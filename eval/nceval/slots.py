"""Machine-wide CPU slots shared by every agent's evaluation runs.

Each agent used to run its own worker pool and sleep while `uptime` showed
a load above 24. Seven uncoordinated pools oversubscribed the machine at
some moments and left it idle at others. Instead, every CPU-heavy
subprocess group of the harness (an encode with its streamed metrics, a
point's decoders, an anchor encode) holds CPU slots while it runs:

  - The pool is NC_CPU_SLOTS slots (default `DEFAULT_SLOTS`) in
    NC_SLOT_DIR (default ~/nanocodec-data/run/slots), one lock file per slot.
  - A slot is an exclusive flock(2) on its file. The kernel releases it when
    the holder exits, even after a crash or kill -9, so slots never leak and
    there is no daemon. Each acquisition opens its own file descriptions, so
    threads and processes are limited by the same pool.
  - A request for k slots takes all k or none (a partial set is released
    before retrying), so two multi-slot requests cannot deadlock.
  - Requests are served FIFO machine-wide: a waiter holds a ticket (a locked
    file in `queue/`), and only the oldest live ticket may take slots, so a
    multi-slot request is never overtaken forever by single-slot ones.
    A ticket whose holder died is unlocked; the next waiter removes it.
  - Fair shares: one process (an agent's job) holds at most
    `owner_share()` slots at a time (max(2, pool/2): 12 of 24), so two big
    jobs can share the machine and neither starves the other; its own
    threads queue FIFO for that share. A real-use point claims 3 slots
    (point.py), so one screen runs 4 points at once. (pool/4, 6 slots, let a
    screen run only ONE point at a time once MS-SSIM made a point claim 5:
    proxy-6 took 13-20 min instead of ~2.)
  - GPU slots (`gpu_slots`): a separate machine-wide pool of
    NC_GPU_SLOTS (default `DEFAULT_GPU_SLOTS`) in NC_SLOT_DIR/gpu, same
    locking and FIFO queue, one slot per point that scores on the GPU
    (VMAFx, nceval.metrics). The GPU saturates at ~300 720p frames/s (~2
    real-use points/s) from about 4 VMAFx processes; beyond that each
    process's CPU time grows with contention (1.0 / 1.5 / 2.8 / 4.0 CPU-s
    per 150-frame pass at 1 / 4 / 8 / 16 processes, 2026-11-11), so more
    concurrent points only burn CPU. Take GPU slots BEFORE CPU slots (never
    the reverse), so holders of both never wait on each other.
  - NVENC slots (`nvenc_slots`): NC_NVENC_SLOTS (default
    `DEFAULT_NVENC_SLOTS`) concurrent hardware encodes machine-wide, in
    NC_SLOT_DIR/nvenc. NVENC is its own block on the GPU, but each session
    takes device memory, which VMAFx shares: a request also waits until the
    GPU has `NVENC_SESSION_MIB` per session plus `NVENC_HEADROOM_MIB` free.
  - Memory: a request also waits while MemAvailable is below
    `MIN_AVAILABLE_BYTES`. A streamed point needs ~100-200 MB, so this only
    trips under real memory pressure.

Workers therefore never sleep on the load average: an agent can submit as
many points as it likes (up to its 8-worker cap) and they run exactly when
slots are free. Do not nest acquisitions in one thread (holding slots while
waiting for more can starve). Timing runs still need a quiet machine.
"""

from __future__ import annotations

import fcntl
import os
import random
import subprocess
import threading
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

DEFAULT_SLOTS = 24  # of 32 hardware threads; everything runs at nice 19, so the desktop stays responsive
# Concurrent GPU-scored points machine-wide. A slot is held for the whole
# streamed encode. Since points share the VMAFx scoring servers (no CUDA
# context per point, nceval.metrics.vmafx_command), 16 concurrent 720p points
# score ~1.35x the frames/s of 8 on a loaded machine (563 vs 419 fps,
# 2026-11-12) for ~6 GB of device memory; per-process contexts made more
# than 8 only burn CPU.
DEFAULT_GPU_SLOTS = 16
# Concurrent NVENC encodes machine-wide (owner, 2026-11-12: at most 2, the
# GPU also scores VMAFx). A 720p session takes ~0.5 GB of device memory, so a
# request waits for that per session plus 1 GiB of headroom.
DEFAULT_NVENC_SLOTS = 2
NVENC_SESSION_MIB = 512
NVENC_HEADROOM_MIB = 1024
# Every harness subprocess runs at the lowest CPU priority, whatever the
# caller's niceness, so interactive work always wins (`low_priority`).
NICENESS = 19
MIN_AVAILABLE_BYTES = 4 << 30
GPU_MEMORY_POLL_SECONDS = 2.0
POLL_SECONDS = (0.02, 0.2)  # randomised back-off between attempts
MIN_OWNER_SHARE = 2
OWNER_SHARE_DIVISOR = 2


def low_priority(cmd: list[str]) -> list[str]:
    """`cmd` run at NICENESS via nice(1) (adds to the caller's niceness, capped at 19)."""
    return ["nice", "-n", str(NICENESS), *cmd]


def slot_count() -> int:
    count = int(os.environ.get("NC_CPU_SLOTS", DEFAULT_SLOTS))
    if count < 1:
        raise ValueError("NC_CPU_SLOTS must be positive")
    return count


def slot_directory() -> Path:
    default = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data")) / "run" / "slots"
    return Path(os.environ.get("NC_SLOT_DIR", default))


def gpu_slot_count() -> int:
    count = int(os.environ.get("NC_GPU_SLOTS", DEFAULT_GPU_SLOTS))
    if count < 1:
        raise ValueError("NC_GPU_SLOTS must be positive")
    return count


def available_memory() -> int | None:
    """Linux MemAvailable in bytes, or None if unknown (then not enforced)."""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return None


def owner_share(total: int | None = None) -> int:
    """The most slots one process may hold at once."""
    total = slot_count() if total is None else total
    return min(total, max(MIN_OWNER_SHARE, total // OWNER_SHARE_DIVISOR))


class _OwnerShare:
    """This process's running total of held slots, granted FIFO to its threads."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._held = 0
        self._queue: deque = deque()

    @contextmanager
    def reserve(self, count: int, cap: int) -> Iterator[None]:
        ticket = object()
        with self._condition:
            self._queue.append(ticket)
            self._condition.wait_for(
                lambda: self._queue[0] is ticket and self._held + count <= cap)
            self._queue.popleft()
            self._held += count
            self._condition.notify_all()
        try:
            yield
        finally:
            with self._condition:
                self._held -= count
                self._condition.notify_all()


# Process-wide on purpose: the share limits the whole process (all its
# worker threads), exactly as the slot files limit the whole machine.
_OWNER = _OwnerShare()


def _try_acquire(directory: Path, wanted: int, total: int) -> list:
    """Lock `wanted` free slots, or none. Returns the open lock files."""
    held = []
    start = random.randrange(total)  # spread contenders over the files
    for offset in range(total):
        handle = (directory / f"slot-{(start + offset) % total:03d}.lock").open("a+b")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        except BaseException:
            handle.close()
            for other in held:
                other.close()
            raise
        held.append(handle)
        if len(held) == wanted:
            return held
    for handle in held:
        handle.close()
    return []


@contextmanager
def cpu_slots(count: int = 1) -> Iterator[None]:
    """Hold `count` machine-wide CPU slots while inside.

    The request is capped at this process's share (`owner_share`). `count` 0
    (a gate served entirely from the verified-stream cache) holds nothing.
    Acquire before starting the processes the slots are for.
    """
    if count < 0:
        raise ValueError("slot count must not be negative")
    if count == 0:
        yield
        return
    total = slot_count()
    cap = owner_share(total)
    wanted = min(count, cap)
    directory = slot_directory()
    directory.mkdir(parents=True, exist_ok=True)
    with _OWNER.reserve(wanted, cap):
        with _machine_slots(directory, wanted, total):
            yield


@contextmanager
def gpu_slots(count: int = 1) -> Iterator[None]:
    """Hold `count` machine-wide GPU slots while inside (0 holds nothing).

    Enter before `cpu_slots`, never inside it (see the module docstring).
    """
    if count < 0:
        raise ValueError("slot count must not be negative")
    if count == 0:
        yield
        return
    total = gpu_slot_count()
    directory = slot_directory() / "gpu"
    directory.mkdir(parents=True, exist_ok=True)
    with _machine_slots(directory, min(count, total), total):
        yield


def gpu_free_mib() -> int | None:
    """Free device memory of the first GPU in MiB (nvidia-smi), or None if unknown."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits", "--id=0"],
                             capture_output=True, text=True, timeout=30, check=True).stdout
        return int(out.split()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


@contextmanager
def nvenc_slots(sessions: int = 1) -> Iterator[None]:
    """Hold NVENC slots for `sessions` concurrent encodes while inside: one slot
    each, or the whole pool for more sessions than it has (an approved parallel
    throughput run, tools/dev/nvenc_timing.py, which then has NVENC to itself),
    once the GPU has the sessions' memory plus headroom free. Enter before
    `cpu_slots`, like `gpu_slots`."""
    total = int(os.environ.get("NC_NVENC_SLOTS", DEFAULT_NVENC_SLOTS))
    directory = slot_directory() / "nvenc"
    directory.mkdir(parents=True, exist_ok=True)
    need = sessions * NVENC_SESSION_MIB + NVENC_HEADROOM_MIB
    with _machine_slots(directory, min(sessions, total), total):
        while (free := gpu_free_mib()) is not None and free < need:
            time.sleep(GPU_MEMORY_POLL_SECONDS)
        yield


@contextmanager
def _machine_slots(directory: Path, wanted: int, total: int) -> Iterator[None]:
    held: list = []
    try:
        with _ticket(directory / "queue") as ticket:
            while True:
                memory = available_memory()
                if (_is_first(ticket) and (memory is None or memory >= MIN_AVAILABLE_BYTES)):
                    held = _try_acquire(directory, wanted, total)
                    if held:
                        break
                time.sleep(random.uniform(*POLL_SECONDS))
        yield
    finally:
        for handle in held:
            handle.close()  # never delete lock files: that would allow duplicate holders


@contextmanager
def _ticket(queue: Path) -> Iterator[Path]:
    """A place in the machine-wide queue, held (flock) until the block ends.

    The ticket is locked under a temporary name and only then renamed into
    the queue, so nobody can mistake a ticket being created for a dead one.
    Names sort by creation time.
    """
    queue.mkdir(parents=True, exist_ok=True)
    name = f"{time.time_ns():020d}-{os.getpid()}-{threading.get_ident()}.ticket"
    pending = queue / f".{name}.pending"
    handle = pending.open("a+b")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX)
        ticket = queue / name
        pending.rename(ticket)
        try:
            yield ticket
        finally:
            ticket.unlink(missing_ok=True)
    finally:
        handle.close()
        pending.unlink(missing_ok=True)


def _is_first(ticket: Path) -> bool:
    """Whether no live ticket is older than `ticket`; removes dead ones."""
    for other in sorted(ticket.parent.glob("*.ticket")):
        if other.name >= ticket.name:
            return True
        try:
            with other.open("a+b") as handle:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False  # a live, older waiter
        except FileNotFoundError:
            continue  # just finished
        other.unlink(missing_ok=True)  # its holder died: unlocked
    return True


def contending_owners() -> set[int]:
    """PIDs of other processes holding slots (from /proc/locks) or queued for them."""
    directory = slot_directory()
    inodes = set()
    for index in range(slot_count()):
        try:
            inodes.add((directory / f"slot-{index:03d}.lock").stat().st_ino)
        except FileNotFoundError:
            pass
    owners = set()
    try:
        for line in Path("/proc/locks").read_text().splitlines():
            fields = line.split()
            if len(fields) > 5 and fields[1] == "FLOCK" and int(fields[5].split(":")[-1]) in inodes:
                owners.add(int(fields[4]))
    except (OSError, ValueError):
        pass
    for ticket in (directory / "queue").glob("*.ticket"):
        try:
            owners.add(int(ticket.name.split("-")[1]))
        except (IndexError, ValueError):
            pass
    owners.discard(os.getpid())
    return owners


def available_slots() -> float:
    """Slots this process can expect right now, for wall-time estimates.

    The nominal share, divided fairly among the processes currently holding or
    waiting for slots, and never more than the idle CPU cores (other work on
    the machine does not take slots).
    """
    total = slot_count()
    fair = total / (len(contending_owners()) + 1)
    try:
        idle = (os.cpu_count() or total) - os.getloadavg()[0]
    except OSError:
        idle = total
    return max(1.0, min(owner_share(total), fair, idle))


def slots_in_use() -> int:
    """How many slots are held right now, by anyone (for status displays)."""
    directory, busy = slot_directory(), 0
    for index in range(slot_count()):
        path = directory / f"slot-{index:03d}.lock"
        if not path.exists():
            continue
        with path.open("a+b") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                busy += 1
    return busy
