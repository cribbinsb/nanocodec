"""Encode on rented boxes, score on this PC's GPU: the encode rounds and the GPU's guards.

tools/dev/aws_measure.py runs a screen, quick or final check this way by default when the
GPU is usable (`gate_reason`). Its command then runs on the PC (in a bounded systemd
scope) with VMAFx CUDA scoring and `cbr.Bench.prebuilt` set, so its results land in the
local point cache under today's keys. The points it needs are encoded on a warm Graviton
box in **encode rounds**:

  - Here, `nceval.realuse.Runner.run_jobs` (the batch boundary) hands each batch to
    `EncodeRounds.ensure`. Points neither cached nor prebuilt start one round:
    `aws_measure.py --encode-round DIR --seed SEED -- COMMAND`, which blocks until the
    box's streams are back in DIR. Then the batch is scored here. A point the round did
    not bring back (e.g. a spatial-layer family) is encoded here, and counted.
  - On the box, the same command runs in encode-only mode (ENCODE_ONLY_ENV). It
    replays the command's decisions from the seed: the results this PC run has used,
    so it reaches the same batch. Every uncached point it reaches is encoded into the
    prebuilt layout (stream, rate log, coded times, per-frame xxh3-128 digests of the
    encoder's reconstruction) and counts as missing (None). Every decision stays here:
    the box's own decisions on missing points only decide what else it encodes ahead.

Guards (owner, 2026-11-24: the card is shared, and the PC crashed at 02:19 that night
on GPU memory exhaustion). A run that cannot meet them is refused, with what is wrong and
how to fix it (`refusal`), and never oversubscribes. Nothing falls back to scoring on a
cloud CPU (owner, 2026-11-27: quality is scored on the local GPU only; no silent
fallbacks; until then a guard sent the command to a Graviton CPU box, and a quick did so
unnoticed when this PC's free disk reached diskguard's floor):
  - one scoring service (`GpuQueue`, SERVICE_SLICE): at most SERVICE_COMMANDS commands,
    one VMAFx server, one CPU quota and memory cap for them all;
  - at least START_FREE_MIB of free GPU memory (nvidia-smi) to start, and at least
    BATCH_FREE_MIB before each batch (our own server's ~1.3 GB is in use by then),
    waiting up to VRAM_WAIT_SECONDS for it;
  - the local work directory's disk budget (nceval.diskguard), checked before every
    batch and so before every encode round (2026-11-24: a quick's round encoded its
    points, then the run died on the budget here);
  - no fixed quiet hours (coordinator, 2026-11-24: the other project's nightly cron jobs
    that used the card are disabled; the free-memory gates above cover anything else on
    it, before every batch).
Mid-run, a guard that fails ends the run with UNAVAILABLE_EXIT; aws_measure then refuses
(its box-encoded streams are kept for the retry). Each batch gets at most one round; what it still lacks is
encoded here.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .diskguard import DiskBudgetExceeded, check_disk

#: On the box: encode the points into the prebuilt layout, score nothing.
ENCODE_ONLY_ENV = "NC_ENCODE_ONLY"
#: The prebuilt directory (cbr.Bench.prebuilt), on the box and here.
PREBUILT_ENV = "NC_PREBUILT_DIR"
#: Here: the command the encode rounds run on the box (a JSON argv list).
ROUND_COMMAND_ENV = "NC_ENCODE_ROUND"
#: Here: the directory of the scoring run's round broker (`RoundBroker`): a command's
#: rounds are requests there, packed with its batch's into one round on one box.
BROKER_ENV = "NC_GPU_ROUND_BROKER"
#: Here: the broker already ran a round for every command of the batch before starting
#: them (their first work, fixedqp ab's rate probes included).
PREFETCHED_ENV = "NC_GPU_PREFETCHED"
#: Here: the most the batch's rounds may cost together (USD; aws_measure --budget).
ROUNDS_BUDGET_ENV = "NC_GPU_ROUNDS_BUDGET"
#: A request waits this long for the batch's other commands to ask too, unless every
#: running command is already waiting.
GATHER_SECONDS = 20
BROKER_POLL_SECONDS = 2
#: After a run's last round, its box stays this long for a follow-up round that scoring
#: may ask for, then is stopped unless another job is on it (`RoundBroker.release`):
#: its idle timer would keep it 10 min, ~$0.03 on a c8g.8xlarge, more than a 6-quick
#: batch's round ($0.021, 2026-11-24). The grace starts only once no command is running:
#: while any command still scores, its next level may ask for another round, and a box
#: released mid-run costs a fresh box's launch and setup (2026-11-25: a staged screen's
#: box stopped 2 min into a 9.5-min scoring pass; its L2/L3 round waited ~10 min for a
#: new box, against ~$0.03 for keeping it).
RELEASE_GRACE_SECONDS = 120
#: While commands still score, the broker renews the last round's box's idle timer this
#: often (`aws_warm.py hold`), well inside the box's own IDLE_MINUTES: the box survives a
#: long scoring pass for the next level's round, and if this PC dies the box still idles
#: out on its own timer.
HOLD_EVERY_SECONDS = 240
AWS_WARM = Path(__file__).resolve().parents[2] / "tools" / "dev" / "aws_warm.py"
#: Here: the commit every encode round's box runs (the GPU run's tree, pinned at its
#: start: the tree may change while it runs).
TREE_COMMIT_ENV = "NC_GPU_TREE_COMMIT"
#: The prebuilt keys this PC already holds, beside the box's prebuilt directory: the
#: box does not encode them again.
HAVE_FILE = "have.txt"
#: The box's rate probes (cbr.Bench.probe), under its prebuilt directory, as its work
#: directory holds them (aws_measure copies them there after an encode round's command).
RATE_PROBES = "rate-probes"
#: The scoring service (owner/coordinator, 2026-11-24: one run at a time sent most
#: quicks to paid CPU scoring while the card had room): up to SERVICE_COMMANDS commands
#: score at once, every one in SERVICE_SLICE (one CPU quota and memory cap for them all),
#: on one VMAFx server, through the machine-wide GPU slots; each gets an equal share of
#: the workers. A command waits for a place up to QUEUE_WAIT_MINUTES (`--gpu-wait`
#: overrides), then is refused.
SERVICE_SLICE = "nanocodec-gpu-score.slice"
# 8 since 2026-11-27 18:00 (6 from 15:00, 3 before): a command holds its place through its
# cloud encode rounds too, so with 6 places the card ran at 25-50 % while a priority screen
# waited for a place.
SERVICE_COMMANDS = 8
QUEUE_WAIT_MINUTES = 10
QUEUE_POLL_SECONDS = 15
#: Where a GPU-scored command that cannot go on writes why (aws_measure reports it).
UNAVAILABLE_REASON_ENV = "NC_GPU_UNAVAILABLE_REASON"
#: A GPU-scored command's exit status when it cannot go on (a guard, a failed round):
#: aws_measure then refuses the measurement.
UNAVAILABLE_EXIT = 75
#: How to fix each kind of reason (its prefix) that the GPU path cannot go on (`refusal`).
FIXES = {
    "disk floor": "free disk under ~/nanocodec-data/work (tools/dev/retention.py lists what "
                  "may go; nceval.diskguard wants 200 GiB free), then run again",
    "GPU memory": "free GPU memory (nvidia-smi: what else holds the card?) or start the "
                  "card's driver again if nvidia-smi fails, then run again",
    "queue wait": "wait for the scoring service's other commands to end, or pass a longer "
                  "--gpu-wait",
    "service memory": "wait for the scoring service's other commands to end (its slice is "
                      "full), or run fewer commands at once",
    "encode round": "read the encode round's log above, fix its cause, then run again (the "
                    "box-encoded streams are kept for the retry)",
}
GENERIC_FIX = "fix the cause above, then run again"
START_FREE_MIB = 8 * 1024
BATCH_FREE_MIB = 4 * 1024
VRAM_WAIT_SECONDS = 180
VRAM_POLL_SECONDS = 15
#: The scoring service's bounds (SERVICE_SLICE) and its GPU slots.
#: 800 % and 6G since 2026-11-24 (coordinator: the PC has 30 threads and 48 GB free; at
#: 400 % three concurrent final-checks would take ~50 min each, one alone took 16.7).
SCOPE_CPU_QUOTA = os.environ.get("NC_GPU_SCORE_CPU_QUOTA", "800%")
#: 10G since 2026-11-24 (coordinator: a 26-screen batch and other commands took the
#: slice to 6.33 of its 6.44 GB).
SCOPE_MEMORY_MAX = os.environ.get("NC_GPU_SCORE_MEMORY_MAX", "10G")
#: Memory a command may need in the slice (measured peaks 2026-11-24: a quick 1.25 GB, a
#: final-check 3.0 GB, page cache included), and how many of a batch's commands run at
#: once: a command or batch is admitted only while the slice has that much headroom for
#: it (`memory_reason`), never to be OOM-killed inside the slice.
COMMAND_MEMORY_BYTES = 3 << 30
BATCH_PARALLEL = 3
LOCAL_SLOTS = 4
VMAFX_SERVERS = 1


def data_root() -> Path:
    return Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))


def queue_dir() -> Path:
    return data_root() / "run" / "gpu-score"


def gpu_free_mib() -> int | None:
    """Free memory of the first GPU (nvidia-smi), None when it cannot be read."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free",
                              "--format=csv,noheader,nounits"], capture_output=True,
                             text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = out.stdout.split()
    return int(lines[0]) if out.returncode == 0 and lines and lines[0].isdigit() else None


def gate_reason(minimum_mib: int, free_mib: Callable[[], int | None] = gpu_free_mib) -> str | None:
    """Why the GPU may not score now (under `minimum_mib` free), or None."""
    free = free_mib()
    if free is None:
        return "GPU memory: nvidia-smi reports none"
    if free < minimum_mib:
        return f"GPU memory: {free} MiB free, under {minimum_mib}"
    return None


def local_reason(work: Path, gpu: Callable[[], str | None] | None = None) -> str | None:
    """Why this PC may not score the next batch: the GPU (`wait_for_gpu`) or the disk
    budget of `work` (diskguard); None when both allow it."""
    reason = (gpu or wait_for_gpu)()
    if reason:
        return reason
    try:
        check_disk(work)
    except DiskBudgetExceeded as error:
        return f"disk floor: {error}"
    return None


def wait_for_gpu(minimum_mib: int = BATCH_FREE_MIB,
                 free_mib: Callable[[], int | None] = gpu_free_mib,
                 wait_s: float = VRAM_WAIT_SECONDS, poll_s: float = VRAM_POLL_SECONDS,
                 sleep: Callable[[float], None] = time.sleep) -> str | None:
    """`gate_reason`, waiting up to `wait_s` for memory to free."""
    deadline = time.monotonic() + wait_s
    while True:
        reason = gate_reason(minimum_mib, free_mib)
        if reason is None or time.monotonic() >= deadline:
            return reason
        sleep(poll_s)


class GpuQueue:
    """A place in the scoring service (SERVICE_COMMANDS flocks under queue_dir): waits
    for one up to `wait_s`."""

    def __init__(self, directory: Path | None = None, places: int = SERVICE_COMMANDS):
        self.directory, self.places = directory or queue_dir(), places
        self._file = None

    def acquire(self, wait_s: float = QUEUE_WAIT_MINUTES * 60,
                sleep: Callable[[float], None] = time.sleep) -> str | None:
        """None once a place is held, else why not (who holds them all)."""
        self.directory.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + wait_s
        while True:
            holders = []
            for place in range(self.places):
                handle = (self.directory / f"place-{place}.lock").open("a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.seek(0)
                    holders.append(handle.read().strip() or "a run")
                    handle.close()
                    continue
                handle.seek(0)
                handle.truncate()
                handle.write(f"pid {os.getpid()} since {time.strftime('%H:%M:%S')}\n")
                handle.flush()
                self._file = handle
                return None
            if time.monotonic() >= deadline:
                return (f"all {self.places} places busy for {wait_s / 60:g} min ("
                        + "; ".join(holders) + ")")
            sleep(QUEUE_POLL_SECONDS)

    def release(self) -> None:
        if self._file is not None:
            fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()
            self._file = None


def ensure_slice() -> None:
    """SERVICE_SLICE with the service's CPU quota and memory cap (idempotent): every
    GPU-scored command's scope is placed in it."""
    quota = SCOPE_CPU_QUOTA
    shown = subprocess.run(["systemctl", "--user", "show", SERVICE_SLICE, "-p",
                            "CPUQuotaPerSecUSec", "--value"], capture_output=True, text=True).stdout
    if shown.strip().endswith("s") and shown.strip()[:-1].replace(".", "").isdigit():
        # never lower a quota raised for a benchmark's scoring (NC_GPU_SCORE_CPU_QUOTA, 1600 %
        # on 2026-11-26) from a command that runs at the default
        current = float(shown.strip()[:-1]) * 100
        quota = f"{max(current, float(quota.rstrip('%'))):.0f}%"
    memory = SCOPE_MEMORY_MAX
    shown_mem = subprocess.run(["systemctl", "--user", "show", SERVICE_SLICE, "-p", "MemoryMax",
                                "--value"], capture_output=True, text=True).stdout.strip()
    if shown_mem.isdigit() and memory.endswith("G") and int(shown_mem) > int(memory[:-1]) << 30:
        memory = shown_mem  # never lowered while a benchmark scores at a higher cap either
    subprocess.run(["systemctl", "--user", "set-property", "--runtime", SERVICE_SLICE,
                    f"CPUQuota={quota}", f"MemoryMax={memory}"],
                   check=True, capture_output=True)


def slice_memory() -> tuple[int, int] | None:
    """SERVICE_SLICE's (memory it cannot give back, limit) in bytes: anonymous, shared and
    kernel memory, not its page cache, which the kernel reclaims under the limit (2026-11-24:
    the slice showed 7.0 of 10 GB in use, all of it page cache, with no command running).
    None when systemd reports no slice (no command in it yet)."""
    shown = subprocess.run(["systemctl", "--user", "show", SERVICE_SLICE, "-p", "ControlGroup",
                            "-p", "MemoryMax"], capture_output=True, text=True).stdout
    values = dict(line.split("=", 1) for line in shown.splitlines() if "=" in line)
    limit, group = values.get("MemoryMax", ""), values.get("ControlGroup", "")
    stat = Path("/sys/fs/cgroup") / group.lstrip("/") / "memory.stat"
    if not limit.isdigit() or not group or not stat.exists():
        return None
    fields = dict(line.split() for line in stat.read_text().splitlines())
    held = sum(int(fields.get(name, 0)) for name in ("anon", "shmem", "kernel"))
    return held, int(limit)


def memory_reason(commands: int,
                  usage: Callable[[], tuple[int, int] | None] = slice_memory) -> str | None:
    """Why `commands` commands (at most BATCH_PARALLEL of a batch at once) may not start in
    the slice now: less than COMMAND_MEMORY_BYTES each of headroom under its limit."""
    found = usage()
    if found is None:
        return None
    current, limit = found
    need = COMMAND_MEMORY_BYTES * min(commands, BATCH_PARALLEL)
    if limit - current < need:
        return (f"the scoring service holds {current / 2**30:.1f} of its {limit / 2**30:.1f} "
                f"GiB (page cache aside); {need / 2**30:.1f} GiB needed")
    return None


def worker_share() -> int:
    """Each command's workers: an equal share of the service's GPU slots (at least 1),
    so commands in the service take turns at the GPU."""
    return max(1, LOCAL_SLOTS * 2 // SERVICE_COMMANDS)


def encode_only() -> bool:
    return bool(os.environ.get(ENCODE_ONLY_ENV))


def prebuilt_dir() -> Path | None:
    value = os.environ.get(PREBUILT_ENV)
    return Path(value) if value else None


def have_keys(prebuilt: Path) -> set[str]:
    """The keys listed in prebuilt/HAVE_FILE (the box: what the PC already holds)."""
    path = prebuilt / HAVE_FILE
    return set(path.read_text().split()) if path.exists() else set()


def import_rate_probes(source: Path, work: Path) -> int:
    """The box's rate probes (cbr.Bench.probe's cache files, under RATE_PROBES in the
    prebuilt directory) into this machine's cache at the same relative paths, without
    their `cpu_s`: that is the box's CPU under its own load, so a probe asked for its
    time (`timed`) is coded here again. A file here already is kept. Returns how many."""
    count = 0
    for path in sorted(source.rglob("rate-*.json")) if source.is_dir() else ():
        target = work / path.relative_to(source)
        if target.exists():
            continue
        probe = {k: v for k, v in json.loads(path.read_text()).items() if k != "cpu_s"}
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f".{target.name}.partial")
        partial.write_text(json.dumps(probe))
        partial.rename(target)
        count += 1
    return count


def file_sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_done(base: Path, suffixes: list[str]) -> None:
    """A prebuilt point's `.done`, written on the box once its encoder has exited and its
    files are renamed into place: each file's size and SHA-256 (point_whole checks them
    after the copy to the PC). Written under a `.partial` name, never copied, and renamed."""
    files = {}
    for suffix in suffixes:
        path = Path(f"{base}{suffix}")
        if path.exists():
            files[suffix] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    partial = Path(f"{base}.done.partial")
    partial.write_text(json.dumps({"files": files}) + "\n")
    partial.rename(f"{base}.done")


def point_whole(done: Path, base: Path | None = None) -> bool:
    """Whether the point `done` marks has every file its `.done` records, whole, beside
    `base` (default: beside `done`). An empty `.done` (written before the manifest) checks
    nothing."""
    text = done.read_text()
    if not text.strip():
        return True
    stem = str(base) if base is not None else str(done)[:-len(".done")]
    for suffix, want in json.loads(text)["files"].items():
        path = Path(stem + suffix)
        if (not path.exists() or path.stat().st_size != want["bytes"]
                or file_sha256(path) != want["sha256"]):
            return False
    return True


def done_keys(prebuilt: Path) -> list[str]:
    """Every complete prebuilt point under `prebuilt`, as its key."""
    return sorted(str(p.relative_to(prebuilt))[:-len(".done")]
                  for p in prebuilt.rglob("*.done"))


#: One EncodeRounds per GPU-scored command process (EncodeRounds.from_env).
_PROCESS_ROUNDS: dict[tuple[str, str, str], "EncodeRounds"] = {}
_PROCESS_ROUNDS_LOCK = threading.Lock()


def refusal(reason: str) -> str:
    """The message that refuses a measurement the GPU cannot score: what is wrong, how to
    fix it, and that nothing was scored elsewhere."""
    fix = next((text for prefix, text in FIXES.items() if reason.startswith(prefix)),
               GENERIC_FIX)
    return (f"refused: the local GPU cannot score this ({reason}). Fix: {fix}. Nothing was "
            f"scored on a cloud CPU instead (owner rule: quality is scored on the local GPU "
            f"only).")


class GpuUnavailable(SystemExit):
    """The GPU path cannot go on: the run stops with UNAVAILABLE_EXIT, and aws_measure
    refuses the measurement (`refusal`)."""

    def __init__(self, reason: str):
        print(f"gpu-score: {refusal(reason)}", flush=True)
        path = os.environ.get(UNAVAILABLE_REASON_ENV)
        if path:
            Path(path).write_text(f"{reason}\n")
        super().__init__(UNAVAILABLE_EXIT)


class EncodeRounds:
    """This PC's side of the encode rounds (module doc), one per scoring run."""

    def __init__(self, command: list[str], prebuilt: Path, work: Path,
                 runner_round: Callable[[list[str]], int] | None = None,
                 gpu_check: Callable[[], str | None] | None = None):
        """`runner_round(argv)` runs a round and returns its status (default: a
        subprocess); `gpu_check()` is the guard before each batch (default
        `local_reason`: the GPU and the disk budget)."""
        self.command, self.prebuilt, self.work = command, prebuilt, work
        self._run = runner_round or self._call
        self._gpu_check = gpu_check or (lambda: local_reason(work))
        self._lock = threading.Lock()
        self.rounds = 0
        self.local_encodes = 0
        #: The rounds' own CPU here (aws_measure: staging, the clips' check decodes,
        #: sync, pulls), apart from the scoring.
        self.round_cpu_s = 0.0
        #: The result files this run has used (cached or measured): the next round's seed.
        self.used: set[Path] = set()

    @classmethod
    def from_env(cls, work: Path) -> "EncodeRounds | None":
        """The process's one instance (every Runner of the command shares it: quick's L1
        and probe each make a Runner, and two instances ran two rounds at once on two
        boxes, 2026-11-24), or None outside a GPU-scored run."""
        value, prebuilt = os.environ.get(ROUND_COMMAND_ENV), prebuilt_dir()
        if not value or prebuilt is None or encode_only():
            return None
        with _PROCESS_ROUNDS_LOCK:
            key = (value, str(prebuilt), str(work))
            if key not in _PROCESS_ROUNDS:
                _PROCESS_ROUNDS[key] = cls(json.loads(value), prebuilt, work)
            return _PROCESS_ROUNDS[key]

    def note(self, result_path: Path) -> None:
        self.used.add(result_path)

    def ensure(self, jobs: list, missing_of: Callable[[list], list]) -> None:
        """Before a batch: the GPU guards, then one round for the points it lacks.
        `missing_of(jobs)`: the jobs neither cached nor prebuilt."""
        reason = self._gpu_check()
        if reason:
            raise GpuUnavailable(reason)
        if not missing_of(jobs):
            return
        with self._lock:  # concurrent batches (quick's L1 and probe) share one round
            missing = missing_of(jobs)
            if not missing:
                return
            self._round()
            still = missing_of(jobs)
            self.local_encodes += len(still)
            print(f"gpu-score: round {self.rounds}: {len(missing) - len(still)} of "
                  f"{len(missing)} points encoded on the box"
                  + (f"; {len(still)} are encoded here" if still else "")
                  + f" (rounds' CPU here so far {self.round_cpu_s:.1f} s)", flush=True)

    def prefetch(self) -> None:
        """One round before any batch (the GPU guards first): a command whose first work
        is not a batch of points (fixedqp ab's rate probes, which choose its quantizers)
        has the box code that too, and later batches find their streams here. The
        broker's round for the whole batch (PREFETCHED_ENV) already was that round."""
        reason = self._gpu_check()
        if reason:
            raise GpuUnavailable(reason)
        if os.environ.get(PREFETCHED_ENV):
            imported = import_rate_probes(self.prebuilt / RATE_PROBES, self.work)
            print(f"gpu-score: the batch's first round brought {len(done_keys(self.prebuilt))}"
                  f" streams and {imported} rate probes", flush=True)
            return
        with self._lock:
            self._round()
            print(f"gpu-score: round {self.rounds} (ahead of the first batch): "
                  f"{len(done_keys(self.prebuilt))} streams here", flush=True)

    def _round(self) -> None:
        """One round (lock held), then its rate probes into the local cache."""
        broker = os.environ.get(BROKER_ENV)
        status = (submit_request(Path(broker), self.round_request()) if broker
                  else self._run(self.round_argv()))
        self.rounds += 1
        if status != 0:
            raise GpuUnavailable(f"encode round {self.rounds} failed (status {status})")
        imported = import_rate_probes(self.prebuilt / RATE_PROBES, self.work)
        if imported:
            print(f"gpu-score: {imported} rate probes from the box", flush=True)

    def _call(self, argv: list[str]) -> int:
        """Run a round; its CPU (it and its children, from its exit status) is counted."""
        process = subprocess.Popen(argv)
        _, status, usage = os.wait4(process.pid, 0)
        self.round_cpu_s += usage.ru_utime + usage.ru_stime
        return os.waitstatus_to_exitcode(status)

    def round_argv(self) -> list[str]:
        """The round's own aws_measure call (no broker)."""
        request = self.round_request()
        commit = os.environ.get(TREE_COMMIT_ENV)
        return [sys.executable, str(AWS_MEASURE), "--encode-round", request["prebuilt"],
                "--seed", request["seed"], "--have", request["have"],
                *(["--tree-commit", commit] if commit else []), "--", *self.command]

    def round_request(self) -> dict:
        """The round's request ({command, prebuilt, seed, have}); writes its seed (links
        to the used results) and the list of points held here."""
        seed = self.prebuilt.parent / "seed"
        for link in sorted(seed.rglob("*"), reverse=True) if seed.exists() else ():
            link.unlink() if link.is_symlink() or link.is_file() else link.rmdir()
        for path in sorted(self.used):
            if path.exists():
                target = seed / path.relative_to(self.work)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(path.resolve())
        seed.mkdir(parents=True, exist_ok=True)
        self.prebuilt.mkdir(parents=True, exist_ok=True)
        have = self.prebuilt.parent / HAVE_FILE
        have.write_text("".join(f"{k}\n" for k in done_keys(self.prebuilt)))
        return {"command": self.command, "prebuilt": str(self.prebuilt), "seed": str(seed),
                "have": str(have)}


AWS_MEASURE = Path(__file__).resolve().parents[2] / "tools" / "dev" / "aws_measure.py"


def submit_request(broker: Path, request: dict) -> int:
    """A round through the batch's broker: the request goes into its directory, and its
    status comes back when the round that packed it ends."""
    path = broker / f"req-{os.getpid()}-{time.monotonic_ns()}.json"
    partial = path.with_name(f".{path.name}")
    partial.write_text(json.dumps(request))
    partial.rename(path)
    status = Path(f"{path}.status")
    while not status.exists():
        time.sleep(BROKER_POLL_SECONDS)
    code = int(status.read_text())
    status.unlink()
    return code


def rounds_spent(log: Path) -> float:
    """What a rounds log's rounds cost together (USD)."""
    if not log.exists():
        return 0.0
    return sum(json.loads(line).get("estimated_usd") or 0.0
               for line in log.read_text().splitlines())


class RoundBroker:
    """The rounds of one scoring run (a batch, or one command): every command's round is
    a request in `directory`, and the requests that wait together go to ONE round on ONE
    box (2026-11-24: a 26-quick batch whose commands each ran their own round launched 14
    new boxes, $1.19 against ~$0.3). Rounds run one at a time, so a batch holds at most
    one box, which later rounds reuse while it is warm. A round that would pass
    `budget_usd` (the rounds so far, `rounds_log`) is refused, and so are its commands."""

    def __init__(self, directory: Path, rounds_log: Path, budget_usd: float | None,
                 call: Callable[[list[str]], int] = subprocess.call,
                 gather_s: float = GATHER_SECONDS, grace_s: float = RELEASE_GRACE_SECONDS,
                 clock: Callable[[], float] = time.time):
        self.directory, self.rounds_log, self.budget = directory, rounds_log, budget_usd
        self._call, self.gather_s, self.grace_s, self._clock = call, gather_s, grace_s, clock
        self.rounds = 0
        #: The box of the last round, its $/h, and when that round ended (None: released,
        #: or no round yet).
        self._box: tuple[str, float, float] | None = None
        #: When the box's idle timer was last renewed (`hold`).
        self._held_at = 0.0

    def run_round(self, requests: list[dict]) -> int:
        spent = rounds_spent(self.rounds_log)
        if self.budget is not None and spent >= self.budget:
            print(f"gpu-score: the batch's rounds cost ${spent:.3f}, its budget is "
                  f"${self.budget:.2f}: no more rounds", flush=True)
            return 1
        self.rounds += 1
        listing = self.directory / f"round-{self.rounds}.json"
        listing.write_text(json.dumps(requests))
        commit = os.environ.get(TREE_COMMIT_ENV)
        status = self._call([sys.executable, str(AWS_MEASURE), "--encode-rounds", str(listing),
                             "--rounds-log", str(self.rounds_log),
                             *(["--tree-commit", commit] if commit else [])])
        last = self.last_round()
        if last.get("warm_box"):
            self._box = (last["warm_box"], last.get("spot_usd_per_hour") or 0.0, self._clock())
        return status

    def last_round(self) -> dict:
        lines = self.rounds_log.read_text().splitlines() if self.rounds_log.exists() else []
        return json.loads(lines[-1]) if lines else {}

    def release(self, force: bool = False, running: int = 0) -> bool:
        """Stop the last round's box once no command is `running` and `grace_s` has passed
        since that round (`force`: now, the run has no more rounds to ask for), unless
        another job is on it; its idle time until then goes into the rounds log, so the
        batch's cost is the real one. Returns whether it was stopped."""
        if self._box is None:
            return False
        box, price, ended = self._box
        idle_s = self._clock() - ended
        if not force and (running or idle_s < self.grace_s):
            return False
        self._box = None
        if self._call([sys.executable, str(AWS_WARM), "stop-if-idle", box]) != 0:
            print(f"gpu-score: warm box {box} is in use: left to its idle timer", flush=True)
            return False
        tail = round(price * idle_s / 3600, 4)
        with self.rounds_log.open("a") as log:
            log.write(json.dumps({"warm_box": box, "idle_tail_s": round(idle_s),
                                  "estimated_usd": tail}) + "\n")
        print(f"gpu-score: warm box {box} stopped after {idle_s:.0f} s idle (${tail})",
              flush=True)
        return True

    def hold(self, running: int) -> bool:
        """Renew the last round's box's idle timer while commands still run (at most every
        HOLD_EVERY_SECONDS). Returns whether it was renewed."""
        if self._box is None or not running or self._clock() - self._held_at < HOLD_EVERY_SECONDS:
            return False
        self._held_at = self._clock()
        if self._call([sys.executable, str(AWS_WARM), "hold", self._box[0]]) != 0:
            print(f"gpu-score: could not renew warm box {self._box[0]}'s idle timer", flush=True)
            return False
        return True

    def pending(self) -> list[Path]:
        return sorted(p for p in self.directory.glob("req-*.json"))

    def serve_once(self, running: int, now: float | None = None) -> int:
        """One look at the requests: when every running command waits, or the oldest has
        waited `gather_s`, they go to one round and get its status. Returns how many."""
        pending = self.pending()
        if not pending:
            return 0
        oldest = min(p.stat().st_mtime for p in pending)
        if len(pending) < running and (now or time.time()) - oldest < self.gather_s:
            return 0
        status = self.run_round([json.loads(p.read_text()) for p in pending])
        for path in pending:
            path.unlink()
            Path(f"{path}.status").write_text(f"{status}\n")
        return len(pending)

    def serve(self, running: Callable[[], int], done: threading.Event) -> None:
        while not (done.is_set() and not self.pending()):
            if not self.serve_once(running()) and not self.release(running=running()) \
                    and not self.hold(running()):
                time.sleep(BROKER_POLL_SECONDS)
        self.release(force=True)


def first_requests(spec: list[dict], directory: Path) -> list[dict]:
    """The batch's first round: every command, unseeded, skipping the streams its prebuilt
    directory already holds (a retry)."""
    requests = []
    for index, entry in enumerate(spec):
        prebuilt = Path(entry["env"][PREBUILT_ENV])
        prebuilt.mkdir(parents=True, exist_ok=True)
        seed, have = directory / f"seed-{index}", directory / f"have-{index}.txt"
        seed.mkdir(exist_ok=True)
        have.write_text("".join(f"{k}\n" for k in done_keys(prebuilt)))
        requests.append({"command": json.loads(entry["env"][ROUND_COMMAND_ENV]),
                         "prebuilt": str(prebuilt), "seed": str(seed), "have": str(have)})
    return requests


def run_batch(spec_path: Path, call: Callable[[list[str]], int] = subprocess.call) -> None:
    """The commands of a GPU-scored batch, BATCH_PARALLEL side by side (aws_measure.
    run_scoring starts this as its scope's main process), each with its own environment
    additions; each one's exit status goes to its "status" file. Their rounds go through
    one RoundBroker, whose first round codes every command's first work at once."""
    spec = json.loads(spec_path.read_text())
    directory = spec_path.parent / "broker"
    directory.mkdir(exist_ok=True)
    budget = os.environ.get(ROUNDS_BUDGET_ENV)
    broker = RoundBroker(directory, spec_path.parent / "rounds.jsonl",
                         float(budget) if budget else None, call)
    first = broker.run_round(first_requests(spec, directory))
    prefetched = {PREFETCHED_ENV: "1"} if first == 0 else {}
    if first:
        print(f"gpu-score: the batch's first round failed (status {first}); each command "
              f"asks for its own", flush=True)
    running, lock, done = [0], threading.Lock(), threading.Event()

    def run(entry: dict) -> None:
        with lock:
            running[0] += 1
        try:
            process = subprocess.Popen(entry["argv"], env={**os.environ, **entry["env"],
                                                           BROKER_ENV: str(directory),
                                                           **prefetched})
            Path(entry["status"]).write_text(f"{process.wait()}\n")
        finally:
            with lock:
                running[0] -= 1
    server = threading.Thread(target=broker.serve, args=(lambda: running[0], done),
                              daemon=True)
    server.start()
    # BATCH_PARALLEL at once: the memory the batch was admitted with (memory_reason)
    with ThreadPoolExecutor(BATCH_PARALLEL) as pool:
        list(pool.map(run, spec))
    done.set()
    server.join()


def main(argv: list[str]) -> None:
    if argv[:1] != ["run"] or len(argv) != 2:
        raise SystemExit("usage: python3 -m nceval.gpuscore run BATCH.json")
    run_batch(Path(argv[1]))


if __name__ == "__main__":
    main(sys.argv[1:])
