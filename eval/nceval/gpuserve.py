"""The machine-wide pool of VMAFx scoring servers (docs/handoff/vmafx-gpu.md).

A point scores VMAF v1 / MS-SSIM as a run of `vmaf --via-server NAME`: the
client hands its arguments and stdin to the `vmaf --serve NAME` process,
starting it if none listens. The pool of one build is `POOL_SIZE` servers;
NAME carries the user, the build's identity (device, inode and mtime of the
binary) and the server's index, so every user of one build shares one pool
and a rebuilt binary never reaches a server of the old one.

Bounds (the GPU is shared with other jobs and another user):
  - a server runs at most 6 runs at once and starts one only while the card
    has 1.5 GB free, else the run waits (VMAFx patch 0008);
  - its unused cached buffers are capped at 1 GiB, and it exits after 60 s
    without work;
  - so a pool holds at most ~POOL_SIZE x 4 GB (typically 1-2 GB per server
    at 720p), and concurrent points beyond its slots wait in the servers
    instead of starting more of them.

Switching builds is one way. Every build has a generation (`vmaf
--serve-generation`, the VMAFx patch that last changed the server; builds
before generations count as 0). The first run of the default build in a
process
  - fails (`OlderThanLivePool`) if a live server of a NEWER generation
    exists: its eval tree is older than the live scoring build and must be
    rebased, never drain the newer pool;
  - drains every server of an OLDER generation (`--drain-server`). A drained
    server finishes its runs, refuses new ones (exit `STATUS_RETIRED`) and
    re-executes into a memory-free stub that keeps the name for a few hours,
    so clients of a retired build fail loudly instead of starting it again.
Servers since generation 10 also refuse a drain from a build that is not
newer, so even an old tree's unconditional drain cannot retire them (on
2026-11-14 an older tree drained the newer serve4 pool and every current job
failed as retired). Servers of the same generation are left alone.

Stalls fail loudly (VMAFx patch 0009, after a silent 15-minute pool wedge on
2026-11-14): the client opens its inputs and passes them as descriptors, so
a server never blocks opening a FIFO whose writer is gone; a run whose client
hung up, or whose inputs stall, is abandoned and frees its slot; a client
that hears no heartbeat from its server fails. Each exits `STATUS_STALLED`
with a message naming the server.

`python3 -m nceval gpu-status` lists the servers, their GPU memory and idle
time.
"""

from __future__ import annotations

import functools
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

#: Servers per build (NC_VMAFX_SERVERS overrides).
POOL_SIZE = 3
POOL_SIZE_ENV = "NC_VMAFX_SERVERS"
#: Where servers publish their state and log (vmaf_serve.cpp).
STATE_DIR = Path("/tmp")
STATE_PREFIX = "vmaf-serve-"
#: A run's exit status when its server was drained, or had no GPU memory in time.
STATUS_RETIRED = 86
STATUS_NO_MEMORY = 87
#: A run's exit status when it was abandoned as stalled: its inputs moved no
#: byte for 180 s (server side), or its client heard nothing from the server
#: for 240 s (a wedged server). vmaf_serve.cpp has the reasons for the values.
STATUS_STALLED = 88
DRAIN_TIMEOUT_SECONDS = 10
#: Generation of builds that predate generations (serve3, serve4).
UNVERSIONED_GENERATION = 0


class OlderThanLivePool(RuntimeError):
    """This process's default VMAFx build is older than a live pool's."""


def name_prefix() -> str:
    return f"nc-vmafx-{os.getuid()}-"


def build_identity(tool: Path) -> str:
    stat = tool.resolve().stat()
    return f"{stat.st_dev:x}-{stat.st_ino:x}-{stat.st_mtime_ns:x}"


def server_name(tool: Path, index: int) -> str:
    """Scoring server `index` of this build of `tool` for this user."""
    return f"{name_prefix()}{build_identity(tool)}-{index}"


def pool_size() -> int:
    size = int(os.environ.get(POOL_SIZE_ENV, POOL_SIZE))
    if size < 1:
        raise ValueError(f"{POOL_SIZE_ENV} must be positive")
    return size


@dataclass
class Server:
    name: str
    pid: int
    tool: str
    retired: bool
    state: dict  # its status file; empty for builds that publish none
    gpu_mib: int | None

    @property
    def generation(self) -> int:
        return int(self.state.get("generation", UNVERSIONED_GENERATION))

    @property
    def idle_seconds(self) -> float | None:
        since = self.state.get("idle_since")
        return time.time() - since if since else None


def servers() -> list[Server]:
    """This user's scoring servers, from the process table (any build)."""
    memory = _gpu_memory_by_pid()
    found = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if len(argv) < 3 or argv[1] not in (b"--serve", b"--serve-retired"):
            continue
        name = argv[2].decode(errors="replace")
        if not name.startswith(name_prefix()):
            continue
        pid = int(proc.name)
        found.append(Server(name, pid, argv[0].decode(errors="replace"),
                            argv[1] == b"--serve-retired", _state(name), memory.get(pid)))
    return sorted(found, key=lambda s: s.name)


@functools.lru_cache(maxsize=None)
def tool_generation(tool: Path) -> int:
    """The server generation of `tool` (UNVERSIONED_GENERATION for builds
    that cannot say)."""
    try:
        run = subprocess.run([str(tool), "--serve-generation"], stdin=subprocess.DEVNULL,
                             capture_output=True, text=True, timeout=DRAIN_TIMEOUT_SECONDS)
    except (OSError, subprocess.SubprocessError):
        return UNVERSIONED_GENERATION
    text = run.stdout.strip()
    return int(text) if run.returncode == 0 and text.isdigit() else UNVERSIONED_GENERATION


def newer_live_servers(found: list[Server], generation: int) -> list[Server]:
    """Live (not retired) servers of a newer generation than `generation`."""
    return [s for s in found if not s.retired and s.generation > generation]


def drain_targets(found: list[Server], tool: Path, generation: int) -> list[Server]:
    """Live servers that `tool` (of `generation`) may drain: older generations only."""
    current = f"{name_prefix()}{build_identity(tool)}-"
    return [s for s in found if not s.retired and not s.name.startswith(current)
            and s.generation < generation]


def drain_older_builds(tool: Path) -> list[str]:
    """Take over the pool for `tool`: fail if a newer build's pool is live,
    else drain every server of an older generation; their names."""
    generation = tool_generation(tool)
    found = servers()
    newer = newer_live_servers(found, generation)
    if newer:
        raise OlderThanLivePool(
            f"your eval tree is older than the live scoring build: its VMAFx {tool} is "
            f"generation {generation}, but server {newer[0].name} runs generation "
            f"{newer[0].generation} ({newer[0].tool}); rebase onto variance-integration")
    drained = []
    for server in drain_targets(found, tool, generation):
        subprocess.run([str(tool), "--drain-server", server.name], stdin=subprocess.DEVNULL,
                       capture_output=True, timeout=DRAIN_TIMEOUT_SECONDS)
        drained.append(server.name)
    return drained


def format_status(found: list[Server]) -> str:
    lines = [f"{'server':<58} {'pid':>8} {'state':<9} {'run':>4} {'wait':>4} {'done':>6} "
             f"{'fail':>5} {'stall':>5} {'gen':>3} {'GPU MiB':>8} {'idle s':>7}  tool"]
    total = 0
    for s in found:
        state = "retired" if s.retired else s.state.get("state", "?")
        idle = s.idle_seconds
        total += s.gpu_mib or 0
        lines.append(f"{s.name:<58} {s.pid:>8} {state:<9} {s.state.get('running', ''):>4} "
                     f"{s.state.get('waiting', ''):>4} {s.state.get('done', ''):>6} "
                     f"{s.state.get('failed', ''):>5} {s.state.get('stalled', ''):>5} {s.generation:>3} {s.gpu_mib if s.gpu_mib is not None else '-':>8} "
                     f"{f'{idle:.0f}' if idle is not None else '-':>7}  {s.tool}")
    lines.append(f"{len(found)} servers, {total} MiB of GPU memory")
    return "\n".join(lines)


def _state(name: str) -> dict:
    try:
        return json.loads((STATE_DIR / f"{STATE_PREFIX}{name}.status").read_text())
    except (OSError, ValueError):
        return {}


def _gpu_memory_by_pid() -> dict[int, int]:
    try:
        run = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=10, check=True)
    except (OSError, subprocess.SubprocessError):
        return {}
    memory = {}
    for line in run.stdout.splitlines():
        pid, _, mib = line.partition(",")
        if pid.strip().isdigit() and mib.strip().isdigit():
            memory[int(pid)] = int(mib)
    return memory
