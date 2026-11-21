#!/usr/bin/env python3
"""User-space instruction and cycle counts of commands, from the kernel's
hardware counters (perf_event_open through ctypes: no perf, valgrind or root
needed, only `perf_event_paranoid` <= 2). Made for the Jetson, which has
neither perf nor valgrind (paranoid 2); on the PC (paranoid 4) use callgrind.

Instruction counts of a deterministic command are exact run to run (unlike
wall time, which moves ~1% on a shared board), so small CPU changes on ARM
can be judged from one round. Cycles follow memory and branch behaviour too;
take the minimum over a few rounds (~0.5% spread).

    python3 tools/dev/jperf.py [--rounds N] [--core C] [--input F]... \\
        BIN [BIN ...] -- ARG ...

Each BIN runs with the arguments after `--`, on core C (default 3), its
output discarded; `{}` in the arguments is replaced by each --input in turn
(one table per input). Rounds interleave the binaries. Prints the minimum
instructions and cycles per binary, relative to the first.

Example (AV1 decode, two builds, three streams):

    python3 tools/dev/jperf.py --rounds 3 --input s/crowd720.ivf --input s/static720.ivf \\
        bins/base bins/cand -- decode --codec av1 --input {} --output /dev/null
"""

from __future__ import annotations

import argparse
import ctypes
import os
import platform
import struct
import sys
import time
from dataclasses import dataclass

# perf_event_open syscall numbers.
SYSCALLS = {"aarch64": 241, "x86_64": 298}
#: perf_event_attr types: the generic hardware events, and a CPU's own (raw) event numbers.
PERF_TYPE_HARDWARE = 0
PERF_TYPE_RAW = 4
HW_CPU_CYCLES = 0
HW_INSTRUCTIONS = 1
ATTR_SIZE = 128
# perf_event_attr flag bits.
DISABLED = 1 << 0
INHERIT = 1 << 1
EXCLUDE_KERNEL = 1 << 5
EXCLUDE_HV = 1 << 6
ENABLE_ON_EXEC = 1 << 12
# perf_event_attr read_format bits: a read also returns how long the event was enabled
# and how long it ran, which differ when the kernel multiplexes more events than the
# PMU has counters (the count is then partial, so it is dropped).
FORMAT_TOTAL_TIME_ENABLED = 1 << 0
FORMAT_TOTAL_TIME_RUNNING = 1 << 1
READ_FORMAT = FORMAT_TOTAL_TIME_ENABLED | FORMAT_TOTAL_TIME_RUNNING
#: (type, config) of the two counts this tool prints.
INSTRUCTIONS = (PERF_TYPE_HARDWARE, HW_INSTRUCTIONS)
CYCLES = (PERF_TYPE_HARDWARE, HW_CPU_CYCLES)

libc = ctypes.CDLL(None, use_errno=True)


@dataclass
class Run:
    """One counted run: each event's user-space count (None where the kernel refused
    to open it, e.g. a VM without that PMU event), the child's CPU time and wall time."""
    counts: dict[str, int | None]  # None: not offered here, or multiplexed
    user_s: float
    system_s: float
    wall_s: float


def event_attr(kind: int, config: int) -> ctypes.Array:
    """A perf_event_attr counting `config` in user space of the process and
    its children, from its exec on."""
    flags = DISABLED | INHERIT | EXCLUDE_KERNEL | EXCLUDE_HV | ENABLE_ON_EXEC
    head = struct.pack("IIQQQQQ", kind, ATTR_SIZE, config, 0, 0, READ_FORMAT, flags)
    return ctypes.create_string_buffer(head + bytes(ATTR_SIZE - len(head)), ATTR_SIZE)


def measure(argv: list[str], core: int, events: dict[str, tuple[int, int]]) -> Run:
    """One run of `argv` pinned to `core`, its output discarded, with `events`
    ({name: (type, config)}) counted where the kernel offers them."""
    syscall = SYSCALLS[platform.machine()]
    ready_r, ready_w = os.pipe()
    pid = os.fork()
    if pid == 0:
        # The child waits until the counters are attached, then execs; if it cannot, it
        # exits (never returning into the caller's code as a second process).
        try:
            os.close(ready_w)
            os.read(ready_r, 1)
            os.sched_setaffinity(0, {core})
            null = os.open(os.devnull, os.O_WRONLY)
            os.dup2(null, 1)
            os.dup2(null, 2)
            os.execv(argv[0], argv)
        finally:
            os._exit(127)
    os.close(ready_r)
    fds: dict[str, int | None] = {}
    for name, (kind, config) in events.items():
        fd = libc.syscall(syscall, event_attr(kind, config), pid, -1, -1, 0)
        fds[name] = fd if fd >= 0 else None
    started = time.perf_counter()
    os.write(ready_w, b"x")
    os.close(ready_w)
    _, status, usage = os.wait4(pid, 0)
    wall = time.perf_counter() - started
    if os.waitstatus_to_exitcode(status) != 0:
        raise RuntimeError(f"jperf: {' '.join(argv)} failed")
    counts = {name: read_count(fd) if fd is not None else None for name, fd in fds.items()}
    for fd in fds.values():
        if fd is not None:
            os.close(fd)
    return Run(counts, usage.ru_utime, usage.ru_stime, wall)


def read_count(fd: int) -> int | None:
    """An event's count, or None when it did not run for the whole of its time."""
    value, enabled, running = struct.unpack("QQQ", os.read(fd, 24))
    return value if running == enabled else None


def count(argv: list[str], core: int) -> tuple[int, int]:
    """(instructions, cycles) of one run of `argv` pinned to `core`."""
    try:
        run = measure(argv, core, {"instructions": INSTRUCTIONS, "cycles": CYCLES})
    except RuntimeError as error:
        sys.exit(str(error))
    if None in run.counts.values():
        sys.exit("jperf: perf_event_open failed; needs /proc/sys/kernel/perf_event_paranoid "
                 "<= 2 (the Jetson has 2)")
    return run.counts["instructions"], run.counts["cycles"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--core", type=int, default=3)
    parser.add_argument("--input", action="append", help="repeatable")
    parser.add_argument("bins", nargs="+")
    if "--" not in sys.argv:
        parser.error("arguments for the binaries follow `--`")
    split = sys.argv.index("--")
    options = parser.parse_args(sys.argv[1:split])
    command = sys.argv[split + 1 :]
    bins = [os.path.abspath(b) for b in options.bins]
    for item in options.input or [None]:
        args = [a.replace("{}", item) if item else a for a in command]
        runs: dict[str, list[tuple[int, int]]] = {b: [] for b in bins}
        for _ in range(options.rounds):
            for b in bins:
                runs[b].append(count([b, *args], options.core))
        base_i = min(i for i, _ in runs[bins[0]])
        base_c = min(c for _, c in runs[bins[0]])
        print(f"== {item or ' '.join(command)}")
        for b in bins:
            i = min(v for v, _ in runs[b])
            c = min(v for _, v in runs[b])
            print(f"  {os.path.basename(b):24s} instr {i / 1e9:8.4f}G ({100 * (i / base_i - 1):+.2f}%)"
                  f"  cycles {c / 1e9:8.4f}G ({100 * (c / base_c - 1):+.2f}%)")


if __name__ == "__main__":
    main()
