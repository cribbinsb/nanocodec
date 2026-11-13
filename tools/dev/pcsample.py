#!/usr/bin/env python3
"""Sampling profiler for the Jetson, where perf is unavailable.

The Jetson has no perf or valgrind and `perf_event_paranoid` 2 stops
perf_event-based profilers such as samply. ptrace still works on a child
(`ptrace_scope` 1), so this tool runs the command as its child, stops it every
INTERVAL microseconds (PTRACE_INTERRUPT), reads its program counter and link
register (aarch64 `NT_PRSTATUS`), and lets it go on. Each sample costs tens
of microseconds, so the run is slower but the profile is not skewed by it.
Single-threaded programs only (the encoder is).

    python3 tools/dev/pcsample.py record OUT [--interval US] -- BINARY ARGS...
    python3 tools/dev/pcsample.py report OUT BINARY [--top N] [--callers]

`record` writes OUT (one sample per line: pc and lr relative to the binary's
load base, then the raw pc) and OUT.maps. `report` prints self time per
function (symbols from `nm`, the binary must not be stripped; shared-library
samples are named from their dynamic symbols), or with `--callers` the
functions whose link register points into each shared-library hot spot
(who calls memcpy, read, ...). Samples landing in a library function show
the name of its nearest exported symbol, which for internal glibc routines
can be a neighbour's. Take the Jetson timing lock while recording
(`~/nanocodec-data/jetson-timed.sh python3 tools/dev/pcsample.py record ...`).
"""

import argparse
import bisect
import collections
import ctypes
import os
import subprocess
import sys
import time

PTRACE_CONT = 7
PTRACE_GETREGSET = 0x4204
PTRACE_SEIZE = 0x4206
PTRACE_INTERRUPT = 0x4207
NT_PRSTATUS = 1
AARCH64_REGS = 34  # x0..x30, sp, pc, pstate
PC, LR = 32, 30
SIGTRAP, SIGSTOP = 5, 19
EXEC_SETTLE_S = 0.05  # child waits this long so the parent can seize it first
MAPS_SETTLE_S = 0.2  # time for the dynamic loader to map the binary
MASK64 = (1 << 64) - 1


class IoVec(ctypes.Structure):
    _fields_ = [("base", ctypes.c_void_p), ("len", ctypes.c_size_t)]


def record(out, interval_us, command):
    libc = ctypes.CDLL(None, use_errno=True)
    libc.ptrace.restype = ctypes.c_long
    libc.ptrace.argtypes = [ctypes.c_long, ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p]
    pid = os.fork()
    if pid == 0:
        time.sleep(EXEC_SETTLE_S)
        os.execv(command[0], command)
    if libc.ptrace(PTRACE_SEIZE, pid, None, None) != 0:
        sys.exit(f"ptrace seize failed: errno {ctypes.get_errno()}")
    time.sleep(MAPS_SETTLE_S)
    exe = os.path.realpath(command[0])
    maps = open(f"/proc/{pid}/maps").read()
    with open(out + ".maps", "w") as f:
        f.write(maps)
    base = next(
        int(f[0].split("-")[0], 16)
        for f in (line.split() for line in maps.splitlines())
        if len(f) >= 6 and os.path.realpath(f[5]) == exe and int(f[2], 16) == 0
    )
    regs = (ctypes.c_uint64 * AARCH64_REGS)()
    iov = IoVec(ctypes.cast(regs, ctypes.c_void_p), ctypes.sizeof(regs))
    samples = []
    while libc.ptrace(PTRACE_INTERRUPT, pid, None, None) == 0:
        _, status = os.waitpid(pid, 0)
        if os.WIFEXITED(status) or os.WIFSIGNALED(status):
            break
        if libc.ptrace(PTRACE_GETREGSET, pid, ctypes.c_void_p(NT_PRSTATUS), ctypes.byref(iov)) == 0:
            samples.append((regs[PC], regs[LR]))
        # Pass on real signals; swallow our own group-stop / interrupt traps.
        signal = 0
        if os.WIFSTOPPED(status) and status >> 16 == 0:
            if os.WSTOPSIG(status) not in (SIGTRAP, SIGSTOP):
                signal = os.WSTOPSIG(status)
        libc.ptrace(PTRACE_CONT, pid, None, ctypes.c_void_p(signal))
        time.sleep(interval_us / 1e6)
    with open(out, "w") as f:
        for pc, lr in samples:
            f.write(f"{(pc - base) & MASK64:x} {(lr - base) & MASK64:x} {pc:x}\n")
    print(f"{len(samples)} samples", file=sys.stderr)


class Symbols:
    """Address -> name lookup over one `nm` listing."""

    def __init__(self, path, dynamic):
        flags = ["-D"] if dynamic else ["-C"]
        listing = subprocess.run(
            ["nm", "-n", "--defined-only", *flags, path], capture_output=True, text=True
        ).stdout
        entries = []
        for line in listing.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[1] in "tTwWiI":
                entries.append((int(parts[0], 16), parts[2]))
        self.addresses = [a for a, _ in entries]
        self.names = [n for _, n in entries]

    def name(self, address):
        i = bisect.bisect_right(self.addresses, address) - 1
        return self.names[i] if i >= 0 else "?"


def report(path, binary, top, callers):
    own = Symbols(binary, dynamic=False)
    libraries = []
    for f in (line.split() for line in open(path + ".maps")):
        if len(f) >= 6 and "x" in f[1] and os.path.basename(f[5]) != os.path.basename(binary):
            low, high = (int(v, 16) for v in f[0].split("-"))
            libraries.append((low, high, int(f[2], 16), f[5]))
    library_symbols = {}

    def library_name(raw):
        for low, high, offset, library in libraries:
            if low <= raw < high:
                if library not in library_symbols:
                    library_symbols[library] = Symbols(library, dynamic=True)
                name = library_symbols[library].name(raw - low + offset)
                return f"[{os.path.basename(library)}] {name}"
        return None

    counts = collections.Counter()
    total = 0
    for line in open(path):
        pc, lr, raw = (int(v, 16) for v in line.split())
        total += 1
        in_library = library_name(raw)
        if callers:
            if in_library:
                counts[f"{in_library} <- {own.name(lr)}"] += 1
        else:
            counts[in_library or own.name(pc)] += 1
    for name, count in counts.most_common(top):
        print(f"{100 * count / total:6.2f}%  {name[:160]}")
    print(f"{total} samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("out")
    rec.add_argument("--interval", type=int, default=200, help="microseconds between samples")
    rec.add_argument("command", nargs=argparse.REMAINDER)
    rep = sub.add_parser("report")
    rep.add_argument("out")
    rep.add_argument("binary")
    rep.add_argument("--top", type=int, default=40)
    rep.add_argument("--callers", action="store_true")
    args = parser.parse_args()
    if args.mode == "record":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        record(args.out, args.interval, command)
    else:
        report(args.out, args.binary, args.top, args.callers)


if __name__ == "__main__":
    main()
