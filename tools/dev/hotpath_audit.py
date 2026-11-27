#!/usr/bin/env python3
"""Hot-path audit of one callgrind profile: the overheads that self-cost rankings hide.

    python3 tools/dev/hotpath_audit.py PROFILE [--binary BIN] [--top N] [--fn SUBSTR [--lines]]

PROFILE is a release callgrind profile (`valgrind --tool=callgrind --dump-instr=yes`,
a `profiling`-profile build so inlined code keeps its source files). Four lists:

1. functions by self cost, each split by the source file its instructions came from
   (inlined `core::iter`, `alloc::vec` or `slice::index` code shows up here; the file
   names where the work came from, not that it is a bounds check);
2. small out-of-line functions: at least MIN_CALLS calls and at most SMALL_IR
   instructions a call, so call overhead (and work a caller discards, which inlining
   would remove) is a large share;
3. memcpy / memset / malloc by caller, with calls and instructions a call: hidden
   copies (a `const` table indexed at run time, an `Option` of a large `Copy` value
   taken by value, a struct swapped by value) appear only as libc symbols otherwise;
4. with --binary (the same build): the fixed byte size of each hot caller's memcpy /
   memset sites, from its disassembly, and the copies whose source is read-only data
   (`lea ...(%rip)`): a constant copied to the stack on every call;
5. tiny hot loops outside the nc-dsp kernels: per function, the most-executed
   instruction's count (the loop's trips) and the instructions executed at least half
   as often (its body); a body of at most TINY_BODY instructions costing LOOP_SHARE of
   the total is a scan, fill or clear inside a larger function (an `any()` over a
   coefficient block, a per-picture scan of a block map), invisible in list 1;
6. with --binary: functions called at least PROLOGUE_CALLS times whose register saves
   and restores (`push`, `pop`, `vzeroupper`, `ret`) are PROLOGUE_SHARE or more of
   their self cost: a hot path paying for registers only a rare path needs (split the
   rare path out of line).

Why (docs/PROCEDURES.md "Hot-path audit"): on 2026-11-27 this found a 1 KiB const
table memcpy'd per call (VP9 -3.8 %), a 760-byte Option copy (H.264 -0.85 %) and a
792-byte struct swap (HEVC -0.6 %) that many rounds of kernel profiling had missed;
lists 5 and 6 come from the decoder audit of the same day (AV2 -3.0 % and HEVC
-1.8 % from scans, AV2 -0.65 % from a four-symbol read's register saves;
docs/handoff/decoder-audit-2026-11-27.md).
Reads the profile only; writes nothing.
"""
from __future__ import annotations

import argparse
import collections
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

#: A function called at least this often...
MIN_CALLS = 200_000
#: ...for at most this many instructions a call is listed as a small out-of-line helper.
SMALL_IR = 40
#: A file contributing less than this share of a function's self cost is not listed.
FILE_SHARE = 0.05
#: libc symbols whose callers are listed.
LIBC = ("memcpy", "memmove", "memset", "malloc", "realloc", "free")
#: Instructions searched back from a memcpy call for its size and source.
TRACE_BACK = 300
#: A loop body of at most this many instructions...
TINY_BODY = 8
#: ...costing at least this share of the total is listed as a tiny hot loop.
LOOP_SHARE = 0.003
#: Functions named with this prefix are kernels, whose loops are the work itself.
KERNEL_PREFIX = "nc_dsp::"
#: A function called at least this often...
PROLOGUE_CALLS = 1_000_000
#: ...whose register saves and restores are at least this share of its self cost.
PROLOGUE_SHARE = 0.2
#: The instructions of a function's register saves and restores.
PROLOGUE_MNEMONICS = ("push", "pop", "vzeroupper", "ret")


@dataclass
class Profile:
    """Self cost per (function, source file), and call edges."""

    self_cost: collections.Counter = field(default_factory=collections.Counter)
    line_cost: collections.Counter = field(default_factory=collections.Counter)
    #: (function, instruction address) -> self Ir, which is its execution count.
    instr_cost: collections.Counter = field(default_factory=collections.Counter)
    #: instruction address -> (source file, line) of its cost.
    instr_line: dict = field(default_factory=dict)
    #: (caller, callee) -> [calls, inclusive Ir]
    edges: dict = field(default_factory=lambda: collections.defaultdict(lambda: [0, 0]))

    def per_function(self) -> collections.Counter:
        out = collections.Counter()
        for (fn, _), ir in self.self_cost.items():
            out[fn] += ir
        return out

    def calls_to(self) -> collections.Counter:
        out = collections.Counter()
        for (_, callee), (calls, _) in self.edges.items():
            out[callee] += calls
        return out


def parse(path: Path) -> Profile:
    """The callgrind format with name compression (`fn=(12) name`, later `fn=(12)`)."""
    names = {"file": {}, "fn": {}}
    profile = Profile()
    fn = file = function_file = callee = None
    pending_calls = None
    line_no = 0
    address = 0

    def resolve(kind: str, value: str) -> str:
        match = re.match(r"\((\d+)\)\s*(.*)", value)
        if not match:
            return value
        ident, name = match.groups()
        if name:
            names[kind][ident] = name
        return names[kind].get(ident, value)

    with open(path, errors="replace") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if not line:
                continue
            key, _, value = line.partition("=")
            if key == "fl":
                file = function_file = resolve("file", value)
            elif key in ("fi", "fe"):
                file = resolve("file", value)
            elif key == "fn":
                fn = resolve("fn", value)
                file = function_file
            elif key in ("cfl", "cfi"):
                resolve("file", value)
            elif key == "cfn":
                callee = resolve("fn", value)
            elif key == "calls":
                pending_calls = int(value.split()[0])
            elif line[0].isdigit() or line[0] in "+-*":
                parts = line.split()
                # `positions: instr line`: the address, absolute or relative.
                if parts[0].startswith("0x"):
                    address = int(parts[0], 16)
                elif parts[0][0] in "+-":
                    address += int(parts[0])
                if pending_calls is not None:
                    edge = profile.edges[(fn, callee)]
                    edge[0] += pending_calls
                    edge[1] += int(parts[-1])
                    pending_calls = None
                    continue
                # `positions: instr line`: address, line, Ir.
                if len(parts) < 3:
                    continue
                position = parts[1]
                if position[0] in "+-":
                    line_no += int(position)
                elif position != "*":
                    line_no = int(position)
                profile.self_cost[(fn, file)] += int(parts[2])
                profile.line_cost[(fn, file, line_no)] += int(parts[2])
                profile.instr_cost[(fn, address)] += int(parts[2])
                profile.instr_line[address] = (file, line_no)
    return profile


def short(path: str) -> str:
    for marker in ("/library/", "/crates/", "/src/"):
        if marker in path:
            return path[path.index(marker) + 1:]
    return path


def report_self(profile: Profile, total: int, top: int, wanted: str | None, lines: bool):
    calls = profile.calls_to()
    ranked = [(f, ir) for f, ir in profile.per_function().most_common()
              if wanted is None or wanted in f][:top]
    print(f"== self cost (total {total:,} Ir)")
    for fn, ir in ranked:
        n = calls.get(fn, 0)
        per_call = f"  {ir / n:.0f}/call x{n:,}" if n else ""
        print(f"{ir:>14,} {100 * ir / total:5.2f}%  {fn[:130]}{per_call}")
        files = sorted(((c, f) for (g, f), c in profile.self_cost.items() if g == fn), reverse=True)
        for cost, file in files[:6]:
            if cost >= FILE_SHARE * ir:
                print(f"{'':>17}{100 * cost / ir:5.1f}%  {short(file)}")
        if lines:
            hot = sorted(((c, f, n) for (g, f, n), c in profile.line_cost.items() if g == fn),
                         reverse=True)
            for cost, file, number in hot[:12]:
                print(f"{'':>21}{cost:>12,}  {short(file)}:{number}")


def report_small(profile: Profile, total: int):
    print(f"\n== small out-of-line functions (>= {MIN_CALLS:,} calls, <= {SMALL_IR} Ir/call)")
    own = profile.per_function()
    rows = sorted(((own[f], n, f) for f, n in profile.calls_to().items()
                   if n >= MIN_CALLS and own[f] <= SMALL_IR * n), reverse=True)
    for ir, n, fn in rows[:25]:
        print(f"{ir:>14,} {100 * ir / total:5.2f}%  {ir / n:5.1f}/call x{n:>10,}  {fn[:120]}")


def report_libc(profile: Profile, total: int) -> list[str]:
    """Callers of the libc copy / fill / allocation symbols; returns the callers listed."""
    print("\n== memcpy / memset / allocation by caller (inclusive)")
    rows = sorted(((ir, n, caller, callee) for (caller, callee), (n, ir) in profile.edges.items()
                   if any(s in callee for s in LIBC) and not any(s in caller for s in LIBC)),
                  reverse=True)
    listed = []
    for ir, n, caller, callee in rows[:30]:
        if ir < total // 1000:
            break
        symbol = next(s for s in LIBC if s in callee)
        print(f"{ir:>14,} {100 * ir / total:5.2f}%  {ir / n:7.0f}/call x{n:>9,}  {symbol:7} "
              f"{caller[:110]}")
        listed.append(caller)
    return listed


def instructions_by_function(profile: Profile) -> dict:
    """function -> {address: self Ir}."""
    out = collections.defaultdict(dict)
    for (fn, address), ir in profile.instr_cost.items():
        out[fn][address] = ir
    return out


def report_loops(profile: Profile, total: int):
    """Tiny hot loops outside the kernels (list 5)."""
    print(f"\n== tiny hot loops (body <= {TINY_BODY} instructions, >= {100 * LOOP_SHARE:.1f} %"
          f" of the total, outside {KERNEL_PREFIX})")
    rows = []
    for fn, costs in instructions_by_function(profile).items():
        # Kernels loop by design; libc's own loops are list 3, by caller.
        if fn.startswith(KERNEL_PREFIX) or any(symbol in fn for symbol in LIBC):
            continue
        trips = max(costs.values())
        body = [address for address, ir in costs.items() if 2 * ir >= trips]
        loop_ir = sum(costs[address] for address in body)
        if len(body) <= TINY_BODY and loop_ir >= LOOP_SHARE * total:
            rows.append((loop_ir, trips, len(body), fn, min(body)))
    for loop_ir, trips, size, fn, first in sorted(rows, reverse=True)[:25]:
        file, number = profile.instr_line.get(first, ("?", 0))
        print(f"{loop_ir:>14,} {100 * loop_ir / total:5.2f}%  {trips:>12,} trips x {size}  "
              f"{short(file)}:{number}  {fn[:90]}")


def report_prologues(binary: Path, profile: Profile, total: int):
    """Hot functions whose register saves and restores are a large share (list 6)."""
    print(f"\n== register saves and restores (>= {PROLOGUE_CALLS:,} calls, >= "
          f"{100 * PROLOGUE_SHARE:.0f} % of self cost)")
    calls = profile.calls_to()
    hot = {fn: costs for fn, costs in instructions_by_function(profile).items()
           if calls.get(fn, 0) >= PROLOGUE_CALLS}
    wanted = {address for costs in hot.values() for address in costs}
    mnemonic = {}
    dump = subprocess.run(["objdump", "-d", "--no-show-raw-insn", str(binary)],
                          capture_output=True, text=True, check=True).stdout
    for line in dump.splitlines():
        match = re.match(r"\s*([0-9a-f]+):\s+(\S+)", line)
        if match and int(match.group(1), 16) in wanted:
            mnemonic[int(match.group(1), 16)] = match.group(2)
    rows = []
    for fn, costs in hot.items():
        own = sum(costs.values())
        saved = sum(ir for address, ir in costs.items()
                    if mnemonic.get(address, "").startswith(PROLOGUE_MNEMONICS))
        if saved >= PROLOGUE_SHARE * own:
            rows.append((own, saved, calls[fn], fn))
    for own, saved, n, fn in sorted(rows, reverse=True)[:25]:
        print(f"{own:>14,} {100 * own / total:5.2f}%  saves {100 * saved / own:4.1f}%  "
              f"x{n:>10,}  {fn[:100]}")


def report_fixed_copies(binary: Path, callers: list[str]):
    """Fixed-size memcpy / memset sites of `callers`, and memcpy from read-only data."""
    print("\n== fixed-size copies in those callers (bytes; * = source is read-only data)")
    dump = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", str(binary)],
                          capture_output=True, text=True, check=True).stdout
    wanted = {unhashed(c) for c in callers}
    for name, body in functions(dump):
        name = unhashed(name)
        if name not in wanted:
            continue
        sites = sorted({site for site in copy_sites(body)})
        if sites:
            text = ", ".join(f"{kind} {size}{'*' if constant else ''}"
                             for kind, size, constant in sites)
            print(f"  {text:40}  {name[:110]}")


def unhashed(name: str) -> str:
    """`name` without the crate disambiguators (`nc_vp9[9ccc6212d1960f00]`) that
    objdump prints and callgrind does not."""
    return re.sub(r"\[[0-9a-f]{16}\]", "", name)


def functions(dump: str):
    name, body = None, []
    for line in dump.splitlines():
        match = re.match(r"^[0-9a-f]+ <(.*)>:$", line)
        if match:
            if name:
                yield name, body
            name, body = match.group(1), []
        elif name:
            body.append(line.rstrip())
    if name:
        yield name, body


def copy_sites(body: list[str]):
    """(memcpy|memset, immediate size, source traced to a rip-relative address)."""
    for i, line in enumerate(body):
        if "call" not in line or not ("memcpy" in line or "memset" in line):
            continue
        kind = "memcpy" if "memcpy" in line else "memset"
        size, source, constant = None, "%rsi", False
        for back in reversed(body[max(0, i - TRACE_BACK):i]):
            # The operands, without objdump's `# address <symbol>` comment.
            back = back.split("#")[0].rstrip()
            if size is None and (m := re.search(r"mov\s+\$0x([0-9a-f]+),%edx", back)):
                size = int(m.group(1), 16)
            m = re.search(r"\s(lea|mov)\s+(\S+),(%r\w+)$", back)
            if kind == "memcpy" and m and m.group(3) == source:
                if m.group(1) == "lea" and "(%rip)" in m.group(2):
                    constant = True
                    break
                if re.fullmatch(r"%r\w+", m.group(2)):
                    source = m.group(2)
                    continue
                break
        if size is not None:
            yield kind, size, constant


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("profile", type=Path)
    parser.add_argument("--binary", type=Path, help="the profiled build, for the copy sizes")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--fn", help="only functions whose name contains this")
    parser.add_argument("--lines", action="store_true", help="hottest source lines too")
    args = parser.parse_args()
    profile = parse(args.profile)
    total = sum(profile.self_cost.values())
    report_self(profile, total, args.top, args.fn, args.lines)
    if args.fn:
        return 0
    report_small(profile, total)
    callers = report_libc(profile, total)
    if args.binary:
        report_fixed_copies(args.binary, callers)
    report_loops(profile, total)
    if args.binary:
        report_prologues(args.binary, profile, total)
    return 0


if __name__ == "__main__":
    sys.exit(main())
