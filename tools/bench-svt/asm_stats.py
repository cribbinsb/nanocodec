#!/usr/bin/env python3
"""Instruction statistics of one function's disassembly (Intel syntax).

Usage:
    asm_stats.py <binary-or-archive> <symbol substring> [--loop]

Disassembles the function whose (demangled) name contains the substring and
prints: instructions, vector instructions, loads, stores, stack (spill)
accesses, calls, panic calls, conditional branches and the loops found
(backward branches) with their instruction counts. With --loop it also
prints the body of the largest innermost loop.

x86-64 and AArch64 are recognised (memory operands `[...]` in both).
"""

from __future__ import annotations

import re
import subprocess
import sys

LINE = re.compile(r"^\s*([0-9a-f]+):\s+(?:[0-9a-f]{2} )*\s*([a-z][\w.]*)\s*(.*)$")


def disassemble(path: str, symbol: str) -> list[tuple[int, str, str]]:
    out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", "-M", "intel", path],
                         capture_output=True, text=True, check=False).stdout
    if "unrecognized option" in out or not out:
        out = subprocess.run(["objdump", "-d", "-C", "--no-show-raw-insn", path],
                             capture_output=True, text=True, check=True).stdout
    lines, inside = [], False
    for raw in out.splitlines():
        if raw.endswith(">:"):
            inside = symbol in raw
            continue
        if inside:
            m = LINE.match(raw)
            if m:
                lines.append((int(m.group(1), 16), m.group(2), m.group(3)))
    # Hand-written asm splits a function into local labels (`name.loop`),
    # each its own symbol: every symbol containing the substring counts.
    return lines


def is_store(op: str, args: str) -> bool:
    if op.startswith(("st", "vst")):  # AArch64 stores
        return True
    first = args.split(",")[0]
    return "[" in first and not op.startswith(("cmp", "test", "prefetch", "vptest"))


def stats(lines: list[tuple[int, str, str]]) -> dict[str, object]:
    s = dict(instructions=len(lines), vector=0, loads=0, stores=0, stack=0, calls=0,
             panics=0, branches=0, loops=[])
    for addr, op, args in lines:
        if re.search(r"\b[xyz]mm\d|\bv\d+\.|\bq\d+\b", args):
            s["vector"] += 1
        if "[" in args:
            if is_store(op, args):
                s["stores"] += 1
            else:
                s["loads"] += 1
            if re.search(r"\[(rsp|rbp|sp)\b", args):
                s["stack"] += 1
        if op.startswith(("call", "bl")) and not op.startswith("blt") or op == "blr":
            s["calls"] += 1
            if re.search(r"panic|bounds_check|slice_(start|end)_index|unwrap_failed", args):
                s["panics"] += 1
        if (op.startswith("j") and op != "jmp") or op.startswith(("b.", "cbz", "cbnz", "tbz", "tbnz")):
            s["branches"] += 1
            target = re.search(r"\b([0-9a-f]+)\b", args)
            if target and int(target.group(1), 16) < addr:
                start = int(target.group(1), 16)
                body = sum(1 for a, _, _ in lines if start <= a <= addr)
                s["loops"].append((start, addr, body))
    return s


def main() -> None:
    path, symbol = sys.argv[1], sys.argv[2]
    lines = disassemble(path, symbol)
    if not lines:
        sys.exit(f"symbol containing {symbol!r} not found in {path}")
    s = stats(lines)
    loops = s.pop("loops")
    print(f"{symbol}: " + ", ".join(f"{k} {v}" for k, v in s.items()))
    for start, end, body in loops:
        print(f"  loop {start:x}..{end:x}: {body} instructions")
    if "--loop" in sys.argv and loops:
        # The largest loop that contains no other loop.
        inner = [l for l in loops if not any(o is not l and l[0] <= o[0] and o[1] <= l[1] for o in loops)]
        start, end, _ = max(inner, key=lambda l: l[2])
        for addr, op, args in lines:
            if start <= addr <= end:
                print(f"    {op:<14}{args}")


if __name__ == "__main__":
    main()
