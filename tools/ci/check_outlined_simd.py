#!/usr/bin/env python3
"""Fail if a built binary holds SIMD code compiled outside its `#[target_feature]` context.

    python3 tools/ci/check_outlined_simd.py BINARY [BINARY ...]

An ISA enabled per function (x86 AVX2/AVX-512, AArch64 extensions such as dotprod) reaches
only the code inlined into a `#[target_feature]` function. A generic helper the compiler
keeps out of line (a closure, `array::map`, `array::from_fn`, an `#[inline(never)]` generic
function) is compiled without that ISA: every intrinsic in it becomes a call to an
out-of-line `core::arch` function. One such fallback made a dense 32x32 AV1 inverse take
29.6 us instead of 0.27 us (docs/handoff/avx512.md, 2026-11-17).

Two mechanical rules over the symbol table / disassembly (ELF through `objdump`, WebAssembly
through `wasm-objdump`):

1. a `core::core_arch` function defined in the binary: an intrinsic that was not inlined
   (only the CPU-feature detection's own `_xgetbv`/`cpuid` are allowed);
2. an out-of-line `array::map` or `array::from_fn` instance whose elements are AVX2/AVX-512
   vectors (`__m256i`, `__m512i` or one of our lane types): it moves them through memory with
   baseline instructions. (An outlined closure that does vector arithmetic calls intrinsics,
   so rule 1 finds it.)

Every hit is listed with the functions that call it; exit status 1 when there is any.
"""

from __future__ import annotations

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

INTRINSIC = re.compile(r"^core::core_arch::")
#: Runtime CPU detection's own helpers (std_detect), not kernels.
DETECTION = re.compile(r"^core::core_arch::(x86|x86_64)::(xsave::_xgetbv|cpuid::)")
#: An array helper (`array::map`, `array::from_fn`) whose element type is a per-function x86
#: vector: the vector type or an AVX2/AVX-512 lane type of ours.
VECTOR = r"(core::core_arch::x86::__m(256|512)i|nc_dsp::[A-Za-z0-9_:]*(avx2|avx512)::[A-Z][A-Za-z0-9]*)"
VECTOR_HELPER = re.compile(rf"(try_from_fn|try_map)::<core::ops::try_trait::NeverShortCircuit<{VECTOR}>"
                           rf"|>::map::<.*, {VECTOR}>$")
ELF_FUNCTION = re.compile(r"^[0-9a-f]+ <(.+)>:$")
ELF_CALL = re.compile(r"\b(?:call|bl|b)\s+[0-9a-f]+ <(.+?)(\+0x[0-9a-f]+)?>$")


def is_hit(name: str) -> bool:
    if INTRINSIC.match(name):
        return not DETECTION.match(name)
    return bool(VECTOR_HELPER.search(name))


def elf_hits(path: Path) -> dict[str, set[str]]:
    """Each offending function defined in the ELF binary, with its callers."""
    dump = subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-C", str(path)],
                          check=True, capture_output=True, text=True).stdout
    hits: dict[str, set[str]] = defaultdict(set)
    current = None
    for line in dump.splitlines():
        if match := ELF_FUNCTION.match(line):
            current = match.group(1)
            if is_hit(current):
                hits.setdefault(current, set())
        elif current and (match := ELF_CALL.search(line)) and is_hit(match.group(1)):
            hits[match.group(1)].add(current)
    return hits


def wasm_hits(path: Path) -> dict[str, set[str]]:
    """Each offending function named in the WebAssembly module (callers not resolved)."""
    dump = subprocess.run(["wasm-objdump", "-x", "-j", "name", str(path)],
                          check=True, capture_output=True, text=True).stdout
    names = re.findall(r'^ - func\[\d+\] <(.+)>$', dump, re.M)
    demangled = subprocess.run(["c++filt"], input="\n".join(names), check=True,
                               capture_output=True, text=True).stdout.splitlines()
    return {name: set() for name in demangled if is_hit(re.sub(r"::h[0-9a-f]{16}$", "", name))}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__, file=sys.stderr)
        return 2
    failed = False
    for arg in sys.argv[1:]:
        path = Path(arg)
        hits = wasm_hits(path) if path.suffix == ".wasm" else elf_hits(path)
        print(f"{path}: {len(hits)} outlined SIMD function(s)")
        for name, callers in sorted(hits.items()):
            failed = True
            print(f"  {name}")
            for caller in sorted(callers):
                print(f"    called from {caller}")
    if failed:
        print("SIMD code was compiled outside its #[target_feature] context: make the helper a "
              "plain #[inline(always)] function or loop (not a closure, array::map or "
              "array::from_fn), or give the outlined function the ISA's target feature.",
              file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
