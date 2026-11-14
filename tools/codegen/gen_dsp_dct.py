#!/usr/bin/env python3
"""Generate crates/nc-dsp/src/txfm/dct.rs: AV1's 1-D DCTs over SIMD lanes.

nc-dsp evaluates each 1-D transform on several columns (or rows) at once:
the butterfly network is identical for every lane, so it is written once,
generic over a minimal lane trait (`ForwardLanes` / `InverseLanes`, which
every `I32Lanes` type implements), and instantiated per ISA (and with a
one-lane scalar type as the reference). Guarded 16-bit NEON lanes implement
the same traits.

libaom writes its 1-D DCTs as long, completely regular butterfly listings
over two ping-pong buffers (`bf1[i] = half_btf(...)`,
`clamp_value(bf0[a] + bf0[b], stage_range[stage])`). This script translates
them mechanically from the pinned libaom tree (BSD-2-Clause, Copyright (c)
2016, Alliance for Open Media; see THIRD_PARTY.md):

  av1/encoder/av1_fwd_txfm1d.c  av1_fdct4/8/16/32/64  (encoder only)
  av1/common/av1_inv_txfm1d.c   av1_idct4/8/16/32/64  (normative, AV1 §7.13.2)

Translation rules:
  - `output` is the `&mut [V; N]` argument, `step` a local array; `bf0` and
    `bf1` are tracked as aliases of `input` / `output` / `step`;
  - `a + b`, `a - b`, `-a + b` become lane `add` / `sub`;
  - `half_btf(w0, a, w1, b, cos_bit)` becomes `V::half_btf` with `cospi[k]`
    taken from the table argument;
  - `clamp_value(x, stage_range[stage])` becomes `V::clamp16`: at 8-bit depth
    every inverse stage range is 16 bits (libaom `av1_gen_inv_stage_range`:
    row `bd + 8`, column `max(bd + 6, 16)`);
  - in the sparse (`_lowN`) inverse graphs, a `half_btf` whose second input is
    a structural zero becomes `V::mul_round(w, a, cos_bit)`;
  - range checks, stage counters and declarations are dropped.

Usage:
    gen_dsp_dct.py <libaom source dir> <output .rs file>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FUNCTIONS = [
    ("av1/encoder/av1_fwd_txfm1d.c", "av1_fdct4", "fdct4", 4),
    ("av1/encoder/av1_fwd_txfm1d.c", "av1_fdct8", "fdct8", 8),
    ("av1/encoder/av1_fwd_txfm1d.c", "av1_fdct16", "fdct16", 16),
    ("av1/encoder/av1_fwd_txfm1d.c", "av1_fdct32", "fdct32", 32),
    ("av1/encoder/av1_fwd_txfm1d.c", "av1_fdct64", "fdct64", 64),
    ("av1/common/av1_inv_txfm1d.c", "av1_idct4", "idct4", 4),
    ("av1/common/av1_inv_txfm1d.c", "av1_idct8", "idct8", 8),
    ("av1/common/av1_inv_txfm1d.c", "av1_idct16", "idct16", 16),
    ("av1/common/av1_inv_txfm1d.c", "av1_idct32", "idct32", 32),
    ("av1/common/av1_inv_txfm1d.c", "av1_idct64", "idct64", 64),
]

REF = r"(\w+)\[(\d+)\]"
WEIGHT = r"(-?)cospi\[(\d+)\]"
ASSIGN = re.compile(r"^\s*bf1\[(\d+)\]\s*=\s*(.+);\s*$")
ALIAS = re.compile(r"^\s*(bf0|bf1)\s*=\s*(\w+);\s*$")
HALF_BTF = re.compile(rf"^half_btf\({WEIGHT},\s*{REF},\s*{WEIGHT},\s*{REF},\s*cos_bit\)$")
CLAMP = re.compile(r"^clamp_value\((.+),\s*stage_range\[stage\]\)$")
SUM = re.compile(rf"^(-?){REF}\s*([+-])\s*{REF}$")
COPY = re.compile(rf"^{REF}$")


def function_body(text: str, name: str) -> list[str]:
    start = text.index(f"void {name}(")
    open_brace = text.index("{", start)
    depth, i = 0, open_brace
    while True:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[open_brace + 1:i].splitlines()


def lane_ref(name: str, index: str, alias: dict[str, str | None]) -> str:
    target = alias[name] if name in alias else name
    if target not in ("input", "output", "step"):
        raise ValueError(f"unknown buffer {name!r}")
    return f"{target}[{index}]"


def translate_expr(expr: str, alias: dict[str, str | None]) -> str:
    expr = expr.strip()
    if m := CLAMP.match(expr):
        return f"V::clamp16({translate_expr(m.group(1), alias)})"
    if m := HALF_BTF.match(expr):
        s0, k0, n0, i0, s1, k1, n1, i1 = m.groups()
        a = lane_ref(n0, i0, alias)
        b = lane_ref(n1, i1, alias)
        return f"V::half_btf({s0}cospi[{k0}], {a}, {s1}cospi[{k1}], {b}, cos_bit)"
    if m := SUM.match(expr):
        neg, n0, i0, op, n1, i1 = m.groups()
        a = lane_ref(n0, i0, alias)
        b = lane_ref(n1, i1, alias)
        if neg:
            if op != "+":
                raise ValueError(f"unsupported expression {expr!r}")
            return f"{b}.sub({a})"
        return f"{a}.add({b})" if op == "+" else f"{a}.sub({b})"
    if m := COPY.match(expr):
        return lane_ref(m.group(1), m.group(2), alias)
    raise ValueError(f"unsupported expression {expr!r}")


def translate_sparse_expr(expr: str, alias: dict[str, str | None], zeros: set[str]) -> str | None:
    """Propagate structural zeros, preserving every nonzero stage clamp/rounding."""
    expr = expr.strip()
    def ref(name, index):
        value = lane_ref(name, index, alias)
        return None if value in zeros else value
    if m := CLAMP.match(expr):
        value = translate_sparse_expr(m.group(1), alias, zeros)
        return None if value is None else f"V::clamp16({value})"
    if m := HALF_BTF.match(expr):
        s0,k0,n0,i0,s1,k1,n1,i1 = m.groups()
        a,b = ref(n0,i0),ref(n1,i1)
        if a is None and b is None:
            return None
        if a is None:
            return f"V::mul_round({s1}cospi[{k1}], {b}, cos_bit)"
        if b is None:
            return f"V::mul_round({s0}cospi[{k0}], {a}, cos_bit)"
        return f"V::half_btf({s0}cospi[{k0}], {a}, {s1}cospi[{k1}], {b}, cos_bit)"
    if m := SUM.match(expr):
        neg,n0,i0,op,n1,i1 = m.groups()
        a,b = ref(n0,i0),ref(n1,i1)
        if neg:
            if op != "+": raise ValueError(expr)
            a,b,op = b,a,"-"
        if a is None and b is None: return None
        if b is None: return a
        if a is None: return b if op == "+" else f"V::splat(0).sub({b})"
        return f"{a}.add({b})" if op == "+" else f"{a}.sub({b})"
    if m := COPY.match(expr): return ref(m.group(1),m.group(2))
    raise ValueError(f"unsupported sparse expression {expr!r}")


def translate(lines: list[str], rust_name: str, size: int, nonzero: int | None = None) -> str:
    alias: dict[str, str | None] = {"bf0": None, "bf1": None}
    body: list[str] = []
    uses_step = False
    zeros = {f"input[{i}]" for i in range(nonzero or size, size)}
    for line in lines:
        if m := ALIAS.match(line):
            alias[m.group(1)] = m.group(2)
            continue
        if m := ASSIGN.match(line):
            target = alias["bf1"]
            if target not in ("output", "step"):
                raise ValueError(f"{rust_name}: bf1 aliases {target!r}")
            slot = f"{target}[{m.group(1)}]"
            rhs = (translate_expr(m.group(2), alias) if nonzero is None else
                   translate_sparse_expr(m.group(2), alias, zeros))
            if rhs is None:
                zeros.add(slot)
                continue
            zeros.discard(slot)
            uses_step |= target == "step"
            uses_step |= "step[" in rhs
            body.append(f"    {target}[{m.group(1)}] = {rhs};")
            continue
        if "stage" in line and "//" in line:
            body.append(f"    // {line.strip().lstrip('/').strip()}")
    if nonzero is not None:
        body.extend(f"    output[{i}] = V::splat(0);" for i in range(size) if f"output[{i}]" in zeros)
    lane_trait = ("super::forward_lanes::ForwardLanes" if rust_name.startswith("fdct")
                  else "super::inverse_lanes::InverseLanes")
    head = [
        f"/// libaom `av1_{rust_name.split('_low')[0]}`"
        + (f" with only the first {nonzero} inputs nonzero" if nonzero else " on every lane")
        + " (GENERATED; see module docs).",
        "#[inline(always)]",
        f"pub(crate) fn {rust_name}<V: {lane_trait}>(",
        f"    input: &[V; {size}],",
        f"    output: &mut [V; {size}],",
        "    cospi: &[i32; 64],",
        "    cos_bit: u32,",
        ") {",
    ]
    if uses_step:
        head.append(f"    let mut step = [V::splat(0); {size}];")
    return "\n".join(head + body + ["}"])


def forward_low_half_specialization(full: str, size: int) -> str:
    """Share the full graph, explicitly pruning dependencies of discarded outputs.

    LLVM does not reliably remove these operations from aliased scratch arrays.
    Reverse liveness must track each assignment, not merely each array slot:
    output/step alternate roles between stages.
    """
    keep = size // 2
    assignments = list(re.finditer(r"((?:output|step)\[\d+\])\s*=\s*(.*?);", full, re.S))
    live = {f"output[{i}]" for i in range(keep)}
    dead = []
    for match in reversed(assignments):
        if match[1] in live:
            live.remove(match[1])
            live.update(re.findall(r"(?:output|step)\[\d+\]", match[2]))
        else:
            dead.append((match.start(), match.end()))
    assert not live
    groups = []
    for start, end in sorted(dead):
        if groups and not full[groups[-1][1]:start].strip():
            groups[-1] = (groups[-1][0], end)
        else:
            groups.append((start, end))
    for start, end in reversed(groups):
        statements = full[start:end].replace("\n", "\n    ")
        full = full[:start] + f"if !LOW{keep} {{\n        " + statements + "\n    }" + full[end:]
    full = full.replace(f"pub(crate) fn fdct{size}<V: super::forward_lanes::ForwardLanes>(",
                        f"fn fdct{size}_impl<V: super::forward_lanes::ForwardLanes, const LOW{keep}: bool>(", 1)
    wrappers = []
    for name, low, description in [
        (f"fdct{size}", "false", f"Full {size}-point graph retained as the scalar/reference implementation."),
        (f"fdct{size}_low{keep}", "true", f"Exact low{keep} outputs; entries{keep}..{size-1} are scratch and must not be consumed."),
    ]:
        if size == 32 and low == "true":
            wrappers.append('#[cfg(any(target_arch = "x86_64", all(target_arch = "wasm32", target_feature = "simd128")))]')
        wrappers.append(f"""/// {description}
#[inline(always)]
pub(crate) fn {name}<V: super::forward_lanes::ForwardLanes>(
    input: &[V; {size}],
    output: &mut [V; {size}],
    cospi: &[i32; 64],
    cos_bit: u32,
) {{
    fdct{size}_impl::<V, {low}>(input, output, cospi, cos_bit);
}}
""")
    return "\n".join(wrappers) + "\n" + full


def inverse32_half_schedule(full: str) -> str:
    """Run independent odd/even halves to completion before the final merge.

    Preserve every expression and stage clamp. Check the independence when
    generating so a future upstream graph cannot silently break the schedule.
    """
    head, body = full.split("    // stage 0;", 1)
    independent, merge = body.split("    // stage 9", 1)
    halves: list[list[str]] = [[], []]
    for line in independent.splitlines():
        if not line.strip() or line.strip().startswith("//"):
            continue
        match = re.fullmatch(r"    (?:step|output)\[(\d+)\] = (.*);", line)
        if match is None:
            raise ValueError(f"unrecognised inverse32 assignment: {line}")
        half = int(match[1]) // 16
        dependencies = re.findall(r"(?:step|output)\[(\d+)\]", match[2])
        if any(int(index) // 16 != half for index in dependencies):
            raise ValueError(f"inverse32 cross-half dependency: {line}")
        halves[half].append(line)
    head = head.replace("fn idct32<", "fn idct32_half_scheduled<")
    head = head.replace("#[inline(always)]", '#[cfg(target_arch = "x86_64")]\n#[inline(always)]')
    return (head + "    // Independent odd half, then even half.\n"
            + "\n".join(halves[1] + halves[0]) + "\n    // stage 9" + merge)


def main() -> None:
    libaom, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    parts = [
        "//! AV1 1-D DCT butterflies over SIMD lanes. GENERATED by",
        "//! `tools/codegen/gen_dsp_dct.py` from libaom (BSD-2-Clause, Copyright (c)",
        "//! 2016, Alliance for Open Media; see `THIRD_PARTY.md`): do not edit by hand.",
        "//!",
        "//! Each function applies one libaom 1-D DCT independently to every lane of",
        "//! its input vectors: `input[i]` holds element `i` of `V::LANES` separate",
        "//! transforms. The inverse transforms clamp every stage to 16 bits, the",
        "//! 8-bit stage range.",
        "",
        "#![allow(clippy::too_many_lines, clippy::inline_always)]",
        "",
    ]
    for source, c_name, rust_name, size in FUNCTIONS:
        text = (libaom / source).read_text()
        translated = translate(function_body(text, c_name), rust_name, size)
        if rust_name in ("fdct32", "fdct64"):
            translated = forward_low_half_specialization(translated, size)
        parts.append(translated)
        parts.append("")
        if rust_name == "idct32":
            parts.append(inverse32_half_schedule(translated))
            parts.append("")
    inverse = (libaom / "av1/common/av1_inv_txfm1d.c").read_text()
    for size, limits in [(32, (8, 16)), (64, (8, 16, 32))]:
        for limit in limits:
            parts.append(translate(function_body(inverse, f"av1_idct{size}"),
                                   f"idct{size}_low{limit}", size, limit))
            parts.append("")
    out_path.write_text("\n".join(parts))


if __name__ == "__main__":
    main()
