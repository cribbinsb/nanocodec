#!/usr/bin/env python3
"""Extract VP9 default probabilities, scans and lookup tables from pinned libvpx.

Usage: gen_vp9_tables.py /path/to/libvpx-v1.16.0
No compiler, full reference build, network or generated object files required.
"""
import ast
import math
from pathlib import Path
import re
import subprocess
import sys

COMMIT = "1024874c5919305883187e2953de8fcb4c3d7fa6"


def array(source, name):
    match = re.search(r"\b" + name + r"\s*\[[^=]+?=\s*(\{.*?\});", source, re.S)
    if match is None:
        raise ValueError(f"missing table {name}")
    value = re.sub(r"/\*.*?\*/|//[^\n]*", "", match[1], flags=re.S)
    return ast.literal_eval(value.replace("{", "[").replace("}", "]"))


def declaration(name, ty, value):
    return f"pub(crate) const {name}: {ty} = {value!r};\n"


def main():
    source = Path(sys.argv[1])
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise ValueError(f"expected {COMMIT}, got {commit}")
    entropy = (source / "vp9/common/vp9_entropy.c").read_text()
    modes = (source / "vp9/common/vp9_entropymode.c").read_text()
    scan = (source / "vp9/common/vp9_scan.c").read_text()
    quant = (source / "vp9/common/vp9_quant_common.c").read_text()
    mvref = (source / "vp9/common/vp9_mvref_common.h").read_text()
    text = '''//! Generated VP9 default probabilities, scans and lookup tables from libvpx
//! v1.16.0 (BSD-3-Clause). Copyright (c) 2010 The `WebM` Project authors.
//! See `THIRD_PARTY.md`. Regenerate with `tools/codegen/gen_vp9_tables.py`;
//! do not edit manually.

#![allow(clippy::unreadable_literal, clippy::excessive_precision)]
'''
    sizes = ("4x4", "8x8", "16x16", "32x32")
    coefficients = []
    for size in sizes:
        model = array(entropy, f"default_coef_probs_{size}")
        for plane in model:
            for reference in plane:
                reference[0] += [[0, 0, 0]] * 3  # band zero has three contexts
        coefficients.append(model)
    text += "/// Node probabilities of one band/context grid: `[band][context][node]`.\n"
    text += "pub(crate) type BandProbs = [[[u8; 3]; 6]; 6];\n"
    text += declaration("COEFFICIENTS", "[[[BandProbs; 2]; 2]; 4]", coefficients)
    for size, count in zip(sizes, (16, 64, 256, 1024)):
        kinds = ("default", "col", "row") if size != "32x32" else ("default",)
        for kind in kinds:
            name = f"{kind}_scan_{size}"
            text += declaration(name.upper(), f"[u16; {count}]", array(scan, name))
            text += declaration(name.upper() + "_NEIGHBORS", f"[u16; {2 * (count + 1)}]",
                                array(scan, name + "_neighbors"))
    text += declaration("BANDS_4X4", "[u8; 16]", array(entropy, "vp9_coefband_trans_4x4"))
    text += declaration("BANDS_8X8_PLUS", "[u8; 1024]", array(entropy, "vp9_coefband_trans_8x8plus"))
    text += declaration("PARETO", "[[u8; 8]; 255]", array(entropy, "vp9_pareto8_full"))
    text += declaration("CATEGORY6", "[u8; 14]", array(entropy, "vp9_cat6_prob"))
    text += declaration("KF_PARTITION", "[[u8; 3]; 16]", array(modes, "vp9_kf_partition_probs"))
    text += declaration("PARTITION", "[[u8; 3]; 16]", array(modes, "default_partition_probs"))
    text += declaration("KF_Y_MODE", "[[[u8; 9]; 10]; 10]", array(modes, "vp9_kf_y_mode_prob"))
    text += declaration("KF_UV_MODE", "[[u8; 9]; 10]", array(modes, "vp9_kf_uv_mode_prob"))
    text += declaration("Y_MODE", "[[u8; 9]; 4]", array(modes, "default_if_y_probs"))
    text += declaration("UV_MODE", "[[u8; 9]; 10]", array(modes, "default_if_uv_probs"))
    text += declaration("INTER_MODE", "[[u8; 3]; 7]", array(modes, "default_inter_mode_probs"))
    text += declaration("SKIP", "[u8; 3]", array(modes, "default_skip_probs"))
    text += declaration("INTRA_INTER", "[u8; 4]", array(modes, "default_intra_inter_p"))
    text += declaration("SINGLE_REF", "[[u8; 2]; 5]", array(modes, "default_single_ref_p"))
    text += declaration("MV_REF_BLOCKS", "[[[i8; 2]; 8]; 13]", array(mvref, "mv_ref_blocks"))
    text += "/// Bits of a symbol of probability `p / 256` (`BIT_COST[p]`, p = 1..=255).\n"
    costs = [0.0] + [-math.log2(p / 256) for p in range(1, 256)]
    text += "pub(crate) const BIT_COST: [f32; 256] = [" + ", ".join(f"{c:.7}" for c in costs) + "];\n"
    text += declaration("DC_QUANT", "[i32; 256]", array(quant, "dc_qlookup"))
    text += declaration("AC_QUANT", "[i32; 256]", array(quant, "ac_qlookup"))
    target = Path(__file__).resolve().parents[2] / "crates/nc-vp9/src/tables.rs"
    target.write_text(text)
    subprocess.run([str(Path.home() / ".cargo/bin/rustfmt"), str(target)], check=True)


if __name__ == "__main__":
    main()
