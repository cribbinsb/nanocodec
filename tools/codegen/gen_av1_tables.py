#!/usr/bin/env python3
"""Generate crates/nc-av1/src/tables/generated.rs from libaom's C sources.

AV1 needs large constant tables (default CDFs, quantiser lookups, scan
orders, coefficient context offsets). Typing them by hand invites errors, so
they are extracted mechanically from the pinned libaom tree (BSD-2-Clause,
Copyright (c) 2016, Alliance for Open Media; see THIRD_PARTY.md).

CDF tables keep libaom's *inverse* CDF convention: each entry is
32768 - cdf, the last real entry is 0, and libaom's trailing adaptation
counter is dropped (the runtime Cdf context owns its adaptation counter).
Short CDFs inside a wider declared slot are zero-padded.

Usage:
    gen_av1_tables.py <libaom source dir> <output .rs file>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

# Constants used in the declarations we extract (libaom enums.h/entropy.h).
CONSTANTS = {
    "CFL_JOINT_SIGNS": 8, "CFL_ALPHA_CONTEXTS": 6, "CFL_ALPHABET_SIZE": 16,
    "DIRECTIONAL_MODES": 8, "MAX_ANGLE_DELTA": 3, "KF_MODE_CONTEXTS": 5, "INTRA_MODES": 13, "CFL_ALLOWED_TYPES": 2, "UV_INTRA_MODES": 14,
    "PARTITION_CONTEXTS": 20, "EXT_PARTITION_TYPES": 10, "EXT_TX_SETS_INTRA": 3,
    "EXT_TX_SIZES": 4, "TX_TYPES": 16, "SKIP_CONTEXTS": 3, "TOKEN_CDF_Q_CTXS": 4,
    "TX_SIZES": 5, "TXB_SKIP_CONTEXTS": 13, "PLANE_TYPES": 2, "EOB_COEF_CONTEXTS": 9,
    "DC_SIGN_CONTEXTS": 3, "SIG_COEF_CONTEXTS_EOB": 4, "SIG_COEF_CONTEXTS": 42,
    "NUM_BASE_LEVELS": 2, "LEVEL_CONTEXTS": 21, "BR_CDF_SIZE": 4, "QINDEX_RANGE": 256,
    "BLOCK_SIZE_GROUPS": 4, "EXT_TX_SETS_INTER": 4, "NEWMV_MODE_CONTEXTS": 6,
    "GLOBALMV_MODE_CONTEXTS": 2, "REFMV_MODE_CONTEXTS": 6, "DRL_MODE_CONTEXTS": 3,
    "INTRA_INTER_CONTEXTS": 4, "REF_CONTEXTS": 3, "SINGLE_REFS": 7,
    "TXFM_PARTITION_CONTEXTS": 21, "MAX_TX_CATS": 4, "TX_SIZE_CONTEXTS": 3, "MAX_TX_DEPTH": 2,
    "FILTER_INTRA_MODES": 5, "BLOCK_SIZES_ALL": 22,
    "SWITCHABLE_FILTER_CONTEXTS": 16, "SWITCHABLE_FILTERS": 3, "INTER_MODE_CONTEXTS": 8,
    "INTER_COMPOUND_MODES": 8, "INTERINTRA_MODES": 4, "MOTION_MODES": 3,
    "MASKED_COMPOUND_TYPES": 2, "COMP_INTER_CONTEXTS": 5, "COMP_REF_TYPE_CONTEXTS": 5,
    "UNI_COMP_REF_CONTEXTS": 3, "UNIDIR_COMP_REFS": 4, "FWD_REFS": 4, "BWD_REFS": 3,
    "SKIP_MODE_CONTEXTS": 3, "COMP_INDEX_CONTEXTS": 6, "COMP_GROUP_IDX_CONTEXTS": 6,
    "MAX_WEDGE_TYPES": 16, "RESTORE_SWITCHABLE_TYPES": 3, "DELTA_Q_PROBS": 3,
    "DELTA_LF_PROBS": 3, "FRAME_LF_COUNT": 4, "SEG_TEMPORAL_PRED_CTXS": 3,
    "SPATIAL_PREDICTION_PROBS": 3, "MAX_SEGMENTS": 8,
    # libaom spells the palette block-size contexts PALATTE_BSIZE_CTXS.
    "PALATTE_BSIZE_CTXS": 7, "PALETTE_Y_MODE_CONTEXTS": 3, "PALETTE_UV_MODE_CONTEXTS": 2,
    "PALETTE_SIZES": 7, "PALETTE_COLOR_INDEX_CONTEXTS": 5, "PALETTE_COLORS": 8,
}

# (C name, source file, Rust name). CDF tables.
CDF_TABLES = [
    ("default_cfl_sign_cdf", "av1/common/entropymode.c", "CFL_SIGN_CDF"),
    ("default_cfl_alpha_cdf", "av1/common/entropymode.c", "CFL_ALPHA_CDF"),
    ("default_kf_y_mode_cdf", "av1/common/entropymode.c", "KF_Y_MODE_CDF"),
    ("default_angle_delta_cdf", "av1/common/entropymode.c", "ANGLE_DELTA_CDF"),
    ("default_uv_mode_cdf", "av1/common/entropymode.c", "UV_MODE_CDF"),
    ("default_partition_cdf", "av1/common/entropymode.c", "PARTITION_CDF"),
    ("default_intra_ext_tx_cdf", "av1/common/entropymode.c", "INTRA_EXT_TX_CDF"),
    ("default_skip_txfm_cdfs", "av1/common/entropymode.c", "SKIP_CDF"),
    ("default_if_y_mode_cdf", "av1/common/entropymode.c", "Y_MODE_CDF"),
    ("default_inter_ext_tx_cdf", "av1/common/entropymode.c", "INTER_EXT_TX_CDF"),
    ("default_newmv_cdf", "av1/common/entropymode.c", "NEWMV_CDF"),
    ("default_zeromv_cdf", "av1/common/entropymode.c", "ZEROMV_CDF"),
    ("default_refmv_cdf", "av1/common/entropymode.c", "REFMV_CDF"),
    ("default_drl_cdf", "av1/common/entropymode.c", "DRL_CDF"),
    ("default_intra_inter_cdf", "av1/common/entropymode.c", "INTRA_INTER_CDF"),
    ("default_single_ref_cdf", "av1/common/entropymode.c", "SINGLE_REF_CDF"),
    ("default_txfm_partition_cdf", "av1/common/entropymode.c", "TXFM_PARTITION_CDF"),
    ("default_tx_size_cdf", "av1/common/entropymode.c", "TX_SIZE_CDF"),
    ("default_filter_intra_cdfs", "av1/common/entropymode.c", "FILTER_INTRA_CDF"),
    ("default_filter_intra_mode_cdf", "av1/common/entropymode.c", "FILTER_INTRA_MODE_CDF"),
    ("av1_default_txb_skip_cdfs", "av1/common/token_cdfs.h", "TXB_SKIP_CDF"),
    ("av1_default_eob_extra_cdfs", "av1/common/token_cdfs.h", "EOB_EXTRA_CDF"),
    ("av1_default_dc_sign_cdfs", "av1/common/token_cdfs.h", "DC_SIGN_CDF"),
    ("av1_default_eob_multi16_cdfs", "av1/common/token_cdfs.h", "EOB_PT_16_CDF"),
    ("av1_default_eob_multi32_cdfs", "av1/common/token_cdfs.h", "EOB_PT_32_CDF"),
    ("av1_default_eob_multi64_cdfs", "av1/common/token_cdfs.h", "EOB_PT_64_CDF"),
    ("av1_default_eob_multi128_cdfs", "av1/common/token_cdfs.h", "EOB_PT_128_CDF"),
    ("av1_default_eob_multi256_cdfs", "av1/common/token_cdfs.h", "EOB_PT_256_CDF"),
    ("av1_default_eob_multi512_cdfs", "av1/common/token_cdfs.h", "EOB_PT_512_CDF"),
    ("av1_default_eob_multi1024_cdfs", "av1/common/token_cdfs.h", "EOB_PT_1024_CDF"),
    ("av1_default_coeff_base_eob_multi_cdfs", "av1/common/token_cdfs.h",
     "COEFF_BASE_EOB_CDF"),
    ("av1_default_coeff_base_multi_cdfs", "av1/common/token_cdfs.h", "COEFF_BASE_CDF"),
    ("av1_default_coeff_lps_multi_cdfs", "av1/common/token_cdfs.h", "COEFF_BR_CDF"),
    # Inter, segmentation, block delta and restoration syntax (the decoder).
    ("default_switchable_interp_cdf", "av1/common/entropymode.c", "SWITCHABLE_INTERP_CDF"),
    ("default_inter_compound_mode_cdf", "av1/common/entropymode.c",
     "INTER_COMPOUND_MODE_CDF"),
    ("default_interintra_cdf", "av1/common/entropymode.c", "INTERINTRA_CDF"),
    ("default_interintra_mode_cdf", "av1/common/entropymode.c", "INTERINTRA_MODE_CDF"),
    ("default_wedge_interintra_cdf", "av1/common/entropymode.c", "WEDGE_INTERINTRA_CDF"),
    ("default_compound_type_cdf", "av1/common/entropymode.c", "COMPOUND_TYPE_CDF"),
    ("default_wedge_idx_cdf", "av1/common/entropymode.c", "WEDGE_IDX_CDF"),
    ("default_motion_mode_cdf", "av1/common/entropymode.c", "MOTION_MODE_CDF"),
    ("default_obmc_cdf", "av1/common/entropymode.c", "OBMC_CDF"),
    ("default_comp_inter_cdf", "av1/common/entropymode.c", "COMP_INTER_CDF"),
    ("default_comp_ref_type_cdf", "av1/common/entropymode.c", "COMP_REF_TYPE_CDF"),
    ("default_uni_comp_ref_cdf", "av1/common/entropymode.c", "UNI_COMP_REF_CDF"),
    ("default_comp_ref_cdf", "av1/common/entropymode.c", "COMP_REF_CDF"),
    ("default_comp_bwdref_cdf", "av1/common/entropymode.c", "COMP_BWDREF_CDF"),
    ("default_skip_mode_cdfs", "av1/common/entropymode.c", "SKIP_MODE_CDF"),
    ("default_compound_idx_cdfs", "av1/common/entropymode.c", "COMPOUND_IDX_CDF"),
    ("default_comp_group_idx_cdfs", "av1/common/entropymode.c", "COMP_GROUP_IDX_CDF"),
    ("default_switchable_restore_cdf", "av1/common/entropymode.c", "SWITCHABLE_RESTORE_CDF"),
    ("default_wiener_restore_cdf", "av1/common/entropymode.c", "WIENER_RESTORE_CDF"),
    ("default_sgrproj_restore_cdf", "av1/common/entropymode.c", "SGRPROJ_RESTORE_CDF"),
    ("default_palette_y_mode_cdf", "av1/common/entropymode.c", "PALETTE_Y_MODE_CDF"),
    ("default_palette_uv_mode_cdf", "av1/common/entropymode.c", "PALETTE_UV_MODE_CDF"),
    ("default_palette_y_size_cdf", "av1/common/entropymode.c", "PALETTE_Y_SIZE_CDF"),
    ("default_palette_uv_size_cdf", "av1/common/entropymode.c", "PALETTE_UV_SIZE_CDF"),
    ("default_palette_y_color_index_cdf", "av1/common/entropymode.c",
     "PALETTE_Y_COLOR_INDEX_CDF"),
    ("default_palette_uv_color_index_cdf", "av1/common/entropymode.c",
     "PALETTE_UV_COLOR_INDEX_CDF"),
    ("default_intrabc_cdf", "av1/common/entropymode.c", "INTRABC_CDF"),
    ("default_delta_q_cdf", "av1/common/entropymode.c", "DELTA_Q_CDF"),
    ("default_delta_lf_multi_cdf", "av1/common/entropymode.c", "DELTA_LF_MULTI_CDF"),
    ("default_delta_lf_cdf", "av1/common/entropymode.c", "DELTA_LF_CDF"),
    ("default_segment_pred_cdf", "av1/common/entropymode.c", "SEGMENT_PRED_CDF"),
    ("default_spatial_pred_seg_tree_cdf", "av1/common/entropymode.c",
     "SPATIAL_PRED_SEG_TREE_CDF"),
]

# (C name, source file, Rust name, Rust element type). Plain integer tables.
INT_TABLES = [
    ("intra_mode_context", "av1/common/common_data.h", "INTRA_MODE_CONTEXT", "u8"),
    ("dc_qlookup_QTX", "av1/common/quant_common.c", "DC_QLOOKUP", "i16"),
    ("ac_qlookup_QTX", "av1/common/quant_common.c", "AC_QLOOKUP", "i16"),
    ("default_scan_4x4", "av1/common/scan.c", "DEFAULT_SCAN_4X4", "u16"),
    ("default_scan_8x8", "av1/common/scan.c", "DEFAULT_SCAN_8X8", "u16"),
    ("default_scan_16x16", "av1/common/scan.c", "DEFAULT_SCAN_16X16", "u16"),
    ("default_scan_32x32", "av1/common/scan.c", "DEFAULT_SCAN_32X32", "u16"),
    ("av1_nz_map_ctx_offset_4x4", "av1/common/txb_common.c", "NZ_MAP_CTX_OFFSET_4X4", "u8"),
    ("av1_nz_map_ctx_offset_8x8", "av1/common/txb_common.c", "NZ_MAP_CTX_OFFSET_8X8", "u8"),
    ("av1_nz_map_ctx_offset_16x16", "av1/common/txb_common.c", "NZ_MAP_CTX_OFFSET_16X16",
     "u8"),
    ("av1_nz_map_ctx_offset_32x32", "av1/common/txb_common.c", "NZ_MAP_CTX_OFFSET_32X32",
     "u8"),
    # Rectangular sizes (the decoder reads every transform size).
    ("default_scan_4x8", "av1/common/scan.c", "DEFAULT_SCAN_4X8", "u16"),
    ("default_scan_8x4", "av1/common/scan.c", "DEFAULT_SCAN_8X4", "u16"),
    ("default_scan_4x16", "av1/common/scan.c", "DEFAULT_SCAN_4X16", "u16"),
    ("default_scan_16x4", "av1/common/scan.c", "DEFAULT_SCAN_16X4", "u16"),
    ("default_scan_8x16", "av1/common/scan.c", "DEFAULT_SCAN_8X16", "u16"),
    ("default_scan_16x8", "av1/common/scan.c", "DEFAULT_SCAN_16X8", "u16"),
    ("default_scan_8x32", "av1/common/scan.c", "DEFAULT_SCAN_8X32", "u16"),
    ("default_scan_32x8", "av1/common/scan.c", "DEFAULT_SCAN_32X8", "u16"),
    ("default_scan_16x32", "av1/common/scan.c", "DEFAULT_SCAN_16X32", "u16"),
    ("default_scan_32x16", "av1/common/scan.c", "DEFAULT_SCAN_32X16", "u16"),
    ("av1_nz_map_ctx_offset_4x8", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_4X8", "u8"),
    ("av1_nz_map_ctx_offset_8x16", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_8X16", "u8"),
    ("av1_nz_map_ctx_offset_16x32", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_16X32", "u8"),
    ("av1_nz_map_ctx_offset_32x16", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_32X16", "u8"),
    ("av1_nz_map_ctx_offset_32x64", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_32X64", "u8"),
    ("av1_nz_map_ctx_offset_64x32", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_64X32", "u8"),
    ("av1_nz_map_ctx_offset_4x16", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_4X16", "u8"),
    ("av1_nz_map_ctx_offset_16x4", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_16X4", "u8"),
    ("av1_nz_map_ctx_offset_8x32", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_8X32", "u8"),
    ("av1_nz_map_ctx_offset_32x8", "av1/common/txb_common.c",
     "NZ_MAP_CTX_OFFSET_32X8", "u8"),
]

CDF_PROB_TOP = 32768


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def find_initializer(text: str, name: str) -> tuple[list[int], str]:
    """Return (declared dims, initializer body) of the C array `name`."""
    match = re.search(rf"\b{name}\s*((?:\[[^\]]*\]\s*)+)\)?\s*=\s*\{{", text)
    if not match:
        raise KeyError(f"array {name} not found")
    dims = [evaluate(d) for d in re.findall(r"\[([^\]]*)\]", match.group(1))]
    start = match.end() - 1
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return dims, text[start:i + 1]
    raise ValueError(f"unterminated initializer for {name}")


def evaluate(expr: str) -> int:
    expr = " ".join(expr.split())  # declarations wrap across lines
    expr = re.sub(r"CDF_SIZE\(([^)]*)\)", r"(\1 + 1)", expr)
    for key, value in CONSTANTS.items():
        expr = re.sub(rf"\b{key}\b", str(value), expr)
    return int(eval(expr, {"__builtins__": {}}))  # noqa: S307 - trusted pinned source


def expand_cdf_macros(body: str) -> str:
    """Replace AOM_CDFn(a, b, ...) with the inverse-CDF values (no counter)."""
    def repl(m: re.Match) -> str:
        args = [evaluate(a) for a in split_args(m.group(2))]
        icdf = [CDF_PROB_TOP - a for a in args] + [0]
        return ", ".join(map(str, icdf))
    pattern = re.compile(r"AOM_CDF(\d+)\(([^()]*(?:\([^()]*\)[^()]*)*)\)")
    return pattern.sub(repl, body)


def split_args(args: str) -> list[str]:
    return [a.strip() for a in args.split(",") if a.strip()]


def parse_braces(body: str):
    """Parse a C brace initializer into nested Python lists of ints."""
    tokens = re.findall(r"\{|\}|[^{},\s][^{},]*", body)
    stack = [[]]
    for tok in tokens:
        tok = tok.strip()
        if tok == "{":
            stack.append([])
        elif tok == "}":
            done = stack.pop()
            stack[-1].append(done)
        elif tok:
            stack[-1].append(evaluate(tok))
    assert len(stack) == 1 and len(stack[0]) == 1
    return stack[0][0]


def shape_to(node, dims: list[int]):
    """Fit `node` to `dims`, zero-filling missing trailing elements/levels."""
    if not dims:
        if isinstance(node, list):
            raise ValueError("initializer nested deeper than its declaration")
        return node
    if not isinstance(node, list):
        node = [node]
    # A `{ 0 }` placeholder for a whole sub-array: flatten scalars into zeros.
    if len(dims) > 1 and node and not isinstance(node[0], list):
        if any(v != 0 for v in node):
            raise ValueError("scalar initializer for a nested level must be zero")
        node = []
    if len(node) > dims[0]:
        raise ValueError(f"too many elements: {len(node)} > {dims[0]}")
    zero = zeros(dims[1:])
    return [shape_to(n, dims[1:]) for n in node] + [zero] * (dims[0] - len(node))


def zeros(dims: list[int]):
    return 0 if not dims else [zeros(dims[1:]) for _ in range(dims[0])]


def rust_type(base: str, dims: list[int]) -> str:
    t = base
    for d in reversed(dims):
        t = f"[{t}; {d}]"
    return t


def rust_value(node) -> str:
    if isinstance(node, list):
        return "[" + ", ".join(rust_value(n) for n in node) + "]"
    return str(node)


def main() -> int:
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    cache: dict[str, str] = {}

    def source(rel: str) -> str:
        if rel not in cache:
            cache[rel] = strip_comments((src / rel).read_text())
        return cache[rel]

    lines = [
        "//! Constant tables extracted from libaom v3.15.3 (BSD-2-Clause, Copyright (c)",
        "//! 2016, Alliance for Open Media). GENERATED by `tools/codegen/gen_av1_tables.py`",
        "//! -- do not edit by hand.",
        "//!",
        "//! CDFs use libaom's inverse convention (32768 - cdf, terminated by 0) and omit",
        "//! the adaptation counter; short CDFs are zero-padded to their slot width.",
        "",
        "// Complete libaom tables: not every transform size is used yet.",
        "#![allow(dead_code, clippy::unreadable_literal, clippy::large_const_arrays)]",
        "",
    ]
    for cname, rel, rname in CDF_TABLES:
        dims, body = find_initializer(source(rel), cname)
        dims[-1] -= 1  # drop the adaptation counter slot
        data = shape_to(parse_braces(expand_cdf_macros(body)), dims)
        lines.append(f"/// libaom `{cname}`.")
        lines.append(f"pub const {rname}: {rust_type('u16', dims)} = {rust_value(data)};")
        lines.append("")
    for cname, rel, rname, ty in INT_TABLES:
        dims, body = find_initializer(source(rel), cname)
        data = shape_to(parse_braces(body), dims)
        lines.append(f"/// libaom `{cname}`.")
        lines.append(f"pub const {rname}: {rust_type(ty, dims)} = {rust_value(data)};")
        lines.append("")
    out.write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
