#!/usr/bin/env python3
"""Generate the general AV2 decoder's transform tables from dav2d's C source.

    tools/codegen/gen_av2_dec_tables.py DAV2D_SRC [--check]

DAV2D_SRC is a dav2d checkout at commit 237229e (e.g.
~/nanocodec-data/anchors/vp9-av2/src/dav2d-237229e). The script parses the
constant arrays named in TABLES out of its C files and writes them, flattened
the way the Rust kernels index them, to

- crates/nc-dsp/src/av2_dec/itx_kernels.rs (the 1-D inverse transform
  matrices of src/itx_1d.c), and
- crates/nc-dsp/src/av2_dec/stx_tables.rs (the secondary transform kernels
  and 8x8 coefficient mappings of src/stx_tables.c).

With --check the files are compared, not written (exit status 1 on a diff).
"""
import argparse
import re
import sys
from math import isqrt, prod
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / 'crates/nc-dsp/src/av2_dec'
DAV2D_COMMIT = '237229e'

# (C file, C name, Rust name, Rust element type, Rust shape, doc). The Rust
# shape's product must equal the C array's; it flattens trailing C axes
# where the kernels index them as one row-major matrix.
ITX_TABLES = [
    ('src/itx_1d.c', 'dct8_kernel', 'DCT8_ODD', 'i8', [4 * 4],
     'Odd half of the 8-point DCT: row `i` weights inputs 1, 3, 5, 7 for outputs `i` and `7 - i`.'),
    ('src/itx_1d.c', 'dct16_kernel', 'DCT16_ODD', 'i8', [8 * 8],
     'Odd half of the 16-point DCT (as [`DCT8_ODD`]).'),
    ('src/itx_1d.c', 'dct32_kernel', 'DCT32_ODD', 'i8', [16 * 16],
     'Odd half of the 32-point DCT (as [`DCT8_ODD`]).'),
    ('src/itx_1d.c', 'adst4_kernel', 'ADST4', 'i8', [4 * 4],
     '4-point ADST: row `i` weights every input for output `i`.'),
    ('src/itx_1d.c', 'adst8_kernel', 'ADST8', 'i8', [8 * 8],
     '8-point ADST (as [`ADST4`]); flipped by output reversal.'),
    ('src/itx_1d.c', 'adst16_kernel', 'ADST16', 'i8', [16 * 16],
     '16-point ADST (as [`ADST4`]).'),
    ('src/itx_1d.c', 'flipadst4_kernel', 'FLIPADST4', 'i8', [4 * 4],
     '4-point flipped ADST, its own matrix (AVM tunes it separately).'),
    ('src/itx_1d.c', 'flipadst16_kernel', 'FLIPADST16', 'i8', [16 * 16],
     '16-point flipped ADST, its own matrix (AVM tunes it separately).'),
    ('src/itx_1d.c', 'ddt8_kernel', 'DDT8', 'i8', [8 * 8],
     '8-point data-driven transform (inter DDT); flipped by output reversal.'),
    ('src/itx_1d.c', 'ddt16_kernel', 'DDT16', 'i8', [16 * 16],
     '16-point data-driven transform (as [`DDT8`]).'),
]

STX_TABLES = [
    ('src/stx_tables.c', 'dav2d_coeff8x8_mapping_idx', 'COEFF8X8_MAPPING_IDX', 'u8', [33],
     'The 8x8 output mapping of each kernel, by `set * 3 + type` (`dav2d_coeff8x8_mapping_idx`).'),
    ('src/stx_tables.c', 'dav2d_coeff8x8_mapping', 'COEFF8X8_MAPPING', 'u8', [3, 48],
     'Raster positions (`y * 8 + x`) in the top-left 8x8 of the 48 outputs of an 8x8 kernel\n'
     '/// (`dav2d_coeff8x8_mapping`).'),
    ('src/stx_tables.c', 'dav2d_stx_4x4_kernel', 'STX4_KERNELS', 'i8', [14, 3, 8 * 16],
     'The 4x4 kernels by set and type: 8 inputs by 16 outputs, row-major by input\n'
     '/// (`dav2d_stx_4x4_kernel`).'),
    ('src/stx_tables.c', 'dav2d_stx_8x8_kernel', 'STX8_KERNELS', 'i8', [11, 3, 32 * 48],
     'The 8x8 kernels by set and type: 32 inputs by 48 outputs, row-major by input\n'
     '/// (`dav2d_stx_8x8_kernel`).'),
]

OUTPUTS = [
    ('itx_kernels.rs', 'src/itx_1d.c',
     'The 1-D inverse transform matrices of the general AV2 decoder\'s [`super::itx`]',
     ITX_TABLES),
    ('stx_tables.rs', 'src/stx_tables.c',
     'The secondary transform kernels and 8x8 output mappings of [`super::stx`]',
     STX_TABLES),
]

# `[static] const <type> [ALIGN(]<name>[dims][, align)] = {`
DECL = re.compile(
    r'(?:static\s+)?const\s+u?int\d+_t\s+(?:ALIGN\(\s*)?(\w+)((?:\[[\d\s*]+\])+)'
    r'(?:\s*,\s*\d+\s*\))?\s*=\s*\{')
NUMBER = re.compile(r'-?\d+')
MAX_ROW = 16  # values per generated line


def strip_comments(text):
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.S)
    return re.sub(r'//[^\n]*', ' ', text)


def parse_arrays(path):
    """Every constant array initializer of `path`: name -> (dims, values)."""
    text = strip_comments(path.read_text())
    arrays = {}
    for match in DECL.finditer(text):
        # Each dimension is a product of literals (`[4 * 4]`).
        dims = [prod(int(f) for f in d.split('*'))
                for d in re.findall(r'\[([^\]]+)\]', match.group(2))]
        end = text.index('};', match.end())
        values = [int(v) for v in NUMBER.findall(text[match.end():end])]
        if len(values) != prod(dims):
            sys.exit(f'{path}: {match.group(1)} has {len(values)} values, dims {dims}')
        arrays[match.group(1)] = (dims, values)
    return arrays


def rust_type(element, shape):
    ty = element
    for dim in reversed(shape):
        ty = f'[{ty}; {dim}]'
    return ty


def rust_value(values, shape, indent):
    pad = '    ' * indent
    if len(shape) == 1:
        # A square matrix prints one matrix row per line.
        side = isqrt(len(values))
        row = side if side * side == len(values) and side <= MAX_ROW else MAX_ROW
        lines = [', '.join(f'{v:4}' for v in values[i:i + row]) + ','
                 for i in range(0, len(values), row)]
        return '[\n' + ''.join(f'{pad}    {line}\n' for line in lines) + f'{pad}]'
    step = prod(shape[1:])
    parts = [rust_value(values[i:i + step], shape[1:], indent + 1)
             for i in range(0, len(values), step)]
    return '[\n' + ''.join(f'{pad}    {p},\n' for p in parts) + f'{pad}]'


def render(source, file_name, c_file, what, tables):
    arrays = parse_arrays(source / c_file)
    out = [
        f'//! {what}.',
        '//!',
        f'//! GENERATED by `tools/codegen/gen_av2_dec_tables.py` from dav2d (commit',
        f'//! {DAV2D_COMMIT}) `{c_file}`; do not edit. Copyright © 2018-2026 `VideoLAN` and',
        '//! dav2d authors, BSD-2-Clause (`docs/licenses/dav2d`).',
        '',
    ]
    for _, c_name, rust_name, element, shape, doc in tables:
        if c_name not in arrays:
            sys.exit(f'{c_file}: no array {c_name}')
        dims, values = arrays[c_name]
        if prod(dims) != prod(shape):
            sys.exit(f'{c_name}: C dims {dims} do not flatten to {shape}')
        out += [
            f'/// {doc}',
            '#[rustfmt::skip]',
            f'pub static {rust_name}: {rust_type(element, shape)} = '
            f'{rust_value(values, shape, 0)};',
            '',
        ]
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('dav2d_src', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    stale = False
    for file_name, c_file, what, tables in OUTPUTS:
        text = render(args.dav2d_src, file_name, c_file, what, tables)
        target = OUT_DIR / file_name
        if args.check:
            if not target.exists() or target.read_text() != text:
                print(f'{target} is stale')
                stale = True
        else:
            target.write_text(text)
            print(f'wrote {target}')
    return 1 if stale else 0


if __name__ == '__main__':
    sys.exit(main())
