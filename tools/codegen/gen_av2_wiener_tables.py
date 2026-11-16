#!/usr/bin/env python3
"""Generate the AV2 Wiener restoration filters' classification and
pretrained tables (crates/nc-dsp/src/av2_dec/wiener_tables.rs) from dav2d's
src/wiener_tables.c and src/tables.c.

    tools/codegen/gen_av2_wiener_tables.py DAV2D_SRC [--write]

DAV2D_SRC is a dav2d checkout (commit 237229e; BSD-2-Clause, see
docs/licenses/dav2d), e.g. ~/nanocodec-data/work/av2-dec/dav2d-debug. The
tables: the class of each of the 4096 feature indices (`lut_to_class`),
the filter each class selects for the pretrained (PC-Wiener) filters and for
the frame's own filters by class count (`pretrained_sub_classify`,
`user_sub_classify`, by quantizer class), and the 64 pretrained filters per
quantizer class (`dav2d_pc_wiener_filters`). Without --write the generated
file is only compared with the checked-in one.
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'crates/nc-dsp/src/av2_dec/wiener_tables.rs'


def flat_array(source, name):
    """The integers of C array `name`'s initializer, flattened."""
    text = re.sub(r'//[^\n]*', '', source)
    text = re.sub(r'/\*.*?\*/', '', text, flags=re.S)
    start = text.index('{', re.search(r'\b' + name + r'\s*\[', text).end())
    depth, i = 0, start
    while True:
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                break
        i += 1
    return [int(v) for v in re.findall(r'-?\d+', text[start:i])]


def rust_array(values, per_line=16):
    lines = [', '.join(str(v) for v in values[i:i + per_line])
             for i in range(0, len(values), per_line)]
    return '[\n    ' + ',\n    '.join(lines) + ',\n]'


def nested(values, shape):
    """Rust nested array literal of `values` in `shape`."""
    if len(shape) == 1:
        return '[' + ', '.join(str(v) for v in values) + ']'
    step = len(values) // shape[0]
    inner = [nested(values[i * step:(i + 1) * step], shape[1:]) for i in range(shape[0])]
    return '[\n' + ',\n'.join(inner) + ',\n]'


def generate(src):
    wiener = (Path(src) / 'src/wiener_tables.c').read_text()
    tables = (Path(src) / 'src/tables.c').read_text()
    lut = flat_array(wiener, 'lut_to_class')
    pretrained = flat_array(wiener, 'pretrained_sub_classify')
    user = flat_array(wiener, 'user_sub_classify')
    filters = flat_array(tables, 'dav2d_pc_wiener_filters')
    assert len(lut) == 4096 and len(pretrained) == 4 * 256
    assert len(user) == 4 * 7 * 256 and len(filters) == 4 * 64 * 13
    out = ['//! The Wiener restoration filters\' classification and pretrained',
           '//! filters (dav2d `wiener_tables.c` and `tables.c`, commit 237229e;',
           '//! Copyright © 2018-2026 `VideoLAN` and dav2d authors, BSD-2-Clause, see',
           '//! `docs/licenses/dav2d`; from AVM v1.0.0\'s trained models). Generated',
           '//! by `tools/codegen/gen_av2_wiener_tables.py DAV2D_SRC --write`: do not',
           '//! edit.',
           '',
           '/// The class of each feature index (dav2d `lut_to_class`).',
           '#[rustfmt::skip]',
           'pub static FEATURE_CLASS: [u8; 4096] = ' + rust_array(lut) + ';',
           '',
           '/// The pretrained filter of each class, by quantizer class (dav2d',
           '/// `pretrained_sub_classify`).',
           '#[rustfmt::skip]',
           'pub static PRETRAINED_FILTER_OF_CLASS: [[u8; 256]; 4] = '
           + nested(pretrained, [4, 256]) + ';',
           '',
           '/// The frame filter of each class, by quantizer class and class count',
           '/// index minus one (dav2d `user_sub_classify`).',
           '#[rustfmt::skip]',
           'pub static FRAME_FILTER_OF_CLASS: [[[u8; 256]; 7]; 4] = '
           + nested(user, [4, 7, 256]) + ';',
           '',
           '/// The pretrained filters by quantizer class: twelve symmetric taps',
           '/// and the centre tap (dav2d `dav2d_pc_wiener_filters`).',
           '#[rustfmt::skip]',
           'pub static PRETRAINED_FILTERS: [[[i8; 13]; 64]; 4] = '
           + nested(filters, [4, 64, 13]) + ';',
           '']
    return '\n'.join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('src')
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    text = generate(args.src)
    if args.write:
        OUT.write_text(text)
        print(f'wrote {OUT}')
    elif OUT.exists() and OUT.read_text() == text:
        print('PASS: generated tables match')
    else:
        sys.exit('FAIL: generated tables differ (run with --write)')


if __name__ == '__main__':
    main()
