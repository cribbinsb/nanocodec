#!/usr/bin/env python3
"""Generate the general AV2 decoder's probability models from AVM v1.0.0's
default tables (crates/nc-av2/src/general/cdfs.rs and general/cdfs/defaults.rs).

    tools/codegen/gen_av2_general_cdfs.py --source AVM_SRC --build AVM_BUILD [--write]

AVM_SRC is an AVM checkout at the pin below (e.g.
~/nanocodec-data/anchors/vp9-av2/src/avm-v1.0.0), AVM_BUILD its build
directory (for avm_config.h). Each model family in SPECS names its field, its
array axes, its symbol count and the AVM expression of one CDF (axes a, b, c;
q is the coefficient quantizer category of quantized families). A small C
program compiled against the pinned headers prints every default with its
adaptation rates, checks the axes with static assertions, and hashes 65 AVM
`update_cdf` steps per model; the generated test replays them through our
`Cdf`. Without --write the files are only checked.

Tool agents add the families their syntax reads to SPECS (docs/handoff/av2.md
"The general block layer").
"""
import argparse
import itertools
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
OUT = ROOT / 'crates/nc-av2/src/general'

# (field, axes, symbols, AVM expression, flags, what it codes). Flags:
# 'q' the family has a table per coefficient quantizer category; 'r' its
# AVM rows are wider than its alphabet (a ragged table).
SPECS = [
    ('part_split', [2, 64], 2, 'default_do_split_cdf[a][b]', '',
     'whether a block splits (plane tree, context)'),
    ('part_square', [8], 2, 'default_do_square_split_cdf[0][a]', '',
     'whether a 128x128 or 256x256 block splits into four squares'),
    ('part_dir', [2, 64], 2, 'default_rect_type_cdf[a][b]', '',
     'the direction of a binary split'),
    ('part_ext', [2, 64], 2, 'default_do_ext_partition_cdf[a][0][b]', '',
     'whether a split is three-way'),
    ('part_4way', [2, 64], 2, 'default_do_uneven_4way_partition_cdf[a][0][b]', '',
     'whether a three-way split is the uneven four-way one instead'),
    ('region_type', [4], 2, 'default_region_type_cdf[a]', '',
     'whether an inter-frame region is intra only (separate trees)'),
    ('skip_mode', [3], 2, 'default_skip_mode_cdfs[a]', '',
     'whether an inter block is in skip mode'),
    ('is_inter', [4], 2, 'default_intra_inter_cdf[a]', '',
     'whether a block of an inter frame is inter'),
    ('intrabc', [3], 2, 'default_intrabc_cdf[a]', '',
     'whether an intra block uses intra block copy'),
    ('skip_txfm', [6], 2, 'default_skip_txfm_cdfs[a]', '',
     'whether a block has no residual (contexts 3..5 in skip mode)'),
    ('gdf', [], 2, 'default_gdf_cdf', '',
     'whether a guided-detail-filter block is filtered'),
    ('cdef_index0', [4], 2, 'default_cdef_strength_index0_cdf[a]', '',
     'whether a CDEF block uses strength set 0'),
    ('cdef_index3', [], 2, 'default_cdef_cdf[0]', 'r',
     'the CDEF strength set beyond 0 of a frame with three sets'),
    ('cdef_index4', [], 3, 'default_cdef_cdf[1]', 'r', 'the same with four sets'),
    ('cdef_index5', [], 4, 'default_cdef_cdf[2]', 'r', 'the same with five sets'),
    ('cdef_index6', [], 5, 'default_cdef_cdf[3]', 'r', 'the same with six sets'),
    ('cdef_index7', [], 6, 'default_cdef_cdf[4]', 'r', 'the same with seven sets'),
    ('cdef_index8', [], 7, 'default_cdef_cdf[5]', 'r', 'the same with eight sets'),
    ('ccso', [3, 4], 2, 'default_ccso_cdf[a][b]', '',
     'whether a CCSO block of a plane is filtered'),
    ('delta_q', [], 8, 'default_delta_q_cdf', '',
     'a superblock quantizer delta (the last symbol escapes)'),
]


def rust_type(base, dims):
    for size in reversed(dims):
        base = f'[{base}; {size}]'
    return base


def probe(source, build):
    """Defaults per category and family, and the update-trace hash per category."""
    c = ['#include <stdio.h>', '#include <stdint.h>', '#include <string.h>',
         '#include "av2/common/av2_common_int.h"',
         '#include "av2/common/entropy_inits_coeffs.h"',
         '#include "av2/common/entropy_inits_modes.h"',
         '#include "av2/common/entropy_inits_mv.h"',
         'int main(void) { for(int q=0;q<4;q++) { uint64_t hash=14695981039346656037ULL;']
    for _, dims, n, expression, flags, _ in SPECS:
        for i, size in enumerate(dims):
            c.append(f'for(int {chr(97+i)}=0;{chr(97+i)}<{size};{chr(97+i)}++) {{')
        relation = '>=' if 'r' in flags else '=='
        c.append(f'{{ _Static_assert(sizeof({expression}) {relation} sizeof(avm_cdf_prob) * ({n} + 4), "axes");')
        c.append(f'avm_cdf_prob model[{n}+4]; memcpy(model,{expression},sizeof(model));')
        c.append(f'for(int j=0;j<{n}+4;j++)printf("%u ",model[j]); puts("");')
        c.append(f'for(int step=0;step<65;step++) {{ update_cdf(model,(step*7+step/3)%{n},{n});'
                 f' for(int j=0;j<{n};j++) hash=(hash^model[j])*1099511628211ULL; }} }}')
        c.extend('}' for _ in dims)
    c.append('printf("HASH %llu\\n",(unsigned long long)hash); } return 0; }')
    with tempfile.TemporaryDirectory(prefix='nc-av2-general-cdf-') as directory:
        path = Path(directory) / 'probe.c'
        path.write_text('\n'.join(c))
        subprocess.run(['cc', '-O1', '-I' + str(source), '-I' + str(build), str(path),
                        '-o', str(path.with_suffix(''))], check=True)
        output = iter(subprocess.check_output([str(path.with_suffix(''))], text=True).splitlines())
    defaults = [[] for _ in SPECS]
    hashes = []
    for _ in range(4):
        for index, (_, dims, n, _, _, _) in enumerate(SPECS):
            rows = []
            for _ in itertools.product(*(range(size) for size in dims)):
                row = list(map(int, next(output).split()))
                assert len(row) == n + 4 and row[n] == 0
                rows.append(row)
            defaults[index].append(rows)
        hashes.append(int(next(output).split()[1]))
    assert next(output, None) is None, 'extra output'
    return defaults, hashes


def generate(defaults, hashes):
    tables = ['// Generated by tools/codegen/gen_av2_general_cdfs.py; do not edit.',
              '// Original AVM copyright Alliance for Open Media.',
              '// BSD-3-Clause-Clear + separate AOM patent terms; see THIRD_PARTY.md.',
              'use super::Initial;']
    for (name, dims, n, _, flags, _), categories in zip(SPECS, defaults):
        quantized = 'q' in flags
        if not quantized:
            assert all(category == categories[0] for category in categories)
        rows = iter(sum(categories, []) if quantized else categories[0])

        def array(shape):
            if not shape:
                row = next(rows)
                return f'Initial {{ inverse: {row[:n]}, rates: {row[n + 1:]} }}'
            return '[' + ',\n'.join(array(shape[1:]) for _ in range(shape[0])) + ']'
        shape = ([4] if quantized else []) + dims
        tables.append(f'pub(super) const {name.upper()}: {rust_type(f"Initial<{n}>", shape)} = {array(shape)};')
    tables.append('#[cfg(test)] pub(super) const TRACE_HASHES: [u64; 4] = ['
                  + ', '.join(f'{h:_}' for h in hashes) + '];')

    def build(expression, depth):
        if depth == 0:
            return expression + '.build()'
        if depth == 1:
            return expression + '.map(Initial::build)'
        return expression + '.map(|value| ' + build('value', depth - 1) + ')'

    def visit(root, action):
        lines = []
        for name, dims, *_ in SPECS:
            expression = f'{root}.{name}'
            if not dims:
                lines.append(action(expression if root == 'self' else '&mut ' + expression))
                continue
            expression = '&mut ' + expression
            for depth in range(len(dims)):
                lines.append(f'for v{depth} in {expression} {{')
                expression = f'v{depth}'
            lines.append(action(expression))
            lines.extend('}' for _ in dims)
        return lines

    model = ['//! The general AV2 decoder\'s probability models: every AVM v1.0.0 default',
             '//! its syntax reads so far, with AVM\'s adaptation rates (generated by',
             '//! `tools/codegen/gen_av2_general_cdfs.py`, which tool agents extend).',
             '//!', '//! Kept apart from [`crate::cdf_models::TileCdfs`], the encoder\'s subset,',
             '//! so the encoder\'s hot model copies stay small; both use the shared',
             '//! [`Cdf`] and entropy reader.',
             'use crate::entropy::Cdf;', '#[path = "cdfs/defaults.rs"] mod defaults;',
             '#[derive(Clone, Copy)] struct Initial<const N: usize> { inverse: [u16; N], rates: [u8; 3] }',
             'impl<const N: usize> Initial<N> { fn build(self) -> Cdf<N> {',
             'Cdf::new(self.inverse, self.rates).expect("independently verified AVM defaults") } }',
             '/// The coefficient CDF quantizer category of a frame base qindex (AVM `get_q_ctx`).',
             '#[allow(dead_code, reason = "the coefficient families use it")]',
             'fn category(qindex: u16) -> usize { match qindex { 0..=90 => 0, 91..=140 => 1, 141..=190 => 2, _ => 3 } }',
             '/// One tile\'s models.',
             '#[derive(Debug, Clone, PartialEq, Eq)] pub struct Cdfs {']
    for name, dims, n, expression, _, what in SPECS:
        model += [f'/// {what[0].upper() + what[1:]} (AVM `{expression}`).',
                  f'pub {name}: {rust_type(f"Cdf<{n}>", dims)},']
    model += ['}', 'impl Cdfs {', '/// The defaults of a frame at base quantizer `qindex`.',
              '#[must_use] pub fn new(qindex: u16) -> Self {',
              '#[allow(unused_variables, reason = "the coefficient families use it")] let q = category(qindex);',
              'Self {']
    for name, dims, _, _, flags, _ in SPECS:
        model.append(f'{name}: {build(f"defaults::{name.upper()}" + ("[q]" if "q" in flags else ""), len(dims))},')
    model += ['} }', '/// Move every model back along its adaptation-rate schedule (AVM',
              '/// `av2_reset_cdf_symbol_counters`) before later frames inherit them.',
              'pub fn restart_adaptation(&mut self) {']
    model += visit('self', lambda e: f'{e}.restart_adaptation();')
    model += ['} }', '#[cfg(test)] mod tests { use super::*; use crate::entropy::Writer;',
              'fn trace<const N: usize>(cdf: &mut Cdf<N>, hash: &mut u64) { let mut writer = Writer::new();',
              'for step in 0..65 { writer.symbol((step * 7 + step / 3) % N, cdf).unwrap();',
              'for &value in cdf.inverse() { *hash = (*hash ^ u64::from(value)).wrapping_mul(1_099_511_628_211); } } }',
              '/// Our models update as AVM\'s `update_cdf` does, from the same defaults.',
              '#[test] fn original_c_update_traces() {',
              'for (q, expected) in [0, 91, 141, 191].into_iter().zip(defaults::TRACE_HASHES) {',
              'let mut models = Cdfs::new(q); let mut hash = 14_695_981_039_346_656_037;']
    model += visit('models', lambda e: f'trace({e}, &mut hash);')
    model += ['assert_eq!(hash, expected); } } }']
    return {'cdfs/defaults.rs': '\n'.join(tables) + '\n', 'cdfs.rs': '\n'.join(model) + '\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    head = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == PIN, f'AVM at {head}, expected {PIN}'
    defaults, hashes = probe(args.source, args.build)
    for name, text in generate(defaults, hashes).items():
        with tempfile.TemporaryDirectory(prefix='nc-av2-general-cdf-rust-') as directory:
            path = Path(directory) / Path(name).name
            path.write_text(text)
            subprocess.run([str(Path.home() / '.cargo/bin/rustfmt'), '--edition', '2024',
                            '--config', 'skip_children=true', str(path)], check=True)
            text = path.read_text()
        target = OUT / name
        if args.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        else:
            assert target.read_text() == text, f'{name} differs; regenerate with --write'
    print(f'PASS: {len(SPECS)} model families; update-trace hashes {hashes}')


if __name__ == '__main__':
    main()
