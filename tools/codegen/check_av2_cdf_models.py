#!/usr/bin/env python3
"""Extract subset CDF identities/rates and check original AVM update_cdf traces.

--write regenerates new model files; default checks exact reproducibility.
Only original pinned headers are compiled, in a temporary directory. No YUV.
"""
import argparse
import ast
import itertools
import re
from pathlib import Path
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'


def specs():
    tree = ast.parse((ROOT / 'crates/nc-av2/examples/gen_coeff_tables.py').read_text())

    def assigned(name):
        return next(ast.literal_eval(n.value) for n in tree.body
                    if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    old = assigned('specs')
    unquantized = assigned('UNQUANTIZED')
    result = []
    for name, dims, expr in old:
        if name == 'SIGN':
            result.append(('sign', [2, 3], 2, 'av2_default_dc_sign_cdfs[q][a][0][b]', True))
        elif name in unquantized:
            result.append((name.lower(), dims[:-1], dims[-1], expr, False))
        else:
            # The first fixed-table axis is the quantizer category, selected
            # at construction: rename its index to q and shift the others.
            expr = re.sub(r'\[([a-d])\]', lambda m: '[' + ('q' if m[1] == 'a' else chr(ord(m[1]) - 1)) + ']', expr)
            result.append((name.lower(), dims[1:-1], dims[-1], expr, True))
    additions = [
        ('split', [16], 2, 'default_do_split_cdf[0][a]'),
        ('rect', [28], 2, 'default_rect_type_cdf[0][a]'),
        ('mode_set', [], 4, 'default_y_mode_set_cdf'),
        ('y_mode', [3], 8, 'default_y_mode_idx_cdf[a]'),
        ('y_mode_offset', [3], 6, 'default_y_mode_idx_offset_cdf[a]'),
        ('uv_mode', [2], 8, 'default_uv_mode_cdf[a]'),
        ('dpcm_y', [], 2, 'default_dpcm_cdf'),
        ('dpcm_uv', [], 2, 'default_dpcm_uv_cdf'),
        ('lossless_size', [4, 2], 2, 'default_lossless_tx_size_cdf[a][b]'),
        ('lossless_inter_tx', [], 2, 'default_lossless_inter_tx_type_cdf'),
        # Contexts 3..5 are skip-mode blocks' (av2_get_skip_txfm_context).
        ('block_skip', [6], 2, 'default_skip_txfm_cdfs[a]'),
        ('skip_mode', [3], 2, 'default_skip_mode_cdfs[a]'),
        ('skip_drl', [3], 2, 'default_skip_drl_cdf[a]'),
        ('is_inter', [4], 2, 'default_intra_inter_cdf[a]'),
        ('inter_mode', [5], 3, 'default_inter_single_mode_cdf[a]'),
        ('drl', [3, 5], 2, 'default_drl_cdf[a][b]'),
        ('mv_set', [], 2, 'default_nmv_context.joint_shell_set_cdf'),
        ('mv_class_low', [], 8, 'default_nmv_context.joint_shell_class_cdf_0[MV_PRECISION_QTR_PEL]'),
        ('mv_class_high', [], 8, 'default_nmv_context.joint_shell_class_cdf_1[MV_PRECISION_QTR_PEL]'),
        ('mv_low_offset', [2], 2, 'default_nmv_context.shell_offset_low_class_cdf[a]'),
        ('mv_class2_offset', [], 2, 'default_nmv_context.shell_offset_class2_cdf'),
        ('mv_offset', [16], 2, 'default_nmv_context.shell_offset_other_class_cdf[0][a]'),
        ('mv_col_greater', [2], 2, 'default_nmv_context.col_mv_greater_flags_cdf[a]'),
        ('mv_col_index', [4], 2, 'default_nmv_context.col_mv_index_cdf[a]'),
        ('cdef_index0', [4], 2, 'default_cdef_strength_index0_cdf[a]'),
        # Cross-component sample offset unit flags, by plane and context.
        ('ccso', [3, 4], 2, 'default_ccso_cdf[a][b]'),
        # Superblock quantizer deltas (read_delta_qindex: DELTA_Q_PROBS + 1 symbols).
        ('delta_q', [], 8, 'default_delta_q_cdf'),
        # Single-reference, direction-0 contexts only (offsets 0..3 of
        # av2_get_pred_context_switchable_interp).
        ('switchable_interp', [4], 3, 'default_switchable_interp_cdf[a]'),
    ]
    # cdef_cdf is ragged: row k codes the index of a frame with k + 3
    # strength sets, an alphabet of k + 2 symbols.
    additions += [(f'cdef_index{k + 3}', [], k + 2, f'default_cdef_cdf[{k}]') for k in range(6)]
    result.extend((*s, False) for s in additions)
    # TCQ's second-quantizer base sets go last: they live in their own
    # TcqCdfs, which only TCQ frames build (see `main`).
    return sorted(result, key=lambda group: group[0].endswith('_tcq'))


def rust_type(base, dims):
    for size in reversed(dims):
        base = f'[{base}; {size}]'
    return base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    assert subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip() == PIN
    groups = specs()
    c = ['#include <stdio.h>', '#include <stdint.h>', '#include <string.h>',
         '#include "av2/common/av2_common_int.h"',
         '#include "av2/common/entropy_inits_coeffs.h"',
         '#include "av2/common/entropy_inits_modes.h"',
         '#include "av2/common/entropy_inits_mv.h"',
         'int main(void) { for(int q=0;q<4;q++) { uint64_t hash=14695981039346656037ULL;']
    for _, dims, n, expression, _ in groups:
        for i, size in enumerate(dims):
            c.append(f'for(int {chr(97+i)}=0;{chr(97+i)}<{size};{chr(97+i)}++) {{')
        c.append(f'{{ avm_cdf_prob model[{n}+4]; memcpy(model,{expression},sizeof(model));')
        c.append(f'for(int j=0;j<{n}+4;j++)printf("%u ",model[j]); puts("");')
        c.append(f'for(int step=0;step<65;step++) {{ update_cdf(model,(step*7+step/3)%{n},{n});'
                 f' for(int j=0;j<{n};j++) hash=(hash^model[j])*1099511628211ULL; }} }}')
        c.extend('}' for _ in dims)
    c.append('printf("HASH %llu\\n",(unsigned long long)hash); } return 0; }')
    with tempfile.TemporaryDirectory(prefix='nc-av2-cdf-') as directory:
        path = Path(directory) / 'probe.c'
        path.write_text('\n'.join(c))
        subprocess.run(['cc', '-O1', '-I'+str(args.source), '-I'+str(args.build), str(path), '-o', str(path.with_suffix(''))], check=True)
        output = subprocess.check_output([str(path.with_suffix(''))], text=True).splitlines()
    it = iter(output)
    defaults = [[] for _ in groups]
    hashes = []
    for q in range(4):
        for index, (_, dims, n, _, _) in enumerate(groups):
            values = []
            for _ in itertools.product(*(range(size) for size in dims)):
                row = list(map(int, next(it).split()))
                assert len(row) == n + 4 and row[n] == 0
                values.append(row)
            defaults[index].append(values)
        hashes.append(int(next(it).split()[1]))
    try:
        next(it)
        raise AssertionError('extra output')
    except StopIteration:
        pass
    lines = ['// Generated by tools/codegen/check_av2_cdf_models.py; do not edit.',
             '// Original AVM copyright Alliance for Open Media.',
             '// BSD-3-Clause-Clear + separate AOM patent terms; see THIRD_PARTY.md.',
             'use super::Initial;']
    for (name, dims, n, _, quantized), categories in zip(groups, defaults):
        if not quantized:
            assert all(cat == categories[0] for cat in categories)
        values = iter(sum(categories, []) if quantized else categories[0])
        def array(shape):
            if not shape:
                row = next(values)
                return f'Initial {{ inverse: {row[:n]}, rates: {row[n+1:]} }}'
            return '[' + ',\n'.join(array(shape[1:]) for _ in range(shape[0])) + ']'
        shape = ([4] if quantized else []) + dims
        lines.append(f'pub(super) const {name.upper()}: {rust_type(f"Initial<{n}>", shape)} = {array(shape)};')
    lines.append('#[cfg(test)] pub(super) const TRACE_HASHES: [u64; 4] = [' + ', '.join(f'{h:_}' for h in hashes) + '];')
    generated = '\n'.join(lines)+'\n'
    main_groups = [g for g in groups if not g[0].endswith('_tcq')]
    tcq_groups = [g for g in groups if g[0].endswith('_tcq')]
    tcq_field = lambda name: name[:-len('_tcq')]
    def mapped(expr, depth):
        if depth == 0: return expr+'.build()'
        if depth == 1: return expr+'.map(Initial::build)'
        return expr+'.map(|value| '+mapped('value',depth-1)+')'
    def visit(root, members, action):
        out = []
        for field, dims in members:
            expr = f'{root}.{field}'
            if not dims:
                # `restart_adaptation` takes the field by method call; `trace`
                # wants a mutable reference.
                out.append(action(expr if root == 'self' else '&mut ' + expr)); continue
            expr = '&mut ' + expr
            for depth in range(len(dims)):
                out.append(f'for v{depth} in {expr} {{'); expr = f'v{depth}'
            out.append(action(expr))
            out.extend('}' for _ in dims)
        return out
    model = ['//! Mutable per-tile AV2 models for optional within-tile adaptation.',
             '//!', '//! Models preserve AVM identities even for equal defaults. Reset for each',
             '//! tile/frame while cross-frame context is disabled. RD trials price and adapt',
             '//! a scratch copy of these models (`entropy::Writer` estimates), so rejected',
             '//! trials never touch live state. Chroma DC signs in the current subset are bypass:',
             '//! the separate sign plane defaults are retained for explicit future use.',
             '//! TCQ\'s second-quantizer base models are a separate [`TcqCdfs`], built',
             '//! only by TCQ frames: kept among the hot models they cost the default',
             '//! encoder about 1% Jetson cycles.',
             'use crate::entropy::Cdf;', '#[path = "cdf_defaults.rs"] mod defaults;',
             '#[derive(Clone, Copy)] struct Initial<const N: usize> { inverse: [u16; N], rates: [u8; 3] }',
             'impl<const N: usize> Initial<N> { fn build(self) -> Cdf<N> {',
             'Cdf::new(self.inverse, self.rates).expect("independently verified AVM defaults") } }',
             '/// The coefficient CDF quantizer category of a frame base qindex (AVM `get_q_ctx`).',
             'fn category(qindex: u8) -> usize { if qindex <= 90 { 0 } else if qindex <= 140 { 1 } else if qindex <= 190 { 2 } else { 3 } }',
             '/// Probability models for the coded AV2 subset (coefficients TX4..TX64, modes, motion).',
             '/// Axes are named in field documentation; there is no pointer/value deduplication.',
             '/// A clone is one ~8 KiB block copy (plus TCQ\'s models where built).',
             '#[derive(Debug, Clone, PartialEq, Eq)] pub struct TileCdfs {']
    for name, dims, n, expr, _ in main_groups:
        model += [f'/// AVM `{expr}` (q selected at construction; a/b are array axes).',
                  f'pub {name}: {rust_type(f"Cdf<{n}>", dims)},']
    model += ['/// TCQ\'s second-quantizer base models, built on first use ([`Self::tcq`]).',
              'tcq: Option<Box<TcqCdfs>>,',
              '/// The quantizer category these models started from, which TCQ\'s take too.',
              'category: u8,', '}']
    model += ['/// The luma base models of TCQ\'s second quantizer (states 2, 3, 6 and 7).',
              '#[derive(Debug, Clone, Copy, PartialEq, Eq)] pub struct TcqCdfs {']
    for name, dims, n, expr, _ in tcq_groups:
        model += [f'/// AVM `{expr}` (q selected at construction; a/b are array axes).',
                  f'pub {tcq_field(name)}: {rust_type(f"Cdf<{n}>", dims)},']
    model += ['}', 'impl TcqCdfs {', '/// Fresh normative defaults of quantizer category `q`.',
              'fn new(q: usize) -> Self { Self {']
    for name, dims, _, _, quantized in tcq_groups:
        model.append(f'{tcq_field(name)}: {mapped(f"defaults::{name.upper()}[q]", len(dims))},')
    model += ['} }', '/// [`TileCdfs::restart_adaptation`] of these models.',
              'fn restart_adaptation(&mut self) {']
    model += visit('self', [(tcq_field(n), d) for n, d, *_ in tcq_groups], lambda e: f'{e}.restart_adaptation();')
    model += ['} }']
    model += ['impl TileCdfs {', '/// Fresh normative defaults for one tile at the frame base quantizer.',
              '#[must_use] pub fn new(qindex: u8) -> Self {',
              'let q = category(qindex);', 'Self {']
    for name, dims, _, _, quantized in main_groups:
        expression = f'defaults::{name.upper()}' + ('[q]' if quantized else '')
        model.append(f'{name}: {mapped(expression,len(dims))},')
    model += ['tcq: None, category: q as u8,', '} }',
              '/// TCQ\'s models, at their defaults (of the category these models started',
              '/// from, as AVM\'s frame contexts carry them) until a TCQ frame adapts them.',
              'pub fn tcq(&mut self) -> &mut TcqCdfs {',
              'let q = usize::from(self.category);',
              'self.tcq.get_or_insert_with(|| Box::new(TcqCdfs::new(q))) }',
              '/// TCQ\'s models where a TCQ frame built them ([`Self::tcq`]).',
              '#[must_use] pub fn built_tcq(&self) -> Option<&TcqCdfs> { self.tcq.as_deref() }',
              '/// Move every model back along its adaptation-rate schedule, keeping',
              '/// its probabilities: AVM `av2_reset_cdf_symbol_counters` on a',
              '/// frame\'s final models before later frames inherit them. TCQ\'s',
              '/// unbuilt defaults have nothing to move.',
              '#[allow(clippy::too_many_lines)] pub fn restart_adaptation(&mut self) {']
    model += visit('self', [(n, d) for n, d, *_ in main_groups], lambda e: f'{e}.restart_adaptation();')
    model += ['if let Some(tcq) = &mut self.tcq { tcq.restart_adaptation(); }', '} }',
              '#[cfg(test)] mod tests { use super::*; use crate::entropy::Writer;',
              'fn trace<const N:usize>(cdf:&mut Cdf<N>, hash:&mut u64) { let mut writer=Writer::new();',
              'for step in 0..65 { writer.symbol((step*7+step/3)%N,cdf).unwrap();',
              'for &value in cdf.inverse() { *hash=(*hash ^ u64::from(value)).wrapping_mul(1_099_511_628_211); } } }',
              '#[test] #[allow(clippy::too_many_lines)] fn original_c_update_traces_and_reset() {',
              'for (q, expected) in [0,91,141,191].into_iter().zip(defaults::TRACE_HASHES) {',
              'let mut models=TileCdfs::new(q); let snapshot=models.clone(); let mut hash=14_695_981_039_346_656_037;']
    model += visit('models', [(n, d) for n, d, *_ in main_groups], lambda e: f'trace({e}, &mut hash);')
    model += ['let tcq = models.tcq();']
    model += [l.replace('&mut tcq.', '&mut tcq.') for l in visit('tcq', [(tcq_field(n), d) for n, d, *_ in tcq_groups], lambda e: f'trace({e}, &mut hash);')]
    model += ['assert_eq!(hash,expected); assert_ne!(models,snapshot); assert_eq!(TileCdfs::new(q),snapshot);',
              'let mut restarted=models.clone(); restarted.restart_adaptation(); assert_ne!(restarted,models);',
              'assert_eq!(restarted.drl[0][0].inverse(),models.drl[0][0].inverse());',
              'models=snapshot.clone(); assert_eq!(models,snapshot); } }',
              '#[test] fn equal_defaults_have_independent_state() { let mut models=TileCdfs::new(100);',
              'let uv=models.dpcm_uv; let sign_uv=models.sign[1];',
              'Writer::new().symbol(0,&mut models.dpcm_y).unwrap();',
              'Writer::new().symbol(1,&mut models.sign[0][0]).unwrap();',
              'assert_eq!(models.dpcm_uv,uv); assert_eq!(models.sign[1],sign_uv);',
              'assert_ne!(models.dpcm_y.inverse(),models.dpcm_uv.inverse()); }',
              '#[test] fn tcq_models_start_at_their_categorys_defaults() {',
              'for q in [0,91,141,191] { let mut models=TileCdfs::new(q); assert!(models.built_tcq().is_none());',
              'let built=*models.tcq(); assert_eq!(built, TcqCdfs::new(category(q))); assert_eq!(models.built_tcq(), Some(&built)); } }',
              '#[test] fn quantizer_category_boundaries() {',
              'for (a,b) in [(0,90),(91,140),(141,190),(191,255)] { assert_eq!(TileCdfs::new(a),TileCdfs::new(b)); }',
              'for (a,b) in [(90,91),(140,141),(190,191)] { assert_ne!(TileCdfs::new(a),TileCdfs::new(b)); } }', '}']
    outputs = {'cdf_defaults.rs': generated, 'cdf_models.rs': '\n'.join(model)+'\n'}
    for name, text in outputs.items():
        # Format in temporary storage so --check is read-only.
        with tempfile.TemporaryDirectory(prefix='nc-av2-cdf-rust-') as directory:
            path = Path(directory)/name
            path.write_text(text)
            subprocess.run([str(Path.home()/'.cargo/bin/rustfmt'), '--edition', '2024', '--config', 'skip_children=true', str(path)], check=True)
            text=path.read_text()
        target=ROOT/'crates/nc-av2/src'/name
        if args.write:
            target.write_text(text)
        else:
            assert target.read_text()==text, f'{name} differs; regenerate with --write'
    print(f'PASS: {len(groups)} model families, four q categories, 65 C updates/model; hashes {hashes}')


if __name__ == '__main__':
    main()
