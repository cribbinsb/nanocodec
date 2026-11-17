// Fixed QP on the standards lens (fixed-qp.html): every encoder at fixed quantizers on
// the AOM low-delay common test conditions, against x264 medium, in the Strict,
// Profile and CBR-settings columns (fixed-qp.html says what each is). Data: data/fixed-qp-lens.json (the fixed-QP lens run; rows
// {test, column, anchor, weighted, j, psnr_yuv, vmaf, ms_ssim, xpsnr?, n}), backed by a
// docs/results CSV.
import { NOT_MEASURED, control, el, fail, loadData, pct, segmented, sources, table } from './common.js';

const COLUMNS = { strict: 'Strict', profile: 'Profile', cbr_settings: 'CBR settings' };
const METRICS = {
  weighted: 'Weighted (0.92 Y + 0.04 Cb + 0.04 Cr PSNR)',
  j: 'J (PSNR-YUV, VMAF v1, MS-SSIM)',
  psnr_yuv: 'PSNR-YUV',
  vmaf: 'VMAF v1',
  ms_ssim: 'MS-SSIM',
  xpsnr: 'XPSNR-YUV (not part of J)',
};
const state = { metric: 'weighted' };

/** The rate-mode switch shared with the CBR overview: this page is the fixed-QP side. */
function modeSwitch(root) {
  root.append(control('Rate mode', segmented('Rate mode', [
    { value: 'cbr', label: 'CBR (real time)' }, { value: 'qp', label: 'Fixed QP' },
  ], 'qp', (value) => { if (value === 'cbr') window.location.href = './'; })));
}

function render(data, names, root) {
  const tests = [...new Set(data.rows.map((r) => r.test))];
  const find = (test, column) => data.rows.find((r) => r.test === test && r.column === column);
  const ours = (t) => t.startsWith('nanocodec-');
  const key = (t) => Object.keys(COLUMNS).map((c) => find(t, c)).find(Boolean)[state.metric] ?? Infinity;
  tests.sort((a, b) => (ours(b) - ours(a)) || (key(a) - key(b)));
  root.querySelector('.lens-table').replaceChildren(
    el('p', { class: 'note' }, `${METRICS[state.metric]}, BD-rate against ${data.rows[0].anchor}; negative is fewer bits. Sequences in brackets.`),
    table([
      { label: 'Encoder', cell: (t) => [el('span', { class: `swatch ${ours(t) ? 'ours' : 'ref'}` }), names(t)] },
      ...Object.entries(COLUMNS).map(([column, label]) => ({
        label, num: true,
        cell: (t) => {
          // a reference encoder has one setting: the same value in every column
          const r = find(t, column) || (ours(t) ? null : data.rows.find((x) => x.test === t));
          if (!r) return '';
          return r[state.metric] != null ? `${pct(r[state.metric])} (${r.n})` : NOT_MEASURED;
        },
      })),
    ], tests, (t) => (ours(t) ? 'ours' : '')),
    sources([data.source].flat()));
}

async function main() {
  modeSwitch(document.querySelector('.mode-switch'));
  const root = document.getElementById('fixed-qp');
  let data;
  let names;
  try {
    data = await loadData('fixed-qp-lens');
    // display names: the overview's, through its fixed-QP view's lens-test map
    const chart = await loadData('chart');
    const lens = chart.views.find((v) => v.id === 'qp')?.lens_names || {};
    names = (t) => chart.points[lens[t] || t]?.label || t;
  } catch (error) {
    root.querySelector('.run').textContent = 'The fixed-QP lens has not been published for this build yet.';
    return;
  }
  try {
    root.querySelector('.run').textContent = `Build ${data.build_commit}; condition ${data.condition}.`;
    root.querySelector('.controls').append(control('Metric', segmented('Metric',
      Object.keys(METRICS).map((value) => ({ value, label: value === 'weighted' ? 'Weighted' : METRICS[value].split(' (')[0] })),
      state.metric, (v) => { state.metric = v; render(data, names, root); })));
    render(data, names, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
