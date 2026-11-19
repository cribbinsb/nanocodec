// Coding efficiency vs rate control (rate-control.html): CBR vs CBR, fixed QP vs
// fixed QP and each encoder's own rate-control cost, per metric, with a few
// sentences computed from the same rows. Data: data/rc-cost.json.
import { METRIC_NAMES, METRIC_SHORT, control, el, fail, loadData, pct, segmented, sources, table } from './common.js';

const PSNR_ZERO = 'x264-medium~psnr';
const DEFAULT_ZERO = 'x264-medium';
const LABELS = { [PSNR_ZERO]: 'x264 medium --tune psnr', [DEFAULT_ZERO]: 'x264 medium, default tuning' };
const COLUMNS = [
  { line: 'cbr_vs_cbr', anchor: PSNR_ZERO, label: 'CBR vs CBR (x264 --tune psnr)' },
  { line: 'qp_vs_qp', anchor: PSNR_ZERO, label: 'QP vs QP (x264 --tune psnr)' },
  { line: 'cbr_vs_cbr', anchor: DEFAULT_ZERO, label: 'CBR vs CBR (x264 default)' },
  { line: 'qp_vs_qp', anchor: DEFAULT_ZERO, label: 'QP vs QP (x264 default)' },
  { line: 'rc_cost', anchor: null, label: 'Own CBR vs own fixed QP' },
];
const state = { metric: 'j' };

function name(family) {
  if (LABELS[family]) return LABELS[family];
  const codec = family.replace(/^nanocodec-/, '');
  return `nanocodec ${{ h264: 'H.264', hevc: 'HEVC', vvc: 'VVC', av1: 'AV1', vp9: 'VP9', av2: 'AV2' }[codec] || codec} Medium (default)`;
}

function render(data, root) {
  const find = (test, col) => data.rows.find((r) => r.test === test && r.line === col.line && (col.anchor ? r.anchor === col.anchor : r.anchor === test));
  const tests = [...new Set(data.rows.map((r) => r.test))];
  const ours = tests.filter((t) => t.startsWith('nanocodec-'))
    .sort((a, b) => find(a, COLUMNS[0]).j - find(b, COLUMNS[0]).j);
  const rows = [...ours, ...tests.filter((t) => !t.startsWith('nanocodec-'))];
  root.querySelector('.rc-table').replaceChildren(
    el('p', { class: 'note' }, `${METRIC_NAMES[state.metric]}; negative is fewer bits.`),
    table([
      { label: 'Encoder', cell: (t) => [el('span', { class: `swatch ${t.startsWith('nanocodec-') ? 'ours' : 'ref'}` }), name(t)] },
      ...COLUMNS.map((c) => ({ label: c.label, num: true, cell: (t) => {
        const r = find(t, c);
        return r ? `${pct(r[state.metric])} (${r.clips})` : '';
      } })),
    ], rows, (t) => (t.startsWith('nanocodec-') ? 'ours' : '')), sources(data.source));

  const cost = (t) => find(t, COLUMNS[4])[state.metric];
  const qp = (t, anchor) => find(t, { line: 'qp_vs_qp', anchor })[state.metric];
  const cbr = (t, anchor) => find(t, { line: 'cbr_vs_cbr', anchor })[state.metric];
  const newer = ours.filter((t) => t !== 'nanocodec-h264');
  const range = (values) => `${pct(Math.max(...values))} to ${pct(Math.min(...values))}`;
  const h264 = 'nanocodec-h264';
  const m = METRIC_SHORT[state.metric];
  root.querySelector('.findings').replaceChildren(
    el('li', {}, el('strong', {}, 'Rate-control cost: '),
      `ours ${pct(Math.min(...ours.map(cost)))} to ${pct(Math.max(...ours.map(cost)))} (${m}) against our own fixed QP; x264 medium ${pct(cost(PSNR_ZERO))} with --tune psnr and ${pct(cost(DEFAULT_ZERO))} at its default tuning.`),
    el('li', {}, el('strong', {}, 'H.264 Medium (default) at fixed QP: '),
      `${pct(qp(h264, PSNR_ZERO))} against --tune psnr and ${pct(qp(h264, DEFAULT_ZERO))} against the default tuning, against ${pct(cbr(h264, PSNR_ZERO))} and ${pct(cbr(h264, DEFAULT_ZERO))} in CBR.`),
    el('li', {}, el('strong', {}, 'Newer-format Medium (default) presets at fixed QP: '),
      `${range(newer.map((t) => qp(t, PSNR_ZERO)))} against x264 --tune psnr, against ${range(newer.map((t) => cbr(t, PSNR_ZERO)))} in CBR.`),
  );
}

async function main() {
  const root = document.getElementById('rc');
  try {
    const data = await loadData('rc-cost');
    root.querySelector('.run').textContent = `${data.measured.caption} Ladder ${data.ladder_kbps} kbps at 720p30, ${data.frames} frames, one key frame.`;
    root.querySelector('.controls').append(control('Metric', segmented('Metric',
      Object.keys(METRIC_SHORT).map((key) => ({ value: key, label: METRIC_SHORT[key] })), state.metric,
      (v) => { state.metric = v; render(data, root); })));
    render(data, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
