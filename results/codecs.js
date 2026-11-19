// Results by codec (codecs.html): for each format, our presets and the reference
// encoders of that format, on the fresh and all held-out clips and the development
// clips, for the chosen zero line and metric. Data: data/chart.json.
import {
  CODEC_NAMES, METRIC_NAMES, METRIC_SHORT, ZERO_NAMES, control, el, fail, fps, loadData, measuredNote, pct,
  segmented, sources, table,
} from './common.js';

const TUNED_ZERO = 'x264-medium~psnr';
const PRESET_ORDER = ['Slow', 'Medium', 'Fast', 'Faster', 'Veryfast'];
const state = { zero: TUNED_ZERO, metric: 'j' };

function views(data) {
  const zid = state.zero === TUNED_ZERO ? 'psnr' : 'default';
  const byId = Object.fromEntries(data.views.map((v) => [v.id, v]));
  return [
    { title: 'Held-out, fresh', view: byId[`fresh-${zid}`] },
    { title: 'Held-out, all', view: byId[`heldout-${zid}`] },
    { title: 'Development', view: byId[`dev-${zid}`] },
    ...Object.entries(data.datasets).map(([id, d]) => ({ title: d.short, view: byId[`${id}-${zid}`] })),
  ].filter((c) => c.view);
}

function cellOf(view, family) {
  const v = view.values[family];
  if (!v || v[state.metric] === null || v[state.metric] === undefined) return '–';
  return pct(v[state.metric]) + (v.clips_scored ? ` (${v.clips_scored})` : '');
}

function render(data, root) {
  const columns = views(data);
  const fresh = columns[0].view;
  const out = data.codec_order.map((codec) => {
    const families = Object.keys(data.points).filter((f) => {
      const p = data.points[f];
      return p.codec === codec && columns.some((c) => c.view.values[f]) && (!p.tune || f === state.zero);
    });
    const ours = families.filter((f) => data.points[f].ours)
      .sort((a, b) => PRESET_ORDER.indexOf(data.points[a].preset) - PRESET_ORDER.indexOf(data.points[b].preset));
    const refs = families.filter((f) => !data.points[f].ours)
      .sort((a, b) => (fresh.values[b]?.fps.jetson ?? 0) - (fresh.values[a]?.fps.jetson ?? 0));
    const speedOf = (f) => columns.map((c) => c.view.values[f]).find(Boolean)?.fps;
    return el('section', {},
      el('h2', { id: codec }, CODEC_NAMES[codec]),
      table([
        { label: 'Encoder', cell: (f) => [el('span', { class: `swatch ${data.points[f].ours ? 'ours' : 'ref'}` }), data.points[f].label] },
        ...columns.map((c) => ({ label: `${c.title} (clips)`, num: true, cell: (f) => cellOf(c.view, f) })),
        { label: 'fps Jetson', num: true, cell: (f) => fps(speedOf(f)?.jetson) },
        { label: 'fps PC', num: true, cell: (f) => fps(speedOf(f)?.x86) },
      ], [...ours, ...refs], (f) => (data.points[f].ours ? 'ours' : '') + (f === state.zero ? ' zero' : '')));
  });
  root.querySelector('.codec-tables').replaceChildren(
    el('p', { class: 'note' }, `${METRIC_NAMES[state.metric]} against ${ZERO_NAMES[state.zero]}; in brackets, the clips with a reliable BD. Clip counts per set: ${columns.filter((c) => c.view.clips).map((c) => `${c.title} ${c.view.clips}`).join(', ')}.`),
    ...out, ...columns.map((c) => measuredNote(`${c.title}: ${c.view.measured.caption}`)),
    sources([...new Set(columns.flatMap((c) => c.view.source))]));
}

async function main() {
  const root = document.getElementById('codecs');
  try {
    const data = await loadData('chart');
    // the lede names each published outside dataset (data.datasets) after our clip sets
    document.querySelector('.datasets-note').textContent = Object.values(data.datasets)
      .map((d) => `, and on ${d.name}, ${d.credit}, numbers only`).join('');
    root.querySelector('.controls').append(
      control('Metric', segmented('Metric', ['j', 'psnr', 'vmaf'].map((m) => ({ value: m, label: METRIC_SHORT[m] })),
        state.metric, (v) => { state.metric = v; render(data, root); })),
    );
    render(data, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
