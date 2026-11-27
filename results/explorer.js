// The overview (index.html): BD-rate against encode speed for every family of a benchmark
// report, by timing machine, dataset and metric, on the common set, with the Pareto frontiers
// of what is shown, and the same view as a table. Data: data/bp3.json (tools/bench/preview.py).
import { paretoFront, frontierLine } from './frontier.js';
import { drawScatter, pinSwatch, swatch } from './chart.js';
import { CODEC_NAMES, control, el, fail, loadData, pct, segmented, table } from './common.js';
import {
  NAMES, SHORT, SPEED_NAME, SPEED_UNIT, banner, chosenMachine, clipsNote, interval, machinePicker, mean, speed, statement,
  viewOf, viewTitle, viewsOn, zeroFamily, zeroName,
} from './bp3.js';

const SHAPES = {
  nanocodec: 'circle', x264: 'square', x265: 'diamond', 'SVT-AV1': 'triangle', openh264: 'triangle-down',
  libvpx: 'cross', libaom: 'plus', Kvazaar: 'pentagon', VVenC: 'hexagon',
};
const ALL = 'all';
const PIN_NOTE = 'points beyond the axes are pinned to the edge (arrow)';
const state = {
  machine: null, dataset: null, metric: 'j', frontier: true, preset: ALL, codec: ALL,
  hiddenOwners: new Set(), tuned: false, drop: false, selected: null,
};

function visible(data, view) {
  const zero = zeroFamily(view, state.metric);
  return Object.entries(view.values).filter(([family, v]) => {
    const p = data.points[family];
    if (mean(v.bd[state.metric]) === null) return false;
    if (state.hiddenOwners.has(p.owner)) return false;
    if (state.codec !== ALL && p.codec !== state.codec) return false;
    if (p.ours && state.preset !== ALL && p.preset !== state.preset) return false;
    if (p.drop && !state.drop) return false;
    if (p.metric_tuned && family !== zero && !state.tuned) return false;
    return true;
  });
}

function describe(data, view, family) {
  const p = data.points[family];
  const v = view.values[family];
  const bd = v.bd[state.metric];
  return el('div', {},
    el('strong', {}, p.label),
    el('div', {}, el('span', { class: 'value' }, pct(bd.mean)), ` ${SHORT[state.metric]} vs ${zeroName(view, state.metric)}`),
    el('div', { class: 'dim' }, `95 % interval ${interval(bd)}, ${clipsNote(v)}`),
    el('div', {}, `${speed(v.speed)} ${SPEED_UNIT} over ${v.speed_clips} clips`),
    el('div', { class: 'dim' }, 'Select the point for every metric'));
}

function detail(data, view, family) {
  const p = data.points[family];
  const v = view.values[family];
  const c = v.compliance;
  const x = v.exclusion_check;
  const share = (s) => `${(100 * s).toFixed(2)}%`;
  const rows = [
    ['Encoder', p.owner], ['Codec', CODEC_NAMES[p.codec]],
    ['Preset', p.preset + (p.tune ? `, tune ${p.tune}` : '') + (p.drop ? ', its own frame dropping on (+drop)' : '')],
    ...data.metrics.map((m) => [`${NAMES[m]} (${clipsNote(v)})`, `${pct(mean(v.bd[m]), 2)} (95 % interval ${interval(v.bd[m])})`]),
    ['Its own anchor-valid clips', String(v.own_clips)],
    ['Clips lost to a malformed curve', String(v.lost)],
    ['Encode speed', `${speed(v.speed)} ${SPEED_UNIT} (${SPEED_NAME}) over ${v.speed_clips} clips, normalised to the timing references (raw ${speed(v.speed_raw)})`],
    ['On the report’s front (common set, BD-J and speed, all families)', view.front.includes(family) ? 'yes' : 'no'],
    c ? ['Rate compliance (one bucket for every encoder)', `${share(c.share_frames_over)} of ${c.frames} frames over the bucket; ${share(c.share_points_over)} of ${c.points} points with a frame over it; worst excess ${c.worst_excess_buffers.toFixed(2)} buffers; 95th-percentile peak ${c.p95_peak_buffers.toFixed(2)} buffers; rate ${c.mean_rate_over_target.toFixed(3)} of target`] : null,
    x ? ['Exclusion check', `on its ${x.missing_clips} lost clips, rate ${x.rate_over_target_missing === null ? '–' : x.rate_over_target_missing.toFixed(3)} of target (kept clips ${x.rate_over_target_kept.toFixed(3)})`] : null,
    Object.keys(v.envelope_dropped).length ? ['Points dropped by the monotone envelope', Object.entries(v.envelope_dropped).map(([m, n]) => `${SHORT[m]} ${n}`).join(', ')] : null,
  ].filter(Boolean);
  return el('section', { class: 'detail', 'aria-live': 'polite' },
    el('h3', {}, p.label),
    el('dl', {}, rows.flatMap(([k, val]) => [el('dt', {}, k), el('dd', {}, val)])));
}

function render(data, root) {
  const views = viewsOn(data, state.machine);
  if (!views.length) {
    root.querySelector('.controls').replaceChildren();
    machinePicker(data.machines, state.machine, (machine) => pickMachine(data, root, machine));
    root.querySelector('.view-note').textContent = `No report found on ${data.machines[state.machine].name} yet.`;
    for (const slot of ['.chart', '.detail-slot', '.table-slot']) root.querySelector(slot).replaceChildren();
    root.querySelector('.chart-title').textContent = '';
    root.querySelector('.chart-sub').textContent = '';
    return;
  }
  const view = viewOf(data, state.dataset, state.machine) || views[0];
  state.dataset = view.dataset;
  controls(data, root);
  machinePicker(data.machines, state.machine, (machine) => pickMachine(data, root, machine));
  root.querySelector('.view-note').replaceChildren(statement(data, view));

  const entries = visible(data, view);
  const zero = zeroFamily(view, state.metric);
  const points = entries.map(([family, v]) => {
    const p = data.points[family];
    const bd = v.bd[state.metric];
    const base = family.split('~')[0].replace('+drop', '');
    const soft = p.drop || (p.metric_tuned && family !== zero);
    return {
      id: family, x: v.speed, y: bd.mean, iqr: family === zero ? null : [bd.low, bd.high],
      kind: p.ours ? 'ours' : (soft ? 'tuned' : 'ref'),
      shape: SHAPES[p.owner] || 'circle', label: p.label.replace(/^nanocodec /, ''),
      twin: soft && view.values[base] ? base : null,
      priority: (p.ours ? 2 : 1) + (p.preset === 'Medium' ? 1 : 0),
      aria: `${p.label}: ${pct(bd.mean)}, ${speed(v.speed)} ${SPEED_UNIT}`,
    };
  }).filter((p) => p.x);
  const chart = root.querySelector('.chart');
  const title = root.querySelector('.chart-title');
  title.textContent = `${NAMES[state.metric]} vs ${zeroName(view, state.metric)}, against encode speed`;
  const sub = root.querySelector('.chart-sub');
  sub.textContent = `${viewTitle(data, view)} · ${data.machines[view.machine].label} · ${view.procedure} · common set (${view.common_clips} of ${view.view_clips} clips) · lower is fewer bits, right is faster · bars: 95 % bootstrap interval over clips`;
  let pinned = [];
  if (!points.length) {
    chart.replaceChildren(el('p', { class: 'note' }, 'No points match these filters.'));
  } else {
    const frontiers = !state.frontier ? [] : ['ours', 'ref'].map((kind) => {
      const front = paretoFront(points.filter((p) => (kind === 'ours') === (p.kind === 'ours')));
      return { kind, line: frontierLine(front), ids: new Set(front.map((p) => p.id)) };
    });
    pinned = drawScatter(chart, points, {
      frontiers,
      xLabel: `${SPEED_NAME}, ${SPEED_UNIT}, ${data.machines[view.machine].name} (log scale)`,
      yLabel: `${SHORT[state.metric]} BD-rate (lower is better)`,
      ariaLabel: title.textContent,
      selected: state.selected,
      describe: (p) => describe(data, view, p.id),
      onSelect: (p) => { state.selected = p.id; render(data, root); root.querySelector('.detail-slot').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); },
    }).pinned;
  }
  root.querySelector('.pin-key').hidden = !pinned.length;
  if (pinned.length) sub.textContent += ` · ${PIN_NOTE}`;
  const twins = Object.entries(view.same_streams);
  if (twins.length) {
    chart.append(el('p', { class: 'note' }, `Shown once (the report measured the twin identically): ${twins.map(([twin, kept]) => `${twin} as ${kept}`).join('; ')}.`));
  }
  const slot = root.querySelector('.detail-slot');
  const chosen = state.selected && view.values[state.selected];
  slot.replaceChildren(chosen ? detail(data, view, state.selected)
    : el('p', { class: 'note' }, 'Select a point (click, tap or Enter) for every metric, its lost clips and its rate compliance.'));

  const rows = entries.map(([family, v]) => ({ family, v, p: data.points[family] }))
    .sort((a, b) => mean(a.v.bd[state.metric]) - mean(b.v.bd[state.metric]));
  const m = SHORT[state.metric];
  root.querySelector('.table-slot').replaceChildren(
    el('p', { class: 'note' }, `Sorted by the chart's BD. Front: on the report's front, the families no other family beats on both BD-J and speed on the common set. A +drop row has only its own clips. Own clips: each family's own anchor-valid clips.`),
    table([
      { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.p.ours ? 'ours' : 'ref'}` }), r.p.label] },
      { label: 'Front', cell: (r) => (view.front.includes(r.family) ? 'front' : '') },
      { label: `BD-${m}`, num: true, cell: (r) => pct(mean(r.v.bd[state.metric]), 2) },
      { label: '95 % interval', num: true, cell: (r) => interval(r.v.bd[state.metric]) },
      { label: 'Clips', num: true, cell: (r) => String(r.v.clips) + (r.v.own_only ? ' (own)' : '') },
      { label: 'Own clips', num: true, cell: (r) => String(r.v.own_clips) },
      { label: 'Lost', num: true, cell: (r) => String(r.v.lost) },
      { label: SPEED_UNIT, num: true, cell: (r) => speed(r.v.speed) },
      { label: 'Timed clips', num: true, cell: (r) => String(r.v.speed_clips) },
    ], rows, (r) => (r.p.ours ? 'ours' : '') + (r.family === zero ? ' zero' : '')),
    el('p', { class: 'sources' }, 'Data: ', el('code', {}, view.source)));
}

function toggle(label, title, key, kind, data, root) {
  return el('button', {
    type: 'button', 'aria-pressed': String(state[key]), title,
    onclick: () => { state[key] = !state[key]; legend(data, root); render(data, root); },
  }, swatch('square', kind), label);
}

function legend(data, root) {
  const owners = [...new Set(Object.values(data.points).map((p) => p.owner))];
  owners.sort((a, b) => (a === 'nanocodec' ? -1 : b === 'nanocodec' ? 1 : a.localeCompare(b)));
  root.querySelector('.legend').replaceChildren(...owners.map((owner) => el('button', {
    type: 'button', 'aria-pressed': String(!state.hiddenOwners.has(owner)),
    onclick: () => {
      if (state.hiddenOwners.has(owner)) state.hiddenOwners.delete(owner); else state.hiddenOwners.add(owner);
      legend(data, root);
      render(data, root);
    },
  }, swatch(SHAPES[owner], owner === 'nanocodec' ? 'ours' : 'ref'), owner === 'nanocodec' ? 'nanocodec (ours)' : owner)),
  toggle('metric-tuned variants (~psnr, ~ssim)', 'The references’ own metric tunes; each is linked to its default-tune twin', 'tuned', 'tuned', data, root),
  toggle('+drop rows', 'libvpx and libaom with their own frame dropping on, each on its own clips', 'drop', 'tuned', data, root),
  el('span', { class: 'frontier-key', title: 'A point is on its frontier when no other shown point of its group is both faster and lower (better) on this metric' },
    el('span', { class: 'frontier-swatch ours' }), 'our Pareto frontier ',
    el('span', { class: 'frontier-swatch ref' }), 'the references’ (of the points shown)'),
  el('span', { class: 'pin-key', hidden: true }, pinSwatch(), PIN_NOTE));
}

/** A new timing machine: the banner and everything below it, the selection cleared. */
function pickMachine(data, root, machine) {
  state.machine = machine;
  state.selected = null;
  banner(data, machine);
  render(data, root);
}

function controls(data, root) {
  const rerender = () => render(data, root);
  const open = () => { state.selected = null; rerender(); };
  const presets = [...new Set(Object.values(data.points).filter((p) => p.ours).map((p) => p.preset))];
  const select = (label, key, options) => {
    const node = el('select', { 'aria-label': label, onchange: (e) => { state[key] = e.target.value; rerender(); } },
      options.map(([value, text]) => el('option', { value }, text)));
    node.value = state[key];
    return node;
  };
  const datasets = Object.keys(data.datasets).filter((d) => viewOf(data, d, state.machine));
  root.querySelector('.controls').replaceChildren(
    control('Dataset', segmented('Dataset', datasets.map((d) => ({ value: d, label: data.datasets[d].name })),
      state.dataset, (v) => { state.dataset = v; open(); })),
    control('Metric', segmented('Metric', data.metrics.map((m) => ({ value: m, label: SHORT[m] })),
      state.metric, (v) => { state.metric = v; rerender(); })),
    control('Pareto frontier', segmented('Pareto frontier', [{ value: 'on', label: 'On' }, { value: 'off', label: 'Off' }],
      state.frontier ? 'on' : 'off', (v) => { state.frontier = v === 'on'; rerender(); })),
    control('Codec', select('Codec', 'codec', [[ALL, 'All codecs'], ...data.codec_order.map((c) => [c, CODEC_NAMES[c]])])),
    control('Preset', select('Preset', 'preset', [[ALL, 'All presets'], ...presets.map((p) => [p, `${p} (ours)`])])),
  );
}

async function main() {
  const root = document.getElementById('explorer');
  try {
    const data = await loadData('bp3');
    state.machine = chosenMachine(data.machines);
    banner(data, state.machine);
    legend(data, root);
    render(data, root);
    let width = root.clientWidth;
    new ResizeObserver(() => {
      if (Math.abs(root.clientWidth - width) > 8) { width = root.clientWidth; render(data, root); }
    }).observe(root);
  } catch (error) {
    fail(root, error);
  }
}

main();
