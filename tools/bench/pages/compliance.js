// Rate control (compliance.html): for one dataset at a time on the chosen timing machine, how
// every encoder's CBR streams use the shared 1 s buffer, from the report's one external bucket
// replay (procedure section 8): a strip chart of each family's peak fullness, the figures as a
// table, and the exclusion check of each family that lost clips (section 9).
// Data: data/bp3.json (tools/bench/preview.py).
import { control, el, fail, loadData, pct, segmented, table } from './common.js';
import { banner, chosenMachine, machinePicker, statement, viewTitle, viewsOn } from './bp3.js';

const SVG = 'http://www.w3.org/2000/svg';
const DATASET_PARAM = 'set';
// The bucket replay's two drains (report.py): the stream's own achieved rate (the fair one:
// an encoder cannot gain by undershooting) and the target rate.
const MEASURES = { own_rate: 'At the stream\'s own rate', target_rate: 'At the target rate' };
const state = { dataset: new URLSearchParams(location.search).get(DATASET_PARAM), measure: 'own_rate' };

const share = (value) => `${(100 * value).toFixed(1)}%`;
const peak = (measure, key) => (measure ? measure[key].toFixed(2) : '–');
const over = (measure) => (measure ? share(measure.share_points_over_1) : '–');
const ratio = (value) => (value === null || value === undefined ? '–' : value.toFixed(3));

function svg(tag, attrs = {}, text = null) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== null) node.textContent = text;
  return node;
}

/** One row per family: a bar from its median to its 95th-percentile peak, a tick at its
 *  worst point, and the one-buffer line; log scale, since the worst peaks reach many buffers. */
function stripChart(rows, measure) {
  const row = 17;
  const label = 230;
  const right = 20;
  const top = 26;
  const width = 900;
  const height = top + rows.length * row + 34;
  const values = rows.flatMap((r) => [r.c[measure].median_peak_buffers, r.c[measure].worst_peak_buffers]);
  const lo = Math.min(0.25, ...values) * 0.9;
  const hi = Math.max(2, ...values) * 1.1;
  const x = (v) => label + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * (width - label - right);
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-label': 'Peak buffer fullness per encoder' });
  const grid = svg('g', { class: 'grid' });
  for (const t of [0.25, 0.5, 1, 2, 5, 10, 20, 50].filter((v) => v >= lo && v <= hi)) {
    grid.append(svg('line', { x1: x(t), x2: x(t), y1: top - 6, y2: height - 30 }));
    grid.append(svg('text', { x: x(t), y: height - 14, 'text-anchor': 'middle' }, String(t)));
  }
  root.append(grid);
  root.append(svg('line', { class: 'zero-line', x1: x(1), x2: x(1), y1: top - 10, y2: height - 30 }));
  root.append(svg('text', { class: 'axis-title', x: x(1) + 4, y: top - 12 }, 'one buffer (1 s)'));
  root.append(svg('text', { class: 'axis-title', x: label + (width - label) / 2, y: height - 1, 'text-anchor': 'middle' },
    'peak fullness in buffers (log scale): bar median to 95th percentile of points, tick the worst point'));
  rows.forEach((r, i) => {
    const m = r.c[measure];
    const y = top + i * row + row / 2;
    const kind = r.p.ours ? 'ours' : 'ref';
    const g = svg('g');
    g.append(svg('title', {}, `${r.p.label}: median ${m.median_peak_buffers.toFixed(2)}, 95th percentile ${m.p95_peak_buffers.toFixed(2)}, worst ${m.worst_peak_buffers.toFixed(2)} buffers; ${share(m.share_points_over_1)} of points over one buffer`));
    g.append(svg('text', { x: label - 8, y: y + 4, 'text-anchor': 'end', class: `point-label ${kind}` }, r.p.label));
    g.append(svg('line', { x1: x(m.median_peak_buffers), x2: x(m.worst_peak_buffers), y1: y, y2: y, class: `strip-thin ${kind}` }));
    g.append(svg('rect', { x: x(m.median_peak_buffers), y: y - 4, width: Math.max(2, x(m.p95_peak_buffers) - x(m.median_peak_buffers)), height: 8, rx: 2, class: `strip ${kind}` }));
    g.append(svg('line', { x1: x(m.worst_peak_buffers), x2: x(m.worst_peak_buffers), y1: y - 5, y2: y + 5, class: `strip-thin ${kind}` }));
    root.append(g);
  });
  return el('div', { class: 'chart-card' }, el('p', { class: 'chart-title' }, `Peak buffer fullness, ${MEASURES[measure].toLowerCase()}`),
    el('p', { class: 'chart-sub' }, 'Each CBR point\'s frames replayed through the 1 s bucket; lower is calmer. Hover a row for its numbers.'),
    el('div', { class: 'chart strips' }, root));
}

function section(data, view) {
  const rows = Object.entries(view.values).filter(([, v]) => v.compliance)
    .map(([family, v]) => ({ family, v, c: v.compliance, x: v.exclusion_check, p: data.points[family] }))
    .sort((a, b) => (b.p.ours - a.p.ours) || a.p.label.localeCompare(b.p.label));
  const charted = [...rows].sort((a, b) => a.c[state.measure].p95_peak_buffers - b.c[state.measure].p95_peak_buffers);
  const cls = (r) => (r.p.ours ? 'ours' : '');
  const name = { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.p.ours ? 'ours' : 'ref'}` }), r.p.label] };
  return el('section', {},
    el('h2', { id: view.id }, viewTitle(data, view)),
    el('p', { class: 'note measured' }, statement(data, view)),
    stripChart(charted, state.measure),
    el('details', { class: 'table-view' }, el('summary', {}, 'The figures as a table'),
      table([
        name,
        { label: 'Points', num: true, cell: (r) => String(r.c.points) },
        { label: 'Frames dropped', num: true, cell: (r) => (r.c.dropped_frames ? share(r.c.share_frames_dropped) : '0') },
        { label: 'Own rate: 95th-percentile peak (buffers)', num: true, cell: (r) => peak(r.c.own_rate, 'p95_peak_buffers') },
        { label: 'Own rate: worst peak', num: true, cell: (r) => peak(r.c.own_rate, 'worst_peak_buffers') },
        { label: 'Own rate: points over 1.0', num: true, cell: (r) => over(r.c.own_rate) },
        { label: 'Target rate: worst peak', num: true, cell: (r) => peak(r.c.target_rate, 'worst_peak_buffers') },
        { label: 'Target rate: points over 1.0', num: true, cell: (r) => over(r.c.target_rate) },
        { label: 'First key, median (s of own rate)', num: true, cell: (r) => (r.c.first_key_seconds ? r.c.first_key_seconds.median.toFixed(2) : '–') },
        { label: 'Rate over target', num: true, cell: (r) => ratio(r.c.mean_rate_over_target) },
      ], rows, cls)),
    el('h3', {}, 'Exclusion check: the families that lost clips'),
    el('p', { class: 'note' }, 'For each family with a malformed curve on some clip: its rate over target on the lost and on the kept clips, and our BD-J on those clips (as the report prints it). The Clips page lists which clips.'),
    table([
      name,
      { label: 'Lost clips', num: true, cell: (r) => String(r.x.missing_clips) },
      { label: 'Rate over target, lost', num: true, cell: (r) => ratio(r.x.rate_over_target_missing) },
      { label: 'Rate over target, kept', num: true, cell: (r) => ratio(r.x.rate_over_target_kept) },
      { label: 'Our BD-J on the lost clips', num: true, cell: (r) => pct(r.x.ours_bd_j_on_missing, 2) },
      { label: 'Our BD-J on the kept clips', num: true, cell: (r) => pct(r.x.ours_bd_j_on_kept, 2) },
    ], rows.filter((r) => r.x && r.x.missing_clips), cls));
}

async function main() {
  const root = document.getElementById('compliance');
  try {
    const data = await loadData('bp3');
    let machine = chosenMachine(data.machines);
    const show = () => {
      banner(data, machine);
      const views = viewsOn(data, machine);
      const view = views.find((v) => v.dataset === state.dataset) || views[0];
      root.querySelector('.view').replaceChildren(view ? section(data, view)
        : el('p', { class: 'note' }, 'No report found on this machine yet.'));
    };
    if (!(state.dataset in data.datasets)) [state.dataset] = Object.keys(data.datasets);
    const controls = root.querySelector('.controls');
    controls.append(
      control('Dataset', segmented('Dataset', Object.entries(data.datasets).map(([value, d]) => ({ value, label: d.name })),
        state.dataset, (v) => {
          state.dataset = v;
          const url = new URL(location.href);
          url.searchParams.set(DATASET_PARAM, v);
          history.replaceState(null, '', url);
          show();
        })),
      control('Bucket drained', segmented('Bucket drained', Object.entries(MEASURES).map(([value, label]) => ({ value, label })),
        state.measure, (v) => { state.measure = v; show(); })));
    machinePicker(data.machines, machine, (m) => { machine = m; show(); });
    show();
  } catch (error) {
    fail(root, error);
  }
}

main();
