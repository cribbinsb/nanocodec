// Spatial scalability (svc.html): SVC, simulcast and single-layer points against
// single-layer x264 medium, as a chart (platform selectable) and a table.
// Data: data/svc.json.
import { drawScatter } from './chart.js';
import { control, el, fail, fps, loadData, measuredNote, pct, segmented, sources, table } from './common.js';

const SHAPES = { libaom: 'plus', libvpx: 'cross', x264: 'square' };
const PLATFORMS = { jetson: 'Jetson Orin Nano core', x86: 'Ryzen 9 5950X core' };
const state = { platform: 'jetson', selected: null };

function shapeOf(row) {
  const project = Object.keys(SHAPES).find((p) => row.family.startsWith(p));
  return project ? SHAPES[project] : 'circle';
}

function render(data, root) {
  const chart = root.querySelector('.chart');
  // a point not timed on this machine has no place on the speed axis (the table lists it)
  drawScatter(chart, data.rows.filter((r) => r.fps[state.platform] != null).map((r) => ({
    id: r.family, x: r.fps[state.platform], y: r.j, iqr: r.ours || r.j !== 0 ? r.iqr : null,
    kind: r.ours ? 'ours' : 'ref', shape: shapeOf(r), label: r.label, flagged: false, twin: null,
    priority: r.ours ? 2 : 1, aria: `${r.label}: ${pct(r.j)}, ${fps(r.fps[state.platform])} fps`,
  })), {
    xLabel: `encode fps (estimated), one ${PLATFORMS[state.platform]} (log scale)`,
    yLabel: 'BD-J, 720p output (lower is better)',
    ariaLabel: 'SVC: bitrate saving against encode speed',
    selected: state.selected,
    describe: (p) => {
      const r = data.rows.find((row) => row.family === p.id);
      return el('div', {}, el('strong', {}, r.label),
        el('div', {}, el('span', { class: 'value' }, pct(r.j)), ' BD-J of the 720p output'),
        r.base_j !== null ? el('div', {}, `base layer ${pct(r.base_j)} vs the codec's 360p alone`) : null,
        el('div', {}, `${fps(r.fps.jetson)} fps Jetson core · ${fps(r.fps.x86)} fps PC core`),
        el('div', { class: 'dim' }, `${r.clips} clips`));
    },
    onSelect: (p) => { state.selected = p.id; render(data, root); },
  });
}

async function main() {
  const root = document.getElementById('svc');
  try {
    const data = await loadData('svc');
    root.querySelector('.controls').append(control('Speed on', segmented('Platform', [
      { value: 'jetson', label: 'Jetson (Arm)' }, { value: 'x86', label: 'PC (x86)' },
    ], state.platform, (v) => { state.platform = v; render(data, root); })));
    render(data, root);
    root.querySelector('.svc-table').replaceChildren(table([
      { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.ours ? 'ours' : 'ref'}` }), r.label] },
      { label: 'BD-J, 720p (total rate)', num: true, cell: (r) => pct(r.j) },
      { label: '360p view vs its codec alone at 360p', num: true, cell: (r) => (r.base_j === null ? '' : pct(r.base_j)) },
      { label: '180p view vs its codec alone at 180p', num: true, cell: (r) => (r.op180_j === null ? '' : pct(r.op180_j)) },
      { label: 'fps Jetson', num: true, cell: (r) => fps(r.fps.jetson) },
      { label: 'fps PC', num: true, cell: (r) => fps(r.fps.x86) },
    ], data.rows, (r) => (r.ours ? 'ours' : '')), measuredNote(data.measured.caption), sources(data.source));
    let width = root.clientWidth;
    new ResizeObserver(() => {
      if (Math.abs(root.clientWidth - width) > 8) { width = root.clientWidth; render(data, root); }
    }).observe(root);
  } catch (error) {
    fail(root, error);
  }
}

main();
