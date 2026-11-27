// The clips (clips.html): each benchmark dataset's clips as the benchmark coded them (the
// committed manifest's names and numbers, the shared ladder's rungs) and, from the chosen
// timing machine's report, the families whose curves on a clip were malformed, which is what
// takes a clip out of the common set. Data: data/clips.json (tools/bench/preview_pages.py).
import { control, el, fail, loadData, segmented, table } from './common.js';
import { banner, chosenMachine, machinePicker } from './bp3.js';

const DATASET_PARAM = 'set';
const state = { dataset: new URLSearchParams(location.search).get(DATASET_PARAM) };

/** The families that lost a clip's window on this machine, and whether that takes the window
 *  out of the common set (a +drop row's loss does not). */
function losses(data, view, clip, window) {
  const lost = view?.lost[clip.name] || {};
  const families = [...(lost[String(window)] || []), ...(lost.all || [])];
  return { families, out: families.some((f) => !f.endsWith(data.drop_suffix)) };
}

function lostCell(data, view, clip) {
  if (!view) return 'no report';
  const parts = clip.windows.map((w) => ({ w, ...losses(data, view, clip, w) }))
    .filter((p) => p.families.length);
  if (!parts.length) return 'none';
  return parts.map((p) => el('div', { title: p.families.join(', ') },
    `${clip.windows.length > 1 ? `${p.w}: ` : ''}${p.families.length === 1 ? p.families[0] : `${p.families.length} families`}`,
    p.out ? el('strong', {}, ' (out of the common set)') : ''));
}

function render(data, machine, root) {
  const d = data.datasets[state.dataset];
  const view = d.views[machine];
  const windows = d.clips.reduce((n, c) => n + c.windows.length, 0);
  root.querySelector('.summary').replaceChildren(
    `${d.name}: ${d.about}. ${d.clips.length} clips, ${windows} windows (150 frames on every clip, also 600 where the source allows it; AOM-CTC: the CTC's 130). `,
    view ? `On this machine's report: ${view.view_clips} windows in the view, ${view.common_clips} in the common set (every family but the +drop rows has well-formed curves there). ` : 'No report on this machine yet. ',
    `Rungs: the shared ${data.ladder_kbps.join(' / ')} kbps ladder at 720p30, scaled to each clip and capped at half its source rate. Manifest: `,
    el('code', {}, d.manifest), '.');
  root.querySelector('.clip-table').replaceChildren(table([
    { label: 'Clip', cell: (c) => c.name },
    { label: 'Class', cell: (c) => c.class },
    { label: 'Size', cell: (c) => `${c.width}×${c.height}` },
    { label: 'fps', num: true, cell: (c) => String(c.fps) },
    { label: 'Windows (frames)', num: true, cell: (c) => c.windows.join(' + ') },
    { label: 'Source kbps', num: true, cell: (c) => c.source_kbps.toLocaleString('en') },
    { label: 'Rungs (kbps)', num: true, cell: (c) => c.rungs.join(' / ') },
    { label: 'Malformed curves (hover for the families)', cell: (c) => lostCell(data, view, c) },
  ], d.clips, (c) => (view && c.windows.every((w) => losses(data, view, c, w).out) ? 'out' : '')));
}

async function main() {
  const root = document.getElementById('clips');
  try {
    const [data, bp3] = await Promise.all([loadData('clips'), loadData('bp3')]);
    if (!(state.dataset in data.datasets)) [state.dataset] = Object.keys(data.datasets);
    let machine = chosenMachine(bp3.machines);
    const show = () => { banner(bp3, machine); render(data, machine, root); };
    const controls = root.querySelector('.controls');
    controls.append(control('Dataset', segmented('Dataset',
      Object.entries(data.datasets).map(([value, d]) => ({ value, label: d.name })), state.dataset,
      (v) => {
        state.dataset = v;
        const url = new URL(location.href);
        url.searchParams.set(DATASET_PARAM, v);
        history.replaceState(null, '', url);
        show();
      })));
    machinePicker(bp3.machines, machine, (m) => { machine = m; show(); });
    show();
  } catch (error) {
    fail(root, error);
  }
}

main();
