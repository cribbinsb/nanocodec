// Rate compliance (compliance.html): for each dataset on the chosen timing machine, every encoder's figures from the one
// external bucket replay (procedure section 8), and the exclusion check of each family that
// lost clips (section 9). Data: data/bp3.json (tools/bench/preview.py).
import { el, fail, loadData, pct, table } from './common.js';
import { banner, chosenMachine, machinePicker, statement, viewTitle, viewsOn } from './bp3.js';

const share = (value) => `${(100 * value).toFixed(2)}%`;
// The bucket replay's two measures (report.py): drained at the stream's own achieved rate (the
// fair one: an encoder cannot gain by undershooting) and at the target rate.
const peak = (measure, key) => (measure ? measure[key].toFixed(2) : '–');
const over = (measure) => (measure ? share(measure.share_points_over_1) : '–');
const ratio = (value) => (value === null || value === undefined ? '–' : value.toFixed(3));

function section(data, view) {
  const rows = Object.entries(view.values).filter(([, v]) => v.compliance)
    .map(([family, v]) => ({ family, v, c: v.compliance, x: v.exclusion_check, p: data.points[family] }))
    .sort((a, b) => (b.p.ours - a.p.ours) || a.p.label.localeCompare(b.p.label));
  const cls = (r) => (r.p.ours ? 'ours' : '');
  const name = { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.p.ours ? 'ours' : 'ref'}` }), r.p.label] };
  return el('section', {},
    el('h2', { id: view.id }, viewTitle(data, view)),
    el('p', { class: 'note measured' }, statement(data, view)),
    table([
      name,
      { label: 'Points', num: true, cell: (r) => String(r.c.points) },
      { label: 'Frames', num: true, cell: (r) => String(r.c.frames) },
      { label: 'Own rate: 95th-percentile peak (buffers)', num: true, cell: (r) => peak(r.c.own_rate, 'p95_peak_buffers') },
      { label: 'Own rate: worst peak', num: true, cell: (r) => peak(r.c.own_rate, 'worst_peak_buffers') },
      { label: 'Own rate: points over 1.0', num: true, cell: (r) => over(r.c.own_rate) },
      { label: 'Target rate: worst peak', num: true, cell: (r) => peak(r.c.target_rate, 'worst_peak_buffers') },
      { label: 'Target rate: points over 1.0', num: true, cell: (r) => over(r.c.target_rate) },
      { label: 'First key, median (s of own rate)', num: true, cell: (r) => (r.c.first_key_seconds ? r.c.first_key_seconds.median.toFixed(2) : '–') },
      { label: 'Rate over target', num: true, cell: (r) => ratio(r.c.mean_rate_over_target) },
    ], rows, cls),
    el('h3', {}, 'Exclusion check: the families that lost clips'),
    el('p', { class: 'note' }, 'For each family with a malformed curve on some clip: its rate over target on the lost and on the kept clips, and our BD-J on those clips (as the report prints it).'),
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
    const show = (machine) => {
      banner(data, machine);
      const views = viewsOn(data, machine);
      root.replaceChildren(...(views.length ? views.map((v) => section(data, v)) : [el('p', { class: 'note' }, 'No report found on this machine yet.')]));
    };
    machinePicker(data.machines, chosenMachine(data.machines), show);
    show(chosenMachine(data.machines));
  } catch (error) {
    fail(root, error);
  }
}

main();
