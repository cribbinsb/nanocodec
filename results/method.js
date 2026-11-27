// How each view on the chosen timing machine was measured (method.html): the provenance each benchmark report carries
// (procedure commits, phase, series, clip counts, zero lines, envelope drops, refused timing
// windows, patches, exceptions, binaries). Data: data/bp3.json (tools/bench/preview.py).
import { el, fail, loadData, table } from './common.js';
import { SHORT, allChoices, banner, chosenMachine, machinePicker, statement, viewTitle, viewsOn, zeroName } from './bp3.js';

const SHA_SHOWN = 16;

function section(data, view) {
  const dropped = Object.values(view.values).reduce((n, v) => n + Object.values(v.envelope_dropped).reduce((a, b) => a + b, 0), 0);
  const refused = Object.entries(view.refused_windows);
  const twins = Object.entries(view.same_streams);
  const rows = [
    ['Report', el('code', {}, view.source)],
    ['Procedure', `${view.procedure} (the report's record); run manifests at commits ${view.procedure_commits.map((c) => c.slice(0, 9)).join(', ')}`],
    ['Timing machine', `${data.machines[view.machine].about} (series ${view.timing_series})`],
    ['Quality series', view.quality_series === view.timing_series ? `${view.quality_series} (the timed streams)` : `${view.quality_series}; the timed streams equal them byte for byte`],
    ['Phase', `${view.phase_name} (${Object.entries(view.phase).map(([k, v]) => `${k} ${v}`).join(', ')})`],
    ['nanocodec build', view.build.nanocodec ? `commit ${view.build.nanocodec.commit} (${view.build.nanocodec.date || 'date not recorded'}); ${view.build.nanocodec.version}; ${view.build.nanocodec.build}; SHA-256 ${view.build.nanocodec.sha256.slice(0, SHA_SHOWN)}` : el('strong', { class: 'not-recorded' }, 'not recorded')],
    ...view.build.references.map((r) => [`${r.name} build`, r.version ? `${r.version} (${r.commit}); LTO: ${r.lto}; PGO: ${r.pgo}` : el('strong', { class: 'not-recorded' }, 'not recorded')]),
    ['Build record', el('code', {}, view.build.record)],
    ['Clips', `${view.view_clips} in the view; ${view.common_clips} in the common set`],
    ['Zero line', data.metrics.map((m) => `${SHORT[m]}: ${zeroName(view, m)}`).join('; ')],
    ['Monotone envelope', view.envelope ? `on; ${dropped} points dropped over all families and metrics` : 'off'],
    ['Speed', view.speed_provisional ? 'provisional (a timing check failed)' : 'every timing check passed for the times used'],
    ['Refused timing windows (coded again)', String(refused.length)],
    ['Families on the report’s front', [...new Set(view.front.map((f) => data.points[view.same_streams[f] || f].label))].join('; ')],
    twins.length ? ['Twins shown once', twins.map(([twin, kept]) => `${twin} as ${kept}`).join('; ')] : null,
    ...view.patches.map((p) => ['Patch before scoring', `${p.file} at ${p.commit}: ${p.why}`]),
    ...view.exceptions.map((e) => ['Exception', `${e.rule}: ${e.value} (${e.why})${e.withdrawn ? `; withdrawn ${e.withdrawn}` : ''}`]),
  ].filter(Boolean);
  return el('section', {},
    el('h2', { id: view.id }, viewTitle(data, view)),
    el('p', { class: 'note measured' }, statement(data, view)),
    el('div', { class: 'detail' }, el('dl', {}, rows.flatMap(([k, v]) => [el('dt', {}, k), el('dd', {}, v)]))),
    el('details', {}, el('summary', {}, 'Refused timing windows'),
      el('ul', { class: 'note' }, refused.map(([where, why]) => el('li', {}, `${where}: ${why}`)))),
    el('details', {}, el('summary', {}, 'Binaries (SHA-256, first characters)'),
      table([
        { label: 'Program', cell: ([name]) => name },
        { label: 'SHA-256', cell: ([, sha]) => el('code', {}, sha.slice(0, SHA_SHOWN)) },
      ], Object.entries(view.binaries))));
}

async function main() {
  const root = document.getElementById('method');
  try {
    const data = await loadData('bp3');
    const show = (machine) => {
      banner(data, machine);
      const views = viewsOn(data, machine);
      root.replaceChildren(...(views.length ? views.map((v) => section(data, v)) : [el('p', { class: 'note' }, 'No report found on this machine yet.')]));
    };
    const machine = chosenMachine(data.machines, data.views);
    machinePicker(data.machines, machine, show, allChoices(data));
    show(machine);
  } catch (error) {
    fail(root, error);
  }
}

main();
