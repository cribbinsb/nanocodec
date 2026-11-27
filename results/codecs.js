// Results by codec (codecs.html): for each format, our presets and the reference encoders of
// that format in every dataset reported on the chosen timing machine, on the common set, for
// the chosen metric. Data: data/bp3.json (tools/bench/preview.py).
import { CODEC_NAMES, control, el, fail, loadData, pct, segmented, table } from './common.js';
import { NAMES, SHORT, SPEED_UNIT, banner, chosenMachine, machinePicker, mean, speed, statement, viewTitle, viewsOn } from './bp3.js';

const state = { metric: 'j', machine: null };

/** A family's figures in a view; a twin the report measured identically has its kept row's. */
const valuesOf = (view, family) => view.values[view.same_streams[family] || family];

/** "BD (clips)" of a family in a view, or a dash where it has none; a +drop row's clips are
 * its own. */
function cell(view, family) {
  const v = valuesOf(view, family);
  const value = v ? mean(v.bd[state.metric]) : null;
  return value === null ? '–' : `${pct(value)} (${v.clips}${v.own_only ? ', own' : ''})`;
}

function render(data, root) {
  const views = viewsOn(data, state.machine);
  const speedOf = (f) => views.map((v) => v.values[f]?.speed).find(Boolean) ?? 0;
  const out = data.codec_order.map((codec) => {
    const families = Object.keys(data.points).filter((f) => data.points[f].codec === codec && views.some((v) => valuesOf(v, f)))
      .sort((a, b) => (data.points[b].ours - data.points[a].ours) || speedOf(a) - speedOf(b));
    if (!families.length) return null;
    return el('section', {},
      el('h2', { id: codec }, CODEC_NAMES[codec]),
      table([
        { label: 'Encoder', cell: (f) => [el('span', { class: `swatch ${data.points[f].ours ? 'ours' : 'ref'}` }), data.points[f].label] },
        ...views.flatMap((view) => [
          { label: `${viewTitle(data, view)}: BD (clips)`, num: true, cell: (f) => cell(view, f) },
          { label: 'own clips', num: true, cell: (f) => String(valuesOf(view, f)?.own_clips ?? '–') },
          { label: SPEED_UNIT, num: true, cell: (f) => speed(valuesOf(view, f)?.speed) },
        ]),
      ], families, (f) => (data.points[f].ours ? 'ours' : '')));
  });
  root.querySelector('.codec-tables').replaceChildren(
    el('p', { class: 'note' }, `${NAMES[state.metric]} on ${data.machines[state.machine].name}; in brackets, the clips averaged: the common set (every family over the same clips), or a +drop row's own clips. Each dataset has its BD, each family's own anchor-valid clip count, then its loaded speed. A tune twin that a report measured identically to another row (the overview lists them) shows that row's figures.`),
    ...(views.length ? out.filter(Boolean) : [el('p', { class: 'note' }, 'No report found on this machine yet.')]),
    ...views.map((view) => el('p', { class: 'note measured' }, el('strong', {}, `${viewTitle(data, view)}. `), statement(data, view))));
}

async function main() {
  const root = document.getElementById('codecs');
  try {
    const data = await loadData('bp3');
    state.machine = chosenMachine(data.machines);
    machinePicker(data.machines, state.machine, (machine) => { state.machine = machine; banner(data, machine); render(data, root); });
    banner(data, state.machine);
    root.querySelector('.controls').append(control('Metric', segmented('Metric',
      data.metrics.map((m) => ({ value: m, label: SHORT[m] })), state.metric, (v) => { state.metric = v; render(data, root); })));
    render(data, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
