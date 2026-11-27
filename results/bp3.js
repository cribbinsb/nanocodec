// Shared parts of the benchmark results pages (tools/bench/pages, assembled by
// tools/bench/preview.py): the timing machine selector every page carries (kept in the URL),
// the statement under every view (procedure, machine, dataset and clip counts, zero line,
// phase), view lookup and number formats. Every figure comes from data/bp3.json,
// which is the benchmark reports' own numbers (tools/bench/report.py).
import { METRIC_NAMES, METRIC_SHORT, REPO, control, el, pct, segmented } from './common.js';

// The report's metric keys under the site's names.
export const NAMES = { j: METRIC_NAMES.j, psnr_yuv: METRIC_NAMES.psnr, vmaf_v1: METRIC_NAMES.vmaf, ms_ssim: METRIC_NAMES.ms_ssim };
export const SHORT = { j: METRIC_SHORT.j, psnr_yuv: METRIC_SHORT.psnr, vmaf_v1: METRIC_SHORT.vmaf, ms_ssim: METRIC_SHORT.ms_ssim };
// The zero line's tunes (procedure section 9), by the report's family names.
const ZERO_TUNES = {
  'x264-medium': 'x264 medium, default tune (--tune zerolatency)',
  'x264-medium~psnr': 'x264 medium --tune zerolatency,psnr',
  'x264-medium~ssim': 'x264 medium --tune zerolatency,ssim',
};
export const SPEED_UNIT = 'Mpx/CPU-s';
/** How a machine's speed figures were taken, as the charts' axes name them. */
export const speedName = (data, machine) => data.machines[machine].speed_name;
// The URL parameter that keeps the chosen timing machine across pages and reloads.
const MACHINE_PARAM = 'machine';

export const zeroName = (view, metric) => ZERO_TUNES[view.zero_line[metric]] || view.zero_line[metric];
/** The family plotted as a metric's zero line: the zero itself, or the twin it is shown as. */
export const zeroFamily = (view, metric) => view.same_streams[view.zero_line[metric]] || view.zero_line[metric];
export const viewOf = (data, dataset, machine) => data.views.find((v) => v.dataset === dataset && v.machine === machine);
export const viewsOn = (data, machine) => data.views.filter((v) => v.machine === machine);
export const viewTitle = (data, view) => data.datasets[view.dataset].name;

/** The machines that have at least one report (all of `machines` where views are not given). */
export const machinesWithViews = (machines, views) => Object.keys(machines)
  .filter((m) => !views || views.some((v) => v.machine === m));

/** The timing machine in the URL when it has a report (`views`), else the first that has one
 *  (x86). */
export function chosenMachine(machines, views = null) {
  const asked = new URLSearchParams(location.search).get(MACHINE_PARAM);
  const ready = machinesWithViews(machines, views);
  return ready.includes(asked) ? asked : ready[0] || Object.keys(machines)[0];
}

/** The machine selector's entries for one dataset: the machines it is measured on, those
 *  without a report yet shown but not selectable. */
export function datasetChoices(data, dataset) {
  const shown = data.datasets[dataset].machines;
  return { shown, ready: shown.filter((m) => viewOf(data, dataset, m)) };
}

/** The machine selector's entries on a page that shows every dataset: every machine, those
 *  without any report yet shown but not selectable. */
export const allChoices = (data) => ({ shown: Object.keys(data.machines), ready: machinesWithViews(data.machines, data.views) });

/** A dataset's view on `machine`, else on the first machine that has it (the machine to switch
 *  to when the reader picks a dataset the current machine does not have). */
export function machineFor(data, dataset, machine) {
  return viewOf(data, dataset, machine) ? machine : datasetChoices(data, dataset).ready[0];
}

/** Keep `machine` in this page's URL and in the links to the other results pages. */
function keepMachine(machine) {
  const url = new URL(location.href);
  url.searchParams.set(MACHINE_PARAM, machine);
  history.replaceState(null, '', url);
  for (const link of document.querySelectorAll('.results-nav a')) {
    const target = new URL(link.getAttribute('href'), location.href);
    target.searchParams.set(MACHINE_PARAM, machine);
    link.setAttribute('href', `${target.pathname.split('/').pop() || './'}${target.search}`);
  }
}

/** The timing machine selector, first among the page's other controls (the results bar on a
 *  page that has none); onChange(machine) re-renders the page. A page that rebuilds its
 *  controls calls this again after them. `choices` ({shown, ready}, datasetChoices or
 *  allChoices): the machines listed and those selectable (the rest are marked "not yet");
 *  every machine by default. */
export function machinePicker(machines, machine, onChange, choices = null) {
  keepMachine(machine);
  const shown = choices ? choices.shown : Object.keys(machines);
  const ready = choices ? choices.ready : shown;
  const host = document.querySelector('.controls') || document.querySelector('.machine-picker');
  host.querySelector('.machine-control')?.remove();
  const node = control('Timing machine', segmented('Timing machine',
    shown.map((value) => {
      const m = machines[value];
      return ready.includes(value) ? { value, label: m.label, title: m.name }
        : { value, label: `${m.label} (not yet)`, title: `${m.name}: no report yet; shown when it arrives`, disabled: true };
    }), machine,
    (v) => { keepMachine(v); onChange(v); }));
  node.classList.add('machine-control');
  host.prepend(node);
}

/** Megapixels per CPU-second: one decimal below 100, whole numbers above. */
export function speed(value) {
  if (value === null || value === undefined) return '–';
  return value >= 100 ? value.toFixed(0) : value.toFixed(1);
}

/** A BD's 95 % bootstrap interval over clips. */
export function interval(bd) {
  return bd && bd.low !== null && bd.low !== undefined ? `${pct(bd.low)} to ${pct(bd.high)}` : '–';
}

export const mean = (bd) => (bd ? bd.mean : null);

/** "(n clips)" of a family's headline figure; a +drop row averages its own clips. */
export const clipsNote = (v) => `${v.clips} clips${v.own_only ? ', its own (+drop row)' : ''}`;

const notRecorded = () => el('strong', { class: 'not-recorded' }, 'not recorded');

/** One build identity: our source commit, its date, version and build type, and each
 * reference encoder's version, as the report's provenance checked them. */
function buildLines(build, views) {
  const own = build.nanocodec;
  return el('div', { class: 'build-identity' },
    el('div', {}, el('strong', {}, 'nanocodec build: '),
      own ? [`source commit ${own.commit}, `, own.date || ['commit date ', notRecorded()], `; ${own.version}; build type: ${own.build}`] : notRecorded(),
      el('span', { class: 'dim' }, ` (${views})`)),
    el('div', {}, el('strong', {}, 'Reference encoders: '), build.references.flatMap((r, i) => [
      i ? '; ' : '', `${r.name} `, r.version ? `${r.version} (${r.commit})` : notRecorded()]),
      '. Each is built with LTO and PGO where its build allows (the Method page lists them).'));
}

/** The builds one machine's results are for: one block where every view used the same
 * binaries, else one per group of views. */
function buildIdentity(data, views) {
  const groups = new Map();
  for (const view of views) {
    const key = JSON.stringify(view.build);
    groups.set(key, [...(groups.get(key) || []), view]);
  }
  return [...groups.values()].map((group) => buildLines(group[0].build,
    groups.size === 1 ? 'every dataset' : group.map((v) => viewTitle(data, v)).join('; ')));
}

/** The procedure versions one machine's reports were made at. */
const proceduresOf = (views) => [...new Set(views.map((v) => v.procedure))].join(', ') || 'none';

/** The banner for one machine: the procedure, the machine and the reports not there yet. */
export function banner(data, machine) {
  const views = viewsOn(data, machine);
  const missing = data.missing.filter((m) => m.machine === machine).map((m) => data.datasets[m.dataset].name);
  const absent = Object.entries(data.datasets).filter(([, d]) => !d.machines.includes(machine)).map(([, d]) => d.name);
  document.querySelector('.results-banner').replaceChildren(
    `Benchmark procedure ${proceduresOf(views)} (the reports' own record${proceduresOf(views) === data.current_procedure ? '' : `; the procedure is now at ${data.current_procedure}`}); CBR; every figure is from the procedure's own reports, timed on ${data.machines[machine].name}. `,
    missing.length ? `No report found on this machine yet for (not shown): ${missing.join('; ')}. ` : '',
    absent.length ? `Not measured on this machine: ${absent.join('; ')}.` : '',
    el('div', { class: 'machine-conditions' }, el('strong', {}, 'How this machine is timed: '),
      `${data.machines[machine].about}. Each machine is timed under its own conditions, so speeds compare between encoders on one machine, never across machines.`),
    ...buildIdentity(data, views));
}

/** A dataset's description, with a link to its analysis where it has one. */
export function aboutDataset(d) {
  return d.doc ? [d.about, ' (', el('a', { href: REPO + d.doc }, 'how it was chosen'), ')'] : [d.about];
}

/** What a view measured, on what, against what: the statement under every chart and table
 *  (text and links, for el() or replaceChildren(...)). */
export function statement(data, view) {
  const d = data.datasets[view.dataset];
  const zeros = `BD-J and BD-VMAF v1 against ${zeroName(view, 'j')}, BD-PSNR-YUV against ${zeroName(view, 'psnr_yuv')}, BD-MS-SSIM against ${zeroName(view, 'ms_ssim')}`;
  const quality = view.quality_series === view.timing_series ? ''
    : ` Quality is scored from the ${view.quality_series} series' streams, which the timed streams equal byte for byte.`;
  return [`Procedure ${view.procedure}, CBR, ${d.name}: `, ...aboutDataset(d), '. '
    + (view.common_clips > view.distinct_clips
      ? `View: ${view.view_clips} clip-windows (each clip's 150-frame window, and its 600-frame window where the source allows) where the zero line's curves are well formed; common set: ${view.common_clips} of them, on ${view.distinct_clips} clips, where every family but the +drop rows has well-formed curves; each clip counts once, its windows averaged. `
      : `View: ${view.view_clips} clips where the zero line's curves are well formed; common set: ${view.common_clips} of them, where every family but the +drop rows has well-formed curves. `)
    + (Object.keys(view.excluded_clips || {}).length ? `Left out of every family: the ${Object.keys(view.excluded_clips).length} clips a tuning set of ours used (in-sample). ` : '')
    + 'Every BD is over the common set (a +drop row\'s over its own clips); each family\'s own anchor-valid clip count is in the table. '
    + `Zero line: ${zeros}. Speed: ${speedName(data, view.machine)}, megapixels per CPU-second, geometric mean over clips, on ${data.machines[view.machine].about} (series ${view.timing_series}).${quality} `
    + `Phase: ${view.phase_name}.`];
}
