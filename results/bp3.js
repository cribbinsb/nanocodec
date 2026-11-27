// Shared parts of the benchmark results pages (tools/bench/pages, assembled by
// tools/bench/preview.py): the timing machine selector every page carries (kept in the URL),
// the statement under every view (procedure, machine, dataset and clip counts, zero line,
// phase), view lookup and number formats. Every figure comes from data/bp3.json,
// which is the benchmark reports' own numbers (tools/bench/report.py).
import { METRIC_NAMES, METRIC_SHORT, control, el, pct, segmented } from './common.js';

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
export const SPEED_NAME = 'loaded speed (half load)';
// The URL parameter that keeps the chosen timing machine across pages and reloads.
const MACHINE_PARAM = 'machine';

export const zeroName = (view, metric) => ZERO_TUNES[view.zero_line[metric]] || view.zero_line[metric];
/** The family plotted as a metric's zero line: the zero itself, or the twin it is shown as. */
export const zeroFamily = (view, metric) => view.same_streams[view.zero_line[metric]] || view.zero_line[metric];
export const viewOf = (data, dataset, machine) => data.views.find((v) => v.dataset === dataset && v.machine === machine);
export const viewsOn = (data, machine) => data.views.filter((v) => v.machine === machine);
export const viewTitle = (data, view) => data.datasets[view.dataset].name;

/** The timing machine in the URL, else the first of `machines` (x86). */
export function chosenMachine(machines) {
  const asked = new URLSearchParams(location.search).get(MACHINE_PARAM);
  return asked in machines ? asked : Object.keys(machines)[0];
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
 *  controls calls this again after them. */
export function machinePicker(machines, machine, onChange) {
  keepMachine(machine);
  const host = document.querySelector('.controls') || document.querySelector('.machine-picker');
  host.querySelector('.machine-control')?.remove();
  const node = control('Timing machine', segmented('Timing machine',
    Object.entries(machines).map(([value, m]) => ({ value, label: m.label, title: m.name })), machine,
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
  document.querySelector('.results-banner').replaceChildren(
    `Benchmark procedure ${proceduresOf(views)} (the reports' own record${proceduresOf(views) === data.current_procedure ? '' : `; the procedure is now at ${data.current_procedure}`}); CBR; every figure is from the procedure's own reports, timed on ${data.machines[machine].name}. `,
    missing.length ? `No report found on this machine for (not shown): ${missing.join('; ')}.` : '',
    ...buildIdentity(data, views));
}

/** What a view measured, on what, against what: the statement under every chart and table. */
export function statement(data, view) {
  const d = data.datasets[view.dataset];
  const zeros = `BD-J and BD-VMAF v1 against ${zeroName(view, 'j')}, BD-PSNR-YUV against ${zeroName(view, 'psnr_yuv')}, BD-MS-SSIM against ${zeroName(view, 'ms_ssim')}`;
  const quality = view.quality_series === view.timing_series ? ''
    : ` Quality is scored from the ${view.quality_series} series' streams, which the timed streams equal byte for byte.`;
  return `Procedure ${view.procedure}, CBR, ${d.name}: ${d.about}. `
    + `View: ${view.view_clips} clips where the zero line's curves are well formed; common set: ${view.common_clips} of them, where every family but the +drop rows has well-formed curves. Every BD is over the common set (a +drop row's over its own clips); each family's own anchor-valid clip count is in the table. `
    + `Zero line: ${zeros}. Speed: ${SPEED_NAME}, megapixels per CPU-second, geometric mean over clips, on ${data.machines[view.machine].about} (series ${view.timing_series}).${quality} `
    + `Phase: ${view.phase_name}.`;
}
