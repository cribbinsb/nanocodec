// The results explorer (index.html): bitrate saving vs encode speed for every
// encoder and preset, with the clip set, rate mode, metric, platform,
// codecs and presets selectable, a legend that toggles projects, a tooltip per
// point, a detail panel with the exact command line, and the same view as a table.
// A link ?codec=&preset= (deep-link.js) opens it filtered to that preset, selected;
// a selected point links to the demo with its settings (or its encoder's stream).
// Data: data/chart.json (python3 -m nceval.site, from docs/results).
import { paretoFront, frontierLine } from './frontier.js';
import { drawScatter, pinSwatch, swatch } from './chart.js';
import { chartFamily, demoQuery, manifestEncoder, parseResultsQuery } from '../deep-link.js';
import {
  CODEC_NAMES, METRIC_NAMES, METRIC_SHORT, NOT_MEASURED, ZERO_NAMES, control, el, fail, fps, loadData, pct,
  segmented, sources, table,
} from './common.js';

const SHAPES = { nanocodec: 'circle', x264: 'square', x265: 'diamond', 'SVT-AV1': 'triangle', openh264: 'triangle-down', libvpx: 'cross', libaom: 'plus' };
// The speed machines, in the data's order (site.PLATFORMS): what one core is, and its short name.
const PLATFORM_LABELS = { jetson: 'Jetson Orin Nano core', x86: 'AMD EPYC 9R45 (Zen 5) core', zen3: 'AMD EPYC 7R13 (Zen 3, no AVX-512) core', graviton: 'AWS Graviton4 (Neoverse V2) core' };
const PLATFORM_SHORT = { jetson: 'Jetson', x86: 'Zen 5', zen3: 'Zen 3', graviton: 'Graviton4' };
const PLATFORMS = Object.keys(PLATFORM_LABELS);
// The fixed-QP chart's extra metric (the standards' weighted PSNR); the rest are common.js's.
// XPSNR is recorded beside J (never part of it) where a view's points have it.
const NAMES = { ...METRIC_NAMES, weighted: 'BD-rate, weighted PSNR (0.92 Y + 0.04 Cb + 0.04 Cr)', xpsnr: 'BD-XPSNR-YUV (not part of J)' };
const SHORT = { ...METRIC_SHORT, weighted: 'Weighted', xpsnr: 'XPSNR' };
const TUNED_ZERO = 'x264-medium~psnr';
const ALL = 'all';
const PIN_NOTE = 'points beyond the axes are pinned to the edge (arrow)';
// The demo's pre-encoded streams (for a reference point's demo link), loaded on first use.
const MANIFEST_URL = '../streams-ref/manifest.json';
let manifest = null;

// The Clip set control offers each mode its own sets, as data/chart.json has views for them:
// the Fixed QP chart's (CTC: the standards lens; the others samples of those CBR sets) and the
// CBR chart's.
const qpViewOf = (data, clipset) => data.views.find((v) => v.rate_mode === 'qp' && v.clipset === clipset)?.id;
const hasCbrView = (data, clipset) => data.views.some((v) => v.rate_mode === 'cbr' && v.clipset === clipset);
// How each Fixed QP sample was drawn from its clip set (an outside dataset's: data.datasets).
const QP_SAMPLE_RULES = {
  fresh: 'the held-out, fresh set (videos none of our tuning used; the clips are picked by a fixed rule, content classes in turn and within a class by a hash of the name)',
};
const state = {
  clipset: 'fresh', qpClipset: 'ctc', zero: TUNED_ZERO, mode: 'cbr', metric: 'j', platform: 'jetson', frontier: true,
  preset: ALL, codecs: new Set(Object.keys(CODEC_NAMES)), hiddenOwners: new Set(), tuned: false,
  selected: null,
};

function viewId(data) {
  if (state.mode === 'qp') return qpViewOf(data, state.qpClipset);
  const zid = state.zero === TUNED_ZERO ? 'psnr' : 'default';
  return `${state.clipset}-${zid}`;
}

function versionOf(data, view, family) {
  const point = data.points[family];
  if (point.ours) {
    if (view.build_commit) return `build ${view.measured.labels[view.build_commit.split(' ')[0]]}`;
    const build = view.family_builds?.[family];
    return build ? `build ${view.builds[build] || build}` : '';
  }
  const tools = view.tools || data.views.find((v) => v.tools)?.tools || {};
  return point.version_key && tools[point.version_key] ? tools[point.version_key] : '';
}

function visible(data, view) {
  return Object.entries(view.values).filter(([family, v]) => {
    const p = data.points[family];
    if (!p || v[state.metric] === null || v[state.metric] === undefined || !v.fps?.[state.platform]) return false;
    if (state.hiddenOwners.has(p.owner)) return false;
    if (p.codec && !state.codecs.has(p.codec)) return false;
    if (p.ours && state.preset !== ALL && p.preset !== state.preset) return false;
    if (!p.ours && p.tune && family !== view.zero && !state.tuned) return false;
    return true;
  });
}

function describe(data, view, family, v) {
  const p = data.points[family];
  const version = versionOf(data, view, family);
  return el('div', {},
    el('strong', {}, p.label), version ? el('span', { class: 'dim' }, ` · ${version}`) : null,
    el('div', {}, el('span', { class: 'value' }, pct(v[state.metric])), ` ${SHORT[state.metric]} vs ${ZERO_NAMES[view.zero]}`),
    v.iqr && v.iqr[0] !== null && family !== view.zero
      ? el('div', { class: 'dim' }, `per-clip interquartile range ${pct(v.iqr[0])} to ${pct(v.iqr[1])}`) : null,
    el('div', {}, PLATFORMS.filter((m) => v.fps[m]).map((m) => `${fps(v.fps[m])} fps ${PLATFORM_SHORT[m]} core`).join(' · ')),
    v.clips_scored ? el('div', { class: 'dim' }, `${v.clips_scored} clips with a reliable BD`
      + (state.metric === 'xpsnr' && xpsnrClips(v) ? ` (XPSNR ${xpsnrClips(v)})` : '')) : null,
    el('div', { class: 'dim' }, 'Select the point for its command line'));
}

/** A metric's BD-rate, or why there is none in a view that records the metric. */
function metricText(value) {
  return value === null || value === undefined ? NOT_MEASURED : pct(value);
}

/** XPSNR may average fewer clips than J (where its curves overlap the zero line's):
 * 'over 135/138 clips' then, so it is never silently another clip set. */
function xpsnrClips(v) {
  return v.xpsnr_clips && v.clips_scored && v.xpsnr_clips !== v.clips_scored
    ? `over ${v.xpsnr_clips}/${v.clips_scored} clips` : '';
}

/** metricText, with XPSNR's clip count when it is not J's. */
function metricCell(v, m) {
  const text = metricText(v[m]);
  return m === 'xpsnr' && v[m] !== null && v[m] !== undefined && xpsnrClips(v) ? `${text} (${xpsnrClips(v)})` : text;
}

function detail(data, view, family) {
  const p = data.points[family];
  const v = view.values[family];
  const rows = [
    ['Encoder', `${p.owner}${versionOf(data, view, family) ? `, ${versionOf(data, view, family)}` : ''}`],
    ['Codec', CODEC_NAMES[p.codec] || p.codec],
    ['Preset', p.preset + (p.tune ? `, --tune ${p.tune}` : '')],
    ...view.metrics.map((m) => [NAMES[m], metricCell(v, m)]),
    v.iqr && v.iqr[0] !== null && family !== view.zero ? ['Per-clip spread (J)', `${pct(v.iqr[0])} to ${pct(v.iqr[1])} (interquartile)`] : null,
    v.clips_scored ? ['Clips scored', String(v.clips_scored)] : null,
    v.rate_invalid_points ? ['Rate-invalid points', `${v.rate_invalid_points} (kept on their curves)`] : null,
    v.chained ? ['Chained clips', `${v.chained} (through one intermediate encoder)`] : null,
    ['Encode speed', PLATFORMS.filter((m) => v.fps[m]).map((m) => `${fps(v.fps[m])} fps on one ${PLATFORM_LABELS[m]}`).join(', ') || 'not timed'],
  ].filter(Boolean);
  const commands = [];
  if (state.mode === 'cbr' && p.cbr_command) {
    commands.push(el('p', { class: 'note' }, `CBR command at the ${data.example.kbps} kbps rung of a ${data.example.width}×${data.example.height} clip at ${data.example.fps} fps (the ladder scales the rate per clip):`),
      el('code', { class: 'command' }, p.cbr_command));
  }
  if (p.timing) {
    commands.push(el('p', { class: 'note' }, `Timing command (fixed QP chosen for about ${data.timing_rung_kbps} kbps, shown on ${p.timing.clip})${p.timing_is_twin ? ', the default-tuned twin: this variant is not timed itself' : ''}:`),
      el('code', { class: 'command' }, p.timing.command));
  }
  return el('section', { class: 'detail', 'aria-live': 'polite' },
    el('h3', {}, p.label),
    el('p', { class: 'demo-link-slot' }),
    el('dl', {}, rows.flatMap(([k, val]) => [el('dt', {}, k), el('dd', {}, val)])),
    commands);
}

/** The demo link of a point: our preset at the example rung, or a reference encoder's
 * pre-encoded stream (preferring the chart's size) beside nanocodec matched to it. */
async function demoLink(data, family) {
  const p = data.points[family];
  const ex = data.example;
  if (p.ours) {
    const preset = p.preset.toLowerCase();
    return { href: `../${demoQuery({ codec: p.codec, preset, width: ex.width, height: ex.height, fps: ex.fps, kbps: ex.kbps })}`,
      text: `Try ${p.label} in the demo (${ex.width}×${ex.height}, ${ex.fps} fps, ${ex.kbps} kbps)` };
  }
  const encoder = manifestEncoder(family);
  if (!encoder || !p.codec) return null;
  manifest ??= await fetch(MANIFEST_URL).then((r) => (r.ok ? r.json() : { streams: [] }), () => ({ streams: [] }));
  const streams = manifest.streams.filter((s) => s.encoder === encoder);
  const stream = streams.find((s) => s.chartConfiguration && s.width === ex.width) ?? streams[0];
  if (!stream) return null;
  return { href: `../${demoQuery({ codec: p.codec, stream: stream.id })}`,
    text: `See a ${stream.label} stream in the demo, beside nanocodec matched to it` };
}

function fillDemoLink(data, family, root) {
  demoLink(data, family).then((link) => {
    const slot = root.querySelector('.demo-link-slot');
    if (slot && link && state.selected === family) slot.replaceChildren(el('a', { href: link.href, class: 'demo-link' }, link.text));
  });
}

function viewNote(data, view) {
  if (view.rate_mode === 'qp' && view.clipset === 'ctc') {
    return `Bits: BD-rate against x264 medium at fixed QP on the AOM CTC low-delay sequences (${view.condition}, ${view.clips} sequences), ours with its fixed-QP profile (column "${view.column}"; all three columns on the Fixed QP page), ${view.measured.quality} ${qpSpeedNote(view)} ${view.measured.dated}`;
  }
  if (view.rate_mode === 'qp') {
    return `Bits: BD-rate against x264 medium at fixed QP on ${view.clips} clips of ${QP_SAMPLE_RULES[view.clipset] || data.datasets[view.clipset].sample_rule}, ${view.frames} frames each, one key frame, every encoder at ${view.points} fixed quantizers per clip spanning the CBR ladder's rates; J is the mean of the full and lower-half BD-rates over PSNR-YUV, VMAF v1 and MS-SSIM, with its 90% interval over clips. ${view.measured.quality} ${qpSpeedNote(view)} ${view.measured.dated}${datasetNote(data, view)}`;
  }
  const tools = view.tools || data.views.find((v) => v.tools).tools;
  const refs = `x264 ${tools.x264}, x265 ${tools.x265}, SVT-AV1 ${tools.svt}, libaom ${tools.libaom}, libvpx ${tools.libvpx}`;
  return `${data.clipsets[view.clipset]}: ${view.clips} clips, ${data.frames} frames each, one key frame, CBR ladder ${data.ladder_kbps.join('/')} kbps at 720p30 scaled to each clip (plus ${data.low_ladder_kbps.join('/')} kbps where curves need them), ${data.buffer_ms} ms buffer. Ours: ${oursBuilds(data, view)}; references ${refs}. ${cbrSpeedNote(data)} ${view.measured.dated}${datasetNote(data, view)}`;
}

/** An outside dataset's selection, credit and licence note (data.datasets), for its views. */
function datasetNote(data, view) {
  const d = data.datasets?.[view.clipset];
  if (!d) return '';
  return ` Clips: ${d.clips} of ${d.candidates}, stratified by ${d.stratified_by.join(' × ')} (${d.strata} strata, each filled evenly). Data: ${d.credit}, ${d.url}; ${d.note}.`;
}

/** What the CBR charts' speed axis is on the selected machine: its timing record's platform,
 * and the build our encoders were timed at where one session timed them all (Zen 3, the Graviton). */
function cbrSpeedNote(data) {
  const platform = data.platforms[state.platform];
  if (!platform) {
    return `Speed: not timed on the ${PLATFORM_LABELS[state.platform].replace(' core', '')} for this chart; choose another machine under Speed on.`;
  }
  const build = data.timing_builds?.[state.platform];
  return `Speed: ${platform}${build ? `; ours timed at build ${build}` : ''}.`;
}

/** What the Fixed QP chart's speed axis is on the selected machine (view.speed, from the
 * lens timing record), or that this machine has no such timing. */
function qpSpeedNote(view) {
  const speed = view.speed?.[state.platform];
  if (!speed) {
    return `Speed: not timed on the ${PLATFORM_LABELS[state.platform].replace(' core', '')} for this chart; choose another machine under Speed on.`;
  }
  const what = view.clipset === 'ctc' ? 'sequences' : 'clips';
  const quantizers = speed.quantizers === view.points ? 'the same fixed quantizers' : `${speed.quantizers} of the same fixed quantizers`;
  return `Speed: single-thread encode fps on the same ${speed.sequences} ${what} at ${quantizers} (each encoder's own encodes of the quality run, ${speed.frames} frames each): CPU time on one pinned core of ${speed.platform}, ours at build ${speed.build}; fps is 1000 over the geometric mean of the ${what}' milliseconds per frame (${view.size_range.includes(' to ') ? `they run from ${view.size_range}, so the fps is not a 720p figure` : `all ${view.size_range}`}).`;
}

/** Which build each of our families on a view was measured at, grouped by build (a
 * chart's points come from several builds: each family is re-measured when it changes). */
function oursBuilds(data, view) {
  const byBuild = new Map();
  for (const [family, build] of Object.entries(view.family_builds)) {
    if (!family.startsWith('nanocodec-')) continue;
    const label = (data.points[family]?.label || family).replace(/^nanocodec /, '');
    byBuild.set(build, [...(byBuild.get(build) || []), label]);
  }
  return [...byBuild.entries()].sort((a, b) => b[1].length - a[1].length)
    .map(([build, labels]) => `${labels.join(', ')} at ${view.measured.labels[build] || view.builds[build] || build}`).join('; ');
}

function render(data, root) {
  const view = data.views.find((v) => v.id === viewId(data));
  if (!view.metrics.includes(state.metric)) state.metric = 'j';
  // A Fixed QP sample view is timed on fewer machines than the CTC lens (VC-1K and
  // fresh: x86 only): show it on a machine it has, and disable the others.
  const timed = view.rate_mode === 'qp' ? Object.keys(view.speed || {}) : null;
  if (timed?.length && !timed.includes(state.platform)) state.platform = timed[0];
  for (const b of root.querySelectorAll('[data-control="platform"] button')) {
    b.setAttribute('aria-pressed', String(b.dataset.value === state.platform));
    b.disabled = Boolean(timed?.length) && !timed.includes(b.dataset.value);
  }
  const clipset = state.mode === 'qp' ? state.qpClipset : state.clipset;
  for (const [name, value] of [['clipset', clipset], ['metric', state.metric]]) {
    for (const b of root.querySelectorAll(`[data-control="${name}"] button`)) {
      b.setAttribute('aria-pressed', String(b.dataset.value === value));
      if (name === 'clipset') {
        b.disabled = state.mode === 'qp'
          ? !qpViewOf(data, b.dataset.value)
          : !hasCbrView(data, b.dataset.value);
      }
      if (name === 'metric') {
        b.disabled = !view.metrics.includes(b.dataset.value);
        b.title = !b.disabled ? ''
          : b.dataset.value === 'xpsnr' ? 'XPSNR is not measured for this view yet'
            : 'Not in this run’s data (the held-out and development charts record J, PSNR-YUV and VMAF v1)';
      }
    }
  }
  const note = root.querySelector('.view-note');
  note.textContent = `${viewNote(data, view)} `;
  note.append(el('a', { href: `clips.html?set=${view.rate_mode === 'qp' && view.clipset !== 'ctc' ? view.id : view.clipset}` }, 'The clips.'));

  const entries = visible(data, view);
  const points = entries.map(([family, v]) => {
    const p = data.points[family];
    const base = family.split('~')[0];
    return {
      id: family, x: v.fps[state.platform], y: v[state.metric],
      iqr: state.metric === 'j' && v.iqr && v.iqr[0] !== null && family !== view.zero ? v.iqr : null,
      kind: p.ours ? 'ours' : (p.tune && family !== view.zero ? 'tuned' : 'ref'),
      shape: SHAPES[p.owner] || 'circle', label: p.label.replace(/^nanocodec /, '').replace(/ Medium \(default\)$/, ''),
      flagged: v.flagged, twin: p.tune && view.values[base] && family !== view.zero ? base : null,
      priority: (p.ours ? 2 : 1) + (p.preset === 'Medium' ? 1 : 0),
      aria: `${p.label}: ${pct(v[state.metric])}, ${fps(v.fps[state.platform])} fps`,
    };
  });
  const chart = root.querySelector('.chart');
  const title = root.querySelector('.chart-title');
  const zero = view.rate_mode === 'qp' ? 'x264 medium at fixed QP' : ZERO_NAMES[view.zero];
  title.textContent = `${NAMES[state.metric]} vs ${zero}, against single-thread encode speed`;
  root.querySelector('.chart-sub').textContent = view.rate_mode === 'qp'
    ? `Fixed QP · ${view.clipset === 'ctc' ? 'AOM CTC low delay' : `${view.clips} clips of ${data.clipsets[view.clipset].split(':')[0]}`}, build ${view.build_commit.split(' ')[0]} · lower is fewer bits, right is faster`
    : `Real-time CBR · ${data.clipsets[view.clipset].split(':')[0]} · lower is fewer bits, right is faster`;
  let pinned = [];
  if (!points.length) {
    chart.replaceChildren(el('p', { class: 'note' }, 'No points match these filters.'));
  } else {
    // Pareto frontiers of what is shown (filters, metric and machine applied; a
    // point not yet timed is not in `points`): ours, and the references.
    const frontiers = !state.frontier ? [] : ['ours', 'ref'].map((kind) => {
      const front = paretoFront(points.filter((p) => (kind === 'ours') === (p.kind === 'ours')));
      return { kind, line: frontierLine(front), ids: new Set(front.map((p) => p.id)) };
    });
    pinned = drawScatter(chart, points, {
      frontiers,
      xLabel: view.rate_mode === 'qp'
        ? `encode fps, ${view.clipset === 'ctc' ? 'AOM CTC sequences' : `${data.clipsets[view.clipset].split(':')[0]} clips`} (geometric mean), one ${PLATFORM_LABELS[state.platform]} (log scale)`
        : `encode fps, 720p, one ${PLATFORM_LABELS[state.platform]} (log scale)`,
      yLabel: `${SHORT[state.metric]} BD-rate (lower is better)`,
      ariaLabel: title.textContent,
      selected: state.selected,
      describe: (p) => describe(data, view, p.id, view.values[p.id]),
      onSelect: (p) => { state.selected = p.id; render(data, root); root.querySelector('.detail-slot').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); },
    }).pinned;
  }
  root.querySelector('.pin-key').hidden = !pinned.length;
  if (pinned.length) root.querySelector('.chart-sub').textContent += ` · ${PIN_NOTE}`;
  // Encoders with a value but no speed on this machine (never timed here): listed, not dropped.
  const untimed = Object.entries(view.values)
    .filter(([f, v]) => data.points[f] && !v.fps?.[state.platform] && v[state.metric] !== null)
    .map(([f, v]) => `${data.points[f].label} (${pct(v[state.metric])})`);
  const unmeasured = Object.entries(view.values)
    .filter(([f, v]) => data.points[f] && (v[state.metric] === null || v[state.metric] === undefined))
    .map(([f]) => data.points[f].label);
  if (state.metric === 'xpsnr' && unmeasured.length) {
    chart.append(el('p', { class: 'note' }, `XPSNR ${NOT_MEASURED} yet, so not plotted: ${unmeasured.join('; ')}.`));
  }
  if (untimed.length) {
    chart.append(el('p', { class: 'note off-scale' }, `Not yet timed, so not plotted: ${untimed.join('; ')}.`));
  }
  if (view.rate_mode === 'qp' && view.clipset === 'ctc') {
    chart.append(el('p', { class: 'note' }, el('a', { href: 'fixed-qp.html' }, 'All three fixed-QP columns, per encoder: the Fixed QP page.')));
  }
  const slot = root.querySelector('.detail-slot');
  const chosen = state.selected && view.values[state.selected];
  slot.replaceChildren(chosen ? detail(data, view, state.selected) : el('p', { class: 'note' }, 'Select a point (click, tap or Enter) for its versions, every metric, its exact command line and a link to try it in the demo.'));
  if (chosen) fillDemoLink(data, state.selected, root);

  const rows = entries.map(([family, v]) => ({ family, v, p: data.points[family] }))
    .sort((a, b) => a.v[state.metric] - b.v[state.metric]);
  root.querySelector('.table-slot').replaceChildren(table([
    { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.p.ours ? 'ours' : 'ref'}` }), r.p.label] },
    ...view.metrics.map((m) => ({ label: SHORT[m], num: true, cell: (r) => metricCell(r.v, m) })),
    { label: 'Clips', num: true, cell: (r) => (r.v.clips_scored ? String(r.v.clips_scored) : '–') },
    ...PLATFORMS.map((m) => ({ label: `fps ${PLATFORM_SHORT[m]}`, num: true, cell: (r) => fps(r.v.fps[m]) })),
  ], rows, (r) => (r.p.ours ? 'ours' : '') + (r.family === view.zero ? ' zero' : '')), sources(view.source));
}

function legend(data, root, owners) {
  const box = root.querySelector('.legend');
  box.replaceChildren(...owners.map((owner) => el('button', {
    type: 'button', 'aria-pressed': String(!state.hiddenOwners.has(owner)),
    onclick: (e) => {
      if (state.hiddenOwners.has(owner)) state.hiddenOwners.delete(owner); else state.hiddenOwners.add(owner);
      e.currentTarget.setAttribute('aria-pressed', String(!state.hiddenOwners.has(owner)));
      render(data, root);
    },
  }, swatch(SHAPES[owner], owner === 'nanocodec' ? 'ours' : 'ref'), owner === 'nanocodec' ? 'nanocodec (ours)' : owner)),
  el('button', {
    type: 'button', 'aria-pressed': String(state.tuned),
    title: 'The references’ metric-oriented tunings (x264/x265 --tune psnr or ssim, libvpx --tune ssim)',
    onclick: (e) => { state.tuned = !state.tuned; e.currentTarget.setAttribute('aria-pressed', String(state.tuned)); render(data, root); },
  }, swatch('square', 'tuned'), 'metric-tuned variants'),
  el('span', { class: 'frontier-key', title: 'A point is on its frontier when no other point of its group is both faster and lower (better) on this metric and machine' },
    el('span', { class: 'frontier-swatch ours' }), 'our Pareto frontier ',
    el('span', { class: 'frontier-swatch ref' }), 'the references’ (emphasised points are on it)'),
  el('span', { class: 'pin-key', hidden: true }, pinSwatch(), PIN_NOTE));
}

function controls(data, root) {
  const presets = [...new Set(Object.values(data.points).filter((p) => p.ours).map((p) => p.preset))];
  const rerender = () => render(data, root);
  const codecSelect = el('select', { 'aria-label': 'Codec', onchange: (e) => {
    state.codecs = new Set(e.target.value === ALL ? Object.keys(CODEC_NAMES) : [e.target.value]);
    rerender();
  } }, el('option', { value: ALL }, 'All codecs'), data.codec_order.map((c) => el('option', { value: c }, CODEC_NAMES[c])));
  if (state.codecs.size === 1) codecSelect.value = [...state.codecs][0];
  const presetSelect = el('select', { 'aria-label': 'Preset', onchange: (e) => { state.preset = e.target.value; rerender(); } },
    el('option', { value: ALL }, 'All presets'), presets.map((p) => el('option', { value: p }, `${p} (ours)`)));
  presetSelect.value = state.preset;
  const withData = (name, widget) => { widget.dataset.control = name; return widget; };
  root.querySelector('.controls').replaceChildren(
    control('Clip set', withData('clipset', segmented('Clip set', [
      { value: 'fresh', label: 'Held-out, fresh' }, { value: 'heldout', label: 'Held-out, all' }, { value: 'dev', label: 'Development' },
      ...Object.entries(data.datasets).map(([value, d]) => ({ value, label: d.name })),
      { value: 'ctc', label: 'AOM CTC (Fixed QP)' },
    ], state.clipset, (v) => { if (state.mode === 'qp') state.qpClipset = v; else state.clipset = v; rerender(); }))),
    // Fixed QP: the standards lens at the chart's build, on the same axes.
    control('Rate mode', segmented('Rate mode', [
      { value: 'cbr', label: 'CBR' }, { value: 'qp', label: 'Fixed QP' },
    ], state.mode, (v) => { state.mode = v; rerender(); })),
    control('Metric', withData('metric', segmented('Metric', Object.keys(SHORT).map((m) => ({ value: m, label: SHORT[m] })),
      state.metric, (v) => { state.metric = v; rerender(); }))),
    control('Speed on', withData('platform', segmented('Platform', [
      { value: 'jetson', label: 'Jetson (Arm)' }, { value: 'x86', label: 'x86 (Zen 5)' },
      { value: 'zen3', label: 'x86 without AVX-512 (Zen 3)' },
      { value: 'graviton', label: 'Server Arm (Graviton4)' },
    ], state.platform, (v) => { state.platform = v; rerender(); }))),
    control('Pareto frontier', segmented('Pareto frontier', [
      { value: 'on', label: 'On' }, { value: 'off', label: 'Off' },
    ], state.frontier ? 'on' : 'off', (v) => { state.frontier = v === 'on'; rerender(); })),
    control('Codec', codecSelect),
    control('Preset', presetSelect),
  );
}

/** The page's note on each published outside dataset (data.datasets), after its clip sets. */
function datasetsNote(data) {
  document.querySelector('.datasets-note').replaceChildren(...Object.values(data.datasets).map((d) =>
    el('span', {}, ` ${d.name} is a sample of `, el('a', { href: d.url }, d.short), ` ${d.about}`)));
}

/** A demo link (deep-link.js `resultsQuery`): filter to that codec and preset and select its point. */
function applyLink(data) {
  const link = parseResultsQuery(location.search);
  if (!link || !data.codec_order.includes(link.codec)) return;
  state.codecs = new Set([link.codec]);
  const preset = link.preset[0].toUpperCase() + link.preset.slice(1);
  if (Object.values(data.points).some((p) => p.ours && p.codec === link.codec && p.preset === preset)) {
    state.preset = preset;
    state.selected = chartFamily(link);
  }
}

async function main() {
  const root = document.getElementById('explorer');
  try {
    const data = await loadData('chart');
    datasetsNote(data);
    applyLink(data);
    const owners = [...new Set(Object.values(data.points).map((p) => p.owner))].filter((o) => SHAPES[o]);
    owners.sort((a, b) => (a === 'nanocodec' ? -1 : b === 'nanocodec' ? 1 : a.localeCompare(b)));
    controls(data, root);
    legend(data, root, owners);
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
