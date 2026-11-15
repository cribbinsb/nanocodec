// The results explorer (index.html): bitrate saving vs encode speed for every
// encoder and preset, with the clip set, rate mode, metric, platform,
// codecs and presets selectable, a legend that toggles projects, a tooltip per
// point, a detail panel with the exact command line, and the same view as a table.
// A link ?codec=&preset= (deep-link.js) opens it filtered to that preset, selected;
// a selected point links to the demo with its settings (or its encoder's stream).
// Data: data/chart.json (python3 -m nceval.site, from docs/results).
import { paretoFront, frontierLine } from './frontier.js';
import { drawScatter, swatch } from './chart.js';
import { chartFamily, demoQuery, manifestEncoder, parseResultsQuery } from '../deep-link.js';
import {
  CODEC_NAMES, METRIC_NAMES, METRIC_SHORT, ZERO_NAMES, control, el, fail, fps, loadData, pct,
  segmented, sources, table,
} from './common.js';

const SHAPES = { nanocodec: 'circle', x264: 'square', x265: 'diamond', 'SVT-AV1': 'triangle', openh264: 'triangle-down', libvpx: 'cross', libaom: 'plus' };
const PLATFORM_LABELS = { jetson: 'Jetson Orin Nano core', x86: 'Ryzen 9 5950X core' };
// The fixed-QP chart's extra metric (the standards' weighted PSNR); the rest are common.js's.
const NAMES = { ...METRIC_NAMES, weighted: 'BD-rate, weighted PSNR (0.92 Y + 0.04 Cb + 0.04 Cr)' };
const SHORT = { ...METRIC_SHORT, weighted: 'Weighted' };
// Fixed-QP speed: the lens run's own measurement, on the PC only.
const QP_SPEED = 'the chart\'s single-thread timing at fixed QP (720p), on the machine selected';
const TUNED_ZERO = 'x264-medium~psnr';
const ALL = 'all';
// The demo's pre-encoded streams (for a reference point's demo link), loaded on first use.
const MANIFEST_URL = '../streams-ref/manifest.json';
let manifest = null;

const state = {
  clipset: 'fresh', zero: TUNED_ZERO, mode: 'cbr', metric: 'j', platform: 'jetson', frontier: true,
  preset: ALL, codecs: new Set(Object.keys(CODEC_NAMES)), hiddenOwners: new Set(), tuned: false,
  selected: null,
};

function viewId() {
  if (state.mode === 'qp') return 'qp';
  const zid = state.zero === TUNED_ZERO ? 'psnr' : 'default';
  return `${state.clipset}-${zid}`;
}

function versionOf(data, view, family) {
  const point = data.points[family];
  if (point.ours) {
    if (view.build_commit) return `build ${view.build_commit}`;
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
    el('div', {}, `${fps(v.fps.jetson)} fps Jetson core · ${fps(v.fps.x86)} fps PC core`),
    v.clips_scored ? el('div', { class: 'dim' }, `${v.clips_scored} clips with a reliable BD`) : null,
    el('div', { class: 'dim' }, 'Select the point for its command line'));
}

function detail(data, view, family) {
  const p = data.points[family];
  const v = view.values[family];
  const rows = [
    ['Encoder', `${p.owner}${versionOf(data, view, family) ? `, ${versionOf(data, view, family)}` : ''}`],
    ['Codec', CODEC_NAMES[p.codec] || p.codec],
    ['Preset', p.preset + (p.tune ? `, --tune ${p.tune}` : '')],
    ...view.metrics.map((m) => [NAMES[m], pct(v[m])]),
    v.iqr && v.iqr[0] !== null && family !== view.zero ? ['Per-clip spread (J)', `${pct(v.iqr[0])} to ${pct(v.iqr[1])} (interquartile)`] : null,
    v.clips_scored ? ['Clips scored', String(v.clips_scored)] : null,
    v.rate_invalid_points ? ['Rate-invalid points', `${v.rate_invalid_points} (kept on their curves)`] : null,
    v.chained ? ['Chained clips', `${v.chained} (through one intermediate encoder)`] : null,
    ['Encode speed', `${fps(v.fps.jetson)} fps on one ${PLATFORM_LABELS.jetson}, ${fps(v.fps.x86)} fps on one ${PLATFORM_LABELS.x86}`],
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
  if (view.rate_mode === 'qp') {
    const dev = data.views.find((v) => v.id === 'dev-psnr');
    return `Bits: BD-rate against x264 medium at fixed QP on the AOM CTC low-delay sequences (${view.condition}, ${view.clips} sequences), ours with its fixed-QP profile (column "${view.column}"; all three columns on the Fixed QP page), build ${view.build_commit}. Speed: single-thread encode fps at 720p, fixed QP, on the timing cores of the ${PLATFORM_LABELS[state.platform].replace(' core', '')} (${data.platforms[state.platform]}), from the chart timing at build ${dev.builds[dev.base_build] || dev.base_build}; an encoder without that timing is listed under the chart.`;
  }
  const tools = view.tools || data.views.find((v) => v.tools).tools;
  const refs = `x264 ${tools.x264}, x265 ${tools.x265}, SVT-AV1 ${tools.svt}, libaom ${tools.libaom}, libvpx ${tools.libvpx}`;
  const base = view.builds[view.base_build] || view.base_build;
  return `${data.clipsets[view.clipset]}: ${view.clips} clips, ${data.frames} frames each, one key frame, CBR ladder ${data.ladder_kbps.join('/')} kbps at 720p30 scaled to each clip (plus ${data.low_ladder_kbps.join('/')} kbps where curves need them), ${data.buffer_ms} ms buffer. Ours at ${base}; references ${refs}. Speed: ${data.platforms[state.platform]}.`;
}

function render(data, root) {
  const view = data.views.find((v) => v.id === viewId());
  if (!view.metrics.includes(state.metric)) state.metric = 'j';
  for (const b of root.querySelectorAll('[data-control="platform"] button')) {
    b.setAttribute('aria-pressed', String(b.dataset.value === state.platform));
  }
  for (const [name, value] of [['clipset', state.clipset], ['metric', state.metric]]) {
    for (const b of root.querySelectorAll(`[data-control="${name}"] button`)) {
      b.setAttribute('aria-pressed', String(b.dataset.value === value));
      if (name === 'clipset') b.disabled = state.mode === 'qp';
      if (name === 'metric') {
        b.disabled = !view.metrics.includes(b.dataset.value);
        b.title = b.disabled ? 'Not in this run’s data (the held-out and development charts record J, PSNR-YUV and VMAF v1)' : '';
      }
    }
  }
  const note = root.querySelector('.view-note');
  note.textContent = `${viewNote(data, view)} `;
  note.append(el('a', { href: `clips.html?set=${view.rate_mode === 'qp' ? 'ctc' : view.clipset}` }, 'The clips.'));

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
    ? `Fixed QP · AOM CTC low delay, build ${view.build_commit} · lower is fewer bits, right is faster`
    : `Real-time CBR · ${data.clipsets[view.clipset].split(':')[0]} · lower is fewer bits, right is faster`;
  if (!points.length) {
    chart.replaceChildren(el('p', { class: 'note' }, 'No points match these filters.'));
  } else {
    // Pareto frontiers of what is shown (filters, metric and machine applied; a
    // point not yet timed is not in `points`): ours, and the references.
    const frontiers = !state.frontier ? [] : ['ours', 'ref'].map((kind) => {
      const front = paretoFront(points.filter((p) => (kind === 'ours') === (p.kind === 'ours')));
      return { kind, line: frontierLine(front), ids: new Set(front.map((p) => p.id)) };
    });
    drawScatter(chart, points, {
      frontiers,
      xLabel: `encode fps, 720p, one ${PLATFORM_LABELS[state.platform]} (log scale)`,
      yLabel: `${SHORT[state.metric]} BD-rate (lower is better)`,
      ariaLabel: title.textContent,
      selected: state.selected,
      describe: (p) => describe(data, view, p.id, view.values[p.id]),
      onSelect: (p) => { state.selected = p.id; render(data, root); root.querySelector('.detail-slot').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); },
    });
  }
  // Encoders with a value but no speed on this machine (never timed here): listed, not dropped.
  const untimed = Object.entries(view.values)
    .filter(([f, v]) => data.points[f] && !v.fps?.[state.platform] && v[state.metric] !== null)
    .map(([f, v]) => `${data.points[f].label} (${pct(v[state.metric])})`);
  if (untimed.length) {
    chart.append(el('p', { class: 'note off-scale' }, `Not yet timed, so not plotted: ${untimed.join('; ')}.`));
  }
  if (view.rate_mode === 'qp') {
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
    ...view.metrics.map((m) => ({ label: SHORT[m], num: true, cell: (r) => pct(r.v[m]) })),
    { label: 'Clips', num: true, cell: (r) => (r.v.clips_scored ? String(r.v.clips_scored) : '–') },
    { label: 'fps Jetson', num: true, cell: (r) => fps(r.v.fps.jetson) },
    { label: 'fps PC', num: true, cell: (r) => fps(r.v.fps.x86) },
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
    el('span', { class: 'frontier-swatch ref' }), 'the references’ (emphasised points are on it)'));
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
    ], state.clipset, (v) => { state.clipset = v; rerender(); }))),
    // Fixed QP: the standards lens at the chart's build, on the same axes.
    control('Rate mode', segmented('Rate mode', [
      { value: 'cbr', label: 'CBR' }, { value: 'qp', label: 'Fixed QP' },
    ], state.mode, (v) => { state.mode = v; rerender(); })),
    control('Metric', withData('metric', segmented('Metric', Object.keys(SHORT).map((m) => ({ value: m, label: SHORT[m] })),
      state.metric, (v) => { state.metric = v; rerender(); }))),
    control('Speed on', withData('platform', segmented('Platform', [
      { value: 'jetson', label: 'Jetson (Arm)' }, { value: 'x86', label: 'PC (x86)' },
    ], state.platform, (v) => { state.platform = v; rerender(); }))),
    control('Pareto frontier', segmented('Pareto frontier', [
      { value: 'on', label: 'On' }, { value: 'off', label: 'Off' },
    ], state.frontier ? 'on' : 'off', (v) => { state.frontier = v === 'on'; rerender(); })),
    control('Codec', codecSelect),
    control('Preset', presetSelect),
  );
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
