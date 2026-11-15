// The results explorer (index.html): bitrate saving vs encode speed for every
// encoder and preset, with the clip set, zero line, rate mode, metric, platform,
// codecs and presets selectable, a legend that toggles projects, a tooltip per
// point, a detail panel with the exact command line, and the same view as a table.
// A link ?codec=&preset= (deep-link.js) opens it filtered to that preset, selected;
// a selected point links to the demo with its settings (or its encoder's stream).
// Data: data/chart.json (python3 -m nceval.site, from docs/results).
import { drawScatter, swatch } from './chart.js';
import { chartFamily, demoQuery, manifestEncoder, parseResultsQuery } from '../deep-link.js';
import {
  CODEC_NAMES, METRIC_NAMES, METRIC_SHORT, ZERO_NAMES, control, el, fail, fps, loadData, pct,
  segmented, sources, table,
} from './common.js';

const SHAPES = { nanocodec: 'circle', x264: 'square', x265: 'diamond', 'SVT-AV1': 'triangle', openh264: 'triangle-down', libvpx: 'cross', libaom: 'plus' };
const PLATFORM_LABELS = { jetson: 'Jetson Orin Nano core', x86: 'Ryzen 9 5950X core' };
const TUNED_ZERO = 'x264-medium~psnr';
const ALL = 'all';
// The demo's pre-encoded streams (for a reference point's demo link), loaded on first use.
const MANIFEST_URL = '../streams-ref/manifest.json';
let manifest = null;

const state = {
  clipset: 'fresh', zero: TUNED_ZERO, mode: 'cbr', metric: 'j', platform: 'jetson',
  preset: ALL, codecs: new Set(Object.keys(CODEC_NAMES)), hiddenOwners: new Set(), tuned: false,
  selected: null,
};

function viewId() {
  const zid = state.zero === TUNED_ZERO ? 'psnr' : 'default';
  if (state.mode === 'qp') return `rc-qp-${zid}`;
  if (state.clipset === 'dev') return zid === 'default' ? 'dev-default' : 'rc-cbr-psnr';
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
    el('div', {}, el('span', { class: 'value' }, pct(v[state.metric])), ` ${METRIC_SHORT[state.metric]} vs ${ZERO_NAMES[view.zero]}`),
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
    ...view.metrics.map((m) => [METRIC_NAMES[m], pct(v[m])]),
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
  const tools = view.tools || data.views.find((v) => v.tools).tools;
  const refs = `x264 ${tools.x264}, x265 ${tools.x265}, SVT-AV1 ${tools.svt}, libaom ${tools.libaom}, libvpx ${tools.libvpx}`;
  if (view.clipset === 'dev-rc') {
    return `${view.rate_mode === 'qp' ? 'Fixed QP against x264 medium at fixed QP' : 'CBR against x264 medium in CBR'}: a separate run on the development clips (ladder ${view.ladder_kbps} kbps, ${view.frames} frames, one key frame), our Default presets and x264 ${tools.x264} only, at build ${view.build_commit}. Speed is the development chart's timing record. MS-SSIM is available for this run only.`;
  }
  const base = view.builds[view.base_build] || view.base_build;
  return `${data.clipsets[view.clipset]}: ${view.clips} clips, ${data.frames} frames each, one key frame, CBR ladder ${data.ladder_kbps.join('/')} kbps at 720p30 scaled to each clip (plus ${data.low_ladder_kbps.join('/')} kbps where curves need them), ${data.buffer_ms} ms buffer. Ours at ${base}; references ${refs}. Speed: ${data.platforms[state.platform]}.`;
}

function render(data, root) {
  const view = data.views.find((v) => v.id === viewId());
  if (!view.metrics.includes(state.metric)) state.metric = 'j';
  for (const [name, value] of [['clipset', state.mode === 'qp' ? 'dev' : state.clipset], ['metric', state.metric]]) {
    for (const b of root.querySelectorAll(`[data-control="${name}"] button`)) {
      b.setAttribute('aria-pressed', String(b.dataset.value === value));
      if (name === 'clipset') b.disabled = state.mode === 'qp' && b.dataset.value !== 'dev';
      if (name === 'metric') {
        b.disabled = !view.metrics.includes(b.dataset.value);
        b.title = b.disabled ? 'Not in this run’s data (the held-out and development charts record J, PSNR-YUV and VMAF v1)' : '';
      }
    }
  }
  root.querySelector('.view-note').textContent = viewNote(data, view);

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
  title.textContent = `${METRIC_NAMES[state.metric]} vs ${ZERO_NAMES[view.zero]}, against single-thread encode speed`;
  root.querySelector('.chart-sub').textContent = `${view.rate_mode === 'qp' ? 'Fixed QP' : 'Real-time CBR'} · ${view.clipset === 'dev-rc' ? 'development clips, Defaults' : data.clipsets[view.clipset].split(':')[0]} · lower is fewer bits, right is faster`;
  if (!points.length) {
    chart.replaceChildren(el('p', { class: 'note' }, 'No points match these filters.'));
  } else {
    drawScatter(chart, points, {
      xLabel: `encode fps, 720p, one ${PLATFORM_LABELS[state.platform]} (log scale)`,
      yLabel: `${METRIC_SHORT[state.metric]} BD-rate (lower is better)`,
      ariaLabel: title.textContent,
      selected: state.selected,
      describe: (p) => describe(data, view, p.id, view.values[p.id]),
      onSelect: (p) => { state.selected = p.id; render(data, root); root.querySelector('.detail-slot').scrollIntoView({ block: 'nearest', behavior: 'smooth' }); },
    });
  }
  const slot = root.querySelector('.detail-slot');
  const chosen = state.selected && view.values[state.selected];
  slot.replaceChildren(chosen ? detail(data, view, state.selected) : el('p', { class: 'note' }, 'Select a point (click, tap or Enter) for its versions, every metric, its exact command line and a link to try it in the demo.'));
  if (chosen) fillDemoLink(data, state.selected, root);

  const rows = entries.map(([family, v]) => ({ family, v, p: data.points[family] }))
    .sort((a, b) => a.v[state.metric] - b.v[state.metric]);
  root.querySelector('.table-slot').replaceChildren(table([
    { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.p.ours ? 'ours' : 'ref'}` }), r.p.label] },
    ...view.metrics.map((m) => ({ label: METRIC_SHORT[m], num: true, cell: (r) => pct(r.v[m]) })),
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
  }, swatch('square', 'tuned'), 'metric-tuned variants'));
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
    control('Zero line', segmented('Zero line', [
      { value: TUNED_ZERO, label: 'x264 medium, --tune psnr' }, { value: 'x264-medium', label: 'x264 medium, default' },
    ], state.zero, (v) => { state.zero = v; rerender(); })),
    control('Rate mode', segmented('Rate mode', [
      { value: 'cbr', label: 'CBR' }, { value: 'qp', label: 'Fixed QP' },
    ], state.mode, (v) => { state.mode = v; rerender(); })),
    control('Metric', withData('metric', segmented('Metric', Object.keys(METRIC_SHORT).map((m) => ({ value: m, label: METRIC_SHORT[m] })),
      state.metric, (v) => { state.metric = v; rerender(); }))),
    control('Speed on', segmented('Platform', [
      { value: 'jetson', label: 'Jetson (Arm)' }, { value: 'x86', label: 'PC (x86)' },
    ], state.platform, (v) => { state.platform = v; rerender(); })),
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
