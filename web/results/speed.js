// Speed (speed.html): encode fps per preset on each machine (data/chart.json),
// the browser build's decode and encode fps (data/wasm.json), our decoders against
// the reference decoders per platform (data/decoders.json; the browser against
// ffmpeg.wasm and its encode speed against quality: data/wasm-yardstick.json)
// and the hardware encoders
// (data/hardware.json).
import { drawRatioBars, drawScatter } from './chart.js';
import { CODEC_NAMES, control, el, fail, fps, loadData, pct, ratio, segmented, sources, table } from './common.js';

const SPEED_VIEW = 'fresh-psnr'; // every view carries the same timing record

function encode(chart, root) {
  const view = chart.views.find((v) => v.id === SPEED_VIEW);
  const rows = Object.entries(view.values)
    .filter(([family]) => !chart.points[family].tune || family === view.zero)
    .map(([family, v]) => ({ family, p: chart.points[family], v }))
    .sort((a, b) => b.v.fps.jetson - a.v.fps.jetson);
  const machines = { jetson: 'Jetson', x86: 'x86', zen3: 'x86 without AVX-512', graviton: 'Server Arm' };
  root.querySelector('.platforms').textContent = Object.entries(machines).filter(([m]) => chart.platforms[m])
    .map(([m, name]) => `${name}: ${chart.platforms[m]}${chart.timing_builds?.[m] ? `; ours timed at build ${chart.timing_builds[m]}` : ''}.`).join(' ');
  root.querySelector('.encode').replaceChildren(table([
    { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.p.ours ? 'ours' : 'ref'}` }), r.p.label] },
    { label: 'Codec', cell: (r) => CODEC_NAMES[r.p.codec] },
    { label: 'fps, Jetson core', num: true, cell: (r) => fps(r.v.fps.jetson) },
    { label: 'fps, x86 core', num: true, cell: (r) => fps(r.v.fps.x86) },
    { label: 'fps, Zen 3 core', num: true, cell: (r) => fps(r.v.fps.zen3) },
    { label: 'fps, Graviton4 core', num: true, cell: (r) => fps(r.v.fps.graviton) },
  ], rows, (r) => (r.p.ours ? 'ours' : '')), sources(view.source));
}

function wasm(data, root) {
  const key = (r) => `${r.direction}|${r.codec}|${r.preset || ''}|${r.res}`;
  const byKey = new Map(data.rows.map((r) => [key(r), r]));
  const cell = (direction, codec, preset, res) => {
    const r = byKey.get(`${direction}|${codec}|${preset}|${res}`);
    if (!r) return '–';
    return `${fps(r.wasm_fps)} (${ratio(r.native_fps / r.wasm_fps)})${r.seed_caveat ? ' *' : ''}`;
  };
  const codecs = [...new Set(data.rows.map((r) => r.codec))].sort((a, b) => Object.keys(CODEC_NAMES).indexOf(a) - Object.keys(CODEC_NAMES).indexOf(b));
  const columns = [{ label: 'Codec', cell: (c) => CODEC_NAMES[c] }];
  for (const res of ['720p', '360p']) columns.push({ label: `Decode ${res}`, num: true, cell: (c) => cell('decode', c, '', res) });
  for (const res of ['720p', '360p']) {
    for (const preset of ['default', 'fast']) {
      columns.push({ label: `Encode ${res} ${preset === 'default' ? 'Medium (default)' : 'Fast'}`, num: true, cell: (c) => cell('encode', c, preset, res) });
    }
  }
  const builds = [...new Set(data.rows.flatMap((r) => r.builds))];
  root.querySelector('.wasm').replaceChildren(table(columns, codecs),
    el('p', { class: 'note' }, `Builds ${builds.join(', ')}; the "${data.module}" module; frame-weighted over ${Math.max(...data.rows.map((r) => r.clips))} clips per row.`),
    sources(data.source));
  if (data.rows.some((r) => r.seed_caveat)) {
    root.querySelector('.wasm-caveat').textContent = '* Measured with a different first-key quantizer than the demo uses (fixed in the bench afterwards): the WebAssembly side coded a larger first key frame and costlier frames after it, so these rows understate its speed.';
  }
}

const DECODE_PLATFORMS = { x86: 'PC (x86)', jetson: 'Jetson (Arm)', wasm: 'Browser (WebAssembly)' };
const decodeState = { platform: 'x86' };

/** Per stream on the chosen platform: ours and the reference in ms per frame and their ratio
 * (the browser: ffmpeg.wasm from the same session; null where nothing was measured). */
function decodeRows(native, browser) {
  const rows = native.rows.map((r) => {
    if (decodeState.platform !== 'wasm') {
      const v = r[decodeState.platform];
      return { r, reference: r.reference, ours: v?.ours_ms ?? null, ref: v?.ref_ms ?? null, ratio: v?.ratio ?? null,
        second: v?.second_ratio != null ? `vs ${r.second_reference} ${v.second_ratio.toFixed(2)}` : '' };
    }
    const w = browser.decode.find((d) => d.codec === r.codec && d.stream === r.stream);
    return { r, reference: 'ffmpeg.wasm', ours: w?.ours_ms ?? null, ref: w?.ref_ms ?? null, ratio: w?.ratio ?? null, second: '' };
  });
  if (decodeState.platform !== 'wasm') return rows;
  // Browser streams the native run did not use (e.g. HEVC on the 720p crowd stream).
  const extra = browser.decode.filter((d) => !native.rows.some((r) => r.codec === d.codec && r.stream === d.stream))
    .map((d) => ({ r: d, reference: 'ffmpeg.wasm', ours: d.ours_ms, ref: d.ref_ms, ratio: d.ratio, second: '' }));
  return [...rows, ...extra];
}

function decoders(native, browser, root) {
  const rows = decodeRows(native, browser);
  const ms = (v) => (v === null ? 'no data' : `${v.toFixed(2)} ms`);
  root.querySelector('.decoder-note').textContent = decodeState.platform === 'wasm'
    ? `Browser: our module (build ${browser.build}) and ffmpeg.wasm, one PC core, same session. ffmpeg.wasm has no SIMD and no AV1, VVC or AV2 decoder; streams it was not run on show no data.`
    : `${DECODE_PLATFORMS[decodeState.platform]}: build ${native.build}, every decoder single-threaded.`;
  drawRatioBars(root.querySelector('.ratio-chart'), rows.map((x) => ({
    label: `${CODEC_NAMES[x.r.codec]} · ${x.r.stream}`, ratio: x.ratio,
    tip: x.ratio === null ? '' : `${CODEC_NAMES[x.r.codec]} ${x.r.stream}: ours ${ms(x.ours)}, ${x.reference} ${ms(x.ref)}`,
  })), { axisLabel: 'our decode time ÷ the reference\u2019s (below 1: ours took less time)', missingText: 'no data' });
  root.querySelector('.decoders').replaceChildren(table([
    { label: 'Codec', cell: (x) => CODEC_NAMES[x.r.codec] },
    { label: 'Stream', cell: (x) => x.r.stream },
    { label: 'Reference', cell: (x) => x.reference },
    { label: 'Ours', num: true, cell: (x) => ms(x.ours) },
    { label: 'Reference', num: true, cell: (x) => ms(x.ref) },
    { label: 'Ours ÷ ref', num: true, cell: (x) => (x.ratio === null ? '–' : `${x.ratio.toFixed(2)}${x.second ? ` (${x.second})` : ''}`) },
  ], rows), sources(decodeState.platform === 'wasm' ? browser.source : native.source));
}

function wasmEncode(data, root) {
  const chart = root.querySelector('.wasm-encode-chart');
  const ours = (r) => r.module !== 'ffmpeg.wasm';
  const draw = () => drawScatter(chart, data.encode.map((r) => ({
    id: r.name, x: r.fps, y: r.j, iqr: null, kind: ours(r) ? 'ours' : 'ref', shape: ours(r) ? 'circle' : 'square',
    label: ours(r) ? r.label.replace(/^nanocodec /, '') : `${r.label} (ffmpeg.wasm)`, flagged: false, twin: null,
    priority: ours(r) ? 2 : 1, aria: `${r.label}: ${fps(r.fps)} fps`,
  })), {
    xLabel: 'browser encode fps, 720p, one PC core (log scale)', yLabel: 'BD-J of the same encoder (lower is better)',
    ariaLabel: 'Browser encode speed against quality', selected: null, onSelect: () => {},
    describe: (p) => {
      const r = data.encode.find((e) => e.name === p.id);
      return el('div', {}, el('strong', {}, `${r.label} · ${ours(r) ? 'our module' : 'ffmpeg.wasm (no SIMD)'}`),
        el('div', {}, el('span', { class: 'value' }, `${fps(r.fps)} fps`), ` (${r.ms.toFixed(2)} ms per frame)`),
        el('div', {}, `${pct(r.j)} BD-J: the chart's native point`));
    },
  });
  draw();
  new ResizeObserver(draw).observe(chart.parentElement);
  root.querySelector('.wasm-encode-table').replaceChildren(table([
    { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${ours(r) ? 'ours' : 'ref'}` }), r.label] },
    { label: 'Module', cell: (r) => (ours(r) ? `ours (build ${data.build})` : 'ffmpeg.wasm, no SIMD') },
    { label: 'fps', num: true, cell: (r) => fps(r.fps) },
    { label: 'BD-J (native chart point)', num: true, cell: (r) => pct(r.j) },
  ], data.encode, (r) => (ours(r) ? 'ours' : '')), sources(data.source));
}

function hardware(data, root) {
  root.querySelector('.hardware').replaceChildren(table([
    { label: 'Encoder (GPU)', cell: (r) => r.label },
    { label: 'BD-J vs x264 medium', num: true, cell: (r) => pct(r.j) + (r.chained ? ' ‡' : '') },
    { label: 'Aggregate fps', num: true, cell: (r) => `${fps(r.fps_aggregate)} (${r.aggregate_streams} streams)` },
    { label: 'One-stream fps', num: true, cell: (r) => fps(r.fps_1stream) },
  ], data.rows), el('p', { class: 'note' }, '‡ one clip chained through an intermediate encoder (no direct overlap with the zero line).'), sources(data.source));
}

async function main() {
  const root = document.getElementById('speed');
  try {
    const [chart, wasmData, dec, browser, hw] = await Promise.all([
      loadData('chart'), loadData('wasm'), loadData('decoders'), loadData('wasm-yardstick'), loadData('hardware')]);
    encode(chart, root);
    wasm(wasmData, root);
    wasmEncode(browser, root);
    root.querySelector('.decoder-controls').append(control('Platform', segmented('Platform',
      Object.entries(DECODE_PLATFORMS).map(([value, label]) => ({ value, label })), decodeState.platform,
      (v) => { decodeState.platform = v; decoders(dec, browser, root); })));
    decoders(dec, browser, root);
    hardware(hw, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
