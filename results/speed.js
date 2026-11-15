// Speed (speed.html): encode fps per preset on both machines (data/chart.json),
// the browser build's decode and encode fps (data/wasm.json), our decoders against
// the reference decoders (data/decoders.json) and the hardware encoders
// (data/hardware.json).
import { CODEC_NAMES, el, fail, fps, loadData, pct, ratio, sources, table } from './common.js';

const SPEED_VIEW = 'fresh-psnr'; // every view carries the same timing record

function encode(chart, root) {
  const view = chart.views.find((v) => v.id === SPEED_VIEW);
  const rows = Object.entries(view.values)
    .filter(([family]) => !chart.points[family].tune || family === view.zero)
    .map(([family, v]) => ({ family, p: chart.points[family], v }))
    .sort((a, b) => b.v.fps.jetson - a.v.fps.jetson);
  root.querySelector('.platforms').textContent = `Jetson: ${chart.platforms.jetson}. PC: ${chart.platforms.x86}.`;
  root.querySelector('.encode').replaceChildren(table([
    { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.p.ours ? 'ours' : 'ref'}` }), r.p.label] },
    { label: 'Codec', cell: (r) => CODEC_NAMES[r.p.codec] },
    { label: 'fps, Jetson core', num: true, cell: (r) => fps(r.v.fps.jetson) },
    { label: 'fps, PC core', num: true, cell: (r) => fps(r.v.fps.x86) },
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
      columns.push({ label: `Encode ${res} ${preset === 'default' ? 'Default' : 'Fast'}`, num: true, cell: (c) => cell('encode', c, preset, res) });
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

function decoders(data, root) {
  const time = (r, platform) => {
    const v = r[platform];
    if (!v || v.ratio === null) return v ? `${v.ours_ms.toFixed(2)} ms` : '–';
    const second = v.second_ratio !== null ? ` (vs ${r.second_reference} ${v.second_ratio.toFixed(2)})` : '';
    return `${v.ours_ms.toFixed(2)} / ${v.ref_ms.toFixed(2)} ms = ${v.ratio.toFixed(2)}${second}`;
  };
  root.querySelector('.decoders').replaceChildren(table([
    { label: 'Codec', cell: (r) => CODEC_NAMES[r.codec] },
    { label: 'Stream', cell: (r) => r.stream },
    { label: 'Reference', cell: (r) => r.reference },
    { label: 'PC: ours / ref', num: true, cell: (r) => time(r, 'x86') },
    { label: 'Jetson: ours / ref', num: true, cell: (r) => time(r, 'jetson') },
    { label: 'WebAssembly (PC)', num: true, cell: (r) => (r.wasm ? `${r.wasm.ms.toFixed(2)} ms (${ratio(r.wasm.ratio)} native)` : '–') },
  ], data.rows), el('p', { class: 'note' }, `Build ${data.build}; milliseconds per frame, best round.`), sources(data.source));
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
    const [chart, wasmData, dec, hw] = await Promise.all([
      loadData('chart'), loadData('wasm'), loadData('decoders'), loadData('hardware')]);
    encode(chart, root);
    wasm(wasmData, root);
    decoders(dec, root);
    hardware(hw, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
