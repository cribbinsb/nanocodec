// Decoder speed (decoders.html): our native decoders against each format's reference decoders
// on the chosen timing machine; our browser module against WebAssembly builds of the references
// on the same machine; and, apart, against the browser's own (native) decoders through WebCodecs.
// Data: data/decoders.json (tools/bench/preview_decoders.py, from the decoder-speed record).
import { drawRatioBars } from './chart.js';
import { CODEC_NAMES, control, el, fail, loadData, segmented, table } from './common.js';
import { chosenMachine, machinePicker } from './bp3.js';

// The stream groups the native chart can average over: the summary's ratio field and count.
const GROUPS = {
  all: { label: 'All streams', ratio: 'ratio', count: 'streams' },
  own: { label: 'Our encoder’s streams', ratio: 'ratio_own', count: 'own_streams' },
  other: { label: 'Other encoders’ streams', ratio: 'ratio_other', count: 'other_streams' },
};
// The reference each format's headline figure is against: the decoder that format is usually
// played with (FFmpeg for H.264 and HEVC, dav1d for AV1, libvpx for VP9, VVdeC for VVC, dav2d
// for AV2); the chart shows every reference.
const HEADLINE_REFERENCE = { h264: 'ffmpeg', hevc: 'ffmpeg', vvc: 'vvdec', av1: 'dav1d', vp9: 'vpxdec', av2: 'dav2d' };
const AXIS = 'decode time, ours ÷ the reference’s (below 1: ours took less time)';

const state = { machine: null, group: 'all' };

const two = (v) => (v === null || v === undefined ? '–' : v.toFixed(2));
const msCell = (t) => (t ? `${t.ms.toFixed(2)} (${t.lo.toFixed(2)}–${t.hi.toFixed(2)})` : '–');
const fpsOf = (t) => (t ? (1000 / t.ms).toFixed(0) : '–');
const size = (r) => `${r.width}×${r.height}`;
const mpx = (v) => (v === null || v === undefined ? '–' : v.toFixed(0));

/** The summary rows of one machine in the page's format order. */
function ordered(data, view) {
  const order = (s) => data.codec_order.indexOf(s.codec) * 10 + Object.keys(data.references).indexOf(s.reference);
  return [...view.summary].sort((a, b) => order(a) - order(b));
}

function headline(data, view) {
  return data.codec_order.map((codec) => {
    const s = view.summary.find((x) => x.codec === codec && x.reference === HEADLINE_REFERENCE[codec]);
    if (!s) return null;
    return el('div', {},
      el('span', {}, `${CODEC_NAMES[codec]} against ${data.short[s.reference]}`),
      el('strong', {}, `${two(s.ratio)}×`),
      el('span', {}, `the time, ${s.streams} streams`));
  }).filter(Boolean);
}

function native(data, view, root) {
  const group = GROUPS[state.group];
  const rows = ordered(data, view);
  root.querySelector('.headline-figures').replaceChildren(...headline(data, view));
  root.querySelector('.native-sub').textContent = `${data.machines[state.machine].name}, one core, every decoder one-threaded; ${group.label.toLowerCase()}.`;
  drawRatioBars(root.querySelector('.native-chart'), rows.map((s) => ({
    label: `${CODEC_NAMES[s.codec]} · ${data.short[s.reference]} (${s[group.count]})`,
    ratio: s[group.ratio] ?? null,
    tip: s[group.ratio] ? `${CODEC_NAMES[s.codec]}: ours ${two(s[group.ratio])}× the time of ${data.references[s.reference]}, geometric mean over ${s[group.count]} streams` : '',
  })), { axisLabel: AXIS, missingText: 'no streams in this group' });
  root.querySelector('.native-note').textContent = `Each bar is the geometric mean, over the streams of the group, of our median decode time divided by the reference's on the same stream. Our encoder's streams use only the coding tools our encoder chooses; other encoders' streams (x264, x265, SVT-AV1, libaom, libvpx, openh264 and camera recordings) exercise more of each format. VVC and AV2 have only our encoder's streams.`;
  root.querySelector('.native-table').replaceChildren(table([
    { label: 'Format', cell: (s) => CODEC_NAMES[s.codec] },
    { label: 'Reference', cell: (s) => data.references[s.reference] },
    { label: 'Streams', num: true, cell: (s) => String(s.streams) },
    { label: 'Ours ÷ ref, all', num: true, cell: (s) => two(s.ratio) },
    { label: 'our streams', num: true, cell: (s) => (s.own_streams ? `${two(s.ratio_own)} (${s.own_streams})` : '–') },
    { label: 'other streams', num: true, cell: (s) => (s.other_streams ? `${two(s.ratio_other)} (${s.other_streams})` : '–') },
    { label: 'Plain build ÷ ref, all', num: true, cell: (s) => two(s.ratio_plain) },
    { label: 'Ours, Mpx/s', num: true, cell: (s) => mpx(s.ours_mpx_s) },
    { label: 'Ref, Mpx/s', num: true, cell: (s) => mpx(s.ref_mpx_s) },
  ], rows), el('p', { class: 'note' }, 'Mpx/s: decoded megapixels per second (luma width × height ÷ median time per frame), geometric mean over the streams; the ratios are geometric means of per-stream time ratios. Ours is the 0.7.0 PGO release build; the plain build (the same commit without PGO, as crates.io builds it) was timed in the same rounds, against the same reference runs.'));
  root.querySelector('.native-streams').replaceChildren(...data.codec_order.map((codec) => {
    const streams = view.native.filter((r) => r.codec === codec);
    if (!streams.length) return null;
    const refs = Object.keys(data.references).filter((ref) => streams.some((r) => r.decoders[ref]));
    return el('details', { class: 'stream-detail' }, el('summary', {}, `${CODEC_NAMES[codec]}: ${streams.length} streams`),
      el('div', { class: 'table-wrap' }, table([
        { label: 'Stream', cell: (r) => r.stream },
        { label: 'Encoder', cell: (r) => r.encoder },
        { label: 'Size', cell: size },
        { label: 'Frames', num: true, cell: (r) => String(r.frames) },
        { label: 'Ours ms/frame', num: true, cell: (r) => msCell(r.decoders.ours) },
        { label: 'Plain ms/frame', num: true, cell: (r) => msCell(r.decoders['ours-plain']) },
        ...refs.flatMap((ref) => [
          { label: `${data.short[ref]} ms/frame`, num: true, cell: (r) => msCell(r.decoders[ref]) },
          { label: `÷ ${data.short[ref]}`, num: true, cell: (r) => (r.decoders[ref] ? two(r.decoders.ours.ms / r.decoders[ref].ms) : '–') },
        ]),
      ], streams)));
  }).filter(Boolean), el('p', { class: 'note' }, 'ms per frame: the median of the passes (in brackets, the fastest and slowest pass).'));
}

const BROWSER_AXIS = 'browser decode time, ours ÷ the yardstick’s (below 1: ours took less time)';

/** Summary rows of one comparison (wasm or WebCodecs) in the page's format order. */
function orderedBy(data, summary, references) {
  const order = (s) => data.codec_order.indexOf(s.codec) * 10 + Object.keys(references).indexOf(s.reference);
  return [...summary].sort((a, b) => order(a) - order(b));
}

/** Every stream of a browser comparison, one details block per format. */
function streamDetails(data, rows, ours, references, short, extra = []) {
  return data.codec_order.map((codec) => {
    const streams = rows.filter((r) => r.codec === codec);
    if (!streams.length) return null;
    const refs = Object.keys(references).filter((ref) => streams.some((r) => r.decoders[ref]));
    return el('details', { class: 'stream-detail' }, el('summary', {}, `${CODEC_NAMES[codec]}: ${streams.length} streams`),
      table([
        { label: 'Stream', cell: (r) => r.stream },
        { label: 'Encoder', cell: (r) => r.encoder },
        { label: 'Size', cell: size },
        { label: 'Ours ms/frame', num: true, cell: (r) => msCell(r.decoders[ours]) },
        ...refs.flatMap((ref) => [
          { label: `${short[ref]} ms/frame`, num: true, cell: (r) => msCell(r.decoders[ref]) },
          { label: `÷ ${short[ref]}`, num: true, cell: (r) => (r.decoders[ref] && r.decoders[ours] ? two(r.decoders[ours].ms / r.decoders[ref].ms) : '–') },
        ]),
        ...extra,
      ], streams));
  }).filter(Boolean);
}

function wasm(data, view, root) {
  const w = view.wasm;
  const machine = data.machines[state.machine].name;
  root.querySelector('.wasm-sub').textContent = `node ${view.platform.node || ''} (V8) on ${machine}, one core, every decoder one-threaded; geometric mean over the streams each yardstick decodes.`;
  if (!w) {
    root.querySelector('.wasm-chart').replaceChildren(el('p', { class: 'note' }, 'Not measured on this machine.'));
    root.querySelector('.wasm-table').replaceChildren();
    root.querySelector('.wasm-streams').replaceChildren();
    return;
  }
  const rows = orderedBy(data, w.summary, data.wasm_references);
  drawRatioBars(root.querySelector('.wasm-chart'), rows.map((s) => ({
    label: `${CODEC_NAMES[s.codec]} · ${data.wasm_short[s.reference]} (${s.streams})`,
    ratio: s.ratio,
    tip: `${CODEC_NAMES[s.codec]}: our module ${two(s.ratio)}× the time of ${data.wasm_references[s.reference]}, geometric mean over ${s.streams} streams`,
  })), { axisLabel: BROWSER_AXIS, missingText: '' });
  root.querySelector('.wasm-table').replaceChildren(table([
    { label: 'Format', cell: (s) => CODEC_NAMES[s.codec] },
    { label: 'Yardstick', cell: (s) => data.wasm_references[s.reference] },
    { label: 'Streams', num: true, cell: (s) => String(s.streams) },
    { label: 'Ours ÷ yardstick, all', num: true, cell: (s) => two(s.ratio) },
    { label: 'our streams', num: true, cell: (s) => (s.own_streams ? `${two(s.ratio_own)} (${s.own_streams})` : '–') },
    { label: 'other streams', num: true, cell: (s) => (s.other_streams ? `${two(s.ratio_other)} (${s.other_streams})` : '–') },
    { label: 'Ours, Mpx/s', num: true, cell: (s) => mpx(s.ours_mpx_s) },
    { label: 'Yardstick, Mpx/s', num: true, cell: (s) => mpx(s.ref_mpx_s) },
  ], rows), el('p', { class: 'note' }, `${data.about.wasm_yardsticks} ${Object.keys(w.errors).length ? `Runs that failed and are left out: ${Object.entries(w.errors).map(([k, v]) => `${k.split('|')[0]} on ${data.wasm_short[k.split('|')[2]]} (${v.slice(0, 80)})`).join('; ')}.` : ''}`));
  root.querySelector('.wasm-streams').replaceChildren(...streamDetails(data, w.rows, 'ours-wasm', data.wasm_references, data.wasm_short, [
    { label: 'Ours native ms/frame', num: true, cell: (r) => (r.native ? r.native.ms.toFixed(2) : '–') },
    { label: 'Browser ÷ native', num: true, cell: (r) => (r.native && r.decoders['ours-wasm'] ? two(r.decoders['ours-wasm'].ms / r.native.ms) : '–') },
  ]), el('p', { class: 'note' }, 'ms per frame: the median of the runs (in brackets, the fastest and slowest run); every decoder\'s time is divided by the same frame count.'));
}

/** One WebCodecs block: a chart of ours ÷ each mode per format, and what was unavailable. */
function webcodecsBlock(data, wc, where) {
  if (!wc) return [el('p', { class: 'note' }, `Not measured on ${where}.`)];
  const rows = orderedBy(data, wc.summary, data.webcodecs);
  const chart = el('div', { class: 'chart' });
  // grouped by reason: "reason (H.264 hardware, HEVC software, ...)"
  const byReason = {};
  for (const [k, reason] of Object.entries(wc.unsupported)) {
    const [scope, mode] = k.split('|');
    (byReason[reason] ||= []).push(`${CODEC_NAMES[scope] || scope} ${data.webcodecs_short[mode].replace('WebCodecs ', '')}`);
  }
  const unsupported = Object.entries(byReason).map(([reason, list]) => `${list.join(', ')} (${reason})`);
  const parts = [
    el('p', { class: 'chart-sub' }, `${wc.channel === 'chrome' ? 'Google Chrome' : 'Chromium (Playwright build)'} ${wc.browser}, ${wc.headed ? 'in a window on a private headless Wayland compositor (headless Chrome cannot set up hardware decode output)' : 'headless'}, on ${where}; the browser bound to one core.`),
    chart,
    table([
      { label: 'Format', cell: (s) => CODEC_NAMES[s.codec] },
      { label: 'Browser decoder', cell: (s) => data.webcodecs[s.reference] },
      { label: 'Streams', num: true, cell: (s) => String(s.streams) },
      { label: 'Ours ÷ browser', num: true, cell: (s) => two(s.ratio) },
      { label: 'Ours, Mpx/s', num: true, cell: (s) => mpx(s.ours_mpx_s) },
      { label: 'Browser, Mpx/s', num: true, cell: (s) => mpx(s.ref_mpx_s) },
    ], rows),
    el('p', { class: 'note' }, `Not available through WebCodecs here: ${unsupported.join('; ') || 'none'}.${wc.gpu_video.length ? ` chrome://gpu: ${wc.gpu_video.join('; ')}.` : ''}`),
    el('details', { class: 'table-view' }, el('summary', {}, 'Every stream'),
      ...streamDetails(data, wc.rows, 'ours-chrome', data.webcodecs, data.webcodecs_short)),
  ];
  requestAnimationFrame(() => drawRatioBars(chart, rows.map((s) => ({
    label: `${CODEC_NAMES[s.codec]} · ${data.webcodecs_short[s.reference]} (${s.streams})`,
    ratio: s.ratio,
    tip: `${CODEC_NAMES[s.codec]}: our module ${two(s.ratio)}× the time of ${data.webcodecs[s.reference]}, geometric mean over ${s.streams} streams`,
  })), { axisLabel: 'decode time in Chrome, our module ÷ the browser’s decoder', missingText: '' }));
  return parts;
}

function webcodecs(data, view, root) {
  root.querySelector('.webcodecs-machine').replaceChildren(
    el('h3', {}, `Software, on ${data.machines[state.machine].name}`),
    ...webcodecsBlock(data, view.webcodecs, data.machines[state.machine].name));
  root.querySelector('.webcodecs-pc').replaceChildren(
    el('h3', {}, `Software and hardware, on ${data.pc.name}`),
    ...webcodecsBlock(data, data.pc.webcodecs, data.pc.name));
  root.querySelector('.webcodecs-note').textContent = data.about.webcodecs;
}

function method(data, view, root) {
  const a = data.about;
  const p = view.platform;
  root.querySelector('.results-banner').replaceChildren(
    el('strong', {}, `nanocodec ${a.release} PGO release build (${a.commit}). `),
    `Timed on ${data.machines[state.machine].name}${p.cpu ? `, ${p.cpu}` : ''}: one decode per physical core, every decoder one-threaded, `,
    `median of 15 passes per decoder and stream. Our own measurements; the references are timed one-threaded here and are usually run with more threads.`,
    data.missing.length ? ` Not measured yet: ${data.missing.join(', ')}.` : '');
  root.querySelector('.method-detail').replaceChildren(
    el('p', {}, el('strong', {}, 'Our decoders: '), `${a.build}. ${p.nanocodec ? `Version line: ${p.nanocodec}.` : ''} ${a.plain_build}.`),
    el('p', {}, el('strong', {}, 'Machine: '), `AWS ${data.machines[state.machine].type} (spot, eu-north-1)${p.cpu ? `, ${p.cpu}` : ''}. The decoders were split over three of its physical cores, one process per core and the box otherwise idle (${p.pinning || 'pinning not recorded'}); each stream's decoders ran in the same process's interleaved rounds.`),
    el('p', {}, el('strong', {}, 'Native method: '), `${a.method} ${a.harnesses}`),
    el('p', {}, el('strong', {}, 'References: '), Object.values(data.references).join('; ') + `. ${data.machines[state.machine].anchors}`),
    el('p', {}, el('strong', {}, 'Streams: '), a.streams),
    el('p', {}, el('strong', {}, 'Browser: '), `${a.browser_build}. ${a.browser_method} ${a.ffmpeg_wasm}. ${a.wasm_builds}`),
    el('p', {}, el('strong', {}, 'Exactness: '), 'checked separately, not here. Every decoder passes the landing conformance check (bit-exact against two standard decoders on our own streams). On other encoders\' streams: HEVC passes the official HEVC_v1 suite (147/147); VVC 226 of the 283 official H.266.1 streams; H.264 110 of 190 official H.264.1 streams (interlaced streams are not decoded yet); AV1 and VP9 pass the libaom (242/242) and libvpx (305/305) encoder test vectors, not official suites, and AV1 does not yet decode some valid Main tools; AV2 has no suite (30 of 30 foreign streams, 8-bit single tile). No decoder decodes every valid stream of its profiles yet: ', el('a', { href: 'https://github.com/cribbinsb/nanocodec/blob/main/docs/FAQ.md#conformance' }, 'the full list of gaps'), '.'),
    el('p', {}, el('strong', {}, 'Record: '), el('a', { href: `https://github.com/cribbinsb/nanocodec/tree/main/${data.record}` }, data.record), ' (every timed pass).'));
}

function render(data, root) {
  const view = data.views[state.machine];
  method(data, view, root);
  native(data, view, root);
  wasm(data, view, root);
  webcodecs(data, view, root);
}

async function main() {
  const root = document.getElementById('decoders');
  try {
    const data = await loadData('decoders');
    state.machine = chosenMachine(data.machines);
    root.querySelector('.controls').append(control('Streams in the native chart', segmented('Streams',
      Object.entries(GROUPS).map(([value, g]) => ({ value, label: g.label })), state.group,
      (v) => { state.group = v; render(data, root); })));
    machinePicker(data.machines, state.machine, (machine) => { state.machine = machine; render(data, root); });
    render(data, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
