// Spatial scalability (svc.html): SVC, simulcast and single-layer points of AV1 and VP9 against
// single-layer x264 medium, as a chart (speed on the chosen timing machine) and a table with
// each stream's lower operating points. Data: data/svc.json (nceval.svc_chart `site`, through
// tools/bench/preview_pages.py); every figure is the record's.
import { drawScatter } from './chart.js';
import { control, el, fail, fps, loadData, pct, segmented, table } from './common.js';
import { chosenMachine, machinePicker } from './bp3.js';

const CODECS = { all: 'AV1 and VP9', av1: 'AV1', vp9: 'VP9' };
const SHAPES = { aom: 'plus', vpx: 'cross', simulcast: 'diamond', anchor: 'square' };
const KIND_NOTE = 'Marks: ● nanocodec (SVC and single-layer), ◆ nanocodec simulcast, ✚ libaom svc_encoder_rtc, ✖ libvpx vp9_spatial_svc_encoder, ■ x264 medium (the zero line).';
const state = { codec: 'all', selected: null };

const label = (r) => (r.ours ? `nanocodec ${r.label}` : r.label);
const shapeOf = (r) => (r.family === 'x264-medium' ? 'square' : SHAPES[r.kind] || 'circle');
const shown = (r) => state.codec === 'all' || r.codec === state.codec || r.family === 'x264-medium';

function render(data, machine, root) {
  const rows = data.rows.filter(shown);
  const timed = rows.filter((r) => r.speed[machine] != null && r.bd.j != null);
  drawScatter(root.querySelector('.chart'), timed.map((r) => ({
    id: r.family, x: r.speed[machine], y: r.bd.j, iqr: r.family === 'x264-medium' ? null : r.iqr,
    kind: r.ours ? 'ours' : 'ref', shape: shapeOf(r), label: label(r), flagged: false, twin: null,
    priority: r.ours ? 2 : 1, aria: `${label(r)}: ${pct(r.bd.j)}, ${fps(r.speed[machine])} fps`,
  })), {
    xLabel: `encode frames per CPU-second, all layers, one core of AWS ${data.timing[machine].machine}, ${data.machines[machine].label} (log scale)`,
    yLabel: 'BD-J of the 720p output (lower is better)',
    ariaLabel: 'SVC: bits saved against encode speed',
    selected: state.selected,
    describe: (p) => {
      const r = data.rows.find((row) => row.family === p.id);
      return el('div', {}, el('strong', {}, label(r)),
        el('div', {}, el('span', { class: 'value' }, pct(r.bd.j)), ' BD-J, 720p at the total rate'),
        r.base_360p != null ? el('div', {}, `360p view ${pct(r.base_360p)} against ${r.codec.toUpperCase()} alone at 360p`) : null,
        el('div', {}, Object.entries(data.machines).map(([m, d]) => `${fps(r.speed[m])} fps ${d.label}`).join(' · ')),
        el('div', { class: 'dim' }, `${r.clips} clips; bar: per-clip interquartile range`));
    },
    onSelect: (p) => { state.selected = p.id; render(data, machine, root); },
  });
  const untimed = rows.filter((r) => r.speed[machine] == null).map(label);
  root.querySelector('.legend-note').textContent = `${KIND_NOTE}${untimed.length ? ` Not timed on this machine (table only): ${untimed.join(', ')}.` : ''}`;
  const lower = (key) => (r) => (r[key] == null ? '' : pct(r[key]));
  root.querySelector('.svc-table').replaceChildren(table([
    { label: 'Encoder', cell: (r) => [el('span', { class: `swatch ${r.ours ? 'ours' : 'ref'}` }), label(r)] },
    { label: 'BD-J, 720p', num: true, cell: (r) => pct(r.bd.j) },
    { label: 'BD-PSNR-YUV', num: true, cell: (r) => pct(r.bd.psnr_yuv) },
    { label: 'BD-VMAF v1', num: true, cell: (r) => pct(r.bd.vmaf_v1) },
    { label: 'BD-MS-SSIM', num: true, cell: (r) => pct(r.bd.ms_ssim) },
    { label: '720p receiver vs single layer', num: true, cell: lower('receiver_720p') },
    { label: '360p view vs alone', num: true, cell: lower('base_360p') },
    { label: '180p view vs alone', num: true, cell: lower('view_180p') },
    { label: 'Lower layers\' share of rate', num: true, cell: (r) => (r.base_share == null ? '' : `${Math.round(100 * r.base_share)}%`) },
    ...Object.entries(data.machines).map(([m, d]) => ({ label: `fps, ${d.label}`, num: true, cell: (r) => fps(r.speed[m]) })),
  ], [...rows].sort((a, b) => (b.ours - a.ours) || ((a.bd.j ?? 1e9) - (b.bd.j ?? 1e9))),
  (r) => (r.family === 'x264-medium' ? 'zero' : r.ours ? 'ours' : '')));
}

/** A few sentences computed from the rows: what the layers cost against the codec alone, and
 *  where the reference SVC encoders stand. */
function findings(data) {
  const row = (family) => data.rows.find((r) => r.family === family);
  const out = [];
  for (const codec of ['av1', 'vp9']) {
    const single = row(`nanocodec-${codec}`);
    const l2 = row(`nanocodec-${codec}@l2-medium`);
    const key = row(`nanocodec-${codec}@l2-medium+L2T1_KEY`);
    const simulcast = row(`simulcast-${codec}`);
    if (!single || !l2) continue;
    out.push(el('li', {}, el('strong', {}, `${codec.toUpperCase()}: `),
      `two layers (L2T1, 720p + 360p) ${pct(l2.bd.j)} against x264 medium at the total rate, where ${codec.toUpperCase()} coding 720p alone is ${pct(single.bd.j)}`,
      simulcast ? ` and two independent streams (simulcast) ${pct(simulcast.bd.j)}` : '',
      `; its 360p view is ${pct(l2.base_360p)} against ${codec.toUpperCase()} alone at 360p.`,
      key ? ` What a 720p receiver downloads, against the single-layer 720p stream: ${pct(l2.receiver_720p)} BD-J in L2T1 (it takes every layer), ${pct(key.receiver_720p)} in K-SVC (L2T1_KEY: the 360p layer on key frames only).` : ''));
  }
  const refs = data.rows.filter((r) => r.kind === 'aom' || r.kind === 'vpx');
  if (refs.length) {
    const best = (kind) => refs.filter((r) => r.kind === kind).sort((a, b) => a.bd.j - b.bd.j)[0];
    const aom = best('aom');
    const vpx = best('vpx');
    out.push(el('li', {}, el('strong', {}, 'Reference SVC encoders: '),
      [aom && `libaom's example at best ${pct(aom.bd.j)} (${aom.label})`, vpx && `libvpx's at best ${pct(vpx.bd.j)} (${vpx.label})`].filter(Boolean).join('; '),
      ', against x264 medium on the same clips and rates.'));
  }
  return out;
}

function conditions(data) {
  const scored = data.clips.filter((c) => c.scored);
  const timings = Object.entries(data.timing).map(([m, t]) => `${data.machines[m].label}: AWS ${t.machine} (${t.region}, spot; one encode at a time), ${t.date}, ${t.rounds} rounds`);
  return [
    ['Build', `nanocodec ${data.build.slice(0, 9)} (release 0.7.0). Quality from the gate build of that commit (the same streams); speed from its plain release build (fat LTO, no PGO) on x86, built on the timing box on Arm.`],
    ['Clips', `${scored.length} of ${data.clips.length} development clips, 720p, ${data.frames} frames each, one key frame (${data.clips.filter((c) => !c.scored).map((c) => c.name).join(', ') || 'none'} left out for every point: some point lacks a reliable rate overlap with x264 medium there). These are development clips, not the benchmark's held-out sets.`],
    ['Rates', `the benchmark ladder, ${data.ladder_kbps.join(' / ')} kbps at 720p30 (scaled per clip, capped at half its source rate), CBR, ${data.buffer_ms} ms buffer; the rate is the whole stream, every layer.`],
    ['Ours', `the benchmark's CBR conditions (${data.own_conditions.join(' ')}), presets l2-medium, l2-fast, l3-medium, l3-fast; the spatial layers split the rate by our default shares.`],
    ['References', 'libaom svc_encoder_rtc v3.15.3 and libvpx vp9_spatial_svc_encoder v1.16.0: their projects\' real-time SVC examples, one thread, CBR, no frame drops, full quantizer range, cyclic-refresh AQ; example programs, not tuned products. x264 medium: the benchmark\'s CBR command.'],
    ['Quality', 'J (PSNR-YUV, VMAF v1, MS-SSIM in equal thirds) of every frame, scored on the PC\'s GPU; our 720p from the encoder\'s reconstruction, the lower views decoded (dav1d / vpxdec) and checked against the encoder\'s layer hashes; the references decoded at each operating point and cross-checked by a second decoder.'],
    ['Speed', `CPU time of one encode pinned to one physical core, every layer, ${timings.join('; ')}: the median over rounds per clip, the geometric mean over three 720p clips at the 600 kbps rung. Simulcast is not timed (two or three independent encodes). Reference builds: x264 and svc_encoder_rtc the benchmark\'s LTO + PGO builds; vp9_spatial_svc_encoder a plain build of libvpx v1.16.0 (the benchmark keeps no build of the example). Not comparable with the benchmark pages\' half-load Mpx/CPU-s.`],
  ];
}

async function main() {
  const root = document.getElementById('svc');
  try {
    const data = await loadData('svc');
    let machine = chosenMachine(data.machines);
    document.querySelector('.svc-banner').replaceChildren(
      `Measured ${data.date} on nanocodec 0.7.0 (${data.build.slice(0, 9)}), on development clips with the benchmark's ladder and CBR conditions; not part of the benchmark procedure's reports. `,
      el('a', { href: '#conditions' }, 'Conditions'), '.');
    root.querySelector('.svc-findings').replaceChildren(...findings(data));
    root.querySelector('.svc-conditions').replaceChildren(...conditions(data).map(([k, v]) => el('li', {}, el('strong', {}, `${k}: `), v)));
    root.querySelector('h2:last-of-type').id = 'conditions';
    root.querySelector('.controls').append(control('Codec', segmented('Codec',
      Object.entries(CODECS).map(([value, l]) => ({ value, label: l })), state.codec,
      (v) => { state.codec = v; render(data, machine, root); })));
    machinePicker(data.machines, machine, (m) => { machine = m; render(data, machine, root); });
    render(data, machine, root);
    let width = root.clientWidth;
    new ResizeObserver(() => {
      if (Math.abs(root.clientWidth - width) > 8) { width = root.clientWidth; render(data, machine, root); }
    }).observe(root);
  } catch (error) {
    fail(root, error);
  }
}

main();
