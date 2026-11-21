// The method page (method.html): the clip sets, ladder, machines, versions and
// command lines from data/chart.json, and the window-length comparison from
// data/window.json, all filled in from the generated data.
import {
  METRIC_SHORT, ZERO_NAMES, control, el, fail, loadData, measuredNote, pct, segmented, sources, table,
} from './common.js';

const WINDOW_METRICS = { j: 'j', psnr: 'psnr_yuv', vmaf: 'vmaf_v1', ms_ssim: 'ms_ssim' };
const state = { zero: 'x264-medium', metric: 'j' };

function facts(chart, root) {
  const views = Object.fromEntries(chart.views.map((v) => [v.id, v]));
  const counts = {
    fresh: `${views['fresh-psnr'].clips} clips scored against x264 --tune psnr`,
    heldout: `${views['heldout-psnr'].clips} clips scored against x264 --tune psnr`,
    dev: `${views['dev-psnr'].clips} clips scored against x264 --tune psnr`,
    ...Object.fromEntries(Object.entries(chart.datasets).map(([id, d]) => [id,
      `${views[`${id}-psnr`].clips} clips scored against x264 --tune psnr; data ${d.credit}, ${d.url}, ${d.note}`])),
  };
  root.querySelector('.clipsets').replaceChildren(...Object.entries(chart.clipsets).map(([id, text]) =>
    el('li', {}, el('strong', {}, text.split(':')[0]), `: ${text.split(':').slice(1).join(':').trim()}; ${counts[id]}.`)));
  root.querySelector('.ladder').textContent = `Rates: CBR at ${chart.ladder_kbps.join(', ')} kbps for 720p at 30 fps, scaled to each clip's pixel rate, plus ${chart.low_ladder_kbps.join(' and ')} kbps on the clips where some curve has no reliable overlap with the zero line otherwise (for every encoder alike). ${chart.frames} frames per clip. A clip where some point still has no overlap is left out for every point.`;
  root.querySelector('.buffer').textContent = `a ${chart.buffer_ms} ms buffer`;
  root.querySelector('.machines').replaceChildren(...[
    el('li', {}, el('strong', {}, 'Arm: '), chart.platforms.jetson, ' (NVIDIA Jetson Orin Nano).'),
    el('li', {}, el('strong', {}, 'x86: '), chart.platforms.x86, ' (AMD).'),
    chart.platforms.zen3 && el('li', {}, el('strong', {}, 'x86 without AVX-512: '), chart.platforms.zen3, ' (AMD Zen 3).'),
    chart.platforms.graviton && el('li', {}, el('strong', {}, 'Server Arm: '), chart.platforms.graviton, ' (AWS Graviton4).'),
  ].filter(Boolean));
  const chartViews = chart.views.filter((v) => v.tools);
  root.querySelector('.versions').replaceChildren(table([
    { label: 'Chart', cell: (v) => `${chart.clipsets[v.clipset].split(':')[0]}, zero line ${ZERO_NAMES[v.zero]}` },
    { label: 'nanocodec build', cell: (v) => v.builds[v.base_build] || v.base_build },
    { label: 'References', cell: (v) => Object.entries(v.tools).map(([k, ver]) => `${k} ${ver}`).join(', ') },
  ], chartViews), el('p', { class: 'note' }, 'Also: ',
    Object.entries(chart.other_tools).map(([k, ver]) => `${k} ${ver}`).join(', '),
    ' (openh264 on every chart, HM and VVenC on the fixed-QP lens).'));
  const families = Object.keys(chart.points).filter((f) => chart.points[f].cbr_command)
    .sort((a, b) => Number(chart.points[b].ours) - Number(chart.points[a].ours) || chart.points[a].label.localeCompare(chart.points[b].label));
  root.querySelector('.commands').replaceChildren(
    el('p', { class: 'note' }, `Example: a ${chart.example.width}×${chart.example.height} clip at ${chart.example.fps} fps, the ${chart.example.kbps} kbps rung.`),
    ...families.map((f) => {
      const p = chart.points[f];
      return el('details', {}, el('summary', {}, p.label),
        el('code', { class: 'command' }, p.cbr_command),
        p.timing ? el('p', { class: 'note' }, `Timed at fixed QP${p.timing_is_twin ? ' (as its default-tuned twin)' : ''}:`) : null,
        p.timing ? el('code', { class: 'command' }, p.timing.command) : null);
    }),
    sources(['docs/results/bd-vs-speed-plan.json', 'eval/nceval/cbr.py']));
}

function windowTable(data, chart, root) {
  const metric = WINDOW_METRICS[state.metric];
  const cell = (entry) => (entry ? `${pct(entry[metric][0])} [${pct(entry[metric][1])}, ${pct(entry[metric][2])}]` : '–');
  const label = (f) => chart.points[f]?.label || f;
  const rows = Object.keys(data.families).filter((f) => data.families[f].short || data.families[f].long);
  rows.sort((a, b) => (data.families[a].short?.[state.zero]?.[metric][0] ?? 0) - (data.families[b].short?.[state.zero]?.[metric][0] ?? 0));
  root.querySelector('.window').replaceChildren(
    el('p', { class: 'note' }, `${METRIC_SHORT[state.metric]} BD-rate against ${ZERO_NAMES[state.zero]}, ${data.clips} held-out sources.`),
    table([
      { label: 'Encoder', cell: (f) => [el('span', { class: `swatch ${f.startsWith('nanocodec-') ? 'ours' : 'ref'}` }), label(f)] },
      { label: '150 frames, one key', num: true, cell: (f) => (f === state.zero ? '0' : cell(data.families[f].short?.[state.zero])) },
      { label: '600 frames, key every 10 s', num: true, cell: (f) => (f === state.zero ? '0' : cell(data.families[f].long?.[state.zero])) },
      { label: 'Rate ÷ target at 600 (median)', num: true, cell: (f) => data.families[f].rate_long?.ratio_median?.toFixed(3) ?? '–' },
      { label: 'Points over a strict 1 s buffer at 600', num: true, cell: (f) => {
        const r = data.families[f].rate_long;
        return r ? `${r.points_over_buffer} of ${r.points}` : '–';
      } },
    ], rows, (f) => (f.startsWith('nanocodec-') ? 'ours' : '') + (f === state.zero ? ' zero' : '')),
    measuredNote(data.measured.caption), sources(data.source));
}

async function main() {
  const root = document.getElementById('method');
  try {
    const [chart, windowData] = await Promise.all([loadData('chart'), loadData('window')]);
    facts(chart, root);
    root.querySelector('.buffer-text').textContent = chart.buffer_note;
    root.querySelector('.window-controls').append(
      control('Zero line', segmented('Zero line', [
        { value: 'x264-medium', label: 'x264 medium, default' }, { value: 'x264-medium~psnr', label: 'x264 medium, --tune psnr' },
      ], state.zero, (v) => { state.zero = v; windowTable(windowData, chart, root); })),
      control('Metric', segmented('Metric', Object.keys(METRIC_SHORT).map((m) => ({ value: m, label: METRIC_SHORT[m] })),
        state.metric, (v) => { state.metric = v; windowTable(windowData, chart, root); })));
    windowTable(windowData, chart, root);
  } catch (error) {
    fail(root, error);
  }
}

main();
