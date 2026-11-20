// The headline chart's history (history.html): one frame per day, each the last
// integration build of that day re-measured under one protocol (tools/dev/history_chart.py),
// drawn in the overview chart's style with fixed axes, the reference encoders fixed,
// a play / pause button and a day slider. ?frame=N&still=1 shows one frame without
// controls (the GIF and MP4 frames are screenshots of it).
// Data: data/history.json (docs/results/history-2026-11-20/history.json).
import { paretoFront, frontierLine } from './frontier.js';
import { drawScatter } from './chart.js';
import { plotDomain } from './domain.js';
import { CODEC_NAMES, el, fail, fps, loadData, pct, table } from './common.js';

const SHAPES = { nanocodec: 'circle', x264: 'square', x265: 'diamond', 'SVT-AV1': 'triangle', openh264: 'triangle-down', libvpx: 'cross', libaom: 'plus' };
const FRAME_MS = 1100;
const X_LABEL = 'single-thread encode fps, 720p, one AMD EPYC 9R45 (Zen 5) core; timed on AWS c8a (log scale; faster →)';

function referencePoints(data) {
  return Object.entries(data.references).map(([family, r]) => ({
    id: family, x: r.fps_x86, y: r.j, iqr: family === data.zero ? null : r.iqr, kind: 'ref',
    shape: SHAPES[r.owner] || 'circle', label: r.label, flagged: r.flagged, priority: 1,
    aria: `${r.label}: ${pct(r.j)}, ${fps(r.fps_x86)} fps`,
  }));
}

const STEM_NAMES = { slow: 'Slow', fast: 'Fast', faster: 'Faster', veryfast: 'Veryfast' };
// Earlier frames' frontiers fade with age: the newest ghost at GHOST_OPACITY, older ones less.
const GHOST_OPACITY = 0.7;
const GHOST_FADE = 0.6;

/** `hevc@fast` -> HEVC Fast; `hevc` (the Default) -> HEVC. */
function presetLabel(key) {
  const [codec, stem] = key.split('@');
  return stem ? `${CODEC_NAMES[codec]} ${STEM_NAMES[stem] || stem}` : CODEC_NAMES[codec];
}

/** Our points of a day: the presets on the day's frontier, small and labelled; with `all`,
 * the others too, faint and unlabelled. A Default is ringed. */
function ourPoints(day, all = true) {
  const front = new Set(day.frontier || []);
  return Object.entries(day.values).filter(([k, v]) => v.j !== null && v.fps_x86 && (all || front.has(k))).map(([key, v]) => ({
    id: `nanocodec-${key}`, x: v.fps_x86, y: v.j, iqr: null, kind: 'ours', shape: 'circle',
    label: front.has(key) ? presetLabel(key) : '', flagged: v.flagged, priority: front.has(key) ? 3 : 2,
    classes: [front.has(key) ? '' : 'off-front', key.includes('@') ? '' : 'default-preset'].join(' ').trim(),
    scale: front.has(key) ? 0.8 : 0.6,
    aria: `${presetLabel(key)} on ${day.date}: ${pct(v.j)}, ${fps(v.fps_x86)} fps `,
  }));
}

const xy = (v) => [v.fps_x86, v.j];

/** A day's frontier as a line, slowest first. */
function frontLine(day) {
  return (day.frontier || []).map((k) => xy(day.values[k]));
}

/** Each codec's presets that day, slowest first: thin lines under the marks. */
function ladders(day) {
  return Object.keys(CODEC_NAMES).map((codec) => Object.entries(day.values)
    .filter(([k, v]) => k.split('@')[0] === codec && v.j !== null && v.fps_x86)
    .map(([, v]) => xy(v)).sort((p, q) => p[0] - q[0]))
    .map((line) => ({ line, cls: 'ladder' }));
}

/** The earlier frames' frontiers, older ones fainter. */
function ghosts(data, index) {
  return data.days.slice(0, index).map((d, i) => ({
    line: frontLine(d), cls: 'ghost-front', opacity: (GHOST_OPACITY * GHOST_FADE ** (index - 1 - i)).toFixed(3),
  }));
}

const AREAS = { rust: 'Rust (crates/)', eval_tools: 'eval and tools', docs: 'docs' };
const count = (n) => n.toLocaleString('en-US');

/** One frame's activity line: commits (by the agent its trailers name), lines added and
 * removed per area, and the Rust lines at the day's build. */
function activityText(day) {
  const a = day.activity;
  if (!a) return '';
  const agents = Object.entries(a.agents).sort((x, y) => y[1] - x[1])
    .map(([name, n]) => `${name === 'unmarked' ? 'no agent trailer' : name} ${count(n)}`).join(', ');
  const lines = Object.entries(AREAS).map(([k, label]) => `+${count(a.lines[k][0])} / −${count(a.lines[k][1])} ${label}`).join(', ');
  const period = a.since === day.date ? day.date : `${a.since} to ${day.date}`;
  return `Git, ${period}: ${count(a.commits)} commits (${agents}); lines ${lines}; ${count(a.rust_lines_total)} lines of Rust at this build.`;
}

function describe(p) {
  return el('div', {}, el('strong', {}, p.label), el('div', {}, p.aria));
}

function draw(root, data, index, domain) {
  const day = data.days[index];
  const showAll = root.querySelector('.all-toggle')?.checked ?? false;
  const points = [...referencePoints(data), ...ourPoints(day, showAll)];
  const refs = paretoFront(points.filter((p) => p.kind !== 'ours'));
  const ours = new Set((day.frontier || []).map((k) => `nanocodec-${k}`));
  const frontiers = [
    { kind: 'ref', line: frontierLine(refs), ids: new Set(refs.map((p) => p.id)) },
    { kind: 'ours bold', line: frontLine(day), ids: ours },
  ];

  root.querySelector('.frame-day').textContent = `${day.date} · build ${day.commit}`;
  root.querySelector('.frame-activity').textContent = activityText(day);
  root.querySelector('.frame-caption').replaceChildren(...(day.caption || []).map((line) => el('li', {}, line)));
  drawChange(root, data, index);
  drawScatter(root.querySelector('.chart'), points, {
    xLabel: X_LABEL, yLabel: 'BD-J, % bits for the same quality (lower is fewer bits)', describe, onSelect: () => {},
    frontiers, lines: [...ghosts(data, index), ...(showAll ? ladders(day) : [])], domain, ariaLabel: `Bits against speed on ${day.date}`,
  });
}

const SVG_NS = 'http://www.w3.org/2000/svg';
const PANEL = { width: 360, height: 150, left: 50, right: 24, top: 22, bottom: 26 };

function svgEl(tag, attrs, text) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}

/** One small line chart of a whole-front measure over the frames, the current one marked. */
function changePanel(data, index, title, value, format, zero) {
  const vals = data.days.map(value);
  const known = vals.filter((v) => v !== null && v !== undefined);
  const lo = Math.min(zero, ...known);
  const hi = Math.max(zero, ...known);
  const span = hi - lo || 1;
  const w = PANEL.width - PANEL.left - PANEL.right;
  const h = PANEL.height - PANEL.top - PANEL.bottom;
  const sx = (i) => PANEL.left + (data.days.length > 1 ? (i / (data.days.length - 1)) * w : 0);
  const sy = (v) => PANEL.top + ((hi - v) / span) * h;
  const root = svgEl('svg', { viewBox: `0 0 ${PANEL.width} ${PANEL.height}`, class: 'change-panel', role: 'img', 'aria-label': title });
  root.append(svgEl('text', { x: PANEL.left, y: 14, class: 'axis-title' }, title));
  root.append(svgEl('line', { x1: PANEL.left, x2: PANEL.width - PANEL.right, y1: sy(zero), y2: sy(zero), class: 'zero-line' }));
  for (const v of [lo, hi]) root.append(svgEl('text', { x: PANEL.left - 6, y: sy(v) + 4, 'text-anchor': 'end' }, format(v)));
  const pts = vals.map((v, i) => (v === null || v === undefined ? null : [sx(i), sy(v)]));
  root.append(svgEl('polyline', { class: 'change-line', points: pts.filter(Boolean).map((p) => p.join(',')).join(' ') }));
  pts.forEach((p, i) => {
    if (p) root.append(svgEl('circle', { cx: p[0], cy: p[1], r: i === index ? 5 : 2.5, class: i === index ? 'change-dot current' : 'change-dot' }));
  });
  data.days.forEach((d, i) => {
    if (i === 0 || i === index || i === data.days.length - 1) root.append(svgEl('text', { x: sx(i), y: PANEL.height - 8, 'text-anchor': 'middle' }, d.date.slice(5)));
  });
  return root;
}

function drawChange(root, data, index) {
  const base = data.front_base_day;
  const day = data.days[index];
  const b = day.front_vs_base;
  const prev = day.front_vs_previous;
  const both = (c) => (c && c.j_points !== null ? `${pct(c.j_points).replace('%', '')} points of BD-J at equal speed` : 'no overlap')
    + (c && c.speed_factor !== null ? `, ${c.speed_factor.toFixed(2)}× speed at equal J` : '');
  root.querySelector('.change-charts').replaceChildren(
    changePanel(data, index, `BD-J gap to the ${base} front (percentage points)`, (d) => d.front_vs_base?.j_points, (v) => pct(v), 0),
    changePanel(data, index, `speed factor at equal J vs the ${base} front`, (d) => d.front_vs_base?.speed_factor, (v) => `${v.toFixed(2)}×`, 1));
  root.querySelector('.frame-front').textContent = `Front: against the previous frame ${prev ? both(prev) : '–'}; against the ${base} front ${both(b)}.`;
  root.querySelector('.change-note').textContent = 'Each front is BD-J as a piecewise-linear function of log fps; the J gap is the mean difference at equal speed over the speeds both fronts cover, the speed factor the mean at equal BD-J over the BD-J range both cover; nothing is extrapolated.';
}

function dayTable(data) {
  return table([
    { label: 'Day', cell: (d) => d.date },
    { label: 'Build', cell: (d) => el('code', {}, d.commit) },
    { label: 'Main changes since the previous frame', cell: (d) => (d.caption || []).join(' ') },
    { label: 'Activity', cell: (d) => activityText(d) },
    { label: 'Our Pareto frontier, slowest first', cell: (d) => (d.frontier || []).map((k) => `${presetLabel(k)} ${pct(d.values[k].j)} at ${fps(d.values[k].fps_x86)} fps`).join('; ') },
  ], data.days);
}

function note(data) {
  const s = data.slice;
  return `Each frame is the last commit of that day on the integration branch (every other day, and today's head), built as a plain release build and measured now with one protocol: `
    + `every speed-ladder preset that build shipped, with its own values (before 2026-11-15 the files <codec>-fast, -faster and -fastest are Fast, Faster and Veryfast; files off the ladder are left out), real-time CBR on the chart's ladder (${data.ladder_kbps.join('/')} kbps at 720p30 scaled to each clip, `
    + `plus ${data.low_ladder_kbps.join('/')} kbps on ${data.low_rung_clips.length} clips whose curves need them), ${data.frames} frames, one key frame, `
    + `our reconstruction scored. Bits: BD-J (PSNR-YUV, VMAF v1 and MS-SSIM in equal thirds, class-balanced) against ${data.zero_label} `
    + `on a fixed slice of ${s.clips.length} of the ${data.fresh_clips} held-out fresh clips (${Object.entries(s.per_class).map(([k, n]) => `${n} ${k}`).join(', ')}; `
    + `within a class the first by a hash of the name), averaged over the ${data.clips_used} clips on which every point of every day and every reference overlaps the zero line. `
    + `Speed: fps timed on AWS c8a (Zen 5), one core, by the chart's own timing: its commands (three 720p clips, fixed QP found per build for about ${data.timing_rung_kbps} kbps), `
    + `each run back to back with x264 medium on one c8a.large, the median time ratio times the chart's stored x264 medium time; every day here is a plain release build (the overview's builds are PGO). The references are the overview chart's, `
    + `their quality re-averaged on the same clips and their fps as timed on the c8a. `
    + `The bold line is our Pareto frontier that day over every codec and preset (no other of our points is both faster and lower); the faint blue lines are the earlier frames' frontiers, older ones fainter; `
    + `the marks on it are its presets, a ringed one a codec's Default. "Show every preset" adds the presets off the frontier, faint, with thin lines joining each codec's presets.`;
}

async function main() {
  const root = document.getElementById('history');
  try {
    const data = await loadData('history');
    const all = [...referencePoints(data), ...data.days.flatMap(ourPoints)];
    const domain = plotDomain(all);
    const params = new URLSearchParams(location.search);
    const still = params.has('still');
    let index = Math.min(Number(params.get('frame') ?? data.days.length - 1), data.days.length - 1);
    root.querySelector('.view-note').textContent = note(data);
    const slider = root.querySelector('input[type=range]');
    const button = root.querySelector('.play');
    slider.max = String(data.days.length - 1);
    slider.value = String(index);
    let timer = null;
    const show = (i) => { index = i; slider.value = String(i); draw(root, data, i, domain); };
    const stop = () => { clearInterval(timer); timer = null; button.textContent = 'Play'; button.setAttribute('aria-pressed', 'false'); };
    button.addEventListener('click', () => {
      if (timer) { stop(); return; }
      if (index === data.days.length - 1) show(0);
      button.textContent = 'Pause';
      button.setAttribute('aria-pressed', 'true');
      timer = setInterval(() => { if (index >= data.days.length - 1) stop(); else show(index + 1); }, FRAME_MS);
    });
    slider.addEventListener('input', () => { stop(); show(Number(slider.value)); });
    root.querySelector('.all-toggle').addEventListener('change', () => draw(root, data, index, domain));
    if (still) document.body.classList.add('still');
    else root.querySelector('.table-slot').append(dayTable(data));
    show(index);
    window.addEventListener('resize', () => draw(root, data, index, domain));
    document.body.dataset.ready = 'true';
  } catch (error) {
    fail(root, error);
  }
}

main();
