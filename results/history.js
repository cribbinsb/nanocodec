// The headline chart's history (history.html): one frame per day, each the last
// integration build of that day re-measured under one protocol (tools/dev/history_chart.py),
// drawn in the overview chart's style with fixed axes, the reference encoders fixed,
// a play / pause button (autoplays on a loop) and a day slider. ?frame=N&still=1 shows one frame without
// controls (the GIF and MP4 frames are screenshots of it).
// Data: data/history.json (docs/results/history-2026-11-20/history.json).
import { paretoFront, frontierLine } from './frontier.js';
import { drawScatter } from './chart.js';
import { plotDomain } from './domain.js';
import { CODEC_NAMES, control, el, fail, fps, pct, segmented, table } from './common.js';

/** Set an element's text if the page has it (a page cached from before the element existed
 * still renders). */
function setText(root, selector, text) {
  const node = root.querySelector(selector);
  if (node) node.textContent = text;
}

/** Replace an element's children if the page has it. */
function setChildren(root, selector, children) {
  const node = root.querySelector(selector);
  if (node) node.replaceChildren(...children);
}

// The asset version this module was loaded with (history.html: history.js?v=...), so the
// data it reads is the one published with it.
const VERSION = new URL(import.meta.url).searchParams.get('v') || '';

const SHAPES = { nanocodec: 'circle', x264: 'square', x265: 'diamond', 'SVT-AV1': 'triangle', openh264: 'triangle-down', libvpx: 'cross', libaom: 'plus' };
const FRAME_MS = 550;
const X_LABEL = 'single-thread encode fps, 720p, one AMD EPYC 9R45 (Zen 5) core; timed on AWS c8a (log scale; faster →)';

function referencePoints(data) {
  return Object.entries(data.references).map(([family, r]) => ({
    id: family, x: r.fps_x86, y: r.j, iqr: family === data.zero ? null : r.iqr, kind: 'ref',
    shape: SHAPES[r.owner] || 'circle', label: r.label, flagged: r.flagged, priority: 1,
    aria: `${r.label}: ${pct(r.j)}, ${fps(r.fps_x86)} fps`,
  }));
}

const STEM_NAMES = { veryslow: 'Veryslow', slower: 'Slower', slow: 'Slow', fast: 'Fast', faster: 'Faster', veryfast: 'Veryfast' };
// A rung renamed away keeps its old stem behind this prefix (history_chart.py RETIRED).
const RETIRED = 'retired-';
// Earlier frames' frontiers fade with age: the newest ghost at GHOST_OPACITY, older ones less.
const GHOST_OPACITY = 0.7;
const GHOST_FADE = 0.6;

// The codec filter: 'all' (our whole front) or one codec's own presets (?codec=av1).
const ALL = 'all';
const view = { codec: ALL };
const inView = (key) => view.codec === ALL || key.split('@')[0] === view.codec;
/** The day's frontier in the current view: over every codec, or the codec's presets alone. */
const frontKeys = (day) => (view.codec === ALL ? day.frontier : day.codec_fronts?.[view.codec]) || [];
const vsBase = (day) => (view.codec === ALL ? day.front_vs_base : day.codec_vs_base?.[view.codec]);
const vsPrevious = (day) => (view.codec === ALL ? day.front_vs_previous : day.codec_vs_previous?.[view.codec]);
const baseDay = (data) => (view.codec === ALL ? data.front_base_day : data.codec_front_base_days?.[view.codec]);

/** `hevc@fast` -> HEVC Fast; `hevc` (the Default) -> HEVC; `h264@retired-faster` ->
 * H.264 Faster (retired). */
function presetLabel(key) {
  const [codec, stem] = key.split('@');
  if (!stem) return CODEC_NAMES[codec];
  const retired = stem.startsWith(RETIRED);
  const name = stem.slice(retired ? RETIRED.length : 0);
  return `${CODEC_NAMES[codec]} ${STEM_NAMES[name] || name}${retired ? ' (retired)' : ''}`;
}

/** Our points of a day: the presets on the day's frontier, small and labelled; with `all`,
 * the others too, faint and unlabelled. A Default is ringed. */
function ourPoints(day, all = true) {
  const front = new Set(frontKeys(day));
  return Object.entries(day.values).filter(([k, v]) => inView(k) && v.j !== null && v.fps_x86 && (all || front.has(k))).map(([key, v]) => ({
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
  return frontKeys(day).map((k) => xy(day.values[k]));
}

/** Each codec's presets that day, slowest first: thin lines under the marks. */
function ladders(day) {
  return Object.keys(CODEC_NAMES).filter((codec) => view.codec === ALL || codec === view.codec).map((codec) => Object.entries(day.values)
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
  const ours = new Set(frontKeys(day).map((k) => `nanocodec-${k}`));
  const frontiers = [
    { kind: 'ref', line: frontierLine(refs), ids: new Set(refs.map((p) => p.id)) },
    { kind: 'ours bold', line: frontLine(day), ids: ours },
  ];

  fillHead(root, data, index);
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

/** "… points of BD-J at equal speed, …× speed at equal J" for a front change, or why not. */
function frontChange(c) {
  if (!c || (c.j_points === null && c.speed_factor === null)) return 'n/a (a front needs two presets and an overlap)';
  return (c.j_points !== null ? `${pct(c.j_points).replace('%', '')} points of BD-J at equal speed` : 'no overlap')
    + (c.speed_factor !== null ? `, ${c.speed_factor.toFixed(2)}× speed at equal J` : '');
}

/** The text above the chart for frame `index`: date and build, captions, git activity and
 * the front's change. */
function fillHead(root, data, index) {
  const day = data.days[index];
  setText(root, '.frame-day', `${day.date} · ${day.release ? `release ${day.release} · ` : ''}build ${day.commit}`);
  setChildren(root, '.frame-caption', (day.caption || []).map((line) => el('li', {}, line)));
  setText(root, '.frame-activity', activityText(day));
  const name = view.codec === ALL ? 'Front' : `${CODEC_NAMES[view.codec]} front`;
  const base = baseDay(data);
  setText(root, '.frame-front', `${name}: against the previous frame ${index ? frontChange(vsPrevious(day)) : '–'}; `
    + (base ? `against the ${base} front ${frontChange(vsBase(day))}.` : 'n/a (no day with two presets).'));
}

/** Reserve the tallest frame's text height above the chart (at the current width), so the
 * chart never moves between frames. */
function fixHeadHeight(root, data, index) {
  const head = root.querySelector('.frame-head');
  if (!head) return;
  head.style.minHeight = '';
  let tallest = 0;
  data.days.forEach((_, i) => { fillHead(root, data, i); tallest = Math.max(tallest, head.getBoundingClientRect().height); });
  head.style.minHeight = `${Math.ceil(tallest)}px`;
  fillHead(root, data, index);
}

function drawChange(root, data, index) {
  const base = baseDay(data);
  const who = view.codec === ALL ? '' : `${CODEC_NAMES[view.codec]}: `;
  setChildren(root, '.change-charts', [
    changePanel(data, index, `${who}BD-J gap to the ${base} front (points)`, (d) => vsBase(d)?.j_points, (v) => pct(v), 0),
    changePanel(data, index, `${who}speed factor at equal J vs the ${base} front`, (d) => vsBase(d)?.speed_factor, (v) => `${v.toFixed(2)}×`, 1)]);
  setText(root, '.change-note', 'Each front is BD-J as a piecewise-linear function of log fps; the J gap is the mean difference at equal speed over the speeds both fronts cover, the speed factor the mean at equal BD-J over the BD-J range both cover; nothing is extrapolated.');
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

/** The timing's own noise (history_chart.py noise_text says the same in the docs). */
function noiseText(noise) {
  if (!noise || noise.within_session_median_deviation_pct === null) return '';
  const moves = Object.values(noise.unchanged.fps_change_pct);
  const signed = (v) => `${v > 0 ? '+' : v < 0 ? '−' : ''}${Math.abs(v).toFixed(1)}%`;
  return `Each preset is timed in five repeats and the median kept; within a session the repeats deviate from their median by ${noise.within_session_median_deviation_pct.toFixed(1)}% (median over presets)`
    + (moves.length ? `; the ${moves.length} presets whose streams did not change from ${noise.unchanged.from} to ${noise.unchanged.to}, each timed in one session, moved by ${signed(Math.min(...moves))} to ${signed(Math.max(...moves))} in fps. ` : '. ');
}

function note(data) {
  const s = data.slice;
  return `Each frame is the last commit of that day on the integration branch, built as a plain release build and measured with one protocol when the frame was added (preset names follow later renames: 0.5.0's Slow is Slower, Veryslow on HEVC and VVC; the old H.264 Fast is Faster, and the old H.264 Faster, retired on 2026-11-21, ends there): `
    + `every speed-ladder preset that build shipped, with its own values (before 2026-11-15 the files <codec>-fast, -faster and -fastest are Fast, Faster and Veryfast; files off the ladder are left out), real-time CBR on the chart's ladder (${data.ladder_kbps.join('/')} kbps at 720p30 scaled to each clip, `
    + `plus ${data.low_ladder_kbps.join('/')} kbps on ${data.low_rung_clips.length} clips whose curves need them), ${data.frames} frames, one key frame, `
    + `our reconstruction scored. Bits: BD-J (PSNR-YUV, VMAF v1 and MS-SSIM in equal thirds, class-balanced) against ${data.zero_label} `
    + `on a fixed slice of ${s.clips.length} of the ${data.fresh_clips} held-out fresh clips (${Object.entries(s.per_class).map(([k, n]) => `${n} ${k}`).join(', ')}; `
    + `within a class the first by a hash of the name), averaged over the ${data.clips_used} clips on which every point of every day and every reference overlaps the zero line. `
    + `Speed: fps timed on AWS c8a (Zen 5), one core, by the chart's own timing: its commands (three 720p clips, fixed QP found per build for about ${data.timing_rung_kbps} kbps), `
    + `each run back to back with x264 medium, to 2026-11-20 one c8a.large per preset holding that preset's build of every one of those days (so their day-to-day speed differences are the same machine's), later frames in their own sessions; the median time ratio times the chart's stored x264 medium time; every day here is a plain release build (the overview's builds are PGO). `
    + `${noiseText(data.timing_noise)}${data.days.filter((d) => d.note).map((d) => `${d.date}: ${d.note} `).join('')}The references are the overview chart's, `
    + `their quality re-averaged on the same clips and their fps as timed on the c8a. `
    + `The bold line is our Pareto frontier that day over every codec and preset (no other of our points is both faster and lower); the faint blue lines are the earlier frames' frontiers, older ones fainter; `
    + `the marks on it are its presets, a ringed one a codec's Default. "Show every preset" adds the presets off the frontier, faint, with thin lines joining each codec's presets.`;
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
// A tick label needs about this many pixels; narrower, every other one is left out.
const TICK_PX = 48;

/** "2026-11-13" -> "Nov 13". */
function shortDate(date) {
  const [, month, day] = date.split('-').map(Number);
  return `${MONTHS[month - 1]} ${day}`;
}

/** Date labels under the slider at each frame's position, thinned to fit the width
 * (the first and last always shown). */
function drawTicks(root, data) {
  const ticks = root.querySelector('.day-ticks');
  if (!ticks) return;
  const n = data.days.length;
  const step = Math.max(1, Math.ceil((TICK_PX * n) / Math.max(1, ticks.clientWidth)));
  ticks.replaceChildren(...data.days.map((d, i) => {
    // the last always; a thinned label too close before it is left out
    const shown = i === n - 1 || (i % step === 0 && n - 1 - i >= step);
    return el('span', { style: `left: ${n > 1 ? (i / (n - 1)) * 100 : 0}%`, class: shown ? '' : 'thinned' }, shortDate(d.date));
  }));
}

/** data/history.json, at the version this module was published with. */
async function loadHistory() {
  const response = await fetch(`data/history.json${VERSION ? `?v=${VERSION}` : ''}`);
  if (!response.ok) throw new Error(`data/history.json: HTTP ${response.status}`);
  return response.json();
}

async function main() {
  const root = document.getElementById('history');
  try {
    const data = await loadHistory();
    const all = [...referencePoints(data), ...data.days.flatMap(ourPoints)];
    const domain = plotDomain(all);
    const params = new URLSearchParams(location.search);
    const still = params.has('still');
    // the axes are the whole front's, so every codec view shares them
    const requested = params.get('codec');
    view.codec = requested in CODEC_NAMES ? requested : ALL;
    let index = Math.min(Number(params.get('frame') ?? data.days.length - 1), data.days.length - 1);
    setText(root, '.view-note', note(data));

    const slider = root.querySelector('input[type=range]');
    const button = root.querySelector('.play');
    slider.max = String(data.days.length - 1);
    slider.value = String(index);
    let timer = null;
    const show = (i) => {
      index = i;
      slider.value = String(i);
      setText(root, '.day-label', `${data.days[i].date} · frame ${i + 1} of ${data.days.length}`);
      draw(root, data, i, domain);
    };
    drawTicks(root, data);
    const stop = () => { clearInterval(timer); timer = null; button.textContent = 'Play'; button.setAttribute('aria-pressed', 'false'); };
    // plays on a loop: after the last day it starts again from the first
    const play = () => {
      button.textContent = 'Pause';
      button.setAttribute('aria-pressed', 'true');
      timer = setInterval(() => show(index >= data.days.length - 1 ? 0 : index + 1), FRAME_MS);
    };
    button.addEventListener('click', () => { if (timer) stop(); else play(); });
    slider.addEventListener('input', () => { stop(); show(Number(slider.value)); });
    root.querySelector('.all-toggle')?.addEventListener('change', () => draw(root, data, index, domain));
    const codecs = [{ value: ALL, label: 'All' }, ...Object.entries(CODEC_NAMES).map(([value, label]) => ({ value, label }))];
    root.querySelector('.controls')?.prepend(control('Front of', segmented('Front of', codecs, view.codec, (value) => {
      view.codec = value;
      const url = new URL(location.href);
      if (value === ALL) url.searchParams.delete('codec'); else url.searchParams.set('codec', value);
      history.replaceState(null, '', url);
      fixHeadHeight(root, data, index);
      draw(root, data, index, domain);
    })));
    if (still) document.body.classList.add('still');
    else root.querySelector('.table-slot')?.append(dayTable(data));
    fixHeadHeight(root, data, index);
    show(index);
    // autoplay unless a single frame was asked for (?frame=N or ?still)
    if (!still && !params.has('frame')) play();
    window.addEventListener('resize', () => { fixHeadHeight(root, data, index); drawTicks(root, data); draw(root, data, index, domain); });
    document.body.dataset.ready = 'true';
  } catch (error) {
    fail(root, error);
  }
}

main();
