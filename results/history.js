// History (history.html): our presets and their Pareto frontier, one frame per day's build,
// from the BP-4 history report of the selected timing machine (procedure section 14), with the
// reference encoders of the same report as fixed points; CBR only. The form is the earlier
// history page's (web/results/history.js): fixed axes, earlier frames' frontiers fading,
// play / pause and a day slider. The machine selector and the key-interval control switch
// between the report's frame lists. Data: data/history.json (tools/bench/preview_history.py).
import { paretoFront, frontierLine } from './frontier.js';
import { drawScatter } from './chart.js';
import { plotDomain } from './domain.js';
import { CODEC_NAMES, control, el, fail, loadData, pct, segmented, table } from './common.js';
import { SPEED_UNIT, chosenMachine, machinePicker, speed, zeroName } from './bp3.js';

const SHAPES = {
  nanocodec: 'circle', x264: 'square', x265: 'diamond', 'SVT-AV1': 'triangle', openh264: 'triangle-down',
  libvpx: 'cross', libaom: 'plus', Kvazaar: 'pentagon', VVenC: 'hexagon',
};
const FRAME_MS = 700;
const GHOST_OPACITY = 0.7;
const GHOST_FADE = 0.6;
const ALL = 'all';
const state = { codec: ALL, index: 0, timer: null, machine: null, key: null };

const viewOf = (data) => data.views[state.machine];
const framesOf = (data) => viewOf(data).series[state.key];
const inCodec = (key) => state.codec === ALL || key.split('@')[0] === state.codec;
const presetLabel = (key) => {
  const [codec, preset] = key.split('@');
  return `${CODEC_NAMES[codec]} ${preset[0].toUpperCase()}${preset.slice(1)}`;
};

/** Our points of a frame in the current view, as {key, x, y}. */
function ours(frame) {
  return Object.entries(frame.view.values).filter(([k, v]) => inCodec(k) && v.j !== null && v.speed)
    .map(([key, v]) => ({ key, x: v.speed, y: v.j, v }));
}

const frontOf = (frame) => paretoFront(ours(frame));

function ourPoints(frame, all) {
  const front = new Set(frontOf(frame).map((p) => p.key));
  return ours(frame).filter((p) => all || front.has(p.key)).map((p) => ({
    id: `nanocodec-${p.key}`, x: p.x, y: p.y, iqr: front.has(p.key) && p.v.low !== null ? [p.v.low, p.v.high] : null,
    kind: 'ours', shape: 'circle', label: front.has(p.key) ? presetLabel(p.key) : '', priority: front.has(p.key) ? 3 : 2,
    classes: front.has(p.key) ? '' : 'off-front', scale: front.has(p.key) ? 0.8 : 0.6,
    aria: `${presetLabel(p.key)}, ${frame.label}: ${pct(p.y)}, ${speed(p.x)} ${SPEED_UNIT}`,
  }));
}

function referencePoints(data) {
  return Object.entries(viewOf(data).references).filter(([, r]) => r.j !== null && r.speed).map(([family, r]) => ({
    id: family, x: r.speed, y: r.j, iqr: null, kind: 'ref', shape: SHAPES[r.owner] || 'circle', label: r.label, priority: 1,
    aria: `${r.label}: ${pct(r.j)} on the common set of ${r.clips} clips, ${speed(r.speed)} ${SPEED_UNIT}`,
  }));
}

function ghosts(frames, index) {
  return frames.slice(0, index).map((f, i) => ({
    line: frontierLine(frontOf(f)), cls: 'ghost-front', opacity: (GHOST_OPACITY * GHOST_FADE ** (index - 1 - i)).toFixed(3),
  }));
}

function ladders(frame) {
  return Object.keys(CODEC_NAMES).filter((c) => state.codec === ALL || c === state.codec)
    .map((codec) => ({ cls: 'ladder', line: ours(frame).filter((p) => p.key.split('@')[0] === codec).sort((a, b) => a.x - b.x).map((p) => [p.x, p.y]) }));
}

const rungsText = (frame) => Object.entries(frame.rungs).map(([codec, rungs]) => `${CODEC_NAMES[codec]} (${rungs.join(', ')})`).join('; ');
const notRecorded = () => el('strong', { class: 'not-recorded' }, 'not recorded');
const complianceText = (view) => (view?.compliance?.length ? view.compliance.map((c) => `${c.label} ${c.value}`).join('; ') : null);

function fillHead(root, data, frame) {
  const { view } = frame;
  root.querySelector('.zero-name').textContent = zeroName(viewOf(data), 'j');
  root.querySelector('.frame-day').replaceChildren(
    `${frame.label} · build `, frame.commit ? el('code', {}, frame.commit) : notRecorded(),
    ` · CBR, ${data.keys[state.key]} · common set of ${view.clips} history clips`);
  root.querySelector('.frame-presets').textContent = `Codecs and presets in this build: ${rungsText(frame)}.`;
  const compliance = complianceText(view);
  root.querySelector('.frame-compliance').replaceChildren('Measured compliance, this build: ', compliance || notRecorded(), '.');
  root.querySelector('.frame-caption').replaceChildren(...(frame.caption || []).map((line) => el('li', {}, line)));
}

/** Reserve the tallest frame's text height above the chart, over every frame at the current
 * width, so the chart never moves between frames. */
function fixHeadHeight(root, data) {
  const head = root.querySelector('.frame-head');
  head.style.minHeight = '';
  let tallest = 0;
  for (const frame of framesOf(data)) {
    fillHead(root, data, frame);
    tallest = Math.max(tallest, head.getBoundingClientRect().height);
  }
  head.style.minHeight = `${Math.ceil(tallest)}px`;
}

function draw(root, data, domain) {
  const frames = framesOf(data);
  const frame = frames[state.index];
  const all = root.querySelector('.all-toggle').checked;
  const points = [...referencePoints(data), ...ourPoints(frame, all)];
  const refs = paretoFront(points.filter((p) => p.kind !== 'ours'));
  const front = frontOf(frame);
  fillHead(root, data, frame);
  root.querySelector('.day-label').textContent = `${frame.label} · frame ${state.index + 1} of ${frames.length}`;
  drawScatter(root.querySelector('.chart'), points, {
    xLabel: `encode speed, ${SPEED_UNIT}, ${data.machines[state.machine].name}, half load (log scale; faster →)`,
    yLabel: 'BD-J, % bits for the same quality (lower is fewer bits)',
    describe: (p) => el('div', {}, el('strong', {}, p.label || p.id), el('div', {}, p.aria)), onSelect: () => {},
    frontiers: [
      { kind: 'ref', line: frontierLine(refs), ids: new Set(refs.map((p) => p.id)) },
      { kind: 'ours bold', line: frontierLine(front), ids: new Set(front.map((p) => `nanocodec-${p.key}`)) },
    ],
    lines: [...ghosts(frames, state.index), ...(all ? ladders(frame) : [])], domain,
    ariaLabel: `Bits against speed, ${frame.label}`,
  });
}

/** Fixed axes over every frame, so the chart does not move between frames. */
function domainOf(data) {
  const points = [...referencePoints(data), ...framesOf(data).flatMap((f) => { const c = state.codec; state.codec = ALL; const p = ourPoints(f, true); state.codec = c; return p; })];
  return points.length ? plotDomain(points) : null;
}

function dayTable(data) {
  const front = (frame) => frontOf(frame).map((p) => `${presetLabel(p.key)} ${pct(p.y)} at ${speed(p.x)}`).join('; ');
  return table([
    { label: 'Frame', cell: (f) => f.label },
    { label: 'Build', cell: (f) => (f.commit ? el('code', {}, f.commit) : notRecorded()) },
    { label: 'Codecs and presets', cell: (f) => rungsText(f) },
    { label: `Front, slowest first (BD-J at ${SPEED_UNIT})`, cell: front },
    { label: 'Compliance', cell: (f) => complianceText(f.view) || notRecorded() },
    { label: 'Main changes since the previous frame', cell: (f) => (f.caption || []).join('; ') },
  ], framesOf(data));
}

/** The fixed reference points: BD-J on the common set, speed. */
function referenceTable(data) {
  const view = viewOf(data);
  return el('div', {},
    el('h3', {}, 'References: the same report, on the same common set'),
    el('p', { class: 'note' }, `${view.view_clips} history clip windows have a well-formed zero line; every family, ours of every day and the references, has well-formed curves on ${view.common_clips} of them (each clip's 150- and 600-frame windows, counted once per clip in the BDs).`),
    table([
      { label: 'Reference', cell: ([, r]) => r.label },
      { label: 'BD-J (clips)', num: true, cell: ([, r]) => (r.j !== null ? `${pct(r.j)} (${r.clips})` : '–') },
      { label: SPEED_UNIT, num: true, cell: ([, r]) => speed(r.speed) },
    ], Object.entries(view.references)));
}

function setTicks(root, frames) {
  const n = frames.length;
  root.querySelector('.day-ticks').replaceChildren(...frames.map((f, i) => el('span', {
    style: `left: ${n > 1 ? (i / (n - 1)) * 100 : 0}%`, class: i === 0 || i === n - 1 || i % 3 === 0 ? '' : 'thinned',
  }, f.tick)));
}

/** The animation: controls, slider, play / pause and the tables; returns its stop(). */
function animate(root, data) {
  const slider = root.querySelector('input[type=range]');
  const button = root.querySelector('.play');
  let domain = domainOf(data);
  const reset = (index) => {
    const frames = framesOf(data);
    state.index = Math.min(index ?? frames.length - 1, frames.length - 1);
    slider.max = String(frames.length - 1);
    slider.value = String(state.index);
    setTicks(root, frames);
    domain = domainOf(data);
    draw(root, data, domain);
  };
  const show = (i) => { state.index = i; slider.value = String(i); draw(root, data, domain); };
  const stop = () => { clearInterval(state.timer); state.timer = null; button.textContent = 'Play'; button.setAttribute('aria-pressed', 'false'); };
  const play = () => {
    button.textContent = 'Pause'; button.setAttribute('aria-pressed', 'true');
    state.timer = setInterval(() => show(state.index >= framesOf(data).length - 1 ? 0 : state.index + 1), FRAME_MS);
  };
  button.addEventListener('click', () => (state.timer ? stop() : play()));
  slider.addEventListener('input', () => { stop(); show(Number(slider.value)); });
  root.querySelector('.all-toggle').addEventListener('change', () => draw(root, data, domain));
  root.querySelector('.controls').prepend(
    control('Key frames', segmented('Key frames', Object.entries(data.keys).map(([value, label]) => ({ value, label })),
      state.key, (v) => { state.key = v; refresh(); })),
    control('Front of', segmented('Front of', [{ value: ALL, label: 'All' }, ...Object.entries(CODEC_NAMES).map(([value, label]) => ({ value, label }))],
      state.codec, (v) => { state.codec = v; draw(root, data, domain); })));
  // a new machine or key interval: its own frames, axes, tables and text height; from the last frame
  const refresh = () => {
    root.querySelector('.table-slot').replaceChildren(dayTable(data), referenceTable(data));
    fixHeadHeight(root, data);
    reset();
  };
  root.querySelector('.table-slot').append(dayTable(data), referenceTable(data));
  fixHeadHeight(root, data);
  const params = new URLSearchParams(location.search);
  reset(params.has('frame') ? Number(params.get('frame')) : undefined);
  if (!params.has('frame')) play();
  window.addEventListener('resize', () => {
    if (root.hidden) return;
    fixHeadHeight(root, data);
    draw(root, data, domain);
  });
  return { stop, refresh };
}

const points = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}`;

/** The mean offset of our families (ours) or of the references. */
function meanOf(offsets, ours) {
  const values = Object.entries(offsets).filter(([f]) => f.startsWith('nanocodec-') === ours).map(([, o]) => o);
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** The warning beside a graph that fails the tracking check (V17), with its numbers. */
function trackingWarning(check) {
  const outside = Object.entries(check.outside);
  return [
    el('strong', {}, 'Read this graph for the trend across days, not the level. '),
    `It fails the procedure's tracking check (V17, section 14): the history clips do not reproduce the full benchmark's level, and the cause is not yet established. `
    + `On the current build, each family's BD-J on the history common set (${check.subset_clips} clip-windows) minus its BD-J on the full "ours" set (${check.full_clips}) must lie within ${check.limit} point of the median offset (${points(check.median_offset)} points); `
    + `${outside.length} of ${check.families} families lie outside it${check.front_same ? '' : ', and the front differs'}. `
    + `Mean offset: our families ${points(meanOf(check.offsets, true))}, the references ${points(meanOf(check.offsets, false))} (positive: the subset reads the family worse). `,
    el('details', {}, el('summary', {}, 'Every family\'s offset (points of BD-J, subset minus full set)'),
      table([
        { label: 'Family', cell: ([f]) => f },
        { label: 'Offset', num: true, cell: ([, o]) => points(o) },
        { label: 'From the median', num: true, cell: ([, o]) => points(o - check.median_offset) },
        { label: 'Within the limit', cell: ([f]) => (f in check.outside ? 'no' : 'yes') },
      ], Object.entries(check.offsets))),
    el('p', { class: 'note' }, `Data: ${check.source}`),
  ];
}

async function main() {
  const root = document.getElementById('history');
  try {
    const data = await loadData('history');
    const procedures = [...new Set(Object.values(data.views).map((v) => v.procedure || 'not recorded'))].join(', ');
    document.querySelector('.results-banner').append(`Benchmark procedure ${procedures} (the history reports' record; current: ${data.current_procedure}).`);
    document.querySelector('.history-notes').replaceChildren(...data.notes.map((n) => el('li', {}, n)));
    const absent = document.querySelector('.history-absent');
    state.key = data.primary_key;
    let animation = null;
    const pick = (machine) => {
      state.machine = machine;
      const shown = Boolean(data.views[machine]);
      const check = shown ? data.views[machine].tracking_check : null;
      const warn = Boolean(check && !check.passed);
      root.hidden = !shown;
      absent.hidden = shown && !warn;
      absent.replaceChildren(...(!shown ? [`No history report for ${data.machines[machine].name} yet; this page fills in when it arrives.`]
        : warn ? trackingWarning({ ...check, source: data.views[machine].source }) : []));
      if (!shown) animation?.stop();
      else if (animation) animation.refresh();
      else animation = animate(root, data);
      document.querySelector('.history-source').textContent = shown ? `Data: ${data.views[machine].source}` : '';
    };
    machinePicker(data.machines, chosenMachine(data.machines), pick);
    pick(chosenMachine(data.machines));
    document.body.dataset.ready = 'true';
  } catch (error) {
    fail(root, error);
  }
}

main();
