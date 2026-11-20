// A small dependency-free SVG scatter chart for the results pages: log-scale
// speed across, BD-rate up (lower is better), per-clip spread as vertical bars,
// a marker shape per project, direct labels where they fit, and a tooltip and a
// selection per point (pointer, touch and keyboard). The axes cover the bulk of the
// points (domain.js); a point beyond them is pinned to the plot edge with an arrow
// pointing off the chart, and lines running to it are clipped at the edge. It draws
// what it is given; the pages decide which points and values.
import { plotDomain } from './domain.js';

const SVG = 'http://www.w3.org/2000/svg';
const MARGIN = { top: 14, right: 18, bottom: 46, left: 56 };
const LABEL_CHAR_PX = 6.4; // average label glyph width at the chart's label size
// Label distances tried beyond the point's own side, in label heights.
const LEADER_RINGS = [0, 1, 2, 3];
const LABEL_HEIGHT = 13;
const HIT_RADIUS = 14; // generous hit target around an 8-12 px marker
const LOG_TICK_STEPS = [1, 2, 5];
const LOG_NARROW_RATIO = 5;
// A pinned point's centre sits this far inside the plot edge, leaving room for its arrow.
const PIN_INSET = 16;
const PIN_ARROW_GAP = 5; // from the marker's edge to the arrow
const PIN_ARROW_SIZE = 5; // the chevron's half-height (and about its reach)
const PIN_ARROW = 'M-3.5,-5L1.5,0L-3.5,5'; // a chevron pointing along +x
// Charts drawn so far, numbering each one's clip path (several charts share a page).
let clipCount = 0;

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== null && v !== undefined) node.setAttribute(k, v);
  return node;
}

/** The path of a marker shape of radius r centred on (0, 0). */
export function shapePath(shape, r) {
  const t = r * 1.15;
  switch (shape) {
    case 'square': return `M${-r},${-r}h${2 * r}v${2 * r}h${-2 * r}z`;
    case 'diamond': return `M0,${-t}L${t},0L0,${t}L${-t},0z`;
    case 'triangle': return `M0,${-t}L${t},${t * 0.8}L${-t},${t * 0.8}z`;
    case 'triangle-down': return `M0,${t}L${t},${-t * 0.8}L${-t},${-t * 0.8}z`;
    case 'cross': {
      const a = r * 0.42;
      return `M${-r},${-r + a}L${-a},0L${-r},${r - a}L${-r + a},${r}L0,${a}L${r - a},${r}L${r},${r - a}L${a},0L${r},${-r + a}L${r - a},${-r}L0,${-a}L${-r + a},${-r}z`;
    }
    case 'plus': {
      const a = r * 0.36;
      return `M${-a},${-r}h${2 * a}v${r - a}h${r - a}v${2 * a}h${a - r}v${r - a}h${-2 * a}v${a - r}h${a - r}v${-2 * a}h${r - a}z`;
    }
    default: return `M${r},0A${r},${r} 0 1,1 ${-r},0A${r},${r} 0 1,1 ${r},0z`;
  }
}

/** A small legend swatch of a marker. */
export function swatch(shape, kind) {
  const node = svg('svg', { viewBox: '-8 -8 16 16', 'aria-hidden': 'true' });
  node.append(svg('path', { d: shapePath(shape, 6), class: `mark ${kind}` }));
  return node;
}

/** A legend swatch of the arrow beside a point pinned to the edge. */
export function pinSwatch() {
  const node = svg('svg', { viewBox: '-5 -5 10 10', 'aria-hidden': 'true' });
  node.append(svg('path', { d: PIN_ARROW }));
  return node;
}

function niceStep(span, target) {
  const raw = span / target;
  const power = 10 ** Math.floor(Math.log10(raw));
  const unit = [1, 2, 2.5, 5, 10].find((m) => m * power >= raw) * power;
  return unit;
}

function logTicks(lo, hi) {
  const ticks = [];
  // A narrow range (a few points) gets every integer multiple, so it keeps several ticks.
  const steps = hi / lo < LOG_NARROW_RATIO ? [1, 2, 3, 4, 5, 6, 7, 8, 9] : LOG_TICK_STEPS;
  for (let p = Math.floor(Math.log10(lo)); p <= Math.ceil(Math.log10(hi)); p++) {
    for (const m of steps) {
      const v = m * 10 ** p;
      if (v >= lo && v <= hi) ticks.push(v);
    }
  }
  return ticks;
}

/** A chevron pointing off the chart from a pinned marker of radius r: `dx` and `dy` are
 * -1, 0 or 1 per axis (screen directions), so a corner point points diagonally. */
function pinArrow(dx, dy, r) {
  const angle = (Math.atan2(dy, dx) * 180) / Math.PI;
  return svg('path', { class: 'pin-arrow', d: PIN_ARROW, transform: `rotate(${angle}) translate(${r + PIN_ARROW_GAP} 0)` });
}

/**
 * Draw `points` into `container`:
 *   points: [{id, x, y, iqr: [lo, hi] | null, kind: 'ours'|'ref'|'tuned', shape, label,
 *             flagged, twin: id | null, priority}]
 *   opts: {xLabel, yLabel, describe(point) -> Node (tooltip), onSelect(point), selected,
 *          frontiers: [{kind: 'ours'|'ref', line: [[x, y], ...], ids: Set}] (Pareto, frontier.js),
 *          lines: [{line: [[x, y], ...], cls, opacity?}] (extra polylines under the marks: the
 *            history page's preset ladders and earlier days' frontiers),
 *          domain: {x: [lo, hi], y: [lo, hi]} (fixed axes across redraws; default plotDomain)}
 * Returns {pinned: [id, ...]}, the points beyond the axes, so the page can say so.
 */
export function drawScatter(container, points, opts) {
  const width = Math.max(300, container.clientWidth);
  const narrow = width < 560;
  const height = Math.round(Math.min(640, Math.max(330, width * (narrow ? 1.3 : 0.6))));
  const radius = narrow ? 5 : 6;
  const plotW = width - MARGIN.left - MARGIN.right;
  const plotH = height - MARGIN.top - MARGIN.bottom;

  const domain = opts.domain || plotDomain(points);
  const ys = [...domain.y, 0];
  const xLo = domain.x[0] * 0.82;
  const xHi = domain.x[1] * 1.22;
  const yStep = niceStep(Math.max(...ys) - Math.min(...ys), narrow ? 5 : 8);
  const yLo = Math.floor(Math.min(...ys) / yStep) * yStep;
  const yHi = Math.ceil(Math.max(...ys) / yStep) * yStep;
  const sx = (v) => MARGIN.left + ((Math.log10(v) - Math.log10(xLo)) / (Math.log10(xHi) - Math.log10(xLo))) * plotW;
  const sy = (v) => MARGIN.top + ((yHi - v) / (yHi - yLo)) * plotH;
  // Where each point is drawn: in place, or pinned inside the edge it lies beyond
  // (out: the screen direction off the chart, per axis).
  const at = new Map(points.map((p) => {
    const outX = p.x < xLo ? -1 : p.x > xHi ? 1 : 0;
    const outY = p.y > yHi ? -1 : p.y < yLo ? 1 : 0;
    const edge = (out, lo, hi, v) => (out < 0 ? lo + PIN_INSET : out > 0 ? hi - PIN_INSET : v);
    return [p.id, {
      x: edge(outX, MARGIN.left, width - MARGIN.right, sx(p.x)),
      y: edge(outY, MARGIN.top, height - MARGIN.bottom, sy(p.y)),
      out: outX || outY ? [outX, outY] : null,
    }];
  }));

  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'group', 'aria-label': opts.ariaLabel });
  // Lines (frontiers, spread bars, twin links) keep their true geometry and stop at the plot edge.
  clipCount += 1;
  const clipId = `plot-clip-${clipCount}`;
  const clip = svg('clipPath', { id: clipId });
  clip.append(svg('rect', { x: MARGIN.left, y: MARGIN.top, width: plotW, height: plotH }));
  const defs = svg('defs');
  defs.append(clip);
  root.append(defs);
  const clipped = (cls) => svg('g', { class: cls, 'clip-path': `url(#${clipId})` });
  const grid = svg('g', { class: 'grid' });
  for (let v = yLo; v <= yHi + 1e-9; v += yStep) {
    grid.append(svg('line', { x1: MARGIN.left, x2: width - MARGIN.right, y1: sy(v), y2: sy(v) }));
    const t = svg('text', { x: MARGIN.left - 8, y: sy(v) + 4, 'text-anchor': 'end' });
    t.textContent = v === 0 ? '0' : `${v > 0 ? '+' : '−'}${Math.abs(v)}%`;
    grid.append(t);
  }
  for (const v of logTicks(xLo, xHi)) {
    grid.append(svg('line', { x1: sx(v), x2: sx(v), y1: MARGIN.top, y2: height - MARGIN.bottom }));
    const t = svg('text', { x: sx(v), y: height - MARGIN.bottom + 16, 'text-anchor': 'middle' });
    t.textContent = String(v);
    grid.append(t);
  }
  root.append(grid);
  root.append(svg('line', { class: 'zero-line', x1: MARGIN.left, x2: width - MARGIN.right, y1: sy(0), y2: sy(0) }));
  const xTitle = svg('text', { class: 'axis-title', x: MARGIN.left + plotW / 2, y: height - 8, 'text-anchor': 'middle' });
  xTitle.textContent = opts.xLabel;
  const yTitle = svg('text', { class: 'axis-title', transform: `translate(14 ${MARGIN.top + plotH / 2}) rotate(-90)`, 'text-anchor': 'middle' });
  yTitle.textContent = opts.yLabel;
  root.append(xTitle, yTitle);

  const byId = new Map(points.map((p) => [p.id, p]));
  const links = clipped('links');
  for (const p of points) {
    const twin = p.twin && byId.get(p.twin);
    if (twin) links.append(svg('line', { class: 'twin-link', x1: sx(p.x), y1: sy(p.y), x2: sx(twin.x), y2: sy(twin.y) }));
  }
  root.append(links);

  // Pareto frontiers, under the marks: a line through each group's frontier points.
  const fronts = clipped('frontiers');
  for (const f of opts.frontiers || []) {
    if (f.line.length > 1) {
      fronts.append(svg('polyline', { class: `frontier ${f.kind}`, points: f.line.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ') }));
    }
  }
  root.append(fronts);
  const extra = clipped('extra-lines');
  for (const { line, cls, opacity } of opts.lines || []) {
    if (line.length > 1) extra.append(svg('polyline', { class: cls, opacity, points: line.map(([x, y]) => `${sx(x)},${sy(y)}`).join(' ') }));
  }
  root.append(extra);
  const onFront = new Set((opts.frontiers || []).flatMap((f) => [...f.ids]));

  const bars = clipped('bars');
  for (const p of points) {
    if (!p.iqr || at.get(p.id).out) continue; // a pinned point's spread would sit off its marker
    bars.append(svg('line', { class: `iqr ${p.kind === 'ours' ? 'ours' : 'ref'}`, x1: sx(p.x), x2: sx(p.x), y1: sy(p.iqr[0]), y2: sy(p.iqr[1]) }));
  }
  root.append(bars);

  // Labels: the highest-priority points first, each at the first free side.
  const placed = points.map((p) => {
    const { x, y, out } = at.get(p.id);
    // a pinned point's arrow, on its off-chart side, keeps labels off it too
    const [dx, dy] = (out || [0, 0]).map((d) => d * (radius + PIN_ARROW_GAP + PIN_ARROW_SIZE));
    return { x0: x - radius + Math.min(dx, 0), x1: x + radius + Math.max(dx, 0), y0: y - radius + Math.min(dy, 0), y1: y + radius + Math.max(dy, 0) };
  });
  // The zero line keeps its labels off it, so it stays readable.
  placed.push({ x0: MARGIN.left, x1: width - MARGIN.right, y0: sy(0) - 2, y1: sy(0) + 2 });
  const labels = svg('g');
  const overlaps = (b) => placed.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  for (const p of [...points].sort((a, b) => b.priority - a.priority)) {
    if (!p.label) continue; // a point drawn without a label (the history page's off-frontier presets)
    const w = p.label.length * LABEL_CHAR_PX;
    const { x: cx, y: cy } = at.get(p.id);
    // The four sides next to the point, then the same farther out on a leader line:
    // a crowded cluster (the slow rungs beside the Defaults) used to drop labels.
    const sidesAt = (gap) => [
      { x: cx + gap, y: cy + 4, anchor: 'start', box: { x0: cx + gap, x1: cx + gap + w, y0: cy - 8, y1: cy + 5 } },
      { x: cx - gap, y: cy + 4, anchor: 'end', box: { x0: cx - gap - w, x1: cx - gap, y0: cy - 8, y1: cy + 5 } },
      { x: cx, y: cy - gap - 2, anchor: 'middle', box: { x0: cx - w / 2, x1: cx + w / 2, y0: cy - gap - LABEL_HEIGHT, y1: cy - gap } },
      { x: cx, y: cy + gap + 10, anchor: 'middle', box: { x0: cx - w / 2, x1: cx + w / 2, y0: cy + gap, y1: cy + gap + LABEL_HEIGHT } },
    ];
    const near = radius + 4;
    const sides = LEADER_RINGS.flatMap((ring) => sidesAt(near + ring * LABEL_HEIGHT));
    const inside = (b) => b.x0 >= 2 && b.x1 <= width - 2 && b.y0 >= 0 && b.y1 <= height - MARGIN.bottom;
    const side = sides.find((s) => inside(s.box) && !overlaps(s.box))
      // our own points are always labelled, overlapping if nothing is free
      || (p.kind === 'ours' ? sides.find((s) => inside(s.box)) : null);
    if (!side) continue;
    placed.push(side.box);
    if (Math.hypot(side.x - cx, side.y - cy) > near + LABEL_HEIGHT) {
      labels.append(svg('line', { class: 'leader', x1: cx, y1: cy, x2: side.x, y2: side.y - 4 }));
    }
    const t = svg('text', { class: `point-label ${p.kind === 'ours' ? 'ours' : ''}`, x: side.x, y: side.y, 'text-anchor': side.anchor });
    t.textContent = p.label;
    labels.append(t);
  }
  root.append(labels);

  const marks = svg('g');
  const tooltip = container.parentElement.querySelector('.tooltip');
  const show = (p, node) => {
    tooltip.replaceChildren(opts.describe(p));
    tooltip.hidden = false;
    const card = container.parentElement.getBoundingClientRect();
    const box = node.getBoundingClientRect();
    const left = Math.min(box.left - card.left + 16, card.width - tooltip.offsetWidth - 6);
    const top = box.top - card.top - tooltip.offsetHeight - 10;
    tooltip.style.left = `${Math.max(6, left)}px`;
    tooltip.style.top = `${top < 4 ? box.bottom - card.top + 10 : top}px`;
  };
  const hide = () => { tooltip.hidden = true; };
  for (const p of [...points].sort((a, b) => a.priority - b.priority)) {
    const { x, y, out } = at.get(p.id);
    const g = svg('g', { class: `point${opts.selected === p.id ? ' active' : ''}${out ? ' pinned' : ''}`, transform: `translate(${x} ${y})` });
    const cls = `mark ${p.kind}${p.flagged ? ' flagged' : ''}${p.kind === 'tuned' ? ' ref' : ''}${onFront.has(p.id) ? ' on-front' : ''}${p.classes ? ` ${p.classes}` : ''}`;
    const r = (p.kind === 'ours' ? radius + 0.5 : radius) * (p.scale || 1);
    g.append(svg('path', { d: shapePath(p.shape, r), class: cls }));
    if (out) g.append(pinArrow(out[0], out[1], r));
    const aria = out ? `${p.aria} (beyond the axes, pinned to the edge)` : p.aria;
    const hit = svg('circle', { r: HIT_RADIUS, class: 'hit', tabindex: 0, role: 'button', 'aria-label': aria });
    hit.addEventListener('pointerenter', () => show(p, hit));
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('focus', () => show(p, hit));
    hit.addEventListener('blur', hide);
    hit.addEventListener('click', () => opts.onSelect(p));
    hit.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); opts.onSelect(p); }
    });
    g.append(hit);
    marks.append(g);
  }
  root.append(marks);
  container.replaceChildren(root);
  hide();
  return { pinned: points.filter((p) => at.get(p.id).out).map((p) => p.id) };
}

const BAR = { height: 18, gap: 6, right: 56, top: 24, bottom: 30 };

/**
 * Horizontal bars of time ratios (ours ÷ reference) on a log scale around 1:
 * rows [{label, ratio, tip}], a bar from 1 to each ratio, its value at the end,
 * the tip as the bar's tooltip. Rows without a ratio are listed as missing.
 */
export function drawRatioBars(container, rows, { axisLabel, missingText }) {
  const width = Math.max(300, container.clientWidth);
  const longest = Math.max(...rows.map((r) => r.label.length)) * LABEL_CHAR_PX + 16;
  const labelWidth = Math.min(longest, width * (width < 560 ? 0.5 : 0.4));
  const height = BAR.top + rows.length * (BAR.height + BAR.gap) + BAR.bottom;
  const ratios = rows.map((r) => r.ratio).filter((r) => r !== null);
  const lo = Math.min(0.5, ...ratios) * 0.9;
  const hi = Math.max(2, ...ratios) * 1.1;
  const plotW = width - labelWidth - BAR.right;
  const sx = (v) => labelWidth + ((Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo))) * plotW;
  const root = svg('svg', { viewBox: `0 0 ${width} ${height}`, width, height, role: 'img', 'aria-label': axisLabel });
  const grid = svg('g', { class: 'grid' });
  for (const v of [0.25, 0.5, 1, 2, 4].filter((t) => t >= lo && t <= hi)) {
    grid.append(svg('line', { x1: sx(v), x2: sx(v), y1: BAR.top - 6, y2: height - BAR.bottom }));
    const t = svg('text', { x: sx(v), y: height - BAR.bottom + 16, 'text-anchor': 'middle' });
    t.textContent = `${v}×`;
    grid.append(t);
  }
  root.append(grid);
  const title = svg('text', { class: 'axis-title', x: width / 2, y: 14, 'text-anchor': 'middle' });
  title.textContent = axisLabel;
  root.append(title);
  rows.forEach((r, i) => {
    const y = BAR.top + i * (BAR.height + BAR.gap);
    const name = svg('text', { class: 'point-label', x: labelWidth - 8, y: y + BAR.height * 0.7, 'text-anchor': 'end' });
    const fits = Math.floor((labelWidth - 16) / LABEL_CHAR_PX);
    name.textContent = r.label.length > fits ? `${r.label.slice(0, fits - 1)}…` : r.label;
    root.append(name);
    if (r.ratio === null) {
      const t = svg('text', { x: sx(1) + 6, y: y + BAR.height * 0.7 });
      t.textContent = missingText;
      root.append(t);
      return;
    }
    const x0 = Math.min(sx(1), sx(r.ratio));
    const bar = svg('rect', { class: `mark ${r.ratio <= 1 ? 'ours' : 'ref'}`, x: x0, y, width: Math.max(2, Math.abs(sx(r.ratio) - sx(1))), height: BAR.height, rx: 3, tabindex: 0 });
    const tip = svg('title');
    tip.textContent = r.tip;
    bar.append(tip);
    const value = svg('text', { class: 'point-label', x: Math.max(sx(1), sx(r.ratio)) + 6, y: y + BAR.height * 0.7 });
    value.textContent = r.ratio.toFixed(2);
    root.append(bar, value);
  });
  root.append(svg('line', { class: 'zero-line', x1: sx(1), x2: sx(1), y1: BAR.top - 6, y2: height - BAR.bottom }));
  container.replaceChildren(root);
}
