// The plotted range of a results scatter (chart.js): the span of the bulk of the
// points, so that one very slow encoder (HM at under 1 fps) or one very poor one
// (x264 ultrafast at +150 %) does not squash the rest into a corner. Points beyond
// it are pinned to the plot edge by the chart. Pure, so web/tests/domain.test.js
// checks it under node.
import { paretoFront } from './frontier.js';

// Tukey's outer fences: a value more than this many interquartile ranges beyond the
// quartiles is "far out". The inner fences (1.5) also cut the CBR charts' ordinary
// spread (openh264, x264 ultrafast at +30 to +60 %); 3 leaves those in place and
// pins HM and x264 ultrafast on the Fixed QP charts.
const FENCE_IQRS = 3;
// Below this many points the quartiles say little, so every point sets the range.
const MIN_POINTS_FOR_FENCES = 5;
// An outlier this close to the bulk (as a fraction of the bulk's span) is plotted in
// place: widening the axes a little reads better than pinning a point just past them.
const NEAR_SLACK = 0.25;

function quantile(sorted, q) {
  const at = (sorted.length - 1) * q;
  const lo = Math.floor(at);
  return sorted[lo] + (sorted[Math.min(lo + 1, sorted.length - 1)] - sorted[lo]) * (at - lo);
}

/** The [lo, hi] fences of `values`, or an unbounded range when there are too few. */
function fences(values) {
  if (values.length < MIN_POINTS_FOR_FENCES) return [-Infinity, Infinity];
  const sorted = [...values].sort((a, b) => a - b);
  const q1 = quantile(sorted, 0.25);
  const q3 = quantile(sorted, 0.75);
  return [q1 - FENCE_IQRS * (q3 - q1), q3 + FENCE_IQRS * (q3 - q1)];
}

/** The references' frontier points inside our speed range: the ones our frontier is
 * judged against, which must stay on the chart whatever the fences say. */
function boundingRefs(points) {
  const ours = points.filter((p) => p.kind === 'ours');
  if (!ours.length) return [];
  const lo = Math.min(...ours.map((p) => p.x));
  const hi = Math.max(...ours.map((p) => p.x));
  return paretoFront(points.filter((p) => p.kind !== 'ours')).filter((p) => p.x >= lo && p.x <= hi);
}

/** [lo, hi] of the core values, widened to take in the values within the slack of it. */
function span(core, all) {
  const lo = Math.min(...core);
  const hi = Math.max(...core);
  const slack = (hi - lo) * NEAR_SLACK;
  const near = all.filter((v) => v >= lo - slack && v <= hi + slack);
  return [Math.min(lo, ...near), Math.max(hi, ...near)];
}

/**
 * The range of `points` ({x > 0, y, kind, iqr?: [lo, hi]}) to plot: {x: [lo, hi], y: [lo, hi]}, x in
 * fps (the chart's axis is log, so the fences are on log10 x). The core is every point
 * inside the fences on both axes, every one of ours, and the references' frontier
 * points inside our speed range; the range is the core's, widened to take in points
 * just past it. Points outside are the chart's to pin.
 */
export function plotDomain(points) {
  const logX = (p) => Math.log10(p.x);
  const [xLo, xHi] = fences(points.map(logX));
  const [yLo, yHi] = fences(points.map((p) => p.y));
  const keep = new Set(boundingRefs(points));
  const core = points.filter((p) => p.kind === 'ours' || keep.has(p)
    || (logX(p) >= xLo && logX(p) <= xHi && p.y >= yLo && p.y <= yHi));
  const [lx0, lx1] = span(core.map(logX), points.map(logX));
  // a per-clip spread bar widens the range like a point near the bulk; a longer one is clipped
  const ys = points.flatMap((p) => [p.y, ...(p.iqr || [])]);
  // the range's own fps values, not 10 ** log10(fps), so a point at the end is inside exactly
  const xIn = points.filter((p) => logX(p) >= lx0 && logX(p) <= lx1).map((p) => p.x);
  return { x: [Math.min(...xIn), Math.max(...xIn)], y: span(core.map((p) => p.y), ys) };
}
