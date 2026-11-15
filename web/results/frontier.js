// The Pareto frontier of a chart's points (overview, explorer.js): the points no other
// point of the same group beats on both axes, i.e. faster (higher x) and better
// (lower y) at once. Pure, so web/tests/frontier.test.js checks it under node.

/** Whether `a` dominates `b`: at least as fast and as good, strictly better in one. */
export function dominates(a, b) {
  return a.x >= b.x && a.y <= b.y && (a.x > b.x || a.y < b.y);
}

/** The points of `points` ({x, y}) no other point dominates, slowest first. Ties
 * (identical points) are both kept: neither is better. */
export function paretoFront(points) {
  return points.filter((p) => !points.some((q) => q !== p && dominates(q, p)))
    .sort((a, b) => a.x - b.x || a.y - b.y);
}

/** The frontier as a line, [[x, y], ...] slowest first, straight from one frontier
 * point to the next (a reading aid: presets are discrete, so the points between
 * two frontier points are not themselves reachable). */
export function frontierLine(front) {
  return front.map((p) => [p.x, p.y]);
}
