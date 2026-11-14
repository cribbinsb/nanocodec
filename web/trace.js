// A short record of the page's comparison lifecycle (settings changes, restarts,
// starts, pauses, decoder switches, failures, stream choices, page errors), for
// diagnosing rare races: each event goes to the console at debug level with a
// millisecond timestamp, where the product smoke test collects it and prints it
// when it fails, and the last TRACE_KEPT events are kept for the debug report
// (debug-report.js `traceTail`).

/** Events kept for the debug report. */
const TRACE_KEPT = 300;
const kept = [];

/** Note one lifecycle event. */
export function trace(...parts) {
  const line = `${Math.round(performance.now())} ${parts.join(' ')}`;
  console.debug(`page: trace ${line}`);
  kept.push(line);
  if (kept.length > TRACE_KEPT) kept.shift();
}

/** The kept events, oldest first, as "ms text" lines. */
export function traceTail() { return [...kept]; }
