// A short record of the page's comparison lifecycle (settings changes, restarts,
// starts, failures, stream choices), for diagnosing rare races: each event goes to
// the console at debug level with a millisecond timestamp, where the product smoke
// test collects it and prints it when it fails. Nothing else reads it.

/** Note one lifecycle event. */
export function trace(...parts) {
  console.debug(`page: trace ${Math.round(performance.now())} ${parts.join(' ')}`);
}
