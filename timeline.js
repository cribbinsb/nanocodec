// The source timeline both views share. Every source is sampled on one grid of
// ticks at SOURCE_FPS: tick c is source time c / SOURCE_FPS. A view at a lower
// frame rate codes every Nth tick (N = SOURCE_FPS / fps, an integer) and holds
// its last picture in between, so two views always have ticks in common: the
// multiples of the least common multiple of their N. Pause and frame stepping
// land on those common instants (app.js), so both views show the same moment.
//
// A video's own frames map onto the grid as a 30 Hz sampler would see them:
// tick c shows source frame floor(c * F / SOURCE_FPS) of an F fps video, so a
// 25 fps clip repeats a frame every sixth tick. tools/web/make-reference-streams.py
// samples the clips the same way for the pre-encoded streams.

/** Ticks per second of the shared source timeline. */
export const SOURCE_FPS = 30;
/**
 * The frame rates a view can run at: SOURCE_FPS divided by an integer, and whole
 * numbers, as the encoder takes an integer rate (nc-web BrowserEncoder), so 6 fps
 * rather than 7.5.
 */
export const FRAME_RATES = [1, 2, 3, 5, 6].map(divisor => SOURCE_FPS / divisor);
// Rounding guard for frame times computed in floating point.
const EPSILON = 1e-6;

/** The tick step of a view at `fps` (1 at SOURCE_FPS); throws for a rate off the grid. */
export function tickStep(fps) {
  const step = SOURCE_FPS / fps;
  if (!Number.isInteger(step) || step < 1) throw new Error(`${fps} fps does not divide ${SOURCE_FPS} fps`);
  return step;
}

/** Whether a view with tick step `step` codes tick `tick`. */
export function codesTick(step, tick) { return tick % step === 0; }

const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/** The step of the ticks every view codes: the least common multiple of their steps. */
export function commonStep(steps) { return steps.reduce((a, b) => a * b / gcd(a, b), 1); }

/** The first tick after `tick` that every view (tick steps `steps`) codes. */
export function nextCommonTick(tick, steps) {
  const step = commonStep(steps);
  return (Math.floor(tick / step) + 1) * step;
}

/** Source time of a tick, in milliseconds. */
export function tickMs(tick) { return tick * 1000 / SOURCE_FPS; }

/** The tick nearest to a source time in milliseconds. */
export function tickAt(ms) { return Math.round(ms * SOURCE_FPS / 1000); }

/** The frame of an `fps` video shown at `tick`. */
export function frameAtTick(tick, fps) { return Math.floor(tick * fps / SOURCE_FPS + EPSILON); }

/** The ticks on which frame `frame` of an `fps` video is shown: [first, last], or null for none. */
export function ticksOfFrame(frame, fps) {
  const first = Math.max(0, Math.ceil(frame * SOURCE_FPS / fps - EPSILON));
  const last = Math.ceil((frame + 1) * SOURCE_FPS / fps - EPSILON) - 1;
  return first <= last ? [first, last] : null;
}

/** Ticks in one loop of a video of `frames` frames at `fps`. */
export function loopTicks(frames, fps) { return Math.round(frames * SOURCE_FPS / fps); }
