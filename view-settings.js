// The per-view picture and rate choices the page offers (panel.js), and how a
// pre-encoded reference stream (reference.js) maps onto them: the settings that
// make the other view a live nanocodec encode of the same clip, frames, size,
// frame rate and bitrate ("Match in the other view"), and the check that every
// stream in the manifest can be matched (tests/reference.test.js).
import { CODECS } from './codecs.js';
import { FRAME_RATES } from './timeline.js';

/** Picture sizes a view can run at, smallest first. */
export const RESOLUTIONS = [[384, 216], [640, 360], [854, 480], [1280, 720]];
export const DEFAULT_RESOLUTION = [854, 480];
export const DEFAULT_FPS = 15;
/** The target and peak bitrate inputs' range, in kbps. */
export const RATE_LIMITS_KBPS = { min: 10, max: 100000, step: 10 };
/** A new view's peak rate is this multiple of its target (300 / 600 kbps by default). */
export const PEAK_TO_TARGET = 2;
/** Keyframe interval choices, in seconds. */
export const GOP_SECONDS = [1, 2, 5, 10, 30];
/**
 * The key-interval choice "On clip start": a key frame where a sample clip loops, as a
 * pre-encoded stream (one loop of its clip, replayed from its key frame) keys. A live
 * view's default on a clip (owner, 2026-11-22: the views keyed at different times,
 * unseen); a source that does not loop (webcam, scene, file) is not offered it.
 */
export const KEY_AT_CLIP_START = 'clip-start';
/** Rate-control buffer choices, in milliseconds. */
export const BUFFER_CHOICES_MS = [250, 500, 1000, 2000];
/** The page's own colour handling: 8-bit 4:2:0, which every stream must also be. */
const PIXEL_FORMAT = { bitDepth: 8, chroma: '4:2:0' };

/**
 * The live settings that reproduce a reference stream's conditions in another view:
 * its size, frame rate, target bitrate, buffer and key interval, and its codec as
 * the starting codec (the visitor may pick another; the preset is left alone).
 */
export function matchingSettings(entry) {
  return {
    width: entry.width, height: entry.height, fps: entry.fps, codec: entry.codec, rateMode: 'cbr',
    targetKbps: entry.targetKbps, peakKbps: entry.targetKbps * PEAK_TO_TARGET, bufferMs: entry.bufferMs,
    gop: entry.keyInterval >= entry.frames ? KEY_AT_CLIP_START : entry.keyInterval / entry.fps,
  };
}

/** Why a reference stream cannot be matched by a live view (empty when it can). */
export function matchProblems(entry, clipIds) {
  const settings = matchingSettings(entry), problems = [];
  if (!clipIds.includes(entry.clip)) problems.push(`clip ${entry.clip} is not a hosted sample clip`);
  if (!RESOLUTIONS.some(([w, h]) => w === entry.width && h === entry.height)) problems.push(`${entry.width}x${entry.height} is not offered`);
  if (!FRAME_RATES.includes(entry.fps)) problems.push(`${entry.fps} fps is not offered`);
  if (!(entry.codec in CODECS)) problems.push(`codec ${entry.codec} is not offered`);
  for (const rate of [settings.targetKbps, settings.peakKbps]) {
    if (rate < RATE_LIMITS_KBPS.min || rate > RATE_LIMITS_KBPS.max || rate % RATE_LIMITS_KBPS.step) problems.push(`${rate} kbps is outside the rate inputs`);
  }
  if (!BUFFER_CHOICES_MS.includes(settings.bufferMs)) problems.push(`a ${settings.bufferMs} ms buffer is not offered`);
  if (settings.gop !== KEY_AT_CLIP_START && !GOP_SECONDS.includes(settings.gop)) problems.push(`a ${settings.gop} s key interval is not offered`);
  if (entry.bitDepth !== PIXEL_FORMAT.bitDepth || entry.chroma !== PIXEL_FORMAT.chroma) problems.push('not 8-bit 4:2:0');
  return problems;
}

/**
 * Whether a view codes a clip at more than half its source file's bitrate: the source's
 * own coding then shows in the comparison (the project codes a clip at most at half its
 * source bitrate, CLAUDE.md "Testing"). Unknown (null) source rates are not flagged.
 */
export const SOURCE_RATE_SHARE = 0.5;
export function overSourceShare(targetKbps, sourceKbps) {
  return Number.isFinite(sourceKbps) && targetKbps > SOURCE_RATE_SHARE * sourceKbps;
}

/** The longest periodic key interval a live encoder takes (nc-web BrowserEncoder), in seconds. */
const MAX_KEY_SECONDS = 30;

/**
 * Keys "On clip start" for a view coding every `step`th tick of a `loopTicks`-tick clip
 * loop: the encode worker requests a key at the first frame it codes in each new loop
 * (encode-worker.js), and the periodic interval is one loop of frames when the loop is a
 * whole number of them (its keys fall on those same frames, so `--intra-period` alone
 * reproduces the stream), else the longest the encoder takes, which then adds a key every
 * MAX_KEY_SECONDS (`extraSeconds`; shown in the view). `requested`: the frames the
 * worker requests keys at, for the first `frames` frames.
 */
export function clipStartKeys(loopTicks, step, fps, frames = 0) {
  const whole = loopTicks % step === 0;
  const requested = [];
  for (let loop = 1; Math.ceil(loop * loopTicks / step) < frames; loop++) requested.push(Math.ceil(loop * loopTicks / step));
  return { keyInterval: whole ? loopTicks / step : MAX_KEY_SECONDS * fps, extraSeconds: whole ? null : MAX_KEY_SECONDS, requested: whole ? [] : requested };
}
