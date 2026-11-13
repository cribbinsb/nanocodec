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
    gopSeconds: entry.keyInterval / entry.fps,
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
  if (!GOP_SECONDS.includes(settings.gopSeconds)) problems.push(`a ${settings.gopSeconds} s key interval is not offered`);
  if (entry.bitDepth !== PIXEL_FORMAT.bitDepth || entry.chroma !== PIXEL_FORMAT.chroma) problems.push('not 8-bit 4:2:0');
  return problems;
}
