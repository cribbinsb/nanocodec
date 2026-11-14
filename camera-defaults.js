// Encoder settings the page adds to a live encode of the webcam: a still
// camera's sensor noise is not coded and the target becomes a ceiling
// (docs/analysis/static-noise-2026-11-14.md). Without them a webcam on an
// empty room spends the whole target re-coding noise; with them, at the
// page's default 300 kbps, AV1 codes it at about 200 kbps, HEVC 190 and
// H.264 160, within 0.3 dB of the full-rate picture against the scene.
// Settings the viewer typed in the overrides box win.

/** Engine settings for every codec: the noise skip and the skip-aware budget. */
export const CAMERA_SETTINGS = ['decision.noise_skip=4', 'decision.noise_skip_chroma_ratio=2', 'rate.skip_budget=1'];
/**
 * Codecs whose partition planner gives still content large leaves also split
 * them where the noise skip passes in part (AV1 at 1 Mbps: 476 -> 285 kbps at
 * -0.1 dB; HEVC lost 0.4 dB there, so it is left out).
 */
export const CAMERA_CODEC_SETTINGS = { av1: ['planner.noise_split=1'] };

const settingName = line => line.split('#')[0].split('=')[0].trim();

/** `config` (a view's live settings) with the webcam settings it does not already override. */
export function withCameraSettings(config) {
  if (config.entry) return config;
  const own = new Set((config.overrides || '').split('\n').map(settingName).filter(Boolean));
  const added = [...CAMERA_SETTINGS, ...(CAMERA_CODEC_SETTINGS[config.codec] || [])]
    .filter(line => !own.has(settingName(line)));
  if (!added.length) return config;
  return { ...config, overrides: [config.overrides, ...added].filter(Boolean).join('\n') };
}
