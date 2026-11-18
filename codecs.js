// What the page needs to know about each codec, in picker order: its label,
// its native quantizer scale (the encoder's fixed-quantizer argument), the
// speed presets the UI offers (`presets`, names of the ladder PRESETS, Medium
// first), the spatial layer counts above one it offers (`spatialLayers`: two
// layers are operating points 0 / 1, three are 0 / 1 / 2; layered only at
// LAYERED_PRESETS), and how a recording is stored (IVF with a FourCC, or a raw
// Annex-B elementary stream with a file extension). What each codec
// supports beyond that (film grain, block QP, ...) comes from the codec module (options.js).
/**
 * The codec-independent speed ladder, as [name, label], Medium (each codec's
 * Default, what nc-web codes without a preset) first, then Slow (the quality-first
 * rung, about 1.5-2.5x Medium's CPU) and the faster rungs.
 */
export const PRESETS = [['medium', 'Medium (default)'], ['slow', 'Slow'], ['fast', 'Fast'], ['faster', 'Faster'], ['veryfast', 'Veryfast']];
export const DEFAULT_PRESET = 'medium';
/** The presets a spatially layered (SVC) stream can be coded at. */
export const LAYERED_PRESETS = ['medium', 'fast'];

export const CODECS = {
  h264: { label: 'H.264', quantizer: { name: 'QP', min: 0, max: 51, initial: 32 }, presets: ['medium', 'slow', 'fast', 'faster', 'veryfast'], spatialLayers: [], annexB: '264' },
  hevc: { label: 'HEVC', quantizer: { name: 'QP', min: 0, max: 51, initial: 32 }, presets: ['medium', 'slow', 'fast', 'faster'], spatialLayers: [], annexB: 'hevc' },
  av1: { label: 'AV1', quantizer: { name: 'qindex', min: 1, max: 255, initial: 180 }, presets: ['medium', 'slow', 'fast', 'faster', 'veryfast'], spatialLayers: [2, 3], ivf: 'AV01' },
  vvc: { label: 'VVC', quantizer: { name: 'QP', min: 0, max: 63, initial: 32 }, presets: ['medium', 'slow', 'fast'], spatialLayers: [], annexB: 'vvc' },
  vp9: { label: 'VP9', quantizer: { name: 'qindex', min: 0, max: 255, initial: 180 }, presets: ['medium', 'slow', 'fast', 'faster', 'veryfast'], spatialLayers: [2, 3], ivf: 'VP90' },
  av2: { label: 'AV2', quantizer: { name: 'qindex', min: 0, max: 255, initial: 180 }, presets: ['medium', 'slow', 'fast', 'faster'], spatialLayers: [], ivf: 'AV02' },
};

/** The download's file extension for `codec`. */
export function streamExtension(codec) {
  return CODECS[codec].ivf ? 'ivf' : CODECS[codec].annexB;
}
