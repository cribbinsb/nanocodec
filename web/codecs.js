// What the page needs to know about each codec, in picker order: its label,
// its native quantizer scale (the encoder's fixed-quantizer argument), whether
// the UI offers its Fast (and Faster) preset, spatial SVC (two layers, op 0 / op 1) and three-layer SVC
// (`svc3`: op 0 / 1 / 2), and how a recording is stored (IVF with a
// FourCC, or a raw Annex-B elementary stream with a file extension). What each codec
// supports beyond that (film grain, block QP, ...) comes from the codec module (options.js).
export const CODECS = {
  h264: { label: 'H.264', quantizer: { name: 'QP', min: 0, max: 51, initial: 32 }, fast: true, faster: true, fastest: true, annexB: '264' },
  hevc: { label: 'HEVC', quantizer: { name: 'QP', min: 0, max: 51, initial: 32 }, fast: true, faster: true, annexB: 'hevc' },
  av1: { label: 'AV1', quantizer: { name: 'qindex', min: 1, max: 255, initial: 180 }, fast: true, faster: true, svc: true, svc3: true, ivf: 'AV01' },
  vvc: { label: 'VVC', quantizer: { name: 'QP', min: 0, max: 63, initial: 32 }, fast: true, annexB: 'vvc' },
  vp9: { label: 'VP9', quantizer: { name: 'qindex', min: 0, max: 255, initial: 180 }, fast: true, faster: true, fastest: true, svc: true, ivf: 'VP90' },
  av2: { label: 'AV2', quantizer: { name: 'qindex', min: 0, max: 255, initial: 180 }, fast: false, ivf: 'AV02' },
};

/** The download's file extension for `codec`. */
export function streamExtension(codec) {
  return CODECS[codec].ivf ? 'ivf' : CODECS[codec].annexB;
}
