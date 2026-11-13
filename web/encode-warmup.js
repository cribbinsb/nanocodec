// Start-up warm-up for the encode worker, the encoder's counterpart of the
// decode worker's warmup.js. V8 compiles each WASM function on its first call
// and optimises the hot ones later, so a fresh module codes its first frames
// much slower than a warm one (node, 640x360: the first 10 frames of H.264
// Default took 340-940 ms cold against 160-330 ms after a warm-up; AV1
// 560-1110 against 280-620 ms; docs/handoff/wasm-perf.md "Encoder warm-up").
//
// After 'ready', while the encoder still has no frame to code (the capture of
// the first frames, and frame 0 held for the start-up lookahead), a throwaway
// encoder of the same codec, preset and rate mode codes a few tiny synthetic
// frames, one key and the rest inter with motion, then is discarded: the
// functions compiled on the way are the ones the real frames need. It yields
// to the event loop before every frame and stops as soon as `stop()` reports
// real work, so a real frame waits for at most one tiny frame.

/** Size of the warm-up frames: small, but above every codec's minimum and with several superblocks. */
const WARM_UP_WIDTH = 256, WARM_UP_HEIGHT = 144;
/** One key frame and inter frames: enough to compile the inter and rate-control paths. */
const WARM_UP_FRAMES = 8;
/** Pixels the synthetic texture moves per frame, so motion search finds real motion. */
const MOTION_PER_FRAME = 2;

/** Let queued messages (a real frame) run first. */
const yieldToEvents = () => new Promise(resolve => setTimeout(resolve, 0));

/** Synthetic RGBA frame `index`: a textured pattern moving diagonally, with smooth colour ramps. */
function syntheticFrame(rgba, index) {
  const shift = index * MOTION_PER_FRAME;
  for (let y = 0; y < WARM_UP_HEIGHT; y++) {
    for (let x = 0; x < WARM_UP_WIDTH; x++) {
      const o = 4 * (y * WARM_UP_WIDTH + x);
      rgba[o] = (((x + shift) * 7) ^ ((y + shift) * 13)) & 255;
      rgba[o + 1] = (x * 3 + y + 2 * shift) & 255;
      rgba[o + 2] = (y * 5 - x) & 255;
      rgba[o + 3] = 255;
    }
  }
}

/**
 * Code WARM_UP_FRAMES synthetic frames with a throwaway encoder of `module`
 * configured like `config` (the worker's init config) unless `stop()` turns
 * true. Resolves to `{ ms, frames, longestFrameMs, stopped }`, as warmup.js.
 */
export async function warmUpEncoder(module, config, stop) {
  const start = performance.now();
  const result = (frames, longestFrameMs, stopped) =>
    ({ ms: performance.now() - start, frames, longestFrameMs, stopped });
  await yieldToEvents();
  if (stop()) return result(0, 0, true);
  const created = performance.now();
  const c = config;
  const encoder = new module.BrowserEncoder(c.codec, WARM_UP_WIDTH, WARM_UP_HEIGHT, c.qp, c.fps,
    c.keyInterval, c.preset, c.overrides, c.filmGrain);
  if (c.rateMode === 'cbr') encoder.set_rate_control(c.targetKbps, c.peakKbps, c.bufferMs);
  let frames = 0, longestFrameMs = performance.now() - created;
  try {
    const rgba = new Uint8Array(WARM_UP_WIDTH * WARM_UP_HEIGHT * 4);
    for (let index = 0; index < WARM_UP_FRAMES; index++) {
      await yieldToEvents();
      if (stop()) return result(frames, longestFrameMs, true);
      syntheticFrame(rgba, index);
      const frameStart = performance.now();
      encoder.encode(rgba);
      longestFrameMs = Math.max(longestFrameMs, performance.now() - frameStart);
      frames++;
    }
  } finally {
    encoder.free();
  }
  return result(frames, longestFrameMs, false);
}
