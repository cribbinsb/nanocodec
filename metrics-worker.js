// MS-SSIM for analysis mode, in its own worker so it never holds up decoding:
// ~100 ms per 720p frame (nc-web `ms_ssim_luma`, the evaluation's definition,
// nc-dsp `ms_ssim`), against a few ms of decode. A view's decode worker starts
// one with analysis on and sends it the newest shown frame's luma and its
// source luma at most once a second (decode-worker.js `queueScore`,
// SCORE_INTERVAL_MS): frames shown meanwhile are skipped, and the last one
// shown (a paused frame) is always scored. Each score goes back with the tick it belongs to.
import { ANALYSIS_MODULE_URL } from './wasm.js';

let module = null;
// Resolves once the analysis module is instantiated ('init' carries the compiled module).
let ready = null;

self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      ready = import(ANALYSIS_MODULE_URL).then(async loaded => {
        await loaded.default({ module_or_path: message.wasmModule });
        module = loaded;
      });
      return;
    }
    await ready;
    const start = performance.now();
    const value = module.ms_ssim_luma(new Uint8Array(message.reference), new Uint8Array(message.distorted), message.width, message.height) ?? null;
    self.postMessage({ type: 'scored', tick: message.tick, value, ms: performance.now() - start });
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
