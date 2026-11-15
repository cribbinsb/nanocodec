// The worker of an "Original" view (pipeline.js OriginalView): no encoder and
// no decoder. It takes each captured frame from the capture worker, places it
// at the view's size exactly as a live view's encoder is given it
// (view-source-frame.js), and posts it to the page to show, in the format the
// view's renderer asks for: the packed I420 itself for WebGL (renderer.js
// converts it as it converts decoded frames) or RGBA converted by colour.js
// `i420ToRgba`, which equals nc-dsp's conversion (web/tests/colour-wasm.mjs).
// The placed frame also goes back as the view's source, as a decode worker
// returns it, for the magnifier and the debug report.
import { MODULE_URL } from './wasm.js';
import { i420ToRgba } from './colour.js';
import { ViewSourceFrame } from './view-source-frame.js';

// Started at once, awaited by 'init' (the message handler must exist synchronously).
const codecModule = import(MODULE_URL);
let viewSource = null, width = 0, height = 0;
// Displayed pixel buffers the page has given back, reused by size (as decode-worker.js).
const MAX_SPARE_PIXELS = 5;
const sparePixels = [];

/** A reused (or new) buffer of `length` bytes. */
function pixelBuffer(length) {
  const match = sparePixels.findIndex(p => p.length === length);
  return match >= 0 ? sparePixels.splice(match, 1)[0] : new Uint8Array(length);
}
/** The placed frame in the renderer's `display` format. */
function displayed(source, colour, display) {
  const packed = new Uint8Array(source);
  if (display === 'i420') {
    const pixels = pixelBuffer(packed.length);
    pixels.set(packed);
    return { format: 'i420', pixels };
  }
  return { format: 'rgba', pixels: new Uint8Array(i420ToRgba(packed, width, height, colour).buffer) };
}
/** One captured frame: place it, and post it to the page as this view's shown frame. */
function receiveCapture({ data: { id, timestamp, frame, display } }) {
  try {
    const start = performance.now();
    const { source, colour } = viewSource.place({ frame, width, height });
    const { format, pixels } = displayed(source, colour, display);
    const ms = performance.now() - start;
    self.postMessage({ type: 'frame', id, timestamp, source, colour, format, pixels, width, height, streamWidth: width, streamHeight: height,
      key: false, repeat: false, psnr: null, decodeMs: null, decodeStageMs: ms }, [source, pixels.buffer]);
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}
self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      const module = await codecModule;
      const wasm = await module.default({ module_or_path: message.wasmModule });
      viewSource = new ViewSourceFrame(module, wasm);
      ({ width, height } = message);
      message.port.onmessage = receiveCapture;
      self.postMessage({ type: 'ready' });
    } else if (message.type === 'recycle') {
      if (sparePixels.length < MAX_SPARE_PIXELS) sparePixels.push(new Uint8Array(message.buffer));
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
