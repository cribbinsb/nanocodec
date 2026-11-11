// Decoder half of one comparison view; encode-worker.js is the other half.
// Packets arrive from the view's encode worker over a MessageChannel port
// with the RGBA source frame each was coded from. This worker decodes them,
// measures PSNR against that source (in WASM, off the encoder's path) and
// posts the frame to show, with every timing, to the page.
//
// Buffers circulate instead of being allocated per frame: the source buffer
// goes back to the page with the result, and the page gives back each
// displayed pixel buffer once the view no longer needs it.
//
// Decoded frames go out in the format the page's renderer asks for with each
// frame (renderer.js): packed I420 for a WebGL view, which converts on the GPU,
// or RGBA converted here in WASM for the CPU path.
import { MODULE_URL } from './wasm.js';
// Started at once, awaited by 'init' (the message handler must exist synchronously).
const codecModule = import(MODULE_URL);
let decoder, wasm;
// Returned display buffers kept for reuse: one per frame the page can hold in
// flight or on screen (pipeline.js), so none is allocated in steady state.
const MAX_SPARE_PIXELS = 4;
const sparePixels = [];
/** Copy the decoder's latest frame, in `format`, out of WASM memory into a reused buffer. */
function decodedPixels(format) {
  const { width, height } = decoder;
  const length = format === 'i420' ? width * height + 2 * Math.ceil(width / 2) * Math.ceil(height / 2) : width * height * 4;
  // Convert first: it may grow WASM memory, which detaches older buffers.
  const offset = format === 'i420' ? decoder.i420_frame() : decoder.rgba_frame();
  const view = new Uint8Array(wasm.memory.buffer, offset, length);
  let pixels = sparePixels.pop();
  if (!pixels || pixels.length !== length) pixels = new Uint8Array(length);
  pixels.set(view);
  return pixels;
}
/**
 * Decode one coded frame and score it against its source. `decodeMs` is the
 * decoder alone; `decodeStageMs` is all the time the frame held this worker.
 * A repeat (frame hold: nothing coded) shows the last decoded frame again.
 */
function decodeFrame({ rgba, packet, ...coded }) {
  const start = performance.now();
  if (!coded.repeat) decoder.decode(packet);
  const decoded = performance.now();
  const psnr = decoder.psnr(new Uint8Array(rgba));
  return { ...coded, packet, source: rgba, psnr, decodeMs: decoded - start, decodeStageMs: performance.now() - start };
}
/** One frame from the encode worker: decode it and post it to the page. */
function receivePacket({ data: message }) {
  try {
    if (message.held) {
      self.postMessage({ type: 'frame', id: message.id, held: true });
      return;
    }
    // Two coded frames after the start-up lookahead: frame 0 is `previous`.
    const decoded = message.coded.map(decodeFrame);
    const result = decoded.at(-1), previous = decoded.length > 1 ? decoded[0] : null;
    const displayStart = performance.now();
    const format = message.display === 'i420' ? 'i420' : 'rgba';
    const pixels = decodedPixels(format);
    result.decodeStageMs += performance.now() - displayStart;
    const transfer = [pixels.buffer, ...decoded.flatMap(d => [d.packet.buffer, d.source])];
    self.postMessage({ type: 'frame', id: message.id, ...result, previous, format, pixels, width: decoder.width,
      height: decoder.height, bufferFullness: message.bufferFullness, bufferOverflows: message.bufferOverflows }, transfer);
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}
self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      const module = await codecModule;
      wasm = await module.default();
      decoder = new module.BrowserDecoder(message.codec);
      message.port.onmessage = receivePacket;
      self.postMessage({ type: 'ready' });
    } else if (message.type === 'recycle') {
      if (sparePixels.length < MAX_SPARE_PIXELS) sparePixels.push(new Uint8Array(message.buffer));
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
