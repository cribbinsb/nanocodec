// One independent codec state per comparison view. No frames queue behind encoding.
// Pixel buffers circulate instead of being allocated per frame: the page sends
// each source frame in a buffer this worker returns with the result, and gives
// back each displayed pixel buffer once the view no longer needs it.
//
// Decoded frames go out in the format the page's renderer asks for with each
// frame (renderer.js): packed I420 for a WebGL view, which converts on the GPU,
// or RGBA converted here in WASM for the CPU path.
//
// The module build (relaxed SIMD or plain SIMD128) is chosen in wasm.js.
import { MODULE_URL, RELAXED_SIMD } from './wasm.js';
// Started at once, awaited by 'init' (the message handler must exist synchronously).
const codecModule = import(MODULE_URL);
let BrowserEncoder, BrowserDecoder, encoder, decoder, wasm;
const sparePixels = [];
// Frame 0, held back one frame while the encoder looks ahead to frame 1
// (encoder.wants_look_ahead: rate control sizes the first key from start-up
// camera motion).
let held = null;
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
/** Encode and decode one RGBA frame; the result fields a panel records. */
function codeFrame(rgba, timestamp) {
  const start = performance.now();
  const packet = encoder.encode(rgba);
  const encoded = performance.now();
  decoder.decode(packet);
  const decoded = performance.now();
  return { timestamp, packet, key: encoder.key, encodeMs: encoded - start, decodeMs: decoded - encoded,
    psnr: encoder.quality(decoder), quantizer: encoder.quantizer };
}
function configureRate(c) {
  if (c.rateMode === 'cbr') encoder.set_rate_control(c.targetKbps, c.peakKbps, c.bufferMs);
  else encoder.disable_rate_control();
}
self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      const module = await codecModule;
      ({ BrowserEncoder, BrowserDecoder } = module);
      wasm = await module.default();
      const c = message.config;
      encoder = new BrowserEncoder(c.codec, c.width, c.height, c.qp, c.fps, c.keyInterval, c.preset, c.overrides, c.filmGrain);
      decoder = new BrowserDecoder(c.codec);
      configureRate(c);
      self.postMessage({ type: 'ready', relaxedSimd: RELAXED_SIMD });
    } else if (message.type === 'keyframe') {
      encoder.request_keyframe();
    } else if (message.type === 'rate') {
      configureRate(message.config);
    } else if (message.type === 'recycle') {
      if (sparePixels.length < 2) sparePixels.push(new Uint8Array(message.buffer));
    } else if (message.type === 'frame') {
      const source = new Uint8Array(message.rgba);
      if (!held && encoder.wants_look_ahead) {
        // Nothing to show yet: hand the buffer back and wait for frame 1.
        held = { rgba: source.slice(), timestamp: message.timestamp };
        self.postMessage({ type: 'frame', id: message.id, held: true, source: message.rgba }, [message.rgba]);
        return;
      }
      let previous = null;
      if (held) {
        encoder.look_ahead(source);
        previous = codeFrame(held.rgba, held.timestamp);
        held = null;
      }
      const result = codeFrame(source, message.timestamp);
      const format = message.display === 'i420' ? 'i420' : 'rgba';
      const pixels = decodedPixels(format);
      const transfer = [result.packet.buffer, pixels.buffer, message.rgba];
      if (previous) transfer.push(previous.packet.buffer);
      self.postMessage({ type: 'frame', id: message.id, ...result, previous, format, pixels, source: message.rgba,
        width: decoder.width, height: decoder.height, bufferFullness: encoder.buffer_fullness,
        bufferOverflows: encoder.buffer_overflows }, transfer);
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
