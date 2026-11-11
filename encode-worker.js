// Encoder half of one comparison view; decode-worker.js is the other half.
// The page sends each RGBA source frame here. The packet goes straight to the
// view's decode worker over a MessageChannel port, never through the page, so
// this worker encodes frame N+1 while the decode worker still decodes frame N.
// The source buffer travels on with the packet: the decode worker measures
// PSNR against it and hands it back to the page with the decoded frame.
//
// The module build (relaxed SIMD or plain SIMD128) is chosen in wasm.js.
import { MODULE_URL } from './wasm.js';
// Started at once, awaited by 'init' (the message handler must exist synchronously).
const codecModule = import(MODULE_URL);
let encoder, decoderPort;
// Frame 0, held back one frame while the encoder looks ahead to frame 1
// (encoder.wants_look_ahead: rate control sizes the first key from start-up
// camera motion).
let held = null;
/**
 * Encode one source frame; the fields the decode worker passes on to the page.
 * `repeat`: the input repeated the picture (frame hold) and nothing was coded:
 * the packet is empty and the decoder shows its last frame again.
 */
function encodeFrame({ rgba, timestamp }) {
  const start = performance.now();
  const packet = encoder.encode(new Uint8Array(rgba));
  return { timestamp, rgba, packet, repeat: encoder.repeat, key: encoder.key, quantizer: encoder.quantizer,
    encodeMs: performance.now() - start };
}
function configureRate(c) {
  if (c.rateMode === 'cbr') encoder.set_rate_control(c.targetKbps, c.peakKbps, c.bufferMs);
  else encoder.disable_rate_control();
}
self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      const module = await codecModule;
      await module.default();
      const c = message.config;
      encoder = new module.BrowserEncoder(c.codec, c.width, c.height, c.qp, c.fps, c.keyInterval, c.preset, c.overrides, c.filmGrain);
      decoderPort = message.port;
      configureRate(c);
      self.postMessage({ type: 'ready' });
    } else if (message.type === 'keyframe') {
      encoder.request_keyframe();
    } else if (message.type === 'rate') {
      configureRate(message.config);
    } else if (message.type === 'frame') {
      if (!held && encoder.wants_look_ahead) {
        // Nothing to show yet: the page still hears that frame 0 is done, in order.
        held = message;
        decoderPort.postMessage({ id: message.id, held: true });
        return;
      }
      const coded = [];
      if (held) {
        encoder.look_ahead(new Uint8Array(message.rgba));
        coded.push(encodeFrame(held));
        held = null;
      }
      coded.push(encodeFrame(message));
      decoderPort.postMessage({ id: message.id, display: message.display, coded, bufferFullness: encoder.buffer_fullness,
        bufferOverflows: encoder.buffer_overflows }, coded.flatMap(c => [c.packet.buffer, c.rgba]));
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
