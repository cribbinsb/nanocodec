// Encoder half of one comparison view; decode-worker.js is the other half.
// The capture worker (capture-worker.js) sends each captured frame here (YUV
// planes, or RGBA) over a MessageChannel port; the page sends only control
// messages. This worker converts it to the view's size (nc-web `SourceFrame`)
// and encodes it. The packet goes straight to the view's decode worker over a
// MessageChannel port, never through the page, so this worker encodes frame
// N+1 while the decode worker still decodes frame N. The converted source
// (packed I420) travels on with the packet: the decode worker measures PSNR
// against it and hands it back to the page as the original frame.
//
// The stream signals the colour of the samples it codes (colour.js). The
// encoder is made for the configuration's default colour and made again,
// before the first frame is coded, if the source's colour differs; a later
// change of colour (the capture path changed) restarts the stream the same
// way, with a key frame.
//
// The module build (relaxed SIMD or plain SIMD128) is chosen in wasm.js.
import { MODULE_URL } from './wasm.js';
// Frame timestamps from the page are in milliseconds; the encoder takes microseconds.
const MICROSECONDS_PER_MS = 1000;
import { warmUpEncoder } from './encode-warmup.js';
// The colour of the sample clips (limited-range BT.709): the usual first frame's.
const DEFAULT_COLOUR = 'bt709';
// Started at once, awaited by 'init' (the message handler must exist synchronously).
const codecModule = import(MODULE_URL);
let module, wasm, encoder, decoderPort, config;
// The view's converted source frame, reused.
let source;
// Frame 0, held back one frame while the encoder looks ahead to frame 1
// (encoder.wants_look_ahead: rate control sizes the first key from start-up
// camera motion).
let held = null;
// Set when the first real frame is coded: start-up warm-up (encode-warmup.js) stops then.
let coding = false;
/** Convert a captured frame (capture-worker.js) into `source`. */
function convert(frame) {
  if (frame.kind === 'planes') {
    source.from_planes(new Uint8Array(frame.data), frame.format, frame.colour, frame.width, frame.height, Uint32Array.from(frame.layout));
  } else {
    source.from_rgba(new Uint8Array(frame.data));
  }
  if (encoder.colour !== source.colour) createEncoder(source.colour);
}
/** `source` packed as I420, copied out of WASM memory. */
function packedSource() {
  const offset = source.packed();
  return new Uint8Array(wasm.memory.buffer, offset, source.packed_length).slice();
}
/**
 * Encode one captured frame; the fields the decode worker passes on to the page.
 * `repeat`: the input repeated the picture (frame hold) and nothing was coded:
 * the packet is empty and the decoder shows its last frame again.
 */
function encodeFrame({ frame, timestamp }) {
  const start = performance.now();
  convert(frame);
  // A live source's frames carry their capture time (some may have been skipped):
  // rate control budgets each for the real time since the previous one.
  const packet = config.timed ? encoder.encode_at(source, timestamp * MICROSECONDS_PER_MS) : encoder.encode(source);
  const packed = packedSource();
  return { timestamp, source: packed.buffer, colour: source.colour, packet, repeat: encoder.repeat, key: encoder.key,
    quantizer: encoder.quantizer, encodeMs: performance.now() - start };
}
function configureRate(c) {
  if (c.rateMode === 'cbr') encoder.set_rate_control(c.targetKbps, c.peakKbps, c.bufferMs);
  else encoder.disable_rate_control();
}
/** (Re)create the encoder for input of `colour`. */
function createEncoder(colour) {
  const c = config;
  encoder?.free();
  encoder = new module.BrowserEncoder(c.codec, c.width, c.height, c.qp, c.fps, c.keyInterval, c.preset, c.overrides, c.filmGrain, colour);
  configureRate(c);
}
/** Handle a page or capture message; any failure is reported to the page. */
async function receive({ data: message }) {
  try {
    if (message.type === 'init') {
      module = await codecModule;
      // The page posts the module it compiled once (wasm.js compiledModule).
      wasm = await module.default({ module_or_path: message.wasmModule });
      config = message.config;
      createEncoder(DEFAULT_COLOUR);
      source = new module.SourceFrame(config.width, config.height);
      decoderPort = message.port;
      message.capturePort.onmessage = receive;
      self.postMessage({ type: 'ready' });
      // Not awaited: real frames are handled as they come and stop it.
      if (message.warmUp !== false) {
        warmUpEncoder(module, config, () => coding)
          .then(report => self.postMessage({ type: 'warmed', ...report }))
          .catch(error => console.warn('encoder warm-up failed:', error));
      }
    } else if (message.type === 'decoder') {
      // The view's decoder was replaced (analysis switched, pipeline.js
      // `switchDecoders`): packets go to the new one from the next frame on.
      decoderPort = message.port;
      self.postMessage({ type: 'decoder-connected' });
    } else if (message.type === 'keyframe') {
      encoder.request_keyframe();
    } else if (message.type === 'rate') {
      config = message.config;
      configureRate(config);
    } else if (message.type === 'frame') {
      if (!held && encoder.wants_look_ahead) {
        // Nothing to show yet: the page still hears that frame 0 is done, in order.
        held = message;
        decoderPort.postMessage({ id: message.id, held: true });
        return;
      }
      coding = true;
      const coded = [];
      if (held) {
        convert(message.frame);
        encoder.look_ahead(source);
        coded.push(encodeFrame(held));
        held = null;
      }
      coded.push(encodeFrame(message));
      decoderPort.postMessage({ id: message.id, display: message.display, coded, bufferFullness: encoder.buffer_fullness,
        bufferOverflows: encoder.buffer_overflows }, coded.flatMap(c => [c.packet.buffer, c.source]));
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}
self.onmessage = receive;
