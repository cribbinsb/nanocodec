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
// Over an RTP link (config.link, rtp-link.js) each coded frame is also
// packetized here (nc-web `RtpSender`) and its packets travel with it; the
// decode worker's loss feedback comes back on the same port, stamped with the
// source time it reaches the encoder, and is applied before the first frame
// captured at or after that time, so the round trip is the emulated one,
// whatever the machine's speed.
//
// The module build (relaxed SIMD or plain SIMD128) is chosen in wasm.js.
import { MODULE_URL } from './wasm.js';
// Frame timestamps from the page are in milliseconds; the encoder takes microseconds.
const MICROSECONDS_PER_MS = 1000;
import { warmUpEncoder } from './encode-warmup.js';
import { tickAt } from './timeline.js';
import { DEPENDENCY_DESCRIPTOR_ID } from './rtp-link.js';
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
// A key frame was requested, or the rate settings changed, and the next coded
// frame is the first to have it: that frame says so (`keyRequested`,
// `rateApplied`), so the page's debug report can replay it at exactly that
// frame (these messages and the captures come on different ports).
let keyRequested = false, rateApplied = null;
// The clip loop the previous coded frame was in, for keys "On clip start"
// (config.loopTicks, view-settings.js `clipStartKeys`).
let lastLoop = null;
// The RTP sender over an RTP link (null for the bitstream link), and loss
// feedback waiting for its arrival time ({ kind, frame, at }, in arrival order).
let sender = null, feedback = [];
/**
 * Request a key for the first frame coded in a new loop of the clip (timestamps are
 * source times). A loop of whole frames needs none: its periodic interval is the loop.
 */
function keyAtClipStart(timestamp) {
  if (!config.loopTicks || config.loopTicks % config.step === 0) return;
  const loop = Math.floor(tickAt(timestamp) / config.loopTicks);
  if (lastLoop !== null && loop > lastLoop) { encoder.request_keyframe(); keyRequested = true; }
  lastLoop = loop;
}
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
  keyAtClipStart(timestamp);
  const repairing = applyFeedback(timestamp);
  // A live source's frames carry their capture time (some may have been skipped):
  // rate control budgets each for the real time since the previous one.
  const packet = config.timed ? encoder.encode_at(source, timestamp * MICROSECONDS_PER_MS) : encoder.encode(source);
  const packed = packedSource();
  const requested = keyRequested, rate = rateApplied;
  keyRequested = false; rateApplied = null;
  return { timestamp, source: packed.buffer, colour: source.colour, packet, repeat: encoder.repeat, key: encoder.key, keyRequested: requested, rateApplied: rate,
    quantizer: encoder.quantizer, encodeMs: performance.now() - start, rtp: packetize(packet, timestamp), repairing };
}
/**
 * Apply the loss feedback that has reached the encoder by source time `timestamp`
 * (nc-web `report_lost_frame`, `acknowledge_frame`, `request_keyframe`). Returns
 * whether any of it asks for a repair, which the frame about to be coded makes.
 */
function applyFeedback(timestamp) {
  let repairing = false;
  while (feedback.length && feedback[0].at <= timestamp) {
    const item = feedback.shift();
    if (item.kind === 'ack') encoder.acknowledge_frame(item.frame);
    else if (item.kind === 'loss') { encoder.report_lost_frame(item.frame); repairing = true; }
    else { encoder.request_keyframe(); keyRequested = true; repairing = true; }
  }
  return repairing;
}
/** Over an RTP link: the coded frame's packets and each one's layer ({ packets, layers }); else null. */
function packetize(packet, timestamp) {
  if (!sender) return null;
  const count = sender.packetize(encoder, packet, timestamp * MICROSECONDS_PER_MS);
  const packets = [], layers = [];
  for (let i = 0; i < count; i++) {
    packets.push(sender.packet(i).buffer);
    layers.push([sender.packet_spatial_id(i), sender.packet_temporal_id(i)]);
  }
  return { packets, layers };
}
/** The loss feedback the decode worker sent: kept in arrival order. */
function receiveFeedback({ data }) {
  if (data.type !== 'feedback') return;
  feedback.push(...data.items);
  feedback.sort((a, b) => a.at - b.at);
}
function configureRate(c) {
  if (c.rateMode === 'cbr') encoder.set_rate_control(c.targetKbps, c.peakKbps, c.bufferMs);
  else encoder.disable_rate_control();
}
/** (Re)create the encoder for input of `colour`. */
function createEncoder(colour) {
  const c = config;
  encoder?.free();
  encoder = new module.BrowserEncoder(c.codec, c.width, c.height, c.qp, c.fps, c.keyInterval, c.preset, c.overrides, c.filmGrain, colour,
    Boolean(c.stillCamera), c.spatialLayers);
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
      const link = config.link;
      sender = link && link.mode !== 'bitstream' ? new module.RtpSender(config.codec, link.network.mtu, DEPENDENCY_DESCRIPTOR_ID) : null;
      decoderPort = message.port;
      decoderPort.onmessage = receiveFeedback;
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
      decoderPort.onmessage = receiveFeedback;
      self.postMessage({ type: 'decoder-connected' });
    } else if (message.type === 'keyframe') {
      encoder.request_keyframe();
      keyRequested = true;
    } else if (message.type === 'rate') {
      config = message.config;
      configureRate(config);
      rateApplied = { rateMode: config.rateMode, targetKbps: config.targetKbps, peakKbps: config.peakKbps, bufferMs: config.bufferMs };
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
        bufferOverflows: encoder.buffer_overflows }, coded.flatMap(c => [c.packet.buffer, c.source, ...(c.rtp?.packets ?? [])]));
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}
self.onmessage = receive;
