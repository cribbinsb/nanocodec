// Decoder half of one comparison view; encode-worker.js is the other half.
// Packets arrive from the view's encode worker over a MessageChannel port
// with the source frame each was coded from (packed I420 of the view size,
// and its colour). This worker decodes them, measures PSNR against that
// source plane by plane (in WASM, off the encoder's path) and posts the frame
// to show, with every timing, to the page. A view playing a pre-encoded stream
// has no encode worker: the page posts the stream's packets on the same port
// with the captured frame (pipeline.js), which this worker converts to the
// view's size itself as the encode worker would; frames decoded only to keep
// the references (no capture attached) are not scored.
//
// Buffers circulate instead of being allocated per frame: the source buffer
// goes back to the page with the result, and the page gives back each
// displayed pixel buffer once the view no longer needs it.
//
// Decoded frames go out in the format the page's renderer asks for with each
// frame (renderer.js): packed I420 for a WebGL view, which converts on the GPU,
// or RGBA converted here in WASM for the CPU path, both with the source's
// colour matrix and range (colour.js).
//
// Spatial SVC (the AV1 and VP9 `svc` presets): the decoder keeps only the
// operating point the viewer picked (BrowserDecoder.set_operating_point: AV1
// drops the OBUs of other layers, AV1 section 7.18; VP9, which has no layer
// ids, decodes only the base frame of each [S0, S1] superframe, VP9 Annex B,
// and shows it). Dropping to the base takes effect at
// once; the full-size layer predicts from its own earlier pictures, which were
// not decoded meanwhile, so going back up waits for the key frame the page
// asks the encoder for (pipeline.js) and switches just before decoding it.
// A smaller base picture is shown scaled to the stream size (nc-web).
import { MODULE_URL } from './wasm.js';
import { warmUp } from './warmup.js';
// Started at once, awaited by 'init' (the message handler must exist synchronously).
const codecModule = import(MODULE_URL);
let module, decoder, wasm;
// A pre-encoded view's converted source frame and its size (made on its first capture).
let source = null, sourceSize = '';
// The operating point to switch up to at the next key frame (null: none pending).
let pendingOperatingPoint = null;
// The colour frames are shown in: the latest source's (colour.js).
let displayColour = 'bt709';
// Set by the first real packet: start-up warm-up (warmup.js) stops then.
let realData = false;
// Returned display buffers kept for reuse: one per frame the page can hold in
// flight (MAX_FRAMES_IN_FLIGHT, pipeline.js) or on screen, so none is allocated in steady state.
const MAX_SPARE_PIXELS = 5;
const sparePixels = [];
/** Copy the decoder's latest frame, in `format` (RGBA in `colour`), out of WASM memory into a reused buffer. */
function decodedPixels(format, colour) {
  const { width, height } = decoder;
  const length = format === 'i420' ? width * height + 2 * Math.ceil(width / 2) * Math.ceil(height / 2) : width * height * 4;
  // Convert first: it may grow WASM memory, which detaches older buffers.
  const offset = format === 'i420' ? decoder.i420_frame() : decoder.rgba_frame(colour);
  const view = new Uint8Array(wasm.memory.buffer, offset, length);
  const match = sparePixels.findIndex(p => p.length === length);
  const pixels = match >= 0 ? sparePixels.splice(match, 1)[0] : new Uint8Array(length);
  pixels.set(view);
  return pixels;
}
/**
 * Decode one coded frame and score it against its source. `decodeMs` is the
 * decoder alone; `decodeStageMs` is all the time the frame held this worker.
 * A repeat (frame hold: nothing coded) shows the last decoded frame again.
 */
function decodeFrame({ capture, packet, ...coded }) {
  const start = performance.now();
  if (coded.key && pendingOperatingPoint !== null) {
    decoder.set_operating_point(pendingOperatingPoint);
    pendingOperatingPoint = null;
  }
  if (!coded.repeat) decoder.decode(packet);
  const decoded = performance.now();
  if (capture) Object.assign(coded, convertCapture(capture));
  const psnr = coded.source ? decoder.psnr(new Uint8Array(coded.source)) : null;
  return { ...coded, packet, psnr, width: decoder.width, height: decoder.height,
    streamWidth: decoder.stream_width, streamHeight: decoder.stream_height, spatialId: decoder.spatial_id,
    decodeMs: decoded - start, decodeStageMs: performance.now() - start };
}
/** A pre-encoded view's captured frame at the view's size: `{ source, colour }` as the encode worker sends them. */
function convertCapture({ frame, width, height }) {
  if (sourceSize !== `${width}x${height}`) {
    source?.free();
    source = new module.SourceFrame(width, height);
    sourceSize = `${width}x${height}`;
  }
  if (frame.kind === 'planes') {
    source.from_planes(new Uint8Array(frame.data), frame.format, frame.colour, frame.width, frame.height, Uint32Array.from(frame.layout));
  } else {
    source.from_rgba(new Uint8Array(frame.data));
  }
  const offset = source.packed();
  return { source: new Uint8Array(wasm.memory.buffer, offset, source.packed_length).slice().buffer, colour: source.colour };
}
/** One frame from the encode worker: decode it and post it to the page. */
function receivePacket({ data: message }) {
  realData = true;
  try {
    if (message.held) {
      self.postMessage({ type: 'frame', id: message.id, held: true });
      return;
    }
    // Several coded frames: frame 0 after the start-up lookahead, or the frames
    // a pre-encoded stream skipped; the last is shown, the `earlier` ones counted.
    const decoded = message.coded.map(decodeFrame);
    const result = decoded.at(-1), earlier = decoded.slice(0, -1);
    const displayStart = performance.now();
    const format = message.display === 'i420' ? 'i420' : 'rgba';
    // The last colour seen (frames decoded only for their references carry none).
    if (result.colour) displayColour = result.colour;
    const pixels = decodedPixels(format, displayColour);
    result.decodeStageMs += performance.now() - displayStart;
    const transfer = [pixels.buffer, ...decoded.flatMap(d => [d.packet.buffer, d.source]).filter(Boolean)];
    self.postMessage({ type: 'frame', id: message.id, ...result, colour: displayColour, earlier, format, pixels, width: decoder.width,
      height: decoder.height, streamWidth: decoder.stream_width, streamHeight: decoder.stream_height,
      spatialId: decoder.spatial_id, bufferFullness: message.bufferFullness, bufferOverflows: message.bufferOverflows }, transfer);
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}
self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      module = await codecModule;
      wasm = await module.default();
      decoder = new module.BrowserDecoder(message.codec);
      if (message.operatingPoint) decoder.set_operating_point(message.operatingPoint);
      message.port.onmessage = receivePacket;
      self.postMessage({ type: 'ready' });
      // Not awaited: real packets are handled as they come and stop it.
      if (message.warmUp !== false) {
        warmUp(module, message.codec, () => realData)
          .then(report => self.postMessage({ type: 'warmed', ...report }))
          .catch(error => console.warn('decoder warm-up failed:', error));
      }
    } else if (message.type === 'operating-point') {
      // Down to the base at once; up to the full size at the next key frame.
      if (message.operatingPoint > 0) {
        pendingOperatingPoint = null;
        decoder.set_operating_point(message.operatingPoint);
      } else {
        pendingOperatingPoint = message.operatingPoint;
      }
    } else if (message.type === 'recycle') {
      if (sparePixels.length < MAX_SPARE_PIXELS) sparePixels.push(new Uint8Array(message.buffer));
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
