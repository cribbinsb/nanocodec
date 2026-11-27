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
// Spatial SVC (AV1 and VP9 with 2 or 3 spatial layers, config `spatialLayers`): the decoder keeps only the
// operating point the viewer picked (BrowserDecoder.set_operating_point: AV1
// drops the OBUs of other layers, AV1 section 7.18; VP9, which has no layer
// ids, decodes only the base frame of each [S0, S1] superframe, VP9 Annex B,
// and shows it). Dropping to the base takes effect at
// once; the full-size layer predicts from its own earlier pictures, which were
// not decoded meanwhile, so going back up waits for the key frame the page
// asks the encoder for (pipeline.js) and switches just before decoding it.
// A smaller base picture is shown scaled to the stream size (nc-web).
//
// Analysis (analysis-view.js): a view with analysis on gets a decode worker
// whose decoder keeps each picture's block records
// (BrowserDecoder.set_analysis), and every shown frame carries them as one
// packed Int16Array buffer (`analysis`), with `analysisState` 'on', or
// 'unsupported' for codecs whose decoder has none yet. Switching analysis
// replaces this worker with one recording or not (pipeline.js
// `switchDecoders`): the page posts it a 'replay' of the stream from its last
// key frame to the picture on screen, decoded here at once and answered with
// that picture ('replayed', as a 'frame'). With analysis on, a shown frame
// that has its source also carries the PSNR map (`errorMap`, error-map.js):
// the decoder's squared error per block against that source (nc-web
// `error_map`, which reuses the source the frame's `psnr` call unpacked),
// turned into dB here, off the page's thread. MS-SSIM is scored by a metrics
// worker this worker starts (metrics-worker.js): the newest shown frame
// whenever it is free, sent to the page as 'ms-ssim' with its tick.
// RTP link (config.link, rtp-link.js): the coded frames arrive with their RTP
// packets, which go through the link's SFU and network emulator to an RTP
// receiver; what it delivers by the capture's source time is decoded, and the
// view shows the newest decoded picture. A capture for which nothing decodes
// shows the last decoded picture again (frozen, `link.frozen`): what the
// decoder really outputs, no concealment. PSNR is measured only on pictures
// decoded at that capture, against their own source frame. The receiver's
// loss feedback goes back to the encode worker on the packet port.
import { MODULE_URL } from './wasm.js';
import { RtpLink, rtpTimestamp } from './rtp-link.js';
import { warmUp } from './warmup.js';
import { errorMap, errorMapBuffers } from './error-map.js';
import { tickAt } from './timeline.js';
import { ViewSourceFrame } from './view-source-frame.js';
// Started at once, awaited by 'init' (the message handler must exist synchronously).
const codecModule = import(MODULE_URL);
let module, decoder, wasm;
// A pre-encoded view's captured frames, placed at the view's size (view-source-frame.js).
let viewSource = null;
// The active operating point and any operating point waiting for the next key frame.
let currentOperatingPoint = 0, pendingOperatingPoint = null;
// Analysis: 'off', 'on' or 'unsupported' (this codec's decoder records nothing).
let analysisState = 'off';
// The colour frames are shown in: the latest source's (colour.js).
let displayColour = 'bt709';
// Set by the first real packet: start-up warm-up (warmup.js) stops then.
let realData = false;
// The RTP link (null for the bitstream link), the packet port (feedback goes back on
// it), the coded frames sent and not yet decoded or given up, by RTP timestamp, and
// the last picture shown ({ pixels, format, ...sizes }) for frozen captures.
let link = null, packetPort = null, lastPicture = null;
const pending = new Map();
// Coded frames older than this (ms of source time) are given up: their source is dropped.
const PENDING_MS = 5000;
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
/** Switch on the decoder's analysis records. */
function startAnalysis() {
  analysisState = decoder.set_analysis(true) ? 'on' : 'unsupported';
}
/** The shown picture's packed block records, copied out of WASM memory (null without analysis). */
function analysisRecords() {
  if (analysisState !== 'on') return null;
  const offset = decoder.analysis_frame();
  return new Int16Array(wasm.memory.buffer, offset, decoder.analysis_length).slice().buffer;
}
// MS-SSIM (analysis mode): the metrics worker, whether it is scoring a frame,
// and the newest frame waiting for it ({ tick, width, height, reference, distorted }).
let metrics = null, metricsBusy = false, nextScore = null;
/**
 * A score starts at most this often (ms), whatever the frame rate: each costs
 * ~110 ms of CPU at 720p, which competes with the encoders and decoders on a
 * busy machine. Scoring every frame it could took a slowed 1280x720 + 640x360
 * comparison from 244 to 74 delivered frames (web/tests/psnr-map.py, 2026-11-14).
 * The newest frame waits for the next slot, so a paused frame is scored within it.
 */
const SCORE_INTERVAL_MS = 1000;
// When the last score started, and the timer for the next slot (0: none).
let lastScoreAt = -Infinity, scoreTimer = 0;
/** Start the metrics worker on the codec module (`wasmModule`, compiled by the page). */
function startMetrics(wasmModule) {
  metrics = new Worker(new URL('./metrics-worker.js', import.meta.url), { type: 'module' });
  metrics.onmessage = ({ data }) => {
    if (data.type === 'error') { self.postMessage(data); return; }
    self.postMessage({ type: 'ms-ssim', tick: data.tick, value: data.value, ms: data.ms });
    metricsBusy = false;
    sendScore();
  };
  metrics.postMessage({ type: 'init', wasmModule });
}
/** Keep the shown frame of `tick` (scored against `source`, packed I420) as the next to score, replacing an unscored one. */
function queueScore(tick, source) {
  if (!metrics || !source) return;
  const width = decoder.error_map_width, height = decoder.error_map_height;
  // After the frame's psnr call: the scored size is the source's. The copy may grow
  // WASM memory (detaching older views), so the view is made after it.
  const offset = decoder.scored_luma();
  const distorted = new Uint8Array(wasm.memory.buffer, offset, width * height).slice();
  const reference = new Uint8Array(source, 0, width * height).slice();
  nextScore = { tick, width, height, reference: reference.buffer, distorted: distorted.buffer };
  sendScore();
}
/** Hand the metrics worker the waiting frame if it is free and its slot has come (SCORE_INTERVAL_MS). */
function sendScore() {
  if (metricsBusy || !nextScore) return;
  const wait = lastScoreAt + SCORE_INTERVAL_MS - performance.now();
  if (wait > 0) {
    scoreTimer ||= setTimeout(() => { scoreTimer = 0; sendScore(); }, wait);
    return;
  }
  lastScoreAt = performance.now();
  metricsBusy = true;
  const score = nextScore;
  nextScore = null;
  metrics.postMessage({ type: 'score', ...score }, [score.reference, score.distorted]);
}
/** The PSNR map of the decoder's latest frame against `result`'s source, scored last (null without analysis or a source). */
function blockErrors(result) {
  if (analysisState !== 'on' || !result.source) return null;
  const offset = decoder.error_map();
  const sums = new Uint32Array(wasm.memory.buffer, offset, decoder.error_map_length);
  return errorMap(sums, decoder.error_map_width, decoder.error_map_height);
}
/**
 * Decode one coded frame and score it against its source. `decodeMs` is the
 * decoder alone; `decodeStageMs` is all the time the frame held this worker.
 * A repeat (frame hold: nothing coded) shows the last decoded frame again.
 */
function decodeFrame({ capture, packet, ...coded }) {
  const start = performance.now();
  if (coded.key && pendingOperatingPoint !== null) {
    currentOperatingPoint = pendingOperatingPoint;
    decoder.set_operating_point(pendingOperatingPoint);
    pendingOperatingPoint = null;
  }
  if (!coded.repeat) decoder.decode(packet);
  const decoded = performance.now();
  const shown = coded.repeat || !('frame_shown' in decoder) || decoder.frame_shown;
  if (shown && capture) Object.assign(coded, viewSource.place(capture));
  const psnr = shown && coded.source && decoder.width > 0 ? decoder.psnr(new Uint8Array(coded.source)) : null;
  return { ...coded, packet, psnr, shown, width: decoder.width, height: decoder.height,
    streamWidth: decoder.stream_width, streamHeight: decoder.stream_height, spatialId: decoder.spatial_id,
    decodeMs: decoded - start, decodeStageMs: performance.now() - start };
}
/**
 * Decode a message's coded frames (`coded`; the page's renderer format in
 * `display`) and post the last as `type`, with its pixels, records and every
 * timing, to the page.
 */
function decodeAndPost(type, message) {
  // Several coded frames: frame 0 after the start-up lookahead, the frames a
  // pre-encoded stream skipped, or a replay; the last is shown, the `earlier` ones counted.
  const decoded = message.coded.map(decodeFrame);
  const result = decoded.at(-1), earlier = decoded.slice(0, -1);
  if (!result.shown && decoder.width === 0) {
    const transfer = decoded.flatMap(d => [d.packet.buffer, d.source]).filter(Boolean);
    self.postMessage({ type, id: message.id, held: true }, transfer);
    return;
  }
  const displayStart = performance.now();
  const format = message.display === 'i420' ? 'i420' : 'rgba';
  // The last colour seen (frames decoded only for their references carry none).
  if (result.colour) displayColour = result.colour;
  const pixels = decodedPixels(format, displayColour);
  // After the pixels: packing may grow WASM memory too, and is copied at once.
  const analysis = analysisRecords(), errors = blockErrors(result);
  // A replay's frames carry no timestamp: the page names the tick on screen.
  queueScore(message.tick ?? tickAt(result.timestamp), result.source);
  result.decodeStageMs += performance.now() - displayStart;
  const transfer = [pixels.buffer, ...decoded.flatMap(d => [d.packet.buffer, d.source]).filter(Boolean), ...(analysis ? [analysis] : []),
    ...(errors ? errorMapBuffers(errors) : [])];
  self.postMessage({ type, id: message.id, ...result, colour: displayColour, earlier, format, pixels, width: decoder.width,
    height: decoder.height, streamWidth: decoder.stream_width, streamHeight: decoder.stream_height,
    spatialId: decoder.spatial_id, bufferFullness: message.bufferFullness, bufferOverflows: message.bufferOverflows,
    analysis, analysisState, errorMap: errors }, transfer);
}
/** Copy of the shown picture's pixels (the posted buffer goes to the page). */
function keepPicture(result, pixels) {
  lastPicture = { pixels: pixels.slice(), format: result.format, width: result.width, height: result.height,
    streamWidth: result.streamWidth, streamHeight: result.streamHeight, spatialId: result.spatialId };
}
/** Which layer the shown picture is (the link monitor's "On screen"). */
function layerName(spatialId, spatialLayers) {
  if (spatialLayers < 2) return 'full picture';
  return spatialId === spatialLayers - 1 ? `full size (S${spatialId})` : `base layer S${spatialId}, upscaled`;
}
/**
 * One capture over the RTP link: send its coded frames' packets, decode what the
 * receiver delivers by the capture's time, and post the picture on screen.
 */
function receiveOverRtp(message) {
  const sent = message.coded.filter(c => !c.repeat && c.rtp);
  for (const coded of message.coded) {
    if (coded.repeat || !coded.rtp) continue;
    link.send(coded.rtp, coded.timestamp, coded.key);
    if (coded.repairing && !coded.key) link.repaired(coded.timestamp);
    pending.set(rtpTimestamp(coded.timestamp), coded);
  }
  const now = message.coded.at(-1).timestamp;
  // Whether a picture was decoded at this capture, and the coded frame it came from (its source scores it).
  let decodedNow = false, shown = null, refused = false;
  const decodedIds = [];
  for (const { unit, timestamp, frameIds } of link.receive(now)) {
    const coded = pending.get(timestamp);
    pending.delete(timestamp);
    try {
      decoder.decode(unit);
    } catch {
      refused = true;
      continue;
    }
    decodedIds.push(...frameIds);
    if (!('frame_shown' in decoder) || decoder.frame_shown) { decodedNow = true; shown = coded ?? null; }
  }
  for (const [timestamp, coded] of pending) if (coded.timestamp < now - PENDING_MS) pending.delete(timestamp);
  const items = link.feedback(decodedIds, refused);
  if (items.length) packetPort.postMessage({ type: 'feedback', items });
  // The coded frames are the page's record of what was sent (bitrate, download).
  const record = sent.map(c => ({ ...c, rtp: null, psnr: null, decodeMs: null, source: null }));
  const transfer = record.map(c => c.packet.buffer);
  const state = layers => link.state(layerName(decoder.spatial_id, layers));
  const spatialLayers = link.layers.spatialLayers;
  if (decodedNow) {
    const start = performance.now();
    const psnr = shown?.source && decoder.width > 0 ? decoder.psnr(new Uint8Array(shown.source)) : null;
    if (shown?.colour) displayColour = shown.colour;
    const format = message.display === 'i420' ? 'i420' : 'rgba';
    const pixels = decodedPixels(format, displayColour);
    const result = { width: decoder.width, height: decoder.height, streamWidth: decoder.stream_width, streamHeight: decoder.stream_height,
      spatialId: decoder.spatial_id, format };
    keepPicture(result, pixels);
    self.postMessage({ type: 'frame', id: message.id, ...result, timestamp: now, key: Boolean(shown?.key), repeat: false, quantizer: shown?.quantizer ?? null,
      encodeMs: null, psnr, decodeMs: performance.now() - start, decodeStageMs: performance.now() - start, packet: new Uint8Array(0),
      earlier: record, colour: displayColour, pixels, source: shown?.source, bufferFullness: message.bufferFullness,
      bufferOverflows: message.bufferOverflows, analysisState, link: state(spatialLayers) }, [pixels.buffer, shown?.source, ...transfer].filter(Boolean));
    return;
  }
  if (!lastPicture) {
    self.postMessage({ type: 'frame', id: message.id, held: true, earlier: record, link: state(spatialLayers) }, transfer);
    return;
  }
  // Frozen (or between two frames still in flight): the last picture again.
  const pixels = lastPicture.pixels.slice();
  self.postMessage({ type: 'frame', id: message.id, ...lastPicture, pixels, timestamp: now, key: false, repeat: true, quantizer: null,
    encodeMs: null, psnr: null, decodeMs: null, decodeStageMs: 0, packet: new Uint8Array(0), earlier: record, colour: displayColour,
    bufferFullness: message.bufferFullness, bufferOverflows: message.bufferOverflows, analysisState, link: state(spatialLayers) },
  [pixels.buffer, ...transfer]);
}
/** One frame from the encode worker: decode it and post it to the page. */
function receivePacket({ data: message }) {
  realData = true;
  try {
    if (message.held) {
      self.postMessage({ type: 'frame', id: message.id, held: true });
      return;
    }
    if (link) receiveOverRtp(message);
    else decodeAndPost('frame', message);
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
}
self.onmessage = async ({ data: message }) => {
  try {
    if (message.type === 'init') {
      module = await codecModule;
      // The page posts the module it compiled once (wasm.js compiledModule).
      wasm = await module.default({ module_or_path: message.wasmModule });
      decoder = new module.BrowserDecoder(message.codec);
      viewSource = new ViewSourceFrame(module, wasm);
      currentOperatingPoint = message.operatingPoint || 0;
      pendingOperatingPoint = null;
      if (currentOperatingPoint) decoder.set_operating_point(currentOperatingPoint);
      if (message.analysis) { startAnalysis(); startMetrics(message.wasmModule); }
      link = message.link && message.link.mode !== 'bitstream'
        ? new RtpLink(module, message.codec, message.link, message.layers, message.frameMs) : null;
      packetPort = message.port;
      message.port.onmessage = receivePacket;
      self.postMessage({ type: 'ready' });
      // Not awaited: real packets are handled as they come and stop it.
      if (message.warmUp !== false) {
        warmUp(module, message.codec, () => realData)
          .then(report => self.postMessage({ type: 'warmed', ...report }))
          .catch(error => console.warn('decoder warm-up failed:', error));
      }
    } else if (message.type === 'operating-point') {
      // Down to a smaller spatial layer at once; up to a larger one at the next key frame.
      if (message.operatingPoint >= currentOperatingPoint) {
        pendingOperatingPoint = null;
        currentOperatingPoint = message.operatingPoint;
        decoder.set_operating_point(message.operatingPoint);
      } else {
        pendingOperatingPoint = message.operatingPoint;
      }
    } else if (message.type === 'network') {
      link?.configure(message.network);
    } else if (message.type === 'replay') {
      realData = true;
      decodeAndPost('replayed', message);
    } else if (message.type === 'recycle') {
      if (sparePixels.length < MAX_SPARE_PIXELS) sparePixels.push(new Uint8Array(message.buffer));
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  }
};
