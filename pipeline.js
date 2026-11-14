// The page side of the comparison pipeline: a capture worker that converts
// each captured picture for every view at that view's own size
// (capture-worker.js), and per view either a live encode worker and a decode
// worker joined by a MessageChannel (encode-worker.js, decode-worker.js), or,
// for a pre-encoded reference stream, the decode worker alone, to which the
// page passes the stream's packets with each captured frame. The page never
// reads pixels back: it hands the capture worker a picture (capture.js).
//
// Frames are captured on the shared source timeline (timeline.js). Each view
// codes only the ticks of its own frame rate and holds its last picture in
// between; a pre-encoded view shows the stream's frame for that tick, first
// decoding any frames of the stream the capture skipped (each predicts from
// the one before). Several captures may be in flight at once
// (MAX_FRAMES_IN_FLIGHT), so each stage works on its own frame: encoders on
// frame N+1 while decoders finish frame N. Results are matched by capture id
// and released in capture order once every view that codes that tick has
// decoded it, so each view shows the newest tick it has, and views at the same
// rate show the same tick. The page captures nothing while the queue is full
// (`full`), which bounds the added latency.
//
// Switching analysis on or off (analysis-view.js) changes only the decoders'
// module (`switchDecoders`): once no capture is in flight, each view starts a
// decode worker on the other build and has it decode the view's stream again
// from its last key frame up to the picture on screen, so the new decoder holds
// the same references and shows the same picture (the bitstream is unchanged).
// Live encoders keep running and are given the new decoder's port. A live view
// keeps the packets since its last key frame for this (`sinceKey`); a
// pre-encoded view has its whole stream.
import { codesTick, tickAt, tickMs } from './timeline.js';
import { framesToDecode, streamFrameAtTick } from './reference.js';
import { SourcePicture } from './colour.js';
import { compiledAnalysisModule, compiledModule } from './wasm.js';

/**
 * Frames captured but not yet decoded by both views: one per stage (capture
 * worker, encoders, decoders, page), so every stage can work at once. Three
 * left a stage idle once capture became its own worker: headless Chromium,
 * H.264 vs AV1 Default at 30 fps, three interleaved rounds, delivered fps
 * 854x480 23.3 15.6 16.0 -> 23.3 28.0 22.5, 1280x720 13.2 10.5 15.9 -> 15.5 15.2 13.9.
 */
export const MAX_FRAMES_IN_FLIGHT = 4;
const START_TIMEOUT_MS = 30000;
const LOAD_FAILED = 'Could not load the WASM codec. Build the demo with tools/web/build.sh, then serve the web directory over HTTP.';
/**
 * A live view keeps at most this many bytes of packets since its last key frame
 * for re-decoding (one key interval: 30 s at the default 300 kbps is ~1 MiB).
 * Past it the view cannot switch decoders in place until its next key frame,
 * and switching analysis restarts the comparison instead.
 */
const MAX_REPLAY_BYTES = 64 * 1024 * 1024;

const startWorker = script => new Worker(new URL(script, import.meta.url), { type: 'module' });

/** What every view has: its workers (the decode worker last) and the panel it shows on. */
class View {
  constructor(panel, workers, onMessage, onFailure) {
    this.panel = panel;
    this.workers = workers;
    this.decoder = workers.at(-1);
    this.onMessage = onMessage;
    this.failed = event => onFailure(new Error(event.message || LOAD_FAILED));
    this.ready = Promise.all(workers.map(worker => new Promise(resolve => {
      worker.onmessage = ({ data }) => { if (data.type === 'ready') resolve(); else onMessage(this, data); };
      worker.onerror = this.failed;
    })));
  }
  /** Whether analysis is wanted for this view (analysis-view.js). */
  get analysisWanted() { return Boolean(this.panel.analysis?.enabled); }
  /**
   * Start the decode worker on `port`, where packets will arrive, with the
   * compiled codec module: the analysis build when the view's analysis is on
   * (analysis-view.js). `warmUp`: run the decoder's start-up warm-up (warmup.js).
   */
  initDecoder(port, { wasmModule, analysisModule }, warmUp = true) {
    this.analysing = this.analysisWanted;
    this.decoder.postMessage({ type: 'init', codec: this.panel.current.codec, port, analysis: this.analysing, warmUp,
      wasmModule: this.analysing ? analysisModule : wasmModule, operatingPoint: +(this.panel.operatingPoint?.value ?? 0) }, [port]);
  }
  /**
   * Replace the decode worker with one on the module analysis now wants, and
   * have it decode `replayFrames()` again; resolves to the result of the last of
   * them (the picture on screen, decode-worker.js 'replayed'), or null when the
   * view has shown nothing yet. Only while no capture is in flight.
   */
  async switchDecoder(modules) {
    const old = this.decoder, worker = startWorker('./decode-worker.js');
    const channel = new MessageChannel();
    const replayed = new Promise((resolve, reject) => {
      worker.onmessage = ({ data }) => {
        if (data.type === 'ready') this.sendReplay(worker, resolve);
        else if (data.type === 'replayed') resolve(data);
        else if (data.type === 'error') reject(new Error(`Version ${this.panel.letter}: ${data.message}`));
      };
      worker.onerror = event => reject(new Error(event.message || LOAD_FAILED));
    });
    const analysing = this.analysing;
    this.decoder = worker;
    this.initDecoder(channel.port1, modules, false);
    let result;
    try { result = await replayed; } catch (error) {
      worker.terminate(); this.decoder = old; this.analysing = analysing;
      throw error;
    }
    worker.onmessage = ({ data }) => this.onMessage(this, data);
    worker.onerror = this.failed;
    old.terminate();
    this.workers[this.workers.length - 1] = worker;
    await this.connectDecoder(channel.port2);
    return result.empty ? null : result;
  }
  /** Post the frames to decode again to a new decode worker, with the shown frame's source for its PSNR. */
  sendReplay(worker, resolve) {
    const frames = this.replayFrames(), original = this.panel.original;
    if (!frames.length) { resolve({ empty: true }); return; }
    const coded = frames.map(({ packet, key }) => ({ packet: packet.slice(), key, repeat: false, quantizer: null, encodeMs: null }));
    if (original) Object.assign(coded.at(-1), { source: original.i420.slice().buffer, colour: original.colour });
    worker.postMessage({ type: 'replay', coded, display: this.panel.renderer.format, tick: +this.panel.badge.dataset.tick },
      coded.flatMap(c => [c.packet.buffer, ...(c.source ? [c.source] : [])]));
  }
  /** Whether `switchDecoders` can re-decode this view's picture (see MAX_REPLAY_BYTES). */
  get replayable() { return true; }
  /** Note a decoded result's coded frames for `replayFrames` (a pre-encoded view has its stream). */
  retain() {}
  /** Ask for a key frame (only a live encoder can). */
  requestKeyframe() {}
  /** Apply the panel's current rate settings (only a live encoder has any). */
  updateRate() {}
  terminate() { for (const worker of this.workers) worker.terminate(); }
}

/** A live nanocodec encode: the encoder takes frames from the capture worker and posts packets straight to the decoder. */
class LiveView extends View {
  constructor(panel, onMessage, onFailure) {
    super(panel, [startWorker('./encode-worker.js'), startWorker('./decode-worker.js')], onMessage, onFailure);
    this.encoder = this.workers[0];
    // The packets since the last key frame ({ packet, key }), and their
    // bytes; null past MAX_REPLAY_BYTES until the next key frame.
    this.sinceKey = [];
    this.sinceKeyBytes = 0;
    // The shown SVC layer changes on the fly: only the decode worker hears of it.
    panel.operatingPoint?.addEventListener('change', () => {
      this.decoder.postMessage({ type: 'operating-point', operatingPoint: +panel.operatingPoint.value });
      if (+panel.operatingPoint.value === 0) this.encoder.postMessage({ type: 'keyframe' });
    });
  }
  /** Start both workers; the encoder takes its frames from `capturePort`. Resolves once both are ready. */
  init(capturePort, modules) {
    const channel = new MessageChannel();
    this.initDecoder(channel.port1, modules);
    this.encoder.postMessage({ type: 'init', config: this.panel.current, port: channel.port2, capturePort, wasmModule: modules.wasmModule },
      [channel.port2, capturePort]);
    return this.ready;
  }
  codes(tick) { return codesTick(this.panel.current.step, tick); }
  requestKeyframe() { this.encoder.postMessage({ type: 'keyframe' }); }
  updateRate() { this.encoder.postMessage({ type: 'rate', config: this.panel.current }); }
  /** Keep a decoded result's coded frames (frame holds code nothing) for `replayFrames`. */
  retain({ earlier = [], ...shown }) {
    for (const coded of [...earlier, shown]) {
      if (coded.repeat) continue;
      if (coded.key) { this.sinceKey = []; this.sinceKeyBytes = 0; }
      if (!this.sinceKey) continue;
      this.sinceKeyBytes += coded.packet.byteLength;
      if (this.sinceKeyBytes > MAX_REPLAY_BYTES) this.sinceKey = null;
      else this.sinceKey.push({ packet: coded.packet, key: coded.key });
    }
  }
  get replayable() { return this.sinceKey !== null; }
  replayFrames() { return this.sinceKey; }
  /**
   * The encoder posts its packets to the new decoder from its next frame on;
   * resolves once it has the port, so no later capture can reach the old one
   * (captures come on another port, in no order with this message).
   */
  connectDecoder(port) {
    return new Promise(resolve => {
      this.decoderConnected = resolve;
      this.encoder.postMessage({ type: 'decoder', port }, [port]);
    });
  }
}

/** A pre-encoded reference stream (reference.js), decoded in the browser. */
class ReferenceView extends View {
  constructor(panel, onMessage, onFailure) {
    super(panel, [startWorker('./decode-worker.js')], onMessage, onFailure);
    // The stream of the settings the view started with (panel.js `prepare`).
    this.stream = panel.stream;
    // The last stream frame sent to the decoder (null before the first).
    this.last = null;
  }
  /** Start the decoder; source frames arrive on `capturePort` and go on with the stream's packets. */
  init(capturePort, modules) {
    const channel = new MessageChannel();
    this.initDecoder(channel.port1, modules);
    this.port = channel.port2;
    this.capturePort = capturePort;
    capturePort.onmessage = ({ data }) => this.submit(data);
    return this.ready;
  }
  codes(tick) { return streamFrameAtTick(this.stream, tick) !== null; }
  /** Send the decoder every stream frame up to the one of this capture's tick, with the captured frame of that one. */
  submit({ id, timestamp, frame: captured, display }) {
    const tick = tickAt(timestamp), target = streamFrameAtTick(this.stream, tick), { packets, step, entry } = this.stream;
    const frames = framesToDecode(this.last, target);
    this.last = target;
    // Packets are views into the fetched file: each frame gets its own copy to transfer.
    const coded = frames.map(frame => ({
      timestamp: tickMs(tick - (target - frame) * step), packet: packets[frame].slice(), key: frame % entry.keyInterval === 0,
      repeat: false, quantizer: null, encodeMs: null,
      capture: frame === target ? { frame: captured, width: this.panel.current.width, height: this.panel.current.height } : null,
    }));
    this.port.postMessage({ id, display, coded, bufferFullness: null, bufferOverflows: 0 }, [captured.data, ...coded.map(c => c.packet.buffer)]);
  }
  /** The stream frames from the key frame at or before the last one decoded up to it. */
  replayFrames() {
    if (this.last === null) return [];
    const { packets, entry } = this.stream, first = this.last - this.last % entry.keyInterval;
    return Array.from({ length: this.last - first + 1 }, (_, i) => ({ packet: packets[first + i], key: i === 0 }));
  }
  /** Stream packets go to the new decoder from the next capture on. */
  connectDecoder(port) { this.port.close(); this.port = port; }
  terminate() { this.port?.close(); this.capturePort?.close(); super.terminate(); }
}

export class Comparison {
  /**
   * `onFrame(results, originals, tick)`: every view that codes `tick` decoded it
   * (`results[i]` is view i's decode-worker result, or null for a view that holds
   * its picture; `originals[i]` view i's source frame, a colour.js SourcePicture, or null).
   * `onError(error)`: a worker failed after start-up.
   */
  constructor(panels, { onFrame, onError }) {
    this.onFrame = onFrame;
    this.onError = onError;
    this.closed = false;
    // Rejects start() while the workers are starting.
    this.startFailed = null;
    this.nextId = 0;
    // Capture id -> { tick, coding, results }, in capture order (Map keeps insertion order).
    this.inFlight = new Map();
    this.views = panels.map(panel => {
      const ViewKind = panel.current.source === 'reference' ? ReferenceView : LiveView;
      return new ViewKind(panel, (view, data) => this.receive(view, data), error => this.fail(error));
    });
    this.capture = startWorker('./capture-worker.js');
    this.capture.onmessage = ({ data }) => { if (!this.closed) this.fail(new Error(`Capture: ${data.message}`)); };
    this.capture.onerror = event => this.fail(new Error(event.message || 'The capture worker failed.'));
  }
  /** Start every worker; rejects on the first error or after a timeout. */
  start() {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('The WASM worker did not start. Check the build and browser console.')), START_TIMEOUT_MS);
      this.startFailed = error => { clearTimeout(timeout); reject(error); };
      const ports = this.views.map(() => new MessageChannel());
      const views = this.views.map(({ panel }, i) => ({ width: panel.current.width, height: panel.current.height, port: ports[i].port1 }));
      this.capture.postMessage({ type: 'init', views }, ports.map(p => p.port1));
      const analysis = this.views.some(({ panel }) => panel.analysis?.enabled);
      Promise.all([compiledModule(), analysis ? compiledAnalysisModule() : null])
        .then(([wasmModule, analysisModule]) => Promise.all(this.views.map((view, i) => view.init(ports[i].port2, { wasmModule, analysisModule }))))
        .then(() => {
          clearTimeout(timeout);
          this.startFailed = null;
          resolve();
        }, error => this.fail(error));
    });
  }
  /** Whether the queue is full: the page should not capture until a frame completes. */
  get full() { return this.inFlight.size >= MAX_FRAMES_IN_FLIGHT; }
  /** Whether no capture is in flight (every view has shown every frame sent). */
  get idle() { return this.inFlight.size === 0; }
  /** Whether some view's decoder is not on the module analysis now wants. */
  get decodersStale() { return this.views.some(view => view.analysing !== view.analysisWanted); }
  /** Whether every view can switch its decoder in place (`switchDecoders`). */
  get canSwitchDecoders() { return this.views.every(view => view.replayable); }
  /**
   * Put every view whose decoder is on the wrong module (analysis switched) on
   * the right one, without restarting: see the file comment. Call only when
   * `idle`. Resolves to each view's re-decoded result (or null: not switched,
   * or nothing shown yet).
   */
  async switchDecoders() {
    const [wasmModule, analysisModule] = await Promise.all([compiledModule(), this.views.some(v => v.analysisWanted) ? compiledAnalysisModule() : null]);
    return Promise.all(this.views.map(view => (view.analysing === view.analysisWanted ? null : view.switchDecoder({ wasmModule, analysisModule }))));
  }
  /** Which views code `tick`. */
  coding(tick) { return this.views.map(view => view.codes(tick)); }
  /**
   * Send the pictures of `tick` to the views that code it (`coding`): `shots`
   * holds one per source, `{ picture, width, height, views }` (capture.js
   * `snapshot`, its size, the view indices it is for). The capture worker takes
   * them over and closes them.
   */
  send(tick, shots, coding) {
    const id = ++this.nextId;
    this.inFlight.set(id, { tick, coding, results: this.views.map(() => null) });
    // Each frame names the format the view's renderer draws, which changes if a GL context is lost.
    const displays = this.views.map(({ panel }) => panel.renderer.format);
    Promise.all(shots.map(shot => shot.picture)).then(pictures => {
      if (this.closed) { for (const picture of pictures) picture.close(); return; }
      const sources = shots.map((shot, i) => ({ picture: pictures[i], sourceWidth: shot.width, sourceHeight: shot.height, views: shot.views }));
      this.capture.postMessage({ type: 'frame', id, timestamp: tickMs(tick), sources, displays, coding }, pictures);
    }).catch(error => this.fail(error));
  }
  requestKeyframes() { for (const view of this.views) view.requestKeyframe(); }
  /** Apply view `index`'s current rate settings from its next frame on. */
  updateRate(index) { this.views[index].updateRate(); }
  /** Give view `index`'s decode worker back a displayed pixel buffer. */
  recycle(index, buffer) { this.views[index].decoder.postMessage({ type: 'recycle', buffer }, [buffer]); }
  terminate() {
    this.closed = true;
    this.capture.terminate();
    for (const view of this.views) view.terminate();
  }
  receive(view, data) {
    if (this.closed) return;
    if (data.type === 'error') {
      view.panel.element.classList.add('error');
      this.fail(new Error(`Version ${view.panel.letter}: ${data.message}`));
    } else if (data.type === 'ms-ssim') {
      view.panel.scoredMsSsim(data);
    } else if (data.type === 'decoder-connected') {
      view.decoderConnected();
    } else if (data.type === 'frame') {
      const frame = this.inFlight.get(data.id);
      if (!frame) return;
      if (!data.held) view.retain(data);
      frame.results[this.views.indexOf(view)] = data;
      this.release();
    }
  }
  /**
   * Hand every leading capture all its coding views have decoded to `onFrame`, in
   * capture order, with each view's source pixels (which its decode worker returns)
   * as its original; none for frame 0 held at start-up.
   */
  release() {
    for (const [id, frame] of this.inFlight) {
      if (frame.coding.some((coding, i) => coding && !frame.results[i])) return;
      this.inFlight.delete(id);
      const originals = frame.results.map((result, i) => {
        const { width, height } = this.views[i].panel.current;
        return result?.source ? new SourcePicture(new Uint8Array(result.source), width, height, result.colour) : null;
      });
      this.onFrame(frame.results, originals, frame.tick);
      if (this.closed) return;
    }
  }
  fail(error) {
    if (this.closed) return;
    if (this.startFailed) this.startFailed(error);
    else this.onError(error);
  }
}
