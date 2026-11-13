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
import { codesTick, tickAt, tickMs } from './timeline.js';
import { framesToDecode, streamFrameAtTick } from './reference.js';
import { SourcePicture } from './colour.js';
import { compiledModule } from './wasm.js';

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

const startWorker = script => new Worker(new URL(script, import.meta.url), { type: 'module' });

/** What every view has: its workers (the decode worker last) and the panel it shows on. */
class View {
  constructor(panel, workers, onMessage, onFailure) {
    this.panel = panel;
    this.workers = workers;
    this.decoder = workers.at(-1);
    const failed = event => onFailure(new Error(event.message || LOAD_FAILED));
    this.ready = Promise.all(workers.map(worker => new Promise(resolve => {
      worker.onmessage = ({ data }) => { if (data.type === 'ready') resolve(); else onMessage(this, data); };
      worker.onerror = failed;
    })));
  }
  /** Start the decode worker on `port`, where packets will arrive, with the compiled codec module. */
  initDecoder(port, wasmModule) {
    this.decoder.postMessage({ type: 'init', codec: this.panel.current.codec, port, wasmModule,
      operatingPoint: +(this.panel.operatingPoint?.value ?? 0) }, [port]);
  }
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
    // The shown SVC layer changes on the fly: only the decode worker hears of it.
    panel.operatingPoint?.addEventListener('change', () => {
      this.decoder.postMessage({ type: 'operating-point', operatingPoint: +panel.operatingPoint.value });
      if (+panel.operatingPoint.value === 0) this.encoder.postMessage({ type: 'keyframe' });
    });
  }
  /** Start both workers; the encoder takes its frames from `capturePort`. Resolves once both are ready. */
  init(capturePort, wasmModule) {
    const channel = new MessageChannel();
    this.initDecoder(channel.port1, wasmModule);
    this.encoder.postMessage({ type: 'init', config: this.panel.current, port: channel.port2, capturePort, wasmModule },
      [channel.port2, capturePort]);
    return this.ready;
  }
  codes(tick) { return codesTick(this.panel.current.step, tick); }
  requestKeyframe() { this.encoder.postMessage({ type: 'keyframe' }); }
  updateRate() { this.encoder.postMessage({ type: 'rate', config: this.panel.current }); }
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
  init(capturePort, wasmModule) {
    const channel = new MessageChannel();
    this.initDecoder(channel.port1, wasmModule);
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
      compiledModule()
        .then(wasmModule => Promise.all(this.views.map((view, i) => view.init(ports[i].port2, wasmModule))))
        .then(() => {
          clearTimeout(timeout);
          this.startFailed = null;
          resolve();
        }, error => this.fail(error));
    });
  }
  /** Whether the queue is full: the page should not capture until a frame completes. */
  get full() { return this.inFlight.size >= MAX_FRAMES_IN_FLIGHT; }
  /** Which views code `tick`. */
  coding(tick) { return this.views.map(view => view.codes(tick)); }
  /**
   * Send the picture (capture.js `snapshot`, of a `sourceWidth` x `sourceHeight`
   * source) captured at `tick` to the views that code it (`coding`); the capture
   * worker takes it over and closes it.
   */
  send(tick, picture, sourceWidth, sourceHeight, coding) {
    const id = ++this.nextId;
    this.inFlight.set(id, { tick, coding, results: this.views.map(() => null) });
    // Each frame names the format the view's renderer draws, which changes if a GL context is lost.
    const displays = this.views.map(({ panel }) => panel.renderer.format);
    Promise.resolve(picture).then(frame => {
      if (this.closed) { frame.close(); return; }
      this.capture.postMessage({ type: 'frame', id, timestamp: tickMs(tick), picture: frame, sourceWidth, sourceHeight, displays, coding }, [frame]);
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
    } else if (data.type === 'frame') {
      const frame = this.inFlight.get(data.id);
      if (!frame) return;
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
