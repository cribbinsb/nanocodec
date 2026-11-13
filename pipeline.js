// The page side of the comparison pipeline: a capture worker that converts
// each captured picture for every view (capture-worker.js), per view an encode
// worker and a decode worker joined by a MessageChannel (encode-worker.js,
// decode-worker.js), and a short queue of frames in flight. The page never
// reads pixels back: it hands the capture worker a VideoFrame.
//
// Several frames may be in flight at once (MAX_FRAMES_IN_FLIGHT), so each
// stage works on its own frame: encoders on frame N+1 while decoders finish
// frame N. Every frame goes to both views; results are matched by frame id and
// released in capture order only once both views have decoded that frame, so
// the two views always show the same source frame. The page captures no new
// frame while the queue is full (`full`), which bounds the added latency.

/**
 * Frames captured but not yet decoded by both views: one per stage (capture
 * worker, encoders, decoders, page), so every stage can work at once. Three
 * left a stage idle once capture became its own worker: headless Chromium,
 * H.264 vs AV1 Default at 30 fps, three interleaved rounds, delivered fps
 * 854x480 23.3 15.6 16.0 -> 23.3 28.0 22.5, 1280x720 13.2 10.5 15.9 -> 15.5 15.2 13.9.
 */
export const MAX_FRAMES_IN_FLIGHT = 4;
const START_TIMEOUT_MS = 30000;

/**
 * A transferable picture of `source` (video element or canvas) for the capture
 * worker: a VideoFrame (WebCodecs), else a promised ImageBitmap.
 */
export function snapshot(source) {
  return typeof VideoFrame === 'function' ? new VideoFrame(source, { timestamp: 0 }) : createImageBitmap(source);
}

/** One view's pair of workers. */
class ViewWorkers {
  constructor(panel, onMessage, onFailure) {
    this.panel = panel;
    this.encoder = new Worker(new URL('./encode-worker.js', import.meta.url), { type: 'module' });
    this.decoder = new Worker(new URL('./decode-worker.js', import.meta.url), { type: 'module' });
    const failed = event => onFailure(new Error(event.message || 'Could not load the WASM codec. Build the demo with tools/web/build.sh, then serve the web directory over HTTP.'));
    // The shown SVC layer changes on the fly: only the decode worker hears of it.
    panel.operatingPoint?.addEventListener('change', () => {
      this.decoder.postMessage({ type: 'operating-point', operatingPoint: +panel.operatingPoint.value });
      if (+panel.operatingPoint.value === 0) this.encoder.postMessage({ type: 'keyframe' });
    });
    this.ready = Promise.all([this.encoder, this.decoder].map(worker => new Promise(resolve => {
      worker.onmessage = ({ data }) => { if (data.type === 'ready') resolve(); else onMessage(this, data); };
      worker.onerror = failed;
    })));
  }
  /**
   * Start both workers, joined by a MessageChannel; the encoder takes its frames
   * from `capturePort`. Resolves once both are ready.
   */
  init(capturePort) {
    const channel = new MessageChannel();
    this.decoder.postMessage({ type: 'init', codec: this.panel.current.codec, port: channel.port1,
      operatingPoint: +(this.panel.operatingPoint?.value ?? 0) }, [channel.port1]);
    this.encoder.postMessage({ type: 'init', config: this.panel.current, port: channel.port2, capturePort }, [channel.port2, capturePort]);
    return this.ready;
  }
  terminate() {
    this.encoder.terminate();
    this.decoder.terminate();
  }
}

export class Comparison {
  /**
   * `onFrame(results, original)`: both views decoded one frame (`results[i]`
   * is view i's decode-worker result; `original` the captured ImageData).
   * `onError(error)`: a worker failed after start-up.
   */
  constructor(panels, { onFrame, onError }) {
    this.onFrame = onFrame;
    this.onError = onError;
    this.closed = false;
    // Rejects start() while the workers are starting.
    this.startFailed = null;
    this.nextId = 0;
    // Frame id -> { results }, in capture order (Map keeps insertion order).
    this.inFlight = new Map();
    this.views = panels.map(panel => new ViewWorkers(panel, (view, data) => this.receive(view, data), error => this.fail(error)));
    this.capture = new Worker(new URL('./capture-worker.js', import.meta.url), { type: 'module' });
    this.capture.onmessage = ({ data }) => { if (!this.closed) this.fail(new Error(`Capture: ${data.message}`)); };
    this.capture.onerror = event => this.fail(new Error(event.message || 'The capture worker failed.'));
  }
  /** Start every worker; rejects on the first error or after a timeout. */
  start() {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('The WASM worker did not start. Check the build and browser console.')), START_TIMEOUT_MS);
      this.startFailed = error => { clearTimeout(timeout); reject(error); };
      const ports = this.views.map(() => new MessageChannel());
      const sizes = this.views.map(({ panel }) => ({ width: panel.current.width, height: panel.current.height }));
      this.capture.postMessage({ type: 'init', views: sizes.map((size, i) => ({ ...size, port: ports[i].port1 })) }, ports.map(p => p.port1));
      Promise.all(this.views.map((view, i) => view.init(ports[i].port2))).then(() => {
        clearTimeout(timeout);
        this.startFailed = null;
        resolve();
      });
    });
  }
  /** Whether the queue is full: the page should not capture until a frame completes. */
  get full() { return this.inFlight.size >= MAX_FRAMES_IN_FLIGHT; }
  /**
   * Send one captured picture (`snapshot`) of a `sourceWidth` x `sourceHeight`
   * source to both views; the capture worker takes it over and closes it.
   */
  send(picture, sourceWidth, sourceHeight, timestamp) {
    const id = ++this.nextId;
    this.inFlight.set(id, { results: this.views.map(() => null) });
    // Each frame names the format the view's renderer draws, which changes if a GL context is lost.
    const displays = this.views.map(({ panel }) => panel.renderer.format);
    Promise.resolve(picture).then(frame => {
      if (this.closed) { frame.close(); return; }
      this.capture.postMessage({ type: 'frame', id, timestamp, picture: frame, sourceWidth, sourceHeight, displays }, [frame]);
    }).catch(error => this.fail(error));
  }
  requestKeyframes() { for (const view of this.views) view.encoder.postMessage({ type: 'keyframe' }); }
  /** Apply view `index`'s current rate settings from its next frame on. */
  updateRate(index) {
    const view = this.views[index];
    view.encoder.postMessage({ type: 'rate', config: view.panel.current });
  }
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
   * Hand every leading frame both views have decoded to `onFrame`, in capture
   * order, with the first view's source pixels (which the decode worker
   * returns) as the original; none for frame 0 held at start-up.
   */
  release() {
    for (const [id, frame] of this.inFlight) {
      if (!frame.results.every(Boolean)) return;
      this.inFlight.delete(id);
      const { source } = frame.results[0], { width, height } = this.views[0].panel.current;
      this.onFrame(frame.results, source ? new ImageData(new Uint8ClampedArray(source), width, height) : null);
      if (this.closed) return;
    }
  }
  fail(error) {
    if (this.closed) return;
    if (this.startFailed) this.startFailed(error);
    else this.onError(error);
  }
}
