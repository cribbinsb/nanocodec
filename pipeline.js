// The page side of the comparison pipeline: per view, an encode worker and a
// decode worker joined by a MessageChannel (encode-worker.js,
// decode-worker.js), and a short queue of frames in flight.
//
// Several frames may be in flight at once (MAX_FRAMES_IN_FLIGHT), so each
// stage works on its own frame: encoders on frame N+1 while decoders finish
// frame N. Every frame goes to both views; results are matched by frame id and
// released in capture order only once both views have decoded that frame, so
// the two views always show the same source frame. The page captures no new
// frame while the queue is full (`full`), which bounds the added latency.

/** Frames captured but not yet decoded by both views: enough to overlap encode and decode with slack for the page. */
export const MAX_FRAMES_IN_FLIGHT = 3;
// Each view's encoder may also hold one source frame back at start-up (encode-worker.js).
const MAX_SOURCE_BUFFERS = MAX_FRAMES_IN_FLIGHT + 1;
const START_TIMEOUT_MS = 30000;

/** One view's pair of workers. */
class ViewWorkers {
  constructor(panel, onMessage, onFailure) {
    this.panel = panel;
    this.sourceBuffers = [];
    this.encoder = new Worker(new URL('./encode-worker.js', import.meta.url), { type: 'module' });
    this.decoder = new Worker(new URL('./decode-worker.js', import.meta.url), { type: 'module' });
    const failed = event => onFailure(new Error(event.message || 'Could not load the WASM codec. Build the demo with tools/web/build.sh, then serve the web directory over HTTP.'));
    this.ready = Promise.all([this.encoder, this.decoder].map(worker => new Promise(resolve => {
      worker.onmessage = ({ data }) => { if (data.type === 'ready') resolve(); else onMessage(this, data); };
      worker.onerror = failed;
    })));
  }
  /** Start both workers, joined by a MessageChannel; resolves once both are ready. */
  init() {
    const channel = new MessageChannel();
    this.decoder.postMessage({ type: 'init', codec: this.panel.current.codec, port: channel.port1 }, [channel.port1]);
    this.encoder.postMessage({ type: 'init', config: this.panel.current, port: channel.port2 }, [channel.port2]);
    return this.ready;
  }
  /** Send a copy of `rgba` (in a buffer this view returned earlier, if one is free). */
  encode(id, timestamp, rgba) {
    let buffer = this.sourceBuffers.pop();
    if (!buffer || buffer.byteLength !== rgba.length) buffer = new ArrayBuffer(rgba.length);
    new Uint8ClampedArray(buffer).set(rgba);
    // Each frame names the format the view's renderer draws, which changes if a GL context is lost.
    this.encoder.postMessage({ type: 'frame', id, timestamp, rgba: buffer, display: this.panel.renderer.format }, [buffer]);
  }
  /** Keep a source buffer the decode worker handed back. */
  returnSource(buffer) {
    if (buffer && this.sourceBuffers.length < MAX_SOURCE_BUFFERS) this.sourceBuffers.push(buffer);
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
    // Frame id -> { original, results }, in capture order (Map keeps insertion order).
    this.inFlight = new Map();
    this.views = panels.map(panel => new ViewWorkers(panel, (view, data) => this.receive(view, data), error => this.fail(error)));
  }
  /** Start every worker; rejects on the first error or after a timeout. */
  start() {
    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('The WASM worker did not start. Check the build and browser console.')), START_TIMEOUT_MS);
      this.startFailed = error => { clearTimeout(timeout); reject(error); };
      Promise.all(this.views.map(view => view.init())).then(() => {
        clearTimeout(timeout);
        this.startFailed = null;
        resolve();
      });
    });
  }
  /** Whether the queue is full: the page should not capture until a frame completes. */
  get full() { return this.inFlight.size >= MAX_FRAMES_IN_FLIGHT; }
  /** Send one captured frame (ImageData) to both views. */
  send(original, timestamp) {
    const id = ++this.nextId;
    this.inFlight.set(id, { original, results: this.views.map(() => null) });
    for (const view of this.views) view.encode(id, timestamp, original.data);
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
      view.returnSource(data.source);
      view.returnSource(data.previous?.source);
      frame.results[this.views.indexOf(view)] = data;
      this.release();
    }
  }
  /** Hand every leading frame both views have decoded to `onFrame`, in capture order. */
  release() {
    for (const [id, frame] of this.inFlight) {
      if (!frame.results.every(Boolean)) return;
      this.inFlight.delete(id);
      this.onFrame(frame.results, frame.original);
      if (this.closed) return;
    }
  }
  fail(error) {
    if (this.closed) return;
    if (this.startFailed) this.startFailed(error);
    else this.onError(error);
  }
}
