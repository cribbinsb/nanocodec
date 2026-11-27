// The page side of the multi-engine conformance check
// (tools/web/browser_engines.py): it reports which module the demo would load
// here (web/wasm.js's feature detection) and decodes streams in a pool of
// module workers (engine-conformance-worker.mjs), one stream per worker at a
// time, so a check uses several cores as the node runner does
// (conformance-runner.mjs). Each stream's result is the MD5 of the shown
// pictures, compared by the driver with the manifest's.
import { MODULE_URL, RELAXED_SIMD, SIMD128 } from '../../web/wasm.js';

const idle = [];
const waiting = [];

/** A free worker, once there is one. */
function worker() {
  return idle.length ? Promise.resolve(idle.pop()) : new Promise(resolve => waiting.push(resolve));
}

function release(w) {
  if (waiting.length) waiting.shift()(w);
  else idle.push(w);
}

window.conformance = {
  detected: { relaxed: RELAXED_SIMD, simd128: SIMD128, module: new URL(MODULE_URL).pathname },
  /** Starts COUNT workers. */
  start(count) {
    for (let i = 0; i < count; i++) idle.push(new Worker(new URL('engine-conformance-worker.mjs', import.meta.url), { type: 'module' }));
  },
  /** Decodes one stream: { pkg, codec, url, deep } -> { pictures, md5 } or { pictures, error }. */
  async decode(job) {
    const w = await worker();
    try {
      return await new Promise(resolve => {
        w.onmessage = e => resolve(e.data);
        w.onerror = e => resolve({ pictures: 0, error: `worker: ${e.message}` });
        w.postMessage(job);
      });
    } finally {
      release(w);
    }
  },
};
window.dispatchEvent(new Event('conformance-ready'));
