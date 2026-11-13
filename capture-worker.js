// Capture stage of the comparison pipeline (pipeline.js), off the page's main
// thread: turns each captured picture (a VideoFrame, or an ImageBitmap where
// WebCodecs is missing, of the playing video, the camera or the test scene)
// into the RGBA frame each view's encoder takes, and posts it straight to that
// encode worker over a MessageChannel port.
//
// Reading a video frame back as RGBA is the costliest step outside the codecs:
// the page's drawImage + getImageData took 15-90 ms per 720p frame in headless
// Chromium under load, during which the page could neither capture nor
// receive and draw results (docs/handoff/wasm-perf.md, "Demo end-to-end").
// Here it blocks only this worker. Where the view has the picture's own size,
// VideoFrame.copyTo converts it without a canvas (asynchronously; 3-4x faster
// than the canvas when frames are in CPU memory). Both paths give the page's
// former pixels exactly (Chromium 148, every sample clip; the test
// web/tests/capture-identity.py).
//
// Each view gets the picture letterboxed into its own size (aspect kept, black
// bars), as the page did before; views of one size share a single conversion.

let views = [];
// One reused 2D canvas per frame size.
const contexts = new Map();
// Whether VideoFrame.copyTo converts to RGBA here (older engines reject the format).
let copyToRgba = true;
// Frames are converted one after another, so they reach the encoders in order.
let previous = Promise.resolve();

function context(width, height) {
  const key = `${width}x${height}`;
  if (!contexts.has(key)) contexts.set(key, new OffscreenCanvas(width, height).getContext('2d', { willReadFrequently: true }));
  return contexts.get(key);
}
/** `picture` (`sourceWidth` x `sourceHeight`) letterboxed into width x height on a canvas, as RGBA bytes. */
function drawn(picture, sourceWidth, sourceHeight, width, height) {
  const ctx = context(width, height);
  const scale = Math.min(width / sourceWidth, height / sourceHeight);
  const w = sourceWidth * scale, h = sourceHeight * scale;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, width, height);
  ctx.drawImage(picture, (width - w) / 2, (height - h) / 2, w, h);
  return ctx.getImageData(0, 0, width, height).data.buffer;
}
/** `picture` as RGBA bytes of width x height. */
async function convert(picture, sourceWidth, sourceHeight, width, height) {
  if (copyToRgba && typeof VideoFrame === 'function' && picture instanceof VideoFrame && width === sourceWidth && height === sourceHeight
    && picture.visibleRect.width === width && picture.visibleRect.height === height) {
    const rgba = new ArrayBuffer(width * height * 4);
    try {
      await picture.copyTo(rgba, { format: 'RGBA' });
      return rgba;
    } catch {
      copyToRgba = false;
    }
  }
  return drawn(picture, sourceWidth, sourceHeight, width, height);
}
/** Convert one picture for every view and post each its frame. */
async function capture({ id, timestamp, picture, sourceWidth, sourceHeight, displays }) {
  try {
    const bySize = new Map();
    const frames = [];
    for (const { width, height } of views) {
      const key = `${width}x${height}`;
      // A second view of the same size takes a copy: each buffer is transferred.
      if (!bySize.has(key)) bySize.set(key, await convert(picture, sourceWidth, sourceHeight, width, height));
      frames.push(frames.some(f => f === bySize.get(key)) ? bySize.get(key).slice(0) : bySize.get(key));
    }
    views.forEach(({ port }, i) => port.postMessage({ type: 'frame', id, timestamp, rgba: frames[i], display: displays[i] }, [frames[i]]));
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  } finally {
    picture.close();
  }
}

self.onmessage = ({ data: message }) => {
  if (message.type === 'init') {
    // [{ port, width, height }] per view, in view order.
    views = message.views;
  } else if (message.type === 'frame') {
    previous = previous.then(() => capture(message));
  }
};
