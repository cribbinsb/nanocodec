// Capture stage of the comparison pipeline (pipeline.js), off the page's main
// thread: turns each captured picture (a VideoFrame, or an ImageBitmap where
// WebCodecs is missing, of the playing video, the camera or the test scene)
// into the frame each view's encoder takes, and posts it straight to that
// encode worker over a MessageChannel port.
//
// A video or camera VideoFrame in a YUV layout the codecs read (I420, NV12)
// goes as its own decoded planes (`VideoFrame.copyTo` in its own format, once
// for all views): no YUV -> RGB -> YUV round trip, 1.5 instead of 4 bytes per
// pixel, and a copy 4-30x cheaper than RGBA when frames live on the GPU
// (docs/handoff/wasm-perf.md, "Native YUV capture"). Each view's worker
// scales and letterboxes the planes to its size in WASM (nc-web
// `SourceFrame`) and codes them in their own colour (colour.js), which the
// stream signals. Everything else (the test scene, other formats or
// colours, engines without VideoFrame) takes the RGBA path below.
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
// On the RGBA path each view gets the picture letterboxed into its own size
// (aspect kept, black bars), as the page did before; views of one size share a
// single conversion. Only the views that code the frame's tick get it
// (`coding`, timeline.js). A frame may come with several pictures, one per
// source (each view codes its own rendition of a clip, exact-source.js), each
// converted for the views it is for.
import { RGBA_COLOUR, colourOf } from './colour.js';

/** VideoFrame formats whose planes nc-web `SourceFrame.from_planes` reads. */
const PLANE_FORMATS = new Set(['I420', 'I420A', 'NV12']);

let views = [];
// One reused 2D canvas per frame size.
const contexts = new Map();
// Whether VideoFrame.copyTo converts to RGBA here (older engines reject the format).
let copyToRgba = true;
// Whether VideoFrame.copyTo gives the planes (cleared if it ever fails, e.g. a
// GPU frame that cannot be read back): the session then stays on RGBA.
let copyPlanes = true;
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
/**
 * The picture's own YUV planes, `{ kind: 'planes', data, format, colour,
 * width, height, layout }` (`layout`: each plane's offset and stride), or null
 * when it must take the RGBA path.
 */
async function planes(picture) {
  if (!copyPlanes || typeof VideoFrame !== 'function' || !(picture instanceof VideoFrame) || !PLANE_FORMATS.has(picture.format)) return null;
  const colour = colourOf(picture);
  if (!colour) return null;
  try {
    const data = new ArrayBuffer(picture.allocationSize());
    const layout = await picture.copyTo(data);
    const { width, height } = picture.visibleRect;
    return { kind: 'planes', data, format: picture.format, colour, width, height, layout: layout.flatMap(p => [p.offset, p.stride]) };
  } catch {
    copyPlanes = false;
    return null;
  }
}
/** The frame for each coding view (null for the others); a frame shared by views is copied, as each is transferred. */
async function frames(picture, sourceWidth, sourceHeight, coding) {
  const yuv = await planes(picture);
  if (yuv) {
    let first = true;
    return coding.map(c => {
      if (!c) return null;
      const frame = first ? yuv : { ...yuv, data: yuv.data.slice(0) };
      first = false;
      return frame;
    });
  }
  const bySize = new Map();
  const result = [];
  for (const [i, { width, height }] of views.entries()) {
    if (!coding[i]) { result.push(null); continue; }
    const key = `${width}x${height}`;
    if (!bySize.has(key)) bySize.set(key, await convert(picture, sourceWidth, sourceHeight, width, height));
    const data = result.some(f => f?.data === bySize.get(key)) ? bySize.get(key).slice(0) : bySize.get(key);
    result.push({ kind: 'rgba', data, colour: RGBA_COLOUR, width, height });
  }
  return result;
}
/** Convert each source's picture for the coding views it is for, and post each view its frame. */
async function capture({ id, timestamp, sources, displays, coding }) {
  try {
    for (const { picture, sourceWidth, sourceHeight, views: forViews } of sources) {
      const converted = await frames(picture, sourceWidth, sourceHeight, coding.map((c, i) => c && forViews.includes(i)));
      views.forEach(({ port }, i) => {
        const frame = converted[i];
        if (frame) port.postMessage({ type: 'frame', id, timestamp, frame, display: displays[i] }, [frame.data]);
      });
    }
  } catch (error) {
    self.postMessage({ type: 'error', message: error instanceof Error ? error.message : String(error) });
  } finally {
    for (const { picture } of sources) picture.close();
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
