// Decoder side of the browser display benchmark (display-bench.py): decodes one
// packet per request and hands the frame over as the demo worker does (web/decode-worker.js):
// packed I420 for a WebGL view or RGBA converted in WASM for the CPU path, copied
// out of WASM memory into a reused buffer and transferred to the page.
import init, { BrowserDecoder } from '../../web/pkg/relaxed/nc_web.js';
import { streamPackets } from './packets.mjs';

let wasm, decoder, packets;
const spare = [];

function copyOut(format) {
  const { width, height } = decoder;
  const length = format === 'i420' ? width * height + 2 * Math.ceil(width / 2) * Math.ceil(height / 2) : width * height * 4;
  const offset = format === 'i420' ? decoder.i420_frame() : decoder.rgba_frame();
  let pixels = spare.pop();
  if (!pixels || pixels.length !== length) pixels = new Uint8Array(length);
  pixels.set(new Uint8Array(wasm.memory.buffer, offset, length));
  return pixels;
}

self.onmessage = async ({ data }) => {
  if (data.type === 'open') {
    wasm ??= await init();
    packets = streamPackets(new Uint8Array(await (await fetch(data.url)).arrayBuffer()), data.codec);
    decoder?.free();
    decoder = new BrowserDecoder(data.codec);
    self.postMessage({ type: 'opened', frames: packets.length });
  } else if (data.type === 'recycle') {
    if (spare.length < 2) spare.push(new Uint8Array(data.buffer));
  } else if (data.type === 'frame') {
    const start = performance.now();
    decoder.decode(packets[data.index]);
    const decoded = performance.now();
    const pixels = copyOut(data.format);
    const handedOff = performance.now();
    // Untimed: the CPU conversion of the same frame, for the pixel comparison.
    const reference = data.check ? copyOut('rgba').slice() : null;
    self.postMessage({ type: 'frame', format: data.format, pixels, reference, width: decoder.width, height: decoder.height,
      decodeMs: decoded - start, handoffMs: handedOff - decoded }, [pixels.buffer]);
  }
};
