// Start-up warm-up for the decode worker. V8 compiles each WASM function on its
// first call, so a fresh module decodes its first frame 5-10x slower than a
// warm one (72-155 ms at 720p) and its first second 1.7-3.4x slower
// (docs/results/competitor-yardsticks.md §3.5): what a viewer joining a stream
// sees. Before the first real packet, the worker decodes a tiny built-in
// stream of its codec (warmup/<codec>.ivf: eight 256x144 frames, one key and
// seven inter, made by tools/web/make-warmup.sh), converts each frame for both
// display paths and scores it, then discards everything. The functions compiled
// on the way are the ones the real stream needs.
//
// It never holds up a real frame for long: it yields to the event loop before
// every frame and stops as soon as `stop()` reports real data, so a packet that
// arrives mid-warm-up waits for at most one tiny frame (whose compilation the
// real frame would have paid for anyway).

const IVF_HEADER_BYTES = 32;
const IVF_FRAME_HEADER_BYTES = 12;

/** The packets of an IVF file. */
function ivfPackets(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const packets = [];
  for (let at = IVF_HEADER_BYTES; at + IVF_FRAME_HEADER_BYTES <= bytes.length;) {
    const size = view.getUint32(at, true);
    at += IVF_FRAME_HEADER_BYTES;
    packets.push(bytes.subarray(at, at + size));
    at += size;
  }
  return packets;
}

/** Let queued messages (a real packet) run first. */
const yieldToEvents = () => new Promise(resolve => setTimeout(resolve, 0));

/**
 * Decode `codec`'s warm-up stream with a throwaway decoder of `module` unless
 * `stop()` turns true. Resolves to `{ ms, frames, longestFrameMs, stopped }`:
 * the time spent, the tiny frames decoded, the longest synchronous step (a
 * frame, or creating the throwaway decoder: the most a real packet can wait)
 * and whether real data stopped it. A stream that cannot be
 * fetched counts as stopped with no frames: warm-up is only an optimisation.
 */
export async function warmUp(module, codec, stop) {
  const start = performance.now();
  const result = (frames, longestFrameMs, stopped) =>
    ({ ms: performance.now() - start, frames, longestFrameMs, stopped });
  let response;
  try {
    response = await fetch(new URL(`./warmup/${codec}.ivf`, import.meta.url));
  } catch {
    return result(0, 0, true);
  }
  if (!response.ok || stop()) return result(0, 0, true);
  const packets = ivfPackets(new Uint8Array(await response.arrayBuffer()));
  if (stop()) return result(0, 0, true);
  const created = performance.now();
  const decoder = new module.BrowserDecoder(codec);
  let frames = 0, longestFrameMs = performance.now() - created;
  try {
    let source = null;
    for (const packet of packets) {
      await yieldToEvents();
      if (stop()) return result(frames, longestFrameMs, true);
      const frameStart = performance.now();
      decoder.decode(packet);
      decoder.rgba_frame();
      decoder.i420_frame();
      source ??= new Uint8Array(decoder.width * decoder.height * 4);
      decoder.psnr(source);
      longestFrameMs = Math.max(longestFrameMs, performance.now() - frameStart);
      frames++;
    }
  } finally {
    decoder.free();
  }
  return result(frames, longestFrameMs, false);
}
