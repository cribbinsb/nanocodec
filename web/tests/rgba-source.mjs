// Node tests code synthetic RGBA frames the way the demo codes RGBA input
// (capture-worker.js's RGBA path): through an nc-web SourceFrame, whose packed
// I420 is also what the decode worker scores PSNR against.

/**
 * A function `(encoder, width, height, rgba) -> { packet, packed }` for the
 * module's `SourceFrame` class and WASM `memory`: `packet` is the encoded
 * frame, `packed` the source as packed I420.
 */
export function rgbaEncoder(SourceFrame, memory) {
  return (encoder, width, height, rgba) => {
    const source = new SourceFrame(width, height);
    try {
      source.from_rgba(rgba);
      const packet = encoder.encode(source);
      const offset = source.packed(); // may grow WASM memory: before reading the buffer
      const packed = new Uint8Array(memory.buffer, offset, source.packed_length).slice();
      return { packet, packed };
    } finally {
      source.free();
    }
  };
}
