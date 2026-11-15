// A captured frame placed at a view's size, exactly as the view's encoder is
// given it (encode-worker.js): nc-web `SourceFrame` from the capture worker's
// planes or RGBA (capture-worker.js), packed I420 without row padding, with
// its colour. Workers without an encoder use it: a pre-encoded view's decoder
// scores its stream against this frame (decode-worker.js), and an "Original"
// view shows it (original-worker.js), so all three see the same pixels.

export class ViewSourceFrame {
  /** `module`: the loaded codec module; `wasm`: its instance exports (for the memory). */
  constructor(module, wasm) {
    this.module = module; this.wasm = wasm;
    this.frame = null; this.size = '';
  }
  /** `{ source, colour }`: the captured `frame` at width x height, `source` a fresh packed I420 ArrayBuffer. */
  place({ frame, width, height }) {
    if (this.size !== `${width}x${height}`) {
      this.frame?.free();
      this.frame = new this.module.SourceFrame(width, height);
      this.size = `${width}x${height}`;
    }
    if (frame.kind === 'planes') {
      this.frame.from_planes(new Uint8Array(frame.data), frame.format, frame.colour, frame.width, frame.height, Uint32Array.from(frame.layout));
    } else {
      this.frame.from_rgba(new Uint8Array(frame.data));
    }
    const offset = this.frame.packed();
    return { source: new Uint8Array(this.wasm.memory.buffer, offset, this.frame.packed_length).slice().buffer, colour: this.frame.colour };
  }
}
