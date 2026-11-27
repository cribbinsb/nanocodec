// Incremental MD5 (RFC 1321) for the in-browser conformance page
// (engine-conformance.html): the conformance manifests give each stream's
// decoded output as an MD5, and Web Crypto has no MD5. It hashes a whole
// stream's pictures as they are decoded, so no output is kept or posted back.

// Per-round left rotations (RFC 1321 §3.4).
const SHIFTS = [7, 12, 17, 22, 5, 9, 14, 20, 4, 11, 16, 23, 6, 10, 15, 21];
// floor(abs(sin(i + 1)) * 2^32), the sine table of RFC 1321 §3.4.
const SINES = Uint32Array.from({ length: 64 }, (_, i) => Math.floor(Math.abs(Math.sin(i + 1)) * 2 ** 32));
const BLOCK_BYTES = 64;

export class Md5 {
  constructor() {
    this.state = Uint32Array.of(0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476);
    this.pending = new Uint8Array(BLOCK_BYTES);
    this.pendingBytes = 0;
    this.totalBytes = 0;
    this.words = new Uint32Array(16);
  }

  /** Adds bytes (a Uint8Array) to the message. */
  update(bytes) {
    this.totalBytes += bytes.length;
    let at = 0;
    if (this.pendingBytes) {
      const take = Math.min(BLOCK_BYTES - this.pendingBytes, bytes.length);
      this.pending.set(bytes.subarray(0, take), this.pendingBytes);
      this.pendingBytes += take;
      at = take;
      if (this.pendingBytes < BLOCK_BYTES) return this;
      this.block(this.pending, 0);
      this.pendingBytes = 0;
    }
    for (; at + BLOCK_BYTES <= bytes.length; at += BLOCK_BYTES) this.block(bytes, at);
    this.pending.set(bytes.subarray(at));
    this.pendingBytes = bytes.length - at;
    return this;
  }

  /** The digest as lower-case hex; the hash takes no more input. */
  hex() {
    const bits = this.totalBytes * 8;
    const tail = new Uint8Array((this.pendingBytes < 56 ? 64 : 128) - this.pendingBytes);
    tail[0] = 0x80;
    const view = new DataView(tail.buffer);
    view.setUint32(tail.length - 8, bits >>> 0, true);
    view.setUint32(tail.length - 4, Math.floor(bits / 2 ** 32), true);
    this.update(tail);
    return [...new Uint8Array(this.state.buffer)].map(b => b.toString(16).padStart(2, '0')).join('');
  }

  block(bytes, at) {
    const w = this.words;
    for (let i = 0; i < 16; i++) {
      const j = at + i * 4;
      w[i] = bytes[j] | bytes[j + 1] << 8 | bytes[j + 2] << 16 | bytes[j + 3] << 24;
    }
    let [a, b, c, d] = this.state;
    for (let i = 0; i < 64; i++) {
      const round = i >> 4;
      let f, g;
      if (round === 0) { f = (b & c) | (~b & d); g = i; }
      else if (round === 1) { f = (d & b) | (~d & c); g = (5 * i + 1) & 15; }
      else if (round === 2) { f = b ^ c ^ d; g = (3 * i + 5) & 15; }
      else { f = c ^ (b | ~d); g = (7 * i) & 15; }
      const sum = (a + f + SINES[i] + w[g]) | 0;
      const shift = SHIFTS[round * 4 + (i & 3)];
      a = d; d = c; c = b;
      b = (b + ((sum << shift) | (sum >>> (32 - shift)))) | 0;
    }
    this.state[0] += a; this.state[1] += b; this.state[2] += c; this.state[3] += d;
  }
}
