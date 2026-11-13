// Splits a stored stream into the per-frame packets the demo worker passes to
// BrowserDecoder.decode. Shared by the node benchmark (bench-runner.mjs) and the
// browser display benchmark (display-bench.py), so it uses no node modules.

const IVF_HEADER_BYTES = 32;
const IVF_FRAME_HEADER_BYTES = 12;

export function ivfFrames(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const frames = [];
  for (let at = IVF_HEADER_BYTES; at + IVF_FRAME_HEADER_BYTES <= bytes.length;) {
    const size = view.getUint32(at, true);
    at += IVF_FRAME_HEADER_BYTES;
    frames.push(bytes.subarray(at, at + size));
    at += size;
  }
  return frames;
}

// Start offsets of every Annex-B start code (00 00 01, optionally preceded by a zero).
function nalStarts(bytes) {
  const starts = [];
  for (let i = 0; i + 3 <= bytes.length; i++) {
    if (bytes[i] === 0 && bytes[i + 1] === 0 && bytes[i + 2] === 1) {
      starts.push(i > 0 && bytes[i - 1] === 0 ? i - 1 : i);
      i += 2;
    }
  }
  return starts;
}

// Whether a NAL header's first two bytes start a coded slice: H.264 types 1-5,
// HEVC types below 32 (§7.4.2.2), VVC types below 12 (H.266 §7.4.2.2).
function isVcl(codec, first, second) {
  if (codec === 'h264') return (first & 0x1f) >= 1 && (first & 0x1f) <= 5;
  if (codec === 'vvc') return (second >> 3) < 12;
  return ((first >> 1) & 0x3f) < 32;
}

// One packet per picture: parameter sets and SEI attach to the following VCL NAL
// (our streams carry exactly one slice per picture).
export function annexBFrames(bytes, codec) {
  const starts = nalStarts(bytes);
  const frames = [];
  let first = starts[0];
  for (let n = 0; n < starts.length; n++) {
    const end = n + 1 < starts.length ? starts[n + 1] : bytes.length;
    let header = starts[n];
    while (bytes[header] === 0) header++;
    if (isVcl(codec, bytes[header + 1], bytes[header + 2])) { frames.push(bytes.subarray(first, end)); first = end; }
  }
  return frames;
}

/** Packets of an IVF (AV1, VP9, AV2) or Annex-B (H.264, HEVC, VVC) stream. */
export function streamPackets(bytes, codec) {
  return ['h264', 'hevc', 'vvc'].includes(codec) ? annexBFrames(bytes, codec) : ivfFrames(bytes);
}
