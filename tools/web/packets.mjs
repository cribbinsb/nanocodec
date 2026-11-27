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

// The VVC picture header NAL type (H.266 Table 5): it opens a picture whose
// slices then carry sh_picture_header_in_slice_header_flag = 0.
const VVC_PH_NUT = 19;

// Whether a NAL opens a new picture; `nal` indexes its first header byte.
// H.264: a slice with first_mb_in_slice = 0, whose ue(v) is the single bit 1
// (§7.3.3); HEVC: first_slice_segment_in_pic_flag (§7.3.6.1); VVC: a picture
// header NAL, or a slice with sh_picture_header_in_slice_header_flag
// (H.266 §7.3.7.1). Streams of several slices per picture (conformance
// vectors) need this; only the first bit after the NAL header is read, which
// emulation prevention cannot change.
function startsPicture(codec, bytes, nal, vcl) {
  if (codec === 'vvc' && (bytes[nal + 1] >> 3) === VVC_PH_NUT) return true;
  if (!vcl) return false;
  return (bytes[codec === 'h264' ? nal + 1 : nal + 2] & 0x80) !== 0;
}

// One packet per picture (access unit): every slice of a picture, with the
// parameter sets and SEI that precede its first slice. A NAL between two
// pictures' slices goes with the later picture.
export function annexBFrames(bytes, codec) {
  const starts = nalStarts(bytes);
  const frames = [];
  let first = starts[0];
  let sliceEnd = null; // end of the current picture's last slice
  for (let n = 0; n < starts.length; n++) {
    const end = n + 1 < starts.length ? starts[n + 1] : bytes.length;
    let header = starts[n];
    while (bytes[header] === 0) header++;
    const nal = header + 1; // past the start code's final 01
    const vcl = isVcl(codec, bytes[nal], bytes[nal + 1]);
    if (sliceEnd !== null && startsPicture(codec, bytes, nal, vcl)) {
      frames.push(bytes.subarray(first, sliceEnd));
      first = sliceEnd;
      sliceEnd = null;
    }
    if (vcl) sliceEnd = end;
  }
  if (sliceEnd !== null) frames.push(bytes.subarray(first, bytes.length));
  return frames;
}

/** Packets of an IVF (AV1, VP9, AV2) or Annex-B (H.264, HEVC, VVC) stream. */
export function streamPackets(bytes, codec) {
  return ['h264', 'hevc', 'vvc'].includes(codec) ? annexBFrames(bytes, codec) : ivfFrames(bytes);
}
