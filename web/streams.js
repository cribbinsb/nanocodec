// Containers, kept separate from the codecs: AV1/VP9/AV2 downloads use IVF
// timestamps; H.264/HEVC/VVC are Annex-B elementary streams (codecs.js). IVF is
// also read: the decoder warm-up streams and the pre-encoded reference streams.
import { CODECS } from './codecs.js';
export const MAX_RECORDING_BYTES = 16 * 1024 * 1024;
const IVF_HEADER_BYTES = 32;
const IVF_FRAME_HEADER_BYTES = 12;

/** The packets of an IVF file, as views into `bytes`; throws if a frame runs past the end. */
export function ivfPackets(bytes) {
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const packets = [];
  for (let at = IVF_HEADER_BYTES; at + IVF_FRAME_HEADER_BYTES <= bytes.length;) {
    const size = view.getUint32(at, true);
    at += IVF_FRAME_HEADER_BYTES;
    if (at + size > bytes.length) throw new Error('Truncated IVF stream');
    packets.push(bytes.subarray(at, at + size));
    at += size;
  }
  return packets;
}
export function streamBlob(codec, width, height, frames) {
  const fourcc = CODECS[codec].ivf;
  if (!fourcc) return new Blob(frames.map(f => f.data), { type: 'application/octet-stream' });
  const header = new ArrayBuffer(32), view = new DataView(header);
  new Uint8Array(header).set([68, 75, 73, 70], 0); // DKIF
  view.setUint16(6, 32, true);
  new TextEncoder().encodeInto(fourcc, new Uint8Array(header, 8, 4));
  view.setUint16(12, width, true); view.setUint16(14, height, true);
  view.setUint32(16, 1_000_000, true); view.setUint32(20, 1, true);
  view.setUint32(24, frames.length, true);
  const parts = [header];
  const origin = frames[0]?.timestamp ?? 0;
  for (const frame of frames) {
    const h = new DataView(new ArrayBuffer(12));
    h.setUint32(0, frame.data.byteLength, true);
    h.setBigUint64(4, BigInt(Math.max(0, Math.round((frame.timestamp - origin) * 1000))), true);
    parts.push(h.buffer, frame.data);
  }
  return new Blob(parts, { type: 'video/x-ivf' });
}
export function download(blob, filename) {
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob); link.href = url; link.download = filename;
  document.body.append(link); link.click(); link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
