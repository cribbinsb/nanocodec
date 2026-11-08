// Container writing is separate from the codec: AV1/VP9 downloads use IVF timestamps.
export const MAX_RECORDING_BYTES = 16 * 1024 * 1024;
export function streamBlob(codec, width, height, frames) {
  if (!['av1', 'vp9', 'av2'].includes(codec)) return new Blob(frames.map(f => f.data), { type: 'application/octet-stream' });
  const header = new ArrayBuffer(32), view = new DataView(header);
  new Uint8Array(header).set([68, 75, 73, 70], 0); // DKIF
  view.setUint16(6, 32, true);
  new Uint8Array(header).set(codec === 'vp9' ? [86, 80, 57, 48] : [65, 86, 48, codec === 'av2' ? 50 : 49], 8); // VP90 / AV01 / AV02
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
