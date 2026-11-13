// Re-frames a stored stream as IVF (one packet per picture, the packets the
// demo worker passes to BrowserDecoder.decode), so web/warmup.js needs only an
// IVF reader for every codec. Run by tools/web/make-warmup.sh.
//
//   node make-warmup.mjs CODEC IN_STREAM OUT_IVF WIDTH HEIGHT
import { readFile, writeFile } from 'node:fs/promises';
import { streamPackets } from './packets.mjs';

const [codec, input, output, width, height] = process.argv.slice(2);
const packets = streamPackets(new Uint8Array(await readFile(input)), codec);
const header = new DataView(new ArrayBuffer(32));
const fourcc = codec.toUpperCase().padEnd(4, '0').slice(0, 4);
[...'DKIF'].forEach((c, i) => header.setUint8(i, c.charCodeAt(0)));
header.setUint16(6, 32, true);
[...fourcc].forEach((c, i) => header.setUint8(8 + i, c.charCodeAt(0)));
header.setUint16(12, +width, true);
header.setUint16(14, +height, true);
header.setUint32(16, 15, true);
header.setUint32(20, 1, true);
header.setUint32(24, packets.length, true);
const parts = [new Uint8Array(header.buffer)];
packets.forEach((packet, index) => {
  const frame = new DataView(new ArrayBuffer(12));
  frame.setUint32(0, packet.length, true);
  frame.setUint32(4, index, true);
  parts.push(new Uint8Array(frame.buffer), packet);
});
await writeFile(output, Buffer.concat(parts));
console.log(`${output}: ${packets.length} packets`);
