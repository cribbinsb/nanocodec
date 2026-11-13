import test from 'node:test';
import assert from 'node:assert/strict';
import { streamBlob } from '../streams.js';

test('IVF carries actual packet lengths, dimensions and capture timestamps', async () => {
  const frames = [{ data: new Uint8Array([1, 2, 3]), timestamp: 100 }, { data: new Uint8Array([4, 5]), timestamp: 166.666 }];
  const bytes = await streamBlob('av1', 640, 360, frames).arrayBuffer(), view = new DataView(bytes);
  assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'DKIF');
  assert.equal(new TextDecoder().decode(bytes.slice(8, 12)), 'AV01');
  assert.equal(view.getUint16(12, true), 640); assert.equal(view.getUint16(14, true), 360);
  assert.equal(view.getUint32(16, true), 1_000_000); assert.equal(view.getUint32(24, true), 2);
  assert.equal(view.getUint32(32, true), 3); assert.equal(view.getBigUint64(36, true), 0n);
  assert.equal(view.getUint32(47, true), 2); assert.equal(view.getBigUint64(51, true), 66666n);
  assert.deepEqual([...new Uint8Array(bytes, 44, 3)], [1, 2, 3]);
});
test('Annex-B packets remain byte-for-byte unchanged', async () => {
  const data = [{ data: new Uint8Array([0, 0, 0, 1, 0x67]) }, { data: new Uint8Array([0, 0, 1, 0x65]) }];
  for (const codec of ['h264', 'hevc', 'vvc']) assert.deepEqual([...new Uint8Array(await streamBlob(codec, 640, 360, data).arrayBuffer())], [0, 0, 0, 1, 0x67, 0, 0, 1, 0x65]);
});

for (const [codec, fourcc] of [['vp9', 'VP90'], ['av2', 'AV02']]) test(`${codec} recordings identify the native bitstream as ${fourcc}`, async () => {
  const bytes = await streamBlob(codec, 72, 40, [{ data: new Uint8Array([0x87]), timestamp: 0 }]).arrayBuffer();
  assert.equal(new TextDecoder().decode(bytes.slice(8, 12)), fourcc);
  assert.equal(new DataView(bytes).getUint32(32, true), 1);
  assert.equal(new Uint8Array(bytes)[44], 0x87);
});
