// Actual WASM lifecycle/isolation check. Run after tools/web/build.sh.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import init, { BrowserEncoder, BrowserDecoder } from '../pkg/nc_web.js';
const binary = await readFile(new URL('../pkg/nc_web_bg.wasm', import.meta.url));
const wasm = await init({ module_or_path: binary });
const count = 1200;
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
function open(codec, id, width = 72) {
  return { codec, id, width, encoder: new BrowserEncoder(codec, width, 64, 160, 30, 900, 'default'), decoder: new BrowserDecoder(codec) };
}
function close(s) { s.encoder.free(); s.decoder.free(); }
function step(s, frame) {
  if (frame === 1000) { close(s); Object.assign(s, open(s.codec, s.id, 64)); }
  if (frame === 10 || frame === 300) s.encoder.set_rate_control(frame === 10 ? 100 : 150, 250, 1000);
  if (frame === 600) s.encoder.disable_rate_control();
  if (frame === 30) s.encoder.request_keyframe();
  if (frame === 450) assert.throws(() => s.decoder.decode(new Uint8Array([0])));
  const input = new Uint8Array(s.width * 64 * 4);
  for (let y = 0; y < 64; y++) for (let x = 0; x < s.width; x++) {
    const n = 4 * (y * s.width + x), shift = Math.floor(frame / 4);
    input.set([(x * 3 + shift + s.id * 41) & 255, (y * 3 + s.id * 29) & 255, (x + y + shift) & 255, 255], n);
  }
  const packet = s.encoder.encode(input);
  assert.equal(s.encoder.key, [0, 30, 900, 1000].includes(frame), `${s.codec} key at ${frame}`);
  s.decoder.decode(packet);
  assert.equal(s.decoder.width, s.width);
  assert.equal(s.decoder.height, 64);
  assert.ok(Number.isFinite(s.encoder.quality(s.decoder)));
  return [hash(packet), hash(s.decoder.decoded_i420()), s.encoder.quantizer];
}
const streams = ['vp9', 'av2'].flatMap(codec => [open(codec, 0), open(codec, 1)]);
const expected = streams.map(() => []), memory = [];
const started = performance.now();
try {
  for (let frame = 0; frame < count; frame++) {
    streams.forEach((s, i) => expected[i].push(step(s, frame)));
    if (frame % 100 === 99) memory.push({ frames: frame + 1, bytes: wasm.memory.buffer.byteLength });
  }
} finally { streams.forEach(close); }
// Replay each independently, proving interleaving did not contaminate state.
for (let i = 0; i < streams.length; i++) {
  const s = open(streams[i].codec, streams[i].id);
  try { for (let frame = 0; frame < count; frame++) assert.deepEqual(step(s, frame), expected[i][frame]); }
  finally { close(s); }
}
const steady = memory.filter(s => s.frames >= 400).map(s => s.bytes);
assert.equal(Math.max(...steady), Math.min(...steady), 'WASM linear memory plateaus after warmup, including recreation');
assert.equal(wasm.memory.buffer.byteLength, steady[0], 'isolated replays reuse freed allocations');
console.log(JSON.stringify({ status: 'pass', wasm_sha256: hash(binary), frames_per_instance: count,
  instances: 4, total_encode_decode_pairs: count * 8, memory, seconds: (performance.now() - started) / 1000 }));
