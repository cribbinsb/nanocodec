// Run after tools/web/build.sh: exercise the actual browser WASM module in Node.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import init, { BrowserEncoder, BrowserDecoder, SourceFrame } from '../pkg/nc_web.js';
import { rgbaEncoder } from './rgba-source.mjs';

const wasm = await init({ module_or_path: await readFile(new URL('../pkg/nc_web_bg.wasm', import.meta.url)) });
const encodeRgba = rgbaEncoder(SourceFrame, wasm.memory);

function obuTypes(packet) {
  const types = [];
  for (let offset = 0; offset < packet.length;) {
    let size = 0, shift = 0, byte;
    do {
      assert.ok(offset < packet.length && shift <= 49, 'bounded OBU length');
      byte = packet[offset++];
      size += (byte & 127) * 2 ** shift;
      shift += 7;
    } while (byte & 128);
    assert.ok(size > 0 && offset + size <= packet.length, 'complete OBU payload');
    types.push((packet[offset] >> 2) & 31);
    offset += size;
  }
  return types;
}

function pixels(width, height, frame) {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const index = (y * width + x) * 4;
    // Colored texture exercises all three planes, including lossy reference drift.
    rgba.set([(x * 7 + frame * 13) & 255, (y * 11 + frame * 3) & 255,
      (x * 3 + y * 5 + frame * 17) & 255, 255], index);
  }
  return rgba;
}

// Any even size is coded (padded to 8, cropped by the conformance window);
// odd sizes are rejected.
for (const [width, height] of [[320, 180], [66, 64], [64, 66]]) {
  const evenEncoder = new BrowserEncoder('av2', width, height, 0, 30, 900, 'default');
  const evenDecoder = new BrowserDecoder('av2');
  try {
    const rgba = pixels(width, height, 0);
    const { packet, packed } = encodeRgba(evenEncoder, width, height, rgba);
    evenDecoder.decode(packet);
    assert.ok(evenDecoder.rgba_frame('bt601') + rgba.length <= wasm.memory.buffer.byteLength, `${width}x${height} RGBA frame`);
    assert.equal(evenDecoder.width, width);
    assert.equal(evenDecoder.height, height);
    assert.equal(evenDecoder.psnr(packed), 100, `${width}x${height} lossless round trip`);
  } finally { evenEncoder.free(); evenDecoder.free(); }
}
for (const [width, height] of [[65, 64], [64, 65]]) {
  assert.throws(() => new BrowserEncoder('av2', width, height, 0, 30, 900, 'default'), /multiples of 2/);
}
assert.throws(() => new BrowserEncoder('av2', 64, 64, 256, 30, 900, 'default'), /quantizer|quantiser/i);

const encoder = new BrowserEncoder('av2', 64, 64, 0, 30, 900, 'default');
const decoder = new BrowserDecoder('av2');
try {
  {
    const small = new SourceFrame(72, 64);
    try { assert.throws(() => encoder.encode(small), /size does not match/); } finally { small.free(); }
  }
  for (let frame = 0; frame < 7; frame++) {
    if (frame === 3) { encoder.set_rate_control(100, 200, 1000); encoder.request_keyframe(); }
    if (frame === 5) encoder.disable_rate_control();
    if (frame === 6) encoder.request_keyframe();
    if (frame === 2) assert.throws(() => decoder.decode(new Uint8Array([0])));
    const rgba = pixels(64, 64, frame);
    const { packet, packed } = encodeRgba(encoder, 64, 64, rgba);
    const key = frame === 0 || frame === 3 || frame === 6;
    assert.equal(encoder.key, key);
    assert.ok(obuTypes(packet).includes(key ? 4 : 7), 'actual key/forward OBU type');
    decoder.decode(packet);
    assert.ok(decoder.rgba_frame('bt601') + rgba.length <= wasm.memory.buffer.byteLength, 'RGBA frame in WASM memory');
    assert.equal(decoder.width, 64);
    assert.equal(decoder.height, 64);
    assert.equal(decoder.decoded_i420().length, 64 * 64 * 3 / 2);
    assert.ok(Number.isFinite(decoder.psnr(packed)));
    if (frame === 3 || frame === 4) {
      assert.ok(encoder.quantizer >= 1 && encoder.quantizer <= 255);
      assert.ok(Number.isFinite(encoder.buffer_fullness) && encoder.buffer_fullness >= 0);
      assert.ok(Number.isInteger(encoder.buffer_overflows) && encoder.buffer_overflows >= 0);
    } else {
      assert.equal(encoder.quantizer, 0, 'rate-control disable restores fixed quantizer');
      assert.equal(decoder.psnr(packed), 100, 'lossless reconstructed I420 equals source');
    }
    if (key) {
      const fresh = new BrowserDecoder('av2');
      try {
        fresh.decode(packet);
        assert.deepEqual(fresh.decoded_i420(), decoder.decoded_i420(), 'forced key starts a fresh decoder');
      } finally { fresh.free(); }
    }
  }
} finally { encoder.free(); decoder.free(); }

// Non-square, partial-superblock geometry is supported when divisible by eight.
const partialEncoder = new BrowserEncoder('av2', 72, 64, 0, 30, 900, 'default');
const partialDecoder = new BrowserDecoder('av2');
try {
  for (let frame = 0; frame < 2; frame++) {
    const { packet, packed } = encodeRgba(partialEncoder, 72, 64, pixels(72, 64, frame));
    partialDecoder.decode(packet);
    assert.equal(partialEncoder.key, frame === 0);
    assert.equal(partialDecoder.width, 72);
    assert.equal(partialDecoder.height, 64);
    assert.equal(partialDecoder.psnr(packed), 100);
  }
} finally { partialEncoder.free(); partialDecoder.free(); }

console.log('PASS AV2 WASM: 9 key/forward frames, lossless pixels, rate control, forced keys, geometry and error recovery');
