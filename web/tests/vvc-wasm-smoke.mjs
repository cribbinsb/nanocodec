// Run after tools/web/build.sh: exercise the actual browser WASM module in Node.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import init, { BrowserEncoder, BrowserDecoder, SourceFrame } from '../pkg/nc_web.js';
import { rgbaEncoder } from './rgba-source.mjs';

const wasm = await init({ module_or_path: await readFile(new URL('../pkg/nc_web_bg.wasm', import.meta.url)) });
const encodeRgba = rgbaEncoder(SourceFrame, wasm.memory);

// H.266 §7.4.2.2 NAL unit types of our streams.
const TRAIL_NUT = 0, IDR_N_LP = 8, SPS_NUT = 15, PPS_NUT = 16;
// VVC's QP range (no lossless coding in our subset, so quality is a threshold).
const MAX_QP = 63;
const MIN_PSNR_DB = 30;

/** NAL unit types of one Annex-B access unit. */
function nalTypes(packet) {
  const types = [];
  for (let i = 0; i + 4 < packet.length; i++) {
    if (packet[i] === 0 && packet[i + 1] === 0 && packet[i + 2] === 1) {
      types.push(packet[i + 4] >> 3);
      i += 4;
    }
  }
  return types;
}

function pixels(width, height, frame) {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    // Moving coloured texture exercises all three planes and inter prediction.
    rgba.set([(x * 7 + frame * 13) & 255, (y * 11 + frame * 3) & 255,
      (x * 3 + y * 5 + frame * 17) & 255, 255], (y * width + x) * 4);
  }
  return rgba;
}

/** Encode and decode `frames` frames; returns each frame's PSNR. */
function roundTrip(encoder, decoder, width, height, frames) {
  const psnr = [];
  for (let frame = 0; frame < frames; frame++) {
    const rgba = pixels(width, height, frame);
    const { packet, packed } = encodeRgba(encoder, width, height, rgba);
    decoder.decode(packet);
    assert.ok(decoder.rgba_frame('bt601') + rgba.length <= wasm.memory.buffer.byteLength, `${width}x${height} RGBA frame`);
    assert.equal(decoder.width, width);
    assert.equal(decoder.height, height);
    assert.equal(decoder.decoded_i420().length, width * height * 3 / 2);
    // The demo's decode worker scores against the packed source (decode-worker.js).
    psnr.push(decoder.psnr(packed));
  }
  return psnr;
}

// Any even size is coded (padded to 8, cropped by the conformance window),
// in both presets, across several CTUs.
for (const [width, height] of [[320, 180], [66, 64], [64, 66], [136, 72]]) {
  for (const preset of ['default', 'fast']) {
    const encoder = new BrowserEncoder('vvc', width, height, 22, 30, 900, preset);
    const decoder = new BrowserDecoder('vvc');
    try {
      for (const psnr of roundTrip(encoder, decoder, width, height, 3)) {
        assert.ok(psnr > MIN_PSNR_DB, `${width}x${height} ${preset} PSNR ${psnr}`);
      }
    } finally { encoder.free(); decoder.free(); }
  }
}
for (const [width, height] of [[65, 64], [64, 65]]) {
  assert.throws(() => new BrowserEncoder('vvc', width, height, 22, 30, 900, 'default'), /multiples of 2/);
}
assert.throws(() => new BrowserEncoder('vvc', 64, 64, MAX_QP + 1, 30, 900, 'default'), /quantizer|quantiser/i);
{
  const fresh = new BrowserDecoder('vvc');
  assert.throws(() => fresh.psnr(new Uint8Array(64 * 64 * 3 / 2)), /No decoded frame/);
  fresh.free();
}

const encoder = new BrowserEncoder('vvc', 128, 96, 27, 30, 900, 'default');
const decoder = new BrowserDecoder('vvc');
try {
  {
    const small = new SourceFrame(64, 64);
    try {
      assert.throws(() => small.from_rgba(new Uint8Array(3)), /RGBA frame length/);
      assert.throws(() => encoder.encode(small), /size does not match/);
    } finally { small.free(); }
  }
  for (let frame = 0; frame < 8; frame++) {
    if (frame === 3) { encoder.set_rate_control(100, 200, 1000); encoder.request_keyframe(); }
    if (frame === 5) encoder.disable_rate_control();
    if (frame === 6) encoder.request_keyframe();
    if (frame === 2) assert.throws(() => decoder.decode(new Uint8Array([0])));
    const { packet, packed } = encodeRgba(encoder, 128, 96, pixels(128, 96, frame));
    const key = frame === 0 || frame === 3 || frame === 6;
    assert.equal(encoder.key, key);
    const types = nalTypes(packet);
    assert.ok(types.includes(key ? IDR_N_LP : TRAIL_NUT), `frame ${frame} NAL types ${types}`);
    if (key) assert.ok(types.includes(SPS_NUT) && types.includes(PPS_NUT), 'key carries parameter sets');
    decoder.decode(packet);
    assert.equal(decoder.width, 128);
    assert.equal(decoder.height, 96);
    const psnr = decoder.psnr(packed);
    if (frame === 3 || frame === 4) {
      assert.ok(encoder.quantizer <= MAX_QP);
      assert.ok(Number.isFinite(encoder.buffer_fullness) && encoder.buffer_fullness >= 0);
      assert.ok(Number.isInteger(encoder.buffer_overflows) && encoder.buffer_overflows >= 0);
      assert.ok(Number.isFinite(psnr));
    } else {
      assert.ok(Math.abs(encoder.quantizer - 27) <= 6, `fixed QP cadence ${encoder.quantizer}`);
      assert.ok(psnr > MIN_PSNR_DB, `frame ${frame} PSNR ${psnr}`);
    }
    if (key) {
      const fresh = new BrowserDecoder('vvc');
      try {
        fresh.decode(packet);
        assert.deepEqual(fresh.decoded_i420(), decoder.decoded_i420(), 'forced key starts a fresh decoder');
      } finally { fresh.free(); }
    }
  }
  // A P picture without its reference fails on a fresh decoder.
  const orphan = new BrowserDecoder('vvc');
  try {
    assert.throws(() => orphan.decode(encodeRgba(encoder, 128, 96, pixels(128, 96, 8)).packet));
    assert.equal(orphan.width, 0, 'nothing shown after a failed decode');
  } finally { orphan.free(); }
} finally { encoder.free(); decoder.free(); }

console.log('PASS VVC WASM: 8 key/P frames, both presets, even geometry, rate control, forced keys and error recovery');
