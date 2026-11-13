// Run after tools/web/build.sh: exercises the actual browser WASM module in Node.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import init, { BrowserEncoder, BrowserDecoder, SourceFrame, check_parameters, codec_options, parameter_registry } from '../pkg/nc_web.js';
import { rgbaEncoder } from './rgba-source.mjs';

const wasm = await init({ module_or_path: await readFile(new URL('../pkg/nc_web_bg.wasm', import.meta.url)) });
const encodeRgba = rgbaEncoder(SourceFrame, wasm.memory);
// Any even size codes (180 is not a multiple of 8); odd sizes do not.
new BrowserEncoder('vp9', 322, 180, 0, 30, 900, 'default').free();
assert.throws(() => new BrowserEncoder('vp9', 321, 180, 0, 30, 900, 'default'), /multiples of/);
// Film grain synthesis is AV1-only; AV1 accepts it.
assert.throws(() => new BrowserEncoder('vp9', 64, 64, 0, 30, 900, 'default', '', true), /only available for AV1/);
new BrowserEncoder('av1', 64, 64, 180, 30, 900, 'default', '', true).free();
// Parameter overrides (the demo's Advanced editor) are checked against the registry.
assert.deepEqual(JSON.parse(check_parameters('vp9', '# ok\nrate.aq_strength=1')), []);
assert.match(JSON.parse(check_parameters('vp9', 'nope.nothing=1'))[0].message, /unknown parameter/);
assert.throws(() => new BrowserEncoder('vp9', 64, 64, 0, 30, 900, 'default', 'layers.count=9'), /parameter line 1/);
assert.ok(JSON.parse(parameter_registry('vp9')).some(p => p.name === 'rate.aq_strength' && p.doc));
assert.equal(JSON.parse(codec_options('vp9')).filmGrain, false);
const encoder = new BrowserEncoder('vp9', 64, 64, 0, 30, 900, 'default');
const decoder = new BrowserDecoder('vp9');
try {
  for (let frame = 0; frame < 4; frame++) {
    if (frame === 2) { encoder.set_rate_control(100, 200, 1000); encoder.request_keyframe(); }
    if (frame === 3) encoder.disable_rate_control();
    const rgba = new Uint8Array(64 * 64 * 4);
    for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
      const i = (y * 64 + x) * 4, gray = (x * 3 + y * 2 + frame * 4) & 255;
      rgba.set([gray, gray, gray, 255], i);
    }
    const { packet, packed } = encodeRgba(encoder, 64, 64, rgba);
    assert.equal(encoder.key, frame === 0 || frame === 2);
    decoder.decode(packet);
    assert.ok(decoder.rgba_frame('bt601') + rgba.length <= wasm.memory.buffer.byteLength, 'RGBA frame in WASM memory');
    assert.equal(decoder.width, 64); assert.equal(decoder.height, 64);
    assert.equal(decoder.decoded_i420().length, 64 * 64 * 3 / 2);
    if (frame !== 2) {
      assert.equal(encoder.quantizer, 0);
      assert.equal(decoder.psnr(packed), 100);
    } else {
      assert.ok(encoder.quantizer >= 1 && encoder.quantizer <= 255);
      assert.ok(Number.isFinite(decoder.psnr(packed)));
    }
  }
  console.log('PASS VP9 WASM: lossless inter, forced keyframe, rate-control enable/disable, 30-second GOP');
} finally { encoder.free(); decoder.free(); }

// Lossless P frames above use TX4; separately exercise mixed-size lossy TX8.
const lossy = new BrowserEncoder('vp9', 64, 64, 160, 30, 900, 'default');
const lossyDecoder = new BrowserDecoder('vp9');
try {
  for (let frame = 0; frame < 6; frame++) {
    if (frame === 3) lossy.request_keyframe();
    if (frame === 2) assert.throws(() => lossyDecoder.decode(new Uint8Array([0])));
    const rgba = new Uint8Array(64 * 64 * 4);
    for (let y = 0; y < 64; y++) for (let x = 0; x < 64; x++) {
      rgba.set([(x * 2 + y + frame) & 255, (x + y * 2 + frame) & 255,
        (x + y + frame * 2) & 255, 255], (y * 64 + x) * 4);
    }
    const { packet, packed } = encodeRgba(lossy, 64, 64, rgba);
    assert.equal(lossy.key, frame === 0 || frame === 3);
    // Key frames report the configured qindex; P frames follow the engine's
    // fixed-QP cadence (rate.cadence_fixed_qp), whose offsets stay well within
    // half an octave of quantizer step (about 32 qindex here).
    if (lossy.key) assert.equal(lossy.quantizer, 160);
    else assert.ok(Math.abs(lossy.quantizer - 160) <= 32, `P-frame qindex ${lossy.quantizer}`);
    lossyDecoder.decode(packet);
    assert.ok(lossyDecoder.rgba_frame('bt601') + rgba.length <= wasm.memory.buffer.byteLength, 'RGBA frame in WASM memory');
    assert.equal(lossyDecoder.decoded_i420().length, 64 * 64 * 3 / 2);
    assert.ok(Number.isFinite(lossyDecoder.psnr(packed)));
  }
  console.log('PASS VP9 WASM: six lossy TX8 frames, chroma, forced reset and error recovery');
} finally { lossy.free(); lossyDecoder.free(); }

// Spatial SVC (L2T1, 2:1 scaling 128x72 -> 64x36 base): full superframe view
// and base-only operating point (set_operating_point(1)) both decode and score PSNR.
const svcEnc = new BrowserEncoder(
  'vp9', 128, 72, 140, 30, 30, 'fast',
  'spatial.layers=2\nspatial.base_scale=0.5\nhold.enable=0', false
);
const svcFullDec = new BrowserDecoder('vp9');
const svcBaseDec = new BrowserDecoder('vp9');
svcBaseDec.set_operating_point(1);
try {
  for (let frame = 0; frame < 4; frame++) {
    const rgba = new Uint8Array(128 * 72 * 4);
    for (let y = 0; y < 72; y++) for (let x = 0; x < 128; x++) {
      rgba.set([(x * 3 + y + frame * 7) & 255, (x + y * 5 + frame * 3) & 255,
        (x * 2 + y * 2 + frame * 11) & 255, 255], (y * 128 + x) * 4);
    }
    const { packet, packed } = encodeRgba(svcEnc, 128, 72, rgba);
    svcFullDec.decode(packet);
    svcBaseDec.decode(packet);
    assert.equal(svcFullDec.width, 128);
    assert.equal(svcFullDec.height, 72);
    assert.equal(svcFullDec.spatial_id, 1);
    assert.equal(svcBaseDec.width, 64);
    assert.equal(svcBaseDec.height, 36);
    assert.equal(svcBaseDec.spatial_id, 0);
    assert.equal(svcBaseDec.stream_width, 128);
    assert.equal(svcBaseDec.stream_height, 72);
    assert.ok(svcFullDec.psnr(packed) > 20);
    assert.ok(svcBaseDec.psnr(packed) > 18);
  }
  console.log('PASS VP9 WASM: spatial SVC full (128x72) and base-only (64x36) views decode and score PSNR');
} finally { svcEnc.free(); svcFullDec.free(); svcBaseDec.free(); }
