// Run after tools/web/build.sh: the page's JS display conversion (colour.js,
// used for the original frame and mirrored by the WebGL shader's weights)
// against nc-dsp's in WASM (BrowserDecoder.rgba_frame), for every colour, on
// a lossless VP9 frame of random samples (every Y, Cb, Cr combination range).
// Also: SourceFrame passes planes of the view size through untouched, and the
// encoder takes only frames of its own colour.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import init, { BrowserEncoder, BrowserDecoder, SourceFrame } from '../pkg/nc_web.js';
import { DISPLAY_WEIGHTS, i420ToRgba } from '../colour.js';

const wasm = await init({ module_or_path: await readFile(new URL('../pkg/nc_web_bg.wasm', import.meta.url)) });
const width = 128, height = 96, chroma = (width / 2) * (height / 2);
const planes = new Uint8Array(width * height + 2 * chroma);
let seed = 12345;
for (let i = 0; i < planes.length; i++) { seed = (seed * 1103515245 + 12345) >>> 0; planes[i] = seed >>> 24; }
const layout = Uint32Array.from([0, width, width * height, width / 2, width * height + chroma, width / 2]);
for (const colour of Object.keys(DISPLAY_WEIGHTS)) {
  const source = new SourceFrame(width, height);
  const encoder = new BrowserEncoder('vp9', width, height, 0, 30, 900, 'default', undefined, false, colour);
  const decoder = new BrowserDecoder('vp9');
  try {
    source.from_planes(planes, 'I420', colour, width, height, layout);
    assert.equal(source.colour, colour);
    const packedAt = source.packed(); // may grow WASM memory: before reading the buffer
    assert.deepEqual(new Uint8Array(wasm.memory.buffer, packedAt, source.packed_length).slice(), planes, 'planes pass through');
    decoder.decode(encoder.encode(source));
    assert.deepEqual(decoder.decoded_i420(), planes, 'lossless');
    const offset = decoder.rgba_frame(colour);
    const wasmRgba = new Uint8Array(wasm.memory.buffer, offset, width * height * 4).slice();
    assert.deepEqual(new Uint8Array(i420ToRgba(planes, width, height, colour).buffer), wasmRgba, `${colour}: JS and WASM display conversions`);
    const other = colour === 'bt601' ? 'bt709' : 'bt601';
    const wrong = new BrowserEncoder('vp9', width, height, 0, 30, 900, 'default', undefined, false, other);
    try { assert.throws(() => wrong.encode(source), /colour does not match/); } finally { wrong.free(); }
  } finally { source.free(); encoder.free(); decoder.free(); }
}
console.log(`PASS colour: SourceFrame planes, encoder colour check, JS = WASM display conversion for ${Object.keys(DISPLAY_WEIGHTS).join(', ')}`);
