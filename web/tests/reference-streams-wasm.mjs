// Run after tools/web/build.sh: every pre-encoded reference stream
// (streams-ref/, made by tools/web/make-reference-streams.py) decodes in the
// browser's WASM decoders, both module builds, bit-exactly: each frame's I420
// MD5 must equal the reference decoder's (ffmpeg, dav1d or vpxdec), which the
// generator recorded in hashes.json after checking our native decoder agrees.
// The page plays these streams through the same BrowserDecoder.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { ivfPackets } from '../streams.js';

const streams = new URL('../streams-ref/', import.meta.url);
const manifest = JSON.parse(await readFile(new URL('manifest.json', streams)));
const expected = JSON.parse(await readFile(new URL('hashes.json', streams)));
const md5 = bytes => createHash('md5').update(bytes).digest('hex');
const report = [];
for (const build of ['', 'relaxed/']) {
  const pkg = new URL(`../pkg/${build}`, import.meta.url);
  const module = await import(new URL('nc_web.js', pkg));
  await module.default({ module_or_path: await readFile(new URL('nc_web_bg.wasm', pkg)) });
  for (const entry of manifest.streams) {
    const packets = ivfPackets(new Uint8Array(await readFile(new URL(entry.file, streams))));
    const decoder = new module.BrowserDecoder(entry.codec);
    try {
      const hashes = packets.map((packet, frame) => {
        try { decoder.decode(packet); } catch (error) { throw new Error(`${entry.id} frame ${frame}: ${error}`); }
        return md5(decoder.decoded_i420());
      });
      const first = hashes.findIndex((hash, frame) => hash !== expected[entry.id][frame]);
      assert.equal(first, -1, `${build || 'simd128/'} ${entry.id}: frame ${first} differs from the reference decoder`);
      assert.equal(hashes.length, expected[entry.id].length);
    } finally { decoder.free(); }
    report.push(`${build || 'simd128/'}${entry.id}`);
  }
}
console.log(JSON.stringify({ status: 'pass', streams: report.length }));
