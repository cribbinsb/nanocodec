// WASM decode/encode benchmark runner for one stream or clip, driven by
// tools/web/bench.py. Loads the built browser module (web/pkg, SIMD128) under
// node, single-threaded, and prints one JSON object on stdout.
//
//   node bench-runner.mjs decode --pkg DIR --codec C --stream FILE [--repeat N] [--hash]
//   node bench-runner.mjs encode --pkg DIR --codec C --yuv FILE --width W --height H
//        --frames N --fps F --kbps R --peak-kbps P --buffer-ms B --key-seconds K [--fast] [--repeat N]
//
// decode: splits IVF (AV1/VP9/AV2) or Annex-B (H.264/HEVC) into the per-frame
// packets the demo worker passes to BrowserDecoder.decode, and times each frame
// as the demo worker pays it: decode, then the display hand-off (I420->RGBA and
// the copy out of WASM memory into a reused buffer). The two parts are also
// reported separately (`decode_ms_per_frame`, `display_ms_per_frame`). Modules
// built before the hand-off API (`rgba_frame`) return RGBA from `decode`; their
// hand-off cannot be separated.
// --hash adds a separate untimed pass hashing every decoded I420 frame, for the
// bit-exactness check against the native decoder.
// encode: times BrowserEncoder.encode per frame as the demo pays it: from the
// clip's I420 planes through a SourceFrame (the demo's video capture path,
// modules with SourceFrame), or, for older modules, from RGBA converted from
// the clip once (untimed) through the encoder's own RGBA->I420;
// --fast selects the Fast preset (the demo's "Encoder preset"). The encoder's
// quantizer argument (under rate control, the seed of the first key frame) is
// the demo's initial one per codec (web/codecs.js), as the native CLI seeds it:
// q-index 180 for AV1, VP9 and AV2, QP 32 otherwise.
// Every pass uses fresh codec instances; the best (minimum-total) pass is reported
// together with the per-pass totals, so noise from a loaded host is visible.
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { streamPackets } from './packets.mjs';
import { CODECS } from '../../web/codecs.js';

function parseArgs(argv) {
  const [mode, ...rest] = argv;
  const args = { mode, repeat: 3, hash: false, fast: false };
  for (let i = 0; i < rest.length; i++) {
    const key = rest[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    if (key === 'hash' || key === 'fast') args[key] = true;
    else args[key] = rest[++i];
  }
  return args;
}

async function loadModule(pkg) {
  const module = await import(pathToFileURL(path.join(pkg, 'nc_web.js')).href);
  const binary = await readFile(path.join(pkg, 'nc_web_bg.wasm'));
  const exports = await module.default({ module_or_path: binary });
  return { module: { ...module, wasmMemory: () => exports.memory }, wasmSha256: createHash('sha256').update(binary).digest('hex') };
}

function summarize(passes, frames) {
  const totals = passes.map(p => p.reduce((a, b) => a + b, 0));
  const best = passes[totals.indexOf(Math.min(...totals))];
  const sorted = [...best].sort((a, b) => a - b);
  const total = Math.min(...totals);
  return {
    frames, pass_totals_ms: totals.map(t => +t.toFixed(2)),
    ms_per_frame: +(total / frames).toFixed(3), fps: +(1000 * frames / total).toFixed(2),
    median_frame_ms: +sorted[Math.floor(sorted.length / 2)].toFixed(3),
    max_frame_ms: +sorted[sorted.length - 1].toFixed(3),
  };
}

async function decode(args) {
  const { module, wasmSha256 } = await loadModule(args.pkg);
  const bytes = new Uint8Array(await readFile(args.stream));
  const packets = streamPackets(bytes, args.codec);
  const result = { mode: 'decode', codec: args.codec, stream: args.stream, wasm_sha256: wasmSha256 };
  if (args.hash) {
    const decoder = new module.BrowserDecoder(args.codec);
    const digest = createHash('sha256');
    for (const packet of packets) { decoder.decode(packet); digest.update(decoder.decoded_i420()); }
    result.i420_sha256 = digest.digest('hex');
    result.width = decoder.width; result.height = decoder.height;
    decoder.free();
  }
  const handoff = 'rgba_frame' in module.BrowserDecoder.prototype;
  const passes = [], decodeOnly = [];
  for (let pass = 0; pass < +args.repeat; pass++) {
    const decoder = new module.BrowserDecoder(args.codec);
    const times = [], decodeTimes = [];
    let canvas = null;
    for (const packet of packets) {
      const start = performance.now();
      decoder.decode(packet);
      const decoded = performance.now();
      if (handoff) {
        const length = decoder.width * decoder.height * 4;
        // The colour argument (newer modules) is ignored by older ones.
        const offset = decoder.rgba_frame('bt601'); // may grow (and detach) WASM memory
        const view = new Uint8Array(module.wasmMemory().buffer, offset, length);
        if (!canvas || canvas.length !== length) canvas = new Uint8Array(length);
        canvas.set(view);
      }
      const end = performance.now();
      times.push(end - start);
      decodeTimes.push(decoded - start);
    }
    decoder.free();
    passes.push(times);
    decodeOnly.push(decodeTimes);
  }
  if (!passes.length) return result; // --repeat 0: hash only
  const best = summarize(passes, packets.length);
  if (handoff) {
    const totals = passes.map(p => p.reduce((a, b) => a + b, 0));
    const pass = totals.indexOf(Math.min(...totals));
    const decodeMs = decodeOnly[pass].reduce((a, b) => a + b, 0) / packets.length;
    result.decode_ms_per_frame = +decodeMs.toFixed(3);
    result.display_ms_per_frame = +(best.ms_per_frame - decodeMs).toFixed(3);
  }
  return { ...result, ...best };
}

function i420ToRgba(frame, width, height) {
  const rgba = new Uint8Array(width * height * 4);
  const chromaWidth = width / 2, uBase = width * height, vBase = uBase + chromaWidth * height / 2;
  const clamp = v => (v < 0 ? 0 : v > 255 ? 255 : v);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const c = (y >> 1) * chromaWidth + (x >> 1);
    const l = frame[y * width + x] - 16, u = frame[uBase + c] - 128, v = frame[vBase + c] - 128;
    const o = 4 * (y * width + x);
    rgba[o] = clamp((298 * l + 409 * v + 128) >> 8);
    rgba[o + 1] = clamp((298 * l - 100 * u - 208 * v + 128) >> 8);
    rgba[o + 2] = clamp((298 * l + 516 * u + 128) >> 8);
    rgba[o + 3] = 255;
  }
  return rgba;
}

async function encode(args) {
  const { module, wasmSha256 } = await loadModule(args.pkg);
  const width = +args.width, height = +args.height, count = +args.frames, fps = +args.fps;
  const frameBytes = width * height * 3 / 2;
  const yuv = await readFile(args.yuv);
  const planes = 'SourceFrame' in module;
  const frames = [];
  for (let i = 0; i < count; i++) {
    const frame = yuv.subarray(i * frameBytes, (i + 1) * frameBytes);
    frames.push(planes ? frame : i420ToRgba(frame, width, height));
  }
  const layout = Uint32Array.from([0, width, width * height, width / 2, width * height * 5 / 4, width / 2]);
  const passes = [];
  let bytes = 0;
  for (let pass = 0; pass < +args.repeat; pass++) {
    const encoder = new module.BrowserEncoder(args.codec, width, height, CODECS[args.codec].quantizer.initial, fps, +args.keySeconds * fps,
      args.fast ? 'fast' : 'default', undefined, false, 'bt709');
    encoder.set_rate_control(+args.kbps, +args.peakKbps, +args.bufferMs);
    const source = planes ? new module.SourceFrame(width, height) : null;
    const times = [];
    bytes = 0;
    for (const frame of frames) {
      const start = performance.now();
      if (planes) source.from_planes(frame, 'I420', 'bt709', width, height, layout);
      const packet = encoder.encode(planes ? source : frame);
      times.push(performance.now() - start);
      bytes += packet.length;
    }
    encoder.free();
    source?.free();
    passes.push(times);
  }
  return { mode: 'encode', codec: args.codec, yuv: args.yuv, wasm_sha256: wasmSha256,
    achieved_kbps: +(bytes * 8 * fps / count / 1000).toFixed(1), ...summarize(passes, count) };
}

const args = parseArgs(process.argv.slice(2));
const run = { decode, encode }[args.mode];
if (!run) { console.error('usage: bench-runner.mjs decode|encode ...'); process.exit(2); }
console.log(JSON.stringify(await run(args)));
