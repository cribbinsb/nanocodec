// Official conformance corpora through the browser module (web/pkg) under node,
// split into packets exactly as the page's benchmarks split them (packets.mjs).
// The native decoder's corpus runs (tools/ci/conformance.py) do not see the
// browser entry point or the splitter: the HEVC vector DELTAQP_B_SONY_3 (four
// slices per picture) decoded natively but failed here, because the splitter
// then cut one packet per slice (2026-11-22).
//
//   node conformance-runner.mjs --pkg DIR --codec C --manifest JSON --streams DIR
//        [--must-pass FILE] [--only A,B] [--jobs N] [--json OUT]
//
// Each 4:2:0 progressive stream is decoded with BrowserDecoder.decode,
// one packet per picture, taking every picture it shows (next_frame, and
// finish at the end, where the module has them), cropped Y, Cb, Cr: 8-bit
// streams as bytes (decoded_i420), deeper ones as the native samples in
// 16-bit little-endian words (decoded_i420_u16), the reference's own format.
// The browser decoders show pictures in output order, so their output must
// equal the reference's: the verdict is the whole-output MD5, as for the native
// runs (tools/ci/conformance.py). The manifest's per-picture MD5s only name the
// first differing picture: they are the reference output cut at the first
// picture's size, so they do not split it where pictures change size (VVC RPR_*,
// BOUNDARY_A_3) or are field pairs shown as frames (FIELD_*), where seven VVC
// streams equal to the reference failed on the count alone (2026-11-23).
// 4:0:0 and interlaced streams are left out (the browser shows 4:2:0 frames).
// --json writes each compared stream's result and the counts.
// Exit status 0 when every stream matches, or with --must-pass FILE (one name
// per line, # comments) every listed one.
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { availableParallelism } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { streamPackets } from './packets.mjs';

/** The shown pictures' MD5s and the MD5 of all of them, or the error that stopped the decode. */
async function decodeStream({ pkg, codec, file, deep }) {
  const module = await import(pathToFileURL(path.join(pkg, 'nc_web.js')).href);
  await module.default({ module_or_path: readFileSync(path.join(pkg, 'nc_web_bg.wasm')) });
  const decoder = new module.BrowserDecoder(codec);
  const md5s = [], whole = createHash('md5');
  const take = () => {
    const samples = deep ? decoder.decoded_i420_u16() : decoder.decoded_i420();
    // WebAssembly memory is little-endian, so a u16 view's bytes are the reference's layout.
    const picture = new Uint8Array(samples.buffer, samples.byteOffset, samples.byteLength);
    whole.update(picture);
    md5s.push(createHash('md5').update(picture).digest('hex'));
  };
  // Modules before output-order display have neither.
  const drain = () => { while ('next_frame' in decoder && decoder.next_frame()) take(); };
  try {
    for (const packet of streamPackets(new Uint8Array(readFileSync(file)), codec)) {
      decoder.decode(packet);
      if (decoder.frame_shown) take();
      drain();
    }
    if ('finish' in decoder) decoder.finish();
    drain();
    return { md5s, whole: whole.digest('hex') };
  } catch (error) {
    return { md5s, error: String(error) };
  } finally {
    decoder.free();
  }
}

if (!isMainThread) {
  parentPort.on('message', async job => parentPort.postMessage(await decodeStream({ ...workerData, ...job })));
} else {
  await main();
}

function parseArgs(argv) {
  const args = { jobs: Math.min(8, availableParallelism()), only: '' };
  for (let i = 0; i < argv.length; i += 2) args[argv[i].replace(/^--/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase())] = argv[i + 1];
  for (const key of ['pkg', 'codec', 'manifest', 'streams']) if (!args[key]) throw new Error(`--${key} is required`);
  return args;
}

/** The stream names in a list file: each line that is not a comment. */
function nameList(file) {
  return new Set(readFileSync(file, 'utf8').split('\n').map(l => l.trim()).filter(n => n && !n.startsWith('#')));
}

/** Whether a manifest entry has samples deeper than 8 bits. */
function isDeep(entry) {
  return entry.bit_depth?.some(d => d > 8) ?? false;
}

/** The manifest's entries and those this check can compare: 4:2:0 progressive. */
function comparable(manifest, only) {
  const data = JSON.parse(readFileSync(manifest, 'utf8'));
  const names = new Set(only ? only.split(',') : []);
  const all = (Array.isArray(data) ? data : data.streams).filter(e => !names.size || names.has(e.name));
  const entries = all.filter(e => e.chroma_format === '4:2:0' && (e.scan ?? 'progressive') === 'progressive');
  return { all, entries };
}

/** Why the shown pictures are not the reference's, or null when they are. */
function mismatch(entry, { md5s, whole, error }) {
  const want = entry.frame_md5?.length ? entry.frame_md5 : null;
  if (error) return `${error} after ${md5s.length}${want ? ` of ${want.length}` : ''} pictures`;
  if (whole === entry.md5) return null;
  const first = want?.length === md5s.length ? md5s.findIndex((m, i) => m !== want[i]) : -1;
  return first < 0 ? `the ${md5s.length} pictures differ from the reference's output`
    : `picture ${first} (output order) differs from the reference`;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const { all, entries } = comparable(args.manifest, args.only);
  const mustPass = args.mustPass ? nameList(args.mustPass) : null;
  const results = new Map();
  const queue = [...entries];
  const workers = Array.from({ length: Math.min(+args.jobs, queue.length) }, () =>
    new Worker(new URL(import.meta.url), { workerData: { pkg: path.resolve(args.pkg), codec: args.codec } }));
  await Promise.all(workers.map(worker => new Promise(resolve => {
    let entry;
    const next = () => {
      entry = queue.shift();
      if (!entry) { worker.terminate(); resolve(); return; }
      worker.postMessage({ file: path.join(args.streams, entry.bitstream), deep: isDeep(entry) });
    };
    worker.on('message', result => { results.set(entry.name, mismatch(entry, result)); next(); });
    worker.on('error', error => { results.set(entry.name, `worker: ${error}`); resolve(); });
    next();
  })));
  let failedRequired = 0;
  for (const entry of entries) {
    const problem = results.has(entry.name) ? results.get(entry.name) : 'not run';
    const required = !mustPass || mustPass.has(entry.name);
    if (problem && required) failedRequired++;
    if (problem) console.log(`${required ? 'FAIL' : 'fail (not required)'} ${entry.name}: ${problem}`);
  }
  const passed = entries.filter(e => results.has(e.name) && !results.get(e.name)).length;
  const deep = entries.filter(isDeep);
  const deepPassed = deep.filter(e => results.has(e.name) && !results.get(e.name)).length;
  console.log(`${args.codec} WASM (${path.basename(path.resolve(args.pkg))}): ${passed}/${entries.length} streams match the reference`
    + ` (deeper than 8 bits ${deepPassed}/${deep.length}; ${all.length - entries.length} of the manifest's ${all.length} not 4:2:0 progressive)`
    + (mustPass ? `, ${failedRequired} required failing` : ''));
  if (args.json) {
    const streams = Object.fromEntries(entries.map(e => [e.name, {
      bit_depth: e.bit_depth, profile: e.profile ?? null,
      required: !mustPass || mustPass.has(e.name),
      problem: results.has(e.name) ? results.get(e.name) : 'not run',
    }]));
    const skipped = all.filter(e => !entries.includes(e))
      .map(e => ({ name: e.name, chroma_format: e.chroma_format, scan: e.scan ?? 'progressive' }));
    writeFileSync(args.json, JSON.stringify({
      codec: args.codec, pkg: path.basename(path.resolve(args.pkg)), manifest: path.basename(args.manifest),
      manifest_streams: all.length, compared: entries.length, passed,
      deep_compared: deep.length, deep_passed: deepPassed, required_failing: failedRequired,
      skipped, streams,
    }, null, 1) + '\n');
  }
  process.exit(failedRequired ? 1 : 0);
}
