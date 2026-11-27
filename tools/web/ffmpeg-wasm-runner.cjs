// One timed ffmpeg.wasm (@ffmpeg/core, single-thread) run under node, for
// tools/web/ffmpeg_wasm_yardstick.py: the yardstick our browser module is compared with.
//
//   node ffmpeg-wasm-runner.cjs CORE_JS INPUT_FILE OUTPUT_NAME -- FFMPEG_ARGS...
//
// INPUT_FILE is written into the module's memory file system as "in" (untimed);
// FFMPEG_ARGS name it "in" and write OUTPUT_NAME (or -f null -). The run adds
// FFmpeg's -benchmark_all, which logs the wall time of every decoder and encoder
// call (fftools: update_benchmark around each avcodec call, microseconds).
// Prints one JSON object:
//   rc, frames (from ffmpeg's progress log), outBytes (OUTPUT_NAME's size, 0 if none);
//   decode_ms, encode_ms: the sums of the decode_video and encode_video calls, the
//     codec alone: no module start-up, probing, demuxing, command parsing, filters
//     or muxing (what our runner times for ours);
//   startup_ms: module instantiation (createFFmpegCore);
//   exec_ms: the whole exec (probe, demux, codec, filters, mux, and the logging);
//   ms: startup_ms + exec_ms, the measure the records before 2026-11-23 used.
const fs = require('node:fs');

/** `bench: USER user SYS sys REAL real TASK` (fftools/ffmpeg.c update_benchmark). */
const BENCH = /bench:\s+\d+ user\s+\d+ sys\s+(\d+) real (decode_video|encode_video)/;

async function main() {
  const [coreJs, input, outName, sep, ...args] = process.argv.slice(2);
  if (sep !== '--') throw new Error('usage: CORE_JS INPUT OUTPUT_NAME -- ARGS...');
  const data = fs.readFileSync(input);
  // The core is built for browsers and workers: it reads `self` and is given its
  // .wasm (beside CORE_JS) instead of fetching it.
  globalThis.self = Object.assign(globalThis, { location: { href: `file://${coreJs}` } });
  const wasmBinary = fs.readFileSync(coreJs.replace(/\.js$/, '.wasm'));
  const createFFmpegCore = require(coreJs);
  let frames = 0;
  const codecUs = { decode_video: 0, encode_video: 0 };
  const started = performance.now();
  const core = await createFFmpegCore({ wasmBinary });
  const ready = performance.now();
  core.setLogger(({ message }) => {
    const m = /frame=\s*(\d+)/.exec(message);
    if (m) frames = Math.max(frames, Number(m[1]));
    const b = BENCH.exec(message);
    if (b) codecUs[b[2]] += Number(b[1]);
  });
  core.FS.writeFile('in', data);
  const run = performance.now();
  const rc = core.exec('-nostdin', '-threads', '1', '-benchmark_all', ...args);
  const done = performance.now();
  let outBytes = 0;
  try { outBytes = core.FS.stat(outName).size; } catch { /* no output file (-f null) */ }
  const startupMs = ready - started, execMs = done - run;
  process.stdout.write(JSON.stringify({
    rc, frames, outBytes, decode_ms: codecUs.decode_video / 1000, encode_ms: codecUs.encode_video / 1000,
    startup_ms: startupMs, exec_ms: execMs, ms: startupMs + execMs,
  }) + '\n');
}
main().catch((e) => { console.error(e); process.exit(1); });
