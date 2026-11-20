// One timed ffmpeg.wasm (@ffmpeg/core, single-thread) run under node, for
// tools/web/ffmpeg_wasm_yardstick.py: the yardstick our browser module is compared with.
//
//   node ffmpeg-wasm-runner.cjs CORE_JS INPUT_FILE OUTPUT_NAME -- FFMPEG_ARGS...
//
// INPUT_FILE is written into the module's memory file system as "in" (untimed);
// FFMPEG_ARGS name it "in" and write OUTPUT_NAME (or -f null -). Prints one JSON
// object: rc, ms (module start-up plus the run, without the input copy), outBytes
// (OUTPUT_NAME's size, 0 if none) and frames (from ffmpeg's progress log).
const fs = require('node:fs');

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
  const started = performance.now();
  const core = await createFFmpegCore({ wasmBinary });
  const ready = performance.now();
  core.setLogger(({ message }) => {
    const m = /frame=\s*(\d+)/.exec(message);
    if (m) frames = Math.max(frames, Number(m[1]));
  });
  core.FS.writeFile('in', data);
  const run = performance.now();
  const rc = core.exec('-nostdin', '-threads', '1', ...args);
  const done = performance.now();
  let outBytes = 0;
  try { outBytes = core.FS.stat(outName).size; } catch { /* no output file (-f null) */ }
  process.stdout.write(JSON.stringify({ rc, ms: (ready - started) + (done - run), outBytes, frames }) + '\n');
}
main().catch((e) => { console.error(e); process.exit(1); });
