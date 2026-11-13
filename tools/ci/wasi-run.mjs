// Run a wasm32-wasip1 binary under Node's WASI implementation.
//
//   node tools/ci/wasi-run.mjs target/wasm32-wasip1/release/nc-checkasm.wasm [args...]
//
// Used by check.sh to run nc-checkasm with the SIMD128 kernels. Exits with
// the program's exit code.
import { spawnSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import { argv, execArgv, execPath, exit } from "node:process";

// Node 22.21/22.22 crash (SIGSEGV in V8's
// InnerPointerToCodeCache::GetCacheEntry during GC, returning through a
// WASI write) on the large checkasm module unless V8's fast API calls are
// off (docs/handoff/wasm-perf.md, "Node WASI validation caveat"). Re-run
// this script with the flag, so every caller gets it.
const FAST_API_OFF = "--no-turbo-fast-api-calls";
if (!execArgv.includes(FAST_API_OFF)) {
  const child = spawnSync(execPath, [FAST_API_OFF, ...execArgv, ...argv.slice(1)], {
    stdio: "inherit",
  });
  exit(child.status ?? 1);
}

const [wasmPath, ...args] = argv.slice(2);
if (!wasmPath) {
  console.error("usage: wasi-run.mjs <module.wasm> [args...]");
  exit(2);
}
const wasi = new WASI({ version: "preview1", args: [wasmPath, ...args], returnOnExit: true });
const module = await WebAssembly.compile(await readFile(wasmPath));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
exit(wasi.start(instance));
