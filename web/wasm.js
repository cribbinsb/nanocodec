// Which build of the codec module this browser loads, shared by the workers
// and the page (options.js), so all fetch the same file: the relaxed-SIMD
// build where the engine supports relaxed SIMD (bit-identical output, faster
// lane selects), else the plain SIMD128 build, else (an engine without
// WebAssembly SIMD: Safari before 16.4, Firefox before 89, Chrome before 91)
// the scalar build, nc-dsp's scalar reference kernels (same output, slower).
//
// The page compiles the module once (`compiledModule`) and posts the compiled
// WebAssembly.Module to every codec worker, which instantiates it: four workers
// each fetching and compiling the 7 MB module themselves took 2.3x as long to
// be ready (headless Chromium, fresh context, six interleaved loads: median
// 884 ms against 378 ms until four workers had made an encoder).
// The probes are functions returning a v128: `i8x16.relaxed_swizzle` of two
// splats, and `i8x16.splat` of a constant.
const RELAXED_SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
  3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11]);
const SIMD128_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
  3, 2, 1, 0, 10, 8, 1, 6, 0, 65, 0, 253, 15, 11]);
export const SIMD128 = WebAssembly.validate(SIMD128_PROBE);
export const RELAXED_SIMD = SIMD128 && WebAssembly.validate(RELAXED_SIMD_PROBE);
/** The module directory under pkg/ ('' for the SIMD128 build) and how the page names it. */
export const BUILD = RELAXED_SIMD ? { dir: 'relaxed/', name: 'SIMD128 + relaxed SIMD (pkg/relaxed)' }
  : SIMD128 ? { dir: '', name: 'SIMD128 (pkg)' }
    : { dir: 'scalar/', name: 'scalar, no WebAssembly SIMD (pkg/scalar)' };
export const MODULE_URL = new URL(`./pkg/${BUILD.dir}nc_web.js`, import.meta.url).href;
const WASM_URL = new URL(`./pkg/${BUILD.dir}nc_web_bg.wasm`, import.meta.url).href;

// All builds carry the decoders' analysis records (analysis-view.js), off
// until a decoder's `set_analysis(true)`, so analysis mode needs no other module.

let compiled = null;
/**
 * The codec module's WebAssembly.Module, compiled once per page while it
 * downloads (a server that does not send `application/wasm` gets a plain
 * compile of the fetched bytes).
 */
export function compiledModule() {
  compiled ??= WebAssembly.compileStreaming(fetch(WASM_URL))
    .catch(async () => WebAssembly.compile(await (await fetch(WASM_URL)).arrayBuffer()));
  return compiled;
}
