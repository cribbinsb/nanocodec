// Which build of the codec module this browser loads, shared by the workers
// and the page (options.js), so both fetch the same file: the relaxed-SIMD
// build where the engine supports relaxed SIMD (bit-identical output, faster
// lane selects), else the plain SIMD128 build.
// The probe is a function returning `i8x16.relaxed_swizzle` of two splats.
const RELAXED_SIMD_PROBE = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123,
  3, 2, 1, 0, 10, 15, 1, 13, 0, 65, 1, 253, 15, 65, 2, 253, 15, 253, 128, 2, 11]);
export const RELAXED_SIMD = WebAssembly.validate(RELAXED_SIMD_PROBE);
export const MODULE_URL = new URL(RELAXED_SIMD ? './pkg/relaxed/nc_web.js' : './pkg/nc_web.js', import.meta.url).href;
