# Browser engines, 2026-11-27

`tools/web/browser_engines.py` (Playwright 1.60, headless, Linux x86-64). Conformance: the
official corpora decoded inside each engine's page with the module `web/wasm.js` picks there
(`engines.json`, build box); product smoke: `web/tests/browser-product-smoke.py` per engine
(`smoke-pc.json`, PC, no display).

| Engine | Module picked | H.264 (113, 110 required) | HEVC 147 | VVC 221 | AV1 240 | VP9 305 | Demo smoke |
|---|---|---|---|---|---|---|---|
| Chromium 148 | relaxed SIMD | 108, 0 required failing | 147 | 221 | 240 | 305 | pass |
| Firefox 150 | relaxed SIMD | 108, 0 required failing | 147 | 221 | 240 | 305 | pass (CPU display: no WebGL2 headless) |
| WebKit 26.4 | SIMD128 | 108, 0 required failing | 147 | **18 (fail)** | 240 | 305 | pass (test scene for the webcam steps) |
| Chromium 148 without SIMD | scalar | 108, 0 required failing | 147 | 221 | 240 | 305 | pass |

The five H.264 streams not decoded use FMO or SP/SI slices (not required; as natively).

WebKit and VVC: the 203 failing streams are the 10-bit ones; JavaScriptCore's optimizing tier
miscompiles `nc_dsp::vvc_intra::high_simd::simd128::predict_rect` (found by bisecting
`JSC_wasmFunctionIndexRangeToCompile`; with `JSC_useOMGJIT=0`, or the scalar build, they decode
exactly). Not fixed. Our encoders make 8-bit streams only, which decode exactly in WebKit; the gap
is other encoders' 10-bit VVC in Safari's engine. A first JSC miscompile of the same kind (H.264
chroma deblocking, `deblock::simd128::chroma_pair`, 51 streams) is worked around in this change.

Not tested: real Safari on macOS or iOS (other JIT back end on Apple silicon, other media stack),
mobile browsers, the webcam in WebKit (Playwright's Linux WebKit refuses getUserMedia).
