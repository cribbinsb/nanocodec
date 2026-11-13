# VP9 and AV2 v1 implementation contract

Working scope, 8 November 2026. This extends the
[initial implementation plan](initial-av2-vp9-plan.md), following the request to
complete both native implementations autonomously. Work remains isolated on
`feature/vp9-av2`; other agents continue the main branch.

## Release meaning

V1 means a useful native Rust low-delay video encoder and matching emitted-subset
decoder, not a reference-library wrapper or merely headers. Each codec must work
through CLI and WASM/demo with fixed quantizer and shared realtime rate control.
Every emitted picture must reproduce encoder reconstruction in the own decoder
and two independent external implementations. A header or primitive oracle is
necessary development evidence, not a picture-conformance result.

The common initial format is 8-bit 4:2:0, one tile, one temporal layer, forward
prediction, keyframe requests and periodic reference resets. Supported geometry
must be stated explicitly; unsupported syntax must fail cleanly. High bit depth,
compound/warped prediction, film grain, screen-content tools, temporal layering
and multiple tiles are outside v1. Optional tool omissions must be legally
signalled, especially for AV2.

## Basic coding tools and integration

| Area | VP9 implementation | AV2 implementation |
|---|---|---|
| Framing and headers | Profile 0 key/inter headers; VP90 IVF | Final v1.0.0 OBUs, sequence and frame/tile headers |
| Entropy | Boolean coder, coefficient tokens and spatial contexts | Final AV2 range coder, true bypass, CDF adaptation and normative contexts |
| Intra | DC, vertical, horizontal, true motion; matching hybrid transforms | Small useful legal predictor set with normative transforms and contexts |
| Inter | One prior reconstruction, zero/new vectors, integer and subpixel compensation | One prior reconstruction, legal reference slots, motion candidates and subpixel compensation |
| Residual | Quantized AC/DC; coefficient scan variants; lossless path | Actual quantized AC/DC, legal scan/transform types; lossless if supported by subset |
| Partition/transform sizes | Start fixed 8x8 / 4x4, expand useful legal sizes before calling speed/quality competitive | Start smallest practical legal traversal, then expand useful sizes |
| Skip/filter | Skip syntax, effective skip contexts, normative in-loop deblocking | Skip syntax plus legally disabled or implemented normative filters |
| Realtime controls | Shared GOP, force keyframe, CBR controls, runtime target changes, rate logs | Same API with independently calibrated quantizer mapping |
| Product | CLI encode/decode, evaluator, WASM own decoder, side-by-side demo | Same; no AV2 UI selection until native picture streams are valid |

The fixed-block scalar VP9 baseline is not a performance target. Larger transforms,
partitions, improved search and SIMD remain necessary performance work. Record
support separately from conformance and competitiveness; do not silently turn a
feature milestone into a speed claim.

## Evidence and efficient execution

- Use the current shared streamed point engine and CPU-slot pool. Small gates may
  check the decoders sequentially; this changes scheduling, never acceptance.
- Generate diagnostic sources in memory or pipes. Keep compressed failures only
  when needed to reproduce a bug; do not write source/reconstruction/decoded YUV.
- Unit-test independent primitive goldens, then gate short complete streams.
  Cover spatial edges, quantizer extremes, static and changing pictures, motion,
  reference resets, and decoder errors before broad quality sweeps.
- Use source and held-out camera clips for quality/speed only after the full path
  is conformant. Measure speed separately from metric scoring. No BD-rate claim
  without a verified overlapping quality range and matching configurations.
- VP9 external pair: libvpx and FFmpeg-native. AV2 pair: pinned final AVM and a
  verified compatible dav2d revision. dav2d and rav2d are one implementation family.
- Full workspace CI, actual WASM roundtrips and a bounded malformed-input check
  accompany release integration. ARM compile checks do not claim ARM runtime.

Current evidence and exact remaining work live in
the implementation handoff (internal).

## Paper to guide AV2 tool exploration

User-requested research input (8 November 2026):
[Transform and Entropy Coding in AV2, Nalci et al., January 2026](https://arxiv.org/html/2601.02712v1).
Use this paper to inform tool selection and experiments, rather than assuming
AV2 gains require expensive mode searches. The detailed tool descriptions (§III-A–O), results (§IV-A–C) and Appendix A
have now been reviewed in the [entropy audit](av2-entropy-exploration.md).
That audit distinguishes published contributions from our measured results.
Check proposed syntax against our pinned final AVM v1.0.0: this paper predates
the final release.

Prioritize improvements to the basic coding path that can reduce bitrate without
expanding prediction or transform search substantially:

- Probability Adaptation Rate Adjustment (PARA, section III-I):
  per-tile CDF adaptation with trained rates is now enabled by default and
  independently picture-validated. Cross-frame context inheritance remains
  disabled. See the entropy audit for the measured cost and bitrate floor.
- Adaptive Transform Coding (ATC, III-J) and adaptive truncated Rice coding
  (III-K): audit which coefficient contexts, scanning rules and magnitude coding
  improvements our baseline already uses, then identify missing opportunities.
- True bypass coding: exact multi-bit batching is now implemented and
  checked against original-C bytes. Hardware throughput figures are not evidence
  of an equivalent software speedup.
- Consult the remaining transform, quantization and coefficient tools, and
  section IV's per-tool gain estimates, to rank subsequent experiments. Do not
  assume trellis quantization or additional transform searches are cheap simply
  because they appear alongside entropy improvements.

For each candidate, record existing support, incremental implementation/search
cost, and a focused experiment. Where syntax permits, compare enabled/disabled
variants with prediction and transform decisions held fixed to isolate entropy
effects; then measure end-to-end bitrate/quality and CPU on broad, held-out
content. Run the standard independent-decoder gates. Published gains motivate
experiments, but are neither additive nor measurements of nanocodec.

## v1 acceptance evidence

Both native I/P implementations, emitted-subset decoders, CLI and browser paths
are implemented. The internal handoff has exact commits,
binaries, controls, benchmark bounds and reproduction commands.

- Normative primitives and complete streams pass independent decoder checks;
  TX4/TX8, mixed VP9 block sizes, lossless extremes and malformed-input recovery
  are covered.
- Shared CQP/CBR, target updates, forced/periodic keys, optional scene detection,
  input layouts, VFR timestamps and dropped-frame resumption are integrated.
- Actual WASM tests cover 9,600 encode/decode pairs, simultaneous instance
  isolation, memory stability and real Chromium synthetic-webcam product flows.
- Host CI, actual WASM SIMD, native ARM tests and NEON checkasm pass. Quiet
  Jetson runs report encode/decode time; busy-host measurements use instructions.
- Initial fully gated, source-cap-compliant quality comparisons against libvpx,
  AVM and the frozen AV1/HEVC/SVT baselines are recorded. Missing rate ranges and
  insufficient overlap are explicit. These prototypes do not meet reference
  compression/CPU parity; performance remains experimental as permitted by M5.

The automated browser source is synthetic; physical-camera visual assessment
and broader long-session/long-GOP performance remain further product evaluation.
No general-purpose decoder, high-bit-depth, multilayer or reference-parity claim
is implied by this basic v1 release.
