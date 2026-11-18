# Changelog

## 0.3.0 (2026-11-18)

- **Speed figures.** The results site's speeds are measured on this release's profile-guided (PGO)
  builds of the command-line tool and C API (x86_64 and aarch64, attached to the GitHub release).
  crates.io carries source: a crate built by cargo is a plain release build, which took 1.4-3.2 % more
  CPU on x86 than PGO (AV1, HEVC and H.264 Faster, 2026-11-11). The browser (WASM) module is a plain
  build.

### Decoders
- **VVC Main 10 SIMD.** The 10-bit VVC kernels (ALF and CC-ALF, MC with RPR and GPM, DMVR/BDOF/PROF,
  transforms, dequantisation, LMCS, deblock, SAO, intra) have AVX2, AVX-512, NEON and WASM tiers beside
  the scalar reference, each checked bit-exact against it.
- **Faster, output unchanged:**
  - AV1 decode -2.5 % cycles from the shared block copies (Jetson, fixed clock, 90 frames, all frames
    identical);
  - VP9 decode -0.8 % instructions (packed inverse transform);
  - H.264, HEVC and VVC deblocking filter independent edges together in SIMD lanes;
  - AV1/AV2 CDEF on NEON (below).

### Encoders
- **New preset: AV1 Veryfast**, below Faster. Its keys and base frames take Faster's values of its four
  per-block shortcuts, and temporal layers switch on camera jitter alone. On the Fixed QP lens (against x264 medium)
  its last change took it from +7.7 / -6.2 to +3.3 / -10.4 (Weighted / J BD-rate) at 1024-1044 fps
  on the c8a fixed-QP chart path (SVT-AV1 M13: 979-987 fps in the same sessions).
- **Fast presets code keys and base frames with the Default's per-block tools** (base-only effort);
  Fixed QP lens, Weighted / J against x264 medium, before -> after:
  - VP9 Fast +0.5 / -13.7 -> -7.6 / -18.2;
  - AV2 Fast -11.7 / -20.6 -> -16.1 / -24.7;
  - HEVC Fast -3.1 / -11.9 -> -6.0 / -15.5 (intra NxN and trellis stay off);
  - AV1 Faster takes Fast's per-block tools the same way.
- **Adaptive tool thresholds** (`adapt.*`): per clip and frame role, a gate's threshold follows the
  blocks just inside it. On in AV1 Faster, HEVC Fast and Faster and VVC Fast: CBR pool J -0.6 to
  -1.4 % (L3, 2026-11-17 draw, CPU-credited).
- **Rate control.** Two engine defaults now on: `rate.key_reserve` (the frames before a periodic key
  leave room for it; H.264 Default J -0.06 %, AV1 Default -0.01 %, 600-frame final-check) and
  `rate.key_quality_match` (a key is planned at its inter frames' measured quality where the inter
  level asks less; only with the quiet run-up before keys). Every rate-controlled preset with
  periodic keys codes different streams. `rate.key_level_mean` is new and off by default.

The fixed-QP and CBR results are on the
[results site](https://cribbinsb.github.io/nanocodec/results/).

### Speed
All exact (every shipped preset of every codec codes identical streams):
- **CDEF on NEON** ported from dav1d (BSD-2-Clause): byte filters and one-pass direction search, shared
  by AV1 and AV2. Whole AV1 encode on the Jetson (720p, 150 frames, five pairs): Faster -0.90 %,
  Veryfast -0.26 % cycles.
- **One block-copy family** (plain, rounded average, byte to i16) for prediction, reconstruction and
  decoder MC in AV1, VP9 and AV2, on every ISA tier. Jetson AV1 encode -0.66 % (Faster) and -0.64 %
  (Veryfast) cycles.
- **AV1 coefficient coding and pricing** (per-size scan tables, a level map saturated at the base
  levels as in libaom, a one-interval symbol replay): x86 instructions AV1 Default -3.0 %, Fast -1.8 %,
  Faster -2.0 % (40 frames, one 720p clip).
- **AVX-512 (Ice Lake / Zen 4) forward transforms and VVC ALF row:** Zen 5 (c8a) encode throughput
  AV1 Default +1.4 %, Fast +1.5 %, VP9 +4.3 %, HEVC +1.3 %, VVC +2.7 %; H.264 measured 0.96 %
  slower in the same run, with no change in its executed instructions.
- **Deblocking edge batches** for H.264, HEVC and VVC on every tier (patterns from dav1d, BSD-2-Clause):
  Jetson Default encode cycles H.264 -0.90 %, VVC -0.42 %, HEVC level; Zen 5 H.264 time +1.4 %.
- **Picture borders** with ordinary stores on every ISA: Zen 5 encode -3.2 to -4.0 % CPU with AVX2,
  Zen 3 -0.6 %.
- AVX-512 keeps the AVX2 kernels where they are faster (small-block variance and sums, guarded 16-bit
  AV1 inverses), and AVX2 fallbacks are compiled for AVX2.

### Tools
- **Release builds.** Releases ship PGO builds of the CLI and C API for x86_64 and aarch64, trained on
  public Xiph 720p sequences that no chart uses, with identical streams to the plain build checked for
  every preset (`tools/release/release.py`).
- **Results site:** speed series on AWS c6a (Zen 3, AVX2 only) and c8g (Graviton4) beside Zen 5 (c8a)
  and the Jetson Orin Nano.
- **nc-checkasm:** benchmarks for the VVC rectangular and approximate 64x64 forward transforms and
  rectangular variance and sums.

### Breaking
- **nc-common:** the `BorderStores` trait and the `*_with` border-extension variants of `Plane` and
  `Picture` are removed; `extend_borders` / `extend_border_rows` take no dispatch table.

## 0.2.0 (2026-11-16)

### Decoders
- **10-bit decoding.** HEVC Main 10, AV1 Main 10 (also 10-bit monochrome and film grain), VP9 profile 2
  (10 and 12-bit), VVC Main 10 and H.264 High 10, chosen per sequence. Deep pictures are returned as
  `DecodedPicture` with u16 samples. The existing 8-bit calls stay 8-bit only.
- **Conformance:**
  - HEVC_v1 147/147.
  - AV1 libaom 242/242 (66 of them 10-bit).
  - VP9 libvpx 305/305.
  - VVC official 224/226 (field-pair output missing).
  - H.264 ITU-T H.264.1 110/190: every progressive Main, High and High 10 stream, plus B slices. Interlace,
    FMO/ASO and SP/SI are not supported.
  - AV2 30/30 foreign streams (inter prediction, segmentation, film grain).
- **10-bit SIMD.** AV1 has AVX2, AVX-512, NEON and WASM tiers, at about 1.64× dav1d's instructions on a
  1080p 10-bit stream. VP9 has AVX2 and NEON. HEVC/VVC have WASM for their shared kernels.
- **nc-web** decodes 10-bit streams: native u16 output, depth getters, and 8-bit display conversion.

### Encoders
- **New presets:**
  - AV1 Fast, Faster and Default re-pointed.
  - HEVC Fast re-pointed.
  - H.264 Default prices 5 Intra_4x4 modes.

  The fixed-QP and CBR results are on the
  [results site](https://cribbinsb.github.io/nanocodec/results/).
- **Rate control.** A quiet second before periodic keys is now the engine default, which bounds the
  key-frame dip on still scenes. `decision.exact_rank_margin` now defaults to 0.
- **Speed, all exact** (streams identical):
  - shared SIMD kernels across codecs: quantiser, intra, MC, reconstruction, end of block, deblock, CABAC
    encoder, butterfly transforms;
  - AVX-512 (Ice Lake / Zen 4 feature set) MC, deblock and CDEF;
  - AV1/VP9/VVC entropy coding and pricing;
  - mode decision;
  - pre-analysis statistics.
- **New default-off controls:** per-superblock effort budget (`effort.*`, `light.*`), frame-level filter
  gates (`filter.*`), quantizer/energy work gates (`gate.*`).

### Breaking
- **AVX-512 tier:** it now requires the Ice Lake / Zen 4 feature set (GFNI, VBMI, VBMI2…). Older AVX-512
  CPUs use the AVX2 tier.
- **Decoder API:** frame-by-frame decoding goes through `push_nal` / `next_output` / `flush` (H.264, VVC,
  AV1, VP9), which return `DecodedPicture`.

## 0.1.0 (2026-11-15)
- First release on crates.io.
