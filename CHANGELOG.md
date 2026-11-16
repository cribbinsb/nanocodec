# Changelog

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
