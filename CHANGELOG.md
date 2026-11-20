# Changelog

## 0.5.0 (2026-11-20)

- **Speed figures.** The results site's speeds are measured on this release's profile-guided (PGO)
  builds of the command-line tool and C API (x86_64 and aarch64, attached to the GitHub release).
  crates.io carries source: a crate built by cargo is a plain release build, which took 1.4-3.2 % more
  CPU on x86 than PGO (AV1, HEVC and H.264 Faster, 2026-11-11). The browser (WASM) module is a plain
  build.
- **x86-64-v3 builds.** The repository's `.cargo/config.toml` now builds x86_64 with
  `-C target-cpu=x86-64-v3` (Haswell / Zen 1 or later): encode instructions -2.7 to -4.4 %, decode
  -0.4 to -1.7 %, streams identical. It does not reach crates built by cargo from crates.io; set
  `RUSTFLAGS` yourself (README, "Build").

### Presets: a better-spaced ladder
- **Slow** was re-picked for every codec (everything heavy on, then cut back to the knee), measured
  against each Default at fixed QP: H.264 J -4.4 %, HEVC -6.6 %, VVC -4.2 %, AV1 -5.5 %, VP9 -2.0 %,
  AV2 -7.0 %, at 1.6-2.5x the Default's P-frame instructions. The old Slow presets were within
  -2.1..+0.2 % of their Defaults.
- **Fast, Faster and Veryfast** take the Defaults' cheap levers where they pay at the faster rungs'
  CPU weighting (pool screens, J with CPU credit): VVC Fast -10.3 %, HEVC Fast -6.4 %, HEVC Faster
  -5.5 %, H.264 Fast -5.9 %, Faster -3.9 %, Veryfast -3.9 %, VP9 Fast -2.2 %, Faster -3.7 %,
  Veryfast -5.9 %, AV1 Fast -1.9 %, Faster -1.2 %, AV2 Fast -0.3 %, Faster -2.9 %; each within
  about +-2 % of its old CPU.

### Encoders
- **Keys in rate control.** A key now matches the quality the inter frames hold on the unchanged part
  of a still picture, an earlier key's unpaid loan carries into the next key, and with key periods up
  to 5 s the frames before a key no longer save for it (the key's loan is repaid after it). 600-frame
  final-checks J -0.01 to -0.73 % across 11 presets; with keys every 2 s, BD PSNR-YUV -0.3 to -1.0 %
  and the frames before a key up to 2.7 dB better on still scenes. On still scenes with 2 s keys the
  first 2.5 s can run up to 1.17x the target; every later window and the long-term rate stay at or
  under it.
- **VVC 64-point transform units** in the Default, with the planner pricing the detail a 64-point unit
  cannot code (H.266 zero-out above frequency 32): pool J -0.9 %, 600-frame final-check -0.75 %.
- **Still-scene level damping** on in the AV1, VVC and AV2 Defaults (VVC final-check -0.65 %).
- **Encode speed (exact, streams identical):** AV2 Default -7.3 % and Faster -15.2 % instructions
  against 0.4.0 (measured 2026-11-20 morning); AV1 Faster -6.4 % and VP9 Faster/Veryfast -2.3/-2.8 %
  from per-block work, then 16-bit forward DCTs (AV1/VP9 16-64 points, one driver for AVX2, NEON and
  SIMD128): VP9 -4.5 %, AV1 -2.6..-2.8 %; HEVC Faster -1.4 %, VVC Fast -1.9 %. Every change kept was
  also faster on the Jetson.

### Decoders
- **AV1 decode** on a libaom 720p stream: 1259 M -> 1193 M instructions (-5.2 %, dav1d 909 M).
  Deblocking moved to dav1d's lane layout (its kernel now runs fewer instructions than dav1d's);
  self-guided restoration, filter intra, small-block directional prediction, edge reading and inverse
  transforms ported from or modelled on dav1d. On the Jetson (NEON) the directional, self-guided and
  Wiener kernels followed: about -2 % cycles.

### Measurement
- **PSNR is capped at 60 dB per plane** (libvmaf's 8-bit cap; was 100 dB). A perfectly coded flat plane
  no longer counts as 100 dB, which swung grayscale clips by several dB. Published J moves by more
  than 0.1 on 7 rows, all on two flat-chroma clips. Results before 2026-11-20 used 100 dB.
- **History page:** the results site animates our Pareto front, 2026-11-07 to 2026-11-20, every preset
  re-measured on a fixed 28-clip slice and timed on AWS c8a.

## 0.4.0 (2026-11-19)

- **Speed figures.** The results site's speeds are measured on this release's profile-guided (PGO)
  builds of the command-line tool and C API (x86_64 and aarch64, attached to the GitHub release).
  crates.io carries source: a crate built by cargo is a plain release build, which took 1.4-3.2 % more
  CPU on x86 than PGO (AV1, HEVC and H.264 Faster, 2026-11-11). The browser (WASM) module is a plain
  build.

### New results set
- **VC-1K phone video calls.** A CBR chart (100 clips) and a Fixed QP view from Meta's VC-1K dataset
  (CC BY-NC 4.0; numbers only). A further 30 VC-1K clips, disjoint from the measurement set, joined the
  development pool as a "videocall" class. Most changes below came from fixing what this set exposed.

### Encoders
- **RD final decision (H.264 Default and Slow).** A shared engine stage weighs the ranking's winner, the
  skip vector and the best 16x16 intra prediction by luma coded cost (CABAC bits) where the cheap ranking
  cannot separate them. VC-1K six worst clips J -9.8 % (fixed QP); CBR pool J -2.5 % (108 clips),
  600-frame final-check -2.7 %; about +7-11 % instructions on Default, ~0 on presets with it off.
- **Winner walk (H.264 Default).** The ranking's winning vector is refined over its quarter-sample
  neighbours (never the skip vector or a static block): pool J -0.3 %, final-check -1.9 %, +1 % CPU.
- **Moving-camera keys at fixed QP (HEVC, AV1, VP9, AV2 Default).** Under camera motion a fixed-QP key is
  coded up to 6 QP coarser, ramping with motion; a fixed camera keeps its fine key. Fixed-QP pool J
  -0.8 % (HEVC), -1.5 % (AV1, AV2), -1.9 % (VP9); about -4 % on each codec's worst VC-1K clips. CBR is
  unchanged.
- **AV2 Fast and Faster CDEF.** A measured probe (off, the old curve and a short strength ladder on
  sampled rows) replaces the quantiser-only strength model, which was far too weak on dark and flat
  video: VC-1K J -14.0 % / -15.7 %, CBR final-check -5.5 % / -6.3 %, +4 % time.
- **AV2 coefficient rate model.** Mode-decision trials price coefficients with a fitted model in AV2
  Default, Fast and Faster; Default codes 7.7 % faster at the same quality.
- **Shared SAO for HEVC and VVC** (HM's encoder structure, exact context rates, band offset available):
  VVC CBR J -0.3 to -0.5 %, HEVC level.
- **VP9 Veryfast** takes 16x16 inter transforms and a lower pre-transform skip threshold: CBR pool J
  -6.2 % (CPU-credited), +3 % instructions.
- **AV1 Faster** uses causal layer propagation (an MB-tree within the layer period) on base blocks.
- **Rate control (H.264 Default).** On still pictures the level's per-frame step is damped by the share of
  unchanged blocks, so tiny decision changes no longer swing a still clip's bitrate by up to 14 %.

### Decoders
- AV1 10-bit decode closer to dav1d (1.64x -> 1.21x its time); VVC Main 10 CABAC and kernel speed-ups;
  VP9 10-bit deblock, MC and inverse transforms in SIMD on every tier.

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
