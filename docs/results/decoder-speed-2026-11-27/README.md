# Decoder speed of release 0.7.0, x86 and Arm (2026-11-27)

The results site's Decoders page (`tools/bench/pages/decoders.html`, data built by
`tools/bench/preview_decoders.py` from this directory). Same method and harnesses as
`../decoder-speed-2026-11-23/` (decode alone on both sides), now on the benchmark's two
timing families and on the 0.7.0 release.

- **Ours:** the `v0.7.0` tag (`ce89975ea`) built with `cargo build --release` in the
  repository (fat LTO, one codegen unit, x86-64-v3 code generation on x86; no PGO, as
  published on crates.io), on the build box: x86_64 natively, aarch64 cross-compiled. The
  version line ends in `-dirty`: the build's only change from the tag is Cargo.lock's stale
  `nc-checkasm` version (0.6.0 to 0.7.0; the tag's lockfile fails `--locked`). Browser: the
  tag's `tools/web/build.sh` module, its relaxed-SIMD build.
- **Machines:** AWS c8i.2xlarge (Intel Xeon 6975P-C, Granite Rapids; our decoders on their
  AVX-512 tier) and c8g.2xlarge (Graviton 4, Neoverse V2; NEON and SVE2), spot, eu-north-1
  (`tools/dev/aws_decode_speed.py --machine ... --cpus 1,2,3`). The streams were split over
  three physical cores (vCPUs 1-3, the SMT siblings idle on c8i), one `decoder_speed.py`
  process per core, the box otherwise idle; each stream's decoders ran in the same process's
  interleaved rounds. Browser runs: node v22.21.1 on vCPU 1 after the decoder runs.
- **Method:** `tools/dev/decoder_speed.py`, 5 interleaved rounds x 3 passes: the median of
  15 passes per decoder and stream. Browser: `tools/web/ffmpeg_wasm_yardstick.py`, 7 rounds.
- **References:** the same pinned harnesses as 2026-11-23: FFmpeg n8.0.1, dav1d 1.5.4 (through
  libavcodec), libvpx v1.16.0, VVdeC d261469, AVM v1.0.0, dav2d 237229e with its assembly. x86:
  `~/nanocodec-data/anchors/x86_64/bin`; Arm: the generic aarch64 builds made on the Jetson
  (no `-march`; `build_anchors.sh avbench decbench`), copied to
  `~/nanocodec-data/anchors/aarch64/bin`. ffmpeg.wasm: `@ffmpeg/core` 0.12.10.
- **Streams:** `streams.json` (size and encoder of each). The 2026-11-23 yardstick streams
  (`~/nanocodec-data/results/decoder-speed-2026-11-23/streams`, `streams2`), plus 77 BP-4
  benchmark streams: 7 clips of the "ours" held-out set, one per content class (960x540 to
  1280x720, 600 frames, the 1200 kbps rung), coded by our six encoders at Medium and by x264
  medium, x265 medium, libvpx VP9 rt 7, SVT-AV1 M10 and libaom rt 8 (from
  `~/nanocodec-data/bench/bp4-ours-{ours,refs}-c8g`). Numbers only: no stream is published.
- **Files:** `<machine>/decoders-cpu{1,2,3}.json` (every pass), `<machine>/yardstick.json`,
  the platform notes, `about.json` (the page's statements).
- **Cost:** c8i 16.3 min, $0.02; c8g 21.0 min, $0.03 (spot); builds on the shared build box (15 min of one job slot).

## Summary

Decode time, ours ÷ reference, geometric mean over streams (all / our encoder's / other encoders' streams), and geometric mean Mpx/s.

| format | reference | streams | c8i all | c8i ours / other | c8i Mpx/s ours / ref | c8g all | c8g ours / other | c8g Mpx/s ours / ref |
|---|---|---|---|---|---|---|---|---|
| H.264 | FFmpeg | 18 | 1.16 | 1.16 / 1.16 | 644 / 747 | 0.94 | 0.94 / 0.94 | 616 / 581 |
| HEVC | FFmpeg | 23 | 0.83 | 0.80 / 0.84 | 613 / 506 | 0.87 | 0.81 / 0.90 | 527 / 456 |
| VVC | FFmpeg | 9 | 0.45 | 0.45 / – | 346 / 156 | 0.33 | 0.33 / – | 307 / 100 |
| VVC | VVdeC | 9 | 0.66 | 0.66 / – | 346 / 226 | 0.62 | 0.62 / – | 307 / 191 |
| AV1 | dav1d | 26 | 1.23 | 1.18 / 1.26 | 411 / 507 | 1.02 | 0.97 / 1.04 | 330 / 337 |
| VP9 | FFmpeg | 17 | 0.97 | 0.96 / 0.98 | 549 / 532 | 1.08 | 1.08 / 1.07 | 463 / 498 |
| VP9 | libvpx | 17 | 1.09 | 1.07 / 1.11 | 549 / 599 | 1.07 | 1.06 / 1.07 | 463 / 494 |
| AV2 | dav2d | 9 | 1.32 | 1.32 / – | 312 / 411 | 0.77 | 0.77 / – | 270 / 208 |
| AV2 | AVM | 9 | 0.39 | 0.39 / – | 312 / 121 | 0.29 | 0.29 / – | 270 / 79 |

Browser (node v22.21.1), ms per frame, median of 7: ours, ffmpeg.wasm, ours ÷ ffmpeg.wasm, ours browser ÷ ours native.

| format | stream | c8i ours | c8i ffmpeg.wasm | ratio | ÷ native | c8g ours | c8g ffmpeg.wasm | ratio | ÷ native |
|---|---|---|---|---|---|---|---|---|---|
| AV1 | `crowd_mot17_09.600k.av1.ivf` | 4.29 | no decoder | – | 2.61 | 5.88 | no decoder | – | 3.04 |
| AV1 | `crowd1080.3000k.ours.av1.ivf` | 10.29 | no decoder | – | 1.94 | 14.29 | no decoder | – | 2.15 |
| AV2 | `crowd_mot17_09.600k.av2.ivf` | 4.45 | no decoder | – | 2.18 | 5.67 | no decoder | – | 2.43 |
| AV2 | `crowd1080.3000k.ours.av2.ivf` | 10.85 | no decoder | – | 1.78 | 14.28 | no decoder | – | 1.96 |
| H.264 | `crowd_mot17_09.600k.h264.264` | 1.90 | 2.17 | 0.88 | 2.07 | 2.47 | 3.05 | 0.81 | 2.67 |
| H.264 | `crowd1080.3000k.x264.h264.264` | 6.04 | 8.25 | 0.73 | 1.77 | 8.00 | 10.22 | 0.78 | 2.24 |
| HEVC | `cam150.axis-m5074-720p.hevc` | 2.33 | 2.53 | 0.92 | 2.28 | 2.94 | 3.22 | 0.91 | 2.85 |
| HEVC | `crowd_mot17_09.600k.hevc.hevc` | 2.48 | 3.95 | 0.63 | 2.31 | 3.32 | 4.89 | 0.68 | 2.90 |
| HEVC | `crowd1080.3000k.ours.hevc.hevc` | 7.05 | 15.10 | 0.47 | 1.89 | 10.06 | 17.54 | 0.57 | 2.16 |
| VP9 | `crowd_mot17_09.600k.vp9.ivf` | 2.34 | 5.67 | 0.41 | 2.13 | 3.12 | 6.83 | 0.46 | 2.61 |
| VP9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 6.74 | 17.40 | 0.39 | 1.79 | 9.96 | 22.52 | 0.44 | 2.02 |
| VVC | `crowd_mot17_09.600k.vvc.vvc` | 3.75 | no decoder | – | 2.18 | 4.58 | no decoder | – | 2.60 |
| VVC | `crowd1080.3000k.ours.vvc.vvc` | 13.17 | no decoder | – | 1.96 | 16.76 | no decoder | – | 2.17 |
