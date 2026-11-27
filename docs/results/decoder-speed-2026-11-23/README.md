# Decoder speed, decode alone on both sides (2026-11-23)

Re-measure for finding 10 of `docs/analysis/results-audit-2026-11-23.md`: the earlier records
(`decoder-speed-2026-11-19`, `wasm-yardstick-2026-11-19`) charged the reference its start-up.
Here neither side is charged process or module start-up, probing, demuxing or output.

- **Machines:** AWS c8a.large (AMD EPYC 9R45, Zen 5; one vCPU per physical core), eu-west-1, on
  demand (`tools/dev/aws_decode_speed.py`), every timed process pinned to CPU 1 and run one after
  another, our decoder on its AVX-512 tier; Jetson Orin Nano, one Cortex-A78AE core (core 3, clock
  fixed at 1.728 GHz) under the board's exclusive timing lock.
- **Method:** `tools/dev/decoder_speed.py`, 5 interleaved rounds x 3 passes per process, median of
  15 passes; the harnesses and what a pass covers are described under "Plain build" below.
- **Ours, the site's table (`x86.json`, `jetson.json`):** 0.6.0's shipped decoder: release commit
  `207745d7` plus only the `decode --bench` CLI path (1da4349e2, 3ca7f3061: `crates/nc-cli`
  `decode.rs`, `help.rs`; no decoder code differs), commit `f9483e0e` (branch
  `audit-decoders-pgo-207745d7`), built the release way: `tools/dev/pgo_build.sh` with 0.6.0's four
  public derf training clips (`release.py` `TRAINING_CLIPS`, 20 frames; 0.6.0's own profile data was
  not kept, so it was regenerated), x86_64 on the build box, aarch64 on the Jetson. Decoded frames
  equal the shipped 0.6.0 binaries' (MD5 of every frame, HEVC and AV1 720p streams, both machines).
- **Also:** a plain `--release` build of `1da4349e` (`x86-plain.json`, `jetson-plain.json`), and
  0.4.0's release commit `c9cff89f` plus the same CLI path (`fcaa3395`, PGO the same way) on x86
  (`x86-pgo-0.4.0.json`), to separate the method change from the build change.
- **Browser:** the plain browser module of `1da4349e` (`tools/web/build.sh`, `relaxed`: SIMD128
  plus relaxed SIMD); WASM is never a PGO build.

## Shipped build (0.6.0 PGO): x86, `x86.json`

One c8a.large session (9.1 min, $0.02), every reference in the same session.

| codec | stream | frames | ours ms/frame | reference | reference ms/frame | ratio |
|---|---|---|---|---|---|---|
| av1 | `aom720_good4_1000k.av1.ivf` | 40 | 2.36 (2.31–2.53) | dav1d 1.5.4 (libavcodec libdav1d) | 1.65 (1.59–1.71) | 1.43 |
| av1 | `crowd1080.3000k.ours.av1.ivf` | 150 | 3.84 (3.79–3.95) | dav1d 1.5.4 (libavcodec libdav1d) | 3.32 (3.27–3.37) | 1.16 |
| av1 | `crowd1080.3000k.svt.av1.ivf` | 150 | 2.12 (2.08–2.33) | dav1d 1.5.4 (libavcodec libdav1d) | 1.75 (1.72–1.88) | 1.21 |
| av1 | `crowd_mot17_09.600k.av1.ivf` | 150 | 1.12 (1.11–1.13) | dav1d 1.5.4 (libavcodec libdav1d) | 0.97 (0.96–0.97) | 1.16 |
| av1 | `svt720_p8_1000k.av1.ivf` | 150 | 1.48 (1.45–1.54) | dav1d 1.5.4 (libavcodec libdav1d) | 1.10 (1.08–1.12) | 1.35 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 4.83 (4.72–4.88) | AVM v1.0.0 | 12.10 (10.10–12.87) | 0.40 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 4.83 (4.72–4.88) | dav2d 237229e | 3.36 (3.34–3.42) | 1.44 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 1.48 (1.47–1.53) | AVM v1.0.0 | 3.61 (3.52–3.77) | 0.41 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 1.48 (1.47–1.53) | dav2d 237229e | 1.05 (1.04–1.06) | 1.41 |
| h264 | `crowd1080.3000k.ours.h264.264` | 150 | 2.31 (2.18–2.54) | FFmpeg n8.0.1 | 2.02 (1.96–2.09) | 1.14 |
| h264 | `crowd1080.3000k.x264.h264.264` | 150 | 2.60 (2.58–2.68) | FFmpeg n8.0.1 | 2.20 (2.18–2.21) | 1.18 |
| h264 | `crowd_mot17_09.600k.h264.264` | 150 | 0.63 (0.63–0.64) | FFmpeg n8.0.1 | 0.58 (0.58–0.58) | 1.10 |
| h264 | `openh264_cabac_1080p.3000k.264` | 150 | 2.28 (2.27–2.32) | FFmpeg n8.0.1 | 2.00 (1.99–2.03) | 1.14 |
| hevc | `cam150.axis-m5074-720p.hevc` | 150 | 0.69 (0.68–0.70) | FFmpeg n8.0.1 | 0.87 (0.87–0.88) | 0.79 |
| hevc | `cam150.axis-p1375e-1080p.hevc` | 150 | 1.67 (1.51–1.71) | FFmpeg n8.0.1 | 2.05 (1.96–2.23) | 0.82 |
| hevc | `cam150.axis-p3267lv-5mp.hevc` | 150 | 3.24 (3.11–3.80) | FFmpeg n8.0.1 | 5.07 (4.76–5.26) | 0.64 |
| hevc | `cam150.hanwha-qnvc9011r-4k.hevc` | 150 | 6.87 (6.26–7.22) | FFmpeg n8.0.1 | 8.11 (7.83–8.27) | 0.85 |
| hevc | `crowd1080.3000k.ours.hevc.hevc` | 150 | 2.78 (2.76–2.85) | FFmpeg n8.0.1 | 3.49 (3.41–3.64) | 0.80 |
| hevc | `crowd_mot17_09.600k.hevc.hevc` | 150 | 0.74 (0.73–0.75) | FFmpeg n8.0.1 | 0.94 (0.93–0.94) | 0.79 |
| hevc | `novatek-bodycam-2460x1848-1s.hevc` | 25 | 4.09 (3.76–4.38) | FFmpeg n8.0.1 | 4.62 (4.25–4.78) | 0.89 |
| hevc | `x265-1080p-b.hevc` | 150 | 3.45 (3.25–3.68) | FFmpeg n8.0.1 | 3.69 (3.52–4.01) | 0.93 |
| hevc | `x265-4k-b.hevc` | 120 | 5.74 (5.32–5.94) | FFmpeg n8.0.1 | 7.29 (6.91–7.38) | 0.79 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 2.72 (2.69–2.77) | FFmpeg n8.0.1 | 2.86 (2.84–2.88) | 0.95 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 2.72 (2.69–2.77) | libvpx v1.16.0 | 2.67 (2.63–2.71) | 1.02 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 2.63 (2.57–2.77) | FFmpeg n8.0.1 | 2.60 (2.54–2.62) | 1.01 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 2.63 (2.57–2.77) | libvpx v1.16.0 | 2.52 (2.42–2.68) | 1.04 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 0.73 (0.72–0.73) | FFmpeg n8.0.1 | 0.80 (0.79–0.80) | 0.91 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 0.73 (0.72–0.73) | libvpx v1.16.0 | 0.72 (0.72–0.77) | 1.00 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 4.95 (4.90–5.11) | FFmpeg n8.0.1 | 12.15 (10.99–12.64) | 0.41 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 4.95 (4.90–5.11) | VVdeC d261469 | 9.08 (8.65–9.44) | 0.55 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 1.21 (1.20–1.22) | FFmpeg n8.0.1 | 2.94 (2.84–3.14) | 0.41 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 1.21 (1.20–1.22) | VVdeC d261469 | 2.37 (2.18–2.44) | 0.51 |

## Shipped build (0.6.0 PGO): Jetson, `jetson.json`

| codec | stream | frames | ours ms/frame | reference | reference ms/frame | ratio |
|---|---|---|---|---|---|---|
| av1 | `aom720_good4_1000k.av1.ivf` | 40 | 8.94 (8.88–9.15) | dav1d 1.5.4 (libavcodec libdav1d) | 7.69 (7.64–7.79) | 1.16 |
| av1 | `crowd1080.3000k.ours.av1.ivf` | 150 | 14.83 (14.72–14.94) | dav1d 1.5.4 (libavcodec libdav1d) | 14.53 (14.43–14.63) | 1.02 |
| av1 | `crowd1080.3000k.svt.av1.ivf` | 150 | 10.31 (10.19–10.44) | dav1d 1.5.4 (libavcodec libdav1d) | 9.77 (9.71–9.88) | 1.06 |
| av1 | `crowd_mot17_09.600k.av1.ivf` | 150 | 4.40 (4.39–4.42) | dav1d 1.5.4 (libavcodec libdav1d) | 4.67 (4.65–4.70) | 0.94 |
| av1 | `svt720_p8_1000k.av1.ivf` | 150 | 5.76 (5.73–5.83) | dav1d 1.5.4 (libavcodec libdav1d) | 5.22 (5.18–5.27) | 1.10 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 17.39 (17.34–17.48) | AVM v1.0.0 | 61.69 (61.43–61.99) | 0.28 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 17.39 (17.34–17.48) | dav2d 237229e | 20.16 (20.07–20.24) | 0.86 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 5.46 (5.44–5.64) | AVM v1.0.0 | 19.83 (19.72–19.95) | 0.28 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 5.46 (5.44–5.64) | dav2d 237229e | 7.77 (7.74–7.80) | 0.70 |
| h264 | `crowd1080.3000k.ours.h264.264` | 150 | 6.55 (6.39–6.59) | FFmpeg n8.0.1 | 7.14 (7.07–7.19) | 0.92 |
| h264 | `crowd1080.3000k.x264.h264.264` | 150 | 7.53 (7.46–7.58) | FFmpeg n8.0.1 | 7.69 (7.61–7.82) | 0.98 |
| h264 | `crowd_mot17_09.600k.h264.264` | 150 | 1.93 (1.93–1.94) | FFmpeg n8.0.1 | 2.06 (2.05–2.07) | 0.94 |
| h264 | `openh264_cabac_1080p.3000k.264` | 150 | 6.77 (6.69–6.87) | FFmpeg n8.0.1 | 7.05 (6.98–7.13) | 0.96 |
| hevc | `cam150.axis-m5074-720p.hevc` | 150 | 2.17 (2.16–2.22) | FFmpeg n8.0.1 | 2.59 (2.58–2.62) | 0.84 |
| hevc | `cam150.axis-p1375e-1080p.hevc` | 150 | 5.21 (5.18–5.24) | FFmpeg n8.0.1 | 6.21 (6.19–6.22) | 0.84 |
| hevc | `cam150.axis-p3267lv-5mp.hevc` | 150 | 11.51 (11.49–11.53) | FFmpeg n8.0.1 | 13.77 (13.72–13.85) | 0.84 |
| hevc | `cam150.hanwha-qnvc9011r-4k.hevc` | 150 | 19.96 (19.82–20.09) | FFmpeg n8.0.1 | 23.72 (23.63–23.79) | 0.84 |
| hevc | `crowd1080.3000k.ours.hevc.hevc` | 150 | 11.20 (11.14–11.27) | FFmpeg n8.0.1 | 13.18 (13.10–13.26) | 0.85 |
| hevc | `crowd_mot17_09.600k.hevc.hevc` | 150 | 2.70 (2.69–2.73) | FFmpeg n8.0.1 | 3.35 (3.34–3.43) | 0.81 |
| hevc | `novatek-bodycam-2460x1848-1s.hevc` | 25 | 16.27 (16.21–16.38) | FFmpeg n8.0.1 | 17.06 (16.89–17.16) | 0.95 |
| hevc | `x265-1080p-b.hevc` | 150 | 14.95 (14.87–15.14) | FFmpeg n8.0.1 | 15.43 (15.37–15.49) | 0.97 |
| hevc | `x265-4k-b.hevc` | 120 | 22.96 (22.79–23.14) | FFmpeg n8.0.1 | 25.82 (25.52–26.13) | 0.89 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 10.70 (10.59–10.76) | FFmpeg n8.0.1 | 10.06 (10.04–10.12) | 1.06 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 10.70 (10.59–10.76) | libvpx v1.16.0 | 10.58 (10.52–10.65) | 1.01 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 11.95 (11.88–12.05) | FFmpeg n8.0.1 | 11.13 (11.11–11.17) | 1.07 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 11.95 (11.88–12.05) | libvpx v1.16.0 | 11.65 (11.55–11.71) | 1.03 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 2.75 (2.74–2.77) | FFmpeg n8.0.1 | 2.63 (2.62–2.64) | 1.05 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 2.75 (2.74–2.77) | libvpx v1.16.0 | 2.71 (2.71–2.73) | 1.01 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 18.53 (18.44–18.57) | FFmpeg n8.0.1 | 52.98 (52.87–53.45) | 0.35 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 18.53 (18.44–18.57) | VVdeC d261469 | 31.93 (31.52–32.14) | 0.58 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 3.92 (3.92–3.94) | FFmpeg n8.0.1 | 11.83 (11.77–11.89) | 0.33 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 3.92 (3.92–3.94) | VVdeC d261469 | 7.91 (7.88–7.95) | 0.50 |

## Method or build: x86 ratios before and after

Ours ÷ reference, x86 (c8a.large). 2026-11-19: the published table (0.4.0 PGO `c9cff89f`, best
whole-process wall time of 7 rounds, FFmpeg writing raw video through a muxer, the reference
CLIs vpxdec, VVdeC and avmdec). The other two columns: in process, median of 15, the 0.4.0 PGO
build (+ the bench CLI) and the 0.6.0 PGO build (+ the bench CLI), each in its own session.

| codec | stream | reference | 2026-11-19 published | 0.4.0 PGO in process | 0.6.0 PGO in process |
|---|---|---|---|---|---|
| av1 | `aom720_good4_1000k.av1.ivf` | dav1d | 1.56 | 1.59 | 1.43 |
| hevc | `cam150.axis-m5074-720p.hevc` | ffmpeg | 0.69 | 0.81 | 0.79 |
| hevc | `cam150.axis-p1375e-1080p.hevc` | ffmpeg | 0.64 | 0.78 | 0.82 |
| hevc | `cam150.axis-p3267lv-5mp.hevc` | ffmpeg | 0.54 | 0.66 | 0.64 |
| hevc | `cam150.hanwha-qnvc9011r-4k.hevc` | ffmpeg | 0.68 | 0.81 | 0.85 |
| av1 | `crowd1080.3000k.ours.av1.ivf` | dav1d | 1.15 | 1.14 | 1.16 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | avmdec | 0.47 | 0.46 | 0.40 |
| h264 | `crowd1080.3000k.ours.h264.264` | ffmpeg | 0.94 | 1.06 | 1.14 |
| hevc | `crowd1080.3000k.ours.hevc.hevc` | ffmpeg | 0.76 | 0.83 | 0.80 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | vpxdec | 1.05 | 1.05 | 1.02 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | vvdec | 0.57 | 0.58 | 0.55 |
| av1 | `crowd1080.3000k.svt.av1.ivf` | dav1d | 1.23 | 1.25 | 1.21 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | vpxdec | 1.09 | 1.08 | 1.04 |
| h264 | `crowd1080.3000k.x264.h264.264` | ffmpeg | 1.03 | 1.14 | 1.18 |
| av1 | `crowd_mot17_09.600k.av1.ivf` | dav1d | 1.17 | 1.16 | 1.16 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | avmdec | 0.43 | 0.42 | 0.41 |
| h264 | `crowd_mot17_09.600k.h264.264` | ffmpeg | 0.90 | 1.06 | 1.10 |
| hevc | `crowd_mot17_09.600k.hevc.hevc` | ffmpeg | 0.73 | 0.81 | 0.79 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | vpxdec | 1.03 | 1.01 | 1.00 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | vvdec | 0.57 | 0.57 | 0.51 |
| hevc | `novatek-bodycam-2460x1848-1s.hevc` | ffmpeg | 0.70 | 0.91 | 0.89 |
| h264 | `openh264_cabac_1080p.3000k.264` | ffmpeg | 0.98 | 1.11 | 1.14 |
| av1 | `svt720_p8_1000k.av1.ivf` | dav1d | 1.38 | 1.41 | 1.35 |
| hevc | `x265-1080p-b.hevc` | ffmpeg | 0.86 | 0.94 | 0.93 |
| hevc | `x265-4k-b.hevc` | ffmpeg | 0.60 | 0.76 | 0.79 |

## Plain build (`1da4349e`): x86, `x86-plain.json`

`tools/dev/decoder_speed.py`: 5 interleaved rounds (order reversed every other round), each
process decoding its stream 3 times: 15 timed passes per decoder and stream. x86: a third
c8a.large session (9.0 min, $0.02), same binary; it replaced the first session's x86.json
(which had only FFmpeg and dav1d; its raw output is kept with the streams).

- Ours: `nanocodec decode --bench 3`: the stream read into memory, then per pass a fresh
  decoder over it, frames discarded. A pass includes the decoder's construction and the
  container framing over memory (IVF headers, Annex-B start-code scan).
- FFmpeg / dav1d: `libavcodec-bench --repeat 3` (`tools/anchors/libavcodec_bench.c`) against the
  pinned static FFmpeg n8.0.1 libraries (gcc 15.2, `-O3`, x86 assembly on: AVX2 and AVX-512
  kernels where FFmpeg has them; `tools/anchors/build_anchors.sh avbench`). Packets demuxed and
  parsed by libavformat before timing; a pass is avcodec_send_packet / avcodec_receive_frame
  over every packet plus the flush, one thread, frames unreferenced. Decoder open is not timed.
  AV1 goes to dav1d 1.5.4 through libavcodec's libdav1d wrapper.
- libvpx v1.16.0 (vpxdec's library), VVdeC d261469 and AVM v1.0.0 (avmdec's library):
  `vpx-bench`, `vvdec-bench`, `avm-bench` (`tools/anchors/{vpx,vvdec,avm}_bench.c`,
  `decode_bench.h`; `build_anchors.sh decbench`), each against its pinned static library as its
  own decoder application is built. IVF frames or Annex-B NAL units (VVdeC takes one NAL unit
  per call, as vvdecapp feeds it) split in memory before timing; a pass is the library's decode
  call over every unit plus its flush, one thread, frames not copied. Decoder open not timed.
- dav2d 237229e (AV2): `dav2d-bench` (`tools/anchors/dav2d_bench.c`) against the pinned dav2d
  built with its assembly (`build_anchors.sh dav2dasm`; the validated conformance oracle `dav2d`
  is built without it). On both AV2 streams its output MD5 equals the oracle's, on x86 and on the
  Jetson. One thread, frame delay 1 (as `dav2d --threads 1 --framedelay 1`).
- AV2 rows (ours, AVM, dav2d) come from a later c8a.large session of the same day (2.5 min,
  $0.01); an earlier one timed the scalar dav2d build (raw output kept, not used).

ms per frame: median of the 15 passes (min–max); ratio = ours median / reference median.

| codec | stream | frames | ours ms/frame | reference | reference ms/frame | ratio |
|---|---|---|---|---|---|---|
| av1 | `aom720_good4_1000k.av1.ivf` | 40 | 2.30 (2.28–2.32) | dav1d 1.5.4 (libavcodec libdav1d) | 1.60 (1.59–1.63) | 1.43 |
| av1 | `crowd1080.3000k.ours.av1.ivf` | 150 | 3.88 (3.87–3.90) | dav1d 1.5.4 (libavcodec libdav1d) | 3.33 (3.32–3.34) | 1.17 |
| av1 | `crowd1080.3000k.svt.av1.ivf` | 150 | 2.06 (2.05–2.09) | dav1d 1.5.4 (libavcodec libdav1d) | 1.73 (1.71–1.80) | 1.20 |
| av1 | `crowd_mot17_09.600k.av1.ivf` | 150 | 1.14 (1.13–1.15) | dav1d 1.5.4 (libavcodec libdav1d) | 0.97 (0.96–0.97) | 1.18 |
| av1 | `svt720_p8_1000k.av1.ivf` | 150 | 1.47 (1.46–1.51) | dav1d 1.5.4 (libavcodec libdav1d) | 1.08 (1.08–1.09) | 1.35 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 4.77 (4.75–4.79) | AVM v1.0.0 | 10.38 (10.17–11.33) | 0.46 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 4.77 (4.75–4.79) | dav2d 237229e | 3.34 (3.33–3.35) | 1.43 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 1.51 (1.50–1.56) | AVM v1.0.0 | 3.55 (3.49–3.65) | 0.43 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 1.51 (1.50–1.56) | dav2d 237229e | 1.04 (1.04–1.05) | 1.45 |
| h264 | `crowd1080.3000k.ours.h264.264` | 150 | 2.12 (2.11–2.19) | FFmpeg n8.0.1 | 1.96 (1.96–1.98) | 1.08 |
| h264 | `crowd1080.3000k.x264.h264.264` | 150 | 2.52 (2.51–2.53) | FFmpeg n8.0.1 | 2.19 (2.18–2.21) | 1.15 |
| h264 | `crowd_mot17_09.600k.h264.264` | 150 | 0.63 (0.63–0.64) | FFmpeg n8.0.1 | 0.58 (0.57–0.59) | 1.09 |
| h264 | `openh264_cabac_1080p.3000k.264` | 150 | 2.26 (2.25–2.27) | FFmpeg n8.0.1 | 1.99 (1.98–2.00) | 1.13 |
| hevc | `cam150.axis-m5074-720p.hevc` | 150 | 0.71 (0.70–0.71) | FFmpeg n8.0.1 | 0.87 (0.87–0.87) | 0.82 |
| hevc | `cam150.axis-p1375e-1080p.hevc` | 150 | 1.56 (1.55–1.58) | FFmpeg n8.0.1 | 1.95 (1.94–1.97) | 0.80 |
| hevc | `cam150.axis-p3267lv-5mp.hevc` | 150 | 3.26 (3.20–3.50) | FFmpeg n8.0.1 | 4.79 (4.74–4.81) | 0.68 |
| hevc | `cam150.hanwha-qnvc9011r-4k.hevc` | 150 | 6.71 (6.52–6.87) | FFmpeg n8.0.1 | 7.85 (7.78–7.92) | 0.85 |
| hevc | `crowd1080.3000k.ours.hevc.hevc` | 150 | 2.89 (2.88–2.91) | FFmpeg n8.0.1 | 3.42 (3.40–3.48) | 0.85 |
| hevc | `crowd_mot17_09.600k.hevc.hevc` | 150 | 0.77 (0.77–0.78) | FFmpeg n8.0.1 | 0.93 (0.93–0.97) | 0.83 |
| hevc | `novatek-bodycam-2460x1848-1s.hevc` | 25 | 3.97 (3.94–4.10) | FFmpeg n8.0.1 | 4.25 (4.18–4.33) | 0.93 |
| hevc | `x265-1080p-b.hevc` | 150 | 3.13 (3.11–3.27) | FFmpeg n8.0.1 | 3.52 (3.49–3.63) | 0.89 |
| hevc | `x265-4k-b.hevc` | 120 | 5.29 (5.23–5.69) | FFmpeg n8.0.1 | 6.93 (6.90–7.15) | 0.76 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 2.79 (2.78–2.80) | FFmpeg n8.0.1 | 2.84 (2.83–2.85) | 0.98 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 2.79 (2.78–2.80) | libvpx v1.16.0 | 2.65 (2.64–2.67) | 1.05 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 2.64 (2.62–2.67) | FFmpeg n8.0.1 | 2.55 (2.54–2.56) | 1.03 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 2.64 (2.62–2.67) | libvpx v1.16.0 | 2.43 (2.42–2.55) | 1.09 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 0.73 (0.73–0.74) | FFmpeg n8.0.1 | 0.79 (0.79–0.79) | 0.93 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 0.73 (0.73–0.74) | libvpx v1.16.0 | 0.72 (0.72–0.73) | 1.02 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 5.11 (5.09–5.13) | FFmpeg n8.0.1 | 11.11 (10.96–11.29) | 0.46 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 5.11 (5.09–5.13) | VVdeC d261469 | 8.82 (8.61–8.94) | 0.58 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 1.29 (1.28–1.29) | FFmpeg n8.0.1 | 2.85 (2.82–2.88) | 0.45 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 1.29 (1.28–1.29) | VVdeC d261469 | 2.23 (2.17–2.37) | 0.58 |

## Plain build (`1da4349e`): Jetson, `jetson-plain.json`

Same method and harnesses, built for aarch64 on the board (`build_anchors.sh avbench decbench`):
Jetson Orin Nano, one Cortex-A78AE core (core 3, clock fixed at 1.728 GHz), under the board's
exclusive timing lock (`jetson-timed.sh taskset -c 3`), 5 interleaved rounds x 3 passes, median
of 15. Ours: the board's release build of `1da4349e` (`nceval.jetson.ensure_build`). FFmpeg
n8.0.1 with its NEON assembly; libvpx v1.16.0; VVdeC d261469; AVM v1.0.0 and dav2d 237229e with
its assembly (both built on the board for this, `build_anchors.sh avm dav2dasm decbench`). The AV2
rows come from a second run under the lock the same day (5 rounds x 3 passes, clock again 1.728 GHz).

| codec | stream | frames | ours ms/frame | reference | reference ms/frame | ratio |
|---|---|---|---|---|---|---|
| av1 | `aom720_good4_1000k.av1.ivf` | 40 | 9.16 (9.04–9.36) | dav1d 1.5.4 (libavcodec libdav1d) | 7.70 (7.66–7.79) | 1.19 |
| av1 | `crowd1080.3000k.ours.av1.ivf` | 150 | 15.52 (15.36–15.61) | dav1d 1.5.4 (libavcodec libdav1d) | 14.55 (14.51–14.62) | 1.07 |
| av1 | `crowd1080.3000k.svt.av1.ivf` | 150 | 10.83 (10.80–10.94) | dav1d 1.5.4 (libavcodec libdav1d) | 9.75 (9.72–9.77) | 1.11 |
| av1 | `crowd_mot17_09.600k.av1.ivf` | 150 | 4.56 (4.55–4.59) | dav1d 1.5.4 (libavcodec libdav1d) | 4.67 (4.66–4.70) | 0.98 |
| av1 | `svt720_p8_1000k.av1.ivf` | 150 | 6.00 (5.95–6.08) | dav1d 1.5.4 (libavcodec libdav1d) | 5.21 (5.19–5.28) | 1.15 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 18.54 (18.49–18.61) | AVM v1.0.0 | 61.86 (61.37–62.07) | 0.30 |
| av2 | `crowd1080.3000k.ours.av2.ivf` | 150 | 18.54 (18.49–18.61) | dav2d 237229e | 20.16 (20.09–20.30) | 0.92 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 5.61 (5.60–5.81) | AVM v1.0.0 | 19.84 (19.67–19.93) | 0.28 |
| av2 | `crowd_mot17_09.600k.av2.ivf` | 150 | 5.61 (5.60–5.81) | dav2d 237229e | 7.78 (7.75–7.81) | 0.72 |
| h264 | `crowd1080.3000k.ours.h264.264` | 150 | 6.87 (6.78–6.90) | FFmpeg n8.0.1 | 7.13 (7.07–7.18) | 0.96 |
| h264 | `crowd1080.3000k.x264.h264.264` | 150 | 7.82 (7.68–7.87) | FFmpeg n8.0.1 | 7.66 (7.61–7.72) | 1.02 |
| h264 | `crowd_mot17_09.600k.h264.264` | 150 | 2.03 (2.02–2.05) | FFmpeg n8.0.1 | 2.07 (2.06–2.08) | 0.98 |
| h264 | `openh264_cabac_1080p.3000k.264` | 150 | 7.14 (7.02–7.22) | FFmpeg n8.0.1 | 7.04 (6.98–7.06) | 1.01 |
| hevc | `cam150.axis-m5074-720p.hevc` | 150 | 2.33 (2.31–2.37) | FFmpeg n8.0.1 | 2.59 (2.58–2.64) | 0.90 |
| hevc | `cam150.axis-p1375e-1080p.hevc` | 150 | 5.53 (5.51–5.55) | FFmpeg n8.0.1 | 6.21 (6.19–6.22) | 0.89 |
| hevc | `cam150.axis-p3267lv-5mp.hevc` | 150 | 12.26 (12.23–12.34) | FFmpeg n8.0.1 | 13.76 (13.73–13.78) | 0.89 |
| hevc | `cam150.hanwha-qnvc9011r-4k.hevc` | 150 | 20.59 (20.50–20.71) | FFmpeg n8.0.1 | 23.73 (23.65–23.86) | 0.87 |
| hevc | `crowd1080.3000k.ours.hevc.hevc` | 150 | 11.34 (11.22–11.36) | FFmpeg n8.0.1 | 13.19 (13.12–13.37) | 0.86 |
| hevc | `crowd_mot17_09.600k.hevc.hevc` | 150 | 2.71 (2.70–2.75) | FFmpeg n8.0.1 | 3.36 (3.35–3.38) | 0.81 |
| hevc | `novatek-bodycam-2460x1848-1s.hevc` | 25 | 16.32 (16.18–16.43) | FFmpeg n8.0.1 | 17.16 (17.06–17.45) | 0.95 |
| hevc | `x265-1080p-b.hevc` | 150 | 14.76 (14.72–14.89) | FFmpeg n8.0.1 | 15.47 (15.41–15.83) | 0.95 |
| hevc | `x265-4k-b.hevc` | 120 | 23.09 (22.85–23.35) | FFmpeg n8.0.1 | 25.88 (25.61–26.27) | 0.89 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 10.87 (10.80–10.99) | FFmpeg n8.0.1 | 10.07 (10.05–10.12) | 1.08 |
| vp9 | `crowd1080.3000k.ours.vp9.ivf` | 150 | 10.87 (10.80–10.99) | libvpx v1.16.0 | 10.60 (10.54–10.70) | 1.03 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 12.24 (12.10–12.32) | FFmpeg n8.0.1 | 11.14 (11.10–11.17) | 1.10 |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 150 | 12.24 (12.10–12.32) | libvpx v1.16.0 | 11.68 (11.64–11.70) | 1.05 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 2.83 (2.82–2.85) | FFmpeg n8.0.1 | 2.63 (2.62–2.64) | 1.08 |
| vp9 | `crowd_mot17_09.600k.vp9.ivf` | 150 | 2.83 (2.82–2.85) | libvpx v1.16.0 | 2.72 (2.71–2.73) | 1.04 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 18.57 (18.53–18.63) | FFmpeg n8.0.1 | 52.96 (52.71–53.51) | 0.35 |
| vvc | `crowd1080.3000k.ours.vvc.vvc` | 150 | 18.57 (18.53–18.63) | VVdeC d261469 | 31.93 (31.65–32.07) | 0.58 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 4.10 (4.09–4.12) | FFmpeg n8.0.1 | 11.86 (11.77–11.91) | 0.35 |
| vvc | `crowd_mot17_09.600k.vvc.vvc` | 150 | 4.10 (4.09–4.12) | VVdeC d261469 | 7.93 (7.90–7.99) | 0.52 |

## Browser: node, `../wasm-yardstick-2026-11-23/wasm.json`

`tools/web/ffmpeg_wasm_yardstick.py` under node v22.21.1 (V8) on the same machine type and core
(a second c8a.large session, 2026-11-23), 7 interleaved rounds, one process per run, one
decode or encode per process: 7 values per side. Both sides are timed on the codec calls
alone: neither is charged module start-up, probing, demuxing or command parsing.

- Ours: the `BrowserDecoder.decode` calls (`decode_ms_per_frame`; the display hand-off,
  I420 to RGBA, is reported beside it and not included; includes copying each packet into
  the module's memory) and the `BrowserEncoder.encode` calls (`encode_ms_per_frame`;
  loading each frame into a `SourceFrame` is reported beside it, in `ms_per_frame`).
- ffmpeg.wasm: `@ffmpeg/core` 0.12.10 (latest on npm on 2026-11-23; no SIMD
  variant is published: `@ffmpeg/core-mt` is the multi-thread build), FFmpeg 5.1.4 and x264
  under emscripten with `--disable-asm` and `-O3 -msimd128`: compiler-generated SIMD128 only,
  FFmpeg's and x264's hand-written SIMD kernels are off. One thread. Time = the sum of its
  `decode_video` / `encode_video` calls as FFmpeg's `-benchmark_all` logs them.
- Start-up (module instantiation) and the whole-run measure the 2026-11-19 record used
  (start-up plus the whole `exec`, here also carrying `-benchmark_all`'s logging) are
  recorded beside.

ms per frame: median of 7 (min–max); ratio = ours median / ffmpeg.wasm median.

### Decode

| codec | stream | ours ms/frame | ffmpeg.wasm decode ms/frame | ratio | ffmpeg.wasm start-up ms | ffmpeg.wasm whole run ms/frame |
|---|---|---|---|---|---|---|
| hevc | `axis720.cam.hevc` | 1.80 (1.78–1.80) | 2.01 (2.00–2.10) | 0.89 | 114 (113–116) | 3.55 (3.50–3.67) |
| h264 | `crowd_mot17_09.600k.h264.264` | 1.49 (1.47–1.50) | 1.80 (1.78–1.81) | 0.83 | 114 (113–115) | 3.50 (3.47–3.56) |
| hevc | `crowd_mot17_09.600k.hevc.hevc` | 1.89 (1.84–1.92) | 3.06 (3.03–3.09) | 0.62 | 113 (113–115) | 4.56 (4.53–4.61) |
| h264 | `crowd1080.3000k.x264.h264.264` | 5.09 (5.04–5.43) | 6.79 (6.74–6.88) | 0.75 | 114 (114–116) | 8.75 (8.68–8.84) |
| hevc | `crowd1080.3000k.ours.hevc.hevc` | 5.63 (5.59–5.75) | 11.35 (11.32–11.48) | 0.50 | 115 (113–117) | 13.10 (13.08–13.29) |
| vp9 | `crowd1080.3000k.vpxenc.vp9.ivf` | 5.05 (4.93–5.20) | 13.31 (13.26–13.55) | 0.38 | 115 (113–116) | 15.32 (15.26–15.59) |

### Encode

crowd_mot17_09, 1280x720, 150 frames, 30 fps. Both sides have the same rate
limits (`encode_settings` in the record): target 600 kbps, peak
1200 kbps, buffer 600 kbit (1 s at the target). Ours:
`set_rate_control(600, 1200, 1000): target kbps, peak kbps, buffer ms`. x264: `-b:v 600k -maxrate 1200k -bufsize 600k -tune zerolatency -bf 0 -rc-lookahead 0`.

| run | encoder | encode calls ms/frame | whole (ours: with frame loading; ffmpeg.wasm: start-up and exec) ms/frame | achieved kbps (median) |
|---|---|---|---|---|
| h264 | ours | 9.71 (9.61–9.76) | 9.81 (9.70–9.88) | 586 |
| h264-fast | ours | 7.18 (7.15–7.28) | 7.27 (7.24–7.37) | 587 |
| hevc-fast | ours | 9.43 (9.33–9.59) | 9.51 (9.44–9.68) | 586 |
| vp9 | ours | 14.15 (14.07–14.48) | 14.24 (14.16–14.58) | 590 |
| av1-fast | ours | 9.60 (9.53–9.71) | 9.71 (9.63–9.80) | 591 |
| x264 veryfast | ffmpeg.wasm x264 | 30.12 (29.96–31.57) | 31.14 (31.01–32.65) | 527 |
| x264 ultrafast | ffmpeg.wasm x264 | 11.94 (11.80–12.59) | 13.03 (12.89–13.76) | 558 |
