# Decoder speed of release 0.7.0, x86 and Arm (2026-11-27)

The results site's Decoders page (`tools/bench/pages/decoders.html`, data built by
`tools/bench/preview_decoders.py` from this directory). Same method and harnesses as
`../decoder-speed-2026-11-23/` (decode alone on both sides), now on the benchmark's two
timing families and on the 0.7.0 release.

- **Ours (the page):** the 0.7.0 PGO release build, the binaries of the GitHub release
  assets `nanocodec-0.7.0-{x86_64,aarch64}-linux.tar.gz` (commit `ce89975ea`,
  `tools/dev/pgo_build.sh`; copies in `~/nanocodec-data/release/builds/ce89975e/`). Timed in
  the same rounds, against the same reference runs: the plain release build of that commit
  (`.../ce89975e/<arch>/plain/nanocodec`, no PGO, as crates.io builds it), as `ours-plain`.
  These are the second session on each machine (`<machine>/decoders-cpu*.json`).
- **First session (kept, not on the page):** `first-session-plain/<machine>/`: our own plain
  `cargo build --release` of the tag in the repository (version line `-dirty`: the tag's
  Cargo.lock had a stale `nc-checkasm` version, fixed on main in 3db6eec8d), with its own
  reference runs. Its ratios match the second session's plain column within a few hundredths.
- **Browser:** the first session's runs (`<machine>/yardstick.json`), the tag's
  `tools/web/build.sh` module, its relaxed-SIMD build (WASM is never a PGO build).
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
- **Files:** `<machine>/decoders-cpu{1,2,3}.json` (every pass, PGO and plain), `<machine>/yardstick.json`,
  `<machine>/wasm-yardsticks-cpu{1,2,3}.json`, `<machine>/webcodecs-cpu{1,2,3}.json` (+ `.gpu.txt`,
  chrome://gpu), `pc/webcodecs.json` (+ `.gpu.txt`), `wasm-build-flags.json` (section below),
  the platform notes, `about.json` (the page's statements).
- **Cost:** first sessions c8i 16.3 min, $0.02, c8g 21.0 min, $0.03; PGO re-time c8i 14.4 min, $0.02, c8g 16.8 min, $0.02; one aborted launch (a bad --pkg path, terminated during upload) under $0.01; all spot. Builds on the shared build box (15 min of one job slot).

## Summary

Decode time, ours ÷ reference, geometric mean over streams: the PGO release build (all / our encoder's / other encoders' streams), the plain release build timed in the same rounds, and both decoders' geometric mean Mpx/s (PGO).

| format | reference | streams | c8i PGO | c8i ours / other | c8i plain | c8i Mpx/s ours / ref | c8g PGO | c8g ours / other | c8g plain | c8g Mpx/s ours / ref |
|---|---|---|---|---|---|---|---|---|---|---|
| H.264 | FFmpeg | 18 | 1.23 | 1.24 / 1.23 | 1.16 | 599 / 739 | 0.99 | 1.00 / 0.99 | 0.95 | 585 / 580 |
| HEVC | FFmpeg | 23 | 0.87 | 0.84 / 0.89 | 0.84 | 570 / 497 | 0.89 | 0.84 / 0.94 | 0.87 | 505 / 452 |
| VVC | FFmpeg | 9 | 0.47 | 0.47 / – | 0.45 | 327 / 154 | 0.34 | 0.34 / – | 0.33 | 287 / 99 |
| VVC | VVdeC | 9 | 0.67 | 0.67 / – | 0.65 | 327 / 219 | 0.65 | 0.65 / – | 0.62 | 287 / 187 |
| AV1 | dav1d | 26 | 1.25 | 1.20 / 1.28 | 1.23 | 394 / 494 | 1.03 | 0.99 / 1.06 | 1.02 | 324 / 335 |
| VP9 | FFmpeg | 17 | 0.98 | 0.97 / 0.99 | 0.96 | 534 / 522 | 1.11 | 1.11 / 1.11 | 1.08 | 448 / 497 |
| VP9 | libvpx | 17 | 1.11 | 1.09 / 1.12 | 1.09 | 534 / 591 | 1.10 | 1.10 / 1.10 | 1.07 | 448 / 491 |
| AV2 | dav2d | 9 | 1.39 | 1.39 / – | 1.32 | 291 / 403 | 0.82 | 0.82 / – | 0.77 | 253 / 207 |
| AV2 | AVM | 9 | 0.41 | 0.41 / – | 0.39 | 291 / 119 | 0.31 | 0.31 / – | 0.29 | 253 / 78 |

First session (own plain build, own reference runs), all streams: c8i H.264/FFmpeg 1.16, HEVC/FFmpeg 0.83, VVC/FFmpeg 0.45, VVC/VVdeC 0.66, AV1/dav1d 1.23, VP9/FFmpeg 0.97, VP9/libvpx 1.09, AV2/dav2d 1.32, AV2/AVM 0.39; c8g H.264/FFmpeg 0.94, HEVC/FFmpeg 0.87, VVC/FFmpeg 0.33, VVC/VVdeC 0.62, AV1/dav1d 1.02, VP9/FFmpeg 1.08, VP9/libvpx 1.07, AV2/dav2d 0.77, AV2/AVM 0.29.

In these paired rounds the plain build took 1-6 % less time than the PGO build on every format and both machines (no cause investigated).

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

## Stronger browser yardsticks (third session, 2026-11-27)

ffmpeg.wasm 0.12.10 is built without FFmpeg's SIMD kernels, a soft yardstick. Added, on every stream
of the native table each one decodes (102 streams, 6 `pan_*` excluded as natively):

- **WebAssembly builds of the references, by us** (`tools/web/build_wasm_yardsticks.sh`, on the build
  box; emscripten 4.0.10, `-O3 -msimd128`, no threads; every flag in `wasm-build-flags.json`):
  FFmpeg n8.0.1 (h264, hevc, vvc, vp9 decoders; `--arch=wasm --enable-simd128`, so its only
  hand-written WebAssembly SIMD, HEVC's inverse transforms and SAO in `libavcodec/wasm/hevc`, is on;
  the rest is compiler-vectorised C), dav1d 1.5.4 (C, `-Denable_asm=false`: no WebAssembly kernels
  upstream; through libavcodec as natively) and libvpx v1.16.0 (`generic-gnu` C: no WebAssembly
  kernels upstream). Timed by the native harnesses (`libavcodec_bench.c`, `vpx_bench.c`) compiled
  with them, under node v22.21.1 beside our module and ffmpeg.wasm: `tools/web/wasm_yardsticks.py`,
  5 interleaved rounds, one process per run, each process the faster of two passes (fresh decoder
  each; the first warms V8's optimising tier), ffmpeg.wasm one pass; median over rounds. Our module:
  the v0.7.0 relaxed-SIMD build rebuilt from the tag (with the Cargo.lock fix), byte-identical to the
  first session's (sha256 `08777898...`). Split over vCPUs 1-3 as the decoder table.
- **The browser's own decoders (WebCodecs, native code)**: `tools/web/webcodecs_yardstick.py`, headless
  Chrome through Playwright 1.60, the whole browser bound to one vCPU with taskset; our module in the
  same browser; both the faster of two passes per fresh page, median over rounds. c8i: Google Chrome
  158.0.8127.88; c8g: Playwright's Chromium 148.0.7778.0 (Google ships no Linux Arm Chrome; this
  build has no H.264 decoder); the PC: Google Chrome 154.0.8037.97, vCPU 14 (host-timed), 3 rounds.
  Hardware decode on the PC: see "Hardware decode on the PC" below; the first PC run (before the
  VA-API driver, 3 rounds) is kept in `pc/before-vaapi-driver/`.

Ours ÷ yardstick, geometric mean of per-stream decode-time ratios (below 1: ours took less time).

| format | yardstick | streams | c8i | c8g |
|---|---|---|---|---|
| H.264 | FFmpeg n8.0.1 WASM SIMD (ours-built) | 18 | 0.67 | 0.71 |
| H.264 | ffmpeg.wasm 0.12.10 (official, no SIMD) | 18 | 0.58 | 0.60 |
| HEVC | FFmpeg n8.0.1 WASM SIMD (ours-built) | 23 | 0.63 | 0.81 |
| HEVC | ffmpeg.wasm 0.12.10 | 23 | 0.41 | 0.46 |
| VVC | FFmpeg n8.0.1 WASM SIMD (ours-built) | 9 | 0.36 | 0.43 |
| AV1 | dav1d 1.5.4 WASM SIMD (ours-built) | 26 | 0.36 | 0.45 |
| VP9 | FFmpeg n8.0.1 WASM SIMD (ours-built) | 17 | 0.37 | 0.55 |
| VP9 | libvpx v1.16.0 WASM SIMD (ours-built) | 17 | 0.24 | 0.34 |
| VP9 | ffmpeg.wasm 0.12.10 | 17 | 0.30 | 0.35 |

Ours in Chrome ÷ Chrome's own software decoder (WebCodecs, native code with assembly), same browser:

| format | streams | c8i (Chrome 158) | c8g (Chromium 148) | PC (Chrome 154) | PC hardware (NVDEC) |
|---|---|---|---|---|---|
| H.264 | 18 | 1.55 | not in this build | 1.51 | 0.55 |
| HEVC | 23 | no software decoder | no software decoder | no software decoder | 0.88 |
| AV1 | 26 | 1.62 | 1.53 | 1.39 | failed on all 26 streams |
| VP9 | 17 | 1.33 | 1.59 | 1.27 | 0.59 |
| VVC, AV2 | | no WebCodecs codec string | | | |

No run failed. Cost: c8i wasm session $0.02 (20 min; a first c8i launch was spot-reclaimed after one
round, $0.01, its partial data discarded), c8g wasm $0.03 (23 min; its WebCodecs job failed at
Playwright's install on the Ubuntu 26.04 image and was rerun with `PLAYWRIGHT_HOST_PLATFORM_OVERRIDE`),
WebCodecs reruns c8i $0.01, c8g $0.01: $0.08, all spot. Builds: ~8 job-slot minutes on the build box.
PC: 18 min of one timing core.

### Hardware decode on the PC (after nvidia-vaapi-driver 0.0.14 was installed)

- **Driver:** `/usr/bin/ffmpeg -init_hw_device vaapi` loads `nvidia_drv_video.so` ("VA-API NVDEC driver
  [direct backend]", kernel driver 610.57.04) without any environment variable; `vainfo` is not installed.
  ffmpeg's VA-API decode of our AV1, SVT-AV1 and H.264 streams ran without errors.
- **Chrome 154** (`--enable-features=AcceleratedVideoDecodeLinuxGL,VaapiOnNvidiaGPUs,VaapiIgnoreDriverChecks
  --ignore-gpu-blocklist --ozone-platform=wayland`): VA-API now initialises, and `isConfigSupported`
  with `prefer-hardware` is true for H.264, HEVC, VP9 and AV1. Headless, every hardware decode failed
  (`ChromeosStatusCode::6` while creating the decoder's image processor, all flag variants tried), so the
  timed run drew into a private headless Wayland compositor (`dbus-run-session gnome-shell --headless
  --wayland --no-x11 --virtual-monitor 1280x720`, its own socket; nothing on the desktop). One earlier
  probe opened a window on the desktop; it was closed when the owner objected.
- **Used hardware:** H.264, HEVC and VP9: every stream decoded under `prefer-hardware`, which fails
  rather than fall back to software, and Chrome's log in the probe runs shows `VaapiVideoDecoder`; `nvidia-smi dmon`
  (`pc/nvidia-smi-dmon.txt`, 2 s samples) showed decoder use in 15 of 509 samples, at most 26 %.
  **AV1: no**, every stream fails in Chrome (`vaEndPicture failed, VA error: internal decoding error`)
  although ffmpeg decodes them through the same driver.
- **Run:** 2 interleaved rounds, all streams, Chrome bound to vCPU 14 (host-timed), 16 min; ours, Chrome's
  software and hardware decoders in the same rounds (`pc/webcodecs.json`). GPU shared: 13 % busy, 4.6 GB
  used before the run (other projects and the VMAF scorer).
- **Reading the hardware column:** ours in WebAssembly on one core took 0.55 / 0.88 / 0.59 of the time
  Chrome's hardware path took for H.264 / HEVC / VP9 (Mpx/s, geometric means: ours 341 / 306 / 324,
  hardware 189 / 271 / 192). What limits the hardware path's throughput was not measured: the GPU's
  decoder was mostly idle.
