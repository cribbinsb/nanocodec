# Speed claims against independent yardsticks

<sub>Measured 2026-11-12. Native build abcf696b (variance-integration, plain release); browser module
bcbc5601 relaxed SIMD (the build behind the README's WASM table).</sub>

Until now, our decoder speed and our WebAssembly figures were compared only with our own native build. This page
lists every speed or efficiency claim in [README.md](../../README.md) and [docs/results](README.md), names the
yardstick each one uses, and measures each self-only one against the best independent implementation we could run.

## Method

- **Machine.** Ryzen 9 5950X. Every timed process was pinned to timing core 15 under `tools/dev/host-timed.sh`.
  The PC was shared with other agents (1-minute load 23–42 on the other cores during the runs). Runs were
  interleaved: each round ran every command once, with the order rotated between rounds. The tables show the best
  of 3 rounds (5 for start-up, 2 for WASM). Wall time is process wall time. Peak RSS comes from `wait4`.
- **Single thread everywhere:** `ffmpeg -threads 1`, `dav1d --threads 1 --framedelay 1`, `vpxdec --threads=1`,
  `vvdecapp -t 1`, `avmdec --threads=1`, `dav2d --threads 1`, `x264 --threads 1`, SVT `--lp 1`, and so on. The
  binaries are the pinned anchors in `~/nanocodec-data/anchors`. FFmpeg n8.0.1 writes rawvideo to `/dev/null`,
  which is slightly pessimistic for FFmpeg.
- **Streams.**
  - Ours: the README's WASM streams (crowd_mot17_09 and pan_jaad_0117 at 720p and 600 kbps), plus new 1080p
    encodes at 3 Mbit/s of crowd_pp22_189_1080p (source 6.3 Mbit/s), one per codec.
  - Third-party:
    - x264 `veryfast` with no B-frames at 1080p and 3 Mbit/s;
    - SVT-AV1 M10 at 1080p and 3 Mbit/s;
    - libvpx VP9 `--good --cpu-used=4` at 1080p and 3 Mbit/s;
    - two real ONVIF camera recordings from `~/Downloads/hevc-real-clips` (Axis M5074 720p with 901 frames,
      Axis P1375-E 1080p with 754 frames), extracted with `-c copy`.

  Our decoders were checked first on every third-party stream: each output is MD5-identical to FFmpeg's.
- **WASM yardstick.** We used `@ffmpeg/core` 0.12.10, the official single-thread ffmpeg.wasm, under the same node
  22 and the same pinned core. It is a generic C build with no hand-written SIMD, and it has no AV1 or VVC decoder,
  so it is a weak yardstick. The rows marked **est.** take a competitor's measured native time and multiply it by
  the WASM/native ratio that we ourselves pay (README: 1.36x for H.264/HEVC decode, about 2x for H.264 encode,
  2.8x for AV1 Fast and 2.5x for VP9 encode). They estimate what a SIMD WASM port of that competitor would reach.
  They are not measurements.
- **Encode.** crowd_mot17_09 at 720p, 150 frames, CBR 600 kbps with a 1 s buffer. Each competitor ran at its
  chart command line (zero latency, no B-frames, no lookahead). "Start-up" is the wall time of a process that
  encodes one frame: process start, initialisation and the first key frame. This is the time to the first packet.
  "Per frame" is (150-frame wall − start-up) / 149.

## 1. Claims and their yardsticks

| # | Claim (where) | Yardstick until today |
|---|---|---|
| 1 | Native encode speed: fps per preset on Jetson and x86, "faster than SVT M12/M13", "close to openh264's speed" (README charts and presets; bd-vs-speed.md) | **Independent**: x264, x265, SVT-AV1, libaom, libvpx, openh264, NVENC, timed in the same rounds |
| 2 | "WASM decode is fast: on average only 1.4x the native decode time"; WASM decode fps table (README, results README, wasm-speed.md) | **Ourselves only** (our native build) |
| 3 | WASM encode fps and "1.7–3.0x to encode" (same places) | **Ourselves only** |
| 4 | Native decode speed of H.264, HEVC, AV1 and VP9 (implied by "decode anywhere"; no figure published) | **Ourselves only**. Now known to be 1.3–2.2x slower than FFmpeg/dav1d; fixes in progress elsewhere |
| 5 | VVC and AV2 decoders (WASM/native ratio only) | **Ourselves only** |
| 6 | Demo display hand-off "about 0.7 ms per 720p frame", WebGL 0.15–0.3 ms (results README, wasm-perf handoff) | **Ourselves only** |
| 7 | Live PSNR "about 2 ms per 720p frame" (web/README) | **Ourselves only** (not re-measured here; the RGBA copy dominates it) |
| 8 | "Low delay" (README intro) | Throughput only (claim 1). Start-up and per-frame latency were **never measured** |
| 9 | Memory use | **No claim and never measured** |
| 10 | CLI and C API | No speed claim. The CLI is what claim 1 times. The C API wraps the same encoder; its overhead was not measured (not done here either) |
| 11 | Headless Chromium cross-check of WASM decode (wasm-speed.md) | **Ourselves only** (node vs Chromium, same module) |

## 2. Measurements against independent implementations

Ratio = ours ÷ theirs in time (> 1 means we are slower). Times are ms per frame unless marked otherwise.

| Claim | Yardstick | Stream / setting | Ours | Theirs | Ratio |
|---|---|---|---:|---:|---:|
| 4 native decode H.264 | FFmpeg | ours 720p (2 clips) | 1.30–1.33 | 1.12–1.19 | **1.12–1.16** |
| 4 native decode H.264 | FFmpeg | ours 1080p 3M / x264 1080p 3M | 5.51 / 5.95 | 4.65 / 4.31 | **1.19 / 1.38** |
| 4 native decode H.264 | openh264 | x264 1080p 3M | 5.95 | 9.01 | 0.66 |
| 4 native decode HEVC | FFmpeg | ours 720p (2 clips) | 1.91–2.11 | 1.69–1.79 | **1.14–1.18** |
| 4 native decode HEVC | FFmpeg | ours 1080p 3M | 9.55 | 7.51 | **1.27** |
| 4 native decode HEVC | FFmpeg | Axis 720p camera / Axis 1080p camera | 1.55 / 6.47 | 1.57 / 5.58 | 0.99 / **1.16** |
| 4 native decode AV1 | dav1d | ours 720p (2 clips) | 2.61–3.17 | 1.67–2.09 | **1.52–1.57** |
| 4 native decode AV1 | dav1d | ours 1080p 3M / SVT 1080p 3M | 12.4 / 12.3 | 5.35 / 4.40 | **2.31 / 2.80** |
| 4 native decode VP9 | vpxdec | ours 720p, ours 1080p, vpxenc 1080p | 2.25–9.93 | 1.03–4.93 | **2.01–2.19** |
| 4 native decode VP9 | FFmpeg | same | | | **1.43–1.66** |
| 5 native decode VVC | VVdeC `-t 1` | ours 720p (2) / ours 1080p 3M | 3.50–3.69 / 14.2 | 7.21–9.13 / 28.6 | 0.40–0.50 |
| 5 native decode VVC | FFmpeg native vvc | same | | | 0.51–0.53 |
| 5 native decode AV2 | avmdec (reference) | ours 720p (2) / ours 1080p 3M | 2.77–2.82 / 10.2 | 6.47–6.76 / 25.5 | 0.40–0.44 |
| 5 native decode AV2 | dav2d (early) | same | | | 0.28–0.42 |
| 2 WASM decode H.264 | ffmpeg.wasm (no SIMD) | ours 720p / x264 1080p 3M | 2.57 / 9.40 | 3.65 / 11.1 | 0.70 / 0.85 |
| 2 WASM decode H.264 | **est.** FFmpeg × 1.36 | same | 2.57 / 9.40 | 1.52 / 5.86 | **1.69 / 1.60** |
| 2 WASM decode HEVC | ffmpeg.wasm (no SIMD) | Axis 720p camera / ours 720p / ours 1080p | 2.35 / 3.66 / 14.1 | 2.67 / 5.10 / 18.5 | 0.88 / 0.72 / 0.76 |
| 2 WASM decode HEVC | **est.** FFmpeg × 1.36 | same | | 2.13 / 2.29 / 10.2 | **1.10 / 1.60 / 1.38** |
| 2 WASM decode VP9 | ffmpeg.wasm (no SIMD) | vpxenc 1080p 3M | 14.9 | 21.2 | 0.70 |
| 2 WASM decode VP9 | **est.** vpxdec × 1.36 | same | 14.9 | 6.71 | **2.21** |
| 6 display I420→RGBA, WASM (demo CPU path, incl. copy out) | libyuv `I420ToABGR`, native AVX2 | 720p / 1080p | 0.84–1.02 / 2.74–3.65 | 0.196 / 0.59 | **4.3–6.2** (cross-platform); fixed, see 3.1 |
| 6 display I420→RGBA, native | libyuv `I420ToABGR`, AVX2 | 360p / 720p / 1080p | 0.84 / 3.36 / 7.63 | 0.047 / 0.196 / 0.59 | **13–18** (our AVX2 and NEON tables use the scalar row); fixed, see 3.1 |
| 3 WASM encode H.264 | ffmpeg.wasm x264 veryfast / ultrafast (no SIMD) | 720p 600k | 11.6 (Default) / 11.0 (Fast) | 46.5 / 18.6 | 0.25 / 0.59 |
| 3 WASM encode H.264 | **est.** x264 veryfast × 2.0 / openh264 × 2.0 | 720p 600k | 11.6 / 11.0 | 15.1 / 7.7 | 0.77 / **1.43** |
| 3 WASM encode AV1 Fast | **est.** SVT M12 rtc × 2.8 | 720p 600k | 22.6 | 19.1 | **1.19** |
| 3 WASM encode VP9 | **est.** libvpx rt 9 × 2.5 | 720p 600k | 25.8 | 17.4 | **1.48** (ours: BD-J −47.8 % vs libvpx rt 9's −5.7 %, both against x264 medium) |
| 8 encoder start-up (first packet) H.264 | x264 veryfast / openh264 | 1 frame 720p | 31.9 (Default), 22.3 (Faster) | 14.8 / 24.1 | **2.16** / 1.32; Faster 0.93 |
| 8 encoder start-up AV1 | SVT-AV1 M12 rtc | 1 frame 720p | 121.5 (Default), 52.9 (Faster) | 51.6 | **2.35**; Faster 1.03 |
| 8 encoder start-up HEVC / VP9 | x265 ultrafast / libvpx rt 9 | 1 frame 720p | 72.7 / 97.3 | 84.0 / 91.6 | 0.87 / 1.06 |
| 8 encoder start-up VVC | none available; x265 ultrafast for scale | 1 frame 720p | 235.6 | 84.0 | (2.8) |
| 8 per-frame encode latency H.264 | x264 veryfast / openh264 | 720p steady state | 5.85 (Default), 3.88 (Faster) | 7.56 / 3.84 | 0.77 / 1.01 (Faster) |
| 8 per-frame encode latency AV1 | SVT M12 rtc | 720p | 5.85 (Faster), 9.93 (Default) | 6.81 | 0.86 / 1.46 |
| 9 encoder peak RSS | x264 vf / openh264 / SVT M12 / libvpx rt9 / x265 uf | 720p, 150 frames | 19–30 MB | 27 / 21 / 32 / 87 / 68 MB | 0.4–1.1 |
| 9 decoder peak RSS | FFmpeg / dav1d / VVdeC / avmdec | 1080p | 17–27 MB (AV1 on SVT stream 51) | 23–47 / 14–33 / 128 / 205 MB | 0.1–0.9; **AV1 1.5–1.9** vs dav1d |

Per-run data (JSON, interleaved rounds):
`~/nanocodec-data/work/speed-audit/out/{decode,encode,startup,wasm}.json` and `display.txt`.
Scripts: `bench.py` and `wasm_bench.py`, kept with the session notes (not in the repo).

### 2.1 Decoders at the close of decoder speed work (2026-11-13)

This section supersedes the claim-4/5 decode rows above and §3.8 for current figures (build d007869b). The full
per-stream table and method are in
[wasm-speed.md "Decoder close-out"](wasm-speed.md#decoder-close-out-2026-11-13-dec-winddown).
Ratio = ours ÷ reference in time, same streams, single thread.
- **x86:** core 14, best of 7 interleaved rounds. The host was loaded, so read each ratio as about ±10 %.
- **Jetson:** core 3, best of 3 rounds.

| Codec | Reference | x86: our streams | x86: third-party | Jetson: our streams | Jetson: third-party | WASM / our native |
|---|---|---:|---:|---:|---:|---:|
| H.264 | FFmpeg | 0.88–0.90 | 0.95–1.00 (x264, openh264) | 0.77–0.78 | 0.82–0.85 | 1.75–2.13 |
| HEVC | FFmpeg | 0.73–0.78 | 0.57–0.94 (cameras, x265 B) | 0.72–0.85 | 0.75–0.92 | 1.21–1.79 |
| AV1 | dav1d | 1.13–1.27 | 1.52–1.58 (SVT, libaom) | 1.09–1.20 | 1.32–1.43 | 1.55–1.98 |
| VP9 | vpxdec (FFmpeg) | 1.32–1.36 (0.98–1.05) | 1.41 (0.98) | 1.18–1.24 (1.07–1.10) | 1.14 (1.07) | 1.65–2.05 |
| VVC | VVdeC | 0.39–0.47 | – | 0.63–0.70 | – | 1.69–1.70 |
| AV2 | avmdec | 0.33–0.39 | – | 0.31–0.32 | – | 1.69–1.88 |

**VP9 against vpxdec after vp9-dec-parity (2026-11-13).** Ours ÷ vpxdec in instructions (x86 callgrind) and cycles
(Jetson core 3, jperf). These are counts, not wall time. Build 80a0111a (Jetson: the same decoder at 159804aa).

| Stream | x86 instructions | Jetson cycles |
|---|---:|---:|
| ours 720p / 1080p | 1.163 / 1.224 | 1.218 / 1.170 |
| ours SVC L2T1 / L2T3 720p | 1.117 / 1.116 | 1.192 / 1.177 |
| vpxenc rt 720p / good 1080p | 1.248 / 1.271 | 1.277 / 1.137 |
| resize_inter 1080p (scaled references) | 1.430 | 1.290 |

At adb57e6e the same streams were:
- x86: 1.25–1.35, with SVC 1.98–2.08;
- Jetson: 1.14–1.27, with SVC 1.48–1.49.

Method and per-change detail: `docs/handoff/vp9.md` "Decoder speed against vpxdec".

**Like for like at 270b535e.** Neither decoder writes output: ours runs `--discard 1`, vpxdec `--noblit`.

| Stream | x86 instructions | Jetson cycles |
|---|---:|---:|
| ours 720p / 1080p | 1.123 / 1.185 | 1.188 / 1.149 |
| ours SVC L2T1 / L2T3 720p | 1.079 / 1.081 | 1.152 / 1.144 |
| vpxenc rt 720p / good 1080p | 1.198 / 1.229 | 1.242 / 1.122 |
| resize_inter 1080p | 1.405 | 1.255 |

The remaining Jetson gap is in the loop-filter and MC NEON kernels (`docs/handoff/vp9.md`, round 2).

**HEVC against Antare's `hevc.wasm`** (their numbers scaled by our own base-to-head ratio, as in
[wasm-speed.md](wasm-speed.md#final-re-time-2026-11-13-dec-winddown)): we are now faster on all five rows:
- Novatek −20 %;
- x265 1080p B −72 %;
- x265 4K B −55 %;
- our 720p −23 %;
- our 1080p −16 %.

On 2026-11-12 we were behind on three of those rows.

## 3. Worst gaps, ranked

1. **Demo display hand-off (I420→RGBA): 4.3–6.2x libyuv** in the browser (0.84–1.02 ms per 720p frame vs 0.20 ms).
   Natively the conversion is 13–18x slower, because the AVX2 and NEON tables point at the scalar row (no native
   product uses it yet). The WebGL path (0.15–0.3 ms) avoids most of this cost, but the CPU fallback is what the
   README quotes. **Fixed the same day (3.1).**
2. **AV1 decode: 1.5–2.8x dav1d** natively (2026-11-13 close-out: 1.13–1.58x on x86, 1.09–1.43x on the Jetson, §2.1). The gap is worst on SVT's 1080p stream, which uses more tools than ours,
   and our AV1 decoder uses 1.5–1.9x dav1d's memory. Already being fixed.
3. **WASM decode is behind an estimated SIMD port of FFmpeg/libvpx (est. 1.1–2.2x).** Ours beat ffmpeg.wasm, which
   has no SIMD, by only 12–30 %. This is consistent with the competitor HEVC WASM beating us by 7–52 %. The README's
   "only 1.4x native" compares us with ourselves, so it hides this gap. **2026-11-13, same estimate (reference
   native × 1.36):** HEVC 0.51–1.12x (and ahead of the competitor HEVC WASM on every row), H.264 1.13–1.51x,
   AV1 1.5–1.9x, VP9 1.6–2.1x. The rest is SIMD128 (no counterpart to the AVX2 / NEON gains at V8's register
   limit) and the AV1 / VP9 native gaps.
4. **Encoder start-up (time to the first packet): 2.2x x264 for H.264 Default and 2.35x SVT M12 for AV1 Default**
   (32 vs 15 ms and 122 vs 52 ms at 720p). VVC Default takes 236 ms. The Faster presets are at parity, so most of
   the cost is the Default key frame.
5. **VP9 decode: 2.0–2.2x vpxdec** (1.4–1.7x FFmpeg). 2026-11-13: 1.32–1.41x vpxdec and level with FFmpeg on x86;
   1.14–1.24x vpxdec on the Jetson (§2.1).
6. **H.264 and HEVC decode: 1.1–1.4x FFmpeg.** Closest on the Axis 720p camera stream, where we are level.
   **Closed 2026-11-13:** H.264 0.88–1.00x and HEVC 0.57–0.94x FFmpeg on x86, both under FFmpeg on the Jetson (§2.1).
7. **WASM encode estimates:** H.264 Fast 1.4x an openh264-class WASM, VP9 1.5x libvpx rt and AV1 Fast 1.2x SVT M12.
   These are estimates only. H.264 Default beats both x264 veryfast in ffmpeg.wasm and its estimated SIMD port.

**No gap:**
- VVC decode is about 2x faster than VVdeC and FFmpeg.
- AV2 decode is 2.3–3.5x faster than avmdec and dav2d, but those are a reference decoder and an early decoder,
  so they are weak yardsticks.
- Encode memory is the lowest of all the encoders measured.
- Steady-state encode latency agrees with the independent encode-speed charts (claim 1).

### 3.1 Display conversion after the fix (2026-11-12)

New exact kernels for every ISA (`nc-dsp` `convert/to_rgba_{avx2,neon,simd128}.rs`). They work on row pairs in
16-bit lanes, using an exact rewrite of the 8.8 BT.601 inverse (`bt601_split`). The output is unchanged: checkasm
compares every (Y, Cb, Cr) against the scalar reference on each ISA. The timings are the best of 3 interleaved
rounds of 300 conversions, each pinned to one core: PC core 15 (SIMD128 under node 22 via WASI) and Jetson core 3.

| ISA | 360p | 720p | 1080p | libyuv `I420ToABGR`, same core (360p / 720p / 1080p) | Ratio | Before (720p) |
|---|---:|---:|---:|---:|---:|---:|
| AVX2 (PC) | 0.039 | 0.157 | 0.389 | 0.048 / 0.188 / 0.427 | **0.82–0.91** | 3.36 (scalar) |
| SIMD128 (PC, node) | 0.059 | 0.230 | 0.542 | libyuv AVX2: 0.048 / 0.188 / 0.427 | **1.22–1.27** | 0.60 |
| NEON (Jetson Orin) | 0.233 | 0.931 | 2.134 | 0.286 / 1.143 / 2.571 | **0.81–0.83** | 3.81 (scalar) |

The demo's whole CPU hand-off, which is the conversion plus the JS copy out of WASM memory (bench-runner, relaxed
module, pinned A/B against the module built from variance-integration), now takes:

| Resolution | Before (ms) | After (ms) |
|---|---:|---:|
| 360p | 0.18 | 0.10 |
| 720p | 0.82 | 0.47 |
| 1080p | 2.69 | 1.95 |

The rest of each frame's hand-off is the JS `Uint8Array.set` copy out of WASM memory: about 0.25 ms at 720p and
1.4 ms at 1080p. A WASM kernel cannot write outside its linear memory, so this copy cannot be fused into the
conversion. Only the WebGL path, which hands off I420 at 1.5 bytes per pixel, avoids it.

### 3.2 RGBA→I420 (browser encode input), 2026-11-12

The output is unchanged: checkasm now also passes all 16.7M RGB colours through each ISA. Times are the best of
4 interleaved rounds, measured the same way as in 3.1, against libyuv `ABGRToI420` on the same core:

| ISA | 360p | 720p | 1080p | libyuv (360p / 720p / 1080p) | Ratio | Before (720p) |
|---|---:|---:|---:|---:|---:|---:|
| AVX2 (PC) | 0.044 | 0.168 | 0.415 | 0.038 / 0.150 / 0.390 | **1.06–1.15** | 0.23 |
| NEON + dot product (Jetson Orin) | | 0.593 | | 0.791 (720p) | **0.75** | 1.22 |
| SIMD128 (PC, node) | 0.067 | 0.264 | 0.62 | libyuv AVX2: 0.038 / 0.150 / 0.390 | **1.6–1.8** | unchanged |

- **AVX2** works straight from packed pixels, with no planar transpose. Luma comes from `vpmaddubsw` with the byte
  weights [66, −127, 25, 0] plus the masked G byte as 256 G, then one `vphaddw`. Chroma comes from a shuffle and a
  `vpmaddubsw` by ones per pixel pair, then exact 2x2 means, `vpmaddwd` and `vphaddd`.
- **NEON** forms luma from one `udot` per four packed pixels, onto the rounding and offset, with byte 1 of each
  32-bit lane taken by two unzips. The Orin's single multiply pipe made the old 12 widening multiplies per 16
  pixels the bottleneck. Without the dot-product extension (detected at run time) the kernel falls back to
  widening multiply-accumulates.
- **SIMD128** stays 1.6–1.8x libyuv AVX2. WebAssembly has no unsigned byte-weight multiply-add, and its
  `i32x4.dot_i16x8_s` path was already the cheapest exact form we found. This is about 2 % of a 720p browser
  encode.

### 3.3 Demo display path in Chromium (default check), 2026-11-12

The demo's renderer choice is `auto`. It picks WebGL2, then WebGL1, on a hardware GPU (via
`failIfMajorPerformanceCaveat`), and falls back to the CPU path. The worker then hands off I420 (1.5 bytes per
pixel) instead of RGBA. We re-measured this with the new CPU kernel:
- **Setup:** headless Chromium with ANGLE/Vulkan on the RTX 5090; the whole browser pinned to cores 14 and 15.
- **Method:** `tools/web/display-bench.py --gpu vulkan --async`, best of 3 interleaved rounds, median over three
  clips.
- **Exactness:** WebGL2 output is bit-identical to the CPU path (max |d| 0).

End-to-end milliseconds per frame:

| Codec | 720p CPU | 720p WebGL2 | 360p CPU | 360p WebGL2 |
|---|---:|---:|---:|---:|
| H.264 | 3.51 | 2.32 | 0.79 | 0.71 |
| HEVC | 4.36 | 2.99 | 1.12 | 0.95 |
| AV1 | 5.59 | 3.93 | 1.35 | 1.31 |
| VP9 | 4.49 | 2.95 | 1.02 | 0.89 |

WebGL2 is 30–34 % faster at 720p and 3–15 % faster at 360p. Its worker hand-off is 0.12–0.16 ms at 720p, against
0.69–0.86 ms on the CPU path. WebGL is already the default, with the CPU path as the fallback, so nothing changes.

### 3.4 Encoder start-up (time to the first packet), profiled 2026-11-12

720p crowd_mot17_09 at 600 kbps CBR. Each command was run 7 times, interleaved and pinned (PC core 15, Jetson
core 3); the table gives the best wall time in ms, with CPU time and minor page faults from `wait4`.

| Encoder | PC wall (user / sys, faults) | Jetson wall (user / sys, faults) |
|---|---:|---:|
| ours H.264 Default, 1 frame | 24.7–27.9 (14 / 10–14, 3 750) | 42.2 (30.0 / 11.3, 3 377) |
| x264 veryfast, 1 frame | 13.6 (11.1 / 2.0, 993) | 26.7 (17.4 / 8.7, 1 402) |
| ours AV1 Default, 1 frame | 119.5 (105 / 14, 4 670) | 245.6 (232 / 12, 4 316) |
| SVT-AV1 M12 rtc, 1 frame | 52.6 (40 / 12, 5 273) | 114.8 (98 / 16, 4 212) |
| ours H.264, 0 frames (start-up and read-ahead only) | 6.0 (0 / 5.8, 1 359) | |

Where the time goes (callgrind, x86):
- **Start-up proper is small.** Building the encoder, the tables and the first-key read-ahead takes about 3M
  instructions and 5–6 ms of wall time on the PC, almost all of it kernel time spent faulting in memory. Of the
  H.264 1-frame run's 151M instructions, 148M are in `encode_frame`, and 132M of those are the key picture's
  coding. For AV1, the key picture is 1.30G of 1.32G instructions: `KeyPlanner::plan_node` 35 %, residual RD
  cost 45 % (`CoeffCdfs::cost` alone 14 %), filter-intra prediction 7 %. A second, P, frame adds only 44M
  (H.264) or 97M (AV1).
- **On the Jetson the gap is all user time:** the key frame's mode decision. Page-fault time there is level
  with x264 and SVT, because the Jetson uses transparent huge pages for every allocation. On the PC, transparent
  huge pages are opt-in (`madvise`). x264 opts in, which is why it takes 993 faults to our 3 750. Opting in
  needs an `madvise` call, which means `unsafe` code outside nc-dsp, so we did not.
- **No identity-preserving cut is worth landing.** The largest one found removes the CLI's double buffering of
  the read-ahead frames, about 1.4 MB of first touches (about 1–1.5 ms on the PC). The rest of the gap is
  key-frame RDO work spread across many functions. Cutting it means faster intra RD costing, which is ordinary
  encoder-speed work: the same functions dominate every intra block in later frames.
- **The larger latency term is by design.** With rate control, `rate.first_key_lookahead` holds the first packet
  until the second frame has arrived (`Encoder::wants_look_ahead`), so a live encoder's first packet costs one
  frame interval more (33 ms at 30 fps). This changes the stream, so it is a preset decision, not an identity
  fix.

### 3.5 Browser module build, 2026-11-12

- **Compile settings: already right.** The browser module is built with the workspace release profile: fat LTO,
  `codegen-units = 1`, `opt-level = 3` and `panic = "abort"`. rustc 1.99's wasm32 baseline already enables
  bulk memory, sign extension and non-trapping float-to-int, and the module uses them: 2 432 `memory.copy`,
  1 425 `memory.fill`, 296 `i32.extend8_s` and 112 `trunc_sat`.
- **Module choice: right.** The demo loads the relaxed-SIMD module wherever the engine validates relaxed SIMD
  (`web/wasm.js`), and it instantiates with `WebAssembly.instantiateStreaming` (wasm-bindgen's `default()`
  fetch), so Chromium can code-cache the module on repeat visits.
- **wasm-opt is a net loss, so the pipeline stays without it.** We tried binaryen 132 (`-O3`/`-O4`, with
  `--enable-simd`, `relaxed-simd`, `bulk-memory`, `sign-ext`, `nontrapping-float-to-int`, `mutable-globals`,
  `multivalue` and `reference-types`) on the relaxed module. Both builds decode bit-identically and encode to
  the same sizes. Measured pinned, 2 interleaved rounds, all six codecs:

  | Change vs no wasm-opt | Decode | Encode |
  |---|---|---|
  | AV1 | −3.1 % (`-O3`), −5.1 % (`-O4`) | worse by 12–16 % |
  | AV2, VVC | within ±1 % | worse by 2–10 % |
  | H.264, HEVC | worse by 1–3 % | worse by 5–11 % |
  | VP9 | worse by 11–13 % | worse by 6–7 % |

- **V8 tier-up: the first second is slow.** A fresh module, under node 22 pinned, decoded 720p crowd:

  | Codec | First 30 frames, cold (ms) | First 30 frames, warm (ms) | Frame 0, cold (ms) | Frame 0, warm (ms) |
  |---|---:|---:|---:|---:|
  | H.264 | 161 | 57 | 77 | 10 |
  | HEVC | 241 | 91 | 100 | 14 |
  | AV1 | 432 | 126 | 155 | 16 |
  | VP9 | 163 | 77 | 72 | 10 |
  | AV2 | 357 | 157 | 134 | 24 |
  | VVC | 334 | 195 | 136 | 33 |

  From frame 30 on, cold and warm match within 3 %. So the cost is V8's lazy, per-function compilation on
  first call, not slow Liftoff code: `--no-liftoff` makes frame 0 slower (96–226 ms), not faster. Instantiation
  itself takes about 50 ms.

  **Landed (2026-11-12):** the decode worker now pre-warms on a built-in 256x144 stream of its codec
  (`web/warmup.js`, see `web/README.md`). In headless Chromium, which compiles more in the background than node,
  the cold cost is smaller than the node figures above. There, pinned, the warm-up cuts frame-0 decode time by
  30–55 % (H.264 28 → 12–18 ms, VVC 77 → 36 ms) and the first 30 frames by 8–17 %.

  The page cannot choose eager compilation. Two mitigations remain:
  - Chromium's wasm code cache, already reachable through streaming compilation, helps on repeat visits.
  - A worker could pre-warm by decoding a tiny built-in stream of the chosen codec before the first real frame.
    This is not done yet: it covers only the code paths that stream exercises.

### 3.6 Decoder memory (peak heap, 1080p), 2026-11-12

DHAT peak heap of our decoders on the 1080p 3 Mbit/s streams from section 2:

| Decoder | Peak heap | Largest parts |
|---|---:|---|
| AV1, SVT's stream | 48.8 MB | 27 bordered picture planes (9 pictures, about 3.9 MB each), 34.8 MB; 8 saved motion fields of 2.1 MB; CDEF scratch 4.5 MB |
| AV1, our stream | 23.5 MB | 9 planes (3 pictures), 11.6 MB; motion and CDEF scratch |
| HEVC, Axis 1080p camera (reorder depth 3) | 34.5 MB | 12 decoded pictures, 15.6 MB, plus 12 output copies (`Frame::output`), 12.7 MB; PU motion 5.2 MB |
| VP9, libvpx stream | 27.8 MB | 15 planes (5 pictures), 19.5 MB; a 3.15 MB per-block buffer |
| VVC, ours | 13.9 MB | 8 planes, 7.0 MB; a full-picture copy for SAO, 3.5 MB; motion grid 3.1 MB |
| AV2, ours | 11.7 MB | 6 planes, 6.4 MB; the CLI's output copy, 3.2 MB; CDEF 2.1 MB |
| H.264, x264 stream | 9.4 MB | 6 planes, 7.8 MB |

The large items are in the AV1, HEVC and VP9 decoders. Those crates are owned by the decoder agents right now, so
this audit hands the list over instead of editing them:
- **AV1:** about 100-sample picture borders (dav1d uses none) and one picture per distinct reference slot.
- **HEVC:** a copy per output picture in bumping order.
- **VP9:** five bordered pictures.

The AV2 and VVC copies are about 3 MB each at 1080p and transient. Removing the AV2 copy in the CLI changed peak
RSS by under 1 MB, so it was not landed.

### 3.7 Key-frame encode cost, 2026-11-12

**Measurement.** 720p crowd and pan clips, 30 frames, CBR 600 kbps, pinned, 2 interleaved rounds. "Key extra" is
(time with a key every 5 frames − time with one key) / 5: the extra cost of a key frame over an inter frame. This
is the latency hit on joining a stream, on recovery, and on a key sent on request.

| Encoder | Key extra (ms) | Inter frame (ms) | | Encoder | Key extra (ms) | Inter frame (ms) |
|---|---:|---:|---|---|---:|---:|
| ours H.264 | 1.0 | 7.1 | | x264 veryfast | 0.6 | 6.2 |
| ours H.264 Fast | 2.2 | 5.6 | | SVT-AV1 M12 rtc | 11.2 | 7.3 |
| ours HEVC | 7.0 | 11.1 | | SVT-AV1 M10 rtc | 31.9 | 11.3 |
| ours HEVC Fast | 4.4 | 7.4 | | libvpx VP9 rt 9 | 7.6 | 8.9 |
| ours AV1 | 20.0 | 14.3 | | | | |
| ours AV1 Fast | 11.8 | 9.2 | | | | |
| ours VP9 | 13.1 | 14.6 | | | | |
| ours AV2 | 11.5 | 15.9 | | | | |
| ours VVC | 58.9 | 23.4 | | | | |
| ours VVC Fast | 29.2 | 13.0 | | | | |

**Comparisons.** Rate control shifts bits between key and inter frames, so the column is approximate; x264
`medium` even comes out negative, which is why it is not listed.
- **AV1:** Default costs about 1.8x SVT M12's key extra; Fast is level with it.
- **H.264 and HEVC:** keys cost little.
- **VVC Default:** the largest key extra by far. There is no real-time VVC competitor to compare with.

**Exact work, landed (2b8f4754).** `dc_value` now sums the left column from one slice, which cuts AV1 key-frame
instructions by 1.5 % with byte-identical streams for every codec and preset. In AV1 Default key frames,
key-block planning takes 34 % and the mode decision 69 % (residual RD cost 44 %, prediction 27 %). Those paths are
already SIMD and pruned (`forward_64_pruned`, sparse EOB, cached CDF costs), and no other exact change found was
above 1 %. Two VVC scratch-buffer resizes were tried: they saved 0.15 %, against the 2–3 % that callgrind's
inlined-caller attribution suggested, and were dropped.

**Lever, screened and rejected.**
- **Change:** Fast's key-frame intra settings in AV1 Default (`av1.keyframe_directional=0`,
  `keyframe_angle_deltas=0`, `keyframe_filter_intra=0`).
- **Effect on keys:** key-frame time −31 % (fixed QP 120, all-key), +2.4 % key bytes.
- **Screen:** L2 adopt over 48 pool clips gives J +0.95 % [+0.71, +1.22] and fails the guard
  `bd_worst_vmaf_v1` (+2.22). Verdict: WORSE.
- **Why it can't pay off on the screen:** J prices whole streams, whose keys are a small share of the CPU,
  so a key-latency gain earns almost no CPU credit there.
- **What adoption would need:** an owner decision on how to weigh key-frame latency. For example, a cheaper key
  search used only for keys requested mid-stream, screened against a latency objective.

### 3.8 Native decode on the Jetson (Cortex-A78AE NEON), 2026-11-12

**Setup.**
- **Build:** head 1b2c955c, built on the Jetson.
- **References:** the pinned aarch64 anchors: FFmpeg n8.0.1 with NEON assembly, `-threads 1`; dav1d 1.5.4
  `--threads 1`; vpxdec 1.16 `--threads=1`; VVdeC `-t 1`. avmdec v1.0.0 was built on the Jetson for this run.
- **Method:** every process pinned to core 3 under `jetson-timed.sh`, 3 interleaved rounds, best of 3. Times are
  per frame. Ratio = ours ÷ reference (> 1 means we are slower).

| Codec | Stream | Ours (ms/frame) | Reference (ms/frame) | Jetson ratio | x86 ratio (§2) |
|---|---|---:|---|---:|---:|
| H.264 | ours 720p crowd / pan | 2.18 / 2.36 | FFmpeg 2.68 / 2.91 | **0.81** | 1.12–1.16 |
| H.264 | ours 1080p 3M / x264 1080p 3M | 7.10 / 8.68 | FFmpeg 8.57 / 9.13 | **0.83 / 0.95** | 1.19 / 1.38 |
| HEVC | ours 720p crowd / pan / 1080p 3M | 3.47 / 4.24 / 13.38 | FFmpeg 4.18 / 4.78 / 14.71 | **0.83–0.91** | 1.14–1.27 |
| HEVC | Axis 720p / Axis 1080p / Axis 5MP / Hanwha 4K cameras (150 frames each) | 2.72 / 6.34 / 13.70 / 23.69 | FFmpeg 3.37 / 7.79 / 16.59 / 27.96 | **0.81–0.85** | 0.99–1.16 |
| AV1 | ours 720p crowd / pan | 6.04 / 7.99 | dav1d 4.75 / 6.22 | **1.27–1.28** | 1.52–1.57 |
| AV1 | ours 1080p 3M / SVT 1080p 3M | 19.57 / 15.65 | dav1d 14.55 / 9.84 | **1.35 / 1.59** | 2.31 / 2.80 |
| VP9 | ours 720p crowd / pan | 3.76 / 4.66 | vpxdec 2.74 / 3.60; FFmpeg 3.20 / 3.98 | **1.30–1.37** (FFmpeg 1.17–1.18) | 2.09–2.19 |
| VP9 | ours 1080p 3M / vpxenc 1080p 3M | 13.49 / 14.15 | vpxdec 10.60 / 11.64; FFmpeg 11.47 / 12.47 | **1.21–1.27** (FFmpeg 1.13–1.18) | 2.01–2.17 |
| VVC | ours 720p crowd / pan / 1080p 3M | 6.96 / 7.66 / 27.09 | VVdeC 8.17 / 9.66 / 32.50; FFmpeg 13.36 / 16.19 / 56.31 | **0.79–0.85** (FFmpeg 0.47–0.52) | 0.40–0.53 |
| AV2 | ours 720p crowd / pan / 1080p 3M | 6.36 / 7.11 / 21.00 | avmdec 19.87 / 24.46 / 62.39 | **0.29–0.34** | 0.40–0.44 |

- **H.264, HEVC and VVC are faster than the references on the Jetson**, though H.264 and HEVC are slower than
  FFmpeg on x86. The x86 gaps are therefore an x86 (AVX2) problem for those two codecs.
- **AV1 (1.27–1.59x dav1d) and VP9 (1.13–1.37x) are the Jetson gaps.**
- **Peak RSS:** AV1 26–51 MB against dav1d's 10–32 MB. Elsewhere we use less memory than the reference
  (VVdeC 59–126 MB, avmdec 129–206 MB).

**Where the time goes on the Jetson** (`tools/dev/pcsample.py`, self time; libc `memcpy` and `memset` show as
`strerrordesc_np` at 2–6 %):
- **AV1, SVT stream:** `predict_compound` 16 %, MC `prep_w32`/`prep_w64` 20 % (compound prediction, which SVT
  uses and our encoder does not), `transform_block` 15 %, `warp` 2 %.
- **AV1, our stream:** deblock `neon::av1_edge` 16.5 %, `transform_block` 11 %, MC `neon_dotprod` 20 %,
  `inter_modes` 7.5 %, CDEF 9 %.
- **VP9:** the NEON 8-tap MC driver (`vp9_mc_simd::driver::by_width`, 165 taps) 38 %, the loop filter
  (`vp9_loop_filter::packed::run`) 18 %, `decode_block` 13 %, `residual` 8 %.
