# Browser (WebAssembly) speed: raw data

<sub>Updated 2026-11-10, builds f2f7a221 (H.264–AV2) and bcbc5601 (VVC)</sub>

Data behind [README.md § In the browser](README.md#in-the-browser-webassembly); every row of
[wasm-speed.csv](wasm-speed.csv) is one stream (decode) or clip (encode) per module, best over rounds.

**Setup.** AMD Ryzen 9 5950X, node v22.21.1 (V8, the engine in Chrome), one thread: the browser module has
no threads. Every timed process (node and native) pinned to core 15 with `taskset`; the PC was shared with other
agents (1-minute load 8.7–17.9 during the H.264–AV2 runs, 4.0–11.0 during the VVC run). Three interleaved
rounds: each round runs every stream or clip once per module and once natively; the best round counts. Decode
processes run two passes each (the first warms V8's tier-up), 720p encode one (the warm-up effect is within noise
there).

**Builds.** H.264, HEVC, AV1, VP9, AV2: f2f7a221 (WASM sha256 simd128 `5f53d1960b9b`, relaxed `41ac3112a308`;
native `d963520192e5`). VVC (added to the browser build by 57e3cdd2): bcbc5601 (simd128 `63eb6e6cd605`, relaxed
`ecc099eb19d6`; native `543c34e53177`). The commits between the two builds change no H.264–AV2 code on x86 or
WASM (NEON-only CDEF, VVC); the VVC run re-measured HEVC as a control (below).

**Streams.** Our own CBR encodes (peak 2x, 1 s buffer, default key interval, 150 frames at the clip's 30 fps) of
crowd_mot17_09, pan_jaad_0117, static_chirla_s000c3 (one clip per content class; static is 1080x720): 600 kbps
at 720p, 300 kbps at 360p (≤ 50 % of every source's bitrate). Every module's decoded I420 is hash-checked
against the native decoder on every stream (all bit-exact).

**Caution (2026-11-11):** the AV1, VP9 and AV2 encode rows below were measured with QP 32 as the
first-key seed instead of the demo's q-index 180 (bench-runner fixed 2026-11-11); their WASM side coded a twice-larger
first key and costlier frames after it.

**Encode** at the demo's settings: RGBA in (converted untimed beforehand; the timed call is the demo's
`BrowserEncoder.encode`, RGBA to I420 plus encode), 15 fps, CBR 300 kbps at 360p (the demo's default) and
600 kbps at 720p, peak 2x, 1 s buffer, 30 s keyframes, 150 frames. Fast only where the demo offers it (H.264,
HEVC, AV1, VVC). Native: `nanocodec encode` with the same settings and `--preset <codec>-fast`.

**Native** is process wall time (`nanocodec decode … --output /dev/null`, `nanocodec encode`), best of the
same rounds; at 360p decode the process start-up (a few ms per 150 frames) makes native look up to ~5 % slower.
WASM/native compares WASM decode alone (without the display hand-off) with native decode.

Commands (the bitstreams, under `~/nanocodec-data/work/wasm-speed/streams/`, are the only files written):

```sh
tools/web/build.sh   # PKG_DIR=<dir> for a pair outside web/pkg
common="--bin $CARGO_TARGET_DIR/release --pkg <pkg> --modules simd128,relaxed --targets 720p=600,360p=300 \
  --presets default,fast --cpu 15 --workers 4"
python3 tools/web/bench.py $common --encode-res 360p --repeat 2 --rounds 3 --json a.json --csv a.csv
python3 tools/web/bench.py $common --skip-decode --encode-res 720p --repeat 1 --rounds 3 --json b.json --csv b.csv
# VVC: the same two commands with --codecs vvc,hevc at bcbc5601
taskset -c 15 python3 tools/web/display-bench.py --streams <stream dir> --codecs h264,hevc,av1,vp9,av2 \
  --kbps 720p=600,360p=300 --renderers cpu --rounds 3
```

## Native YUV capture: capture, encode and decode (2026-11-13, wasm-perf4)

**Decode, WASM ÷ native** (720p crowd 600 kbps, 150 frames, adb57e6e; ms per frame, three interleaved rounds
on timing core 14, `work/wasm-perf4/xdec-base.json`):

| Codec | native | WASM SIMD128 | WASM relaxed | relaxed ÷ native | reference decoder |
|---|---|---|---|---:|---|
| H.264 | 0.98-1.06 | 1.73-1.92 | 1.78-1.85 | 1.82 | FFmpeg 1.11-1.22 |
| HEVC | 1.32-1.41 | 2.22-2.28 | 2.15-2.25 | 1.63 | FFmpeg 1.73-1.79 |
| AV1 | 1.85-1.91 | 3.57-3.68 | 3.61-3.64 | 1.95 | dav1d 1.63-1.65 |
| VP9 | 1.40-1.44 | 2.50-2.61 | 2.52-2.60 | 1.80 | vpxdec 1.05-1.06, FFmpeg 1.36-1.38 |
| AV2 | 2.50-2.52 | 4.30-4.44 | 4.21-4.36 | 1.69 | avmdec 6.97-7.24 |
| VVC | 2.78-2.84 | 4.55-4.84 | 4.49-4.89 | 1.62 | VVdeC 8.15-10.06 |

Per-stage split and the reasons: `docs/handoff/wasm-perf.md` "Native YUV capture and the per-stage decode gap".

**Capture worker cost per frame** (`work/wasm-perf4/capture/wbench.py`: a worker given 30 `VideoFrame`s of
the playing 720p or 360p mot17-mall-walk file; min and median ms, three rounds; headless Chromium 148 on a
loaded PC, load ~30):

| Frames | Old, scaled view (canvas draw + read back: 854x480 from 720p, 384x216 from 360p) | Old, view at the file's size (`copyTo` RGBA) | New, any size (`copyTo` own format) |
|---|---|---|---|
| 720p, software decode (I420) | 7.5 / 12.9, 10.0 / 17.9, 15.4 / 27.6 | 2.1 / 4.4, 3.3 / 6.3, 3.8 / 9.9 | 0.2 / 0.6, 0.4 / 1.0, 0.5 / 1.2 |
| 720p, GPU (Vulkan, NV12) | 27.0 / 47.5, 22.9 / 41.5, 30.6 / 51.2 | 18.6 / 40.4, 20.4 / 47.6, 20.4 / 43.5 | 3.8 / 12.7, 3.2 / 13.3, 4.2 / 9.7 |
| 360p, software (I420) | 1.6 / 6.1, 1.7 / 6.7, 1.8 / 8.0 | 0.9 / 1.3, 0.9 / 1.3, 0.8 / 1.5 | 0.0 / 0.2, 0.1 / 0.2, 0.0 / 0.2 |
| 360p, GPU (Vulkan, NV12) | 3.1 / 9.6, 4.8 / 12.7, 6.3 / 10.8 | 3.5 / 8.4, 4.3 / 12.3, 7.2 / 13.3 | 1.7 / 5.4, 2.1 / 6.7, 2.8 / 7.5 |

What moved into the encode worker in exchange (node 22, relaxed module, timing core 15, median of 60): placing
720p planes in an 854x480 view 0.93 ms (scaled; first version 1.52 ms, then vertical pass first), in a 640x360
view 0.55 ms, at 1280x720 0.09 ms; the old RGBA conversion took 0.18, 0.10 and 0.48 ms, plus the copy of 4 instead of
1.5 bytes per pixel into WASM. At the default size the capture worker saves 12-40 ms per frame (median)
and the encoder pays ~0.7 ms more.

**Encode, WASM ÷ native** (720p crowd, the demo's settings: 15 fps, CBR 600 kbps, 2x peak, 1 s buffer, 90
frames; relaxed module; best of 2 interleaved rounds on timing core 14; `work/wasm-perf4/encab.json`). Old
= RGBA input (the clip converted to RGBA once, untimed; the encoder converts back to BT.601), new = the clip's
own planes through `SourceFrame`, as the page now codes video. The inputs differ (BT.601 round trip against
the original planes), so rate control's choices differ slightly too:

| Codec | Preset | WASM old ms/f | WASM new ms/f | change | native ms/f | old ÷ native | new ÷ native |
|---|---|---:|---:|---:|---:|---:|---:|
| H.264 | Default | 18.7 | 17.0 | -9.2 % | 9.2 | 2.03 | 1.85 |
| H.264 | Fast | 10.1 | 10.7 | +6.4 % | 4.8 | 2.12 | 2.26 |
| HEVC | Default | 22.2 | 21.5 | -3.0 % | 11.0 | 2.02 | 1.96 |
| HEVC | Fast | 17.6 | 14.5 | -17.7 % | 7.9 | 2.22 | 1.83 |
| AV1 | Default | 28.2 | 26.2 | -7.2 % | 14.4 | 1.95 | 1.81 |
| AV1 | Fast | 19.9 | 16.6 | -16.5 % | 9.1 | 2.19 | 1.83 |
| VP9 | Default | 31.9 | 28.9 | -9.3 % | 16.3 | 1.95 | 1.77 |
| AV2 | Default | 32.7 | 31.7 | -3.2 % | 17.4 | 1.88 | 1.82 |
| VVC | Default | 38.1 | 35.5 | -6.9 % | 21.8 | 1.75 | 1.63 |
| VVC | Fast | 24.6 | 20.5 | -16.8 % | 13.0 | 1.89 | 1.57 |

**Demo end to end** (`work/wasm-perf4/e2e2.py`: headless Chromium 148, ANGLE/Vulkan on the RTX 5090, a fresh
browser context per load; 4 interleaved rounds of old (adb57e6e) and new pages, each case measured 8 s after
3 s of warm-up; delivered fps per round, target in brackets). The PC ran at load 30-45 throughout, so the old
page delivered about half of what demo-wasm-speed measured in the morning; read the medians, not single
rounds. Indicative only:

| Case | old fps | new fps | old median | new median |
|---|---|---|---:|---:|
| default (854x480 from the 720p file, H.264 vs AV1 Default) [15] | 9.5, 6.3, 7.4, 5.8 | 6.2, 8.5, 8.0, 7.6 | 6.9 | 7.8 |
| 1280x720, H.264 vs AV1 Default [30] | 7.1, 5.6, 3.0, 11.6 | 5.4, 13.9, 12.4, 14.2 | 6.4 | 13.2 |
| 1280x720, VP9 Default vs HEVC Fast [30] | 9.2, 5.8, 19.7, 5.9 | 5.7, 11.4, 15.0, 14.7 | 7.6 | 13.1 |
| time to first frame, ms (default page, no code cache) | 4340, 4013, 3081, 861 | 2079, 2230, 764, 1037 | 3547 | 1558 |

## Encode against native, and the demo end to end (2026-11-13, demo-wasm-speed)

Analysis and the per-stage breakdown: our internal notes (not published) "Demo end-to-end".

**Encode, WASM against native, 720p.** Build b8f93e74 (WASM relaxed module sha256 `28bf26171657`; native
release CLI from the same commit). `tools/dev/host-timed.sh python3 tools/web/bench.py --bin <release> --pkg
<pkg> --modules relaxed --skip-decode --encode-res 720p --presets default,fast --rounds 1 --repeat 2 --cpu 14`:
crowd_mot17_09, pan_jaad_0117, static_chirla_s000c3, 600 kbps CBR (peak 2x, 1 s buffer), 15 fps, 30 s keys,
150 frames, frame-weighted; load 39-41 on the other cores. Evidence:
`~/nanocodec-data/work/demo-wasm-speed/bench-encode-720p.{json,log}`.

| Codec | Preset | WASM encode fps | WASM ms/f | Native fps | WASM/native time |
|---|---|---:|---:|---:|---:|
| H.264 | Default | 60.0 | 16.7 | 127.2 | 2.12x |
| H.264 | Fast | 79.6 | 12.6 | 177.5 | 2.23x |
| HEVC | Default | 42.9 | 23.3 | 82.5 | 1.92x |
| HEVC | Fast | 69.4 | 14.4 | 141.6 | 2.04x |
| AV1 | Default | 33.7 | 29.7 | 62.9 | 1.87x |
| AV1 | Fast | 53.5 | 18.7 | 119.0 | 2.22x |
| VP9 | Default | 33.7 | 29.7 | 63.3 | 1.88x |
| AV2 | Default | 26.6 | 37.7 | 52.4 | 1.97x |
| VVC | Default | 26.6 | 37.5 | 49.5 | 1.86x |
| VVC | Fast | 44.9 | 22.3 | 85.7 | 1.91x |

**Demo end to end, delivered fps** (headless Chromium 148, ANGLE/Vulkan, WebGL2 display, H.264 Default vs AV1
Default, interleaved rounds of 8 s after 3 s warm-up, load 36-52; `work/demo-wasm-speed/ab{2,3}.json`):

| Size @ target | before (main-thread capture) | capture worker (dabaf0ca) | + 4 in flight (16a135db) |
|---|---|---|---|
| 854x480 @ 15 (default) | 11.5 14.5 8.3 | 14.2 15.4 14.5 | |
| 854x480 @ 30 | | 23.3 15.6 16.0 | 23.3 28.0 22.5 |
| 1280x720 @ 30 | 10.0 11.7 8.8 | 13.2 17.2 14.1 / 13.2 10.5 15.9 | 15.5 15.2 13.9 |
| 640x360 @ 30 | 29.3 28.3 27.8 | 27.0 29.5 29.4 | |

The two capture-worker columns at 720p come from two separate A/B sessions (each against its own control).

## Decoder close-out (2026-11-13, dec-winddown)

The final decoder table before decoder speed work was closed (internal notes, not published). One build of every decoder: variance-integration **d007869b** (native release; WASM
relaxed-SIMD module from the same commit). Ours / ref < 1 means we are faster than the single-threaded reference.

**Method.**
- **x86:** Ryzen 9 5950X. Every process was pinned to timing core 14 under `tools/dev/host-timed.sh`.
- **Rounds:** 7 interleaved rounds (3 + 4; order reversed every other round). Each cell is the best round.
- **Native timing:** process wall time, decoding to `/dev/null`.
- **WASM timing:** `tools/web/bench-runner.mjs decode`, decode-only ms per frame (best of 2 passes per process).
- **References:** pinned anchors, all single-threaded:
  - FFmpeg n8.0.1 `-threads 1` (H.264, HEVC);
  - dav1d 1.5.4 `--threads 1 --framedelay 1` (AV1);
  - vpxdec 1.16 `--threads=1 --noblit` (VP9; FFmpeg in brackets);
  - VVdeC `-t 1` (VVC);
  - avmdec `--threads=1` (AV2).
- **Noise:** the PC was not quiet. The 1-minute load on the other cores was 20–55, against the 5–15 that we
  expected. For one stream and decoder, max/min over the rounds had a median of 19 % and a 90th percentile of 64 %.
  The worst cells were the 4K and Novatek WASM runs, up to 136 %. Treat x86 ratios as about ±10 % and the WASM
  ms as upper bounds.
- **Jetson:** Orin Nano, one Cortex-A78AE core (core 3, `jetson-timed.sh`). Built on the board from the same
  commit; best of 3 interleaved rounds. The board was otherwise idle, and its run-to-run noise is about 1 %.
- **Streams** (150 frames unless noted):
  - Ours: crowd_mot17_09 720p 600 kbit/s and crowd_pp22_189 1080p 3 Mbit/s.
  - HEVC "ours 720p" on x86 is `hevc-wasm-speed/streams/ours-720p-600k.hevc` (the Antare row's stream). On the
    Jetson it is the README crowd stream.
  - Third-party:
    - x264 and openh264 CABAC 1080p 3 Mbit/s;
    - Antare's camera clips: Novatek 2460x1848 (25 frames), Axis 720p and 1080p, plus Axis 5 MP and Hanwha 4K on
      the Jetson;
    - x265 `medium` 1080p with B-frames, and x265 `fast` 4K with B-frames (120 frames);
    - SVT-AV1 M10 1080p 3 Mbit/s, SVT-AV1 preset 8 720p 1 Mbit/s, and libaom good-quality 720p 1 Mbit/s
      (40 frames);
    - vpxenc 1080p 3 Mbit/s.
- **Exactness:**
  - Native output on every x86 stream in the table has the same whole-output MD5 as its reference decoder's
    (FFmpeg, dav1d, vpxdec, VVdeC or avmdec). The Antare clips also match their `.framehash` files.
  - On the Jetson, the AV1 third-party streams and our 720p stream also match dav1d.
  - WASM I420 equals native on every stream checked. The exception is x265 B, whose pictures the browser decoder
    emits in decoding order (its low-delay API): the per-picture MD5 multiset is identical.

| Codec | Stream | x86 ours ms | x86 ref ms | **x86 ours/ref** | Jetson ours ms | Jetson ref ms | **Jetson ours/ref** | WASM ms (x86) | WASM/native |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H.264 | ours 720p | 0.95 | 1.08 | **0.88** | 2.06 | 2.69 | **0.77** | 1.66 | 1.75 |
| H.264 | ours 1080p 3M | 3.12 | 3.47 | **0.90** | 6.71 | 8.60 | **0.78** | 6.23 | 2.00 |
| H.264 | x264 CABAC 1080p | 4.40 | 4.38 | **1.00** | 7.71 | 9.11 | **0.85** | 8.97 | 2.04 |
| H.264 | openh264 CABAC 1080p | 3.43 | 3.62 | **0.95** | 6.95 | 8.47 | **0.82** | 7.31 | 2.13 |
| HEVC | ours 720p | 1.31 | 1.67 | **0.78** | 3.00 | 4.18 | **0.72** | 2.08 | 1.59 |
| HEVC | ours 1080p 3M | 5.19 | 7.10 | **0.73** | 12.45 | 14.70 | **0.85** | 9.27 | 1.79 |
| HEVC | Novatek 2460x1848 | 9.89 | 13.37 | **0.74** | 17.68 | 21.33 | **0.83** | 14.35 | 1.45 |
| HEVC | Axis 720p / 1080p camera | 1.16 / 3.50 | 1.66 / 5.30 | **0.70 / 0.66** | 2.53 / 5.87 | 3.37 / 7.80 | **0.75 / 0.75** | 1.85 / 5.53 | 1.60 / 1.58 |
| HEVC | Axis 5MP / Hanwha 4K camera | | | | 13.19 / 21.83 | 16.52 / 28.07 | **0.80 / 0.78** | | |
| HEVC | x265 1080p B | 7.02 | 7.47 | **0.94** | 15.63 | 16.99 | **0.92** | 11.36 | 1.62 |
| HEVC | x265 4K B | 15.61 | 27.39 | **0.57** | 23.16 | 30.17 | **0.77** | 18.84 | 1.21 |
| AV1 | ours 720p | 1.82 | 1.62 | **1.13** | 5.17 | 4.75 | **1.09** | 3.38 | 1.85 |
| AV1 | ours 1080p 3M | 6.74 | 5.33 | **1.27** | 17.40 | 14.54 | **1.20** | 13.32 | 1.98 |
| AV1 | SVT M10 1080p 3M | 5.33 | 3.39 | **1.57** | 13.02 | 9.83 | **1.32** | 8.28 | 1.55 |
| AV1 | SVT p8 720p | 2.78 | 1.83 | **1.52** | 7.21 | 5.29 | **1.36** | 4.44 | 1.60 |
| AV1 | libaom good 720p | 4.80 | 3.04 | **1.58** | 11.33 | 7.91 | **1.43** | 7.96 | 1.66 |
| VP9 | ours 720p | 1.43 | 1.05 (FFmpeg 1.36) | **1.36** (1.05) | 3.40 | 2.74 (FFmpeg 3.19) | **1.24** (1.07) | 2.35 | 1.65 |
| VP9 | ours 1080p 3M | 5.08 | 3.83 (5.20) | **1.32** (0.98) | 12.58 | 10.62 (11.47) | **1.18** (1.10) | 9.55 | 1.88 |
| VP9 | vpxenc 1080p 3M | 5.98 | 4.24 (6.11) | **1.41** (0.98) | 13.36 | 11.73 (12.48) | **1.14** (1.07) | 12.25 | 2.05 |
| VVC | ours 720p | 2.73 | 7.01 | **0.39** | 5.16 | 8.14 | **0.63** | 4.64 | 1.70 |
| VVC | ours 1080p 3M | 10.17 | 21.63 | **0.47** | 22.65 | 32.52 | **0.70** | 17.16 | 1.69 |
| AV2 | ours 720p | 2.44 | 6.28 | **0.39** | 6.20 | 19.88 | **0.31** | 4.13 | 1.69 |
| AV2 | ours 1080p 3M | 7.81 | 23.55 | **0.33** | 19.83 | 62.31 | **0.32** | 14.71 | 1.88 |

**Summary by codec.**
- **H.264, HEVC, VVC and AV2** are at or faster than their reference on both platforms. Against FFmpeg, H.264 is
  0.88–1.00 on x86 and 0.77–0.85 on the Jetson. HEVC is 0.57–0.94 and 0.72–0.92.
- **AV1** is still slower than dav1d. On our own streams it is 1.09–1.27x. On third-party streams it is 1.32–1.58x,
  because of compound, warped motion, masks and loop restoration (which our encoder does not use).
- **VP9** is 1.14–1.41x vpxdec but level with FFmpeg's VP9 decoder (0.98–1.10x).
- **WASM costs 1.2–2.1x native.** The median is about 1.7x, higher than the 1.4x of 2026-11-10, because the
  native AVX2 paths gained more than SIMD128 did.

Per-run data: `~/nanocodec-data/work/dec-winddown/` (`xdec-final{,2}.json` x86, `jdec-final.json` Jetson,
`streams.tsv`, `xdec.py`, `table.py`). The Jetson harness is `~/speed-audit/jdec2.py` on the board.

## Aggregates (frame-weighted over the three clips)

"Demo path" is what the demo worker pays per frame with the CPU display path: decode plus the I420-to-RGBA
hand-off out of WASM memory (the WebGL path hands off I420 instead, about 0.15–0.3 ms per 720p frame).

| Codec | Res | Module | Decode-only fps | Demo path fps (+ RGBA hand-off) | Native fps | WASM/native time | Achieved kbps |
|---|---|---|---:|---:|---:|---:|---:|
| H.264 | 720p | relaxed | 650 | 453 | 879 | 1.35x | 633 |
| H.264 | 720p | simd128 | 657 | 458 | 879 | 1.34x | 633 |
| H.264 | 360p | relaxed | 1774 | 1349 | 2416 | 1.36x | 318 |
| H.264 | 360p | simd128 | 1774 | 1348 | 2416 | 1.36x | 318 |
| HEVC | 720p | relaxed | 428 | 334 | 589 | 1.37x | 638 |
| HEVC | 720p | simd128 | 426 | 330 | 589 | 1.38x | 638 |
| HEVC | 360p | relaxed | 1177 | 970 | 1597 | 1.36x | 319 |
| HEVC | 360p | simd128 | 1167 | 961 | 1597 | 1.37x | 319 |
| AV1 | 720p | relaxed | 276 | 231 | 412 | 1.49x | 638 |
| AV1 | 720p | simd128 | 249 | 206 | 412 | 1.65x | 638 |
| AV1 | 360p | relaxed | 727 | 633 | 1117 | 1.54x | 318 |
| AV1 | 360p | simd128 | 724 | 636 | 1117 | 1.54x | 318 |
| VP9 | 720p | relaxed | 336 | 273 | 479 | 1.43x | 648 |
| VP9 | 720p | simd128 | 329 | 264 | 479 | 1.45x | 648 |
| VP9 | 360p | relaxed | 942 | 798 | 1351 | 1.43x | 322 |
| VP9 | 360p | simd128 | 910 | 766 | 1351 | 1.48x | 322 |
| AV2 | 720p | relaxed | 224 | 189 | 399 | 1.78x | 645 |
| AV2 | 720p | simd128 | 235 | 200 | 399 | 1.70x | 645 |
| AV2 | 360p | relaxed | 677 | 604 | 1111 | 1.64x | 325 |
| AV2 | 360p | simd128 | 668 | 595 | 1111 | 1.66x | 325 |
| VVC | 720p | relaxed | 214 | 187 | 251 | 1.17x | 662 |
| VVC | 720p | simd128 | 213 | 186 | 251 | 1.18x | 662 |
| VVC | 360p | relaxed | 585 | 529 | 774 | 1.32x | 320 |
| VVC | 360p | simd128 | 588 | 531 | 774 | 1.32x | 320 |

| Codec | Preset | Res | Module | WASM encode fps | Native fps | WASM/native time | Achieved kbps (WASM) |
|---|---|---|---|---:|---:|---:|---:|
| H.264 | Default | 720p | relaxed | 89.1 | 179.5 | 2.01x | 602 |
| H.264 | Default | 720p | simd128 | 83.2 | 179.5 | 2.16x | 602 |
| H.264 | Default | 360p | relaxed | 283.1 | 538.8 | 1.90x | 299 |
| H.264 | Default | 360p | simd128 | 280.5 | 538.8 | 1.92x | 299 |
| H.264 | Fast | 720p | relaxed | 96.3 | 224.3 | 2.33x | 599 |
| H.264 | Fast | 720p | simd128 | 95.2 | 224.3 | 2.36x | 599 |
| H.264 | Fast | 360p | relaxed | 333.3 | 688.3 | 2.07x | 301 |
| H.264 | Fast | 360p | simd128 | 340.4 | 688.3 | 2.02x | 301 |
| HEVC | Default | 720p | relaxed | 46.9 | 85.9 | 1.83x | 601 |
| HEVC | Default | 720p | simd128 | 46.9 | 85.9 | 1.83x | 601 |
| HEVC | Default | 360p | relaxed | 149.7 | 254.2 | 1.70x | 305 |
| HEVC | Default | 360p | simd128 | 144.9 | 254.2 | 1.75x | 305 |
| HEVC | Fast | 720p | relaxed | 71.9 | 155.4 | 2.16x | 609 |
| HEVC | Fast | 720p | simd128 | 72.2 | 155.4 | 2.15x | 609 |
| HEVC | Fast | 360p | relaxed | 260.1 | 507.4 | 1.95x | 303 |
| HEVC | Fast | 360p | simd128 | 252.0 | 507.4 | 2.01x | 303 |
| AV1 | Default | 720p | relaxed | 26.1 | 61.7 | 2.36x | 604 |
| AV1 | Default | 720p | simd128 | 25.7 | 61.7 | 2.40x | 604 |
| AV1 | Default | 360p | relaxed | 74.9 | 174.2 | 2.33x | 300 |
| AV1 | Default | 360p | simd128 | 71.6 | 174.2 | 2.43x | 300 |
| AV1 | Fast | 720p | relaxed | 46.2 | 137.0 | 2.96x | 599 |
| AV1 | Fast | 720p | simd128 | 45.2 | 137.0 | 3.03x | 599 |
| AV1 | Fast | 360p | relaxed | 160.3 | 433.8 | 2.71x | 301 |
| AV1 | Fast | 360p | simd128 | 149.7 | 433.8 | 2.90x | 301 |
| VP9 | Default | 720p | relaxed | 24.4 | 63.2 | 2.59x | 604 |
| VP9 | Default | 720p | simd128 | 24.7 | 63.2 | 2.56x | 604 |
| VP9 | Default | 360p | relaxed | 70.2 | 172.9 | 2.46x | 300 |
| VP9 | Default | 360p | simd128 | 70.2 | 172.9 | 2.46x | 300 |
| AV2 | Default | 720p | relaxed | 18.3 | 50.3 | 2.74x | 603 |
| AV2 | Default | 720p | simd128 | 17.8 | 50.3 | 2.83x | 603 |
| AV2 | Default | 360p | relaxed | 55.9 | 138.1 | 2.47x | 302 |
| AV2 | Default | 360p | simd128 | 55.7 | 138.1 | 2.48x | 302 |
| VVC | Default | 720p | relaxed | 35.4 | 67.5 | 1.91x | 610 |
| VVC | Default | 720p | simd128 | 34.8 | 67.5 | 1.94x | 610 |
| VVC | Default | 360p | relaxed | 111.8 | 190.9 | 1.71x | 299 |
| VVC | Default | 360p | simd128 | 109.9 | 190.9 | 1.74x | 299 |
| VVC | Fast | 720p | relaxed | 49.8 | 104.1 | 2.09x | 598 |
| VVC | Fast | 720p | simd128 | 49.8 | 104.1 | 2.09x | 598 |
| VVC | Fast | 360p | relaxed | 169.6 | 298.7 | 1.76x | 303 |
| VVC | Fast | 360p | simd128 | 168.0 | 298.7 | 1.78x | 303 |

Per clip (relaxed module; decode-only fps / native fps, encode fps / native fps):

| Direction | Codec | Preset | Res | Clip | WASM fps | Native fps |
|---|---|---|---|---|---:|---:|
| decode | H.264 | – | 720p | crowd_mot17_09 | 627 | 830 |
| decode | H.264 | – | 720p | pan_jaad_0117 | 565 | 811 |
| decode | H.264 | – | 720p | static_chirla_s000c3 | 799 | 1024 |
| decode | H.264 | – | 360p | crowd_mot17_09_360p | 1736 | 2323 |
| decode | H.264 | – | 360p | pan_jaad_0117_360p | 1600 | 2362 |
| decode | H.264 | – | 360p | static_chirla_s000c3_360p | 2041 | 2579 |
| decode | HEVC | – | 720p | crowd_mot17_09 | 430 | 582 |
| decode | HEVC | – | 720p | pan_jaad_0117 | 347 | 510 |
| decode | HEVC | – | 720p | static_chirla_s000c3 | 556 | 705 |
| decode | HEVC | – | 360p | crowd_mot17_09_360p | 1230 | 1648 |
| decode | HEVC | – | 360p | pan_jaad_0117_360p | 989 | 1436 |
| decode | HEVC | – | 360p | static_chirla_s000c3_360p | 1381 | 1739 |
| decode | AV1 | – | 720p | crowd_mot17_09 | 284 | 415 |
| decode | AV1 | – | 720p | pan_jaad_0117 | 202 | 331 |
| decode | AV1 | – | 720p | static_chirla_s000c3 | 415 | 539 |
| decode | AV1 | – | 360p | crowd_mot17_09_360p | 746 | 1146 |
| decode | AV1 | – | 360p | pan_jaad_0117_360p | 559 | 911 |
| decode | AV1 | – | 360p | static_chirla_s000c3_360p | 1000 | 1398 |
| decode | VP9 | – | 720p | crowd_mot17_09 | 333 | 461 |
| decode | VP9 | – | 720p | pan_jaad_0117 | 295 | 426 |
| decode | VP9 | – | 720p | static_chirla_s000c3 | 395 | 571 |
| decode | VP9 | – | 360p | crowd_mot17_09_360p | 884 | 1283 |
| decode | VP9 | – | 360p | pan_jaad_0117_360p | 930 | 1311 |
| decode | VP9 | – | 360p | static_chirla_s000c3_360p | 1022 | 1474 |
| decode | AV2 | – | 720p | crowd_mot17_09 | 201 | 388 |
| decode | AV2 | – | 720p | pan_jaad_0117 | 197 | 361 |
| decode | AV2 | – | 720p | static_chirla_s000c3 | 297 | 462 |
| decode | AV2 | – | 360p | crowd_mot17_09_360p | 663 | 1048 |
| decode | AV2 | – | 360p | pan_jaad_0117_360p | 572 | 1020 |
| decode | AV2 | – | 360p | static_chirla_s000c3_360p | 853 | 1306 |
| decode | VVC | – | 720p | crowd_mot17_09 | 213 | 244 |
| decode | VVC | – | 720p | pan_jaad_0117 | 195 | 236 |
| decode | VVC | – | 720p | static_chirla_s000c3 | 240 | 277 |
| decode | VVC | – | 360p | crowd_mot17_09_360p | 603 | 761 |
| decode | VVC | – | 360p | pan_jaad_0117_360p | 538 | 787 |
| decode | VVC | – | 360p | static_chirla_s000c3_360p | 622 | 773 |
| encode | H.264 | Default | 720p | crowd_mot17_09 | 88 | 207 |
| encode | H.264 | Default | 720p | pan_jaad_0117 | 85 | 167 |
| encode | H.264 | Default | 720p | static_chirla_s000c3 | 95 | 170 |
| encode | H.264 | Default | 360p | crowd_mot17_09_360p | 300 | 577 |
| encode | H.264 | Default | 360p | pan_jaad_0117_360p | 263 | 497 |
| encode | H.264 | Default | 360p | static_chirla_s000c3_360p | 289 | 549 |
| encode | H.264 | Fast | 720p | crowd_mot17_09 | 99 | 216 |
| encode | H.264 | Fast | 720p | pan_jaad_0117 | 86 | 208 |
| encode | H.264 | Fast | 720p | static_chirla_s000c3 | 106 | 254 |
| encode | H.264 | Fast | 360p | crowd_mot17_09_360p | 348 | 691 |
| encode | H.264 | Fast | 360p | pan_jaad_0117_360p | 322 | 670 |
| encode | H.264 | Fast | 360p | static_chirla_s000c3_360p | 331 | 705 |
| encode | HEVC | Default | 720p | crowd_mot17_09 | 47 | 84 |
| encode | HEVC | Default | 720p | pan_jaad_0117 | 41 | 80 |
| encode | HEVC | Default | 720p | static_chirla_s000c3 | 54 | 96 |
| encode | HEVC | Default | 360p | crowd_mot17_09_360p | 157 | 269 |
| encode | HEVC | Default | 360p | pan_jaad_0117_360p | 124 | 224 |
| encode | HEVC | Default | 360p | static_chirla_s000c3_360p | 178 | 277 |
| encode | HEVC | Fast | 720p | crowd_mot17_09 | 68 | 152 |
| encode | HEVC | Fast | 720p | pan_jaad_0117 | 68 | 133 |
| encode | HEVC | Fast | 720p | static_chirla_s000c3 | 81 | 192 |
| encode | HEVC | Fast | 360p | crowd_mot17_09_360p | 266 | 497 |
| encode | HEVC | Fast | 360p | pan_jaad_0117_360p | 240 | 471 |
| encode | HEVC | Fast | 360p | static_chirla_s000c3_360p | 277 | 563 |
| encode | AV1 | Default | 720p | crowd_mot17_09 | 26 | 76 |
| encode | AV1 | Default | 720p | pan_jaad_0117 | 24 | 62 |
| encode | AV1 | Default | 720p | static_chirla_s000c3 | 29 | 52 |
| encode | AV1 | Default | 360p | crowd_mot17_09_360p | 87 | 214 |
| encode | AV1 | Default | 360p | pan_jaad_0117_360p | 59 | 154 |
| encode | AV1 | Default | 360p | static_chirla_s000c3_360p | 85 | 165 |
| encode | AV1 | Fast | 720p | crowd_mot17_09 | 44 | 138 |
| encode | AV1 | Fast | 720p | pan_jaad_0117 | 39 | 122 |
| encode | AV1 | Fast | 720p | static_chirla_s000c3 | 61 | 155 |
| encode | AV1 | Fast | 360p | crowd_mot17_09_360p | 146 | 411 |
| encode | AV1 | Fast | 360p | pan_jaad_0117_360p | 140 | 417 |
| encode | AV1 | Fast | 360p | static_chirla_s000c3_360p | 213 | 480 |
| encode | VP9 | Default | 720p | crowd_mot17_09 | 23 | 67 |
| encode | VP9 | Default | 720p | pan_jaad_0117 | 22 | 61 |
| encode | VP9 | Default | 720p | static_chirla_s000c3 | 28 | 62 |
| encode | VP9 | Default | 360p | crowd_mot17_09_360p | 66 | 190 |
| encode | VP9 | Default | 360p | pan_jaad_0117_360p | 62 | 168 |
| encode | VP9 | Default | 360p | static_chirla_s000c3_360p | 86 | 164 |
| encode | AV2 | Default | 720p | crowd_mot17_09 | 18 | 60 |
| encode | AV2 | Default | 720p | pan_jaad_0117 | 17 | 51 |
| encode | AV2 | Default | 720p | static_chirla_s000c3 | 21 | 43 |
| encode | AV2 | Default | 360p | crowd_mot17_09_360p | 55 | 163 |
| encode | AV2 | Default | 360p | pan_jaad_0117_360p | 48 | 132 |
| encode | AV2 | Default | 360p | static_chirla_s000c3_360p | 67 | 125 |
| encode | VVC | Default | 720p | crowd_mot17_09 | 34 | 66 |
| encode | VVC | Default | 720p | pan_jaad_0117 | 33 | 63 |
| encode | VVC | Default | 720p | static_chirla_s000c3 | 39 | 74 |
| encode | VVC | Default | 360p | crowd_mot17_09_360p | 112 | 190 |
| encode | VVC | Default | 360p | pan_jaad_0117_360p | 104 | 177 |
| encode | VVC | Default | 360p | static_chirla_s000c3_360p | 120 | 208 |
| encode | VVC | Fast | 720p | crowd_mot17_09 | 48 | 103 |
| encode | VVC | Fast | 720p | pan_jaad_0117 | 46 | 96 |
| encode | VVC | Fast | 720p | static_chirla_s000c3 | 58 | 115 |
| encode | VVC | Fast | 360p | crowd_mot17_09_360p | 174 | 306 |
| encode | VVC | Fast | 360p | pan_jaad_0117_360p | 156 | 280 |
| encode | VVC | Fast | 360p | static_chirla_s000c3_360p | 182 | 312 |

## Native decode on the Jetson (2026-11-12)

The same 720p streams (crowd, pan), decoded natively on one Jetson Orin Nano core (Cortex-A78AE, NEON, pinned,
best of 3 interleaved rounds, head 1b2c955c), next to the single-threaded references there. ms per frame:

| Codec | Ours (crowd / pan) | Reference (crowd / pan) | Ours ÷ reference |
|---|---:|---|---:|
| H.264 | 2.18 / 2.36 | FFmpeg 2.68 / 2.91 | 0.81 |
| HEVC | 3.47 / 4.24 | FFmpeg 4.18 / 4.78 | 0.83–0.89 |
| AV1 | 6.04 / 7.99 | dav1d 4.75 / 6.22 | 1.27–1.28 |
| VP9 | 3.76 / 4.66 | vpxdec 2.74 / 3.60 | 1.30–1.37 |
| AV2 | 6.36 / 7.11 | avmdec 19.87 / 24.46 | 0.29–0.32 |
| VVC | 6.96 / 7.66 | VVdeC 8.17 / 9.66 | 0.79–0.85 |

1080p, camera clips and profiles: [competitor-yardsticks.md §3.8](competitor-yardsticks.md).

## Headless Chromium cross-check (decode)

Headless Chromium 148 (Playwright, `tools/web/display-bench.py`: a module worker with the relaxed module, CPU
display path, page draws every frame; the whole browser pinned to core 15), same streams, best of 3 rounds,
mean over the three clips, against node's relaxed decode-only time from the main run:

| Codec | Res | Chromium decode ms | node decode ms | Chromium / node | Chromium end to end ms |
|---|---|---:|---:|---:|---:|
| H.264 | 720p | 1.90 | 1.54 | 1.24 | 4.03 |
| H.264 | 360p | 0.57 | 0.56 | 1.02 | 1.06 |
| HEVC | 720p | 2.86 | 2.33 | 1.22 | 4.70 |
| HEVC | 360p | 0.93 | 0.85 | 1.09 | 1.46 |
| AV1 | 720p | 4.48 | 3.63 | 1.23 | 6.56 |
| AV1 | 360p | 1.30 | 1.38 | 0.95 | 1.83 |
| VP9 | 720p | 3.85 | 2.98 | 1.29 | 5.85 |
| VP9 | 360p | 1.02 | 1.06 | 0.96 | 1.53 |
| AV2 | 720p | 5.36 | 4.47 | 1.20 | 7.48 |
| AV2 | 360p | 1.50 | 1.48 | 1.01 | 1.98 |

At 360p Chromium's decode matches node within −5 to +9 %. At 720p it is 20–29 % slower: with the whole browser
on one core, the worker's decode shares it with the page's 3.7 MB-per-frame transfer and software-GL
(SwiftShader) compositing. "End to end" adds the hand-off, transfer to the page and the draw. A real browser on
an unpinned machine runs those in other processes, so node's single-core figures are the codec's cost.

## HEVC control in the VVC run

| | Decode 720p / 360p (relaxed, decode-only fps) | Encode Default 720p / 360p | Encode Fast 720p / 360p |
|---|---:|---:|---:|
| main run (load 8.7–17.9) | 428 / 1177 | 46.9 / 149.7 | 71.9 / 260.1 |
| VVC run (load 4.0–11.0) | 423 / 1187 | 52.0 / 159.6 | 79.2 / 282.9 |
| native, main run / VVC run | 589 / 1597 → 577 / 1619 | 85.9 / 254.2 → 102.6 / 265.6 | 155.4 / 507.4 → 178.3 / 518.8 |

Decode repeats within 2 %; encode ran 7–11 % faster in WASM and 2–19 % faster natively in the quieter VVC window,
so the VVC encode rows read a few percent fast next to the others, and absolute fps on this shared PC carry
about ±10 %. Ratios are steadier than absolute fps.

## HEVC vs Antare `hevc.wasm` (2026-11-12, hevc-wasm-speed)

Antare's numbers are their retest (pinned core, interleaved rounds) of their `hevc.wasm` and of our module
at abcf696b, on their machine. We do not have their module, so "ours now" is their measurement of our
abcf696b scaled by our own base→head ratio: WASM decode-only ms/frame, relaxed module, `tools/web/bench-runner.mjs`
pinned to one reserved core (14/15), interleaved base/head rounds, best round (PC load ~20 throughout, so ratios,
not absolutes, are the reliable part). Head = f21003cc (+ the CABAC table commit, not in the WASM timing).
The x265 rows use our own x265 encodes (`--preset medium` 1080p 2.5 Mbit/s, `--preset fast` 4K of the Hanwha
clip, default B-frames and weighted prediction), not Antare's files, so their scaled values are indicative.

| Row | Antare | ours abcf696b (their run) | ours base→head here | ours now (scaled) | vs Antare |
|---|---:|---:|---:|---:|---:|
| Novatek bodycam 2460x1848, P-only | 7.9–8.4 | 11.1–12.1 | 33.0 → 23.7 (−28 %) | ~8.3 | +3 % |
| x265 1080p, B-frames | 6.7–6.8 | 9.4–9.5 | 62.1 → 24.1 (−61 %) | ~3.7 | −45 % |
| x265 4K, B-frames | 25.9–26.3 | 39.1–40.1 | 80.1 → 41.0 (−49 %) | ~20 | −23 % |
| our encoder, 720p 600 kbit/s | 1.84 | 1.98 | 2.54 → 2.47 (−3 %) | ~1.93 | +5 % |
| our encoder, 1080p 3 Mbit/s | 6.38 | 7.11 | 10.93 → 10.62 (−3 %) | ~6.9 | +8 % |

Native against `ffmpeg -threads 1` (callgrind instructions, deterministic under load): Novatek bodycam ours 2.79G
against FFmpeg's 2.26G for the whole process. On our 720p stream the remaining time splits (callgrind, head):
deblocking 20 % (luma kernel 9 %, boundary strengths and loops 8 %), inter prediction 17 % (kernels 12 %),
SAO 4 % (edge kernel), inverse transforms 3.6 %, CABAC and residual parsing about 8 %. The kernels (HEVC
compound prep SIMD, row-band deblocking, SAO edge) are Astra's queue (`feature/decoder-kernels`).
All five Antare clips decode bit-exactly against their `.framehash` files (every picture), before and after.

### Final re-time (2026-11-13, dec-winddown)

The same method at head d007869b. Base (abcf696b, `hevc-wasm-speed/pkg-base`) and head relaxed modules were
interleaved in the same 7 rounds as the close-out table above, on core 14 (loaded host, see there), best round
each. "Ours now" is Antare's measurement of our abcf696b times our base-to-head ratio.

| Row | Antare | ours abcf696b (their run) | ours base→head here (ms) | ours now (scaled) | vs Antare |
|---|---:|---:|---:|---:|---:|
| Novatek bodycam 2460x1848, P-only | 7.9–8.4 | 11.1–12.1 | 25.3 → 14.35 (−43 %) | ~6.3–6.9 | **−20 %** |
| x265 1080p, B-frames | 6.7–6.8 | 9.4–9.5 | 56.5 → 11.4 (−80 %) | ~1.9 | **−72 %** |
| x265 4K, B-frames | 25.9–26.3 | 39.1–40.1 | 63.7 → 18.8 (−70 %) | ~11.6–11.9 | **−55 %** |
| our encoder, 720p 600 kbit/s | 1.84 | 1.98 | 2.92 → 2.08 (−29 %) | ~1.41 | **−23 %** |
| our encoder, 1080p 3 Mbit/s | 6.38 | 7.11 | 12.34 → 9.27 (−25 %) | ~5.34 | **−16 %** |

We are now ahead of Antare's `hevc.wasm` on all five rows. The x265 rows still use our own x265 encodes, so
they are indicative. With the round spread on this host, read each row's margin as about ±10 points. The
Novatek and 4K rows were the noisiest cells.

## AV1 native decode against dav1d (2026-11-12, av1-dec-speed)

`tools/web/bench.py --codecs av1 --targets 720p=600,360p=300 --presets default --skip-encode
--repeat 2 --rounds 3`, run without pinning: the timing cores were not available to this agent.
Load was 13–18, so treat the absolute fps as pessimistic and compare ratios.
- The reference column is `dav1d --threads 1` on the same streams.
- "Base" is abcf696b; "now" is bbb1c255, and the table shows the better of two interleaved "now"
  runs.
- Both used the same streams: the current encoder's, under `work/av1-dec-speed/bench-streams`.
- Every WASM output hash-matched native.

Frame-weighted over the three clips (crowd, pan, static):

| Res | Build | Native fps | dav1d ms/f | Native/dav1d | WASM ms/f (simd128) | WASM/native |
|---|---|---:|---:|---:|---:|---:|
| 720p | base | 382 | 1.60 | 1.64x | 5.23 | 1.60x |
| 720p | now | 452 | 1.52 | 1.45x | 4.20 | 1.58x |
| 360p | base | 1179 | 0.53 | 1.60x | 1.46 | 1.48x |
| 360p | now | 1336 | 0.53 | 1.42x | 1.17 | 1.40x |

Deterministic counts (callgrind) back the wall-time ratios:
- Native instructions: crowd720 -8.9% (4.16G → 3.79G against dav1d's 2.61G); 1080p 3 Mbit/s
  -16.8% (14.46G → 12.03G against 8.32G).
- WASM instructions (callgrind on node, TurboFan only): crowd720 -9.4%, static720 -3.5%.

The changes and the remaining gap (deblocking, MC wrappers, CDEF, coefficients) are in
the AV1 decoder notes (internal, not published), session 2026-11-12.

## H.264 decoder vs ffmpeg and other WASM H.264 decoders (2026-11-12, h264-dec-speed)

**Builds.**
- Before: f2f7a221 (`nanocodec-base`, the build in the tables above).
- After: 685bce33 (variance-integration, after the h264-dec-speed landings).
- Native: profiling profile, which is release code generation plus debug info.
- WASM: the relaxed module.

**Streams.** 150 frames each, in `~/nanocodec-data/work/h264-dec-speed/streams/`:
- Ours: crowd_mot17_09 720p 600 kbps and crowd_pp22_189 1080p 3 Mbit/s (Default preset, CABAC).
- x264 and openh264 (through ffmpeg's libx264/libopenh264), 1080p 3 Mbit/s, CABAC and CAVLC, no B frames.
- 3 Mbit/s is ≤ 50 % of the 6.3 Mbit/s source.

**Exactness.** Every build's per-frame MD5 equals ffmpeg's on all six streams. H.264 conformance passes 65/65. The WASM I420 SHA-256 equals native on every stream.

**Native: CPU time vs `ffmpeg -threads 1 -f null`.**
- Setup: anaconda ffmpeg 7.1.1, single-threaded, best of 9 interleaved rounds, on timing core 14 under `tools/dev/host-timed.sh`.
- Ours writes I420 to /dev/null, which ffmpeg's null muxer does not.
- ffmpeg's ~25 ms of process start (dynamic linking) is included.

| Stream | ffmpeg ms | before ms | after ms | before / ffmpeg | after / ffmpeg |
|---|---:|---:|---:|---:|---:|
| ours 720p 600k | 168 | 187 | 139 | 1.11x | **0.83x** |
| ours 1080p 3M | 497 | 654 | 488 | 1.32x | **0.98x** |
| x264 CABAC 1080p 3M | 632 | 883 | 678 | 1.40x | 1.07x |
| x264 CAVLC 1080p 3M | 476 | 737 | 545 | 1.55x | 1.14x |
| openh264 CABAC 1080p 3M | 461 | 691 | 522 | 1.50x | 1.13x |
| openh264 CAVLC 1080p 3M | 375 | 605 | 416 | 1.61x | 1.11x |

**WASM decode ms/frame vs other WASM H.264 decoders.**
- Ours: `bench-runner.mjs` decode-only, best of 3 interleaved rounds, pinned to core 14.
- ffmpeg.wasm: `@ffmpeg/core` 0.12.10 (single-threaded ffmpeg CLI, `-f null`, best of 2).
- tinyh264 0.0.7 (h264bsd; Constrained Baseline only, so only the openh264 CAVLC stream).
- All run under node 22.

| Stream | ours before | ours after | ffmpeg.wasm | tinyh264 |
|---|---:|---:|---:|---:|
| ours 720p 600k | 1.73 | **1.51** | 2.59 | – |
| ours 1080p 3M | 7.27 | **6.23** | 9.45 | – |
| x264 CABAC 1080p 3M | 9.47 | **8.08** | 13.03 | – |
| x264 CAVLC 1080p 3M | 10.39 | **7.84** | 11.69 | – |
| openh264 CABAC 1080p 3M | 7.60 | **6.58** | 8.40 | – |
| openh264 CAVLC 1080p 3M | 6.62 | **5.57** | 7.49 | 6.28 |

**What remains vs ffmpeg 8** (callgrind, symbolized from Ubuntu debuginfod; per stream in our internal notes (not published) "H.264 decoder"):
- Deblocking: +370–390M instructions on the x264/openh264 streams. The control side is ours; the run kernel is Astra's (`work/runtraces/h264/`).
- Branch mispredicts: about 32M vs 13M on openh264 CAVLC; 7.5M are in residual parsing, now a separate agent's.
- Per-macroblock bookkeeping: +130–190M.
- Inverse transform + add: +40–110M.
- Intra: +70–110M.

## VP9 decoder against the references (2026-11-12, vp9-dec-speed)

Native: CPU time (user + system) of one decode of 150 frames to `/dev/null`, pinned to core 14
under `tools/dev/host-timed.sh`, best of 7 interleaved rounds (load ~15-30 elsewhere). Before =
variance-integration 461ae318 (start of this work); after = 4b75de37. References: libvpx 1.16.0
`vpxdec --threads=1 --noblit` and FFmpeg 8.0.1 `ffmpeg-decode -threads 1 ... -f rawvideo /dev/null`.
WASM: `tools/web/bench-runner.mjs decode`, relaxed module, decode-only ms per frame, best of 3
rounds x 3 passes, core 14. All outputs identical before and after (frame MD5s), VP9 conformance
303/303.

| Stream | Ours before (ms/frame) | Ours after | vpxdec | ffmpeg | After / ffmpeg | After / vpxdec | WASM before | WASM after | WASM / native after |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| crowd 720p 600k (ours) | 2.18 | 1.69 | 0.99 | 1.31 | 1.28x | 1.70x | 3.25 | 2.34 | 1.38x |
| pan 720p 600k (ours) | 2.34 | 1.82 | 1.11 | 1.41 | 1.29x | 1.63x | 3.66 | 2.86 | 1.57x |
| static 360p 300k (ours) | 0.69 | 0.55 | 0.31 | 0.43 | 1.26x | 1.78x | 0.95 | 0.61 | 1.12x |
| crowd 1080p 3M (ours) | 7.31 | 5.83 | 3.64 | 4.68 | 1.25x | 1.60x | 13.6 | 9.86 | 1.69x |
| crowd 1080p 3M (vpxenc) | 8.53 | 7.02 | 4.50 | 5.84 | 1.20x | 1.56x | 14.6 | 10.7 | 1.52x |

Instructions (callgrind, crowd 720p / vpxenc 1080p): 3.80G / 10.94G before, 2.74G / 8.65G after;
libvpx 1.84G / 5.70G. Per-stage gaps and what was tried are in our internal notes (not published)
("VP9 decoder speed", "VP9 MC kernels"). The WASM / native ratio rose on the 1080p rows because
the native AVX2 gains (MC row pairs, fused inverse) have no SIMD128 counterpart at V8's register
limit.

Jetson Orin Nano (Cortex-A78AE, NEON; core 3 under `jetson-timed.sh`, best of 3, after =
4b75de37 release build; ffmpeg-decode and vpxdec from the aarch64 anchors):

| Stream | Ours (ms/frame) | vpxdec | ffmpeg | Ours / ffmpeg | Ours / vpxdec |
|---|---:|---:|---:|---:|---:|
| crowd 720p 600k (ours) | 3.73 | 2.72 | 3.16 | 1.18x | 1.37x |
| crowd 1080p 3M (ours) | 13.5 | 10.6 | 11.4 | 1.18x | 1.27x |
| crowd 1080p 3M (vpxenc) | 14.1 | 11.6 | 12.4 | 1.14x | 1.21x |

### Round 2 (2026-11-12, vp9-dec-speed2)

After a74d5cb8 (libvpx eob-class inverse + add, sliding AVX2 vertical MC, no per-frame zeroing)
and its follow-up "deblocking views per plane row, recycled deblocking units, inline intra stores".
CPU time (user + sys, `getrusage`) of one decode, best of 5 (Jetson) or 5 (x86, core 14 under
`host-timed.sh`, machine load ~40 elsewhere, so x86 ratios carry a few percent of noise);
"before" is this work's base 11fcb4e3 on the Jetson and the older 461ae318 on x86.
Script: `~/nanocodec-data/work/vp9-dec-speed2/tools/cputime.py`.

| Stream | Jetson before / ffmpeg | Jetson after / ffmpeg | x86 after / ffmpeg |
|---|---:|---:|---:|
| crowd 720p 600k (ours) | 1.18x | 1.10x | 1.09x |
| pan 720p 600k (ours) | 1.17x | 1.12x | 1.09x |
| static 360p 300k (ours) | 1.23x | 1.13x | 1.00x |
| crowd 1080p 3M (ours) | 1.17x | 1.13x | 1.11x |
| crowd 1080p 3M (vpxenc) | 1.14x | 1.09x | 1.18x |

Instructions (callgrind, crowd 720p): 2.72G -> 2.41G (libvpx 1.84G); inverse + add 214M -> 75M,
MC 409M -> 327M. WASM decode-only (relaxed module, node, core 14, ms/frame, previous agent's
module -> this one): crowd 720p 2.45 -> 2.24, pan 2.87 -> 2.94, static 0.63 -> 0.63, crowd 1080p
9.39 -> 9.98, vpxenc 1080p 13.4 -> 12.6 (within the loaded host's noise except crowd 720p): the
gains are AVX2-side; SIMD128 MC is unchanged (see our internal notes (not published), "VP9 MC kernels").
