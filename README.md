# nanocodec

An experimental, low-delay video encoder for live camera footage, written in Rust, with matching decoders that
also run in a web page as WebAssembly. One shared engine drives six formats: **H.264, HEVC, AV1, VP9, AV2 and
VVC (H.266)**. It aims at streaming from small edge devices (cameras, bodycams, drones, robots) at low bitrate
to the cloud or a browser.

**[Try the browser demo](https://cribbinsb.github.io/nanocodec/)**: compare two encoders side by side on your
webcam, a sample clip or a test scene. Encoding and decoding run in your browser.

> **Built on others' work.** Most of what works here was learned from, or ported from, open-source codecs and
> the standards behind them: above all [SVT-AV1](https://gitlab.com/AOMediaCodec/SVT-AV1), whose real-time
> pipeline is our main design guide, and also [libaom](https://aomedia.googlesource.com/aom/),
> [dav1d](https://code.videolan.org/videolan/dav1d), [libvpx](https://chromium.googlesource.com/webm/libvpx),
> [openh264](https://github.com/cisco/openh264), [kvazaar](https://github.com/ultravideo/kvazaar),
> [VVenC](https://github.com/fraunhoferhhi/vvenc) / [VVdeC](https://github.com/fraunhoferhhi/vvdec),
> [AVM](https://gitlab.com/AOMediaCodec/avm), [Thor](https://github.com/cisco/thor),
> [libyuv](https://chromium.googlesource.com/libyuv/libyuv), [libvmaf](https://github.com/Netflix/vmaf) and the
> JM / HM / VTM reference software; and on the ITU-T, ISO/IEC and Alliance for Open Media experts who wrote the
> standards. The coding tools are the standards'; the ideas are largely those projects'. See
> [Acknowledgements](#acknowledgements) and [THIRD_PARTY.md](THIRD_PARTY.md).

> **How it was made.** The code was written and is maintained by AI coding agents (mostly Anthropic's Claude,
> also OpenAI's Codex) working under the direction of one person. How the code is checked, and what that does
> and does not guarantee, is in the [FAQ](docs/FAQ.md#ai-authorship).

> **Notice.** Experimental software, provided "as is". The formats it implements are covered by third-party
> patents: neither this code nor the demo gives you any patent licence.
> **[Credits, licences, patents and disclaimers](docs/LEGAL.md)**.

**Read more:** [questions a sceptic would ask (FAQ)](docs/FAQ.md) · [how it is built](docs/ARCHITECTURE.md) ·
[the story of building it](docs/RETROSPECTIVE.md) · [detailed results and method](docs/results/README.md).

## Scope: what it does and what it leaves out

nanocodec covers a deliberately narrow subset of each standard, for one use case:

- **Low delay:** forward prediction only (no B-frames, no lookahead), one slice / tile per frame, 8-bit 4:2:0
  only, single-threaded.
- **Real-time CBR** with a 1 s buffer, plus loss resilience (temporal layers, long-term references, intra refresh,
  key frame on request). The controller treats the target as a ceiling and may spend less. The first key frame
  may borrow up to about 1.4 buffers and repays it over the next seconds (10 s by default), so the strict 1 s
  leaky bucket is exceeded at the start of a stream, and over the chart's 150-frame clips our streams end a median
  3–5% above target. Strict settings (`rate.recode_overflow=1`, `rate.recode_overflow_inter=1`,
  `rate.key_cap_buffer=1`, `rate.key_repay_seconds=1`) keep every chart point inside the bucket and cost 4.7–6.9%
  more bits at equal quality (see [rate control](docs/FAQ.md#rate-control)).
- **Input frames that would not change the picture are not coded** (a "hold": exact or near-exact repeats, as in
  recompressed or frame-doubled sources); the reference encoders code every frame. No frame was held on any of the
  chart's points.
- **Its own decoders** decode the profiles listed under [Decoders](#decoders); the encoders emit only the subset
  of tools they use.

The projects we compare with are general-purpose, mature, multi-threaded encoders and decoders that cover the full
standards and many more use cases (offline, high bit depth, lookahead, B-frames, many threads). We measured only
their real-time settings on one thread; their other modes may do better on these clips. nanocodec had the
advantage of being built after them, with their code to learn from.

## Quality and speed

The charts compare every encoder in its own real-time rate control at the same bitrate targets, on one CPU core.

![Bitrate saving vs encode speed, one Jetson Orin Nano core](docs/results/bd-vs-speed.svg)

![Bitrate saving vs encode speed, one Ryzen 9 5950X core](docs/results/bd-vs-speed-x86.svg)

<sub>**What is measured.** 720p (one 1884×720) at 24–30 fps, 150 frames per clip with one key frame (5 s at 30 fps; see
[window length](#window-length): longer windows give smaller savings); CBR targets
of 150–1200 kbps (scaled to each clip's pixel rate and capped at half the source's bitrate), 1 s buffer. Quality
axis: BD-rate on the achieved bitrates against x264 0.165 `--preset medium --tune zerolatency --bframes 0
--rc-lookahead 0 --threads 1` (ABR with a 1 s VBV). "BD-J" is our composite: the mean of the BD-rates on
PSNR-YUV, VMAF (model v1.0.16) and MS-SSIM, averaged over content classes and over the full and lower-rate
halves of the curves; plain per-metric numbers are in the [full table](docs/results/bd-vs-speed.md). Speed axis:
single-thread encode CPU time at a fixed quantizer near 600 kbps (not in CBR mode; our encoders take a median
about 2% more CPU in CBR, 0–7% by preset; the references' CBR cost was not measured). Jetson: one Cortex-A78AE core
locked at 1.728 GHz. PC: one Ryzen 9 5950X core on a shared, loaded machine (load ~40–60), each run normalised
by x264 `medium` runs in the same round, so read PC differences of a few percent as noise. Our points are scored
on the encoder's reconstruction; the reference encoders' on their decoded streams. Builds, dates and every
command line: [bd-vs-speed.md](docs/results/bd-vs-speed.md).</sub>

**The chart's clips and held-out clips.** The chart uses 11 natural camera clips (our VAL set). These clips were
also used to rank candidate changes during development, so the chart is not independent of the tuning. We
therefore re-measured every encoder on 71 clips from 21 other datasets
([held-out check](docs/results/heldout-chart-2026-11-12.md), same builds as the chart of 2026-11-12):

| Encoder (settings as above; 150 frames, one key frame) | BD-J, chart's 11 clips | BD-J, 71 held-out clips | fps, Jetson core | fps, PC core |
|---|---:|---:|---:|---:|
| nanocodec VVC Default | −53.4% | −46.3% | 27.5 | 67.5 |
| nanocodec AV2 Default | −53.6% | −44.5% | 31.5 | 85.5 |
| nanocodec AV1 Default | −50.9% | −42.5% | 41.1 | 102.6 |
| nanocodec HEVC Default | −49.3% | −41.7% | 51.0 | 121.0 |
| nanocodec VP9 Default | −47.9% | −41.2% | 34.4 | 94.2 |
| nanocodec AV1 Fast | −45.2% | −35.9% | 65.2 | 165.8 |
| SVT-AV1 v4.3.0 M8 (one thread) | −45.6% | −35.5% | 14.3 | 45.2 |
| x265 4.2 `medium` (zero latency, no B-frames, one thread) | −40.7% | −28.0% | 8.0 | 25.6 |
| SVT-AV1 v4.3.0 M12 `--rtc` (one thread) | −34.9% | −20.4% | 70.9 | 182.1 |
| libaom v3.15.3 realtime `--cpu-used=8` (one thread) | −28.7% | −23.0% | 17.4 | 42.1 |
| nanocodec H.264 Default | −20.4% (−23.0% at the current build) | −17.9% | 74.4 | 175.9 |
| libvpx v1.16.0 VP9 realtime `--cpu-used=9` (one thread) | −5.7% | −0.1% | 47.8 | 113.0 |
| x264 0.165 `medium` (the anchor above) | 0 | 0 | 26.7 | 67.6 |
| x264 0.165 `veryfast` (same settings) | +2.3% | +8.1% | 85.4 | 206.4 |
| openh264 v2.7.0 realtime, medium complexity | +24.6% | +19.7% | 111.5 | 260.3 |

<sub>Negative: fewer bits than x264 `medium` for the same quality. BD-J columns from the held-out check
(2026-11-12 builds); fps from the [current chart](docs/results/bd-vs-speed.md) (2026-11-13; H.264 Default has
since moved to −23.0% on the chart's clips). The held-out clips are stored high-quality encodes of graded
sources, not raw captures; since 2026-11-11 the same clip pool has also fed our rotating adoption screens, so
they are held out from most, not all, tuning.</sub>

How to read this: on the held-out clips every encoder's saving against x264 `medium` is smaller, by a median of
7.6 points, ours and the references alike: x264 does relatively better there (likely in part because many
held-out sources were themselves H.264-encoded; we have not proven the cause). Relative to the other encoders,
our presets keep their ranks within about one place; the size of any saving against x264 depends on the clips.
Where the references are ahead: SVT-AV1 M12 and M13 `--rtc`, x264 `veryfast` / `ultrafast` and openh264 encode
faster than our AV1, HEVC and VVC Defaults; our VVC Default is below 30 fps at 720p on the Jetson core; x264
`medium` is closer to us on the held-out set. Per-metric numbers, per-clip spread and known weaknesses:
**[docs/results/](docs/results/README.md)**.

**The anchor's settings matter.** x264's and x265's defaults are tuned for how the picture looks (psychovisual
optimisations), which PSNR, VMAF and MS-SSIM cannot judge, and no subjective test was run here; our presets were
tuned on exactly that metric mix. So every x264 and x265 preset was also measured at the metric-oriented tunings
its own documentation provides (`--tune psnr`, `--tune ssim`), and libaom / libvpx at `--tune ssim`. Where a tuning
is distinguishably better for its encoder (direct comparison, 95% interval excluding zero), the charts plot it
beside the default point (lightly filled, joined by a dashed line; its fps is the default run's until re-timed):
x264 medium and x265 medium / superfast with `--tune psnr`, libvpx rt 9 with `--tune ssim`. The other variants
were within the noise of their defaults; all are in the [table](docs/results/bd-vs-speed.md#metric-tuned-references).
The AV1 and VP9 references already run their PSNR tunes. x264 `medium --tune psnr` needs 9.8% fewer bits than the chart's anchor on
BD-J (PSNR-YUV −8.3%, VMAF v1 −19.4%, MS-SSIM −1.8%); x265 `medium --tune psnr` moves from −40.7% to −44.1%.
Against the PSNR-tuned x264 `medium`, every saving is smaller, and our H.264 Faster and Fastest presets save no
bits:

| Encoder | BD-J vs x264 `medium` (default psy settings, the chart's anchor) | BD-J vs x264 `medium --tune psnr` |
|---|---:|---:|
| nanocodec VVC Default | −54.0% | −47.1% |
| nanocodec AV2 Default | −53.7% | −46.8% |
| nanocodec AV1 Default | −50.9% | −43.6% |
| nanocodec HEVC Default | −49.3% | −41.7% |
| nanocodec VP9 Default | −48.1% | −40.2% |
| nanocodec AV1 Fast | −44.9% | −36.7% |
| x265 4.2 `medium` | −40.7% | −31.6% |
| SVT-AV1 v4.3.0 M12 `--rtc` | −34.9% | −25.1% |
| nanocodec H.264 Default | −23.0% | −13.3% (95% interval −20.7 to −5.4) |
| nanocodec H.264 Fast | −16.2% | −5.8% (−13.5 to +2.1) |
| nanocodec H.264 Faster | −8.5% | +2.5% (−5.0 to +9.8) |
| nanocodec H.264 Fastest | −0.8% | +10.7% (+3.8 to +17.6) |
| x264 0.165 `medium --tune psnr` itself | −9.8% | 0 |
| openh264 v2.7.0 | +24.6% | +42.8% |

<sub>Chart's 11 clips, 150 frames, one key frame; intervals are 95% bootstrap intervals over clips within content
class (11 clips, so read them as approximate). `--tune ssim` changes the anchor by −1.4%.</sub>

#### Window length

Every BD number on this page comes from 150-frame windows with one key frame (5 s at 30 fps),
and our presets were tuned on such windows. That favours our encoders: in a 5 s window our first key frame takes
8–10% of the bits and our streams end 5–6% over target (the key frame's loan is not yet repaid), which shows up as
quality. On 5 clips (3 of them used for tuning, 2 from the chart set, so neither independent nor enough for
intervals), the same builds over 600 frames (20 s, key frames at 0 and 300) gave a smaller saving for all three
codecs tested, with the rate on target and the first frame at 2–3% of the bits:

| Encoder | BD-J vs x264 `medium`, 150 frames | BD-J vs x264 `medium`, 600 frames | vs x264 `medium --tune psnr`, 600 frames |
|---|---:|---:|---:|
| nanocodec AV1 Default | −54.7% | −50.6% | −42.0% |
| nanocodec HEVC Default | −52.2% | −46.7% | −37.6% |
| nanocodec H.264 Default | −24.5% | −15.9% | −4.1% |

A chart on longer windows is planned. Until then, read the 150-frame numbers as favourable to us by several
points, most for H.264.

### Spatial scalability (SVC)

One AV1 or VP9 stream carrying a 360p30 base layer and the 720p30 picture (L2T1 / L2T3), so a receiver or relay
can drop to the base without a second encode: `--preset av1-svc` / `vp9-svc` and `--preset av1-svc-fast` /
`vp9-svc-fast`. AV1 also codes three layers, 180p30 + 360p30 + 720p30 (L3T1 / L3T3): `--preset av1-svc3` /
`av1-svc3-fast`. Rate is the **total** stream rate, quality the 720p output (BD-J vs single-layer x264 `medium`,
the scale above), on 8 of the chart's clips; speed is one core, timed in pairs with x264 `medium`.

![Spatial SVC: bitrate saving of the 720p output vs encode speed, one Ryzen 9 5950X core](docs/results/svc-chart.svg)

| Encoder | BD-J (720p, total rate) | base layer BD-J vs the codec's 360p alone | fps (Jetson, 1 core) | fps (x86, 1 core) |
|---|---:|---:|---:|---:|
| nanocodec AV1 SVC L2T3 (`av1-svc`, 3 temporal layers) | −46.6% | −8.6% | 25.5 | 57.9 |
| nanocodec AV1 SVC L2T1 (`av1-svc`) | −44.9% | −6.5% | 24.0 | 54.8 |
| nanocodec AV1 SVC Fast L2T1 (`av1-svc-fast`) | −40.1% | +7.0% | 42.4 | 78.4 |
| nanocodec VP9 SVC L2T3 (`vp9-svc`, 3 temporal layers) | −42.4% | −3.1% | 23.4 | 58.8 |
| nanocodec VP9 SVC L2T1 (`vp9-svc`) | −40.7% | −0.7% | 22.3 | 56.2 |
| nanocodec VP9 SVC Fast L2T1 (`vp9-svc-fast`) | −35.2% | +16.4% | 33.7 | 74.2 |
| nanocodec AV1 SVC L3T3 (`av1-svc3`, 3 temporal layers) | −41.7% | +17.5% (360p op) | 23.3 | 54.3 |
| nanocodec AV1 SVC L3T1 (`av1-svc3`) | −39.6% | +20.5% (360p op) | 22.3 | 51.9 |
| nanocodec AV1 SVC Fast L3T1 (`av1-svc3-fast`) | −33.7% | +37.2% (360p op) | 39.9 | 75.4 |
| nanocodec AV1 / VP9 simulcast 720p + 360p (two streams) | −33.3% / −29.6% | | | 72.3 / 73.2 |
| nanocodec AV1 simulcast 720p + 360p + 180p (three streams) | −33.7% | | | 72.7 |
| libaom v3.15.3 `svc_encoder_rtc` L3T1, speed 10 | −6.7% | +80.3% | 27.6 | 64.5 |
| libaom v3.15.3 `svc_encoder_rtc` L2T1, speed 10 | −8.7% | +42.6% | 30.6 | 68.6 |
| libaom v3.15.3 `svc_encoder_rtc` L2T3, speed 10 | +18.1% | +59.7% | 32.8 | 77.1 |
| libvpx v1.16.0 `vp9_spatial_svc_encoder` L2T1, speed 9 | +27.4% | +108.2% | 37.6 | 72.4 |
| libvpx v1.16.0 `vp9_spatial_svc_encoder` L2T1, speed 7 | +30.3% | +114.8% | 22.9 | 53.3 |

The libaom and libvpx rows use those projects' example SVC encoders (one thread, CBR, no frame drops) at the
settings listed in [svc-chart.md](docs/results/svc-chart.md); the examples demonstrate the API and are not tuned
products, and other settings may do better. libaom's three-temporal-layer rows split each spatial layer's rate
60/20/20% across temporal layers, as libaom's own tests do (re-measured 2026-11-13; with WebRTC's 40/20/40 they
had read +48.9%). On these clips and settings, libaom L2T1 at speed 10 is −8.7% and our AV1 SVC L2T1 −44.9%;
libaom L2T3 is +18.1% and ours −46.6%. Single-layer
AV1 / VP9 Default on the same clips: −53.0% / −50.1%. The three-layer streams' 180p operating point is +4.9%
(L3T1) against AV1 Default coded alone at 180p.

### In the browser (WebAssembly)

Frames per second of the browser build on one PC core (AMD Ryzen 9 5950X, shared machine, ±10%),
single-threaded (the WebAssembly module has no threads), under V8 (node 22, cross-checked in headless
Chromium). Decode: our CBR streams at 600 kbps (720p) and 300 kbps (360p). Encode: the demo's settings, Default /
Fast preset (VP9 and AV2 have no Fast in the demo).

<sub>Measured 2026-11-10, builds f2f7a221 (H.264–AV2) and bcbc5601 (VVC)</sub>

| Codec | Decode 720p | Decode 360p | Encode 720p | Encode 360p |
|---|---:|---:|---:|---:|
| H.264 | 650 | 1774 | 89 / 96 | 283 / 333 |
| HEVC | 428 | 1177 | 47 / 72 | 150 / 260 |
| AV1 | 276 | 727 | 26 / 46 | 75 / 160 |
| VP9 | 336 | 942 | 24 | 70 |
| AV2 | 224 | 677 | 18 | 56 |
| VVC | 214 | 585 | 35 / 50 | 112 / 170 |

WebAssembly took 1.2–1.8x the native time to decode and 1.7–3.0x to encode (2026-11-10 builds); with later
native decoder speed-ups, decode is 1.2–2.1x (median about 1.7x; 2026-11-13,
[decoder close-out](docs/results/wasm-speed.md#decoder-close-out-2026-11-13-dec-winddown)). On the five test rows
published with the `hevc.wasm` decoder, our HEVC WASM decoder took less time. Method and raw data:
[docs/results/](docs/results/README.md#in-the-browser-webassembly).

## Presets

Each codec has a **Default** (quality point) and a **Fast** preset (speed point):
`--preset h264-fast | hevc-fast | av1-fast | vp9-fast | av2-fast | vvc-fast`. Some codecs have further speed
points: `h264-faster`, `h264-fastest`, `hevc-faster`, `av1-faster`, `vp9-faster`, `vp9-fastest`. Measured on the
chart's 11 clips (150 frames, one key frame) and one Jetson core (fps), against the reference encoders at the
settings above:

- H.264 Fast: 114 fps vs openh264 (realtime, medium complexity) 111 fps; BD-J −25.6% against openh264.
- H.264 Faster: 134 fps; BD-J −8.5% vs x264 `medium` (+2.5% vs x264 `medium --tune psnr`, i.e. no saving) and
  −39.7% vs x264 `ultrafast` (191 fps).
- AV1 Faster: 81 fps vs SVT-AV1 M12 `--rtc` 71 fps; BD-J −7.4% against M12 (per-clip range −19.1% to +5.4%).
- HEVC Faster: 98 fps vs SVT-AV1 M13 `--rtc` 79 fps (PC: 0.89x M13's time in the same rounds); BD-J −4.4%
  against M13 (BD-PSNR-YUV +1.0%, i.e. no saving on PSNR alone).

AV1 adds an optional, experimental **film grain** option on top of any preset (`--film-grain [STRENGTH]`, the C
API's `film_grain`, a checkbox in the browser demo): decoders re-synthesize the texture coding smoothed away. It is
off by default, since it lowers PSNR and VMAF and suits stochastic texture best.

## Decoders

Our own Rust decoders, natively and in WebAssembly. 8-bit only (no 10-bit), single-threaded.

| Codec | What has been tested | Our decode time ÷ the reference's, x86 / Jetson |
|---|---|---|
| H.264 | Constrained High profile: 65/65 of the conformance streams we test in that profile, MD5-exact | FFmpeg: 0.88–1.00x / 0.77–0.85x |
| HEVC | Main profile: 136/136 of the conformance streams we test, exact | FFmpeg: 0.57–0.94x / 0.72–0.92x |
| AV1 | Main profile, 8-bit 4:2:0: 176/176 libaom test vectors | dav1d: 1.13–1.58x / 1.09–1.43x (dav1d faster) |
| VP9 | Profile 0: 303/303 test vectors | vpxdec: 1.32–1.41x / 1.14–1.24x (vpxdec faster; FFmpeg: 0.98–1.10x) |
| AV2 | Our own streams only, bit-exact with avmdec and dav2d | avmdec (AVM's research reference decoder, not built for speed): 0.33–0.39x / 0.31–0.32x |
| VVC | 8-bit Main: 9/9 official 8-bit conformance streams (ITU-T H.266.1) tested so far; not a full Main-profile conformance claim | VVdeC: 0.39–0.47x / 0.63–0.70x |

Ratios are single-threaded on the same streams (all references were run with one thread; they are normally run
multi-threaded): ours at 720p and 1080p, plus third-party streams (x264, openh264, x265 with B-frames, IP cameras,
SVT-AV1, libaom, vpxenc). Below 1 means ours took less time. The PC was a loaded Ryzen 9 5950X, so read x86 ratios
as about ±10 %; the Jetson is one Orin Nano core. Measured 2026-11-13 at d007869b
([table](docs/results/wasm-speed.md#decoder-close-out-2026-11-13-dec-winddown)). Decoder robustness: see
[FAQ: security](docs/FAQ.md#security-and-fuzzing).

**Encoder conformance.** No custom syntax is used. Before a build's numbers are published, short streams from
every shipped preset of every codec (temporal layers, holds, intra refresh, the main coding tools) are decoded
by two standard decoders per codec (H.264: JM and FFmpeg; HEVC: HM and FFmpeg; AV1: aomdec and dav1d; VP9:
vpxdec and FFmpeg; AV2: avmdec and dav2d; VVC: VVdeC and FFmpeg) and by ours, and must match the encoder's
reconstruction frame by frame (hashes); a randomized syntax-stress test forces random legal coding decisions.
The longer chart streams themselves are scored on the encoder's reconstruction and are not each decoded.

## Quick start

```sh
cargo build --release
cargo test --workspace
```

Encode raw I420 (or `--input-format yv12|nv12`) in real-time CBR, then decode:

```sh
target/release/nanocodec encode --codec av1 --input input.yuv --width 1280 --height 720 \
  --fps 30 --frames 150 --bitrate 600 --buffer-ms 1000 --output video.ivf
target/release/nanocodec decode --codec av1 --input video.ivf --output decoded.yuv
```

`--codec h264 | hevc | av1 | vp9 | av2 | vvc`. H.264, HEVC and VVC use Annex-B; AV1, VP9 and AV2 use IVF. Fixed
quantizer, temporal layers, timestamps, intra refresh and the rate-control options are in
[CLI usage](crates/nc-cli/README.md). Libraries expose `Decoder::decode(packet)`; a C API (static or dynamic
library) is in `crates/nc-capi`.

### Browser demo

**Live: https://cribbinsb.github.io/nanocodec/**. Two independent encoder/decoder workers process the same frames
(webcam, local video or the built-in scene), each with its own codec, bitrate or quantizer, preset and keyframe
interval. It shows the decoded pixels, bitrate, timing and PSNR (against the frame given to that encoder), and can
download the real streams. Your video is not uploaded. All six codecs run in the demo. A version can also play
streams encoded offline by other encoders at the chart's settings (named, with version and command line).

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-bindgen-cli --version 0.2.105 --locked
tools/web/build.sh
python3 -m http.server 8765 --bind 127.0.0.1 --directory web   # open http://localhost:8765
```

See [the demo guide](web/README.md).

## Project layout

```
crates/
  nc-common/     planes, bit I/O, YUV I/O, errors
  nc-dsp/        pixel kernels: scalar reference + AVX2 / NEON / WASM SIMD128 (with nc-capi, the only unsafe code)
  nc-analysis/   pyramid, motion estimation, pre-analysis, adaptive quantisation
  nc-encoder/    shared engine: mode decision, rate control, references, the CodecBackend trait
  nc-h264/ nc-hevc/ nc-av1/ nc-vp9/ nc-av2/ nc-vvc/   one backend + subset decoder per codec
  nc-capi/       C ABI          nc-cli/  command line          nc-web/  WASM bindings
  nc-checkasm/   SIMD-vs-reference tests and kernel benchmarks
eval/   evaluation harness       tools/   build, CI and dev scripts       web/   browser demo
```

Design: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). The story so far: [docs/RETROSPECTIVE.md](docs/RETROSPECTIVE.md).

## Acknowledgements

nanocodec would not exist without the people who built the open video ecosystem. Thank you to:

- the **SVT-AV1** team (Alliance for Open Media; originally Intel and Netflix), whose real-time pipeline is the
  model for our shared engine;
- the authors of **libaom**, **AVM**, **dav1d** (VideoLAN), **libvpx** (WebM Project), **openh264** (Cisco),
  **Thor** (Cisco), **kvazaar** (Ultra Video Group), **VVenC** and **VVdeC** (Fraunhofer HHI), **libyuv**, and
  of the **JM**, **HM** and **VTM** reference software, from which we learned, and in places ported, the
  algorithms credited module by module in [THIRD_PARTY.md](THIRD_PARTY.md);
- the experts of **ITU-T VCEG, ISO/IEC MPEG, JVT, JCT-VC, JVET and the Alliance for Open Media**, who designed
  H.264, HEVC, VVC, VP9, AV1 and AV2: every coding tool here is theirs;
- **x264, x265, FFmpeg, libaom, libvpx, SVT-AV1, openh264 and NVIDIA NVENC**, which we use as comparison points
  and, with JM, HM, VVdeC, dav1d, dav2d and aomdec, as independent decoders to check our streams;
- **Netflix** for VMAF and its open test sequences, the authors of **MS-SSIM** (Wang, Simoncelli and Bovik) and
  of the **BD-rate** method (Bjøntegaard);
- the providers of test material: **Xiph.org** (derf's collection), the **Blender Foundation**, **SVT (Sveriges
  Television)**, **Netflix**, **MOTChallenge** and the other datasets listed in the
  [clip attribution](web/clips/ATTRIBUTION.md).

What this project adds is narrow: one shared low-delay engine across six formats, a browser decode path for its
own streams, and an account of building it with AI agents.

## Licence, credits and patents

BSD 3-Clause; see [LICENSE](LICENSE). This is a copyright licence only: **it grants no patent rights, and using
nanocodec or its demo does not license you under any patent on H.264, HEVC, VVC, AV1, VP9 or AV2.** Obtaining
any patent licence your use needs is your responsibility.

nanocodec reimplements ideas and code from BSD-style and other permissively licensed projects; each is attributed
in [THIRD_PARTY.md](THIRD_PARTY.md). GPL and LGPL tools (x264, x265, FFmpeg) are run only as separate external
programs for comparison and verification; none of their code is copied into or distributed with nanocodec.

The full explanation, with links to the patent licensing bodies, is in
**[Credits, licences, patents and disclaimers](docs/LEGAL.md)**. It is general information, not legal advice.

## Contact

<bernandocribbenza@gmail.com>, or open an issue on GitHub.
