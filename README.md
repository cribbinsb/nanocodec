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
  key frame on request). The target is a long-run ceiling: from about 5 s on, the stream so far is at or under
  it, and undershoot is allowed. Single seconds can exceed it (a key-frame second or a static-scene refinement
  burst, up to about 2×), and the first key frame may borrow up to about 1.4 buffers, so a strict 1 s receiver
  buffer is exceeded at key frames. At the builds the charts were measured on (before this behaviour landed on
  2026-11-13) the key loan was repaid over 10 s, so over the chart's 150-frame clips our streams end a median 3–5%
  above target, and strict buffer settings cost our Defaults 4.7–6.9% more bits (about 2 points of BD-J; 3–4 on
  held-out clips). Details: [rate control](docs/FAQ.md#rate-control).
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

The charts compare every encoder in its own real-time rate control at the same bitrate targets, on one CPU core,
on **138 held-out clips from sources none of our tuning, screens or charts had used** (137 scored; see below).
The clips we tune beside give more favourable numbers for most of our presets; they are on a
[development page](docs/results/development-chart.md).

![Bitrate saving vs encode speed on 137 held-out clips, one Jetson Orin Nano core](docs/results/bd-vs-speed-independent_fresh_2026-11-13.svg)

![Bitrate saving vs encode speed on 137 held-out clips, one Ryzen 9 5950X core](docs/results/bd-vs-speed-independent_fresh_2026-11-13-x86.svg)

<sub>**What is measured.** Clips: 138 never-used camera sources (`independent_fresh_2026-11-13`, drawn by a fixed
seeded rule; [list and rules](docs/results/independent-2026-11-13.md#clips)), mostly 540p–720p; 137 scored (one
clip has no reliable curve overlap for libaom rt 8 and is left out for every encoder). Window: 150 frames per clip
from frame 0, one key frame (5 s at 30 fps; see [window length](#window-length)). Rates: CBR at 150–1200 kbps for
720p30, scaled to each clip's pixel rate and capped at half the source's bitrate, plus 40/75 kbps on clips that
need them, 1 s buffer (see [bitrate range](#bitrate-range)). Our encoders at commit 8b492382 (with the rate-control
change of 2026-11-13), measured 2026-11-13; references: x264 0.165, x265 4.2, SVT-AV1 v4.3.0, libaom v3.15.3,
libvpx v1.16.0, openh264 v2.7.0, one thread, low delay, settings in [the method](docs/results/bd-vs-speed.md#method).
Metric: BD-rate on achieved bitrates against x264 `medium --tune zerolatency --bframes 0 --rc-lookahead 0 --threads 1`
(ABR, 1 s VBV); BD-J = PSNR-YUV, VMAF v1 (model v1.0.16) and MS-SSIM in equal thirds, averaged over content
classes and over the full and lower-rate halves of the curves. The bars are the per-clip interquartile range;
95% bootstrap intervals for an earlier build of the same clips are in the
[independent report](docs/results/independent-2026-11-13.md). Scoring: stock libvmaf 3.2.1 on rented CPUs and our
GPU path, which agree at stored precision. Speed: not measured on these clips but on the main chart's three timing
clips (single thread, fixed quantizer near 600 kbps; CBR takes our encoders a median ~2% more CPU, up to 7% for HEVC
Default): Jetson, one Cortex-A78AE core at 1.728 GHz; PC, one Ryzen 9 5950X core on a shared, loaded machine,
normalised by x264 `medium` runs in the same rounds (read PC differences of a few percent as noise). Full pages:
[fresh 138](docs/results/bd-vs-speed-independent_fresh_2026-11-13.md),
[all 209](docs/results/bd-vs-speed-independent_2026-11-13.md).</sub>

| Encoder | BD-J, 137 fresh clips | BD-J, 207 of 209 held-out clips | vs x264 `medium --tune psnr`, fresh | vs x264 `medium --tune psnr`, all | fps, Jetson core | fps, PC core |
|---|---:|---:|---:|---:|---:|---:|
| nanocodec VVC Default | −51.1% | −48.6% | −44.3% | −41.6% | 27.9 | 71.8 |
| nanocodec AV2 Default | −49.2% | −46.5% | −42.3% | −39.4% | 31.5 | 85.5 |
| nanocodec HEVC Default | −47.0% | −43.9% | −39.8% | −36.5% | 51.0 | 121.0 |
| nanocodec VP9 Default | −45.4% | −43.4% | −37.8% | −35.8% | 34.4 | 94.2 |
| nanocodec AV1 Default | −45.3% | −43.5% | −37.9% | −35.9% | 41.1 | 102.6 |
| nanocodec AV1 Fast | −40.2% | −37.3% | −32.0% | −28.9% | 65.2 | 165.8 |
| nanocodec HEVC Fast | −40.0% | −37.8% | −31.6% | −29.5% | 78.5 | 175.8 |
| SVT-AV1 v4.3.0 M8 | −42.2% | −36.6% | −34.0% | −27.5% | 14.3 | 45.2 |
| x265 4.2 `medium --tune psnr` | −33.7% | −31.0% | −24.8% | −22.0% | 8.0* | 25.6* |
| x265 4.2 `medium` | −30.3% | −27.8% | −20.5% | −17.9% | 8.0 | 25.6 |
| SVT-AV1 v4.3.0 M12 `--rtc` | −29.1% | −21.9% | −19.1% | −11.3% | 70.9 | 182.1 |
| SVT-AV1 v4.3.0 M13 `--rtc` | −27.0% | −20.1% | −16.8% | −9.2% | 79.2 | 193.1 |
| nanocodec H.264 Default | −28.7% | −25.9% | −19.1% | −16.4% | 74.4 | 175.9 |
| libaom v3.15.3 realtime `--cpu-used=8` | −27.7% | −24.3% | −17.0% | −14.0% | 17.4 | 42.1 |
| nanocodec H.264 Fast | −21.5% | −18.3% | −11.2% | −8.1% | 113.5 | 230.2 |
| nanocodec H.264 Faster | −14.5% | −11.1% | −3.3% | −0.0% | 133.7 | 265.7 |
| x264 0.165 `medium --tune psnr` | −10.3% | −10.1% | 0 | 0 | 26.7* | 67.6* |
| nanocodec H.264 Fastest | −7.4% | −2.7% | +4.5% | +9.1% | 155.4 | 313.7 |
| libvpx v1.16.0 VP9 realtime `--cpu-used=9` | −2.1% | −0.1% | +11.7% | +13.1% | 47.8 | 113.0 |
| x264 0.165 `veryfast` | −0.1% | +3.9% | +13.5% | +18.0% | 85.4 | 206.4 |
| x264 0.165 `medium` (the anchor) | 0 | 0 | +13.4% | +13.0% | 26.7 | 67.6 |
| openh264 v2.7.0 | +14.2% | +13.9% | +30.9% | +29.4% | 111.5 | 260.3 |
| x264 0.165 `ultrafast` | +33.1% | +39.4% | +49.4% | +56.5% | 190.5 | 547.1 |

<sub>Negative: fewer bits than x264 `medium` (default tuning) for the same quality. "Fresh": 137 of the 138
never-used clips. "All": 207 of 209 clips (the fresh set plus 71 clips of an earlier held-out check, which our
rotating screens have drawn on since 2026-11-11). * the default-tuned run's fps (the metric-tuned run is not yet
timed). These pages use the chart tool's method (one clip set shared by every encoder, clips without an overlap
left out); the [independent report](docs/results/independent-2026-11-13.md), on an earlier build, uses a
per-encoder class-balanced estimate with bootstrap intervals; the two differ by up to about 1.3 points.</sub>

**How to read this.** x264's and x265's defaults are tuned for how the picture looks (psychovisual
optimisations), which PSNR, VMAF and MS-SSIM cannot judge, and no subjective test was run; our presets are tuned
on exactly this metric mix. So every number is also given against x264 `medium --tune psnr`, its documented
metric-oriented tuning, which needs 10.3% fewer bits than the default-tuned anchor on these clips. The charts
also plot each reference's metric-oriented tuning where it is distinguishably better for that encoder (lightly
filled, joined to its default point by a dashed line). Where we are behind:

- **Against x264 `medium --tune psnr`**, our H.264 Faster saves 3.3% on the fresh clips and nothing on all 209
  (−0.0%), and H.264 Fastest needs more bits (+4.5% fresh, +9.1% all).
- **Speed per quality:** SVT-AV1 M12 and M13 `--rtc`, x264 `veryfast` / `ultrafast` and openh264 encode faster
  than our AV1, HEVC and VVC Defaults, and x264 `ultrafast` is faster than any preset of ours; our VVC Default
  runs below 30 fps at 720p on the Jetson core.
- **On the fresh clips SVT-AV1 M8** (14 fps on the Jetson) needs fewer bits than our AV1 Fast and HEVC Fast (on
  all 209 it does not).

Per-metric numbers, per-clip spread and known weaknesses: **[docs/results/](docs/results/README.md)**.

#### Window length

The headline numbers come from 150-frame windows with one key frame (5 s at 30 fps), and our presets were tuned
on such windows. A 5 s window favours an encoder that invests in its first key frame, as ours does. The same
commit (8b492382) was therefore also measured over 600 frames (20 s) with a key frame every 10 s, on the 71 fresh
held-out sources long enough for it (crowd 19, pan 17, static 16, body-cam 13, low light 6), beside the same
sources at 150 frames. BD-J vs x264 `medium`, 95% bootstrap intervals:

| Encoder (71 fresh sources) | BD-J, 150 frames, one key | BD-J, 600 frames, key every 10 s | vs x264 `--tune psnr`, 150 → 600 | achieved ÷ target at 600 (median) |
|---|---:|---:|---:|---:|
| nanocodec VVC Default | −52.8 [−55.2, −50.0] | −45.3 [−47.8, −42.7] | −45.3 → −36.6 | 0.988 |
| nanocodec AV2 Default | −51.1 [−53.8, −48.1] | −43.7 [−46.3, −41.0] | −43.5 → −34.8 | 0.989 |
| nanocodec HEVC Default | −48.3 [−50.7, −45.6] | −40.9 [−43.3, −38.5] | −40.4 → −31.7 | 0.988 |
| nanocodec AV1 Default | −47.1 [−49.3, −44.6] | −40.8 [−43.1, −38.4] | −39.1 → −31.6 | 0.989 |
| nanocodec VP9 Default | −47.4 [−49.9, −44.5] | −40.1 [−42.5, −37.6] | −39.3 → −30.6 | 0.989 |
| nanocodec H.264 Default | −29.8 [−32.7, −26.8] | −19.8 [−22.0, −17.7] | −19.6 → −8.2 | 0.988 |
| SVT-AV1 M8 | −43.8 [−45.9, −41.5] | −41.1 [−43.2, −39.0] | −34.7 → −31.4 | 1.004 |
| SVT-AV1 M12 | −31.6 [−34.8, −27.9] | −27.1 [−30.2, −23.9] | −21.0 → −15.6 | 1.004 |
| x265 medium `--tune psnr` | −34.1 [−36.5, −31.4] | −36.3 [−38.7, −33.7] | −24.4 → −26.7 | 0.986 |
| x265 medium | −30.4 [−32.6, −28.1] | −32.2 [−34.5, −29.6] | −19.7 → −21.3 | 0.989 |
| libaom rt 8 | −29.4 [−31.3, −27.5] | −21.2 [−23.3, −18.8] | −18.1 → −9.2 | 1.009 |
| libvpx VP9 rt 9 | −4.6 [−6.6, −2.4] | +2.4 [−0.0, +5.0] | +10.1 → +17.6 | 1.005 |
| x264 medium `--tune psnr` | −10.9 [−12.3, −9.6] | −11.4 [−12.8, −9.9] | | 0.963 |
| x264 medium | 0 | 0 | | 0.962 |

What it shows:

- **Over 20 s our savings against x264 `medium` are 6–10 points smaller** than over 5 s (AV1 −47.1% → −40.8%,
  H.264 −29.8% → −19.8%); against x264 `medium --tune psnr` our Defaults are −31% to −37% at 600 frames, H.264
  −8.2%.
- **The references move less or gain:** SVT-AV1 M8 2.7 points smaller, SVT-AV1 M12 4.5, while x265 `medium`
  gains 1.8 points.
- **At 600 frames SVT-AV1 M8 (−41.1%) is level with our AV1, HEVC and VP9 Defaults (−40.1% to −40.9%),** at a
  much lower speed: 14.3 fps on the Jetson core and 45.2 on the PC core, against 34.4–51.0 and 94–121 for those
  presets of ours (main chart timing).
- **Our rate is on target over 20 s** (median 0.988–0.989 of target); x264 `medium` lands at 0.962.
- **Buffer:** 74–94 of 252 points per Default of ours still peak above a strict 1 s buffer at 600 frames (at key
  frames, which our controller does not cap), against 10 for x264 `medium`, 18 for x265 `medium` and 20–27 for
  SVT-AV1 M12 / M8.

Limits: 71 sources from 5 classes; the key period differs between the columns (one key at 150 frames, one every
10 s at 600); the 600-frame low rungs ran on 9 clips only. Source and CSVs:
[independent report, update section](docs/results/independent-2026-11-13.md#update-2026-11-13-later-the-new-rate-control-and-a-20-s-window).

#### Bitrate range

The headline numbers cover 150–1200 kbps at 720p30 (scaled to each clip's pixel rate, capped at half the source's
bitrate, plus 40/75 kbps on four clips): the low rates we target. Our savings are largest there. On the 10–11
held-out clips with room for 2,400 and 4,000 kbps, BD-J over the upper range (600 kbps to 4 Mbps at 720p30) was
about half the chart-rate figure (independent audit, 30-clip subset): AV1 −24.8% vs −47.2%, HEVC −26.9% vs
−48.1%, VVC −30.4% vs −53.4%, H.264 −9.3% vs −29.0%; the references' gaps shrink too (x265 `medium` −22.6% vs
−30.8%, SVT-AV1 M12 −4.2% vs −28.2%). With 10–11 clips these upper-range figures are indicative only.

### Spatial scalability (SVC)

One AV1 or VP9 stream carrying a 360p30 base layer and the 720p30 picture (L2T1 / L2T3), so a receiver or relay
can drop to the base without a second encode: `--preset av1-svc` / `vp9-svc` and `--preset av1-svc-fast` /
`vp9-svc-fast`. AV1 also codes three layers, 180p30 + 360p30 + 720p30 (L3T1 / L3T3): `--preset av1-svc3` /
`av1-svc3-fast`. Rate is the **total** stream rate, quality the 720p output (BD-J vs single-layer x264 `medium`,
the scale above), on 8 of the 11 development clips (the [development chart](docs/results/development-chart.md)'s); speed is one core, timed in pairs with x264 `medium`.

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

AV1 also codes the WebRTC K-SVC modes (`--scalability L2T1_KEY` … `L3T3_KEY`): the upper layers predict from the
layer below only in key units, so between keys a relay forwards each receiver one layer. Each view scored at the
rate its receiver downloads, on the same 8 clips (build 95728f25, every anchor from that build, 2026-11-14):
L3T1_KEY's 720p, 360p and 180p views are +4.3%, −1.3% and +1.6% against AV1 Default coded alone at that size
(full SVC L3T1: +35.7%, +21.4%, +2.1%), its total −9.8% against three-way simulcast, at 1.53× the encode
instructions of single-layer AV1 Default (full SVC L3T1: 1.87×). L3T3_KEY and the method:
[svc-chart.md](docs/results/svc-chart.md#k-svc-key-modes-av1-l3t1_key-and-l3t3_key).
VP9 codes the same three-layer and K-SVC modes (`--preset vp9-svc3` / `vp9-svc3-fast`): VP9 L3T1_KEY's views
are +3.5%, +1.2% and +4.8% against VP9 Default alone (+5.9% for 180p on all 11 clips, above the +5% target), its
total −11.8% against three-way simulcast, at 1.64× the encode instructions of single-layer VP9 Default
([svc-chart.md](docs/results/svc-chart.md#vp9-three-layers-l3t1-l3t3-and-k-svc-l3t1_key-l3t3_key)).

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
11 development clips (150 frames, one key frame; held-out figures for the same pairs are on the [fresh-clip page](docs/results/bd-vs-speed-independent_fresh_2026-11-13.md#selected-pairs)) and one Jetson core (fps), against the reference encoders at the
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
The chart streams themselves are scored on the encoder's reconstruction; an independent check re-encoded 144
chart points (every codec's Default, 11 clips) and decoded them with both standard decoders and ours: all were
bit-exact with the reconstruction. On the 209 held-out clips every point of ours was scored from a standard
decoder's output, which equalled the reconstruction frame by frame.

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
