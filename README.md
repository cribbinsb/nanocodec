# nanocodec

[![crates.io](https://img.shields.io/crates/v/nanocodec.svg)](https://crates.io/crates/nanocodec) [![docs.rs](https://img.shields.io/docsrs/nanocodec)](https://docs.rs/nanocodec) [![License: BSD-3-Clause](https://img.shields.io/badge/license-BSD--3--Clause-blue.svg)](LICENSE)

**Six video formats, one encoder engine, written in Rust by AI agents.**

nanocodec encodes and decodes **H.264, HEVC, VVC, AV1, VP9 and AV2** for live, low-delay camera video:
cameras, drones and robots streaming at low bitrate to the cloud or straight into a browser.

**[Try it in your browser](https://cribbinsb.github.io/nanocodec/)** · **[Interactive results](https://cribbinsb.github.io/nanocodec/results/)** ·
[Docs](https://cribbinsb.github.io/nanocodec/docs/) · [FAQ](docs/FAQ.md)

![Bitrate vs encode speed on the held-out clips (audited once, never tuned on), Arm (Graviton 4)](docs/images/bd-vs-speed-bp4.svg)

*Lower = fewer bits for the same picture quality; right = faster. Blue is nanocodec, orange the established
encoders, all in the same low-delay, single-thread live setting, timed on Arm (AWS Graviton 4); x86 is on the
results pages ([how we measured](https://cribbinsb.github.io/nanocodec/results/method.html); the procedure's
[known limitations](https://cribbinsb.github.io/nanocodec/results/#limitations) are listed beside the results). The held-out clips
are "audited once, never tuned on"; rate compliance and each family's own-clip figure are on the results pages.
Ours is development build 23c962d8b, measured before 0.7.0's last encoder changes.*

## Why it's interesting

- **About a third fewer bits than x264.** At their `medium` presets our VVC, AV2, AV1 and HEVC encoders need
  30–36 % fewer bits than x264 `medium` (tuned for PSNR) for the same quality on our held-out camera clips
  (audited once, never tuned on), at a similar or higher
  speed. ([All results](https://cribbinsb.github.io/nanocodec/results/))
- **One engine, six formats.** Rate control, motion search and mode decisions are shared; each format adds
  only its own syntax. An improvement lands once and every format gets it.
- **Plays in the browser.** All six decoders compile to WebAssembly (SIMD128: Chrome/Edge 91+, Firefox 89+,
  Safari 16.4+; tested in Chromium only), so a page can play HEVC, VVC or AV2 without the browser's own codecs.
- **Standard streams, checked.** After every landing, short test streams of every shipped preset of every
  format (layers, holds, intra refresh, starved keys and the main coding tools) must decode bit-exactly in
  two independent decoders per format and in ours. That checks the presets' coding paths, not every stream
  an encoder can make; the benchmark itself does not cross-decode. Our decoders also decode other encoders'
  streams, but not yet every valid one:
  - HEVC: 147/147 of the official ITU-T/ISO HEVC_v1 suite (Main, Main 10); no known gap in those profiles.
  - VVC: 226/283 of the official H.266.1 suite (all its Main 10 streams but two); reserved NAL types,
    layer ids above 0 and a VPS id above 0 are rejected, not yet ignored.
  - H.264: 110/190 of the official H.264.1 suite (its AVCv1, FRExt and High 10 sets); not yet interlaced
    streams (field pictures, MBAFF) or pictures above level 5.2.
  - AV1 and VP9: 242/242 libaom and 314/314 libvpx test vectors (encoder project vectors, not official
    suites). AV1 not yet: quantizer matrices, loop-filter deltas, segment features other than the
    quantizer, lossless segments in mixed frames, odd frame sizes, switch frames. VP9: all four profiles.
  - AV2 (a draft): 30 streams (12 dav2d vectors, 18 we made with avmenc); no suite exists. Not yet:
    10-bit, monochrome, several tiles, reference scaling.

  Work on the gaps is under way; full list: [Conformance](docs/FAQ.md#conformance).
  Every known gap and its status (in progress, open, or proposed exception): [decoder compliance register](docs/decoder-compliance.md).
- **Scalable video (SVC).** Spatial and temporal layers for AV1 and VP9 in the WebRTC-SVC modes (L1T1–L3T3
  and K-SVC), so one stream can serve viewers at different resolutions and bitrates; every layer is checked in
  the standard decoders.
- **RTP / WebRTC transport.** Packetizers and depacketizers for all five standard formats (`nanocodec-rtp`;
  AV1 with the Dependency Descriptor for SVC forwarding), tested against GStreamer and Pion. Headless Chrome
  decodes our H.264, AV1 (including two-layer SVC) and VP9 packets over a real WebRTC session, bit-exact. VP9
  spatial layers are not carried yet ([details](docs/handoff/rtp.md)).
- **Built for loss.** Temporal layers and key frames on request (all six formats), long-term references with
  loss feedback (H.264, HEVC, VVC, AV1, VP9, AV2) and intra refresh (H.264, HEVC, VVC, AV1). Recovery from a lost
  frame is tested with the standard decoders for H.264, HEVC, AV1 and AV2.
- **Fast on x86, Arm and the web.** Hand-written SIMD for AVX2, AVX-512, NEON and WebAssembly.
- **Built by AI.** About 4,000 commits and half a million lines of Rust written by AI coding agents
  (mostly Anthropic's Claude) in its first fifteen days, directed by one person ([how it's checked](docs/FAQ.md#ai-authorship)).
- **Permissive licence.** BSD 3-Clause; safe Rust except the DSP crate (SIMD kernels, assembly and their dispatch) and the C
  API; every other crate has `#![forbid(unsafe_code)]`.

**Where we're behind:** our H.264 is close to x264 rather than ahead of it, and the fastest settings of
SVT-AV1, libvpx and openh264 run faster than any of our `medium` presets. These are our own measurements, not
an independent benchmark.

## Quick start

From [crates.io](https://crates.io/crates/nanocodec-cli):

```sh
cargo install nanocodec-cli
nanocodec encode --codec av1 --input input.yuv --width 1280 --height 720 \
  --fps 30 --frames 150 --bitrate 600 --buffer-ms 1000 --output video.ivf
nanocodec decode --codec av1 --input video.ivf --output decoded.yuv
```

Or from this repository:

```sh
cargo build --release
target/release/nanocodec encode --codec av1 --input input.yuv --width 1280 --height 720 \
  --fps 30 --frames 150 --bitrate 600 --buffer-ms 1000 --output video.ivf
target/release/nanocodec decode --codec av1 --input video.ivf --output decoded.yuv
```

`--codec h264 | hevc | vvc | av1 | vp9 | av2`. Building for x86_64, Arm and WebAssembly, every option, the C API
and the browser module: the [usage docs](https://cribbinsb.github.io/nanocodec/docs/).

## Use from Rust

`cargo add nanocodec` ([crates.io](https://crates.io/crates/nanocodec), [docs](https://docs.rs/nanocodec)).
Each codec is a cargo feature (`h264`, `hevc`, `vvc`, `av1`, `vp9`, `av2`; all on by default), and the crate
documentation has an encode and decode example. The engine, codecs, C API and browser bindings are also published
separately as `nanocodec-*` crates.

## Presets

Pick a speed with `--preset`: `veryfast`, `faster`, `fast`, `medium` (the default), `slow`, `slower`,
`veryslow`. Faster presets use less CPU for a few more bits; not every codec has every name
(`nanocodec params --codec C` lists them). They are our own ladder per codec, not x264's speeds: the
[results](https://cribbinsb.github.io/nanocodec/results/) show what each one measures. Spatial SVC for AV1 and VP9:
`--scalability L2T1` or `L3T1`.

## Built on others' work

nanocodec learned from, and in places ports, open-source codecs: above all
[SVT-AV1](https://gitlab.com/AOMediaCodec/SVT-AV1), whose real-time pipeline is our main guide, and also
[libaom](https://aomedia.googlesource.com/aom/), [dav1d](https://code.videolan.org/videolan/dav1d),
[libvpx](https://chromium.googlesource.com/webm/libvpx), [openh264](https://github.com/cisco/openh264),
[kvazaar](https://github.com/ultravideo/kvazaar), [VVenC](https://github.com/fraunhoferhhi/vvenc) /
[VVdeC](https://github.com/fraunhoferhhi/vvdec), [AVM](https://gitlab.com/AOMediaCodec/avm),
[libvmaf](https://github.com/Netflix/vmaf) and the JM / HM / VTM reference software, and on the experts who
wrote the standards. Details: [THIRD_PARTY.md](THIRD_PARTY.md).

## Licence and patents

BSD 3-Clause ([LICENSE](LICENSE)), a copyright licence only: **it grants no patent rights, and using nanocodec or
its demo does not license you under any patent on H.264, HEVC, VVC, AV1, VP9 or AV2.** Experimental software,
provided "as is". No GPL or LGPL code is in nanocodec's libraries, binaries or demo. Our
benchmark tools run x264, x265 and FFmpeg as separate programs, or link FFmpeg's libraries to time its decoders
(`tools/anchors/libavcodec_bench.c`, not shipped). Details: [Credits, licences, patents and disclaimers](docs/LEGAL.md). The demo's sample media are not BSD: see [MEDIA-LICENSES.md](MEDIA-LICENSES.md).

Contact: <bernandocribbenza@gmail.com>, or open an issue on GitHub.
