# Launch notes (draft, not posted)

Drafts for the owner to edit and post, or not. Nothing here has been posted anywhere. Numbers must be re-checked
against `docs/results/` on the day; if the independent re-run's held-out table has landed, use its figures.

## Show HN

**Title:** Show HN: nanocodec – a low-delay encoder for six video formats, written by AI agents, in Rust

**First comment:**

> This is an experiment: one shared encoder engine (motion search, mode decision, rate control) driving H.264,
> HEVC, AV1, VP9, AV2 and VVC backends, for low-delay live camera streaming at low bitrate on one CPU core, plus
> decoders for our own streams that run in the browser as WebAssembly. There's a demo that encodes your webcam
> with two formats side by side, locally: https://cribbinsb.github.io/nanocodec/
>
> First, credit where it's due: very little here is new. The coding tools are the standards', and most of the
> encoder design was learned from, and in places ported from, SVT-AV1 (whose real-time pipeline is our model),
> libaom, dav1d, libvpx, openh264, kvazaar, VVenC/VVdeC, AVM and the JM/HM/VTM reference software. What the
> project adds is the combination: one low-delay engine across six formats, a browser decode path, and a written
> account of building it.
>
> The code was written by AI coding agents (mostly Claude, some Codex) directed by one person. It has not had
> line-by-line human review; what checks it is the compiler (no `unsafe` outside SIMD kernels and the C ABI),
> SIMD-vs-scalar tests, and conformance checks against two independent decoders per format.
>
> It is a narrow subset: 8-bit 4:2:0, forward prediction only, no lookahead, one thread, CBR. The comparisons in
> the README are against other encoders' real-time settings on one thread (x264 zerolatency without B-frames,
> SVT-AV1 `--rtc`, libvpx/libaom realtime); those are mature, general-purpose encoders that do better in other
> modes. At low rates (150–1200 kbps at 720p; savings are about half at higher rates), on 137 held-out clips
> from sources we had never used (5-second windows, one key frame), our AV1 Default needs about 45% fewer bits
> than x264 `medium` at those settings, and about 38% fewer than x264 `medium --tune psnr`; our fastest H.264
> preset needs more bits than the latter. On the 11 clips we tune beside the AV1 figure is about 51%, and over
> 20-second windows (5 clips, an earlier build) about 4 points less. It is slower than SVT-AV1's fastest real-time presets; our decoders are slower than dav1d and vpxdec.
>
> The FAQ answers the questions I expected (benchmarks, rate control, overfitting, licences, what's missing):
> https://github.com/cribbinsb/nanocodec/blob/main/docs/FAQ.md. We also had an agent red-team our own claims
> before posting; its notes are in docs/review/. Corrections welcome.

## video-dev Slack / AV1 community / Doom9 post

> I've been running an experiment in building a low-delay, single-thread encoder for live camera footage
> across H.264, HEVC, AV1, VP9, AV2 and VVC on one shared engine, written by AI agents under my direction:
> https://github.com/cribbinsb/nanocodec (browser demo: https://cribbinsb.github.io/nanocodec/).
>
> It leans heavily on SVT-AV1's real-time design and on libaom, dav1d, libvpx, openh264, kvazaar, VVenC/VVdeC and
> AVM, all credited in THIRD_PARTY.md; thank you to everyone who built those.
>
> I'd value expert eyes on the method more than anything. Comparisons are BD-rate (PSNR-YUV / VMAF v1 / MS-SSIM,
> each also reported alone) at 150–1200 kbps 720p CBR with a 1 s buffer, 150-frame clips, against x264
> zerolatency/no-B, SVT-AV1 `--rtc`, libvpx/libaom realtime, all one thread. Known caveats are listed in the FAQ:
> our key frame may borrow up to ~1.4 buffers, we don't code exact-repeat frames, timing is at fixed QP, our
> streams are scored on the encoder's reconstruction (conformance-checked separately). If you think a reference
> is configured unfairly for low-delay one-thread use, please tell me the settings and I'll re-run it.

(For r/rust: lead with the structure: one engine crate behind a `CodecBackend` trait, `unsafe` confined to one
SIMD crate with scalar references and bit-exactness tests, WASM SIMD128 builds; and with the AI-authorship
experiment.)

## The three questions most likely to come first

1. **"Isn't x264 medium with zerolatency and no B-frames a weak anchor, and SVT-AV1 at one thread hobbled?"**
   It's the configuration our use case requires (live, low delay, one core), and we ran every encoder that
   way, including ours. Those encoders are better in their usual configurations; we don't claim otherwise. If
   you know better low-delay one-thread settings for any of them, we'll re-run with them and publish the result.

2. **"Did you just overfit to your test clips?"** The presets were found by automated search, and the chart's
   clips were used to rank changes, so the headline chart is on 137 clips from sources we had never used (plus
   a 209-clip set beside it). Our savings against x264 are 2.6–5.4 points smaller there for the newer codecs (AV1
   −45.3% vs −50.7%), the references' shrink more, and our H.264 does better. The tuning-adjacent chart is on a
   separate development page.

3. **"Who reviewed AI-written decoder code that parses untrusted input?"** Nobody line by line. The decoders are
   safe Rust outside the SIMD kernels, but they can still panic on malformed input, fuzzing so far has been
   short (about 30 min per decoder, VVC not yet), and we say in the FAQ not to expose them to untrusted input in
   production.
