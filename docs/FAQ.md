# Questions a sceptic would ask

Short, plain answers. Where something is not done or not known, we say so. Numbers link to the result pages that
hold them; if a number here and a results page disagree, the results page is right and this page is stale.

## AI authorship

**Who wrote this?** AI coding agents, mostly Anthropic's Claude, with some work by OpenAI's Codex, under the
direction of one person, who set the goals, priorities and rules and made the judgement calls. Commits carry
`Co-Authored-By` trailers naming the model. The commit author name is the project's.

**Has a human reviewed the code?** Not line by line. The safeguards are mechanical:

- the Rust compiler and `#![forbid(unsafe_code)]` in every crate except `nc-dsp` (pixel kernels) and `nc-capi`
  (the C ABI); every `unsafe` block there carries a `// SAFETY:` comment, and every SIMD kernel has a scalar
  reference and a bit-exactness test (`nc-checkasm`);
- `cargo clippy -D warnings`, `cargo fmt`, unit and round-trip tests;
- conformance checks against independent decoders (below) before numbers are published;
- an adversarial review of the public claims ([review notes](review/adversarial-review-2026-11-13.md)).

None of this proves the code is correct or well designed; it catches crashes, mismatches with the standards'
decoders and regressions in what is measured.

**Is any of it copied from GPL code?** We have found none. x264, x265, FFmpeg and libde265 are run only as
external programs. A spot check of distinctive x264/FFmpeg functions, tables and names (cost tables, AQ, psy-rd,
SATD, rate control) found only credited ideas reimplemented differently (for example our AQ uses log-variance, not
x264's energy formula). Code learned from permissively licensed projects (SVT-AV1, libaom, dav1d, libvpx,
openh264, kvazaar, VVenC/VVdeC, AVM, libyuv, Thor, libvmaf, JM/HM/VTM) is credited module by module in
[THIRD_PARTY.md](../THIRD_PARTY.md). If you find something we missed, please open an issue.

**Are the spec citations real?** The code carries over 2,000 `§` references to the standards. Two samples (30 and
then 33, across H.264, HEVC, AV1, VP9 and VVC), checked against the published texts we could read (the AV1 spec,
ITU-T H.264 2003–2007 editions, H.265 2013 edition and 2023 contents, H.266 2023, the VP9 v0.6 draft), found
most correct and a few pointing at a neighbouring clause, usually from a different edition's numbering (for
example HEVC planar/DC prediction, the H.264 CABAC encoder initialisation, VVC transform-type derivation); those
and the same errors elsewhere were corrected (26 files). AV2 citations (draft specification) and HEVC equation
numbers were not checked. Treat a citation as a pointer, not a proof.

**Were the numbers made up by a model?** Every number on the results pages is written by the evaluation harness
(`eval/nceval`) from measured runs, with the builds and command lines on the page. The harness itself is also
AI-written; see "What do the benchmarks show" below for what it does and does not check.

## What the benchmarks show, and what they do not

**What is compared?** Each encoder in its own real-time, low-delay rate control at the same bitrate targets on one
CPU thread: 720p, 150 frames per clip with one key frame, 150–1200 kbps (plus 40 and 75 kbps on four clips, for
every encoder), 1 s buffer. Reference encoders and versions: x264 0.165, x265 4.2, SVT-AV1 v4.3.0, libaom v3.15.3,
libvpx v1.16.0, openh264 v2.7.0, NVENC (RTX 5090). Command lines: [bd-vs-speed.md](results/bd-vs-speed.md#method)
and [tools/anchors/README.md](../tools/anchors/README.md).

**Is that fair to them?** It is one configuration: the one our use case needs (no B-frames, no lookahead, one
thread, CBR with a 1 s buffer). Those projects are general-purpose and are usually run with more threads,
lookahead or B-frames, where they will do better. x264's and x265's defaults are tuned for how the picture looks
(psychovisual optimisations), which PSNR, VMAF and MS-SSIM cannot judge, and we ran no subjective test; our
presets were tuned on exactly the PSNR / VMAF / MS-SSIM mix we report. So the chart also shows x264 and x265 at
the metric-oriented tuning their own documentation provides (`--tune psnr` or `--tune ssim`, whichever scores
better for each), beside their default-tuned points. The AV1 and VP9 references (SVT-AV1 `--rtc`, libaom,
libvpx realtime) already run their PSNR tunes, and openh264 and NVENC run with adaptive quantisation off. That matters: on the chart's clips x264 `medium --tune psnr` (otherwise the same settings)
needs 9.8% fewer bits than the default-tuned anchor on BD-J, and against it our savings are 7–12 points smaller:
AV1 Default −43.6% instead of −50.9%, HEVC −41.7%, VVC −47.1%, H.264 Default −13.3% instead of −23.0%, H.264 Fast
−5.8% (interval spans zero), and our H.264 Faster (+2.5%) and Fastest (+10.7%) save no bits at all
([table](../README.md#quality-and-speed)). We would welcome better settings for any reference within the same
constraints.

**Why a composite metric ("BD-J")?** We tune on it, so we report it, but every table also gives the plain
BD-PSNR-YUV and BD-VMAF numbers ([full table](results/bd-vs-speed.md), [per metric](results/README.md#per-metric)).
BD-J is the mean of the BD-rates on PSNR-YUV (Y:U:V weighted 6:1:1), VMAF and MS-SSIM, averaged over content classes
and over the full and lower-rate halves of each curve (`eval/nceval/jreal.py`). VMAF is Netflix's model
`vmaf_v1.0.16` (the "v1" model of libvmaf 3), computed by libvmaf 3.2.1 or by VMAFx, a GPU implementation with
our patches. Does the GPU fork change the scores? An independent cross-check of 200 held-out points (stock libvmaf 3.2.1 on CPU with `vmaf_v1.0.16_3d0h.json`, plus our MS-SSIM, against the GPU path on the same streams) found PSNR-YUV within 7e-15 dB, VMAF identical, MS-SSIM within 3.9e-6 and identical packet sizes. One caveat: the GPU scorer was not run-to-run reproducible on 8 of 144 chart points (sequence VMAF moved by up to 0.037, single frames by up to 5.5, on bit-identical pictures), so scores are reproducible to about 0.04 VMAF at sequence level; this is a known, open scorer issue. MS-SSIM is scored on every frame
by our reimplementation of libvmaf's float MS-SSIM (agreeing to about 1e-6).

**Are your streams actually decoded, or do you score your own reconstruction?** Our points are scored on our
encoder's reconstruction; the references' on their standard-decoded output. That is valid only if the
reconstruction equals what a standard decoder outputs. We check that before a build's numbers are published, on
short streams of every shipped preset of every codec, decoded by two standard decoders per codec and by ours,
compared frame by frame. An independent check also re-encoded 144 chart points (every codec's Default, all 11
clips) and passed them through both standard decoders and ours: all bit-exact with the scored reconstruction. On
the 209 held-out clips every point of ours was scored from a standard decoder's output, checked equal to the
reconstruction frame by frame.

**Only 150 frames with one key frame?** Yes (5 s at 30 fps), and our presets were tuned on such windows. It
favours us: in 5 s our first key frame takes 8–10% of the bits and our streams end 5–6% over target (the key
frame's loan is not yet repaid). On 5 clips (not independent: 3 used for tuning, 2 from the chart set; too few
for intervals) over 600 frames with key frames at 0 and 300, our savings against x264 `medium` were smaller for
all three codecs tested: AV1 −54.7% → −50.6%, HEVC −52.2% → −46.7%, H.264 −24.5% → −15.9%; against x264 `medium
--tune psnr` over 600 frames, AV1 −42.0%, HEVC −37.6%, H.264 −4.1%. A longer-window chart is planned; until then
read the 150-frame numbers as favourable to us by several points.

**Why these bitrates?** Our use case is low-bitrate streaming. Each clip's ladder is scaled to its pixel rate and
capped at half its source's bitrate, because many sources are already compressed and coding above that measures
the source's artefacts. Our savings are largest at these low rates: on the 10–11 held-out clips with room for
2,400 and 4,000 kbps (720p30), BD-J over 600 kbps–4 Mbps was about half the chart-rate figure (AV1 −24.8% vs
−47.2%, H.264 −9.3% vs −29.0%), and the references' gaps shrank too. Read every headline number as "at
150–1200 kbps, 720p30".

**Which clips?** The chart uses 11 natural camera clips (bodycam, crowd, low light, pan, static), our VAL set. They
were also used to confirm which candidate changes were kept, so they are not independent of the tuning. Many
sources are not redistributable, so the clips themselves cannot be published; the clip lists are in
`eval/clipset/`.

**Did you overfit?** Our presets were found by automated search on clips that include the chart's, so an
independent re-run measured every encoder (same builds and settings, 150 frames, one key frame) on 138 fresh clips
from sources no clip set of ours had ever used, plus the 71 clips of an earlier held-out check. BD-J vs x264
`medium` with 95% intervals: our AV1 Default −45.6% [−47.8, −43.5] on the fresh clips (−50.9% on the chart's),
HEVC −46.9% (−49.3%), VVC −51.1% (−54.0%), H.264 −28.5% (−23.0%); the references moved more (SVT-AV1 M12
`--rtc` −29.7% vs −34.9%, x265 `medium` −30.6% vs −40.7%). So the size of a saving depends on the clips, our
position among the encoders tested held, and our H.264 did better than on the chart's clips. Against a PSNR-tuned
x264 `medium` on all 209 clips: AV1 −37.5%, H.264 −15.6%, H.264 Faster +0.7% (not ahead), H.264 Fastest +10.7%
(behind). The full table is in the [README](../README.md#quality-and-speed).

**Speed: how measured?** Encode CPU time per frame, one thread, at a fixed quantizer that lands each encoder near
600 kbps on three clips; not timed in CBR mode. In CBR our encoders take a median about 2% more CPU (0–7% by
preset, three clips at 600 kbps, x86); the references' CBR cost was not measured. Jetson Orin Nano: one
Cortex-A78AE core, clock locked. PC: one Ryzen 9 5950X core on a shared, loaded machine, each run normalised by
x264 `medium` runs in the same round; read PC differences of a few percent as noise.

## Rate control

**What does "the CBR target is a ceiling" mean?** Our controller may spend less than the target when more bits
would not buy visible quality (a static scene, say); it should not spend more. BD-rate uses the bitrate each
encoder actually achieved, so spending less is not an advantage in the numbers: it moves the point left on its
own curve.

**Do you overshoot?** Over short windows, yes; over the long run, no. Since 2026-11-13 (62788659, 71cad753) our controller works like this: from about 5 s into a stream, the stream
so far is at or under the target on average and stays there (it aims 1% under); the first key frame may borrow
bits (it is not capped at one buffer: up to about 1.4 buffers) and repays them within 5 s; any other spend above
the target line is repaid too. Measured limits on long single-key clips: single seconds holding a key frame, or a
refinement burst on a static scene, can reach about 2× the target; 5–10 s windows on static scenes +8% to +20%;
windows of 30 s at most about 1% over, and whole streams at or under the target. Motion content stays within a few percent
over any 5 s window. Strict buffer settings (`rate.recode_overflow=1 rate.key_cap_buffer=1`) keep every frame,
keys included, inside a one-buffer receiver, at a quality cost on static scenes. The references run
with a strict 1 s VBV.

An independent audit of the chart's points (42 per encoder, 150 frames), **at the builds the charts were measured
on, before that change** (the key loan was then repaid over 10 s and the long-run rate was not held under the
target), found:

- our streams end a median 3–5% above target (about on target once the unrepaid loan is excluded); x264 and x265
  end 3–10% below target, SVT-AV1 on target;
- in a strict 1 s leaky bucket, 5–12 of 42 points per preset of ours peak above the buffer, at most 1.40 buffers
  (the first key frame); x264 `medium` 2 points (at most 2.11), SVT-AV1 1–2 (at most 2.35), x265 none; libaom,
  libvpx and openh264 6–17 points, at most 9.6, 13.6 and 22.5 buffers;
- with strict settings (`rate.recode_overflow=1`, `rate.recode_overflow_inter=1`, `rate.key_cap_buffer=1`,
  `rate.key_repay_seconds=1`) no point of ours exceeds the bucket, and our Defaults need 4.7–6.9% more bits at
  equal quality: BD-J against x264 moves about 2 points (AV1 −50.9% → −49.0%, H.264 −23.0% → −20.7%).

Every encoder's actual bytes, the loan included, are what the BD axis uses. Our looser "rate valid" flag in the
tables (bucket peak at most 1.5× the buffer; average at most 10% plus the buffer's share of the window over target)
is applied to every encoder alike.

**Do you skip frames?** Our encoder does not code an input frame that would not change the picture (exact or
near-exact repeats, typical of recompressed or frame-doubled sources). Such "held" frames cost no bits and no
encode time and are scored as the repeated picture, which is what a viewer sees. The references code every frame
(cheaply, as skip frames). On the chart's points no frame was held (independent audit), so holds do not affect
the published numbers. No encoder in the charts drops frames.

## Patents and licences

**Can I use this commercially?** The code is BSD 3-Clause, a copyright licence only. It gives you no patent
licence: H.264, HEVC and VVC are covered by patent pools and other holders, and AV1, VP9 and AV2, though designed
to be royalty-free, also have third-party patent claims. See [LEGAL.md](LEGAL.md). It is experimental and not
production-ready.

**What about the demo clips?** They keep their own licences ([clip attribution](../web/clips/ATTRIBUTION.md));
some are non-commercial (Netflix CC BY-NC-ND, MOTChallenge CC BY-NC-SA). The Netflix excerpts are format
conversions only and are not re-encoded into comparison streams. Ask and we will remove any clip.

**The demo's pre-encoded streams were made with x264 and x265, which are GPL.** Running a GPL program does not put
its output under the GPL; we distribute only the output streams, not the programs.

## What is not done

- **Single-threaded** encode and decode; no slices or tiles beyond one per frame.
- **8-bit 4:2:0 only.** No 10-bit, no 4:4:4, no HDR.
- **Subset encoders:** forward prediction only (no B-frames), no lookahead, CBR and fixed-QP only, one tile/slice;
  several tools of the newer standards are unused (for example compound prediction and loop restoration in
  AV1/AV2).
- **Decoders:** H.264 (Constrained High), HEVC (Main), AV1 (Main, 8-bit 4:2:0) and VP9 (profile 0) pass the
  conformance streams we test in those profiles; VVC passes the 9 official 8-bit streams tested so far and is not
  a full Main-profile decoder; the AV2 decoder decodes our own streams only. Details in the
  [README](../README.md#decoders).
- **Speed gaps:** our AV1 and VP9 decoders are slower than dav1d and vpxdec single-threaded; SVT-AV1 M12/M13 and
  x264's fast presets encode faster than our Default presets; VVC Default runs below 30 fps at 720p on one Jetson
  core.
- **Known quality weak points:** fine pristine texture (water, foliage), where x264 and x265 need fewer bits;
  H.264 flicker; VP9 trails our HEVC; SVC costs 6–14 points of BD-J against single-layer streams; AV2 and VVC are
  young (rate control and quality work ongoing). See [known weaknesses](results/README.md#known-weaknesses).

## Security and fuzzing

The decoders are written in safe Rust (no `unsafe` outside the SIMD kernels and the C ABI), so memory-safety bugs
are unlikely outside those kernels, but a malformed stream can still make a decoder panic (abort) or misbehave.
Each decoder has a `cargo fuzz` target (`fuzz/`) that drives the browser, C API and command-line decode paths,
seeded from our streams and the official conformance vectors. On 2026-11-13 every decoder, VVC included, had 0.8 to
2.6 worker-hours of fuzzing (about 725,000 executions in all). Fuzzing and a review of the size limits found and
fixed four issues: in HEVC a slice naming an out-of-range parameter set crashed the decoder; in AV1 invalid film
grain points crashed it, frames whose size is not a multiple of 8 overflowed in deblocking, and the picture size was
not capped at the level 6.3 limit. Nothing was found in H.264, VP9, VVC or AV2. That is a bounded search, not a
proof: see [results/fuzzing.md](results/fuzzing.md) for the numbers and what was not tested. No corpus is published.
Do not expose these decoders to untrusted input in production. The C ABI catches panics at the boundary. The browser
demo decodes only streams it encoded itself and the project's hosted streams.

## Why build this at all?

To see how far one shared, low-delay engine can go across six formats on a small CPU, and to have our own
browser decoders for our own streams. The coding tools are the standards'; the ideas are largely those of the
projects we credit. What is new here is the combination, the shared engine, and an account of building it with AI
agents ([RETROSPECTIVE.md](RETROSPECTIVE.md)).
