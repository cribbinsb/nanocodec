# Questions a sceptic would ask

Short, plain answers. Where something is not done or not known, we say so. Numbers link to the result pages that
hold them; if a number here and a results page disagree, the results page is right and this page is stale.

## AI authorship

**Who wrote this?** AI coding agents, mostly Anthropic's Claude, with some work by OpenAI's Codex and Google's
Gemini, under the direction of one person, who set the goals, priorities and rules and made the judgement calls.

**How long did it take?** Fifteen days to release 0.5.0. The first commit is dated 6 November 2026 and 0.5.0
20 November 2026; in between are about 4,000 commits and about 510,000 lines of Rust (tests and tables included), plus the
evaluation harness and the web pages. Those counts are from the development history; the public repository
receives it as periodic sync commits. Commits carry
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

**What is compared?** Each encoder in its own low-delay rate control at the same bitrate targets on one CPU
thread, against x264 `medium --tune zerolatency,psnr` as the zero line: 150 frames per clip with one key frame, a
1 s buffer, and a ladder of 150, 300, 600 and 1200 kbps at 720p30, scaled down by pixel rate for smaller clips
(most held-out clips are below 720p), plus 40 and 75 kbps (scaled the same way) for every encoder on any clip
where some curve does not overlap the zero line's. The exact clip counts, ladder and builds of each chart are
stated with it on the [results pages](https://cribbinsb.github.io/nanocodec/results/). Reference encoders and versions: x264 0.165, x265 4.2, SVT-AV1 v4.3.0, libaom v3.15.3,
libvpx v1.16.0, openh264 v2.7.0, NVENC (RTX 5090). Command lines: [bd-vs-speed.md](results/bd-vs-speed.md#method)
and [tools/anchors/README.md](../tools/anchors/README.md).

**Is the comparison fair to them?** It is one configuration: the one our use case needs (no B-frames, no
lookahead, one thread, CBR with a 1 s buffer). Those projects are general-purpose and are usually run with more
threads, lookahead or B-frames, where they will do better. Two settings are known to cost references bits: openh264
runs Constrained Baseline with CAVLC, and with CABAC on it needed about 17 % fewer bits on five 720p development
clips; libvpx and libaom run without cyclic-refresh AQ (`--aq-mode=3`), which moved libvpx by about −7 % and
libaom by −2 to −4 % on average on the same five clips, with mixed signs by clip (one run each,
[audit, finding 3](analysis/results-audit-2026-11-23.md#3-reference-settings)). Ours is a PGO + LTO build; the
references are stock release builds. Within that configuration we made three choices:

- **The zero line is x264 `medium` set up as a careful live sender for these metrics:** `--tune
  zerolatency,psnr` (psychovisual optimisation and adaptive quantisation off), no B-frames, no lookahead, ABR with
  the same 1 s VBV our encoders get. x264's default tuning favours how the picture looks, which PSNR, VMAF and
  MS-SSIM cannot judge (we ran no subjective test), so against the metrics we report it needed 10.3% more bits than
  the PSNR tune on the fresh held-out clips (2026-11-13 builds); it stays in the table as a row. Our presets are tuned on exactly this
  metric mix.
- **Every other reference at its own metric-oriented setting:** x264 and x265 also at `--tune psnr` / `--tune
  ssim` where that is distinguishably better for them; SVT-AV1 `--rtc`, libaom and libvpx realtime already run
  their PSNR tunes; openh264 and NVENC run with adaptive quantisation off (turning it on did not help them).
- **Rate control is part of what is compared.** Every encoder runs its own low-delay CBR, so a margin mixes coding
  efficiency with how much each rate control costs. Measured separately on 2026-11-14 on the development clips
  (preset names of that date) for our AV1 Default: at fixed QP against fixed QP it needs 33.9% fewer bits than PSNR-tuned x264 `medium` (38.3% in CBR
  against CBR); CBR costs our encoders 10.7–21.5% against their own fixed QP, x264 `--tune psnr` 48.6% and
  default-tuned x264 70.5% ([rc-cost-2026-11-14.csv](results/rc-cost-2026-11-14.csv)). Our H.264 Default is level
  with x264 `medium` at fixed QP (+0.4% / +2.6% against `--tune psnr`): its whole CBR margin is rate control, and
  our H.264 Fastest needs about 30% more bits than x264 `medium` at fixed QP.

The current figures for every preset against that anchor, with their clip counts and builds, are in the
README's [headline](../README.md#headline-results) and the [per-codec table](https://cribbinsb.github.io/nanocodec/results/codecs.html).
We would welcome better settings for any reference within the same constraints.

**Why a composite metric ("BD-J")?** We tune on it, so we report it, but every table also gives the plain
BD-PSNR-YUV and BD-VMAF numbers ([full table](results/bd-vs-speed.md), [per metric](results/README.md#per-metric)).
BD-J is the mean of the BD-rates on PSNR-YUV (Y:U:V weighted 6:1:1, each plane's per-frame PSNR capped at 60 dB,
libvmaf's 8-bit cap; 100 dB in charts published before 2026-11-20), VMAF and MS-SSIM, averaged over content classes.
Charts rendered before 2026-11-23 (release 0.6.0's) also average the full overlap of each curve with its
lower-rate half, so the low rates count twice; charts rendered from 16de42521 on use the whole overlap and give
the BD by rate regime (up to 600 kbps, and 600 kbps–4 Mbps, at 720p30 scaled) beside it
(`eval/nceval/chart.py`, `eval/nceval/jreal.py`). VMAF is Netflix's model
`vmaf_v1.0.16` (the "v1" model of libvmaf 3), computed by libvmaf 3.2.1 on the CPU (points scored on rented machines) or by VMAFx, a GPU (CUDA) build of
libvmaf (points scored on our PC), at each clip's own size: most held-out clips are below 720p, so these VMAF values are not
comparable with figures from the 1080p VMAF protocol. Does the GPU fork change the scores? An independent cross-check of 200 held-out points (stock libvmaf 3.2.1 on CPU with `vmaf_v1.0.16_3d0h.json`, plus our MS-SSIM, against the GPU path on the same streams) found PSNR-YUV within 7e-15 dB, VMAF identical, MS-SSIM within 3.9e-6 and identical packet sizes. One caveat: the GPU scorer was not run-to-run reproducible on 8 of 144 chart points (sequence VMAF moved by up to 0.037, single frames by up to 5.5, on bit-identical pictures), so scores are reproducible to about 0.04 VMAF at sequence level; this is a known, open scorer issue. MS-SSIM is scored on every frame
by our reimplementation of libvmaf's float MS-SSIM (agreeing to about 1e-6).

**Are your streams actually decoded, or do you score your own reconstruction?** Both, by where a point is
scored. The references' points are always scored on a standard decoder's output. Ours scored on rented machines
(the held-out and VC-1K charts and, since 2026-11-21, the chart refreshes) are scored on the first standard
decoder's output, checked equal to the encoder's reconstruction frame by frame; ours scored on our PC are scored
on the encoder's reconstruction, with PSNR from the encoder's own squared-error log (on 65 re-encoded points it
matched libvmaf's PSNR within 5e-7 dB per plane per frame;
[metric validation](results/metric-validation-2026-11-23.md)). Scoring the reconstruction is valid only if it
equals what a standard decoder outputs. We check that before a build's numbers are published, on
short streams of every shipped preset of every codec, decoded by two standard decoders per codec and by ours,
compared frame by frame. An independent check also re-encoded 144 chart points (every codec's Default, all 11
clips) and passed them through both standard decoders and ours: all bit-exact with the scored reconstruction. One open caveat: the reference frames of clips stored compressed are decoded by the first FFmpeg on the
machine's path, which is not pinned; FFmpeg 7.1.1 and 8.0.1 decode 3 of 38 sampled VC-1K clips differently, and
the stored chart values match 8.0.1.

**Only 150 frames with one key frame?** Yes (5 s at 30 fps), and our presets were tuned on such windows. It
favours us: in 5 s our first key frame takes 8–10% of the bits and our streams end 5–6% over target (the key
frame's loan is not yet repaid). On 71 fresh held-out sources at commit 8b492382, over 600 frames (20 s) with a
key every 10 s, our savings against x264 `medium` are 6–10 points smaller than over 150 frames (AV1 −47.1% →
−40.8%, HEVC −48.3% → −40.9%, H.264 −29.8% → −19.8%), while the references move less or gain (SVT-AV1 M8 −43.8% →
−41.1%, x265 `medium` −30.4% → −32.2%); SVT-AV1 M8 is then level with our AV1, HEVC and VP9 Defaults, at well
under half their speed. Against x264 `medium --tune psnr` at 600 frames our Defaults are −31% to −37%, H.264
−8.2%. The table is in the [method page](https://cribbinsb.github.io/nanocodec/results/method.html#window).

**Why these bitrates?** Our use case is low-bitrate streaming. Each clip's ladder is scaled to its pixel rate and
capped at half its source's bitrate, because many sources are already compressed and coding above that measures
the source's artefacts. Our savings are largest at these low rates: on the 10–11 held-out clips with room for
2,400 and 4,000 kbps (720p30), BD-J over 600 kbps–4 Mbps was about half the chart-rate figure (AV1 −24.8% vs
−47.2%, H.264 −9.3% vs −29.0%), and the references' gaps shrank too. Read every headline number as "at the
150–1200 kbps ladder at 720p30, scaled down for smaller pictures (down to about 30 kbps, and lower where the 40
and 75 kbps rungs are added)".

**Which clips?** The headline chart uses the 138 fresh held-out clips (a fixed, seeded draw of clips no clip set
of ours had used; most come from the same public datasets as the development clips), with the 209-clip held-out
set (the fresh clips plus 71 clips of an earlier check) beside it; the README states how many of each were
scored. The fresh set has been re-measured at seven or more builds since 2026-11-13, with its numbers seen by
the people and agents choosing what to work on next. The
[development chart](results/development-chart.md) uses the 11 natural camera clips we tune beside (our VAL
set); they were used to confirm which candidate changes were kept, so they are not independent of the tuning, and
they give most of our presets more favourable numbers. Many
sources are not redistributable, so the clips themselves cannot be published; the clip lists are in
`eval/clipset/`.

**Did you overfit?** Our presets were found by automated search on clips that include the development chart's,
so the headline chart is measured on held-out clips (above). At commit 8b492382 (2026-11-13, preset names of that
date; 150 frames, one key frame), BD-J vs the default-tuned x264 `medium`, held-out (137 fresh) vs development
(11 clips): our AV1
Default −45.3% vs −50.7%, HEVC −47.0% vs −49.6%, VVC −51.1% vs −54.4%, H.264 −28.7% vs −24.0%; the references
moved more (SVT-AV1 M12 `--rtc` −29.1% vs −34.9%, x265 `medium` −30.3% vs −40.7%). So the size of a saving depends
on the clips, our position among the encoders tested held, and our H.264 does better on the held-out clips. The
independent report (an earlier build) gives 95% bootstrap intervals, about ±2 points for these figures. The full
table is on the [results pages](https://cribbinsb.github.io/nanocodec/results/).

**Speed: how measured?** In the published 0.6.0 charts: encode CPU time per frame, one thread, at a fixed
quantizer that lands each encoder near 600 kbps, on three 720p development clips (`lowlight_city_night`,
`crowd_mot17_09`, `bodycam_bc1`), 150 frames; not in CBR mode and not on the clips the quality is measured on
(re-timing the CBR encodes on each chart's own clips is in progress). In CBR our encoders took a median about 2%
more CPU (0–7% by preset, the same three clips, x86); the references' CBR cost was not measured. Jetson Orin Nano:
one Cortex-A78AE core, clock locked at 1.728 GHz. x86: one core of a rented AMD EPYC 9R45 (AWS c8a.large); our
times there are scaled per clip to an earlier session's x264 `medium` time, and the references keep their own
earlier sessions' times, so points of different sessions share one axis. The metric-tuned references carry the
time of their default-tuned runs.

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
keys included, inside a one-buffer receiver, at a quality cost on static scenes. x264 and x265 run with a
hard 1 s VBV whose maximum rate is the target; SVT-AV1, libaom and libvpx with their real-time CBR and a 1 s
buffer; openh264 has no buffer setting. Our encoder's default peak rate is twice the target.

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

Every encoder's actual bytes, the loan included, are what the BD axis uses. The "rate-invalid" count in the
tables (bucket peak above 1.5× the buffer, or an average more than 10% plus the buffer's share of the window over
target) is looser than one buffer, and for our points only the outstanding key loan is deducted first, so it
flags few of ours; the share of each encoder's points over one buffer is on the
[buffer page](https://cribbinsb.github.io/nanocodec/results/buffer.html).

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

## Conformance

Our decoders also decode other encoders' streams, natively and in the browser, 8-bit and 10-bit (AV2:
8-bit only), but **none of them decodes every valid stream of its codec's main profiles yet**. Below, each
line names the suite, says whether it is an official ITU-T/ISO conformance suite or an encoder project's
test vectors, gives the count, and lists every valid-stream feature of the main profiles that decoder
still rejects. Streams decoded natively and compared picture by picture with the reference decoder named,
on 2026-11-23 at 98069a8b3 ([record](results/conformance-2026-11-23/summary.json)); the gap lists are
from the [audit of 2026-11-27](analysis/decoder-compliance-gaps-2026-11-27.md).

- **HEVC** (Main, Main Still Picture, Main 10): 147/147 of the official ITU-T/ISO JCT-VC HEVC_v1
  conformance suite (H.265.1; 136 8-bit, 11 10-bit), against HM. No known gap in these profiles. Not
  supported (other profiles): the range extensions (4:0:0, 4:2:2, 4:4:4, 12-bit); the RExt, SCC,
  MV-HEVC, SHVC and 3D-HEVC suites are not run.
- **VVC** (Main 10 and Main 10 Still Picture, 4:2:0 and 4:0:0): 226 of the 283 streams of the official
  ITU-T H.266.1 first-edition conformance suite (9 8-bit, 217 10-bit), against VTM 24.0: all of its Main 10
  streams but HRD_B_2 (VTM itself fails it) and VPS_C_2 (not run). The other 55 are 4:2:2 / 4:4:4,
  multilayer, palette and ACT streams (other profiles, not supported). STILL_B_1 is compared with VTM's
  output, which differs from the MD5 in its zip. **Not yet** (valid Main 10 streams rejected): reserved
  and unspecified NAL unit types and NAL units with `nuh_layer_id` above 0 are an error instead of being
  ignored (§7.4.2.2), and an SPS with `sps_video_parameter_set_id` above 0 is rejected. Also 138 of 138
  short 8-bit streams we coded with VTM and VVenC to cover the tools one by one (our own streams, not a suite).
- **H.264** (Constrained Baseline, Main, High, High 10): 110 of the 190 streams of the official ITU-T
  H.264.1 conformance suite's AVCv1, FRExt and Professional (High 10) sets, against JM. **Not yet**:
  interlaced streams (field pictures and MBAFF; 75 of the 80 failures are such Main/High streams) and
  pictures larger than level 5.2 allows (levels 6 to 6.2, 8K). The other 5 failures use FMO/ASO or SP/SI
  slices (Baseline/Extended only, not supported). The suite's 4:2:2, 4:4:4, MVC and SVC streams are not
  fetched (other profiles, not supported).
- **AV1** (Main, 8- and 10-bit): 242/242 libaom v3.15.3 test vectors (176 8-bit, 66 10-bit), against
  aomdec. These are an encoder project's test vectors, **not a conformance suite**; the AOM Argon AV1
  conformance streams have not been run. **Not yet** (valid Main streams rejected): quantizer matrices,
  block loop-filter deltas, segment features other than the quantizer, lossless segments in a frame that
  is not wholly lossless, odd frame sizes, switch frames. A fix is in progress. High and Professional
  profiles are not supported.
- **VP9** (profiles 0 and 2, 8- to 12-bit): 305/305 libvpx v1.16.0 test vectors (303 8-bit, one 10-bit,
  one 12-bit), against vpxdec. These are an encoder project's test vectors (the de facto suite; VP9 has no
  official one), and only two cover profile 2. **Not yet**: profiles 1 and 3 (4:2:2, 4:4:0, 4:4:4); their
  vectors are not run.
- **AV2** (a draft standard, AVM v1.0.0): 30 of 30 foreign streams (12 dav2d test vectors, 18 we made with
  avmenc v1.0.0), all 8-bit, single tile, 640x360 or smaller, bit-exact against avmdec v1.0.0 and dav2d
  (re-run on 2026-11-23 against AVM's frame hashes: 30/30; not part of the record above). No AV2
  conformance suite exists; this is a small check corpus, not conformance. **Not yet**: 10-bit, monochrome
  (4:0:0), several tiles or tile groups, references of another size (reference scaling), multi-frame
  headers, bridge frames, wedge compound when motion fields are stored, embedded/extended layers, and the
  bilinear chroma phase of odd eighth-sample vectors.

In the browser (the WebAssembly module, plain and relaxed SIMD), on the suites' 4:2:0 progressive streams:
HEVC 147/147, AV1 240/240, VP9 305/305 (10- and 12-bit included) and H.264 108/113 (same record), and VVC
221/221 by whole-output MD5 against VTM (d010259d0: the record's 214/221 was a runner fault on seven streams
whose picture size changes or whose fields are output as frames). The 5 H.264 streams not decoded use FMO or
SP/SI slices, which the native decoder does not support either. Monochrome (4:0:0) and interlaced streams
are left out of every browser run, so monochrome decoding in the browser is untested; AV2 has no browser
run in the record.

Browser decode time against ffmpeg.wasm 0.12.10 (the official build: FFmpeg 5.1.4 with `--disable-asm`, so
without FFmpeg's hand-written SIMD kernels), decode alone on both sides, stream by stream, is recorded in
[docs/results/wasm-yardstick-2026-11-23](results/wasm-yardstick-2026-11-23/). 10-bit decoding has SIMD on
AVX2, AVX-512, NEON and WASM for AV1 and VP9, plus WASM for the kernels HEVC and VVC share; VVC's native (x86/Arm) and H.264's 10-bit paths
are scalar so far.

## What is not done

- **Single-threaded** encode and decode; no slices or tiles beyond one per frame.
- **8-bit 4:2:0 encoders.** The encoders code 8-bit 4:2:0 only (no 10-bit, no 4:4:4, no HDR). The decoders
  also decode 10-bit (HEVC Main 10, AV1 10-bit, VP9 profile 2 at 10 and 12 bits, H.264 High 10, VVC Main 10),
  natively and in the browser.
- **Decoders:** no decoder decodes every valid stream of its codec's main profiles yet; the counts, which
  suites are official and which are encoder test vectors, are under [Conformance](#conformance). Not yet
  decoded (valid streams of the main profiles): H.264 interlace (field pictures, MBAFF) and pictures above
  level 5.2; AV1 quantizer matrices, block loop-filter deltas, segment features other than the quantizer,
  lossless segments in mixed frames, odd frame sizes and switch frames (fix in progress); VVC reserved NAL
  types, `nuh_layer_id` above 0 and a VPS id above 0 (rejected instead of ignored); VP9 profiles 1 and 3;
  AV2 10-bit, monochrome, several tiles, reference scaling, multi-frame headers, bridge frames and layers.
  Not checked: the AV1 Argon conformance streams; no AV2 suite exists (30 streams only); monochrome in the
  browser. Other profiles not supported: H.264 Baseline/Extended tools (FMO/ASO, SP/SI) and 4:2:2/4:4:4;
  HEVC range extensions; VVC 4:2:2 / 4:4:4, multilayer, palette and ACT.
- **Speed gaps:** decoding in process with one thread (2026-11-23, decode alone on both sides, release 0.6.0's
  PGO build, [record](results/decoder-speed-2026-11-23/README.md)), on one Zen 5 core our H.264 decoder took
  1.10–1.18x FFmpeg's time, our AV1 decoder 1.16–1.43x dav1d's, our VP9 decoder 1.00–1.04x libvpx's (0.91–1.01x
  FFmpeg's) and our AV2 decoder 1.41–1.44x dav2d's. On one Jetson Orin Nano core, the same way: H.264
  0.92–0.98x FFmpeg's, AV1 0.94–1.16x dav1d's, VP9 1.01–1.03x libvpx's (1.05–1.07x FFmpeg's), AV2 0.70–0.86x
  dav2d's; SVT-AV1 M12/M13 and
  x264's fast presets encode faster than our Default presets; VVC Default is the slowest of our Defaults at 720p
  on one Jetson Orin Nano core (the charts' speed axis; speed depends on content).
- **Known quality weak points:** fine pristine texture (water, foliage), where x264 and x265 need fewer bits;
  H.264 flicker; VP9 trails our HEVC; SVC costs 6–14 points of BD-J against single-layer streams; AV2 and VVC are
  young (rate control and quality work ongoing). See [known weaknesses](results/README.md#known-weaknesses).

## Security and fuzzing

The decoders are written in safe Rust (no `unsafe` outside the SIMD kernels and the C ABI), so memory-safety bugs
are unlikely outside those kernels, but a malformed stream can still make a decoder panic (abort) or misbehave.
Each decoder has a `cargo fuzz` target (`fuzz/`) that drives the browser, C API and command-line decode paths,
seeded from our streams and the conformance and test-vector streams. On 2026-11-13/14 (an earlier build, before the
10-bit, general VVC and general AV2 decoders) every decoder had 2.1 to 3.8 worker-hours of fuzzing in two rounds
(about 2.75 million executions; the second round used small inputs and per-codec dictionaries), and every corpus
input was replayed through the WASM build without a trap; fuzzing and a review of every stream-derived index,
size and sum found and fixed seven issues (two crashes reachable from every entry point: an out-of-range HEVC
parameter-set id and non-increasing AV1 film grain points; an AV1 deblocking overflow on frames whose size is not
a multiple of 8; a missing AV1 picture-size cap; and arithmetic that could overflow before its range check in
H.264, HEVC and VVC header parsing, two of them only on 32-bit WASM). On 2026-11-23, at 98069a8b3 (release 0.6.0's
decoders), a third round ran each of the six decoders for 25 minutes with eight workers (about 3.3 worker-hours
each, 20 in all, about 1.3 million executions), seeded with the 10-bit conformance vectors, FFmpeg-encoded 8- and
10-bit streams and our own: no finding in H.264, HEVC, VVC, AV1 or VP9 (VP9: five timeouts on declared pictures of
up to 2896x65536, whose decode time scales with their area); in AV2, one crash (three inputs, one cause): a band-only CCSO with `max_band_log2` 7 indexed past the 72-entry offset table and
panicked, in release builds too. Both are fixed in 71f6d92b2: AV2 now rejects band-only CCSO beyond 64 bands as
a stream error, as AVM does, and VP9 caps the picture area at level 6.2's 35,651,584 samples; the inputs are
regression tests. Every round-3 corpus input was then replayed through the WASM build of all six decoders
without a trap (the build before the fix traps on exactly the three AV2 inputs). That is a bounded search, not a proof: see
[results/fuzzing.md](results/fuzzing.md) for the numbers and what was not tested. No corpus is published. Do not
expose these decoders to untrusted input in production. The C ABI catches panics at the boundary. The browser demo
decodes only streams it encoded itself and the project's hosted streams.

## Why build this at all?

To see how far one shared, low-delay engine can go across six formats on a small CPU, and to have our own
browser decoders for our own streams. The coding tools are the standards'; the ideas are largely those of the
projects we credit. What is new here is the combination, the shared engine, and an account of building it with AI
agents ([RETROSPECTIVE.md](RETROSPECTIVE.md)).
