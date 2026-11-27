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

- the Rust compiler and `#![forbid(unsafe_code)]` in every crate except `nc-dsp` (SIMD kernels, assembly and
  their dispatch) and `nc-capi` (the C ABI); every `unsafe` block there carries a `// SAFETY:` comment, and every SIMD kernel has a scalar
  reference and a bit-exactness test (`nc-checkasm`);
- `cargo clippy -D warnings`, `cargo fmt`, unit and round-trip tests (run in a post-landing check; not every
  release has a recorded pass of them);
- after each change lands, short test streams of every shipped preset of every format are decoded by two
  independent decoders and ours and compared frame by frame with the encoder's own pictures (below); this
  check runs beside landing, not as a gate;
- an adversarial review of the public claims ([review notes](review/adversarial-review-2026-11-13.md)).

None of this proves the code is correct or well designed; it catches crashes, mismatches with the standards'
decoders and regressions in what is measured.

**Is any of it copied from GPL code?** We have found none. x264, x265, FFmpeg and libde265 are run as external
programs; one benchmark tool (`tools/anchors/libavcodec_bench.c`, not shipped) links FFmpeg's libraries to time
its decoders. A spot check of distinctive x264/FFmpeg functions, tables and names (cost tables, AQ, psy-rd,
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

**What is compared?** The charts follow benchmark procedure BP-4
([benchmark-procedure.md](benchmark-procedure.md); each encoder's command, flag by flag with its reason, in
[benchmark-fairness.toml](benchmark-fairness.toml)). In short:

- **CBR only**, low delay (no B-frames, no reordering, no lookahead), one thread, a 1 s buffer that starts full,
  peak rate equal to the target. Ours runs `--strict-buffer`: it never passes the 1 s buffer.
- **Rates:** 300, 500, 800 and 1200 kbps at 720p30, scaled per clip by (pixels / 1280×720)^0.75 ×
  (fps / 30)^0.5, and never above half the source's bitrate. Nothing above 1200 kbps (720p30-equivalent) is
  measured.
- **Windows:** 150 frames per clip, starting with a key frame, with a key every 150 frames; on sources long
  enough, 600 frames with a key every 150 as well.
- **Each reference at its own real-time defaults** (for libvpx and libaom, their projects' real-time example
  settings), with only the two shared constraints above. Every reference is built with LTO and PGO where its
  build allows (fairness record). Versions: x264 0.165, x265 4.2, SVT-AV1 v4.3.0, libaom v3.15.3, libvpx
  v1.16.0, openh264 v2.7.0, Kvazaar v2.3.2. VVenC is not charted: its only single-pass rate control is VBR with
  a look-ahead.
- **Zero line:** x264 `medium` in its best tune for each metric: `--tune zerolatency,psnr` for BD-PSNR, BD-VMAF
  and BD-J, `--tune zerolatency,ssim` for BD-MS-SSIM. Default-tuned x264 `medium` needs about 14 % more bits
  (BD-J) on the same chart, so the PSNR tune is the harder zero line for us.
- **Every scored stream is decoded**, ours and the references' alike; no score is taken from an encoder's own
  reconstruction (details below).

**Is the comparison fair to them?** It is one configuration: the one our use case needs (no B-frames, no
lookahead, one thread, CBR with a 1 s buffer). Those projects are general-purpose and are usually run with more
threads, lookahead or B-frames, where they will do better. openh264 is charted with CAVLC (Constrained Baseline)
and with CABAC. libvpx and libaom also have labelled `+drop` rows with their own frame dropping on; the main rows
code every frame, as every encoder does. Our presets are tuned at these rates on clips of this kind; no
reference is. The remaining asymmetries, per encoder, are listed in the fairness record and in
[section 16 of the procedure](benchmark-procedure.md#16-known-limitations).

- **Rate control is part of what is compared.** Every encoder runs its own low-delay CBR, so a margin mixes coding
  efficiency with how much each rate control costs. History (2026-11-14 builds, development clips, preset names
  of that date, an earlier procedure): our AV1 Default at fixed QP against fixed QP needed 33.9% fewer bits than
  PSNR-tuned x264 `medium` (38.3% in CBR against CBR); CBR cost our encoders 10.7–21.5% against their own fixed
  QP, x264 `--tune psnr` 48.6% and default-tuned x264 70.5%
  ([rc-cost-2026-11-14.csv](results/rc-cost-2026-11-14.csv)). Our H.264 Default was then level with x264
  `medium` at fixed QP (+0.4% / +2.6% against `--tune psnr`).

The current figures for every preset against that anchor, with their clip counts and build, are in the
[chart at the top of the README](../README.md) and on the
[results pages](https://cribbinsb.github.io/nanocodec/results/codecs.html).
We would welcome better settings for any reference within the same constraints.

**Why a composite metric ("BD-J")?** We tune on it, so we report it, but every table also gives the plain
BD-PSNR-YUV and BD-VMAF numbers ([full table](results/bd-vs-speed.md), [per metric](results/README.md#per-metric)).
BD-J is the mean of the BD-rates on PSNR-YUV (Y:U:V weighted 6:1:1, each plane's per-frame PSNR capped at 60 dB,
libvmaf's 8-bit cap; 100 dB in charts published before 2026-11-20), VMAF and MS-SSIM, each against the zero line's tune for that metric, averaged with equal weight per clip.
Charts rendered before 2026-11-23 (release 0.6.0's) also average the full overlap of each curve with its
lower-rate half, so the low rates count twice; charts rendered from 16de42521 on use the whole overlap and give
the BD by rate regime (up to 600 kbps, and 600 kbps–4 Mbps, at 720p30 scaled) beside it
(`eval/nceval/chart.py`, `eval/nceval/jreal.py`). VMAF is Netflix's model
`vmaf_v1.0.16` (the "v1" model of libvmaf 3), computed by libvmaf 3.2.1 on the CPU (points scored on rented machines) or by VMAFx, a GPU (CUDA) build of
libvmaf (points scored on our PC), at each clip's own size: most held-out clips are below 720p, so these VMAF values are not
comparable with figures from the 1080p VMAF protocol. Does the GPU fork change the scores? An independent cross-check of 200 held-out points (stock libvmaf 3.2.1 on CPU with `vmaf_v1.0.16_3d0h.json`, plus our MS-SSIM, against the GPU path on the same streams) found PSNR-YUV within 7e-15 dB, VMAF identical, MS-SSIM within 3.9e-6 and identical packet sizes. One caveat: the GPU scorer was not run-to-run reproducible on 8 of 144 chart points (sequence VMAF moved by up to 0.037, single frames by up to 5.5, on bit-identical pictures), so scores are reproducible to about 0.04 VMAF at sequence level; this is a known, open scorer issue. MS-SSIM is scored on every frame
by our reimplementation of libvmaf's float MS-SSIM (agreeing to about 1e-6).

**Are your streams actually decoded, or do you score your own reconstruction?** In the charts, decoded: every
scored stream, ours and the references', is decoded and scored on that output. In the BP-4 series our own
decoder (the frozen 23c962d8b build) decoded every family's streams; where it exited with an error, the codec's
standard decoder did (the pinned FFmpeg 8.0.1, with dav1d for AV1; avmdec for AV2), and the score record names
it. A sample of the series' streams, plus 13 more reference streams checked in the 2026-11-27 integrity audit,
decoded bit-identically in ours and the standard decoders (the
[method page](https://cribbinsb.github.io/nanocodec/results/method.html) lists them). VMAF v1 and MS-SSIM are
scored on our PC's GPU by VMAFx, checked once per scorer build against libvmaf 3.2.1 on CPU; PSNR comes from
exact squared-error sums with libvmaf's 60 dB cap. Separately, the landing check decodes short streams of every
shipped preset in two standard decoders and ours, frame by frame against the encoder's reconstruction. Our
development screens (not published) score the encoder's reconstruction, which that check shows is what a
standard decoder outputs.

**Only 150 frames?** Each clip is scored over 150 frames (5 s at 30 fps) starting with a key frame, and, where
the source is long enough (40 of the held-out clips), also over 600 frames with a key every 150 frames, so
mid-sequence keys count. History (commit 8b492382, 2026-11-13, an earlier procedure with one key per window): on
71 held-out sources over 600 frames with a key every 10 s, our savings against default-tuned x264 `medium` were
6–10 points smaller than over 150 frames (AV1 −47.1% → −40.8%, HEVC −48.3% → −40.9%, H.264 −29.8% → −19.8%),
while the references moved less or gained.

**Why these bitrates?** Our use case is low-bitrate streaming. Each clip's ladder is scaled to its pixel rate and
capped at half its source's bitrate, because many sources are already compressed and coding above that measures
the source's artefacts. Our savings are largest at low rates: in an earlier procedure (0.6.0), on the 10–11
held-out clips with room for 2,400 and 4,000 kbps (720p30), BD-J over 600 kbps–4 Mbps was about half the
chart-rate figure (AV1 −24.8% vs −47.2%, H.264 −9.3% vs −29.0%), and the references' gaps shrank too. Read every
headline number as "on the 300–1200 kbps ladder at 720p30, scaled down for smaller pictures".

**Which clips?** The headline is BP-4's held-out set: 97 camera clips (`eval/clipset/heldout_2026-11-23b.manifest.json`),
84 of them audited once and never tuned on, 13 unseen. The set has been measured at several builds, each
reading logged in [the sealed log](results/sealed-log.md). The README chart averages every encoder over the
common set, the clips where every encoder's curves are well formed (92 of 97 on the chart's build); the clips
it lacks are listed with the reason on the [Clips page](https://cribbinsb.github.io/nanocodec/results/clips.html).
The results pages also give a video-call set (VC-1K, 100 clips) and the AOM common test conditions' clips. Many
sources are not redistributable, so the clips themselves cannot be published; the clip lists are in
`eval/clipset/`. "Ours" sources are x264 re-encodes of the original videos; their effect on the ranking is not
measured yet.

**Did you overfit?** Our presets were found by automated search on development clips, disjoint from the
held-out set by content (`nceval.content_id`), so the headline is measured on clips the search never saw. The
held-out set's results have been seen by the people and agents choosing what to work on next, which the
sealed log records. History (commit 8b492382, 2026-11-13, an earlier procedure): BD-J against default-tuned x264
`medium`, held-out (137 fresh clips) vs development (11 clips): AV1 Default −45.3% vs −50.7%, HEVC −47.0% vs
−49.6%, VVC −51.1% vs −54.4%, H.264 −28.7% vs −24.0%; the references moved more (SVT-AV1 M12 `--rtc` −29.1% vs
−34.9%, x265 `medium` −30.3% vs −40.7%). So the size of a saving depends on the clips, and our position among
the encoders tested held.

**Speed: how measured?** From the same CBR encodes that are scored (each timed stream's SHA-256 must equal the
scored one). Each encode's CPU time (user + sys) is measured on a bare-metal AWS machine, one per chip: c8i.metal-48xl
(Intel Xeon 6975P-C, Granite Rapids) and c8g.metal-24xl (Graviton 4), while half the machine's physical cores
run the same encoder family ("loaded speed", the way a server full of streams runs). Against an idle machine
this slows encoders by ×1.00–1.07 on Intel and ×1.01–1.23 on Graviton, our fast presets and SVT-AV1's among
the most. The figure is megapixels per CPU-second, geometric mean over the clips; it is not a frames-per-second
or real-time claim. Every machine runs a check set when it starts, and a machine outside ±1 % of the others is
replaced. Ours on Graviton uses NEON plus SVE2 for some kernel families; each encoder uses its own best SIMD.

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

In the BP-4 charts ours runs `--strict-buffer` and never passes the 1 s buffer; each encoder's buffer use is on
the Rate control page. History: an independent audit of the 0.6.0 charts' points (42 per encoder, 150 frames),
**at the builds those charts were measured on, before that change** (the key loan was then repaid over 10 s and the long-run rate was not held under the
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
[Rate control page](https://cribbinsb.github.io/nanocodec/results/compliance.html).

**Do you skip frames?** Our encoder does not code an input frame that would not change the picture (exact or
near-exact repeats, typical of recompressed or frame-doubled sources). Such "held" frames cost no bits and no
encode time and are scored as the repeated picture, which is what a viewer sees. The references code every frame
(cheaply, as skip frames). In the BP-4 charts ours codes a held frame as an all-skip frame (`hold.code_skip=1`),
so every encoder codes every input frame on the main rows; only the labelled `+drop` rows of libvpx and libaom
drop frames.

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

Our decoders also decode other encoders' streams, natively and in the browser, 8-bit and 10-bit, but
**none of them decodes every valid stream of its codec's main profiles yet**. Below, each
line names the suite, says whether it is an official ITU-T/ISO conformance suite or an encoder project's
test vectors, gives the count, and lists every valid-stream feature of the main profiles that decoder
still rejects. Streams decoded natively and compared picture by picture with the reference decoder named,
on 2026-11-23 at 98069a8b3 ([record](results/conformance-2026-11-23/summary.json)); the gap lists are
from the [audit of 2026-11-27](analysis/decoder-compliance-gaps-2026-11-27.md). Every gap, with its status (in progress, open, or proposed
exception with its reason), is in the [decoder compliance register](decoder-compliance.md), which a check
keeps in step with the decoders' rejection paths.

- **HEVC** (Main, Main Still Picture, Main 10): 147/147 of the official ITU-T/ISO JCT-VC HEVC_v1
  conformance suite (H.265.1; 136 8-bit, 11 10-bit), against HM. No known gap in these profiles.
  **Range extensions** (4:0:0, 4:2:2, 4:4:4, 8 to 16 bits and every RExt tool): 49/49 of the JCT-VC RExt
  conformance set (15 8-bit, 13 10-bit, 15 12-bit, 6 16-bit; 4:0:0 3/3, 4:2:0 8/8, 4:2:2 7/7, 4:4:4 31/31),
  against HM 19.0 built with high bit depth and the streams' shipped MD5s (2026-11-27, not part of the
  record above). **Not yet**: inter prediction above 12 bits (Monochrome 16; no conformance stream codes
  it). Not supported: screen content coding; the SCC, MV-HEVC, SHVC and 3D-HEVC suites are not run.
- **VVC** (Main 10, Main 10 4:4:4 and their Still Picture profiles: 4:0:0, 4:2:0, 4:2:2 and 4:4:4,
  palette and ACT): 272 of the 283 streams of the official ITU-T H.266.1 first-edition conformance suite
  (11 8-bit, 261 10-bit), against VTM 24.0: all of its single-layer streams but HRD_B_2 (VTM itself fails
  it) and VPS_C_2 (not run). The other 9 are multilayer streams (other profiles, not supported). STILL_B_1
  and STILL444_B_1 are compared with VTM's output, which differs from the MD5 in their zips. **Not yet** (valid Main 10 streams rejected): reserved
  and unspecified NAL unit types and NAL units with `nuh_layer_id` above 0 are an error instead of being
  ignored (§7.4.2.2), and an SPS with `sps_video_parameter_set_id` above 0 is rejected. Also 138 of 138
  short 8-bit streams we coded with VTM and VVenC to cover the tools one by one (our own streams, not a suite).
  Range extensions (Main 12, Main 12 4:4:4, Main 16 4:4:4 and their Intra/Still Picture profiles): 72 of
  the 116 streams of the H.266.1 second-edition range-extension set we run (suite `vvc-rext`), against
  VTM built for high bit depth: all 72 12-bit streams (4:0:0 to 4:4:4, every range-extension tool). Not
  supported: 16-bit samples (the other 44 streams are refused).
- **H.264** (Baseline, Constrained Baseline, Extended, Main, High, High 10, High 4:2:2, High 4:4:4
  Predictive and their Intra profiles, CAVLC 4:4:4 Intra; 8 to 14 bits; progressive and interlaced:
  field pictures and MBAFF frames): all 190 of the 190 streams of the official ITU-T H.264.1 conformance
  suite's AVCv1, FRExt and Professional (High 10) sets, against JM (2026-11-27 natively, at the MBAFF
  landing; the record above had 110), the 75 interlaced ones (37 field-picture, 38 MBAFF) among them.
  Interlaced streams of x264 `--interlaced` and ffmpeg `-flags +ilme+ildct` (CABAC and CAVLC, B frames,
  spatial and temporal direct, weighted prediction) decode bit-exactly against ffmpeg. **Not yet**: MBAFF
  frames with slice groups or data partitioning (Extended only; no stream of either in the suites). The
  suite's 4:2:2 and 4:4:4 sets (fetched as their own suite, `h264-chroma`): 34 of 38 against JM
  (2026-11-27 natively; 8-bit 7/7, 10-bit 14/18, 14-bit 13/13: every progressive one, separate colour
  planes and lossless included, and the 2 field-picture ones); **not yet** the 4 MBAFF ones (MBAFF frames
  in 4:2:2 and 4:4:4 are refused). Every 4:4:4 stream of the suite codes its colour planes separately, so
  4:4:4 with Cb and Cr coded like luma and lossless coding in every chroma format are covered by ten x264
  streams we made (`h264-x264-chroma`, 10/10, JM and ffmpeg agree on each). MVC and SVC streams are not
  fetched (other profiles, not supported).
- **AV1** (Main, High and Professional: 8- to 12-bit, 4:2:0, 4:2:2, 4:4:4 and 4:0:0), natively on
  2026-11-27 at ec8cd468b ([record](results/conformance-2026-11-27-av1/summary.json)):
  - 3343 of 3343 streams of the official AOM **Argon** conformance suite (v2.1.1), in 18 sets: for each
    of profiles 0, 1 and 2, the core, core special, not-Annex-B, not-Annex-B special and stress sets
    (3008 streams) each match aomdec's output on the same stream (`aomdec --rawvideo`; not Argon's own
    MD5s, which hash aomdec's all-layers output), and the error sets' 335 deliberately invalid streams
    end without a crash or hang (their output is not compared). Some streams have frames larger than
    level 6.3 allows: our decoder refuses those by default (a bound for untrusted input) and decodes them
    with `--unbounded 1`, as this run did. The first profile 0 run that day had 790 of 889: 61 streams
    decoded to the wrong output, from two bugs since fixed (the loop filter's chroma transform size in
    lossless blocks, and intra block copy near the frame's right and bottom edges). The large-scale-tile
    sets were not run (not supported). The nightly verifier runs every set but stress.
  - 242/242 libaom v3.15.3 test vectors (176 8-bit, 66 10-bit), against aomdec: an encoder project's
    test vectors, not a conformance suite.
  - 28/28 short streams we made with libaom (aomenc, and its ROI example encoder) and SVT-AV1 v4.3.0 for
    the Main tools those vectors lack (quantizer matrices, superblock quantizer and loop-filter deltas,
    segmentation with loop-filter, reference and skip features, lossless, odd frame sizes, switch
    frames), each kept only where aomdec and dav1d agree, against aomdec. Our own streams, not a suite.

  - High and Professional profiles (8-, 10- and 12-bit; 4:2:0, 4:2:2, 4:4:4, monochrome, RGB): libaom
    ships no such vectors, so 27 streams we made with aomenc and ffmpeg's libaom
    (`tools/dev/av1_profile_streams.sh`: 4:4:4 and 4:2:2 at 8/10/12 bits, 12-bit 4:2:0 and monochrome,
    sRGB and identity-matrix RGB, screen content with palette and intra block copy, film grain, superres,
    lossless, quantizer matrices, 128x128 superblocks, odd sizes) match `aomdec --rawvideo` 27/27 (High
    11/11, Professional 16/16; dav1d agrees on all but monochrome, which it writes as luma only). Our own
    streams, not a suite (Argon's profile 1 and 2 sets, above, are).

  No Main, High or Professional tool is refused any more (quantizer matrices, block loop-filter deltas,
  all segment features, lossless segments, odd frame sizes and switch frames, and the High and
  Professional profiles were added on 2026-11-27). Large-scale tile decoding is not supported.
- **VP9** (profiles 0 to 3, 8- to 12-bit, 4:2:0, 4:2:2, 4:4:0, 4:4:4): 314/314 libvpx v1.16.0 test
  vectors (306 8-bit, four 10-bit, four 12-bit; 303 profile 0, 3 profile 1, 2 profile 2, 6 profile 3),
  against vpxdec and the vectors' shipped MD5s. These are an encoder project's test vectors (the de facto
  suite; VP9 has no official one), and only eleven cover profiles 1 to 3; 26 more profile 1 and 3 streams
  we made with ffmpeg's libvpx-vp9 (every format and depth, sRGB, odd sizes, lossless, alt-ref) also
  decode bit-exact against vpxdec (`docs/handoff/chroma-formats.md`).
- **AV2** (a draft standard, AVM v1.0.0): 117 of the 128 streams of our check corpus bit-exact against
  avmdec v1.0.0, 98 of them also against dav2d 237229e (the other 19 are dav2d test vectors newer than that
  dav2d, which decodes them differently; their MD5s in the test data agree with avmdec), on 2026-11-27 at
  df87b4f57 (`tools/dev/av2_foreign/compare.py`; not part of the record above). The corpus: dav2d's 8-bit
  and deeper test vectors and avmenc v1.0.0 streams we made, at 8 and 10 bits, 4:2:0 and 4:0:0, odd and
  non-multiple-of-8 sizes, several tiles and tile groups, references of another size, multi-frame
  headers, bridge frames, embedded layers, 256x256 superblocks and film grain. The 11 not counted:
  seven 4:2:2/4:4:4 streams (other profiles, rejected by name) and four 12-bit vectors that avmdec v1.0.0
  itself refuses (our decoder decodes 12-bit, unchecked). No AV2 conformance suite exists; this is a
  check corpus, not conformance. **Not yet**: extended layers (several streams multiplexed; avmenc
  cannot make them), 4:2:2 and 4:4:4.

In the browser (the WebAssembly module, plain and relaxed SIMD), on the suites' 4:2:0 progressive streams:
HEVC 147/147, AV1 240/240, VP9 305/305 (10- and 12-bit included) and H.264 108/113 (same record), and VVC
221/221 by whole-output MD5 against VTM (d010259d0: the record's 214/221 was a runner fault on seven streams
whose picture size changes or whose fields are output as frames). Since 2026-11-27 the H.264 browser run
takes every stream of the suite, interlaced and 4:0:0 ones included: 190/190 in both builds (gate
profile, at the MBAFF landing). Monochrome streams of the other codecs are still left out, so their
monochrome decoding in the browser is untested; AV2 has no browser run in the record.

Browser decode time against ffmpeg.wasm 0.12.10 (the official build: FFmpeg 5.1.4 with `--disable-asm`, so
without FFmpeg's hand-written SIMD kernels), decode alone on both sides, stream by stream, is recorded in
[docs/results/wasm-yardstick-2026-11-23](results/wasm-yardstick-2026-11-23/). 10-bit decoding has SIMD on
AVX2, AVX-512, NEON and WASM for AV1 and VP9, plus WASM for the kernels HEVC and VVC share; VVC's native (x86/Arm) and H.264's 10-bit paths
are scalar so far.

## Which resilience tools does each codec support?

Checked by running the encoder (2026-11-27, `tools/ci/realuse_conformance.sh`: every stream below decodes
bit-exactly against our reconstruction in both standard decoders of its codec and in ours, with a simulated
loss where the row names one). "Yes" means the encoder codes it, alone and combined with the other columns,
except where a cell says otherwise.

| Codec | Temporal layers | Long-term reference (LTR) | Intra refresh | Key frame on request | Loss recovery (feedback) |
|---|---|---|---|---|---|
| H.264 | yes (2-3, fixed or adaptive) | yes | yes | yes | from the LTR, else a key frame |
| HEVC | yes | yes | yes | yes | from the LTR, else a key frame |
| VVC | yes | yes | yes | yes | from the LTR, else a key frame |
| AV1 | yes | yes | yes¹ | yes | from the LTR, else a key frame |
| VP9 | yes | yes | no² | yes | from the LTR, else a key frame |
| AV2 | yes | yes | no² | yes | from the LTR, else a key frame |

- How the LTR is signalled: H.264 by long-term marking (MMCO), HEVC and VVC as a long-term reference picture
  (`long_term_ref_pics_present_flag`, `sps_long_term_ref_pics_flag`), AV1, VP9 and AV2 as a held reference slot.
- With temporal layers, LTRs are marked on base-layer frames and intra refresh sweeps across the base-layer
  frames only (as many as cover the period, at least two). With an LTR, intra refresh restarts its sweep at a
  recovery frame.
- ¹ AV1's intra refresh repairs a decoder whose reference pictures are wrong; a standard decoder (dav1d,
  aomdec) that missed a whole frame stops there, because inter frames load the probability tables their
  reference saved: after a dropped frame, recovery is by LTR or key frame.
- ² Refused: VP9's backward-adapted probabilities and previous-frame motion vectors, and AV2's
  `primary_ref_frame`, carry state a lost frame changes, so vpxdec and avmdec do not recover by a sweep
  ([handoff](handoff/loss-resilience.md#intra-refresh-after-a-dropped-frame-2026-11-15)).
- Each sweep's first frame is marked as a recovery point, so a decoder joining mid-stream knows when its
  pictures become exact: H.264 and HEVC write a recovery-point SEI, VVC codes the frame as a GDR picture.
  VVdeC-based players skip the first refresh sweep after a stream start (they withhold the first GDR
  picture's recovering pictures even after an IDR picture); the stream is spec-conformant per H.266
  §8.1.1/§8.1.2, and FFmpeg and our decoder show every frame.
- Not combined: intra refresh with spatial layers (SVC) or with an explicit second reference
  (`refs.second` 1).

## What is not done

- **Single-threaded** encode and decode; no slices or tiles beyond one per frame.
- **8-bit 4:2:0 encoders.** The encoders code 8-bit 4:2:0 only (no 10-bit, no 4:4:4, no HDR). The decoders
  also decode 10-bit (HEVC Main 10, AV1 10-bit, VP9 profile 2 at 10 and 12 bits, H.264 High 10, VVC Main 10),
  natively and in the browser, and natively HEVC's range extensions (4:0:0 to 4:4:4, up to 16 bits).
- **Decoders:** no decoder decodes every valid stream of its codec's main profiles yet; the counts, which
  suites are official and which are encoder test vectors, are under [Conformance](#conformance). Not yet
  decoded (valid streams of the main profiles): AV1 quantizer matrices, block loop-filter deltas, segment features other than the quantizer,
  lossless segments in mixed frames, odd frame sizes and switch frames (fix in progress); VVC reserved NAL
  types, `nuh_layer_id` above 0 and a VPS id above 0 (rejected instead of ignored);
  AV2 extended layers.
  Not checked: the AV1 Argon conformance streams; no AV2 suite exists (a 128-stream check corpus only);
  AV2 12-bit and the AV2 browser path at 10 bits; monochrome in the
  browser. Other profiles not supported: H.264 MBAFF frames with slice groups or data partitioning
  (Extended only) and 4:2:2/4:4:4;
  HEVC screen content coding; VVC multilayer and 12/16-bit (range extensions).
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

The decoders are written in safe Rust (no `unsafe` outside the DSP crate, which holds the SIMD kernels, assembly
and their dispatch, and the C ABI), so memory-safety bugs are unlikely outside those, but a malformed stream can still make a decoder panic (abort) or misbehave.
Each decoder has a `cargo fuzz` target (`fuzz/`) that drives the browser, C API receiver (its Rust session, not the `extern "C"` layer) and
command-line decode paths,
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
expose these decoders to untrusted input in production. The C library, built as documented (`cargo build --profile capi`, which unwinds), returns `NC_ERROR_INTERNAL`
from a panic instead of aborting the host (`tools/ci/capi_check.sh` forces one through the C ABI); a `--release`
build aborts on a panic. The browser demo
decodes only streams it encoded itself and the project's hosted streams.

## Why build this at all?

To see how far one shared, low-delay engine can go across six formats on a small CPU, and to have our own
browser decoders for our own streams. The coding tools are the standards'; the ideas are largely those of the
projects we credit. What is new here is the combination, the shared engine, and an account of building it with AI
agents ([RETROSPECTIVE.md](RETROSPECTIVE.md)).
