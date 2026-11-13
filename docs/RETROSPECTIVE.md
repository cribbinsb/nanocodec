# Building nanocodec: a retrospective

nanocodec is a real-time video encoder for live camera footage (bodycams,
fixed cameras, drones, crowds) that has to stream at low bitrate from a small
ARM board to a browser. It codes H.264, HEVC, AV1, VP9, AV2 and VVC from one
shared engine, and ships its own decoders, which also run in the browser as
WebAssembly. This is the story of how it got there: what we set out to do,
what surprised us, what we tried, and which ideas paid off. Every number here
comes from a measurement recorded at the time, in `docs/results/` or in our
internal working notes (not published).

> **Dated.** Written 2026-11-10 to 2026-11-11 and only lightly updated since.
> Its numbers are from the builds and scores of that time (some sections use
> the earlier VMAF NEG composite) and differ from the current
> [results](results/README.md). They were measured on the 11 VAL clips, which
> were also used to confirm which changes were kept; on 71 held-out clips
> every encoder's saving against x264 `medium` is smaller (see the
> [held-out check](results/heldout-chart-2026-11-12.md)). The code was written
> by AI coding agents under one person's direction; "we" below means that
> team.
How the code is organised is in [ARCHITECTURE.md](ARCHITECTURE.md). The
current results are in [results/README.md](results/README.md).

The short version: at real-time settings, on our 11 VAL clips, our AV1 needed
about half the bits of x264 `medium` for the same quality (−49.8% at the time;
−42.5% on 71 held-out clips with the 2026-11-12 builds), and our H.264 13.6%
fewer.
About three quarters of the AV1 gain comes from AV1's coding tools. The other
quarter comes from how the encoder is built.

## 1. What we set out to do

The brief was narrow on purpose:

- **Real time on one core.** 720p30 live video on one Cortex-A78AE core of a
  Jetson Orin Nano, the edge device we target. The PC (a Ryzen 9 5950X) is
  the secondary platform.
- **Low delay.** Only forward prediction: every frame predicts from frames
  already sent, so the encoder never waits for future frames. No B-frames,
  no lookahead.
- **Real-time rate control.** Each stream gets a bitrate target and a 1 s
  buffer. It must not overshoot, because the network behind it can't absorb
  bursts.
- **Strictly standard bitstreams.** Every stream must decode bit-exactly in
  two independent standard decoders (for example ffmpeg and JM for H.264, or
  dav1d and aomdec for AV1), and also in our own decoder.
- **Targets set by the baselines.** First openh264, the H.264 encoder this
  replaces. Soon after that, SVT-AV1's fast real-time mode (M12 `--rtc`), a
  much stronger encoder.

Early on we measured SVT-AV1 M12 at −48.6% BD-rate against openh264 for
1.24–1.48× its CPU. *BD-rate* is the average change in bitrate at equal
quality between two rate–quality curves; negative means fewer bits. The
owner's verdict was "SVT-AV1 is truly excellent … use that as your primary
guide." So SVT-AV1's real-time pipeline became the reference architecture,
and its `--rtc` presets remain the reference we measure our HEVC and AV1
against.

## 2. The shape of a real-time encoder

A video encoder splits each frame into blocks and, for each block, predicts
it, either from pixels already coded in the same frame (*intra*) or from an
earlier frame shifted by a *motion vector* (*inter*). It then codes only the
difference. That residual goes through a transform (a DCT-like change of
basis that packs energy into a few coefficients), is *quantised* (divided by
a step size and rounded, which is where quality is lost), and is
entropy-coded. A *loop filter* then smooths block edges in the
reconstruction, which later frames predict from.

What separates encoders is how they *choose*: block sizes, prediction modes,
vectors and quantised levels. Slow reference encoders do *rate–distortion
optimisation* (RDO). They code many candidates for real and keep the one with
the lowest distortion + λ·bits, where λ prices a bit in units of distortion.
At real-time speed you cannot afford that. A real-time encoder:

1. **analyses the source once**: a ½/¼-resolution pyramid, per-8×8
   statistics and a still-block map;
2. **searches motion once**, hierarchically on the source frames, and shares
   the vectors;
3. **plans the partition from statistics**: SVT-AV1's "very light PD0"
   estimates each block size's cost from the motion search's prediction
   variance and picks the tree before coding anything;
4. **decides modes cheaply**: candidates are ranked by residual variance
   plus λ·estimated bits, and only the winner is refined to sub-pixel
   precision and coded for real.

nanocodec does exactly that, once, in a codec-independent engine. Each codec
is a backend behind one trait (`CodecBackend`) that supplies syntax, bit
costs, transforms and filters. So an engine improvement lands in six codecs
at once.

## 3. What surprised us

**The metric was flattering us.** For a while our HEVC and AV1 looked well
ahead of SVT-AV1 on VMAF. An adversarial review of the chart found that VMAF
v1's "enhancement gain", which rewards sharpening-like changes, credited us
with about 20 points of BD-rate against SVT-AV1. Those points vanish under
VMAF NEG ("no enhancement gain"), and under NEG we were about level with SVT.
We moved every decision to the mean of PSNR-YUV, VMAF NEG and MS-SSIM, and
re-screened every default that had been adopted on VMAF v1. One reversal
stands out. AV1's deblocking filter had been rejected because VMAF v1 scored
it +103% worse on one clip (bodycam_airport). Under the new metrics it gave
−6.6% on VAL with 0 of 11 clips worse, and it is now on in the Default.

**Real-time rate control changes the ranking.** Fixed-quantiser comparisons
measure the coding tools and nothing else. The board we trust now runs every
encoder, ours and the references, in its own real-time CBR mode on the same
ladder (150–1200 kbps at 720p30, 1 s buffer), and replays every point through
a strict 1 s leaky bucket. Several references break the bucket at low rates.
openh264 has 22 rate-invalid points; on static clips its first key frame
alone takes 1.4–2.6 s of the target rate. libvpx has 15–16 and libaom 6–8.
Ours have none on the chart under that rule, which judges our points without
the first key frame's outstanding loan (our controller may borrow up to about
1.4 buffers for it and repay it later), so it is not a strict 1 s VBV. Real-time modes are tuned for latency, so a
reference that looks strong at fixed QP can be weaker when it has to hold a
buffer.

**The first rate controller cost a lot.** Against our own fixed-QP curves,
the first CBR controller lost +32.2% (AV1) and +10.8% (H.264) PSNR BD-rate.
Most of the loss was key-frame sizing. A fixed key budget left the key frame
coarser than the P frames after it on cheap static content, and every later
frame inherited the poor key (AV1 lowlight +112%). A key-level offset and a
per-layer QP cadence ("keycad") recovered −5.7% (H.264) and −5.3% (AV1), most
of it on lowlight (−13% / −19%). A later policy made the target a ceiling,
not a rate to hit: bits that no longer buy quality are not spent.

**Short screens miss slow drift.** A skip lever that zeroed small chroma
residuals passed its 150-frame screen (−1.6% J_real). On 300-frame
single-key runs it decayed by 0.34 → 0.52 → 0.70 dB per third on a pan clip,
because the skipped chroma error builds up over a long chain of P frames.
Every inter, loop-filter and rate-control change now also gets a
single-key-frame full-length run with a per-third trend.

**"Off" was not free.** Each experiment lands behind a registered parameter,
and its off setting must reproduce the old streams byte for byte. Identical
streams turned out not to mean identical cost. AV2 parity hiding, landed
*off*, still cost +1.2% x86 and +0.9% Jetson encode time, through one
per-coefficient tally in the shared coefficient walk. The rule is now "off
means free": identical streams, the same instruction counts within layout
noise, and the same Jetson cycles. The fix is a pattern: test the flag once
per block and dispatch to a specialised walk, rather than test it per
coefficient.

## 4. What we tried, by area

*J_real* is our adoption score: the class-balanced mean of full-range and
low-rate BD-rate on (PSNR-YUV + VMAF NEG + MS-SSIM) / 3, plus a CPU term
weighted by a factor k (0.1 for the Default preset, which buys quality, and
0.6 for Fast, which buys speed). Guards reject a change outright if any
content class loses more than 3%, the worst 1% of frames lose more than 2%,
or flicker or the key-frame pulse grows. Negative is better.

### Motion search and partitioning

The engine follows SVT-AV1: zero-vector SAD gates, quarter-resolution
hierarchical search, a small full-pel window, then a statistics-driven
partition plan per superblock. That structure is the main reason our CPU is
competitive. Most attempts to do more search inside it failed:

| Tried | Effect | Verdict |
|---|---|---|
| HEVC vertical HME range ±32 | one clip −2.5%, another +2.1% | rejected |
| AV1 zero-vector gates 2/3/4 | +1% bits for −1.1% CPU | rejected |
| HEVC "skip-first" CU decision | +6.4 … +31.6% | rejected |
| HEVC 8×8 inter CUs | +8 … +10% | rejected |
| VVC multi-type tree, RD-confirmed shapes (`rect_rd`) | −0.97% fixed-QP BD for +32% (crowd) / +70% (pan) instructions | off |
| Still detector (8-frame run + 3×3 guard) | precision 0.45 → 0.82, recall 1.00 → 0.80 | adopted as the gate for static levers |
| L1T2 layering on static scenes | static clips lose 0.4–8% quality; HEVC −9% CPU | a CPU lever only, not in presets |

VVC's multi-type tree (binary and ternary splits beside the quadtree) shows
the general lesson. The shapes save bits at matched QP, but only when
priced exactly at sub-pixel-refined vectors, and that pricing costs more
than the gain is worth in real time.

### Mode decision

| Tried | Effect | Verdict |
|---|---|---|
| H.264 psychovisual term 0.25 | quality −1.1%, +2.7% instructions | adopted |
| HEVC psy 0.25 | +1.3% (crowd guard) | rejected |
| AV1 P-frame intra modes + CfL | +0.05% | rejected |
| AV1 exact 8-tap sub-pel refinement | +30% CPU for no gain | rejected |
| H.264 half-pel only | +24.8% | rejected |
| AV1 Default preset arms (sub-pel margins, ungated full-pel probe, inter transform split) | −6.5% fixed-QP BD for +29.7% P instructions | adopted (Default/Fast split) |

### Quantisation: trellis, TCQ, dependent quantisation

Plain quantisation rounds each coefficient on its own. A *trellis* (RDOQ)
searches over the levels of a whole block with a dynamic program, pricing
each choice with the real entropy-coder contexts, and keeps the cheapest
sequence in distortion + λ·bits. One shared trellis in the engine, with a
per-codec rate model, is now adopted in five Default presets:

| Codec | Quality (TRAIN or VAL) | CPU |
|---|---|---|
| AV1 | VAL −2.4% full / −2.6% low | P frames +15% host, +20% Jetson |
| HEVC | TRAIN −2.4% full / −2.0% low | marginal J, adopted for Default |
| H.264 | TRAIN −2.0%, every class better | +18.1% weighted; J −0.2 |
| VP9, AV2 | adopted, P-frame luma, one survivor | Fast presets pin it off |

The newer standards' state-machine quantisers did not make it. AV2's
trellis-coded quantisation (TCQ, eight states whose parity steers the next
level's reconstruction) gave −2.6 … −2.9% quality. It needs an exact Viterbi
search inside every mode-decision trial, so it cost +72% CPU, and J_real came
out +5.1% worse; break-even is about +26%. The cheaper variants lost most of
the gain, which taught us that mode decision must see the levels TCQ will
code. VVC's dependent quantisation, with its own four-state Viterbi, cost
+107% for mixed gains (+0.05 dB on a static clip, −0.24 dB on crowd).

### Rate control, AQ and psy

- **keycad** (key-level offset + per-layer cadence): −5.7% H.264, −5.3% AV1.
  Adopted.
- **Variance AQ**, a per-block quantiser from luma variance: PSNR +7.0%,
  VMAF +13.0%. Rejected twice. On camera footage, "flat" blocks are often
  noise, and noise does not mask quantisation error.
- **Content-adaptive periodic key boost:** rejected.
- **Re-coding a frame that would overflow the buffer:** adopted with a
  ceiling of 1.4× the buffer (under the 1.5× validity limit). A ceiling of
  1.0, meaning no overrun at all, cost static_cam22 +3.5% J_real.

### Loop filters

| Tried | Effect | Verdict |
|---|---|---|
| AV1 deblocking in Default | VAL quality −6.6%; +7.5% instructions after SIMD; J −3.6% at k 0.2 | adopted (Fast off) |
| HEVC luma SAO off | J +4.4% | SAO stays |
| HEVC chroma SAO off | J −0.2% for −1.6 … −3.2% CPU | fails bodycam bar in Fast |
| AV1 joint luma/chroma CDEF | VAL PSNR +11.5%, homes390 +108% | not promoted |
| AV1 CDEF strength reuse every 4 frames (Fast) | −2.2% instructions; long GOP +1.5%, bodycam +3.2% | drifts; not landed |

### Entropy coding

*CABAC* (H.264/HEVC) and AV1's multi-symbol coder are adaptive arithmetic
coders: their probability models learn the statistics of the stream as it is
coded. Two results stood out:

- **CAVLC would not have saved H.264.** The older variable-length coder is
  what openh264 uses by default. Measured inside openh264 and x264, it costs
  +12.5 … +14% bits, more at the low rates we target, and would save only
  about 6% CPU. That is a J of about +9, against a margin of −21% over
  openh264. Not built.
- **Cross-frame CDFs in AV2.** Our AV2 had coded every frame from the default
  probability tables, while our AV1 inherits the previous frame's adapted
  ones. Inheriting them saved 0.8–4.9% of bytes at equal PSNR for no CPU.
  On VAL it gave J_real −2.6% with no long-GOP drift, and it moved AV2's lead
  over our AV1 from −1.25% to −4.3%. Ironically, the reason AV2 first trailed
  AV1 was a missing default, not a missing tool.

### Speed kernels

Speed lives in small SIMD kernels (AVX2, NEON, WASM SIMD128). Each has a
scalar reference and a bit-exactness test. The big single wins were porting
whole stages:

| Kernel | Effect |
|---|---|
| VP9 inverse transform, AVX2 | VP9 encode −11.2% |
| VP9 loop filter edges, AVX2 | VP9 −7.3% |
| AV1 deblocking, packed lanes | deblocking overhead +17.1% → +4.9% (crowd) |
| AV2 typed transforms | AV2 −4.1% |
| Const-width variance / residual kernels, Q15 bilinear taps (shared) | −3.6% (AV1), −4.3% (HEVC), −6.5% (H.264) instructions |

What didn't help is just as telling. Transparent huge pages cut L1 TLB
refills by 85% and CPU time by nothing (the L2 TLB caught them).
`target-cpu=cortex-a78` was noise, because the kernels already dispatch at
run time.

## 5. How we do well against the others

On the real-use board of that time (11 natural VAL clips, which were used to
confirm changes, CBR ladder, every stream through both standard decoders),
head to head on the same points, with the references at their real-time
settings on one thread:

| Ours | vs | BD (J metric set) | Jetson fps, ours / theirs |
|---|---|---:|---|
| H.264 | openh264 | −23.0% | 96 / 111 |
| H.264 | x264 `veryfast` | −19.3% | 96 / 85 |
| HEVC | x265 `medium` | −8.6% | 49 / 8 |
| HEVC Fast | x265 `superfast` | −2.4% | 73 / 12 |
| AV1 | SVT-AV1 M12 `--rtc` | −19.1% | 40 / 71 |
| AV1 Fast | SVT-AV1 M12 `--rtc` | −7.7% | 61 / 71 |
| AV1 | SVT-AV1 M8 | +1.2% | 40 / 14 |
| VP9 | libvpx rt (cpu-used 7) | −39.8% | 33 / 30 |

These are the design choices behind the margin, as far as we can trace them:

- **Rate-control discipline in the 1 s buffer.** After the first key frame's
  loan is repaid, our encoders hold the bucket at every rung (the loan itself
  is up to about 1.4 buffers). The worst 1% of frames are where this shows: against SVT-AV1
  our worst frames are 33–39% better, and our key frames pulse 1.8–2.8 dB
  less. A key frame that blows the buffer, as openh264's does on static
  clips, buys a sharp first frame with seconds of starved ones after it.
- **The statistics-driven planner.** Partitioning from motion-search
  statistics instead of trial encodes is what let our HEVC run 6× faster than
  x265 `medium` (zero latency, no B-frames, one thread) on the Jetson and
  still code 8.6% fewer bits at the time.
- **One engine, six codecs.** The trellis, the still detector, keycad and
  the planner were each written once and adopted per codec after their own
  screen. Our VP9 needed 39.8% fewer bits than libvpx's real-time mode
  (cpu-used 7, one thread) mostly because it inherits an engine built for
  HEVC and AV1.
- **Joint tuning into Pareto presets.** Every heuristic threshold is a
  registered parameter. Presets come from a multi-objective search (quality
  vs CPU) over those parameters, not from hand-picked flags. Default and Fast
  are two points on that front, with Default meant to cost at least 1.5×
  Fast's CPU. Hand bundles did
  poorly: one AV1 Fast bundle saved 14.5% CPU for +23% BD-rate, which is why
  Fast is found by search.
- **Cost-aware adoption.** Each lever is judged on quality *and* CPU on both
  platforms (Jetson weighted 2:1). That rejected many changes a pure-quality
  view would have kept (MTT, TCQ, dependent quantisation), and kept the
  encoders inside real time.

Gaps remain. On fine pristine texture (water, foliage, grass), x264 and x265
need fewer bits than ours. Our H.264 flickers more than openh264 and x264 (+0.2 dB
mean frame-to-frame PSNR change). Our VP9 still trails our HEVC on crowd and
pan. Our AV1 Default is slower than SVT-AV1 M12 (40 vs 71 fps on the Jetson).
It needs fewer bits than M12 by spending that time on search, and its Fast
preset (61 fps) is the closer comparison.

## 6. Modern design vs the coding standard

How much of the gain is the engine, and how much is the newer standard? We
can separate them because we hold one factor fixed at a time.

**Engine margin at a fixed standard.** Same standard, different encoders:

| Standard | Comparison | BD | Speed |
|---|---|---:|---|
| H.264 | ours vs x264 `medium` | −13.6% | ours 3.6× faster |
| H.264 | ours vs x264 `veryfast` (similar speed) | −19.3% | 96 vs 85 fps |
| H.264 | ours vs openh264 | −23.0% | 0.87× its speed |
| HEVC | ours vs x265 `medium` | −8.6% | ours 6× faster |
| AV1 | ours vs SVT-AV1 M12 `--rtc` | −19.1% | ours 0.56× its speed |
| AV1 | ours vs SVT-AV1 M10 (same speed, chained) | about −9% | 40 vs 39 fps |

At matched speed the engine is worth about 9–20%, depending on the
competitor.

**Standard steps on our own engine.** Same engine, different standards, from
the chart (each against x264 `medium`, chained; a direct pair where we have
one):

| Step | Chained | Direct |
|---|---:|---:|
| H.264 → HEVC | −37.1% | — |
| HEVC → AV1 | −7.5% | — |
| AV1 → AV2 | −2.7% | −4.1% (J_real −7.9%) |
| HEVC → VVC | −5.5% | −6.7% (J_real −4.6%) |
| HEVC → VP9 | −2.5% | J_real +1.6% (level) |

The H.264 → HEVC step is the largest, and it matches the reference encoders: x265
`medium` is −41.3% against x264 `medium`. The newer standards add single
digits each on our engine so far. Part of that is our subset: none of our
backends uses more than one active reference, compound prediction or loop
restoration yet, and those are where the newer standards add tools.

**The decomposition.** Our AV1 is −49.8% against x264 `medium`. Two ways to
split it agree:

- *Engine first:* our H.264 engine gives −13.6% at the same standard. Moving
  that engine from H.264 to AV1 gives a further −41.9% (0.502 / 0.864). In
  log terms, **about 21% of the gain is the engine and about 79% is the
  standard.**
- *Standard first:* SVT-AV1 M12 `--rtc`, a strong real-time AV1 encoder,
  gets −38.9%. Our engine adds −19.1% on top, head to head. That chains to
  −50.6% (the direct figure is −49.8%), with **about 30% engine and 70%
  standard.**

So, roughly: **of our AV1's −49.8%, about −14 to −19% comes from the engine
and about −39 to −42% from AV1's tools.** That is a quarter to a third
engine, the rest standard.

Assumptions and caveats:

- The two factors are treated as independent and multiplicative. That is
  only approximately true: a better engine exploits more of a standard's
  tools.
- x264 `medium` and SVT-AV1 M12 stand in for "a competent encoder" of each
  standard. Against openh264 the engine share would look larger, and against
  a slow reference encoder it would look smaller. SVT-AV1 M8 reaches what our
  AV1 reaches (+1.2%), at a third of our speed. The engine's contribution is
  getting a standard's gain *at real-time speed*, not exceeding what the
  standard allows.
- Chaining BD-rates through an anchor is approximate. Our H.264 vs openh264
  is −23.0% directly but −29.1% when chained through x264, because openh264's
  rate-invalid points shrink the overlap. Read the chained numbers to within
  a few points.
- These come from 11 short clips (150 frames), where differences of a few
  percent are within the per-clip spread.

## 7. What we would tell the next person

- Measure the thing you ship. Fixed-QP and VMAF v1 both pointed us the wrong
  way, more than once.
- Real time is a budget. Most "better" tools (MTT, TCQ, dependent
  quantisation, exact sub-pel filters) are better at fixed QP and worse at a
  fixed CPU budget.
- Write the improvement once in the shared engine, then screen it per codec:
  it won't win everywhere (psy did in H.264, not in HEVC).
- Long-GOP runs and the off-means-free check are cheap, and they catch what
  short screens miss.
