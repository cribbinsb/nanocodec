# Ideas from the literature, ranked for us (2026-11-21)

**Question.** Which published techniques (papers, standards contributions, encoder
documentation and source of the BSD encoders) could still move our frontier, given
what the ideas ledger says we already tried?

**Method.**
- Read first: `docs/ARCHITECTURE.md`, `docs/handoff/presets.md`,
  `docs/handoff/rd-mode-decision.md`, `docs/handoff/fast-paths.md`,
  `docs/analysis/tools-that-hurt-2026-11-20.md`,
  `docs/analysis/silent-noop-audit-2026-11-20.md`.
- Web survey of the seven areas in the brief (about 140 searches and page reads, by three
  research sub-agents and directly). No encoding runs.
- Every candidate was searched in the ledger (`nceval.levers ideas search`, about 200
  phrasings); ledger slugs are quoted with their outcome.
- GPL encoders (x264, x265, FFmpeg) were not read; nothing here comes from their code.

**Reading the entries.**
- **Status** of a source: *verified* = the page, abstract or source file was opened and
  states what is quoted (most pages went through a summarising fetch tool, not a raw
  read); *partial* = title and authors confirmed from a bibliographic record or search
  result, numbers from a snippet or not seen. Nothing unverified is cited.
- **Reported** numbers are the source's own, on its own baseline (usually an RD-search
  reference encoder, often random access). None was reproduced here. They overstate
  what a pipeline that already avoids the RD search can save.
- **Fit** and the expected sizes under it are my estimates, not measurements.
- **Effort**: S = days, one module; M = a week, a hook per codec; L = new structure.

## Summary: the top 15

| # | Idea | Stage, codecs | Ledger | Fit | Effort |
|---|---|---|---|---|---|
| 1 | Classifier-gated RD partition search (uncertainty zones) | planner, shared; VVC first, then HEVC/AV1 | RD search adopted in VVC Slow, rejected at Default on CPU; gating by a classifier not tried | high | M |
| 2 | Causal temporal filter of the source (last frame, static blocks) | analysis, shared | **rejected 2026-11-21** (diagnostic: worse on real pool clips) | medium-high | S |
| 3 | Cheap RDOQ substitute on the fast rungs (zero-block bypass, high-frequency level-1 rule) | quantiser, shared | parts tried (eob cap rejected, HEVC shave adopted) | medium | S-M |
| 4 | Sub-pel points chosen from the integer cost surface | mode decision, shared | **rejected 2026-11-21** as a probe saver (no Jetson cycle gain); **landed** as a predicted quarter step on VVC Fast, VP9 Fast/Faster, HEVC Faster | medium | S |
| 5 | Temporal-dependency λ in the other codecs and presets | rate/QP, shared | adopted for AV1 Faster only | medium | S-M |
| 6 | AV1 local warped motion | AV1 (AV2) backend | not tried in the encoder | medium | M |
| 7 | Transform-type shortlist by SATD and neighbours | AV2, AV1, VVC backends | type search rejected on CPU | medium | S-M |
| 8 | Integral-projection seed for large camera motion | analysis, shared | not tried | low-medium | S-M |
| 9 | LTR feedback audit against published pitfalls; layer-weighted protection | resilience, shared + transport | LTR adopted | medium (correctness) | S |
| 10 | Modelled background as a long-term reference | references, shared | second reference held/partly adopted; background model not tried | medium-low | L |
| 11 | VMAF-coupled block λ | mode decision / AQ, shared | variance AQ rejected three times | low-medium | M |
| 12 | XPSNR-style spatio-temporal activity QP | AQ, shared | variance AQ rejected; temporal term not tried | low-medium | S |
| 13 | Cyclic refresh with a coded-Q map, static blocks only | rate/AQ, shared | a rotating-band variant rejected | low-medium | M |
| 14 | AV1 global motion from the camera model we already fit | AV1 (AV2) backend | not tried in the encoder | low-medium | M |
| 15 | Statistics-only SAO class and ALF decisions | HEVC, VVC filters | SAO search interval and merges done | low-medium | S-M |

The first seven are held in the ledger (see "Ledger records" at the end).

## 1. Classifier-gated RD partition search

- **Sources.**
  - T. Amestoy, A. Mercat, W. Hamidouche, C. Bergeron, D. Menard, "Random Forest Oriented
    Fast QTBT Frame Partitioning", ICASSP 2019, pp. 1837-1841
    (<https://researchportal.tuni.fi/en/publications/random-forest-oriented-fast-qtbt-frame-partitioning/>),
    verified (abstract).
  - A. Mercat, A. Lemmetti, M. Viitanen, J. Vanne, "Acceleration of Kvazaar HEVC Intra
    Encoder With Machine Learning", ICIP 2019, pp. 2676-2680
    (<https://researchportal.tuni.fi/en/publications/acceleration-of-kvazaar-hevc-intra-encoder-with-machine-learning/>),
    verified (abstract).
  - libaom `av1/encoder/speed_features.h` (`ml_prune_partition`,
    `simple_motion_search_split`, `ml_early_term_after_part_split_level`;
    <https://aomedia.googlesource.com/aom/+/refs/heads/main/av1/encoder/speed_features.h>),
    verified (header comments; no numbers).
  - SVT-AV1 `Docs/Appendix-Mode-Decision.md`, depth refinement
    (<https://gitlab.com/AOMediaCodec/SVT-AV1/-/raw/master/Docs/Appendix-Mode-Decision.md>),
    verified (no numbers).
- **What.** A small classifier (decision tree or random forest on features available
  before coding) predicts the split decision. Amestoy et al. add "uncertainty zones":
  the classifier decides alone where it is confident, and only the uncertain nodes go
  to the full RD comparison. Kvazaar's trees predict a depth interval for the search.
- **Reported.** Amestoy: about 30 % encode time for +0.57 % BD-rate (JEM-7.0, random
  access). Mercat: 1.35x speed of Kvazaar veryslow at +0.49 % bitrate (intra only).
- **Us.** `planner.rd_split` already is the RD comparison, and its value is measured:
  VVC Slow J -9.1 % at +99 % P instructions (`rd-partition-search-planner-rd-split-in-vvc-slow`,
  adopted); VVC Default -5.7 % at +78 %, rejected at k 0.1 with the note "needs ~half
  the CPU" (`rd-partition-search-in-vvc-default`); HEVC and AV1 share the
  transform-blind 16->8 loss (`rd-partition-search-for-hevc-av1-and-the-other-codecs`,
  held; `hevc-shadow-rd-ceiling-full-coder-all-planes`: ceiling 7.4-8.7 % of coded cost,
  mostly 16->8 splits). The prunes tried so far are structural (depth window, skip
  exit, cost bound). Not tried: deciding *which nodes* get the RD comparison from the
  planner's own evidence. The search itself produces the labels (plan against RD
  winner per node), so a shallow tree on planner features (variance-cost margin
  between leaf and split, `me_dist`, vector spread of the quarters, quantiser, noise
  sigma, block size) can be fitted offline and checked on held-out clips.
- **Ledger caution.** `learned-superblock-effort-ranking` (rejected): a gradient-boosted
  model did not rank superblocks for *effort* better than the proxy. That predicted
  where extra effort gains; this predicts the split decision at a node and keeps the
  RD comparison where the prediction is uncertain. A different target, but the same
  risk: check offline that the tree separates nodes before any encoder work.
- **Fit: high.** The ceiling is our own measurement and the labels cost nothing. The
  open number is what fraction of nodes must still be priced; halving the +78 % would
  put VVC Default near break-even at k 0.1, and HEVC/AV1 8x8 are next.
- **Effort: M.** Shared engine (`rd_split.rs` gate + planner features); trained on TRAIN,
  judged on the rotating pool, so it cannot be fitted to the clips that screen it.

## 2. Causal temporal filter of the source

- **Sources.**
  - libaom commit "Improve RTC temporal filtering", Yunqing Wang, 2022-08-30
    (<https://aomedia.googlesource.com/aom/+/009c7c4ad99865f2d70fd059b4ae57fe06d87321%5E%21/>),
    verified.
  - libvpx `vp9/encoder/vp9_denoiser.c`
    (<https://chromium.googlesource.com/webm/libvpx/+/refs/heads/main/vp9/encoder/vp9_denoiser.c>),
    verified (no numbers).
  - Past-only configuration of the VTM/HM temporal pre-filter (JVET-O0549 lineage; HM
    merge request <https://vcgit.hhi.fraunhofer.de/jvet/HM/-/merge_requests/54>),
    partial (no low-delay number seen).
- **What.** Per 64x64 superblock the current source is filtered against the co-located
  block of the *previous source frame*: no motion search, no lookahead. libaom's gates:
  variance of the difference at most 250 x the AC quantiser step, small mean
  difference, no high frame SAD, neighbours' vectors at most 24, and the zero-offset
  SAD clearly below the SAD at the four one-sample offsets.
- **Reported.** libaom, rtc_derf set, speeds 9 / 10: -0.405 / -0.344 (average PSNR) and
  -0.547 / -0.494 (SSIM); the commit message does not label the unit (presumably
  BD-rate %). Encoder slowdown at most 0.5 %.
- **Us.** Shared analysis stage, every codec. The gate inputs exist (8x8 and 4x4 SADs
  against the previous source, static flag, noise sigma). It serves the stated
  principle "noise is removed, not reproduced", and it also steadies the static
  detector.
- **Ledger.** No temporal-filter idea. `denoise-before-encoding-with-grain-signalling`
  (AV1, rejected, "not built": sizes moved <= 0.34 % at fixed QP) was an estimate for
  denoise plus grain synthesis, not this filter.
- **Fit: medium-high.** Cheap and shared. The open risk is our score: J is measured
  against the unfiltered source, as libaom's numbers are, so the gain should carry,
  but low-light clips (high sigma) are where it could cost PSNR.
- **Effort: S.**

**Outcome 2026-11-21: rejected, not built** (ledger `causal-temporal-source-filter`;
scripts in `~/nanocodec-data/work/lit-ideas/tf/`). Offline diagnostic first: a recursive
causal filter in Python (each 8x8 block motion-compensated at its best full-pel offset
within ±2 against the previous *filtered* frame; per-pixel blend weight 0.5 x (1 -
|d| / 3σ); a block is left alone when its mean |d| exceeds 2 x 1.128σ; σ from the
flattest tenth of blocks), the filtered clip coded with our encoder and scored against
the unfiltered source (PSNR-YUV 6:1:1 from `--sse-log`; fixed QP, 60 frames).

- **Real pool clips lose.** Noise σ over the pool (this estimator, 12 frames): 179 of
  221 clips below 0.8, 17 at 1.5-3 (mostly moving water and texture), 1 above 3. On
  the four noisiest real clips, H.264 Fast, QP 34-46, BD-rate PSNR-YUV of filtered
  against unfiltered: pool_otw_homes_00168 (σ 1.25) +0.65 %, pan_bdd_00cef86b +0.81 %,
  vc1k android lowlight medium motion (σ 1.5) +1.50 %, pool_bodycam_oxford_street
  (σ 2.3) +4.20 %. A gentler setting (blend 0.25, gate 1.0σ, co-located only): +0.13 % /
  +0.93 % (H.264 Fast), +0.24 % / +2.92 % (AV1 Faster) on the lowlight and bodycam
  clips. VMAF v1 also falls (vc1k lowlight, QP 38 / 42: 66.04 -> 65.76, 53.57 -> 53.16
  at equal rate). At CBR 300-1000 kbps the PSNR-YUV moved by -0.05..+0.15 dB with the
  rate differences rate control left.
- **Synthetic noise wins, against the clean picture.** Scored against the clean
  reference (the main set's `*_noise_*` clips, which the pool excludes): pan_jaad_0147
  high (σ 5.5) -22.9 %, static_london_bus low (σ 3.3) -7.1 %, crowd_pp22_027 low (σ 2.8)
  -1.2 %, pan_jaad_0147 low -0.3 %; crowd_pp22_027 high (σ 6.4) and static_london_bus
  high are better at every QP (crowd high: -12..-16 % bits at +0.08..+0.68 dB; their
  BD fit is not meaningful, the curves saturate). Scored against the noisy source,
  static_chirla_s000c3 noise_high saves 29-47 % of the bits for 0.02-0.2 dB at QP 38-48.
- **Why not built.** Adoption evidence comes only from the pool, which holds no clip
  where the filter pays; gated to the σ where it does (about 3 and up) it would be inert
  on the whole pool. Revisit only for genuinely noisy camera input, judged on clean
  references.

## 3. Cheap RDOQ substitute on the fast rungs

- **Sources.**
  - M. Xu, T. Nguyen Canh, B. Jeon, "Simplified Rate-Distortion Optimized Quantization for
    HEVC", IEEE BMSB 2018
    (<https://pure.skku.edu/en/publications/simplified-rate-distortion-optimized-quantization-for-hevc/>),
    verified (abstract).
  - "Fast rate distortion optimized quantization for HEVC", IEEE VCIP 2015
    (<https://unpaywall.org/10.1109%2FVCIP.2015.7457854>), partial (authors not seen).
  - J. Cui, R. Xiong et al., "Hybrid All Zero Soft Quantized Block Detection for HEVC",
    IEEE TIP 27(10), 2018 (<https://dblp1.uni-trier.de/rec/journals/tip/CuiXZWWMG18.html>),
    partial.
- **What.** Two pieces. (a) Skip the level optimisation where it cannot matter:
  all-zero blocks and blocks where only DC survives. (b) Replace the search by a rule:
  level-1 coefficients at set high-frequency positions go to zero, with a smaller
  rounding offset at high frequencies. Cui et al. add "pseudo" zero blocks: blocks the
  plain quantiser keeps but RDOQ would zero, found by an RD estimate before the
  transform.
- **Reported.** Xu: 15.6 % quantisation time for 0.01 % BD-rate (all intra), 13.17 % for
  0.07 % (random access). VCIP 2015: 34 % of RDOQ time, "slight" loss. These are
  against RDOQ on; the gain of the rule against RDOQ *off* is not reported.
- **Us.** Trellis is worth -2.2 % quality on AV1 Default at +15-20 % CPU
  (`av1-default-trellis-quantisation`) and was too dear for H.264 Fast (-2.1 quality
  for +26.6 %, `h-264-fast-default-s-heavy-levers-final-stage-winner-walk-trellis-probes`). The fast rungs therefore run a dead-zone quantiser. A positional
  level-1 rule is a registered-parameter version of what HEVC's "shave" already does
  (`hevc-shave-gap-8-inter-rounding-128-perceptual-dead-zone`, adopted;
  `shared-trellissettings-planned-dead-zone-and-isolated-ones-shave`, partial), shared
  across codecs and tuned per rung.
- **Ledger.** `shared-trellis-max-eob-threshold-quant-trellis-max-eob` rejected (cap 16:
  -5.4 % instructions; cap 32: J +1.19 %): cutting the trellis by block size loses.
  `provable-all-zero-test-before-the-forward-transform` rejected (never fired);
  `pre_tx_skip` adopted. The pseudo-zero test (what the trellis *would* zero) is not
  in the ledger.
- **Fit: medium.** Zero CPU on rungs that have no trellis; the size of the gain is
  unknown (0 to perhaps a third of the trellis gain, my guess).
- **Effort: S-M** (one shared rule in the quantiser wrapper, per-codec scan positions).

## 4. Sub-pel points chosen from the integer cost surface

- **Sources.**
  - P. R. Hill, T.-K. Chiew, D. R. Bull, C. N. Canagarajah, "Interpolation free subpixel
    accuracy motion estimation", IEEE TCSVT 16(12):1519-1526, 2006
    (<https://research-information.bris.ac.uk/en/publications/interpolation-free-subpixel-accuracy-motion-estimation/>),
    verified (abstract; no numbers in it).
  - libaom `speed_features.h`: `SUBPEL_TREE_PRUNED`, `SUBPEL_TREE_PRUNED_MORE`,
    `use_adaptive_subpel_search` (URL as in 1), verified (comments; no numbers).
- **What.** The SADs at the best whole-sample vector and its four neighbours define a
  cost surface. Either solve a parabola for the sub-sample minimum (Hill), or use the
  surface only to choose which half-sample positions to test (libaom). libaom also
  prunes harder when the whole-sample match was good and the zero-vector SAD or source
  variance is low.
- **Reported.** Hill: "reduced complexity ... with a controllable tradeoff", used for a
  real-time H.264 encoder; no figure in the abstract.
- **Us.** The engine refines the winner with a fixed cross, half then quarter (4 + 4
  positions). The motion search's full-pel window already has the neighbours' SADs.
  Testing two of the four positions per stage, on the side the surface slopes to,
  would remove up to half the refinement predictions on every codec.
- **Ledger.** Nothing under parabola, error surface or cost list. Related and adopted:
  `backend-sub-pel-skip-variance`; `adaptive-threshold-controller-sub-pel-rank-probe-a3`
  (partial). `engine-skip-quarter-sample-stage-when-half-keeps-centre` rejected (quarter-pel is
  essential), so
  this prunes directions, never a stage.
- **Fit: medium.** Sub-pel was 8 % of AV1 P-frame work in the 2026-11-09 profile; a
  1-3 % CPU saving on the fast rungs is plausible (estimate). The source-frame SADs
  describe the source, while refinement runs on the reconstruction: that mismatch is
  the risk.
- **Effort: S.**

**Outcome 2026-11-21: rejected, not landed** (ledger
`sub-pel-points-from-integer-cost-surface`; code on branch `lit-ideas2`, data in
`~/nanocodec-data/work/lit-ideas/`). Built as a shared engine lever,
`decision.subpel_surface` (`nc-encoder/src/subpel_surface.rs`): along an axis whose two
far costs are known (the quarter stage after an unmoved half stage: the half probes;
the half stage, and the quarter stage's move axis: the integer probe's whole-sample
neighbours), 1 probes only the side of the lower far cost, 2 fits a parabola.

- **Diagnostic** (branch `lit-ideas-diag`: the full and the pruned search run on the
  same cached costs at every refinement; 60 frames, CBR 600, the four quick clips,
  14 fast rungs). The side rule ends at the full search's vector in 97.9-99.9 % of
  refinements (97.6 % on VVC's few secondary-reference refinements) on AV1 / AV2 / VVC / HEVC Fast / VP9 Fast (refinement cost +0.007 % at
  most) and prunes 20-24 % of the probes on AV1, 34 % on AV2, 35 % on VVC Fast, 20 % on
  HEVC Fast, 10.5 % on VP9 Fast. The parabola (reach 1/8) prunes 26-41 % at 95.8-99.5 %
  agreement, cost up to +0.03 % (VVC +0.12 %). On H.264 Fast/Faster/Veryfast, HEVC
  Faster and VP9 Faster/Veryfast nothing is known for free (the quarter stage runs only
  after a half move, `decision.subpel_quarter_needs_half_move`, and there is no integer
  probe): 0 % pruned. Measuring the neighbours on purpose (the "oracle" variant) prunes
  28-50 % at similar agreement but costs four whole-sample measurements to save about
  two bilinear ones.
- **CPU, side rule on (same binary, `--param`):** x86 release instructions (31 f, CBR
  600, five quick clips): HEVC Fast -1.09 %, AV2 Faster -0.91 %, AV1 Veryfast -0.87 %,
  AV1 Faster -0.83 %, AV2 Fast -0.67 %, VVC Fast -0.42 %, VP9 Fast -0.35 %, AV1 Fast
  -0.26 %. Jetson (jperf, 90 f, four clips, min of 3): instructions -0.3..-1.6 %, but
  cycles mean per rung HEVC Fast +0.39 %, AV1 Faster +0.13 %, AV1 Veryfast +0.08 %,
  AV2 Faster +0.05 %, VVC Fast +0.03 %, AV1 Fast -0.03 %, AV2 Fast -0.02 %, VP9 Fast
  -0.27 %: the skipped bilinear evaluations are not on the board's critical path.
  dCPU = (2 x Jetson + x86) / 3 is -0.1..-0.3 %, about 0.1-0.2 % of J at k 0.6 before
  any quality cost.
- **Off path not free.** With the lever off, x86 instructions against the parent
  (release builds of both, same toolchain, five quick clips): first build AV1 Faster
  +0.42 %, H.264 Faster +0.29 %, HEVC Fast +0.29 %, VP9 Faster +0.07 %; with separate
  off and on instances of the search, H.264 Faster +0.39 %, HEVC Fast +0.12 %, VP9 Faster
  +0.12 %. Per function (H.264 Faster, crowd): the bilinear refinement's batched cost
  (`cost_four_below`) is no longer inlined; forcing it inline made the total worse
  (+0.63 %). Keeping the default free would need a second copy of the
  refinement loops.
- **Why rejected.** No Jetson cycle saving on any rung, a cost at defaults, and code
  that would need duplicating to be free: not screened for quality.

**Follow-up 2026-11-21: the same fit as a quality lever, landed on four rungs**
(ledger `finer-subpel-step-from-current-costs`; code 4c0e7b9e9, presets 66d8932a7 and the
VP9 Faster / HEVC Faster commit after it; data in `~/nanocodec-data/work/lit-ideas/`).
`decision.subpel_quarter_predict` (`nc-encoder/src/quarter_predict.rs`): where the
search skips its quarter stage after a half stage that did not move
(`decision.subpel_quarter_needs_half_move`, `subpel_half_only`), the parabola through the
half stage's costs predicts the best quarter neighbour; 1 probes it once and keeps it only
if cheaper (never worse than the half-sample vector), 2 takes it unmeasured where the
predicted gain is at least 5 % of the distortion. Off: 306/306 identical, x86 instructions
-0.01..+0.02 % against its parent.

- **Upper bound** (full quarter stage where gated; fixed QP, quick draw, 24 clips, J
  quality only): VP9 Veryfast -3.04 % [-4.27, -1.74], VP9 Faster -2.80 %, H.264 Faster
  -1.02 %, H.264 Veryfast -0.98 %, VP9 Fast (light frames) -0.98 %, HEVC Faster -0.79 %,
  VVC Fast -0.78 %, H.264 Fast -0.51 %. The AV1/AV2 walk always runs its quarter stage.
- **Agreement** (diagnostic, 60 f, CBR 600, four quick clips): one probe ends at the full
  quarter search's vector in 98.1-99.9 % of the skipped refinements (VVC 84.7 %), keeping
  91-99.5 % of its decision-time gain.
- **Recovered, fixed QP** (probe / no-probe model): VP9 Faster -2.70 / -3.30, VP9 Veryfast
  -2.66 / -3.15, H.264 Faster -1.03 / +2.14, H.264 Veryfast -1.00 / +3.19, VP9 Fast -0.95 /
  -0.40, HEVC Faster -0.76 / -1.09, VVC Fast -0.66 / -0.53, H.264 Fast -0.48 / -0.14.
- **CPU, probe** (whole-encode cycles, `aws_cpu_ab`, Graviton3 / Graviton4 / Zen 5): VP9
  Faster +2.10 / +1.87 / +1.75 %, VP9 Veryfast +3.38 / +3.25 / +2.41 %, H.264 Faster
  +0.82 / +0.92 / +1.24 %, H.264 Veryfast +0.95 / +1.12 / +1.46 %, H.264 Fast +0.82 /
  +0.74 / +1.67 %, VP9 Fast +0.38 / +0.66 / +0.50 %, HEVC Faster +0.68 / +0.78 / +0.66 %,
  VVC Fast +0.38 / +0.11 / +0.38 %; model, HEVC Faster +1.03 / +0.56 / +0.69 %, VP9
  Faster +2.68 / +2.91 / +1.76 %. (Jetson cycles, probe: VP9 Faster +1.80 %, Veryfast
  +3.02 %, H.264 +0.81..+0.94 %, VP9 Fast +1.09 %, HEVC Faster +0.60 %, VVC Fast -0.16 %.)
- **Checks** (k 0.6; the screens' dCPU was the Jetson/x86 mean, the last column re-prices
  L3 / final-check at the cloud dCPU, ARM and x86 equal):

  | rung | mode | stage 1 L1 / probe J | L3 J | final-check J | cloud dCPU | re-priced L3 / final | outcome |
  |---|---|---|---|---|---:|---|---|
  | VVC Fast | probe | -0.43 / -0.71 | -0.48 [-0.90, -0.06] | -0.44 [-0.60, -0.18] | +0.31 | -0.30 / -0.26 | landed |
  | VP9 Fast | probe | +0.11 / +0.26 | -0.05 [-0.21, +0.09] | -0.77 [-1.06, -0.40] | +0.51 | -0.21 / -0.93 | landed |
  | VP9 Faster | probe | -0.60 / -0.03 | -0.44 [-0.84, -0.06] | -0.68 [-1.19, -0.36] | +1.87 | -0.20 / -0.44 | landed |
  | HEVC Faster | model | -0.74 / -0.28 | -0.26 [-0.55, +0.03] | -0.70 [-1.02, -0.49] | +0.74 | -0.22 / -0.66 | landed |
  | H.264 Fast | probe | -0.20 / +0.48 | -0.34 [-0.71, +0.01] | -0.24 [-0.52, -0.02] | +1.23 | -0.06 / +0.04 | not landed (neutral) |
  | H.264 Faster | probe | +0.11 / +0.20 | +1.11 [-0.45, +3.85], videocall guard +9.7 % | -0.06 | - | - | not landed |
  | H.264 Veryfast | probe | +0.01 / +0.24 | +0.48 [-0.55, +2.09], videocall guard +5.1 % | -0.02 | - | - | not landed |
  | VP9 Veryfast | probe / model | killed: probe J +0.91 [+0.51, +1.32] / +1.87 | - | - | - | - | not landed |

  VP9 Fast's L3 was labelled BETTER by the train rule (mean below 0, upper bound at most
  +0.3); its final check carries the gain.
- **Not carried over:** the model mode loses on H.264 (its bilinear ranking keeps the
  unmeasured cost); AV1 / AV2 never skip the quarter stage; Default and Slow run it in full.

**1/8-sample vectors (VP9, the only codec whose encoder has them): rejected**
(ledger `vp9-high-precision-mv-default-slow`). `vp9.high_precision_qindex` (1/8 allowed
below that qindex): Default, all qindex: quick +1.39 % [+0.70, +2.18], adopt +0.43 %
[+0.02, +0.89], final +1.04 % [+0.78, +1.32] (n 108); gate 200: +1.05 / +0.87 %; gate 128:
+0.05 / -0.06 / +0.03 % (no gain); Slow +1.03 % [+0.64, +1.38]. By rung (quick): qindex
120-180 +1.2..+2.8 % bits with PSNR -0.02..-0.05 dB, about half rate and half distortion;
the finest rungs (qindex 45-90, 5 points) +0.2..+0.45 % bits for no quality. By source
origin (adopt + final, per clip): H.264-origin +1.27 % [+0.81, +1.85] (n 134), lossless
FFV1 / raw origin +0.81 % [+0.28, +1.30] (n 15): smaller, not reversed; the raw
`texture_stockholm_360p` is among the worst. With the frame flag on, every NEWMV near a
small reference vector codes a precision bit per component (VP9 `use_mv_hp`); how much
of the rate is that bit was not separated. AV1 and AV2 code quarter samples only
(`allow_high_precision_mv` 0; our AV2 decoder subset is quarter-only), so no test there.

## 5. Temporal-dependency λ in the other codecs and presets

- **Source.** Y. Gao, C. Zhu, S. Li, T. Yang, "Temporally Dependent Rate-Distortion
  Optimization for Low-Delay Hierarchical Video Coding", IEEE TIP 26(9):4457-4470, 2017
  (<https://ro.uow.edu.au/eispapers1/707>), partial (abstract from a search excerpt; the
  page refused a direct read).
- **What.** Propagation chains between coding units over time give each unit a λ (and
  optionally QP) from how much later frames depend on it, instead of a fixed
  low-delay cascade.
- **Reported.** 2.5 % BD-rate (low-delay P) from λ alone, about 5.4 % with QP adaptation
  (PSNR, HEVC reference). Whether the published scheme needs future frames was not
  checked.
- **Us.** We have the causal form: `causal-tpl-layer-propagation` (adopted, AV1 Faster:
  pool L3 J -1.32 %, final-check -0.88 %; "AV2 Fast L1 -0.30 % not pursued").
  `causal-tpl-reuse-forecast-aq` (rejected, +6 %) was the forecast-driven AQ variant.
  Open: the adopted mechanism in the other layered presets and codecs (HEVC, VVC, VP9,
  H.264), and on the flat stream's QP cadence, where "base" frames play the T0 role.
- **Fit: medium.** The mechanism is built and shared; this is the "consider every codec
  and preset" pass, with the paper as evidence that the low-delay P case pays.
- **Effort: S-M** (screens, plus block-QP signalling checks per codec).

## 6. AV1 local warped motion

- **Sources.**
  - H.-C. Chuang, Z. Lei, A. Opalach, A. Norkin, "Analysis of AV1 Coding Tools" (Meta,
    Netflix; <https://research.facebook.com/file/115187374892843/Analysis_of_AV1_Coding_Tools.pdf>),
    verified (tables read from the PDF text).
  - S. Parker, Y. Chen, et al., "Global and locally adaptive warped motion compensation
    in video compression", ICIP 2017, pp. 275-279
    (<https://doi.org/10.1109/ICIP.2017.8296286>), partial (record only).
- **What.** A block's affine model is fitted from the vectors of its neighbours, so the
  encoder searches nothing: one more candidate per block that has a new or predicted
  vector.
- **Reported.** libaom-research, random access, tool on alone: -1.21 / -1.26 / -1.44 %
  (PSNR-Y / SSIM / VMAF) at 102.8 % encode and 101.4 % decode time; tool off from the
  full set: +0.86 / +1.00 / +0.89 %.
- **Us.** The AV1 encoder writes `enable_warped_motion` 0 (`nc-av1/src/obu.rs`); the
  decoder has the warp filter and the model fit. Bodycam and drone clips zoom and
  rotate, which translation cannot follow. The candidate needs the final stage's
  pricing or the ranking's variance measure on a warped prediction.
- **Ledger.** No encoder idea under warp. `av1-search-free-obmc` (rejected, VMAF -4.6) is
  the nearest: another search-free motion mode that lost.
- **Fit: medium.** Small average, low-delay single-reference not measured by the source,
  decode cost small. AV2 only if its subset gains the mode.
- **Effort: M.**

## 7. Transform-type shortlist by SATD and neighbours

- **Sources.**
  - SVT-AV1 `Docs/Appendix-TX-Search.md`
    (<https://gitlab.com/AOMediaCodec/SVT-AV1/-/raw/master/Docs/Appendix-TX-Search.md>),
    verified (no numbers).
  - L. He, S. Xiong, R. Yang, X. He, H. Chen, "Low-Complexity Multiple Transform Selection
    Combining Multi-Type Tree Partition Algorithm for Versatile Video Coding", Sensors
    22(15):5523, 2022 (<https://pmc.ncbi.nlm.nih.gov/articles/PMC9331267>), verified.
- **What.** SVT tests types in ordered groups, drops a type whose SATD is too far from
  the best (threshold scaled by QP) and stops when distortion is already small. He et
  al.: if no spatial neighbour used a non-default type, test the default only;
  otherwise test in order of neighbour use and stop when the cost stops improving.
- **Reported.** He: -26.4 % time for +0.13 % BD-rate (VTM-3.0, all intra).
- **Us.** `av2-inter-transform-types-search` was rejected on cost (J -0.7 % quality at
  +7.1 % P instructions); VVC MTS lost outright (`vvc-mtt-mts-and-dependent-quantisation`,
  +1.1 J). A neighbour gate plus SATD pruning aims at the AV2 case: keep the -0.7 % at
  a fraction of the +7.1 %.
- **Fit: medium** for AV2 and AV1 Slow/Default; low for VVC until its MTS decision is
  fixed.
- **Effort: S-M.**

## 8. Integral-projection seed for large camera motion

- **Source.** libvpx commit "Integral projection based motion estimation", Jingning Han,
  2015-02-13 (<https://aomedia.googlesource.com/avm/+/ed2dc59c1b363ab510344b28ee0367c50c5080c4%5E%21>),
  verified.
- **What.** A 64x64 block and a +-32 reference area are reduced to row and column sums;
  each axis is matched by a 1-D SAD, coarse to fine. The vector seeds the search.
- **Reported.** VP9 real time, speed 6, 1080p 500 kbps: blue_sky -30 % bits and 25 %
  faster, pedestrian_area -4 %; about 2 % more CPU on low-motion content.
- **Us.** We seed from last frame's vector or the camera model and run HME at 1/4
  resolution, so most of this exists. It could replace HME where the seed fails
  (motion onset, whip pans) at lower cost than the HME window.
- **Ledger.** No projection search. `hme-range-reduction` and
  `full-pel-window-early-exit-by-centre-sad` are held; the seed-gated HME skip was
  rejected (`fast-paths.md` item 4: -0.5..-1.0 % on pans only).
- **Fit: low-medium.** HME is 0-4 % of Jetson time; the gain would be quality at motion
  onset, not speed.
- **Effort: S-M.**

## 9. LTR feedback audit; layer-weighted protection

- **Sources.**
  - F. Zhou, F. Lyu (Meta), "Enhancing Video Network Resiliency With LTR and RS Code",
    @Scale, 2024-03-20
    (<https://atscaleconference.com/enhancing-video-network-resiliency-with-ltr-and-rs-code/>),
    verified.
  - RFC 4585 sections 6.3.1 and 6.3.3 (<https://www.rfc-editor.org/rfc/rfc4585.txt>),
    verified. RFC 5109 (<https://www.rfc-editor.org/rfc/rfc5109>), partial.
- **What.** The same design as ours (acknowledged long-term reference, recovery by a P
  frame from it), with production findings: a recovery frame is 40-50 % smaller than a
  key and key frames under loss fell about 20 %. Pitfalls they hit: acknowledging on
  receipt instead of after decode caused long freezes; acknowledgement state must be
  cleared after a key (encoder deadlock). Unequal protection: more redundancy on the
  base temporal layer than on the layers above it.
- **Us.** `ltr-with-loss-feedback-one-long-term-slot` and
  `ltr-recovery-vs-key-only-repair-under-loss` are adopted (recovery 26 kB against a
  150 kB key on crowd 720p). The audit is whether our simulator and API contract cover
  the two pitfalls (ack after decode; state reset on every key, including requested
  and scene keys), and a documented recommendation to weight transport protection by
  the `discardable` / temporal-layer packet metadata we already emit.
- **Fit: medium** (correctness and guidance, not J). **Effort: S.**

## 10. Modelled background as a long-term reference

- **Sources.**
  - X. Zhang, T. Huang, Y. Tian, W. Gao, "Background-Modeling-Based Adaptive Prediction for
    Surveillance Video Coding", IEEE TIP 23(2):769-784, 2014
    (<https://dblp.dagstuhl.de/rec/journals/tip/ZhangHTG14.html>), partial.
  - X. Zhang, Y. Tian, T. Huang, S. Dong, W. Gao, "Optimizing the Hierarchical Prediction
    and Coding in HEVC for Surveillance and Conference Videos With Background Modeling",
    IEEE TIP 23(10):4511-4526, 2014, partial.
  - "Introduction to AVS2 Scene Video Coding Techniques", ZTE Communications 2016
    (<https://www.zte.com.cn/global/about/magazine/zte-communications/2016/1/en_214/448974.html>),
    partial.
- **What.** A background picture is built from the source over time, coded at a fine
  quantiser, optionally not displayed, and kept as a long-term reference; blocks that
  uncover background predict from it.
- **Reported.** 39.09 % bit saving on surveillance and 5.27 % on conference video (HEVC
  low delay, second paper); AVS2 "nearly 40 %" on scene video.
- **Us.** Our second reference is the last high-quality frame:
  `h-264-second-reference-last-hq-frame` (held: J -1.2 % at +23 % P instructions),
  `hevc-second-reference-refs-second` (held: L3 -1.28 % at +16.9 % Jetson),
  `reference-plan-second-reference-on-deep-hq-frames-structure-unified` (adopted: AV1
  -0.53 %, VP9 -0.67 %, HEVC -0.33 %). `docs/handoff/engine.md` found that static
  backgrounds gain nothing from it (the previous frame already holds the detail). A
  modelled background differs in one way: it holds what is *currently occluded*, which
  no recent frame has.
- **Fit: medium-low.** The published sizes are against an anchor without any long-term
  reference; ours would be the uncovered-area share on fixed cameras with traffic.
  Costs: a non-displayed picture (AV1/VP9/AV2 natively, HEVC `pic_output_flag`), its
  bits inside a 1 s buffer, and the second reference's per-block trial cost.
- **Effort: L.** Check first, offline: the share of coded bits in blocks that a
  running-median background would predict better than the previous frame.

## 11. VMAF-coupled block λ

- **Source.** Z. Luo, C. Zhu, Y. Huang, R. Xie, L. Song, C.-C. J. Kuo, "VMAF Oriented
  Perceptual Coding Based on Piecewise Metric Coupling", IEEE TIP 2021
  (<http://medialab.sjtu.edu.cn/publication/luo-2021-vop/>), verified (publication page).
- **What.** VMAF is modelled as locally linear in squared error near a reference
  distortion, giving a per-block λ without pre-filtering.
- **Reported.** 3.61 % bit saving at equal VMAF, HEVC low-delay P (2.67 % random access).
  PSNR and MS-SSIM effects are not on the page.
- **Us / ledger.** Every block-QP redistribution so far lost on J: `variance-aq-at-strength-1`
  (PSNR -0.4..-0.9 dB for MS-SSIM +0.05..+0.29 dB), `variance-block-aq-rate-aq-strength-for-j`
  (held-out -0.09 %), `svt-style-one-sided-variance-boost-aq` (+9.5..+31 % J_real),
  `h-264-variance-aq`. VMAF is one third of J and v1, not the paper's version.
- **Fit: low-medium.** Worth an offline check only: replay block λ offsets on logged
  per-block SSE and score all three metrics before any encoder work.
- **Effort: M.**

## 12. XPSNR-style spatio-temporal activity QP

- **Source.** C. R. Helmrich, S. Bosse, H. Schwarz, D. Marpe, T. Wiegand, "A Study of the
  Extended Perceptually Weighted Peak Signal-to-Noise Ratio (XPSNR) for Video
  Compression with Different Resolutions and Bit Depths", ITU Journal: ICT Discoveries
  3(1), 2020 (<https://www.itu.int/dms_pub/itu-s/opb/journal/S-JOURNAL-ICTS.V3I1-2020-8-PDF-E.pdf>),
  verified (pages 1-6).
- **What.** Block weight (a_pic / a_k)^0.5 with activity a_k from a 3x3 spatial
  high-pass plus twice the temporal high-pass (frame difference), with a floor. The
  QPA lineage (JVET-H0047, K0206) turns the weight into a block QP offset.
- **Reported.** Correlation with subjective scores over eight datasets: Pearson 0.822
  (VMAF 0.822, MS-SSIM 0.758, PSNR 0.702). No BD-rate in the paper.
- **Us / ledger.** As 11. The new element against our rejected variance AQ is the
  temporal term (moving detail takes a coarser step). It would help the BD-XPSNR
  column we publish; on J it has to beat the three rejections above.
- **Fit: low-medium. Effort: S** (the pre-analysis has the frame-difference SADs).

## 13. Cyclic refresh with a coded-Q map, static blocks only

- **Source.** libvpx `vp9/encoder/vp9_aq_cyclicrefresh.c`
  (<https://chromium.googlesource.com/webm/libvpx/+/refs/heads/main/vp9/encoder/vp9_aq_cyclicrefresh.c>),
  verified (mechanism; no numbers, no paper found).
- **What.** Each frame a rotating 10 % of blocks is coded at a finer quantiser so static
  areas converge. Blocks that are moving, intra or high-distortion are excluded; a
  per-8x8 map of the last coded Q stops blocks already at a fine Q from being
  refreshed again; off on upper temporal layers and at very low Q.
- **Ledger.** `cyclic-spatial-qp-cadence-band` / `cyclic-spatial-cadence-band-rate-cadence-spatial`
  rejected: AV1 Default 600 f final J +2.01 % with the pan guard tripped, CPU +15.6 %
  on static. That variant moved the base-frame boost to a rotating band of still
  squares. It is not clear from the record whether it kept a coded-Q map; the pan
  guard says it acted on moving content.
- **Fit: low-medium.** Our base-frame cadence already does the same job per frame, and
  delta-Q signalling cost AV1 118 -> 862 bits per P frame in the AQ screens. Re-test
  only as a static-camera variant with the coded-Q map.
- **Effort: M.**

## 14. AV1 global motion from the camera model we already fit

- **Sources.** Chuang et al. (as in 6), verified. SVT-AV1 `Docs/Appendix-Global-Motion.md`
  (<https://gitlab.com/AOMediaCodec/SVT-AV1/-/raw/master/Docs/Appendix-Global-Motion.md>),
  verified.
- **What.** Frame-level warp parameters per reference; blocks select them with a cheap
  mode. SVT can derive a translation-only model from its motion vectors without
  feature matching and skips the search when vectors are uniform.
- **Reported.** libaom tool on alone: -0.31 / -0.38 / -0.57 % at 135.7 % encode time (the
  RANSAC fit); tool off: +0.06 / 0.00 / -0.09 %. "Highly sequence dependent": BlueSky
  -8.61 / -10.90 / -12.16 %, minor elsewhere.
- **Us.** `nc-analysis/src/global.rs` already fits translation + zoom per frame, so the
  fit costs nothing; the work is signalling, the GLOBALMV candidate and a warped
  prediction path. Not in the ledger for the encoder.
- **Fit: low-medium** (near zero on average, large on rare zoom/pan clips; 6 covers more
  of the same content block by block). **Effort: M.**

## 15. Statistics-only SAO class and ALF decisions

- **Sources.**
  - Joo, Choi, Lee, "Fast sample adaptive offset encoding algorithm for HEVC based on
    intra prediction mode", ICCE 2013; Joo and Choi, dominant-edge-direction SAO, ICIP
    2014 (records <https://manuscript.isc.ac/Inventory/49/664852.htm>,
    <https://manuscript.isc.ac/Inventory/49/249102.htm>), partial.
  - Meng, Zhang, Jia, Zhang, Wang, Ma, "Optimized Adaptive Loop Filter in Versatile Video
    Coding" (<https://arxiv.org/abs/2607.05737>), verified (abstract).
- **What.** Test one edge-offset class per CTB, chosen from the intra mode or a local
  edge-direction histogram, instead of four. ALF: estimate distortion from the
  covariance statistics without filtering, in one pass.
- **Reported.** SAO: 77.3 % of SAO encode time for 0.8 % BD-rate (HM 10.0); 42.1 % for
  0.25 % (HM 12.0). ALF: about -25 % ALF encode time, "negligible" loss (random access).
- **Ledger.** `shared-hevc-vvc-sao-decision-hm-structure` adopted; `sao-type-search-every-picture`
  rejected (the type search already runs every fourth picture); `hevc-sao-skip-in-unchanged-ctbs`
  adopted; `vvc-alf-exact-decision-only-on-estimate-on-ctbs` rejected (-0.56 %).
- **Fit: low-medium.** With the search interval at 4 most of the SAO saving is already
  taken; the remaining target is the fast rungs where SAO off buys -5.8 % CPU for
  +4.4 % quality (`hevc-veryfast-rung-parameters`).
- **Effort: S-M.**

## The rest, briefly

Covered by what we have, or tried and closed (do not re-propose without a new reason):

- **Early skip / merge termination** (ESD: Kim, Yang, Won, Jeon, PCS 2012, verified, up to
  36 % time against HM 4.0; ECU/CFM and merge-CBF rules: Wu et al., KSII TIIS 2016,
  verified, -29..-46 % time at +0.3..2.4 %). Ledger: six early-skip variants rejected
  (`early-skip-test-before-motion-search`: crowd +8.8 %, lowlight +6.4 %); the RD
  partition search has the skip exit.
- **Zero-block detection by SAD/SATD thresholds** (Lv et al., SPIE 2014, partial; libaom
  `set_force_skip_flag`, verified). `pre_tx_skip`, `still_skip` adopted; the provable
  test never fired.
- **Variance partitioning and superblock zero-vector skip from source SAD** (libaom
  `var_based_part.c`, verified). Ours is SVT's light PD0 with static bypass; the
  libaom 8x8 min/max forced split is the one rule we lack (small).
- **Depth range from neighbours or the reference frame's partition** (Shen et al., IEEE
  TMM 2013, partial; Qureshi, Amirpour, Timmerer, arXiv:2503.01415, verified, about 21 %
  time against VVenC medium). Ledger: `av2-motion-gated-base-partition-depth` rejected
  (J +0.32 %).
- **CNN partition predictors** (Paul, Norkin, Bovik, arXiv:1906.06476; Tissier et al.,
  arXiv:2103.05319; both verified, intra only, -47..-70 % time at +0.9..2.6 %). Intra
  results against full RDO; too heavy for one A78 core.
- **Gradient/HOG intra pruning** (libaom `intra_pruning_with_hog`, verified; Jiang, Ma,
  Chen, CECNet 2012, partial). Ledger: `av1-key-frame-direction-pruning-by-gradient-histogram`
  rejected (no saving after kernel speed-ups).
- **Decision trees for AV1 inter modes** (Kim et al., arXiv:1908.11166, verified, -43 %
  against the full RD search): no RD search to prune here.
- **R-λ rate control and CTU-level allocation** (Li, Li, Li, Zhang, IEEE TIP 2014, partial;
  S. Li, Xu, Wang, Sun, IEEE TCSVT 2017, partial; Yang, Song, Luo, Wang, BMSB 2014,
  verified abstract). We have a per-codec rate model and `inframe` steering;
  `in-frame-row-steering-as-re-code-replacement-inframe-room` rejected.
- **libvpx/SVT one-pass CBR details** (libvpx `vp9_ratectrl.c`, SVT `Appendix-Rate-Control.md`,
  verified): buffer-zone Q ceiling, damped rate-model correction with a dead band, a
  clamp between the last two Q after an over-then-undershoot, overshoot re-encode.
  Mostly present; the oscillation clamp is a cheap candidate for the still-scene
  instability (`rate-control-instability-on-still-scenes`, partial).
- **ρ-domain rate model** (He, Mitra, IEEE TCSVT 2002, partial): our models are fitted
  per codec already.
- **Adaptive rounding offset** (G. Sullivan, JVT-N011, 2005, verified): up to 1 dB at
  high PSNR, "little or none at low fidelity" by its own account.
- **Sign data hiding** (JCTVC-G271, H0481, partial): adopted in HEVC and VVC. AV2 parity
  hiding is `unclear` in the ledger with no quality number: a loose end to measure.
- **Table and model bit estimates** (JCTVC-G763, partial; SVT eob-only rate): done
  (`symbol_cost`, `av2-fitted-coefficient-level-rate-model-for-trials`); AV1 linear
  pricing rejected.
- **SSIM-based RDO and divisive normalisation** (Wang, Rehman, Wang, Ma, Gao, IEEE TCSVT
  2012 and TIP 2013, partial): same family as 11-12.
- **VMAF pre-processing** (Deng, Han, Xu, MMSP 2020, partial, about 37 % VMAF BD-rate;
  libaom `tune_vmaf.c`): sharpening the source games one third of J and costs the
  other two; not proposed.
- **SVT variance boost** (`Appendix-Variance-Boost.md`, verified): rejected here
  (`svt-style-one-sided-variance-boost-aq`).
- **Psy-RD / AC bias** (SVT `--ac-bias`, partial): `h-264-psy-strength-0-25` partial,
  `hevc-psy-term-on-vc-1k` rejected.
- **EPZS predictors and thresholds** (Tourapis et al., JVT-E023, verified): our seeds
  cover the list except the accelerator vector; SEA/PDE early exits are inside the
  SAD kernels' cost range already.
- **SVT pre-HME line search and search-area divisors**
  (`Appendix-Open-Loop-Motion-Estimation.md`, verified): `hme-range-reduction` held.
- **Deblock level and CDEF strength from Q; CDEF reuse from references; filter off on
  skip-heavy frames** (libaom `picklpf.c`, `pickcdef.c`; SVT `Appendix-CDEF.md`; all
  verified, no numbers). `cdef_model` adopted; reuse drifted on long GOPs and
  `av2-cdef-reuse-motion-deblock-bundle`, `av1-frame-level-loop-filter-gate`,
  `av1-per-block-cdef-skip-on-static-content` are rejected.
- **Loop restoration shortcuts** (SVT `Appendix-Restoration-Filter.md`, verified):
  `av1-loop-restoration` rejected at +142 % instructions; previous-frame coefficients
  would not close that.
- **Rectangular AV1 partitions** (Chuang et al.: tool off +5.11 %, the largest
  partition tool, with an RD search): `engine-rectangular-partition-shapes` held; VVC
  shows shapes lose without coded-cost decisions, so this follows 1.
- **AV1 motion-field vectors** (Chuang et al.: -1.06 % on, +0.77 % off, decode time
  +17 %): the decode cost is against our WASM priority.
- **Golden refresh gated on low motion, golden boost in CBR** (libvpx, verified): our QP
  cadence and second reference.
- **Loss-aware mode decision (ROPE)** (Zhang, Regunathan, Rose, IEEE JSAC 2000, partial)
  and **leaky prediction**: built for partial-frame loss or outside standard syntax;
  with whole-frame loss and feedback, LTR recovery is the fit.
- **Quantisation matrices**: SVT-AV1 ships them off by default; no measured gain was
  found in any source read, so not ranked.

## Not found

No source was retrieved for: JVET-K0390 / M0600 result numbers, JCTVC-M0036, a cyclic
refresh paper, sign-data-hiding gain figures, a fast-CfL paper, JVET-O0549 / X0116
low-delay numbers, deblock-offset-against-VMAF studies, or a published WebRTC L1T3
experience paper.

## Ledger records

Held (queued) ideas added 2026-11-21, each pointing at its section here:
`classifier-gated-rd-partition-search`, `causal-temporal-source-filter`,
`cheap-rdoq-substitute-fast-rungs`, `sub-pel-points-from-integer-cost-surface`,
`temporal-dependency-lambda-other-codecs`, `av1-local-warped-motion-encoder`,
`transform-type-shortlist-satd-neighbours`.
