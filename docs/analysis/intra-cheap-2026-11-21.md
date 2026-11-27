# Key-frame gains without the cost (2026-11-21)

**Question (owner).** The key-frame RD searches of `key-frame-rdo-2026-11-21.md` make a key 4-14 %
cheaper at 1.4-9x its CPU. How much of that can be had at close to today's key CPU, and how does our
front of key BD-rate against key CPU compare with SVT-AV1's still-picture front (the bar: at or below
it at every CPU point)?

Method: `nceval.fixedqp ab --frames 1 --key-share 0.15`, quick level of draw `fixedqp-2026-11-21`
(24 clips), BD-J against each codec's medium key, key CPU as the whole one-frame process's CPU
seconds (as SVT's points were timed). Ranking evidence only; nothing here changes a preset.

## 1. What SVT-AV1 does on a still (`--avif 1`), M8 / M6 / M4

Read from v4.3.0 (`Source/Lib/Codec/enc_mode_config.c` `svt_aom_sig_deriv_*_allintra`,
`product_coding_loop.c`, `enc_dec_process.c`, `md_process.c`, `rd_cost.c`, `full_loop.c`);
`[inferred]` marks judgement. `--avif 1` sets `allintra`, its own preset tables (no base/reference
branches), and switches off TPL, temporal filtering, motion estimation, the downsampled pictures and
QP scaling. Light PD1 never applies to it. Thresholds marked `·q` scale with
q = max(10, qp)/63 below qp 46, else 1.05 − exp(−(qp − 35)/10).

| Control | M2 | M4 | M6 | M8 |
|---|---|---|---|---|
| Superblock | 64 | 64 | 64 | 64 |
| Smallest block | 4 | 8 | 8 | 8 |
| PD0 (planning pass) | level 1 | level 1 | level 1 | level 1 |
| Depth refinement after PD0 | level 6: ±1 depth, gated | level 6 | level 10: PD0's tree is coded | level 10 |
| Rectangular shapes | H, V, H4, V4 (pruned) | none | none | none |
| Luma candidates ranked by SATD | 61 + filter DC | 61 + filter DC | DC, V, H, smooth, filter DC | DC, V, H, smooth, pruned to 2-4 |
| Candidates to luma-only coefficient cost (MDS1) / full RD (MDS3) | 48 / 12 | 24·q / 6·q, within 20 % then 15 % of the best | 24·q / 6·q, tighter | 0 / 1 (MDS1 skipped) |
| Transform size search | 2 levels | 1 level (stops if depth 0 has no coefficients, or on cost) | 1 level | none |
| Transform types (blocks under 16 / from 16) | all 6 groups | all 6 groups, SATD exit 15·q % | 5 / 4 groups, exit 10·q % | 3 / 2 groups, exit 10·q % |
| RDOQ | full | full | by complexity: off / cut-off 60-80 % | as M6 |
| Coefficient rate in full RD | exact walk | exact walk | exact walk | 6000 + 1000·eob for sparse blocks, else the walk with contexts 0 |
| CDF / rate-table update per superblock | yes | yes | yes | no: frame-initial tables |
| Chroma mode | searched over the survivors' modes (+DC) | same | follows luma | follows luma |
| CfL | on (blocks to 32, chroma variance gate) | on | on | off |
| Separate final coding pass | yes | no: MDS3's result is the stream | no | no |
| Deblocking | full-image search | search with early exits | from Q | from Q |
| CDEF | 15 strength pairs | 3 strengths | 2 strengths, subsampled 4 | from QP |
| Wiener restoration | on, refined | on | on | off |

**The planning pass (PD0), the same at M2-M8.** Top-down over the square sizes 64 to 8. Per block:
one candidate (DC), luma only, predicted from the *source* neighbours (no reconstruction, so no
inverse transform), one DCT at the largest transform, plain quantiser, coefficient-domain distortion,
rate from half a coefficient walk with zero contexts (M7-M8: `6000 + 500·eob` for sparse blocks).
Depth exits: the split is not tried when the parent's cost is under 20x the split flag's cost; the
remaining quadrants stop once parent ≤ split bits + quadrants so far. This is the structure of our
`key_plan.rs` (which cites it) with our exits.

**Depth refinement (M2-M4, level 6).** Per PD0 leaf at most one parent depth and one child depth are
added to PD1, and only where PD0's own costs leave it open: the parent where its PD0 cost is within
15 % of the four current blocks' (and above a floor), the children where their PD0 cost plus the
split flag is within 15 % of the current block's; neither beyond the range of sizes PD0 chose in the
superblock. PD1 then decides by full RD with the parent biased by 0.995, stopping the remaining
quadrants on the parent's cost, and skips a child depth of blocks to 16x16 when the four quadrant
distortions are even (standard deviation under 250) and under 15 % of the coefficients are nonzero.
M6 and M8 code PD0's tree as it is.

**Staged mode decision (PD1).** MDS0 predicts every candidate from the reconstruction and ranks by
Hadamard SATD + λ · mode bits (with the chroma mode's bits assumed to follow luma). MDS1 (M2-M6):
luma only, DCT, largest transform, plain quantiser, coefficient-domain distortion, exact rate. MDS3:
transform size and type, RDOQ, spatial SSE, chroma and CfL, on the survivors only. At M8 one SATD
winner of 2-4 candidates goes straight to MDS3.

**Why M8 costs a quarter of our key [inferred].** One full RD per leaf after 2-4 SATDs (we
transform-price about ten modes); the planning pass's exits; the planned tree coded once with MDS3's
output as the stream; modelled rate for sparse blocks and frozen rate tables; chroma once, following
luma; no filter search.

**Where M4's 14 % over M8 comes from [inferred].** Directional modes with angle deltas through the
staged decision (M6 and M8 test none); depth refinement by full RD; an independent chroma mode over
the survivors; the full transform-type search with RDOQ on every tested type; transform size depth 1;
searched deblocking and CDEF.

M2 adds the four rectangular shapes, 4x4 blocks, transform depth 2, twice the candidates per stage,
a larger CDEF search and a separate coding pass.

## 2. Where our key's time goes (perf, hardware cycles)

`eval/intracheap/key_profile.sh` on the build box (c7a, one pinned core, `perf stat -r 5` and
`perf record -e cycles`), one 720x1280 crowd frame coded as a key at our medium settings, SVT-AV1 in
its still mode beside it. Our `--qp` and SVT's are different scales and the sizes below are far
apart, so this table gives shares and orders of magnitude, not the rate-matched comparison (that is
the harness's, section 4).

| Encoder | quantizer | bytes | CPU ms |
|---|---|---|---|
| ours AV1 medium | 120 / 180 / 225 / 250 | 373 k / 192 k / 93 k / 53 k | 250 / 188 / 146 / 109 |
| SVT-AV1 still M8 | 35 / 50 / 58 / 63 | 101 k / 26 k / 8.3 k / 1.9 k | 50 / 32 / 25 / 21 |
| SVT-AV1 still M6 | same | 101 k / 28 k / 8.8 k / 2.0 k | 179 / 102 / 74 / 56 |
| SVT-AV1 still M4 | same | 100 k / 28 k / 8.9 k / 1.8 k | 700 / 530 / 350 / 238 |
| ours HEVC medium | 30 / 40 / 46 / 51 | 290 k / 124 k / 59 k / 26 k | 208 / 142 / 95 / 61 |
| ours H.264 medium | 30 / 40 | 289 k / 115 k | 94 / 69 |
| ours VP9 / AV2 medium | 180 | 197 k / 107 k | 241 / 156 |
| ours VVC medium | 38 | 111 k | 331 |

Self cycles by symbol, medium key:

| Codec | Pricing coefficients (the context walk into a bit counter) | Other large items |
|---|---|---|
| AV1 | `CoeffCdfs::cost` 17.6 % at qindex 250, 29.7 % at 180, 39 % at 120; range symbols into `BitCounter` 3-6 % | intra prediction and edge preparation 16 %, quantiser 10 %, the real entropy write 3-7 %, CfL 4 %, memmove 3-5 % |
| HEVC | `write_residual::<BitEstimator>` 17-21 % | `code_residual` 16-18 %, trellis rate model 7 %, the real CABAC write 5-7 % |
| VP9 | `coeff::cost` 21.8 % | `code_residuals` 13.8 %, prediction 6 %, trellis level rates 4.8 % |
| VVC | `walk_sized::<BitCost>` 19.9 % | `quantize_rect` 16 %, prediction 5 % |
| H.264 | residual and syntax into `BitCounter` 22.7 % | Intra_4x4 RDO 11.6 % + `code_intra_4x4_mb` 10 % |
| AV2 | `write_typed_with_stats` 9.8 % | parity hiding's `conform` 8 % (on in AV2's Default), prediction 9 %, CDEF 3.4 % |

SVT-AV1 M8 spends 1-3 % in `svt_av1_cost_coeffs_txb` and under 5 % in RDOQ; M4, with about ten
times the candidates, 6.7 % and 17.5 %.

The same walk on inter frames (31 frames, one key, fixed QP): AV1 `CoeffCdfs::cost` 15.7 % of the
whole encode (trellis 7.6 %); HEVC `write_residual::<BitEstimator>` 9.5 %, `code_residual` 13.6 %,
trellis rate model 7.3 %.

**Reading.** Every codec of ours ranks about ten luma candidates per leaf by transform, quantiser
and an exact context walk where SVT-AV1 ranks by SATD and walks the coefficients of one (M8) to at
most six (M4) candidates, from rate tables. That, not the partition search, is the first structural
difference at equal work; AV1's speed rungs already have the staged form
(`av1.keyframe_satd_modes`).

## 3. The staged decision, shared (design)

SVT-AV1's structure, as it would sit in the shared engine for intra and inter candidates alike. Each
stage is a cost every backend can already produce; what is missing is the staging itself and the
table-driven rate of stage 1.

| Stage | SVT-AV1 | Cost | Ours today | Shared form |
|---|---|---|---|---|
| 0 | MDS0, every candidate | prediction SATD + λ · mode bits | AV1's speed rungs (`av1.keyframe_satd_modes`), HEVC and VVC rankings, `rank_intra`; inter: the ranking of `choose_inter_mode` | `nc_encoder::rank_intra` / the inter ranking, one list of (estimate, candidate) |
| 1 | MDS1, at most N1, within a cost band of stage 0's best | luma only, DCT at the largest size, plain quantiser, coefficient-domain distortion, rate from tables | AV1 medium runs this stage on every candidate with the exact context walk (17-39 % of the key); `av1.keyframe_model_candidates` is a modelled-rate form of it; HEVC `intra_rd_candidates` | one engine driver over a backend's `luma_transform_cost(candidate)`; the band and N1 are engine parameters |
| 3 | MDS3, at most N3, within a band of stage 1's best | everything coded: transform size and type, RDOQ, chroma, spatial error, real contexts | `leaf_rd.rs` (`decision.intra_rd_modes`, with `decision.intra_rd_margin` as the band); inter: `final_stage.rs`, `cu_pricing.rs` | `leaf_rd.rs`'s node trials, for inter leaves as for intra |

- **The band is the lever, not the count.** `decision.intra_rd_margin` is stage 3's band on the
  backend's own estimate; section 4 has its curve.
- **Stage 1's rate.** The exact walk recomputes contexts per coefficient. SVT prices from rate tables
  built from the CDFs once per superblock (M4-M6) or frame (M8), and by `6000 + 1000 · eob` for sparse
  blocks at M8. AV1's existing linear end-of-block rate (`av1.key_linear_rate_eob_share`) used *inside
  partition trials* lost 3.8 of the search's 9.0 %: a modelled rate may rank modes, not decide
  partitions.
- **Depth refinement** is the partition search inside a window with the margin gate
  (`planner.key_rd_margin`, agent intra-selector) on the plan's own costs; SVT's level 6 is window
  1/1 with a 15 % band.
- Not built here: the engine driver for stage 1 and the per-superblock rate tables. They are the
  next step, for inter frames too (section 2: the walk is 10-16 % of the whole AV1 and HEVC encode).

## 4. Results

One frame, 24 clips, J % against each codec's medium key. CPU x: the candidate's whole-process CPU
over a null candidate's (a setting that changes nothing) timed in the same run
(`eval/intracheap/front.py`); between runs on the shared box the same candidate moved by up to 0.1.
`S(d/s, g)`: `planner.key_rd_split` with `key_rd_deeper` d, `key_rd_shallower` s and the margin gate
`planner.key_rd_margin` g (agent intra-selector's). Candidate sets: `eval/intracheap/b<n>-<codec>.txt`.

### 4.1 The ladder and the yardsticks on the key axes

`eval/intracheap/ladder.py` on a `fixedqp gap` run of every shipped preset and the references
(CPU x that codec's medium key process: AV1 38.6 ms, HEVC 29.8, H.264 17.2, VP9 53.8, VVC 89.8,
AV2 52.7). The SVT ratio is the same at every size (M8 over our medium: 0.54 at 1080p, 0.48 at 720p,
0.54 below), so start-up does not make it.

| AV1 | J % | CPU x | | HEVC | J % | CPU x |
|---|---|---|---|---|---|---|
| ours veryfast / faster / fast | +22.8 / +8.5 / +8.3 | 0.43 / 0.48 / 0.47 | | ours faster / fast | +18.2 / +10.5 | 0.44 / 0.51 |
| ours slow / slower | 0.0 / -5.8 | 1.01 / 1.35 | | ours slow / slower / veryslow | 0.0 / -1.5 / -3.5 | 1.01 / 1.46 / 2.02 |
| SVT-AV1 still M9 / M8 | +7.1 / +1.9 | 0.49 / 0.52 | | x265 ultrafast | -0.2 | 2.06 |
| SVT-AV1 still M6 / M5 | -1.2 / -8.1 | 1.57 / 3.58 | | x265 veryfast / fast / medium | -8.3 / -8.3 / -9.3 | 6.3 / 6.5 / 6.6 |
| SVT-AV1 still M4 / M3 / M2 | -12.0 / -15.2 / -17.3 | 6.4 / 13.8 / 37.7 | | x265 slow | -12.2 | 8.9 |
| libaom allintra cpu 8 / 6 | +21.8 / +5.1 | 1.10 / 4.44 | | | | |

| H.264 | J % | CPU x | | VP9 / VVC / AV2 (ours) | J % | CPU x |
|---|---|---|---|---|---|---|
| ours veryfast | +3.5 | 0.65 | | VP9 veryfast / faster / fast | +54.8 / +11.5 / +7.0 | 0.29 / 0.56 / 0.67 |
| ours faster / fast / slow | 0.0 | 1.0 | | VP9 slow / slower | 0.0 / -2.0 | 1.01 / 2.16 |
| ours slower | -3.2 | 2.94 | | VVC faster / fast | +6.7 / +4.1 | 0.51 / 0.61 |
| x264 ultrafast / veryfast | +73.5 / +3.3 | 0.42 / 0.64 | | VVC slow / slower / veryslow | 0.0 / -1.1 / -0.3 | 1.01 / 1.20 / 1.39 |
| x264 fast / medium / slow | -8.0 / -7.9 / -10.7 | 0.90 / 0.90 / 1.70 | | AV2 faster / fast | +5.6 / +0.9 | 0.54 / 0.82 |
| openh264 baseline | +51.4 | 0.82 | | AV2 slow / slower | 0.0 / -0.6 | 0.99 / 0.91 |

- Every codec's `slow` key is its medium key, and H.264's faster, fast and slow keys are too: the
  ladders have no key rung between fast and slower.
- No VP9, VVC or AV2 reference of the same standard was run (vpxenc is not stageable on the rented
  box; VVenC and AVM were not tried).

### 4.2 The partition search, gated

| Codec | S(0/2, 1.1) | S(0/2, 1.2) | S(1/1, 1.1) | S(1/1, 1.2) | S(1/2, 1.2) | S(2/2, 1.2) | S(2/2, 1.5) | ungated 2/2 |
|---|---|---|---|---|---|---|---|---|
| HEVC | -3.8 @1.26 | -4.5 @1.33 | -5.3 @1.41 | -6.0 @1.56 | -7.3 @1.74 | -7.5 @2.0 | -8.2 @2.2 | -8.2 @2.4 |
| VP9 | -5.1 @1.28 | -6.5 @1.38 | -4.2 @1.58 | -5.4 @1.89 | -6.9 @2.0 | -7.1 @2.6 | -7.5 @2.9 | -7.5 @3.2 |
| VVC | -3.9 @1.21 | -5.3 @1.35 | -6.1 @1.48 | -7.8 @1.69 | -8.9 @1.86 | -9.6 @2.2 | -10.4 @2.5 | -10.4 @2.7 |
| AV1, key trellis on | -6.1 @1.26 | -6.2 @1.35 | -7.5 @1.64 | -8.3 @1.93 | -8.3 @2.0 | -8.9 @2.6 | -8.9 @3.0 | -9.0 @3.2 |
| AV2, key trellis on | -4.4 @1.23 | -5.0 @1.33 | -6.3 @1.60 | -7.6 @1.90 | -7.6 @1.98 | -8.6 @2.7 | -9.1 @3.2 | -9.1 @3.35 |

- **Direction.** The merge-only window (no deeper level) holds 91 % of the full search's gain in VP9,
  56 % in HEVC and VVC, for a quarter to a third of its extra time: their plans over-split.
  Deepen-only gains nothing in VP9 (-0.4) and half in HEVC and VVC. AV1 and AV2 gain only from
  deepening (ungated 0/2 without the trellis: AV1 +0.5, AV2 -0.8).
- **A plan biased to split, then merge-only** (`planner.intra_bias` 1250, ungated 0/2): HEVC -7.4
  @1.85, VP9 -6.9 @2.1, VVC -8.6 @2.2: on the gated front's line, not below it.
- The key trellis alone (`quant.trellis_key`, off in AV1's and AV2's Default): AV1 -5.0 @1.03 (four
  states -6.1 @1.12), AV2 -3.5 @1.09 (MS-SSIM +0.9; four states -4.5 @1.17).

### 4.3 Modes by coded cost, cheaper (HEVC)

`decision.intra_rd_modes` N with `decision.intra_rd_margin` (only modes whose SATD estimate is
within the margin are coded) and `planner.key_rd_leaf_pass` (the modes are searched on the chosen
partition only):

| HEVC | J % | CPU x |
|---|---|---|
| 2 modes / 4 modes, no margin | -4.9 / -6.5 | 1.83 / 2.48 |
| 4 within 5 % / 4 within 10 % / 8 within 10 % / 8 within 20 % | -4.6 / -5.2 / -6.4 / -7.5 | 1.63 / 1.90 / 2.09 / 2.8 |
| S(1/2, 1.2) + 4 within 10 %, leaf pass | -9.7 | 2.81 |
| S(1/2, 1.2) + 8 within 10 %, leaf pass / in the trials | -9.8 / -11.8 | 3.05 / 4.39 |
| S(2/2) + 4 within 10 %, leaf pass / in the trials | -10.2 / -11.8 | 3.4 / 5.0 |
| S(2/2) + 4, no margin, in the trials (key-rdo's -12.5) | -12.5 | about 5.9 |

The margin trims a fifth of the mode search's time at equal J (8 within 10 % against 4 unmargined);
the leaf pass costs 2 points of J for a third less time. Both are points on the front, neither is
free. AV1: coded-cost modes stay under 1 % (key trellis + 2 within 5 %: -5.6 @1.31).

### 4.4 AV1 against SVT-AV1 still

| SVT-AV1 | J % | CPU x | Ours | J % | CPU x |
|---|---|---|---|---|---|
| M9 | +7.1 | 0.49 | shipped fast key | +8.3 | 0.47 |
| M8 | +1.9 | 0.52 | fast's key settings + key trellis | +1.7 | 0.54 |
| | | | the same, 2 coded candidates + CfL | -0.5 | 0.62 |
| | | | SATD-ranked directions, 6 by modelled cost, 3 coded, key trellis | -1.3 | 0.89 |
| M6 | -1.2 | 1.57 | medium, 1 transform-priced candidate, key trellis | -4.7 | 0.97 |
| | | | medium + key trellis | -5.0 | 1.03 |
| | | | + S(0/2, 1.2) | -6.2 | 1.31 |
| | | | + S(1/1, 1.2) | -8.3 | 1.94 |
| M5 | -8.1 | 3.58 | + S(2/2, 1.2) | -8.9 | 2.6 |
| M4 | -12.0 | 6.4 | S(2/2, 1.5), trellis 4 states, 8 candidates, deltas 2 / + 4 transform types by coded cost (4.10) | -10.3 / -12.0 | 4.2 / 9.5 |
| M3 / M2 | -15.2 / -17.3 | 13.8 / 37.7 | none | | |

With the key trellis on, our AV1 key front is at or below SVT-AV1's from M8's CPU to M5's, level with
M8, about 1 % short at M4's CPU, and has nothing at M3 and M2 (transform types, shapes and 4x4 blocks: agent
av1-key-tools). Fast's key settings are the staged decision of section 3 (SATD rank, one coded).

### 4.5 Tools that buy nothing on these keys

| Setting | J % [interval] | CPU x |
|---|---|---|
| HEVC `intra_nxn` 0 + `intra_tu_depth` 0 | 0.0 [-0.3, +0.4] | 0.82 |
| HEVC `intra_tu_depth` 1 / `intra_nxn` 0 | -0.1 / 0.0 | 0.96 / 0.97 |
| VP9 `intra_tx_depth` 0 | -0.3 [-0.6, +0.2] | 0.82 |
| AV2 `intra_chroma_candidates` 1 | -0.2 [-0.6, +0.2] | 0.96 |

At key sizes of 0.15 buffer, one frame. They also act on intra blocks of P frames (VP9's and AV2's
can be set for keys alone, `key.<name>`; HEVC's two are not re-read per frame).

### 4.6 Tried and removed

| Idea | Result | Why |
|---|---|---|
| Coded-leaf cost exit before the split (30 λ per 8x8) | HEVC +5.6 % J, VP9 -4.5 against -6.8 | coded leaves cost hundreds of λ per 8x8; a threshold that is safe skips nothing |
| Leaf coded before a planned split (the split bounded by it) | same streams, CPU 1.49 against 1.50 | the bound seldom stops a planned split early |
| Directional correction of the plan's DC cost (`intra_plan_direction` 0.25-1) | AV1, AV2 0.0; HEVC +1.0 to +10.0; VP9 +2.1 to +16.7; VVC +0.7 to +9.6 | every size is corrected, small blocks most: the over-splitting plans split more |
| AV1 linear end-of-block rate in trials / for ranking (existing parameter) | search 2/2 -5.2 against -9.0; alone +8.5 | the model is too coarse to decide by |
| HEVC coarse key mode search, VVC no angular candidates, VP9 4+2 mode search | +5.1 / +4.4 / +1.9 | the modes carry the gain |
| HEVC `intra_rd_candidates` 2 under merge-only | -5.4 @1.55 against S(0/2) -4.6 @1.46 | on the line |

### 4.7 Whole stream

`nceval.rc_oracle`, CBR, base `rate.recode_overflow` 1.0 and `rate.key_cap_buffer` 0.8, J % against
it. 150 frames: 24 clips, one key each. 600 frames with 10 s keys: two clips per class (indicative).
`rate.key_level_finer` 2 changes nothing at 150 frames (the only key is the first, which has no inter
level), so that column is the saving banked as fewer bits.

| Codec, key set | 150 f | 600 f | 600 f with `key_level_finer` 2 (alone) |
|---|---|---|---|
| HEVC S(1/2, 1.2) | -0.83 | -1.89 | -1.81 (-0.02) |
| HEVC S(0/2, 1.2) | -0.42 | | |
| VP9 S(0/2, 1.2) | -0.49 | -0.68 | -0.81 (+0.04) |
| VVC S(1/2, 1.2) / S(0/2, 1.2) | -0.23 / -0.40 | -0.85 | -1.20 (-0.14) |
| AV1 key trellis | -0.74 | -0.23 | -0.44 (+0.07) |
| AV1 key trellis + S(1/1, 1.2) | -0.87 | -0.43 | -0.45 |
| AV2 key trellis | +0.55 | +0.33, one buffer overflow | +0.30, one overflow (-0.08) |
| AV2 S(1/1, 1.2) | | -0.50 | -0.59 |
| AV2 key trellis + S(1/1, 1.2) | -0.52 | -1.04 | -1.17 |
| HEVC `intra_nxn` 0 + `intra_tu_depth` 0 | -0.15 | | |
| the same under S(1/2, 1.2) | -0.43 (the search alone -0.83) | | |
| VP9 `intra_tx_depth` 0, every frame / keys only | +0.19 / +0.22 | | |
| the keys-only cut under S(0/2, 1.2) | -0.07 (the search alone -0.49) | | |

- The gated search is worth 0.2-0.9 % of whole-stream J at 150 frames and 0.4-1.9 % with 10 s keys.
- The cuts of 4.5 are not free in a stream: about 0.4 % under the search in HEVC and VP9, whose
  first keys here are larger and finer than the one-frame keys they were free on.
- AV2's key trellis alone loses in both runs and overflowed a buffer once; with the search it gains.

### 4.8 The independent study (agent Gemini, branch `feature/gemini-intra-cheap`), folded in

| Its mechanism | Outcome |
|---|---|
| `key_rd_leaf_first` (the leaf coded before a planned split) | kept as `planner.key_rd_leaf_first`: hardware cycles of one key under search 2/2 on Zen 5 (`aws_cpu_ab`, 5 rounds): AV1 -2.24 % [-2.43, -2.05], VP9 -0.29 %, AV2 -0.22 %, VVC -0.08 %, HEVC +0.12 % (instructions +0.45 %). A gain in AV1 only |
| `key_rd_deepen_margin` / `key_rd_merge_margin` | one mechanism: `planner.key_rd_merge_margin` is the landed margin gate's value for merges (0: `key_rd_margin` for both); section 4.9 |
| `key_rd_flat_exit`, `intra_rd_flat_exit` | not taken: its HEVC search 2/2 with exit 8 / 16 / 24 is -4.75 / +0.18 / +3.30 against -8.21 (4.6 here: the same form) |
| `intra_rd_satd_margin` | the same idea as `decision.intra_rd_margin`, which is kept; the two measure alike (4 modes within 10 %: -5.18 @1.91 there, -5.2 @1.90 here) |
| AV1 modelled level rate for mode ranking (`KEY_LEVEL_RATE` for `CoeffCdfs::cost`) | its reject: +4.49 % |
| HEVC plan by source-SATD modes | its reject: +5.6 / +6.4 % |
| HEVC per-CU trial bits cached from the CTU's initial contexts | its reject: search 2/2 -6.70 against -8.21 |
| H.264 `intra8x8_gate_satd` 3 + `intra_rd_candidates` 3 (-1.45 % @1.03) | passed to agent h264-gap; covered by its landed Intra_8x8 search (292f0b725) |

Its window tables agree with 4.2 to the decimal (same harness and draw).

### 4.9 The margin gate per direction

`planner.key_rd_merge_margin` m with `planner.key_rd_margin` d (deepening), window 1/2 unless noted,
against the single value in the same run:

| Codec | per direction | single value |
|---|---|---|
| VP9 | d 1.05, m 1.5: -7.1 @1.73; d 1.1, m 1.5: -7.2 @1.88; d 1.2, m 2.0: -7.4 @2.13 | 1.2: -6.9 @2.03; 1.5: -7.5 @2.28 |
| VVC | d 1.1, m 1.5: -8.8 @1.86; d 1.2, m 2.0: -9.5 @2.03 | 1.2: -8.9 @1.89; 1.5: -9.6 @2.12 |
| HEVC | d 1.5, m 1.1: -7.2 @1.72; d 1.1, m 1.5: -6.6 @1.70 | 1.2: -7.3 @1.75; 1.5: -7.6 @1.91 |
| AV1 (key trellis on) | d 1.05, m 1.5: -7.5 @1.67; 2/2 d 1.5, m 1.1: -9.1 @2.90 | 1.2: -8.3 @2.00; 2/2 1.2: -8.9 @2.63 |
| AV2 | d 1.5, m 1.2: -5.7 @2.00 | 1.5: -5.7 @2.07 |

A wider merge margin with a narrow deepen margin is a better point in VP9 (0.2 % better at 15 %
less key CPU than the single value); elsewhere the two are the same curve. A wide deepen margin
with a narrow merge margin loses in VP9 and VVC (-5.8 and -7.9).

### 4.10 Stacks (a later build: AV1 with transform types landed, so its numbers moved by up to 0.8)

| Codec | Set | J % [interval] | CPU x |
|---|---|---|---|
| AV1 | fast's key settings + key trellis | +0.9 [-0.2, +1.8] | 0.54 |
| AV1 | the same + 7 candidates by modelled cost | -0.8 [-1.8, 0.0] | 0.60 |
| AV1 | the same + S(1/1, 1.2) | -1.2 | 0.85 |
| AV1 | medium, 1 transform-priced candidate, key trellis | -5.3 [-5.7, -4.8] | 0.99 |
| AV1 | the same + S(1/1, 1.2) / with 4 trellis states | -8.6 / -9.4 | 1.81 / 2.03 |
| AV1 | key trellis + 4 transform types by coded cost (`decision.intra_rd_transforms`) | -6.7 | 1.78 |
| AV1 | the same + S(1/1, 1.2) | -10.3 | 3.82 |
| AV1 | S(2/2, 1.5), trellis 4 states, 8 candidates, angle deltas 2 | -10.3 | 4.21 |
| AV1 | the same + 4 transform types by coded cost | -12.0 [-12.3, -11.5] | 9.51 |
| HEVC | `intra_nxn` 0, `intra_tu_depth` 0 + S(1/2, 1.2) | -7.2 | 1.48 |
| HEVC | the same + 4 / 8 modes within 10 % | -11.2 / -11.3 | 3.12 / 3.79 |
| HEVC | cuts + S(2/2, 1.5) + 8 modes within 20 % | -13.0 [-13.7, -12.1] | 6.28 |
| VP9 | S(0/2, 1.2) + `intra_rd_trellis` | -7.3 | 1.62 |
| VP9 | `intra_tx_depth` 0 + S(0/2, 1.5) + `intra_rd_trellis` + 5 candidates | -9.2 | 1.86 |
| VVC | `angular_candidates` 1 / 2 (Default 3) | -1.5 [-2.0, -0.9] / -1.1 | 0.94 / 1.00 |
| VVC | `angular_candidates` 2 + S(0/2, 1.2) / S(1/2, 1.2) / S(2/2, 1.5) | -6.6 / -9.7 / -11.1 | 1.33 / 1.89 / 2.54 |
| AV2 | `intra_chroma_candidates` 1 + key trellis + S(1/1, 1.2) / with 4 trellis states | -8.5 / -9.5 | 1.86 / 2.05 |

- **AV1 at SVT-AV1 M4's CPU is 1 % short, not level.** Section 4.4's M4 row took its CPU from
  another run; measured in one run the front passes -10.3 @4.2 and -12.0 @9.5 (about -11.0 at M4's
  6.4, where SVT-AV1 has -12.0). The type search by coded cost is what is dear (x2.3 for 1.7 %).
- AV1's fast key with the trellis and seven modelled candidates is 2.7 % better than SVT-AV1 M8
  at 15 % more CPU (0.60 against 0.52); at M8's own CPU it is level (+0.9 against +1.9).
- VVC's third angular candidate costs J on these keys: one fewer is 1.1-1.5 % better and no slower.
  Not checked in a stream.
- HEVC's and VP9's cuts stack without loss on one frame; in a stream they cost 0.4 % (4.7).

## 5. Ready list for 0.6.1 (nothing here changes a preset; one-frame figures unless marked)

| Codec | Fast rungs | Medium | Slow | Slower |
|---|---|---|---|---|
| AV1 | key trellis needs a key-only scope first: `quant.trellis` 0 switches it off (fast keys +9.2 without, +0.9 with) | `quant.trellis_key` 1 (-5.0 @1.03; stream -0.74 / -0.23) | + S(1/1, 1.2), `key_rd_leaf_first` (-8.3 @1.9; stream -0.87 / -0.43) | + S(2/2, 1.5) with slower's key tools (-10.3 @4.2) |
| AV2 | | S(1/1, 1.2) with `quant.trellis_key` 1, never the trellis alone (-7.6 @1.9; stream -0.52 / -1.04) | the same | S(2/2, 1.5) + trellis (-9.1 @3.2) |
| HEVC | | S(0/2, 1.2) (-4.5 @1.33; stream -0.42) or S(1/1, 1.1) (-5.3 @1.41) | S(1/2, 1.2) (-7.3 @1.74; stream -0.83 / -1.89) | S(2/2, 1.5) + 8 modes within 10-20 % (-11.8 to -13.0 @4.4-6.3) |
| VP9 | | S(0/2, 1.2) (-6.5 @1.38; stream -0.49 / -0.68) | + `vp9.intra_rd_trellis` 1 (-7.3 to -7.7 @1.6) or S(1/2) deepen 1.05, merge 1.5 (-7.1 @1.73) | as slow |
| VVC | | S(0/2, 1.2) (-5.3 @1.35; stream -0.40) | S(1/2, 1.2) (-8.9 @1.86; stream -0.23 / -0.85) | S(2/2, 1.5) (-10.4 @2.5) |

Stream figures: 150 f / 600 f with 10 s keys (4.7). To check before any of it lands: the pool's
rotating draw (`nceval.quick` → train), the key's share of encode time per rung on a timing box, and
for VVC `angular_candidates` 2 in a stream.

## 6. Open

- The staged decision's stage 1 as one engine driver with per-superblock rate tables (section 3):
  not built. It is the lever for the pricing walk's 10-39 % of every encode (section 2).
- A key-only trellis scope for the rungs that pin `quant.trellis` 0: `TrellisSettings` is read as
  "the trellis is configured" (`is_some`) in every backend's forward policy and trimming, so the
  scope needs each of those sites audited.
- AV1 above M4: 4x4 blocks and the cost of the type search (agent av1-key-tools has types and H/V
  shapes landed, default off).
- VP9, VVC and AV2 have no same-standard yardstick on these axes.

## 7. The staged decision, built (agent staged-md, 2026-11-22)

Branch `staged-md`. Everything below lands default off with identical streams
(`preset_identity`, all codecs and presets); forced on, landing conformance passed 606/606
(AV1, HEVC, VVC, VP9, AV2, every preset).

### 7.1 How many candidates are priced exactly, and where the winner ranks

Probe build (branch `staged-md-probe`, never landed), medium, 4 quick clips x 2 quantizers,
31 frames (1 key + 30 P). "Exact" is each codec's luma coefficient-domain cost with the exact
context walk (AV1 `residual_rd_cost`, HEVC `luma_transform_cost`, + λ · mode bits: SVT's MDS1);
stage 0 is prediction SATD + √λ · mode bits. Loss: exact cost lost by keeping only the SATD
top N, as a share of the exact winners' total.

| List | per leaf | winner SATD rank 1 / top 2 / top 3 / top 6 | loss top 1 / 2 / 3 / 4 / 6 | 20 % band: loss / kept |
|---|---|---|---|---|
| AV1 key intra (medium prices all) | 10 | 49 / 64 / 73 / 92 % | 3.70 / 2.10 / 1.41 / 0.80 / 0.23 % | 0.44 % / 4.7 |
| HEVC key intra (8 forced; medium prices 1) | 8 | 50 / 67 / 77 / 94 % | 2.74 / 1.37 / 0.86 / 0.53 / 0.16 % | 0.35 % / 4.1 |
| AV1 P intra (7 forced; medium prices 2) | 7 | 57 / 76 / 89 / 100 % | 3.71 / 1.34 / 0.40 / 0.05 / 0 % | 0.72 % / 3.3 |
| HEVC P intra (8 forced; medium SATD rank 1) | 8 | 55 / 73 / 81 / 95 % | 2.33 / 1.16 / 0.70 / 0.41 / 0.12 % | 0.13 % / 4.9 |
| AV1 P inter final (winner, skip, intra; forced `final_stage`) | 2.3 | 82 / 98 / 100 % | 1.60 / 0.16 / 0 % | 0.77 % / 1.3 |
| HEVC P inter final (same) | 2.3 | 82 / 97 / 100 % | 2.08 / 0.50 / 0 % | 1.20 % / 1.3 |

In medium P frames no codec prices ten candidates per leaf: AV1 prices 2 of 7 intra modes and
codes one inter candidate; HEVC takes the SATD-best intra mode. The ten-candidate exact pricing
is on keys.

### 7.2 Where the walk is: key against P (callgrind, crowd 720p, medium, 1 frame against 9)

AV1: the key is 2095 M instructions, a P frame 99 M (a key = 21 P frames), so in a 31-frame run
the one key is about 41 % of the instructions: section 2's "15.7 % of the whole encode" is
mostly the key. P frames alone: `tree_transform_cost` (the transform-split trials) 6.2 %,
`code_plane` 19 % (the transform RD check inside it), `trellis_quantize` 8.5 %,
`CoeffCdfs::cost` 4-5 %. Key: `rd_best_luma` 38 % (MDS1 on every candidate, again inside the
key RD search's trials), `key_intra_shortlist` 24 % (its SATD pool), the key planner 21 %.
HEVC: key 1621 M, P 128 M; in P frames the pricing walk (`write_residual::<BitEstimator>`,
`residual_bits`) is about 2 %, the P cost is the inter luma tree's coding and the inter ranking.

### 7.3 The shared intra cut (`decision.intra_stage_*`, `decision.inter_intra_stage_*`)

Every intra candidate ranked by SATD + √λ · mode bits; the first `keep`, within `band` per
mille of the best, go on to the codec's transform-costed estimate (key frames and the intra
blocks of inter frames take separate cuts). One frame (key-share 0.15, quick draw
`fixedqp-2026-11-21`, n=24), J % against medium @ key CPU x:

| Codec | keep 1 / 2 / 3 / 4 | band 10 / 20 / 30 % |
|---|---|---|
| AV1 (medium prices all ten) | +5.0 @0.76 / +2.2 @0.83 / +1.8 @0.86 / +1.3 @0.90 | +0.8 @0.91 / +0.2 @0.97 / -0.1 @1.01 |
| HEVC (medium prices the SATD best) | - / -0.3 @1.19 / -1.1 @1.25 / -0.8 @1.32 | -1.9 @1.34 / -2.1 @1.51 / - |
| VVC | +3.0 @0.90 / +1.7 @0.93 / +1.2 @0.96 / 0 (its default) | -0.5 @1.00 / -1.0 @1.02 / -1.3 @1.04 |
| AV2 | +2.2 @0.92 / 0 (default) / -0.6 @1.08 / -1.2 @1.17 | -1.0 @1.27 / -1.5 @1.49 / -1.5 @1.65 |
| VP9 | +2.4 @0.71 / +0.5 @0.92 / 0 (default) / -0.2 @1.12 | 0.0 @1.08 / -0.4 @1.27 / -0.4 @1.44 |

AV1's cut trades on the same line as `av1.keyframe_intra_rd_candidates` (its SATD stage
predicts every candidate again, and its key SATD pool is already 24 % of the key). Streams
(150 f, key cut only): VVC band 20 / 30 % +0.2 / +0.3 (worse, despite the one-frame gain),
HEVC band 10 % -0.2 [-0.3, +0.1], AV2 keep 4 -0.4 [-0.7, -0.1], band 20 % -0.5 [-0.7, -0.3];
the AV2 inter-frame cut changes nothing. Whole-encode cycles (aws_cpu_ab, CBR 600, 150 f,
graviton3 / graviton4 / zen5): AV2 keep 4 +1.9 / +1.7 / +2.3 %, AV2 band 20 % +4.4 / +3.9 /
+5.1 %, HEVC band 10 % +1.8 / +1.9 / +1.6 %. None is a free point; no preset change.

### 7.4 The transform-split gate (`gate.tx_split_min_nonzero`, AV1)

The tx-split trial prices its four quarters only where the whole transform kept at least N
levels (1, the default, is the old exit). Fixed QP 150 f, n=24, N=9: medium +0.1 [-0.0, +0.1],
slow +0.0, slower +0.1 (N=17: +0.3 to +0.4). Cycles N=9: medium -1.04 / -1.14 / -1.15 %
(graviton3 / graviton4 / zen5), slow -1.39 / -1.44 / -1.82 %, slower -0.6 to -0.8 %. Quick
(k 0.1): medium L1 -0.29 [-0.41, -0.11], probe +0.23; slow L1 -0.21, probe +0.95 (static
class +4.79). Train (draw 2026-11-22): medium L3 +0.42 [+0.04, +0.92] WORSE, final check +0.38
[+0.18, +0.58] WORSE; slow L3 +0.15 AMBIGUOUS, final check +0.19 AMBIGUOUS. Ejected: the
skipped splits matter over long windows; the default stays 1.
