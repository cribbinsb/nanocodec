# RD mode decision audit (2026-11-19)

Where each codec and preset decides by rate-distortion cost, by a cheap metric, or by a fixed
choice; what each RD path costs; and the candidates for adding RD cheaply or making it cheaper.
Branch `rd-audit`, scratch and logs in `~/nanocodec-data/work/rd-audit/`. Background:
[rd-mode-decision.md](../handoff/rd-mode-decision.md), [rd-depth.md](../handoff/rd-depth.md),
[hevc.md](../handoff/hevc.md) (VC-1K sections), [vp9-av2.md](../handoff/vp9-av2.md) ("AV2 on the
Pareto frontier").

## How the costs were measured

Callgrind instructions on the build box, gate build of `cd145e8b0`, Johnny and WestWindEasy 720p,
each codec's Default, QP 36 (qindex 160 for AV1, VP9 and AV2). "Key" is a one-frame encode, "P" the
mean of the next 8 frames (`probe2.sh`, `probes.txt`). Cells read Johnny / WestWindEasy; "+x %" is the
cost of the RD path against the same encode with it off (or reduced as stated).

| codec | key (M instr) | P (M instr per frame) |
|---|---:|---:|
| H.264 | 189 / 389 | 52 / 167 |
| HEVC | 893 / 1232 | 69 / 222 |
| VVC | 2468 / 3330 | 96 / 317 |
| AV1 | 1460 / 1979 | 56 / 255 |
| VP9 | 1092 / 1355 | 70 / 256 |
| AV2 | 892 / 1355 | 43 / 213 |

Rungs: VF Veryfast, Fr Faster, Fa Fast, D Default, S Slow. HEVC has no VF, VVC no VF or Fr, AV2 no
VF. Values from `nanocodec params --codec C` and `crates/nc-encoder/presets/`.

## Inventory

### H.264

- Key intra: Intra_16x16 luma and chroma modes by SATD on every rung. Intra_4x4 / Intra_8x8 only
  above fixed SATD gates (`h264.intra4x4_gate_satd` 24.3, 64 on VF/Fr; not λ-scaled:
  `h264.intra_nxn_gate_lambda` 0). D and S: `h264.intra_rd` 2 codes Intra_4x4 by RD over the
  5 SATD-best modes (9 on S) and picks 16x16 against NxN by full cost: key +30 % / +70 %, P +0 / +7 %.
  Fa, Fr, VF: SATD only.
- Inter vs intra and skip: engine ranking (mean-removed variance + λ·bits), DC intra test, intra
  gate variance 16 (25 VF/Fr, 50 Fa). D/S: the RD final stage (winner, skip vector, intra by
  coded luma cost + CABAC bits within 0.03 λ, 0.05 on S): P +8.0 / +9.6 %. Pinned off on Fa/Fr/VF.
- Sub-sample: winner walk (exact error + λ·bits) D/S only; `h264.rd_refine` (coded-cost walk) S
  only.
- Partition: 16x8/8x16/8x8 searched above a SATD gate (λ gate 0.8 on D/S); VF none.
- Transform size: heuristic; RD (`h264.transform_size_rd`) S only. Residual RD
  (`h264.residual_rd_lambda`) S only; JM-style coefficient elimination on all rungs.
- Trellis D/S: P +0.4 / +7.1 %. Deblocking fixed, no search.

### HEVC

- Key intra mode: SATD + √λ·bits only, a coarse-to-fine angular sweep measured on the first
  transform block of large blocks (`backend/intra.rs`); no RD between modes on any rung (full
  sweep vs coarse: key +3.8 / +4.5 %). Chroma is always the derived mode: no chroma search.
- Key transform tree and PART_NxN by RD (D/S; `hevc.intra_tu_depth` 2, `hevc.intra_nxn`): key
  +34 / +48 %. Key CU sizes planned by transform cost (`hevc.intra_leaf_cost`, D/S).
- P-frame intra: coarse set (planar, DC, H, V, MPMs) by SATD; angular sweep only on S.
- Inter transform tree: RD trials on luma, gated by residual spread (D/S/Fa): P +0.6 / +5.0 %.
- Whole-CU pricing (all planes, real coder; intra leaves within 0.1 λ): D only, P +0.3 / +7.6 %.
  Slow pins it 0 ("not screened").
- Merge candidates: every one ranked with the exact filter, no RD. Intra gate variance 0.
- SAO: RD, type search every 4th picture.

### VVC

- Key intra: SATD shortlist, full RD on 4 directions (2 on Fa): 4 vs 1 costs key +11 / +14 %.
  ISP by RD: key +29 / +26 %. Chroma direction search by RD on keys: +10 / +8 %.
- P-frame intra: coarse set, full RD on the 2 best (`vvc.coarse_rd_candidates`); Fa DC only.
- Per-unit residual keep/drop RD (`vvc.unit_zero_rd` 0.5): P +0.5 / +1.2 %. Pairwise merge
  candidates confirmed by luma RD.
- ALF (exact decision) and SAO by RD. Off: MTS, MTT and `planner.rect_rd`, dependent quantisation.
  `inter_leaf_cost` exists but no shipped preset uses it.
- Slow keeps `decision.exact_rank_margin` 1.1 while D is 0 ("not screened").

### AV1

- Key intra: D/S full RD with a coefficient walk on 5 base modes (DC, smooth, Paeth, V, H) plus the
  3 best of a SATD shortlist (directions, deltas, filter intra), CfL search. Against the Fast
  recipe (DC-prediction gate 150 with size scale 3, SATD, model re-rank of 7, top 1): **key +122 /
  +109 %** (top 2: +112 / +89 %). Fr/Fa use that recipe; VF DC only (temporary, see cheap-intra).
- P-frame intra: D SATD then 2 transform-costed candidates; faster rungs DC only (D: P +0.1 / +3.5 %).
- Inter transform split: RD trial gated by energy 0.25 λ (D/S): P +5.1 / +16.7 %.
- Trellis D/S: P +5.0 / +19.5 %. Skip: SVT light-PD1 pre-transform skip and skip bias; VF/Fr/Fa
  also reject luma before the inverse with SVT's end-of-block rate.
- CDEF searched per superblock on sampled rows; deblocking level from Q; restoration off.
- `decision.intra_test_dc` 0 (every other codec 1).

### VP9

- Key intra (Fa/D/S): all ten modes by SATD + √λ·bits, **no RD**; against DC/V/H: key +15 / +17 %.
  VF/Fr: DC/V/H by SSE on the first transform block, chroma DC.
- P-frame intra: DC only on every rung, except Fast's non-base frames, which search
  (`nonbase.vp9.inter_intra_dc_only` 0): a faster rung doing more than D.
- Inter transform size: RD trials with exact token bits (D/S/Fa): **P +4.2 / +36.9 %**. Faster
  prices them at 4 bits per level (libvpx `block_yrd`); on D that saves P 0 / 4 %.
- Residual drop RD on every rung. Switchable interpolation filter. Loop filter level from Q.

### AV2

- Key intra: SATD + bits, full RD on 2 modes (+10 / +11 % key over 1), fitted coefficient-rate model
  on D/Fa/Fr, exact on S. Transform split by RD: key +11 / +12 %, P 0 / +6.5 %.
- P-frame intra: DC only (`av2.p_intra_modes` 0).
- Coefficient trimming RD; TCQ on S; CDEF full search on D, probe on Fa/Fr.
- Slow keeps `decision.exact_rank_margin` 1.1 while D is 0 ("not screened").

## Inconsistencies

- One pattern, two engine stages: H.264 uses the luma-only final stage (`final_stage.rs`), HEVC the
  all-plane whole-CU pricing (`cu_pricing.rs`); AV1, VP9, AV2 and VVC have neither.
- Five coefficient-rate estimates for one job: AV2's fitted level model (codec-local), the engine
  `LevelRateModel` (AV1 keys only), VP9's bits per level, AV1's SVT end-of-block estimates, H.264's
  exact CABAC.
- Key mode metric: SATD only in HEVC and VP9 (and H.264's 16x16); full RD in VVC, AV1 Default and AV2.
- P-frame intra gate: absolute variance 16 / 0 / 64 / 64 / 6.9 / 64 (H.264 / HEVC / VVC / AV1 /
  VP9 / AV2); AV1 alone has `intra_test_dc` 0.
- The "minimum nonzero levels before a split trial" gate exists in VP9, AV2 and VVC's MTS, not in
  HEVC or AV1.
- Slower rung doing less: HEVC Slow without whole-CU pricing; VVC and AV2 Slow with a coarser
  exact-rank margin than D; VP9 Fast searching P intra where D does not.
- Forced or DC-only choices left: VP9 and AV2 P-frame intra, HEVC chroma intra, VP9 VF/Fr keys
  (DC/V/H), AV1 VF keys (temporary).

## Candidates (ranked by expected J per CPU)

Parameters only:
1. Slow inversions: HEVC Slow `decision.cu_price_intra_lambdas` 0.1 (after the HEVC conformance
   fix lands); VVC and AV2 Slow `decision.exact_rank_margin` 0.
2. AV1 Default key frames on the Fast recipe at top 2: key -53 / -47 % instructions, about 4 % of a
   Johnny encode with 10 s keys. Offline, model top 1 left 25 % of DC-only's regret. Risk: key
   quality and the periodic-key pulse.
3. AV1 Default `av1.inter_tx_split_energy` 1 (held in the ledger: Jetson -2.3 %, L1 J -0.03 %).

Add RD where it is missing:
4. VP9 key frames: SATD, then a `LevelRateModel` re-rank with VP9 weights.
5. HEVC key frames: the same (HEVC all-intra is already level with x265, so a smaller gain).
6. VP9 P-frame intra with the core set, the prediction gate and the model (a plain ranking was
   rejected on 2026-11-19).

Make existing RD cheaper:
7. VP9 transform-size trials priced by the shared model instead of the token walk.
8. VVC key: model re-rank before the 4 full-RD directions, exact RD on 2.
9. A shared minimum-levels gate for HEVC's inter transform-split trials.

Structural: a final stage or CU pricing for AV1, VP9 and AV2 on one engine mechanism; one shared
coefficient-rate model.

In flight elsewhere: `cheap-intra` (HEVC Fast and VVC Fast P intra, AV2 keys, AV1 reuse rule),
`chroma-cost` (HEVC chroma trellis), `hevc-vvc-held` (HEVC TU16, deblocking).

## Tried

All fixed-QP rows: `nceval.fixedqp ab --clipset` on the codec's VC-1K worst six
(`work/vc1k-harvest/vc1k-worst6-*.json`, H.264 `work/rd-depth/vc1k-worst6.json`), matched QP, J %
against the same build's Default (or the named preset); they rank ideas only. All-intra rows: 6
frames, every frame a key, five public 720p clips, four quantizers, PSNR-YUV (6:1:1) BD-rate
(`work/rd-audit/intra.py`), with callgrind instructions of one key frame (Johnny / WestWindEasy).

| candidate | result | outcome |
|---|---|---|
| AV1 Default keys on the Fast recipe, exact RD on 1 / 2 | key instr -55 / -53 %; J +0.7 [+0.1, +1.4] / +0.8 [+0.2, +1.4] | rejected |
| AV1 Default `av1.inter_tx_split_energy` 1 | J +0.3 [-0.1, +0.8] | held |
| VVC Slow `decision.exact_rank_margin` 0 | J +0.3 [+0.1, +0.5] | rejected |
| AV2 Slow `decision.exact_rank_margin` 0 | J -0.2 [-0.4, -0.0]; pool k 0: L1 +0.11 %, L2 -0.21 %, L3 +0.01 % [-0.14, +0.16] AMBIGUOUS at CBR instr +0.5 / +0.8 % | rejected |
| HEVC Slow `decision.cu_price_intra_lambdas` 0.1 | J -1.8 [-2.2, -1.5] (VMAF -3.4); pool k 0: L1 -1.26 %, L2 -1.20 %, L3 -0.64 % [-0.83, -0.45]; final-check -0.92 % [-1.21, -0.63], no guard; CBR instr +0.6 / +18.0 % | **adopted** (cd589e129) |
| VP9 Default `vp9.inter_tx_depth` 1 | J -0.2 [-0.8, +0.3]; P instr -1.6 / -9.3 %; pool k 0.1, dCPU -3.2: L3 +0.49 % [+0.32, +0.65] WORSE | rejected |
| VP9 Default `vp9.inter_tx_split_spread` 1.5 / 2 | J +0.6 / +0.9; P instr -10.5 / -20.8 % (WWE) | not pursued |

**VP9 key intra modes by coded cost** (`vp9.intra_rd_candidates`, `vp9.intra_rd_trellis`, landed
off in 17f033419). The SATD-best k luma modes (each with its best chroma mode) are coded at the
largest transform in all planes and the cheapest by SSE + λ · (token + mode bits) kept.

| variant | all-intra BD | key instr |
|---|---:|---:|
| k 2 | -2.58 % | +43 / +47 % |
| k 3 | -3.49 % | +68 / +75 % |
| k 4 | -3.99 % | +94 / +104 % |
| k 2, trials without trellis | -2.23 % | +27 / +26 % |
| **k 3, trials without trellis** | **-3.00 %** | +43 / +43 % |
| k 4, trials without trellis | -3.41 % | +60 / +60 % |
| k 2 / k 4, x264-style SATD margin 1.1 | -2.13 / -2.86 % | +31 / +59 % |
| k 3, RD only on blocks up to 16x16 | -3.00 % | +47 / +70 % |
| k 2, fitted `LevelRateModel` instead of the token walk (VAL error 7-9 %) | -1.61 % | +24 / +23 % |

The trellis inside the trials was the cost to cut (callgrind: trellis about 18 % of a k 2 key,
the token walk 10 %); the level-rate model lost a third of the gain for 2 % of the key, the
margin and size gates bought nothing, and all three were removed. CBR 600 kbps, 150 frames: k 3
without trellis +1.65 / +1.94 % instructions.

**VP9 Default adopted (52d08f15e):** 3 candidates, trials without trellis. Pool CBR (k 0.1, dCPU
1.8, draw 2026-11-19): L1 J -0.56 % [-0.85, -0.26], L2 -0.35 % [-0.64, -0.06], L3 -0.44 % [-0.72,
-0.17] n 108 BETTER; final-check -0.70 % [-0.99, -0.42] n 36 BETTER, no guard (dKey1 +0.10 dB);
static_recovery 13/13; preset_identity 297/306, the nine VP9 Default only. Other VP9 presets pin 1.

Carried to the other codecs and presets (CLAUDE.md, every change considered everywhere):

| where | result | outcome |
|---|---|---|
| VP9 Slow, k 3 without trellis | CBR 150 f instr +3.3 / +2.8 %; pool k 0: L1 -0.57 %, L2 -0.52 %, L3 -0.68 % [-0.91, -0.45]; final-check -0.60 % [-0.80, -0.38], no guard | **adopted** (91326d11c) |
| VP9 Fast, k 2 without trellis | CBR 150 f instr +2.0 / +19.5 % | not viable at k 0.6 |
| HEVC key (`hevc.intra_rd_candidates`) | below; CBR 150 f instr k 2 +1.1 / +9.4 %, k 4 +2.6 / +18.1 % | k 2 pool L1 (k 0.1, dCPU 5.2) +0.50 % [+0.22, +0.79]: rejected |
| H.264 `Intra_16x16` mode by RD (prototype, removed) | all-intra k 2 -0.64 % at +48 / +23 % key instr, k 3 -0.85 % at +71 / +37 % | rejected: a whole macroblock coding per trial |
| VVC key | trials are already plainly quantised (no trellis to drop); `vvc.isp_residual_gate` 1 +0.01 % all-intra for key -3.2 / -0.8 %, `vvc.chroma_rd_candidates` 2 +0.12 % for -3.1 / -2.2 %, `vvc.rd_candidates` 3 +0.70 % for -3.0 / -3.6 % | ISP gate: small bundle item |
| AV1 Default, AV2 keys | already transform-costed RD with plain quantisation (`quant.trellis_key` 0) | n/a |

**HEVC key intra mode by transform cost** (`hevc.intra_rd_candidates`, landed off in 1aefcad9f): the
SATD-best k modes priced on their first transform block by the key planner's
`luma_transform_cost` (plain quantisation, CABAC level bits, coefficient-domain error). k 2 -0.63 %
at +13 / +12 % key instructions, k 4 -1.25 % at +26 / +22 %, k 8 -1.80 % at +51 / +44 %: HEVC's keys
already decide transform trees and NxN by RD, so less is left than in VP9.

- H.264 `decision.final_price_min_lambdas` (close candidates below x λ per sample keep their screen
  cost): VC-1K worst six fixed QP J +8.2 [+6.5, +9.8] at 0.05, +11.2 at 0.1, +8.1 at 0.25, for
  whole-encode instructions -0.5 to -2.6 %. Rejected: the stage's gain is on near-uncoded blocks.

## HEVC Fast smooth P-frame intra made cheaper (owner priority, 2026-11-19)

`hevc.inter_intra_smooth` (planar or DC on HEVC Fast's light superblocks, by SATD + √λ · bits,
derived chroma, no trials, unconditional inter fallback; landed off in b51ddbb40) measured worst six
J -1.8 [-2.4, -1.3] but pool L2 +0.83 % WORSE at k 0.6. Callgrind (HEVC Fast, CBR 600, WestWindEasy
60 f) of the lever's +6.1 %: the full mode search's scalar transpose of the source block (78 M of 260 M,
never needed for planar or DC), a second reference gather and prediction when the CU was coded (37 M
each), two SATDs (20 M), filtered references, copies.

Landed exact (5bd134be0, identical streams): the choice gathers once, predicts both modes in coding
orientation and hands the winner's prediction to the block's coding; `choose_intra_mode` transposes
the source only on its first horizontal mode. Lever cost, CBR 600, 120 f, keys every 60 (WestWindEasy /
ControlledBurn / Johnny): +7.06 / +5.39 / -0.15 % (mean +4.10) -> +3.22 / +2.79 / -0.26 % (+1.92).

Cheaper decisions, worst six fixed QP J and the same cost:

| variant | J | cost (mean) |
|---|---:|---:|
| SATD (exact version) | -1.8 [-2.4, -1.3] | +1.92 % |
| SAD instead of SATD | -1.7 [-2.0, -1.3] | +1.79 % |
| planar only below 0.8 x DC's SATD | -0.8 [-1.1, -0.5] | +1.09 % |
| planar only below 0.7 x DC's SATD | -0.4 [-0.5, -0.3] | +0.97 % |

What is left is the coding of planar CUs (luma planar, derived-mode chroma planar, filtered
references): it scales with the gain, so a margin trades them about evenly.

Pool screen of the exact version (k 0.6, dCPU 1.9, draw 2026-11-19): L1 J +0.03 % [-0.34, +0.38],
L2 +0.56 % [+0.28, +0.81] (quality about -0.6 %); stopped there. At k 0.6 that quality pays for
about +1.0 % instructions, half the cost left. Not passing; the lever stays off. VVC Fast, whose
planar-or-DC gained J -0.3 [-0.6, +0.1] on its worst six with the same planar coding cost, was not
ported.
