# Tools that hurt at fixed QP (2026-11-20)

**Question (owner).** When each codec's Slow was re-picked "all heavy settings on, cut to the
knee" (k 0, quality only; `docs/handoff/presets.md`), several standard coding tools measured
WORSE at fixed QP. In a reference encoder a standard tool never loses quality with CPU free,
so each loss points at a bug or a wrong cost/decision in our implementation. Diagnose each,
biggest first; fix the clear ones.

**Method.** Each tool was first reproduced on 2-4 clips at fixed QP with
`~/nanocodec-data/work/tools-hurt/probe.py`: direct encodes of raw clips (`quick/` set:
crowd_mot17_09, pan_jaad_0117, lowlight_pp22_004, static_chirla_s000c3, bodycam_airport; 30-40
frames), four quantizers, per-plane BD-rate (PSNR-Y/U/V and PSNR-YUV 6:1:1) against the
codec's Default from `--sse-log` (no YUV written). Then the cause was located in the code and
against the reference encoder, and fixes were measured with `nceval.fixedqp ab` (draw
fixedqp-2026-11-20, 22-24 clips, J = PSNR-YUV / VMAF v1 / MS-SSIM thirds). Head 80dff233b.
Note for reading J: VMAF v1 (`vmaf_v1.0.16_3d0h`) has a `speed_chroma_uv` feature, so a chroma
loss moves VMAF as well as PSNR-YUV.

## Summary

| tool (codec) | Slow re-pick J | reproduced (probe) | cause | fix | result |
|---|---:|---|---|---|---|
| `decision.final_stage` (VP9, AV1; also HEVC, VVC, AV2) | VP9 +5.6 %, AV1 +2.4 % | VP9 PSNR-YUV BD +9.8 / +4.2 %, AV1 +3.6 / +0.7 % | **bug (shared engine)**: without a codec pricing the stage keeps the ranking's winner, and it replaces the DC intra test, so inter frames lose intra and the skip vector | landed 8a9e12f4b: the stage runs only where `CodecBackend::prices_final_candidates` | the tool is now inert (identical streams) outside H.264; the gain needs pricing hooks (below) |
| H.264 split search everywhere | +1.3 % (Default), +1.9 % (Slow, this run) | lowlight +6 % rate at matched QP | decision on a cheap proxy: 16x16 vs 16x8/8x16/8x8 by SATD + λ·vector bits against the 16x16 predictor; on noisy video split vectors fit noise | none needed in a preset: with the coded-cost decision (`h264.rd_refine` 2) split everywhere is neutral | Slow pool: split +1.9 % [+1.3, +2.5]; rd 2 -0.9 % [-1.3, -0.5]; rd 2 + split -0.9 % [-1.5, -0.4] |
| HEVC split spread + five merges + TU16 chroma | +0.6 % | not reproduced on PSNR: YUV BD -0.6 / -1.7 / -1.0 % | within noise; the pool loss was VMAF (+0.4) | none | - |

| AV1 chroma trellis | +2.5 % | chroma BD +14..+41 % | no bug: chroma λ weighting as HEVC/VP9/H.264; a chroma-for-rate trade J (VMAF v1 reads chroma) rejects | keep off | - |
| AV1 8x8 inter blocks | +2.4 % | YUV BD up to +3.9 % | planner overrates 16->8 splits (quarters priced at own vectors, coded at predicted ones; transform blind) | proposal: coded-cost split confirmation (`inter_leaf_cost` for AV1) | not done |
| AV1 `residual_rd_max_mv` 64 | +0.9 % | luma worse on 4 of 4 | greedy per-block residual drop, blind to reference use | none short of reference-aware λ | - |
| VVC dependent quantisation | +6.6 % | +14..23 % bits / +0.3..0.9 dB at matched QP | levels fine; block and mode choices around DQ worse (planner, 8x8) | expected ≈ 0 even when fixed (VVenC ablation) | not fixed |
| VVC multi-type tree | +4.6 % | luma +4 %, chroma +12..14 % | variance-model shapes blind to transform; chroma of small/rect CUs | `rect_rd` 1: +1.2 %; real fix is full-RD splits | not fixed |
| VVC affine beside second reference | refused | - | deliberate: affine candidate lists know one reference | none (VVenC: +0.7 % at our point) | - |
| AV2 every transform type and split | +5.4 % | pool +5.6 % | **settings, not tools**: `tx_split` 2 forces the split, `intra_modes` 4 is narrower than the Default's 3 | landed aaf73462d (AV2 Slow: split gate off + P intra widened) | final-check -1.08 % [-1.28, -0.75] vs old Slow |
| VP9 1/8-sample vectors | +1.6 % | pool +1.7 % [+0.6, +3.1] | **bug**: odd NEARESTMV/NEARMV truncated to quarter samples, then priced and coded as NEWMV | landed 95874a490 | bodycam +7.2 -> +0.7 %; lowlight still +4.9 % |

## 1. The shared final decision stage (VP9 +5.6 %, AV1 +2.4 %)

**Reproduced.** Default plus `decision.final_stage` 1, 40 frames, qindex 80-200, PSNR-YUV BD:
VP9 crowd +9.8 %, pan +4.2 %, lowlight +1.4 %; AV1 crowd +3.6 %, pan +0.7 %, lowlight +0.7 %,
at no extra work (the Slow re-pick measured 0 % and +1 % instructions).

**Cause (a real bug in the shared engine).** `final_stage.rs` decides between the ranking's
winner, the codec's skip vector and intra by a screen, then prices the close ones with
`CodecBackend::final_coded_cost`. Only H.264 implements that pricing (HEVC has the intra screen
alone). Without it `price_close` returns the winner. And the stage *replaces* the ranking's DC
intra test (`decision.rs`, `intra_or_inter`): so with the stage on, a codec without a pricing
never chose intra in an inter frame, nor the skip vector over the winner. Proof: VP9 and AV1
with the stage on code byte-identical streams to the stage on with intra disabled
(`decision.intra_gate_variance` 1024, `intra_directional` 0, `intra_test_dc` 0), and close to
the stage off with intra disabled (VP9 crowd +7.2 %, AV1 +2.9 %). This is also what sank the
HEVC stage (ledger `hevc-rd-final-stage-hooks-skip-vector-`, VC-1K +2.9 %): HEVC screens intra
but cannot price it.

**Fix (landed 8a9e12f4b).** `CodecBackend::prices_final_candidates` (default false; H.264: CABAC
and no random decisions, the same condition its pricing had) gates the stage in `decision.rs`;
a codec without a pricing keeps the ranking's decision as with the stage off. No shipped
preset sets the stage outside H.264, so `preset_identity` is 306/306 identical; VP9 and AV1 with
the stage on now code the same streams as with it off.

**Independent confirmation.** The silent no-op audit
(`docs/analysis/silent-noop-audit-2026-11-20.md`, ceb49fae4, item (c) 1) found the same: outside
H.264 the stage lacks `final_coded_cost`, `final_intra_screen` (VVC, AV1, VP9, AV2) and
`skip_vector` (HEVC, AV1, VP9, AV2), so five of its six sub-parameters are no-ops and it only
drops the DC intra test. The VP9 +5.6 % and AV1 +2.4 % results measured that hollow stage, not
the RD stage. Wiring the stage for those codecs is another agent's task; since 8a9e12f4b it
stays off until a codec's `prices_final_candidates` returns true, which that work sets with
its pricing.

**What the gain needs (proposal, for that agent).** The H.264 stage bought fixed QP J -1.6 % at +15 %
instructions on its Default (h264-spacing). For the other codecs:
- VVC already has the inter half: `inter_leaf_cost` is exactly the stage's coded luma cost
  (luma SSE after transform/quant/recon + λ · coefficient and mode bits). It lacks an intra
  pricing from the reconstruction. The engine should let a codec price the inter finalists
  only and keep the DC intra test for intra (run `intra_or_inter` on the stage's inter
  choice when `final_intra_screen` is `None`); VVC then plugs in with a few lines.
- AV1, VP9, AV2, HEVC need `final_coded_cost` (and AV1/VP9/AV2 an intra screen): AV1's
  `residual_rd_cost` and HEVC's whole-CU trial coder are the pieces. Expected gain: of the
  order of H.264's (-1 to -2 % J where the stage's flat/dark content is), not measured.

## 2. H.264 split search on every macroblock (+1.3 % Default, +1.9 % Slow)

**Reproduced.** Default, `h264.split_gate_lambda` 0 + `split_gate_satd` 0 +
`split_refine_mv16` 1: PSNR-YUV BD crowd -1.3 %, pan -2.2 %, lowlight +0.7 % (rate +5.7 % at
matched QP for +0.2 dB Y, chroma +4 %). On the pool (Slow, fixedqp-2026-11-20, 22 clips)
J +1.9 % [+1.3, +2.5] (PSNR +1.3, VMAF +1.9, MS-SSIM +2.0).

**Cause (decision on a cheap proxy).** `nc-h264/src/partition.rs::choose_partition` picks
16x16 / 16x8 / 8x16 / 8x8 by prediction SATD + λ_SATD · (shape bits + vector bits), every
quadrant's vector priced against the 16x16 predictor. It never sees the residual's coding.
Behind the gate only blocks with large 16x16 SATD reach it; with the gate open, every
macroblock does, and on noisy and dark video four vectors fit noise: the SATD drops, the coded
residual does not. With the coded-cost decision between the best split and 16x16
(`h264.rd_refine` 2, the x264 `--subme 7+` approach) the same open search is neutral or better:

| Slow + | probe PSNR-YUV BD crowd / pan / lowlight / bodycam | pool J (22 clips) |
|---|---|---:|
| split everywhere | -1.0 / -1.7 / +0.8 / -0.4 % | +1.9 % [+1.3, +2.5] |
| `rd_refine` 2 | -3.2 / -2.4 / -1.7 / -1.5 % | -0.9 % [-1.3, -0.5] |
| `rd_refine` 2 + split everywhere | -4.2 / -3.4 / -2.8 / -1.6 % | -0.9 % [-1.5, -0.4] |

**Fix.** No code bug; the open split search is only valid behind the RD decision. Not adopted in
Slow: `rd_refine` costs +41 % instructions (h264-spacing) for -0.9 %, below Slow's knee.

## 3. HEVC split spread + five merge candidates + TU16 chroma (+0.6 %)

Not reproduced on PSNR: the bundle's PSNR-YUV BD was -0.6 / -1.7 / -1.0 % on crowd / pan /
lowlight (`inter_tu_split_spread` alone -0.8 / -1.7 / -0.8 %; five merges and TU16 chroma each
within ±0.1 %). The pool's +0.6 % [+0.2, +1.2] was PSNR +0.1 / VMAF +0.4 / MS-SSIM +0.1: a
tool at the edge of noise, not a loss to chase. The HEVC 8x8 inter CUs (+0.3 %) are the known
pan loss (docs/handoff/hevc.md) and share the AV1 8x8 cause below.

## 4. AV1: chroma trellis (+2.5 %), 8x8 inter blocks (+2.4 %), residual check on every vector (+0.9 %)

AV1 Default, qindex 80/120/160/200, 30-40 frames; two pool clips added for the 8x8 case
(pool_in_to_tree "tree", pool_bodycam_antare_refused_entry_bwc_02 "bwc02", decoded to scratch
and deleted). Work: `~/nanocodec-data/work/tools-hurt/av1/`, branch `tools-hurt-av1`.

**Chroma trellis (`quant.trellis_chroma` 1): no bug; a squared-error trade J rejects.** Probe:
PSNR-Y BD -1.7 / -0.4 / -1.0 %, U/V +14 / +41 / +14 % (pan worst) for -0.5..-1.7 % rate at
matched qindex. AV1's chroma λ is the luma λ times `chroma_scale` × (chroma step / luma step)²,
the same weighting as HEVC, VP9 and H.264 (VVC passes a per-plane λ), and the rate model's
indices match the coefficient writer's. VMAF v1 reads chroma (`speed_chroma_uv`), hence the
pool's VMAF +6.5 % with MS-SSIM -2.3 %. No chroma scale wins everywhere (YUV BD, crowd / pan /
tree / bwc02): 0.05: -0.17 / -0.04 %; 0.1 (tree, bwc02): +0.89 / +1.26 %; 1: +0.64 / +3.12 /
+2.43 / -0.94 %. At small scales the trellis rounds chroma past the dead-zone quantiser and
spends more chroma bits; at 1 it crushes chroma. Keep it off (as every codec's default since
2026-11-15).

**8x8 inter blocks (`planner.inter_min_log2` 3): the planner overrates 16->8 splits.** YUV BD
tree +3.9 %, bwc02 +2.4 %, pan +0.75 %, crowd -0.2 %, bodycam -0.9 %; rate +5..9 % at matched
qindex for +0.04..0.07 dB Y. Every 8x8 split is marginal: a diagnostic 8 extra bits per 8x8
leaf removes all of them (Default streams). Two mechanisms:
- tree: the planner prices each quarter at its own whole-sample search vector with the
  cheapest NEAREST signalling (`planner_leaf_bits`, no vector-difference bits), but mode
  decision codes most quarters at a predicted vector (22,237 of 26,720 8x8 leaves skipped, new
  vectors 7,259 -> 7,192): the split pays partition and mode bits for no motion gain (P bits
  287K -> 348K);
- bwc02: four 8x8 transforms code worse than one 16x16, which the planner's variance estimate
  cannot see (5,632 of 24,140 8x8 leaves skipped).
A motion-separation gate (`planner.small_split_motion_gate`, branch commit 4ceeca67a, off,
not landed) takes tree +3.9 -> +1.1 % but leaves bwc02 at +2.4 %: evidence, not a fix.
**Fix proposal:** confirm each 16->8 quad split by the codec's coded cost, as VVC's
`planner.rect_rd` does through `inter_leaf_cost` (VVC's 8x8 CUs measured -0.6 % that way). AV1
needs an `inter_leaf_cost` (luma prediction + the existing `residual_rd_cost` + mode bits;
`&self`, so scratch behind a `RefCell` as in VVC). Expected: neutral to -0.5 % J (estimate).
Side finding, not chased: `planner.leaf_bits` 12 makes the AV1 Default much worse (pan +19 %,
crowd +17 % YUV BD).

**Residual check on every vector (`av1.residual_rd_max_mv` 64): greedy per-block drop.** Small
on the probe (YUV BD +0.15 / +0.11 / +0.44 / -0.27 % pan / lowlight / bwc02 / tree) but luma
worse on all four (+0.13..+0.98 %) as rate falls 0.1-4.3 %; not the skip bias (bias off:
lowlight +1.0 %). Each residual drop is decided at the frame's λ for the block alone, blind to
its use as a reference; the pool loss sits on one dark clip. No fix short of reference-aware λ
(TPL-like); the Default limit 8 is the guard.

## 5. VVC: dependent quantisation (+6.6 %), multi-type tree (+4.6 %), affine refused

Pool fixed QP (fixedqp-2026-11-20, 24 clips) and probe (crowd / pan / lowlight / static). Work:
`~/nanocodec-data/work/tools-hurt/vvc/`, branch `tools-hurt-vvc` (clean, nothing committed).
Streams with DQ and with MTT 2 decode to the encoder's reconstruction in our decoder and ffmpeg
(identical MD5s): no encoder/decoder mismatch.

**Dependent quantisation (`vvc.dep_quant` 1): J +6.5 % [+5.5, +7.9]; the decisions around it,
not its levels, lose.** Probe at matched QP: +14..23 % bits for +0.3..0.9 dB Y, chroma BD up to
+20 % (pan). Key-frame λ scale 1: +6.6 % (not the λ defaults). The quantiser itself works on
its own transform units: P frames J -2.2 % (crowd), -0.4 % (static) at the mode λ; key frames
with planning off at 32x32 blocks BD-Y -3.7 % / -4.1 % (static / crowd), at 16x16 about -1.5 %.
At 8x8 and with the planner it loses (BD-Y +4.5 % / +0.8 %): inside a DQ frame the plain levels
would give SSE 11.61M, while a frame coded plain reaches 10.68M after filtering, so the DQ frame's
block and mode choices are worse. Ruled out: plain-semantics trial pricing, ALF off, exact
zero bounds, exact forward transforms (each little change). Not yet tested: the rate model
ignores the coded-group flags and builds its context template from nearest-level guesses; a
synthetic 8x8 test has DQ only 0.4 % better than the best dead-zone quantiser. Expected gain
of a full fix: about zero (VVenC's own ablation at our operating point puts DQ within noise,
docs/handoff/vvc.md); a DQ-on-P-slices-only variant needs decoder work for at most ~1 %
(estimate). Not fixed.

**Multi-type tree (`vvc.mtt` 2): J +4.6 %; +1.2 % [+0.7, +2.0] with `planner.rect_rd` 1
(+1.5 % with `rect_bias` 1300).** Two causes: (1) luma: shapes are chosen by the variance
model, blind to the transform (alone BD-Y +2.7 % / +1.9 % crowd / pan; with `rect_rd` luma
gains 0.2-0.9 %); (2) chroma: smaller and rectangular CUs (8x8 squares too) code chroma 0.1-0.3
dB worse at the same QP, chroma BD +5..+18 % even where luma gains. Pricing both chroma planes
in `inter_leaf_cost` moved YUV BD -0.3..+0.2 % (reverted); an exact pre-transform zero bound
had no effect; `unit_zero_rd` off recovered part of crowd only. VVenC gains -4..-6 % from MTT
here, but with full-RD split decisions at 1.9-2.3x CPU: that is the real fix, not a small one.

**Affine beside the second reference: a deliberate refusal.** `check_config`
(`crates/nc-vvc/src/backend.rs`) rejects it because the affine candidate lists know one
reference. VVenC's ablation has affine at +0.7 % at our point: nothing to buy.

Note (silent no-op audit, item 180): VVC Slow's `planner.rect_rd` 1 is dead while `vvc.mtt` is
0 (no rectangular shapes; identical streams at QP 27/37/47). The -0.6 % credited to "8x8 inter
CUs with shape RD" is `planner.inter_min_log2` 3 alone. `rect_rd` matters only with MTT on
(the +1.2 % row above).

## 6. AV2: every transform type and split (+5.4 %), intra widened (+0.5 %); VP9 1/8-sample vectors (+1.6 %)

Fork branch `tools-hurt-av2vp9`; work `~/nanocodec-data/work/tools-hurt/av2vp9/`.

**AV2: the losses were the settings screened, not the tools.** Pool fixed QP (24 clips): the
re-pick's bundle +5.6 % [+4.7, +6.7]; split into parts, the transform split alone -0.4 %
[-0.5, -0.2] and the transform types alone +0.2 % [+0.1, +0.3]; the corrected bundle -0.3 %
[-0.5, -0.2]. The re-pick had set `tx_split` 2, which forces the split instead of trialling it,
and `intra_modes` 4, which is narrower than the Default's 3 (so "intra widened" narrowed it).
Corrected levers against the old AV2 Slow: `av2.inter_split_min_nonzero` 0 J -0.4 % [-0.6,
-0.3] at +11.0 % P instructions; P-frame intra widened (`p_intra_modes` 3,
`p_intra_rd_candidates` 4, `p_intra_chroma_candidates` 13, `p_intra_tx_split` 1,
`p_intra_type_retry` 1, `decision.intra_gate_variance` 16) -0.7 % [-1.0, -0.5] at +13.8 %;
both +24.8 % P instructions (AV2 Slow about x3.07 of the Default, accepted by the coordinator).
Landed aaf73462d: CBR k 0 L1 -1.04 % [-1.57, -0.41], L2 -1.06 % [-1.32, -0.78]; final-check
against 95874a490 J -1.08 % [-1.28, -0.75] BETTER, no guard; preset_identity: only AV2 Slow
changes.

**VP9 1/8-sample vectors (`vp9.high_precision_qindex` 256): a real bug in the pricing and
coding of reference vectors.** With 1/8-sample vectors allowed, a block's NEARESTMV/NEARMV can be
odd. The engine sees vectors truncated to quarter samples (`to_engine_mv`), and `mode_bits` /
`predict_inter_block` converted the candidate back as a plain vector one eighth away: priced
and coded as NEWMV, so blocks beside 1/8 vectors lost their cheap modes (libvpx keeps the exact
reference vector, `use_mv_hp` per vector). Fix landed 95874a490: `coded_vector` maps the
engine's candidate back to the exact reference vector (zero stays ZEROMV). Probe PSNR-YUV BD
(hp 256 vs off, 30 f): bodycam_airport +7.2 -> +0.7 %, crowd -0.4 -> +0.3 %, pan +0.2 -> +0.1 %,
lowlight +1.5 -> +4.9 % (the remaining loss is the post-hoc 1/8 refinement, not located).
Streams with hp off unchanged (preset_identity VP9 81/81); an hp stream decodes bit-exactly in
our decoder and vpxdec. 1/8 vectors stay off in every preset.

## Outcome

Two real bugs fixed (the hollow final stage made inert, 8a9e12f4b; VP9 odd reference vectors,
95874a490), one preset mis-setting corrected (AV2 Slow, aaf73462d, -1.08 % final-check). The
rest are decisions on cheap proxies (H.264 splits, AV1 8x8, VVC MTT and DQ's surrounding
decisions) whose fix is coded-cost confirmation of partitions, or trades J rejects (AV1 chroma
trellis, AV1 residual drop). Open next steps: the final-stage wiring (another agent);
coded-cost confirmation of 16->8 splits for AV1 (and HEVC 8x8); full-RD MTT splits for VVC;
the VP9 1/8 lowlight residue.
