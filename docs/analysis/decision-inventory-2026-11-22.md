# Decision inventory: thresholds that could become functions of features (2026-11-22)

**Why.** The owner's design for learned decisions: take a fixed threshold compared against a
computed quantity and replace it with `threshold(features) = bias + w · features`. The bias starts
at today's value and w at 0, so today's rule is a degenerate case of the function. Then tune the
terms jointly on real encodes (`nceval.tune`). Where today's tree has one fixed test, the extra
inputs and alternatives start at "off". The first instance is the RD partition search's context
gate (`planner.rd_split_{deepen,merge}_shift_*`, `crates/nc-encoder/src/linear_threshold.rs`,
`rd_split.rs` `ContextShift`).

This file lists the registered decision parameters that are compared against a computed quantity.
For each it gives what the threshold is compared against, the features at that point, where it is
used, and its measured leverage. It is a read-only survey of the source, presets, code comments and
the ideas ledger at d386bf350. Numbers are quoted only where they were read, and "unknown" means
none was found. The generalisation options are proposals, each starting off. "med" is the backend's
`DEFAULT_PARAMS`.

## Two findings that shape the plan

- **The main planner decision is already in this form.** `planner.split_selector` multiplies the
  split bias by e^h, with h linear in node features, so w = 0 is the old rule. It ships at scale 0.5
  on the medium rungs of AV1, HEVC, VVC, VP9 and AV2. Pool L3 J ranged from −0.20 % (VVC) to
  −1.53 % (AV2); the VP9 and HEVC fast rungs did not pass (`docs/handoff/planner-selector.md`). It
  reads no depth context and no frame role.
- **The predictive features are coarse.** In `docs/analysis/adaptive-effort-2026-11-16.md` §0 and
  §3, per-superblock pixel features did not predict where effort pays: gain and cost rose together
  in every quintile, and a GBDT on ten features ranked no better than the proxy. Two things did
  predict. One was value per clip and per frame role (a 7x range). The other was the neighbour and
  co-located leaf depth, the only predictor found on the 1.16 M-node RD-split dump. So feature terms
  should favour frame role, per-clip running statistics, noise σ, λ/QP, block size and the depth
  context.

## (1) Planner depth decisions (`partition_plan.rs`)

| param (default) | compared against | features there | non-default use | leverage (read) | generalisation options (off) |
|---|---|---|---|---|---|
| `planner.bias_base` / `bias_q_slope` / `bias_min` / `bias_max` (1100/150/900/1200 ‰) | `bias·leaf ≤ 1000·split` (`plan_node`) | leaf distortion, residual var/mean, quadrant var ratio and imbalance, quadrant MV distance, split/whole SAD, size, q, 8x8 cost spread, noise | AV1 l2-medium | a global lower bias lost fixed-QP J +3..+7 % (selector doc) | generalised by `split_selector`; add depth-context and frame-role terms |
| `planner.split_selector_scale` (1.0; 0.5 at every medium) | scales h | as above | 5 mediums 0.5; VVC slower | L3 −0.20..−1.53 % | per-codec and per-rung refit; context term; frame-role term |
| `planner.lambda_scale` (2.0) | split vs leaf λ weight | λ, q | HEVC/AV2 2.58, VVC 4.0, VP9 2.38 at medium; many fast rungs | not quantified | per-size and per-frame-role value |
| `planner.split_rate_factor` (2.0) | split-symbol bits weight | partition bits | HEVC medium and fast 2.26 | VP9: no effect | — |
| `planner.distortion_cap_per_lambda` (0.125) | per-sample distortion cap | residual var, λ | none | forcing larger blocks +41.8/+58.3 % vs SVT (rejected); named as a source of planner errors | function of noise σ and residual mean |
| `planner.high_8x8_cost_variance` (150 000) | 8x8 cost spread × qp weight × headroom → cap at 32x32 | spread, qp, headroom | none | AV1 J −0.6 %, HEVC −0.3 % (500 000: −0.7 %) | noise and motion term |
| `planner.depth_removal_th_32_16` / `_16_8` (0/0; fast rungs 25/50) | improvement < th·scale (`remove_below_32`) | ME distortion, spread, qp | fast/faster/veryfast of AV1, VP9, HEVC, VVC | removal off: HEVC −1.01, AV2 −3.12, AV1 −0.66, VVC/VP9 ≈0 (L3) | frame-role value; noise term |
| `planner.depth_removal_qp_scale` (2) | threshold multiplier | headroom, band | AV2 med 1, AV1 veryfast 8, HEVC faster 4 | AV2 1: TRAIN −0.7 / VAL −1.2; AV1 8: CPU −2 % for +2.8 % | — |
| `planner.depth_removal_min_quantizer` (0) | relative q ≥ value | q | med HEVC/VP9 0.86, VVC/AV1 0.8; fast rungs | HEVC quality −1.7 %, AV1 −3.5/−1.5 % at +4.6/+9.7 % instr | a smooth ramp in q instead of a step |
| `planner.depth_removal_sad_per_step` (0) | 32x32 SAD/sample < k·step | SAD, step | none | 0.5: +11.4 % BD for −8 % time (rejected) | — |
| `planner.zero_out_gate` (0.2) | residual σ < k·step | σ, step, size | none | ungated +14 % instr; 0.4/0.7: −1.84/−3.09 % x86, static class +3.6/+6.0 % | still/noise term |
| `planner.rect_motion_gate` (1) | MV separation ≥ k samples | quadrant MVs | none | unknown | SAD-ratio term |
| `planner.inter_floor_motion_gate` / `_sad_ratio` (0/1) | MV difference / SAD ratio | quadrant MVs, SADs | none | camera pool neutral; text −74..−80 % P bytes | — |
| `planner.intra_bias` (1050) | key-plan parent multiplier | intra leaf cost | none | all-intra BD −1.1..−1.9 %; J −0.4..−0.6 % | λ and texture term |
| `effort.regular_fraction` / `regular_proxy` (1/0) | superblock rank by SAD + spread proxy | proxy, frame role | AV1 faster/veryfast, HEVC fast, VP9 fast | rf 0: +11.1 % BD at 0.78x; per-SB proxy ≈ random in CBR | frame-role and per-clip value |

## (2) RD partition search gates (`rd_split.rs`)

| param (default) | compared against | features | use | leverage | generalisation options (off) |
|---|---|---|---|---|---|
| `planner.rd_split_deepen_context` / `merge_context` (−1/−1) | depth context (min; clamped sum) | neighbours, co-located, size, planner margin, residual | slow rungs | the gate: HEVC Default −4.2 % at +40 % vs the full search's −5.4 % at +86 % | **done**: `planner.rd_split_{deepen,merge}_shift_*` (11 terms) |
| `planner.rd_split_margin` (0) | planner alternative < m × chosen | margin, direction, size, q | none shipped | change rate flat 21-31 % over margin buckets; VP9/VVC deepen-only m1.2 L3 −0.30/−0.50 at +13..17 % cycles (held) | margin × context product; per-direction values |
| `planner.rd_split_deepen_residual` (0) | residual var/sample vs k·λ | residual var | none | unknown | — |
| `planner.rd_split_skip_exit` (switch) | planned leaf coded as skip | skip flag | all | "rarely worse" | exit when the leaf's coded residual is under kλ |
| `planner.rd_split_merge_skipped` (0) | all children skip | skip flags | none | unknown | — |
| `planner.rd_split_shape_gate` (0) | MV separation ≥ k | MVs | none | unknown | — |
| `planner.rd_split_tt_margin` (0.05) | binary shape within m of best | coded costs | VVC MTT (off) | VVC mtt2 −10.7 % at +283 % instr (rejected) | — |
| `planner.rd_split_tail_satd` (0) | SATD tail at the bound | SATD | slow/slower | −1.4..−2.8 % cycles for +0.07..+0.44 % J | — |
| `planner.rd_split_frame_period` (1) | frame index | frame role | none | unknown | base-only search (frame role) |
| `planner.key_rd_margin` / `key_rd_merge_margin` (0/0; med 1.1, AV2 1.2, slow/slower 1.2-1.5) | key-plan alternative < m × chosen | margin, size, intra leaf cost/λ, λ | 5 tree codecs, medium and slow | L3 −0.10..−0.55, final −0.34..−0.64 at +2.0..+3.9 % cycles | generalised by `key_rd_gate` |
| `planner.key_rd_gate` (0) | tree score ≥ g | margin, direction, size, leaf cost/λ, λ | none | offline 85-95 % of gain vs 68-75 % for the margin; pool train partial (landed off) | — |

## (3) Mode decision: early exits and gates (`decision.rs`, `final_stage.rs`, `cu_pricing.rs`, `still_skip.rs`, backends)

| param (default) | compared against | features | use | leverage | generalisation options (off) |
|---|---|---|---|---|---|
| `decision.codec_subpel_seed_lambdas` (0.03) | seed residual var/sample < k·λ | seed var, λ, area, size | H.264 0.1, VVC 0.05, HEVC/VP9 0 at medium; fast rungs 0.01-0.3 | HEVC 0.05: instr −6.7 %, TRAIN +0.87 %; AV1 0.03: CPU −7.0/−4.0 %; `adapt.subpel` fast rungs L3 −0.99..−1.43 % | per-size, frame role, noise σ |
| `decision.skip_no_refine_lambdas` (0.25) | skip residual < k·λ·area | skip-vector residual, λ | H.264 0.21, HEVC 0; many rungs 0 | unknown | frame role |
| `decision.refine_cheap_min_lambdas` (0.05) / `seed_refine_ratio` (0; AV1 med 1.1) | distortion ≥ k·λ·area; seed vs cheap ratio | distortion, λ | several | VVC 1.1: Jetson −1.7 %, L1 −0.09 (held) | — |
| `decision.codec_fullpel_probe_min_lambdas` (0; AV1 fast rungs 0.0094) | seed var ≥ k·λ·area | seed var, λ | AV1 fast/faster | gated probes TRAIN BD −1.7/−1.9 % | — |
| `decision.exact_rank_margin` (0; fast rungs 1.0-1.5) | bilinear cost > m × best exact | costs | fast rungs | engine 0: VVC L3 −0.43 %; 1.0 on fast rungs instr −0.14..−0.53 % (rejected) | — |
| `decision.winner_walk_min_lambdas` (0; H.264/VP9 med 0.25) | winner var ≥ k·λ·area | var, λ | H.264, VP9 | unknown | — |
| `decision.intra_gate_variance` (64; H.264 16, HEVC 0, VP9 6.86) / `intra_gate_lambdas` (0) | inter residual var/sample > g | residual var, DC error, λ | widely varied | λ form held: VC-1K worst six −5.8 % (dark −12/−22), L3 −0.04 %; AV1/AV2/VP9 λ gate −0.7..−3.1 % on VC-1K (held) | var + λ jointly, noise σ, brightness |
| `decision.final_margin_lambdas` (0.1; H.264/VVC 0.03) | screen cost within m·λ·area | screen costs, λ | H.264/VVC medium, HEVC slower | H.264 stage VC-1K −9.8 %; VVC stage L3 −0.76 at +7.5..8.7 % instr | per-size, frame role |
| `decision.final_max_winner_lambdas` / `final_intra_gate_lambdas` / `final_intra_energy_ratio` | winner error, residual, AC energy vs k | λ, energy | H.264, VVC | within the stage's numbers | — |
| `decision.cu_price_intra_lambdas` (0; HEVC med 0.1) and its error/ratio gates | intra margin < k·λ/sample | intra margin, winner error, intra/inter ratio | HEVC medium | HEVC L3 −1.00 %, final −0.80 % at +0.5..+6.8 % instr | carry to other codecs |
| `decision.intra_directional_dc_ratio` (4; HEVC/VVC 2) | DC cost ≥ r × inter | costs | HEVC, VVC | unknown | — |
| `decision.noise_skip` (0) / chroma ratio | zero-vector MSE ≤ k × noise var | MSE, noise σ, still flag | none shipped | stills up to +2 dB, 18-23 % fewer instr; fast rungs +4..+20 % P instr (rejected) | — |
| `h264.split_gate_satd` / `split_gate_lambda` (33.3/0; med λ 0.8) | 16x16 SATD/sample | SATD, √λ | H.264 | λ 0.8: quality −0.31 % at +2.8 % CPU | — |
| `h264.intra4x4_gate_satd` / `intra8x8_gate_satd` / `intra_nxn_gate_lambda` | I16x16 SATD/sample | SATD, √λ | H.264 | 16: +0.54 %; speed variants +2.3..+3.6 % | — |
| `h264.subpel_skip_variance` / `_lambda` (24/0) | residual var | var, λ | none | fixed gate ~16 % rate on grainy high-rate content | — |
| `av1.pre_tx_skip_energy` / `_score` / `_rd` / `texture_scale` | 100·SSE < e·area·qindex; score; RD | SSE, qindex, neighbour skips | AV1 all rungs; AV2 fast | P instr −5.5 %, PSNR-YUV +0.82 %; texture 8: −3.8 % instr | frame role (`_score_nonbase` exists) |
| `av1.skip_bias_pct` / `_base` / `skip_bias_energy` | skip distortion % vs energy·λ | energy, λ, frame role | slower 100 | unknown | — |
| `av1.chroma_skip_sad_num` / `zero_scale` / `min_error` | sampled SAD; zero-residual veto | SAD, q | fast rungs 0 | veto J −0.46..−0.99 % at +1.7 % cycles | — |
| `av1.keyframe_flat_gate` / `av2.intra_flat_gate` | DC SSE/sample | SSE, size | AV1 fast..veryfast, AV2 faster | 150 searches about half the key blocks | already per-size |
| transform-split gates (`av1.inter_tx_split_energy`, `gate.tx_split_*`, `vp9.inter_tx_split_spread`, `hevc.*_tu_split_*`) | prediction error / spread / nonzeros | energy, q | VP9 med 1.047, fast 1.5 | VP9 2: Jetson −14.4 %, x86 −8 % for J +2..+3.5 % | — |
| `vp9.nonrd_skip_scale`, `vp9.tx_search_min_levels`, `hevc.tu16_chroma_concentration` | residual var vs AC thr; levels; chroma share | var, levels | VP9, HEVC | nonrd: Jetson −5.0 %, J +2.0 %; tu16 off: callgrind −3.3 %, L3 −3.10 | — |
| `vvc.unit_zero_rd` / `_max_levels`, `zero_residual_texture`, `affine_*_gate`, `early_merge_lambdas`, `mts_min_levels` | squared error, levels, prediction error | λ, levels | VVC | early merge 0.03: −0.9 % for +0.27 % (rejected) | — |

## (4) Motion analysis and frame-level gates

- **Zero-vector SAD gates in `nc-analysis` `motion.rs`.** These are
  `analysis.static_sad_per_pixel`, `one_position_sad_per_pixel` and `skip_hme_sad_per_pixel`.
  - With the gates off (held), crowd 480p BD-J was HEVC −6.5, VP9 −2.2, AV1 −10.7, VVC −7.9 and
    AV2 −8.2 %; crowd 720p was −7.7..−12.6 %. CPU was not measured.
  - HME plus the full-pel search is 0.2-6.9 % of x86 time.
- **`analysis.window_exit_sad_per_pixel` and `hme_level1_max_variance`:** leverage unknown.
- **`hold.change`, `scene.*`, `layers.adaptive_*`, `filter.*_proxy`, `filter.usage_quiet_share`,
  `refs.second_*_margin`:** frame level, numbers unknown.
- **`rate.aq_strength` and `persist.strength`:** off everywhere; every AQ variant in the ledger was
  rejected.

## Ranking by expected leverage (judgement from the numbers above)

1. The `analysis.*_sad_per_pixel` zero-vector gates.
2. `decision.codec_subpel_seed_lambdas`.
3. `planner.depth_removal_*`.
4. `split_selector` with depth-context and frame-role terms.
5. `decision.intra_gate_variance` / `_lambdas`.
6. The `rd_split` margin family.
7. `key_rd_margin`.
8. The `final_stage` margins.
9. The `cu_price_*` gates.
10. AV1 `pre_tx_skip`.
11. `effort.regular_*`.
12. `planner.high_8x8_cost_variance` and `distortion_cap_per_lambda`.
13. The transform-split gates.
14. `exact_rank_margin`, the probe and refine exits.
15. `planner.zero_out_gate`.

## Staged tuning groups (10-25 parameters each)

- **G1, planner depth (~18).** Parameters: the `bias_*` four, `split_selector_scale`,
  `lambda_scale`, `split_rate_factor`, `distortion_cap_per_lambda`, `high_8x8_cost_variance`,
  `depth_removal_*` (four), `zero_out_gate`, `rect_motion_gate` and `effort.regular_proxy`. New
  terms: depth context, frame role, noise σ. Next after the context gate, at medium.
- **G2, motion and sub-pel exits (~14).** Parameters: the three zero-vector gates,
  `zero_gate_cell`, `window_exit_sad_per_pixel`, `hme_level1_max_variance`, the `subpel_seed` pair,
  `skip_no_refine`, `refine_cheap`, `seed_refine_ratio`, `fullpel_probe`, `exact_rank_margin` and
  `winner_walk_min`. New terms: per-size values and noise σ.
- **G3, intra and final-stage gates (~14).** The `intra_gate` pair, `intra_directional_dc_ratio`,
  the `final_*` margins, the four `cu_price_*`, and the H.264 intra gates in λ form.
- **G4, RD partition search at the slow rungs (~10).** `rd_split_margin`, `deepen_residual`,
  `merge_skipped`, `shape_gate`, `tt_margin`, `tail_satd`, `frame_period`, `key_rd_margin`,
  `key_rd_merge_margin` and `key_rd_gate`.
- **G5, codec-local transform and skip gates (~20, per codec).**

## Results, 2026-11-22 (nn-tools)

**Method.** Each mechanism was tuned on real encodes (`nceval.tune search --window-frames 600`; the key studies also used `--second-key-share 5:150`), warm-started at the shipped point. The best trial-clip points were then held out: a long-window L3 at 1 key in 150 frames and at 1 key in 300 (`nceval.screen --levels final --final-frames`), with each share's CPU charged at the rung's front k.

| mechanism | rung | trial clips | held out (mean of shares) | CPU 1/150, 1/300 | outcome |
|---|---|---|---|---|---|
| key RD search, cheap form | HEVC medium | t2 | -0.47 % [-0.82, -0.08] BETTER | Graviton +1.89 / +0.93; Zen 5 +2.67 / +1.95 | landed 35009d02a |
| context-gate shift | HEVC slow | t4 | -0.13 % [-0.33, +0.15] BETTER (k 0.0697) | Graviton -3.28 / -2.67; Zen 5 -3.45 / -2.49 | landed fc351808e |
| key RD search | VP9 medium | t7 | -0.09 % [-0.31, +0.04] | Graviton +6.60 / +2.22 | not landed: crosses 0 at a large key CPU |
| split selector corrections (G1) | HEVC medium | t53 (-0.91 %) | -0.12 % [-0.32, +0.06], AMBIGUOUS at 1/300 | Graviton -0.61 / +0.53 | dropped |
| post-trial stop | AV1 slow | t15 (-0.70 %) | +0.04 % [-0.30, +0.38] WORSE | Graviton -0.57 / -0.58 | rejected |
| context-gate shift | AV1 slow | t36 (-0.64 %) | +0.09 % [-0.07, +0.23] AMBIGUOUS | Graviton -1.77 / -2.39 | not landed |
| key-distance terms (v2) | HEVC medium vs head | t19 | +0.28 % [+0.06, +0.49] WORSE (1/150 +0.63 %) | Graviton -1.45 / -1.18; Zen 5 -1.77 / -1.50 | rejected |
| key-distance terms (v2) | AV2 medium | t10 | -0.05 % [-0.19, +0.09] WORSE | Graviton +3.62 / +0.02; Zen 5 +4.60 / +0.25 | rejected |

**Correction.** The commit messages of 35009d02a and fc351808e give their CPU as "Graviton 3/4 + Zen 5". That is wrong. `aws_cpu_ab`'s default targets are `graviton3,graviton4`, so those numbers are Graviton only. Zen 5 was measured afterwards and is in the table:
- HEVC slow's saving holds.
- HEVC medium's key search costs more on Zen 5. With equal Arm/x86 weight its held-out mean is about -0.35 %, still better.

Always pass `--targets zen5,graviton3,graviton4`.

**Lessons.**
- 150-frame trial gains did not survive. AV1 medium G1 measured -1.55 % at 150 f, -0.34 % at 600 f, and held out lower still.
- The tool's callgrind CPU estimate missed the key-frame cost at 1 key in 150: VP9 t7 and AV2 t10 measured +4 to +7 % there on rented cores.
- `nceval.screen --preset` sets only k, not the rung's parameters. Pass the rung's preset file values to both arms; read them from the binary's own `presets/`.
