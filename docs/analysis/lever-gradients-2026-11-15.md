# Lever gradients and the slow rung (2026-11-15)

Every rung of the preset ladder should come from one ranked list of measured
levers, cut at the rung's exchange rate between J and CPU. This page holds the
list for `slow`, built from results already on disk (no new quality runs), and
records how well the additive estimate predicted each codec's measured bundle.
The repo-wide lever ledger (`nceval.levers`, being built) supersedes the
throwaway replay used here once it lands.

## The exchange rate: k_slow = 0.025

Screens adopt on J + k·dCPU (J in %, dCPU in %; negative is better). Medium
(each codec's Default) uses k = 0.1: 1 % CPU is worth 0.1 % J. Slow buys
quality inside about 1.5-2.5x Medium's CPU, so its k must let a lever that
costs a lot of CPU through when it gains clearly:

- At k 0.025 a +100 % CPU bundle (2x) must buy at least 2.5 % J.
- The first slow point, HEVC's (below), measured J -5.59 % at +72 % CPU
  (Jetson cycles x1.74, x86 x1.69, weighted 2:1): 0.078 % J per 1 % CPU, so it
  nets J -3.8 % at k 0.025 and would fail at k 0.1 (+1.6 %).
- Levers between 0.025 and 0.1 % J per 1 % CPU are therefore slow material;
  under 0.025 they are not worth their CPU even at the top of the ladder.

## Replay of the stored results at k 0.1 and k 0.025

Sources: the fixed-QP ablations and costs (`work/fixedqp/results/abl-*.json`,
`cost-*.json`, `work/fixedqp2/results/`), the unified-structure, tuner and
rate-control screens (`work/{unified-structure,tuner4,rc-cost2}/*.json`), and
Jetson cycle ratios measured for this page where the x86 cost was missing
(`nceval.jetson` cost clips, 31 f, fixed QP; marked J). J is the stored
interval with its own CPU charge removed. A setting passes at k when J + k·dCPU
< 0 and the upper bound + k·dCPU <= +0.3 (the L3 rule). Fixed-QP J is matched
QP, 16 clips of the fixedqp draw; CBR J is the pool levels. Rows already in a
codec's Default are left out.

| Codec | Setting | J [90 % CI] | Kind | dCPU | k 0.1 | k 0.025 |
|---|---|---|---|---|---|---|
| all | second reference every frame, full decision (`refs.second`, `refs.second_full`, `structure.unified` 0) | HEVC -1.65 [-2.25, -1.16]; AV1 -2.00 [-2.48, -1.61]; VVC -1.91 [-2.46, -1.44]; AV2 -1.46 [-2.00, -0.96] | CBR L3 | J +24.8 / +22 / +13.7 / +11.9 | fail | **pass** |
| VP9 | same | -2.20 [-3.67, -0.87] | fixed QP | J +8.3 | pass | **pass** |
| H.264 | same | refused beside the Default's temporal layers | | | | |
| HEVC | `hevc.coarse_inter_intra` 0 | -0.51 [-0.70, -0.29] | fixed QP | +5.9 | fail | **pass** |
| HEVC | refinement early exits off | -0.40 [-0.58, -0.28] | fixed QP | +5.9 | fail | **pass** |
| HEVC | `hevc.inter_tu_split_spread` 1 | -0.27 [-0.52, -0.13] | fixed QP | +9.0 | fail | **pass** |
| HEVC | `quant.trellis_states` 8 | -0.52 [-0.80, -0.30] | fixed QP | +78.0 (J 4 states +23.5) | fail | fail |
| HEVC | exact forward transforms | -0.02 [-0.27, +0.25] | fixed QP | ? | fail | fail |
| AV1 | `av1.pre_tx_skip` 0 | -2.27 [-3.30, -1.15] | fixed QP | +8.1 | pass | **pass** |
| AV1 | no depth removal | -0.72 [-0.95, -0.48] | fixed QP | 0.0 | pass | **pass** |
| AV1 | refinement early exits off | -0.60 [-0.77, -0.43] | fixed QP | +13.6 | fail | **pass** |
| AV1 | biased skip off (`av1.skip_bias_pct*` 100) | -0.60 [-1.14, +0.02] | fixed QP | +0.6 | pass | **pass** |
| AV1 | motion search without early exits | -0.47 [-0.66, -0.27] | fixed QP | +1.8 | pass | **pass** |
| AV1 | `av1.inter_tx_split_energy` 0 | -0.39 [-0.51, -0.28] | fixed QP | +9.4 | fail | **pass** |
| AV1 | `decision.subpel_walk` 1 | -0.32 [-0.46, -0.17] | fixed QP | +0.1 | pass | **pass** |
| AV1 | `av1.inter_intra_search` 2 | -0.30 [-0.47, -0.12] | fixed QP | +0.9 | pass | **pass** |
| AV1 | `quant.trellis_states` 8 | -0.53 [-0.74, -0.32] | fixed QP | +40.8 (J 4 states +12.3) | fail | fail |
| AV1 | loop restoration | -0.88 [-1.71, -0.11] | fixed QP | +145.9 | fail | fail |
| AV1 | `av1.residual_rd_max_mv` 64 | +0.74 [+0.38, +1.11] | fixed QP | | fail | fail |
| AV1 | `av1.chroma_skip_sad_num` 0 | +2.44 [+0.46, +4.58] | fixed QP | | fail | fail |
| VVC | `planner.inter_min_log2` 3 | -0.64 [-1.26, -0.07] | fixed QP | J +8.3 | pass | **pass** |
| VVC | affine | -0.69 [-1.19, -0.19] | fixed QP | | | refused beside the second reference |
| VVC | trellis 8, refinement exits off, motion search, full P intra, no depth removal | -0.1 to +0.1 | fixed QP | J +8 (exits) | fail | fail |
| VVC | MTT 1 / 2 with `planner.rect_rd` | +1.3 / +0.9 | fixed QP | | fail | fail |
| VP9 | transform split and size search everywhere | -0.36 [-0.54, -0.14] | fixed QP | J +1.4 (with the next two) | pass | **pass** |
| VP9 | no depth removal | -0.22 [-0.38, -0.06] | fixed QP | (above) | pass | **pass** |
| VP9 | motion search without early exits | -0.18 [-0.29, -0.06] | fixed QP | J +0.6 | pass | **pass** |
| VP9 | `quant.trellis_states` 8 | -0.34 [-0.61, -0.07] | fixed QP | J +128 (4 states +54) | fail | fail |
| AV2 | TCQ on inter frames, pruned (`av2.tcq` 2, `tcq_prune_bits` 5) | -2.27 [-2.59, -1.95] / -2.67 [-3.25, -2.13] (two draws) | fixed QP | +40.9 (J +30.3) | fail | **pass** |
| AV2 | `av2.interp_switchable` 1 | -1.20 [-2.28, -0.24] | fixed QP | +1.7 | pass | **pass** |
| AV2 | motion search without early exits | -0.62 [-1.01, -0.24] | fixed QP | ~+1 | pass | **pass** |
| AV2 | `quant.trellis_states` 8 | -0.49 [-0.71, -0.28] | fixed QP | J +10.2 | fail | **pass** (overlaps TCQ) |
| AV2 | `av2.ref_mv_bank` 1 | -0.41 [-0.75, -0.07] | fixed QP | J +0.6 | pass | **pass** |
| AV2 | `decision.subpel_walk` 1 | -0.37 [-0.54, -0.20] | fixed QP | J +0.9 | pass | **pass** |
| AV2 | `av2.max_drl_bits` 3 | -0.32 [-0.60, -0.04] | fixed QP | J +1.6 | pass | **pass** |
| AV2 | refinement early exits off | -0.11 [-0.29, +0.05] | fixed QP | J +4.2 | fail | fail |
| H.264 | partition search everywhere | -0.70 [-1.08, -0.36] | fixed QP | +17.8 | fail | **pass** |
| H.264 | RD transform size | -0.51 [-0.71, -0.29] | fixed QP | +21.2 | fail | fail (+0.02) |
| H.264 | `quant.trellis_states` 8 | -0.21 [-0.37, -0.05] | fixed QP | +32.9 | fail | fail |
| H.264 | refinement early exits off | +0.06 [-0.12, +0.23] | fixed QP | | fail | fail |

Several cheap levers pass at Medium's own k 0.1 but are not in the Defaults
(AV1 pre-transform skip off, no depth removal, motion search, sub-sample walk,
inter intra search 2, biased skip off; AV2 switchable filters, MV bank, DRL 3,
walk, motion search; VP9's three). They were measured at fixed QP only; each
is a Medium candidate for its own CBR screen.

## Slow bundles: predicted and measured

Predicted J is the sum of the passing levers' J (the additive estimate, no
overlap discount); predicted CPU the product of their ratios. Measured J is the
L3 confirmation against Medium on the day's draw, at k 0.

| Codec | Bundle | Predicted J | Predicted CPU | Measured CPU (Jetson cycles / x86) | Measured L3 J |
|---|---|---|---|---|---|
| HEVC | (screened before this method) | | | x1.74 / x1.69 | **-5.59 % [-6.48, -4.71]** (n 78, L2 -4.61 %) |
| AV1 | (screened before this method; carries `residual_rd_max_mv` 64, a fixed-QP loser) | | | x1.60 / x1.65 | **-2.07 % [-2.63, -1.55]** (n 78, ladder; L2 -2.47 %) |
