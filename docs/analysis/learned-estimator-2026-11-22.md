# A learned correction of the planner's cost estimate (2026-11-22)

**Question (owner, via the learned-search line).** The encoder predicts rate and distortion
before it codes. Can a model that starts at today's hand-made estimate and learns a correction
from zero predict the coded costs well enough to pick the real trial's winner more often? And
does conditioning on the coded neighbours ("NNUE-style": each coded block updates the context
of later predictions) help?

**Answer, offline only.** The correction helps, but not by enough to try it in the encoder.
The coded-context part adds nothing. Nothing here changes a stream, and nothing is wired in.

Scripts: `eval/estimator/extract.py` builds the table, `eval/estimator/accuracy.py` fits and
scores. Data: nn-data v0.3 (manifest `87f87010`): all 2639 shards, at most 3000 rows per shard,
4.48M rows. Held out: `nndata.split`, seed `nn-estimator-1`, whole source groups, 640k rows.

## 1. Which estimate is measured

v0.3 records one estimate together with its exact coded labels: the inter planner's
leaf-vs-quad-split comparison (`node/plan_costs`: `bias * leaf` against `1000 * split`,
including the split selector's shift; `partition_plan.rs`). Its labels are the RD partition
search's full codings of both slots (`node/cost`).

Rows: inter-frame nodes that the planner weighed both ways and the search coded both ways,
neither slot abandoned. They come from medium's search twins and from slow, for HEVC, AV1 and
VVC.

The dataset does not record mode decision's per-candidate estimates (SATD / SAD plus rate
tables) or the coefficient-price model's predictions, so this study does not cover them.

## 2. Today's estimator

log(coded / estimated): the median is the bias, IQR / 1.349 the spread. The split side is
counted only on planned-split nodes, because a planned leaf's split estimate is the early
exit's partial sum.

| codec, rung | leaf bias / spread | split bias / spread | order agrees | Spearman | regret, % of node J | leaf share: coded / planner |
|---|---|---|---|---|---|---|
| HEVC medium twin | +0.12 / 0.31 | +0.19 / 0.41 | 74.5 % | 0.57 | 1.45 | 0.69 / 0.56 |
| HEVC slow | +0.15 / 0.32 | +0.28 / 0.46 | 71.2 % | 0.58 | 1.46 | 0.64 / 0.48 |
| AV1 medium twin | +0.09 / 0.21 | +0.11 / 0.28 | 81.9 % | 0.62 | 1.64 | 0.77 / 0.73 |
| AV1 slow | +0.12 / 0.27 | +0.25 / 0.43 | 77.3 % | 0.67 | 1.59 | 0.69 / 0.58 |
| VVC medium twin | -0.02 / 0.29 | -0.03 / 0.34 | 78.2 % | 0.62 | 1.19 | 0.66 / 0.61 |
| VVC slow | -0.02 / 0.29 | +0.03 / 0.37 | 74.6 % | 0.62 | 1.23 | 0.61 / 0.52 |

- Per candidate the estimate is off by 20-45 % (log spread 0.2-0.46). Replacing trials needs
  2-5 %, so it is about ten times too coarse.
- 32x32 nodes are the worst: agreement 70-78 %, regret up to 2.1 %. 64x64 nodes are better:
  78-88 %, regret 0.6-1.0 %.
- A constant shift of the comparison, fitted per codec, rung and size for minimum regret,
  lands at 0 to -0.06 and gains at most 0.04 points of regret. The planner is calibrated on
  average; its errors depend on the content.

## 3. The learned correction

The corrected log ratio is `log(leaf / split)_estimated + f(inputs)`, with f starting at zero.
f is a gradient-boosted regression of the estimate's clipped log-ratio error (sklearn HGB, 300
trees, 63 leaves). The `decision_*` rows are a LightGBM logistic model of "the leaf codes
cheaper" instead. It starts from today's score, `-k * log ratio`, and each node is weighted by
`|J leaf - J split| / lambda`.

Held out, all sizes, as agreement / Spearman / regret %:

| inputs | HEVC med twin | HEVC slow | AV1 med twin | AV1 slow | VVC med twin | VVC slow |
|---|---|---|---|---|---|---|
| today | .743/.55/1.78 | .717/.56/1.72 | .824/.65/1.65 | .787/.67/1.58 | .784/.60/1.47 | .749/.61/1.49 |
| base: estimate, node, frame, superblock, motion search, pre-analysis, trial bytes | .774/.70/1.57 | .745/.67/1.63 | .842/.80/1.46 | .810/.79/1.40 | .813/.76/1.08 | .771/.72/1.29 |
| base + coded context | .776/.71/1.54 | .745/.67/1.68 | .842/.81/1.47 | .808/.79/1.43 | .815/.77/1.06 | .774/.73/1.28 |
| decision loss, base + context | .781/.65/1.51 | .751/.65/1.65 | .842/.76/1.49 | .811/.76/1.46 | .816/.72/1.15 | .775/.70/1.31 |
| decision loss, before-children inputs | .775/.64/1.62 | .747/.63/1.68 | .840/.76/1.53 | .808/.76/1.47 | .810/.71/1.20 | .770/.69/1.38 |
| planner-held only (17 inputs) | .764/.67/1.66 | .739/.64/1.74 | .836/.79/1.62 | .805/.77/1.56 | .798/.74/1.34 | .766/.70/1.42 |
| planner-held, 64 trees of depth 4 | .760/.65/1.73 | .736/.63/1.71 | .835/.78/1.55 | .804/.76/1.48 | .789/.72/1.47 | .759/.68/1.48 |

- **Coded context.** These inputs are the left, above and co-located superblocks' estimate
  calibration (log coded / estimated of their planned leaves and splits, all sizes and the
  node's size), their coded J per sample over lambda, and the co-located node's calibration.
  They move agreement by -1.0 to +0.7 points and regret by -0.07 to +0.05. The planner's
  neighbours' errors do not predict its error here.
- **Best correction** (base inputs, which need the split estimate): regret -5 % (HEVC slow) to
  -27 % (VVC medium twin), agreement +1.8..+3.1 points, Spearman +0.11..+0.16.
- **Deployable without breaking the planner's exact early exit** (inputs known before the
  children are planned, decision loss): regret -2 % to -18 %, agreement +1.6..+3.2 points. A
  squared-loss regression on those inputs is worse than today (regret +0.4..+0.6), because
  without the estimate it cannot learn a decision.
- **What the planner already holds** (trial bytes, estimate, size, lambda): regret +1 %
  (HEVC slow) to -9 %. A small tree keeps 0 % to -6 %.
- Permutation importance (base): the estimated ratio dominates, followed by the split estimate
  per sample, the margin, trial byte 2, size and the leaf estimate. Pre-analysis and motion
  search inputs follow at a tenth of that or less.

## 4. Why no encode A/B

The brief's bar for wiring was a clearly better rank accuracy. Where the correction is cheap
enough to deploy, it gains 1.5-3 points of agreement and a few percent to 18 % of a
single-node regret. That regret is itself 1.2-1.8 % of node J and myopic: it ignores later
frames.

The two nearest attempts gained more offline and lost in the encoder:
- the distilled flip guide: 11-12 % of its oracle offline, J 0 to +2 % in the encoder
  (distill-fast-2026-11-22.md section 2);
- the learned value gate (learned-search-2026-11-22.md).

The linear split selector, which shifts the same comparison by the same mechanism, is already
shipped (-0.93 % L3). The remaining content-dependent gain is the size of the margin that
those closed loops lost.

What would change this: per-candidate labels for mode decision's estimate stage and the
coefficient-price model. Neither is in v0.3. A dataset column for each candidate's estimate
next to its coded parts would let the same study run where the estimate's spread may be
smaller.

## 5. At the encoder's cost limit (coordinator's request, 17:10)

The coordinator asked for one encode test of the deployable correction. The model limit was
64 trees of depth 4 (a few hundred cycles a node at medium) or a linear fit within a point.
The VVC regret cut it cited (1.47 to 1.08-1.2) came from 400-tree, 63-leaf models, so it was
first re-measured at that size, offline.

Script: `eval/estimator/deployable.py`, with `--probe` for the wider input sets. It uses the
same data and split as above, with the decision loss and one model shared by every codec.

The inputs are integers the encoder computes exactly: trial bytes 0 and 3-10 (size and the
SplitEvidence bytes, which do not depend on the margin or the choice), plus the leaf estimate
per sample over lambda and lambda itself, as `log2_steps`.

Held out, as agreement / regret %:

| model | HEVC med twin | HEVC slow | AV1 med twin | AV1 slow | VVC med twin | VVC slow |
|---|---|---|---|---|---|---|
| today | .743/1.78 | .717/1.72 | .824/1.65 | .787/1.58 | .784/1.47 | .749/1.49 |
| 64 trees of depth 4, planner bytes | .764/1.74 | .742/1.77 | .833/1.68 | .802/1.60 | .793/1.47 | .763/1.53 |
| linear, planner bytes | .756/1.76 | .732/1.80 | .829/1.59 | .799/1.52 | .779/1.49 | .745/1.55 |
| 64x4, superblock / pre-analysis / search stats | .759/1.76 | .735/1.86 | .832/1.68 | .799/1.61 | .780/1.51 | .751/1.60 |
| 64x4, stats + planner bytes | .763/1.73 | .741/1.78 | .833/1.67 | .802/1.61 | .790/1.48 | .762/1.52 |
| 64x4, stats + depth context | .767/1.70 | .740/1.72 | .837/1.54 | .808/1.43 | .796/1.45 | .762/1.48 |

- **Under the cost limit the regret gain disappears.** Agreement still rises 1-2.5 points,
  but regret moves -0.04 to +0.14 at VVC and HEVC, and VVC medium stays at 1.47.
- **The only small model that cuts regret reads the depth context:** the final leaf sizes of
  the left, above and co-located neighbours, which are coded context. It gains AV1 -7 % /
  -9 % and HEVC medium -4 %, but only -1 % at VVC. Medium does not keep that depth map: it
  exists only with `planner.rd_split_context`.
- **Not wired.** With no offline regret gain at the allowed size for VVC / HEVC medium, the
  encode A/B would test a regret-neutral model. The one candidate left is the depth-context
  model at AV1 medium and slow. It needs the depth map kept at medium (its cost) and has not
  been tried.
