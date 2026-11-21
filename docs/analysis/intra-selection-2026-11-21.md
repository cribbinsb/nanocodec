# Cheap intra selection: the key plan against the key RD search (2026-11-21)

Owner ask: "lots of easy wins here ... extending the cheap selection ideas we already use to
I-frames". Question: how much of the key RD partition search's gain
(`planner.key_rd_split`, docs/analysis/key-frame-rdo-2026-11-21.md) can medium and fast
presets get at about zero trials, and which few trials are worth paying for.

## Labels

Diag branch `intra-selector-diag` (never land): `NC_KEYRD_DUMP` prints, per key superblock,
every square node's zero-trial evidence (`KEST`: λ, the key plan's transform estimate,
partition bits, source mean / variance, gradients and their coherence, quadrant variances)
and, with the key search unbounded, each node's coded leaf and split cost (`KRD`). Runs:
`--preset medium`, `planner.key_rd_split=1`, window 3/3 (exhaustive from 64 to 8),
`--key-qp-offset 0`; 56 pool clips (planner-selector's sample: 4 per stratum and family-hash
half, `fit` / `test` split by family), frame 0 and the mid-clip frame each coded as one key,
AV1 / VP9 qindex 80 / 130 / 190, HEVC QP 24 / 30 / 36; 918 runs (5 clips' raw inputs failed
to decode), 12.5 M nodes. Costs are coded J (D + λR, all planes), so the shares below are
of key J, not BD-rate. Scripts: `~/nanocodec-data/scratch/intra-selector/py/` (disposable).
Scoring as planner-selector: recursive regret, a split's cost holding its children at their
best; the offline key planner (`sim.py`) reproduces the dumped plan exactly (agreement 1.0).

## How good the key plan is

RD gain over the plan (% of key J, mean per run): AV1 1.81, HEVC 2.98, VP9 3.20; by QP
AV1 1.28 / 1.75 / 2.41, HEVC 2.51 / 2.91 / 3.52, VP9 2.43 / 2.83 / 4.33; largest on
videocall, low light and bodycam, smallest on "other" and pan.

| codec | agreement | 16 plan leaf / split overturned | 32 | 64 | deepen side | merge side |
|---|---|---|---|---|---|---|
| AV1 | 0.84 | 14 % / 21 % | 17 % / 12 % | 19 % / 6 % | 1.21 | 0.63 |
| HEVC | 0.75 | 12 % / 52 % | 16 % / 25 % | 13 % / 17 % | 1.04 | 2.00 |
| VP9 | 0.73 | 14 % / 49 % | 18 % / 27 % | 24 % / 21 % | 1.05 | 2.26 |

(deepen / merge side: the gain an oracle recovers on the plan's leaves / splits alone.)
HEVC and VP9 over-split (half of their planned 16→8 splits code worse than the 16x16 leaf);
AV1 under-splits. Below a planned leaf the coded cost splits again two levels down in at
most 3.6 % of nodes (key-rdo: window 1/1 holds ~80 % of the 2/2 gain).

It is not a bias a constant fixes: a uniform shift of `intra_bias` (h = ±0.1, ±0.2, -0.4 in
log units) loses on every codec (h -0.1: regret +0.4..1.5× the gain; +0.1: +0.6..0.7×).

## A zero-trial selector

Features known to the key planner before it plans a node's children: log variance / λ,
quadrant variance gain and imbalance, gradient energy / λ, gradient coherence, the
estimate per sample / λ, log λ, size; plus the planner's own margin. Gain-weighted AUC of
"split codes cheaper" (fit half): the planner margin is the strongest (0.62–0.97 by cell),
then source variance (0.62–0.93), gradient energy, estimate per sample, quadrant imbalance;
coherence, λ and the quadrant gain are weak (≤ 0.08 strength).

Held-out share of the RD gain recovered (test half):

| model | AV1 | HEVC | VP9 |
|---|---|---|---|
| linear shift of the bias, Powell on recursive regret (inter selector's form), s 0.75 | 5.4 % | 11.0 % | 11.0 % |
| same, s 1.0 | 3.2 % | 11.8 % | 12.9 % |
| gradient-boosted trees, content features + margin (ceiling) | 16.3 % | 23.5 % | 28.6 % |
| trees, + codec | 16.0 % | 23.4 % | 29.0 % |
| trees, margin + size + codec only | -2.5 % | 0.5 % | -3.5 % |

So a content selector can reach a quarter of the gain at best and the linear form about a
tenth: about 0.1–0.4 % of key J. Not worth a mechanism on its own; not landed.

## The gate: trial only where the estimate nearly tied

Offline, with the margin gate (trial where |ln(bias · leaf / (1000 · split))| < t, plan
elsewhere), share of the RD gain against extra coded area (reached trialled nodes' area over
the frame's, one extra candidate each; the unbounded exhaustive search is ~2 frames):

| t (ratio) | AV1 | HEVC | VP9 |
|---|---|---|---|
| 0.05 (1.05) | 52 % at 0.43 | 51 % at 0.67 | 48 % at 0.54 |
| 0.1 (1.105) | 76 % at 0.80 | 73 % at 1.09 | 73 % at 1.06 |
| 0.2 (1.22) | 93 % at 1.29 | 89 % at 1.62 | 91 % at 1.61 |
| 0.3 (1.35) | 98 % at 1.57 | 95 % at 1.88 | 96 % at 1.86 |

The estimate's errors sit at small margins: that is where the trials go. Landed as
`planner.key_rd_margin` (default off; the key plan's margins in the shared `NodeMargins`,
the same gate as the inter search's `rd_split_margin`).

Real encodes (one-frame fixed-QP ab, key-share 0.15, draw fixedqp-2026-11-21 n 24, medium
with `key_rd_split=1`, window 1/1; BD-J against medium, key CPU from the probe):

| codec | full search | margin 1.05 | 1.10 | 1.20 |
|---|---|---|---|---|
| AV1 | -2.7 [-3.2,-2.3] 2.15x | -1.2 [-1.5,-0.9] 1.36x | -2.0 [-2.3,-1.6] 1.58x | -2.7 [-3.1,-2.3] 1.90x |
| HEVC | -6.6 [-7.3,-5.7] 1.75x | -4.0 [-4.3,-3.7] 1.29x | -5.3 [-5.6,-4.9] 1.42x | -6.0 [-6.5,-5.3] 1.60x |
| VP9 | -5.9 [-6.2,-5.6] 2.11x | -2.5 [-2.9,-2.0] 1.34x | -4.2 [-4.6,-3.6] 1.59x | -5.4 [-5.7,-4.8] 1.89x |
| AV2 | -5.8 [-6.4,-5.3] 2.01x | -2.0 [-2.5,-1.7] 1.26x | -3.4 [-3.9,-3.1] 1.47x | -4.7 [-5.4,-4.2] 1.74x |
| VVC | -8.9 [-9.7,-8.0] 1.90x | -3.8 [-4.2,-3.1] 1.29x | -6.1 [-6.6,-5.3] 1.51x | -7.8 [-8.5,-6.9] 1.76x |

Margin 1.10 keeps 59-80 % of the search's gain for 47-57 % of its extra key time, 1.20
81-100 % for 73-84 %. HEVC with `intra_rd_candidates` 2 on top of margin 1.10: -5.8
[-6.2,-5.2] at 1.69x. Ready for the 0.6.1 screens (keys only, not adoption evidence yet):
`planner.key_rd_split=1 planner.key_rd_margin=1.1` per codec (1.05 on the fast rungs), HEVC
medium `hevc.intra_rd_candidates=2`. Fast / faster rungs: not measured.

## Mode and transform size (one level down)

One-frame fixed-QP ab, key-share 0.15, draw fixedqp-2026-11-21 n 24, medium, BD-J against
medium (key CPU ratio from the probe, coarse):

| change | J | key CPU |
|---|---|---|
| HEVC `intra_rd_candidates` 2 (SATD's top 2 re-ranked by coded cost) | -1.5 [-1.8,-1.2] | ~1.06x |
| HEVC `intra_rd_candidates` 3 | -1.6 [-1.9,-1.3] | ~1.11x |
| HEVC `intra_tu_split_min_cost` 0 | +0.6 | ~1.10x |
| HEVC `intra_nxn_min_cost` 0 | +0.6 | ~1.07x |
| VP9 `intra_rd_candidates` 2 (medium has 3) | +1.0 | ~0.86x |
| VP9 `intra_tx_depth` 3 / 1 (medium 2) | +0.4 / +0.1 | ~0.99x / 0.93x |
| AV1 `keyframe_intra_rd_candidates` 1 / 8 (medium 3) | +0.8 / +0.2 | ~0.91x / 1.16x |
| AV1 `keyframe_angle_deltas` 2 | 0.0 | ~1.00x |

The only cheap prize is HEVC's mode re-rank (medium takes SATD's best alone); VP9 and AV1
medium are at their optimum, and the transform-size gates are past it.

## Intra blocks in P frames

P-frame partitions come from the inter planner (and `split_selector`); intra there is a
leaf-level mode decision, so neither the key plan's margins nor the key gate apply. The
share of P-frame bits that is intra on the pool: not known (no counter in the CLI; a
per-frame intra-area count in the encoder's stats would answer it).
