# nn dataset: shard schema (format version 7)

The labelled dataset that learned encoder models train and are judged on
(nn-lead's per-superblock policy network, the probe agent's trial gate and
controller oracle). The encoder writes it: `nanocodec encode ... --label-dump DIR`
(`crates/nc-encoder/src/labels.rs`, `crates/nc-cli/src/label_dump.rs`). It is
read only through `eval/nceval/nndata` (no private parsers).

## Guarantees

- **The stream is the preset's own.** The dump codes extra candidates and
  codes every candidate without a bound or early exit, but it chooses
  exactly as the preset does, among the candidates the preset would code.
  Every shard is generated next to a plain encode, and the two streams must
  be byte-identical, or the shard is quarantined. First shard: HEVC slow and
  medium, identical.
- **Only the kept coding is recorded.** When rate control re-codes a frame,
  the records of the discarded coding are dropped.
- **Costs are the engine's own J.** That is distortion plus λ·rate, all
  planes, with entropy-coded bits (`CodecBackend::node_trial_cost`), in the
  units the RD search compares.

## Layout

A shard is one encode, as one directory:

```
labels.json      {"format_version": 6, "codec", "candidate_slots": 6, "patch_side": 32,
                  "tools": ["preset", ...], "tables": {table: [{"name", "dtype", "shape"}]}}
meta.json        provenance (written by nceval.nndata; see below)
frame/*.npy      one row per coded frame
sb/*.npy         one row per superblock (CTU), coding order
node/*.npy       one row per node the RD partition search decided
leaf/*.npy       one row per leaf of the final coding
```

Each column is a plain `.npy` array whose first axis is the row. The columns
are declared once, in `crates/nc-cli/src/label_columns.rs`. Each shard's
`labels.json` lists them with dtype and row shape (-1: the tool-slot count).
The reader checks the arrays against that list, and checks the list against
the columns it needs (`eval/nceval/nndata/schema.py` `COLUMNS`). Adding a
column is one line in the writer and the value in `nc_encoder::labels`; the
reader loads it without a change. Format 4 shards (v0) have no list; they
load as the format-4 set. Format 5 shards load without the format-6 columns
(the trial summaries and search positions, marked (v6) below). Key:
`(clip, start frame, frame, sb x, sb y)`. The same clip coded at two presets
has the same keys, for distillation.

### frame/

| column | dtype | meaning |
|---|---|---|
| index | u4 | input frame index |
| key | u1 | 1: key frame |
| temporal_layer | u1 | temporal layer |
| quantizer | u4 | the frame's planned quantizer (codec units: HEVC QP, AV1 qindex); a fixed-QP key is coded finer by the codec's key offset (HEVC at a 10 s period: 15 QP), which this column does not show |
| lambda | f4 | the backend's mode-decision λ at the quantizer it codes (`CodecBackend::lambda`): the scale of the frame's J; use it, not `quantizer`, for the rate-distortion trade-off |
| noise_sigma | f4 | pre-analysis noise σ (luma samples) |
| width, height | u4 | coded size |
| superblock_log2, min_block_log2 | u1 | the codec's block-size limits |
| key_distance | u4 | frames since the last key frame (v5) |
| rate | f4 [4] | rate control at the frame's start: buffer fullness, the frame's bit budget, the level quantizer, the cadence offset (NaN at fixed QP) (v5) |

### sb/

All values are known before the superblock is coded, except `cost` and `ns`.

| column | dtype, shape | meaning |
|---|---|---|
| frame, x, y | u4 | input frame index, luma position |
| inside | u1 | wholly inside the coded picture |
| cost | f4 | coded J of the superblock; NaN at picture edges and where the backend has no node trials (H.264) |
| ns | u8 | wall time of the dump's coding of it (with the full candidate search, so slower than the preset; use tool_ns[0] for the preset's time), ns |
| luma_mean, luma_variance | f4 | source luma |
| gradient | f4 [2] | mean abs horizontal, vertical gradient |
| coherence | f4 | gradient structure-tensor coherence |
| block_variance | f4 [2] | mean and std of the 8x8 variances (pre-analysis) |
| static_share | f4 | share of 8x8 cells pre-analysis marks static |
| still | f4 [2] | share of cells marked still, mean still run (frames, cap 255) |
| zero_sad | f4 | zero-vector SAD per sample against the last source frame ("changed since last frame") |
| best_sad | f4 | motion search's best 8x8 SAD per sample |
| motion | f4 [4] | 8x8 field: mean mv x, y and std x, y (luma samples) |
| search_vector | i2 [2] | the superblock's search vector (quarter samples) |
| residual_energy | f4 | mean squared residual at the search vector |
| second_residual | f4 [2] | against the frame's second reference where it carries one (else 0): mean squared residual at the search vector scaled to its distance, and at the zero vector (v7) |
| patch | u1 [32,32] | source luma, averaged down from the superblock (64: 2x2 means) |
| residual | i2 [32,32] | source minus reference at the whole-sample search vector, averaged the same way; 0 on key frames |
| block_sad | f4 [8,8] | the motion search's best 8x8 SAD per sample, pooled to an 8x8 grid over the superblock (v5) |
| block_mv | f4 [8,8,2] | its 8x8 vectors (luma samples), pooled the same way (v5) |
| cell_stats | u2 [8,8,15] | per 8x8 cell, from the full-resolution source and the residual (`nc-analysis` `cell_stats.rs`, the routine a model in deployment uses): 0 mean, 1 variance, 2 gradient energy (mean (dx²+dy²)/4, central differences), 3-10 orientation histogram (8 unsigned bins from horizontal, each the summed abs(dx)+abs(dy) of its samples / 4), 11 zero-vector SAD, 12 SAD at the superblock's search vector, 13 residual mean square there, 14 noise floor (the smallest 4x4-quarter variance). u16, saturating. Reference values are 0 on key frames. 128-pixel superblocks pool 2x2 cells; smaller ones fill the top-left (v5). 8x8x15x2 bytes = 1.9 kB per superblock |
| tool_cost | f4 [T] | per tool slot (labels.json `tools`: slot 0 `preset`, then each `--label-tool`): the superblock's coded J coded with that tool flipped from the preset's setting, everything else (the rest of the frame, the state before it) as the preset; NaN where unpriced |
| tool_work | u8 [T] | as tool_ns, in `cand_work` units (v0.2) |
| tool_positions | u4 [T] | as tool_ns, in `cand_positions` units (v6) |
| tool_ns | u8 [T] | as tool_cost, wall time; slot 0 is the preset's real time (the search as the preset runs it, unlike `ns`) |

Pre-analysis and motion statistics are 0 on key frames and on the first
frame.

### node/

One row per node that the RD partition search decided (`rd_split.rs`):
key frames under `planner.key_rd_split`, inter frames under
`planner.rd_split`. At medium, HEVC and AV1 search only their key frames;
their slow rungs also search inter frames. A forced edge split has no row,
and its children are roots (parent -1).

| column | dtype, shape | meaning |
|---|---|---|
| frame, x, y, log2 | | node |
| sb | u4 | row of its superblock in `sb/` |
| parent | i4 | row of its parent node in `node/` (-1: root); parents come before children |
| planned | u1 | the slot the plan proposed, 255 for nodes below a planned leaf |
| chosen | u1 | the slot chosen (always the cheapest eligible one) |
| eligible | u1 | bit s: the preset would have coded slot s |
| plan_margin | f4 | the planner's margin: its rejected over its chosen cost estimate, at least 1 (1 where unknown, NaN below a planned leaf); the margin gates compare it with `planner.rd_split_margin` / `key_rd_margin` |
| gates | u1 | the preset's gates' verdicts at the node (what each would have decided, while the dump codes every candidate), bits: 1 the margin gate lets a planned leaf try the split, 2 a planned split try the leaf, 4 the context gate lets a planned leaf try the split, 8 the context gate lets a planned split try the leaf, 16 (v5) the residual gate lets a planned leaf try the split (format 4: bit 4 was context and residual together); a gate the preset has off allows |
| plan_costs | f4 [2] | the planner's compared leaf and split costs (bias·leaf, 1000·split rate-scaled, as it compares them; NaN where it did not weigh both) (v5) |
| final_context | i1 [2] | the depth context from the dump's own map of final leaves, on every rung: left/above neighbours' and co-located leaf log2 minus the node's, from the superblocks coded before (v5) |
| search | f4 [4] | the motion search at the node's size (16/32/64 aggregated fields): vector x, y (samples), whole SAD and split SAD per sample; 0 elsewhere and on key frames (v5) |
| pre | f4 [5] | pre-analysis over the node's 8x8 cells: static share, still share, zero-vector SAD per sample, mean and max 8x8 variance (v5) |
| first | f4 [3] | what the first candidate (the plan) coded, known when it is coded: its leaves' count, their skip share (skip CUs and static skips), their intra share (v5) |
| abandoned | u1 | bit s: the preset abandoned slot s at its bound (the best cost so far); its `cost` holds that bound, a lower bound of its cost, and `distortion`/`bits` are NaN (v0.2; earlier versions coded it on in full, which left state behind on AV2 slow) |
| cand_work | u8 [6] | deterministic work of coding each slot (its subtree): luma samples of every leaf decided and coded plus every extra coding by the key-frame leaf search. The same on every machine and run (repeat-checked); weights against cycles are fitted separately (v0.2) |
| cand_positions | u4 [6] | the second deterministic work counter of each slot's subtree, every coding counted as in `cand_work`: block predictions the leaf decisions measured, i.e. the motion search's positions (`decision.rs` `Measurer`: one per vector measured, whole-sample, sub-sample or a compound half, whatever its kernel; the planner's own measurements before the search are not counted). 0 where no leaf decision ran (key frames) (v6) |
| nonzero | u2 [6] | what each slot's coding holds, read where its cost is (`CodecBackend::node_trial_summary`, `NodeTrialSummary`): its non-zero coefficient levels, all planes. 65535 where the slot was not coded (or abandoned) or the backend keeps no summary (v6) |
| tx_blocks | u2 [6] | as `nonzero`: its transform blocks with coefficients, each plane's block counted once (v6) |
| max_tx | u1 [6] | as `nonzero`: log2 of the longer side of its largest coded luma transform; 0 where it codes no luma coefficient, 255 where unknown (v6) |
| distortion, bits | f4 [6] | each slot's cost in its parts (`CodecBackend::node_trial_parts`): cost = distortion + λ·bits, distortion the codec's weighted SSE over all planes (v5) |
| has_inputs | u1 | 1 where `inputs` holds the planner's value-network inputs (0: key frames, static nodes, nodes below a planned leaf, nodes the planner did not weigh both ways) |
| inputs | u1 [11] | the trial value network's integer inputs at the node, exactly the bytes the planner computes (`crates/nc-encoder/src/trial_inputs.rs`, `TrialInputs`): train/serve parity |
| context | i1 [2] | the depth context (`depth_context.rs`): left/above neighbours' and co-located leaf log2 minus the node's (0 unknown; recorded only where the preset keeps the map, i.e. with `planner.rd_split_context`) |
| cost | f4 [6] | coded J of each slot, coded in full; NaN where the slot is not a candidate |
| ns | u8 [6] | wall time of coding each slot, ns |

The slots are 0 leaf, 1 quad split, then 2 horizontal, 3 vertical,
4 horizontal triple and 5 vertical triple (`PartitionShape::ALL`). The first
candidate's values (what a gate reads after coding the plan) are at slot
`planned`; below a planned leaf (`planned` 255) the first candidate is the
leaf, slot 0.

#### Trial summaries and work counters (v6)

- **What `nonzero`, `tx_blocks` and `max_tx` count.** The coding in place when
  the slot's cost is read: its kept leaves only, never an undone candidate's.
  The backends compute them in the same call a learned gate would make in
  deployment (no train/serve skew):
  - HEVC and VVC read them back from the syntax recorded since the trial's
    mark: the CU items, their transform units and levels. An undo truncates
    those, so nothing leaks.
  - AV1 writes its symbols as it goes and keeps no syntax record. So it counts
    each transform block as its coefficients are written. Each trial notes its
    parent's count and starts from zero; an undo starts it again; a kept
    coding carries its count; the trial's end adds its count to its parent's.
  - Counted: every plane's coded block, as each codec records it (each
    transform unit's `cbf` planes in HEVC and VVC, where a VVC ISP part is one
    luma block). AV1 64-point transforms count their coded 32x32 levels.
  - H.264 has no node trials, so it has no summary (65535 / 255). Its `node/`
    table is empty anyway.
  - AV2 and VP9 have none yet: their node trials are being rewritten.
- **The cost proxy.** `cand_work` (leaf samples) and `cand_positions` count
  every coding of the subtree, undone ones included, so between them they
  see per-leaf effort that a samples-only counter misses: the motion search's
  depth differs between rungs. `tx_blocks` adds transform-block counts, but
  only of the kept coding. Fit them as separate regressors against cycles;
  they are not combined in the dump.
- **H.264** gets the engine's counter per leaf (`leaf/positions`, filled for
  every codec); it has no tool trials, so its `sb/tool_positions` stay 0.

The candidates are the preset's candidates plus those its gates leave out:
the same depth window with no margin, context, residual or motion gate, and
every legal shape where a leaf is tried. Static nodes keep the plan.

#### leaf/ (v5)

One row per leaf of the final coding (trial codings are undone with the
engine's state, so only the kept leaves remain).

| column | dtype, shape | meaning |
|---|---|---|
| frame, x, y | u4 | frame and position |
| sb | u4 | row of its superblock in `sb/` |
| log2 | u1 [2] | log2 width, height |
| mode | u1 | 0 intra, 1 inter (primary), 2 inter (secondary), 3 compound, 4 static skip |
| skip | u1 | the backend coded it as a skip CU |
| mv | i2 [2] | vector into the primary reference (quarter samples) |
| qp_offset | i2 | block quantizer offset |
| positions | u2 | block predictions its leaf decision measured (`cand_positions` units; 0 on key frames and static skips) (v6) |

### Medium's inter search

Medium codes no inter RD partition search, so every medium job has a twin
with `planner.rd_split=1`, `planner.rd_split_context=0` and
`planner.rd_split_margin=0` (meta.json `job.params`; directory
`<codec>-medium-<rate>+planner.rd_split=1+...`), whose node/ rows cover
medium's inter frames with the gates off. Its stream is that configuration's,
not medium's.

### Tools (sb/tool_cost)

Each superblock inside the picture is coded, before its real coding, once
per tool slot as an undone trial: slot 0 exactly as the preset (its J equals
`cost`; the validator checks it), and slot t with tool t flipped from the
preset's value (on where the preset has it off, off where on), so
`tool_cost[t] - tool_cost[0]` is what running (or dropping) that tool on that
superblock alone changes, and `tool_ns[t] - tool_ns[0]` its time. The tools
(`nceval.nndata.generate.TOOL_CATALOGUE`) are `planner.rd_split`,
`planner.key_rd_split`, `quant.trellis`, `decision.winner_walk_steps`,
`decision.final_stage`, `decision.intra_rd_modes`,
`decision.intra_rd_transforms`. Labelled on the medium and slowest rungs.
Frame-level only, so not labelled per superblock: the second reference
(`refs.second`), the loop filters' searches (SAO, CDEF), `quant.trellis_states`
and `quant.trellis_key`. AV1's intra_rd tools are not labelled yet: their
trials leave state behind (the dump's stream then differs).

## Reading the costs and times

- **Split costs and times cover the whole subtree.** A split's children are
  searched by the same full rule, and its `ns` includes their searches. The
  child rows (`parent`) carry their own costs and times, so a split's own
  time is its `ns` minus its children's chosen-candidate times.
- **A superblock's `cost` equals its root node's chosen cost exactly**
  (first shard: max difference 0).
- **The causal context is not stored.** The left, above and co-located
  final depths, and the co-located superblock's previous decision and cost,
  are derived by join: every superblock and node of every frame is in the
  shard. `nceval.nndata` provides the join.

## Timing caveats

`ns` is wall time from a monotonic clock on whatever box made the shard. A
shared or loaded box makes it noisy: it is good for the order of magnitude
of a budget oracle, not for precise cost models. nn-lead takes precise
timing from its own quiet cloud runs. The noise is measured on a repeat
encode of a shard and stated in each version's card.

## Provenance (meta.json)

meta.json records:
- commit and binary hash;
- codec, preset, rate mode, QP or kbps, params;
- clip name, source path, dataset or camera, content class, grade,
  source kbps and the source file's sha256;
- start frame and frame count.

A version directory (`nn-data/vN/`) holds `manifest.json`, which lists
every shard's content hash, and a dataset card. A released version is never
edited: changes make vN+1.

## Known limits (v1)

- **Not recorded yet:**
  - bits per leaf and per superblock;
  - the trial summaries of AV2 and VP9, and of H.264 leaves (no trial scope);
  - effort spent inside a codec's own searches (VVC's affine, ISP and MTS
    trials; AV1's transform-type and intra searches) beyond what the leaf
    samples and the engine's search positions see;
  - instruction counts.
- **H.264** has no node trials, so it gets `frame/` and `sb/` only, with no
  costs.
- **Spatial-layer encodes** are not labelled.
