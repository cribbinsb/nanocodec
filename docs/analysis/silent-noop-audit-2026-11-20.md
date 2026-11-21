# Silent no-op audit: registered parameters per codec (2026-11-20)

The VP9 winner walk did nothing until today: VP9 returned the default `None` from
`CodecBackend::luma_filter`, so the engine took the bilinear ranking and every
`decision.winner_walk_*` setting was inert. The 2026-11-19 screen recorded
"VP9 +0.0 (no effect)", and nobody saw why. Wiring the hook gave VP9 Slow
−1.9 J more and VP9 Default −1.5 % on final-check (`docs/handoff/presets.md`).
This audit looks for every other setting that silently does nothing.

**Method.** At landed head `80dff233b` (gate build, `tools/dev/binary_for.sh`), for
each of the six codecs and each registered parameter (the 270 shared ones and
the codec's own prefix: 1,981 pairs), the scan encoded at the codec's Default
(`medium`). The parameter was set to its range's minimum and then its maximum
(whichever differ from the Default value), and the stream SHA-256 was compared
with Default's. The scan stopped at the first stream that differed.

| pass | clips (720p) | modes | what it tests |
|---|---|---|---|
| 1 | crowd_mot17_09, pan_bdd_00c12bd0, 60 frames | CBR 600 key 30; fixed QP 32 (H.264/HEVC/VVC), qindex 140 (AV1/VP9/AV2) | every pair |
| 2 | crowd_mot17_09, static_otw_homes390, lowlight_pp22_004, 240 frames | CBR 600 key 120; fixed QP | the 213 pass-1 no-ops among the 80 shared parameters that are live in some codecs, plus the first 53 H.264 pairs of the rest (stopped: too slow on a loaded machine) |
| 3 | pass-1 clips | as pass 1 | each gated family with its gate turned on (`decision.final_stage` 1, `decision.cu_price_error_lambdas` 1, winner walk 3, `decision.noise_skip` 2, `vvc.mtt` 1, depth removal 40/50, `chroma.policy` 1, `rate.window_seconds` 1, the effort overlay, `adapt.*` with their tools) |

Scripts, hashes and logs are in `~/nanocodec-data/work/noop-audit/`
(`noop_scan.py`, `scan_default.jsonl`, `pass2_*.jsonl`, `p3_*.jsonl`). No streams
were kept. A "no-op" here means identical streams on these clips and modes.
Parameters that act only on cuts, long keys, repeats or layers can still be
live elsewhere. Pass 2 turned 12 pairs live: H.264 `analysis.zero_sads_from_pre`,
`rate.cadence_*`, `rate.key_level_mean`, `rate.prekey_quiet`, `rate.scene_hint_ratio`;
AV1 and VP9 `refs.second_full` / `refs.second_refine_margin`; AV2 `rate.max_quality_qp`
(static clip only, see Bugs). The other 206 of the 213 mixed pairs stayed no-ops, so the
classification below holds for them on 240-frame static, noisy and crowd content too.

Pass 1 counts (live / no-op / rejected): H.264 144/156/12, HEVC 155/149/12, VVC
167/167/14, AV1 176/174/0, VP9 146/152/12, AV2 167/165/13. Rejected values fail
loudly, with the reason given: `grain.*` outside AV1 ("unknown parameter"),
`resolution.adaptive` without scaled references, `spatial.layers` 3 where the
codec has no spatial scalability, and `vvc.affine` 1 ("second reference: not
with vvc.affine yet"). Most no-ops are category (b), a setting gated off at
Default (below). The items that need a decision are the wiring gaps (c), two
bugs the same scan exposed, and inert lines in shipped presets (one of them
VVC Slow's shape RD).

## Category (c): missing CodecBackend hooks, ranked

Each row is a shared mechanism the engine runs through an optional
`CodecBackend` hook. A codec that keeps the trait default gets an inert or
hollow mechanism. Its parameters are then silent no-ops, and any screen of them
measures nothing. The rows are ranked by likely value: measured gains on codecs
that have the hook, and whether a past rejection measured the hollow version.

| # | Mechanism (parameters) | Hook missing | Codecs lacking it | Measured where wired | Note |
|---|---|---|---|---|---|
| 1 | RD final decision stage (`decision.final_stage`, `final_margin_lambdas`, `final_skip_candidate`, `final_intra_candidate`, `final_intra_gate_lambdas`, `final_intra_energy_ratio`) | `final_coded_cost` (all but H.264), `final_intra_screen` (VVC, AV1, VP9, AV2), `skip_vector` (HEVC, AV1, VP9, AV2) | HEVC, VVC, AV1, VP9, AV2 | H.264 Default: VC-1K worst six J −9.8 %, pool L3 −2.53 % [−3.08, −2.02], final-check −2.68 % (ledger `shared-rd-final-decision-stage-…`) | Pass 3 with the stage on: margin, skip, intra candidate, intra gate and energy ratio are all no-ops in all five codecs, so the stage only drops the DC intra test. The rejected `vp9-final-decision-stage` (fixed QP J +5.6 % "at 0 % instr") and AV1's +2.4 % screened this hollow stage, not the RD stage. HEVC's hooks were built on a branch and rejected (+2.9 J, luma-only coded cost at the largest TBs); they did not land. |
| 2 | Codec skip vector (`skip_vector`): the winner walk's skip exemption, the final stage's skip candidate, `decision.skip_no_refine_lambdas` | `skip_vector` | HEVC, AV1, VP9, AV2 | H.264: walking the skip vector cost J +7.2 % on pool_chad_3_086 fixed QP; keeping it −0.3 % (`decision.rs` comment at the walk) | Without the hook, the winner walk also moves skip and merge winners to sub-sample neighbours: on VP9 Default (walk 3 from 0.25 λ, landed today) and on AV1/AV2/HEVC Slow (walk on). Not measured on those codecs. Cheap to add: HEVC merge index 0, the AV1/VP9/AV2 NEARESTMV/ZEROMV-type cheapest predictor. Needs a check per codec of which vector "skip" means. |
| 3 | Whole-CU pricing (`decision.cu_price_error_lambdas`, `cu_price_intra_lambdas`, `cu_price_intra_ratio`; `cu_price_screen_ratio` also needs `final_intra_screen`) | `begin_cu_trial` / `cu_trial_cost` / `undo_cu_trial` | AV1, VP9, AV2 (screen: VVC too) | HEVC Default `cu_price_intra_lambdas` 0.1: pool L3 −1.00 %, final-check −0.80 %; HEVC Slow L3 −0.64 %; VVC Slow −0.7 J for +87 points (held) | Known for AV1: held because a trial needs writer + CDF snapshots (`av1-whole-cu-pricing-trials-hevc-port`). VP9's coder has no per-block CDF adaptation (frame-level probabilities), so a trial is cheaper to undo there than in AV1. |
| 4 | Noise skip and static skip (`decision.noise_skip`, `noise_skip_chroma_ratio`, `rate.skip_budget*`; `planner.noise_split` is live) | `try_static_skip` (and `static_skip_vector` in VP9, AV2) | VVC, VP9, AV2 | AV1/HEVC/H.264 still cameras: same rate, equal or up to +2 dB better against clean source, 18–23 % fewer instructions; default off, kept in the still-camera profile (`noise-skip-on-still-cameras`, partial) | Matters for the still-camera profile (edge cameras), not for Default. |
| 5 | Frame-level filter effort (`filter.deblock_skip_share`, `deblock_skip_proxy`, `cdef_reuse_proxy`, `cdef_max_reuse`, `adapt.cdef`) | `set_filter_effort`, `coded_skip_share`, `filter_search_evidence` | H.264, HEVC, VVC, VP9, AV2 (AV2 has CDEF; pass 3: `adapt.cdef` live in AV1, no-op in AV2) | AV1: not a large lever (proxy 0.5: 152 → 159 fps, Weighted −17.9 → −15.1; `av1-deblocking-skip-filter-gate`, `av1-frame-level-loop-filter-gate`, both rejected) | Speed lever only, and weak on AV1; low value. AV2 tried its own codec-local CDEF reuse (rejected 2026-11-19, J +3.0 %). |
| 6 | Quantizer work gates (`gate.tx_split_skip_min_quantizer`, `gate.partial_transform_min_quantizer`) | read only by `nc-av1` (`WorkGate`) | HEVC (`hevc.partial_forward`, inter TU-tree trials), VVC, AV2 (transform-split / type search) | AV1 Fast gates 0.78: final-check J +1.10 % worse (`shared-quantizer-work-gates-workgate`) | Shared infrastructure with one consumer. Low value until a gate wins on AV1. |
| 7 | Rectangular inter shapes in the planner (`planner.rect_*`) | `partition_shapes` / `shape_bits` / `write_shape` (`RECTANGULAR_LEAVES`) | HEVC (2NxN/Nx2N PUs), AV1, VP9, AV2 (HORZ/VERT partitions) | VVC: off at Default (`vvc.mtt` 0); with `vvc.mtt` 1 every `planner.rect_*` is live (pass 3); used in VVC Slow (`planner.rect_rd`) | No measurement outside VVC. AV1/VP9/AV2 code only square partitions today. |
| 8 | Compound prediction (`refs.second_compound_margin`) | `supports_compound_reference` / `encode_compound_block` | AV2 (has compound syntax); VP9 (compound needs opposite sign bias on the two references) | AV1 only | H.264/HEVC/VVC would need B slices, which are outside the P-only subset: (a) there. |
| 9 | Zero-out confirmation of large leaves (`planner.zero_out_gate`) | `full_spectrum_log2` + `inter_leaf_cost` | AV1, AV2 (64-point transforms also code only their 32x32 lowest frequencies) | VVC only | Applies only where our AV1/AV2 actually code a 64-point transform (`av1.ceiling_tx_split` splits 64x64 inter blocks at coarse plans). Low. |

Rows 1 and 2 were wired the same day for all five codecs, and both audit bugs fixed
(`docs/handoff/rd-mode-decision.md`, "The stage in HEVC, VVC, AV1, VP9 and AV2").

The light-parameter hooks (`set_light_params` / `use_light_params`: HEVC and AV1
only) are not a silent gap. Engine `light.*` settings work in every codec (VP9
Fast and AV2 Fast/Faster streams change with `effort.regular_fraction`), and a
codec-own `light.<codec>.*` that the codec does not re-read is rejected loudly
("cannot take a non-base frame value").

## Bugs found by the same scan (not no-ops)

1. **AV2's quality ceiling is 8x too fine** (`rate.max_quality_qp`, and the noise
   floor `rate.noise_floor_scale` through the same function).
   `nc-encoder/src/quality_cap.rs` `pixel_step_per_unit` maps `Codec::Av2` to 1
   ("nc-av2 already divides its Q3 step by eight"). nc-av2's own `RATE_MODEL`
   comment says the opposite: after that division its steps are AV1's scale,
   eighths of a sample step, the same mistake once fixed for the reuse term.
   Measured, 30 frames of static_otw_homes390 at 8 Mbps, stream bytes:

   | `rate.max_quality_qp` | 10 (Default) | 24 | 36 | 51 |
   |---|---:|---:|---:|---:|
   | AV1 | 995,298 | 822,420 | 48,464 | 4,508 |
   | AV2 | 865,853 | 865,853 | 842,905 | 117,397 |

   At 600 kbps on crowd and pan, AV2's streams did not change from 0 to 51
   (pass 1); only the 240-frame static clip moved at 51 (pass 2). So on AV2
   the ceiling acts about 18 QP finer than set, and at the Default 10 it never
   binds. A one-line fix would make AV2 match the other codecs (map Av2 like
   Av1/VP9). Better, the shared code would read `RateModel::sample_step_scale`,
   which every backend already supplies, instead of a per-codec match.
2. **The recode contract fails in H.264 and AV1** (`debug.recode_identity` 1 must
   be byte-identical to 0). Fixed QP on crowd_mot17_09 (60 frames):
   - H.264 QP 32: reconstruction differs from frame 58, and the stream by one
     byte.
   - AV1 qindex 140: reconstruction differs from frame 21 on, the stream by
     +731 bytes.

   CBR at 600 kbps matched in both, and HEVC, VVC, VP9 and AV2 matched in every
   mode. A frame re-coded for a buffer overflow (`rate.recode_overflow`) goes
   through the same undo path, so it may carry state from the discarded
   attempt. The existing identity tests run on HEVC and VP9 only
   (`nc-hevc/src/dec/tests.rs`, `nc-vp9/tests/roundtrip.rs`).

## Category (b): gated off at Default (no action)

Families no-op because another setting is off at that codec's Default; each
gate itself is live. Pass 3 turned each gate on to check that its children
become live.

- **Bilinear-ranking settings are dead wherever the exact ranking runs.**
  `decision.skip_no_refine_lambdas`, `h264.subpel_skip_variance`,
  `h264.subpel_skip_lambda` and the backend hook `subpel_skip_variance` are read
  only by the bilinear ranking (`refine_subpel`). HEVC, VVC, AV1 and AV2 return
  `Some` from `luma_filter` unconditionally, so the bilinear path never runs
  there. H.264 and VP9 use it only on the rungs that pin `exact_ranking` 0
  (H.264 Faster/Veryfast; VP9 Fast/Faster/Veryfast and the layered files). So
  several shipped presets set a value that does nothing:
  `decision.skip_no_refine_lambdas` in `hevc/fast.toml` (also as `light.`),
  `hevc/faster.toml`, `vvc/slow.toml`, `av2/slow.toml` and `vp9/slow.toml`.
  Any tuning that searched it there searched noise. These lines can be
  removed, which leaves streams identical.
- `decision.final_*`: stage off outside H.264 (but see (c) 1).
- `decision.winner_walk_square`, `winner_walk_min_lambdas`: walk off in HEVC, VVC,
  AV1 and AV2 Default; live with the walk on (pass 3).
- `decision.subpel_walk_max_noise_sigma`: coordinate walk on only in H.264.
  H.264's fixed-stage settings (`subpel_half_only`, `subpel_quarter_needs_half_move`,
  `subpel_fractional_bias`) are no-ops there for the same reason and live
  with `decision.subpel_walk` 0.
- `decision.intra_directional_dc_ratio`, `intra_direction_bits`: estimate off in
  H.264, VP9 and AV2.
- `decision.cu_price_*`: off except HEVC (`cu_price_intra_lambdas` 0.1);
  `cu_price_screen_ratio` showed no change at 64 (its only non-default value
  tried) even in HEVC.
- `layers.*` (adaptive temporal layers) and `refs.second`: superseded by
  `structure.unified` in HEVC, VVC, AV1, VP9 and AV2; `structure.*` the reverse in
  H.264 (not unified).
- `decision.base_period`: with `base_lambda_scale` 1 and no `nonbase.*` settings
  only the cadence it defines is unused (live in H.264 through its layer cadence).
- `planner.depth_removal_qp_scale`, `adapt.depth`: depth removal thresholds 0 at
  Default; live with 40/50 in HEVC, AV1, VP9, AV2. In VVC they stayed no-ops at
  QP 32 and CBR 600, because removal starts at relative quantizer 0.8 (QP 50 of
  63). At QP 52 VVC Fast's thresholds change the stream (checked directly).
- `planner.rect_*`: `vvc.mtt` 0 (live at 1).
- `chroma.offset`: `chroma.policy` 0 (live at 1).
- `rate.window_tolerance`, `rate.window_keys`: `rate.window_seconds` 0
  (`window_keys` live at 1 in H.264, HEVC, VVC and AV2).
- `adapt.*` children: their tools off; `adapt.rank` family live everywhere with
  `decision.exact_rank_margin` 1.5. `adapt.subpel` is a no-op in HEVC and VP9,
  whose seed gate (`decision.codec_subpel_seed_lambdas`) is 0.
- `effort.*`: overlay off at Default (`effort.regular_fraction` 1).
- `quant.trellis_key` in H.264: H.264 trellises intra macroblocks only with
  `h264.trellis_i16x16` / `h264.trellis_i4x4` (both 0). `quant.trellis_states` in
  VVC: VVC's trellis documents that it does not use it (`nc-vvc/src/trellis.rs`).
- Shared rate control, content-dependent on these clips; no hook is involved, and
  none of these was traced further: `rate.first_key_lookahead(_off)` (HEVC, VVC,
  AV2: their 0.5-buffer cap does not bind); `rate.period_ceiling` (HEVC, VVC);
  `rate.lambda_headroom_octaves` (live only in H.264 and HEVC); `rate.fixed_qp_moving_cadence` (H.264, AV1, VP9); `hold.enable`
  (live only in H.264, on the pass-1 clips; the reason is not known).
- Codec-own families gated by a codec toggle: `h264.rd_refine_*` (`h264.rd_refine` 0),
  `hevc.inter_intra_*` (`hevc.inter_intra_dc_only` 0), `vvc.affine_*`, `vvc.mts_*`,
  `vvc.dep_quant_*` (tool off), `av1.restoration_*`, `av1.keyframe_*` (`av1.keyframe_satd_modes`
  0), `av2.pre_tx_*` (`av2.pre_tx_skip` 0), `av2.skip_mode_candidates` (`av2.skip_mode` 0),
  `av2.tcq_prune_bits` (`av2.tcq` 0), `vp9.nonrd_block_test` (`vp9.nonrd_skip_scale` 0).
  `hevc.forward_zero_columns` is an exact speed-up and identical by design.

## Inert lines in shipped presets

The same test, applied to the presets: for every `name = value` line of each shipped
preset file that differs from the codec's Default (712 lines; 106 `light.` /
`nonbase.` lines left out), the scan encoded the preset with the line put back to
the Default value (pass-1 clips and modes, `preset_lines.py`). 602 lines change
the streams. 110 do not; `~/nanocodec-data/work/noop-audit/preset_inert.txt`
lists them. Many act only on content these clips lack: key and still
settings such as `rate.key_still_boost`, `rate.fixed_qp_moving_key` and
`rate.period_key_saving`, and the adaptive-layer limits. The lines below are
dead by construction, because the mechanism they tune is off or unwired in that
preset:

| preset(s) | inert line | why |
|---|---|---|
| `vvc/slow` | `planner.rect_rd = 1` ("rectangular shapes confirmed by coded cost") | no rectangular shapes without `vvc.mtt` (0 in Slow; `mtt` 2 screened +4.6 %). Checked at QP 27/37/47: identical; `vvc.mtt` 1 changes them. The −0.6 % credited to "8x8 inter CUs with shape RD" in `f213f834a` is `planner.inter_min_log2` 3 alone. The AV1 Slow pick also noted "`rect_rd` and depth removal: no effect" (AV1 has no shapes hook, (c) 7). |
| `hevc/faster`, `vvc/slow`, `av2/slow`, `vp9/slow` | `decision.skip_no_refine_lambdas` | bilinear ranking only (above) |
| `vp9/faster`, `vp9/veryfast`, `vp9/l2-fast`, `vp9/l3-fast` | `decision.codec_subpel_seed_lambdas` | exact-filter ranking only; these rungs pin `vp9.exact_ranking` 0 |
| `hevc/fast`, `vvc/fast` | `adapt.rank = 1` | adapts `decision.exact_rank_margin` only when above 1; it is 0 there |
| `hevc/fast`, `hevc/faster`, `vvc/fast` | `adapt.probe = 1` | adapts `decision.codec_fullpel_probe_min_lambdas` only when above 0; it is 0 there |
| `av1/fast`, `av1/faster`, `av1/veryfast`, `av2/fast`, `av2/faster` | `decision.subpel_walk_max_noise_sigma` | `decision.subpel_walk` 0 there |
| `vp9/fast`, `vp9/faster`, `vp9/veryfast`, `vp9/l2-fast`, `vp9/l3-fast`, `h264/fast`, `h264/faster`, `h264/veryfast` | `quant.trellis_key`, `quant.trellis_lambda_scale` | `quant.trellis` 0 there |
| `h264/faster`, `h264/veryfast` | `decision.winner_walk_steps`, `subpel_walk`, `refine_cheap_vector`, `codec_fullpel_probe` | exact-filter features; `h264.exact_ranking` 0 there (pins kept on purpose when the Default changed) |

These lines leave streams identical whether present or not, so removing them
is an exact change. Their cost is that they mislead: a tuner or a reader takes
them for live levers.

`nanocodec params --codec C --preset P` accepts `--preset` and prints the
Default's values, not the rung's. A check of a rung's settings through it
reads the wrong numbers.

## Category (a): the codec lacks the tool

- H.264: every `planner.*` (macroblocks are fixed 16x16; sub-partitions are
  decided in the backend) and `decision.codec_subpel_seed_lambdas_large` (no
  block of 32x32 or more). `adapt.motion` is also a no-op for H.264 on every
  clip, which is not traced; `analysis.zero_sads_from_pre` is live on the
  240-frame clips.
- `planner.zero_out_gate` in H.264 and HEVC: no 64-point transform.
- Compound outside AV1 for H.264/HEVC/VVC: B slices are outside the P-only subset.
- `grain.*` outside AV1 and `spatial.*` outside AV1/VP9: rejected loudly.

## Hook coverage at `80dff233b`

Optional `CodecBackend` methods each backend overrides (Y). Methods every
backend implements are left out.

| hook | H.264 | HEVC | VVC | AV1 | VP9 | AV2 |
|---|---|---|---|---|---|---|
| `skip_vector` | Y | – | Y | – | – | – |
| `final_coded_cost` | Y | – | – | – | – | – |
| `final_intra_screen` | Y | Y | – | – | – | – |
| `begin_cu_trial` (+ cost/undo/end) | – | Y | Y | – | – | – |
| `try_static_skip` | Y | Y | – | Y | – | – |
| `static_skip_vector` | (skip_vector) | Y | (skip_vector) | Y | – | – |
| `set_filter_effort`, `coded_skip_share`, `filter_search_evidence` | – | – | – | Y | – | – |
| `partition_shapes`, `shape_bits`, `write_shape` | – | – | Y | – | – | – |
| `inter_leaf_cost` | – | – | Y | – | – | – |
| `supports_compound_reference`, `encode_compound_block` | – | – | – | Y | – | – |
| `set_light_params`, `use_light_params` | – | Y | – | Y | – | – |
| `set_refresh` (intra refresh) | Y | Y | Y | Y | – | – |
| `copied_reference` | Y | – | – | Y | – | – |
| `set_noise_sigma` | – | – | – | Y | – | Y |
| `block_qp_offset` | – | – | Y | Y | Y | – |

A new optional hook should come with a check that names the codecs keeping its
default. A cheap one: `nanocodec params` could list, per codec, the parameters
whose mechanism that codec does not wire, so a screen of an inert setting is
visible before it runs.
