# Shared vs codec-specific code: audit (2026-11-15)

Owner's direction (2026-11-15): "Check our methodology for including things in
shared code vs codec specific; is there anything more we can make shared? We
ought to take into account non-shared cost when developing, even at equal or
slightly worse we should favour doing things in a common way as it's likely to
be more robust and less over-fitting."

Scope: the encoder side of the six codec crates (`nc-h264`, `nc-hevc`,
`nc-vvc`, `nc-av1`, `nc-vp9`, `nc-av2`) at `449f45e8`, against the shared
engine (`nc-encoder`, `nc-analysis`). Decoders are out of scope except where an
encoder helper duplicates its own decoder's. Line references are at that
commit. The inventory is from reading the code (three parallel read-only
passes: quantisation and cost models; mode decision and search; loop filters,
references and plumbing), plus a parameter matrix extracted mechanically from
every backend's `DEFAULT_PARAMS` / `FIXED_QP_PARAMS` and `presets/*.toml`.
"Exact" below means byte-identical streams for every Default and preset,
which a refactor proves with `tools/dev/preset_identity.py` before it lands.

## 1. Summary

- **The architecture holds at the top level.** Partition planning, motion
  search, the intra gates, rate control, structure and references are engine
  code; backends supply bits, leaf costs and syntax. The one backend-owned
  split decision is H.264's sub-macroblock split (`nc-h264/src/partition.rs:148`).
- **Below the `CodecBackend` line, the same mechanisms are written 2-6 times.**
  The backend files are large (AV1 `backend.rs` 5955 lines, VP9 3905, HEVC
  3278, H.264 2613, VVC 2278, AV2 2039) and most of their non-syntax code is
  one of ~25 patterns that recur across codecs: trellis glue, dead zones,
  trailing-one shaving, residual-drop RD, transform-split gates, key-frame
  quantisers, chroma offsets and weights, filter search cadence, reference
  slot tables. Roughly 1,000-1,300 lines are exact duplicates or
  near-duplicates that could be one shared function each (ranked list, §3).
- **The same job carries different parameter names per codec** (§4.2: 22 jobs,
  e.g. quantiser rounding in three unit systems under six names), and **shared
  engine parameters carry per-codec values** that read as per-codec tuning
  rather than codec reasons (§4.1). Several engine defaults are overridden to
  the same value by all six codecs: the engine default is simply stale.
- **The process did not push towards sharing.** The only rule is "port
  speed-ups across codecs" (CLAUDE.md), which ports a copy rather than
  sharing the code; its tracking table (`docs/handoff/cross-codec-speedups.md`)
  has 21 "not checked" cells out of 102. Nothing in the briefs, landing
  checks or adoption rule counts the cost of a second implementation, and
  the adoption rule (J at the bootstrap bound) rewards per-codec constants
  found by screening each codec separately, which is the overfitting the
  owner warns against (§5).
- **Six correctness findings** surfaced on the way (§6); two are confirmed
  dead/ignored code paths, the others need a check by their owners.

## 2. Method notes

- Patterns were classed as: *exact duplicate* (same arithmetic, sharing is
  byte-identical), *near-duplicate* (same idea, different constants or
  comparators; sharing is exact only if each caller keeps its own constants,
  else stream-changing), *one-codec heuristic* (would apply elsewhere).
- Estimates of lines removed are rough (±50 %); they count the copies that
  would go, minus the shared version.
- Integer widths: the integer-width encoder work narrowed AV1/VP9 and (landing
  now) H.264 quant/levels to i16; its HEVC/VVC narrowing is shelved on
  `feature/int-width-enc7-hevc` (Jetson regression). Shared quant helpers below
  are written generic over the level type (or i16-ready) so that branch can
  adopt them; the trellis `Candidates.levels` stay wide
  (`docs/analysis/integer-width-audit-2026-11-14.md`).

## 3. Ranked candidates to share

Rank = duplication x lines x low risk. Codecs: 2=H.264 H=HEVC V=VVC 1=AV1
9=VP9 A=AV2.

| # | What | Where (file:line) | Codecs | Lines | Risk | Kind |
|---|---|---|---|---|---|---|
| 1 | **Consensus engine defaults**: engine default differs from a value every codec sets (`quant.trellis` 1, `rate.cadence_motion_gate` 1, `rate.cadence_startup_period` 8, `rate.period_ceiling` 1, `quant.trellis_states` 1 [VVC ignores it]); majority values (`rate.key_still_boost` 6, `rate.cadence_startup_base_boost` 4) | every `backend.rs` `DEFAULT_PARAMS`; `nc-encoder/src/params.rs` | 6 | ~30 table rows | low | exact |
| 2 | Trellis settings parsed from `QuantParams` (`lambda_scale`, `chroma_scale`, `key_frames`, `max_states`, key gate) | 2 `backend.rs:374`, H `backend.rs:527`, V `trellis.rs:598`, 1 `backend.rs:541`, 9 `backend.rs:733`, A `backend.rs:160` | 6 | ~70 | low | exact |
| 3 | Trellis wrapper (last level, `Candidates::around`, `search.quantize`, signed write-back); per codec only `nearest` and the distortion closure differ | 2 `residual_trellis.rs:242`, H `residual_trellis.rs:436`, 1 `coeff_trellis.rs:55`, 9 `coeff_trellis.rs:226`, A `coeff/trellis.rs:117`, V `trellis.rs:547` | 6 | ~120 | low-med (hot) | exact |
| 4 | Planned-step dead zone for λ headroom: `ceil((1-r)·sqrt(scale)).max(1)` then zero smaller levels | 2 `macroblock.rs:499,1224`, H `backend.rs:242,2725`, 1 `backend.rs:4706,5105`, 9 `transform.rs:222,233` | 4 (V, A lack it) | ~60 | low | exact |
| 5 | Key-frame quantiser at fixed QP: the `begin_frame` branch, and "qindex whose step ≥ scale·step(q)" search | 2 `:2312`, H `:1447`, V `:1102`, 1 `:1672` + `scaled_qindex:5182`, 9 `:1466` + `scaled_qindex:3612`, A `:1239` + `key_qindex:147` | 6 | ~60 | low | exact (keep each codec's offset); unifying the offset rule is stream-changing |
| 6 | Isolated trailing ±1 shave ("SVT shave_coeff") | H `quant.rs:119`, V `recon.rs:712`, 9 `coeff.rs:446`, 1 `mds3.rs:29` (zero-run off by one, may drop DC) | 4 (+A's RD trim) | ~40 | low | exact for H/V/9; AV1 stream-changing |
| 7 | HEVC re-implements nc-dsp's `HevcQuantizer` arithmetic (`QUANT_SCALE`, `quant_params`, zero limits); VVC already uses nc-dsp's | H `quant.rs:25-97` vs `nc-dsp/src/codec_txfm/mod.rs:168`; also `step_sse = 2^((qp-4)/3)` in H `backend.rs:525`, V `trellis.rs:47`, `dep_quant.rs:32` | 2 | ~50 | low | exact |
| 8 | VP9 / AV2 reference store (`HeldReference`, held/commit spatial, scale checks) | 9 `backend.rs:440-515,1103,1153` vs 1 `reference_store.rs:31-153`, `backend.rs:1319` | 2 | ~70 | med (SVC) | exact |
| 9 | Chroma policy mapping: `closest_index` over the codec's chroma QP table / delta step | 2 `chroma_offsets.rs:65`, H `backend.rs:2665`, 1 `:5552`, 9 `:3588`, A `:1514` | 5 | ~30 | low | exact |
| 10 | Adaptive chroma error weight and gap-driven chroma delta (VP9 copied AV2) | 9 `backend.rs:2466,2490,1692,1765`, A `chroma_weight.rs:53`, `backend.rs:700,1405,1480` | 2 | ~45 | low | exact |
| 11 | Intra-fallback vector stashed per backend (`Option<(pos, Mv)>`, take + position check) | engine `encoder.rs:2529`; 2 `backend.rs:346,2137`, H `:472,852`, 1 `:484,1621`, V `:391,550` | 4 | ~40 | low | exact |
| 12 | Filter search cadence (interval, key/POC reuse, skip unchanged CTBs keep old class) | H `backend.rs:1556` (hard-coded 4), V `decide_sao:1964`, V `alf_search.rs`, 1 `cdef_frame_choice:2795`; none for A CDEF/CCSO, AV1 restoration | 4 (3 gaps) | ~50 | med | exact as helper; filling gaps is stream-changing |
| 13 | Quadrant residual-concentration gate before a transform split | H `backend/rd.rs:117`, 9 `backend.rs:2600`, 1 `backend.rs:3189` (`>` vs `>=`) | 3 | ~45 | low | exact with own comparator |
| 14 | Quantisation-group QP state (HEVC/VVC) and per-block qindex (AV1/AV2) | H `quant_group.rs:23`, V `quant_group.rs:41`; 1 `backend/delta_q.rs:71`, A `delta_q.rs:79` | 2+2 | ~60 | low | exact |
| 15 | Block SSE wrappers (`dsp.variance(..).sse` over two regions) | 9 `:2449,2510`, H `backend/rd.rs:98`, 2 `residual_rd.rs:108`, A `block_coder.rs:3988`, V `backend.rs:1690` | 5 | ~40 | low | exact |
| 16 | Still-skip (`try_static_skip`, `static_skip_vector`) | 1 `backend.rs:1529,1514`, H `backend.rs:1337,1327`; V/9/A implement neither, so `decision.noise_skip` is inert there | 2 (3 gaps) | ~25 | low | exact as engine default; filling gaps stream-changing |
| 17 | Angle-delta coarse-to-fine search, and hand-rolled SATD shortlists bypassing `intra_candidates::shortlist_intra` (only VVC uses it) | 1 `backend.rs:3848,3809`, `p_intra.rs:117`, A `block_coder.rs:1577` | 2-3 | ~50 | low | exact (ties aside) |
| 18 | CDEF: AV1 keeps its own strength-set choice; only AV2 uses the engine's `select_strength_sets`; `adjust_strength` and `filtered_units` duplicated | 1 `cdef.rs:98-290,482,618`, A `cdef.rs:522,703,1000` | 2 | ~50 (+~190 if AV1 adopts the engine search) | med | helpers exact; search stream-changing |
| 19 | Early-merge shortcut (full-pel, SSE ≤ k·λ·area, first merge = vector) | H `backend.rs:1273`, V `backend.rs:1026` | 2 (applies to all) | ~15 | low | exact |
| 20 | Inter-frame intra search tiers (DC only / coarse / marked / full) under 6 names | 2 `:118,561`, 9 `:254,2272`, H `:572-590`, V `inter_intra.rs`, 1 `p_intra.rs:41`, A | 6 | ~60 | med | enum exact; behaviour stream-changing |
| 21 | Transform split/type early exits ("too few levels", "unsplit coded nothing", "pre-tx-skipped") with off-by-one conventions | 9 `:2576,2788,2789`, A `:2679,2838`, V `:1781`, H `inter_tree.rs:140`, 1 `:3298` | 5 | ~50 | low | exact as helpers |
| 22 | Reference choice → list index; per-slot refresh bookkeeping | 2 `:535`, H `:368`, V `:1481`, A `:478`; slot tables in all six | 4-6 | ~70 | low | exact |
| 23 | Two SAO edge-offset searches (HEVC rounded mean vs VVC RD per magnitude); CCSO fit uses the same arithmetic | H `sao.rs:222-440`, V `sao.rs:121-265`, A `ccso_search.rs:101` | 2-3 | ~250 | med | stream-changing |
| 24 | λ = k·step² in two notations (H.26x `C·2^((QP-12)/3)`; AV `k·(step/8)²`); k ≈ 0.09 (H, V, 9), 0.135 (2, 1, A) | 2 `:257`, H `:226`, V `:1115`, 1 `:1447`, 9 `:1209`, A `:1119` | 6 | ~15 | med | stream-changing unless operand order kept |
| 25 | Chroma λ (λ_C = λ_Y·(step_C/step_Y)²) in four arithmetics; HEVC scales only with the policy on | 1 `:4652`, 9 `:2914`, 2 `:1274`, H `:2346`, V `recon.rs:116`, A `block_coder.rs:3820` | 6 | ~25 | med | stream-changing |
| 26 | Residual-drop RD (skip vs coded, SVT skip bias), luma-only or all planes | 1 `:4739`, 9 `:2369`, 2 `residual_rd.rs:126`, H `inter_tree.rs:159`, V `:1668`, A | 6 | low value | med | stream-changing |
| 27 | AV2's legacy standalone encoder (`stream::Encoder`, `inter.rs` 942 lines, `search.rs` 196 lines with an unregistered second `"av2"` param group, encode half of `intra.rs`), used only by AV2 tests and examples | `nc-av2/src/{stream,inter,search,intra}.rs` | 1 | ~1,300 | med (tests need streams from `Av2Backend`) | exact (no shipped path) |
| 28 | Engine `OWN_PICTURE_PIPELINE` / `encode_picture` and their guards: no implementer left | `nc-encoder/src/backend.rs:169-182`, `encoder.rs:328,380,938,1165` | engine | ~40 | low | exact |

One-codec heuristics that would apply elsewhere (each a stream-changing
screen; the code gives no codec reason unless quoted):

- AV2's RD trailing-one trim (`coeff_trim.rs:73`, measured pan -3.2 %) vs the
  fixed-gap shave of #6 in HEVC/VVC/VP9/AV1.
- AV2's `inter_split_min_nonzero` (J -7.4 %, dCPU -12.8 %): no AV1/HEVC
  equivalent.
- VVC's intra-dense picture gate (`inter_intra.rs`): its stated reason is
  generic (fast-motion P pictures coded mostly intra); HEVC's
  `coarse_inter_intra` is on without it.
- Chroma intra mode search exists in VVC/AV2/VP9/H.264; AV1 always codes
  UV DC/CfL, HEVC always DM. No codec reason given.
- VVC `trellis_key_lambda_scale` ("the copies after a key inherit its
  quality"): a generic argument.
- Planned dead zone (#4) and λ headroom are missing in VVC and AV2.
- `coded_bits` (in-frame rate control) only in H.264 and AV1; HEVC/VVC have
  CABAC writers that could report it.

## 4. Parameters

### 4.1 Shared engine parameters with per-codec values

From every backend's `DEFAULT_PARAMS` (engine default first). Verdict:
**C** codec reason (a syntax or tool difference changes what the value
means), **T** looks like per-codec tuning (same mechanism, same units; value
found by screening one codec), **S** stale engine default (every codec agrees).

| Parameter | Engine | H.264 | HEVC | VVC | AV1 | VP9 | AV2 | Verdict |
|---|---|---|---|---|---|---|---|---|
| `quant.trellis` | 0 | 1 | 1 | 1 | 1 | 1 | 1 | S |
| `quant.trellis_states` | 4 | 1 | 1 | (unused) | 1 | 1 | 1 | S |
| `rate.cadence_motion_gate` | 0 | 1 | 1 | 1 | 1 | 1 | 1 | S |
| `rate.cadence_startup_period` | 0 | 8 | 8 | 8 | 8 | 8 | 8 | S |
| `rate.period_ceiling` | 0 | 1 | 1 | 1 | 1 | 1 | 1 | S (every non-Default preset pins 0) |
| `rate.key_still_boost` | 0 | 6 | 4 | 6 | 6 | 6 | 4 | S for 6; HEVC/AV2 4 is T (6 tripped HEVC's pulse guard) |
| `rate.cadence_startup_base_boost` | 0 | 4 | 6 | 6 | 4 | 4 | 5 | T |
| `quant.trellis_chroma` | 0.25 | 0 | 0 | 0 | (0.25) | 0 | (0.25) | S for 4; AV1/AV2 inherit 0.25 unscreened |
| `quant.trellis_key` | 1 | – | 1 | – | 0 | 1 | 0 | T |
| `quant.trellis_lambda_scale` | 1 | 1.3 | – | – | 0.6 | 1.139 | 1 | C/T: rate models differ in accuracy per entropy coder, but values come from per-codec screens |
| `planner.lambda_scale` | 2 | – | 2.58 | 4 | – | 2.38 | 2.58 | T (presets 2-6.7); partition bits are already the codec's own (`planner_leaf_bits`) |
| `planner.depth_removal_min_quantizer` | 0 | – | 0.86 | 0.8 | 0.8 | 0.86 | – | T (all Fast presets pin 0) |
| `decision.codec_subpel_seed_lambdas` | 0.03 | 0.1 | 0.05 | 0.05 | – | 0 | – | T |
| `decision.subpel_move_bias` / `_nonzero` / `_fractional_bias` | 1 / 1 / 1 | 1.20 / 1.07 / 1.1 | 1.1 ×3 | – | – | – | – | T (every Fast preset: 1.1 ×3) |
| `decision.intra_gate_variance` | 64 | 16 | 0 | – | – | 6.857 | – | C/T: the gate admits the codec's intra search, whose cost differs, but values are screens (presets 25-256) |
| `decision.exact_rank_margin` | 1.1 | 0 | 0 | – | – | – | 1.1 | T (HEVC measured +0.4 % at 1.1) |
| `decision.intra_directional` (+`_dc_ratio`) | 0 (4) | – | 1 (2) | 1 (2) | 1 | – | – | C: needs directional intra modes (VP9 Default is DC-only in P frames); H.264 and AV2 have them: unscreened |
| `decision.lambda_scale` | 1 | 0.66 | – | 0.91 | – | – | – | T |
| `decision.refine_cheap_vector` | 1 | 1 | – | – | – | – | 0 | T |
| `structure.unified` / `inherited_depth` | 0 | – | 1 | – | 1 | 1 | – | rollout in progress (structure agent) |

### 4.2 Codec-namespaced parameters doing one job

Defaults (`DEFAULT_PARAMS` override after the slash).

| Job | H.264 | HEVC | VVC | AV1 | VP9 | AV2 |
|---|---|---|---|---|---|---|
| Inter rounding | `inter_rounding_divisor` 5.6 | `inter_rounding` 128/109 (1/512) | `inter_rounding` 128/95 | `inter_rounding_q7` 48/41 (1/128) | `inter_rounding_divisor` 4/3.78 | `inter_rounding_q7` 32/36 |
| Intra/key rounding | – | `intra_rounding` 171/209 | `intra_rounding` 171/196 | `keyframe_rounding_q7` 48 | – | `key_rounding_q7` 48 |
| Chroma inter rounding | – | `chroma_inter_rounding` 128 | same, 128/170 | `inter_chroma_rounding_q7` 48 | `chroma_rounding` (mode) | `inter_chroma_rounding_delta_q7` 0/22 |
| Fixed-QP key strength | `keyframe_qp_offset` 6 (8 fixed-QP) | `key_qp_offset_delta` 0 (+ engine) | `key_qp_offset_delta` 0 | `keyframe_step_permille` 240 | same 240 | same 240 |
| λ per step² | (frame weights) | `lambda_scale` 1 | – | `lambda_per_step_squared` .135 | same .09/.1006 | same .135/.125 |
| Chroma offset | `chroma_qp_offset` -3 | same -3 | same 0/-1 | `chroma_delta_q` -32 | same -8/-14 | same -16 |
| Trailing ±1 shave | – | `shave_gap` 8 | same 8/9 | `shave_min_zero_run` 8 | `shave_gap` 0 | `coeff_trim` (RD) |
| Zero bound before transform | `inter_zero_*_scale` 1 | `pre_transform_zero_scale` 2 | same 1/2 | `pre_tx_texture_scale` 8 | `pre_transform_zero_scale` 1 | `pre_tx_texture_scale` 8 |
| Merge list size | – | `merge_candidates` 2/4 | same 2/6 | `near_count` 3 | – | `drl_ranked_entries` 6 |
| P-frame intra DC only | `inter_intra_dc_only` 0 | same 0 | same 0 | `inter_intra_search` | same 0/1 | – |
| Still skip | – | `still_skip`, `still_skip_limit_scale` | – | same | – | – |
| Early merge | – | `early_merge_lambdas` 0 | same 0 | – | – | – |
| Filter λ scale | – | `sao_luma_lambda_scale` 1 | same 1/2.1, `alf_lambda_scale` | `restoration_lambda_scale` | – | – |
| Filter stats row step | – | `sao_stats_row_step` 4 | same, `alf_stats_row_step` 2/4 | `cdef_search_row_step` 4, `restoration_row_step` 2 | – | `cdef_search_row_period` 8 |
| Filter search interval | (const 4) | – | `sao_search_interval` 1/4, `alf_bank_interval` 1/4 | `cdef_local_search_period` 1 | – | – |
| Deblock off on inter | `deblock_inter` | same | – | `loop_filter` | `skip_filter_off` | `deblock` |
| Loop-filter level scale / sharpness | – | – | – | `loop_filter_level_scale`, `_sharpness` | `filter_level_scale` 1/.63, `filter_sharpness` | – |
| Partial forward transform | – | `partial_forward` | (`fast_forward`) | `partial_forward` | `partial_forward` | – |
| Cadence base period | – | – | – | `av1.base_period` 4 (= `decision.base_period`) | – | – |

Units differ (1/512, 1/128, a divisor), so moving these into engine groups
is a rename plus a unit conversion; H.264's and VP9's divisors are not exactly
representable in 1/512 (1/5.6, 1/3.78), so a common unit changes their
streams by a rounding step. The names should converge when each mechanism
moves into the engine (§3), not as a separate rename that breaks presets.

## 5. Process

What exists:

- CLAUDE.md: one-way dependencies, `CodecBackend` only, and "**Port speed-ups
  across codecs**: check whether the same pattern exists in the others and
  port it, or say in the commit why it doesn't apply". ARCHITECTURE.md §2:
  "One engine, many codecs ... each codec adds only its syntax, transforms,
  entropy coding, prediction filters and loop filters".
- `briefs/borrow-and-share.md` (decoder speed, 2026-11-12): "Share between our
  codecs (one implementation, many users) ... put it in nc-dsp / nc-common with
  a neutral name". This is the only brief text that asks for one shared
  implementation, and it covers kernels and decoders, not encoder heuristics.
- `docs/handoff/cross-codec-speedups.md`: the porting ledger; 21 of 102 cells
  "not checked".
- Landing checks (`land.sh`, `land_scope.py`, `land_guard.py`,
  `check_layering.py`) enforce layering only; nothing measures duplication.

Gaps:

1. "Port" invites a copy. Most duplicates in §3 are exactly that: the
   speed-up was ported (VP9 copied AV2's chroma weight, AV1 copied AV2's
   angle search, VP9 copied AV1's reference store) instead of moved down a
   layer.
2. Nothing counts the cost of non-shared code. A codec-local version that is
   0.1 % better on one codec's screen wins today, though it adds a second
   implementation, a second parameter name and a second screen to maintain.
3. Per-codec values of shared parameters need no justification: a screen per
   codec finds a constant per codec, which is fitting each codec's draw.
   Screens are rotated and pooled, but the per-codec degree of freedom
   remains.
4. New heuristics get no "codec reason" check: the code rarely says why a
   mechanism is codec-local (§3 one-codec list), so the next agent cannot
   tell a syntax constraint from an accident.
5. Engine defaults are not kept current: when every codec overrides a
   default to the same value, the default is stale and a new codec starts
   from the wrong point (§4.1 "S").

## 6. Correctness findings (outside the duplication brief)

1. **`--key-qp-offset` is silently ignored by H.264 and AV1** (confirmed:
   neither reads `config.key_qp_offset`; HEVC/VVC honour it, VP9/AV2 reject
   it). Shared key-frame quantiser (#5) fixes it.
2. **The engine's picture-owning-backend guards are dead** (confirmed: no
   backend sets `OWN_PICTURE_PIPELINE`): the intra-refresh and
   `set_resolution` guards that name "VP9, AV2" never fire, so those codecs
   now run intra refresh without `set_refresh`. VP9 still reads the previous
   frame's vectors (`previous_vectors`, `nc-vp9/src/backend.rs:1901`), which
   HEVC/VVC turn off under refresh; recovery after a loss may be broken for
   VP9 (unverified: needs a `realuse_conformance` / `static_recovery` run
   with refresh on VP9 and AV2).
3. **HEVC writes SAO reuse state in `filter_frame`, not `commit_frame`**
   (`nc-hevc/src/backend.rs:1586`; VVC defers it, `:2004`/`:1342`): a re-coded
   or dropped frame can change later frames' reuse (unverified effect).
4. **VVC `loop_filter_reach` is the default 8** (`nc-vvc/src/backend.rs:914`):
   long deblocking reads 8 and SAO and ALF follow; matters for intra refresh
   only (unverified).
5. **VVC's intra path may not force DC on the refresh dirty edge** as
   H.264/HEVC/AV1 do (unverified).
6. AV2's second `"av2"` parameter group (`search.rs:6`) is unregistered and
   clashes in name with `ToolParams` (`av2.pre_tx_skip_energy` 30 vs 15); only
   the legacy encoder (#27) uses it.

## 7. Proposed methodology rule

Added to CLAUDE.md under Rules as "Shared first" (owner, 2026-11-15):

- One mechanism, one implementation: a heuristic, cost model or search that
  is not bitstream syntax lives in the engine (or nc-dsp), and the codec
  supplies only its syntax-dependent inputs through `CodecBackend`. Prefer it
  even at equal or slightly worse measured results (within about +0.3 % J,
  no guard tripped): it is more robust and harder to overfit.
- Count non-shared code as a cost: porting a speed-up means moving it into
  shared code, not copying it.
- A codec-local heuristic states its codec reason (the syntax or tool
  difference) in its doc comment.
- A codec's value for a shared parameter is justified in its
  `DEFAULT_PARAMS` comment by a codec reason, or it is the shared default.
  When every codec sets the same value, it becomes the engine default.

## 8. Plan and status

Exact refactors land as staged commits, each proved with
`tools/dev/preset_identity.py` (every codec, Default and every preset, three
clips, CBR at two rates and fixed QP) plus the crates' tests and clippy.
Stream-changing unifications are screened with `nceval.screen --clips adopt`
(L1-L3, 600-frame final) and adopted at equal or slightly worse J per the
rule. Coordination (2026-11-15): #2-#4, #6, #7 wait for the integer-width
H.264 landing and build on its i16 types; HEVC/VVC stay generic for the
shelved `feature/int-width-enc7-hevc`.

Progress is recorded below as it lands.

| Date | Item | Result |
|---|---|---|
| 11-15 | #27 AV2 legacy encoder, #28 `OWN_PICTURE_PIPELINE` | deleted (owner: "design that out and completely remove it"); 2,899 lines out; AV2 decoder tests moved onto `Av2Backend`; `tools/ci/check_encode_paths.py` keeps one encode path per codec. Other crates audited: no second encoder (front ends all use `nc_encoder::Encoder`). |
| 11-15 | #1 consensus defaults (quant) | `quant.trellis` 1 and `quant.trellis_states` 1 are the engine defaults; the six codecs' copies (and redundant `trellis_key` 1, AV2 `trellis_lambda_scale` 1 and `exact_rank_margin` 1.1 pins) removed; exact. The `rate.*` consensus values are left to the rate-control owners (engine RC unit tests run on the engine defaults). |
| 11-15 | #9 chroma policy mapping, #10 adaptive weight | `ChromaPolicy::closest_qp_offset` (H.264, HEVC), `closest_step_delta` (AV1, VP9, AV2), `adaptive_error_weight` (VP9, AV2); exact. |
| 11-15 | #5 key quantizer | `scaled_quantizer` (AV1, VP9, AV2 `keyframe_step_permille` search) beside `offset_key_quantizer` (a73985d0, audit-fixes); exact. |
| 11-15 | #2 trellis settings, #4 planned dead zone, #6 shave | `nc_encoder::TrellisSettings` (all six codecs), `planned_min_level` / `apply_planned_dead_zone` (H.264, HEVC, AV1, VP9; generic over i16/i32 levels for the parked HEVC/VVC i16 branch), `shave_isolated_ones` (HEVC, VVC; VP9's and AV1's rules differ and stay theirs); exact. |
| 11-15 | #11 intra fallback | `BlockDecision::intra_fallback` replaces the `CodecBackend::intra_fallback` hook and four (position, vector) stashes; exact. |
| 11-15 | #13, #15 block and quadrant SSE | `nc_encoder::block_sse` (HEVC, VP9, H.264), `quadrant_sse` under the HEVC/VP9/AV1 split gates (each keeps its comparison); exact (252/252). |
| 11-15 | #17 angle deltas, #7 HEVC quantiser | `AngleDeltaSearch` (AV1, AV2); HEVC's quantiser terms from nc-dsp's `HevcQuantizer` (`forward_terms`, `largest_zero_coefficient`); exact (252/252). |
| 11-15 | VVC intra refresh | verified: `loop_filter_reach` 16 (4f0d8f05) and dirty-edge DC forcing (044e52a5) landed upstream; `refresh_tests.rs` (corrupted references, Default params) recovers within two sweeps. That test also passes with reach 8, so it does not pin the reach; 16 stands on the filter analysis. |
| 11-15 | #3, #16, #19 | not worth sharing: the trellis wrapper's shared part is already `Trellis`; still-skip and early merge keep only codec state and an off-by-default lever around the shared tests. |
| 11-15 | #8 held reference slots | `nc_encoder::HeldSlots<T>` and `scaled_reference_allowed` replace VP9's copy of AV1's reference store; exact (AV1, VP9 135/135). |
| 11-15 | §4.1 `quant.trellis_chroma` | stream-changing: the shared default is 0 (H.264, HEVC, VVC, VP9 pinned it); AV1 and AV2 move onto it. Pool levels of draw 2026-11-15, L3 single point, 78 clips: AV1 J -0.46% [-0.90, -0.03], AV2 J -0.49% [-0.81, -0.19], both BETTER, no guard at L2/L3 (AV1's L1 flagged its 4 static clips, +3.96%; static at L3 -0.42%). |
| 11-15 | #18 CDEF strength adjustment, #22 reference index | AV1's and AV2's encoder copies of `adjust_strength` call nc-dsp's existing `cdef_adjust_strength`; `ReferenceChoice::single_index` replaces H.264/HEVC/VVC `ref_idx_of` and AV2 `reference_rank`; exact. |
| 11-15 | #12 filter search cadence | not shared: the common part is one test (`key || index % interval == 0`); sharing the mechanism means one engine interval parameter for HEVC SAO (hard-coded 4), VVC SAO (4) and AV1 CDEF (1), a stream-changing unification left with tuning winding down. |
| 11-15 | #14 | not shared: HEVC/VVC `QuantGroup` and AV1/AV2 `block_qindex` are each codec's delta-QP syntax state, which belongs in the codec crate. |

## 9. Final status (2026-11-15)

Work stopped here at the coordinator's request; tuning is winding down.

**Done** (landed on `variance-integration`; exact ones proved with
`tools/dev/preset_identity.py`, every codec's Default and every preset):

- #27, #28: AV2's standalone encoder and the engine's `OWN_PICTURE_PIPELINE`
  hook deleted; `tools/ci/check_encode_paths.py` keeps one encode path per
  codec.
- #1 (quant part): `quant.trellis` 1 and `quant.trellis_states` 1 are engine
  defaults.
- #2 `TrellisSettings`, #4 planned dead zone, #5 `scaled_quantizer`, #6 shave
  (HEVC, VVC), #7 HEVC quantiser from nc-dsp, #8 `HeldSlots` and
  `scaled_reference_allowed`, #9 chroma policy mapping, #10 adaptive chroma
  weight, #11 `BlockDecision::intra_fallback`, #13 `quadrant_sse`, #15
  `block_sse`, #17 `AngleDeltaSearch`, #18 CDEF strength adjustment from
  nc-dsp, #22 `ReferenceChoice::single_index`: all exact.
- §4.1 `quant.trellis_chroma`: shared default 0 (stream-changing, L3 better
  for AV1 and AV2, no guard at L2/L3).
- VVC intra refresh (§6.4, §6.5): verified fixed upstream (4f0d8f05,
  044e52a5); §6.1-§6.3 were fixed by the coordinator's audit-fixes work.

**Decided not to share** (reasons in §8's table):

- #3 trellis wrapper: the shared part is already `Trellis`; what remains per
  codec is its own candidate and write-back arithmetic.
- #12 filter search cadence: one shared test; a real unification needs one
  engine interval parameter (HEVC SAO 4 hard-coded, VVC SAO 4, AV1 CDEF 1),
  which is stream-changing.
- #14 quantisation-group QP state: each codec's delta-QP syntax.
- #16 still skip, #19 early merge: codec state around shared tests, and the
  levers are off by default.
- #6 for VP9 and AV1: their trailing-one rules behave differently.

**Open** (no clear case while tuning winds down; each needs one L3 screen):

- #20 inter-frame intra tiers, #21 transform-split early exits (small,
  off-by-one conventions differ), #23 one SAO edge-offset search (HEVC vs
  VVC's RD offsets), #24/#25 the λ forms (one `λ = k·step²` model, k 0.09 vs
  0.135, and four chroma-λ arithmetics), #26 residual-drop RD.
- The per-codec parameter spreads in §4.1 marked T. Among them
  `rate.cadence_startup_base_boost` is 4 (H.264, AV1, VP9), 5 (AV2) and 6
  (HEVC, VVC), with no known codec reason: the start-up cadence is the
  engine's (`qp_cadence.rs`) and the values come from per-codec screens. It
  is the clearest candidate for one shared value, left to the rate-control
  owners with the other `rate.*` consensus defaults (§4.1 "S" rows:
  `rate.cadence_motion_gate` 1, `rate.cadence_startup_period` 8,
  `rate.period_ceiling` 1, `rate.key_still_boost` 6 in four codecs).
