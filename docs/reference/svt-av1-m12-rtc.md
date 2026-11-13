# SVT-AV1 v4.3.0 `--rtc 1 --preset 12`: what it does per frame and per superblock

SVT-AV1 real-time M12 is nanocodec's primary design guide and bar
(`docs/ARCHITECTURE.md` §2). This is an implementation-level reading of the code path
we measured: the pinned v4.3.0 tree in the anchors build, full build
(`RTC_BUILD=OFF`, `MOBILE_BUILD=OFF`). All `file:function` references are
under `Source/Lib/Codec/` unless the path is given. **[inferred]** marks
reasoning rather than code that was read. SVT-AV1 is BSD-3 + AOM patent
licence: its algorithms may be reimplemented with attribution.

Measured on our quick suite (CQP, flat IPP, 1 thread):
- about −49% BD-rate (PSNR-YUV) vs openh264-baseline;
- CPU 1.24–1.48× openh264-baseline on the PC, 1.27–1.65× on the Jetson.

## The six things that change the picture

1. **Our measurements were made in CQP, which runs a narrower configuration than
   "rtc".** The adapter passes `--rc 0 --aq-mode 0 --scd 0
   --hierarchical-levels (TL-1) --lp 1`.
   - With 1 temporal layer this is the *flat-IPP* path
     (`use_flat_ipp = hierarchical_levels==0`).
   - Cyclic refresh is CBR-only, so **it was off in every measured number**.
   - The defaults differ from what we ran: rtc with automatic levels gives 2
     hierarchical levels (`Globals/enc_handle.c`, about line 4567).
2. **One reference, previous frame only.** From M9 up in rtc, `mrp_level=0`
   (`Globals/enc_handle.c: set_mrp_ctrl` / `set_mrp_ctrl_with_level`): one
   list-0 reference, no list 1, no compound.
3. **The AV1 deblocking filter is off at M11, M12 and M13.**
   - `enc_mode_config.c: get_dlf_level_rtc` returns 0 above M10, and level 0
     means `enabled=0` in `svt_aom_set_dlf_controls`.
   - Only CDEF runs. Loop restoration is off for rtc at every resolution
     (`svt_aom_get_wn_filter_level_rtc` and `_sg_` both return 0).
4. **Intra is DC_PRED only from M11 up, on every frame including keyframes.**
   - `svt_aom_get_intra_mode_levels_rtc` sets
     `intra_level = MAX_INTRA_LEVEL-1 = 9`, meaning `intra_mode_end = DC_PRED`.
   - On P frames intra is skipped entirely unless `norm_me_dist > 5000`, about
     1.2 SAD per pixel (`sig_deriv_mode_decision_config_rtc`, `skip_intra`).
5. **P frames never run regular RD mode decision from M11 up.** Partitioning is a
   variance-cost "very light PD0". PD1 then codes only that partition
   (`pic_block_based_depth_refinement_level=10`, i.e.
   `PD0_DEPTH_PRED_PART_ONLY`), through the light-PD1 path, which takes exactly
   one candidate to transform.
6. **Several fast paths are switched by picture variance.** `pic_avg_variance`
   is the per-pixel variance of 64×64 blocks, averaged over the frame. It is
   compared with `RTC_LOW_VARIANCE_TH=1000` and `RTC_ULTRA_LOW_VARIANCE_TH=350`
   (`enc_mode_config.c:17–18`). Most of the M12-vs-M13 difference is these
   variance-gated protections.

---

## 1. Frame-level flow

**Process chain:** resource coordination → picture analysis → picture decision
→ ME → initial RC → RC → MD config → EncDec (mode decision) → DLF (no-op) →
CDEF → rest (no-op) → entropy coding → packetisation.
- `pic_bypass_encdec = 1` for 8-bit from M3 up (`svt_aom_get_bypass_encdec_rtc`),
  so **the MD reconstruction is the final reconstruction**. There is no separate
  encode pass.

**Picture analysis** (`pic_analysis_process.c: svt_aom_picture_analysis_kernel_iter`)
- Pads the frame and builds ½ and ¼ downsampled luma
  (`svt_aom_downsample_filtering_input_picture`). SVT calls these "quarter" and
  "sixteenth" because they are ¼ and 1/16 of the area.
- Computes per-SB variance at 8/16/32/64 from 8×8 means and mean-squares
  (`compute_b64_variance`). Its mean is `pic_avg_variance`.
- Runs screen-content detection (`svt_aom_is_screen_content`; default mode 2,
  ≤1080p) and luma-dominance detection.
- No temporal filtering: low delay forces TF off. No TPL:
  `Globals/enc_handle.c: get_tpl` returns 0 for low delay and for `aq_mode 0`.

**Motion estimation** (`motion_estimation.c: svt_aom_motion_estimation_b64`,
per 64×64; settings from `enc_mode_config.c: svt_aom_sig_deriv_me`)
- **Open loop:** it searches source pictures (the PA reference), not
  reconstructions.
- **Zero-MV SAD gates.** SAD is measured on every other row and doubled;
  `me_early_exit_th = 64·64·8`.
  - Static bypass, `me_static_b64_bypass`: if the zero-MV SAD is under 1 per
    pixel (`me_static_b64_th = 4096`), skip HME and integer search, set MV = 0,
    and fill the 32/16/8 SADs as zz/4, /16, /64.
  - HME level 0 is skipped if zz_SAD < 8192 (`hme_level0_b64`).
  - The integer search window collapses to 1×1 if zz_SAD < 5461
    (`integer_search_b64`, about line 1260).
- **HME level 0** runs on the 1/16-area picture with 2×2 search quadrants.
  - `hme_l0_sa` is min {8,8}, max {96,96} at M9 and up (`set_hme_search_params`).
  - For distance 1, `get_hme_l0_search_area` makes each quadrant 16×4, so the
    total is 32×8 at ¼ linear resolution, about ±64×±16 full-res pixels.
  - Uses `SUB_SAD_SEARCH` (every other row) with `svt_sad_loop_kernel`.
  - Scales by distance (`ceil(5·dist/8)`) and by QP.
- **HME level 1** (½ resolution, 8×3 refinement):
  - Flat IPP: on for M≤11; at M12 only if `pic_avg_variance < 1000`; off at
    M13 (`svt_aom_sig_deriv_multi_processes_rtc`). Always on for hierarchical
    structures.
  - Level 2 is off. Pre-HME is off for flat M11+ (it is 4 at M9/M10).
- **Full-pel search** (`set_me_search_params`):
  - Window `{8,1}` for flat M11+, for M12/M13 in any structure, and for
    hierarchical M≥12 — clamped up to 8×3 by the QP-scaling minimums. That is
    24 positions around the HME centre (±4 horizontally, ±1 vertically).
    M10, and M11 with hierarchical layers, use {24,16}→{32,16} below 1080p.
  - The zero-MV-versus-HME-centre check (`check_00_center`) is skipped because
    `me_early_exit_th` is non-zero.
  - Kernel: `svt_ext_all_sad_calculation_8x8_16x16` plus
    `svt_ext_eight_sad_calculation_32x32_64x64`. One pass over 8 x-positions
    gives the best MV and SAD for all 85 square blocks
    (1×64, 4×32, 16×16, 64×8).
- **QP-dependent scaling:** every ME/HME size and the depth/PD0 thresholds use
  `svt_aom_get_qp_based_th_scaling_factors` with the *configured* QP:
  `max(10,qp)/63` below QP 46, and `1.05 − e^{−(qp−35)/10}` from 46 up. All
  flags are on for rtc (`Globals/enc_handle.c: set_qp_based_th_scaling_ctrls_rtc`).
- **What is stored** (`compute_distortion`):
  - Per SB: `me_64x64/32x32/16x16/8x8_distortion` (sums, normalised to 4096
    pixels).
  - `me_8x8_cost_variance`: the variance of the 64 8×8 SADs.
  - `rc_me_distortion`.
  - Per block, the best MV and SAD. ME 8×8 storage is off from M9 up
    (`svt_aom_get_enable_me_8x8`), so 8×8 blocks would use the 16×16 MV.
  - Only the best uni-pred candidate is kept (`use_best_me_unipred_cand_only=1`).
  - `norm_me_dist` is the mean of `me_8x8_distortion` over SBs
    (`initial_rc_process.c`, line 718).
- **Also off:** global motion (M5+), MFMV (`svt_aom_set_mfmv_config`: off for
  rtc), warped motion, OBMC (M9+), inter-intra, filter-intra, compound,
  palette/IntraBC (screen content only, ≤M8).

**Reference structure** (`pd_process.c`, about line 1946)
- Flat rtc: `ref_dpb_index[LAST] = previous frame`, using `flat_max_refs = 1`
  DPB slot.
- LD with layers in CQP: `LAST` is the newest base frame for layer-1 positions;
  frames after the layer-1 picture use it instead of the base. This is the
  standard L1T3 pattern. With layers, the "long base" every 128 frames is
  refreshed into slot 7 (the CQP branch) but is unused with one reference.
- `set_frame_update_type`: flat frames get pseudo-layers — every 4th frame
  (`frame_offset % 4 == 0`) is GF_UPDATE ("base/boosted"), odd frames are LF
  ("leaf"), the rest INTNL_ARF.
  - Many tool levels key on `is_base` and `is_leaf`, so flat IPP is effectively
    a 4-frame cadence of effort.

**What base frames get that other layers don't, at M12:** a CDEF search (others
reuse the reference's strength); a gentler light-PD1 level (4 instead of 5 or
8); a lower QP and λ only in layered CQP (§6); and stricter skip-DC thresholds
on non-leaf frames.

---

## 2. Partitioning (P frames at M12)

**Sizes and shapes**
- SB64 always for rtc (`Globals/enc_handle.c`, about line 4109).
- No 4×4 (`svt_aom_get_disallow_4x4_rtc` returns true).
- **No 8×8 from M10 up** (`svt_aom_get_disallow_8x8_rtc`), except where frame
  edges force it (`dimensions_require_8x8`). P-frame blocks are therefore
  **64, 32 and 16 square only**.
- NSQ geometry level 3 (H/V only, min 8, no HA/HB/H4), but
  `md_disallow_nsq_search=1` in PD0 and light PD1. **NSQ is effectively used
  only on I frames**, through regular PD1 with `nsq_search_level 19`.

**Maximum size per SB** (`get_max_block_size_rtc`): 32 if
`me_8x8_cost_variance > HIGH_8x8_DIST_VAR_TH (50000) × q_weight`, else 64.

**Depth removal** (`set_depth_removal_level_controls`): level 7 at M12, or 6 if
`pic_avg_variance < 350`; level 7 at M13.
- Costs are `RDCOST(fast_λ, 0, me_dist_N)`. Deviations are
  `dev_32→16 = (c32−c16)·1000/c16` and so on.
- Below-32 is disallowed if `c32 < RDCOST(fast_λ, 8192, 512·6)` or both
  deviations are under their thresholds. The base thresholds are 50 (16→8) and
  25 (32→16), scaled by `me_8x8_cost_variance/(63−qp−10)` bands (<25000 ×4,
  <50000 ×2/½, else 0) and by `max((63−qp−10)>>4, 1) · qp_scale_factor(2)`.
- Below-16 uses the same `RDCOST(fast_λ, 8192, 512·6)` cost test.
- In rtc, reference-frame information is not used here.

**PD0** (`enc_dec_process.c: svt_aom_mode_decision_kernel_iter`)
- `set_pic_pd0_lvl_rtc`: M10+ uses `pic_pd0_lvl = 7`, i.e. **PD0_LVL_6, "very
  light PD0", no transform**, on P frames. I frames and transitions get 6
  (PD0_LVL_5).
- **The PD0 classifier is not used in rtc** (`use_pd0_classifier = !rtc`).
- Block cost (`product_coding_loop.c: compute_lpd0_cost_inter`,
  `compute_lpd0_cost_from_variance`):
  - For up to 3 ME reference candidates: compute
    `variance(src, fullpel_ME_pred on the RECON reference)` with
    `svt_aom_mefn_ptr[bs].vf`.
    This cap is over reference/direction entries, not three alternative MVs
    for one reference (`product_coding_loop.c:8261–8328`, pinned commit above).
    Each block/reference indexes one stored ME vector; active M12 RTC uses
    one L0 reference and no L1, so the generic cap does not add three vectors.
  - `cost = RDCOST(λ_full, rate(PARTITION_NONE), min(var/area, λ>>10)·area)`.
    The per-pixel distortion is capped at λ/1024, a "noise floor" cap.
- Split decision (`test_split_partition_pd0`):
  - `split_cost = RDCOST(λ, 2 × rate(PARTITION_SPLIT), 0)`; the rate is doubled
    because `use_accurate_part_ctx=false` above M7.
  - The children's costs are added one by one. The search exits early when
    `parent·1000·bias ≤ split_so_far·10⁶`; `depth_early_exit_lvl=0` in rtc
    LVL_6 gives thresholds of 1000.
  - The parent is kept if `bias·parent ≤ 1000·split`.
- **Parent bias** (`svt_aom_sig_deriv_enc_dec_pd0`): `1100 − 150·base_q/255`,
  plus 25, 50 or 75 when `me_8x8_cost_variance` exceeds 500, 1000 or 2000,
  clipped to [900, 1200].
  - At M13 the variance part is scaled by a weight from the ratio
    `me_64x64_dist/me_8x8_dist` (`pd0_cost_bias_weight = 600`). This is a
    **real M12/M13 difference**.

**PD1** uses `pred_depth_only` and a fixed partition (no ±1 refinement from M8
up).

---

## 3. Mode decision

### Light-PD1 level per frame (`sig_deriv_mode_decision_config_rtc`)

**M12:**
- Base frames: `pic_lpd1_lvl = 4` (LPD1_LVL_1), or 5 if
  `hierarchical_levels == 1`.
- Other frames: `low_var ? 5 (LVL_3) : 8 (LVL_6)`, where low_var means
  `pic_avg_variance < 1000`.
- I frames: 0, i.e. regular PD1.

**Per-SB changes:**
- In CBR, cyclic-refresh SBs drop 2 levels, capped at 2 on base frames and 4
  otherwise (`svt_aom_sig_deriv_enc_dec_common`).
- `lpd1_detector_skip_pd0` (`enc_dec_process.c`) lowers the level by scoring
  the co-located SB in the reference: +5 each if it was intra, was not skip, has
  dist 64 > 3× the reference's, or has 8×8 variance > 3×. It then compares with
  the left and top SBs (dist > (L+T)<<5) and checks whether the neighbours were
  intra.
- `pd1_lvl_refinement = 2` stops the cascade at LVL_1, so **every P-frame SB
  uses the light path from M11 up**.

### Candidates (`mode_decision.c: generate_md_stage_0_cand_light_pd1`)
- **NEARESTMV**: always, from the full AV1 MVP stack
  (`svt_aom_generate_av1_mvp_table`; spatial only, MFMV off). Skipped when both
  neighbours are intra (`use_neighbouring_mode`).
- **NEARMV**: up to `near_count` — 1 at `cand_reduction` level ≥3. LPD1 levels
  map to 2 (LVL_0), 3 (≤LVL_2), 4 (LVL_3), 5 (LVL_4+), each floored at the
  picture level of 1.
- **NEWMV**: 1, the best ME MV after subpel refinement, de-duplicated against
  MVs already injected.
- **No GLOBALMV and no explicit ZEROMV**; zero MV is injected only as a
  fallback (`inject_zz_backup_candidate`).
- **Intra**: DC only, and only if `!skip_intra`.
  - DC is dropped at MDS0 if the best inter variance is under
    `skip_dc_th·area`: 15 on leaf frames, 5 otherwise, at cand_reduction 5.
  - The GLOBALMV MDS0 bypass (`lpd1_try_mds0_bypass`) exists but its threshold
    is 0 in rtc, so it never fires.

### Subpel (`product_coding_loop.c: read_refine_me_mvs_light_pd1`)
- Runs on the **reconstructed** reference, before candidate injection, on the
  ME MV only.
- The level is `max(lpd1-derived level, pcs->me_subpel_level)`. The picture
  level is 6 at M10–M12 and 9 at M13.
  - LVL_0 → 6: `SUBPEL_TREE_PRUNED`, 4-tap, 1 iteration, quarter-pel,
    `bias_fp 110`.
  - LVL_1–3 → 7, LVL_4+ → 8: `md_subpel_search_fixed_stage`.
- **Fixed-stage search:** a full-pel variance baseline, then the 4 half-pel cross
  positions, then the 4 quarter-pel cross positions around the best. It uses the
  `svf` sub-pixel variance (bilinear, libaom-style). Half-pel positions use
  the original ME centre; each quarter-pel position uses the **updated** best
  vector, allowing a coordinate walk. Direction order is +x, -x, +y, -y.
  The 110% integer margin applies only while the best offset **relative to
  the initial ME vector** is zero; after the first accepted move it is off.
  This is not a test of whether the absolute MV is zero.
  - Level 8 adds an early exit when var < 10·area; level 9 uses 20·area and also
    skips when the variance of the full-pel prediction is < 100.
- **Subpel is skipped entirely** at LVL_4+ when the reference SB was skip (with
  ref_skip% > 50) or used the 64×64 MVP, and `me_8x8_var < 200·qp` and
  `me_64_dist < 200·qp`.
- Max precision is quarter-pel. `allow_high_precision_mv` only applies at
  ≤480p and q_idx < 128.

### MDS0 (`md_stage_0_light_pd1` → `fast_loop_core_light_pd1`)
- Luma prediction only, with **EIGHTTAP_REGULAR fixed**
  (`interpolation_search_level = 0` from M11 up, frame filter not switchable).
- **Distortion is the variance of the residual** (mean-removed), not SAD or
  SATD.
- `cost = luma_rate_approx·λ + (var<<4)·λ`, with `approx_inter_rate = 2`: the
  NEWMV predictor is always `ref_mv_stack[0]` and reference-frame bits are
  skipped (`rd_cost.c`).
- A candidate is dropped as soon as its distortion-only cost exceeds the best so
  far.
- **Exactly one candidate goes on to MDS3.**

### MDS3 (`full_loop_core_light_pd1`)
1. **Transform-skip score** (`lpd1_should_perform_tx`). With
   SSE < `dist_energy_th(30–40)·area·qindex/100`:
   - +50;
   - +20 if both neighbours are skip, +15 if both are NEAREST, +15 more if both
     conditions hold;
   Separately, add +150 if `RDCOST(skip) < rd_skip_th%` of a reference
   cost, then subtract the low-QP penalty. The RD signal does not require
   the energy condition.
   - The transform is skipped when the score reaches `skip_tx_score_th`: 125
     (lpd1 tx-skip levels 1–2), 70 (level 3), 50 (levels 4–5). LVL_5/6 SBs with
     low `me_8x8_var`/`me_dist` move up to levels 4/5.
   Nanocodec exposes this score as an optional experiment:
   `av1.pre_tx_skip=1`, with `pre_tx_skip_energy`, `pre_tx_skip_score` and
   `pre_tx_skip_rd` controls (defaults 30, 125, 100). Score zero disables
   the gate; RD zero disables its RD signal. `pre_tx_luma_dominant=1`
   conservatively applies SVT's luma-dominant low-qindex protection to every
   input because no equivalent input classifier is available. Setting it to
   zero uses the source's nondominant penalties. The gate uses prediction
   SSE, current syntax CDF rates and nanocodec's lambda. It skips luma only;
   existing chroma decisions remain separate. It is disabled by default.
2. **Luma transform** (`perform_dct_dct_tx_light_pd1`): **DCT_DCT only**, at the
   **largest transform size** (= block size; 64-point transforms keep the
   top-left 32×32).
   - Plain AV1 `quantize_b` (no RDOQ from M11 up), then **coefficient shaving**
     (`full_loop.c: shave_coeff`): trailing ±1 coefficients separated by ≥8
     zeros are removed, then an RD check on the trailing ±1s.
3. **Coefficient rate estimate** (`rate_est_level 0`, i.e. `coeff_rate_est_lvl 0`):
   - `6000 + eob·1000` if eob < area/64, else `6000 + eob·400`. Units are 1/512
     bit.
   - **No context modelling at all.**
4. **Block-skip check** (`lpd1_blk_skip_luma_rd`): force skip if
   `skip_cost·pct < code_cost·100`, with pct = 90 / 75 / 60 by LPD1 level.
5. **Chroma:**
   - Prediction follows luma (`CHROMA_MODE_1`, chroma level 5 from M11 up). No
     independent UV search, no CfL on P frames.
   - `lpd1_chroma_energy_skip`: a per-plane SAD on every other row; skip the
     chroma transform if SAD < 5·area_uv/8.
   - A chroma-complexity detector forces chroma when luma is zero (lpd1_tx
     level 5 at ≤LVL_4, level 6 above).
6. `svt_aom_full_cost`, then inverse transform and reconstruction
   (`av1_perform_inverse_transform_recon`).

### I frames at M12
- PD0_LVL_5: light PD0 with a transform, PD0 intra at level 8 (DC…SMOOTH,
  pruned).
- Then regular PD1 (`svt_aom_sig_deriv_enc_dec_rtc`) at the predicted depth:
  - **DC only, plus CfL** (cfl_level 2 on I frames);
  - NSQ H/V testing;
  - txs 0 at M12, so TX_MODE_LARGEST;
  - txt 0 (DCT only);
  - no RDOQ;
  - NIC level 11.

### Lambda (`rc_process.c: svt_aom_compute_rd_mult`)
- libaom `def_*_rd_multiplier(q)·q²` on `dc_q`.
- Multiplied by the frame-type factor: 150/128 for TL0 and flat frames,
  180/128 for leaf layers.
- KF additionally ×100/128 (`RTC_KF_LAMBDA_BOOST`).
- Per-SB λ modulation from ME is on only for M≤10 in rtc.

---

## 4. Loop filters

**Deblocking:** off at M11 and up.
- M10 is `dlf_level 7`: SB-based, `LPF_PICK_FROM_Q`, zero-strength level 3. On
  non-base frames it is turned off when the reference skip % is above 95 or 75
  (`dlf_level_modulation`, mode 3).
- `--enable-dlf 2` would act as if the preset were 3 lower.

**CDEF** (`set_cdef_search_controls`; levels from
`svt_aom_sig_deriv_multi_processes_rtc`):
- Flat IPP: I frames level 5; P frames at M10/M11 level 7; M12 level 7 if
  variance < 350, else 9; M13 level 9. With layers: level 8 from M9 up.
- **Level 9 (typical at M12):**
  - Base frames search only 2 luma strengths, `pf_gi[0]=0` and `pf_gi[15]`
    (pri 15, sec 0), with no secondary strengths.
  - Distortion is measured on every 4th row (`subsampling_factor 4`).
  - UV strength comes from a QP model (`svt_pick_cdef_from_qp`).
  - Non-base frames **do not search**: Y strength is the midpoint of the
    reference's strengths, UV from QP
    (`md_config_process.c: update_cdef_filters_on_ref_info`).
  - Damping is `3 + (base_q_idx>>6)`.
- **Level 7:** the same 2 strengths plus a secondary pass at +2, with a UV
  search.
- **Level 5 (I frames):** 3 primary strengths (0, 7, 15) plus a +2 secondary.
- **Application:** filter blocks with no non-skip blocks are skipped
  (`cdef_count == 0`). The encoder applies CDEF only to reference or
  reconstruction frames, so non-reference top-layer frames are signalled but not
  filtered by the encoder (`cdef_process.c: svt_aom_cdef_kernel_iter`).

**Loop restoration:** off.

---

## 5. Entropy and bitstream

**Inside mode decision:** CDF tracking is off from M9 up (`update_cdf_level 0`).
Rates come from the frame-start context tables.

**In-frame CDF adaptation (`disable_cdf_update`), from M11 up**
(`should_disable_cdf_update`):
- It is disabled once 30 frames have passed since the key frame.
- It is re-enabled once every 9 frames (`frames_since_cdf_update` cycles 0…8).
- `frame_end_cdf_update_mode = 1`, so contexts are refreshed backward.
- One tile; standard range coder.

---

## 6. Rate control and QP

### Fixed QP — what we measured (`rc_crf_cqp.c: cqp_qindex_calc`)
- **Flat-IPP P frames:** the configured qindex, with no offsets.
- **Key frame:** `q_target = q·(1−0.76)`, i.e. quantiser step ×0.24. The table
  is `md_process.c: percents[1] = {76, 60, 30, 15, 8, 4}`.
- **With layers:** TL0 ×0.40, TL1 ×0.70, top layer (non-reference) ×1.0.
  - Layers above 0 also get `non_base_boost`: qstep reduced by
    `(ref intra-SB %)/4` %.
  - These steps are converted to qindex deltas with `svt_av1_compute_qdelta`.

### CBR rtc (`rc_rtc_cbr.c`)
- **Model:** `bits = rcf · max(me_dist, 1024) · sqrt(q_ref)/q / q`. For key
  frames, complexity = 5600 and the reference scale is 1.
- The frame qindex is found by binary search against the target, bounded by the
  reference q_idx −16…+64 (−0…+128 for higher layers).
- `rcf` is Kalman-filtered in the log domain, per mini-GOP position (Q = 0.04,
  adaptive R ≥ 0.04).
- **Flat IPP uses a virtual 3-layer bit allocation** (`calc_pframe_target_size`;
  base weight `w0 = −9.32·rcf + 0.023·avg_q + 0.034·avg_zeromv + 8.14`).
- Leaky-bucket buffer, one recode if the VBV would overflow, frame-skip credits.
- **Cyclic refresh** (`rtc_cyclic_refresh_init`,
  `rc_aq.c: svt_aom_cyclic_refresh_setup` / `cyclic_sb_qp_assignment`):
  - CBR only, base-layer inter frames only, SB64.
  - Requires average qindex in [16, 118·255/128] and
    `avg_frame_low_motion ≥ 50`.
  - 20% of SBs per frame in a round-robin band. Only "static" SBs (|ME MV| ≤ 1
    and 8×8 distortion < 2·mean) are boosted.
  - Segment 1 gets a `rate_ratio = 2.7 + avg_q/170`, capped at 4× bits.
    Segment 2 (below-mean distortion) gets a larger boost (`rate_boost_fac` 18
    plus a quadratic term in the deviation), capped at 8×.
  - The delta is limited to −60% of qindex.
- AQ mode 0 was used in our runs; variance boost and TPL were off.

---

## 7. M10 → M13 delta list (rtc, flat IPP unless noted)

| Feature | M10 | M11 | M12 | M13 |
|---|---|---|---|---|
| Deblocking | level 7 (from Q) | off | off | off |
| ME window (<1080p) | {24,16}→{32,16} | {8,1} (flat) | {8,1} | {8,1} |
| Pre-HME | 4 | 0 (flat) | 0 | 0 |
| HME level 1 | on | on | on if var < 1000 | off |
| CDEF on P frames | 7 | 7 | 7 if var < 350, else 9 | 9 |
| Interpolation filter | switchable search (level 4; off when ref skip % high) | REGULAR fixed | REGULAR | REGULAR |
| RDOQ | I frames only | off | off | off |
| Coefficient rate estimate | real (level 1) | linear eob (0) | 0 | 0 |
| `disable_cdf_update` cycling | no | yes | yes | yes |
| Chroma level | 4 (independent UV on intra) | 5 | 5 | 5 |
| TX type search | 9 | 9 | off | off |
| TX size search | I frames (4) | I frames (4) | off | off |
| Depth removal | 5 | 6 | 6 if var < 350, else 7 | 7 |
| Light-PD1, base frames | regular PD1 | 2 | 4 | 4 if var < 1000, else 6 |
| Light-PD1, other frames | 3 / 4 by var < 1000 | 3 / 5 | 5 / 8 | 5 (<350) / 6 (<1000) / 8 |
| Subpel floor | 6 | 6 | 6 | 9 |
| PD0 bias weight | 0 | 0 | 0 | 600 |
| Intra levels | 4/6, ang 2 | DC only | DC only | DC only |
| SB λ modulation | on | off | off | off |
| Candidate buffer size (memory only) | 65 | 15 | 10 | 10 |

The same at every level: 8×8 disallowed from M10 up, NIC 11, PD0 = VLPD0 from
M10 up, `skip_intra` gate from M9 up.

**So M12 is essentially M13 plus protections for low-variance frames (HME L1,
CDEF 7, depth removal 6, gentler light-PD1) and better subpel.** On
high-variance content M12 and M13 differ only in subpel and the PD0 bias, which
fits the measured gap: −48.6% vs −46.5%, CPU 1.24× vs 1.00× on bodycam.

---

## 8. Where the time likely goes [inferred — not profiled]

**P frames at M12 per SB:**
- ME: zz-SAD, HME L0 over 512 positions at ¼ resolution with half the rows, 24
  full-res positions. Small.
- PD0: 21 variance evaluations. Very small.
- Per final block:
  - MVP stack construction (scalar, branchy);
  - ≤8 bilinear subpel variances;
  - 2–4 8-tap luma predictions;
  - one luma transform/quant/inverse plus distortion;
  - chroma prediction and transforms;
  - reconstruction copy.

**Per frame:** CDEF (direction search and filtering; the search only on base
frames) and entropy coding (scalar).

**Likely ranking:** inter prediction (luma 8-tap for each candidate plus chroma),
then transform/quant/inverse transform, then CDEF, then entropy coding with MVP
and context logic, then ME.

**SIMD on the hot path** (dispatch through `aom_dsp_rtcd.h` /
`common_dsp_rtcd.h`):
- `svt_ext_all_sad_calculation_8x8_16x16`,
  `svt_ext_eight_sad_calculation_32x32_64x64`, `svt_sad_loop_kernel`,
  `svt_nxm_sad_kernel`
- `svt_aom_variance{N}`, `svt_aom_sub_pixel_variance{N}`
- `svt_av1_convolve_2d_sr` / `_x_sr` / `_y_sr` / copy
- `svt_residual_kernel8bit`, `svt_av1_fwd_txfm2d_{N}`, `svt_aom_quantize_b`,
  `svt_av1_inv_txfm2d_add_{N}`, `svt_full_distortion_kernel32_bits`
- `svt_aom_cdef_find_dir(_dual)_8bit`, `svt_cdef_filter_block_8bit`,
  `svt_aom_compute_cdef_dist_8bit`
- `svt_aom_downsample_2d`, `svt_compute_mean_8x8`, `svt_av1_get_nz_map_contexts`

---

## 9. Mapping onto our engine

### Copy algorithmically (shared engine)
1. **ME:** the zz-SAD gates (1, 1.33 and 2 per pixel), HME L0 with 2×2
   quadrants at ¼ resolution, a tiny 8×3 full-res refinement, the single
   all-blocks 8×8-to-64 SAD kernel, and `me_8x8_cost_variance`. It is simpler
   than our planned EPZS search; start with SVT's version and add EPZS
   predictors only if measurements show a gain.
2. **Partitioning:** VLPD0 as specified — variance of the full-pel
   ME prediction on the reconstructed reference, capped at λ/1024 per pixel,
   doubled split rate, and the QP/variance parent bias in [900, 1200]. Plus
   `depth_removal` and the max-size cap. No ±1 refinement at our default tier.
3. **Mode decision:** {NEAREST, 1×NEAR, NEW} plus gated DC; residual-variance
   ranking; one survivor; the transform-skip score, block-skip RD, chroma energy
   skip; coefficient shaving; the linear eob rate. This is close to our original
   plan (SATD ranking, 1–2 survivors) but cheaper: variance instead of SATD,
   and one survivor.
4. **Fixed-stage subpel** (4 half-pel + 4 quarter-pel, bilinear `svf`, 110% bias,
   skip on static references) and the per-SB reference statistics (`sb_intra`,
   `sb_skip`, `sb_64x64_mvp`, me_dist/var) that drive the detectors.
5. **The RTC RC model** (ME-distortion quadratic model with a Kalman rcf) is
   worth trying against R-λ, since it reuses our ME output directly.

### Shareable with HEVC and H.264
- ME, partitioning and the candidate/skip detectors all carry over.
  - HEVC: NEAREST/NEAR become merge candidates 0/1, NEW becomes AMVP.
  - H.264: run VLPD0 at 16×16 with 16×8/8×16 shapes.
- Coefficient shaving maps onto openh264-style isolated-±1 elimination.
- CDEF "search on base frames, reuse on others, QP model for chroma" maps onto
  SAO strength reuse.
- **Do not copy "deblock off" to HEVC or H.264 [inferred]:** they have no CDEF
  to cover for it, and their deblocking is cheap.

### Skip, or do better, for live camera content
- **Keyframes are DC-only with no TX split.** That is weak; adding H/V/planar
  shortlists to our I-frame path is a cheap win. GDR makes it matter less.
- **Cyclic refresh did not contribute to the −49%.** It was off in our CQP runs,
  and in SVT it is base-layer and CBR-only. Measure it separately in CBR.
- **No LTR or background reference in flat rtc** (one LAST reference). For
  static surveillance scenes, a background LTR and a guarded static fast path
  are things SVT does not do.
- **The static-block test is only "zz-SAD < 1 per pixel".** Our pre-analysis
  background map, with a chroma guard and noise awareness, should do better.
  Sensor noise easily exceeds 1/px, so SVT's bypass rarely fires on camera
  content [inferred].
- **Skip as non-applicable at this tier:** TPL, TF, palette/IBC,
  OBMC/warp/global motion, compound, MFMV, RDOQ, TX type/size search, loop
  restoration.

### Two configuration notes
- Add `--rc 1` (CBR) and default-`--hierarchical-levels` lines to the reference
  set, so we also have SVT's *actual* rtc behaviour (cyclic refresh,
  virtual-layer RC, 2-level LD).
- Flat-IPP results already include SVT's hidden 4-frame effort cadence.
