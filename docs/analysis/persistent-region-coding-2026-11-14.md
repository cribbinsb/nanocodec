# Persistent-region coding: what the still-scene bursts really do, and a deliberate design (2026-11-14)

**Question (owner).** "It sounds like we are accidentally doing a half job of predicting and encoding future
static regions better? I think we need a stronger analysis, I think we are missing something here."

**Short answer.** Yes. On still scenes our quality comes from refining the static background, and the base
frames that the rate controller happens to land deep do that refining. The valuable refinements are the deep
ones early in a key period. Most burst bits are spent later, at lower value. Some of them go into the second
before a periodic key, which then throws the background quality away (its still-region gain is negative). A
deliberate design needs three parts. Each part is measured separately below:

- (a) no refinement just before a scheduled key;
- (b) a still scene's periodic key planned at the quality the picture holds, with its loan repaid afterwards;
- (c) loan-funded refinement, planned per block by forward value, while ordinary frames hold still blocks at
  the quality they already have.

Measured so far (AV1 and H.264 Default, six still points and two moving controls, section 5):

- (b) gains on every still point, at the cost of a larger 1 s window around the key.
- The hold gains on H.264, but not on every point.
- (c) as built does not pass.

The root cause of the bursts is not the cadence. It is the inter level falling below the quality the still
picture holds (section 4). Fixing it in AV1 needs per-8x8 quantizers (segmentation), because AV1's 64x64
delta-q dilutes the hold.

Build: worktree `persistence` (branch `analysis/persistence`), base `variance-integration` 7fd7eb1e. All
figures: 600 frames from frame 0, a key every 10 s, CBR on the clip's real-use ladder, 1 s buffer, our
reconstruction. Scripts and per-run JSON: `~/nanocodec-data/work/persistence/` (`persist_probe.py`,
`summarize.py`, `compare.py`, `out/`, `arms/`).

## 1. The mechanism, quantified

### 1.1 Method

`persist_probe.py` encodes a point with `--display-recon` into a FIFO, so no files are kept. It reads the
source alongside and follows every 8x8 luma cell:

- **Persistence:** whether the cell's reconstruction changed from the previous frame, and the run lengths
  of unchanged copies.
- **Inherited gain of a change:** the squared error it saves against the counterfactual "the cell kept its
  previous copy", summed over every frame until the cell changes again. It is credited to the frame that
  made the change.
- **Validity:** the counterfactual holds only where the source was still at the change (mean
  |src_t − src_t−1| < 1.5), so only those cells are counted. Moving-cell gains are listed apart.
- **Unit:** gains are given in "dB-frames": 4.343 · ΣΔSSE / (pixels · mean MSE). That is the change in mean
  PSNR-Y over the clip times its frame count.
- **Frame classes:**
  - key;
  - base burst: a cadence base frame over one cycle budget (4U, where U = target/fps);
  - cheap base;
  - other P.

### 1.2 Results (AV1 Default, `summarize.py out/*-av1.json`)

| Point | copies (cell-frames) | burst frames | burst share of bits | burst share of still gain | periodic key's share of still gain | still gain per kbit, bursts / other P |
|---|---:|---:|---:|---:|---:|---:|
| chirla 47 | 92.7% | 22 | 59% | 88% | −5% | 1.42 / 0.49 |
| chirla 187 | 91.1% | 31 | 73% | 94% | −9% | 0.29 / 0.21 |
| meva admin 97 | 92.2% | 27 | 66% | 102% | −20% | 0.33 / 0.19 |
| meva admin 388 | 89.5% | 27 | 82% | 98% | −8% | 0.09 / 0.08 |
| meva hospital 53 | 94.2% | 15 | 64% | 127% | −41% | 1.13 / 0.37 |
| meva hospital 212 | 92.1% | 25 | 70% | 84% | −6% | 0.31 / 0.30 |
| otw_homes_00004 150 | 87.5% | 28 | 68% | 95% | −1% | 0.15 / 0.02 |
| otw_homes_00004 600 | 85.0% | 23 | 59% | 88% | −1% | 0.05 / 0.01 |
| chad_4_109 150 (people moving) | 83.1% | 12 | 36% | 108% | 0% | 0.15 / −0.03 |
| pan_jaad_0194 150 (control) | 25.9% | 22 | 27% | — (negligible) | — | 0.004 / 0.000 |
| crowd mot20_02 85 (control) | 43.4% | 23 | 19% | — (small) | — | 0.07 / 0.01 |

What the bursts refine, and when:

- **Only the static background.** 89-100% of the cells a burst changes on the still points are cells whose
  source was still. The moving parts get essentially nothing.
- **Value is concentrated early in the key period.**
  - The best quarter of bursts, ranked by gain, gives 50-87% of all burst gain with 27-42% of burst bits.
    These bursts sit 8-60 frames after a key (a few later).
  - The other 58-73% of burst bits come a median 166-208 frames after the key. They buy 13-50% of the gain.
  - On 5 of the 8 still points, 51-79% of burst bits buy less gain per kbit than ordinary P frames do.
- **Refinement is repeated in increments.** A burst's refinement survives a median of 4-24 frames before the
  next burst refines the same cells again.
- **Some bursts refine below the noise.** On chirla 187, frames 88 and 168 at qindex 16-18 survived a median
  of 2-12 frames.
- **The periodic key throws the background away.**
  - 8-20% of burst bits land in the 30 frames before a periodic key.
  - The key then codes the background coarser than it was held, so its still-region gain is negative.
  - chirla 187: frame 299 is a 4.2U burst at qindex 18 (46.5 dB). The key at frame 300 is coded at qindex 126
    for 16U and gives 38.7 dB. It is starved by 0.93 buffer of excess.

**How much of the static-scene quality comes from long-persistent regions.** On the still points, 85-94% of
cell-frames are unchanged copies. Nearly all (84-127%) of the background's quality gain after a key comes
from deep refinements of static background that later frames copy. A deliberate version could do better in
two ways (an estimate from these numbers, not measured):

- refining deeply once, early in a key period, would keep the best quarter's 50-87% of the gain for about a
  third of the burst bits;
- not refining in the run-up to a key would avoid discarding 8-20% of them.

## 2. How mature real-time encoders do it (read from source; BSD-style only)

| Encoder | Signal without lookahead | Sizing and schedule | Rate-control accounting | Prediction |
|---|---|---|---|---|
| libvpx VP9 (`vp9_aq_cyclicrefresh.c`, `vp9_ratectrl.c`) | per-8x8 `consec_zero_mv`. A block qualifies if its last coded q is coarser than the boost q or it has under 100 zero-mv frames (60 if noisy). Also the running `avg_frame_low_motion`. | 10% of blocks per frame (5% if few qualify), walking superblocks round-robin. Delta from `rate_ratio_qdelta` 3.0 for the first 4·(100/percent) frames after a key, then 2.0, capped at 60% of q. Off at very fine or very coarse average q, at low-motion share < 20, and above temporal layer 0. | Base q from a segment-weighted bits/MB (`vp9_cyclic_refresh_rc_bits_per_mb`). The post-encode estimate uses the actual segment fractions, so the correction factor is not polluted. Golden target = avg·N·pct/(100N+pct−100), others reduced so N frames spend N·avg (`vp9_calc_pframe_target_size_one_pass_cbr`). `gf_cbr_boost_pct` is 0 by default. | Golden = LAST copied every min(4·100/percent, 40) frames (`vp9_cyclic_refresh_set_golden_update`). Refresh is cancelled on frames with low-content share < 0.65. Golden NEWMV is skipped. |
| libvpx VP8 (`onyx_if.c`, `ratectrl.c`) | `zeromv_count`, LAST (0,0) | `cyclic_background_refresh` MBs/20 per frame, segment Q = Q/2, open loop | golden refreshed when zero-mv > half the MBs; target × (100+`gf_cbr_boost_pct`)/100, with no cut elsewhere | golden every 6-40 frames |
| libaom RT (`aq_cyclicrefresh.c`, `ratectrl.c`) | as VP9, plus a per-SB source SAD (always refresh SBs below 2·64², never above 6-8·64²) | 10 ± 5% of blocks, ratio 3.0 early then 2.25 (+0-0.25). Closed loop: a correction factor > 1.25 lowers both, < 0.5 raises them. | same segment-weighted model. `av1_adjust_gf_refresh_qp_one_pass_rt` refreshes golden only from a frame finer than average (cancel if base_q > avg_q; force if base_q < 0.87·avg_q) | GOLDEN in a fixed slot, refreshed every ~80 frames (16 if low motion < 40). ALTREF lags 3-6 frames. |
| SVT-AV1 `--rtc` (`rc_rtc_cbr.c`, `rc_aq.c`, `pd_process.c`) | ME vector within ±1 and 8x8 distortion < 2× average; `avg_frame_low_motion` ≥ 50 to enable | 20% of SBs as delta-q. Ratio 2.7 + avg_q/170, capped at 4; a second tier below average distortion, capped at 8. Delta solved from the R-Q model, capped at −60%. | `av1_estimate_frame_size` blends the tiers by their ME distortion, so the base q is searched against the target. The boost is self-funding. Layer weights rise with the zero-mv share. | flat IPP uses the last 4 frames. Layered: GOLD = oldest base, LAST3 a long base reference refreshed every 128 frames. |
| openh264 (`BackgroundDetection.cpp`, `svc_mode_decision.cpp`, `ref_list_mgr_svc.cpp`) | 16x16 background map from MAD/SAD | **no boost**: background blocks are forced to P-skip when the reference QP is within 3 of the current one | rate-neutral AQ (mean removed) | LTR only after loss feedback |
| VVenC / JVET LD-P (`lowdelay_medium.cfg`, `EncSlice.cpp`) | none (fixed structure) | GOP 8, QP offsets by POC 5,4,5,4,5,4,5,**1**: every 8th frame about 4 QP finer | single-pass RC needs lookahead (not real time) | references: the previous frame **plus the last three GOP-end (high-quality) frames** (POC −1, −9, −17, −25) |
| x264 MB-tree / HEVC-AV1 TPL (documentation only) | lookahead propagation: propagate = (intra + in)·(1 − inter/intra) | ΔQP = −strength·log2(1 + propagate/intra), strength 5·(1−qcomp) = 2 | per-block QP within the frame budget | — (needs lookahead, off with zero latency) |

What they share:

1. **The signal is history.** It is zero-motion counts and low source SAD, never a forecast. Our still runs,
   noise map and reuse term already hold the same signal.
2. **The refinement is stopped once done.** libvpx and libaom skip a block already coded at or finer than the
   boost q, and libaom refreshes golden only from a better-than-average frame. Our cadence refines every 4th
   frame whatever the picture already holds.
3. **The boost is budgeted, never discovered.** The cyclic-refresh encoders put the boosted segment's bits
   into the base-q search (segment-weighted bits). libvpx's golden target spreads its boost over the golden
   interval (others get less). Our bursts are unplanned: the size model mispredicts them 13-22×.
4. **The high-quality copy is kept as a reference.** VVenC/JVET LD-P keep the last three GOP-end frames as
   references. libvpx, libaom and SVT keep a golden or long reference. So the background survives while LAST
   tracks motion. Our `refs.second` does this (H.264 only, off: +16% CPU for −1.4% J_real on a deep cadence,
   most of it one class).
5. **None of them re-codes a periodic key from scratch in a still scene in their default real-time setups.**
   WebRTC streams run very long key intervals and recover with LTR or intra refresh. That is outside this
   note's 10 s-key test conditions, but relevant to (b) below.

The standards lens's low-delay configurations (JVET LD-P, AOM CTC LD) are exactly a fixed, scheduled
"refinement frame" every 8 or 16 frames, kept as an extra reference. Their gain over flat IPP comes from the
schedule and the long reference, not from luck. Our fixed-QP cadence (`rate.cadence_fixed_qp`) copies the
schedule but not the reference.

## 3. Our design against theirs: what we are missing

| Piece | Ours today | Missing |
|---|---|---|
| (a) Persistence estimate per block | still runs, noise map, reuse term per-cell error (`reuse.rs`). `persist.strength` (motion-error EWMA, zero-mean redistribution every frame) is off: studies found +4-8% J_real WORSE, mostly bodycam (`docs/handoff/block-aq.md`), and an mbtree-style one-step reuse forecast fared no better | Nothing is missing in the signal. The failures were zero-mean, every-frame redistributions. The bursts' value comes from the **timing and depth** of refinement, not from a per-frame tilt. |
| (b) Scheduled, rate-controlled refinement | none: base frames refine when the level's limit cycle lands them deep | a refinement planned when and where it pays, sized and budgeted |
| (c) A second reference | `refs.second` (H.264 only, off) | the high-quality background as a reference that survives LAST tracking motion. This matters most for moving people over a static background (meva). |
| (d) Knowing when refinement is pointless | reuse term noise floor; quality cap; noise skip | ordinary frames are free to refine still blocks whenever the level falls below what they hold. That is the root cause (section 4). |
| (e) Accounting | key loan only. Bursts become debt and the correction repays them at once, which starves later frames and the next key | a planned loan repaid on schedule, like the key's |
| Periodic key | starved by debt and the 1.4-buffer re-code room; re-codes the held background coarser | a key planned at the held quality, or no from-scratch key on still scenes (LTR recovery) |

## 4. Root cause found while building the experiment

With the cadence flat (no base boost), the bursts did not go away. They moved to ordinary frames (AV1 chirla
93: frames of 10-14U at the inter level, against a 0.5U budget):

1. On a still scene ordinary frames copy nearly everything and cost almost nothing.
2. The controller therefore lowers the inter level until it is finer than the quality the still picture
   holds.
3. The next frame that codes a still block at that level refines it. The first one that codes many refines
   the whole picture.

The cadence's base frames are just the most likely frames for this to happen, which is why the bursts look
like a cadence property. **The fix is that ordinary frames hold still blocks at the quality they have; only
planned refinements refine.** Part (c) of the experiment implements this (`ReuseTerm::hold_still`): each still
cell is held at the finest step it was coded at since its source settled.

## 5. Experiment (registered, all off by default)

Code (branch `analysis/persistence`, head 70d8d277, not landed). With every switch off, all 56 streams
(every codec's Default and every shipped preset, two 720p clips) are byte-identical to the base build. The
switches are separate, so each can be screened alone or together:

- `rate.hold_still` (0/1): ordinary flat inter frames hold still blocks at the finest step they were coded at
  since their source settled (`ReuseTerm::hold_still`, planned by `StillPlan` in `planned_refinement.rs`).
  The frame's level is planned, and its model learnt, without those offsets. This is the root-cause fix of
  section 4.
- `rate.prekey_quiet` (s): in the last N s before a scheduled periodic key, no cadence base boost
  (`StillRate::quieted`) and the hold. Part (a).
- `rate.key_hold_still` (buffers): a periodic key of a still scene (running unchanged share ≥
  `rate.key_repay_still_from`) is planned at the held level. It is bounded by the strict buffer, with room of
  at least N buffers whatever the debt, and not by its repay phase or the correction; the frames after it repay
  it (`key_step`, `key_allowance`, `strict_room`). Part (b).
- `rate.refine` with `refine_budgets`, `refine_min_budgets`, `refine_interval`, `refine_horizon`,
  `refine_payback`, `refine_max_offset`, `refine_startup`. Part (c):
  - `planned_refinement.rs` decides when: at most once per interval, never within the horizon of the next key.
  - `ReuseTerm::plan_refinement` decides where. It ranks still cells by forward value per estimated bit,
    R·(error − D(price step)) / bits. The price is λ/(1+R), with R = min(still run, frames to the next key),
    never below the cell's noise step. The bits are a high-rate estimate.
  - `refinement_loan.rs` (`RefinementLoan`, inside `StillRate`) pays for it. The loan is on top of the frame's ordinary budget, bounded by
    `refine_budgets`·U and the strict room. It is scheduled debt (outside the correction's excess), repaid
    over `refine_payback` by equal cuts of later inter budgets, the way a key's loan is. Its cost scale is
    learnt from what refinement frames really cost beyond their ordinary prediction.
  - The frame's level is planned without the refined blocks.
  - The hold (`rate.hold_still`) is on with it, on every frame including refinement frames.
  - The steady-state cadence (and with `refine_startup` 0 the start-up cadence) gets no base boost.

### 5.1 Measurements

Setup: 600 frames from frame 0, a key every 10 s, our reconstruction, `compare.py` / `table.py` over
`arms/`. Each cell lists:

- rate (achieved / target, 600 f);
- max 1 s window (bits / target over the worst 30-frame window, the key's included);
- max U (largest inter frame in uniform budgets);
- re-coded frames;
- Δ mean PSNR-Y;
- Δ worst-1% PSNR-Y (mean of the worst 1% of frames);
- Δ PSNR-Y over frames 300-359 (the window after the second key).

Arms:

- `quiet`: `rate.prekey_quiet=1`, before the hold existed, so no base boost in the last second only.
- `qhold`: the same with the hold in that second.
- `hold`: `rate.hold_still=1`.
- `holdkeyN`: `rate.key_hold_still=N`.
- `refine`: `rate.refine=1`, which includes the hold.
- `all`: refine, quiet and holdkey2.

**AV1 Default**:

| Point | Default: rate / max 1 s / max U / re-coded / mean Y / worst-1% Y / Y f300-359 | `quiet`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 | `qhold`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 | `hold`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 | `holdkey2`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 | `refine`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 |
|---|---|---|---|---|---|---|
| chad 150 | 0.973 / 2.12 / 31 / 3 / 24.55 / 21.46 / 23.94 | 0.971 / 2.11 / 30 / 3 / -0.02 / +0.00 / +0.12 |  |  | 0.978 / 2.63 / 30 / 3 / +0.09 / +0.00 / +0.65 | 0.985 / 2.33 / 32 / 4 / -0.59 / +0.03 / -0.05 |
| chad 600 | 0.978 / 1.86 / 31 / 1 / 30.23 / 27.40 / 28.86 | 0.973 / 1.91 / 31 / 1 / +0.01 / +0.00 / +0.30 |  |  | 0.978 / 2.12 / 31 / 1 / +0.08 / +0.00 / +0.77 | 0.990 / 2.01 / 21 / 1 / -0.70 / -0.04 / -0.30 |
| chirla 47 | 1.001 / 1.97 / 40 / 22 / 38.52 / 28.42 / 33.95 | 0.998 / 1.91 / 40 / 22 / +0.59 / +3.09 / +1.83 | 0.985 / 2.01 / 40 / 19 / +0.39 / +3.11 / +1.48 | 0.994 / 1.97 / 37 / 32 / -0.14 / +0.08 / -0.35 | 0.984 / 3.35 / 40 / 21 / +0.74 / +3.11 / +4.62 | 1.008 / 2.06 / 36 / 37 / -0.60 / -0.78 / -0.90 |
| chirla 93 | 1.037 / 2.20 / 39 / 8 / 42.67 / 34.96 / 37.77 | 0.997 / 2.07 / 40 / 8 / +0.29 / +0.00 / +1.92 | 1.010 / 2.08 / 32 / 8 / +0.33 / +0.00 / +2.27 | 1.028 / 2.18 / 36 / 15 / -0.17 / -1.81 / -1.15 | 1.020 / 2.63 / 34 / 7 / +0.29 / +0.00 / +2.31 | 1.031 / 1.63 / 28 / 26 / +0.18 / -0.19 / +0.09 |
| chirla 187 | 1.015 / 1.80 / 30 / 23 / 45.20 / 38.55 / 42.07 | 1.012 / 2.02 / 30 / 22 / -0.48 / -5.03 / -4.03 | 1.059 / 1.80 / 31 / 22 / -0.21 / -2.99 / -2.27 | 1.058 / 2.00 / 30 / 17 / +0.06 / -0.07 / -0.64 | 1.050 / 2.41 / 32 / 24 / +0.36 / +0.00 / +2.87 | 1.025 / 2.04 / 33 / 19 / -0.05 / +0.00 / -0.53 |
| crowd 85 | 0.991 / 1.61 / 13 / 0 / 26.51 / 24.73 / 25.72 | 0.992 / 1.69 / 13 / 0 / +0.07 / +0.15 / +0.07 | 0.992 / 1.69 / 13 / 0 / +0.07 / +0.21 / +0.11 | 0.991 / 1.61 / 12 / 0 / -0.02 / +0.05 / -0.01 | 0.991 / 1.61 / 13 / 0 / +0.01 / +0.09 / -0.00 | 0.993 / 1.61 / 6 / 0 / -0.05 / +0.13 / +0.10 |
| meva 97 | 0.984 / 2.05 / 30 / 16 / 35.61 / 29.03 / 32.49 | 0.969 / 2.09 / 30 / 21 / -0.14 / +0.06 / +0.04 | 0.933 / 2.12 / 30 / 22 / -0.07 / +0.06 / +0.03 | 0.965 / 1.91 / 30 / 22 / -0.18 / -0.35 / -1.22 | 0.958 / 2.47 / 30 / 14 / +0.16 / +0.13 / +0.62 | 1.018 / 1.99 / 33 / 13 / -0.45 / -1.67 / -3.22 |
| meva 194 | 1.033 / 2.12 / 37 / 8 / 39.01 / 32.57 / 35.04 | 1.009 / 2.12 / 37 / 9 / +0.09 / +0.09 / +0.33 | 0.993 / 2.23 / 40 / 10 / +0.02 / -0.00 / -0.43 | 0.994 / 1.95 / 36 / 14 / -0.66 / -3.26 / -1.20 | 1.027 / 2.53 / 37 / 9 / +0.14 / +0.09 / +0.94 | 1.015 / 1.91 / 30 / 16 / -0.50 / -3.00 / -2.49 |
| meva 388 | 1.028 / 2.13 / 37 / 7 / 41.20 / 33.64 / 36.68 | 0.997 / 2.13 / 37 / 4 / +0.30 / +2.51 / +2.67 | 0.992 / 2.13 / 37 / 6 / +0.26 / +2.51 / +2.59 | 1.002 / 2.29 / 34 / 3 / +0.07 / +2.39 / +1.54 | 0.984 / 2.44 / 37 / 6 / +0.19 / +2.51 / +2.06 | 1.028 / 2.17 / 29 / 13 / -0.12 / +1.86 / +0.74 |
| pan 150 | 0.988 / 1.85 / 17 / 0 / 24.07 / 20.50 / 23.82 | 0.988 / 1.96 / 20 / 0 / -0.00 / +0.03 / +0.19 | 0.988 / 1.95 / 20 / 0 / -0.01 / +0.02 / +0.15 | 0.988 / 1.86 / 17 / 0 / -0.00 / +0.03 / +0.03 | 0.988 / 1.85 / 17 / 0 / +0.00 / +0.00 / +0.00 | 0.985 / 1.42 / 9 / 0 / -0.10 / -0.14 / -0.03 |

AV1 `refine` is the first build (refinement frames did not hold their other still blocks). Its rerun after that fix was dropped because AV1's 64x64 pooling fails the hold anyway (see `hold`).

**H.264 Default**:

| Point | Default: rate / max 1 s / max U / re-coded / mean Y / worst-1% Y / Y f300-359 | `qhold`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 | `hold`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 | `holdkey2`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 | `refine`: rate / max 1 s / max U / re-coded / Δmean / Δworst-1% / Δf300-359 |
|---|---|---|---|---|---|
| chirla 47 | 0.987 / 2.15 / 30 / 12 / 38.63 / 30.24 / 35.30 | 0.994 / 2.18 / 30 / 16 / +0.22 / +0.04 / +0.96 | 0.998 / 2.30 / 34 / 19 / +0.09 / +0.04 / +1.19 | 0.996 / 2.40 / 30 / 19 / +0.30 / +0.04 / +1.32 | 1.024 / 2.17 / 35 / 21 / -0.94 / -3.62 / -3.57 |
| chirla 93 | 0.987 / 2.00 / 32 / 4 / 42.07 / 33.17 / 36.69 | 0.984 / 1.88 / 31 / 5 / +0.55 / +0.36 / +2.30 | 1.009 / 2.02 / 32 / 3 / +0.14 / +0.66 / +1.98 | 0.991 / 2.29 / 32 / 4 / +0.17 / +0.66 / +0.86 | 1.016 / 1.78 / 25 / 10 / +0.19 / +0.60 / +0.10 |
| chirla 187 | 0.997 / 2.06 / 32 / 2 / 45.31 / 37.31 / 43.00 | 0.984 / 2.04 / 30 / 4 / +0.01 / +0.00 / +1.10 | 1.007 / 1.91 / 29 / 3 / +0.01 / +0.00 / +0.63 | 1.008 / 2.16 / 31 / 2 / +0.10 / +0.00 / +0.81 | 1.018 / 1.76 / 24 / 5 / -0.39 / +0.00 / -1.08 |
| crowd 85 | 0.989 / 1.57 / 5 / 0 / 24.97 / 23.07 / 24.12 | 0.989 / 1.57 / 5 / 0 / +0.00 / +0.00 / +0.00 | 0.989 / 1.57 / 5 / 0 / +0.00 / +0.00 / +0.00 | 0.989 / 1.57 / 5 / 0 / +0.00 / +0.00 / +0.00 | 0.989 / 1.57 / 5 / 0 / +0.00 / +0.00 / +0.00 |
| meva 97 | 0.962 / 2.27 / 28 / 16 / 34.59 / 26.71 / 29.89 | 0.988 / 2.28 / 35 / 19 / +0.09 / +0.44 / -0.01 | 1.001 / 2.06 / 37 / 14 / +0.58 / +1.70 / +1.69 | 0.974 / 3.41 / 28 / 17 / +0.55 / +1.70 / +2.28 | 1.034 / 1.98 / 27 / 19 / -0.15 / -0.85 / -1.94 |
| meva 194 | 1.006 / 2.18 / 37 / 6 / 38.37 / 29.34 / 31.86 | 0.974 / 2.19 / 37 / 10 / +0.34 / +1.75 / +2.94 | 1.006 / 2.28 / 41 / 8 / +0.31 / +1.52 / +2.05 | 1.007 / 2.74 / 37 / 9 / +0.35 / +1.75 / +2.72 | 1.022 / 2.02 / 34 / 10 / -0.25 / +0.60 / -0.02 |
| meva 388 | 1.027 / 2.01 / 33 / 3 / 41.26 / 34.80 / 38.21 | 0.978 / 2.18 / 38 / 4 / -0.00 / -0.94 / -0.29 | 1.002 / 2.26 / 32 / 3 / -0.28 / -0.85 / -1.75 | 0.994 / 2.01 / 31 / 4 / +0.08 / +0.14 / +0.52 | 1.023 / 2.01 / 28 / 6 / -0.34 / -1.17 / -1.60 |
| pan 150 | 0.989 / 1.26 / 7 / 0 / 22.22 / 17.70 / 22.20 | 0.990 / 1.32 / 7 / 0 / +0.00 / +0.02 / +0.06 | 0.989 / 1.25 / 7 / 0 / -0.00 / +0.03 / -0.05 | 0.989 / 1.26 / 7 / 0 / +0.00 / +0.00 / +0.00 | 0.989 / 1.30 / 7 / 0 / -0.01 / +0.03 / -0.11 |

**(b) trade-off: worst 1 s window against post-key quality** (AV1, still points):

| Point | N = 0 (Default): max 1 s / Y f300-359 / mean Y / rate | N = 1.5: max 1 s / Y f300-359 / mean Y / rate | N = 2: max 1 s / Y f300-359 / mean Y / rate | N = 3: max 1 s / Y f300-359 / mean Y / rate |
|---|---|---|---|---|
| chirla 47 | 1.97 / 33.95 / 38.52 / 1.001 | 2.70 / 36.56 / 38.96 / 0.986 | 3.35 / 38.57 / 39.26 / 0.984 | 3.92 / 40.13 / 39.88 / 0.983 |
| chirla 93 | 2.20 / 37.77 / 42.67 / 1.037 | 2.15 / 38.98 / 42.83 / 1.006 | 2.63 / 40.08 / 42.96 / 1.020 | 3.14 / 41.70 / 43.05 / 1.040 |
| chirla 187 | 1.80 / 42.07 / 45.20 / 1.015 | 2.17 / 43.88 / 45.47 / 1.053 | 2.41 / 44.94 / 45.56 / 1.050 | 2.75 / 45.71 / 45.65 / 1.049 |
| meva 97 | 2.05 / 32.49 / 35.61 / 0.984 | 2.10 / 32.74 / 35.56 / 0.898 | 2.47 / 33.10 / 35.77 / 0.958 | 2.53 / 33.17 / 35.74 / 0.998 |
| meva 194 | 2.12 / 35.04 / 39.01 / 1.033 | 2.16 / 34.57 / 39.03 / 0.990 | 2.53 / 35.97 / 39.15 / 1.027 | 2.78 / 36.56 / 39.22 / 1.026 |
| meva 388 | 2.13 / 36.68 / 41.20 / 1.028 | 2.27 / 38.09 / 41.33 / 1.002 | 2.44 / 38.75 / 41.38 / 0.984 | 2.44 / 38.75 / 41.38 / 0.984 |

Each cell: worst 1 s window (× target) / PSNR-Y f300-359 / mean PSNR-Y / rate. N = 1 changed nothing on chirla 93: there the strict room is already at about one buffer, and a key at the held quality needs more.

### 5.2 Reading

- **(b) `key_hold_still` is the only part that gains on every still point at N = 2.**
  - Gains, both codecs: mean PSNR-Y +0.1 to +0.7 dB, worst-1% 0 to +3.1 dB, the window after the key
    +0.5 to +4.6 dB. The moving controls are unchanged (their running still share keeps it off).
  - The curve is not monotone at small N: at N = 1.5, meva 194's post-key window is 0.5 dB *lower*.
  - Price: the key's own 1 s window, from 1.8-2.3× target to 2.0-3.4× target at N = 2 (2.4-3.9× at N = 3).
  - Rate over 600 f: 0.96-1.05 of target. AV1 chirla 187 ends at 1.050 against the Default's 1.015, so that
    point fails the long-window ceiling.
  - The N curve above is the owner's trade-off.
  - Re-codes and the largest inter frame do not change: (b) does not touch the bursts.
- **`hold` / `qhold` (the root-cause fix, alone or in the last second before a key) gain on H.264 but not
  cleanly.**
  - H.264 `qhold`: mean +0.0 to +0.55 dB, the window after the key −0.3 to +2.9 dB, rate at or under the
    Default on 5 of 6 points.
  - H.264 `qhold` still misses: meva 388 worst-1% −0.9 dB, and re-codes rise slightly (+1 to +4 per 600).
  - H.264 `hold` everywhere: similar gains, but meva 388 loses 0.3 dB mean.
  - AV1: mixed, with chirla 187 −3.0 dB worst-1% and 1.059 of target for `qhold`. The 64x64 delta-q pools the
    hold over cells that are not held, and bursts leak through (section 4).
  - Neither arm makes the largest frame smaller, so neither passes the coordinator's gates as built.
- **(c) `refine` does not pass on either codec.**
  - H.264: rate 1.016-1.034, re-codes up, mean −0.94 to +0.19 dB.
  - A first H.264 version that let refinement frames refine unplanned blocks too did better (chirla 93 +0.14
    mean / +0.66 worst-1%, meva 194 +0.06 / +1.56). So the planned refinements are too narrow.
  - The loan is too weak where ordinary still frames undershoot their budget anyway, because a budget cut
    repays nothing there. The rate then ends over target.
  - Value per bit is right in principle (section 1), but the per-cell selection (known, clean cells only,
    measured one row in four) sees too little of the picture, and its high-rate cost estimate needs the learnt
    scale to swing 5-20×.
- **What this establishes.**
  - The quality the Default draws from bursts is real, and the periodic key wastes much of it. Holding the key
    at the held quality recovers it at a peak cost.
  - Refining only on purpose needs per-8x8 quantizers in every codec, and a refinement planner that sees every
    still cell (not only freshly measured ones).

## 6. Designs, ranked

Ranked by expected gain for this product (still scenes at low rate, CBR, 10 s keys). "Expected" is a
judgement from the measurements above.

1. **Hold still blocks on ordinary frames and plan every refinement (hold + (c)), for all codecs, with
   per-8x8 quantizers.** This attacks the root cause (section 4):
   - Every unplanned burst, re-code and long-window overshoot on still content comes from ordinary frames
     refining what the picture already holds.
   - The deliberate refinement then spends where the forward value is: early in a key period, at λ/(1+R),
     above the noise.
   - Status:
     - It works on H.264 (16x16 block QP).
     - It fails on AV1, because 64x64 delta-q pools a hold or refinement over moving and unknown cells, and
       bursts leak through.
     - AV1 (and AV2) need segmentation with per-8x8 ids (AV1 §5.9.14, §5.11.9), as libaom's cyclic refresh
       uses. VP9 has segmentation but codes one offset per 64x64 superblock today (`BLOCK_QP_LOG2` 6); an
       8x8 grid is a small change there. HEVC/VVC can use finer quant groups (`diff_cu_qp_delta_depth`).
   - Expected:
     - the Default's still-scene quality, or better, at controlled frame sizes;
     - re-codes and long-window overshoot on still content mostly gone;
     - CPU about +4% (block QP active on still frames: the cyclic-cadence measurement on H.264 was +3.5%).
   - Standards lens: at fixed QP the level cannot fall, so the hold changes little. A planned refinement at
     fixed QP is a per-block, value-ranked version of the cadence.
   - Experiment:
     - Build AV1 segmentation for hold and refinement, behind the same parameters.
     - Screen the hold alone (`rate.hold_still`), then hold plus refinement, on AV1 and H.264.
     - Pass gates: on the six still points, no 20 s window over target, fewer re-codes, smaller largest frame,
       mean and worst-1% PSNR-Y no worse than the Default. Then L1, L2 and L3 pool screens, and the
       600-frame final.
2. **Periodic key on still scenes planned at the held quality, with a size cap ((b), `rate.key_hold_still`
   = N buffers).**
   - Measured: +0.1 to +0.7 dB mean and up to +4.6 dB after the key, at a real cost in the worst 1 s window
     (section 5.1). The owner chooses the trade-off along the N curve.
   - Note: a from-scratch key at the held quality costs more than one buffer on a still scene (N = 1 changes
     nothing on chirla 93). Within 1 s of buffer the only way to keep the background across a periodic
     recovery point is not to re-code it from scratch:
     - recovery from an acknowledged long-term reference (`ltr.rs`, with feedback);
     - or an intra-refresh sweep (`refresh.period`) instead of periodic keys on still scenes.
     That is a product decision (recovery semantics), recorded here, not built.
   - Standards lens: none (the lens uses one key).
3. **A second, long-lived reference holding the last refinement (golden / long-term).**
   - The JVET LD-P structure keeps the last three GOP-end frames, and libvpx/libaom/SVT keep a golden or
     long reference.
   - Our `refs.second` (H.264 only) measured −1.4% J_real at +16% CPU on a deep cadence, mostly one class
     (fixed camera with moving people).
   - With planned refinements it has a natural target: the refinement frame becomes the long reference. Moving
     people uncover background that the reference still holds at the refined quality.
   - Expected: modest on our CBR tests, larger on the standards lens (its LD configs get exactly this). The
     CPU cost is the blocker; gate the second search to blocks next to motion.
   - Experiment: after design 1, a refinement frame refreshes the long-term slot. Measure meva and
     chad-type clips and the lens.
4. **(a) no refinement in the last N s before a scheduled key (`rate.prekey_quiet`).**
   - As "no base boost" alone it is mixed: the root cause still refines. As "hold still blocks in the
     window" it is the cheap part of design 1, and it is measured in section 5.1.

## 7. What to run first

1. **Tuner screens of what exists, each alone** (all registered, off):
   - `rate.key_hold_still` at 1.5 and 2 on AV1, H.264 and HEVC Default. Use the L1/L2/L3 pool levels and the
     600-frame final. Report the worst 1 s window beside J, because the owner sets N.
   - `rate.prekey_quiet=1` on H.264 Default.
   - The hold (`rate.hold_still`) on H.264.
   - These are the cheap, measured gains. (b) needs no new code.
2. **Build per-8x8 quantizers for the hold in AV1:**
   - Segmentation (AV1 §5.9.14 `segmentation_params`, §5.11.9 segment ids with spatial prediction, §7.12.2
     per-segment qindex; at most 7 offsets per frame, snapped as `nc-vp9/src/block_qp.rs` does).
   - Our AV1 decoder subset rejects segmentation today (`decoder/subset.rs`), so it needs decoding too.
   - Conformance against aomdec and dav1d through the landing check.
   - Then re-run `hold` / `qhold` on the six still points. Pass gates as in design 1.
   - Estimate: one to two agent-days with conformance.
3. **Then redo (c) as a whole-picture planner:**
   - It plans every still cell from its held step, not only cells measured clean this frame.
   - It costs the refinement from the held-to-price step ratio, learnt per codec.
   - It repays the loan out of the undershoot that still frames already leave, as debt rather than budget cuts.
   - Same six points, then L1.
