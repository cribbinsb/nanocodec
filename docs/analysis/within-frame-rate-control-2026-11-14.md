# Within-frame rate control: ours against the reference encoders (2026-11-14)

**Question (owner).** How do we distribute bits *inside* one frame, and do we steer a frame's size while
coding it? How do the standard real-time encoders do both, and what should we change: better quality from
the allocation, or better bit control?

**Scope.** Part A is *allocation*: how the quantizer varies between the blocks of one frame. Part B is
*control*: whether a frame's size is steered while it is coded, and what happens when it overshoots. Our side
was read from the code at `variance-integration` 0c8ff7ba. The references were read from their source
(openh264, libvpx, libaom, SVT-AV1, kvazaar, VVenC; all BSD-style). For x264 and x265 (GPL) only their
documentation and published descriptions were used; no code was read. Facts measured earlier are cited, not
re-derived.

## 1. Ours

### 1.1 Allocation: what per-block quantizer variation exists

The engine plans one quantizer per frame (`RateController::select`) and an optional per-block offset field
(`aq.rs`, `BlockAq`), in QP units (2^(1/6) step ratio), pooled per square of the codec's block-QP
granularity and converted to native units through each codec's step table (`aq::native_offset`). Three terms
add:

| Term | Parameter (engine default) | Signal and law | On in a shipped preset? |
|---|---|---|---|
| Variance | `rate.aq_strength` (0) | `strength · (log2 var − frame mean log2 var)` per 8x8 cell from the pre-analysis' `sum_sq_dev` (no pixel pass), symmetric, clamped to `±rate.aq_max_offset` (8) | No (every codec, every preset) |
| Persistence | `persist.strength` (0) | `strength · (log2 motion-error EWMA − frame mean)` | No |
| Reuse | `persist.reuse` (1) | Still cells whose measured reconstruction fell below their long-run quality get `−3·log2(1+R)` QP (λ/(1+R), R = still run to the next key) (`reuse.rs`) | **Yes, every codec and preset**; it fires only on a few cells of some frames |

How the offset reaches the block and λ:

- Per leaf, `Encoder::leaf_qp_offset` asks the field for the leaf's offset and the backend for what it can code
  (`CodecBackend::block_qp_offset`; VP9 snaps to one of 7 segment values). The decision then runs with
  `λ · (step_block / step_frame)²` (`block_lambda_ratio`): **λ already follows the block quantizer**, as in
  HM, kvazaar and VVenC.
- Rate control budgets the offsets *before* the frame: `effective_block_offset` folds the planned field into
  one equivalent offset through the rate model (bits ∝ step^−exponent), and the frame quantizer is chosen so
  the effective step meets the budget. After the frame the *realised* offsets replace it
  (`set_realised_block_offset`).
- A static skip is never taken on a block coded finer than the frame (it proves a zero residual only at the
  frame quantizer).
- A frame whose field is all zero skips block AQ entirely (`BlockAq::active`); the variance path, once on,
  costs ~5.7% CPU at any strength (H.264 Default study, 2026-11-13).

Measured spread (docs/analysis/steady-state-tools-2026-11-13.md): area-weighted SD of block QP within a frame
0.2 (ours) against 1–13 qindex for SVT-AV1 M8 `--rtc` (mean 6), 2–6 QP for x264 medium (mean 3.8), 1.1 QP for
x265 medium, and up to ±12–13 QP deltas on a quarter to a third of quant groups in the body-cam SoC's hardware
encoder (soc-encoder-tools.md). Two studies found variance AQ worth nothing on J as implemented (H.264 Default
best −0.39% J at +5.7% CPU; HEVC −1.14% in study, −0.09% [−1.42, +0.55] at 600 frames held out with a
worst-1% VMAF guard miss), trading −0.5…−0.9 dB PSNR for +0.1…+0.3 dB MS-SSIM in every window.

**The frame-level QP cadence does at frame granularity what other encoders do per block.** `qp_cadence.rs`
gives every 4th inter frame (8th during start-up) a finer quantizer (`rate.cadence_base_boost` 3, start-up
4–6), scaled by the still share. On still content the finer base frame refines everything its predecessors
left and the following frames inherit it by skipping. This is where the AV1 still-scene overshoot comes from
(audit addenda): base frame costs are bimodal (80% cost almost nothing, the rest burst to 1–9 cycle budgets),
the per-role model mispredicts the bursts 13–22× in AV1, and the level swings ~40 qindex within a cycle.

### 1.2 Control: what is decided before the frame, and what can change during it

- **Before the frame** (`select`): budget from the virtual 1 s buffer, debt correction (×0.25…×2), key loan and
  repay, cadence offset, quality cap, layer share; one model per frame role predicts bits from the
  pre-analysis complexity and the step; the inter level moves at most `level_step` per frame.
- **During the frame: nothing.** The superblock loop in `Encoder::code_picture` codes every superblock at the
  planned frame quantizer plus the planned field. No backend reports its bit position mid-frame, and no offset
  depends on what the frame has spent so far.
- **After the frame:** `recode_quantizer`. If the frame would fill the strict-CBR bucket (drained at the
  target) past `rate.recode_overflow` × capacity (1.4; AV2 inter 1.1), the whole frame is coded again, coarser,
  at most twice (`MAX_RECODES`), the first re-code assuming a bits-vs-step slope of 3. Every backend supports
  it (`CodecBackend::RECODE`). On the audit's six still points the Default re-codes 7–26 of 600 frames. A
  re-coded frame is coded twice: a deadline risk on the Jetson. Frame dropping exists but is off
  (`rate.frame_drop`).
- Probes of bounding bursts after the fact (`rate.burst_cycles`, debt-scaled base boost) brought every
  point inside 2.5% but starved the frames right after a key: the bound scaled with the *corrected* budget, and
  the stream enters the key at the correction floor (audit addendum).

### 1.3 What each codec's syntax allows that we use or not

| Codec | Block-QP syntax | Granularity we use | Smallest the syntax allows | Cost of a zero delta | Notes |
|---|---|---|---|---|---|
| H.264 | `mb_qp_delta` (§7.3.5, §7.4.5), −26…+25, predicted from previous MB | 16x16 MB | 16x16 MB | 1 bit (CAVLC `se(v)`) / one CABAC bin, only in MBs with residual or I_16x16 | Skipped MBs keep the predicted QP: a change takes effect at the next coded MB |
| HEVC | `cu_qp_delta_abs/sign` (§7.3.8.14), PPS `cu_qp_delta_enabled_flag` | 64x64 CTB quant group (`diff_cu_qp_delta_depth` 0) | 8x8 (depth 3 at 64 CTB) | one CABAC bin per quant group with a coded TU | Enabled stream-wide whenever block AQ exists, i.e. in every Default (reuse term) |
| VVC | `cu_qp_delta` (§7.3.11.10) | 64x64 | down to 8x8 | one bin per quant group with a coded TU | Single-layer streams only (layered SPS path lacks it) |
| AV1 | `delta_q_present`, `delta_q_res`, `delta_qindex` at the first block of each SB (§5.9.17, §5.11.12) | 64x64 SB, `delta_q_res` 0 (1 qindex) | 64x64 SB (128 with 128 SBs); `delta_q_res` 0–3 | one adapted symbol per SB not wholly skipped | Present only on frames with offsets (`FramePlan::block_qp`); decided before the frame |
| AV2 | as AV1 (AVM `read_delta_qindex`) | 64x64 SB | SB | as AV1 | Deblocking reads each block's own qindex |
| VP9 | segmentation `SEG_LVL_ALT_Q` (§6.2.11/§7.2.10), 8 segments, per-block ids | 64x64, ≤7 distinct offsets per frame | 8x8 ids | tree-coded segment id per block; temporal id prediction unused | `nc-vp9/src/block_qp.rs` |

Unused: finer quant groups in HEVC/VVC, `delta_q_res` > 0 in AV1/AV2 (coarser, cheaper deltas), VP9 temporal
segment-id prediction (the cheap path for a stable cyclic-refresh map), and above all **any mid-frame use of
the syntax**: every codec can change the quantizer at the next superblock (VP9 only among 8 segment values
fixed in the header).

## 2. The references

Read from source (BSD-style licences) at their current heads: openh264 (`codec/encoder/core/src/ratectl.cpp`),
libvpx (`vp9/encoder/vp9_aq_cyclicrefresh.c`, `vp9_ratectrl.c`, `vp9_speed_features.c`), libaom
(`av1/encoder/aq_cyclicrefresh.c`, `ratectrl.c`, `speed_features.c`, `av1_cx_iface.c`), SVT-AV1
(`Source/Lib/Codec/rc_rtc_cbr.c`, `rc_aq.c`, `enc_handle.c`; local checkout e3d6340), kvazaar
(`src/rate_control.c`, `encoderstate.c`), VVenC (`BitAllocation.cpp`, `RateCtrl.cpp`). HM's `TEncRateCtrl`
(JCTVC-K0103) is described from the proposal and checked against kvazaar's port of it. x264 and x265 are
described from their documentation and developer mailing-list posts only.

**Headline: the real-time AV1/VP9 encoders do not steer a frame mid-way; the H.26x encoders that do so use small
row steps.** libvpx, libaom and SVT-AV1 fix every block's quantizer before the frame and handle overshoot after
it: SVT with one re-code, the libs with a scene-change maximum-Q rule and optional drops. Row- or CTU-level feedback exists in
openh264 (on by default), kvazaar's opt-in OBA mode, HM, and x264/x265 VBV. **Their per-block allocation in
real time is cyclic refresh, not variance AQ:** they boost a rotating subset of static blocks.

### 2.1 Allocation

- **libvpx VP9 (aq-mode 3, the WebRTC default).** Up to 3 segments. Each frame marks about 10% of the 8x8 blocks
  (5% when few qualify), walking superblocks from where the last frame stopped. A block qualifies if it has not been
  refreshed recently, and was coded coarser than the boost quantizer or has had a zero vector for under 100 frames.
  The boost delta is the qindex at which the rate model gives `rate_ratio_qdelta` times the bits: 3.0 for the
  first 4 cycles after a key, then 2.0. The delta is capped at 60% of q. During mode decision, a block that moves or is intra goes back to the base segment, and a large
  still block with low rate goes to a second, stronger segment. The base Q is chosen from a bits model that
  mixes the base and boost quantizers by the boosted fraction. λ for boosted blocks comes from the boosted Q. It is
  off on keys, in temporal layers above 0, at very fine or very coarse average Q, under low-motion statistics,
  and on scene changes. Cost: negligible (mode-decision statistics).
- **libaom real time.** The default RT config is cyclic refresh with delta-q off (`av1_cx_iface.c`). It works as libvpx,
  plus a source-SAD gate per superblock. The refresh fraction is 10 ± 5% and the ratio 2.25–3.0. Both adapt
  to the last frame's rate miss: a correction factor above 1.25 lowers both. A variance-boost SB delta-q exists
  in the nonrd path but is off by default.
- **SVT-AV1 `--rtc`.** Cyclic refresh signalled as per-64x64 `delta_q`, not segmentation. It runs on CBR base-layer inter
  frames when average Q is mid-range and low motion is common. The band is 20% of the superblocks and moves
  through the frame. A superblock qualifies when its motion-estimation (ME) vector is within ±1 and its distortion is under twice the
  average. Below-average-distortion superblocks get a second, stronger tier. The ratio is `2.7 + avgQ/170`, capped at 4 (8 for
  the second tier). Each tier's delta comes from the R–Q model fed with that tier's own ME distortion, and the base Q
  from a tier-weighted frame-size estimate. Variance boost is forced off in CBR, and the ME-based λ modulation
  is off at M11/M12. λ follows the superblock qindex. **The 7–9 qindex block spread we measured for SVT M8 is
  this cyclic-refresh band**, not perceptual AQ.
- **openh264.** A variance AQ exists but is hard-disabled (`encoder_ext.cpp`: "turn off adaptive quant now").
  That explains the byte-identical streams the audit saw with AQ on.
- **kvazaar.** VAQ `0.1·vaq·(ln var_CTU − ln var_frame)` per CTU, off by default. λ is recomputed from the CTU QP.
- **VVenC.** Perceptual QPA, on in its presets: `QP_CTU = sliceQP + round(3·log2(activity · norm))`, where activity
  is spatial plus temporal high-pass over the source and two previous source frames. Sub-CTU quant groups. `λ_CTU =
  λ_slice · 2^(ΔQP/3)`. Decided in pre-analysis.
- **x264 / x265 (documentation).** AQ mode 1 is `strength · (log2 energy − c)`, modes 2–3 auto-variance (mode 3 with a dark
  bias), x265 mode 4 adds edges. AQ is on by default, runs in the lookahead, and λ follows the block QP.

### 2.2 Control

- **openh264 GOM control (on for P frames, single slice).** Groups of 2–4 MB rows. Each group's target is the
  slice bits left times its share of the remaining pre-analysis SAD. At each group boundary the QP moves +1/+2 or −1 from the ratio of
  bits left to bits left after the group's miss. The QP is clamped to the frame QP ± 3. The next frame starts from the average
  MB QP actually used. There is no re-code, and frames are skipped before coding on buffer state.
- **kvazaar OBA (opt-in) and HM K0103.** After each CTU, the actual CABAC bits feed a target for the next few CTUs:
  the picture target minus the bits so far, shared by weight. λ is solved per CTU (R-λ), with QP within ±1 of the previous CTU and
  ±2 of the picture. No re-code. kvazaar's default λ mode has no in-frame feedback.
- **x264 / x265 VBV (documentation).** Per-row bit predictors at candidate QPs and per-row QP step limits. x265
  re-encodes a row on a large QP jump or buffer underflow.
- **libvpx / libaom real time.** No re-code in one pass. A scene-change frame (high source SAD) is coded at or near the
  maximum Q, decided before encoding or (libvpx at ≤CIF) by one re-encode when it is over 8× the average frame.
  They also drop frames on buffer state, and optionally drop after encoding (libaom: more than 8× average and the buffer under a quarter
  of optimal). No max frame size is guaranteed in real time.
- **SVT-AV1 `--rtc`.** One re-code when the *mode-decision-estimated* frame size would overflow the buffer or a
  max-bitrate share. The new Q comes from the R–Q model corrected by actual/estimated, at least base + 4, with delta-q off
  in the re-code.
- **VVenC.** Frame/GOP-level rate control only (two-pass or lookahead).

### 2.3 Comparison

| Encoder | Per-block signal and law | On in real time? | λ coupling | Cost | In-frame feedback | Overshoot |
|---|---|---|---|---|---|---|
| **ours** | variance / persistence (off); reuse term on still cells whose quality decayed (on, sparse); frame-level QP cadence | reuse only | λ ∝ block step² | variance path ~5.7% CPU | **none** | whole-frame re-code past 1.4 buffers (≤2) |
| openh264 | variance AQ (disabled) | no AQ; GOM on | λ table[MB QP] | ~0 | **yes**: per 2–4 MB rows, ±1/2 QP, frame QP ± 3 | pre-encode skip |
| libvpx VP9 rt | cyclic refresh, ~10% static blocks, rate-ratio delta 2–3× | yes (aq 3) | segment rdmult | ~0 | none | scene-change max-Q; drop |
| libaom rt | cyclic refresh 10 ± 5%, ratio 2.25–3, adapted to rate miss | yes | from boosted q | ~0 | none | scene-change max-Q; optional post-drop |
| SVT-AV1 rtc | cyclic refresh as SB delta-q, 20% band, two tiers from ME distortion | yes (CBR base layer) | from SB qindex | ~0 | none | one re-code on estimated size |
| kvazaar | VAQ (off); CTU R-λ | off; λ RC with `--bitrate` | λ-first | one variance pass | OBA only (opt-in) | none |
| VVenC | QPA spatio-temporal activity, 3·log2 | yes | 2^(ΔQP/3) | activity filter | none | GOP capper |
| HM K0103 | — | — | λ-first | ~0 | yes, LCU ±1/±2 QP | none |
| x264 / x265 (docs) | AQ 1–4 (variance, auto, dark, edge) | yes | from block QP | lookahead | yes, row VBV | x265 row re-encode |

### 2.4 What this says about us

- **Our QP cadence is a frame-granular cyclic refresh.** Cyclic refresh boosts ~10–20% of the still blocks every
  frame. Our cadence boosts the still picture every 4th frame (8th during start-up), so the boost arrives as a
  burst that a per-role size model cannot predict. The references' rate models also count the boosted segment's bits
  when choosing the base Q (libvpx's mixed bits-per-MB, SVT's per-tier ME distortion); ours folds planned block
  offsets in the same way (`effective_block_offset`), so that part carries over.
- **Mid-frame control is uncommon but proven in the H.26x encoders, always with small steps** (±1–2 QP per row
  group or CTU, ±2–3 of the frame QP). None of them guarantees a max frame size; the AV1/VP9 real-time encoders
  rely on a re-code (SVT) or on a scene-change maximum Q.
- **No real-time reference runs a variance AQ by default except x264/x265.** That fits our measured result that
  variance AQ does not pay on J.
- **λ coupling is the same everywhere** (λ follows the block quantizer), and ours already does it.


## 3. What we should do

Ranked by expected value for this product (low-rate live camera, one core in real time, the target is a
ceiling, worst frames matter) against cost. "Expected" is a judgement from the evidence above, not a
measurement; the prototypes' measurements follow in section 4.

| # | Change | What it fixes | Expected effect | Cost | Risk |
|---|---|---|---|---|---|
| 1 | **In-frame (superblock-row) bit control** with bounded steps: after each superblock row, project the frame's size from the bits spent and the pre-analysis weight still to code; coarsen the remaining rows (never finer than the plan) when the projection passes a limit. Hard limit: the room the re-code would enforce (`strict_room`), so a whole-frame re-code becomes the rare backstop. Optional soft limit: a multiple of the *uncorrected* frame budget (target/fps), aimed at the 12–35-budget refinement bursts | Re-codes (7–26 per 600 on still points, each a second coding of a frame: Jetson deadline risk); the bursts behind AV1's 3–5% long-window overshoot; a basis for a max-frame-size guarantee | Bit control: re-codes → few; burst size bounded without a debt-scaled cut. Quality: neutral by construction at the hard limit (it acts where a re-code acts today, but keeps the top rows at the planned quantizer instead of coarsening the whole frame) | One bit-position query per superblock row and a few flops; a `coded_bits` query per backend (exact for bit writers and CABAC, a cost sum for deferred range coders). AV1/AV2 frames must carry `delta_q_present` while armed | A frame coarsened mid-way is spatially uneven (top finer than bottom); the soft limit can starve post-key frames if it scaled with the corrected budget, which is why it is tied to the uncorrected budget |
| 2 | **Cyclic, spatial cadence on still regions**: give a rotating 1/period of the still superblocks the base boost every frame instead of the whole frame every period: what every real-time AV1/VP9 reference does (cyclic refresh, 10–20% of static blocks per frame, delta from a rate ratio of 2–4, base Q from a segment-weighted bits model) | The bimodal base-frame cost the size model cannot represent (the root cause the audit names) | Smooth frame sizes on still content; the same refinement per cycle; possibly better worst frames | Block QP on every still frame (the active-AQ path costs ~4% instructions on `pan_jaad`); AV1 delta-q symbols per superblock; VP9 one segment | Medium: a quality bet the cadence's own +0.4 dB must survive; overlaps the predictor fix in worktree `still-overshoot` |
| 3 | **A new variance law** (x264 AQ-mode-2-style normalised energy, one-sided boost as SVT's variance boost) | MS-SSIM third, the long-window loss | Unclear: two studies and the 2026-11-11 re-screen found every variance or temporal AQ term worse or flat on J (PSNR loss outweighs MS-SSIM) | ~5.7% CPU for the active path unless pooled cheaper | High: repeats a measured negative |
| 4 | **Max frame size for transport** (`max_frame_bits` user cap) | Low-latency transports that must bound each packet | A guarantee instead of a re-code ceiling | Free once #1 exists (its hard limit, then a last-resort maximum quantizer on the remaining rows) | Low |
| 5 | Finer quant groups (HEVC/VVC depth > 0), `delta_q_res` > 0 (AV1), VP9 temporal segment ids | Signalling cost of block QP | Small, only once a per-block allocation is on | Low | Low |
| — | λ coupling to block QP | — | Nothing to do: λ already follows the block step squared (`block_lambda_ratio`), as in HM, kvazaar and VVenC | — | — |

**Prototyped:** #1 (in-frame control), in the shared engine, registered and off by default, for the codecs whose
backends report a bit position. #2 next if #1 leaves the bursts in place.

## 4. Prototype measurements

### 4.1 What was built (`inframe.*`, registered, off)

`nc-encoder/src/inframe.rs`. After each superblock row the encoder asks the backend for the frame's bits so far
(`CodecBackend::coded_bits`). It projects the rest of the frame from the bits per unit of weight spent so far,
where a row's weight is block AQ's expected bit weight from the pre-analysis (no new pass). When the
projection passes the frame's limit, the remaining rows are coarsened. The rules:

- **Bounded steps.** Rows only get coarser than the plan, by at most `inframe.row_step` (3) QP per row and
  `inframe.max_offset` (12) in all.
- **Engine plumbing.** The offset travels through the engine's block-QP path, so λ follows the coded step.
- **Rate model.** The coarsened rows are reported to rate control as a realised offset (`set_inframe_offset`),
  so the size model learns the frame's real step.

Two limits, either or both:

- **`inframe.room`** is a share of the room the whole-frame re-code would enforce: a re-code replacement.
- **`inframe.budgets`** is a multiple of the *uncorrected* frame budget (target / fps). It is a burst bound
  that does not shrink with the debt, unlike the `rate.burst_cycles` probe.

Keys are left to the whole-frame re-code unless `inframe.keys` is 1.

**Backends.** H.264 reports exact CAVLC bits and CABAC bits including the carry-pending bytes. AV1 reports range-coder
bytes, or for deferred tiles the information content of the recorded intervals, priced once per query.

**AV1 cost.** AV1 frames carry `delta_q_present` while armed, so an armed AV1 stream changes even where no row is
coarsened.

**Checks.**
- With both limits off, every shipped preset of every codec is byte-identical to the integration branch: 52 of
  52 streams, two 720p clips, CBR 600 kbps.
- With in-frame control forced on (`budgets=4`, `room=0.9`, `keys=1`), the landing conformance check passes for
  H.264 and AV1: all 225 streams are bit-exact in JM, ffmpeg, aomdec, dav1d and ours, covering every preset, layers,
  intra refresh and starved keys.

**Not wired.** HEVC (whose slice is written at `end_frame`), VP9 (whose symbols are recorded for the frame's
probabilities), VVC and AV2 report nothing, so in-frame control stays off for them.

### 4.2 Bit control: the audit's six still points

Setup: 600 frames from frame 0, a key every 10 s, our reconstruction, build `feature/inframe-rc` (in-frame control off is byte-identical to the integration branch).
Script: `bitctl.py`, which runs the shared point engine with a `--rate-log` encode per point.

Column notes:

- **U** is one uncorrected frame budget (target/fps). Inter frame sizes are given in U as p95 / max.
- **Re-coded** counts frames re-coded per 600.
- **Worst-1%** is the worst-1% PSNR-Y.
- **Rate** is achieved / target over 600 frames.

AV1 Default:

| Point | Default: rate / max U / re-coded / worst-1% / PSNR-YUV | `budgets=8` (keys exempt) | `room=0.9` (keys exempt) | `room=0.9` (keys steered too) |
|---|---|---|---|---|
| chirla 47 | 1.001 / 40.3 / 22 / 28.81 / 40.11 | 1.011 / 18.8 / 11 / 29.10 / 39.44 | 1.018 / 33.9 / 13 / 29.05 / 39.98 | 1.008 / 34.2 / 14 / **27.31** / 39.59 |
| chirla 93 | 1.037 / 38.6 / 8 / 35.40 / 43.57 | 1.008 / 19.4 / 6 / 35.08 / 43.51 | 0.995 / 36.2 / 3 / 35.51 / 43.55 | 1.006 / 36.2 / 3 / 35.51 / 43.56 |
| chirla 187 | 1.030 / 35.6 / 26 / 38.97 / 45.76 | 1.026 / 22.5 / 1 / 38.72 / 45.72 | 1.051 / 30.3 / 3 / 39.32 / 45.97 | 1.042 / 30.3 / 6 / 39.32 / 45.80 |
| meva 97 | 0.984 / 30.2 / 16 / 29.14 / 38.16 | 1.014 / 14.8 / 5 / 29.27 / 37.77 | 1.009 / 30.7 / 0 / **26.98** / 37.54 | 1.011 / 30.3 / 0 / **26.96** / 37.66 |
| meva 194 | 1.033 / 37.1 / 8 / 33.06 / 40.94 | 0.998 / 9.4 / 1 / 32.77 / 40.45 | 1.041 / 32.2 / 0 / 32.36 / 40.80 | 1.014 / 31.5 / 0 / 32.45 / 40.76 |
| meva 388 | 1.028 / 36.6 / 7 / 33.91 / 42.73 | 1.004 / 9.1 / 0 / 36.32 / 42.58 | 1.016 / 31.8 / 0 / 35.64 / 42.64 | 1.039 / 29.8 / 0 / 35.61 / 42.76 |

H.264 Default:

| Point | Default: rate / max U / re-coded / worst-1% / PSNR-YUV | `budgets=8` | `room=0.9` (keys exempt) |
|---|---|---|---|
| chirla 47 | 0.987 / 30.0 / 12 / 30.88 / 40.33 | 0.976 / 13.9 / 4 / 30.55 / 39.98 | 1.003 / 28.5 / 4 / **25.42** / 39.31 |
| chirla 93 | 0.987 / 31.8 / 4 / 33.69 / 43.44 | 0.977 / 13.0 / 2 / 34.08 / 43.37 | 0.998 / 36.9 / 0 / 33.20 / 43.19 |
| chirla 187 | 0.997 / 32.4 / 2 / 38.43 / 46.49 | 0.980 / 12.9 / 0 / 37.77 / 45.98 | 1.006 / 31.9 / 1 / 38.38 / 46.61 |
| meva 97 | 0.962 / 27.7 / 16 / 26.97 / 37.04 | 0.977 / 8.3 / 1 / 28.54 / 36.80 | 1.021 / 35.4 / 4 / **24.08** / 36.75 |
| meva 194 | 1.006 / 37.2 / 6 / 29.73 / 40.42 | 0.994 / 8.2 / 1 / 31.28 / 40.08 | 1.020 / 33.5 / 0 / 31.59 / 40.72 |
| meva 388 | 1.027 / 33.2 / 3 / 35.37 / 43.03 | 0.997 / 7.9 / 0 / 34.98 / 42.51 | 1.004 / 31.8 / 0 / 34.60 / 42.81 |

Other limits tried: `budgets=12` (AV1 rate 0.999–1.055, max 12–25 U); `budgets=16`; `room=0.9` with
`budgets=8`, which matches `budgets=8` within noise.

What the tables show:

- **`budgets=8` does the bit-control job.** It bounds every frame's size: the largest inter frame falls from 28–40 U to 8–23 U, with
  AV1's misses past 8 U coming from rows coded before the projection caught up. Re-codes fall from 87 to 24 per
  3,600 frames in AV1 and from 43 to 8 in H.264. Every point lands at 0.976–1.026 of target. AV1's four points over 2.5% (+2.8% to +3.7%)
  end at −0.2% to +2.6%. Worst-1% PSNR-Y moves −0.7 to +2.4 dB. The large drops of the
  after-the-fact bounds (−1.8 to −2.9 dB on meva 194, audit addenda) do not appear.
- **It costs mean quality on still content, and the lower rate explains little of it.** PSNR-YUV is −0.05 to −0.67 dB, against rate
  changes of −3.4% to +3.0%. At matched rate (the Default's three ladder points interpolated in log-rate to the
  candidate's achieved rate) the loss is:

  | | chirla 47 / 93 / 187 | meva 97 / 194 / 388 |
  |---|---|---|
  | AV1 PSNR-YUV | −0.72 / +0.08 / −0.03 | −0.51 / −0.36 / −0.09 |
  | AV1 VMAF v1 | −0.44 / +0.58 / +0.62 | −1.56 / −0.53 / +0.25 |
  | H.264 PSNR-YUV | −0.30 / −0.03 / −0.43 | −0.31 / −0.28 / −0.41 |
  | H.264 VMAF v1 | +0.23 / +0.40 / +0.17 | −0.22 / +0.32 / −0.11 |

  MS-SSIM at matched rate moves −0.0033 to +0.0005. Still-scene refinement bursts are efficient: they buy quality
  that every later frame inherits. A bound gives part of that up, and a row bound also refines the top of the
  picture first.
- **`room` is not a re-code replacement.** With keys steered, or with only inter frames, it removes most
  re-codes but drops worst-1% PSNR-Y by 1.5–5.5 dB on chirla 47 and meva 97. The frames that reach the re-code room are the
  ones after a key. A re-code coarsens them evenly; row steering leaves their lower rows up to 12 QP coarser.
  On static content those rows stay coarse until refined again.

### 4.3 Quality: L1 screen (rotation 2026-11-14 quick draw, 24 clips, 150 frames, `budgets=8` vs off)

| Codec | J (90% CI) | Clips changed | Largest per-clip effects |
|---|---|---|---|
| AV1 Default | −1.18% [−2.38, +0.52] | 24 of 24 (armed frames carry `delta_q_present`) | `otw_portrait_crowd_1080p` −22.4%, `drone_rounda` −5.7%; static `bdd_00dc5030` +8.8%, `long_static_chirla` +1.6% |
| H.264 Default | −0.38% [−1.50, +0.63] | 10 of 24 | `otw_portrait_crowd_1080p` −16.7%; `aboda_video11` +5.1%, static `bdd_00dc5030` +3.8%, `pan_jaad_0101` +1.8% |

Both intervals include 0. The mean rests on one clip: without `otw_portrait_crowd_1080p` both are about 0 or
slightly worse. Static clips lose, as in 4.2.

### 4.4 Re-code census (owner question: how often do we re-encode?)

Each point was encoded once with `--rate-log`; the new `codings` and `encode_us` columns give each frame's
codings and wall time. Streams were deleted. Build `feature/inframe-rc` with in-frame control off, which is
byte-identical to the integration branch.

Column notes:

- **base** means a cadence base frame (a negative cadence offset).
- **Time** is a re-coded inter frame's wall time over its stream's median single-coded inter frame. It was
  measured on a loaded machine, so read it as a ratio within each stream.
- **Most codings** is the most times any one frame was coded.

**Day's L1 draw** (24 clips, every rung, 150 frames, one key; 93 streams, ~13,870 frames per config):

| Config | Re-coded per 1000: all / key / base / other inter | Streams with none | Most codings | Time: median, max |
|---|---|---:|---:|---|
| AV1 Default / Fast | 1.4 / 0 / 8.6 / 0.0; 0.9 / 0 / 5.0 / 0.1 | 86% / 90% | 3 | 10.3x, 15.9x; 7.8x, 10.9x |
| HEVC Default / Fast | 1.7 / 0 / 10.4 / 0.0; 0.6 / 0 / 4.1 / 0.0 | 90% / 92% | 3 | 8.4x, 17.1x; 5.4x, 11.0x |
| H.264 Default | 1.1 / 0 / 12.1 / 0.1 | 89% | 3 | 6.3x, 16.1x |
| H.264 Fast / Faster / Fastest | 0.5 / 0.6 / 1.2 (nearly all base frames) | 95% / 94% / 91% | 3 | 3.0–3.2x, max 4.8x |
| VP9 Default / Fast | 1.1 / 0.5 | 89% / 94% | 3 | 11.2x (max 21.2x) / 7.0x |
| VVC Default / Fast | 0.9 / 0.7 | 90% / 92% | 3 | 8.0x / 5.8x |
| AV2 Default / Fast | 2.4 / 1.3 | 85% / 90% | 3 | 9.3x (max 21.6x) / 5.3x |
| H.264 Default, `inframe.budgets=8` | 0.0 | 100% | 1 | — |
| AV1 Default, `inframe.budgets=8` | 0.1 | 99% | 3 | 7.6x |

By class, almost all re-codes are on drone clips (5–22 per 1000, 4 streams) and static clips (0.4–6.2). Crowd
has 0–2.7, pan 0–0.8, body-cam 0–1, and lowlight and texture 0. Over the 14 configs without in-frame control,
163 frames were coded twice and 43 three times.

**Long window** (the draw's 30 clips with ≥600 frames, every rung, 600 frames, a key every 10 s; 114 streams,
~68,200 frames per config):

| Config | Re-coded per 1000: all / key / base / other inter | Streams with none | Most codings | Time: median, max |
|---|---|---:|---:|---|
| AV1 Default | 4.2 / 44 / 16.2 / 1.0 | 68% | 3 | 10.5x, 39x |
| HEVC Default | 4.0 / 40 / 17.0 / 0.6 | 70% | 3 | 8.9x, 23x |
| H.264 Default | 3.3 / 32 / 25.7 / 0.7 | 68% | 3 | 6.5x, 21x |
| AV1 Default, `inframe.budgets=8` | 1.0 / 32 / 0.6 / 1.0 | 79% | 3 | 4.2x, 13x |
| H.264 Default, `inframe.budgets=8` | 0.8 / 40 / 0.3 / 0.7 | 80% | 3 | 2.8x, 14x |

By class, static clips have 7.8–9.9 per 1000 (1.9–2.4 with `budgets=8`), pan ≤0.1, and crowd and lowlight 0.

What the census shows:

- **No first key was re-coded** (0 of 114 in every config).
- **Periodic keys were re-coded 8–11 times out of 137 (6–8%).** In-frame control leaves keys alone by default,
  so this is the same with it on.
- **A frame is coded at most three times** (`MAX_RECODES` 2).
- **What would make inter re-codes unnecessary.** In-frame control with a hard guarantee: a last resort of the
  maximum offset, then zero residual, in the remaining rows. That bounds a frame at the room plus one superblock
  row. The `room` probe (4.2) shows the cost: steering at the hard limit hurts the frames after a key more than a
  uniform re-code does. So the guarantee should stay a rare backstop behind a soft bound and a better base-frame
  predictor.
- **Periodic keys.** Either keep strict-buffer room before the key, which bounded bursts now make feasible, or
  size the key from an intra pre-estimate.

### 4.5 Verdict and hand-off

- **`inframe.budgets` is a working bit-control lever.** It bounds frame size in budget units and removes nearly
  every base-frame re-code (4.4), with no worst-frame collapse. But it costs real mean quality on still scenes:
  −0.03 to −0.72 dB PSNR-YUV at matched rate. On the L1 draw it is J-neutral within noise. It is landed off by
  default for deployments that value steady frame sizes and few re-codes: low-latency transport, real-time
  deadlines. It is **not handed to the tuner for the Defaults**, because it fails the still-scene quality gate.
- **`inframe.room` is rejected as a re-code replacement.** Whole-frame re-codes beat row steering on the frames
  after a key. It stays registered because `budgets` uses the same machinery and the hard limit is the basis of a max-frame-size
  option, but it is not recommended.
- **Next, if pursued:**
  - Arm AV1's `delta_q` only on frames that can burst: cadence base frames, or frames finer than their reference.
  - Wire HEVC. Its slice is written at `end_frame` from recorded CUs, so it needs either a running estimate (the
    CABAC bin costs the decisions already price, summed per coded CU into a counter `coded_bits` reads) or
    per-CTU writing. The estimate is cheap: one add per CU.
  - Most promising for quality: recommendation #2, the cyclic spatial cadence. It attacks the bimodal base-frame cost at
    its cause instead of bounding it.

## 5. Cyclic spatial cadence (`rate.cadence_spatial`, registered, off)

Recommendation #2, built in `nc-encoder/src/cyclic_cadence.rs`. In steady state (after the start-up cadence that
follows every key), the cadence's base boost goes to a rotating band of quantizer squares instead of to whole
base frames. The band is `1 / cadence_period` of the candidate squares, taken in raster order from where the last
frame's band stopped.

- **Candidates:** squares whose every 8x8 cell has been static for at least 2 frames, and whose last coded step is
  coarser than the band's step (libvpx's "coded coarser than the boost" rule).
- **Keys:** every key, periodic or first, resets each square to the key's step.
- **Rate control:** it sees a flat steady-state cadence, and budgets the band as planned block offsets.
- **Mode decision:** band squares take the zero vector with no candidate, sub-pel or intra search. They are
  static and re-coded only to refine them, as in the references' cyclic refresh.

`inframe.startup_only` limits the in-frame soft limit to the start-up cadence; it is used by variant v3 below.

**Variants**

- **v1:** the band from the first inter frame after a key, candidates from the still map.
- **v2:** v1 plus the start-up frame cadence kept, plus short-run static candidates.
- **v3:** v2 plus `inframe.budgets=8` during start-up only.

The figures below are for v2 with the zero-vector path. Setup: the six still points, 600 frames, a key every
10 s. Each column is against the Default of the same build; "matched" is at matched rate against the Default's
curve, and f300-360 is the window after the second key.

| Point | AV1: rate; re-codes; worst-1% Y; matched PSNR-YUV / VMAF; f300-360 | H.264: same |
|---|---|---|
| chirla 47 | 1.001→0.998; 22→12; +0.49; −0.18 / −0.03; +0.15 | 0.987→0.992; 12→5; +0.05; −0.02 / +0.43; +0.08 |
| chirla 93 | 1.037→1.015; 8→6; −0.52; −0.12 / +0.10; +0.04 | 0.987→0.995; 4→4; +0.77; +0.05 / +0.24; +1.06 |
| chirla 187 | 1.015→0.995; 23→10; +0.60; +0.05 / −0.14; +1.31 | 0.997→0.985; 2→2; 0.00; −0.25 / +0.05; +0.46 |
| meva 97 | 0.984→1.025; 16→7; +0.32; −0.25 / −0.31; −0.29 | 0.962→1.002; 16→10; +1.63; +0.02 / +0.78; +0.18 |
| meva 194 | 1.033→0.994; 8→9; +0.01; +0.12 / +0.52; +0.58 | 1.006→0.992; 6→5; +1.82; +0.25 / +1.04; +2.08 |
| meva 388 | 1.028→0.994; 7→2; +2.94; +0.15 / +0.49; +2.12 | 1.027→0.995; 3→1; +0.13; 0.00 / +0.21; +0.07 |

Frames 0-60 are identical to the Default, because the start-up cadence is unchanged.

**How the earlier variants did**

- **v1** removed the bursts: largest frame 10-25 U, re-codes 87→26 (AV1). But it lost frames 0-60 at all 12
  points (−0.2 to −1.4 dB), and lost 0.8-0.9 dB PSNR-YUV at matched rate on the lowest rungs.
- **v3** had the strongest bit control: largest frame 8-25 U, re-codes 87→23 (AV1) and 43→10 (H.264). But it lost
  frames 0-60 by 0.6-1.5 dB everywhere: the start-up bursts are worth their bits.
- **v2 without the zero-vector path** lost 0.97 dB worst-1% on H.264 chirla 47.

**L1 screen** (v2 before the zero-vector path; today's quick draw, 150 frames): AV1 J +0.11% [−0.25, +0.81],
H.264 −0.72% [−1.03, +0.41]. With `cadence_startup_frames` at 100 the band is active for only about 50 of the 150
frames, so the 600-frame long window is its real test.

**Long-window rate** (AV1, the draw's 30 clips with ≥600 frames, 114 streams, Default → band):

| | Default | Band |
|---|---:|---:|
| Median achieved / target | 0.989 | 0.991 |
| Streams more than 2.5% over target | 26 | 23 |
| Re-codes | 276 | 178 |

The largest rises lift undershooting static streams to about target.

**CPU** (x86 instructions; static_cam22, 250 coded frames, 150 kbps, one key):

- **H.264:** +3.5%. Without the zero-vector path it was +5.6%: band squares cannot take the static-skip shortcut.
- **AV1:** +15.6%. On this clip the band changed adaptive resolution's trajectory: 100 frames at 720p instead of
  91, and a late switch up left the 10 s window at about 160 kbps on a 150 kbps target.

**Noise skip.** Band squares stay open to the noise skip (`decision.noise_skip`), and to that test alone. A square
whose zero-vector residual is within the camera's noise is skipped and marked settled, so with the noise skip on
the band never refines noise. Not yet measured with both on; that waits for the skip-aware budget.

### 5.1 Follow-up: noise-robust candidates and the noise skip (2026-11-14)

**Candidates.** Candidates now come from each cell's zero vector predicting it about as well as its best match
(`noise_skip::zero_explains`) for 2 frames in a row, instead of pre-analysis' static flag. On a noisy still that
flag reads mid-tones as changed, so the band found few squares and lost the frame cadence's steady-state
refinement.

**Noise skip.** A band square whose zero-vector residual is within the camera's noise (the noise map's test,
while `decision.noise_skip` is on) is coded at the frame quantizer and settled.

**The six still points with the new candidates** (same setup as section 5; vs the Default of the same build):

| Point | AV1: rate; re-codes; worst-1% Y; matched PSNR-YUV / VMAF; f300-360 | H.264: same |
|---|---|---|
| chirla 47 | 1.001→0.995; 22→15; +1.62; +0.07 / +0.68; +0.75 | 0.987→0.996; 12→6; +0.05; −0.03 / +0.47; −0.03 |
| chirla 93 | 1.037→1.004; 8→5; +0.20; +0.40 / +0.56; +1.78 | 0.987→0.995; 4→2; +0.77; +0.02 / +0.27; +1.26 |
| chirla 187 | 1.015→1.014; 23→12; +0.60; +0.04 / +0.22; +1.60 | 0.997→0.989; 2→2; 0.00; −0.33 / 0.00; +0.05 |
| meva 97 | 0.984→1.003; 16→5; +0.32; −0.09 / −0.18; −0.53 | 0.962→0.998; 16→6; +1.31; +0.02 / +0.36; −0.39 |
| meva 194 | 1.033→0.993; 8→7; −0.27; +0.03 / +0.24; −0.20 | 1.006→1.000; 6→6; +1.82; +0.17 / +0.89; +1.59 |
| meva 388 | 1.028→0.991; 7→2; +2.66; +0.15 / +0.48; +1.81 | 1.027→0.992; 3→2; −0.52; −0.15 / −0.11; −0.94 |

- Re-codes fall 84→46 (AV1) and 43→20 (H.264).
- No AV1 point ends more than 1.4% over target, and none gets worse on rate.
- The worst worst-1% change is H.264 meva 388 at −0.52 dB.

**Synthetic noisy stills.** The static-noise agent's recipe: frame 0 of `static_meva_hospital` at 854x480, 300
frames, `add_noise.py`, 15 fps, CBR 1000 kbps, one key. Each cell is kbps / PSNR-Y against the clean original.
"skip" is `decision.noise_skip=4`, `noise_skip_chroma_ratio=2` and `rate.skip_budget=1`; "both" adds the band.

| sigma | codec | off | band | skip | both |
|---|---|---|---|---|---|
| 1.15 | AV1 | 988 / 43.73 | 991 / 43.44 | 845 / 43.83 | 893 / 43.49 |
| | H.264 | 985 / 44.05 | 987 / 43.99 | 361 / 44.04 | 369 / 44.00 |
| 2.9 | AV1 | 988 / 37.31 | 991 / 37.05 | 265 / 37.94 | 378 / 37.54 |
| | H.264 | 986 / 38.06 | 987 / 38.05 | 109 / 38.95 | 109 / 38.97 |
| 7 | AV1 | 988 / 31.55 | 991 / 31.22 | 728 / 31.51 | 770 / 31.24 |
| | H.264 | 985 / 32.32 | 984 / 32.37 | 575 / 32.77 | 574 / 32.83 |

- **H.264:** the band leaves the noise skip's undershoot and quality intact. Both together match the skip alone
  within 2% of rate and 0.06 dB.
- **AV1:** the band still costs on noisy stills. Alone it loses 0.26-0.33 dB against the clean scene at the same
  rate. With the skip it adds 6-43% rate and loses 0.27-0.40 dB.
- Before the new candidates the AV1 losses were twice as large: band alone −0.94 / −0.98 / −0.43 dB, and with the
  skip 910 / 390 / 774 kbps.
- AV1's noise skip is itself weak here (845 kbps at sigma 1.15 against H.264's 361; the static-noise agent's open
  AV1 finding). So the band should stay off for AV1 on noisy still scenes until that gap is understood.

**Moving content** (a5392646, lowest rung, 600 frames, band on vs off):

| Clip | Codec | Rate | PSNR-YUV | VMAF v1 | Worst-1% Y |
|---|---|---|---|---|---|
| pan_jaad_video_0194 | AV1 | 0.988→0.986 | −0.07 | −0.56 | −0.11 |
| | H.264 | 0.989→0.989 | −0.01 | −0.35 | +0.06 |
| bodycam_bc1_540p | AV1 | 0.989→0.990 | −0.02 | −0.53 | 0.00 |
| | H.264 | 0.987→0.987 | −0.04 | −0.26 | 0.00 |

No re-codes in either arm. The loss came from flat areas on a moving camera counting as still (section 5.2).

### 5.2 Tuner result and per-frame diagnosis: not adopted (2026-11-14)

**The tuner's screens of `rate.cadence_spatial=1` on AV1 Default were worse.**

| Level | J (90% interval) | Notes |
|---|---|---|
| L1 | −0.59% | |
| L2 | +0.50% | static class +1.7% |
| L3 | +0.51% [+0.22, +0.80], n = 127 | |
| 600-frame final (30 clips, key every 10 s) | +2.01% [+1.33, +2.66] | static +2.59%; pan guard tripped (+3.43%) |

The rate side worked in the final: the Default's meva_admin overshoot was gone and no point went over target.

**Per frame** (AV1 Default against band, every rung, one key):

- **Before the band acts.** On fit_static_otw283, fit_static_otw399 and pool_lowlight_jaad_0026_540p, the two
  arms are identical until frame 104, where the start-up cadence ends.
- **Then the band loses ground.** It gains 0.2-0.4 dB in frames 104-119, then loses 0.3-1.3 dB in frames 120-149.
- **What the Default does there.** Its base frames in that window are deep refinements of the whole picture, far
  below the cadence's 3 QP because the level's limit cycle lands them deep:
  - otw283 at 600 kbps, frame 124: qindex 58 and 76 kB, against 112-120 around it.
  - otw399 at 1200 kbps, frame 116: qindex 75 and 65 kB.
  - Every later frame inherits these refinements. The band arm codes the same frames at qindex 114 for 2.5 kB.
- **pool_chad_4_109 over 600 frames.** Rate is the same in both arms (0.978-0.989 against 0.986). The Default's
  bursts at frames 236, 424, 484 and 544 (64-118 kB at qindex 77-116) are each 3.5-5.5 dB above the band arm at
  the same frame. Mean PSNR-Y over 600 frames is 0.2-0.56 dB lower with the band.

**Deeper bands do not recover it.** Band boost 6 and 9 QP instead of 3: rate control raises the level to pay for
the band (otw283 at 600 kbps: q 112 → 141 / 162). Frames 120-149 still lose 0.2-1.5 dB on all three clips,
roughly as at boost 3.

**Moving content.** The VMAF loss on moving clips came from flat areas on a moving camera: they match equally at
every vector, so they counted as still and were coded as copies. Two probes:

- Keeping the frame cadence for the share outside the band made it worse (AV1 pan VMAF −1.21).
- A camera-still condition on the candidates removed most of it (pan_jaad_video_0194: AV1 −0.56 → −0.10, H.264
  −0.35 → −0.06).

Neither was landed, because the lever is not adopted.

**Conclusion.**

- **Where the quality comes from.** The Default's quality on static scenes comes from the same deep, rare
  whole-picture refinements that cause the bursts and the overshoot. A whole-picture refinement is inherited by
  every later frame until the scene changes, and at equal rate it buys more than the same bits spread over a
  budgeted band.
- **Why the band cannot match it.** The band is paid for in every frame's budget, so it can only refine shallowly.
  Refining deeply means borrowing bits that later frames repay, which is the burst again.
- **What could be tried, not tried.** A band refined deeply and rarely, with its bits treated as a cycle-long
  loan, would be bursts by area rather than by frame. It may still keep frame sizes smoother, but nothing measured
  here says it would win J.
- **Decision.** `rate.cadence_spatial` stays registered and off.
