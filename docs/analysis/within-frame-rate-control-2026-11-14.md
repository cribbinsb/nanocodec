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

Pending: in-frame control (`inframe.*`, branch `feature/inframe-rc`).
