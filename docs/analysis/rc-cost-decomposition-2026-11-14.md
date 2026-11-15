# What CBR rate control costs against fixed QP, part by part (2026-11-14)

**Question (coordinator).** Our CBR costs +10.7% (H.264 Default) to +21.5% (HEVC) BD-J against our own fixed
QP (`docs/results/rc-cost-2026-11-14.csv`). Where does that come from: frame-to-frame quantizer jitter, the
size model's misses, re-codes, key frames, slow drift, or the tools that only run under rate control? Then fix
the largest part that moving content (crowd, pan, bodycam, lowlight) pays. Still-scene drift and bursts belong
to other work (`rate.period_ceiling`, `rate.key_hold_still`, `rate.held_quality`, `rate.hold_recover`) and are
reported here only.

## Method: replaying quantizer sequences

`nanocodec encode --qp-replay FILE` (new, `crates/nc-cli/src/qp_replay.rs`) imposes one quantizer per input
frame on rate control's plan (`Encoder::force_next_quantizer`, `RateController::force`). Everything else runs
as usual: the budgets, buffers, models, the reuse term and the hold. A forced frame is never re-coded.
Replaying a stream's own `--rate-log` reproduces it byte for byte (checked on AV1 and H.264, `pan_jaad_0101`,
300 kbps), so a replayed sequence differs from CBR only in its quantizers.

`python3 -m nceval.rc_oracle` (`eval/nceval/rc_oracle.py`) encodes every CBR point once more with
`--rate-log`, and replays four variants of its logged sequence:

| Variant | Inter frames | Key |
|---|---|---|
| `replay` | as logged | as logged |
| `smooth` | each level replaced by its centred 1 s mean (step domain); cadence and layer offsets kept | as logged |
| `flat` | every level at the sequence's mean step | as logged |
| `flat_key` | as `flat` | the fixed-QP key's step ratio to the level (H.264 −6 QP, HEVC −15 QP, AV1 ×0.24 step) |

Each variant's four ladder points make a curve; BD-J (the project's J) between neighbours in the chain
CBR → replay → smooth → flat → flat_key → fixed QP splits the whole cost into:

- **re-codes** (CBR vs replay): what coding a frame twice changes;
- **fast jitter** (replay vs smooth): level moves faster than 1 s, including those the size model's misses cause;
- **slow drift** (smooth vs flat): the start-up descent and content-driven moves over seconds;
- **key allocation** (flat vs flat_key): CBR's key size against fixed QP's key-to-level ratio;
- **the rest** (flat_key vs fixed QP): what the rate-control-only tools and remaining schedule differences do.

The steps are BDs between different curves, so they add up only approximately. Setup: rc_cost's set (the 11
natural VAL clips, 150 frames, one key, the real-use ladder 150/300/600/1200 kbps at 720p30), our
reconstruction, class-balanced. With two clips per class the class columns are indicative only.

## Decomposition

BD-J of each step (positive: the step costs bits). Build dfbd4692^ (the integration head 2dc89a37 plus the
replay hook, byte-identical without `--qp-replay`). Class columns are the means of their two clips.

| Step | AV1 Default | crowd / pan / bodycam / lowlight / static | H.264 Default | crowd / pan / bodycam / lowlight / static |
|---|---:|---|---:|---|
| **Whole cost** (CBR vs fixed QP) | **+16.0** | +6.4 / −1.2 / +2.2 / −0.4 / +73.0 | **+10.7** | 0.0 / +1.0 / +4.8 / −2.0 / +49.7 |
| re-codes (CBR vs replay) | 0.0 | 0 / 0 / 0 / 0 / +0.1 | 0.0 | 0 / 0 / 0 / 0 / 0 |
| fast jitter (smooth → replay) | +1.7 | +1.1 / +3.7 / +3.4 / +0.3 / +0.2 | +0.4 | +0.9 / +1.8 / +3.9 / +0.7 / −5.4 |
| slow drift (flat → smooth) | +5.2 | +0.9 / −0.8 / −0.6 / +0.8 / +25.9 | +9.3 | +1.0 / −0.5 / −0.1 / +0.9 / +45.1 |
| key allocation (flat_key → flat) | +12.6 | +3.6 / −4.1 / 0.0 / −1.7 / +65.0 | +6.1 | +3.5 / −0.5 / +1.5 / −0.9 / +26.9 |
| the rest (fixed QP → flat_key) | −0.3 | 0.0 / +0.2 / −0.5 / +0.2 / −1.4 | −1.4 | −5.1 / 0.0 / −0.2 / −2.5 / +0.6 |

Achieved rate over target (median of the 44 points): replay 0.985 / 0.979 (AV1 / H.264), smooth 0.949 / 0.960,
flat 0.953 / 0.945, flat_key 1.025 / 0.960. The BDs use achieved rates.

Controller statistics from the rate logs (mean per point; AV1 qindex, H.264 QP):

| Class | key share of bits (AV1 / H.264) | level SD | mean \|Δlevel\| per frame | size model: log2(actual/predicted) SD, inter | misses ≥ 4x | re-codes per point |
|---|---|---|---|---|---|---|
| static | 0.12 / 0.11 | 21.1 / 3.4 | 5.8 / 1.1 | 1.35 / 1.38 | 13% / 16% | 0.6 / 0.4 |
| crowd | 0.10 / 0.10 | 8.1 / 1.4 | 5.1 / 1.6 | 0.37 / 0.34 | 0.3% / 0% | 0 |
| pan | 0.07 / 0.06 | 13.1 / 2.3 | 3.1 / 1.0 | 0.51 / 0.31 | 1.3% / 0.1% | 0 |
| bodycam | 0.04 / 0.04 | 11.4 / 1.9 | 6.7 / 1.2 | 0.55 / 0.63 | 1.3% / 2.4% | 0 |
| lowlight | 0.07 / 0.07 | 7.4 / 1.7 | 2.9 / 1.2 | 0.27 / 0.27 | 0.3% / 0.3% | 0 |

### What it says

- **Still scenes carry most of the whole cost**: AV1 static +73% and H.264 +50%, against −2..+6% for every
  moving class. Two parts make it up. Slow drift costs +26% (AV1) and +45% (H.264): the level starts fine
  after the key and drifts, and on static content a constant level is far better. The key costs +65% / +27%:
  CBR's key is about 35 qindex (AV1) or 4 QP (H.264) finer than the level, while fixed QP's is ×0.24 step
  (−12 QP). On a still scene every later frame inherits the key, so the finer key pays. This is the
  still-scene area (other work); the window is 150 frames with one key, which favours investing in the key.
- **Re-codes cost nothing measurable** (≤ 0.1). On moving content they do not happen.
- **The size model's misses are a still-scene problem.** On moving content the model predicts inter frames
  within 1.2-1.5x (log2 SD 0.27-0.63), with almost no 4x misses. The 13-22x misses the audit found are
  static refinement bursts (13-16% of static inter frames miss by 4x or more).
- **On moving content the largest part is fast jitter**: AV1 +3.7 (pan), +3.4 (bodycam), +1.1 (crowd); H.264
  +3.9 (bodycam), +1.8 (pan), +0.9 (crowd). Slow drift is about neutral there (−0.8..+1.0): following real
  content changes is not what costs.
- **Key allocation on moving content has both signs**: crowd +3.6 / +3.5 (a finer key would pay), pan
  −4.1 (AV1) and lowlight −1.7 / −0.9 (CBR's coarser-than-fixed key is better there). A key sized by how much
  of the picture will persist would take both; it is the second lever.
- **The rate-control-only tools help a little**: the rest is −0.3 (AV1), −1.4 (H.264; crowd −5.1).

### Where the jitter comes from

The inter level is re-solved every frame from that frame's own complexity hint (the summed square roots of its
8x8 SADs against the previous frame), so a frame 30% busier than the last scales the level's step by
about 1.3^(1/1.1) = 1.27: 8-10 qindex in AV1. `rate.level_step_scale` (0.5) bounds it to 10 qindex (AV1) / 2 QP
per frame, but does not remove it. AV1 Default, `handheld_mot17_10`, 300 kbps: at a steady 9.1-9.4 kbit
budget the level moved 5-10 qindex frame to frame (mean 5.4). Fixed QP codes a busier frame at the same
quantizer and lets it cost more; the 1 s buffer can absorb that.

### Candidate 1: plan the level from the mean hint (`rate.level_hint_weight`, registered, off)

The level is planned from a running geometric mean of the inter hints instead of the frame's own; the frame's
size is still predicted from its own hint. At 0.25, `handheld_mot17_10` (AV1, 300 kbps): mean |Δlevel| 5.4 →
1.7 qindex per frame at the same rate (279 kbps).

Screen, AV1 Default, `rate.level_hint_weight=0.25`, pool draw 2026-11-14:

| Level | J (90% interval) | Notes |
|---|---|---|
| L1 (24 clips, single point) | −0.24% [−0.67, +0.59] | |
| L2 (48 clips, ladder) | +0.30% [−0.09, +0.70] | worst-1% PSNR-Y −1.4%, worst-1% VMAF −1.0%, flicker −0.01 dB; crowd +0.6, static +0.2 |

**Not adopted.** The causal mean steadies the level but does not recover the centred mean's gain. Either the
gain needs foresight (the centred mean knows the next half second), or the per-frame response does some
good. A `trailing` oracle variant (the mean of the second up to each frame) separates the two. It is in
`rc_oracle.py`, but was parked unmeasured when the key-period ceiling took priority (owner, 2026-11-14:
steady frame sizes are the direction for still scenes).

## The key-period ceiling's key reserve (`rate.period_ceiling`), fixed and in AV1 Default

Owner, 2026-11-14: steady frame sizes are the direction for still scenes, so the key-period ceiling
(`period_ledger.rs`, persistence agent) took priority over the jitter work. Its first form gave AV1 Default
J −1.97% at 600 frames, with every point at or under target. But it squeezed the frames before every key:
- crowd worst-1% term +101%, pan +42%;
- periodic-key pulse guard +2.44 dB against a +0.25 dB limit;
- re-codes 6.0 → 22.8 per 1000 frames on the static and crowd points.

**Two causes, both in the ledger's key accounting.**

1. **A periodic key was charged twice.** The ledger periods are the key period, so a periodic key starts the
   next period, and the debt it leaves is repaid there. The ledger also reserved the coming key's size from
   the period *before* it. Every frame of that period paid for a key that the next period pays for again.
   With keys every 2 s (the pulse check) that halved the inter budget.
2. **The strict-buffer reserve was a whole buffer on any "fixed camera".** A key due before the deadline kept
   the larger of the last key and a whole buffer, scaled by the unchanged-block share from 0.8 to 1. Crowd
   and street scenes have still backgrounds, so they qualified, although their moving part keeps their keys
   small.

**Fix** (`period_ledger.rs`):
- A key that starts the next period is charged to the period before only in proportion to the still share,
  through the same 0.8 → 1 map. A wholly still scene's key holds refined quality, and reserving for it keeps
  a burst from starving it. A moving scene's key is left to its own period.
- A key requested inside the period is still charged in full.
- The strict-buffer reserve is `2 × last key × still`, not a buffer. The 2 comes from the measured growth of
  a still scene's key: 31 budgets after a 17-budget key on chirla 93 kbps.

**600-frame long-window L3** (AV1 Default, keys every 10 s, 31 points, draw 2026-11-14, single point;
`final_only.py` runs L3 alone because the ceiling is inert within one key period, so L1/L2 are identical):

| Variant | J (90% interval) | static | crowd | pan | lowlight | periodic-key pulse (crowd / lowlight) | guards |
|---|---|---|---|---|---|---|---|
| first form (tuner, 64187d24) | −1.97 [−2.86, −1.17] | | | | | +2.44 | pulse |
| reserve scaled by the last key | −1.94 [−2.91, −1.08] | −7.43 | −0.05 | −0.39 | +0.11 | +2.25 (+3.93 / +4.51) | pulse |
| + periodic key not charged before | −1.10 [−2.00, −0.39] | −4.40 | 0.00 | 0.00 | 0.00 | +0.03 | none |
| **+ charged by still share (landed)** | **−1.27 [−2.23, −0.49]** | **−4.98** | −0.10 | 0.00 | 0.00 | −0.10 | none |

Landed form:
- **Rate:** achieved/target max 1.033 → 1.000 (static median 0.993 → 0.977). Worst 1 s window median 1.64 →
  1.64 and max 2.28 → 2.28 (×target). Largest inter frame median 8.5 → 8.8 frame budgets.
- **Re-codes** on the 22 static and crowd points: 6.1 → 11.0 per 1000 frames (first form: 22.8).
- **Single key:** inert. No ledger period ends inside a 1800-frame chirla or 1500-frame crowd stream with one
  key, so the streams are byte-identical.
- **The moving classes are now unchanged** (0.00 is byte-identical). The static gain is about two thirds of
  the first form's. The rest came with the pre-key squeeze.

**Bundle:**
- AV1 Default takes `rate.period_ceiling = 1` (`nc-av1` backend defaults).
- `av1-fast`, `-faster`, `-svc`, `-svc3`, `-svc-fast` and `-svc3-fast` pin it off. They are byte-identical
  (12/12 streams, static_cam29, 2 rungs, 130 frames, keys every 50).
- The new Default equals the old with `--param rate.period_ceiling=1` (2/2).
- `static_recovery` passes (13/13).

**H.264 Default** (same fix, same draw and setup), against the tuner's first-form result J −0.04% [−0.38,
+0.28], which had moving classes +0.3..+1.1 and a +2.52 dB pulse:
- **J:** −0.19% [−0.41, +0.00] BETTER, no guard. Static −0.75%; crowd, pan and lowlight unchanged. Pulse −0.36 dB.
- **Rate:** achieved/target max 0.997 → 0.999. Worst 1 s window max 2.52 → 2.32 (×target).
- **Re-codes:** 6.6 → 7.3 per 1000 frames on the 22 static and crowd points.
- **Bundle:** H.264 Default takes `rate.period_ceiling = 1`. `h264-fast`, `-faster`, `-fastest` and `-balanced`
  pin it off (8/8 streams byte-identical). `static_recovery` passes.
- **Why the gain is small:** H.264 Default already ran under target (prekey quiet second, since 6e1a695b).

**Landed** (verified by the landing conformance check, 470/470 streams bit-exact in every decoder):
- 14553dee (AV1 Default) and 4d23a3c7 (H.264 Default).
- The reserve fix: 11c4c7fe, d5ddaaad and 3de876e0. Clean-up: bc79caf0. Test fix: 80a85d76.
- Tools: 84b65d5f (`--qp-replay`, `rc_oracle`, `rate.level_hint_weight`).

**AV1 final-check** (3de876e0 → 14553dee): full ladder, 600 frames, keys every 10 s, 31 clips. J −1.27%
[−2.11, −0.57] BETTER, no guard.

**H.264 final-check** (bc79caf0 → 4d23a3c7, same setup): J −0.39% [−0.62, −0.18] BETTER, no guard.

## Next steps

- **HEVC** was not decomposed. It has the largest whole cost (+21.5%), so it comes first, with the same tool:
  `python3 -m nceval.rc_oracle --codec hevc --clips val`. Then the pool's `quick` draw for more clips per class.
- **Port the ceiling** to HEVC, VVC, VP9 and AV2 Default: the same 600-frame L3 for each (`final_only.py` in
  `~/nanocodec-data/work/rc-cost/`, a wrapper that runs `nceval.screen` at L3 only).
- **Still-scene key allocation** is the largest part left: +65% (AV1) / +27% (H.264) of the static cost on the
  150-frame window. A first key sized by the start-up lookahead's still share would take crowd's +3.6 and
  avoid pan's −4.1. Measure it at 600 frames, where the window does not reward the key so much.
- **Jitter:** measure the `trailing` oracle. If a causal mean recovers the centred one's −1.7% (AV1), the
  level filter needs a different form, such as a filtered budget correction instead of a filtered hint. If it
  does not, the gain needs lookahead, which low-delay real time does not have.

## Follow-up (2026-11-14 evening)

### The ceiling in every Default

Same fix, same 600-frame L3 (keys every 10 s, 31 points, draw 2026-11-14). Each codec's other presets pin
it off (byte-identical).

| Codec | J (90% interval) | static | moving classes | achieved/target max | re-codes per 1000 (static + crowd) | landed |
|---|---|---|---|---|---|---|
| AV1 | −1.27 [−2.23, −0.49] | −5.0 | ≤ 0.1 | 1.033 → 1.000 | 6.1 → 11.0 | 14553dee |
| H.264 | −0.19 [−0.41, +0.00] | −0.75 | 0 | 0.997 → 0.999 | 6.6 → 7.3 | 4d23a3c7 |
| HEVC (with `rate.prekey_quiet`) | −0.20 [−0.40, −0.00] | −0.78 | 0 | | 6.8 → 8.0 | 0e0da898 |
| VVC | −0.83 [−1.46, −0.30] | −3.3 | 0 | 1.020 → 0.999 | 4.9 → 6.7 | 935b1726 |
| VP9 | −1.33 [−2.53, −0.32] | −5.5 | crowd +0.2 | 1.007 → 1.000 | 5.5 → 10.8 | d5cf4fe9 |
| AV2 | −0.62 [−1.38, +0.05] | −2.8 | crowd +0.3 | 0.999 → 1.000 | 13.0 → 17.2 | 487dff46 |

No guard tripped in any codec. HEVC was screened twice. On the earlier base it gave −0.84 (static −3.4). It
landed after the quiet second before keys (`rate.prekey_quiet`, 98bbd409), which takes most of the same
still-scene gain.

### Key still boost (`rate.key_still_boost`), in AV1 Default at 6

A key is coded `boost × fixed_camera(still share)` QP units finer. The still share is the running share of
unchanged blocks, or the start-up lookahead's share for the first key, and `fixed_camera` is the ledger's
0.8 → 1 map. A boosted key's whole size is scheduled as its loan. On top of AV1 Default with the ceiling,
draw 2026-11-14:

| Boost | L1 | L2 | 600-frame L3 | Guards |
|---|---|---|---|---|
| 6 | −0.57 [−1.24, −0.03] | −1.87 [−2.49, −1.25]; worst-1% PSNR-Y −4.3%, VMAF −5.7% | −1.95 [−2.51, −1.41]; static −7.1, crowd −0.7, pan and lowlight 0 | none |
| 10 | | | −2.63 [−3.34, −1.91]; static −9.2, crowd −1.3 | periodic-key pulse +0.52 dB |

At 6:
- **Rate:** achieved/target max 1.000; the worst 1 s window is unchanged.
- **Re-codes** (static and crowd): 11.0 → 15.8 per 1000 frames, all on still scenes.

### The trailing oracle: how much of the jitter gain a causal controller can reach

AV1 Default, VAL (as in the decomposition, rebuilt at the head):

| Variant vs replay | J | bodycam | crowd | lowlight | pan | static |
|---|---|---|---|---|---|---|
| centred 1 s mean (`smooth`) | −1.5 | −3.4 | −1.0 | −0.3 | −4.2 | +1.4 |
| trailing 1 s mean (`trailing`) | −0.6 | +0.4 | −0.9 | +0.6 | −1.4 | −1.5 |

About 40% of the gain is reachable without foresight, almost all of it on pan and crowd. Body-cam's −3.4
needs the next half second, which low-delay coding does not have. This fits the causal `rate.level_hint_weight`
screen (L2 +0.30%). The jitter lever is worth at most about −0.6% here, so it is not pursued further.

### HEVC decomposition

HEVC Default at the head before the HEVC ceiling (with `rate.prekey_quiet`), VAL, same setup as above. BD-J by
step, whole and crowd / pan / bodycam / lowlight / static:

| Step | HEVC | by class |
|---|---:|---|
| **Whole cost** | **+21.5** | +7.8 / −2.4 / +2.8 / −1.2 / +100.3 |
| re-codes | 0.0 | 0 |
| fast jitter (smooth → replay) | −0.1 | −0.8 / +2.2 / +0.7 / +0.2 / −2.7 |
| slow drift (flat → smooth) | +8.5 | +2.1 / −0.5 / +1.6 / +0.3 / +38.9 |
| key allocation (flat_key → flat) | +16.8 | +6.1 / −5.0 / +0.1 / −1.7 / +84.6 |
| the rest | +0.2 | small |

- **What dominates:** as in AV1 and H.264, still scenes and the key. Here the key is the largest single part,
  and crowd pays the most for it (+6.1).
- **Jitter:** HEVC's fast jitter is small. Its level moves 0.5–1.2 QP per frame on moving content.
- **Size model:** within 1.3–1.5x on moving content (log2 SD 0.30–0.56). 17% of static inter frames miss by
  4x or more.
- **Trailing mean:** it beats the replay on static (−11%, the drift cut short), not on moving content.

### Key still boost in every Default

The same lever, screened through L1 and L2 (150 frames) and the 600-frame L3 (draw 2026-11-14) for each codec:

| Codec | Boost (QP) | L1 | L2 | 600-frame L3 | static / crowd (L3) | re-codes per 1000 (static + crowd) | guards |
|---|---|---|---|---|---|---|---|
| AV1 | 6 | −0.57 | −1.87 | −1.95 [−2.51, −1.41] | −7.1 / −0.7 | 11.0 → 15.8 | none |
| H.264 | 6 | −0.70 | −1.95 | −1.58 [−2.19, −1.05] | −5.8 / −0.5 | 8.0 → 9.7 | none |
| VVC | 6 | −1.67 | −2.78 | −1.66 [−2.40, −1.04] | −6.1 / −0.6 | 6.7 → 10.8 | none |
| VP9 | 6 | −0.16 | −1.69 | −1.84 [−2.42, −1.30] | −6.8 / −0.6 | 10.8 → 15.7 | none |
| HEVC | 4 | −1.64 | −1.02 | −0.42 [−0.90, −0.00] | −1.75 / +0.07 | 8.0 → 10.6 | none |
| AV2 | 4 | −1.46 | −1.69 | −1.69 [−2.36, −1.05] | −6.2 / −0.5 | 17.2 → 20.4 | none |

- **Pan and lowlight:** unchanged in every codec.
- **Rate:** achieved/target max ≤ 1.000 (H.264 0.997).
- **Why HEVC and AV2 run at 4:** at 6 they tripped the periodic-key pulse guard: HEVC +0.36 dB (L3 −0.90%),
  AV2 +0.46 dB (L3 −2.02%).
- **Bundle:** the other presets of each codec pin it off. The new Defaults equal the previous head with the
  parameter set, and the pinned presets are byte-identical (42/42 streams).

**AV1 key still final-check** (195d1297^ → 195d1297, full ladder, 600 frames, keys every 10 s, 31 clips):
J −1.52% [−1.92, −1.09] BETTER, no guard.

**Landed:**
- The ceiling: HEVC 0e0da898, VVC 935b1726, VP9 d5cf4fe9, AV2 487dff46.
- The key still boost: AV1 195d1297, H.264 61c3817e, VVC 39b54eab, VP9 7752ddef, HEVC c2346bf1, AV2 e17fce9d.
- Final-checks of the other nine bundles: `~/nanocodec-data/work/rc-cost/fc-*.log`.

### Fixed per-frame overhead (`rate.fixed_overhead`, registered, off)

The unified-structure agent found that the size model has no fixed term. Header bytes are therefore read as
content that scales with the step, and a byte or three more per frame (an AV1 OBU extension, `av1.frame_ids`)
made still-scene rate control drift.

`fixed_overhead.rs` models `bits = F + m · hint / step^e`:
- `F` is learnt from inter frames with at least 98% unchanged blocks. It follows their smallest size down at
  once, rises by 1/32 of the gap per frame, and is at most half a frame budget.
- The content model learns from `bits − F`, and the level is solved for `budget − F`.

**Probe** at the head with the ceiling and key still boost. AV1, 600 frames, keys every 10 s, the 14 static
long clips each at its rung, PSNR-Y at equal rate; `fo_analyse.py`. The `gflat` arm is the adaptive-layers
stream configured so that it never layers.

| Pair | mean | per clip |
|---|---|---|
| `av1.frame_ids=1` vs flat | +0.03 dB | −0.23 .. +0.63 |
| same pair, both arms with the overhead model | −0.01 dB | −0.09 .. +0.09 |
| `gflat` vs flat | +0.01 dB | −0.18 .. +0.41 |
| same pair, both arms with the overhead model | −0.04 dB | −0.39 .. +0.20 |
| plain flat stream, with vs without | 0.00 dB | −0.43 .. +0.35 |

- **On this head the drift barely shows.** The ceiling appears to have damped it.
- **The model makes the stream about 3x less sensitive to added header bytes** (mean |Δ| 0.11 → 0.04 dB).
- **`gflat` is unchanged.** Its spread is CBR run-to-run sensitivity, not an adaptive-path bias: the two
  streams diverge at frame ~13, when a 2-bit budget difference rounds the level differently (unified-structure
  agent).

**Pool screen** (AV1 Default, draw 2026-11-14): L1 +0.12% [−0.00, +0.12], L2 +0.04% [−0.14, +0.20],
600-frame L3 −0.02% [−0.39, +0.38] AMBIGUOUS. **Neutral, so not adopted.** It stays registered and off.

### Final-checks of every bundle

`nceval.levels final-check`, each bundle commit against its parent: full ladder, 600 frames, keys every 10 s,
31 clips, draw 2026-11-14. No guard tripped in any of them.

| Codec | Key-period ceiling | Key still boost |
|---|---|---|
| AV1 | −1.27 [−2.11, −0.57] (14553dee) | −1.52 [−1.92, −1.09] (195d1297) |
| H.264 | −0.39 [−0.62, −0.18] (4d23a3c7) | −1.37 [−1.74, −1.00] (61c3817e) |
| HEVC | −0.26 [−0.45, −0.10] (0e0da898) | −0.43 [−0.85, −0.06] (c2346bf1) |
| VVC | −1.03 [−1.61, −0.50] (935b1726) | −1.58 [−2.04, −1.14] (39b54eab) |
| VP9 | −1.51 [−2.51, −0.66] (d5cf4fe9) | −1.49 [−1.92, −1.07] (7752ddef) |
| AV2 | −0.82 [−1.38, −0.33] (487dff46) | −1.19 [−1.60, −0.79] (e17fce9d) |

## Re-measure at c323b045 (2026-11-15)

Taken after the key-period ceiling, the key still boost, `rate.prekey_quiet` (H.264, HEVC), the unified
reference plan (AV1, VP9, HEVC) and the fixed-QP profiles had landed. Setup as above (`nceval.rc_cost`,
VAL, 11 clips, 150 frames, one key, 150-1200 kbps); `docs/results/rc-cost-2026-11-15.csv`.

**The fixed-QP side now uses each codec's fixed-QP profile** (`rate.fixed_qp_profile`, set by `--qp`):
- H.264: leaf +2, cadence scale 1, base λ ×0.8, key 8 QP finer;
- HEVC, VVC, AV2: the cadence at fixed strength;
- AV1, VP9: no profile (the anchor is unchanged).

Where a profile improved the anchor, part of the change below is the anchor's, not rate control's.
`rc_oracle`'s `flat_key` now uses H.264's profile key (−8 QP).

| Default | CBR vs own fixed QP, BD-J (2026-11-14) | fixed-QP anchor |
|---|---:|---|
| AV1 | **+13.6** (+16.0) | no profile |
| HEVC | **+21.1** (+21.5) | profile |
| VVC | **+15.0** (+17.1) | profile |
| VP9 | **+14.6** (+16.4) | no profile |
| AV2 | **+12.2** (+13.2) | profile |
| H.264 | **+10.9** (+10.7) | profile |

**Decomposition** (`rc_oracle`, same build). Each cell is the whole-set BD-J, then crowd / pan / bodycam /
lowlight / static:

| Step | AV1 | HEVC | H.264 |
|---|---|---|---|
| **Whole cost** | **+13.6** · +6.4 / −1.4 / −1.3 / −0.3 / +64.6 | **+21.1** · +9.6 / +0.6 / +5.7 / −1.1 / +90.6 | **+10.9** · +0.7 / −1.3 / +1.7 / −2.1 / +55.5 |
| re-codes | 0.0 | 0.0 | 0.0 |
| fast jitter | +2.3 · +0.7 / +4.0 / +1.9 / +0.7 / +4.0 | +0.9 · −0.7 / +2.9 / +2.4 / +0.1 / −0.1 | +2.7 · +0.7 / +1.5 / +4.9 / +0.6 / +5.9 |
| slow drift | **+4.5** · +0.7 / −1.2 / −0.4 / +0.8 / **+22.4** | **+7.3** · +2.1 / −0.7 / +1.1 / +0.2 / **+33.7** | **+4.9** · +1.0 / −0.4 / −0.4 / +0.7 / **+23.5** |
| key allocation | +8.9 · +3.8 / −4.7 / −3.3 / −1.7 / +50.6 | +16.1 · +6.7 / −5.0 / −0.9 / −1.2 / +81.2 | +6.0 · +3.7 / −1.7 / +0.1 / −1.3 / +29.0 |
| the rest | 0.0 | +1.5 | −1.0 |
| trailing mean vs replay | −1.0 · static −4.5 | −3.1 · static −12.6, bodycam −2.0 | −2.5 · static −5.2, bodycam −3.9 |

Achieved rate over target (median): replay 0.985 / 0.978 / 0.982, flat 0.954 / 0.919 / 0.951.

- **Key allocation is still the largest part, on still scenes** (static +51 / +81 / +29). The key still
  boost halved AV1's (+65 → +51). HEVC's runs at 4 QP, against a fixed-QP key 15 QP finer than its P frames.
- **Slow drift is a third of the whole cost** (4.5 / 7.3 / 4.9 of 13.6 / 21.1 / 10.9), almost all on still
  scenes. It is the next lever (section below).
- **The size model on still scenes:** inter frames are still missed by 4x or more 10-15% of the time
  (log2 SD 1.2-1.4). Moving content stays within 1.3-1.6x.
- **Measuring the oracle arms:** they are now scored like every other evaluation point (our reconstruction,
  no decode). On this head, AV1 Default's CBR stream `crowd_mot20_06` at 175 kbps fails the standard
  decoders' bit-exact check at frame 72, inside the adaptive-resolution path; with
  `resolution.adaptive=0` it passes. Reported to the coordinator.

### What slow drift is on a still scene

AV1 Default, `static_london_bus`, 300 kbps (bytes per frame against a budget of about 1000):
- **Descent.** Between bursts, frames cost 30-200 bytes at any level coarser than the quality the picture
  holds. The size model reads that as room and walks the level down at its full limit (10 qindex per frame).
- **Burst.** A cadence base frame (−4.2 to −5.6 QP units) then lands below the held quality and refines the
  whole picture at once: 6-40 kbytes, 6-40 budgets.
- **Climb.** The burst's debt is about a buffer. The proportional correction drops to its floor, and the
  level climbs back 60-120 qindex within half a second.

The cycle repeats every 8-16 frames, and the level swings 110-230. HEVC behaves the same at 2 QP per frame
(level 33-52). The `flat` arm's constant level keeps the cadence, so its base frames should still refine (its replay
logs were not kept to confirm it). If so, what it removes is the half-second climbs, in which whatever did
change is coded coarse. That points at the repayment of a burst, not at the descent. The descent is what
`rate.held_quality` stopped, and that lost still-scene quality (`docs/handoff/tuning.md`).

### Three causal levers against the drift, none adopted (2026-11-15)

Each one was built as a registered parameter (off) on c323b045 and screened on the pool draw 2026-11-15.
The code is not kept: none came near a gain, and each is a few lines to rebuild from the description.

| Lever | What it changes | Six still points: AV1, 600 f, key every 10 s, mean Δ PSNR-Y / rate | Pool screen |
|---|---|---|---|
| `rate.held_quality=1` (landed, off; the earlier attempt) | prices the refinement into the plan, so the level stops before it | −0.15 dB (−0.44..+0.04) / +0.2% (on the old base: down to −0.69) | not re-screened |
| `rate.empty_level_step=0.25` | after a near-empty frame (≤ ¼ budget), the level may go below the key's and every later inter level only at ¼ of its step limit | +0.01 dB (−0.22..+0.35) / −0.4% | AV1 L1 **+0.94%** [0.00, +2.78]; static **+6.6%**; rate guard (1.07% over target) |
| `rate.burst_loan=8` | an unplanned inter frame over 8 budgets has its overrun scheduled as a loan repaid over 5 s (the refinement loan), so the level does not jump after a burst | 0.00 dB (−0.19..+0.29) / −1.1% | AV1 L1 0.00% [−0.88, +1.10]; rate guard (**1.74%** over target, anchor 0.36%) |
| `rate.level_smoothing=0.25` | the inter level step is a running geometric mean of the solved steps (the causal counterpart of the oracle's `trailing` arm) | — | L1: HEVC −0.25%, H.264 +0.08% (drone +3.5 guard), AV1 +0.14%. HEVC L2 +0.02% [−0.51, +0.50] (static −1.7, pan −1.1, others +0.1..+0.6) |

What this says:
- **Slowing the descent loses what the coordinator predicted:** the still-scene refinements (static +6.6%).
- **Spreading the burst's repayment** (burst loan) gains no quality and spends past the target. The
  proportional correction already repays a burst about as fast as the target allows. A longer schedule
  only carries the debt out of the 150-frame window.
- **A closed-loop level filter does not reproduce the open-loop trailing arm's gain**
  (HEVC −3.1% replayed, +0.02% at L2). The oracle replays a level sequence against budgets that did not
  react to it. A real filter delays the controller's own response, so the correction pushes harder.
- **The baseline has moved.** At 600 frames with keys every 10 s, the six still points now run at
  0.92-0.99 of target (they were 0.98-1.04 before the ceiling). On that window there is little drift
  left to take. The 150-frame single-key window of `rc_cost` charges the start-up descent and
  the first bursts in full.

**Conclusion.** Slow drift is still a third of the 150-frame cost (AV1 +4.5, HEVC +7.3, H.264 +4.9),
but no causal controller change tried reaches it, and the two that touch the burst cycle cost quality or
rate. What is left of it looks like foresight: the `flat` arm knows the clip's mean level from frame 1.
Next: measure the drift step on the 600-frame long window (keys every 10 s) before more controller work,
because the 150-frame window may overstate it. Key allocation (+8.9 / +16.1 / +6.0, static +29..+81) is
still the largest part. For HEVC, a stronger still-key boost is capped by the periodic-key pulse guard
(boost 6: +0.36 dB).

### The product window: start-up descent against steady-state drift (2026-11-15)

`rc_oracle --clips final --frames 600 --key-seconds 10 --per-class 2 --variants replay,startup,flat,flat_key`
at c323b045, on the 2026-11-15 draw. Only static, crowd and lowlight have 600-frame clips in L3. Clips:
- static: chirla cam 7, aboda video 11;
- crowd: pedestrians, uvg volleyball;
- lowlight: bdd night, bdd100 00cee0e0.

That is 6 clips (5 scored for HEVC), so the class columns are indicative only. The new `startup` arm
flattens only the first second after each key, to the mean level of the rest of that key's period. Its
gain over the replay is what the start-up descent costs. Without a `smooth` arm, flat → replay holds
the drift and the jitter together.

| Step (BD-J · crowd / lowlight / static) | AV1 | HEVC | H.264 |
|---|---|---|---|
| **Whole cost** | **+23.4** · +22.3 / −0.3 / +48.2 | **+13.9** · +21.1 / +1.0 / +19.5 | **+26.9** · +11.9 / +1.2 / +67.6 |
| drift + jitter (flat → replay) | +8.1 · −3.8 / +1.4 / **+26.8** | +12.0 · −7.1 / +2.6 / **+40.6** | +13.1 · −3.2 / +2.1 / **+40.5** |
| of it, start-up descent (startup vs replay) | −0.1 · +3.8 / −0.3 / **−3.9** | −0.8 · +6.7 / +0.1 / **−9.2** | −1.0 · +2.0 / −0.2 / **−4.7** |
| key allocation (flat_key → flat) | +15.3 · **+24.7** / −2.3 / +23.4 | +15.0 · **+28.7** / −2.8 / +19.0 | +8.0 · **+14.6** / −0.8 / +10.3 |
| the rest | +0.3 | +1.0 | +3.2 |

Achieved rate over target (median): replay 0.988 / 0.981 / 0.982, startup 0.974 / 0.982 / 0.974.

- **Start-up descent is a minor part.** It is 10-23% of the static drift (−3.9 of +26.8, −9.2 of +40.6,
  −4.7 of +40.5). It also has the wrong sign on crowd (+2.0..+6.7): there the descent after a key pays.
  A start-up level choice would win at most a few percent on still scenes and give part of it back on
  crowd. **The drift line stops here.**
- **The steady-state drift is the still-scene burst cycle** described above. None of the three causal
  levers reached it.
- **Key allocation on crowd** is the largest single part on the product window: +25 / +29 / +15, with
  CBR's key 49 qindex (AV1), 9 QP (HEVC) or 6 QP (H.264) finer than the level. Fixed QP's key is finer
  still, and with keys every 10 s its quality carries over the crowd's still background. The key still
  boost does not act there: it starts at a running unchanged share of 0.8, which crowd scenes stay
  below. That is the next lever, and it needs no foresight. It needs a sample with pan and bodycam
  too (no 600-frame clips of theirs in this L3 draw), because on the 150-frame VAL window pan pays the
  other way (−4.7 / −5.0 / −1.7).
