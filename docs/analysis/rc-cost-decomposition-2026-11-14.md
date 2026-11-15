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
