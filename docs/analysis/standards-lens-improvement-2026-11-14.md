# Standards lens: explaining the gap to our chart, and improving under it (2026-11-14)

**Question (owner).** "Improve our performance in the standards based test ... without too much penalty in
our normal test; or if we find our test methodology is poor we should improve it." Overfitting to the
lens's sequences is allowed for this project (they stay sealed against the normal tuner).

The lens: [standards-ctc-2026-11-14.md](../results/standards-ctc-2026-11-14.md) (AOM CTC v8 low delay, 12
eight-bit sequences, 130 frames, fixed QP; JVET LD-P class E). HM, VTM and libaom speed 0 are exhaustive
reference software; ours is a real-time subset. Comparisons below against x264 medium are at the lens's
settings unless a row says otherwise.

## How it was measured

`python3 -m nceval.ctc local` (new, `eval/nceval/ctc/local.py`) codes one variant on this PC with the lens's
own jobs and worker, scores our shown pictures (no decode: conformance is the landing check), and `compare`
reports BD-rates against the anchor curves already in the results page's points.csv. The `lean` metric set
(`worker.py`) computes only the overall PSNR, MS-SSIM (dB) and VMAF v1 the Weighted figure and J read; it is a
quarter of `--aom_ctc v6.0`'s CPU and gave identical numbers on the check point. Reproduction: the local AV1
Default at variance-integration 7fd7eb1e reproduces the published lens exactly (Weighted −14.4, J −23.5 vs x264
medium; CBR −3.9 / −11.6), and its fixed-QP streams match the cloud run's rates and PSNR bit for bit.

## Part A: the gap between the chart and the lens

**Answer.** Our chart gives AV1 Default BD-J −50.9% against x264 medium; the lens gives −23.5%. Changing one
thing at a time accounts for all of it. The largest single part is the anchor's mode: x264 medium in its own
zero-latency CBR (1 s VBV, no lookahead, default AQ and psy) needs +55 to +71% bits on J against itself at
fixed QP. Ours needs +9 to +10%. The chart measures both encoders as they are used, which is legitimate, but
its headline reads as coding efficiency and most of the margin is x264's rate control.

### Decomposition (AV1 Default, BD-J vs x264 medium)

| Step (one change each) | Clips | J | change |
|---|---|---:|---:|
| Chart as published: ours CBR vs x264 CBR, low-rung rule | 11 VAL, 150 f | −50.9 | |
| 1. Without the low-rung rule (ladder only) | 8 with overlap | −45.4 | +5.5 |
| 2. x264 anchor CBR → fixed QP (ours still CBR) | 11 | −24.3 | +21.1 |
|    of which x264 `--tune psnr` alone (still CBR) | 9 | −38.3 | +7.1 |
| 3. Ours CBR → fixed QP (both fixed QP) | 11 | −37.1 | −12.8 |
| 4. Content: chart clips → lens sequences (both fixed QP; 130 f, the lens's QP points) | 12 lens | −23.5 | +13.6 |
| 5. Metric: J → AOM Weighted (0.92 Y + 0.04 U + 0.04 V) | 12 lens | −14.4 | +9.1 |

Steps 1-3 on the chart's 11 natural VAL clips at 150 frames, fixed-QP points chosen by `cbr.measure_all` to span
each CBR curve's rates (so the rate range stays the chart's). The two fixed-QP rows differ in `n` because
overlap differs. Script: `partA_chart.py` (scratch, results kept in
`~/nanocodec-data/work/ctc-improve/partA-chart-av1.json`).

### Each encoder's own rate-control cost (CBR vs its own fixed QP, BD-J)

Chart-clip figures: [rc-cost-2026-11-14.csv](../results/rc-cost-2026-11-14.csv), written by
`python3 -m nceval.rc_cost` (11 natural VAL clips, 150 frames, one key, ladder 150/300/600/1200 kbps at
720p30 pixel-rate scaled, no low rungs; our build's crates are variance-integration 7fd7eb1e; columns `line`
= cbr_vs_cbr / qp_vs_qp / rc_cost, `test`, `test_mode`, `anchor`, `anchor_mode`, `bd_j`, `bd_psnr_yuv`,
`bd_vmaf_v1`, `bd_ms_ssim`, `clips`). BD-rates are not symmetric: our fixed QP against our CBR is −8.5, our
CBR against our fixed QP +16.0; the cost line is the latter.

| Encoder | chart clips | lens (12 seq) | lens Weighted |
|---|---:|---:|---:|
| ours, AV1 Default | +16.0 | +9.9 | +17.3 |
| x264 medium, default tuning | +70.5 | +54.6 | +74.7 |
| x264 medium, `--tune psnr` | +48.6 | +32.8 | +35.5 |

The same on the lens, our AV1 against each x264 mode (J / Weighted): x264 fixed QP −23.5 / −14.4 (ours fixed
QP) and −11.6 / −3.9 (ours CBR); x264 CBR −37.8 / −36.7 (ours CBR, the chart's pairing on the lens's content);
x264 `--tune psnr` CBR −29.3 / −21.3; x264 `--tune psnr` fixed QP −20.8 / −9.1 (ours fixed QP).

### Every Default (and H.264 Fastest) on the chart's clips (rc-cost-2026-11-14.csv, BD-J)

| Encoder | CBR vs x264 CBR | CBR vs x264 `--tune psnr` CBR | fixed QP vs x264 fixed QP | fixed QP vs x264 `--tune psnr` fixed QP | own RC cost |
|---|---:|---:|---:|---:|---:|
| VVC | −49.8 | −42.5 | −40.4 | −39.2 | +17.1 |
| AV2 | −47.6 | −41.4 | −39.1 | −37.7 | +13.2 |
| AV1 | −45.4 | −38.3 | −37.1 | −33.9 | +16.0 |
| HEVC | −44.4 | −38.1 | −34.7 | −31.4 | +21.5 |
| VP9 | −42.9 | −38.6 | −33.0 | −29.0 | +16.4 |
| H.264 | −18.0 | −13.1 | +0.4 | +2.6 | +10.7 |
| H.264 Fastest | +0.7 | +8.6 | +29.6 | +33.1 | +14.9 |
| x264 medium / `--tune psnr` | | | | | +70.5 / +48.6 |

Clips with a reliable BD differ by row (7-11 of 11). H.264 Default's chart margin over x264 medium is all rate
control: at fixed QP the two are level on these clips.

### Other factors (already measured in the audit, not re-measured)

- Window: over 600 frames our BD-J vs x264 is 4-9 points less favourable than over 150 frames
  ([audit §3](../results/rate-control-audit-2026-11-13.md)).
- Rate range: our gaps are about half over the upper range ([audit §5](../results/rate-control-audit-2026-11-13.md)).
  The lens shows the same: AV1 vs x264 at fixed QP is J −27.9 on the QP 32-47 / qindex 160-235 points and
  −11.8 on QP 22-37 / qindex 110-185 (H.264: −5.3 and +4.6; VVC −35.8 and −17.5).
- Metric: the Weighted figure counts chroma at 8%; our chroma BD-rates are −30 to −50% against x264 (we spend
  relatively more on chroma), which J credits more (PSNR-YUV 6:1:1 is one third of J).

### Is our methodology flattering us?

Partly. Nothing is mis-measured: every BD uses achieved bytes and the encoders run as configured. But the
headline pairs our CBR with x264's weakest mode, with x264's psycho-visual defaults scored on metrics that
penalise them, on content (noisy camera footage) that favours us by about 14 points against clean conferencing
sequences, with a low-rung rule worth another 5.5 points, at low rates where all our gaps are largest. Each of
these is disclosed somewhere; together they make the −50.9 read as a codec-efficiency figure, which it is not.

### Proposed methodology changes (for the coordinator and owner; README and charts not changed)

1. **Anchor in its best real-time configuration for these metrics:** x264 medium `--tune zerolatency,psnr`,
   `--bframes 0 --rc-lookahead 0`, ABR with `--vbv-maxrate` = target and `--vbv-bufsize` = the same 1 s our
   encoders get (a live sender at a 1 s receiver buffer cannot use more), one key. A larger VBV or capped CRF is
   not what a 1 s-buffer sender can do; a lookahead adds latency. With it AV1 Default is −38.3 ladder-only and
   −43.5 with the low-rung rule (the chart's existing tuned-twin row).
2. **A fixed-QP line beside every CBR headline:** ours fixed QP vs the anchor fixed QP on the same clips
   (AV1: −33.9 vs x264 `--tune psnr` on the chart clips; −20.8 on the lens).
3. **An "own rate-control cost" line per encoder** (CBR vs its own fixed QP): ours +16.0 (chart clips) / +9.9 (lens), x264 +33 to +71.
4. Keep the standards lens as the content-independent cross-check and quote it beside the chart.

Implemented in `eval/` for these: `python3 -m nceval.rc_cost` (lines 2 and 3, and the chart pairing without
the low-rung rule, against x264 medium at both tunings; every Default and H.264 Fastest measured); `nceval.ctc local --x264 cbr|psnr-cbr|psnr-qp` (the x264 modes on the lens)
and `nceval.ctc compare --anchor ENCODER:KIND` (any curve as the zero line).

## Part B: improving under the lens

### What our fixed-QP path does

`--qp` is IPPP with one key frame. The key is finer than the P frames (AV1/VP9/AV2 `keyframe_step_permille`
240: about 12 QP units; HEVC/VVC `fixed_qp_key_offset` of the key period). The QP cadence applies at fixed QP
(`rate.cadence_fixed_qp` 1): period 8 with a 4-6 QP base boost for the first 100 frames after a key (every
Default), then period 4 with 3; leaf frames +1; offsets scaled by the frame's still-block share, or full on
low-motion frames (`rate.cadence_motion_gate` 1). Trellis is on, AQ off, the quality cap and every other
rate-control feature are inactive. So fixed QP already has the frame-level structure of our CBR path; what it
lacks against HM LD-P is the reference structure (HM: four references, the three older ones the previous GOP-8
base frames, QP offsets +1 base / +4..+7 others with lambda per position).

Fine-QP check: against HM, our AV1/VVC/HEVC gaps widen at the fine end (AV1 J +47 on QP 32-47 vs +65 on 22-37;
VVC +31 vs +48), while x265's and H.264's narrow. SVT-AV1 M10 widens too (+61 vs +70).

### Candidates, AV1 Default at fixed QP (paired BD vs the Default's lens curves; J / Weighted)

Measured with `nceval.ctc local --kinds qp` on the 12 AOM-LD sequences.

| Candidate | Paired vs Default: Weighted / J | vs x264 medium: Weighted / J (Default −14.4 / −23.5) |
|---|---:|---:|
| `rate.cadence_static_scaling` 0 (full cadence always) | −0.6 / −1.1 | −13.2 / −22.7 |
| `rate.cadence_startup_frames` 3600, startup boost 6 (period 8 throughout) | +1.9 / +1.6 | −11.5 / −21.2 |
| `chroma.policy` 1, `chroma.offset` 1 | +2.4 / +2.4 | −15.6 / −22.3 |
| `av1.inter_rounding_q7` 48 | +0.3 / +0.1 | −14.0 / −23.3 |

(Parameter probes only; the cadence items touch machinery another agent is changing, so no code change there.)
