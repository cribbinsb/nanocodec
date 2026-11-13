# AV1 held-out gap against SVT-AV1 M12: diagnosis (2026-11-08)

**Question:** AV1 Default is at parity with SVT M12 RTC on TRAIN (PSNR-YUV BD
−2.6%, VMAF v1 −0.1%) but trails on held-out VAL (+10.6% / −4.0%; Fast +12.6% /
−0.7%). Why?

**Sources:**
- Cached certified curves: `work/overnight/current-cross-codec/report.json`, and
  the curve files it cites.
- Matched-size stream statistics: `tools/dev/compare_av1_svt.py` with
  `NC_COMPARE_CLIPSET=main`, binary built from `variance-integration` 00f2151.
  Output in `work/av1-heldout/cmp/`; summariser `work/av1-heldout/cmpsum.py`.

VAL clips were used only for **diagnosis**. Every fix below is selected on TRAIN
(`work/av1-heldout/screen.py`).

## 1. The VAL mean is one clip, plus a high-rate effect common to both splits

- **bodycam_airport is +115.5% PSNR BD** (overlap 0.85, VMAF +19%).
  - Without it, the VAL mean is −1.1% and the median −0.8%: parity, like TRAIN.
  - The gap is real, not a BD artefact. At about 290 kbps we reach 45.7 dB PSNR-YUV; SVT reaches 47.1 dB, and our HEVC, on the same engine, 48.6 dB.
- **On both TRAIN and VAL, our relative rate grows towards the high-quality end of
  each curve.**
  - Rate difference against SVT at 10% / 50% / 90% of the overlapping PSNR range:

    | Split | Clip | Low | Mid | High |
    |---|---|---|---|---|
    | VAL | lowlight_pp22_049 | −5 | +9 | +44 |
    | VAL | lowlight_bdd_00ce95b0 | +4 | +7 | +24 |
    | VAL | crowd_pp22_051 | −10 | −6 | +16 |
    | TRAIN | lowlight_bdd_00c497ae | +6 | +9 | +31 |
    | TRAIN | lowlight_pp22_004 | +11 | +14 | +23 |
    | TRAIN | crowd_pp22_027 | −5 | +2 | +13 |
    | TRAIN | static_otw_homes432 | −8 | −1 | +13 |

  - Mid and low rates are at or better than parity on most clips.
  - This is not overfitting to TRAIN: the effect is systematic. The splits differ mainly in composition (VAL holds bodycam_airport and two low-light clips).

## 2. Matched-size stream statistics (same bytes as SVT, whole clip)

| | bodycam_airport q22: SVT | ours | lowlight_pp22_049 q27: SVT | ours | lowlight_pp22_049 q48: SVT | ours |
|---|---|---|---|---|---|---|
| P base qindex | **88** | 111 | **108** | 127 | **192** | 201 |
| P luma PSNR (dB) | **47.09** | 44.78 | **41.20** | 39.65 | **34.85** | 34.65 |
| P chroma U/V (dB) | 47.0/46.8 | **47.9/47.7** | 48.8/48.8 | **49.5/49.3** | 46.6/46.6 | **47.5/47.3** |
| coded transform area | **10.7%** | 21.5% | **14.3%** | 24.6% | **7.4%** | 10.9% |
| skip area | **89%** | 79% | **86%** | 75% | **93%** | 89% |
| NEW area | **15.8%** | 4.8% | **22.9%** | 12.8% | **16.0%** | 5.6% |
| MV bits per P frame | **724** | 202 | **3088** | 1554 | **1044** | 469 |
| header bits per P frame | 458 | 615 | 832 | 901 | 383 | 580 |
| key-frame bytes | 29 618 | 41 570 | 85 769 | 76 287 | 23 153 | 22 545 |
| key-frame luma / chroma U (dB) | **54.8**/57.2 | 53.8/**61.9** | **50.3**/53.0 | 48.3/**53.9** | **42.0**/49.7 | 41.3/49.7 |
| key-frame blocks | 64/32/16 mixed | 16×16 only | 32/16 mixed | 16×16 only | 64/32/16 mixed | 16×16 only |
| P bytes by frame index mod 4 | 1216/969/1106/1129 | flat | 4583/3177/3672/3602 | flat | 838/424/516/532 | flat |

Luma over the bodycam_airport clip (frames 0–9, 10–49, 50–99, 100–199, 200–299):
SVT 51.2 / 48.1 / 47.4 / 47.2 / 46.1 dB; ours 50.7 / 47.0 / 45.6 / 44.6 / 43.2 dB.

**Diagnostic** (VAL, not used for selection): on bodycam_airport, chroma_delta_q
−24 versus 0 costs +7.2% bytes at q111 and +12.5% at q100, for +0.2 dB chroma and
no change in luma.

## 3. Ranked hypotheses, each with a TRAIN-measurable fix

1. **P-frame operating point: residual on too much area at too coarse a quantiser.**
   - SVT keeps more skip and more NEW at a finer qindex; we spread coefficients over twice the area. The luma deficit grows with rate and with clip length.
   - Fix to screen: a larger mode-decision λ relative to the quantiser step (`av1.lambda_per_step_squared` ×2, ×4), so that at equal rate the encoder sits at a finer qindex with more skip. Related: `decision.new_rate_scale`, the skip-bias family.
2. **Weak key frames at high rate.** Fixed 16×16 DC: luma is 1–2 dB below SVT's at the same or more bytes.
   - Fix to screen: existing default-off tools `planner.intra_plan=1`, `av1.keyframe_intra_modes=2`, `av1.keyframe_cfl=1`.
3. **Chroma over-allocation on clean content.** `av1.chroma_delta_q −24` only attenuates for noisy input. On clean, high-quality content it buys about +0.2 dB chroma for 7–12% rate.
   - Fix to screen: `chroma_delta_q=0`, or an offset that also fades towards zero at low base qindex.
4. **No 4-frame effort cadence.** SVT spends 20–45% more bytes on every 4th P frame. Earlier engine-only λ cadence tests were neutral (see av1.md); the backend-level version is untested.
5. **Header overhead:** 70–200 bits more per P frame. This matters only at low rate.

## 4. TRAIN screen results

(8 TRAIN clips, 150 frames, cached TRAIN quantizers, PSNR-YUV from the reconstruction
and pinned VMAF v1; negative BD means fewer bits.)

BD against SVT M12 / against our Default. Mean over the 8 clips (lowlight_pp22_004,
lowlight_bdd_00c497ae, lowlight_city_night, crowd_pp22_027, static_otw_homes432,
bodycam_antare_0815, bodycam_bc1, pan_jaad_0223). Screen runner: `work/av1-heldout/screen.py`,
log `screens.log`. These screen points are not conformance-gated (encoder
reconstruction only); acceptance uses the gated full-TRAIN run in §5.

| Candidate | vs SVT PSNR | vs SVT VMAF | vs Default PSNR | vs Default VMAF | Verdict |
|---|---|---|---|---|---|
| Default (base) | +1.86 | +0.89 | — | — | reproduces cached curves exactly |
| `av1.chroma_delta_q=0` | +7.73 | +5.76 | **+6.77** | **+4.93** | rejected: worse on all 8 clips |
| `av1.lambda_per_step_squared=0.27` (λ×2) | +11.24 | +11.83 | **+10.49** | **+11.10** | rejected (λ×4 skipped) |
| `av1.lambda_per_step_squared=0.10` (λ×0.75) | +1.92 | +0.74 | −0.44 | −0.17 | noise; homes432 +6.9/+5.2 |
| key-frame tools (`planner.intra_plan=1`, `av1.keyframe_intra_modes=2`, `av1.keyframe_cfl=1`) | +1.47 | −1.45 | **−0.32** | **−2.18** | to full TRAIN |
| `decision.new_rate_scale=0.5` | +0.95 | −0.15 | **−0.95** | **−1.06** | to full TRAIN |

What this says about the hypotheses:

- **H3 (chroma) is wrong as stated.** Chroma's extra bits buy back more than they
  cost in PSNR-YUV, which weights chroma at 1/3, and in VMAF through better
  references. The VAL diagnostic exaggerated the cost because it looked at rate
  only.
- **H1 is not a global λ problem.** Raising λ moves skip and qindex the
  SVT way but loses heavily: the SVT gap lies in *which* blocks get residual
  (search and NEW quality), not in a global trade-off. Cheaper NEW pricing (more NEW, as
  SVT does) helps, mostly on bodycam_bc1 (−6.6% / −7.1%).
- **H2 (key frames) is confirmed modestly.** Gains concentrate on the low-light
  clips, where SVT's key-frame luma lead was largest.

## 5. Gated full TRAIN (21 clips, Default grid, 50% cap)

Runner `work/av1-heldout/fulltrain/run.py`, frozen binary SHA256
`0b42423e…8c67` (`variance-integration` 00f2151). Each point needs own decoder +
dav1d/ffmpeg + aomdec equal to the encoder reconstruction. The anchor is the
certified Default TRAIN curve set.

| Candidate | vs Default PSNR | vs Default VMAF v1 | Notes |
|---|---|---|---|
| key-frame tools | −0.08 | −1.82 | low light −2.6/−4.4, homes432 +6.5/+2.0; +2.4% CPU amortised over a 300-frame GOP |
| `decision.new_rate_scale=0.5` | −0.42 | −0.56 | bodycam −3.1/−3.1, crowd +1.4/+1.5; CPU neutral |
| both | −0.21 | −2.50 | |

All three are below the adoption bar: no VAL run, nothing adopted. They were
superseded by §6.

## 6. Root cause: P-frame CDEF too weak (2026-11-08)

The two VAL clips discussed here, bodycam_airport and the
`work/coord-airport/static.yuv` segment, were used for diagnosis only.

**Ruled out** on static.yuv (each tested directly; none moved the curve more than about 5%):
- motion vectors: within about 25% SAD of a ±12 px full search, and HEVC shares them;
- decision λ ×0.5 to ×4, planner λ, splitting, depth removal, forced 16x16;
- chroma offset, rounding 24/64, the residual-RD vector limit;
- interpolation filter (SVT also uses regular) and MV precision;
- sub-pel move biases and walks.

**64x64 transforms are not a bug.** Our 64-point transform matches an ideal
32x32 low-frequency truncation (28.48 vs 28.50 dB). The planner picks 64x64
only on flat areas unless forced to.

**Cause:**
- SVT M12 does not deblock: `--enable-dlf 0/1` gives the same md5.
- With `--enable-cdef 0`, SVT drifts exactly as we do: static.yuv qp18 goes from 276 kbps/46.03 dB to 301/44.14, and luma decays over the clip.
- Our P-frame CDEF is nearly off at high quality: the QP model gives Y primary 1, UV 0 at q90.
- The existing strong per-64x64 luma candidate was default-off. Its 350 variance gate excludes bodycam_airport.

**Fix:** `av1.cdef_per_superblock=1`, `av1.cdef_local_strong_luma=1`,
`av1.cdef_local_strong_variance_min=0`.

| Test | Before | After |
|---|---|---|
| static.yuv q90 | 285.5 kbps / 44.67 dB, Y decaying | 272.9 / 47.40, flat |
| bodycam_airport q111 (VAL, diagnosis) | 284.7 / 44.81 Y | 264.0 / 46.66 |

6-clip TRAIN screen (150 f, gated), BD vs Default, PSNR-YUV/VMAF v1:

| Clip | PSNR | VMAF v1 |
|---|---|---|
| homes432 | −7.06 | −7.41 |
| lowlight_pp22_004 | −8.15 | −7.27 |
| crowd_pp22_027 | −4.80 | −0.92 |
| meva_hospital | −1.17 | −1.04 |
| pan_jaad_0223 | −0.43 | +0.98 |
| bodycam_bc1 | +0.51 | +1.08 |
| **mean** | **−3.52** | **−2.43** |

Callgrind cost: +2.3% to +5.9%.

**Follow-up VAL** (10 natural clips, full length, streamed engine, BD vs the
certified Default VAL curves):

| Clip | PSNR | VMAF v1 |
|---|---|---|
| bodycam_airport | −47.8% | −15.1% |
| lowlight_pp22_049 | −14.3% | −6.5% |
| crowd_pp22_051 | −6.0% | −3.7% |
| pan_bdd_00cef86b | −5.3% | −1.9% |
| static_cam29 | −4.3% | −3.4% |
| static_london_bus | −3.6% | −5.6% |
| handheld_mot17_10 | −2.4% | +0.1% |
| pan_jaad_0101 | −1.6% | +0.9% |
| lowlight_bdd_00ce95b0 | −1.4% | +1.6% |
| static_otw_homes390 | +12.8% | −10.4% |
| **mean** | **−7.39%** | **−4.38%** |

- Noise clips improve at matched rate: +0.2–0.6 dB PSNR, +0.8–2.9 VMAF.
- Against SVT the VAL mean is now −3.0% / −8.6%; it was +10.6% / −4.0%.
- Default adopted.

**Still pending:**
- the Fast-preset arm (cap3, gate 300 vs 0);
- long single-keyframe chains.

**Remaining gap after the fix** (bodycam_airport 720p, 150 f, matched Y about
47.0): we need 149 kbps where SVT needs 116 (+28%).
- About 45% of the extra bytes are the key frame: 30.2 KB vs 20.9 KB, at −0.4 dB.
- About 55% are P frames: +22% bytes per frame.
- The keyframe and P-frame partition work has moved to another agent.
- Next here: CDEF strength selection. SVT's header luma primaries are about 10–14 and its chroma 4–7; ours are 1–4 and 0–2.

## 7. CDEF strength selection (2026-11-08)

Staged screens with `nceval.screen`, BD vs the default at the time, PSNR-YUV/VMAF v1:

| Change | Result | PSNR | VMAF v1 | Outcome |
|---|---|---|---|---|
| strong secondary 1 vs 0 | BETTER (full TRAIN proxy) | −1.89% | −1.45% | landed a019f6c |
| strong secondary 2 vs 1 | BETTER | −0.97% | −0.53% | landed 3a1f8a4 (default 2) |
| chroma strength search, primary 4 | WORSE | −0.08% | +1.05% | — |
| chroma strength search, primary 8 | cut | | | — |
| frame-level CDEF on every block, SVT level 9 style, staged TRAIN | WORSE (proxy-6) | +2.50% | +1.47% | — |
| same, static set (meva, homes432, cam22, PP22 low light) | WORSE | +11.2% | +8.1% | — |
| key-frame luma primary 7 | flat after two stages | +0.01% | +0.01% | option removed |
| key-frame luma primary 15 | AMBIGUOUS at proxy-4 | −0.46% | −0.25% | option removed |
| strong primary 10 (gentler, with secondary 2) | stopped: worse at proxy-4 and proxy-6 | +1.62% | +0.57% | rejected |
| strong primary 7 | not run: primary 10 already loses | | | rejected |

Notes:
- **Secondary 2:** the default now applies a secondary strength of 2 to the strong per-superblock luma candidate.
- **Chroma search:** the option stays (`av1.cdef_uv_search_primary`, default off).
- **Why frame-level CDEF was tried:** aom-inspect shows SVT M12 filtering about 90% of blocks with one frame strength: (15,0) on base frames and the midpoint (7,2) elsewhere. Our per-superblock search leaves about 64% unfiltered on meva.
- **Frame-level CDEF result:** it was worse everywhere, worst on static content, and cheaper (−4% to −6% instructions). The option was removed. Our per-superblock strong selection beats SVT's frame policy on our encoder.
- **VAL-150 confirmation of secondary 2 vs 0:** 11 natural clips, run in two halves. Mean −3.44% PSNR / −4.82% VMAF; worst homes390 +5.3/+2.4.
- **Gentler strong primary:** it barely changes meva texture retention at q200 (0.375 at primary 15, 0.382 at 10, 0.388 at 7) and loses on TRAIN.
- **meva VMAF gap vs our HEVC** (AV1 about −3.4 VMAF at equal rate, equal PSNR): this is texture loss. Luma Laplacian energy, recon/source: AV1 0.375 (0.413 without strong CDEF) vs HEVC 0.482. Most of the gap is already present on the key frame and early P frames, so it is a partition and transform-size question rather than CDEF.
