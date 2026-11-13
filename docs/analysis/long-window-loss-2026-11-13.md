# Why our gaps shrink from a 150-frame to a 600-frame window (2026-11-13)

**Question.** On the 71 fresh held-out sources long enough for 20 s, our BD-J against x264 `medium` shrinks by 6–10
points from 150 frames (one key) to 600 frames (a key every 10 s). x265 `medium` gains 1.8 points over the same
change. Why? This is a diagnosis; nothing was changed.

**Data.**

- **Build.** Ours at the verified head 8b492382, which carries the rate-control change 62788659 + 71cad753.
- **Clips.** The same 71 sources at both lengths: `ind_*` excerpts (150 frames) and `indl_*` (600 frames), cut
  from 2 s into each source. The first 150 frames of the two forms carry the same bits and PSNR to within 0.01 dB.
- **Measurement.** The independent test's points (independent-2026-11-13.md, "Update"). Window BD uses each
  600-frame stream's own per-frame PSNR-YUV, VMAF and bits.
- **Counterfactuals.** Run locally on the 6 clips where AV1 lost most, at 600 frames.
- **Scripts.** Working scripts and CSVs are in `~/nanocodec-data/work/independent/longloss/`.

**Headline.** The shrink is not a slow decay along the inter chain, not the debit term, and not the key cost.
Most of it comes from where a 5-second window sits.

- **Start-up favours us.** Our encoder invests in the first key and the first second, while x264, x265 and
  SVT-AV1 start low and ramp up. The first 150 frames are our best window against x264, and the later windows
  settle about 4–6 points lower.
- **The rung rule adds up to 3 points.** The chart's low-rung rule fires on fewer clips at 600 frames.
- **The periodic key costs AV1 about 2 points** on the clips where it lost most.

## Causes, ranked by points explained (AV1 Default, 71 clips: −47.1 → −40.8, +6.2 points)

| # | Cause | Points (AV1) | Evidence | Where it points |
|---|---|---:|---|---|
| 1 | **Start-up asymmetry.** Our first window is our best against the anchor; the references improve after their own start-up | ~4.2 of 6.2 (H.264 ~7.6 of 10.0) | Window BD within the same 600-frame streams (PSNR-YUV + VMAF): AV1 −48.9 / −44.6 / −42.9 / −42.4, mean −44.7. H.264 −32.9 / −24.6 / −21.9 / −21.8. x265 −29.6 / −32.0 / −33.7 / −34.0. At an unchanged rate, the references' PSNR-Y rises after window 0 (x264 +0.36 dB, x265 +0.33, SVT-AV1 M8 +0.35), ours falls (AV1 −0.26, H.264 −0.41, HEVC −0.23). On the 6 worst clips: our first 30 frames take 1.5× the average bits at +0.8 dB over frames 30–149, the references' first 30 frames sit 1.8–2.7 dB below; x264 lands at 0.93 of target in window 0 and 0.99 in windows 1–2 | Rate control's steady state after the first key: the first-key budget (`rate.first_key_boost_scale` 2.5), the 5 s repay (`rate.key_repay_seconds`) and the QP the controller settles to afterwards. Our quality is front-loaded; the references' rises over the first seconds |
| 2 | **Fewer clips need the low rungs at 600 frames** | 1.1 (H.264 2.2, HEVC 2.1, VP9 2.6, AV2 2.8, VVC 2.7) | The chart's rule adds 40/75 kbps rungs on 22 of 71 clips at 150 frames and on 9 at 600. On the clips with a BD in both, ladder only, the shrink is AV1 +5.1, H.264 +7.8, HEVC +5.2, VP9 +4.7, AV2 +4.7, VVC +4.8 (rule: +6.2 / +10.0 / +7.3 / +7.3 / +7.5 / +7.5) | Measurement, not encoder. Over 20 s x264's curve overlaps ours on more clips, so fewer clips get the low rungs where x264 is weakest |
| 3 | **The periodic key at 10 s** | ~2.1 on the 6 worst clips; H.264 0.2 | Counterfactual on 6 clips, 600 frames, x264 with the same key structure: AV1 −30.8 with a key every 10 s, −32.9 with a single key. H.264 −16.0 / −16.2. x265 gains from the periodic key (−22.3 vs −19.0). Window 300–449 (after the key at 300) is our weakest window, −0.6 to −0.8 dB below window 0 vs −0.27 to +0.07 for the references. Our keys take 3.4–3.8% of the bits vs 0.9–2.1% | Key budget and repay for periodic keys; `rate.recode_overflow`; the starved-key behaviour after a key |
| 4 | **Debit term** (`rate.debit`) | 0 | Counterfactual: AV1 −30.7 with `rate.debit=0` vs −30.8; H.264 −16.1 vs −16.0 | Not a cause |
| 5 | **First-key boost** | negative, i.e. it helps | `rate.first_key_boost_scale=1` makes the 600-frame BD-J worse: AV1 −28.9 vs −30.8, H.264 −14.6 vs −16.0. The boost pays over 20 s too, just less than over 5 s | Not a cause; part of why window 0 is strong |

**What the metrics say.** The shrink is largest on MS-SSIM: AV1 −40.5 → −30.8 (PSNR-YUV −50.8 → −46.2, VMAF
−49.8 → −45.4; chart rule, 71 clips). MS-SSIM weights structure and detail, so our steady-state frames lose more
structural detail against x264 than the first second shows. A slow decay along the inter chain would show as a
falling trend through windows 1–3. The decline there is small (AV1 −44.6 → −42.4), so drift is at most a minor
part.

**Where, by class** (BD-J change from 150 to 600 frames, chart rule; clip counts):

| | bodycam (13) | crowd (19) | lowlight (6) | pan (17) | static (16) |
|---|---:|---:|---:|---:|---:|
| AV1 | +3.6 | +8.8 | +6.2 | +4.8 | +7.8 |
| H.264 | +11.3 | +8.8 | +10.4 | +12.1 | +7.6 |
| HEVC | +6.3 | +7.9 | +8.8 | +6.3 | +7.4 |
| SVT-AV1 M8 | +1.4 | +3.5 | +5.5 | +0.7 | +2.6 |
| x265 medium | −4.3 | +0.2 | −3.4 | −3.3 | +2.0 |

AV1 loses most on crowd and static, H.264 on pan and body-cam. The 6 clips where AV1 lost most (+16 to +19
points): bwc_videotext_video190 (body-cam), personpath22 00177 / 00212 (crowd), bdd100k 00d1bafa / 00d9e313
(pan), personpath22 00190 (static).

**Why x265 gains.** x265 medium improves against x264 window by window (−29.6 → −34.0). At an unchanged rate its
quality rises after its start-up like x264's, and slightly more, and it gains from the periodic key. Its ABR
converges over the first seconds; ours is front-loaded by design.

## What this cannot show

- **Clip counts.** 71 sources; the counterfactuals ran on only 6, and those 6 were chosen as AV1's worst. Their
  numbers show direction, not the average size.
- **Window BD.** It uses PSNR-YUV and VMAF only, per-frame means; MS-SSIM is not stored per frame.
- **What this does not separate.** "Steady-state rate allocation" from "reference-quality drift" inside cause 1.
  A run with the first key at the steady-state budget *and* a re-targeted steady-state QP would.
- **Nothing was tuned here.** Presets were tuned on 150-frame windows. Re-tuning on longer windows is the
  obvious next experiment.

## Suggested next steps (not done)

1. Tune or screen on 600-frame windows (the tuner's `--final-frames`). Check whether the steady-state QP after
   the first key's repay is too coarse.
2. Periodic-key budget for AV1: a key at 10 s costs ~2 points on the worst clips.
3. Publish BD for both windows. The 5 s figure measures a session's start, the 20 s figure a running stream.
