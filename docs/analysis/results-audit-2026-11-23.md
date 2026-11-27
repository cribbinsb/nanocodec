# Adversarial audit of the published results (2026-11-23)

What a skeptical expert can attack in the README, `docs/results/` and the results site as published at
974ad4629 (release 0.6.0), most severe first. Each finding says what is claimed, how the number is made, the
verdict, the size and the fix. Evidence is the code and data that make the number, not the docs' description of
it.

- **Not covered:** speed timed at a fixed QP while quality is CBR (`docs/results/bd-vs-speed-plan.json`,
  `tools/dev/aws_timing.py`, `tools/dev/history_chart.py`). That is being fixed in `worktrees/chart-cbr`.
- **Measured here:** one probe on the build box (finding 3; the script was scratch and is not landed). Every
  other size is read from published data or from an earlier measurement, named where it is used.
- **Not re-run:** the conformance suites, the fuzzing rounds and the chart points themselves.
- **Verdicts:** *wrong* (the statement is false), *misleading* (true as far as it goes, but a reader draws a
  wrong conclusion), *sound*.

## Summary

| # | Finding | Verdict | Size |
|---|---|---|---|
| 1 | Speed and quality are measured on different clips, resolutions and rungs | misleading | relative speed moves 20–45 % with the clip set |
| 2 | The headline saving is a very-low-rate number and does not say so | misleading | 4–7 points from the low rungs; gaps about half above 600 kbps |
| 3 | Reference settings: openh264 is run without CABAC; libvpx and libaom without cyclic-refresh AQ | misleading | openh264 about 17 % (measured here); libvpx about 7 % on average, mixed by clip |
| 4 | SVC against simulcast: the layer split differs and only one layer is scored | misleading | may be the whole 10-point lead (estimate) |
| 5 | "Fixed QP" is not fixed QP for ours; psy-tuned zero line; J as the default metric | misleading | 5–13 points between metrics; rank flips |
| 6 | x86 fps are rescaled across sessions; Zen 3 mixes releases; charts name the wrong CPU | wrong (label), misleading (fps) | 3–8 % fps; 10 % on Zen 3 |
| 7 | Metrics: 6:1:1 PSNR and native-size VMAF favour our allocation; our PSNR is self-reported | misleading | about 10 points on AV1 against SVT-AV1 M12 (indirect) |
| 8 | "Held-out" is file-level only, re-measured every release, and now used for a parameter sweep | misleading | not measurable |
| 9 | The "rate-invalid" column uses a rule that suits our buffer | misleading | ours 0–5 points flagged; 10–16 % pass one buffer |
| 10 | ffmpeg.wasm and FFmpeg decoder comparisons charge the reference its start-up | misleading | H.264 0.43 → about 0.53; native H.264 level, not ahead |
| 11 | The buffer statement publishes the smaller of two measured costs, on 24 clips | misleading | 3–5.5 % stated, 10–13 % also measured |
| 12 | Build asymmetry: ours PGO + LTO, references stock builds | under-disclosed | 1.4–3.2 % (project's own figure) |
| 13 | Conformance denominators, stale counts and contradictory texts | wrong in the stale copies | — |
| 14 | Fuzzing claim predates most decoder code | misleading | — |
| 15 | Stale tables beside current ones (NVENC, decoders, WASM, FAQ) | wrong in the stale copies | up to 5 points (NVENC) |
| 16 | Smaller items | — | — |

## 1. Speed and quality are measured on different clips, resolutions and rungs

- **Claim:** every CBR chart plots bits against fps; README line 64: "25–66 fps at 720p on one Jetson Orin Nano
  core, 102–198 on one AMD EPYC 9R45 (Zen 5) core (that x264: 27 and 113)"; README line 46: fps "on the charts'
  clips".
- **How it is made:**
  - fps is from three development clips at 720p (`lowlight_city_night`, `crowd_mot17_09`, `bodycam_bc1`), at one
    rung (600 kbps), 150 frames (`docs/results/bd-vs-speed-timing*.json`: `clips`, `rung_kbps`).
  - The same three-clip fps is attached to every clip set's chart: fresh, all held-out and VC-1K
    (`eval/nceval/chart.py` `CLIPSET_SPEED_NOTE`, `with_speed`).
  - The headline range is the six `medium` presets only (`eval/nceval/site.py:1140-1143`); the sentence does not
    say so.
- **Why it is not like for like:**
  - The fresh quality set is 36 of 138 clips at 720p; 66 are 960x540 and 23 are 640x360 or smaller, down to
    424x240 (`web/results/data/clips.json`, set `fresh`). Frame rates run from 10 to 30 fps.
  - Quality is over four to six rungs per clip; speed is one rung.
  - The zero line, x264 `medium --tune zerolatency,psnr`, was never timed. It and every other metric-tuned point
    carry the default-tuned twin's fps (`chart.json` `timing_is_twin: true`; table note "re-timing pending").
    "That x264: 27 and 113" is therefore the fps of a different command.
- **Size:** relative speed depends on content. On one c8a core the Fixed QP views, each timed on its own clips,
  give x264 `medium` 186 / 265 / 96 fps on the CTC, fresh and VC-1K samples, and our HEVC `medium` over x265
  `medium` 3.25x / 3.81x / 4.76x (`chart.json`, views `qp`, `qp-fresh`, `qp-vc1k`). A speed ratio on the plotted
  clip set can differ by 20–45 % from the three-clip one.
- **Verdict:** misleading. The method page and each chart page state the three timing clips; the README headline
  and line 46 do not, and line 46 is wrong for the held-out chart.
- **Fix:**
  - Time the encodes that are scored: a class-stratified subset of each chart's own clips, every rung.
  - Time the tuned twins.
  - Headline: "medium presets; three 720p development clips at 600 kbps" until then. Correct line 46.

## 2. The headline saving is a very-low-rate number and does not say so

- **Claim:** README line 63: "our default (`medium`) presets need 25.7–50.6% fewer bits for the same quality
  than x264 0.165 `medium --tune zerolatency,psnr` on 137 fresh held-out clips".
- **How it is made** (`eval/nceval/chart.py`, `eval/nceval/jreal.py`):
  - Ladder 150 / 300 / 600 / 1200 kbps at 720p30, scaled by (pixel rate)^0.75 (`cbr.clip_targets`). The fresh
    set's median scale is 0.65 and its smallest 0.19: the bottom rung is 29 kbps on the 424x240 clips, and the top
    rung's median is 780 kbps.
  - On a clip where any encoder's curve does not overlap the anchor's, every encoder also codes 40 and 75 kbps,
    scaled the same way (`LOW_LADDER_KBPS`, `low_rung_clips`). That is 62 of the 138 fresh clips (the chart page's
    "Low rungs" line): about 26 and 49 kbps at 540p, and under 10 kbps at 240p.
  - BD-J is the mean of the full-overlap BD and the BD over its lower-rate half (`chart.per_clip_j`): the low
    half counts twice. BD-PSNR-YUV and BD-VMAF on the charts are built the same way, so no column is a plain
    BD-rate.
  - A curve needs three points, not four (`jreal.BD_MIN_POINTS = 3`).
- **Evidence the references are at their floor there:** rate-invalid points on the fresh chart
  (`chart.json`, view `fresh-psnr`, `rate_invalid_points`): x264 `medium --tune psnr` 58, x265 `medium` 43,
  SVT-AV1 M12 29, libaom rt 8 106, libvpx rt 7 170, openh264 268; ours 0–5. They stay on their curves.
- **Size, from the project's own 2026-11-13 audit** (`docs/results/rate-control-audit-2026-11-13.md` section 5;
  `docs/results/independent-2026-11-13/heldout-ladder.csv`; an older build, against the default-tuned x264):
  - The low-rung rule is worth 4–7 points to every encoder against x264.
  - Over the upper range (600 kbps to 4 Mbps at 720p30, 10–11 clips with room) the gaps are about half: AV1
    −24.8 % against −47.2 % by the chart's rule on the same clips, H.264 −9.3 % against −29.0 %, H.264 Faster
    +25.4 % against −14.9 %.
  - That audit's own instruction, "'at low rates' belongs in every headline sentence", was not carried out: the
    headline says "same targets" and gives no rates.
- **Not measured here:** the low-half double weighting alone, and the saving on the 720p clips alone, at 0.6.0.
- **Verdict:** misleading. Each rule is on the method page; the headline sentence and the chart axis read as a
  general saving.
- **Fix:**
  - Headline: "at 30–1200 kbps (150–1200 kbps at 720p30, scaled down for smaller pictures)".
  - Publish a plain full-range BD-rate on the ladder alone beside BD-J, and a view restricted to the 720p clips
    and to the upper two rungs.
  - Require four points per curve for a published number.

## 3. Reference settings

- **Claim:** every chart point of openh264, libvpx and libaom; the selected pairs "H.264 against openh264"
  (−36.0 % on the fresh clips, `medium`); `eval/nceval/chart.py` `METHODOLOGY`: "libaom and libvpx default to
  `--tune psnr` with AQ off".
- **How they are run** (`eval/nceval/cbr.py:296-303,340-366`):
  - openh264: Constrained Baseline with CAVLC (`--cabac 0`), against our High-profile CABAC H.264. The method
    page states it; no page says what it costs openh264.
  - libvpx VP9 and libaom AV1: `--rt --end-usage=cbr` with no `--aq-mode`, and both programs default to 0
    (their `--help`). Their usual real-time setting is cyclic-refresh AQ (`--aq-mode=3`). The 2026-11-13 audit
    tried it only together with `--max-q=52` and ±50 % under/overshoot, which broke the rate at these targets, and
    it was dropped.
- **Measured here** (build box, 2026-11-23): five 720p development clips (`crowd_pp22_051`, `pan_jaad_0101`,
  `lowlight_bdd_00ce95b0`, `static_london_bus`, `handheld_mot17_10`), 150 frames, the chart's own CBR command
  with one option changed, ladder 150 / 300 / 600 / 1200 kbps. Plain full-range BD-rate of the variant against
  the chart's setting, mean of the five clips; libvmaf on CPU. One run, no interval.

  | Variant against the chart's setting | PSNR-YUV | VMAF v1 | MS-SSIM | Per-clip range (PSNR-YUV) |
  |---|---:|---:|---:|---|
  | openh264 `--cabac 1` | −17.3 % | −17.8 % | −17.4 % | −9.6 to −23.2 |
  | libvpx rt 7 `--aq-mode=3` | −7.2 % | −7.4 % | −8.0 % | −22.8 to +5.5 |
  | libvpx rt 9 `--aq-mode=3` | −7.1 % | −6.0 % | −8.3 % | −27.0 to +5.2 |
  | libaom rt 8 `--aq-mode=3` | −3.8 % | −2.7 % | −3.6 % | −18.1 to +2.2 |
  | libaom rt 10 `--aq-mode=3` | −2.3 % | +0.3 % | −3.1 % | −11.7 to +4.1 |

  - **openh264:** CABAC saves bits on every clip and every metric. The chart's openh264 is about 17 % worse than
    openh264 can be at the same speed class, and every "against openh264" figure is that much too favourable to
    us. CABAC's CPU cost for openh264 was not timed.
  - **libvpx:** better on the crowd and static clips (11–35 %), worse on the low-light and handheld clips
    (1–7 %), better on average. On the 150 kbps rung it overshoots more with AQ (up to 2.3x target against 1.9x).
    Five clips do not settle it.
  - **libaom:** the crowd clip gains 11–19 % and the static clip 1–12 %; the others lose 1–8 %.
- **Other settings, from the 2026-11-13 audit:** SVT-AV1's flat P scores 3.5–4.5 % better than its default
  hierarchy, and one thread better than four for x264 and x265, so those choices favour the references. x264
  and x265 are charted at their metric tuning. Sound.
- **Harness:** `~/nanocodec-data/anchors/x86_64/bin/openh264-harness` (2026-11-06) has no `--bitrate`; the charts
  use `~/nanocodec-data/work/ratecontrol/bin/openh264-harness` (2026-11-19), which the build box does not have.
  Two binaries under one name is a reproducibility trap.
- **Verdict:** openh264: misleading, about 17 %. libvpx: probably handicapped by several percent on average; not
  established. libaom: small.
- **Fix:**
  - Chart openh264 with CABAC on (and keep the Baseline row if the product case needs it), re-time it, and
    recompute every pair against it.
  - Measure `--aq-mode=3` for libvpx and libaom on the full clip sets and chart whichever setting scores better
    for each, as is already done for the x264 and x265 tunes.
  - Install one harness binary.

## 4. SVC against simulcast: the layer split differs and only one layer is scored

- **Claim:** `docs/results/svc-chart.md` and `web/results/svc.html`: AV1 L2T1 −48.0 % against AV1 simulcast
  −37.8 % and libaom SVC rt 8 −8.5 % (BD-J of the 720p output; rate is the whole stream).
- **How it is made:** the references and simulcast are given 35 % of the target for the 360p layer
  (`eval/nceval/svcbase.py:57` `BASE_RATE_SHARE`, `split_rate`). Our encoder starts from 35 % and "adapts it per
  frame" (`svc-chart.md` Method).
- **Evidence:** the table's own share column (`docs/results/svc-chart.csv` `base_share`): ours 0.209–0.219,
  simulcast 0.347, libaom 0.303–0.316. Ours puts about 79 % of the bits into the only layer the headline
  column scores; simulcast puts 65 % there.
- **Size:** not measured. As an estimate only, simulcast at a 21 % share would need about 0.65 / 0.79 of its rate
  for the same 720p quality, which moves −37.8 % to about −49 % and removes the lead.
- **Also:** eight development clips, the default-tuned x264 zero line, build c9cff89f (release 0.4.0). The x86 fps
  are 5950X time ratios multiplied by the c8a x264 fps (`docs/results/provenance.json`, `svc-chart.csv`): the fps
  of no machine.
- **Verdict:** misleading.
- **Fix:** code simulcast and libaom at our achieved split (or pin ours at 35 %), or score both layers in one
  objective. Label the speed "relative to x264 medium".

## 5. "Fixed QP" is not fixed QP for ours; psy-tuned zero line; J as the default metric

- **Claim:** `web/results/fixed-qp.html`: "every encoder codes at fixed quantizers… Rate control is taken out";
  `web/results/index.html`: "the AOM low-delay test conditions, the standards' way".
- **Ours at `--qp` is not constant QP:**
  - The CLI turns on `rate.fixed_qp_profile` (`crates/nc-cli/src/main.rs:443-451`).
  - The frame QP cadence stays on (`rate.cadence_fixed_qp` default 1, `crates/nc-encoder/src/params.rs:1125`,
    applied at `encoder.rs:778-789`): base frames finer, the frames between coarser.
  - Key frames take a finer QP (H.264 `keyframe_qp_offset` 8, `crates/nc-h264/src/backend.rs:2156-2161`).
  - Block-level offsets are built in either rate mode (`encoder.rs:398`, `persist.reuse` default 1). Whether
    the published streams carry non-zero block offsets was not inspected.
- **The references are pinned flat:** libaom and libvpx `--end-usage=q --min-q=Q --max-q=Q --aq-mode=0`, SVT-AV1
  `--rc 0 --aq-mode 0 --hierarchical-levels 0`, x264 and x265 plain `--qp`
  (`eval/nceval/encoders/anchors.py:38-41,60-66,168-232`). None of our three columns (Strict, Profile, CBR
  settings) switches our cadence or key offset off (`eval/nceval/ctc/lensrows.py:44-53`).
- **Zero line:** x264 `medium` with psy-rd on (`chart.json` view `qp`, `zero`). The CBR pages call that anchor
  unfair to x264 and use `--tune psnr`. On the development clips our fixed-QP savings are 1–4 points smaller
  against the PSNR-tuned x264 (`docs/results/rc-cost-2026-11-19.csv`: AV1 −39.0 → −35.9, H.264 −6.5 → −4.3).
- **Default metric:** the overview opens the Fixed QP view on J (`web/results/explorer.js:42`), not the standards'
  weighted PSNR. On the published view (`chart.json`, view `qp`, weighted / J):

  | Encoder | Weighted PSNR | J |
  |---|---:|---:|
  | nanocodec HEVC medium | −17.4 | −29.7 |
  | x265 medium | −19.9 | −20.9 |
  | nanocodec AV1 medium | −22.7 | −29.4 |
  | SVT-AV1 M8 | −30.3 | −30.8 |
  | nanocodec H.264 medium | −1.5 | −8.6 |
  | nanocodec AV1 veryfast | −0.5 | −13.9 |
  | HM low delay P | −47.8 | −48.6 |

  x265 `medium` is ahead of our HEVC `medium` on the standards' metric and nine points behind on J.
- **"The standards' way":** 12 of the CTC's 8-bit A3–A5 sequences; A2 and B1 are absent, A4 is coded at 8 bits
  where the CTC codes 10, and the anchor and encoder configurations are not the CTC's
  (`docs/analysis/standards-evaluation-methodology-2026-11-14.md:62,106,143`; `eval/nceval/ctc/sequences.py`).
- **What the page does say:** its column notes name a "QP cadence" and a "fixed-QP profile" for ours. It does
  not say the references have none.
- **Sample views:** on `qp-fresh` libaom rt 8 / 10 and libvpx rt 7 are scored on 7 of 12 clips, libvpx rt 9 on 6,
  sixteen other points on 11; nothing is flagged because the test is a strict "fewer than half"
  (`chart.py` `MIN_SCORED_FRACTION`, `flagged`). The CBR charts enforce one clip set for all points; these do not.
- **Sound parts:** BD is PCHIP over the overlap with no extrapolation, and speed is timed on the scored encodes
  (`tools/dev/fixedqp_timing.py:17-24`).
- **Verdict:** misleading.
- **Fix:**
  - Add a constant-QP row for ours (`rate.cadence_fixed_qp=0`, key offset 0, `persist.reuse=0`), or give the
    references their own key and layer offsets, and say which on the page.
  - Zero line x264 `medium --tune psnr`; default metric weighted PSNR; show the luma column.
  - Reword to "12 of the AOM CTC's 8-bit A3–A5 sequences at the CTC's quantizer points; not its configuration
    or anchor".
  - One clip set per view.

## 6. x86 fps are rescaled across sessions; Zen 3 mixes releases; charts name the wrong CPU

- **Wrong CPU on the charts.** `docs/results/bd-vs-speed-x86.svg` and
  `bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr-x86.svg` are titled "one x86-64 core (AMD Ryzen 9
  5950X)" with the axis "one Ryzen 9 5950X core"; the five chart pages say "The same on x86 (one Ryzen 9 5950X
  core)" and describe the timing as "on a Ryzen 9 5950X, a shared machine". The data is an AMD EPYC 9R45
  (`bd-vs-speed-timing-x86.json` `platform`). Hard-coded at `eval/nceval/chart.py:746-747` and in `METHODOLOGY`.
  **Wrong.** Fix: take the label from the timing file.
- **Cross-session rescale.** Our published ms per frame is our time over the paired x264 time, multiplied by an
  older session's stored x264 time (`tools/dev/chart_refresh.py` `merge_timing`; the timing file's `note`). The
  references keep older sessions' times.
  - Published over raw time clusters by session: references 0.97–1.04, ours 1.04–1.08. The headline "102–198" is
    108–210 in the raw times.
  - `svtav1-m13@asm-avx2` against `svtav1-m13` gives SVT's AVX-512 gain as 14.6 % published and 9.2 % raw.
  - The stated repeatability, "repeats within 0.4 %, separate machines within 1.3 %"
    (`docs/results/README.md`), is within a session; the spread between sessions is 3–8 %.
  - The rescale makes ours slower than the raw times, so it does not favour us; the fault is that points of
    different sessions share one axis.
  - **Misleading precision.** Fix: time every point, references included, in one session per release and publish
    the raw fps, or state ±4 % between sessions.
- **Zen 3.** 19 of our 35 points in `bd-vs-speed-timing-zen3.json` were timed at 20d31869 (release 0.5.0),
  among them every `medium` except VVC; their plotted quality is 4fc9a1de. On Zen 5 the same step moved `medium`
  fps by −10 % (VP9), −11 % (VVC) and +4 % (AV1) (`docs/results/history-2026-11-21/README.md`). The caption says
  so; the head commit's subject ("PGO builds of 207745d7 on every chart machine") does not hold for this file.
  Fix: re-time or drop the 19.
- **Sound:** quality at 4fc9a1de with speed at 207745d7 is one encoder: the only change under `crates/` and
  `presets/` between them is `crates/nanocodec/README.md` (`git diff --stat 4fc9a1de 207745d7`). H.264 `medium`
  (c9cff89f) and `veryfast` (20d31869) rest on the refresh's stream-hash check, which was not re-run here.

## 7. Metrics: what the composite rewards, and who measures it

- **Chroma weight.** PSNR-YUV is 6:1:1 (`eval/nceval/metrics.py`), so chroma is a quarter of that third of J.
  On the 138 fresh clips at the 2026-11-13 build (`docs/results/independent-2026-11-13.csv`, against the
  default-tuned x264 `medium`):

  | Encoder | BD PSNR-Y | BD PSNR-YUV | BD VMAF v1 | BD MS-SSIM |
  |---|---:|---:|---:|---:|
  | nanocodec AV1 | −46.5 | −50.3 | −48.8 | −37.8 |
  | nanocodec HEVC | −47.7 | −48.5 | −52.1 | −40.1 |
  | nanocodec H.264 | −29.6 | −32.0 | −34.6 | −18.9 |
  | SVT-AV1 M12 | −35.7 | −30.8 | −33.3 | −25.1 |
  | x265 medium | −33.8 | −29.7 | −32.3 | −29.9 |

  - Ours gain from adding chroma (AV1 3.8 points) and the references lose (SVT-AV1 M12 4.9, x265 4.1). Dividing
    the columns, our AV1 against SVT-AV1 M12 is about −28 % on PSNR-YUV and about −17 % on PSNR-Y. That is an
    indirect ratio of two BD figures on an older build, not a direct BD.
  - At fixed QP on the CTC clips our BD on U and V is −31 to −53 % with Y at +9 to −20 %
    (`docs/results/standards-ctc-2026-11-14/bd.csv`): the same allocation.
  - MS-SSIM, the one luma-structure metric in J, gives every one of our codecs its smallest saving.
- **VMAF protocol.** The charts' VMAF is model `vmaf_v1.0.16_3d0h` scored at each clip's own size
  (`metrics.py:1-13`). The module itself calls that "an INTERNAL relative metric… not comparable with published
  (Netflix/AOM CTC) 1080p-protocol VMAF". Most fresh clips are below 720p. By default it is computed by VMAFx on
  CUDA with the project's own patches, not by libvmaf (`metrics.py:33-39`). The method page says only "VMAF is
  libvmaf's v1 model, scored on every frame".
- **Who measures ours.** Our points are scored on the encoder's own reconstruction output (`--display-recon`,
  `cbr.py:265`) and their PSNR is computed inside our encoder (`nceval/point.py:160` `EncoderPsnrMetric`); the
  references are decoded by a standard decoder and measured by the harness. The method page states the first
  half. `docs/results/README.md:82-83` and `docs/FAQ.md:97` still say every held-out point of ours "was scored
  from a standard decoder's output", which was true of the 2026-11-13 run only.
- **Sources.** The clips are decodes of already-compressed video, some stored at half size
  (`eval/clipset/select_independent.py`: "downscaled 2x when that lifts it over the floor"). The reference picture
  carries another encoder's artefacts. Disclosed in one sentence on the method page.
- **Verdict:** misleading by omission. No number is wrong.
- **Fix:**
  - Show PSNR-Y and MS-SSIM columns beside J on every chart.
  - State the VMAF protocol (native size, VMAFx) on the method page, or score the 1080p protocol.
  - Run published charts with `--gate sample` or `--gate full`, so a decoder's output is what is scored.
  - Correct the two stale "standard decoder's output" sentences.

## 8. "Held-out" is file-level only, re-measured every release, and now used for a parameter sweep

- **Claim:** README line 81: "held-out camera clips from sources none of our tuning, screens or charts had used".
- **True at file level:** no source file of `eval/clipset/independent.json` appears in any other clip-set file
  (checked against 24 files in `eval/clipset/` and `~/nanocodec-data/clips/`), and the set is sealed from the
  screening pool (`eval/nceval/pool.py:57`).
- **Not independent in kind:** 113 of its 150 clips come from source directories the development set also draws
  from (PersonPath22 38, JAAD 26, BDD100K MOT 24, …), and 140 of 150 from directories the development set or the
  screening pool uses. The site's overview says "most come from the same public collections as the development
  clips"; the README does not.
- **Adaptive reuse:** the fresh chart has been re-measured at seven or more builds since 2026-11-13 (commits of
  11-13, 11-15, 11-16, 11-17, 11-18, 11-20 and 11-21), with its numbers in front of the people and agents choosing
  what to work on next.
- **A parameter sweep on it:** `docs/analysis/frame-allocation-2026-11-21.md` sections 6 and 7 sweep
  `rate.key_cap_buffer` (0.10 / 0.15 / 0.22 / 0.30) and three buffer arms on 24 clips of `independent` and 12 of
  `independent_long`. The planned hard-buffer default will be chosen from that evidence.
- **VC-1K:** the 100 measured clips are unused, but 30 other clips of the same dataset have been in the screening
  pool as class `videocall` since 2026-11-18 (`eval/nceval/pool.py:46-50`), three days before the published build.
  "Used by none of our tuning" is true of the clips, not of the dataset.
- **Size:** not measurable from here.
- **Verdict:** misleading wording; the selection itself is sound.
- **Fix:**
  - Say "clips never used; same public datasets as the development clips".
  - Seal `independent` from parameter sweeps as well as screens, and choose the buffer default on pool clips.
  - Draw a new sealed set for the next milestone, and say how many times each held-out set has been measured.

## 9. The "rate-invalid" column uses a rule that suits our buffer

- **Claim:** the chart tables' notes column ("58 rate-invalid pts" for x264, none for most of ours).
- **How it is made:** a point is invalid when its one-second leaky bucket peaks above 1.5 buffers
  (`eval/nceval/jreal.py:187`) or its average rate is more than 10 % plus buffer / window over target
  (`eval/nceval/realuse.py:320-331`). For ours only, the outstanding key loan is deducted first
  (`eval/nceval/cbr.py:830-841`).
- **Why it matters:** our encoder stops at 1.4 buffers by design, just under the 1.5 threshold. So ours shows 0–5
  flagged points while 10–16 % of its points pass one buffer (the published buffer statement).
- **Verdict:** misleading. The rule is on the method page; the column reads as "ours holds its rate, theirs do
  not".
- **Fix:** one rule for all, with no loan deduction: report "points over 1.0 buffer" and "points more than 10 %
  over target" as two counts.

## 10. ffmpeg.wasm and FFmpeg decoder comparisons charge the reference its start-up

- **Claim:** README line 40: "On the H.264, HEVC and VP9 streams we timed, they needed less time per frame than
  ffmpeg.wasm"; site ratios 0.33–0.59 (`web/results/data/wasm-yardstick.json`).
- **How it is made** (`tools/web/ffmpeg_wasm_yardstick.py`, `tools/web/ffmpeg-wasm-runner.cjs:22-35`,
  `eval/nceval/site.py:1033-1037`):
  - ffmpeg.wasm is `@ffmpeg/core` 0.12.10, built without SIMD; ours is the relaxed-SIMD module. The site says
    so; the README sentence does not.
  - Their time is module instantiation plus the whole `exec` (probe, demux, command line) over 150 frames. Ours
    is the sum of the `decode()` calls (`bench-runner.mjs:99-101`).
- **Size:** `createFFmpegCore` alone took 91–115 ms on the PC (three runs, unpinned) against their totals of
  508–2296 ms. Taking 100 ms off moves H.264 720p from 0.43 to about 0.53 and Axis HEVC from 0.52 to about 0.65.
  The record does not store start-up, so the exact figure on the timing machine is not known.
- **A published text contradicts the impression:** `docs/results/competitor-yardsticks.md:224-225` says we "beat
  ffmpeg.wasm, which has no SIMD, by only 12–30 %" and trail an estimated SIMD port.
- **Browser encode chart:** ours is given `--peak-kbps 1200` (twice the target) where x264 gets
  `-maxrate 600k -bufsize 600k` (`ffmpeg_wasm_yardstick.py:59-78`); one clip; the quality axis is the native
  chart's BD-J, not these encodes (`site.py:1024,1052`).
- **Native decoders:** `tools/dev/decoder_speed.py:49-51,94-103` takes whole-process wall time, and FFmpeg writes
  raw video to `/dev/null` through a muxer where dav1d and vpxdec discard frames. On a one-frame 720p H.264
  stream FFmpeg's process costs about 6.6 ms more than ours (unpinned 5950X): about 6 % of the 720p row, so
  H.264 on x86 moves from 0.90 to about 0.95. H.264 is level with FFmpeg, not ahead. The HEVC and Jetson margins
  are far larger than this bias.
- **Verdict:** misleading; the direction holds for HEVC and VP9.
- **Fix:** record and subtract start-up (or time ours end to end); "ffmpeg.wasm (official build, no SIMD)" in the
  README; give both encoders the same peak; drop the quality axis from the browser encode chart.

## 11. The buffer statement publishes the smaller of two measured costs

- **Claim** (`eval/nceval/buffer_note.py`, on every CBR chart and in the README): holding our stream inside one
  buffer "as the references' is" costs 3–5.5 %.
- **What was measured** (`docs/analysis/frame-allocation-2026-11-21.md` sections 5–7):
  - Never passing one buffer, with a first key of about 0.3 buffer: +3 to +5.5 %.
  - Also matching the references' key size (0.10–0.24 buffer against our 0.33–0.45) and holding each one-second
    window to the target: +10 to +13 %. With that, AV1 against SVT-AV1 M12 moves from −31 % to −25 % and H.264
    `medium` from level to about 10 % behind.
  - Commit 0b6cc062a removed the second set and the key-size and key-second figures from the statement.
- **Peak rate:** our encoder's default peak is twice the target; x264, x265 and NVENC are given a peak equal to
  the target (`--vbv-maxrate`, `-maxrate`; section 5's table of what each encoder is given). The statement does
  not mention it.
- **Assessment:**
  - The one-buffer rule is the constraint the references are actually given, so 3–5.5 % is the defensible
    number. Their small keys are their own choice.
  - The removed facts are still what a streaming engineer asks first: our key second runs at 1.3–1.4x the target,
    theirs at 0.9–1.05x.
  - The 3–5.5 % rests on 24 clips at 150 frames and 12 at 600, one run, no interval, and the published headline
    numbers are still the loose-buffer ones.
- **Verdict:** misleading by omission; the statement itself is accurate.
- **Fix:** restore the key-size and key-second sentence; re-measure the charts with the hard buffer before the
  next release rather than after.

## 12. Build asymmetry

- **Ours:** PGO, fat LTO, one codegen unit, `-C target-cpu=x86-64-v3` (`Cargo.toml:71-73`, `.cargo/config.toml`,
  `docs/results/release.json`).
- **References:** stock CMake or configure release builds (`tools/anchors/build_anchors.sh:171-275`).
- **Graviton:** the reference binaries are the Jetson's (`tools/dev/aws_timing.py:92-95`), with x265 built
  `-DENABLE_SVE=OFF -DENABLE_SVE2=OFF` (`build_anchors.sh:242-246`). The effect was not measured.
- **Users:** `cargo install nanocodec-cli` gives neither PGO nor `x86-64-v3` (the README notes the second).
- **PGO training:** on four Xiph clips, not the timing clips (`tools/release/release.py:236-242`). Sound.
- **Size:** the project measured PGO at 1.4–3.2 % (`docs/PROCEDURES.md:381-384`; CHANGELOG: 4–7 % for 0.6.0).
- **Not verified:** whether each reference dispatches AVX-512 on Zen 5.
- **Fix:** print the reference build flags on the method page; publish the plain-build fps beside the PGO one;
  rebuild x265 with SVE2 for Graviton.

## 13. Conformance: denominators, stale counts, contradictory texts

- **VVC count:** `README.md:31` says 226/226; `docs/FAQ.md:211-214` and `docs/results/README.md:66` say 224/226.
- **VVC denominator:** the suite has 283 streams; 226 are the Main 10 / Main 10 Still Picture 4:2:0 and 4:0:0
  ones. `HRD_B_2` and `VPS_C_2` are left out because VTM fails on them, and `STILL_B_1` passes against VTM's
  output, which disagrees with the suite's own MD5 (`docs/handoff/vvc-foreign-corpus.md:21-25`). "226/226 of the
  official conformance streams" does not say this.
- **H.264:** "110 of the 190 ITU-T H.264.1 streams"; 190 is AVCv1, FRExt and High 10 only: 4:2:2, 4:4:4, MVC and
  SVC streams are not fetched (`tools/dev/conformance_fetch.py:46-50`). `docs/FAQ.md:202` calls it "the whole"
  suite.
- **VP9:** the 64 resize vectors are checked against vpxdec alone.
- **Browser bit depth:** `README.md:25-27` says 8- and 10-bit in the browser (`crates/nc-web/src/lib.rs:923-936`
  backs it); `docs/FAQ.md:196-198` and `docs/results/README.md:56-57` say 8-bit only.
- **No stored evidence at 0.6.0:** only the H.264 suite is in `tools/ci/check.sh:155`; the others are manual
  release steps and no result file is committed. They were not re-run here.
- **"Every shipped preset's streams are checked"** (`README.md:56`): the check codes short purpose-made streams
  per preset (`tools/ci/landing_conformance.py:17-62`), not the chart streams.
- **Sound:** HEVC 147/147 and AV1 242/242 match the manifests.
- **Fix:** generate the counts from one file as the chart figures are; write the denominators out ("226 of the
  suite's 283: Main 10 4:2:0 / 4:0:0"); commit each suite's `--json` result per release.

## 14. Fuzzing claim predates most decoder code

- **Claim:** `docs/FAQ.md:230-236`: every decoder "had 2.1 to 3.8 worker-hours"; `docs/results/fuzzing.md`
  (rounds of 2026-11-13 and 11-14).
- **Since then** (`git log -- fuzz`): HEVC Main 10, AV1 10-bit and film grain, VP9 profile 2, H.264 High 10, the
  general VVC decoder (9 → 226 streams) and the general AV2 decoder. `fuzzing.md:104-105` lists "10-bit and other
  profiles" as not tested. The README now advertises decoding other encoders' 8- and 10-bit streams.
- **Verdict:** misleading by staleness.
- **Fix:** re-run at 0.6.0 with 10-bit and foreign seeds, or date the claim in the FAQ.

## 15. Stale tables beside current ones

- **NVENC:** `docs/results/README.md` gives H.264 P4 −18.7 %, HEVC −43.0 %, AV1 −36.4 %; the published
  `bd-vs-speed-hardware.csv` gives −13.8, −40.5 and −33.3.
- **Decoder ratios:** `docs/results/README.md:59-73` (2026-11-13, a loaded 5950X) against the site (2026-11-19,
  c8a): VP9 against vpxdec 1.32–1.41 against 1.03–1.09; VVC against VVdeC 0.39–0.47 against 0.57.
- **Releases:** every decoder, WASM, SVC and rate-control figure on the site is release 0.4.0; captions say so.
- **`docs/results/not-measured.json`:** marks nine rungs "the next release measures it" on views whose data
  contains them.
- **History page:** text and links say "to 2026-11-20"; the data runs to 2026-11-21.
- **`web/results/yardsticks.html:38,54`:** "timed in the same rounds", "reserved core of a shared PC"; the data is
  from rented machines in separate sessions.
- **FAQ:** preset names and figures of 2026-11-15 (`docs/FAQ.md:75-79`).
- **Fix:** delete the tables in `docs/results/README.md` that the site regenerates, and link to the site.

## 16. Smaller items

- **Selected pairs** on the chart pages compare our H.264 with the default-tuned x264 `medium` (−34.2 %), not the
  tuned zero line the headline uses (−25.7 %).
- **The Pareto front** leaves out any point whose fps is its twin's (`chart.pareto`), which today is every
  metric-tuned reference.
- **Bars** are per-clip interquartile ranges, not confidence intervals; the fresh chart publishes no interval for
  its headline numbers.
- **CBR against fixed QP:** our AV1 `medium` is −22.7 % against x264 `medium` at fixed QP on the CTC clips
  (weighted PSNR) and −43 % in CBR on the fresh clips. Much of the CBR lead is what the references' real-time
  rate control costs them at these rates (`web/results/rate-control.html` shows this; the headline does not).
- **Window:** 150 frames with one key frame favours our key investment; the README says so ("over 20 s our
  savings shrink by several points"). Sound disclosure.
- **Wording:** no superlatives and no yes/no real-time claims were found in the README, the FAQ or the results
  pages.

## What was checked and found sound

- Clip selection for the fresh set is a seeded, rule-based draw with no source shared with any other set.
- BD-rate: PCHIP over the overlap, no extrapolation, one clip set for every point on the CBR charts.
- x264 and x265 are shown at their metric tuning, and the zero line is the tuned x264.
- SVT-AV1 runs flat P, which scores better for it than its default hierarchy (+3.5 / +4.5 % with the hierarchy);
  one thread scores better for x264 and x265 than four (2026-11-13 audit).
- The quality and speed builds of 0.6.0 are the same encoder.
- PGO was not trained on the timing clips.

## Coverage

What was read for this audit, so that a claim missing above is known to be unexamined, not cleared.

| Published item | Covered in |
|---|---|
| README headline (bits, speed, where behind), "Read these numbers with their limits" | 1, 2, 7, 8, 11 |
| README feature list: conformance counts, ffmpeg.wasm sentence, "every shipped preset" | 10, 13 |
| Overview and By codec pages: fresh, all held-out, development and VC-1K CBR charts | 1, 2, 3, 6, 7, 8, 9, 16 |
| Speed page: encode on four machines, browser, decoders, NVENC | 1, 6, 10, 12, 15 |
| Fixed QP page and the three Fixed QP views | 5 |
| Rate control page | 15, 16 (release 0.4.0 data; clip counts differ between its columns) |
| SVC page | 4 |
| History page | 6, 15 |
| Method page: score, clips, settings, buffer, window length | 2, 3, 7, 9, 11 |
| Yardsticks page | 10, 15 |
| Clips page | 8 |
| `docs/results/README.md`, `docs/FAQ.md` | 13, 14, 15 |

**Not examined:** time to first packet and memory figures (`docs/results/competitor-yardsticks.md`), the colour
conversion comparison with libyuv, the demo page's on-screen numbers, the usage docs under `web/docs/`,
`docs/LEGAL.md`, the temporal-layer, intra-refresh and loss-resilience claims, and the window-length table beyond
its disclosure.

## Fix status

Updated as fixes land (agent `audit-text` keeps this table). "Text" is the wording of the README, FAQ,
`docs/results/README.md` and the results pages; "data" is a re-measure or a change to how a number is computed.
An owner named by its worktree (`~/nanocodec-data/worktrees/<name>`) has the item in progress; a hash is landed
on `variance-integration`.

| # | Finding | Text | Data / computation |
|---|---|---|---|
| 1 | Speed and quality on different clips | 400b8fbd1 (headline names the three timing clips and rung; README feature line corrected), 4d2034845 (method, speed pages) | `chart-cbr` (main chart), `chart-retime` (held-out, VC-1K, Arm, Jetson, history): time every quality encode; landed 305114ca3, b93a7fc87, 991b973ec; re-time pending |
| 2 | Low-rate headline | 400b8fbd1 (headline states the ladder, low rungs and BD-J definition; README limits "Rates") | 16de42521 (plain BD-J, LOW and UPPER regimes, PSNR-Y and MS-SSIM columns; takes effect when the charts are re-rendered); headline BD-J wording follows the render (5350cfbcb, 5a3ef330b) |
| 3 | Reference settings | 400b8fbd1 (README, FAQ), 4d2034845, c07e4863d (method page) | 98069a8b3, 2b8b831d6 (openh264 at two settings, libvpx / libaom `--aq-mode=3`); measurement: `chart-cbr`, `chart-retime` |
| 4 | SVC split | 06eec4c9e (x86 axis says it is a 5950X ratio times EPYC x264 fps) | `svc-fair` |
| 5 | Fixed QP | ba4c6ed8e (which CTC sequences, not its configuration or anchor; overview card) | feb163654 (constant-QP row, PSNR-YUV first, x264 `--tune psnr` zero, one clip set per view); re-measure by `fixedqp-retime`: 38f1e6b5c (lens data against x264 `medium --tune zerolatency,psnr`, Constant QP column, references on their opt builds), 35c2dc358, 01abe162b, bfc8a2a60, a74e15c48, 464cce243; 2a7f30938 (lens re-timed on the references' opt builds), 4b92c21fc (samples re-measured, one clip set per view) |
| 6 | x86 rescale, Zen 3, CPU label | 400b8fbd1 (headline states the rescale), 4d2034845 (method, speed, yardsticks pages) | `audit-method` (machine label from the timing file); headline speed sentence follows the record (2c24c7b1e); one-session re-time: `chart-cbr`, `chart-retime` |
| 7 | Metrics disclosure | 400b8fbd1 (README "Scoring", FAQ: VMAF protocol, our PSNR, the stale standard-decoder sentence), 4d2034845 (method page) | 16de42521 (PSNR-Y and MS-SSIM columns), 89b58631d (metric validation: VMAFx against CPU libvmaf, encoder PSNR against libvmaf); text cb56fc120 (which points are scored on a decoder's output); unpinned FFmpeg for source decode: follow-up of `audit-metrics` |
| 8 | Held-out wording | 400b8fbd1 (README and FAQ: same public datasets; re-measured at seven or more builds) | 3a2194039, e84855b2c, 7faf4d7ce (sealed set `heldout_2026-11-23`, fresh by source collection, 100 clips, runtime seal and CI check; now the main chart set for the re-measure); choosing the buffer default on pool clips: open |
| 9 | Rate-invalid rule | 400b8fbd1 (FAQ states the loan deduction) | 16de42521 (one rule for every encoder, nothing deducted: over one buffer, over target; on re-render) |
| 10 | Decoder yardsticks | a6c485c90 (README, FAQ, speed and yardsticks pages; ffmpeg.wasm named exactly) | 1da4349e2, e67bb7a8e, a6c4aecca (decode alone on both sides: native in process against FFmpeg n8.0.1 and dav1d on c8a.large; browser rows from ffmpeg.wasm's per-call decode times). 21abcbef6, a4a0a95b0 (site native table from the in-process record, whole-process figures removed; browser encode re-measured with start-up excluded on both sides and the same peak; `avbench` in every anchor set, Jetson included); page text in this row's next commit. a7f6abbf1 (libvpx, VVdeC and AVM in process; browser encode's borrowed quality axis dropped: speed at one rate only). 50894a361 (the Jetson in process, one Cortex-A78AE core at a fixed clock), a7639e270 (AV2: AVM on the Jetson, dav2d with its assembly on both machines) |
| 11 | Buffer statement | 400b8fbd1 (caption "looser than the references'" replaced by the measured share over one buffer) | 16de42521 (`buffer_note.facts_table`, intervals; filled when buffer use is re-run on the re-measured charts) |
| 12 | Build asymmetry | 400b8fbd1 (README), 4d2034845 (method page) | `audit-method` with `audit-builds` (reference flags, plain-build fps) |
| 13 | Conformance | 9423a197a (README and FAQ counts from the record, denominators, reference decoders, browser results) | 55092b606 (every native and browser suite at 98069a8b3, `docs/results/conformance-2026-11-23/summary.json`); browser VVC 221/221 after the runner fix d010259d0; AV2's 30 foreign streams 30/30 and VP9 305/305 after 71f6d92b2 |
| 14 | Fuzzing | 1e95779dd, 2188278d4 (FAQ: both rounds dated, round 3's numbers, the fixes and the WASM replay) | ac9ed8f19 (round 3 at 98069a8b3: six decoders × 25 min × 8 workers, 10-bit and foreign seeds, about 1.3 million executions; one AV2 crash in CCSO; both round-3 findings fixed in 71f6d92b2, corpora replayed in WASM without a trap) |
| 15 | Stale tables | 400b8fbd1 (`docs/results/README.md` tables removed, FAQ figures, history dates, yardsticks page); `not-measured.json` already cleared in 5f2f2555a | — |
| 16 | Smaller items | 06eec4c9e (selected pairs say their references are default-tuned; bars labelled interquartile, not confidence intervals) | tuned twins timed (Pareto front): `chart-cbr` (main), `chart-retime` (others) |

Beyond the list: the internal decision tools' timing consistency (`measure-consistency`): 1f425f5ef, a4e89c4a2,
7c5b79d94, 54b292c38.

## Damage assessment

Each published claim as it stood at 974ad4629 (release 0.6.0), and the corrected value where a re-measure has
landed. "Pending" means the re-measure is still running (owner in "Fix status"). A corrected value can differ
from the old one through the method change and through the build: say which where both changed.

### Decoder speed (finding 10)

Ours ÷ reference, time per frame, range over the streams of each codec; below 1 means ours took less time.
Published: whole processes, start-up and output included, release 0.4.0 PGO build (c9cff89f), best of 7 (x86) or
5 (Jetson) rounds, 2026-11-19. Corrected: decode alone in process on both sides, median of 15 passes,
2026-11-23 (`docs/results/decoder-speed-2026-11-23/`): the method change alone (0.4.0's PGO build plus only
the bench CLI path, x86) and the shipped decoder (0.6.0's PGO build 207745d7 plus only the bench CLI path; its
decoded frames equal the shipped binary's; PGO regenerated the release way, as 0.6.0's profile data was not
kept). References in process: FFmpeg n8.0.1, dav1d 1.5.4 through libavcodec, libvpx v1.16.0, VVdeC d261469,
AVM v1.0.0, dav2d 237229e with its assembly.

| Codec, reference | x86 published | x86, method only (0.4.0 PGO) | x86, 0.6.0 PGO | Jetson published | Jetson, 0.6.0 PGO |
|---|---|---|---|---|---|
| H.264, FFmpeg | 0.90–1.03 | 1.06–1.14 | 1.10–1.18 | 0.76–0.83 | 0.92–0.98 |
| HEVC, FFmpeg | 0.54–0.86 | 0.66–0.94 | 0.64–0.93 | 0.68–0.90 | 0.81–0.97 |
| VP9, vpxdec / libvpx | 1.03–1.09 | 1.01–1.08 | 1.00–1.04 | 1.02–1.04 | 1.01–1.03 |
| VP9, FFmpeg | 0.83–0.93 | — | 0.91–1.01 | 0.89–0.96 | 1.05–1.07 |
| VVC, VVdeC | 0.57 | 0.57–0.58 | 0.51–0.55 | 0.51–0.58 | 0.50–0.58 |
| VVC, FFmpeg | not measured | — | 0.41 | not measured | 0.33–0.35 |
| AV1, dav1d | 1.15–1.56 | 1.14–1.59 | 1.16–1.43 | 0.96–1.26 | 0.94–1.16 |
| AV2, avmdec / AVM | 0.43–0.47 | 0.42–0.46 | 0.40–0.41 | 0.28–0.29 | 0.28 |
| AV2, dav2d | not measured | — | 1.41–1.44 | not measured | 0.70–0.86 |

- **Damage:** the H.264 claim reverses, and the method change alone does it (0.90–1.03 → 1.06–1.14 at the same
  0.4.0 build). Ours was published as level with or faster than FFmpeg on x86 and 17–24 % faster on the Jetson;
  for the shipped 0.6.0 decoder, FFmpeg's H.264 needs less time than ours on all four x86 streams (ours
  1.10–1.18x) and ours is 2–8 % faster on the Jetson. HEVC against FFmpeg moves 7–19 points towards FFmpeg on x86; on
  the Jetson VP9 goes from faster than FFmpeg to slower. VVC, AV1 and the libvpx comparison barely move. AV2 was
  published only against AVM, a research decoder not built for speed; against dav2d ours takes 1.41–1.44x its
  time on x86 and 0.70–0.86x on the Jetson.

### Browser decode and encode (finding 10)

ffmpeg.wasm 0.12.10. Old: its module start-up and the whole command counted, ours the decode calls (c9cff89f,
2026-11-19). New: decode or encode calls alone on both sides (1da4349e, 2026-11-23, c8a.large, node).

| Stream (decode) | Old ratio | New ratio |
|---|---:|---:|
| H.264 ours 720p crowd 600k | 0.43 | 0.83 |
| H.264 x264 CABAC 1080p 3M | 0.59 | 0.75 |
| HEVC Axis 720p camera | 0.52 | 0.89 |
| HEVC ours 1080p 3M | 0.44 | 0.49 |
| HEVC ours 720p crowd 600k | 0.43 | 0.61 |
| VP9 vpxenc 1080p 3M | 0.33 | 0.37 |

- **Damage:** ours still takes less time on every stream, but the margin shrinks from 41–67 % to 11–63 %; the
  H.264 and Axis HEVC margins shrink most. The README sentence's direction holds; "no SIMD" was inexact
  (ffmpeg.wasm has compiler SIMD128, without FFmpeg's hand-written kernels).
- **Browser encode:** ms per frame old → new: ours H.264 fast 5.35 → 7.15, HEVC fast 7.81 → 9.33, AV1 fast
  9.24 → 9.53, H.264 medium 9.81 → 9.61, VP9 13.01 → 14.07; x264 ultrafast 12.95 → 11.80, veryfast 31.3 →
  29.96. Old: x264 given a peak equal to the target, ours twice it, and ffmpeg.wasm's start-up counted; new:
  the same target, peak and buffer on both sides, start-up excluded on both. The old quality axis (BD-J of
  the native chart, other clips and build) is removed: the browser encodes have no quality figure.
- **Damage:** our fastest browser encode (H.264 fast) went from 2.4x to 1.7x x264 ultrafast's frame rate;
  H.264 medium from 1.32x to 1.23x.

### Conformance (finding 13)

| Claim | Published | Measured 2026-11-23 at 98069a8b3 |
|---|---|---|
| VVC | README 226/226; FAQ and results README 224/226 ("the 2 that fail use field output") | 226/226 of the suite's 283 (exclusions named) |
| H.264 | 110 of 190 ("the whole" suite in the FAQ) | 110/190 of the AVCv1, FRExt and High 10 sets fetched; 4:2:2, 4:4:4, MVC and SVC not fetched |
| HEVC, AV1, VP9 | 147/147, 242/242, 305/305 | the same |
| Browser bit depth | README: 8- and 10-bit; FAQ and results README: 8-bit only | 10-bit passes in the browser for every codec (HEVC 11, VVC 214, AV1 65, VP9 2, H.264 7 streams) |
| Browser counts | not published | HEVC 147/147, AV1 240/240, VP9 305/305, VVC 221/221, H.264 108/113 |
| AV2 foreign streams | 30/30 | 30/30 against AVM's frame hashes (after 71f6d92b2); VP9 305/305 after its area cap |

- **Damage:** none to the counts we claimed in the README; the FAQ and results README copies were wrong
  (224/226, 8-bit only in the browser) and the denominators were not stated.

### Fuzzing (finding 14)

- **Published:** every decoder had 2.1–3.8 worker-hours in two rounds (2026-11-13/14), about 2.75 million
  executions, seven issues found and fixed, nothing in VP9 or AV2.
- **Measured 2026-11-23 at 98069a8b3:** six decoders × 25 min × 8 workers with 10-bit and foreign seeds, about
  1.3 million executions: no finding in H.264, HEVC, VVC, AV1 or VP9 (VP9: five timeouts on very large declared
  pictures); one AV2 crash (CCSO band-only `max_band_log2` 7, `crates/nc-av2/src/ccso.rs:131`, panicked in
  release builds too). Both fixed in 71f6d92b2 (AV2: a clean error beyond 64 bands, as AVM; VP9: area cap at
  level 6.2), with the inputs as regression tests; every round-3 corpus input replays in WASM without a trap.
- **Damage:** "nothing was found in AV2" did not hold for 0.6.0's decoder (one crash, now fixed); the claim
  predated the 10-bit and general VVC / AV2 decoders.

### Encoding charts (findings 1–9, 11, 12, 16)

| Claim (0.6.0) | Published | Corrected |
|---|---|---|
| Headline bits, `medium` vs x264 `medium --tune psnr`, fresh / all held-out | 25.7–50.6 % / 22.5–47.6 % fewer bits | pending (plain BD-J, regimes, reference settings, new sealed set) |
| Headline speed, Jetson / Zen 5 (x264) | 25–66 / 102–198 fps (27 / 113) | pending (timed on the quality encodes) |
| openh264 pairs (H.264 `medium` vs openh264, fresh) | −36.0 % | pending; openh264 with CABAC needed about 17 % fewer bits on five clips (finding 3), so the gap shrinks by about that much |
| Fixed QP lens, weighted PSNR / J (HEVC, AV1 `medium`; x265 `medium`; SVT-AV1 M8) | vs default-tuned x264: HEVC −17.4 / −29.7, AV1 −22.7 / −29.4, x265 −19.9 / −20.9, M8 −30.3 / −30.8 | vs x264 `--tune psnr` (38f1e6b5c, profile column): HEVC −12.3 / −27.5, AV1 −17.8 / −26.9, x265 −15.9 / −17.5, M8 −26.2 / −28.1 |
| Fixed QP samples, BD-J (AV1, HEVC `medium`; x265 `medium`; SVT-AV1 M8) | vs default-tuned x264, clip sets differing by point: fresh −30.1, −32.2, −23.8, −25.5; VC-1K −48.9, −43.3, −48.5, −49.2 | vs x264 `--tune psnr`, one clip set per view (4b92c21fc; fresh 6 of 12 clips, VC-1K 10 of 12): fresh −27.2, −28.8, −23.3, −25.1; VC-1K −46.7, −41.6, −46.4, −46.8 |
| SVC, AV1 L2T1 vs simulcast | −48.0 % vs −37.8 % | pending (`svc-fair`) |
| Rate-invalid counts | ours 0–5 points | pending (one rule for all, on re-render) |
| Buffer statement | 3–5.5 % cost | pending (buffer facts on the re-measured charts) |
| NVENC table (results README) | H.264 −18.7, HEVC −43.0, AV1 −36.4 % | the published CSV's −13.8, −40.5, −33.3 %; the stale table is removed |

- **Damage so far:** on the Fixed QP lens, against the PSNR-tuned zero line, every saving shown is 4–5 points smaller
  in weighted PSNR, and x265 `medium` stays ahead of our HEVC `medium` on that metric (−15.9 against −12.3).
  The CBR headline's damage is not known until the re-measure lands.

### Main chart, 0.6.0 as published and re-measured (findings 1, 2, 3, 12, 16)

Re-measured 2026-11-23 on the main chart's 11 VAL clips (not a published chart view; the
held-out views follow). Quality: every encode of the chart (the real-use ladder, the upper
rungs, the low rungs on four clips), plain full-range BD-J against default-tuned x264
`medium`, references on the timed LTO/PGO builds with 2b8b831d6's settings. Speed: c8a.large,
the 0.6.0 PGO builds (207745d7; H.264 Default and Veryfast on their own quality builds'
PGO binaries c9cff89f and 20d31869), every encode its quality point scores, paired rounds on
one pinned core. Both columns change through the method (quality) and the timing (speed); the
quality change is mostly the plain BD (no double-weighted low half) and the upper rungs.
Records: ~/nanocodec-data/charts/timing-cbr/main-060-vs-head-2026-11-23/ (tables.md has
every point).

| Point (0.6.0) | Published BD-J / fps | Corrected BD-J / fps |
|---|---:|---:|
| VVC Veryslow | −57.9 / 30.9 | −52.6 / 38.7 |
| VVC | −54.4 / 101.7 | −47.7 / 111.2 |
| AV2 | −52.9 / 150.7 | −46.3 / 148.2 |
| AV1 | −50.1 / 179.0 | −43.1 / 180.8 |
| AV1 Fast | −48.4 / 252.9 | −40.7 / 255.0 |
| HEVC | −49.2 / 167.9 | −42.0 / 175.1 |
| HEVC Fast | −45.2 / 281.8 | −37.4 / 279.8 |
| AV1 Veryfast | −40.1 / 419.4 | −27.9 / 402.6 |
| VP9 Veryfast | −33.4 / 393.8 | −21.7 / 429.3 |
| H.264 | −23.1 / 198.4 | −19.7 / 217.3 |
| H.264 Veryfast | −5.3 / 510.6 | +7.1 / 541.9 |
| x264 `medium` (zero line) | 0 / 112.8 | 0 / 100.3 |
| x264 `ultrafast` | +48.6 / 1017.0 | +64.3 / 728.5 |
| SVT-AV1 M13 | −28.8 / 445.4 | −18.9 / 446.1 |
| libvpx VP9 cpu7 (AQ 3 now) | +1.6 / 122.7 | −9.0 / 186.3 |

- **Our savings are 3–12 points smaller** on every rung; the fast rungs lose most (AV1 Veryfast
  −40.1 → −27.9, VP9 Veryfast −33.4 → −21.7, H.264 Veryfast −5.3 → +7.1). The references
  shrink too (SVT-AV1 M13 −28.8 → −18.9), so the ranking moves less than the numbers.
- **Speed:** most of our fast and medium rungs are within ±10 % of the published fps; the
  slow rungs gain 12–28 %. x264 `ultrafast` and `veryfast` lose 25–28 % and x265 gains 1–24 %.
  Mode (fixed QP against CBR), clips (three training clips against every quality encode) and
  reference builds all changed at once; how much each contributes is not measured.
- **H.264 Default and Veryfast (0.6.0) were timed on another build than their quality**
  (quality c9cff89f / 20d31869, timing 207745d7, which codes other CBR bytes, e.g. 92,251
  against 92,168 on one encode). The corrected values time each on its own quality build;
  since 644e71b24 a timed encode whose stream hash is not its quality stream's is refused.
- **Head (7c1c5334f) against 0.6.0, both corrected:** the front is +0.38 BD-J points
  (worse) at equal speed and 0.98× the speed at equal BD-J. By codec (J points, speed):
  H.264 −0.81, 1.038×; HEVC +0.12, 0.985×; AV1 +0.45, 0.958×; VP9 +0.37, 0.967×;
  AV2 −0.07, 1.011×; VVC −0.14, 1.019×.
