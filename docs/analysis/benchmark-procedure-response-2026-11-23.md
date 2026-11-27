# Response to the hostile review of BP-1 (2026-11-23)

> **Superseded by BP-3.10** ([benchmark-procedure.md](../benchmark-procedure.md), 2026-11-24). This dated response is kept as the record of its review; where it differs from the current procedure, the procedure governs.

This answers every finding of
[benchmark-procedure-review-2026-11-23.md](benchmark-procedure-review-2026-11-23.md) ("the
review"). Each finding is accepted, rejected with evidence, or measured. The procedure is now
draft BP-2 ([docs/benchmark-procedure.md](../benchmark-procedure.md)). Its per-encoder fairness
record is [docs/benchmark-fairness.toml](../benchmark-fairness.toml), rendered into the
procedure's section 17.

The coordinator decided the open questions under the owner's delegation; they are listed at the
end.

## How the measurements were made

All numbers are measured today at head 0d9f32bc (our release build) with the timed reference
builds (`anchors/x86_64-opt`). There was no compute on the PC except the analysis.

- **Quality and rate runs:** 20 clips, on c7a.24xlarge spot boxes, packed, which is fine for
  streams and scores.
  - The clips: the 4 main-chart 720p clips, and 16 held-out clips from 240p to 720p, two from
    each of 8 classes.
  - The ladder: 40, 75, 150, 300, 600 and 1200 kbps at 720p30, scaled per clip by pixel
    rate^0.75. 150 frames, one key frame.
  - The encoders:
    - ours: H.264 medium/veryfast, HEVC, AV1 medium/veryfast, VP9, VVC and AV2 medium;
    - x264 medium (and `~psnr`) and veryfast; x265 medium and ultrafast; openh264; SVT-AV1
      M12/M13; libvpx rt 7/9; libaom rt 8/10;
    - libvpx's and libaom's own real-time example programs.
  - Our variants:
    - `def` (default);
    - `nohold` (`hold.enable=0`);
    - `nola` (`rate.first_key_lookahead_off=1`);
    - `strict` (`--strict-buffer`);
    - `fair` (all three plus `rate.key_cap_buffer=0.6`);
    - `bp2` (`fair` plus `--peak-bitrate` = target; its streams equal `fair`'s);
    - `nores` (AV1 `resolution.adaptive=0`).
  - Reference variants: `+F06` (x264/x265 `--vbv-init 0.6`) and `+resize` (dynamic resize).
- **Scoring:** every stream decoded by ffmpeg 8 (Ubuntu 26.04's), or our decoder for AV2. Held
  frames were refilled with the previous picture. libvmaf 3.2.1 VMAF v1 + PSNR. BD is PCHIP
  with at least 4 points, plain mean over clips.
  - Size-changed frames were rescaled by ffmpeg's automatic (bicubic) scaler. BP-2 pins the
    player model instead, so the resize figures below are indicative.
  - The crowd clip lost a few points to a full scratch disk on one box (19 of 6,782); listed
    as missing.
- **Timing:** one c8a.large, vCPU 1, 3 clips at 720p, 600 kbps, 5 rounds:
  - x265 with and without `--asm avx512`;
  - the RT examples against the chart commands;
  - plain builds of every reference and of our 0.6.0 against the timed builds.
- **Spend:** this response cost $0.35 of AWS (sessions in the ledger, `Purpose=bench-method`). The
  bench-method total is $1.98. No instance was left running.
- **Records:** `~/nanocodec-data/bench-method-records/` (`all-resp.jsonl`, `resp-analysis.txt`,
  `tools/`), kept by a retention entry.

## The BLOCKERs

### B1. Frame hold: accepted; all-skip frames for ours, decoded count = input count for all

**How often hold fires at BP settings** (`def`, 20 clips x 6 rungs per family):

| Family | Streams with holds | Held frames, share of all frames | Most in one stream |
|---|---:|---:|---:|
| H.264 medium | 8.2 % | 0.29 % | 8 |
| H.264 veryfast | 5.8 % | 0.27 % | 10 |
| HEVC medium | 9.8 % | 1.75 % | 133 |
| AV1 medium | 8.2 % | 0.69 % | 44 |
| AV1 veryfast | 2.5 % | 0.12 % | 11 |
| VP9 medium | 9.0 % | 0.31 % | 12 |
| VVC medium | 9.1 % | 0.91 % | 117 |
| AV2 medium | 8.2 % | 1.12 % | 91 |

- **By class:** static 1.48 % and videocall 1.26 % of frames; every other class 0-0.07 %.
- **By rung:** spread from 0 to 2.3 %.
- **On a still picture repeated 150 times:** HEVC held 133 frames and AV2 91.
- **Rate-control dropping** (`drop_due`) is off at BP settings: it needs `--frame-drop`, which
  is not passed (`crates/nc-cli/src/rate_control.rs:75`). No `Dropped` frame was seen in any
  stream.
- **Cost of turning hold off** (`nohold` against `def`, BD-rate): -0.28 to +0.54 % PSNR-YUV and
  -0.25 to +0.51 % VMAF.

**What an all-skip frame costs** (the owner's proposal). These are the smallest coded P frames
on the repeated still picture: each encoder's cheapest "nothing changed" frame.

| Encoder | 720p | 1080p |
|---|---:|---:|
| openh264 | 14 B | 14 B |
| SVT-AV1 M12 / M13 | 26 / 27 B | 27 / 23 B |
| x265 ultrafast | 31 B | 51 B |
| x264 medium `~psnr` / veryfast | 40 / 39 B | 74 / 69 B |
| ours: H.264 / HEVC / VVC / AV1 / VP9 / AV2 (hold off) | 35 / 23 / 30 / 29 / 20 / 17 B | 67 / 34 / 40 / 64 / 28 / 77 B |
| libaom rt 8 / 10, libvpx rt 9 | 58 / 66 / 92 B | 140 / 123 / 1,095 B |

A skip frame costs tens of bytes in every codec. Coding held frames as all-skip frames:
- removes the asymmetry ("ours may send nothing");
- makes the decoded count equal the input for every encoder;
- costs ours about 30 B per held frame, under 0.1 % of a typical stream at the measured hold
  rates.

The analysis the encoder skips for a held frame is a legitimate early decision, like every
encoder's skip decision, and the fairness record says so.

**Resolution (decision 3):**
- Ours codes a held frame as a conformant all-skip frame: engine task T1, specified in BP-2
  section 18 and assigned by the coordinator. Until it lands, BP-2 runs `hold.enable=0`.
- V9 requires decoded frames = input frames for every encoder.
- BP-1's sentence "ours never drops in these modes" is removed.

### B2. Start-up lookahead: accepted; off in benchmarks (decision 1); cost measured

`rate.first_key_lookahead_off=1` against the default, BD-rate (positive = ours worse):

| Family | PSNR-YUV | VMAF |
|---|---:|---:|
| H.264 medium | +2.39 % | +2.18 % |
| H.264 veryfast | +3.01 % | +1.98 % |
| HEVC medium | +1.44 % | +0.46 % |
| AV1 medium | +2.27 % | +1.94 % |
| AV1 veryfast | +1.57 % | +0.79 % |
| VP9 medium | +1.34 % | +0.99 % |
| VVC medium | +2.27 % | +3.53 % |
| AV2 medium | +1.40 % | +1.86 % |

- The first key frame's size changes on 84 of 120 points per family. The ratio to the default
  key runs 0.23-4.24, median 1.00: on moving clips the lookahead lowers the key, and on static
  ones it raises it.
- The key is a median 8.3-10.2 % of a 150-frame stream.
- So one frame of lookahead was worth 1.3-3.0 % PSNR-YUV on this 5 s window. The references get
  none, so BP-2 turns it off.
- The CLI still reads frame 1 early (I/O only). With the cap off, nothing in the encoder uses it.

### B3. Buffer: accepted; one bucket, one replay, invalid points out for all (decision 4)

**What each encoder was told before BP-2** (verified in `eval/nceval/cbr.py`):
- x264/x265: peak = target, `--vbv-init` 0.9 (default).
- SVT, libvpx and libaom: initial and optimal levels 600 ms.
- Ours: peak 2x the target (`rate_control.rs:451`, default), the key up to 1 buffer
  (`rate.key_cap_buffer` 1.0), and re-codes allowed up to 1.15 buffers (`rate.recode_overflow`).
- openh264: no buffer controls.

**Replay of every stream through one 1 s bucket** drained at the target, every bit counted. The
share of points over one buffer, ladder rungs (150-1200 kbps) / low rungs (40, 75):

| Encoder | Bucket starts empty (F = 1.0 s) | F = 0.6 s | Rate / target, median (ladder) |
|---|---|---|---:|
| x264 medium | 2.5 % / 66.7 % | 3.8 % / 76.9 % | 0.96 |
| x264 veryfast | 0.0 % / 5.1 % | 1.2 % / 17.9 % | 0.95 |
| x265 medium | 2.5 % / 71.1 % | 13.8 % / 89.5 % | 1.00 |
| SVT-AV1 M12 | 1.2 % / 56.4 % | 12.5 % / 64.1 % | 1.01 |
| libvpx rt 9 | 27.5 % / 94.9 % | 45.0 % / 100 % | 1.04 |
| libaom rt 10 | 27.5 % / 92.3 % | 51.2 % / 97.4 % | 1.03 |
| openh264 | 35.0 % / 97.4 % | 52.5 % / 100 % | 1.00 |
| ours default (H.264, HEVC, AV1, VP9, VVC, AV2 medium) | 17.5-27.5 % / 10-30 % | 52-65 % / 56-68 % | 0.98-0.99 |
| ours `strict` | 0 % / 0-5 % | 52-66 % / 55-68 % | 0.98-0.99 |
| ours `bp2` (strict, key cap 0.6) | 0-1.7 % on both | 51-83 % / 55-88 % | 0.98-1.00 |

**What this shows:**
- Ours by default uses up to 1.15 buffers (peak 1.13-1.15) on a fifth to a quarter of its
  points; the references are told 1 buffer. That was the asymmetry the review found.
  `--strict-buffer` removes it at a cost of +0.20 to +0.79 % PSNR-YUV.
- **Our rate control has no initial-fullness setting.** It plans as if the bucket starts empty.
  So even with the key capped at 0.6 s, it overflows a bucket that starts 60 % full on
  51-88 % of points.
  - Until engine task T2 adds `--buffer-initial-ms`, the only initial fullness every encoder can
    be configured to exactly is F = 1.0 s, a decoder buffer full at start.
  - BP-2 states F = 600 ms as the target and needs T2 for ours. If T2 is late, the coordinator
    decides between F = 1.0 s for all (references set to `--vbv-init 1` and
    `--buf-initial-sz 1000`) and waiting.
- **The references overshoot the low rungs.** At 40 and 75 kbps (720p30), 56-100 % of every
  reference's points are over one buffer even with an empty start, at 1.1-11x their target.
  Their encoders cannot code that little (quantizer floors).
  - Excluding over-buffer points for everyone (decision 4) therefore leaves most references with
    no valid low-rung points.
  - By the one-clip-set rule (B5), a clip needs 4 valid points per family, so the low rungs will
    mostly drop out of the headline.
  - That matches the audit's finding 2 (the headline was a very-low-rate number). It is stated,
    and the low rungs remain a separate, labelled very-low-rate view.
- x264/x265 at `--vbv-init 0.6` instead of 0.9: x264 medium's BD changes by +1.39 % PSNR-YUV
  (worse for x264), x265's by -0.12 %. x264's over-buffer share barely moves (2.5 to 1.2 %).

**Loan-adjusted rate:** used for ours only, today, in:
- `eval/nceval/cbr.py:878-903` (`add_loan_adjusted_rate`, `overshoot`, `reachable`);
- `nceval.realuse.rate_valid` (screens, ours against ours);
- `nceval.independent` (`loan_ratio`);
- `nceval.chart` chained comparisons through our own family (`per_clip_rows(paired=...)`,
  line 744, which drops rungs by `rate_valid`).

The BP-2 report never imports these or reads `key_loan_bits` or `loan_adjusted_rate_error`
(V15). Screens keep them: there both arms are ours and judged alike, owner-sanctioned. The
chained chart path is replaced by BP-2's report.

**Resolution:**
- One bucket: B = 1 s, F = 600 ms, peak = target, configured per encoder (fairness record).
- One external replay judges validity. The average-rate rule counts the same initial credit for
  all (R x T + F x R, 2 % tolerance).
- Over-buffer and over-rate points are excluded from every curve alike, and counted per encoder.
  The all-points view is secondary.

### B4. SEI: accepted; stripped from every stream (decision 13); effect measured

**Information SEI per stream:**

| Encoder | SEI bytes | Share of stream at the 40 kbps rung (median / max) | at 1200 kbps |
|---|---:|---|---|
| x264 | 656-682 B | 3.3-3.9 % / 15-16 % | 0.2 % |
| x265 | 2,399-2,405 B | 17.4 % / 39 % | 0.7 % |

- Ours, openh264, SVT-AV1 and libaom write no SEI or metadata OBU: OBU types seen were
  sequence header, temporal delimiter and frame only.

**Effect on BD-PSNR-YUV against x264 medium** (20 clips), SEI counted versus stripped:
- x265 medium: -30.92 to -34.69 % (3.77 points better for x265);
- x265 ultrafast: 4.29 points better;
- every other encoder: 0.5-1.1 points worse. The zero line was inflated: ours +0.57 to +0.82,
  SVT +0.88, libaom +0.52, libvpx +0.71, openh264 +1.09.

**Resolution:**
- `nceval.streamstrip` strips SEI (H.264 6, HEVC 39/40, VVC 23/24) and AV1 metadata OBUs from
  every stream after the timed encode, ours included.
- Decoded frames are checked identical before and after. Bytes removed are recorded. x265 also
  runs `--no-info`; x264 has no such switch (`x264 --fullhelp`).
- Tests: `eval/tests/test_streamstrip.py`.
- Ours will write an information SEI / metadata too (task T3), stripped the same way.

### B5. Result-dependent clip subsets: accepted (decision 14)

- One clip set per view: a clip on which any charted family fails a curve rule is excluded for
  every family. The count per family is a column of the headline table.
- The numeric rules are in BP-2 section 4, and the report's constants are tested against the
  text (V20): at least 4 points, overlap at least 0.5, ties, monotonicity tolerances. The old
  `BD_MIN_POINTS = 3` is not used.
- Speed is over the same clips and points as the BD it is plotted with.
- Measured here: the VMAF BD was reliable on 12-14 of 20 clips for most families against x264
  medium, against 19-20 for PSNR-YUV. On these clips the one-clip-set rule would cost about a
  third of the clips for the VMAF column, from overlap failures at the low rungs. The final count
  comes from the first BP-2 run.

### B6. Weighting: accepted (decision 9)

- The headline is equal weight per clip within each dataset, as the mean of per-clip log rate
  ratios. Per-class means are shown separately with their clip counts.
- The bootstrap is over clips, unstratified. A cluster bootstrap by source collection is the
  sensitivity figure.
- The "ours" top-up (pan and crowd to about 12 each, grade-B at half size) is drawn by the
  script before any BP-2 measurement and sealed under a new name. "ours" has never been used for
  a published result.
- The seal's contents are recorded as final before any measurement, and every later measurement
  is logged (T6).
- AOM-CTC classes (A2-A5) are not weights either: equal per clip.

## MAJORs

### M1. "ours" sources are x264 re-encodes: accepted

- BP-2 section 3 describes the pipeline truthfully: Lanczos-scaled where larger, libx264
  `-preset medium -crf 16`, 1 s closed GOPs. Every "ours" number carries it.
- The manifest gets the missing fields before the run.
- The ranking bias is **not measured** (task T7: code pristine CTC sequences directly and
  through `build_pool.py`, compare). The next draw is stored near-lossless with no periodic keys.

### M2. x265 AVX-512: accepted, measured (decision 6)

`--asm avx512` makes x265 medium take 0.924x the time and ultrafast 0.942x (c8a.large, 3 clips,
5 rounds). Its streams differ. Under the single-run rule each run's own stream is scored, so
the old reason not to use it is void. BP-2 runs `--asm avx512`.

### M3. Build asymmetry: accepted, measured (decision 7)

Plain build's time over the timed build's (c8a.large, 3 clips, 5 rounds):

| Encoder | Timed build | Plain / timed |
|---|---|---:|
| x264 medium / veryfast / ultrafast | LTO + PGO | 1.051 / 1.058 / 1.014 |
| x265 medium | LTO | 1.017 |
| SVT-AV1 M12 / M13 | LTO + PGO | 1.045 / 1.047 |
| libvpx rt 9 / 7 | LTO + PGO | 1.034 / 1.018 |
| libaom rt 10 / 8 | LTO | 1.031 / 1.028 |
| ours 0.6.0 (H.264 medium, veryfast, HEVC medium, AV1 medium, VP9 veryfast) | PGO | 1.007, 0.984, 0.988, 1.018, 1.031 |

- Streams were identical except x264 veryfast's rebuilt binary.
- PGO is worth -1.6 to +3.1 % for ours and up to 5.8 % for x264.
- BP-2 builds every reference with PGO once per version and target (task T4 for x265, libaom
  and openh264), and records flags, training set and effect in the fairness record.
- Ours is also compiled for `x86-64-v3` while the references' control code is baseline
  x86-64: stated in the record, not measured.

### M4. Fixed QP: decided by the owner (decision 5); key offsets measured

"Fixed QP" is a fixed base QP with each encoder's own frame and block behaviour.

**Measured at QP 32 / q 40:**
- x264 and x265 code the key at QP 29 against 32 for P frames (`--ipratio` 1.4, 3 QP finer).
- SVT-AV1 `--rc 0 --qp 40` averages QP 39.85 over 150 frames (`--enable-stat-report`): some
  frames, most likely the key, are coded finer. Per-frame QPs were not parsed.
- libvpx and libaom with min = max = q have no key offset.
- Ours: not measured in this run (our CLI refuses `--rate-log` at fixed QP). Our fixed-QP mode
  has a codec-chosen key offset by default (`--key-qp-offset`, "the codec chooses"), and a
  QP cadence.

BP-2 writes the actual tables (AOM CTC qindex points) into section 4 and reports fixed-QP speed
beside the rates reached, never as one ranking.

### M5. libaom's real-time configuration: accepted, measured; the record now uses the projects' own RT flags

Programs compared: `svc_encoder_rtc` (libaom's real-time example, one layer) against `aomenc
--rt` as charted, and `vpx_temporal_svc_encoder` (libvpx's) against `vpxenc --rt`.

| | Time (example / chart CLI) | BD-rate (example vs chart) PSNR-YUV / VMAF |
|---|---:|---|
| libaom speed 10 | 0.557 | +1.27 % / -0.34 % (18 clips) |
| libaom speed 8 | 0.567 | +4.05 % / +1.55 % |
| libvpx speed 9 | 1.147 | -4.00 % / -6.83 % (16 clips) |
| libvpx speed 7 | 1.091 | -5.24 % / -7.17 % |

- **libaom** with its project's real-time tool set is **1.8x faster** at about the same
  efficiency. The review was right: the charted libaom speed understated it by 44 %.
- **libvpx's** example is 4-7 % more efficient and 9-15 % slower than `vpxenc --rt`.
- The examples' rate control overshoots more at the low rungs (median 9-11x the target at 40-75
  kbps against 2.5-3.4x for the CLIs), so their low rungs are mostly over-buffer.
- **Resolution:** the fairness record gives `aomenc` the example's controls (CDEF on;
  warped/OBMC/global motion, order hint and TPL off; deltaq 0; cost updates every 3; CfL, smooth,
  angle-delta and filter intra off; default-tx-only intra; loop-filter control 1; CDF update mode 1;
  lines 2036-2067 of
  `svc_encoder_rtc.cc`). It gives `vpxenc` the example's VP9 controls (`vpx_temporal_svc_encoder.c`
  lines 907-916). Each with a citation.
- The loop-filter control and CDF update mode are passed explicitly; both match the example
  and the default.
- Palette and intraBC are set by the example only for screen content.
- The record's `aomenc` command must be re-timed against the example before freezing (part of
  T10's pipe-clean).

### M6. Stream identity across machines for the references: accepted

Each machine series scores its own streams. The hash rule (V11) is for ours only.

### M7. Timing residuals: accepted

- The canaries now include the fastest encoders, with per-encoder limits (±4 %, set from the
  measured single-run sd), and a mid-run canary.
- The encoder runs in a cpuset cgroup, not under `taskset`. Threads are sampled while it runs.
- User and sys are reported separately, with a user-only sensitivity column. Any major page
  fault refuses the encode.
- 5 % of encodes are timed again on a second box, and the per-class spread is published.
- The headline is published on Zen 5 and Graviton4.
- The CLI file readers stay in the timed span (the program a user runs). Their cost is stated:
  up to 4 % for libvpx (study section 5).
- Memory at 1080p is not measured; the major-fault refusal catches pressure.

### M8. The 5 s window: accepted

A 600-frame one-key window and a 600-frame window with a key every 150 run on the history
subset beside the headline. The headline is labelled "5 s, one key frame".

### M9. Zero line and twins: decided (owner, decision 8)

- The zero line is x264 medium `--tune zerolatency`: familiar, reproducible, and the front does
  not depend on it.
- The twins are separate labelled rows. A reference's BD-J is "one configuration" (its default
  tune), never a per-metric mix.

### M10. Reference coverage: partly accepted

- Claims are limited to "against these configurations".
- Kvazaar (pinned and built) and uvg266 are an owner question, task T9.
- 10-bit for the A2 sources is stated as a limitation: our encoders are 8-bit.
- The openh264 harness source and hash are published.

### M11. PGO overlap check: accepted, implemented

- `nceval.content_id` adds the curated alias and shoot table (OldTownCross =
  `old_town_cross_420_720p50`, and the SVT 2006 shoot).
- `eval/tests/test_content_id.py` fails on today's pair, as the review asked.
- A2 holds OldTownCross and CrowdRun, both from the training clips' shoot. DucksTakeOff, InToTree
  and ParkJoy are not in A2's 22 (listing read 2026-11-23).
- The training set moves for ours and every reference (T5).

### M12. Pre-registration: accepted

- The order is fixed: scripts, then a pipe-clean on a throwaway set, then freeze (commit hash
  in the manifest), then the top-up seal, then one run.
- A deviation rule (fix and re-run, never waive; report incomplete if money is short) and a
  sealed-set log (T6) are written into BP-2.
- The history subset's rate-behaviour stratum uses the published 0.6.0 data, so the selection
  does not wait for, and is never redone after, a BP-2 run.
- Budget: the about $90 estimate fits the about $110 left. A shortfall is flagged to the
  coordinator (decision 11).

### M13. Reproducibility: accepted in part

- VC-1K is numbers-only by licence and cannot be reproduced by a third party; BP-2 says so.
- "ours" gets per-clip URLs and recipes where the licences allow.
- The timed binaries with their profile data are published.
- The study's records and its scratch tools are now kept in `~/nanocodec-data/bench-method-records/`,
  under a retention entry.

### M14. Speed unit: accepted

Megapixels per second per encoder, plus a per-resolution fps table; never a bare fps.

### M15. Statistics: accepted

- Mean of log rate ratios.
- The bootstrap is 95 % with 10,000 resamples in the new report's own constants, not
  `nceval.interval`'s.
- A cluster bootstrap by source collection is the sensitivity figure.
- Holm correction over a pre-declared list of headline claims; everything else is labelled
  exploratory.
- Box noise is measured by the repeats, not assumed independent.

## MINORs

| Item | Response |
|---|---|
| 10-bit to 8-bit | Verified against `(x + 2) >> 2` on one A2 sequence before the first run (BP-2 section 3); not yet done. |
| "Comparability with published CTC numbers" | Removed. |
| Manifest name | The clip-set manifests are `~/nanocodec-data/clips/vc1k_2026-11-18.json` (the run reads that file); `eval/clipset/vc1k.json` is its selection rule. BP-2 names both. |
| VC-1K independence | Stated: unseen clips, not an unseen dataset. |
| V10 covers ours only | A 2 % sample of reference streams is cross-decoded (V10). |
| V17 cannot fail | Replaced: per-rung offset within 1 point of the median offset, front membership and speed within 3 %. |
| History frames' PGO | Kept release PGO binaries where they exist; daily frames are plain builds, labelled. Nothing regenerated without its profile. |
| Zen 3 series | Stated in the record (one core, two threads). |
| Size and crop | Exact visible size, per frame, for every decoder path (BP-2 section 3); size changes upscaled by the pinned player model. |
| VMAF model | The model file's source (Netflix/vmaf v3.2.1) and SHA-256 in the manifest; "a 3H-distance model on 240p-540p pictures" stated. |
| Cost estimates | The decode share and the 1080p scaling are marked as estimates in BP-2 section 15. |

## The adaptive-resolution question (decision 2), measured

**How often each encoder codes smaller frames** (20 clips x 6 rungs):

| Encoder | Streams that switch | Share of frames coded smaller |
|---|---:|---:|
| ours AV1 medium (default on) | 23-24 % (28-29 of 120), all at 40-150 kbps | 15.3-15.6 % |
| ours AV1 veryfast | 15 % | 10.4-10.8 % |
| libaom rt 10 / 8 with `--resize-mode=3` | 62 % / 61 % | 15.3 % / 15.1 % |
| libvpx rt 9 / 7 with `--resize-allowed=1` | 142 of 240 streams | not split |
| SVT-AV1 M12 with `--resize-mode 3` | 19 % | 4.8 % |
| every reference at its chart / example setting | 0 % | 0 % |

**BD effect:**
- Turning ours off (`nores` against default) costs +1.63 % PSNR-YUV / +0.17 % VMAF (AV1 medium)
  and +1.26 / +0.07 % (veryfast).
- Turning the references' dynamic resize on, BD against x264 medium (PSNR-YUV; VMAF):
  - libaom rt 10: -25.72 to -22.52 % (-34.02 to -28.48 % on VMAF);
  - libaom rt 8: -28.31 to -25.41 %;
  - libvpx rt 9: -1.74 to +0.08 % (VMAF unchanged);
  - libvpx rt 7: -7.70 to -7.02 % (VMAF -19.14 to -20.06 %);
  - SVT M12: -23.96 to -20.29 % (on fewer valid clips).

So dynamic resize makes every reference worse here, while ours gains a little from its own.

**Resolution:** ours keeps it on. Each reference runs as its project's real-time example sets
it, which is off: libvpx `rc_resize_allowed = 0`, libaom `rc_resize_mode = 0`, SVT RTC
default. The fairness record cites these lines. Size-changed frames are scored after the pinned
player-model upscale for everyone (V9).

## Decisions taken (coordinator, under the owner's delegation)

| # | Question | Decision | Measured cost or effect |
|---|---|---|---|
| 1 | Our start-up lookahead in benchmarks | Off | Ours loses 1.3-3.0 % PSNR-YUV, 0.5-3.5 % VMAF (B2) |
| 2 | Adaptive resolution | On for ours; references as their projects' real-time configurations set it (off) | Ours off would cost +1.3 to +1.6 % PSNR; references' resize on makes them worse (above) |
| 3 | Frame hold | All-skip frames (T1); decoded count = input for all | Holds on 6-10 % of streams, 0.1-1.8 % of frames; a skip frame is tens of bytes |
| 4 | Over-buffer points | Excluded for every encoder alike, counted | At the low rungs 56-100 % of references' points are over-buffer: the low rungs largely leave the headline |
| 5 | Fixed QP | Base QP, each encoder's own frame/block QP behaviour | x264/x265 keys 3 QP finer; SVT slightly finer; libvpx/libaom none |
| 6 | x265 AVX-512 | On | 5.8-7.6 % faster |
| 7 | Reference PGO | Every reference that supports it, built once | References gain 1.4-5.8 % today where PGO/LTO exists |
| 8 | Zero line | x264 medium default tune | `vbv-init 0.6` costs x264 1.4 % |
| 9 | Weighting | Equal per clip; per-class shown | - |
| 10 | "ours" top-up | Yes, before first use, sealed under a new name | - |
| 11 | Budget | Proceed within about $110 | - |
| 12 | Jetson | History subset only | - |
| 13 | SEI / metadata | Stripped from all, verified | x265 3.8-4.3 points better; others 0.5-1.1 points worse against x264 (B4) |
| 14 | One clip set per view | Yes | about a third of VMAF clips at risk (B5) |

## Later decisions (coordinator, 2026-11-23; BP-2.1)

1. **Initial fullness:** 1.0 s for every encoder until T2 lands, then 600 ms for all. Both are
   in the fairness record's `[phase]` table, and the commands take the current phase.
2. **The 40/75 kbps rungs:** a separate, labelled very-low-rate view, out of the headline. Equal
   exclusion applies there too, with counts shown.
3. **References:**
   - Kvazaar v2.3.2 is the real-time HEVC reference.
   - VVenC v1.14.0 low-delay faster/fast is the VVC reference (the CTC run's build; its
     low-delay configs are experimental in that release).
   - uvg266 only once a pinned build exists (none in `tools/anchors/versions.env`).
   - The fairness record states that AV2 has no real-time reference.
   - T9 covers building them with PGO and the pipe-clean.

T2 (initial fullness) and T3 (ours writes an information SEI) go to the engine agent.
