# The fast subset (2026-11-27)

**Owner decision (2026-11-27):** fix the history clip subset. Declare it the common fast subset. Run
the slow platforms (Mac M4, Jetson Orin Nano) only on it.

This note covers four things: why the BP-4 history subset fails V17, how the new subset was chosen,
how robust it is, and what the runs will cost.

- **Record:** `eval/clipset/fast-subset.json` (30 clips; sealed log entry 31).
- **Work area:** `~/nanocodec-data/bench/fast-subset-2026-11-27/`. It holds the per-clip tables, the
  scripts and `final.json`.
- **Data:** every figure comes from the BP-4 per-clip BD tables. They were rebuilt with the report's
  own `clip_table` from `bench/bp4-ours-{refs,ours}-{c8i,c8g}`, the c8g fill run and
  `bench/bp4-hist-c8i`.
  - The full-set BD-J values match `bp4-prep/reports/ours-*.json` exactly.
  - The c8g history table was not rebuilt (the report loads everything into memory). Its headline
    figures are the audit's.

## 1. Diagnosis: why the history subset fails V17

**Published:** on c8i, the history page's common set has 18 clip-windows on 9 clips. On it:

| | value |
|---|---|
| median offset | +5.20 |
| our families' mean offset | +5.48 |
| the references' mean offset | +2.04 |
| families outside V17's limit | 37 of 64 |

c8g is the same: +5.0 median, ours +5.2, references +1.8.

**Decomposition** (c8i; offset = subset BD-J minus full common-set BD-J, in points):

| View | clip-windows / clips | median | ours | refs | outside | max dev |
|---|---|---|---|---|---|---|
| Published: history common set, both windows | 18 / 9 | +5.20 | +5.48 | +2.04 | 37 | 12.8 |
| Same 9 clips, 150 f windows only | 9 / 9 | +2.52 | +2.82 | -0.76 | 44 | 16.4 |
| Same 9 clips, 600 f windows only | 9 / 9 | +7.55 | +8.24 | +4.98 | 42 | 13.1 |
| All 21 history clips, current families' common set | 29 / 20 | +0.12 | +0.13 | -0.35 | 21 | 3.8 |
| All 21, 150 f only | 20 / 20 | -0.84 | -1.00 | -1.57 | 23 | 4.9 |
| Full set: the long-source clips, 150 f only | 36 / 36 | +1.50 | +2.34 | +0.04 | 38 | 4.4 |
| Full set: the long-source clips, 600 f only | 37 / 37 | +4.49 | +4.85 | +3.17 | 37 | 8.2 |
| Full set: every clip, 150 f only | 91 / 91 | -0.36 | -0.41 | -0.58 | 7 | 1.9 |

**(d) Common-set shrinkage. This is the root cause, and it is structural.** Early builds' failed
curves play no part.
- The history report holds 946 families. Of these, 456 are the single-key `~k600` day families.
  They are coded only on the 10 history clips that have a 600-frame source, so they have no point on
  the other 11 clips.
- The common set requires every family. It therefore keeps only those 10 clips. One more,
  `ho_wm_texture_limestone_outcrops_and_mountain_scrub_on`, is lost: 27 current families have no BD
  there (AV1, AV2, HEVC, VVC and VP9 rungs of ours, libaom and SVT M11).
- Apart from these, no day build loses a single clip-window. 421 key-every-150 day families hold
  every window.

**(a) Window weighting: about +2.7 of the +5.2.**
- On the 9 surviving clips, every clip has both windows, so the 600-frame window carries half the
  weight. In the full set, 40 of 97 clips have a 600-frame window, so it carries about 21 %.
- On the same 9 clips, 150 frames alone read +2.52 and 600 frames alone +7.55.
- In the full set, the 600-frame window reads about 3 points worse than the 150-frame window. For
  our families the gap is +4.85 against +2.34 (2.5 points). For the references it is +3.17 against
  +0.04.

**(b) Clip choice: about +2.9.**
- The survivors are all long-source clips. Those clips are harder for every encoder, ours more so:
  at 150 frames they read +1.50 against -0.36 for the full set.
- The particular 9 add about +1.0 more.
- Class mix of the 9: lowlight 2, drone 2, static 2, bodycam 1, pan 1, texture 1. There is no
  videocall clip and no crowd clip, although these are 14 and 11 of the 91.

**(c) Clip count.**
- Even unbiased, 21 clips leave 21 families outside (max deviation 3.8).
- Random class-stratified subsets (c8i, 200 draws per size) never pass V17:

  | clips | median families outside | median max deviation |
  |---|---|---|
  | 15 | 35 | 11.4 |
  | 20 | 31 | 9.4 |
  | 25 | 28 | 8.1 |
  | 30 | 24 | 7.1 |
  | 40 | 23 | 5.7 |
  | 50 | 15 | 4.3 |

- **Sampling noise is larger than V17's limit.** The full set's own bootstrap (its 91 clips
  resampled) moves the largest family deviation by 4.7 points at the median (95th percentile:
  10.0). A 1-point V17 therefore tests how closely a subset tracks *these* 91 clips, not the clip
  population. Any subset that passes it is fitted to them.

**Fix for the history report (done 2026-11-27: `report.py` `OWN_CLIP_ROWS`; `--subset` reads a report on the
subset):**
- The single-key `~k600` families must not define the common set. Show them on their own
  long-window clips, labelled, as the `+drop` rows are.
- The day series and V17 use the key-every-150 families' common set.
- Without this fix, any subset with clips shorter than 600 frames shrinks again.

## 2. Selection

**Universe:** the 91 clips whose every window is in both chips' BP-4 common set.
- No "ours" clip is in `tuned_on.json`. The ours set is already V3-disjoint from the PGO and tuning
  sets.
- The one pan clip is outside this universe.

**Constraints:**
- Per-class counts are proportional to the universe (largest remainder).
- Long-source clips are in the full set's share: 12 of 30, against 36 of 91. Each clip's windows are
  averaged as in the report, so the 600-frame weight equals the full set's.

**Search:** seeded simulated annealing over swaps that keep class and window kind.
- **Objective (both chips, all 64 current families):** squared deviations above 0.8 points, plus
  the mean squared deviation, plus front-membership differences.
- **Sizes and seeds:**
  - sizes 15, 18, 20, 22, 25 and 30, with 6 seeds each;
  - sizes 20, 22 and 25 again, with 10 more seeds each.
- **Result:** no search at 15-25 clips passed on both chips. The best 25-clip result had 0 families
  outside on both chips, with one front difference on c8i. Several 30-clip searches passed.

**The subset: 30 clips.** By class: bodycam 5, static 5, videocall 5, drone 4, crowd 4, texture 4,
lowlight 3. 12 are long-source clips. The list is in `eval/clipset/fast-subset.json`.

**V17 per machine (BP-4, 23c962d8b):**

| chip | median offset | largest deviation | outside | front |
|---|---|---|---|---|
| c8i | +0.36 | 0.69 | 0 | same |
| c8g | +0.48 | 0.73 | 0 | same |

Our families' mean offset is +0.39 (c8i) and +0.50 (c8g). The references' is +0.33 and +0.47.

## 3. Robustness

**Other builds (BP-3 CBR, c8i, 150 f, its own families and zero line), out of sample:**

| build | median | ours mean | refs mean | max dev | outside | front differs on |
|---|---|---|---|---|---|---|
| 74aaa1fcd | -0.02 | -0.06 | +0.08 | 1.75 | 5 | nanocodec-vp9@faster, openh264-baseline (+~aq), x264-ultrafast |
| 889309c99 | +0.21 | +0.20 | +0.40 | 1.66 | 5 | the same four |

- For comparison, random stratified 30-clip subsets on 74aaa1fcd have a median max deviation of
  5.7, with 38 families outside.
- The old 21-clip subset, whose extension was chosen on these very BP-3 reference figures, has a max
  deviation of 3.2 and 26 families outside.
- **The level tracks across builds.** Ours against the references moves by at most 0.2 points. The
  strict ±1 per family and the front at the fast end do not carry over fully.

**Leave-families-out:** the selection was run on half the 13 encoder groups (4 splits × 2 seeds,
at 20, 25 and 30 clips) and judged on the held-out groups' families.
- At 25 clips, the held-out max deviation has a median of about 2.7, against 5.6 for random
  subsets. Families outside: about 7-10, against 15.
- At 30 clips: about 2.7 against 5.1, and about 8 against 14.
- **The selection roughly halves the error on families it never saw, but they never pass V17.** For
  any family not in the 64 (a new preset, a new reference), the subset's V17 pass is in-sample.

**Bootstrap of the full set:** resampling the 91 reference clips gives no draw in which the subset
passes; the median max deviation is 4.7. This is the same noise floor as in section 1, and it
applies to the full set itself.

**History days (c8i, 6 of the 30 clips are history clips, 150 f):**
- For each preset, the day builds' offset (the 6 clips against all 20) minus the current build's
  offset:
  - the median of its mean over days is 0.48;
  - the median of its largest absolute value is 1.2.
- **Exception:** H.264 veryfast's early days differ by up to 13.5 points (mean -8.9). That preset
  changed most over the period.
- Six clips is a weak check. The re-run itself (section 4) gives the full answer.

**Reading:** the subset passes V17 on both BP-4 chips and tracks the level (ours against the
references) on two other builds. Use it for the trend across days and for the slow platforms'
relative standing. Keep V17 as its admission test on each new full measurement. Expect a family or
two near the fast end of the front to flip as builds change.

## 4. Plan and estimates (estimate only; nothing launched)

**AWS history re-run (all 19 days, the 30 clips, our families only):**
- The references and the zero line already exist on all 97 clips in the BP-4 refs runs.
- **Measured base:** the BP-4 history run on 21 clips took 30.6 CPU-h on c8i (0.64 h on one
  48-slot c8i.metal-48xl) and 36.7 CPU-h on c8g (0.77 h, c8g.metal-24xl). That was 50,552 jobs per
  chip.
- **Scale:** our families' Mpx on the 30 against the 21 is 1.62x. That gives about 50 CPU-h (c8i)
  and 59 CPU-h (c8g), and about 82k jobs per chip.
  - Wall: about 1.1 h per chip, plus about 0.3 h boot and staging. The day builds (x86 and aarch64
    PGO) already exist from BP-4.
  - Six of the 30 clips are old history clips, so their points can be reused. That saves about
    20 %.
- **Price:** `tools/dev/aws_choose.py timing` today gives eu-north-1 spot at $8.4 (c8i) and $9.6
  (c8g) per 1000 vCPU-h. Metal is not in the tool, so it is priced at the same per-vCPU rate:
  - c8i.metal-48xl (192 vCPU): about $1.6/h;
  - c8g.metal-24xl (96 vCPU): about $0.9/h.
- **Total:** about **$3.5-4.5**, about **1.5 h wall** with the chips in parallel.
- **Scoring:** on the local GPU, about 9.4 M 720p-equivalent frames per chip at about 1,100 frames/s.
  That is about 2.4 h if c8g's identical streams copy c8i's scores, as in BP-4.
- **Rule:** procedure section 6 timing rules on the same metal types (V6-V8, V24, V28). Size cap
  $6; report the box and spend against the estimate.

**Slow platforms: the full family set (69 families, 8,373 jobs) on the 30 clips:**
- **Method:** each job's Mpx divided by the family's measured Mpx per wall-second on that machine,
  from its BP-4 records, over 3 slots at the run's median hourly slot utilisation.
  - Mac: 55k records, 0.65 utilisation.
  - Jetson: 18k VC-1K records, 0.93 utilisation.
- **Check:** the same method predicts 7.5 h for the Mac's full "ours" run. It took 6.8 h.

| machine | encode-h | wall | full "ours" set (97 clips), same method |
|---|---|---|---|
| Mac M4 (3 slots) | 5.3 | **about 2.7 h** | 7.5 h predicted, 6.8 h actual |
| Jetson Orin Nano (3 slots) | 19.3 | **about 6.9 h** | 18.9 h |

**Existing points:** the Mac has already finished the whole "ours" set under BP-4. Its fast-subset
figures can be read from that run once it is scored, with no new encodes. Only the Jetson (and any later Mac build) needs
the subset run. Scoring is on the local GPU.
