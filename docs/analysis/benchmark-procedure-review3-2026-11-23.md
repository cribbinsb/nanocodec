# Third hostile review: benchmark procedure BP-3 (2026-11-23)

The same reviewer and role as reviews 1 and 2 (a rival encoder's lead engineer). This reads BP-3
at 9f88abf7a: `docs/benchmark-procedure.md`, the answer to review 2
([benchmark-procedure-response2-2026-11-23.md](benchmark-procedure-response2-2026-11-23.md),
"response 2"), `docs/benchmark-fairness.toml` ("the record"), `docs/results/sealed-log.md` and
`tools/bench/plan.py`.

- **Verified in code or data:** each item marked *verified*.
- **Light computations here:** the ladder's targets, bits per pixel and legal-rung counts on the
  re-sealed set's manifest (`~/nanocodec-data/clips/heldout_2026-11-23b.json`), and a listing of
  the stored chart results under `~/nanocodec-data/work/heldout-chart/cloud-results/f150-ip300`.
- **Not run:** no encode, no timing, no AWS.

## Summary

- Review 2's three BLOCKERs: N1 resolved, N3 partly, **N2 not resolved**.
- **One BLOCKER remains:** the re-sealed set is described as 84 unseen clips plus 13 redrawn. All
  84 have stored results from every one of the chart's 68 families, made today, and response 2
  read those per-clip results. The sealed log does not record it.
- Three MAJORs a competitor would lead with: the compliance marker (C1), the ladder (C2) and
  libaom's configuration (C4).

## Part 1. Review 2's items

| # | Item | Verdict | Evidence |
|---|---|---|---|
| N1 | Headline clip set collapses | RESOLVED | The exclusion rule is withdrawn; all points count (BP §7). Response 2 confirmed the collapse (0 of 20 clips). New concerns about what replaced it: C1. |
| N2 | Sealed set used for design | **NOT** | See C3. The 16 design clips are excluded and logged (*verified:* `sealed_exclusions.json`, `plan.py --sealed`). The remaining 84 are not unseen. |
| N3 | Fixed QP is four modes | PARTLY | *Verified in the record:* SVT-AV1 is now true CQP (`--aq-mode 0`), libvpx and libaom are unpinned `--end-usage=q`, and every encoder's key and block behaviour is measured, ours included. Left: C5. |
| N4 | Zero line | PARTLY | Per-metric twins for single-metric figures, and both numbers in any headline (BP §8). But BD-J is still against the default (psycho-visual) tune, and the VMAF twin is assigned by name, not by measurement. |
| N5 | Adaptive resolution and the scaler | PARTLY | Re-measured under the pinned scaler on unsealed clips, which is what was asked: the references do worse or the same with resize on. The scaler sensitivity is still unmeasured (T11), and only our streams pass through it. |
| N6 | libaom/libvpx commands untested | PARTLY | Run and compared with the examples, but at max-q 52; the record's commands use the default 63 (*verified:* no `--max-q` in either CBR command). The libaom speed gap is open: C4. |
| N7 | Several runs, budget | RESOLVED | Nothing is published at the interim phase; re-costed at about $119. |
| N8 | `content_id` shoots | RESOLVED per response 2 | The "ter" shoot and source-path comparison are reported added; not re-read here. |
| N9 | Smaller items | RESOLVED | AV2 stripping, test coverage and the plan-stage box are answered. |
| Record | Kvazaar `--no-psnr --no-wpp` | RESOLVED | *Verified* in both Kvazaar commands. |
| Record | VVenC peak, intra period, placement | RESOLVED | `--MaxBitrate` set; tested; charted in Fixed QP only. |
| Record | Missing presets | RESOLVED in the record | Six x264 presets in the record's comment; BP §2 still lists three. |
| Record | openh264 version | RESOLVED in the record | v2.7.0 there; BP §2 still says v2.6.0. |
| Record | libaom/libvpx omitted settings | RESOLVED | `--max-intra-rate` and `--gf-cbr-boost` now in the commands; quantizer limits explained. |

## Part 2. New findings

### C3. BLOCKER. The re-sealed set `heldout_2026-11-23b` is not unseen, and the log does not say so

- **Claim:** response 2 and sealed-log entry 2: "the 84 unseen kept clips of the first set" plus 13
  redrawn.
- **Verified:** `~/nanocodec-data/work/heldout-chart/cloud-results/f150-ip300/` holds a directory
  for every one of those 84 clips, each with 68 family entries (`x264-medium`, `svtav1-m12`,
  `openh264-baseline~cabac`, `nanocodec-h264@slower`, ...), written today between 12:11 and 12:15.
  The audit's fix table already called this set "now the main chart set for the re-measure".
- **Verified:** response 2's section "Owner question: smaller clip sets" computes per-clip BD-J
  "from the recorded chart points ... in `work/heldout-chart/cloud-results/f150-ip300`" for
  "heldout_2026-11-23 (80 clips have all families)". So the bench-method agent read per-clip
  results of the sealed set, for every family, while writing BP-3.
- **The sealed log has three entries (1, 2, 2a).** None records the chart re-measure or that
  analysis. BP §12 says every measurement of a sealed set is logged.
- **Why it matters:** after those results existed, the ladder changed twice (to 150-1200, then
  300-1200), the upper rungs were dropped, the exclusion rule was replaced and the zero line was
  redefined. Whether or not any of those choices looked at the sealed results, the procedure
  cannot show that they did not. "Unseen" is the one property this set was built to have.
- **Also:** the manifest is not committed (*verified:* `git ls-files eval/clipset` has only the
  `.draw.json` and `.outcome.json`; the set file is in the data root). BP §3 says each manifest is
  "a JSON file committed before the run".
- **Fix:**
  - Log the chart re-measure and the subsampling analysis now, as entries before entry 2.
  - Then choose and state one of: (a) "ours" is published as a held-out-by-source set that was
    measured once under the old method before BP-3 was fixed, in those words, with only the 13
    redrawn clips called unseen; or (b) a new draw from a collection not yet touched.
  - Commit the manifest with its decoded-frame hashes.
  - Make the seal a property the storage enforces: results for a sealed set may be written only by
    the frozen procedure's run id.

### C1. MAJOR. The compliance axis: the threshold and the label can flatter us

- **Rule** (BP §7): an encoder with under 90 % compliant points gets a hollow marker labelled "not
  buffer-compliant".
- **Where the references land** (response 2, final ladder, mean of the four rungs): libvpx about
  87-88 %, libaom about 87-89 %, openh264 about 82-90 %, some SVT-AV1 presets about 89 %. Ours is
  100 %. The 90 % line sits one to three points above three reference families, and it was set
  after that table existed, on the 20 design clips.
- **Each side's own mechanism is treated differently:**
  - Ours holds the bucket by re-coding frames (`--strict-buffer`). Allowed.
  - libvpx, libaom and openh264 hold it by dropping frames; the examples also cap the quantizer at
    52 because they can drop. BP-3 turns dropping off for all, then labels them non-compliant.
  - After task T1 ours will send all-skip frames for held pictures, which is our form of "nothing
    new this frame". A reference's dropped frame is the same thing to a viewer.
- **Zero tolerance:** one bit over at one frame makes a point non-compliant. The marker hides
  whether an encoder missed by 1 % or by 3x.
- **Unspecified:** which points form the denominator (all coded points, or only clips with four
  legal rungs), and that the share depends on the phase (F = 1.0 s or 600 ms).
- **Undershoot is free:** an encoder that spends less than the target is compliant by
  construction. Ours treats the target as a ceiling.
- **Fix:**
  - No binary marker. Publish the continuous figures only: share over one buffer, the
    95th-percentile peak in buffers, and rate over target.
  - Or keep a marker with a threshold and tolerance fixed now from an outside source (for example
    a stated receiver model), not from this table, and say it was set after seeing design data.
  - Add one row per frame-dropping reference with its drop on, dropped frames scored as the
    previous picture and counted, with compliance recomputed. That is the configuration its
    authors ship.
  - Print "compliant by re-coding" beside ours, with the share of frames re-coded and its time.

### C2. MAJOR. The ladder: three in one day, no upper range, and a 16x spread in bits per pixel

- **History:** 150-4000 plus 40/75 (BP-1, BP-2), 150-1200 (response 2's interim), 300-1200
  (final). Each change followed measurements on the design clips. The last one is fairer to the
  references than the first (their compliance at 300 is far better than at 150), and it should be
  credited. It is still a post-hoc choice.
- **No upper range.** The audit's finding 2 measured that the gaps roughly halve above 600 kbps
  (AV1 -24.8 % against -47.2 %). The 2400 and 4000 rungs that showed this are gone, and BP §8
  still describes a "low" and an "upper (600-4000)" regime that no longer exists. 1200 kbps at
  720p30 is 0.043 bits per pixel; the ladder's bottom is 0.011.
- **The sub-linear rule** (pixels^0.75 x (fps/30)^0.5) is plausible, and close to common streaming
  ladders. It is cited as "the owner's judgement" with no source. Its effect, *computed here from
  `plan.scaled_kbps`*:

  | Clip | Targets, kbps | Bits per pixel |
  |---|---|---|
  | 424x240, 16 fps | 42-168 | 0.026-0.103 |
  | 640x360, 30 fps | 106-424 | 0.015-0.061 |
  | 1280x720, 30 fps | 300-1200 | 0.011-0.043 |
  | 1280x720, 60 fps | 424-1697 | 0.008-0.031 |
  | 1920x1080, 60 fps | 779-3118 | 0.006-0.025 |

  - The same rung name covers a 16x range of bits per pixel across the benchmark.
  - The AOM-CTC A2 class (1080p, up to 60 fps), the only pristine high-resolution content, gets the
    lowest rates per pixel: 0.006-0.025. Response 2's own table says references are compliant on
    only 79 % of points at 0.005-0.01 bits per pixel. So the very-low-rate regime that was removed
    from the headline returns for A2 through the scaling.
  - On the re-sealed "ours" set the bottom rung is 0.011-0.022 bits per pixel (median 0.014).
- **T12 moves our tuning screens to the same rungs and scaling.** Our presets will then be tuned
  on exactly the benchmark's rates, window and key structure; only the clips differ. No reference
  is tuned to this ladder.
- **Fix:**
  - Publish the bits-per-pixel range per dataset and per resolution class beside every headline.
  - Run the dropped 2400 and 4000 rungs once on AOM-CTC and VC-1K (their sources allow it) as a
    labelled upper-range figure, so the reader sees the gap at higher rates.
  - Report A2 separately from the other CTC classes.
  - State T12 in the limitations: "our presets are tuned at these rates and this window".
  - Fix §8's regime text and §3's "ladder plus the upper and low rungs".

### C4. MAJOR. libaom is charted through a command known to be 6-11 % slower than its own example

- Response 2: the libaom example takes 0.94x (speed 10) and 0.90x (speed 8) the CLI command's
  time, "a gap not yet explained. The record keeps the CLI".
- That comparison was run with max-q 52; the record's command has max-q 63. The record's command
  has therefore still not been timed against the example.
- **Fixed QP is worse** (*verified* in the record): the libaom and libvpx `qp` commands carry none
  of the example's real-time controls (no `--enable-global-motion=0`, `--enable-obmc=0`,
  cost-update frequencies, intra tool switches). Review 1's M5 showed those controls are worth
  1.8x for libaom. So libaom's fixed-QP speed is charted in the slow configuration.
- **Fix:** chart libaom from its example program (one layer), or find the difference before
  freezing; until then print "CLI 6-11 % slower than the project's example" on the libaom rows.
  Give the fixed-QP commands the same real-time controls as the CBR ones.

### C5. MAJOR. Fixed QP: SVT-AV1 loses its own adaptation, ours keeps its own

- The definition allows "its own frame- and block-level QP behaviour". Ours keeps a P-frame
  cadence of 85-138 qindex, block offsets up to 48 and a HEVC key 9 QP finer (response 2).
  libaom and libvpx keep their constant-quality boosts.
- SVT-AV1 is run with `--aq-mode 0`, because its default (2) is CRF by its help text. That removes
  SVT's own block-level behaviour while ours stays.
- **Fix:** chart SVT-AV1 in both modes (`--aq-mode 0` and its default), labelled; keep our
  `rate.constant_qp=1` row beside our default, as response 2 offers, on the chart itself.

### C6. MAJOR. `tools/bench/plan.py` contradicts the procedure in three places

- *Verified, line "decoder":* every job of ours is marked `"decoder": "ours"`, every reference
  `"ffmpeg"`. BP §6 says every stream, ours included, is decoded by the standard decoder and
  cross-checked. The design measurements behind BP-3 therefore scored our streams from our own
  decoder. The audit's finding 7 was about exactly this kind of path.
- *Verified:* the 50 % source-rate rule applies only `if clip.get("source_kbps")`. A manifest
  without the field gets every rung, silently. Principle 2 says refuse.
- *Verified:* `--sealed` checks only the exclusion list. It does not check the sealed log, the
  frozen procedure hash or four legal rungs. Fixed QP plans one quantizer, not the six of §4.
- **Fix:** standard decoder for all in the plan; refuse a clip without `source_kbps`; refuse a
  sealed run without a log entry and a frozen commit.

### C7. MAJOR. Clips and classes of the re-sealed set

*Computed from the manifest:*
- 97 clips; 85 have four legal rungs under the 50 % rule (9 have three, 3 have two). The headline
  is over at most 85.
- With four rungs: bodycam 15, videocall 15, texture 13, static 12, drone 11, crowd 10, lowlight
  8, pan 1.
- 59 of 97 are Wikimedia Commons uploads; 540 lines 52, 360 lines 28, 720 lines 9, 240 lines 7.
- **Pan is one clip.** The set has no vehicle- or panning-camera class in any usable number. Any
  sentence about "moving camera" content cannot come from this set; drone clips are a different
  motion.
- The 12 clips without four rungs are the lowest-rate sources. Dropping them selects against the
  most compressed sources, which is reasonable, and must be stated with the count.
- **Fix:** state "85 clips, no pan class" with the headline; name the content the set does not
  cover.

### C8. MAJOR. The two headline machines are both AMD, and one runs beside its own harness

- The trimmed plan runs the full sets on c8a (Zen 5) and c6a (Zen 3). BP §5 and §15's scope list
  still say the headline is on Zen 5 and Graviton4.
- c6a.large is one core with two threads; the encoder shares the core with the harness thread
  that samples it every 100 ms. The study measured +66 % for two busy siblings on Intel. The
  harness is light, but its effect on c6a is not measured.
- No Intel and no Arm machine carries the full sets.
- **Fix:** measure the harness's effect on c6a (same encodes with sampling on and off), or use a
  c6a.xlarge and leave the sibling idle; correct §5 and §15; say "two AMD generations".

### C9. MINOR. Text that still contradicts the decisions (*verified*)

- Status block: "Until the rest exist, no result is labelled BP-2".
- §2: families list three presets of x264/x265, four of SVT, and VVenC as a CBR family; openh264
  v2.6.0; "no B-frames" without the VVenC note the record has.
- §3 table: the old 100-clip description and "about 111"; "CBR (the ladder plus the upper and low
  rungs)".
- §8: regimes "upper (600-4000)".
- §5: "The headline is published on ... Zen 5 (c8a) and Graviton4 (c8g)".

### C10. Still open from earlier reviews, and required before any run

- T1 (all-skip), T2 (initial fullness), T4 (reference PGO), T5 (PGO training set), T7 (source
  pipeline bias), T10 (box, score and report scripts), T11 (scaler sensitivity), and the 10-bit
  conversion check. None is done. The procedure cannot yet be executed, and its thresholds
  (canaries, refusals) have never been exercised.

## Part 3. What a competitor would publish against a BP-3 result

1. "Their held-out set had been measured with all 68 encoders before the procedure was fixed, and
   their own log omits it." (C3)
2. "They forbid our frame dropping, then mark us 'not buffer-compliant' at a 90 % line drawn just
   above our 87 %." (C1)
3. "They stop at 1200 kbps for 720p and give 1080p60 0.006 bits per pixel; their own audit says
   the gap halves at higher rates." (C2)
4. "They run libaom through a command they measured as slower than our example and did not find
   out why; at fixed QP they left our real-time controls off." (C4)
5. "Their encoder is tuned on the benchmark's own ladder, window and key structure." (C2, T12)
6. "Their composite score is quoted against x264 with psycho-visual tuning on." (N4)
7. "Their source clips are x264 re-encodes, and that bias is still unmeasured." (review 1 M1, T7)
8. "Both headline machines are AMD." (C8)

## What BP-3 now gets right

- The collapse found in review 2 was measured honestly (0 of 20 clips) and the rule withdrawn.
- All points on the curve, with rate behaviour published rather than used to select clips.
- The 16 design clips are excluded by a file the planner enforces.
- Fixed-QP behaviour is verified from sources and measured per encoder, ours included, and ours is
  reported as the encoder with the most variation.
- The references' resize setting was re-measured under the pinned scaler on unsealed clips.
- Kvazaar's, VVenC's and SVT-AV1's entries were corrected; VVenC left the real-time chart.
- Commands now come from the record through a planner that exists.
- Nothing is to be published at the interim phase.

## Remaining BLOCKERs

1. **C3:** the re-sealed "ours" set is not unseen (84 of 97 clips have stored results from every
   family, read during BP-3's design), the sealed log omits it, and the manifest is not
   committed. Log it and either relabel the set truthfully or redraw from an untouched
   collection.

No other BLOCKER. C1 (the compliance marker), C2 (ladder disclosure and an upper-range figure)
and C4 (libaom) are cheap to fix and should be fixed before the procedure is frozen.

## Verification of BP-3.1 (9827bed01)

A short check of each BLOCKER and MAJOR above against the code and documents at 9827bed01, and
against [benchmark-procedure-response3-2026-11-23.md](benchmark-procedure-response3-2026-11-23.md).
Nothing was run.

| # | Finding | Verdict | Evidence |
|---|---|---|---|
| C3 | The re-sealed set is not unseen (BLOCKER) | RESOLVED by relabelling | *Verified:* sealed-log entries 1b (all 100 clips measured by 68 families, 12:11-12:15), 1c (the per-clip results read for response 2) and 3 (the relabel). The committed `eval/clipset/heldout_2026-11-23b.manifest.json` has 97 clips, each with `decoded_sha256`, `source_kbps` and `status`: 84 "audited once, never tuned on", 13 "unseen (redrawn)". BP §3 and §16 carry the label. `plan.py --sealed` now needs a log entry, a frozen commit and a clean tree. Not done, as response 3 says: a seal the storage enforces. |
| C1 | Compliance marker | RESOLVED | *Verified:* BP §7 has no threshold, marker or label, states the denominator, and prints ours as "compliant by re-coding". The `+drop` families exist in `fairness.py` and the record for libvpx, libaom and openh264. Notes below. |
| C2 | Ladder | PARTLY | Bits per pixel and a separate A2 figure are in BP §4, and the rule is stated as the owner's judgement with no outside source. The upper-range figure is not decided; a stated limitation for it is **pending** (coordinator). |
| C4 | libaom | PARTLY | *Verified:* libaom CBR runs `svc_encoder_rtc`; the libaom and libvpx Fixed QP commands now carry the real-time controls. Open points below. |
| C5 | SVT-AV1 `--aq-mode 0` at Fixed QP | NOT yet; **pending** | *Verified:* the record still has `--aq-mode 0`. The coordinator reports the fix in progress: every encoder keeps its own default block adaptation. |
| C6 | `plan.py` | RESOLVED | *Verified:* `STANDARD_DECODER` for every codec, ours included; a clip without `source_kbps` is refused; six quantizers; sealed-run checks. |
| C7 | 23b clips and classes | RESOLVED | Stated in BP §3 (85 clips with four rungs, no pan class, 59 Wikimedia). |
| C8 | Machines | RESOLVED in text | c8a is the one headline machine; "two AMD generations" is said. The c6a harness effect is a task (T15), not yet measured. |
| C9 | Stale text | RESOLVED | *Verified by search:* none of the five stale statements remains. |

**Notes on what was added:**

- **The libaom example command has no frame count** (*verified:* the record's `cbr` list for
  libaom has no `-f`; the example has a `-f/--frames` option). The manifest's clips store 162-168
  frames for 150 measured. Unless the box hands the example a file of exactly N frames, it codes
  more frames than every other encoder and V9 refuses it. Add `-f {N}`.
- **The example takes an integer frame rate** (the record says so: 29 for 30000/1001). Its rate
  control then plans for a slightly different rate than the replay judges it by, and more so on
  the set's 16-28 fps clips. State the size or patch the timebase.
- **In the interim phase the example starts at 600 ms** while every other encoder gets 1000 ms.
  The record states it. Nothing is published at the interim phase, so it matters for design runs
  only.
- **Which libvpx/libaom row is the main one.** The main rows now use the examples' max-q 52 with
  frame dropping off; response 2 said that combination "would handicap them". The `+drop` rows
  are the shipped configuration. Show both with equal weight on every chart and in every headline
  that names libvpx or libaom, never the no-drop row alone.
- **The drop threshold (30)** is described in the record as "the threshold WebRTC uses, not
  verified in source here". Verify it, or cite the example's own default.
- **The record's libaom entry still names its tool as `aomenc`** while its CBR command runs
  `svc_encoder_rtc`. The CLI's 9-14 % gap to the example stays unexplained and is stated on the
  Fixed QP rows, where the CLI is still used.

**Remaining BLOCKERs: none.** C5 and the upper-range limitation are pending with the
coordinator; C2 and C4 are partly resolved, with the open points listed above. The tasks of C10
(T1, T2, T4, T5, T7, T11, the fleet driver and `report.py`, the 10-bit check) still stand between
this procedure and a first run.
