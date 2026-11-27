# Hostile review of benchmark procedure BP-1 (2026-11-23)

A review of `docs/benchmark-procedure.md` (draft BP-1) written as a rival encoder's lead engineer
would write it: every way the procedure, as specified, could give results that are wrong, unfair
to the references, flattering to nanocodec, irreproducible or later invalidated.

- **Read:** `docs/benchmark-procedure.md` (BP-1), `docs/analysis/benchmark-method-2026-11-23.md`
  ("the study"), `docs/analysis/results-audit-2026-11-23.md` ("the audit"),
  `docs/analysis/heldout-2026-11-23.md`, `docs/PROCEDURES.md` "Reference builds", and the code
  named in each finding.
- **Verified in code or data:** every finding marked *verified*. Line numbers are at 1e17861a3.
- **Not run:** no encode, no timing, no AWS. Sizes marked *not measured* are estimates or unknown.
- **Severity:** BLOCKER (results invalid or unfair as specified), MAJOR, MINOR.
- **Reference to BP-1 text:** `BP §n` and the line in `docs/benchmark-procedure.md`.

## Summary of BLOCKERs

| # | Finding | Favours |
|---|---|---|
| B1 | Our encoder holds and drops frames by default; BP-1 says it never does, and forbids the references to | ours |
| B2 | Our encoder reads the second frame before coding the first; BP-1 says no encoder has lookahead | ours |
| B3 | "One shared 1 s buffer" is three different constraints, and rate-invalid points stay on the curve | ours |
| B4 | x264's and x265's information SEI is counted as bits; ours writes none; the low rungs make it 4-40 % of a stream | every encoder against the zero line; everyone against x265 |
| B5 | Each pair is scored on its own, result-dependent clip subset; speed is over another subset | unknown, per pair |
| B6 | Class-balanced means give 3 clips an eighth of the "ours" headline and 2 clips a quarter of the CTC one | unknown; intervals invalid |

Each is cheap to fix before the first run. None is fixable after it without re-running.

## BLOCKERs

### B1. Our encoder skips frames by default; the references are forbidden to

- **Severity:** BLOCKER. *Verified.*
- **Text:** BP §6 lines 263-266: "No encoder may drop frames. openh264 runs with `--frame-skip 0`;
  ours never drops in these modes". V9 (line 340) requires the decoded frame count to equal the
  clip's.
- **Code:**
  - `crates/nc-encoder/src/params.rs:1702`: `hold.enable` defaults to 1.0: "holds the last picture
    instead of coding a frame that would not change it (nothing is sent ...)". The threshold
    (`hold.change` 0.5) is half a quantizer step, not an exact repeat.
  - `crates/nc-encoder/src/hold.rs:1-20`: a held frame is "not coded at all".
  - `crates/nc-encoder/src/encoder.rs:3285-3303`: rate control can also drop a frame
    (`should_drop`, `drop_frame`).
  - `eval/nceval/framestats.py:1-11,101-111`: today's harness puts held and dropped frames back as
    empty entries (`with_held_frames`), and `eval/nceval/sweep.py:584-587` counts IVF frames "when
    frames were held or dropped".
  - `tools/dev/aws_heldout_chart.py:199`: `nohold` (`hold.enable=0`) exists only as an extra
    variant, so the chart commands run with hold on.
- **Failure scenario:** on a static or low-light clip at a low rung our stream has fewer coded
  pictures than the clip. Then one of two things happens. Either V9 refuses our points on exactly
  the clips where hold fires (a result-dependent exclusion), or the builder of `score.py` reuses
  `with_held_frames`, and ours is scored with repeated pictures while openh264 is run with frame
  skip off (its rate control depends on skipping) and libvpx and libaom with frame dropping off. A
  rival reads this as: nanocodec may skip, we may not.
- **Not measured:** how often hold or the rate-control drop fires at BP-1's settings.
- **Fix:** pick one rule and write it in §2.
  - Either `hold.enable=0` in every BP-1 command of ours, plus a check that the rate-control drop
    cannot fire (count coded pictures on the box, refuse on any shortfall);
  - or allow every encoder its skip mechanism (openh264 frame skip on, libvpx/libaom
    `--drop-frame`), and score a skipped frame as the previous decoded picture for all.
  - Correct the sentence "ours never drops in these modes".

### B2. Our encoder has a start-up lookahead; the references' lookahead is switched off

- **Severity:** BLOCKER as a false fairness statement; size not measured. *Verified.*
- **Text:** BP §2 lines 78-79: "every encoder runs without B-frames, lookahead or scene-cut keys".
- **Code:**
  - `crates/nc-cli/src/input.rs:103-131` (`look_ahead_second`), called at
    `crates/nc-cli/src/main.rs:1004`: the CLI reads two frames and hands the second to the encoder
    before the first is encoded.
  - It is on by default: `first_key_lookahead` is 0.3 in `crates/nc-h264/src/backend.rs:330` and
    `crates/nc-av1/src/backend.rs:151`, and 0.5 in `crates/nc-hevc/src/backend.rs:118` and
    `crates/nc-av2/src/backend.rs:519`.
  - It sizes the first key frame from the motion between frames 0 and 1
    (`crates/nc-encoder/src/params.rs:1326-1333`).
- **Failure scenario:** the test window is 150 frames with one key frame, so the first key's size
  is the largest single rate-control decision in every point. Ours makes it with one frame of
  future; x264 and x265 are run with `--rc-lookahead 0`, libvpx and libaom with
  `--lag-in-frames=0`. A reviewer who finds `look_ahead_second` after publication calls the whole
  CBR chart "low delay for them, lookahead for you".
- **Fix:** `rate.first_key_lookahead_off=1` in every BP-1 command of ours; or keep it, state "one
  frame of start-up delay" beside every CBR number, and measure its BD effect on calibration
  clips so the size is published.

### B3. "One shared 1 s buffer" is not one constraint

- **Severity:** BLOCKER. *Verified in the commands; our encoder's current peak behaviour not
  re-measured.*
- **Text:** BP §2 line 81 ("the same rate-control constraint for all"), §7 lines 287-295.
- **What each encoder is actually told** (`eval/nceval/cbr.py`):
  - x264, x265 (lines 341-343): `--vbv-maxrate` = target, `--vbv-bufsize` = 1 s. Peak rate equals
    the target. `--vbv-init` is left at its default 0.9.
  - SVT-AV1 (301-303), libvpx, libaom (376-378): a 1 s buffer with initial and optimal levels of
    600 ms (`SVT_BUFFER_LEVEL_MS`), so their first frame is planned against 0.6 s of bits.
  - Ours (282): `--bitrate K --buffer-ms 1000`. The audit (finding 11) found our default peak is
    twice the target and that 10-16 % of our points passed one buffer; `rate.key_cap_buffer`
    defaults to a whole buffer for a key (`crates/nc-encoder/src/params.rs:1361`).
- **The validity rule does not match any of them** (`eval/nceval/cbr_stats.py:8-12,35-49`): the
  replay bucket "starts empty", so a first frame of up to 1.0 s of bits is valid. That is the
  allowance ours uses and the one the libaom, libvpx and SVT commands are configured not to use.
- **The "10 % over target" rule flags compliant streams:** a 150-frame clip is 5 s at 30 fps. An
  encoder that legitimately spends its initial buffer credit (x264: 0.9 s) is up to 18 % over
  target on that window while obeying its VBV exactly. The published "rate-invalid" count then
  reads as a reference failure.
- **Invalid points stay on the curve** (lines 293-295): an encoder that breaks the buffer gains
  quality for it and keeps the gain in the headline; the count sits in a side column. The audit
  measured what holding one buffer costs ours: 3-5.5 %, and 10-13 % with reference-sized keys.
- **Failure scenario:** the headline CBR saving includes several points bought with a looser
  buffer and a larger first key than the references were allowed, in the window (one key, 5 s)
  where that matters most.
- **Fix:**
  - One written bucket: capacity 1 s, a stated initial fullness, drain at target; every encoder
    configured to exactly it (`--vbv-init`, `--buf-initial-sz`, our peak equal to the target).
  - Our encoder run in a mode that cannot exceed it, verified per point.
  - The headline computed on buffer-valid points only, for every encoder; the all-points view is
    the secondary one.
  - Drop the "10 % over the whole clip" rule or scale it to the window (target x duration plus the
    initial credit).

### B4. The references' information SEI is counted as their bits

- **Severity:** BLOCKER (cheap to fix, inflates every headline number). *Verified in code; sizes
  computed from the code's own figures.*
- **Text:** BP §6 lines 281-283: "Bits: the elementary stream's bytes per frame ... The same rule
  applies to every encoder."
- **Code:** `eval/nceval/infosei.py:1-8`: x264 and x265 open every stream with a
  `user_data_unregistered` SEI of "about 0.6 kB and 1.5-2 kB ... Our encoder writes none".
  `eval/nceval/cbr.py:265`: `info_sei` defaults to True, so chart points keep it. The module calls
  it "noise" over 150 frames.
- **It is not noise on BP-1's ladder.** BP §4 codes the 40 and 75 kbps rungs on every clip, scaled
  by pixel rate. For the "ours" set's commonest size (540p, scale 0.65) the six lowest targets are
  about 26 to 780 kbps, or 16 to 490 kB per 5 s clip:
  - x264's 0.6 kB is 3.7 % of the lowest rung and about 1.2 % averaged over the curve in log rate;
    about 2.3 % at 360p; 13 % of the lowest rung at 240p (7.6 kbps, 4.75 kB).
  - x265's 1.5-2 kB is 9-12 % of the lowest 540p rung and 32-42 % of the lowest 240p rung.
- **Failure scenario:** x264 medium is the zero line. Its inflated rate lowers every other
  encoder's BD-rate against it by roughly 1.5-2.5 points on "ours" (*estimate from the sizes
  above; not measured*). x265 loses several points against every AV1 and VP9 encoder and ours.
  The bias is largest where the headline is weighted (low rates, small pictures).
- **Fix:** strip the SEI for all (x265 `--no-info`; x264 through `strip_info_sei` or a bitstream
  filter) and say so; or count only VCL and parameter-set NAL units for every encoder.

### B5. Each pair is scored on its own clip subset, chosen by the results

- **Severity:** BLOCKER. *Verified against the text and the BD code.*
- **Text:** BP §4 lines 170-173 (four points inside the overlap; a non-monotone curve excludes the
  clip "for that pair"), §8 lines 304-305 ("Clips without a valid BD for the pair are listed and
  excluded for that pair only"), §8 lines 308-310 (speed over the clip's points, all clips).
- **Code that will be reused:**
  - `eval/nceval/bdrate.py:119-122`: equal quality at two rates raises an error.
  - `eval/nceval/jreal.py:410-416`: `curve_bd` returns None when the overlap is under half of the
    smaller span (`MIN_OVERLAP_FRACTION`), a threshold BP-1 does not mention.
  - `eval/nceval/jreal.py:185`: `BD_MIN_POINTS = 3`, against BP-1's four.
- **Why the exclusions are not random:** a curve fails when an encoder sits at its quantizer
  ceiling (rungs coincide), overshoots the low rungs (no overlap with the zero line) or saturates
  a metric. Those are the clips where that encoder is weakest or strongest. The audit already
  found this on the fixed-QP views (finding 5: libaom on 7 of 12 clips) and required "one clip set
  per view"; BP-1 goes back to per-pair sets.
- **Speed and quality separate again:** BD is over the pair's surviving clips, speed over all
  clips. Principle 1 ("speed and quality from the same encodes") then holds per encode but not
  for the two numbers plotted as one point.
- **Failure scenario:** openh264 cannot reach the 40 kbps rung on static 240p clips; those clips
  drop out of its pair; its BD is computed on the clips where it copes. Or the reverse for ours on
  clips where the quality cap makes rungs coincide.
- **Fix:**
  - One clip set per chart view: a clip missing for any charted family is missing for all, and the
    headline is over that intersection. Publish the per-pair figures beside it.
  - State every exclusion rule numerically in §4 (overlap fraction, tie handling, minimum points)
    and make the code's constants equal to it.
  - Speed over the same clips and the same points as the BD it is plotted with.
  - Report the excluded clips' count per family in the headline table, not in a list elsewhere.

### B6. Class weighting gives a few clips most of the weight, and breaks the intervals

- **Severity:** BLOCKER for the headline and every interval. *Verified in the manifest.*
- **Text:** BP §8 lines 303-305 (class-balanced mean), 311-315 (bootstrap stratified by class),
  §3 line 93 (classes; a top-up "drawn only after the owner confirms").
- **Data** (`eval/clipset/heldout_2026-11-23.json` with its `.outcome.json`, the 100 kept clips):
  - pan: 3 clips, all from one dataset (Caltech Pedestrian, 640x480 dash camera). Under class
    balance they carry 12.5 % of the headline; one clip carries 4.2 %, against 0.83 % for a
    bodycam clip.
  - AOM-CTC: if the classes are A2-A5, the two A5 clips carry 25 % of the mean and the 22 A2 clips
    25 %.
  - VC-1K is one class, so there "class-balanced" is the plain mean: the three sets' headlines are
    weighted by different rules under one name.
- **Class is the collection, not the content** (`docs/analysis/heldout-2026-11-23.md:115`): drone,
  texture, low light and crowd are all Wikimedia Commons (59 of 100 clips); bodycam is EPIC-Kitchens
  (12 of 15), which is head-mounted kitchen video; static is UCF-Crime plus Wikimedia; videocall is
  VCD2. A class mean is a source-collection mean.
- **Intervals:** a stratum of 3 resampled with replacement has 10 distinct outcomes, and of 2, 3.
  The stratified bootstrap understates the uncertainty of exactly the strata that carry the most
  weight per clip. `eval/nceval/interval.py:95-110` merges singleton strata, which is a third rule.
- **The top-up is a post-hoc change to a sealed set:** it is drawn after the set was sealed, by a
  rule changed for the purpose ("grade-B sources at half size", refused in the original draw).
- **Fix:**
  - Headline: the plain mean over clips, with the class means in a table. Or fix the class weights
    in the manifest before the run and require a minimum of about 10 clips per class.
  - Draw the top-up now, by the script, before any BP-1 measurement, and seal the result as a new
    set name. Never add clips to a set after a measurement on it exists.
  - Bootstrap over clips without strata for the headline, and cluster by source collection as a
    sensitivity figure.
  - State the weighting rule per set beside each headline.

## MAJOR

### M1. The "ours" sources are x264 re-encodes, scaled by us; §3 says there is no scaling

- *Verified.* BP §3 lines 127-129: "4:2:0, 8-bit, limited range, as stored. There is no scaling,
  cropping or frame-rate change".
- `docs/analysis/heldout-2026-11-23.md:69-78` and `eval/clipset/build_pool.py:151-164`: each clip
  is the original (already compressed) video, Lanczos-scaled to 720, 540 or 360 lines, then
  re-encoded with libx264 `-preset medium -crf 16`, capped at 8 Mbps per 720p, with closed 1 s
  GOPs. The benchmark's "source" is the decode of that file.
- **Consequences:**
  - The reference picture is the output of x264 medium, which is also the zero line and the
    encoder family of three reference rungs. Re-coding x264's own output on the same macroblock
    grid is not neutral between H.264 encoders and the others. The sign and size are *not
    measured*.
  - The source has an I-frame every second: a quality pulse every 16-30 frames that every encoder
    must reproduce or smooth, and that PSNR rewards reproducing.
  - "Grade A, or grade B downscaled" (line 126) contradicts "no scaling" (line 128) in the same
    section. The scaler and its chroma siting are part of the source definition.
  - Source rates are 272-8556 kbps; with the 50 % rule some clips are coded only at the low rungs.
- **Manifest:** BP §3 lines 111-121 list per-clip fields (frame count, width, fps, licence, source
  bitrate, SHA-256 of decoded frames). The committed manifest's entries hold only `name`, `source`,
  `start` (seconds), `seconds`, `content_class`, `family`, `height`, `source_fps`, `draw`.
- **Fix:** describe the pipeline truthfully in §3 and with every "ours" number. Add the missing
  manifest fields before the run. For the next draw store clips near-lossless (or raw) with no
  periodic keys. Measure the bias once: code a few pristine CTC sequences directly and through the
  `build_pool.py` pipeline and compare the encoders' ranking.

### M2. x265 is held to AVX2 while ours runs AVX-512; the stated reason no longer applies

- *Verified.* BP §5 lines 231-232: "the tier is whatever the CPU offers, the same for all
  encoders". `docs/PROCEDURES.md:1399-1400,1410-1413`: x265 runs AVX-512 only with `--asm avx512`,
  the chart does not pass it, and with it x265 medium takes 0.93x the time on Zen 5.
- The old reason (`eval/nceval/chart.py:1709-1711`) was that the flag changes x265's stream
  between the quality and timing machines. BP-1 has one machine and one run per encode, so the
  reason is gone, and the handicap (7 %, their own measurement) stays.
- **Fix:** chart x265 at `--asm avx512` on AVX-512 machines (or both, the faster as the family),
  and replace the sentence with each encoder's recorded dispatch.

### M3. Build treatment is unequal between references, and BP-1 describes it wrongly

- *Verified.* BP §5 lines 237-239 and §16 line 526 say the references are "default optimised
  builds" and "not PGO". `tools/anchors/build_anchors.sh:316-336,368-391,414-435` and
  `docs/PROCEDURES.md:1372-1383`: the `x86_64-opt` builds BP §2 names give x264, SVT-AV1 and libvpx
  LTO and PGO; x265 and libaom get LTO only; openh264 gets neither.
- So the disclosure BP-1 would print is false, and the chart's HEVC, AV1-rt and H.264-rt
  comparators (x265, libaom, openh264) are the ones left without PGO. "Documents no PGO route" is
  not a reason: libvpx was done with plain `-fprofile-generate` flags (lines 419-431), which works
  for any gcc project.
- Ours is also built `-C target-cpu=x86-64-v3` (audit finding 12); the references get no `-march`.
  Their C control code is compiled for baseline x86-64.
- x264's PGO trains on its Makefile's own option sets (B-frames, not `zerolatency`); SVT and libvpx
  train on the chart's exact commands. No reference and not ours is trained in fixed QP.
- **Fix:** generic gcc PGO for x265, libaom and openh264 with the chart's commands; train x264 on
  the chart's commands; publish plain-build fps for ours beside the PGO fps; make the text match
  `build-flags.json`.

### M4. Fixed QP: x264 and x265 are not at one quantizer, and the ladder text does not match the code

- *Verified.* BP §2 lines 74-75, 83: "every encoder holds one quantizer ... for every frame and
  block". `eval/nceval/encoders/anchors.py:29-37` and `cbr.py:343`: x264 and x265 get plain `--qp`.
  Both apply their I-frame ratio (`--ipratio` 1.4, about QP-3 on the key frame) in that mode. Ours
  with `rate.constant_qp=1` has no key offset (`crates/nc-encoder/src/params.rs:1633`). Whether
  SVT-AV1 `--rc 0 --aq-mode 0` scales the key frame's quantizer was not checked.
- BP §4 lines 164-167 says the tables are "matched mean rates on a calibration set". The code's
  tables are the AOM CTC qindex points (`eval/nceval/ctc/conditions.py:12-16,46`), with the 0-63
  encoders at rounded equivalents (112-236 against 110-235). "The same table applies to all
  encoders of a codec" cannot be literal for SVT, libaom and ours on three scales.
- Speed at fixed QP is compared at different operating points: each codec's table lands on
  different rates and qualities, and encoders are faster at coarser quantizers.
- **Fix:** `--ipratio 1 --pbratio 1` for x264/x265 (or give every encoder its natural key offset
  and call the mode "fixed base QP"); write the actual tables into §4; report fixed-QP speed
  interpolated to a common quality range.

### M5. libaom's real-time speed looks wrong; verify its configuration before publishing it

- The study's own table (section 5): libaom rt 10 takes 1.934 s for 150 720p frames (78 fps),
  2.5x libvpx rt 9 (0.788 s) and 4x SVT-AV1 M13 (0.490 s). libaom's real-time speed 10 is normally
  in the same class as those. *My expectation, not a measurement here.*
- `eval/nceval/cbr.py:387-392`: plain `aomenc --rt` plus buffer options and `--aq-mode=3`. libaom's
  own real-time example (`examples/svc_encoder_rtc`) and WebRTC set many further controls (among
  them global motion, warped motion, OBMC, TPL, cost-update frequencies, CDEF and palette
  settings) that `aomenc` leaves at defaults.
- **Failure scenario:** libaom's maintainers show their encoder at several times the published
  fps with their documented real-time settings; every AV1-against-libaom speed claim is void.
- **Fix:** time libaom through its documented real-time configuration (the flag set of
  `svc_encoder_rtc` for one layer) and chart the faster; ask the same question of `vpxenc --rt`
  against WebRTC's VP9 settings. State who chose each reference configuration and against what.

### M6. Stream identity across machines is assumed for the references and already known false

- BP §5 lines 245-246 and V11: a machine series "whose streams differ is refused"; §15 line 490
  scores once because "streams are the same on every machine".
- `docs/PROCEDURES.md:1414-1418`: on Graviton x265 codes different streams with the compiler
  version; `:1397-1398,1406-1408`: x264 medium "codes different bytes on the rebuilt binary" of
  the same pinned source. The study's identity result (section 1) covers x86 boxes with one set
  of binaries only.
- **Failure scenario:** the Graviton or Jetson series is refused for x265 (and any encoder whose
  rate control uses floating point), or the rule is waived on the day. Either way the series'
  speed is plotted against quality from other streams: the audit's finding 1 again.
- **Fix:** score each machine series' own streams (budget it), and keep the hash rule for ours
  only, where bit-exactness across tiers is a design property.

### M7. Timing: the residual cloud effects are not symmetric between encoders

- **Neighbours.** The study shows contention on the same Zen 5 host slows fast encoders most
  (section 2: openh264 2.07x against x265 1.62x at full density; +6.8 % at quarter density).
  c8a.large shares its host. BP §5's argument that random box assignment "spreads box bias ...
  over every encoder alike" (line 186) holds for a box-wide factor, not for an effect that depends
  on the encoder. A neighbour load below the ±2 % canary limit biases fast encoders slower than
  slow ones, in one direction, on every box. The evidence that neighbours "did not move" times is
  five boxes on one day in one zone.
- **Canaries.** 12 encodes of 60 frames are the noisiest kind (study section 3: the shortest
  encodes have the largest jitter); the limits (±2 %, ±5 %) and the "second refusal fails the run"
  rule have no measured false-refusal rate. The reference table's own box bias is not stated.
- **V8 is vacuous as written.** A process pinned to one CPU can never show CPU time above wall,
  whatever its thread count; and `/proc/<pid>/status` "when it ends" shows a zombie with one
  thread. Only the plan-stage unpinned run tests anything, once, on one clip. `taskset` does not
  stop an encoder that sets its own affinity.
- **The timed span includes each CLI's file reader** (study section 5): `vpxenc` and `aomenc` read
  with 50,600 4 KB `read()` calls, up to 4 % of libvpx's time; ours uses 50 reads. That is the
  demo program's I/O, not the library a product links. Progress output to stderr is also timed.
- **Memory.** A c8a.large has 4 GiB; a 1080p clip takes 0.47 GiB of tmpfs, and the encoders'
  own memory at 1080p is unmeasured. Major faults are recorded (line 202) but refuse nothing.
- **One machine.** Zen 5 is the one CPU where our AVX-512 tier, x264's and SVT's run without a
  frequency penalty; the study's Zen 4 and Intel runs moved encoders ±11 % relative to it. A
  single-machine headline is a choice that needs a second machine beside it, not a limitation
  note.
- **Fix:** canary sets that include the fastest encoders, with a per-encoder (not only geometric
  mean) limit; a mid-run canary; repeat 5 % of encodes on a second box and publish the observed
  between-box spread per encoder class; cpuset cgroup instead of `taskset`; count threads while
  running; report user and sys separately and a user-only sensitivity column; refuse on major
  faults; publish the headline on two CPU families.

### M8. The window: 150 frames, one key frame, about 5 seconds

- BP §3 lines 150-151, §16 lines 519-520. The first key and the start-up transient are the whole
  rate-control story of a 5 s clip, and B2 and B3 both act there. The audit already records "over
  20 s our savings shrink by several points". The owner's own rule is to judge key-related changes
  at two key shares.
- "Stated, not corrected" does not stop a reader taking the 5 s number as the steady-state number.
- **Fix:** a second published window from the same procedure (for example 600 frames with one
  key, and a periodic-key run), on a subset if cost binds, shown beside the headline.

### M9. The zero line and the metric twins are a forking path

- BP §2 line 82: "each reference is shown at its best setting for the metric it is scored on";
  §8 lines 301-302: zero line x264 medium `--tune zerolatency,psnr`.
- BD-J is one number from three metrics. Which twin's curve enters a reference's BD-J, and which
  twin is the zero line for the VMAF and MS-SSIM thirds, is not specified. `--tune psnr` turns
  x264's adaptive quantization off, which is not its best setting for MS-SSIM or VMAF.
- The audit's finding 5 shows the choice moves savings by 1-4 points.
- **Fix:** write the rule: for each reference and each metric the best twin, selected on the
  calibration set and frozen in the manifest; the zero line's twin per metric likewise; BD-J of a
  reference stated as either "one configuration" or "per-metric best", never unlabelled.

### M10. Reference coverage does not support codec-level claims

- No same-codec reference for VVC or AV2 in BP §2 (VVenC and AVM exist in `cbr.py:394-426` for
  fixed QP only; uvg266 is absent). Their rungs can be compared only with other codecs.
- Kvazaar is pinned and built (`tools/anchors/versions.env`) and not charted; it is the usual
  real-time HEVC comparator.
- Three presets each of x264 and x265, four of SVT-AV1 and two each of libvpx and libaom, against
  35 rungs of ours. A "front" drawn through three reference points is mostly interpolation.
- openh264 is run through the project's own harness (`tools/openh264-harness`), not Cisco's
  `h264enc`; the audit found it at a setting 17 % worse than it can do. The harness is under the
  authors' control and must be published with its source hash.
- One thread, 8-bit, flat P for all: stated, but every reference is designed around threads, and
  10-bit is the better mode for x265, SVT and libaom on the 11 A2 sequences.
- **Fix:** add Kvazaar and uvg266 (or state why not), the missing fast presets, and limit claims
  to "against these configurations".

### M11. PGO training overlap: the check as written does not catch the collision it cites

- BP §3 lines 132-141, V3: content id is "the file stem with resolution and frame-rate suffixes
  removed". `OldTownCross` and `old_town_cross_420_720p50` do not match under that rule (case,
  underscores, `_420_`). No content-id helper exists in `eval/` or `tools/` (*verified by search*).
- The four PGO clips (`tools/release/release.py:248-249`) and A2's CrowdRun come from one SVT
  shoot (same camera, film stock and grain). Disjoint by name is not disjoint in kind. Whether A2
  also holds DucksTakeOff, InToTree or ParkJoy was not checked here; the check must cover them.
- The references' PGO uses the same four clips (`build_anchors.sh:81`), so the fix must rebuild
  them too.
- **Fix:** a curated alias table per sequence, checked by a test that must fail on today's
  OldTownCross pair; PGO training for ours and the references moved to sequences from no CTC class
  and no dataset's source collection.

### M12. Pre-registration is weaker than §1 claims

- The scripts do not exist (`tools/bench/` is absent), so every threshold in §5 and §10 is
  untested, and the procedure will be edited while it is first run against the real datasets.
- The audit's fix table says the sealed set is "now the main chart set for the re-measure". If any
  result on it existed before BP-1's choices (AQ mode 3, zero line, low rungs on every clip, class
  weights, the top-up), those choices were not made blind. VC-1K and the CTC sequences have been
  charted repeatedly.
- The seal is an environment variable any agent can set (`eval/nceval/sealing.py:27-28`).
- §14 selects the history subset using BD-J spread from "the first BP-1 run" (line 438), and V17
  can refuse a subset that §14 says is "never changed".
- Budget: about $90 planned against about $110 left (§15). One failed run consumes the
  contingency; the pressure will be to relax a refusal rather than re-run.
- **Fix:** build and pipe-clean the scripts on a throwaway clip set; then freeze BP-1 (commit
  hash in the manifest); then draw a new sealed set and run once. Log every measurement of a
  sealed set (who, when, why) in the repository. Write the deviation rule now: what happens when a
  check refuses and money is short.

### M13. A third party cannot reproduce two of the three sets, or the timed binary

- "Ours" comes from private paths under `/mldata`; VC-1K is "numbers only, never frames or
  streams" (`eval/clipset/vc1k.json`). A SHA-256 of decoded frames proves nothing to someone who
  cannot obtain the frames. Only AOM-CTC is reproducible, and it is the set whose protocol differs
  most from the product case.
- BP §12 step 2: a PGO binary cannot be rebuilt bit-identically, and the audit notes 0.6.0's
  profile data was not kept. `X264_REF=stable` names a branch (the commit beside it is the pin).
- The study's raw records and tools are in a scratch directory and "not landed" (study, lines
  13-15); the project's own retention rule lets scratch be deleted. The evidence for §5 can
  vanish.
- **Fix:** publish, per clip, the public URL, the extraction recipe and the hashes; publish the
  timed binaries and profile data; move the study's records to a kept directory and commit its
  tools.

### M14. Speed is reported in a unit readers will misread

- BP §8 lines 308-310: the geometric mean over clips of seconds per frame, as "fps". On "ours" the
  clips run from 240p to 720p (0.41 of 720p's pixels on average), on CTC from 270p to 1080p. The
  number is the fps of no resolution, and differs by set for the same encoder.
- **Fix:** report megapixels per second (or 720p-equivalent fps with the scaling law stated), and
  a per-resolution table; never a bare "fps" without the picture size.

### M15. Statistics

- Clips within a collection are treated as independent (59 of 100 from Wikimedia Commons; VC-1K
  from one protocol and a few device types). Intervals are too narrow by an unknown factor.
- "Two encoders differ when the interval excludes 0" is applied across 68 families, three metrics,
  regimes and three sets with no multiplicity control.
- The set mean is an arithmetic mean of BD-rate percentages, which is asymmetric (-50 % and +100 %
  are the same ratio). Use the mean of log rate ratios, as `bdrate.py:83-84` already has per clip.
- `eval/nceval/interval.py:33-34` is 90 % and 2,000 draws; BP-1 says 95 % and 10,000. The new
  report must not import the old constants.
- "Box noise is below 0.1 %" (line 313) assumes independent box biases; region, zone and day are
  shared across boxes.

## MINOR

- **10-bit to 8-bit (BP §3 lines 105-108).** "Rounding without dither" through `-sws_dither none`
  is asserted, not tested; swscale's unscaled depth conversion has its own dither tables. Verify
  against `(x + 2) >> 2` computed independently on one A2 sequence, and record the result.
- **"Comparability with published CTC numbers" (line 95)** is not available in any case: 8-bit
  conversion, a different anchor, different configurations and one thread. Remove the claim.
- **Manifest name:** BP §3 names `vc1k_2026-11-18.json`; the file is `eval/clipset/vc1k.json`.
- **VC-1K independence:** 30 other clips of the same dataset are in the tuning pool
  (`eval/nceval/pool.py:47-50`). "Disjoint by content id" is true and the dataset is not unseen.
- **V10 covers our streams only.** The references are scored from one decoder (FFmpeg's native
  H.264, HEVC and VP9). A decoder defect would hit references alone; cross-check a sample with
  JM/HM/vpxdec.
- **V17** (rank correlation at least 0.98 over 35 rungs spanning 60 points of BD) cannot fail;
  it tests nothing.
- **History frames** use PGO builds regenerated without the original profiles, so they are not
  the released binaries.
- **Zen 3 series:** c6a.large is one core with two threads; the harness runs on the sibling.
- **Decoded size and crop:** state how non-multiple-of-8 sizes (270x480, 1884x720) are cropped by
  each decoder path before hashing.
- **VMAF model:** publish where `vmaf_v1.0.16_3d0h.json` comes from and its hash, and say that a
  3H-viewing-distance model is applied to 240p-540p pictures.
- **Cost table:** "decoding +20 %" and 1080p scaling are estimates feeding a budget with 20 %
  headroom.

## What an honest BP-1 result could claim

- "On these three clip sets, at these rates, one thread, 8-bit, flat P, one key frame in 5 s, on
  one Zen 5 core, against these named configurations of the references."
- Not: savings "against x264/x265/SVT-AV1" in general; real-time capability; steady-state
  streaming efficiency; anything about VVC or AV2 relative to their own references; anything on
  Arm until M6 is fixed.
- Where a reader is most likely to be misled: the low-rate weighting of the full-range headline
  (two extra low rungs on every clip of a set that averages 0.41 of 720p), the 5 s window, the
  class-balanced mean, and "fps".

## What the procedure gets right

- One run per encode, timed and scored from the same stream; it removes the audit's worst fault.
- No packing for timing, with measurements to show why; CPU time with wall beside it.
- Refusing instead of warning; one build per run, identified by hash.
- Every stream scored from a standard decoder's output, with ours cross-decoded.
- Low rungs on every clip, so the point set no longer depends on results.
- PCHIP over the overlap with no extrapolation; no outlier removal; exclusions listed.
- Rate-validity counts by one rule with no loan deduction (the rule itself needs B3's fix).
- Input verified resident; concatenation rejected with data; canaries instead of a paired anchor.
- A limitations section, and metric-tuned twins for x264 and x265.
- libvmaf on CPU as the published scorer, pinned by version and model hash.
