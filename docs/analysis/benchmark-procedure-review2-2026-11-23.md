# Second hostile review: benchmark procedure BP-2.1 and its fairness record (2026-11-23)

The same reviewer as
[benchmark-procedure-review-2026-11-23.md](benchmark-procedure-review-2026-11-23.md) ("review 1"),
in the same role: a rival encoder's lead engineer. This reads the authors' answer
([benchmark-procedure-response-2026-11-23.md](benchmark-procedure-response-2026-11-23.md), "the
response"), the procedure as it stands at 98a59db9c (BP-2.1, `docs/benchmark-procedure.md`), the
fairness record (`docs/benchmark-fairness.toml`, "the record") and the new modules
(`eval/nceval/fairness.py`, `streamstrip.py`, `content_id.py`).

- **Verified in code or data:** each item marked *verified*. Reference sources read:
  `~/nanocodec-data/anchors/x86_64/src/libaom-v3.15.3`, `libvpx-v1.16.0`; reference `--help`
  output from `anchors/x86_64-opt/bin` and `anchors/x86_64/bin`.
- **One light computation:** a replay of the response's own stream records
  (`~/nanocodec-data/bench-method-records/all-resp.jsonl`) through the validity bucket (N1).
- **Not run:** no encode, no timing, no AWS.
- **From my own knowledge, not verified here:** marked so where used (x264/x265 behaviour in CQP,
  VVenC's intra-period rule, the Xiph "ter" sequences).

## Summary

- Of review 1's 6 BLOCKERs: 3 resolved (B1, B2, B4), 1 resolved in text (B6), 2 partly (B3, B5).
- Of its 15 MAJORs: 4 resolved, 8 partly, 3 not (M4, M9, M12).
- **Three BLOCKERs remain or are new:**
  - **N1.** The validity rule plus the one-clip-set rule leaves almost no clips. On the response's
    own streams, 3 of 20 clips survive, and libvpx, libaom and openh264 each keep 5 of 20.
  - **N2.** Sixteen clips of the sealed "ours" set were measured to design BP-2. The set is no
    longer unseen.
  - **N3.** Fixed QP is four different modes under one name (Fixed QP charts only).
- The fairness record is a real improvement: its libaom and libvpx citations check out. It still
  has wrong or uncited statements for x264, x265, SVT-AV1, libvpx, libaom, Kvazaar, VVenC and
  openh264, listed in part 2.

## Part 1. Review 1's findings: resolved or not

### BLOCKERs

| # | Finding | Verdict | Evidence |
|---|---|---|---|
| B1 | Frame hold and drop | RESOLVED (interim) | *Verified:* `hold.enable=0` in both commands of ours (record lines for `cbr`, `qp`); `eval/tests/test_fairness_record.py` asserts it; `--frame-drop` is not passed. The all-skip mode (T1) changes our streams, so switching to it is a procedure version bump, not a quiet swap. |
| B2 | Start-up lookahead | RESOLVED | *Verified:* `rate.first_key_lookahead_off=1` in the CBR command and asserted by the test. Measured cost published (1.3-3.0 %). |
| B3 | One bucket | PARTLY | Peak, strict buffer and one initial fullness are now in the record (*verified*). Left open: the interim configuration (F = 1.0 s) is unmeasured for every reference ("0.9 -> 1.0 not measured", record); openh264, Kvazaar and VVenC cannot be configured to the bucket at all; and the zero-tolerance exclusion that follows creates N1. |
| B4 | Information SEI | RESOLVED, one gap | *Verified:* `streamstrip.py` parses Annex B and AV1 OBUs correctly as far as read; x265 gets `--no-info`. Gap: `strip()` raises for AV2 (line "no metadata rule"), and our five AV2 rungs are charted, so V19 fails or is special-cased for them. |
| B5 | Result-dependent clip subsets | PARTLY | The rule is written (BP §4) and uniform. Its constants live in `bench/report.py`, which does not exist. Its consequence is N1. |
| B6 | Class weighting | RESOLVED in text | Equal weight per clip, log-ratio mean, unstratified bootstrap (BP §8). The §3 table row still says the top-up is "drawn only after the owner confirms". The seal itself: N2. |

### MAJORs

| # | Finding | Verdict | Evidence |
|---|---|---|---|
| M1 | "ours" sources are x264 re-encodes | PARTLY | Disclosed truthfully (BP §3). The bias is still unmeasured (T7), and the committed manifest still lacks the fields §3 lists (*verified:* `eval/clipset/heldout_2026-11-23.json` unchanged). |
| M2 | x265 AVX-512 | RESOLVED | *Verified:* `--asm avx512` in both x265 commands; measured 5.8-7.6 %. |
| M3 | Build asymmetry | PARTLY | Policy written; T4 not done. The record's build lines still describe today's unequal builds, and the `x86-64-v3` effect is "not measured". |
| M4 | Fixed QP | NOT | See N3. |
| M5 | libaom/libvpx real-time configuration | PARTLY | The example's controls are in the record and the citation is right (*verified:* `svc_encoder_rtc.cc` lines 2035-2065, `vpx_temporal_svc_encoder.c` lines 907-920). But the record's commands have never been run: the response measured the example programs, not `aomenc` with these flags. See N6. |
| M6 | Stream identity across machines | RESOLVED in text | Each series scores its own streams (BP §5, V11). |
| M7 | Timing residuals | PARTLY | Specified (cpuset, sampled threads, mid-run canary, 5 % repeats, two CPU families). Nothing is built. The §5 decision table still says "pinned (`taskset`)". The encoder-dependent neighbour effect is observed by the repeats, not removed. |
| M8 | The 5 s window | RESOLVED as specified | Two 600-frame windows, on the 30-clip history subset only. |
| M9 | Zero line and twins | NOT | See N4. |
| M10 | Reference coverage | PARTLY | Kvazaar and VVenC added, but their entries are unmeasured and have errors (part 2). |
| M11 | PGO overlap check | PARTLY | `content_id` works on the cited pair (*verified by reading the test*). Its shoot table has an error and a gap (N8). T5 is open. |
| M12 | Pre-registration | NOT | See N2. Also `tools/bench/` still does not exist, and nothing imports `nceval.fairness` (*verified by search*): the "record, document and runs cannot drift" claim is not yet true. |
| M13 | Reproducibility | PARTLY | Stated honestly; records moved to a kept directory (*verified:* `bench-method-records/NANOCODEC-KEEP.md`). Publication of binaries, profiles and recipes is a promise. |
| M14 | Speed unit | RESOLVED | Megapixels per second and a per-resolution table (BP §8). |
| M15 | Statistics | RESOLVED as specified | Log ratios, own constants, cluster bootstrap, Holm over a declared list. Unbuilt. |

### MINORs

- **Resolved:** the CTC-comparability claim (removed); V17 (replaced); a 2 % cross-decode of
  reference streams (V10); history PGO builds; VC-1K's status stated.
- **Not yet:** the 10-bit to 8-bit check is still "not yet done"; the VMAF model's provenance is
  promised for the manifest.
- **Stale text left in BP-2.1** (*verified*): §1 principle 3 and the §13 plan row still name
  `eval/nceval/cbr.py` / `cbr.Family.command` as the command source; §5 "SVT `--asm max`" is not
  in any command; §4 lists our AV1/VP9/AV2 points as 85-210 while
  `eval/nceval/ctc/conditions.py:46` has 110-235.

## Part 2. The fairness record, entry by entry

"Example" below means the project's own real-time example program that the record cites.

### Shared constraints

- `bframes = 0` and `lookahead_frames = 0` are not checked for Kvazaar or VVenC by the test, and
  VVenC's low-delay configurations code B slices (the record says so under VVenC). Either the
  shared constraint is "no backward references" or VVenC breaks it. Say which.
- `initial_level_ms` has two phases. Results from the two phases are two procedure versions and
  must never share a chart; BP §12 should say so explicitly.
- `resize`: see N5.
- `zero_line`: see N4.

### nanocodec

- **Fair as written:** peak equal to target, strict buffer, lookahead off, hold off.
- **`--strict-buffer` is a re-coding mode.** Ours meets the bucket by encoding a frame again when
  it would overflow. No reference has that. The time is counted, which is right, but it is why
  ours passes validity on 100 % of points and three references on a quarter (N1). The record's
  "remaining" line says this; the headline rule does not account for it.
- **Interim `rate.key_cap_buffer=1.0`:** a first key of a whole buffer. Whether libvpx, libaom and
  SVT-AV1 use a larger initial level for a larger key is not measured; their key sizing is a fixed
  heuristic. The interim phase may give ours a credit the references are configured for but do
  not use. Measure key sizes per encoder at F = 1.0 s before any interim run.
- **Adaptive resolution on (AV1 only):** N5.
- **Fixed QP:** "ours keeps its own frame-level cadence, key offset and block offsets", and the
  response says our key offset was "not measured in this run". The one encoder whose fixed-QP
  behaviour is undocumented is ours. See N3.
- **Build:** `-C target-cpu=x86-64-v3` against baseline x86-64 for every reference's C code,
  "not measured". Measure it (one plain `x86-64` build of ours on the timing clips) or give the
  references `-march=x86-64-v3`.

### x264

- **`--qp` entry is wrong:** "x264's own `--ipratio` 1.4 ... and its AQ stay". From my knowledge
  of x264 (not verified here): in CQP x264 switches AQ and MB-tree off. So at fixed QP x264 has no
  block-level adaptation while ours keeps block offsets. Check with `x264 --qp 32 -v` and correct
  the entry.
- **`--tune` entry contradicts BP §8:** the record says "the twin that suits each metric is
  charted (... the rule per metric is fixed in the manifest)"; §8 says a reference's BD-J is its
  default tune and the twins are separate rows. See N4.
- **`--vbv-init`:** x264's default is 0.9; 0.6 costs it 1.4 % (response). The final phase
  therefore handicaps x264 against its default by a measured amount to match the libvpx example's
  600 ms. State the 1.4 % beside the final-phase numbers.
- **Presets:** ultrafast, veryfast, medium. `superfast`, `faster` and `fast` are missing with no
  reason given; ours has six rungs per codec.

### x265

- **`--qp`:** the same AQ statement as x264, the same doubt.
- **`--asm avx512`:** right. The banner check that AVX-512 was actually used should be a V-check.
- **Presets:** ultrafast, superfast, medium; `veryfast`, `faster`, `fast` missing, no reason.
- **Build:** LTO only today.

### openh264

- **Version:** the record and BP §2 say v2.6.0; `tools/anchors/versions.env` pins
  `OPENH264_REF=v2.7.0` (*verified*). One of them is wrong.
- **Build:** "-O3 -O2", no LTO, no PGO, "not measured". It is the only encoder with no measured
  build effect.
- **No buffer controls and frame skip off:** openh264's rate control relies on skipping. With it
  off, 35 % of its ladder points are over-buffer (response), and under N1 it keeps 5 of 20 clips.
  This configuration makes openh264 invalid by construction. Either allow its skip (scored as a
  repeated frame, as the response's own B1 logic allows for ours once T1 lands) or chart it only
  in the all-points view and say why.
- **The harness** is the authors' program. "Source is published" is a promise; the record should
  carry its commit and SHA-256 now.

### SVT-AV1

- **CBR:** consistent with its `--help` (*verified:* `--rtc`, `--buf-initial-sz`, `--keyint`).
- **Fixed QP entry is mislabelled:** the command is `--rc 0 --qp` with the default `--aq-mode 2`.
  SVT's own help says that is CRF ("0: CRF or CQP (if `--aq-mode` is 0)", *verified*). The record
  calls it "Fixed base QP (CQP)". See N3.
- **Presets:** M8, M10, M12, M13; M9 and M11 missing, no reason.

### libvpx (vpxenc)

- **Citation correct** for AQ 3, denoiser off, static threshold 1, GF CBR boost 0, periodic boost
  0, 50/50, 1000/600/600, resize off (*verified*).
- **Example settings the record leaves out without listing them** (*verified in the example*):
  `rc_min_quantizer = 2`, `rc_max_quantizer = 52` for VP9 (lines 812-814);
  `VP8E_SET_MAX_INTRA_BITRATE_PCT` 1000 (line 942); `VP9E_SET_POSTENCODE_DROP 0`,
  `VP9E_SET_DISABLE_OVERSHOOT_MAXQ_CBR 0`, `VP9E_SET_TILE_COLUMNS`. The "remaining" line names
  only two. Leaving max-q at 63 probably helps libvpx reach low rates, so the omission may favour
  it; it must still be listed and justified.
- **`--end-usage=q` with min = max:** "the only way vpxenc holds a base QP" is false. `--end-usage=q
  --cq-level=Q` without the pin is libvpx's own constant-quality mode and keeps its key and
  golden-frame offsets. See N3.
- **Not measured as written** (N6).

### libaom (aomenc)

- **Citation correct** for the 2036-2067 block (*verified*; every flag exists in `aomenc --help`).
- **Left out without listing** (*verified in the example*): `rc_min_quantizer = 2`,
  `rc_max_quantizer = 52` (lines 1907-1908); `AV1E_SET_GF_CBR_BOOST_PCT 0` (aomenc has
  `--gf-cbr-boost`; the libvpx entry sets it, the libaom entry does not);
  `AOME_SET_MAX_INTRA_BITRATE_PCT 300` (line 2122; aomenc has `--max-intra-rate`);
  `AV1E_SET_AUTO_TILES 1`; `AV1E_SET_POSTENCODE_DROP_RTC 1`.
- **Fixed QP:** as libvpx.
- **Not measured as written** (N6). The 1.8x speed correction is for the example program.

### Kvazaar (new)

- **PSNR is computed inside the timed run.** `kvazaar --help`: "`--(no-)psnr`: Calculate PSNR for
  frames. [enabled]" (*verified*). The record's command has no `--no-psnr`. Kvazaar would be
  charged for a metric pass no other encoder runs.
- **WPP is on by default** ("`--(no-)wpp` ... [enabled]", *verified*) and the command has no
  `--no-wpp`. x265 gets `--no-wpp`. Wavefront coding costs bits (entropy sync per row) for no
  benefit on one thread.
- **"No VBV":** true of its options as read, so it joins openh264 under N1.
- **Built only in the plain prefix** (`anchors/x86_64/bin/kvazaar`), not in `x86_64-opt`. Its
  build line cites a task, not a build.
- **Never run under this procedure** ("not yet measured"; key offset "not measured").

### VVenC (new)

- **"No ... peak option" is wrong.** `vvencFFapp --fullhelp` lists `--MaxBitrate`: "approximate
  maximum instantaneous bitrate" (*verified*). Set it to the target, as for x264.
- **`--IntraPeriod 300` with `GOPSize 8`** (`lowdelay_faster.cfg`, *verified*): from my knowledge
  VVenC requires the intra period to be a multiple of the GOP size; 300 is not. Not run here. The
  existing fixed-QP path passes no intra period. Run it once before the record is frozen.
- **Not a real-time encoder,** as the record says. Then its place on a real-time chart needs the
  label on the chart, not only in the record.

## Part 3. New findings

### N1. BLOCKER. The headline's clip set collapses, and the rule that collapses it is one only ours can meet

- **Rule:** BP §7 excludes every over-buffer point; §4 needs 4 valid points per curve and excludes
  a clip for all families when any family fails; BP-2.1 moved the 40/75 kbps rungs out of the
  headline, leaving 150/300/600/1200 plus 2400/4000 where the source allows.
- **Measured here** on the response's own streams (20 clips, the four ladder rungs, bucket of 1 s
  starting empty, which is F = 1.0 s; references at their old chart commands; ours `strict`):

  | Family | Clips with 4 valid ladder points, of 20 |
  |---|---:|
  | ours, all eight families | 20 |
  | x264 veryfast | 20 |
  | x264 medium, x265 medium/ultrafast, SVT-AV1 M12/M13 | 18-19 |
  | libvpx rt 7 / 9 | 5 |
  | libaom rt 8 / 10 | 5 |
  | openh264 | 5 |
  | **All charted families together** | **3** |

  The three survivors are two video-call clips and one night scene.
- **Limits of this figure:** SEI not stripped; the upper rungs were not coded in the response, so
  a clip with six rungs could keep four; the record's new libvpx/libaom flags and
  `--buf-initial-sz 1000` may behave differently. It is the response's own 27-35 % over-buffer
  share, compounded over four rungs.
- **A second cut the authors have not counted:** the 50 % source-rate rule. "ours" sources run
  from 272 kbps. A clip whose source allows fewer than four headline rungs cannot have four points
  at all now that the low rungs are out. The manifest has no source bitrate, so the count is
  unknown.
- **Why it is unfair as well as empty:** ours passes because `--strict-buffer` re-codes frames.
  A zero-tolerance bucket is a rule calibrated to that mechanism. The surviving clips are the
  ones every one-pass controller finds easy.
- **Fix, before freezing:**
  - Measure the surviving clip count per set on throwaway clips with the record's commands.
  - Decide the rule from that, in writing: for example a stated overflow tolerance for all, or the
    all-points BD as the headline with compliance as its own published axis, or a minimum
    surviving-clip count below which a family (not the clips) leaves the headline.
  - Count legal headline rungs per clip from the manifests and state it.

### N2. BLOCKER. The sealed set was used to design the procedure

- **Verified:** the response's measurement set contains 16 clips of `heldout_2026-11-23`
  (`all-resp.jsonl`, field `input`: `ho_caltech_set02_v003`, `ho_caltech_set07_v008`,
  `ho_epic_kitchens_p01_107`, `ho_epic_kitchens_p02_116`, `ho_stock_sea_waves_01_4k`,
  `ho_ucf_crimes_burglary050_x264`, `ho_ucf_crimes_fighting043_x264`, two `ho_vcd2_*`, two
  `ho_wm_drone_*`, two `ho_wm_festivals_*`, `ho_wm_night_adygea_night`,
  `ho_wm_night_canmore_fireworks`, `ho_wm_texture_2019_06_05_orage_belfort`). The response says
  "16 held-out clips ... two from each of 8 classes".
- Every encoder and every variant of ours was coded and scored on them, and decisions 1-8 and 13
  (lookahead, resize, hold, buffer, zero line, SEI) cite those numbers.
- BP §12 says the pipe-clean runs "on a throwaway clip set (never a sealed one)". The response
  says the set "has never been used for a published result", which is a narrower statement.
- **Fix:** move those 16 clips out of the sealed set into a development set, record it in the
  sealed-set log as the first entry, and draw replacements with the top-up before sealing
  `heldout_2026-11-23b`. Any further BP design measurement uses pool or throwaway clips only. Make
  the seal a check the measurement tools enforce (the scratch tools bypassed it).

### N3. BLOCKER for the Fixed QP charts. "Fixed QP" is four different modes

- **The owner's rule** (BP §2): a fixed base QP, each encoder's own frame and block behaviour.
- **What the record does** (*verified in the commands*):
  - SVT-AV1: `--rc 0 --qp`, default AQ: CRF by its own help.
  - x264, x265: `--qp`: CQP with a 3 QP key offset; AQ most likely off (see part 2).
  - libvpx, libaom: quantizer pinned with min = max: no key offset, no adaptation, although their
    unpinned `--end-usage=q` is their "own behaviour".
  - Ours: cadence, key offset (size unmeasured) and block offsets, all developed on this
    150-frame, one-key window.
  - Kvazaar: "key offset not measured".
- So the rule is applied to ours, SVT and x264, and denied to libvpx and libaom. The response
  itself shows finer keys help on this window.
- **Fix:** one of two coherent definitions, for everyone:
  - *Constant QP:* every picture and block at one quantizer (ours `rate.constant_qp=1`, x264/x265
    `--ipratio 1`, SVT `--aq-mode 0` with its key offset off, libvpx/libaom pinned); or
  - *Constant quality:* each encoder's own quality mode (x264/x265 `--crf`, SVT CRF, libvpx/libaom
    `--end-usage=q` unpinned, ours as now), labelled "constant quality", not "Fixed QP".
  - Measure and print every encoder's key-frame and block QP spread, ours first.

### N4. MAJOR. The zero line went back to the configuration the audit called unfair to x264

- BP-1 used x264 medium `--tune zerolatency,psnr`; BP-2 uses the default tune (psy on), "familiar".
- The audit measured the difference on the published charts: our H.264 medium is -34.2 % against
  the default tune and -25.7 % against the tuned one (audit, item 16). That is 8.5 points added to
  a headline sentence by the choice of anchor.
- "Positions on the front do not depend on the anchor" is true of the chart and false of any
  sentence "N % fewer bits than x264".
- A reference's main row is now its default (psy) tune, scored on PSNR, MS-SSIM and VMAF, while
  ours is tuned to that composite. The fair row is the per-metric best twin; it is the secondary
  one.
- **Fix:** every sentence and table that quotes a saving against x264 or x265 quotes it against the
  best twin for that metric (chosen on calibration clips, frozen), with the default-tune figure
  beside it, never alone.

### N5. MAJOR. Adaptive resolution is on for ours alone, scored through our own upscaler

- Ours (AV1) codes 15 % of frames at half size at the low rungs; every reference has resize off.
- The justification is that the references score worse with resize on (response). That was
  measured with ffmpeg's bicubic scaler, "indicative" by the response's own words, on sealed
  clips. BP-2 scores through `nceval.display`, "the bit-exact twin of `nc_dsp::display`": the
  player model our encoder's resolution decision is built against.
- So the evidence for the references' setting was taken under one scaler, and the published score
  uses another, chosen by us, that only our streams pass through.
- **Fix:** re-measure the references' dynamic resize under the pinned scaler on throwaway clips,
  and give each reference whichever setting scores better; publish our AV1 rows with resize off
  beside the default; show the sensitivity of our gain to the scaler (bilinear, bicubic, Lanczos).

### N6. MAJOR. The libaom and libvpx commands in the record are untested

- The response timed and scored the example programs. The record then builds `aomenc` and
  `vpxenc` commands from a subset of the examples' settings, with different quantizer limits and
  key-size limits (part 2). No stream of these commands exists.
- T10 says the `aomenc` command "is re-timed against the example before freezing". Its BD must be
  re-measured too, and the same for `vpxenc`, whose example was 4-7 % more efficient than the old
  CLI command.
- **Fix:** an acceptance check in the record: each CLI command within a stated margin of its
  example program in time and BD, on throwaway clips; otherwise chart the example program itself.

### N7. MAJOR. Interim and final phases, T1 and T4 make at least three different BP-2 runs

- Interim F = 1.0 s, then final 600 ms; hold off, then all-skip; today's reference builds, then
  PGO for all. Each changes streams or times.
- The budget (§15, about $90 of about $110) was computed for one pass of BP-1's scope. BP-2 adds a
  second headline CPU family, two 600-frame windows, 5 % repeats, plan-stage unpinned runs and
  two more reference families. It has not been re-costed.
- **Fix:** run nothing published at the interim phase; finish T1, T2 and T4, then freeze once.
  Re-cost §15.

### N8. MINOR. `content_id` has one wrong shoot and one missing

- `720p50_parkrun_ter` is mapped into the SVT 2006 HD shoot. From my knowledge of the Xiph
  collection, the `_ter` 720p sequences (parkrun, shields, mobcal, stockholm) are an earlier SVT
  set. A4's Stockholm is from that set, so today's PGO clip parkrun shares a shoot with a CTC
  sequence that the table does not flag, and BP-1's suggested replacement (`shields`) would too.
- Name matching cannot see that two private clips with different names come from one source
  file. V3 should also compare source paths and collections, as `select_heldout.py used` does.

### N9. MINOR. Smaller items

- `streamstrip` removes every SEI, including buffering-period and picture-timing messages. That is
  fine for decoding and should be said, since those are not "information".
- The record's test checks chosen flags per tool; it does not check `bframes` or lookahead for
  Kvazaar and VVenC.
- V8's plan-stage run needs "a box with free cores"; which box, and whether it is in the budget, is
  not stated.

## Remaining BLOCKERs

1. **N1:** fix the validity and clip-set rules so the headline is not computed on a handful of
   easy clips, and so validity is not defined by a mechanism only ours has. Measure the surviving
   clip count before freezing.
2. **N2:** remove the 16 measured clips from the sealed set, log it, and redraw before sealing.
3. **N3** (Fixed QP charts only): one definition of the mode for every encoder.

Close behind, and cheap: N4 (which x264 the headline quotes), the Kvazaar `--no-psnr` /
`--no-wpp` omissions, and running the record's libaom, libvpx, Kvazaar and VVenC commands at
least once before they are called the fairness record.
