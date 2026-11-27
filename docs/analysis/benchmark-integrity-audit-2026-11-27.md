# BP-4 benchmark integrity audit (2026-11-27)

This is a read-only adversarial audit of the BP-4 results published at
https://cribbinsb.github.io/nanocodec/results/ and in the README. The question: does the
benchmark have the same flaw as the decoder-compliance claim, where a claim rests on a check
that does not cover what it says it covers?

- **Sources:**
  - `docs/benchmark-procedure.md` (BP-4.0);
  - the reports in `~/nanocodec-data/bench/bp4-prep/reports/`;
  - the run records in `~/nanocodec-data/bench/bp4-*/`;
  - `tools/bench/`;
  - `docs/results/sealed-log.md`;
  - the live site data `results/data/bp3.json`;
  - the README.
- **Earlier review:** many points below repeat `bp4-adversarial-review-2026-11-25.md`. Where that
  review's finding is still open in what was published, it is marked "(open since the 11-25 review)".
- **Approach:** no code was changed. The only compute was 14 stream decodes on the PC (item 5).
- **Labels:** "Unverified" means not checked.

## Summary, ranked by how much each could change a reader's conclusion

| # | Finding | Direction | Size |
|---|---|---|---|
| 1 | The VC-1K set, presented as sealed, was tuned on: 9 of its 100 clips were the lever-ranking sets for every codec on 2026-11-18/19, after its results were seen. None of this is in the sealed log, and V3 does not test tuning sets | favours ours on VC-1K | not measured |
| 2 | The AOM-CTC low-delay view (12 of its 34 clips) is in-sample for ours (the standards lens). The site does not say so | favours ours on AOM-CTC | not measured (open since the 11-25 review) |
| 3 | The start-up burst. Our first key is 0.35-0.58 s of channel (median) against 0.09-0.13 s for x264, x265 and SVT-AV1. The 150-frame window carries most of the weight, and every buffer starts full | favours ours | 5-8 % J by the project's own measurement (open since the 11-25 review) |
| 4 | The site and README omit disclosures the procedure requires before publication (D17, section 9): the limitations list, compliance beside the BD, the re-coded share, our own-set figure and untimed points. The README overclaims stream checking | wording | — |
| 5 | Scoring decoder. The procedure says the standard decoders decode every stream; in fact our own decoder decoded every family, references included. Its identity with the standard decoders was checked on one sample that had no x265 stream | could penalise references | none found: 13 more reference streams checked here were bit-identical |
| 6 | On c8i "ours", our families and the references were timed on different machines whose check sets used different clips. Agreement between those two machines was never measured, so the V6 check could not apply | either | unknown; where measurable, 0.2-1.2 % |
| 7 | History page: the V17 tracking check fails (37 of 64 families more than 1 point off). Section 14 says the graph is then withdrawn; it is published. Its common set is 9 clips | against ours (ours reads ~3.4 points worse relative to the references) | — |
| 8 | The slow end of the speed front is uncontested: no reference is charted slower than x265 medium, SVT M8, libaom cpu8 or libvpx cpu7 | favours ours at the slow end | disclosure (open since the 11-25 review) |
| 9 | Nothing above 1200 kbps is measured, and section 16 still says the gap "narrows" there, while the measured result was a reversal for the fast rungs | favours ours' fast rungs | disclosure (open since the 11-25 review) |
| 10 | Minor: scene-cut handling is not uniform; the docs contradict the commands; scores are copied between runs keyed only on stream bytes; site bugs | small | — |

**What holds up:**
- All 414 family rows on the site match the reports.
- One thread for every encoder; the same 1 s bucket, starting full, for all.
- LTO and PGO for every encoder whose build allows them, trained on the same four clips, which are disjoint from the datasets.
- Ours alone is held to `--strict-buffer` (a handicap to ours).
- The `+drop` rows are kept out of the common set.
- Every scored stream is decoded, never taken from the encoder's reconstruction.
- No real-time, "fair" or superlative claim was found on the site.

## 1. The VC-1K set was tuned on

**Evidence.**
- Mark's plan for 2026-11-18, recorded in memory: "fix/improve as many issues relating to the poor
  meta sequences performance as possible". This came after the VC-1K results were charted that day
  (commits d9c6967d7, 8a2005ed4).
- That night's agents ranked levers on "the six worst clips of the VC-1K Fixed QP sample", for each
  codec:
  - the clip sets are `~/nanocodec-data/work/vc1k-harvest/vc1k-worst6-{av1,av2,hevc,vp9,vvc}.json`,
    each `"sample_of": "vc1k_2026-11-18"`;
  - they were built by `mkset.py`, which takes each codec's worst gap to a reference;
  - for H.264, `work/rd-depth/vc1k-worst6.json`, now deleted.
- The five codec files hold 9 distinct clips. All 9 are in
  `eval/clipset/vc1k_2026-11-18.manifest.json`, the BP-4 VC-1K set.
- These docs record the use:
  - `docs/handoff/hevc.md` (VC-1K sections);
  - `docs/handoff/rd-depth.md:16` and `:126`;
  - `docs/analysis/port-hevc-wins-2026-11-19.md:9`;
  - `docs/analysis/rd-mode-audit-2026-11-19.md:151`;
  - `CHANGELOG.md:224` ("VC-1K six worst clips J -9.8 %").
- Changes ranked this way landed and are in the frozen build:
  - HEVC TU16 heuristic off;
  - HEVC deblocking +2/+2, carried to other codecs;
  - the moving-camera key;
  - the RD final decision stage.
- Adoption went through the pool screens, which include a VC-1K "dev" class: 30 clips drawn from
  the same 825-clip source, disjoint by clip (262abde0f). But the clips and the problems to fix were
  chosen from measurement-set results.
- `docs/results/sealed-log.md` starts on 2026-11-23. Its first VC-1K entry (9) says "nothing else
  of it was read". The 11-18/19 readings and the tuning use are not logged.
- V3's test (`eval/tests/test_content_id.py:66`, `test_pgo_training_set_is_disjoint_from_every_dataset`)
  checks only the PGO set against the datasets. It does not check tuning pools or ad-hoc ranking
  sets, although procedure section 4 says the datasets are "disjoint from every ... tuning set".

**Effect.** Not measured. Ours on VC-1K is partly in-sample: directly on 9 of the 100 clips, and on
the rest through problem selection from the same source.

**What a reader is told.** The site gives VC-1K only as "phone video calls from Meta's VC-1K
dataset". It makes no held-out claim, but the procedure (section 4) calls the set sealed.

## 2. The AOM-CTC low-delay view is in-sample (open since the 11-25 review)

- The 12 A3-A5 clips in `aomctc_bp3.manifest.json` are the standards-lens sequences:
  - Johnny, KristenAndSara, ControlledBurn, WestWindEasy, BlueSky, Stockholm, RedKayak,
    SnowMountain, SpeedBag, TouchdownPass, FourPeople and Vertical_Bayshore;
  - the lens tuned ours on them (review 2026-11-25, finding 4);
  - the frozen commit still cites the lens (`crates/nc-av1/src/backend.rs:1478`).
- The site describes AOM-CTC as "the AOM common test set", with no "tuned on" label. V3 does not
  cover the lens.

## 3. The start-up burst (open since the 11-25 review)

First key in seconds of the target rate, BP-4 `ours-c8i.json`, `compliance.first_key_seconds`:

| Family | median | p95 |
|---|---:|---:|
| nanocodec medium presets (AV1, AV2, H.264, HEVC, VP9, VVC) | 0.35-0.46 | 0.51-0.66 |
| nanocodec veryfast (AV1, H.264, VP9) | 0.44-0.58 | 0.62-0.75 |
| x264 (all charted presets), x265 | 0.11-0.21 | 0.18-0.34 |
| SVT-AV1 M8 / M12 | 0.09 / 0.12 | 0.15 / 0.23 |

- **Weighting:** the headline averages each clip once over its windows. 97 clips have the 150-frame
  window and only 40 also have 600 frames. Each window starts with a key, and every buffer starts
  full at 1.0 s.
- **Size:** the project measured that capping our first key costs ours 5.0-7.8 % J
  (`first-key-2026-11-25.md`, quoted in the 11-25 review). The references' start-up loss is not
  measured.
- **Disclosure:** the site does not mention this.
- **The compliance page:** it leads with the bucket at each stream's own rate. On that measure 28-41 %
  of our medium points pass one buffer, against 7-11 % for x264 and x265 medium. At the target rate,
  ours is 0-1.1 %. Neither figure sits beside the BD (item 4).

## 4. Disclosures required before publication are missing (site and README)

D17 and section 15 make "compliance beside every BD" and "the known limitations of section 16
listed beside it" conditions of publication. Section 9 says a claim about ours quotes the
all-anchor-valid figure with the common-set figure beside it.

**Missing:**
- **Section 16 limitations:** listed nowhere on the site or in the README. Among them: "our presets
  are tuned at these rates; no reference is", nothing above 1200 kbps, sources are x264 re-encodes,
  and thin classes.
- **Compliance beside the BD:** absent from the overview table. It appears only on a separate page
  and in the click panel, which is broken (`tools/bench/pages/explorer.js:63` reads fields the data
  no longer has).
- **"Compliant by re-coding" with the re-coded share** (section 8): absent.
- **Our own-set figure:** absent; the site shows only the common set.
  - On VC-1K the common set is the more flattering figure for 32 of 35 of our families, by up to
    0.93 points (AV1 fast -27.74 shown, -27.05 the claim figure).
  - On "ours" the common set is the less flattering figure.
- **Untimed points:** 2 on ours-c8i and 1 on vc1k-c8i are not shown. "Timed clips" equals "Clips"
  everywhere.
- **Lost clips:** the names and reasons are only in the reports.

**Wording:**
- **README:** "Every stream our encoders make is checked against two independent decoders". The
  landing check codes short streams per shipped preset (`landing_conformance.py`), and the benchmark
  does no cross-decoding (D18). This is the same kind of claim as the compliance one: it asserts more
  than the check covers.
- **README chart tooltips:** they say "fps" for Mpx/CPU-s (`tools/bench/readme_chart.py:56`).
- **The method page:** out of date (it says one timed machine; it says history and buffer use are not
  measured).
- **"Held-out":** the README and site call the "ours" set held-out without the "audited once, never
  tuned on" label (sealed-log entry 3: "Every 'ours' headline carries the label"). Between 11-23
  and 11-26 the set was read for quality in at least four release measurements (entries 1b, 11, 15
  and 25) while our encoder kept changing.
  - No record shows a lever chosen from held-out results.
  - That none was is unverified.
  - The curve rules changed after results were seen (entry 17). That change made ours look worse
    (AV1 medium went from -40.64 % to -38.66 %).

## 5. Scoring decoder: ours decodes every family

- **Procedure:** section 7 and `score.py:66-67` say each stream is decoded by its codec's standard
  decoder.
- **Practice:** in BP-4, `NC_SCORE_OURS_DECODER` was set by hand. Our frozen 23c962d8b decoder then
  decoded every family, references included (`score.py:161-194`).
  - Score records name `decoder: "ours"`: for example, 13,357 of 13,357 in `bp4-ours-refs-c8i`.
  - A few records are `"standard"`: AV1 streams that change size, which go to aomdec.
  - 307 records have no `decoder` field: 77 in `bp4-aomctc-ours-c8i` and 230 in
    `bp4-vc1k-refs-c8g`.
- **Fallback:** the standard decoder is used only when ours exits with an error. A silent mismatch
  would not be caught.
- **Identity check:** `~/nanocodec-data/bench/bp4-prep/decode-check.log`, summarised in
  `bench/decode-identity-2026-11-26.md`. It sampled 3-6 streams per encoder group. It covered no
  x265 stream, one SVT preset (M10), and no VVC.
- **Checked here:** 14 decodes, 13 distinct reference streams from `bp4-ours-refs-c8i` (one clip at
  cbr800): x265 ultrafast, superfast, veryfast, faster and medium; SVT M8, M12 and M13; openh264
  CABAC; x264 ultrafast and faster; Kvazaar veryfast; libaom cpu10. Ours against the pinned ffmpeg
  8.0.1: all bit-identical.
  - So no harm was found, but the documented safeguard was not the one used.
- **V27 scorer proof:** `bench/proven-scorers.json` rests on `v27-verifyv27-0.json`, 20 streams, all
  ours (AV1, AV2, H.264, HEVC), with no reference, VP9 or VVC stream. D19 says "210 streams"; that
  may be an earlier build (unverified).
- **Metrics:** every record is `metrics_backend: vmafx`, VMAF `vmaf_v1.0.16_3d0h`, MS-SSIM, and PSNR
  at 6:1:1, the same for all families. The CLAUDE.md `--display-recon` rule does not apply here: no
  bench path reads a reconstruction.
- **Copying scores between runs:** `localscore.py:513-560` copies scores between runs by stream
  SHA-256 only, rewriting id, family and series. No clip, window or scorer check.
  - In practice it is safe: no SHA is shared by two clips among 108,588 streams.
  - All our c8g streams equal the c8i ones. About 3,400 reference streams per run differ by chip
    and were scored on their own.
  - The report's V23 guard only checks that a score's SHA is in the series' set (`report.py:228-235`).

## 6. Speed comparability between our families and the references (c8i "ours")

- **Which machines:** ours-c8i took our families' times from box0-154435-0 (run `bp4-ours-ours-c8i`,
  check clips ho_caltech/ho_epic_kitchens). The references' times came from box0-175911-0 (run
  `bp4-ours-refs-c8i`, chained, check clips vc1k_android_*).
- **Why V6 could not apply:** `wholebox.check_set_levels` compares machines only on shared check
  jobs. With disjoint clips, each machine is judged only against itself, so V6 cannot fail. How
  closely the two machines agree is unmeasured. D13 (amended) assumes machines of one type agree.
- **Where agreement could be measured, it is close:**
  - c8g "ours" (both runs used the same check clips): within 0.1-0.5 % per check job;
  - c8i VC-1K, ours against refs: 0.2-1.2 %, with ours' machine slightly slower.
- **Unequal load:** each encode is loaded by its own run's mixed queue (D13). Our encodes share the
  machine with ours, the references with references, so the load is not identical. The direction is
  unknown.
- **Other refusals:** a third machine, box0-181904-0, was refused (no start check set). Its points
  were re-timed on box0-175911-0, apart from the 2 untimed points.
- **Both sides PGO:** yes, trained on the same four `pgo_training_2026-11-23` clips (ours 20 frames
  of every preset; references 30 frames). libvpx has no LTO, a handicap to it. Speed is shown only
  from valid timings; untimed points are left out of speed (V13 amended, sealed-log 27), which hides
  them from view (item 4).

## 7. History page

- **Clips:** the common set is 18 clip-windows (9 clips) of the 137 in the view (`history-c8i.txt:213`).
  The others are lost because early daily builds coded fewer clips. Section 14's 21-clip "ours"
  extension is not what is shown.
- **V17 tracking check:** "every rung's subset-minus-full BD-J offset within 1 point of the median
  offset". Computed here from `history-{c8i,c8g}.json` against `ours-*.json`: 37 of 64 current
  families are outside it on each chip. The median offset is about +5; x264 is about -9 against
  that, and openh264 -13.
  - Section 14: "A failure withdraws the history graph until the owner decides." The graph is
    published.
- **Direction:** against ours. On the 9 clips our families read 5.2-5.5 points worse than on the full
  set, the references 1.8-2.0.
- **Provenance:** the daily builds show "LTO None, PGO None, ?". Their BUILD.json files (for example
  history-2026-11-23) say PGO, so the report's provenance block is incomplete.
- **Refused reports:** `history-{c8i,c8g}.refused.txt` hold a ZeroDivisionError in
  `report.compliance_figures` from a report attempt. Which attempt produced the published data is
  unverified.

## 8 and 9. Front coverage and the rate range (open since the 11-25 review)

- **Slowest references charted:** x265 medium (25.4 Mpx/CPU-s on c8i), SVT M8 (52.9), libaom cpu8
  (89.5) and libvpx cpu7 (110.7).
- **Our front members slower than all of them:**
  - VVC veryslow (18.8), slower (29.3) and slow (43.5) all sit below SVT M8 and are on the c8i
    front.
  - No reference is charted at x265 slow, SVT M5-M7, libaom cpu6/7 or libvpx cpu5/6.
- **Rate range:** the ladder stops at 1200 kbps (720p30-equivalent). Section 16 still says the gap
  "narrows" above it. The measured result (`upper-rates-diagnosis-2026-11-23.md`) was AV1 veryfast
  +5.4 % and H.264 veryfast +41.9 % at 600-4000 kbps.

## 10. Minor inconsistencies

- **Scene cuts:** x264's CBR command lets x264 choose scene-cut keys (rule 4). x265's still passes
  `--no-scenecut --min-keyint 150`, SVT passes `--scd 0`, and libvpx/libaom pass
  `--kf-min-dist=--kf-max-dist=150` (`bp4-ours-refs-c8i/jobs.json`). Rule 4 is not applied uniformly.
- **Docs against commands:**
  - section 16 says "one thread, 8-bit, flat P for everyone", but SVT runs 2 levels and Kvazaar
    `lp-g8d4t1`;
  - section 17's shared table says `keyint 300`, but the commands use 150 (D8);
  - section 9 says lost clips "raised our headline". In BP-4 (`ours-c8i.txt:149-157`), 8 of the 9
    clip-windows lost from the "ours" common set are lost wholly or partly because our curves lie
    above x264's quality range, so dropping them lowers our headline. The ninth is lost because our
    AV1/AV2 MS-SSIM curve has one rising point;
  - section 7 names the wrong decoder (item 5).
- **+drop intervals:** the site shows the own-set bootstrap interval for `+drop` rows, while the report
  text prints `bd`. The means are the same; the intervals differ by up to 0.8 points.
- **Wording:** "common set of 128 clips" counts clip-windows; the per-family "on 92" counts clips.

## Not checked

- Whether any held-out reading after 11-23 steered a lever.
- How much items 1 and 2 are worth.
- Whether the c8i "ours" machines agree with each other.
- The "By codec" page.
- Jetson and Mac VC-1K records that were copied into scores.

## Fixes (2026-11-27, same day)

- **1, 2:** `eval/clipset/tuned_on.json` records the 9 VC-1K and 12 AOM-CTC in-sample clips;
  V3's test (`eval/tests/test_content_id.py`) requires every dataset-tuning overlap to be listed
  there; `report.py --exclude-clips` leaves them out. Reports
  `bench/bp4-prep/reports/{vc1k,aomctc}-out-of-sample-{c8i,c8g}.json`. VC-1K's site headline is
  the view without the 9, the full set beside it; AOM-CTC is labelled with A2 alone beside it.
  Without the 9, every one of our families reads 0.08-1.31 points *better* on VC-1K (mean 0.41;
  the references -0.44 to +1.07): AV1 medium -31.01 % against -30.84 % (c8i), so the tuning did
  not inflate VC-1K. Sealed log 28-30.
- **3:** each family's median first key and its 600-frame-window BD-J are in the overview table;
  the start-up statement (5.0-7.8 % J, references unmeasured) is beside every view and in
  section 16.
- **4:** section 16's limitations, compliance, own-clip BDs, untimed points and lost clips with
  reasons are on the overview; the click panel is fixed; README wording (stream checking,
  held-out label, chart tooltip unit) corrected.
- **5:** procedure section 7 and the Method page state what was done and the identity evidence.
- **6:** stated on the c8i "ours" view and in section 16.
- **7:** the site withdraws the history graph itself on a V17 failure (`preview_history.tracking_check`),
  showing the offsets instead.
- **10:** scene cuts and keyint 150 in the fairness record; section 16's structures; section 9's
  lost-clip direction.
