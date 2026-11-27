# Benchmark docs consistency review (2026-11-24)

Reviewed at `5a617ea59` (BP-3.10). Checked against the current decisions: c8i.2xlarge on
eu-north-1 spot as the timing machine, with the two-reference V6 check; c8g.2xlarge spot as
the second family; c6a.4xlarge and the Jetson on the history subset only; the c8a.large and
c8a.2xlarge results superseded; the ladder, anchor, BD, retention, budget and stream rules as
given by the coordinator.

Files read: `docs/benchmark-procedure.md` (BP), `docs/benchmark-fairness.toml`,
`docs/results/sealed-log.md`, `docs/analysis/benchmark-procedure-response{,2,3}-2026-11-23.md`,
`docs/PROCEDURES.md` (release, speed and results-chart sections) and `tools/bench/{plan,box,score,report,fleet}.py`.

Severity: **WRONG** means it contradicts a current decision or the code. **STALE** means it
was true once and is now outdated. **AMBIGUOUS** means a reader cannot tell which rule applies.

## A. Timing machine and machine series (the main contradiction)

| # | File:line | Says | Should say | Sev |
|---|---|---|---|---|
| A1 | BP:337 | "Timing machine: AWS c8a.large (AMD EPYC 9R45, Zen 5, 2 vCPUs, no SMT), on demand." | "Timing machine: AWS c8i.2xlarge (Intel Granite Rapids), eu-north-1 spot, one encode at a time on vCPU 1 (BP-3.10)." Move the c8a history to a "superseded" note | WRONG |
| A2 | BP:341 | "One encode at a time per c8a.large … vCPU 0 runs the harness" | One encode at a time per c8i.2xlarge on vCPU 1. The harness runs on another physical core and vCPU 1's SMT sibling stays idle (c8i has SMT; `box.pin_harness` already avoids the sibling) | WRONG |
| A3 | BP:342 | "c8a.large rather than one encode per big box: same time, a fraction of the price" | Delete it, or mark it superseded by BP-3.7/3.10 (c8a.large's L3 slice gave a 12.6 % spread between boxes) | WRONG |
| A4 | BP:343 | "On demand rather than spot … A reclaimed timing box would lose its encodes" | "Spot (BP-3.10): reclaimed boxes' jobs are re-run by `fleet.py missing`" | WRONG |
| A5 | BP:345 | "No paired x264 anchor … Canary sets detect bad boxes instead" | Replace with BP-3.6: two interleaved timing references per box (x264 medium and our AV1 medium, every 20 jobs and at the end), plus the V6 divergence check (3-point median, 2 %) | WRONG |
| A6 | BP:353 | Step 1: "Refuse a model other than the manifest's (EPYC 9R45)" | The c8i model, as the manifest records it. The code does not check it at all (see D1) | WRONG |
| A7 | BP:354-355 | "On c8a.large (4 GiB RAM) one 1080p clip, 466 MB, fits" | c8i.2xlarge has 16 GiB, or no instance claim at all | STALE |
| A8 | BP:361 | "The reference table is the median of five c8a.large boxes" | State what the canaries are compared with today (in the code: each box against its own start, see D3) | STALE |
| A9 | BP:398-399, 406-407 | "On Zen 5 that is AVX-512"; x265 AVX-512 "5.8-7.6 % faster on Zen 5" | "On the timing CPU (c8i: AVX-512, see BP:495)". The Zen 5 figure can stay as evidence, labelled as such | STALE |
| A10 | BP:421-422 | "Graviton4: c8g.large, vCPU 1" | "c8g.2xlarge, spot" (BP:492) | WRONG |
| A11 | BP:423-427 vs BP:470 | "Zen 3: **c6a.xlarge**" vs "**c6a: c6a.4xlarge**" | One value: c6a.4xlarge, history subset only | WRONG |
| A12 | BP:462 | "**The timing box is c8a.2xlarge (BP-3.7, coordinator).**" (bold, reads as current) | Mark it "superseded by BP-3.10" in the same line. Keep the measurements as history | STALE |
| A13 | BP:493 | "The c8a.2xlarge results of 2026-11-24 are superseded" | "The c8a.large (2026-11-23) and c8a.2xlarge (2026-11-24) speeds are superseded" (both, in one place) | AMBIGUOUS |
| A14 | BP:492 vs BP:504 | "c8g.2xlarge (spot) is the second CPU family" vs budget "Zen 3 (c6a.4xlarge) and Graviton (c8g) run the history subset only" | Decide whether c8g runs the full sets or the history subset, and say it once. The decision list makes c8g the second family and puts only c6a and the Jetson on the history subset, so BP:504 is probably wrong | AMBIGUOUS |
| A15 | BP:511-514 | "The headline machine is c8a (Zen 5) … c6a (Zen 3) carries the full sets too … No Intel machine carries the full sets." | "The headline machine is c8i.2xlarge (Intel), stated on every chart. c8g.2xlarge (Graviton 4) is the second family. c6a.4xlarge and the Jetson run the history subset only (BP-3.8)." | WRONG |
| A16 | BP:150 | "about 15.7 c8a core-hours" | Re-cost on c8i, or label it "(c8a estimate, BP-3.0)" | STALE |
| A17 | BP:219-221 | "$0.02 per 720p clip … (c8a.large on demand) … about $20 for both modes on the c8a series" | Re-cost on c8i spot, or label it historical | STALE |
| A18 | BP:778-780 | Reproduction step 5: "Re-time on an AWS c8a.large, one encode on vCPU 1 … within about ±2 %" | "Re-time on an AWS c8i.2xlarge …", with c8i's measured spread (1.4-1.5 % between boxes) | WRONG |
| A19 | BP:814-815 | "`aws_session.py` stays the launcher: on demand for timing, spot for scoring" | "spot for timing (c8i, c8g; BP-3.10) and for scoring" | WRONG |
| A20 | BP:820 | "Timing wall (400 c8a.large)" | Re-cost on c8i, or label it historical | STALE |
| A21 | BP:883-944 (§15) | "c8a (the headline) and c6a on the full sets"; "c8a (Zen 5) and c6a (Zen 3) run all three datasets"; cost lines for c8a and c6a; "c6a.xlarge instead of c6a.large +$22" | The whole section describes the pre-BP-3.8 plan. Replace it with the BP-3.8/3.10 plan (c8i headline, c8g second family, c6a.4xlarge and Jetson history-only, VC-1K Fixed QP deferred, history frames deferred), or mark it superseded and point to BP:499-510 | WRONG |
| A22 | BP:948-950 | "**One timing CPU** (Zen 5 in AWS c8a)" | "(Intel Granite Rapids in AWS c8i.2xlarge)" | WRONG |
| A23 | BP:959 | "no Intel or Arm machine on the full sets" | Intel is now the headline. State only what is true (for example "no AMD machine on the full sets", if c8g runs them) | WRONG |
| A24 | BP:968-970 | "c8a.large shares its host's L3 … five boxes agreed within -0.5 to +0.9 %" | The c8i limitation: a 1.4-1.5 % spread between boxes, not corrected by the within-box references (BP:490-491) | STALE |
| A25 | BP:971-972 | "A single busy core ran at about 4.5 GHz" | c8i: 3.84 GHz (BP:487) | STALE |
| A26 | BP:1182 (App. A #4) | "4.52-4.54 GHz on c8a" | c8i figures, or label it historical | STALE |
| A27 | BP:1183 (App. A #5) | "One encode per c8a.large (no SMT) on vCPU 1" | "One encode per c8i.2xlarge on vCPU 1, its SMT sibling idle; two timing references" | WRONG |
| A28 | BP:1202 (App. A #24) | "Timing on demand; scoring on spot … c8a spot: no capacity" | "Timing on spot (BP-3.10), reclaims re-run by `fleet.py missing`" | WRONG |
| A29 | BP:711 (V11) | "every machine set's streams of ours equal the c8a run's" | "equal the headline (c8i) series' streams", or "the run's first series" | WRONG |
| A30 | BP:337 and others | No explicit statement whether c8i has SMT, or how vCPU 1's sibling is kept idle | Add one sentence on the c8i topology (from lscpu) and the idle-sibling rule | AMBIGUOUS |

## B. Other contradictions inside the procedure

| # | File:line | Says | Should say | Sev |
|---|---|---|---|---|
| B1 | BP:1 vs BP:7 | Title "BP-3.10"; Status "draft BP-3.1, for owner review" | Status: BP-3.10, with the BP-3.4 to 3.10 changes listed (the list stops at BP-3.3) | STALE |
| B2 | BP:20 vs BP:24 | "a pipe-clean run is in progress (T10)" vs "The pipe-clean passed" | Delete "in progress" | STALE |
| B3 | BP:41-44 (Principle 1) vs BP:474-477, 478-494 | "no timing re-runs. Speed and quality therefore come from the same encodes by construction" vs encode-only re-timing on another machine (c8i) supplying the speeds | One explicit rule: a re-timing is allowed only when it reproduces every scored stream byte for byte (V11). The speed then comes from an encode that produced the identical stream, not from the encode that was scored. Say this in Principle 1, or the principle is false as written for every current run | AMBIGUOUS |
| B4 | BP:55-57, 746-748 (§12) vs BP-3.4 to 3.10 | "A change to any of them is a new procedure version, and results of different versions are not compared" | BP-3.7 and 3.10 combine c8a-coded scored streams with c8i timing, and BP-3.9 changed the curve rules after scoring. State how this squares with the rule (for example "timing-only versions keep BD comparable"), or the rule is broken | AMBIGUOUS |
| B5 | BP:139 (§3 table, "ours") | "97 (85 in the headline)" | The headline is the common set (41 clips, BP:284). 85 is the count of clips with four legal rungs. Write "85 with four legal rungs; headline on the common set, its size stated" | WRONG |
| B6 | BP:283 vs BP:139 | "the 9 clips with three legal rungs stay out" vs 97 - 85 = 12 without four rungs | Give the full split (9 with three rungs, 3 with fewer?) | AMBIGUOUS |
| B7 | BP:284-285 vs BP:625-626 | View = 94 clips (zero line well formed, helped by the anchor's extra rungs) vs §7 compliance denominator "the view's clip set (the clips with four legal rungs)" | The view is no longer "the clips with four legal rungs". Define the compliance denominator by one name | AMBIGUOUS |
| B8 | BP:141 vs BP:152 | A2: "11 of them 10-bit" vs "Nine A2 sequences are 10-bit … 13 are 8-bit" | One count, checked against the manifest (`eval/clipset/aomctc_bp3.manifest.json`) | WRONG |
| B9 | BP:143-144 vs BP:507 | "All three sets are run in both modes … all encoders and all rungs" vs "VC-1K CBR only in this round" | "Both modes, except VC-1K Fixed QP (deferred, BP-3.8)" | AMBIGUOUS |
| B10 | BP:684-687 (§9), 852-862 (§14) vs BP:506 | 600-frame windows and the history graph's frames presented as part of the run | The budget defers the other history frames. Say whether the windows are run in this round | AMBIGUOUS |
| B11 | BP:311 vs BP:316-318 | "Any claim about ours … over all anchor-valid clips (coordinator, pending the owner)" vs "Secondary, labelled: each family's mean over its own well-formed clips" | For ours, the own-set figure is the claim figure, not secondary. Say "for references, secondary; for ours, the claim figure (all anchor-valid clips)". Drop "pending the owner": it is decided | AMBIGUOUS |
| B12 | BP:792 | "Retention (BP-3.5, coordinator, pending the owner's answer)" | Decided: drop "pending" | STALE |
| B13 | BP:567-572 (§6 Bits) | Lists stripping and "x265 also runs --no-info. x264 has no switch." Ours' `stream.encoder_info=0` is missing | Add "ours runs `--param stream.encoder_info=0`" (it is in the record, toml:61-62, 159) | STALE |
| B14 | BP:1014 (toml `sei`) | "SEI NAL units … and AV1 metadata OBUs" | "… AV1 and AV2 metadata OBUs; ours runs stream.encoder_info=0, x265 --no-info" (as BP:568-569) | STALE |
| B15 | BP:599-602 vs sealed-log:21 (entry 9), :23 (entry 11), toml:25 | "**Nothing is published at the interim phase** … The first published run waits for T1, T2 and T4" vs "A publishable run of the frozen procedure" / "release measurement", with the phase still `interim` | Either the current runs are not publishable (say so in the sealed log), or the rule changed (say so in §7). `--buffer-initial-ms` now exists in nc-cli (`rate_control.rs:18`), so T2's status needs updating too | WRONG |
| B16 | BP:1191 (App. A #13) | "over-buffer points excluded for all" | §7: all points count in the BD, and compliance is published beside it | WRONG |
| B17 | BP:1192 (App. A #14) | "one bucket (1 s, 600 ms initial, …)" | "the phase's initial fullness (1.0 s interim, 600 ms final)" | STALE |
| B18 | BP:1195 (App. A #17) | "low rungs on every clip" | No low rungs exist. Say "the four-rung ladder; the anchor's extra points (BP-3.4)" | STALE |
| B19 | BP:705 (V6) vs BP:454 | V6 lists the CPU model and the canary limits only | Add "the two timing references within 2 % (3-point median, BP-3.6/T8)", which BP:454 assigns to V6 | AMBIGUOUS |
| B20 | BP:1151 (§18 title), T8 at :1162, T10 at :1164, T2 at :1156, T15 at :1169 | "Tasks before the first BP-3 run"; T10 "encode, score and report remain"; T8 open; T2 open; T15 "c6a.xlarge" | Update the statuses (T10 done; T8 decided: 3-point median; T2: the flag exists, phase not switched; T15 superseded by c6a.4xlarge) | STALE |
| B21 | BP:800 vs repo CLAUDE.md "Bounded disk" | "The PC keeps at least 50 GB free; downloads pause below that" | The project rule stops at under 200 GiB free. Also no 50 GB floor exists in `fleet.py` (see D6) | AMBIGUOUS |

## C. Text against the scripts (named in the text but missing or different)

| # | File:line | Says | Code | Sev |
|---|---|---|---|---|
| C1 | BP:784-790, 832, 1200 | One driver `tools/bench/bench.py` (`run/status/report`, `history-select`, `status` prints spend) | No `bench.py` exists: `tools/bench/` has plan, box, score, report and fleet only. Describe the real commands (`plan.py` → `fleet.py stage/run/score/missing/prune` → `report.py`), or mark bench.py as planned | WRONG |
| C2 | BP:746-748 | "the scripts embed it (`PROCEDURE_VERSION`)" and the report refuses mixed versions | No `PROCEDURE_VERSION` anywhere in `tools/` or `eval/`. The record says `version = "BP-3"` (toml:18), not BP-3.10 | WRONG |
| C3 | BP:807 (§13 table) | plan's input is "`cbr.Family.command`" | `plan.py` uses `nceval.fairness.command` (BP:49, BP:812) | WRONG |
| C4 | BP:358-360 | Canary set = "12 fixed encodes … x264 ultrafast, openh264, SVT-AV1 M13, our H.264 veryfast, x264 medium and x265 medium, each on two clips" | `fleet.py:105`: `canaries.json` is a seeded random draw of N jobs (`random.Random("canaries").sample(jobs, N)`), not a fixed list | WRONG |

## D. Checks the text says are enforced but the code does not enforce

| # | File:line | Says | Code | Sev |
|---|---|---|---|---|
| D1 | BP:352-353, BP:705 (V6) | Record lscpu/microcode, "Refuse a model other than the manifest's" | `box.py` has no CPU-model, microcode or lscpu check (grep: no `model`, `lscpu`, `cpuinfo`) | WRONG |
| D2 | BP:373, 377-378, BP:706 (V7) | MHz sampled every 100 ms; refuse an encode whose median MHz is below 95 % of the canary median | `box.py` refuses on steal, wall/CPU and major faults only (`box.py:238-242`); there is no MHz sampling or check. Also its `SAMPLE_S = 0.05`, not 100 ms | WRONG |
| D3 | BP:361-363 | The canaries are compared against a five-box reference table | `report.py` V6 compares each box's canaries with its own (CANARY_SET 0.02, CANARY_ONE 0.04); no reference table | AMBIGUOUS |
| D4 | BP:394-395 | "Second box: 5 % of encodes … timed again on a second box" | Not in `plan.py`/`fleet.py` (no repeat sampling found) | WRONG (or mark as not yet built) |
| D5 | BP:710-711 (V11) | "a 2 % random sample of our encodes re-run on a different box codes byte-identical streams" | No such sampling in the scripts. Only whole re-timing runs check hashes (`report.py --timing`) | WRONG (or mark as not yet built) |
| D6 | BP:800 | "downloads pause below 50 GB free" | No free-space check in `fleet.py` | WRONG |
| D7 | `report.py:491-497` vs BP:429-436, V23 | Each series is timed on its own machine | With one `--timing` run, `report.py` applies that series' times to **every** series ("with one timing series, it times every series"). A c8g or c6a scored series reported beside one c8i re-timing would silently get c8i times. Restrict it to the superseded series of the same ISA, or require an explicit `--timing SERIES=RUN` map | WRONG |

## E. fleet.py / box.py comments and defaults

| # | File:line | Says | Should say | Sev |
|---|---|---|---|---|
| E1 | fleet.py:6 | Usage example `--series c8a --type c8a.large --region eu-west-1` | `--series c8i --type c8i.2xlarge --region eu-north-1 --spot` | STALE |
| E2 | fleet.py:21 | "`run` shards the jobs over K on-demand timing boxes" | "spot timing boxes (BP-3.10; `--spot`), or on demand" | STALE |
| E3 | fleet.py:141, 144, 491-492 | "timing boxes on demand (a reclaim would waste the timing)"; on-demand is the default and spot needs `--spot` | Make spot the default for timing, or require the flag explicitly, so a run cannot fall back to on-demand silently. Fix the comment | AMBIGUOUS |
| E4 | fleet.py:341 | "c8a.large has 4 GiB" | A machine-neutral statement (c8i.2xlarge: 16 GiB) | STALE |
| E5 | box.py:381 | `--series` help "e.g. c8a" | "e.g. c8i" | STALE |

## F. PROCEDURES.md

| # | File:line | Says | Should say | Sev |
|---|---|---|---|---|
| F1 | PROCEDURES.md:1513 | "Benchmark procedure (draft BP-1, 2026-11-23) … c8a.large timing … Until its scripts exist, the rules below describe what the current charts did." | "BP-3.10 is normative for every published result: c8i.2xlarge spot timing, `tools/bench/*`. The rules below are the pre-BP-3 charts' method (legacy, 0.6.0 and earlier)." | WRONG |
| F2 | PROCEDURES.md:1514-1550 ("Results charts") | Ladder 150/300/600/1200 plus upper rungs 2400/4000 and 40/75 low rungs; low and upper regimes; class-balanced BD-J; the "10 % over target" rate rule | Label the whole block "legacy charts (pre-BP-3)", or replace it with a pointer to BP §4, §7 and §8 (300-1200 only, no regimes, equal weight per clip, the bucket replay) | WRONG |
| F3 | PROCEDURES.md:1831-1832 | Sealed set "today `heldout_2026-11-23`" | `heldout_2026-11-23b` (97 clips, "audited once, never tuned on"; sealed log) | STALE |
| F4 | PROCEDURES.md:504-506, 634, 651-660, 728, 769-772 (Release) | `release.py` times charts on c8a, c6a, c8g and the Jetson (`chart_refresh`, `aws_timing`, `fixedqp_timing`, history on four c8a.large) | BP:3-5 says a result made any other way is not published. State that the release's chart stage must run BP-3 (`tools/bench`), or that these chains are retired | AMBIGUOUS |
| F5 | PROCEDURES.md:1257-1286 (Measure speed) | The site's speed series on c8a.large, c8g.large and c6a.large (one pair at a time) | Mark them as legacy series (pre-BP-3), superseded for published results by BP-3.10 | STALE |

## G. Response documents and sealed log (dated records)

These are historical records. Their statements were true when written, so they should not be
rewritten. Each response needs a one-line banner, "Machine and plan decisions superseded by
BP-3.10 (c8i.2xlarge headline, c8g second family, c6a and Jetson history only)", so that no
reader takes them as current.

| # | File:line | Says | Sev |
|---|---|---|---|
| G1 | response3:17 (C8) | "**The headline machine is c8a (Zen 5)** … c6a runs on c6a.xlarge (+$22)" | STALE (banner) |
| G2 | response2:26, 268 | "Timing on c8a.large on demand"; "c8a and c6a on all three datasets … c8g on the history subset … about $119" | STALE (banner) |
| G3 | response:45, 261, 267 | c8a.large timings | STALE (historical evidence, fine with the banner) |
| G4 | sealed-log:23 (entry 11) | Records the c8i switch, but does not record c8g.2xlarge as the second family or that the c8a.large speeds (entry 7) are also superseded | AMBIGUOUS (add one clause) |

## Consistent with the decisions (checked, no action)

- The ladder: BP:232-237, toml:39 and `plan.py:38-43` agree (300/500/800/1200, ^0.75, ^0.5, 50 % cap via `SOURCE_RATE_SHARE`). There is no very-low view and no upper rungs (BP:233, :951).
- The anchor extension: BP:292-304 matches `plan.py:66-72` (0.5x/2x, QP step 5, max 51) and `report.py`.
- BD: `MIN_POINTS = 4`, `MIN_OVERLAP = 0.5`, the monotone tolerances, the envelope (`ENVELOPE = True`) and the common set over the non-drop families (`report.py:521-524`) match BP §4. Ours' "ALL anchor-valid clips" is printed first (`report.py:579-583`).
- V6 divergence: `REFERENCE_DIVERGENCE = 0.02`, `REFERENCE_SMOOTHING = 3` and the references (`fleet.py:361-362`: x264 medium and nanocodec-av1@medium at 500, `REFERENCE_EVERY = 20`) match BP:448-459.
- Input on tmpfs, verified resident (`box.check_resident`); SEI stripping (`box.py` V19); our `stream.encoder_info=0` and x265 `--no-info` in the record; Fixed QP pruning with SHA-256 and size (`fleet.prune`); `fleet.py missing` exists.
- Datasets: 97 / 84 + 13 / VC-1K 100 / AOM-CTC 12 + 22 = 34, BP:139-141 and the sealed log agree (apart from B5, B6 and B8).

## Re-check at a4b5d0116 (2026-11-24)

I re-read the rewritten procedure (2c13bace7) and the review fixes (a4b5d0116): the procedure,
the fairness record, the sealed log, the three response banners, PROCEDURES.md lines 504-507,
1261, 1520-1521 and 1834-1847, and `tools/bench/{fleet,box,report}.py`.

### Original items

| Items | Status | How |
|---|---|---|
| A1-A30 (machines) | RESOLVED | One authoritative Machines section (BP §2): c8i.2xlarge spot is the headline; c8g.2xlarge is the second family, history subset this round; c8a.large and c8a.2xlarge are superseded; c6a and the Jetson are not run this round. The SMT sibling stays idle (§6). §15 and §16 are rewritten; the reproduction step, the launcher text and Appendix A are gone or replaced |
| B1, B2, B20 | RESOLVED | Status block rewritten; §18 lists open tasks only; §19 is the change log |
| B3, B4 | RESOLVED | Principle 1 states the re-timing exception and ties it to V11 identity, across machine series within one version |
| B5, B6, B7 | RESOLVED | "97 (85 with four legal rungs)"; the headline is the common set; compliance counts over "the view's clips". See N4 for one remaining number |
| B8 | RESOLVED | Nine 10-bit A2 sources |
| B9, B10 | RESOLVED | §15 plan table; VC-1K Fixed QP and the 600-frame windows are deferred |
| B11, B12 | RESOLVED | Claim figure and secondary figure separated; "pending" dropped |
| B13, B14 | RESOLVED | §7 Bits and the toml `sei` row now name ours' `stream.encoder_info=0` and AV2 OBU types 8/9 |
| B15 | RESOLVED | Status: nothing published, interim phase; sealed-log 9 and 11 corrected |
| B16, B17, B18 | RESOLVED | Appendix A removed |
| B19 | RESOLVED | V6 now includes the reference divergence |
| B21, D6 | RESOLVED in the text and the code | `fleet.py` refuses below `NC_MIN_FREE_GIB` (default 200). See N5 for the round's exception |
| C1, C2, C3 | RESOLVED | `bench.py` and `PROCEDURE_VERSION` removed from the text; §13 lists the real commands; the toml says `version = "BP-3.10"` |
| C4 | RESOLVED in principle | Canaries are now each box's own jobs (`fleet.py:404-407`). See N1 for the count |
| D1, D2 | RESOLVED | `box.cpu_model`/`cpu_mhz` recorded; `report.py:178-198` enforces V6 (one model per series) and V7 (95 % of the box's median clock) |
| D3 | RESOLVED | The text says each box's normalised canaries are checked (§6, V6) |
| D4, D5 | RESOLVED | Removed from the text (not built) |
| D7 | RESOLVED | `report.py --timing … --timing-series S`: a re-timing applies to exactly one quality series, from runs that hold only the named series, with V11 per point |
| E1, E5 | RESOLVED | |
| E2, E3, E4 | PARTLY: see N2 and N3 | The default is now spot and `--on-demand` is opt-in; the "c8a.large has 4 GiB" comment is gone; two comments are still stale |
| F1-F5 | RESOLVED | PROCEDURES.md: BP-3.10 named as normative; "Results charts" labelled as the old method; the release chains and speed series labelled legacy; 23b named. See N6 |
| G1-G4 | RESOLVED | Superseded banners on all three responses; sealed-log 11 names c8g and both superseded c8a speeds |

### Remaining and new (7)

| # | File:line | Says | Should say | Sev |
|---|---|---|---|---|
| N1 | BP:190 vs `fleet.py:476` | "Canaries: three of the box's own jobs" | `fleet.py stage --canaries` defaults to 6, and each box draws that many from its shard (`fleet.py:405-406`). Set the default to 3, or write 6 | WRONG |
| N2 | `fleet.py:21` | "`run` shards the jobs over K on-demand timing boxes" | "spot timing boxes (the default; `--on-demand` to override)" | STALE |
| N3 | `fleet.py:155` | Comment "timing boxes on demand (a reclaim would waste the timing)" | "timing boxes on spot by default (BP-3.10; reclaims refilled by `missing`); packed runs always spot" | STALE |
| N4 | BP:435 (§16) | "41 of 93 view clips on 'ours' CBR" | The earlier text gave the view as 94 with the envelope and 93 without, and the envelope is the default. State which view the 93 is, or use the envelope figure | AMBIGUOUS |
| N5 | BP:365-366 (§12) vs sealed-log:24 (entry 12) | §12: the 200 GiB floor, no exception | Entry 12 sets `NC_MIN_FREE_GIB=60` for this round. §12 or §15 should mention the round's exception. Entry 12 also mixes "60 GiB" (the setting) and "60 GB" | AMBIGUOUS |
| N6 | PROCEDURES.md:1847 | Example `NC_RELEASE_MEASURE=1 … --clipset ~/nanocodec-data/clips/heldout_2026-11-23.json` | Use the current set (`heldout_2026-11-23b`) in the example, or mark it as the legacy chart path | STALE |
| N7 | `fleet.py:12` | "`canaries.json` (the first N jobs of a fixed seeded draw)" | "the canary count; each box draws its canaries from its own shard" | STALE |

**Result: not clean. 7 items remain** (1 WRONG, 2 AMBIGUOUS, 4 STALE). All are small; N1 is the
only one where the text and the code disagree on behaviour.
