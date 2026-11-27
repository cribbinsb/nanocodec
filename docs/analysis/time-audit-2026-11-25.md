# Time audit, 2026-11-25

Owner: "Audit everything that takes more than 2-5 minutes, every possible delay; only keep
what is truly earning its keep." Every step or safeguard over ~2 min that an agent or the
owner waits on, with measured times, its catch record and a verdict.

**Sources.** The landing and verify logs (`run/land-logs`, 5 days, 360 verifies). The AWS
ledger (`work/aws/ledger.md`, rows since 2026-11-22). The build box occupancy records
(`work/aws/buildbox*/occupancy.jsonl`, 644 jobs). The bench run logs (`bench/bp4trial`,
`bench/gate.jsonl`). Tool output lines quoted in the agents' session transcripts: 325
landings, 190 pre-land profiles, 784 quick verdicts, and 851 distinct `[aws ...]` stage lines.

**Not in scope.** Another agent covers the dev measurement loop: the warm box held for the
whole run, staged key-share probes, the baked measurement image, the stream-hash score
cache, GPU scoring throughput, and phase timelines with a smoke run. Those rows are
marked *other agent* and were not changed here.

## Ranked by time saved per day

The "saved" column is an estimate unless it is marked as measured.

| # | Item | Time (median / worst, source) | Caught real problems? | Verdict | Change |
|---|---|---|---|---|---|
| 1 | `static_recovery.sh` built `cargo test --release` (fat LTO) | 249 s / 484 s on the box (40 runs, occupancy). **Measured after the change: 7 s warm, 172 s in a cold target.** The test itself runs in 0.7 s | The test checks streams, and the gate profile codes byte-identical streams to release | **MAKE LIGHTWEIGHT:** `--profile gate`, which shares the agent's own gate build | 230122c40. Saves ~4 min per run, ~1-2 h/day of agent waiting at ~25-30 runs a day |
| 2 | Build box idle end after 10 min | 42 of 124 relaunches in 10 days came 2-30 min after an idle end (ledger). Each paid 1.1 min to start (p90 2.9, n=53), plus cold builds from an image baked 2026-11-16 | Saves ~$0.15-0.20 per idle end; during the day the box was relaunched within minutes anyway | **MAKE LIGHTWEIGHT:** 30 min | 5bb12722f. ~4 relaunches a day avoided, ~10-30 min/day |
| 3 | Bench capacity search: a spill region over the session budget raised `BudgetExceeded` out of `open_session`, counted as a lost machine | Today: c8i refused 3 times in 2 min, the run gave up and was relaunched by hand (~5 min lost). D14's 10-min search never ran | Bug, not a guard | **FIX:** skip that region; refuse at once only when every region is over budget; size the budget for the spill regions D14 allows (1.5x home) | f18376caa |
| 4 | Hardware counters in bench: a `linux-tools` apt install in every timing machine's setup, and `perf stat` beside references and canaries | Seconds to a minute of setup per machine (not timed separately; the trial's machines logged "perf missing" anyway) | D19 (owner): no hardware-counter diagnostics | **REMOVE** | bfeb4e0e0 |
| 5 | `localscore --follow` exited when the shared GPU queue was busy (10-min wait) | A run's streams stayed unscored until someone restarted it | Not a guard | **MAKE LIGHTWEIGHT:** a follower waits and tries again | c815da9c3 |
| 6 | Build box image 9 days stale (last `nc-remote bake` 2026-11-16) | Every one of ~20 box starts a day begins with 9-day-old cargo targets. Cold release build 193 s (PROCEDURES); warm no-op 2.4 s | — | **MAKE LIGHTWEIGHT:** baked now; a daily automatic bake from `nc-remote reap` is still proposed | `nc-remote bake` 2026-11-25 15:07 (ami-0fb0f7765da4be442; ~58 GB used, ≈ $0.10/day of snapshot storage, the older image pruned once it is available) |
| 7 | Verify *tests*: browser module step | 483 s median, 1042 s worst (41 runs; 74 logs with the step), on the build box after landing | **0 failures in 74 runs.** It runs for every nc-dsp/nc-common change | **MAKE LIGHTWEIGHT (proposed):** for a kernel-only change, run the SIMD128 checkasm under node (one nc-checkasm WASM build) instead of both browser modules plus the corpora. Nobody waits on it, but it holds build-box slots ~1 h/day | Not done |
| 8 | Verify *tests*, the rest (check.sh, eval suite, syntax stress) | 201 s median, 890 s p90, 1882 s worst (176 runs) | **Yes, 17 real failures in 5 days:** unit tests in dependent crates (6), clippy on the box's AVX-512 lints (2), syntax stress (2), a page test (1), eval tests (several). Plus 6 infra failures (SSH mux, box git) | **KEEP.** Already after landing | — |
| 9 | Verify *streams* (landing conformance, 725 streams, every decoder) | 116 s median, 227 s p90, 429 s worst (184 runs) | 0 real mismatches in 5 days. All 5 failures were infra: vpxdec missing on the box, SSH mux | **KEEP:** correctness, not to be dropped. Already after landing and cached by stream hash | — |
| 10 | Pre-land checks (land.sh, in the lander's tree) | Whole landing 32 s median, 278 s p90, 772 s worst (325). Pre-land 138 s median, 354 s p90 (190). Unit tests are most of it: 108 s median, 626 s worst | **Yes, often:** unit-test refusals (84 quoted lines), clippy (54), fmt (10), all-features (4), outlined SIMD (2) | **KEEP.** Its "over the 90 s budget" note fires on most landings: noise, not a delay | — |
| 11 | Revert guard (land_guard.py) | Seconds | Yes: 44 refusals quoted, including two stale-tree squashes. It correctly flagged this audit's own deliberate revert of the counters | **KEEP** | — |
| 12 | `preset_identity.py` | 200 s median, 882 s worst (50 box runs) | Catch rate not measured (its output is not logged centrally) | **KEEP**, the only proof for exact changes. **Proposed:** `--codecs` defaults to the codecs the change reaches (land_scope's `conformance_codecs`), as landing conformance already does | Not done |
| 13 | Build box 12 h hard lifetime | Forces ~2 make-before-break replacements a day of a box that runs 17-26 h | Backstop for a lost PC | **KEEP:** a hard cap on spend. It costs far less than the idle ends (#2) | — |
| 14 | Per-agent build-box pools (`NC_REMOTE_POOL=buildbox-<name>`) | 18 pools, each with its own launch, cold targets and idle timer | No x86 need found | **Proposed:** x86 agents share the main box; pools only for Arm | Coordinator's call (briefs) |
| 15 | nc-remote sync (`rsync --checksum`), slot queue, overflow box | Sync 1 s median, 97 s p99 (1823 syncs). Slot wait p90 0 s, 1.0 h total over 644 jobs | Prevents stale-mtime builds and oversubscription | **KEEP** | — |
| 16 | `binary_for.sh` / build cache | 15 s median, 142 s worst (30) | Replaced 10+ min builds | **KEEP** | — |
| 17 | Dev measure: box launch+setup | 4.3 min median, 15.1 min p90 (21 rounds); "box ready" 1.1 min median | — | *Other agent* (baked image, held box) | — |
| 18 | Dev measure: waiting for a warm box being set up | Up to 705 s (291 progress lines) | — | *Other agent* | — |
| 19 | Dev measure: warm spend cap $6/h queues new boxes | Queued up to 810 s (`WAITING: ... $6.00/h cap`) | Cost cap | **MAKE LIGHTWEIGHT** (coordinator): $12/h; the account stop stays | c2f1e1c8c |
| 20 | Dev measure readiness (version stamps, 2 decode checks, decoder `ldd`, backend import) | Prepare 0.4 min median, 1.4 min p90 (228) | The decode check never tripped; "missing anchor decoder" did (23 quoted lines) | **KEEP:** cheap, and fails fast | — |
| 21 | quick (L1 + probes) on the box | 2.5 min median, 4.7 min p90, 12.2 min worst (784 verdicts). The owner's "tens of minutes" is the box setup and scoring around it (#17-19) | — | *Other agent* (staged probes) | — |
| 22 | Bench rehearsal gate keyed on HEAD | Each rehearsal costs 1-2 metal machines at ~6 min. Gate log: 56 records over 14 commits; 8 launches went ungated | Yes (5 catches: launch without `--type`, history builds, settle bug, no scoring box, cgroup lock) | **MAKE LIGHTWEIGHT** (coordinator): keyed on the git trees of tools/bench, eval/nceval and aws_session.py; principle 6 text updated | a76c09fd0 |
| 23 | Bench check set after every family (3 rounds, 11-13 s) | ~36 sets a machine: +49 % machine time on the trial plan | V6 refused boxes in BP-3 | **Owner decision:** implemented | a7268327d |
| 24 | Bench V29: an extra ffprobe decode of every H.264/HEVC/VVC stream | Doubles the decode CPU for those codecs (0.48 s against a 0.47 s decode, one 1080p stream). It also uses the ffprobe on PATH, not the pinned ffmpeg | No logged refusal | **Owner decision:** implemented | cbcfacf1d |
| 25 | Bench warm-up runs each binary's *longest* job serially | Minutes per binary per machine (not measured on a full run) | The warm-up has a reason; its length has none | **MAKE LIGHTWEIGHT** (coordinator): the shortest job | ea403b2b6 |
| 26 | Bench capacity: D14's 10-min search | c8i.metal: 0 of 18 launches today; c8g.metal: 13 of 52 | The mechanism, not a guard | **Owner** (D14) | — |

## Measured and estimated savings

- **static_recovery (measured):** 249 s → 7 s warm, 172 s cold.
- **Build box idle (estimated):** ~4 relaunches a day avoided, ~10-30 min/day across agents.
- **Bench capacity bug:** ~5 min and a manual relaunch each time it struck, and up to 30 min when a search ran out.
- **Not counted:** the hardware-counter setup and the localscore restarts.

**Spend.** No instance launched. Two `static_recovery` runs went on the running shared build
box (~3 min of it, ≈ $0.03).

## For the owner

Done on the coordinator's direction (2026-11-25): the warm spend cap (#19), the gate key
(#22), the warm-up (#25) and the image bake (#6).

Owner decisions, 2026-11-25, implemented:
1. **Check sets (D13, V6, V24):** one when a machine starts, judged at once against the
   series' machines (a failing machine does no work and is replaced). One more at its end,
   whose drift is reported. None between families; a family settles the moment it
   finishes (a7268327d).
2. **V29 (D19):** no extra decode. H.264/HEVC/VVC reordering is no longer read; their
   first-picture key comes from the NAL unit type. The AV1/VP9 header checks are kept
   (cbcfacf1d).
3. **D14's 10-min capacity search.** Kept as is (coordinator).
