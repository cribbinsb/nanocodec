# Benchmark procedure (BP-4.0, 2026-11-25)

This is the one normative procedure for every published compression and speed result of
nanocodec: the results site's charts and tables, the README figures, and any number quoted
outside the project. A result made any other way is not published.

**How to read this document.**
- Sections 1-18 state the procedure **as it is now**, and nothing else.
- Section 19 is the change log: every version from BP-1 to BP-4.0, with its date and reason.
  The full text of each earlier version is in git history.
- The evidence behind each rule is in:
  - [analysis/benchmark-method-2026-11-23.md](analysis/benchmark-method-2026-11-23.md), "the
    study";
  - [analysis/whole-box-timing-2026-11-25.md](analysis/whole-box-timing-2026-11-25.md), the
    whole-machine timing test behind BP-4's timing (sections 2 and 6);
  - the three hostile reviews and their responses in `docs/analysis/`
    (benchmark-procedure-review*, benchmark-procedure-response*);
  - `docs/results/sealed-log.md`.
- Each encoder's exact command, flag by flag with its reason, is the fairness record
  [benchmark-fairness.toml](benchmark-fairness.toml), rendered in section 17.

**Status.**
- **BP-4.0 measures CBR only.** The Fixed QP comparison is removed (owner, 2026-11-25): it was
  not well enough defined to be fair or useful. Each encoder's fixed-quantizer mode varies the
  quantizer by its own design (ours varies it over time; SVT-AV1 only with temporal layers;
  x264 and x265 not at all), so equal "QP" meant different things.
- **BP-4.0 (owner, 2026-11-25) changes how encodes are timed and where they are scored**
  (sections 2, 6 and 7). Its timing scheduler and its GPU scoring path are **not built yet**
  (T23, T25), and no run has been made under it: the trial (T24) comes first.
- Planning, staging, scoring on cloud CPU and reporting are built and in use:
  `tools/bench/plan.py`, `box.py`, `score.py`, `report.py`, `fleet.py` (section 13).
- The runs made so far are BP-3.x runs: "ours" CBR and Fixed QP, AOM-CTC CBR and Fixed QP, and
  VC-1K CBR. Their Fixed QP reports are withdrawn. They keep their version. Their CBR quality
  points stay valid. Their speeds measure a
  different thing (one encode on an idle core) and are never mixed with BP-4 speeds.
- **Nothing is published**: the publish freeze (2026-11-23) holds until the owner decides.
  The runs are release-measure runs, made with the buffer starting full (1.0 s, section 8). BP-3
  runs also ran ours at `--strict-buffer`, which BP-4 drops.

## Settled decisions (read before proposing any change)

These questions have been argued, often more than once, and the owner has decided them. **No
agent, review or coordinator reopens one.** A proposal touching one quotes its entry here, gives
new evidence the entry did not have, and goes to the owner as a single question. Each entry
gives the decision, the alternatives that were considered and why they were rejected.

| # | Question | Decision (date) | Alternatives rejected, and why |
|---|---|---|---|
| D1 | Which modes are measured? | **CBR only** (2026-11-25) | **Fixed QP:** not well enough defined to be fair or useful. Each encoder's fixed-quantizer mode varies the quantizer by its own design: ours over time, SVT-AV1 only with temporal layers, x264/x265 not at all. Matching the standards' low-delay test conditions, and "no layers plus each encoder's own cadence", were both tried the same day and dropped with the mode. |
| D2 | How is each encoder configured? | **Section 3's rule:** the encoder's own real-time defaults (for libvpx and libaom, their shipped real-time example settings). Only two constraints: no added latency (no lookahead, no reordering; temporal layers allowed) and the 1 s buffer. Plus four measurement conditions. Everything else default (2026-11-25) | **One shared structure for all (flat P):** it overrode encoders' own defaults (SVT-AV1's 2-level CBR structure). **Each reference's best-scoring settings:** that tunes references on our own data. **Bare `--rt` defaults for libvpx:** they lose its real-time AQ (+10-11 % BD-J) and are not what the project ships for real time. |
| D3 | libvpx/libaom maximum quantizer | **63 on main rows, 52 on `+drop` rows** (2026-11-25) | **52 everywhere:** 52 comes from real-time setups that rely on frame dropping. Without dropping it overshoots the low rungs 1.4-1.9x and drops 5-6 clips from their BD set, a selection that flatters them. |
| D4 | Our buffer bound | **`--strict-buffer`: ours holds the 1 s buffer strictly** (2026-11-25, final) | **Our 1.15 default bound** (decided 2026-11-22, restored and withdrawn 2026-11-25): x264, x265 and SVT-AV1 never pass one buffer, so ours passing it is arguable. Strict may cost ours a little and ends the argument. **"Nobody hard-enforces":** the same objection. |
| D5 | Initial buffer level | **Full, 1.0 s, for every encoder** (2026-11-25) | **A 600 ms "final" phase:** dropped. A second phase split charts, and it waited on an engine task. |
| D6 | Rate ladder | **300/500/800/1200 kbps at 720p30, scaled per clip** (2026-11-23) | **Higher rungs (2000-3000 kbps at 720p):** far above what a real-time modern-codec 720p encoder uses. **150/2400/4000:** dropped at BP-3. |
| D7 | Windows | **150 frames on every clip, plus 600 frames where the source allows (40 held-out, 4 VC-1K). Each clip coded once at its longest window; the 150 window is read from the start of that encode** (2026-11-25) | **150 only:** too little to show mid-sequence key frames. **150 + 300:** chosen against, because 600 shows mid-sequence keys properly. **Coding a clip twice:** wasteful. **A 600-frame headline:** too few long sources. |
| D8 | Key interval | **A key every 150 frames in every window,** matching dev's L3 and final check (600 f / key every 150) (2026-11-25) | **Every 300:** one mid-sequence key in 600 frames; dev's long windows decide at 150 (and 900 f / 300). |
| D9 | Baseline | **One common baseline, x264 medium; BD of every encoder against it** (2026-11-23) | **Pairwise or per-pair clip sets:** never. They break the one comparison every number shares. |
| D10 | Zero line tune | **x264 medium's best tune per metric:** PSNR, VMAF and J against its PSNR twin; MS-SSIM against its SSIM twin (2026-11-25) | **The default tune for VMAF and J:** x264's worst for J (the PSNR twin beats it by 12.5 % on J and 20.6 % on VMAF). The intent had always been the best twin per metric. |
| D11 | Clips lost from the common set | **Extend the anchor:** x264 medium also codes 0.5x the bottom rung and 2x and 4x the top (2026-11-23, 4x 2026-11-25). Curve rules as BP-3.14 | **Excluding clips or requiring 4 points or half overlap:** they shrank the headline to 41 of 94 clips and biased it. |
| D12 | Held-out set | **Stays at its 97 clips. No clip is ever excluded for having been used: each use is logged and disclosed** (2026-11-25) | **Exclusion and redraw:** exclusion is selection bias. **A 60-clip subset:** saves under $2 a series under BP-4 and moves levels by over 1 BD point for a third of families. |
| D13 | How speed is timed | **Amended again 2026-11-26 21:28 (owner): machines of one instance type are assumed equivalent ("we need to assume the same machine type gives same results or the whole thing breaks anyway"). A point's time stands on its own: valid when its machine's start check set passed (V6) and the encode passes V7/V8, whichever machine of the type coded the rest of its family; a family may span machines. A machine may serve several runs of its chip in turn (`--chain`); its check sets run at its start and final end, and a chained run's records name the run that holds them.** **Amended 2026-11-26 (owner): one longest-first queue of all of a machine's timed jobs, no filler (`halfload --queue mixed`); each encode is loaded by the run's own mixed jobs. One family at a time kept a 48-slot machine about 25 % busy.** Originally: **Loaded speed:** one encoder family at a time on every second physical core of one whole bare-metal machine per chip, longest first, with one check set when a machine starts (compared with the series' other machines; a failing machine does no work and is replaced) and one at its end if it survives (drift recorded and reported); families settle the moment they finish, so a reclaim loses only the family in progress (owner 2026-11-25: minimal overhead; whole machines do not drift, measured < 0.35 % over 40 min, half-load repeats <= 0.28 %) (2026-11-25) | **One encode per small shared box (BP-3):** slow and dear, and 8-35 % of times were refused. **Full load:** Graviton slows encoders unevenly (x1.01-1.50). **An idle machine:** nobody runs a real-time encoder on an idle server; each encoder competing with its own work is fair. |
| D14 | Timing machines | **Bare metal only: c8i.metal-48xl and c8g.metal-24xl; capacity sought for 10 minutes across zones and cheap regions; never another size** (2026-11-25) | **Non-metal sizes:** 2.3-3.1x the price. **Mixing sizes:** two sizes of one chip differ (measured 0.4-4.2 %). |
| D15 | Reference results | **Coded once, reused for every build of ours; re-timed when a run's check set moves more than 1 %** (2026-11-25) | **Re-running references every build:** about two thirds of the cost, for nothing. |
| D16 | Scoring | **On the PC's GPU** (the existing VMAFx service), with a 1 % libvmaf CPU parity sample; cloud CPU only as a labelled fallback (2026-11-24/25) | **Cloud CPU scoring by default:** about 97 % of earlier dev spend for work the local GPU does free. |
| D17 | Publication | **Only with both platforms complete, the speed measure stated, compliance beside every BD, and the limitations listed; the owner lifts the freeze** (2026-11-25) | **Publishing one platform:** a benchmark was once reported complete with Graviton missing. |
| D18 | Does the benchmark prove our streams are standard-conformant? | **No. Benchmarking and correctness are separate procedures. The benchmark is optimised for cost and time: each stream is decoded once, no cross-decoding** (2026-11-25; which decoder: section 7) | **Cross-decoding every one of our streams in every decoder (V10 as written at BP-3):** it re-proves what the landing check (`land.sh --verify`: every shipped preset bit-exact in both standard decoders and ours, cached by stream hash) already proves, and it made scoring 2-5x slower (110-245 frames/s against the GPU's 550). Raised and rejected more than once (owner, 2026-11-25: "BENCHMARKING IS NOT PROVING CORRECTNESS"). **A 2 % sample:** proposed the same day; rejected for the same reason. |
| D19 | Which checks run in every benchmark run? | **Only what the fair result needs: the shared conditions are met (V2, V4, V5, V7-V9, V24, V28-V31), the same input for all, and the timing machine is sound (check sets, V6). Correctness and tool verification are separate procedures, never steps of a benchmark run: the GPU scorer's agreement with libvmaf is `tools/bench/verify_scorer.py`, run once per scorer build (V27), and benchmark scoring only refuses an unverified build; ISA bit-exactness and stream conformance in the landing check (V10, V12 removed); no hardware-counter diagnostics; no extra decode for stream structure: V29 reads only what the stream's bytes give (AV1/VP9 frame headers, every codec's first-picture key), and H.264/HEVC/VVC output order is not checked** (2026-11-25) | **Per-pass V27 parity on a cloud box:** it re-proves the same scorer build every run (already proven on 210 streams, identical). **ISA bit-exactness at plan time (V12):** the landing check's job. **Hardware counters on every family:** diagnostics, not results. **V29 by a second decode of every H.264/HEVC/VVC stream (ffprobe packet positions):** doubled the scoring decode CPU of those codecs (no refusal logged); a per-build sample check was also rejected (owner, 2026-11-25). Owner, 2026-11-25: the benchmark is optimised for fair results quickly and cheaply, nothing else |
| D20 | Which reference rows carry metric twins (PSNR/SSIM tunes)? | **For now only the zero line, x264 medium. The other twins (x264 below medium, x265, libvpx and libaom `~ssim`) are a later addition, backfilled when time and budget allow** (2026-11-25) | **Coding every twin in the first run:** about 38 % more reference encodes, delaying the first results |
| D21 | Does ours run frame hold in the main rows? | **Yes: hold on with all-skip output (`hold.enable=1 hold.code_skip=1`); every frame is still coded** (2026-11-25) | **Hold off:** our own default disabled although the benchmark sets contain repeats (held-out 20 of 97 clips, 2.4 % of frames, mostly static CCTV and video calls; sealed log 23), and all-skip output is conformant (725/725). **A send-nothing `+drop` row for ours:** not needed |

## 1. Principles

1. **One run per encode.** Each encode runs once, on a box that gives a valid time. That run
   yields the stream (scored afterwards) and the time, so speed and quality come from the same
   encodes.
   - **The one exception is an encode-only re-timing** (section 6). It is valid **only because
     V11 proves it reproduced every scored stream byte for byte**: the timed encode then is the
     scored encode.
   - The same identity is what allows streams coded on one machine series (c8a) to be timed on
     another (c8i), and BP-3 streams to be timed under BP-4, with no new scoring.
   - Without V11 identity the report refuses.
2. **Refuse, never warn.** Every check in section 10 stops the run, or rejects the box or
   point, with a message naming the failure. A refused point is re-run, never waived.
3. **One source per fact.**
   - Commands come from the fairness record through `nceval.fairness.command`, and
     `tools/bench/plan.py` makes every job from it.
   - Builds are identified by SHA-256.
   - Clips are identified by the SHA-256 of their decoded frames.
   - Metrics are identified by tool version and model hash.
4. **Fixed before measuring.** The datasets, rungs, references and rules are frozen in each
   run's manifest (procedure commit) before its first encode. A rule decided after seeing
   results is logged as such in `docs/results/sealed-log.md` and in section 19.
5. **No manual steps.** Stage, run, score and report are commands of `tools/bench/fleet.py` and
   `report.py` (section 13).
6. **Rehearse, then run (BP-3.11).** No full run starts until `fleet.py rehearse` has passed on
   the same tools (the git trees of `tools/bench`, `eval/nceval` and `tools/dev/aws_session.py`,
   so a commit elsewhere keeps a rehearsal valid; 2026-11-25), and until `fleet.py preflight`
   has shown its checks and cost envelope and the envelope has been approved once
   (`preflight --approve USD`). `fleet.py run` enforces it: it refuses unless both are recorded
   in `bench/gate.jsonl` for the committed tools it runs; `--ungated REASON` overrides, printed and recorded. Progress, ETAs and spend come from
   `fleet.py status`, never from hand estimates.

## 2. Machines (authoritative)

| Role | Machine | How | Status |
|---|---|---|---|
| **Timing, Intel** | **c8i.metal-48xl** (bare metal; Xeon 6975P-C, Granite Rapids, 1 socket, 96 cores × 2 threads, 480 MB L3), **spot** | whole machine, half load: one encoder family at a time on every second physical core (section 6) | BP-4; scheduler to build (T23) |
| **Timing, Graviton** | **c8g.metal-24xl** (bare metal; Graviton 4, 1 socket, 96 cores, 36 MB L3), spot | the same, with the aarch64 builds | BP-4; scheduler to build (T23) |
| **Scoring** (decode and metrics) | the PC: decoding on its CPU (bounded workers), VMAF v1 and MS-SSIM on its GPU (the VMAFx CUDA server, section 7) | untimed | BP-4; bench path to build (T25). Until then, and as the fallback: c8g.8xlarge spot (c7i/c6i .24xlarge when no Graviton is free) |
| Superseded timing | c8i.2xlarge and c8g.2xlarge, one encode per box with timing references (BP-3.10 to BP-3.14); before them c8a.2xlarge and c8a.large | — | their speeds stay with their BP-3.x runs; their streams stay valid (V11) |
| Not run | Zen 3 (c6a), Jetson Orin Nano | — | owner, 2026-11-24: c6a is on demand, not offered in eu-north-1, and dearer; the Jetson is optional and free of AWS cost |

**One instance type per chip, bare metal, never mixed** (V28; owner, 2026-11-25).
- Two sizes of one chip do not agree. Measured 2026-11-25, alone, the same jobs:
  - c8g.metal-48xl ran 3.7-4.2 % slower than c8g.metal-24xl, nearly uniformly;
  - the non-metal c8g.24xlarge ran 0.4-1.3 % slower, graded by encoder.
- Bare metal is the cheapest whole machine per hour:
  - c8i.metal-48xl $0.96 against c8i.48xlarge's $2.95;
  - c8g.metal-24xl $0.41 against c8g.24xlarge's $0.94;
  - that is 2.3-3.1 times less.
- It also runs no hypervisor beside the encoders.
- Non-metal sizes are never used for timing.

**Getting bare-metal capacity.** Bare-metal spot is often short. On 2026-11-25:
- c8i.metal-48xl: none in eu-north-1 or us-west-2 (4 tries);
- c8g.metal-24xl: launched once in three tries.

So the driver retries for up to 10 minutes (owner, 2026-11-25):
- each round tries every zone of eu-north-1, then every zone of any region whose spot price
  for the type is within 1.5 × eu-north-1's (`aws_choose.py`'s spill rule). Clips may go to any
  of our AWS boxes (owner, 2026-11-20). The region never changes the machine type;
- a new round starts every minute;
- after 10 minutes without enough machines, the run reports what it got and asks the owner;
- it never falls back to another size or to non-metal. On demand (about 10 times the spot
  price) needs the owner's approval.

A run that has some machines keeps going on them; more join as capacity appears.

**Stability**, measured on whole machines (alone, full and half load interleaved over
40 minutes; 17 families × 8 clips;
[analysis/whole-box-timing-2026-11-25.md](analysis/whole-box-timing-2026-11-25.md)):

| | Graviton 4 (c8g.metal-24xl) | Intel (c8i.48xlarge: no bare-metal capacity that day) |
|---|---|---|
| Half load, step to step, worst family | 0.06 % | 0.28 % |
| Alone, 20 minutes apart, worst family | 0.18 % | 0.15 % |
| One encode's spread at half load (sd / median per job) | 0.33 % | 0.72 % |
| Half-load time against alone, range over 17 families | ×1.01-1.23 | ×1.00-1.07 |

- BP-3's small boxes, for comparison:
  - c8i.2xlarge: 1.4-1.5 % between boxes, with 8 % of timings refused;
  - c8g.2xlarge: 0.4-1.1 % between boxes, with 35 % refused.
- **Not measured yet:**
  - whether two machines of one type agree. BP-4 assumes it, as every comparison with past
    results already does, and checks it on every run (V6) and in the trial (T24);
  - Intel on bare metal. The Intel figures above are from c8i.48xlarge (the same chip and
    socket), and the trial measures c8i.metal-48xl.

**ISA tiers.**
- **On c8i** (lscpu: avx512f/bw/dq/vl/vbmi2/vnni/fp16): each encoder's runtime dispatch tier
  is read from its banner in the run's records (V21) and listed in the fairness record. x265
  runs `--asm avx512` on x86_64 and its own dispatch on aarch64 (`{X265_ASM}`).
- **Arm SIMD tiers on c8g** (NEON, SVE or SVE2), from each encoder's banner (V21), are listed in
  the record too.
  - x265 is built with SVE2 for Graviton.
  - **Ours: NEON, with SVE2 kernels for the residual (diff) and transform families since
    7464aac9e (2026-11-24, in build 889309c99); the rest is NEON.**
  - Each encoder uses its own best code, which is fair, but the asymmetry is stated beside
    every c8g figure.

## 3. Encoders and builds

### The configuration rule (owner, 2026-11-25; final, changed only by the owner)

Every encoder's command, ours included, follows these rules and nothing else. The benchmark is
CBR only. The
fairness record (section 17) applies them flag by flag.

1. **Its own real-time defaults.** Each encoder runs its own defaults for its real-time or
   low-delay mode:
   - x264 and x265 `--tune zerolatency`;
   - SVT-AV1 `--rtc 1 --pred-struct 1`;
   - libvpx and libaom: their project's own shipped real-time configuration, the settings of
     their real-time example programs (libvpx `examples/vpx_temporal_svc_encoder.c`; libaom
     `examples/svc_encoder_rtc.cc`, carried to `aomenc --rt` on the main rows), except settings
     that presuppose frame dropping, which the main rows forbid (the examples' max-q 52; the
     main rows run 63). The `+drop` rows run the examples' settings as shipped. Evidence:
     `docs/analysis/reference-fair-config-2026-11-25.md`;
   - openh264 `camera-realtime`;
   - Kvazaar's low-delay configuration;
   - ours, its preset.
2. **Only two constraints limit the encoder's coding choices:**
   - **No added latency:** no lookahead, and no picture references a later one (no
     reordering). Temporal layers are allowed: all their references are in the past.
   - **The 1-second buffer at the target rate:** a 1 s buffer, starting full, drained at the
     target rate (peak = target).
3. **Four measurement conditions make the numbers comparable.** They are not choices about
   coding:
   - the same input: 8-bit, the clip's frames, and one key frame at its start (the test
     window). After that, key frames are the encoder's own choice and fit the buffer like any
     other frame;
   - the same target rate per clip and rung (section 5);
   - one thread, so each encode runs on its one pinned core (section 6);
   - main rows code every frame. Frame dropping appears only in separately labelled `+drop`
     rows.
4. **Everything else is the encoder's own default:**
   - prediction structure and temporal layers;
   - QP and lambda cadence;
   - scene-cut keys;
   - quantizer range (libvpx and libaom: max 63);
   - AQ;
   - tools.

   A flag that changes a default is allowed only to impose rule 2 or 3, and it names which
   constraint in the record. No flag is added for quality, for speed, or to make encoders
   alike.
5. **There is no Fixed QP mode** (owner, 2026-11-25; see Status).

Applying the rule to an encoder is a mapping, not a new decision: the record shows each flag
and the rule it comes from. The mapping for every family, with its measured effect, is in
`docs/analysis/reference-fair-config-2026-11-25.md`. Changed families are coded again once per
platform.

- **Ours:** every shipped single-layer rung of H.264, HEVC, VVC, AV1, VP9 and AV2 (35 families
  at 0d9f32bc), one build per run.
  - Published runs use the release PGO build, trained on the PGO set of
    `eval/clipset/pgo_training_2026-11-23.json`, disjoint from every dataset (V3).
  - x86_64: `anchors/x86_64-opt-bp3/nanocodec` (commit 74aaa1fcd).
  - aarch64: `anchors/aarch64-graviton-opt-bp3/nanocodec`.
- **References**, each with LTO and PGO where its build allows, trained on the same set with
  the record's own commands (`tools/anchors/pgo_train.py`):
  - stores: `anchors/x86_64-opt-bp3` and `anchors/aarch64-graviton-opt-bp3`;
  - families:
    - x264 and x265, each preset ultrafast to medium, at the real-time tune and its PSNR and
      SSIM twins;
    - openh264 (CAVLC; CABAC twin), AQ on (its default);
    - SVT-AV1 M8-M13;
    - libvpx rt 7/9;
    - libaom rt 8/10 (main rows: `aomenc --rt` with its real-time example's settings; `+drop`
      rows: the example `svc_encoder_rtc` as shipped);
    - Kvazaar ultrafast-veryfast;
  - Labelled extra CBR rows: libvpx and libaom with their own frame dropping on (`+drop`).
- **Two twins are the same encoder and are shown once** (measured):
  - x265's PSNR and SSIM tunes are byte-identical at ultrafast and superfast.
- **Zero line:** x264 medium, with the tune matched to each metric (section 9).
- The commands, the shared constraints and the per-flag reasons are section 17.

## 4. Datasets

| Set | Manifest (committed) | Clips | Frames | Notes |
|---|---|---:|---:|---|
| ours | `eval/clipset/heldout_2026-11-23b.manifest.json` (sealed, release-only) | 97 (85 with four legal rungs) | 150 | **"Audited once, never tuned on"** for 84 clips, unseen for 13 (sealed log 1b-3). No pan class |
| VC-1K | `eval/clipset/vc1k_2026-11-18.manifest.json` (sealed) | 100 | 150 | video calls; numbers only (licence). **9 clips are in-sample:** each codec's six worst were lever-ranking sets on 2026-11-18/19 (`eval/clipset/tuned_on.json`, sealed log 28). The headline leaves them out (91 clips); the full 100 is shown beside it (owner, 2026-11-27) |
| AOM-CTC | `eval/clipset/aomctc_bp3.manifest.json` | 34: 12 low-delay (A3-A5) + 22 A2 (1080p) | 130 | **The 12 low-delay clips are in-sample:** they are the standards lens ours was developed on (`tuned_on.json`, sealed log 29). The headline is all 34, labelled; the view without them (A2 only) is shown beside it. 9 A2 sources 10-bit, converted to 8-bit by the pinned ffmpeg 8.0.1 (checked exactly against `(x + 2) >> 2`) |

- Each manifest lists, per clip:
  - the source and start;
  - the frames, size and rate;
  - the class and licence;
  - the source bitrate;
  - the SHA-256 of the decoded I420 frames.

  A run whose decoded frames do not match is refused (V4).
- "ours" sources are x264 re-encodes of the original videos (`eval/clipset/build_pool.py`). Their
  effect on the ranking is not yet measured (task T7).
- Datasets are disjoint from every PGO training set and tuning set, and from each other, by
  `nceval.content_id` (aliases, shoots, source paths; V3), except the in-sample clips recorded in
  `eval/clipset/tuned_on.json` (VC-1K 9, AOM-CTC 12): a clip a tuning or lever-ranking set used
  is listed there with the set, its date and what landed, and the V3 test requires every
  overlap between a dataset and a tuning set to be exactly those clips.
- **In-sample clips (owner, 2026-11-27).** A report can leave them out
  (`report.py --exclude-clips eval/clipset/tuned_on.json`: every family loses them, and the
  report lists them). The results site shows, per dataset, the headline and the other view beside
  it, with the clips and why: VC-1K's headline leaves its 9 out; AOM-CTC's includes its 12,
  labelled. The held-out set keeps its D12 rule (logged and disclosed, never excluded).
- A sealed set is read only by release measurements (`nceval.sealing`), and every reading is
  logged in `docs/results/sealed-log.md`.

## 5. Rates and quantizers

**CBR ladder (the only one):** 300 / 500 / 800 / 1200 kbps at 720p30.
- Scaled per clip: target = rung × (pixels / (1280×720))^0.75 × (fps / 30)^0.5 (the owner's
  real-time judgement; no outside source is claimed). `plan.scaled_kbps` implements it.
- Bits per pixel this gives:
  - "ours": 0.011-0.023 at the bottom rung, 0.043-0.090 at the top;
  - VC-1K: 0.011 at the bottom, 0.043-0.049 at the top;
  - 720p60: 0.008 at the bottom, 0.031 at the top;
  - 1080p60: 0.006 at the bottom, 0.025 at the top.
- A rung above half the clip's source bitrate is not coded. A clip without a source bitrate is
  refused.
- Nothing above 1200 kbps (720p30-equivalent) is measured.

**Anchor extension (for BD overlap only):** on every clip, x264 medium and its two tune twins
also code points past each end:
- 0.5× the bottom rung, and 2× and 4× the top rung (4× added by the owner, 2026-11-25, so that
  encoders whose curves lie above x264's quality range keep their clips; T30). Where a top point
  is over half the source rate, the anchor codes the cap if it lies above the top rung's target
  and has not been coded already, and nothing otherwise.
- These points enter only the anchor's BD curve: never compliance, speed, or the anchor's own
  row as a test family (`plan.anchor_points`, `report.curves`).

## 6. Timing

**What is measured: loaded speed (owner, 2026-11-25).**
- Each encode's CPU time (user + sys, `wait4`), while half of the machine's physical cores run
  the same encoder family.
- Each encoder competes only with its own work, the way a server full of streams runs it. It is
  not single-encode speed on an idle core, and is never called that.
- The difference is measured, and stated beside every speed figure. At half load, against alone
  (17 families, 2026-11-25):
  - Intel: ×1.00-1.07;
  - Graviton: ×1.01-1.23;
  - the fast presets of ours and SVT-AV1 slow most; on Graviton, our h264 and av1 veryfast reach
    ×1.23.

**Machine layout.**
- Physical cores come from `lscpu -p`, one hardware thread per core.
- **Slots:** every second physical core, from core 1. That is 48 slots on a 96-core socket.
  The other cores and every SMT sibling stay idle.
- **Core 0** holds the harness and the machine's own work: staging, decompression and copying
  records off. No encode runs there.
- **Pinning:** each encode is pinned to its slot's core by CPU affinity, inherited at the fork
  (V8). A cpuset move per spawn queued 48 slots on the kernel's cgroup migration lock, up to
  0.14 s each, which V7 counted as wall time (rehearsal, 2026-11-25).
- **Inputs:** clips on tmpfs, wholly resident (V5). Each binary's shortest job is coded once,
  untimed, as its warm-up before its first timed encode.

**Schedule: one family at a time, longest first.**
- **Plan before launch.** `fleet.py plan-timing` (T23):
  - takes each job's expected seconds from its family's last run on the series, else from the
    family's per-pixel rate;
  - orders the families by total expected time, longest first;
  - assigns whole families to the run's machines, each family to the machine with the least
    planned work;
  - writes the plan (machine, family order, expected seconds, expected wall and cost) into the
    run manifest before the first encode (principle 4). `preflight` prices it.
- **On each machine,** families run one after another.
  - A family's timed jobs go longest first to the next free slot.
  - When a slot has no timed job left while others still run, it codes an untimed copy of one
    of the family's jobs (filler). Every timed encode therefore runs under the same load, made
    only of its own family. Filler stops when the family's last timed encode ends.
  - A family never shares a machine with another family's encodes. A family never spans two
    machines.
- **The same machine runs the next family.** Setup (upload, decompression, warm-ups) happens
  once per machine.

**Machine check (V6), instead of BP-3's timing references.**
- Each machine runs the **check set** once when it starts and once at its end if it survives,
  never between families (owner, 2026-11-25: minimal overhead; whole bare-metal machines do
  not drift, measured < 0.35 % over 40 min, half-load repeats <= 0.28 %). A check set measured
  11-13 s on a c8g.metal-24xl with the AOM-CTC check clips (1080p and 360p, 130 frames), about
  $0.0015:
  - **encoders:** x264 medium, x264 ultrafast, our AV1 medium and our H.264 veryfast;
  - **clips and repeats:** 2 fixed clips each, three times;
  - **layout:** at half load, on all slots, one encoder at a time.
- The start set's geometric mean, relative to the series median of the run's machines' start
  sets, must lie within ±1 %. The driver judges it as soon as it is back, against the machines
  so far: a machine that fails does no work and is replaced. The report judges every machine's
  start set against them all; a machine that fails there has its families coded again.
- The end set is not a pass/fail check: each surviving machine's drift (end over start) is
  recorded and reported.
- **The check never rescales any time.** Its results are reported per machine: the run's
  measured agreement between machines and its drift.

**Per encode.**
- V7 refuses:
  - steal of 0.5 % or more;
  - wall over CPU by more than 3 % (and 20 ms);
  - a major page fault;
  - a slot core's clock below 95 % of the machine's median during an encode of 0.25 s or more.
- A refused encode is coded again in the same family pass; a family with more than 1 % refused
  is coded again whole.
- **Designed for failure** (spot reclaims; the owner's four rules in `docs/aws-cost-guide.md`).
1. **Copy off as units finish.** Every 30 s, and at the end of each family, the driver copies
   the machine's new records and streams to the PC.
2. **Act on the notice.** The machine reads the spot interruption notice between encodes. On a
   notice it starts no new encode and copies everything off.
3. **Retry only what is missing.** A family settles the moment it finishes: all its timed
   encodes are on the PC, from one machine whose start set passed, under the planned load
   (V24). A reclaim loses only the family in progress, which is coded again on a replacement.
   - A family cut short by a reclaim is coded again whole on a replacement machine of the same
     type; its partial times are discarded.
   - Families already settled are kept.
4. **A reclaim test.** `fleet.py rehearse` forces one reclaim mid-family (T22).

**Re-timing.**
- `fleet.py run --discard-streams` codes a scored run's jobs again for their times only,
  recording each stream's SHA-256 and size.
- Every stream must equal the scored one (V11). `report.py --timing` then takes the times from
  it.
- This lets BP-3 runs' scored streams be timed under BP-4 with no new scoring.

**Reference reuse (owner, 2026-11-25).**
- The references' binaries and commands are frozen. Their streams, scores and times are reused
  by every later run of ours (`report.py --parts`, V25).
- A new build of ours codes only our families: about a third of a full series.
- The references are coded again only when:
  - their command or build changes;
  - a dataset or the ladder changes;
  - **a run's check set (section 6) moves more than 1 % from the check sets of the runs that
    timed the references** (the geometric mean over its encoders, on the same instance type).
    Then the references are re-timed (encode-only, V11) before the run's report is made.

**Speed figure.**
- Per encoder: the geometric mean over the view's clips of megapixels per CPU-second over the
  curve's points, labelled "loaded speed (half load)".
- It is never a bare "fps" without picture size, and never a real-time claim.

## 7. Decoding, metrics and bits

- **Decoders (as BP-4 ran, corrected 2026-11-27).** Every scored stream is decoded; no score
  is read from an encoder's reconstruction.
  - **Our decoder decodes every family's streams, the references' included** (owner,
    2026-11-26: `NC_SCORE_OURS_DECODER`, the frozen 23c962d8b build; `score.py`). Score records
    name `decoder: "ours"`.
  - **Fallback:** where ours exits with an error, the stream is decoded by the codec's standard
    decoder and its record names it (`"standard"`): ffmpeg 8.0.1 for H.264, HEVC, VVC and VP9;
    ffmpeg with dav1d for AV1; avmdec for AV2. AV1 streams whose frames change size go to
    aomdec. A silent difference would not be caught by this fallback.
  - **Identity evidence:** ours against the standard decoders on a sample of the series' real
    streams (26 AV1, 6 AV2, 36 H.264, 24 HEVC, 12 VP9: `bench/decode-identity-2026-11-26.md`),
    all identical. That sample had no x265 stream, one SVT-AV1 preset (M10) and no VVC stream;
    the 2026-11-27 integrity audit decoded 13 more reference streams (x265 ultrafast, superfast,
    veryfast, faster and medium; SVT-AV1 M8, M12 and M13; openh264 CABAC; x264 ultrafast and
    faster; Kvazaar veryfast; libaom cpu10), all bit-identical to the pinned ffmpeg 8.0.1. Every
    VVC stream of the series is ours, which the landing check proves bit-exact against VVdeC and
    ffmpeg. 307 BP-4 score records carry no `decoder` field (77 in `bp4-aomctc-ours-c8i`, 230 in
    `bp4-vc1k-refs-c8g`).

  ffmpeg runs with `-fps_mode passthrough` (one frame per coded frame).
  - Each stream is decoded once. The benchmark does not cross-decode (D18).
  - A stream whose frames change size is decoded at each frame's size and upscaled by the
    pinned player model (`nceval.display`, bilinear).
  - A `+drop` row's dropped frames show the previous picture, placed by IVF timestamps, and
    are counted.
  - Every other stream must decode to exactly the clip's frames (V9).
- **Metrics:** libvmaf 3.2.1 (SHA-256 in the manifest), on every frame, at the clip's own
  size:
  - VMAF v1 (`vmaf_v1.0.16_3d0h.json`);
  - MS-SSIM (`float_ms_ssim`, BD in dB);
  - PSNR per plane, combined as (6 Y + U + V) / 8.

- **Where scoring runs (owner, 2026-11-25: locally where possible).**
  - **On the PC** (`fleet.py score --local`, `tools/bench/localscore.py`): `score.py` itself,
    with its `vmafx` backend, so the decodes, checks and records are a scoring box's.
    - The standard decoders (the run's `stage/bin`, ffmpeg the pinned 8.0.1) run on the CPU in
      the GPU scoring service's slice, at most 8 workers, each pass's clips decompressed into
      its own directory and deleted after it.
    - VMAF v1 and MS-SSIM run on the GPU through the one bounded VMAFx CUDA scoring service
      that the dev tools use (`nceval.metrics`, `nceval.gpuscore`: a place in the service and
      8 GiB of free GPU memory to start, or the pass is refused with the cloud path named; the
      patched build and its verification are in `docs/handoff/vmafx-gpu.md`). PSNR comes from
      exact per-plane squared-error sums with libvmaf's 60 dB cap.
    - `--follow MINUTES` repeats passes while streams arrive, so scoring runs beside encoding.
    - A pass's scores enter the series only after its V27 sample agrees.
  - **Agreement with libvmaf.** On 200 points, VMAFx agreed with libvmaf 3.2.1 on CPU:
    - VMAF v1: identical per frame;
    - PSNR-YUV: within 7e-15 dB;
    - MS-SSIM: within 3.9e-6.

    The bench path, checked 2026-11-25: 210 streams of the BP-3 AOM-CTC CBR run (6 clips, all
    35 of our families, all six codecs) scored locally and compared per frame with their
    stored cloud libvmaf scores: VMAF v1 identical on every frame, MS-SSIM within 4.7e-6,
    PSNR per plane within 5e-7 dB.
  - **Frame-0 fix.** Patch 0001 fixed a frame-0 initialisation race that had moved single
    frames on about 10 % of runs.
  - **Parity check (V27).** Each scoring pass also scores a 1 % sample of its streams (at
    least 20, covering every codec in the pass) with libvmaf 3.2.1 on CPU, on a small cloud
    Graviton box (`fleet.py score --ids`). VMAF v1 must match per frame at libvmaf's printed
    precision, MS-SSIM within 1e-5 and PSNR per plane within 1e-4 dB; a mismatch stops the pass
    and its scores are not merged (`v27-<tag>.json` lists each difference).
  - **The fallback** is cloud CPU scoring: Graviton c8g.8xlarge spot, else c7i or c6i
    .24xlarge (`fleet.py score`, in use). It is used when the GPU or the PC is unavailable,
    never silently: each score record names its backend.
  - **Cloud CPU details:**
    - decodes live in memory (the box's tmpfs), and the workers are capped by the memory they
      need;
    - each pass runs the current `score.py` and records its commit;
    - Graviton's decoders and libvmaf built for aarch64 (`--score-bin`) equal x86's frame by
      frame (`arm-parity.json`).
- **Bits:**
  - After the timed span, every stream loses its SEI and metadata units (`nceval.streamstrip`:
    H.264 6, HEVC 39/40, VVC 23/24, AV1 OBU 5, AV2 OBU 8/9). The stripped stream must decode
    identically, and the bytes removed are recorded.
  - Ours runs with `stream.encoder_info=0`, and x265 with `--no-info`.
  - x264's SEI cannot be switched off. It is stripped: 662 B per stream at the median.

## 8. Rate compliance (CBR)

- **One bucket shape for every encoder:** capacity 1 s, the decoder side starting full
  (1.0 s), every counted bit in. Each encoder is configured to it as far as its options allow
  (section 17). The 600 ms "final" phase of BP-3 is dropped (owner, 2026-11-25).
- **Ours holds the buffer strictly** (`--strict-buffer`; owner, 2026-11-25, final). The
  references run their own CBR rate control against the same configured buffer. In the
  2026-11-24 report, x264, x265 and SVT-AV1 never passed one buffer (worst 0.82-0.99), while
  libaom, libvpx, openh264 and Kvazaar passed it on 11-14 % of points (worst 2-3 buffers). How
  close each encoder holds the bucket is a result, published as follows:
- **Every point counts in the BD.** Compliance is published beside it per encoder, with no
  threshold or marker. Denominator: every coded point and frame of the view's clips.
  - **Primary, the bucket at the stream's own rate:** drained at the point's achieved average
    rate and holding one second of it. A point is plotted at its achieved rate, so an encoder
    that undershoots is not credited with a larger buffer (the results site's buffer page,
    `nceval.cbr_stats.own_rate_fullness`; `report.replay_own_rate`).
  - **Secondary, the bucket at the target:** drained at the target (as BP-3 to BP-3.11
    published).
  - **The worse of the two,** per point: the larger of the two peaks.
  - For each of the three: the median, 95th-percentile and worst peak in buffers, the share of
    points over 1.0 and over 1.15 buffers, and the share of frames over one buffer; the rate
    over target (median and mean); and the first key's size in seconds of the stream's own
    rate (median, p95, worst). Overall, per class, and per frame-rate band (≤ 20 fps,
    > 20 fps).
  - A second series of ours at another bound is not run until the owner decides (BP-3.13).
- Ours is printed "compliant by re-coding", with its re-coded share. No encoder's own rate log
  is read (V15).

## 9. Curves, clip sets and aggregation

**Per clip and encoder** (`tools/bench/report.py`):
1. **Monotone envelope:** per metric, a point is dropped when another point of the same encoder
   on the clip has a lower or equal achieved rate and a higher or equal quality. This applies
   identically to every encoder, ours and the anchor included, and the drops are counted.
2. **Curve rules** (`nceval.bdj`, the one definition the dev screens share). The aim is a BD
   for every encoder on every clip, with nothing excluded (owner, 2026-11-25, BP-3.14):
   - at least two points after the envelope (`MIN_POINTS = 2`). The interpolation is defined
     from two, so a clip whose ladder has fewer than four rungs under the 50 %-of-source cap
     is usable;
   - a point whose stored score lacks a metric is not a point of that metric's curve. It
     stays in the curves of the metrics it has, and the report counts such points per family;
   - a point whose quality does not rise keeps the lower-rate one;
   - a fall beyond 0.05 dB PSNR, 0.1 VMAF or 0.0005 MS-SSIM refuses the curve. With the
     envelope this never acts: a point with more rate and no more quality is dominated, so
     what the envelope keeps rises strictly. It acts on the `--no-envelope` view only.
3. **BD-rate** (`nceval.bdrate`): PCHIP in quality space over whatever quality range the
   curve and the zero line share, with no extrapolation and no minimum overlap. There is no
   BD only when the two ranges do not touch at all; the report then says which curve lies
   above the other.
4. **Overlap, shown not excluded:** beside each family's figure the report gives the median
   overlap (the shared range as a fraction of the smaller curve's quality span, the thinnest
   of the clip's BDs) and the number of clips under the old line
   (`THIN_OVERLAP_FRACTION = 0.5`).

**Zero line:** x264 medium.
- BD-PSNR-YUV is against `--tune zerolatency,psnr`.
- BD-MS-SSIM is against `--tune zerolatency,ssim`.
- BD-VMAF is against `--tune zerolatency,psnr`.
- **BD-J** is the mean of the three BDs, each against `--tune zerolatency,psnr`.
- Each metric's zero is x264 medium's best tune for that metric, measured (owner,
  2026-11-25): on the BP-3 "ours" CBR points the PSNR twin beat the default tune by 20.6 % on
  VMAF v1 and 12.5 % on J (docs/analysis/bp4-adversarial-review-2026-11-25.md, finding 1).
- Every chart names its zero's tune.

**Clip sets:**
- **The view:** the clips where the zero line's three curves are well formed.
- **A family without a BD on a clip loses only that clip**, counted. Every clip the common
  set lacks is listed with the families that lose it and the exact reason: one point after
  the envelope, ranges that do not touch (and on which side), or a missing metric.
- **The headline is the common set:** the view's clips where every non-`+drop` family is well
  formed. Every encoder is averaged over the same clips, and the set's size is stated.
  `+drop` rows are shown on their own clips, labelled.
- **Any claim about ours** quotes its BD over all its anchor-valid clips (the view less its
  own malformed curves), with the common-set figure beside it. The more flattering of the two
  is never shown alone.
- **Secondary, labelled:** each family's mean over its own clips.
- **Exclusion check** (printed for each family that loses clips):
  - its rate over target on the lost and kept clips;
  - our BD-J on those clips.

  On the first run, the lost clips were hard low-rate clips where the references overshoot,
  and dropping them raised our headline. In BP-4 ("ours", c8i) it is the other way: 8 of the 9
  clip-windows lost from the common set are lost wholly or partly because our curves lie above
  x264's quality range, so dropping them lowers our headline; the ninth is lost to one rising
  point of our AV1/AV2 MS-SSIM curve. The site shows each family's own-clip figure beside the
  common set.

**Aggregation:**
- **Set mean:** equal weight per clip. The mean of the per-clip log rate ratio,
  log(1 + BD/100), reported as a percentage. Per-class means appear in a separate table with
  clip counts; classes with fewer than 8 clips are stated as thin.
- **Intervals:** a 95 % bootstrap over clips, 10,000 resamples, seeded, for every BD and speed
  figure. A cluster bootstrap over source collections is a sensitivity figure.
- **Front:** on the common set, the families no other family beats on both BD-J and speed.
- **Calling a difference:** only when the bootstrap interval of the per-clip difference
  excludes 0. Headline claims are a short list fixed before the run, held with a Holm
  correction.
- **Outliers:** none removed. A clip leaves only by a rule above or a refused check, and it is
  listed.

## 10. Validity checks (each refuses)

| # | Check | Where |
|---|---|---|
| V1 | Manifest frozen: procedure commit, dataset manifests, ladders, families, binaries' SHA-256 | plan, stage |
| V2 | Every binary's SHA-256 equals the manifest's on every box | box |
| V3 | Datasets disjoint from every PGO training set, tuning set and each other (`nceval.content_id`); the only overlaps with tuning sets are the in-sample clips of `eval/clipset/tuned_on.json` (`eval/tests/test_content_id.py`) | plan, tests |
| V4 | The clip's decoded frames hash and frame count match the manifest; the staged input holds exactly the job's frames | box, scorer |
| V5 | Input on tmpfs and wholly resident before the encode | box |
| V6 | One CPU model for every machine of a series; each machine's start check set (section 6) within ±1 % of the series median of the machines' start sets (judged by the driver as the set comes back: a failing machine does no work and is replaced; and by the report over every machine); the end set's drift is reported, never a refusal (D13); times are never rescaled | fleet, report |
| V7 | Steal < 0.5 %; wall ≤ 1.03 × CPU (and CPU + 20 ms); no major page fault (box, per encode); the timing CPU's median clock during each encode of 0.25 s or more at least 95 % of its box's median, where the CPU reports one (report; a slow encode's time is refused and the job coded again) | box, report |
| V8 | Each encode confined to its slot's core (CPU affinity, checked from its Cpus_allowed_list during the encode); slots on every second physical core, no slot on core 0 or on an SMT sibling; while a family is timed, every slot runs that family (timed or filler) and nothing else | box |
| V9 | Encoder exit 0, stream non-empty, decoded frames = input frames (dropped frames only in `+drop` rows, placed by timestamp) | box, scorer |
| V10 | Removed from benchmark runs (D18): the benchmark does not cross-decode. Conformance of every stream we ship is proved by the landing check (`land.sh --verify`), a separate procedure. `score.py --cross-sample` remains for an explicit investigation only | — |
| V11 | Re-timed and re-coded streams equal the scored streams byte for byte | report |
| V12 | Removed from benchmark runs (D18): ISA-tier bit-exactness is the landing check's job (checkasm, `tools/ci/isa_identity.sh`) | — |
| V13 | Every planned point has a time, a stream and three metrics, over all of a dataset's runs | report | **Amended 2026-11-26 (owner): only a missing stream or score refuses; a point without a valid time is listed, counted for quality and BD and left out of its family's speed.**
| V14 | The curve and clip-set rules of section 9; lost clips, reasons and the exclusion check printed | report |
| V15 | One bucket replay for every encoder; no encoder's own rate log read | report |
| V16 | Provenance complete for every point (section 11) | report |
| V17 | The history subset tracks the full sets on the current build (section 14) | report |
| V18 | Every command is `nceval.fairness.command`'s for its family; `eval/tests/test_fairness_record.py` passes. A reused point (a mixed report) is refused unless its command equals, byte for byte, the command a plan made at the report's tools commit gives the same job | plan, tests, report |
| V19 | SEI and metadata stripped from every stream; bytes removed recorded | box |
| V20 | The report's statistics constants equal section 9's text (a test compares them) | tests |
| V21 | Each encoder's banner shows the dispatch tier the record names | box records |
| V22 | A sealed run uses no excluded clip, runs a frozen clean commit, and has a sealed-log entry (`plan.py --sealed`) | plan |
| V23 | Every point is scored from a stream its own machine series coded (score records carry the stream SHA-256 and series) | report |
| V24 | A family's times count only when all its timed encodes ran on one machine whose start check set passed, under the planned half load of its own family; a family settles the moment it finishes (D13); an encode is settled only when its record (with the stream's SHA-256, and the stream unless a re-timing) is on the PC | fleet, report |
| V25 | One recorded build identity: every binary of every quality and timing run has the same SHA-256 across the runs, each encoder's hash has a build record (our commit, version and build; each reference's source commit, flags, LTO/PGO and toolchain), and every run records its tools commit; the report opens with this provenance block. A mixed report (our families re-measured on a new build, the other families reused: `report.py --parts`) checks it per family: each encoder has one identity across the parts, and every binary no ours-part uses (the references, the decoders, the scorer) is byte-identical in every part that records it; only our superseded build may differ, and the block does not list it | report |
| V26 | A point whose stream fails the codec's standard decoder (V9) and is deterministic (coded more than once, always to the same SHA-256) is a failure of that encoder: listed at the top of the report with its stream hash and counted against it (the point is absent, so its clip loses the family's curve where the minimum of points is missed). It is never dropped silently, and never waived for one encoder without the same record for another. The stream and every decoder's output are kept (bench/encoder-failures/). An encode that fails with the same error on every box it was given (the queue quarantines it after two) is the same kind of failure, without a stream | report |
| V27 | Not a benchmark check (D19). Benchmark scoring refuses a VMAFx build absent from `bench/proven-scorers.json`; a build is entered there by the separate procedure `tools/bench/verify_scorer.py`, run once per build (a 1 % sample, at least 20 streams and every codec, scored again by libvmaf 3.2.1 on CPU: VMAF v1 per frame at printed precision, MS-SSIM within 1e-5, PSNR per plane within 1e-4 dB) | verify_scorer.py |
| V28 | One instance type for every machine of a timing series (from instance metadata), as section 2 names it | box records, report |
| V29 | No picture is coded ahead of its display, read from each stream's bytes with no extra decode (`tools/bench/streamcheck.py`): AV1/VP9 frame headers have no `show_frame = 0` and no `show_existing_frame`. A failed probe refuses too. H.264/HEVC/VVC output order is not read (D19: it needs a second decode of every stream); their points are listed per family. AV2's headers are not parsed: its points are listed as counted only | scorer, report |
| V30 | Each stream's first frame is a key (the window starts with one; H.264/HEVC/VVC: the first coded picture's NAL unit type is IDR/IRAP); AV1/VP9 keys per stream are reported per family (scene-cut keys are the encoder's own choice, D2) | scorer, report |
| V31 | Each encode used one core: CPU time at most 1.10 x wall time (+20 ms) | report |

## 11. Provenance, versioning and sealing

- **Each point's records** (encode, score, job):
  - procedure commit;
  - family and exact command;
  - binary SHA-256;
  - clip and decoded-frame hash;
  - target;
  - machine series, box and instance;
  - CPU, wall, steal, faults, working threads and allowed CPUs;
  - the slot core, the family's load (slots busy at its start) and the machine's check sets;
  - stream bytes, SHA-256, per-frame bytes and SEI bytes removed;
  - decoder;
  - metrics, per frame and pooled.
- **Versioning:** the procedure version is this document's (BP-4.0). A run's manifest carries
  its commit. A patch applied to a staged run before any of its scores was read is recorded in
  the run manifest's `patches`.
- **Sealing:** every reading of a sealed set, every relabelling, and every rule set after
  seeing results is an entry in `docs/results/sealed-log.md`.
- **Reproduction:** AOM-CTC is public; "ours" can be rebuilt from its per-clip sources where
  their licences allow; VC-1K is numbers only. The stream of any point can be re-coded from its
  command, build and clip, and must match its SHA-256.

## 12. Retention

- **CBR streams are kept.**
- BP-3's Fixed QP streams were deleted once scored (`fleet.py prune`); BP-4 codes none.
- Staged clips are links to their kept sources (`fleet.py stage`).
- Runs live under `~/nanocodec-data/bench/<run>/` (`tools/dev/retention.toml` "bench").
- **Disk floor:** `fleet.py run` and `score` refuse to start when the PC has less than the
  project's 200 GiB free (`NC_MIN_FREE_GIB`, as `eval/nceval/diskguard.py`). A 60 GiB
  exception for this round (sealed-log entry 12) was withdrawn unused (entry 13).

## 13. Scripts

| Step | Command | Does |
|---|---|---|
| plan | `tools/bench/plan.py --clips MANIFEST --mode cbr [--sealed --frozen-commit C]` | one job per encode from the record, with the ladder, scaling, anchor extension, source cap and Fixed QP table; refuses clips without a source rate and excluded sealed clips |
| stage | `fleet.py stage RUN --jobs J --bin DIR --cfg DIR --clips DIR` | freezes the run: jobs, manifest (binary hashes, phase, commit), binaries, clips, scripts |
| run (BP-3 layout) | `fleet.py run RUN --series S --boxes K [--fleet-budget USD] [--discard-streams] [--packed N] [--type T --region R --on-demand]` | in use for BP-3 runs and packed untimed encodes; BP-4 timing uses `--layout half` |
| plan-timing (BP-4, to build: T23) | `fleet.py plan-timing RUN... --series S --machines M` | section 6's plan: family order, whole families to machines, expected wall and cost, into the manifest |
| run, BP-4 layout (to build: T23) | `fleet.py run RUN --series S --layout half` | whole machines of the series' one type; families one at a time, longest first, filler, check sets, family-level settling (section 6) |
| score, local | `fleet.py score RUN --series S --local [--follow MINUTES]` | `score.py --metrics-backend vmafx` on the PC (`localscore.py`): decoding on its CPU, VMAF v1 and MS-SSIM on its GPU scoring service, the V27 sample on a small cloud box; passes merged only after V27 |
| score | `fleet.py score RUN --series S --boxes K` | in use; under BP-4 the fallback: `score.py` on spot scoring boxes (Graviton c8g.8xlarge by default, x86 c7i/c6i as fallback), for streams without a valid score; slots replace lost boxes |
| preflight | `fleet.py preflight RUN...` | disk, quota and cap, the PC's address, prices and reclaim rates by zone, the cost envelope |
| rehearse | `fleet.py rehearse RUN... --out DIR` | the whole pipeline on whole sample clips of each run, with a forced reclaim |
| status | `fleet.py status RUN... [--group] [--since ISO]` | measured progress, rates, ETAs from remaining expected seconds, the benchmark's own spend |
| missing | `fleet.py missing RUN... --series S --out J` | the jobs without a valid record (a last resort) |
| prune | `fleet.py prune RUN --series S` | section 12 |
| report | `report.py RUN... [--timing RUN... --timing-series S] [--no-envelope]` | checks and tables of sections 8-10, the front, both clip-set views. A re-timing applies only to one quality series, from runs that hold exactly the named timing series, with V11 identity for every point |

`aws_session.py` launches every box (tags, ledger, lifetime and budget caps). When a poll
fails it adds the PC's current public address to the box's SSH rule. `fleet.py run` and
`score` drive all their boxes from one local process, one thread each, with shared AWS clients
and each box's output in its own log. They refuse to start outside a memory-capped scope
(about 70 MB per 20 boxes' clients, measured).

## 14. History set

- **The subset:** 30 clips, 10 from each dataset (`eval/clipset/history-set.json`, drawn once by
  `eval/clipset/history_select.py`, never redrawn).
  - It is stratified over class, resolution, frame rate, SI/TI, noise and low light, and
    source.
  - Two strata of the written rule are not applied: rate behaviour, and SI/TI for "ours"
    (sealed). The file says so.
- **The "ours" clips of the daily history series (owner, 2026-11-25):** 21 clips, the ten of the
  draw and eleven added (`ours_extension` in the same file; `tools/bench/history_clips.py` is
  the one definition every history tool reads).
  - Why: on the ten, our families read 3.4 to 4.8 BD-J points worse than on the full 97 and the
    references up to 3.3 points better, so our front looked 2.2 to 7.1 points closer to the
    references' than on the main page (builds 74aaa1fcd and 889309c99, both modes).
  - Rule: clips are added one at a time, each the clip that brings the fixed reference
    encoders' BD-J on the subset closest to their full-set figures, among the clips whose
    content class is still under its share of the 97. No result of ours enters the choice.
  - Measured on the 21: our families' mean is within 0.2 points of the full set and the
    references' within 0.3 (both builds, both modes); the fronts' gap reads 0.8 to 1.4 points
    wider than on the full set. Random class-stratified subsets of 21 are not this close
    (median 1.5 to 1.6 points for ours at 20 to 25 clips).
  - The added clips' zero line comes from the main record's runs (the same x264 build and
    commands), so the daily runs code our families only on them.
  - Record: `~/nanocodec-data/bench/history-days/subset-2026-11-25/`.
- **Mode:** CBR on the full ladder.
- **Machines:** timed as section 6, on the same types as the main series (c8i.metal-48xl and
  c8g.metal-24xl), so history and the main results measure the same thing. **The history builds
  are re-timed under BP-4** (owner, 2026-11-25): encode-only, their streams kept (V11), so every
  history point carries a BP-4 speed (T31).
- **Long windows (owner, 2026-11-25):** the history clips also run 600-frame windows, one with a
  single key and one with a key every 150 frames, beside the 150-frame view. Key-frame and
  start-up behaviour is judged there (T31).
- Earlier release and daily frames are deferred.
- **Tracking check (V17):** on the current build, every rung's subset-minus-full BD-J offset is
  within 1 point of the median offset, with the same front membership. Owner decision
  2026-11-27: a failure no longer withdraws the graph; the graph is shown with the failed check
  and every family's offset beside it, to be read for the trend across days, not the level, until
  the cause is established and fixed. The results site applies it itself
  (`tools/bench/preview_history.py`, `tracking_check`). BP-4 fails it on both chips (37 of
  64 families outside, median offset +5.2 / +5.0 points; our families' mean offset +5.5 / +5.2,
  the references' +2.0 / +1.8; the front differs; common set 18 clip-windows, 9 clips), so the
  graph is withdrawn.

## 15. This round's plan (owner, 2026-11-25)

1. **Build** the BP-4 scheduler (T23) and local scoring (T25).
2. **Trial (T24):** "ours" CBR on two c8g.metal-24xl machines, and on one c8i.metal-48xl
   when capacity allows. The same subset of families goes
   on both, to measure agreement between machines of one type, and the rest is split between
   them. Report the plan's estimate against the actual wall time and cost.
3. **The full series, both chips,** under BP-4:
   - **datasets:** "ours", AOM-CTC and VC-1K, CBR;
   - **families:** every family of the record;
   - **history:** the history subset's builds.

**Estimated cost** (from the staged Graviton plan's 124.9 encode-hours a chip, × the measured
average half-load slowdown, ÷ 48 slots; about 5 % for each family's tail and 10 minutes of
setup a machine). These are estimates until the trial measures them:

| | Machine-hours | Spot cost | Wall time, 1 machine | Wall time, 5 machines |
|---|---|---|---|---|
| Graviton (c8g.metal-24xl, $0.41/h) | about 3.0 | about $1.25 | about 3 h | about 45 min |
| Intel (c8i.metal-48xl, $0.96/h) | about 2.8 | about $2.70 | about 2.8 h | about 45 min |
| Scoring | — | $0 on the PC's GPU (V27's sample: well under $1) | — | — |

BP-3, for comparison:
- Intel: $11-24 of encoding, measured;
- Graviton: $29 of encoding plus $4.60 of scoring, pre-flight, never run in full;
- 8 % (Intel) and 35 % (Graviton) of the timings refused.

**Re-timing instead of re-coding.** Where a BP-3 run's build is still the one to report, its
scored streams can be timed under BP-4 (`--discard-streams`, V11), with no new scoring.

**Deferred:**
- the c6a (Zen 3) series;
- the Jetson series.

**Before the full series (owner, 2026-11-25):**
- **The held-out set stays as it is: 97 clips** (owner, 2026-11-25). No clip is ever excluded
  for having been used: a use is logged in `docs/results/sealed-log.md` and disclosed instead.
- **The tuning pool gains the same strata** (low-fps camera; 50/60 fps), drawn disjoint from
  every sealed set (T32).

**Publication (owner, 2026-11-25).** The publish freeze stands until the owner lifts it. A
result is published only with:
- both platforms complete;
- the speed measure stated ("loaded speed, half load");
- compliance beside every BD;
- the known limitations of section 16 listed beside it.

**Budget:**
- The AWS cap is $800, and the account's stop line is $790. $769.03 was spent at 2026-11-25
  08:45Z.
- The trial and both full series fit within it at the estimates above. `preflight`'s envelope
  is approved by the owner before each full run (principle 6).

## 16. Known limitations

- **Machines:** Intel Granite Rapids and Graviton 4, reported separately. CPUs order encoders
  differently: in the study's single-encode runs, Zen 4 and Intel moved encoders by up to
  ±11 % against Zen 5. No AMD series is in this round.
- **Loaded speed, not single-encode speed** (section 6).
  - On Graviton, half load moves encoders against each other by up to 22 % relative to an
    idle machine (×1.01-1.23).
  - Ours and SVT-AV1's fast presets are the most affected; that is the measure, stated, and
    our encoder's cache sensitivity is open work (T28).
  - On Intel the range is ×1.00-1.07.
- **Agreement between machines of one type** is assumed. It is checked on every run (V6), and
  each report states it.
- **Rates:** nothing above 1200 kbps (720p30-equivalent) is measured. A dev diagnosis at
  600-4000 kbps (`docs/analysis/upper-rates-diagnosis-2026-11-23.md`, four 720p pool clips)
  measured a reversal for our fast presets there: AV1 veryfast +5.4 % and H.264 veryfast
  +41.9 % against x264 medium.
- **Slow end:** no reference is charted slower than x265 medium, SVT-AV1 M8, libaom cpu8 or
  libvpx cpu7, so our slowest presets (VVC slow to veryslow) have no reference beside them.
- **Tuning:** our presets are tuned at these rates; no reference is. Screens rank at 150 frames, but
  changes to rate, keys, cadence or key quality, and every preset bundle's final check, are
  decided on long windows (`nceval.levels`, `nceval.key_shares`): 600 frames with a key every
  150, and 900 frames with a key every 300.
- **In-sample clips:** 9 VC-1K clips were lever-ranking sets for every codec on 2026-11-18/19
  (the VC-1K headline leaves them out; the full set is shown beside it), and AOM-CTC's 12
  low-delay clips are the standards lens ours was developed on (labelled; A2 alone is shown
  beside it). The rest of VC-1K shares its source with the clips whose problems were chosen.
- **Start-up:** every window starts with a key frame and a full 1 s buffer; each "ours" clip
  counts once, and only 40 of its 97 clips also have a 600-frame window beside the 150-frame one. Our first key is 0.35-0.58 s of the
  channel (median) against 0.09-0.21 s for x264, x265 and SVT-AV1 ("ours", c8i); capping it
  costs ours 5.0-7.8 % J by our own measurement (`docs/analysis/first-key-2026-11-25.md`). The
  references' start-up effect is not measured. The site shows each family's first key beside its
  BD and the 600-frame windows' BD beside the headline.
- **Timing machines:** on c8i "ours", our families and the references were timed on two machines
  whose start check sets used different clips, so their agreement was not measured.
- **Clip sets:**
  - the "ours" set was audited once, and read for quality by release measurements on
    2026-11-23 to 11-26 while our encoder changed; no lever is recorded as chosen from it;
  - it has no pan or moving-camera class;
  - pan, lowlight and drone are thin.
- **Common set:** in BP-4 (both chips) 92 of the 97 clips on "ours" (128 of 137 clip-windows),
  99 of 100 on VC-1K (90 of 91 without its in-sample clips) and 34 of 34 on AOM-CTC. On
  "ours", 8 of the 9 lost clip-windows are lost wholly or partly because our own curves lie
  above x264 medium's quality range (section 9). Each report lists the lost clips with the
  reason. Both the common-set and the own-set figures are published.
- **Thin overlaps and two-point curves** count in full since BP-3.14. A BD over a narrow
  shared range, or through two points, is a weaker estimate than one over four points and a
  wide range; the reports show each family's overlap so that this is visible.
- **Sources:**
  - variable-frame-rate sources are converted to a constant rate by the pinned ffmpeg
    (identical input for every encoder);
  - "ours" sources are x264 re-encodes;
  - CrowdRun and OldTownCross carry a standards-only licence notice and are flagged.
- **Encoders:**
  - one thread, 8-bit, no reordering for everyone; the prediction structure is each encoder's
    own real-time default (SVT-AV1 runs its 2-level structure, Kvazaar `lp-g8d4t1`);
  - scene-cut keys are not uniform: x264 and openh264 choose their own, the other references
    run without them (section 17, `scene_cut`);
  - VVenC is not charted: its rate control is VBR with a look-ahead, so it has no real-time CBR
    mode, and BP-4 has no Fixed QP mode.
- **Window:** the 150-frame unit includes one key frame.
- **VMAF v1 at native size** is a relative metric, not comparable with published 1080p VMAF.

## 17. Fairness record (rendered from docs/benchmark-fairness.toml)

### Shared constraints

| Constraint | Value |
|---|---|
| rate_kbps | the only ladder: 300 / 500 / 800 / 1200 kbps at 720p30, target = rung x (pixels / 1280x720)^0.75 x (fps / 30)^0.5, capped at half the source bitrate (owner, 2026-11-23) |
| buffer_ms | 1000 |
| initial_level_ms | 1000: the buffer starts full for every encoder (owner, 2026-11-25) |
| optimal_level_ms | 600 |
| peak | equal to the target (no encoder gets a higher drain rate) |
| frames | 150 ("ours", VC-1K); 130 (AOM-CTC, the CTC's count) |
| keyint | 150: a key every 150 frames in every window (D8; tools/bench/plan.py --keyint, KEY_INTERVAL) |
| bframes | 0: no picture references a later one (no reordering, past references only); VVenC's low-delay B slices reference past pictures only (verified, VVenC entry) |
| lookahead_frames | 0 |
| scene_cut | not uniform (audit 2026-11-27, finding 10): x264 and openh264 choose their own scene-cut keys in CBR (procedure section 3, rule 4); x265 (--no-scenecut --min-keyint), SVT-AV1 (--scd 0) and libvpx and libaom (--kf-min-dist = --kf-max-dist) run without them; ours runs its preset defaults. Keys per stream are reported per family (V30) |
| threads | 1 |
| frame_skip | off on every main row (no encoder sends nothing for a frame); libvpx and libaom also have labelled +drop rows with their own frame dropping on (CBR only); ours codes a held frame as an all-skip frame (engine task T1; until it lands, hold off) |
| resize | allowed where it is the encoder's own recommended real-time setting: ours on (AV1), the references as their projects' real-time examples set it (off); size-changed frames are scored after one pinned upscale (nceval.display, bilinear centre-aligned) for all |
| sei | every stream loses its SEI NAL units (H.264 6, HEVC 39/40, VVC 23/24), AV1 metadata OBUs (5) and AV2 metadata OBUs (8/9) after the timed encode (nceval.streamstrip); decoded frames are checked identical in both standard decoders and ours; bytes removed recorded per point; x265 also runs --no-info |
| zero_line | x264 medium in its best tune for each metric (owner, 2026-11-25): BD-PSNR and BD-VMAF and the composite BD-J against --tune zerolatency,psnr, BD-MS-SSIM against --tune zerolatency,ssim; every chart names its zero's tune |
| validity | one external replay: 1 s bucket, initial decoder fullness 1000 ms, drain at the target, every counted bit; published per encoder as continuous compliance figures beside the BD (all points count in the BD; procedure section 8) |

Phase now: **full-start**; values per phase:

| Phase | INIT_MS | VBV_INIT |
|---|---|---|
| full-start | 1000 | 1.0 |

### nanocodec-{codec}@{preset} (nanocodec)

- Version: the run's commit (manifest), one binary for every rung
- Build: release, fat LTO, -C target-cpu=x86-64-v3 (workspace .cargo/config.toml); PGO only for a release, trained on clips disjoint from every dataset (nceval.content_id). Measured on 0.6.0: the plain build takes 0.984-1.031x the PGO build's time (response M3). The BP-3 run's PGO build: anchors/x86_64-opt-bp3/nanocodec (BUILD.json: commit, SHA-256, training on eval/clipset/pgo_training_2026-11-23.json)

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--bitrate --buffer-ms 1000 --peak-bitrate {KBPS}` | CBR target; 1 s buffer; drain at the target | the shared rate, buffer and peak | the default peak is 2x the target (rate_control.rs); BP-2 sets it to the target, as x264/x265 --vbv-maxrate | none known |
| `--strict-buffer` | never pass one buffer (re-codes a frame that would); the key cap stays at its default, one buffer (params.rs) | the shared 1 s buffer, held | owner, 2026-11-25 (final): ours holds the 1 s buffer strictly, even though x264, x265 and SVT-AV1 merely stay inside it by their own rate control and libaom, libvpx, openh264 and Kvazaar exceed it on 11-14 % of points (2026-11-24 report). It may cost ours a little; it removes the argument. Supersedes the 1.15 default bound of 2026-11-22 and option 5a | none |
| `--param rate.first_key_lookahead_off=1` | the start-up lookahead (frame 1 read before frame 0 is coded) no longer sizes the first key | off | every reference runs with zero lookahead (--rc-lookahead 0, --lag-in-frames=0) | the CLI still reads frame 1 early (I/O only); measured effect of the cap: response B2 |
| `--param hold.enable=1 --param hold.code_skip=1` | frame hold with all-skip output: a near-exact repeat of the last coded frame is coded as a frame whose every block copies it (no residual); every input frame is still coded | our default hold, with all-skip output (owner, 2026-11-25, D21) | every frame is coded, as on every main row. Repeats exist in the benchmark sets: held-out 20 of 97 clips (2.4 % of frames, mostly static CCTV and video calls), VC-1K 10 of 100 (0.2 %) (sealed log 23). Conformance: 725/725 streams bit-exact in every standard decoder and ours | none |
| `(adaptive coded resolution: default, on for AV1)` | AV1 codes at half size per dimension at very low rates (standard reference scaling, no key frame) | on (owner, 2026-11-23: an allowed coding tool in standard syntax) | each reference uses resize where its project's recommended real-time configuration does: libvpx's and libaom's real-time examples set it off (vpx_temporal_svc_encoder.c rc_resize_allowed = 0; svc_encoder_rtc.cc rc_resize_mode = 0), SVT's RTC default is off; scoring upscales any size-changed frame with one pinned filter for all (section 6) | measured: response Q3 (how often ours switches, its BD effect, and the references with dynamic resize on) |
| `--intra-period {KEYINT} --temporal-layers 1` | one key frame per clip; flat P | the shared key interval and structure | same as every reference (--keyint, --kf-max-dist, -k) | none |
| `--preset {PRESET}` | the rung | every shipped single-layer rung | the references are charted at several presets each | 35 rungs against 2-4 per reference: a front through few reference points is interpolation (review M10) |
| `(no thread flag)` | threads | our encoder is single-threaded | every reference runs one thread | none |
| `(DSP tier: runtime dispatch)` | SIMD tier | AVX-512 on the timing CPU, recorded; tiers are bit-exact (tools/ci/isa_identity.sh) | every reference uses its own best runtime dispatch on the same CPU (x265 --asm avx512, below) | none known |
| `--qp {QP}` | Fixed QP = no rate control, a fixed base QP; ours keeps its own frame-level cadence, key offset and block offsets (rate.constant_qp off) | the codec's base-QP table (procedure section 5) | owner, 2026-11-23: every encoder keeps its own frame- and block-level QP behaviour in fixed-QP mode (x264/x265 --ipratio, SVT and the others as listed in their tables) | measured key-frame offsets per encoder: response M4 |

### x264-{preset}[~{tune}] (x264)

- Version: 0.165 (anchors build_anchors.sh, X264_COMMIT in tools/anchors/versions.env)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): -O3, LTO (x264 configure --enable-lto), PGO through x264's own CFLAGSPROF/LDFLAGSPROF hooks (gcc -fprofile-generate / -fprofile-use -fprofile-partial-training), trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Measured on the earlier x86_64-opt build (its own fprofiled training): the plain build takes 1.051 / 1.058 / 1.014x the time (medium / veryfast / ultrafast; response M3)

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--tune zerolatency[,psnr|,ssim]` | low-delay settings (no B-frames, no lookahead, no mb-tree, sliced-threads off at 1 thread); the twins turn psycho-visual tuning to the metric | x264's own real-time tune (x264 --fullhelp: 'zerolatency'); twins as x264 documents for metric runs | the twin that suits each metric is charted (review M9: the rule per metric is fixed in the manifest) | --tune psnr disables AQ, which is not x264's best for VMAF/MS-SSIM: the default-tune family is charted beside it |
| `--bitrate --vbv-maxrate {KBPS} --vbv-bufsize {KBPS} --vbv-init {VBV_INIT}` | ABR inside a VBV: peak = target, 1 s buffer, decoder buffer {VBV_INIT} full at start | the shared bucket: 1 s, starting full | same bucket as everyone; x264's default --vbv-init is 0.9 | measured: 0.9 -> 0.6 costs x264 medium 1.4 % BD-PSNR-YUV (response B3); 0.9 -> 1.0 not measured |
| `--bframes 0 --rc-lookahead 0 --keyint {KEYINT}` | no B-frames, no lookahead (zerolatency's own values, restated); one key at the start, later keys x264's own choice | rule 2 (no added latency) and rule 3 (one key at the start of the test window) | procedure section 3 (BP-4.0): scene-cut keys are x264's own choice (rule 4; default scenecut threshold 40, common/base.c:388), so --no-scenecut and --min-keyint {KEYINT} are dropped from the CBR command; x264's scene-cut detection runs without lookahead (encoder/encoder.c:1618, slicetype.c:1523). Audit: docs/analysis/reference-fair-config-2026-11-25.md | the Fixed QP command keeps --no-scenecut and --min-keyint; BP-4 measures no Fixed QP |
| `--threads 1` | one thread (also no lookahead thread) | the shared thread count | same for all | x264 is designed around threads (review M10): stated |
| `(SEI, post-processing)` | x264 writes a ~0.6 kB user-data SEI with its options; x264 0.165 has no switch to omit it (x264 --fullhelp) | every SEI NAL is stripped after the timed encode (nceval.streamstrip), the decoded frames checked identical, the bytes removed recorded | the same post-processing strips SEI and metadata from every encoder's stream, ours included. Ours runs with --param stream.encoder_info=0 (its encoder-information SEI/metadata, about 135 B per key since 117de821a, is charged by our rate control), as x265 runs --no-info. Asymmetry, stated: x264 cannot omit its SEI, so its rate control may spend bits on it that are then stripped; measured (response 2) 662 B median per stream, 0.8 % of the stream median, up to 15 % on the smallest streams | none |
| `--qp {QP}` | CQP: no rate control, fixed base QP; key frame at QP - 6 log2(ipratio 1.4) = about 3 QP finer; AQ and MB-tree are switched off by x264 itself in CQP | x264's own CQP mode | Fixed QP = each encoder's no-rate-control mode with its own frame and block QP behaviour (procedure section 5) | verified in x264 source at the pinned commit (encoder/encoder.c lines 951-966: i_aq_mode = 0, b_mb_tree = 0; qp_i = qp_p - 6*log2f(f_ip_factor)) and measured (key QP 29 at --qp 32, response M4). Ours keeps block offsets where x264 has none in this mode: stated |

### x265-{preset}[~{tune}] (x265)

- Version: 4.2 (X265_COMMIT)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): LTO (CMake CMAKE_INTERPROCEDURAL_OPTIMIZATION), PGO by gcc's generic -fprofile-generate / -fprofile-use -fprofile-partial-training (x265 documents no route), trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Measured on the earlier LTO-only build: the plain build takes 1.017x the time (response M3); plain vs this PGO build not yet measured

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--tune psnr / --tune ssim at ultrafast and superfast` | x265's metric tunes | measured (30 frames, 600 kbps, superfast): the two tunes code byte-identical streams; x265 reports AQ strength 0.0 for --tune ssim at these presets. The first BP-3 run's BDs agree (ultrafast -4.77 % both, superfast -21.28 % both) | at these two presets the PSNR and SSIM twins are one encoder, shown once and labelled | none |
| `--tune zerolatency|psnr|ssim (+ --no-cutree with a metric tune)` | low delay; metric twins (x265 takes one tune, so --no-cutree restores the rest of zerolatency) | x265's own real-time tune | as x264 | as x264 |
| `--vbv-maxrate {KBPS} --vbv-bufsize {KBPS} --vbv-init {VBV_INIT}` | the shared bucket | as x264 | as x264 | measured: response B3 |
| `--frame-threads 1 --pools none --no-wpp` | one thread | the shared thread count (rule 3) | WPP and frame threads exist only for parallelism; with one thread they would cost coding efficiency for nothing, so they are off (procedure section 3 rule 3) | x265 is designed around WPP threads: stated |
| `--no-info` | drops x265's 1.5-2.4 kB user-data SEI | off | SEI is not counted for any encoder anyway; this removes it from the stream too | none |
| `(no --asm flag: x265's own runtime dispatch)` | x265 picks its SIMD tier itself; its default leaves AVX-512 off on x86_64 (source/common/cpu.cpp:158-226: cpu_detect adds AVX-512 only when asked) and uses NEON/SVE on aarch64 | x265's default | procedure section 3 rule 4 (BP-4.0): a speed flag is not a shared constraint, so the earlier --asm avx512 on x86_64 is dropped (audit, docs/analysis/reference-fair-config-2026-11-25.md). Output is unchanged by design; speed may move | the x86_64 speed effect of the change is measured by the first BP-4 timing run |
| `--qp {QP}` | CQP: fixed base QP, key about 3 QP finer (ipratio); AQ, hevc-aq, AQ-motion and CU-tree switched off by x265 itself in CQP | x265's own CQP mode | as x264 | verified in x265 source at the pinned commit (source/encoder/encoder.cpp lines 4049-4056; ratecontrol.cpp 345-350) and measured (key QP 29 at --qp 32) |

### openh264-baseline[~cabac][+drop] (openh264-harness (tools/openh264-harness, SHA-256 in the manifest; the harness source is published))

- Version: openh264 v2.7.0 (OPENH264_COMMIT 1bca6231, tools/anchors/versions.env); harness commit 745b0166 (tools/openh264-harness), SHA-256 recorded in the manifest; the T16 harness (2026-11-25) adds --coded-frames and continues past a skip on the +drop row, main-row streams byte-identical (2 pool clips checked), and is rebuilt into the anchor stores before the next run
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): libopenh264.a -O3 with gcc -flto=auto (archived with gcc-ar; the harness, -O2, links with -flto) and gcc's generic -fprofile-generate / -fprofile-use -fprofile-partial-training (its Makefile offers neither), trained through the harness, trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Plain vs PGO not yet measured Plain-vs-PGO speed not measurable: the plain harness predates --bitrate

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--usage camera-realtime --complexity medium|high --cabac 0|1` | openh264's real-time camera mode; CAVLC (baseline) or CABAC (~cabac twin) | openh264's own usage type for live camera; both entropy coders charted | the CABAC twin is about 17 % better (audit finding 3) and is the H.264-to-H.264 comparison | none |
| `--adaptive-quant 1 --scene-change 1` | openh264's adaptive quantisation and scene-change detection | the library defaults (codec/api/wels/param_svc.h:166 bEnableSceneChangeDetect = true, :168 bEnableAdaptiveQuant = true); --background-detection 1 --denoise 0 --ltr 0 --temporal-layers 1 also equal the defaults (param_svc.h:165, 167, 170) | procedure section 3 rule 4 (BP-4.0): defaults unless a shared constraint needs otherwise. AQ 1 codes byte-identical streams to AQ 0 in this mode (84/84 in the audit), so the former ~aq twin is gone; scene-change on: median BD-J unchanged, 3 of 23 streams gained keys (audit, docs/analysis/reference-fair-config-2026-11-25.md) | the Fixed QP command keeps --scene-change 0; BP-4 measures no Fixed QP |
| `--bitrate {KBPS}` | openh264's rate control | the shared rate | openh264 exposes no buffer size, initial level or peak: it is judged by the same replay | its buffer cannot be set to the shared bucket: stated; measured compliance: response B3 |
| `--frame-skip {SKIP} {SKIP_ARGS}` | openh264's frame skipping; on the '+drop' row the harness also lists the input frame of each coded picture (--coded-frames {OUTPUT}.frames) | 0 on the main rows; 1 on the labelled 'openh264-baseline+drop' row, the library default (codec/api/wels/param_svc.h:169 bEnableFrameSkip = true) | the main rows code every frame, as every encoder's (procedure section 3, measurement condition). The '+drop' row is the configuration openh264 is designed for: its rate control cannot hold the rate without skipping (its own warning: 'bitrate can't be controlled ... without enabling skip frame'). Annex B has no timestamps, so the harness's coded-frame list places each picture: a skipped frame shows the previous picture and is counted, and its slot carries 0 bytes in the rate and the buffer replay, as the libvpx and libaom '+drop' rows (tools/bench/score.py drop_slots, bytes_per_slot) | none |

### svtav1-m{PRESET} (SvtAv1EncApp)

- Version: v4.3.0 (SVTAV1_COMMIT)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): LTO (SVT_AV1_LTO, its default) + PGO by SVT's own route (SVT_AV1_PGO: PGOCompileGen / PGOCompileUse), trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Measured on the earlier x86_64-opt build: the plain build takes 1.045 / 1.047x the time (M12 / M13)

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--rtc 1 --pred-struct 1` | SVT's real-time mode, low-delay prediction, with SVT's own CBR structure | SVT-AV1's documented RTC configuration (Docs/Parameters.md --rtc); low-delay CBR defaults to 2 hierarchical levels and is capped at 2 (Source/Lib/Globals/enc_handle.c:4570-4589); all references are past pictures (verified in the streams) | procedure section 3 rule 4 (BP-4.0): the encoder's own real-time structure, temporal layers included, so the earlier --hierarchical-levels 0 is dropped from the CBR command. Measured +4.2 to +7.2 % BD-J against flat P at M8-M13 (worse; compliance unchanged): the default is what the rule requires, not what scores best (audit, docs/analysis/reference-fair-config-2026-11-25.md) | the Fixed QP command keeps --hierarchical-levels 0; BP-4 measures no Fixed QP |
| `--rc 2 --tbr --buf-sz 1000 --buf-initial-sz {INIT_MS} --buf-optimal-sz 600` | CBR with the shared bucket | the shared bucket | same as all | SVT has no peak setting separate from the target: CBR drains at the target |
| `--lp 1` | one thread | shared | same for all | none |
| `--scd 0 --keyint {KEYINT}` | no scene-change keys; one key | shared | same for all | none |
| `--rc 0 --qp {QP}` | SVT's fixed base quantizer with its own default adaptation (--aq-mode at its default 2; SVT's own key and layer qindex assignment) | coordinator decision on review 3 C5: in Fixed QP every encoder keeps its own default block adaptation. SVT's help calls --rc 0 with AQ on 'CRF'; in --rtc mode its AQ is inactive: --qp 40 with and without --aq-mode 0 gives byte-identical streams at M12 and M8 (measured, 30 frames of a 720p clip; aom-inspect shows no delta-q in either) | Fixed QP = each encoder's no-rate-control mode with its own frame and block behaviour | key offset measured in response 2 (N3): key qindex 67 against 160 |

### libvpx-vp9-cpu{PRESET}[~ssim][+drop] (vpxenc)

- Version: v1.16.0 (LIBVPX_COMMIT)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): no LTO (its configure offers none), PGO by libvpx's own route (configure --use-profile after a -fprofile-generate build), trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Measured on the earlier x86_64-opt build: the plain build takes 1.034 / 1.018x the time (rt 9 / 7)

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--rt --cpu-used --aq-mode=3 --noise-sensitivity=0 --static-thresh=1 --gf-cbr-boost=0 --frame-boost=0 --undershoot-pct=50 --overshoot-pct=50` | libvpx's real-time controls | as libvpx's own real-time example sets them (examples/vpx_temporal_svc_encoder.c, VP9 branch: AQ 3, denoiser off, static threshold 1, GF CBR boost 0, periodic boost 0, 50/50 under/overshoot) | procedure section 3 rule 1 (BP-4.0, owner 2026-11-25): libvpx runs its project's own shipped real-time configuration, the example's settings (examples/vpx_temporal_svc_encoder.c lines 812-943, quoted in docs/analysis/reference-fair-config-2026-11-25.md section 4). Without them (vpxenc --rt defaults) it measures +10-11 % BD-J, mostly from AQ 3 | every example setting, kept or dropped (examples/vpx_temporal_svc_encoder.c): KEPT with the example's value: undershoot/overshoot 50, buffer 1000 / initial and optimal levels (the shared bucket's), VP8E_SET_MAX_INTRA_BITRATE_PCT 1000 (lines 939-943), CPUUSED, AQ_MODE 3, GF_CBR_BOOST_PCT 0, FRAME_PERIODIC_BOOST 0, NOISE_SENSITIVITY off, STATIC_THRESHOLD 1, resize off (rc_resize_allowed 0); DEFAULT already: FRAME_PARALLEL_DECODING 0, TUNE_CONTENT 0, DISABLE_LOOPFILTER 0, DISABLE_OVERSHOOT_MAXQ_CBR 0, ROW_MT 0 at one thread; KEPT: rc_min_quantizer 2 (line 812); rc_max_quantizer 52 only on the +drop rows (see --max-q); DROPPED: rc_dropframe_thresh (frame dropping off for every encoder), POSTENCODE_DROP (a drop mechanism), TILE_COLUMNS (log2 of threads = 0 at one thread), SVC (one layer) |
| `--max-q={MAX_Q}` | the largest quantizer | 63 (libvpx's full range, its default) on the main rows; 52 (the example's, examples/vpx_temporal_svc_encoder.c:814) on the '+drop' rows | owner decision, 2026-11-25 (procedure section 3 rule 1): the example's 52 presupposes frame dropping, which the main rows forbid; without dropping the 52 clamp overshoots the bottom rung (1.5-1.9x the target in the audit) and drops clips from the common set (BD on 18-19 of 24 clips against 23 with 63; points over one buffer 44 -> 14 of 84). Audit: docs/analysis/reference-fair-config-2026-11-25.md | on the clips the clamp keeps, 63 measures 2-4 % worse BD-J: the clamp's extra bits, not efficiency |
| `--end-usage=cbr --target-bitrate --buf-sz=1000 --buf-initial-sz={INIT_MS} --buf-optimal-sz=600` | CBR with the shared bucket | the shared bucket: 1 s, starting full (the example uses 1000/600/600; the optimal level stays 600 ms, the rate controller's target level) | same as all | none |
| `--drop-frame={DROP} --resize-allowed=0` | frame dropping (buffer-level threshold, %), no dynamic resize | drop 0 on the main rows; 30 on the labelled '+drop' row: the threshold libvpx's own example test passes (libvpx v1.16.0 test/vpx_temporal_svc_encoder.sh:43 frame_drop_thresh=30, read as rc_dropframe_thresh by examples/vpx_temporal_svc_encoder.c:810); resize off | the main rows match every encoder; the '+drop' row is the configuration its authors ship, dropped frames shown as the previous picture and counted | ours holds the bucket by re-coding: its re-coded share and time are printed beside it |
| `--threads=1 --lag-in-frames=0 --passes=1` | one thread, no lookahead, one pass | shared | same for all | none |
| `(file reader)` | vpxenc reads the input with 4 KB stdio reads | as shipped | the CLI is the program a user runs | up to about 4 % of its time is input reading (study section 5): stated |
| `--end-usage=q --cq-level={QP}` | libvpx's own constant-quality mode: a base quantizer with its own key and golden-frame offsets, no bitrate target | libvpx's native fixed-quality mode (review 2, N3: BP-2's min = max pin removed its offsets) | Fixed QP = each encoder's no-rate-control mode with its own frame behaviour | key offset: measured in the BP-3 response (N3) |

### libaom-av1-cpu{PRESET}[~ssim] (aomenc)

- Version: v3.15.3 (LIBAOM_COMMIT)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): LTO (CMake CMAKE_INTERPROCEDURAL_OPTIMIZATION), PGO by gcc's generic -fprofile-generate / -fprofile-use -fprofile-partial-training (libaom documents no route), one profile for svc_encoder_rtc and aomenc, trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Measured on the earlier LTO-only aomenc: the plain build takes 1.031 / 1.028x the time (rt 10 / 8) Measured (BP-3.1, c8a.large, 3 clips x 5 rounds, the record's CBR command at the 800 rung): the PGO build takes 0.937x the plain build's time

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `(CBR) aomenc --rt with the example's real-time settings: --min-q=2 --undershoot-pct=50 --overshoot-pct=50 --buf-optimal-sz=600 --max-intra-rate=300 --aq-mode=3 --gf-cbr-boost=0 --enable-cdef=1 --loopfilter-control=1 --enable-warped-motion=0 --enable-obmc=0 --enable-global-motion=0 --enable-order-hint=0 --enable-tpl-model=0 --deltaq-mode=0 --*-cost-upd-freq=3 --cdf-update-mode=1 --enable-cfl-intra=0 --enable-smooth-intra=0 --enable-angle-delta=0 --enable-filter-intra=0 --use-intra-default-tx-only=1` | libaom's real-time configuration on the main rows and the ~ssim twins | the settings of libaom's own real-time example (examples/svc_encoder_rtc.cc: min-q 2 1907, under/overshoot 50 1909-1910, optimal level 600 1912, max intra 300 % 1720, AQ 3 from -aq 1 2037, GF CBR boost 0 2038, CDEF on 2039, loop-filter control 1 2040, warped/OBMC/global motion/order hint/TPL off 2041-2045, deltaq 0 2046, cost updates 3 2053-2056, CDF update mode 1 2058, CfL/smooth/angle-delta/filter intra off and default-tx-only intra 2061-2065; reference tree b2347b8), carried to the CLI | procedure section 3 rule 1 (BP-4.0, owner 2026-11-25): libaom runs its project's own shipped real-time configuration. The example program itself hard-codes a 600 ms initial buffer (svc_encoder_rtc.cc:1911) and max-q 52 (1908), which break the shared full-start buffer and, without dropping, overshoot; so the main rows run the same settings through aomenc with the shared bucket. Measured -0.5 % (cpu8) / -1.0 % (cpu10) BD-J against the example; clips with a BD 18 -> 23 of 24; points over one buffer 35-36 -> 10 of 84 (audit, docs/analysis/reference-fair-config-2026-11-25.md). With aomenc, --tune=ssim exists in CBR, so the ~ssim twin is charted in CBR too | the CLI's reference set is wider than the example's: 6-11 % slower with the same flags (BP-3 response 2, N6); the BP-4 timing run measures it on this command. The PGO profile was trained on the BP-3 commands (example for CBR, aomenc for Fixed QP) |
| `--buf-sz=1000 --buf-initial-sz={INIT_MS} --max-q=63 --drop-frame=0` | the shared bucket, starting full; the full quantizer range; no frame dropping | rules 2 and 3; max-q 63 is libaom's default | owner decision, 2026-11-25: the example's max-q 52 presupposes frame dropping, which the main rows forbid; the '+drop' rows keep the example as shipped | none |
| `(Fixed QP) aomenc --end-usage=q --cq-level={QP} with the example's real-time tool flags` | libaom's constant-quality mode with the real-time controls of the example (review 3 C4) | the example has no fixed-quality mode, so the CLI is used with the example's tool set: --enable-cdef=1 --enable-warped-motion=0 --enable-obmc=0 --enable-global-motion=0 --enable-order-hint=0 --enable-tpl-model=0 --deltaq-mode=0 --*-cost-upd-freq=3 --enable-cfl-intra=0 --enable-smooth-intra=0 --enable-angle-delta=0 --enable-filter-intra=0 --use-intra-default-tx-only=1 --cdf-update-mode=1 --loopfilter-control=1 --resize-mode=0 | the same tool set as the CBR rows | the CLI's reference set is wider than the example's (6-11 % slower in CBR): stated on the Fixed QP rows |
| `(file reader)` | aomenc reads with 4 KB stdio reads | as shipped | as libvpx | stated |
| `--end-usage=q --cq-level={QP}` | libaom's own constant-quality mode, as libvpx | libaom's native fixed-quality mode | as libvpx | key offset: measured in the BP-3 response (N3) |

### libaom-av1-cpu{PRESET}+drop (svc_encoder_rtc (libaom's real-time example program, as shipped))

- Version: v3.15.3 (LIBAOM_COMMIT)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): LTO (CMake CMAKE_INTERPROCEDURAL_OPTIMIZATION), PGO by gcc's generic -fprofile-generate / -fprofile-use -fprofile-partial-training (libaom documents no route), one profile for svc_encoder_rtc and aomenc, trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Measured on the earlier LTO-only aomenc: the plain build takes 1.031 / 1.028x the time (rt 10 / 8) Measured (BP-3.1, c8a.large, 3 clips x 5 rounds, the record's CBR command at the 800 rung): the PGO build takes 0.937x the plain build's time

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `(CBR) svc_encoder_rtc -sl 1 -tl 1 -lm 0 -sp {PRESET} -aq 1 -th 1 -k {KEYINT}` | libaom's own real-time example program, one spatial and one temporal layer (layering mode 0: LAST is the only reference) | review 3 C4 / coordinator decision 4: the libaom project's own real-time configuration is its example, used as it is except frame dropping. Its settings (examples/svc_encoder_rtc.cc lines 1907-1913, 2036-2085): min/max q 2/52, undershoot/overshoot 50, buffer 1000 ms with initial and optimal level 600 ms, CDEF on, warped/OBMC/global motion/order hint/TPL off, deltaq 0, cost updates every 3, CfL/smooth/angle-delta/filter intra off, default-tx-only intra, CDF update mode 1, loop-filter control 1, auto tiles, AQ 3, GF CBR boost 0, max intra rate 300 % | the '+drop' rows (BP-4.0): the program the libaom project ships as its real-time reference, as shipped (its 600 ms start and max-q 52 included); the main rows run its settings through aomenc. The CLI with the same flags was measured 6-11 % slower (response 2, N6; the stream inspection attributes it to the CLI's wider reference set) | the example fixes its own bucket: initial and optimal level 600 ms, so on the '+drop' rows it starts with less credit than the shared full 1000 ms: stated. It is given the exact timebase (-t {FPS_DEN}/{FPS_NUM}), but its rate control uses the integer rate den/num (examples/svc_encoder_rtc.cc:1985: 29 for 30000/1001), so on NTSC clips it budgets about 3.4 % more bits per frame than the target; the replay uses the clip's true rate for every encoder, and the excess shows in its rate over target. Its -f frame count is refused with these arguments, so the box feeds it exactly N frames (box.py refuses an input of any other size). It has no --tune, so the '+drop' rows have no SSIM twin |
| `--drop-frame={DROP}` | the example's frame dropping (buffer-level threshold, %) | 0 on the main rows; 30 on the labelled '+drop' row: the threshold libaom's own real-time SVC rate tests use (test/svc_datarate_test.cc:1376, cfg_.rc_dropframe_thresh = 30, in the reference tree at b2347b8); the example itself defaults to 0 (examples/svc_encoder_rtc.cc:407-408 sets it only from --drop-frame) | the main rows match every encoder (no drops); the '+drop' row is the configuration its authors ship, dropped frames shown as the previous picture and counted | ours holds the bucket by re-coding instead: its re-coded share and time are printed beside it |

### kvazaar-{PRESET} (kvazaar)

- Version: v2.3.2 (KVAZAAR_COMMIT 6040962b, tools/anchors/versions.env)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): LTO (CMake CMAKE_INTERPROCEDURAL_OPTIMIZATION), PGO by gcc's generic -fprofile-generate / -fprofile-use -fprofile-partial-training (Kvazaar documents no route), trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Plain vs PGO not yet measured

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--preset ultrafast|superfast|veryfast` | Kvazaar's speed presets | its real-time range (coordinator, 2026-11-23: the real-time HEVC reference) | Kvazaar is a real-time HEVC encoder; its fast presets are its live-encoding settings | not yet measured in this procedure |
| `--gop lp-g8d4t1 --period {KEYINT} --no-open-gop` | Kvazaar's low-delay P GOP: 8 pictures, 4 layers, every picture decodable (t1), references past pictures only; one key at the start | rule 2 (no added latency) needs an override: Kvazaar's default GOP is a 16-picture random-access pyramid with reordering (src/cfg.c:137-138, kvz_gop_ra16). Kvazaar has no default low-delay GOP; its low-delay family is lp-g{G}d{D}t{T} (README 'LP-GOP syntax'). d4: the deepest its CBR rate control supports (src/encoder.c 4-layer low-delay weights; 5 layers refused at encoder.c:800, all 252 audit encodes of lp-g16d5t1 failed); g8: the shortest GOP holding 4 layers (cfg.c:1445-1461); t1: no reference skipping (cfg.c:1464-1491). QP offsets are Kvazaar's own (+1 to +4 by layer, cfg.c:1466) | procedure section 3 rules 2 and 4 (BP-4.0): the encoder's own low-delay structure, no reordering. Measured -6.1 to -8.6 % BD-J against --gop 0 (audit, docs/analysis/reference-fair-config-2026-11-25.md). The Fixed QP command keeps --gop 0; BP-4 measures no Fixed QP | the choice of d and g within Kvazaar's low-delay family is resolved from its source as above, not a Kvazaar default |
| `--bitrate {KBPS}000` | Kvazaar's rate control (lambda domain); its unit is bits per second (kvazaar --help: 'N: Target N bits per second') | the shared rate. Until BP-3.1's pipe-clean run (2026-11-23) the record passed kbps, a target 1000x too low: every Kvazaar CBR point measured before then (response 2's N1 table, 75-100 % valid) undershot by construction and is void | Kvazaar has no VBV size, initial level or peak option: judged by the same replay, like openh264 | its buffer cannot be set to the shared bucket: stated |
| `--threads 0 --owf 0 --no-wpp --no-psnr --no-info` | no worker threads or frame parallelism; no wavefront substreams (as x265 --no-wpp); no PSNR computed inside the timed run (Kvazaar's --psnr is on by default); no information SEI | one thread, no in-encoder metric pass | same for all: no other encoder computes a metric while timed | none |
| `--qp {QP}` | Fixed base QP with Kvazaar's own frame/block behaviour | its defaults in CQP | owner rule: every encoder keeps its own frame/block QP behaviour | key offset not measured |

### vvenc-{PRESET} (vvencFFapp)

- Version: v1.14.0 (VVENC_COMMIT 9428ea86, tools/anchors/versions.env; the build the CTC run used)
- Build: x86_64: anchors/x86_64-opt-bp3 (built 2026-11-23 by tools/anchors/build_anchors.sh --opt on the build box, gcc 15.2; SHA-256s in its SHA256SUMS, flags in its build-flags.json): LTO (VVENC_ENABLE_LINK_TIME_OPT, its default), PGO by gcc's generic -fprofile-generate / -fprofile-use -fprofile-partial-training (VVenC documents no route), trained on the PGO training set eval/clipset/pgo_training_2026-11-23.json (NASA/NOAA, disjoint from every dataset, nceval.content_id) with this record's own commands, every family but the +drop rows, CBR 600 kbps and Fixed QP at the middle of the tool's CTC table (tools/anchors/pgo_train.py). Plain vs PGO not yet measured

| Flags | Controls | Value | Why fair | Remaining asymmetry |
|---|---|---|---|---|
| `--TicksPerSecond {VVENC_TICKS}` | VVenC's timing clock; it must be a multiple of the frame rate (vvencFFapp: 'TicksPerSecond must be a multiple of FrameRate/Framescale') | the largest multiple of the rate's numerator not above 27000000: 27000000 itself for NTSC (VVenC's own advice), and a valid clock for any rate num/den. Found by the first BP-3 Fixed QP run: VCD2 webcams at 17, 23 and 28 fps and Wikimedia uploads at 24.39, 29.089 and 29.833 fps were refused with the default | timing metadata only: the coded pictures are unchanged | none |
| `-c experimental/lowdelay_faster.cfg | lowdelay_fast.cfg` | VVenC's low-delay configuration at its fastest presets | VVenC's own low-delay configs (cfg/experimental/ in the v1.14.0 tree, as the CTC lens used: nceval.ctc.reference_encoders VVENC_CONFIG); the fastest two | the coordinator's VVC reference (2026-11-23); the same build as the AOM-CTC run | the low-delay configs are marked experimental in v1.14.0; low delay codes B slices (no P-only mode); VVenC has no real-time speed class: stated |
| `(coding order and references; evidence)` | whether the low-delay configs reorder pictures or reference future ones | verified on v1.14.0 streams, 2026-11-23: with lowdelay_faster and lowdelay_fast (cfg/experimental, PicReordering 0, GOPSize 8, every RPL entry a positive POC delta), at fixed QP and with single-pass rate control. The encoder's per-picture log (--Verbosity=6) shows coding order = display order (POC 0, 1, 2, ... for 24 of 24 pictures) and every L0/L1 entry a past picture (GPB: L1 = L0). Its temporal filter (MCTF, BIM) is set to past pictures only (MCTFFutureReference 0). Logs and configs: bench-method-records/vvenc-lowdelay/ | no picture reordering and no future reference, like every other encoder | none in the stream structure |
| `-c rc1p.cfg --LookAhead=1 --TargetBitrate {KBPS}000 --MaxBitrate {KBPS_X1500}` | VVenC's single-pass rate control: a VBR controller with a pre-analysis look-ahead (its log: 'VBR single-pass', 'LookAhead:1') | the shared rate (bits/s). VVenC refuses one-pass rate control without the look-ahead ('Look-ahead encoding must be used with one-pass rate control') | the only single-pass rate control VVenC offers. --MaxBitrate ('approximate maximum instantaneous bitrate', vvencFFapp --fullhelp) is set to 1.5x the target, the lowest VVenC accepts ('MaxBitrate must be at least 1.5*TargetBitrate', tested 2026-11-23); every other peak is the target. VVenC has no buffer size or initial level, so it is judged by the same replay | VVenC's CBR points have a rate-control look-ahead and no buffer model, an advantage to VVenC that no other encoder has (depth not measured): stated beside every VVC comparison. VVenC's fixed-QP view has none |
| `--Threads=1 --MaxParallelFrames=0` | one thread, no frame parallelism | one thread | same for all | VVenC is designed around threads: stated |
| `--IntraPeriod {KEYINT}` | one intra picture per clip | the shared structure; tested with GOPSize 8: v1.14.0 accepts 300 (its log: 'Intra period (keyframe): 300'), as it does 296 and 304, and codes one intra picture (bench-method-records/vvenc-lowdelay) | same as all | none |

## 18. Open tasks

| # | Task | Done when |
|---|---|---|
| T1 | Ours with hold on and all-skip output in the main rows. The engine mode exists since 96d1626d6 (`hold.code_skip`, default off; conformance 655/655 with it on, 2026-11-23); the owner decides after its 2026-11-25 screen | the owner's decision on the screen |
| T2 | Initial-fullness setting for our rate control (engine); then the final phase (F = 600 ms) | the replay at 600 ms shows no start-caused overflow |
| T7 | Source-pipeline bias: pristine CTC sequences coded directly and through `build_pool.py` | the number is published |
| T9 | Kvazaar and VVenC record commands pass a full pipe-clean | done in this round's runs |
| T11 | Upscaler sensitivity (bilinear, bicubic, Lanczos) for size-changing streams | the table is published |
| T12 | Screens' rate scaling and ladder moved to section 5's | screens and plan give equal targets |
| T13 | VVenC look-ahead latency, to decide whether it may run CBR | measured |
| T17 | ISA tier per encoder on c8i (x86) and c8g (NEON / SVE / SVE2), from the runs' banners, into the record | the record lists them |
| T18 | `fleet.py preflight`: disk, quota and the benchmark cap, the PC's address, spot prices and reclaim rates by zone, the cost envelope to approve once | done (landed); the next run starts with it |
| T22 | A live `fleet.py rehearse` on the tools version of the next run (stage, encode and time with a forced reclaim, validity, Arm scoring, report) | it passes before any full run |
| T19 | Superseded by BP-4: scoring is local (T25) | — |
| T20 | Superseded by BP-4: no canaries or references; the check set (section 6) replaces them | — |
| T21 | `fleet.py recover`: book and clean up orphaned boxes after a PC crash (today a scratch tool) | landed and tested |
| T23 | BP-4 scheduler: `fleet.py plan-timing` and `run --layout half` (section 6: slots, family-at-a-time with filler, longest first, check sets, family settling, reclaim handling); the scratch runner `bench-method-records/whole-box-2026-11-25/tools/wb_remote.py` is the starting point | landed with tests and a rehearsal (T22) |
| T24 | Trial: "ours" CBR on two c8g.metal-24xl (a shared subset on both) and one c8i.metal-48xl when capacity allows | agreement between machines, wall time and cost measured against the plan's estimate; the owner decides on the full runs |
| T25 | Bench scoring on the existing local GPU scorer: `fleet.py score --local` (`localscore.py`, `score.py --metrics-backend vmafx`), V27, V29-V31 (`streamcheck.py`) | done 2026-11-25: 210 AOM-CTC streams identical to their cloud scores (section 7) |
| T26 | Dev CPU A/Bs (`aws_cpu_ab.py`, screens' timing) moved to section 6's loaded condition on the same types (dev matches benchmark; `eval/tests/test_dev_matches_benchmark.py`) | the test checks it |
| T30 | Anchor 4× top point in `plan.anchor_points` and `report.curves` | landed with tests; the common-set loss per family reported |
| T31 | History: re-time every history build under BP-4; add the 600-frame windows (one key; a key every 150) | the history page shows BP-4 speeds and the long-window view |
| T32 | Strata: low-fps camera and 50/60 fps clips in the tuning pool and the redraw | in the pool and the held-out manifest |
| T33 | Windows: plan 150 + 600 (each clip once, 150 read from the 600 encode's start); report each clip once in the headline and each window as its own view; key every 150 frames | landed in plan.py and report.py with tests |
| T28 | Our encoders' cache sensitivity under load (brief `work/briefs/cache-sensitivity-2026-11-25.md`) | our fast presets' half-load slowdown no worse than x264 ultrafast's on Graviton |

## 19. Change log

| Version | Date | Change | Reason |
|---|---|---|---|
| BP-1 | 2026-11-23 | First normative procedure: single run per encode, c8a.large timing, cloud libvmaf scoring | The benchmark-method study |
| BP-2, BP-2.1 | 2026-11-23 | Answers to the first hostile review; fairness record; initial-fullness phases; Kvazaar and VVenC added | Review 1 |
| BP-3 | 2026-11-23 | Ladder 300/500/800/1200 with sub-linear scaling; all points count, compliance as figures; per-encoder Fixed QP modes; sealed-set exclusions; AV2 stripping | Review 2; owner ladder decision |
| BP-3.1 | 2026-11-23 | `heldout_2026-11-23b` relabelled "audited once"; 90 % marker dropped; libaom CBR via its example; standard decoders for all; c8a headline | Review 3 |
| BP-3.2 | 2026-11-23 | A malformed curve costs only its own family the clip | The pipe-clean: the strict rule kept 8 of 15 design clips |
| BP-3.3 | 2026-11-23 | The headline averages every encoder over the common clip set | Owner: one baseline, the same clips for all (before scoring) |
| BP-3.4 | 2026-11-23 | The anchor codes one point past each end of the ladder | Owner: BD overlap without extrapolation (before scoring) |
| BP-3.5 | 2026-11-23 | Fixed QP streams pruned once scored; staged clips linked | Disk |
| BP-3.6 | 2026-11-23 | Two interleaved timing references per box; normalised speed | 24 of 49 c8a.large boxes drifted -8 % to +14 % |
| BP-3.7 | 2026-11-24 | Timing box c8a.2xlarge; encode-only re-timing (`--timing`) | c8a.large shares L3: 12.6 % spread |
| BP-3.8 | 2026-11-24 | The budgeted plan: VVenC timed on a subset, Zen 3 and Graviton on the history subset only, VC-1K CBR only | Budget |
| T8 | 2026-11-24 | Reference divergence on a 3-point moving median (2 % kept) | Single-point noise; set after seeing results |
| BP-3.9 | 2026-11-24 | The monotone envelope | References' non-monotone curves; decided after seeing results (it changed nothing on "ours" CBR) |
| BP-3.10 | 2026-11-24 | Timing on c8i.2xlarge spot (eu-north-1), c8g.2xlarge the second family; c6a and Jetson dropped from this round | Stability within 1.5 % at about 1/5.8 the cost |
| BP-3.11 | 2026-11-24 | Fleet designed for failure: one job queue, small units, 30 s checkpoints, spot notices, automatic replacement, zones in turn, settled jobs only (V24); one local driver process. V6 judged per reference window, not per box; V7's clock check refuses single encodes. References and canaries record the timing CPU's hardware counters (cycles, instructions, cache and LLC misses, backend stalls, as available; `perf stat -C` beside the encode) and the effective clock. A packing test (`--background-share`): a share of a fleet's boxes loop benchmark encodes on their idle physical cores, to measure whether packed boxes keep timing accuracy; their jobs pass every check as usual. V7's clock check skips encodes under 0.25 s. Scoring slots replace reclaimed boxes; a cap of 1000 vCPU on all benchmark boxes; Every stop of a box closes it with references (driver STOP); references also every 5 minutes of encode time. eu-north-1a avoided (lessons: analysis/bp3-run-lessons-2026-11-24.md) Scoring decodes in memory with a memory-derived worker cap, on c7i/c6i | Owner: a reclaimed box lost all its work (records came back only at its end), and 12 of 119 c8i boxes were reclaimed in the first hour. x264 ultrafast encodes of 0.02-0.04 s failed V7's clock check on every re-run (one sample, the clock still rising from idle). Owner: tell clock drops from neighbour cache or memory contention in the slow episodes of box11 and box17. One window's 2.4 % divergence refused all 1,302 jobs of a box (owner: references exist to judge each window). Scoring boxes stalled: 90 workers' 1080p decodes filled the 40 GB root disk (measured: 38 of 38 GB used, I/O pressure) |
| BP-3.12 | 2026-11-24 | Ours runs CBR at its default buffer bound (1.15 buffers) and default key cap (1.0 buffer): `--strict-buffer` and the key-cap override removed from the record. Our CBR points re-coded on c8i (same build), competitors reused; the strict points kept, labelled "strict buffer (superseded)" Compliance's primary measure is the bucket at each stream's own rate (the buffer page's), the target-rate bucket secondary | Owner decision, restored (docs/analysis/cbr-buffer-references-2026-11-22.md, "Decision taken before this table"; the review response's change e7611a666, 2026-11-23, reversed it without his approval): the bound was settled before the review and is stated on the public site (web/docs/data/cli.json); the review's change (e7611a666, BP-3.1) reversed it without his approval. Measured in the BP-3.10 "ours" CBR run: x264 and x265 peak at 0.74 buffers, SVT-AV1 at 1.04-1.08, libaom, libvpx and openh264 at 14-57; strict cost us +0.20 to +0.79 % BD. Compliance, published beside every result, judges every encoder's bound alike. Set after seeing results |
| BP-3.13 | 2026-11-24 | BP-3.12's record change reverted: ours runs CBR at one buffer (`--strict-buffer`) with the key cap as run on 2026-11-24; the 1.15-buffer re-run cancelled before launch (its plans kept, unlaunched, in bench/bp312/). Compliance's measures stay: the bucket at the stream's own rate, at the target, and the worse of the two, the first key's size, per class and frame-rate band | Owner decision 2026-11-24 18:10: the headline conditions stay as run, pending his decision on the bound and the key-frame strategy; inputs: docs/analysis/cbr-buffer-references-2026-11-22.md and the fairness review (scratch/fair-cbr/fair-cbr-procedure-2026-11-25.md) |
| BP-3.13, mixed reports | 2026-11-24 | `report.py --parts`: a report takes each family from the part that holds it (the re-measure: ours from the new build's runs, the references reused from the 74aaa1fcd runs); V25 per family and V18 on the reused commands; V26 also counts an encode that fails alike on every box (the 11-06 and 11-07 builds refuse 960x540 in HEVC and AV1). `fleet.py score` prunes the scored Fixed QP streams as each pass ends (BP-3.5); `preflight` checks the disk floor against the runs' expected stream bytes and the account's stop line ($790 of the $800 cap) against the envelope, and no box launches whose worst case would pass that line; the benchmark plans share one vCPU cap (Purpose `bench` and its plans) | Coordinator for the owner, 2026-11-24 evening: the re-measure on the new head reuses today's reference points; the disk fell under the floor with 90 GB of scored Fixed QP streams; the owner raised the cap to $800 |
| BP-3.13, V26 | 2026-11-24 | V26: a deterministic standard-decoder failure is reported as that encoder's failure. First case: ours, nanocodec-av1@slow, ControlledBurn_1280x720p30_420, qp160 (build 74aaa1fcd): stream sha256 e0a02a081836fa5b2a69aa52759c90fc65892e6e35a2680e348bf55435e36a6c, 796,998 bytes, coded twice identically; dav1d, aomdec and our decoder all fail at frame 81 of 130 (kept with their outputs in bench/encoder-failures/e0a02a081836/) | Coordinator for the owner: a refused point is re-run, never waived, but a deterministic stream fails every re-run; the failure is a result, counted against the encoder |
| BP-3.14 | 2026-11-25 | Curve rules: a curve needs 2 points after the envelope (4 before); no minimum overlap, the BD is integrated over whatever quality range the curve and the zero line share, and only ranges that do not touch give no BD; each family's median overlap and its clips under the old 0.5 line are reported beside its figure; a point without a metric is out of that metric's curve only, counted; every clip the common set lacks is listed with its families and reasons. Kept: the envelope, the one baseline, the common-set headline with each family's own-set figure, the fall refusal (which the envelope leaves nothing to act on). Reports regenerated from the stored points, no new encode; the BP-3.13 reports kept beside them as `report-bp313.*`. Common sets: "ours" CBR 41 of 94 to 84 of 97 (83 of 97 on the 74aaa1fcd build), VC-1K CBR 77 of 99 to 99 of 100, AOM-CTC CBR 9 of 34 to 24 of 34; both Fixed QP views were already whole (97 of 97, 34 of 34). | Owner decision 2026-11-25, **after seeing results**: everything is BD against one baseline and every encoder should have a BD on every clip; the 4-point and half-overlap rules had cut the "ours" CBR headline to 41 of 94 clips (sealed log 17) |
| BP-4.0 | 2026-11-25 | **Timing:** loaded speed on whole machines replaces one encode per small box. One bare-metal instance type per chip (c8i.metal-48xl, c8g.metal-24xl; owner, the same day: bare metal only, 2.3-3.1 times cheaper than the non-metal sizes first written here), never mixed, with capacity sought for up to 10 minutes across all zones and cheap regions. One encoder family at a time on every second physical core, longest first, with same-family filler so every timed encode runs under the same load; whole families per machine. A check set at each machine's start and end (V6) replaces the timing references, normalisation and canaries; times are never rescaled. Family-level settling (V24), slot confinement (V8), one type per series (V28). **Scoring:** on the PC (decoders on CPU, VMAF v1 and MS-SSIM on the GPU's VMAFx server), with a 1 % libvmaf CPU parity sample (V27); cloud CPU scoring is the fallback. Scheduler to be built and local scoring to be wired in (T23, T25) and trialled (T24) before any run | Owner decision 2026-11-25, on the whole-machine timing test (analysis/whole-box-timing-2026-11-25.md; 100,368 encodes, $3.27): BP-3 timing was slow and dear and refused 8 % (Intel) to 35 % (Graviton) of its times; whole machines repeat to 0.06-0.28 % at half load. Owner: each encoder competing with its own work is a fair test, arguably fairer, and nobody runs an encoder on an unloaded server; half load is the compromise, and our cache sensitivity is to be optimised. Owner: score locally where possible (95 %+ of earlier dev spend was metric computation the local GPU does free). Decided after seeing timing results on 8 sealed clips (sealed log 18); no quality was read |
| BP-4.0, configuration rule | 2026-11-25 | Section 3's five-rule configuration rule: real-time defaults; overrides only for shared constraints; Fixed QP: no temporal layers, and each encoder's own QP cadence (no cadence imposed or removed; amended the same day from "match the standards' low-delay conditions"); CBR keeps each encoder's own real-time structure and cadence; everything else default, including the quantizer range (max 63). Applied to the record after the mapping audit | Owner, 2026-11-25: one definitive answer to what a fair configuration is, after repeated rounds; max-q 52 overshot the low rungs and flattered libvpx/libaom through the common set |
| BP-4.0, no Fixed QP | 2026-11-25 | The Fixed QP comparison is removed: BP-4 measures CBR only. BP-3's Fixed QP reports are withdrawn. VVenC (charted in Fixed QP only) leaves the charts. Configuration rule 3 now says there is no Fixed QP mode; rules 1, 2, 4 and 5 apply to CBR | Owner, 2026-11-25: Fixed QP is not well enough defined to be fair or useful. Read from source the same day: at Fixed QP ours varies its quantizer over time (`qp_cadence.rs`, `rate.cadence_fixed_qp`), SVT-AV1 only with temporal layers (`rc_crf_cqp.c: cqp_qindex_calc`), x264 and x265 not at all (`--qp` forces AQ and mb-tree off), and x264's and x265's `--qp` mode forces AQ off. (An earlier text of this row said the record pinned libaom and libvpx to one quantizer; that was wrong: their Fixed QP rows used their own `--end-usage=q` mode, checked in the record on 2026-11-25.) |
| BP-4.0, CBR definition | 2026-11-25 | Configuration rule rewritten as the final CBR definition. The encoder's choices are limited only by no added latency (no lookahead, no reordering; temporal layers allowed) and the 1 s buffer at the target rate. Four measurement conditions (same input with one key at the start, same target, one thread, every frame coded on main rows). Everything else default, including structure, cadence, scene-cut keys, quantizer range and AQ. Record diffs follow the CBR mapping audit | Owner, 2026-11-25: "the only constraints should be the 1 second buffer, no lookahead"; reordering, rate, threads, input and dropping kept as latency or measurement conditions with his agreement |
| BP-4.0, buffer | 2026-11-25 | Every encoder runs its own rate control against the configured 1 s buffer and none is forced to hard-enforce it: ours drops `--strict-buffer` and the key-cap override, back to its default 1.15-buffer bound. One phase: the buffer starts full (1.0 s) for everyone; the 600 ms phase and its key cap are dropped. Record, render and tests updated | Owner, 2026-11-25 (options 5a and 6): one rule for every encoder; compliance stays published beside every BD at the stream's own rate and at the target |
| BP-4.0, owner decisions | 2026-11-25 | The 8 clips of sealed-log 18 stay; the held-out set is redrawn back to 97 with low-fps camera and 50/60 fps strata; references are reused across our builds and re-timed when a run's check set moves more than 1 %; the x264 anchor also codes 4× the top rung; history builds are re-timed under BP-4, with 600-frame windows on the history clips; publication only with both platforms, the stated speed measure, compliance and limitations | Owner, 2026-11-25: agreed the eight open decisions as recommended |
| BP-4.0, reference commands | 2026-11-25 | The audit's CBR record changes applied (docs/analysis/reference-fair-config-2026-11-25.md section 6): x264 drops --no-scenecut/--min-keyint; x265 drops --asm avx512 (and the per-architecture argument mechanism, now unused); openh264 AQ and scene change on, ~aq twin removed; SVT-AV1 drops --hierarchical-levels 0; libvpx max-q 63 on main rows, 52 on +drop rows; libaom main rows and ~ssim twins through aomenc --rt with the example's settings, +drop rows the example as shipped; Kvazaar --gop lp-g8d4t1. Rule 1 names libvpx's and libaom's shipped real-time example configurations. Fixed QP commands untouched (not measured). Coded again once per platform: libaom cpu8/cpu10 main rows and their new ~ssim twins (aomenc), libvpx cpu7/cpu9 main rows (max-q 63), SVT-AV1 M8-M13 (its own 2-level structure), Kvazaar ultrafast/superfast/veryfast (lp-g8d4t1), openh264 baseline and ~cabac (AQ and scene change on), x264 every preset and tune (scene-cut keys allowed; the zero line, so every BD moves slightly), and x265 every preset and tune on x86_64 (default dispatch: speed only, streams identical by design) | Owner, 2026-11-25: the configuration rule applied flag by flag, each default cited from source and measured on 24 tuning-pool clips |
| BP-4.0, strict buffer | 2026-11-25 | Ours runs `--strict-buffer` again (key cap at its default, one buffer), superseding the same day's option 5a and the 2026-11-22 1.15 bound | Owner, 2026-11-25: "we can go with strict, even if it penalises us, I am just sick of the argument"; final |
| BP-4.0, held-out set | 2026-11-25 | The held-out set stays at its 97 clips; the 12 excluded by sealed-log 20 are reinstated, and no clip is excluded for use from now on (uses are logged and disclosed). No redraw | Owner, 2026-11-25: exclusion creates selection bias |
| BP-4.0, windows | 2026-11-25 | Two window lengths: 150 frames on every clip, and 600 frames on every clip whose source allows it (40 held-out, 4 VC-1K at the decision). Each clip is coded once, at its longest window; its 150-frame window is the first 150 frames of that encode (no clip coded twice). The 600-frame window shows mid-sequence key frames. Each clip counts once in the headline. Key every 150 frames in every window, as dev's L3 and final check (`nceval.key_shares`): the 600-frame window has keys at 0, 150, 300 and 450 (T33) | Owner, 2026-11-25: "we need 600 to show mid-sequence I frames properly"; cost about 1.6x today's frames |
| BP-4.0, zero line | 2026-11-25 | BD-VMAF and BD-J are measured against x264 medium's PSNR twin, x264's best tune for both; BD-PSNR and BD-MS-SSIM unchanged | Owner, 2026-11-25: the intent was always the best twin per metric; the default tune is x264's worst for J (review finding 1: our BD-J against x264 moves 8-14 points) |
| BP-4.0, settled decisions | 2026-11-25 | A "Settled decisions" section (D1-D17) records each argued question, its decision and the rejected alternatives with reasons; none is reopened without new evidence and the owner | Owner, 2026-11-25: stop the repeated loops |
| BP-4.0, openh264 +drop | 2026-11-25 | T16 done: the labelled `openh264-baseline+drop` row, openh264's own frame skipping on (its library default). The harness continues past a skipped frame and lists the input frame of each coded picture (`--coded-frames`); the box keeps the list in the encode record; scoring places the pictures from it (a skipped frame shows the previous picture, counted). Every '+drop' row's rate and buffer replay now count a dropped frame's slot as 0 bytes over the whole duration (they had counted coded pictures only, which overstated a '+drop' row's rate). Main rows are byte-identical (checked on 2 pool clips). The harness in the anchors stores is rebuilt before the next run | Owner, 2026-11-25: give openh264 the same labelled '+drop' row as libvpx and libaom |
| BP-4.0, local scoring and stream checks | 2026-11-25 | `fleet.py score --local` scores on the PC's existing GPU service through `score.py`'s `vmafx` backend, with V27's cloud libvmaf sample before any merge. V29-V31: reordering and hidden frames read from every stream, the first frame a key and keys per stream reported, one core per encode | D16; owner: verify behaviour from the streams, not the commands. A check on packet timestamps (the 2026-11-25 audit's) passes every raw Annex-B stream, reordered or not: V29 reads output frames' packet positions instead |
| BP-4.0, no cross-decoding | 2026-11-25 | V10 removed from benchmark runs; each stream decoded once by its standard decoder; D18 records why | Owner, 2026-11-25: benchmarking is optimised for cost and time; correctness is the landing check's job |
| BP-4.0, per-run checks | 2026-11-25 | D19: V27 once per scorer build; V12 removed; no hardware counters; every remaining per-run check serves the fair result | Owner, 2026-11-25: fair results quickly and cheaply, nothing else |
| BP-4.0, twins | 2026-11-25 | D20: metric twins only for x264 medium for now; the others are backfilled later | Owner, 2026-11-25: "make the twins a later addition" |
| BP-4.0, hold | 2026-11-25 | D21: ours runs frame hold with all-skip output in the main rows; conditions lock updated | Owner, 2026-11-25 |
| BP-4.0, check sets | 2026-11-25 | D13, V6, V24: one check set when a machine starts (judged against the series' other machines; a failing machine does no work and is replaced) and one at its end (drift reported); none between families; a family settles the moment it finishes | Owner, 2026-11-25: minimal overhead; whole machines do not drift (< 0.35 % over 40 min; half-load repeats <= 0.28 %). The set after every family cost +49 % machine time on the trial plan (docs/analysis/time-audit-2026-11-25.md) |
| BP-4.0, stream structure | 2026-11-25 | D19, V29, V30: no extra decode for stream structure; H.264/HEVC/VVC reordering is no longer read (their first-picture key is, from its NAL unit type); AV1/VP9 checks unchanged | Owner, 2026-11-25: the ffprobe pass doubled the decode CPU of every H.26x stream (docs/analysis/time-audit-2026-11-25.md) |
| BP-4.0, one type | 2026-11-26 | D13, V6, V24: a point's time is valid on its own (its machine's start check set passed, V7/V8 per encode); the rule that a family's times count only from one machine is removed; chained machines (one machine serving several runs of its chip) carry their start check set by reference | Owner, 2026-11-26 21:28: machines of one type are equivalent; the one-machine rule left 3,759 points without a valid time after machines were lost mid-family |
| BP-4.0, untimed points | 2026-11-26 | V13 refuses only a missing stream or score. A point with a valid stream and score but no valid time (a refused box, a timing refusal) counts for quality and BD, is left out of its family's speed figure, and is listed with its reason at the top of the report | Owner, 2026-11-26 (no stupid rules): two held-out c8i points of one box without a start check set would otherwise have held the whole report |
| BP-4.0, audit fixes | 2026-11-27 | Section 4, V3: the in-sample record `eval/clipset/tuned_on.json` (VC-1K 9, AOM-CTC 12) and `report.py --exclude-clips`; VC-1K's headline leaves its 9 out, the full set beside; AOM-CTC labelled, A2 alone beside. Section 7 corrected to the decoder BP-4 used (ours for every family, the standard decoder on error) with the identity evidence. Section 14: the site withdraws the history graph on a V17 failure. Section 16: in-sample clips, start-up, timing machines, slow end, rates (reversal measured, not "narrows"), prediction structures and scene cuts. Section 17: keyint 150, scene cuts as run. Section 9: BP-4's lost clips lower our headline | Benchmark integrity audit 2026-11-27, findings 1-7 and 10; owner, 2026-11-27: fix, not only document |
