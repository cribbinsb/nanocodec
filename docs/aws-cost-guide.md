# AWS cost guide (2026-11-24)

This guide says which machine each workload rents. Every tool default and every agent
follows it. The measurements were made once, on 2026-11-24, for $0.24 (ledger rows with
`Purpose=aws-cost`). Raw records and the scratch tools that made them are in
`~/nanocodec-data/aws-cost-records` (kept). The 20-minute loops are under
`runs/<type>/drift.jsonl`.

Before you launch, check:

```sh
python3 tools/dev/aws_choose.py WORKLOAD --hours H --boxes N
```

WORKLOAD is one of: `timing`, `ab`, `packed`, `packed-arm`, `build`, `build-arm`, `all`.
The tool is read-only. It prints today's spot price for each candidate in eu-north-1 and
in the spill regions, the free spot quota, and the estimate for N boxes for H hours. It
then names the pick.

## Rules (every agent, every tool)

1. **The cheapest machine that works (owner, 2026-11-24).** Never x86 where AWS Arm is
   cheaper and will do. Graviton4 (c8g) costs less per unit of work than the x86 pair:
   2.3x the scored points and 2.6x the encodes per dollar (below). Every tool defaults to
   Arm where the work allows it. Each x86-only path states why it is **required**, not
   merely current, in the table "Arm first, per launch path".
2. **Check offerings and quotas before choosing.** Run `aws_choose.py`. Families differ by
   region: eu-north-1 has no c8a. Spot prices differ by up to 2x between zones of one
   region.
3. **Spot first, in eu-north-1.**
   - Spill regions cost 2.8-3.6x more. On 2026-11-24, c7i.24xlarge spot was $1.58/h in
     eu-central-1 and $2.00/h in eu-west-1, against $0.56/h in eu-north-1.
   - Waiting for home quota is usually cheaper than spilling. Spill only when a result is
     needed now, and say so.
4. **Use the smallest box that keeps the measurement valid** (table below). One encode at
   a time needs one quiet physical core, not a big box.
5. **Estimate the cost before launching, and report spend against the estimate** when the
   job ends. The ledger's `session cost` line is the spend.
6. **No on-demand without a stated reason** in the command's output or the commit. The
   only standing reasons are:
   - nc-remote's reclaim rule;
   - a type with no spot capacity in any region, when the job cannot wait.
7. **No idle boxes.**
   - Every box self-terminates: idle watchdog plus hard lifetime.
   - `aws_warm.py stop`, `nc-remote stop` and `aws_warm.py status` exist; use them when
     you finish.
   - Confirm `describe-instances` shows none of yours before you report done.
8. **Never a 48xlarge, metal or m-family box for packed work.** On 2026-11-24 they cost
   1.6-2.3x more per vCPU than c7i.24xlarge in eu-north-1:

   | Instance | $/h |
   |---|---|
   | c7i.48xlarge | 2.60 |
   | c7a.48xlarge | 2.25 |
   | m7i.24xlarge | 1.02 |
   | c7i.24xlarge | 0.56 |

9. **c7a (Zen 4) is not used (owner).** Its AVX-512 is double-pumped.

## Arm first, per launch path (2026-11-24)

| Path | Default | x86 required? Why |
|---|---|---|
| aws_measure screen / quick / levels (final-check), aws_warm | **Arm** (`--arch auto`, 7c2cf813a); x86 when a build is dirty or no c8g box launches | - |
| aws_measure fixedqp gap | x86 | **Required:** reference encoders' quality builds are x86 (`nceval.config.QUALITY_MACHINE`, BP) |
| aws_measure tune | x86 (per study) | **Required per study:** its callgrind CPU objective counts the x86 kernels and a study is pinned to its family. A new study may start on Arm |
| aws_measure rc_oracle, arm_oracle | x86 | Not required: they code our encoder only. Move them after an Arm parity run |
| tools/bench/fleet.py score | **Arm** (c8g.8xlarge; x86 fallback after 10 min without capacity: 10e0c027b) | - |
| tools/bench/fleet.py packed encodes, aws_heldout_chart, heldout_fleet | x86 | **Required:** the benchmark's and charts' reference encoders run their x86 quality builds (BP) |
| aws_cpu_ab, benchmark timing, aws_timing, fixedqp_timing | both families | **By definition:** the CPU is what is measured |
| nc_remote build box | x86 (c6i / c7i .24xlarge) | **Partly.** Proposal under "Build box" |
| aws_ctc | x86 | Not required: the HM / VVenC encoders have no aarch64 build yet (rare use) |
| eval/distill/box_run | x86 (r7i / m7i / c6a; c7a dropped) | Not required: torch-CPU, lightgbm and sklearn run on arm64 (r8g / m7g, cheaper), not yet tried |

**Build box split (proposed; coordinator decision).**
- **x86 required:**
  - builds whose binaries come back to the PC (x86);
  - checkasm on the AVX2 / AVX-512 tiers;
  - callgrind counts used in decisions;
  - x86 PGO release builds;
  - the verifier's x86 runs.
- **Arm will do:**
  - tests, clippy, fmt;
  - preset_identity (streams are ISA-independent);
  - conformance (the aarch64 decoders are built: "Graviton4 packed path");
  - aarch64 builds for aws_cpu_ab: native c8g.8xlarge, cold release 243 s for $0.011, against cross-compiling on the x86 box.
- **Occupancy, the first 4 h logged (136 jobs):**
  - no job waited;
  - at most 7 of 10 agent slots were busy, and 3 or fewer 78 % of the time;
  - a 64-vCPU box (6 agent slots) would have been full 0.8 % of the time, a 48-vCPU box 11.7 %.
- **Proposal:** the x86 box at 64 vCPU, plus a c8g.8xlarge pool for the Arm-able jobs (`nc-remote run --arch`), decided on the full 24 h log.

## Spot reclaims: every tool is designed for them (owner, 2026-11-24)

Spot reclaimed about 10 % of boxes in one night. A tool that rents spot meets four rules,
and a new tool is not landed without them:

1. **Copy off as work finishes.** Each finished work unit leaves the box within a few
   minutes (`aws_session.Session.pulling`, or `download_new` checkpoints), never only at
   the end.
2. **Act on the notice.** The 2-minute spot interruption notice (IMDSv2
   `spot/instance-action`) triggers a final pull and stops new work. `Session.pulling`
   checks it every 15 s and pulls at once (493dd7eff).
3. **Retry only what is missing.** A retry or replacement reruns only the missing units,
   automatically, within the job's budget.
4. **Test it.** A test simulates a reclaim mid-run and asserts that at most the in-flight
   unit is lost.

**Finished outputs leave the box before any planned stop (2026-11-24).** A tool never ends
a box (idle stop, replacement, drain, scale-down) while finished work on it has not been
copied off. It copies the work off first, and refuses to stop until it has. Only a forced
stop, the budget or the hard lifetime may end such a box, and each says what is lost. A
box's home does not outlive the box: outputs go where the tool fetches them. (19 finished
PGO builds, ~$2.7, were lost to an idle replacement of the build box when they were
written outside the job's tree.)

What a reclaim loses, per tool (audit of 2026-11-24, from the code):

| Tool | Unit | Before | Now | Fix, test |
|---|---|---|---|---|
| aws_measure, one-off box | point | the run's points since the last pull; the retry recomputed all | points in flight | 493dd7eff (reseed; notice pull) |
| aws_measure, warm box (aws_warm) | point | the same; the retry on another warm box recomputed all (no seed) | points in flight | a4a832359 (warm reseed, `cp -n` seed) |
| aws_cpu_ab | job (a rung x clip, all rounds) | the whole shard: results copied back only at the end; the retry reran every job | the job in flight | ef812fa30 (pulling, `done.json` skip, runs kept across attempts) |
| nc_remote (build box) | one command | the command in flight; retried once, whole | unchanged unit. On a notice, running jobs are told and new jobs are refused (254) and run on the replacement | dc404a7ef |
| nc_remote (build box), finished outputs | a job's `--fetch` paths | lost if the caller was gone at the job's end, and with an idle stop or replacement; undeclared home writes always lost | fetched by `reap` when the caller has gone; idle watchdog, reap, drain, replace and `stop` keep a box holding them (fetch first); `stop --force` names them; undeclared home writes warned at job end | this change; eval/tests/test_nc_remote_outputs.py |
| aws_heldout_chart / heldout_fleet | stream, scored point | minutes since the last pull; a lane is relaunched and skips what it has | as before, plus the notice pull | existing lane tests; 493dd7eff |
| tools/bench/fleet.py (timing, score) | timed unit, score | the unit in flight (`box.py` reads the notice, checkpoints, replacements rerun what is missing) | unchanged | reference design |
| aws_ctc, distill box_run | checkpoint | minutes since the last checkpoint; a rerun is whole | not changed (rare use) | open |
| aws_timing, fixedqp_timing | - | on demand by design (timing) | - | - |

## Allowed instance types and regions (tools/ci/check_aws_launchers.py reads this table)

A tool may name only these types and regions; any other needs a row here first, with its use.
Spot unless the use says on demand (and the check's registry states why).

| Type | Use |
|---|---|
| c8i.2xlarge | benchmark timing (BP-3.10), CPU A/B graniterapids, decoder speed (aws_decode_speed) |
| c8g.2xlarge | benchmark timing (second family), CPU A/B graviton4, decoder speed (aws_decode_speed) |
| c8g.medium, c8g.large | CPU A/B graviton4medium; the charts' Graviton timing (c8g.large) |
| c8g.4xlarge, c8g.8xlarge, c8g.12xlarge, c8g.16xlarge | packed encodes and scoring on Graviton4 (aws_measure, aws_warm, fleet score), native aarch64 builds |
| c7i.4xlarge, c7i.8xlarge, c7i.12xlarge, c7i.16xlarge, c7i.24xlarge | packed x86 encodes and scoring, build box, pipe-cleaners |
| c6i.8xlarge, c6i.12xlarge, c6i.16xlarge, c6i.24xlarge | the same, the cheaper x86 when spot allows |
| c8i.4xlarge | measured candidate (throughput table) |
| c8a.large, c8a.2xlarge, c8a.4xlarge, c8a.metal-24xl | Zen 5: CPU A/B zen5 (on demand, eu-west-1: no c8a in eu-north-1), the charts' x86 timing series |
| c7i.metal-24xl, c8g.metal-24xl, c8i.metal-48xl | the charts' paired bare-metal timing (aws_timing); BP-4 benchmark timing (c8i.metal-48xl, c8g.metal-24xl: procedure D14) |
| c6a.large | the charts' Zen 3 timing series (on demand) |
| c7g.large | the decoder yardstick's Graviton3 core |
| r7i.16xlarge, m7i.16xlarge, c6a.16xlarge | NN fits that need RAM (distill box_run) |

Regions: eu-north-1 (home), eu-west-1 and eu-central-1 (spill, and c8a), us-east-1.

## Quotas and offerings (Service Quotas and describe-instance-type-offerings, 2026-11-24)

| Region | On-demand standard | Spot standard | c-families offered |
|---|---|---|---|
| eu-north-1 | 512 vCPU | **1200 vCPU** | c6i c7a c7g c7i c8g c8i (no c8a) |
| eu-west-1 | 512 | 512 | c6i c7a c7g c7i c8a c8g c8i |
| eu-central-1 | 512 | 512 | as eu-west-1 |
| us-east-1 | 512 | 512 | as eu-west-1 |

- **eu-north-1 sizes:** c8g from `.medium` (1 core); c8i from `.large` (one core plus its
  SMT sibling).
- **Spot is shared.** eu-north-1's spot quota is shared by every agent, the benchmark
  fleet and the build box. On 2026-11-24 at 08:08 it had 7-143 vCPU free.

## Per workload

| Workload | Recommended | Measured | Runner-up | Fall back when |
|---|---|---|---|---|
| 1. Benchmark absolute timing | **c8i.2xlarge spot, eu-north-1** (BP-3.10, unchanged) | 0.3-0.4 % per-encode sd; 0.1-0.4 % drift in 20 min; boxes within 1.3 %. $0.057-0.074/h = **$/encode-hour**. | c8g.2xlarge spot ($0.063-0.124/h), the second family | No spot capacity: wait. Never a .large (see below). |
| 1b. Graviton timing on c8g.medium | **Not for absolute timing** (confirmed on 3 boxes) | Per box quiet (sd 0.4-0.9 %, drift ≤0.4 %), but the boxes differ: AV1 1.549 / 1.553 / 1.620 s (4.6 % apart), x264 1.6 % apart; c8g.2xlarge pair 0.2 % | c8g.2xlarge (unchanged) | - |
| 2. Paired CPU A/B (aws_cpu_ab) | **c8i.2xlarge + c8g.2xlarge spot, eu-north-1** (current default) | as row 1 | `--targets graviton4medium` (c8g.medium, vCPU 0, harness sharing it): A/A +0.21 % [-0.17, +0.59], round sd 0.80 %, $0.0057/h; box bias cancels in a pair | zen5 (c8a.2xlarge, on demand, eu-west-1, $0.46/h): only when x86 AMD is the question |
| 3. Metric scoring (CPU libvmaf), packed: **the fallback** for dev measurement (row 3b is the default) | **Graviton4 (c8g, 29.6k points/$) via `--arch auto`**; else c7i.24xlarge or c6i.24xlarge spot, eu-north-1, whichever is cheaper per vCPU at launch | c7i: 7,294 points/h, **13.0k points/$** at $0.562/h. c6i: 6,149 points/h, **11.7k/$** at $0.526/h (14.1k/$ at its $0.437 best-zone price) | c8i.4xlarge: 1,787/h, 10.5k/$ | Quota: wait. Spill only if needed now (rule 3). |
| 3b. Metric scoring for dev measurement: **the local GPU scoring service, the default** | **`aws_measure --score auto` (the default) for quick, screen, final-check and train batches: encoded on warm c8g boxes, scored by VMAFx CUDA in the PC's scoring service (3 commands at once, CPUQuota 800 %, MemoryMax 10G)**; Graviton CPU scoring (row 3) is the fallback: service full for 10 min, a guard tripped, or `--score cpu` | quick: $0.006 against $0.035 on the CPU path; final-check: $0.015 against $0.146, 16.7 against 30 min; decisions equal, scores within 4.6e-7 (MS-SSIM) (section "Scoring on the PC's GPU") | row 3 | Any guard (section "Scoring on the PC's GPU") |
| 4. Untimed packed encodes | **Graviton4 (c8g) via `--arch auto`**, else the row 3 pair | Our AV1 medium, 720p, 150 frames. c7i: 92,790 encodes/h, **165k/$**. c6i: 94,245/h, 179k/$ (216k/$ at best-zone price) | **c8g.8xlarge (Graviton4): 71,068/h at $0.164/h = 434k/$, 2.6x cheaper**. Needs aarch64 builds and tools, see "Not yet possible" | Same as row 3 |
| 5. Builds and tests (nc-remote) | c7i/c6i .24xlarge spot (current), **recommend 48 vCPU, below** | Cold `cargo build --release -p nanocodec-cli`: c7i.24xlarge 369 s ($0.058); c8i.4xlarge 359 s ($0.017); c8g.8xlarge 243 s ($0.011, aarch64) | c8g.8xlarge for aarch64 work | Reclaim rule (nc-remote) |
| 6. History, retime, other occasional runs | Packed: as row 3. Timed: as row 1. | - | - | - |

**$/encode-hour** is the box's hourly price, because a timing box runs one encode at a time.
Add $0.005/h of public IPv4 and about $0.002/h of root volume to each box. On a c8g.medium
these are about as much as the instance itself.

### 1. Timing: small sizes measured (20-min loop: x264 medium and our AV1 medium, 720p, 150 frames, alternating)

The harness waited in `wait4` during each encode. It ran no sampler and no perf, and took
CPU time from rusage. The timed vCPU was 1, or 0 on the 1-vCPU c8g.medium. There was one
box per size.

The c8i.2xlarge and c8g.2xlarge rows are the BP-3.10 records (`runs-stock2-*`, first 20
minutes, two boxes each). Their harness also ran perf and a 100 ms frequency sampler.

| Box | Spot $/h | x264 CPU s | sd | drift | AV1 CPU s | sd | drift | vs 2xlarge mean (x264, AV1) |
|---|---|---|---|---|---|---|---|---|
| c8g.medium | 0.0057 | 3.074 | 0.42 % | +0.01 % | 1.549 | 0.70 % | +0.36 % | +0.8 %, +2.1 % |
| c8g.large | 0.0181 | 2.998 | 1.30 % | +0.10 % | 1.487 | 1.81 % | -0.06 % | -1.6 %, -2.0 % |
| c8g.2xlarge a/b | 0.063-0.124 | 3.047/3.050 | 0.40/0.45 % | +0.2/-0.3 % | 1.516/1.519 | 1.15 % | +0.8/-0.9 % | reference |
| c8i.large | 0.0142 | 2.243 | **5.25 %** | -1.34 % | 1.305 | **3.25 %** | -0.63 % | +5.5 %, +2.7 % |
| c8i.xlarge | 0.0289 | 2.154 | 1.30 % | **-2.70 %** | 1.272 | 0.56 % | -1.05 % | +1.3 %, +0.1 % |
| c8i.2xlarge a/b | 0.057-0.074 | 2.139/2.113 | 0.42/0.37 % | -0.4/+0.4 % | 1.279/1.262 | 0.32/0.29 % | -0.1/+0.2 % | reference |

*Drift* is the last third's mean over the first third's.

**Findings:**
- **c8i.large is not valid for timing.** The harness and system share the timed core's SMT
  sibling, which gives a 3-5 % sd and windows up to 7.9 % apart. This agrees with c6a.large
  (+7.6 %) and c7i.large (3.2 % round sd).
- **c8i.xlarge is noisier than c8i.2xlarge:** 2.7 % x264 drift and 1.3 % sd. It saves
  about $0.03/h.
- **c8g.medium is quiet per box but not between boxes** (owner-approved confirmation,
  2026-11-24, two more boxes, same loop, `aws-cost-records/runs-confirm`):

  | Box | x264 CPU s, sd, drift | AV1 CPU s, sd, drift | vs c8g.2xlarge mean (x264, AV1) |
  |---|---|---|---|
  | c8g.medium #1 | 3.074, 0.42 %, +0.01 % | 1.549, 0.70 %, +0.36 % | +0.8 %, +2.1 % |
  | c8g.medium #2 | 3.072, 0.42 %, -0.01 % | 1.553, 0.94 %, +0.04 % | +0.8 %, +2.4 % |
  | c8g.medium #3 | 3.122, 0.39 %, -0.14 % | 1.620, 0.84 %, -0.29 % | +2.4 %, +6.8 % |

  The harness shares the only vCPU but only waits (wall/CPU 1.008-1.010). Box #3 is 4.6 %
  slower on AV1 than #1, against 0.2 % between the two c8g.2xlarge boxes, so c8g.medium is
  not valid for absolute timing: the benchmark keeps c8g.2xlarge (no BP change proposed).
  - For a paired A/B the box's bias cancels. aws_cpu_ab's real harness on c8g.medium
    (`--targets graviton4medium`, `pinning.pin_harness(single_cpu=True)`), base = candidate,
    AV1 medium, 4 clips x 5 rounds, 2 boxes: cycles +0.21 % [-0.17, +0.59], instructions
    -0.00 %, round sd 0.80 % (Graviton4 .large 0.69 %, 2026-11-21). At $0.0057/h against
    c8g.2xlarge's $0.063-0.124/h it is a valid, cheaper Arm A/B box. It is not the default:
    dev A/B runs on the benchmark's machines (BP-3.10), so that is the coordinator's call.
- **c8g.large was noisier than c8g.medium** on this one box: 1.3-1.8 % sd.

### 3-4. Packed throughput

Every vCPU ran one worker. The encode batch was 20 encodes per worker. The scoring batch
was 2 scores per worker: libvmaf 3.2.1 with VMAF v1 + float_ms_ssim + psnr, `--threads 1`,
720p, 150 frames.

| Box | vCPU (cores) | Spot $/h paid | Encodes/h | Encodes/$ | Points/h | Points/$ |
|---|---|---|---|---|---|---|
| c7i.24xlarge (Sapphire Rapids) | 96 (48) | 0.5618 | 92,790 | 165k | 7,294 | 13.0k |
| c6i.24xlarge (Ice Lake) | 96 (48) | 0.5261 (best zone 0.4368) | 94,245 | 179k (216k) | 6,149 | 11.7k (14.1k) |
| c8i.4xlarge (Granite Rapids) | 16 (8) | 0.1699 | 26,559 | 156k | 1,787 | 10.5k |
| c8g.8xlarge (Graviton4) | 32 (32) | 0.1639 | 71,068 / 71,343 | **434k** | 4,853 | **29.6k** |
| c7a.24xlarge (2026-11-23, excluded) | 96 (96) | 0.6148 | - | - | 9,590 | 15.6k |

- **Encodes:**
  - Graviton4 gives about 2.6x the packed encodes per dollar of any x86 box. Each vCPU is
    a physical core, and its spot price per vCPU was the lowest (c8g.8xlarge $5.12 per
    1000 vCPU-h).
  - c8g.16xlarge was $0.35/h ($5.47 per 1000 vCPU-h); no larger c8g was cheaper per vCPU.
- **Scoring:** c7i and c6i are within ±10 % per dollar, depending on the zone price. Take
  whichever is cheaper per vCPU at launch: `aws_warm` already picks the cheapest per vCPU.
- **Spot reclaim:** the c8i.4xlarge was reclaimed 11 minutes in, after its build.

### 5. Build box

| | c7i.24xlarge | c8i.4xlarge | c8g.8xlarge (aarch64) |
|---|---|---|---|
| Cold `cargo build --release -p nanocodec-cli` (fat LTO), `-j nproc` | 369 s | 359 s | 243 s |
| $ per cold release build | $0.058 | $0.017 | $0.011 |
| Gate-profile `cargo test --workspace --lib --bins` | 110 s | (reclaimed) | 63 s |

- **Test timing caveat:** the test runs stopped at `help::tests::cli_reference_is_current`,
  which needs `docs/` (not uploaded). So the test times are mostly compile time.
- **A release build does not use 96 vCPUs.** The fat-LTO link is serial, so a 16-vCPU box
  builds as fast. The 96-vCPU box earns its size only from concurrent jobs.
- **Utilisation** (CloudWatch CPUUtilization, every build box in eu-north-1, last 3 days,
  61 box-hours, 5-minute averages):

  | Statistic | CPU |
  |---|---|
  | mean | 15.2 % |
  | median | 7.7 % |
  | p75 | 15.6 % |
  | p90 | 35.7 % |
  | p95 | 82.7 % |

  - 33.5 % of the time was under 5 %, and 92.7 % was at or under 50 % (48 vCPU).
  - On one `nc-remote status` sample (08:17 UTC) all 12 slots were free.
  - The box runs 17-26 h a day for about $10-22/day. In total: c6i $77.5, c7i $38.5 and
    c7a $33.1 under `buildbox`, plus the overflow pool.
- **Warm measurement boxes idle out after 10 min** (coordinator, 2026-11-24; was 20: about
  -$13.5 net over the ledger period for 39 extra launches, replayed on 131 boxes).
- **The build box's idle watchdog (10 min, overflow 20 min) is short enough.** The box is rarely idle
  for 10 minutes: it is mostly up but nearly empty. The cost is the size, not the idle
  tail.
- **Recommendation (coordinator decision; not changed here):** a 48-vCPU box (c6i.12xlarge
  $0.218/h or c7i.12xlarge $0.43/h spot today) with the existing overflow box for the p95
  bursts.
  - This is half the price or less.
  - Caveat: CloudWatch measures CPU, not slot occupancy. The 2026-11-21 change to 96 vCPU
    was made for slot queueing (verify, identity, callgrind). Log slot occupancy for a day
    first.
  - Separately, aarch64 builds for aws_cpu_ab are 5 min cold, cross-compiled on the x86
    box. A native c8g.8xlarge builds in 4 min for $0.011.

## Graviton4 packed path (owner, 2026-11-24: the default)

- **Tools:** the aarch64 standard decoders (dav1d, dav2d, ffmpeg-decode, ldecod, TAppDecoder,
  vvdecapp, aomdec, vpxdec, avmdec) and libvmaf 3.2.1 with its models, built by
  `tools/anchors/build_anchors.sh` on one c8g.8xlarge (5.2 min, $0.03), are in
  `~/nanocodec-data/anchors/aarch64` (`SHA256SUMS-c8g-decoders`, `build-flags-c8g-decoders.json`).
  They are BSD/LGPL builds as on x86 (ffmpeg-decode is the LGPL decode-only build); the clip
  decoder is Ubuntu's ffmpeg package, `8.0.1-3ubuntu2` on arm64 as on the PC.
- **Parity, same streams** (`aws-cost-records/parity`):
  - 13 decodes (6 codecs, every standard decoder plus ffmpeg-decode) of our streams: every
    output identical (SHA-256) to x86's.
  - libvmaf, 30 frames of a 720p AV1 recon: VMAF v1, float_ms_ssim, PSNR and every
    feature identical per frame to the printed digit.
  - The same `nceval.screen` (h264 faster, 2 clips, 60 frames) on c8g.4xlarge and on an
    x86 warm box: the same streams and decision (BETTER); VMAF fields identical, PSNR to
    1e-13, the MS-SSIM BDs (nceval.msssim, numpy) within 4.8e-5 percentage points, J_real
    within 1.3e-5 pp (the PC-GPU against cloud-CPU gap was 8e-6 pp).
- **Throughput:** c8g.8xlarge scores 29.6k points per dollar against c7i.24xlarge's 13.0k
  (2.3x) and encodes 434k against 165k (2.6x).
- **The switch:** `aws_measure --arch auto|x86_64|aarch64` (`DEFAULT_ARCH = "auto"`, owner, 2026-11-24).
  - `auto` takes aarch64 for nceval.screen / quick / levels when every build is a clean commit
    (its aarch64 release build comes from aws_cpu_ab's cache, else the build box: 188 s
    cold) and the aarch64 anchors are here.
  - Otherwise, or when no c8g box launches, it takes x86, with the reason printed.
  - Warm boxes keep to their machine: `aws_warm.ami_arch`, family c8g at 16/8/12xlarge.
  - Pipe-clean (`--arch aarch64 --cold`, c8g.4xlarge): 3.9 min, $0.01; prepare 1.4 min.

## Scoring on the PC's GPU (2026-11-24)

`aws_measure --score gpu|auto|cpu` (`eval/nceval/gpuscore.py`; docs/PROCEDURES.md "Encode on
boxes, score on the PC's GPU"). `auto` is the default and takes the GPU for nceval.quick,
nceval.screen, nceval.levels final-check, nceval.fixedqp ab and a `--batch` (train's
screens, side by side in one scope); `--score cpu` keeps today's CPU scoring on a box.
`fixedqp gap` and `cost` stay on boxes (gap's reference encoders run x86 quality builds).
**Encode rounds are packed (2026-11-24):** a 26-quick batch whose commands each ran
their own round launched 14 new boxes ($1.19 against ~$0.3). Now every GPU-scored run
(a batch or one command) has one round broker: its first round carries every command at
once, later rounds carry every request waiting together, and rounds run one at a time, so a
batch holds at most one warm box. `aws_measure` prints the planned first round's dollars
before it starts and refuses a batch whose plan passes `--budget`; the broker refuses a
round once the batch's rounds have cost `--budget`. After the run's last round its box
stays 2 min for a follow-up round, then the broker stops it unless another job is on it
(`aws_warm.py stop-if-idle`); that idle time is counted in the batch's cost (the 10-min idle
timer had cost ~$0.03, more than a 6-quick batch's round). Each result carries `batch_rounds`,
`batch_rounds_usd` and `batch_boxes`.
fixedqp ab's first round runs ahead of its rate probes: the box codes the probes that choose
the quantizers and every point, and the probes come back without their CPU times. The command runs here with VMAFx CUDA; each batch's missing points
are encoded on a warm Graviton box (an encode round: the command run encode-only, replayed
from the results this run has used); the PC decodes each stream once, checks it against the
encoder's recon digests and scores it. Results enter the local point cache under today's
keys.

**Guards** (unit-tested; each falls back to the Graviton CPU path, mid-run by exit 75):
**one scoring service** for every agent: up to 3 commands score at once (places under
`~/nanocodec-data/run/gpu-score/`), all in one systemd slice (`nanocodec-gpu-score.slice`:
CPUQuota 800 %, MemoryMax 10G for them all), on one VMAFx server and 4 machine-wide GPU
slots, each command with an equal share of the workers; a command waits for a place up to
10 min (`--gpu-wait`), then scores on a box's CPU. (Until 2026-11-24 one run at a time:
other agents' quicks held it most of the time and fell back to paid CPU scoring.)
8 GB free to start and 4 GB before each batch (nvidia-smi, 3-minute wait), at any hour (no fixed
window since 2026-11-24: the free-memory gate covers anything else on the card); the disk budget
before every batch.

**Measured** (h264 faster + quant.trellis=1, d51e02c3, draw 2026-11-24):

| Run | Points | PC CPU | PC peak memory | VMAFx server VRAM | Wall | AWS |
|---|---|---|---|---|---|---|
| screen pipe-clean, 2 clips, 60 frames | 24 | 27.6 s (1.15 s a point) | 1.40 GB | - | 4.3 min (2.5 min a new warm box) | $0.013, 2 rounds |
| quick, first (two rounds at once, fixed in ad955800d) | 164 | 1008 s | 4.30 GB (the cap, page cache included) | 1.98 GB peak; card free ≥ 9.5 GB | 9.0 min (two new warm boxes) | $0.046 |
| quick, fixed | 164 (116 x 150 f, 24 x 600 f, 24 x 900 f = 356 150-frame equivalents) | 972 s = **2.73 s per 150-frame point**; the round's own CPU 5.5 s | 1.25 GB | as above | 5.6 min | **$0.006**, 1 round |
| quick, Graviton CPU path (before) | 164 | - | - | - | 7.5 min run on c8g.8xlarge | **$0.035** |
| final-check, h264 faster d51e02c3 vs 3212a2c47, `--estimator single`, GPU (shared source decode) | 303 (20 x 150 f, 167 x 600 f, 116 x 900 f = 1,384 150-frame equivalents) | 3,148 s = **2.27 s per 150-frame point** | 3.0 GB | - | 16.7 min | **$0.015**, 2 rounds |
| the same final-check, Graviton CPU path (before) | 303 | - | - | - | 26.0 min run (+ 4.3 min launch) on c8g.12xlarge | **$0.146** |
| fixedqp ab, h264 faster + trellis, 3 clips, 60 frames, GPU | 24 (+ 24 rate probes from the box, none coded here) | 38.6 s in its own scope (the VMAFx server ran in another) | 0.5 GB | - | 9.0 min, of which 4.8 a new warm box | **$0.024**, 1 round |
| the same fixedqp ab, CPU path on x86 (since d6ce3ca0f its CPU path is Graviton: parity run, same streams, J within 1.0e-5 pp, $0.009 against $0.026) | 24 | - | - | - | 6.8 min, of which 4.9 a new c6i.24xlarge | **$0.052** |
| batch of 2 screens (2 clips, 60 frames; one all cached) | 7 + cached | - | - | - | 1.6 min | $0.006, 1 round |

- **PC decode per point** (ffmpeg-decode, H.264, one thread, process start included): 0.06-0.34
  CPU-s at 150 frames (360p-720p), 0.39 at 900 frames (360p), 0.76 at 600 frames (1080p). The
  rest of the 2.73 s is not broken down yet. The scope also holds the reference clip's decode,
  the frames' path through Python and the VMAFx client and server.
- **fixedqp ab equality (24 points):** streams, VMAF and PSNR equal (PSNR to 1.4e-14),
  MS-SSIM within 8.5e-7; BD-J -2.766428 (GPU) against -2.766434 (CPU), 6.2e-6 pp. Both runs
  launched a new box, which dominates their wall time and cost.
- **Final-check equality (298 points on both paths):** streams, PSNR and every per-clip
  quality identical; MS-SSIM within 4.6e-7, a frame's VMAF v1 within 4e-6 on 18 points;
  NEUTRAL on both.
- **NVDEC for the source decode was tried and not used:** identical frames on 11 pool clips
  (H.264), but no less CPU (0.5-1.2 against 0.2-1.2 CPU-s a 150-frame window): the cost is
  moving raw frames, not decoding. A clip window's points in flight share one decode instead.
- **Equality, the same 164 points on both paths:** streams, VMAF v1 and PSNR identical; MS-SSIM
  within 3.4e-6. Quick L1 J -3.408837 (GPU) against -3.408841 (CPU), 3.8e-6 pp; probe mean
  3.2e-5 pp (the 900-frame share 6.2e-5, one clip 3.1e-4); decision CONTINUE on both. The
  screen pipe-clean: J -3.521278 against -3.521288 on Graviton (1.1e-5 pp), BETTER on both.

## What the tools do by default now

- **A fallback from GPU to CPU scoring is loud and recorded (2026-11-24).** Every result
  has `scored_on` (GPU on the PC, or CPU on a box) and, after a fallback, `fallback_reason`
  (`disk floor: ...`, `GPU memory: ...`, `queue wait: ...`, `service memory: ...`, `encode
  round ... failed`); ledger rows carry the same in their `source`. The fallback prints one
  `!!!` line with the CPU path's planned cost against the GPU path's, and is refused
  (nothing launched) when that CPU cost passes `--budget`. (A batch had gone to CPU on the
  disk floor at about 5x the cost, silently.)

- **A run is planned against the run cap before launch (2026-11-24).** `aws_measure` plans
  a screen, quick, final-check or fixedqp ab run as max(job_points x 60 vCPU-s / box vCPUs,
  its longest command's floor: 26 min a long-window final check, 8 min a quick) and needs it
  under 0.8 x `--run-minutes`. A run over it takes the machine's largest auto size; a batch
  that fits no one box runs as sub-batches on their own boxes (first fit, largest first);
  a command over it on its own is refused, nothing launched. (An 8-screen L3-ladder batch,
  3,856 planned points, had hit the 35-min cap on one c8g.12xlarge with none of its 8
  results back.)

- **Scoring for dev measurement (quick, screen, final-check, train):** the local GPU scoring
  service by default (`aws_measure --score auto`: encodes on warm c8g boxes, VMAFx CUDA here,
  3 commands at once in `nanocodec-gpu-score.slice`, CPUQuota 800 %, MemoryMax 10G); Graviton
  CPU scoring on a box is the fallback (`--score cpu`, a full service for 10 min, or any guard).
- **Packed x86 boxes** use the c7i/c6i pair; no c7a, m-family, 48xlarge or metal:
  - `aws_session.THROUGHPUT_X86_TYPES` = c7i.24xlarge, c6i.24xlarge.
  - `aws_measure.AUTO_SIZES` = c7i/c6i at 8/12/16/24xlarge.
  - `aws_warm.WARM_TYPES` = c7i/c6i at 24/16/12xlarge, cheapest per vCPU first.
  - `aws_heldout_chart` and `heldout_fleet` default to c7i.24xlarge (pipe-cleaner
    c7i.4xlarge).
- **`tools/dev/aws_choose.py`** is the check for rule 2.
- **Build-box occupancy is logged** from d51e02c3f (2026-11-24 08:56 UTC): every job's slot,
  wait and run time (`nc-remote occupancy --hours 24`). Resize after 24 h, not before.
- **Unchanged:**
  - `tools/bench/fleet.py`: c8i.2xlarge, BP-3.10.
  - `aws_cpu_ab`: c8i.2xlarge + c8g.2xlarge spot, eu-north-1.
  - `nc_remote` sizes: see the build box recommendation.

## Recommendations for the owner (not applied)

1. **Graviton timing on c8g.medium: rejected for the benchmark** (4.6 % between boxes on
   AV1, above). Offered for paired A/B only (`graviton4medium`, coordinator's call).
2. **Build box at 48 vCPU** with overflow, after a day of slot-occupancy logging.
3. **Packed work on Graviton4: prepared, switched on after the benchmark run** (see
   "Graviton4 packed path").

## Not yet possible / not measured

- **Between-box spread** for c8g.large, c8i.large and c8i.xlarge: one box each, by the
  owner's budget (c8g.medium: three, above).
- **Whole test suite:** time on any box (see the test caveat).
