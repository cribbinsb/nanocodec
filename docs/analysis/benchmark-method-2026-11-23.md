# How to run the published benchmarks: measurements (2026-11-23)

The owner is redoing every published result on three datasets ("ours", VC-1K, AOM-CTC), each in
CBR and Fixed QP, for all our rungs and the references. The binding rule is that **each encode
runs once, timed as it happens, and that one run yields both the stream (scored for quality
afterwards) and the time**. This note measures how to do that: on which machine and with what
packing, with what noise and how many rounds, where to score the metrics, the cost and wall time,
and what start-up, input I/O and concatenation do to the time. It also settles the reported
"ISA bug". The normative procedure built on it is [docs/benchmark-procedure.md](../benchmark-procedure.md).

Every number below is a measurement made today on rented AWS machines, unless it is marked as
coming from a cited document or as not measured. The PC did no compute: it ran two sub-second
ISA probes (40 encodes of a 480x360 clip) and the analysis. The scratch tools and raw records
are kept in `~/nanocodec-data/bench-method-records/` (`tools/`, `runs*/`, `*.txt`; retention
entry `bench-method-records`).

## Summary

| Question | Answer (measured) |
|---|---|
| One run for both axes? | Yes. 8,046 timed encodes coded identical streams on 7 machine types and configurations. The stream of a timed run is the stream to score. |
| Valid timing box | **c8a.large (Zen 5, no SMT), one encode at a time on vCPU 1.** One encode alone on a c8a.24xlarge or c8a.48xlarge gives the same time (0.997, 1.011). No packed configuration is valid: 1/4 density on a c8a.48xlarge adds +6.8 %, full density +81 %, and fast encoders are slowed most. |
| Noise | Single run on c8a.large: sd 0.57-0.94 % (per job, three rounds). Box-to-box bias over 5 Zen 5 boxes: -0.5 to +0.9 %. One run per encode is enough; the aggregate's precision comes from the many encodes per encoder spread over many boxes. |
| Anchor pairing (x264 medium beside every encode) | Not needed. It doubles the cost, and on one box of five it made the bias worse (1.006 to 0.980). Per-box canary sets replace it. |
| Start-up | Process plus library start: 0.7-1.5 ms for every encoder. Our encoders' time outside frame encoding (init, output, exit): 2.4-6.8 ms per 150-frame encode, 0.3-1.4 %. The large "intercepts" (up to 7 % of a 150-frame encode) are the first key frame's encode, which depends on content and rate: real work, not overhead. Time the whole process; subtract nothing. |
| Input I/O | See "Input I/O". Read cost from RAM is small against encode cost; a cold disk read is not. Inputs go on tmpfs and are checked resident. |
| Concatenating clips into one encode | See "Concatenation". Rejected. |
| Metric scoring | Cloud CPU: libvmaf 3.2.1 (VMAF v1 + float_ms_ssim + psnr) costs 20.2 CPU-s per 720p 150-frame point alone and 35.3 CPU-s at 94 per c7a.24xlarge, which is 9,600 points/h for $0.61/h spot, about $0.00006 per 720p point. The PC's GPU (documented, not re-measured) does about 19,000 points/h but needs about 2.2 PC CPU-s per point. Score on cloud CPU boxes. |
| The "Zen 4 ISA bug" | **Not a CPU or ISA difference: a build mix-up.** Every build codes one stream on Zen 3, Zen 4, Zen 5 and Intel Granite Rapids, on every tier (auto, avx512, avx2, scalar), alone or on a loaded box. 20d31869 and c9cff89f code 27,444 B, while 4fc9a1de and 207745d7 code 27,433 B. The chart's quality path and timing path ran different commits for H.264 veryfast. |
| Spot capacity | c8a.48xlarge and c8a.12xlarge had no spot capacity in eu-west-1, eu-central-1 or us-east-1. Spot placement scores for c8a sizes are 1-3 of 10 in six regions. c8a.large on demand launched at once, every time (13 launches). |
| Cost | About $0.02 to encode and time one 720p clip in one mode (68 families x 6 points, 9.5 core-minutes on c8a.large on demand), plus about $0.03 to score it on spot c7a. Section "Cost and wall time" has the full table. |

AWS spend for this study: see "Spend" at the end.

## 1. One run for both axes

The design the owner fixed: every encode runs exactly once, pinned to a core that gives a valid
time, and writes its stream. The timed process's CPU time is the speed point. Its stream is
decoded by a standard decoder and scored afterwards, anywhere, because scoring needs no timing.
Quality and speed then come from the same encodes by construction.

The scratch runner did this end to end on every box. It ran each encode with `taskset` and
recorded `wait4` rusage, wall time, and the stream's byte count and SHA-256:
`tools/bm_remote.py` in the scratch directory. Its streams are the scored streams.

**Stream identity across machines (the precondition):**
- All 8,046 timed encodes coded identical streams on every machine and configuration: 108 chart
  commands + 54 start-up variants, each a unique command.
- The machines and configurations were c8a.large x3, c8a.24xlarge, c8a.48xlarge, c7a.24xlarge and
  c8i.24xlarge, each serial and packed.
- The encoders were our six codecs and x264, x265, openh264, SVT-AV1, libaom and libvpx at the
  chart's flags.
- No encode failed.

So a stream from a timed run on any of these machines is the stream any other machine would make.

## 2. Valid timing per configuration

**Jobs:** 108 encodes = 27 encoders x 4 clips of the main chart:
- `crowd_mot20_06` (1884x720 25 fps, 350 kbps), `handheld_mot17_10` (1200), `lowlight_bdd_00ce95b0`
  (300), `pan_jaad_0101` (1200);
- 150 frames each, the chart's CBR commands.

**Encoders:** ours at H.264 veryfast/medium/slower, HEVC faster/medium/veryslow, AV1
veryfast/medium/slower, VP9 veryfast/medium, AV2 faster/medium, VVC faster/medium; x264
ultrafast/veryfast/medium, x265 ultrafast/medium, openh264, SVT-AV1 M8/M13, libvpx rt 7/9 and
libaom rt 8/10.

**Build:** ours is today's head 0d9f32bc, plain release (fat LTO). The references are the timed
reference builds (`anchors/x86_64-opt`).

**Statistic:** the median CPU time (user+sys) per job, as a ratio to the same job on c8a.large #1
run serially.

**Packing:** a packed configuration keeps every slot busy until the last recorded encode ends
(unrecorded filler encodes).

| Configuration | Concurrent | Bias (geo-mean ratio) | Per-job sd | p5-p95 | Encoder rank rho | Relative move between encoders | Single-run sd |
|---|---:|---:|---:|---|---:|---|---:|
| c8a.large #1, serial (reference) | 1 | 1.000 | - | - | - | - | 0.57 % |
| c8a.large #2, serial | 1 | 1.007 | 0.9 % | 0.996-1.022 | 1.000 | 0.989-1.023 | 0.94 % |
| c8a.large #3, serial | 1 | 1.002 | 0.7 % | 0.993-1.015 | 1.000 | 0.989-1.017 | 0.69 % |
| c8a.large, both vCPUs busy | 2 | 1.042-1.053 | 2.4-2.7 % | 1.01-1.10 | 0.999 | 0.96-1.06 | 1.6-2.0 % |
| c8a.24xlarge, one encode | 1 | 0.997 | 0.6 % | 0.988-1.007 | 1.000 | 0.991-1.008 | - |
| c8a.24xlarge, every 2nd core | 47 | 1.290 | 5.0 % | 1.19-1.44 | 0.999 | 0.92-1.06 | 3.1 % |
| c8a.24xlarge, all cores | 94 | 1.725 | 7.5 % | 1.54-1.96 | 0.999 | 0.89-1.16 | 3.1 % |
| c8a.48xlarge, one encode | 1 | 1.011 | 0.8 % | 0.997-1.023 | 1.000 | 0.981-1.012 | - |
| c8a.48xlarge, every 4th core | 48 | 1.068 | 2.5 % | 1.04-1.12 | 1.000 | 0.977-1.039 | 2.1 % |
| c8a.48xlarge, every 2nd core | 95 | 1.320 | 4.5 % | 1.23-1.43 | 0.999 | 0.93-1.05 | 2.9 % |
| c8a.48xlarge, all cores | 190 | 1.812 | 7.2 % | 1.63-2.06 | 0.998 | 0.89-1.14 | 3.1 % |
| c7a.24xlarge (Zen 4), one encode | 1 | 1.481 | 3.6 % | 1.39-1.57 | 0.998 | 0.94-1.07 | - |
| c7a.24xlarge, every 2nd core | 47 | 1.746 | 5.8 % | 1.58-1.92 | 0.999 | 0.90-1.10 | 2.5 % |
| c7a.24xlarge, all cores (today's quality boxes) | 94 | 2.592 | 11.3 % | 2.14-3.12 | 0.994 | 0.82-1.25 | 3.4 % |
| c8i.24xlarge (Granite Rapids), one encode | 1 | 1.405 | 7.0 % | 1.27-1.56 | 0.995 | 0.89-1.11 | - |
| c8i.24xlarge, one per physical core | 46 | 1.548 | 8.5 % | 1.38-1.80 | 0.993 | 0.90-1.15 | 1.1 % |
| c8i.24xlarge, both SMT threads | 92 | 2.567 | 8.7 % | 2.30-2.96 | 0.996 | 0.90-1.17 | 3.2 % |

**How to read the columns:**
- *Encoder rank rho:* the Spearman correlation of the 27 encoders' mean log time with the reference's.
- *Relative move:* each encoder's ratio over the configuration's bias, as the min-max over encoders.
- *Single-run sd:* the pooled sd of one run around the job's mean, over that configuration's repeats.

**Topology (lscpu on the boxes):**
- c8a.large, c8a.24xlarge, c8a.48xlarge and c7a.24xlarge have one thread per core (no SMT): each vCPU is a physical core.
- c8a.24xlarge: 96 cores, 12 L3 instances.
- c8a.48xlarge: 2 sockets, 192 cores, 24 L3 instances.
- c8i.24xlarge: 48 cores x 2 threads; vCPU n and n+48 are siblings.
- Option (c) of the brief, "both SMT siblings busy", exists only on the Intel box. There it adds
  +66 % to each encode's CPU time over one encode per physical core (2.567 / 1.548).

**Findings:**
- **Packing is not valid timing on any box tried.** On Zen 5 even one encode per four cores adds
  6.8 % and moves encoders by ±3 % relative to each other. Full density adds 72-81 % and moves them
  by -11 to +16 %.
- **Contention penalises the fast encoders most.** At full density on c8a.48xlarge, openh264 is
  slowed 2.07x, SVT M13 2.00x, our HEVC faster 1.98x and H.264 veryfast 1.97x. x265 ultrafast is
  slowed 1.62x, x265 medium 1.64x and our HEVC veryslow 1.61x. On c7a.24xlarge the spread is
  2.12x (x265 medium) to 3.24x (SVT M13). Per-encoder table: `fam-table.md` in the scratch
  directory. A packed timing therefore understates fast encoders' speed relative to slow ones by
  up to about 25 %.
- **Today's 2.0-3.2x "inflation" decomposes** into the CPU generation (Zen 4 alone: 1.48x on this
  set) and contention (1.75x more at full density).
- **The frequency snapshot fits this.** A single busy vCPU reported 4,519-4,535 MHz on c8a and
  3,696-3,716 MHz on c7a. The rest reported the 2,600 MHz nominal. The snapshot is from
  `/proc/cpuinfo` at the end of each configuration, so frequency under full load was not
  captured; "frequency vs L3/memory" is not separated here.
- **c8a.large's L3 is not exclusive.** lscpu reports an 8 MiB L3 slice on c8a.large; the host's
  other tenants share the rest. Even so, three c8a.large and two quiet big boxes agree within
  -0.5 to +1.1 %. As measured today, neighbours did not move c8a.large times.
- **Steal time** was at most 0.07 % of any timed CPU's jiffies in any configuration.
  `/proc/stat` was read per configuration.

## 3. Run-to-run noise, rounds and anchors

**Same box:**
- A single run's deviation from its job's mean has sd 0.57 %, 0.94 % and 0.69 % on the three
  c8a.large boxes (3 rounds each, rotated order).
- The median absolute deviation per encoder is 0.13 % (libaom rt 8) to 0.48 % (our HEVC faster).
  The fastest encodes are the noisiest: 0.2-0.5 s encodes, where a few ms of jitter matter.

**Across boxes:** the box biases against the median of five Zen 5 boxes (each job's median over
the boxes) were:

| Box | Bias | Per-job sd | x264-medium anchor (4 jobs) | Bias after normalising by the anchor |
|---|---:|---:|---:|---:|
| c8a.large #1 | 0.998 | 0.43 % | 0.996 | 1.002 |
| c8a.large #2 | 1.006 | 0.74 % | 1.026 | 0.980 |
| c8a.large #3 | 1.000 | 0.47 % | 1.001 | 0.999 |
| c8a.24xlarge (alone) | 0.995 | 0.51 % | 0.996 | 1.000 |
| c8a.48xlarge (alone, us-west-2) | 1.009 | 0.69 % | 1.005 | 1.004 |

**Rounds:** one. Each encode runs once (the binding rule).
- An encoder's speed is a geometric mean over many encodes: clips x points, e.g. 100 clips x 6
  points = 600.
- The single-run noise then averages to about 0.6 % / sqrt(600) = 0.02 %.
- The box bias averages over the boxes the encodes were spread across. With random assignment to
  B boxes and a box sd of about 0.5 %, it is about 0.5 % / sqrt(B): 0.05 % at B = 100.
- The chart's speed differences between encoders are tens of percent, so the precision is ample.
  The limiting uncertainty is the clip sample: a bootstrap over clips.

**Anchors:** pairing every encode with an x264-medium anchor on the same core is not needed:
- It doubles the CPU cost.
- With a few anchor encodes per box it adds noise. Box #2's anchor read +2.6 % while the box was
  +0.6 %.
- Detecting a bad box is done better by a canary set (procedure, "Machines"): the same 12 short
  encodes at the start and end of every box, compared with the reference table, refusing the box
  beyond ±2 %. In the table above, every box's 4-job anchor was within ±2.6 %.

**Not measured:**
- the box bias over a large fleet. Five boxes is a small sample; the canary sets will measure it
  in every run;
- run-to-run noise on other days;
- noise on Graviton and the Jetson.

## 4. Start-up overhead

**Design** (owner's request):
- c8a.large, one pinned core, sources raw in the page cache.
- 23 encoders at their chart flags: ours at medium for all six codecs plus H.264/AV1/VP9
  veryfast and HEVC faster; x264 medium/veryfast/ultrafast, x265 medium/ultrafast, openh264,
  SVT-AV1 M8/M12/M13, libaom rt 8/10, libvpx rt 7/9.
- 3 resolutions: 1280x720, and 960x540 and 640x360 bicubic-scaled from it.
- 3 clips: handheld, low light, pan.
- 2 CBR rates: 300 and 1200 kbps at 720p, scaled by (pixels)^0.75.
- N = 1, 2, 5, 10, 30, 75, 150 frames, with 10 repeats in shuffled order.
- Plus the bare process: `--version`, or `--help` for aomenc, vpxenc and openh264-harness.
- 29,210 encodes on 8 boxes, 0 failures. Fit t(N) = a + b*N per encoder x resolution x clip x
  rate. Full table: `su-table.txt` / `su.csv` in the scratch directory.

**Results:**

| Encoder (720p) | Bare process | Fitted a (median [min, max] over clips x rates) | t(1) | a / t(150) | t(1) at 300 / 1200 kbps |
|---|---:|---|---:|---:|---|
| x264 ultrafast | 0.8 ms | 3.7 [3.2, 4.0] ms | 3.9 ms | 1.8 % | 3.8 / 4.1 |
| x264 veryfast | 0.8 | 3.6 [1.1, 5.6] | 7.3 | 0.7 % | 6.8 / 8.2 |
| x264 medium | 0.9 | -2.0 [-10.8, 5.6] | 10.6 | 0.0 % | 9.0 / 12.1 |
| x265 ultrafast | 1.3 | 31.0 [29.4, 35.3] | 42.3 | 1.8 % | 38.6 / 46.9 |
| x265 medium | 1.4 | 112.9 [105, 137] | 137.0 | 3.3 % | 124.5 / 153.6 |
| openh264 | 1.0 | 8.4 [5.5, 11.7] | 9.8 | 2.8 % | 9.6 / 10.2 |
| SVT-AV1 M13 | 0.8 | 16.0 [13.7, 18.8] | 16.1 | 4.7 % | 15.8 / 16.3 |
| SVT-AV1 M12 | 0.7 | 16.7 [13.0, 21.9] | 17.2 | 4.4 % | 17.1 / 17.2 |
| SVT-AV1 M8 | 0.9 | 74.3 [52.7, 104] | 80.8 | 4.5 % | 60.7 / 92.5 |
| libvpx rt 9 | 0.9 | 22.3 [20.2, 22.6] | 24.1 | 3.3 % | 23.1 / 25.7 |
| libvpx rt 7 | 0.8 | 25.6 [19.6, 26.2] | 29.0 | 2.4 % | 26.6 / 32.9 |
| libaom rt 10 | 1.5 | 51.3 [33.8, 79.1] | 56.2 | 2.9 % | 49.0 / 63.4 |
| libaom rt 8 | 1.2 | 58.0 [40.2, 90.5] | 76.2 | 2.9 % | 61.9 / 97.2 |
| ours H.264 veryfast | 1.0 | 11.9 [11.0, 13.3] | 11.7 | 3.8 % | 10.7 / 12.0 |
| ours H.264 medium | 0.8 | 14.8 [12.3, 23.0] | 15.1 | 2.3 % | 13.2 / 15.5 |
| ours HEVC faster | 0.8 | 14.5 [13.0, 17.1] | 14.2 | 3.5 % | 12.9 / 15.7 |
| ours HEVC medium | 0.8 | 68.9 [43.6, 99.3] | 67.7 | 6.3 % | 44.3 / 79.0 |
| ours AV1 veryfast | 0.8 | 20.9 [16.6, 27.0] | 20.2 | 5.3 % | 16.4 / 24.0 |
| ours AV1 medium | 1.1 | 53.0 [34.8, 68.5] | 49.3 | 5.2 % | 35.5 / 58.4 |
| ours VP9 veryfast | 0.7 | 20.1 [14.6, 24.2] | 20.1 | 5.0 % | 16.5 / 24.0 |
| ours VP9 medium | 1.0 | 57.2 [39.8, 73.1] | 56.2 | 5.3 % | 43.1 / 67.6 |
| ours AV2 medium | 0.9 | 64.4 [44.2, 90.8] | 66.9 | 5.7 % | 46.4 / 74.8 |
| ours VVC medium | 0.9 | 101.9 [71.2, 140] | 104.6 | 7.2 % | 75.5 / 123.6 |

At 360p, a / t(150) is 2.4-7.5 % (x264 ultrafast 4.3 %, x264 medium about 0 %).

**What "a" is.** For our encoders the CLI's per-frame timer (`--rate-log` `encode_us`: reading
the frame plus encoding it) splits t(1) and t(150). 10 repeats; handheld clip; 1200 kbps scaled:

| Ours (720p) | t(1) | Frame 0 encode | Outside frames at N=1 | t(150) | Outside frames at N=150 | Key frame / mean P frame | Frames 1-9 / frames 140-149 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H.264 veryfast | 11.6 ms | 8.6 | 3.0 | 427 | 4.6 (1.1 %) | 3.2 | 1.00 |
| H.264 medium | 16.6 | 13.8 | 2.8 | 1019 | 4.6 (0.5 %) | 2.0 | 1.05 |
| HEVC faster | 16.4 | 13.1 | 3.2 | 547 | 4.8 (0.9 %) | 3.7 | 1.16 |
| HEVC medium | 80.5 | 77.0 | 3.5 | 1300 | 5.3 (0.4 %) | 9.4 | 1.21 |
| AV1 veryfast | 25.0 | 21.7 | 3.3 | 516 | 5.4 (1.0 %) | 6.7 | 1.22 |
| AV1 medium | 58.8 | 55.5 | 3.3 | 1242 | 4.6 (0.4 %) | 6.9 | 1.25 |
| VP9 veryfast | 24.5 | 21.7 | 2.8 | 510 | 4.4 (0.9 %) | 6.7 | 1.18 |
| VP9 medium | 68.5 | 65.6 | 2.9 | 1444 | 4.7 (0.3 %) | 7.2 | 1.24 |
| AV2 medium | 75.7 | 72.8 | 2.8 | 1608 | 5.1 (0.3 %) | 7.2 | 1.15 |
| VVC medium | 125.1 | 121.4 | 3.6 | 1876 | 6.8 (0.4 %) | 10.5 | 1.18 |

At 360p, the outside-frames time is 1.8-2.5 ms at N=1 and 2.4-3.5 ms at N=150.

**Conclusions:**
- **Process and library start-up are about 1 ms** (0.7-1.5 ms) for every encoder. Our
  encoders' non-frame time is 2-4 ms at N=1 and 2.4-6.8 ms at N=150, which includes writing the
  stream: 0.3-1.4 % of a 150-frame encode.
- **The intercept a is the first key frame.** Ours at medium spends 2-12x a P frame on it. It
  varies with content (our HEVC medium t(1): 77-107 ms over clips) and with rate (44 vs 79 ms at
  300 vs 1200 kbps). The repeat sd of t(1) is 0.2-1.5 ms, so it is repeatable but not a constant.
- **It cannot be measured once and subtracted:** it is encode work that depends on content and
  rate.
- **It need not be timed separately either.** Every encoder codes the same structure: 150
  frames, one key frame. The whole process's CPU time is the honest cost of "150 frames
  including one key".
- **The early frames:** ours spends 0-25 % more per frame on frames 1-9 than on frames 140-149.
  For the references, a fit over N >= 30 predicts t(1) above its measured value (negative
  "first-frame excess", `su-table.txt`), so their later frames are not cheaper than their first.
- **Wall vs CPU:** 1.000-1.012, with the highest on the shortest encodes (x264 ultrafast 360p:
  58 ms in all).
- **Decision:** time the whole process (rusage user+sys); record wall beside it. The procedure
  states the timed unit, "150 frames including one key frame, process start to exit". A key-frame
  interval longer than the clip, as in real use, would lower the key frame's share. That is a
  property of the test, stated, not corrected.

## 5. Input I/O

**Design:**
- One c8a.large, one pinned core (vCPU 1). Every encoder of the start-up set, 150 frames of
  `handheld_mot17_10`, at 720p (1200 kbps) and 360p (424 kbps).
- Five runs in each state:
  - *warm:* the raw input in the page cache, read once before;
  - *cold:* the page cache dropped, the encoder binary re-warmed with a bare run, and `fincore`
    confirming 0 % of the input resident;
  - *tmpfs:* the input copied to `/dev/shm` on the box.
- A `cat` of the input in each state, and one `strace -c` run per encoder (read-family calls).
- Medians of five; `ccio-table.txt` in the scratch directory.

| Encoder (720p) | Warm CPU | Warm sys | tmpfs CPU | Cold CPU | Cold wall | read() calls | mmap calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| x264 ultrafast | 0.239 s | 0.003 | 0.242 | 0.267 | 1.542 | 2 | 170 |
| x264 medium | 2.364 | 0.012 | 2.367 | 2.409 | 3.644 | 2 | 172 |
| x265 medium | 5.703 | 0.033 | 5.641 | 5.729 | 5.763 | 232 | 79 |
| openh264 | 0.466 | 0.010 | 0.475 | 0.480 | 1.537 | 229 | 38 |
| SVT-AV1 M13 | 0.490 | 0.009 | 0.491 | 0.516 | 1.548 | 4 | 485 |
| libvpx rt 9 | 0.788 | 0.034 | 0.787 | 0.810 | 1.654 | 50,627 | 45 |
| libaom rt 10 | 1.934 | 0.024 | 1.973 | 1.972 | 2.863 | 50,629 | 44 |
| ours H.264 veryfast | 0.412 | 0.013 | 0.419 | 0.435 | 1.576 | 50 | 55 |
| ours HEVC faster | 0.531 | 0.013 | 0.530 | 0.547 | 1.590 | 50 | 58 |
| ours AV1 medium | 1.231 | 0.014 | 1.228 | 1.237 | 1.879 | 50 | 58 |
| ours VVC medium | 1.786 | 0.015 | 1.779 | 1.805 | 2.460 | 50 | 56 |

**Raw read cost:** `cat` of the 207 MB 720p input costs 4-5 ms CPU from the page cache or
tmpfs. A cold read takes 0.67-1.60 s of wall, the volume's read time, but almost no CPU.

**Findings:**
- **(1) Residency.** After a warm-up read the input is resident. Dropping the cache left 0 %
  resident (`fincore`), so the cold runs were truly cold.
- **(2) Cold vs warm.** CPU time rises by 1-12 %. x264 ultrafast is the worst: 0.239 to 0.267 s,
  page-cache fill counted as sys. The wall rises by up to 1.3 s, the disk wait, which CPU time
  does not count but a wall-clock timing would. Timing a cold input is therefore wrong on either
  clock.
- **(3) Read cost from RAM.**
  - Every encoder's sys time is 0.3-4.3 % of its CPU time at 720p. libvpx is highest: 34 ms of
    788, with stdio's 4 KB reads (about 50,600 read calls). x264 is lowest: it maps the input
    (2 reads, 170 mmaps), 3-12 ms.
  - sys also counts page faults on the encoders' own buffers, so it bounds the input cost from
    above. The copy itself (the `cat`) is 4-5 ms, which is 2 % of x264 ultrafast's 239 ms and
    under 1 % for everything else.
- **(4) The encoders read differently.** mmap (x264), a few large reads (SVT-AV1, ours: 50
  reads), medium reads (x265, openh264: about 230) and 4 KB stdio reads (libvpx, libaom:
  50,600). From RAM the difference is at most about 30 ms per 720p clip (libvpx), 4 % of its
  fastest encode.
- **tmpfs vs page cache:** no difference beyond noise. CPU within ±1-3 %, sys within a few ms.

**Decision (owner, 2026-11-23):**
- The input is decompressed onto a tmpfs on the timing box, verified 100 % resident (`fincore`)
  before each encode, and removed after.
- The read cost from RAM is part of each encoder's timed cost, as in real use. It is at most
  about 4 % for libvpx and about 2 % for the fastest x264, and the procedure states this.
- It is not subtracted: the encoders' own I/O choices are part of the program.

## 6. Concatenating clips into one encode

**Design:**
- One c8a.large, vCPU 1. Three 720p clips (handheld, pan, low light), 150 frames each, treated
  at 30 fps.
- 10 encoders: ours H.264/HEVC/AV1/VP9 medium, x264 medium, x265 medium, SVT-AV1 M12, libvpx rt
  9, libaom rt 10, openh264.
- Each clip encoded alone (150 frames, one key), against one concatenated 450-frame encode with a
  forced key every 150 frames.
- CBR 600 kbps (one target for all three clips, as concatenation forces) and Fixed QP (ours and
  x264/x265/openh264 at QP 32; SVT, libvpx and libaom at q 40). Three timed rounds each.

| Encoder | Mode | CPU: 3 clips alone | CPU: concatenated | Ratio | Segment bits vs the clip alone (clip 1 / 2 / 3) |
|---|---|---:|---:|---:|---|
| ours H.264 medium | CBR | 2.05 s | 2.05 | 0.999 | +0.1 / +2.2 / +2.1 % |
| ours H.264 medium | QP | 2.87 | 2.89 | 1.005 | 0.0 / +4.3 / +3.1 % |
| ours HEVC medium | CBR | 2.75 | 2.73 | 0.993 | 0.0 / -6.3 / +4.5 % |
| ours HEVC medium | QP | 5.35 | 5.30 | 0.989 | -0.2 / -0.5 / -0.5 % |
| ours AV1 medium | CBR | 2.64 | 2.63 | 0.995 | 0.0 / -5.5 / +2.1 % |
| ours VP9 medium | CBR | 3.04 | 3.00 | 0.985 | 0.0 / -4.9 / +0.5 % |
| x264 medium | CBR | 4.51 | 4.57 | 1.013 | +0.1 / +3.5 / +10.1 % |
| x264 medium | QP | 7.20 | 7.21 | 1.001 | 0.0 / 0.0 / -0.1 % |
| x265 medium | CBR | 10.93 | 10.69 | 0.978 | 0.0 / +9.1 / -14.1 % |
| x265 medium | QP | 18.85 | 18.80 | 0.998 | 0.0 / -0.1 / -0.4 % |
| SVT-AV1 M12 | CBR | 1.18 | 1.17 | 0.991 | 0.0 / +1.1 / -0.9 % |
| libvpx rt 9 | CBR | 1.78 | 1.75 | 0.987 | 0.0 / +0.6 / +0.3 % |
| libaom rt 10 | CBR | 4.64 | 4.60 | 0.991 | 0.0 / -3.2 / -2.3 % |
| openh264 | CBR | 0.96 | 0.96 | 0.994 | 0.0 / 0.0 / -0.2 % |

Full table, every encoder in both modes: `ccio-table.txt`.
- The per-segment PSNR comparison in that run is **not valid**. ffmpeg's psnr filter aligned the
  raw elementary streams with the reference by timestamp, and most pairs scored about 16 dB,
  which is misaligned. So quality per segment was not measured.
- The packet counts include one extra packet (a header) for most encoders. That shifts the
  segment bits by at most one packet.

**Findings:**
- **Speed:** concatenation saves 0-2 % of CPU time. The ratio is 0.978-1.013, within noise. That
  follows from section 4: start-up is about 1 ms, and the key frames are coded either way.
- **Independence:**
  - In CBR, the rate control carries buffer and model state across the boundary. The same clip
    gets -14 % to +10 % different bits inside the long stream (x265 -14.1 %, x264 +10.1 %, ours
    HEVC -6.3 %).
  - At fixed QP our H.264 still differs by up to 4.3 % (adaptive structure and QP cadence carry
    over). The references at fixed QP differ by 0-0.5 %.
  - The first clip is identical, as expected.
- **Per-clip ladders and frame rates:** each clip has its own rate targets (pixel-rate scaled)
  and its own frame rate (10-30 fps in our sets). A concatenated stream has one target and one
  frame rate, so it cannot code the ladder.
- **Cold start:** every clip alone starts cold, with a key frame and an empty buffer, which is
  the real-time start the product sees. Concatenated, only the first clip does.
- **Failure blast radius:** one failed or refused encode would lose every clip in the stream.
- **Statistics:** per-clip BD and the bootstrap over clips assume independent per-clip curves,
  which concatenation breaks.
- **Decision:** encode each clip alone. Concatenation was rejected: no speed gain, and it changes
  the streams being measured.
- **Alternatives considered:**
  - Subtracting a separately measured start-up is not valid: the intercept is content- and
    rate-dependent key-frame work (section 4).
  - Timing encode calls only (in-encoder timers) is possible for ours (`encode_us`) but not
    uniformly for the references, and it would drop real costs (init, I/O).
  - Whole-process CPU time is the one uniform measure.

## 7. Metric scoring: where

**libvmaf 3.2.1 CPU (the pinned `anchors/x86_64/bin/vmaf`)** was measured on a c7a.24xlarge
(Zen 4, eu-north-1 spot $0.6148/h). It ran VMAF v1 (`vmaf_v1.0.16_3d0h`) + `float_ms_ssim` +
`psnr` in one pass, `--threads 1`, on 150-frame I420 pairs:

| | 720p | 360p |
|---|---:|---:|
| One process alone | 20.2 CPU-s | 4.96 CPU-s |
| 94 processes at once (60 rounds) | 35.3 CPU-s | 8.53 CPU-s |
| Throughput, one c7a.24xlarge packed | 9,590 points/h | 39,700 points/h |
| Cost per point at $0.6148/h | $0.000064 | $0.000015 |

Scoring time is not a measurement, so packing is fine for it.

**The PC's GPU (VMAFx CUDA, docs/handoff/vmafx-gpu.md; not re-measured today):**
- 786 fps aggregate at 16 points at once, which is about 5.2 points/s (19,000 points/h) for 720p
  150-frame points.
- About 2.2 PC CPU-s per point (metrics.py comment) for VMAF + MS-SSIM, plus the decode and PSNR
  on the PC.
- 100,000 points would be about 61 CPU-hours on the PC, which the PC rule forbids. The GPU is
  also shared with another project.

**Decision:** score on cloud CPU boxes (spot, packed, any x86).
- Two c7a.24xlarge match the GPU's throughput for about $1.2/h.
- libvmaf 3.2.1 CPU is a public, pinned implementation a third party can run.
- The stored values agree with VMAFx at 6 decimals (docs/results/metric-validation-2026-11-23.md),
  so the existing cache stays comparable.

**Decoding for scoring** (the pinned ffmpeg 8.0.1 or the codec's standard decoder) was not timed
here. It is a small share: a 720p decode runs at hundreds of fps.

## 8. The "Zen 4 ISA bug"

**Reported:** the 0.6.0-era H.264 builds 20d31869 and c9cff89f code different streams on Zen 4
(27,433 B) than on Zen 3 / Zen 5 (27,444 B), for example `ho_caltech_set02_v003` at 37 kbps,
H.264 veryfast.

**Measured:**
- **Machines:** that command (the h23 plan's, 480x360, 150 frames) was coded on the PC (Zen 3,
  EPYC-class 5950X), c7a.24xlarge (Zen 4, EPYC 9R14), c8a.large x3, c8a.24xlarge and
  c8a.48xlarge (Zen 5, EPYC 9R45), and c8i.24xlarge (Xeon 6975P-C).
- **Tiers:** `NC_DSP_ISA` unset, `avx512`, `avx2` and `scalar`. avx512 is refused on Zen 3, as
  designed.
- **Load:** alone, and on fully loaded boxes (94 or 190 at once).
- **Builds:** the chart builds 4fc9a1de, 20d31869 and c9cff89f (gate, plus 20d31869/c9cff89f
  PGO); the release builds 20d31869 and 207745d7 (PGO and plain); and today's head.

| Build | Stream on every machine, tier and load |
|---|---|
| 4fc9a1de (gate) | 27,433 B, sha 8bb24164 |
| 207745d7 (release PGO and plain) | 27,433 B, sha 8bb24164 |
| 20d31869 (gate, PGO, release PGO, release plain) | 27,444 B, sha 374c17bc |
| c9cff89f (gate, PGO) | 27,444 B, sha 374c17bc |
| 0d9f32bc (head, release) | 27,517 B, sha 03778cdc |

**The cause:** the held-out chart's build record (`docs/results/bd-vs-speed-independent_2026-11-13-builds.json`)
names two builds for H.264 veryfast:
- `families` names 20d31869;
- `timed_on` names 4fc9a1de.

The h23 timing plan records 4fc9a1de's binary hash (`nanocodec_sha256` f451e51a =
`charts/builds/4fc9a1de/nanocodec`), while its command passes `--binary-for
nanocodec-h264@veryfast=.../20d31869/nanocodec`. One path ran 4fc9a1de and the other 20d31869.
Those commits code this clip differently on every CPU, and the CPU was a confound. Under the
single-run rule a family has exactly one build, and the record holds the binary's hash, so this
cannot recur.

**Today's head across tiers:** `tools/ci/isa_identity.sh` (new) codes every shipped preset of
every codec with the same binary at `NC_DSP_ISA=scalar` against `avx2`, and `avx2` against
`avx512` (`scalar` against `neon` on aarch64). It uses preset_identity's clips and modes (three
clips, CBR at two rates and fixed QP) and requires identical streams.
- x86 result at head 0d9f32bc, on the build box (AVX-512 capable): scalar = avx2 387/387 streams
  identical (239 s), and avx2 = avx512 387/387 identical (131 s).
- **NEON:** the script supports it (`--local --clip-dir` on the board or a Graviton box). It was
  not run today.
- A whole-encode check across tiers did not exist before: nc-checkasm checks kernels against
  their scalar references, and preset_identity compared two builds on one tier.

## 9. Wall time, regions and quota

**Quotas** (Service Quotas, standard families, vCPU, read 2026-11-23):

| Region | Spot quota | On-demand quota | In use now (ours; spot / on demand) |
|---|---:|---:|---|
| eu-west-1 | 512 | 512 | 26 / 244 |
| eu-north-1 | 1200 | 512 | 288 / 0 |
| eu-central-1 | 512 | 512 | 0 / 0 |
| eu-west-2 | 512 | 512 | 0 / 0 |
| us-east-1 | 512 | 512 | 0 / 0 |
| us-west-2 | 512 | 512 | 0 / 0 |
| eu-west-3, us-east-2 | 5 | 5 | - |

c8a is not offered in eu-north-1 or eu-west-2.

**Spot prices now, cheapest zone, $/h:**

| Type | eu-west-1 | eu-central-1 | us-east-1 | us-west-2 |
|---|---:|---:|---:|---:|
| c8a.large | 0.037 | 0.046 | 0.031 | 0.030 |
| c8a.24xlarge | 2.10 | 1.66 | 0.88 | 1.46 |
| c8a.48xlarge | 2.29 | 2.41 | 2.29 | 3.20 |
| c7a.24xlarge (eu-north-1: 0.61) | 1.51 | 1.53 | 0.93 | 1.38 |

c8a.large on demand: $0.1156/h in eu-west-1 (the session's own price lookup).

**Spot capacity, measured by launching:**
- No capacity in any zone: c8a.48xlarge in eu-west-1, eu-central-1 and us-east-1; c8a.12xlarge
  in eu-west-1.
- Launched: c8a.48xlarge in us-west-2d and c8a.24xlarge in eu-central-1a.
- `get-spot-placement-scores` (1-10; 384 vCPU target) gave:
  - single c8a types: 1-3 in every region;
  - c8a.large at 40 vCPU: 1 everywhere;
  - c8a 12/16/24/48xl/metal together: us-west-2 9, eu-central-1 8, eu-west-1 1, us-east-1 1;
  - c7a.24xlarge: 1-3.
- c8a.large on demand launched at once 13 times out of 13 (eu-west-1a).

**Launch plan** (timing on demand, so a reclaim cannot cost a measurement; scoring on spot):
- c8a.large on demand, about 120 boxes per region: 240 vCPU of the 512 quota, leaving room for
  other users. eu-west-1 (244 vCPU in use today, so about 100 boxes), eu-central-1, us-east-1 and
  us-west-2.
- About 400 boxes in four regions. The clips go up once per box (owner, 2026-11-20).
- At 9.5 core-minutes per 720p clip-mode (section 10), 400 boxes do about 2,500 clip-modes per
  hour.

**Per-box overhead, measured:** 1.6-2.1 min billed from launch to the first encode, plus the
download and termination. 15 sessions; the uploads were 100-860 MB over this PC's 64 MB/s
uplink.

## 10. Cost and wall time per clip and per dataset

**Encode and time (c8a.large on demand, $0.1156/h, one encode at a time):**
- *Per encode:* the 108-job round took 147.6 s on c8a.large, a mean of 1.37 s per 720p
  150-frame encode over this mix.
- *Families per mode:* the chart has 68 families: 35 of our single-layer rungs and 33
  reference families, including the metric-tuned twins (`bd-vs-speed-plan`).
- *Points per family and clip:* about 6 (CBR: the 4-rung ladder plus 2 upper rungs where the
  source allows; Fixed QP: 6 quantizers).
- *Per 720p clip in one mode:* 408 encodes x 1.37 s = 9.3 core-minutes, or $0.018. With the
  per-box overhead (about 2 min per box-hour) and the canary sets (24 short encodes per box) it
  is about $0.020.
- *Scaling with resolution:* time scales roughly with pixels (`su-table.txt`: our medium rungs
  at 360p take 0.33-0.35x their 720p time; x264 medium 0.31x). A 360p clip costs about a third
  and 1080p about 2.2x (not measured).

**Score (c7a.24xlarge spot, packed):** 408 points x 35.3 CPU-s = 4.0 core-hours, which is $0.026
per 720p clip-mode at $0.0064 per vCPU-hour. Decoding is not included (not measured; small).

**Per dataset x mode** (720p-equivalent clips, 150 frames; wall assumes 400 timing boxes and
enough spot scoring boxes):

| Clips | Encode + time | Score | Total $ | Wall (timing) |
|---:|---:|---:|---:|---|
| 50 | $1.0 | $1.3 | $2.3 | about 5 min + 2 min setup |
| 100 | $2.0 | $2.6 | $4.6 | about 5 min + 2 min |
| 200 | $4.0 | $5.2 | $9.2 | about 7 min + 2 min |
| 500 | $10 | $13 | $23 | about 15 min + 2 min |
| 1,000 | $20 | $26 | $46 | about 25 min + 2 min |

**How the wall is reached:**
- 400 boxes x 60 min / 9.5 min is about 2,500 clip-modes per hour.
- A box takes whole clips (all 408 encodes of a clip, 9.5 min). The wall is about max(9.5 min,
  clips x 9.5 / 400 min).
- To reach it, the 408 encodes of a clip are split over boxes by family: random assignment,
  which also spreads box bias.
- Frames scale linearly: 300-frame clips double every column.

**All six dataset-modes** (three datasets x two modes) at 200 clips each: about $55 and well
under an hour of timing wall, with scoring in parallel.

**Not measured:**
- 1080p and 4K costs;
- the AOM-CTC set's mix of resolutions;
- the Jetson, Graviton and Zen 3 timing series, which the owner may want as extra machine sets.

## 11. What changes in our tools

Designed in docs/benchmark-procedure.md, "Scripts". In short:
- one `tools/bench/` driver replaces the separate quality and timing paths;
- it times on c8a.large (one encode, vCPU 1, tmpfs input);
- it scores on spot CPU boxes;
- it enforces the checks and writes one record per point.

Not built yet: the owner and the coordinator decide first.

## Spend

$1.63 in 27 ledger rows, `Purpose=bench-method`:
- 13 c8a.large on demand;
- 1 c8a.24xlarge, 1 c8a.48xlarge, 2 c7a.24xlarge and 1 c8i.24xlarge spot;
- 2 sessions stopped at upload by a scratch-tool bug, at $0.06;
- the attempts refused for spot capacity cost nothing.

The isa_identity run used the shared build box (nc-remote slots). After the last session, no
instance tagged `Purpose=bench-method` was running in eu-west-1, eu-north-1, eu-central-1,
us-east-1 or us-west-2 (describe-instances).

## 12. History set: how well a subset tracks the full set

**Measured on the published data, not re-run:**
- Each history frame is coded on a 28-clip stratified slice of the fresh held-out set (5-6 per
  class, first by SHA-256 of the name; `web/results/data/history.json` `slice`).
- Its 0.6.0 frame (207745d77) against the full 137-clip fresh chart (`chart.json` view
  `fresh-psnr`, the same 0.6.0 presets) gives, over our 35 rungs:
  - the slice's BD-J is lower by 1.12 points on average, sd 0.35, max |difference| 1.79;
  - Spearman rank correlation 0.999;
  - per codec, every rung moves by a near-constant offset (H.264 -1.4 to -1.7, AV1 -0.5 to
    -1.4, VVC -0.7 to -1.2).
- So the slice tracks rank and change, with a level offset of about 1 point.

**Speed tracking** (subset vs full) cannot be measured on existing data: both used the same
three timing clips. It is measured by the first BP-1 run, which times every encode of the full
sets.

## 13. Full regeneration cost

The per-line table is in docs/benchmark-procedure.md, "Cost of regenerating everything". It is
computed by `tools/cost.py` in the scratch directory from the measured per-encode time, scoring
time, prices and box overheads. Lines that rest on an estimate are marked there.
