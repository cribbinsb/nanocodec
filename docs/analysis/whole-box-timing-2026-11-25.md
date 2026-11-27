# Whole-machine timing test (2026-11-25)

The evidence behind BP-4.0's timing method (docs/benchmark-procedure.md sections 2 and 6).
The question: can whole machines, each running one encoder family at a time on many cores,
replace BP-3's one encode per small spot box? That method ran about a hundred boxes, divided
each time by noisy reference encodes, and discarded 8 % (Intel) to 35 % (Graviton) of its
timings.

## What was run

- **Machines (eu-north-1 spot):**
  - c8g.metal-24xl: Graviton 4, 1 socket, 96 cores, 36 MB shared L3;
  - c8g.metal-48xl: Graviton 4, 2 sockets, 192 cores, 2 × 36 MB L3;
  - c8i.48xlarge: Intel Xeon 6975P-C, 1 socket, 96 cores × 2 threads, 480 MB L3;
  - c8g.24xlarge, for the 12- and 24-encode loads only.
- **Jobs:** 17 encoder families × 8 clips, each at its middle CBR rung, with the benchmark's
  own commands (BP-3 job files) and binaries:
  - ours: build 889309c99;
  - references: the bp3 anchors for each architecture.

  The clips came from the sealed "ours" set by mistake (sealed-log entry 18). This was timing
  only: no stream was scored.
- **Layout:** one encode per physical core, pinned with `taskset`. SMT siblings stayed idle,
  and core 0 was left to the harness.
- **Loads:**
  - **alone:** one encode on an otherwise idle machine;
  - **loaded, one family at a time:** each family's jobs on every Nth core, with untimed copies
    of the same family keeping every core busy until the last timed encode ended;
  - **N =** 95 (full), 47 (half), 24 and 12.
- **Order:** alone, full and half steps were interleaved over about 40 minutes, so drift shows.
- **Volume:** 100,368 encodes, none failed. Every job coded the same stream every time.
- **Records:** `~/nanocodec-data/bench-method-records/whole-box-2026-11-25/` (runner `tools/wb_remote.py`,
  analysis `tools/wb_analyse.py`).
- **Cost:** $3.27 (four sessions).

## Repeatability

Each figure is per family: the geometric mean of its 8 jobs' median CPU times, compared between
steps on one machine.

| | Graviton 96 core | Graviton 192 core | Intel 96 core |
|---|---|---|---|
| Alone, two rounds about 20 min apart: worst family | 0.18 % | 0.33 % | 0.15 % |
| Full load, four steps: worst family against the median | 0.30 % | 0.54 % | 1.27 % |
| Half load, two steps: worst family against the median | 0.06 % | 0.13 % | 0.28 % |
| One encode's spread (sd / median per job): alone / half / full | 0.19 / 0.33 / 0.41 % | 0.19 / 0.36 / 0.47 % | 0.19 / 0.72 / 1.45 % |

BP-3 on small boxes, for comparison:
- c8i.2xlarge: 1.4-1.5 % spread between boxes;
- c8g.2xlarge: 0.4-1.1 % spread between boxes;
- identical streams up to 1.5 % apart between runs.

## Between machines

Graviton 192 core against Graviton 96 core, the same jobs:

| Load | Mean ratio | Worst family against that mean | sd |
|---|---|---|---|
| alone | 1.0388 | 0.35 % | 0.18 % |
| half | 1.0401 | 0.99 % | 0.40 % |
| full | 1.0438 | 2.44 % | 0.84 % |

The c8g.24xlarge alone, against the c8g.metal-24xl, was 0.4-1.3 % slower, graded by family
(one round each; x264 ultrafast's 8 % is a single round of 0.19 s encodes and not trusted).

So:
- **Two sizes of one chip do not agree.** A series must use one instance type.
- **Agreement between two machines of the same type was not measured.** BP-4 checks it on
  every run (procedure V6) and in the trial (T24).

## Load effect: loaded time against alone, per family

| Encoder | Graviton alone (s) | 12 | 24 | 47 (half) | 95 (full) | Intel alone (s) | 47 (half) | 95 (full) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| kvazaar-ultrafast | 2.653 | 1.003 | 1.005 | 1.008 | 1.015 | 1.113 | 1.024 | 1.062 |
| nanocodec-av2@slower | 4.874 | 1.006 | 1.008 | 1.015 | 1.032 | 4.806 | 1.019 | 1.056 |
| x264-medium | 1.355 | 1.036 | 1.044 | 1.056 | 1.078 | 1.040 | 1.042 | 1.139 |
| nanocodec-vp9@slower | 2.850 | 1.007 | 1.016 | 1.055 | 1.080 | 2.435 | 1.019 | 1.103 |
| openh264-baseline | 0.283 | 1.019 | 1.045 | 1.071 | 1.131 | 0.191 | 1.000 | 1.052 |
| x265-medium | 2.962 | 1.026 | 1.043 | 1.102 | 1.142 | 2.352 | 1.058 | 1.136 |
| svtav1-m8 | 1.415 | 1.035 | 1.055 | 1.116 | 1.164 | 1.169 | 1.027 | 1.119 |
| x264-ultrafast | 0.190 | 1.025 | 1.045 | 1.076 | 1.172 | 0.146 | 1.021 | 1.076 |
| x265-ultrafast | 1.505 | 1.027 | 1.061 | 1.132 | 1.176 | 1.109 | 1.067 | 1.153 |
| libaom-av1-cpu8 | 1.025 | 1.054 | 1.072 | 1.112 | 1.180 | 0.925 | 1.022 | 1.132 |
| nanocodec-vvc@medium | 1.099 | 1.032 | 1.060 | 1.128 | 1.187 | 0.985 | 1.038 | 1.136 |
| nanocodec-av1@medium | 0.620 | 1.033 | 1.056 | 1.111 | 1.189 | 0.593 | 1.031 | 1.099 |
| nanocodec-hevc@medium | 0.603 | 1.042 | 1.077 | 1.147 | 1.220 | 0.607 | 1.032 | 1.112 |
| libvpx-vp9-cpu9 | 0.440 | 1.060 | 1.097 | 1.151 | 1.262 | 0.389 | 1.007 | 1.080 |
| svtav1-m13 | 0.314 | 1.061 | 1.121 | 1.223 | 1.396 | 0.277 | 1.012 | 1.080 |
| nanocodec-av1@veryfast | 0.311 | 1.067 | 1.122 | 1.233 | 1.469 | 0.280 | 1.048 | 1.161 |
| nanocodec-h264@veryfast | 0.209 | 1.064 | 1.125 | 1.229 | 1.502 | 0.222 | 1.028 | 1.124 |

Where each Graviton column was measured:
- the 12 and 24 columns on the c8g.24xlarge, each against its own alone rounds;
- the 47 and 95 columns on the c8g.metal-24xl.

What the table shows:

- **Where the time goes:** the slowdown is in user time, not the kernel. Our h264 veryfast on
  Graviton, full load: user 0.236 s → 0.382 s, sys 0.009 s → 0.031 s.
- **Graviton:** the slowdown grows about linearly with the number of encodes running at once.
  For the fastest presets that is about 0.5 % per extra encode. No load above one encode
  leaves the encoders' order unchanged. Our fast presets and SVT-AV1 M13 are the most
  sensitive.
- **Intel:** half load keeps every family within ×1.00-1.07 of alone.
- **The cause is not measured.** No cache counters were recorded. The pattern (worst on the
  chip with 36 MB of L3 for 96 cores) points at shared-cache or memory pressure. The Jetson
  brief `work/briefs/cache-sensitivity-2026-11-25.md` is to find out.

## Machine availability (spot, 2026-11-25 morning)

| Type | Tries | Launched |
|---|---|---|
| c8i.metal-48xl | eu-north-1 ×2, us-west-2 ×2 | 0 |
| c8i.metal-96xl | eu-north-1 ×2 | 0 |
| c8i.48xlarge | eu-north-1 ×1 | 1 (eu-north-1b, $2.95/h) |
| c8g.metal-24xl | eu-north-1 ×3 | 1 ($0.41/h) |
| c8g.metal-48xl | eu-north-1 ×2 | 1 ($0.82/h) |
| c8g.24xlarge | eu-north-1 ×1 | 1 (eu-north-1c, $0.96/h) |

## Owner decisions (2026-11-25)

- **Loaded speed is the measure.** Each encoder competes only with its own work; nobody runs
  an encoder on an idle server.
- **Half load is the reasonable compromise.** Our encoders' cache sensitivity is to be
  optimised rather than hidden.
- **Machines of one type are assumed to agree**, as every comparison with past results
  already assumes; each run checks it.
- **Scoring is done locally (GPU) where possible.**
