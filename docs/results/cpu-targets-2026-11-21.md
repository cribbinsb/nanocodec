# CPU targets for everyday A/Bs: Graviton, Zen 5 and the Jetson (2026-11-21)

Question (owner): could AWS Graviton be the everyday Arm measurement platform? The Jetson
Orin Nano (Cortex-A78AE, 256 KB L2 per core, 2 MB L3, LPDDR5) is one shared board where
jobs queue one at a time. Owner correction the same day: the targets are CPUs, not judges.
Where they differ on a change, that is a fact about the change, as AVX2 against AVX-512 is.
This file describes where the targets agree and where they differ, so the target set and
the decision rule can be chosen (docs/PROCEDURES.md "CPU A/B on rented cores").

## Method

- **Calibration set.** 62 rung A/Bs recorded on the Jetson on 2026-11-21 (jperf, min of 3
  rounds, cycles and instructions per clip), six changes:
  - `exact`: the exact analysis commits, 87fe15c9 to da273f66 (14 rungs).
  - zero SADs from pre-analysis: 332be8fd, `analysis.zero_sads_from_pre=1` (13 rungs).
  - even rows with a decimated half level: 332be8fd (12 rungs).
  - temporal partition prior: 9b42dceb (11 rungs).
  - seeded HME: 87fe15c9 (4 rungs).
  - sub-pel points from the integer cost surface: 995d48bd (8 rungs).

  Sources: docs/handoff/fast-analysis.md and the lit-ideas surface lever. These rungs
  include the memory-side items (the even rows and the zero SADs) and the case where
  instructions fall but Jetson cycles do not (the surface lever). Clips: the four quick
  clips each handoff used. CBR 600, 150 frames (90 for the surface lever).
- **Builds.** On Arm, the Jetson's own release binaries (`charts/builds/<commit>`, native
  builds, rustc 1.99.1, default target CPU), copied unchanged, so Arm instruction streams
  differ only by run-time dispatch. On x86, `cargo build --release` of the same commits on
  the build box (x86-64-v3).
- **Protocol.** `tools/dev/aws_cpu_ab.py`: .large boxes, spot, eu-west-1. Base and
  candidate run on vCPU 1 of the same box in 5 interleaved rounds (A B, B A, ...). Five
  boxes per target, all at once. The figure per rung is the mean over its clips of the
  per-round paired log ratio (the "paired" column below); "min" uses the min-of-rounds ratio,
  as jperf does.
- **Jetson reproducibility.** The same runner on the Jetson, for two of the rungs (zero SADs,
  HEVC Faster and AV1 Veryfast), 5 rounds: -0.89 % [-1.12, -0.66] and -0.96 % [-1.27, -0.66],
  against -0.88 % and -1.04 % recorded. Per-clip numbers moved by up to 0.26.

## Agreement with the Jetson, per target

Rung level, n = 62. Sign agreement counts the rungs where both moved 0.3 % or more.

| target | instance | rung r (paired / min) | sign agree | opposite sign, both >= 0.3 % | median gap | round sd | 95 % half-width per rung | cycles ~ instructions r |
|---|---|---|---|---|---|---|---|---|
| Jetson (A78AE) | board | - | - | - | - | 0.49-0.65 % | 0.23-0.31 | 0.59 |
| Graviton2 (N1) | c6g.large | 0.78 / 0.69 | 39/40 | 1 | 0.30 | 0.96 % | 0.44 | 0.67 |
| Graviton3 (V1) | c7g.large | 0.83 / 0.76 | 38/38 | 0 | 0.22 | 0.72 % | 0.33 | 0.79 |
| Graviton4 (V2) | c8g.large | 0.80 / 0.76 | 36/36 | 0 | 0.37 | 0.69 % | 0.27 | 0.85 |
| Zen 5 | c8a.large | 0.62 / 0.59 | 29/29 | - | 0.42 | 0.64 % | - | 0.89 |
| Sapphire Rapids | c7i.large (SMT pair) | 0.51 / 0.62 | 35/40 | - | 0.71 | 3.24 % | - | 0.68 |

Between the Graviton generations: G2 against G3 r 0.78, G3 against G4 0.85, G2 against G4
0.79. G3 against Zen 5 0.78, G4 against Zen 5 0.67. On the same Arm binaries, the
instruction deltas match the Jetson's at r 0.99.

## Where they differ (mean cycles change per lever, paired)

| lever (rungs) | Jetson instr | Jetson | G2 | G3 | G4 | Zen 5 | c7i |
|---|---|---|---|---|---|---|---|
| exact (14) | -0.51 | -0.56 | -0.68 | -0.64 | -0.73 | -0.83 | -0.71 |
| zero SADs (13) | -0.21 | -0.65 | -0.73 | -0.49 | -0.49 | -0.12 | -0.43 |
| even rows (12) | -0.91 | -0.44 | -0.46 | -0.56 | -0.81 | -0.24 | -1.22 |
| temporal prior (11) | -0.82 | -0.61 | -0.62 | -0.83 | -0.83 | -1.73 | -1.67 |
| seeded HME (4) | -1.30 | -0.92 | -1.47 | -1.15 | -1.50 | -0.95 | -1.47 |
| sub-pel surface (8) | -0.83 | +0.04 | -0.26 | -0.33 | -0.41 | -0.53 | -0.13 |

- **Instruction cuts.** The sub-pel surface, seeded HME and temporal prior: the bigger cores
  keep more of the cut. The surface lever costs nothing in cycles on the Jetson and saves
  0.3-0.4 % on Graviton3/4. Cycles follow instructions more closely, the bigger the core
  (last column above).
- **Memory side.** The zero SADs reuse pre-analysis SADs instead of reading both frames
  again: -0.65 % on the Jetson and -0.73 % on Graviton2 (whose L2 refills fall 2.7 %), about
  -0.5 % on Graviton3/4 and -0.12 % on Zen 5. Even rows: Graviton4 saves the most (L2D
  refills -5.0 %, Graviton2 -3.3 %, Graviton3 -0.7 %).
- **Largest single gaps (rung, Jetson / target).**
  - Graviton2: exact AV1 Faster -0.36 / -1.36; exact AV1 Fast +0.02 / -0.80. Its backend
    stalls fall 2-5 % there.
  - Graviton3: temporal prior AV1 Faster -0.48 / -1.16.
  - Graviton4: seeded HME AV2 Fast -1.34 / -2.27 (backend stalls -6.4 %).
  - The one opposite sign beyond 0.3 % on both sides: exact H.264 Veryfast, -0.61 on the
    Jetson against +0.35 on Graviton2.
- **x86.** The x86 builds' instruction deltas differ from Arm's (r 0.80), as expected for
  another ISA. The temporal prior saves 2-3x more on x86 than on any Arm core.

## Counters and noise on the boxes

- **PMU.** c6g, c7g and c8g .large count cycles, instructions and one more event at a time.
  With seven events at once, every count multiplexed and was dropped. The runner probes what
  fits and rotates the extra events over the rounds: L2D refill, LL miss, backend stall,
  L1D refill and branch mispredicts on Arm; cache misses and branch mispredicts, two at a
  time, on c8a and c7i.
- **Instructions in these VMs** are not exact: about 0.03 % run to run.
- **c7i.large** runs its two vCPUs as one core's SMT threads, so the timed thread shares a
  core with the system. Its round sd was 3.2 %: unsuited for everyday A/Bs.
- **c8a.large spot** was reclaimed on 3 of 5 boxes within minutes. The tool's on-demand
  retry recovered all of them.

## Cost and time

- Whole calibration: 32 box sessions including the pipe-cleaners, $0.24.
- Arm set: 62 rungs on 15 boxes, 20 min wall. The slowest box was Graviton2's, at 18.6 min
  for its fifth of the set.
- One rung A/B (4 clips, 5 rounds) is about 1.5 min of encodes after about 2 min of launch
  and upload, under a cent.
- The Jetson rerun of 2 rungs waited about 2 min in the board's lock queue behind other
  agents' jobs, then ran for 140 s.

## Charts

The site already carries a Graviton4 series beside the Jetson (c8g.large,
[graviton-speed-2026-11-17.md](graviton-speed-2026-11-17.md)). Readers can reproduce it on
a rented core. Nothing on the charts changes here.
