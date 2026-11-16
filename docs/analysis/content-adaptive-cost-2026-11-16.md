# Where our CPU goes against SVT-AV1 and x264: Zen 5 scaling, the Fast rungs at fixed QP, content adaptivity (2026-11-16)

Diagnosis only: no encoder code was changed. The brief changed four times during the work
(content adaptivity against M10; block-level counts; the Fixed QP chart's frontier; the Zen 5
scalar/vector question). All four are answered here, in the order of the last brief: Zen 5 first,
the Fast rungs at fixed QP second, content adaptivity third. Every number is measured or read
unless it is marked **[inferred]**, which also names the test that would confirm it.

## 0. Findings

1. **Zen 5: our kernels scale like SVT's; our problem is how much scalar work we do, not how wide
   our vectors are.** From Zen 3 to Zen 5, vector kernels get 1.7-2.5x faster in both encoders
   and scalar code 1.4-1.8x (§1.1). SVT M10 spends 58-62 % of its Zen 5 cycles in kernels, we
   spend 30-40 % (§1.3). We retire as many vector uops as M10 on the same pictures and, on top,
   1.7-1.8x its non-vector instructions (§1.2). On the lowlight clip the 2.0x time gap to M10
   is a 2.0x instruction gap at the same IPC (12.84 G against 6.45 G instructions, IPC 3.29
   and 3.24, this morning's counters).
2. **What is really narrow or scalar where it should not be** is a short list worth about
   8-12 % of Zen 5 time in all **[inferred from measured shares]** (§1.5): the AV1 deblocking
   edge kernel (3.6-6.8 % of cycles, 88-93 % of its instructions non-vector, its vector uops all
   128-bit), small-block variance, pixel and statistics loops that live outside nc-dsp
   (`intra_estimate::estimate`, `preanalysis::analyze`, `noise::update`, `quantized_eob`), then
   16-wide sub-pel variance, CDEF blocks and the AVX2-only AV1 MC.
3. **The larger Zen 5 prize is scalar work SVT does not do at all on P frames**: coefficient
   coding is 12-28 % of our Zen 5 cycles against 5-12 % of M10's and 2-11 % of M12's (trellis,
   live-CDF rate walks for the transform-split trial, and a coefficient writer and range coder
   that take twice M10's instructions for the same bytes).
4. **Why the Fast rungs collapse at fixed QP: one mechanism, not the profile.** AV1 Fast drops
   the high frequencies of every 32x32 and 64x64 transform at every quantizer
   (`av1.partial_forward 2`), and its two other settings close the ways round it: depth removal
   at every quantizer keeps blocks at 32x32 and larger, and the transform split is off. Putting
   back any one of the three recovers 2.7-3.1 dB PSNR-Y at qindex 110 (all of the Default
   together: 3.8 dB), 0.3-0.4 dB at qindex 185 and 0.02-0.22 dB at qindex 210-235; the CBR
   chart's 600 kbit/s rung is at qindex 188-216 on its clips (§2.2).
   The fixed-QP profile being inert on Fast accounts for 2.8 of the 27.7 points between AV1 Fast
   and Default, and 0.2 of 21.6 for HEVC (§2.1, §2.3).
5. **No re-mix of existing levers beats M10 or M12 outright on Zen 5** (§2.4, timed on c8a).
   The best new points are on the frontier between them: AV1 Fast with the Default's depth
   settings and full forward transform is 250 fps at Weighted -6.5 (M12: 356 fps, -1.5; M10:
   216 fps, -14.2), and with deblocking 232 fps at -9.2. To beat M10 we need the -16 point
   (Default without trellis and the key intra set, 172 fps) about 26 % faster; to beat M12 we
   need the -3.8 point (Fast with depth, 253 fps) about 41 % faster.
6. **Content adaptivity** (§3): AV1 Default costs 84.6 M instructions per P frame on the dark
   static clip and 84.4 M on the busy one; SVT M10 51.3 and 79.1; our own AV1 Fast 42.0 and
   54.4. Default codes as many blocks on the easy clip as on the hard one (1009 against 1093
   leaves per frame; M10 549 against 846) and does *more* transform-split trials, trellis and
   coefficient-rate walks there; it is coded at a finer quantizer for the same bytes (qindex 188
   against 214), which is the likely reason **[inferred]**. The many-clip
   confirmation the second brief asked for was stopped by the third brief and is **not done**.

## Method and sources

- **Builds.** Ours: `variance-integration` head `461896365`, `--profile profiling` (release code
  generation with debug info) for callgrind and for the c8a timing (stripped copy); the lens
  build `bad14976` (`~/nanocodec-data/work/fixedqp2/bin/cut`) for the ablation, which reproduces
  the published lens (AV1 Fast +1.6, Default -17.9; HEVC Fast +7.9, Default -14.9) and gives the
  same figures as head, to the last digit, for every variant run on both. The morning probe's build `d43e1cbf4` for
  the Zen 5 counters, so they line up with `docs/handoff/avx512.md`. References: the anchors'
  SVT-AV1 v4.3.0 and x264 0.165, and an uninlined SVT build of the same source
  (`-fno-inline`, byte-identical streams) for call counts.
- **Clips.** The two private chart clips (lowlight_city_night, crowd_mot17_09) were profiled on
  the PC only. Everything run on AWS used public AOM CTC sequences.
- **Instruction counts** are callgrind on the PC (AVX2; valid under load). **Zen 5 numbers** are
  from one c8a.large core: the morning's records in `~/nanocodec-data/work/zen5/probe-zen{3,5}`
  and two new sessions (instruction mix, candidate timing; $0.03 in all, in the ledger).
- **Fixed-QP figures** are the lens's Weighted BD-rate (0.92 Y + 0.04 U + 0.04 V of the PSNR of
  the mean MSE) on its 12 sequences, 130 frames, six quantizers, against the stored x264 medium
  curves, computed from the encoder's `--sse-log` (no pictures written). AV1 rates exclude the
  IVF container, as the lens's do. J was not computed (it needs VMAF and MS-SSIM scoring).
- Scripts and raw results: `~/nanocodec-data/work/adaptive-diag/` (`classes.py`, `mix/`,
  `ablate.py`, `abl/`, `cgstat.py`, `stages_*.py`, `counts_*.py`, `cg/`).
- The SVT citations in §3.4 come from a helper's code trace handed over by the coordinator; the
  ones this document relies on were re-read in the source and are marked "checked".

## 1. Zen 5: scalar against vector

### 1.1 Speed-up from Zen 3 to Zen 5 by kind of work

Lowlight clip, 150 frames, the morning's perf records on c6a (Zen 3) and c8a (Zen 5): each
symbol's share times the run's median CPU time, summed by class. Symbols under 0.2 % are not
listed by perf and are the "below threshold" row. With about 7000 samples a run, single symbols
are noisy; classes are not.

ours-av1: Zen 3 1491 ms, Zen 5 894 ms, x1.67

| class | Zen 3 ms | Zen 5 ms | Zen 5 share | speedup |
|---|---:|---:|---:|---:|
| coefficient coding, scalar (trellis, CDF cost/write, range coder, EOB) | 338 | 246 | 28% | 1.38 |
| backend block coding, scalar (code_block/plane, mvref, mode bits, loop-filter control) | 196 | 126 | 14% | 1.56 |
| below the 0.2% listing threshold | 177 | 115 | 13% | 1.55 |
| kernels: variance/sub-pel variance/SAD/SATD | 223 | 113 | 13% | 1.97 |
| mode decision and planning, scalar (nc_encoder) | 167 | 103 | 12% | 1.63 |
| kernels: transform/quant | 160 | 68 | 8% | 2.36 |
| kernels: loop filters (deblock, CDEF) | 92 | 54 | 6% | 1.69 |
| kernels: MC and intra prediction | 51 | 25 | 3% | 2.01 |
| libc, kernel, other | 32 | 21 | 2% | 1.53 |
| source analysis, scalar (nc_analysis) | 31 | 19 | 2% | 1.63 |
| kernels: other nc_dsp | 24 | 4 | 0% | 5.60 |

ours-h264: Zen 3 952 ms, Zen 5 619 ms, x1.54

| class | Zen 3 ms | Zen 5 ms | Zen 5 share | speedup |
|---|---:|---:|---:|---:|
| kernels: variance/sub-pel variance/SAD/SATD | 244 | 136 | 22% | 1.80 |
| mode decision and planning, scalar (nc_encoder) | 193 | 126 | 20% | 1.53 |
| coefficient coding, scalar (trellis, CDF cost/write, range coder, EOB) | 139 | 95 | 15% | 1.47 |
| backend block coding, scalar (code_block/plane, mvref, mode bits, loop-filter control) | 139 | 86 | 14% | 1.62 |
| below the 0.2% listing threshold | 78 | 51 | 8% | 1.52 |
| kernels: transform/quant | 65 | 51 | 8% | 1.26 |
| source analysis, scalar (nc_analysis) | 46 | 33 | 5% | 1.41 |
| libc, kernel, other | 15 | 19 | 3% | 0.79 |
| kernels: other nc_dsp | 12 | 8 | 1% | 1.44 |
| kernels: MC and intra prediction | 10 | 8 | 1% | 1.24 |
| kernels: loop filters (deblock, CDEF) | 10 | 7 | 1% | 1.47 |

svt-m10: Zen 3 888 ms, Zen 5 445 ms, x2.00

| class | Zen 3 ms | Zen 5 ms | Zen 5 share | speedup |
|---|---:|---:|---:|---:|
| mode decision, ME control, syntax, scalar C | 146 | 83 | 19% | 1.76 |
| kernels: variance/sub-pel variance/SAD/SATD | 155 | 78 | 17% | 2.00 |
| kernels: transform/quant | 184 | 74 | 17% | 2.49 |
| below the 0.2% listing threshold | 159 | 66 | 15% | 2.41 |
| coefficient coding, scalar (optimize_b, cost_coeffs, write_coeffs, entropy coder) | 74 | 52 | 12% | 1.42 |
| kernels: loop filters (deblock, CDEF) | 82 | 46 | 10% | 1.76 |
| kernels: MC and intra prediction | 51 | 29 | 7% | 1.74 |
| kernels: other vector | 19 | 10 | 2% | 1.88 |
| libc, kernel, other | 18 | 6 | 1% | 2.86 |

svt-m13: Zen 3 453 ms, Zen 5 236 ms, x1.92

| class | Zen 3 ms | Zen 5 ms | Zen 5 share | speedup |
|---|---:|---:|---:|---:|
| mode decision, ME control, syntax, scalar C | 112 | 64 | 27% | 1.75 |
| kernels: variance/sub-pel variance/SAD/SATD | 69 | 32 | 14% | 2.14 |
| below the 0.2% listing threshold | 61 | 31 | 13% | 1.97 |
| kernels: transform/quant | 71 | 26 | 11% | 2.77 |
| coefficient coding, scalar (optimize_b, cost_coeffs, write_coeffs, entropy coder) | 34 | 23 | 10% | 1.47 |
| kernels: MC and intra prediction | 38 | 19 | 8% | 2.00 |
| libc, kernel, other | 23 | 19 | 8% | 1.24 |
| kernels: loop filters (deblock, CDEF) | 27 | 16 | 7% | 1.69 |
| kernels: other vector | 18 | 7 | 3% | 2.74 |

x264-medium: Zen 3 1606 ms, Zen 5 988 ms, x1.62

| class | Zen 3 ms | Zen 5 ms | Zen 5 share | speedup |
|---|---:|---:|---:|---:|
| kernels: SAD/SATD/variance | 476 | 320 | 32% | 1.49 |
| mode decision, ME control, syntax, scalar C | 386 | 245 | 25% | 1.58 |
| kernels: MC and intra prediction | 399 | 200 | 20% | 2.00 |
| coefficient coding (CABAC, trellis) | 105 | 82 | 8% | 1.28 |
| below the 0.2% listing threshold | 130 | 69 | 7% | 1.89 |
| kernels: transform/quant | 75 | 50 | 5% | 1.51 |
| kernels: loop filter | 34 | 18 | 2% | 1.86 |
| libc, kernel, other | 0 | 5 | 1% | 0.00 |

Read together: in our AV1 and in SVT the transform and quantisation kernels gain 2.4-2.8x and
the variance and SAD family 2.0-2.1x, while scalar coefficient coding gains 1.3-1.5x in all
five encoders. (H.264-sized kernels gain less: ours 1.3-1.8x, x264's 1.5-2.0x.) Our AV1 classes
scale like SVT's; our mix does not. On Zen 5 the scalar classes are 494 ms of our 894 ms (55 %), and 135 ms
of M10's 445 ms (30 %).

### 1.2 Exact counters on Zen 5 (AOM CTC sequences)

c8a.large, one pinned core, AVX-512 tiers unless marked, 130 frames at the lens's fourth
quantizer (ours qindex 185 or QP 37, SVT 46, x264 37; the bytes are close but not matched).
`fp_ops_retired_by_width` counts packed integer and floating-point vector uops by register
width; scalar-FP is scalar float arithmetic.

| run (130 frames) | CPU ms | instructions G | IPC | 128-bit uops G | 256-bit G | 512-bit G | scalar-FP G | non-vector instr G | vector uops / instr | bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ours-av1-Johnny | 456 | 5.83 | 3.28 | 0.86 | 0.35 | 0.75 | 0.06 | 3.81 | 34% | 84340 |
| ours-av1-WestWindEasy | 1636 | 22.63 | 3.23 | 2.74 | 1.21 | 1.73 | 0.30 | 16.65 | 25% | 1009246 |
| ours-av1-fast-Johnny | 219 | 3.01 | 3.20 | 0.31 | 0.24 | 0.38 | 0.03 | 2.05 | 31% | 86601 |
| ours-av1-fast-WestWindEasy | 709 | 10.20 | 3.36 | 1.03 | 0.70 | 1.31 | 0.07 | 7.09 | 30% | 984206 |
| ours-av1@avx2tier-Johnny | 449 | 6.69 | 3.48 | 0.94 | 1.55 | 0.04 | 0.07 | 4.09 | 38% | 84340 |
| ours-av1@avx2tier-WestWindEasy | 1741 | 24.32 | 3.26 | 2.74 | 4.06 | 0.08 | 0.31 | 17.13 | 28% | 1009246 |
| ours-h264-Johnny | 345 | 5.11 | 3.46 | 0.71 | 0.24 | 0.47 | 0.05 | 3.64 | 28% | 78573 |
| ours-h264-WestWindEasy | 987 | 13.69 | 3.24 | 2.58 | 0.99 | 0.91 | 0.15 | 9.05 | 33% | 928826 |
| ours-hevc-Johnny | 590 | 9.05 | 3.58 | 0.76 | 0.65 | 1.33 | 0.08 | 6.23 | 30% | 136016 |
| ours-hevc-WestWindEasy | 1825 | 25.42 | 3.26 | 2.52 | 1.72 | 1.89 | 0.36 | 18.92 | 24% | 1604029 |
| svt-m10-Johnny | 306 | 4.32 | 3.30 | 0.32 | 0.82 | 0.97 | 0.00 | 2.21 | 49% | 72052 |
| svt-m10-WestWindEasy | 1116 | 15.41 | 3.23 | 1.57 | 3.78 | 0.72 | 0.02 | 9.32 | 39% | 1030916 |
| svt-m12-Johnny | 190 | 2.63 | 3.23 | 0.15 | 0.41 | 0.64 | 0.00 | 1.42 | 46% | 64787 |
| svt-m12-WestWindEasy | 667 | 9.10 | 3.19 | 0.55 | 2.12 | 0.51 | 0.00 | 5.92 | 35% | 1012757 |
| svt-m13-Johnny | 167 | 2.15 | 3.01 | 0.14 | 0.37 | 0.35 | 0.00 | 1.28 | 40% | 64427 |
| svt-m13-WestWindEasy | 639 | 8.51 | 3.11 | 0.51 | 1.93 | 0.45 | 0.00 | 5.61 | 34% | 994391 |
| x264-medium-Johnny | 588 | 7.55 | 3.00 | 1.05 | 1.07 | 0.89 | 0.01 | 4.52 | 40% | 88535 |
| x264-medium-WestWindEasy | 3553 | 38.24 | 2.52 | 7.06 | 4.12 | 3.00 | 0.09 | 23.98 | 37% | 1516202 |

- Our AV1 Default retires about as many vector uops as M10 on both sequences (1.96 G against
  2.11 G on Johnny, 5.68 G against 6.07 G on WestWind), but **2.7x and 1.7x its 128-bit uops**
  (0.86 against 0.32 G; 2.74 against 1.57 G), and **1.7x and 1.8x its non-vector instructions**
  (3.81 against 2.21 G; 16.6 against 9.3 G).
- Our scalar float work is visible (0.06-0.36 G uops: f32/f64 cost arithmetic); SVT's is zero
  (integer costs).
- Forcing our AVX2 tier costs 0 % on Johnny and 6 % on WestWind: the AVX-512 tier is not where
  the gap is.

### 1.3 Where each encoder's Zen 5 cycles and instructions are, by class

Per-symbol samples of cycles, instructions and the three width counters, scaled to the exact
totals above (perf throttled the sampling on the longer runs, so shares are sampled and totals
are exact). "Non-vector" is instructions minus vector and scalar-FP uops.

Ours, AV1 Default, WestWindEasy:

totals: cycles 7.01G instructions 22.63G vector uops 128/256/512 2.74/1.21/1.73G scalar-fp 0.30G

| class | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coefficient coding, scalar (trellis, CDF cost/write, range coder, EOB) | 26.0 | 5443 | 2.98 | 398 | 58 | 158 | 121 | 4708 | 11% | 65% |
| backend block coding, scalar (code_block/plane, mvref, mode bits, loop-filter control) | 18.4 | 3978 | 3.08 | 517 | 200 | 147 | 42 | 3073 | 22% | 60% |
| mode decision and planning, scalar (nc_encoder) | 17.0 | 3892 | 3.26 | 520 | 159 | 358 | 77 | 2778 | 27% | 50% |
| kernels: variance/sub-pel variance/SAD/SATD | 11.7 | 3234 | 3.93 | 524 | 217 | 554 | 28 | 1911 | 40% | 40% |
| kernels: transform/quant | 6.9 | 1562 | 3.25 | 240 | 295 | 317 | 6 | 703 | 55% | 28% |
| kernels: loop filters (deblock, CDEF) | 6.7 | 1695 | 3.60 | 164 | 88 | 61 | 2 | 1380 | 18% | 52% |
| libc, kernel, other | 5.0 | 792 | 2.27 | 89 | 25 | 60 | 10 | 608 | 22% | 51% |
| kernels: MC and intra prediction | 3.0 | 783 | 3.68 | 147 | 99 | 20 | 9 | 508 | 34% | 55% |
| source analysis, scalar (nc_analysis) | 1.8 | 514 | 4.19 | 46 | 10 | 16 | 1 | 440 | 14% | 63% |
| kernels: other nc_dsp | 1.4 | 333 | 3.32 | 56 | 41 | 20 | 5 | 211 | 35% | 48% |

Ours, AV1 Default, Johnny:

totals: cycles 1.78G instructions 5.83G vector uops 128/256/512 0.86/0.35/0.75G scalar-fp 0.06G

| class | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kernels: variance/sub-pel variance/SAD/SATD | 20.8 | 1311 | 3.55 | 200 | 91 | 379 | 8 | 634 | 51% | 30% |
| backend block coding, scalar (code_block/plane, mvref, mode bits, loop-filter control) | 16.0 | 908 | 3.19 | 144 | 39 | 38 | 8 | 678 | 24% | 65% |
| mode decision and planning, scalar (nc_encoder) | 15.5 | 884 | 3.21 | 146 | 32 | 109 | 27 | 569 | 33% | 51% |
| coefficient coding, scalar (trellis, CDF cost/write, range coder, EOB) | 12.2 | 668 | 3.08 | 58 | 7 | 19 | 12 | 572 | 13% | 69% |
| libc, kernel, other | 8.1 | 310 | 2.16 | 23 | 7 | 36 | 2 | 241 | 22% | 35% |
| kernels: loop filters (deblock, CDEF) | 7.8 | 575 | 4.15 | 72 | 14 | 22 | 1 | 467 | 19% | 67% |
| kernels: MC and intra prediction | 5.5 | 358 | 3.66 | 99 | 97 | 9 | 3 | 150 | 57% | 48% |
| source analysis, scalar (nc_analysis) | 5.2 | 326 | 3.49 | 52 | 7 | 16 | 0 | 250 | 23% | 69% |
| kernels: transform/quant | 3.9 | 234 | 3.35 | 31 | 34 | 101 | 0 | 66 | 71% | 19% |
| kernels: other nc_dsp | 2.5 | 122 | 2.72 | 20 | 15 | 8 | 1 | 78 | 35% | 47% |

SVT M10, WestWindEasy and Johnny:

totals: cycles 4.77G instructions 15.41G vector uops 128/256/512 1.57/3.78/0.72G scalar-fp 0.02G

| class | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mode decision, ME control, syntax, scalar C | 27.3 | 3892 | 2.99 | 329 | 638 | 124 | 5 | 2797 | 28% | 30% |
| kernels: variance/sub-pel variance/SAD/SATD | 24.8 | 4040 | 3.42 | 430 | 1077 | 273 | 10 | 2251 | 44% | 24% |
| kernels: loop filters (deblock, CDEF) | 12.2 | 2190 | 3.75 | 387 | 365 | 2 | 0 | 1435 | 34% | 51% |
| kernels: transform/quant | 10.7 | 1664 | 3.26 | 99 | 920 | 222 | 0 | 424 | 75% | 8% |
| coefficient coding, scalar (optimize_b, cost_coeffs, write_coeffs, entropy coder) | 8.9 | 1333 | 3.13 | 66 | 235 | 10 | 0 | 1022 | 23% | 21% |
| kernels: MC and intra prediction | 8.4 | 1469 | 3.65 | 179 | 267 | 48 | 1 | 973 | 34% | 36% |
| libc, kernel, other | 2.7 | 143 | 1.13 | 6 | 3 | 14 | 0 | 120 | 16% | 26% |
| kernels: other vector | 2.2 | 311 | 2.97 | 55 | 223 | 15 | 0 | 18 | 94% | 19% |

totals: cycles 1.31G instructions 4.32G vector uops 128/256/512 0.32/0.82/0.97G scalar-fp 0.00G

| class | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kernels: transform/quant | 23.1 | 1057 | 3.50 | 31 | 347 | 465 | 0 | 214 | 80% | 4% |
| mode decision, ME control, syntax, scalar C | 22.4 | 983 | 3.35 | 32 | 96 | 115 | 1 | 739 | 25% | 13% |
| kernels: variance/sub-pel variance/SAD/SATD | 20.0 | 901 | 3.44 | 64 | 149 | 273 | 1 | 414 | 54% | 13% |
| kernels: loop filters (deblock, CDEF) | 8.6 | 465 | 4.14 | 53 | 85 | 1 | 0 | 326 | 30% | 38% |
| kernels: MC and intra prediction | 7.2 | 336 | 3.56 | 97 | 23 | 56 | 0 | 160 | 52% | 55% |
| libc, kernel, other | 6.5 | 99 | 1.15 | 1 | 1 | 33 | 0 | 63 | 36% | 4% |
| coefficient coding, scalar (optimize_b, cost_coeffs, write_coeffs, entropy coder) | 5.4 | 225 | 3.15 | 3 | 18 | 7 | 0 | 197 | 12% | 10% |
| kernels: other vector | 3.1 | 159 | 3.91 | 34 | 92 | 9 | 0 | 24 | 85% | 25% |

SVT M12, WestWindEasy and Johnny:

totals: cycles 2.85G instructions 9.10G vector uops 128/256/512 0.55/2.12/0.51G scalar-fp 0.00G

| class | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| mode decision, ME control, syntax, scalar C | 31.7 | 2783 | 3.08 | 136 | 375 | 94 | 1 | 2177 | 22% | 22% |
| kernels: variance/sub-pel variance/SAD/SATD | 17.5 | 1933 | 3.86 | 144 | 529 | 226 | 1 | 1033 | 46% | 16% |
| kernels: transform/quant | 11.2 | 1015 | 3.18 | 62 | 495 | 110 | 0 | 349 | 66% | 9% |
| kernels: loop filters (deblock, CDEF) | 10.9 | 1120 | 3.59 | 90 | 221 | 2 | 0 | 807 | 28% | 29% |
| coefficient coding, scalar (optimize_b, cost_coeffs, write_coeffs, entropy coder) | 10.6 | 782 | 2.59 | 23 | 109 | 6 | 0 | 645 | 18% | 17% |
| kernels: MC and intra prediction | 9.5 | 901 | 3.32 | 51 | 201 | 44 | 0 | 605 | 33% | 17% |
| libc, kernel, other | 3.6 | 142 | 1.37 | 4 | 1 | 16 | 0 | 121 | 15% | 19% |
| kernels: other vector | 2.0 | 238 | 4.09 | 32 | 168 | 8 | 0 | 30 | 87% | 15% |

totals: cycles 0.81G instructions 2.63G vector uops 128/256/512 0.15/0.41/0.64G scalar-fp 0.00G

| class | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| kernels: variance/sub-pel variance/SAD/SATD | 31.5 | 978 | 3.83 | 49 | 116 | 377 | 1 | 435 | 55% | 9% |
| mode decision, ME control, syntax, scalar C | 27.6 | 681 | 3.04 | 26 | 61 | 81 | 1 | 512 | 25% | 16% |
| libc, kernel, other | 12.5 | 130 | 1.28 | 2 | 1 | 26 | 0 | 101 | 22% | 8% |
| kernels: transform/quant | 9.1 | 265 | 3.60 | 10 | 79 | 105 | 0 | 71 | 73% | 5% |
| kernels: MC and intra prediction | 7.8 | 203 | 3.22 | 7 | 24 | 43 | 0 | 129 | 36% | 10% |
| kernels: loop filters (deblock, CDEF) | 6.1 | 188 | 3.80 | 19 | 38 | 0 | 0 | 130 | 31% | 34% |
| kernels: other vector | 3.8 | 123 | 4.00 | 34 | 87 | 3 | 0 | 0 | 101% | 28% |
| coefficient coding, scalar (optimize_b, cost_coeffs, write_coeffs, entropy coder) | 2.2 | 53 | 3.04 | 1 | 2 | 0 | 0 | 51 | 4% | 22% |

Ours, AV1 Fast, WestWindEasy:

totals: cycles 3.03G instructions 10.20G vector uops 128/256/512 1.03/0.70/1.31G scalar-fp 0.07G

| class | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coefficient coding, scalar (trellis, CDF cost/write, range coder, EOB) | 21.4 | 2003 | 3.09 | 106 | 9 | 32 | 0 | 1855 | 7% | 72% |
| backend block coding, scalar (code_block/plane, mvref, mode bits, loop-filter control) | 17.0 | 1825 | 3.54 | 188 | 88 | 92 | 14 | 1443 | 20% | 51% |
| kernels: variance/sub-pel variance/SAD/SATD | 13.9 | 1487 | 3.54 | 208 | 117 | 331 | 15 | 815 | 44% | 32% |
| mode decision and planning, scalar (nc_encoder) | 13.0 | 1358 | 3.44 | 153 | 26 | 156 | 33 | 989 | 25% | 46% |
| kernels: transform/quant | 9.1 | 933 | 3.39 | 72 | 161 | 445 | 2 | 254 | 73% | 11% |
| kernels: loop filters (deblock, CDEF) | 8.0 | 948 | 3.90 | 121 | 104 | 118 | 0 | 605 | 36% | 35% |
| libc, kernel, other | 6.5 | 413 | 2.09 | 32 | 32 | 63 | 3 | 284 | 31% | 25% |
| source analysis, scalar (nc_analysis) | 3.9 | 499 | 4.23 | 42 | 13 | 34 | 1 | 409 | 18% | 47% |
| kernels: other nc_dsp | 2.9 | 304 | 3.46 | 35 | 41 | 19 | 1 | 208 | 31% | 36% |
| kernels: MC and intra prediction | 2.3 | 280 | 4.04 | 65 | 99 | 8 | 2 | 108 | 61% | 38% |

Kernel classes together: ours 30 % (WestWind) and 41 % (Johnny) of cycles; M10 58 % and 62 %;
M12 51 % and 58 %. Stage by stage, in instructions on WestWind:

| Stage | ours AV1 Default | SVT M10 | What differs |
|---|---:|---:|---|
| Coefficient coding (scalar in both) | 5443 M | 1333 M | We trellis every coded P-frame block and price trials on live CDFs; M10 does neither on P frames (§3.4). |
| Mode decision, block glue, analysis (scalar in both) | 8384 M | 3892 M | More blocks, decisions and variance probes per frame on the profiled clips (§3.3). |
| Variance, sub-pel variance, SAD | 3234 M, 40 % vector, 40 % of that 128-bit | 4040 M, 44 % vector, 24 % 128-bit | Ours has more 128-bit work (16-wide blocks). |
| Transform and quantisation | 1562 M, 55 % vector, 28 % 128-bit | 1664 M, 75 % vector, 8 % 128-bit | Level; ours narrower on small transforms. |
| Loop filters | 1695 M, 18 % vector, 52 % 128-bit | 2190 M, 34 % vector, 51 % 128-bit | Ours mostly non-vector instructions (the deblocking edge kernel). |
| MC and intra prediction | 783 M, 34 % vector | 1469 M, 34 % vector | Level. |

### 1.4 Our hottest functions on Zen 5, with their mix

WestWindEasy:

totals: cycles 7.01G instructions 22.63G vector uops 128/256/512 2.74/1.21/1.73G scalar-fp 0.30G

| symbol | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nc_av1::backend::trellis_quantize | 8.6 | 1691 | 2.80 | 202 | 5 | 26 | 91 | 1367 | 14% | 87% |
| nc_av1::coeff::CoeffCdfs::cost | 5.6 | 1195 | 3.04 | 17 | 12 | 40 | 0 | 1126 | 6% | 25% |
| nc_dsp::diff::avx512::variance | 4.5 | 1177 | 3.77 | 219 | 77 | 172 | 15 | 694 | 40% | 47% |
| nc_encoder::decision::choose_inter_mode | 4.3 | 948 | 3.16 | 165 | 65 | 78 | 29 | 611 | 32% | 54% |
| nc_av1::backend::Av1Backend::code_block | 3.8 | 810 | 3.07 | 78 | 56 | 21 | 5 | 650 | 19% | 50% |
| nc_dsp::deblock::avx2::av1_edge | 3.6 | 989 | 3.96 | 72 | 0 | 0 | 2 | 915 | 7% | 100% |
| nc_dsp::diff::avx512::sub_pixel_variance_w16 | 3.5 | 1098 | 4.48 | 174 | 57 | 245 | 8 | 613 | 43% | 36% |
| nc_av1::range_coder::SymbolWriter::write_interval | 3.3 | 634 | 2.77 | 0 | 0 | 0 | 0 | 634 | 0% | 0% |
| nc_av1::coeff::CoeffCdfs::write | 3.0 | 616 | 2.95 | 35 | 0 | 5 | 0 | 576 | 6% | 88% |
| nc_dsp::txfm::avx512_quant::levels | 2.4 | 566 | 3.39 | 41 | 145 | 145 | 3 | 232 | 59% | 12% |
| nc_av1::backend::Av1Backend::code_plane | 2.0 | 545 | 3.91 | 81 | 45 | 17 | 5 | 397 | 26% | 57% |
| nc_av1::backend::inter_mode_bits | 2.0 | 362 | 2.61 | 61 | 13 | 30 | 11 | 247 | 29% | 59% |
| nc_dsp::cdef::avx512::block | 1.6 | 428 | 3.72 | 58 | 14 | 61 | 0 | 295 | 31% | 44% |
| nc_av1::backend::Av1Backend::predict_plane | 1.6 | 396 | 3.62 | 60 | 17 | 11 | 7 | 301 | 22% | 68% |
| nc_dsp::mc::av1_avx2::put_taps | 1.5 | 380 | 3.52 | 59 | 77 | 7 | 5 | 232 | 38% | 41% |
| nc_av1::backend::quantized_eob | 1.4 | 428 | 4.36 | 20 | 34 | 70 | 0 | 303 | 29% | 16% |
| nc_encoder::symbol_cost::SymbolCosts<_>::refill | 1.4 | 210 | 2.16 | 10 | 1 | 7 | 2 | 189 | 9% | 55% |
| nc_av1::mvref::find_mv_stack | 1.3 | 167 | 1.84 | 13 | 0 | 0 | 0 | 155 | 8% | 100% |
| <nc_av1::coeff::trellis::RateModel as nc_encoder::trellis::CoeffRateModel>::level_rate | 1.2 | 244 | 2.91 | 38 | 0 | 0 | 13 | 194 | 15% | 100% |
| nc_av1::range_coder::SymbolWriter::write_fixed | 1.2 | 290 | 3.53 | 23 | 2 | 4 | 1 | 260 | 10% | 79% |
| nc_encoder::intra_estimate::estimate | 1.2 | 346 | 4.22 | 8 | 0 | 0 | 3 | 336 | 2% | 100% |
| <nc_av1::backend::Av1Backend as nc_encoder::backend::CodecBackend>::end_frame | 1.1 | 267 | 3.40 | 0 | 0 | 0 | 0 | 267 | 0% | 0% |
| nc_encoder::decision::Measurer::measure_four | 1.0 | 204 | 2.85 | 50 | 10 | 51 | 2 | 91 | 54% | 45% |
| nc_encoder::encoder::Encoder<B>::code_leaf | 1.0 | 174 | 2.51 | 26 | 0 | 0 | 1 | 147 | 15% | 100% |
| nc_dsp::txfm::avx512::forward_16 | 1.0 | 229 | 3.33 | 33 | 14 | 71 | 0 | 110 | 52% | 28% |
| nc_dsp::pixel::avx512::scan | 1.0 | 278 | 4.10 | 40 | 2 | 22 | 0 | 214 | 23% | 62% |
| nc_av1::mvcoding::MvCdfs::bits | 1.0 | 235 | 3.46 | 31 | 4 | 10 | 5 | 185 | 19% | 69% |
| nc_dsp::pixel::avx512::grid_x4 | 0.9 | 192 | 2.98 | 25 | 5 | 19 | 1 | 142 | 26% | 52% |
| nc_analysis::motion::SbSearchContext::run | 0.9 | 215 | 3.57 | 33 | 7 | 15 | 0 | 160 | 26% | 59% |
| nc_encoder::partition_plan::Planner<B>::plan_node | 0.9 | 260 | 4.32 | 41 | 11 | 33 | 9 | 166 | 33% | 48% |
| nc_av1::backend::Av1Backend::tree_transform_cost | 0.8 | 222 | 3.77 | 59 | 14 | 12 | 3 | 133 | 39% | 69% |
| nc_encoder::decision::Measurer::measure | 0.8 | 186 | 3.27 | 28 | 6 | 19 | 3 | 129 | 29% | 53% |
| nc_av1::mvref::Scan<C>::row | 0.8 | 115 | 2.03 | 11 | 0 | 0 | 1 | 103 | 10% | 100% |
| nc_dsp::cdef::avx2_interior::block4_pair | 0.8 | 138 | 2.59 | 19 | 27 | 0 | 0 | 92 | 33% | 41% |

Johnny:

totals: cycles 1.78G instructions 5.83G vector uops 128/256/512 0.86/0.35/0.75G scalar-fp 0.06G

| symbol | cycles % | instr M | IPC | 128-bit M | 256-bit M | 512-bit M | scalar-fp M | non-vector instr M | vector uops / instr | of vector: 128-bit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| nc_dsp::diff::avx512::variance | 8.9 | 586 | 3.70 | 85 | 33 | 167 | 4 | 297 | 49% | 30% |
| nc_dsp::deblock::avx2::av1_edge | 6.8 | 486 | 4.00 | 59 | 0 | 0 | 1 | 425 | 12% | 100% |
| nc_av1::backend::Av1Backend::code_block | 4.3 | 221 | 2.91 | 31 | 4 | 9 | 1 | 177 | 19% | 72% |
| nc_av1::coeff::CoeffCdfs::cost | 4.3 | 268 | 3.52 | 4 | 2 | 5 | 0 | 257 | 4% | 39% |
| nc_dsp::pixel::avx512::grid_x4 | 4.0 | 163 | 2.27 | 45 | 3 | 38 | 1 | 76 | 53% | 52% |
| nc_encoder::decision::choose_inter_mode | 3.5 | 189 | 3.05 | 36 | 10 | 27 | 6 | 109 | 39% | 49% |
| nc_av1::backend::trellis_quantize | 2.8 | 141 | 2.83 | 23 | 0 | 1 | 10 | 107 | 17% | 95% |
| nc_dsp::mc::av1_avx2::put_taps | 2.7 | 175 | 3.59 | 44 | 87 | 1 | 1 | 41 | 76% | 33% |
| nc_dsp::diff::avx512::sub_pixel_variance_wide | 2.3 | 143 | 3.57 | 8 | 0 | 84 | 0 | 51 | 64% | 9% |
| nc_av1::backend::Av1Backend::code_plane | 2.0 | 142 | 4.02 | 38 | 22 | 5 | 1 | 76 | 46% | 59% |
| nc_encoder::partition_plan::Planner<B>::plan_node | 1.7 | 121 | 3.99 | 24 | 3 | 9 | 5 | 81 | 29% | 67% |
| nc_analysis::preanalysis::analyze | 1.6 | 113 | 3.98 | 0 | 4 | 2 | 0 | 107 | 5% | 0% |
| nc_dsp::diff::avx512::sub_pixel_variance_w16 | 1.5 | 118 | 4.39 | 27 | 6 | 33 | 1 | 52 | 55% | 41% |
| nc_av1::range_coder::SymbolWriter::write_interval | 1.5 | 79 | 3.08 | 0 | 0 | 0 | 0 | 79 | 0% | 0% |
| nc_analysis::motion::SbSearchContext::run | 1.5 | 93 | 3.60 | 39 | 3 | 13 | 0 | 38 | 59% | 72% |
| nc_analysis::noise::update | 1.3 | 49 | 2.12 | 0 | 0 | 0 | 0 | 49 | 0% | 0% |
| nc_av1::backend::Av1Backend::predict_plane | 1.3 | 100 | 4.44 | 21 | 3 | 5 | 2 | 70 | 29% | 74% |
| _copy_to_iter | 1.3 | 5 | 0.24 | 1 | 0 | 5 | 0 | 0 | 113% | 13% |
| nc_dsp::pixel::avx512::scan | 1.3 | 79 | 3.54 | 18 | 1 | 14 | 0 | 46 | 42% | 54% |
| nc_dsp::txfm::avx512_quant::levels | 1.2 | 78 | 3.57 | 6 | 17 | 32 | 0 | 22 | 71% | 10% |
| nc_av1::mvref::find_mv_stack | 1.2 | 48 | 2.22 | 3 | 0 | 0 | 0 | 45 | 6% | 100% |
| nc_av1::backend::inter_mode_bits | 1.1 | 58 | 3.06 | 10 | 1 | 7 | 2 | 37 | 32% | 56% |
| nc_av1::loopfilter::transform_starts | 1.0 | 22 | 1.22 | 0 | 0 | 0 | 0 | 22 | 0% | 0% |
| nc_av1::coeff::CoeffCdfs::write | 1.0 | 41 | 2.25 | 5 | 0 | 0 | 0 | 36 | 12% | 100% |

### 1.5 Candidates, ranked by expected Zen 5 gain

Shares are of our AV1 Default's Zen 5 cycles on WestWindEasy / Johnny / lowlight (the last from
the morning's record). The gain column is **[inferred]**: share x (1 - 1/k) for the stated
instruction reduction k; the confirming test is the kernel's `nc-checkasm --bench` on c8a and a
paired timing of the encoder.

| # | Work | Cycles % (WestWind / Johnny / lowlight) | Measured mix | SVT's equivalent | Assumed k | Gain % |
|---|---|---|---|---|---|---|
| 1 | AV1 deblocking edge kernel `nc_dsp::deblock::avx2::av1_edge` (one edge per call, 16-byte lanes, `deblock/avx2.rs:31`), plus its control (`loopfilter::transform_starts`, `filter_frame`) | 3.6 / 6.8 / 4.1, control +0.5-1.7 | 7-12 % vector uops per instruction, 100 % of them 128-bit; 915 of 989 M instructions non-vector | `svt_aom_lpf_*_sse2` are 128-bit too, but M10 runs them on 2 of 240 superblocks per frame on lowlight and M11-M13 never (§3.4) | 3 (whole 32/64-sample edge runs per call, 256/512-bit) | 2.4 / 4.5 / 2.7 |
| 2 | Pixel and statistics loops outside nc-dsp: `nc_encoder::intra_estimate::estimate` (1.2 %, 2 % vector), `nc_analysis::preanalysis::analyze` (0.6-1.6 %, 5 % vector), `nc_analysis::noise::update` (0.65-1.3 %, no vector), `nc_av1::backend::quantized_eob` (0.8-1.5 %), `zero_residual::residual_quantises_to_zero_scaled` (0.5-0.7 %), `still::StillTracker::update` (0.4 %) | 3.3 / 3.7 / 3.7 | scalar | SVT's analysis is kernels (`compute_b64_variance`, `svt_compute_interm_var_four8x8_avx2`); its quantiser returns the end of block | 4 | 2.5 / 2.8 / 2.8 |
| 3 | Small-block variance `nc_dsp::diff::avx512::variance` | 4.5 / 8.9 / 5.6 | 40-49 % vector, 30-47 % of it 128-bit, 59 % non-vector instructions (22 000 calls a frame) | `svt_aom_variance16x16_avx2` 38 % vector, 62 % non-vector: no better per call; M10 makes 4400 calls a frame against our 22 000 | 1.5 (batches of four 16x16 per call) | 1.5 / 3.0 / 1.9 |
| 4 | `sub_pixel_variance_w16` | 3.5 / 1.5 / 2.2 | 43-55 % vector, 36-41 % 128-bit | `svt_aom_sub_pixel_variance16xh_avx2` 68 % vector, 23 % 128-bit | 1.3 | 0.8 / 0.4 / 0.5 |
| 5 | CDEF block kernels (`cdef::avx512::block`, `avx2_interior::block4_pair`, `direction_avx2::find_dir`) | 3.1 / n.l. / 2.0 | 31-42 % vector, 25-44 % 128-bit | `svt_av1_cdef_filter_block_4xn_8_native_avx2` 27 % vector | 1.3 | 0.7 / - / 0.5 |
| 6 | AV1 MC `mc::av1_avx2::put_taps` (no AVX-512 version) | 1.5 / 2.7 / 0.7 | 38-76 % vector, none 512-bit | `convolve_2d_sr_*_avx512` | 1.5 | 0.5 / 0.9 / 0.2 |
| 7 | 8-point forward transform `txfm::avx2::forward_8` | 0.7 / n.l. / 0.6 | 96 % of its vector uops 128-bit | - | 1.5 | 0.2 |
| | **Sum of 1-7** | | | | | **about 8-12** |

Not on this list, because they are not a width problem:

- **Coefficient coding** (26.0 / 12.2 / 28 % of cycles): `trellis_quantize` 8.6 / 2.8 / 10.0 %,
  `CoeffCdfs::cost` 5.6 / 4.3 / 8.9 %, the writer (`SymbolWriter::write_interval`,
  `CoeffCdfs::write`, `write_fixed`) 7.5 / 3.2 / 4.9 %. Vectorising the context derivation was
  measured this morning and gained nothing (`docs/handoff/avx512.md`, "not landed"). Two facts
  are new here: on WestWind we write 1.01 MB with 1540 M instructions in the coefficient writer
  and range coder, M10 writes 1.03 MB with 761 M (`av1_write_coeffs_txb_1d` +
  `svt_od_ec_encode_cdf_q15`), so **ours cost twice M10's per byte**; and M10 runs no RDOQ on
  P frames (0 `svt_av1_optimize_b` calls) and 241-528 coefficient-rate walks a P frame against
  our 1070-1417 (§3.3).
- **Block glue** (`code_block`, `code_plane`, `predict_plane`, `choose_inter_mode`,
  `inter_mode_bits`, `mvref`): 14-17 % of cycles, 60-75 % non-vector. It follows the number of
  blocks and candidates (§3.3), not the instruction width.

H.264 Default has the same shape (WestWind: coefficient coding 16.7 % of cycles, mode decision
18.4 %, block coding 15.1 %, kernels 41 %; its SAD/variance kernels have 57 % of their vector
uops at 128 bits because macroblocks are 16 wide).

## 2. The Fast rungs at fixed QP

### 2.1 AV1 Fast, one lever group at a time

Each row is AV1 Fast (the lens's own parameter list) with one group of parameters back at the
Default's values. Weighted BD-rate in % against Fast per sequence (negative: fewer bits), the
mean, and the mean against x264 medium (the chart's figure). `fast-A-B` puts back several groups.

| variant | Johnny | KristenAn | Controlle | WestWindE | BlueSky | Stockholm | RedKayak | SnowMount | SpeedBag | Touchdown | FourPeopl | Vertical | mean | vs x264 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | -31.9 | -28.3 | -21.3 | -22.5 | -30.4 | -32.2 | -28.3 | -14.4 | -21.7 | -46.2 | -30.1 | -25.1 | -27.71 | -17.9 |
| fast-depth-txsplit-partial_fwd-deblock-unified_only-key_intra-cdef | -25.8 | -24.8 | -20.1 | -20.1 | -24.5 | -21.9 | -24.2 | -14.6 | -16.1 | -42.0 | -30.1 | -19.3 | -23.63 | -14.8 |
| fast-depth-txsplit-partial_fwd-deblock-unified_only | -22.2 | -21.6 | -18.6 | -19.7 | -25.0 | -20.7 | -18.9 | -12.4 | -12.8 | -38.1 | -27.6 | -18.0 | -21.29 | -12.8 |
| fast-depth-txsplit-partial_fwd-deblock | -20.6 | -21.0 | -18.0 | -18.9 | -24.9 | -17.8 | -18.8 | -3.3 | -12.2 | -37.5 | -27.5 | -17.7 | -19.85 | -11.2 |
| fast-depth-partial_fwd-deblock | -20.2 | -20.2 | -17.6 | -18.2 | -23.5 | -17.9 | -17.6 | -2.9 | -12.5 | -33.9 | -25.7 | -17.2 | -18.95 | -9.2 |
| fast-depth-txsplit-partial_fwd | -17.4 | -15.6 | -9.5 | -10.5 | -25.6 | -22.0 | -6.1 | -6.0 | -4.1 | -33.8 | -25.5 | -13.4 | -15.80 | -8.4 |
| fast-depth-txsplit | -17.4 | -15.7 | -9.3 | -10.5 | -23.0 | -21.2 | -7.5 | -5.3 | -3.4 | -33.0 | -26.0 | -14.3 | -15.56 | -8.1 |
| fast-depth-partial_fwd | -16.7 | -15.4 | -9.3 | -9.8 | -24.5 | -21.3 | -4.9 | -5.6 | -4.7 | -30.6 | -23.9 | -13.1 | -15.00 | -6.5 |
| fast-depth | -16.6 | -15.4 | -9.0 | -9.6 | -13.6 | -20.0 | -6.3 | +1.6 | -3.9 | -29.2 | -24.2 | -11.3 | -13.12 | -3.8 |
| fast-depth_th | -16.6 | -15.4 | -9.0 | -9.6 | -13.6 | -20.0 | -6.3 | +1.6 | -3.9 | -29.2 | -24.2 | -11.3 | -13.12 | -3.8 |
| fast-depth_minq | -14.7 | -13.8 | -8.6 | -9.6 | -12.4 | -18.7 | -6.3 | +1.6 | -3.6 | -27.8 | -22.4 | -10.7 | -12.27 | -3.3 |
| fast-txsplit | -8.8 | -7.2 | -8.6 | -7.4 | -17.2 | -15.2 | -6.3 | -7.9 | -2.8 | -18.9 | -10.6 | -9.4 | -10.04 | -0.6 |
| fast-partial_fwd | -4.6 | -3.3 | -7.5 | -7.3 | -18.7 | -14.0 | -4.2 | -8.9 | -1.8 | -9.2 | -3.2 | -6.6 | -7.44 | +3.8 |
| fast-deblock | -2.8 | -4.7 | -7.4 | -7.5 | -0.0 | +3.9 | -10.9 | +1.3 | -8.1 | -4.1 | -1.3 | -4.4 | -3.85 | -0.9 |
| fast-structure | -2.3 | -1.0 | -1.4 | -1.7 | -0.0 | -5.0 | -0.4 | -16.7 | -0.9 | -2.6 | -0.4 | -0.7 | -2.76 | -0.2 |
| fast-unified_only | -2.3 | -1.0 | -1.4 | -1.7 | -0.0 | -5.0 | -0.4 | -16.7 | -0.9 | -2.6 | -0.4 | -0.7 | -2.76 | -0.2 |
| fast-key_intra | -4.3 | -3.7 | -1.1 | -0.2 | -1.0 | -2.3 | -0.5 | -5.2 | -1.6 | -0.4 | -2.9 | -1.6 | -2.07 | -0.1 |
| fast-cdef | -0.2 | -0.7 | -2.0 | +0.1 | +3.0 | +1.8 | -8.4 | +0.1 | -2.1 | -11.8 | +0.6 | -1.3 | -1.74 | +0.0 |
| fast-subpel_bias | +0.0 | -0.2 | -0.8 | -0.4 | -6.4 | -2.5 | -1.6 | +0.6 | -0.6 | -2.8 | +0.2 | -2.6 | -1.42 | +0.7 |
| fast-trellis | -1.0 | -1.0 | -0.9 | -1.1 | +1.1 | -0.2 | -5.3 | -0.6 | -1.4 | -1.1 | -1.3 | -2.1 | -1.24 | +0.4 |
| fast-p_intra | -0.3 | -0.2 | -0.7 | -1.7 | -0.4 | -0.1 | -2.1 | -0.0 | -1.0 | -0.4 | -0.4 | -0.8 | -0.67 | +0.9 |
| fast-txrd | -1.8 | -0.8 | -0.6 | -0.3 | -0.4 | +0.7 | -0.8 | -0.3 | -0.8 | -1.5 | +0.0 | -1.2 | -0.65 | +1.5 |
| fast-cheap_refine | -1.2 | -0.5 | +0.0 | +0.1 | +1.0 | +5.3 | -0.3 | -8.7 | +0.2 | -1.4 | -0.1 | -1.9 | -0.63 | +1.0 |
| fast-pretx | -0.6 | -1.1 | -0.4 | +1.1 | -0.9 | +0.1 | +0.3 | -0.5 | -0.8 | +0.2 | +0.4 | -0.0 | -0.19 | +1.1 |
| fast-probe_gate | -0.2 | +0.0 | -0.1 | +0.1 | -0.5 | -0.0 | -0.2 | +0.0 | +0.0 | +0.0 | +0.0 | -0.9 | -0.15 | +1.5 |
| fast-chroma_q | +0.2 | +0.3 | +0.8 | -1.3 | +1.1 | +0.4 | -0.5 | -0.1 | -0.1 | +0.1 | +0.2 | -2.2 | -0.09 | +2.0 |
| fast | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +1.6 |
| fast-seed_walk | -1.0 | -0.3 | -0.1 | +0.2 | +8.9 | +7.4 | -0.2 | -0.0 | -1.1 | -0.6 | +0.9 | -0.4 | +1.14 | +2.4 |

Groups: `depth` = `planner.depth_removal_min_quantizer` 0 and the thresholds 25/50 (the Default
has thresholds 0, i.e. no depth removal, and 0.8); `txsplit` = `av1.inter_tx_split` 0;
`partial_fwd` = `av1.partial_forward` 2; `deblock` = `av1.loop_filter` 0; `structure` =
`structure.unified` 0, `structure.inherited_depth` 0, `rate.key_still_boost` 0,
`rate.period_ceiling` 0; `unified_only` = `structure.unified` alone (with it on, the fixed-QP
profile's `structure.secondary_min_depth 0` and `refs.second_full 1` become active); `key_intra`
= the `av1.keyframe_*` set; `cdef` = the `av1.cdef_local_*` values; `subpel_bias`, `seed_walk`,
`cheap_refine`, `probe_gate` = the `decision.*` motion levers; `pretx`, `txrd` = the
pre-transform and transform-RD skips; `p_intra` = `decision.intra_directional` and
`av1.inter_intra_search`; `trellis` = `quant.trellis` 0; `chroma_q` = `av1.chroma_delta_q` and
noise scale.

- The three large groups are `depth` (-13.1), `txsplit` (-10.0) and `partial_fwd` (-7.4);
  then `deblock` (-3.9), the unified structure (-2.8), key intra (-2.1), CDEF (-1.7).
- **The fixed-QP profile is not the cause**: `unified_only` and `structure` are the same -2.8
  (the other three structure parameters do nothing at fixed QP), of which SnowMountain is -16.7
  and the other eleven sequences -0.0 to -5.0.
- The two columns do not rank alike, because the figure against x264 integrates over x264's
  quality range only: the full forward transform alone is -7.4 against Fast but +3.8 against
  x264 (Fast itself: +1.6), since at the coarse quantizers it spends more bytes than it gains
  (§2.2). It pays only together with the depth settings (-3.8 to -6.5).
- The pre-transform skip, transform-RD skip, probe gate, chroma offset and P-frame intra search
  (the levers that make Fast content-adaptive, §3) cost 0.1-0.7 each at fixed QP.

### 2.2 The mechanism: large transforms without their high frequencies

The same variants by quantizer (mean over the 12 sequences of the change in bytes and in
PSNR-Y against Fast at the same quantizer):

| AV1 Fast with one group at the Default's values | q110 | q135 | q160 | q185 | q210 | q235 |
|---|---:|---:|---:|---:|---:|---:|
| depth removal thresholds | +28% bytes, +2.72 dB | +15% bytes, +1.53 dB | +10% bytes, +0.93 dB | +3% bytes, +0.44 dB | +1% bytes, +0.22 dB | +1% bytes, +0.13 dB |
| inter transform split on | +34% bytes, +2.87 dB | +22% bytes, +1.76 dB | +11% bytes, +0.86 dB | +3% bytes, +0.30 dB | +0% bytes, +0.07 dB | -0% bytes, +0.02 dB |
| full forward transform | +47% bytes, +3.12 dB | +34% bytes, +1.92 dB | +21% bytes, +0.95 dB | +10% bytes, +0.36 dB | +3% bytes, +0.11 dB | +1% bytes, +0.04 dB |
| deblocking on | -1% bytes, +0.03 dB | -0% bytes, +0.03 dB | -1% bytes, +0.04 dB | -1% bytes, +0.07 dB | -1% bytes, +0.10 dB | -1% bytes, +0.15 dB |
| all (the Default) | +32% bytes, +3.77 dB | +21% bytes, +2.58 dB | +10% bytes, +1.61 dB | +2% bytes, +0.98 dB | -3% bytes, +0.58 dB | -3% bytes, +0.47 dB |

At qindex 110 any one of the first three levers is worth 2.7-3.1 dB, and all of the Default
together 3.8 dB: they are not three independent losses. The reading below is from the code and
fits those figures; that the depth and split levers work by moving blocks onto exact 16x16
transforms is **[inferred]** (test: the transform-size histogram of Fast and Fast + depth at
qindex 110, `eval/nceval/streamstats.py`).

- `av1.partial_forward 2` computes only the low-frequency part of every 32x32 and 64x64 forward
  transform, in coding as well as in trials, at every quantizer
  (`nc-av1/src/backend.rs` `forward_policy`: `large: applies`). Its 16x16 counterpart is already
  gated by quantizer (`av1.partial_forward16_min_q`, `medium: applies && frame_q_idx >= ...`).
- Depth removal at every quantizer (`planner.depth_removal_min_quantizer 0`, thresholds 25/50)
  keeps superblocks from splitting below 32x32, so the blocks stay at the transform sizes the
  approximation hits. Without it blocks split to 16x16, whose transform is exact.
- `av1.inter_tx_split 0` removes the other route (four 16x16 transforms inside a 32x32 block).

At qindex 210-235 the same levers move PSNR-Y by 0.02-0.22 dB, and the full forward transform
spends 1-3 % more bytes for 0.04-0.11 dB; at qindex 185 they are worth 0.3-0.4 dB for 3-10 %
more bytes. The CBR chart's 600 kbit/s rung is at qindex 188-216 on its clips, the lens spans
110-235: that is why Fast holds up on the chart (J -38.2 against the Default's -46.2 there) and
not on the lens.

SVT's equivalent is gated on the block's measured energy, not unconditional: light PD1 uses its
N4 partial transform only when `100 * MDS0 distortion < use_mds3_shortcuts_th * area * qindex`
(`product_coding_loop.c:9148-9153`, checked), and regular PD1 when the earlier stage found
almost no coefficients (`apply_pf_on_coeffs`, `product_coding_loop.c:4612-4618`, checked). M12
and M13 have no transform split either (`txs_level 0`, `enc_mode_config.c:9929-9935`, checked)
and use depth removal levels 6-7, and still hold -1.5 and +4.1 on this lens.

### 2.3 HEVC Fast

| variant | Johnny | KristenAn | Controlle | WestWindE | BlueSky | Stockholm | RedKayak | SnowMount | SpeedBag | Touchdown | FourPeopl | Vertical | mean | vs x264 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| default | -23.7 | -24.8 | -5.5 | -17.0 | -31.3 | -45.4 | -2.9 | -8.3 | -18.3 | -31.9 | -28.4 | -21.8 | -21.60 | -14.9 |
| fast-planner-tu_split-motion-intra-structure-merge | -24.0 | -24.0 | -6.6 | -15.9 | -31.7 | -44.0 | -2.5 | -9.2 | -17.5 | -31.0 | -25.5 | -23.0 | -21.24 | -14.5 |
| fast-planner-tu_split-motion-intra-structure | -22.6 | -22.6 | -6.2 | -15.8 | -30.2 | -44.1 | -2.2 | -8.9 | -15.6 | -30.0 | -25.1 | -20.6 | -20.32 | -13.8 |
| fast-planner-tu_split-motion-intra | -19.5 | -20.7 | -7.0 | -11.8 | -27.3 | -15.5 | -8.3 | -4.1 | -14.2 | -23.1 | -23.5 | -20.5 | -16.29 | -10.2 |
| fast-planner-tu_split-motion | -18.1 | -19.0 | -2.3 | -8.6 | -27.3 | -15.5 | -6.4 | -3.1 | -11.8 | -23.0 | -22.5 | -20.3 | -14.81 | -8.9 |
| fast-planner-tu_split-intra | -13.2 | -16.1 | -7.0 | -11.7 | -18.0 | -13.0 | -8.4 | -4.2 | -10.5 | -19.4 | -22.1 | -14.8 | -13.20 | -6.8 |
| fast-planner-tu_split | -11.8 | -14.4 | -2.4 | -8.5 | -17.9 | -12.7 | -6.5 | -3.6 | -7.7 | -18.9 | -20.8 | -14.8 | -11.67 | -5.4 |
| fast-depth_th | -8.9 | -10.2 | -0.2 | -5.1 | -13.9 | -9.3 | -4.9 | +0.1 | -3.6 | -14.0 | -13.9 | -12.0 | -7.98 | -1.3 |
| fast-planner | -8.5 | -9.9 | -0.6 | -5.4 | -14.0 | -9.4 | -5.1 | -0.2 | -3.6 | -14.1 | -13.2 | -11.8 | -7.98 | -1.4 |
| fast-depth_minq | -8.5 | -9.9 | -0.2 | -5.1 | -13.7 | -9.2 | -4.9 | +0.1 | -3.6 | -13.8 | -13.7 | -11.8 | -7.85 | -1.3 |
| fast-tu_split | -4.1 | -6.0 | -2.3 | -3.9 | -5.3 | -4.2 | -1.3 | -3.9 | -4.9 | -7.5 | -8.9 | -3.8 | -4.68 | +2.9 |
| fast-motion | -6.1 | -4.4 | +0.2 | -0.0 | -8.5 | -3.5 | +0.2 | +0.1 | -3.6 | -3.9 | -0.6 | -4.5 | -2.91 | +4.8 |
| fast-intra | -2.8 | -3.3 | -4.6 | -3.5 | -0.8 | -1.5 | -1.6 | -3.8 | -3.7 | -1.2 | -4.5 | -1.0 | -2.68 | +5.3 |
| fast-structure | -2.7 | -2.2 | +1.0 | -1.1 | -1.2 | -13.5 | +6.6 | -5.4 | +0.2 | -5.1 | -1.4 | +1.1 | -1.99 | +6.7 |
| fast-trellis | -2.3 | -3.4 | -1.1 | -1.5 | -0.2 | -0.1 | -3.8 | +0.4 | -2.6 | -2.3 | -4.6 | -0.5 | -1.83 | +5.3 |
| fast-merge | -2.9 | -2.8 | -0.3 | -0.2 | -2.2 | -0.7 | -0.3 | -0.7 | -1.7 | -1.1 | +0.1 | -2.2 | -1.25 | +6.8 |
| fast-profile_only | +0.0 | +0.0 | +1.8 | -0.5 | -0.9 | -10.7 | +6.6 | +0.0 | +0.5 | -0.7 | +0.0 | +1.4 | -0.22 | +8.5 |
| fast-planner_lambda | +0.4 | +0.2 | -0.3 | -0.2 | -0.5 | -0.5 | -0.1 | -0.5 | +0.4 | -0.2 | +0.2 | -0.1 | -0.10 | +7.7 |
| fast-cadence_boost | +1.0 | +0.8 | -0.2 | -0.4 | -0.2 | -0.4 | +0.0 | +0.0 | -0.9 | -0.9 | +0.6 | -0.1 | -0.05 | +8.1 |
| fast-sao | -0.2 | -0.6 | +0.1 | +0.1 | -0.3 | -0.2 | -0.0 | +0.0 | +0.3 | +0.6 | +0.1 | -0.1 | -0.03 | +7.9 |
| fast | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +7.9 |
| fast-chroma_q | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +7.9 |
| fast-fast_fwd | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +7.9 |
| fast-pre_zero | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.0 | +0.00 | +7.9 |
| fast-rounding | +1.0 | +0.7 | +1.8 | +2.3 | +1.5 | +0.7 | +2.0 | -4.9 | +2.3 | +1.0 | -0.1 | +1.0 | +0.77 | +8.9 |

Groups: `planner` = depth removal at every quantizer with thresholds 25/50, and
`planner.lambda_scale` 2 / `split_rate_factor` 2 (the Default: thresholds 0, 0.86, 2.58 and
2.26); `tu_split` = `hevc.inter_tu_split` 0, `hevc.intra_tu_depth` 0; `motion` = the
`decision.*` motion levers; `intra` = `hevc.intra_nxn`, `intra_leaf_cost`, `inter_intra_dc_only`,
`decision.intra_directional`; `structure` as for AV1 plus `rate.fixed_qp_profile` 0;
`profile_only` = `rate.fixed_qp_profile` alone.

- Depth removal is again the largest single lever: -8.0, all of it from switching the removal
  off (`depth_th` -8.0, `depth_minq` -7.9, the whole `planner` group -8.0; the planner's λ pair
  alone is -0.1). Then the transform tree (-4.7), motion (-2.9), intra (-2.7), structure (-2.0), trellis
  (-1.8).
- **The profile pinned off is -0.2** (Stockholm -10.7, RedKayak +6.6, the rest within 2).
- `fast_fwd`, `pre_zero` and `chroma_q` change nothing: those values are the Default's already.

### 2.4 Candidate points, timed on Zen 5

c8a.large, one pinned core, head build, median of 3 rotated rounds after a warm-up (worst
max/min of a point 1.06), on the four 720p lens sequences at two of its quantizers (ours AV1
qindex 160 and 210, HEVC QP 32 and 42, SVT 40 and 52, x264 32 and 42), 130 frames. fps is frames
per CPU second over those eight encodes. This is not the chart's timing (three private clips at
the 600 kbit/s rung), so compare points within this table: here M12 is 356 fps (chart 409), M10
216 (229), our AV1 Fast 306 (290), AV1 Default 150 (158). BD is the full 12-sequence lens.

| point | fps, both quantizers | fps, fine point | fps, coarse point | Weighted BD vs x264 medium (lens, 12 sequences) | vs our Fast |
|---|---:|---:|---:|---:|---:|
| ours AV1 faster | 414 | 366 | 476 | +14.5 | +29.67 |
| SVT M13 | 401 | 346 | 475 | +4.1 |  |
| SVT M12 | 356 | 310 | 419 | -1.5 |  |
| ours AV1 faster+depth | 350 | 287 | 448 | +6.5 | +10.58 |
| ours AV1 faster+depth+pf | 345 | 284 | 439 | +6.4 | +7.42 |
| ours HEVC faster | 324 | 265 | 416 | +17.4 | +12.24 |
| ours AV1 faster+depth+pf+deblock | 315 | 260 | 399 | +2.6 | +2.14 |
| ours AV1 fast | 306 | 263 | 364 | +1.6 | +0.00 |
| ours HEVC faster+depth | 269 | 210 | 375 | +4.1 | -0.68 |
| ours HEVC faster+planner | 266 | 208 | 370 | +4.0 | -0.82 |
| ours HEVC fast | 264 | 217 | 335 | +7.9 | +0.00 |
| ours AV1 fast+depth | 253 | 203 | 337 | -3.8 | -13.12 |
| ours AV1 fast+depth+pf | 250 | 202 | 330 | -6.5 | -15.00 |
| ours AV1 fast+depth+pf+deblock | 232 | 188 | 305 | -9.2 | -18.95 |
| ours HEVC fast+planner | 230 | 181 | 315 | -1.4 | -7.98 |
| ours HEVC fast+depth | 225 | 177 | 307 | -1.3 | -7.98 |
| SVT M10 | 216 | 196 | 240 | -14.2 |  |
| ours HEVC fast+planner+tu | 209 | 163 | 292 | -5.4 | -11.67 |
| ours AV1 fast+depth+pf+deblock+txsplit | 206 | 160 | 289 | -11.2 | -19.85 |
| ours AV1 fast+depth+pf+deblock+txsplit+unified | 195 | 152 | 275 | -12.8 | -21.29 |
| ours AV1 default-trellis-keyintra | 172 | 135 | 238 | -16.1 | -25.92 |
| ours AV1 default-trellis | 167 | 131 | 228 | -16.9 | -26.92 |
| ours AV1 default | 150 | 115 | 214 | -17.9 | -27.71 |
| ours HEVC default-trellis | 148 | 120 | 194 | -13.3 | -20.58 |
| ours HEVC default | 125 | 96 | 178 | -14.9 | -21.60 |
| x264 medium | 80 | 60 | 117 | 0.0 |  |

`depth` = the Default's depth settings (thresholds 0), `pf` = `av1.partial_forward 0`,
`deblock` = `av1.loop_filter 1`, `txsplit` = `av1.inter_tx_split 1`, `unified` =
`structure.unified 1`, `planner` = depth plus the Default's planner λ pair, `tu` = the
Default's transform tree.

What each lever costs and buys on AV1 Fast (Zen 5 fps, Weighted against x264 medium):

| Step | fps | Weighted | Step cost | Step gain |
|---|---:|---:|---:|---:|
| Fast | 306 | +1.6 | | |
| + depth | 253 | -3.8 | -17 % fps | -5.4 |
| + full forward transform | 250 | -6.5 | -1 % | -2.7 |
| + deblocking | 232 | -9.2 | -7 % | -2.7 |
| + transform split | 206 | -11.2 | -11 % | -2.0 |
| + unified structure | 195 | -12.8 | -5 % | -1.6 |
| Default without trellis and key intra set | 172 | -16.1 | | |
| Default without trellis | 167 | -16.9 | | |
| Default | 150 | -17.9 | | |

Against the frontier, in this table's own speeds:

- **M10 (216 fps, -14.2).** Nothing we have dominates it. Fast + depth + pf + deblock is faster
  (232) and 5 points behind; the Default without trellis is 2.7 points ahead and 23 % slower.
  Fast + 5 levers (195 fps, -12.8) is dominated by M10.
- **M12 (356 fps, -1.5).** Nothing dominates it. Fast + depth (253 fps, -3.8) and Fast + depth +
  pf (250 fps, -6.5) are better in BD and 29 % slower. Faster + depth + pf + deblock (315 fps,
  +2.6) is dominated.
- **M13 (401 fps, +4.1).** Our AV1 Faster is faster (414) at +14.5.
- HEVC: Fast + depth is 225 fps at -1.3 (dominated by M12); HEVC Default is 125 fps at -14.9.

### 2.5 Recommendations

1. **Re-point AV1 Fast now** (parameters only): Fast with the Default's depth settings and
   `av1.partial_forward 0`: 250 fps, -6.5, against today's 306 fps, +1.6. It leaves the
   dominated region (it sits between M12 and M10 on the frontier) and costs 18 % of Fast's
   speed. Adding `av1.loop_filter 1` gives a second point at 232 fps, -9.2. Both need the CBR
   screen before landing: at the chart's coarse quantizers depth removal off is still a gain
   (+1 % bytes for +0.13-0.22 dB), the full forward transform is slightly negative (+1-3 % bytes
   for +0.04-0.11 dB).
2. **Gate the three levers by quantizer or by block energy instead of pinning them** (code;
   shared engine for depth removal, per codec for the transform levers). The speed they buy is
   needed only where they are harmless. The cheapest version is the existing pattern
   (`planner.depth_removal_min_quantizer`, `av1.partial_forward16_min_q`) applied to
   `av1.partial_forward` on large transforms and to the transform split; SVT's version is the
   energy gate of §2.2. **[inferred]** from the per-quantizer table: with the levers on only
   from about qindex 185, Fast would keep its coarse-point speed (364 fps here) and lose only
   at the fine points. Test: the lens with the gates at 160, 185 and 210.
3. **Beating M10 needs the Default about 26 % faster at -16**, and the measured candidates for
   that are in §1 and §3, not in the presets: trellis off is +11 % fps for +1.0 Weighted (but
   L1 J +1.7 % in CBR, `docs/handoff/avx512.md`); the key intra set +3 % for +0.8; the
   deblocking kernel (§1.5 #1) and the non-kernel pixel loops (#2) about 5-7 % **[inferred]**;
   a cheaper coefficient writer (twice M10's instructions per byte) up to 3-4 % **[inferred:
   1540 M against 761 M of 22.6 G instructions]**.
4. **Beating M12 needs 356 fps at -1.5 or better**: Fast + depth is 253 fps. That is 41 %, more
   than any measured lever; the point to build towards is Faster's speed (414) with depth and
   deblocking restored (315 fps, +2.6 today) plus the gates of 2.
5. **HEVC Fast**: the same depth fix (thresholds 0) takes it from +7.9 to -1.3 for 15 % of its
   speed (264 to 225 fps). It stays behind the AV1 points at equal speed.

## 3. Content adaptivity (lowlight against crowd)

The many-clip confirmation asked for in the second brief (20 pool clips, 11 encoder variants)
was stopped at the QP search by the third brief. What follows is the two chart clips only, at
the chart's matched-bytes quantizers, 150 frames, callgrind on the PC, per P frame (the 150-frame
run minus a key-only run, over 149). Note that the chart's "600 kbit/s rung" commands are fixed
QP at matched bytes (`--qp 188` and `--qp 214` for our AV1), not CBR.

### 3.1 Per stage

Ours, AV1 Default:

| per P frame, M instructions (inclusive) | lowlight | crowd | lowlight/crowd |
|---|---:|---:|---:|
| whole encoder | 84.6 | 84.4 | 1.00 |
| encode_frame (everything but file I/O) | 84.2 | 84.1 | 1.00 |
|   source analysis (pre-analysis, motion search, noise, stillness) | 5.4 | 7.1 | 0.75 |
|     motion search | 2.0 | 3.7 | 0.54 |
|   code_picture | 76.9 | 75.0 | 1.02 |
|     partition planner (VLPD0) | 4.1 | 4.2 | 0.98 |
|     mode decision (choose_inter_mode, incl. second reference) | 18.4 | 20.5 | 0.90 |
|       sub-pel refinement of seeds | 7.8 | 6.0 | 1.30 |
|     block coding (encode_block) | 38.1 | 33.6 | 1.13 |
|       transform split trial (choose_tx_split's tree_transform_cost) | 8.2 | 3.6 | 2.27 |
|       trial transform + rate (residual_rd_cost, all callers) | 7.2 | 3.9 | 1.82 |
|       trellis quantisation | 10.0 | 6.0 | 1.67 |
|       coefficient rate walks (CoeffCdfs::cost) | 4.9 | 1.7 | 2.85 |
|       P-frame intra search (intra_prediction_satd) | 0.1 | 0.9 | 0.07 |
|     loop filters (filter_frame) | 9.7 | 12.5 | 0.78 |
|       CDEF search + filter (apply_cdef) | 5.0 | 7.5 | 0.67 |
|       deblocking edges (kernel only) | 3.6 | 3.9 | 0.92 |
|     end_frame (CDF averaging, packet) | 2.1 | 2.3 | 0.92 |

| per P frame, calls | lowlight | crowd | lowlight/crowd |
|---|---:|---:|---:|
| superblocks planned | 240 | 240 | 1.00 |
| planner nodes costed (plan_node) | 1025 | 974 | 1.05 |
| leaves coded (code_leaf) | 1009 | 1093 | 0.92 |
| mode decisions (choose_inter_mode + second reference) | 1337 | 1292 | 1.04 |
| whole-sample variance probes (Measurer::measure) | 9354 | 11018 | 0.85 |
| sub-pel probe batches of 4 (Measurer::measure_four) | 2687 | 2160 | 1.24 |
| variance kernel calls | 22074 | 25796 | 0.86 |
| sub-pel variance kernel calls | 11656 | 9434 | 1.24 |
| plane codings (code_plane: predict, transform, quantise) | 2979 | 3242 | 0.92 |
| trial transforms with rate (residual_rd_cost) | 1216 | 928 | 1.31 |
| transform split trials (tree_transform_cost) | 1205 | 789 | 1.53 |
| forward transform kernel calls | 3018 | 2829 | 1.07 |
| trellis quantisations | 574 | 649 | 0.88 |
| coefficient rate walks (CoeffCdfs::cost) | 1417 | 1070 | 1.32 |
| coefficient blocks written (CoeffCdfs::write) | 1032 | 1603 | 0.64 |
| pre-transform skip tests (skip_luma_before_transform) | 1214 | 1141 | 1.06 |
| zero-residual tests | 1023 | 1009 | 1.01 |
| P-frame intra SATD candidates | 24 | 323 | 0.07 |
| MV stack builds (stack_for) | 2332 | 2306 | 1.01 |
| predictions (predict_plane) | 4148 | 4815 | 0.86 |
| deblocking edge calls (av1_edge / luma) | 6235 | 6667 | 0.94 |
| CDEF 8x8 blocks filtered | 2157 | 3354 | 0.64 |

Ours, AV1 Fast:

| per P frame, M instructions (inclusive) | lowlight | crowd | lowlight/crowd |
|---|---:|---:|---:|
| whole encoder | 42.0 | 54.4 | 0.77 |
| encode_frame (everything but file I/O) | 41.7 | 54.1 | 0.77 |
|   source analysis (pre-analysis, motion search, noise, stillness) | 5.4 | 7.1 | 0.75 |
|     motion search | 2.0 | 3.7 | 0.54 |
|   code_picture | 34.8 | 45.5 | 0.77 |
|     partition planner (VLPD0) | 2.0 | 3.2 | 0.64 |
|     mode decision (choose_inter_mode, incl. second reference) | 8.0 | 10.6 | 0.75 |
|       sub-pel refinement of seeds | 2.5 | 2.0 | 1.28 |
|     block coding (encode_block) | 17.2 | 20.0 | 0.86 |
|       coefficient rate walks (CoeffCdfs::cost) | 0.5 | 0.3 | 1.57 |
|     loop filters (filter_frame) | 4.8 | 8.0 | 0.60 |
|       CDEF search + filter (apply_cdef) | 4.8 | 8.0 | 0.60 |
|     end_frame (CDF averaging, packet) | 1.8 | 2.4 | 0.76 |

| per P frame, calls | lowlight | crowd | lowlight/crowd |
|---|---:|---:|---:|
| superblocks planned | 240 | 240 | 1.00 |
| planner nodes costed (plan_node) | 1032 | 979 | 1.06 |
| leaves coded (code_leaf) | 698 | 1046 | 0.67 |
| mode decisions (choose_inter_mode + second reference) | 698 | 1046 | 0.67 |
| whole-sample variance probes (Measurer::measure) | 4296 | 8083 | 0.53 |
| sub-pel probe batches of 4 (Measurer::measure_four) | 336 | 428 | 0.79 |
| variance kernel calls | 5110 | 10012 | 0.51 |
| sub-pel variance kernel calls | 2630 | 3224 | 0.82 |
| plane codings (code_plane: predict, transform, quantise) | 1859 | 2947 | 0.63 |
| forward transform kernel calls | 1431 | 2228 | 0.64 |
| coefficient rate walks (CoeffCdfs::cost) | 70 | 83 | 0.85 |
| coefficient blocks written (CoeffCdfs::write) | 518 | 1463 | 0.35 |
| pre-transform skip tests (skip_luma_before_transform) | 677 | 975 | 0.69 |
| MV stack builds (stack_for) | 1394 | 2040 | 0.68 |
| predictions (predict_plane) | 2098 | 3142 | 0.67 |
| CDEF 8x8 blocks filtered | 2118 | 3677 | 0.58 |

Ours, H.264 Default:

| per P frame, M instructions (inclusive) | lowlight | crowd | lowlight/crowd |
|---|---:|---:|---:|
| whole encoder | 58.6 | 61.1 | 0.96 |
| encode_frame (everything but file I/O) | 58.3 | 60.8 | 0.96 |
|   source analysis (pre-analysis, motion search, noise, stillness) | 10.2 | 10.8 | 0.94 |
|     motion search | 6.6 | 7.2 | 0.92 |
|   code_picture | 46.9 | 49.0 | 0.96 |
|     mode decision (choose_inter_mode, incl. second reference) | 20.9 | 18.2 | 1.14 |
|       sub-pel refinement of seeds | 8.3 | 5.4 | 1.53 |
|     block coding (encode_block) | 22.2 | 25.1 | 0.88 |
|       trellis quantisation | 4.6 | 1.3 | 3.40 |
|       deblocking edges (kernel only) | 0.6 | 1.4 | 0.44 |

| per P frame, calls | lowlight | crowd | lowlight/crowd |
|---|---:|---:|---:|
| superblocks planned | 3600 | 3600 | 1.00 |
| leaves coded (code_leaf) | 3600 | 3600 | 1.00 |
| mode decisions (choose_inter_mode + second reference) | 4550 | 3631 | 1.25 |
| whole-sample variance probes (Measurer::measure) | 18398 | 15451 | 1.19 |
| sub-pel probe batches of 4 (Measurer::measure_four) | 2131 | 1368 | 1.56 |
| variance kernel calls | 13207 | 14854 | 0.89 |
| sub-pel variance kernel calls | 16965 | 10875 | 1.56 |
| forward transform kernel calls | 4534 | 8274 | 0.55 |
| trellis quantisations | 720 | 480 | 1.50 |
| deblocking edge calls (av1_edge / luma) | 2841 | 5823 | 0.49 |

SVT M10 (anchor build; the deblocking of M10 runs inside the mode-decision kernel, 0.14 M
instructions per frame on lowlight and 0.76 M on crowd):

| stage | lowlight M/P-frame | calls | crowd M/P-frame | calls | lowlight/crowd |
|---|---:|---:|---:|---:|---:|
| whole encoder | 51.3 | | 79.1 | | 0.65 |
| picture analysis | 1.0 | 1 | 1.0 | 1 | 1.00 |
| motion estimation (open loop) | 3.6 | 1 | 10.5 | 1 | 0.34 |
| mode decision + coding (EncDec) | 34.4 | 1 | 50.3 | 1 | 0.68 |
|   PD0 partition | 1.9 | 240 | 1.9 | 240 | 0.96 |
|   light PD1 (SBs) | 18.6 | 177 | 24.4 | 178 | 0.76 |
|   regular PD1 (SBs) | 12.9 | 63 | 21.6 | 62 | 0.60 |
| deblocking | 0.0 | 1 | 0.0 | 1 | 1.00 |
| CDEF search + filter | 8.6 | 1 | 12.9 | 1 | 0.66 |
| entropy coding | 2.7 | 1 | 3.4 | 1 | 0.79 |
| MD config + RC | 0.6 | 1 | 0.6 | 1 | 1.00 |

x264 medium:

| per P frame (3600 macroblocks) | lowlight calls | lowlight M instr | crowd calls | crowd M instr | calls ratio |
|---|---:|---:|---:|---:|---:|
| whole encoder | | 90.0 | | 119.2 | |
| macroblocks analysed | 3600 | 71.6 | 3600 | 97.9 | 1.00 |
| P-skip probes (x264_macroblock_probe_skip) | 3141 | 3.5 | 2918 | 3.2 | 1.08 |
| 16x16 inter analyses | 3600 | 37.3 | 3600 | 40.5 | 1.00 |
| motion searches (x264_me_search_ref) | 8409 | 43.1 | 11509 | 57.7 | 0.73 |
| sub-pel refinements (refine_subpel) | 11179 | 37.5 | 15389 | 48.5 | 0.73 |
| 8x8 partition analyses | 528 | 16.1 | 772 | 28.0 | 0.68 |
| 16x8 analyses | 0 | 0.0 | 0 | 0.0 | nan |
| 8x16 analyses | 0 | 0.0 | 0 | 0.0 | nan |
| intra analyses in P frames (mb_analyse_intra) | 528 | 2.8 | 772 | 7.6 | 0.68 |
| RD cost evaluations (rd_cost_mb) | 2241 | 8.5 | 3378 | 11.5 | 0.66 |
| macroblock encodes incl. RD trials (x264_macroblock_encode) | 5841 | 8.4 | 6978 | 9.0 | 0.84 |
| qpel RD refinements (me_refine_qpel_rd) | 0 | 0.0 | 0 | 0.0 | nan |
| 8x8 trellis quantisations | 1593 | 1.4 | 2580 | 1.1 | 0.62 |
| 4x4 trellis quantisations | 3530 | 1.1 | 3300 | 0.5 | 1.07 |
| macroblocks written (macroblock_write_cabac) | 513 | 1.0 | 770 | 1.3 | 0.67 |

Which stages shrink on the easy clip:

- **SVT M10** (0.65): motion estimation 0.34, light PD1 0.76, regular PD1 0.60, CDEF 0.66.
- **x264** (0.76): motion search and sub-pel 0.73, 8x8 partitions and intra 0.68, RD 0.66. It
  still runs the 16x16 search on all 3600 macroblocks, of which 513 a frame reach
  `macroblock_write_cabac`.
- **Our AV1 Fast** (0.77): planner 0.64, mode decision 0.75, block coding 0.86, CDEF 0.60.
- **Our AV1 Default** (1.00): analysis 0.75 and CDEF 0.67 shrink; mode decision 0.90; block
  coding *grows* 1.13 (transform-split trials 2.27, trellis 1.67, rate walks 2.85); the easy
  clip is coded at a finer quantizer for the same bytes (qindex 188 against 214).
- **Our H.264 Default** (0.96): motion search 0.92 (its zero-vector shortcuts are switched off
  in the Default, `analysis.static_sad_per_pixel 0` and two more, for J -0.17 %), mode decision
  1.14, trellis 3.4x.

### 3.2 Block-level counts, SVT M10 (uninlined build)

| per P frame (240 superblocks) | lowlight | crowd | lowlight/crowd |
|---|---:|---:|---:|
| superblocks through PD0 (VLPD0) | 240 | 240 | 1.00 |
| PD0 blocks costed (variance model) | 1787 | 2037 | 0.88 |
| superblocks in light PD1 | 177 | 178 | 1.00 |
| superblocks in regular PD1 | 63 | 62 | 1.01 |
| blocks coded by light PD1 | 388 | 613 | 0.63 |
| blocks coded by regular PD1 | 161 | 233 | 0.69 |
| MDS0 candidates, light PD1 | 805 | 2680 | 0.30 |
| MDS0 candidates, regular PD1 | 421 | 1573 | 0.27 |
| full-loop (MDS3) evaluations, light PD1 | 388 | 613 | 0.63 |
| full-loop evaluations, regular PD1 (MDS1+MDS3) | 211 | 445 | 0.47 |
| light-PD1 transform decisions (lpd1_should_perform_tx) | 388 | 613 | 0.63 |
| transform-type searches (regular PD1) | 98 | 149 | 0.66 |
| forward transforms, all planes and types (estimate_transform) | 2290 | 5285 | 0.43 |
| quantisations (svt_aom_quantize_inv_quantize) | 1093 | 2611 | 0.42 |
| RDOQ (svt_av1_optimize_b) | 0 | 0 | nan |
| coefficient cost walks (svt_av1_cost_coeffs_txb) | 241 | 528 | 0.46 |
| sub-pel searches (fixed stage / tree) | 714 | 1169 | 0.61 |
| variance kernel calls | 4396 | 8272 | 0.53 |
| sub-pel variance kernel calls | 9757 | 13191 | 0.74 |
| intra predictions (svt_av1_predict_intra_block) | 94 | 2292 | 0.04 |
| inter predictions, luma+chroma (svt_aom_inter_pu_prediction_av1*) | 1681 | 3019 | 0.56 |
| MVP table builds | 548 | 846 | 0.65 |
| 64x64 ME searches (svt_aom_motion_estimation_b64) | 240 | 240 | 1.00 |
| HME level-0 SAD loops (svt_sad_loop_kernel) | 262 | 822 | 0.32 |
| full-pel 8-position SAD passes (all_sad_calculation_8x8_16x16) | 562 | 2540 | 0.22 |
| superblocks deblocked (svt_aom_loop_filter_sb) | 2 | 32 | 0.05 |
| CDEF filter-block searches (cdef_seg_search) | 0 | 0 | 1.00 |
| CDEF 8x8 blocks filtered | 16805 | 29288 | 0.57 |
| coefficient blocks written (av1_write_coeffs_txb_1d) | 474 | 1103 | 0.43 |

- Every fourth frame (the "base" frames of flat IPP) runs regular PD1 on all 240 superblocks:
  63 superblocks per frame on average; the other frames run light PD1 (`pic_lpd1_lvl = is_base
  ? 0 : ...`, `enc_mode_config.c:10043-10044`, checked). Regular PD1 is 25 % of the frames and
  38-43 % of the mode-decision instructions.
- No RDOQ call on any P frame; one transform-type search per regular-PD1 block on base frames
  (98-149 a frame); no transform-size search.
- Intra predictions in P frames: 94 a frame on lowlight against 2292 on crowd (the frame-level
  `skip_intra` on `norm_me_dist <= 5000`, `enc_mode_config.c:9799-9806`, checked).
- 2 superblocks a frame are deblocked on lowlight, 32 on crowd.

### 3.3 The same counts for us, and the comparison

The call rows of §3.1 against M10's, per P frame on lowlight / crowd:

| | ours AV1 Default | ours AV1 Fast | SVT M10 |
|---|---:|---:|---:|
| Blocks coded | 1009 / 1093 | 698 / 1046 | 549 / 846 |
| Partition nodes costed | 1025 / 974 | 1032 / 979 | 1787 / 2037 |
| Mode decisions | 1337 / 1292 (incl. the second reference of the fixed-QP profile) | 698 / 1046 | 549 / 846 |
| Whole-sample variance kernel calls | 22074 / 25796 | 5110 / 10012 | 4396 / 8272 |
| Sub-pel variance kernel calls | 11656 / 9434 | 2630 / 3224 | 9757 / 13191 |
| Forward transforms (kernel calls; SVT: `estimate_transform`, all planes and types) | 3018 / 2829 | 1431 / 2228 | 2290 / 5285 |
| Trial transforms with a rate walk | 1216 / 928 | 0 | 241 / 528 (`svt_av1_cost_coeffs_txb`) |
| Trellis / RDOQ | 574 / 649 | 0 | 0 / 0 |
| Coefficient blocks written | 1032 / 1603 | 518 / 1463 | 474 / 1103 |
| P-frame intra candidates | 24 / 323 | 0 | 94 / 2292 |
| Deblocking | 6235 / 6667 edge calls | off | 2 / 32 superblocks |

- We cost fewer partition nodes than M10 but **code 1.8x the blocks on the easy clip** (1.3x on
  the busy one). Why our planner ends with more leaves on lowlight is **not known yet**; the
  test is a partition-size histogram of both streams on the same frames
  (`eval/nceval/streamstats.py`, `aom-inspect`).
- We make **5x M10's whole-sample variance calls** on lowlight (22 000 against 4400); the
  sub-pel count is level with M10's.
- Our Fast is close to M10 in these counts and below it in instructions (42.0 / 54.4 M against
  51.3 / 79.1 M per P frame).

### 3.4 SVT's mechanisms and our counterparts

From the helper's trace of v4.3.0 (`emc` = `Source/Lib/Codec/enc_mode_config.c`); "checked"
means re-read here.

| SVT mechanism (rtc, M10; M12 where it differs) | Citation | Ours | State in AV1 Default |
|---|---|---|---|
| Static 64x64 ME bypass (zero-MV SAD < 1 per pixel) and zero-MV exits for HME and the window | emc:840-844 (checked), `motion_estimation.c:1259-1263, 1569-1576, 1784-1795` | `nc-analysis/src/motion.rs`, `analysis.static_sad_per_pixel` 1, `one_position_sad_per_pixel` 1.33, `skip_hme_sad_per_pixel` 2 | On (motion search 2.0 M on lowlight, 3.7 M on crowd). **Off in H.264 Default** (all three 0). |
| Frame-level `skip_intra` when `norm_me_dist <= 5000` | emc:9799-9806 (checked) | Per-block gate `decision.intra_gate_variance` (`decision.rs` `intra_or_inter`) | On: 24 candidates a frame on lowlight. No prize. |
| VLPD0 partition from variance, parent bias, early exit | `product_coding_loop.c:8261-8337, 10448-10533`; emc:7429-7473 | `nc-encoder/src/partition_plan.rs` (same model, plus rectangular shapes and the mean-residual charge) | On. |
| Depth removal from ME costs; M10 level 5, M12 6-7 | emc:10001-10015 (checked), 3109-3421 | `partition_plan.rs` `remove_below_32`, `planner.depth_removal_*` | **Off since 2026-11-15** (thresholds 0); Fast pins 25/50 at every quantizer (§2). |
| 8x8 blocks disallowed from M10 | emc:8365-8375 | `planner.inter_min_log2` 4 | Same (16x16 floor). |
| Light PD1 on non-base frames, regular PD1 on base frames; per-superblock demotion from the reference's superblock (intra, skip, ME distortion and variance against 3x the reference's) | emc:10035-10057 (checked), `enc_dec_process.c:2214-2344` (checked) | Frame roles only: `frame_role.rs`, the `nonbase.*` parameter overlay | No per-superblock complexity class; no use of the previous frame's block decisions. |
| Light-PD1 transform skip score (energy, neighbours skip / NEAREST, RD of skipping) | emc:7872-7893, `product_coding_loop.c:6330-6414` | `nc-encoder/src/pre_tx_skip.rs`, `Av1Backend::skip_luma_before_transform` | On with the zero-residual model (`av1.pre_tx_skip_model 1`, energy 15): 1214 tests a frame. Fast uses SVT's own values. |
| Light-PD1 luma RD skip, chroma energy skip, no chroma without luma | emc:7913-7919, `product_coding_loop.c:6421-6541` | `drop_unprofitable_luma_residual` (only for vectors up to `av1.residual_rd_max_mv`), `mds3::skip_chroma_residual`; `av1.transform_rd_skip` | Luma RD drop on; transform-RD skip and its chroma bypass off (Fast: on). |
| No RDOQ on P frames from M8 (flat IPP); coefficient shaving instead | emc:9717-9739 (checked) | `quant.trellis` 1 on P frames; `mds3::shave_trailing_ones` | **Trellis on**: 10.0 M (12 %) on lowlight, 6.0 M (7 %) on crowd. |
| Transform size search on I frames only (M8-M11), none at M12 | emc:9929-9935 (checked) | `av1.inter_tx_split` 1, `Av1Backend::choose_tx_split`, gates `inter_tx_split_energy` 0.25, `inter_tx_split_concentration` 0 | **On**: 1205 trials a frame on lowlight (8.2 M, 10 %), 789 on crowd (3.6 M). |
| Rate estimation from frame-start tables, no CDF update in mode decision (M9+); linear EOB estimate at M11+ | emc:8694-8702, 9740-9744 (checked) | `CoeffCdfs::cost` walks the live CDFs; `av1.coeff_rate_linear` 0 | Live walks: 1417 a frame on lowlight (4.9 M). |
| Deblocking: off on non-base frames when the reference's skip share is over 75 %; luma level 0 under an ME-distortion floor; M11+ off | emc:1448-1471, 1526-1547 (checked); `deblocking_filter.c:948-1003` | `av1.loop_filter` 1, level from Q | **Always on**: 6235 edge calls a frame on lowlight, 6667 on crowd. |
| CDEF: non-base frames reuse the reference's strength or search {0, reference}; filter blocks with no coded block are skipped | emc:2370-2381 (checked), `md_config_process.c:697-786` | `cdef_search.rs`, `av1.cdef_local_search_period` 1, per-superblock on/off | Adapts: 5.0 M on lowlight against 7.5 M on crowd; cheaper than M10's 8.6 / 12.9 M. |
| Sub-pel skipped when the reference superblock was skip and ME variance is low (light PD1 level 4+, so M12 not M10); early exit under 10 per pixel | emc:7821-7870 (checked) | `decision.codec_subpel_seed_lambdas` 0.03, `decision.seed_refine_ratio` 1.1 | On; our sub-pel calls are level with M10's. |
| Interpolation filter search off when the reference's skip share is over 50 % (M10), always off M11+ | emc:9854-9868 (checked) | No filter search | Not applicable. |
| Partial-frequency transforms gated on block energy | `product_coding_loop.c:9148-9153, 4612-4618` (checked) | `av1.partial_forward` (unconditional on 32/64) | Off in Default, 2 in Fast (§2.2). |

x264 0.165 (`encoder/analyse.c` at the pinned commit, read only): the P-skip probe before
analysis (`x264_macroblock_probe_pskip` when `b_fast_pskip`, lines 3002-3008, and again after
the 16x16 search, 1294-1297), early termination of the sub-partition search on the 16x16 cost
(3060, 3096-3098), intra tried against the inter cost with `b_fast_intra` thresholds (446-457,
696), and RD refinement only within 5/4 of the best SATD (2549-2552). In this fixed-QP run every
macroblock of every P frame still gets the 16x16 search (3600 `mb_analyse_inter_p16x16` calls a
frame on both clips).

### 3.5 The prize on easy content, ranked

Per P frame of AV1 Default on lowlight (84.6 M instructions; crowd 84.4 M). Which codecs each
applies to is from the engine's structure; the savings are the measured inclusive costs, and
what a gated version would keep of them is not measured.

| # | Mechanism we lack | Lowlight | Crowd | Engine-wide? | Known cost of simply switching it off |
|---|---|---:|---:|---|---|
| 1 | No trellis where it cannot pay (SVT: none on P frames) | 10.0 M (12 %) | 6.0 M (7 %) | Yes: the shared `nc_encoder::trellis` (H.264, HEVC, AV1, VP9 and AV2 have trellis entries in `eval/results/levers.jsonl`) | `quant.trellis 0`: CBR L1 J +1.7 %; fixed QP +1.0 Weighted; +11 % fps on Zen 5 |
| 2 | Transform-split trial gated by energy, quantizer or the previous frame (SVT: none on P frames) | 8.2 M (10 %) | 3.6 M (4 %) | Per codec: AV1 here, HEVC has the same lever (`hevc.inter_tu_split`); the others were not checked | `av1.inter_tx_split 0`: CBR L1 J +1.3 %; fixed QP 2.0-10.0 Weighted |
| 3 | Deblocking skipped on low-activity frames (reference skip share, ME distortion) | 4.7 M (6 %) | 5.0 M (6 %) | Yes as a frame-level decision; each codec's filter | `av1.loop_filter 0`: fixed QP 2.7-3.9 Weighted; a content-gated version is unmeasured |
| 4 | Fewer blocks on easy content (M10 codes 549 where we code 1009) | **[inferred]** up to 20 M if block coding and decisions (56.5 M) fell with the block count | - | Yes: shared planner | Depth removal as Fast pins it: fixed QP 5.4-13.1 Weighted; cause of the difference not known (§3.3) |
| 5 | Cheaper whole-sample variance probing (22 000 calls against 4400) | part of mode decision's 18.4 M | 20.5 M | Yes: `decision.rs` `Measurer` | Not measured |
| 6 | Per-superblock complexity class from the reference's superblock (SVT's light-PD1 detector) to pick the cheap path per superblock rather than per preset | not measurable from profiles | | Yes: engine | Not built |

Items 1-3 are 22.9 M of 84.6 M (27 %) on lowlight and 14.6 M (17 %) on crowd.

## 4. Not done, and limits

- The many-clip survey of §3 (stopped by the third brief; `survey.py` and its partial QP table
  are in the work directory).
- J for the ablation and the candidates (Weighted only); the candidates need the CBR screens
  before any preset change.
- The Jetson: nothing here was timed on it.
- The gains in §1.5 and the gated versions in §2.5 and §3.5 are inferences; each names its test.
- The c8a timing covers 4 of the lens's 12 sequences and 2 of its 6 quantizers.
- The morning's `ours-av1@avx2tier` width counters in `probe-zen5` are identical to the
  AVX-512 tier's and so were not taken with the tier forced; the new session's are correct
  (0.04 G 512-bit uops).
