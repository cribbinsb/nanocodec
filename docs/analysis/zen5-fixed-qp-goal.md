# Goal: back ahead on the Fixed QP chart on Zen 5 (living record)

Owner's goal (Mark, 2026-11-16): get our points back ahead on the results site's **Fixed QP**
chart. That chart shows BD-rate against x264 medium at fixed QP, AOM CTC LD, 12 sequences. Its
speed is the chart timing. The gap is **Zen 5 only** (x86 timing on AWS c8a.large). On the Zen 3
PC and on the Jetson we are not behind.

Read this first. Update it whenever something is established or ruled out. Record facts only, each
with its source.

## Established facts

1. **The gap on the chart.** On Zen 5 at fixed QP, SVT-AV1 M10, M12 and M13 beat every point of
   ours between about 185 and 445 fps. Weighted BD %, fps:
   - SVT: M13 +4.1 at 445, M12 −1.5 at 409, M10 −14.2 at 229.
   - Ours, dominated: HEVC Fast +7.9 at 316, AV1 Fast +1.6 at 290, H.264 Default +3.9 at 216,
     AV2 Fast −11.7 at 190, VVC Fast −6.9 at 185.
   - Ours, still on the frontier: HEVC −14.9 at 172, AV1 −17.9 at 158, VP9 −18.6 at 144, AV2 −25.2
     at 134.
   - Source: `eval/nceval/site.py` `fixed_qp_view()` and `docs/results/fixed-qp-lens-2026-11-15.csv`.
2. **It is Zen 5 scaling.** The bisection is in `docs/handoff/avx512.md`, "Bisection on one build".
   Going from Zen 3 to Zen 5:
   - SVT M10 gets 1.95× faster; ours 1.68×; x264 1.63×. With AVX2 only: SVT 1.75×, ours 1.55×.
   - On the Zen 3 PC, M10 and our AV1 Default measure level (106.6 against 103.8 fps, commit
     b2057e5f4).
3. **Mechanism, from the same bisection.**
   - IPC on Zen 5 is equal: ours 3.29, M10 3.24.
   - About 44% of M10's instructions are vector uops, mostly 256/512-bit. Ours are about 27%,
     many of them 128-bit.
   - About 55% of our Zen 5 cycles are outside the SIMD kernels.
   - Zen 5's gains are largest for vector work.
4. **Our cost hardly varies with content; SVT's does.** Chart timing, 3 clips, `ms_per_frame` in
   `docs/results/bd-vs-speed-timing-x86.json`: the max/min ratio across clips is 1.14 for our AV1
   Default and 1.93 for SVT M10. This is 3 clips only and is not yet confirmed on more.
5. **Our Fast rungs lose much more at fixed QP than in CBR.** AV1 goes −17.9 → +1.6 Weighted
   (Default → Fast) at fixed QP, against −46.2 → −38.2 J in CBR. HEVC goes −14.9 → +7.9. The
   cause is not known yet.

6. **Diagnosis landed** (`docs/analysis/content-adaptive-cost-2026-11-16.md`, aa88b7518).
   - **Instruction mix:** we retire about as many vector uops as M10, plus 1.7–1.8× its non-vector
     instructions. Widening candidates add up to about 8–12 % of Zen 5 time: the av1_edge deblock,
     the scalar pixel and statistics loops outside nc-dsp, and small-block variance (22,000 calls a
     frame against M10's 4,400).
   - **The Fast fixed-QP loss is three AV1 levers together:** `partial_forward 2`, depth removal at
     every quantizer and transform split off. They cost most at low qindex. The fixed-QP profile
     pin explains only 2.8 of the 27.7 points.
   - **Timed on c8a, 4 sequences:**
     - AV1 Fast + Default depth + full forward: 250 fps, −6.5. This is on the frontier, between
       M12 (356 fps, −1.5) and M10 (216 fps, −14.2).
     - HEVC Fast + depth: from +7.9 to −1.3 for 15 % of its speed.
   - **AV1 coefficient writer and range coder:** 1540M instructions against M10's 761M for the
     same bytes.

7. **The chart's speed axis is now timed on the lens's own encodes** (2026-11-16,
   [fixed-qp-speed-2026-11-16.md](../results/fixed-qp-speed-2026-11-16.md), build b11b2f83).
   - Facts 1 and 4 used the CBR chart's timing on 3 private clips. The axis is now the same 12 CTC sequences and 6
     quantizers the BD comes from, on c8a.large.
   - Weighted, fps: M13 +4.1 at 984; AV1 Faster +2.1 at 901; M12 -1.5 at 899; AV1 Fast -11.0 at 524; M10 -14.2 at
     519; AV2 -25.2 at 306; AV1 -18.9 at 304; M8 -30.3 at 169.
   - AV1 Faster and AV1 Fast now run level with M12 and M10. They are 3.6 and 3.2 Weighted points behind them, and on
     J level or ahead (-13.5 against -10.4; -19.6 against -19.5).
   - A second view on 12 held-out clips (J only) puts HEVC Faster ahead of M12 and AV1 Faster, HEVC Fast and AV1 Fast
     ahead of M10. Nothing of ours is ahead of M13 in any view on x86.

8. **Chart refresh at 2860f897** (2026-11-17, [fixed-qp-speed-2026-11-16.md](../results/fixed-qp-speed-2026-11-16.md)
   "Update 2026-11-17"). AV1 Fast, AV1, VP9, VVC and HEVC code the same lens streams as b11b2f83 and were only
   re-timed; AV1 Faster (base-only) changed; AV1 Veryfast is new. Weighted / J, c8a fps, Jetson fps (SVT stored;
   the same-session check read 0.4-1.0 % faster on x86):

   | Point | Weighted | J | x86 fps | Jetson fps |
   |---|---:|---:|---:|---:|
   | SVT-AV1 M13 | +4.1 | -4.8 | 984.1 | 196.1 |
   | AV1 Veryfast | +7.8 | -6.2 | 1078.3 | 255.7 |
   | SVT-AV1 M12 | -1.5 | -10.4 | 899.0 | 178.1 |
   | AV1 Faster | +2.1 → -5.7 | -13.5 → -18.3 | 900.8 → 786.4 | 222.0 → 203.4 |
   | AV1 Fast | -11.0 | -19.6 | 523.6 → 526.5 | 142.8 → 143.7 |
   | SVT-AV1 M10 | -14.2 | -19.5 | 518.9 | 99.0 |
   | AV1 | -18.9 | -26.7 | 303.7 → 304.9 | 90.3 → 90.7 |

   - **x86, J:** AV1 Veryfast is ahead of M13 (first point of ours ahead of M13 on Zen 5). AV1 Faster is 7.9 J
     points ahead of M12 but 12.5 % slower, so no longer level with it on speed. AV1 Fast stays level with M10.
   - **x86, Weighted:** still nothing of ours ahead of M10, M12 or M13. Veryfast is 3.7 points behind M13 at
     +9.6 % speed; Faster is 4.2 points ahead of M12 at -12.5 % speed.
   - **Jetson:** AV1 Faster is ahead of M12 and M13 in Weighted and J; Veryfast is ahead of M13 in J.
   - **CBR held-out charts** (AV1 Faster and Veryfast re-measured on both clip sets; vs x264 medium / vs x264
     medium --tune psnr; CBR chart timing, c8a): fresh, AV1 Faster -38.8 → -41.7 / -31.1 → -34.2 at
     346.7 → 327.4 fps; AV1 Veryfast -35.3 / -26.9 at 447.1 fps against M13 -27.5 / -18.2 at 445.4 (same session
     1.017x M13's time). All held-out: Faster -36.0 → -38.3 / -27.8 → -30.2; Veryfast -32.0 / -23.1 against M13
     -20.6 / -10.2.
   - **AV1 Veryfast base-only** (2caf7d21d, re-measured at d40f86b3, same method; results doc "Update 2026-11-17
     (2)"): lens +7.8 / -6.2 → +3.2 / -10.3 (unrounded +3.25 / -10.35); 1078.3 → 1055.0 fps on c8a (this
     session's M13 1006.8, stored 984.1), 255.7 → 244.9 on the Jetson. **Ahead of M13 in Weighted and J on both
     machines**; faster than M12 on both, level with it on J (-10.35 against -10.36), 4.8 Weighted points behind
     it. CBR held-out: fresh -37.4 / -29.2 (vs x264 medium / tuned psnr), all -34.1 / -25.6, at 410.7 c8a
     (M13 445.4, M12 408.7) and 87.7 Jetson (M13 79.2): ahead of M12 on both machines, and of M13 on the Jetson.
     On c8a in CBR it is 7.8 % slower than M13.

## Ruled out (measured; do not repeat)

All of these were measured on 2026-11-16 and are recorded in `docs/handoff/avx512.md`.
- **AVX-512 alone as the cause:** it explains about 3 points of the scaling gap.
- **AV1 trellis and coefficient-cost reductions:**
  - trellis off: +1.72% J;
  - an end-of-block gate: +1.19%;
  - SVT's greedy limits, the end-of-block window and the 60% cut-off: +1.24%;
  - a linear rate for the transform-split trial: +0.31% to +0.43%;
  - precomputing the context walk: exact but slower on c8a;
  - inner-loop micro-edits: more instructions.
- **Frame-level loop-filter gate on AV1 Default (`filter.*`, 7c2b840cf) as a large speed lever**
  (`feature/light-pipeline`, c8a session 2, 4 sequences x 2 QPs; box lens): deblocking off on
  non-base frames after a 75 %-skipped reference: 153 fps (Default 152), Weighted -17.7 / J -26.1
  (Default -17.9 / -26.0); frame proxy 0.05 (deblocking off and CDEF reuse on well-predicted
  frames): 153 fps, -17.5 / -25.5; proxy 0.5 (fires on almost every frame): 159 fps (+5 %),
  -15.1 / -24.2. Deblocking and CDEF control are not where Default's time goes on these clips;
  the gate may still pay on a Fast base with deblocking on (being measured).
- **Per-filter-block CDEF skip on static content** (`feature/av1-lowlight`, 2026-11-16; patch kept
  at `~/nanocodec-data/work/av1-lowlight/cdef-static-rejected.patch`). A 64x64 block whose
  zero-vector SAD against the previous source, per sample, is under `t` quantizer steps is left
  out of CDEF (no direction search, no filtering; a third strength set, all off). AV1 Faster,
  chart QPs, 150 f, one key (bytes / PSNR-YUV against `t` = 0); lowlight callgrind per P frame
  (30 f):

  | t | lowlight | crowd | bodycam | Ir per P frame |
  |---|---|---|---|---:|
  | 0 | 374884 / 33.269 | 347060 / 31.203 | 384485 / 33.092 | 29.61M |
  | 0.02 | +1.0 % / -0.00 dB | +0.9 % / -0.15 dB | +0.7 % / -0.02 dB | 28.81M (-2.7 %) |
  | 0.05 | +1.9 % / -0.05 dB | +1.2 % / -0.18 dB | +0.8 % / -0.03 dB | 28.23M (-4.7 %) |
  | 0.1 | +3.2 % / -0.12 dB | +1.7 % / -0.27 dB | +0.8 % / -0.09 dB | 27.96M (-5.6 %) |

  Static blocks are where CDEF holds back the coding noise of residual-coded static areas
  (`av1.cdef_per_superblock`'s own evidence), so leaving them unfiltered costs bits and quality
  out of proportion to the time (2-5 % rate for 3-5 % of lowlight's P-frame instructions, more
  than the M12-M10 frontier slope). Not landed. SVT's own skip (`me_based_cdef_skip`) is per
  frame and also requires the references' CDEF gain to have been small.
- **VVC key-frame bundle:** passed the pool screen but failed the 10 s key pulse guard on static
  clips. The cause is rate control's second-key budget (`feature/vvc-speed` report).
- **HEVC Fast re-pointed to the Default's depth settings (parameters only)**
  (`feature/hevc-h264-speed`, build at c770b3c42's code, 2026-11-16):
  - Lens (12 AOM CTC sequences, current build): Fast +7.9 -> -1.3 Weighted (reproduces §2.3 of
    the diagnosis). Gating depth removal by quantizer instead (Fast's 25/50 thresholds with
    `planner.depth_removal_min_quantizer` 0.84 / 0.78 / 0.70) gives -1.3 / -0.6 / +1.6.
  - CBR pool screen against Fast (2026-11-16 draw, k 0.6, dCPU +2.7% passed in): L1 J -0.80%,
    L2 +0.59%, L3 (78 clips, ladder) **+0.43% [+0.03, +0.83] WORSE**; quality J full -1.8%,
    low -0.6% (lowlight -5.6% / -3.8%, the other classes -1.7% to +1.0%), no guard.
  - The dCPU passed in was low: x86 callgrind at CBR 600 kbit/s, 90 f, is +2.2% crowd09,
    +4.7% bodycam_bc1, +9.1% lowlight_city_night (mean +5.3%); the 0.84 quantizer gate is
    +0.2% / +1.9% / +7.2% (lowlight codes at fine quantizers there).
  - So neither the plain re-point nor a quantizer-only gate pays for its CPU in CBR; the
    shared energy gates (AV1/AV2 agent) are the route.
- **HEVC Fast with light superblocks (light pipeline, 3df562c10) and Faster-based points**
  (`feature/hevc-h264-speed`, 2026-11-16; box lens via `nceval.ctc local`, Weighted / J against
  x264 medium; x86 callgrind for CPU; CBR pool screens against Fast, k 0.6):
  - Regular superblocks with the Default's depth (thresholds 0) and transform tree, light ones with
    Fast's (25/50, no tree), by `effort.regular_fraction`: 1 (all regular) -3.7 / -12.4, 0.75
    -3.4, 0.5 -2.2, 0.25 +0.9; Fast +7.9 / -4.2. Fixed-QP instructions against Fast (crowd09 and
    lowlight_city_night, QP 32 / 42, 30 f): +18 / +10-11 %, +17 / +10-11 %, +14 / +9-10 %, +8-9 /
    +5-6 %. The light split interpolates; it does not beat the line between Fast and all-regular.
  - CBR, 0.25: L3 J **+0.59 % [+0.14, +1.04] WORSE** (quality full -2.5 %, low -1.3 %; CPU at
    600 kbit/s +4.1 %).
  - Faster base with the Default's depth and transform tree everywhere: lens **+1.7 / -6.6**
    (Faster alone +17.4 / +4.8) at fewer instructions than Fast (-7 to -11 % at QP 37; -12 % at
    600 kbit/s CBR). Faster base with light shares 0.5 / 0.25: +3.6 / +7.6. In CBR the same point
    is **quality +7.4 % full / +9.4 % low against Fast, L3 J +1.06 % WORSE, every class over the
    quality guard and the key-pulse guard**: the fixed-QP lens and the CBR pool disagree by
    about 10 points on this change (Faster's other levers, not depth or the tree, are what CBR
    punishes; not yet split).

## In progress (who)

- **Diagnosis (Fable agent, `feature/adaptive-diag`):**
  - (a) which of our hot work is scalar or 128-bit where SVT's is 256/512-bit vector, with the
    expected gain on Zen 5;
  - (b) why our Fast rungs lose so much efficiency at fixed QP.
  - Deliverable: `docs/analysis/content-adaptive-cost-2026-11-16.md`.
  - SVT M10/M12 rtc skip trace (code-level, cited): the final message of helper task
    a927dfc1e03ab9f90.
- **Kernel sharing (`feature/kernel-share`):** merged kernels get 256/512-bit widest tiers; it
  lists any hot kernels still on 128-bit lanes.
- **Astra:** AVX-512 tier for the shared deblock family.
- **Exact speed-ups, identity-only landing** (each moves our points right):
  - engine and VP9: `feature/zen5-codecs`;
  - HEVC and H.264: `feature/hevc-h264-speed`;
  - AV1 and AV2: `feature/av1-av2-speed`;
  - polish: `feature/polish`.

## Progress (exact, streams identical)

- 2026-11-17, `feature/av1-faster-exact`: the AVX-512 tier writes picture borders with ordinary
  stores, not non-temporal ones (c9e59feed), and AV1 CDEF uses the backend's `Dsp` (4d317fa52).
  AV1 Faster fixed-QP chart timing on c8a, 810.6 -> 845.2 fps (+4.3 %), M12 903.4 in the same
  session. Section "AV1 Faster exact speed-ups" below.

- 2026-11-16, `feature/av1-lowlight`: the AV1 key-frame CfL slope search (on in the Default
  since 79d53ba8b) runs on a shared nc-dsp CfL prediction-SSE kernel (`Dsp::cfl_sse`; AVX2,
  NEON, SIMD128 on the shared 16-bit lanes; checkasm on every tier). Lowlight Default key frame
  2072M -> 1922M Ir (-7.2 %). Jetson (chart builds, crowd 720p): all-key Default cycles
  -5.15 %, Default CBR 600 31 f cycles -1.74 %. Handoff: `docs/handoff/av1-lowlight.md`.

- 2026-11-16, `feature/av1-lowlight`: AV1 CDEF finds its units from a coded-unit map the block
  writer fills (the decoder's `CdefMap`, now shared in `nc-av1/src/cdef_map.rs`) instead of
  scanning the mode-info grid, and frames that reuse their per-block choices filter through the
  decoder's band driver (one pass per unit). Faster on lowlight_city_night at qindex 184
  (callgrind, 30 f): CDEF 3.46M -> 2.78M per P frame, whole P frame 29.61M -> 28.96M (-2.2 %);
  key frame +0.9M (525.0M -> 525.9M: the map is filled block by block).

- 2026-11-16 to 2409144f3: kernel-sharing merges cut whole-program instructions (callgrind, crowd
  720p, 30 f, release) by HEVC −6.25 %, VVC −3.93 %, H.264 −0.05 %, VP9 +0.02 %, AV2 +0.05 %,
  AV1 +0.12 %. Exact engine/codec cuts: VP9 about −4 to −5 % (end-of-block tests, quantiser
  setup), AV1/AV2 −0.5 to −0.7 % (CDEF search, intra edges), H.264/HEVC intra cleanups. Not yet
  re-timed on Zen 5.

## HEVC Fast re-point (feature/hevc-h264-speed, landed 6c208433e, 2026-11-16)

- **What:** the Default's depth removal (thresholds 0) and inter transform tree, plus Faster's
  CBR-harmless settings (inter rounding 85, SAO chroma off, partial forward with zero scales
  3 / 4). A pool L1 split of Faster's groups against Fast (quality J): rounding -0.52 %, SAO
  chroma +0.06 %, partial forward +2.65 % (for -6 % CPU); left out motion +7.36 %, max
  transform 16 +2.06 %, planner pair +2.00 %, coarse key intra +1.18 %.
- **CBR pool screen against the previous Fast** (k 0.6, x86 dCPU +1.1 % at 600 kbit/s): L3 J
  -1.31 % [-2.17, -0.48] BETTER, no guard.
- **final-check** (600 f, keys every 10 s, 28 long clips): J -2.93 % [-5.36, -0.53]; the long
  periodic-key pulse guard is at its limit on one clip (pool_external_bdd_night_new_720p_360p
  +0.504 dB against +0.50; balanced -0.22 dB, crowd -0.99 dB), so the tool says WORSE. Kept on
  CLAUDE.md's guard rule (a guard missed by hundredths while the score improves).
  Looked at (old and new Fast, CBR 212 kbit/s = the clip's rung 2, 600 f, keys every 10 s):
  every frame around the key at 300 is better with the new Fast (P frames 296-299 +1.2 to
  +1.7 dB, the key +0.65 dB, frames 301-304 +0.9 to +1.5 dB); the key now sits 0.54 dB below
  its predecessors where it sat 0.24 dB above, because the P frames gained more than the key.
  The key and its predecessor show no visible artefact in either build.
- **Charts (refresh at 6c208433e; Zen 5 c8a timing):** CBR chart J against x264 medium -44.16 %
  at 315.5 fps -> **-43.61 % at 308.9 fps**. Fixed-QP lens (box, 12 AOM CTC sequences)
  Weighted / J against x264 medium +7.9 / -4.2 -> **-3.1 / -11.9**, at the same chart speed.
- **Against the frontier, Zen 5 fps:**

| point | fixed QP Weighted | CBR J | fps | line between neighbours at that fps |
|---|---:|---:|---:|---|
| HEVC Default | -14.9 | -50.98 | 171.7 | |
| **HEVC Fast (new)** | **-3.1** | **-43.61** | **308.9** | HEVC Default-Faster: fixed QP +5.3, CBR -42.5 (above both); SVT M10-M12: fixed QP -8.6 (below), CBR -38.5 (above) |
| HEVC Fast (old) | +7.9 | -44.16 | 315.5 | HEVC Default-Faster: fixed QP +6.3 (below), CBR -42.1 (above) |
| HEVC Faster | +17.4 | -37.45 | 390.8 | |
| SVT-AV1 M10 | -14.2 | -41.30 | 228.9 | |
| SVT-AV1 M12 | -1.5 | -34.91 | 408.7 | |

  So at fixed QP the new Fast moves from below our own Default-Faster line to well above it,
  but stays under the SVT M10-M12 line (5.5 points short at its speed); in CBR it keeps its
  place above both lines, 0.55 points lower than before at 2 % less speed.

### HEVC Fast against SVT M12: stages and the light pipeline (2026-11-16)

- **Per P frame, box callgrind, 60 f, HEVC Fast QP 37 / M12 `--rtc` qp 46:** Johnny 33.8 M against
  26.1 M (1.29x), WestWindEasy 96.7 M against 73.4 M (1.32x). Ours: mode decision 21 %, block
  coding 26-38 % (of which the inter transform tree 12-18 %), planner 6-12 %, CABAC writing
  3-9 %. M12 codes every superblock with light PD1 (regular PD1 0 instructions); its CDEF is
  7-18 % and entropy coding 3-13 %. Script: `work/hevc-h264-speed/stages.py`.
- **The trade the SVT M10-M12 line allows at 309 fps:** about 0.22 Weighted points per 1 % of
  speed.
- **Light superblocks on the new Fast** (light = no transform tree and Fast's old 25 / 50 depth
  removal, optionally Faster's motion values; lens Weighted against x264 medium, x86 instructions
  against the new Fast at QP 32 / 42, crowd09 and lowlight):
  - regular share 0.5: -1.2 at -2.8 %;
  - regular share 0.25: +1.9 at -6.9 %;
  - with Faster's motion values, share 0.5: -0.5 at -4.5 %;
  - with Faster's motion values, share 0: +14.6 at -18.4 %.
  - So every light trade costs 0.58-0.96 points per 1 % of speed, three to four times the line's
    rate: they move the point away from the line, not onto it.
- **The other direction, one Default group added to the new Fast** (lens; instructions at QP 37,
  30 f, key frame included):
  - merge candidates: -4.3 at +3.9 % (0.31 points per 1 %);
  - structure group: -4.3 at +4.3 % (0.28);
  - motion levers: -6.3 at +18 % (0.17);
  - trellis: -4.5 at +31 % (0.05);
  - intra bundle: -6.0 at +41 % (0.07).
  - Only merge and structure beat the line's rate. Both together would be about -5.5 at about
    286 fps, against a line at -10.2 there: the gap narrows from 5.5 to about 4.7 points and does
    not close.

## AV1 Fast re-point and quantizer gates (feature/av1-av2-speed, 2026-11-16)

- **Gates landed** (6d4d2c636; streams identical at the defaults).
  - Types `nc_encoder::{GateInput, GateParams, WorkGate}`. A cheap path is taken from a relative
    quantizer up, or below it where the error per sample is under `max_energy` squared steps.
    The energy half has no registered parameter yet.
  - Registered parameters: `gate.partial_transform_min_quantizer` and
    `gate.tx_split_skip_min_quantizer`.
  - In AV1 the gates swap with the light-superblock settings.
- **Fixed-QP lens** (12 AOM CTC sequences, Weighted vs x264 medium, box runs;
  `~/nanocodec-data/work/av1-av2-speed/lens`). "Re-pointed Fast" (RF) is Fast with the
  Default's depth and full forward.

  | Point | Weighted |
  |---|---:|
  | Fast | +1.6 |
  | RF | −6.5 |
  | Depth and partial forward gated at relative quantizer 0.78 | −6.2 |
  | The same at 0.86 | −6.6 |
  | RF + deblocking | −9.2 |
  | RF + split trial (gate 0.7, `av1.inter_tx_split_energy` 1) | −7.7 |
  | RF + deblocking + that split | −10.4 |

- **c8a timing** (one core, median of 3 rounds after a warm-up):
  - Fixed QP: four CTC sequences, qindex 160 and 210.
  - CBR: 600 kbps on crowd and lowlight.

  | Point | fps fixed QP | fps CBR 600 |
  |---|---:|---:|
  | Fast | 311 | 326 |
  | RF | 255–256 | 297–298 |
  | Gates 0.78 | 262 | 306 |
  | RF + deblocking | 242 | 278 |
  | RF + deblocking + split | 231 | 274 |
  | Default | 153 | 180 |
  | SVT M12 | 355–362 | — |
  | SVT M10 | 216–219 | — |

- **CBR pool screens** (draw 2026-11-16):
  - RF vs Fast: L3 J −3.95 % [−4.59, −3.25] BETTER.
  - Gates 0.78 vs RF: +0.30 % [−0.01, +0.60], equal.
  - RF + split at 0.7 vs Fast: J −3.97 % BETTER, but two guards missed: rate overshoot on 5
    rungs and flicker +0.07.
  - RF + deblocking + split vs RF: J −2.99 %, WORSE on the periodic-key pulse (+1.00 dB) and on
    the ceiling (+1.69 % over target).
- **RF landed, was reverted, and was re-landed on judgement.** It first landed as 5e21b7a80
  and was reverted by ab5c67dff.
  - The bundle final-check (600 frames, keys every 10 s) gave J −6.37 % [−7.51, −5.17], but
    WORSE on the periodic-key pulse: worst clip +1.51 dB, all static clips (otw_homes_00004,
    chad_4_109, meva_school330, london_bus).
  - Measured on pool_otw_homes_00004 (600 frames, key every 300, PSNR-Y, old → new):

    | Rate | Second key | P frames before it |
    |---|---:|---:|
    | 300 kbps | 31.25 → 31.14 | 36.99 → 37.27 |
    | 600 kbps | 34.69 → 34.71 | 38.83 → 39.75 |

    The keys are unchanged and the P frames improve, so the dip deepens; no frame gets worse.
  - Neither key-side variant clears the guard (J, worst-clip pulse):
    - the Default key intra set: −6.79 %, +1.10 dB;
    - `rate.key_still_boost` 2: −6.40 %, +3.38 dB;
    - boost 3: −6.74 %, +2.30 dB;
    - boost 6: −7.35 %, +4.56 dB.

    With the boost the worst clip becomes pool_meva_hospital_g301. Why it blows up there is
    not known.
  - Re-landed on judgement (coordinator): no frame gets worse, the deeper dip is the better P
    frames, and the Default already has these settings.
- **Ruled out after the re-land:**
  - Gates at 0.78 on the new Fast: final-check J +1.10 % [+0.81, +1.40], WORSE, with
    `bd_worst_vmaf_v1` +3.47. The 150-frame screen had called it equal.
  - The energy half of the partial-transform gate: the partial forward transform where the
    residual is under e squared steps per sample, wired per transform so a trial and its coding
    decide alike. On the lens against the new Fast, at 0.99× box time:

    | e | Weighted vs new Fast | vs x264 |
    |---|---:|---:|
    | 0.25 | +1.4 % | −5.1 |
    | 0.5 | +3.3 % | −3.2 |
    | 1 | +5.1 % | −2.5 |
    | 2 | +6.0 % | −1.1 |

    BlueSky and SnowMountain lose 13–35 %: smooth blocks with weak residuals still need their
    high frequencies, and the approximation saves almost no time there. Not landed. Patch:
    `~/nanocodec-data/work/av1-av2-speed/energy-gate-rejected.patch`.
- **Still-scene key dip (rate control, `feature/rc-keydip`, 2026-11-16).** Traces: AV1 Fast, 5
  static clips × 300/600 kbps, 600 f, key every 300, rate logs + per-frame PSNR-Y
  (`~/nanocodec-data/work/rc-keydip`).
  - **Why P frames out-climb the key.** In 8 of 10 traces the strict buffer is empty (0.00)
    before key 2, and the key fills it to 0.87–1.00 (0.94–1.15 × capacity). On still content P
    frames cost a few hundred bytes, so the level walks down 60–80 qindex over the period
    (homes 300: 212 → 130; base frames to q82), and P PSNR-Y climbs about 6 dB (31 → 37). The key
    is bounded by the buffer (homes 300: q163, 31.1 dB), so a finer key cannot be bought.
    london_bus is the exception: its key uses 0.5–0.7 of the buffer.
  - **The starved key (f-c3).** Homes 300: a q54 cadence base frame 4 frames before the key cost
    32 kB (13 kB predicted), took the buffer to 0.79, and the key coded q235, 25.0 dB. AV1 had no
    `rate.prekey_quiet`.
  - **Base-frame bits on still scenes.** Their cost is set by how far their quantizer is finer
    than the quality the picture holds. On chirla, 100+ kbit frames were predicted at
    0.02–0.26×. `rate.refine_cost` (registered, off) prices them at 0.8–1.5×, but using it raised
    the stream's rate 1–4%. The cause is not known yet.
  - **Ruled out: the key-dip bound** (`rate.key_dip`: hold P PSNR-Y within N dB of the expected
    key). Final-check against old Fast at 4 dB: J −6.47%, worst-clip pulse +1.02 dB (chad_4_109),
    hospital 300 kbps coded 37% under target. Deleted.
  - **Landed: the measured quiet run-up.** `rate.prekey_quiet` now scales by how the inter
    frames' running PSNR-Y stands against the last key: whole from equal quality up, off once the
    key stands 2 dB above. The plain switch lifted keys that already stood above their P frames
    (bdd_night_new 300 kbps pulse +2.87 → +4.05 dB; unchanged with the measured form). A ramp off
    from equal quality tripped HEVC Default's pulse guard on mot17_04 (+0.93 dB), so it is centred
    on the key standing above. AV1 Fast takes it.
    - Against the re-pointed Fast: J −0.67% [−0.90, −0.45] BETTER, no guard, pulse −0.24 dB
      (static −0.59).
    - Against the Fast before the re-point (an earlier form, with the quiet whole wherever P
      stood 1 dB above the key): J −7.22% [−8.35, −6.05], worst-clip pulse +1.19 dB (chad_4_109;
      was +1.51), long-window +0.69. That still misses the +0.50 guard. On chad_4_109 the re-point
      lifts P frames by 1.3–1.6 dB, the key is unchanged, and no rate-control lever tried closes
      the gap without giving that back.
    - H.264 and HEVC Default (they already ran the quiet second; same draw): H.264 J +0.04%
      [−0.04, +0.13], no guard. HEVC J +0.08% [−0.04, +0.21], pulse −0.26 dB balanced (lowlight
      −0.92), but one clip, crowd mot17_04 at 2 s keys, +0.74 dB. Landed on judgement: the
      balanced pulse improves, J is level, and on mot17_04 the keys stand 1–4 dB above their
      P frames, so less quiet lowers the high-rate pulses (+3.5 → +2.1 dB at 1200 kbps) while
      the low-rate ones swing around 0.
    - f-c3 on homes 300: key 25.0 → 30.4 dB.
    - **Engine default since the next landing** (shared first). Final-checks against 86d268d54,
      no guard in any: AV1 Default J −0.13% [−0.42, +0.16], VP9 −0.14% [−0.38, +0.07], VVC
      −0.24% [−0.49, −0.02], AV2 −0.32% [−0.57, −0.10], HEVC Fast (unpinned) −0.47% [−0.84,
      −0.08]; pulse −0.16 to −0.35 dB balanced. The fast rungs of VP9, VVC and AV2 and AV1's
      layered fast rungs pin it off (AV1 Faster took it with f-c3n); H.264's and HEVC's
      Faster rungs keep their pins.
  - **Why `rate.refine_cost` raises the rate (chirla, gated split):** its floor keeps a burst
    inside the strict room (frame 80: 25 kB → 3.8 kB), so the base-role model no longer learns
    burst sizes. The level then stays 50–90 qindex finer (frames 88/96: 207/251 → 140/157), and
    the next burst is larger (frame 104: 2.5 → 40.6 kB, buffer to 1.00). Next: plan the level
    with the predicted refinement price, not with the role model alone.
    - t33 (light-pipeline binary, plain switch): homes 600 pulse delta against old Fast +3.88 →
      +0.38 dB, london_bus 600 +2.00 → +0.70, hospital 600 +2.74 → +1.57.

## AV1 Fast: the Default's cheap quality groups (feature/light-pipeline, 2026-11-16)

Lever-group ablation of Fast against the Default (each group of Fast's values put back to
the Default's, box fixed-QP lens and c8a fixed-QP timing; docs/handoff/light-pipeline.md).
Five groups were above the Fast -> Default line: CDEF settings, sub-pel move biases,
cheap-vector and seed refinement, pre-transform skip energy/model/RD, deblocking. Together
on the landed Fast (ca8429a43, with rate.prekey_quiet from 86d268d54):

| point | c8a fps (4 CTC seq.) | chart-path time vs Fast | lens Weighted / J |
|---|---:|---:|---:|
| previous Fast | 249 | 1.00 | -6.5 / -17.1 |
| new Fast | 226 | 1.104 | -11.0 / -19.7 |
| SVT M10 | 215 | 1.175 | -14.2 / -19.5 |

- New Fast beats M10 on fixed-QP J at higher speed; M10 keeps 3.2 Weighted points.
- CBR quality vs the previous Fast (draw 2026-11-16, dCPU 0): L3 -3.31% [-4.15, -2.31],
  n=78, against a Fast -> Default line of about -2.0% at x1.104 (Default -11.98% at x1.61).
- Jetson (Orin Nano, one core, jperf min of 3 under jetson-timed, 150 f CBR 600 kbit/s,
  45ba872af -> 011bfa1cc): cycles crowd +6.84%, bodycam +2.22%, lowlight +7.71%; sum
  12.07 -> 12.70 G, x1.052 (instructions x1.061). x86 chart path x1.104.
- Final-check (600 f, keys every 10 s): J -3.25% [-4.02, -2.53], no guards. Without the
  prekey_quiet fix the same bundle failed the pulse guard (+0.84 dB); without deblocking the
  static class failed (+3.7%).

## AV1 Default: three of Slow's cheap groups (feature/light-pipeline, 2026-11-16)

Group ablation of Default against Slow: only exact_rank_margin 0, cdef_search_row_step 1
and keyframe_cfl 1 were above the Default -> Slow line; landed in AV1 DEFAULT_PARAMS. Lens
-17.9 / -26.0 -> -18.9 / -26.7 at +2.4 % c8a time; CBR L1 quality -0.80 %; final-check
-0.30 % [-0.49, -0.11], no guards. Default with trellis off (+8 % speed) or also Fast's key
intra set (+15 %) lose in CBR (L1 +1.14 % / +3.93 %): no Default re-mix reaches M10's
speed at its Weighted. exact_rank_margin 0 engine-wide, L1 quality: H.264, HEVC and VP9
+0.00 % (already 0 or inert), VVC -0.17 % [-0.61, +0.20], AV2 -0.53 % [-0.77, -0.03].

- **Engine default 0 (feature/rank-margin, 2026-11-16).** Quality-only levels (dCPU 0,
  draw 2026-11-16): VVC L3 -0.43 % [-0.74, -0.15] n 78 BETTER; AV2 L3 +0.12 % [-0.19,
  +0.46] AMBIGUOUS (ceiling guard 0.65 % over target against 0.35 % + 0.25). Final-check
  (600 f, keys every 10 s, ladder, n 28): VVC -0.28 % [-0.51, -0.09], periodic-key pulse
  guard +0.68 dB on one clip (pool_external_meva_admin_720p_536p, the clip HEVC's merge
  attempt also failed on; balanced dKeyN +0.00 dB); AV2 -0.19 % [-0.48, +0.05], no guards.
  Adopted as the shared default (equal or better on both; shared-first). VP9 is inert: it
  has no `luma_filter`, so it never reaches the exact ranking (identity confirms). Preset
  identity: only VVC and AV2 Default change; VVC and AV2 Slow pin 1.1 (not screened at 0).
- **Jetson, AV1 Default 79d53ba8b against its parent** (jperf, core 3, min of 3, CBR 600,
  150 f, crowd / bodycam / static 720p): cycles +4.53 / +4.96 / +5.04 %, geometric mean
  **x1.048**; instructions +6.86 / +6.70 / +5.54 % (x1.064). x86 c8a time was +2.4 %.

## Assigned 2026-11-16 09:10

- **`feature/av1-av2-speed`:** re-point AV1 Fast (depth + full forward), then gate the three
  levers by quantizer or energy, then gated trellis, transform-split trial and deblock skip.
- **`feature/hevc-h264-speed`:** re-point HEVC Fast (depth), then gating.
- **`feature/zen5-codecs`:** small-block variance call count; AV1 coefficient writer and range
  coder (exact).
- **`feature/kernel-share`:** scalar pixel and statistics loops into shared wide kernels.
- **Astra:** av1_edge AVX-512.

- **Bold structural work** (Mark: "be bold … don't get stuck at our local maximum"). Whole
  configurations are judged on the frontier, with no per-change +0.3 % J bar.
  - `feature/light-pipeline`: an SVT-style light decision pipeline in the shared engine. It has a
    partition pre-pass, light mode decision on a fixed partition, and ME static/zero bypasses,
    behind parameters, AV1 first. It owns planner/decision/ME and the variance call count.
  - `feature/av1-writer`: AV1 coefficient writer and range coder, 1540M against M10's 761M
    instructions for the same bytes. Restructure as needed; exact.

- **Design direction (Mark, 2026-11-16):** effort is continuous, not binary per preset.
  - Expensive tools run where per-block or per-superblock gates (quantizer, energy per step²) say
    so.
  - A bounded-work budget, `effort.regular_fraction`, gives the expensive path to at most the worst
    X % of superblocks, ranked by a pre-coding proxy, OR to any superblock above an absolute
    threshold.
  - Presets become points along one tuned curve of continuous knobs.

- **Light pipeline S1/S2 landed** (6b6b965c7, default off).
  - **First AV1 lens sweep:** regular_fraction 0.75 → +0.38 % BD against Default at 0.98× time;
    0.5 → +0.97 % at 0.97×; 0.25 → +3.65 % at 0.90×; 0 → +11.1 % at 0.78×. Fast is +40.6 % at
    0.56×. Times are user CPU on the loaded PC, a ranking only.
  - **So most of Default's cost is outside the per-block codec levers:** engine mode decision and
    motion, frame-level deblock/CDEF, key frames.
  - **Absolute depth removal at 0.5 steps:** a bad trade (+11.4 %).

- **First Zen 5 frontier, AV1** (c8a, 4 CTC sequences × 2 QPs, 130 f, light-pipeline agent).
  Weighted BD: SVT M13 401 fps +4.1; M12 357 −1.5; our Fast 311 +1.6; M10 216 −14.2; Default with
  every SB light 200 −9.5; regular_fraction 0.25 171 −15.8; 0.5 158 −17.2; Default 152 −17.9.
  - **Single levers with every SB light, speed gain on Default:** trellis off +10 %, tx-split off
    +8 %, decision/motion +5 %, the rest ≤ +1 %.
  - **The per-SB budget on a Default base tops out at 200 fps.** Fast's extra ~110 fps is
    frame-level (deblock/CDEF, key-frame intra, structure).
  - **Next:** a Fast base with heavier regular SBs, and a frame-level deblock/CDEF gate.
- **AV1 Fast re-point** (feature/av1-av2-speed): the shared gate landed (6d4d2c636).
  - **Lens:** Fast with Default depth + full forward −6.5. Gated from quantizer 0.78: −6.2. Plus
    the split trial below 0.7: −7.9.
  - **CBR pool screen of the first:** L1 J −4.36 %, L2 −2.95 %; L3 running.

- **New AV1 Fast landed** (5e21b7a80): lens Weighted +1.6 → −6.5; CBR L3 J −3.95 %; c8a about
  255 fps at fixed QP.
- **Zen 5 session 2** (light pipeline, c8a). Weighted / J against x264 medium:

  | Point | fps | Weighted | J |
  |---|---:|---:|---:|
  | M12 | 349 | −1.5 | −10.4 |
  | Fast-base rf 0.25 | 235 | −3.3 | −17.3 |
  | M10 | 210 | −14.2 | −19.5 |
  | Fast-base rf 0.5 | 207 | −7.1 | −18.8 |
  | Default + light bundle rf 0.25 | 174 | −14.5 | −24.0 |
  | Default | 152 | −17.9 | −26.0 |

  - **Weighted:** nothing beats M10 yet.
  - **J:** we are about level with M10 at its speed.
  - **Frame-level filter gate (S5, 7c2b840cf):** at most +5 % speed on Default, so deblock/CDEF
    is not where Default's time goes.
  - **Next:** stage-by-stage profile of Default rf 0 against Fast.
- **Astra:** AVX-512 ICL/GFNI deblocking, shared by AV1, AV2 and VP9 (bb69841d2), 532/532 streams
  verified.

- **Where Default rf 0 still loses to Fast** (callgrind, lowlight, 130 f, one key): 2.15 G
  instructions in all. **The key frame is 1.2 G (56 %)**: Default's key intra set and planner leaf
  cost; key frames are always regular. P frames are 0.94 G: deblock, more second-reference
  decisions (`structure.unified` with `refs.second_full`), and mode decision.
- **c8a session 3:** Fast-base rf 0.25 + deblock + filter gates runs at 228 fps, J −19.9. That
  beats M10 (213 fps, −19.5 J) **in J** on the lens. CBR is not screened yet. Its Weighted is
  −9.4 against M10's −14.2.
- **Next:** key-frame superblock classes (`effort.key_regular_fraction`, a spatial proxy), then the
  second-reference trial as a light lever, then a joint tune and the CBR screens.

- **AV1/AV2 writer landing 1** (cec603273, exact). The writer path goes from about 1,790 to
  about 1,050 Ir per byte (SVT M10 is about 760); whole AV1 Default encode −3.5 % (callgrind, 40 f
  CTC). The range coder is now a 64-bit window, with a flat symbol log and replay, sized level
  maps, an eob==1 shortcut, and unrolled CDF adaptation. AV2 got the same coder.
- **AV1 writer landing 2** (exact; `feature/av1-writer2`). Callgrind, WestWindEasy 40 f, qindex 185,
  `~/nanocodec-data/work/av1-writer2/` (`cg.sh`, `writer.py`). Writer path = `CoeffCdfs::write`
  inclusive + the deferred frame's replay (`encode_run`, every symbol of the frame):
  - Default 397.5M → 370.3M, **1,062 → 990 Ir/byte**; Fast 1,134 → 1,052; Faster 1,074 → 1,001.
  - Whole encode: Default −1.81 % (7803.6M → 7662.4M), Fast −1.19 %, Faster −1.29 %. Most of it
    is rate estimation (`CoeffCdfs::cost` 704.8M → 598.7M on Default), which shares the level map.
  - The level map holds magnitudes saturated at 3, so the `coeff_base` sum is five plain bytes;
    `coeff_br` reads levels above 2 back from the block. `cost` uses the sized maps too.
  - The replay codes every logged symbol as one general interval (binary, literal and omitted
    alike) with the coder state in registers: 38.5 → 35.3 Ir a symbol (3.56M symbols, Default).
  - SVT M10 on the same content, same definition (`av1_write_coeffs_txb_1d`, `get_nz_map_contexts`,
    `txb_init_levels`, `od_ec_encode_cdf_q15`, flush): 324.0M for 394,638 bytes, about 820 Ir/byte;
    its binary symbols are coded inline in their callers and are not in that count.
  - What is left is mostly the deferred frame itself: every symbol is logged (about 25 Ir) and
    replayed (35 Ir), where SVT codes once (about 30). Removing the log needs CDEF decided before
    a superblock's symbols are coded, which changes streams. AV2 is not affected: its recording
    writer is a different structure (models re-adapted on replay).
- **AV1 coefficient pricing against SVT M10** (`feature/av1-writer2`, same harness).
  - **Why ours cost ~8x on Default:** call count first. On Default, `CoeffCdfs::cost` runs
    266K times (260K from `residual_rd_cost`) at 2,245 Ir a call: 598.7M. M10's full walk
    (`svt_av1_cost_coeffs_txb`) runs 74K times at 1,317 Ir: 97.4M; `txb_estimate_coeff_bits` adds
    8.3M over 107K calls. On Fast and Faster, ours is 124M / 114M, about M10's total. Nearly all
    of those calls come from the one key frame's planner.
  - **SVT M10 prices exactly:** real time M10 runs `rate_est_level` 1, the full CDF walk. Only
    M10 with one hierarchical level, and M11+, use the linear end-of-block model (level 0).
    `coeff_rate_est_lvl` 2, linear below an end-of-block threshold, is used only in light PD0.
    Our linear model is already ported (`av1.key_linear_rate_eob_share`, `coeff_rate_linear`).
  - **Exact landing 3:** the magnitude walk runs over per-size scan tables (coefficient, map
    index, `nz_map_ctx_offset` by scan index, SVT-style prepared contexts), with the DC step
    peeled and the base costs summed in a local. Whole encode: Default 7662.4M → 7566.2M
    (−1.26 %), Fast −0.60 %, Faster −0.69 %. `cost` 598.7M → 511.9M on Default. Writer path
    990 → **959 Ir/byte** (Fast 1,017, Faster 973).
  - Tried and dropped: one packed array of steps instead of three (+0.06 % on Default).
  - Not shared with AV2: its contexts clip neighbours at 5 or 3 by region (AVM `clip_max5` /
    `clip_max3`), so the saturate-at-3 map does not carry over.
  - **Linear pricing of inter-frame residual trials (`av1.linear_rate_eob_share`, not landed;
    branch `feature/av1-linear-inter` 9fccf8bb2):** the non-key counterpart of
    `key_linear_rate_eob_share`, SVT `coeff_rate_est_lvl` 0.
    - **Fast and Faster:** no effect, with identical callgrind counts. They run no inter-frame
      residual trials; their pricing is the key frame's, already screened (+1.04 % J, above).
    - **Default, share 1 against 0, CPU:** x86 callgrind −1.66 % (CBR 1500) and −2.07 % (fixed
      qindex 185).
    - **Pool levels with that CPU (2026-11-17 draw):** L1 −1.13 %, L2 −0.81 %, L3 J −0.29 %
      [−0.55, −0.04] (n 78): BETTER.
    - **Final-check (600 f, quality only):** J +0.49 % [+0.29, +0.68]; MS-SSIM is worse on 27 of 29
      clips. The periodic-key pulse guard fails at +0.92 dB (meva_school_g336, dKeyN), against
      +0.50. With the −1.7 % CPU at k 0.6 that is about −0.5 % J_real.
    - **Not landed:** the gain is all CPU credit, quality is worse almost everywhere, the pulse
      guard fails by 0.4 dB, and the Jetson is not measured. The trade, about 0.3 % quality per
      1 % CPU, is cheaper than the preset levers' 1.3–4 %. If wanted, a Jetson count and a look at
      meva_school's keys are the next step.
- **AV1 Fast re-point key pulse:** Default's key intra set doesn't fix it (worst clip +1.10 dB);
  key boost makes it worse (+4.56). Being tested: partial forward on keys only, and a coarser key
  QP.

- **CBR timing via the chart path (c8a, 3 chart clips):** Fast 301 fps, fr25lfg 238, M10 233,
  Default 171. fr25lfg sits about 1 J point *below* the Fast→Default line in CBR (estimated J);
  the earlier "above" was withdrawn.
- **Key budget** (key_regular_fraction 0 → 1 on fr25lfg): 227 → 213 fps, J −19.9 → −21.7. It is
  smooth, with no knee, and does not fix the pulse.
- **Gap in the chart itself:** the Fixed QP chart has none of our Faster or Veryfast rungs, only
  Slow, Default and Fast, while M12/M13 own 350–450 fps. The chart's default metric is J.
  - **Next:** lens and c8a-time the Faster/Veryfast rungs at fixed QP, then add them to the lens
    and site if they compete.

- **Faster/Veryfast rungs at fixed QP** (c8a chart-path time, box lens; chart fps scaled from
  M13 = 445): AV1 Faster about 355 fps, Weighted +14.5 / J −5.5; HEVC Faster 402, +17.4 / +4.8;
  VP9 Faster 320, +12.8 / −3.5; VP9 Veryfast 334; H.264 Faster 448, +46.6 / +29.2; H.264
  Veryfast 524. **All are dominated by M12 (417 fps, −1.5 / −10.4)**, so adding them to the
  chart would not help. The M12/M13 band is a real gap. Closest: AV1 Faster needs about +17 %
  speed and about −5 J.
- **Next:** budget and gates on an AV1 Faster base aimed at M12. A joint tune on the Fast base
  is running (CBR J against CPU, lens check at acceptance).

- **AV1 Faster candidate f-c3** (Faster with its cheap-but-costly groups restored to Fast's
  values: subpel seed lambda, exact-rank margin, pre_tx_skip_energy, skip_bias_energy,
  analysis.window_*, chroma_skip_sad_num, near_count, coeff_rate_linear).
  - **Lens:** −1.1 Weighted / −15.9 J at 388 c8a fixed-QP fps, against M12 357 fps, −1.5 / −10.4.
  - **f-c4** (+ Default depth): 331 fps, −7.8 / −19.6, above the M12–M10 line.
  - **Pending:** chart-path timing, CBR screen, final-check.
- **Light-pipeline tune front points** (t33/t42/t59) sit *below* the Fast→Default line on the
  fixed-QP lens, e.g. t33 230 fps −4.5 / −18.2 against a line value of −8.7 / −20.4. Effort
  reallocation on a Fast base does not help the goal chart.
  - **Next:** a lever-group ablation of Fast against Default, the method that found f-c3.

- **HEVC Fast + merge alone (2026-11-16, feature/hevc-speed2): not landed.** Lens -3.1 -> -4.3
  Weighted at +3.5 % c8a time (0.34 points per 1 %, above the line's 0.22), CBR L1 quality -2.3 %,
  but final-check WORSE on the key pulse (pool_external_meva_admin_720p_536p +1.98 dB, J +10.3 % on
  that clip). Details: `docs/handoff/hevc.md`, "HEVC speed agent 2".
- **HEVC exact 916c569eb:** per P frame -0.15 % Johnny, -1.11 % WestWindEasy (box callgrind, Fast QP 37).

- **AV1 Faster = f-c3n with `rate.prekey_quiet` 1** (`feature/av1-fc3n`, 2026-11-16; the
  rate-control fix 86d268d54 un-starves f-c3's still-scene key). Against the previous Faster:
  - final-check (600 f, keys every 10 s, 27 clips, ladder): quality J **-9.78 % [-10.64,
    -9.01]**, every class better, key pulses -0.20 dB (first) / -0.14 dB (periodic); guard: one
    more rung of pool_meva_hospital_g301 overshoots (> 10 %), as in f-c3; that clip's periodic key
    pulse improves 0.66 dB.
  - x86 callgrind, CBR 600 kbit/s, 90 f: +14.0 % (crowd +11.3, bodycam +14.4, lowlight +16.4),
    so J at k 0.6 about -1.4 %; CBR L1 with that dCPU: J -0.77 % [-2.98, +1.07].
  - Zen 5 chart path (c8a.large, QPs matched to the chart bytes, 2 repeats): Faster 376 fps,
    **f-c3n 340**, f-c3 331, SVT M12 427 (`~/nanocodec-data/work/av1-lowlight/fc/t1`).
  - f-c3 (f-c3n plus pre-transform skip energy 30 and no skip bias): quality J -11.74 %
    [-12.73, -10.82] for +18.0 % CPU (about -0.9 % at k 0.6), L1 -1.27 % [-3.44, +0.58],
    331 fps; f-c3n is the better J per CPU and the faster point, so it is the one landed.

- **AV1 key frames: where the excess is, and a linear key rate** (`feature/av1-lowlight`,
  2026-11-16). Lowlight key frame at qindex 184, old Faster: ours 525M Ir against M12a's 378M.
  Ours: the key planner's leaf costs are 287M, and 239M of that is `residual_rd_cost`, whose live
  coefficient-rate walk (`CoeffCdfs::cost`) is 149M. Coding is 171M and CDEF 19M. M12a: CDEF
  115M, entropy coding 88M, mode decision about 140M. SVT M11+ real time prices coefficients with
  its linear end-of-block model on every frame, key frames included (`rate_est_level` 0). New
  parameter `av1.key_coeff_rate_linear` (default 0, streams unchanged) does the same for our key
  planner and intra trials. Key frame, lowlight q184: Faster 526M -> 356M, Fast 597M -> 383M,
  Default 1807M -> 1021M.

  | base | chart path, c8a fps | x86 Ir, CBR 600, 90 f | fixed-QP lens Weighted / J |
  |---|---|---|---|
  | Faster (f-c3n) | 342.6 -> 350.3 (+2.2 %) | -1.7 % | (old Faster) +14.5 / -5.5 -> +14.1 / -5.3 |
  | Fast | 254.4 -> 257.9 (+1.4 %) | -2.0 % | -11.0 / -19.6 -> -13.1 / -20.6 |
  | Default | 171.4 -> 181.7 (+6.0 %) | -5.5 % | -17.9 / -26.0 -> -18.3 / -25.4 |

  CBR on Faster: L1 J +0.46 % [-0.44, +1.51] (with its dCPU) and the static-class quality guard
  (+5.2 %; one chirla clip +18 %). Final-check (600 f, keys every 10 s): quality J **+1.04 %
  [+0.57, +1.50] WORSE**. Static keys get worse (dKey1 +0.21 dB, dKeyN +0.18 dB on static; worst
  periodic pulse +1.28 dB against the 0.50 guard). The other classes' pulses improve by about
  0.2-0.3 dB. At k 0.6 the -1.7 % CPU about cancels the quality loss. Not landed on Faster. The
  linear model ignores level magnitudes, which matter on fine-quantizer static keys. Next:
  linear pricing only for sparse blocks (SVT `coeff_rate_est_lvl` 2: linear below an end-of-block
  threshold, the full walk above), measured on Default first, where the gain is largest.

- **Where the linear key rate loses, and sparse-only pricing** (`feature/av1-lowlight`,
  2026-11-16).
  - Static clip pool_external_meva_admin_720p_536p, Faster, CBR 300 kbit/s, 600 f, key every
    300 (rate log and display recon). The first key has the same q141 and the same bytes (29.6 KB)
    with and without the linear rate, but 0.23 dB lower PSNR-Y (34.74 -> 34.51): the model costs
    key efficiency at equal rate. The second key moves with rate control's state: budget 86 kbit
    against 33 kbit, q152 against q171, 34.35 against 32.01 dB, after a 17.8 KB inter frame two
    frames earlier. The pulse loss is that efficiency loss, amplified by the budget the run-up
    leaves.
  - `av1.key_linear_rate_eob_share` (replaces `av1.key_coeff_rate_linear`, default 0): linear only
    below that share of a block's coefficients, the CDF walk above (SVT `coeff_rate_est_lvl` 2).
    Default, one key at fixed q, bytes / PSNR-Y against share 0; lowlight key-frame Ir:

    | share | admin (q150) | lowlight (q150) | crowd (q170) | bodycam (q170) | lowlight key Ir |
    |---|---|---|---|---|---:|
    | 0 | 82672 / 43.013 | 192394 / 43.192 | 80123 / 42.242 | 50453 / 43.885 | 2077M |
    | 0.03 | +0.1 % / -0.00 | -0.0 % / -0.01 | -0.4 % / -0.05 | +0.5 % / -0.00 | 2068M |
    | 0.1 | +0.4 % / -0.05 | +0.1 % / -0.01 | +0.2 % / -0.08 | +1.0 % / -0.07 | 2028M |
    | 0.25 | +0.9 % / -0.08 | +0.4 % / -0.01 | +2.6 % / -0.08 | +2.9 % / -0.08 | 1951M (-6 %) |
    | 1 | +16.6 % / -0.17 | +10.8 % / -0.27 | +10.2 % / -0.20 | +7.1 % / -0.15 | 1229M (-41 %) |

    Almost all of the time is in dense blocks, which is exactly where the linear model misprices,
    because it ignores level magnitudes. Sparse-only pricing saves little; full pricing costs
    7-17 % of the key's bytes at fine key quantizers. A threshold on end-of-block share (block
    density) is the energy dependence; it does not give a useful trade. Not screened. The open
    route is a magnitude-aware cheap estimate for dense blocks (end of block plus a per-level
    log-magnitude term, fitted to the walk), or an exact speed-up of the walk itself (2,560 Ir per
    call).

- **AV1 Veryfast** (`feature/av1-veryfast`, 2026-11-16; docs/handoff/av1-veryfast.md): first Faster
  plus fixed L1T3, chroma skip 7, 32x32 minimum inter blocks, depth scale 8, one NEARMV, no integer probes,
  variance-planned keys: 451-461 fps, final-check J +11.86 %, but its fixed layers code keys 1.8x
  larger and 2-3 dB above their P frames (periodic-key pulse +1.15 dB, worst +4.05 dB). Re-pointed
  to the same without fixed layers: c8a chart path 348 -> 419 fps (M13 459, M12 430 same session),
  Jetson cycles -11.2 %, final-check J +10.42 %, pulse +0.29 dB; lens J / Weighted -7.0 / +8.4 (M13
  -4.8 / +4.1, M12 -10.4 / -1.5). Other levers on top to win speed back cost +18 to +29 % CBR J.

- **Effort by frame role, "base-only"** (`feature/base-only`, 2026-11-16; study:
  docs/analysis/adaptive-effort-2026-11-16.md §5.1). `effort.regular_fraction 0` with
  `effort.light_nonbase_only 1`: base frames (and keys) get the next rung's per-block tools, and the
  `light.*` overlay gives non-base frames the rung's own values. Draw 2026-11-16; c8a chart path
  (aws_timing, 2 repeats); the lines are against the neighbouring presets at the same time. Work
  dir `~/nanocodec-data/work/base-only/` (parameter files `cand.params`, `faster-cand.params`).
  - **On Fast** (regular = the Default's 12 differing per-block values): 223.8 fps (x1.148 of Fast's
    time; Default x1.501); lens W / J -12.3 / -20.8 (line -13.3 / -21.7); CBR L1 -2.55 % (line
    -2.78), final-check -2.44 % [-2.98, -1.91] (line -2.85), pulse +1.86 dB
    (pool_external_meva_admin_720p_536p, J +11.2 %: its second key is starved, 93k against 158k
    bits at 194k, 2.6 dB worse, and the next 90 frames are 1.8-2.8 dB worse). Below the line on
    both charts: **out**. With `rate.key_reserve` (engine default since 3652eedf8) the CBR
    final-check is -2.90 % [-3.43, -2.41], no guard (meva_admin J +11.2 -> -4.2 %): on the CBR
    line by c8a time (-2.85), below it by callgrind instructions (x1.476 of Fast under CBR,
    Default x1.935: line -4.91); lens unchanged. Still **out** (docs/handoff/rate-control.md
    "Key reserve").
  - **On Faster** (regular = Fast's 14 differing per-block values): 313.7 fps (x1.094 of Faster's
    time; Fast x1.336); lens W / J -5.7 / -18.3 (line -1.5 / -15.2); CBR L1 -7.84 % (line -3.22),
    final-check -8.27 % [-9.60, -6.90] (line -3.38). Above the line on both charts, but the pulse
    guard trips (+1.11 dB, long +0.70; Fast itself against Faster: +1.29 dB on the same clip). On
    pool_static_london_bus_540p the P frames before the key gain 0.7-1.0 dB; at 388k the key is
    0.73 dB worse (294k against 327k bits) and its next frames about 0.6 dB worse until about K+30.
    Not a clean judgement case: **held, blocked on key funding** (the same family as the
    rate-control agent's layered-key work). Re-screen it once keys after a better-coded run-up get
    their bits. 2026-11-16 (rc-keyfollow): neither the key reserve (london_bus key not starved,
    byte-identical) nor a key planned against a mean of recent levels (london_bus pulse +1.11 ->
    +0.91, J -8.17 %) clears the guard: `pool_fit_static_meva_school330_540p` (+1.07) and
    `pool_chad_4_109` (+1.06) were over it all along, behind london_bus as the worst clip. Still
    held; what binds those two keys is not measured yet.
    2026-11-16 judgement (faster-baseonly; frame by frame on the final-check's own cached points,
    600 f, keys at 0 and 300, all rungs of the three clips; docs/handoff/light-pipeline.md
    "Faster base-only: the pulse judgement"): **not adopted, still held**. The three clips' pulse
    readings come from the long-window ladder pass, not the 2 s pass. The second key is
    absolutely worse than Faster's at 3 of 10 rung points (london_bus 389k -0.73 dB Y / -1.6
    VMAF, chad_4 300k -0.28 / -1.6, meva_school 97k -0.41 / -2.8), each with 6-12 % fewer key
    bits; elsewhere it is within ±0.3 dB. Frames after the key are never worse in PSNR-Y (K+1..29
    -0.06 to +0.57 dB, K+30..89 +0.05 to +1.14), while VMAF there moves -1.6 to +1.6. The chad
    and meva keys are allowance-bound in both arms with an empty buffer before the key. Faster's
    key overshoots its predicted size by more (+13 % / +25 % against +7.5 % / +12 %), and that is
    where its extra key bits come from. The candidate's first key, coded with Fast's tools, costs
    4 % / 12 % more at the same qindex, and its second key is planned 2-6 qindex coarser.
    **Adopted** later 2026-11-16 (coordinator; e097d830b): pool L3 with its CPU (dCPU +9.4 % at
    k 0.6) J -0.41 % [-1.39, +0.61] AMBIGUOUS, no guard; quality alone J -7.0 %. AV1 Faster is
    now 313.7 fps on the c8a chart path (x1.094 of the previous Faster's time, same session).
    Key fix forward: `rate.key_quality_match` (engine default 1) lifts the level-bound
    london_bus 389k key above Faster's (32.98 -> 34.30 dB, Faster 33.71); the chad_4 300k and
    meva_school 97k keys are bound by the physical buffer in both arms, and Faster's are
    larger only by overflowing it further (docs/handoff/rate-control.md "Key quality match").

## Veryfast and Faster base-only re-mix (feature/av1-weighted, 2026-11-17)

Goal: AV1 points ahead of SVT on the Fixed QP chart on Zen 5 in Weighted as well as J. Build e329c8b59.
Lens = `nceval.ctc local` on the build box (Weighted / J against x264 medium); fps = c8a.large fixed-QP
chart timing (`fixedqp_timing.py`, all 12 sequences x 6 quantizers, one instance per session, variants as
the rung's lens command plus `--param` pairs; SVT M12/M13 in every session). CBR CPU = x86 callgrind,
release build, the three chart clips at CBR 600 kbit/s, 90 f (geometric mean). Pool screens: draw
2026-11-17, k 0.6. Work dir `~/nanocodec-data/work/av1-weighted/` (`v/*.params`, `t/rec*.json`). Every lens
and timing row is in the lever ledger (base `veryfast` / `faster`, rate `lens`).

"Base-only" = `effort.regular_fraction 0` + `effort.light_nonbase_only 1`: keys and base (T0) frames take
the plain values, the other frames the `light.*` values.

**Veryfast** (shipped Veryfast's lever values stay on non-base frames; the base frames take):

| candidate | base frames take | Weighted / J | fps (Veryfast; M13, same session) | CBR |
|---|---|---:|---|---|
| Veryfast | - | +7.7 / -6.2 | 1087-1099 | |
| vI | Faster's `planner.inter_min_log2` 4 | +7.5 / -6.7 | 1067.3 (1087.3; 986.4) | |
| vM | + Faster's near count, integer probes, chroma skip | +6.1 / -8.5 | 1054.7 (1087.3; 986.4) | |
| vMAB | vM + Fast's subpel biases and refinement | +5.0 / -8.9 | 1038.4 (1087.3; 986.4) | |
| **vMC** | vM + Fast's pre-transform skip and skip bias | **+3.3 / -10.4** | 1044.3 / 1037.2 / 1024.9 / 1042.6 (1087.3 / 1082.4 / 1078.5 / 1090.9; M13 986.4 / 978.9 / 986.9 / 993.5) | x86 +4.9 %; L1 quality -3.29 %; L3 J **-1.27 % [-1.89, -0.68] BETTER** (quality -4.2 %), no guard; final-check quality J **-5.87 % [-6.32, -5.40]**, no guard |
| vMCAB | vMC + vMAB's | +2.4 / -10.6 | 1022.2 / 1007.7 (1082.4 / 1078.5) | x86 +7.4 %; L3 J -0.46 % [-1.11, +0.15] BETTER |
| vbo14 | Fast's 14 Faster base-only values only | +5.4 / -6.4 | 1018.4 (1088.5) | |
| vboDE | Fast's depth removal and full forward only | +9.4 / -3.6 | | |
| vboDEm | vboDE + Faster's inter_min_log2 | +6.0 / -7.0 | 990.3 (1088.5) | |
| vbo14m | all 14 + the four levers | +1.2 / -10.9 | 948.7 (1088.5; 983.9) | L1 quality -5.89 % |

- **Landed vMC** (2caf7d21d): AV1 Veryfast is ahead of M13 on both axes on the lens (+3.3 / -10.4 against
  +4.1 / -4.8), 3.9-6.0 % faster than M13 in four sessions. Changed presets: AV1 Veryfast only (288/297
  identical). Web speed hint 0.36 -> 0.38. Jetson not measured. Not yet on the site (needs the chart
  refresh's lens row and timing).
- On Veryfast's base frames, its own four levers and Fast's pre-transform skip carry the gain; Fast's depth
  removal and full forward transform cost speed for little (vbo14m against vMCAB: -1.2 W for -7 % fps).

**Faster** (target: ≥ M12's speed at ≤ -1.5 Weighted; M12 897-911 fps in these sessions):

| candidate | change on Faster | Weighted / J | fps (Faster; M12, same session) | CBR |
|---|---|---:|---|---|
| Faster | - | -5.7 / -18.3 | 799-818 | |
| fL | Veryfast's layer gates + `rate.key_level_mean` 1 | -5.8 / -16.9 | 843.9 (806.7; 903.0) | |
| fLCM | fL + light chroma 7, depth scale 8, NEARMV 1, inter_min_log2 5 | -4.4 / -16.0 | 865.8 (806.7) | |
| fLCMP | fLCM + light integer probes off | -3.9 / -15.7 | 879.4 / 882.1 | L1 quality +3.76 % |
| fLCMPh | fLCMP + non-base HME trio | -3.4 / -15.3 | 895.5 / 890.2 / 894.2 | |
| fLCMPw | fLCMP + non-base window 2x0 | -2.4 / -14.4 | 902.3 (806.9; 903.0) | |
| fLCMPk5 | fLCMP + half the key superblocks variance-planned (`effort.key_regular_fraction` 0.5, light `planner.intra_plan` 1) | -1.9 / -14.1 | 903.0 / 893.0 (M12 903.0 / 897.3) | |
| fLCMPi | fLCMP + `planner.intra_plan` 1 | +1.2 / -11.0 | 951.4 | |
| fLCMPhk75 | fLCMPh + key fraction 0.75 | -3.0 / -14.7 | 900.8 / 884.0 / 905.9 (M12 897.3 / 903.5 / 908.5) | x86 -6.5 % |
| fLCMPhn | fLCMPh, base frames without Fast's depth removal and full forward | -0.1 / -13.2 | 980.0 | |
| **fhkD** | fLCMPhk75 + partial forward on base frames | **-4.2 / -16.2** | 925.3 / 934.0 (M12 908.5 / 910.7) | x86 -7.4 % (lowlight +4.7 %); L1 quality +5.12 %; **L3 J +2.31 % [+0.92, +3.80] WORSE** with that CPU, class quality guards and periodic-key pulse +1.08 dB |
| fhkDnpc | fhkD + base NEARMV 1, probes off, chroma 7 | -3.6 / -15.0 | 952.6 / 945.2 (M12 910.7 / 910.5) | x86 -9.4 % |
| fhkDk5 | fhkD with key fraction 0.5 | -3.7 / -15.6 | 949.9 (910.7) | |
| fhkDk5npc | both | -2.5 / -14.4 | 963.9 (910.7) | |
| fhkI | fLCMPhk75 + inter_min_log2 5 on every frame | +3.3 / -9.4 | 981.6 | |
| **fD** | partial forward on base frames only | **-5.7 / -18.7** | 823.4 (811.9; 910.5) | x86 -0.6 %; L3 J **-0.29 % [-0.75, +0.17] BETTER**, no guard; final-check quality J **+0.42 % [+0.03, +0.83] WORSE** (no CPU credit), no guard |
| fDL / fDLh / fDLhk75 | fD + fL / + HME / + key 0.75 | -6.0 / -5.4 / -4.8 | 860.2 / 872.0 / 884.9 (811.9) | L1 quality +0.98 / +1.85 / +2.02 %; fDL x86 lowlight +12.8 % |
| fDh | fD + non-base HME trio | not on the lens | not timed | x86 -1.8 %; L1 quality -0.30 % |

- Faster's speed can be bought on the fixed-QP chart (fhkD/fhkDnpc: ahead of M12 in Weighted and J at
  +1.8 to +4.6 % of its speed), but in CBR the same levers lose: fhkD's pool L3 is WORSE with its CPU credit.
  The losses are the non-base light levers (chroma, depth 8, NEARMV, inter_min_log2 5, probes) and the layer
  gates (more lowlight instructions under CBR). Not landed.
- **fD not landed:** lens -5.7 / -18.7 at +1.4 % fixed-QP speed and CBR L3 BETTER with its CPU, but the
  long-window final-check is +0.42 % quality (about +0.06 % with -0.6 % CPU at k 0.6): level, and too small
  to move the chart. AV1 Faster stays 9-10 % slower than M12 on the Fixed QP chart (-5.7 Weighted against
  -1.5). Open: a Faster speed lever that holds in CBR; the light levers above do not.
- Fast's full forward transform on Faster's base frames costs Weighted on the lens (fhkD -4.2 against
  fLCMPhk75 -3.0) and on Veryfast (vbo14m against vMCAB, with depth) little for its time.
- AWS: 8 on-demand c8a.large sessions (2.5-12.6 min), $0.16, all terminated and in the ledger; lens runs on the shared build box.

## AV1 Faster exact speed-ups (feature/av1-faster-exact, 2026-11-17)

Goal: close AV1 Faster's ~10 % speed gap to M12 on the Fixed QP chart with stream-identical changes.
Builds: `--profile profiling` (release code with debug info). Callgrind: x86 PC, `NC_DSP_ISA=avx2`,
the lens's own commands (`av1-weighted/t/plan6.json`). Zen 5: c8a.large, `perf` cycles by symbol
(`adaptive-diag/mix/remote_mix.py`), and A/B chart timing of two builds with M12 in each session (12
sequences x 6 quantizers, 4 rotated rounds, `fixedqp_timing_remote.py`). Scripts and raw output:
`~/nanocodec-data/work/av1-faster-exact/` (`cg.py`, `fnrank.py`, `zen5/ab.py`, `zen5/agg.py`).

**Where the gap is.**
- **By sequence.** In the chart record at e329c8b59 (`t/rec7.json`), Faster's ms/frame against
  M12 runs from 0.91 (RedKayak) and 0.95 (WestWindEasy) to 1.20-1.22 (Stockholm, SpeedBag), 1.37
  (SnowMountain) and 1.50 (FourPeople 480x270). Most of the gap is on the 270p and 360p sequences.
- **Instructions (callgrind, AVX2).** On WestWindEasy, Johnny and SpeedBag at q135/q210, 130 f,
  Faster runs 29.93 G instructions and M12 (qp 34/52) 32.39 G. On FourPeople, SnowMountain and
  Stockholm, Faster runs 7.53 G and M12 6.78 G (869 M of M12's are `memset`).
- **Zen 5 cycles.** Johnny and WestWindEasy: Faster 1126 / 3529 / 753 / 2210 M against M12
  1076 / 3879 / 789 / 2328 M (q135/qp34, then q210/qp52). The three small sequences: Faster 2101 M,
  M12 1658 M (+27 %).
- **Zen 5 symbols, small sequences, share of cycles:**
  - ours: `CoeffCdfs::write` 5.6 %, `code_block` 4.6 %, `CoeffCdfs::cost` 4.6 %,
    `choose_inter_mode` 3.8 %, `code_plane` 3.6 %, `grid_x4` 3.4 %, `encode_run` 2.9 %, border
    stores 3.5 % (`border_store::avx2::copy` 1.8 + `fill` 1.7), libc copies and fills 5.0 %;
  - M12: `av1_write_coeffs_txb_1d` 4.2 %, `svt_aom_quantize_b_avx2` 4.1 %, `od_ec_encode_cdf_q15`
    3.2 %, 8x8/16x16 SAD 3.0 %, libc 5.1 %.
  - Coefficient coding is 13 % of our cycles there against about 8.5 % of M12's.
  - `CoeffCdfs::cost` is the key frame's planner (`KeyPlanner::leaf_cost` -> `residual_rd_cost`):
    its share is 2.7 points higher on the small sequences than on 720p.
  - The border stores are 2.2 points higher on the small sequences than on 720p.
- **Callgrind ranking, 720p/360p set** (self instructions): `CoeffCdfs::write` 5.5 %, CDEF
  `block::<8>` 4.7 %, `code_block` 4.1 %, `sad_grid_best_64x64` 3.9 %, `code_plane` 3.5 %,
  `variance_w32` 3.2 %, `quantize_levels` 2.7 %, `encode_run` 2.6 %, `sad_scan_16x16_even_rows`
  2.6 %, `forward_16` 2.5 %. CDEF is 13 % inclusive; M12's CDEF kernels are about the same count.
- Callgrind's `memcpy`/`memset` counts overstate cheap large copies (the 16 KiB CDF context
  clone counts 58 K instructions); the copies callgrind ranks (CDF save, analyzer moves) are not
  where Zen 5 time goes and were not changed.

**Landed (exact).** Both changes: `preset_identity` 297/297, lens stream sizes equal on c8a, ledger rows.
- **c9e59feed: the AVX-512 tier writes picture borders with ordinary stores.** The AVX2 tier
  keeps the non-temporal stores (HEVC Faster, Zen 3: -5.7 %).
  - c8a A/B against the e329c8b59 code: 819.8 -> 847.6 fps (+3.4 %), M12 914.7.
  - Every sequence was faster: FourPeople -5.3 %, SnowMountain -6.9 %, Johnny -3.1 %,
    WestWindEasy -0.8 %.
  - Why Zen 5 differs from Zen 3 is not known.
- **4d317fa52: AV1 CDEF uses the backend's `Dsp`.** CDEF built a new `Dsp` five times a frame.
  Callgrind: Faster -0.03 %, Fast -0.03 %, Default -0.06 %.
- **Final pair** (c8a, one session, 4 rounds): Faster d40f86b38 810.6 -> head 845.2 fps (+4.3 %),
  M12 903.4. Faster is now 6.4 % slower than M12, from 10.3 %.

| sequence | before | after | M12 (ms/frame) |
|---|---:|---:|---:|
| FourPeople 480x270 | 0.468 | 0.439 | 0.312 |
| SnowMountain 360p | 0.506 | 0.470 | 0.371 |
| SpeedBag 360p | 0.814 | 0.777 | 0.670 |
| Stockholm 360p | 0.889 | 0.849 | 0.742 |
| TouchdownPass 360p | 0.942 | 0.902 | 0.804 |
| BlueSky 360p | 1.118 | 1.079 | 1.009 |
| Vertical_Bayshore 270x480 | 0.626 | 0.594 | 0.627 |
| RedKayak 360p | 2.040 | 1.981 | 2.265 |
| Johnny 720p | 1.672 | 1.606 | 1.583 |
| KristenAndSara 720p | 1.752 | 1.689 | 1.645 |
| ControlledBurn 720p | 3.742 | 3.666 | 3.752 |
| WestWindEasy 720p | 4.917 | 4.845 | 5.215 |

**Zen 5 kernel tiers** (`nc-checkasm --bench`, c8a, `--isa avx2` against `--isa avx512`, one
run each). The AVX-512 tier is slower than AVX2 on Zen 5 for:
- AV1 decoder inverse transforms (`av1_inverse_*`): 1.2-2.25x slower, for example 32x64
  `dct_dct` 448 -> 890 ns and 64x64 `i16_dct_dct` 1228 -> 1962 ns. This matters for decode
  speed and was not changed here.
- The encoder's sparse inverse DCT (`inverse_32_sparse` / `inverse_64_sparse`): low8 32x32
  144 -> 234 ns, low16 149 -> 205 ns. These are about 0.3 % of Faster's Zen 5 cycles.
- Narrow variance at height 8: 4x4 1.33x, 8x8 1.37x slower; 8x32 level.
- The AVX2 narrow inverse with dense large 32x32 coefficients took 29.6 us against 0.33 us on
  the AVX-512 tier (the bench's `low32` rows). That is the AVX2 tier's guarded fallback; how
  often real streams hit it is not measured.
- Everything else on Faster's path (SAD grids, sub-pixel variance at 32/64, forward transforms,
  quantiser, CDEF) is 1.2-3.4x faster on AVX-512.

**Open.** The remaining gap is on the small, easy sequences and spread over many functions:
coefficient coding, block glue, and key-frame RD pricing (`CoeffCdfs::cost` 4.6 % of small-sequence
cycles). The coefficient writer and range coder have been optimised several times
(e677b1a0f, ce82237f3); no exact item left there was found in this pass.

AWS: 5 on-demand c8a.large sessions (2.0-6.1 min each, about $0.03 in all), all terminated and
in the ledger.

## Next decision points

- When the diagnosis lands, brief vectorisation and widening work ranked by Zen 5 gain.
- Propose AV1 and HEVC preset points that beat M10 (≥ 229 fps at ≤ −14.2 Weighted) and M12
  (≥ 409 fps at ≤ −1.5), confirmed by a Zen 5 chart re-time.
