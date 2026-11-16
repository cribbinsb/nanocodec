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

- 2026-11-16 to 2409144f3: kernel-sharing merges cut whole-program instructions (callgrind, crowd
  720p, 30 f, release) by HEVC −6.25 %, VVC −3.93 %, H.264 −0.05 %, VP9 +0.02 %, AV2 +0.05 %,
  AV1 +0.12 %. Exact engine/codec cuts: VP9 about −4 to −5 % (end-of-block tests, quantiser
  setup), AV1/AV2 −0.5 to −0.7 % (CDEF search, intra edges), H.264/HEVC intra cleanups. Not yet
  re-timed on Zen 5.

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
- **RF landed and was reverted** (5e21b7a80, reverted by ab5c67dff). The bundle final-check (600
  frames, keys every 10 s) gave J −6.37 % [−7.51, −5.17] but WORSE on the periodic-key pulse:
  worst clip +1.51 dB, all static clips (otw_homes_00004, chad_4_109, meva_school330,
  london_bus). The P frames got better; the DC-only key frames did not.
- **Open:**
  - Why the pulse grows that much is not measured. Next test: RF with the Default's key intra
    set (`av1.keyframe_*`; −2.1 Weighted on its own in the diagnosis) through the same
    final-check.
  - Why the split trial overshoots in CBR is not known.

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
- **AV1 Fast re-point key pulse:** Default's key intra set doesn't fix it (worst clip +1.10 dB);
  key boost makes it worse (+4.56). Being tested: partial forward on keys only, and a coarser key
  QP.

## Next decision points

- When the diagnosis lands, brief vectorisation and widening work ranked by Zen 5 gain.
- Propose AV1 and HEVC preset points that beat M10 (≥ 229 fps at ≤ −14.2 Weighted) and M12
  (≥ 409 fps at ≤ −1.5), confirmed by a Zen 5 chart re-time.
