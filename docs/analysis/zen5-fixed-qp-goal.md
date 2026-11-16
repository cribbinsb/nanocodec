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
- **VVC key-frame bundle:** passed the pool screen but failed the 10 s key pulse guard on static
  clips. The cause is rate control's second-key budget (`feature/vvc-speed` report).

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

## Next decision points

- When the diagnosis lands, brief vectorisation and widening work ranked by Zen 5 gain.
- Propose AV1 and HEVC preset points that beat M10 (≥ 229 fps at ≤ −14.2 Weighted) and M12
  (≥ 409 fps at ≤ −1.5), confirmed by a Zen 5 chart re-time.
