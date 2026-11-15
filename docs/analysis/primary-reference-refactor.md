# A per-frame primary reference: what the engine assumes (2026-11-15)

**Why.** H.264's cheapest modes (`P_Skip`, and the partitions the SATD search
ranks) predict from list index 0 only. With the second reference on, a still
block reaches the finer high-quality (HQ) frame only as a full inter
macroblock, so it stays on skip from the coarser previous frame. Fixed QP 32,
60 frames, flat: ref 1 wins 0.8 % of `static_cam22`'s area against 80 % skip,
where HEVC, AV1 and VP9 skip or merge from the second reference and gain most
on static content. The direct fix is to list the HQ frame first on still
frames, so `P_Skip` copies it.

**What was tried (reverted).** `refs.second_primary_still` swapped
`FrameReferences::reference` and `secondary` on frames after a still one, and
H.264 listed the long-term picture at index 0 (`long_term_pic_num`, then a
`Subtract` command for the previous frame). The streams decode bit-exactly in
ffmpeg and our decoder. The quality did not hold:

- Swapping the analysis link as well: `static_cam22` +12 % bytes, -0.44 dB.
- Swapping only the coded references: +29 % to +45 % bytes.

The swapped frames lost skip (80 % to 52 %) and the cadence base boost (QP 28
became 31). The engine reads `reference` both as "the frame the pixels are
predicted from" and as "the analysed previous frame".

## Paths that assume reference = analysed previous frame

| Path | Where | What it reads | What it needs |
|---|---|---|---|
| Analysis link | `encoder.rs` `apply_search_role`, `analyze_written_timed(planned.link())`; `nc_analysis::analyzer` slot sources, HME pyramids | motion field, pre-analysis moments, noise, all against `reference.slot` | the previous frame as the analysed link, plus (optional) a motion search against the primary slot when it differs |
| Pre-analysis static share | `preanalysis.rs` `static_fraction`, read by the cadence (`cadence_depth`), AQ, rate control | the change against the analysed reference | always against the previous frame |
| Still detector | `nc_analysis::still` (`min_run` consecutive static frames, `guard`) | per-block static runs against the analysed reference | the previous-frame chain; a swapped frame must not reset the runs |
| Static skip / still skip / noise skip | `StaticSkipTest`, `hevc/av1.still_skip`, `decision.noise_skip`, H.264 `static_skip` | the zero-vector residual against `pictures.reference` | the test against the picture the skip actually copies (the primary), with thresholds re-checked for an HQ frame several frames back |
| Reuse chain | `encoder.rs` ~1476 (`follows`, `term.set_reference`), `reuse.rs`, `persist.*` | `reference.distance == 1` or the reference is the last frame the term saw | follow the analysed chain: a swapped frame currently resets the term, so its refinement offsets are re-spent |
| Cadence strength | `cadence_depth`, `cadence_motion_hint(m, reference.distance)` | the static share and motion at the primary's distance | the analysed (previous) link's static share and distance |
| Scene cut, structure motion | `is_scene_cut`, `structure.observe(motion_signal(.., planned.reference.distance))`, `scene.rs` history per distance | SAD and motion at the reference distance | the analysed link |
| Secondary seeds | `with_secondary` `scaled_vector(search_mv, ratio)`, `secondary_ratio` | an older secondary (ratio > 1) seeded from the primary's search | a seed for a younger secondary: the analysed vector itself |
| Slot bookkeeping | `second_reference.rs` `apply`/`coded`, `ltr.rs`, `feedback` | slot refreshes | unchanged (the swap moves no slots) |

## Proposed shape and size

Split the frame's references into `analysed: ReferenceLink` (the previous
frame, which analysis, the still detector, the reuse chain, the cadence and
scene logic read) and `primary`/`secondary` (prediction only, in list order).
The engine decides the order (`refs.second_primary_still`, or "HQ finer by
N QP"). A backend opts in with `SECOND_REFERENCE_AS_PRIMARY` and maps the
order onto its syntax: H.264 about 40 lines (done once, bit-exact), and HEVC
and VVC need a reference picture list order of their own. Motion search
against the HQ slot runs only on frames that swap.

Estimate: about 2-3 days. That is about 10 call sites in `encoder.rs`, the
analyzer's second search, re-checking the skip-test thresholds, landing
conformance for H.264 (JM, ffmpeg, ours), then fixed-QP and CBR L1-L3 for
H.264 medium and slow. The expected gain is unmeasured; the bound is the static
class's share of the second reference's gain in HEVC and AV1 (static 5.7 % of
block cost in AV1's traces, `docs/handoff/tuning.md`).

Also measured and dropped: sub-block (8x8) vectors for the secondary reference
in the shared engine. The primary's analysis vectors, scaled to the
secondary's distance, were passed to H.264's partition search, at fixed QP 32
with 60 frames. Results: crowd +0.03 % bytes / -0.005 dB, static_cam29
+0.03 % / +0.006 dB, pan_jaad_0101 +0.07 % / +0.004 dB. That is within noise,
and only H.264 uses sub-block motion.
