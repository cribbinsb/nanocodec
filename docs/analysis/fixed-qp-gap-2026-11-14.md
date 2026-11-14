# Fixed-QP coding efficiency: the gap map (2026-11-14)

**Question (owner).** "Establish a comprehensive program to improve our performance at fixed QP."
Rate control costs our encoders +10.7 to +21.5 % BD-J against their own fixed QP
([rc-cost-2026-11-14.csv](../results/rc-cost-2026-11-14.csv)); other agents are working on that. This note
covers the other half: how efficient each codec's tools are with rate control out of the picture, how far
each codec is from reference encoders, and which tools account for the gap. It is phase 1 of the program.
Phase 2 works through the ranked list at the end.

## Method

`python3 -m nceval.fixedqp` (new, `eval/nceval/fixedqp.py`) does three things:

- **Clips.** It draws from the graded pool (`nceval.pool.draw`) with the rotation key `fixedqp-<UTC date>`.
  The draw is stratified and family-disjoint like the screens' levels, but it is a different draw from the
  day's adoption levels. Ranking ideas here therefore does not use up the clips that later judge them. This
  run used `fixedqp-2026-11-14`, level `quick`: 24 clips (4 crowd, 4 low light, 4 pan, 4 static, 4 body-cam,
  3 texture, 1 drone; 360p to 1080p), 150 frames each (a short clip is measured whole). No clip was picked by
  name. The AOM/JVET sequences stay sealed.
- **Quantizers.** Every encoder runs at fixed QP: our `--qp` with one key frame, and each anchor in its
  fixed-quantiser mode with AQ off. Each clip gets four quantizers that span the real-use ladder's rates
  (150 to 1200 kbps at 720p30, scaled by pixel rate, ±25 %, the top capped at half the source rate). They
  are found from cached rate probes.
- **Scores.** It reports BD-J (`nceval.jreal`: the mean of the full-range and lower-half BD-rates over
  PSNR-YUV, VMAF v1 and MS-SSIM) and each metric's full-range BD-rate. Values are class-balanced, with a
  90 % bootstrap interval over clips (`nceval.interval`). `gap` compares families. `ab` compares candidates
  with their anchor at the anchor's own quantizers (matched QP, paired per clip). `cost` gives callgrind
  P-frame instructions (`tune.instruction_cost`).

The new anchor kinds are `vvenc-ld-<preset>` (VVenC's low-delay configuration files, 8-bit, one thread) and
`avm-ld-s<speed>` (AVM). Both are fixed-QP only and live in `nceval.cbr`. AVM turned out to be too slow to
include (one 20-frame 540p point at speed 6 took more than 3 minutes), so AV2 is compared against AV1
anchors and our own AV1.

Build: variance-integration 7f1ecc82. Results: `~/nanocodec-data/work/fixedqp/results/`.

## 1. Where each codec stands (Default and Fast presets at fixed QP, 24-clip draw)

BD-rates in % (negative: fewer bits at equal quality); the interval is for J; n = clips with a reliable BD.

| Ours | vs | J [90 % CI] | PSNR-YUV | VMAF v1 | MS-SSIM | n |
|---|---|---|---:|---:|---:|---:|
| H.264 Default | x264 medium `--tune psnr` | +4.3 [+2.2, +7.1] | +0.7 | +8.2 | +4.6 | 22 |
| H.264 Fast | x264 medium psnr | +14.8 [+12.6, +17.0] | +12.1 | +20.3 | +17.7 | 22 |
| HEVC Default | x265 medium psnr | +0.5 [-3.0, +5.5] | +1.3 | -1.6 | +6.3 | 23 |
| HEVC Fast | x265 medium psnr | +17.0 [+13.6, +21.4] | +20.5 | +13.6 | +22.7 | 22 |
| VVC Default | x265 medium psnr | -9.1 [-11.8, -5.5] | -8.1 | -10.2 | -3.3 | 24 |
| VVC Fast | x265 medium psnr | +6.4 [+3.5, +9.8] | +12.2 | +4.7 | +9.4 | 24 |
| AV1 Default | SVT-AV1 M12 rtc | -24.3 [-27.0, -20.7] | -28.2 | -27.8 | -17.4 | 23 |
| AV1 Fast | SVT-AV1 M12 rtc | -14.5 [-17.4, -10.8] | -12.7 | -19.7 | -9.4 | 23 |
| VP9 Default | libvpx rt cpu 7 | -38.7 [-45.9, -30.7] | -41.2 | -35.3 | -35.5 | 13 |
| VP9 Fast | libvpx rt cpu 7 | -27.7 [-35.6, -19.2] | -29.1 | -24.7 | -23.6 | 13 |
| AV2 Default | SVT-AV1 M12 rtc | -27.9 [-29.9, -25.0] | -30.9 | -32.2 | -19.6 | 22 |
| AV2 Fast | SVT-AV1 M12 rtc | -18.4 [-21.3, -14.4] | -17.6 | -20.9 | -12.2 | 22 |

The same Defaults against one common yardstick, x264 medium psnr: H.264 +4.3, HEVC -27.3, VP9 -26.3,
AV1 -25.6, AV2 -30.0, VVC -34.2. Between our own codecs: AV2 vs AV1 -4.0, VVC vs HEVC -9.4, VP9 vs HEVC
+2.5. Fast vs Default per codec: +10.9 (H.264) to +18.3 (VVC). Context anchors vs x264 medium psnr:
x264 medium default tuning +5.0, x265 medium psnr -23.6, SVT M12 -6.4, libvpx cpu 7 +35.0 (n 12: its
rate floor leaves few overlaps).

Reading:

- **H.264 is the one codec behind its real-time reference at fixed QP** (+4.3 J vs x264 medium psnr, worst
  on VMAF at +8.2). That matches the chart-clip figure (qp_vs_qp +0.4 / +2.6 in rc-cost).
- **HEVC Default is level with x265 medium psnr.** VVC Default is 9 % ahead of x265 medium psnr and 9 %
  ahead of our HEVC. AV1, AV2 and VP9 are far ahead of their real-time references (SVT M12 rtc, libvpx rt).
- **The Fast presets give up 11-18 % J against their Defaults.** That is the largest single block of
  coding efficiency in this map, and it is a CPU trade by design.
- **MS-SSIM is our weakest metric against the x26x anchors** (HEVC +6.3 while PSNR is +1.3). VMAF is the
  weakest for H.264.

Slow references (x264 veryslow, x265 slow, SVT M8, libvpx good cpu 2, VVenC faster LD): section 4 (being
measured).

## 2. Tool attribution (matched-QP ablations on our Defaults, first 16 clips of the draw)

Each line switches one tool off or to its maximum against the codec's Default, at the Default's
quantizers. Negative J means the change helps. "0.0 everywhere" means the parameter does not reach that
codec's backend.

### H.264 (n = 14)

| Change | J [90 % CI] | PSNR | VMAF | MS-SSIM |
|---|---|---:|---:|---:|
| motion search maximal (`analysis.*` early exits off) | -1.0 [-1.9, -0.2] | -0.9 | -1.0 | -1.1 |
| partition search everywhere (`h264.split_gate_satd` 0, uniform-vector skip off, refine 16x16 quadrants) | -0.7 [-1.1, -0.4] | -1.8 | -0.4 | -0.9 |
| RD 4x4/8x8 transform choice (`h264.transform_size_rd` 1) | -0.5 [-0.7, -0.3] | -1.3 | -0.6 | -0.3 |
| more intra trials (`decision.intra_gate_variance` 16) | -0.3 [-0.4, -0.1] | -0.4 | 0.0 | -0.8 |
| trellis 8 states | -0.2 [-0.4, 0.0] | -0.2 | -0.3 | -0.1 |
| flat P instead of the adaptive 3 layers | +2.5 [-0.6, +5.7] | +3.5 | +0.7 | +4.4 |
| trellis off | +2.3 | | | |
| deblocking off in P | +8.3 | | | |
| directional intra estimate on | +6.1 (harmful for H.264) | | | |
| λ x0.85 / x1.15 | +0.6 / +2.2 (tuned) | | | |
| exact-ranking early exits off, sub-pel gates off, partition planner levers | 0.0 to +0.1 (no effect) | | | |

The search-depth levers that exist add up to about 2.5 % J at most. The remaining +4 % to x264 medium is
structural: x264 medium codes with 3 references, sub-pel RD (subme 7) and an RD mode decision over all
partitions. Of those, H.264 here has only the 3-layer temporal structure, and the second reference is
rejected with layers.

### HEVC (n = 12-15; three points lost to the VMAFx outage, refilled in the final table)

| Change | J [90 % CI] | PSNR | VMAF | MS-SSIM |
|---|---|---:|---:|---:|
| trellis 8 states | -0.5 [-0.8, -0.3] | -0.8 | -0.4 | -0.8 |
| full P-frame intra search (`hevc.coarse_inter_intra` 0) | -0.5 [-0.7, -0.3] | -0.4 | -0.7 | -0.7 |
| exact-ranking/sub-pel early exits off (`rdo_max`) | -0.4 [-0.6, -0.3] | -0.4 | -0.4 | -0.3 |
| no depth removal (`planner.depth_removal_th_*` 0) | -0.4 [-0.7, -0.3] | -0.4 | -0.4 | -0.4 |
| inter TU split on every block (`hevc.inter_tu_split_spread` 1) | -0.3 [-0.6, -0.2] | -0.6 | -0.3 | -0.6 |
| motion search maximal | -0.1 | | | |
| hierarchical P (3 layers, adaptive) | +2.3 [-1.5, +6.6] | +2.2 | +3.3 | +0.5 |
| 8x8 inter leaves | +0.6 | | | |
| QP cadence off at fixed QP | +6.4 | | | |
| SAO off / deblocking off in P / inter TU split off / trellis off | +7.7 / +6.5 / +1.0 / +2.3 | | | |
| chroma policy (JVET map) | +0.4 (PSNR +2.7, MS-SSIM -5.4) | | | |

HEVC's search levers add up to about 2 % J. Default is already level with x265 medium psnr. x265 medium's
own extra tools are 3 references, 3 merge candidates and AMP. The second reference is not supported in our
HEVC backend yet.

### VVC (n = 14-16)

| Change | J [90 % CI] | PSNR | VMAF | MS-SSIM |
|---|---|---:|---:|---:|
| affine (`vvc.affine` 1) | -0.7 [-1.2, -0.2] | -1.2 | -0.1 | -1.3 |
| hierarchical P (3 layers, adaptive) | -0.7 [-3.6, +2.1] | -0.6 | +1.9 | -2.7 |
| 8x8 inter leaves | -0.6 [-1.3, -0.1] | -0.7 | -1.4 | -0.1 |
| 64-point transform (`vvc.tx64`) | -0.4 [-1.0, +0.3] | | | |
| trellis 8, rdo_max, me_max, depth_keep, full P intra, sub-pel walk | -0.1 to +0.1 | | | |
| MTS | +1.1 [+0.7, +1.5] | | | |
| MTT depth 1 / 2 | +3.3 / +3.0 | | | |
| dependent quantisation | +5.0 [+4.3, +5.7] | +5.2 | +2.0 | +6.7 |
| chroma policy | +0.5 (PSNR +2.4, MS-SSIM -3.5) | | | |

VVC's search is already saturated: none of the shared search levers moves it. Its unused normative
tools (MTT, MTS, dependent quantisation) all lose, as `docs/handoff/vvc.md` found. There they were
diagnosed as a symptom of the engine's partition and mode-decision RD (VVenC's MTT gains 4-6 % where ours
loses 3-6 %, and DQ's rate model is a template approximation). They are not levers until that is fixed.

**Coded split decisions (tried, rejected).** `planner.split_rd`: the VVC backend's luma RD cost
(`inter_leaf_cost`, the estimate `planner.rect_rd` uses) decides quad splits that the variance model finds
marginal. Result: band 25 % +1.1 J [+0.6, +1.5], band 100 % +2.1 J [+1.5, +2.7]. The coded estimate (luma
only, neighbour state at planning time) splits more than the tuned variance model (`planner.lambda_scale`
4 for VVC) and loses. The code was dropped. A better partition RD needs chroma and the real mode decision,
not this proxy.

### AV1, VP9, AV2

Being re-measured after the VMAFx server change (queue `~/nanocodec-data/work/fixedqp/queue.txt`).

## 3. Phase 2 so far: costs of the gains (matched QP, same 16 clips; x86 callgrind P-frame instructions)

CPU is `nceval.fixedqp cost`: callgrind, key frame plus 12 P frames on `tune.COST_CLIPS`, P-frame
geometric mean, relative to the Default.

| Candidate | J [90 % CI] | PSNR | VMAF | MS-SSIM | x86 P instr. | J per +1 % CPU |
|---|---|---:|---:|---:|---:|---:|
| H.264 motion search maximal (`analysis.*` early exits off) | -1.0 [-1.9, -0.2] | -0.9 | -1.0 | -1.1 | +2.1 % | 0.48 |
| H.264 more intra trials (`decision.intra_gate_variance` 16) | -0.3 [-0.4, -0.1] | -0.4 | 0.0 | -0.8 | 0.0 % | free |
| H.264 partition search everywhere | -0.7 [-1.1, -0.4] | -1.8 | -0.4 | -0.9 | +17.8 % | 0.04 |
| H.264 RD transform size | -0.5 [-0.7, -0.3] | -1.3 | -0.6 | -0.3 | +21.2 % | 0.02 |
| H.264 trellis 8 states | -0.2 | | | | +32.9 % | 0.01 |
| H.264 all five | -2.4 [-3.1, -1.8] | -4.1 | -1.8 | -3.1 | +73.2 % | 0.03 |
| HEVC `hevc.merge_candidates` 3 / 4 / 5 | -1.3 / -1.7 / -1.7 | | | | +1.1 / +2.2 / +2.7 % | 1.2 / 0.77 / 0.63 |
| HEVC no depth removal | -0.4 [-0.7, -0.3] | -0.4 | -0.4 | -0.4 | 0.0 % | free |
| HEVC full P-frame intra search | -0.5 [-0.7, -0.3] | | | | +5.9 % | 0.08 |
| HEVC rdo_max (exact-ranking and sub-pel early exits off) | -0.4 [-0.6, -0.3] | | | | +5.9 % | 0.07 |
| HEVC TU split spread 1 | -0.3 [-0.5, -0.1] | | | | +9.0 % | 0.03 |
| HEVC trellis 8 states | -0.5 | | | | +78.0 % | 0.01 |
| HEVC all five search levers | -2.3 [-2.8, -1.9] | -2.4 | -2.4 | -2.5 | +103 % | 0.02 |
| VVC MTT 1 / 2 with `planner.rect_rd` | +1.3 / +0.9 | | | | | worse |

At the Default's exchange rate (k 0.1, so 1 % CPU is worth 0.1 % J), the candidates that pay are
H.264's motion-search early exits off plus the wider intra gate (about -1.3 J for +2 %), and HEVC's
four-entry merge list plus no depth removal (about -2.1 J for +2.2 %). Both went to the adoption screen
(`nceval.screen --clips adopt`, k 0.1, with dcpu). The rest are tuner arms at best: they gain at fixed QP
but cost 6-100 % CPU.

`hevc.merge_candidates` (new, default 2, so streams are unchanged) sets `MaxNumMergeCand` (HEVC
§7.4.7.1). The merge list then carries the remaining spatial candidates and TMVP. It is the cheap HEVC
form of VVC's adopted six-candidate list. Merge 1, 3 and 5 decode bit-exactly in HM, ffmpeg and our
decoder (`landing_conformance.py --param`, 21/21 streams each). It landed off in dbf385e3.

## 4. Ranked list

| # | Tool | Est. BD-J (fixed QP) | CPU (x86) | Codecs | Status |
|---|---|---|---|---|---|
| 1 | HEVC merge list 4 + no depth removal | -2.1 | +2.2 % x86, +1.7 % Jetson cycles | HEVC (Fast too) | adoption screen |
| 2 | H.264 motion-search early exits off + intra gate 16 | ~ -1.3 | +2.1 % x86, +2.8 % Jetson cycles | H.264 | **adopted** (screen L3 BETTER, J -0.17 [-0.33, -0.02], n 126, no guard) |
| 3 | Second reference ported to HEVC/VVC/AV2 (backend signalling only; the engine feature exists) | AV1 -5 / VP9 -4 on the lens; tuner, CBR: AV1 -2.2 J at +17.6 % Jetson | high | HEVC, VVC, AV2 | next code work; design for CPU |
| 4 | Expensive search levers (partition search, RD transform size, trellis 8, full P intra, TU spread) | -0.2 to -0.7 each | +6 to +78 % | H.264, HEVC | tuner arms, not Default |
| 5 | VVC partition/mode-decision RD (needed before MTT/MTS/DQ can pay) | MTT +1 to +3 now; VVenC's MTT -4..-6 | high | VVC (engine: all) | research |
| 6 | Fast presets toward Default | 11-18 | the designed CPU trade | all Fast | tuner |

Not levers at fixed QP on the pool: hierarchical P for HEVC (+2.3) and VVC (-0.7, ambiguous), the chroma
policy (J-neutral), λ moves, the planner's rectangle RD for H.264/HEVC (no effect), and coded split
decisions (above).
