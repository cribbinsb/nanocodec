# Porting HEVC's VC-1K wins to the other codecs (2026-11-19)

HEVC landed three changes on 2026-11-19 ([hevc.md](../handoff/hevc.md), the VC-1K sections): the
TU16 chroma heuristic off, deblocking offsets +2 / +2, and whole-CU pricing of close-call intra
leaves (`decision.cu_price_intra_lambdas`, engine `cu_pricing.rs`). This note records where the
same pattern exists in H.264, VVC, AV1, VP9 and AV2, what was built, and what it measured.
Branch `port-hevc-wins`; scratch, logs and sets in `~/nanocodec-data/work/port-hevc-wins/`.

Ranking: `nceval.fixedqp ab --clipset` on each codec's six worst VC-1K clips
(`work/vc1k-harvest/vc1k-worst6-<codec>.json`, H.264 `work/rd-depth/vc1k-worst6.json`), matched
QP, J % against the same build's Default, 90 % interval. These rank ideas only; adoption is the
pool screen.

## 1. Stronger deblocking

New parameters (all default 0, so streams are unchanged until a preset sets them):
- `h264.deblock_alpha_offset`, `h264.deblock_beta_offset`: `slice_alpha_c0_offset_div2` /
  `slice_beta_offset_div2` (§7.4.3); the slice header and the shared filter already carried them.
- `vvc.deblock_beta_offset`, `vvc.deblock_tc_offset`: `pps_luma_beta_offset_div2` /
  `pps_luma_tc_offset_div2` (§7.4.3.5; chroma inferred equal). The encoder's filter
  (`filter.rs` `Offsets`) indexes β and tC at the offset QP; our own fast decoder now reads the
  two varying PPS fields (block QP, offsets) and requires the rest to be the PPS our encoder writes.
- `av2.deblock_delta`: the frame header's two-bit per-channel deltas (−2..1, units of
  `DF_DELTA_SCALE` 8 qindex) on every filtered channel.
- AV1 and VP9 already had a level multiplier (`av1.loop_filter_level_scale`,
  `vp9.filter_level_scale`).

Conformance (`tools/ci/landing_conformance.py --param ...`): VVC β +2 / tC −3 21/21, H.264
α +2 / β −3 33/33, AV2 delta +1 23/23 bit-exact in every decoder.

| codec | candidate | worst six J [90 %] | PSNR | VMAF | MS-SSIM |
|---|---|---:|---:|---:|---:|
| H.264 | α/β +1 / +1 | +0.1 [-0.1, +0.4] | -0.3 | +0.4 | +0.1 |
| H.264 | α/β +2 / +2 | +0.2 [-0.1, +0.5] | -0.4 | +0.9 | -0.1 |
| VVC | β/tC +1 / +1 | -0.9 [-1.3, -0.5] | -1.3 | -1.2 | -0.3 |
| VVC | β/tC +2 / +2 | -1.7 [-2.2, -1.2] | -2.8 | -0.6 | -1.3 |
| AV1 | level × 1.25 | -0.4 [-0.7, -0.1] | -0.6 | -0.1 | -0.4 |
| AV1 | level × 1.5 | -0.7 [-1.1, -0.3] | -1.3 | -0.1 | -0.6 |
| AV1 | level × 2 | -0.7 [-1.3, +0.0] | -1.6 | +0.1 | -0.7 |
| AV1 | × 1.5, sharpness 2 | +0.4 [+0.2, +0.8] | +0.5 | +0.5 | +0.1 |
| VP9 | level × 1.25 | -1.6 [-2.1, -1.1] | -1.6 | -1.5 | -1.5 |
| VP9 | level × 1.5 | -1.3 [-2.0, -0.7] | -1.5 | -1.0 | -1.5 |
| AV2 | delta +1 | -0.5 [-1.0, +0.1] | -0.9 | +0.1 | -0.9 |

Pool (CBR, k 0.1, draw 2026-11-19, same binary with the parameter against without):
- VP9 level × 1.25: L1 +0.57 % [-0.27, +1.50], L2 +0.45 % [+0.19, +0.71]: worse; rejected.
- VVC +2 / +2: L1 -0.08 % [-0.83, +0.67], L2 +0.22 % [-0.04, +0.47]; stopped before L3 once HEVC's
  +2 / +2 had failed its final-check and was reverted (784ba2ade); rejected.
- H.264 (no gain on the six clips) rejected; AV1 × 1.5 and AV2 +1 held, not screened.

Every parameter stays at 0 / 1 in every preset: the VC-1K gains did not carry to the pool.

## 2. Transform-size caps and biases

HEVC's lesson: a heuristic that forced 16x16 inter transforms where an RD tree would keep 32x32
cost J on flat phone video. Where each codec stands:
- H.264: inter macroblocks with luma always take the 8x8 transform (the larger); 4x4 only by RD
  (`h264.transform_size_rd`, Slow). Nothing biases smaller.
- VVC: a CU is one transform unit (implicit split only past `MaxTbSizeY`); no inter transform
  tree, so no split heuristic. The one cap is `vvc.tx64` 0 (64x64 CUs as four 32x32).
- AV1: the inter split is an RD trial (`av1.inter_tx_split`, energy gate 0.25 λ, biasing to the
  larger size); `av1.ceiling_tx_split` splits 64x64 only past qindex 255 (λ scale ≥ 2), never at
  fixed QP; `av1.inter_tx_split_concentration` is off.
- AV2: RD split with a minimum-levels gate (`av2.inter_split_min_nonzero` 4, biasing larger).
- VP9: Default searches up to 32x32 by RD; Veryfast's inter cap was lifted on 2026-11-18.

**VVC `vvc.tx64` 1 in the Default** (a 64x64 CU as one 64-point transform unit of its 32x32
lowest frequencies, not four 32x32): the cap HEVC's lesson points at.
- Worst six fixed QP: J -3.5 [-5.8, -1.1] (PSNR -3.0, VMAF -4.8, MS-SSIM +0.1).
- Instructions (gate, QP 36, 30 f; Johnny / KristenAndSara / WestWindEasy): -5.9 / -5.6 / -0.3 %.
- Pool (CBR, k 0.1, dCPU -3.9, draw 2026-11-19, `bin` with the parameter against without): L1 -2.31 %
  [-3.34, -1.33], L2 -1.47 % [-2.11, -0.87], L3 (ladder, n 108) -1.18 % [-1.55, -0.79] BETTER, no
  guard. Quality alone -0.55 % balanced: videocall -2.2, pan -1.7, lowlight -1.1, drone -0.8,
  bodycam -0.5, crowd +0.1, **static +2.3 %**: five static clips lose 4-14 %
  (`pool_static_chirla_chirla_seq_024_camera_7_2023` +14.4, `pool_static_chirla_s024c2_360p` +14.4,
  `pool_static_london_bus_540p` +13.9, `pool_static_cevo_ukof_ld_indoor_light_ohcam_0` +10.5,
  `pool_meva_hospital_g301` +7.1, PSNR-YUV +14..+23 %). `bodycam_airport` (the 2026-11-11 loser) is
  not in the draw. A 60-frame CBR 300 kbps encode of the local `static_london_bus` 720p clip did not
  reproduce it (key 32.49 / 32.48 dB, P mean 33.78 / 33.81 dB): cause not known yet.
- Keys every 2 s (`pulsecheck.py`, 6 static + 6 moving, 600 f): pulse -0.04 dB, worst-1 % Y BD
  -4.9 %, PSNR-YUV BD +0.49 %, VMAF BD -0.7 %.
- final-check (`nceval.levels final-check`, 36 clips, 600 f, keys every 10 s) against 7d0d75b5e: J
  -0.21 % [-0.74, +0.40] AMBIGUOUS, static class guard +4.52 % (> 3): `pool_static_chirla_chirla_seq_024`
  +26.2 %, `pool_meva_hospital_g301` +19.8 % with the first key 2.0 dB and later keys 2.1 dB worse.
  The handoff's note on `vvc.tx64` names the cause measured on 2026-11-11: the approximate 64-point
  forward (2x2 means into a 32-point DCT) costs 0.4-0.6 dB at QP 25-32. An exact forward (2026-11-19,
  [vvc.md](../handoff/vvc.md) "Exact 64-point forward") did not remove these key losses: the
  approximation was not their cause.

**Landed: `vvc.tx64` 1 with `vvc.tx64_min_qp` 34** (new: pictures finer than QP 34 plan no leaf above
32x32, `VvcBackend::max_leaf_log2`).
- final-check: min QP 38 J -0.48 % [-0.69, -0.29] WORSE on the periodic-key pulse alone (+0.66 dB on
  `pool_static_chirla_chirla_seq_024`, J +0.2 %); **min QP 34 J -0.75 % [-1.33, -0.18] BETTER, no
  guard**: static +0.1 %, lowlight -1.7, drone -1.3, pan -0.7, bodycam -0.6, crowd +0.1; dKey1 -0.02
  dB, dKeyN -0.05 dB.
- Pool levels (the gated build with `vvc.tx64` 0 as anchor, k 0.1, dCPU -3.9): L1 -2.43 % [-3.60,
  -1.30], L2 -1.44 % [-1.96, -0.99], L3 (single, n 108) -0.86 % [-1.42, -0.32]; quality -0.47 %
  balanced (videocall -3.1, pan -1.4, lowlight -1.2, bodycam -0.3, crowd -0.3, static +0.5, drone
  +2.6; worst `fit_static_otw399` +12.7 %, `pool_static_london_bus_540p` +7.7 %,
  `pool_otw_homes_00004` +7.6 %). Verdict WORSE on two guards, judged marginal and not a block: the
  long-window pulse +0.27 dB against +0.25, and the rate ceiling 0.48 % mean over target against
  the anchor's 0.17 % + 0.25. The 600-frame final-check, which has the longer windows, tripped
  neither.
- Keys every 2 s (12 clips; `pulsecheck.py` derives the candidate from the anchor, so the candidate
  here carries `--param vvc.tx64=1` over the anchor's `vvc.tx64=0`): pulse +0.01 dB, worst-1 % Y
  -3.6 %, PSNR-YUV BD +0.14 %, VMAF BD +0.10 %.
- Instructions (gate, 30 f): -5.9 / -5.6 / -0.3 % at QP 36, -0.3 / -0.6 / -1.1 % at QP 30.
- `preset_identity`: only VVC medium (Default) streams change (9 of 27); VVC Fast pins `vvc.tx64` 0,
  Slow now pins it 0 (not screened). Conformance 21/21.
- Open: the static clips that still lose at 150 frames (`fit_static_otw399` +12.7 %); an exact
  64-point forward would remove the reason for the QP gate.

Measurement pitfalls met on the way (both in the eval tooling, fixed the same day):
- `nc-remote` builds carried the worktree's HEAD as a clean commit stamp, and the result caches key a
  clean build on its commit (`buildinfo.stream_identity`): two builds of uncommitted variants on one
  HEAD shared their cached points. Now stamped `<commit>-dirty.<diff hash>` (`tools/dev/build_stamp.py`).
- `work/intra-p-ranking/pulsecheck.py` built the candidate from the anchor after the anchor's
  `--anchor-param`s were applied, so with `--anchor-param` alone both arms were the same. The tool
  is now `tools/dev/pulsecheck.py` (arms built independently, identical arms refused).

## 3. Whole-CU pricing for VVC

`CodecBackend::begin_cu_trial` / `cu_trial_cost` / `undo_cu_trial` / `end_cu_trial` for VVC
(`crates/nc-vvc/src/backend/cu_trial.rs`). A trial notes the CU's reconstruction (three planes),
its motion-grid cells, the HMVP history list and affine models (`MotionGrid::save_cu` /
`restore_cu`), the item list, the quantization group, the cached predictors and the P-picture
intra count; the cached luma prediction is dropped at begin and undo. The price is the three
planes' squared error (chroma weighted by λ / λ_chroma) plus λ times the CU's bits, priced by
the slice writer's own syntax walk: `syntax::block` is now generic over `coeff::Bins`, so the
same walk writes (CABAC) or prices (`BitCost`, at the slice's initial contexts). Only square CUs
are tried (rectangular ones come only from `vvc.mtt`, off). `vvc.trial_full_intra` 1 gives intra
CUs coded in a trial the key pictures' direction search (HEVC's `hevc.trial_full_intra`).

Conformance with `decision.cu_price_intra_lambdas` 0.1 (and with `vvc.trial_full_intra` 1):
21/21 bit-exact. Instructions (gate build, 30 frames, QP 36; Johnny / KristenAndSara /
WestWindEasy 720p): +0.00 / +0.06 / +1.16 %.

| candidate | worst six J [90 %] | PSNR | VMAF | MS-SSIM |
|---|---:|---:|---:|---:|
| intra leaves 0.05 | -0.3 [-0.5, -0.0] | -0.0 | -0.4 | +0.0 |
| intra leaves 0.1 | -0.2 [-0.4, +0.1] | +0.0 | -0.6 | +0.1 |
| intra leaves 0.2 | -0.4 [-0.7, -0.1] | -0.1 | -0.9 | -0.1 |

AV1 was not built: its encoder writes each block's symbols into the range coder as it codes and
adapts the CDFs at once, so a trial must snapshot and restore the writer, the whole CDF context,
the mode-info, CDEF-map, coefficient and transform-size context grids, the segment, delta-q and
CDEF-index state. Left in the ideas ledger.
