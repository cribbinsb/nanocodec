# Key-frame coding efficiency: headroom and a key-frame RD search (2026-11-21)

**Question (owner).** Under the 0.6.1 buffer rule the key frame's bits are scarce and the frames after
it inherit its quality (`frame-allocation-2026-11-21.md` sections 5-7). Does more compute on the key
frame buy quality at equal bits? The earlier test could only switch on our slowest preset's intra
tools (2.5-9 % for 1.3-2.4x the key's time).

**Answer.**

- **Our key is 10-18 % more expensive than strong intra encoders of the same standard** at these
  sizes: AV1 17-18 % (SVT-AV1 still-picture M0-M2), HEVC 12-13 % (x265 slow to placebo), H.264
  10-11 % (x264 slow).
- **Compute pays in the encoders that decide by coded cost**: SVT-AV1's still mode gains 14 % from M8
  to M4 for 12x its CPU, then 6-7 % more from M4 to M0 for another 20x. x264 and x265 stop gaining at
  `slow`.
- **Partition and transform search carry it, not extra prediction tools** (libaom's own switches).
- **No codec of ours decided a key partition by coded cost.** The key-frame RD partition search
  (`planner.key_rd_split`, the P-frame search's driver on key nodes) makes the key 3.6-10.4 % cheaper
  at equal quality for about 2-3x the key's time.

Nothing here changes a stream: every lever is default off.

## Method

`python3 -m nceval.fixedqp gap|ab --frames 1 --key-share 0.15` (agent key-study's harness): the first
frame of each clip of the quick level of draw `fixedqp-2026-11-21` (24 clips: crowd 4, lowlight 4, pan
4, static 3, bodycam 3, videocall 3, texture 2, drone 1), coded as a key frame at four quantizers
spanning keys of 0.15 of a one-second buffer at each ladder rate. BD-J is the project's J (PSNR-YUV,
VMAF v1, MS-SSIM in thirds; mean of the full and lower-half BD-rates), class-balanced, with a
bootstrap interval. Negative: the test is cheaper. Ours are scored from the encoder's reconstruction,
references from a standard decoder. Fixed sets and this draw rank ideas; nothing here is adoption
evidence.

Added for this study:

| Addition | What it does |
|---|---|
| `libaom-av1-cpuN~allintra`, `~good`, `+option=value` | libaom `--allintra` / `--good` at speed N, with any aomenc option |
| `svtav1-mN~still`, `~ra`, `+option=value` | SVT-AV1 without `--rtc`: still-picture mode (`--avif 1`) or its default settings |
| `cpu_s` in the result | the encoder process's CPU seconds per curve point (its own exit status) |
| `nceval/infosei.py` | x264's and x265's information SEI (0.6 kB and more) is dropped from one-frame streams: it was a large share of a 3-30 kB key and made x265 look 36 % worse than our HEVC |
| `eval/keyrdo/headroom.py` | the tables below from a result file |

CPU seconds are whole one-frame processes on a busy 96-vCPU c7i box (mixed 360p-1080p frames): ratios
within one run, not timings.

**Key sizes this small sit at the quantizer ceiling.** At 0.15 buffer the lowest goal is below what the
coarsest quantizer codes on 18 of 24 clips for our AV1 (SVT 16, libaom 22 of 24): BD-rates compare the
part of the curves that overlaps, mostly the upper three rungs.

## 1. Headroom against strong intra encoders

Against our medium key of the same standard (build 55aac741 = `variance-integration` cd723bbc5).

| AV1 | J % [interval] | PSNR-YUV / VMAF v1 / MS-SSIM | CPU s per key |
|---|---|---|---|
| ours `slower` | -5.8 [-6.1, -5.3] | -4.5 / -8.1 / -4.3 | 0.085 |
| SVT-AV1 still M8 | +1.9 [+0.6, +3.4] | +6.8 / +1.3 / -1.8 | 0.019 |
| SVT-AV1 still M6 | -1.2 [-3.1, +0.4] | +3.6 / +0.5 / -8.5 | 0.058 |
| SVT-AV1 still M4 | -12.0 [-13.8, -10.4] | -8.4 / -9.9 / -17.9 | 0.24 |
| SVT-AV1 still M2 | -17.3 [-19.3, -15.7] | -13.7 / -14.5 / -23.0 | 1.29 |
| SVT-AV1 still M0 | -18.4 [-20.6, -16.6] | -15.4 / -14.4 / -25.3 | 4.98 |
| libaom allintra cpu 6 | +5.1 [+0.4, +8.2] | +14.9 / +3.0 / -10.5 | 0.16 |
| libaom allintra cpu 4 | -8.9 [-11.7, -6.5] | -1.5 / -8.5 / -18.4 | 0.95 |
| libaom allintra cpu 2 | -7.9 [-11.0, -5.5] | -1.9 / -7.2 / -17.9 | 1.91 |
| libaom allintra cpu 0 | -10.6 [-13.9, -8.4] | -5.0 / -10.1 / -18.6 | 6.21 |
| libaom good cpu 6 | -5.4 [-8.0, -3.4] | -0.1 / -1.7 / -16.7 | 0.33 |
| libaom good cpu 2 | -14.6 [-17.4, -12.5] | -8.9 / -10.8 / -24.9 | 2.28 |
| libaom allintra cpu 4, `tune=ssim` | -11.4 [-13.6, -9.6] | -2.9 / -6.4 / -26.4 | 0.85 |
| libaom allintra cpu 4, `tune=iq` (n 17) | -13.8 [-15.1, -12.5] | -5.2 / -12.4 / -25.3 | 1.06 |
| libaom allintra cpu 4, `deltaq-mode=3` | -5.0 [-7.8, -2.8] | +1.1 / -4.9 / -14.9 | 0.89 |
| SVT-AV1 still M4, `tune=2` / `enable-qm=1` | -11.5 / -12.0 | | 0.26 / 0.28 |

| H.264 / HEVC | J % [interval] | PSNR-YUV / VMAF v1 / MS-SSIM | CPU s per key |
|---|---|---|---|
| ours H.264 medium / `slower` | 0 / -3.2 [-4.0, -2.5] | -3.8 / -4.1 / -4.4 | 0.019 / 0.078 |
| x264 medium `--tune psnr` | -7.9 [-9.9, -6.7] | -0.8 / -13.1 / -11.2 | 0.014 |
| x264 slow | -10.7 [-12.4, -9.7] | -4.5 / -14.7 / -13.4 | 0.027 |
| x264 veryslow | -10.4 [-12.3, -9.3] | -4.5 / -14.6 / -13.1 | 0.028 |
| x264 veryslow, psy defaults (n 17) | +3.6 [+0.5, +5.4] | +8.5 / -2.7 / -0.5 | 0.029 |
| ours HEVC medium / `slower` | 0 / -1.5 [-2.8, -0.4] | -1.2 / +5.3 / -8.0 | 0.030 / 0.061 |
| x265 medium `--tune psnr` | -9.3 [-11.0, -7.8] | -8.0 / -0.4 / -17.2 | 0.17 |
| x265 slow | -12.2 [-13.7, -10.7] | -11.0 / -3.1 / -20.4 | 0.24 |
| x265 veryslow | -12.2 [-13.9, -10.7] | -11.7 / -2.4 / -20.8 | 0.29 |
| x265 placebo | -12.6 [-14.1, -11.1] | -11.8 / -2.9 / -20.8 | 0.34 |

Ours, `slower` against medium, the other codecs: VP9 -2.0 [-2.1, -1.7], VVC -1.1 [-1.5, -0.7], AV2 -0.6
[-0.9, -0.3].

- H.264's gap is efficiency, not compute: x264 medium is 7.9 % cheaper at 0.7x our key's CPU.
- With fixed QP, x264 and x265 run no adaptive quantisation, so `--tune ssim` equals `--tune psnr`.
- Not measured: VVC, VP9 and AV2 references (VVenC and vpxenc are not in the stageable anchors
  directory; AVM was not run). The splice of a reference's key into our P-frame path (inheritance) was
  not done: it needs a reconstruction import.

## 2. Where the references' gain sits

libaom `--allintra --cpu-used=3`, 12 clips, one tool switched off (J % lost):

| Switch | J % [interval] | CPU s (base 1.96) |
|---|---|---|
| `enable-rect-partitions=0` (also removes AB and 4-way) | +25.5 [+21.6, +27.8] | 0.45 |
| `enable-directional-intra=0` | +10.5 [+8.6, +11.5] | 0.86 |
| `enable-angle-delta=0` | +6.5 [+5.6, +7.0] | 1.08 |
| `enable-tx-size-search=0` | +5.1 [+4.0, +6.5] | 0.93 |
| `use-intra-default-tx-only=1` | +3.0 [+1.9, +4.3] | 1.08 |
| `enable-rect-tx=0` | +2.7 [+2.0, +3.5] | 1.00 |
| `max-partition-size=32` | +2.7 [+1.5, +3.4] | 1.33 |
| `use-intra-dct-only=1` | +2.3 [+1.6, +3.3] | 1.06 |
| `enable-smooth-intra=0` | +1.0 [+0.5, +1.6] | 1.81 |
| `enable-1to4-partitions=0` | +0.8 [+0.3, +1.1] | 1.17 |
| `enable-tx64=0` | +0.6 [+0.1, +1.1] | 1.51 |
| `enable-restoration=0` | +0.3 [+0.2, +0.4] | 1.41 |
| palette, CfL, filter intra, quant matrices, chroma delta-q, flip/identity transforms, AB partitions | within +-0.5 | |
| `enable-intra-edge-filter=0` | -0.7 [-1.4, -0.3] | 1.47 |
| `sb-size=64` | -0.7 [-1.1, -0.2] | 1.44 |
| `min-partition-size=8` | -1.2 [-1.7, -0.6] | 1.20 |
| `sharpness=2` | -1.4 [-2.3, -0.6] | 1.39 |
| `disable-trellis-quant=1` | -1.5 [-2.2, -0.8] (VMAF -4.3, MS-SSIM +1.2) | 1.24 |

`enable-cdef=0` gave identical streams. The rectangular-partition figure is at key sizes near libaom's
coarsest quantizer, and whether the switch also changes its partition pruning is not known: an upper
bound until our own number exists. At these rates trellis quantisation, 4x4 blocks and 128x128
superblocks cost J.

**Our side** (code read, 2026-11-21):

- Key partitions come from `key_plan.rs`: a DC prediction from source neighbours and one transform per
  candidate leaf (luma only in HEVC and VVC), on a quad tree. `planner.rd_split` was inter-only.
- AV1 keys: square blocks 8-64, one DCT_DCT the size of the block, about ten modes priced by a
  transform-domain distortion with empty neighbour contexts, chroma DC or CfL.
- HEVC medium picks the luma mode by SATD (`hevc.intra_rd_candidates` 1); chroma always follows luma.
- H.264: I16x16, I8x8 and chroma modes by SATD; intra macroblocks are never trellised.
- Only VVC codes rectangular blocks, in P frames, default off (`vvc.mtt`).
- The ledger's "key RD search <= 0.5 %" (`key-frame-rd-partition-search`, HEVC medium) built no
  search: it was `hevc.intra_rd_candidates` 8 and `planner.intra_bias` 1150 / 1250 as a proxy.

## 3. The key-frame RD partition search (`planner.key_rd_split`)

`rd_split.rs` on key nodes: around the key plan, each node codes the leaf and the quad split inside a
depth window (`planner.key_rd_deeper`, `key_rd_shallower`) through the real leaf path and keeps the
cheaper by coded cost (node trials: all planes, reconstruction error, entropy-coded bits). AV1, HEVC,
VP9, AV2 and VVC; H.264 has no tree. Forced on, landing conformance: 606 of 606 streams bit-exact in
every decoder. Off: 387 of 387 preset streams identical.

One frame, 24 clips, against each codec's medium key, BD-J % [interval]:

| Codec | window 1/1 | 2/2 | 3/3 | 2/2 with the slower rung's intra settings (those alone) |
|---|---|---|---|---|
| VVC | -8.9 [-9.7, -8.0] | -10.4 [-11.4, -9.1] | -10.9 [-11.7, -9.7] | |
| HEVC | -6.6 [-7.3, -5.7] | -8.2 [-8.8, -7.3] | -7.9 [-8.4, -7.0] | -10.0 [-10.7, -9.3] (-2.7) |
| VP9 | -5.9 [-6.2, -5.6] | -7.5 [-7.9, -6.9] | -7.7 [-8.3, -7.0] | |
| AV2 | -5.8 [-6.4, -5.3] | -6.6 [-7.4, -6.0] | -6.6 [-7.4, -6.1] | |
| AV1 | -2.7 [-3.2, -2.3] | -3.6 [-4.2, -3.1] | -3.7 [-4.2, -3.2] | -10.2 [-10.6, -9.6] (-5.9) |

The three metrics agree (HEVC 2/2: PSNR-YUV -8.8, VMAF v1 -7.6, MS-SSIM -6.2). Window 2/2 is the knee.
In AV1 the search is worth three times more with a better mode decision under it (8 RD candidates,
angle deltas 2, key trellis with 4 states).

Candidate sets: `eval/keyrdo/krd-<codec>.txt`.

## 4. The key-frame leaf search (`decision.intra_rd_modes`, `intra_rd_chroma`)

`leaf_rd.rs`: the backend declares what else it could code a key leaf with
(`CodecBackend::intra_alternatives`: next-ranked luma modes, transform layouts, chroma modes); each is
coded through the real leaf path in a node trial and the cheapest kept, one axis at a time. AV1 offers
its luma ranking; HEVC its SATD ranking and the four explicit chroma modes (its chroma always followed
luma). No backend offers a transform axis yet. Forced on with the partition search: 291 of 291 AV1 and
HEVC streams bit-exact in every decoder; off, 387 of 387 preset streams identical.

One frame, 24 clips, against the medium key. Key CPU is the whole one-frame process against medium's
(49 ms HEVC, 63 ms AV1, start-up and file reading included, so the ratios understate the encode's):

| HEVC | J % [interval] | PSNR-YUV / VMAF v1 / MS-SSIM | key CPU |
|---|---|---|---|
| 2 luma alternatives | -4.9 [-5.3, -4.6] | -5.1 / -4.2 / -4.1 | x1.1 |
| 4 | -6.5 [-7.0, -6.1] | -7.0 / -5.5 / -5.7 | x1.6 |
| 8 | -7.8 [-8.3, -7.3] | -8.6 / -6.0 / -7.0 | x2.5 |
| chroma modes | -1.3 [-1.8, -0.9] | -2.6 / +0.2 / -0.9 | x1.5 |
| partition search 2/2 | -8.2 [-8.8, -7.3] | -8.8 / -7.6 / -6.2 | x1.4 |
| partition search + 4 | -12.5 [-13.0, -11.6] | -13.8 / -10.8 / -10.4 | x4.5 |
| partition search + 4 + chroma | -12.5 [-13.1, -11.6] | -14.5 / -10.4 / -10.2 | x7.1 |
| partition search + 4 + slower's intra settings | -14.0 [-14.5, -13.2] | -14.3 / -12.1 / -13.1 | x8.7 |
| partition search + 8 + chroma + slower's | -14.4 [-15.1, -13.6] | -15.9 / -11.1 / -13.8 | x19.9 |

| AV1 | J % [interval] | PSNR-YUV / VMAF v1 / MS-SSIM | key CPU |
|---|---|---|---|
| 2 / 4 / 8 luma alternatives | +0.1 / +0.2 / +0.2 | | x0.9 / x1.1 / x1.4 |
| partition search 2/2 | -3.6 [-4.2, -3.1] | -3.7 / -3.1 / -4.3 | x2.0 |
| partition search + 4 | -4.3 [-5.0, -3.7] | -4.2 / -3.7 / -4.9 | x3.7 |
| partition search + slower's key tools | -10.2 [-10.6, -9.6] | -9.3 / -12.2 / -8.6 | x2.6 |
| partition search + slower's key tools + 4 | -11.4 [-11.9, -10.7] | -10.9 / -13.5 / -9.4 | x5.3 |
| the same + 8 | -11.4 [-11.9, -10.7] | | x7.9 |

- HEVC with the two searches reaches x265 placebo's key (-12.6 % in section 1).
- AV1's transform-priced ranking is already good: coded cost over its next modes adds about 1 %. AV1
  stays 6 % short of SVT-AV1 still M2; what it cannot code on keys is in section 2 (transform size and
  type, rectangular blocks).
- Chroma by coded cost pays 1.3 % alone and nothing beside the luma searches.

Candidate sets: `eval/keyrdo/lrd-<codec>.txt`.

## 5. Existing settings, λ and the plan bias on a key

One frame, 24 clips, against the medium key (`eval/keyrdo/misc-<codec>.txt`):

| Codec | Setting | J % [interval] |
|---|---|---|
| AV1 | `quant.trellis_key` 1 | -5.0 [-5.4, -4.5] (VMAF v1 -7.6) |
| AV1 | `quant.trellis_key` 1, `trellis_states` 4 | -6.1 [-6.6, -5.6] |
| AV1 | `av1.keyframe_intra_rd_candidates` 8 / `keyframe_angle_deltas` 2 | +0.2 / 0.0 |
| AV1 | best set (search 2/2, slower's key tools, 4 alternatives) | -11.4 |
| AV1 | best, `av1.lambda_per_step_squared` 0.09 / 0.2 (0.135) | -12.1 / -9.6 |
| AV1 | best, `av1.intra_rate_scale` 0.7 / 1.4 | -11.6 / -11.3 |
| AV1 | best, `av1.restoration` 1 | -11.7 (PSNR-YUV -11.5 against -10.9) |
| AV1 | best, `planner.intra_bias` 950 / 1150 (1050) | -11.0 / -11.0 |
| HEVC | best set (search 2/2, 4 alternatives) | -12.5 |
| HEVC | best, `hevc.lambda_scale` 0.7 / 1.4 | -12.6 / -11.3 |
| HEVC | best, `planner.intra_bias` 950 / 1150 | -11.4 / -12.2 |
| HEVC | best, `hevc.sao_band` 1 | -12.5 |
| H.264 | `h264.trellis_i16x16` 1 / `trellis_i4x4` 1 / both | +0.9 / +1.5 / +1.9 |
| H.264 | slower's intra settings / with both trellises | -3.3 / -1.9 |
| VP9 | search 2/2 / slower's intra settings / both | -7.5 / -2.0 / -8.9 |
| VVC | search 2/2 / slower's intra settings / both | -10.4 / -1.3 / -11.6 |
| VVC | both, `vvc.isp` 1 / `vvc.mts` 1 | -11.6 (identical streams) / -10.4 |
| AV2 | search 2/2 / `av2.intra_rd_candidates` 4 / both | -6.6 / -1.3 / -7.2 |

- AV1's slower-rung key gain is the key trellis (off in AV1's Default), not the candidate counts.
- λ and the plan's bias are flat at equal bits on one frame. What a key λ does to the frames that
  inherit it is not measured by one frame.
- H.264's intra trellis loses at these sizes.

## 6. AV1 intra transform split (built, measured, removed)

`TX_MODE_SELECT` on key frames, an intra block coding four quarter-size luma transforms each predicted
from the one before (`tx_depth` 1), chosen per block by the leaf search (`decision.intra_rd_transforms`).
dav1d bit-exact; AV1 landing conformance passed forced on. One frame, 24 clips, against the medium key:

| Setting | J % [interval] |
|---|---|
| transform sizes signalled, never split | +0.4 [+0.4, +0.5] |
| split by coded cost | +0.1 [-0.3, +0.4] |
| partition search 2/2 + split | -4.0 [-4.6, -3.4] (search alone -3.6) |
| best set (section 5) + split | -11.2 [-11.7, -10.4] (best alone -11.4) |

The quad partition search already offers the smaller transforms, with their own modes; libaom's 5.1 %
for its transform-size search (section 2) does not carry to our structure. The code is not landed
(branch `key-rdo-tx-split`). The leaf search keeps its transform axis for a backend that has one.

## 7. Whole stream: the gain is not spent on the key

`python3 -m nceval.rc_oracle --arm ... --base-arm mid`, draw `fixedqp-2026-11-21`, 150 f (24 clips, one
key) and 600 f with 10 s keys (two clips per class, 6 scored: indicative). `mid` is the 0.6.1 buffer
rule of `frame-allocation-2026-11-21.md` (`rate.recode_overflow` 1.0, `rate.key_cap_buffer` 0.30).
Partition search 2/2 on mid, J % against mid; key times are wall ms from the rate logs of a shared box:

| Codec | 150 f | 600 f | on shipped, 150 f | key ms, 150 f | key's share of bits |
|---|---|---|---|---|---|
| AV1 | -0.62 | -0.14 | -0.39 | 100 -> 253 | 6.3 -> 6.2 % |
| HEVC | -0.82 | -0.08 | -0.59 | 78 -> 144 | 4.6 -> 4.3 % |
| VVC | -0.70 | -1.89 | -0.64 | 185 -> 581 | 5.0 -> 4.7 % |
| VP9 | -0.46 | -1.05 | -0.80 | 123 -> 374 | |
| AV2 | -0.55 | +0.25 | -1.30 | 121 -> 369 | |

AV1 with the slower rung's key tools as well (one-frame key -10.2 %): -0.01 % at 150 f, +0.69 % at 600 f.

**Why so little.** Rate control plans the key's quantizer from a size model that does not know the key
got cheaper. The key is coded at the same quantizer (`key_minus_level` unchanged) and comes out smaller:
its share of the stream's bits falls and the model's log2 error moves by the same amount (HEVC -0.22 ->
-0.33). The saved bits go to the P frames, where a bit buys less than on the key
(`frame-allocation-2026-11-21.md` section 7: 47 % more key bits is J -2.0 to -3.2 %).

## 8. Our front against the yardsticks (key bits at equal quality against key CPU)

One frame, 24 clips, J % against our medium key and CPU seconds per key (whole one-frame process on the
box; ours from the ratios of sections 3-4 on medium's 0.063 s AV1 and 0.030-0.049 s HEVC, which two runs
measured differently: ratios are the firmer figure).

| AV1: ours | J % | CPU s | SVT-AV1 still | J % | CPU s |
|---|---|---|---|---|---|
| medium | 0 | 0.063 | M8 | +1.9 | 0.019 |
| slower | -5.8 | 0.085 | M6 | -1.2 | 0.058 |
| partition search 2/2 | -3.6 | 0.12 | M4 | -12.0 | 0.24 |
| + slower's key tools | -10.2 | 0.17 | M2 | -17.3 | 1.29 |
| + 4 luma alternatives | -11.4 | 0.33 | M0 | -18.4 | 4.98 |

| HEVC: ours | J % | key CPU | x265 `--tune psnr` | J % | key CPU against our medium |
|---|---|---|---|---|---|
| medium | 0 | x1 | medium | -9.3 | x5.7 |
| slower | -1.5 | x2.0 | slow | -12.2 | x7.8 |
| partition search 2/2 | -8.2 | x1.4 | veryslow | -12.2 | x9.7 |
| 4 luma alternatives | -6.5 | x1.6 | placebo | -12.6 | x11.2 |
| both | -12.5 | x4.5 | | | |
| both + slower's intra settings | -14.0 | x8.7 | | | |

- **HEVC: our front is at or below x265's from x1.4 up**: the partition search alone is within 1 % of
  x265 medium at a quarter of its key CPU, and both searches match placebo at 40 % of it.
- **AV1: our front is above SVT-AV1's everywhere.** At our medium's CPU SVT (M6) is level; at 0.17-0.33
  s we are 1-2 % short of M4 (0.24 s); nothing of ours reaches M2. The missing 6 % is what our AV1 keys
  cannot code: rectangular blocks, transform types, 4x4 blocks, and at the cheap end SVT's staged mode
  decision (M8 codes our medium's quality in a third of its time).
- H.264: x264 medium is 7.9 % ahead at 0.7x our key CPU (section 1); H.264 has no tree for these
  searches. VP9, VVC, AV2: no same-standard yardstick was run.

**With the key put back on its budget** (`rate.key_recode_miss`, agent key-study: a key whose size
misses its budget by more than N octaves is coded again at the step its miss implies). Arms on mid:
the re-code alone, and the re-code with the best key set of sections 4-5 (HEVC: search 2/2 and 4 luma
alternatives; AV1: those and the key trellis, 8 candidates, angle deltas 2; VVC: search 2/2). J %
against mid; the last column is the best set against the re-code alone, the like-for-like figure:

| 150 f, 24 clips | re-code 0.25 / 0.5 | best + re-code 0.25 / 0.5 | best against re-code alone | key's share of bits (mid -> re-code -> best) |
|---|---|---|---|---|
| AV1 | +3.23 / +2.80 | +1.61 / +1.78 | -1.6 / -1.0 | 7.7 -> 5.6-5.9 -> 5.6 % |
| HEVC | +0.37 / +0.14 | -1.24 / -1.32 | -1.6 / -1.5 | 5.5 -> 5.4-5.5 -> 5.5-5.6 % |
| VVC | +0.88 / +0.87 | -0.50 / -0.37 | -1.4 / -1.2 | 6.0 -> 5.2-5.4 -> 5.3-5.5 % |

| 600 f, 10 s keys, 6 clips | re-code 0.25 / 0.5 | best + re-code 0.25 / 0.5 | best against re-code alone |
|---|---|---|---|
| AV1 | +4.46 / +4.23 | +2.21 / +1.69 | -2.2 / -2.5 |
| HEVC | +1.07 / +0.31 | -1.34 / -0.93 | -2.4 / -1.2 |
| VVC | +1.19 / +1.53 | +0.27 / +0.11 | -0.9 / -1.4 |
| VP9 | +3.29 / +3.09 | +1.53 / +1.74 | -1.8 / -1.4 |
| AV2 | +4.69 / +4.76 | +2.00 / +1.92 | -2.7 / -2.8 |

(The 600 f rows are from the run's printed reports; its 150 f VP9 and AV2 commands did not finish.)

- **Spent on the key, the searches are worth about 1.0-1.6 % of whole-stream J at 150 f and 1-3 % at
  10 s keys** (6 clips), against 0.5-0.8 % banked as fewer bits.
- **The re-code alone costs J on this draw**, most in AV1 (+2.8 to +4.5 %): mid's AV1 keys come out
  larger than their budget (size-model log2 bias +0.27) and the re-code shrinks them (key share 7.7 ->
  5.6 %). That is the key-size lever of `frame-allocation-2026-11-21.md` section 7 again, not the
  search; no buffer overflowed in any arm (fullness at most 0.80).
- Key time with both the searches and the re-code: 0.77-1.08 s against 0.08-0.21 s (wall, shared box),
  whole encode x1.03-1.20 at 150 f. A first pass that codes the plan and a searched second pass would
  halve it; not built.

## 9. Rectangular partitions in P frames (a reference's switch)

libaom `--good --cpu-used=6 --lag-in-frames=0`, fixed QP, 60 frames with one key, 8 clips of the same
draw: `--enable-rect-partitions=0` costs J +1.5 % [+0.7, +2.1] (PSNR-YUV +1.9, VMAF v1 +0.4, MS-SSIM
+2.0). One speed, one reference: the all-intra figure of section 2 (+25.5 % at key sizes, cpu 3) is 17
times that, so rectangular blocks are a key-frame tool first. Only VVC codes rectangular blocks today
(P frames, `vvc.mtt`, default off).

## 10. AV1 intra transform types on keys (built, landed default off)

What was missing (section 2): no forward ADST, FLIPADST or identity, and no type through AV1's
quantise / code / reconstruct path. Built:

- `nc-dsp/src/av1_forward.rs`, `Dsp::av1_forward`: libaom's forward of every size and type (ADST4/8/16,
  identity4-32, flips, 2:1 scale, 64-point zero-out), scalar; square `DCT_DCT` stays on the SIMD DCTs.
  Tested against the tuned DCTs (bit-exact) and against the normative inverse (round trip within 2).
- The type through the coding path: `coeff::TypeSyntax` (inter, or intra with its direction and type),
  the intra sets (`TX_SET_INTRA_1` / `_2`), the typed forward and inverse in `code_plane`. The encoder
  codes the five class-2D types (`DCT_DCT`, `ADST_DCT`, `DCT_ADST`, `ADST_ADST`, `IDTX`): they share
  `DCT_DCT`'s scan and contexts. Choice (`tx_type.rs`):
  - `av1.keyframe_mode_tx_type`: the type the luma mode implies (`Mode_To_Txfm`), no search;
  - the leaf search's transform axis (`decision.intra_rd_transforms`): the other types, each a whole
    leaf coding;
  - `av1.keyframe_tx_type_search`: the types ranked by a transform-domain estimate (forward, quantise,
    coefficient bits with the real contexts, the error the dequantised coefficients leave), as libaom's
    and SVT-AV1's intra type searches price them. It stops when the first type quantises to nothing.
- Forced on, landing conformance passed 242 of 242 AV1 streams in every decoder (each switch). Off,
  preset identity was 90 of 90.

One frame, 24 clips, against the medium key. Key CPU is relative to medium within one run, so compare
ratios within a block of rows:

| Setting | J % [interval] | PSNR-YUV / VMAF v1 / MS-SSIM | key CPU |
|---|---|---|---|
| mode-implied type | +0.7 [+0.5, +0.9] | +0.9 / +1.3 / +0.7 | x1.01 |
| leaf search, 4 types | -0.4 [-0.6, -0.2] | -1.0 / -0.1 / -0.2 | x1.71 |
| estimate (`keyframe_tx_type_search`) | -0.5 [-0.8, -0.3] | -1.3 / +0.2 / -0.6 | x1.11 |
| best set (section 5) | -11.4 [-11.9, -10.7] | -10.9 / -13.5 / -9.4 | x8.6 |
| best + mode-implied type | -12.5 [-13.0, -11.7] | -11.9 / -15.1 / -10.0 | (x1.08 best) |
| best + leaf search, 4 types | -13.1 [-13.7, -12.4] | -12.7 / -15.5 / -10.4 | x14.6 |
| best + estimate | -13.3 [-13.9, -12.5] | -13.0 / -16.2 / -10.5 | x11.4 |
| best + estimate + leaf search of its second | -13.5 [-14.1, -12.7] | -13.2 / -16.6 / -10.6 | x15.4 |

- The estimate beats a whole coding per type, at 1.32x the best set's key CPU (the leaf search of 4
  types costs 1.69x). That is libaom's +2.3 % for DCT only (section 2) recovered, about 2 %.
- The mode-implied type loses alone and wins inside the best set: it is a cheap partial of the search.
- The rest of the gap to SVT-AV1 still M2-M3 is rectangular blocks (section 2: +25.5 % when libaom
  turns them off).

Sets: `eval/keyrdo/txt-av1.txt`, `eval/keyrdo/tts-av1.txt`.

Also landed default off: `rate.key_plan_first` (the plan-only first key coding under
`rate.key_recode_miss`; conformance 242/242 AV1 and 49/49 HEVC forced on). It was not measured because
the re-code loses on every base of 2026-11-21.

## 11. Rectangular blocks on AV1 keys (built, landed default off)

`av1.key_rect_partitions` with `planner.key_rd_shapes` (on top of `planner.key_rd_split`): the RD
partition search also codes 16x16-64x64 key nodes inside the picture as two halves
(`PARTITION_HORZ` / `VERT`; the halves are at least 16x8, so no chroma block is below 4x4). Built
in landable steps, each identity-clean with the feature off (preset identity 90/90 against its
integration parent):

- One `TxSize` (`tx_size.rs`) and one `BlockSize` (`block_size.rs`) for encoder and decoder.
- Coefficient coding of the 2:1 rectangles: scans, `nz_map_ctx_offset` tables (libaom's sharing), eob
  alphabets 32/128/512, transform-type sets by `Tx_Size_Sqr` / `Sqr_Up`. Every square and 2:1 shape
  and type round-trips through the decoder's reader, and `cost` equals `write` on them.
- Mode info with a width and height, the loop filter's transform edges, mv-ref scans, partition and
  entropy contexts, node-trial snapshots of rectangles.
- Intra prediction of rectangles on the decoder's any-shape edges and kernels (`LumaIntra::block_edges`
  / `predict_block`), tested equal to the square path on squares for every mode, delta and filter
  mode; edge availability of the halves (`edge_extension`), tested against decoding order.
- `code_plane` on `w x h` transforms (DCT of rectangles through `Dsp::av1_forward`), the key trellis
  on rectangles (their default scans visit anti-diagonals in order, tested), CfL through square tiles
  (the SSE kernel is square), the luma mode by the transform-domain estimate over the best
  `keyframe_intra_rd_candidates` by prediction cost plus the winner's angle deltas, and the leaf
  search's luma and type alternatives on rectangles.
- Forced on (with the key trellis, CfL, the type estimate and leaf alternatives), landing conformance
  passed 242 of 242 AV1 streams in every decoder.

One frame, 24 clips, draw `fixedqp-2026-11-21`, against the medium key of this record's base (build
5f2f2555: the medium key before 3ebca4466 turned the key RD search on in medium). Candidates on head
38eeaf659; key CPU relative to that anchor within this run:

| Setting | J % [interval] | PSNR-YUV / VMAF v1 / MS-SSIM | key CPU |
|---|---|---|---|
| medium on head 38eeaf659 | -2.0 [-2.3, -1.6] | -2.0 / -1.6 / -2.4 | x1.5 |
| partition search 2/2 | -2.7 [-3.1, -2.2] | -2.9 / -2.1 / -3.3 | x2.0 |
| + rectangles | -5.3 [-5.9, -4.7] | -5.6 / -3.6 / -5.8 | x5.5 |
| best set + type estimate | -12.1 [-12.7, -11.4] | -11.9 / -15.0 / -9.4 | x7.3 |
| + rectangles | -14.0 [-14.5, -13.3] | -13.8 / -16.1 / -11.3 | x20.9 |
| + rectangles + one leaf-search type | -14.2 [-14.8, -13.6] | -14.0 / -16.4 / -11.5 | x28.4 |

(The best set's partition search is gated on this head by medium's key RD margin of 3ebca4466, so it
gains 1.2 % less than in section 10 against the same anchor.)

- Rectangles add 1.9-2.6 % on either base. With them our key is 2 % past SVT-AV1 still M4 (-12.0 %,
  0.24 s) and 3 % short of M2 (-17.3 %, 1.29 s), at about 0.9 s of key CPU on this box (the same box
  measured the anchor at 0.043 s; the record's 0.063 s was another box).
- Their CPU is the cost to cut: x2.9 the set without them in this run (x1.8 in a quieter one). The
  rectangles' transforms run the scalar `av1_forward` (no SIMD for rectangular DCT or for ADST), and
  every rectangle's luma candidates are transform-costed.
- An earlier run without CfL and with the luma by prediction SSE gained 1.0-1.3 %: CfL and the
  transform-costed luma are worth about 0.6-1.3 %.

Sets: `eval/keyrdo/rect-av1.txt`.

## Open

- AV1 keys against SVT-AV1 still M2-M0: 3 % short of M2 (section 11). Next: SIMD (or lanes) for the
  rectangular and ADST forward transforms, the rectangles in P frames through the same driver, AB and
  4:1 partitions, and 4x4 blocks.
- A plan-only first key pass with the searched pass on the re-code (halves the key's time under
  `rate.key_recode_miss`). Not built.
- How often the key re-code fires per key was not read from the logs.
- VP9, VVC and AV2 yardsticks (libvpx good, VVenC, AVM), and kvazaar for HEVC.
- Intra blocks in P frames: their share of bits and whether these searches pay there. Not measured.
- x86 cycles and Jetson cost of the key searches: only wall times from rate logs so far.
- The splice of a reference encoder's key into our P-frame path.
