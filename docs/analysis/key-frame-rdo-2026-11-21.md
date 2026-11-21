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
