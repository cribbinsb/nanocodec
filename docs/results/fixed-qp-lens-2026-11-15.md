# Standards lens and slow references after the fixed-QP program (2026-11-15)

Run on this PC with `nceval.ctc local` (12 AOM CTC v8 LD sequences, 130 f; JVET class E, first 150 f),
against the x264 medium curves of [standards-ctc-2026-11-14.md](standards-ctc-2026-11-14.md). Our
streams are scored on the encoder's shown pictures. Figures are BD-rates in %, Weighted (0.92 Y +
0.04 U + 0.04 V) / J. The fixed-QP profiles and the analysis are in
[fixed-qp-gap-2026-11-14.md](../analysis/fixed-qp-gap-2026-11-14.md). The queue was stopped before
the fresh head run covered every codec. Rows marked "paired" give the profile's change against the
same build without it.

| Encoder (fixed QP) | 2026-11-14 published | 2026-11-15 | Build |
|---|---|---|---|
| H.264 Default, AOM LD | +5.8 / -2.3 | +5.4 / -4.0 | 7f5d917f (head) |
| H.264 Default, JVET-E | n/a | -7.3 / -15.4 (profile paired: -0.6 / -2.4) | 7f5d917f |
| H.264 Fast, AOM LD | +29.4 / +13.9 | +29.4 / +13.9 | 7f5d917f |
| HEVC Default, AOM LD | -11.0 / -22.8 | -11.8 / -23.7; profile paired +0.1 / -1.4 | c93c6b2e (after 03339919) |
| HEVC Default, JVET-E | n/a | -25.6 / -33.9; profile paired 0.0 / 0.0 | 4358da05 |
| VVC Default, AOM LD | -21.6 / -29.9 | -22.5 / -30.6; profile paired -0.0 / -1.0 | 4358da05 |
| AV1 Default, AOM LD | -14.4 / -23.5 | -14.4 / -23.5 (no profile) | c93c6b2e |
| AV2 Default, AOM LD | -18.4 / -30.3 | -18.4 / -30.3; profile paired -0.5 / -0.8 | 4358da05 |

Slow references on the pool (fixed QP, draw `fixedqp-2026-11-14`, 24 clips, J): H.264 +8.5 vs x264
veryslow psnr; HEVC +8.4 vs x265 slow psnr; AV1 +5.7 and AV2 +2.1 vs SVT-AV1 M8 rtc (MS-SSIM +16-18,
VMAF -4 to -8); VP9 -15.7 vs libvpx good cpu 2. VVenC low-delay faster did not finish within three
30-minute runs and is not reported.

## Head bad14976: the full fixed-QP lens (2026-11-15 evening)

The lens was run on this PC with `nceval.ctc local`, AOM CTC v8 LD (12 sequences, 130 f), on a gate build of
variance-integration bad14976. BD-rates are in % against x264 medium at fixed QP, given as Weighted / J.
Data: [fixed-qp-lens-2026-11-15.csv](fixed-qp-lens-2026-11-15.csv), shown on the results site's Fixed QP
page (`web/results/data/fixed-qp-lens.json`). An earlier run at 77e93089 is superseded.

Columns:

- **Strict**: a fully static pattern, as the CTCs expect. The cadence has fixed strength
  (`rate.cadence_static_scaling 0`), there are no adaptive layers (H.264 is flat), and every
  high-quality frame carries the second reference. For HEVC, VVC and AV2 their fixed-QP profile
  already is this pattern.
- **Profile**: the preset at `--qp`, with the codec's fixed-QP profile (`FIXED_QP_PARAMS`). The slow
  rung keeps it on; the fast rung pins it off.
- **CBR settings**: the Default with the profile off (`rate.fixed_qp_profile 0`), i.e. exactly the CBR
  encoder with the quantizer fixed.

fps is encoded frames per CPU second (user time) over the curve's 72 points. For our rungs and the
anchors run here (x264 ultrafast/veryfast/veryslow, x265 ultrafast/superfast/slow, SVT M8/M12/M13,
libaom rt 10, libvpx rt 9 and good 2) it was measured on this PC under shared load. For the stored
curves (x264 medium, x265 medium, SVT M10, libaom rt 8, libvpx rt 7, openh264, HM, VVenC) it comes from
the 2026-11-14 lens run. Treat it as an order of magnitude, not a timing.

| Encoder | Strict | Profile | CBR settings | fps |
|---|---:|---:|---:|---:|
| H.264 medium (Default) | +5.3 / -0.9 | +3.9 / -4.8 | +3.8 / -3.6 | 109 |
| HEVC medium (Default) | -14.9 / -27.1 | -14.9 / -27.1 | -13.5 / -24.9 | 57 |
| VVC medium (Default) | -24.0 / -32.5 | -24.0 / -32.5 | -23.3 / -31.1 | 41 |
| AV1 medium (Default) | -16.9 / -25.6 | -17.9 / -26.0 | -16.9 / -25.3 | 81 |
| VP9 medium (Default) | -18.2 / -27.4 | -18.6 / -27.5 | -18.3 / -27.3 | 79 |
| AV2 medium (Default) | -25.2 / -34.7 | -25.2 / -34.7 | -24.2 / -33.4 | 76 |
| H.264 slow | | -1.7 / -7.6 |  | 71 |
| HEVC slow | | -20.9 / -30.5 |  | 32 |
| VVC slow | | -26.9 / -33.8 |  | 37 |
| AV1 slow | | -21.4 / -28.5 |  | 55 |
| VP9 slow | | -20.4 / -28.7 |  | 52 |
| AV2 slow | | -28.4 / -36.3 |  | 26 |
| H.264 fast | | +29.4 / +13.9 | (profile pinned off) | 149 |
| HEVC fast | | +7.9 / -4.2 | (profile pinned off) | 108 |
| VVC fast | | -6.9 / -16.5 | (profile pinned off) | 66 |
| AV1 fast | | +1.6 / -14.1 | (profile pinned off) | 150 |
| VP9 fast | | +0.5 / -13.7 | (profile pinned off) | 74 |
| AV2 fast | | -11.7 / -20.6 | (profile pinned off) | 85 |

Anchors, each at its own fixed-QP setting (low delay, one thread, AQ off where it has a switch), by J:

| Anchor | Weighted / J | fps |
|---|---:|---:|
| hm-ldp | -47.8 / -48.6 | 0.353 |
| vvenc-faster-ld | -48.1 / -46.9 | 8.21 |
| svtav1-m8 | -30.3 / -30.8 | 32.7 |
| x265-slow | -25.7 / -27.8 | 5.87 |
| x265-medium | -19.9 / -20.9 | 32.8 |
| svtav1-m10 | -14.2 / -19.5 | 234 |
| vpx-vp9-good-s2 | -20.4 / -16.6 | 8.3 |
| x265-superfast | -10.3 / -15.7 | 45.7 |
| svtav1-m12 | -1.5 / -10.4 | 283 |
| svtav1-m13 | +4.1 / -4.8 | 291 |
| x264-medium | +0.0 / +0.0 | 78.1 |
| x264-veryslow | +0.6 / +1.8 | 7.88 |
| aom-rt-s8 | -2.2 / +3.2 | 82 |
| aom-rt-s10 | +6.0 / +5.8 | 99.9 |
| x265-ultrafast | +20.4 / +6.0 | 63.5 |
| x264-veryfast | +14.6 / +19.2 | 226 |
| vpx-vp9-rt-s7 | +15.5 / +25.5 | 156 |
| vpx-vp9-rt-s9 | +46.9 / +41.4 | 210 |
| openh264-baseline | +50.0 / +54.4 | 483 |
| x264-ultrafast | +155.7 / +137.9 | 732 |

x264 veryslow measures level with x264 medium on this lens, because the lens runs x264 with its
default psycho-visual tuning. With `--tune psnr` on the pool it gains 3.7 % J (fixed-qp-gap).

## Pool findings of this session (nceval.fixedqp, matched QP)

- **Unified plan in the fixed-QP profile (landed).** The second reference goes on every high-quality
  frame (`structure.secondary_min_depth 0`), with the full decision against it (`refs.second_full 1`).
  AV1: J -1.9 [-2.9, -0.9] and -2.4 [-4.2, -0.8] on draws `fixedqp-2026-11-15` / `-second`; x86 P
  instructions +2.0 %. VP9: -1.5 [-2.4, -0.7] and -1.4 [-2.6, -0.3], +1.0 %. HEVC: only
  `refs.second_full` (the depth gate is inert at full cadence strength), -1.3 [-2.1, -0.6] and -1.7
  [-3.0, -0.5], +2.0 %.
- **Multi-reference is not the H.264/HEVC gap to the slow references.** x264 with one reference against
  its default (draw `fixedqp-2026-11-14`, 22-23 clips, `--tune psnr`): veryslow (16 refs) J -0.2
  [-1.0, +1.1]; medium (3 refs) -0.0. Our own H.264 second reference: flat + second reference vs the
  layered Default -1.5 [-4.3, +1.5] on one draw, +1.4 [-1.6, +4.4] on the other, so it is not a lever.
- **The identified H.264 lead: RD refinement of motion vectors and partitions.** x264 veryslow
  (one ref) at `--subme 7` instead of 10: J +2.5 [+1.9, +3.2] (16 clips). `--subme 9` is identical to 10,
  so x264's subme 8-9 RD refinement carries most of veryslow's 3.2 % over medium. This is ledgered, not
  built.
