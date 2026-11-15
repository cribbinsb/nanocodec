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

## Cut head 77e93089: the full fixed-QP lens (2026-11-15 afternoon)

The lens was run on this PC with `nceval.ctc local`, AOM CTC v8 LD (12 sequences, 130 f), on a gate build of
variance-integration 77e93089. BD-rates are in % against x264 medium at fixed QP, given as Weighted / J.
Data: [fixed-qp-lens-2026-11-15.csv](fixed-qp-lens-2026-11-15.csv), shown on the results site's Fixed QP
page (`web/results/data/fixed-qp-lens.json`). Columns:

- **Strict**: a fully static pattern, as the CTCs expect. The cadence has fixed strength
  (`rate.cadence_static_scaling 0`), there are no adaptive layers (H.264 is flat), and the second
  reference is either on every high-quality frame (H.264, HEVC, AV1, VP9, AV2) or off (VVC). For
  HEVC, VVC and AV2 their fixed-QP profile already is this pattern.
- **Profile**: the Default at `--qp`, with the codec's fixed-QP profile (`FIXED_QP_PARAMS`).
- **CBR settings**: the Default with the profile off (`rate.fixed_qp_profile 0`), i.e. exactly the CBR
  encoder with the quantizer fixed.

| Encoder | Strict | Profile | CBR settings |
|---|---:|---:|---:|
| H.264 Default | +5.3 / -0.9 | +3.9 / -4.8 | +3.8 / -3.6 |
| HEVC Default | -14.9 / -27.1 | -14.9 / -27.1 | -13.5 / -24.9 |
| VVC Default | -24.0 / -32.5 | -24.0 / -32.5 | -23.3 / -31.1 |
| AV1 Default | -16.9 / -25.6 | -17.9 / -26.0 | -16.9 / -25.3 |
| VP9 Default | -18.2 / -27.4 | -18.6 / -27.5 | -18.3 / -27.3 |
| AV2 Default | -25.2 / -34.7 | -25.2 / -34.7 | -24.2 / -33.4 |
| H.264 fast | | +29.4 / +13.9 | (same) |
| HEVC fast | | +7.9 / -4.2 | (same) |
| VVC fast | | -6.9 / -16.5 | (same) |
| AV1 fast | | +1.6 / -14.1 | (same) |
| VP9 fast | | +0.5 / -13.7 | (same) |
| AV2 fast | | -11.7 / -20.6 | (same) |

The fast rungs pin the profile off, so they have one column.

Anchors, each at its own fixed-QP setting (low delay, one thread, AQ off where it has a switch):

| Anchor | Weighted / J | | Anchor | Weighted / J |
|---|---:|---|---|---:|
| x264 veryslow | +0.5 / +1.8 | | SVT-AV1 M8 (rtc) | -30.3 / -30.8 |
| x265 medium | -19.9 / -20.9 | | SVT-AV1 M10 (rtc) | -14.2 / -19.5 |
| x265 slow | -25.7 / -27.8 | | libvpx VP9 good cpu 2 | -20.4 / -16.6 |
| VVenC faster LD | -48.1 / -46.9 | | libvpx VP9 rt 7 | +15.5 / +25.5 |
| HM LD-P | -47.8 / -48.6 | | libaom rt 8 | -2.2 / +3.2 |

The anchor curves are new on this page for x264 veryslow, x265 slow, SVT M8 and libvpx good (the `SLOW`
group of `nceval.ctc.conditions`); the others are the stored curves of
[standards-ctc-2026-11-14](standards-ctc-2026-11-14.md). x264 veryslow measures level with x264 medium on
this lens, because the lens runs x264 with its default psycho-visual tuning. With `--tune psnr` on the
pool it gains 3.7 % J (fixed-qp-gap). Reading:

- The profile pays most for H.264 (J -3.9 against strict, -1.2 against CBR settings) and HEVC (-2.2
  against CBR settings). For AV1, VP9, VVC and AV2 the three columns are within 1.4 % J.
- Only for H.264 does strict cost much: the flat structure gives up the adaptive layers (Weighted +1.4).
- Against the same formats' slow settings (differences of the x264-medium BD-rates, not paired BDs):
  HEVC Default is 0.7 J points behind x265 slow (Weighted 10.8 points behind); VP9 is 10.9 J points
  ahead of libvpx good cpu 2; AV1 is 4.8 J points behind SVT-AV1 M8 (Weighted 12.4 points behind; SVT's
  lead is in PSNR-YUV and MS-SSIM, not VMAF).

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
