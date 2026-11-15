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
