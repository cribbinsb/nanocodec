# Standards-body test conditions: our encoders under the AOM and JVET low-delay CTCs (2026-11-14)

A second, independent lens beside our own charts: the sequences, configurations, metrics and BD-rate method
that the AOM and JVET codec groups use, kept as close as we could to low delay and low bitrate. Why these
conditions were chosen: [the methodology note](../analysis/standards-evaluation-methodology-2026-11-14.md).
This page is a data collection for the owner; it changes no headline, chart or README.

**Read this first.** The two reference encoders here (HM and VTM) and libaom at speed 0 are the formats'
reference and research software. They search exhaustively by design. In this test they used about 200 to 4,000 times
the CPU time of our encoders (EncT columns). Ours is a narrow real-time subset: one reference structure, no B
slices, single-threaded real-time presets. The BD-rates against them measure how far a real-time subset sits
from each format's exhaustive reference at CTC settings. They do not measure competitiveness. The real-time
encoders (x264, x265, SVT-AV1, libaom rt, libvpx rt, openh264, kvazaar, VVenC) are the practical
comparison. Every figure is at **fixed QP**. That measures coding efficiency without rate control, which is
not how our encoders are used. Our charts measure CBR. The CBR rows below are a labelled extra.

## Results

All numbers are BD-rates in percent: the extra (+) or saved (−) bits at equal quality, as a mean over the
sequences with a reliable BD-rate (`n`, counted for the first column; another column can rest on one
sequence fewer). "Weighted" is the AOM CTC's 0.92 Y + 0.04 U + 0.04 V combination of
the overall-PSNR BD-rates. "J" is our own composite, for cross-reference. EncT is the test encoder's
single-thread CPU time relative to the anchor's: 1/387 means the test encoder used 1/387 of the anchor's
time. The last column is the same Weighted figure over the free-licence sequences only (6 of 12: Johnny,
KristenAndSara, BlueSky, Stockholm, FourPeople, Vertical_Bayshore). The research-use sequences are included
everywhere else, by the owner's decision.

2,805 of 2,814 planned points were measured. 0 jobs failed, and all 1,944 of our streams decoded bit-exactly in
both standard decoders. Nine reference-encoder points did not finish within a launch's 30-minute limit and
are missing. Eight are libaom speed 0: RedKayak q28-q52 (so RedKayak has no libaom speed-0 curve),
Stockholm q28/q34 and SnowMountain q28. The ninth is HM's WestWindEasy at QP 22. Those curves are shorter,
and BD-rate uses their overlap only.

### 1. AOM CTC low delay, 12 sequences, against HM-19.0 LD-P

| Encoder | Weighted | Y | U | V | APSNR-YUV | MS-SSIM | VMAF | J | n | EncT | Weighted, free tier only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nanocodec-av1 | +72.7 | +78.2 | -11.3 | -5.3 | +72.2 | +64.7 | +61.8 | +51.8 | 11 | 1/387 | +91.2 |
| nanocodec-av1@av1-fast | +111.4 | +119.3 | +12.1 | +28.7 | +115.4 | +80.1 | +80.3 | +75.5 | 12 | 1/740 | +130.3 |
| nanocodec-av2 | +64.5 | +71.1 | -14.1 | -8.9 | +66.2 | +56.6 | +52.2 | +40.6 | 12 | 1/382 | +82.9 |
| nanocodec-av2@av2-fast | +84.2 | +84.1 | +71.7 | +98.7 | +83.4 | +74.1 | +60.5 | +60.1 | 12 | 1/525 | +100.0 |
| nanocodec-vp9 | +66.5 | +73.8 | -19.9 | -15.4 | +68.2 | +60.9 | +50.6 | +47.6 | 12 | 1/332 | +83.2 |
| nanocodec-vp9@vp9-fast | +106.6 | +115.8 | -2.2 | +3.5 | +109.4 | +92.5 | +78.8 | +75.7 | 12 | 1/487 | +130.6 |
| nanocodec-vvc | +47.0 | +50.8 | -3.4 | +9.6 | +47.2 | +43.1 | +37.0 | +33.1 | 12 | 1/200 | +58.3 |
| nanocodec-vvc@vvc-fast | +76.5 | +76.9 | +59.0 | +85.6 | +75.0 | +68.0 | +63.9 | +61.5 | 12 | 1/333 | +94.6 |
| nanocodec-hevc | +72.7 | +75.0 | -11.9 | +4.4 | +68.4 | +60.5 | +60.0 | +48.2 | 11 | 1/323 | +85.8 |
| nanocodec-hevc@hevc-fast | +106.7 | +113.9 | +17.9 | +29.4 | +106.3 | +98.6 | +104.8 | +86.0 | 12 | 1/593 | +137.9 |
| nanocodec-h264 | +115.8 | +123.9 | +18.7 | +26.3 | +117.7 | +106.7 | +115.7 | +101.3 | 12 | 1/575 | +121.7 |
| nanocodec-h264@h264-fast | +168.4 | +178.5 | +46.1 | +57.3 | +170.9 | +151.5 | +147.8 | +135.9 | 12 | 1/925 | +172.8 |
| vtm-ldp | -26.3 | -25.6 | -33.1 | -34.9 | -26.0 | -24.1 | -26.9 | -27.1 | 2 | 4.42x | -26.3 |
| aom-good-s0-ld | -25.4 | -24.8 | -31.0 | -32.8 | -25.6 | -27.1 | -28.0 | -21.7 | 7 | 2.79x | -24.0 |
| vvenc-faster-ld | +0.0 | +0.1 | -5.2 | +2.4 | -0.1 | -1.9 | +0.3 | +2.6 | 12 | 0.0427x | +5.9 |
| x265-medium | +55.6 | +53.3 | +80.5 | +82.7 | +54.2 | +51.2 | +59.8 | +57.0 | 12 | 1/107 | +61.5 |
| kvazaar-medium | +68.7 | +60.5 | +159.0 | +168.0 | +64.5 | +62.0 | +53.0 | +76.2 | 12 | 0.0187x | +72.2 |
| svtav1-m10 | +74.5 | +76.0 | +52.2 | +61.8 | +74.0 | +63.1 | +62.7 | +63.2 | 12 | 1/692 | +90.8 |
| aom-rt-s8 | +104.1 | +97.4 | +166.9 | +195.3 | +100.3 | +94.1 | +92.3 | +110.7 | 12 | 1/234 | +119.4 |
| x264-medium | +104.6 | +105.0 | +87.9 | +113.8 | +104.4 | +107.7 | +111.9 | +110.9 | 12 | 1/263 | +117.7 |
| vpx-vp9-rt-s7 | +152.1 | +144.7 | +218.6 | +255.9 | +148.1 | +144.5 | +128.5 | +158.3 | 12 | 1/455 | +169.1 |
| openh264-baseline | +217.8 | +209.3 | +291.8 | +339.2 | +214.0 | +219.7 | +257.2 | +235.9 | 12 | 1/1,597 | +251.0 |

VTM and libaom speed 0 rows cover only the classes they ran on (A5, and A4/A5), so they have small `n`.

### 2. The same streams against x264 medium (our charts' anchor), at fixed QP

| Encoder | Weighted | Y | U | V | APSNR-YUV | MS-SSIM | VMAF | J | n | EncT | Weighted, free tier only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nanocodec-av1 | -14.4 | -11.4 | -49.4 | -49.2 | -13.8 | -18.9 | -19.0 | -23.5 | 12 | 0.679x | -11.5 |
| nanocodec-av1@av1-fast | +1.6 | +4.9 | -37.0 | -36.4 | +3.4 | -12.8 | -10.9 | -14.1 | 12 | 0.355x | +3.8 |
| nanocodec-av2 | -18.4 | -15.4 | -52.3 | -53.8 | -17.5 | -23.2 | -24.5 | -30.3 | 12 | 0.687x | -15.2 |
| nanocodec-av2@av2-fast | -11.7 | -9.7 | -16.6 | -9.2 | -10.1 | -15.7 | -20.0 | -20.6 | 11 | 0.5x | -7.9 |
| nanocodec-vp9 | -17.3 | -14.2 | -52.4 | -53.0 | -16.4 | -21.1 | -25.0 | -26.6 | 12 | 0.791x | -15.3 |
| nanocodec-vp9@vp9-fast | +0.5 | +4.3 | -43.0 | -44.3 | +1.7 | -7.5 | -11.9 | -13.7 | 12 | 0.539x | +4.2 |
| nanocodec-vvc | -21.6 | -20.2 | -37.6 | -37.1 | -21.5 | -25.9 | -29.0 | -29.9 | 12 | 1.31x | -20.9 |
| nanocodec-vvc@vvc-fast | -6.9 | -7.0 | -5.8 | -4.5 | -7.5 | -13.6 | -15.9 | -16.5 | 12 | 0.788x | -3.9 |
| nanocodec-hevc | -11.0 | -8.1 | -44.2 | -43.1 | -11.0 | -17.7 | -17.6 | -22.8 | 12 | 0.812x | -8.7 |
| nanocodec-hevc@hevc-fast | +7.9 | +11.1 | -28.9 | -28.9 | +7.8 | +0.6 | +4.5 | -4.2 | 12 | 0.443x | +16.6 |
| nanocodec-h264 | +5.8 | +9.1 | -31.0 | -33.0 | +6.8 | +0.2 | +3.6 | -2.3 | 12 | 0.456x | +2.2 |
| nanocodec-h264@h264-fast | +29.4 | +33.4 | -16.2 | -17.6 | +30.5 | +21.3 | +19.0 | +13.9 | 12 | 0.284x | +25.7 |
| hm-ldp | -47.8 | -48.4 | -36.8 | -46.4 | -48.2 | -48.9 | -49.0 | -48.6 | 12 | 263x | -51.8 |
| x265-medium | -19.9 | -21.4 | +2.4 | -8.8 | -20.7 | -23.4 | -20.8 | -20.9 | 12 | 2.44x | -22.7 |
| svtav1-m10 | -14.2 | -13.4 | -20.3 | -25.7 | -14.2 | -20.2 | -19.7 | -19.5 | 12 | 0.379x | -12.9 |
| aom-rt-s8 | -2.2 | -1.9 | +29.1 | +31.7 | -0.5 | -3.8 | -5.5 | +3.2 | 11 | 1.12x | +0.5 |
| vpx-vp9-rt-s7 | +15.5 | +20.5 | +50.4 | +61.6 | +22.1 | +19.8 | +11.7 | +25.5 | 11 | 0.577x | +21.2 |
| openh264-baseline | +50.0 | +45.9 | +96.8 | +97.9 | +48.4 | +49.0 | +64.0 | +54.4 | 12 | 0.164x | +56.4 |

### 3. Our CBR mode (labelled extra), against x264 medium at fixed QP

Each of our Default encoders also ran its real-time rate control (`--bitrate`, 1 s buffer) at the rate its
own fixed-QP point reached. This mixes two modes: our rate control against x264 at fixed QP. It shows what
our rate control costs on these sequences. It is not a like-for-like comparison.

| Encoder | Weighted | Y | U | V | APSNR-YUV | MS-SSIM | VMAF | J | n | EncT | Weighted, free tier only |
|---|---|---|---|---|---|---|---|---|---|---|---|
| nanocodec-av1 (CBR extra) | -3.8 | -0.7 | -39.4 | -40.8 | -2.8 | -9.3 | -6.9 | -11.6 | 12 | — | -8.0 |
| nanocodec-av2 (CBR extra) | -7.7 | -4.4 | -44.9 | -46.5 | -6.5 | -13.1 | -13.3 | -20.1 | 12 | — | -8.6 |
| nanocodec-vp9 (CBR extra) | -5.7 | -2.7 | -39.4 | -42.0 | -4.6 | -11.1 | -13.9 | -13.8 | 12 | — | -9.1 |
| nanocodec-vvc (CBR extra) | -10.2 | -8.8 | -25.3 | -26.2 | -10.0 | -16.6 | -17.0 | -17.7 | 12 | — | -13.7 |
| nanocodec-hevc (CBR extra) | -0.0 | +2.7 | -31.9 | -32.1 | +0.2 | -8.8 | -7.0 | -11.2 | 12 | — | -2.6 |
| nanocodec-h264 (CBR extra) | +17.8 | +21.0 | -18.4 | -20.4 | +18.8 | +9.8 | +16.6 | +10.9 | 12 | — | +9.8 |

### 4. JVET low-delay P, class E (FourPeople, Johnny, KristenAndSara, 720p60, first 150 frames), against HM-19.0 LD-P

Frame-averaged PSNR, as HM reports it. The "@ctc" columns use JVET's QP 22-37 points for QP-scale encoders.
"Y-fa" uses all six points (QP 42 and 47 added, the low-rate extension). Our AV1, VP9 and AV2 use the qindex
points throughout.

| Encoder | Y-fa@ctc | U-fa@ctc | V-fa@ctc | Y-fa | J | n | EncT | Y-fa@ctc, free tier only |
|---|---|---|---|---|---|---|---|---|
| nanocodec-av1 | +55.6 | -39.4 | -37.8 | +50.5 | +33.2 | 3 | 1/612 | +55.6 |
| nanocodec-av1@av1-fast | +115.6 | -14.4 | -10.1 | +95.2 | +63.6 | 3 | 1/1,028 | +115.6 |
| nanocodec-av2 | +62.6 | -34.1 | -32.4 | +41.5 | +23.4 | 3 | 1/630 | +62.6 |
| nanocodec-av2@av2-fast | +82.6 | +2.0 | +18.8 | +53.0 | +34.9 | 3 | 1/771 | +82.6 |
| nanocodec-vp9 | +62.2 | -34.3 | -34.9 | +56.1 | +39.5 | 3 | 1/528 | +62.2 |
| nanocodec-vp9@vp9-fast | +107.7 | -17.1 | -18.4 | +87.2 | +60.8 | 3 | 1/659 | +107.7 |
| nanocodec-vvc | +46.6 | -18.4 | -21.4 | +33.5 | +20.2 | 3 | 1/304 | +46.6 |
| nanocodec-vvc@vvc-fast | +80.6 | +25.0 | +26.3 | +58.7 | +41.3 | 3 | 1/460 | +80.6 |
| nanocodec-hevc | +76.8 | — | — | +57.0 | +31.7 | 3 | 1/423 | +76.8 |
| nanocodec-hevc@hevc-fast | +121.9 | -10.3 | -9.7 | +96.4 | +65.6 | 3 | 1/742 | +121.9 |
| nanocodec-h264 | +124.4 | -4.0 | +0.2 | +115.6 | +97.9 | 3 | 1/837 | +124.4 |
| nanocodec-h264@h264-fast | +166.1 | +12.1 | +18.7 | +147.4 | +123.3 | 3 | 1/1,144 | +166.1 |
| vvenc-faster-ld | +0.7 | -10.0 | -1.8 | +1.2 | +5.7 | 3 | 0.0262x | +0.7 |
| x265-medium | +52.2 | +68.8 | +74.3 | +49.5 | +47.5 | 3 | 1/159 | +52.2 |
| kvazaar-medium | +74.5 | +139.5 | +139.8 | +74.3 | +83.4 | 3 | 0.0125x | +74.5 |
| svtav1-m10 | +83.2 | +12.1 | +15.5 | +72.6 | +64.6 | 3 | 1/1,077 | +83.2 |
| aom-rt-s8 | +110.8 | +149.4 | +149.7 | +122.8 | +147.8 | 3 | 1/308 | +110.8 |
| x264-medium | +109.2 | +63.2 | +72.9 | +127.1 | +125.9 | 3 | 1/402 | +109.2 |
| vpx-vp9-rt-s7 | +177.0 | +171.0 | +181.1 | +185.8 | +204.0 | 3 | 1/643 | +177.0 |
| openh264-baseline | +267.5 | +269.9 | +301.9 | +259.3 | +256.1 | 3 | 1/2,566 | +267.5 |

Per-class tables for every codec, the tables against VTM and libaom speed 0, and the bitrates each
quantiser point gives are in [tables.md](standards-ctc-2026-11-14/tables.md). The rates HM's points reach
are relevant to the low-bitrate question (median kbps):

| HM LD-P, QP | 22 | 27 | 32 | 37 | 42 | 47 |
|---|---|---|---|---|---|---|
| A3 (720p, 4 sequences) | 1,981 | 1,204 | 571 | 285 | 137 | 81 |
| A4 (360p, 6) | 1,326 | 454 | 194 | 94 | 46 | 25 |
| A5 (270p, 2) | 744 | 304 | 141 | 66 | 33 | 19 |
| E (720p60, 3) | 1,995 | 670 | 322 | 173 | 96 | 60 |

JVET's own points (QP 22-37) end at medians of roughly 70-290 kbps at these sizes. The extra QP 42/47 points
(and the AOM CTC's qindex 235) reach medians of about 20-140 kbps, the range our product targets.

## Reading

**Against the reference software (HM, VTM, libaom speed 0).** Every one of our encoders needs more bits than
HM's low-delay P configuration at equal PSNR. The Default presets need +47% (VVC) to +116% (H.264) on the
Weighted figure. AV2, VP9, AV1 and HEVC fall between +64% and +73%. The Fast presets need +76% to +168%. On
our J the Defaults are +33% (VVC) to +101% (H.264). In the same table, the real-time encoders are also behind
HM: x265 medium +56%, SVT-AV1 M10 (rtc) +75%, x264 medium +105%, libaom rt 8 +104%, libvpx rt 7 +152%,
openh264 +218%. VVenC's low-delay `faster` preset is level with HM (+0.0%). Our encoders used 1/200 to 1/925
of HM's CPU time. This is the expected position for a narrow real-time subset (one reference, no B slices, no
GOP-8 QP hierarchy, real-time search) against an exhaustive reference encoder at CTC settings. The size of the
gap is the useful number. On class E (JVET table, CTC points) it is +47% for our VVC and +56% for our AV1,
against +52% for x265 medium and +109% for x264 medium. VTM and libaom speed 0 measure 25-26% below HM on
the classes they ran, so the gap to them is larger (tables.md).

**Against the real-time encoders, at fixed QP.** Against x264 medium, our Default presets measure −22% (VVC),
−18% (AV2), −17% (VP9), −14% (AV1), −11% (HEVC) and +6% (H.264) on the Weighted figure. On J they measure
−30% to −2%. The Fast presets measure −12% to +29% (J −21% to +14%). For comparison in the same table:
x265 medium −20%, SVT-AV1 M10 −14%, libaom rt 8 −2%, libvpx rt 7 +16%, openh264 +50%. Our encoders used 0.28
to 1.31 times x264 medium's CPU time. Our chroma BD-rates are strongly negative against every anchor. We
spend relatively more on chroma at a given luma rate. The CTC's Weighted figure, 92% luma, credits little of
that.

**Where this lens disagrees with our own charts, and why we think so.** Our published chart gives AV1
Default a BD-J of −50.9% against x264 medium. This lens gives −23.5% at fixed QP, and −11.6% when our
rate control runs against x264 at fixed QP. Measured here: our rate control costs our Default encoders 10-13 points
(Weighted and J) on these sequences against our own fixed-QP curves (table 3 against table 2). The rest of the difference has
causes we have not tested separately: (a) the anchor's mode. Our charts run x264 with its own CBR rate
control and its default psychovisual tuning. Its metric-tuned twins in our audit score better on these
metrics. Here x264 runs at fixed QP. (b) Content. These are clean, mostly 25-60 fps test sequences, several of them static
conferencing scenes. Our pool is 30 fps camera footage with noise and motion. (c) The rate range: here it runs from about 10 kbps to several Mbps across 270p-720p; our charts use
150-1,200 kbps at 720p30. Both lenses put VVC and AV2 first and H.264 last among our Defaults. Our charts
give BD-J −54.4 / −54.0 / −50.9 / −49.6 / −48.2 / −24.0 for VVC / AV2 / AV1 / HEVC / VP9 / H.264. This lens
gives J −29.9 / −30.3 / −23.5 / −22.8 / −26.6 / −2.3.

**What this lens adds.** It is a reproducible, public, recognisable measurement. Every sequence is
downloadable with a checksum, and every anchor is the formats' reference at its published configuration. It
is also the clearest statement of headroom: 47-116% to HM and more to VTM. Most of that headroom is
structural (multiple references, hierarchical QP, exhaustive search), which this project trades for speed by
design.

**Caveats.** n is 12 sequences (3 for JVET-E). Fixed QP is not how our encoders run. KristenAndSara was in our
screening pool before today. The licence-tier split moves the Weighted figures by up to about 30 points (last
column). The free six include the static conferencing scenes. Our guess, not tested, is that HM's
multi-reference structure gains most there. No figure here is a confidence interval.


## Method

| Item | AOM CTC v8 low delay (primary) | JVET LD-P class E (secondary) |
|---|---|---|
| Source of the conditions | CWG-F384o *AV2 CTC v8* §4.3 (2025-10-10) | JCTVC-L1100 / JVET-T2010 low-delay P |
| Sequences | classes A3, A4, A5, 8-bit sources only: 12 sequences (below) | FourPeople, Johnny, KristenAndSara 1280x720p60 (derf copies) |
| Frames | 130, one key frame | **first 150** (CTC: all 600) |
| Quantiser points | QP codecs 22, 27, 32, 37, 42, 47; our AV1/VP9/AV2 qindex 110-235 (the CTC's); libaom/SVT/libvpx q 28-59 (same qindex via `quantizer_to_qindex`) | same points; the JVET figure uses QP 22-37 for QP codecs, the extension all six |
| Metrics | `vmaf --aom_ctc v6.0` (libvmaf v3.2.1): overall PSNR Y/U/V (MSE averaged over frames), SSIM, MS-SSIM (dB), VMAF and VMAF-NEG v0.6.1, CIEDE2000, PSNR-HVS; plus VMAF v1 (3d0h) for our J | frame-averaged PSNR Y/U/V (as HM prints) |
| BD-rate | log rate, PCHIP over quality, overlap only (`nceval.bdrate`, CTC §5.5); BD-RateWeighted = 0.92 Y + 0.04 U + 0.04 V | same interpolation |
| Rate | payload bytes (no IVF headers: the CTC's `.obu` bytes; Annex-B files whole) x 8 x fps / frames | same |
| Reporting | per class (equal weight per sequence) and overall mean; n = sequences with a reliable BD-rate (overlap ≥ 50% of the shorter curve) | overall |
| Time | EncT = geometric mean over sequences of the summed encode user time, test / anchor (`os.wait4`), one thread, on the cloud machine (AMD EPYC 9R14), not our Jetson | same |
| Our J (cross-reference) | per sequence the mean of the full-overlap and lower-half BD-rates over PSNR-YUV (6:1:1), VMAF v1 and MS-SSIM, as `docs/PROCEDURES.md` defines it | same |

Our streams: every one was decoded by both standard decoders of its codec (H.264: ffmpeg + JM; HEVC: ffmpeg + HM;
VVC: ffmpeg + VVdeC; AV1: ffmpeg/dav1d + aomdec; VP9: ffmpeg + vpxdec; AV2: avmdec + dav2d). Each decoded frame's
MD5 had to equal the encoder's reconstruction. In fixed-QP runs the shown pictures (what we score) had to equal
the coded ones. A stream failing either check counts as a failed job. Per stream, the frame count and the SHA-256 of
its per-frame MD5 list are in `docs/results/standards-ctc-2026-11-14/streams.csv`. The lists themselves are kept
outside the repository (`~/nanocodec-data/work/standards-eval/res-*/hashes`). No decoded video was kept, and none left the cloud machines.

### Encoders

| Group | Encoder (name in tables) | Version | Configuration |
|---|---|---|---|
| ours | nanocodec-{h264,hevc,vvc,av1,vp9,av2} and `@<codec>-fast` | 1668a7bd (verified head) | `--qp Q --intra-period <frames>`; Default (engine defaults) and the Fast preset; CBR extra: `--bitrate K --buffer-ms 1000` at the rate the same point reached at fixed QP |
| standards | hm-ldp | HM-19.0 (24c021c6) | `encoder_lowdelay_P_main.cfg` unchanged, `--ConformanceWindowMode=1` |
| standards | vtm-ldp | VTM-24.0 (ca331de2) | `encoder_lowdelay_P_vtm.cfg`, `--InternalBitDepth=8` (CTC: 10); **class A5 only** (time) |
| standards | aom-good-s0-ld | libaom v3.15.3 | `--good --cpu-used=0` with the AOM CTC LD flags libaom has (`--lag-in-frames=0`, 16-frame pyramid, fixed QP offsets, no TPL / delta-q); **classes A4/A5 only** (time) |
| practical | x264-medium | 0.165 (b35605a) | `--preset medium --tune zerolatency --bframes 0 --rc-lookahead 0 --threads 1 --qp Q` |
| practical | x265-medium | 4.2 | `--preset medium --tune zerolatency --bframes 0`, one thread, `--qp Q` |
| practical | openh264-baseline | v2.7.0 | Constrained Baseline, camera real time, medium complexity, fixed QP |
| practical | kvazaar-medium | v2.3.2 | `--preset medium --gop 0 --no-bipred`, one thread, `--qp Q` |
| practical | vvenc-faster-ld | VVenC v1.14.0 (9428ea86) | `cfg/experimental/lowdelay_faster.cfg` (low delay with B slices from past pictures only), 8-bit, one thread |
| practical | svtav1-m10 | SVT-AV1 v4.3.0 | `--rtc 1 --preset 10 --pred-struct 1 --rc 0 --aq-mode 0 --qp Q`, one thread |
| practical | aom-rt-s8 | libaom v3.15.3 | `--rt --cpu-used=8 --lag-in-frames=0`, fixed q, `--aq-mode=0` |
| practical | vpx-vp9-rt-s7 | libvpx v1.16.0 | `--rt --cpu-used=7`, fixed q, `--aq-mode=0` |

The exact command lines are generated by `eval/nceval/ctc` (`python3 -m nceval.ctc stage`) from the adapters in
`eval/nceval/encoders/` and `eval/nceval/ctc/reference_encoders.py`. Metric tool: libvmaf v3.2.1. Decoders:
`tools/anchors/versions.env`.

### Sequences and licences

Downloaded on the rented machines from the URLs below and checked against the CTC's MD5 (derf publishes none,
so the digests we received are recorded in `docs/results/standards-ctc-2026-11-14/sources.json`). Nothing is
redistributed. Each licence is summarised from the sequence's own notice; the notice URL is given.

| Class | Sequence | Size, fps | Tier | Licence (from the notice) | Source | MD5 (CTC) |
|---|---|---|---|---|---|---|
| A3 | Johnny_1280x720_60 | 1280x720, 60 | free | public domain (author: Vidyo) ([notice](https://media.xiph.org/video/aomctc/test_set/a3_720p/Johnny_1280x720_60_Copyright.txt)) | [Johnny_1280x720_60.y4m](https://media.xiph.org/video/aomctc/test_set/a3_720p/Johnny_1280x720_60.y4m) | 1d6aab4003385c255262a88f4df7cccc |
| A3 | KristenAndSara_1280x720_60 | 1280x720, 60 | free | public domain (author: Vidyo) ([notice](https://media.xiph.org/video/aomctc/test_set/a3_720p/KristenAndSara_1280x720_60_Copyright.txt)) | [KristenAndSara_1280x720_60.y4m](https://media.xiph.org/video/aomctc/test_set/a3_720p/KristenAndSara_1280x720_60.y4m) | 43bb17b78086d0183642c7b74cc1c903 |
| A3 | ControlledBurn_1280x720p30_420 | 1280x720, 30 | research | NTIA/ITS: research use only, acknowledgement kept ([notice](https://media.xiph.org/video/aomctc/test_set/a3_720p/ControlledBurn_1280x720p30_420_Copyright.txt)) | [ControlledBurn_1280x720p30_420.y4m](https://media.xiph.org/video/aomctc/test_set/a3_720p/ControlledBurn_1280x720p30_420.y4m) | 875d857d8ef1b7f1b653c4c530669005 |
| A3 | WestWindEasy_1280x720p30_420 | 1280x720, 30 | research | NTIA/ITS: research use only, acknowledgement kept ([notice](https://media.xiph.org/video/aomctc/test_set/a3_720p/WestWindEasy_1280x720p30_420_Copyright.txt)) | [WestWindEasy_1280x720p30_420.y4m](https://media.xiph.org/video/aomctc/test_set/a3_720p/WestWindEasy_1280x720p30_420.y4m) | da8afe9c91a260c835b39f58b13c99e7 |
| A4 | BlueSky_360p25_v2 | 640x360, 25 | free | TU München / Taurus Media: "Restrictions of use: No restrictions" ([notice](https://media.xiph.org/video/aomctc/test_set/a4_360p/BlueSky_360p25_Copyright.txt)) | [BlueSky_360p25_v2.y4m](https://media.xiph.org/video/aomctc/test_set/a4_360p/BlueSky_360p25_v2.y4m) | 722261f9e84578203ea9486673ffb274 |
| A4 | Stockholm_640x360_5994 | 640x360, 59.94 | free | SVT (2001 set): "Restrictions of use: No restrictions" ([notice](https://media.xiph.org/video/aomctc/test_set/a4_360p/Stockholm_640x360_5994_Copyright.txt)) | [Stockholm_640x360_5994.y4m](https://media.xiph.org/video/aomctc/test_set/a4_360p/Stockholm_640x360_5994.y4m) | 62d126930a6c8f61f5267f0b21746cde |
| A4 | RedKayak_360_2997 | 640x360, 29.97 | research | NTIA/ITS: research use only, acknowledgement kept ([notice](https://media.xiph.org/video/aomctc/test_set/a4_360p/RedKayak_360_2997_Copyright.txt)) | [RedKayak_360_2997.y4m](https://media.xiph.org/video/aomctc/test_set/a4_360p/RedKayak_360_2997.y4m) | 4205131788b94668c4feef73ce44aec9 |
| A4 | SnowMountain_640x360_2997 | 640x360, 29.97 | research | NTIA/ITS: research use only, acknowledgement kept ([notice](https://media.xiph.org/video/aomctc/test_set/a4_360p/SnowMountain_640x360_2997_Copyright.txt)) | [SnowMountain_640x360_2997.y4m](https://media.xiph.org/video/aomctc/test_set/a4_360p/SnowMountain_640x360_2997.y4m) | 63ae6b7fb786883c2144552e98d0c4dd |
| A4 | SpeedBag_640x360_2997 | 640x360, 29.97 | research | NTIA/ITS: research use only, acknowledgement kept ([notice](https://media.xiph.org/video/aomctc/test_set/a4_360p/SpeedBag_640x360_2997_Copyright.txt)) | [SpeedBag_640x360_2997.y4m](https://media.xiph.org/video/aomctc/test_set/a4_360p/SpeedBag_640x360_2997.y4m) | 8bfe53f0215b749744b3947e16ab09a4 |
| A4 | TouchdownPass_640x360_2997 | 640x360, 29.97 | research | NTIA/ITS: research use only, acknowledgement kept ([notice](https://media.xiph.org/video/aomctc/test_set/a4_360p/TouchdownPass_640x360_2997_Copyright.txt)) | [TouchdownPass_640x360_2997.y4m](https://media.xiph.org/video/aomctc/test_set/a4_360p/TouchdownPass_640x360_2997.y4m) | dd98868ad8286eb0fb0b8c00827098a1 |
| A5 | FourPeople_480x270_60 | 480x270, 60 | free | public domain (author: Vidyo) ([notice](https://media.xiph.org/video/aomctc/test_set/a5_270p/FourPeople_480x270_60_Copyright.txt)) | [FourPeople_480x270_60.y4m](https://media.xiph.org/video/aomctc/test_set/a5_270p/FourPeople_480x270_60.y4m) | 4d35db57a2377e4211134db345dea225 |
| A5 | Vertical_Bayshore_270x480_2997 | 270x480, 29.97 | free | CC BY-SA 4.0, (c) James Leslie Skinner ([notice](https://media.xiph.org/video/aomctc/test_set/a5_270p/Vertical_Bayshore_270x480_2997_Copyright.txt)) | [Vertical_Bayshore_270x480_2997.y4m](https://media.xiph.org/video/aomctc/test_set/a5_270p/Vertical_Bayshore_270x480_2997.y4m) | 6ec078264f2ae2c99e2fbe108b774847 |
| E | FourPeople_1280x720_60_E | 1280x720, 60 | free | public domain (derf listing; author Vidyo, as in the AOM CTC copies' notices) ([notice](https://media.xiph.org/video/derf/)) | [FourPeople_1280x720_60.y4m](https://media.xiph.org/video/derf/y4m/FourPeople_1280x720_60.y4m) | not published |
| E | Johnny_1280x720_60_E | 1280x720, 60 | free | public domain (derf listing; author Vidyo, as in the AOM CTC copies' notices) ([notice](https://media.xiph.org/video/derf/)) | [Johnny_1280x720_60.y4m](https://media.xiph.org/video/derf/y4m/Johnny_1280x720_60.y4m) | not published |
| E | KristenAndSara_1280x720_60_E | 1280x720, 60 | free | public domain (derf listing; author Vidyo, as in the AOM CTC copies' notices) ([notice](https://media.xiph.org/video/derf/)) | [KristenAndSara_1280x720_60.y4m](https://media.xiph.org/video/derf/y4m/KristenAndSara_1280x720_60.y4m) | not published |

Not used: Vidyo3, Vidyo4 (A3) and ParkJoy (A5), whose notices limit use to developing standards. DrivingPOV,
RollerCoaster (A3) and SparksElevator (A5) are 10-bit sources.

**KristenAndSara was not held out.** An excerpt of the same Xiph file has been in our screening pool
(`pool_kristen_sara*`), so earlier tuning may have seen it. From today the lens's families are sealed
(`nceval.pool.SEALED_FAMILIES`), so no screen can draw it again. The other sequences were never in our clip
sets.

### Deviations from the CTCs

| Deviation | Why |
|---|---|
| No B slices and no 4-reference GOP-8 structure in our streams; our fixed-QP mode is IPPP with a key QP offset and our own QP cadence | the project's real-time subset |
| VTM on A5 only; libaom speed 0 on A4/A5 only; A3's standards anchor is HM; nine slow reference points missing (above) | each launch had to finish within 30 minutes (owner rule). One 130-frame 270p point takes VTM 12.6 min and libaom speed 0 7.7 min on the cloud machine; some 360p libaom speed-0 points and one 720p HM point took longer than a launch allows |
| JVET class E: the first 150 of 600 frames | the same time rule (HM: ~15 min per QP-22 point at 150 frames) |
| Everything coded at 8 bits (VTM and VVenC `--InternalBitDepth=8`; the AOM CTC codes A4 at 10 bits) | 8-bit project |
| QP 42/47 beyond JVET's 22-37 | low-bitrate interest; reported separately in the JVET table |
| AOM CTC anchor is AVM (AV2); here libaom (AV1) at the CTC's LD settings | AVM at speed 0 is out of the time budget, and our AV2 is an initial subset |
| Tiles/threads: one thread and one tile everywhere (the CTC allows tiles on A2/A3 for speed) | single-threaded comparison |

### How it was run

| Launch | Instance | Jobs | Cost |
|---|---|---|---|
| pipe-clean | i-081b34d80e49fe9cc c7a.4xlarge | 69 (every encoder, 1 point) | $0.04 |
| wave 1, AOM shard 1/3 | i-0133497ee3fb2d76b c7a.48xlarge | 810 planned, 501 home | $0.60 |
| wave 1, AOM shard 2/3 | i-0e1ac037bbf95a768 c7a.48xlarge | 810 planned, 503 home | $0.64 |
| wave 2, shards 1-8 of the rest | i-02f81e2609a68dbcf, i-0d3bf80bf13577407, i-029abf0f1f83c0591, i-01624b147b409c78b, i-03a88fb75ebe0aa60, i-08ebf904ce690747f, i-079cfde8f0fceb6c9, i-016fe7b3b037e5fb7 (c7a.32xlarge) | 1,810 | $2.89 |
| cleanups (CBR extras left by stopped shards) | i-06fd68486b78c66ac, i-0b1791a9ab4d6e2a8 (c7a.32xlarge) | 290 + 228 | $0.11 |
| **total** | 13 spot launches, eu-north-1, each ≤ 30 min of running, ExpectedEnd ≤ 40 min | 2,805 points | **$4.28** |

Wave 1 lost its last ten minutes of results. SSH closes five minutes before a self-terminating instance
shuts down, and a checkpoint raced with bitstream deletion. Both are fixed in `tools/dev/aws_ctc.py`; the lost
jobs re-ran in wave 2.

```sh
# build HM-19.0, VTM-24.0, VVenC v1.14.0 (tags above) and put bin/{TAppEncoder,EncoderApp,vvencFFapp}
# and the three cfg files in REFS; our build at the verified head in NC_NANOCODEC_BIN
cd eval
python3 -m nceval.ctc stage --run-dir RUN --references REFS --conditions aom-ld,jvet-e [--shard I/N]
python3 ../tools/dev/aws_ctc.py --run-dir RUN --results RES --instance-type c7a.32xlarge --workers 64 \
    --lifetime 35 --minutes 26 --budget 1.5
python3 -m nceval.ctc report --results RES/results[,...] --out OUT     # points.csv, bd.csv, tables.md
```

Raw data: `docs/results/standards-ctc-2026-11-14/points.csv` (every point: rate, user time, all metrics),
`bd.csv` (every BD-rate with its overlap and reliability), and `tables.md` (all tables, including the
per-class ones for every codec).
