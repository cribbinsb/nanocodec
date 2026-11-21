# Bitrate vs encode speed

<sub>Updated 2026-11-21, build 730e4e16 (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, AV1, AV1 Fast, VP9, VP9 Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, AV1 Veryfast, HEVC Faster, VP9 Faster, VP9 Veryfast, HEVC Slow, AV1 Slow, VVC Slow, H.264 Slow, VP9 Slow, AV2 Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264, H.264 Slower, H.264 Slow, H.264 Fast, H.264 Faster, H.264 Veryfast, HEVC Veryslow, HEVC Slower, HEVC Slow, HEVC, HEVC Fast, HEVC Faster, AV1 Slower, AV1 Slow, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9 Slower, VP9 Slow, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2 Slower, AV2 Slow, AV2, AV2 Fast, AV2 Faster, VVC Veryslow, VVC Slower, VVC Slow, VVC, VVC Fast, VVC Faster timed on 207745d7 (same streams)</sub>

![bitrate vs speed](bd-vs-speed.svg)

**Rate control and the buffer.** Every encoder in the CBR charts is asked for a 1 s buffer at the target rate. x264, x265 and NVENC are held to it by a hard VBV, and SVT-AV1, libaom and libvpx by their 1000 ms real-time buffer. nanocodec 0.6.0 treats its buffer as a target with a 1.4x stop and sizes its key frames more generously: in these charts its buffer peak passes one buffer on 10-16 % of points (theirs: 0-1 %). openh264 is given no buffer setting and behaves like nanocodec in this respect. That is worth quality. On a sample of the held-out clips (24 clips at 150 frames, 12 at 600 frames; AV1, HEVC and H.264 medium), holding nanocodec's stream inside one buffer, as the references' is, cost it 3-5.5 % in bits at equal quality: its lead over SVT-AV1 M12 at 150 frames moved from 31 % to 28 % (AV1) and from 33 % to 29 % (HEVC), H.264 medium moved from level with SVT-AV1 M12 to about 5 % behind, and the leads over x264 medium moved by 0.6-2.2 points. The Fixed QP charts have no rate control and are not affected. A later release will make the buffer a hard bound by default and re-measure these charts.

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Veryslow** | -57.9% | -59.5% | -62.6% | -62.8 … -53.7 | 7.6 | 0.29× | 30.8 | 0.27× | 10/10 | Pareto; Pareto x86 |
| **VVC Slower** | -56.9% | -58.5% | -62.0% | -61.3 … -52.8 | 13.6 | 0.51× | 51.9 | 0.46× | 10/10 | Pareto; Pareto x86 |
| **AV2 Slower** | -56.9% | -61.0% | -61.5% | -61.9 … -53.3 | 7.9 | 0.30× | 26.6 | 0.24× | 10/10 |  |
| **VVC Slow** | -56.3% | -57.5% | -61.4% | -62.8 … -51.3 | 18.0 | 0.68× | 73.0 | 0.65× | 10/10 | Pareto; Pareto x86 |
| **VVC** | -54.4% | -55.8% | -60.1% | -60.7 … -49.3 | 25.0 | 0.94× | 101.7 | 0.90× | 10/10 | Pareto; Pareto x86 |
| **HEVC Veryslow** | -54.0% | -55.3% | -58.7% | -59.9 … -49.3 | 10.2 | 0.38× | 36.9 | 0.33× | 10/10 |  |
| **AV2 Slow** | -53.9% | -58.1% | -59.4% | -60.8 … -49.4 | 27.5 | 1.03× | 120.8 | 1.07× | 10/10 | Pareto; Pareto x86 |
| **AV1 Slower** | -53.8% | -57.6% | -56.6% | -61.6 … -48.9 | 14.5 | 0.54× | 59.5 | 0.53× | 10/10 |  |
| **AV2** | -52.9% | -57.0% | -58.8% | -60.8 … -48.5 | 35.4 | 1.33× | 150.7 | 1.34× | 10/10 | Pareto; Pareto x86 |
| **HEVC Slower** | -52.3% | -53.2% | -57.7% | -58.2 … -48.2 | 17.3 | 0.65× | 68.2 | 0.60× | 10/10 |  |
| **AV1 Slow** | -51.6% | -55.7% | -55.8% | -60.0 … -45.8 | 28.5 | 1.07× | 115.3 | 1.02× | 10/10 |  |
| **HEVC Slow** | -51.4% | -52.4% | -57.7% | -57.3 … -46.9 | 33.2 | 1.25× | 121.4 | 1.08× | 10/10 |  |
| **AV2 Fast** | -51.0% | -54.0% | -57.0% | -57.0 … -47.1 | 42.7 | 1.60× | 172.7 | 1.53× | 10/10 | Pareto; Pareto x86 |
| **VP9 Slower** | -50.7% | -54.4% | -53.4% | -55.1 … -46.2 | 13.0 | 0.49× | 51.5 | 0.46× | 10/10 |  |
| **VVC Fast** | -50.5% | -51.8% | -56.3% | -58.6 … -44.9 | 44.8 | 1.68× | 177.3 | 1.57× | 10/10 | Pareto; Pareto x86 |
| **AV1** | -50.1% | -54.5% | -54.3% | -59.6 … -44.0 | 45.2 | 1.70× | 179.0 | 1.59× | 10/10 | Pareto; Pareto x86 |
| **HEVC** | -49.1% | -50.1% | -55.6% | -55.3 … -44.3 | 45.2 | 1.69× | 167.9 | 1.49× | 10/10 |  |
| **VP9 Slow** | -49.0% | -52.9% | -51.7% | -55.8 … -43.2 | 21.9 | 0.82× | 91.8 | 0.81× | 10/10 |  |
| **AV1 Fast** | -48.4% | -52.4% | -52.8% | -57.8 … -42.5 | 64.7 | 2.42× | 252.9 | 2.24× | 10/10 | Pareto; Pareto x86 |
| **VVC Faster** | -47.6% | -48.9% | -54.4% | -55.7 … -41.9 | 59.4 | 2.23× | 219.8 | 1.95× | 10/10 |  |
| **VP9** | -46.9% | -50.7% | -49.9% | -53.4 … -41.3 | 36.6 | 1.37× | 145.6 | 1.29× | 10/10 |  |
| **AV2 Faster** | -46.7% | -49.0% | -50.7% | -56.1 … -41.2 | 61.9 | 2.32× | 265.7 | 2.36× | 10/10 | Pareto x86 |
| **AV1 Faster** | -45.7% | -48.5% | -51.3% | -56.5 … -41.3 | 77.1 | 2.89× | 313.5 | 2.78× | 10/10 | Pareto; Pareto x86 |
| **HEVC Fast** | -45.2% | -45.8% | -50.6% | -51.7 … -40.6 | 72.0 | 2.70× | 281.8 | 2.50× | 10/10 |  |
| **VP9 Fast** | -42.1% | -45.9% | -44.7% | -45.0 … -37.1 | 59.8 | 2.24× | 247.1 | 2.19× | 10/10 |  |
| SVT-AV1 M8 | -42.1% | -45.8% | -40.6% | -46.2 … -39.8 | 14.3 | 0.53× | 80.8 | 0.72× | 10/10 | 2 rate-invalid pts |
| x265 medium, metric-tuned (--tune psnr) | -41.5% | -42.6% | -45.7% | -46.3 … -35.6 | 8.0 | 0.30× | 38.0 | 0.34× | 10/10 | fps of the default-tuned run; re-timing pending |
| **VP9 Faster** | -40.5% | -42.2% | -42.5% | -52.7 … -34.0 | 79.2 | 2.97× | 311.9 | 2.77× | 10/10 | Pareto |
| **AV1 Veryfast** | -40.1% | -41.8% | -44.6% | -46.1 … -34.9 | 95.2 | 3.57× | 419.5 | 3.72× | 10/10 | Pareto; Pareto x86 |
| **HEVC Faster** | -40.0% | -40.3% | -45.6% | -46.0 … -34.3 | 90.0 | 3.37× | 350.2 | 3.11× | 10/10 |  |
| x265 medium | -37.9% | -37.0% | -40.4% | -42.8 … -33.1 | 8.0 | 0.30× | 38.0 | 0.34× | 10/10 |  |
| SVT-AV1 M10 | -37.7% | -39.9% | -39.2% | -41.6 … -33.1 | 39.3 | 1.47× | 228.9 | 2.03× | 10/10 | 2 rate-invalid pts |
| x265 superfast, metric-tuned (--tune psnr) | -37.6% | -37.8% | -41.8% | -45.1 … -30.2 | 12.1 | 0.45× | 57.9 | 0.51× | 10/10 | fps of the default-tuned run; re-timing pending |
| x265 superfast | -36.2% | -35.1% | -41.1% | -43.7 … -29.3 | 12.1 | 0.45× | 57.9 | 0.51× | 10/10 |  |
| x265 ultrafast, metric-tuned (--tune psnr) | -33.7% | -33.2% | -38.2% | -41.3 … -23.3 | 17.1 | 0.64× | 85.1 | 0.75× | 10/10 | fps of the default-tuned run; re-timing pending |
| **VP9 Veryfast** | -33.4% | -34.3% | -38.5% | -39.1 … -26.8 | 100.7 | 3.77× | 393.8 | 3.49× | 10/10 | Pareto |
| x265 ultrafast | -33.0% | -30.9% | -38.5% | -39.4 … -24.6 | 17.1 | 0.64× | 85.1 | 0.75× | 10/10 |  |
| SVT-AV1 M12 | -30.4% | -30.7% | -35.4% | -36.1 … -24.5 | 70.9 | 2.67× | 408.7 | 3.62× | 10/10 | 2 rate-invalid pts |
| SVT-AV1 M13 | -28.8% | -28.8% | -34.5% | -32.9 … -22.9 | 79.2 | 2.96× | 445.4 | 3.95× | 10/10 | Pareto x86; 1 rate-invalid pts |
| libaom rt 10 | -25.5% | -25.2% | -27.9% | -28.5 … -22.9 | 22.3 | 0.83× | 90.5 | 0.80× | 10/10 | 8 rate-invalid pts |
| **H.264 Slower** | -25.2% | -28.2% | -31.5% | -46.2 … -19.3 | 37.4 | 1.40× | 113.3 | 1.00× | 10/10 |  |
| libaom rt 8 | -25.1% | -22.9% | -28.4% | -26.4 … -22.7 | 17.4 | 0.65× | 68.8 | 0.61× | 10/10 | 6 rate-invalid pts |
| **H.264 Slow** | -24.9% | -27.8% | -31.4% | -46.5 … -19.2 | 51.9 | 1.95× | 160.7 | 1.43× | 10/10 |  |
| **H.264** | -23.1% | -25.5% | -30.4% | -45.8 … -16.6 | 66.2 | 2.48× | 198.4 | 1.76× | 10/10 |  |
| **H.264 Fast** | -17.9% | -19.2% | -26.0% | -42.0 … -11.7 | 96.9 | 3.63× | 283.4 | 2.51× | 10/10 |  |
| **H.264 Faster** | -16.3% | -18.0% | -23.5% | -40.8 … -9.4 | 118.3 | 4.44× | 369.8 | 3.28× | 10/10 | Pareto |
| x264 medium, metric-tuned (--tune psnr) | -8.1% | -6.8% | -16.6% | -10.2 … -6.1 | 26.7 | 1.00× | 112.8 | 1.00× | 10/10 | 5 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Veryfast** | -5.3% | -8.3% | -11.4% | -34.6 … +6.3 | 161.9 | 6.07× | 510.6 | 4.53× | 10/10 | Pareto; Pareto x86 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -3.3% | +2.6% | -8.2% | -8.9 … -0.4 | 47.8 | 1.78× | 204.4 | 1.81× | 10/10 | 17 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -2.5% | +2.6% | -7.9% | -8.1 … -0.7 | 47.8 | 1.78× | 204.4 | 1.81× | 10/10 | 16 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 10/10 | 4 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.4% | +6.2% | +1.1% | -6.2 … +6.2 | 30.0 | 1.12× | 122.7 | 1.09× | 10/10 | 13 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +1.6% | +5.2% | -4.2% | -6.7 … +8.5 | 30.0 | 1.12× | 122.7 | 1.09× | 10/10 | 15 rate-invalid pts |
| x264 veryfast | +7.6% | +6.6% | +4.2% | -12.6 … +15.3 | 85.4 | 3.18× | 363.3 | 3.22× | 10/10 |  |
| openh264 | +34.4% | +31.4% | +33.3% | -15.6 … +66.0 | 111.5 | 4.17× | 524.7 | 4.65× | 10/10 | Pareto x86; 22 rate-invalid pts |
| x264 ultrafast, metric-tuned (--tune ssim) | +40.1% | +53.4% | +27.1% | +15.1 … +50.5 | 190.5 | 7.11× | 1017.0 | 9.02× | 10/10 | fps of the default-tuned run; re-timing pending |
| x264 ultrafast | +48.6% | +63.6% | +17.3% | +9.3 … +64.5 | 190.5 | 7.11× | 1017.0 | 9.02× | 10/10 | Pareto; Pareto x86 |

Low rungs (40 / 75 kbps, every encoder) on: static_otw_homes390, static_cam29, static_london_bus, bodycam_airport. Clips left out for every encoder: 1; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 1, crowd 2, lowlight 2, pan 2, static 3 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slower | -25.2% | -30.1% |
| H.264 Slow | -24.9% | -29.8% |
| H.264 | -23.1% | -28.3% |
| H.264 Fast | -17.9% | -23.6% |
| H.264 Faster | -16.3% | -22.1% |
| H.264 Veryfast | -5.3% | -11.9% |
| HEVC Veryslow | -54.0% | -55.3% |
| HEVC Slower | -52.3% | -53.8% |
| HEVC Slow | -51.4% | -53.1% |
| HEVC | -49.1% | -51.1% |
| HEVC Fast | -45.2% | -47.1% |
| HEVC Faster | -40.0% | -42.0% |
| AV1 Slower | -53.8% | -55.5% |
| AV1 Slow | -51.6% | -53.6% |
| AV1 | -50.1% | -52.3% |
| AV1 Fast | -48.4% | -50.4% |
| AV1 Faster | -45.7% | -47.9% |
| AV1 Veryfast | -40.1% | -42.3% |
| VP9 Slower | -50.7% | -52.4% |
| VP9 Slow | -49.0% | -51.0% |
| VP9 | -46.9% | -48.9% |
| VP9 Fast | -42.1% | -43.8% |
| VP9 Faster | -40.5% | -43.1% |
| VP9 Veryfast | -33.4% | -36.0% |
| AV2 Slower | -56.9% | -58.3% |
| AV2 Slow | -53.9% | -55.6% |
| AV2 | -52.9% | -54.6% |
| AV2 Fast | -51.0% | -52.7% |
| AV2 Faster | -46.7% | -48.9% |
| VVC Veryslow | -57.9% | -59.3% |
| VVC Slower | -56.9% | -58.4% |
| VVC Slow | -56.3% | -57.9% |
| VVC | -54.4% | -56.1% |
| VVC Fast | -50.5% | -52.5% |
| VVC Faster | -47.6% | -49.7% |
| openh264 | +34.4% | +22.6% |
| x264 ultrafast | +48.6% | +40.9% |
| x264 veryfast | +7.6% | +3.2% |
| x265 ultrafast | -33.0% | -33.0% |
| x265 superfast | -36.2% | -36.3% |
| x265 medium | -37.9% | -37.6% |
| SVT-AV1 M8 | -42.1% | -42.9% |
| SVT-AV1 M10 | -37.7% | -38.7% |
| SVT-AV1 M12 | -30.4% | -32.3% |
| SVT-AV1 M13 | -28.8% | -31.0% |
| libvpx VP9 rt 7 | +1.6% | +3.4% |
| libvpx VP9 rt 9 | -2.5% | -3.0% |
| libaom rt 8 | -25.1% | -25.9% |
| libaom rt 10 | -25.5% | -26.9% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +34.4% | +31.4% | +33.3% | +38.4% | 10/10 |  |
| openh264, adaptive quantisation on | +34.4% | +31.4% | +33.3% | +38.4% | 10/10 | -0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +48.6% | +63.6% | +17.3% | +64.9% | 10/10 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +48.7% | +63.8% | +17.3% | +64.9% | 10/10 | +0.1 [-0.0, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) ● | +40.1% | +53.4% | +27.1% | +39.8% | 10/10 | -3.5 [-6.1, -0.9] better |
| x264 veryfast | +7.6% | +6.6% | +4.2% | +11.9% | 10/10 |  |
| x264 veryfast, metric-tuned (--tune psnr) | +8.2% | +9.4% | -8.3% | +23.5% | 10/10 | +0.1 [-1.1, +1.2] not distinguishable |
| x264 veryfast, metric-tuned (--tune ssim) | +7.2% | +7.3% | +3.8% | +10.5% | 10/10 | -0.3 [-1.0, +0.6] not distinguishable |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 10/10 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -8.1% | -6.8% | -16.6% | -1.0% | 10/10 | -8.1 [-8.8, -7.3] better |
| x264 medium, metric-tuned (--tune ssim) | -0.8% | -0.7% | +1.2% | -2.8% | 10/10 | -0.8 [-1.5, +0.6] not distinguishable |
| x265 ultrafast | -33.0% | -30.9% | -38.5% | -29.5% | 10/10 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -33.7% | -33.2% | -38.2% | -29.8% | 10/10 | -1.2 [-2.3, -0.1] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -33.7% | -33.2% | -38.2% | -29.8% | 10/10 | -1.2 [-2.3, -0.1] better |
| x265 superfast | -36.2% | -35.1% | -41.1% | -32.3% | 10/10 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -37.6% | -37.8% | -41.8% | -33.1% | 10/10 | -2.4 [-3.1, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -37.6% | -37.8% | -41.8% | -33.1% | 10/10 | -2.4 [-3.1, -1.6] better |
| x265 medium | -37.9% | -37.0% | -40.4% | -36.4% | 10/10 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -41.5% | -42.6% | -45.7% | -36.2% | 10/10 | -5.9 [-7.5, -4.4] better |
| x265 medium, metric-tuned (--tune ssim) | -39.1% | -39.3% | -40.9% | -37.1% | 10/10 | -1.9 [-2.8, -1.1] better |
| libvpx VP9 rt 7 | +1.6% | +5.2% | -4.2% | +3.7% | 10/10 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +1.4% | +6.2% | +1.1% | -3.1% | 10/10 | -1.0 [-2.1, -0.0] better |
| libvpx VP9 rt 9 | -2.5% | +2.6% | -7.9% | -2.3% | 10/10 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | -3.3% | +2.6% | -8.2% | -4.4% | 10/10 | -1.1 [-1.8, -0.4] better |
| libaom rt 8 | -25.1% | -22.9% | -28.4% | -24.1% | 10/10 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -26.3% | -22.6% | -25.2% | -30.9% | 10/10 | -0.2 [-2.0, +1.5] not distinguishable |
| libaom rt 10 | -25.5% | -25.2% | -27.9% | -23.4% | 10/10 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -26.1% | -24.5% | -24.7% | -29.2% | 10/10 | +0.4 [-1.1, +1.8] not distinguishable |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Veryslow** | -51.8% | -52.1% | -51.0% | -52.2% | 10/10 |
| **AV2 Slower** | -50.9% | -53.7% | -50.2% | -48.8% | 10/10 |
| **VVC Slower** | -50.7% | -50.9% | -50.2% | -51.1% | 10/10 |
| **VVC Slow** | -49.8% | -49.5% | -49.3% | -50.6% | 10/10 |
| **VVC** | -47.8% | -47.6% | -47.7% | -48.1% | 10/10 |
| **AV2 Slow** | -47.5% | -50.4% | -47.4% | -44.7% | 10/10 |
| **HEVC Veryslow** | -47.5% | -47.5% | -46.2% | -48.7% | 10/10 |
| **AV1 Slower** | -47.1% | -49.9% | -43.8% | -47.7% | 10/10 |
| **AV2** | -46.4% | -49.1% | -46.6% | -43.6% | 10/10 |
| **HEVC Slower** | -45.4% | -44.8% | -44.8% | -46.5% | 10/10 |
| **AV1 Slow** | -44.7% | -47.6% | -42.6% | -43.8% | 10/10 |
| **HEVC Slow** | -44.4% | -43.8% | -44.6% | -44.8% | 10/10 |
| **AV2 Fast** | -43.9% | -45.5% | -43.9% | -42.5% | 10/10 |
| **VP9 Slower** | -43.5% | -46.4% | -39.2% | -45.0% | 10/10 |
| **VVC Fast** | -43.3% | -42.9% | -42.9% | -44.2% | 10/10 |
| **AV1** | -43.0% | -46.4% | -40.8% | -41.7% | 10/10 |
| **HEVC** | -41.9% | -41.4% | -42.2% | -42.2% | 10/10 |
| **VP9 Slow** | -41.6% | -44.7% | -37.1% | -43.0% | 10/10 |
| **AV1 Fast** | -40.9% | -43.8% | -38.7% | -40.3% | 10/10 |
| **VVC Faster** | -40.1% | -39.7% | -40.6% | -40.0% | 10/10 |
| **VP9** | -39.2% | -42.3% | -34.8% | -40.3% | 10/10 |
| **AV2 Faster** | -38.9% | -39.9% | -36.2% | -40.8% | 10/10 |
| **AV1 Faster** | -37.9% | -39.4% | -36.6% | -37.6% | 10/10 |
| **HEVC Fast** | -37.4% | -36.6% | -35.8% | -39.8% | 10/10 |
| **VP9 Fast** | -33.7% | -36.7% | -28.6% | -35.8% | 10/10 |
| SVT-AV1 M8 | -33.6% | -36.7% | -23.5% | -40.5% | 10/10 |
| x265 medium, metric-tuned (--tune psnr) | -33.5% | -33.8% | -30.4% | -36.2% | 10/10 |
| **VP9 Faster** | -31.7% | -32.7% | -25.8% | -36.5% | 10/10 |
| **HEVC Faster** | -31.6% | -31.1% | -29.8% | -34.0% | 10/10 |
| **AV1 Veryfast** | -31.6% | -32.5% | -28.4% | -33.8% | 10/10 |
| x265 medium, metric-tuned (--tune ssim) | -30.5% | -30.1% | -24.3% | -37.3% | 10/10 |
| x265 medium | -29.2% | -27.3% | -23.6% | -36.6% | 10/10 |
| x265 superfast, metric-tuned (--tune psnr) | -28.8% | -28.1% | -25.1% | -33.1% | 10/10 |
| x265 superfast, metric-tuned (--tune ssim) | -28.8% | -28.1% | -25.1% | -33.1% | 10/10 |
| SVT-AV1 M10 | -28.5% | -29.9% | -21.5% | -34.0% | 10/10 |
| x265 superfast | -27.0% | -24.9% | -24.0% | -32.3% | 10/10 |
| **VP9 Veryfast** | -24.5% | -25.1% | -21.5% | -26.9% | 10/10 |
| x265 ultrafast, metric-tuned (--tune psnr) | -24.4% | -23.3% | -20.5% | -29.3% | 10/10 |
| x265 ultrafast, metric-tuned (--tune ssim) | -24.4% | -23.3% | -20.5% | -29.3% | 10/10 |
| x265 ultrafast | -23.5% | -20.8% | -20.7% | -29.0% | 10/10 |
| SVT-AV1 M12 | -20.6% | -20.5% | -17.0% | -24.2% | 10/10 |
| SVT-AV1 M13 | -19.0% | -18.8% | -16.0% | -22.2% | 10/10 |
| **H.264 Slower** | -16.8% | -20.2% | -15.4% | -14.8% | 10/10 |
| **H.264 Slow** | -16.5% | -19.6% | -15.2% | -14.6% | 10/10 |
| libaom rt 8, metric-tuned (--tune ssim) | -16.3% | -12.9% | -5.3% | -30.7% | 10/10 |
| libaom rt 10, metric-tuned (--tune ssim) | -16.0% | -14.6% | -4.4% | -29.0% | 10/10 |
| libaom rt 10 | -15.0% | -15.1% | -6.7% | -23.2% | 10/10 |
| libaom rt 8 | -14.9% | -13.1% | -7.8% | -24.0% | 10/10 |
| **H.264** | -14.5% | -17.3% | -13.9% | -12.4% | 10/10 |
| **H.264 Fast** | -9.2% | -11.1% | -9.2% | -7.3% | 10/10 |
| **H.264 Faster** | -7.2% | -9.6% | -6.1% | -5.8% | 10/10 |
| **H.264 Veryfast** | +4.8% | +0.2% | +8.4% | +5.9% | 10/10 |
| x264 medium, metric-tuned (--tune ssim) | +9.3% | +7.4% | +22.1% | -1.6% | 10/10 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +9.7% | +15.1% | +18.0% | -4.0% | 10/10 |
| x264 medium | +9.9% | +7.6% | +20.8% | +1.2% | 10/10 |
| libvpx VP9 rt 9 | +10.7% | +15.5% | +18.5% | -1.8% | 10/10 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +13.5% | +17.2% | +25.6% | -2.2% | 10/10 |
| libvpx VP9 rt 7 | +14.5% | +17.3% | +21.8% | +4.5% | 10/10 |
| x264 veryfast, metric-tuned (--tune ssim) | +17.8% | +15.0% | +26.4% | +12.0% | 10/10 |
| x264 veryfast | +18.3% | +14.4% | +27.1% | +13.4% | 10/10 |
| x264 veryfast, metric-tuned (--tune psnr) | +18.3% | +17.5% | +11.9% | +25.6% | 10/10 |
| openh264, adaptive quantisation on | +52.1% | +47.9% | +68.1% | +40.3% | 10/10 |
| openh264 | +52.1% | +47.9% | +68.1% | +40.3% | 10/10 |
| x264 ultrafast, metric-tuned (--tune ssim) | +52.2% | +60.4% | +53.8% | +42.5% | 10/10 |
| x264 ultrafast | +60.8% | +70.9% | +43.0% | +68.6% | 10/10 |
| x264 ultrafast, metric-tuned (--tune psnr) | +60.8% | +71.0% | +42.9% | +68.6% | 10/10 |

## Hardware encoders

NVIDIA NVENC (NVIDIA GeForce RTX 5090 NVENC, driver 610.57.04, ffmpeg 7.1.1) through ffmpeg, in low-delay CBR on the same clips, rungs and buffer as every point above, no B-frames; scored the same way; p4 is preset P4 with `-tune ll`, p7 preset P7 with `-tune hq`. Speeds are GPU wall-clock frames per second at 720p (fixed-function hardware, not CPU time; tools/dev/nvenc_timing.py): the aggregate throughput of N parallel streams (the best of 1, 2, 4 and 8) and one stream, both encoding frames already in GPU memory, and one stream through ffmpeg's read-convert-upload pipeline (the earlier figure, bound by that pipeline). The x86 chart plots the aggregate, never on the Pareto front; the Jetson chart leaves them out.

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | aggregate fps (N streams) | 1-stream encode fps | 1-stream fps via ffmpeg pipeline | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| NVENC HEVC p4 | -40.5% | -41.1% | -44.6% | -40.8 … -36.9 | 2815 (8) | 798 | 328 | 10/10 | 3 rate-invalid pts |
| NVENC HEVC p7 | -40.3% | -41.1% | -44.4% | -42.1 … -36.5 | 1160 (8) | 336 | 224 | 10/10 | 3 rate-invalid pts |
| NVENC AV1 p7 | -36.7% | -40.6% | -35.8% | -40.9 … -29.7 | 2291 (8) | 679 | 202 | 10/10 | 8 rate-invalid pts |
| NVENC AV1 p4 | -33.3% | -37.3% | -32.3% | -38.9 … -23.9 | 3785 (8) | 1065 | 249 | 10/10 | 9 rate-invalid pts |
| NVENC H.264 p7 | -14.2% | -14.1% | -21.8% | -24.7 … -5.9 | 2038 (8) | 548 | 319 | 10/10 | 8 rate-invalid pts |
| NVENC H.264 p4 | -13.8% | -13.5% | -21.6% | -24.9 … -5.1 | 3358 (8) | 949 | 352 | 10/10 | 8 rate-invalid pts |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **bodycam_airport** (bodycam): no reliable overlap for AV2 Slower.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | bodycam_airport |
|---|---:|
| VVC Veryslow | -82.4% |
| VVC Slower | -82.6% |
| AV2 Slower | n/a |
| VVC Slow | -81.6% |
| VVC | -81.5% |
| HEVC Veryslow | -76.5% |
| AV2 Slow | -82.5% |
| AV1 Slower | -80.3% |
| AV2 | -82.0% |
| HEVC Slower | -76.5% |
| AV1 Slow | -81.0% |
| HEVC Slow | -75.5% |
| AV2 Fast | -80.4% |
| VP9 Slower | -75.9% |
| VVC Fast | -80.2% |
| AV1 | -80.3% |
| HEVC | -75.1% |
| VP9 Slow | -75.0% |
| AV1 Fast | -78.0% |
| VVC Faster | -75.8% |
| VP9 | -74.9% |
| AV2 Faster | -79.9% |
| AV1 Faster | -77.2% |
| HEVC Fast | -70.6% |
| VP9 Fast | -72.2% |
| SVT-AV1 M8 | -75.3% |
| x265 medium, metric-tuned (--tune psnr) | n/a |
| VP9 Faster | -72.9% |
| AV1 Veryfast | -73.5% |
| HEVC Faster | -68.7% |
| x265 medium | -67.5% |
| SVT-AV1 M10 | -72.5% |
| x265 superfast, metric-tuned (--tune psnr) | n/a |
| x265 superfast | -63.4% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a |
| VP9 Veryfast | -58.8% |
| x265 ultrafast | -61.5% |
| SVT-AV1 M12 | -71.0% |
| SVT-AV1 M13 | -68.3% |
| libaom rt 10 | -65.4% |
| H.264 Slower | -58.5% |
| libaom rt 8 | -57.8% |
| H.264 Slow | -58.7% |
| H.264 | -57.2% |
| H.264 Fast | -47.9% |
| H.264 Faster | -47.8% |
| x264 medium, metric-tuned (--tune psnr) | n/a |
| H.264 Veryfast | -43.2% |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a |
| libvpx VP9 rt 9 | -34.9% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a |
| libvpx VP9 rt 7 | -32.2% |
| x264 veryfast | -28.8% |
| openh264 | -27.8% |
| x264 ultrafast, metric-tuned (--tune ssim) | n/a |
| x264 ultrafast | -21.3% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slower | openh264 | -40.8% | -41.7% | -45.9% | -47.0 … -31.8 | 11/11 |
| H.264 Slow | openh264 | -40.6% | -41.2% | -46.0% | -46.8 … -32.0 | 11/11 |
| H.264 | openh264 | -39.2% | -39.5% | -45.3% | -45.6 … -30.2 | 11/11 |
| H.264 | x264 veryfast | -35.1% | -36.2% | -40.9% | -47.9 … -24.3 | 11/11 |
| H.264 | x264 medium | -28.6% | -30.6% | -36.5% | -52.5 … -16.7 | 11/11 |
| H.264 Fast | openh264 | -33.7% | -32.9% | -41.6% | -38.8 … -24.7 | 11/11 |
| H.264 Fast | x264 veryfast | -29.5% | -29.5% | -36.2% | -40.1 … -18.9 | 11/11 |
| H.264 Fast | H.264 | +9.8% | +11.8% | +7.3% | +6.4 … +10.9 | 11/11 |
| H.264 Faster | openh264 | -32.0% | -31.8% | -40.0% | -35.6 … -22.5 | 11/11 |
| H.264 Faster | x264 ultrafast | -47.3% | -53.4% | -41.0% | -55.6 … -41.4 | 10/11 |
| H.264 Faster | H.264 Fast | +2.1% | +0.9% | +3.4% | +0.5 … +2.8 | 11/11 |
| H.264 Veryfast | openh264 | -20.2% | -21.6% | -29.7% | -27.7 … -7.2 | 11/11 |
| H.264 Veryfast | x264 ultrafast | -41.8% | -49.7% | -32.8% | -50.5 … -35.3 | 11/11 |
| H.264 Veryfast | H.264 Faster | +15.6% | +14.1% | +13.9% | +6.6 … +20.3 | 11/11 |
| HEVC Veryslow | SVT-AV1 M12 | -37.2% | -37.4% | -40.0% | -47.4 … -25.9 | 11/11 |
| HEVC Slower | SVT-AV1 M12 | -34.9% | -34.5% | -38.1% | -42.3 … -27.4 | 11/11 |
| HEVC Slow | SVT-AV1 M12 | -33.8% | -33.2% | -39.0% | -41.0 … -27.6 | 11/11 |
| HEVC | SVT-AV1 M12 | -30.3% | -29.7% | -35.6% | -37.1 … -23.4 | 11/11 |
| HEVC | x265 medium | -22.5% | -23.8% | -31.6% | -37.4 … -9.9 | 11/11 |
| HEVC Fast | SVT-AV1 M12 | -22.1% | -20.5% | -25.5% | -28.1 … -16.6 | 11/11 |
| HEVC Fast | x265 superfast | -18.2% | -19.3% | -21.1% | -28.9 … -8.4 | 11/11 |
| HEVC Fast | HEVC | +11.7% | +12.4% | +15.6% | +7.6 … +13.7 | 11/11 |
| HEVC Faster | SVT-AV1 M12 | -13.2% | -11.4% | -17.4% | -17.5 … -8.5 | 11/11 |
| HEVC Faster | SVT-AV1 M13 | -17.7% | -15.9% | -21.1% | -22.5 … -11.0 | 11/11 |
| HEVC Faster | HEVC Fast | +12.0% | +12.1% | +11.6% | +7.1 … +15.7 | 11/11 |
| AV1 Slower | SVT-AV1 M12 | -38.7% | -41.8% | -37.8% | -43.2 … -34.9 | 11/11 |
| AV1 Slow | SVT-AV1 M12 | -36.3% | -39.3% | -38.5% | -41.2 … -32.3 | 11/11 |
| AV1 | SVT-AV1 M12 | -34.1% | -37.6% | -36.0% | -38.7 … -29.8 | 11/11 |
| AV1 | openh264 | -60.2% | -61.9% | -62.3% | -68.0 … -48.5 | 11/11 |
| AV1 | SVT-AV1 M8 | -14.4% | -13.9% | -25.7% | -24.3 … -5.2 | 11/11 |
| AV1 | libaom rt 8 | -33.4% | -37.0% | -38.7% | -45.4 … -27.2 | 11/11 |
| AV1 Fast | SVT-AV1 M12 | -30.1% | -33.3% | -30.9% | -31.7 … -27.4 | 11/11 |
| AV1 Fast | openh264 | -58.2% | -59.4% | -60.0% | -66.9 … -45.4 | 11/11 |
| AV1 Fast | AV1 | +6.3% | +7.0% | +8.2% | +3.0 … +8.4 | 11/11 |
| AV1 Faster | SVT-AV1 M12 | -25.8% | -27.4% | -28.3% | -30.0 … -20.1 | 11/11 |
| AV1 Faster | AV1 Fast | +5.9% | +8.7% | +3.2% | +3.6 … +9.0 | 11/11 |
| AV1 Veryfast | SVT-AV1 M12 | -15.6% | -14.8% | -16.7% | -25.1 … -6.5 | 11/11 |
| AV1 Veryfast | SVT-AV1 M13 | -20.1% | -19.5% | -20.4% | -26.0 … -13.3 | 11/11 |
| AV1 Veryfast | AV1 Faster | +14.8% | +18.6% | +16.9% | +9.3 … +19.8 | 11/11 |
| VP9 Slower | HEVC Slower | +3.7% | -3.6% | +12.0% | -0.4 … +6.6 | 11/11 |
| VP9 Slow | HEVC Slow | +6.3% | -1.4% | +16.5% | +1.6 … +11.7 | 11/11 |
| VP9 | HEVC | +5.7% | -1.8% | +15.1% | +1.2 … +11.1 | 11/11 |
| VP9 | libvpx VP9 rt 7 | -47.0% | -50.9% | -46.5% | -61.4 … -40.3 | 11/11 |
| VP9 Fast | HEVC Fast | +6.6% | -1.6% | +14.1% | -1.0 … +12.1 | 11/11 |
| VP9 Fast | libvpx VP9 rt 9 | -41.9% | -48.0% | -40.7% | -49.0 … -35.7 | 11/11 |
| VP9 Fast | VP9 | +12.1% | +12.0% | +14.1% | +8.2 … +13.9 | 11/11 |
| VP9 Faster | HEVC Faster | -2.3% | -5.9% | +6.0% | -7.6 … +0.8 | 11/11 |
| VP9 Faster | libvpx VP9 rt 9 | -40.2% | -44.4% | -38.1% | -52.3 … -30.7 | 11/11 |
| VP9 Faster | VP9 Fast | +2.9% | +7.3% | +3.5% | +0.2 … +7.6 | 11/11 |
| VP9 Veryfast | HEVC Faster | +17.3% | +13.9% | +16.1% | +14.4 … +19.5 | 11/11 |
| VP9 Veryfast | libvpx VP9 rt 9 | -30.3% | -34.8% | -33.0% | -43.0 … -19.9 | 11/11 |
| VP9 Veryfast | VP9 Faster | +22.2% | +22.0% | +10.1% | +14.5 … +23.9 | 11/11 |
| AV2 Slower | AV1 Slower | -8.8% | -9.9% | -13.4% | -9.8 … -7.4 | 11/11 |
| AV2 Slow | AV1 Slow | -6.6% | -7.4% | -9.3% | -8.7 … -3.9 | 11/11 |
| AV2 | AV1 | -7.2% | -7.1% | -10.9% | -11.0 … -4.6 | 11/11 |
| AV2 | SVT-AV1 M8 | -20.8% | -20.3% | -34.2% | -29.8 … -14.6 | 11/11 |
| AV2 Fast | AV1 Fast | -7.0% | -5.0% | -12.0% | -11.4 … -1.8 | 11/11 |
| AV2 Fast | AV2 | +6.4% | +9.3% | +6.4% | +2.7 … +9.5 | 11/11 |
| AV2 Faster | AV1 Faster | -3.3% | -3.0% | +0.2% | -7.8 … +1.0 | 11/11 |
| AV2 Faster | SVT-AV1 M12 | -29.1% | -30.2% | -28.6% | -36.6 … -24.4 | 11/11 |
| AV2 Faster | AV2 Fast | +10.3% | +11.2% | +17.1% | +5.6 … +12.5 | 11/11 |
| VVC Veryslow | HEVC Veryslow | -10.6% | -11.3% | -9.2% | -13.3 … -6.4 | 11/11 |
| VVC Slower | HEVC Slower | -13.2% | -14.6% | -11.6% | -14.5 … -10.0 | 11/11 |
| VVC Slow | HEVC Slow | -12.7% | -13.4% | -9.7% | -15.0 … -9.2 | 11/11 |
| VVC | HEVC | -13.2% | -14.2% | -11.2% | -16.7 … -7.9 | 11/11 |
| VVC | x265 medium | -32.4% | -34.2% | -39.3% | -47.9 … -21.9 | 11/11 |
| VVC Fast | HEVC Fast | -14.3% | -15.4% | -14.5% | -18.1 … -7.5 | 11/11 |
| VVC Fast | VVC | +10.5% | +11.2% | +11.5% | +7.4 … +14.4 | 11/11 |
| VVC Faster | HEVC Faster | -17.3% | -18.6% | -19.8% | -19.4 … -14.3 | 11/11 |
| VVC Faster | VVC Fast | +9.3% | +8.7% | +4.8% | +6.6 … +8.4 | 11/11 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 730e4e16 (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, AV1, AV1 Fast, VP9, VP9 Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, AV1 Veryfast, HEVC Faster, VP9 Faster, VP9 Veryfast, HEVC Slow, AV1 Slow, VVC Slow, H.264 Slow, VP9 Slow, AV2 Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264, H.264 Slower, H.264 Slow, H.264 Fast, H.264 Faster, H.264 Veryfast, HEVC Veryslow, HEVC Slower, HEVC Slow, HEVC, HEVC Fast, HEVC Faster, AV1 Slower, AV1 Slow, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9 Slower, VP9 Slow, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2 Slower, AV2 Slow, AV2, AV2 Fast, AV2 Faster, VVC Veryslow, VVC Slower, VVC Slow, VVC, VVC Fast, VVC Faster timed on 207745d7 (same streams), its presets), openh264
  realtime (medium complexity, CAVLC), SVT-AV1 v4.3.0 `--rtc --rc 2` (CBR, 1 s buffer),
  x264 / x265 4.2 as the anchor (zero latency, no B-frames, one thread / no pools),
  libvpx v1.16.0 VP9 and libaom v3.15.3 AV1 in realtime CBR (`--rt --end-usage=cbr
  --lag-in-frames=0 --threads=1`, `--cpu-used` as named, 1 s buffer, frame dropping off).
  A point is counted as rate-invalid (it stays on its curve) when its 1 s leaky bucket,
  drained at the target, peaks above 1.5x the buffer, or its average rate exceeds the
  target by more than 10% plus the buffer's share of the window (+30% at 150 frames);
  ours are judged without the first key frame's outstanding loan (up to about 1.4
  buffers, repaid over the following seconds), the references with a strict 1 s VBV.
  The BD axis uses every bit. No encoder here drops frames; ours do not code input
  frames that would not change the picture (holds: no bits, scored as the repeated
  picture), the references code every frame.
- **Ladder and clips:** 150 / 300 / 600 / 1200 kbps at 720p30, scaled by pixel rate and
  capped at half the source bitrate; 11 natural VAL clips in 5 content
  classes, 150 frames, one key frame. Metrics are scored on our encoder's
  reconstruction (the references' on their standard-decoded output); the build passed
  the landing conformance check (short streams of every shipped preset decoded by two
  standard decoders and ours, bit-exact with the reconstruction); the chart streams
  themselves are not each decoded.
- **Quality axis:** BD-rate on the achieved-rate curves (PCHIP, reliable-overlap rule).
  BD-J = (CB(Q_full) + CB(Q_low)) / 2, Q = mean BD-rate over PSNR-YUV, VMAF v1 and MS-SSIM,
  CB = mean over content classes of the class means, Q_low = the lower-rate half of
  the overlap (the J_real quality composite, eval/nceval/jreal.py). BD-PSNR-YUV and BD-VMAF v1 are the
  same with PSNR-YUV or VMAF v1 alone. The bars are the interquartile
  range of the per-clip BD-J. **Low rungs:** x264 medium undershoots its targets and has
  lower quality than most points at a given rate, so on some clips (static,
  bodycam_airport) their whole curves lay above its quality range, with no reliable
  overlap. On every clip where some point has no reliable overlap on the ladder above,
  EVERY encoder (ours and the references, the anchor too) also runs 40 / 75 kbps (720p30,
  scaled the same way), which reaches down into the anchor's range; BD is then direct
  over the overlap. **One clip set for all:** every point is averaged over the same clips,
  those on which every point overlaps the anchor directly; a clip where some point still
  has none is left out for every point (listed, with each point's direct BD there, under
  "Excluded clips"), unless that would remove a whole content class: then its clips are
  kept by chaining (‡) every unscored point through ONE intermediate encoder that
  overlaps both, BD(point→anchor) = (1 + BD(point→intermediate)) × (1 + BD(intermediate→
  anchor)) − 1, per metric and term, the second leg always direct (never two hops), the
  intermediate of the point's own codec lineage first, then of its own side, then the best
  overlap, then by name, and only when every unscored point chains.
  Nothing is extrapolated; a point with BD on fewer than 50% of the clips is
  flagged.
- **Speed axis:** Jetson Orin Nano, Cortex-A78AE core 3 at 1.728 GHz, one thread, encode CPU time per frame (rusage), median of three
  rotated rounds after a warm-up, the anchor timed in the same lock window as the point
  or, since the core's clock is locked and the anchor repeats within ~0.3 ms/frame
  across lock windows, one fresh anchor timing reused for several points; fixed
  quantizer chosen on the PC to land each encoder at about 600 kbps
  (150 frames, 720p) on lowlight_city_night, crowd_mot17_09 and bodycam_bc1;
  fps = 1000 / geometric-mean ms per frame; speed = geometric mean of the per-clip
  anchor / point time ratios.
- **x86 speed (`bd-vs-speed-x86.svg`, `.csv`):** the same streams (same quantizers and
  binaries' x86 builds) on a Ryzen 9 5950X, a shared machine, so each run is pinned
  to one logical CPU (one per physical core) and every point is paired with an x264
  medium run on the same CPU in interleaved rounds (machine load slows both); the
  minimum of 5 rounds per clip, after a warm-up.
  Refresh of 2026-11-11: only our points were re-timed (the reference encoders' timings
  are from the earlier session). The PC ran about 9% faster that night (paired x264 medium
  73 vs 67.6 fps), so each of our per-clip times is scaled by the reference session's x264
  medium time over its paired x264 medium time; speed ratios are unchanged and the raw
  times are kept in `bd-vs-speed-timing-x86.json` (`raw_ms_per_frame`).
- **Caveats:** our VVC and AV2 are young (rate control and quality work are
  ongoing); the main speed axis is single-core on one ARM core (x86 ratios differ,
  mostly in SIMD coverage); fixed-QP timing switches x264/x265 adaptive quantisation off
  (cheaper than their CBR mode by a few percent); the quality axis is 11 short clips,
  so differences of a few percent are within the per-clip spread, and these VAL clips
  were also used to rank our candidate changes (see heldout-chart-*.md for 71 other
  clips); the reference encoders' real-time modes are tuned for latency, not for these
  metrics, and their other modes (threads, lookahead, B-frames) may do better.
- **Metric-tuned references:** x264 and x265 default to psychovisual optimisations
  (psy-rd, psy-rdoq, adaptive quantisation) tuned for how the picture looks, which
  PSNR, VMAF and MS-SSIM cannot judge, and no subjective test was run here. So each of
  them is also shown at the metric-oriented tuning its own documentation provides,
  choosing the one that scores best for it on our composite; our encoders are tuned
  to that composite. x264 0.165 `--tune zerolatency,psnr` (`--aq-mode 0 --no-psy`) and
  `--tune zerolatency,ssim` (`--aq-mode 2 --no-psy`); x265 4.2 `--tune psnr`
  (`--aq-strength 0 --psy-rd 0 --psy-rdoq 0`) and `--tune ssim` (`--aq-mode 2
  --psy-rd 0 --psy-rdoq 0`), with `--bframes 0 --rc-lookahead 0 --no-scenecut
  --no-cutree --frame-threads 1` restoring x265's zero-latency settings (it applies
  one `--tune`; checked: the same encoder settings as zerolatency with those psy and AQ
  values set by hand, decoded pictures identical); every other setting as above. Only
  x264 and x265 run at visually tuned defaults on this chart: SVT-AV1 `--rtc` already
  runs its PSNR tune (it offers no other in low delay), libaom and libvpx default to
  `--tune psnr` with AQ off, and openh264 and NVENC run with adaptive quantisation off. They are scored on the chart's clip set and rungs and never change
  it. The libaom and libvpx `--tune ssim` and openh264 adaptive-quantisation variants
  are measured the same way. A variant is plotted only where it is distinguishably
  better for its encoder than the charted setting: its direct paired BD-J against that
  setting, on the same clips and rungs, is negative and the 95% bootstrap interval
  (clips resampled within content class) excludes zero; with two tunes, the larger
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 ultrafast --tune ssim (-3.5 [-6.1, -0.9]%); x264 medium --tune psnr (-8.1 [-8.8, -7.3]%); x265 ultrafast --tune psnr (-1.2 [-2.3, -0.1]%); x265 superfast --tune psnr (-2.4 [-3.1, -1.6]%); x265 medium --tune psnr (-5.9 [-7.5, -4.4]%); libvpx VP9 rt 7 --tune ssim (-1.0 [-2.1, -0.0]%); libvpx VP9 rt 9 --tune ssim (-1.1 [-1.8, -0.4]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
