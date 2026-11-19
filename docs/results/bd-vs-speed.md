# Bitrate vs encode speed

<sub>Updated 2026-11-19, build 730e4e16 (H.264 Fast, H.264 Faster, H.264 Veryfast, VP9 Faster: 478e8249; VP9, AV1 Slow: 77e93089; AV1, AV1 Fast, VP9 Fast, AV1 Veryfast: b33ba3bb; VP9 Slow, AV2 Slow: bad14976; H.264, HEVC, HEVC Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, HEVC Faster, VP9 Veryfast, HEVC Slow, VVC Slow, H.264 Slow, AV2 Faster: c9cff89f); H.264 Fast, H.264 Faster, H.264 Veryfast, VP9 Faster, AV1 Slow, VP9, AV2 Slow, VP9 Slow, AV1, AV1 Fast, AV1 Veryfast, VP9 Fast timed on c9cff89f (same streams)</sub>

![bitrate vs speed](bd-vs-speed.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **AV2 Slow** | -56.3% | -59.9% | -61.6% | -61.2 … -49.0 | 22.0 | 0.82× | 81.6 | 0.72× | 11/11 | Pareto; Pareto x86 |
| **VVC Slow** | -56.1% | -56.8% | -61.6% | -60.9 … -49.4 | 24.8 | 0.93× | 97.7 | 0.87× | 11/11 | Pareto; Pareto x86 |
| **VVC** | -56.0% | -56.7% | -61.6% | -62.3 … -48.0 | 28.9 | 1.08× | 118.1 | 1.05× | 11/11 | Pareto; Pareto x86 |
| **AV2** | -55.0% | -58.7% | -60.6% | -59.6 … -47.5 | 37.9 | 1.42× | 164.2 | 1.46× | 11/11 | Pareto; Pareto x86 |
| **AV1 Slow** | -53.5% | -57.4% | -58.4% | -61.8 … -44.3 | 29.7 | 1.11× | 116.6 | 1.03× | 11/11 |  |
| **AV1** | -53.3% | -57.2% | -57.8% | -61.8 … -44.2 | 45.0 | 1.69× | 171.4 | 1.52× | 11/11 | Pareto; Pareto x86 |
| **HEVC Slow** | -52.4% | -53.2% | -57.6% | -57.7 … -44.4 | 31.0 | 1.16× | 116.4 | 1.03× | 11/11 |  |
| **HEVC** | -51.1% | -52.3% | -57.1% | -56.7 … -44.2 | 50.4 | 1.89× | 201.2 | 1.78× | 11/11 | Pareto; Pareto x86 |
| **AV2 Fast** | -50.3% | -52.6% | -54.3% | -53.8 … -43.2 | 50.0 | 1.87× | 223.4 | 1.98× | 11/11 | Pareto x86 |
| **VP9 Slow** | -50.3% | -54.2% | -52.8% | -58.5 … -41.8 | 36.7 | 1.38× | 151.8 | 1.35× | 11/11 |  |
| **AV1 Fast** | -49.7% | -53.4% | -53.4% | -53.5 … -41.6 | 65.5 | 2.46× | 280.4 | 2.49× | 11/11 | Pareto; Pareto x86 |
| **VP9** | -49.4% | -53.5% | -52.1% | -55.5 … -40.9 | 43.2 | 1.62× | 172.4 | 1.53× | 11/11 |  |
| **AV2 Faster** | -48.9% | -51.0% | -53.1% | -53.4 … -40.9 | 59.5 | 2.23× | 254.8 | 2.26× | 11/11 |  |
| **HEVC Fast** | -47.0% | -47.4% | -52.5% | -51.6 … -40.3 | 78.2 | 2.93× | 304.6 | 2.70× | 11/11 | Pareto; Pareto x86 |
| **VVC Fast** | -46.9% | -46.0% | -53.6% | -52.4 … -39.2 | 55.1 | 2.07× | 221.2 | 1.96× | 11/11 |  |
| **AV1 Faster** | -46.6% | -47.7% | -53.8% | -52.6 … -38.9 | 78.4 | 2.94× | 312.7 | 2.77× | 11/11 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -45.5% | -49.4% | -44.5% | -48.3 … -40.1 | 14.3 | 0.53× | 80.8 | 0.72× | 11/11 | 2 rate-invalid pts |
| **VP9 Fast** | -45.0% | -48.5% | -47.2% | -49.4 … -36.7 | 54.4 | 2.04× | 227.3 | 2.02× | 11/11 |  |
| x265 medium, metric-tuned (--tune psnr) | -44.1% | -45.5% | -48.2% | -48.1 … -36.0 | 8.0 | 0.30× | 38.0 | 0.34× | 11/11 | fps of the default-tuned run; re-timing pending |
| **AV1 Veryfast** | -41.4% | -41.2% | -48.0% | -50.3 … -32.5 | 95.9 | 3.59× | 425.3 | 3.77× | 11/11 | Pareto; Pareto x86 |
| SVT-AV1 M10 | -41.3% | -43.9% | -43.0% | -42.9 … -34.1 | 39.3 | 1.47× | 228.9 | 2.03× | 11/11 | 2 rate-invalid pts |
| x265 medium | -40.7% | -40.2% | -43.2% | -43.3 … -33.2 | 8.0 | 0.30× | 38.0 | 0.34× | 11/11 |  |
| x265 superfast, metric-tuned (--tune psnr) | -40.1% | -40.6% | -44.4% | -46.8 … -30.5 | 12.1 | 0.45× | 57.9 | 0.51× | 11/11 | fps of the default-tuned run; re-timing pending |
| x265 superfast | -38.8% | -38.1% | -43.8% | -45.0 … -29.5 | 12.1 | 0.45× | 57.9 | 0.51× | 11/11 |  |
| **HEVC Faster** | -38.6% | -36.8% | -44.2% | -46.1 … -31.3 | 101.4 | 3.80× | 417.8 | 3.71× | 11/11 | Pareto |
| **VP9 Faster** | -37.8% | -37.2% | -44.4% | -45.2 … -28.2 | 92.0 | 3.45× | 354.1 | 3.14× | 11/11 |  |
| x265 ultrafast | -35.6% | -33.8% | -41.2% | -42.6 … -25.0 | 17.1 | 0.64× | 85.1 | 0.75× | 11/11 |  |
| SVT-AV1 M12 | -34.9% | -35.7% | -39.8% | -42.5 … -24.7 | 70.9 | 2.67× | 408.7 | 3.62× | 11/11 | 2 rate-invalid pts |
| **VP9 Veryfast** | -33.5% | -33.8% | -42.2% | -44.8 … -22.9 | 94.3 | 3.53× | 390.7 | 3.47× | 11/11 |  |
| SVT-AV1 M13 | -33.2% | -33.7% | -38.8% | -41.8 … -23.2 | 79.2 | 2.96× | 445.4 | 3.95× | 11/11 | Pareto x86; 1 rate-invalid pts |
| libaom rt 10 | -29.8% | -29.9% | -32.3% | -31.3 … -23.1 | 22.3 | 0.83× | 90.5 | 0.80× | 11/11 | 8 rate-invalid pts |
| libaom rt 8 | -28.7% | -26.8% | -32.1% | -31.6 … -23.5 | 17.4 | 0.65× | 68.8 | 0.61× | 11/11 | 6 rate-invalid pts |
| **H.264** | -28.6% | -30.6% | -36.5% | -52.5 … -16.7 | 65.9 | 2.47× | 205.2 | 1.82× | 11/11 |  |
| **H.264 Slow** | -28.2% | -30.8% | -35.7% | -51.5 … -17.3 | 42.0 | 1.57× | 123.7 | 1.10× | 11/11 |  |
| **H.264 Fast** | -16.7% | -18.3% | -27.1% | -42.5 … -3.6 | 121.2 | 4.54× | 379.5 | 3.37× | 11/11 | Pareto |
| x264 medium, metric-tuned (--tune psnr) | -9.8% | -8.3% | -19.4% | -10.3 … -6.7 | 26.7 | 1.00× | 112.8 | 1.00× | 11/11 | 5 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Faster** | -9.5% | -12.1% | -19.0% | -40.2 … +6.4 | 145.7 | 5.46× | 486.1 | 4.31× | 11/11 | Pareto; Pareto x86 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -6.5% | -1.4% | -12.1% | -10.8 … -0.6 | 47.8 | 1.78× | 204.4 | 1.81× | 11/11 | 17 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -5.7% | -1.2% | -11.9% | -10.3 … -0.8 | 47.8 | 1.78× | 204.4 | 1.81× | 11/11 | 16 rate-invalid pts |
| **H.264 Veryfast** | -5.4% | -7.3% | -15.3% | -37.4 … +10.6 | 166.2 | 6.23× | 526.2 | 4.67× | 11/11 | Pareto; Pareto x86 |
| libvpx VP9 rt 7 | -1.5% | +1.6% | -7.7% | -9.0 … +8.0 | 30.0 | 1.12× | 122.7 | 1.09× | 11/11 | 15 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 11/11 | 4 rate-invalid pts |
| x264 veryfast | +2.3% | +1.8% | -2.2% | -19.2 … +14.7 | 85.4 | 3.18× | 363.3 | 3.22× | 11/11 |  |
| openh264 | +24.6% | +22.2% | +21.8% | -27.9 … +62.5 | 111.5 | 4.17× | 524.7 | 4.65× | 11/11 | 22 rate-invalid pts |
| x264 ultrafast | +38.9% | +53.4% | +8.5% | -1.5 … +63.1 | 190.5 | 7.11× | 1017.0 | 9.02× | 11/11 | Pareto; Pareto x86 |

Low rungs (40 / 75 kbps, every encoder) on: static_otw_homes390, static_cam29, static_london_bus, bodycam_airport. Clips left out for every encoder: 0; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 2, crowd 2, lowlight 2, pan 2, static 3 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -28.2% | -30.6% |
| H.264 | -28.6% | -31.0% |
| H.264 Fast | -16.7% | -19.4% |
| H.264 Faster | -9.5% | -12.7% |
| H.264 Veryfast | -5.4% | -8.6% |
| HEVC Slow | -52.4% | -53.1% |
| HEVC | -51.1% | -51.8% |
| HEVC Fast | -47.0% | -47.6% |
| HEVC Faster | -38.6% | -39.7% |
| AV1 Slow | -53.5% | -54.6% |
| AV1 | -53.3% | -54.4% |
| AV1 Fast | -49.7% | -50.5% |
| AV1 Faster | -46.6% | -47.6% |
| AV1 Veryfast | -41.4% | -42.8% |
| VP9 Slow | -50.3% | -51.4% |
| VP9 | -49.4% | -50.4% |
| VP9 Fast | -45.0% | -45.9% |
| VP9 Faster | -37.8% | -39.1% |
| VP9 Veryfast | -33.5% | -35.1% |
| AV2 Slow | -56.3% | -57.2% |
| AV2 | -55.0% | -55.8% |
| AV2 Fast | -50.3% | -51.1% |
| AV2 Faster | -48.9% | -49.7% |
| VVC Slow | -56.1% | -56.8% |
| VVC | -56.0% | -56.8% |
| VVC Fast | -46.9% | -47.5% |
| openh264 | +24.6% | +18.0% |
| x264 ultrafast | +38.9% | +35.2% |
| x264 veryfast | +2.3% | +0.3% |
| x265 ultrafast | -35.6% | -35.6% |
| x265 superfast | -38.8% | -38.8% |
| x265 medium | -40.7% | -40.3% |
| SVT-AV1 M8 | -45.5% | -45.8% |
| SVT-AV1 M10 | -41.3% | -41.7% |
| SVT-AV1 M12 | -34.9% | -35.8% |
| SVT-AV1 M13 | -33.2% | -34.4% |
| libvpx VP9 rt 7 | -1.5% | +0.2% |
| libvpx VP9 rt 9 | -5.7% | -5.9% |
| libaom rt 8 | -28.7% | -28.8% |
| libaom rt 10 | -29.8% | -30.4% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +24.6% | +22.2% | +21.8% | +29.9% | 11/11 |  |
| openh264, adaptive quantisation on | +24.6% | +22.2% | +21.8% | +29.9% | 11/11 | +0.0 [+0.0, +0.0] not distinguishable |
| x264 ultrafast | +38.9% | +53.4% | +8.5% | +54.8% | 11/11 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +38.9% | +53.5% | +8.5% | +54.8% | 11/11 | +0.0 [-0.1, +0.2] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +32.1% | +44.5% | +18.4% | +33.5% | 11/11 | -1.8 [-5.5, +1.8] not distinguishable |
| x264 veryfast | +2.3% | +1.8% | -2.2% | +7.3% | 11/11 |  |
| x264 veryfast, metric-tuned (--tune psnr) | +1.5% | +3.1% | -15.4% | +16.6% | 11/11 | -2.3 [-5.4, +0.9] not distinguishable |
| x264 veryfast, metric-tuned (--tune ssim) | +1.1% | +1.5% | -3.6% | +5.4% | 11/11 | -1.8 [-3.8, +0.3] not distinguishable |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 11/11 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -9.8% | -8.3% | -19.4% | -1.8% | 11/11 | -9.8 [-12.0, -7.6] better |
| x264 medium, metric-tuned (--tune ssim) | -1.4% | -1.3% | -0.2% | -2.8% | 11/11 | -1.4 [-2.8, -0.0] better |
| x265 ultrafast | -35.6% | -33.8% | -41.2% | -31.7% | 11/11 |  |
| x265 ultrafast, metric-tuned (--tune psnr) | -36.2% | -35.9% | -40.8% | -31.8% | 11/11 | -1.1 [-2.2, +0.1] not distinguishable |
| x265 ultrafast, metric-tuned (--tune ssim) | -36.2% | -35.9% | -40.8% | -31.8% | 11/11 | -1.1 [-2.2, +0.1] not distinguishable |
| x265 superfast | -38.8% | -38.1% | -43.8% | -34.6% | 11/11 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -40.1% | -40.6% | -44.4% | -35.3% | 11/11 | -2.2 [-3.0, -1.4] better |
| x265 superfast, metric-tuned (--tune ssim) | -40.1% | -40.6% | -44.4% | -35.3% | 11/11 | -2.2 [-3.0, -1.4] better |
| x265 medium | -40.7% | -40.2% | -43.2% | -38.7% | 11/11 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -44.1% | -45.5% | -48.2% | -38.4% | 11/11 | -5.9 [-7.4, -4.4] better |
| x265 medium, metric-tuned (--tune ssim) | -41.8% | -42.4% | -43.7% | -39.3% | 11/11 | -1.9 [-2.8, -1.0] better |
| libvpx VP9 rt 7 | -1.5% | +1.6% | -7.7% | +1.5% | 11/11 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | -1.2% | +3.0% | -2.1% | -4.4% | 11/11 | -0.3 [-1.8, +1.2] not distinguishable |
| libvpx VP9 rt 9 | -5.7% | -1.2% | -11.9% | -4.0% | 11/11 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | -6.5% | -1.4% | -12.1% | -6.1% | 11/11 | -1.0 [-1.7, -0.3] better |
| libaom rt 8 | -28.7% | -26.8% | -32.1% | -27.2% | 11/11 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -29.3% | -26.1% | -28.7% | -33.1% | 11/11 | +0.6 [-1.5, +2.8] not distinguishable |
| libaom rt 10 | -29.8% | -29.9% | -32.3% | -27.2% | 11/11 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -30.1% | -28.9% | -29.1% | -32.4% | 11/11 | +1.2 [-0.6, +3.1] not distinguishable |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **AV2 Slow** | -49.9% | -51.9% | -50.0% | -47.7% | 11/11 |
| **VVC Slow** | -49.4% | -48.5% | -49.5% | -50.2% | 11/11 |
| **VVC** | -49.3% | -48.3% | -49.5% | -50.1% | 11/11 |
| **AV2** | -48.3% | -50.5% | -48.7% | -45.7% | 11/11 |
| **AV1 Slow** | -46.5% | -49.0% | -45.6% | -45.0% | 11/11 |
| **AV1** | -46.3% | -48.7% | -44.9% | -45.2% | 11/11 |
| **HEVC Slow** | -45.2% | -44.4% | -44.6% | -46.7% | 11/11 |
| **HEVC** | -43.8% | -43.3% | -43.7% | -44.3% | 11/11 |
| **AV2 Fast** | -42.7% | -43.3% | -40.5% | -44.1% | 11/11 |
| **VP9 Slow** | -42.6% | -45.6% | -38.3% | -44.0% | 11/11 |
| **AV1 Fast** | -42.0% | -44.3% | -39.3% | -42.4% | 11/11 |
| **VP9** | -41.6% | -44.8% | -37.4% | -42.7% | 11/11 |
| **AV2 Faster** | -41.2% | -41.7% | -39.3% | -42.7% | 11/11 |
| **HEVC Fast** | -38.9% | -37.5% | -38.0% | -41.2% | 11/11 |
| **VVC Fast** | -38.9% | -36.2% | -39.5% | -41.0% | 11/11 |
| **AV1 Faster** | -38.7% | -38.2% | -39.9% | -38.1% | 11/11 |
| SVT-AV1 M8 | -37.0% | -40.1% | -28.0% | -43.0% | 11/11 |
| **VP9 Fast** | -36.5% | -39.1% | -31.5% | -38.9% | 11/11 |
| x265 medium, metric-tuned (--tune psnr) | -35.8% | -36.4% | -32.7% | -38.3% | 11/11 |
| x265 medium, metric-tuned (--tune ssim) | -32.9% | -32.8% | -26.6% | -39.2% | 11/11 |
| **AV1 Veryfast** | -32.8% | -31.6% | -32.4% | -34.6% | 11/11 |
| SVT-AV1 M10 | -32.1% | -33.6% | -25.9% | -36.9% | 11/11 |
| x265 medium | -31.6% | -30.1% | -26.0% | -38.6% | 11/11 |
| x265 superfast, metric-tuned (--tune psnr) | -30.8% | -30.3% | -27.1% | -34.9% | 11/11 |
| x265 superfast, metric-tuned (--tune ssim) | -30.8% | -30.3% | -27.1% | -34.9% | 11/11 |
| **HEVC Faster** | -29.6% | -26.9% | -27.8% | -34.2% | 11/11 |
| x265 superfast | -29.2% | -27.3% | -26.2% | -34.2% | 11/11 |
| **VP9 Faster** | -29.0% | -27.5% | -28.2% | -31.2% | 11/11 |
| x265 ultrafast, metric-tuned (--tune psnr) | -26.3% | -25.4% | -22.4% | -31.1% | 11/11 |
| x265 ultrafast, metric-tuned (--tune ssim) | -26.3% | -25.4% | -22.4% | -31.1% | 11/11 |
| x265 ultrafast | -25.6% | -23.0% | -22.7% | -31.0% | 11/11 |
| SVT-AV1 M12 | -25.1% | -24.9% | -21.8% | -28.5% | 11/11 |
| **VP9 Veryfast** | -24.4% | -24.0% | -25.6% | -23.6% | 11/11 |
| SVT-AV1 M13 | -23.3% | -23.1% | -20.7% | -26.2% | 11/11 |
| libaom rt 10, metric-tuned (--tune ssim) | -19.5% | -18.4% | -8.1% | -32.0% | 11/11 |
| **H.264** | -19.5% | -21.1% | -20.0% | -17.4% | 11/11 |
| libaom rt 10 | -19.1% | -19.3% | -11.3% | -26.9% | 11/11 |
| **H.264 Slow** | -19.1% | -21.3% | -18.9% | -17.0% | 11/11 |
| libaom rt 8, metric-tuned (--tune ssim) | -18.4% | -15.7% | -7.0% | -32.6% | 11/11 |
| libaom rt 8 | -18.0% | -16.5% | -10.7% | -26.8% | 11/11 |
| **H.264 Fast** | -6.6% | -8.0% | -9.0% | -2.9% | 11/11 |
| **H.264 Faster** | +1.3% | -1.9% | +0.6% | +5.0% | 11/11 |
| **H.264 Veryfast** | +5.5% | +3.1% | +4.8% | +8.8% | 11/11 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +7.6% | +12.3% | +15.9% | -5.3% | 11/11 |
| libvpx VP9 rt 9 | +8.5% | +12.7% | +16.1% | -3.2% | 11/11 |
| x264 medium, metric-tuned (--tune ssim) | +11.1% | +8.7% | +25.3% | -0.7% | 11/11 |
| x264 veryfast, metric-tuned (--tune psnr) | +12.5% | +12.5% | +5.8% | +19.1% | 11/11 |
| libvpx VP9 rt 7 | +12.7% | +14.9% | +20.5% | +2.7% | 11/11 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +12.7% | +15.4% | +25.6% | -3.0% | 11/11 |
| x264 veryfast, metric-tuned (--tune ssim) | +12.9% | +10.6% | +20.7% | +7.4% | 11/11 |
| x264 medium | +13.0% | +9.8% | +27.1% | +2.0% | 11/11 |
| x264 veryfast | +14.6% | +11.1% | +23.4% | +9.4% | 11/11 |
| openh264 | +42.8% | +39.8% | +56.7% | +31.9% | 11/11 |
| openh264, adaptive quantisation on | +42.8% | +39.8% | +56.7% | +31.9% | 11/11 |
| x264 ultrafast, metric-tuned (--tune ssim) | +45.8% | +53.0% | +47.4% | +37.0% | 11/11 |
| x264 ultrafast, metric-tuned (--tune psnr) | +52.3% | +62.2% | +35.5% | +59.3% | 11/11 |
| x264 ultrafast | +52.4% | +62.1% | +35.7% | +59.3% | 11/11 |

## Hardware encoders

NVIDIA NVENC (NVIDIA GeForce RTX 5090 NVENC, driver 610.57.04, ffmpeg 7.1.1) through ffmpeg, in low-delay CBR on the same clips, rungs and buffer as every point above, no B-frames; scored the same way; p4 is preset P4 with `-tune ll`, p7 preset P7 with `-tune hq`. Speeds are GPU wall-clock frames per second at 720p (fixed-function hardware, not CPU time; tools/dev/nvenc_timing.py): the aggregate throughput of N parallel streams (the best of 1, 2, 4 and 8) and one stream, both encoding frames already in GPU memory, and one stream through ffmpeg's read-convert-upload pipeline (the earlier figure, bound by that pipeline). The x86 chart plots the aggregate, never on the Pareto front; the Jetson chart leaves them out.

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | aggregate fps (N streams) | 1-stream encode fps | 1-stream fps via ffmpeg pipeline | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---|
| NVENC HEVC p4 | -43.0% | -43.7% | -47.1% | -45.0 … -37.1 | 2815 (8) | 798 | 328 | 11/11 | 3 rate-invalid pts |
| NVENC HEVC p7 | -42.9% | -43.9% | -46.9% | -45.8 … -36.5 | 1160 (8) | 336 | 224 | 11/11 | 3 rate-invalid pts |
| NVENC AV1 p7 | -39.6%‡ | -43.1% | -39.0% | -45.6 … -32.3 | 2291 (8) | 679 | 202 | 11/11 | ‡ 1 clip chained; 8 rate-invalid pts |
| NVENC AV1 p4 | -36.4%‡ | -40.0% | -35.9% | -43.2 … -25.3 | 3785 (8) | 1065 | 249 | 11/11 | ‡ 1 clip chained; 9 rate-invalid pts |
| NVENC H.264 p7 | -19.1% | -19.0% | -26.9% | -26.2 … -6.5 | 2038 (8) | 548 | 319 | 11/11 | 8 rate-invalid pts |
| NVENC H.264 p4 | -18.7% | -18.5% | -26.7% | -26.7 … -5.6 | 3358 (8) | 949 | 352 | 11/11 | 8 rate-invalid pts |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slow | openh264 | -38.9% | -39.9% | -44.6% | -43.8 … -30.3 | 11/11 |
| H.264 | openh264 | -39.2% | -39.5% | -45.3% | -45.6 … -30.2 | 11/11 |
| H.264 | x264 veryfast | -35.1% | -36.2% | -40.9% | -47.9 … -24.3 | 11/11 |
| H.264 | x264 medium | -28.6% | -30.6% | -36.5% | -52.5 … -16.7 | 11/11 |
| H.264 Fast | openh264 | -26.0% | -26.3% | -36.6% | -32.1 … -15.3 | 11/11 |
| H.264 Fast | x264 veryfast | -23.2% | -23.9% | -30.6% | -34.8 … -12.8 | 11/11 |
| H.264 Fast | H.264 | +20.9% | +21.6% | +17.2% | +15.5 … +26.5 | 11/11 |
| H.264 Faster | openh264 | -19.1% | -20.5% | -29.5% | -25.8 … -11.8 | 11/11 |
| H.264 Faster | x264 ultrafast | -39.7% | -47.8% | -31.0% | -46.9 … -33.7 | 11/11 |
| H.264 Faster | H.264 Fast | +10.0% | +8.5% | +10.8% | +4.2 … +14.5 | 11/11 |
| H.264 Veryfast | openh264 | -13.7% | -14.7% | -25.2% | -21.5 … -0.6 | 11/11 |
| H.264 Veryfast | x264 ultrafast | -36.6% | -44.5% | -27.9% | -44.1 … -31.4 | 11/11 |
| H.264 Veryfast | H.264 Faster | +5.9% | +7.3% | +4.3% | +3.4 … +5.9 | 11/11 |
| HEVC Slow | SVT-AV1 M12 | -29.5% | -29.0% | -32.5% | -37.8 … -21.1 | 11/11 |
| HEVC | SVT-AV1 M12 | -27.6% | -27.5% | -32.0% | -35.5 … -20.2 | 11/11 |
| HEVC | x265 medium | -19.9% | -21.9% | -28.0% | -35.7 … -6.0 | 11/11 |
| HEVC Fast | SVT-AV1 M12 | -18.8% | -17.3% | -22.1% | -28.8 … -7.9 | 11/11 |
| HEVC Fast | x265 superfast | -15.6% | -16.5% | -18.4% | -25.8 … -7.8 | 11/11 |
| HEVC Fast | HEVC | +11.8% | +13.7% | +14.0% | +7.3 … +14.7 | 11/11 |
| HEVC Faster | SVT-AV1 M12 | -2.5% | +2.7% | -6.1% | -10.1 … +3.8 | 11/11 |
| HEVC Faster | SVT-AV1 M13 | -7.0% | -1.9% | -9.8% | -15.7 … -0.2 | 11/11 |
| HEVC Faster | HEVC Fast | +21.2% | +25.2% | +21.8% | +15.7 … +28.6 | 11/11 |
| AV1 Slow | SVT-AV1 M12 | -33.6% | -37.3% | -36.2% | -36.0 … -30.2 | 11/11 |
| AV1 | SVT-AV1 M12 | -33.3% | -36.9% | -35.2% | -35.7 … -29.8 | 11/11 |
| AV1 | openh264 | -59.7% | -61.4% | -61.9% | -68.1 … -48.6 | 11/11 |
| AV1 | SVT-AV1 M8 | -13.4% | -13.0% | -24.9% | -22.6 … -5.4 | 11/11 |
| AV1 | libaom rt 8 | -32.8% | -36.4% | -38.1% | -43.8 … -26.7 | 11/11 |
| AV1 Fast | SVT-AV1 M12 | -25.7% | -29.7% | -25.3% | -27.4 … -21.6 | 11/11 |
| AV1 Fast | openh264 | -55.7% | -57.3% | -57.2% | -65.9 … -41.8 | 11/11 |
| AV1 Fast | AV1 | +10.8% | +11.2% | +14.3% | +5.6 … +14.0 | 11/11 |
| AV1 Faster | SVT-AV1 M12 | -20.0% | -19.7% | -26.3% | -21.9 … -14.9 | 11/11 |
| AV1 Faster | AV1 Fast | +7.8% | +14.5% | -1.4% | +3.1 … +12.1 | 11/11 |
| AV1 Veryfast | SVT-AV1 M12 | -9.9% | -6.1% | -15.9% | -20.5 … -1.3 | 11/11 |
| AV1 Veryfast | SVT-AV1 M13 | -14.6% | -11.2% | -19.5% | -21.5 … -6.4 | 11/11 |
| AV1 Veryfast | AV1 Faster | +13.0% | +16.8% | +14.4% | +7.2 … +17.8 | 11/11 |
| VP9 Slow | HEVC Slow | +5.8% | -2.4% | +13.6% | -0.4 … +10.7 | 11/11 |
| VP9 | HEVC | +4.9% | -3.0% | +14.1% | +0.1 … +10.1 | 11/11 |
| VP9 | libvpx VP9 rt 7 | -45.9% | -50.5% | -44.6% | -60.4 … -39.3 | 11/11 |
| VP9 Fast | HEVC Fast | +4.6% | -3.6% | +13.1% | -1.7 … +11.8 | 11/11 |
| VP9 Fast | libvpx VP9 rt 9 | -41.1% | -47.3% | -39.3% | -48.9 … -35.1 | 11/11 |
| VP9 Fast | VP9 | +10.9% | +12.4% | +12.3% | +8.0 … +12.2 | 11/11 |
| VP9 Faster | HEVC Faster | +0.5% | -2.8% | -1.2% | -5.0 … +6.6 | 11/11 |
| VP9 Faster | libvpx VP9 rt 9 | -32.5% | -35.0% | -35.8% | -45.7 … -19.0 | 11/11 |
| VP9 Faster | VP9 Fast | +17.1% | +26.9% | +5.7% | +7.6 … +20.7 | 11/11 |
| VP9 Veryfast | HEVC Faster | +11.4% | +5.1% | +3.4% | +5.9 … +13.8 | 11/11 |
| VP9 Veryfast | libvpx VP9 rt 9 | -25.8% | -30.6% | -32.3% | -43.0 … -11.4 | 11/11 |
| VP9 Veryfast | VP9 Faster | +12.1% | +9.3% | +4.8% | +5.0 … +12.5 | 11/11 |
| AV2 Slow | AV1 Slow | -7.3% | -7.2% | -9.5% | -11.3 … -4.5 | 11/11 |
| AV2 | AV1 | -4.5% | -4.5% | -7.6% | -7.2 … -3.1 | 11/11 |
| AV2 | SVT-AV1 M8 | -17.7% | -17.3% | -31.2% | -27.1 … -12.0 | 11/11 |
| AV2 Fast | AV1 Fast | -2.4% | +0.9% | -3.4% | -5.7 … +3.0 | 11/11 |
| AV2 Fast | AV2 | +13.6% | +17.4% | +19.9% | +10.8 … +16.0 | 11/11 |
| AV2 Faster | AV1 Faster | -5.6% | -7.8% | +1.2% | -11.5 … +4.2 | 11/11 |
| AV2 Faster | SVT-AV1 M12 | -25.8% | -26.7% | -26.1% | -29.5 … -20.4 | 11/11 |
| AV2 Faster | AV2 Fast | +3.4% | +4.3% | +2.3% | +0.5 … +3.4 | 11/11 |
| VVC Slow | HEVC Slow | -9.7% | -9.3% | -10.3% | -11.9 … -6.1 | 11/11 |
| VVC | HEVC | -11.9% | -10.9% | -11.2% | -17.6 … -7.3 | 11/11 |
| VVC | x265 medium | -29.1% | -30.1% | -36.2% | -42.3 … -17.6 | 11/11 |
| VVC Fast | HEVC Fast | -0.5% | +2.2% | -3.1% | -1.5 … +5.4 | 11/11 |
| VVC Fast | VVC | +26.5% | +30.1% | +24.9% | +21.4 … +30.0 | 11/11 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 730e4e16 (H.264 Fast, H.264 Faster, H.264 Veryfast, VP9 Faster: 478e8249; VP9, AV1 Slow: 77e93089; AV1, AV1 Fast, VP9 Fast, AV1 Veryfast: b33ba3bb; VP9 Slow, AV2 Slow: bad14976; H.264, HEVC, HEVC Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, HEVC Faster, VP9 Veryfast, HEVC Slow, VVC Slow, H.264 Slow, AV2 Faster: c9cff89f); H.264 Fast, H.264 Faster, H.264 Veryfast, VP9 Faster, AV1 Slow, VP9, AV2 Slow, VP9 Slow, AV1, AV1 Fast, AV1 Veryfast, VP9 Fast timed on c9cff89f (same streams), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 medium --tune psnr (-9.8 [-12.0, -7.6]%); x265 superfast --tune psnr (-2.2 [-3.0, -1.4]%); x265 medium --tune psnr (-5.9 [-7.4, -4.4]%); libvpx VP9 rt 9 --tune ssim (-1.0 [-1.7, -0.3]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
