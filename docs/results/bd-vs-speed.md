# Bitrate vs encode speed

<sub>Updated 2026-11-20, build 730e4e16 (H.264 Fast, H.264 Faster, HEVC, HEVC Fast, AV1, AV1 Fast, VP9, VP9 Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, AV1 Veryfast, HEVC Faster, H.264 Veryfast, VP9 Faster, VP9 Veryfast, HEVC Slow, AV1 Slow, VVC Slow, H.264 Slow, VP9 Slow, AV2 Slow, AV2 Faster: 20d31869; H.264: c9cff89f); H.264 timed on 20d31869 (same streams)</sub>

![bitrate vs speed](bd-vs-speed.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **AV2 Slow** | -57.7% | -61.2% | -62.9% | -62.1 … -51.3 | 16.1 | 0.60× | 62.3 | 0.55× | 11/11 | Pareto; Pareto x86 |
| **VVC Slow** | -57.5% | -58.8% | -62.8% | -63.0 … -50.7 | 16.0 | 0.60× | 64.4 | 0.57× | 11/11 | Pareto x86 |
| **VVC** | -56.2% | -57.4% | -61.9% | -61.5 … -48.0 | 29.8 | 1.12× | 121.0 | 1.07× | 11/11 | Pareto; Pareto x86 |
| **AV1 Slow** | -55.3% | -58.9% | -58.4% | -63.1 … -47.5 | 26.6 | 1.00× | 113.3 | 1.00× | 11/11 |  |
| **AV2** | -55.3% | -59.0% | -60.9% | -63.2 … -47.3 | 39.2 | 1.47× | 170.1 | 1.51× | 11/11 | Pareto; Pareto x86 |
| **HEVC Slow** | -54.0% | -55.1% | -59.1% | -58.9 … -46.4 | 19.0 | 0.71× | 76.9 | 0.68× | 11/11 |  |
| **AV1** | -53.7% | -57.5% | -58.0% | -62.7 … -44.2 | 45.7 | 1.71× | 181.4 | 1.61× | 11/11 | Pareto; Pareto x86 |
| **HEVC** | -51.7% | -52.9% | -58.0% | -57.2 … -45.0 | 45.3 | 1.70× | 171.9 | 1.52× | 11/11 |  |
| **VVC Fast** | -51.5% | -52.6% | -58.7% | -59.5 … -42.5 | 53.2 | 1.99× | 205.2 | 1.82× | 11/11 | Pareto; Pareto x86 |
| **AV2 Fast** | -51.3% | -53.4% | -55.4% | -57.9 … -44.1 | 51.8 | 1.94× | 230.8 | 2.05× | 11/11 | Pareto x86 |
| **AV1 Fast** | -51.0% | -54.5% | -54.8% | -60.1 … -41.8 | 66.2 | 2.48× | 267.6 | 2.37× | 11/11 | Pareto; Pareto x86 |
| **VP9 Slow** | -50.4% | -54.3% | -52.8% | -57.8 … -42.0 | 26.7 | 1.00× | 100.0 | 0.89× | 11/11 |  |
| **VP9** | -49.7% | -53.7% | -52.5% | -57.1 … -41.1 | 42.1 | 1.58× | 171.0 | 1.52× | 11/11 |  |
| **AV2 Faster** | -49.7% | -51.7% | -53.7% | -58.6 … -40.8 | 62.9 | 2.36× | 274.8 | 2.44× | 11/11 | Pareto x86 |
| **HEVC Fast** | -48.8% | -49.2% | -54.1% | -54.0 … -41.2 | 72.0 | 2.70× | 293.2 | 2.60× | 11/11 | Pareto; Pareto x86 |
| **AV1 Faster** | -47.3% | -48.4% | -54.4% | -59.0 … -38.9 | 79.4 | 2.98× | 323.9 | 2.87× | 11/11 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -45.5% | -49.4% | -44.5% | -48.3 … -40.1 | 14.3 | 0.53× | 80.8 | 0.72× | 11/11 | 2 rate-invalid pts |
| **VP9 Fast** | -45.2% | -48.8% | -47.3% | -49.6 … -37.1 | 55.8 | 2.09× | 233.7 | 2.07× | 11/11 |  |
| x265 medium, metric-tuned (--tune psnr) | -44.1% | -45.5% | -48.2% | -48.1 … -36.0 | 8.0 | 0.30× | 38.0 | 0.34× | 11/11 | fps of the default-tuned run; re-timing pending |
| **HEVC Faster** | -43.3% | -43.4% | -49.2% | -50.5 … -34.5 | 90.3 | 3.38× | 364.2 | 3.23× | 11/11 | Pareto; Pareto x86 |
| **VP9 Faster** | -41.5% | -41.4% | -46.1% | -53.0 … -31.3 | 92.3 | 3.46× | 368.7 | 3.27× | 11/11 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -41.4% | -41.2% | -48.1% | -50.7 … -32.5 | 96.5 | 3.62× | 431.4 | 3.83× | 11/11 | Pareto; Pareto x86 |
| SVT-AV1 M10 | -41.3% | -43.9% | -43.0% | -42.9 … -34.1 | 39.3 | 1.47× | 228.9 | 2.03× | 11/11 | 2 rate-invalid pts |
| x265 medium | -40.7% | -40.2% | -43.2% | -43.3 … -33.2 | 8.0 | 0.30× | 38.0 | 0.34× | 11/11 |  |
| x265 superfast, metric-tuned (--tune psnr) | -40.1% | -40.6% | -44.4% | -46.8 … -30.5 | 12.1 | 0.45× | 57.9 | 0.51× | 11/11 | fps of the default-tuned run; re-timing pending |
| x265 superfast | -38.8% | -38.1% | -43.8% | -45.0 … -29.5 | 12.1 | 0.45× | 57.9 | 0.51× | 11/11 |  |
| **VP9 Veryfast** | -37.2% | -38.2% | -43.0% | -45.4 … -28.5 | 95.8 | 3.59× | 385.9 | 3.42× | 11/11 |  |
| x265 ultrafast | -35.6% | -33.8% | -41.2% | -42.6 … -25.0 | 17.1 | 0.64× | 85.1 | 0.75× | 11/11 |  |
| SVT-AV1 M12 | -34.9% | -35.7% | -39.8% | -42.5 … -24.7 | 70.9 | 2.67× | 408.7 | 3.62× | 11/11 | 2 rate-invalid pts |
| SVT-AV1 M13 | -33.2% | -33.7% | -38.8% | -41.8 … -23.2 | 79.2 | 2.96× | 445.4 | 3.95× | 11/11 | Pareto x86; 1 rate-invalid pts |
| **H.264 Slow** | -30.4% | -33.1% | -37.4% | -53.1 … -19.4 | 37.2 | 1.39× | 115.3 | 1.02× | 11/11 |  |
| libaom rt 10 | -29.8% | -29.9% | -32.3% | -31.3 … -23.1 | 22.3 | 0.83× | 90.5 | 0.80× | 11/11 | 8 rate-invalid pts |
| libaom rt 8 | -28.7% | -26.8% | -32.1% | -31.6 … -23.5 | 17.4 | 0.65× | 68.8 | 0.61× | 11/11 | 6 rate-invalid pts |
| **H.264** | -28.6% | -30.6% | -36.5% | -52.5 … -16.7 | 66.4 | 2.49× | 202.2 | 1.79× | 11/11 |  |
| **H.264 Fast** | -21.5% | -23.0% | -30.1% | -47.2 … -9.5 | 119.5 | 4.48× | 370.1 | 3.28× | 11/11 | Pareto |
| **H.264 Faster** | -13.2% | -15.5% | -21.9% | -44.8 … +3.8 | 146.9 | 5.51× | 460.1 | 4.08× | 11/11 | Pareto; Pareto x86 |
| **H.264 Veryfast** | -11.3% | -13.7% | -20.0% | -43.4 … +6.0 | 164.1 | 6.15× | 520.2 | 4.61× | 11/11 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -9.8% | -8.3% | -19.4% | -10.3 … -6.7 | 26.7 | 1.00× | 112.8 | 1.00× | 11/11 | 5 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -6.5% | -1.4% | -12.1% | -10.8 … -0.6 | 47.8 | 1.78× | 204.4 | 1.81× | 11/11 | 17 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -5.7% | -1.2% | -11.9% | -10.3 … -0.8 | 47.8 | 1.78× | 204.4 | 1.81× | 11/11 | 16 rate-invalid pts |
| libvpx VP9 rt 7 | -1.5% | +1.6% | -7.7% | -9.0 … +8.0 | 30.0 | 1.12× | 122.7 | 1.09× | 11/11 | 15 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 11/11 | 4 rate-invalid pts |
| x264 veryfast | +2.3% | +1.8% | -2.2% | -19.2 … +14.7 | 85.4 | 3.18× | 363.3 | 3.22× | 11/11 |  |
| openh264 | +24.6% | +22.2% | +21.8% | -27.9 … +62.5 | 111.5 | 4.17× | 524.7 | 4.65× | 11/11 | Pareto x86; 22 rate-invalid pts |
| x264 ultrafast | +38.9% | +53.4% | +8.5% | -1.5 … +63.1 | 190.5 | 7.11× | 1017.0 | 9.02× | 11/11 | Pareto; Pareto x86 |

Low rungs (40 / 75 kbps, every encoder) on: static_otw_homes390, static_cam29, static_london_bus, bodycam_airport. Clips left out for every encoder: 0; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 2, crowd 2, lowlight 2, pan 2, static 3 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -30.4% | -32.7% |
| H.264 | -28.6% | -31.0% |
| H.264 Fast | -21.5% | -24.4% |
| H.264 Faster | -13.2% | -16.6% |
| H.264 Veryfast | -11.3% | -14.8% |
| HEVC Slow | -54.0% | -54.6% |
| HEVC | -51.7% | -52.3% |
| HEVC Fast | -48.8% | -49.6% |
| HEVC Faster | -43.3% | -44.4% |
| AV1 Slow | -55.3% | -56.3% |
| AV1 | -53.7% | -54.8% |
| AV1 Fast | -51.0% | -52.2% |
| AV1 Faster | -47.3% | -48.5% |
| AV1 Veryfast | -41.4% | -42.7% |
| VP9 Slow | -50.4% | -51.4% |
| VP9 | -49.7% | -50.8% |
| VP9 Fast | -45.2% | -46.1% |
| VP9 Faster | -41.5% | -43.0% |
| VP9 Veryfast | -37.2% | -38.5% |
| AV2 Slow | -57.7% | -58.4% |
| AV2 | -55.3% | -56.2% |
| AV2 Fast | -51.3% | -52.3% |
| AV2 Faster | -49.7% | -50.8% |
| VVC Slow | -57.5% | -58.2% |
| VVC | -56.2% | -57.0% |
| VVC Fast | -51.5% | -52.5% |
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
| **AV2 Slow** | -51.5% | -53.5% | -51.7% | -49.4% | 11/11 |
| **VVC Slow** | -51.0% | -50.6% | -51.1% | -51.3% | 11/11 |
| **VVC** | -49.6% | -48.9% | -49.9% | -49.8% | 11/11 |
| **AV2** | -48.6% | -50.8% | -49.0% | -46.0% | 11/11 |
| **AV1 Slow** | -48.5% | -50.7% | -46.0% | -49.0% | 11/11 |
| **HEVC Slow** | -47.1% | -46.6% | -46.5% | -48.2% | 11/11 |
| **AV1** | -46.6% | -49.1% | -45.3% | -45.6% | 11/11 |
| **HEVC** | -44.5% | -43.8% | -44.9% | -44.6% | 11/11 |
| **VVC Fast** | -44.1% | -43.0% | -45.8% | -43.4% | 11/11 |
| **AV2 Fast** | -43.8% | -44.4% | -41.8% | -45.3% | 11/11 |
| **AV1 Fast** | -43.4% | -45.6% | -40.9% | -43.8% | 11/11 |
| **VP9 Slow** | -42.7% | -45.8% | -38.2% | -44.2% | 11/11 |
| **AV2 Faster** | -42.0% | -42.6% | -39.8% | -43.7% | 11/11 |
| **VP9** | -42.0% | -45.0% | -37.9% | -43.1% | 11/11 |
| **HEVC Fast** | -40.9% | -39.6% | -40.0% | -43.2% | 11/11 |
| **AV1 Faster** | -39.5% | -38.8% | -40.6% | -38.9% | 11/11 |
| SVT-AV1 M8 | -37.0% | -40.1% | -28.0% | -43.0% | 11/11 |
| **VP9 Fast** | -36.7% | -39.4% | -31.5% | -39.3% | 11/11 |
| x265 medium, metric-tuned (--tune psnr) | -35.8% | -36.4% | -32.7% | -38.3% | 11/11 |
| **HEVC Faster** | -34.9% | -33.7% | -33.9% | -37.0% | 11/11 |
| **VP9 Faster** | -32.9% | -32.1% | -30.2% | -36.4% | 11/11 |
| x265 medium, metric-tuned (--tune ssim) | -32.9% | -32.8% | -26.6% | -39.2% | 11/11 |
| **AV1 Veryfast** | -32.9% | -31.5% | -32.6% | -34.5% | 11/11 |
| SVT-AV1 M10 | -32.1% | -33.6% | -25.9% | -36.9% | 11/11 |
| x265 medium | -31.6% | -30.1% | -26.0% | -38.6% | 11/11 |
| x265 superfast, metric-tuned (--tune psnr) | -30.8% | -30.3% | -27.1% | -34.9% | 11/11 |
| x265 superfast, metric-tuned (--tune ssim) | -30.8% | -30.3% | -27.1% | -34.9% | 11/11 |
| x265 superfast | -29.2% | -27.3% | -26.2% | -34.2% | 11/11 |
| **VP9 Veryfast** | -28.2% | -28.2% | -26.5% | -29.8% | 11/11 |
| x265 ultrafast, metric-tuned (--tune psnr) | -26.3% | -25.4% | -22.4% | -31.1% | 11/11 |
| x265 ultrafast, metric-tuned (--tune ssim) | -26.3% | -25.4% | -22.4% | -31.1% | 11/11 |
| x265 ultrafast | -25.6% | -23.0% | -22.7% | -31.0% | 11/11 |
| SVT-AV1 M12 | -25.1% | -24.9% | -21.8% | -28.5% | 11/11 |
| SVT-AV1 M13 | -23.3% | -23.1% | -20.7% | -26.2% | 11/11 |
| **H.264 Slow** | -21.5% | -23.8% | -21.1% | -19.7% | 11/11 |
| libaom rt 10, metric-tuned (--tune ssim) | -19.5% | -18.4% | -8.1% | -32.0% | 11/11 |
| **H.264** | -19.5% | -21.1% | -20.0% | -17.4% | 11/11 |
| libaom rt 10 | -19.1% | -19.3% | -11.3% | -26.9% | 11/11 |
| libaom rt 8, metric-tuned (--tune ssim) | -18.4% | -15.7% | -7.0% | -32.6% | 11/11 |
| libaom rt 8 | -18.0% | -16.5% | -10.7% | -26.8% | 11/11 |
| **H.264 Fast** | -11.7% | -12.9% | -12.3% | -9.9% | 11/11 |
| **H.264 Faster** | -2.6% | -5.3% | -2.5% | -0.1% | 11/11 |
| **H.264 Veryfast** | -0.7% | -3.5% | -0.6% | +2.0% | 11/11 |
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
| H.264 Slow | openh264 | -40.7% | -41.7% | -45.9% | -47.0 … -31.8 | 11/11 |
| H.264 | openh264 | -39.2% | -39.5% | -45.3% | -45.6 … -30.2 | 11/11 |
| H.264 | x264 veryfast | -35.1% | -36.2% | -40.9% | -47.9 … -24.3 | 11/11 |
| H.264 | x264 medium | -28.6% | -30.6% | -36.5% | -52.5 … -16.7 | 11/11 |
| H.264 Fast | openh264 | -31.9% | -31.7% | -39.9% | -35.6 … -21.7 | 11/11 |
| H.264 Fast | x264 veryfast | -28.3% | -28.9% | -34.2% | -39.5 … -17.8 | 11/11 |
| H.264 Fast | H.264 | +12.2% | +13.0% | +11.2% | +8.3 … +13.7 | 11/11 |
| H.264 Faster | openh264 | -23.8% | -24.7% | -32.7% | -28.8 … -16.5 | 11/11 |
| H.264 Faster | x264 ultrafast | -43.1% | -50.7% | -34.1% | -51.5 … -36.4 | 11/11 |
| H.264 Faster | H.264 Fast | +12.1% | +10.9% | +11.6% | +4.3 … +18.4 | 11/11 |
| H.264 Veryfast | openh264 | -20.2% | -21.6% | -29.7% | -27.7 … -7.2 | 11/11 |
| H.264 Veryfast | x264 ultrafast | -41.8% | -49.7% | -32.8% | -50.5 … -35.3 | 11/11 |
| H.264 Veryfast | H.264 Faster | +2.8% | +2.7% | +1.8% | +0.9 … +3.5 | 11/11 |
| HEVC Slow | SVT-AV1 M12 | -32.7% | -32.5% | -35.7% | -40.5 … -23.5 | 11/11 |
| HEVC | SVT-AV1 M12 | -28.9% | -28.7% | -34.0% | -36.1 … -19.5 | 11/11 |
| HEVC | x265 medium | -21.3% | -23.1% | -30.2% | -36.8 … -8.1 | 11/11 |
| HEVC Fast | SVT-AV1 M12 | -22.7% | -21.0% | -25.9% | -29.8 … -16.4 | 11/11 |
| HEVC Fast | x265 superfast | -18.8% | -19.7% | -21.5% | -29.1 … -8.5 | 11/11 |
| HEVC Fast | HEVC | +9.1% | +10.7% | +12.4% | +4.9 … +12.7 | 11/11 |
| HEVC Faster | SVT-AV1 M12 | -12.9% | -11.0% | -17.3% | -16.8 … -8.8 | 11/11 |
| HEVC Faster | SVT-AV1 M13 | -17.4% | -15.4% | -21.0% | -22.5 … -10.9 | 11/11 |
| HEVC Faster | HEVC Fast | +13.3% | +13.5% | +12.4% | +7.2 … +18.5 | 11/11 |
| AV1 Slow | SVT-AV1 M12 | -36.5% | -39.9% | -35.8% | -40.5 … -32.6 | 11/11 |
| AV1 | SVT-AV1 M12 | -33.7% | -37.3% | -35.5% | -37.3 … -29.7 | 11/11 |
| AV1 | openh264 | -60.1% | -61.8% | -62.2% | -68.0 … -48.8 | 11/11 |
| AV1 | SVT-AV1 M8 | -14.0% | -13.4% | -25.4% | -22.5 … -5.4 | 11/11 |
| AV1 | libaom rt 8 | -33.3% | -36.9% | -38.7% | -46.0 … -27.1 | 11/11 |
| AV1 Fast | SVT-AV1 M12 | -28.6% | -32.0% | -28.7% | -32.1 … -24.9 | 11/11 |
| AV1 Fast | openh264 | -57.5% | -58.8% | -59.0% | -66.0 … -44.6 | 11/11 |
| AV1 Fast | AV1 | +7.9% | +8.7% | +10.9% | +5.1 … +9.3 | 11/11 |
| AV1 Faster | SVT-AV1 M12 | -21.6% | -21.1% | -27.8% | -27.0 … -15.7 | 11/11 |
| AV1 Faster | AV1 Fast | +9.5% | +16.2% | +0.3% | +5.8 … +15.4 | 11/11 |
| AV1 Veryfast | SVT-AV1 M12 | -10.0% | -6.1% | -16.4% | -18.9 … -1.4 | 11/11 |
| AV1 Veryfast | SVT-AV1 M13 | -14.7% | -11.3% | -19.9% | -19.5 … -7.4 | 11/11 |
| AV1 Veryfast | AV1 Faster | +15.4% | +19.9% | +16.1% | +7.9 … +21.7 | 11/11 |
| VP9 Slow | HEVC Slow | +10.1% | +1.9% | +19.3% | +4.0 … +16.3 | 11/11 |
| VP9 | HEVC | +5.9% | -2.3% | +16.2% | +2.2 … +12.7 | 11/11 |
| VP9 | libvpx VP9 rt 7 | -46.3% | -50.7% | -45.1% | -60.9 … -39.7 | 11/11 |
| VP9 Fast | HEVC Fast | +8.5% | -0.2% | +17.5% | +0.6 … +16.7 | 11/11 |
| VP9 Fast | libvpx VP9 rt 9 | -41.4% | -47.6% | -39.5% | -49.2 … -35.1 | 11/11 |
| VP9 Fast | VP9 | +11.4% | +12.4% | +13.3% | +7.4 … +12.8 | 11/11 |
| VP9 Faster | HEVC Faster | +3.8% | +3.3% | +6.9% | -4.3 … +9.8 | 11/11 |
| VP9 Faster | libvpx VP9 rt 9 | -36.9% | -39.4% | -37.9% | -51.8 … -26.1 | 11/11 |
| VP9 Faster | VP9 Fast | +9.1% | +17.6% | +2.2% | +3.5 … +15.6 | 11/11 |
| VP9 Veryfast | HEVC Faster | +15.5% | +12.2% | +15.1% | +10.9 … +17.5 | 11/11 |
| VP9 Veryfast | libvpx VP9 rt 9 | -30.9% | -35.2% | -33.3% | -42.4 … -20.7 | 11/11 |
| VP9 Veryfast | VP9 Faster | +12.8% | +9.9% | +8.0% | +5.8 … +12.3 | 11/11 |
| AV2 Slow | AV1 Slow | -6.7% | -6.9% | -12.0% | -8.6 … -5.7 | 11/11 |
| AV2 | AV1 | -4.5% | -4.6% | -7.5% | -7.2 … -2.7 | 11/11 |
| AV2 | SVT-AV1 M8 | -18.2% | -17.9% | -31.7% | -28.7 … -11.7 | 11/11 |
| AV2 Fast | AV1 Fast | -1.8% | +1.5% | -2.9% | -5.1 … +3.1 | 11/11 |
| AV2 Fast | AV2 | +11.4% | +15.6% | +16.7% | +8.9 … +11.7 | 11/11 |
| AV2 Faster | AV1 Faster | -6.2% | -8.7% | +1.3% | -12.0 … +0.7 | 11/11 |
| AV2 Faster | SVT-AV1 M12 | -27.6% | -28.5% | -27.4% | -33.3 … -24.0 | 11/11 |
| AV2 Faster | AV2 Fast | +4.0% | +4.2% | +4.5% | +0.4 … +4.6 | 11/11 |
| VVC Slow | HEVC Slow | -9.2% | -9.5% | -9.4% | -12.6 … -3.4 | 11/11 |
| VVC | HEVC | -11.5% | -11.7% | -10.1% | -17.3 … -5.7 | 11/11 |
| VVC | x265 medium | -29.8% | -31.5% | -37.2% | -44.5 … -18.6 | 11/11 |
| VVC Fast | HEVC Fast | -7.4% | -8.4% | -12.5% | -13.4 … -1.6 | 11/11 |
| VVC Fast | VVC | +14.9% | +15.7% | +9.5% | +11.4 … +15.5 | 11/11 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 730e4e16 (H.264 Fast, H.264 Faster, HEVC, HEVC Fast, AV1, AV1 Fast, VP9, VP9 Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, AV1 Veryfast, HEVC Faster, H.264 Veryfast, VP9 Faster, VP9 Veryfast, HEVC Slow, AV1 Slow, VVC Slow, H.264 Slow, VP9 Slow, AV2 Slow, AV2 Faster: 20d31869; H.264: c9cff89f); H.264 timed on 20d31869 (same streams), its presets), openh264
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
