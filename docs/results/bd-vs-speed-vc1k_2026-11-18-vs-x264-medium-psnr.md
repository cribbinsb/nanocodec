# Bitrate vs encode speed, VC-1K phone video calls (vc1k_2026-11-18), zero line x264-medium~psnr

<sub>Updated 2026-11-19, build 730e4e16 (H.264 Fast, H.264 Faster, H.264 Veryfast, VP9 Faster: 478e8249; VP9, AV1 Slow: 77e93089; AV1, AV1 Fast, VP9 Fast, AV1 Veryfast: b33ba3bb; VP9 Slow, AV2 Slow: bad14976; H.264, HEVC, HEVC Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, HEVC Faster, VP9 Veryfast, HEVC Slow, VVC Slow, H.264 Slow, AV2 Faster: c9cff89f); H.264 Fast, H.264 Faster, H.264 Veryfast, VP9 Faster, AV1 Slow, VP9, AV2 Slow, VP9 Slow, AV1, AV1 Fast, AV1 Veryfast, VP9 Fast timed on c9cff89f (same streams)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 100 clips of `vc1k_2026-11-18`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-vc1k_2026-11-18-vs-x264-medium-psnr.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-vc1k_2026-11-18-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **AV2 Slow** | -50.5% | -53.9% | -49.4% | -54.7 … -44.8 | 22.0 | 0.82× | 81.6 | 0.72× | 100/100 | Pareto; Pareto x86 |
| **AV2** | -49.0% | -52.5% | -48.2% | -53.1 … -43.1 | 37.9 | 1.42× | 164.2 | 1.46× | 100/100 | Pareto; Pareto x86 |
| **VVC** | -47.6% | -48.3% | -45.4% | -53.1 … -41.9 | 28.9 | 1.08× | 118.1 | 1.05× | 100/100 |  |
| **VVC Slow** | -47.1% | -47.8% | -45.2% | -52.3 … -40.9 | 24.8 | 0.93× | 97.7 | 0.87× | 100/100 |  |
| **AV1 Slow** | -45.9% | -49.2% | -43.7% | -49.3 … -39.6 | 29.7 | 1.11× | 116.6 | 1.03× | 100/100 |  |
| **AV1** | -45.9% | -49.6% | -42.9% | -49.6 … -39.9 | 45.0 | 1.69× | 171.4 | 1.52× | 100/100 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -44.9% | -48.2% | -37.7% | -50.7 … -38.9 | 14.3 | 0.53× | 80.8 | 0.72× | 100/100 | 2 rate-invalid pts |
| **AV2 Fast** | -44.4% | -46.1% | -42.4% | -49.3 … -37.8 | 50.0 | 1.87× | 223.4 | 1.98× | 100/100 | Pareto; Pareto x86 |
| **AV1 Fast** | -43.8% | -47.1% | -40.5% | -47.8 … -37.2 | 65.5 | 2.46× | 280.4 | 2.49× | 100/100 | Pareto; Pareto x86 |
| **VP9 Slow** | -43.0% | -49.7% | -35.4% | -46.4 … -36.7 | 36.7 | 1.38× | 151.8 | 1.35× | 100/100 |  |
| **HEVC Slow** | -42.3% | -43.7% | -39.4% | -48.7 … -36.4 | 31.0 | 1.16× | 116.4 | 1.03× | 100/100 |  |
| **VP9** | -42.3% | -49.2% | -35.1% | -45.8 … -36.5 | 43.2 | 1.62× | 172.4 | 1.53× | 100/100 |  |
| **HEVC** | -41.4% | -43.2% | -38.6% | -47.1 … -35.2 | 50.4 | 1.89× | 201.2 | 1.78× | 100/100 |  |
| x265 medium | -40.8% | -40.7% | -36.0% | -44.4 … -35.4 | 8.0 | 0.30× | 38.0 | 0.34× | 100/100 | 1 rate-invalid pts |
| SVT-AV1 M10 | -39.7% | -43.3% | -33.7% | -45.6 … -33.2 | 39.3 | 1.47× | 228.9 | 2.03× | 100/100 | 2 rate-invalid pts |
| **VVC Fast** | -38.9% | -37.6% | -39.9% | -45.2 … -31.0 | 55.1 | 2.07× | 221.2 | 1.96× | 100/100 |  |
| **AV1 Faster** | -38.8% | -40.2% | -38.5% | -43.7 … -31.1 | 78.4 | 2.94× | 312.7 | 2.77× | 100/100 | Pareto; Pareto x86 |
| **AV2 Faster** | -38.7% | -39.1% | -36.9% | -45.4 … -32.0 | 59.5 | 2.23× | 254.8 | 2.26× | 100/100 |  |
| **VP9 Fast** | -38.5% | -44.6% | -31.5% | -41.7 … -32.0 | 54.4 | 2.04× | 227.3 | 2.02× | 100/100 |  |
| x265 superfast | -36.2% | -36.2% | -32.2% | -41.5 … -29.3 | 12.1 | 0.45× | 57.9 | 0.51× | 100/100 |  |
| **HEVC Fast** | -36.1% | -36.5% | -32.6% | -41.4 … -29.2 | 78.2 | 2.93× | 304.6 | 2.70× | 100/100 |  |
| x265 ultrafast | -33.9% | -33.9% | -29.3% | -40.3 … -26.3 | 17.1 | 0.64× | 85.1 | 0.75× | 100/100 |  |
| **AV1 Veryfast** | -33.6% | -34.5% | -30.5% | -39.1 … -26.2 | 95.9 | 3.59× | 425.3 | 3.77× | 100/100 | Pareto; Pareto x86 |
| SVT-AV1 M12 | -30.4% | -31.4% | -27.9% | -36.1 … -22.7 | 70.9 | 2.67× | 408.7 | 3.62× | 100/100 | 2 rate-invalid pts |
| **HEVC Faster** | -28.9% | -26.7% | -25.7% | -33.6 … -20.3 | 101.4 | 3.80× | 417.8 | 3.71× | 100/100 | Pareto |
| libaom rt 8 | -25.8% | -26.9% | -18.2% | -31.4 … -20.8 | 17.4 | 0.65× | 68.8 | 0.61× | 100/100 | 32 rate-invalid pts |
| SVT-AV1 M13 | -25.7% | -26.8% | -23.5% | -31.8 … -17.3 | 79.2 | 2.96× | 445.4 | 3.95× | 100/100 | Pareto x86; 2 rate-invalid pts |
| **VP9 Faster** | -25.6% | -25.9% | -25.4% | -29.7 … -16.6 | 92.0 | 3.45× | 354.1 | 3.14× | 100/100 |  |
| libaom rt 10 | -24.0% | -26.3% | -16.7% | -28.1 … -19.0 | 22.3 | 0.83× | 90.5 | 0.80× | 100/100 | 28 rate-invalid pts |
| **VP9 Veryfast** | -16.4% | -19.8% | -18.2% | -23.1 … -7.8 | 94.3 | 3.53× | 390.7 | 3.47× | 100/100 |  |
| libvpx VP9 rt 7 | -1.2% | -1.5% | +3.7% | -8.2 … +2.1 | 30.0 | 1.12× | 122.7 | 1.09× | 100/100 | 50 rate-invalid pts |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 100/100 | 15 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +0.5% | +1.6% | +5.7% | -5.9 … +5.2 | 47.8 | 1.78× | 204.4 | 1.81× | 100/100 | 50 rate-invalid pts |
| **H.264 Slow** | +0.9% | -4.6% | +2.5% | -11.1 … +11.1 | 42.0 | 1.57× | 123.7 | 1.10× | 100/100 |  |
| **H.264** | +1.8% | -3.6% | +3.5% | -10.4 … +12.3 | 65.9 | 2.47× | 205.2 | 1.82× | 100/100 |  |
| x264 medium | +13.5% | +7.6% | +27.4% | +9.2 … +17.1 | 26.7 | 1.00× | 112.8 | 1.00× | 100/100 | 14 rate-invalid pts |
| **H.264 Fast** | +19.7% | +14.4% | +16.9% | +12.6 … +30.3 | 121.2 | 4.54× | 379.5 | 3.37× | 100/100 | Pareto |
| **H.264 Faster** | +28.7% | +20.9% | +30.2% | +20.6 … +42.9 | 145.7 | 5.46× | 486.1 | 4.31× | 100/100 | Pareto; Pareto x86 |
| **H.264 Veryfast** | +35.3% | +27.8% | +39.2% | +26.6 … +47.3 | 166.2 | 6.23× | 526.2 | 4.67× | 100/100 | Pareto; Pareto x86 |
| x264 veryfast | +35.8% | +26.4% | +50.2% | +28.8 … +44.9 | 85.4 | 3.18× | 363.3 | 3.22× | 100/100 |  |
| x264 ultrafast | +75.4% | +79.4% | +58.4% | +69.4 … +91.1 | 190.5 | 7.11× | 1017.0 | 9.02× | 100/100 | Pareto; Pareto x86 |
| openh264 | +84.1% | +77.1% | +106.9% | +70.1 … +99.2 | 111.5 | 4.17× | 524.7 | 4.65× | 100/100 | 123 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: vc1k_android_indoor_fastmotion_darklight_720x1280_30_3683766447, vc1k_android_outdoor_fastmotion_mediumlight_720x1280_30_3685120542, vc1k_android_outdoor_mediummotion_darklight_720x1280_30_3685120604, vc1k_ios_indoor_mediummotion_goodlight_1280x720_30_3685486349, vc1k_android_indoor_mediummotion_lowlight_720x1280_30_3688853878, vc1k_ios_outdoor_mediummotion_lowlight_720x1280_30_3685586878, vc1k_android_outdoor_mediummotion_lowlight_720x1280_30_3684302390, vc1k_android_indoor_slowmotion_darklight_1280x720_30_3688244638, vc1k_android_indoor_slowmotion_darklight_1280x720_30_3689905294, vc1k_android_outdoor_slowmotion_darklight_1280x720_30_3689905293, vc1k_ios_indoor_slowmotion_goodlight_1280x720_30_3687390482, vc1k_ios_indoor_slowmotion_goodlight_720x1280_30_3683837796, vc1k_ios_indoor_slowmotion_lowlight_720x1280_30_3685486350, vc1k_android_indoor_slowmotion_lowlight_720x1280_30_3684329765, vc1k_android_indoor_slowmotion_lowlight_720x1280_30_3683570068, vc1k_android_indoor_slowmotion_lowlight_1280x720_30_3689905288, vc1k_android_outdoor_slowmotion_lowlight_1280x720_30_3685800581, vc1k_ios_outdoor_slowmotion_mediumlight_720x1280_30_3683837800. Clips left out for every encoder: 0; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (videocall 100 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +13.5% | +13.5% |
| H.264 Slow | +0.9% | +0.9% |
| H.264 | +1.8% | +1.8% |
| H.264 Fast | +19.7% | +19.7% |
| H.264 Faster | +28.7% | +28.7% |
| H.264 Veryfast | +35.3% | +35.3% |
| HEVC Slow | -42.3% | -42.3% |
| HEVC | -41.4% | -41.4% |
| HEVC Fast | -36.1% | -36.1% |
| HEVC Faster | -28.9% | -28.9% |
| AV1 Slow | -45.9% | -45.9% |
| AV1 | -45.9% | -45.9% |
| AV1 Fast | -43.8% | -43.8% |
| AV1 Faster | -38.8% | -38.8% |
| AV1 Veryfast | -33.6% | -33.6% |
| VP9 Slow | -43.0% | -43.0% |
| VP9 | -42.3% | -42.3% |
| VP9 Fast | -38.5% | -38.5% |
| VP9 Faster | -25.6% | -25.6% |
| VP9 Veryfast | -16.4% | -16.4% |
| AV2 Slow | -50.5% | -50.5% |
| AV2 | -49.0% | -49.0% |
| AV2 Fast | -44.4% | -44.4% |
| AV2 Faster | -38.7% | -38.7% |
| VVC Slow | -47.1% | -47.1% |
| VVC | -47.6% | -47.6% |
| VVC Fast | -38.9% | -38.9% |
| openh264 | +84.1% | +84.1% |
| x264 ultrafast | +75.4% | +75.4% |
| x264 veryfast | +35.8% | +35.8% |
| x265 ultrafast | -33.9% | -33.9% |
| x265 superfast | -36.2% | -36.2% |
| x265 medium | -40.8% | -40.8% |
| SVT-AV1 M8 | -44.9% | -44.9% |
| SVT-AV1 M10 | -39.7% | -39.7% |
| SVT-AV1 M12 | -30.4% | -30.4% |
| SVT-AV1 M13 | -25.7% | -25.7% |
| libvpx VP9 rt 7 | -1.2% | -1.2% |
| libvpx VP9 rt 9 | +0.5% | +0.5% |
| libaom rt 8 | -25.8% | -25.8% |
| libaom rt 10 | -24.0% | -24.0% |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **AV2 Slow** | -60.5% | -63.4% | -65.1% | -53.1% | 100/100 |
| **AV2** | -59.3% | -62.3% | -64.3% | -51.3% | 100/100 |
| **VVC** | -58.0% | -58.1% | -62.3% | -53.7% | 100/100 |
| **VVC Slow** | -57.7% | -57.7% | -62.2% | -53.2% | 100/100 |
| **AV1 Slow** | -56.4% | -58.8% | -60.7% | -49.6% | 100/100 |
| **AV1** | -56.3% | -59.2% | -59.9% | -49.9% | 100/100 |
| SVT-AV1 M8 | -55.7% | -57.9% | -56.0% | -53.3% | 100/100 |
| **AV2 Fast** | -55.5% | -56.6% | -60.1% | -49.9% | 100/100 |
| **AV1 Fast** | -54.6% | -57.0% | -58.1% | -48.7% | 100/100 |
| **VP9 Slow** | -54.1% | -59.2% | -54.5% | -48.7% | 100/100 |
| **VP9** | -53.5% | -58.8% | -54.2% | -47.5% | 100/100 |
| **HEVC Slow** | -53.3% | -53.7% | -57.6% | -48.7% | 100/100 |
| **HEVC** | -52.5% | -53.3% | -57.0% | -47.2% | 100/100 |
| x265 medium | -51.6% | -50.3% | -54.4% | -50.2% | 100/100 |
| SVT-AV1 M10 | -51.2% | -53.4% | -52.9% | -47.3% | 100/100 |
| **AV2 Faster** | -50.6% | -50.1% | -56.4% | -45.3% | 100/100 |
| **VP9 Fast** | -50.3% | -54.8% | -51.6% | -44.5% | 100/100 |
| **VVC Fast** | -50.0% | -47.9% | -58.0% | -44.2% | 100/100 |
| **AV1 Faster** | -49.9% | -50.1% | -56.7% | -42.7% | 100/100 |
| **HEVC Fast** | -47.9% | -47.0% | -52.4% | -44.3% | 100/100 |
| x265 superfast | -47.6% | -46.1% | -51.6% | -45.1% | 100/100 |
| x265 ultrafast | -45.6% | -43.9% | -49.3% | -43.6% | 100/100 |
| **AV1 Veryfast** | -45.2% | -44.1% | -50.8% | -40.7% | 100/100 |
| SVT-AV1 M12 | -42.7% | -42.1% | -48.6% | -37.5% | 100/100 |
| **HEVC Faster** | -40.6% | -36.4% | -46.4% | -39.1% | 100/100 |
| SVT-AV1 M13 | -38.5% | -37.6% | -45.2% | -32.6% | 100/100 |
| **VP9 Faster** | -37.9% | -36.1% | -46.9% | -30.6% | 100/100 |
| libaom rt 8 | -37.8% | -36.0% | -40.2% | -37.2% | 100/100 |
| libaom rt 10 | -36.4% | -36.0% | -39.3% | -33.8% | 100/100 |
| **VP9 Veryfast** | -28.2% | -28.9% | -40.5% | -15.4% | 100/100 |
| libvpx VP9 rt 7 | -14.1% | -10.5% | -21.2% | -10.7% | 100/100 |
| libvpx VP9 rt 9 | -13.2% | -8.2% | -20.5% | -11.0% | 100/100 |
| **H.264 Slow** | -10.7% | -12.3% | -20.0% | +0.2% | 100/100 |
| x264 medium, metric-tuned (--tune psnr) | -10.6% | -6.6% | -20.6% | -4.6% | 100/100 |
| **H.264** | -10.0% | -11.3% | -19.3% | +0.7% | 100/100 |
| **H.264 Fast** | +7.5% | +7.3% | -8.4% | +23.6% | 100/100 |
| **H.264 Faster** | +15.9% | +14.3% | +2.1% | +31.3% | 100/100 |
| x264 veryfast | +20.4% | +18.5% | +17.8% | +24.8% | 100/100 |
| **H.264 Veryfast** | +22.3% | +21.9% | +9.4% | +35.4% | 100/100 |
| x264 ultrafast | +59.8% | +72.1% | +25.1% | +82.1% | 100/100 |
| openh264 | +61.1% | +62.5% | +61.5% | +59.3% | 100/100 |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slow | openh264 | -43.6% | -44.7% | -49.7% | -49.1 … -38.9 | 100/100 |
| H.264 | openh264 | -43.1% | -44.0% | -49.2% | -48.7 … -38.3 | 100/100 |
| H.264 | x264 veryfast | -27.4% | -27.6% | -33.1% | -34.0 … -20.5 | 100/100 |
| H.264 | x264 medium | -10.0% | -11.3% | -19.3% | -20.1 … +2.0 | 100/100 |
| H.264 Fast | openh264 | -31.1% | -30.7% | -42.5% | -37.3 … -26.1 | 99/100 |
| H.264 Fast | x264 veryfast | -13.7% | -12.6% | -23.3% | -19.5 … -5.4 | 100/100 |
| H.264 Fast | H.264 | +21.5% | +23.6% | +14.6% | +12.4 … +26.6 | 100/100 |
| H.264 Faster | openh264 | -25.8% | -26.2% | -35.8% | -32.4 … -21.9 | 99/100 |
| H.264 Faster | x264 ultrafast | -29.7% | -38.2% | -20.3% | -35.0 … -23.5 | 100/100 |
| H.264 Faster | H.264 Fast | +9.6% | +7.6% | +12.4% | +4.2 … +13.6 | 100/100 |
| H.264 Veryfast | openh264 | -21.5% | -20.8% | -31.1% | -28.8 … -18.0 | 99/100 |
| H.264 Veryfast | x264 ultrafast | -25.3% | -33.5% | -14.5% | -31.6 … -20.1 | 100/100 |
| H.264 Veryfast | H.264 Faster | +7.2% | +8.8% | +7.5% | +4.7 … +9.2 | 100/100 |
| HEVC Slow | SVT-AV1 M12 | -19.7% | -21.6% | -18.7% | -29.3 … -11.1 | 99/100 |
| HEVC | SVT-AV1 M12 | -18.0% | -20.8% | -16.9% | -28.0 … -9.9 | 99/100 |
| HEVC | x265 medium | -1.5% | -5.7% | -5.5% | -10.7 … +8.9 | 100/100 |
| HEVC Fast | SVT-AV1 M12 | -7.5% | -6.6% | -6.1% | -17.0 … -1.5 | 99/100 |
| HEVC Fast | x265 superfast | +0.8% | +0.1% | -0.8% | -14.1 … +11.0 | 100/100 |
| HEVC Fast | HEVC | +12.9% | +17.5% | +13.5% | +10.0 … +15.2 | 100/100 |
| HEVC Faster | SVT-AV1 M12 | +7.3% | +15.0% | +7.3% | -4.4 … +14.4 | 99/100 |
| HEVC Faster | SVT-AV1 M13 | -0.4% | +7.3% | -0.8% | -12.3 … +7.6 | 100/100 |
| HEVC Faster | HEVC Fast | +18.9% | +26.7% | +16.0% | +14.1 … +23.5 | 100/100 |
| AV1 Slow | SVT-AV1 M12 | -26.8% | -32.4% | -25.7% | -33.9 … -21.9 | 99/100 |
| AV1 | SVT-AV1 M12 | -27.3% | -34.0% | -24.3% | -33.9 … -23.0 | 99/100 |
| AV1 | openh264 | -70.8% | -72.3% | -72.4% | -74.8 … -68.2 | 82/100 |
| AV1 | SVT-AV1 M8 | +2.3% | +1.7% | -7.8% | -9.2 … +9.8 | 99/100 |
| AV1 | libaom rt 8 | -27.0% | -29.5% | -32.9% | -34.0 … -23.6 | 100/100 |
| AV1 Fast | SVT-AV1 M12 | -24.0% | -29.9% | -20.7% | -29.1 … -20.2 | 99/100 |
| AV1 Fast | openh264 | -69.9% | -71.3% | -71.2% | -73.6 … -67.5 | 90/100 |
| AV1 Fast | AV1 | +5.2% | +6.8% | +5.5% | +2.6 … +7.6 | 100/100 |
| AV1 Faster | SVT-AV1 M12 | -13.7% | -15.2% | -17.5% | -22.9 … -6.7 | 99/100 |
| AV1 Faster | AV1 Fast | +14.8% | +22.5% | +4.0% | +8.0 … +19.8 | 100/100 |
| AV1 Veryfast | SVT-AV1 M12 | -4.0% | -3.6% | -4.2% | -15.8 … +4.3 | 99/100 |
| AV1 Veryfast | SVT-AV1 M13 | -13.2% | -13.3% | -11.6% | -23.7 … -5.1 | 99/100 |
| AV1 Veryfast | AV1 Faster | +12.6% | +16.5% | +16.3% | +6.4 … +18.6 | 100/100 |
| VP9 Slow | HEVC Slow | -1.4% | -13.5% | +9.0% | -6.6 … +3.5 | 100/100 |
| VP9 | HEVC | -2.0% | -13.5% | +8.2% | -7.0 … +2.1 | 100/100 |
| VP9 | libvpx VP9 rt 7 | -43.6% | -49.7% | -39.0% | -48.1 … -39.1 | 100/100 |
| VP9 Fast | HEVC Fast | -5.6% | -18.1% | +2.2% | -9.8 … -2.9 | 100/100 |
| VP9 Fast | libvpx VP9 rt 9 | -42.6% | -48.9% | -38.0% | -46.5 … -36.3 | 100/100 |
| VP9 Fast | VP9 | +9.1% | +12.1% | +7.2% | +6.2 … +11.9 | 100/100 |
| VP9 Faster | HEVC Faster | +5.0% | -1.0% | -0.9% | -1.9 … +10.9 | 100/100 |
| VP9 Faster | libvpx VP9 rt 9 | -27.7% | -28.3% | -31.6% | -34.8 … -19.6 | 100/100 |
| VP9 Faster | VP9 Fast | +36.6% | +57.2% | +12.8% | +24.4 … +45.2 | 99/100 |
| VP9 Veryfast | HEVC Faster | +28.6% | +18.5% | +15.1% | +12.9 … +43.6 | 94/100 |
| VP9 Veryfast | libvpx VP9 rt 9 | -14.8% | -19.9% | -22.7% | -24.7 … -7.0 | 100/100 |
| VP9 Veryfast | VP9 Faster | +28.3% | +20.2% | +17.2% | +14.4 … +34.1 | 99/100 |
| AV2 Slow | AV1 Slow | -11.9% | -14.0% | -12.8% | -15.1 … -8.7 | 99/100 |
| AV2 | AV1 | -8.8% | -10.7% | -12.1% | -11.5 … -5.8 | 99/100 |
| AV2 | SVT-AV1 M8 | -8.3% | -8.1% | -20.2% | -15.7 … -0.8 | 99/100 |
| AV2 Fast | AV1 Fast | -4.2% | -0.5% | -6.4% | -7.3 … +1.6 | 100/100 |
| AV2 Fast | AV2 | +11.8% | +18.7% | +13.6% | +8.7 … +15.2 | 100/100 |
| AV2 Faster | AV1 Faster | -5.1% | -4.5% | +0.2% | -12.0 … +2.3 | 100/100 |
| AV2 Faster | SVT-AV1 M12 | -18.7% | -18.3% | -18.0% | -28.6 … -8.6 | 99/100 |
| AV2 Faster | AV2 Fast | +14.6% | +18.7% | +12.5% | +3.5 … +22.3 | 100/100 |
| VVC Slow | HEVC Slow | -10.5% | -8.9% | -11.8% | -14.2 … -8.3 | 100/100 |
| VVC | HEVC | -13.3% | -11.2% | -13.5% | -17.1 … -10.9 | 100/100 |
| VVC | x265 medium | -14.3% | -16.3% | -18.2% | -21.9 … -5.6 | 100/100 |
| VVC Fast | HEVC Fast | -6.3% | +0.7% | -13.7% | -10.8 … -3.1 | 99/100 |
| VVC Fast | VVC | +25.0% | +31.7% | +13.7% | +18.5 … +28.1 | 100/100 |

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
  capped at half the source bitrate; 100 natural VAL clips in 1 content
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
- **x86 speed (`bd-vs-speed-vc1k_2026-11-18-vs-x264-medium-psnr-x86.svg`, `.csv`):** the same streams (same quantizers and
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
  gain. No variant is distinguishably better than its charted setting. The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
