# Bitrate vs encode speed, VC-1K phone video calls (vc1k_2026-11-18), zero line x264-medium~psnr

<sub>Updated 2026-11-20, build 730e4e16 (H.264 Fast, H.264 Faster, HEVC, HEVC Fast, AV1, AV1 Fast, VP9, VP9 Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, AV1 Veryfast, HEVC Faster, H.264 Veryfast, VP9 Faster, VP9 Veryfast, HEVC Slow, AV1 Slow, VVC Slow, H.264 Slow, VP9 Slow, AV2 Slow, AV2 Faster: 20d31869; H.264: c9cff89f); H.264 timed on 20d31869 (same streams)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 100 clips of `vc1k_2026-11-18`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-vc1k_2026-11-18-vs-x264-medium-psnr.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-vc1k_2026-11-18-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **AV2 Slow** | -52.6% | -55.8% | -51.5% | -55.9 … -47.0 | 16.1 | 0.60× | 62.3 | 0.55× | 100/100 | Pareto; Pareto x86 |
| **AV2** | -49.3% | -52.9% | -48.6% | -53.2 … -43.2 | 39.2 | 1.47× | 170.1 | 1.51× | 100/100 | Pareto; Pareto x86 |
| **AV1 Slow** | -49.3% | -52.9% | -45.2% | -53.3 … -43.1 | 26.6 | 1.00× | 113.3 | 1.00× | 100/100 |  |
| **VVC Slow** | -49.1% | -50.3% | -47.2% | -53.1 … -42.3 | 16.0 | 0.60× | 64.4 | 0.57× | 100/100 |  |
| **VVC** | -48.5% | -49.5% | -46.7% | -53.0 … -42.4 | 29.8 | 1.12× | 121.0 | 1.07× | 100/100 |  |
| **AV1** | -46.2% | -49.9% | -43.4% | -49.6 … -39.8 | 45.7 | 1.71× | 181.4 | 1.61× | 100/100 | Pareto; Pareto x86 |
| **HEVC Slow** | -45.8% | -47.4% | -43.5% | -51.1 … -39.6 | 19.0 | 0.71× | 76.9 | 0.68× | 100/100 |  |
| SVT-AV1 M8 | -45.5% | -48.8% | -38.3% | -51.3 … -38.9 | 14.3 | 0.53× | 80.8 | 0.72× | 100/100 | 2 rate-invalid pts |
| **AV2 Fast** | -44.8% | -46.4% | -42.8% | -48.7 … -37.8 | 51.8 | 1.94× | 230.8 | 2.05× | 100/100 | Pareto; Pareto x86 |
| **AV1 Fast** | -44.1% | -47.4% | -40.8% | -48.6 … -37.5 | 66.2 | 2.48× | 267.6 | 2.37× | 100/100 | Pareto; Pareto x86 |
| **VP9 Slow** | -43.5% | -50.2% | -36.0% | -46.9 … -36.9 | 26.7 | 1.00× | 100.0 | 0.89× | 100/100 |  |
| **HEVC** | -43.5% | -45.1% | -41.6% | -48.9 … -36.8 | 45.3 | 1.70× | 171.9 | 1.52× | 100/100 |  |
| **VP9** | -42.7% | -49.6% | -35.7% | -46.2 … -36.6 | 42.1 | 1.58× | 171.0 | 1.52× | 100/100 |  |
| x265 medium | -41.3% | -41.2% | -36.6% | -45.7 … -35.5 | 8.0 | 0.30× | 38.0 | 0.34× | 100/100 | 1 rate-invalid pts |
| **VVC Fast** | -41.3% | -42.2% | -41.9% | -48.2 … -34.4 | 53.2 | 1.99× | 205.2 | 1.82× | 100/100 |  |
| SVT-AV1 M10 | -40.3% | -43.9% | -34.2% | -45.8 … -33.5 | 39.3 | 1.47× | 228.9 | 2.03× | 100/100 | 2 rate-invalid pts |
| **VP9 Fast** | -39.3% | -45.3% | -32.4% | -42.7 … -32.4 | 55.8 | 2.09× | 233.7 | 2.07× | 100/100 |  |
| **AV1 Faster** | -39.2% | -40.5% | -38.9% | -43.9 … -31.2 | 79.4 | 2.98× | 323.9 | 2.87× | 100/100 | Pareto; Pareto x86 |
| **AV2 Faster** | -39.0% | -39.4% | -37.0% | -45.2 … -32.0 | 62.9 | 2.36× | 274.8 | 2.44× | 100/100 |  |
| **HEVC Fast** | -38.6% | -39.2% | -34.9% | -43.1 … -31.3 | 72.0 | 2.70× | 293.2 | 2.60× | 100/100 |  |
| x265 superfast | -36.8% | -36.8% | -32.8% | -43.4 … -30.5 | 12.1 | 0.45× | 57.9 | 0.51× | 100/100 |  |
| x265 ultrafast | -34.6% | -34.6% | -30.0% | -42.6 … -27.5 | 17.1 | 0.64× | 85.1 | 0.75× | 100/100 |  |
| **HEVC Faster** | -34.6% | -34.8% | -31.0% | -38.0 … -27.1 | 90.3 | 3.38× | 364.2 | 3.23× | 100/100 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -34.1% | -35.0% | -31.1% | -39.4 … -26.0 | 96.5 | 3.62× | 431.4 | 3.83× | 100/100 | Pareto; Pareto x86 |
| SVT-AV1 M12 | -30.9% | -31.8% | -28.4% | -37.0 … -22.7 | 70.9 | 2.67× | 408.7 | 3.62× | 100/100 | 2 rate-invalid pts |
| **VP9 Faster** | -30.3% | -31.6% | -27.2% | -34.7 … -22.0 | 92.3 | 3.46× | 368.7 | 3.27× | 100/100 |  |
| SVT-AV1 M13 | -26.3% | -27.3% | -24.1% | -32.6 … -17.3 | 79.2 | 2.96× | 445.4 | 3.95× | 100/100 | Pareto x86; 2 rate-invalid pts |
| libaom rt 8 | -25.8% | -26.8% | -18.2% | -31.4 … -20.8 | 17.4 | 0.65× | 68.8 | 0.61× | 100/100 | 37 rate-invalid pts |
| **VP9 Veryfast** | -24.8% | -28.4% | -21.5% | -30.4 … -15.7 | 95.8 | 3.59× | 385.9 | 3.42× | 100/100 |  |
| libaom rt 10 | -24.1% | -26.5% | -16.9% | -28.4 … -19.0 | 22.3 | 0.83× | 90.5 | 0.80× | 100/100 | 32 rate-invalid pts |
| **H.264 Slow** | -2.1% | -7.8% | -0.2% | -14.1 … +7.2 | 37.2 | 1.39× | 115.3 | 1.02× | 100/100 |  |
| libvpx VP9 rt 7 | -1.2% | -1.4% | +3.7% | -7.8 … +2.4 | 30.0 | 1.12× | 122.7 | 1.09× | 100/100 | 56 rate-invalid pts |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 100/100 | 20 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +0.4% | +1.6% | +5.6% | -5.9 … +5.2 | 47.8 | 1.78× | 204.4 | 1.81× | 100/100 | 55 rate-invalid pts |
| **H.264** | +1.4% | -3.9% | +3.0% | -11.4 … +12.3 | 66.4 | 2.49× | 202.2 | 1.79× | 100/100 |  |
| **H.264 Fast** | +12.2% | +6.9% | +13.6% | +5.1 … +22.3 | 119.5 | 4.48× | 370.1 | 3.28× | 100/100 | Pareto |
| x264 medium | +13.2% | +7.3% | +27.0% | +8.9 … +16.7 | 26.7 | 1.00× | 112.8 | 1.00× | 100/100 | 18 rate-invalid pts |
| **H.264 Faster** | +23.8% | +16.0% | +27.2% | +15.8 … +34.5 | 146.9 | 5.51× | 460.1 | 4.08× | 100/100 | Pareto; Pareto x86 |
| **H.264 Veryfast** | +26.1% | +17.7% | +31.8% | +18.4 … +36.8 | 164.1 | 6.15× | 520.2 | 4.61× | 100/100 | Pareto; Pareto x86 |
| x264 veryfast | +34.7% | +25.1% | +48.7% | +28.2 … +44.8 | 85.4 | 3.18× | 363.3 | 3.22× | 100/100 |  |
| x264 ultrafast | +73.6% | +76.6% | +57.3% | +66.5 … +91.2 | 190.5 | 7.11× | 1017.0 | 9.02× | 100/100 | Pareto; Pareto x86 |
| openh264 | +84.0% | +77.2% | +106.8% | +70.1 … +98.7 | 111.5 | 4.17× | 524.7 | 4.65× | 100/100 | 129 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: vc1k_android_indoor_fastmotion_darklight_720x1280_30_3683766447, vc1k_android_outdoor_fastmotion_mediumlight_720x1280_30_3685120542, vc1k_android_outdoor_mediummotion_darklight_720x1280_30_3685120604, vc1k_ios_indoor_mediummotion_goodlight_1280x720_30_3685486349, vc1k_android_indoor_mediummotion_lowlight_720x1280_30_3688853878, vc1k_ios_outdoor_mediummotion_lowlight_720x1280_30_3685586878, vc1k_android_outdoor_mediummotion_lowlight_720x1280_30_3684302390, vc1k_ios_indoor_mediummotion_mediumlight_720x1280_30_3685486318, vc1k_android_indoor_slowmotion_darklight_1280x720_30_3688244638, vc1k_android_indoor_slowmotion_darklight_1280x720_30_3689905294, vc1k_android_outdoor_slowmotion_darklight_1280x720_30_3689905293, vc1k_ios_indoor_slowmotion_goodlight_1280x720_30_3687390482, vc1k_ios_indoor_slowmotion_goodlight_720x1280_30_3683837796, vc1k_ios_indoor_slowmotion_lowlight_720x1280_30_3685486350, vc1k_android_indoor_slowmotion_lowlight_720x1280_30_3684329765, vc1k_android_indoor_slowmotion_lowlight_720x1280_30_3683570068, vc1k_android_indoor_slowmotion_lowlight_1280x720_30_3689905288, vc1k_android_outdoor_slowmotion_lowlight_1280x720_30_3685800581, vc1k_android_indoor_slowmotion_mediumlight_720x1280_30_3685591320, vc1k_ios_indoor_slowmotion_mediumlight_1280x720_30_3687663173, vc1k_ios_outdoor_slowmotion_mediumlight_720x1280_30_3683837800. Clips left out for every encoder: 0; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (videocall 100 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +13.2% | +13.2% |
| H.264 Slow | -2.1% | -2.1% |
| H.264 | +1.4% | +1.4% |
| H.264 Fast | +12.2% | +12.2% |
| H.264 Faster | +23.8% | +23.8% |
| H.264 Veryfast | +26.1% | +26.1% |
| HEVC Slow | -45.8% | -45.8% |
| HEVC | -43.5% | -43.5% |
| HEVC Fast | -38.6% | -38.6% |
| HEVC Faster | -34.6% | -34.6% |
| AV1 Slow | -49.3% | -49.3% |
| AV1 | -46.2% | -46.2% |
| AV1 Fast | -44.1% | -44.1% |
| AV1 Faster | -39.2% | -39.2% |
| AV1 Veryfast | -34.1% | -34.1% |
| VP9 Slow | -43.5% | -43.5% |
| VP9 | -42.7% | -42.7% |
| VP9 Fast | -39.3% | -39.3% |
| VP9 Faster | -30.3% | -30.3% |
| VP9 Veryfast | -24.8% | -24.8% |
| AV2 Slow | -52.6% | -52.6% |
| AV2 | -49.3% | -49.3% |
| AV2 Fast | -44.8% | -44.8% |
| AV2 Faster | -39.0% | -39.0% |
| VVC Slow | -49.1% | -49.1% |
| VVC | -48.5% | -48.5% |
| VVC Fast | -41.3% | -41.3% |
| openh264 | +84.0% | +84.0% |
| x264 ultrafast | +73.6% | +73.6% |
| x264 veryfast | +34.7% | +34.7% |
| x265 ultrafast | -34.6% | -34.6% |
| x265 superfast | -36.8% | -36.8% |
| x265 medium | -41.3% | -41.3% |
| SVT-AV1 M8 | -45.5% | -45.5% |
| SVT-AV1 M10 | -40.3% | -40.3% |
| SVT-AV1 M12 | -30.9% | -30.9% |
| SVT-AV1 M13 | -26.3% | -26.3% |
| libvpx VP9 rt 7 | -1.2% | -1.2% |
| libvpx VP9 rt 9 | +0.4% | +0.4% |
| libaom rt 8 | -25.8% | -25.8% |
| libaom rt 10 | -24.1% | -24.1% |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **AV2 Slow** | -62.2% | -64.9% | -66.4% | -55.2% | 100/100 |
| **AV2** | -59.5% | -62.5% | -64.4% | -51.7% | 100/100 |
| **VVC Slow** | -59.5% | -60.2% | -63.7% | -54.6% | 100/100 |
| **AV1 Slow** | -59.1% | -62.0% | -61.2% | -54.2% | 100/100 |
| **VVC** | -58.9% | -59.4% | -63.2% | -54.1% | 100/100 |
| **AV1** | -56.6% | -59.4% | -60.2% | -50.2% | 100/100 |
| **HEVC Slow** | -56.4% | -57.0% | -60.7% | -51.4% | 100/100 |
| SVT-AV1 M8 | -56.1% | -58.2% | -56.3% | -53.7% | 100/100 |
| **AV2 Fast** | -55.8% | -56.7% | -60.3% | -50.3% | 100/100 |
| **AV1 Fast** | -54.8% | -57.2% | -58.2% | -49.0% | 100/100 |
| **VP9 Slow** | -54.5% | -59.5% | -54.8% | -49.4% | 100/100 |
| **HEVC** | -54.4% | -55.0% | -59.3% | -48.9% | 100/100 |
| **VP9** | -53.8% | -59.0% | -54.5% | -47.9% | 100/100 |
| **VVC Fast** | -52.4% | -52.6% | -59.5% | -45.0% | 100/100 |
| x265 medium | -51.9% | -50.6% | -54.7% | -50.5% | 100/100 |
| SVT-AV1 M10 | -51.6% | -53.8% | -53.1% | -47.8% | 100/100 |
| **VP9 Fast** | -50.9% | -55.3% | -52.1% | -45.3% | 100/100 |
| **AV2 Faster** | -50.8% | -50.3% | -56.3% | -45.8% | 100/100 |
| **AV1 Faster** | -50.1% | -50.3% | -57.0% | -43.1% | 100/100 |
| **HEVC Fast** | -50.0% | -49.2% | -54.0% | -46.7% | 100/100 |
| x265 superfast | -47.9% | -46.3% | -51.7% | -45.5% | 100/100 |
| x265 ultrafast | -46.0% | -44.2% | -49.6% | -44.2% | 100/100 |
| **HEVC Faster** | -46.0% | -44.4% | -50.7% | -42.8% | 100/100 |
| **AV1 Veryfast** | -45.5% | -44.4% | -51.1% | -41.1% | 100/100 |
| SVT-AV1 M12 | -43.0% | -42.4% | -48.8% | -37.9% | 100/100 |
| **VP9 Faster** | -42.3% | -41.7% | -48.1% | -37.2% | 100/100 |
| SVT-AV1 M13 | -38.8% | -37.9% | -45.5% | -33.1% | 100/100 |
| libaom rt 8 | -37.6% | -35.7% | -40.0% | -37.1% | 100/100 |
| **VP9 Veryfast** | -36.5% | -37.6% | -43.0% | -28.7% | 100/100 |
| libaom rt 10 | -36.4% | -35.9% | -39.4% | -33.9% | 100/100 |
| libvpx VP9 rt 7 | -14.0% | -10.3% | -21.0% | -10.6% | 100/100 |
| **H.264 Slow** | -13.4% | -15.3% | -22.1% | -2.8% | 100/100 |
| libvpx VP9 rt 9 | -13.2% | -8.0% | -20.4% | -11.1% | 100/100 |
| x264 medium, metric-tuned (--tune psnr) | -10.4% | -6.4% | -20.3% | -4.5% | 100/100 |
| **H.264** | -10.1% | -11.4% | -19.4% | +0.5% | 100/100 |
| **H.264 Fast** | +0.3% | -0.1% | -10.7% | +11.9% | 100/100 |
| **H.264 Faster** | +11.2% | +9.5% | -0.1% | +24.0% | 100/100 |
| **H.264 Veryfast** | +13.3% | +11.5% | +3.6% | +24.7% | 100/100 |
| x264 veryfast | +19.7% | +17.6% | +17.0% | +24.5% | 100/100 |
| x264 ultrafast | +58.6% | +70.0% | +24.6% | +81.0% | 100/100 |
| openh264 | +61.1% | +62.6% | +61.4% | +59.3% | 100/100 |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slow | openh264 | -45.0% | -46.3% | -50.7% | -49.7 … -40.7 | 100/100 |
| H.264 | openh264 | -43.0% | -43.9% | -49.1% | -48.1 … -38.3 | 100/100 |
| H.264 | x264 veryfast | -27.1% | -27.3% | -32.7% | -33.6 … -20.5 | 100/100 |
| H.264 | x264 medium | -10.1% | -11.4% | -19.4% | -20.4 … +2.0 | 100/100 |
| H.264 Fast | openh264 | -36.3% | -36.0% | -43.9% | -42.0 … -31.3 | 99/100 |
| H.264 Fast | x264 veryfast | -18.6% | -18.2% | -24.7% | -23.7 … -10.9 | 100/100 |
| H.264 Fast | H.264 | +13.3% | +14.3% | +11.6% | +7.3 … +17.1 | 100/100 |
| H.264 Faster | openh264 | -28.9% | -29.3% | -36.9% | -34.4 … -25.1 | 99/100 |
| H.264 Faster | x264 ultrafast | -32.3% | -40.9% | -21.7% | -38.2 … -26.7 | 100/100 |
| H.264 Faster | H.264 Fast | +12.5% | +11.0% | +12.8% | +6.4 … +16.6 | 100/100 |
| H.264 Veryfast | openh264 | -27.4% | -27.9% | -34.5% | -33.4 … -23.5 | 99/100 |
| H.264 Veryfast | x264 ultrafast | -30.7% | -39.8% | -18.9% | -36.4 … -25.2 | 100/100 |
| H.264 Veryfast | H.264 Faster | +3.0% | +2.8% | +3.9% | +0.9 … +4.4 | 100/100 |
| HEVC Slow | SVT-AV1 M12 | -26.2% | -28.6% | -25.4% | -34.0 … -18.6 | 99/100 |
| HEVC | SVT-AV1 M12 | -22.0% | -24.5% | -22.2% | -30.4 … -14.4 | 99/100 |
| HEVC | x265 medium | -5.8% | -9.5% | -10.7% | -13.1 … +4.7 | 100/100 |
| HEVC Fast | SVT-AV1 M12 | -11.2% | -11.0% | -9.0% | -21.2 … -5.0 | 99/100 |
| HEVC Fast | x265 superfast | -3.5% | -4.6% | -4.2% | -19.5 … +5.7 | 100/100 |
| HEVC Fast | HEVC | +13.4% | +17.3% | +16.4% | +10.8 … +14.7 | 100/100 |
| HEVC Faster | SVT-AV1 M12 | -3.4% | -1.8% | -1.8% | -10.3 … +1.7 | 98/100 |
| HEVC Faster | SVT-AV1 M13 | -10.7% | -9.1% | -9.4% | -18.7 … -5.4 | 99/100 |
| HEVC Faster | HEVC Fast | +11.4% | +13.2% | +9.5% | +7.0 … +15.1 | 100/100 |
| AV1 Slow | SVT-AV1 M12 | -32.3% | -39.4% | -26.4% | -38.8 … -28.9 | 99/100 |
| AV1 | SVT-AV1 M12 | -27.2% | -34.0% | -24.2% | -34.1 … -23.5 | 99/100 |
| AV1 | openh264 | -71.1% | -72.7% | -72.7% | -74.9 … -68.0 | 85/100 |
| AV1 | SVT-AV1 M8 | +2.2% | +1.5% | -8.0% | -9.6 … +10.1 | 99/100 |
| AV1 | libaom rt 8 | -27.5% | -30.0% | -33.4% | -34.9 … -23.2 | 100/100 |
| AV1 Fast | SVT-AV1 M12 | -24.0% | -30.0% | -20.6% | -28.9 … -19.9 | 99/100 |
| AV1 Fast | openh264 | -70.1% | -71.5% | -71.3% | -74.1 … -67.0 | 89/100 |
| AV1 Fast | AV1 | +5.3% | +6.6% | +6.1% | +2.9 … +7.4 | 100/100 |
| AV1 Faster | SVT-AV1 M12 | -13.6% | -15.3% | -17.5% | -22.9 … -6.5 | 99/100 |
| AV1 Faster | AV1 Fast | +14.9% | +22.7% | +4.0% | +8.4 … +19.9 | 100/100 |
| AV1 Veryfast | SVT-AV1 M12 | -4.1% | -3.8% | -4.3% | -16.1 … +4.4 | 99/100 |
| AV1 Veryfast | SVT-AV1 M13 | -13.2% | -13.4% | -11.6% | -23.7 … -4.8 | 99/100 |
| AV1 Veryfast | AV1 Faster | +12.5% | +16.4% | +15.9% | +5.8 … +18.4 | 100/100 |
| VP9 Slow | HEVC Slow | +5.9% | -6.3% | +18.4% | +0.5 … +11.7 | 100/100 |
| VP9 | HEVC | +2.2% | -10.0% | +14.2% | -3.1 … +6.3 | 100/100 |
| VP9 | libvpx VP9 rt 7 | -44.2% | -50.1% | -39.6% | -48.5 … -39.1 | 99/100 |
| VP9 Fast | HEVC Fast | -2.1% | -14.7% | +5.1% | -7.4 … +1.8 | 100/100 |
| VP9 Fast | libvpx VP9 rt 9 | -43.2% | -49.5% | -38.8% | -46.8 … -37.0 | 99/100 |
| VP9 Fast | VP9 | +8.6% | +11.5% | +7.0% | +5.5 … +10.5 | 100/100 |
| VP9 Faster | HEVC Faster | +7.0% | +4.8% | +5.5% | +3.2 … +12.7 | 100/100 |
| VP9 Faster | libvpx VP9 rt 9 | -33.2% | -34.7% | -33.4% | -37.4 … -26.0 | 100/100 |
| VP9 Faster | VP9 Fast | +25.4% | +42.3% | +10.2% | +19.0 … +30.7 | 100/100 |
| VP9 Veryfast | HEVC Faster | +24.8% | +15.4% | +19.7% | +14.8 … +30.3 | 99/100 |
| VP9 Veryfast | libvpx VP9 rt 9 | -25.3% | -30.2% | -26.5% | -32.9 … -17.8 | 100/100 |
| VP9 Veryfast | VP9 Faster | +19.2% | +12.6% | +14.6% | +8.5 … +22.9 | 99/100 |
| AV2 Slow | AV1 Slow | -9.2% | -9.9% | -14.5% | -12.0 … -6.1 | 99/100 |
| AV2 | AV1 | -8.9% | -10.3% | -11.8% | -11.3 … -5.8 | 99/100 |
| AV2 | SVT-AV1 M8 | -6.4% | -8.5% | -20.3% | -15.2 … -1.0 | 100/100 |
| AV2 Fast | AV1 Fast | -4.3% | -0.4% | -6.7% | -8.1 … +1.6 | 100/100 |
| AV2 Fast | AV2 | +11.8% | +19.2% | +13.4% | +8.3 … +15.0 | 100/100 |
| AV2 Faster | AV1 Faster | -5.0% | -4.4% | +0.6% | -11.9 … +2.6 | 100/100 |
| AV2 Faster | SVT-AV1 M12 | -18.6% | -18.3% | -17.7% | -27.8 … -8.4 | 99/100 |
| AV2 Faster | AV2 Fast | +14.6% | +18.3% | +13.3% | +3.5 … +22.6 | 100/100 |
| VVC Slow | HEVC Slow | -8.2% | -7.7% | -8.5% | -12.6 … -6.0 | 100/100 |
| VVC | HEVC | -11.8% | -11.2% | -11.3% | -16.8 … -10.4 | 100/100 |
| VVC | x265 medium | -16.3% | -19.2% | -20.4% | -24.3 … -8.9 | 100/100 |
| VVC Fast | HEVC Fast | -8.5% | -6.5% | -14.0% | -12.9 … -5.8 | 99/100 |
| VVC Fast | VVC | +20.0% | +23.8% | +12.3% | +13.3 … +24.6 | 99/100 |

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
