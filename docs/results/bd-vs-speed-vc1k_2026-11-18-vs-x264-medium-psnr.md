# Bitrate vs encode speed, VC-1K phone video calls (vc1k_2026-11-18), zero line x264-medium~psnr

<sub>Updated 2026-11-21, build 730e4e16 (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, AV1, AV1 Fast, VP9, VP9 Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, AV1 Veryfast, HEVC Faster, VP9 Faster, VP9 Veryfast, HEVC Slow, AV1 Slow, VVC Slow, H.264 Slow, VP9 Slow, AV2 Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264, H.264 Veryfast timed on 4fc9a1de (same streams)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 100 clips of `vc1k_2026-11-18`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-vc1k_2026-11-18-vs-x264-medium-psnr.svg)

**Rate control and the buffer.** Every encoder in the CBR charts is asked for a 1 s buffer at the target rate. x264, x265 and NVENC are held to it by a hard VBV, and SVT-AV1, libaom and libvpx by their 1000 ms real-time buffer. nanocodec 0.6.0 treats its buffer as a target with a 1.4x stop and sizes key frames more generously. In these charts its first key frame is about 0.33-0.45 of a buffer, against 0.10-0.24 for those references. Its buffer peak passes one buffer on 10-16 % of points (theirs: 0-1 %), and its key second runs at about 1.3-1.4x the target (theirs: 0.9-1.05x). openh264 is given no buffer setting and behaves like nanocodec in this respect. The looser buffer is worth quality. On a sample of the held-out clips (24 clips at 150 frames, 12 at 600 frames; AV1, HEVC and H.264 medium), holding nanocodec inside one buffer like the references cost it 10-13 % in bits at equal quality. At 150 frames that moved its lead over SVT-AV1 M12 from 31 % to 25 % (AV1) and from 33 % to 25 % (HEVC), and moved H.264 medium from level with SVT-AV1 M12 to about 10 % behind; against x264 medium the leads moved by 2-5 points. The Fixed QP charts have no rate control and are not affected. A later release will make the buffer a hard bound by default and re-measure these charts.

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-vc1k_2026-11-18-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **AV2 Slower** | -56.3% | -59.7% | -54.6% | -60.5 … -50.4 | n/a | n/a | 26.6 | 0.24× | 100/100 | Pareto x86 |
| **VVC Veryslow** | -55.5% | -57.2% | -52.9% | -61.5 … -49.8 | n/a | n/a | 30.8 | 0.27× | 100/100 | Pareto x86 |
| **VVC Slower** | -54.5% | -56.2% | -51.9% | -60.7 … -48.5 | n/a | n/a | 51.9 | 0.46× | 100/100 | Pareto x86 |
| **VVC Slow** | -54.2% | -56.0% | -51.3% | -61.4 … -48.0 | 16.0 | 0.60× | 73.0 | 0.65× | 100/100 | Pareto; Pareto x86 |
| **AV2 Slow** | -53.3% | -56.8% | -52.3% | -58.3 … -46.8 | 16.1 | 0.60× | 120.8 | 1.07× | 100/100 | Pareto; Pareto x86 |
| **AV1 Slower** | -52.9% | -56.5% | -48.6% | -58.2 … -47.4 | n/a | n/a | 59.5 | 0.53× | 100/100 |  |
| **VVC** | -52.0% | -53.9% | -49.4% | -59.0 … -45.8 | 29.8 | 1.12× | 101.7 | 0.90× | 100/100 | Pareto |
| **AV2** | -51.8% | -55.3% | -50.8% | -57.3 … -45.3 | 39.2 | 1.47× | 150.7 | 1.34× | 100/100 | Pareto; Pareto x86 |
| **AV1 Slow** | -50.5% | -53.7% | -47.5% | -57.1 … -44.2 | 26.6 | 1.00× | 115.3 | 1.02× | 100/100 |  |
| **HEVC Veryslow** | -49.7% | -51.6% | -47.1% | -56.0 … -43.1 | n/a | n/a | 36.9 | 0.33× | 100/100 |  |
| **AV2 Fast** | -49.2% | -51.4% | -48.6% | -55.8 … -42.6 | 51.8 | 1.94× | 172.7 | 1.53× | 100/100 | Pareto; Pareto x86 |
| **HEVC Slower** | -49.2% | -50.6% | -47.2% | -55.4 … -42.5 | n/a | n/a | 68.2 | 0.60× | 100/100 |  |
| **VP9 Slower** | -48.6% | -54.6% | -42.4% | -55.3 … -41.6 | n/a | n/a | 51.5 | 0.46× | 100/100 |  |
| **AV1** | -48.5% | -51.9% | -45.5% | -55.4 … -41.6 | 45.7 | 1.71× | 179.0 | 1.59× | 100/100 | Pareto x86 |
| **HEVC Slow** | -47.4% | -49.1% | -45.4% | -53.4 … -40.5 | 19.0 | 0.71× | 121.4 | 1.08× | 100/100 |  |
| **AV1 Fast** | -46.9% | -49.8% | -43.9% | -53.8 … -40.0 | 66.2 | 2.48× | 252.9 | 2.24× | 100/100 | Pareto; Pareto x86 |
| **VP9 Slow** | -46.8% | -52.8% | -40.6% | -53.3 … -39.8 | 26.7 | 1.00× | 91.8 | 0.81× | 100/100 |  |
| **VVC Fast** | -46.8% | -47.9% | -44.0% | -53.6 … -40.5 | 53.2 | 1.99× | 177.3 | 1.57× | 100/100 |  |
| SVT-AV1 M8 | -46.6% | -49.9% | -39.6% | -53.8 … -40.0 | 14.3 | 0.53× | 80.8 | 0.72× | 100/100 | 2 rate-invalid pts |
| **HEVC** | -45.1% | -46.6% | -43.5% | -51.4 … -38.0 | 45.3 | 1.70× | 167.9 | 1.49× | 100/100 |  |
| **VP9** | -44.5% | -50.5% | -38.4% | -52.7 … -37.0 | 42.1 | 1.58× | 145.6 | 1.29× | 100/100 |  |
| **AV1 Faster** | -43.3% | -45.8% | -40.7% | -52.0 … -35.3 | 79.4 | 2.98× | 313.5 | 2.78× | 100/100 | Pareto; Pareto x86 |
| **VVC Faster** | -42.5% | -43.5% | -42.2% | -50.8 … -35.9 | n/a | n/a | 219.8 | 1.95× | 100/100 |  |
| x265 medium | -42.4% | -42.4% | -37.7% | -49.0 … -36.0 | 8.0 | 0.30× | 38.0 | 0.34× | 100/100 | 2 rate-invalid pts |
| **AV2 Faster** | -41.8% | -42.5% | -39.3% | -50.7 … -33.1 | 62.9 | 2.36× | 265.7 | 2.36× | 100/100 |  |
| SVT-AV1 M10 | -41.6% | -45.1% | -35.5% | -48.5 … -34.7 | 39.3 | 1.47× | 228.9 | 2.03× | 100/100 | 2 rate-invalid pts |
| **VP9 Fast** | -41.4% | -47.4% | -34.8% | -49.4 … -34.5 | 55.8 | 2.09× | 247.1 | 2.19× | 100/100 |  |
| **HEVC Fast** | -40.3% | -40.8% | -36.6% | -48.1 … -32.3 | 72.0 | 2.70× | 281.8 | 2.50× | 100/100 |  |
| **AV1 Veryfast** | -38.5% | -40.8% | -33.1% | -45.7 … -28.9 | 96.5 | 3.62× | 419.5 | 3.72× | 100/100 | Pareto; Pareto x86 |
| x265 superfast | -38.0% | -38.0% | -34.1% | -45.9 … -31.6 | 12.1 | 0.45× | 57.9 | 0.51× | 100/100 |  |
| **VP9 Faster** | -37.6% | -39.9% | -30.8% | -44.5 … -28.8 | 92.3 | 3.46× | 311.9 | 2.77× | 100/100 |  |
| **HEVC Faster** | -36.5% | -36.8% | -32.9% | -44.2 … -28.8 | 90.3 | 3.38× | 350.2 | 3.11× | 100/100 |  |
| x265 ultrafast | -36.0% | -36.1% | -31.4% | -44.2 … -28.7 | 17.1 | 0.64× | 85.1 | 0.75× | 100/100 |  |
| SVT-AV1 M12 | -32.5% | -33.4% | -30.0% | -42.4 … -23.8 | 70.9 | 2.67× | 408.7 | 3.62× | 100/100 | 2 rate-invalid pts |
| SVT-AV1 M13 | -28.0% | -29.0% | -25.9% | -35.5 … -17.8 | 79.2 | 2.96× | 445.4 | 3.95× | 100/100 | Pareto x86; 2 rate-invalid pts |
| **VP9 Veryfast** | -26.5% | -30.0% | -23.0% | -33.3 … -17.7 | 95.8 | 3.59× | 393.8 | 3.49× | 100/100 |  |
| libaom rt 8 | -26.2% | -27.0% | -18.7% | -31.5 … -20.8 | 17.4 | 0.65× | 68.8 | 0.61× | 100/100 | 46 rate-invalid pts |
| libaom rt 10 | -24.8% | -26.9% | -17.7% | -28.8 … -19.2 | 22.3 | 0.83× | 90.5 | 0.80× | 100/100 | 40 rate-invalid pts |
| **H.264 Slow** | -3.1% | -9.0% | -0.4% | -14.8 … +7.1 | 37.2 | 1.39× | 160.7 | 1.43× | 100/100 |  |
| **H.264 Slower** | -3.1% | -8.9% | -1.0% | -15.6 … +6.8 | n/a | n/a | 113.3 | 1.00× | 100/100 |  |
| libvpx VP9 rt 7 | -1.3% | -1.4% | +3.7% | -8.0 … +2.7 | 30.0 | 1.12× | 122.7 | 1.09× | 100/100 | 66 rate-invalid pts |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 100/100 | 29 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +0.2% | +1.4% | +5.4% | -6.0 … +4.9 | 47.8 | 1.78× | 204.4 | 1.81× | 100/100 | 65 rate-invalid pts |
| **H.264** | +0.3% | -5.2% | +2.1% | -12.6 … +12.0 | 66.4 | 2.49× | 198.4 | 1.76× | 100/100 |  |
| **H.264 Fast** | +6.7% | +2.0% | +7.2% | -4.4 … +17.3 | 119.5 | 4.48× | 283.4 | 2.51× | 100/100 | Pareto; 1 rate-invalid pts |
| **H.264 Faster** | +10.3% | +4.7% | +12.4% | -2.5 … +20.6 | 146.9 | 5.51× | 369.8 | 3.28× | 100/100 | Pareto |
| x264 medium | +12.7% | +6.7% | +26.1% | +8.5 … +16.3 | 26.7 | 1.00× | 112.8 | 1.00× | 100/100 | 25 rate-invalid pts |
| **H.264 Veryfast** | +23.1% | +14.6% | +29.8% | +9.3 … +34.1 | 164.1 | 6.15× | 510.6 | 4.53× | 100/100 | Pareto; Pareto x86 |
| x264 veryfast | +32.8% | +22.6% | +46.6% | +23.2 … +43.7 | 85.4 | 3.18× | 363.3 | 3.22× | 100/100 |  |
| x264 ultrafast | +70.3% | +71.1% | +55.5% | +60.8 … +88.7 | 190.5 | 7.11× | 1017.0 | 9.02× | 100/100 | Pareto; Pareto x86 |
| openh264 | +83.7% | +77.0% | +106.3% | +68.8 … +98.7 | 111.5 | 4.17× | 524.7 | 4.65× | 100/100 | 143 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: vc1k_android_indoor_fastmotion_darklight_720x1280_30_3683766447, vc1k_android_outdoor_fastmotion_mediumlight_720x1280_30_3685120542, vc1k_android_outdoor_mediummotion_darklight_1280x720_25_3686179217, vc1k_android_outdoor_mediummotion_darklight_720x1280_30_3685120604, vc1k_ios_indoor_mediummotion_goodlight_1280x720_30_3685486349, vc1k_ios_indoor_mediummotion_goodlight_720x1280_30_3685120568, vc1k_android_indoor_mediummotion_lowlight_720x1280_30_3688492094, vc1k_android_indoor_mediummotion_lowlight_720x1280_30_3688853878, vc1k_ios_outdoor_mediummotion_lowlight_720x1280_30_3685586878, vc1k_ios_outdoor_mediummotion_lowlight_1280x720_30_3687663171, vc1k_android_outdoor_mediummotion_lowlight_720x1280_30_3684302390, vc1k_ios_indoor_mediummotion_mediumlight_720x1280_30_3685486318, vc1k_android_indoor_slowmotion_darklight_720x1280_30_3684302409, vc1k_android_indoor_slowmotion_darklight_1280x720_30_3688244638, vc1k_android_indoor_slowmotion_darklight_720x1280_25_3686179205, vc1k_android_indoor_slowmotion_darklight_1280x720_30_3689905294, vc1k_android_outdoor_slowmotion_darklight_1280x720_30_3689905293, vc1k_ios_indoor_slowmotion_goodlight_1280x720_30_3687390482, vc1k_ios_indoor_slowmotion_goodlight_720x1280_30_3683837796, vc1k_ios_outdoor_slowmotion_goodlight_1280x720_30_3687533665, vc1k_ios_indoor_slowmotion_lowlight_720x1280_30_3685486350, vc1k_android_indoor_slowmotion_lowlight_720x1280_30_3684329765, vc1k_android_indoor_slowmotion_lowlight_720x1280_30_3683570068, vc1k_android_indoor_slowmotion_lowlight_1280x720_30_3689905288, vc1k_android_outdoor_slowmotion_lowlight_1280x720_30_3685800581, vc1k_android_indoor_slowmotion_mediumlight_720x1280_30_3685591320, vc1k_ios_indoor_slowmotion_mediumlight_1280x720_30_3687663173, vc1k_ios_outdoor_slowmotion_mediumlight_720x1280_30_3683837800. Clips left out for every encoder: 0; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (videocall 100 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +12.7% | +12.7% |
| H.264 Slower | -3.1% | -3.1% |
| H.264 Slow | -3.1% | -3.1% |
| H.264 | +0.3% | +0.3% |
| H.264 Fast | +6.7% | +6.7% |
| H.264 Faster | +10.3% | +10.3% |
| H.264 Veryfast | +23.1% | +23.1% |
| HEVC Veryslow | -49.7% | -49.7% |
| HEVC Slower | -49.2% | -49.2% |
| HEVC Slow | -47.4% | -47.4% |
| HEVC | -45.1% | -45.1% |
| HEVC Fast | -40.3% | -40.3% |
| HEVC Faster | -36.5% | -36.5% |
| AV1 Slower | -52.9% | -52.9% |
| AV1 Slow | -50.5% | -50.5% |
| AV1 | -48.5% | -48.5% |
| AV1 Fast | -46.9% | -46.9% |
| AV1 Faster | -43.3% | -43.3% |
| AV1 Veryfast | -38.5% | -38.5% |
| VP9 Slower | -48.6% | -48.6% |
| VP9 Slow | -46.8% | -46.8% |
| VP9 | -44.5% | -44.5% |
| VP9 Fast | -41.4% | -41.4% |
| VP9 Faster | -37.6% | -37.6% |
| VP9 Veryfast | -26.5% | -26.5% |
| AV2 Slower | -56.3% | -56.3% |
| AV2 Slow | -53.3% | -53.3% |
| AV2 | -51.8% | -51.8% |
| AV2 Fast | -49.2% | -49.2% |
| AV2 Faster | -41.8% | -41.8% |
| VVC Veryslow | -55.5% | -55.5% |
| VVC Slower | -54.5% | -54.5% |
| VVC Slow | -54.2% | -54.2% |
| VVC | -52.0% | -52.0% |
| VVC Fast | -46.8% | -46.8% |
| VVC Faster | -42.5% | -42.5% |
| openh264 | +83.7% | +83.7% |
| x264 ultrafast | +70.3% | +70.3% |
| x264 veryfast | +32.8% | +32.8% |
| x265 ultrafast | -36.0% | -36.0% |
| x265 superfast | -38.0% | -38.0% |
| x265 medium | -42.4% | -42.4% |
| SVT-AV1 M8 | -46.6% | -46.6% |
| SVT-AV1 M10 | -41.6% | -41.6% |
| SVT-AV1 M12 | -32.5% | -32.5% |
| SVT-AV1 M13 | -28.0% | -28.0% |
| libvpx VP9 rt 7 | -1.3% | -1.3% |
| libvpx VP9 rt 9 | +0.2% | +0.2% |
| libaom rt 8 | -26.2% | -26.2% |
| libaom rt 10 | -24.8% | -24.8% |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **AV2 Slower** | -64.9% | -67.7% | -68.3% | -58.6% | 100/100 |
| **VVC Veryslow** | -64.4% | -65.5% | -67.5% | -60.3% | 100/100 |
| **VVC Slower** | -63.6% | -64.6% | -66.7% | -59.4% | 100/100 |
| **VVC Slow** | -63.3% | -64.4% | -66.3% | -59.3% | 100/100 |
| **AV2 Slow** | -62.4% | -65.3% | -66.7% | -55.3% | 100/100 |
| **AV1 Slower** | -61.7% | -64.5% | -63.3% | -57.3% | 100/100 |
| **VVC** | -61.4% | -62.6% | -64.8% | -56.9% | 100/100 |
| **AV2** | -61.2% | -63.9% | -65.7% | -53.9% | 100/100 |
| **AV1 Slow** | -59.7% | -62.2% | -62.7% | -54.3% | 100/100 |
| **HEVC Veryslow** | -59.3% | -60.3% | -63.0% | -54.5% | 100/100 |
| **AV2 Fast** | -59.1% | -60.7% | -64.1% | -52.4% | 100/100 |
| **HEVC Slower** | -58.7% | -59.3% | -63.0% | -53.7% | 100/100 |
| **VP9 Slower** | -58.5% | -62.8% | -59.3% | -53.2% | 100/100 |
| **AV1** | -58.1% | -60.6% | -61.3% | -52.3% | 100/100 |
| **HEVC Slow** | -57.3% | -58.1% | -61.7% | -52.1% | 100/100 |
| **VVC Fast** | -57.1% | -57.5% | -60.8% | -53.0% | 100/100 |
| **VP9 Slow** | -56.9% | -61.3% | -57.8% | -51.4% | 100/100 |
| **AV1 Fast** | -56.8% | -58.9% | -60.2% | -51.2% | 100/100 |
| SVT-AV1 M8 | -56.6% | -58.7% | -56.9% | -54.4% | 100/100 |
| **HEVC** | -55.2% | -55.8% | -60.2% | -49.8% | 100/100 |
| **VP9** | -54.9% | -59.3% | -56.2% | -49.1% | 100/100 |
| **AV1 Faster** | -53.6% | -55.1% | -58.1% | -47.7% | 100/100 |
| **VVC Faster** | -53.1% | -53.4% | -59.4% | -46.6% | 100/100 |
| **AV2 Faster** | -52.8% | -52.6% | -57.6% | -48.2% | 100/100 |
| x265 medium | -52.3% | -51.0% | -55.1% | -50.9% | 100/100 |
| SVT-AV1 M10 | -52.3% | -54.4% | -53.8% | -48.7% | 100/100 |
| **VP9 Fast** | -52.3% | -56.6% | -53.4% | -46.7% | 100/100 |
| **HEVC Fast** | -50.9% | -50.2% | -54.7% | -47.7% | 100/100 |
| **AV1 Veryfast** | -49.5% | -49.9% | -52.4% | -46.2% | 100/100 |
| **VP9 Faster** | -48.9% | -49.5% | -50.6% | -46.6% | 100/100 |
| x265 superfast | -48.4% | -46.8% | -52.1% | -46.2% | 100/100 |
| **HEVC Faster** | -47.1% | -45.7% | -51.7% | -44.0% | 100/100 |
| x265 ultrafast | -46.6% | -44.8% | -50.1% | -44.9% | 100/100 |
| SVT-AV1 M12 | -44.0% | -43.4% | -49.6% | -39.1% | 100/100 |
| SVT-AV1 M13 | -39.9% | -39.0% | -46.4% | -34.4% | 100/100 |
| **VP9 Veryfast** | -37.5% | -38.5% | -43.5% | -30.4% | 100/100 |
| libaom rt 8 | -37.5% | -35.3% | -39.9% | -37.1% | 100/100 |
| libaom rt 10 | -36.5% | -35.8% | -39.5% | -34.1% | 100/100 |
| **H.264 Slow** | -14.0% | -16.0% | -21.7% | -4.2% | 100/100 |
| **H.264 Slower** | -13.8% | -15.9% | -22.2% | -3.4% | 100/100 |
| libvpx VP9 rt 7 | -13.8% | -10.0% | -20.6% | -10.7% | 100/100 |
| libvpx VP9 rt 9 | -13.1% | -7.9% | -20.1% | -11.3% | 100/100 |
| **H.264** | -10.7% | -12.1% | -19.5% | -0.4% | 100/100 |
| x264 medium, metric-tuned (--tune psnr) | -10.0% | -5.9% | -19.8% | -4.3% | 100/100 |
| **H.264 Fast** | -4.2% | -4.5% | -15.0% | +7.0% | 100/100 |
| **H.264 Faster** | -0.9% | -1.7% | -11.0% | +9.9% | 100/100 |
| **H.264 Veryfast** | +10.9% | +8.9% | +2.8% | +21.0% | 100/100 |
| x264 veryfast | +18.5% | +15.8% | +16.3% | +23.6% | 100/100 |
| x264 ultrafast | +56.0% | +65.4% | +24.1% | +78.6% | 100/100 |
| openh264 | +61.0% | +62.5% | +61.2% | +59.2% | 100/100 |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slower | openh264 | -45.0% | -46.3% | -50.8% | -49.7 … -41.0 | 100/100 |
| H.264 Slow | openh264 | -45.2% | -46.5% | -50.6% | -49.9 … -41.2 | 100/100 |
| H.264 | openh264 | -43.0% | -43.9% | -49.3% | -48.0 … -38.5 | 100/100 |
| H.264 | x264 veryfast | -27.2% | -27.5% | -32.3% | -33.8 … -21.0 | 100/100 |
| H.264 | x264 medium | -10.7% | -12.1% | -19.5% | -22.7 … +1.7 | 100/100 |
| H.264 Fast | openh264 | -39.2% | -38.7% | -46.6% | -44.4 … -34.3 | 99/100 |
| H.264 Fast | x264 veryfast | -21.5% | -20.8% | -27.7% | -27.3 … -15.7 | 100/100 |
| H.264 Fast | H.264 | +8.9% | +10.3% | +6.1% | +5.1 … +9.8 | 100/100 |
| H.264 Faster | openh264 | -36.6% | -36.5% | -44.0% | -42.0 … -31.5 | 99/100 |
| H.264 Faster | x264 ultrafast | -38.5% | -45.4% | -29.9% | -44.4 … -33.0 | 98/100 |
| H.264 Faster | H.264 Fast | +3.5% | +2.8% | +5.0% | +1.2 … +5.7 | 100/100 |
| H.264 Veryfast | openh264 | -28.1% | -28.5% | -34.8% | -33.6 … -23.5 | 99/100 |
| H.264 Veryfast | x264 ultrafast | -31.5% | -40.6% | -19.1% | -37.3 … -26.4 | 100/100 |
| H.264 Veryfast | H.264 Faster | +14.7% | +12.8% | +16.6% | +8.1 … +18.1 | 100/100 |
| HEVC Veryslow | SVT-AV1 M12 | -30.3% | -33.2% | -29.0% | -37.5 … -21.8 | 99/100 |
| HEVC Slower | SVT-AV1 M12 | -29.5% | -31.6% | -29.5% | -35.9 … -22.1 | 99/100 |
| HEVC Slow | SVT-AV1 M12 | -26.3% | -28.8% | -26.4% | -34.0 … -18.9 | 99/100 |
| HEVC | SVT-AV1 M12 | -22.4% | -24.5% | -23.2% | -29.9 … -14.2 | 99/100 |
| HEVC | x265 medium | -7.2% | -10.8% | -12.5% | -14.3 … +2.9 | 100/100 |
| HEVC Fast | SVT-AV1 M12 | -11.4% | -11.1% | -9.3% | -20.4 … -4.4 | 99/100 |
| HEVC Fast | x265 superfast | -4.9% | -6.3% | -5.4% | -19.1 … +5.0 | 100/100 |
| HEVC Fast | HEVC | +13.9% | +17.5% | +17.5% | +11.4 … +15.4 | 100/100 |
| HEVC Faster | SVT-AV1 M12 | -2.7% | -0.9% | -2.6% | -10.4 … +2.4 | 99/100 |
| HEVC Faster | SVT-AV1 M13 | -11.0% | -9.7% | -9.7% | -18.4 … -5.2 | 99/100 |
| HEVC Faster | HEVC Fast | +10.8% | +12.4% | +8.9% | +7.0 … +14.3 | 100/100 |
| AV1 Slower | SVT-AV1 M12 | -35.4% | -42.3% | -29.2% | -41.5 … -31.3 | 99/100 |
| AV1 Slow | SVT-AV1 M12 | -31.7% | -37.6% | -28.5% | -37.9 … -27.3 | 99/100 |
| AV1 | SVT-AV1 M12 | -28.7% | -35.0% | -25.4% | -34.7 … -24.7 | 99/100 |
| AV1 | openh264 | -72.5% | -73.8% | -73.7% | -76.2 … -69.7 | 82/100 |
| AV1 | SVT-AV1 M8 | -0.9% | -1.5% | -9.9% | -10.4 … +8.6 | 99/100 |
| AV1 | libaom rt 8 | -30.1% | -32.4% | -35.5% | -35.9 … -26.0 | 100/100 |
| AV1 Fast | SVT-AV1 M12 | -26.5% | -31.8% | -23.8% | -31.1 … -22.7 | 99/100 |
| AV1 Fast | openh264 | -71.7% | -73.0% | -72.9% | -76.2 … -68.8 | 88/100 |
| AV1 Fast | AV1 | +3.9% | +5.3% | +3.8% | +1.6 … +5.6 | 100/100 |
| AV1 Faster | SVT-AV1 M12 | -19.9% | -24.0% | -18.9% | -26.8 … -12.7 | 99/100 |
| AV1 Faster | AV1 Fast | +9.6% | +12.0% | +7.1% | +5.1 … +14.6 | 100/100 |
| AV1 Veryfast | SVT-AV1 M12 | -12.2% | -15.1% | -6.5% | -22.1 … -3.9 | 99/100 |
| AV1 Veryfast | SVT-AV1 M13 | -20.3% | -23.6% | -13.4% | -29.1 … -12.1 | 99/100 |
| AV1 Veryfast | AV1 Faster | +11.6% | +14.4% | +17.0% | +6.2 … +16.6 | 100/100 |
| VP9 Slower | HEVC Slower | +0.7% | -10.4% | +11.8% | -2.2 … +3.2 | 100/100 |
| VP9 Slow | HEVC Slow | +1.4% | -9.2% | +12.0% | -2.1 … +3.9 | 100/100 |
| VP9 | HEVC | +1.3% | -9.3% | +12.1% | -2.4 … +4.3 | 100/100 |
| VP9 | libvpx VP9 rt 7 | -45.7% | -50.9% | -42.2% | -52.2 … -40.1 | 99/100 |
| VP9 Fast | HEVC Fast | -3.5% | -15.8% | +3.4% | -8.4 … +0.8 | 100/100 |
| VP9 Fast | libvpx VP9 rt 9 | -44.9% | -51.3% | -40.7% | -50.8 … -38.6 | 99/100 |
| VP9 Fast | VP9 | +8.2% | +8.9% | +8.3% | +4.5 … +10.0 | 100/100 |
| VP9 Faster | HEVC Faster | -5.5% | -9.7% | +1.8% | -7.8 … -0.7 | 100/100 |
| VP9 Faster | libvpx VP9 rt 9 | -40.9% | -43.7% | -36.8% | -47.1 … -32.9 | 99/100 |
| VP9 Faster | VP9 Fast | +9.5% | +20.6% | +7.3% | +6.7 … +13.2 | 100/100 |
| VP9 Veryfast | HEVC Faster | +25.5% | +16.8% | +21.4% | +17.2 … +32.1 | 99/100 |
| VP9 Veryfast | libvpx VP9 rt 9 | -26.3% | -31.4% | -27.2% | -32.8 … -18.8 | 100/100 |
| VP9 Veryfast | VP9 Faster | +37.7% | +33.1% | +21.6% | +23.9 … +43.9 | 98/100 |
| AV2 Slower | AV1 Slower | -11.0% | -12.4% | -15.8% | -13.8 … -6.8 | 100/100 |
| AV2 Slow | AV1 Slow | -9.6% | -11.9% | -12.7% | -13.6 … -5.5 | 100/100 |
| AV2 | AV1 | -10.3% | -12.0% | -13.1% | -12.4 … -6.7 | 100/100 |
| AV2 | SVT-AV1 M8 | -9.6% | -11.5% | -22.5% | -20.5 … -3.6 | 100/100 |
| AV2 Fast | AV1 Fast | -8.3% | -7.5% | -12.2% | -12.9 … -2.9 | 100/100 |
| AV2 Fast | AV2 | +6.3% | +10.7% | +4.8% | +2.7 … +9.0 | 100/100 |
| AV2 Faster | AV1 Faster | -0.6% | +3.4% | -0.3% | -8.6 … +6.3 | 100/100 |
| AV2 Faster | SVT-AV1 M12 | -21.2% | -21.4% | -19.4% | -30.2 … -11.2 | 99/100 |
| AV2 Faster | AV2 Fast | +20.5% | +26.3% | +23.6% | +10.4 … +28.9 | 100/100 |
| VVC Veryslow | HEVC Veryslow | -15.4% | -15.5% | -14.3% | -18.5 … -12.3 | 100/100 |
| VVC Slower | HEVC Slower | -14.3% | -15.3% | -11.7% | -17.4 … -11.7 | 100/100 |
| VVC Slow | HEVC Slow | -17.0% | -17.9% | -14.0% | -20.4 … -13.7 | 100/100 |
| VVC | HEVC | -16.3% | -17.9% | -13.0% | -20.4 … -13.0 | 100/100 |
| VVC | x265 medium | -21.4% | -25.8% | -23.7% | -26.2 … -15.8 | 100/100 |
| VVC Fast | HEVC Fast | -16.0% | -18.4% | -15.8% | -21.0 … -13.2 | 100/100 |
| VVC Fast | VVC | +14.2% | +17.6% | +13.9% | +9.8 … +14.6 | 100/100 |
| VVC Faster | HEVC Faster | -17.0% | -18.3% | -19.2% | -20.9 … -13.6 | 99/100 |
| VVC Faster | VVC Fast | +11.6% | +13.6% | +4.7% | +6.3 … +15.8 | 99/100 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 730e4e16 (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, AV1, AV1 Fast, VP9, VP9 Fast, AV2, AV2 Fast, VVC, VVC Fast, AV1 Faster, AV1 Veryfast, HEVC Faster, VP9 Faster, VP9 Veryfast, HEVC Slow, AV1 Slow, VVC Slow, H.264 Slow, VP9 Slow, AV2 Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264, H.264 Veryfast timed on 4fc9a1de (same streams), its presets), openh264
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
