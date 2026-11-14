# Bitrate vs encode speed, held-out clips (independent_fresh_2026-11-13), zero line x264-medium~psnr

<sub>Updated 2026-11-14, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 138 clips of `independent_fresh_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC** | -44.7% | -44.9% | -44.2% | -52.4 … -38.4 | 27.9 | 1.05× | 71.8 | 1.06× | 135/135 | Pareto; Pareto x86 |
| **AV2** | -42.8% | -46.1% | -41.3% | -50.4 … -36.5 | 31.5 | 1.18× | 85.5 | 1.27× | 135/135 | Pareto; Pareto x86 |
| **HEVC** | -40.2% | -40.3% | -39.0% | -47.4 … -33.3 | 51.0 | 1.91× | 121.0 | 1.79× | 135/135 | Pareto; Pareto x86 |
| **AV1** | -38.5% | -42.3% | -34.5% | -45.5 … -34.2 | 44.1 | 1.65× | 107.0 | 1.58× | 135/135 | 1 rate-invalid pts |
| **VP9** | -38.3% | -40.4% | -34.6% | -46.5 … -31.1 | 39.2 | 1.47× | 100.0 | 1.48× | 135/135 | 2 rate-invalid pts |
| **VVC Fast** | -37.3% | -36.1% | -35.5% | -45.5 … -28.6 | 47.3 | 1.77× | 115.4 | 1.71× | 135/135 |  |
| **AV2 Fast** | -36.2% | -37.5% | -33.4% | -43.7 … -27.6 | 44.1 | 1.65× | 125.4 | 1.86× | 135/135 | Pareto x86 |
| SVT-AV1 M8 | -34.4% | -36.5% | -26.3% | -38.9 … -25.4 | 14.3 | 0.53× | 45.2 | 0.62× | 135/135 | 22 rate-invalid pts |
| **AV1 Fast** | -32.6% | -32.9% | -29.8% | -41.5 … -26.0 | 66.7 | 2.50× | 160.5 | 2.38× | 135/135 | Pareto; Pareto x86 |
| **VP9 Fast** | -32.1% | -33.2% | -28.9% | -40.7 … -23.3 | 47.2 | 1.77× | 117.0 | 1.73× | 135/135 | 1 rate-invalid pts |
| **HEVC Fast** | -32.1% | -32.3% | -30.0% | -40.0 … -25.6 | 78.5 | 2.94× | 175.8 | 2.60× | 135/135 | Pareto; Pareto x86 |
| **VP9 Faster** | -27.7% | -27.1% | -25.1% | -38.5 … -17.6 | 78.4 | 2.94× | 180.9 | 2.68× | 135/135 | Pareto x86; 1 rate-invalid pts |
| SVT-AV1 M10 | -27.0% | -27.8% | -20.3% | -32.0 … -18.2 | 39.3 | 1.47× | 106.6 | 1.58× | 135/135 | 20 rate-invalid pts |
| **AV1 Faster** | -26.6% | -26.8% | -23.0% | -36.4 … -19.4 | 80.5 | 3.02× | 174.9 | 2.59× | 135/135 | Pareto |
| **HEVC Faster** | -25.9% | -24.8% | -22.0% | -35.6 … -17.1 | 98.0 | 3.67× | 206.4 | 3.05× | 135/135 | Pareto; Pareto x86 |
| x265 medium, metric-tuned (--tune psnr) | -25.5% | -26.2% | -20.3% | -31.7 … -18.2 | 8.0 | 0.30× | 25.6 | 0.37× | 135/135 | 28 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 superfast, metric-tuned (--tune psnr) | -23.0% | -22.4% | -17.7% | -28.5 … -16.3 | 12.1 | 0.45× | 38.8 | 0.54× | 135/135 | 26 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 superfast | -21.4% | -19.3% | -16.7% | -25.7 … -13.7 | 12.1 | 0.45× | 38.8 | 0.54× | 135/135 | 27 rate-invalid pts |
| x265 medium | -21.1% | -19.2% | -13.6% | -29.0 … -14.0 | 8.0 | 0.30× | 25.6 | 0.37× | 135/135 | 30 rate-invalid pts |
| **VP9 Fastest** | -20.2% | -19.6% | -19.9% | -35.3 … -11.1 | 88.0 | 3.30× | 192.4 | 2.85× | 135/135 |  |
| SVT-AV1 M12 | -19.9% | -19.7% | -14.8% | -24.3 … -8.7 | 70.9 | 2.67× | 182.1 | 2.69× | 135/135 | 17 rate-invalid pts |
| **H.264** | -19.0% | -21.8% | -16.6% | -32.3 … -11.0 | 74.4 | 2.79× | 175.9 | 2.60× | 135/135 |  |
| SVT-AV1 M13 | -17.6% | -17.1% | -12.9% | -23.2 … -6.5 | 79.2 | 2.96× | 193.1 | 2.86× | 135/135 | 17 rate-invalid pts |
| libaom rt 8 | -17.2% | -15.0% | -11.6% | -23.9 … -10.0 | 17.4 | 0.65× | 42.1 | 0.61× | 135/135 | 76 rate-invalid pts |
| libaom rt 10 | -16.6% | -16.5% | -9.5% | -23.1 … -8.5 | 22.3 | 0.83× | 49.2 | 0.74× | 135/135 | 80 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -15.3% | -13.7% | -9.9% | -22.8 … -5.9 | 17.1 | 0.64× | 53.5 | 0.78× | 135/135 | 26 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -14.5% | -11.6% | -9.6% | -21.2 … -5.1 | 17.1 | 0.64× | 53.5 | 0.78× | 135/135 | 26 rate-invalid pts |
| **H.264 Fast** | -11.1% | -14.3% | -9.7% | -25.0 … +0.9 | 113.5 | 4.25× | 230.2 | 3.41× | 135/135 | Pareto; Pareto x86 |
| **H.264 Faster** | -3.2% | -7.9% | +0.4% | -21.4 … +14.4 | 133.7 | 5.01× | 265.7 | 3.93× | 135/135 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 67.6 | 1.00× | 135/135 | 36 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Fastest** | +4.7% | -0.7% | +7.8% | -15.3 … +24.7 | 153.5 | 5.76× | 311.8 | 4.61× | 135/135 | Pareto; Pareto x86 |
| x264 veryfast, metric-tuned (--tune psnr) | +9.7% | +6.3% | +8.9% | +4.3 … +19.0 | 85.4 | 3.18× | 206.4 | 3.05× | 135/135 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 medium, metric-tuned (--tune ssim) | +11.3% | +9.1% | +27.3% | +7.0 … +14.9 | 26.7 | 1.00× | 67.6 | 1.00× | 135/135 | 32 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +11.4% | +18.2% | +16.2% | +5.5 … +18.4 | 47.8 | 1.78× | 113.0 | 1.53× | 135/135 | 141 rate-invalid pts |
| x264 veryfast | +13.4% | +7.8% | +29.1% | +4.8 … +23.6 | 85.4 | 3.18× | 206.4 | 3.05× | 135/135 | 1 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +13.8% | +18.5% | +23.0% | +4.0 … +19.1 | 30.0 | 1.12× | 68.1 | 1.06× | 135/135 | 113 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 medium | +13.8% | +11.8% | +27.8% | +8.3 … +16.5 | 26.7 | 1.00× | 67.6 | 1.00× | 135/135 | 33 rate-invalid pts |
| libvpx VP9 rt 7 | +13.9% | +18.4% | +18.0% | +4.6 … +21.7 | 30.0 | 1.12× | 68.1 | 1.06× | 135/135 | 133 rate-invalid pts |
| openh264 | +31.7% | +28.1% | +42.3% | +0.0 … +63.3 | 111.5 | 4.17× | 260.3 | 3.85× | 135/135 | 225 rate-invalid pts |
| x264 ultrafast | +48.5% | +56.9% | +38.0% | +28.4 … +69.7 | 190.5 | 7.11× | 547.1 | 8.09× | 135/135 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 3; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 12, crowd 38, lowlight 8, pan 42, static 35 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +13.8% | +12.8% |
| H.264 | -19.0% | -20.4% |
| H.264 Fast | -11.1% | -12.0% |
| H.264 Faster | -3.2% | -3.5% |
| H.264 Fastest | +4.7% | +4.6% |
| HEVC | -40.2% | -40.6% |
| HEVC Fast | -32.1% | -33.0% |
| HEVC Faster | -25.9% | -26.0% |
| AV1 | -38.5% | -40.0% |
| AV1 Fast | -32.6% | -33.9% |
| AV1 Faster | -26.6% | -27.7% |
| VP9 | -38.3% | -38.4% |
| VP9 Fast | -32.1% | -32.2% |
| VP9 Faster | -27.7% | -27.6% |
| VP9 Fastest | -20.2% | -20.8% |
| AV2 | -42.8% | -43.5% |
| AV2 Fast | -36.2% | -36.0% |
| VVC | -44.7% | -45.2% |
| VVC Fast | -37.3% | -37.0% |
| openh264 | +31.7% | +29.8% |
| x264 ultrafast | +48.5% | +54.3% |
| x264 veryfast | +13.4% | +14.3% |
| x265 ultrafast | -14.5% | -13.1% |
| x265 superfast | -21.4% | -20.4% |
| x265 medium | -21.1% | -20.9% |
| SVT-AV1 M8 | -34.4% | -32.7% |
| SVT-AV1 M10 | -27.0% | -25.6% |
| SVT-AV1 M12 | -19.9% | -18.1% |
| SVT-AV1 M13 | -17.6% | -16.1% |
| libvpx VP9 rt 7 | +13.9% | +15.9% |
| libvpx VP9 rt 9 | +11.4% | +12.1% |
| libaom rt 8 | -17.2% | -16.2% |
| libaom rt 10 | -16.6% | -15.8% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +31.7% | +28.1% | +42.3% | +24.8% | 135/135 |  |
| openh264, adaptive quantisation on | +31.7% | +28.1% | +42.3% | +24.8% | 135/135 | +0.0 [+0.0, +0.0] not distinguishable |
| x264 ultrafast | +48.5% | +56.9% | +38.0% | +50.5% | 135/135 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +48.4% | +56.7% | +38.1% | +50.5% | 135/135 | +0.0 [-0.2, +0.2] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +47.3% | +54.2% | +59.9% | +27.9% | 135/135 | +0.4 [-1.1, +1.8] not distinguishable |
| x264 veryfast | +13.4% | +7.8% | +29.1% | +3.4% | 135/135 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +9.7% | +6.3% | +8.9% | +13.9% | 135/135 | -2.1 [-3.2, -1.0] better |
| x264 veryfast, metric-tuned (--tune ssim) | +14.6% | +9.6% | +31.8% | +2.3% | 135/135 | +0.7 [+0.1, +1.2] worse |
| x264 medium | +13.8% | +11.8% | +27.8% | +1.9% | 135/135 |  |
| x264 medium, metric-tuned (--tune ssim) ● | +11.3% | +9.1% | +27.3% | -2.5% | 135/135 | -2.2 [-3.0, -1.4] better |
| x265 ultrafast | -14.5% | -11.6% | -9.6% | -22.3% | 135/135 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -15.3% | -13.7% | -9.9% | -22.2% | 135/135 | -1.1 [-1.4, -0.8] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -15.3% | -13.7% | -9.9% | -22.2% | 135/135 | -1.1 [-1.4, -0.8] better |
| x265 superfast | -21.4% | -19.3% | -16.7% | -28.4% | 135/135 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -23.0% | -22.4% | -17.7% | -29.0% | 135/135 | -2.1 [-2.5, -1.8] better |
| x265 superfast, metric-tuned (--tune ssim) | -23.0% | -22.4% | -17.7% | -29.0% | 135/135 | -2.1 [-2.5, -1.8] better |
| x265 medium | -21.1% | -19.2% | -13.6% | -30.5% | 135/135 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -25.5% | -26.2% | -20.3% | -30.1% | 135/135 | -5.7 [-6.3, -5.1] better |
| x265 medium, metric-tuned (--tune ssim) | -22.7% | -22.2% | -14.6% | -31.2% | 135/135 | -2.0 [-2.3, -1.7] better |
| libvpx VP9 rt 7 | +13.9% | +18.4% | +18.0% | +5.1% | 135/135 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +13.8% | +18.5% | +23.0% | -0.1% | 135/135 | -0.8 [-1.3, -0.3] better |
| libvpx VP9 rt 9 | +11.4% | +18.2% | +16.2% | -0.1% | 135/135 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.4% | +18.6% | +17.1% | -1.3% | 135/135 | -0.3 [-0.7, +0.2] not distinguishable |
| libaom rt 8 | -17.2% | -15.0% | -11.6% | -25.0% | 135/135 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -16.7% | -13.5% | -6.1% | -30.5% | 135/135 | +0.9 [+0.3, +1.7] worse |
| libaom rt 10 | -16.6% | -16.5% | -9.5% | -23.8% | 135/135 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -16.4% | -14.9% | -5.4% | -29.1% | 135/135 | +0.7 [+0.1, +1.4] worse |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC** | -51.8% | -53.8% | -57.1% | -44.5% | 135/135 |
| **AV2** | -50.1% | -54.9% | -54.7% | -40.7% | 135/135 |
| **HEVC** | -47.7% | -49.4% | -53.0% | -40.8% | 135/135 |
| **VP9** | -46.1% | -49.4% | -49.5% | -39.4% | 135/135 |
| **AV1** | -46.1% | -50.8% | -49.2% | -38.2% | 135/135 |
| **VVC Fast** | -45.5% | -46.1% | -50.5% | -40.0% | 135/135 |
| **AV2 Fast** | -44.4% | -47.2% | -48.6% | -37.4% | 135/135 |
| SVT-AV1 M8 | -43.0% | -46.4% | -42.6% | -40.0% | 135/135 |
| **AV1 Fast** | -41.0% | -42.7% | -45.6% | -34.8% | 135/135 |
| **VP9 Fast** | -40.8% | -43.2% | -45.1% | -34.2% | 135/135 |
| **HEVC Fast** | -40.8% | -42.3% | -46.1% | -34.0% | 135/135 |
| **VP9 Faster** | -36.9% | -37.7% | -42.2% | -30.8% | 135/135 |
| SVT-AV1 M10 | -36.7% | -39.1% | -38.4% | -32.6% | 135/135 |
| **AV1 Faster** | -35.7% | -36.8% | -40.2% | -30.0% | 135/135 |
| **HEVC Faster** | -35.4% | -35.3% | -39.9% | -31.1% | 135/135 |
| x265 medium, metric-tuned (--tune psnr) | -34.7% | -36.3% | -37.9% | -30.0% | 135/135 |
| x265 superfast, metric-tuned (--tune psnr) | -32.8% | -33.4% | -36.0% | -28.9% | 135/135 |
| x265 superfast, metric-tuned (--tune ssim) | -32.8% | -33.4% | -36.0% | -28.9% | 135/135 |
| x265 medium, metric-tuned (--tune ssim) | -32.5% | -32.9% | -33.6% | -31.1% | 135/135 |
| x265 superfast | -31.5% | -30.7% | -35.3% | -28.3% | 135/135 |
| x265 medium | -31.2% | -30.4% | -32.9% | -30.4% | 135/135 |
| SVT-AV1 M12 | -30.2% | -31.5% | -33.9% | -25.2% | 135/135 |
| **VP9 Fastest** | -30.2% | -30.5% | -38.2% | -21.8% | 135/135 |
| **H.264** | -28.9% | -32.4% | -35.1% | -19.1% | 135/135 |
| SVT-AV1 M13 | -28.2% | -29.1% | -32.3% | -23.1% | 135/135 |
| libaom rt 8, metric-tuned (--tune ssim) | -27.8% | -25.4% | -27.3% | -30.7% | 135/135 |
| libaom rt 8 | -27.8% | -26.7% | -31.7% | -25.2% | 134/135 |
| libaom rt 10, metric-tuned (--tune ssim) | -27.7% | -27.1% | -26.8% | -29.2% | 135/135 |
| libaom rt 10 | -27.7% | -28.6% | -30.4% | -24.0% | 135/135 |
| x265 ultrafast, metric-tuned (--tune psnr) | -26.0% | -25.6% | -30.1% | -22.5% | 135/135 |
| x265 ultrafast, metric-tuned (--tune ssim) | -26.0% | -25.6% | -30.1% | -22.5% | 135/135 |
| x265 ultrafast | -25.4% | -23.7% | -29.9% | -22.6% | 135/135 |
| **H.264 Fast** | -21.8% | -25.3% | -29.6% | -10.5% | 135/135 |
| **H.264 Faster** | -14.7% | -19.2% | -21.5% | -3.5% | 135/135 |
| x264 medium, metric-tuned (--tune psnr) | -10.7% | -10.0% | -20.6% | -1.5% | 135/135 |
| **H.264 Fastest** | -7.6% | -12.3% | -15.4% | +5.0% | 135/135 |
| libvpx VP9 rt 9 | -2.9% | +2.7% | -10.4% | -1.1% | 135/135 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -2.9% | +3.2% | -9.6% | -2.3% | 135/135 |
| x264 veryfast, metric-tuned (--tune psnr) | -2.4% | -4.8% | -14.6% | +12.0% | 135/135 |
| x264 medium, metric-tuned (--tune ssim) | -2.2% | -2.0% | -0.5% | -4.1% | 135/135 |
| x264 veryfast | -0.6% | -3.6% | -0.2% | +2.0% | 135/135 |
| libvpx VP9 rt 7 | -0.0% | +4.2% | -7.9% | +3.6% | 135/135 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +0.1% | +4.7% | -2.8% | -1.7% | 135/135 |
| x264 veryfast, metric-tuned (--tune ssim) | +0.4% | -1.9% | +2.1% | +1.0% | 135/135 |
| openh264 | +14.4% | +10.3% | +10.0% | +22.9% | 135/135 |
| openh264, adaptive quantisation on | +14.4% | +10.3% | +10.0% | +22.9% | 135/135 |
| x264 ultrafast, metric-tuned (--tune ssim) | +28.6% | +38.7% | +21.9% | +25.2% | 135/135 |
| x264 ultrafast, metric-tuned (--tune psnr) | +31.7% | +41.7% | +6.7% | +46.8% | 135/135 |
| x264 ultrafast | +31.7% | +41.9% | +6.6% | +46.7% | 135/135 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium, metric-tuned (--tune psnr)'s (even with its extra rungs, which the 50% source cap limits).

- **ind_bwc_videotext_video190** (bodycam): no reliable overlap for libaom rt 8, libaom rt 10.
- **ind_personpath22_uid_vid_00190** (static): no reliable overlap for H.264, HEVC, AV1 Fast, VP9, VVC, VVC Fast.
- **ind_personpath22_uid_vid_00212** (crowd): no reliable overlap for HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, VP9, VP9 Fast, VP9 Faster, VP9 Fastest, AV2, AV2 Fast, VVC, VVC Fast, SVT-AV1 M8, SVT-AV1 M10.

Direct BD-J vs x264 medium, metric-tuned (--tune psnr) on them, where it exists:

| Encoder | ind_bwc_videotext_video190 | ind_personpath22_uid_vid_00190 | ind_personpath22_uid_vid_00212 |
|---|---:|---:|---:|
| VVC | -28.3% | n/a | n/a |
| AV2 | -23.7% | -42.8% | n/a |
| HEVC | -20.3% | n/a | n/a |
| AV1 | -18.5% | -40.3% | n/a |
| VP9 | -23.8% | n/a | n/a |
| VVC Fast | -21.1% | n/a | n/a |
| AV2 Fast | -19.4% | -30.8% | n/a |
| SVT-AV1 M8 | -23.3% | -27.7% | n/a |
| AV1 Fast | -15.1% | n/a | n/a |
| VP9 Fast | -18.3% | -32.4% | n/a |
| HEVC Fast | -13.4% | -30.3% | n/a |
| VP9 Faster | -17.8% | -31.9% | n/a |
| SVT-AV1 M10 | +5.7% | -19.7% | n/a |
| AV1 Faster | -16.2% | -28.5% | n/a |
| HEVC Faster | -10.2% | -24.7% | n/a |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 superfast | +10.6% | +2.4% | -9.9% |
| x265 medium | +0.8% | +4.4% | -17.4% |
| VP9 Fastest | -16.5% | -30.0% | n/a |
| SVT-AV1 M12 | +8.8% | -16.3% | -11.9% |
| H.264 | -17.3% | n/a | -31.4% |
| SVT-AV1 M13 | +10.1% | -14.4% | -11.3% |
| libaom rt 8 | n/a | +0.2% | -16.9% |
| libaom rt 10 | n/a | +4.6% | -16.6% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 ultrafast | +20.2% | +18.6% | +5.7% |
| H.264 Fast | -8.0% | -29.0% | -24.3% |
| H.264 Faster | -6.4% | -27.1% | -16.6% |
| H.264 Fastest | -0.7% | -22.8% | -10.9% |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x264 medium, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9 | +8.6% | +24.3% | +19.5% |
| x264 veryfast | +24.1% | +0.3% | -9.1% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| x264 medium | +2.6% | -5.5% | -13.2% |
| libvpx VP9 rt 7 | -10.5% | +24.1% | +28.3% |
| openh264 | +26.1% | -30.1% | +7.2% |
| x264 ultrafast | +64.9% | +74.1% | +58.7% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 | openh264 | -32.2% | -33.0% | -35.8% | -44.5 … -23.2 | 138/138 |
| H.264 | x264 veryfast | -33.4% | -35.2% | -39.8% | -41.0 … -27.2 | 138/138 |
| H.264 | x264 medium | -28.7% | -32.1% | -34.7% | -39.2 … -18.6 | 138/138 |
| H.264 Fast | x264 veryfast | -26.5% | -28.3% | -34.5% | -34.6 … -19.5 | 138/138 |
| H.264 Fast | openh264 | -23.9% | -25.1% | -29.3% | -36.9 … -13.9 | 138/138 |
| H.264 Fast | H.264 | +11.9% | +11.8% | +9.8% | +9.3 … +15.2 | 138/138 |
| H.264 Faster | x264 ultrafast | -40.6% | -49.8% | -32.2% | -49.1 … -35.5 | 127/138 |
| H.264 Faster | H.264 Fast | +10.8% | +8.8% | +13.0% | +5.2 … +15.9 | 138/138 |
| H.264 Fastest | x264 ultrafast | -34.5% | -44.8% | -25.8% | -45.5 … -28.9 | 128/138 |
| H.264 Fastest | H.264 Faster | +11.0% | +10.6% | +9.0% | +8.6 … +13.6 | 138/138 |
| HEVC | x265 medium | -24.3% | -27.6% | -30.3% | -34.9 … -16.5 | 138/138 |
| HEVC Fast | x265 superfast | -13.5% | -17.1% | -16.4% | -27.1 … -7.8 | 138/138 |
| HEVC | SVT-AV1 M12 | -26.2% | -27.2% | -29.5% | -35.4 … -25.0 | 138/138 |
| HEVC Fast | HEVC | +16.5% | +16.7% | +17.6% | +11.0 … +19.4 | 138/138 |
| AV1 | openh264 | -49.1% | -51.6% | -49.9% | -62.4 … -40.4 | 130/138 |
| AV1 Fast | openh264 | -43.4% | -42.8% | -45.8% | -58.7 … -32.4 | 135/138 |
| AV1 | SVT-AV1 M8 | -2.7% | -7.6% | -8.2% | -16.0 … -2.6 | 138/138 |
| AV1 | SVT-AV1 M12 | -24.1% | -30.2% | -23.5% | -34.0 … -24.0 | 138/138 |
| AV1 Fast | SVT-AV1 M12 | -15.6% | -16.4% | -17.9% | -27.3 … -15.4 | 138/138 |
| AV1 | libaom rt 8 | -21.7% | -29.1% | -22.2% | -34.3 … -18.9 | 138/138 |
| AV1 Fast | AV1 | +12.3% | +21.3% | +8.6% | +9.7 … +18.2 | 138/138 |
| AV1 Faster | SVT-AV1 M12 | -6.7% | -6.8% | -9.4% | -22.7 … -3.7 | 138/138 |
| AV1 Faster | AV1 Fast | +11.0% | +12.7% | +11.2% | +6.7 … +15.9 | 138/138 |
| HEVC Faster | SVT-AV1 M13 | -8.6% | -7.0% | -10.3% | -18.4 … -3.6 | 138/138 |
| HEVC Faster | HEVC Fast | +11.2% | +14.1% | +13.8% | +7.8 … +17.8 | 138/138 |
| VP9 | libvpx VP9 rt 7 | -42.2% | -46.4% | -40.6% | -50.1 … -37.2 | 138/138 |
| VP9 Fast | libvpx VP9 rt 9 | -37.5% | -42.9% | -37.1% | -45.7 … -32.7 | 138/138 |
| VP9 Fast | VP9 | +12.1% | +15.0% | +10.3% | +8.2 … +15.7 | 138/138 |
| VP9 Faster | libvpx VP9 rt 9 | -32.7% | -36.6% | -33.5% | -42.5 … -26.5 | 138/138 |
| VP9 Faster | VP9 Fast | +8.4% | +12.0% | +6.2% | +3.6 … +12.0 | 138/138 |
| VP9 Fastest | libvpx VP9 rt 9 | -23.4% | -28.3% | -27.3% | -34.5 … -17.5 | 138/138 |
| VP9 Fastest | VP9 Faster | +15.6% | +14.9% | +9.4% | +5.6 … +19.0 | 138/138 |
| AV2 | SVT-AV1 M8 | -9.8% | -13.6% | -18.8% | -20.7 … -8.9 | 138/138 |
| AV2 | AV1 | -7.1% | -6.5% | -10.6% | -8.9 … -1.8 | 138/138 |
| AV2 Fast | AV2 | +14.6% | +20.4% | +15.9% | +12.3 … +21.9 | 138/138 |
| VVC | x265 medium | -30.6% | -34.1% | -36.9% | -41.4 … -24.4 | 138/138 |
| VVC | HEVC | -9.3% | -9.5% | -10.0% | -11.6 … -7.1 | 138/138 |
| VVC Fast | VVC | +16.0% | +19.7% | +18.3% | +13.5 … +21.2 | 138/138 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753), its presets), openh264
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
  capped at half the source bitrate; 138 natural VAL clips in 5 content
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
- **x86 speed (`bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr-x86.svg`, `.csv`):** the same streams (same quantizers and
  binaries' x86 builds) on a Ryzen 9 5950X, a shared machine, so each run is pinned
  to one logical CPU (one per physical core) and every point is paired with an x264
  medium run on the same CPU in interleaved rounds (machine load slows both); the
  minimum of 5 rounds per clip, after a warm-up.
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-2.1 [-3.2, -1.0]%); x264 medium --tune ssim (-2.2 [-3.0, -1.4]%); x265 ultrafast --tune psnr (-1.1 [-1.4, -0.8]%); x265 superfast --tune psnr (-2.1 [-2.5, -1.8]%); x265 medium --tune psnr (-5.7 [-6.3, -5.1]%); libvpx VP9 rt 7 --tune ssim (-0.8 [-1.3, -0.3]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
