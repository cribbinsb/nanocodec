# Bitrate vs encode speed, held-out clips (independent_2026-11-13), zero line x264-medium~psnr

<sub>Updated 2026-11-14, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 209 clips of `independent_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_2026-11-13-vs-x264-medium-psnr.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_2026-11-13-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC** | -41.8% | -41.7% | -39.6% | -49.8 … -37.8 | 27.9 | 1.05× | 71.8 | 1.06× | 205/205 | Pareto; Pareto x86 |
| **AV2** | -39.6% | -42.6% | -36.4% | -48.3 … -35.3 | 31.5 | 1.18× | 85.5 | 1.27× | 205/205 | Pareto; Pareto x86 |
| **HEVC** | -36.7% | -36.2% | -34.5% | -45.8 … -32.7 | 51.0 | 1.91× | 121.0 | 1.79× | 205/205 | Pareto; Pareto x86 |
| **AV1** | -36.1% | -39.1% | -30.3% | -44.0 … -32.9 | 44.1 | 1.65× | 107.0 | 1.58× | 205/205 | 1 rate-invalid pts |
| **VP9** | -35.9% | -37.4% | -31.1% | -43.9 … -30.9 | 39.2 | 1.47× | 100.0 | 1.48× | 205/205 | 2 rate-invalid pts |
| **VVC Fast** | -33.9% | -31.5% | -30.6% | -42.3 … -27.8 | 47.3 | 1.77× | 115.4 | 1.71× | 205/205 |  |
| **AV2 Fast** | -32.5% | -33.3% | -27.3% | -41.6 … -27.1 | 44.1 | 1.65× | 125.4 | 1.86× | 205/205 | Pareto x86 |
| **HEVC Fast** | -29.7% | -29.2% | -25.9% | -38.3 … -24.8 | 78.5 | 2.94× | 175.8 | 2.60× | 205/205 | Pareto; Pareto x86 |
| **VP9 Fast** | -29.7% | -29.1% | -25.7% | -37.7 … -23.1 | 47.2 | 1.77× | 117.0 | 1.73× | 205/205 | 1 rate-invalid pts |
| **AV1 Fast** | -29.1% | -27.8% | -24.5% | -39.8 … -25.6 | 66.7 | 2.50× | 160.5 | 2.38× | 205/205 |  |
| SVT-AV1 M8 | -27.7% | -31.6% | -14.5% | -38.1 … -25.0 | 14.3 | 0.53× | 45.2 | 0.62× | 205/205 | 31 rate-invalid pts |
| **VP9 Faster** | -25.1% | -23.1% | -21.2% | -35.0 … -17.2 | 78.4 | 2.94× | 180.9 | 2.68× | 205/205 | Pareto x86; 1 rate-invalid pts |
| **HEVC Faster** | -23.4% | -21.9% | -17.4% | -33.2 … -16.2 | 98.0 | 3.67× | 206.4 | 3.05× | 205/205 | Pareto; Pareto x86 |
| **AV1 Faster** | -23.4% | -21.8% | -17.9% | -34.7 … -17.8 | 80.5 | 3.02× | 174.9 | 2.59× | 205/205 |  |
| x265 medium, metric-tuned (--tune psnr) | -22.3% | -23.3% | -15.0% | -32.3 … -18.0 | 8.0 | 0.30× | 25.6 | 0.37× | 205/205 | 32 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M10 | -20.8% | -22.7% | -9.5% | -30.8 … -17.0 | 39.3 | 1.47× | 106.6 | 1.58× | 205/205 | 27 rate-invalid pts |
| x265 superfast, metric-tuned (--tune psnr) | -19.0% | -18.8% | -11.6% | -28.1 … -15.6 | 12.1 | 0.45× | 38.8 | 0.54× | 205/205 | 30 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 medium | -18.1% | -16.6% | -8.2% | -28.4 … -13.1 | 8.0 | 0.30× | 25.6 | 0.37× | 205/205 | 34 rate-invalid pts |
| x265 superfast | -17.5% | -15.5% | -10.6% | -25.5 … -13.0 | 12.1 | 0.45× | 38.8 | 0.54× | 205/205 | 31 rate-invalid pts |
| **VP9 Fastest** | -16.7% | -14.5% | -15.1% | -32.3 … -8.8 | 88.0 | 3.30× | 192.4 | 2.85× | 205/205 |  |
| **H.264** | -16.4% | -18.6% | -12.6% | -31.8 … -6.3 | 74.4 | 2.79× | 175.9 | 2.60× | 205/205 |  |
| libaom rt 8 | -14.1% | -12.0% | -6.7% | -22.9 … -10.0 | 17.4 | 0.65× | 42.1 | 0.61× | 205/205 | 113 rate-invalid pts |
| libaom rt 10 | -13.1% | -12.7% | -3.8% | -21.5 … -7.9 | 22.3 | 0.83× | 49.2 | 0.74× | 205/205 | 123 rate-invalid pts |
| SVT-AV1 M12 | -11.6% | -11.6% | -1.4% | -23.4 … -8.4 | 70.9 | 2.67× | 182.1 | 2.69× | 205/205 | 24 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -11.4% | -10.5% | -3.7% | -22.6 … -4.3 | 17.1 | 0.64× | 53.5 | 0.78× | 205/205 | 30 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -11.0% | -8.5% | -3.7% | -20.9 … -2.5 | 17.1 | 0.64× | 53.5 | 0.78× | 205/205 | 30 rate-invalid pts |
| SVT-AV1 M13 | -9.5% | -9.0% | +0.6% | -21.5 … -6.0 | 79.2 | 2.96× | 193.1 | 2.86× | 205/205 | 24 rate-invalid pts |
| **H.264 Fast** | -8.1% | -10.4% | -4.9% | -24.1 … +2.5 | 113.5 | 4.25× | 230.2 | 3.41× | 205/205 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 67.6 | 1.00× | 205/205 | 44 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Faster** | +0.0% | -3.5% | +5.4% | -19.8 … +18.9 | 133.7 | 5.01× | 265.7 | 3.93× | 205/205 | Pareto; Pareto x86 |
| **H.264 Fastest** | +9.2% | +4.4% | +14.2% | -13.9 … +28.4 | 153.5 | 5.76× | 311.8 | 4.61× | 205/205 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune ssim) | +10.5% | +9.2% | +25.5% | +6.9 … +14.7 | 26.7 | 1.00× | 67.6 | 1.00× | 205/205 | 39 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.8% | +19.7% | +18.5% | +5.0 … +16.4 | 47.8 | 1.78× | 113.0 | 1.53× | 205/205 | 206 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +13.0% | +20.2% | +19.1% | +5.6 … +18.1 | 47.8 | 1.78× | 113.0 | 1.53× | 205/205 | 207 rate-invalid pts |
| x264 medium | +13.1% | +12.7% | +25.2% | +7.6 … +16.3 | 26.7 | 1.00× | 67.6 | 1.00× | 205/205 | 40 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.1% | +19.2% | +25.1% | +4.5 … +18.4 | 30.0 | 1.12× | 68.1 | 1.06× | 205/205 | 169 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | +14.1% | +12.1% | +14.3% | +6.8 … +23.0 | 85.4 | 3.18× | 206.4 | 3.05× | 205/205 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +15.1% | +20.5% | +20.9% | +5.4 … +21.2 | 30.0 | 1.12× | 68.1 | 1.06× | 205/205 | 198 rate-invalid pts |
| x264 veryfast | +18.0% | +14.4% | +33.6% | +6.5 … +26.1 | 85.4 | 3.18× | 206.4 | 3.05× | 205/205 | 1 rate-invalid pts |
| openh264 | +29.8% | +26.0% | +42.9% | -4.5 … +69.3 | 111.5 | 4.17× | 260.3 | 3.85× | 205/205 | 331 rate-invalid pts |
| x264 ultrafast | +56.1% | +69.7% | +45.3% | +41.0 … +78.3 | 190.5 | 7.11× | 547.1 | 8.09× | 205/205 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: pool_barscene_pristine, pool_bodycam_bwc_video118, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_ducks_take_off_pristine, pool_meva_hospital_g301, pool_static_chirla_chirla_seq_024_camera_7_2023, ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 4; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 22, crowd 56, drone 3, lowlight 15, pan 53, static 54, texture 2 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +13.1% | +12.5% |
| H.264 | -16.4% | -17.3% |
| H.264 Fast | -8.1% | -8.5% |
| H.264 Faster | +0.0% | +0.4% |
| H.264 Fastest | +9.2% | +9.1% |
| HEVC | -36.7% | -38.9% |
| HEVC Fast | -29.7% | -31.2% |
| HEVC Faster | -23.4% | -24.3% |
| AV1 | -36.1% | -38.7% |
| AV1 Fast | -29.1% | -32.3% |
| AV1 Faster | -23.4% | -26.0% |
| VP9 | -35.9% | -37.2% |
| VP9 Fast | -29.7% | -30.7% |
| VP9 Faster | -25.1% | -26.1% |
| VP9 Fastest | -16.7% | -18.6% |
| AV2 | -39.6% | -42.0% |
| AV2 Fast | -32.5% | -34.4% |
| VVC | -41.8% | -43.6% |
| VVC Fast | -33.9% | -35.3% |
| openh264 | +29.8% | +32.7% |
| x264 ultrafast | +56.1% | +58.6% |
| x264 veryfast | +18.0% | +17.1% |
| x265 ultrafast | -11.0% | -12.3% |
| x265 superfast | -17.5% | -19.6% |
| x265 medium | -18.1% | -20.3% |
| SVT-AV1 M8 | -27.7% | -31.5% |
| SVT-AV1 M10 | -20.8% | -24.3% |
| SVT-AV1 M12 | -11.6% | -16.5% |
| SVT-AV1 M13 | -9.5% | -14.4% |
| libvpx VP9 rt 7 | +15.1% | +15.4% |
| libvpx VP9 rt 9 | +13.0% | +12.4% |
| libaom rt 8 | -14.1% | -15.6% |
| libaom rt 10 | -13.1% | -14.9% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +29.8% | +26.0% | +42.9% | +20.6% | 205/205 |  |
| openh264, adaptive quantisation on | +29.8% | +26.0% | +42.9% | +20.6% | 205/205 | +0.0 [+0.0, +0.0] not distinguishable |
| x264 ultrafast | +56.1% | +69.7% | +45.3% | +53.3% | 205/205 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +56.2% | +69.9% | +45.1% | +53.6% | 205/205 | +0.1 [-0.1, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +54.8% | +68.3% | +65.2% | +31.0% | 205/205 | +0.1 [-1.6, +1.6] not distinguishable |
| x264 veryfast | +18.0% | +14.4% | +33.6% | +6.1% | 205/205 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +14.1% | +12.1% | +14.3% | +16.0% | 205/205 | -1.9 [-3.2, -0.6] better |
| x264 veryfast, metric-tuned (--tune ssim) | +19.2% | +16.1% | +37.0% | +4.7% | 205/205 | +1.0 [+0.4, +1.6] worse |
| x264 medium | +13.1% | +12.7% | +25.2% | +1.5% | 205/205 |  |
| x264 medium, metric-tuned (--tune ssim) ● | +10.5% | +9.2% | +25.5% | -3.2% | 205/205 | -2.5 [-3.2, -1.7] better |
| x265 ultrafast | -11.0% | -8.5% | -3.7% | -20.7% | 205/205 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -11.4% | -10.5% | -3.7% | -20.1% | 205/205 | -0.7 [-1.1, -0.3] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -11.4% | -10.5% | -3.7% | -20.1% | 205/205 | -0.7 [-1.1, -0.3] better |
| x265 superfast | -17.5% | -15.5% | -10.6% | -26.3% | 205/205 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -19.0% | -18.8% | -11.6% | -26.7% | 205/205 | -2.0 [-2.5, -1.7] better |
| x265 superfast, metric-tuned (--tune ssim) | -19.0% | -18.8% | -11.6% | -26.7% | 205/205 | -2.0 [-2.5, -1.7] better |
| x265 medium | -18.1% | -16.6% | -8.2% | -29.7% | 205/205 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -22.3% | -23.3% | -15.0% | -28.5% | 205/205 | -5.0 [-5.6, -4.4] better |
| x265 medium, metric-tuned (--tune ssim) | -20.1% | -20.1% | -9.7% | -30.4% | 205/205 | -2.3 [-2.6, -2.1] better |
| libvpx VP9 rt 7 | +15.1% | +20.5% | +20.9% | +4.0% | 205/205 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +14.1% | +19.2% | +25.1% | -2.1% | 205/205 | -1.8 [-2.4, -1.1] better |
| libvpx VP9 rt 9 | +13.0% | +20.2% | +19.1% | -0.3% | 205/205 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | +11.8% | +19.7% | +18.5% | -2.9% | 205/205 | -1.3 [-1.8, -0.9] better |
| libaom rt 8 | -14.1% | -12.0% | -6.7% | -23.7% | 205/205 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -13.5% | -10.2% | -1.0% | -29.2% | 205/205 | +1.0 [+0.3, +1.7] worse |
| libaom rt 10 | -13.1% | -12.7% | -3.8% | -22.6% | 205/205 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -13.1% | -11.2% | +0.1% | -28.2% | 205/205 | +0.4 [-0.3, +1.2] not distinguishable |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC** | -48.9% | -50.8% | -52.3% | -43.4% | 205/205 |
| **AV2** | -46.8% | -51.7% | -49.7% | -39.0% | 205/205 |
| **HEVC** | -44.2% | -45.8% | -48.1% | -38.8% | 205/205 |
| **AV1** | -43.8% | -48.4% | -44.9% | -38.1% | 205/205 |
| **VP9** | -43.7% | -46.8% | -45.6% | -38.6% | 205/205 |
| **VVC Fast** | -42.0% | -41.8% | -45.2% | -39.0% | 205/205 |
| **AV2 Fast** | -40.6% | -43.4% | -42.5% | -36.1% | 205/205 |
| **VP9 Fast** | -38.1% | -39.3% | -41.3% | -33.8% | 205/205 |
| **HEVC Fast** | -38.1% | -39.3% | -41.4% | -33.5% | 205/205 |
| **AV1 Fast** | -37.7% | -38.3% | -40.2% | -34.5% | 205/205 |
| SVT-AV1 M8 | -36.9% | -42.1% | -32.3% | -36.3% | 205/205 |
| **VP9 Faster** | -34.0% | -33.8% | -37.8% | -30.6% | 205/205 |
| **HEVC Faster** | -32.5% | -32.5% | -34.7% | -30.4% | 205/205 |
| **AV1 Faster** | -32.5% | -32.5% | -34.9% | -30.0% | 205/205 |
| x265 medium, metric-tuned (--tune psnr) | -31.4% | -33.9% | -32.4% | -28.0% | 205/205 |
| SVT-AV1 M10 | -30.9% | -34.5% | -28.5% | -29.8% | 205/205 |
| x265 medium, metric-tuned (--tune ssim) | -29.8% | -31.1% | -28.3% | -30.0% | 205/205 |
| x265 superfast, metric-tuned (--tune psnr) | -28.8% | -30.3% | -29.7% | -26.2% | 205/205 |
| x265 superfast, metric-tuned (--tune ssim) | -28.8% | -30.3% | -29.7% | -26.2% | 205/205 |
| x265 medium | -28.2% | -28.2% | -27.1% | -29.2% | 205/205 |
| x265 superfast | -27.4% | -27.4% | -29.0% | -25.9% | 205/205 |
| **VP9 Fastest** | -26.2% | -25.5% | -32.9% | -20.1% | 205/205 |
| **H.264** | -25.9% | -29.2% | -30.3% | -18.2% | 205/205 |
| libaom rt 8 | -24.3% | -23.8% | -26.0% | -23.4% | 204/205 |
| libaom rt 8, metric-tuned (--tune ssim) | -24.2% | -22.0% | -21.5% | -28.9% | 205/205 |
| libaom rt 10, metric-tuned (--tune ssim) | -24.0% | -23.4% | -20.7% | -27.8% | 205/205 |
| libaom rt 10 | -23.7% | -24.9% | -24.0% | -22.2% | 205/205 |
| SVT-AV1 M12 | -22.4% | -24.0% | -21.6% | -21.7% | 205/205 |
| x265 ultrafast, metric-tuned (--tune psnr) | -22.1% | -22.9% | -23.6% | -20.0% | 205/205 |
| x265 ultrafast, metric-tuned (--tune ssim) | -22.1% | -22.9% | -23.6% | -20.0% | 205/205 |
| x265 ultrafast | -21.8% | -21.0% | -23.7% | -20.6% | 205/205 |
| SVT-AV1 M13 | -20.6% | -21.5% | -20.0% | -20.2% | 205/205 |
| **H.264 Fast** | -18.5% | -21.6% | -24.2% | -9.6% | 205/205 |
| **H.264 Faster** | -11.2% | -15.1% | -16.0% | -2.5% | 205/205 |
| x264 medium, metric-tuned (--tune psnr) | -10.3% | -10.6% | -19.3% | -1.0% | 205/205 |
| **H.264 Fastest** | -2.7% | -7.5% | -8.8% | +8.0% | 205/205 |
| x264 medium, metric-tuned (--tune ssim) | -2.5% | -2.9% | -0.0% | -4.5% | 205/205 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.5% | +4.5% | -6.1% | -3.1% | 205/205 |
| libvpx VP9 rt 9 | -0.4% | +4.9% | -5.6% | -0.6% | 205/205 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.0% | +5.5% | +0.5% | -3.0% | 205/205 |
| x264 veryfast, metric-tuned (--tune psnr) | +1.8% | -0.3% | -8.8% | +14.5% | 205/205 |
| libvpx VP9 rt 7 | +1.9% | +6.1% | -3.6% | +3.2% | 205/205 |
| x264 veryfast | +3.7% | +1.3% | +5.1% | +4.7% | 205/205 |
| x264 veryfast, metric-tuned (--tune ssim) | +4.6% | +2.7% | +7.7% | +3.4% | 205/205 |
| openh264 | +13.9% | +9.5% | +12.8% | +19.5% | 205/205 |
| openh264, adaptive quantisation on | +13.9% | +9.5% | +12.8% | +19.5% | 205/205 |
| x264 ultrafast, metric-tuned (--tune ssim) | +35.7% | +50.2% | +28.3% | +28.7% | 205/205 |
| x264 ultrafast | +38.9% | +52.3% | +14.2% | +50.1% | 205/205 |
| x264 ultrafast, metric-tuned (--tune psnr) | +39.0% | +52.4% | +14.1% | +50.4% | 205/205 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium, metric-tuned (--tune psnr)'s (even with its extra rungs, which the 50% source cap limits).

- **pool_ducks_take_off_pristine** (texture): no reliable overlap for SVT-AV1 M8, SVT-AV1 M10, libvpx VP9 rt 9.
- **ind_bwc_videotext_video190** (bodycam): no reliable overlap for libaom rt 8, libaom rt 10.
- **ind_personpath22_uid_vid_00190** (static): no reliable overlap for H.264, HEVC, AV1 Fast, VP9, VVC, VVC Fast.
- **ind_personpath22_uid_vid_00212** (crowd): no reliable overlap for HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, VP9, VP9 Fast, VP9 Faster, VP9 Fastest, AV2, AV2 Fast, VVC, VVC Fast, SVT-AV1 M8, SVT-AV1 M10.

Direct BD-J vs x264 medium, metric-tuned (--tune psnr) on them, where it exists:

| Encoder | pool_ducks_take_off_pristine | ind_bwc_videotext_video190 | ind_personpath22_uid_vid_00190 | ind_personpath22_uid_vid_00212 |
|---|---:|---:|---:|---:|
| VVC | -50.9% | -28.3% | n/a | n/a |
| AV2 | -49.1% | -23.7% | -42.8% | n/a |
| HEVC | -36.4% | -20.3% | n/a | n/a |
| AV1 | -51.3% | -18.5% | -40.3% | n/a |
| VP9 | -45.5% | -23.8% | n/a | n/a |
| VVC Fast | -48.1% | -21.1% | n/a | n/a |
| AV2 Fast | -37.9% | -19.4% | -30.8% | n/a |
| HEVC Fast | -40.5% | -13.4% | -30.3% | n/a |
| VP9 Fast | -33.9% | -18.3% | -32.4% | n/a |
| AV1 Fast | -46.7% | -15.1% | n/a | n/a |
| SVT-AV1 M8 | n/a | -23.3% | -27.7% | n/a |
| VP9 Faster | -34.4% | -17.8% | -31.9% | n/a |
| HEVC Faster | -23.3% | -10.2% | -24.7% | n/a |
| AV1 Faster | -44.2% | -16.2% | -28.5% | n/a |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a | n/a |
| SVT-AV1 M10 | n/a | +5.7% | -19.7% | n/a |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a | n/a | n/a |
| x265 medium | -41.3% | +0.8% | +4.4% | -17.4% |
| x265 superfast | -41.9% | +10.6% | +2.4% | -9.9% |
| VP9 Fastest | -34.2% | -16.5% | -30.0% | n/a |
| H.264 | +6.5% | -17.3% | n/a | -31.4% |
| libaom rt 8 | -5.9% | n/a | +0.2% | -16.9% |
| libaom rt 10 | -3.1% | n/a | +4.6% | -16.6% |
| SVT-AV1 M12 | -23.8% | +8.8% | -16.3% | -11.9% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a | n/a | n/a |
| x265 ultrafast | -41.9% | +20.2% | +18.6% | +5.7% |
| SVT-AV1 M13 | -23.5% | +10.1% | -14.4% | -11.3% |
| H.264 Fast | +16.9% | -8.0% | -29.0% | -24.3% |
| H.264 Faster | +40.5% | -6.4% | -27.1% | -16.6% |
| H.264 Fastest | +38.8% | -0.7% | -22.8% | -10.9% |
| x264 medium, metric-tuned (--tune ssim) | n/a | n/a | n/a | n/a |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a | n/a | n/a | n/a |
| libvpx VP9 rt 9 | n/a | +8.6% | +24.3% | +19.5% |
| x264 medium | +15.9% | +2.6% | -5.5% | -13.2% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a | n/a | n/a |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a | n/a | n/a |
| libvpx VP9 rt 7 | +8.7% | -10.5% | +24.1% | +28.3% |
| x264 veryfast | +13.9% | +24.1% | +0.3% | -9.1% |
| openh264 | +109.2% | +26.1% | -30.1% | +7.2% |
| x264 ultrafast | +29.5% | +64.9% | +74.1% | +58.7% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 | openh264 | -30.8% | -31.0% | -34.3% | -43.2 … -23.1 | 209/209 |
| H.264 | x264 veryfast | -31.8% | -34.1% | -36.0% | -40.7 … -26.6 | 209/209 |
| H.264 | x264 medium | -25.0% | -29.2% | -28.4% | -37.6 … -16.3 | 209/209 |
| H.264 Fast | x264 veryfast | -24.6% | -27.0% | -29.8% | -33.8 … -18.3 | 209/209 |
| H.264 Fast | openh264 | -22.1% | -22.4% | -27.2% | -36.3 … -13.5 | 209/209 |
| H.264 Fast | H.264 | +11.8% | +11.9% | +10.4% | +9.5 … +15.3 | 209/209 |
| H.264 Faster | x264 ultrafast | -38.2% | -49.3% | -27.9% | -48.2 … -34.3 | 196/209 |
| H.264 Faster | H.264 Fast | +11.1% | +9.3% | +13.6% | +5.0 … +15.4 | 209/209 |
| H.264 Fastest | x264 ultrafast | -32.1% | -44.0% | -23.5% | -44.9 … -27.9 | 199/209 |
| H.264 Fastest | H.264 Faster | +11.4% | +11.1% | +8.2% | +8.5 … +13.4 | 209/209 |
| HEVC | x265 medium | -20.7% | -23.5% | -26.9% | -34.7 … -13.2 | 209/209 |
| HEVC Fast | x265 superfast | -12.4% | -15.1% | -14.8% | -26.7 … -6.8 | 209/209 |
| HEVC | SVT-AV1 M12 | -26.6% | -26.8% | -32.6% | -35.3 … -23.5 | 209/209 |
| HEVC Fast | HEVC | +14.2% | +15.1% | +16.2% | +11.0 … +19.5 | 209/209 |
| AV1 | openh264 | -48.8% | -50.8% | -48.9% | -62.7 … -39.9 | 197/209 |
| AV1 Fast | openh264 | -42.5% | -40.5% | -43.6% | -59.3 … -30.7 | 204/209 |
| AV1 | SVT-AV1 M8 | -5.0% | -6.3% | -13.6% | -16.5 … -1.6 | 209/209 |
| AV1 | SVT-AV1 M12 | -27.5% | -31.9% | -29.1% | -34.3 … -23.9 | 209/209 |
| AV1 Fast | SVT-AV1 M12 | -18.6% | -17.3% | -22.6% | -26.9 … -13.8 | 209/209 |
| AV1 | libaom rt 8 | -23.2% | -29.2% | -23.8% | -32.6 … -17.9 | 209/209 |
| AV1 Fast | AV1 | +13.7% | +23.4% | +10.5% | +9.9 … +18.1 | 209/209 |
| AV1 Faster | SVT-AV1 M12 | -9.6% | -7.5% | -13.5% | -22.6 … -0.4 | 209/209 |
| AV1 Faster | AV1 Fast | +11.3% | +12.9% | +12.1% | +6.9 … +16.2 | 209/209 |
| HEVC Faster | SVT-AV1 M13 | -10.3% | -8.3% | -14.1% | -18.4 … -3.0 | 209/209 |
| HEVC Faster | HEVC Fast | +12.0% | +14.4% | +14.6% | +7.7 … +17.8 | 209/209 |
| VP9 | libvpx VP9 rt 7 | -41.7% | -45.6% | -40.3% | -47.9 … -35.1 | 209/209 |
| VP9 Fast | libvpx VP9 rt 9 | -36.0% | -40.6% | -36.1% | -44.2 … -30.6 | 209/209 |
| VP9 Fast | VP9 | +12.5% | +16.6% | +10.1% | +8.9 … +16.5 | 209/209 |
| VP9 Faster | libvpx VP9 rt 9 | -30.6% | -33.9% | -31.3% | -40.9 … -24.3 | 209/209 |
| VP9 Faster | VP9 Fast | +8.5% | +11.4% | +7.1% | +3.7 … +13.7 | 209/209 |
| VP9 Fastest | libvpx VP9 rt 9 | -20.7% | -24.3% | -24.2% | -34.5 … -14.5 | 208/209 |
| VP9 Fastest | VP9 Faster | +16.2% | +15.4% | +10.0% | +5.9 … +20.6 | 209/209 |
| AV2 | SVT-AV1 M8 | -10.8% | -11.7% | -22.1% | -20.5 … -6.7 | 209/209 |
| AV2 | AV1 | -5.5% | -5.5% | -8.5% | -8.9 … -1.8 | 209/209 |
| AV2 Fast | AV2 | +15.6% | +21.2% | +17.9% | +12.1 … +21.8 | 209/209 |
| VVC | x265 medium | -28.8% | -31.5% | -34.2% | -40.1 … -22.3 | 209/209 |
| VVC | HEVC | -10.3% | -10.8% | -9.5% | -12.1 … -6.6 | 209/209 |
| VVC Fast | VVC | +16.8% | +22.1% | +18.9% | +13.7 … +20.8 | 209/209 |

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
  capped at half the source bitrate; 209 natural VAL clips in 7 content
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
- **x86 speed (`bd-vs-speed-independent_2026-11-13-vs-x264-medium-psnr-x86.svg`, `.csv`):** the same streams (same quantizers and
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.9 [-3.2, -0.6]%); x264 medium --tune ssim (-2.5 [-3.2, -1.7]%); x265 ultrafast --tune psnr (-0.7 [-1.1, -0.3]%); x265 superfast --tune psnr (-2.0 [-2.5, -1.7]%); x265 medium --tune psnr (-5.0 [-5.6, -4.4]%); libvpx VP9 rt 7 --tune ssim (-1.8 [-2.4, -1.1]%); libvpx VP9 rt 9 --tune ssim (-1.3 [-1.8, -0.9]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
