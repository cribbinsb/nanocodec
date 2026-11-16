# Bitrate vs encode speed, held-out clips (independent_2026-11-13), zero line x264-medium~psnr

<sub>Updated 2026-11-16, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, HEVC Fast, AV1, AV1 Fast, AV1 Faster, AV2, VVC: 4a5b8dd1; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976); AV2 Fast, H.264 Fast, H.264 Faster, HEVC Faster, VP9 Fast, VP9 Faster, VP9 Veryfast, VVC Fast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow timed on 1139f610 (same streams)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 209 clips of `independent_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_2026-11-13-vs-x264-medium-psnr.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_2026-11-13-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -44.7% | -44.7% | -42.4% | -54.4 … -40.3 | 23.3 | 0.88× | 89.3 | 0.79× | 206/206 | Pareto; Pareto x86 |
| **VVC** | -44.2% | -44.2% | -41.6% | -53.6 … -39.5 | 27.0 | 1.01× | 109.3 | 0.97× | 206/206 | Pareto; Pareto x86 |
| **AV2 Slow** | -44.1% | -46.8% | -40.7% | -53.5 … -40.6 | 19.3 | 0.72× | 71.4 | 0.63× | 206/206 |  |
| **AV2** | -42.5% | -45.4% | -39.1% | -52.3 … -38.5 | 32.5 | 1.22× | 140.8 | 1.25× | 206/206 | Pareto; Pareto x86 |
| **HEVC Slow** | -41.0% | -40.0% | -38.5% | -52.0 … -36.0 | 28.9 | 1.08× | 114.0 | 1.01× | 206/206 |  |
| **AV1 Slow** | -39.6% | -42.6% | -34.8% | -50.2 … -36.4 | 27.2 | 1.02× | 112.8 | 1.00× | 206/206 | 1 rate-invalid pts |
| **HEVC** | -39.1% | -38.6% | -37.2% | -49.9 … -34.3 | 46.6 | 1.75× | 179.9 | 1.60× | 206/206 | Pareto; Pareto x86 |
| **AV1** | -38.7% | -41.7% | -33.8% | -49.5 … -35.6 | 40.9 | 1.53× | 161.0 | 1.43× | 206/206 | 1 rate-invalid pts |
| **VP9 Slow** | -38.2% | -40.3% | -33.6% | -47.9 … -32.9 | 34.5 | 1.29× | 131.3 | 1.16× | 206/206 | 4 rate-invalid pts |
| **VP9** | -37.7% | -39.9% | -32.9% | -47.6 … -32.2 | 40.7 | 1.53× | 165.6 | 1.47× | 206/206 | 4 rate-invalid pts |
| **VVC Fast** | -34.3% | -31.8% | -30.9% | -43.4 … -27.9 | 49.4 | 1.85× | 190.5 | 1.69× | 206/206 | Pareto; Pareto x86 |
| **AV1 Fast** | -34.0% | -36.3% | -27.5% | -43.8 … -30.2 | 59.6 | 2.23× | 250.8 | 2.22× | 206/206 | Pareto; Pareto x86 |
| **AV2 Fast** | -32.9% | -33.8% | -27.5% | -42.0 … -27.1 | 44.6 | 1.67× | 207.7 | 1.84× | 206/206 |  |
| **HEVC Fast** | -30.3% | -29.8% | -24.6% | -39.2 … -25.0 | 78.0 | 2.92× | 303.5 | 2.69× | 206/206 | Pareto; Pareto x86 |
| **VP9 Fast** | -30.0% | -29.5% | -26.0% | -38.8 … -23.3 | 54.2 | 2.03× | 205.1 | 1.82× | 206/206 | 1 rate-invalid pts |
| SVT-AV1 M8 | -28.1% | -31.8% | -14.9% | -38.5 … -25.2 | 14.3 | 0.53× | 80.8 | 0.72× | 206/206 | 34 rate-invalid pts |
| **AV1 Faster** | -27.8% | -26.8% | -23.4% | -40.2 … -23.1 | 74.6 | 2.80× | 346.7 | 3.07× | 206/206 | Pareto x86 |
| **VP9 Faster** | -25.6% | -23.6% | -21.7% | -35.8 … -17.5 | 86.5 | 3.24× | 328.1 | 2.91× | 206/206 | Pareto; 1 rate-invalid pts |
| **HEVC Faster** | -23.8% | -22.3% | -17.8% | -33.8 … -16.4 | 98.9 | 3.71× | 409.9 | 3.64× | 206/206 | Pareto; Pareto x86 |
| x265 medium, metric-tuned (--tune psnr) | -22.6% | -23.6% | -15.5% | -32.5 … -18.1 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 35 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M10 | -21.3% | -23.1% | -10.0% | -31.3 … -17.2 | 39.3 | 1.47× | 228.9 | 2.03× | 206/206 | 30 rate-invalid pts |
| x265 superfast, metric-tuned (--tune psnr) | -19.4% | -19.2% | -12.0% | -28.8 … -15.9 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 33 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Slow** | -19.2% | -22.6% | -14.5% | -34.8 … -8.7 | 45.7 | 1.71× | 140.1 | 1.24× | 206/206 |  |
| x265 medium | -18.5% | -17.0% | -8.7% | -29.0 … -13.6 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 37 rate-invalid pts |
| **H.264** | -18.1% | -20.3% | -13.9% | -34.5 … -7.6 | 71.9 | 2.69× | 235.5 | 2.09× | 206/206 |  |
| x265 superfast | -17.9% | -15.9% | -11.0% | -27.2 … -13.5 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 34 rate-invalid pts |
| **VP9 Veryfast** | -17.3% | -15.0% | -15.8% | -33.4 … -9.0 | 96.5 | 3.62× | 334.9 | 2.97× | 206/206 |  |
| libaom rt 8 | -14.2% | -12.0% | -6.8% | -22.7 … -9.6 | 17.4 | 0.65× | 68.8 | 0.61× | 206/206 | 126 rate-invalid pts |
| libaom rt 10 | -13.3% | -12.9% | -4.1% | -21.5 … -8.1 | 22.3 | 0.83× | 90.5 | 0.80× | 206/206 | 136 rate-invalid pts |
| SVT-AV1 M12 | -12.2% | -12.1% | -2.1% | -23.8 … -8.5 | 70.9 | 2.67× | 408.7 | 3.62× | 206/206 | 26 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -12.1% | -11.2% | -4.4% | -23.2 … -5.8 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 33 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -11.6% | -9.2% | -4.3% | -22.2 … -4.0 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 33 rate-invalid pts |
| SVT-AV1 M13 | -10.2% | -9.5% | -0.1% | -22.6 … -6.1 | 79.2 | 2.96× | 445.4 | 3.95× | 206/206 | Pareto x86; 26 rate-invalid pts |
| **H.264 Fast** | -8.6% | -10.8% | -5.3% | -24.6 … +2.2 | 113.3 | 4.25× | 377.1 | 3.34× | 206/206 | Pareto |
| **H.264 Faster** | -0.5% | -4.1% | +4.9% | -20.1 … +18.2 | 135.4 | 5.08× | 462.9 | 4.10× | 206/206 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 49 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Veryfast** | +5.8% | +1.4% | +12.0% | -16.7 … +23.5 | 153.2 | 5.75× | 503.6 | 4.47× | 206/206 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune ssim) | +10.3% | +8.8% | +25.3% | +6.9 … +14.6 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 45 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.4% | +19.6% | +18.0% | +5.0 … +16.3 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 221 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +12.6% | +20.1% | +18.7% | +5.7 … +18.0 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 222 rate-invalid pts |
| x264 medium | +12.9% | +12.3% | +25.1% | +7.4 … +16.3 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 46 rate-invalid pts |
| x264 veryfast, metric-tuned (--tune psnr) | +13.5% | +11.4% | +13.6% | +4.6 … +22.4 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.2% | +19.5% | +25.3% | +4.5 … +19.6 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 185 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +15.2% | +20.7% | +21.1% | +5.3 … +22.0 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 215 rate-invalid pts |
| x264 veryfast | +17.0% | +13.2% | +32.4% | +5.2 … +25.4 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts |
| openh264 | +29.3% | +25.5% | +42.2% | -4.7 … +69.0 | 111.5 | 4.17× | 524.7 | 4.65× | 206/206 | Pareto x86; 353 rate-invalid pts |
| x264 ultrafast | +54.5% | +67.6% | +44.0% | +33.2 … +77.1 | 190.5 | 7.11× | 1017.0 | 9.02× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: pool_barscene_pristine, pool_bodycam_antare_antare_test, pool_bodycam_bwc_video118, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_crowd_person_uid_vid_00222, pool_ducks_take_off_pristine, pool_lowlight_cevo_inof_md_indoor_night_ohcam_0, pool_meva_hospital_g301, pool_static_chirla_chirla_seq_024_camera_7_2023, ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 3; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 22, crowd 56, drone 3, lowlight 15, pan 53, static 55, texture 2 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +12.9% | +12.3% |
| H.264 Slow | -19.2% | -20.7% |
| H.264 | -18.1% | -19.6% |
| H.264 Fast | -8.6% | -9.2% |
| H.264 Faster | -0.5% | -0.6% |
| H.264 Veryfast | +5.8% | +4.9% |
| HEVC Slow | -41.0% | -43.4% |
| HEVC | -39.1% | -41.6% |
| HEVC Fast | -30.3% | -32.1% |
| HEVC Faster | -23.8% | -24.9% |
| AV1 Slow | -39.6% | -42.9% |
| AV1 | -38.7% | -42.1% |
| AV1 Fast | -34.0% | -37.0% |
| AV1 Faster | -27.8% | -30.9% |
| VP9 Slow | -38.2% | -39.9% |
| VP9 | -37.7% | -39.4% |
| VP9 Fast | -30.0% | -31.3% |
| VP9 Faster | -25.6% | -26.8% |
| VP9 Veryfast | -17.3% | -19.5% |
| AV2 Slow | -44.1% | -46.4% |
| AV2 | -42.5% | -44.9% |
| AV2 Fast | -32.9% | -34.9% |
| VVC Slow | -44.7% | -46.8% |
| VVC | -44.2% | -46.3% |
| VVC Fast | -34.3% | -35.9% |
| openh264 | +29.3% | +32.0% |
| x264 ultrafast | +54.5% | +55.9% |
| x264 veryfast | +17.0% | +15.7% |
| x265 ultrafast | -11.6% | -13.2% |
| x265 superfast | -17.9% | -20.2% |
| x265 medium | -18.5% | -20.9% |
| SVT-AV1 M8 | -28.1% | -32.0% |
| SVT-AV1 M10 | -21.3% | -24.9% |
| SVT-AV1 M12 | -12.2% | -17.3% |
| SVT-AV1 M13 | -10.2% | -15.2% |
| libvpx VP9 rt 7 | +15.2% | +15.8% |
| libvpx VP9 rt 9 | +12.6% | +12.2% |
| libaom rt 8 | -14.2% | -15.5% |
| libaom rt 10 | -13.3% | -15.2% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +29.3% | +25.5% | +42.2% | +20.1% | 206/206 |  |
| openh264, adaptive quantisation on | +29.3% | +25.5% | +42.2% | +20.1% | 206/206 | -0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +54.5% | +67.6% | +44.0% | +52.0% | 206/206 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +54.6% | +67.7% | +43.8% | +52.2% | 206/206 | +0.1 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +52.8% | +65.4% | +63.2% | +29.9% | 206/206 | -0.2 [-1.7, +1.4] not distinguishable |
| x264 veryfast | +17.0% | +13.2% | +32.4% | +5.4% | 206/206 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +13.5% | +11.4% | +13.6% | +15.4% | 206/206 | -1.7 [-3.0, -0.5] better |
| x264 veryfast, metric-tuned (--tune ssim) | +18.2% | +14.9% | +35.6% | +4.1% | 206/206 | +0.9 [+0.4, +1.6] worse |
| x264 medium | +12.9% | +12.3% | +25.1% | +1.5% | 206/206 |  |
| x264 medium, metric-tuned (--tune ssim) ● | +10.3% | +8.8% | +25.3% | -3.2% | 206/206 | -2.5 [-3.2, -1.7] better |
| x265 ultrafast | -11.6% | -9.2% | -4.3% | -21.4% | 206/206 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -12.1% | -11.2% | -4.4% | -20.7% | 206/206 | -0.7 [-1.1, -0.2] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -12.1% | -11.2% | -4.4% | -20.7% | 206/206 | -0.7 [-1.1, -0.2] better |
| x265 superfast | -17.9% | -15.9% | -11.0% | -26.7% | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -19.4% | -19.2% | -12.0% | -27.1% | 206/206 | -2.0 [-2.4, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -19.4% | -19.2% | -12.0% | -27.1% | 206/206 | -2.0 [-2.4, -1.6] better |
| x265 medium | -18.5% | -17.0% | -8.7% | -30.0% | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -22.6% | -23.6% | -15.5% | -28.9% | 206/206 | -5.0 [-5.5, -4.4] better |
| x265 medium, metric-tuned (--tune ssim) | -20.4% | -20.4% | -10.2% | -30.7% | 206/206 | -2.3 [-2.5, -2.1] better |
| libvpx VP9 rt 7 | +15.2% | +20.7% | +21.1% | +3.9% | 206/206 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +14.2% | +19.5% | +25.3% | -2.2% | 206/206 | -1.8 [-2.4, -1.0] better |
| libvpx VP9 rt 9 | +12.6% | +20.1% | +18.7% | -0.8% | 206/206 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | +11.4% | +19.6% | +18.0% | -3.4% | 206/206 | -1.3 [-1.8, -0.8] better |
| libaom rt 8 | -14.2% | -12.0% | -6.8% | -24.0% | 206/206 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -13.6% | -10.1% | -1.3% | -29.5% | 206/206 | +0.9 [+0.2, +1.7] worse |
| libaom rt 10 | -13.3% | -12.9% | -4.1% | -22.9% | 206/206 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -13.5% | -11.5% | -0.4% | -28.5% | 206/206 | +0.4 [-0.4, +1.1] not distinguishable |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -51.4% | -53.2% | -54.6% | -46.4% | 206/206 |
| **VVC** | -51.0% | -52.8% | -54.0% | -46.2% | 206/206 |
| **AV2 Slow** | -50.7% | -55.1% | -53.2% | -43.8% | 206/206 |
| **AV2** | -49.3% | -53.9% | -51.9% | -42.1% | 206/206 |
| **HEVC Slow** | -48.0% | -48.9% | -51.4% | -43.7% | 206/206 |
| **AV1 Slow** | -46.8% | -51.2% | -48.4% | -40.8% | 206/206 |
| **HEVC** | -46.4% | -47.7% | -50.4% | -41.0% | 206/206 |
| **AV1** | -46.0% | -50.4% | -47.7% | -40.1% | 206/206 |
| **VP9 Slow** | -45.6% | -49.0% | -47.6% | -40.1% | 206/206 |
| **VP9** | -45.1% | -48.7% | -47.1% | -39.5% | 206/206 |
| **VVC Fast** | -42.2% | -41.9% | -45.4% | -39.3% | 206/206 |
| **AV1 Fast** | -42.0% | -45.7% | -42.7% | -37.6% | 206/206 |
| **AV2 Fast** | -40.9% | -43.5% | -42.7% | -36.5% | 206/206 |
| **HEVC Fast** | -38.7% | -39.7% | -40.4% | -35.9% | 206/206 |
| **VP9 Fast** | -38.4% | -39.5% | -41.5% | -34.2% | 206/206 |
| SVT-AV1 M8 | -37.2% | -42.2% | -32.6% | -36.7% | 206/206 |
| **AV1 Faster** | -36.3% | -36.9% | -39.4% | -32.6% | 206/206 |
| **VP9 Faster** | -34.4% | -34.0% | -38.1% | -31.0% | 206/206 |
| **HEVC Faster** | -32.8% | -32.6% | -35.0% | -30.7% | 206/206 |
| x265 medium, metric-tuned (--tune psnr) | -31.6% | -33.9% | -32.7% | -28.3% | 206/206 |
| SVT-AV1 M10 | -31.2% | -34.6% | -28.8% | -30.3% | 206/206 |
| x265 medium, metric-tuned (--tune ssim) | -30.0% | -31.2% | -28.7% | -30.2% | 206/206 |
| x265 superfast, metric-tuned (--tune psnr) | -29.0% | -30.4% | -30.0% | -26.6% | 206/206 |
| x265 superfast, metric-tuned (--tune ssim) | -29.0% | -30.4% | -30.0% | -26.6% | 206/206 |
| x265 medium | -28.4% | -28.3% | -27.4% | -29.5% | 206/206 |
| **H.264 Slow** | -28.4% | -32.5% | -31.9% | -20.7% | 206/206 |
| x265 superfast | -27.7% | -27.6% | -29.3% | -26.3% | 206/206 |
| **H.264** | -27.4% | -30.6% | -31.5% | -20.2% | 206/206 |
| **VP9 Veryfast** | -26.6% | -25.7% | -33.4% | -20.7% | 206/206 |
| libaom rt 8 | -24.3% | -23.5% | -26.1% | -23.6% | 205/206 |
| libaom rt 8, metric-tuned (--tune ssim) | -24.2% | -21.7% | -21.7% | -29.1% | 206/206 |
| libaom rt 10, metric-tuned (--tune ssim) | -24.2% | -23.3% | -21.1% | -28.1% | 206/206 |
| libaom rt 10 | -23.8% | -24.7% | -24.2% | -22.5% | 206/206 |
| SVT-AV1 M12 | -22.9% | -24.2% | -22.1% | -22.3% | 206/206 |
| x265 ultrafast, metric-tuned (--tune psnr) | -22.6% | -23.2% | -24.0% | -20.5% | 206/206 |
| x265 ultrafast, metric-tuned (--tune ssim) | -22.6% | -23.2% | -24.0% | -20.5% | 206/206 |
| x265 ultrafast | -22.2% | -21.4% | -24.1% | -21.2% | 206/206 |
| SVT-AV1 M13 | -21.0% | -21.7% | -20.5% | -20.9% | 206/206 |
| **H.264 Fast** | -18.8% | -21.7% | -24.6% | -10.0% | 206/206 |
| **H.264 Faster** | -11.6% | -15.3% | -16.4% | -3.0% | 206/206 |
| x264 medium, metric-tuned (--tune psnr) | -10.2% | -10.3% | -19.2% | -1.0% | 206/206 |
| **H.264 Veryfast** | -5.8% | -10.1% | -10.6% | +3.3% | 206/206 |
| x264 medium, metric-tuned (--tune ssim) | -2.5% | -2.8% | -0.1% | -4.4% | 206/206 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.7% | +4.7% | -6.3% | -3.5% | 206/206 |
| libvpx VP9 rt 9 | -0.6% | +5.1% | -5.8% | -1.1% | 206/206 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.3% | +6.2% | +0.8% | -3.1% | 206/206 |
| x264 veryfast, metric-tuned (--tune psnr) | +1.3% | -0.6% | -9.3% | +13.9% | 206/206 |
| libvpx VP9 rt 7 | +2.1% | +6.6% | -3.4% | +3.1% | 206/206 |
| x264 veryfast | +3.0% | +0.7% | +4.3% | +4.1% | 206/206 |
| x264 veryfast, metric-tuned (--tune ssim) | +3.9% | +2.1% | +6.8% | +2.9% | 206/206 |
| openh264 | +13.6% | +9.3% | +12.3% | +19.1% | 206/206 |
| openh264, adaptive quantisation on | +13.6% | +9.3% | +12.3% | +19.1% | 206/206 |
| x264 ultrafast, metric-tuned (--tune ssim) | +34.5% | +48.4% | +27.2% | +27.7% | 206/206 |
| x264 ultrafast | +37.8% | +51.1% | +13.5% | +49.0% | 206/206 |
| x264 ultrafast, metric-tuned (--tune psnr) | +37.9% | +51.2% | +13.4% | +49.2% | 206/206 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium, metric-tuned (--tune psnr)'s (even with its extra rungs, which the 50% source cap limits).

- **pool_ducks_take_off_pristine** (texture): no reliable overlap for SVT-AV1 M8, SVT-AV1 M10, libvpx VP9 rt 9.
- **ind_bwc_videotext_video190** (bodycam): no reliable overlap for libaom rt 8, libaom rt 10.
- **ind_personpath22_uid_vid_00212** (crowd): no reliable overlap for HEVC Slow, HEVC, HEVC Fast, HEVC Faster, AV1 Slow, AV1, AV1 Fast, AV1 Faster, VP9 Slow, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2 Slow, AV2, AV2 Fast, VVC Slow, VVC, VVC Fast, SVT-AV1 M8, SVT-AV1 M10.

Direct BD-J vs x264 medium, metric-tuned (--tune psnr) on them, where it exists:

| Encoder | pool_ducks_take_off_pristine | ind_bwc_videotext_video190 | ind_personpath22_uid_vid_00212 |
|---|---:|---:|---:|
| VVC Slow | -60.0% | -40.0% | n/a |
| VVC | -58.7% | -35.3% | n/a |
| AV2 Slow | -55.2% | -37.4% | n/a |
| AV2 | -52.9% | -33.6% | n/a |
| HEVC Slow | -48.7% | -38.7% | n/a |
| AV1 Slow | -50.4% | -36.0% | n/a |
| HEVC | -44.5% | -29.3% | n/a |
| AV1 | -49.4% | -31.9% | n/a |
| VP9 Slow | -43.3% | -35.1% | n/a |
| VP9 | -44.4% | -30.4% | n/a |
| VVC Fast | -48.1% | -21.1% | n/a |
| AV1 Fast | -44.0% | -19.0% | n/a |
| AV2 Fast | -37.9% | -19.4% | n/a |
| HEVC Fast | -30.4% | -14.0% | n/a |
| VP9 Fast | -33.9% | -18.3% | n/a |
| SVT-AV1 M8 | n/a | -23.3% | n/a |
| AV1 Faster | -42.2% | -19.6% | n/a |
| VP9 Faster | -34.4% | -17.8% | n/a |
| HEVC Faster | -23.3% | -10.2% | n/a |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| SVT-AV1 M10 | n/a | +5.7% | n/a |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| H.264 Slow | +11.0% | -18.1% | -34.2% |
| x265 medium | -41.3% | +0.8% | -17.4% |
| H.264 | +6.4% | -18.0% | -32.2% |
| x265 superfast | -41.9% | +10.6% | -9.9% |
| VP9 Veryfast | -34.2% | -16.5% | n/a |
| libaom rt 8 | -5.9% | n/a | -16.9% |
| libaom rt 10 | -3.1% | n/a | -16.6% |
| SVT-AV1 M12 | -23.8% | +8.8% | -11.9% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 ultrafast | -41.9% | +20.2% | +5.7% |
| SVT-AV1 M13 | -23.5% | +10.1% | -11.3% |
| H.264 Fast | +16.9% | -8.0% | -24.3% |
| H.264 Faster | +40.5% | -6.4% | -16.6% |
| H.264 Veryfast | +31.5% | -3.3% | -12.5% |
| x264 medium, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9 | n/a | +8.6% | +19.5% |
| x264 medium | +15.9% | +2.6% | -13.2% |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 7 | +8.7% | -10.5% | +28.3% |
| x264 veryfast | +13.9% | +24.1% | -9.1% |
| openh264 | +109.2% | +26.1% | +7.2% |
| x264 ultrafast | +29.5% | +64.9% | +58.7% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 | openh264 | -32.2% | -32.3% | -35.2% | -43.5 … -24.0 | 209/209 |
| H.264 | x264 veryfast | -33.1% | -35.3% | -36.7% | -43.2 … -26.8 | 209/209 |
| H.264 | x264 medium | -26.6% | -30.6% | -29.6% | -39.8 … -17.7 | 209/209 |
| H.264 Fast | x264 veryfast | -24.6% | -27.0% | -29.7% | -33.9 … -18.3 | 209/209 |
| H.264 Fast | openh264 | -21.6% | -21.9% | -26.6% | -36.3 … -13.0 | 209/209 |
| H.264 Fast | H.264 | +14.3% | +14.4% | +12.4% | +10.7 … +18.9 | 209/209 |
| H.264 Faster | x264 ultrafast | -38.3% | -49.4% | -27.8% | -48.4 … -34.7 | 200/209 |
| H.264 Faster | H.264 Fast | +10.9% | +9.2% | +13.5% | +5.1 … +15.2 | 209/209 |
| H.264 Veryfast | x264 ultrafast | -34.3% | -45.8% | -24.8% | -46.2 … -31.3 | 201/209 |
| H.264 Veryfast | H.264 Faster | +7.7% | +7.7% | +6.5% | +4.9 … +9.5 | 209/209 |
| HEVC | x265 medium | -23.8% | -26.3% | -30.0% | -37.8 … -17.9 | 209/209 |
| HEVC Fast | x265 superfast | -13.1% | -15.5% | -12.9% | -27.1 … -6.7 | 209/209 |
| HEVC | SVT-AV1 M12 | -29.5% | -29.7% | -35.6% | -37.9 … -26.2 | 209/209 |
| HEVC Fast | HEVC | +18.7% | +19.2% | +25.5% | +15.5 … +24.9 | 209/209 |
| AV1 | openh264 | -51.4% | -53.1% | -51.7% | -64.0 … -42.2 | 203/209 |
| AV1 Fast | openh264 | -46.0% | -47.3% | -44.8% | -61.7 … -36.6 | 206/209 |
| AV1 | SVT-AV1 M8 | -8.5% | -9.7% | -17.6% | -22.0 … -5.2 | 209/209 |
| AV1 | SVT-AV1 M12 | -30.4% | -34.7% | -32.7% | -39.4 … -25.3 | 209/209 |
| AV1 Fast | SVT-AV1 M12 | -24.1% | -27.5% | -25.3% | -32.6 … -20.4 | 209/209 |
| AV1 | libaom rt 8 | -26.1% | -32.0% | -27.2% | -37.2 … -20.8 | 209/209 |
| AV1 Fast | AV1 | +10.1% | +11.9% | +11.9% | +5.9 … +14.0 | 209/209 |
| AV1 Faster | SVT-AV1 M12 | -15.1% | -14.0% | -19.8% | -27.4 … -9.6 | 209/209 |
| AV1 Faster | AV1 Fast | +12.6% | +20.4% | +7.4% | +6.8 … +16.8 | 209/209 |
| HEVC Faster | SVT-AV1 M13 | -10.1% | -8.1% | -14.1% | -18.4 … -3.0 | 209/209 |
| HEVC Faster | HEVC Fast | +12.8% | +15.6% | +11.4% | +8.0 … +18.8 | 209/209 |
| VP9 | libvpx VP9 rt 7 | -43.1% | -47.6% | -41.7% | -51.2 … -37.1 | 209/209 |
| VP9 Fast | libvpx VP9 rt 9 | -35.9% | -40.6% | -36.0% | -44.2 … -31.4 | 209/209 |
| VP9 Fast | VP9 | +15.7% | +21.2% | +13.7% | +11.1 … +22.1 | 209/209 |
| VP9 Faster | libvpx VP9 rt 9 | -30.6% | -34.0% | -31.3% | -41.6 … -24.3 | 209/209 |
| VP9 Faster | VP9 Fast | +8.3% | +11.3% | +6.9% | +3.6 … +13.6 | 209/209 |
| VP9 Veryfast | libvpx VP9 rt 9 | -20.7% | -24.4% | -24.3% | -34.7 … -14.5 | 208/209 |
| VP9 Veryfast | VP9 Faster | +16.0% | +15.5% | +9.4% | +6.2 … +20.8 | 209/209 |
| AV2 | SVT-AV1 M8 | -14.8% | -15.9% | -25.0% | -24.8 … -11.0 | 209/209 |
| AV2 | AV1 | -6.4% | -6.6% | -8.6% | -8.6 … -1.6 | 209/209 |
| AV2 Fast | AV2 | +21.4% | +27.1% | +23.9% | +15.4 … +29.3 | 209/209 |
| VVC | x265 medium | -31.9% | -34.4% | -36.6% | -42.7 … -24.3 | 209/209 |
| VVC | HEVC | -10.8% | -11.2% | -8.9% | -11.8 … -6.8 | 209/209 |
| VVC Fast | VVC | +23.0% | +28.2% | +24.5% | +17.5 … +30.1 | 209/209 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, HEVC Fast, AV1, AV1 Fast, AV1 Faster, AV2, VVC: 4a5b8dd1; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976); AV2 Fast, H.264 Fast, H.264 Faster, HEVC Faster, VP9 Fast, VP9 Faster, VP9 Veryfast, VVC Fast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow timed on 1139f610 (same streams), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.7 [-3.0, -0.5]%); x264 medium --tune ssim (-2.5 [-3.2, -1.7]%); x265 ultrafast --tune psnr (-0.7 [-1.1, -0.2]%); x265 superfast --tune psnr (-2.0 [-2.4, -1.6]%); x265 medium --tune psnr (-5.0 [-5.5, -4.4]%); libvpx VP9 rt 7 --tune ssim (-1.8 [-2.4, -1.0]%); libvpx VP9 rt 9 --tune ssim (-1.3 [-1.8, -0.8]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
