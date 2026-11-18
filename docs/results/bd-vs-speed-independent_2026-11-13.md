# Bitrate vs encode speed, held-out clips (independent_2026-11-13)

<sub>Updated 2026-11-18, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, AV1, AV1 Fast, AV2, VVC: 4a5b8dd1; HEVC Fast, HEVC Faster, AV1 Faster, VP9 Fast, AV2 Fast, VVC Fast: b33ba3bb; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976; AV2 Faster: c9e4ced3; AV1 Veryfast: d40f86b3); H.264 Fast, H.264 Faster, VP9 Faster, VP9 Veryfast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow, AV1, AV1 Fast, AV2, H.264, VVC timed on b33ba3bb (same streams)</sub>

Quality on the 209 clips of `independent_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_2026-11-13.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -51.2% | -53.0% | -54.4% | -60.5 … -45.0 | 24.3 | 0.91× | 92.7 | 0.82× | 207/207 | Pareto; Pareto x86 |
| **VVC** | -50.8% | -52.5% | -53.7% | -60.1 … -44.4 | 27.9 | 1.05× | 118.9 | 1.05× | 207/207 | Pareto; Pareto x86 |
| **AV2 Slow** | -50.5% | -54.8% | -52.9% | -58.6 … -44.7 | 21.0 | 0.79× | 77.8 | 0.69× | 207/207 |  |
| **AV2** | -49.1% | -53.7% | -51.6% | -57.2 … -43.0 | 35.7 | 1.34× | 152.1 | 1.35× | 207/207 | Pareto; Pareto x86 |
| **HEVC Slow** | -47.9% | -48.7% | -51.3% | -57.2 … -41.7 | 30.1 | 1.13× | 117.1 | 1.04× | 207/207 |  |
| **AV1 Slow** | -46.6% | -51.0% | -48.2% | -55.6 … -41.3 | 28.7 | 1.08× | 112.2 | 1.00× | 207/207 | 1 rate-invalid pts |
| **HEVC** | -46.1% | -47.5% | -50.1% | -55.5 … -40.1 | 48.7 | 1.83× | 195.6 | 1.73× | 207/207 | Pareto; Pareto x86 |
| **AV1** | -45.9% | -50.2% | -47.4% | -54.8 … -40.3 | 44.1 | 1.65× | 172.7 | 1.53× | 207/207 | 1 rate-invalid pts |
| **VP9 Slow** | -45.4% | -48.8% | -47.4% | -53.4 … -38.2 | 36.3 | 1.36× | 145.2 | 1.29× | 207/207 | 4 rate-invalid pts |
| **VP9** | -44.9% | -48.5% | -46.9% | -53.2 … -37.8 | 42.9 | 1.61× | 168.8 | 1.50× | 207/207 | 4 rate-invalid pts |
| **AV2 Fast** | -42.7% | -45.2% | -44.9% | -51.5 … -35.9 | 47.8 | 1.79× | 204.8 | 1.82× | 207/207 | Pareto x86 |
| **VVC Fast** | -41.9% | -41.5% | -44.9% | -51.8 … -33.5 | 53.7 | 2.01× | 212.9 | 1.89× | 207/207 | Pareto; Pareto x86 |
| **AV1 Fast** | -41.7% | -45.4% | -42.4% | -50.7 … -36.2 | 64.4 | 2.41× | 265.6 | 2.36× | 207/207 | Pareto; Pareto x86 |
| **HEVC Fast** | -41.1% | -42.7% | -43.5% | -50.5 … -34.8 | 75.8 | 2.84× | 303.8 | 2.69× | 207/207 | Pareto; Pareto x86 |
| **AV2 Faster** | -40.4% | -42.4% | -43.0% | -51.1 … -33.4 | 56.0 | 2.10× | 251.2 | 2.23× | 207/207 |  |
| **VP9 Fast** | -39.9% | -43.0% | -40.6% | -48.6 … -32.8 | 53.9 | 2.02× | 216.2 | 1.92× | 207/207 |  |
| **AV1 Faster** | -38.3% | -40.4% | -40.7% | -49.3 … -32.0 | 78.2 | 2.93× | 320.3 | 2.84× | 207/207 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -36.9% | -41.9% | -32.4% | -46.6 … -30.6 | 14.3 | 0.53× | 80.8 | 0.72× | 207/207 | 34 rate-invalid pts |
| **AV1 Veryfast** | -34.1% | -33.7% | -36.4% | -46.1 … -25.6 | 87.7 | 3.29× | 410.7 | 3.64× | 207/207 | Pareto; Pareto x86 |
| **VP9 Faster** | -34.1% | -33.7% | -37.7% | -44.9 … -25.3 | 90.4 | 3.39× | 355.5 | 3.15× | 207/207 | Pareto; 1 rate-invalid pts |
| **HEVC Faster** | -33.0% | -32.9% | -35.2% | -42.6 … -24.3 | 98.9 | 3.71× | 399.6 | 3.54× | 207/207 | Pareto |
| x265 medium, metric-tuned (--tune psnr) | -31.3% | -33.6% | -32.4% | -42.6 … -25.6 | 8.0 | 0.30× | 38.0 | 0.34× | 207/207 | 35 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M10 | -30.8% | -34.2% | -28.4% | -41.0 … -24.3 | 39.3 | 1.47× | 228.9 | 2.03× | 207/207 | 30 rate-invalid pts |
| x265 superfast, metric-tuned (--tune psnr) | -28.6% | -29.9% | -29.7% | -39.3 … -22.9 | 12.1 | 0.45× | 57.9 | 0.51× | 207/207 | 33 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Slow** | -28.3% | -32.4% | -31.8% | -40.5 … -19.7 | 46.4 | 1.74× | 138.1 | 1.23× | 207/207 |  |
| x265 medium | -28.1% | -28.0% | -27.2% | -38.8 … -22.3 | 8.0 | 0.30× | 38.0 | 0.34× | 207/207 | 37 rate-invalid pts |
| **H.264** | -27.4% | -30.5% | -31.4% | -39.9 … -17.9 | 75.5 | 2.83× | 237.6 | 2.11× | 207/207 |  |
| x265 superfast | -27.3% | -27.1% | -29.0% | -37.2 … -20.9 | 12.1 | 0.45× | 57.9 | 0.51× | 207/207 | 34 rate-invalid pts |
| **VP9 Veryfast** | -26.4% | -25.4% | -33.1% | -40.3 … -17.3 | 102.7 | 3.85× | 358.2 | 3.18× | 207/207 | Pareto |
| libaom rt 8 | -24.4% | -23.5% | -26.1% | -32.9 … -18.2 | 17.4 | 0.65× | 68.8 | 0.61× | 207/207 | 126 rate-invalid pts |
| libaom rt 10 | -23.7% | -24.7% | -24.1% | -31.8 … -16.6 | 22.3 | 0.83× | 90.5 | 0.80× | 207/207 | 136 rate-invalid pts |
| SVT-AV1 M12 | -22.4% | -23.7% | -21.7% | -35.1 … -15.5 | 70.9 | 2.67× | 408.7 | 3.62× | 207/207 | 26 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -22.0% | -22.6% | -23.6% | -34.9 … -12.4 | 17.1 | 0.64× | 85.1 | 0.75× | 207/207 | 33 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -21.7% | -20.8% | -23.7% | -34.0 … -11.8 | 17.1 | 0.64× | 85.1 | 0.75× | 207/207 | 33 rate-invalid pts |
| SVT-AV1 M13 | -20.6% | -21.3% | -20.0% | -32.2 … -13.9 | 79.2 | 2.96× | 445.4 | 3.95× | 207/207 | Pareto x86; 26 rate-invalid pts |
| **H.264 Fast** | -18.7% | -21.6% | -24.4% | -32.1 … -8.7 | 118.8 | 4.46× | 389.6 | 3.45× | 207/207 | Pareto |
| **H.264 Faster** | -11.5% | -15.2% | -16.3% | -27.5 … +5.1 | 142.9 | 5.36× | 454.8 | 4.03× | 207/207 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -10.0% | -10.2% | -19.1% | -12.8 … -6.3 | 26.7 | 1.00× | 112.8 | 1.00× | 207/207 | 49 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Veryfast** | -5.7% | -9.9% | -10.5% | -23.8 … +10.8 | 164.0 | 6.15× | 542.7 | 4.81× | 207/207 | Pareto; Pareto x86 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.5% | +5.1% | -6.1% | -9.1 … +6.6 | 47.8 | 1.78× | 204.4 | 1.81× | 207/207 | 221 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -0.4% | +5.4% | -5.6% | -8.8 … +8.6 | 47.8 | 1.78× | 204.4 | 1.81× | 207/207 | 222 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 207/207 | 46 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.4% | +6.3% | +0.9% | -6.5 … +11.1 | 30.0 | 1.12× | 122.7 | 1.09× | 207/207 | 185 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | +1.6% | -0.4% | -9.1% | -9.8 … +11.1 | 85.4 | 3.18× | 363.3 | 3.22× | 207/207 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +2.2% | +6.8% | -3.2% | -6.8 … +13.7 | 30.0 | 1.12× | 122.7 | 1.09× | 207/207 | 215 rate-invalid pts |
| x264 veryfast | +3.2% | +0.9% | +4.4% | -3.7 … +10.2 | 85.4 | 3.18× | 363.3 | 3.22× | 207/207 | 1 rate-invalid pts |
| openh264 | +13.6% | +9.3% | +12.4% | -15.2 … +49.1 | 111.5 | 4.17× | 524.7 | 4.65× | 207/207 | 353 rate-invalid pts |
| x264 ultrafast | +38.3% | +51.7% | +13.7% | +17.5 … +57.1 | 190.5 | 7.11× | 1017.0 | 9.02× | 207/207 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: pool_barscene_pristine, pool_bodycam_antare_antare_test, pool_bodycam_bwc_video118, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_crowd_person_uid_vid_00222, pool_ducks_take_off_pristine, pool_lowlight_cevo_inof_md_indoor_night_ohcam_0, pool_meva_hospital_g301, pool_static_chirla_chirla_seq_024_camera_7_2023, ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 2; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 22, crowd 57, drone 3, lowlight 15, pan 53, static 55, texture 2 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -28.3% | -29.0% |
| H.264 | -27.4% | -28.0% |
| H.264 Fast | -18.7% | -18.6% |
| H.264 Faster | -11.5% | -10.9% |
| H.264 Veryfast | -5.7% | -5.9% |
| HEVC Slow | -47.9% | -49.4% |
| HEVC | -46.1% | -47.7% |
| HEVC Fast | -41.1% | -42.4% |
| HEVC Faster | -33.0% | -33.8% |
| AV1 Slow | -46.6% | -49.0% |
| AV1 | -45.9% | -48.3% |
| AV1 Fast | -41.7% | -43.8% |
| AV1 Faster | -38.3% | -41.1% |
| AV1 Veryfast | -34.1% | -36.1% |
| VP9 Slow | -45.4% | -46.4% |
| VP9 | -44.9% | -46.0% |
| VP9 Fast | -39.9% | -41.1% |
| VP9 Faster | -34.1% | -34.7% |
| VP9 Veryfast | -26.4% | -28.0% |
| AV2 Slow | -50.5% | -52.0% |
| AV2 | -49.1% | -50.7% |
| AV2 Fast | -42.7% | -44.2% |
| AV2 Faster | -40.4% | -42.8% |
| VVC Slow | -51.2% | -52.5% |
| VVC | -50.8% | -52.0% |
| VVC Fast | -41.9% | -42.8% |
| openh264 | +13.6% | +16.8% |
| x264 ultrafast | +38.3% | +40.3% |
| x264 veryfast | +3.2% | +2.8% |
| x265 ultrafast | -21.7% | -22.5% |
| x265 superfast | -27.3% | -28.8% |
| x265 medium | -28.1% | -29.6% |
| SVT-AV1 M8 | -36.9% | -39.6% |
| SVT-AV1 M10 | -30.8% | -33.2% |
| SVT-AV1 M12 | -22.4% | -26.2% |
| SVT-AV1 M13 | -20.6% | -24.3% |
| libvpx VP9 rt 7 | +2.2% | +3.6% |
| libvpx VP9 rt 9 | -0.4% | -0.2% |
| libaom rt 8 | -24.4% | -25.0% |
| libaom rt 10 | -23.7% | -24.9% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +13.6% | +9.3% | +12.4% | +19.0% | 207/207 |  |
| openh264, adaptive quantisation on | +13.6% | +9.3% | +12.4% | +19.0% | 207/207 | -0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +38.3% | +51.7% | +13.7% | +49.4% | 207/207 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +38.3% | +51.8% | +13.6% | +49.7% | 207/207 | +0.1 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +34.7% | +49.0% | +27.3% | +27.8% | 207/207 | -0.3 [-1.9, +1.3] not distinguishable |
| x264 veryfast | +3.2% | +0.9% | +4.4% | +4.3% | 207/207 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +1.6% | -0.4% | -9.1% | +14.2% | 207/207 | -1.6 [-2.9, -0.4] better |
| x264 veryfast, metric-tuned (--tune ssim) | +4.1% | +2.3% | +6.9% | +2.9% | 207/207 | +0.9 [+0.3, +1.5] worse |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 207/207 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -10.0% | -10.2% | -19.1% | -0.8% | 207/207 | -10.0 [-11.1, -8.8] better |
| x264 medium, metric-tuned (--tune ssim) | -2.5% | -2.8% | -0.1% | -4.5% | 207/207 | -2.5 [-3.2, -1.7] better |
| x265 ultrafast | -21.7% | -20.8% | -23.7% | -20.7% | 207/207 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -22.0% | -22.6% | -23.6% | -19.9% | 207/207 | -0.6 [-1.0, -0.2] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -22.0% | -22.6% | -23.6% | -19.9% | 207/207 | -0.6 [-1.0, -0.2] better |
| x265 superfast | -27.3% | -27.1% | -29.0% | -25.8% | 207/207 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -28.6% | -29.9% | -29.7% | -26.1% | 207/207 | -2.0 [-2.4, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -28.6% | -29.9% | -29.7% | -26.1% | 207/207 | -2.0 [-2.4, -1.6] better |
| x265 medium | -28.1% | -28.0% | -27.2% | -29.3% | 207/207 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -31.3% | -33.6% | -32.4% | -27.9% | 207/207 | -4.9 [-5.4, -4.3] better |
| x265 medium, metric-tuned (--tune ssim) | -29.8% | -30.9% | -28.4% | -29.9% | 207/207 | -2.3 [-2.5, -2.0] better |
| libvpx VP9 rt 7 | +2.2% | +6.8% | -3.2% | +3.0% | 207/207 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +1.4% | +6.3% | +0.9% | -3.1% | 207/207 | -1.8 [-2.5, -1.1] better |
| libvpx VP9 rt 9 | -0.4% | +5.4% | -5.6% | -0.9% | 207/207 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | -1.5% | +5.1% | -6.1% | -3.4% | 207/207 | -1.3 [-1.8, -0.8] better |
| libaom rt 8 | -24.4% | -23.5% | -26.1% | -23.6% | 207/207 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -24.2% | -21.7% | -21.6% | -29.2% | 207/207 | +0.9 [+0.2, +1.6] worse |
| libaom rt 10 | -23.7% | -24.7% | -24.1% | -22.5% | 207/207 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -24.1% | -23.3% | -21.0% | -28.1% | 207/207 | +0.4 [-0.4, +1.1] not distinguishable |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -44.6% | -44.5% | -42.3% | -47.0% | 206/207 |
| **VVC** | -44.1% | -44.0% | -41.4% | -46.9% | 206/207 |
| **AV2 Slow** | -43.9% | -46.6% | -40.6% | -44.6% | 206/207 |
| **AV2** | -42.3% | -45.3% | -38.9% | -42.8% | 206/207 |
| **HEVC Slow** | -40.9% | -39.9% | -38.4% | -44.4% | 206/207 |
| **AV1 Slow** | -39.5% | -42.5% | -34.6% | -41.4% | 206/207 |
| **HEVC** | -39.0% | -38.4% | -37.0% | -41.5% | 206/207 |
| **AV1** | -38.6% | -41.6% | -33.6% | -40.7% | 206/207 |
| **VP9 Slow** | -38.1% | -40.1% | -33.5% | -40.8% | 206/207 |
| **VP9** | -37.6% | -39.7% | -32.8% | -40.1% | 206/207 |
| **AV2 Fast** | -35.1% | -35.7% | -30.5% | -39.1% | 206/207 |
| **VVC Fast** | -34.0% | -31.6% | -30.3% | -39.9% | 206/207 |
| **AV1 Fast** | -33.8% | -36.1% | -27.2% | -38.1% | 206/207 |
| **HEVC Fast** | -33.2% | -33.0% | -28.6% | -38.1% | 206/207 |
| **AV2 Faster** | -32.4% | -32.6% | -27.9% | -36.7% | 206/207 |
| **VP9 Fast** | -31.8% | -33.5% | -25.0% | -36.8% | 206/207 |
| **AV1 Faster** | -30.1% | -30.6% | -25.1% | -34.4% | 206/207 |
| SVT-AV1 M8 | -27.9% | -31.7% | -14.7% | -37.4% | 206/207 |
| **AV1 Veryfast** | -25.4% | -23.4% | -19.9% | -32.9% | 206/207 |
| **VP9 Faster** | -25.4% | -23.3% | -21.4% | -31.4% | 206/207 |
| **HEVC Faster** | -24.2% | -22.7% | -18.3% | -31.5% | 206/207 |
| x265 medium, metric-tuned (--tune psnr) | -22.4% | -23.3% | -15.2% | -28.7% | 207/207 |
| SVT-AV1 M10 | -21.0% | -22.8% | -9.6% | -30.5% | 206/207 |
| x265 medium, metric-tuned (--tune ssim) | -20.2% | -20.1% | -10.0% | -30.6% | 207/207 |
| **H.264 Slow** | -19.3% | -22.6% | -14.5% | -20.8% | 207/207 |
| x265 superfast, metric-tuned (--tune psnr) | -19.0% | -18.7% | -11.7% | -26.7% | 207/207 |
| x265 superfast, metric-tuned (--tune ssim) | -19.0% | -18.7% | -11.7% | -26.7% | 207/207 |
| x265 medium | -18.3% | -16.7% | -8.4% | -29.9% | 207/207 |
| **H.264** | -18.2% | -20.3% | -13.9% | -20.3% | 207/207 |
| x265 superfast | -17.5% | -15.5% | -10.7% | -26.4% | 207/207 |
| **VP9 Veryfast** | -17.1% | -14.8% | -15.6% | -20.9% | 206/207 |
| libaom rt 8 | -14.2% | -12.1% | -6.9% | -23.9% | 206/207 |
| libaom rt 8, metric-tuned (--tune ssim) | -13.6% | -10.2% | -1.5% | -29.4% | 206/207 |
| libaom rt 10, metric-tuned (--tune ssim) | -13.4% | -11.5% | -0.5% | -28.5% | 206/207 |
| libaom rt 10 | -13.2% | -13.0% | -4.2% | -22.9% | 206/207 |
| SVT-AV1 M12 | -11.9% | -11.7% | -1.7% | -22.2% | 207/207 |
| x265 ultrafast, metric-tuned (--tune psnr) | -11.6% | -10.7% | -3.9% | -20.3% | 207/207 |
| x265 ultrafast, metric-tuned (--tune ssim) | -11.6% | -10.7% | -3.9% | -20.3% | 207/207 |
| x265 ultrafast | -11.2% | -8.7% | -3.9% | -21.0% | 207/207 |
| SVT-AV1 M13 | -9.9% | -9.2% | +0.3% | -20.7% | 207/207 |
| **H.264 Fast** | -8.6% | -10.7% | -5.2% | -9.7% | 207/207 |
| **H.264 Faster** | -0.5% | -4.0% | +4.9% | -2.6% | 207/207 |
| **H.264 Veryfast** | +5.8% | +1.5% | +12.0% | +3.9% | 207/207 |
| x264 medium, metric-tuned (--tune ssim) | +10.2% | +8.8% | +25.1% | -3.4% | 207/207 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.5% | +19.7% | +18.1% | -3.4% | 207/207 |
| libvpx VP9 rt 9 | +12.7% | +20.3% | +18.8% | -0.9% | 207/207 |
| x264 medium | +12.8% | +12.2% | +25.0% | +1.3% | 207/207 |
| x264 veryfast, metric-tuned (--tune psnr) | +13.6% | +11.6% | +13.7% | +15.5% | 207/207 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.1% | +19.5% | +25.3% | -2.4% | 207/207 |
| libvpx VP9 rt 7 | +15.2% | +20.7% | +21.1% | +3.7% | 207/207 |
| x264 veryfast | +17.1% | +13.4% | +32.4% | +5.4% | 207/207 |
| x264 veryfast, metric-tuned (--tune ssim) | +18.2% | +15.1% | +35.6% | +4.0% | 207/207 |
| openh264 | +29.0% | +25.2% | +41.9% | +19.8% | 207/207 |
| openh264, adaptive quantisation on | +29.0% | +25.2% | +41.9% | +19.8% | 207/207 |
| x264 ultrafast, metric-tuned (--tune ssim) | +53.0% | +65.9% | +63.2% | +29.9% | 207/207 |
| x264 ultrafast | +54.8% | +68.1% | +44.1% | +52.1% | 207/207 |
| x264 ultrafast, metric-tuned (--tune psnr) | +54.8% | +68.2% | +43.9% | +52.4% | 207/207 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **pool_ducks_take_off_pristine** (texture): no reliable overlap for SVT-AV1 M8, SVT-AV1 M10, libvpx VP9 rt 9.
- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | pool_ducks_take_off_pristine | ind_bwc_videotext_video88 |
|---|---:|---:|
| VVC Slow | -67.8% | -65.6% |
| VVC | -67.9% | -65.3% |
| AV2 Slow | -62.7% | -65.1% |
| AV2 | -61.0% | -64.9% |
| HEVC Slow | -56.1% | -55.7% |
| AV1 Slow | -58.2% | -56.1% |
| HEVC | -52.7% | -56.4% |
| AV1 | -57.2% | -55.3% |
| VP9 Slow | -50.0% | -56.0% |
| VP9 | -50.6% | -55.3% |
| AV2 Fast | -52.6% | -66.5% |
| VVC Fast | -56.8% | -64.0% |
| AV1 Fast | -53.1% | -57.9% |
| HEVC Fast | -43.3% | -55.6% |
| AV2 Faster | -52.9% | -64.9% |
| VP9 Fast | -45.8% | -56.2% |
| AV1 Faster | -51.8% | -57.2% |
| SVT-AV1 M8 | n/a | -60.8% |
| AV1 Veryfast | -52.0% | -55.7% |
| VP9 Faster | -46.3% | -58.3% |
| HEVC Faster | -31.3% | -50.7% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a |
| SVT-AV1 M10 | n/a | -56.2% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a |
| H.264 Slow | -5.7% | -24.1% |
| x265 medium | -49.5% | -35.4% |
| H.264 | -9.7% | -23.5% |
| x265 superfast | -47.3% | -47.1% |
| VP9 Veryfast | -44.9% | -52.4% |
| libaom rt 8 | -18.7% | n/a |
| libaom rt 10 | -16.0% | -38.6% |
| SVT-AV1 M12 | -35.1% | -50.6% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 ultrafast | -47.2% | -45.2% |
| SVT-AV1 M13 | -34.7% | -49.0% |
| H.264 Fast | -0.6% | -21.7% |
| H.264 Faster | +20.6% | -16.9% |
| x264 medium, metric-tuned (--tune psnr) | n/a | n/a |
| H.264 Veryfast | +9.6% | -11.9% |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a | n/a |
| libvpx VP9 rt 9 | n/a | -19.2% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a |
| libvpx VP9 rt 7 | -6.0% | -13.6% |
| x264 veryfast | +2.7% | -3.5% |
| openh264 | +89.8% | +32.2% |
| x264 ultrafast | +15.0% | +10.5% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slow | openh264 | -33.2% | -34.5% | -35.6% | -44.7 … -25.3 | 209/209 |
| H.264 | openh264 | -32.2% | -32.3% | -35.2% | -43.5 … -24.0 | 209/209 |
| H.264 | x264 veryfast | -33.1% | -35.3% | -36.7% | -43.2 … -26.8 | 209/209 |
| H.264 | x264 medium | -26.6% | -30.6% | -29.6% | -39.8 … -17.7 | 209/209 |
| H.264 Fast | openh264 | -21.6% | -21.9% | -26.6% | -36.3 … -13.0 | 209/209 |
| H.264 Fast | x264 veryfast | -24.6% | -27.0% | -29.7% | -33.9 … -18.3 | 209/209 |
| H.264 Fast | H.264 | +14.3% | +14.4% | +12.4% | +10.7 … +18.9 | 209/209 |
| H.264 Faster | openh264 | -13.2% | -14.5% | -17.4% | -27.8 … -5.1 | 209/209 |
| H.264 Faster | x264 ultrafast | -38.3% | -49.4% | -27.8% | -48.4 … -34.7 | 200/209 |
| H.264 Faster | H.264 Fast | +10.9% | +9.2% | +13.5% | +5.1 … +15.2 | 209/209 |
| H.264 Veryfast | openh264 | -6.1% | -8.2% | -10.4% | -24.4 … +3.2 | 208/209 |
| H.264 Veryfast | x264 ultrafast | -34.3% | -45.8% | -24.8% | -46.2 … -31.3 | 201/209 |
| H.264 Veryfast | H.264 Faster | +7.7% | +7.7% | +6.5% | +4.9 … +9.5 | 209/209 |
| HEVC Slow | SVT-AV1 M12 | -32.7% | -32.3% | -37.5% | -40.5 … -29.3 | 209/209 |
| HEVC | SVT-AV1 M12 | -29.5% | -29.7% | -35.6% | -37.9 … -26.2 | 209/209 |
| HEVC | x265 medium | -23.8% | -26.3% | -30.0% | -37.8 … -17.9 | 209/209 |
| HEVC Fast | SVT-AV1 M12 | -22.0% | -22.1% | -25.7% | -29.4 … -19.8 | 209/209 |
| HEVC Fast | x265 superfast | -17.7% | -20.5% | -18.4% | -31.3 … -13.3 | 209/209 |
| HEVC Fast | HEVC | +11.9% | +11.7% | +16.5% | +8.6 … +16.0 | 209/209 |
| HEVC Faster | SVT-AV1 M12 | -8.5% | -6.1% | -12.8% | -16.9 … -1.9 | 209/209 |
| HEVC Faster | SVT-AV1 M13 | -11.1% | -9.1% | -15.1% | -19.2 … -5.4 | 209/209 |
| HEVC Faster | HEVC Fast | +18.8% | +22.5% | +18.9% | +12.8 … +27.1 | 209/209 |
| AV1 Slow | SVT-AV1 M12 | -31.4% | -35.7% | -33.8% | -40.3 … -26.5 | 209/209 |
| AV1 | SVT-AV1 M12 | -30.4% | -34.7% | -32.7% | -39.4 … -25.3 | 209/209 |
| AV1 | openh264 | -51.4% | -53.1% | -51.7% | -64.0 … -42.2 | 203/209 |
| AV1 | SVT-AV1 M8 | -8.5% | -9.7% | -17.6% | -22.0 … -5.2 | 209/209 |
| AV1 | libaom rt 8 | -26.1% | -32.0% | -27.2% | -37.2 … -20.8 | 209/209 |
| AV1 Fast | SVT-AV1 M12 | -24.1% | -27.5% | -25.3% | -32.6 … -20.4 | 209/209 |
| AV1 Fast | openh264 | -46.0% | -47.3% | -44.8% | -61.7 … -36.6 | 206/209 |
| AV1 Fast | AV1 | +10.1% | +11.9% | +11.9% | +5.9 … +14.0 | 209/209 |
| AV1 Faster | SVT-AV1 M12 | -18.7% | -20.0% | -22.8% | -30.0 … -15.1 | 209/209 |
| AV1 Faster | AV1 Fast | +6.8% | +10.3% | +3.2% | +0.9 … +9.6 | 209/209 |
| AV1 Veryfast | SVT-AV1 M12 | -12.3% | -9.6% | -16.1% | -25.6 … -6.5 | 209/209 |
| AV1 Veryfast | SVT-AV1 M13 | -15.0% | -12.8% | -18.6% | -27.9 … -8.6 | 209/209 |
| AV1 Veryfast | AV1 Faster | +8.9% | +14.7% | +9.3% | +4.1 … +18.0 | 209/209 |
| VP9 Slow | HEVC Slow | +6.1% | +0.5% | +9.8% | +3.2 … +11.6 | 209/209 |
| VP9 | HEVC | +2.9% | -2.1% | +7.9% | +0.2 … +9.2 | 209/209 |
| VP9 | libvpx VP9 rt 7 | -43.1% | -47.6% | -41.7% | -51.2 … -37.1 | 209/209 |
| VP9 Fast | HEVC Fast | +2.7% | -1.1% | +6.5% | -1.6 … +8.2 | 209/209 |
| VP9 Fast | libvpx VP9 rt 9 | -37.9% | -44.5% | -35.5% | -47.3 … -33.5 | 208/209 |
| VP9 Fast | VP9 | +11.1% | +12.1% | +14.1% | +6.8 … +15.1 | 209/209 |
| VP9 Faster | HEVC Faster | -2.7% | -3.3% | -4.2% | -5.2 … +2.2 | 209/209 |
| VP9 Faster | libvpx VP9 rt 9 | -30.6% | -34.0% | -31.3% | -41.6 … -24.3 | 209/209 |
| VP9 Faster | VP9 Fast | +12.8% | +19.9% | +6.6% | +8.6 … +18.1 | 209/209 |
| VP9 Veryfast | HEVC Faster | +12.3% | +11.1% | +4.3% | +3.8 … +18.7 | 209/209 |
| VP9 Veryfast | libvpx VP9 rt 9 | -20.7% | -24.4% | -24.3% | -34.7 … -14.5 | 208/209 |
| VP9 Veryfast | VP9 Faster | +16.0% | +15.5% | +9.4% | +6.2 … +20.8 | 209/209 |
| AV2 Slow | AV1 Slow | -8.1% | -7.8% | -9.8% | -10.2 … -3.3 | 209/209 |
| AV2 | AV1 | -6.4% | -6.6% | -8.6% | -8.6 … -1.6 | 209/209 |
| AV2 | SVT-AV1 M8 | -14.8% | -15.9% | -25.0% | -24.8 … -11.0 | 209/209 |
| AV2 Fast | AV1 Fast | -1.9% | +1.2% | -4.4% | -3.5 … +3.1 | 209/209 |
| AV2 Fast | AV2 | +15.8% | +21.8% | +17.0% | +12.3 … +21.2 | 209/209 |
| AV2 Faster | AV1 Faster | -3.8% | -2.7% | -4.2% | -6.4 … -0.0 | 209/209 |
| AV2 Faster | SVT-AV1 M12 | -22.7% | -23.0% | -26.5% | -31.8 … -19.6 | 209/209 |
| AV2 Faster | AV2 Fast | +5.1% | +6.5% | +3.6% | +0.2 … +5.8 | 209/209 |
| VVC Slow | HEVC Slow | -8.1% | -9.3% | -7.9% | -10.0 … -3.6 | 209/209 |
| VVC | HEVC | -10.8% | -11.2% | -8.9% | -11.8 … -6.8 | 209/209 |
| VVC | x265 medium | -31.9% | -34.4% | -36.6% | -42.7 … -24.3 | 209/209 |
| VVC Fast | HEVC Fast | -2.3% | +1.4% | -3.0% | -4.6 … +2.3 | 209/209 |
| VVC Fast | VVC | +23.1% | +28.2% | +24.8% | +17.6 … +29.4 | 209/209 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, AV1, AV1 Fast, AV2, VVC: 4a5b8dd1; HEVC Fast, HEVC Faster, AV1 Faster, VP9 Fast, AV2 Fast, VVC Fast: b33ba3bb; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976; AV2 Faster: c9e4ced3; AV1 Veryfast: d40f86b3); H.264 Fast, H.264 Faster, VP9 Faster, VP9 Veryfast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow, AV1, AV1 Fast, AV2, H.264, VVC timed on b33ba3bb (same streams), its presets), openh264
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
- **x86 speed (`bd-vs-speed-independent_2026-11-13-x86.svg`, `.csv`):** the same streams (same quantizers and
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.6 [-2.9, -0.4]%); x264 medium --tune psnr (-10.0 [-11.1, -8.8]%); x265 ultrafast --tune psnr (-0.6 [-1.0, -0.2]%); x265 superfast --tune psnr (-2.0 [-2.4, -1.6]%); x265 medium --tune psnr (-4.9 [-5.4, -4.3]%); libvpx VP9 rt 7 --tune ssim (-1.8 [-2.5, -1.1]%); libvpx VP9 rt 9 --tune ssim (-1.3 [-1.8, -0.8]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
