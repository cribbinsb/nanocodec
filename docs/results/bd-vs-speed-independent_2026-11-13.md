# Bitrate vs encode speed, held-out clips (independent_2026-11-13)

<sub>Updated 2026-11-20, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Fast, H.264 Faster, H.264 Veryfast, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster: 20d31869; H.264: c9cff89f)</sub>

Quality on the 209 clips of `independent_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_2026-11-13.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -52.7% | -54.4% | -55.9% | -61.3 … -46.1 | 16.0 | 0.60× | 64.4 | 0.57× | 206/206 | Pareto; Pareto x86 |
| **AV2 Slow** | -51.6% | -55.9% | -54.0% | -60.1 … -46.7 | 16.1 | 0.60× | 62.3 | 0.55× | 206/206 | Pareto |
| **VVC** | -51.3% | -53.2% | -54.6% | -60.1 … -44.9 | 29.8 | 1.12× | 121.0 | 1.07× | 206/206 | Pareto; Pareto x86 |
| **HEVC Slow** | -49.9% | -51.4% | -52.8% | -59.4 … -43.5 | 19.0 | 0.71× | 76.9 | 0.68× | 206/206 |  |
| **AV2** | -49.2% | -53.8% | -52.0% | -58.9 … -43.1 | 39.2 | 1.47× | 170.1 | 1.51× | 206/206 | Pareto; Pareto x86 |
| **AV1 Slow** | -48.6% | -52.6% | -49.2% | -56.8 … -43.2 | 26.6 | 1.00× | 113.3 | 1.00× | 206/206 | 1 rate-invalid pts |
| **HEVC** | -47.3% | -49.0% | -50.9% | -57.7 … -40.6 | 45.3 | 1.70× | 171.9 | 1.52× | 206/206 | Pareto; Pareto x86 |
| **VVC Fast** | -46.4% | -48.0% | -49.7% | -56.8 … -38.7 | 53.2 | 1.99× | 205.2 | 1.82× | 206/206 | Pareto; Pareto x86 |
| **AV1** | -46.3% | -50.6% | -47.7% | -56.0 … -40.8 | 45.7 | 1.71× | 181.4 | 1.61× | 206/206 | 1 rate-invalid pts |
| **VP9 Slow** | -46.1% | -49.5% | -47.8% | -55.1 … -38.9 | 26.7 | 1.00× | 100.0 | 0.89× | 206/206 | 3 rate-invalid pts |
| **VP9** | -45.4% | -48.9% | -47.3% | -54.3 … -38.1 | 42.1 | 1.58× | 171.0 | 1.52× | 206/206 | 4 rate-invalid pts |
| **AV2 Fast** | -44.5% | -47.8% | -45.4% | -55.0 … -38.3 | 51.8 | 1.94× | 230.8 | 2.05× | 206/206 | Pareto x86 |
| **HEVC Fast** | -43.4% | -44.9% | -45.3% | -53.5 … -36.7 | 72.0 | 2.70× | 293.2 | 2.60× | 206/206 | Pareto; Pareto x86 |
| **AV1 Fast** | -42.8% | -46.3% | -43.8% | -52.7 … -37.0 | 66.2 | 2.48× | 267.6 | 2.37× | 206/206 |  |
| **AV2 Faster** | -42.7% | -45.8% | -43.6% | -53.9 … -35.7 | 62.9 | 2.36× | 274.8 | 2.44× | 206/206 |  |
| **VP9 Fast** | -40.5% | -43.5% | -41.2% | -49.6 … -33.2 | 55.8 | 2.09× | 233.7 | 2.07× | 206/206 | 1 rate-invalid pts |
| **AV1 Faster** | -39.6% | -41.2% | -42.6% | -52.3 … -33.9 | 79.4 | 2.98× | 323.9 | 2.87× | 206/206 | Pareto; Pareto x86 |
| **HEVC Faster** | -37.9% | -39.2% | -40.1% | -48.9 … -30.1 | 90.3 | 3.38× | 364.2 | 3.23× | 206/206 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -37.1% | -42.0% | -32.6% | -46.8 … -30.7 | 14.3 | 0.53× | 80.8 | 0.72× | 206/206 | 38 rate-invalid pts |
| **VP9 Faster** | -36.7% | -36.8% | -39.3% | -48.7 … -28.1 | 92.3 | 3.46× | 368.7 | 3.27× | 206/206 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -34.5% | -34.0% | -37.0% | -46.4 … -25.7 | 96.5 | 3.62× | 431.4 | 3.83× | 206/206 | Pareto; Pareto x86 |
| **VP9 Veryfast** | -33.5% | -34.3% | -36.6% | -44.8 … -25.7 | 95.8 | 3.59× | 385.9 | 3.42× | 206/206 |  |
| **H.264 Slow** | -32.3% | -36.2% | -36.0% | -43.7 … -23.6 | 37.2 | 1.39× | 115.3 | 1.02× | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) | -31.5% | -33.7% | -32.6% | -42.7 … -25.8 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 38 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M10 | -31.0% | -34.3% | -28.6% | -42.1 … -24.3 | 39.3 | 1.47× | 228.9 | 2.03× | 206/206 | 34 rate-invalid pts |
| **H.264** | -30.3% | -33.7% | -34.6% | -41.9 … -21.7 | 66.4 | 2.49× | 202.2 | 1.79× | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) | -28.8% | -30.1% | -29.9% | -39.6 … -22.9 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 36 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 medium | -28.4% | -28.2% | -27.4% | -39.1 … -22.4 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 40 rate-invalid pts |
| x265 superfast | -27.5% | -27.3% | -29.2% | -37.8 … -21.1 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 37 rate-invalid pts |
| libaom rt 8 | -24.4% | -23.4% | -26.1% | -32.9 … -18.1 | 17.4 | 0.65× | 68.8 | 0.61× | 206/206 | 137 rate-invalid pts |
| libaom rt 10 | -23.8% | -24.6% | -24.1% | -31.8 … -16.6 | 22.3 | 0.83× | 90.5 | 0.80× | 206/206 | 146 rate-invalid pts |
| **H.264 Fast** | -22.7% | -25.5% | -27.4% | -36.3 … -11.7 | 119.5 | 4.48× | 370.1 | 3.28× | 206/206 | Pareto |
| SVT-AV1 M12 | -22.6% | -23.8% | -21.9% | -35.5 … -15.6 | 70.9 | 2.67× | 408.7 | 3.62× | 206/206 | 29 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -22.4% | -22.9% | -23.9% | -35.5 … -13.2 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 36 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -22.1% | -21.1% | -24.0% | -34.4 … -12.3 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 36 rate-invalid pts |
| SVT-AV1 M13 | -20.8% | -21.4% | -20.2% | -32.7 … -14.5 | 79.2 | 2.96× | 445.4 | 3.95× | 206/206 | Pareto x86; 29 rate-invalid pts |
| **H.264 Faster** | -15.1% | -18.2% | -20.0% | -31.6 … +2.5 | 146.9 | 5.51× | 460.1 | 4.08× | 206/206 | Pareto; Pareto x86 |
| **H.264 Veryfast** | -11.4% | -15.5% | -15.8% | -29.0 … +6.0 | 164.1 | 6.15× | 520.2 | 4.61× | 206/206 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -10.0% | -10.1% | -19.0% | -12.6 … -6.3 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 58 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.3% | +5.3% | -6.0% | -9.0 … +6.6 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 234 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -0.2% | +5.6% | -5.5% | -8.6 … +8.6 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 235 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 53 rate-invalid pts |
| x264 veryfast, metric-tuned (--tune psnr) | +1.3% | -0.7% | -9.2% | -10.6 … +10.2 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.5% | +6.5% | +1.0% | -6.5 … +11.0 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 198 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +2.4% | +7.1% | -3.1% | -6.6 … +13.5 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 228 rate-invalid pts |
| x264 veryfast | +2.8% | +0.4% | +4.1% | -4.9 … +9.4 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts |
| openh264 | +13.6% | +9.4% | +12.4% | -15.2 … +50.4 | 111.5 | 4.17× | 524.7 | 4.65× | 206/206 | Pareto x86; 367 rate-invalid pts |
| x264 ultrafast | +37.6% | +50.7% | +13.4% | +15.1 … +56.7 | 190.5 | 7.11× | 1017.0 | 9.02× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: pool_barscene_pristine, pool_bodycam_antare_antare_test, pool_bodycam_bwc_video118, pool_chad_1_093, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_crowd_person_uid_vid_00222, pool_ducks_take_off_pristine, pool_lowlight_cevo_inof_md_indoor_night_ohcam_0, pool_meva_hospital_g301, pool_static_chirla_chirla_seq_024_camera_7_2023, ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bdd100k_mot_bdd_00d9e313_7d75bb18, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video317, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_chirla_chirla_seq_026_camera_3_2023_12_05_12_12_13, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00212, ind_personpath22_uid_vid_00214, ind_personpath22_uid_vid_00216, ind_personpath22_uid_vid_00231, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 3; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 22, crowd 57, drone 3, lowlight 15, pan 53, static 54, texture 2 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -32.3% | -32.6% |
| H.264 | -30.3% | -30.6% |
| H.264 Fast | -22.7% | -23.0% |
| H.264 Faster | -15.1% | -14.8% |
| H.264 Veryfast | -11.4% | -11.6% |
| HEVC Slow | -49.9% | -51.5% |
| HEVC | -47.3% | -48.8% |
| HEVC Fast | -43.4% | -44.8% |
| HEVC Faster | -37.9% | -39.2% |
| AV1 Slow | -48.6% | -50.6% |
| AV1 | -46.3% | -48.5% |
| AV1 Fast | -42.8% | -45.0% |
| AV1 Faster | -39.6% | -42.6% |
| AV1 Veryfast | -34.5% | -36.6% |
| VP9 Slow | -46.1% | -47.1% |
| VP9 | -45.4% | -46.5% |
| VP9 Fast | -40.5% | -41.7% |
| VP9 Faster | -36.7% | -37.9% |
| VP9 Veryfast | -33.5% | -34.8% |
| AV2 Slow | -51.6% | -53.5% |
| AV2 | -49.2% | -51.0% |
| AV2 Fast | -44.5% | -46.8% |
| AV2 Faster | -42.7% | -45.2% |
| VVC Slow | -52.7% | -53.9% |
| VVC | -51.3% | -52.5% |
| VVC Fast | -46.4% | -47.8% |
| openh264 | +13.6% | +17.1% |
| x264 ultrafast | +37.6% | +39.3% |
| x264 veryfast | +2.8% | +2.3% |
| x265 ultrafast | -22.1% | -23.1% |
| x265 superfast | -27.5% | -29.1% |
| x265 medium | -28.4% | -30.0% |
| SVT-AV1 M8 | -37.1% | -39.9% |
| SVT-AV1 M10 | -31.0% | -33.5% |
| SVT-AV1 M12 | -22.6% | -26.4% |
| SVT-AV1 M13 | -20.8% | -24.6% |
| libvpx VP9 rt 7 | +2.4% | +3.7% |
| libvpx VP9 rt 9 | -0.2% | -0.0% |
| libaom rt 8 | -24.4% | -25.0% |
| libaom rt 10 | -23.8% | -24.9% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +13.6% | +9.4% | +12.4% | +19.1% | 206/206 |  |
| openh264, adaptive quantisation on | +13.7% | +9.4% | +12.4% | +19.1% | 206/206 | +0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +37.6% | +50.7% | +13.4% | +48.6% | 206/206 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +37.6% | +50.8% | +13.3% | +48.9% | 206/206 | +0.1 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +34.0% | +47.8% | +26.8% | +27.3% | 206/206 | -0.4 [-2.0, +1.2] not distinguishable |
| x264 veryfast | +2.8% | +0.4% | +4.1% | +4.0% | 206/206 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +1.3% | -0.7% | -9.2% | +13.8% | 206/206 | -1.6 [-2.8, -0.3] better |
| x264 veryfast, metric-tuned (--tune ssim) | +3.7% | +1.8% | +6.6% | +2.6% | 206/206 | +0.9 [+0.3, +1.5] worse |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 206/206 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -10.0% | -10.1% | -19.0% | -0.8% | 206/206 | -10.0 [-11.1, -8.9] better |
| x264 medium, metric-tuned (--tune ssim) | -2.5% | -2.8% | -0.1% | -4.5% | 206/206 | -2.5 [-3.2, -1.7] better |
| x265 ultrafast | -22.1% | -21.1% | -24.0% | -21.1% | 206/206 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -22.4% | -22.9% | -23.9% | -20.3% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -22.4% | -22.9% | -23.9% | -20.3% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 superfast | -27.5% | -27.3% | -29.2% | -26.1% | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -28.8% | -30.1% | -29.9% | -26.4% | 206/206 | -2.0 [-2.3, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -28.8% | -30.1% | -29.9% | -26.4% | 206/206 | -2.0 [-2.3, -1.6] better |
| x265 medium | -28.4% | -28.2% | -27.4% | -29.5% | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -31.5% | -33.7% | -32.6% | -28.2% | 206/206 | -4.9 [-5.4, -4.3] better |
| x265 medium, metric-tuned (--tune ssim) | -29.9% | -31.1% | -28.6% | -30.2% | 206/206 | -2.2 [-2.5, -2.0] better |
| libvpx VP9 rt 7 | +2.4% | +7.1% | -3.1% | +3.1% | 206/206 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +1.5% | +6.5% | +1.0% | -3.0% | 206/206 | -1.9 [-2.5, -1.0] better |
| libvpx VP9 rt 9 | -0.2% | +5.6% | -5.5% | -0.9% | 206/206 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | -1.3% | +5.3% | -6.0% | -3.4% | 206/206 | -1.3 [-1.7, -0.8] better |
| libaom rt 8 | -24.4% | -23.4% | -26.1% | -23.6% | 206/206 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -24.2% | -21.6% | -21.7% | -29.2% | 206/206 | +0.9 [+0.2, +1.6] worse |
| libaom rt 10 | -23.8% | -24.6% | -24.1% | -22.5% | 206/206 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -24.1% | -23.2% | -21.0% | -28.1% | 206/206 | +0.4 [-0.4, +1.1] not distinguishable |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -46.3% | -46.2% | -44.3% | -48.4% | 205/206 |
| **AV2 Slow** | -45.3% | -48.0% | -42.1% | -45.8% | 206/206 |
| **VVC** | -44.8% | -44.7% | -42.6% | -47.1% | 206/206 |
| **HEVC Slow** | -43.2% | -43.0% | -40.3% | -46.4% | 206/206 |
| **AV2** | -42.6% | -45.5% | -39.4% | -42.9% | 206/206 |
| **AV1 Slow** | -41.7% | -44.4% | -36.0% | -44.7% | 206/206 |
| **HEVC** | -40.3% | -40.1% | -38.0% | -42.7% | 206/206 |
| **VVC Fast** | -39.2% | -38.7% | -36.5% | -42.4% | 206/206 |
| **AV1** | -39.1% | -42.1% | -34.0% | -41.2% | 206/206 |
| **VP9 Slow** | -38.9% | -41.0% | -33.9% | -41.7% | 206/206 |
| **VP9** | -38.2% | -40.3% | -33.3% | -40.9% | 206/206 |
| **AV2 Fast** | -37.0% | -38.6% | -30.9% | -41.4% | 206/206 |
| **HEVC Fast** | -35.7% | -35.6% | -30.8% | -40.8% | 206/206 |
| **AV1 Fast** | -35.1% | -37.2% | -29.0% | -39.1% | 206/206 |
| **AV2 Faster** | -34.9% | -36.4% | -28.7% | -39.7% | 206/206 |
| **VP9 Fast** | -32.5% | -34.2% | -25.7% | -37.5% | 206/206 |
| **AV1 Faster** | -31.6% | -31.6% | -27.6% | -35.6% | 206/206 |
| **HEVC Faster** | -29.6% | -29.4% | -24.4% | -35.1% | 206/206 |
| **VP9 Faster** | -28.3% | -26.9% | -23.4% | -34.8% | 206/206 |
| SVT-AV1 M8 | -28.2% | -31.9% | -15.0% | -37.6% | 206/206 |
| **AV1 Veryfast** | -26.0% | -23.9% | -20.7% | -33.3% | 206/206 |
| **VP9 Veryfast** | -24.8% | -24.2% | -20.0% | -30.1% | 206/206 |
| **H.264 Slow** | -23.8% | -26.8% | -19.8% | -24.7% | 206/206 |
| x265 medium, metric-tuned (--tune psnr) | -22.7% | -23.6% | -15.5% | -28.9% | 206/206 |
| **H.264** | -21.5% | -24.0% | -17.9% | -22.6% | 206/206 |
| SVT-AV1 M10 | -21.2% | -23.0% | -9.9% | -30.8% | 206/206 |
| x265 medium, metric-tuned (--tune ssim) | -20.5% | -20.4% | -10.3% | -30.8% | 206/206 |
| x265 superfast, metric-tuned (--tune psnr) | -19.3% | -19.0% | -11.9% | -27.0% | 206/206 |
| x265 superfast, metric-tuned (--tune ssim) | -19.3% | -19.0% | -11.9% | -27.0% | 206/206 |
| x265 medium | -18.7% | -17.1% | -8.8% | -30.2% | 206/206 |
| x265 superfast | -17.9% | -15.8% | -11.0% | -26.7% | 206/206 |
| libaom rt 8 | -14.2% | -12.1% | -7.0% | -24.0% | 205/206 |
| libaom rt 8, metric-tuned (--tune ssim) | -13.6% | -10.3% | -1.6% | -29.5% | 205/206 |
| libaom rt 10, metric-tuned (--tune ssim) | -13.5% | -11.6% | -0.6% | -28.6% | 205/206 |
| libaom rt 10 | -13.3% | -13.0% | -4.3% | -23.0% | 205/206 |
| **H.264 Fast** | -13.0% | -15.0% | -8.9% | -15.2% | 206/206 |
| SVT-AV1 M12 | -12.1% | -11.9% | -2.0% | -22.4% | 206/206 |
| x265 ultrafast, metric-tuned (--tune psnr) | -12.1% | -11.2% | -4.4% | -20.7% | 206/206 |
| x265 ultrafast, metric-tuned (--tune ssim) | -12.1% | -11.2% | -4.4% | -20.7% | 206/206 |
| x265 ultrafast | -11.7% | -9.2% | -4.4% | -21.4% | 206/206 |
| SVT-AV1 M13 | -10.2% | -9.5% | -0.1% | -20.9% | 206/206 |
| **H.264 Faster** | -4.7% | -7.4% | +0.3% | -6.9% | 206/206 |
| **H.264 Veryfast** | -0.5% | -4.5% | +5.4% | -2.3% | 206/206 |
| x264 medium, metric-tuned (--tune ssim) | +10.1% | +8.6% | +25.1% | -3.4% | 206/206 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.5% | +19.8% | +18.2% | -3.4% | 206/206 |
| x264 medium | +12.7% | +12.0% | +24.9% | +1.3% | 206/206 |
| libvpx VP9 rt 9 | +12.8% | +20.4% | +18.8% | -0.9% | 206/206 |
| x264 veryfast, metric-tuned (--tune psnr) | +13.2% | +11.1% | +13.5% | +15.0% | 206/206 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.2% | +19.7% | +25.4% | -2.4% | 206/206 |
| libvpx VP9 rt 7 | +15.3% | +20.9% | +21.2% | +3.7% | 206/206 |
| x264 veryfast | +16.6% | +12.7% | +32.0% | +5.0% | 206/206 |
| x264 veryfast, metric-tuned (--tune ssim) | +17.7% | +14.4% | +35.2% | +3.6% | 206/206 |
| openh264 | +29.0% | +25.3% | +41.9% | +19.9% | 206/206 |
| openh264, adaptive quantisation on | +29.0% | +25.3% | +41.9% | +19.9% | 206/206 |
| x264 ultrafast, metric-tuned (--tune ssim) | +52.0% | +64.3% | +62.4% | +29.2% | 206/206 |
| x264 ultrafast | +53.9% | +66.7% | +43.6% | +51.3% | 206/206 |
| x264 ultrafast, metric-tuned (--tune psnr) | +53.9% | +66.8% | +43.4% | +51.5% | 206/206 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **pool_ducks_take_off_pristine** (texture): no reliable overlap for SVT-AV1 M8, SVT-AV1 M10, libvpx VP9 rt 9.
- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.
- **ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508** (static): no reliable overlap for AV2 Slow, AV2.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | pool_ducks_take_off_pristine | ind_bwc_videotext_video88 | ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508 |
|---|---:|---:|---:|
| VVC Slow | -69.1% | -66.8% | -65.3% |
| AV2 Slow | -64.6% | -65.2% | n/a |
| VVC | -66.9% | -66.5% | -65.6% |
| HEVC Slow | -54.3% | -61.0% | -68.4% |
| AV2 | -60.7% | -64.7% | n/a |
| AV1 Slow | -60.2% | -56.5% | -67.0% |
| HEVC | -51.1% | -59.1% | -66.1% |
| VVC Fast | -55.8% | -63.4% | -61.9% |
| AV1 | -57.1% | -55.9% | -64.9% |
| VP9 Slow | -50.3% | -55.9% | -65.8% |
| VP9 | -50.6% | -55.3% | -64.2% |
| AV2 Fast | -53.5% | -65.5% | -66.0% |
| HEVC Fast | -53.5% | -56.0% | -63.1% |
| AV1 Fast | -52.9% | -58.1% | -65.0% |
| AV2 Faster | -52.7% | -64.7% | -66.4% |
| VP9 Fast | -45.7% | -59.5% | -59.6% |
| AV1 Faster | -48.8% | -56.6% | -64.5% |
| HEVC Faster | -47.7% | -52.6% | -59.7% |
| SVT-AV1 M8 | n/a | -60.8% | -55.6% |
| VP9 Faster | -47.5% | -55.9% | -57.4% |
| AV1 Veryfast | -52.2% | -55.8% | -62.9% |
| VP9 Veryfast | -49.3% | -51.6% | -56.1% |
| H.264 Slow | -18.9% | -24.1% | -61.9% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| SVT-AV1 M10 | n/a | -56.2% | -52.5% |
| H.264 | -11.4% | -22.1% | -60.6% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 medium | -49.5% | -35.4% | -40.3% |
| x265 superfast | -47.3% | -47.1% | -42.3% |
| libaom rt 8 | -18.7% | n/a | -22.4% |
| libaom rt 10 | -16.0% | -38.6% | -32.2% |
| H.264 Fast | -1.9% | -22.4% | -58.4% |
| SVT-AV1 M12 | -35.1% | -50.6% | -53.7% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 ultrafast | -47.2% | -45.2% | -41.7% |
| SVT-AV1 M13 | -34.7% | -49.0% | -52.3% |
| H.264 Faster | -8.3% | -19.0% | -57.7% |
| H.264 Veryfast | +0.1% | -15.3% | -54.7% |
| x264 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9 | n/a | -19.2% | -13.7% |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 7 | -6.0% | -13.6% | +24.9% |
| x264 veryfast | +2.7% | -3.5% | -22.7% |
| openh264 | +89.8% | +32.2% | -66.5% |
| x264 ultrafast | +15.0% | +10.5% | -10.6% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slow | openh264 | -37.7% | -38.6% | -40.4% | -47.4 … -30.1 | 209/209 |
| H.264 | openh264 | -35.4% | -35.9% | -38.8% | -46.1 … -26.4 | 209/209 |
| H.264 | x264 veryfast | -35.7% | -37.7% | -40.5% | -44.9 … -30.1 | 209/209 |
| H.264 | x264 medium | -29.5% | -33.2% | -33.6% | -42.0 … -21.7 | 209/209 |
| H.264 Fast | openh264 | -26.6% | -26.6% | -30.4% | -39.5 … -17.1 | 209/209 |
| H.264 Fast | x264 veryfast | -28.4% | -30.7% | -32.2% | -37.7 … -20.8 | 209/209 |
| H.264 Fast | H.264 | +12.8% | +12.7% | +13.9% | +8.2 … +16.4 | 209/209 |
| H.264 Faster | openh264 | -17.6% | -18.1% | -21.9% | -29.7 … -10.3 | 209/209 |
| H.264 Faster | x264 ultrafast | -42.3% | -51.5% | -34.6% | -53.2 … -37.1 | 198/209 |
| H.264 Faster | H.264 Fast | +11.1% | +10.8% | +10.4% | +5.7 … +16.6 | 209/209 |
| H.264 Veryfast | openh264 | -12.6% | -14.4% | -16.8% | -27.9 … -4.7 | 209/209 |
| H.264 Veryfast | x264 ultrafast | -39.2% | -49.9% | -30.3% | -50.5 … -34.8 | 201/209 |
| H.264 Veryfast | H.264 Faster | +5.9% | +3.9% | +6.8% | +2.7 … +6.8 | 209/209 |
| HEVC Slow | SVT-AV1 M12 | -36.2% | -36.5% | -40.3% | -42.5 … -32.0 | 209/209 |
| HEVC | SVT-AV1 M12 | -31.6% | -32.2% | -37.3% | -39.1 … -27.9 | 209/209 |
| HEVC | x265 medium | -25.9% | -28.8% | -31.7% | -39.1 … -18.3 | 209/209 |
| HEVC Fast | SVT-AV1 M12 | -25.3% | -25.5% | -28.2% | -33.7 … -22.0 | 209/209 |
| HEVC Fast | x265 superfast | -21.1% | -23.7% | -21.3% | -33.9 … -17.2 | 209/209 |
| HEVC Fast | HEVC | +9.9% | +10.8% | +14.4% | +7.4 … +12.5 | 209/209 |
| HEVC Faster | SVT-AV1 M12 | -16.2% | -16.0% | -20.0% | -26.7 … -10.0 | 209/209 |
| HEVC Faster | SVT-AV1 M13 | -18.6% | -18.9% | -22.2% | -28.9 … -13.8 | 209/209 |
| HEVC Faster | HEVC Fast | +13.0% | +13.2% | +12.4% | +7.7 … +17.7 | 209/209 |
| AV1 Slow | SVT-AV1 M12 | -34.5% | -38.1% | -35.4% | -42.5 … -30.6 | 209/209 |
| AV1 | SVT-AV1 M12 | -30.6% | -35.0% | -32.8% | -39.2 … -25.6 | 209/209 |
| AV1 | openh264 | -51.8% | -53.5% | -52.1% | -64.2 … -42.2 | 206/209 |
| AV1 | SVT-AV1 M8 | -8.8% | -10.1% | -17.8% | -21.5 … -5.1 | 209/209 |
| AV1 | libaom rt 8 | -26.5% | -32.5% | -27.5% | -37.5 … -21.3 | 209/209 |
| AV1 Fast | SVT-AV1 M12 | -25.3% | -28.6% | -26.9% | -33.8 … -21.5 | 209/209 |
| AV1 Fast | openh264 | -47.7% | -48.7% | -47.0% | -62.2 … -37.5 | 207/209 |
| AV1 Fast | AV1 | +8.2% | +10.2% | +9.4% | +5.2 … +11.1 | 209/209 |
| AV1 Faster | SVT-AV1 M12 | -20.0% | -20.8% | -24.5% | -32.8 … -15.9 | 209/209 |
| AV1 Faster | AV1 Fast | +7.0% | +11.2% | +3.2% | +0.3 … +9.7 | 209/209 |
| AV1 Veryfast | SVT-AV1 M12 | -12.6% | -9.9% | -16.7% | -26.3 … -6.7 | 209/209 |
| AV1 Veryfast | SVT-AV1 M13 | -15.3% | -13.1% | -19.2% | -28.3 … -9.2 | 209/209 |
| AV1 Veryfast | AV1 Faster | +10.3% | +15.4% | +11.3% | +6.2 … +19.2 | 209/209 |
| VP9 Slow | HEVC Slow | +10.0% | +5.5% | +13.4% | +6.7 … +15.2 | 209/209 |
| VP9 | HEVC | +4.8% | +0.5% | +9.4% | +2.0 … +10.0 | 209/209 |
| VP9 | libvpx VP9 rt 7 | -43.6% | -48.1% | -42.2% | -53.1 … -37.9 | 209/209 |
| VP9 Fast | HEVC Fast | +6.7% | +2.5% | +10.0% | +2.0 … +12.4 | 209/209 |
| VP9 Fast | libvpx VP9 rt 9 | -38.5% | -45.1% | -36.0% | -47.6 … -34.6 | 209/209 |
| VP9 Fast | VP9 | +11.0% | +12.0% | +13.9% | +6.6 … +15.3 | 209/209 |
| VP9 Faster | HEVC Faster | +2.3% | +3.6% | +2.3% | -2.1 … +6.7 | 209/209 |
| VP9 Faster | libvpx VP9 rt 9 | -33.5% | -37.2% | -33.0% | -45.2 … -26.6 | 209/209 |
| VP9 Faster | VP9 Fast | +8.4% | +14.4% | +4.3% | +4.6 … +14.1 | 209/209 |
| VP9 Veryfast | HEVC Faster | +9.2% | +8.8% | +8.0% | +4.5 … +14.4 | 209/209 |
| VP9 Veryfast | libvpx VP9 rt 9 | -29.6% | -34.6% | -29.4% | -41.2 … -24.3 | 209/209 |
| VP9 Veryfast | VP9 Faster | +7.2% | +5.3% | +6.0% | +2.5 … +10.8 | 209/209 |
| AV2 Slow | AV1 Slow | -6.7% | -7.3% | -9.9% | -9.8 … -4.2 | 209/209 |
| AV2 | AV1 | -6.4% | -6.4% | -8.9% | -8.6 … -2.9 | 209/209 |
| AV2 | SVT-AV1 M8 | -14.9% | -16.0% | -25.4% | -25.7 … -10.7 | 209/209 |
| AV2 Fast | AV1 Fast | -3.7% | -1.9% | -3.8% | -8.6 … +0.5 | 209/209 |
| AV2 Fast | AV2 | +11.6% | +15.7% | +15.5% | +6.9 … +13.9 | 209/209 |
| AV2 Faster | AV1 Faster | -6.4% | -8.1% | -2.8% | -9.2 … -1.5 | 209/209 |
| AV2 Faster | SVT-AV1 M12 | -26.2% | -28.0% | -27.4% | -35.1 … -21.3 | 209/209 |
| AV2 Faster | AV2 Fast | +4.0% | +4.3% | +4.2% | +0.8 … +5.9 | 209/209 |
| VVC Slow | HEVC Slow | -6.7% | -6.9% | -7.7% | -7.8 … -3.1 | 209/209 |
| VVC | HEVC | -9.7% | -9.8% | -9.0% | -11.3 … -5.6 | 209/209 |
| VVC | x265 medium | -32.7% | -35.4% | -37.9% | -43.0 … -25.5 | 209/209 |
| VVC Fast | HEVC Fast | -7.4% | -7.7% | -9.6% | -9.3 … -3.5 | 209/209 |
| VVC Fast | VVC | +12.8% | +13.7% | +13.7% | +9.9 … +14.4 | 209/209 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Fast, H.264 Faster, H.264 Veryfast, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster: 20d31869; H.264: c9cff89f), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.6 [-2.8, -0.3]%); x264 medium --tune psnr (-10.0 [-11.1, -8.9]%); x265 ultrafast --tune psnr (-0.6 [-1.0, -0.2]%); x265 superfast --tune psnr (-2.0 [-2.3, -1.6]%); x265 medium --tune psnr (-4.9 [-5.4, -4.3]%); libvpx VP9 rt 7 --tune ssim (-1.9 [-2.5, -1.0]%); libvpx VP9 rt 9 --tune ssim (-1.3 [-1.7, -0.8]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
