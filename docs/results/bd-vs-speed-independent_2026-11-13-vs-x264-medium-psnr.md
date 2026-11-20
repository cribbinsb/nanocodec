# Bitrate vs encode speed, held-out clips (independent_2026-11-13), zero line x264-medium~psnr

<sub>Updated 2026-11-20, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Fast, H.264 Faster, H.264 Veryfast, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster: 20d31869; H.264: c9cff89f); H.264 timed on 20d31869 (same streams)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 209 clips of `independent_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_2026-11-13-vs-x264-medium-psnr.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_2026-11-13-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -46.4% | -46.3% | -44.4% | -55.9 … -41.8 | 16.0 | 0.60× | 64.4 | 0.57× | 206/206 | Pareto; Pareto x86 |
| **AV2 Slow** | -45.4% | -48.1% | -42.2% | -55.3 … -41.9 | 16.1 | 0.60× | 62.3 | 0.55× | 206/206 | Pareto |
| **VVC** | -45.0% | -44.8% | -42.8% | -55.0 … -39.8 | 29.8 | 1.12× | 121.0 | 1.07× | 206/206 | Pareto; Pareto x86 |
| **HEVC Slow** | -43.3% | -43.1% | -40.4% | -53.3 … -38.3 | 19.0 | 0.71× | 76.9 | 0.68× | 206/206 |  |
| **AV2** | -42.7% | -45.6% | -39.6% | -53.1 … -38.4 | 39.2 | 1.47× | 170.1 | 1.51× | 206/206 | Pareto; Pareto x86 |
| **AV1 Slow** | -41.8% | -44.5% | -36.1% | -52.0 … -38.6 | 26.6 | 1.00× | 113.3 | 1.00× | 206/206 | 1 rate-invalid pts |
| **HEVC** | -40.4% | -40.3% | -38.1% | -51.7 … -35.3 | 45.3 | 1.70× | 171.9 | 1.52× | 206/206 | Pareto; Pareto x86 |
| **VVC Fast** | -39.3% | -38.9% | -36.7% | -51.1 … -33.6 | 53.2 | 1.99× | 205.2 | 1.82× | 206/206 | Pareto; Pareto x86 |
| **AV1** | -39.2% | -42.2% | -34.2% | -49.9 … -35.9 | 45.7 | 1.71× | 181.4 | 1.61× | 206/206 | 1 rate-invalid pts |
| **VP9 Slow** | -39.0% | -41.1% | -34.0% | -48.1 … -33.4 | 26.7 | 1.00× | 100.0 | 0.89× | 206/206 | 3 rate-invalid pts |
| **VP9** | -38.3% | -40.4% | -33.4% | -48.1 … -32.2 | 42.1 | 1.58× | 171.0 | 1.52× | 206/206 | 4 rate-invalid pts |
| **AV2 Fast** | -37.2% | -38.9% | -31.2% | -49.6 … -32.6 | 51.8 | 1.94× | 230.8 | 2.05× | 206/206 | Pareto x86 |
| **HEVC Fast** | -35.9% | -35.8% | -31.0% | -46.1 … -31.1 | 72.0 | 2.70× | 293.2 | 2.60× | 206/206 | Pareto; Pareto x86 |
| **AV1 Fast** | -35.3% | -37.5% | -29.3% | -46.6 … -30.8 | 66.2 | 2.48× | 267.6 | 2.37× | 206/206 |  |
| **AV2 Faster** | -35.2% | -36.7% | -29.0% | -48.2 … -30.2 | 62.9 | 2.36× | 274.8 | 2.44× | 206/206 |  |
| **VP9 Fast** | -32.7% | -34.4% | -26.0% | -42.7 … -26.3 | 55.8 | 2.09× | 233.7 | 2.07× | 206/206 | 1 rate-invalid pts |
| **AV1 Faster** | -31.8% | -31.8% | -27.9% | -45.9 … -27.3 | 79.4 | 2.98× | 323.9 | 2.87× | 206/206 | Pareto; Pareto x86 |
| **HEVC Faster** | -29.8% | -29.6% | -24.6% | -41.3 … -24.2 | 90.3 | 3.38× | 364.2 | 3.23× | 206/206 | Pareto; Pareto x86 |
| **VP9 Faster** | -28.5% | -27.1% | -23.6% | -41.2 … -20.0 | 92.3 | 3.46× | 368.7 | 3.27× | 206/206 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -28.3% | -32.0% | -15.2% | -38.9 … -25.4 | 14.3 | 0.53× | 80.8 | 0.72× | 206/206 | 38 rate-invalid pts |
| **AV1 Veryfast** | -26.1% | -24.1% | -20.9% | -39.6 … -19.1 | 96.5 | 3.62× | 431.4 | 3.83× | 206/206 | Pareto; Pareto x86 |
| **VP9 Veryfast** | -25.0% | -24.4% | -20.3% | -37.8 … -19.0 | 95.8 | 3.59× | 385.9 | 3.42× | 206/206 |  |
| **H.264 Slow** | -23.8% | -26.9% | -19.9% | -39.5 … -14.0 | 37.2 | 1.39× | 115.3 | 1.02× | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) | -22.8% | -23.8% | -15.7% | -32.9 … -18.2 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 38 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M10 | -21.6% | -23.3% | -10.3% | -32.0 … -17.7 | 39.3 | 1.47× | 228.9 | 2.03× | 206/206 | 34 rate-invalid pts |
| **H.264** | -21.5% | -24.0% | -18.0% | -37.7 … -12.4 | 66.4 | 2.49× | 202.2 | 1.79× | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) | -19.6% | -19.4% | -12.2% | -29.2 … -16.0 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 36 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 medium | -18.8% | -17.2% | -9.0% | -29.1 … -14.0 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 40 rate-invalid pts |
| x265 superfast | -18.2% | -16.2% | -11.3% | -27.3 … -13.9 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 37 rate-invalid pts |
| libaom rt 8 | -14.2% | -11.9% | -6.8% | -22.6 … -9.7 | 17.4 | 0.65× | 68.8 | 0.61× | 206/206 | 137 rate-invalid pts |
| libaom rt 10 | -13.3% | -12.8% | -4.1% | -21.5 … -8.1 | 22.3 | 0.83× | 90.5 | 0.80× | 206/206 | 146 rate-invalid pts |
| **H.264 Fast** | -13.2% | -15.2% | -9.1% | -29.5 … -1.5 | 119.5 | 4.48× | 370.1 | 3.28× | 206/206 | Pareto |
| x265 ultrafast, metric-tuned (--tune psnr) | -12.5% | -11.6% | -4.7% | -24.1 … -6.1 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 36 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M12 | -12.5% | -12.3% | -2.4% | -24.3 … -8.6 | 70.9 | 2.67× | 408.7 | 3.62× | 206/206 | 29 rate-invalid pts |
| x265 ultrafast | -12.0% | -9.7% | -4.7% | -23.3 … -5.2 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 36 rate-invalid pts |
| SVT-AV1 M13 | -10.5% | -9.8% | -0.5% | -23.2 … -6.3 | 79.2 | 2.96× | 445.4 | 3.95× | 206/206 | Pareto x86; 29 rate-invalid pts |
| **H.264 Faster** | -4.9% | -7.6% | +0.1% | -26.2 … +14.7 | 146.9 | 5.51× | 460.1 | 4.08× | 206/206 | Pareto; Pareto x86 |
| **H.264 Veryfast** | -0.7% | -4.8% | +5.2% | -21.2 … +17.4 | 164.1 | 6.15× | 520.2 | 4.61× | 206/206 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 58 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 medium, metric-tuned (--tune ssim) | +10.1% | +8.6% | +25.1% | +6.6 … +14.5 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 51 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.5% | +19.7% | +18.1% | +5.0 … +16.7 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 234 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +12.7% | +20.2% | +18.7% | +5.7 … +18.1 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 235 rate-invalid pts |
| x264 medium | +12.8% | +12.1% | +25.0% | +7.7 … +16.1 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 53 rate-invalid pts |
| x264 veryfast, metric-tuned (--tune psnr) | +13.0% | +10.8% | +13.2% | +1.7 … +21.6 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.5% | +19.9% | +25.7% | +4.7 … +20.5 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 198 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +15.5% | +21.1% | +21.4% | +5.5 … +22.7 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 228 rate-invalid pts |
| x264 veryfast | +16.4% | +12.4% | +31.8% | +3.5 … +24.7 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts |
| openh264 | +28.7% | +25.1% | +41.5% | -4.9 … +68.2 | 111.5 | 4.17× | 524.7 | 4.65× | 206/206 | Pareto x86; 367 rate-invalid pts |
| x264 ultrafast | +53.4% | +65.9% | +43.3% | +25.4 … +75.6 | 190.5 | 7.11× | 1017.0 | 9.02× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: pool_barscene_pristine, pool_bodycam_antare_antare_test, pool_bodycam_bwc_video118, pool_chad_1_093, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_crowd_person_uid_vid_00222, pool_ducks_take_off_pristine, pool_lowlight_cevo_inof_md_indoor_night_ohcam_0, pool_meva_hospital_g301, pool_static_chirla_chirla_seq_024_camera_7_2023, ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bdd100k_mot_bdd_00d9e313_7d75bb18, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video317, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_chirla_chirla_seq_026_camera_3_2023_12_05_12_12_13, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00212, ind_personpath22_uid_vid_00214, ind_personpath22_uid_vid_00216, ind_personpath22_uid_vid_00231, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 3; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 22, crowd 57, drone 3, lowlight 15, pan 53, static 54, texture 2 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +12.8% | +12.1% |
| H.264 Slow | -23.8% | -24.9% |
| H.264 | -21.5% | -22.7% |
| H.264 Fast | -13.2% | -14.3% |
| H.264 Faster | -4.9% | -5.4% |
| H.264 Veryfast | -0.7% | -1.9% |
| HEVC Slow | -43.3% | -45.7% |
| HEVC | -40.4% | -42.7% |
| HEVC Fast | -35.9% | -38.2% |
| HEVC Faster | -29.8% | -31.9% |
| AV1 Slow | -41.8% | -44.7% |
| AV1 | -39.2% | -42.5% |
| AV1 Fast | -35.3% | -38.5% |
| AV1 Faster | -31.8% | -35.8% |
| AV1 Veryfast | -26.1% | -29.2% |
| VP9 Slow | -39.0% | -40.7% |
| VP9 | -38.3% | -40.1% |
| VP9 Fast | -32.7% | -34.6% |
| VP9 Faster | -28.5% | -30.5% |
| VP9 Veryfast | -25.0% | -27.0% |
| AV2 Slow | -45.4% | -48.1% |
| AV2 | -42.7% | -45.4% |
| AV2 Fast | -37.2% | -40.5% |
| AV2 Faster | -35.2% | -38.6% |
| VVC Slow | -46.4% | -48.4% |
| VVC | -45.0% | -46.9% |
| VVC Fast | -39.3% | -41.5% |
| openh264 | +28.7% | +31.1% |
| x264 ultrafast | +53.4% | +54.1% |
| x264 veryfast | +16.4% | +14.7% |
| x265 ultrafast | -12.0% | -13.9% |
| x265 superfast | -18.2% | -20.6% |
| x265 medium | -18.8% | -21.3% |
| SVT-AV1 M8 | -28.3% | -32.4% |
| SVT-AV1 M10 | -21.6% | -25.3% |
| SVT-AV1 M12 | -12.5% | -17.6% |
| SVT-AV1 M13 | -10.5% | -15.6% |
| libvpx VP9 rt 7 | +15.5% | +16.3% |
| libvpx VP9 rt 9 | +12.7% | +12.3% |
| libaom rt 8 | -14.2% | -15.5% |
| libaom rt 10 | -13.3% | -15.2% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +28.7% | +25.1% | +41.5% | +19.6% | 206/206 |  |
| openh264, adaptive quantisation on | +28.8% | +25.1% | +41.5% | +19.6% | 206/206 | +0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +53.4% | +65.9% | +43.3% | +50.9% | 206/206 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +53.4% | +66.0% | +43.1% | +51.2% | 206/206 | +0.1 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +51.5% | +63.5% | +62.2% | +29.0% | 206/206 | -0.3 [-1.9, +1.3] not distinguishable |
| x264 veryfast | +16.4% | +12.4% | +31.8% | +4.9% | 206/206 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +13.0% | +10.8% | +13.2% | +14.9% | 206/206 | -1.6 [-2.9, -0.4] better |
| x264 veryfast, metric-tuned (--tune ssim) | +17.5% | +14.0% | +35.0% | +3.6% | 206/206 | +0.9 [+0.3, +1.5] worse |
| x264 medium | +12.8% | +12.1% | +25.0% | +1.4% | 206/206 |  |
| x264 medium, metric-tuned (--tune ssim) ● | +10.1% | +8.6% | +25.1% | -3.3% | 206/206 | -2.5 [-3.2, -1.8] better |
| x265 ultrafast | -12.0% | -9.7% | -4.7% | -21.8% | 206/206 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -12.5% | -11.6% | -4.7% | -21.1% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -12.5% | -11.6% | -4.7% | -21.1% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 superfast | -18.2% | -16.2% | -11.3% | -27.0% | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -19.6% | -19.4% | -12.2% | -27.3% | 206/206 | -2.0 [-2.4, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -19.6% | -19.4% | -12.2% | -27.3% | 206/206 | -2.0 [-2.4, -1.6] better |
| x265 medium | -18.8% | -17.2% | -9.0% | -30.2% | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -22.8% | -23.8% | -15.7% | -29.1% | 206/206 | -4.9 [-5.5, -4.4] better |
| x265 medium, metric-tuned (--tune ssim) | -20.6% | -20.6% | -10.4% | -30.9% | 206/206 | -2.3 [-2.5, -2.0] better |
| libvpx VP9 rt 7 | +15.5% | +21.1% | +21.4% | +4.0% | 206/206 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +14.5% | +19.9% | +25.7% | -2.0% | 206/206 | -1.8 [-2.4, -1.0] better |
| libvpx VP9 rt 9 | +12.7% | +20.2% | +18.7% | -0.8% | 206/206 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | +11.5% | +19.7% | +18.1% | -3.4% | 206/206 | -1.3 [-1.7, -0.8] better |
| libaom rt 8 | -14.2% | -11.9% | -6.8% | -23.9% | 206/206 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -13.6% | -10.1% | -1.4% | -29.5% | 206/206 | +0.9 [+0.2, +1.6] worse |
| libaom rt 10 | -13.3% | -12.8% | -4.1% | -22.9% | 206/206 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -13.5% | -11.4% | -0.5% | -28.6% | 206/206 | +0.4 [-0.4, +1.1] not distinguishable |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -52.8% | -54.7% | -56.1% | -47.8% | 206/206 |
| **AV2 Slow** | -51.7% | -56.1% | -54.2% | -45.0% | 205/206 |
| **VVC** | -51.6% | -53.4% | -54.8% | -46.5% | 206/206 |
| **HEVC Slow** | -50.1% | -51.7% | -52.9% | -45.7% | 206/206 |
| **AV2** | -49.4% | -54.0% | -52.2% | -42.1% | 205/206 |
| **AV1 Slow** | -48.7% | -52.8% | -49.4% | -44.0% | 206/206 |
| **HEVC** | -47.5% | -49.2% | -51.1% | -42.1% | 206/206 |
| **VVC Fast** | -46.7% | -48.2% | -50.0% | -41.8% | 206/206 |
| **AV1** | -46.5% | -50.8% | -48.0% | -40.6% | 206/206 |
| **VP9 Slow** | -46.2% | -49.7% | -47.9% | -41.1% | 206/206 |
| **VP9** | -45.6% | -49.1% | -47.4% | -40.2% | 206/206 |
| **AV2 Fast** | -44.9% | -48.1% | -45.7% | -40.7% | 206/206 |
| **HEVC Fast** | -43.6% | -45.2% | -45.5% | -40.2% | 206/206 |
| **AV1 Fast** | -43.1% | -46.6% | -44.1% | -38.6% | 206/206 |
| **AV2 Faster** | -43.0% | -46.2% | -43.9% | -39.0% | 206/206 |
| **VP9 Fast** | -40.8% | -43.9% | -41.5% | -37.0% | 206/206 |
| **AV1 Faster** | -39.9% | -41.4% | -42.9% | -35.2% | 206/206 |
| **HEVC Faster** | -38.1% | -39.4% | -40.3% | -34.6% | 206/206 |
| SVT-AV1 M8 | -37.4% | -42.3% | -32.8% | -37.0% | 206/206 |
| **VP9 Faster** | -37.0% | -37.2% | -39.6% | -34.3% | 206/206 |
| **AV1 Veryfast** | -34.8% | -34.2% | -37.3% | -32.8% | 206/206 |
| **VP9 Veryfast** | -33.7% | -34.6% | -36.9% | -29.8% | 206/206 |
| **H.264 Slow** | -32.4% | -36.3% | -36.2% | -24.7% | 206/206 |
| x265 medium, metric-tuned (--tune psnr) | -31.7% | -33.9% | -32.9% | -28.4% | 206/206 |
| SVT-AV1 M10 | -31.4% | -34.8% | -29.0% | -30.5% | 206/206 |
| **H.264** | -30.4% | -33.8% | -34.8% | -22.6% | 206/206 |
| x265 medium, metric-tuned (--tune ssim) | -30.2% | -31.3% | -28.8% | -30.4% | 206/206 |
| x265 superfast, metric-tuned (--tune psnr) | -29.1% | -30.5% | -30.1% | -26.8% | 206/206 |
| x265 superfast, metric-tuned (--tune ssim) | -29.1% | -30.5% | -30.1% | -26.8% | 206/206 |
| x265 medium | -28.6% | -28.4% | -27.6% | -29.7% | 206/206 |
| x265 superfast | -27.9% | -27.7% | -29.5% | -26.5% | 206/206 |
| libaom rt 8 | -24.2% | -23.3% | -26.0% | -23.5% | 205/206 |
| libaom rt 10, metric-tuned (--tune ssim) | -24.1% | -23.2% | -21.0% | -28.1% | 206/206 |
| libaom rt 8, metric-tuned (--tune ssim) | -24.1% | -21.6% | -21.6% | -29.1% | 206/206 |
| libaom rt 10 | -23.7% | -24.6% | -24.1% | -22.5% | 206/206 |
| SVT-AV1 M12 | -23.0% | -24.3% | -22.3% | -22.5% | 206/206 |
| **H.264 Fast** | -22.9% | -25.7% | -27.6% | -15.4% | 206/206 |
| x265 ultrafast, metric-tuned (--tune psnr) | -22.8% | -23.5% | -24.2% | -20.8% | 206/206 |
| x265 ultrafast, metric-tuned (--tune ssim) | -22.8% | -23.5% | -24.2% | -20.8% | 206/206 |
| x265 ultrafast | -22.5% | -21.7% | -24.4% | -21.5% | 206/206 |
| SVT-AV1 M13 | -21.2% | -21.8% | -20.7% | -21.2% | 206/206 |
| **H.264 Faster** | -15.4% | -18.5% | -20.2% | -7.5% | 206/206 |
| **H.264 Veryfast** | -11.6% | -15.9% | -16.0% | -3.0% | 206/206 |
| x264 medium, metric-tuned (--tune psnr) | -10.1% | -10.1% | -19.1% | -0.9% | 206/206 |
| x264 medium, metric-tuned (--tune ssim) | -2.5% | -2.9% | -0.1% | -4.5% | 206/206 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.6% | +5.0% | -6.2% | -3.5% | 206/206 |
| libvpx VP9 rt 9 | -0.5% | +5.3% | -5.7% | -1.0% | 206/206 |
| x264 veryfast, metric-tuned (--tune psnr) | +1.0% | -1.0% | -9.5% | +13.5% | 206/206 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.6% | +6.6% | +1.2% | -2.8% | 206/206 |
| libvpx VP9 rt 7 | +2.4% | +7.1% | -3.1% | +3.3% | 206/206 |
| x264 veryfast | +2.6% | +0.1% | +3.9% | +3.7% | 206/206 |
| x264 veryfast, metric-tuned (--tune ssim) | +3.4% | +1.5% | +6.3% | +2.5% | 206/206 |
| openh264 | +13.2% | +9.0% | +11.8% | +18.7% | 206/206 |
| openh264, adaptive quantisation on | +13.2% | +9.0% | +11.8% | +18.8% | 206/206 |
| x264 ultrafast, metric-tuned (--tune ssim) | +33.5% | +47.0% | +26.5% | +27.0% | 206/206 |
| x264 ultrafast | +37.0% | +49.9% | +13.0% | +48.1% | 206/206 |
| x264 ultrafast, metric-tuned (--tune psnr) | +37.1% | +50.0% | +12.9% | +48.4% | 206/206 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium, metric-tuned (--tune psnr)'s (even with its extra rungs, which the 50% source cap limits).

- **pool_ducks_take_off_pristine** (texture): no reliable overlap for SVT-AV1 M8, SVT-AV1 M10, libvpx VP9 rt 9.
- **pool_otw_homes_00015** (static): no reliable overlap for VVC Slow.
- **ind_bwc_videotext_video190** (bodycam): no reliable overlap for libaom rt 8, libaom rt 10.

Direct BD-J vs x264 medium, metric-tuned (--tune psnr) on them, where it exists:

| Encoder | pool_ducks_take_off_pristine | pool_otw_homes_00015 | ind_bwc_videotext_video190 |
|---|---:|---:|---:|
| VVC Slow | -61.0% | n/a | -40.4% |
| AV2 Slow | -56.9% | -59.2% | -38.2% |
| VVC | -59.3% | -59.4% | -33.6% |
| HEVC Slow | -47.4% | -58.2% | -38.9% |
| AV2 | -52.8% | -54.1% | -32.8% |
| AV1 Slow | -53.0% | -50.7% | -36.2% |
| HEVC | -43.6% | -54.8% | -30.5% |
| VVC Fast | -48.0% | -55.6% | -27.9% |
| AV1 | -49.2% | -46.3% | -30.6% |
| VP9 Slow | -43.9% | -49.7% | -36.0% |
| VP9 | -44.9% | -48.0% | -31.4% |
| AV2 Fast | -43.2% | -49.7% | -23.0% |
| HEVC Fast | -45.3% | -51.2% | -22.1% |
| AV1 Fast | -43.6% | -38.7% | -19.4% |
| AV2 Faster | -42.8% | -49.5% | -21.3% |
| VP9 Fast | -38.1% | -41.1% | -21.3% |
| AV1 Faster | -38.1% | -44.6% | -19.3% |
| HEVC Faster | -38.0% | -43.4% | -19.1% |
| VP9 Faster | -35.6% | -37.5% | -18.4% |
| SVT-AV1 M8 | n/a | -38.5% | -23.3% |
| AV1 Veryfast | -40.8% | -43.5% | -25.5% |
| VP9 Veryfast | -38.8% | -35.5% | -16.5% |
| H.264 Slow | -4.4% | -19.8% | -24.4% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| SVT-AV1 M10 | n/a | -31.1% | +5.7% |
| H.264 | +4.2% | -19.0% | -20.9% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 medium | -41.3% | -41.1% | +0.8% |
| x265 superfast | -41.9% | -34.9% | +10.6% |
| libaom rt 8 | -5.9% | -32.7% | n/a |
| libaom rt 10 | -3.1% | -34.9% | n/a |
| H.264 Fast | +15.4% | -7.8% | -8.6% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| SVT-AV1 M12 | -23.8% | -22.6% | +8.8% |
| x265 ultrafast | -41.9% | -22.3% | +20.2% |
| SVT-AV1 M13 | -23.5% | -18.6% | +10.1% |
| H.264 Faster | +7.1% | +27.4% | -7.3% |
| H.264 Veryfast | +19.0% | +23.8% | -4.4% |
| x264 medium, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9 | n/a | +17.4% | +8.6% |
| x264 medium | +15.9% | +6.9% | +2.6% |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 7 | +8.7% | -4.2% | -10.5% |
| x264 veryfast | +13.9% | +30.9% | +24.1% |
| openh264 | +109.2% | +147.7% | +26.1% |
| x264 ultrafast | +29.5% | +99.8% | +64.9% |

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
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Fast, H.264 Faster, H.264 Veryfast, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster: 20d31869; H.264: c9cff89f); H.264 timed on 20d31869 (same streams), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.6 [-2.9, -0.4]%); x264 medium --tune ssim (-2.5 [-3.2, -1.8]%); x265 ultrafast --tune psnr (-0.6 [-1.0, -0.2]%); x265 superfast --tune psnr (-2.0 [-2.4, -1.6]%); x265 medium --tune psnr (-4.9 [-5.5, -4.4]%); libvpx VP9 rt 7 --tune ssim (-1.8 [-2.4, -1.0]%); libvpx VP9 rt 9 --tune ssim (-1.3 [-1.7, -0.8]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
