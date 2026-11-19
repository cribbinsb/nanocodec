# Bitrate vs encode speed, held-out clips (independent_2026-11-13)

<sub>Updated 2026-11-19, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (AV1, AV1 Fast: 4a5b8dd1; VP9 Fast: b33ba3bb; H.264 Veryfast, VP9, AV1 Slow, VP9 Slow, AV2 Slow: bad14976; H.264, HEVC, HEVC Fast, HEVC Faster, AV1 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, VVC Slow, AV2 Faster: c9cff89f; AV1 Veryfast: d40f86b3); H.264 Fast, H.264 Faster, VP9 Faster, AV1 Slow, AV2 Slow, H.264 Veryfast, VP9, VP9 Slow, AV1, AV1 Fast timed on b33ba3bb (same streams)</sub>

Quality on the 209 clips of `independent_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_2026-11-13.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -51.2% | -52.9% | -54.6% | -60.2 … -44.9 | 24.8 | 0.93× | 97.7 | 0.87× | 206/206 | Pareto; Pareto x86 |
| **VVC** | -50.7% | -52.4% | -53.8% | -60.2 … -44.5 | 28.9 | 1.08× | 118.1 | 1.05× | 206/206 | Pareto; Pareto x86 |
| **AV2 Slow** | -50.4% | -54.8% | -52.9% | -58.3 … -44.7 | 22.0 | 0.82× | 81.6 | 0.72× | 206/206 |  |
| **AV2** | -48.9% | -53.5% | -51.6% | -57.0 … -42.6 | 37.9 | 1.42× | 164.2 | 1.46× | 206/206 | Pareto; Pareto x86 |
| **HEVC Slow** | -48.1% | -49.5% | -51.2% | -57.0 … -41.9 | 31.0 | 1.16× | 116.4 | 1.03× | 206/206 |  |
| **AV1 Slow** | -46.6% | -51.0% | -48.2% | -55.5 … -41.1 | 29.7 | 1.11× | 116.6 | 1.03× | 206/206 | 1 rate-invalid pts |
| **HEVC** | -46.4% | -48.3% | -50.0% | -55.6 … -40.2 | 50.4 | 1.89× | 201.2 | 1.78× | 206/206 | Pareto; Pareto x86 |
| **AV1** | -45.8% | -50.2% | -47.4% | -54.7 … -40.3 | 45.0 | 1.69× | 171.4 | 1.52× | 206/206 | 1 rate-invalid pts |
| **VP9 Slow** | -45.4% | -48.7% | -47.4% | -53.4 … -38.1 | 36.7 | 1.38× | 151.8 | 1.35× | 206/206 | 4 rate-invalid pts |
| **VP9** | -44.9% | -48.4% | -46.8% | -53.1 … -37.7 | 43.2 | 1.62× | 172.4 | 1.53× | 206/206 | 4 rate-invalid pts |
| **AV2 Fast** | -42.8% | -46.1% | -43.8% | -52.2 … -36.5 | 50.0 | 1.87× | 223.4 | 1.98× | 206/206 | Pareto x86 |
| **AV2 Faster** | -41.9% | -45.0% | -43.3% | -52.2 … -35.1 | 59.5 | 2.23× | 254.8 | 2.26× | 206/206 | Pareto; Pareto x86 |
| **VVC Fast** | -41.9% | -41.5% | -45.2% | -51.3 … -33.8 | 55.1 | 2.07× | 221.2 | 1.96× | 206/206 |  |
| **AV1 Fast** | -41.7% | -45.4% | -42.3% | -50.7 … -36.2 | 65.5 | 2.46× | 280.4 | 2.49× | 206/206 | Pareto; Pareto x86 |
| **HEVC Fast** | -41.2% | -42.8% | -43.6% | -50.1 … -34.4 | 78.2 | 2.93× | 304.6 | 2.70× | 206/206 | Pareto; Pareto x86 |
| **VP9 Fast** | -39.9% | -43.0% | -40.6% | -48.4 … -32.8 | 54.4 | 2.04× | 227.3 | 2.02× | 206/206 |  |
| **AV1 Faster** | -38.5% | -40.3% | -41.3% | -49.2 … -32.9 | 78.4 | 2.94× | 312.7 | 2.77× | 206/206 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -36.9% | -41.9% | -32.3% | -46.6 … -30.6 | 14.3 | 0.53× | 80.8 | 0.72× | 206/206 | 34 rate-invalid pts |
| **AV1 Veryfast** | -34.1% | -33.6% | -36.4% | -45.7 … -25.6 | 95.9 | 3.59× | 425.3 | 3.77× | 206/206 | Pareto; Pareto x86 |
| **VP9 Faster** | -34.0% | -33.7% | -37.7% | -44.7 … -25.2 | 92.0 | 3.45× | 354.1 | 3.14× | 206/206 | 1 rate-invalid pts |
| **HEVC Faster** | -33.1% | -33.0% | -35.2% | -42.9 … -24.5 | 101.4 | 3.80× | 417.8 | 3.71× | 206/206 | Pareto |
| x265 medium, metric-tuned (--tune psnr) | -31.3% | -33.5% | -32.4% | -42.6 … -25.5 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 35 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M10 | -30.8% | -34.1% | -28.3% | -40.8 … -24.3 | 39.3 | 1.47× | 228.9 | 2.03× | 206/206 | 30 rate-invalid pts |
| **VP9 Veryfast** | -30.1% | -30.6% | -35.3% | -41.5 … -21.2 | 94.3 | 3.53× | 390.7 | 3.47× | 206/206 |  |
| **H.264** | -30.1% | -33.5% | -34.4% | -41.6 … -21.5 | 65.9 | 2.47× | 205.2 | 1.82× | 206/206 |  |
| **H.264 Slow** | -29.6% | -33.5% | -33.7% | -42.0 … -21.8 | 42.0 | 1.57× | 123.7 | 1.10× | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) | -28.5% | -29.9% | -29.7% | -39.2 … -22.8 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 33 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 medium | -28.1% | -28.0% | -27.1% | -38.7 … -22.3 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 37 rate-invalid pts |
| x265 superfast | -27.3% | -27.0% | -29.0% | -37.0 … -20.9 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 34 rate-invalid pts |
| libaom rt 8 | -24.4% | -23.5% | -26.1% | -32.9 … -18.1 | 17.4 | 0.65× | 68.8 | 0.61× | 206/206 | 126 rate-invalid pts |
| libaom rt 10 | -23.7% | -24.7% | -24.1% | -31.8 … -16.6 | 22.3 | 0.83× | 90.5 | 0.80× | 206/206 | 136 rate-invalid pts |
| SVT-AV1 M12 | -22.4% | -23.6% | -21.6% | -35.0 … -15.4 | 70.9 | 2.67× | 408.7 | 3.62× | 206/206 | 26 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -22.0% | -22.6% | -23.5% | -34.6 … -12.3 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 33 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -21.7% | -20.7% | -23.6% | -33.7 … -11.7 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 33 rate-invalid pts |
| SVT-AV1 M13 | -20.5% | -21.2% | -19.9% | -31.9 … -13.9 | 79.2 | 2.96× | 445.4 | 3.95× | 206/206 | Pareto x86; 26 rate-invalid pts |
| **H.264 Fast** | -18.6% | -21.5% | -24.4% | -32.0 … -8.7 | 121.2 | 4.54× | 379.5 | 3.37× | 206/206 | Pareto |
| **H.264 Faster** | -11.4% | -15.1% | -16.2% | -27.4 … +5.1 | 145.7 | 5.46× | 486.1 | 4.31× | 206/206 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -10.0% | -10.2% | -19.1% | -12.8 … -6.3 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 49 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Veryfast** | -5.6% | -9.9% | -10.5% | -23.1 … +10.9 | 166.2 | 6.23× | 526.2 | 4.67× | 206/206 | Pareto; Pareto x86 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.4% | +5.1% | -6.0% | -9.0 … +6.6 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 221 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -0.3% | +5.5% | -5.5% | -8.7 … +8.6 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 222 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 46 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.3% | +6.3% | +0.9% | -6.5 … +11.0 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 185 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | +1.7% | -0.3% | -9.0% | -8.9 … +11.1 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +2.2% | +6.8% | -3.3% | -6.8 … +13.5 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 215 rate-invalid pts |
| x264 veryfast | +3.2% | +0.9% | +4.5% | -3.6 … +10.3 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts |
| openh264 | +13.7% | +9.4% | +12.5% | -15.1 … +50.2 | 111.5 | 4.17× | 524.7 | 4.65× | 206/206 | 353 rate-invalid pts |
| x264 ultrafast | +38.4% | +51.8% | +13.8% | +17.8 … +57.2 | 190.5 | 7.11× | 1017.0 | 9.02× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: pool_barscene_pristine, pool_bodycam_antare_antare_test, pool_bodycam_bwc_video118, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_crowd_person_uid_vid_00222, pool_ducks_take_off_pristine, pool_lowlight_cevo_inof_md_indoor_night_ohcam_0, pool_meva_hospital_g301, pool_static_chirla_chirla_seq_024_camera_7_2023, ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 3; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 22, crowd 57, drone 3, lowlight 15, pan 53, static 54, texture 2 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -29.6% | -30.3% |
| H.264 | -30.1% | -30.3% |
| H.264 Fast | -18.6% | -18.4% |
| H.264 Faster | -11.4% | -10.7% |
| H.264 Veryfast | -5.6% | -5.7% |
| HEVC Slow | -48.1% | -49.5% |
| HEVC | -46.4% | -47.8% |
| HEVC Fast | -41.2% | -42.4% |
| HEVC Faster | -33.1% | -33.8% |
| AV1 Slow | -46.6% | -48.9% |
| AV1 | -45.8% | -48.2% |
| AV1 Fast | -41.7% | -43.7% |
| AV1 Faster | -38.5% | -41.3% |
| AV1 Veryfast | -34.1% | -36.0% |
| VP9 Slow | -45.4% | -46.3% |
| VP9 | -44.9% | -45.9% |
| VP9 Fast | -39.9% | -41.0% |
| VP9 Faster | -34.0% | -34.6% |
| VP9 Veryfast | -30.1% | -31.3% |
| AV2 Slow | -50.4% | -51.9% |
| AV2 | -48.9% | -50.5% |
| AV2 Fast | -42.8% | -44.9% |
| AV2 Faster | -41.9% | -44.2% |
| VVC Slow | -51.2% | -52.5% |
| VVC | -50.7% | -52.0% |
| VVC Fast | -41.9% | -42.8% |
| openh264 | +13.7% | +17.2% |
| x264 ultrafast | +38.4% | +40.6% |
| x264 veryfast | +3.2% | +3.0% |
| x265 ultrafast | -21.7% | -22.4% |
| x265 superfast | -27.3% | -28.7% |
| x265 medium | -28.1% | -29.6% |
| SVT-AV1 M8 | -36.9% | -39.5% |
| SVT-AV1 M10 | -30.8% | -33.1% |
| SVT-AV1 M12 | -22.4% | -26.1% |
| SVT-AV1 M13 | -20.5% | -24.2% |
| libvpx VP9 rt 7 | +2.2% | +3.5% |
| libvpx VP9 rt 9 | -0.3% | -0.1% |
| libaom rt 8 | -24.4% | -25.0% |
| libaom rt 10 | -23.7% | -24.8% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +13.7% | +9.4% | +12.5% | +19.1% | 206/206 |  |
| openh264, adaptive quantisation on | +13.7% | +9.4% | +12.5% | +19.1% | 206/206 | +0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +38.4% | +51.8% | +13.8% | +49.5% | 206/206 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +38.4% | +51.9% | +13.7% | +49.8% | 206/206 | +0.1 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +34.8% | +49.1% | +27.4% | +27.9% | 206/206 | -0.3 [-1.9, +1.3] not distinguishable |
| x264 veryfast | +3.2% | +0.9% | +4.5% | +4.3% | 206/206 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +1.7% | -0.3% | -9.0% | +14.3% | 206/206 | -1.6 [-2.9, -0.4] better |
| x264 veryfast, metric-tuned (--tune ssim) | +4.1% | +2.4% | +7.0% | +3.0% | 206/206 | +0.9 [+0.4, +1.5] worse |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 206/206 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -10.0% | -10.2% | -19.1% | -0.8% | 206/206 | -10.0 [-11.1, -8.9] better |
| x264 medium, metric-tuned (--tune ssim) | -2.4% | -2.8% | -0.1% | -4.5% | 206/206 | -2.4 [-3.2, -1.7] better |
| x265 ultrafast | -21.7% | -20.7% | -23.6% | -20.6% | 206/206 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -22.0% | -22.6% | -23.5% | -19.9% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -22.0% | -22.6% | -23.5% | -19.9% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 superfast | -27.3% | -27.0% | -29.0% | -25.8% | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -28.5% | -29.9% | -29.7% | -26.0% | 206/206 | -2.0 [-2.4, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -28.5% | -29.9% | -29.7% | -26.0% | 206/206 | -2.0 [-2.4, -1.6] better |
| x265 medium | -28.1% | -28.0% | -27.1% | -29.3% | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -31.3% | -33.5% | -32.4% | -27.9% | 206/206 | -4.9 [-5.5, -4.3] better |
| x265 medium, metric-tuned (--tune ssim) | -29.7% | -30.9% | -28.4% | -29.9% | 206/206 | -2.3 [-2.5, -2.0] better |
| libvpx VP9 rt 7 | +2.2% | +6.8% | -3.3% | +3.0% | 206/206 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +1.3% | +6.3% | +0.9% | -3.1% | 206/206 | -1.8 [-2.5, -1.0] better |
| libvpx VP9 rt 9 | -0.3% | +5.5% | -5.5% | -0.9% | 206/206 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | -1.4% | +5.1% | -6.0% | -3.3% | 206/206 | -1.3 [-1.8, -0.8] better |
| libaom rt 8 | -24.4% | -23.5% | -26.1% | -23.6% | 206/206 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -24.1% | -21.7% | -21.6% | -29.1% | 206/206 | +1.0 [+0.2, +1.7] worse |
| libaom rt 10 | -23.7% | -24.7% | -24.1% | -22.4% | 206/206 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -24.1% | -23.3% | -21.0% | -28.0% | 206/206 | +0.4 [-0.4, +1.1] not distinguishable |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -44.7% | -44.5% | -42.6% | -47.0% | 205/206 |
| **VVC** | -44.1% | -43.9% | -41.5% | -46.8% | 205/206 |
| **AV2 Slow** | -43.9% | -46.6% | -40.5% | -44.5% | 205/206 |
| **AV2** | -42.2% | -45.1% | -38.9% | -42.4% | 205/206 |
| **HEVC Slow** | -41.2% | -40.8% | -38.3% | -44.4% | 205/206 |
| **AV1 Slow** | -39.5% | -42.5% | -34.6% | -41.4% | 205/206 |
| **HEVC** | -39.3% | -39.4% | -36.9% | -41.6% | 205/206 |
| **AV1** | -38.6% | -41.6% | -33.6% | -40.7% | 205/206 |
| **VP9 Slow** | -38.1% | -40.1% | -33.5% | -40.7% | 205/206 |
| **VP9** | -37.5% | -39.7% | -32.8% | -40.1% | 205/206 |
| **AV2 Fast** | -35.0% | -36.6% | -29.0% | -39.3% | 205/206 |
| **AV2 Faster** | -34.1% | -35.4% | -28.3% | -38.5% | 205/206 |
| **VVC Fast** | -34.1% | -31.5% | -30.7% | -39.9% | 205/206 |
| **AV1 Fast** | -33.8% | -36.1% | -27.2% | -38.1% | 205/206 |
| **HEVC Fast** | -33.3% | -33.1% | -28.7% | -38.1% | 205/206 |
| **VP9 Fast** | -31.7% | -33.5% | -24.9% | -36.8% | 205/206 |
| **AV1 Faster** | -30.4% | -30.5% | -25.9% | -34.7% | 205/206 |
| SVT-AV1 M8 | -27.9% | -31.7% | -14.7% | -37.3% | 205/206 |
| **AV1 Veryfast** | -25.4% | -23.4% | -19.8% | -32.9% | 205/206 |
| **VP9 Faster** | -25.4% | -23.3% | -21.4% | -31.4% | 205/206 |
| **HEVC Faster** | -24.3% | -22.8% | -18.3% | -31.7% | 205/206 |
| x265 medium, metric-tuned (--tune psnr) | -22.4% | -23.3% | -15.2% | -28.7% | 206/206 |
| **H.264** | -21.2% | -23.7% | -17.7% | -22.3% | 206/206 |
| **VP9 Veryfast** | -21.2% | -20.2% | -18.4% | -24.9% | 205/206 |
| SVT-AV1 M10 | -20.9% | -22.7% | -9.5% | -30.5% | 205/206 |
| **H.264 Slow** | -20.8% | -23.8% | -16.8% | -21.6% | 206/206 |
| x265 medium, metric-tuned (--tune ssim) | -20.2% | -20.2% | -9.9% | -30.6% | 206/206 |
| x265 superfast, metric-tuned (--tune psnr) | -19.0% | -18.7% | -11.6% | -26.7% | 206/206 |
| x265 superfast, metric-tuned (--tune ssim) | -19.0% | -18.7% | -11.6% | -26.7% | 206/206 |
| x265 medium | -18.3% | -16.7% | -8.4% | -29.9% | 206/206 |
| x265 superfast | -17.5% | -15.5% | -10.7% | -26.4% | 206/206 |
| libaom rt 8 | -14.2% | -12.1% | -6.9% | -24.0% | 205/206 |
| libaom rt 8, metric-tuned (--tune ssim) | -13.6% | -10.3% | -1.4% | -29.4% | 205/206 |
| libaom rt 10, metric-tuned (--tune ssim) | -13.4% | -11.5% | -0.5% | -28.5% | 205/206 |
| libaom rt 10 | -13.3% | -13.0% | -4.2% | -22.9% | 205/206 |
| SVT-AV1 M12 | -11.8% | -11.7% | -1.6% | -22.1% | 206/206 |
| x265 ultrafast, metric-tuned (--tune psnr) | -11.6% | -10.7% | -3.9% | -20.2% | 206/206 |
| x265 ultrafast, metric-tuned (--tune ssim) | -11.6% | -10.7% | -3.9% | -20.2% | 206/206 |
| x265 ultrafast | -11.2% | -8.7% | -3.9% | -21.0% | 206/206 |
| SVT-AV1 M13 | -9.8% | -9.2% | +0.4% | -20.6% | 206/206 |
| **H.264 Fast** | -8.5% | -10.7% | -5.2% | -9.7% | 206/206 |
| **H.264 Faster** | -0.5% | -3.9% | +5.0% | -2.5% | 206/206 |
| **H.264 Veryfast** | +5.9% | +1.5% | +12.0% | +4.0% | 206/206 |
| x264 medium, metric-tuned (--tune ssim) | +10.2% | +8.8% | +25.1% | -3.4% | 206/206 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.5% | +19.7% | +18.1% | -3.4% | 206/206 |
| libvpx VP9 rt 9 | +12.7% | +20.3% | +18.8% | -0.9% | 206/206 |
| x264 medium | +12.8% | +12.1% | +24.9% | +1.3% | 206/206 |
| x264 veryfast, metric-tuned (--tune psnr) | +13.6% | +11.7% | +13.7% | +15.5% | 206/206 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.0% | +19.4% | +25.1% | -2.5% | 206/206 |
| libvpx VP9 rt 7 | +15.1% | +20.6% | +21.0% | +3.6% | 206/206 |
| x264 veryfast | +17.1% | +13.4% | +32.5% | +5.4% | 206/206 |
| x264 veryfast, metric-tuned (--tune ssim) | +18.3% | +15.1% | +35.7% | +4.0% | 206/206 |
| openh264 | +29.1% | +25.3% | +42.1% | +19.9% | 206/206 |
| openh264, adaptive quantisation on | +29.1% | +25.3% | +42.1% | +19.9% | 206/206 |
| x264 ultrafast, metric-tuned (--tune ssim) | +53.1% | +66.0% | +63.3% | +29.9% | 206/206 |
| x264 ultrafast | +54.9% | +68.2% | +44.2% | +52.2% | 206/206 |
| x264 ultrafast, metric-tuned (--tune psnr) | +54.9% | +68.3% | +44.0% | +52.5% | 206/206 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **pool_ducks_take_off_pristine** (texture): no reliable overlap for SVT-AV1 M8, SVT-AV1 M10, libvpx VP9 rt 9.
- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.
- **ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508** (static): no reliable overlap for VVC Slow.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | pool_ducks_take_off_pristine | ind_bwc_videotext_video88 | ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508 |
|---|---:|---:|---:|
| VVC Slow | -67.9% | -65.8% | n/a |
| VVC | -68.1% | -65.4% | -68.3% |
| AV2 Slow | -62.7% | -65.1% | -68.5% |
| AV2 | -62.3% | -64.6% | -67.5% |
| HEVC Slow | -56.1% | -56.1% | -67.6% |
| AV1 Slow | -58.2% | -56.1% | -66.4% |
| HEVC | -52.9% | -56.8% | -66.6% |
| AV1 | -57.2% | -55.3% | -67.8% |
| VP9 Slow | -50.0% | -56.0% | -63.1% |
| VP9 | -50.6% | -55.3% | -63.3% |
| AV2 Fast | -52.9% | -65.8% | -66.5% |
| AV2 Faster | -53.5% | -64.9% | -64.2% |
| VVC Fast | -56.6% | -63.6% | -59.5% |
| AV1 Fast | -53.1% | -57.9% | -61.9% |
| HEVC Fast | -43.5% | -56.1% | -59.4% |
| VP9 Fast | -45.8% | -56.2% | -59.6% |
| AV1 Faster | -51.2% | -56.9% | -62.9% |
| SVT-AV1 M8 | n/a | -60.8% | -55.6% |
| AV1 Veryfast | -52.0% | -55.7% | -62.1% |
| VP9 Faster | -46.3% | -58.3% | -56.1% |
| HEVC Faster | -31.4% | -50.2% | -57.8% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| SVT-AV1 M10 | n/a | -56.2% | -52.5% |
| VP9 Veryfast | -48.7% | -54.2% | -56.2% |
| H.264 | -11.4% | -22.1% | -60.6% |
| H.264 Slow | -11.2% | -20.6% | -59.7% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 medium | -49.5% | -35.4% | -40.3% |
| x265 superfast | -47.3% | -47.1% | -42.3% |
| libaom rt 8 | -18.7% | n/a | -22.4% |
| libaom rt 10 | -16.0% | -38.6% | -32.2% |
| SVT-AV1 M12 | -35.1% | -50.6% | -53.7% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 ultrafast | -47.2% | -45.2% | -41.7% |
| SVT-AV1 M13 | -34.7% | -49.0% | -52.3% |
| H.264 Fast | -0.6% | -21.7% | -54.8% |
| H.264 Faster | +20.6% | -16.9% | -52.7% |
| x264 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| H.264 Veryfast | +9.6% | -11.9% | -53.5% |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9 | n/a | -19.2% | -13.7% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| libvpx VP9 rt 7 | -6.0% | -13.6% | +24.9% |
| x264 veryfast | +2.7% | -3.5% | -22.7% |
| openh264 | +89.8% | +32.2% | -66.5% |
| x264 ultrafast | +15.0% | +10.5% | -10.6% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slow | openh264 | -34.6% | -35.8% | -37.4% | -46.6 … -26.8 | 209/209 |
| H.264 | openh264 | -35.4% | -35.9% | -38.8% | -46.2 … -26.4 | 209/209 |
| H.264 | x264 veryfast | -35.7% | -37.6% | -40.5% | -45.1 … -30.0 | 209/209 |
| H.264 | x264 medium | -29.3% | -33.0% | -33.4% | -41.8 … -21.5 | 209/209 |
| H.264 Fast | openh264 | -21.6% | -21.9% | -26.7% | -36.3 … -13.0 | 209/209 |
| H.264 Fast | x264 veryfast | -24.6% | -27.0% | -29.7% | -33.9 … -18.3 | 209/209 |
| H.264 Fast | H.264 | +19.5% | +19.2% | +19.2% | +14.4 … +25.2 | 209/209 |
| H.264 Faster | openh264 | -13.2% | -14.5% | -17.4% | -27.8 … -5.1 | 209/209 |
| H.264 Faster | x264 ultrafast | -38.3% | -49.4% | -27.8% | -48.4 … -34.7 | 200/209 |
| H.264 Faster | H.264 Fast | +10.9% | +9.2% | +13.5% | +5.1 … +15.2 | 209/209 |
| H.264 Veryfast | openh264 | -6.1% | -8.2% | -10.4% | -24.4 … +3.2 | 208/209 |
| H.264 Veryfast | x264 ultrafast | -34.3% | -45.8% | -24.8% | -46.2 … -31.3 | 201/209 |
| H.264 Veryfast | H.264 Faster | +7.7% | +7.7% | +6.5% | +4.9 … +9.5 | 209/209 |
| HEVC Slow | SVT-AV1 M12 | -33.1% | -33.3% | -37.5% | -40.9 … -29.4 | 209/209 |
| HEVC | SVT-AV1 M12 | -29.9% | -30.9% | -35.5% | -38.4 … -26.9 | 209/209 |
| HEVC | x265 medium | -24.2% | -27.4% | -29.9% | -38.0 … -18.2 | 209/209 |
| HEVC Fast | SVT-AV1 M12 | -22.1% | -22.3% | -25.9% | -29.6 … -19.9 | 209/209 |
| HEVC Fast | x265 superfast | -17.9% | -20.7% | -18.6% | -31.1 … -13.1 | 209/209 |
| HEVC Fast | HEVC | +12.3% | +13.3% | +16.1% | +8.9 … +16.3 | 209/209 |
| HEVC Faster | SVT-AV1 M12 | -8.7% | -6.3% | -12.9% | -17.4 … -2.1 | 209/209 |
| HEVC Faster | SVT-AV1 M13 | -11.3% | -9.3% | -15.2% | -19.3 … -5.5 | 209/209 |
| HEVC Faster | HEVC Fast | +18.7% | +22.5% | +18.8% | +12.8 … +27.6 | 209/209 |
| AV1 Slow | SVT-AV1 M12 | -31.4% | -35.7% | -33.8% | -40.3 … -26.5 | 209/209 |
| AV1 | SVT-AV1 M12 | -30.4% | -34.7% | -32.7% | -39.4 … -25.3 | 209/209 |
| AV1 | openh264 | -51.4% | -53.1% | -51.7% | -64.0 … -42.2 | 203/209 |
| AV1 | SVT-AV1 M8 | -8.5% | -9.7% | -17.6% | -22.0 … -5.2 | 209/209 |
| AV1 | libaom rt 8 | -26.1% | -32.0% | -27.2% | -37.2 … -20.8 | 209/209 |
| AV1 Fast | SVT-AV1 M12 | -24.1% | -27.5% | -25.3% | -32.6 … -20.4 | 209/209 |
| AV1 Fast | openh264 | -46.0% | -47.3% | -44.8% | -61.7 … -36.6 | 206/209 |
| AV1 Fast | AV1 | +10.1% | +11.9% | +11.9% | +5.9 … +14.0 | 209/209 |
| AV1 Faster | SVT-AV1 M12 | -18.9% | -19.8% | -23.1% | -30.6 … -15.3 | 209/209 |
| AV1 Faster | AV1 Fast | +6.6% | +10.7% | +2.8% | +0.2 … +9.4 | 209/209 |
| AV1 Veryfast | SVT-AV1 M12 | -12.3% | -9.6% | -16.1% | -25.6 … -6.5 | 209/209 |
| AV1 Veryfast | SVT-AV1 M13 | -15.0% | -12.8% | -18.6% | -27.9 … -8.6 | 209/209 |
| AV1 Veryfast | AV1 Faster | +9.0% | +14.1% | +9.8% | +4.3 … +17.7 | 209/209 |
| VP9 Slow | HEVC Slow | +6.7% | +2.1% | +9.7% | +3.4 … +11.7 | 209/209 |
| VP9 | HEVC | +3.5% | -0.5% | +7.6% | +1.2 … +9.2 | 209/209 |
| VP9 | libvpx VP9 rt 7 | -43.1% | -47.6% | -41.7% | -51.2 … -37.1 | 209/209 |
| VP9 Fast | HEVC Fast | +2.9% | -0.9% | +6.6% | -1.6 … +8.0 | 209/209 |
| VP9 Fast | libvpx VP9 rt 9 | -37.9% | -44.5% | -35.5% | -47.3 … -33.5 | 208/209 |
| VP9 Fast | VP9 | +11.1% | +12.1% | +14.1% | +6.8 … +15.1 | 209/209 |
| VP9 Faster | HEVC Faster | -2.4% | -3.1% | -4.0% | -5.1 … +2.4 | 209/209 |
| VP9 Faster | libvpx VP9 rt 9 | -30.6% | -34.0% | -31.3% | -41.6 … -24.3 | 209/209 |
| VP9 Faster | VP9 Fast | +12.8% | +19.9% | +6.6% | +8.6 … +18.1 | 209/209 |
| VP9 Veryfast | HEVC Faster | +5.2% | +2.4% | +0.2% | -0.9 … +10.5 | 209/209 |
| VP9 Veryfast | libvpx VP9 rt 9 | -25.4% | -30.7% | -27.8% | -37.6 … -19.2 | 209/209 |
| VP9 Veryfast | VP9 Faster | +8.1% | +5.7% | +4.9% | +2.0 … +12.5 | 209/209 |
| AV2 Slow | AV1 Slow | -8.1% | -7.8% | -9.8% | -10.2 … -3.3 | 209/209 |
| AV2 | AV1 | -6.2% | -6.3% | -8.7% | -8.3 … -1.3 | 209/209 |
| AV2 | SVT-AV1 M8 | -14.6% | -15.7% | -25.0% | -25.0 … -10.2 | 209/209 |
| AV2 Fast | AV1 Fast | -2.4% | -0.4% | -3.4% | -8.1 … +2.2 | 209/209 |
| AV2 Fast | AV2 | +14.7% | +19.0% | +18.3% | +9.5 … +17.9 | 209/209 |
| AV2 Faster | AV1 Faster | -6.7% | -8.0% | -4.4% | -10.1 … -1.6 | 209/209 |
| AV2 Faster | SVT-AV1 M12 | -25.3% | -26.9% | -27.1% | -33.1 … -21.8 | 209/209 |
| AV2 Faster | AV2 Fast | +2.0% | +2.4% | +1.6% | +0.1 … +2.7 | 209/209 |
| VVC Slow | HEVC Slow | -7.7% | -7.8% | -8.4% | -9.5 … -3.7 | 209/209 |
| VVC | HEVC | -10.3% | -9.6% | -9.5% | -11.8 … -6.5 | 209/209 |
| VVC | x265 medium | -31.9% | -34.4% | -36.8% | -42.9 … -24.6 | 209/209 |
| VVC Fast | HEVC Fast | -2.3% | +1.6% | -3.4% | -4.7 … +2.6 | 209/209 |
| VVC Fast | VVC | +22.9% | +28.0% | +24.5% | +17.5 … +29.3 | 209/209 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (AV1, AV1 Fast: 4a5b8dd1; VP9 Fast: b33ba3bb; H.264 Veryfast, VP9, AV1 Slow, VP9 Slow, AV2 Slow: bad14976; H.264, HEVC, HEVC Fast, HEVC Faster, AV1 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, VVC Slow, AV2 Faster: c9cff89f; AV1 Veryfast: d40f86b3); H.264 Fast, H.264 Faster, VP9 Faster, AV1 Slow, AV2 Slow, H.264 Veryfast, VP9, VP9 Slow, AV1, AV1 Fast timed on b33ba3bb (same streams), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.6 [-2.9, -0.4]%); x264 medium --tune psnr (-10.0 [-11.1, -8.9]%); x265 ultrafast --tune psnr (-0.6 [-1.0, -0.2]%); x265 superfast --tune psnr (-2.0 [-2.4, -1.6]%); x265 medium --tune psnr (-4.9 [-5.5, -4.3]%); libvpx VP9 rt 7 --tune ssim (-1.8 [-2.5, -1.0]%); libvpx VP9 rt 9 --tune ssim (-1.3 [-1.8, -0.8]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
