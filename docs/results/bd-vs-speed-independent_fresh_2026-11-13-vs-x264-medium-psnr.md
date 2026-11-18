# Bitrate vs encode speed, held-out clips (independent_fresh_2026-11-13), zero line x264-medium~psnr

<sub>Updated 2026-11-18, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, AV1, AV1 Fast, AV2, VVC: 4a5b8dd1; HEVC Fast, HEVC Faster, AV1 Faster, VP9 Fast, AV2 Fast, VVC Fast: b33ba3bb; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976; AV1 Veryfast: d40f86b3); H.264 Fast, H.264 Faster, VP9 Faster, VP9 Veryfast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow, AV1, AV1 Fast, AV2, H.264, VVC timed on b33ba3bb (same streams)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 138 clips of `independent_fresh_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -47.5% | -47.8% | -46.7% | -56.8 … -40.6 | 24.3 | 0.91× | 92.7 | 0.82× | 136/136 | Pareto; Pareto x86 |
| **VVC** | -46.9% | -47.0% | -45.9% | -56.8 … -39.9 | 27.9 | 1.05× | 118.9 | 1.05× | 136/136 | Pareto; Pareto x86 |
| **AV2 Slow** | -46.5% | -49.6% | -44.8% | -54.4 … -41.1 | 21.0 | 0.79× | 77.8 | 0.69× | 136/136 |  |
| **AV2** | -45.2% | -48.5% | -43.5% | -53.3 … -39.3 | 35.7 | 1.34× | 152.1 | 1.35× | 136/136 | Pareto; Pareto x86 |
| **HEVC Slow** | -44.8% | -44.6% | -43.3% | -53.1 … -37.2 | 30.1 | 1.13× | 117.1 | 1.04× | 136/136 |  |
| **HEVC** | -42.9% | -43.0% | -41.8% | -52.0 … -35.8 | 48.7 | 1.83× | 195.6 | 1.73× | 136/136 | Pareto; Pareto x86 |
| **AV1 Slow** | -42.3% | -46.0% | -39.5% | -52.2 … -37.0 | 28.7 | 1.08× | 112.2 | 1.00× | 136/136 | 1 rate-invalid pts |
| **AV1** | -41.4% | -45.2% | -38.4% | -51.3 … -36.3 | 44.1 | 1.65× | 172.7 | 1.53× | 136/136 | 1 rate-invalid pts |
| **VP9 Slow** | -40.4% | -42.9% | -36.8% | -49.1 … -33.4 | 36.3 | 1.36× | 145.2 | 1.29× | 136/136 | 3 rate-invalid pts |
| **VP9** | -39.9% | -42.6% | -36.4% | -48.2 … -32.7 | 42.9 | 1.61× | 168.8 | 1.50× | 136/136 | 3 rate-invalid pts |
| **AV2 Fast** | -39.0% | -40.2% | -36.7% | -46.3 … -30.2 | 47.8 | 1.79× | 204.8 | 1.82× | 136/136 | Pareto x86 |
| **VVC Fast** | -37.7% | -36.4% | -35.9% | -46.1 … -28.5 | 53.7 | 2.01× | 212.9 | 1.89× | 136/136 | Pareto; Pareto x86 |
| **AV1 Fast** | -37.2% | -40.4% | -33.1% | -46.0 … -30.9 | 64.4 | 2.41× | 265.6 | 2.36× | 136/136 | Pareto; Pareto x86 |
| **HEVC Fast** | -37.0% | -37.2% | -34.1% | -44.6 … -29.7 | 75.8 | 2.84× | 303.8 | 2.69× | 136/136 | Pareto; Pareto x86 |
| **VP9 Fast** | -35.1% | -37.7% | -29.6% | -44.3 … -26.9 | 53.9 | 2.02× | 216.2 | 1.92× | 136/136 |  |
| SVT-AV1 M8 | -34.8% | -36.7% | -26.7% | -39.0 … -26.0 | 14.3 | 0.53× | 80.8 | 0.72× | 136/136 | 23 rate-invalid pts |
| **AV1 Faster** | -34.3% | -36.2% | -32.0% | -45.3 … -27.3 | 78.2 | 2.93× | 320.3 | 2.84× | 136/136 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -29.2% | -29.1% | -25.4% | -41.3 … -19.8 | 87.7 | 3.29× | 410.7 | 3.64× | 136/136 | Pareto; Pareto x86 |
| **VP9 Faster** | -28.2% | -27.6% | -25.6% | -39.2 … -18.3 | 90.4 | 3.39× | 355.5 | 3.15× | 136/136 | Pareto; 1 rate-invalid pts |
| SVT-AV1 M10 | -27.5% | -28.2% | -20.9% | -32.3 … -18.7 | 39.3 | 1.47× | 228.9 | 2.03× | 136/136 | 21 rate-invalid pts |
| **HEVC Faster** | -27.3% | -26.2% | -23.5% | -36.2 … -18.4 | 98.9 | 3.71× | 399.6 | 3.54× | 136/136 | Pareto |
| x265 medium, metric-tuned (--tune psnr) | -26.0% | -26.5% | -20.9% | -32.0 … -18.4 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 29 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 superfast, metric-tuned (--tune psnr) | -23.5% | -22.8% | -18.2% | -28.8 … -16.7 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 27 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 superfast | -21.9% | -19.7% | -17.2% | -26.2 … -14.5 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 28 rate-invalid pts |
| **H.264 Slow** | -21.7% | -25.6% | -18.5% | -35.7 … -14.4 | 46.4 | 1.74× | 138.1 | 1.23× | 136/136 |  |
| x265 medium | -21.6% | -19.6% | -14.2% | -29.0 … -15.1 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 31 rate-invalid pts |
| **VP9 Veryfast** | -20.9% | -20.1% | -20.5% | -35.5 … -11.1 | 102.7 | 3.85× | 358.2 | 3.18× | 136/136 | Pareto |
| **H.264** | -20.7% | -23.5% | -18.0% | -35.4 … -12.7 | 75.5 | 2.83× | 237.6 | 2.11× | 136/136 |  |
| SVT-AV1 M12 | -20.4% | -20.1% | -15.4% | -24.8 … -8.8 | 70.9 | 2.67× | 408.7 | 3.62× | 136/136 | 18 rate-invalid pts |
| SVT-AV1 M13 | -18.2% | -17.6% | -13.5% | -24.3 … -6.8 | 79.2 | 2.96× | 445.4 | 3.95× | 136/136 | Pareto x86; 18 rate-invalid pts |
| libaom rt 8 | -17.1% | -14.8% | -11.6% | -23.9 … -9.9 | 17.4 | 0.65× | 68.8 | 0.61× | 136/136 | 84 rate-invalid pts |
| libaom rt 10 | -16.7% | -16.5% | -9.6% | -23.5 … -8.9 | 22.3 | 0.83× | 90.5 | 0.80× | 136/136 | 87 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -16.0% | -14.4% | -10.7% | -23.4 … -6.1 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 27 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -15.3% | -12.3% | -10.4% | -22.1 … -5.4 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 27 rate-invalid pts |
| **H.264 Fast** | -11.6% | -14.7% | -10.2% | -25.7 … -0.3 | 118.8 | 4.46× | 389.6 | 3.45× | 136/136 | Pareto |
| **H.264 Faster** | -3.9% | -8.5% | -0.4% | -22.4 … +14.1 | 142.9 | 5.36× | 454.8 | 4.03× | 136/136 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 39 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Veryfast** | +1.1% | -3.6% | +5.0% | -18.5 … +19.7 | 164.0 | 6.15× | 542.7 | 4.81× | 136/136 | Pareto; Pareto x86 |
| x264 veryfast, metric-tuned (--tune psnr) | +8.8% | +5.5% | +7.9% | -0.8 … +18.7 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 medium, metric-tuned (--tune ssim) | +11.1% | +8.8% | +26.9% | +6.9 … +14.7 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 35 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +11.4% | +18.3% | +16.3% | +5.6 … +18.6 | 47.8 | 1.78× | 204.4 | 1.81× | 136/136 | 150 rate-invalid pts |
| x264 veryfast | +12.3% | +6.6% | +27.4% | +2.4 … +21.5 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts |
| x264 medium | +13.6% | +11.4% | +27.5% | +8.1 … +16.5 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 36 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.3% | +19.1% | +23.7% | +4.2 … +20.8 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 124 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +14.3% | +19.0% | +18.6% | +4.6 … +23.2 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 144 rate-invalid pts |
| openh264 | +31.3% | +27.6% | +41.8% | -2.4 … +63.5 | 111.5 | 4.17× | 524.7 | 4.65× | 136/136 | 241 rate-invalid pts |
| x264 ultrafast | +46.4% | +54.2% | +36.2% | +22.7 … +68.0 | 190.5 | 7.11× | 1017.0 | 9.02× | 136/136 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 2; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 12, crowd 38, lowlight 8, pan 42, static 36 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +13.6% | +12.6% |
| H.264 Slow | -21.7% | -23.8% |
| H.264 | -20.7% | -22.7% |
| H.264 Fast | -11.6% | -12.8% |
| H.264 Faster | -3.9% | -4.6% |
| H.264 Veryfast | +1.1% | +0.8% |
| HEVC Slow | -44.8% | -45.4% |
| HEVC | -42.9% | -43.6% |
| HEVC Fast | -37.0% | -37.4% |
| HEVC Faster | -27.3% | -27.6% |
| AV1 Slow | -42.3% | -44.5% |
| AV1 | -41.4% | -43.6% |
| AV1 Fast | -37.2% | -38.6% |
| AV1 Faster | -34.3% | -36.0% |
| AV1 Veryfast | -29.2% | -30.4% |
| VP9 Slow | -40.4% | -41.2% |
| VP9 | -39.9% | -40.7% |
| VP9 Fast | -35.1% | -35.5% |
| VP9 Faster | -28.2% | -28.3% |
| VP9 Veryfast | -20.9% | -21.7% |
| AV2 Slow | -46.5% | -47.7% |
| AV2 | -45.2% | -46.2% |
| AV2 Fast | -39.0% | -39.1% |
| VVC Slow | -47.5% | -48.4% |
| VVC | -46.9% | -47.9% |
| VVC Fast | -37.7% | -37.5% |
| openh264 | +31.3% | +28.9% |
| x264 ultrafast | +46.4% | +51.2% |
| x264 veryfast | +12.3% | +12.6% |
| x265 ultrafast | -15.3% | -14.1% |
| x265 superfast | -21.9% | -21.0% |
| x265 medium | -21.6% | -21.5% |
| SVT-AV1 M8 | -34.8% | -33.3% |
| SVT-AV1 M10 | -27.5% | -26.3% |
| SVT-AV1 M12 | -20.4% | -18.8% |
| SVT-AV1 M13 | -18.2% | -17.0% |
| libvpx VP9 rt 7 | +14.3% | +16.6% |
| libvpx VP9 rt 9 | +11.4% | +12.0% |
| libaom rt 8 | -17.1% | -16.1% |
| libaom rt 10 | -16.7% | -16.0% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +31.3% | +27.6% | +41.8% | +24.4% | 136/136 |  |
| openh264, adaptive quantisation on | +31.3% | +27.6% | +41.8% | +24.4% | 136/136 | +0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +46.4% | +54.2% | +36.2% | +48.8% | 136/136 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +46.3% | +53.9% | +36.2% | +48.8% | 136/136 | -0.0 [-0.2, +0.2] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +44.9% | +50.8% | +57.3% | +26.6% | 136/136 | +0.2 [-1.3, +1.6] not distinguishable |
| x264 veryfast | +12.3% | +6.6% | +27.4% | +2.8% | 136/136 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +8.8% | +5.5% | +7.9% | +13.1% | 136/136 | -1.9 [-3.0, -0.8] better |
| x264 veryfast, metric-tuned (--tune ssim) | +13.4% | +8.4% | +30.2% | +1.7% | 136/136 | +0.7 [+0.1, +1.2] worse |
| x264 medium | +13.6% | +11.4% | +27.5% | +1.9% | 136/136 |  |
| x264 medium, metric-tuned (--tune ssim) ● | +11.1% | +8.8% | +26.9% | -2.5% | 136/136 | -2.2 [-3.0, -1.5] better |
| x265 ultrafast | -15.3% | -12.3% | -10.4% | -23.1% | 136/136 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -16.0% | -14.4% | -10.7% | -22.9% | 136/136 | -1.1 [-1.4, -0.7] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -16.0% | -14.4% | -10.7% | -22.9% | 136/136 | -1.1 [-1.4, -0.7] better |
| x265 superfast | -21.9% | -19.7% | -17.2% | -28.9% | 136/136 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -23.5% | -22.8% | -18.2% | -29.4% | 136/136 | -2.1 [-2.4, -1.7] better |
| x265 superfast, metric-tuned (--tune ssim) | -23.5% | -22.8% | -18.2% | -29.4% | 136/136 | -2.1 [-2.4, -1.7] better |
| x265 medium | -21.6% | -19.6% | -14.2% | -30.9% | 136/136 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -26.0% | -26.5% | -20.9% | -30.5% | 136/136 | -5.7 [-6.2, -5.0] better |
| x265 medium, metric-tuned (--tune ssim) | -23.1% | -22.5% | -15.2% | -31.5% | 136/136 | -2.0 [-2.3, -1.7] better |
| libvpx VP9 rt 7 | +14.3% | +19.0% | +18.6% | +5.2% | 136/136 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +14.3% | +19.1% | +23.7% | +0.0% | 136/136 | -0.8 [-1.3, -0.3] better |
| libvpx VP9 rt 9 | +11.4% | +18.3% | +16.3% | -0.4% | 136/136 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.5% | +18.8% | +17.2% | -1.6% | 136/136 | -0.2 [-0.7, +0.3] not distinguishable |
| libaom rt 8 | -17.1% | -14.8% | -11.6% | -25.1% | 136/136 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -16.7% | -13.3% | -6.1% | -30.6% | 136/136 | +0.9 [+0.2, +1.7] worse |
| libaom rt 10 | -16.7% | -16.5% | -9.6% | -24.1% | 136/136 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -16.6% | -14.9% | -5.5% | -29.3% | 136/136 | +0.7 [+0.1, +1.4] worse |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -54.2% | -56.1% | -59.0% | -47.5% | 136/136 |
| **VVC** | -53.7% | -55.5% | -58.4% | -47.1% | 136/136 |
| **AV2 Slow** | -53.1% | -57.6% | -57.2% | -44.5% | 136/136 |
| **AV2** | -52.0% | -56.7% | -56.3% | -43.2% | 136/136 |
| **HEVC Slow** | -51.8% | -52.9% | -56.3% | -46.1% | 136/136 |
| **HEVC** | -50.1% | -51.7% | -55.2% | -43.4% | 136/136 |
| **AV1 Slow** | -49.3% | -53.9% | -53.1% | -41.0% | 136/136 |
| **AV1** | -48.6% | -53.2% | -52.3% | -40.3% | 136/136 |
| **VP9 Slow** | -47.7% | -51.2% | -51.2% | -40.9% | 136/136 |
| **VP9** | -47.5% | -51.1% | -50.9% | -40.4% | 136/136 |
| **AV2 Fast** | -46.8% | -49.4% | -51.0% | -39.8% | 136/136 |
| **VVC Fast** | -45.8% | -46.2% | -50.7% | -40.4% | 136/136 |
| **HEVC Fast** | -45.1% | -46.7% | -49.3% | -39.3% | 136/136 |
| **AV1 Fast** | -45.0% | -49.0% | -48.2% | -37.8% | 136/136 |
| **VP9 Fast** | -43.4% | -46.9% | -45.6% | -37.6% | 136/136 |
| SVT-AV1 M8 | -43.3% | -46.4% | -42.9% | -40.5% | 136/136 |
| **AV1 Faster** | -42.4% | -45.4% | -47.4% | -34.6% | 136/136 |
| **AV1 Veryfast** | -37.9% | -38.9% | -42.2% | -32.7% | 136/136 |
| **VP9 Faster** | -37.2% | -37.9% | -42.5% | -31.3% | 136/136 |
| SVT-AV1 M10 | -37.1% | -39.2% | -38.7% | -33.2% | 136/136 |
| **HEVC Faster** | -36.5% | -36.4% | -41.0% | -32.2% | 136/136 |
| x265 medium, metric-tuned (--tune psnr) | -35.0% | -36.4% | -38.3% | -30.3% | 136/136 |
| x265 superfast, metric-tuned (--tune psnr) | -33.0% | -33.5% | -36.3% | -29.3% | 136/136 |
| x265 superfast, metric-tuned (--tune ssim) | -33.0% | -33.5% | -36.3% | -29.3% | 136/136 |
| x265 medium, metric-tuned (--tune ssim) | -32.8% | -33.0% | -33.9% | -31.3% | 136/136 |
| x265 superfast | -31.8% | -30.9% | -35.6% | -28.7% | 136/136 |
| x265 medium | -31.5% | -30.6% | -33.2% | -30.7% | 136/136 |
| **H.264 Slow** | -31.3% | -35.5% | -36.7% | -21.6% | 136/136 |
| **VP9 Veryfast** | -30.6% | -30.8% | -38.7% | -22.5% | 136/136 |
| SVT-AV1 M12 | -30.6% | -31.7% | -34.3% | -25.8% | 136/136 |
| **H.264** | -30.3% | -33.7% | -36.3% | -21.0% | 136/136 |
| SVT-AV1 M13 | -28.6% | -29.3% | -32.7% | -23.8% | 136/136 |
| libaom rt 10, metric-tuned (--tune ssim) | -27.7% | -26.9% | -26.9% | -29.4% | 136/136 |
| libaom rt 10 | -27.7% | -28.4% | -30.4% | -24.2% | 136/136 |
| libaom rt 8, metric-tuned (--tune ssim) | -27.6% | -25.0% | -27.2% | -30.7% | 136/136 |
| libaom rt 8 | -27.6% | -26.2% | -31.6% | -25.2% | 135/136 |
| x265 ultrafast, metric-tuned (--tune psnr) | -26.6% | -26.0% | -30.7% | -23.1% | 136/136 |
| x265 ultrafast, metric-tuned (--tune ssim) | -26.6% | -26.0% | -30.7% | -23.1% | 136/136 |
| x265 ultrafast | -25.9% | -24.1% | -30.4% | -23.2% | 136/136 |
| **H.264 Fast** | -22.1% | -25.5% | -29.9% | -11.0% | 136/136 |
| **H.264 Faster** | -15.2% | -19.5% | -22.0% | -4.2% | 136/136 |
| **H.264 Veryfast** | -10.6% | -14.8% | -17.5% | +0.5% | 136/136 |
| x264 medium, metric-tuned (--tune psnr) | -10.6% | -9.7% | -20.4% | -1.6% | 136/136 |
| x264 veryfast, metric-tuned (--tune psnr) | -3.1% | -5.2% | -15.3% | +11.3% | 136/136 |
| libvpx VP9 rt 9 | -2.8% | +3.1% | -10.2% | -1.4% | 136/136 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -2.7% | +3.7% | -9.4% | -2.5% | 136/136 |
| x264 medium, metric-tuned (--tune ssim) | -2.2% | -1.9% | -0.6% | -4.0% | 136/136 |
| x264 veryfast | -1.4% | -4.3% | -1.2% | +1.3% | 136/136 |
| x264 veryfast, metric-tuned (--tune ssim) | -0.4% | -2.7% | +1.0% | +0.4% | 136/136 |
| libvpx VP9 rt 7 | +0.5% | +4.9% | -7.4% | +3.8% | 136/136 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +0.6% | +5.5% | -2.2% | -1.4% | 136/136 |
| openh264 | +14.1% | +10.1% | +9.7% | +22.5% | 136/136 |
| openh264, adaptive quantisation on | +14.1% | +10.1% | +9.7% | +22.5% | 136/136 |
| x264 ultrafast, metric-tuned (--tune ssim) | +27.0% | +36.6% | +20.4% | +24.1% | 136/136 |
| x264 ultrafast, metric-tuned (--tune psnr) | +30.3% | +40.0% | +5.6% | +45.3% | 136/136 |
| x264 ultrafast | +30.3% | +40.2% | +5.5% | +45.2% | 136/136 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium, metric-tuned (--tune psnr)'s (even with its extra rungs, which the 50% source cap limits).

- **ind_bwc_videotext_video190** (bodycam): no reliable overlap for libaom rt 8, libaom rt 10.
- **ind_personpath22_uid_vid_00212** (crowd): no reliable overlap for HEVC Slow, HEVC, HEVC Fast, HEVC Faster, AV1 Slow, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9 Slow, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2 Slow, AV2, AV2 Fast, VVC Slow, VVC, VVC Fast, SVT-AV1 M8, SVT-AV1 M10.

Direct BD-J vs x264 medium, metric-tuned (--tune psnr) on them, where it exists:

| Encoder | ind_bwc_videotext_video190 | ind_personpath22_uid_vid_00212 |
|---|---:|---:|
| VVC Slow | -40.0% | n/a |
| VVC | -35.3% | n/a |
| AV2 Slow | -37.4% | n/a |
| AV2 | -33.6% | n/a |
| HEVC Slow | -38.7% | n/a |
| HEVC | -29.3% | n/a |
| AV1 Slow | -36.0% | n/a |
| AV1 | -31.9% | n/a |
| VP9 Slow | -35.1% | n/a |
| VP9 | -30.4% | n/a |
| AV2 Fast | -20.4% | n/a |
| VVC Fast | -20.3% | n/a |
| AV1 Fast | -19.0% | n/a |
| HEVC Fast | -15.4% | n/a |
| VP9 Fast | -20.7% | n/a |
| SVT-AV1 M8 | -23.3% | n/a |
| AV1 Faster | -19.9% | n/a |
| AV1 Veryfast | -24.9% | n/a |
| VP9 Faster | -17.8% | n/a |
| SVT-AV1 M10 | +5.7% | n/a |
| HEVC Faster | -10.3% | n/a |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 superfast | +10.6% | -9.9% |
| H.264 Slow | -18.1% | -34.2% |
| x265 medium | +0.8% | -17.4% |
| VP9 Veryfast | -16.5% | n/a |
| H.264 | -18.0% | -32.2% |
| SVT-AV1 M12 | +8.8% | -11.9% |
| SVT-AV1 M13 | +10.1% | -11.3% |
| libaom rt 8 | n/a | -16.9% |
| libaom rt 10 | n/a | -16.6% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 ultrafast | +20.2% | +5.7% |
| H.264 Fast | -8.0% | -24.3% |
| H.264 Faster | -6.4% | -16.6% |
| H.264 Veryfast | -3.3% | -12.5% |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a |
| x264 medium, metric-tuned (--tune ssim) | n/a | n/a |
| libvpx VP9 rt 9 | +8.6% | +19.5% |
| x264 veryfast | +24.1% | -9.1% |
| x264 medium | +2.6% | -13.2% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a |
| libvpx VP9 rt 7 | -10.5% | +28.3% |
| openh264 | +26.1% | +7.2% |
| x264 ultrafast | +64.9% | +58.7% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 | openh264 | -33.7% | -34.4% | -36.9% | -45.0 … -23.9 | 138/138 |
| H.264 | x264 veryfast | -34.7% | -36.4% | -40.7% | -43.3 … -28.0 | 138/138 |
| H.264 | x264 medium | -30.2% | -33.5% | -36.0% | -40.4 … -19.8 | 138/138 |
| H.264 Fast | x264 veryfast | -26.5% | -28.3% | -34.4% | -34.6 … -19.5 | 138/138 |
| H.264 Fast | openh264 | -23.7% | -25.0% | -29.1% | -36.9 … -13.2 | 138/138 |
| H.264 Fast | H.264 | +14.4% | +14.3% | +11.9% | +10.7 … +18.5 | 138/138 |
| H.264 Faster | x264 ultrafast | -40.9% | -50.1% | -32.0% | -49.6 … -36.6 | 130/138 |
| H.264 Faster | H.264 Fast | +10.5% | +8.6% | +12.7% | +5.2 … +15.4 | 138/138 |
| H.264 Veryfast | x264 ultrafast | -36.8% | -46.6% | -27.2% | -46.3 … -32.0 | 131/138 |
| H.264 Veryfast | H.264 Faster | +7.3% | +7.5% | +6.8% | +5.5 … +9.7 | 138/138 |
| HEVC | x265 medium | -27.9% | -31.1% | -33.7% | -38.5 … -20.0 | 138/138 |
| HEVC Fast | x265 superfast | -20.6% | -24.0% | -21.5% | -31.7 … -14.8 | 138/138 |
| HEVC | SVT-AV1 M12 | -29.8% | -30.8% | -33.0% | -39.1 … -27.6 | 138/138 |
| HEVC Fast | HEVC | +12.8% | +12.6% | +16.4% | +8.7 … +16.3 | 138/138 |
| AV1 | openh264 | -52.0% | -54.5% | -53.0% | -63.4 … -42.6 | 133/138 |
| AV1 Fast | openh264 | -47.4% | -49.5% | -47.6% | -60.9 … -37.6 | 136/138 |
| AV1 | SVT-AV1 M8 | -7.3% | -12.2% | -13.9% | -22.3 … -6.1 | 138/138 |
| AV1 | SVT-AV1 M12 | -27.8% | -33.8% | -28.3% | -39.6 … -27.4 | 138/138 |
| AV1 Fast | SVT-AV1 M12 | -21.9% | -27.0% | -21.7% | -32.8 … -20.9 | 138/138 |
| AV1 | libaom rt 8 | -25.6% | -32.6% | -26.8% | -38.3 … -21.7 | 138/138 |
| AV1 Fast | AV1 | +9.5% | +11.3% | +10.8% | +6.1 … +14.8 | 138/138 |
| AV1 Faster | SVT-AV1 M12 | -17.6% | -21.1% | -20.4% | -30.6 … -15.8 | 138/138 |
| AV1 Faster | AV1 Fast | +5.1% | +8.0% | +1.3% | +0.7 … +8.5 | 138/138 |
| AV1 Veryfast | SVT-AV1 M13 | -14.0% | -14.3% | -15.3% | -28.7 … -9.7 | 138/138 |
| AV1 Veryfast | AV1 Faster | +9.7% | +15.0% | +11.0% | +4.6 … +17.0 | 138/138 |
| HEVC Faster | SVT-AV1 M13 | -9.7% | -8.3% | -11.5% | -18.5 … -5.8 | 138/138 |
| HEVC Faster | HEVC Fast | +19.8% | +23.3% | +20.0% | +12.9 … +27.2 | 138/138 |
| VP9 | libvpx VP9 rt 7 | -43.8% | -48.7% | -42.2% | -52.6 … -38.4 | 138/138 |
| VP9 Fast | libvpx VP9 rt 9 | -40.3% | -47.1% | -37.6% | -48.2 … -35.8 | 137/138 |
| VP9 Fast | VP9 | +10.0% | +10.8% | +13.0% | +6.8 … +14.6 | 138/138 |
| VP9 Faster | libvpx VP9 rt 9 | -32.9% | -36.9% | -33.7% | -42.7 … -26.5 | 138/138 |
| VP9 Faster | VP9 Fast | +13.6% | +20.6% | +6.7% | +8.0 … +18.4 | 138/138 |
| VP9 Veryfast | libvpx VP9 rt 9 | -23.6% | -28.6% | -27.6% | -35.8 … -17.5 | 138/138 |
| VP9 Veryfast | VP9 Faster | +15.5% | +15.0% | +9.0% | +5.9 … +19.1 | 138/138 |
| AV2 | SVT-AV1 M8 | -13.7% | -17.7% | -21.9% | -25.1 … -12.1 | 138/138 |
| AV2 | AV1 | -6.6% | -6.2% | -8.9% | -8.5 … -1.1 | 138/138 |
| AV2 Fast | AV2 | +14.3% | +20.4% | +14.5% | +12.1 … +21.2 | 138/138 |
| VVC | x265 medium | -33.4% | -36.6% | -38.9% | -43.2 … -26.2 | 138/138 |
| VVC | HEVC | -8.6% | -8.8% | -8.2% | -11.2 … -6.6 | 138/138 |
| VVC Fast | VVC | +21.5% | +25.3% | +22.8% | +17.5 … +29.1 | 138/138 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, AV1, AV1 Fast, AV2, VVC: 4a5b8dd1; HEVC Fast, HEVC Faster, AV1 Faster, VP9 Fast, AV2 Fast, VVC Fast: b33ba3bb; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976; AV1 Veryfast: d40f86b3); H.264 Fast, H.264 Faster, VP9 Faster, VP9 Veryfast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow, AV1, AV1 Fast, AV2, H.264, VVC timed on b33ba3bb (same streams), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.9 [-3.0, -0.8]%); x264 medium --tune ssim (-2.2 [-3.0, -1.5]%); x265 ultrafast --tune psnr (-1.1 [-1.4, -0.7]%); x265 superfast --tune psnr (-2.1 [-2.4, -1.7]%); x265 medium --tune psnr (-5.7 [-6.2, -5.0]%); libvpx VP9 rt 7 --tune ssim (-0.8 [-1.3, -0.3]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
