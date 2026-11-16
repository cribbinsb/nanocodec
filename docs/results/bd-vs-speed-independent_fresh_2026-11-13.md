# Bitrate vs encode speed, held-out clips (independent_fresh_2026-11-13)

<sub>Updated 2026-11-16, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, HEVC Fast, AV1, AV1 Fast, AV1 Faster, AV2, VVC: 4a5b8dd1; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976); AV2 Fast, H.264 Fast, H.264 Faster, HEVC Faster, VP9 Fast, VP9 Faster, VP9 Veryfast, VVC Fast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow timed on 4a5b8dd1 (same streams)</sub>

Quality on the 138 clips of `independent_fresh_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_fresh_2026-11-13.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_fresh_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -53.7% | -55.5% | -58.5% | -62.0 … -45.8 | 23.3 | 0.88× | 89.3 | 0.79× | 137/137 | Pareto; Pareto x86 |
| **VVC** | -53.1% | -54.8% | -57.8% | -61.8 … -45.3 | 27.0 | 1.01× | 109.3 | 0.97× | 137/137 | Pareto; Pareto x86 |
| **AV2 Slow** | -52.6% | -57.0% | -56.7% | -60.1 … -45.0 | 19.3 | 0.72× | 71.4 | 0.63× | 137/137 |  |
| **AV2** | -51.5% | -56.0% | -55.7% | -59.2 … -43.1 | 32.5 | 1.22× | 140.8 | 1.25× | 137/137 | Pareto; Pareto x86 |
| **HEVC Slow** | -51.4% | -52.5% | -55.9% | -59.0 … -42.8 | 28.9 | 1.08× | 114.0 | 1.01× | 137/137 |  |
| **HEVC** | -49.5% | -51.0% | -54.7% | -58.2 … -40.8 | 46.6 | 1.75× | 179.9 | 1.60× | 137/137 | Pareto; Pareto x86 |
| **AV1 Slow** | -49.0% | -53.5% | -52.6% | -57.5 … -42.0 | 27.2 | 1.02× | 112.8 | 1.00× | 137/137 | 1 rate-invalid pts |
| **AV1** | -48.1% | -52.7% | -51.7% | -56.7 … -40.7 | 40.9 | 1.53× | 161.0 | 1.43× | 137/137 | 1 rate-invalid pts |
| **VP9 Slow** | -47.3% | -50.7% | -50.7% | -55.3 … -38.6 | 34.5 | 1.29× | 131.3 | 1.16× | 137/137 | 3 rate-invalid pts |
| **VP9** | -47.0% | -50.6% | -50.4% | -55.1 … -37.9 | 40.7 | 1.53× | 165.6 | 1.47× | 137/137 | 3 rate-invalid pts |
| **VVC Fast** | -45.0% | -45.4% | -49.9% | -53.2 … -35.5 | 49.4 | 1.85× | 190.5 | 1.69× | 137/137 | Pareto; Pareto x86 |
| **AV1 Fast** | -44.3% | -48.2% | -47.3% | -53.3 … -36.5 | 59.6 | 2.23× | 250.8 | 2.22× | 137/137 | Pareto; Pareto x86 |
| **AV2 Fast** | -43.8% | -46.4% | -47.8% | -51.8 … -33.9 | 44.6 | 1.67× | 207.7 | 1.84× | 137/137 |  |
| SVT-AV1 M8 | -42.6% | -45.7% | -42.2% | -46.8 … -32.2 | 14.3 | 0.53× | 80.8 | 0.72× | 137/137 | 23 rate-invalid pts |
| **HEVC Fast** | -41.4% | -42.2% | -45.1% | -49.8 … -32.8 | 78.0 | 2.92× | 303.5 | 2.69× | 137/137 | Pareto; Pareto x86 |
| **VP9 Fast** | -40.4% | -42.5% | -44.6% | -48.0 … -31.2 | 54.2 | 2.03× | 205.1 | 1.82× | 137/137 | 1 rate-invalid pts |
| **AV1 Faster** | -38.8% | -40.3% | -43.9% | -49.8 … -29.2 | 74.6 | 2.80× | 346.7 | 3.07× | 137/137 | Pareto x86 |
| **VP9 Faster** | -36.5% | -37.1% | -41.6% | -47.2 … -27.3 | 86.5 | 3.24× | 328.1 | 2.91× | 137/137 | Pareto; 1 rate-invalid pts |
| SVT-AV1 M10 | -36.0% | -38.1% | -37.7% | -42.2 … -25.1 | 39.3 | 1.47× | 228.9 | 2.03× | 137/137 | 21 rate-invalid pts |
| **HEVC Faster** | -34.9% | -34.6% | -39.3% | -43.3 … -25.2 | 98.9 | 3.71× | 409.9 | 3.64× | 137/137 | Pareto; Pareto x86 |
| x265 medium, metric-tuned (--tune psnr) | -34.2% | -35.5% | -37.5% | -42.5 … -26.3 | 8.0 | 0.30× | 38.0 | 0.34× | 137/137 | 29 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 superfast, metric-tuned (--tune psnr) | -31.9% | -32.3% | -35.5% | -39.6 … -24.3 | 12.1 | 0.45× | 57.9 | 0.51× | 137/137 | 27 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Slow** | -31.1% | -35.3% | -36.4% | -41.2 … -21.2 | 45.7 | 1.71× | 140.1 | 1.24× | 137/137 |  |
| x265 medium | -30.8% | -29.8% | -32.5% | -38.8 … -22.7 | 8.0 | 0.30× | 38.0 | 0.34× | 137/137 | 31 rate-invalid pts |
| x265 superfast | -30.7% | -29.7% | -34.8% | -38.1 … -22.1 | 12.1 | 0.45× | 57.9 | 0.51× | 137/137 | 28 rate-invalid pts |
| **H.264** | -30.2% | -33.5% | -36.1% | -40.4 … -19.8 | 71.9 | 2.69× | 235.5 | 2.09× | 137/137 |  |
| **VP9 Veryfast** | -30.0% | -30.0% | -37.9% | -43.2 … -18.0 | 96.5 | 3.62× | 334.9 | 2.97× | 137/137 |  |
| SVT-AV1 M12 | -29.5% | -30.6% | -33.2% | -35.8 … -16.8 | 70.9 | 2.67× | 408.7 | 3.62× | 137/137 | 18 rate-invalid pts |
| libaom rt 8 | -27.7% | -26.2% | -31.6% | -33.9 … -18.4 | 17.4 | 0.65× | 68.8 | 0.61× | 137/137 | 84 rate-invalid pts |
| libaom rt 10 | -27.6% | -28.3% | -30.1% | -32.7 … -18.1 | 22.3 | 0.83× | 90.5 | 0.80× | 137/137 | 87 rate-invalid pts |
| SVT-AV1 M13 | -27.5% | -28.3% | -31.6% | -33.3 … -15.1 | 79.2 | 2.96× | 445.4 | 3.95× | 137/137 | Pareto x86; 18 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -25.2% | -24.5% | -29.6% | -35.8 … -12.8 | 17.1 | 0.64× | 85.1 | 0.75× | 137/137 | 27 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -24.6% | -22.6% | -29.4% | -34.4 … -11.2 | 17.1 | 0.64× | 85.1 | 0.75× | 137/137 | 27 rate-invalid pts |
| **H.264 Fast** | -21.9% | -25.2% | -29.5% | -33.9 … -10.3 | 113.3 | 4.25× | 377.1 | 3.34× | 137/137 | Pareto |
| **H.264 Faster** | -15.0% | -19.2% | -21.7% | -31.2 … +0.7 | 135.4 | 5.08× | 462.9 | 4.10× | 137/137 | Pareto; Pareto x86 |
| **H.264 Veryfast** | -10.4% | -14.5% | -17.4% | -27.1 … +6.4 | 153.2 | 5.75× | 503.6 | 4.47× | 137/137 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -10.3% | -9.5% | -20.2% | -13.3 … -6.3 | 26.7 | 1.00× | 112.8 | 1.00× | 137/137 | 39 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | -2.4% | -4.6% | -14.8% | -11.3 … +7.5 | 85.4 | 3.18× | 363.3 | 3.22× | 137/137 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -2.2% | +3.9% | -9.6% | -9.0 … +8.1 | 47.8 | 1.78× | 204.4 | 1.81× | 137/137 | 150 rate-invalid pts |
| x264 veryfast | -1.0% | -3.7% | -0.9% | -7.4 … +7.3 | 85.4 | 3.18× | 363.3 | 3.22× | 137/137 | 1 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 137/137 | 36 rate-invalid pts |
| libvpx VP9 rt 7 | +0.7% | +5.3% | -7.0% | -7.0 … +14.1 | 30.0 | 1.12× | 122.7 | 1.09× | 137/137 | 144 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +0.8% | +5.8% | -2.0% | -7.7 … +11.1 | 30.0 | 1.12× | 122.7 | 1.09× | 137/137 | 124 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264 | +14.0% | +9.9% | +9.8% | -15.4 … +43.4 | 111.5 | 4.17× | 524.7 | 4.65× | 137/137 | Pareto x86; 241 rate-invalid pts |
| x264 ultrafast | +31.3% | +41.7% | +6.1% | +12.1 … +49.5 | 190.5 | 7.11× | 1017.0 | 9.02× | 137/137 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 1; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 12, crowd 39, lowlight 8, pan 42, static 36 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -31.1% | -32.1% |
| H.264 | -30.2% | -31.1% |
| H.264 Fast | -21.9% | -22.2% |
| H.264 Faster | -15.0% | -14.9% |
| H.264 Veryfast | -10.4% | -10.0% |
| HEVC Slow | -51.4% | -51.2% |
| HEVC | -49.5% | -49.5% |
| HEVC Fast | -41.4% | -41.1% |
| HEVC Faster | -34.9% | -34.6% |
| AV1 Slow | -49.0% | -50.3% |
| AV1 | -48.1% | -49.5% |
| AV1 Fast | -44.3% | -45.1% |
| AV1 Faster | -38.8% | -39.9% |
| VP9 Slow | -47.3% | -47.5% |
| VP9 | -47.0% | -47.1% |
| VP9 Fast | -40.4% | -40.1% |
| VP9 Faster | -36.5% | -36.2% |
| VP9 Veryfast | -30.0% | -30.2% |
| AV2 Slow | -52.6% | -53.1% |
| AV2 | -51.5% | -51.8% |
| AV2 Fast | -43.8% | -43.3% |
| VVC Slow | -53.7% | -53.8% |
| VVC | -53.1% | -53.3% |
| VVC Fast | -45.0% | -44.3% |
| openh264 | +14.0% | +13.9% |
| x264 ultrafast | +31.3% | +35.8% |
| x264 veryfast | -1.0% | -0.1% |
| x265 ultrafast | -24.6% | -23.3% |
| x265 superfast | -30.7% | -29.4% |
| x265 medium | -30.8% | -30.3% |
| SVT-AV1 M8 | -42.6% | -40.7% |
| SVT-AV1 M10 | -36.0% | -34.4% |
| SVT-AV1 M12 | -29.5% | -27.6% |
| SVT-AV1 M13 | -27.5% | -25.9% |
| libvpx VP9 rt 7 | +0.7% | +4.2% |
| libvpx VP9 rt 9 | -2.2% | -0.4% |
| libaom rt 8 | -27.7% | -25.6% |
| libaom rt 10 | -27.6% | -25.8% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +14.0% | +9.9% | +9.8% | +22.2% | 137/137 |  |
| openh264, adaptive quantisation on | +14.0% | +9.9% | +9.8% | +22.2% | 137/137 | +0.0 [-0.0, +0.0] not distinguishable |
| x264 ultrafast | +31.3% | +41.7% | +6.1% | +46.2% | 137/137 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +31.3% | +41.5% | +6.1% | +46.3% | 137/137 | -0.0 [-0.3, +0.2] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +27.7% | +38.0% | +20.7% | +24.4% | 137/137 | -0.0 [-1.4, +1.2] not distinguishable |
| x264 veryfast | -1.0% | -3.7% | -0.9% | +1.6% | 137/137 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | -2.4% | -4.6% | -14.8% | +12.1% | 137/137 | -1.7 [-2.8, -0.6] better |
| x264 veryfast, metric-tuned (--tune ssim) | -0.1% | -2.1% | +1.3% | +0.5% | 137/137 | +0.6 [+0.1, +1.2] worse |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 137/137 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -10.3% | -9.5% | -20.2% | -1.1% | 137/137 | -10.3 [-11.2, -9.4] better |
| x264 medium, metric-tuned (--tune ssim) | -2.2% | -1.8% | -0.6% | -4.1% | 137/137 | -2.2 [-2.9, -1.4] better |
| x265 ultrafast | -24.6% | -22.6% | -29.4% | -21.9% | 137/137 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -25.2% | -24.5% | -29.6% | -21.6% | 137/137 | -1.0 [-1.3, -0.6] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -25.2% | -24.5% | -29.6% | -21.6% | 137/137 | -1.0 [-1.3, -0.6] better |
| x265 superfast | -30.7% | -29.7% | -34.8% | -27.5% | 137/137 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -31.9% | -32.3% | -35.5% | -28.0% | 137/137 | -2.0 [-2.4, -1.7] better |
| x265 superfast, metric-tuned (--tune ssim) | -31.9% | -32.3% | -35.5% | -28.0% | 137/137 | -2.0 [-2.4, -1.7] better |
| x265 medium | -30.8% | -29.8% | -32.5% | -30.1% | 137/137 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -34.2% | -35.5% | -37.5% | -29.4% | 137/137 | -5.5 [-6.1, -4.8] better |
| x265 medium, metric-tuned (--tune ssim) | -32.1% | -32.2% | -33.3% | -30.7% | 137/137 | -2.0 [-2.3, -1.7] better |
| libvpx VP9 rt 7 | +0.7% | +5.3% | -7.0% | +3.7% | 137/137 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +0.8% | +5.8% | -2.0% | -1.5% | 137/137 | -0.9 [-1.3, -0.4] better |
| libvpx VP9 rt 9 | -2.2% | +3.9% | -9.6% | -1.0% | 137/137 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -2.1% | +4.5% | -8.8% | -2.1% | 137/137 | -0.2 [-0.6, +0.3] not distinguishable |
| libaom rt 8 | -27.7% | -26.2% | -31.6% | -25.3% | 137/137 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -27.6% | -25.0% | -27.2% | -30.8% | 137/137 | +0.8 [+0.2, +1.6] worse |
| libaom rt 10 | -27.6% | -28.3% | -30.1% | -24.2% | 137/137 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -27.6% | -26.8% | -26.8% | -29.3% | 137/137 | +0.7 [+0.1, +1.3] worse |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -47.2% | -47.4% | -46.3% | -47.8% | 136/137 |
| **VVC** | -46.5% | -46.5% | -45.4% | -47.4% | 136/137 |
| **AV2 Slow** | -46.1% | -49.2% | -44.3% | -44.8% | 136/137 |
| **AV2** | -44.8% | -48.1% | -43.0% | -43.4% | 136/137 |
| **HEVC Slow** | -44.7% | -44.3% | -43.1% | -46.5% | 136/137 |
| **HEVC** | -42.6% | -42.5% | -41.4% | -43.7% | 136/137 |
| **AV1 Slow** | -42.1% | -45.7% | -39.2% | -41.3% | 136/137 |
| **AV1** | -41.1% | -44.9% | -37.9% | -40.5% | 136/137 |
| **VP9 Slow** | -40.1% | -42.5% | -36.5% | -41.2% | 136/137 |
| **VP9** | -39.6% | -42.2% | -36.0% | -40.6% | 136/137 |
| **VVC Fast** | -37.2% | -35.8% | -35.2% | -40.5% | 136/137 |
| **AV1 Fast** | -36.7% | -39.8% | -32.4% | -37.7% | 136/137 |
| **AV2 Fast** | -35.9% | -37.1% | -32.8% | -37.7% | 136/137 |
| SVT-AV1 M8 | -34.4% | -36.3% | -26.2% | -40.6% | 136/137 |
| **HEVC Fast** | -33.1% | -32.7% | -29.2% | -37.3% | 136/137 |
| **VP9 Fast** | -32.1% | -33.0% | -28.7% | -34.5% | 136/137 |
| **AV1 Faster** | -30.6% | -30.9% | -28.0% | -32.7% | 136/137 |
| **VP9 Faster** | -27.6% | -26.9% | -24.8% | -31.0% | 136/137 |
| SVT-AV1 M10 | -26.7% | -27.3% | -20.0% | -32.7% | 136/137 |
| **HEVC Faster** | -25.8% | -24.5% | -21.7% | -31.1% | 136/137 |
| x265 medium, metric-tuned (--tune psnr) | -25.4% | -25.8% | -20.2% | -30.1% | 137/137 |
| x265 medium, metric-tuned (--tune ssim) | -22.6% | -21.9% | -14.7% | -31.2% | 137/137 |
| x265 superfast, metric-tuned (--tune psnr) | -22.5% | -21.7% | -17.4% | -28.6% | 137/137 |
| x265 superfast, metric-tuned (--tune ssim) | -22.5% | -21.7% | -17.4% | -28.6% | 137/137 |
| **H.264 Slow** | -21.9% | -25.5% | -18.5% | -21.5% | 137/137 |
| x265 medium | -21.1% | -19.0% | -13.6% | -30.7% | 137/137 |
| x265 superfast | -21.0% | -18.6% | -16.4% | -28.1% | 137/137 |
| **H.264** | -20.8% | -23.4% | -18.0% | -20.9% | 137/137 |
| **VP9 Veryfast** | -20.4% | -19.6% | -19.9% | -21.7% | 136/137 |
| SVT-AV1 M12 | -19.6% | -19.3% | -14.5% | -25.1% | 137/137 |
| SVT-AV1 M13 | -17.5% | -16.9% | -12.6% | -22.9% | 137/137 |
| libaom rt 8 | -17.0% | -15.0% | -11.7% | -25.1% | 136/137 |
| libaom rt 10 | -16.6% | -16.8% | -9.7% | -24.1% | 136/137 |
| libaom rt 8, metric-tuned (--tune ssim) | -16.6% | -13.6% | -6.4% | -30.6% | 136/137 |
| libaom rt 10, metric-tuned (--tune ssim) | -16.5% | -15.2% | -5.8% | -29.3% | 136/137 |
| x265 ultrafast, metric-tuned (--tune psnr) | -14.9% | -13.1% | -9.7% | -21.9% | 137/137 |
| x265 ultrafast, metric-tuned (--tune ssim) | -14.9% | -13.1% | -9.7% | -21.9% | 137/137 |
| x265 ultrafast | -14.2% | -11.0% | -9.4% | -22.1% | 137/137 |
| **H.264 Fast** | -11.6% | -14.5% | -10.0% | -10.4% | 137/137 |
| **H.264 Faster** | -4.0% | -8.3% | -0.3% | -3.3% | 137/137 |
| **H.264 Veryfast** | +1.1% | -3.4% | +5.0% | +1.7% | 137/137 |
| x264 veryfast, metric-tuned (--tune psnr) | +9.2% | +6.0% | +8.2% | +13.3% | 137/137 |
| x264 medium, metric-tuned (--tune ssim) | +10.8% | +8.7% | +26.6% | -2.9% | 137/137 |
| libvpx VP9 rt 9 | +11.6% | +18.7% | +16.5% | -0.5% | 137/137 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.6% | +19.2% | +17.4% | -1.7% | 137/137 |
| x264 veryfast | +12.4% | +7.1% | +27.4% | +2.7% | 137/137 |
| x264 medium | +13.3% | +11.3% | +27.2% | +1.5% | 137/137 |
| x264 veryfast, metric-tuned (--tune ssim) | +13.5% | +8.9% | +30.2% | +1.4% | 137/137 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.0% | +19.1% | +23.5% | -0.4% | 137/137 |
| libvpx VP9 rt 7 | +14.1% | +19.1% | +18.6% | +4.7% | 137/137 |
| openh264 | +30.5% | +26.9% | +41.2% | +23.5% | 137/137 |
| openh264, adaptive quantisation on | +30.5% | +26.9% | +41.2% | +23.5% | 137/137 |
| x264 ultrafast, metric-tuned (--tune ssim) | +45.4% | +52.2% | +57.4% | +26.5% | 137/137 |
| x264 ultrafast, metric-tuned (--tune psnr) | +47.0% | +55.4% | +36.5% | +49.2% | 137/137 |
| x264 ultrafast | +47.1% | +55.6% | +36.5% | +49.1% | 137/137 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | ind_bwc_videotext_video88 |
|---|---:|
| VVC Slow | -65.6% |
| VVC | -65.3% |
| AV2 Slow | -65.1% |
| AV2 | -64.9% |
| HEVC Slow | -55.7% |
| HEVC | -56.4% |
| AV1 Slow | -56.1% |
| AV1 | -55.3% |
| VP9 Slow | -56.0% |
| VP9 | -55.3% |
| VVC Fast | -63.8% |
| AV1 Fast | -57.9% |
| AV2 Fast | -65.8% |
| SVT-AV1 M8 | -60.8% |
| HEVC Fast | -54.2% |
| VP9 Fast | -58.2% |
| AV1 Faster | -58.0% |
| VP9 Faster | -58.3% |
| SVT-AV1 M10 | -56.2% |
| HEVC Faster | -50.8% |
| x265 medium, metric-tuned (--tune psnr) | n/a |
| x265 superfast, metric-tuned (--tune psnr) | n/a |
| H.264 Slow | -24.1% |
| x265 medium | -35.4% |
| x265 superfast | -47.1% |
| H.264 | -23.5% |
| VP9 Veryfast | -52.4% |
| SVT-AV1 M12 | -50.6% |
| libaom rt 8 | n/a |
| libaom rt 10 | -38.6% |
| SVT-AV1 M13 | -49.0% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a |
| x265 ultrafast | -45.2% |
| H.264 Fast | -21.7% |
| H.264 Faster | -16.9% |
| H.264 Veryfast | -11.9% |
| x264 medium, metric-tuned (--tune psnr) | n/a |
| x264 veryfast, metric-tuned (--tune psnr) | n/a |
| libvpx VP9 rt 9 | -19.2% |
| x264 veryfast | -3.5% |
| libvpx VP9 rt 7 | -13.6% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a |
| openh264 | +32.2% |
| x264 ultrafast | +10.5% |

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
| HEVC Fast | x265 superfast | -15.5% | -18.1% | -16.0% | -27.3 … -9.0 | 138/138 |
| HEVC | SVT-AV1 M12 | -29.8% | -30.8% | -33.0% | -39.1 … -27.6 | 138/138 |
| HEVC Fast | HEVC | +20.3% | +21.3% | +25.5% | +15.7 … +25.7 | 138/138 |
| AV1 | openh264 | -52.0% | -54.5% | -53.0% | -63.4 … -42.6 | 133/138 |
| AV1 Fast | openh264 | -47.4% | -49.5% | -47.6% | -60.9 … -37.6 | 136/138 |
| AV1 | SVT-AV1 M8 | -7.3% | -12.2% | -13.9% | -22.3 … -6.1 | 138/138 |
| AV1 | SVT-AV1 M12 | -27.8% | -33.8% | -28.3% | -39.6 … -27.4 | 138/138 |
| AV1 Fast | SVT-AV1 M12 | -21.9% | -27.0% | -21.7% | -32.8 … -20.9 | 138/138 |
| AV1 | libaom rt 8 | -25.6% | -32.6% | -26.8% | -38.3 … -21.7 | 138/138 |
| AV1 Fast | AV1 | +9.5% | +11.3% | +10.8% | +6.1 … +14.8 | 138/138 |
| AV1 Faster | SVT-AV1 M12 | -12.7% | -13.7% | -16.2% | -27.7 … -10.8 | 138/138 |
| AV1 Faster | AV1 Fast | +11.8% | +18.7% | +6.9% | +6.9 … +16.4 | 138/138 |
| HEVC Faster | SVT-AV1 M13 | -8.4% | -6.8% | -10.2% | -18.2 … -3.6 | 138/138 |
| HEVC Faster | HEVC Fast | +13.6% | +15.8% | +12.9% | +8.1 … +18.8 | 138/138 |
| VP9 | libvpx VP9 rt 7 | -43.8% | -48.7% | -42.2% | -52.6 … -38.4 | 138/138 |
| VP9 Fast | libvpx VP9 rt 9 | -37.7% | -43.2% | -37.3% | -46.0 … -32.7 | 138/138 |
| VP9 Fast | VP9 | +15.6% | +20.1% | +13.8% | +10.7 … +21.7 | 138/138 |
| VP9 Faster | libvpx VP9 rt 9 | -32.9% | -36.9% | -33.7% | -42.7 … -26.5 | 138/138 |
| VP9 Faster | VP9 Fast | +8.4% | +12.1% | +6.1% | +3.5 … +12.0 | 138/138 |
| VP9 Veryfast | libvpx VP9 rt 9 | -23.6% | -28.6% | -27.6% | -35.8 … -17.5 | 138/138 |
| VP9 Veryfast | VP9 Faster | +15.5% | +15.0% | +9.0% | +5.9 … +19.1 | 138/138 |
| AV2 | SVT-AV1 M8 | -13.7% | -17.7% | -21.9% | -25.1 … -12.1 | 138/138 |
| AV2 | AV1 | -6.6% | -6.2% | -8.9% | -8.5 … -1.1 | 138/138 |
| AV2 Fast | AV2 | +20.0% | +26.2% | +21.1% | +15.6 … +29.0 | 138/138 |
| VVC | x265 medium | -33.4% | -36.6% | -38.9% | -43.2 … -26.2 | 138/138 |
| VVC | HEVC | -8.6% | -8.8% | -8.2% | -11.2 … -6.6 | 138/138 |
| VVC Fast | VVC | +21.3% | +25.0% | +22.7% | +17.5 … +28.7 | 138/138 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264, HEVC Fast, AV1, AV1 Fast, AV1 Faster, AV2, VVC: 4a5b8dd1; H.264 Veryfast, HEVC, VP9, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow: bad14976); AV2 Fast, H.264 Fast, H.264 Faster, HEVC Faster, VP9 Fast, VP9 Faster, VP9 Veryfast, VVC Fast, AV1 Slow, AV2 Slow, H.264 Slow, H.264 Veryfast, HEVC, HEVC Slow, VP9, VP9 Slow, VVC Slow timed on 4a5b8dd1 (same streams), its presets), openh264
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
- **x86 speed (`bd-vs-speed-independent_fresh_2026-11-13-x86.svg`, `.csv`):** the same streams (same quantizers and
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.7 [-2.8, -0.6]%); x264 medium --tune psnr (-10.3 [-11.2, -9.4]%); x265 ultrafast --tune psnr (-1.0 [-1.3, -0.6]%); x265 superfast --tune psnr (-2.0 [-2.4, -1.7]%); x265 medium --tune psnr (-5.5 [-6.1, -4.8]%); libvpx VP9 rt 7 --tune ssim (-0.9 [-1.3, -0.4]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
