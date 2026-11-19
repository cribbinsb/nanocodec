# Bitrate vs encode speed, held-out clips (independent_fresh_2026-11-13)

<sub>Updated 2026-11-19, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (AV1, AV1 Fast: 4a5b8dd1; VP9 Fast: b33ba3bb; H.264 Veryfast, VP9, AV1 Slow, VP9 Slow, AV2 Slow: bad14976; H.264, HEVC, HEVC Fast, HEVC Faster, AV1 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, VVC Slow, AV2 Faster: c9cff89f; AV1 Veryfast: d40f86b3); H.264 Fast, H.264 Faster, VP9 Faster, AV1 Slow, AV2 Slow, H.264 Veryfast, VP9, VP9 Slow, AV1, AV1 Fast timed on b33ba3bb (same streams)</sub>

Quality on the 138 clips of `independent_fresh_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_fresh_2026-11-13.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_fresh_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -53.7% | -55.4% | -58.6% | -61.7 … -46.4 | 24.8 | 0.93× | 97.7 | 0.87× | 136/136 | Pareto; Pareto x86 |
| **VVC** | -53.1% | -54.7% | -57.9% | -61.5 … -45.5 | 28.9 | 1.08× | 118.1 | 1.05× | 136/136 | Pareto; Pareto x86 |
| **AV2 Slow** | -52.5% | -56.9% | -56.6% | -59.9 … -44.9 | 22.0 | 0.82× | 81.6 | 0.72× | 136/136 |  |
| **HEVC Slow** | -51.4% | -52.9% | -55.6% | -58.7 … -43.2 | 31.0 | 1.16× | 116.4 | 1.03× | 136/136 | Pareto |
| **AV2** | -51.3% | -55.9% | -55.6% | -59.4 … -42.9 | 37.9 | 1.42× | 164.2 | 1.46× | 136/136 | Pareto; Pareto x86 |
| **HEVC** | -49.6% | -51.6% | -54.4% | -58.4 … -41.2 | 50.4 | 1.89× | 201.2 | 1.78× | 136/136 | Pareto; Pareto x86 |
| **AV1 Slow** | -48.9% | -53.4% | -52.6% | -57.3 … -42.0 | 29.7 | 1.11× | 116.6 | 1.03× | 136/136 | 1 rate-invalid pts |
| **AV1** | -48.1% | -52.7% | -51.6% | -56.5 … -40.7 | 45.0 | 1.69× | 171.4 | 1.52× | 136/136 | 1 rate-invalid pts |
| **VP9 Slow** | -47.3% | -50.6% | -50.7% | -55.3 … -38.6 | 36.7 | 1.38× | 151.8 | 1.35× | 136/136 | 3 rate-invalid pts |
| **VP9** | -46.9% | -50.5% | -50.3% | -55.1 … -37.8 | 43.2 | 1.62× | 172.4 | 1.53× | 136/136 | 3 rate-invalid pts |
| **AV2 Fast** | -46.6% | -49.7% | -49.9% | -54.4 … -37.7 | 50.0 | 1.87× | 223.4 | 1.98× | 136/136 | Pareto x86 |
| **AV2 Faster** | -46.1% | -49.0% | -49.4% | -54.9 … -36.5 | 59.5 | 2.23× | 254.8 | 2.26× | 136/136 | Pareto; Pareto x86 |
| **VVC Fast** | -44.9% | -45.3% | -49.9% | -53.4 … -35.7 | 55.1 | 2.07× | 221.2 | 1.96× | 136/136 |  |
| **HEVC Fast** | -44.3% | -45.9% | -48.4% | -52.2 … -35.7 | 78.2 | 2.93× | 304.6 | 2.70× | 136/136 | Pareto; Pareto x86 |
| **AV1 Fast** | -44.2% | -48.2% | -47.3% | -52.9 … -36.4 | 65.5 | 2.46× | 280.4 | 2.49× | 136/136 |  |
| **VP9 Fast** | -42.6% | -46.1% | -44.8% | -51.1 … -33.4 | 54.4 | 2.04× | 227.3 | 2.02× | 136/136 |  |
| SVT-AV1 M8 | -42.5% | -45.7% | -42.1% | -46.7 … -32.1 | 14.3 | 0.53× | 80.8 | 0.72× | 136/136 | 23 rate-invalid pts |
| **AV1 Faster** | -41.8% | -44.3% | -46.8% | -52.4 … -34.6 | 78.4 | 2.94× | 312.7 | 2.77× | 136/136 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -37.2% | -38.2% | -41.3% | -48.7 … -26.1 | 95.9 | 3.59× | 425.3 | 3.77× | 136/136 | Pareto; Pareto x86 |
| **VP9 Faster** | -36.4% | -37.0% | -41.5% | -47.0 … -27.2 | 92.0 | 3.45× | 354.1 | 3.14× | 136/136 | 1 rate-invalid pts |
| SVT-AV1 M10 | -35.9% | -38.0% | -37.5% | -42.0 … -25.1 | 39.3 | 1.47× | 228.9 | 2.03× | 136/136 | 21 rate-invalid pts |
| **HEVC Faster** | -35.8% | -35.5% | -40.0% | -44.4 … -25.9 | 101.4 | 3.80× | 417.8 | 3.71× | 136/136 | Pareto |
| x265 medium, metric-tuned (--tune psnr) | -34.1% | -35.5% | -37.4% | -42.5 … -26.3 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 29 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **VP9 Veryfast** | -32.7% | -34.6% | -38.9% | -45.0 … -22.4 | 94.3 | 3.53× | 390.7 | 3.47× | 136/136 |  |
| **H.264 Slow** | -32.6% | -36.6% | -38.5% | -42.8 … -23.5 | 42.0 | 1.57× | 123.7 | 1.10× | 136/136 |  |
| **H.264** | -32.4% | -36.0% | -38.6% | -42.8 … -22.4 | 65.9 | 2.47× | 205.2 | 1.82× | 136/136 |  |
| x265 superfast, metric-tuned (--tune psnr) | -31.8% | -32.2% | -35.4% | -39.3 … -24.1 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 27 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 medium | -30.8% | -29.8% | -32.4% | -38.5 … -22.7 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 31 rate-invalid pts |
| x265 superfast | -30.6% | -29.6% | -34.7% | -37.5 … -22.0 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 28 rate-invalid pts |
| SVT-AV1 M12 | -29.4% | -30.4% | -33.0% | -35.6 … -16.6 | 70.9 | 2.67× | 408.7 | 3.62× | 136/136 | 18 rate-invalid pts |
| libaom rt 8 | -27.7% | -26.2% | -31.5% | -33.9 … -18.4 | 17.4 | 0.65× | 68.8 | 0.61× | 136/136 | 84 rate-invalid pts |
| libaom rt 10 | -27.5% | -28.2% | -30.1% | -32.7 … -18.1 | 22.3 | 0.83× | 90.5 | 0.80× | 136/136 | 87 rate-invalid pts |
| SVT-AV1 M13 | -27.4% | -28.1% | -31.5% | -33.0 … -14.9 | 79.2 | 2.96× | 445.4 | 3.95× | 136/136 | Pareto x86; 18 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -25.1% | -24.4% | -29.5% | -35.7 … -12.5 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 27 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -24.5% | -22.5% | -29.3% | -34.3 … -11.1 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 27 rate-invalid pts |
| **H.264 Fast** | -21.8% | -25.1% | -29.4% | -33.3 … -10.2 | 121.2 | 4.54× | 379.5 | 3.37× | 136/136 | Pareto |
| **H.264 Faster** | -14.9% | -19.0% | -21.6% | -30.7 … +0.8 | 145.7 | 5.46× | 486.1 | 4.31× | 136/136 | Pareto; Pareto x86 |
| **H.264 Veryfast** | -10.3% | -14.3% | -17.3% | -26.6 … +6.4 | 166.2 | 6.23× | 526.2 | 4.67× | 136/136 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -10.2% | -9.5% | -20.1% | -13.0 … -6.3 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 39 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | -2.3% | -4.4% | -14.7% | -11.2 … +7.6 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -2.1% | +4.0% | -9.5% | -9.0 … +8.1 | 47.8 | 1.78× | 204.4 | 1.81× | 136/136 | 150 rate-invalid pts |
| x264 veryfast | -0.9% | -3.7% | -0.8% | -7.4 … +7.3 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 36 rate-invalid pts |
| libvpx VP9 rt 7 | +0.7% | +5.3% | -7.0% | -7.1 … +13.9 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 144 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +0.7% | +5.8% | -2.1% | -7.8 … +10.9 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 124 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264 | +14.2% | +10.2% | +10.1% | -15.1 … +43.5 | 111.5 | 4.17× | 524.7 | 4.65× | 136/136 | 241 rate-invalid pts |
| x264 ultrafast | +31.5% | +41.9% | +6.2% | +12.5 … +49.5 | 190.5 | 7.11× | 1017.0 | 9.02× | 136/136 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00216, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 2; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 12, crowd 39, lowlight 8, pan 42, static 35 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -32.6% | -33.4% |
| H.264 | -32.4% | -33.2% |
| H.264 Fast | -21.8% | -21.9% |
| H.264 Faster | -14.9% | -14.6% |
| H.264 Veryfast | -10.3% | -9.7% |
| HEVC Slow | -51.4% | -51.1% |
| HEVC | -49.6% | -49.4% |
| HEVC Fast | -44.3% | -44.0% |
| HEVC Faster | -35.8% | -35.5% |
| AV1 Slow | -48.9% | -50.2% |
| AV1 | -48.1% | -49.4% |
| AV1 Fast | -44.2% | -45.0% |
| AV1 Faster | -41.8% | -42.8% |
| AV1 Veryfast | -37.2% | -37.7% |
| VP9 Slow | -47.3% | -47.4% |
| VP9 | -46.9% | -47.0% |
| VP9 Fast | -42.6% | -42.4% |
| VP9 Faster | -36.4% | -36.0% |
| VP9 Veryfast | -32.7% | -33.2% |
| AV2 Slow | -52.5% | -53.0% |
| AV2 | -51.3% | -51.6% |
| AV2 Fast | -46.6% | -46.2% |
| AV2 Faster | -46.1% | -45.8% |
| VVC Slow | -53.7% | -53.8% |
| VVC | -53.1% | -53.2% |
| VVC Fast | -44.9% | -44.2% |
| openh264 | +14.2% | +14.4% |
| x264 ultrafast | +31.5% | +36.1% |
| x264 veryfast | -0.9% | +0.1% |
| x265 ultrafast | -24.5% | -23.1% |
| x265 superfast | -30.6% | -29.3% |
| x265 medium | -30.8% | -30.2% |
| SVT-AV1 M8 | -42.5% | -40.5% |
| SVT-AV1 M10 | -35.9% | -34.3% |
| SVT-AV1 M12 | -29.4% | -27.4% |
| SVT-AV1 M13 | -27.4% | -25.7% |
| libvpx VP9 rt 7 | +0.7% | +4.0% |
| libvpx VP9 rt 9 | -2.1% | -0.3% |
| libaom rt 8 | -27.7% | -25.6% |
| libaom rt 10 | -27.5% | -25.7% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +14.2% | +10.2% | +10.1% | +22.4% | 136/136 |  |
| openh264, adaptive quantisation on | +14.2% | +10.2% | +10.1% | +22.4% | 136/136 | +0.0 [-0.0, +0.1] not distinguishable |
| x264 ultrafast | +31.5% | +41.9% | +6.2% | +46.4% | 136/136 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +31.5% | +41.7% | +6.3% | +46.5% | 136/136 | -0.0 [-0.3, +0.2] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +27.8% | +38.1% | +20.8% | +24.4% | 136/136 | -0.1 [-1.4, +1.2] not distinguishable |
| x264 veryfast | -0.9% | -3.7% | -0.8% | +1.7% | 136/136 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | -2.3% | -4.4% | -14.7% | +12.2% | 136/136 | -1.6 [-2.8, -0.6] better |
| x264 veryfast, metric-tuned (--tune ssim) | -0.0% | -2.0% | +1.5% | +0.6% | 136/136 | +0.7 [+0.1, +1.3] worse |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 136/136 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -10.2% | -9.5% | -20.1% | -1.0% | 136/136 | -10.2 [-11.2, -9.3] better |
| x264 medium, metric-tuned (--tune ssim) | -2.1% | -1.8% | -0.5% | -4.1% | 136/136 | -2.1 [-2.9, -1.4] better |
| x265 ultrafast | -24.5% | -22.5% | -29.3% | -21.8% | 136/136 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -25.1% | -24.4% | -29.5% | -21.5% | 136/136 | -1.0 [-1.3, -0.6] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -25.1% | -24.4% | -29.5% | -21.5% | 136/136 | -1.0 [-1.3, -0.6] better |
| x265 superfast | -30.6% | -29.6% | -34.7% | -27.5% | 136/136 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -31.8% | -32.2% | -35.4% | -27.9% | 136/136 | -2.0 [-2.4, -1.7] better |
| x265 superfast, metric-tuned (--tune ssim) | -31.8% | -32.2% | -35.4% | -27.9% | 136/136 | -2.0 [-2.4, -1.7] better |
| x265 medium | -30.8% | -29.8% | -32.4% | -30.1% | 136/136 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -34.1% | -35.5% | -37.4% | -29.4% | 136/136 | -5.5 [-6.1, -4.7] better |
| x265 medium, metric-tuned (--tune ssim) | -32.0% | -32.2% | -33.2% | -30.6% | 136/136 | -2.0 [-2.2, -1.7] better |
| libvpx VP9 rt 7 | +0.7% | +5.3% | -7.0% | +3.7% | 136/136 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +0.7% | +5.8% | -2.1% | -1.5% | 136/136 | -0.9 [-1.4, -0.4] better |
| libvpx VP9 rt 9 | -2.1% | +4.0% | -9.5% | -0.9% | 136/136 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -2.0% | +4.6% | -8.7% | -2.0% | 136/136 | -0.2 [-0.7, +0.2] not distinguishable |
| libaom rt 8 | -27.7% | -26.2% | -31.5% | -25.3% | 136/136 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -27.6% | -24.9% | -27.1% | -30.7% | 136/136 | +0.9 [+0.2, +1.6] worse |
| libaom rt 10 | -27.5% | -28.2% | -30.1% | -24.2% | 136/136 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -27.5% | -26.7% | -26.7% | -29.3% | 136/136 | +0.7 [+0.1, +1.3] worse |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -47.3% | -47.4% | -46.6% | -47.8% | 135/136 |
| **VVC** | -46.6% | -46.5% | -45.7% | -47.4% | 135/136 |
| **AV2 Slow** | -46.1% | -49.2% | -44.3% | -44.8% | 135/136 |
| **HEVC Slow** | -44.7% | -44.9% | -42.7% | -46.4% | 135/136 |
| **AV2** | -44.7% | -48.0% | -42.9% | -43.1% | 135/136 |
| **HEVC** | -42.7% | -43.2% | -41.1% | -43.6% | 135/136 |
| **AV1 Slow** | -42.1% | -45.8% | -39.1% | -41.3% | 135/136 |
| **AV1** | -41.1% | -44.9% | -37.9% | -40.5% | 135/136 |
| **VP9 Slow** | -40.1% | -42.5% | -36.5% | -41.2% | 135/136 |
| **VP9** | -39.6% | -42.2% | -36.0% | -40.6% | 135/136 |
| **AV2 Fast** | -39.3% | -40.9% | -35.6% | -41.2% | 135/136 |
| **AV2 Faster** | -38.6% | -40.1% | -34.9% | -40.6% | 135/136 |
| **VVC Fast** | -37.2% | -35.8% | -35.2% | -40.4% | 135/136 |
| **AV1 Fast** | -36.6% | -39.8% | -32.3% | -37.7% | 135/136 |
| **HEVC Fast** | -36.5% | -36.6% | -33.3% | -39.3% | 135/136 |
| **VP9 Fast** | -34.6% | -37.1% | -28.9% | -37.6% | 135/136 |
| SVT-AV1 M8 | -34.4% | -36.3% | -26.1% | -40.6% | 135/136 |
| **AV1 Faster** | -33.9% | -35.3% | -31.7% | -34.7% | 135/136 |
| **AV1 Veryfast** | -28.7% | -28.7% | -24.7% | -32.7% | 135/136 |
| **VP9 Faster** | -27.6% | -26.9% | -24.8% | -31.0% | 135/136 |
| **HEVC Faster** | -26.8% | -25.6% | -22.7% | -32.0% | 135/136 |
| SVT-AV1 M10 | -26.7% | -27.3% | -19.9% | -32.7% | 135/136 |
| x265 medium, metric-tuned (--tune psnr) | -25.3% | -25.8% | -20.1% | -30.0% | 136/136 |
| **H.264 Slow** | -23.6% | -27.2% | -21.1% | -22.5% | 136/136 |
| **VP9 Veryfast** | -23.4% | -24.4% | -21.2% | -24.5% | 135/136 |
| **H.264** | -23.3% | -26.3% | -21.2% | -22.5% | 136/136 |
| x265 medium, metric-tuned (--tune ssim) | -22.6% | -21.9% | -14.6% | -31.2% | 136/136 |
| x265 superfast, metric-tuned (--tune psnr) | -22.5% | -21.6% | -17.4% | -28.6% | 136/136 |
| x265 superfast, metric-tuned (--tune ssim) | -22.5% | -21.6% | -17.4% | -28.6% | 136/136 |
| x265 medium | -21.1% | -19.0% | -13.5% | -30.7% | 136/136 |
| x265 superfast | -21.0% | -18.6% | -16.3% | -28.1% | 136/136 |
| SVT-AV1 M12 | -19.5% | -19.2% | -14.3% | -25.0% | 136/136 |
| SVT-AV1 M13 | -17.3% | -16.8% | -12.4% | -22.8% | 136/136 |
| libaom rt 8 | -17.1% | -15.1% | -11.7% | -25.1% | 135/136 |
| libaom rt 10 | -16.7% | -16.8% | -9.7% | -24.1% | 135/136 |
| libaom rt 8, metric-tuned (--tune ssim) | -16.6% | -13.6% | -6.4% | -30.6% | 135/136 |
| libaom rt 10, metric-tuned (--tune ssim) | -16.5% | -15.2% | -5.7% | -29.3% | 135/136 |
| x265 ultrafast, metric-tuned (--tune psnr) | -14.8% | -13.1% | -9.6% | -21.8% | 136/136 |
| x265 ultrafast, metric-tuned (--tune ssim) | -14.8% | -13.1% | -9.6% | -21.8% | 136/136 |
| x265 ultrafast | -14.1% | -11.0% | -9.3% | -22.1% | 136/136 |
| **H.264 Fast** | -11.6% | -14.5% | -9.9% | -10.3% | 136/136 |
| **H.264 Faster** | -3.9% | -8.2% | -0.2% | -3.2% | 136/136 |
| **H.264 Veryfast** | +1.2% | -3.3% | +5.2% | +1.8% | 136/136 |
| x264 veryfast, metric-tuned (--tune psnr) | +9.3% | +6.1% | +8.3% | +13.4% | 136/136 |
| x264 medium, metric-tuned (--tune ssim) | +10.8% | +8.7% | +26.6% | -2.9% | 136/136 |
| libvpx VP9 rt 9 | +11.6% | +18.7% | +16.6% | -0.5% | 136/136 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.7% | +19.2% | +17.5% | -1.7% | 136/136 |
| x264 veryfast | +12.4% | +7.1% | +27.5% | +2.7% | 136/136 |
| x264 medium | +13.3% | +11.2% | +27.1% | +1.4% | 136/136 |
| x264 veryfast, metric-tuned (--tune ssim) | +13.6% | +8.9% | +30.3% | +1.5% | 136/136 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +13.9% | +18.9% | +23.2% | -0.6% | 136/136 |
| libvpx VP9 rt 7 | +14.0% | +18.9% | +18.4% | +4.6% | 136/136 |
| openh264 | +30.7% | +27.0% | +41.5% | +23.6% | 136/136 |
| openh264, adaptive quantisation on | +30.7% | +27.0% | +41.5% | +23.7% | 136/136 |
| x264 ultrafast, metric-tuned (--tune ssim) | +45.4% | +52.3% | +57.5% | +26.5% | 136/136 |
| x264 ultrafast, metric-tuned (--tune psnr) | +47.1% | +55.5% | +36.7% | +49.3% | 136/136 |
| x264 ultrafast | +47.2% | +55.7% | +36.7% | +49.3% | 136/136 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.
- **ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508** (static): no reliable overlap for VVC Slow.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | ind_bwc_videotext_video88 | ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508 |
|---|---:|---:|
| VVC Slow | -65.8% | n/a |
| VVC | -65.4% | -68.3% |
| AV2 Slow | -65.1% | -68.5% |
| HEVC Slow | -56.1% | -67.6% |
| AV2 | -64.6% | -67.5% |
| HEVC | -56.8% | -66.6% |
| AV1 Slow | -56.1% | -66.4% |
| AV1 | -55.3% | -67.8% |
| VP9 Slow | -56.0% | -63.1% |
| VP9 | -55.3% | -63.3% |
| AV2 Fast | -65.8% | -66.5% |
| AV2 Faster | -64.9% | -64.2% |
| VVC Fast | -63.6% | -59.5% |
| HEVC Fast | -56.1% | -59.4% |
| AV1 Fast | -57.9% | -61.9% |
| VP9 Fast | -56.2% | -59.6% |
| SVT-AV1 M8 | -60.8% | -55.6% |
| AV1 Faster | -56.9% | -62.9% |
| AV1 Veryfast | -55.7% | -62.1% |
| VP9 Faster | -58.3% | -56.1% |
| SVT-AV1 M10 | -56.2% | -52.5% |
| HEVC Faster | -50.2% | -57.8% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a |
| VP9 Veryfast | -54.2% | -56.2% |
| H.264 Slow | -20.6% | -59.7% |
| H.264 | -22.1% | -60.6% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 medium | -35.4% | -40.3% |
| x265 superfast | -47.1% | -42.3% |
| SVT-AV1 M12 | -50.6% | -53.7% |
| libaom rt 8 | n/a | -22.4% |
| libaom rt 10 | -38.6% | -32.2% |
| SVT-AV1 M13 | -49.0% | -52.3% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 ultrafast | -45.2% | -41.7% |
| H.264 Fast | -21.7% | -54.8% |
| H.264 Faster | -16.9% | -52.7% |
| H.264 Veryfast | -11.9% | -53.5% |
| x264 medium, metric-tuned (--tune psnr) | n/a | n/a |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a |
| libvpx VP9 rt 9 | -19.2% | -13.7% |
| x264 veryfast | -3.5% | -22.7% |
| libvpx VP9 rt 7 | -13.6% | +24.9% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a |
| openh264 | +32.2% | -66.5% |
| x264 ultrafast | +10.5% | -10.6% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slow | openh264 | -36.4% | -38.1% | -39.4% | -47.8 … -26.4 | 138/138 |
| H.264 | openh264 | -36.1% | -37.2% | -39.4% | -47.5 … -26.3 | 138/138 |
| H.264 | x264 veryfast | -37.0% | -38.8% | -43.5% | -45.1 … -30.9 | 138/138 |
| H.264 | x264 medium | -32.4% | -36.0% | -38.6% | -43.1 … -22.2 | 138/138 |
| H.264 Fast | openh264 | -23.7% | -25.0% | -29.1% | -36.9 … -13.2 | 138/138 |
| H.264 Fast | x264 veryfast | -26.5% | -28.3% | -34.4% | -34.6 … -19.5 | 138/138 |
| H.264 Fast | H.264 | +18.8% | +19.1% | +17.1% | +14.4 … +24.1 | 138/138 |
| H.264 Faster | openh264 | -15.6% | -18.2% | -20.0% | -28.6 … -5.6 | 138/138 |
| H.264 Faster | x264 ultrafast | -40.9% | -50.1% | -32.0% | -49.6 … -36.6 | 130/138 |
| H.264 Faster | H.264 Fast | +10.5% | +8.6% | +12.7% | +5.2 … +15.4 | 138/138 |
| H.264 Veryfast | openh264 | -10.1% | -12.8% | -14.9% | -24.9 … +1.6 | 137/138 |
| H.264 Veryfast | x264 ultrafast | -36.8% | -46.6% | -27.2% | -46.3 … -32.0 | 131/138 |
| H.264 Veryfast | H.264 Faster | +7.3% | +7.5% | +6.8% | +5.5 … +9.7 | 138/138 |
| HEVC Slow | SVT-AV1 M12 | -32.7% | -33.7% | -34.7% | -41.1 … -30.2 | 138/138 |
| HEVC | SVT-AV1 M12 | -30.0% | -31.7% | -32.8% | -38.9 … -28.2 | 138/138 |
| HEVC | x265 medium | -28.1% | -32.0% | -33.5% | -38.9 … -20.2 | 138/138 |
| HEVC Fast | SVT-AV1 M12 | -21.4% | -22.4% | -22.7% | -29.7 … -20.3 | 138/138 |
| HEVC Fast | x265 superfast | -20.6% | -24.1% | -21.4% | -31.6 … -15.0 | 138/138 |
| HEVC Fast | HEVC | +13.1% | +14.0% | +16.1% | +9.0 … +17.1 | 138/138 |
| HEVC Faster | SVT-AV1 M12 | -6.8% | -4.8% | -9.0% | -16.8 … -3.2 | 138/138 |
| HEVC Faster | SVT-AV1 M13 | -9.9% | -8.4% | -11.6% | -18.9 … -6.3 | 138/138 |
| HEVC Faster | HEVC Fast | +19.7% | +23.4% | +19.9% | +13.0 … +27.7 | 138/138 |
| AV1 Slow | SVT-AV1 M12 | -29.0% | -34.8% | -29.7% | -40.8 … -28.9 | 138/138 |
| AV1 | SVT-AV1 M12 | -27.8% | -33.8% | -28.3% | -39.6 … -27.4 | 138/138 |
| AV1 | openh264 | -52.0% | -54.5% | -53.0% | -63.4 … -42.6 | 133/138 |
| AV1 | SVT-AV1 M8 | -7.3% | -12.2% | -13.9% | -22.3 … -6.1 | 138/138 |
| AV1 | libaom rt 8 | -25.6% | -32.6% | -26.8% | -38.3 … -21.7 | 138/138 |
| AV1 Fast | SVT-AV1 M12 | -21.9% | -27.0% | -21.7% | -32.8 … -20.9 | 138/138 |
| AV1 Fast | openh264 | -47.4% | -49.5% | -47.6% | -60.9 … -37.6 | 136/138 |
| AV1 Fast | AV1 | +9.5% | +11.3% | +10.8% | +6.1 … +14.8 | 138/138 |
| AV1 Faster | SVT-AV1 M12 | -17.5% | -20.6% | -20.8% | -30.9 … -15.9 | 138/138 |
| AV1 Faster | AV1 Fast | +5.1% | +8.6% | +0.7% | +0.1 … +8.9 | 138/138 |
| AV1 Veryfast | SVT-AV1 M12 | -10.9% | -10.9% | -12.8% | -27.2 … -7.5 | 138/138 |
| AV1 Veryfast | SVT-AV1 M13 | -14.0% | -14.3% | -15.4% | -28.7 … -9.7 | 138/138 |
| AV1 Veryfast | AV1 Faster | +9.7% | +14.1% | +11.6% | +4.5 … +16.9 | 138/138 |
| VP9 Slow | HEVC Slow | +9.1% | +4.4% | +12.2% | +5.2 … +11.8 | 138/138 |
| VP9 | HEVC | +5.9% | +1.7% | +10.1% | +2.4 … +9.3 | 138/138 |
| VP9 | libvpx VP9 rt 7 | -43.8% | -48.7% | -42.2% | -52.6 … -38.4 | 138/138 |
| VP9 Fast | HEVC Fast | +3.8% | -0.5% | +8.1% | -0.7 … +8.6 | 138/138 |
| VP9 Fast | libvpx VP9 rt 9 | -40.3% | -47.1% | -37.6% | -48.2 … -35.8 | 137/138 |
| VP9 Fast | VP9 | +10.0% | +10.8% | +13.0% | +6.8 … +14.6 | 138/138 |
| VP9 Faster | HEVC Faster | -1.9% | -3.2% | -3.5% | -4.9 … +2.6 | 138/138 |
| VP9 Faster | libvpx VP9 rt 9 | -32.9% | -36.9% | -33.7% | -42.7 … -26.5 | 138/138 |
| VP9 Faster | VP9 Fast | +13.6% | +20.6% | +6.7% | +8.0 … +18.4 | 138/138 |
| VP9 Veryfast | HEVC Faster | +6.6% | +2.0% | +2.7% | -0.4 … +10.2 | 138/138 |
| VP9 Veryfast | libvpx VP9 rt 9 | -27.2% | -33.9% | -29.2% | -39.4 … -21.1 | 138/138 |
| VP9 Veryfast | VP9 Faster | +9.2% | +5.7% | +6.8% | +1.1 … +11.6 | 138/138 |
| AV2 Slow | AV1 Slow | -7.5% | -6.9% | -9.3% | -10.0 … -2.9 | 138/138 |
| AV2 | AV1 | -6.4% | -6.0% | -8.8% | -8.2 … -1.0 | 138/138 |
| AV2 | SVT-AV1 M8 | -13.5% | -17.6% | -21.9% | -25.4 … -12.0 | 138/138 |
| AV2 Fast | AV1 Fast | -5.2% | -2.7% | -6.2% | -8.2 … +2.0 | 138/138 |
| AV2 Fast | AV2 | +11.3% | +15.9% | +14.4% | +9.3 … +16.7 | 138/138 |
| AV2 Faster | AV1 Faster | -8.8% | -9.3% | -5.7% | -10.1 … -2.6 | 138/138 |
| AV2 Faster | SVT-AV1 M12 | -25.8% | -28.2% | -26.3% | -34.3 … -23.1 | 138/138 |
| AV2 Faster | AV2 Fast | +1.3% | +1.6% | +1.3% | +0.0 … +2.2 | 138/138 |
| VVC Slow | HEVC Slow | -5.9% | -5.9% | -7.6% | -8.5 … -3.7 | 138/138 |
| VVC | HEVC | -8.4% | -7.4% | -9.0% | -11.4 … -6.4 | 138/138 |
| VVC | x265 medium | -33.5% | -36.6% | -39.2% | -43.3 … -26.3 | 138/138 |
| VVC Fast | HEVC Fast | -2.1% | +0.9% | -4.0% | -4.1 … +2.5 | 138/138 |
| VVC Fast | VVC | +21.4% | +24.9% | +23.0% | +17.8 … +28.1 | 138/138 |

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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.6 [-2.8, -0.6]%); x264 medium --tune psnr (-10.2 [-11.2, -9.3]%); x265 ultrafast --tune psnr (-1.0 [-1.3, -0.6]%); x265 superfast --tune psnr (-2.0 [-2.4, -1.7]%); x265 medium --tune psnr (-5.5 [-6.1, -4.7]%); libvpx VP9 rt 7 --tune ssim (-0.9 [-1.4, -0.4]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
