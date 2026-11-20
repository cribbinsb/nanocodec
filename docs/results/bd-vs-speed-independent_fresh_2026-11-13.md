# Bitrate vs encode speed, held-out clips (independent_fresh_2026-11-13)

<sub>Updated 2026-11-20, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Fast, H.264 Faster, H.264 Veryfast, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster: 20d31869; H.264: c9cff89f)</sub>

Quality on the 138 clips of `independent_fresh_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_fresh_2026-11-13.svg)

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_fresh_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Slow** | -55.3% | -57.1% | -59.9% | -62.3 … -47.1 | 16.0 | 0.60× | 64.4 | 0.57× | 136/136 | Pareto; Pareto x86 |
| **AV2 Slow** | -54.0% | -58.2% | -58.0% | -61.4 … -47.5 | 16.1 | 0.60× | 62.3 | 0.55× | 136/136 | Pareto |
| **VVC** | -53.6% | -55.4% | -58.6% | -61.8 … -45.6 | 29.8 | 1.12× | 121.0 | 1.07× | 136/136 | Pareto; Pareto x86 |
| **HEVC Slow** | -53.4% | -54.9% | -57.5% | -60.9 … -44.4 | 19.0 | 0.71× | 76.9 | 0.68× | 136/136 |  |
| **AV2** | -51.7% | -56.3% | -56.1% | -60.3 … -44.1 | 39.2 | 1.47× | 170.1 | 1.51× | 136/136 | Pareto; Pareto x86 |
| **HEVC** | -50.8% | -52.6% | -55.6% | -59.5 … -41.5 | 45.3 | 1.70× | 171.9 | 1.52× | 136/136 | Pareto; Pareto x86 |
| **AV1 Slow** | -50.4% | -54.5% | -53.0% | -57.6 … -43.9 | 26.6 | 1.00× | 113.3 | 1.00× | 136/136 | 1 rate-invalid pts |
| **VVC Fast** | -49.4% | -51.4% | -54.3% | -59.2 … -40.0 | 53.2 | 1.99× | 205.2 | 1.82× | 136/136 | Pareto; Pareto x86 |
| **AV1** | -48.3% | -52.9% | -51.8% | -57.1 … -41.6 | 45.7 | 1.71× | 181.4 | 1.61× | 136/136 | 1 rate-invalid pts |
| **AV2 Fast** | -48.1% | -51.0% | -51.2% | -57.7 … -38.7 | 51.8 | 1.94× | 230.8 | 2.05× | 136/136 | Pareto x86 |
| **VP9 Slow** | -48.0% | -51.4% | -51.1% | -55.9 … -38.9 | 26.7 | 1.00× | 100.0 | 0.89× | 136/136 | 2 rate-invalid pts |
| **VP9** | -47.6% | -51.1% | -51.0% | -56.4 … -38.5 | 42.1 | 1.58× | 171.0 | 1.52× | 136/136 | 3 rate-invalid pts |
| **AV2 Faster** | -46.8% | -49.7% | -49.9% | -56.7 … -37.1 | 62.9 | 2.36× | 274.8 | 2.44× | 136/136 | Pareto; Pareto x86 |
| **HEVC Fast** | -46.4% | -48.0% | -50.1% | -54.9 … -37.9 | 72.0 | 2.70× | 293.2 | 2.60× | 136/136 | Pareto; Pareto x86 |
| **AV1 Fast** | -45.1% | -49.1% | -48.3% | -54.9 … -37.2 | 66.2 | 2.48× | 267.6 | 2.37× | 136/136 |  |
| **VP9 Fast** | -43.3% | -46.7% | -45.4% | -51.6 … -34.0 | 55.8 | 2.09× | 233.7 | 2.07× | 136/136 | 1 rate-invalid pts |
| SVT-AV1 M8 | -42.9% | -45.9% | -42.6% | -47.5 … -32.1 | 14.3 | 0.53× | 80.8 | 0.72× | 136/136 | 27 rate-invalid pts |
| **AV1 Faster** | -42.7% | -45.3% | -47.8% | -53.7 … -34.6 | 79.4 | 2.98× | 323.9 | 2.87× | 136/136 | Pareto; Pareto x86 |
| **HEVC Faster** | -40.4% | -41.7% | -44.6% | -50.8 … -30.5 | 90.3 | 3.38× | 364.2 | 3.23× | 136/136 | Pareto; Pareto x86 |
| **VP9 Faster** | -39.6% | -40.5% | -43.5% | -51.1 … -29.9 | 92.3 | 3.46× | 368.7 | 3.27× | 136/136 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -37.9% | -38.9% | -42.1% | -49.9 … -26.4 | 96.5 | 3.62× | 431.4 | 3.83× | 136/136 | Pareto; Pareto x86 |
| SVT-AV1 M10 | -36.3% | -38.3% | -38.1% | -42.9 … -25.1 | 39.3 | 1.47× | 228.9 | 2.03× | 136/136 | 25 rate-invalid pts |
| **VP9 Veryfast** | -36.3% | -38.2% | -40.3% | -47.3 … -26.8 | 95.8 | 3.59× | 385.9 | 3.42× | 136/136 |  |
| **H.264 Slow** | -34.6% | -38.7% | -40.3% | -44.3 … -25.0 | 37.2 | 1.39× | 115.3 | 1.02× | 136/136 |  |
| x265 medium, metric-tuned (--tune psnr) | -34.5% | -35.7% | -37.8% | -42.6 … -26.3 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 32 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264** | -32.8% | -36.4% | -38.9% | -43.3 … -23.5 | 66.4 | 2.49× | 202.2 | 1.79× | 136/136 |  |
| x265 superfast, metric-tuned (--tune psnr) | -32.3% | -32.6% | -35.7% | -40.6 … -24.1 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 30 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 medium | -31.2% | -30.1% | -32.9% | -39.5 … -22.8 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 34 rate-invalid pts |
| x265 superfast | -31.1% | -30.0% | -35.1% | -38.4 … -22.0 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 31 rate-invalid pts |
| SVT-AV1 M12 | -29.8% | -30.7% | -33.6% | -36.4 … -17.1 | 70.9 | 2.67× | 408.7 | 3.62× | 136/136 | 21 rate-invalid pts |
| SVT-AV1 M13 | -27.9% | -28.5% | -32.0% | -35.0 … -15.1 | 79.2 | 2.96× | 445.4 | 3.95× | 136/136 | Pareto x86; 21 rate-invalid pts |
| libaom rt 8 | -27.6% | -26.0% | -31.6% | -33.8 … -18.4 | 17.4 | 0.65× | 68.8 | 0.61× | 136/136 | 94 rate-invalid pts |
| libaom rt 10 | -27.6% | -28.1% | -30.2% | -32.7 … -18.1 | 22.3 | 0.83× | 90.5 | 0.80× | 136/136 | 96 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -25.9% | -25.1% | -30.1% | -36.1 … -13.7 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 30 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Fast** | -25.7% | -28.9% | -32.1% | -37.6 … -13.8 | 119.5 | 4.48× | 370.1 | 3.28× | 136/136 | Pareto |
| x265 ultrafast | -25.3% | -23.3% | -29.9% | -35.2 … -12.2 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 30 rate-invalid pts |
| **H.264 Faster** | -18.4% | -22.2% | -24.7% | -32.7 … -2.3 | 146.9 | 5.51× | 460.1 | 4.08× | 136/136 | Pareto; Pareto x86 |
| **H.264 Veryfast** | -15.6% | -20.0% | -21.7% | -30.9 … +1.3 | 164.1 | 6.15× | 520.2 | 4.61× | 136/136 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -10.1% | -9.2% | -20.1% | -12.6 … -6.3 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 47 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | -3.0% | -5.3% | -15.0% | -11.8 … +6.9 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -2.0% | +4.4% | -9.4% | -9.0 … +8.1 | 47.8 | 1.78× | 204.4 | 1.81× | 136/136 | 161 rate-invalid pts |
| x264 veryfast | -1.7% | -4.7% | -1.5% | -8.8 … +6.9 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 42 rate-invalid pts |
| libvpx VP9 rt 7 | +1.0% | +5.8% | -6.8% | -6.9 … +13.9 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 155 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.1% | +6.3% | -1.8% | -6.9 … +10.9 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 135 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264 | +14.2% | +10.2% | +9.8% | -15.6 … +43.5 | 111.5 | 4.17× | 524.7 | 4.65× | 136/136 | Pareto x86; 253 rate-invalid pts |
| x264 ultrafast | +30.0% | +39.8% | +5.6% | +10.7 … +48.8 | 190.5 | 7.11× | 1017.0 | 9.02× | 136/136 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bdd100k_mot_bdd_00d9e313_7d75bb18, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video295, ind_bwc_videotext_video317, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_chirla_chirla_seq_026_camera_3_2023_12_05_12_12_13, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00212, ind_personpath22_uid_vid_00214, ind_personpath22_uid_vid_00216, ind_personpath22_uid_vid_00231, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 2; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 12, crowd 39, lowlight 8, pan 42, static 35 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slow | -34.6% | -35.4% |
| H.264 | -32.8% | -33.6% |
| H.264 Fast | -25.7% | -26.1% |
| H.264 Faster | -18.4% | -18.4% |
| H.264 Veryfast | -15.6% | -15.3% |
| HEVC Slow | -53.4% | -53.0% |
| HEVC | -50.8% | -50.5% |
| HEVC Fast | -46.4% | -46.6% |
| HEVC Faster | -40.4% | -40.8% |
| AV1 Slow | -50.4% | -51.7% |
| AV1 | -48.3% | -49.7% |
| AV1 Fast | -45.1% | -46.1% |
| AV1 Faster | -42.7% | -44.0% |
| AV1 Veryfast | -37.9% | -38.4% |
| VP9 Slow | -48.0% | -48.2% |
| VP9 | -47.6% | -47.7% |
| VP9 Fast | -43.3% | -43.1% |
| VP9 Faster | -39.6% | -39.4% |
| VP9 Veryfast | -36.3% | -36.6% |
| AV2 Slow | -54.0% | -54.6% |
| AV2 | -51.7% | -52.2% |
| AV2 Fast | -48.1% | -48.0% |
| AV2 Faster | -46.8% | -46.7% |
| VVC Slow | -55.3% | -55.1% |
| VVC | -53.6% | -53.7% |
| VVC Fast | -49.4% | -49.2% |
| openh264 | +14.2% | +14.3% |
| x264 ultrafast | +30.0% | +34.4% |
| x264 veryfast | -1.7% | -0.8% |
| x265 ultrafast | -25.3% | -24.0% |
| x265 superfast | -31.1% | -29.9% |
| x265 medium | -31.2% | -30.7% |
| SVT-AV1 M8 | -42.9% | -41.0% |
| SVT-AV1 M10 | -36.3% | -34.8% |
| SVT-AV1 M12 | -29.8% | -27.9% |
| SVT-AV1 M13 | -27.9% | -26.2% |
| libvpx VP9 rt 7 | +1.0% | +4.4% |
| libvpx VP9 rt 9 | -2.0% | -0.1% |
| libaom rt 8 | -27.6% | -25.6% |
| libaom rt 10 | -27.6% | -25.8% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +14.2% | +10.2% | +9.8% | +22.4% | 136/136 |  |
| openh264, adaptive quantisation on | +14.2% | +10.2% | +9.8% | +22.4% | 136/136 | +0.0 [-0.0, +0.1] not distinguishable |
| x264 ultrafast | +30.0% | +39.8% | +5.6% | +44.6% | 136/136 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +29.9% | +39.5% | +5.6% | +44.7% | 136/136 | -0.1 [-0.3, +0.2] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +26.2% | +35.6% | +19.8% | +23.2% | 136/136 | -0.3 [-1.7, +0.9] not distinguishable |
| x264 veryfast | -1.7% | -4.7% | -1.5% | +1.0% | 136/136 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | -3.0% | -5.3% | -15.0% | +11.3% | 136/136 | -1.5 [-2.5, -0.5] better |
| x264 veryfast, metric-tuned (--tune ssim) | -0.9% | -3.2% | +0.7% | -0.1% | 136/136 | +0.6 [+0.0, +1.2] worse |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 136/136 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -10.1% | -9.2% | -20.1% | -1.0% | 136/136 | -10.1 [-11.0, -9.2] better |
| x264 medium, metric-tuned (--tune ssim) | -2.2% | -1.9% | -0.6% | -4.0% | 136/136 | -2.2 [-2.9, -1.4] better |
| x265 ultrafast | -25.3% | -23.3% | -29.9% | -22.6% | 136/136 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -25.9% | -25.1% | -30.1% | -22.4% | 136/136 | -0.9 [-1.3, -0.6] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -25.9% | -25.1% | -30.1% | -22.4% | 136/136 | -0.9 [-1.3, -0.6] better |
| x265 superfast | -31.1% | -30.0% | -35.1% | -28.1% | 136/136 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -32.3% | -32.6% | -35.7% | -28.5% | 136/136 | -2.0 [-2.3, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -32.3% | -32.6% | -35.7% | -28.5% | 136/136 | -2.0 [-2.3, -1.6] better |
| x265 medium | -31.2% | -30.1% | -32.9% | -30.5% | 136/136 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -34.5% | -35.7% | -37.8% | -29.9% | 136/136 | -5.4 [-6.0, -4.7] better |
| x265 medium, metric-tuned (--tune ssim) | -32.4% | -32.5% | -33.6% | -31.0% | 136/136 | -1.9 [-2.1, -1.6] better |
| libvpx VP9 rt 7 | +1.0% | +5.8% | -6.8% | +3.9% | 136/136 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +1.1% | +6.3% | -1.8% | -1.3% | 136/136 | -0.9 [-1.4, -0.4] better |
| libvpx VP9 rt 9 | -2.0% | +4.4% | -9.4% | -0.9% | 136/136 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.9% | +4.9% | -8.6% | -2.0% | 136/136 | -0.2 [-0.6, +0.3] not distinguishable |
| libaom rt 8 | -27.6% | -26.0% | -31.6% | -25.3% | 136/136 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -27.6% | -24.8% | -27.2% | -30.8% | 136/136 | +0.8 [+0.2, +1.6] worse |
| libaom rt 10 | -27.6% | -28.1% | -30.2% | -24.4% | 136/136 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -27.6% | -26.6% | -26.8% | -29.4% | 136/136 | +0.7 [+0.1, +1.3] worse |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Slow** | -49.1% | -49.3% | -48.4% | -49.6% | 136/136 |
| **AV2 Slow** | -47.9% | -50.9% | -46.3% | -46.6% | 136/136 |
| **VVC** | -47.2% | -47.3% | -46.6% | -47.7% | 136/136 |
| **HEVC Slow** | -47.1% | -47.3% | -45.3% | -48.6% | 136/136 |
| **AV2** | -45.3% | -48.6% | -43.6% | -43.8% | 136/136 |
| **HEVC** | -44.1% | -44.4% | -42.7% | -45.1% | 136/136 |
| **AV1 Slow** | -43.9% | -47.1% | -40.0% | -44.5% | 136/136 |
| **VVC Fast** | -42.3% | -42.6% | -41.0% | -43.3% | 136/136 |
| **AV1** | -41.4% | -45.2% | -38.2% | -40.9% | 136/136 |
| **VP9 Slow** | -41.0% | -43.5% | -37.1% | -42.3% | 136/136 |
| **AV2 Fast** | -40.9% | -42.6% | -37.3% | -42.9% | 136/136 |
| **VP9** | -40.5% | -43.0% | -36.8% | -41.7% | 136/136 |
| **AV2 Faster** | -39.4% | -41.0% | -35.5% | -41.7% | 136/136 |
| **HEVC Fast** | -38.9% | -39.2% | -35.6% | -42.0% | 136/136 |
| **AV1 Fast** | -37.8% | -40.9% | -33.7% | -38.7% | 136/136 |
| **VP9 Fast** | -35.5% | -38.0% | -29.8% | -38.6% | 136/136 |
| **AV1 Faster** | -35.1% | -36.5% | -33.0% | -35.8% | 136/136 |
| SVT-AV1 M8 | -34.9% | -36.7% | -26.8% | -41.2% | 136/136 |
| **HEVC Faster** | -32.1% | -32.3% | -28.6% | -35.3% | 136/136 |
| **VP9 Faster** | -31.2% | -31.0% | -27.3% | -35.2% | 136/136 |
| **AV1 Veryfast** | -29.6% | -29.6% | -25.9% | -33.4% | 136/136 |
| **VP9 Veryfast** | -27.5% | -28.5% | -23.2% | -30.7% | 136/136 |
| SVT-AV1 M10 | -27.2% | -27.8% | -20.6% | -33.3% | 136/136 |
| **H.264 Slow** | -26.0% | -29.6% | -23.4% | -24.9% | 136/136 |
| x265 medium, metric-tuned (--tune psnr) | -25.9% | -26.3% | -20.7% | -30.5% | 136/136 |
| **H.264** | -23.8% | -26.9% | -21.6% | -23.0% | 136/136 |
| x265 medium, metric-tuned (--tune ssim) | -23.1% | -22.4% | -15.2% | -31.7% | 136/136 |
| x265 superfast, metric-tuned (--tune psnr) | -23.1% | -22.3% | -17.9% | -29.2% | 136/136 |
| x265 superfast, metric-tuned (--tune ssim) | -23.1% | -22.3% | -17.9% | -29.2% | 136/136 |
| x265 medium | -21.7% | -19.7% | -14.2% | -31.2% | 136/136 |
| x265 superfast | -21.6% | -19.3% | -16.9% | -28.7% | 136/136 |
| SVT-AV1 M12 | -20.1% | -19.7% | -15.1% | -25.6% | 136/136 |
| SVT-AV1 M13 | -18.0% | -17.3% | -13.3% | -23.4% | 136/136 |
| libaom rt 8 | -17.1% | -15.1% | -11.8% | -25.1% | 135/136 |
| libaom rt 10 | -16.8% | -16.9% | -9.9% | -24.2% | 135/136 |
| libaom rt 8, metric-tuned (--tune ssim) | -16.7% | -13.6% | -6.7% | -30.6% | 135/136 |
| libaom rt 10, metric-tuned (--tune ssim) | -16.6% | -15.2% | -6.0% | -29.4% | 135/136 |
| **H.264 Fast** | -16.0% | -18.7% | -13.3% | -15.8% | 136/136 |
| x265 ultrafast, metric-tuned (--tune psnr) | -15.7% | -14.1% | -10.5% | -22.7% | 136/136 |
| x265 ultrafast, metric-tuned (--tune ssim) | -15.7% | -14.1% | -10.5% | -22.7% | 136/136 |
| x265 ultrafast | -15.1% | -12.0% | -10.3% | -22.9% | 136/136 |
| **H.264 Faster** | -7.8% | -11.8% | -4.2% | -7.5% | 136/136 |
| **H.264 Veryfast** | -4.8% | -9.5% | -0.5% | -4.4% | 136/136 |
| x264 veryfast, metric-tuned (--tune psnr) | +8.4% | +4.9% | +7.8% | +12.5% | 136/136 |
| x264 medium, metric-tuned (--tune ssim) | +10.6% | +8.2% | +26.5% | -3.0% | 136/136 |
| x264 veryfast | +11.4% | +5.7% | +26.6% | +1.9% | 136/136 |
| libvpx VP9 rt 9 | +11.7% | +18.9% | +16.6% | -0.5% | 136/136 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.8% | +19.4% | +17.5% | -1.7% | 136/136 |
| x264 veryfast, metric-tuned (--tune ssim) | +12.4% | +7.3% | +29.2% | +0.6% | 136/136 |
| x264 medium | +13.1% | +10.9% | +27.0% | +1.4% | 136/136 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.2% | +19.4% | +23.6% | -0.4% | 136/136 |
| libvpx VP9 rt 7 | +14.3% | +19.4% | +18.7% | +4.7% | 136/136 |
| openh264 | +30.6% | +27.1% | +41.1% | +23.6% | 136/136 |
| openh264, adaptive quantisation on | +30.6% | +27.1% | +41.1% | +23.6% | 136/136 |
| x264 ultrafast, metric-tuned (--tune ssim) | +43.3% | +48.9% | +55.8% | +25.1% | 136/136 |
| x264 ultrafast, metric-tuned (--tune psnr) | +45.2% | +52.5% | +35.7% | +47.4% | 136/136 |
| x264 ultrafast | +45.3% | +52.8% | +35.7% | +47.4% | 136/136 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.
- **ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508** (static): no reliable overlap for AV2 Slow, AV2.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | ind_bwc_videotext_video88 | ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508 |
|---|---:|---:|
| VVC Slow | -66.8% | -65.3% |
| AV2 Slow | -65.2% | n/a |
| VVC | -66.5% | -65.6% |
| HEVC Slow | -61.0% | -68.4% |
| AV2 | -64.7% | n/a |
| HEVC | -59.1% | -66.1% |
| AV1 Slow | -56.5% | -67.0% |
| VVC Fast | -63.4% | -61.9% |
| AV1 | -55.9% | -64.9% |
| AV2 Fast | -65.5% | -66.0% |
| VP9 Slow | -55.9% | -65.8% |
| VP9 | -55.3% | -64.2% |
| AV2 Faster | -64.7% | -66.4% |
| HEVC Fast | -56.0% | -63.1% |
| AV1 Fast | -58.1% | -65.0% |
| VP9 Fast | -59.5% | -59.6% |
| SVT-AV1 M8 | -60.8% | -55.6% |
| AV1 Faster | -56.6% | -64.5% |
| HEVC Faster | -52.6% | -59.7% |
| VP9 Faster | -55.9% | -57.4% |
| AV1 Veryfast | -55.8% | -62.9% |
| SVT-AV1 M10 | -56.2% | -52.5% |
| VP9 Veryfast | -51.6% | -56.1% |
| H.264 Slow | -24.1% | -61.9% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a |
| H.264 | -22.1% | -60.6% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 medium | -35.4% | -40.3% |
| x265 superfast | -47.1% | -42.3% |
| SVT-AV1 M12 | -50.6% | -53.7% |
| SVT-AV1 M13 | -49.0% | -52.3% |
| libaom rt 8 | n/a | -22.4% |
| libaom rt 10 | -38.6% | -32.2% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a |
| H.264 Fast | -22.4% | -58.4% |
| x265 ultrafast | -45.2% | -41.7% |
| H.264 Faster | -19.0% | -57.7% |
| H.264 Veryfast | -15.3% | -54.7% |
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
| H.264 Slow | openh264 | -38.1% | -39.8% | -40.9% | -48.9 … -29.2 | 138/138 |
| H.264 | openh264 | -36.0% | -37.2% | -39.3% | -47.5 … -26.3 | 138/138 |
| H.264 | x264 veryfast | -37.1% | -38.9% | -43.5% | -44.8 … -30.9 | 138/138 |
| H.264 | x264 medium | -32.8% | -36.4% | -38.9% | -43.5 … -23.2 | 138/138 |
| H.264 Fast | openh264 | -28.2% | -29.1% | -32.1% | -40.3 … -17.2 | 138/138 |
| H.264 Fast | x264 veryfast | -30.1% | -31.7% | -36.7% | -37.2 … -22.7 | 138/138 |
| H.264 Fast | H.264 | +12.3% | +12.9% | +12.2% | +9.1 … +16.7 | 138/138 |
| H.264 Faster | openh264 | -19.3% | -21.3% | -23.5% | -30.0 … -10.2 | 138/138 |
| H.264 Faster | x264 ultrafast | -43.5% | -52.1% | -35.1% | -52.8 … -38.0 | 129/138 |
| H.264 Faster | H.264 Fast | +11.4% | +9.9% | +12.2% | +5.9 … +17.0 | 138/138 |
| H.264 Veryfast | openh264 | -15.7% | -18.6% | -19.9% | -28.5 … -5.6 | 138/138 |
| H.264 Veryfast | x264 ultrafast | -41.2% | -50.7% | -31.7% | -50.7 … -36.3 | 131/138 |
| H.264 Veryfast | H.264 Faster | +4.7% | +3.6% | +5.0% | +2.9 … +6.8 | 138/138 |
| HEVC Slow | SVT-AV1 M12 | -35.5% | -36.6% | -37.5% | -42.8 … -33.6 | 138/138 |
| HEVC | SVT-AV1 M12 | -31.5% | -33.0% | -34.3% | -39.3 … -29.8 | 138/138 |
| HEVC | x265 medium | -29.7% | -33.2% | -35.2% | -39.9 … -21.1 | 138/138 |
| HEVC Fast | SVT-AV1 M12 | -23.9% | -25.1% | -24.7% | -34.9 … -22.7 | 138/138 |
| HEVC Fast | x265 superfast | -23.3% | -26.8% | -23.8% | -34.3 … -18.0 | 138/138 |
| HEVC Fast | HEVC | +11.3% | +11.9% | +14.9% | +7.5 … +12.5 | 138/138 |
| HEVC Faster | SVT-AV1 M12 | -13.7% | -14.7% | -15.4% | -26.7 … -10.7 | 138/138 |
| HEVC Faster | SVT-AV1 M13 | -16.7% | -18.1% | -17.9% | -28.2 … -14.3 | 138/138 |
| HEVC Faster | HEVC Fast | +14.2% | +14.5% | +13.4% | +8.1 … +17.6 | 138/138 |
| AV1 Slow | SVT-AV1 M12 | -30.8% | -36.0% | -29.7% | -42.9 … -30.9 | 138/138 |
| AV1 | SVT-AV1 M12 | -27.5% | -33.6% | -27.9% | -39.3 … -27.5 | 138/138 |
| AV1 | openh264 | -52.2% | -54.8% | -52.9% | -64.1 … -42.3 | 136/138 |
| AV1 | SVT-AV1 M8 | -6.9% | -12.0% | -13.5% | -21.9 … -6.2 | 138/138 |
| AV1 | libaom rt 8 | -25.6% | -32.9% | -26.9% | -40.1 … -21.8 | 138/138 |
| AV1 Fast | SVT-AV1 M12 | -22.5% | -27.8% | -22.5% | -33.8 … -22.3 | 138/138 |
| AV1 Fast | openh264 | -48.7% | -50.6% | -48.9% | -61.6 … -37.8 | 137/138 |
| AV1 Fast | AV1 | +7.6% | +9.4% | +8.7% | +5.4 … +11.6 | 138/138 |
| AV1 Faster | SVT-AV1 M12 | -18.3% | -21.5% | -21.5% | -33.0 … -16.3 | 138/138 |
| AV1 Faster | AV1 Fast | +5.1% | +8.6% | +0.9% | +0.1 … +8.9 | 138/138 |
| AV1 Veryfast | SVT-AV1 M12 | -11.2% | -11.3% | -13.3% | -28.0 … -8.1 | 138/138 |
| AV1 Veryfast | SVT-AV1 M13 | -14.3% | -14.7% | -15.8% | -30.0 … -9.6 | 138/138 |
| AV1 Veryfast | AV1 Faster | +10.4% | +14.9% | +12.0% | +5.1 … +18.7 | 138/138 |
| VP9 Slow | HEVC Slow | +13.1% | +8.2% | +17.1% | +8.4 … +15.6 | 138/138 |
| VP9 | HEVC | +7.4% | +2.9% | +12.1% | +3.6 … +10.0 | 138/138 |
| VP9 | libvpx VP9 rt 7 | -44.6% | -49.4% | -42.9% | -55.2 … -38.7 | 138/138 |
| VP9 Fast | HEVC Fast | +6.9% | +2.5% | +11.0% | +2.5 … +12.7 | 138/138 |
| VP9 Fast | libvpx VP9 rt 9 | -40.9% | -47.7% | -38.0% | -48.8 … -35.9 | 138/138 |
| VP9 Fast | VP9 | +10.2% | +10.8% | +13.1% | +6.5 … +15.1 | 138/138 |
| VP9 Faster | HEVC Faster | +1.4% | +2.3% | +1.9% | -2.1 … +7.2 | 138/138 |
| VP9 Faster | libvpx VP9 rt 9 | -36.1% | -40.5% | -35.4% | -47.1 … -29.1 | 138/138 |
| VP9 Faster | VP9 Fast | +8.5% | +14.4% | +3.9% | +4.4 … +13.3 | 138/138 |
| VP9 Veryfast | HEVC Faster | +9.2% | +7.7% | +9.6% | +5.1 … +13.9 | 138/138 |
| VP9 Veryfast | libvpx VP9 rt 9 | -31.7% | -37.9% | -30.8% | -43.6 … -26.8 | 138/138 |
| VP9 Veryfast | VP9 Faster | +8.4% | +5.7% | +8.2% | +1.8 … +10.4 | 138/138 |
| AV2 Slow | AV1 Slow | -7.7% | -8.0% | -11.3% | -9.9 … -4.4 | 138/138 |
| AV2 | AV1 | -7.3% | -6.8% | -9.6% | -8.5 … -2.7 | 138/138 |
| AV2 | SVT-AV1 M8 | -13.7% | -17.9% | -22.1% | -25.9 … -11.8 | 138/138 |
| AV2 Fast | AV1 Fast | -6.3% | -3.8% | -6.8% | -8.6 … +0.3 | 138/138 |
| AV2 Fast | AV2 | +9.1% | +13.4% | +12.4% | +6.9 … +13.6 | 138/138 |
| AV2 Faster | AV1 Faster | -8.5% | -9.0% | -4.7% | -9.2 … -2.1 | 138/138 |
| AV2 Faster | SVT-AV1 M12 | -26.2% | -28.7% | -26.3% | -35.1 … -22.8 | 138/138 |
| AV2 Faster | AV2 Fast | +2.9% | +3.2% | +3.1% | +0.7 … +4.9 | 138/138 |
| VVC Slow | HEVC Slow | -4.8% | -5.0% | -6.3% | -6.8 … -2.9 | 138/138 |
| VVC | HEVC | -7.0% | -6.9% | -7.8% | -10.3 … -5.6 | 138/138 |
| VVC | x265 medium | -34.0% | -37.4% | -39.9% | -44.0 … -26.5 | 138/138 |
| VVC Fast | HEVC Fast | -7.2% | -7.9% | -10.1% | -8.7 … -3.4 | 138/138 |
| VVC Fast | VVC | +11.1% | +10.9% | +12.1% | +9.5 … +14.2 | 138/138 |

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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): x264 veryfast --tune psnr (-1.5 [-2.5, -0.5]%); x264 medium --tune psnr (-10.1 [-11.0, -9.2]%); x265 ultrafast --tune psnr (-0.9 [-1.3, -0.6]%); x265 superfast --tune psnr (-2.0 [-2.3, -1.6]%); x265 medium --tune psnr (-5.4 [-6.0, -4.7]%); libvpx VP9 rt 7 --tune ssim (-0.9 [-1.4, -0.4]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
