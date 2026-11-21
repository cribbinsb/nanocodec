# Bitrate vs encode speed, held-out clips (independent_fresh_2026-11-13), zero line x264-medium~psnr

<sub>Updated 2026-11-21, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264, H.264 Veryfast timed on 4fc9a1de (same streams)</sub>

**Zero line: x264 medium, metric-tuned (--tune psnr)** (the default-tuned x264 medium is a row of its own). The low rungs are the default chart's, so the points are the same.

Quality on the 138 clips of `independent_fresh_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr.svg)

**Rate control and the buffer.** Every encoder in the CBR charts is asked for a 1 s buffer at the target rate. x264, x265 and NVENC are held to it by a hard VBV, and SVT-AV1, libaom and libvpx by their 1000 ms real-time buffer. nanocodec 0.6.0 treats its buffer as a target with a 1.4x stop and sizes its key frames more generously: in these charts its buffer peak passes one buffer on 10-16 % of points (theirs: 0-1 %). openh264 is given no buffer setting and behaves like nanocodec in this respect. That is worth quality. On a sample of the held-out clips (24 clips at 150 frames, 12 at 600 frames; AV1, HEVC and H.264 medium), holding nanocodec's stream inside one buffer, as the references' is, cost it 3-5.5 % in bits at equal quality: its lead over SVT-AV1 M12 at 150 frames moved from 31 % to 28 % (AV1) and from 33 % to 29 % (HEVC), H.264 medium moved from level with SVT-AV1 M12 to about 5 % behind, and the leads over x264 medium moved by 0.6-2.2 points. The Fixed QP charts have no rate control and are not affected. A later release will make the buffer a hard bound by default and re-measure these charts.

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_fresh_2026-11-13-vs-x264-medium-psnr-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Veryslow** | -54.9% | -55.3% | -54.0% | -60.4 … -48.7 | 7.6 | 0.29× | 30.8 | 0.27× | 137/137 | Pareto; Pareto x86 |
| **VVC Slower** | -53.9% | -54.2% | -53.0% | -59.7 … -47.9 | 13.6 | 0.51× | 51.9 | 0.46× | 137/137 | Pareto; Pareto x86 |
| **VVC Slow** | -52.4% | -52.4% | -51.6% | -59.1 … -46.6 | 18.0 | 0.68× | 73.0 | 0.65× | 137/137 | Pareto; Pareto x86 |
| **HEVC Veryslow** | -52.0% | -52.6% | -50.3% | -57.1 … -46.2 | 10.2 | 0.38× | 36.9 | 0.33× | 137/137 |  |
| **AV2 Slower** | -51.9% | -54.7% | -50.1% | -58.1 … -47.2 | 7.9 | 0.30× | 26.6 | 0.24× | 137/137 | 1 rate-invalid pts |
| **VVC** | -50.6% | -50.8% | -49.8% | -57.6 … -44.2 | 25.0 | 0.94× | 101.7 | 0.90× | 137/137 | Pareto; Pareto x86 |
| **HEVC Slower** | -49.4% | -49.6% | -47.9% | -55.1 … -41.4 | 17.3 | 0.65× | 68.2 | 0.60× | 137/137 |  |
| **AV2 Slow** | -49.1% | -52.2% | -47.5% | -56.1 … -43.3 | 27.5 | 1.03× | 120.8 | 1.07× | 137/137 | Pareto; Pareto x86; 1 rate-invalid pts |
| **HEVC Slow** | -48.6% | -48.9% | -47.4% | -55.0 … -41.6 | 33.2 | 1.25× | 121.4 | 1.08× | 137/137 | Pareto; Pareto x86 |
| **AV2** | -48.0% | -51.0% | -46.7% | -55.6 … -42.6 | 35.4 | 1.33× | 150.7 | 1.34× | 137/137 | Pareto; Pareto x86; 1 rate-invalid pts |
| **AV1 Slower** | -47.2% | -50.3% | -43.5% | -53.6 … -43.5 | 14.5 | 0.54× | 59.5 | 0.53× | 137/137 |  |
| **VP9 Slower** | -47.2% | -49.2% | -44.2% | -53.4 … -40.7 | 13.0 | 0.49× | 51.5 | 0.46× | 137/137 | 3 rate-invalid pts |
| **HEVC** | -46.5% | -46.8% | -45.5% | -53.9 … -38.5 | 45.2 | 1.69× | 167.9 | 1.49× | 137/137 | Pareto; Pareto x86 |
| **VVC Fast** | -46.3% | -46.5% | -45.1% | -54.7 … -39.0 | 44.8 | 1.68× | 177.3 | 1.57× | 137/137 | Pareto x86; 1 rate-invalid pts |
| **AV2 Fast** | -46.0% | -48.1% | -44.7% | -53.1 … -38.7 | 42.7 | 1.60× | 172.7 | 1.53× | 137/137 | 2 rate-invalid pts |
| **VP9 Slow** | -45.8% | -48.0% | -42.7% | -52.5 … -39.6 | 21.9 | 0.82× | 91.8 | 0.81× | 137/137 | 2 rate-invalid pts |
| **AV1 Slow** | -45.6% | -48.9% | -42.6% | -53.4 … -40.9 | 28.5 | 1.07× | 115.3 | 1.02× | 137/137 | 2 rate-invalid pts |
| **VVC Faster** | -44.2% | -44.6% | -42.6% | -52.8 … -35.9 | 59.4 | 2.23× | 219.8 | 1.95× | 137/137 | Pareto; Pareto x86; 1 rate-invalid pts |
| **AV1** | -43.3% | -47.0% | -40.3% | -51.2 … -37.5 | 45.2 | 1.70× | 179.0 | 1.59× | 137/137 | 2 rate-invalid pts |
| **VP9** | -43.2% | -45.5% | -39.9% | -49.9 … -35.5 | 36.6 | 1.37× | 145.6 | 1.29× | 137/137 | 5 rate-invalid pts |
| **AV2 Faster** | -42.4% | -44.1% | -38.9% | -50.9 … -34.0 | 61.9 | 2.32× | 265.7 | 2.36× | 137/137 | Pareto; Pareto x86; 2 rate-invalid pts |
| **AV1 Fast** | -41.1% | -44.3% | -37.5% | -49.3 … -34.1 | 64.7 | 2.42× | 252.9 | 2.24× | 137/137 | Pareto |
| **HEVC Fast** | -40.9% | -41.4% | -37.9% | -48.2 … -33.8 | 72.0 | 2.70× | 281.8 | 2.50× | 137/137 | Pareto; Pareto x86 |
| **AV1 Faster** | -39.8% | -42.5% | -36.4% | -49.1 … -32.6 | 77.1 | 2.89× | 313.5 | 2.78× | 137/137 | Pareto; Pareto x86 |
| **VP9 Fast** | -38.4% | -41.0% | -33.4% | -45.8 … -30.2 | 59.8 | 2.24× | 247.1 | 2.19× | 137/137 | 2 rate-invalid pts |
| **VP9 Faster** | -36.9% | -37.9% | -31.8% | -46.9 … -25.7 | 79.2 | 2.97× | 311.9 | 2.77× | 137/137 | Pareto |
| SVT-AV1 M8 | -36.0% | -37.8% | -28.1% | -40.8 … -26.8 | 14.3 | 0.53× | 80.8 | 0.72× | 137/137 | 35 rate-invalid pts |
| **HEVC Faster** | -34.8% | -35.3% | -31.5% | -44.2 … -26.5 | 90.0 | 3.37× | 350.2 | 3.11× | 137/137 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -34.6% | -36.0% | -29.5% | -45.3 … -23.8 | 95.2 | 3.57× | 419.5 | 3.72× | 137/137 | Pareto; Pareto x86 |
| **VP9 Veryfast** | -29.8% | -31.1% | -25.8% | -39.9 … -20.5 | 100.7 | 3.77× | 393.8 | 3.49× | 137/137 | Pareto; 1 rate-invalid pts |
| SVT-AV1 M10 | -29.0% | -29.5% | -22.6% | -33.3 … -19.1 | 39.3 | 1.47× | 228.9 | 2.03× | 137/137 | 33 rate-invalid pts |
| **H.264 Slower** | -27.5% | -31.3% | -25.2% | -39.9 … -20.3 | 37.4 | 1.40× | 113.3 | 1.00× | 137/137 |  |
| **H.264 Slow** | -27.3% | -30.7% | -25.0% | -39.5 … -19.4 | 51.9 | 1.95× | 160.7 | 1.43× | 137/137 |  |
| x265 medium, metric-tuned (--tune psnr) | -27.1% | -27.9% | -22.1% | -34.1 … -18.8 | 8.0 | 0.30× | 38.0 | 0.34× | 137/137 | 41 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264** | -25.7% | -28.8% | -23.7% | -38.6 … -17.6 | 66.2 | 2.48× | 198.4 | 1.76× | 137/137 |  |
| x265 superfast, metric-tuned (--tune psnr) | -25.1% | -24.7% | -19.6% | -31.8 … -16.9 | 12.1 | 0.45× | 57.9 | 0.51× | 137/137 | 37 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 superfast | -23.7% | -21.9% | -18.7% | -31.1 … -15.3 | 12.1 | 0.45× | 57.9 | 0.51× | 137/137 | 38 rate-invalid pts |
| x265 medium | -22.9% | -21.5% | -15.7% | -29.8 … -15.9 | 8.0 | 0.30× | 38.0 | 0.34× | 137/137 | 43 rate-invalid pts |
| SVT-AV1 M12 | -22.1% | -21.6% | -17.4% | -26.7 … -9.8 | 70.9 | 2.67× | 408.7 | 3.62× | 137/137 | 29 rate-invalid pts |
| **H.264 Fast** | -20.8% | -23.1% | -19.2% | -34.6 … -11.6 | 96.9 | 3.63× | 283.4 | 2.51× | 137/137 |  |
| SVT-AV1 M13 | -20.0% | -19.2% | -15.7% | -26.5 … -7.3 | 79.2 | 2.96× | 445.4 | 3.95× | 137/137 | Pareto x86; 29 rate-invalid pts |
| x265 ultrafast, metric-tuned (--tune psnr) | -18.5% | -17.4% | -12.9% | -28.5 … -7.4 | 17.1 | 0.64× | 85.1 | 0.75× | 137/137 | 37 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264 Faster** | -18.3% | -21.4% | -15.7% | -32.0 … -8.4 | 118.3 | 4.44× | 369.8 | 3.28× | 137/137 | Pareto |
| x265 ultrafast | -17.8% | -15.4% | -12.7% | -27.5 … -5.9 | 17.1 | 0.64× | 85.1 | 0.75× | 137/137 | 37 rate-invalid pts |
| libaom rt 8 | -17.3% | -14.7% | -11.9% | -23.7 … -10.0 | 17.4 | 0.65× | 68.8 | 0.61× | 137/137 | 106 rate-invalid pts |
| libaom rt 10 | -17.3% | -16.9% | -10.3% | -23.7 … -10.1 | 22.3 | 0.83× | 90.5 | 0.80× | 137/137 | 108 rate-invalid pts |
| **H.264 Veryfast** | -7.8% | -12.9% | -3.3% | -26.4 … +12.8 | 161.9 | 6.07× | 510.6 | 4.53× | 137/137 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 137/137 | 58 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | +5.6% | +1.7% | +5.5% | -7.2 … +17.4 | 85.4 | 3.18× | 363.3 | 3.22× | 137/137 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast | +8.7% | +2.3% | +23.7% | -2.2 … +20.2 | 85.4 | 3.18× | 363.3 | 3.22× | 137/137 | 1 rate-invalid pts |
| x264 medium, metric-tuned (--tune ssim) | +10.5% | +7.8% | +26.1% | +6.6 … +14.6 | 26.7 | 1.00× | 112.8 | 1.00× | 137/137 | 50 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | +11.3% | +18.7% | +16.0% | +5.6 … +18.6 | 47.8 | 1.78× | 204.4 | 1.81× | 137/137 | 175 rate-invalid pts |
| x264 medium | +13.0% | +10.7% | +26.5% | +7.4 … +16.1 | 26.7 | 1.00× | 112.8 | 1.00× | 137/137 | 54 rate-invalid pts |
| libvpx VP9 rt 7 | +14.8% | +20.0% | +18.9% | +5.2 … +25.4 | 30.0 | 1.12× | 122.7 | 1.09× | 137/137 | 170 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.8% | +20.1% | +23.9% | +4.7 … +22.5 | 30.0 | 1.12× | 122.7 | 1.09× | 137/137 | 150 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264, adaptive quantisation on | +30.5% | +27.2% | +40.4% | -4.6 … +63.5 | 111.5 | 4.17× | 524.7 | 4.65× | 137/137 | 268 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264 | +30.5% | +27.2% | +40.4% | -4.6 … +63.5 | 111.5 | 4.17× | 524.7 | 4.65× | 137/137 | Pareto x86; 268 rate-invalid pts |
| x264 ultrafast | +39.6% | +45.6% | +31.8% | +15.4 … +66.3 | 190.5 | 7.11× | 1017.0 | 9.02× | 137/137 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d77344_9fe25620, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bdd100k_mot_bdd_00d9e313_7d75bb18, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video283, ind_bwc_videotext_video289, ind_bwc_videotext_video295, ind_bwc_videotext_video317, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video47, ind_bwc_videotext_video58, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_chirla_chirla_seq_026_camera_3_2023_12_05_12_12_13, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00127, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00187, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00212, ind_personpath22_uid_vid_00214, ind_personpath22_uid_vid_00216, ind_personpath22_uid_vid_00231, ind_uvg_vcm_uvgvcm_highwayview, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 1; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 12, crowd 39, lowlight 8, pan 42, static 36 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| x264 medium | +13.0% | +12.1% |
| H.264 Slower | -27.5% | -28.8% |
| H.264 Slow | -27.3% | -28.5% |
| H.264 | -25.7% | -27.0% |
| H.264 Fast | -20.8% | -21.7% |
| H.264 Faster | -18.3% | -18.9% |
| H.264 Veryfast | -7.8% | -7.3% |
| HEVC Veryslow | -52.0% | -52.0% |
| HEVC Slower | -49.4% | -49.1% |
| HEVC Slow | -48.6% | -48.5% |
| HEVC | -46.5% | -46.2% |
| HEVC Fast | -40.9% | -41.1% |
| HEVC Faster | -34.8% | -35.3% |
| AV1 Slower | -47.2% | -49.0% |
| AV1 Slow | -45.6% | -47.6% |
| AV1 | -43.3% | -45.0% |
| AV1 Fast | -41.1% | -42.0% |
| AV1 Faster | -39.8% | -40.9% |
| AV1 Veryfast | -34.6% | -35.0% |
| VP9 Slower | -47.2% | -47.5% |
| VP9 Slow | -45.8% | -46.1% |
| VP9 | -43.2% | -43.4% |
| VP9 Fast | -38.4% | -38.1% |
| VP9 Faster | -36.9% | -36.4% |
| VP9 Veryfast | -29.8% | -30.0% |
| AV2 Slower | -51.9% | -52.7% |
| AV2 Slow | -49.1% | -49.7% |
| AV2 | -48.0% | -48.8% |
| AV2 Fast | -46.0% | -46.3% |
| AV2 Faster | -42.4% | -42.4% |
| VVC Veryslow | -54.9% | -54.9% |
| VVC Slower | -53.9% | -53.8% |
| VVC Slow | -52.4% | -52.5% |
| VVC | -50.6% | -50.8% |
| VVC Fast | -46.3% | -46.5% |
| VVC Faster | -44.2% | -44.1% |
| openh264 | +30.5% | +28.3% |
| x264 ultrafast | +39.6% | +46.1% |
| x264 veryfast | +8.7% | +10.0% |
| x265 ultrafast | -17.8% | -16.2% |
| x265 superfast | -23.7% | -22.5% |
| x265 medium | -22.9% | -22.8% |
| SVT-AV1 M8 | -36.0% | -34.2% |
| SVT-AV1 M10 | -29.0% | -27.4% |
| SVT-AV1 M12 | -22.1% | -20.0% |
| SVT-AV1 M13 | -20.0% | -18.3% |
| libvpx VP9 rt 7 | +14.8% | +17.2% |
| libvpx VP9 rt 9 | +11.3% | +12.1% |
| libaom rt 8 | -17.3% | -16.2% |
| libaom rt 10 | -17.3% | -16.3% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +30.5% | +27.2% | +40.4% | +23.8% | 137/137 |  |
| openh264, adaptive quantisation on ● | +30.5% | +27.2% | +40.4% | +23.8% | 137/137 | -0.0 [-0.0, -0.0] better |
| x264 ultrafast | +39.6% | +45.6% | +31.8% | +41.2% | 137/137 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +39.5% | +45.4% | +31.9% | +41.3% | 137/137 | +0.1 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +38.5% | +42.0% | +51.6% | +21.9% | 137/137 | +0.1 [-1.3, +1.3] not distinguishable |
| x264 veryfast | +8.7% | +2.3% | +23.7% | +0.2% | 137/137 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +5.6% | +1.7% | +5.5% | +9.6% | 137/137 | -1.6 [-2.7, -0.6] better |
| x264 veryfast, metric-tuned (--tune ssim) | +9.8% | +3.8% | +26.3% | -0.8% | 137/137 | +0.5 [-0.0, +1.1] not distinguishable |
| x264 medium | +13.0% | +10.7% | +26.5% | +2.0% | 137/137 |  |
| x264 medium, metric-tuned (--tune ssim) ● | +10.5% | +7.8% | +26.1% | -2.3% | 137/137 | -2.2 [-3.0, -1.4] better |
| x265 ultrafast | -17.8% | -15.4% | -12.7% | -25.4% | 137/137 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -18.5% | -17.4% | -12.9% | -25.2% | 137/137 | -0.9 [-1.3, -0.6] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -18.5% | -17.4% | -12.9% | -25.2% | 137/137 | -0.9 [-1.3, -0.6] better |
| x265 superfast | -23.7% | -21.9% | -18.7% | -30.5% | 137/137 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -25.1% | -24.7% | -19.6% | -31.0% | 137/137 | -1.9 [-2.3, -1.6] better |
| x265 superfast, metric-tuned (--tune ssim) | -25.1% | -24.7% | -19.6% | -31.0% | 137/137 | -1.9 [-2.3, -1.6] better |
| x265 medium | -22.9% | -21.5% | -15.7% | -31.7% | 137/137 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -27.1% | -27.9% | -22.1% | -31.3% | 137/137 | -5.4 [-6.0, -4.8] better |
| x265 medium, metric-tuned (--tune ssim) | -24.3% | -24.1% | -16.6% | -32.3% | 137/137 | -1.8 [-2.1, -1.5] better |
| libvpx VP9 rt 7 | +14.8% | +20.0% | +18.9% | +5.4% | 137/137 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +14.8% | +20.1% | +23.9% | +0.5% | 137/137 | -0.7 [-1.2, -0.2] better |
| libvpx VP9 rt 9 | +11.3% | +18.7% | +16.0% | -0.6% | 137/137 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.5% | +19.3% | +17.0% | -1.7% | 137/137 | -0.1 [-0.5, +0.4] not distinguishable |
| libaom rt 8 | -17.3% | -14.7% | -11.9% | -25.3% | 137/137 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -17.2% | -13.6% | -7.1% | -30.8% | 137/137 | +0.7 [+0.1, +1.4] worse |
| libaom rt 10 | -17.3% | -16.9% | -10.3% | -24.6% | 137/137 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -17.3% | -15.6% | -6.6% | -29.8% | 137/137 | +0.6 [+0.0, +1.3] worse |

## Against x264 medium (default tuning)

The same points with x264 medium (default tuning) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Veryslow** | -60.4% | -62.2% | -64.2% | -55.0% | 137/137 |
| **VVC Slower** | -59.5% | -61.3% | -63.4% | -53.9% | 137/137 |
| **VVC Slow** | -58.3% | -59.9% | -62.4% | -52.7% | 137/137 |
| **HEVC Veryslow** | -57.8% | -59.5% | -61.3% | -52.7% | 137/137 |
| **AV2 Slower** | -57.4% | -61.4% | -60.8% | -50.3% | 136/137 |
| **VVC** | -56.6% | -58.4% | -61.0% | -50.5% | 137/137 |
| **HEVC Slower** | -55.5% | -57.0% | -59.4% | -50.2% | 137/137 |
| **AV2 Slow** | -55.1% | -59.3% | -58.8% | -47.2% | 136/137 |
| **HEVC Slow** | -54.8% | -56.5% | -59.0% | -48.9% | 137/137 |
| **AV2** | -54.0% | -58.2% | -58.2% | -45.9% | 136/137 |
| **AV1 Slower** | -53.3% | -57.0% | -55.5% | -47.5% | 137/137 |
| **VP9 Slower** | -53.3% | -56.1% | -56.2% | -47.5% | 137/137 |
| **VVC Fast** | -53.1% | -54.8% | -57.4% | -47.0% | 137/137 |
| **HEVC** | -53.0% | -54.7% | -57.6% | -46.7% | 137/137 |
| **AV2 Fast** | -52.4% | -55.8% | -56.7% | -44.8% | 137/137 |
| **VP9 Slow** | -52.3% | -55.2% | -55.2% | -46.3% | 137/137 |
| **AV1 Slow** | -51.9% | -55.9% | -54.8% | -45.0% | 137/137 |
| **VVC Faster** | -51.2% | -53.3% | -55.5% | -44.9% | 137/137 |
| **VP9** | -49.9% | -53.1% | -53.0% | -43.7% | 137/137 |
| **AV1** | -49.9% | -54.3% | -53.1% | -42.3% | 137/137 |
| **AV2 Faster** | -49.5% | -52.3% | -52.2% | -43.8% | 137/137 |
| **HEVC Fast** | -48.3% | -49.9% | -51.7% | -43.2% | 137/137 |
| **AV1 Fast** | -48.0% | -51.9% | -51.0% | -41.1% | 137/137 |
| **AV1 Faster** | -46.9% | -50.4% | -50.2% | -40.2% | 137/137 |
| **VP9 Fast** | -45.9% | -49.2% | -48.0% | -40.5% | 137/137 |
| **VP9 Faster** | -44.7% | -46.4% | -46.8% | -40.8% | 137/137 |
| SVT-AV1 M8 | -44.2% | -47.1% | -43.6% | -41.8% | 137/137 |
| **HEVC Faster** | -42.9% | -44.4% | -46.7% | -37.5% | 137/137 |
| **AV1 Veryfast** | -42.3% | -44.5% | -44.6% | -37.8% | 137/137 |
| **VP9 Veryfast** | -38.3% | -40.2% | -42.1% | -32.7% | 137/137 |
| SVT-AV1 M10 | -38.2% | -40.1% | -39.7% | -34.8% | 137/137 |
| **H.264 Slower** | -36.1% | -40.2% | -41.5% | -26.6% | 137/137 |
| **H.264 Slow** | -35.9% | -39.8% | -41.4% | -26.7% | 137/137 |
| x265 medium, metric-tuned (--tune psnr) | -35.5% | -36.9% | -38.6% | -31.1% | 137/137 |
| **H.264** | -34.5% | -38.1% | -40.4% | -24.9% | 137/137 |
| x265 superfast, metric-tuned (--tune psnr) | -34.1% | -34.7% | -36.7% | -30.9% | 137/137 |
| x265 superfast, metric-tuned (--tune ssim) | -34.1% | -34.7% | -36.7% | -30.9% | 137/137 |
| x265 medium, metric-tuned (--tune ssim) | -33.4% | -33.8% | -34.3% | -32.1% | 137/137 |
| x265 superfast | -32.9% | -32.3% | -36.1% | -30.4% | 137/137 |
| x265 medium | -32.3% | -31.6% | -33.7% | -31.6% | 137/137 |
| SVT-AV1 M12 | -31.8% | -32.6% | -35.4% | -27.4% | 137/137 |
| **H.264 Fast** | -30.0% | -32.8% | -36.6% | -20.6% | 137/137 |
| SVT-AV1 M13 | -29.9% | -30.4% | -33.9% | -25.5% | 137/137 |
| x265 ultrafast, metric-tuned (--tune psnr) | -28.3% | -28.0% | -31.6% | -25.5% | 137/137 |
| x265 ultrafast, metric-tuned (--tune ssim) | -28.3% | -28.0% | -31.6% | -25.5% | 137/137 |
| libaom rt 10, metric-tuned (--tune ssim) | -28.1% | -27.0% | -27.2% | -30.0% | 137/137 |
| libaom rt 10 | -27.9% | -28.3% | -30.6% | -24.9% | 137/137 |
| **H.264 Faster** | -27.8% | -31.1% | -33.7% | -18.7% | 137/137 |
| libaom rt 8, metric-tuned (--tune ssim) | -27.8% | -24.8% | -27.4% | -31.0% | 137/137 |
| x265 ultrafast | -27.7% | -26.2% | -31.4% | -25.6% | 137/137 |
| libaom rt 8 | -27.6% | -25.9% | -31.6% | -25.6% | 136/137 |
| **H.264 Veryfast** | -18.3% | -22.9% | -23.5% | -8.6% | 137/137 |
| x264 medium, metric-tuned (--tune psnr) | -10.1% | -9.0% | -19.7% | -1.6% | 137/137 |
| x264 veryfast, metric-tuned (--tune psnr) | -5.5% | -7.9% | -16.5% | +7.9% | 137/137 |
| x264 veryfast | -4.0% | -7.5% | -3.2% | -1.4% | 137/137 |
| x264 veryfast, metric-tuned (--tune ssim) | -3.1% | -6.1% | -1.1% | -2.2% | 137/137 |
| libvpx VP9 rt 9 | -2.7% | +3.8% | -10.2% | -1.6% | 137/137 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -2.5% | +4.5% | -9.3% | -2.6% | 137/137 |
| x264 medium, metric-tuned (--tune ssim) | -2.2% | -2.1% | -0.5% | -3.9% | 137/137 |
| libvpx VP9 rt 7 | +1.1% | +6.1% | -6.9% | +4.0% | 137/137 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.5% | +6.9% | -1.5% | -1.0% | 137/137 |
| openh264, adaptive quantisation on | +13.7% | +9.9% | +9.1% | +22.0% | 137/137 |
| openh264 | +13.7% | +9.9% | +9.1% | +22.0% | 137/137 |
| x264 ultrafast, metric-tuned (--tune ssim) | +22.4% | +30.0% | +17.6% | +19.6% | 137/137 |
| x264 ultrafast, metric-tuned (--tune psnr) | +25.2% | +33.8% | +3.4% | +38.3% | 137/137 |
| x264 ultrafast | +25.2% | +34.0% | +3.3% | +38.2% | 137/137 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium, metric-tuned (--tune psnr)'s (even with its extra rungs, which the 50% source cap limits).

- **ind_bwc_videotext_video190** (bodycam): no reliable overlap for HEVC Veryslow, AV2 Slower, VVC Veryslow, libaom rt 8, libaom rt 10.

Direct BD-J vs x264 medium, metric-tuned (--tune psnr) on them, where it exists:

| Encoder | ind_bwc_videotext_video190 |
|---|---:|
| VVC Veryslow | n/a |
| VVC Slower | -40.8% |
| VVC Slow | -37.9% |
| HEVC Veryslow | n/a |
| AV2 Slower | n/a |
| VVC | -35.4% |
| HEVC Slower | -40.2% |
| AV2 Slow | -36.9% |
| HEVC Slow | -33.4% |
| AV2 | -32.4% |
| AV1 Slower | -39.6% |
| VP9 Slower | -42.4% |
| HEVC | -31.4% |
| VVC Fast | -29.2% |
| AV2 Fast | -25.5% |
| VP9 Slow | -43.2% |
| AV1 Slow | -42.5% |
| VVC Faster | -27.2% |
| AV1 | -29.9% |
| VP9 | -31.5% |
| AV2 Faster | -21.8% |
| AV1 Fast | -20.8% |
| HEVC Fast | -21.1% |
| AV1 Faster | -22.4% |
| VP9 Fast | -19.5% |
| VP9 Faster | -19.5% |
| SVT-AV1 M8 | -23.3% |
| HEVC Faster | -20.4% |
| AV1 Veryfast | -26.4% |
| VP9 Veryfast | -15.3% |
| SVT-AV1 M10 | +5.7% |
| H.264 Slower | -24.4% |
| H.264 Slow | -24.3% |
| x265 medium, metric-tuned (--tune psnr) | n/a |
| H.264 | -20.9% |
| x265 superfast, metric-tuned (--tune psnr) | n/a |
| x265 superfast | +10.6% |
| x265 medium | +0.8% |
| SVT-AV1 M12 | +8.8% |
| H.264 Fast | -13.2% |
| SVT-AV1 M13 | +10.1% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a |
| H.264 Faster | -8.6% |
| x265 ultrafast | +20.2% |
| libaom rt 8 | n/a |
| libaom rt 10 | n/a |
| H.264 Veryfast | -4.4% |
| x264 veryfast, metric-tuned (--tune psnr) | n/a |
| x264 veryfast | +24.1% |
| x264 medium, metric-tuned (--tune ssim) | n/a |
| libvpx VP9 rt 9 | +8.6% |
| x264 medium | +2.6% |
| libvpx VP9 rt 7 | -10.5% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a |
| openh264, adaptive quantisation on | n/a |
| openh264 | +26.1% |
| x264 ultrafast | +64.9% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slower | openh264 | -38.0% | -39.7% | -40.8% | -48.7 … -28.3 | 138/138 |
| H.264 Slow | openh264 | -37.6% | -39.0% | -40.6% | -48.6 … -28.6 | 138/138 |
| H.264 | openh264 | -36.0% | -37.1% | -39.4% | -47.5 … -25.9 | 138/138 |
| H.264 | x264 veryfast | -37.8% | -39.6% | -43.6% | -44.8 … -31.7 | 138/138 |
| H.264 | x264 medium | -34.2% | -37.8% | -40.1% | -44.3 … -24.2 | 138/138 |
| H.264 Fast | openh264 | -31.2% | -31.4% | -35.8% | -43.1 … -21.2 | 138/138 |
| H.264 Fast | x264 veryfast | -32.8% | -33.8% | -39.7% | -39.9 … -26.0 | 138/138 |
| H.264 Fast | H.264 | +8.6% | +10.5% | +6.7% | +6.4 … +11.3 | 138/138 |
| H.264 Faster | openh264 | -28.2% | -29.1% | -32.2% | -40.3 … -17.2 | 138/138 |
| H.264 Faster | x264 ultrafast | -48.5% | -55.4% | -41.6% | -55.5 … -43.9 | 119/138 |
| H.264 Faster | H.264 Fast | +3.3% | +1.9% | +5.1% | +2.2 … +5.7 | 138/138 |
| H.264 Veryfast | openh264 | -15.8% | -18.7% | -20.1% | -28.4 … -6.0 | 138/138 |
| H.264 Veryfast | x264 ultrafast | -42.0% | -51.6% | -31.7% | -51.1 … -37.1 | 133/138 |
| H.264 Veryfast | H.264 Faster | +15.8% | +13.1% | +17.0% | +10.5 … +21.6 | 138/138 |
| HEVC Veryslow | SVT-AV1 M12 | -39.8% | -41.3% | -41.4% | -46.8 … -39.0 | 138/138 |
| HEVC Slower | SVT-AV1 M12 | -36.7% | -37.6% | -38.8% | -43.3 … -34.6 | 138/138 |
| HEVC Slow | SVT-AV1 M12 | -35.1% | -36.5% | -37.7% | -43.3 … -33.6 | 138/138 |
| HEVC | SVT-AV1 M12 | -32.1% | -33.5% | -35.3% | -40.4 … -29.8 | 138/138 |
| HEVC | x265 medium | -31.3% | -34.5% | -37.1% | -40.1 … -23.4 | 138/138 |
| HEVC Fast | SVT-AV1 M12 | -23.8% | -25.0% | -24.7% | -34.4 … -22.7 | 138/138 |
| HEVC Fast | x265 superfast | -24.0% | -27.3% | -24.7% | -33.7 … -17.8 | 138/138 |
| HEVC Fast | HEVC | +12.5% | +12.8% | +16.8% | +8.9 … +14.0 | 138/138 |
| HEVC Faster | SVT-AV1 M12 | -14.4% | -15.6% | -16.1% | -27.0 … -11.9 | 138/138 |
| HEVC Faster | SVT-AV1 M13 | -17.4% | -18.9% | -18.6% | -29.1 … -15.1 | 138/138 |
| HEVC Faster | HEVC Fast | +12.7% | +12.9% | +12.0% | +7.6 … +15.7 | 138/138 |
| AV1 Slower | SVT-AV1 M12 | -33.0% | -37.7% | -31.8% | -44.9 … -34.7 | 138/138 |
| AV1 Slow | SVT-AV1 M12 | -30.8% | -36.2% | -30.7% | -43.2 … -31.9 | 138/138 |
| AV1 | SVT-AV1 M12 | -27.5% | -33.6% | -27.7% | -39.8 … -28.2 | 138/138 |
| AV1 | openh264 | -52.3% | -54.9% | -52.9% | -64.0 … -42.1 | 137/138 |
| AV1 | SVT-AV1 M8 | -7.4% | -12.5% | -13.9% | -22.3 … -6.5 | 138/138 |
| AV1 | libaom rt 8 | -26.2% | -33.6% | -27.3% | -39.7 … -22.1 | 138/138 |
| AV1 Fast | SVT-AV1 M12 | -24.0% | -29.2% | -24.2% | -35.9 … -24.0 | 138/138 |
| AV1 Fast | openh264 | -49.6% | -51.6% | -50.0% | -62.1 … -38.3 | 138/138 |
| AV1 Fast | AV1 | +5.6% | +7.2% | +6.0% | +3.4 … +9.4 | 138/138 |
| AV1 Faster | SVT-AV1 M12 | -22.1% | -26.7% | -22.9% | -36.1 … -20.5 | 138/138 |
| AV1 Faster | AV1 Fast | +2.2% | +3.2% | +1.7% | -1.0 … +5.2 | 138/138 |
| AV1 Veryfast | SVT-AV1 M12 | -15.4% | -17.4% | -15.0% | -30.9 … -11.4 | 138/138 |
| AV1 Veryfast | SVT-AV1 M13 | -18.4% | -20.7% | -17.5% | -31.9 … -13.7 | 138/138 |
| AV1 Veryfast | AV1 Faster | +10.5% | +14.7% | +12.3% | +4.1 … +18.4 | 138/138 |
| VP9 Slower | HEVC Slower | +3.9% | -0.5% | +7.3% | -0.7 … +6.4 | 138/138 |
| VP9 Slow | HEVC Slow | +5.4% | +1.5% | +9.6% | +1.1 … +8.6 | 138/138 |
| VP9 | HEVC | +6.3% | +2.2% | +11.1% | +2.8 … +9.3 | 138/138 |
| VP9 | libvpx VP9 rt 7 | -46.0% | -50.6% | -44.4% | -55.6 … -40.9 | 138/138 |
| VP9 Fast | HEVC Fast | +5.2% | +1.1% | +8.5% | +1.2 … +11.3 | 138/138 |
| VP9 Fast | libvpx VP9 rt 9 | -41.9% | -48.7% | -39.2% | -50.4 … -36.9 | 138/138 |
| VP9 Fast | VP9 | +10.6% | +10.8% | +13.2% | +7.2 … +14.6 | 138/138 |
| VP9 Faster | HEVC Faster | -3.5% | -3.8% | -0.5% | -5.9 … +0.9 | 138/138 |
| VP9 Faster | libvpx VP9 rt 9 | -40.1% | -45.1% | -37.7% | -48.9 … -33.5 | 138/138 |
| VP9 Faster | VP9 Fast | +3.0% | +6.8% | +2.3% | +0.3 … +7.6 | 138/138 |
| VP9 Veryfast | HEVC Faster | +10.9% | +9.8% | +10.7% | +5.9 … +15.6 | 138/138 |
| VP9 Veryfast | libvpx VP9 rt 9 | -31.4% | -37.9% | -30.6% | -43.0 … -26.3 | 138/138 |
| VP9 Veryfast | VP9 Faster | +15.8% | +14.5% | +12.2% | +8.4 … +19.3 | 138/138 |
| AV2 Slower | AV1 Slower | -9.6% | -10.4% | -12.5% | -11.4 … -5.7 | 138/138 |
| AV2 Slow | AV1 Slow | -6.5% | -6.8% | -8.9% | -8.6 … +0.1 | 138/138 |
| AV2 | AV1 | -8.8% | -8.1% | -11.6% | -10.8 … -5.5 | 138/138 |
| AV2 | SVT-AV1 M8 | -15.5% | -19.3% | -24.0% | -28.2 … -14.0 | 138/138 |
| AV2 Fast | AV1 Fast | -9.9% | -8.6% | -13.0% | -13.3 … -4.6 | 138/138 |
| AV2 Fast | AV2 | +4.6% | +7.0% | +4.4% | +2.4 … +8.6 | 138/138 |
| AV2 Faster | AV1 Faster | -5.6% | -3.8% | -4.5% | -7.5 … +0.6 | 138/138 |
| AV2 Faster | SVT-AV1 M12 | -27.2% | -29.8% | -27.3% | -36.2 … -24.0 | 138/138 |
| AV2 Faster | AV2 Fast | +7.5% | +9.0% | +11.8% | +5.5 … +11.1 | 138/138 |
| VVC Veryslow | HEVC Veryslow | -7.2% | -7.1% | -8.1% | -9.8 … -4.1 | 138/138 |
| VVC Slower | HEVC Slower | -10.3% | -11.1% | -10.7% | -13.0 … -8.7 | 138/138 |
| VVC Slow | HEVC Slow | -9.1% | -8.7% | -9.3% | -11.8 … -6.6 | 138/138 |
| VVC | HEVC | -9.0% | -9.1% | -9.0% | -12.4 … -7.1 | 138/138 |
| VVC | x265 medium | -36.9% | -39.9% | -42.4% | -45.1 … -30.3 | 138/138 |
| VVC Fast | HEVC Fast | -11.5% | -11.8% | -13.8% | -13.2 … -8.1 | 138/138 |
| VVC Fast | VVC | +9.5% | +9.8% | +10.7% | +7.3 … +11.7 | 138/138 |
| VVC Faster | HEVC Faster | -18.0% | -19.0% | -19.1% | -19.9 … -13.3 | 138/138 |
| VVC Faster | VVC Fast | +4.7% | +4.0% | +5.2% | +3.6 … +6.5 | 138/138 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264, H.264 Veryfast timed on 4fc9a1de (same streams), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): openh264 AQ on (-0.0 [-0.0, -0.0]%); x264 veryfast --tune psnr (-1.6 [-2.7, -0.6]%); x264 medium --tune ssim (-2.2 [-3.0, -1.4]%); x265 ultrafast --tune psnr (-0.9 [-1.3, -0.6]%); x265 superfast --tune psnr (-1.9 [-2.3, -1.6]%); x265 medium --tune psnr (-5.4 [-6.0, -4.8]%); libvpx VP9 rt 7 --tune ssim (-0.7 [-1.2, -0.2]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
