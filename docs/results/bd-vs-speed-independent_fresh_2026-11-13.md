# Bitrate vs encode speed, held-out clips (independent_fresh_2026-11-13)

<sub>Updated 2026-11-21, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264, H.264 Veryfast timed on 4fc9a1de (same streams)</sub>

Quality on the 138 clips of `independent_fresh_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_fresh_2026-11-13.svg)

**Rate control and the buffer.** Every encoder in the CBR charts is asked for a 1 s buffer at the target rate. x264, x265 and NVENC are held to it by a hard VBV, and SVT-AV1, libaom and libvpx by their 1000 ms real-time buffer. nanocodec 0.6.0 treats its buffer as a target with a 1.4x stop and sizes its key frames more generously: in these charts its buffer peak passes one buffer on 10-16 % of points (theirs: 0-1 %). openh264 is given no buffer setting and behaves like nanocodec in this respect. That is worth quality. On a sample of the held-out clips (24 clips at 150 frames, 12 at 600 frames; AV1, HEVC and H.264 medium), holding nanocodec's stream inside one buffer, as the references' is, cost it 3-5.5 % in bits at equal quality: its lead over SVT-AV1 M12 at 150 frames moved from 31 % to 28 % (AV1) and from 33 % to 29 % (HEVC), H.264 medium moved from level with SVT-AV1 M12 to about 5 % behind, and the leads over x264 medium moved by 0.6-2.2 points. The Fixed QP charts have no rate control and are not affected. A later release will make the buffer a hard bound by default and re-measure these charts.

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_fresh_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Veryslow** | -60.0% | -61.7% | -63.8% | -65.6 … -52.9 | 7.6 | 0.29× | 30.8 | 0.27× | 136/136 | Pareto; Pareto x86 |
| **VVC Slower** | -59.0% | -60.7% | -62.9% | -65.1 … -51.7 | 13.6 | 0.51× | 51.9 | 0.46× | 136/136 | Pareto; Pareto x86 |
| **VVC Slow** | -57.8% | -59.2% | -61.9% | -64.6 … -50.3 | 18.0 | 0.68× | 73.0 | 0.65× | 136/136 | Pareto; Pareto x86 |
| **HEVC Veryslow** | -57.5% | -59.1% | -61.0% | -62.7 … -50.4 | 10.2 | 0.38× | 36.9 | 0.33× | 136/136 |  |
| **AV2 Slower** | -57.1% | -61.0% | -60.4% | -63.2 … -50.5 | 7.9 | 0.30× | 26.6 | 0.24× | 136/136 | 1 rate-invalid pts |
| **VVC** | -56.1% | -57.7% | -60.5% | -63.0 … -48.5 | 25.0 | 0.94× | 101.7 | 0.90× | 136/136 | Pareto; Pareto x86 |
| **HEVC Slower** | -55.1% | -56.5% | -59.0% | -61.7 … -45.7 | 17.3 | 0.65× | 68.2 | 0.60× | 136/136 |  |
| **AV2 Slow** | -54.6% | -58.8% | -58.3% | -61.7 … -47.2 | 27.5 | 1.03× | 120.8 | 1.07× | 136/136 | Pareto; Pareto x86; 1 rate-invalid pts |
| **HEVC Slow** | -54.3% | -55.9% | -58.6% | -60.9 … -45.2 | 33.2 | 1.25× | 121.4 | 1.08× | 136/136 | Pareto; Pareto x86 |
| **AV2** | -53.5% | -57.6% | -57.6% | -60.4 … -46.6 | 35.4 | 1.33× | 150.7 | 1.34× | 136/136 | Pareto; Pareto x86; 1 rate-invalid pts |
| **AV1 Slower** | -53.0% | -56.6% | -55.2% | -59.4 … -47.3 | 14.5 | 0.54× | 59.5 | 0.53× | 136/136 |  |
| **VP9 Slower** | -52.9% | -55.6% | -55.9% | -58.8 … -45.3 | 13.0 | 0.49× | 51.5 | 0.46× | 136/136 | 3 rate-invalid pts |
| **HEVC** | -52.5% | -54.1% | -57.2% | -59.7 … -43.4 | 45.2 | 1.69× | 167.9 | 1.49× | 136/136 | Pareto; Pareto x86 |
| **VVC Fast** | -52.4% | -54.1% | -56.8% | -60.5 … -44.0 | 44.8 | 1.68× | 177.3 | 1.57× | 136/136 | Pareto x86; 1 rate-invalid pts |
| **VP9 Slow** | -51.9% | -54.8% | -54.9% | -58.3 … -45.0 | 21.9 | 0.82× | 91.8 | 0.81× | 136/136 | 2 rate-invalid pts |
| **AV2 Fast** | -51.7% | -55.0% | -55.9% | -59.5 … -43.9 | 42.7 | 1.60× | 172.7 | 1.53× | 136/136 | 2 rate-invalid pts |
| **AV1 Slow** | -51.6% | -55.7% | -54.5% | -59.5 … -46.9 | 28.5 | 1.07× | 115.3 | 1.02× | 136/136 | 2 rate-invalid pts |
| **VVC Faster** | -50.5% | -52.5% | -54.9% | -59.0 … -41.4 | 59.4 | 2.23× | 219.8 | 1.95× | 136/136 | Pareto; Pareto x86; 1 rate-invalid pts |
| **VP9** | -49.4% | -52.6% | -52.6% | -57.0 … -41.1 | 36.6 | 1.37× | 145.6 | 1.29× | 136/136 | 5 rate-invalid pts |
| **AV1** | -49.4% | -53.8% | -52.5% | -57.2 … -42.0 | 45.2 | 1.70× | 179.0 | 1.59× | 136/136 | 2 rate-invalid pts |
| **AV2 Faster** | -48.7% | -51.5% | -51.4% | -57.5 … -39.4 | 61.9 | 2.32× | 265.7 | 2.36× | 136/136 | Pareto; Pareto x86; 2 rate-invalid pts |
| **HEVC Fast** | -47.6% | -49.2% | -51.1% | -55.4 … -38.9 | 72.0 | 2.70× | 281.8 | 2.50× | 136/136 | Pareto; Pareto x86 |
| **AV1 Fast** | -47.3% | -51.2% | -50.3% | -55.8 … -39.5 | 64.7 | 2.42× | 252.9 | 2.24× | 136/136 |  |
| **AV1 Faster** | -46.3% | -49.8% | -49.4% | -55.3 … -38.6 | 77.1 | 2.89× | 313.5 | 2.78× | 136/136 | Pareto; Pareto x86 |
| **VP9 Fast** | -45.2% | -48.4% | -47.2% | -52.8 … -36.3 | 59.8 | 2.24× | 247.1 | 2.19× | 136/136 | 2 rate-invalid pts |
| **VP9 Faster** | -43.9% | -45.6% | -46.0% | -53.9 … -33.2 | 79.2 | 2.97× | 311.9 | 2.77× | 136/136 | Pareto |
| SVT-AV1 M8 | -43.5% | -46.4% | -43.0% | -49.6 … -32.4 | 14.3 | 0.53× | 80.8 | 0.72× | 136/136 | 35 rate-invalid pts |
| **HEVC Faster** | -42.3% | -43.7% | -46.1% | -51.4 … -32.4 | 90.0 | 3.37× | 350.2 | 3.11× | 136/136 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -41.7% | -43.9% | -43.9% | -53.0 … -31.1 | 95.2 | 3.57× | 419.5 | 3.72× | 136/136 | Pareto; Pareto x86 |
| **VP9 Veryfast** | -37.6% | -39.4% | -41.4% | -47.0 … -27.2 | 100.7 | 3.77× | 393.8 | 3.49× | 136/136 | Pareto; 1 rate-invalid pts |
| SVT-AV1 M10 | -37.1% | -39.0% | -38.7% | -44.0 … -25.6 | 39.3 | 1.47× | 228.9 | 2.03× | 136/136 | 33 rate-invalid pts |
| **H.264 Slower** | -36.0% | -40.0% | -41.3% | -44.5 … -26.8 | 37.4 | 1.40× | 113.3 | 1.00× | 136/136 |  |
| **H.264 Slow** | -35.8% | -39.6% | -41.2% | -45.3 … -26.3 | 51.9 | 1.95× | 160.7 | 1.43× | 136/136 |  |
| x265 medium, metric-tuned (--tune psnr) | -34.8% | -36.1% | -37.9% | -43.8 … -27.0 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 41 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| **H.264** | -34.4% | -37.9% | -40.2% | -43.9 … -24.6 | 66.2 | 2.48× | 198.4 | 1.76× | 136/136 |  |
| x265 superfast, metric-tuned (--tune psnr) | -33.1% | -33.6% | -35.9% | -41.0 … -24.5 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 37 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 superfast | -31.9% | -31.1% | -35.3% | -39.7 … -22.3 | 12.1 | 0.45× | 57.9 | 0.51× | 136/136 | 38 rate-invalid pts |
| x265 medium | -31.6% | -30.8% | -33.1% | -39.7 … -22.9 | 8.0 | 0.30× | 38.0 | 0.34× | 136/136 | 43 rate-invalid pts |
| SVT-AV1 M12 | -30.7% | -31.5% | -34.3% | -36.7 … -17.2 | 70.9 | 2.67× | 408.7 | 3.62× | 136/136 | 29 rate-invalid pts |
| **H.264 Fast** | -29.8% | -32.5% | -36.4% | -40.2 … -19.8 | 96.9 | 3.63× | 283.4 | 2.51× | 136/136 |  |
| SVT-AV1 M13 | -28.8% | -29.3% | -32.8% | -36.1 … -15.3 | 79.2 | 2.96× | 445.4 | 3.95× | 136/136 | Pareto x86; 29 rate-invalid pts |
| libaom rt 10 | -27.9% | -28.3% | -30.5% | -32.9 … -18.3 | 22.3 | 0.83× | 90.5 | 0.80× | 136/136 | 108 rate-invalid pts |
| libaom rt 8 | -27.8% | -25.9% | -31.7% | -33.8 … -18.4 | 17.4 | 0.65× | 68.8 | 0.61× | 136/136 | 106 rate-invalid pts |
| **H.264 Faster** | -27.5% | -30.7% | -33.4% | -37.7 … -15.3 | 118.3 | 4.44× | 369.8 | 3.28× | 136/136 | Pareto |
| x265 ultrafast, metric-tuned (--tune psnr) | -27.2% | -26.7% | -30.6% | -37.7 … -15.0 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 37 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 ultrafast | -26.6% | -24.8% | -30.5% | -36.3 … -13.9 | 17.1 | 0.64× | 85.1 | 0.75× | 136/136 | 37 rate-invalid pts |
| **H.264 Veryfast** | -18.0% | -22.4% | -23.3% | -31.8 … -1.9 | 161.9 | 6.07× | 510.6 | 4.53× | 136/136 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -9.9% | -8.9% | -19.5% | -12.6 … -6.1 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 58 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast, metric-tuned (--tune psnr) | -4.9% | -7.3% | -15.9% | -13.8 … +6.9 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast | -3.6% | -6.9% | -2.8% | -12.0 … +6.9 | 85.4 | 3.18× | 363.3 | 3.22× | 136/136 | 1 rate-invalid pts |
| libvpx VP9 rt 9 | -2.1% | +4.5% | -9.6% | -9.1 … +7.9 | 47.8 | 1.78× | 204.4 | 1.81× | 136/136 | 175 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 136/136 | 54 rate-invalid pts |
| libvpx VP9 rt 7 | +1.1% | +6.3% | -6.7% | -6.9 … +13.9 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 170 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.3% | +6.9% | -1.5% | -6.5 … +10.9 | 30.0 | 1.12× | 122.7 | 1.09× | 136/136 | 150 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264, adaptive quantisation on | +13.8% | +9.9% | +9.5% | -16.2 … +43.5 | 111.5 | 4.17× | 524.7 | 4.65× | 136/136 | 268 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264 | +13.8% | +9.9% | +9.5% | -16.2 … +43.5 | 111.5 | 4.17× | 524.7 | 4.65× | 136/136 | Pareto x86; 268 rate-invalid pts |
| x264 ultrafast | +26.1% | +35.4% | +3.9% | +7.4 … +47.1 | 190.5 | 7.11× | 1017.0 | 9.02× | 136/136 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d77344_9fe25620, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bdd100k_mot_bdd_00d9e313_7d75bb18, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video283, ind_bwc_videotext_video289, ind_bwc_videotext_video295, ind_bwc_videotext_video317, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video47, ind_bwc_videotext_video58, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_chirla_chirla_seq_026_camera_3_2023_12_05_12_12_13, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00127, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00187, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00212, ind_personpath22_uid_vid_00214, ind_personpath22_uid_vid_00216, ind_personpath22_uid_vid_00231, ind_uvg_vcm_uvgvcm_highwayview, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 2; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 12, crowd 39, lowlight 8, pan 42, static 35 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slower | -36.0% | -36.2% |
| H.264 Slow | -35.8% | -36.0% |
| H.264 | -34.4% | -34.6% |
| H.264 Fast | -29.8% | -29.7% |
| H.264 Faster | -27.5% | -27.2% |
| H.264 Veryfast | -18.0% | -16.8% |
| HEVC Veryslow | -57.5% | -56.9% |
| HEVC Slower | -55.1% | -54.2% |
| HEVC Slow | -54.3% | -53.7% |
| HEVC | -52.5% | -51.7% |
| HEVC Fast | -47.6% | -47.3% |
| HEVC Faster | -42.3% | -42.1% |
| AV1 Slower | -53.0% | -54.1% |
| AV1 Slow | -51.6% | -53.0% |
| AV1 | -49.4% | -50.5% |
| AV1 Fast | -47.3% | -47.9% |
| AV1 Faster | -46.3% | -46.9% |
| AV1 Veryfast | -41.7% | -41.7% |
| VP9 Slower | -52.9% | -52.8% |
| VP9 Slow | -51.9% | -51.7% |
| VP9 | -49.4% | -49.2% |
| VP9 Fast | -45.2% | -44.5% |
| VP9 Faster | -43.9% | -43.1% |
| VP9 Veryfast | -37.6% | -37.3% |
| AV2 Slower | -57.1% | -57.3% |
| AV2 Slow | -54.6% | -54.6% |
| AV2 | -53.5% | -53.8% |
| AV2 Fast | -51.7% | -51.6% |
| AV2 Faster | -48.7% | -48.3% |
| VVC Veryslow | -60.0% | -59.4% |
| VVC Slower | -59.0% | -58.4% |
| VVC Slow | -57.8% | -57.3% |
| VVC | -56.1% | -55.7% |
| VVC Fast | -52.4% | -52.0% |
| VVC Faster | -50.5% | -49.9% |
| openh264 | +13.8% | +14.1% |
| x264 ultrafast | +26.1% | +32.0% |
| x264 veryfast | -3.6% | -1.9% |
| x265 ultrafast | -26.6% | -25.0% |
| x265 superfast | -31.9% | -30.6% |
| x265 medium | -31.6% | -31.2% |
| SVT-AV1 M8 | -43.5% | -41.4% |
| SVT-AV1 M10 | -37.1% | -35.3% |
| SVT-AV1 M12 | -30.7% | -28.4% |
| SVT-AV1 M13 | -28.8% | -26.8% |
| libvpx VP9 rt 7 | +1.1% | +4.5% |
| libvpx VP9 rt 9 | -2.1% | -0.3% |
| libaom rt 8 | -27.8% | -25.7% |
| libaom rt 10 | -27.9% | -26.1% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +13.8% | +9.9% | +9.5% | +21.9% | 136/136 |  |
| openh264, adaptive quantisation on ● | +13.8% | +9.9% | +9.5% | +21.9% | 136/136 | -0.0 [-0.0, -0.0] better |
| x264 ultrafast | +26.1% | +35.4% | +3.9% | +39.1% | 136/136 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +26.1% | +35.1% | +4.0% | +39.2% | 136/136 | +0.0 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +23.1% | +31.4% | +17.9% | +20.0% | 136/136 | -0.1 [-1.4, +1.1] not distinguishable |
| x264 veryfast | -3.6% | -6.9% | -2.8% | -1.0% | 136/136 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | -4.9% | -7.3% | -15.9% | +8.4% | 136/136 | -1.5 [-2.5, -0.5] better |
| x264 veryfast, metric-tuned (--tune ssim) | -2.7% | -5.5% | -0.7% | -2.0% | 136/136 | +0.6 [-0.0, +1.1] not distinguishable |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 136/136 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -9.9% | -8.9% | -19.5% | -1.3% | 136/136 | -9.9 [-10.8, -9.0] better |
| x264 medium, metric-tuned (--tune ssim) | -2.1% | -1.9% | -0.4% | -3.9% | 136/136 | -2.1 [-2.9, -1.3] better |
| x265 ultrafast | -26.6% | -24.8% | -30.5% | -24.4% | 136/136 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -27.2% | -26.7% | -30.6% | -24.2% | 136/136 | -0.9 [-1.2, -0.6] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -27.2% | -26.7% | -30.6% | -24.2% | 136/136 | -0.9 [-1.2, -0.6] better |
| x265 superfast | -31.9% | -31.1% | -35.3% | -29.4% | 136/136 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -33.1% | -33.6% | -35.9% | -29.8% | 136/136 | -1.9 [-2.2, -1.5] better |
| x265 superfast, metric-tuned (--tune ssim) | -33.1% | -33.6% | -35.9% | -29.8% | 136/136 | -1.9 [-2.2, -1.5] better |
| x265 medium | -31.6% | -30.8% | -33.1% | -31.0% | 136/136 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -34.8% | -36.1% | -37.9% | -30.5% | 136/136 | -5.3 [-5.9, -4.6] better |
| x265 medium, metric-tuned (--tune ssim) | -32.8% | -33.0% | -33.7% | -31.5% | 136/136 | -1.8 [-2.1, -1.5] better |
| libvpx VP9 rt 7 | +1.1% | +6.3% | -6.7% | +3.7% | 136/136 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +1.3% | +6.9% | -1.5% | -1.3% | 136/136 | -0.8 [-1.3, -0.4] better |
| libvpx VP9 rt 9 | -2.1% | +4.5% | -9.6% | -1.3% | 136/136 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -2.0% | +5.1% | -8.7% | -2.3% | 136/136 | -0.1 [-0.6, +0.4] not distinguishable |
| libaom rt 8 | -27.8% | -25.9% | -31.7% | -25.7% | 136/136 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -27.8% | -24.8% | -27.4% | -31.1% | 136/136 | +0.7 [+0.1, +1.4] worse |
| libaom rt 10 | -27.9% | -28.3% | -30.5% | -24.9% | 136/136 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -28.0% | -26.9% | -27.1% | -29.9% | 136/136 | +0.6 [-0.0, +1.2] not distinguishable |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Veryslow** | -54.8% | -55.0% | -53.8% | -55.4% | 135/136 |
| **VVC Slower** | -53.5% | -53.8% | -52.6% | -54.1% | 136/136 |
| **VVC Slow** | -52.0% | -51.9% | -51.2% | -53.0% | 136/136 |
| **HEVC Veryslow** | -51.9% | -52.3% | -50.1% | -53.0% | 135/136 |
| **AV2 Slower** | -51.7% | -54.4% | -49.9% | -50.6% | 135/136 |
| **VVC** | -50.1% | -50.3% | -49.4% | -50.7% | 136/136 |
| **HEVC Slower** | -49.2% | -49.3% | -47.6% | -50.6% | 136/136 |
| **AV2 Slow** | -48.7% | -51.8% | -47.0% | -47.3% | 136/136 |
| **HEVC Slow** | -48.2% | -48.5% | -47.0% | -49.2% | 136/136 |
| **AV2** | -47.5% | -50.5% | -46.2% | -45.9% | 136/136 |
| **AV1 Slower** | -47.0% | -50.0% | -43.3% | -47.8% | 136/136 |
| **VP9 Slower** | -47.0% | -48.9% | -44.0% | -47.9% | 136/136 |
| **HEVC** | -46.1% | -46.3% | -45.1% | -46.8% | 136/136 |
| **VVC Fast** | -45.9% | -45.9% | -44.6% | -47.0% | 136/136 |
| **VP9 Slow** | -45.7% | -47.7% | -42.5% | -46.7% | 136/136 |
| **AV1 Slow** | -45.5% | -48.8% | -42.3% | -45.3% | 136/136 |
| **AV2 Fast** | -45.4% | -47.5% | -44.0% | -44.8% | 136/136 |
| **VVC Faster** | -43.7% | -44.1% | -42.1% | -44.9% | 136/136 |
| **AV1** | -43.0% | -46.7% | -39.8% | -42.4% | 136/136 |
| **VP9** | -42.8% | -45.1% | -39.6% | -43.8% | 136/136 |
| **AV2 Faster** | -41.8% | -43.4% | -38.1% | -43.8% | 136/136 |
| **AV1 Fast** | -40.5% | -43.8% | -36.9% | -41.0% | 136/136 |
| **HEVC Fast** | -40.5% | -40.8% | -37.3% | -43.2% | 136/136 |
| **AV1 Faster** | -39.3% | -42.0% | -35.7% | -40.1% | 136/136 |
| **VP9 Fast** | -37.8% | -40.3% | -32.7% | -40.4% | 136/136 |
| **VP9 Faster** | -36.3% | -37.2% | -31.1% | -40.7% | 136/136 |
| SVT-AV1 M8 | -35.5% | -37.3% | -27.5% | -41.8% | 136/136 |
| **HEVC Faster** | -34.4% | -34.8% | -31.0% | -37.3% | 136/136 |
| **AV1 Veryfast** | -34.1% | -35.5% | -28.9% | -37.8% | 136/136 |
| **VP9 Veryfast** | -29.3% | -30.4% | -25.2% | -32.2% | 136/136 |
| SVT-AV1 M10 | -28.2% | -28.6% | -21.7% | -34.2% | 136/136 |
| **H.264 Slower** | -27.7% | -31.3% | -25.2% | -26.5% | 136/136 |
| **H.264 Slow** | -27.4% | -30.7% | -24.9% | -26.6% | 136/136 |
| x265 medium, metric-tuned (--tune psnr) | -26.6% | -27.3% | -21.5% | -30.9% | 136/136 |
| **H.264** | -25.8% | -28.8% | -23.7% | -24.8% | 136/136 |
| x265 superfast, metric-tuned (--tune psnr) | -24.2% | -23.7% | -18.8% | -30.2% | 136/136 |
| x265 superfast, metric-tuned (--tune ssim) | -24.2% | -23.7% | -18.8% | -30.2% | 136/136 |
| x265 medium, metric-tuned (--tune ssim) | -23.8% | -23.5% | -16.0% | -31.9% | 136/136 |
| x265 superfast | -22.8% | -20.8% | -17.9% | -29.8% | 136/136 |
| x265 medium | -22.5% | -20.9% | -15.1% | -31.4% | 136/136 |
| SVT-AV1 M12 | -21.2% | -20.7% | -16.5% | -26.4% | 136/136 |
| **H.264 Fast** | -20.8% | -23.0% | -19.1% | -20.3% | 136/136 |
| SVT-AV1 M13 | -19.2% | -18.4% | -14.7% | -24.4% | 136/136 |
| **H.264 Faster** | -18.2% | -21.1% | -15.5% | -17.9% | 136/136 |
| x265 ultrafast, metric-tuned (--tune psnr) | -17.4% | -16.2% | -11.9% | -24.2% | 136/136 |
| x265 ultrafast, metric-tuned (--tune ssim) | -17.4% | -16.2% | -11.9% | -24.2% | 136/136 |
| libaom rt 8 | -17.3% | -15.1% | -12.1% | -25.3% | 135/136 |
| libaom rt 10, metric-tuned (--tune ssim) | -17.2% | -15.8% | -6.9% | -29.7% | 135/136 |
| libaom rt 10 | -17.2% | -17.1% | -10.5% | -24.6% | 135/136 |
| libaom rt 8, metric-tuned (--tune ssim) | -17.1% | -13.9% | -7.4% | -30.7% | 135/136 |
| x265 ultrafast | -16.8% | -14.1% | -11.7% | -24.5% | 136/136 |
| **H.264 Veryfast** | -7.7% | -12.5% | -3.2% | -7.3% | 136/136 |
| x264 veryfast, metric-tuned (--tune psnr) | +6.0% | +2.3% | +5.9% | +9.9% | 136/136 |
| x264 veryfast | +9.0% | +2.9% | +23.9% | +0.3% | 136/136 |
| x264 veryfast, metric-tuned (--tune ssim) | +10.0% | +4.4% | +26.5% | -0.9% | 136/136 |
| x264 medium, metric-tuned (--tune ssim) | +10.4% | +7.8% | +25.9% | -2.6% | 136/136 |
| libvpx VP9 rt 9 | +11.5% | +19.0% | +16.3% | -0.7% | 136/136 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.7% | +19.5% | +17.2% | -1.7% | 136/136 |
| x264 medium | +12.8% | +10.5% | +26.2% | +1.7% | 136/136 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.3% | +19.7% | +23.4% | -0.2% | 136/136 |
| libvpx VP9 rt 7 | +14.4% | +19.7% | +18.6% | +4.8% | 136/136 |
| openh264, adaptive quantisation on | +30.1% | +26.7% | +40.2% | +23.3% | 136/136 |
| openh264 | +30.1% | +26.7% | +40.2% | +23.3% | 136/136 |
| x264 ultrafast, metric-tuned (--tune ssim) | +39.1% | +43.6% | +51.8% | +22.0% | 136/136 |
| x264 ultrafast, metric-tuned (--tune psnr) | +40.3% | +46.9% | +32.3% | +41.8% | 136/136 |
| x264 ultrafast | +40.4% | +47.1% | +32.2% | +41.8% | 136/136 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.
- **ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508** (static): no reliable overlap for AV2 Slower, AV2 Slow, AV2.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | ind_bwc_videotext_video88 | ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508 |
|---|---:|---:|
| VVC Veryslow | -68.9% | -68.0% |
| VVC Slower | -68.9% | -69.0% |
| VVC Slow | -68.9% | -64.7% |
| HEVC Veryslow | -63.0% | -68.2% |
| AV2 Slower | -66.6% | n/a |
| VVC | -66.5% | -67.8% |
| HEVC Slower | -60.8% | -67.3% |
| AV2 Slow | -66.8% | n/a |
| HEVC Slow | -59.7% | -65.6% |
| AV2 | -65.5% | n/a |
| AV1 Slower | -57.9% | -64.9% |
| VP9 Slower | -61.8% | -62.8% |
| HEVC | -59.4% | -67.5% |
| VVC Fast | -63.4% | -65.8% |
| VP9 Slow | -60.8% | -62.7% |
| AV2 Fast | -65.6% | -69.0% |
| AV1 Slow | -56.6% | -64.1% |
| VVC Faster | -63.4% | -64.7% |
| VP9 | -58.2% | -63.4% |
| AV1 | -55.8% | -66.3% |
| AV2 Faster | -64.9% | -65.6% |
| HEVC Fast | -55.8% | -62.7% |
| AV1 Fast | -57.8% | -64.0% |
| AV1 Faster | -57.8% | -64.4% |
| VP9 Fast | -58.6% | -61.7% |
| VP9 Faster | -59.0% | -60.6% |
| SVT-AV1 M8 | -60.8% | -55.6% |
| HEVC Faster | -52.8% | -59.5% |
| AV1 Veryfast | -57.6% | -61.9% |
| VP9 Veryfast | -51.1% | -56.8% |
| SVT-AV1 M10 | -56.2% | -52.5% |
| H.264 Slower | -24.1% | -61.9% |
| H.264 Slow | -26.1% | -61.3% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a |
| H.264 | -22.1% | -60.6% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 superfast | -47.1% | -42.3% |
| x265 medium | -35.4% | -40.3% |
| SVT-AV1 M12 | -50.6% | -53.7% |
| H.264 Fast | -18.6% | -60.3% |
| SVT-AV1 M13 | -49.0% | -52.3% |
| libaom rt 10 | -38.6% | -32.2% |
| libaom rt 8 | n/a | -22.4% |
| H.264 Faster | -22.4% | -58.4% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a |
| x265 ultrafast | -45.2% | -41.7% |
| H.264 Veryfast | -15.3% | -54.7% |
| x264 medium, metric-tuned (--tune psnr) | n/a | n/a |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a |
| x264 veryfast | -3.5% | -22.7% |
| libvpx VP9 rt 9 | -19.2% | -13.7% |
| libvpx VP9 rt 7 | -13.6% | +24.9% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a |
| openh264, adaptive quantisation on | n/a | n/a |
| openh264 | +32.2% | -66.5% |
| x264 ultrafast | +10.5% | -10.6% |

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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): openh264 AQ on (-0.0 [-0.0, -0.0]%); x264 veryfast --tune psnr (-1.5 [-2.5, -0.5]%); x264 medium --tune psnr (-9.9 [-10.8, -9.0]%); x265 ultrafast --tune psnr (-0.9 [-1.2, -0.6]%); x265 superfast --tune psnr (-1.9 [-2.2, -1.5]%); x265 medium --tune psnr (-5.3 [-5.9, -4.6]%); libvpx VP9 rt 7 --tune ssim (-0.8 [-1.3, -0.4]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
