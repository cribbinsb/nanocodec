# Bitrate vs encode speed, held-out clips (independent_2026-11-13)

<sub>Updated 2026-11-21, build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264 timed on 20d31869 (same streams)</sub>

Quality on the 209 clips of `independent_2026-11-13`. Speed is the main chart's: timed on its three timing clips (lowlight_city_night, crowd_mot17_09, bodycam_bc1; tools/dev/anchor_chart_timing.py), not on these clips.

![bitrate vs speed](bd-vs-speed-independent_2026-11-13.svg)

**Rate control and the buffer.** Every encoder in the CBR charts is asked for a 1 s buffer at the target rate. x264, x265 and NVENC are held to it by a hard VBV, and SVT-AV1, libaom and libvpx by their 1000 ms real-time buffer. nanocodec 0.6.0 treats its buffer as a target with a 1.4x stop and sizes its key frames more generously: in these charts its buffer peak passes one buffer on 10-16 % of points (theirs: 0-1 %). openh264 is given no buffer setting and behaves like nanocodec in this respect. That is worth quality. On a sample of the held-out clips (24 clips at 150 frames, 12 at 600 frames; AV1, HEVC and H.264 medium), holding nanocodec's stream inside one buffer, as the references' is, cost it 3-5.5 % in bits at equal quality: its lead over SVT-AV1 M12 at 150 frames moved from 31 % to 28 % (AV1) and from 33 % to 29 % (HEVC), H.264 medium moved from level with SVT-AV1 M12 to about 5 % behind, and the leads over x264 medium moved by 0.6-2.2 points. The Fixed QP charts have no rate control and are not affected. A later release will make the buffer a hard bound by default and re-measure these charts.

The same on x86 (one Ryzen 9 5950X core):

![bitrate vs x86 speed](bd-vs-speed-independent_2026-11-13-x86.svg)

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | fps (Jetson, 1 core) | speed vs x264 medium | fps (x86, 1 core) | x86 speed vs x264 medium | clips | notes |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---|
| **VVC Veryslow** | -57.2% | -58.9% | -59.9% | -64.6 … -51.8 | 7.6 | 0.29× | 30.8 | 0.27× | 206/206 | Pareto; Pareto x86 |
| **VVC Slower** | -56.3% | -58.1% | -59.0% | -64.0 … -50.3 | 13.6 | 0.51× | 51.9 | 0.46× | 206/206 | Pareto; Pareto x86 |
| **VVC Slow** | -55.2% | -56.8% | -58.1% | -62.9 … -49.2 | 18.0 | 0.68× | 73.0 | 0.65× | 206/206 | Pareto; Pareto x86 |
| **AV2 Slower** | -54.6% | -58.8% | -56.6% | -62.1 … -49.8 | 7.9 | 0.30× | 26.6 | 0.24× | 206/206 | 1 rate-invalid pts |
| **HEVC Veryslow** | -54.0% | -55.9% | -56.4% | -61.4 … -49.3 | 10.2 | 0.38× | 36.9 | 0.33× | 206/206 |  |
| **VVC** | -53.6% | -55.2% | -56.6% | -61.3 … -47.4 | 25.0 | 0.94× | 101.7 | 0.90× | 206/206 | Pareto; Pareto x86 |
| **AV2 Slow** | -51.8% | -56.3% | -54.1% | -60.5 … -46.5 | 27.5 | 1.03× | 120.8 | 1.07× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |
| **HEVC Slower** | -51.3% | -52.7% | -53.8% | -60.1 … -44.7 | 17.3 | 0.65× | 68.2 | 0.60× | 206/206 |  |
| **AV1 Slower** | -51.1% | -54.9% | -51.3% | -58.7 … -46.5 | 14.5 | 0.54× | 59.5 | 0.53× | 206/206 |  |
| **AV2** | -51.1% | -55.4% | -53.7% | -59.7 … -46.4 | 35.4 | 1.33× | 150.7 | 1.34× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |
| **HEVC Slow** | -50.8% | -52.5% | -54.1% | -59.8 … -44.6 | 33.2 | 1.25× | 121.4 | 1.08× | 206/206 |  |
| **VP9 Slower** | -50.5% | -53.4% | -52.3% | -58.3 … -44.3 | 13.0 | 0.49× | 51.5 | 0.46× | 206/206 | 3 rate-invalid pts |
| **AV1 Slow** | -50.1% | -54.1% | -51.0% | -57.8 … -44.7 | 28.5 | 1.07× | 115.3 | 1.02× | 206/206 | 2 rate-invalid pts |
| **VVC Fast** | -49.8% | -51.4% | -53.0% | -58.6 … -42.8 | 44.8 | 1.68× | 177.3 | 1.57× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |
| **VP9 Slow** | -49.4% | -52.3% | -51.4% | -57.4 … -44.1 | 21.9 | 0.82× | 91.8 | 0.81× | 206/206 | 3 rate-invalid pts |
| **HEVC** | -48.3% | -49.8% | -51.9% | -58.3 … -42.0 | 45.2 | 1.69× | 167.9 | 1.49× | 206/206 | Pareto |
| **AV2 Fast** | -48.2% | -51.8% | -50.7% | -58.2 … -43.2 | 42.7 | 1.60× | 172.7 | 1.53× | 206/206 | 2 rate-invalid pts |
| **VVC Faster** | -47.3% | -48.8% | -50.3% | -56.9 … -40.2 | 59.4 | 2.23× | 219.8 | 1.95× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |
| **AV1** | -47.1% | -51.3% | -48.2% | -56.0 … -41.5 | 45.2 | 1.70× | 179.0 | 1.59× | 206/206 | 2 rate-invalid pts |
| **VP9** | -46.6% | -49.6% | -48.6% | -55.1 … -39.8 | 36.6 | 1.37× | 145.6 | 1.29× | 206/206 | 6 rate-invalid pts |
| **AV1 Fast** | -44.6% | -48.1% | -45.7% | -53.8 … -38.8 | 64.7 | 2.42× | 252.9 | 2.24× | 206/206 | Pareto; Pareto x86 |
| **AV2 Faster** | -44.4% | -47.5% | -44.8% | -56.0 … -38.2 | 61.9 | 2.32× | 265.7 | 2.36× | 206/206 | Pareto x86; 2 rate-invalid pts |
| **HEVC Fast** | -44.0% | -45.5% | -45.7% | -53.8 … -37.3 | 72.0 | 2.70× | 281.8 | 2.50× | 206/206 | Pareto; Pareto x86 |
| **AV1 Faster** | -42.8% | -45.8% | -44.1% | -54.3 … -36.9 | 77.1 | 2.89× | 313.5 | 2.78× | 206/206 | Pareto; Pareto x86 |
| **VP9 Fast** | -42.1% | -45.0% | -43.0% | -50.6 … -34.6 | 59.8 | 2.24× | 247.1 | 2.19× | 206/206 | 2 rate-invalid pts |
| **VP9 Faster** | -40.4% | -41.9% | -41.3% | -51.7 … -32.6 | 79.2 | 2.97× | 311.9 | 2.77× | 206/206 | Pareto |
| **HEVC Faster** | -39.0% | -40.4% | -41.0% | -50.7 … -30.8 | 90.0 | 3.37× | 350.2 | 3.11× | 206/206 | Pareto; Pareto x86 |
| **AV1 Veryfast** | -37.8% | -38.8% | -38.3% | -50.3 … -29.1 | 95.2 | 3.57× | 419.5 | 3.72× | 206/206 | Pareto; Pareto x86 |
| SVT-AV1 M8 | -37.4% | -42.3% | -32.8% | -48.1 … -30.8 | 14.3 | 0.53× | 80.8 | 0.72× | 206/206 | 50 rate-invalid pts |
| **VP9 Veryfast** | -34.3% | -35.0% | -37.4% | -45.8 … -26.0 | 100.7 | 3.77× | 393.8 | 3.49× | 206/206 | Pareto; 1 rate-invalid pts |
| **H.264 Slower** | -33.0% | -36.8% | -36.5% | -44.3 … -24.6 | 37.4 | 1.40× | 113.3 | 1.00× | 206/206 |  |
| **H.264 Slow** | -32.8% | -36.3% | -36.4% | -44.4 … -24.4 | 51.9 | 1.95× | 160.7 | 1.43× | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) | -31.8% | -34.0% | -32.7% | -43.7 … -26.5 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 50 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M10 | -31.5% | -34.7% | -29.0% | -42.5 … -24.6 | 39.3 | 1.47× | 228.9 | 2.03× | 206/206 | 46 rate-invalid pts |
| **H.264** | -31.1% | -34.4% | -35.1% | -43.5 … -22.5 | 66.2 | 2.48× | 198.4 | 1.76× | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) | -29.3% | -30.6% | -30.0% | -40.7 … -23.8 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 45 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x265 medium | -28.7% | -28.6% | -27.5% | -39.6 … -22.8 | 8.0 | 0.30× | 38.0 | 0.34× | 206/206 | 52 rate-invalid pts |
| x265 superfast | -28.1% | -27.9% | -29.4% | -38.9 … -21.8 | 12.1 | 0.45× | 57.9 | 0.51× | 206/206 | 46 rate-invalid pts |
| **H.264 Fast** | -26.1% | -28.3% | -31.2% | -39.1 … -15.9 | 96.9 | 3.63× | 283.4 | 2.51× | 206/206 |  |
| libaom rt 8 | -24.4% | -23.2% | -26.1% | -32.9 … -18.1 | 17.4 | 0.65× | 68.8 | 0.61× | 206/206 | 160 rate-invalid pts |
| libaom rt 10 | -23.9% | -24.6% | -24.2% | -32.2 … -16.8 | 22.3 | 0.83× | 90.5 | 0.80× | 206/206 | 169 rate-invalid pts |
| **H.264 Faster** | -23.7% | -26.4% | -27.9% | -37.6 … -12.8 | 118.3 | 4.44× | 369.8 | 3.28× | 206/206 | Pareto |
| x265 ultrafast, metric-tuned (--tune psnr) | -23.2% | -23.8% | -24.3% | -37.2 … -15.4 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 45 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| SVT-AV1 M12 | -23.1% | -24.2% | -22.3% | -36.4 … -16.1 | 70.9 | 2.67× | 408.7 | 3.62× | 206/206 | 41 rate-invalid pts |
| x265 ultrafast | -22.9% | -22.0% | -24.4% | -36.0 … -15.0 | 17.1 | 0.64× | 85.1 | 0.75× | 206/206 | 45 rate-invalid pts |
| SVT-AV1 M13 | -21.4% | -21.9% | -20.7% | -33.6 … -14.6 | 79.2 | 2.96× | 445.4 | 3.95× | 206/206 | Pareto x86; 40 rate-invalid pts |
| **H.264 Veryfast** | -12.7% | -16.9% | -16.6% | -30.6 … +3.5 | 161.9 | 6.07× | 510.6 | 4.53× | 206/206 | Pareto; Pareto x86 |
| x264 medium, metric-tuned (--tune psnr) | -9.8% | -9.7% | -18.7% | -12.6 … -6.0 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 76 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | -1.3% | +5.5% | -5.8% | -8.9 … +6.6 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 261 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 9 | -0.3% | +5.7% | -5.4% | -8.8 … +8.5 | 47.8 | 1.78× | 204.4 | 1.81× | 206/206 | 261 rate-invalid pts |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 | 26.7 | 1.00× | 112.8 | 1.00× | 206/206 | 70 rate-invalid pts |
| x264 veryfast, metric-tuned (--tune psnr) | +0.2% | -1.8% | -9.7% | -12.4 … +10.1 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| x264 veryfast | +1.8% | -0.9% | +3.4% | -9.7 … +9.4 | 85.4 | 3.18× | 363.3 | 3.22× | 206/206 | 1 rate-invalid pts |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +1.8% | +7.0% | +1.5% | -6.2 … +11.1 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 226 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| libvpx VP9 rt 7 | +2.5% | +7.5% | -2.9% | -6.6 … +13.5 | 30.0 | 1.12× | 122.7 | 1.09× | 206/206 | 256 rate-invalid pts |
| openh264, adaptive quantisation on | +13.4% | +9.2% | +12.2% | -15.8 … +50.4 | 111.5 | 4.17× | 524.7 | 4.65× | 206/206 | 396 rate-invalid pts; fps of the default-tuned run; re-timing pending |
| openh264 | +13.4% | +9.2% | +12.2% | -15.8 … +50.4 | 111.5 | 4.17× | 524.7 | 4.65× | 206/206 | Pareto x86; 396 rate-invalid pts |
| x264 ultrafast | +35.3% | +47.9% | +12.5% | +10.6 … +55.6 | 190.5 | 7.11× | 1017.0 | 9.02× | 206/206 | Pareto; Pareto x86; 1 rate-invalid pts |

Low rungs (40 / 75 kbps, every encoder) on: pool_barscene_pristine, pool_bodycam_antare_antare_test, pool_bodycam_bwc_video118, pool_chad_1_093, pool_chad_3_086, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_crowd_person_uid_vid_00222, pool_crowd_person_uid_vid_00224, pool_ducks_take_off_pristine, pool_gmdcsa24_s1_adl05, pool_jaad_0340, pool_kristen_sara_360p, pool_lowlight_cevo_inof_md_indoor_night_ohcam_0, pool_meva_hospital_g301, pool_otw_homes_00015, pool_smoke_sauna_360p, pool_static_chirla_chirla_seq_024_camera_7_2023, ind_bdd100k_mot_bdd_00cea101_06f21d5e, ind_bdd100k_mot_bdd_00d0f034_6d666f7b, ind_bdd100k_mot_bdd_00d1bafa_1b47b41c, ind_bdd100k_mot_bdd_00d77344_9fe25620, ind_bdd100k_mot_bdd_00d9e313_038310a2, ind_bdd100k_mot_bdd_00d9e313_7d75bb18, ind_bwc_videotext_video219, ind_bwc_videotext_video28, ind_bwc_videotext_video283, ind_bwc_videotext_video289, ind_bwc_videotext_video295, ind_bwc_videotext_video317, ind_bwc_videotext_video321, ind_bwc_videotext_video33, ind_bwc_videotext_video345, ind_bwc_videotext_video347, ind_bwc_videotext_video354, ind_bwc_videotext_video47, ind_bwc_videotext_video58, ind_bwc_videotext_video61, ind_bwc_videotext_video88, ind_cctv_inof_ld_out_light_ffcam_002, ind_cevo_april25_inof_ld_in_light_ffcam_002, ind_cevo_april25_inof_ld_outld_light_ffcam_001, ind_cevo_april25_inof_md_out_light_ffcam_003, ind_chirla_chirla_seq_020_camera_2_2023_07_26_13_21_40, ind_chirla_chirla_seq_026_camera_3_2023_12_05_12_12_13, ind_jaad_video_0015, ind_jaad_video_0018, ind_jaad_video_0019, ind_jaad_video_0042, ind_jaad_video_0051, ind_jaad_video_0076, ind_jaad_video_0102, ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508, ind_meva_2018_03_07_17_05_00_17_10_00_bus_g508, ind_meva_2018_03_13_16_10_00_16_15_00_bus_g505, ind_mot_mot20_01, ind_mp4_waiting_area1, ind_otw_homes_00310, ind_otw_homes_00394, ind_personpath22_uid_vid_00019, ind_personpath22_uid_vid_00079, ind_personpath22_uid_vid_00106, ind_personpath22_uid_vid_00127, ind_personpath22_uid_vid_00142, ind_personpath22_uid_vid_00172, ind_personpath22_uid_vid_00173, ind_personpath22_uid_vid_00177, ind_personpath22_uid_vid_00187, ind_personpath22_uid_vid_00190, ind_personpath22_uid_vid_00195, ind_personpath22_uid_vid_00200, ind_personpath22_uid_vid_00208, ind_personpath22_uid_vid_00212, ind_personpath22_uid_vid_00214, ind_personpath22_uid_vid_00216, ind_personpath22_uid_vid_00231, ind_uvg_vcm_uvgvcm_highwayview, ind_uvg_vcm_uvgvcm_jobfair. Clips left out for every encoder: 3; chained (point, clip) pairs: 0.

## Class-weighted and clip-weighted BD-J

The chart's BD-J weights every content class equally (bodycam 22, crowd 57, drone 3, lowlight 15, pan 53, static 54, texture 2 clips); clip-weighted counts every clip equally. Same clips and rungs; the clip-weighted mean is over the directly compared clips (no chained rows).

| Point | BD-J class-weighted | BD-J clip-weighted |
|---|---:|---:|
| H.264 Slower | -33.0% | -33.3% |
| H.264 Slow | -32.8% | -33.0% |
| H.264 | -31.1% | -31.4% |
| H.264 Fast | -26.1% | -26.5% |
| H.264 Faster | -23.7% | -23.9% |
| H.264 Veryfast | -12.7% | -13.0% |
| HEVC Veryslow | -54.0% | -55.4% |
| HEVC Slower | -51.3% | -52.8% |
| HEVC Slow | -50.8% | -52.1% |
| HEVC | -48.3% | -50.0% |
| HEVC Fast | -44.0% | -45.5% |
| HEVC Faster | -39.0% | -40.4% |
| AV1 Slower | -51.1% | -53.0% |
| AV1 Slow | -50.1% | -51.8% |
| AV1 | -47.1% | -49.4% |
| AV1 Fast | -44.6% | -46.8% |
| AV1 Faster | -42.8% | -45.6% |
| AV1 Veryfast | -37.8% | -39.9% |
| VP9 Slower | -50.5% | -51.8% |
| VP9 Slow | -49.4% | -50.5% |
| VP9 | -46.6% | -48.0% |
| VP9 Fast | -42.1% | -43.1% |
| VP9 Faster | -40.4% | -41.7% |
| VP9 Veryfast | -34.3% | -35.6% |
| AV2 Slower | -54.6% | -56.4% |
| AV2 Slow | -51.8% | -53.6% |
| AV2 | -51.1% | -52.7% |
| AV2 Fast | -48.2% | -50.5% |
| AV2 Faster | -44.4% | -46.9% |
| VVC Veryslow | -57.2% | -58.4% |
| VVC Slower | -56.3% | -57.4% |
| VVC Slow | -55.2% | -56.2% |
| VVC | -53.6% | -54.5% |
| VVC Fast | -49.8% | -50.8% |
| VVC Faster | -47.3% | -48.5% |
| openh264 | +13.4% | +16.9% |
| x264 ultrafast | +35.3% | +36.8% |
| x264 veryfast | +1.8% | +1.2% |
| x265 ultrafast | -22.9% | -24.2% |
| x265 superfast | -28.1% | -29.9% |
| x265 medium | -28.7% | -30.6% |
| SVT-AV1 M8 | -37.4% | -40.3% |
| SVT-AV1 M10 | -31.5% | -34.1% |
| SVT-AV1 M12 | -23.1% | -27.1% |
| SVT-AV1 M13 | -21.4% | -25.2% |
| libvpx VP9 rt 7 | +2.5% | +4.0% |
| libvpx VP9 rt 9 | -0.3% | -0.1% |
| libaom rt 8 | -24.4% | -25.0% |
| libaom rt 10 | -23.9% | -25.1% |

## Metric-tuned references

Each reference at its charted setting and at each alternative its documentation provides (x264 / x265: `--tune psnr`, `--tune ssim`; libaom / libvpx: `--tune ssim`; openh264: adaptive quantisation on), every other setting unchanged; BD-rate vs the chart's anchor (x264 medium, default tuning) on the chart's clips and rungs, and each alternative's direct paired BD-J against the charted setting with its 95% bootstrap interval. The chart also plots, per reference, the alternative with the largest direct gain whose interval excludes zero (●); the other rows are not distinguishable from the charted setting, or worse. The two calculations can differ by a few tenths: the left columns compare each row with the x264 anchor, the last column the two settings directly on the same points. Equal rows: the setting changes nothing there (x264 ultrafast already runs without psy and AQ; x265 ultrafast and superfast give identical results at both tunes; openh264's harness codes the same bytes with adaptive quantisation on).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips | BD-J vs its charted setting (direct, 95% interval) |
|---|---:|---:|---:|---:|---:|---|
| openh264 | +13.4% | +9.2% | +12.2% | +18.8% | 206/206 |  |
| openh264, adaptive quantisation on ● | +13.4% | +9.2% | +12.2% | +18.8% | 206/206 | -0.0 [-0.0, -0.0] better |
| x264 ultrafast | +35.3% | +47.9% | +12.5% | +45.5% | 206/206 |  |
| x264 ultrafast, metric-tuned (--tune psnr) | +35.4% | +48.0% | +12.4% | +45.8% | 206/206 | +0.1 [-0.2, +0.3] not distinguishable |
| x264 ultrafast, metric-tuned (--tune ssim) | +32.0% | +44.9% | +25.8% | +25.4% | 206/206 | -0.4 [-2.0, +1.2] not distinguishable |
| x264 veryfast | +1.8% | -0.9% | +3.4% | +2.9% | 206/206 |  |
| x264 veryfast, metric-tuned (--tune psnr) ● | +0.2% | -1.8% | -9.7% | +12.3% | 206/206 | -1.5 [-2.7, -0.3] better |
| x264 veryfast, metric-tuned (--tune ssim) | +2.6% | +0.5% | +5.8% | +1.6% | 206/206 | +0.9 [+0.3, +1.5] worse |
| x264 medium | +0.0% | +0.0% | +0.0% | +0.0% | 206/206 |  |
| x264 medium, metric-tuned (--tune psnr) ● | -9.8% | -9.7% | -18.7% | -0.9% | 206/206 | -9.8 [-10.9, -8.7] better |
| x264 medium, metric-tuned (--tune ssim) | -2.4% | -2.8% | -0.0% | -4.4% | 206/206 | -2.4 [-3.1, -1.7] better |
| x265 ultrafast | -22.9% | -22.0% | -24.4% | -22.2% | 206/206 |  |
| x265 ultrafast, metric-tuned (--tune psnr) ● | -23.2% | -23.8% | -24.3% | -21.5% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 ultrafast, metric-tuned (--tune ssim) | -23.2% | -23.8% | -24.3% | -21.5% | 206/206 | -0.6 [-1.0, -0.2] better |
| x265 superfast | -28.1% | -27.9% | -29.4% | -26.9% | 206/206 |  |
| x265 superfast, metric-tuned (--tune psnr) ● | -29.3% | -30.6% | -30.0% | -27.2% | 206/206 | -1.9 [-2.3, -1.5] better |
| x265 superfast, metric-tuned (--tune ssim) | -29.3% | -30.6% | -30.0% | -27.2% | 206/206 | -1.9 [-2.3, -1.5] better |
| x265 medium | -28.7% | -28.6% | -27.5% | -29.9% | 206/206 |  |
| x265 medium, metric-tuned (--tune psnr) ● | -31.8% | -34.0% | -32.7% | -28.7% | 206/206 | -4.8 [-5.3, -4.2] better |
| x265 medium, metric-tuned (--tune ssim) | -30.2% | -31.4% | -28.7% | -30.6% | 206/206 | -2.2 [-2.4, -1.9] better |
| libvpx VP9 rt 7 | +2.5% | +7.5% | -2.9% | +3.1% | 206/206 |  |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) ● | +1.8% | +7.0% | +1.5% | -3.0% | 206/206 | -1.8 [-2.4, -1.0] better |
| libvpx VP9 rt 9 | -0.3% | +5.7% | -5.4% | -1.1% | 206/206 |  |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) ● | -1.3% | +5.5% | -5.8% | -3.5% | 206/206 | -1.3 [-1.7, -0.8] better |
| libaom rt 8 | -24.4% | -23.2% | -26.1% | -23.8% | 206/206 |  |
| libaom rt 8, metric-tuned (--tune ssim) | -24.2% | -21.6% | -21.7% | -29.4% | 206/206 | +0.8 [+0.1, +1.5] worse |
| libaom rt 10 | -23.9% | -24.6% | -24.2% | -22.9% | 206/206 |  |
| libaom rt 10, metric-tuned (--tune ssim) | -24.3% | -23.3% | -21.0% | -28.5% | 206/206 | +0.3 [-0.5, +1.1] not distinguishable |

## Against x264 medium, metric-tuned (--tune psnr)

The same points with x264 medium, metric-tuned (--tune psnr) as the zero line instead of the chart's own: direct BD-rate on the chart's clips and rungs (a clip without a direct overlap is left out of that row).

| Encoder | BD-J | BD-PSNR-YUV | BD-VMAF v1 | BD-MS-SSIM | clips |
|---|---:|---:|---:|---:|---:|
| **VVC Veryslow** | -51.6% | -51.6% | -49.5% | -53.5% | 205/206 |
| **VVC Slower** | -50.5% | -50.6% | -48.4% | -52.6% | 206/206 |
| **VVC Slow** | -49.3% | -49.0% | -47.1% | -51.7% | 206/206 |
| **AV2 Slower** | -48.8% | -51.4% | -45.6% | -49.4% | 204/206 |
| **HEVC Veryslow** | -48.0% | -48.3% | -45.1% | -50.5% | 205/206 |
| **VVC** | -47.4% | -47.3% | -45.3% | -49.7% | 206/206 |
| **AV2 Slow** | -45.6% | -48.5% | -42.3% | -46.0% | 206/206 |
| **HEVC Slower** | -44.9% | -44.6% | -41.8% | -48.2% | 206/206 |
| **AV2** | -44.8% | -47.5% | -41.9% | -45.0% | 206/206 |
| **AV1 Slower** | -44.7% | -47.3% | -38.9% | -47.9% | 206/206 |
| **HEVC Slow** | -44.4% | -44.3% | -42.2% | -46.6% | 206/206 |
| **VP9 Slower** | -44.1% | -45.7% | -40.1% | -46.5% | 206/206 |
| **AV1 Slow** | -43.6% | -46.3% | -38.4% | -46.0% | 206/206 |
| **VVC Fast** | -43.1% | -42.7% | -40.6% | -45.9% | 206/206 |
| **VP9 Slow** | -42.9% | -44.3% | -38.8% | -45.4% | 206/206 |
| **HEVC** | -41.5% | -41.2% | -39.4% | -44.0% | 206/206 |
| **AV2 Fast** | -41.5% | -43.3% | -38.0% | -43.0% | 206/206 |
| **VVC Faster** | -40.2% | -39.8% | -37.3% | -43.4% | 206/206 |
| **AV1** | -40.1% | -43.0% | -34.8% | -42.4% | 206/206 |
| **VP9** | -39.7% | -41.3% | -35.3% | -42.5% | 206/206 |
| **AV1 Fast** | -37.3% | -39.5% | -31.7% | -40.7% | 206/206 |
| **AV2 Faster** | -36.9% | -38.5% | -30.5% | -41.7% | 206/206 |
| **HEVC Fast** | -36.5% | -36.5% | -31.7% | -41.5% | 206/206 |
| **AV1 Faster** | -35.3% | -37.0% | -29.8% | -39.1% | 206/206 |
| **VP9 Fast** | -34.4% | -36.1% | -28.3% | -38.9% | 206/206 |
| **VP9 Faster** | -32.6% | -32.8% | -26.1% | -38.8% | 206/206 |
| **HEVC Faster** | -31.0% | -30.9% | -25.8% | -36.3% | 206/206 |
| **AV1 Veryfast** | -29.7% | -29.5% | -22.7% | -36.8% | 206/206 |
| SVT-AV1 M8 | -28.6% | -32.2% | -15.4% | -38.0% | 206/206 |
| **VP9 Veryfast** | -25.9% | -25.4% | -21.4% | -30.8% | 206/206 |
| **H.264 Slower** | -24.7% | -27.8% | -20.7% | -25.6% | 206/206 |
| **H.264 Slow** | -24.4% | -27.1% | -20.5% | -25.6% | 206/206 |
| x265 medium, metric-tuned (--tune psnr) | -23.1% | -24.2% | -15.9% | -29.3% | 206/206 |
| **H.264** | -22.5% | -25.0% | -18.9% | -23.6% | 206/206 |
| SVT-AV1 M10 | -21.8% | -23.5% | -10.6% | -31.4% | 206/206 |
| x265 medium, metric-tuned (--tune ssim) | -21.0% | -21.1% | -10.7% | -31.1% | 206/206 |
| x265 superfast, metric-tuned (--tune psnr) | -20.0% | -19.9% | -12.5% | -27.7% | 206/206 |
| x265 superfast, metric-tuned (--tune ssim) | -20.0% | -19.9% | -12.5% | -27.7% | 206/206 |
| x265 medium | -19.2% | -17.8% | -9.3% | -30.5% | 206/206 |
| x265 superfast | -18.6% | -16.7% | -11.6% | -27.4% | 206/206 |
| **H.264 Fast** | -17.0% | -18.3% | -14.0% | -18.8% | 206/206 |
| libaom rt 8 | -14.3% | -12.0% | -7.1% | -24.1% | 205/206 |
| **H.264 Faster** | -14.2% | -16.4% | -9.9% | -16.4% | 206/206 |
| libaom rt 8, metric-tuned (--tune ssim) | -13.8% | -10.4% | -1.9% | -29.6% | 205/206 |
| libaom rt 10, metric-tuned (--tune ssim) | -13.7% | -11.8% | -1.0% | -28.8% | 205/206 |
| libaom rt 10 | -13.5% | -13.1% | -4.5% | -23.2% | 205/206 |
| x265 ultrafast, metric-tuned (--tune psnr) | -13.1% | -12.4% | -5.3% | -21.7% | 206/206 |
| x265 ultrafast, metric-tuned (--tune ssim) | -13.1% | -12.4% | -5.3% | -21.7% | 206/206 |
| SVT-AV1 M12 | -12.8% | -12.6% | -2.8% | -23.0% | 206/206 |
| x265 ultrafast | -12.7% | -10.5% | -5.3% | -22.4% | 206/206 |
| SVT-AV1 M13 | -10.9% | -10.2% | -0.8% | -21.6% | 206/206 |
| **H.264 Veryfast** | -2.1% | -6.3% | +4.0% | -4.1% | 206/206 |
| x264 medium, metric-tuned (--tune ssim) | +9.9% | +8.1% | +24.6% | -3.2% | 206/206 |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | +11.5% | +19.9% | +18.2% | -3.5% | 206/206 |
| x264 veryfast, metric-tuned (--tune psnr) | +11.8% | +9.5% | +12.4% | +13.5% | 206/206 |
| x264 medium | +12.4% | +11.5% | +24.4% | +1.3% | 206/206 |
| libvpx VP9 rt 9 | +12.7% | +20.4% | +18.8% | -1.0% | 206/206 |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | +14.5% | +20.0% | +25.7% | -2.2% | 206/206 |
| x264 veryfast | +15.1% | +10.9% | +30.4% | +4.0% | 206/206 |
| libvpx VP9 rt 7 | +15.4% | +21.2% | +21.4% | +3.7% | 206/206 |
| x264 veryfast, metric-tuned (--tune ssim) | +16.2% | +12.5% | +33.5% | +2.7% | 206/206 |
| openh264, adaptive quantisation on | +28.7% | +25.1% | +41.4% | +19.6% | 206/206 |
| openh264 | +28.7% | +25.1% | +41.4% | +19.6% | 206/206 |
| x264 ultrafast, metric-tuned (--tune ssim) | +49.2% | +60.3% | +60.0% | +27.3% | 206/206 |
| x264 ultrafast | +50.9% | +62.9% | +41.7% | +48.0% | 206/206 |
| x264 ultrafast, metric-tuned (--tune psnr) | +51.0% | +63.0% | +41.6% | +48.3% | 206/206 |

## Excluded clips

Left out for every encoder, so that all points are averaged over the same clips: on these, some encoders' curves have no reliable overlap with x264 medium's (even with its extra rungs, which the 50% source cap limits).

- **pool_ducks_take_off_pristine** (texture): no reliable overlap for SVT-AV1 M8, SVT-AV1 M10, libvpx VP9 rt 9.
- **ind_bwc_videotext_video88** (bodycam): no reliable overlap for libaom rt 8.
- **ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508** (static): no reliable overlap for AV2 Slower, AV2 Slow, AV2.

Direct BD-J vs x264 medium on them, where it exists:

| Encoder | pool_ducks_take_off_pristine | ind_bwc_videotext_video88 | ind_meva_2018_03_07_17_00_00_17_05_00_bus_g508 |
|---|---:|---:|---:|
| VVC Veryslow | -71.9% | -68.9% | -68.0% |
| VVC Slower | -69.8% | -68.9% | -69.0% |
| VVC Slow | -68.3% | -68.9% | -64.7% |
| AV2 Slower | -67.7% | -66.6% | n/a |
| HEVC Veryslow | -57.5% | -63.0% | -68.2% |
| VVC | -66.5% | -66.5% | -67.8% |
| AV2 Slow | -64.5% | -66.8% | n/a |
| HEVC Slower | -53.7% | -60.8% | -67.3% |
| AV1 Slower | -61.8% | -57.9% | -64.9% |
| AV2 | -60.8% | -65.5% | n/a |
| HEVC Slow | -58.8% | -59.7% | -65.6% |
| VP9 Slower | -52.0% | -61.8% | -62.8% |
| AV1 Slow | -59.3% | -56.6% | -64.1% |
| VVC Fast | -63.6% | -63.4% | -65.8% |
| VP9 Slow | -50.4% | -60.8% | -62.7% |
| HEVC | -51.0% | -59.4% | -67.5% |
| AV2 Fast | -59.9% | -65.6% | -69.0% |
| VVC Faster | -57.8% | -63.4% | -64.7% |
| AV1 | -58.9% | -55.8% | -66.3% |
| VP9 | -49.8% | -58.2% | -63.4% |
| AV1 Fast | -54.9% | -57.8% | -64.0% |
| AV2 Faster | -54.0% | -64.9% | -65.6% |
| HEVC Fast | -53.3% | -55.8% | -62.7% |
| AV1 Faster | -56.3% | -57.8% | -64.4% |
| VP9 Fast | -44.9% | -58.6% | -61.7% |
| VP9 Faster | -47.3% | -59.0% | -60.6% |
| HEVC Faster | -47.3% | -52.8% | -59.5% |
| AV1 Veryfast | -56.7% | -57.6% | -61.9% |
| SVT-AV1 M8 | n/a | -60.8% | -55.6% |
| VP9 Veryfast | -40.7% | -51.1% | -56.8% |
| H.264 Slower | -18.9% | -24.1% | -61.9% |
| H.264 Slow | -17.6% | -26.1% | -61.3% |
| x265 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| SVT-AV1 M10 | n/a | -56.2% | -52.5% |
| H.264 | -11.4% | -22.1% | -60.6% |
| x265 superfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x265 medium | -49.5% | -35.4% | -40.3% |
| x265 superfast | -47.3% | -47.1% | -42.3% |
| H.264 Fast | -7.6% | -18.6% | -60.3% |
| libaom rt 8 | -18.7% | n/a | -22.4% |
| libaom rt 10 | -16.0% | -38.6% | -32.2% |
| H.264 Faster | -1.9% | -22.4% | -58.4% |
| x265 ultrafast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| SVT-AV1 M12 | -35.1% | -50.6% | -53.7% |
| x265 ultrafast | -47.2% | -45.2% | -41.7% |
| SVT-AV1 M13 | -34.7% | -49.0% | -52.3% |
| H.264 Veryfast | +0.1% | -15.3% | -54.7% |
| x264 medium, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| libvpx VP9 rt 9, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 9 | n/a | -19.2% | -13.7% |
| x264 veryfast, metric-tuned (--tune psnr) | n/a | n/a | n/a |
| x264 veryfast | +2.7% | -3.5% | -22.7% |
| libvpx VP9 rt 7, metric-tuned (--tune ssim) | n/a | n/a | n/a |
| libvpx VP9 rt 7 | -6.0% | -13.6% | +24.9% |
| openh264, adaptive quantisation on | n/a | n/a | n/a |
| openh264 | +89.8% | +32.2% | -66.5% |
| x264 ultrafast | +15.0% | +10.5% | -10.6% |

## Selected pairs

Direct BD-rate of our encoders against the reference encoders and against each other, on the same points (no anchor in between; ours vs ours drops the rungs either arm overshot, as on the board).

| nanocodec | vs | BD-J | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | clips |
|---|---|---:|---:|---:|---|---:|
| H.264 Slower | openh264 | -37.5% | -38.5% | -40.3% | -47.4 … -29.8 | 209/209 |
| H.264 Slow | openh264 | -37.2% | -37.8% | -40.1% | -47.7 … -28.9 | 209/209 |
| H.264 | openh264 | -35.3% | -35.8% | -38.7% | -46.1 … -26.1 | 209/209 |
| H.264 | x264 veryfast | -36.1% | -38.0% | -40.4% | -45.1 … -30.3 | 209/209 |
| H.264 | x264 medium | -30.2% | -33.9% | -34.1% | -43.6 … -22.3 | 209/209 |
| H.264 Fast | openh264 | -29.7% | -28.9% | -34.2% | -42.4 … -20.7 | 209/209 |
| H.264 Fast | x264 veryfast | -31.1% | -32.5% | -35.7% | -39.9 … -24.6 | 209/209 |
| H.264 Fast | H.264 | +8.7% | +10.2% | +7.5% | +6.1 … +10.9 | 209/209 |
| H.264 Faster | openh264 | -26.5% | -26.5% | -30.3% | -39.5 … -17.1 | 209/209 |
| H.264 Faster | x264 ultrafast | -47.0% | -55.5% | -38.9% | -56.8 … -44.3 | 185/209 |
| H.264 Faster | H.264 Fast | +3.6% | +2.0% | +5.8% | +2.0 … +5.5 | 209/209 |
| H.264 Veryfast | openh264 | -12.6% | -14.4% | -16.8% | -27.9 … -5.4 | 209/209 |
| H.264 Veryfast | x264 ultrafast | -39.7% | -50.5% | -30.2% | -50.9 … -35.5 | 204/209 |
| H.264 Veryfast | H.264 Faster | +16.9% | +14.6% | +17.0% | +10.1 … +21.4 | 209/209 |
| HEVC Veryslow | SVT-AV1 M12 | -40.6% | -41.5% | -44.2% | -46.6 … -37.5 | 209/209 |
| HEVC Slower | SVT-AV1 M12 | -37.4% | -37.7% | -41.2% | -43.3 … -33.6 | 209/209 |
| HEVC Slow | SVT-AV1 M12 | -35.5% | -35.9% | -40.6% | -42.8 … -32.6 | 209/209 |
| HEVC | SVT-AV1 M12 | -32.2% | -32.5% | -38.1% | -40.3 … -28.6 | 209/209 |
| HEVC | x265 medium | -27.0% | -29.4% | -32.9% | -39.6 … -21.2 | 209/209 |
| HEVC Fast | SVT-AV1 M12 | -25.1% | -25.3% | -28.1% | -33.3 … -22.0 | 209/209 |
| HEVC Fast | x265 superfast | -21.3% | -23.9% | -21.6% | -33.7 … -17.1 | 209/209 |
| HEVC Fast | HEVC | +11.1% | +11.4% | +15.9% | +8.5 … +13.8 | 209/209 |
| HEVC Faster | SVT-AV1 M12 | -16.7% | -16.7% | -20.6% | -26.8 … -10.7 | 209/209 |
| HEVC Faster | SVT-AV1 M13 | -19.1% | -19.5% | -22.9% | -29.0 … -14.5 | 209/209 |
| HEVC Faster | HEVC Fast | +11.8% | +12.2% | +10.6% | +7.0 … +15.7 | 209/209 |
| AV1 Slower | SVT-AV1 M12 | -37.0% | -40.4% | -37.3% | -44.7 … -33.6 | 209/209 |
| AV1 Slow | SVT-AV1 M12 | -34.9% | -38.6% | -36.1% | -43.4 … -29.5 | 209/209 |
| AV1 | SVT-AV1 M12 | -31.2% | -35.4% | -33.1% | -39.8 … -26.5 | 209/209 |
| AV1 | openh264 | -51.9% | -53.7% | -52.1% | -64.4 … -42.1 | 207/209 |
| AV1 | SVT-AV1 M8 | -10.0% | -11.2% | -18.6% | -21.7 … -5.1 | 209/209 |
| AV1 | libaom rt 8 | -27.4% | -33.3% | -28.3% | -38.8 … -21.6 | 209/209 |
| AV1 Fast | SVT-AV1 M12 | -27.2% | -30.5% | -29.2% | -35.6 … -23.1 | 209/209 |
| AV1 Fast | openh264 | -48.9% | -50.1% | -48.6% | -63.2 … -38.0 | 208/209 |
| AV1 Fast | AV1 | +6.3% | +8.1% | +6.2% | +3.2 … +8.8 | 209/209 |
| AV1 Faster | SVT-AV1 M12 | -24.2% | -26.9% | -26.4% | -36.1 … -19.6 | 209/209 |
| AV1 Faster | AV1 Fast | +3.9% | +4.8% | +4.0% | -1.0 … +7.1 | 209/209 |
| AV1 Veryfast | SVT-AV1 M12 | -17.1% | -16.6% | -18.5% | -30.3 … -10.1 | 209/209 |
| AV1 Veryfast | SVT-AV1 M13 | -19.7% | -19.6% | -21.0% | -31.6 … -13.5 | 209/209 |
| AV1 Veryfast | AV1 Faster | +10.8% | +15.8% | +12.0% | +5.7 … +19.2 | 209/209 |
| VP9 Slower | HEVC Slower | +2.1% | -2.0% | +4.4% | -1.7 … +6.0 | 209/209 |
| VP9 Slow | HEVC Slow | +3.3% | -0.3% | +7.1% | -0.3 … +8.3 | 209/209 |
| VP9 | HEVC | +4.0% | +0.1% | +8.3% | +1.1 … +9.1 | 209/209 |
| VP9 | libvpx VP9 rt 7 | -44.8% | -48.9% | -43.7% | -54.1 … -39.6 | 209/209 |
| VP9 Fast | HEVC Fast | +4.7% | +0.7% | +7.1% | +0.0 … +10.8 | 209/209 |
| VP9 Fast | libvpx VP9 rt 9 | -39.7% | -46.3% | -37.7% | -48.8 … -35.2 | 209/209 |
| VP9 Fast | VP9 | +10.8% | +11.0% | +13.5% | +7.2 … +14.9 | 209/209 |
| VP9 Faster | HEVC Faster | -2.7% | -3.9% | +0.5% | -6.2 … +0.3 | 209/209 |
| VP9 Faster | libvpx VP9 rt 9 | -37.3% | -42.6% | -35.2% | -47.6 … -31.6 | 209/209 |
| VP9 Faster | VP9 Fast | +3.5% | +6.4% | +3.3% | +0.3 … +7.5 | 209/209 |
| VP9 Veryfast | HEVC Faster | +10.4% | +10.0% | +9.0% | +6.0 … +16.0 | 209/209 |
| VP9 Veryfast | libvpx VP9 rt 9 | -29.5% | -34.8% | -29.6% | -41.2 … -23.9 | 209/209 |
| VP9 Veryfast | VP9 Faster | +14.2% | +14.8% | +9.2% | +8.7 … +19.9 | 209/209 |
| AV2 Slower | AV1 Slower | -8.7% | -9.6% | -12.1% | -11.5 … -5.8 | 209/209 |
| AV2 Slow | AV1 Slow | -4.7% | -5.3% | -7.9% | -8.7 … -0.2 | 209/209 |
| AV2 | AV1 | -8.3% | -8.3% | -11.1% | -10.8 … -5.1 | 209/209 |
| AV2 | SVT-AV1 M8 | -17.4% | -18.5% | -27.7% | -27.8 … -13.5 | 209/209 |
| AV2 Fast | AV1 Fast | -8.3% | -7.5% | -11.4% | -13.1 … -4.4 | 209/209 |
| AV2 Fast | AV2 | +6.4% | +9.1% | +6.1% | +2.1 … +8.9 | 209/209 |
| AV2 Faster | AV1 Faster | -3.6% | -2.6% | -2.2% | -7.3 … +0.8 | 209/209 |
| AV2 Faster | SVT-AV1 M12 | -27.8% | -29.7% | -28.5% | -36.2 … -23.2 | 209/209 |
| AV2 Faster | AV2 Fast | +9.4% | +10.4% | +14.6% | +5.8 … +11.7 | 209/209 |
| VVC Veryslow | HEVC Veryslow | -8.7% | -8.5% | -9.3% | -10.8 … -4.5 | 209/209 |
| VVC Slower | HEVC Slower | -12.4% | -13.3% | -12.9% | -13.7 … -8.9 | 209/209 |
| VVC Slow | HEVC Slow | -11.3% | -11.1% | -10.4% | -12.9 … -6.7 | 209/209 |
| VVC | HEVC | -11.9% | -12.4% | -10.8% | -13.1 … -7.1 | 209/209 |
| VVC | x265 medium | -35.4% | -37.9% | -40.1% | -44.1 … -29.1 | 209/209 |
| VVC Fast | HEVC Fast | -13.2% | -13.6% | -15.4% | -15.0 … -8.0 | 209/209 |
| VVC Fast | VVC | +9.4% | +9.8% | +9.8% | +7.2 … +11.7 | 209/209 |
| VVC Faster | HEVC Faster | -17.5% | -18.2% | -18.4% | -20.0 … -13.2 | 209/209 |
| VVC Faster | VVC Fast | +6.4% | +6.3% | +7.0% | +3.7 … +7.0 | 209/209 |

## Method

- **Anchor:** x264 0.165 `--preset medium --tune zerolatency --bframes 0
  --rc-lookahead 0 --no-scenecut --threads 1`, ABR with a 1 s VBV at each target.
  BD 0 % and speed 1× by definition.
- **Points:** every encoder in its own real-time rate control at the same targets:
  ours in CBR (`--bitrate`, 1 s buffer; build 8b492382 (encoder sources = 0cdb619d; rate control 62788659 + 71cad753) (H.264 Veryfast: 20d31869; H.264 Fast, H.264 Faster, HEVC, HEVC Fast, HEVC Faster, AV1, AV1 Fast, AV1 Faster, AV1 Veryfast, VP9, VP9 Fast, VP9 Faster, VP9 Veryfast, AV2, AV2 Fast, VVC, VVC Fast, H.264 Slow, HEVC Slow, AV1 Slow, VP9 Slow, AV2 Slow, VVC Slow, AV2 Faster, H.264 Slower, HEVC Veryslow, HEVC Slower, AV1 Slower, VP9 Slower, AV2 Slower, VVC Veryslow, VVC Slower, VVC Faster: 4fc9a1de; H.264: c9cff89f); H.264 timed on 20d31869 (same streams), its presets), openh264
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
  gain. Plotted on these clips (direct BD-J vs the charted setting, 95% interval): openh264 AQ on (-0.0 [-0.0, -0.0]%); x264 veryfast --tune psnr (-1.5 [-2.7, -0.3]%); x264 medium --tune psnr (-9.8 [-10.9, -8.7]%); x265 ultrafast --tune psnr (-0.6 [-1.0, -0.2]%); x265 superfast --tune psnr (-1.9 [-2.3, -1.5]%); x265 medium --tune psnr (-4.8 [-5.3, -4.2]%); libvpx VP9 rt 7 --tune ssim (-1.8 [-2.4, -1.0]%); libvpx VP9 rt 9 --tune ssim (-1.3 [-1.7, -0.8]%). The others stay in the table with their intervals
  (openh264's adaptive quantisation produced byte-identical streams in our harness).
  This direct comparison and the BD-J columns (each setting against the x264 anchor)
  can differ by a few tenths; small disagreements between them are within the
  per-clip noise. Each plotted variant is
  drawn in its default twin's shape, lightly filled, joined to it by a dashed line;
  every variant is in the "Metric-tuned references" table, and every point against
  the better-scoring x264 medium tune is in the table after it. Until they are
  timed themselves they take their default twin's fps (noted in the table).
