# Held-out chart check (2026-11-12)

The published bitrate chart ([bd-vs-speed.md](bd-vs-speed.md)) re-measured on 71 clips that no tuning, screen gate or chart decision has used, as a check against overfitting. Same families, builds, real-use CBR ladder (150 frames, one key frame, rungs within 50% of the source rate, the low rungs), the same BD-J vs x264 medium (nceval.chart), encoded on a rented machine and scored here (`tools/dev/aws_heldout_chart.py`).

Clips: 71 of the screening pool's stored encodes whose source family is in no raw clip set (TRAIN, VAL, TEST, the chart's VAL clips, adv12), one per family, 21 datasets; by class bodycam 10, crowd 18, drone 3, lowlight 7, pan 11, static 19, texture 3.

Every family's BD-J vs x264 medium moved by a median of +7.56 points: on these clips the anchor itself does relatively better, which moves every gap alike. The overfitting signal is a move apart from that common shift ("vs median"): ours median -0.37 (range -5.07 to +1.79), the reference encoders median +1.85 (range -12.45 to +7.44).

Flag: a move more than 2 points from the common shift, or a rank move of more than 2 places. Flagged: SVT-AV1 M8 (+2.54 pts vs median, rank 8 -> 10), AV1 Faster (+1.68 pts vs median, rank 15 -> 12), SVT-AV1 M10 (+4.51 pts vs median, rank 12 -> 13), x265 medium (+5.17 pts vs median, rank 13 -> 15), x265 superfast (+4.83 pts vs median, rank 14 -> 16), libaom rt 8 (-1.90 pts vs median, rank 21 -> 17), x265 ultrafast (+7.27 pts vs median, rank 17 -> 19), SVT-AV1 M12 (+6.96 pts vs median, rank 18 -> 20), SVT-AV1 M13 (+7.44 pts vs median, rank 19 -> 21), H.264 (-5.07 pts vs median, rank 22 -> 22), H.264 Fast (-4.00 pts vs median, rank 23 -> 23), H.264 Faster (-2.82 pts vs median, rank 24 -> 24), libvpx VP9 rt 7 (-3.89 pts vs median, rank 26 -> 27), openh264 (-12.45 pts vs median, rank 30 -> 30).

| Family | BD-J held-out | BD-J chart | Δ pts | Δ vs median | rank held-out | rank chart | BD-PSNR | BD-VMAF | clips | flag |
|---|---|---|---|---|---|---|---|---|---|---|
| VVC | -46.33 | -53.40 | +7.07 | -0.49 | 1 | 2 | -47.48 | -49.71 | 70/70 |  |
| AV2 | -44.45 | -53.55 | +9.10 | +1.55 | 2 | 1 | -49.25 | -47.27 | 70/70 |  |
| AV1 | -42.53 | -50.87 | +8.34 | +0.79 | 3 | 3 | -47.26 | -44.03 | 70/70 |  |
| HEVC | -41.73 | -49.26 | +7.53 | -0.03 | 4 | 4 | -43.22 | -45.73 | 70/70 |  |
| VP9 | -41.16 | -47.90 | +6.74 | -0.82 | 5 | 5 | -44.24 | -42.67 | 70/70 |  |
| VVC Fast | -39.14 | -46.45 | +7.31 | -0.25 | 6 | 6 | -38.92 | -42.05 | 70/70 |  |
| AV2 Fast | -36.90 | -45.56 | +8.66 | +1.11 | 7 | 7 | -40.57 | -37.78 | 70/70 |  |
| AV1 Fast | -35.86 | -45.20 | +9.34 | +1.79 | 8 | 9 | -36.06 | -39.16 | 70/70 |  |
| VP9 Fast | -35.55 | -42.48 | +6.93 | -0.62 | 9 | 11 | -36.57 | -39.08 | 70/70 |  |
| SVT-AV1 M8 | -35.45 | -45.55 | +10.10 | +2.54 | 10 | 8 | -41.11 | -30.22 | 70/70 | FLAG |
| HEVC Fast | -35.19 | -43.40 | +8.21 | +0.65 | 11 | 10 | -36.26 | -38.38 | 70/70 |  |
| AV1 Faster | -29.51 | -38.75 | +9.24 | +1.68 | 12 | 15 | -29.31 | -32.02 | 70/70 | FLAG |
| SVT-AV1 M10 | -29.23 | -41.30 | +12.07 | +4.51 | 13 | 12 | -33.15 | -26.25 | 70/70 | FLAG |
| HEVC Faster | -28.98 | -36.03 | +7.05 | -0.51 | 14 | 16 | -28.60 | -30.94 | 70/70 |  |
| x265 medium | -27.95 | -40.68 | +12.73 | +5.17 | 15 | 13 | -28.24 | -26.56 | 70/70 | FLAG |
| x265 superfast | -26.45 | -38.84 | +12.39 | +4.83 | 16 | 14 | -26.47 | -28.02 | 70/70 | FLAG |
| libaom rt 8 | -23.04 | -28.70 | +5.66 | -1.90 | 17 | 21 | -22.59 | -23.95 | 70/70 | FLAG |
| libaom rt 10 | -22.22 | -29.81 | +7.59 | +0.03 | 18 | 20 | -23.36 | -21.93 | 70/70 |  |
| x265 ultrafast | -20.72 | -35.55 | +14.83 | +7.27 | 19 | 17 | -20.11 | -22.35 | 70/70 | FLAG |
| SVT-AV1 M12 | -20.39 | -34.91 | +14.52 | +6.96 | 20 | 18 | -21.83 | -19.33 | 70/70 | FLAG |
| SVT-AV1 M13 | -18.24 | -33.23 | +14.99 | +7.44 | 21 | 19 | -19.13 | -17.18 | 70/70 | FLAG |
| H.264 | -17.93 | -20.42 | +2.49 | -5.07 | 22 | 22 | -20.90 | -23.00 | 70/70 | FLAG |
| H.264 Fast | -10.38 | -13.94 | +3.56 | -4.00 | 23 | 23 | -13.51 | -16.65 | 70/70 | FLAG |
| H.264 Faster | -3.74 | -8.48 | +4.74 | -2.82 | 24 | 24 | -7.63 | -8.44 | 70/70 | FLAG |
| libvpx VP9 rt 9 | -0.05 | -5.72 | +5.67 | -1.89 | 25 | 25 | +4.97 | -4.38 | 70/70 |  |
| x264 medium | +0.00 | +0.00 |  |  | 26 | 28 | +0.00 | +0.00 | 70/70 |  |
| libvpx VP9 rt 7 | +2.15 | -1.52 | +3.67 | -3.89 | 27 | 26 | +6.05 | -2.36 | 70/70 | FLAG |
| H.264 Fastest | +5.98 | -0.76 | +6.74 | -0.81 | 28 | 27 | +1.30 | -0.85 | 70/70 |  |
| x264 veryfast | +8.07 | +2.28 | +5.79 | -1.77 | 29 | 29 | +6.77 | +8.71 | 70/70 |  |
| openh264 | +19.73 | +24.62 | -4.89 | -12.45 | 30 | 30 | +15.61 | +20.29 | 70/70 | FLAG |
| x264 ultrafast | +47.64 | +38.93 | +8.71 | +1.16 | 31 | 31 | +64.45 | +20.04 | 70/70 |  |

Clips left out for every point (no direct overlap with the anchor): pool_ducks_take_off_pristine.

Clips: pool_aboda_video11, pool_barscene_pristine, pool_bodycam_antare_antare_test, pool_bodycam_antare_pub_garden, pool_bodycam_antare_refused_entry_bwc_02, pool_bodycam_bwc_video118, pool_bodycam_cart_shouting_12, pool_bodycam_knife_drawn_04, pool_bodycam_knife_drawn_05, pool_bodycam_oxford_street, pool_bodycam_queue_scuffle_06, pool_bodycam_shoplifting_10, pool_boxingpractice, pool_caucafall_walk_s9, pool_chad_1_093, pool_chad_3_086, pool_chad_4_109, pool_crosswalk, pool_crowd_MOT20_mot20_08, pool_crowd_cevo_ukof_hd_indoor_light_ohcam_0, pool_crowd_mot_mot17_04, pool_crowd_person_uid_vid_00210, pool_crowd_person_uid_vid_00222, pool_crowd_person_uid_vid_00224, pool_dancers_pristine, pool_drone_rounda_roundabouthd_c001, pool_drone_rounda_roundabouthd_c002, pool_drone_rounda_roundabouthd_c004, pool_ducks_take_off_pristine, pool_foodmarket_pristine, pool_gmdcsa24_s1_adl05, pool_in_to_tree_360p, pool_jaad_0081, pool_jaad_0110, pool_jaad_0340, pool_kristen_sara_360p, pool_lowlight_bdd100_bdd_00cee0e0_5e25daa7, pool_lowlight_bdd100_bdd_00d8d95a_47d98291, pool_lowlight_bdd100_bdd_00db9030_5102ed41, pool_lowlight_cevo_inof_md_indoor_night_ohcam_0, pool_meva_bus_g506, pool_meva_hospital_g301, pool_meva_school_g336, pool_midnight_sun_pristine, pool_narrator_pristine, pool_old_town_cross_360p, pool_otw_homes_00004, pool_otw_homes_00015, pool_otw_homes_00168, pool_otw_lots_064, pool_pan_bdd100_bdd_00ce8219_d0b5582e, pool_pan_bdd100_bdd_00d7268f_fd4487be, pool_pan_jaad_video_0128, pool_pan_jaad_video_0194, pool_pan_jaad_video_0199, pool_pp22_uid_00001, pool_pp22_uid_00070, pool_smoke_sauna_360p, pool_static_antare_cart_shouting_fixed_10, pool_static_bdd100_bdd_00dc5030_237e7f71, pool_static_cevo_ukof_ld_indoor_light_ohcam_0, pool_static_chirla_chirla_seq_024_camera_7_2023, pool_static_otw_homes_00323, pool_tango_pristine, pool_towncentre, pool_vidyo1_pristine, pool_vidyo3_360p, pool_vidyo4, pool_waterfall_360p, pool_wildtrack_cam1, pool_wildtrack_cam3.
## How it was run

- **Encodes:** all 12,958 on one c7a.metal-48xl spot instance in 12.5 min
  ($0.63 including the 2.8 GB download). Our streams are byte-identical to
  local encodes on the clips checked, and two clips' remote source decodes were
  checked against local ones.
- **Scoring:** 9,049 points on c7a.metal-48xl spot instances with the CPU
  metrics backend (libvmaf VMAF v1 + `nceval.msssim`, every frame). That
  backend matches VMAFx at its stored precision; one point checked here showed
  MS-SSIM 2.4e-8 apart and VMAF and PSNR identical.
- **The 34 low-rung points of `pool_ducks_take_off_pristine`:** scored on the
  local GPU (VMAFx). The CPU MS-SSIM then raised "math domain error" on x265
  ultrafast's 35 kbps point there (a negative structure mean); since 2026-11-12
  it takes each term's magnitude like VMAFx and gives 0.244345 (VMAFx 0.244345,
  5.7e-7 apart). That clip is left out of every point's BD anyway.
- **Caveat:** the held-out clips are the pool's stored high-rate encodes
  (graded clean sources), not raw captures. The H.264 family's smaller-than-
  common shift may partly reflect H.264-coded sources.
