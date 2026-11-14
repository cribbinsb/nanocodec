# Spatial scalability (SVC): 720p30 + 360p30 (+ 180p30)

<sub>Updated 2026-11-13, build 52903ab1, vp9-svc at 53890097, vp9-svc+L2T3 at 53890097, vp9-svc-fast at 53890097, av1-svc3 at 4594ae65, av1-svc3+L3T3 at 4594ae65, av1-svc3-fast at 4594ae65; libaom svc_encoder_rtc v3.15.3, libvpx vp9_spatial_svc_encoder v1.16.0; paired-timed x86 speed (see the notes)</sub>

![SVC chart](svc-chart.svg)

Two-layer AV1 and VP9 streams (360p base, operating point 1; 720p, operating point 0) and three-layer AV1 streams (180p, operating point 2; 360p, 1; 720p, 0) on the main chart's clips, rungs, length and key interval. **Rate is the total stream rate (every layer); quality is J_real of the 720p output**, so BD-J is against single-layer x264 medium at 720p, on the same scale as [the main chart](bd-vs-speed.md). The single-layer AV1 and VP9 Default lines show what the lower layers cost at 720p.

| Encoder | BD-J op 0 | BD-PSNR-YUV | BD-VMAF v1 | per-clip IQR (BD-J) | 360p op BD-J | 360p op share of rate | fps (Jetson, 1 core) | fps (x86, 1 core) | x86 speed vs x264 medium | clips |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **AV1 Default 720p (single layer)** | -53.0% | -57.3% | -57.7% | -55.7 … -43.0 |  |  | 41.1 | 92.4 | 1.37× | 8/8 |
| **VP9 Default 720p (single layer)** | -50.1% | -54.4% | -53.6% | -53.8 … -39.6 |  |  | 34.4 | 89.7 | 1.33× | 8/8 |
| **AV1 SVC L2T3** | -46.6% | -50.7% | -52.2% | -49.0 … -34.2 | -8.6% | 23% | 25.5 | 57.9 | 0.86× | 8/8 |
| **AV1 SVC L2T1** | -44.9% | -49.6% | -50.1% | -50.0 … -33.3 | -6.5% | 22% | 24.0 | 54.8 | 0.81× | 8/8 |
| **VP9 SVC L2T3** | -42.4% | -46.2% | -46.8% | -46.6 … -29.2 | -3.1% | 21% | 23.4 | 58.8 | 0.87× | 8/8 |
| **AV1 SVC L3T3** | -41.7% | -45.9% | -48.0% | -43.3 … -28.9 | +17.5% | 28% | 23.3 | 54.3 | 0.80× | 8/8 |
| **VP9 SVC L2T1** | -40.7% | -44.5% | -45.1% | -45.5 … -28.1 | -0.7% | 21% | 22.3 | 56.2 | 0.83× | 8/8 |
| **AV1 SVC Fast L2T1** | -40.1% | -41.7% | -46.7% | -44.5 … -25.7 | +7.0% | 22% | 42.4 | 78.4 | 1.16× | 8/8 |
| **AV1 SVC L3T1** | -39.6% | -44.3% | -45.8% | -44.0 … -28.0 | +20.5% | 27% | 22.3 | 51.9 | 0.77× | 8/8 |
| **VP9 SVC Fast L2T1** | -35.2% | -37.9% | -40.5% | -39.0 … -23.7 | +16.4% | 23% | 33.7 | 74.2 | 1.10× | 8/8 |
| **AV1 SVC Fast L3T1** | -33.7% | -35.5% | -40.6% | -37.0 … -21.3 | +37.2% | 27% | 39.9 | 75.4 | 1.12× | 8/8 |
| **AV1 simulcast 720p+360p+180p** | -33.7% | -38.3% | -39.1% | -35.5 … -20.1 |  | 26% | 31.8* | 72.7 | 1.08× | 8/8 |
| **AV1 simulcast 720p+360p** | -33.3% | -37.9% | -38.8% | -34.9 … -20.1 |  | 35% | 32.9* | 72.3 | 1.07× | 8/8 |
| **VP9 simulcast 720p+360p** | -29.6% | -34.1% | -33.2% | -33.2 … -16.0 |  | 35% | 28.1* | 73.2 | 1.08× | 8/8 |
| libaom SVC L2T1 rt 10 | -8.7% | -14.0% | -11.4% | -17.9 … +10.1 | +42.6% | 32% | 30.6 | 68.6 | 1.01× | 8/8 |
| libaom SVC L2T1 rt 8 | -8.5% | -12.5% | -10.1% | -20.4 … +8.6 | +47.4% | 31% | 24.0 | 54.2 | 0.80× | 8/8 |
| libaom SVC L3T1 rt 10 | -6.7% | -12.7% | -9.3% | -15.4 … +13.3 | +80.3% | 33% | 27.6 | 64.5 | 0.95× | 8/8 |
| libaom SVC L3T1 rt 8 | -1.7% | -5.1% | -4.0% | -15.4 … +16.7 | +87.5% | 32% | 20.3 | 46.2 | 0.68× | 8/8 |
| x264 medium (720p) | +0.0% | +0.0% | +0.0% | +0.0 … +0.0 |  |  | 26.7 | 67.6 | 1.00× | 8/8 |
| libaom SVC L2T3 rt 8 | +16.4% | +14.4% | +12.1% | -4.6 … +37.6 | +60.1% | 31% | 26.0 | 59.4 | 0.88× | 8/8 |
| libaom SVC L2T3 rt 10 | +18.1% | +15.1% | +14.9% | +0.6 … +36.0 | +59.7% | 30% | 32.8 | 77.1 | 1.14× | 8/8 |
| libaom SVC L3T3 rt 8 | +19.1% | +16.3% | +15.6% | -1.3 … +39.7 | +118.0% | 32% | 24.1 | 53.3 | 0.79× | 8/8 |
| libaom SVC L3T3 rt 10 | +21.1% | +17.5% | +17.3% | +2.4 … +39.1 | +121.5% | 32% | 30.2 | 65.9 | 0.97× | 8/8 |
| libvpx SVC L2T1 rt 9 | +27.4% | +29.6% | +17.8% | +13.1 … +36.0 | +108.2% | 30% | 37.6 | 72.4 | 1.07× | 8/8 |
| libvpx SVC L2T1 rt 7 | +30.3% | +31.3% | +20.6% | +18.2 … +46.9 | +114.8% | 31% | 22.9 | 53.3 | 0.79× | 8/8 |

Speed: x264 medium's timed fps (the main chart's) over the point's median per-round time ratio to x264 medium, timed in pairs on one reserved core (`time-host`, the 600 kbps rung, lowlight_city_night, crowd_mot17_09, bodycam_bc1; the Jetson's with the same pairing on its core 3).

\* Estimated, not timed (no paired timing on that platform yet). Each point's fps is the main chart's timed fps of the same program coded single-layer (ours: the codec's Default, or its Fast for an SVC Fast point; libaom: libaom rt at the same speed, aomenc; libvpx: vpxenc rt at the same speed) times that command's instructions over the point's (callgrind instructions per frame (x86-64, AVX2 build), both layers; ours and the references at the 600 kbps rung in CBR, the single-layer commands the main chart's rate-matched fixed-QP ones; lowlight_city_night, crowd_mot17_09, bodycam_bc1). Ratios within one program carry across machines far better than between programs; the Jetson column applies the x86 ratio to the Jetson's timed fps.

## 360p operating point (operating point 1)

360p output against the source downscaled to 360p (ffmpeg Lanczos), at the operating point's own rate (what a 360p receiver gets: the base layer of two, the 180p and 360p layers of three), as BD-J against the same codec's Default coded alone at 360p (the two-layer simulcast's low stream): AV1 points against AV1 Default, VP9 points against VP9 Default.

| Encoder | base BD-J | base BD-PSNR-YUV | base BD-VMAF v1 | base share of total rate | mean base PSNR-YUV | mean base VMAF v1 | clips |
|---|---:|---:|---:|---:|---:|---:|---:|
| AV1 SVC L2T3 | -8.6% | -7.4% | -9.9% | 23% | 31.96 dB | 55.5 | 8/8 |
| AV1 SVC L2T1 | -6.5% | -7.0% | -6.3% | 22% | 31.82 dB | 55.0 | 8/8 |
| AV1 SVC Fast L2T1 | +7.0% | +14.6% | +6.2% | 22% | 31.02 dB | 53.0 | 8/8 |
| AV1 SVC L3T3 | +17.5% | +18.3% | +16.5% | 28% | 31.73 dB | 53.8 | 8/8 |
| AV1 SVC L3T1 | +20.5% | +20.8% | +20.1% | 27% | 31.54 dB | 53.3 | 8/8 |
| AV1 SVC Fast L3T1 | +37.2% | +48.3% | +37.4% | 27% | 30.72 dB | 51.0 | 8/8 |
| libaom SVC L2T1 rt 10 | +42.6% | +48.2% | +51.4% | 32% | 32.55 dB | 63.3 | 8/8 |
| libaom SVC L2T1 rt 8 | +47.4% | +53.3% | +56.0% | 31% | 32.47 dB | 63.6 | 8/8 |
| libaom SVC L2T3 rt 10 | +59.7% | +62.9% | +69.2% | 30% | 32.13 dB | 61.7 | 8/8 |
| libaom SVC L2T3 rt 8 | +60.1% | +65.0% | +66.4% | 31% | 32.09 dB | 61.8 | 8/8 |
| libaom SVC L3T1 rt 10 | +80.3% | +82.9% | +98.5% | 33% | 31.98 dB | 58.5 | 8/8 |
| libaom SVC L3T1 rt 8 | +87.5% | +92.0% | +105.1% | 32% | 31.99 dB | 59.1 | 8/8 |
| libaom SVC L3T3 rt 8 | +118.0% | +121.2% | +133.6% | 32% | 31.33 dB | 56.6 | 8/8 |
| libaom SVC L3T3 rt 10 | +121.5% | +122.7% | +139.0% | 32% | 31.35 dB | 56.5 | 8/8 |
| AV1 Default, 360p alone | +0.0% | +0.0% | +0.0% | 35% | 33.30 dB | 63.2 | |
| VP9 SVC L2T3 | -3.1% | -3.2% | -1.4% | 21% | 31.57 dB | 53.3 | 8/8 |
| VP9 SVC L2T1 | -0.7% | -1.1% | +0.8% | 21% | 31.36 dB | 52.9 | 8/8 |
| VP9 SVC Fast L2T1 | +16.4% | +19.1% | +17.3% | 23% | 31.02 dB | 52.5 | 8/8 |
| libvpx SVC L2T1 rt 9 | +108.2% | +130.3% | +115.2% | 30% | 32.39 dB | 67.9 | 8/8 |
| libvpx SVC L2T1 rt 7 | +114.8% | +131.7% | +119.2% | 31% | 32.41 dB | 68.2 | 7/8 |
| VP9 Default, 360p alone | +0.0% | +0.0% | +0.0% | 35% | 33.25 dB | 63.8 | |

## 180p operating point of three layers (operating point 2)

180p output against the source downscaled to 180p (ffmpeg Lanczos), at the 180p layer's own rate (what a thumbnail receiver gets), as BD-J against AV1 Default coded alone at 180p (the three-way simulcast's lowest stream).

| Encoder | 180p BD-J | 180p BD-PSNR-YUV | 180p BD-VMAF v1 | 180p share of total rate | mean 180p PSNR-YUV | mean 180p VMAF v1 | clips |
|---|---:|---:|---:|---:|---:|---:|---:|
| AV1 SVC L3T3 | +0.8% | +1.4% | +0.5% | 7% | 29.02 dB | 40.8 | 8/8 |
| AV1 SVC L3T1 | +4.9% | +4.8% | +4.9% | 7% | 28.85 dB | 39.8 | 8/8 |
| AV1 SVC Fast L3T1 | +12.4% | +17.2% | +11.1% | 7% | 28.23 dB | 37.6 | 8/8 |
| libaom SVC L3T1 rt 8 | +38.1% | +57.8% | +30.4% | 8% | 29.23 dB | 51.9 | 8/8 |
| libaom SVC L3T3 rt 8 | +42.2% | +60.1% | +34.6% | 8% | 29.07 dB | 50.9 | 8/8 |
| libaom SVC L3T1 rt 10 | +42.2% | +64.1% | +34.1% | 8% | 29.07 dB | 51.4 | 8/8 |
| libaom SVC L3T3 rt 10 | +48.3% | +68.2% | +40.5% | 8% | 28.93 dB | 50.4 | 8/8 |
| AV1 Default, 180p alone | +0.0% | +0.0% | +0.0% | 9% | 30.05 dB | 46.6 | |

## Method

- Every two-layer point targets the total rung with 35% for the base layer (our `spatial.base_rate_share` default; our encoder adapts it per frame, libaom and simulcast keep it); every three-layer point with 8% for 180p and 26% for 360p (our `spatial.s0_rate_share` / `spatial.s1_rate_share` defaults: caps our encoder passes unspent bits up from).
- Three layers (AV1): `presets/av1-svc3.toml` (AV1 Default plus `spatial.layers = 3`, 2:1 steps) in L3T1 and L3T3 (`layers.adaptive=0`), `presets/av1-svc3-fast.toml` (av1-svc-fast's levers) in L3T1; libaom layering modes 6 (L3T1) and 9 (L3T3), its own 4:2:1 downscales; simulcast three independent AV1 Default streams. The 360p operating point is decoded at operating point 1 (dav1d), the 180p at operating point 2, each equal to the encoder's reconstruction of that layer (aomdec's for libaom).
- Ours: `presets/av1-svc.toml` / `presets/vp9-svc.toml` (the codec's Default plus `spatial.layers = 2`), `--temporal-layers 1` (L2T1) or `3` (L2T3, with `layers.adaptive=0`), and `presets/av1-svc-fast.toml` / `presets/vp9-svc-fast.toml` (the codec's Fast levers, a 2x2-mean base downscale and a bilinear search proxy) in L2T1; CBR, 1 s buffer. Our temporal layers have no fixed rate split: the rate controller sets each temporal layer's quantizer step relative to the base from a tuned ladder (`layers.cbr_base_step_ratio`, `layers.cbr_middle_step_ratio`). Operating point 0 is scored from the encoder's reconstruction; the base layer is decoded at operating point 1 (AV1: dav1d; VP9: vpxdec on `nanocodec extract --base-only`'s stream) and must equal the encoder's base reconstruction.
- libaom: `examples/svc_encoder_rtc` of the pinned libaom (tools/anchors `aomsvc`), layering mode 5 (L2T1) or 7 (L2T3), CBR with its 1 s buffer (600 ms start/optimal), no lag, cyclic-refresh AQ (`-aq 1`, as WebRTC), no frame drops, one thread, its own 2:1 downscale; L2T3 and L3T3 temporal split 60/80/100 (cumulative, libaom's own test allocation). Decoded by dav1d (scored) and aomdec (equal hashes) at each operating point.
- libvpx: `examples/vp9_spatial_svc_encoder` of the pinned libvpx v1.16.0 (tools/anchors `vpxsvc`), L2T1 only (`-sl 2 -r 1/2,1/1`; no temporal split), one-pass CBR (`--rc-end-usage=1`), no lag, cyclic-refresh AQ (`-aq 3`, as WebRTC's VP9), no frame drops, inter-layer prediction on, one thread, quantizer 2-63. Its layers are all shown frames, so the base-only stream is each superframe's first frame. Decoded by vpxdec (scored) and ffmpeg (equal hashes) at each operating point.
- Simulcast: the codec's Default at 720p and, on the ffmpeg-Lanczos 360p downscale, at 360p, two independent streams with the rate summed per frame.
- Rate validity flags (realuse.rate_valid) are not shown: our points carry no key-loan adjustment here.
- Clips left out for every point (no reliable overlap): static_cam29, pan_bdd_00cef86b, handheld_mot17_10.
- Low rungs (40/75 kbps at 720p30, every point) on: static_otw_homes390, static_cam29, static_london_bus, pan_bdd_00cef86b, bodycam_airport, handheld_mot17_10.

## K-SVC (KEY) modes: AV1 L3T1_KEY and L3T3_KEY

<sub>Measured 2026-11-14 at the merge of feature/svc-key (build 95728f25: every point and every anchor in this section from that one build and today's presets); not yet in the table and chart above, which are from earlier builds.</sub>

In a KEY mode the upper layers predict from the layer below only in key temporal units, so a relay forwards one receiver `{S0..Sk}` on key units and `{Sk}` alone otherwise. Each view is scored at the rate its receiver downloads (`svcbase.av1_ksvc_receiver_bits`): 720p view against AV1 Default coded alone at 720p, 360p and 180p views against AV1 Default coded alone at that size (the three-way simulcast's streams), total against full SVC and against three-way simulcast. BD-J (mean of the PSNR-YUV, VMAF v1 and MS-SSIM BD-rates); the chart's 8 clips, 150 frames, the chart's ladder and low rungs, one key, 1 s buffer. In brackets, all 11 natural VAL clips (the 8 plus static_cam29, pan_bdd_00cef86b and handheld_mot17_10).

| View (BD-J) | L3T1_KEY | L3T3_KEY | full SVC L3T1 | full SVC L3T3 |
|---|---:|---:|---:|---:|
| 720p view vs AV1 720p alone | +4.3% (+3.2%) | +3.9% (+5.4%) | +35.7% (+34.5%) | +30.7% (+32.0%) |
| 360p view vs AV1 360p alone | -1.3% (+0.0%) | -3.1% (-0.3%) | +21.4% (+22.3%) | +18.0% (+20.8%) |
| 180p view vs AV1 180p alone | +1.6% (+4.8%) | -2.3% (+1.9%) | +2.1% (+5.4%) | -2.2% (+2.4%) |
| total vs full SVC L3T1 | -0.3% (-0.0%) | -1.0% (+1.4%) | | -3.8% (-1.6%) |
| total vs full SVC L3T3 | | +3.0% (+3.0%) | | |
| total vs three-way simulcast | -9.8% (-9.5%) | -10.6% (-8.3%) | -9.5% (-9.5%) | -13.1% (-11.0%) |
| share of the total rate: 720p / 360p / 180p view | 77% / 20% / 7% | 77% / 20% / 7% | 100% / 27% / 7% | 100% / 27% / 7% |

Encode instructions per frame (callgrind, the chart's three timing clips lowlight_city_night, crowd_mot17_09, bodycam_bc1 at the 600 kbps rung, all layers) relative to single-layer AV1 Default at 720p: L3T1_KEY 1.53x (1.58 / 1.51 / 1.51), L3T3_KEY 1.41x (1.53 / 1.40 / 1.31), full SVC L3T1 1.87x. **L3T1_KEY misses the 1.5x target** of the brief (the branch reported 1.48x against an older single-layer build).

libaom `svc_encoder_rtc` v3.15.3 in its KEY mode (layering mode 10, L3T3, CBR, the settings in the method below): on the 11 clips its total-rate curve is +110.6% BD-J (rt 10) and +109.1% (rt 8) against our L3T3_KEY (10 of 11 clips overlap). In that version of the example, modes 9 and 10 produce the same streams: the inter-layer (GOLDEN) reference is only switched on in the code that follows `case 11`, which mode 9 never reaches, so its L3T3 is already key-unit-only inter-layer prediction; the L3T3 rows above score it at full-SVC receiver rates, its KEY rows at K-SVC receiver rates. libaom's SVC encoder is a mature, general implementation; ours codes a narrow real-time subset, so this compares two configurations on these clips, not the encoders in general.

## VP9: three layers (L3T1, L3T3) and K-SVC (L3T1_KEY, L3T3_KEY)

<sub>Measured 2026-11-14 at the merge of feature/svc-key (build 463b5f14: every point and anchor in this section from that one build); not yet in the table and chart above.</sub>

`presets/vp9-svc3.toml` (vp9-svc at three layers, 2:1 steps, shares 0.08 / 0.26; K-SVC 0.08 / 0.24) packs each unit as a VP9 superframe `[S0, S1, S2]` with per-layer probability contexts; lower layers, and in K-SVC the top layer too, are hidden frames shown with `show_existing_frame`. Same method as the AV1 K-SVC section: each view at its receiver's own rate against VP9 Default coded alone at that size, the chart's 8 clips (all 11 in brackets). The svc3 presets keep vp9-svc's values from before the VP9 Default CPU cut (88a246f4); the single-layer and size-alone anchors are today's VP9 Default.

| View (BD-J) | L3T1_KEY | L3T3_KEY | full SVC L3T1 | full SVC L3T3 |
|---|---:|---:|---:|---:|
| 720p view vs VP9 720p alone | +3.5% (+3.2%) | +1.4% (+2.9%) | +35.6% (+35.0%) | +30.5% |
| 360p view vs VP9 360p alone | +1.2% (+2.5%) | -1.9% (+1.0%) | +23.6% (+25.3%) | +19.8% |
| 180p view vs VP9 180p alone | +4.8% (**+5.9%**) | +0.9% (+3.9%) | +5.2% (+6.2%) | +0.0% |
| total vs full SVC L3T1 | -1.5% (-0.9%) | -3.9% (-1.8%) | | -4.2% |
| total vs three-way simulcast | -11.8% (-11.4%) | -14.3% (-12.5%) | -10.2% (-10.5%) | -14.5% |
| share of the total rate: 720p / 360p / 180p view | 78% / 20% / 7% | 78% / 20% / 6% | 100% / 27% / 7% | 100% / 27% / 7% |

Encode instructions per frame (callgrind, the three timing clips at the 600 kbps rung) relative to single-layer VP9 Default at 720p: L3T1_KEY 1.64x, L3T3_KEY 1.50x, full SVC L3T1 1.90x. **L3T1_KEY misses the 1.5x target**, and the L3T1_KEY 180p view misses its +5% target on all 11 clips (+5.9%; +4.8% on the chart's 8). Part of the CPU ratio is the anchor: today's VP9 Default is about 12% cheaper on x86 after its CPU cut, which the svc3 presets do not take.

libvpx `vp9_spatial_svc_encoder` v1.16.0 in L3T1_KEY (settings as in the method below, inter-layer prediction on key frames only): its total-rate curve is +95.9% (speed 9) and +96.9% (speed 7) BD-J against our full-SVC L3T1 on the 11 clips (9 overlapping). libvpx's SVC encoder is a mature, general implementation; ours codes a narrow real-time subset, so this compares two configurations on these clips, not the encoders in general.
