# Server Arm (AWS Graviton4) speed series (2026-11-17)

The results site now has a third speed machine, "Server Arm (Graviton4)", beside the Jetson Orin Nano and the
Zen 5 (c8a) series. It is selectable under "Speed on" on the chart explorer and listed on the speed and method pages.
It covers both charts:

- **Fixed QP (AOM CTC lens):** [fixed-qp-timing-graviton.json](fixed-qp-timing-graviton.json).
- **CBR charts (the speed axis of all three CBR views):** [bd-vs-speed-timing-graviton.json](bd-vs-speed-timing-graviton.json).
- **Not timed here:** the held-out Fixed QP view. Its clips are private and do not go to this machine.

## Method

- **Machine.** AWS c8g.large: Graviton4, Neoverse V2, 2 cores without SMT, eu-west-1, on demand. This is the same size
  and the same method as the c8a.large x86 series: one thread pinned to core 1, core 0 left to the system.
- **SIMD.** `/proc/cpuinfo`: asimd, asimddp, sve, sve2, svei8mm, i8mm, bf16. Our encoder dispatches to its NEON tier,
  its only aarch64 tier. x264 reports "ARMv8 NEON DotProd I8MM SVE SVE2"; SVT-AV1 selects "up to neon_i8mm".
- **Our encoder.** The Jetson's chart build of verified head 2caf7d21 (`nceval.jetson.ensure_build`: release profile, the
  board's toolchain; its stamp is d40f86b3, a docs-only commit on top with the same crates/ and presets/). It was copied
  from the board, so the two Arm series run the same binary.
  - AV1 Veryfast is the exception. Its streams at 2caf7d21 differ from those its plotted quality was measured on
    (2caf7d21 changed its preset), so it was timed on the board's 2860f897 build, its quality build. On the lens it
    ran the 2860f897 lens command lines.
- **References.** The Jetson's aarch64 builds (`~/nanocodec-data/anchors/aarch64/bin` on the board), the same
  versions as x86: SVT-AV1 4.3.0, x264 0.165, x265 4.2, libvpx 1.16.0, libaom 3.15.3, openh264.
  - x265 needs libnuma; the readiness stage installs it when the image lacks it.
  - HM and VVenC have no aarch64 build and are not timed here.
- **Fixed QP lens.** `tools/dev/fixedqp_timing.py plan --machine graviton`, then `run` and `record`, the same way as the
  c8a lens timing (`fixed-qp-speed-2026-11-16.md`):
  - 43 encoders × 12 sequences × 6 quantizers, 3096 encodes, on 29 instances in about 37 min wall;
  - encodes estimated under 20 s took 3 rotated rounds after a warm-up and kept the median; slower ones took 1.
  - Round-to-round spread over the 2387 repeated encodes of 0.2 s or more: largest deviation from the median, median
    0.37 %, 90th percentile 1.20 %.
- **Lens stream identity.** Per encode, the stream bytes match the c8a timing runs' for every encoder of ours (AV1
  Veryfast at 2860f897), for SVT-AV1, libaom, libvpx and openh264, and for x264 ultrafast.
  - x264 veryfast, medium and veryslow and all four x265 presets code other streams on this aarch64 build. Median size
    against x86: x264 medium -0.08 %, x265 medium +1.32 % (one encode +13.5 %), the rest within ±0.35 %.
  - The Jetson series runs the same reference binaries.
- **CBR chart timing.** `tools/dev/aws_timing.py --instance-type c8g.large` (anchor_chart_timing.py `time-cloud`).
  - Every one of the plan's 43 points on the 3 chart clips (150 frames, the plan's quantizers), paired in rotated
    rounds with SVT-AV1 M13 (and x264 medium once per clip), one pair at a time.
  - 5 sessions side by side, one repeat each, plus one for AV1 Veryfast at 2860f897.
  - Merged with `anchor_chart_timing.py merge --machine ... --build 2caf7d21`. The site reads fps straight from that
    record (`nceval.site.cbr_fps`): 1000 over the geometric mean of the per-clip ms per frame, as the chart computes it.
  - Every point's streams equal the plan's (`stream_mismatch` empty).
  - Instance-to-instance difference: AV1 Faster measured 181.2 fps in the pipe-cleaner session and 173.6 in its
    session (4.4 %).
- **Builds differ between the series.** The c8a and Jetson series time our points on their chart builds (lens:
  b11b2f83, 2860f897 for 7 points; CBR: each family's chart build). This series times them at 2caf7d21, which includes
  the exact AV1 writer changes since 2860f897 (e677b1a0f, ce82237f3). The dev CBR view's AV1 Faster quality is from an
  older build than the stream timed here.
- **Cost.** $0.62 in 42 on-demand c8g.large sessions, all terminated, in the ledger.

## Fixed QP lens: fps

The c8a and Jetson columns are the site's stored records. The Jetson timed 2 of the 6 quantizers and not the slow
references.

| Point | c8a (Zen 5) | Jetson | c8g (Graviton4) | c8g/c8a | c8g/Jetson |
|---|---:|---:|---:|---:|---:|
| x264 ultrafast | 1764.5 | 412.2 | 745.1 | 0.42 | 1.81 |
| nanocodec H.264 Veryfast | 1027.2 | 348.4 | 728.4 | 0.71 | 2.09 |
| nanocodec H.264 Faster | 929.7 | 316.6 | 664.0 | 0.71 | 2.10 |
| nanocodec AV1 Veryfast | 1078.3 | 255.7 | 605.0 | 0.56 | 2.37 |
| nanocodec H.264 Fast | 811.8 | 273.0 | 569.6 | 0.70 | 2.09 |
| openh264 | 1049.2 | 258.9 | 542.8 | 0.52 | 2.10 |
| nanocodec VP9 Veryfast | 702.2 | 223.1 | 480.9 | 0.68 | 2.16 |
| SVT-AV1 M13 | 984.1 | 196.1 | 474.4 | 0.48 | 2.42 |
| nanocodec AV1 Faster | 786.4 | 203.4 | 469.9 | 0.60 | 2.31 |
| SVT-AV1 M12 | 899.0 | 178.1 | 431.4 | 0.48 | 2.42 |
| nanocodec VP9 Faster | 684.5 | 195.8 | 428.2 | 0.63 | 2.19 |
| nanocodec HEVC Faster | 627.1 | 182.2 | 398.5 | 0.64 | 2.19 |
| x264 veryfast | 674.5 | 184.2 | 389.6 | 0.58 | 2.12 |
| nanocodec H.264 Medium (default) | 455.7 | 160.3 | 336.1 | 0.74 | 2.10 |
| nanocodec AV1 Fast | 526.5 | 143.7 | 333.6 | 0.63 | 2.32 |
| nanocodec HEVC Fast | 477.4 | 147.4 | 320.8 | 0.67 | 2.18 |
| nanocodec VP9 Fast | 477.6 | 132.7 | 300.4 | 0.63 | 2.26 |
| nanocodec AV2 Fast | 480.9 | 129.2 | 295.7 | 0.61 | 2.29 |
| libvpx VP9 rt 9 | 494.8 | 126.8 | 289.1 | 0.58 | 2.28 |
| SVT-AV1 M10 | 518.9 | 99.0 | 249.0 | 0.48 | 2.52 |
| nanocodec AV1 Medium (default) | 304.9 | 90.7 | 214.5 | 0.70 | 2.36 |
| nanocodec H.264 Slow | 283.6 | 100.4 | 208.3 | 0.73 | 2.07 |
| nanocodec AV2 Medium (default) | 305.7 | 87.4 | 194.5 | 0.64 | 2.23 |
| nanocodec VVC Fast | 277.6 | 88.4 | 192.5 | 0.69 | 2.18 |
| nanocodec VP9 Medium (default) | 284.1 | 83.5 | 186.5 | 0.66 | 2.23 |
| libvpx VP9 rt 7 | 310.9 | 80.3 | 184.2 | 0.59 | 2.29 |
| nanocodec VP9 Slow | 249.6 | 73.5 | 163.4 | 0.65 | 2.22 |
| nanocodec HEVC Medium (default) | 232.0 | 78.2 | 159.2 | 0.69 | 2.04 |
| libaom rt 10 | 224.8 | 57.2 | 144.4 | 0.64 | 2.52 |
| nanocodec AV1 Slow | 192.1 | 56.8 | 130.0 | 0.68 | 2.29 |
| libaom rt 8 | 166.7 | 41.2 | 108.1 | 0.65 | 2.62 |
| nanocodec VVC Medium (default) | 158.4 | 47.5 | 104.9 | 0.66 | 2.21 |
| x264 medium | 186.0 | 50.6 | 103.8 | 0.56 | 2.05 |
| x265 ultrafast | 182.2 | 38.9 | 93.2 | 0.51 | 2.40 |
| nanocodec VVC Slow | 130.8 | 41.0 | 91.8 | 0.70 | 2.24 |
| SVT-AV1 M8 | 168.8 | – | 80.1 | 0.47 | – |
| nanocodec AV2 Slow | 120.1 | 42.2 | 78.3 | 0.65 | 1.86 |
| nanocodec HEVC Slow | 102.5 | 36.6 | 69.2 | 0.68 | 1.89 |
| x265 superfast | 118.7 | 27.2 | 63.8 | 0.54 | 2.35 |
| x265 medium | 73.9 | 17.3 | 40.2 | 0.54 | 2.32 |
| x264 veryslow | 41.6 | – | 24.5 | 0.59 | – |
| x265 slow | 24.9 | – | 13.7 | 0.55 | – |
| HM (reference model), low delay P | 0.7 | – | – | – | – |
| VVenC faster | 17.1 | – | – | – | – |

## CBR charts: fps (720p, 3 chart clips)

The x86 and Jetson columns are the held-out fresh view's (the site's default view).

| Point | c8a (Zen 5) | Jetson | c8g (Graviton4) | c8g/c8a | c8g/Jetson |
|---|---:|---:|---:|---:|---:|
| x264 ultrafast | 1017.0 | 190.5 | 363.0 | 0.36 | 1.91 |
| nanocodec H.264 Veryfast | 503.6 | 153.2 | 356.7 | 0.71 | 2.33 |
| nanocodec H.264 Faster | 462.9 | 135.4 | 309.4 | 0.67 | 2.29 |
| openh264 | 524.6 | 111.5 | 262.5 | 0.50 | 2.36 |
| nanocodec H.264 Fast | 377.1 | 113.3 | 252.3 | 0.67 | 2.23 |
| nanocodec HEVC Faster | 409.9 | 98.9 | 232.3 | 0.57 | 2.35 |
| nanocodec VP9 Veryfast | 334.9 | 96.5 | 222.5 | 0.66 | 2.31 |
| nanocodec AV1 Veryfast | 447.1 | 91.6 | 221.5 | 0.50 | 2.42 |
| SVT-AV1 M13 | 445.4 | 79.2 | 217.9 | 0.49 | 2.75 |
| x264 veryfast | 363.3 | 85.4 | 207.7 | 0.57 | 2.43 |
| nanocodec VP9 Faster | 328.1 | 86.5 | 201.7 | 0.61 | 2.33 |
| SVT-AV1 M12 | 408.6 | 70.9 | 199.5 | 0.49 | 2.81 |
| nanocodec HEVC Fast | 303.5 | 78.0 | 183.6 | 0.60 | 2.35 |
| nanocodec AV1 Faster | 327.4 | 70.2 | 173.6 | 0.53 | 2.47 |
| nanocodec H.264 Medium (default) | 235.5 | 71.9 | 159.0 | 0.68 | 2.21 |
| nanocodec AV1 Fast | 250.8 | 59.6 | 145.1 | 0.58 | 2.44 |
| nanocodec VP9 Fast | 205.1 | 54.1 | 128.7 | 0.63 | 2.38 |
| libvpx VP9 rt 9 | 204.4 | 47.8 | 123.8 | 0.61 | 2.59 |
| nanocodec VVC Fast | 190.5 | 49.4 | 117.8 | 0.62 | 2.38 |
| nanocodec AV2 Fast | 207.7 | 44.6 | 114.5 | 0.55 | 2.56 |
| nanocodec HEVC Medium (default) | 179.9 | 46.6 | 109.8 | 0.61 | 2.36 |
| SVT-AV1 M10 | 228.9 | 39.3 | 103.5 | 0.45 | 2.63 |
| nanocodec H.264 Slow | 140.1 | 45.7 | 101.8 | 0.73 | 2.23 |
| nanocodec AV1 Medium (default) | 161.0 | 40.9 | 101.8 | 0.63 | 2.49 |
| nanocodec VP9 Medium (default) | 165.6 | 40.7 | 97.6 | 0.59 | 2.40 |
| nanocodec AV2 Medium (default) | 140.8 | 32.5 | 82.9 | 0.59 | 2.55 |
| nanocodec VP9 Slow | 131.2 | 34.5 | 82.4 | 0.63 | 2.39 |
| libvpx VP9 rt 7 | 122.7 | 30.0 | 74.8 | 0.61 | 2.49 |
| nanocodec HEVC Slow | 114.0 | 28.9 | 67.5 | 0.59 | 2.33 |
| nanocodec AV1 Slow | 112.8 | 27.2 | 67.0 | 0.59 | 2.46 |
| nanocodec VVC Medium (default) | 109.3 | 27.0 | 65.9 | 0.60 | 2.44 |
| x264 medium | 112.8 | 26.7 | 64.0 | 0.57 | 2.40 |
| libaom rt 10 | 90.5 | 22.3 | 60.9 | 0.67 | 2.73 |
| nanocodec VVC Slow | 89.3 | 23.3 | 56.9 | 0.64 | 2.44 |
| libaom rt 8 | 68.8 | 17.4 | 46.9 | 0.68 | 2.70 |
| nanocodec AV2 Slow | 71.4 | 19.3 | 45.1 | 0.63 | 2.34 |
| x265 ultrafast | 85.1 | 17.1 | 44.9 | 0.53 | 2.62 |
| SVT-AV1 M8 | 80.8 | 14.3 | 38.0 | 0.47 | 2.66 |
| x265 superfast | 57.9 | 12.1 | 31.7 | 0.55 | 2.61 |
| x265 medium | 38.0 | 8.0 | 20.0 | 0.53 | 2.50 |

## Positions against SVT-AV1 M10 / M12 / M13

"Ahead of" means faster and fewer bits, both at once.

- **Fixed QP, Weighted.**
  - AV1 Faster (-5.7 at 469.9 fps) is ahead of M12 (-1.5 at 431.4) on Graviton4. On c8a it is not (786.4 against
    899.0 fps).
  - On the Jetson it is ahead of M12 and M13; on Graviton4 not of M13 (474.4 fps).
  - Nothing else of ours is ahead of an SVT point on any of the three machines.
- **Fixed QP, J.**
  - As on c8a, AV1 Fast is ahead of M10 and AV1 Veryfast of M13.
  - Also ahead on Graviton4: AV1 Faster of M12 (-18.3 against -10.4) and AV2 Fast of M10 (-20.6 at 295.7 fps, against
    -19.5 at 249.0). The Jetson has both too, and AV1 Faster also ahead of M13.
- **CBR, held-out fresh view (J against x264 medium --tune psnr).**
  - Every position of ours on c8a also holds on Graviton4.
  - New on Graviton4: HEVC Default, VP9 Fast, AV2 Fast and VVC Fast ahead of M10; HEVC Faster ahead of M13; VP9 Faster
    ahead of M12; VP9 Veryfast ahead of M12 and M13.
  - Some of those margins are within the 4.4 % instance-to-instance difference above: VP9 Veryfast against M13
    (222.5 against 217.9 fps, 2.1 %) and VP9 Faster against M12 (201.7 against 199.5, 1.1 %).
  - The Jetson has more: AV1 and VP9 Default ahead of M10, H.264 Default ahead of M12, HEVC Fast ahead of M12, VP9
    Faster ahead of M13.
  - The all-held-out view shows the same pattern.
