# Decoder Yardstick (2026-11-22)

- **Commit:** `038043e0f`
- **Method:** Single-thread decode (`tools/dev/decoder_speed.py`, pinned to CPU 1 via `taskset -c 1`, 5 interleaved forward/reverse rounds, best process wall time, null output / `--noblit`).
- **Stream set:** 21 streams (our `medium` + `slower` 720p encodes + conformance streams per codec)

## Machines

| Target | Instance | CPU | nanocodec |
|---|---|---|---|
| graviton3 | `c7g.large` (eu-west-1) | Neoverse-V1 | `nanocodec 0.6.0 038043e0fc8c74ab2730be53e0dfa55dd69005eb` |
| graviton4 | `c8g.large` (eu-west-1) | Neoverse-V2 | `nanocodec 0.6.0 038043e0fc8c74ab2730be53e0dfa55dd69005eb` |
| zen5 | `c8a.large` (eu-west-1) | AMD EPYC 9R45 | `nanocodec 0.6.0 038043e0fc8c74ab2730be53e0dfa55dd69005eb` |

## Summary: geometric mean decode time ratio (`ours / ref`, lower is faster)

| Codec vs Reference | Group | graviton3 | graviton4 | zen5 |
|---|---|---|---|---|
| av1 vs dav1d | ours-720p | 1.14x | 1.15x | 1.13x |
| av1 vs dav1d | conformance | 1.36x | 1.31x | 1.25x |
| av1 vs dav1d | all | 1.25x | 1.23x | 1.19x |
| av1 wasm-relaxed / native | ours-720p | — | — | 1.60x |
| av1 wasm-relaxed / native | conformance | — | — | 1.47x |
| av1 wasm-relaxed / native | all | — | — | 1.53x |
| av1 wasm-simd128 / native | ours-720p | — | — | 1.57x |
| av1 wasm-simd128 / native | conformance | — | — | 1.49x |
| av1 wasm-simd128 / native | all | — | — | 1.53x |
| av2 vs avmdec | ours-720p | — | — | 0.52x |
| av2 vs avmdec | all | — | — | 0.52x |
| av2 vs dav2d | ours-720p | — | — | 0.46x |
| av2 vs dav2d | all | — | — | 0.46x |
| av2 wasm-relaxed / native | ours-720p | — | — | 1.41x |
| av2 wasm-relaxed / native | all | — | — | 1.41x |
| av2 wasm-simd128 / native | ours-720p | — | — | 1.39x |
| av2 wasm-simd128 / native | all | — | — | 1.39x |
| h264 vs ffmpeg | ours-720p | 0.64x | 0.63x | 0.66x |
| h264 vs ffmpeg | conformance | 1.02x | 0.99x | 1.37x |
| h264 vs ffmpeg | all | 0.74x | 0.73x | 0.84x |
| h264 wasm-relaxed / native | ours-720p | — | — | 1.57x |
| h264 wasm-relaxed / native | conformance | — | — | 1.62x |
| h264 wasm-relaxed / native | all | — | — | 1.59x |
| h264 wasm-simd128 / native | ours-720p | — | — | 1.56x |
| h264 wasm-simd128 / native | conformance | — | — | 1.60x |
| h264 wasm-simd128 / native | all | — | — | 1.57x |
| hevc vs ffmpeg | ours-720p | 0.67x | 0.69x | 0.61x |
| hevc vs ffmpeg | conformance | 0.93x | 0.95x | 0.84x |
| hevc vs ffmpeg | all | 0.79x | 0.81x | 0.72x |
| hevc wasm-relaxed / native | ours-720p | — | — | 1.57x |
| hevc wasm-relaxed / native | conformance | — | — | 1.64x |
| hevc wasm-relaxed / native | all | — | — | 1.59x |
| hevc wasm-simd128 / native | ours-720p | — | — | 1.56x |
| hevc wasm-simd128 / native | conformance | — | — | 1.65x |
| hevc wasm-simd128 / native | all | — | — | 1.59x |
| vp9 vs ffmpeg | ours-720p | 0.77x | 0.77x | 0.72x |
| vp9 vs ffmpeg | conformance | 0.91x | 0.93x | 0.93x |
| vp9 vs ffmpeg | all | 0.83x | 0.85x | 0.82x |
| vp9 vs vpxdec | ours-720p | 1.07x | 1.10x | 1.03x |
| vp9 vs vpxdec | conformance | 0.99x | 1.02x | 0.96x |
| vp9 vs vpxdec | all | 1.03x | 1.06x | 0.99x |
| vp9 wasm-relaxed / native | ours-720p | — | — | 1.38x |
| vp9 wasm-relaxed / native | conformance | — | — | 1.36x |
| vp9 wasm-relaxed / native | all | — | — | 1.37x |
| vp9 wasm-simd128 / native | ours-720p | — | — | 1.47x |
| vp9 wasm-simd128 / native | conformance | — | — | 1.40x |
| vp9 wasm-simd128 / native | all | — | — | 1.43x |
| vvc vs ffmpeg | ours-720p | 0.42x | 0.41x | 0.50x |
| vvc vs ffmpeg | conformance | 0.83x | 0.85x | 0.86x |
| vvc vs ffmpeg | all | 0.59x | 0.59x | 0.66x |
| vvc vs vvdec | ours-720p | 0.75x | 0.74x | 0.74x |
| vvc vs vvdec | conformance | 1.25x | 1.28x | 1.14x |
| vvc vs vvdec | all | 0.97x | 0.97x | 0.92x |
| vvc wasm-relaxed / native | ours-720p | — | — | 1.51x |
| vvc wasm-relaxed / native | conformance | — | — | 1.68x |
| vvc wasm-relaxed / native | all | — | — | 1.59x |
| vvc wasm-simd128 / native | ours-720p | — | — | 1.50x |
| vvc wasm-simd128 / native | conformance | — | — | 1.68x |
| vvc wasm-simd128 / native | all | — | — | 1.59x |

## Per-stream single-thread wall time (ms) and ratio (`ours / ref`)

| Stream | Codec | Group | Ref | graviton3 (ours / ref) | graviton4 (ours / ref) | zen5 (ours / ref) |
|---|---|---|---|---|---|---|
| `otc720p_medium_av1.av1.ivf` | av1 | ours-medium | dav1d | 184.1 / 162.4 ms (**1.13x**) | 156.4 / 137.7 ms (**1.14x**) | 88.2 / 77.8 ms (**1.13x**) |
| `otc720p_medium_av1.av1.ivf` | av1 | ours-medium | wasm-relaxed | — | — | 141.5 / 88.2 ms (**1.60x**, 212 fps) |
| `otc720p_medium_av1.av1.ivf` | av1 | ours-medium | wasm-simd128 | — | — | 137.8 / 88.2 ms (**1.56x**, 218 fps) |
| `otc720p_slower_av1.av1.ivf` | av1 | ours-slower | dav1d | 177.2 / 154.2 ms (**1.15x**) | 153.9 / 132.2 ms (**1.16x**) | 80.7 / 71.3 ms (**1.13x**) |
| `otc720p_slower_av1.av1.ivf` | av1 | ours-slower | wasm-relaxed | — | — | 128.7 / 80.7 ms (**1.59x**, 233 fps) |
| `otc720p_slower_av1.av1.ivf` | av1 | ours-slower | wasm-simd128 | — | — | 126.7 / 80.7 ms (**1.57x**, 237 fps) |
| `otc720p_medium_vp9.vp9.ivf` | vp9 | ours-medium | ffmpeg | 124.6 / 160.5 ms (**0.78x**) | 106.9 / 137.2 ms (**0.78x**) | 67.2 / 92.2 ms (**0.73x**) |
| `otc720p_medium_vp9.vp9.ivf` | vp9 | ours-medium | vpxdec | 124.6 / 116.4 ms (**1.07x**) | 106.9 / 96.8 ms (**1.10x**) | 67.2 / 65.2 ms (**1.03x**) |
| `otc720p_medium_vp9.vp9.ivf` | vp9 | ours-medium | wasm-relaxed | — | — | 92.2 / 67.2 ms (**1.37x**, 325 fps) |
| `otc720p_medium_vp9.vp9.ivf` | vp9 | ours-medium | wasm-simd128 | — | — | 96.2 / 67.2 ms (**1.43x**, 312 fps) |
| `otc720p_slower_vp9.vp9.ivf` | vp9 | ours-slower | ffmpeg | 117.8 / 155.5 ms (**0.76x**) | 100.6 / 133.0 ms (**0.76x**) | 62.0 / 86.6 ms (**0.72x**) |
| `otc720p_slower_vp9.vp9.ivf` | vp9 | ours-slower | vpxdec | 117.8 / 109.2 ms (**1.08x**) | 100.6 / 91.6 ms (**1.10x**) | 62.0 / 60.4 ms (**1.03x**) |
| `otc720p_slower_vp9.vp9.ivf` | vp9 | ours-slower | wasm-relaxed | — | — | 85.4 / 62.0 ms (**1.38x**, 351 fps) |
| `otc720p_slower_vp9.vp9.ivf` | vp9 | ours-slower | wasm-simd128 | — | — | 93.7 / 62.0 ms (**1.51x**, 320 fps) |
| `otc720p_medium_h264.264` | h264 | ours-medium | ffmpeg | 78.7 / 126.0 ms (**0.62x**) | 67.6 / 109.8 ms (**0.62x**) | 44.5 / 68.5 ms (**0.65x**) |
| `otc720p_medium_h264.264` | h264 | ours-medium | wasm-relaxed | — | — | 69.7 / 44.5 ms (**1.57x**, 430 fps) |
| `otc720p_medium_h264.264` | h264 | ours-medium | wasm-simd128 | — | — | 68.5 / 44.5 ms (**1.54x**, 438 fps) |
| `otc720p_slower_h264.264` | h264 | ours-slower | ffmpeg | 82.1 / 126.8 ms (**0.65x**) | 70.4 / 110.9 ms (**0.64x**) | 46.5 / 70.1 ms (**0.66x**) |
| `otc720p_slower_h264.264` | h264 | ours-slower | wasm-relaxed | — | — | 73.3 / 46.5 ms (**1.57x**, 410 fps) |
| `otc720p_slower_h264.264` | h264 | ours-slower | wasm-simd128 | — | — | 73.9 / 46.5 ms (**1.59x**, 406 fps) |
| `otc720p_medium_hevc.hevc` | hevc | ours-medium | ffmpeg | 209.8 / 309.2 ms (**0.68x**) | 177.6 / 256.0 ms (**0.69x**) | 116.6 / 186.2 ms (**0.63x**) |
| `otc720p_medium_hevc.hevc` | hevc | ours-medium | wasm-relaxed | — | — | 182.3 / 116.6 ms (**1.56x**, 165 fps) |
| `otc720p_medium_hevc.hevc` | hevc | ours-medium | wasm-simd128 | — | — | 182.4 / 116.6 ms (**1.56x**, 164 fps) |
| `otc720p_slower_hevc.hevc` | hevc | ours-slower | ffmpeg | 200.1 / 303.8 ms (**0.66x**) | 171.6 / 253.2 ms (**0.68x**) | 108.3 / 182.4 ms (**0.59x**) |
| `otc720p_slower_hevc.hevc` | hevc | ours-slower | wasm-relaxed | — | — | 171.0 / 108.3 ms (**1.58x**, 175 fps) |
| `otc720p_slower_hevc.hevc` | hevc | ours-slower | wasm-simd128 | — | — | 169.2 / 108.3 ms (**1.56x**, 177 fps) |
| `otc720p_medium_vvc.vvc` | vvc | ours-medium | ffmpeg | 320.1 / 765.5 ms (**0.42x**) | 262.0 / 638.5 ms (**0.41x**) | 184.8 / 374.7 ms (**0.49x**) |
| `otc720p_medium_vvc.vvc` | vvc | ours-medium | vvdec | 320.1 / 425.2 ms (**0.75x**) | 262.0 / 351.0 ms (**0.75x**) | 184.8 / 248.5 ms (**0.74x**) |
| `otc720p_medium_vvc.vvc` | vvc | ours-medium | wasm-relaxed | — | — | 276.0 / 184.8 ms (**1.49x**, 109 fps) |
| `otc720p_medium_vvc.vvc` | vvc | ours-medium | wasm-simd128 | — | — | 272.7 / 184.8 ms (**1.48x**, 110 fps) |
| `otc720p_slower_vvc.vvc` | vvc | ours-slower | ffmpeg | 339.2 / 796.8 ms (**0.43x**) | 276.7 / 662.1 ms (**0.42x**) | 197.3 / 385.6 ms (**0.51x**) |
| `otc720p_slower_vvc.vvc` | vvc | ours-slower | vvdec | 339.2 / 455.9 ms (**0.74x**) | 276.7 / 374.2 ms (**0.74x**) | 197.3 / 266.4 ms (**0.74x**) |
| `otc720p_slower_vvc.vvc` | vvc | ours-slower | wasm-relaxed | — | — | 302.6 / 197.3 ms (**1.53x**, 99 fps) |
| `otc720p_slower_vvc.vvc` | vvc | ours-slower | wasm-simd128 | — | — | 301.5 / 197.3 ms (**1.53x**, 100 fps) |
| `conf_av1_av1-1-b8-02-allintra.av1.ivf` | av1 | conformance | dav1d | 212.5 / 156.2 ms (**1.36x**) | 174.8 / 135.0 ms (**1.29x**) | 120.7 / 96.0 ms (**1.26x**) |
| `conf_av1_av1-1-b8-02-allintra.av1.ivf` | av1 | conformance | wasm-relaxed | — | — | 190.9 / 120.7 ms (**1.58x**, 204 fps) |
| `conf_av1_av1-1-b8-02-allintra.av1.ivf` | av1 | conformance | wasm-simd128 | — | — | 195.7 / 120.7 ms (**1.62x**, 199 fps) |
| `conf_av1_av1-1-b8-16-intra_only-intrabc-extreme-dv.av1.ivf` | av1 | conformance | dav1d | 62.5 / 46.1 ms (**1.36x**) | 52.5 / 39.6 ms (**1.32x**) | 34.8 / 27.9 ms (**1.25x**) |
| `conf_av1_av1-1-b8-16-intra_only-intrabc-extreme-dv.av1.ivf` | av1 | conformance | wasm-relaxed | — | — | 47.5 / 34.8 ms (**1.36x**, 42 fps) |
| `conf_av1_av1-1-b8-16-intra_only-intrabc-extreme-dv.av1.ivf` | av1 | conformance | wasm-simd128 | — | — | 47.7 / 34.8 ms (**1.37x**, 42 fps) |
| `conf_vp9_vp90-2-02-size-lf-1920x1080.vp9.ivf` | vp9 | conformance | ffmpeg | 2379.4 / 2293.0 ms (**1.04x**) | 2008.4 / 1865.4 ms (**1.08x**) | 1174.9 / 1125.0 ms (**1.04x**) |
| `conf_vp9_vp90-2-02-size-lf-1920x1080.vp9.ivf` | vp9 | conformance | vpxdec | 2379.4 / 2161.8 ms (**1.10x**) | 2008.4 / 1739.2 ms (**1.15x**) | 1174.9 / 1045.0 ms (**1.12x**) |
| `conf_vp9_vp90-2-02-size-lf-1920x1080.vp9.ivf` | vp9 | conformance | wasm-relaxed | — | — | 1792.9 / 1174.9 ms (**1.53x**, 121 fps) |
| `conf_vp9_vp90-2-02-size-lf-1920x1080.vp9.ivf` | vp9 | conformance | wasm-simd128 | — | — | 1927.2 / 1174.9 ms (**1.64x**, 113 fps) |
| `conf_vp9_vp90-2-09-subpixel-00.vp9.ivf` | vp9 | conformance | ffmpeg | 165.7 / 210.0 ms (**0.79x**) | 149.7 / 185.6 ms (**0.81x**) | 104.0 / 125.5 ms (**0.83x**) |
| `conf_vp9_vp90-2-09-subpixel-00.vp9.ivf` | vp9 | conformance | vpxdec | 165.7 / 186.5 ms (**0.89x**) | 149.7 / 165.2 ms (**0.91x**) | 104.0 / 127.3 ms (**0.82x**) |
| `conf_vp9_vp90-2-09-subpixel-00.vp9.ivf` | vp9 | conformance | wasm-relaxed | — | — | 125.6 / 104.0 ms (**1.21x**, 159 fps) |
| `conf_vp9_vp90-2-09-subpixel-00.vp9.ivf` | vp9 | conformance | wasm-simd128 | — | — | 124.0 / 104.0 ms (**1.19x**, 161 fps) |
| `conf_h264_src19td_IBP.264` | h264 | conformance | ffmpeg | 2318.5 / 2275.7 ms (**1.02x**) | 1953.4 / 1975.3 ms (**0.99x**) | 1322.2 / 965.1 ms (**1.37x**) |
| `conf_h264_src19td_IBP.264` | h264 | conformance | wasm-relaxed | — | — | 2143.4 / 1322.2 ms (**1.62x**, 120 fps) |
| `conf_h264_src19td_IBP.264` | h264 | conformance | wasm-simd128 | — | — | 2114.9 / 1322.2 ms (**1.60x**, 122 fps) |
| `conf_hevc_DELTAQP_B_SONY_3.hevc` | hevc | conformance | ffmpeg | 418.4 / 534.9 ms (**0.78x**) | 364.8 / 442.6 ms (**0.82x**) | 245.8 / 354.6 ms (**0.69x**) |
| `conf_hevc_ENTP_A_Qualcomm_1.hevc` | hevc | conformance | ffmpeg | 745.1 / 671.2 ms (**1.11x**) | 587.1 / 536.7 ms (**1.09x**) | 432.8 / 425.7 ms (**1.02x**) |
| `conf_hevc_ENTP_A_Qualcomm_1.hevc` | hevc | conformance | wasm-relaxed | — | — | 708.7 / 432.8 ms (**1.64x**, 34 fps) |
| `conf_hevc_ENTP_A_Qualcomm_1.hevc` | hevc | conformance | wasm-simd128 | — | — | 715.3 / 432.8 ms (**1.65x**, 34 fps) |
| `conf_vvc_8b420_B_2.vvc` | vvc | conformance | ffmpeg | 1410.2 / 1641.4 ms (**0.86x**) | 1189.9 / 1357.3 ms (**0.88x**) | 607.7 / 707.5 ms (**0.86x**) |
| `conf_vvc_8b420_B_2.vvc` | vvc | conformance | vvdec | 1410.2 / 1074.4 ms (**1.31x**) | 1189.9 / 883.0 ms (**1.35x**) | 607.7 / 536.4 ms (**1.13x**) |
| `conf_vvc_8b420_B_2.vvc` | vvc | conformance | wasm-relaxed | — | — | 1016.8 / 607.7 ms (**1.67x**, 48 fps) |
| `conf_vvc_8b420_B_2.vvc` | vvc | conformance | wasm-simd128 | — | — | 1023.8 / 607.7 ms (**1.68x**, 48 fps) |
| `conf_vvc_8b420_A_2.vvc` | vvc | conformance | ffmpeg | 274.7 / 345.1 ms (**0.80x**) | 234.3 / 286.7 ms (**0.82x**) | 115.8 / 133.3 ms (**0.87x**) |
| `conf_vvc_8b420_A_2.vvc` | vvc | conformance | vvdec | 274.7 / 231.7 ms (**1.19x**) | 234.3 / 193.2 ms (**1.21x**) | 115.8 / 100.5 ms (**1.15x**) |
| `conf_vvc_8b420_A_2.vvc` | vvc | conformance | wasm-relaxed | — | — | 195.7 / 115.8 ms (**1.69x**, 250 fps) |
| `conf_vvc_8b420_A_2.vvc` | vvc | conformance | wasm-simd128 | — | — | 194.3 / 115.8 ms (**1.68x**, 252 fps) |
| `otc720p_medium_av2.av2.ivf` | av2 | ours-medium | avmdec | — | — | 142.1 / 285.0 ms (**0.50x**) |
| `otc720p_medium_av2.av2.ivf` | av2 | ours-medium | dav2d | — | — | 142.1 / 311.3 ms (**0.46x**) |
| `otc720p_medium_av2.av2.ivf` | av2 | ours-medium | wasm-relaxed | — | — | 200.6 / 142.1 ms (**1.41x**, 150 fps) |
| `otc720p_medium_av2.av2.ivf` | av2 | ours-medium | wasm-simd128 | — | — | 197.4 / 142.1 ms (**1.39x**, 152 fps) |
| `otc720p_slower_av2.av2.ivf` | av2 | ours-slower | avmdec | — | — | 139.2 / 260.0 ms (**0.54x**) |
| `otc720p_slower_av2.av2.ivf` | av2 | ours-slower | dav2d | — | — | 139.2 / 295.2 ms (**0.47x**) |
| `otc720p_slower_av2.av2.ivf` | av2 | ours-slower | wasm-relaxed | — | — | 195.8 / 139.2 ms (**1.41x**, 153 fps) |
| `otc720p_slower_av2.av2.ivf` | av2 | ours-slower | wasm-simd128 | — | — | 194.3 / 139.2 ms (**1.40x**, 154 fps) |
