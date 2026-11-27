# Metric validation (2026-11-23)

Checks the charts' scorers against reference tools on stored chart points (audit finding 7,
`docs/analysis/results-audit-2026-11-23.md`). Per-point numbers: [metric-validation-2026-11-23.csv](metric-validation-2026-11-23.csv).

## What the charts compute

- **VMAF:** libvmaf's model `vmaf_v1.0.16_3d0h` (the model file of Netflix/vmaf v3.2.1, commit f85a853; CAMBI
  told the encode size), on every frame, at each clip's own 8-bit 4:2:0 size, with no rescaling. The pooled score
  is the mean over frames. This is an internal, relative protocol. The model is calibrated for a 1080p display, so
  the values are not comparable with published 1080p-protocol VMAF.
- **Who scores VMAF:** VMAFx on CUDA (a libvmaf fork, patched; `eval/nceval/metrics.py`) for points scored on the PC.
  libvmaf v3.2.1 on the CPU for points scored on rented boxes (chart refreshes since 2026-11-21, held-out and VC-1K
  charts). Stored points do not record which one scored them.
- **PSNR:** per frame and per plane from exact squared-error sums, 10·log10(255²/MSE), capped at 60 dB. Averaged
  over frames per plane, then combined 6:1:1 (Y:U:V). Our points scored on the PC take the sums from our encoder's
  `--sse-log` on its own reconstruction. Points scored on boxes, and all reference encoders' points, compute them
  in numpy from a standard decoder's output. Our streams scored on boxes are first checked to decode bit-exactly
  to the encoder's reconstruction digests.
- **Reference frames:** a raw clip is read as stored. An encoded clip (VC-1K, the independent sets) is decoded by
  the first `ffmpeg` that `nceval.config.find_tool` finds. See the finding below.

## Sample

143 stored points from three published charts' kept streams (`work/heldout-chart/streams-*`), every family on its
recorded build:

- **BD set:** three VC-1K clips (720×1280 and 1280×720, 30 fps). Six rungs each (40–1200 kbps) of nanocodec AV1,
  HEVC and H.264, and of SVT-AV1 M12, x265 medium and x264 medium. 108 points.
- **Spread:** 35 points.
  - VC-1K and the main chart's 720p clips: every class.
  - The fresh independent set, 240p to 720p: 240, 270, 360, 480, 540, 640 and 720 lines.
  - Families: also nanocodec VP9, VVC and AV2, libvpx-vp9 cpu7, libaom cpu8 and openh264.
  - Rungs rotate lowest / middle / highest.
- **Decoders:** ffmpeg (H.264/HEVC), ffmpeg-dav1d, ffmpeg native VP9 and VVC, avmdec. 77 of our points were each
  checked to equal the encoder's reconstruction digests on every frame.
- **Re-encoded for the encoder's PSNR:** 68 of our points, re-encoded by their recorded build with `--sse-log`. 65
  reproduced the kept stream byte for byte. The other 3 are kept streams of an unrecorded build and are left out.
- **Stored results:** 108 points have one. It is matched to the stream by its frame bits.

## Results

| Check | Points | Max abs difference |
|---|---:|---|
| VMAF v1, VMAFx CUDA vs libvmaf 3.2.1 CPU, per frame | 143 | 0 at libvmaf's printed 6 decimals |
| VMAF v1, per sequence | 143 | 0 |
| VMAF v1, stored chart value vs libvmaf now, per frame | 108 | 0 |
| MS-SSIM, VMAFx vs libvmaf/nceval.msssim, per frame / sequence | 143 | 2.1e-5 / 1.3e-6 |
| PSNR Y, U, V per frame: encoder `--sse-log` vs libvmaf `psnr` feature | 65 | 5.0e-7 dB (libvmaf prints 6 decimals) |
| PSNR-YUV 6:1:1 per sequence: encoder vs libvmaf | 65 | 7.4e-8 dB |
| PSNR Y, U, V per frame: harness numpy vs libvmaf | 143 | 5.0e-7 dB |
| PSNR-YUV stored chart value vs libvmaf now | 108 | 5.6e-8 dB |
| BD-VMAF of the three pairs on the three BD clips (9), VMAFx vs libvmaf | 9 | 0.000 percentage points |

All of these used the same reference frames on both machines (ffmpeg 8.0.1; see below). VMAFx and libvmaf agree
exactly. The encoder's PSNR equals an independent tool's to print precision. The stored chart values are
reproduced.

## Finding: the reference decode depends on the ffmpeg on `PATH`

For encoded clips the reference frames come from whichever `ffmpeg` is first on `PATH`. No pinned binary is used
(`anchors/bin` holds none named `ffmpeg`).

- **The two ffmpeg builds:** the PC's agent shells find a conda ffmpeg 7.1.1 first. The build box and `/usr/bin`
  have 8.0.1.
- **What differs:** 3 of the sample's 38 clips decode to different frames, all VC-1K phone clips: …3685067937,
  …3688492094 and …3688853878. The other 35 decode identically.
- **VMAF:** on those clips the same streams score up to 30.7 VMAF lower against the 7.1.1 decode, per sequence. The
  frames are misaligned.
- **BD-VMAF:** on clip …3685067937 the pairs move by −3.0 (AV1 vs SVT-AV1 M12), −1.9 (HEVC vs x265) and +3.6
  (H.264 vs x264) percentage points.
- **Stored charts:** the stored VC-1K values match the 8.0.1 decode exactly, on all 108 stored points.
- **Risk:** a VC-1K or independent-set point scored on the PC from a shell with the conda ffmpeg first would be
  scored against different reference frames. Its cache key would be the same.

The main chart's clips are raw YUV and are not affected. Pinning the clip decoder (an anchors ffmpeg, or the
decode's version in the clip's identity) is the fix. It is not made here.

## Re-running it

```sh
cd eval
python3 -m nceval.metricval sample --out SAMPLE \
  --chart ~/nanocodec-data/clips/vc1k_2026-11-18.json ~/nanocodec-data/work/heldout-chart/streams-vc1k_2026-11-18 ../docs/results/bd-vs-speed-vc1k_2026-11-18-builds.json \
  --chart ~/nanocodec-data/work/chart-refresh/main-val.json ~/nanocodec-data/work/heldout-chart/streams-main-val ../docs/results/bd-vs-speed-builds.json \
  --chart ~/nanocodec-data/clips/independent_fresh_2026-11-13.json ~/nanocodec-data/work/heldout-chart/streams-independent_fresh_2026-11-13 ../docs/results/bd-vs-speed-independent_fresh_2026-11-13-builds.json \
  --results ~/nanocodec-data/work/heldout-chart/cloud-results/f150-ip300 --results ~/nanocodec-data/work/realuse/f150-ip300 --spread 12
# CPU pass on the build box (send the sample's clips with --clip, sync SAMPLE, the builds and anchors/bin/vpxdec):
nc-remote run --sync SAMPLE ... -- 'cd eval && python3 -m nceval.metricval score --sample SAMPLE --backend libvmaf-cpu --encoder-psnr --workers 24 --out ../remote-out/cpu.json'
# GPU pass here, with the box's ffmpeg first on PATH so both read the same reference frames:
python3 -m nceval.metricval score --sample SAMPLE --backend vmafx-cuda --workers 4 --out gpu.json
python3 -m nceval.metricval compare --sample SAMPLE --cpu cpu.json --gpu gpu.json --out-csv OUT.csv --out-json OUT.json
```

- **What it writes:** the sample is ~40 MB of streams and side files. Every decode and reference goes through pipes,
  so no YUV file is written.
- **Cost (2026-11-23):** the CPU pass took 3.5 min on the shared box with 24 workers. The GPU pass took 2 min on
  the PC.
- **Reading the compare output:**
  - `compare` counts CPU-vs-GPU differences only where both passes read the same reference, and names the clips
    that decoded differently.
  - It compares the encoder's PSNR only where the re-encode reproduced the kept stream.
- **This run:** 2026-11-23, PC RTX 5090 (VMAFx `ext/vmafx-serve6`), shared AWS build box (libvmaf v3.2.1
  f85a853, ffmpeg 8.0.1).
