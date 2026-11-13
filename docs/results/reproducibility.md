# Reproducing our measurements

This page lists what you need to reproduce our numbers, and what you cannot reproduce from outside the project.
It covers three runs:

- (a) one headline number of the published chart ([bd-vs-speed.md](bd-vs-speed.md));
- (b) the independent test of 2026-11-13 ([independent-2026-11-13.md](independent-2026-11-13.md));
- (c) a small run on openly licensed clips that ship in this repository, which anyone can run end to end. Its expected
  numbers are given below.

We followed section (c) ourselves from a clean clone of this repository (2026-11-13). The [log](#what-we-ran-from-a-clean-clone)
at the end lists what broke and what we fixed.

**What cannot be reproduced exactly from outside:** the 11 chart clips and the 209 independent-test clips are cut from
third-party datasets that we do not redistribute (see [Clips](#clips-and-licences)). You can obtain most of those
datasets yourself and cut the same excerpts with the selection files named below, but some sources (YouTube
uploads, internal captures) are not public. Run (c) is the fully public path.

## What a measurement is

All three runs use the same harness (`eval/nceval`, Python 3.12, stdlib + numpy/scipy/opencv/xxhash):

- **Real-use CBR ladder.** Every encoder runs its own real-time rate control at targets of 150, 300, 600 and
  1200 kbps for 720p30. Targets are scaled by pixel rate (exponent in `eval/nceval/sweep.py`) and limited to half
  the clip's source bitrate. On clips where some encoder's curve has no reliable overlap with the anchor's, every
  encoder also gets 40 and 75 kbps rungs (the "low rungs", `nceval.chart.low_rung_clips`).
- **Stream settings.** 150 frames from frame 0, one key frame, 1 s buffer. Command lines per encoder are in
  `eval/nceval/cbr.py` (`Family.command`); our encoder runs `nanocodec encode --bitrate K --buffer-ms 1000` with the
  preset's parameters.
- **Metrics, per frame.** PSNR per plane, combined 6:1:1 into PSNR-YUV. VMAF v1 with the `vmaf_v1.0.16` 3d0h model
  at the clip's native size (libvmaf v3.2.1, Netflix's model file; not the 1080p-upscaled protocol, so not
  comparable with published 1080p VMAF). MS-SSIM (libvmaf's `float_ms_ssim` definition) on every frame.
  `eval/nceval/metrics.py` defines all three.
- **Rate.** The rate axis is the achieved rate: the stream's packet bytes as ffprobe reports them, times fps over
  the frame count. Container overhead is not counted: no IVF file and frame headers, no MP4 boxes. Annex-B start
  codes, parameter sets and SEI (x264's options string included) are counted, because they are inside the packets.
- **BD-rate.** PCHIP over log rate (`eval/nceval/bdrate.py`), on achieved rates. It uses only the curves' reliable
  overlap and never extrapolates.
- **BD-J.** Per clip, the mean of the full-overlap BD-rate and the lower-half BD-rate, each averaged over PSNR-YUV,
  VMAF v1 and MS-SSIM (MS-SSIM as −10·log10(1−x)). Across clips, the class-balanced mean: the mean over content
  classes of each class's mean.
- **Intervals.** 95% percentile bootstrap: 2,000 draws, clips resampled within their class, seed 20261113
  (`eval/nceval/independent.py`).
- **Scoring source.** Our points on the chart are scored on the encoder's reconstruction. Reference encoders, and
  every point of runs (b), are scored on a standard decoder's output; for our streams that output must equal the
  encoder's reconstruction frame by frame (xxh3 hashes).

## (a) One headline chart number: AV1 Default, BD-J −50.87% vs x264 medium

| What | Value |
|---|---|
| Commit | the build recorded for `nanocodec-av1` in `docs/results/bd-vs-speed-builds.json` (e122b42b). The current verified head e1c3d232 (encoder sources equal cba4292f) gives the same BD-J on the same clips: −50.9 [−56.5, −45.3] |
| Toolchain | rustc 1.99.1 (stable), `cargo build --profile gate` for scoring (release codegen without LTO; streams are byte-identical to the release and PGO builds) |
| Anchor | x264 0.165 (stable branch, b35605a), `--preset medium --tune zerolatency --bframes 0 --rc-lookahead 0 --no-scenecut`, ABR with a 1 s VBV, `--threads 1` |
| Clips | the 11 natural VAL clips of `eval/clipset/main.json` (`split: val`): sources and start times are in that file |
| Hardware | quality does not depend on hardware (bit-exact streams). Speed was measured on a Ryzen 9 5950X and a Jetson Orin Nano; it is not part of this number |

Steps:

```sh
git clone https://github.com/cribbinsb/nanocodec && cd nanocodec
cargo build --profile gate -p nc-cli                      # target/gate/nanocodec
tools/anchors/build_anchors.sh python x264 svtav1 libaom dav1d jm hm vmaf ffmpeg
# the 11 VAL clips: obtain each source in eval/clipset/main.json, then cut them (raw YUV)
python3 eval/clipset/extract.py eval/clipset/main.json      # -> $NC_DATA_ROOT/clips/main/*.yuv
mkdir -p build && cp target/gate/nanocodec build/ && cp -r presets build/
cd eval && NC_METRICS_BACKEND=libvmaf-cpu python3 -m nceval.chart measure \
    --binary ../build/nanocodec --presets-dir ../build/presets --points x264-medium,nanocodec-av1
cd .. && python3 tools/dev/independent_report.py --chart --build build --tables bd --out out
```

`NC_METRICS_BACKEND=libvmaf-cpu` scores with stock libvmaf on the CPU. Our own runs use a GPU build of the same
metrics (VMAFx). [The cross-check](rate-control-audit-2026-11-13.md#metrics-gpu-vs-cpu) gives the differences.

What is missing for an outsider: `bodycam_airport` is a YouTube upload, and `static_cam29` and `static_london_bus`
come from internal test collections. These three sources are not public, so (a) on the exact 11 clips is
reproducible only inside the project.

## (b) The independent test (2026-11-13)

| What | Value |
|---|---|
| Our build | verified head e1c3d232; scoring build `charts/builds/cba4292f` (encoder sources identical) |
| References | x264 0.165 (b35605a), x265 4.2, SVT-AV1 v4.3.0, libvpx v1.16.0, libaom v3.15.3, openh264 v2.7.0; pinned in `tools/anchors/versions.env` |
| Clips | 209: 71 screening-pool clips (`heldout_chart_2026-11-12`) and 138 never-used sources; list and SHA-256 in [independent-2026-11-13-clips.csv](independent-2026-11-13-clips.csv) |
| Encode | AWS c7a.metal-48xl spot (AMD EPYC 9R14), eu-north-1: three sessions, 50,142 + 4,848 encodes |
| Scoring | AWS spot, CPU metrics backend (libvmaf v3.2.1 + `nceval.msssim`), three sessions side by side |
| Cost | $8.01 of AWS spot in all ([the run's page](independent-2026-11-13.md#how-it-was-run)) |

Steps:

```sh
# 1. never-used sources (needs the inventory of the source library, eval/inventory)
python3 eval/clipset/select_independent.py --out eval/clipset/independent.json --count 150
python3 eval/clipset/build_pool.py eval/clipset/independent.json
python3 eval/clipset/grade.py independent
# 2. the clip set: yesterday's 71 held-out clips first, then the clean, usable fresh clips
python3 tools/dev/aws_heldout_chart.py select --source-set independent \
    --include $NC_DATA_ROOT/clips/heldout_chart_2026-11-12.json --count 1000 \
    --clipset $NC_DATA_ROOT/clips/independent_2026-11-13.json
# 3. encodes on a rented machine (bitstreams, rate logs and recon digests come back)
R=x264-faster,svtav1-m9,svtav1-m11,x264-medium~psnr,x264-medium~ssim,x265-medium~psnr,x265-medium~ssim,libaom-av1-cpu8~ssim,libvpx-vp9-cpu7~ssim
python3 tools/dev/aws_heldout_chart.py run --clipset $NC_DATA_ROOT/clips/independent_2026-11-13.json \
    --streams STREAMS --build BUILD --references $R --extras 30
# 4. scoring on rented machines (or `score` here), sessions side by side
python3 tools/dev/aws_heldout_chart.py score-cloud --clipset ... --streams STREAMS --build BUILD \
    --references $R --extras 30 --session 1/3          # and 2/3, 3/3
# 5. the tables
python3 tools/dev/independent_report.py --clipset ... --streams STREAMS --build BUILD \
    --references $R --extras 30 --out OUT
```

The audit's other runs ([rate-control-audit-2026-11-13.md](rate-control-audit-2026-11-13.md)):

```sh
# chart clips: extras at the current build, metric-tuned references (scored here), then the tables
python3 tools/dev/independent_report.py --chart --chart-extras --measure --build BUILD \
    --references x264-medium~psnr,x264-medium~ssim,x265-medium~psnr,x265-medium~ssim,libaom-av1-cpu8~ssim,libvpx-vp9-cpu7~ssim,openh264-baseline~aq \
    --tables bd,audit,variants --out OUT
python3 tools/dev/independent_report.py --chart --build BUILD --tables decode-check --out OUT   # full gate on 144 points
python3 tools/dev/independent_report.py --clipset ... --streams STREAMS --build BUILD --references $R \
    --extras 30 --tables gpu-check --session 1/3 --out OUT                                     # 200 points on the GPU path
# window length: the `long` set at 150 and 600 frames
python3 tools/dev/independent_report.py --local --measure --frames 600 --clipset $NC_DATA_ROOT/clips/long.json \
    --build BUILD --families x264-medium,x264-medium~psnr,nanocodec-h264,nanocodec-hevc,nanocodec-av1 --out OUT
# CBR vs fixed-QP CPU time (rented bare metal)
python3 tools/dev/aws_timing.py --points <our chart families> --cbr-variants --repeats 1 --rounds 5 --binary PGO_BUILD
```

## (c) The open subset: anyone, end to end

Ten 720p clips that ship with this repository (the browser demo's samples, `web/clips/`) under licences that
allow redistribution: Blender open movies (CC BY 3.0), SVT Open Content 2022 (CC BY 4.0), and Xiph.org's
Vidyo/JCT-VC conferencing sequences (public domain). Attribution is in
[web/clips/ATTRIBUTION.md](../../web/clips/ATTRIBUTION.md). `tools/dev/open_subset.py` lists them.

These clips are H.264 CRF 22-28 encodes, so the 50%-of-source rule leaves most of them only the 150 and 300 kbps
rungs (plus the low rungs). This run checks the pipeline. Its numbers are not comparable with the chart's.

```sh
git clone https://github.com/cribbinsb/nanocodec && cd nanocodec
cargo build --release -p nc-cli                                  # ~6 min on 4 cores
tools/anchors/build_anchors.sh --prefix ANCHORS --jobs 4 python x264 svtav1 libaom dav1d jm vmaf ffmpeg
mkdir -p build && cp target/release/nanocodec build/ && cp -r presets build/
export NC_DATA_ROOT=$PWD/../ncdata NC_ANCHORS=ANCHORS NC_METRICS_BACKEND=libvmaf-cpu \
       NC_MIN_FREE_GIB=20 NC_CPU_SLOTS=4
mkdir -p $NC_DATA_ROOT/clips
python3 tools/dev/open_subset.py --out $NC_DATA_ROOT/clips/open_subset.json
python3 tools/dev/independent_report.py --local --measure --clipset $NC_DATA_ROOT/clips/open_subset.json \
    --build build --families x264-medium,nanocodec-h264,nanocodec-hevc,nanocodec-av1,svtav1-m12 \
    --workers 4 --tables bd,audit --out out
cat out/bd.md
```

Requirements: Linux x86-64, Rust stable, a C/C++ toolchain, cmake, nasm, meson (installed by
`build_anchors.sh python`), git, ffmpeg/ffprobe on PATH (to read the MP4 clips), Python 3.10+ with numpy, scipy,
opencv-python and xxhash. Disk: about 3 GB. `NC_MIN_FREE_GIB` lowers the harness's free-space floor (200 GiB by
default, which protects the project's shared machine).

Expected result (our run of exactly these commands from a clean clone, 2026-11-13, commit 231a145a; 210 points,
35 min on 4 cores of a Ryzen 9 5950X). BD vs x264 medium in %, with [95% bootstrap over clips]. "clips (J)" counts
the clips with enough curve overlap for a BD. On the four static conferencing clips, our HEVC and AV1 curves have
no reliable overlap with x264's at these rates, so those two rows average 5 of 10 clips. The chart column is the
published chart's value on its own clips, for orientation only.

| Family | BD-J | PSNR-Y | PSNR-YUV | VMAF v1 | MS-SSIM | clips (J) | chart BD-J |
|---|---|---|---|---|---|---|---|
| HEVC | -50.1 [-54.8, -45.5] | -53.8 [-57.3, -50.4] | -53.8 [-56.6, -50.9] | -57.7 [-61.7, -53.5] | -53.1 [-56.8, -49.4] | 5/10 | -49.26 |
| AV1 | -50.1 [-54.3, -45.9] | -57.3 [-60.9, -53.7] | -54.9 [-58.3, -51.6] | -54.4 [-62.0, -47.1] | -56.4 [-59.7, -53.2] | 5/10 | -50.87 |
| H.264 | -37.9 [-44.6, -30.1] | -35.1 [-42.2, -27.6] | -39.2 [-45.3, -31.8] | -39.0 [-46.3, -30.9] | -35.5 [-42.7, -26.9] | 10/10 | -23.04 |
| SVT-AV1 M12 | -36.6 [-49.8, -23.9] | -43.5 [-53.6, -33.3] | -33.7 [-48.9, -18.9] | -35.5 [-48.6, -22.6] | -40.7 [-51.6, -29.8] | 10/10 | -34.91 |

BD-J per class:

| Family | pan (4) | static (4) | texture (2) |
|---|---|---|---|
| H.264 | -22.5 | -46.3 | -45.0 |
| HEVC | -48.8 | n/a | -51.5 |
| AV1 | -51.3 | n/a | -48.9 |
| SVT-AV1 M12 | -33.2 | -50.6 | -26.0 |

## Clips and licences

| Set | Clips | Sources | Redistributable |
|---|---:|---|---|
| Chart (VAL) | 11 | OTW, BDD100K-MOT, JAAD, PersonPath22, MOT17/20 (public research datasets under their own terms), a YouTube upload, two internal clips | no: test inputs only |
| Independent, pool part | 71 | 21 datasets incl. MOT, BDD100K, JAAD, MEVA, CHIRLA, CEVO, Antare body-cam, Xiph/Netflix/SVT open content | partly (the open-content clips) |
| Independent, fresh part | 138 | PersonPath22, JAAD, BDD100K-MOT, BWC-videotext, OTW, CEVO, MEVA, CCTV and others | no: test inputs only |
| Open subset | 10 | Blender, SVT Open Content, Xiph Vidyo | yes (in `web/clips/`) |

## Hardware, run time, cost

| Run | Machine | Time | Cost |
|---|---|---|---|
| Chart quality | AMD Ryzen 9 5950X + RTX 5090 (VMAFx) | about 2 h per full refresh | none |
| Independent encodes | AWS c7a.metal-48xl spot (EPYC 9R14, 192 vCPU) | 50,142 + 4,848 encodes: 45 min (Python-thread runner, GIL-bound, stopped) + 2.5 min + 1.4 min (process runner) | $2.45 |
| Independent scoring | AWS c7a.metal-48xl spot (and two c7a.24xlarge lost to reclaims) | 41,642 points; ~1 h per third of the clips on one metal instance | $5.39 |
| Open subset | 4 cores of a Ryzen 9 5950X | build 6.5 min, anchors 4 min, points 35 min | none |

## Known sources of variation

- **Our encoder is deterministic.** The same build gives byte-identical streams on any x86-64 machine. The
  cloud's streams equalled local encodes on every clip we checked, and the chart's builds equal the current build
  on the chart clips.
- **Reference encoders are deterministic** at the settings we use (one thread). x264 with several sliced
  threads, x265 with WPP, and NVENC are not guaranteed to be.
- **VMAF on the GPU (VMAFx) is not bit-reproducible.** On 144 chart points re-scored from identical decoded
  pictures, 8 sequence VMAF values moved by up to 0.04 points, and single frames by up to 5.5 points.
  PSNR and MS-SSIM were identical. CPU libvmaf versus VMAFx: see
  [the cross-check](rate-control-audit-2026-11-13.md#metrics-gpu-vs-cpu).
- **Clip sets drift.** Grading and pool changes alter which clips a selection script returns: yesterday's
  `select` today returns 72 clips, of which 69 are the same. Every run therefore records its clip set file and the
  stored clips' SHA-256.
- **Speed** depends on the machine, its load and its boost clock. Fps is reported only with its hardware and is
  never mixed across machines.

## Checklist

- [ ] The commit and build: `nanocodec --version` prints the commit.
- [ ] Anchor versions: `tools/anchors/versions.env` (tag + commit; `build_anchors.sh` refuses a mismatch).
- [ ] The clip set file and its clips' SHA-256 (`stored_sha256` in the clips CSV).
- [ ] Metrics backend (`NC_METRICS_BACKEND`), the VMAF model file (`vmaf_v1.0.16_3d0h.json`) and libvmaf v3.2.1.
- [ ] The frames (150, from frame 0), the ladder, the 1 s buffer, one key frame.
- [ ] BD on achieved rate; BD-J definition; class balance; bootstrap seed.
- [ ] Compare against the CSV next to each page, not against rounded numbers in prose.

## What we ran from a clean clone

We cloned the repository into an empty directory, gave it its own data root and anchors prefix, and ran section (c)
as written. Three things broke and one step was missing from the instructions; each was fixed in the repository
and the run repeated:

1. **Free-space floor.** The harness refused to run on a disk with less than 200 GiB free. That floor
   protects the project's shared machine. It can now be lowered with `NC_MIN_FREE_GIB`
   (`eval/nceval/diskguard.py`).
2. **Missing decoders.** Scoring a reference stream needs the pinned standard decoders (`ffmpeg-decode`, and JM
   for H.264). The first anchor list (x264, vmaf) lacked them. The error message named a tool `ffmpeg-decode`
   that `build_anchors.sh` does not know: the tool is called `ffmpeg`. The message now names the build tool
   (`eval/nceval/config.py`), and section (c) lists every tool needed.
3. **CPU slots.** Outside the project's shared pool, `NC_CPU_SLOTS` must be set or the run uses one slot.
4. **Build directory.** The harness expects a directory with `nanocodec` and `presets/`; section (c) creates it
   with two commands.

The open subset run then completed: 210 points, the tables above. No step needed anything outside this
repository except public tool sources (fetched by `build_anchors.sh` at pinned commits), Rust, and system
packages.
