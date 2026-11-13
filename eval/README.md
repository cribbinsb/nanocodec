# eval — the nanocodec evaluation harness

Implements the methodology below ("Methodology"): clip sets, anchor
encoders, a conformance gate on every stream, quality metrics, BD-rate
reports and pinned single-core speed runs.

```
eval/
  inventory/      scan of the source video folders (bitrates, classes)
  clipset/        selection files + extract.py: source video -> raw YUV clip set
  nceval/         the harness (python3 -m nceval ...)
    encoders/     one adapter per anchor configuration; ANCHORS registry
    conformance   two standard decoders per codec, bit-exact or the run fails
    metrics       PSNR-YUV 6:1:1 (own), XPSNR (ffmpeg), VMAF v1/MS-SSIM (VMAFx CUDA; libvmaf)
    sweep         fixed-QP RD sweeps with per-clip quantiser picking
    bdrate        PCHIP BD-rate with overlap checks
    report        BD-rate tables per clip / content class / overall
    speed         taskset-pinned timing, ABBA interleaved, medians
  tests/          unit tests (python3 -m pytest tests)
```

Data lives outside the repo, under `$NC_DATA_ROOT` (default
`~/nanocodec-data`): `anchors/` (tools), `inventory/`, `clips/`, `work/`
(bitstreams), `results/results.sqlite`.

## Methodology

The rules every measurement follows. The real-use objective (CBR ladder,
J_real, guards) is in our internal procedures, "Adoption rule: J_real" (not published); the
published board's method is `docs/results/README.md` "Method".

### Clips

- **Inventory** (`inventory/`): every source file in `/mldata/video` and
  `/mldata/tracking_original`, with its **source bitrate** and a content
  class (static, bodycam, drone, crowd, low-light, pan).
- **Clip sets** (`clipset/`): short native-fps excerpts at 360p–1080p
  (mainly 720p), balanced by class and converted once to raw YUV with a
  recorded hash.
- **Splits:** TRAIN tunes heuristics and thresholds; VAL confirms each
  change and is never screened on; the held-out set is used only for
  reported numbers. Every heuristic needs this split: early tuning overfit
  to single clips.
- **Synthetic noise:** clean clips plus calibrated sensor noise
  (`clipset/add_noise.py`), scored against the clean original
  (`docs/ARCHITECTURE.md` §1).
- **50% rule:** no test point above half the clip's source bitrate (many
  sources are already compressed).

### Anchors

Every anchor runs the same low-delay setup as ours: P frames only (no
B-frames, reordering or lookahead), one thread, fixed intra period, no
scene-cut keyframes, the same frame count, and temporal layers matched where
the anchor supports them. Each configuration is pinned explicitly in
`nceval/encoders/` (versions in `tools/anchors/versions.env`).

- **openh264-baseline** (the H.264 bar): `CAMERA_VIDEO_REAL_TIME`,
  `MEDIUM_COMPLEXITY`, Constrained Baseline / CAVLC, AQ, scene-change and
  background detection at their defaults (on), denoise and LTR off, frame
  skip off.
- **openh264-high:** the same with CABAC, `HIGH_COMPLEXITY` and all analysis
  on. openh264 has no 8×8 transform, so this is the best openh264 can do.
- **SVT-AV1 M12 `--rtc`:** the AV1/HEVC bar (`docs/ARCHITECTURE.md` §1); x264, x265,
  kvazaar, libaom and libvpx are reference points.
- **Preview operating point** (one product use case, reported separately and
  must not regress, not tuned for): openh264 `LOW_COMPLEXITY`, ≤512 px,
  about 6 fps, GOP 5, two temporal layers, 256 kbps. It needs constant
  parameter-set ids across IDRs (the CMAF writer otherwise cuts a segment
  every GOP) and no frame dropping.

### Rate–distortion and metrics

- **Fixed-QP sweeps** (`nceval/sweep.py`) isolate coding efficiency from
  rate control: per-clip quantisers chosen so curves overlap in quality,
  covering about 150–800 kbps at 720p, scaled by pixel count.
- **BD-rate** (`nceval/bdrate.py`): PCHIP over the overlapping range only,
  reported per clip, per content class and overall, because an overall
  number alone hides class-dependent results.
- **Metrics:** adoption decisions use J_real's set: mean BD-rate over
  PSNR-YUV (6:1:1), VMAF v1 and MS-SSIM, in equal thirds (VMAF v0.6.1 NEG is
  no longer used); `nceval/metrics.py` documents the others and the
  measurement profiles. No metric is computed on a stream that failed the
  conformance gate.

### Speed

`nceval/speed.py`: pinned to one core, a warm-up, then interleaved (ABBA)
timed rounds and medians, plus `perf stat` instructions and cycles where
available. Each encoder is timed near the same bitrate, and the result is
the ratio to the anchor on the same machine and clip. Instruction counts
and the Jetson procedure: our internal procedures, "Measure speed" (not published).

### Diagnostics: the standard stats map

Every loss must be explainable without re-running anything heavy, so
bitstreams and per-frame data (bytes, type, layer, QP, per-frame quality) are
kept for every point. `nceval/streamstats.py` extracts a codec-neutral stats
map (bits by category, area by block size and mode, MV and transform
statistics) **decoder-side**, so it works the same for our streams and the
anchors'; usage in our internal procedures, "Inspect a stream" (not published).

## The reference evaluation (maintained entry point)

The anchor evaluation is one command, driven by the checked-in config
`eval/configs/reference.json`:

```
cd eval
python3 -m nceval reference                          # everything, idempotent
python3 -m nceval reference --only report            # just regenerate the report
python3 -m nceval reference --only speed,report --platform-remote jetson
python3 -m nceval reference --only speed --force-speed   # re-time regardless
```

Steps (each skips work that is already up to date):

1. **clips**: `clipset/extract.py` for every suite's selection file, then
   `clipset/add_noise.py` for the suite's `noise_sources`. Existing outputs
   are reused (SHA-256 / deterministic seed).
2. **sweep**: every anchor × clip × structure of each suite, all splits.
   A curve is skipped if it was measured with the current **fingerprint**
   (`nceval/fingerprint.py`): the anchor's pinned commit in
   `tools/anchors/versions.env` (+ the openh264-harness source), the
   adapter's command template, the pinned decoders/libvmaf and
   `metrics.METRICS_REVISION`. Change any of these and the affected curves are
   re-measured on the next run. Bitstreams are kept under
   `work/<encoder>/<structure>/<clip>/<fingerprint>/`; per-frame data
   (`frames` table) is backfilled from them if missing.
3. **speed**: pinned single-core timing of the configured clips, at each
   encoder's sweep point nearest the configured 720p30 target (scaled by pixel
   rate for other clips). Re-timed only when a fingerprint changed. With
   `--platform-remote NAME` the same runs are repeated on that machine
   (`remotes` in the config): code and the needed clips are rsynced over, the
   remote `nceval speed --json` is run with the PC's chosen quantisers, and
   the results are stored locally under the remote's platform name.
4. **report**: always regenerated from the database. Written to
   `$NC_DATA_ROOT/reports/reference-<date>-<time>.md`; the headline
   (BD-rate vs each report anchor on the headline suite/structure, all clips,
   plus speed ratios) is also printed.

The results database is `$NC_DATA_ROOT/results/results.sqlite`
(`nceval/results.py` documents the tables, including per-frame data).

### Adding an anchor

1. If it is a new tool: add `<TOOL>_URL/_REF/_COMMIT` to
   `tools/anchors/versions.env` and a build function to
   `tools/anchors/build_anchors.sh`; rebuild on each machine.
2. Write (or reuse) an adapter in `nceval/encoders/` producing a low-delay,
   single-thread, fixed-QP, P-only command; map its name prefix to the
   versions.env key in `nceval/fingerprint.py` (`_TOOL_KEYS`).
3. Register the configuration in `nceval/encoders/__init__.py` (`ANCHORS`).
   Names are database keys: never rename one.
4. Add the name to the suites' `anchors` lists in `configs/reference.json`
   and run `python3 -m nceval reference`.

### Adding a clip

1. Add an entry to the suite's selection file (`clipset/main.json` or
   `clipset/long.json`): source path (must be in the inventory), start,
   frames, 720p size, class and split. Check the 50% rule leaves room
   (`max_test_kbps` in the inventory) and the source has no duplicated frames.
2. Run `python3 -m nceval reference`: it extracts the clip and measures only
   the new curves.

## Manual workflow (ad-hoc)

All commands run from `eval/`.

1. **Tools** (once per machine): `tools/anchors/build_anchors.sh` builds the
   pinned encoders, reference decoders, libvmaf and `openh264-harness` into
   `$NC_DATA_ROOT/anchors/<arch>/`. `tools/anchors/smoke_test.sh` checks them.
2. **Inventory** (when source folders change):
   `python3 inventory/build_inventory.py` (see `inventory/README.md`).
3. **Clip set**: write a selection file (see `clipset/smoke.json`), then
   `python3 clipset/extract.py clipset/smoke.json`
   → `$NC_DATA_ROOT/clips/smoke.json` + YUV files.
4. **Sweep** (quality):
   `python3 -m nceval sweep --clipset ~/nanocodec-data/clips/smoke.json
   --encoders openh264-baseline,x265-ultrafast --intra-period-seconds 10 --temporal-layers 1`
   Every stream is checked by two standard decoders before any metric is
   computed. A conformance failure makes the command exit non-zero.
5. **Report**:
   `python3 -m nceval report --clipset ... --anchor openh264-baseline
   --metric psnr_yuv,vmaf_v1,xpsnr_y [--per-clip]`
6. **Speed** (after a sweep, so each encoder is timed at a matching bitrate):
   `python3 -m nceval speed --clipset ... --clip NAME --encoders a,b
   [--core 2] [--runs 5]`

The coding structure (`--intra-period-seconds`, `--temporal-layers`, `--max-frames`)
identifies a result set (`ip10s-tl1` etc.); use the same values for sweep,
report and speed. `--single-keyframe` codes one key frame for the whole clip
(`ip1key-tl1`, the long-GOP set that exposes errors accumulating over long
P-frame chains).

7. **Sanity report** (before adoption decisions):
   `python3 -m nceval sanity [--only measure|checks]`, configured by
   `eval/configs/sanity.json`.
8. **Quality under packet loss**: `python3 -m nceval.losssim run|matrix|ltr|startup|checks|table`
   drives `nanocodec session` (encoder + our decoder) through a lossy RTP
   network with receiver feedback and scores what the viewer saw
   (internal notes, not published).

## Rules the harness enforces

- No test point above 50% of the clip's source bitrate (`clips.py`).
- No metric for a stream that did not pass the conformance gate.
- Fixed intra period and no scene-cut keyframes in every anchor config.

## Known limitations

- `perf stat` needs `kernel.perf_event_paranoid <= 2`; otherwise instruction
  counts are omitted and `speed` says so.
- Timings on a machine with the `powersave` governor or boost enabled are
  noisy; `speed` lists such conditions under its table.
- openh264 with RC off still varies QP per temporal layer (and per MB with
  AQ on); its "quantizer" is the base QP.

## Tuning rate domain

The tuner calibrates four quantizers per TRAIN clip inside its actual
`source_kbps / 2` ceiling, using the sweep's scaled operating range and
encoder bitrate floor. It retains a little ceiling headroom; a trial that
still overshoots increases that point's QP before scoring it. Rate probes
also pass both standard decoders, but points above the ceiling never receive
quality metrics. Reports filter historical database points by the same cap.
Missing curves or unreliable quality overlap reject the entire trial.

Cost probes use quantizers selected from the legal curves. Old fixed-QP
instruction helpers remain for reproducing historical profiles; those counts
must be labelled with their QP rather than presented as operating-rate costs.
Study anchors fingerprint the binary, rate-selection code, tools, clips and
source-bitrate metadata. Curve arrays and actual cost QPs are retained per
trial. New protocols require new study names; the uncapped overnight v1 study
was retired without adopting a preset. Presets retain full float precision.
