# Clip inventory

`build_inventory.py` builds the test-clip inventory described in
`eval/README.md` "Methodology: Clips". It needs Python 3 (standard library only) and the
ffprobe/ffmpeg CLI (7.1).

```
python3 eval/inventory/build_inventory.py              # full run, resumes from cache
python3 eval/inventory/build_inventory.py --limit 20   # quick test
python3 eval/inventory/build_inventory.py --force      # ignore the cache
```

The sources (`/mldata/video`, `/mldata/tracking_original`) are only read.
Output goes to `~/nanocodec-data/inventory/` (`--out` to change):

| File | Contents |
|---|---|
| `inventory.json` | `summary` (counts, skipped non-video files by extension, the rules used) and one record per video file in `clips` |
| `inventory.csv` | the main fields, one row per file, for spreadsheets |
| `shortlist.csv` | proposed curated-set candidates for human review: up to 10 per class (≤ 3 per source folder unless the class has too few folders), not derived/edited/duplicate, ≥ 12 s, confidence ≥ 0.6, ranked by 720p30 usability under the 50% rule, fps, confidence and bits per pixel; plus the earlier attempt's sources |
| `cache/` | one JSON per file; a rerun skips files whose size and mtime are unchanged |

Up to `--workers` (default 12, max 16) ffmpeg/ffprobe processes run at
once, each with 2 decode threads and `nice +10`. A broken file gets an
`error` field and the run continues.

## Fields

- **Probe:** `container`, `codec`, `profile`, `width`, `height`, `pix_fmt`,
  `duration_s`, `frame_count`.
- **`fps`** is the real rate: `avg_frame_rate` for containers. Raw `.264/.265/.hevc`
  streams have no container timing, so the rate comes from the filename
  (`_7.5fps`) when present (`fps_source: filename`), else ffprobe's default
  (`assumed`; treat with suspicion). Their frames are counted.
- **`source_kbps`:** the video stream `bit_rate`; otherwise size × 8 /
  duration (`bitrate_source` says which; `file_size_incl_audio` overestimates).
- **`content_hash`:** sha256 of size + first and last 1 MiB.
- **`stats`:** measured on up to 20 s, starting 10% in (max 60 s; raw streams
  from 0), resampled to 5 fps so clips with different frame rates compare:
  - `luma_mean`, `luma_p10`, `sat_mean`: signalstats at 320 px wide;
  - `ydif_median/p90/cv`: mean |Δluma| between sampled frames;
  - `changed_frac_median/p90`: share of pixels changing by more than 10;
  - `camera_motion_frac`: share of frame pairs where a global translation
    (exhaustive ±15 px search on 64×36 thumbnails) cuts the difference by
    ≥ 20%; `camera_motion_px` (median shift, 320-px units) and
    `camera_motion_consistency` (0 = random directions, 1 = one direction);
  - `detail`: mean |x − blur(x)| on a native-resolution 640×360 centre crop;
    `noise_est`: the same in the flattest 10% of 8×8 blocks (crude, and low
    for most sources because compression removed the noise);
  - `scene_cuts`, `cuts_per_min`: scdet hard cuts;
  - `luma_signature`: per-second mean luma, for duplicate detection.

## Auto class (a proposal for review)

Stats first:
1. mean luma < 50, or a monochrome picture (IR night mode) → **low-light**;
2. camera translates in ≥ 20% of frame pairs → **pan** if the direction is
   steady (consistency ≥ 0.6), else **handheld-bodycam**;
3. fixed camera with ≥ 10% of pixels changing per frame → **crowd-busy**;
4. otherwise → **static**.

For low-light clips the motion class is still given in the reasons.
Path hints (`bwc`, `MOT20`, `roundabouthd`, `bdd100k`, `Night`, `OHcam`, …)
then confirm (confidence 0.9) or override where the stats are compatible
(0.6–0.75); a `night` hint makes a clip low-light up to mean luma 90 (lit
night scenes). Stats-based low light always stands (0.75). **drone-aerial** comes only from hints. Forward motion
(dashcams) has no single translation, so it reads as busy unless hinted as
a vehicle; it is filed under **pan**. Disagreement gives 0.45 and the reason
"stats disagree with path hint". `class_reasons` lists the evidence.

Tags: `derived` (composites and re-encoded test copies under
`long_composite/`, `test/`, `converted/`), `edited` (films/web uploads, or
≥ 3 hard cuts per minute).

## Bitrate rule and operating points

A clip is never tested above half its source bitrate: `max_test_kbps =
0.5 × source_kbps`. RD ranges are 150–800 kbps at 720p, scaled by pixel
count: 360p 37.5–200, 1080p 337.5–1800. For each operating point
`operating_points.<op>` gives `resolution_ok` (source at least that size),
`ceiling_kbps` (min of the range top and `max_test_kbps`), `full_range_ok`
and `usable` (the ceiling reaches the range bottom). `ok_720p30_800` /
`ok_720p30_500`: ≥ 720p, ≥ 24 fps and `max_test_kbps` ≥ 800 / 500.

## Duplicates

`duplicate_kind: exact` means the same content hash; `near` means the same
duration (±0.5 s or 1%), aspect ratio, per-second luma signature (mean
difference < 1), dark level, saturation and activity: usually the same
footage re-encoded. Groups are transitive. The representative is an original
(not `derived`) with the highest pixel rate, then bitrate; the others get
`duplicate_of`.

`earlier_attempt` lists the clip names from
`/mldata/benchmarks/preview-codecs-nov05/iteration/inputs.json` whose source
this file is (or an exact copy of).
