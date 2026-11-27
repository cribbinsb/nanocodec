# The 1 s buffer the CBR chart references actually keep (2026-11-22)

**Question (owner).** 0.6.1 makes our 1 s buffer a hard bound. Are we putting ourselves at a
disadvantage against the reference encoders on the CBR charts, which are not all held that tightly?
And does draining the bucket at the *target* (while the BD curves use the *achieved* rate) give an
encoder that undershoots a looser buffer than it appears to have?

**Decision taken before this table (owner, via coordinator):** no point-exclusion rule (it is
complicated and can bias the comparison toward whichever encoder loses points). Our default re-code
ceiling is to match roughly the tighter references' worst case (about 1.15 buffers), with a strict
1.0 as a user option, and an explanatory note. This page is the data for that note.

## Method

- **Points:** every stored point that the published 0.6.0 charts read: main (11 VAL clips), held-out
  (209 clips), fresh (138, a subset of held-out) and VC-1K (100 clips), 150 frames, one key, the
  4-rung ladder plus the low rungs, read from the chart point cache through the chart's own curve
  loader (`nceval.chart.chart_curves`, every family on its recorded build,
  `docs/results/bd-vs-speed*-builds.json`). No encoding. VC-1K: 199 of 23,307 points are not in
  the store (references' tuned/extra presets on a few clips) and are left out.
- **Ours are 0.6.0** (builds 4fc9a1de, c9cff89f, 20d31869: key loan, 1.4-buffer re-code ceiling),
  every preset of a codec pooled into one row. 0.6.1's hard bound is **not** on any chart yet; its
  rows here are not measured.
- **Bucket:** `cbr_stats.replay_buffer` over each point's stored frame sizes, every bit counted (our
  key loan included), 1 s capacity, starting empty.
  - *At target:* drained at the target and holding 1 s of the target (the harness's buffer).
    Points over target by more than 10 % are left out of these columns: they are a rate floor (the
    encoder cannot go that low; libvpx and libaom on the low rungs, openh264), and a target-rate
    bucket reads them as 3-130 buffers, which says nothing about buffer control. Their share is
    its own column.
  - *At own rate:* drained at the stream's own average rate and holding 1 s of it (the owner's
    point: the rate the point is plotted at). Every point.
- Peaks are in buffers (1.0 = one full second queued). "over" columns: share of points whose peak
  exceeds 1.0 / 1.15 / 1.5 buffers.

## Findings (measured)

- **Undershooting does buy a looser buffer at the plotted rate.** Judged at their own rate the
  references' peaks rise: SVT-AV1 M12 p95 0.77 -> 1.46 (main), 0.83 -> 1.00 (held-out); x265 medium
  0.97 -> 1.31 (held-out), 1.00 -> 1.32 (fresh); x264 medium 0.41 -> 0.55 (main), 0.68 -> 0.80
  (held-out). x264 medium's median stream is 0.91-0.96 of target.
- **The references' worst case, at their own rate, is above 1.15 buffers, but on few points.**
  Held-out (the largest set): points over 1.15 are x265 7-8 %, SVT-AV1 2-3 %, libaom 2-3 %, libvpx
  2 %, x264 medium 1-2 %, x264 veryfast/ultrafast 1-2 %, openh264 34 %. Over 1.0: x265 11-12 %,
  SVT-AV1 3-5 %, libaom 4-5 %, libvpx 3-4 %, x264 2-3 %, openh264 42 %. Their maxima reach 2.5-4.4
  buffers (x265 3.5, SVT-AV1 M12 4.2, libvpx 3.1, x264 medium 2.5, openh264 6.6).
- **On VC-1K (phone calls) every reference but openh264 stays at or under one buffer on >= 99 %
  of points** (own rate, p95 0.45-0.76; openh264 7 % over 1.0).
- **Ours (0.6.0) ran the loosest buffer of all but openh264** on main/held-out/fresh: p95
  1.15-1.41 at own rate, 17-38 % of points over 1.0 and 6-28 % over 1.15 (VC-1K: <= 2 % over
  1.0). So on the published 0.6.0 charts the buffer advantage was ours, not the references'.
- **A 1.15 ceiling for us vs the references:** x265 (7-8 % of points over 1.15 on held-out) and
  openh264 would still be looser than us at our ceiling; SVT-AV1, libaom, libvpx and x264 are over
  1.15 on 1-3 % of points, so a hard 1.15 for every frame of ours is about level with them (they
  occasionally exceed it, we never would). A strict 1.0 would hold us tighter than every reference
  on held-out and fresh (each exceeds 1.0 on 2-13 % of points at its own rate; main 0-8 %,
  openh264 32-44 % everywhere but VC-1K); on VC-1K the references stay under 1.0 almost
  everywhere, so there it is level.
- Whether this costs us BD was not measured here (0.6.1's points are not on the charts).

## Tables


### Main chart: 11 clips, 2550 points (0 not stored); our builds 20d31869, 4fc9a1de, c9cff89f

| Encoder | points | over target by > 10 % | achieved / target: median (p5..p95) | at target: peak p95 / max | at target: over 1.0 / 1.15 / 1.5 | at own rate: peak p95 / max | at own rate: over 1.0 / 1.15 / 1.5 |
|---|---:|---:|---|---|---|---|---|
| nanocodec AV1 | 300 | 1 % | 0.986 (0.97..1.03) | 1.31 / 1.39 | 23 / 14 / 0 % | 1.32 / 1.50 | 26 / 13 / 0 % |
| nanocodec AV2 | 250 | 1 % | 0.985 (0.96..1.04) | 1.10 / 1.33 | 19 / 4 / 0 % | 1.15 / 1.49 | 20 / 6 / 0 % |
| nanocodec H.264 | 300 | 0 % | 0.981 (0.97..1.04) | 1.29 / 1.40 | 19 / 16 / 0 % | 1.30 / 1.72 | 21 / 13 / 1 % |
| nanocodec HEVC | 300 | 1 % | 0.978 (0.96..1.04) | 1.30 / 1.40 | 22 / 14 / 0 % | 1.26 / 1.59 | 23 / 15 / 0 % |
| nanocodec VP9 | 300 | 5 % | 0.986 (0.97..1.10) | 1.30 / 1.39 | 23 / 18 / 0 % | 1.28 / 1.52 | 26 / 17 / 0 % |
| nanocodec VVC | 300 | 1 % | 0.975 (0.96..1.03) | 1.33 / 1.40 | 16 / 13 / 0 % | 1.30 / 1.60 | 17 / 11 / 1 % |
| libaom-av1-cpu10 | 50 | 24 % | 1.027 (1.00..1.82) | 0.70 / 0.82 | 0 / 0 / 0 % | 0.69 / 1.08 | 2 / 0 / 0 % |
| libaom-av1-cpu8 | 50 | 22 % | 1.028 (1.00..1.63) | 0.69 / 0.73 | 0 / 0 / 0 % | 0.74 / 1.10 | 2 / 0 / 0 % |
| libvpx-vp9-cpu7 | 50 | 40 % | 1.015 (1.00..2.46) | 0.64 / 0.80 | 0 / 0 / 0 % | 0.80 / 0.85 | 0 / 0 / 0 % |
| libvpx-vp9-cpu9 | 50 | 42 % | 1.020 (1.00..2.16) | 0.56 / 0.67 | 0 / 0 / 0 % | 0.62 / 1.45 | 4 / 4 / 0 % |
| openh264-baseline | 50 | 36 % | 1.002 (1.00..2.84) | 1.80 / 2.35 | 19 / 19 / 12 % | 2.98 / 3.55 | 32 / 28 / 22 % |
| svtav1-m10 | 50 | 6 % | 0.998 (0.90..1.09) | 0.62 / 0.84 | 0 / 0 / 0 % | 0.82 / 1.31 | 2 / 2 / 0 % |
| svtav1-m12 | 50 | 4 % | 0.998 (0.77..1.08) | 0.77 / 1.40 | 2 / 2 / 0 % | 1.46 / 1.84 | 8 / 8 / 6 % |
| svtav1-m13 | 50 | 4 % | 1.001 (0.55..1.07) | 0.66 / 0.85 | 0 / 0 / 0 % | 1.26 / 2.30 | 8 / 8 / 4 % |
| svtav1-m8 | 50 | 6 % | 0.998 (0.93..1.15) | 0.61 / 0.69 | 0 / 0 / 0 % | 0.62 / 0.69 | 0 / 0 / 0 % |
| x264-medium | 50 | 12 % | 0.913 (0.64..1.33) | 0.41 / 0.60 | 0 / 0 / 0 % | 0.55 / 0.73 | 0 / 0 / 0 % |
| x264-medium~psnr | 50 | 14 % | 0.909 (0.66..1.36) | 0.33 / 0.60 | 0 / 0 / 0 % | 0.55 / 0.84 | 0 / 0 / 0 % |
| x264-ultrafast | 50 | 0 % | 0.946 (0.75..1.05) | 0.69 / 0.78 | 0 / 0 / 0 % | 0.87 / 1.04 | 2 / 0 / 0 % |
| x264-veryfast | 50 | 0 % | 0.887 (0.59..1.05) | 0.41 / 0.54 | 0 / 0 / 0 % | 0.66 / 1.04 | 2 / 0 / 0 % |
| x265-medium | 50 | 0 % | 0.962 (0.84..1.02) | 0.83 / 1.36 | 4 / 4 / 0 % | 0.96 / 1.58 | 6 / 4 / 2 % |
| x265-superfast | 50 | 0 % | 0.955 (0.82..1.02) | 0.83 / 1.36 | 4 / 4 / 0 % | 1.00 / 1.62 | 6 / 4 / 2 % |
| x265-ultrafast | 50 | 0 % | 0.967 (0.82..1.02) | 0.84 / 1.38 | 4 / 4 / 0 % | 1.01 / 1.58 | 6 / 4 / 2 % |

### Held-out: 209 clips, 47634 points (0 not stored); our builds 20d31869, 4fc9a1de, c9cff89f

| Encoder | points | over target by > 10 % | achieved / target: median (p5..p95) | at target: peak p95 / max | at target: over 1.0 / 1.15 / 1.5 | at own rate: peak p95 / max | at own rate: over 1.0 / 1.15 / 1.5 |
|---|---:|---:|---|---|---|---|---|
| nanocodec AV1 | 5604 | 1 % | 0.989 (0.96..1.02) | 1.37 / 1.40 | 29 / 22 / 0 % | 1.40 / 2.07 | 30 / 22 / 1 % |
| nanocodec AV2 | 4670 | 1 % | 0.988 (0.96..1.03) | 1.28 / 1.40 | 29 / 13 / 0 % | 1.33 / 2.00 | 30 / 14 / 1 % |
| nanocodec H.264 | 5604 | 0 % | 0.988 (0.96..1.02) | 1.35 / 1.48 | 25 / 17 / 0 % | 1.38 / 2.38 | 26 / 18 / 1 % |
| nanocodec HEVC | 5604 | 1 % | 0.981 (0.95..1.02) | 1.36 / 1.40 | 34 / 25 / 0 % | 1.40 / 2.14 | 37 / 26 / 1 % |
| nanocodec VP9 | 5604 | 1 % | 0.987 (0.96..1.04) | 1.38 / 1.43 | 37 / 28 / 0 % | 1.41 / 2.41 | 38 / 28 / 1 % |
| nanocodec VVC | 5604 | 1 % | 0.977 (0.95..1.01) | 1.33 / 1.40 | 22 / 15 / 0 % | 1.38 / 2.70 | 25 / 16 / 1 % |
| libaom-av1-cpu10 | 934 | 24 % | 1.030 (1.00..2.34) | 0.90 / 1.53 | 3 / 2 / 0 % | 0.99 / 3.16 | 5 / 3 / 1 % |
| libaom-av1-cpu8 | 934 | 23 % | 1.028 (1.00..2.28) | 0.90 / 1.44 | 3 / 2 / 0 % | 0.94 / 2.96 | 4 / 2 / 0 % |
| libvpx-vp9-cpu7 | 934 | 36 % | 1.023 (1.00..3.31) | 0.79 / 1.55 | 3 / 2 / 0 % | 0.93 / 2.57 | 3 / 2 / 1 % |
| libvpx-vp9-cpu9 | 934 | 35 % | 1.025 (1.00..3.15) | 0.82 / 1.84 | 3 / 2 / 1 % | 0.94 / 3.12 | 4 / 2 / 1 % |
| openh264-baseline | 934 | 34 % | 1.003 (0.99..3.87) | 2.20 / 6.64 | 29 / 22 / 15 % | 2.78 / 6.64 | 42 / 34 / 22 % |
| svtav1-m10 | 934 | 8 % | 1.007 (0.94..1.26) | 0.74 / 1.48 | 1 / 0 / 0 % | 0.84 / 3.90 | 3 / 2 / 0 % |
| svtav1-m12 | 934 | 6 % | 1.010 (0.95..1.20) | 0.83 / 1.46 | 2 / 1 / 0 % | 1.00 / 4.17 | 5 / 3 / 1 % |
| svtav1-m13 | 934 | 6 % | 1.010 (0.95..1.18) | 0.80 / 1.58 | 2 / 1 / 0 % | 0.96 / 4.38 | 5 / 3 / 2 % |
| svtav1-m8 | 934 | 11 % | 1.006 (0.92..1.28) | 0.71 / 1.62 | 1 / 0 / 0 % | 0.86 / 3.99 | 3 / 2 / 0 % |
| x264-medium | 934 | 11 % | 0.925 (0.78..1.47) | 0.68 / 3.03 | 0 / 0 / 0 % | 0.80 / 2.50 | 3 / 2 / 0 % |
| x264-medium~psnr | 934 | 11 % | 0.929 (0.78..1.46) | 0.62 / 2.50 | 0 / 0 / 0 % | 0.80 / 2.06 | 2 / 1 / 0 % |
| x264-ultrafast | 934 | 1 % | 0.969 (0.79..1.07) | 0.78 / 1.61 | 1 / 0 / 0 % | 0.92 / 1.68 | 3 / 1 / 0 % |
| x264-veryfast | 934 | 1 % | 0.910 (0.73..1.05) | 0.70 / 0.96 | 0 / 0 / 0 % | 0.83 / 1.57 | 2 / 2 / 0 % |
| x265-medium | 934 | 8 % | 0.996 (0.88..1.15) | 0.97 / 3.24 | 5 / 3 / 0 % | 1.31 / 3.50 | 11 / 7 / 3 % |
| x265-superfast | 934 | 7 % | 0.993 (0.87..1.14) | 0.98 / 2.61 | 5 / 3 / 1 % | 1.32 / 3.02 | 12 / 8 / 4 % |
| x265-ultrafast | 934 | 8 % | 1.003 (0.88..1.15) | 0.97 / 2.64 | 5 / 3 / 1 % | 1.33 / 3.07 | 11 / 7 / 4 % |

### Fresh (held-out, unseen): 138 clips, 31671 points (0 not stored); our builds 20d31869, 4fc9a1de, c9cff89f

| Encoder | points | over target by > 10 % | achieved / target: median (p5..p95) | at target: peak p95 / max | at target: over 1.0 / 1.15 / 1.5 | at own rate: peak p95 / max | at own rate: over 1.0 / 1.15 / 1.5 |
|---|---:|---:|---|---|---|---|---|
| nanocodec AV1 | 3726 | 1 % | 0.989 (0.96..1.03) | 1.34 / 1.40 | 26 / 19 / 0 % | 1.38 / 1.95 | 27 / 19 / 1 % |
| nanocodec AV2 | 3105 | 1 % | 0.988 (0.96..1.03) | 1.26 / 1.40 | 27 / 11 / 0 % | 1.32 / 2.00 | 29 / 11 / 0 % |
| nanocodec H.264 | 3726 | 1 % | 0.989 (0.96..1.02) | 1.33 / 1.48 | 21 / 14 / 0 % | 1.36 / 1.89 | 22 / 15 / 1 % |
| nanocodec HEVC | 3726 | 0 % | 0.981 (0.95..1.02) | 1.35 / 1.40 | 34 / 24 / 0 % | 1.39 / 2.14 | 36 / 26 / 1 % |
| nanocodec VP9 | 3726 | 1 % | 0.987 (0.96..1.03) | 1.36 / 1.43 | 35 / 26 / 0 % | 1.40 / 1.87 | 36 / 26 / 1 % |
| nanocodec VVC | 3726 | 1 % | 0.977 (0.95..1.01) | 1.32 / 1.40 | 19 / 12 / 0 % | 1.36 / 1.71 | 22 / 13 / 1 % |
| libaom-av1-cpu10 | 621 | 24 % | 1.031 (1.00..2.42) | 0.86 / 1.53 | 2 / 1 / 0 % | 0.92 / 3.16 | 4 / 2 / 1 % |
| libaom-av1-cpu8 | 621 | 23 % | 1.031 (1.00..2.46) | 0.85 / 1.44 | 2 / 2 / 0 % | 0.91 / 2.96 | 3 / 1 / 0 % |
| libvpx-vp9-cpu7 | 621 | 36 % | 1.023 (1.00..3.48) | 0.72 / 1.38 | 2 / 1 / 0 % | 0.88 / 2.32 | 3 / 2 / 1 % |
| libvpx-vp9-cpu9 | 621 | 35 % | 1.025 (1.00..3.36) | 0.73 / 1.84 | 1 / 1 / 0 % | 0.92 / 3.12 | 4 / 2 / 1 % |
| openh264-baseline | 621 | 35 % | 1.003 (1.00..3.87) | 2.12 / 4.21 | 29 / 22 / 15 % | 2.68 / 5.60 | 44 / 34 / 23 % |
| svtav1-m10 | 621 | 9 % | 1.009 (0.94..1.32) | 0.72 / 1.46 | 1 / 0 / 0 % | 0.80 / 3.90 | 3 / 2 / 0 % |
| svtav1-m12 | 621 | 7 % | 1.012 (0.96..1.26) | 0.80 / 1.41 | 2 / 1 / 0 % | 0.93 / 4.17 | 4 / 3 / 1 % |
| svtav1-m13 | 621 | 7 % | 1.012 (0.95..1.25) | 0.78 / 1.58 | 2 / 1 / 0 % | 0.94 / 4.38 | 4 / 3 / 2 % |
| svtav1-m8 | 621 | 12 % | 1.009 (0.93..1.31) | 0.68 / 1.62 | 1 / 0 / 0 % | 0.75 / 3.99 | 3 / 1 / 0 % |
| x264-medium | 621 | 12 % | 0.921 (0.78..1.58) | 0.64 / 3.03 | 0 / 0 / 0 % | 0.79 / 2.50 | 2 / 1 / 0 % |
| x264-medium~psnr | 621 | 12 % | 0.925 (0.78..1.54) | 0.61 / 2.50 | 0 / 0 / 0 % | 0.77 / 2.06 | 2 / 2 / 0 % |
| x264-ultrafast | 621 | 1 % | 0.965 (0.79..1.07) | 0.78 / 1.61 | 1 / 0 / 0 % | 0.95 / 1.68 | 3 / 1 / 0 % |
| x264-veryfast | 621 | 1 % | 0.903 (0.73..1.05) | 0.71 / 0.88 | 0 / 0 / 0 % | 0.85 / 1.57 | 2 / 2 / 0 % |
| x265-medium | 621 | 10 % | 0.995 (0.88..1.24) | 1.00 / 3.24 | 5 / 3 / 1 % | 1.32 / 3.50 | 12 / 8 / 3 % |
| x265-superfast | 621 | 9 % | 0.991 (0.87..1.18) | 1.05 / 2.61 | 5 / 3 / 1 % | 1.37 / 2.78 | 13 / 9 / 4 % |
| x265-ultrafast | 621 | 9 % | 1.002 (0.89..1.18) | 0.96 / 2.64 | 5 / 3 / 1 % | 1.37 / 2.84 | 12 / 8 / 4 % |

### VC-1K: 100 clips, 23108 points (199 not stored); our builds 20d31869, 4fc9a1de, c9cff89f

| Encoder | points | over target by > 10 % | achieved / target: median (p5..p95) | at target: peak p95 / max | at target: over 1.0 / 1.15 / 1.5 | at own rate: peak p95 / max | at own rate: over 1.0 / 1.15 / 1.5 |
|---|---:|---:|---|---|---|---|---|
| nanocodec AV1 | 2730 | 0 % | 0.987 (0.97..1.01) | 0.72 / 1.34 | 1 / 0 / 0 % | 0.74 / 1.41 | 1 / 0 / 0 % |
| nanocodec AV2 | 2275 | 0 % | 0.988 (0.97..1.01) | 0.79 / 1.10 | 1 / 0 / 0 % | 0.81 / 1.14 | 2 / 0 / 0 % |
| nanocodec H.264 | 2704 | 0 % | 0.982 (0.96..1.00) | 0.60 / 0.97 | 0 / 0 / 0 % | 0.62 / 1.06 | 0 / 0 / 0 % |
| nanocodec HEVC | 2730 | 0 % | 0.978 (0.96..0.99) | 0.79 / 1.35 | 1 / 0 / 0 % | 0.82 / 1.44 | 2 / 0 / 0 % |
| nanocodec VP9 | 2730 | 0 % | 0.987 (0.97..1.00) | 0.78 / 1.35 | 2 / 1 / 0 % | 0.80 / 1.33 | 2 / 1 / 0 % |
| nanocodec VVC | 2730 | 0 % | 0.975 (0.96..0.99) | 0.71 / 1.35 | 0 / 0 / 0 % | 0.74 / 1.34 | 1 / 0 / 0 % |
| libaom-av1-cpu10 | 452 | 12 % | 1.022 (1.00..1.46) | 0.52 / 0.87 | 0 / 0 / 0 % | 0.52 / 0.89 | 0 / 0 / 0 % |
| libaom-av1-cpu8 | 452 | 14 % | 1.021 (1.00..1.68) | 0.51 / 0.71 | 0 / 0 / 0 % | 0.51 / 0.87 | 0 / 0 / 0 % |
| libvpx-vp9-cpu7 | 445 | 20 % | 1.013 (1.00..2.02) | 0.48 / 0.58 | 0 / 0 / 0 % | 0.56 / 0.95 | 0 / 0 / 0 % |
| libvpx-vp9-cpu9 | 447 | 20 % | 1.013 (1.00..2.03) | 0.50 / 0.70 | 0 / 0 / 0 % | 0.53 / 1.09 | 0 / 0 / 0 % |
| openh264-baseline | 455 | 36 % | 1.001 (1.00..3.60) | 0.78 / 1.69 | 2 / 1 / 0 % | 1.10 / 1.74 | 7 / 5 / 1 % |
| svtav1-m10 | 449 | 1 % | 1.000 (0.95..1.03) | 0.46 / 0.78 | 0 / 0 / 0 % | 0.47 / 0.69 | 0 / 0 / 0 % |
| svtav1-m12 | 455 | 1 % | 1.002 (0.95..1.04) | 0.55 / 1.03 | 0 / 0 / 0 % | 0.56 / 1.10 | 0 / 0 / 0 % |
| svtav1-m13 | 455 | 1 % | 1.003 (0.96..1.04) | 0.61 / 0.92 | 0 / 0 / 0 % | 0.57 / 0.92 | 0 / 0 / 0 % |
| svtav1-m8 | 447 | 1 % | 1.000 (0.95..1.03) | 0.43 / 0.66 | 0 / 0 / 0 % | 0.45 / 0.64 | 0 / 0 / 0 % |
| x264-medium | 455 | 12 % | 0.961 (0.79..1.36) | 0.60 / 0.88 | 0 / 0 / 0 % | 0.59 / 0.97 | 0 / 0 / 0 % |
| x264-medium~psnr | 455 | 12 % | 0.951 (0.77..1.39) | 0.55 / 0.89 | 0 / 0 / 0 % | 0.62 / 1.14 | 0 / 0 / 0 % |
| x264-ultrafast | 449 | 1 % | 0.964 (0.73..1.08) | 0.69 / 0.83 | 0 / 0 / 0 % | 0.76 / 1.99 | 1 / 1 / 0 % |
| x264-veryfast | 448 | 0 % | 0.937 (0.68..1.07) | 0.63 / 0.85 | 0 / 0 / 0 % | 0.72 / 1.11 | 0 / 0 / 0 % |
| x265-medium | 450 | 1 % | 0.981 (0.87..1.06) | 0.66 / 1.57 | 0 / 0 / 0 % | 0.69 / 1.51 | 1 / 0 / 0 % |
| x265-superfast | 447 | 1 % | 0.978 (0.87..1.06) | 0.64 / 1.04 | 0 / 0 / 0 % | 0.67 / 1.11 | 0 / 0 / 0 % |
| x265-ultrafast | 448 | 1 % | 0.980 (0.87..1.06) | 0.65 / 1.05 | 0 / 0 / 0 % | 0.67 / 1.12 | 0 / 0 / 0 % |
