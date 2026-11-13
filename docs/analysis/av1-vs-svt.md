# nanocodec-av1 vs SVT-AV1 M12: where the gap comes from

**Later diagnostic correction (overnight continuation):** M12 uses DC-only
luma intra even on keyframes, as documented in
[the source guide](../reference/svt-av1-m12-rtc.md). Inspection of the saved
bodycam keyframe confirms 100% DC luma; UV area is 94.75% DC and 5.25% CfL.
The intra-mode accounting below therefore does not establish a richer luma
mode set. Variable partitioning, chroma prediction, and keyframe quality
remain relevant; broader luma modes are an optional additional experiment.
The saved flat-IPP frame headers also confirm previous-frame-only references.

## Updated diagnostic checkpoint, 2026-11-07

The original comparison below is historical. Fresh 300-frame, GOP300
stream inspection matches all six native profile streams byte-for-byte;
standard decoder and reconstruction checks passed. These are nearby-rate
operating points, not matched-quality timing or BD-rate measurements.

| Clip / encoder | kbps | P skip area | P NEW area | P PSNR-Y |
|---|---:|---:|---:|---:|
| Bodycam default v7 | 326.41 | 76.81% | 5.15% | 45.437 |
| Bodycam full proxy prototype | 313.44 | 81.87% | 5.43% | 45.338 |
| Bodycam SVT M12 | 317.02 | 89.44% | 15.82% | 47.283 |
| Crowd default v7 | 932.38 | 77.18% | 11.02% | 31.167 |
| Crowd full proxy prototype | 927.14 | 79.39% | 11.05% | 30.950 |
| Crowd SVT M12 | 944.99 | 85.73% | 16.12% | 31.688 |

Evidence: external `work/overnight/av1-v7-rd-native-cycles/streamstats/summary.json`.
The full-proxy prototype in this diagnostic is not the later gain1.25/sigma1
candidate. Its luma PSNR also hides chroma loss; inspect Y/U/V separately.

Native profiles identify forward transforms as 14.87%/14.97% of default
bodycam/crowd cycles, versus SVT 3.88%/6.50%; inverse transforms are
13.24%/8.36%, falling to 3.31%/3.96% with the full proxy. Each percentage
uses that encoder's own total cycles; equal percentages do not imply equal
absolute CPU cost. Narrow-column and rounded-high ARM kernels are now
integrated; pre-transform and later residual exits target work avoidance.

The low NEW share still matters, but lowering NEW rate cost was already
measured and was worse. Existing block comparisons point toward vector
quality; do not repeat a rate-discount experiment based only on NEW share.
Source-guided subpixel walk, seed refinement exits, CfL, keyframe planning,
base-frame effort controls, and local CDEF have now been implemented as
optional controls. Their measured tradeoffs and current validation status
are in the internal overnight notes. The ranked fixes
below describe the original diagnosis, not a list of unimplemented features.

## Original comparison

**Date:** 2026-11-06.
**Builds:** nanocodec tip `dbd4d8d`; SVT-AV1 v4.3.0 with `--rtc 1 --preset 12`, flat IPP, CQP, 1 thread.
**Method:**
- For each clip, SVT was encoded at one QP, and our qindex was bisected until our file was within 3% of SVT's size.
- Both streams were decoded with libaom's inspection decoder (`aom-inspect`, accounting on) and summarised by `eval/nceval/streamstats.py`.
- Quick clips, 720p, 10 s intra period, whole clip (300 frames; low-light 240 frames at 24 fps).
- **Reproduce:**

  ```sh
  python3 -I tools/dev/compare_av1_svt.py <nanocodec> <out_dir> bodycam_airport=22 lowlight_pp22_004=32 crowd_mot17_09=57
  ```

  Per-frame JSON from this run is in `~/nanocodec-data/work/streamstats/cmp/`.

**Context:** the official sweep at `c7f19fd` has AV1 against SVT at PSNR-YUV BD bodycam +110%, low-light +26%, crowd +0.2%. Crowd is the control clip, where we are at parity.

## Summary

At equal file size, SVT spends its bits on **prediction** (more NEW motion vectors, more skip) and on a **better keyframe**. That lets it run a **finer base quantiser**. Its references are then better, and the advantage compounds frame after frame.

We code about twice as much picture area with residual, at a coarser quantiser. Our quality decays faster through the prediction chain: on bodycam the luma gap grows from 0.2 dB in the first frames to 2.7 dB by the end of the clip.

## P frames (mean per frame)

| | bodycam: SVT | bodycam: ours | low-light: SVT | low-light: ours | crowd: SVT | crowd: ours |
|---|---|---|---|---|---|---|
| bytes/frame | 1105 | 1088 | 1733 | 1801 | 1272 | 1267 |
| base qindex | **88** | 107 | **128** | 144 | 228 | 233 |
| PSNR-Y (dB) | **47.09** | 45.02 | **37.44** | 36.39 | 27.25 | 27.29 |
| PSNR-U/V (dB) | 47.0/46.8 | 47.7/47.5 | 42.6/42.5 | 42.8/42.9 | 36.3/35.3 | 36.9/36.0 |
| skip area | **89%** | 78% | **93%** | 85% | **90%** | 82% |
| coded (non-skip) transform area | **11%** | 22% | **7%** | 15% | **10%** | 18% |
| NEW area | **15.8%** | 4.9% | **22.1%** | 11.6% | **14.8%** | 7.2% |
| NEAREST / NEAR area | 66% / 18% | 81% / 14% | 58% / 20% | 67% / 21% | 77% / 6% | 83% / 7% |
| NEW blocks/frame | 99 | 33 | 257 | 114 | 216 | 115 |
| sub-pel share of inter MVs | 43% | 38% | 58% | 64% | 18% | 17% |
| 64×64 / 32×32 / 16×16 area | 67/30/1% | 61/36/3% | 58/32/8% | 21/74/5% | 73/16/9% | 62/26/12% |
| bits: coefficients | 6732 | 7059 | 9776 | 11387 | 6032 | 6654 |
| bits: motion vectors | **724** | 213 | **1866** | 792 | **2225** | 1090 |
| bits: inter mode (NEAREST/NEAR/NEW/DRL) | 656 | 561 | 1070 | 1290 | 665 | 967 |
| bits: partition | 237 | 246 | 424 | 212 | 329 | 405 |
| bits: CDEF | 22 | **0** | 36 | **0** | 0 | **0** |
| bits: header / unaccounted | 458 | **616** | 596 | **711** | 651 | **776** |

## Keyframe

| | bodycam: SVT | bodycam: ours | low-light: SVT | low-light: ours | crowd: SVT | crowd: ours |
|---|---|---|---|---|---|---|
| bytes | **29 618** | 26 194 | **113 777** | 98 187 | 30 116 | 30 267 |
| qindex | **16** | 24 | **36** | 49 | 152 | 157 |
| PSNR-Y (dB) | **54.78** | 53.94 | **45.88** | 44.46 | **35.47** | 35.01 |
| block sizes (area) | 64×64 25%, 32×32 61%, 16×16 12% | **16×16 100%** | 32×32 50%, 16×16 47% | **16×16 100%** | mixed 64…16, incl. rectangles | **16×16 100%** |
| intra-mode bits | 879 | 191 | 6415 | 191 | 3144 | 189 |

## How quality develops over the clip (mean PSNR-Y per window, dB)

| Frames | bodycam: SVT | bodycam: ours | low-light: SVT | low-light: ours | crowd: SVT | crowd: ours |
|---|---|---|---|---|---|---|
| 1–10 | 50.64 | 50.42 | 40.27 | 39.33 | 29.51 | 29.61 |
| 11–50 | 48.09 | 47.14 | 38.38 | 37.34 | 27.87 | 28.17 |
| 51–100 | 47.42 | 45.79 | 37.55 | 36.54 | 27.17 | 27.23 |
| 101–200 | 47.17 | 44.86 | 36.98 | 35.93 | 27.04 | 26.99 |
| 201–end | 46.09 | **43.40** | 36.78 | 35.61 | 27.01 | 27.03 |

**SVT's 4-frame effort cadence** (mean bytes by frame index mod 4, same qindex on every frame):

| Clip | SVT | Ours |
|---|---|---|
| bodycam | 1216 / 969 / 1106 / 1129 | 1034 / 1104 / 1042 / 1169 |
| low-light | **2604** / 1235 / 1513 / 1595 | 1789 / 1780 / 1801 / 1834 (flat) |
| crowd | **1623** / 1084 / 1196 / 1191 | 1278 / 1263 / 1266 / 1263 (flat) |

## Ranked explanation and proposed fixes

**1. We code residual where SVT improves the prediction.**
- **Evidence:** at equal size, we code transform residual on about twice the area. We pick NEW on a third to a half of SVT's area, and spend 30–45% of SVT's motion-vector bits. Coefficient bits are similar, so SVT pays for its extra vectors by coding residual over less area. That buys a finer base qindex (−19 on bodycam, −16 on low-light).
- **Why it hurts:** the finer quantiser makes better references, which is why the gap grows over the clip (bodycam +0.2 → +2.7 dB). It matters most at high quality (bodycam, low-light) and is neutral at crowd's 27 dB.
- **Fix:**
  - Rank NEW the way SVT's light-PD1 does, with an approximate inter-mode rate rather than our exact motion-vector bit cost, so better vectors win more often (docs/reference/svt-av1-m12-rtc.md §3).
  - Bias towards skip and against small residuals when the prediction is good.
  - Track the coded-area share and NEW share as the measures to move towards SVT's.

**2. Our keyframe is weaker: fixed 16×16, DC-only intra.**
- **Evidence:** SVT partitions keyframes into 64/32/16 blocks (and rectangles), spends real intra-mode bits, and codes a 13–16% larger keyframe at a lower qindex, for +0.8 to +1.4 dB.
- **Why it hurts:** most P-frame area is skipped (78–93%), so keyframe quality carries forward into every later frame.
- **Fix:**
  - VLPD0-style partition planning for intra frames;
  - chroma-from-luma (SVT M12 luma remains DC-only; broader luma modes are an additional experiment);
  - check the keyframe qindex mapping (SVT: 88 → 16; ours: 107 → 24).

**3. No effort cadence.**
- **Evidence:** SVT spends more on every 4th frame at the same qindex (low-light +75%, crowd +40%, bodycam +15%), through a gentler light-PD1 or skip level on its "base" frames. Ours is flat. This periodically refreshes reference quality.
- **Fix:** a base-frame cadence. The AV1 agent's `av1.skip_bias_pct_base` is one form of it; the light-PD1 level, sub-pel and CDEF search per cadence position are others.

**4. Per-superblock CDEF.**
- **Evidence:** our CDEF signals 0 bits, so the whole frame uses one strength. SVT signals a per-superblock index (2 strengths; 22–36 bits per P frame).
- **Fix:** 2-strength CDEF with per-superblock selection, as in SVT level 9.

**5. Unaccounted coding bits; header hypothesis ruled out on saved samples.**
- The inspection residual is 120–160 bits larger per P frame, but it is
  actual frame bits minus libaom's symbol-accounting estimates, not a direct
  measurement of header length. It cannot establish removable header syntax.
- Fresh read-only `ffmpeg trace_headers` inspection of the first16 saved
  diagnostic frames on bodycam_airport, crowd_mot17_09 and lowlight_pp22_004
  finds our uncompressed P header **112 bits**, including byte alignment;
  SVT's is **120–152 bits**. Key headers are64 bits vs80–104. OBU framing
  and tile payload are excluded from these header lengths. No TEST inputs
  or new encodes were used, and this is a syntax diagnosis, not a legal-rate
  quality comparison.
- Evidence: work/overnight/actual-header-comparison.json, source hashes and
  per-field traces; reproducible with work/overnight/compare-headers.py.
  The next investigation is symbol-cost accounting/entropy overhead; do
  not spend effort on a speculative header rewrite based on the residual.

**Not a cause:**
- **Chroma:** ours is slightly better than SVT's, a side effect of the coarser luma qindex. Making chroma coarser was measured worse.
- **Sub-pel precision:** our sub-pel share of inter vectors is similar to SVT's.

## Limitations

- **Coefficient bits are not split by plane:** libaom's accounting labels symbols by syntax element only.
- **"Header / unaccounted" is derived:** it's frame bytes × 8 minus the accounted symbols, so it includes any symbol libaom doesn't account for.
- **Single run:** one QP per clip, one matched point. The direction of every finding is the same on all three clips.
