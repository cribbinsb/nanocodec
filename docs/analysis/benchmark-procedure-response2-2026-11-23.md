# Response to the second hostile review (2026-11-23): BP-3

> **Superseded by BP-3.10** ([benchmark-procedure.md](../benchmark-procedure.md), 2026-11-24). This dated response is kept as the record of its review; where it differs from the current procedure, the procedure governs.

This answers every item of
[benchmark-procedure-review2-2026-11-23.md](benchmark-procedure-review2-2026-11-23.md) ("review
2"). The procedure is now BP-3 ([docs/benchmark-procedure.md](../benchmark-procedure.md)) with
its fairness record ([docs/benchmark-fairness.toml](../benchmark-fairness.toml)). The
coordinator's decisions (made under the owner's delegation, 2026-11-23) are listed at the end.
Measured where measurable; every number below is measured today unless marked.

## How the measurements were made

- **The jobs are the record's own commands.** Every encode was planned by `tools/bench/plan.py`
  (new, landed), which takes each command from `nceval.fairness.command` and the fairness record.
  This is the first use of the record by a run (review 2, "cannot drift").
- **Clips:** the response's 20 clips, used for design: 4 main-chart 720p clips and 16
  `heldout_2026-11-23` clips, which now leave the sealed set (N2). The resize re-measure (N5) used
  the 11 main-chart clips, which are not sealed.
- **Settings:** the interim phase (F = 1.0 s), SEI stripped (its bytes subtracted from the first
  frame), 150 frames.
- **Ladder:** BP-2.1's rungs, 40-4000 kbps at 720p30 scaled by (pixel rate)^0.75. The owner's
  BP-3 ladder (150/300/600/1200 with pixels^0.75 x (fps/30)^0.5) differs only in the frame-rate
  exponent. On these 20-30 fps clips that is at most 4.7 % in target (25 fps), so the BP-3
  headline numbers below reuse these encodes at 150-1200.
- **Boxes:**
  - Quality, rate and QP inspection on c7a.24xlarge spot, packed (no timing).
  - Timing on c8a.large on demand, one encode at a time on vCPU 1, 3 clips, 5 rounds.
  - Decoding with the size-change path: a stream whose frame sizes vary is decoded at each
    frame's size (aomdec/vpxdec) and upscaled by the pinned player model (`nceval.display`'s
    filter); every other stream through ffmpeg.
- **Spend:** about $1.6 of AWS for this round (ledger, `Purpose=bench-method`); no instance left
  running.
- **Records:** `~/nanocodec-data/bench-method-records/` (kept).

## The BLOCKERs

### N1. The headline clip set: measured; the zero-tolerance intersection rule is withdrawn

**Valid share per family and rung, first run on BP-2.1's 40-4000 ladder** (peak at most one buffer
from F = 1.0 s, and bits within R x T + F x R + 2 %; record commands with the examples' max-q 52;
20 clips):

| Family | 40 | 75 | 150 | 300 | 600 | 1200 | 2400 | 4000 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ours, 8 families (`--strict-buffer`) | 90-100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| x264 ultrafast / veryfast | 100 / 90 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| x264 medium (all tunes) | 20-25 % | 45 % | 95 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| x265 ultrafast / superfast | 90 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| x265 medium | 40-45 % | 95 % | 100 % | 100 % | 100 % | 100 % | 100 % | 100 % |
| SVT-AV1 M8-M13 | 25-40 % | 35-65 % | 80-95 % | 90-100 % | 90-100 % | 93-100 % | 100 % | 100 % |
| Kvazaar ultrafast-veryfast (**void**: the record gave Kvazaar bits per second as kbps, a target 1000x too low; found by the BP-3.1 pipe-clean run) | - | - | - | - | - | - | - | - |
| openh264 (3 twins) | 0 % | 5-10 % | 25 % | 45-65 % | 90 % | 100 % | 100 % | 100 % |
| libvpx rt 7/9 (record = the example's flags) | 0 % | 0 % | 0 % | 15-20 % | 40-45 % | 93 % | 100 % | 100 % |
| libaom rt 8/10 (record = the example's flags) | 0 % | 0 % | 0 % | 10 % | 35-50 % | 87-100 % | 100 % | 100 % |
| VVenC low-delay faster/fast (one-pass RC is VBR, no buffer) | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % | 0 % |

**Re-measured on the final ladder** (owner: 300/500/800/1200 kbps at 720p30, sub-linear scaling;
run r3fin, 20 clips, every record command including the added presets, libvpx/libaom at their
default max-q 63; 75/150 coded for reference only, the very-low view is dropped). Valid share,
grouped:

| Family | 300 | 500 | 800 | 1200 |
|---|---:|---:|---:|---:|
| ours, 8 families | 100 % | 100 % | 100 % | 100 % |
| x264 ultrafast-medium, x265 ultrafast-medium (all tunes) | 100 % | 100 % | 100 % | 100 % |
| Kvazaar ultrafast-veryfast (**void**, as above) | - | - | - | - |
| SVT-AV1 M8-M13 | 90-100 % | 90-100 % | 83-100 % | 93-100 % |
| openh264 (3 twins) | 55-65 % | 80-95 % | 94-100 % | 100 % |
| libvpx rt 7/9 (max-q 63) | 65-70 % | 90 % | 94 % | 100 % |
| libaom rt 8/10 (max-q 63) | 60-65 % | 95 % | 94 % | 100 % |
| VVenC (one-pass RC is VBR; charted in Fixed QP only) | 0 % | 0 % | 0 % | 0 % |

**Surviving clips:**
- 15 of the 20 clips have four legal rungs under the 50 % source-rate rule. The other five have
  two or three, because low-rate sources cap them.

| Rule (all fixed in advance) | Clips surviving, of 20 |
|---|---:|
| R0 BP-2.1: zero tolerance, 4 valid points per family, one clip set for all families | **0** (VVenC 0; without VVenC at most 8: openh264 and libaom rt 10 have four valid points on only 8 clips) |
| R1 the same with 1.1 buffers tolerated | 0 (without VVenC at most 8) |
| R1b with 1.25 buffers | 0 (without VVenC at most 8) |
| R3 a bits-per-pixel floor: 0.01 / 0.02 / 0.03 bpp | 15 / 1 / 0 clips with four rungs at all, 0 all-valid |
| **R2 all points count; compliance is its own published axis** | **15** (= every clip with four legal rungs) |

- **Valid share by target bits per pixel** (every reference pooled):
  - 59 % below 0.005 bpp;
  - 79 % at 0.005-0.01;
  - 90 % at 0.01-0.02;
  - 94-97 % at 0.02-0.16.
- So no bits-per-pixel floor that keeps the owner's ladder makes every real-time reference
  compliant.
- The interim 150-1200 ladder gave the same: R0-R1b 0 clips, R2 15 (run r3, with the examples'
  max-q 52).

- **The reviewer was right on both counts.** The zero-tolerance intersection leaves no clips, and
  it is a rule calibrated to our re-coding (`--strict-buffer`).

**Resolution (BP-3 section 7), fixed in advance and not dependent on results:**
- The headline BD uses every encoder's full curve: all points, nothing excluded for rate. The
  one-clip-set rule then excludes only clips without four legal rungs (counted per set from the
  manifests before the run) and clips where some curve fails the numeric curve rules.
- **Compliance is its own published axis.** Per encoder, the share of points over one buffer and
  over rate, the 95th-percentile peak, and the first frame's size in buffers. It is printed beside
  every BD number, and the chart marks an encoder with under 90 % compliant points with a hollow
  marker labelled "not buffer-compliant".
- A secondary view shows BD over each pair's both-compliant points. It is labelled and is never
  the headline.
- Re-coding is our rate controller's mechanism. Its time is counted, and the record says so.
  Validity no longer decides which clips count.

**libvpx and libaom quantizer limits: the record now uses their defaults (max-q 63).**
- The example programs pin `rc_max_quantizer = 52` and stay within rate by dropping frames.
  BP-3 turns frame dropping off for every encoder, so with max-q 52 both overshoot most points at
  or below 600 kbps (table above).
- With their default max-q 63 (BP-2's old CLI commands, response 1), they were over one buffer
  on 27-30 % of ladder points instead.
- Keeping the example's 52 without its frame drop would handicap them. The record keeps every
  other example setting and says so.

### N2. The sealed set was used for design: accepted and done in part

- The 16 clips are logged as the first entry of `docs/results/sealed-log.md` and listed in
  `eval/clipset/sealed_exclusions.json`.
- `tools/bench/plan.py --sealed` refuses any of them (tested: "refused: 16 clips were seen before
  sealing").
- All further design measurement uses pool or main-chart clips; N5 below already did.
- **Done: re-sealed as `heldout_2026-11-23b`** (sealed-log entry 2), with 97 clips:
  - the 84 unseen kept clips of the first set;
  - 13 redrawn by the written rule. `select_heldout.py redraw` uses seed `heldout_2026-11-23b`,
    takes only families the first draw never drew, and sets each class's quota to its target
    less its kept unseen clips, with pan and crowd raised to 12. The clips were then built,
    graded and put through `finalize-redraw`.
- **By class:** bodycam 15, crowd 12, drone 13, lowlight 11, pan 1, static 15, texture 15,
  videocall 15.
- **Short of quota:**
  - pan by 11: the pan class is Caltech only, and every Caltech set is already drawn;
  - lowlight by 4 and drone by 2: no families left.
  - A search for an unused pan-like collection found none (sealed-log entry 2a), so the set stays
    at 97. The headline weights clips equally, so the imbalance does not bias it. Per-class
    results are reported only for classes with at least 8 clips; pan (1), lowlight (11) and
    drone (13) are stated as thin.
- `plan.py --sealed` passes on the new set (373 jobs for x264 medium) and refuses the old one.

### N3. Fixed QP: one definition, each encoder's mode verified and measured

**Definition (coordinator):** each encoder's no-rate-control mode with a fixed base quantizer, its
own frame- and block-level QP behaviour allowed.

| Encoder | Mode in the record | Verified how | Measured at base QP 32 (H.264/HEVC) or q 40 / qindex 128 |
|---|---|---|---|
| x264 | `--qp` CQP | Source at the pinned commit (encoder/encoder.c 951-966): in CQP `i_aq_mode = 0`, `b_mb_tree = 0`, key QP = qp - 6 log2(ipratio) | Key QP 29, P frames 32 on every frame, block spread 0. Key/P bytes 8.8x |
| x265 | `--qp` CQP | Source (encoder.cpp 4049-4056: AQ, hevc-aq, AQ-motion, CU-tree off; ratecontrol.cpp 345-350: I-slice QP = qp - ipOffset) | Key 29, P 32, block spread 0. Key/P 8.9x |
| SVT-AV1 | `--rc 0 --aq-mode 0 --qp` (true CQP; BP-2.1 ran CRF) | Its `--help`: "0: CRF or CQP (if `--aq-mode` is 0)" | `aom-inspect` at `--qp 40` (base qindex 160): key qindex 67 (median of 20 clips), every P frame 160, no block variation |
| libaom | `--end-usage=q --cq-level` (native constant quality, unpinned) | `aomenc --help` | At `--cq-level=40`: key qindex 72, P frames 115-160 (golden-frame boosts), no block variation |
| libvpx | `--end-usage=q --cq-level` (native constant quality, unpinned) | `vpxenc --help` | Key/P bytes 15.7x (no QP inspector for VP9) |
| Kvazaar | `--qp` | `kvazaar --help` | Key/P bytes 3.9x |
| VVenC | `-q`, low-delay faster | `vvencFFapp --fullhelp` | Key/P bytes 21.9x |
| **ours** | `--qp` (cadence, key offset, block offsets) | `nanocodec stat` (analysis build) on our streams | **H.264:** key QP 30, P frames mean 33.6 (frame means 30-40, the QP cadence), block spread up to 7 in some frames. **HEVC:** key QP 23 (9 finer), P mean 31.6 (25-33), no block spread. **AV1** (`aom-inspect`, qindex 128): key qindex 36, P frames 85-138 (cadence), block qindex spread up to 48 within a frame (median 2). Key/P bytes 27x (H.264), 20x (HEVC), 13.5x (VP9) |

- **Every AV1 encoder codes the key much finer than its base**: SVT -93 qindex, libaom -88,
  ours -92. The H.264/HEVC references keep 3 QP. **Ours has the most frame- and block-level
  variation in fixed-QP mode**: a P-frame cadence of 85-138 qindex in AV1 and 30-40 QP in H.264,
  block offsets up to 48 qindex, and a HEVC key 9 QP finer. That is "its own behaviour", allowed by the
  definition. It is now measured and stated in the record. The reviewer's point stands that this
  behaviour was developed on this 150-frame one-key window.
- `rate.constant_qp=1` (every frame and block at one QP) stays available as the labelled
  "constant QP" sensitivity row.

### Also measured

- **N6** (the record's commands run as written): **BD**, example program against the record's
  CLI command, 150-1200 kbps:
  - libaom speed 10: -0.17 % PSNR-YUV, +1.41 % VMAF (14 clips);
  - libaom speed 8: +0.24 / +3.98 %;
  - libvpx speed 9: -1.78 / -2.21 % (15 clips);
  - libvpx speed 7: -2.86 / -2.62 %.

  So the CLI commands carrying the examples' flags are within 3 % of the examples' efficiency
  (libvpx's example slightly better), and they overshoot the same way: median rate 1.29x (aomenc)
  against 1.21x (example), 1.58x against 1.41x. **Speed** (c8a.large, one encode at a time on vCPU 1, 3 clips x 5 rounds, median CPU time; the
  CLI commands with the examples' max-q 52):
  - the libaom example takes 0.94x (speed 10) and 0.90x (speed 8) the CLI command's time;
  - the libvpx example takes 1.01x (speed 9) and 0.99x (speed 7).
  - So the CLI command is 6-11 % slower for libaom, and equal for libvpx.
  - **Why: references** (libaom `examples/svc_encoder_rtc.cc` at b2347b8, and the streams).
    - The example runs libaom's SVC path with layering mode 0, which refreshes and references
      LAST only (`set_layer_pattern`, case 0). The CLI's real-time path keeps its default
      reference set.
    - Measured with `aom-inspect -r` on 10 frames of a 720p clip at speed 10 and 600 kbps. The
      example's inter blocks use LAST only. The CLI's blocks use LAST, plus compound LAST +
      ALTREF on 2 % and GOLDEN on 0.1 % of blocks, so it searches more references per block.
    - This matches the BD: the CLI is 0.2-4 % better on VMAF at speed 8, and about equal at
      speed 10 (N6 above).
  - **Ruled out:**
    - tiles: `--auto-tiles=1`, the example's setting, gives a byte-identical CLI stream at 720p;
    - flags: every other control in the example has the same value in the CLI command (record,
      kept/dropped list);
    - threads: both use 1;
    - I/O: both read raw I420 from tmpfs;
    - build: both binaries come from one libaom build.
  - **The confirmation did not confirm it** (T14, c8a, 3 clips, 5 rounds, max-q 52):
    - the CLI with `--reduced-reference-set=1` takes 0.99x / 1.01x its default time (speed 10 / 8);
    - with `--max-reference-frames=3`, 1.01x / 1.01x;
    - with `--auto-tiles=1`, 1.00x / 1.00x;
    - the example takes 0.91x / 0.86x.
    - Neither flag reaches the real-time path's reference choice, so the gap is still not
      attributed. The example's LAST-only references are the measured difference in the
      streams, but not a proven cause.
  - **Closed by BP-3.1** (coordinator decision 4): libaom's CBR rows run the example program
    itself, minus frame dropping. The gap now applies only to the Fixed QP rows (the CLI, with
    the example's real-time tool flags), and is stated there.
  - The record keeps the CLI, which `plan.py` drives, with its default references. That is
    libaom's real-time default and its better BD.
- **N5** (resize under the pinned upscaler, main-chart clips, 40-1200 kbps; dynamic resize on
  against off):
  - libaom rt 10 / 8: +7.12 / +6.98 % PSNR-YUV, +11.2 / +14.5 % VMAF worse with resize on;
    46 of 64 streams switch, and 18.5 % of frames are coded smaller.
  - libvpx rt 9 / 7: +18.5 / +19.6 % PSNR-YUV, +29.3 / +33.0 % VMAF worse; 52-53 of 64 streams
    switch, 75 % of frames smaller.
  - SVT-AV1 M12: -0.03 % / +0.20 %; 9 of 58 streams switch, 3.3 % of frames.
  - Ours AV1 with adaptive resolution off: +2.08 % PSNR-YUV, -0.12 % VMAF (medium); +1.21 /
    -0.35 % (veryfast). 12 of 64 and 8 of 64 streams switch.
  - So under the pinned filter too, every reference does worse or the same with its resize on,
    and ours gains about 2 % PSNR and nothing on VMAF from its own. References keep their
    examples' resize-off; ours keeps adaptive resolution on. Our AV1 rows with it off are
    published beside the default.
  - The sensitivity to the scaler (bilinear, bicubic, Lanczos) is **not measured**; it is task
    T11.
- **Our `x86-64-v3` build** against baseline `x86-64`, same tree (c8a.large, streams byte-identical): the `x86-64-v3` build is 0.1-4.8 % faster.
  The baseline takes 1.024x (H.264 medium), 1.032x (H.264 veryfast), 1.001x (HEVC), 1.048x (AV1)
  and 1.037x (VP9 veryfast) the v3 build's time. The charts use the v3 build, stated in the
  manifest.
- **Kvazaar and VVenC speed** on c8a.large, median seconds per 720p 150-frame encode:
  - Kvazaar ultrafast 1.06, superfast 1.83, veryfast 1.85;
  - VVenC faster 57.9, fast 125.9 (one round);
  - x265 medium 3.51;
  - libaom rt 10 / 8: 0.86 / 1.18;
  - libvpx rt 9 / 7: 0.68 / 1.11;
  - x264 medium: 1.85, from the study.

## Review 2's other items

| Item | Response |
|---|---|
| Part 1, stale text (§1 principle 3, §13, §5 SVT `--asm max`, §4 qindex points) | Fixed in BP-3: commands come from the record through `plan.py`; SVT's banner, not `--asm max`; the qindex table is `conditions.py`'s (110-235, q 28-59). |
| B3: interim phase unmeasured for the references | The interim phase is now measured for every reference (N1 table, F = 1.0 s). Nothing is published at the interim phase (BP-3 §5). The two phases never share a chart. |
| B4 gap: AV2 | `nceval.streamstrip` now strips AV2 metadata OBUs (types 8 and 9, from dav2d's `DAV2D_OBU_METADATA*`; AV2 OBU size includes the header), with a test. It also states that every SEI goes, buffering-period and picture-timing included (N9). |
| B6 §3 stale row | The row now says the top-up is decided and drawn by the rule, less the 16 seen clips. |
| M1 manifest fields, bias | Still open: T7 (bias) and the manifest fields (`heldout_2026-11-23b` will carry them). |
| M3 builds | PGO for every reference that supports it, once per version and target (T4, now including Kvazaar); `x86-64-v3` measured above. |
| M4 Fixed QP | N3. |
| M5 / N6 libaom, libvpx | Every example setting listed in the record as kept, default or dropped, with the reason; the CLI commands measured against the examples (above). |
| M7 timing | BP-3 §5 says cpuset cgroup; the decision table no longer says `taskset`. Still unbuilt (T10). |
| M9 / N4 zero line | Decided: x264 medium with the metric-matched twin per single-metric BD, the default tune for BD-J, the tune in every label, default and PSNR-tune numbers side by side in any headline (record `zero_line`). |
| VVenC placement | **Charted in Fixed QP only** (coordinator): its one-pass RC needs a look-ahead whose latency is not measured as negligible. |
| M10 references | Kvazaar `--no-psnr --no-wpp --no-info` added; VVenC `--MaxBitrate` set to 1.5x the target, the lowest VVenC accepts ("MaxBitrate must be at least 1.5*TargetBitrate", tested); `--IntraPeriod 300` with GOPSize 8 tested and accepted (one intra picture, log kept). VVenC's one-pass RC requires its look-ahead and is VBR: it is 0 % buffer-compliant on every rung and is charted with that compliance shown. |
| M11 / N8 content id | ParkRun, Shields, Mobcal and Stockholm form SVT's "ter" shoot (the reviewer's knowledge, adopted as the safe side), so AOM CTC A4's Stockholm collides with PGO's parkrun (test). Source-path comparison added (`source_overlaps`, test). |
| M12 pre-registration | N2; `plan.py` exists and refuses sealed exclusions; the claim "cannot drift" now holds for planning (commands) and is stated as such until the box and report scripts exist. |
| x264 `--tune` entry vs §8 | Both now follow N4's rule. |
| x264 `--vbv-init` handicap | The final phase's 0.6 costs x264 medium 1.39 % BD-PSNR-YUV (response 1); stated in the record beside the final phase. |
| Presets missing (x264 superfast/faster/fast, x265 veryfast/faster/fast, SVT M9/M11) | Added to the record: every real-time preset between each reference's fastest and medium is charted, so the reference fronts are not interpolation (record `families`, T10 pipe-clean). |
| x265 AVX-512 banner check | V21. |
| openh264 version | v2.7.0 (`versions.env`); harness commit 745b0166, SHA-256 in the manifest. |
| openh264 invalid by construction | Under N1's resolution it is charted on all points, with its compliance shown (25-65 % at 150-300 kbps). Allowing its frame skip would make it the only encoder that drops frames; the decision keeps skip off for all. |
| Kvazaar build | Only a plain build exists today; T4 builds it with LTO and PGO. |
| VVenC on a real-time chart | Its chart marker is labelled "not real-time, VBR with look-ahead". |
| Interim key sizes (ours key cap 1.0 vs the references' heuristics) | First-frame sizes per encoder at F = 1.0 s are in the records (`frame_bytes[0]`) and are published on the compliance axis (first frame in buffers). |
| V8 plan-stage box | The plan stage's unpinned single-thread audit runs on one c8a.4xlarge per run (16 cores, about 1 min). Costed in §15. |
| N7 re-cost | BP-3 §15, re-costed in full on the final ladder (300-1200, 4 rungs). The plan the owner chose: c8a and c6a on all three datasets, both modes; c8g on the history subset; each series scoring its own streams; VVenC in Fixed QP only; history and the windows. About **$119** with a 20 % contingency, within the raised $750 cap. Everything on c8g as well would be about $140 + $34. |
| N9 test gaps | The record test now checks Kvazaar's `--gop 0` and its no-WPP/no-PSNR/no-info flags, and VVenC's low-delay config, peak and intra period. |

## Owner question: smaller clip sets (60 instead of 100)

**Method:**
- Per-clip BD-J against x264 medium, from the recorded chart points: the newest point per
  (clip, family, rung) in `work/heldout-chart/cloud-results/f150-ip300`. This mixes builds where
  several exist, so read it as a sensitivity estimate, not the chart's numbers.
- Sets: heldout_2026-11-23 (80 clips have all families), all held-out (166) and VC-1K (92).
- 1,000 class-stratified draws of n clips (largest-remainder class quotas).
- Each draw is compared with the full set's mean. "Front" uses the published x86 fps.

| Set (clips with all families) | n | 95 % interval width, full -> n (median family, points of BD-J) | Families moving > 0.5 / > 1 point (mean share; worst family's probability) | Front identical to the full set's | Rank correlation, median (5th percentile) |
|---|---:|---|---|---:|---|
| heldout_2026-11-23 (80) | 60 | 7.1 -> 8.2 | 63 % / 33 % (worst 82 % / 67 %) | 16 % | 0.9991 (0.9979) |
| heldout_2026-11-23 (80) | 40 | 7.1 -> 9.8 | 77 % / 55 % | 7 % | 0.9982 (0.9951) |
| all held-out (166) | 80 | 3.7 -> 5.3 | 59 % / 28 % | 61 % | 0.9994 |
| all held-out (166) | 60 | 3.7 -> 6.1 | 67 % / 39 % | 49 % | 0.9991 |
| all held-out (166) | 40 | 3.7 -> 7.3 | 76 % / 53 % | 24 % | 0.9986 |
| VC-1K (92) | 80 | 4.9 -> 5.3 | 32 % / 7 % | 100 % | 0.9997 |
| VC-1K (92) | 60 | 4.9 -> 6.1 | 59 % / 30 % | 90 % | 0.9994 |
| VC-1K (92) | 40 | 4.9 -> 7.3 | 72 % / 48 % | 67 % | 0.9989 |

**Per-class counts left at n = 60** (draw quotas):
- heldout: pan 2, bodycam 11, texture 8, static 9, videocall 10, drone 7, crowd 5, lowlight 8;
- all held-out: crowd 14, static 18, pan 18, lowlight 3, bodycam 5, drone 1, texture 1;
- VC-1K: 60 video calls.

**Findings:**
- **Ranking barely moves** at any size (rank correlation at least 0.995).
- **Levels and the front move:**
  - at 60 clips about a third of families move by more than one BD point against the full set,
    and the front is identical in 16-90 % of draws depending on the set;
  - the 95 % interval widens by about sqrt(100/60), that is 29 %;
  - small classes fall to 1-3 clips, too few for a per-class figure.
- **Cost and time:** both scale linearly with clips. From §15's per-clip costs, 60 clips instead
  of 100 saves 40 % of a dataset-mode's encode and score, about $1 per mode on "ours" and
  $1.7-1.9 on VC-1K. Timing wall time is unchanged (bounded by one clip's box time, about 10 min).
- **Recommendation:** keep about 100 clips for "ours" and VC-1K; no smaller than 80. The saving
  is small in money, and the loss in level and front stability is large. AOM-CTC stays at all 34.

## Decisions taken (coordinator, under the owner's delegation)

1. **N4:** the zero line is x264 medium, with the metric-matched twin per single-metric BD and
   the default tune for BD-J. Recorded.
2. **Ladder:** 150/300/600/1200 kbps at 720p30, target = rung x (pixels/1280x720)^0.75 x
   (fps/30)^0.5, 50 % source cap; very-low view 40/75 scaled the same way. The 2400/4000 upper
   rungs are dropped. Examples (top rung): 1080p30 2205 kbps, 1080p60 3118, 720p60 1697, 360p30
   424, 270p30 276 (`plan.scaled_kbps`, checked against the owner's examples).
3. **History set:** every frame on the full headline ladder (now 150-1200).
4. **Screens:** `cbr.clip_targets` must use the same scaling (a change to land, listed in BP-3 §18
   with its effect).

## Changes to land, listed (not made here)

- **Screen rate scaling.** `eval/nceval/cbr.clip_targets` should use pixels^0.75 x (fps/30)^0.5
  instead of its current scaling. Effect: the screens' rates change only for clips away from
  30 fps; for 25 fps clips the target rises about 4.7 %. The screen clip set and its cost are
  unchanged in count, and encode time changes by well under 1 % (an estimate from the rate
  dependence measured in the start-up runs, not measured). It needs a re-baseline of the screen
  caches, since cached points are keyed by rate.
