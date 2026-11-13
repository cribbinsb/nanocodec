# Initial VP9 efficiency diagnosis

Baseline: `e561e10` (rebased onto main `4cad91d`). This is a bounded diagnostic,
not a parity or BD-rate result. Three existing quick360 clips, first 16 frames,
640×360, 30 fps, one layer, GOP 300. Native and libvpx v1.14.1 realtime CPU8
use the same inputs; reference is one thread, no lookahead/altref/AQ/row threading.
Native qindex 96/160 maps to reference CQ 24/40. PSNR below is average frame
YUV PSNR (6:1:1). All 12 streams passed own and both standard decoder gates.

| Clip | qindex | Native kb/s | Reference kb/s | Native PSNR | Reference PSNR | Source cap kb/s |
|---|---:|---:|---:|---:|---:|---:|
| static_chirla_s000c3_360p | 96 | 2480.190 | 733.920 | 41.4823 | 40.4051 | 585.175 |
| static_chirla_s000c3_360p | 160 | 1407.660 | 268.305 | 35.5621 | 34.7627 | 585.175 |
| crowd_mot17_09_360p | 96 | 5279.595 | 3398.190 | 40.8377 | 39.9218 | 374.750 |
| crowd_mot17_09_360p | 160 | 2760.435 | 1202.640 | 34.3926 | 34.0704 | 374.750 |
| pan_jaad_0117_360p | 96 | 7019.100 | 4722.180 | 41.6021 | 40.9164 | 594.288 |
| pan_jaad_0117_360p | 160 | 3702.030 | 1437.630 | 34.6196 | 34.6009 | 594.288 |

**All points except static reference q160 exceed the required half-source-bitrate
cap.** They cannot establish in-domain quality or production defaults. The static
q96 P-frame bytes are 7.41× reference, but native quality is 1.08 dB higher; this
is an unmatched-quality diagnostic. Pan q160 is almost equal PSNR with 2.58×
bytes, but both points remain out of domain. Two points do not support BD-rate.

Raising static native quantizer to 200/240 still yields 1237.005/1142.625 kb/s
(31.8284/28.2057 dB), both above cap. The bitrate floor calls for syntax diagnosis
before spending more runs on curves.

## Quantized-zero residuals were not explicitly skipped

The original encoder emitted skip only when prediction exactly equaled source.
On static frames 3–11, q160 averaged 3289.8 all-plane-zero-residual blocks out of
3600, but zero explicitly skipped blocks. At q240, 3532.7 blocks had zero residual
and only one was explicitly skipped. Coefficient EOB syntax was needlessly sent.

A test-only replay matched every original tile byte and reconstructed pixel,
then forced skip for these already-zero blocks. Own/libvpx/FFmpeg decoded all
three modified 16-frame streams to exactly the original filtered reconstruction:

| Static qindex | Original total bytes | Explicit-zero-skip bytes | Reduction |
|---|---:|---:|---:|
| 96 | 165346 | 156169 | 5.55% |
| 160 | 93844 | 77586 | 17.32% |
| 240 | 76175 | 57823 | 24.09% |

Loop filtering is secondary here: at q160 and q240 it slightly improves source
SSE, and replacing the previous filtered reference with its prefilter pixels
slightly increases nonzero luma blocks. There is no reconstruction disagreement
with reference decoders suggesting a transform scaling bug.

Validated production fix: cache all quantized inter residuals before
writing skip syntax; emit skip when all coefficients are zero; reuse those
coefficients when coding. Exact predictions still avoid transforms entirely.
Zero-coefficient reconstruction avoids inverse transforms. Effective skip maps
used by neighboring syntax contexts and filtering are unchanged. Native 30 unit
tests and four integration tests pass; all 640 frames in the batched matrix
match own/libvpx/FFmpeg reconstruction. The three real-clip production streams
exactly match the independently forced-skip hashes above, allowing verified
gate-cache reuse.

## Nearest-vector signaling removes the next syntax cost

Pinned libvpx v1.16.0 `vp9_entropymode.c` gives ZEROMV the first false branch;
its default probability is 2/256 for the both-zero-neighbor context. Repeated
ZEROMV therefore costs roughly seven bits per 8×8 block, explaining much of the
remaining static floor. NEARESTMV uses the next branch and can represent the
same zero vector much more cheaply. The encoder now uses NEARESTMV whenever the selected vector equals the
existing nearest predictor, including zero. Neighbor mode counters distinguish
nearest (0), new (1), and zero (3), matching libvpx. Uniform8 single-LAST-reference
blocks use above then left, so the existing predictor already is the normative
nearest candidate; motion search and its costs are unchanged. Decoder support
accepts nearest plus the previously emitted zero/new modes.

| Static qindex | Baseline bytes | Skip-only bytes | Skip + nearest bytes | Total reduction | Final kb/s | In domain? |
|---|---:|---:|---:|---:|---:|---|
| 96 | 165346 | 156169 | 118417 | 28.38% | 1776.255 | No |
| 160 | 93844 | 77586 | 40368 | 56.98% | 605.520 | No |
| 240 | 76175 | 57823 | 20963 | 72.48% | 314.445 | Yes |

All three final streams passed full own/libvpx/FFmpeg gates. Cached independent
per-frame hashes establish exact filtered reconstruction equality with the
skip-only streams (and therefore the original baseline). The full 640-frame
matrix passes, 32 unit tests plus four integration tests pass, and strict VP9
Clippy passes. A new original-C oracle checks mode-tree output across all seven
probability contexts; native tests cover nearest/new/zero vector and counter
roundtrips. This fixes demonstrable overhead but does not establish reference
quality parity: only the final q240 point is in domain, and its quality is low.

No speed claim is made under concurrent machine load. Artifacts are compressed
IVF/JSON/text under `nanocodec-data/work/vp9-initial-efficiency-e561e10`;
`tools/dev/vp9_efficiency_probe.py` reuses shared streamed metrics, gates, cache
and CPU slots. No raw video copies were created.


## Adaptive 16×16 inter blocks

After rebasing onto main `0f9c7d7`, added full 16×16 inter coding with four TX8
luma transforms and one TX8 per chroma plane. Existing shared DCT8 kernels and
coefficient coding are reused. The cheap merge screen tests zero and nearest
predictors via shared DSP SSE; `vp9.merge16_sse_per_lambda` defaults to 0.5 and
zero disables merging. Other blocks retain the existing 8×8 search. This
threshold is empirical and remains available for joint search.

Partition contexts use normative size masks; motion and skip state cover all
constituent 8×8 cells. Mixed-size filtering preserves coding boundaries, omits
internal luma TX8 boundaries for skipped 16×16 blocks and uses TX8 chroma edges.
Keyframes and lossless coding retain 8×8 coding blocks / TX4.

A bounded four-point screen used the same 16-frame inputs, q240 and full gates:

| Clip | 8×8 bytes | Mixed bytes | Reduction | 8×8 YUV PSNR | Mixed YUV PSNR | 8×8 / mixed kb/s | Domain |
|---|---:|---:|---:|---:|---:|---|---|
| static | 20963 | 12185 | 41.87% | 28.2057 | 28.6758 | 314.445 / 182.775 | Both below cap |
| pan | 74688 | 67086 | 10.18% | 26.1217 | 26.9731 | 1120.320 / 1006.290 | Both above cap |

Static luma changes by −0.0078 dB and pan luma by +0.0179 dB; most PSNR gains
come from chroma TX8. Unlike the earlier syntax-only fixes, reconstruction can
change. Disable mode reproduces the previous static bitstream hash exactly.
This small screen supports the tool, not a tuned default or BD-rate comparison.
No CPU speed claim is made under load.

Validation: 33 native unit tests, four integration tests, the full 640-frame
independent decoder matrix, and strict Clippy pass. The original-C oracle adds
48 frame/plane hashes for mixed 8×8/16×16 filter scheduling, including partial
superblocks and skipped/non-skipped blocks. A regression exercises nonzero
luma and both chroma residuals, mode contexts across two larger blocks, and
merge-disable fallback. Final workspace CI/WASM checks are coordinated separately.
