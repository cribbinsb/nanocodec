# Initial VP9 and AV2 support: implementation plan

8 November 2026. Repository baseline: `40c910c`.
Status: **implementation underway**; see [v1 scope](vp9-av2-v1.md) and
the current implementation evidence. No benchmark result
is implied by a completed syntax or conformance milestone.
Companion: [codec expansion assessment](codec-expansion-vp9-vvc-av2.md).

## Intended result and order

Deliver native Rust VP9 and AV2 encoding, matching emitted-subset decoding,
CLI and WASM/demo integration, and measured speed/quality against permissively
licensed references. “Initial support” means useful forward-predicted camera
video with the existing realtime rate-control interface, not just a valid still
image or a wrapper around a reference encoder.

Start with one shared reference/conformance milestone. Then use VP9 as the first
end-to-end implementation because its 8-bit Profile 0 maps closely to the current
engine. Start AV2's syntax/subset investigation alongside that work if engineering
capacity permits; begin its implementation as soon as the AV2 oracle checks pass.
AV2 need not wait for VP9 performance tuning to finish. Keep shared API changes
small and reviewed together so the two backends do not develop incompatible
assumptions.

The plan deliberately does not require exhaustive pre-implementation benchmarking.
A small verified baseline and a legal subset are enough to begin. Broader tuning
comes after useful inter-frame output exists.

## Reference implementations and permitted uses

Pin **source commit, build configuration, compiler and binary hash** for every
anchor. License names below identify software licenses, not universal patent
clearance. Inspect the license and file headers at the actual pinned revision;
preserve notices and update `THIRD_PARTY.md` for adapted code. Follow the repository
rule of idiomatic Rust reimplementation, not wholesale transliteration.

| Implementation | License and primary source | Role |
|---|---|---|
| **libvpx** | BSD-3-Clause: [repository](https://github.com/webmproject/libvpx), [LICENSE](https://raw.githubusercontent.com/webmproject/libvpx/main/LICENSE), [PATENTS](https://raw.githubusercontent.com/webmproject/libvpx/main/PATENTS) | Primary VP9 syntax/reconstruction reference, `vpxenc` realtime quality/speed anchor, `vpxdec` oracle; inspect fast mode decision, probability handling, transforms, filters and SIMD. |
| **AVM, final-compatible release** | BSD-3-Clause-Clear plus separate AOM patent terms: [v1.0.0 source](https://github.com/AOMediaCodec/avm/tree/v1.0.0), [LICENSE](https://raw.githubusercontent.com/AOMediaCodec/avm/v1.0.0/LICENSE), [PATENTS](https://raw.githubusercontent.com/AOMediaCodec/avm/v1.0.0/PATENTS) | Primary AV2 normative implementation and encoder comparison. Pin v1.0.0 initially; investigate later compatible revisions separately. Use `avmenc`/`avmdec` as tools, not runtime dependencies of nanocodec. |
| **dav2d, VideoLAN** | BSD-2-Clause, identified by the [official project page](https://www-test.videolan.org/projects/dav2d/); [upstream source](https://code.videolan.org/videolan/dav2d) | Independent AV2 decoder candidate, decoder-speed anchor, and optimized reconstruction reference. Confirm exact revision compatibility with final AV2 before enabling the gate. |
| **rav2d** | BSD-2-Clause: [repository/status](https://github.com/stukenov/rav2d), [LICENSE](https://raw.githubusercontent.com/stukenov/rav2d/main/LICENSE) | Optional Rust design/kernel reference and extra diagnostic decoder. Evaluate selected safe Rust routines for adaptation; do not import its complete architecture automatically. |

AVM's Clear BSD copyright license explicitly excludes a patent grant; the separate
AOM patent document supplies its own terms. libvpx also has a separate patent
grant. Keep these documents distinct in dependency and attribution records.

The official dav2d page still contains draft-era language despite the final AV2
specification. Its BSD license statement is useful; that page alone cannot certify
final-bitstream compatibility. Its upstream raw source was inaccessible through
the web reader during this review, so checking `COPYING` at the chosen checkout
is an explicit prerequisite to adapting source.

rav2d describes itself as a port of dav2d, with assembly integration and remaining
AVM-vector mismatches. Its status is useful evidence, not a conformance certificate.
**dav2d and rav2d are not two independent implementation families.** For AV2, use
AVM plus a verified dav2d-family implementation as the two external checks.
[Current rav2d status](https://raw.githubusercontent.com/stukenov/rav2d/main/README.md).

For VP9, also use **FFmpeg's native VP9 decoder as an external executable only**.
It is not permissively licensed source to copy into nanocodec; repository policy
already excludes FFmpeg source reuse. Explicitly select the native decoder,
not an FFmpeg wrapper around libvpx. libvpx plus FFmpeg-native supplies the two
independent reconstruction checks. Reference/comparison use does not imply
linking either library into the shipped Rust/WASM implementation.

Keep SVT-AV1 M12 RTC, OpenH264 and current nanocodec AV1/HEVC as existing comparison
points. This plan does not depend on an unverified “SVT-AV2” implementation.
AVM is a research encoder: its runtime is a comparison result, not the production
speed target for our AV2 backend.

## Initial emitted subsets

The specifications, not reference-encoder flags, define legality. Record each
selected tool's syntax location, disabled value/inference, required decoder
behavior and test vector in a per-codec subset document before emitting it.

| Property | VP9 first useful release | AV2 first useful release |
|---|---|---|
| Pixels | Profile 0, 8-bit 4:2:0 | Final v1.0.0, 8-bit 4:2:0, compatible profile/level explicitly established |
| Latency | No lookahead or future-frame dependency; one displayed output per input | Same |
| Pictures | Keyframes and forward inter frames; one actively searched prior reference initially | Same intent, with final AV2 reference-slot/context semantics |
| Partition search | Constrained square partitions fitting the shared planner; add smaller/rectangular candidates only when justified | Small legal set fitting the planner; verify edge and inferred-partition rules before fixing sizes |
| Prediction | DC plus a small useful intra set; translational single-reference inter; start integer motion, then codec-specific subpixel | Same progression; no assumption that AV1 interpolation or intra rules carry over |
| Residuals | Small initial transform set with actual quantized AC/DC coefficients, then expand sizes | One legal transform path first, then sizes/types justified by data |
| Filtering | Required reconstruction first; tune legal filter strengths/options after bit-exactness | Audit all mandatory/inferred operations; disable optional tools only through legal signalling |
| Tiling and layers | One tile, one temporal layer initially | Same, subject to final syntax constraints |
| Quality controls | Constant quantizer first, then shared CBR/VBR interface and force-keyframe | Same |

VP9 Profile 0's format is specified by the [official profile table](https://www.webmproject.org/vp9/profiles/).
AV2 v1.0.0 Table 6.3 supports 8-bit through `bit_depth_idc=1`, so no mandatory
10-bit workspace migration is needed for this first release.
[Final AV2 specification](https://av2.aomedia.org/v1.0.0/index.html).

Initially omit optional compound/warped prediction searches, screen-content tools,
film-grain synthesis, resizing, high bit depth and multi-tile encoding. For AV2,
confirm every proposed omission against the final syntax; this list is a scope
proposal, not a claim that every omitted operation is independently switchable.
Similarly, disabling segmentation or per-block quantizer changes initially must
be signalled consistently with the quantization actually used.

Temporal layers, advanced resilience/LTR, richer AQ and expanded tool searches
follow the first useful release. The initial decoder rejects unsupported streams
cleanly; it is not advertised as a general-purpose VP9/AV2 decoder.

## Milestones and acceptance evidence

### M0 — Reproducible anchors and a working oracle pair

Extend [`tools/anchors`](../../tools/anchors/README.md), version pins and the
existing evaluator rather than creating a second benchmark runner.

- Add VP9 and AV2 identities to `eval/nceval/codecs.py`, reference adapters,
  stream packaging, conformance dispatch and result fingerprints.
- Record exact versions, flags, thread counts, pixel format and decoder family.
  VP9 uses VP9-labelled IVF initially. Establish AV2 framing/container identifiers
  supported by the pinned tools; never label AV2 as AV1 merely to reuse a reader.
- Generate a small reusable corpus: constant frame, ramp, checkerboard, noise,
  translated patch, one textured camera clip, and non-block-aligned frame sizes.
  Include key/inter transitions and reference resets. Compare visible Y/U/V planes,
  dimensions, frame count and ordering byte-for-byte across each oracle pair.
- Check final AV2 compatibility with both AVM and the selected dav2d revision.
  If they disagree, minimize the stream and distinguish revision mismatch from
  a decoder bug. Document and resolve the affected subset; never silently weaken
  the equality gate. AVM-only development can proceed as experimental, but does
  not pass the two-oracle shipping gate.

**Done:** reproducible anchor manifest, small retained compressed corpus, decoded
hashes, and a passing oracle-pair report. No large raw-YUV corpus or full benchmark
sweep is needed for this milestone.

### M1 — VP9 keyframe path and own subset decoder

Create `nc-vp9` following the existing dependency direction. Implement headers,
VP9 boolean entropy coding/decoding, probability state, coefficient tokens,
selected intra prediction, inverse transforms, quantization and filtering. Reuse
bit readers/writers only where their semantics fit.

Develop decoder parsing alongside encoding. Start with tiny diagnostic streams,
then textured residual-bearing frames; an all-zero or flat-image bitstream is
only a smoke test. Extend CLI codec selection and decode dispatch.

**Done:** own decoder and both external decoders reproduce encoder reconstruction
exactly for the emitted intra subset, including edges, quantizer extremes and
truncated/invalid inputs. Unsupported syntax returns an error rather than a panic.

### M2 — VP9 useful low-delay inter coding

Connect `CodecBackend` to shared motion/partition decisions and frame planning.
Implement reference refresh, motion-vector syntax, residual/skip decisions and
frame probability-state evolution. Add subpixel prediction and useful transform
sizes; carry identical reference pictures through encoder and decoder.

Connect the existing rate controller through a VP9-specific quantizer mapping and
lambda/rate model. Do not assume H.264 QP or AV1 qindex calibration is suitable.
Expose fixed-quantizer and target-bitrate modes, keyframe requests and GOP length.

**Done:** multi-frame clips pass all reconstruction checks; keyframe resets,
scene cuts and long GOPs work; target bitrate, actual bitrate, overshoot and
per-frame CPU are reported. The implementation is useful before exhaustive tuning.

### M3 — AV2 legal subset, keyframes and own decoder

Start the subset audit during M0/M1. Create `nc-av2` after the header/reference
contract is understood. Reuse AV1 code only after checking the corresponding
final AV2 process, tables, contexts, rounding and clipping. Keep AV1 and AV2
probability/entropy implementations separate until an identical primitive is
proved worth sharing.

Implement final sequence/frame signalling, entropy state, selected intra and
residual paths, normative reconstruction and the matching parser. Use AVM for
syntax tracing and dav2d/rav2d for a second implementation perspective. A
block-by-block diagnostic should identify the first differing prediction,
coefficient, reconstructed block or filtered row.

**Done:** the same intra gate as M1, against final-compatible AVM and the selected
independent decoder family. Every emitted tool has an entry in the subset map.

### M4 — AV2 useful low-delay inter coding

Connect the existing analysis/planner to AV2 reference slots, motion candidates,
inter syntax and context persistence. Start with a small legal search space and
expand on evidence. Implement the necessary codec-specific interpolation and
reconstruction before borrowing the encoder's heuristics.

Integrate AV2 quantizer/lambda calibration and rate control independently of AV1.
Investigate a few promising AV2 tools using per-tool CPU and rate/distortion
statistics. Expose named tunable thresholds; permit empirical pruning and joint
threshold searches. The goal is a fast AV2 implementation, not AVM's search
strategy re-created in Rust.

**Done:** the M2 inter/CBR gate for AV2, plus a first measured tradeoff against
nanocodec AV1/HEVC and SVT M12. Report missing tools and known performance limits.

### M5 — WASM, SIMD and initial release gates, separately for each codec

This milestone can run for VP9 while AV2 inter coding develops.

- Add encoder/decoder variants and codec capabilities to `nc-web`; expose both
  choices in the side-by-side webcam/local-video demo, including bitrate and
  quantizer controls, pause/resume, keyframe requests and 30-second GOP testing.
- Keep the own decoder as the baseline. Optional WebCodecs use requires correct
  identifiers and runtime capability checks, not an assumed browser support list.
- Profile first, then implement dominant kernels in `nc-dsp` for AVX2, NEON and
  WASM SIMD128, with scalar references and bit-exact kernel checks. No blanket
  port of every reference assembly routine.
- Run x86/ARM/WASM correctness checks, decoder fuzzing and resource-limit tests.
  Check per-instance isolation and memory growth over sustained playback.
- Run held-out quality, single-core encode/decode speed and browser tests; publish
  the supported subset and benchmark configuration beside results.

**Done:** each codec independently passes its external/own-decoder checks and
works end-to-end in the demo. Call performance experimental if it still misses
the target; distinguish “implemented,” “conformant subset” and “competitive.”

## Comparison protocol

For **VP9**, use libvpx realtime, one-pass, single-thread operation, zero lag,
no future alternate-reference generation, matched keyframe interval and matched
layering. Probe a small set of realtime speed settings, then retain the relevant
speed/quality frontier. Verify controls against the pinned build's help/API;
do not publish an untested command line as a working configuration.

For **AV2**, use pinned AVM in a verified forward-only, no-lookahead configuration.
Keep a fuller-tool quality comparison separate from a restricted-subset experiment.
If a reference flag cannot express our exact subset, record that difference rather
than claiming an exact match. Do not infer intrinsic codec speed from AVM runtime.

Reuse existing corpus splits and source-bitrate caps. Screen a few representative
clips at several rates with PSNR/MS-SSIM; rescore retained finalist streams using
the current pinned all-frame VMAF protocol. Confirm on held-out motion, foliage,
noise/low-light and quiet content. Report PSNR and VMAF BD-rate only over overlapping
quality ranges, with per-clip regressions as well as aggregates. LPIPS can remain
an occasional analytics check, not a prerequisite for every experiment.

Time encoding independently from quality scoring. Use resident inputs, repeated
short runs, one pinned CPU core and quiet machines; measure x86 and Jetson
separately. Include keyframe cost, steady inter-frame cost and tail frame latency.
Measure decoder time and WASM render/upload overhead separately. Publish bitrate,
CPU and memory together, including CBR transient behavior; fixed-QP quality alone
does not establish a useful realtime codec.

Reuse cached compressed outputs and existing metric fingerprints. Bound workers,
keep only useful final streams/metrics, and delete decoded scratch after checking.
No new permanent copies of source videos are needed.

## First implementation batch and handoff

The next coding batch is **M0 plus VP9 M1**, with AV2 subset mapping proceeding
in parallel when capacity permits. Its reviewable outputs are:

1. Pinned reference/license manifest and evaluator adapters.
2. Tiny independently checked VP9/AV2 reference vectors and compatibility report.
3. `nc-vp9` keyframe encoder/decoder integrated into the CLI.
4. AV2 syntax/tool subset map with unresolved compatibility questions identified.

After that, advance VP9 M2 and AV2 M3 independently; share only the necessary
engine/interface changes. Keep per-codec handoffs with emitted features, failing
vectors, measured costs and the next concrete milestone. This plan authorizes no
claim of dates or performance gains before those outputs exist.
