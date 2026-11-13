# Should nanocodec add VP9, H.266/VVC or AV2?

Assessment dated **8 November 2026**, against repository commit `7018233`.
This is an architecture and product assessment, not a new benchmark. Relative
implementation effort and priorities below are engineering judgments; no speed
or compression advantage has been measured for these additions.

## Recommendation

**AV2 is the most interesting next compression project; VP9 is the most
straightforward additional implementation; VVC should be driven by a concrete
performance result or deployment requirement.** Investigate the candidates in
the existing evaluation framework before committing to another production codec.
Do not start three implementations together.

| Candidate | Value to nanocodec | Relative implementation effort | Recommended action |
|---|---|---|---|
| VP9 | An additional established delivery format; potentially a useful speed/compression compromise | Lowest of these three, still a new entropy coder, reconstruction path and decoder | Add reference measurements now if considering it; implement when a receiver requirement or measured advantage justifies it |
| H.266 / VVC | Potentially better compression in deployments that accept VVC | High; shared HEVC concepts help, but the backend is substantially different | Run a bounded low-delay feasibility comparison; prioritize implementation only with a strong result or customer need |
| AV2 | The strongest strategic extension of the existing AV1 work | High, with more uncertainty about the best extremely fast subset | First choice for a focused new-codec compression prototype, after a reference/subset investigation |

This ordering is about **our live-camera, single-core, low-delay workload**.
It is not a ranking of the formats' maximum compression efficiency. If the
requirement is “deliver a fourth format quickly,” VP9 moves first. If it is
“improve compression without surrendering speed,” AV2 deserves the first spike.
Existing AV1 and HEVC optimization remains valuable alongside that bounded spike;
there is no need to demand perfect parity before investigating a successor.

## What the current repository already provides

The current implementation is materially beyond the original design sketch:
there are three native Rust encoders and matching subset decoders, a shared
rate-control path, a browser/WASM integration, and evaluation/conformance
infrastructure. Recent commits also add hierarchical-P signalling and reference
handling across H.264, HEVC and AV1. A plan based on an intra-only prototype or
an assumed absence of rate control would overestimate the missing common work.

The relevant boundaries are:

| Existing component | Reuse and limitation |
|---|---|
| [`nc-encoder` backend contract](../../crates/nc-encoder/src/backend.rs) | Frame planning, references, rate-control integration and mode-decision plumbing are reusable. Actual reference syntax, probability state and legal coding decisions belong to each codec. |
| [`nc-analysis`](../../crates/nc-analysis/src/lib.rs) | Motion search and source analysis are reusable starting points. Its documented search organization is 64×64 superblocks and square candidates from 8×8 to 64×64. Richer partition structures need deliberate extensions. |
| [`Plane`](../../crates/nc-common/src/plane.rs) and [`nc-dsp`](../../crates/nc-dsp/src/lib.rs) | Existing planes are explicitly 8-bit. Dispatch, scalar/SIMD testing and some pixel operations transfer; codec-specific interpolation, transforms, rounding and filtering require their own correctness work. |
| [`Codec` configuration](../../crates/nc-encoder/src/config.rs) | The public choices are explicit, not dynamically registered. New codecs need configuration and capability handling. |
| [CLI encoder](../../crates/nc-cli/src/main.rs), [decoder](../../crates/nc-cli/src/decode.rs), [WASM API](../../crates/nc-web/src/lib.rs) | Each explicitly selects existing codec implementations. Adding a crate does not automatically add CLI, browser, packet packaging or demo support. |
| [Evaluation](../../eval/nceval/codecs.py) and procedures | Reuse curve fingerprints, conformance gates, clip sets and metrics. Add codec adapters and independently implemented external decoder checks. |

The current square-quadtree backend interface is a good place to start a
restricted encoder. It is not proof that every useful VVC or AV2 partition fits
without changes. Conversely, supporting a larger format does not require
searching every legal partition. Keep the first implementation constrained and
extend the planner only when a measured gain warrants it.

The existing AV1 backend is useful experience for VP9/AV2, and HEVC for VVC.
None can be converted by changing headers. Entropy contexts, coefficient coding,
prediction rules, reference state, clipping and integer rounding are normative.
Reuse only kernels whose exact behavior has been checked against the new format.

## VP9: sensible for compatibility, less compelling as a successor

VP9 is an established format with libvpx as its reference implementation.
**Profile 0 is 8-bit 4:2:0**, matching nanocodec's current pixel representation.
This makes it the least disruptive candidate architecturally.
Sources: [WebM VP9 overview](https://www.webmproject.org/vp9/),
[official profile table](https://www.webmproject.org/vp9/profiles/).

A sensible initial emitted subset would be Profile 0, one tile, keyframes and
forward-predicted inter frames, a small reference set, and a constrained partition
and transform search. Avoid hidden future-reference frames in the low-delay
configuration. Establish normative reconstruction and loop filtering before
optimizing the decision process. Add temporal layers after ordinary inter-frame
reference handling is proven.

The main new work is the VP9 bitstream/probability machinery, prediction and
coefficient syntax, codec-specific reconstruction kernels, frame-context and
reference updates, and the corresponding parser. AV1's multi-symbol entropy
implementation is not a drop-in VP9 entropy coder. The format specification is
linked from the [official VP9 documentation](https://www.webmproject.org/vp9/).

**The reason to add VP9 should be positive and specific:** an important receiver
supports VP9 well, its hardware decode saves power, or measurements show a useful
CPU/quality position between our existing choices. An older standard can win
under a severe CPU budget; that is a testable possibility, not something to
assume. Equally, merely having fewer coding tools does not establish that our
VP9 implementation would beat our AV1 or HEVC paths.

Use libvpx's realtime configuration as the VP9 speed/quality anchor, with pinned
settings and a small speed-preset sweep. Validate emitted streams with libvpx and
an independently implemented decoder such as FFmpeg's native VP9 decoder. An
FFmpeg invocation that wraps libvpx is not a second independent decoder.

## H.266 / VVC: technically feasible, commercially conditional

H.266 is the standardized name for VVC. It is not a speculative format: the
current ITU recommendation is **H.266 V4, January 2026**.
[ITU recommendation](https://www.itu.int/ITU-T/recommendations/rec.aspx?rec=16662).

A constrained native Rust VVC encoder is feasible. The difficult question is
whether a fast subset retains enough of its compression benefit to repay the
implementation and decoder cost. Start with a simple legal configuration:
forward prediction, limited reference search, constrained partitions, and a
small transform/mode set. Audit each omitted tool's signalling and inference
rules; omitting encoder search does not necessarily eliminate decoder behavior.

HEVC provides useful design experience, but VVC brings different partition,
entropy, prediction and reconstruction rules. The fuller toolset, including
multi-type-tree partitioning and additional prediction/filtering tools, makes
“HEVC with new headers” the wrong implementation model. Use the normative
[H.266 specification](https://www.itu.int/rec/T-REC-H.266) to define an explicit
subset before changing the shared engine.

Use [VVenC](https://github.com/fraunhoferhhi/vvenc) for a practical reference
comparison and VTM for normative investigation. Audit whether the selected
encoder configuration actually meets our no-reordering/no-lookahead constraints;
a default streaming benchmark is not interchangeable with realtime camera
encoding. Failure of one reference configuration to meet our CPU budget does
not prove that a purpose-built fast encoder cannot succeed. It does tell us
which tool costs and quality gains need explaining before a larger investment.

Our own WASM decoder makes limited native-browser adoption an inconvenience,
not an absolute blocker. [VVdeC](https://github.com/fraunhoferhhi/vvdec) already
includes WASM build support, demonstrating feasibility. It does not establish
that our eventual decoder will meet browser CPU, memory or power targets.
Use VVdeC and VTM as distinct reconstruction oracles for the chosen subset.

Licensing needs a deliberate product decision. VVenC's
[Clear BSD license](https://raw.githubusercontent.com/fraunhoferhhi/vvenc/master/LICENSE.txt)
permits source reuse subject to its conditions but explicitly provides no patent
license. A permissive software license alone does not settle VVC distribution
rights. This assessment does not estimate royalties or claim patent clearance.

**Recommendation:** retain VVC as a serious candidate if bandwidth savings or a
receiver requirement can justify it. For a general-purpose browser-facing
nanocodec extension without a named VVC customer, AV2 currently has a clearer
strategic rationale.

## AV2: worth investigating now, with the final specification

The official AV2 release is **v1.0.0, dated 28 May 2026**, with a matching AVM
reference tag. The January 2026 “v13” is a superseded working draft, not a newer
release. Pin the final specification and compatible reference software for the
prototype. [Official specification and release links](https://av2.aomedia.org/).

An important architectural detail: **final AV2 supports 8-bit as well as 10-bit**.
Table 6.3 maps `bit_depth_idc=1` to 8-bit and `0` to 10-bit. An 8-bit 4:2:0
prototype therefore need not begin with a workspace-wide high-bit-depth rewrite.
AV2 still introduces new syntax and reconstruction processes, so this is a
reduction in initial scope, not binary compatibility with AV1.
[Final specification, §6 and Table 6.3](https://av2.aomedia.org/v1.0.0/index.html).

AV2's strategic attraction is the opportunity to choose a small, unusually fast
subset of a newer coding toolbox using what we have learned building AV1. Its
risk is that a stripped-down implementation may discard the very tools that
produce the advertised gains. Treat the benefit of each candidate tool as a
joint speed/quality decision, including interactions with partition search,
quantization, filtering and early-exit thresholds.

AOMedia's October announcement reports approximately 30% greater compression
efficiency than AV1 and describes early browser playback demonstrations and
industry evaluations. That is ecosystem evidence, **not a prediction of 30%
savings for nanocodec at SVT M12 CPU cost**, nor proof of universal native browser
or hardware availability.
[October 2026 ecosystem announcement](https://aomedia.org/press%20releases/AOMedia-Rapid-AV2-Adoption-Across-the-Video-Ecosystem/).

Start by mapping a legal 8-bit, one-tile, forward-predicted subset and its
mandatory reconstruction operations. Then identify a few tools that might earn
their cost on our camera corpus. Do not implement the entire tool inventory or
blindly port AVM's encoder search. A slow reference encoder is useful as an
oracle and experimental quality bound, not as the required production design.

Use final-compatible AVM for initial conformance. Before calling the codec
production-ready, establish the project's normal second independent decoder
check with a verified compatible implementation. Two builds of AVM, or AVM
through two wrappers, do not supply that independence. Availability and maturity
of that second implementation is an explicit investigation item.

AVM's [software license](https://raw.githubusercontent.com/AOMediaCodec/avm/v1.0.0/LICENSE)
and [AOM patent license](https://raw.githubusercontent.com/AOMediaCodec/avm/v1.0.0/PATENTS)
are separate documents. AOM's royalty-free licensing framework is attractive,
but its conditions still apply and it is not a guarantee against all third-party
claims. Likewise VP9's [patent grant](https://raw.githubusercontent.com/webmproject/libvpx/main/PATENTS)
has a defined scope; do not describe either format as simply “patent-free.”

## What “support” should mean

A reference executable in the benchmark harness is useful research support.
Shipping support means a native encoder, matching subset decoder, CLI and WASM
integration, rate-control behavior, and a conformance-tested emitted subset.
Document unsupported input streams explicitly: our decoder need not be a general
replacement for libvpx, VVdeC or a full AV2 decoder.

The encoder already needs reconstruction, so substantial decoder kernels can be
shared. A decoder additionally needs entropy parsing, safe input validation,
reference/context lifetime handling, reset/seek behavior, and malformed-stream
tests. That work is smaller than a full general-purpose decoder, but cannot be
replaced by exposing the encoder's reconstruction buffers.

Retain the current separation of codec modules and shared analysis. Add
`nc-vp9`, `nc-vvc` or `nc-av2` only when beginning its implementation. Keep unsafe
SIMD in `nc-dsp`, with scalar references and checks on x86, ARM and WASM. Do not
preemptively generalize every buffer or partition API for all three candidates.
A future 10-bit path should preserve an efficient 8-bit path and be justified
by its intended content and profile.

For browser delivery, include packet boundaries, timestamps, keyframe/reset
handling, color metadata, supported codec identification and WASM module size in
the design. Container and WebCodecs identifiers must be format-specific; reusing
AV1's identifier for AV2 is invalid. Native decoder capability should be tested
at runtime. The own-decoder path remains the baseline.

## Efficient investigation and decision gates

1. **Add measurement adapters before codec code.** Pin libvpx, VVenC and
   final-compatible AVM versions and configurations. Reuse existing source clips,
   cached streams and metric infrastructure. First run a small, diverse screen
   with matched low-delay settings; do not launch an expensive full-corpus AVM
   sweep just to discover an unsuitable configuration.
2. **Measure the relevant tradeoff.** Compare against current nanocodec AV1/HEVC
   and SVT M12 RTC, keeping OpenH264 as the H.264 speed baseline. Use matched
   dimensions, frame rate, GOP, temporal structure, bit depth, color handling and
   actual bitrate ranges. Report encoder CPU, decoder CPU, latency and memory
   separately. A codec that saves bandwidth but breaks receiver realtime has
   not won the end-to-end comparison.
3. **Use broad content efficiently.** Screen with the existing PSNR/MS-SSIM
   path; score retained finalists with the pinned all-frame VMAF protocol.
   Include motion, low light/noise, foliage, crowds and quiet scenes from the
   existing expanded video/tracking sets. Hold out content for confirmation.
   Observe source-bitrate caps and report excluded or non-overlapping curves;
   do not extrapolate BD-rate outside common quality coverage.
4. **Separate speed from metric cost.** Time repeated short encodes independently
   of VMAF, on a quiet host and on Jetson, using one core and resident input.
   Identify any I/O or conversion cost separately. Test representative easy and
   difficult clips, multiple rates, keyframes and long-GOP inter frames. Browser
   decode measurements must include upload/render overhead separately from the
   decoder itself. Retain metrics and compressed streams, then delete raw
   reconstructed scratch data.
5. **Choose one bounded implementation spike.** Write a legal-subset/tool matrix;
   produce externally verified keyframes, then inter frames and an own decoder.
   Optimize dominant kernels and decision costs before drawing conclusions from
   an unoptimized scalar prototype. Make heuristic constants tunable and search
   important interacting thresholds together. No requirement for exact/provable
   pruning should prevent a measured speed/quality tradeoff.
6. **Promote only on a useful result.** A deployment requirement can justify a
   codec without a compression win. Otherwise require a material improvement
   on the speed/quality frontier across held-out clips, with acceptable decoder
   cost. As an initial *proposed* target, investigate roughly 10% bitrate savings
   at matched quality and comparable CPU, or a similarly meaningful CPU saving
   within an agreed quality budget. These are decision criteria, not forecast
   results. Confirm CBR overshoot, scene cuts, keyframe bursts and temporal
   extraction before shipping.

The first deliverable should be a compact reference comparison and legal-subset
map, followed by **one** prototype if the evidence supports it. AV2 is my preferred
prototype for the compression goal. VP9 is the pragmatic alternative when
compatibility or a demonstrated speed advantage is the goal. VVC remains viable,
but needs the clearest justification for taking effort away from the existing
three codecs.
