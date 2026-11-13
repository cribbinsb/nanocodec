# AV2 native basic subset

Status: experimental syntax implementation, 2026-11-08. Reference is AVM
v1.0.0, commit `966a7d7cd6fcf60360caf5dc413b2aeeb65e144d`, under
BSD-3-Clause-Clear plus the separate AOM patent terms in `docs/licenses/avm`.
Native arbitrary-pixel intra and restricted forward pictures pass independent
AVM and dav2d pixel gates; broad efficiency tuning and product integration remain.

## Sequence contract

`nc_av2::sequence::{write, parse}` implements complete sequence-header payloads,
including bounded field parsing and exact trailing-bit checks. The API supports
single-picture sequences and forward-video sequences; it does not silently turn
video into independent still pictures. Profile is Main 4:2:0 IP0 (profile 0),
8-bit planar 4:2:0 (`bit_depth_idc=1`, `chroma_format_idc=0`). AVM's
`av2_check_profile_interop_conformance` confirms this profile permits both eight
and ten bits despite “10” in its C enum name.

| Group | Emitted syntax and consequences |
|---|---|
| Identification | Sequence ID 0–15, level 0–21 or 31, main tier. Level field validity is checked; picture bitrate/dimensions still require level conformance checks. |
| Dimensions | Maximum coded dimensions 1–65536, coded as whole 8x8 blocks; a right/bottom conformance window (`av2_write_conformance_window`) crops any even size back. Frame implementation imposes smaller resource/tile limits. |
| Ordering | One to three temporal layers (`max_tlayer_id`, default dependency map: each layer references itself and lower layers; tile-group OBU `temporal_id`), one embedded layer, monotonic output, no operating-point set (lower layers are reached by dropping OBUs). Video order hints 1–16 bits; no display/decoder timing model or explicit LCR. |
| Partitions | 64×64 superblocks, shared luma/chroma tree; no extended/uneven partitions; maximum prediction-block aspect ratio 2. Binary partition syntax to uniform 8×8 leaves is implemented. |
| Intra | DIP, edge filter, multi-reference lines, CfL, MHCCP and IBP disabled. The ordinary baseline intra modes and their syntax remain necessary. CfL downsample filter index 0 is still signalled. |
| Inter | Simple translation is always enabled by the specification. No inter-intra, warps, masked compound, reference-MV fields/bank, DRL reordering, TIP, trajectories, BAWP, CWP, optical-flow/MV refinement, adaptive/sign-derived MVD, flexible MV resolution or global motion. |
| References | Video uses eight buffer slots, no long-term frame IDs, DRL maximum one and no per-frame override. Flat sequences derive their single reference implicitly (every P refreshes all slots); layered sequences enable the explicit reference map and name one slot per frame (`num_total_refs = 1`), with the engine's refresh mask. Single pictures infer two slots and minimum DRL. No short refresh syntax. |
| Screen content | Video forces screen tools off. **Single pictures infer selectable screen tools**, so a frame must still signal `allow_screen_content_tools=0`; intra-block-copy permission is separately signalled in frame syntax. |
| Transforms | FSC, intra identity extension, secondary intra/inter transforms, inter DDT, cross-component transforms off; TCQ (`enable_tcq` with per-frame `tcq_mode`) and parity hiding optional (`av2.tcq`, `av2.parity_hiding`; both off by default). Reduced transform partitions on; chroma DCT restriction on. This does not remove required transform selection syntax. |
| Quantization | Equal AC/DC steps; zero base UV AC delta (coded as 23 relative to −23); `uv_ac_delta_q_enabled = 1` in engine sequences, so each frame codes one chroma delta (`read_delta_q`, −64..63, shared by U/V and DC; zero on lossless frames); no separate U/V deltas, no luma DC delta. |
| Entropy | Video disables sequence average-CDF; **single pictures infer average-CDF enabled with type 1**. Frame flags support fixed or trained tile-local updates; the shared encoder and stateful stream now default to trained adaptation. AV2's native baseline entropy coder is implemented; its CDF tables are not AV1 aliases. |
| Filters | CDEF on (frames choose `cdef_frame_enable`; one strength set, so no block-level CDEF index); GDF and restoration off; CCSO optional (`av2.ccso`, off by default; 256×256 units, no filter reuse). Video disables CDEF-on-skipped-transform; single pictures infer adaptive and code it off per frame. Deblocking parameter width is two bits. Actual deblocking permission remains a frame-level choice. |
| Other | No sequence tile config, segmentation data, film grain or sequence extension. One tile is selected by frame syntax. |

Syntax order was audited against `av2_write_sequence_header_obu`, the seven
`write_sequence_*_group_tool_flags` functions and their decoder counterparts.
Unsupported tool bits are rejected with named errors rather than skipped under
an incorrect inferred layout. Parsing does not allocate picture buffers.

## Initial keyframe prefix

`nc_av2::frame_header::{write_keyframe, parse_keyframe}` implements the
byte-aligned header prefix for one `OBU_CLOSED_LOOP_KEY` (type 4) tile group:

- First tile group, no multi-frame header, explicit sequence ID.
- Immediate output in video, no frame-size override; dimensions equal sequence
  maximum. Closed-loop keys infer refresh of every slot for the single layer.
- Screen tools and intra block copy off; primitive fixed-model vectors disable adaptation, while shared encoding enables it by default.
- Uniform one-tile layout, at most 4096 pixels wide and 2304 64×64 superblocks
  in area, conservatively satisfying minimum tile limits at every level.
- Eight-bit qindex; no segmentation, quantizer matrices or delta-Q.
- Qindex zero infers lossless 4×4 transforms and omits filter/transform-size bits.
  Lossy coding turns deblocking off and signals selected or largest transform sizes.
- Reduced transform set 2 (intra DCT-only; inter DCT/identity); zero byte-alignment bits.

This prefix **must be followed by a valid entropy-coded tile**. Returning its
length is useful for a tile decoder but is not evidence of picture conformance.
Restricted forward headers and reference state are implemented as described
below; both intra and initial zero-motion inter picture paths have pixel gates.


## Entropy implementation

`nc_av2::entropy` now supplies fixed/adaptive multi-symbol coding for 2–16 symbols,
true bypass literals, and a bounded independent arithmetic reader. CDFs retain
AV2's three trained adaptation-rate offsets and the count transitions at 16/32;
they are not passed through AV1 adaptation. Probability scaling uses AV2's
seven-bit precision reduction, symbol increment and range rounding to multiples
of eight. A 25,099-byte oracle stream of 20,000 adaptive symbols mixed with
0–16 bypass bits matches the original AVM writer exactly. AVM's scalar decoder
and the native reader both recover every event. Additional unit tests cover all
alphabet sizes and full 32-bit bypass values. The decoder permits at most 15
implicit termination zero bits before reporting truncation.

`examples/gray_keyframe.rs` emits a restricted native 64×64 DC-predicted frame.
The original qindex-100, all-zero-residual stream decoded successfully with
independent dav2d commit `237229ebfefd4eb8e0e25ad9636803bb93bae0c6`: one frame,
6144 bytes all equal to 128, SHA256
`cb11e05cb5da949c0e0f5b5a7cb310df35a96a22c45d1ada70d950859fe697d1`.
The scalar eight-bit dav2d binary SHA256 is
`5fd2c3a38ae7073328b745198b5c62ca8c4010bb0084deaf08ed619a7a9a5128`.
This validates the final sequence/frame/tile syntax for this narrow case;
it does not establish general decoder compatibility or a complete encoder.

The probe is now extended to qindex 255 with optional signed luma DC levels
−4 through 4. `nc_av2::dc_tile` writes/reads the base magnitude, EOB-one and
DC-sign syntax and reconstructs this restricted 64×64 tile natively.
A 94-byte nine-frame sequence passes the full two-family framework gate against
AVM and dav2d, matching all expected reconstruction hashes. Luma values are
68, 83, 98, 113, 128, 143, 158, 173 and 188; chroma remains 128.

Native arbitrary-pixel keyframes now pass the full two-family picture gate:
48 frames covering 8×8, 16×16, 64×64 and partial-superblock 72×40 geometry,
qindices 0, 1, 90, 140, 190 and 255, and flat/textured Y/U/V inputs.
AVM and independently implemented dav2d match the native decoder and encoder
reconstruction for every frame; qindex zero also matches input exactly.
Each geometry is one 12-frame stream formed by concatenating self-contained
keyframe packets, validating repeated sequence headers and frame output order.
Raw pixels remain in memory; only compressed streams are temporary files.

`nc_av2::intra::encode_keyframe(&Picture, qindex)` borrows the shared planar
picture directly and returns `EncodedFrame { packet, reconstruction: Picture }`.
`decode_keyframe` returns a shared `Picture`. The implementation uses 8×8 DC/H/V/Paeth
prediction blocks, four raster 4×4 luma transforms and one 4×4 transform for
each chroma plane. Qindex zero uses WHT; lossy frames use native AV2 DCT and
explicit transform partitioning. Fixed frame CDFs, full AC coefficient coding,
base/range magnitude extensions, adaptive Rice high ranges and signs are present.
Reconstruction is performed once per transform and immediately reused for
prediction. Scratch context arrays live per picture, not per block.

Reproduce the complete streamed pixel gate with:

```sh
NC_ANCHORS=~/nanocodec-data/anchors/vp9-av2 \
  python3 crates/nc-av2/examples/check_intra.py \
  --encoder "$CARGO_TARGET_DIR/debug/examples/intra_frame"
```

The initial forward subset also passes a full two-family 100-frame gate:
8×8, 16×16, 64×64 and 72×40, qindices 0/1/100/200/255, and five-picture
sequences containing one keyframe, unchanged P pictures and changed P pictures.
Every decoded frame matches native reconstruction; lossless frames match input.
The implementation uses one distinct previous reference, zero-motion GLOBALMV,
block skip when every quantized residual vanishes, and the same full residual
path as intra. Every P refreshes all eight reference slots, preserving the single
reference invariant. Cross-frame CDF prediction remains disabled; tile-local adaptation is optional.

`inter::encode_inter(source, reference, qindex, order_hint)` accepts borrowed
shared pictures. `stream::Decoder` retains sequence/reference state transactionally;
`decode_ref` returns a borrowed reconstruction without an extra picture copy,
while `decode` provides a compatible owned result. A failed packet leaves the
previous reference intact. The corresponding stateful encoder resets on requested
keys or dimension changes. Raw OBU packets concatenate into valid sequences.

Reproduce the forward pixel gate with:

```sh
NC_ANCHORS=~/nanocodec-data/anchors/vp9-av2 \
  python3 crates/nc-av2/examples/check_forward.py \
  --encoder "$CARGO_TARGET_DIR/debug/examples/forward_frames"
```

Forward frame prefixes match the original AVM writer in eight independent C
oracle vectors. The quarter-pel MVD shell/pair/sign coder matches a 73-byte
original AVM golden. The first two spatial reference candidates and mode contexts
match original AVM across 20 traces, including superblock boundaries.

`stream::Encoder::with_motion()` enables bounded NEWMV search and quarter-pel
luma/eighth-pel chroma interpolation. Both coarse integer and refined subpixel
versions passed full 125-frame two-family gates, adding 128×136 geometry to the
five-quantizer forward matrix. Append `--motion` to the command above to test it.
The bilinear predictor reuses the existing shared H.264 chroma SIMD kernel;
25,992 original-AVM convolution fixtures prove its exact rounding and edges.
Search exposes tunable early-SSE, diamond-pass and subpixel-step parameters.
Motion stays opt-in: the simple luma-SSE proxy increased compressed bytes on the
synthetic stress sequence, so these results prove correctness, not coding gains.

Four-mode intra selection also passes the full 48-frame external gate. Mode
syntax matches 800 original-C cases and prediction matches 640 original-C
fixtures. AV1 and AV2 now reuse the same DC/H/V/Paeth prediction implementation
in `nc-dsp`; AV2 smooth prediction is deliberately excluded because its formula
differs. The encoder compares inexpensive predictor SSE before performing each
selected transform once. Intra transform set 2 infers DCT independent of mode;
768 original-C transform-type cases confirm no extra entropy symbol is required.

## Validation and next gate

Rust tests cover both sequence forms, dimension boundaries, truncation, malformed
trailing bits, invalid field ranges, hostile short payloads, quantizer extremes,
and frame/tile prefix boundaries. These are internal structural tests only.

`crates/nc-av2/examples/sequence_oracle.c` fills the original AVM `SequenceHeader`
and calls **AVM's actual `av2_write_sequence_header_obu`**; it contains no copied
codec function and no stand-in writer. `sequence_vectors.rs` emits the matching
native fixtures. Compile the C driver against the pinned build's generated
headers and library and compare all output bytes before claiming external
sequence conformance. The C driver is a test executable, never a runtime or
WASM dependency. **Eight complete sequence payloads now match the original pinned AVM writer
byte-for-byte:** video/still at 1×1, 64×48, 1920×1080 and 65536×65536. Golden
payloads are retained in `crates/nc-av2/tests/data/avm-sequence-headers.hex`.
The comparison compiles the original `bitstream.c` with linker section garbage
collection plus three original dependencies, using configured AVM headers;
it needs no function stubs or full library build. Reproduce with:

```sh
python3 crates/nc-av2/examples/check_sequence_oracle.py \
  --source ~/nanocodec-data/anchors/vp9-av2/src/avm-v1.0.0 \
  --build ~/nanocodec-data/anchors/vp9-av2/build/avm-v1.0.0 \
  --vectors "$CARGO_TARGET_DIR/debug/examples/sequence_vectors"
```

Next milestones:

1. Integrate the validated native picture APIs with the shared encoder, rate control,
   CLI and WASM; preserve direct plane borrowing and one-pass reconstruction.
2. Tune motion selection with actual rate/distortion evidence before enabling
   moving prediction by default; retain the two-family sequence gate.
3. Expand transform/mode search and optional filters while retaining
   the established pixel conformance gate and bounded evaluation workflow.
4. Benchmark broad real clips against pinned AVM, reporting speed/quality without
   treating this restricted starting point as performance parity.

## Tile-local CDF adaptation (shared default)

`av2.cdf_adaptation=1` selects final-AVM trained updates independently of motion
search and is the shared encoder default. Set it to zero for fixed-CDF comparisons;
low-level fixed-picture helpers retain their explicit fixed-model behavior. Headers signal the policy per frame; each tile starts fresh models at the
frame quantizer category. Cross-frame CDF inheritance remains disabled. Every
reachable partition, intra/inter mode, transform, MV and coefficient model has
explicit identity; equal initial probabilities do not alias unrelated models.
With TX8 support, models cover 49 families / 386 CDFs in 4,542 bytes. Original-C update
traces cover 65 updates per model in all four quantizer categories, and adaptive
motion syntax additionally matches an independent original-C byte golden.

Adaptive RD trials restore probability models together with arithmetic state and
coefficient-edge context. The decoder supports streams switching between fixed
and adaptive policies, including forced keys and quantizer-category changes.
An integrated mixed-policy roundtrip tests reference preservation after failure.

The combined adaptive-CDF/moving-RD 125-frame full AVM+dav2d pixel gate passes.
Compared with fixed-CDF RD, stream totals decrease from 47,886 to 35,574 bytes
at 64×64, and 229,667 to 163,972 at 128×136. The 27-frame adaptive CLI CQP/CBR
and periodic/forced-key test also passes both external decoders and native decode.
These are correctness and bounded synthetic-byte results, not broad BD-rate.

Fixed-CDF 640×360 screening revealed a content-independent all-zero-residual
floor of 945.69 kbps at qindex 255, above all three source-rate caps. Consequently
fixed-CDF q96/160 real-clip results are explicitly out-of-domain diagnostics.
Adaptive in-domain quantizer bracketing and quality evaluation are separate from
that invalid-rate screening; see `av2-motion-rd.md` for the recorded limitations.

The default promotion is supported by bounded instruction counts at q208:
adaptation adds 2.38% P-picture instructions on static and 3.52% on pan while
preserving pixels exactly. These are deterministic callgrind instruction counts,
not wall-clock speed claims. Motion remains off by default: its initial in-domain
screening settings produced identical streams with additional CPU cost. See the
coupled screening ablation in `av2-motion-rd.md` for the next tunable improvement.

## TX8 and residual-skip extension

Lossy 8×8 luma DCT is the default for both key and forward pictures, selected
through `av2.tx8=1` (zero restores TX4); chroma stays 4×4. Transform partition syntax distinguishes
one TX8 from four TX4s. The decoder accepts both on successive pictures, with
quantizer zero always retaining the lossless WHT4 path. TX8 uses final AVM's
83/35 transform constants: HEVC's 83/36 kernel is not interchangeable. Entropy
coding reuses the existing Rice/sign/range machinery with normative TX8 scans,
EOB64 and neighborhood contexts. Prediction continues to use shared DSP kernels.

Original-C transform checks cover all 255 lossy quantizers and clipped-level
extremes. The TX8 48-keyframe and 125-picture I/P suites match AVM and dav2d,
including partial superblocks, adaptive probabilities and quarter-pel motion.
Malformed TX8 packet checks also verify that an error preserves the prior reference.

## Coefficient syntax for every square transform

`coeff::write` / `coeff::read` code any square transform, TX4 to TX64, in all
planes and for intra and inter blocks (`coeff::TxSize`; TX64 codes its top-left
32×32, laid out with stride 32 as in AVM). Contexts come from
`coeff::Context::derive` (AVM `get_txb_ctx`) and each block leaves
`coeff::entropy_context` in the above/left arrays. Luma base and skip models
are per size, chroma base/low-range models are shared, EOB uses the 16/64/256/1024
families with the capped-symbol overflow bits from TX16, inter luma TX4..TX32
signals DCT in the DCT/IDTX set and TX64 is DCT-only. Lossless stays TX4-only;
chroma stops at TX32 (AVM `av2_get_tx_size`).

The V all-zero context is `get_txb_ctx`'s base (+3 when the chroma block is
larger than its transform) plus 6 when the preceding U block had coefficients
(`xd->eob_u_flag`, added in `av2_read_sig_txtype`). With chroma transforms equal
to chroma blocks the +3 never applies, so existing streams are unchanged
(byte-identical packets before and after the generalisation).

`tools/codegen/check_av2_coeff.py` is the original-C oracle: it derives every
context of `examples/coeff_vectors.rs` with AVM `get_txb_ctx` and decodes our
tiles (all sizes, planes, both prediction kinds, every quantizer category,
fixed and adaptive CDFs) with AVM `av2_read_sig_txtype` / `av2_read_coeffs_txb`.

The shared default now prices a no-residual candidate (`av2.residual_skip_rd=1`)
and screens low-energy prediction before transform/entropy trials
(`av2.pre_tx_skip_energy=30`). This is empirical, lossy-only screening, with
cached prediction reused when the screen fails. Setting either parameter to zero
allows controlled ablation. Motion remains independently selectable; its measured
settings are search lambda 0.135, early SSE/lambda 0.025 and one subpixel step.
See [motion measurements](av2-motion-rd.md)
for quality/CPU evidence and the current TX8 default decision.
