# nanocodec architecture

What nanocodec is for, how it is built, and why it is built that way.
`CLAUDE.md` holds the binding rules; `docs/PROCEDURES.md` holds the how-to
(build, measure, land); `docs/STATUS.md` says where things stand. The story of
what was tried and what paid off is [RETROSPECTIVE.md](RETROSPECTIVE.md).

## 1. Goals and constraints

nanocodec encodes live camera video (fixed cameras, bodycams, drones, robots)
on small edge devices and decodes it anywhere, especially in a web page.

- **Real time on one core.** One Cortex-A78AE core (Jetson Orin Nano class)
  must encode the stream; x86 (AVX2) is the second platform. Single-threaded
  today, but thread-ready: no globals, per-row state.
- **Low delay.** Forward prediction only (P frames, encode order = display
  order), no lookahead, no B-frames.
- **Low bitrate, real rate control.** Typical operating point 720p30 at a few
  hundred kbps. CBR within a 1 s buffer; the target is a ceiling (undershoot
  is allowed, padding never is), with a quality cap so easy scenes do not
  spend bits on noise.
- **Resilience at frame level.** The loss model is "a frame arrives whole or
  not at all", so there are no slices: one slice or tile per frame.
  Resilience comes from temporal layers, a long-term reference with receiver
  feedback, intra refresh and keyframe on request.
- **Standards-strict.** Every stream is plain, conformant H.264, HEVC, AV1,
  VP9, AV2 or VVC. No custom syntax, ever: it would lose the reference
  decoders we test against and hardware decode on receivers. Speed comes
  from the encoder's choices within the standard.
- **WASM decode is the primary browser path.** Our own decoders compile to
  WebAssembly (SIMD128), so a page plays our streams with no dependence on
  the browser's codecs. WebCodecs is an optional extra. 8-bit 4:2:0 only.
- **Noise is removed, not reproduced.** Sensor noise is unwanted at these
  rates: a clean picture is the better result. Film-grain synthesis is an
  experiment only (AV1 `grain.enable`, off; `docs/handoff/av1-grain.md`); the
  encoder drops isolated small coefficients, biases static blocks to skip and
  keeps AQ from reading noise as texture. Noise robustness is measured on
  clean clips with synthetic sensor noise, scored against the clean original.

The bars: H.264 must beat openh264 (realtime, medium complexity) at about its
CPU; AV1 must match or beat **SVT-AV1 M12 `--rtc`** on BD-rate and CPU, on
every content class, on both platforms; HEVC should land near it minus the
format gap; VP9 near our HEVC; AV2 at or below our AV1.

## 2. Design principles

- **SVT-AV1 real-time is the reference architecture.** Its real-time
  pipeline (one cheap analysis pass, motion search once, block sizes from
  statistics, a single light mode-decision pass that also writes the
  bitstream) is followed closely, and departures need a measurement. It is
  BSD-3, so its algorithms are reimplemented with attribution. The
  code-level reading is `docs/reference/svt-av1-m12-rtc.md`. The RD-search
  structure of the HEVC reference encoder, x265 or kvazaar is too slow for
  this budget: trimming it stalls at several times openh264's CPU.
- **One engine, many codecs.** Analysis, motion search, partition planning,
  mode decision, rate control, references and temporal structure are written
  once. Each codec adds only its syntax, transforms, entropy coding,
  prediction filters and loop filters. An engine improvement lands in every
  codec, and building the codecs together kept the abstraction from becoming
  H.264-shaped.
- **Every heuristic is a measured, registered parameter.** Thresholds and
  early exits are named parameters with a default, a range and a stated
  trade-off, so presets are points found by a joint Pareto search (§9), not
  hand-set knobs. A lever that is off must cost nothing.
- **Clean structure over clever code.** Speed lives in small kernels with
  scalar references; everything else reads plainly. Rust was chosen because
  all maintenance is done by AI agents: the compiler enforces memory safety
  and Cargo enforces the module boundaries that conventions alone would not
  keep.
- **Borrowed ideas, not borrowed code.** openh264, kvazaar, SVT-AV1, libaom
  and dav1d (BSD-style) are reimplemented in idiomatic Rust with attribution
  in `THIRD_PARTY.md`. GPL/LGPL code (x264, x265, ffmpeg) is never copied.

## 3. Crates and layering

```
layer 5   nc-capi  nc-cli  nc-web  nc-checkasm      front ends and tools
layer 4   nc-h264  nc-hevc  nc-av1  nc-vp9          one CodecBackend each,
          nc-av2   nc-vvc                           plus that codec's decoder
layer 3   nc-encoder                                shared engine + CodecBackend trait
layer 2   nc-analysis                               pyramid, ME, pre-analysis
layer 1   nc-dsp                                    pixel kernels + SIMD dispatch
layer 0   nc-common                                 pictures, views, bit I/O, YUV I/O
```

A crate depends only on crates in lower layers; `tools/ci/check_layering.py`
enforces it, so the engine can never reach into a codec and codecs never
depend on each other. `unsafe` is allowed only in `nc-dsp` kernels and the C
ABI of `nc-capi`; every other crate has `#![forbid(unsafe_code)]`.

- `nc-capi`: a C ABI (`include/nanocodec.h`) over a safe, codec-independent
  encoder session; panics are caught at the boundary.
- `nc-cli`: the `nanocodec` command (encode, decode, params, stream tools).
- `nc-web`: browser bindings; encoder and decoder run in separate workers
  and only compressed bytes cross between them.
- `nc-checkasm`: SIMD-vs-scalar bit-exactness checks and kernel benchmarks.

## 4. Pictures and views (`nc-common`)

- `Plane` owns one plane of 8-bit samples.
  - **Rows:** the first visible sample of every row is 64-byte aligned
    (`ROW_ALIGNMENT`), and `stride` is a multiple of 64.
  - **Border:** a padded border lets motion compensation and sub-pel filters
    read outside the picture without clamping; `extend_borders()` fills it by
    edge replication, the out-of-picture rule of every codec here.
  - Alignment comes from over-allocating a `Vec<u8>`; no `unsafe`.
- `Picture` is three planes in 8-bit 4:2:0; chroma gets half the luma border.
- `PlaneView` / `PlaneViewMut` are borrowed rectangular windows. The
  constructor checks once that the slice covers every row
  (`len ≥ (h − 1) · stride + w`); SIMD kernels build on that invariant.
  `Plane::region` takes signed coordinates, so a view can reach into the
  border; motion searches address candidates by position inside one padded
  view (`Dsp::sad_x4_at`) because building a view per candidate cost more
  than the SAD itself.
- Bit writers/readers and `yuv::{read_frame, write_frame}` (packed I420) live
  here too.

## 5. Source analysis (`nc-analysis`)

Everything computed from the input pictures alone, once per frame, by
`Analyzer`, before any codec decision:

```
pyramid     1/2, 1/4 luma (2x2 box means)
pre         per 8x8: mean, variance, 8x8 + 4x4 SADs vs the previous source;
            chroma 4x4 SADs; static flag; frame mean luma
motion      per 64x64 superblock, after SVT-AV1 M12 rtc:
            zero-vector SAD gates (static bypass / one position / window at
            zero / full search); HME at 1/4 res seeded by last frame's vector
            or camera motion; a small full-pel window with the 8x8-grid
            kernel, giving the best vector of all 85 square blocks
            8x8..64x64; per-SB me_dist and 8x8 cost variance
global      camera motion (translation + zoom), least squares on the 16x16 field
noise       running sensor-noise sigma from the flattest blocks' best matches
still       persistent stillness: static runs plus a calm neighbourhood
link        the reference each frame searches against under hierarchical-P
```

- A block is **static** when its 8x8 SAD is small, no 4x4 quarter changed on
  its own (a small object moving in a still block) and neither chroma block
  changed (colour-only changes). "Small" scales with the measured noise, so
  sensor noise on a still scene does not read as motion. Static superblocks
  skip the search entirely; this is the main speed lever on fixed cameras.
- Motion search runs on **source** frames, so it is codec-independent and
  shared. Sub-pel refinement and the final vector choice run later, per
  codec, against that codec's own reconstruction.
- Live camera input rarely cuts, but sudden motion (whip-pans, motion after a
  still period) fails the zero-vector gates and gets the full search;
  lighting jumps show in the frame mean luma.
- Timing goes through the `StageTimer` trait (the library never reads a
  clock; wasm32 has none); `examples/analyze.rs` prints per-stage ms/frame
  and search-path counts.

## 6. The shared engine (`nc-encoder`)

Per frame:

```
 input → analysis (nc-analysis)
       → frame plan: frame type, reference, temporal layer, QP (rate control)
       → per superblock: partition plan → per leaf: mode decision
              → backend: predict, transform, quantise (trellis), reconstruct,
                entropy-code immediately (single pass)
       → loop filters (backend; strengths from shared models/searches)
       → packet + per-frame metadata; rate-control update; reference update
```

### 6.1 References, GOP and temporal layers

- `structure` owns which earlier frame each frame predicts from, which
  reference slots it replaces and its temporal layer: a flat chain, or
  hierarchical-P with up to three layers (L1T2/L1T3). Every frame references
  only earlier frames, so dropping the top layer never breaks decoding; a
  per-layer QP cascade gives the base layer more bits.
  Signalling is per codec (H.264 `nal_ref_idc` / sliding window / reference
  list modification, HEVC TemporalId + RPS, AV1/AV2 refresh flags and
  temporal-layer OBU extensions, VP9 reference slots).
- `qp_cadence`: flat IPP streams still give every fourth frame "base"
  treatment, SVT's virtual three-layer allocation, changing only the frame
  quantizer, so the stream stays plain IPP.
- `ltr`: a long-term reference referenced for recovery only after the
  receiver acknowledges it, so recovery after a lost frame is a cheap P frame.
- `second_reference` (`refs.second`, off by default): the last high-quality
  frame (key or cadence base) kept in the long-term slot as a second
  reference for the frames just after it; mode decision chooses the
  reference per block (H.264 so far).
- `refresh`: GDR-style intra refresh, a column of intra blocks sweeping
  across the picture so a decoder that lost a frame or joined late recovers
  without a key frame.
- Key frames: periodic, on request, or at a detected scene cut (`scene`).
- Each packet carries metadata (temporal layer, references used, recovery
  point, discardable) so a transport can act without parsing the bitstream.

### 6.2 Partition planning

The block structure is chosen **from statistics before coding**, not by
trying every split:

- `partition_plan`: SVT-AV1's "very light PD0" for inter frames. The size cap
  comes from the 8x8 cost spread, depth removal from `me_dist`; each leaf is
  costed as its whole-sample prediction variance (capped at λ/8 per sample)
  plus λ times its partition bits, with a parent bias. Fully static nodes are
  never split.
- Rectangular shapes (`shape`) are offered where the backend declares them.
- `key_plan`: the key-frame partition plan per superblock.
- Leaves carry their measured residual statistics so mode decision reuses
  them.

### 6.3 Mode decision

`decision` follows SVT's real-time "light PD1" pass: a non-RD ranking with a
real cost check only where it matters.

- Inter candidates (zero, the codec's cheap predicted vectors, the motion
  search result) are ranked by residual variance + λ·bits; sub-pel
  refinement runs on the winner only (SVT's fixed-stage cross, half then
  quarter, with a bias towards the integer vector).
- Intra is tried only when inter is poor, from a shortlist (`intra_candidates`)
  ranked by SATD; the codec's full mode set is reached through it.
- Still blocks take a zero-residual skip (`still_skip`, `zero_residual`,
  `pre_tx_skip`) with a quality guard: an unguarded background skip costs
  far more bits than it saves.
- `psy` adds a psychovisual term; `random_decisions` drives the syntax-
  coverage stress test (every legal decision exercised, PROCEDURES
  "Syntax-coverage stress test").

### 6.4 Quantisation and RD

- Distortions are sample-domain squared error and `lambda()` is per bit, so
  the engine compares costs across codecs without knowing their syntax.
- `trellis`: shared trellis quantisation (RDOQ). A dynamic program over scan
  positions in reverse scan order chooses a transform block's levels
  together, minimising distortion + λ·bits under the codec's real entropy
  contexts, which each backend supplies. Bitstream syntax is unchanged; only
  levels differ.
- `aq`: spatial AQ from luma variance with a noise floor; `chroma_offset`:
  a noise-adaptive chroma quantizer offset (a finer chroma step pays on clean
  content but would spend bits on noise), used by AV1/AV2 without the policy
  below.
- **Chroma allocation is one engine policy** (`chroma_policy`; design
  decision, Mark 2026-11-10). How finely chroma is quantised relative to
  luma, and how chroma error is weighed against bits, is decided once for
  every codec, in a codec-independent unit (the H.26x QP of a sample-domain
  step, 6 per doubling), and each backend maps it to its own syntax:
  - *Step:* chroma uses luma's step up to QP-eq 36, then gets finer by half
    a QP per luma QP, to 3 QP finer from QP-eq 42 on: the JVET SDR CTC chroma
    map (VVenC's and VTM's default) without its 1-QP coarser bump below 36.
    That is the middle of the reference encoders at their real-time
    defaults: SVT-AV1, libaom, libvpx and AVM code chroma at the luma qindex,
    VVenC (QPA) 0-2 finer, x265 and x264 (their codecs' fixed tables, offset
    0) 3-8 finer at QP 37-45. `chroma.offset` shifts the whole curve.
  - *RD weight:* chroma squared error counts (step_Y / step_C)² times
    luma's, as HM/VTM/VVenC weigh it (2^((QP_Y − QP_C)/3)) and as AV1-family
    encoders do at their equal steps (1).
  - *Mapping:* H.264 sends one PPS per `chroma_qp_index_offset` the policy
    uses with every key and names the picture's in each slice (Table 8-15 is
    fixed and the offset is PPS-only); HEVC sets `pps_cb/cr_qp_offset` 0 and
    sends each picture's offset as `slice_cb/cr_qp_offset` (deblocking reads
    the PPS offset alone, §8.7.2.5.5); VVC signals the policy curve as its
    SPS chroma QP table; AV1, AV2 and VP9 pick each frame's `delta_q_u/v`
    (VP9 `delta_q_uv`) whose step is nearest the target.
  - *Status:* `chroma.policy` (engine default 0) switches it on, at
    `chroma.offset` -1 (1 QP finer than the curve above at every rate, which
    measured better than 0 in every codec). It is off in every codec: AV1
    Default used it (TRAIN J_real -0.0%, VAL -1.4%) until it was turned off
    2026-11-11 by owner decision (on the new score, PSNR-YUV, VMAF v1,
    MS-SSIM in equal thirds, +1.2% to +2.4% worse with a class guard
    tripped). The codecs keep their own chroma settings: every one of them codes chroma finer
    than its reference encoder (0.2-3 dB more chroma PSNR over luma at 720p
    CBR 600) because J_real's PSNR-YUV term (6:1:1) rewards cheap chroma, and
    moving to the policy trades PSNR-YUV (3-6 % worse) for VMAF NEG and
    MS-SSIM (about 2 % better): neutral for H.264, VP9 and VVC (TRAIN within
    +-0.2 %, H.264 tripping the static guard by 0.05), worse for HEVC and AV2
    (pan guard). Evidence and next steps: `docs/handoff/chroma-policy.md`.
- `symbol_cost`: table-driven bit estimates, so ranking never trial-encodes.

### 6.5 Loop-filter decisions

Backends own their filters (deblocking, SAO, CDEF); the engine owns the
shared strength logic: `cdef_model` (strengths from Q) and `cdef_search`
(a small strength-set search, reused across frames where it pays).

### 6.6 Rate control

`rate_control` is low-delay, frame level: packet-size feedback into a virtual
1 s VBV, a per-codec rate model (`CodecBackend::rate_model`), no lookahead,
no re-encode and no padding. `quality_cap` sets the finest quantizer any
frame may use (the target is a ceiling), informed by `refinement_gain`'s
online measure of whether finer inter frames still buy quality;
`startup_lookahead` sizes the first
key frame from the first two source frames' motion. Frame dropping is opt-in
(`RateControlConfig::frame_drop`), because analytics users need every frame.
Fixed QP remains available and is what BD-rate sweeps use.

## 7. The `CodecBackend` trait and the codecs

The engine decides *what* to code (frame type, partitioning, modes); the
backend decides *how* it is written. Through `CodecBackend` a codec declares
its geometry (superblock and minimum block size, legal shapes), supplies
predicted vectors and seeds, partition/mode/vector bit costs, its rate model
and λ, and then encodes each decided block (prediction, transform, quantise,
reconstruct, entropy-code), filters the frame and returns the packet. Default
methods let a codec opt out of capabilities it does not have (temporal
layers, long-term references, intra-leaf costs).

| Codec | Profile and tools (encoder) | Decoder |
|---|---|---|
| H.264 (`nc-h264`) | High, CABAC; 16x16 to 8x8 inter partitions, intra 4x4/16x16, 8x8 transform; quarter-sample MC; trellis; deblocking | 8-bit High conformance suite |
| HEVC (`nc-hevc`) | Main; CTU 64, 2Nx2N CUs 8–64 (NxN intra at 8x8), angular intra, merge/AMVP/TMVP, transform trees to 4x4 with DST; deblocking + SAO edge offset | full Main / Main Still Picture, MD5-exact on the JCT-VC Main set |
| AV1 (`nc-av1`) | Main 8-bit 4:2:0; SB64 with square and rectangular partitions; directional intra with angle deltas, filter intra, CfL; single-reference inter plus LTR slot; adaptive CDFs; trellis on P frames; deblocking + CDEF | toward full Main conformance (libaom vectors, see `docs/handoff/av1-decoder.md`) |
| VP9 (`nc-vp9`) | Profile 0; 64x64 to 8x8 partitions, DC intra, single-reference inter, transforms to 32x32, deblocking | any conforming Profile 0 stream |
| AV2 (`nc-av2`) | 8-bit 4:2:0 I/P; intra modes except CfL, skip, quarter-pel motion | our emitted subset |
| VVC (`nc-vvc`) | 8-bit 4:2:0 quadtree, 67 intra modes, skip/merge/AMVP, DCT-II, deblocking + SAO | our emitted subset |

Each backend keeps its own reconstructed references, since reconstruction
differs per codec. Per-codec detail and supported subsets live in each
crate's module docs and `docs/handoff/<codec>.md`.

## 8. Kernels and dispatch (`nc-dsp`)

```
Dsp (per encoder/decoder instance, Copy)
 ├─ isa: Isa                        Scalar | Avx2 | Neon | Simd128
 ├─ pixel: PixelFns                 SAD (4..64 wide), SAD x4, SATD 4x4/8x8
 └─ sample: SampleFns               moments, 2x downscale, ...
```

- **Selection.** `Dsp::new()` picks `Isa::best()`; `Dsp::with_isa(isa)` picks
  one explicitly (tests, benchmarks). x86-64 detects AVX2 at run time;
  aarch64 always has NEON; wasm32 has SIMD128 when built with
  `-C target-feature=+simd128`. The table belongs to each instance; there is
  no global state.
- **Safe API, unsafe kernels.** `Dsp` methods check views against the block
  size, then call the kernel. Kernels are `unsafe fn`s that may assume the
  slices cover the block, the block edges are valid and the CPU supports the
  ISA; the soundness argument is written once in `dispatch.rs`. `unsafe`
  blocks surround only loads, each with a `// SAFETY:` comment.
- **Scalar is the definition.** Every SIMD kernel returns bit-identical
  results to its scalar reference, checked by `nc-checkasm`. A table entry
  may point at the scalar kernel where that is faster (wasm32 4x4 SAD), with
  the measurement in a comment.
- **Codec kernels are written once** over small lane traits (`simd/`:
  `I32Lanes`, `I16Lanes` and loop-filter extras), and instantiated per ISA by
  thin `#[target_feature]` wrappers that pick the vector width from the
  block width:

```
txfm/        AV1 DCT 4..32 forward / inverse+add, quantise, dequantise
             (1-D butterflies generated from libaom by tools/codegen)
codec_txfm/  HEVC DCT 4..32 + DST 4x4, H.264 4x4 / 8x8 + DC Hadamard
mc/          AV1, HEVC, H.264 interpolation filters
diff/        residual, add-residual, variance, sub-pixel variance
cdef/ sao/ deblock/   loop filters, one lane per line, decisions as lane masks
```

- **References.** The AV1 transform reference is the same butterfly code on
  one `i32` lane, checked bit-exact against libaom's C on random and extreme
  inputs on x86 and the Jetson; HEVC/H.264 kernels are tested against literal
  ports of the backends' scalar code.
- **Branch-free lane decisions:** a comparison is `min(max(b - a, 0), 1)` and
  a select is `old + mask * (new - old)`.
- **SATD** is the sum of absolute 2-D Walsh-Hadamard coefficients of the
  difference: 8x8 tiles when both edges are ≥ 8, else 4x4, normalised as in
  the HEVC reference software (`(s + 1) >> 1`, `(s + 2) >> 2`). The SIMD
  horizontal stages use pairwise add/sub (`phaddw/phsubw`, `uzp1/uzp2`), so
  no transpose is needed.

**Adding a kernel:** declare its type and table field (`pixel/mod.rs` or
`sample/mod.rs`); write an obvious scalar reference with hand-computed unit
tests; add each SIMD version to that ISA's table; add a `Dsp` method that
validates views and calls the kernel inside a commented `unsafe` block; add
it to `nc-checkasm` (`check.rs`, `bench.rs`); run it on the Jetson and under
node (SIMD128), since CI here only cross-checks aarch64.

## 9. Decoders and WASM

- Each codec crate carries its decoder. They share the normative helpers and
  `nc-dsp` kernels with the encoder's reconstruction path, so a decoder is
  essentially entropy decoding plus code the encoder already has.
- The goal is standard profile compliance (8-bit only) without slowing the
  decode of our own streams: H.264, HEVC and VP9 pass their conformance
  suites; AV1 is close; AV2 and VVC decode our emitted subset. A decoder that
  meets syntax it does not support rejects the stream rather than guessing.
- Every stream we report decodes bit-exactly in two standard decoders per
  codec and in our own decoder, compared by per-frame hashes against the
  encoder's reconstruction.
- `nc-web` builds encoder and decoder to WebAssembly with SIMD128; the
  browser demo (`web/`) compares two codecs side by side. WASM decode runs at
  roughly 1.2–1.8× the native time (README "In the browser").

## 10. Presets and parameter tuning

- Every heuristic threshold or multiplier is registered in a `param_group!`
  (`nc-encoder/src/params.rs` for the engine, the backend crates for theirs)
  and read once per encoder. `nanocodec params` lists them; `--preset`,
  `--params FILE` and `--param name=value` set them.
- Each codec has two presets, the built-in files in `presets/`: **Default**
  (the quality point) and **Fast** (the speed point, which carries the CPU
  bar). Default must cost at least 1.5× Fast's CPU and buy real quality for
  it.
- Presets are points on a Pareto front found by a joint multi-objective
  search over the registered parameters (`eval/nceval/tune.py`, Optuna TPE)
  on TRAIN clips, then confirmed on VAL and on the Jetson. Levers interact,
  so tuning them one at a time misses the front.
- CPU is judged on the Jetson (primary) and x86 together, weighted about 2:1;
  a change may not regress one platform to win the other.

## 11. Evaluation

The harness (`eval/`, `python3 -m nceval`) is the ground truth for every
decision. In short:

- **Clips:** live camera content in content classes (static, bodycam, crowd,
  low-light, pan, plus pristine texture), split into TRAIN (tuning), VAL
  (confirmation, never screened on) and held-out sets; never above half the
  source bitrate.
- **Anchors** run the same low-delay, single-thread setup: openh264, x264,
  x265, SVT-AV1 M12 `--rtc`, libvpx and libaom, pinned in
  `tools/anchors/versions.env`.
- **Objective:** every encoder in its own real-time CBR at a common rate
  ladder; J_real is the class-balanced mean BD-rate over PSNR-YUV, VMAF v1 and
  MS-SSIM (full range and the low-rate half) plus a CPU term, with guards on
  every class, worst frames, flicker and buffer compliance.
- **Speed:** single-core CPU per frame at matched rate on the Jetson and the
  PC; instruction counts (callgrind, Jetson counters) for small changes.

Methodology detail: [eval/README.md](../eval/README.md) "Methodology"; the
published results and their method: [docs/results/](results/README.md).

## 12. Checks and toolchain

`tools/ci/check.sh` runs `cargo fmt --check`, clippy with `-D warnings` for
the host, wasm32 and aarch64, the tests, `nc-checkasm` on host SIMD and on
SIMD128 under node's WASI, and the layering check; `--quick` skips the
wasm32 and aarch64 steps. Workspace lints are clippy `pedantic` with a short,
commented allow-list in `Cargo.toml`; public items need docs.

Rust stable via rustup in `~/.cargo` on both machines (not on PATH by
default; scripts prepend it). Targets: `x86_64-unknown-linux-gnu`,
`aarch64-unknown-linux-gnu` (cross-checked here, native on the Jetson),
`wasm32-unknown-unknown` (browser) and `wasm32-wasip1` (tests under node).
