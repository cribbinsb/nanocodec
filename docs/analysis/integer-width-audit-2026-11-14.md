# Integer-width audit: 32-bit storage and compute for 16-bit values (2026-11-14)

**Question (owner).** "If we are using 32b storage/compute for something that is 16b that is STUPID and
should be generally fixed." nanocodec is 8-bit only, so coefficients, residuals, prediction and most
transform intermediates fit in `i16`. dav1d, libvpx, openh264 and VVdeC keep them in 16 bits. That doubles
the SIMD lanes and halves memory traffic. This matters most for the WASM SIMD128 browser decoder and for
NEON on the Jetson.

**Scope.** Every codec crate (encoder and decoder), nc-encoder, nc-analysis and nc-dsp. The AV1 decoder's
dequantised coefficient storage and inverse-transform path (`nc-av1/src/decoder/coefficients.rs:193`,
`nc-dsp/src/av1_inverse/`) is being fixed separately by kernels-q and is not covered here.

## Method

- **Code audit.** Read-only, one pass per area (H.264 + HEVC, VVC, AV1 encoder + AV2, VP9, the shared
  engine and the generic nc-dsp kernels). Each candidate has a range proof from the spec or from exact
  arithmetic.
- **Profiles.** Fresh Jetson `pcsample` profiles at `variance-integration` 98bbd409, on crowd_mot17_09 720p
  CBR 600 kbps:
  - every Default encoder, 60 frames;
  - our decoders, on 150-frame streams of the same clip.

  Cycle shares are self time. In the tables, "enc" and "dec" give each kernel's share of the whole program
  on the Jetson. Each "est. gain" figure is a whole-program estimate for the codec(s) named, from the share
  times the fraction of that work that 16-bit lanes remove. These are estimates, not measurements.
- **Effort.** S is a contained kernel or type change (under half a day). M changes a kernel API and its
  callers in one codec, plus three ISA implementations and checkasm. L reshapes a pipeline across codecs.

### Lanes are not free on NEON

- **Where 16 bits pays.** On NEON, a 16-bit multiply that widens into 32 bits (`smull`/`smlal`, 4 products
  per instruction) runs at the same rate as a 32-bit `mla.4s`. So narrowing pays where the arithmetic
  itself is 16-bit: add, subtract, shift, clamp, min/max, abs, `sqrdmulh`-style rounding products, loads,
  stores and zeroing.
- **Where it pays less on NEON.** For multiply-accumulate transforms (HEVC/VVC/AV2 DCTs, quantiser
  products), 16-bit storage mostly saves loads, narrowing and memory.
- **WASM SIMD128 and AVX2.** These have a real 2x multiply-accumulate on 16-bit pairs
  (`i32x4.dot_i16x8_s`, `pmaddwd`), so the same change gains more in the browser than on the Jetson.

## Ranked table (estimated gain per effort)

| # | Where (file:line) | Now | 8-bit range proof | Saturation needed | Jetson share | Reference | Est. gain | Effort |
|---|---|---|---|---|---|---|---|---|
| 1 | **H.264 decoder coefficients and inverse.** `nc-h264/src/decoder/transform.rs:42,47,50` (`Residual`); kernels `codec_txfm/generic.rs:1499,1537,1620` (`h264_add_scaled_{4x4,mb,8x8}`, `I32x4` lanes) | `i32` storage, 4-lane `i32` butterflies | Stored values are already clamped to −2^15..2^15−1 (`transform.rs:93`, §8.5.12.1). §8.5.12.2 and §8.5.13.2 require every named intermediate of the inverse to lie in −2^15..2^15−1. The butterflies are add, subtract and `>>` only, so wrapping 16-bit arithmetic is exact for conforming streams. | None at the store (clamp exists). The final `(x+32)>>6` overflows 16 bits: use a rounding shift (NEON `vrshr`) or `((x>>5)+1)>>1`, which is exact. | dec: `add_scaled_8x8` 3.2 %, plus `add_scaled_mb`/`_4x4` and the per-block zeroing | openh264 and ffmpeg 8-bit: `int16_t` | 1.5–2.5 % of H.264 decode (Jetson); more on WASM | S–M |
| 2 | **VP9 decoder coefficient buffer.** `nc-vp9/src/decoder/frame.rs:99`, stores `coeff.rs:642`, inverse API `nc-dsp/src/vp9_inverse/packed/mod.rs:62,244` | `[i32; 1024]`, packed on load | §8.6.2 requires every `Dequant[i][j]` to fit in 8+BitDepth = 16 bits. Every 8x8+ consumer already truncates to `i16` on load, so storing `i16` is bit-identical. | Store `as i16`; keep the `level·step` product in `i32` | dec: `inverse_add` 5.8 % (load+pack part of it) | libvpx non-HBD `dqcoeff`: `int16` | 0.7–1.2 % of VP9 decode | M (the encoder recon shares the API) |
| 3 | **Motion-search grid.** `nc-dsp/src/pixel/mod.rs:183` `GridBest.sad`, `keep_min` (`pixel/neon.rs:490`) | `u32` | 8x8 SAD ≤ 64·255 = 16320; 16x16 SAD ≤ 65280 (even-row doubling included) < `u16::MAX` | None (32x32/64x64 entries stay `u32`) | enc: `keep_min` 1.3 % (H.264), `sad_grid_best_x4` 1.4–4.5 % all codecs | SVT-AV1 ME: u16 lanes for 8x8/16x16 | 0.3–0.7 % of every encoder | S–M |
| 4 | **HEVC decoder levels and inverse input.** `nc-hevc/src/dec/ctu.rs:191,193`; `codec_txfm/neon.rs:427` (`inverse_add_buffered` over generic `I32x4`) | `Vec<i32>`, 4-lane `i32` butterflies | Levels rejected beyond −32768..32767 (`dec/residual.rs:367,380`); scaled values clamped (`ctu.rs:1291`, §8.6.3); the first pass is clipped to 16 bits (§8.6.4.2) | Accumulators stay 32-bit (`smlal`, `dot_i16x8`) | dec: `inverse_add_buffered` 6.6 %, `quadtree` zeroing 1.3 % | ffmpeg/x265 int16 (facts only); HM `Int` | 1.5–3 % of HEVC decode; WASM already uses 16-bit pairs | L (needs a 16-bit NEON inverse; shared with VVC and the encoder) |
| 5 | **VVC decoder inverse and dequant.** `codec_txfm/generic.rs:1124` (`hevc_dequantize`), `vvc_mts/generic.rs:98-145`, `nc-vvc/src/decoder_blocks/transform.rs:33`, `nc-dsp/src/vvc_recon.rs:40` (copies into an `i16` scratch) | `i32` | §8.7.3 clamps coefficients to 16 bits; §8.7.4.1 clips the first stage to 16 bits | Already clipped | dec: `hevc_idct16/32_bounded` 4.5 %, `hevc_dequantize` 2.3 % | VVdeC: `TCoeff` int32, SIMD on 16-bit | 1–2 % of VVC decode | L (with #4) |
| 6 | **AV2 inverse output and intermediates.** `nc-dsp/src/av2_txfm/mod.rs:290`, `av2_txfm/generic.rs:297,313`, `nc-av2/src/tile_decoder.rs:103` | `i32` | The output is clamped to −256..255; `t` and `m` are `clamp16` (`generic.rs:302,321`) | Keep the clamps | dec: `residual_planes` 6.1 %, `inverse16/32` 3.4 % | AVM int32; dav2d int16 | 1–2 % of AV2 decode | M |
| 7 | **Encoder coefficient and quantiser path.** Forward output, levels and dequant buffers, per codec: H.264 `nc-h264/src/macroblock.rs:355`; HEVC `nc-hevc/src/backend.rs:650`; VVC `nc-vvc/src/recon.rs:189`; AV1 `nc-av1/src/backend.rs:2277-2302`; VP9 `nc-vp9/src/backend.rs:339-354`, `recon.rs:245`; AV2 `nc-av2/src/block_coder.rs:476`. Kernels `txfm/mod.rs:185-205`, `codec_txfm/mod.rs:654-784`, `vp9_quant/mod.rs:97-124` | `i32` throughout | Forward outputs: H.264 4x4 ≤ 9180, 8x8 ≤ 16320; HEVC/VVC ≤ 32640 (post-shift); AV1 DC peak 32640 at every size; VP9 ≤ 32640 + rounding; AV2 up to ~32965 at 64x64 (beyond `i16`, see below). Levels ≤ 32767 (capped). | Saturating store at the forward output (an encoder-only, non-normative choice). Products `\|c\|·mf + offset` stay 32-bit. | enc: H.264 `quantize` + `forward_quantize_chroma` 3.5 %; HEVC/VVC `hevc_quantize_from` 1.7–1.8 %; AV1/AV2 `quantize_levels` 1.9–2.0 %; VP9 `quantize` 3.1 % + `largest_magnitude` 1.2 % | libaom lowbd rtc quant int16; openh264 int16; libvpx int16 | 0.4–1 % per encoder on the Jetson (the products do not gain); 1–2 % on x86/WASM | L (per codec) |
| 8 | **Variance sum accumulator.** `simd/avx2.rs:797`, `simd/neon.rs:432`, `simd/simd128.rs:418` (`acc_add`) | `i32` lanes, widened per vector | A lane holds ≤ 128 adds of ≤ 255 (32640). Every size up to 32x64 fits one 16-bit pass. | Widen once (64-wide) | enc: `variance::<N>` 1.1–2.7 % | libaom AVX2: sum kept in `i16` | Jetson ≈ 0 (`sadalp` is already cheap); x86 0.3–0.5 % | S |
| 9 | **Pre-analysis SAD quads.** `pixel/mod.rs:94,104`; `nc-analysis/src/preanalysis.rs:103,259-269` | `u32` | 4x4 SAD ≤ 4080; 8x8 ≤ 16320 | None | enc: `sad_quads_8x8_row` 1.2–1.6 %, `sad_4x4_macroblock` 1.9 % (H.264) | openh264 int32 | 0.2–0.4 % | S |
| 10 | **VP9 4x4 inverse and residual.** `nc-dsp/src/vp9_txfm.rs:292-319`, `nc-vp9/src/hybrid.rs:30-95` (scalar, `i64` maths), `nc-vp9/src/recon.rs:246` | `i64`/`i32` | Wrap the inputs to `i16` (§8.6.2); the products then fit `i32` (2·32767·15212 < 2^31) | Wrap input | scalar on every ISA | libvpx int16 in, int32 maths | 0.2–0.5 % of VP9 decode | S |
| 11 | **AV1 loop-restoration SGR rows.** `nc-dsp/src/av1_restoration/self_guided.rs:97,191,121,242` (`sums`, `a`, `filtered`) | 8×`i32` lanes | Box sums ≤ 6375; `a` ∈ [1, 256]; `filtered` < 2^13 | None | not in the top 45 of the crowd decode | dav1d 8bpc: int16 for these rows | < 0.3 % (content-dependent) | M |
| 12 | **VVC encoder dep-quant and BDOF/PROF.** `nc-vvc/src/dep_quant.rs:48-113` (`Step.level`, `nearest`); `vvc_refinement/bdof.rs:26`, `vvc_optical_flow.rs:16` | `u32`/`i32` | Levels ≤ 32767; gradients ≈ ±575; PROF corrections ±31 (§8.5.6.5) | None | not in the top 30 | VVenC `TCoeffSig` int16 | < 0.3 % | S each |
| 13 | **Small items.** `nc-dsp/src/av1_mc/convolve.rs:101` (scalar ISA, `vec![0i32]` per call); `av1_film_grain/kernels.rs:18` scaling LUT `[i32;256]` (fits `u8`); `nc-encoder/src/trellis.rs:47` `Candidates.levels [u32;3]`; `nc-encoder/src/intra_estimate.rs:153` `square_error`; `nc-analysis/src/grain.rs:299-345` (`f64` noise in 1/16 units, exact as `i16`) | `i32`/`u32`/`f64` | See each line | — | — | — | < 0.2 % each; code simplification | S |

**Already narrow (no action).** These already run in 16 bits:

- **Shared and generic kernels:**
  - MC and compound intermediates on every codec;
  - SATD (16-bit Hadamard, `madd` totals);
  - residual differences.
- **AV1/AV2:**
  - the CDEF window;
  - the Wiener ring;
  - CfL AC;
  - the AV1 NEON forward columns.
- **HEVC:**
  - the 16/32-point forward rows;
  - the SIMD128 inverse;
  - the encoder's `level_store`.
- **VVC:**
  - `vvc_mts` (16-bit pairs);
  - ALF;
  - deblock and SAO.
- **VP9:**
  - the packed inverse internals;
  - MC;
  - intra.

## Where 16 bits is NOT safe (do not narrow)

1. **Transform accumulators before their shift.** These cover every HEVC/VVC/AV2 forward and inverse sum
   (32 taps × 90 × 32767 ≈ 2^26.5), the VP9/AV1 butterfly products before `round_shift(·, 14)`, and the
   H.264 forward 8x8 column pass before `ROUND_6` (≈ 2^20). Only the stored, shifted values fit. A single
   product by a doubled constant can use `sqrdmulh`/`pmulhrsw` exactly; sums of two products cannot.
2. **H.264 Intra16x16 luma-DC Hadamard in the encoder** (`nc-h264/src/macroblock.rs:599`). It reaches 65280
   before its `>>1`. It needs 17 bits, and halving earlier changes the rounding.
3. **H.264 inverse `(h + 32) >> 6`.** `h` may be 2^15−1, so the add wraps. Use a rounding shift or the exact
   identity `((h >> 5) + 1) >> 1`.
4. **Dequantisation products.** These are H.264 `level·LevelScale << qp/6`, HEVC/VVC `l·m·scale`,
   AV1/AV2 `level·q` (24-bit masked) and VP9 `|level|·step` (about 30M). All stay 32/64-bit. Only the
   clamped result is stored as `i16`.
5. **Quantiser numerators.** These are `|c|·mf + offset`, and the VP9 32x32 `(|c|<<1) + offset`
   (≈ 66.4k > `u16`).
6. **Hostile decoder input.** Parsed levels must stay `i32` (or saturate) until each codec's range check:
   - H.264: CAVLC escapes and CABAC UEG0 exceed 2^15 (`decoder/cavlc.rs:243`).
   - VVC: the remainder escape reaches 2^16+ (`decoder_blocks/coeff.rs:439,448`, checked at `:363`).
   - AV2: levels are 20-bit (`nc-av2/src/transform.rs:54`, checked at `:738`).

   Narrowed SIMD paths must use wrapping operations, and their scalar references must use `wrapping_*`, so
   that a hostile stream decodes differently but never panics.
7. **Sums of squares and block totals.** This covers all SSE and variance SSE (`u32` even at 4x4),
   `Moments.sum_sq`, SGR squares and `b` (≈ 65280, keep `i32`), `p` and `z`, and the CDEF direction costs.
   It also covers SAD at 16x32 and above (≥ 130560), the 8x8 SATD raw total (130560) and any single 16x16
   Hadamard (65280).
8. **BDOF flow sums** `sums[2..3]` (≈ 122k), the BDOF and PROF correction sums, the MIP dot, CCLM and GEO
   blends (≈ 2^18).
9. **AV2 forward output at 64x64** can reach about 32965 (Cauchy–Schwarz bound), beyond `i16`. A
   saturating store or a range probe (`txfm/range_probe.rs` style) is needed before narrowing. The AV1
   forward output's margin to 32767 is about 127 rounding counts, so it needs a saturating store too.
10. **The Wiener horizontal pass** only fits 16 bits because of its `128·centre` offset and the clamp to
    −2048..6143 (`av1_restoration/wiener.rs:11-16,42`). Keep that trick.

## Bugs found during the audit (beyond width)

- **VVC decoder panic on hostile input.** `nc-dsp/src/vvc_recon.rs:13` asserts `-32767..=32767` in
  `vvc_joint_chroma`. The clipped inverse can legally produce −32768, so a crafted stream panics. A clamp
  to −32767 leaves conforming decode unchanged, because any |residual| > 255 saturates the 8-bit add anyway.
- **VVC encoder dependent quantisation may exceed CoeffMax (unverified).** `nc-vvc/src/dep_quant.rs:67`
  does not cap `k`. On 64x64 blocks at QP near 0 the index 2k−1 can exceed 32767, and our own decoder
  rejects that (`decoder_blocks/coeff.rs:363`). The fix is to cap `k` at 16383.
- **VP9 trellis can step past the dequant cap.** `nc-vp9/src/coeff_trellis.rs:247-249` (`plain + 1`)
  checks `MAX_LEVEL` but not `Vp9QuantStep::cap`. At very low q-index with full-scale DC, dequant can then
  exceed 32767, which breaks §8.6.2.

## Phase 2 (results, updated as changes land)

Each change is exact. Streams are byte-identical old vs new on crowd 720p, and decoded pictures are
hash-identical. The H.264 (65/65), HEVC (136/136) and VP9 (303/303) conformance vectors still match their
reference MD5s. checkasm passes on AVX2, NEON (Jetson) and SIMD128 (node) against the scalar references.

The Jetson figures are jperf cycles, the minimum of 5 interleaved rounds, decoding a 150-frame crowd 720p
600 kbps stream. The x86 figures are callgrind instructions on a 60-frame stream.

| Commit | Change | Jetson cycles | Jetson instr. | x86 instr. |
|---|---|---:|---:|---:|
| 7997dd54 | VVC dep-quant: levels capped so 2k ≤ CoeffMax (a conformance fix; streams change only near QP 0) | — | — | — |
| 03eb2019 | VVC joint CbCr: no assertion on the legal −32768 (hardening; unreachable for chroma ≤ 32 points) | — | — | — |
| 4eab5924 | VP9 trellis respects the 16-bit dequant cap (unreachable at default content) | — | — | — |
| 6e458145 | H.264 decoder: i16 coefficients; 8x8 inverse in eight 16-bit lanes | **−1.95 %** | −1.38 % | −0.10 % |
| 316cbaaf | HEVC decoder: i16 levels/residuals; `hevc_inverse_add` takes i16 | −0.47 % | −0.02 % | −0.11 % |
| 3ce053bc | VP9: i16 coefficients/residuals through the inverse (3000 kbps stream) | −0.18 % | +0.12 % | −0.04 to −0.08 % |
| (this commit set) | VVC dep-quant steps u16; BDOF gradients i16, PROF corrections i8 | +0.07 % (noise) | ±0.00 % | — |
| int-width-vvc | HEVC/VVC dequantise, inverse and residual in i16 (VVC decode; encode −1.05 % cycles) | −1.10 % | −0.37 % | −2.20 % (31 f) |

Our low-delay P-only streams rarely exercise BDOF/PROF, so the VVC decode figure is flat: that change is
pure narrowing.

**HEVC/VVC dequantise, inverse and residual (item 5, the inverse half of item 4).**
- `Dsp::hevc_dequantize` and `vvc_dequantize` write `i16` through `store_sat16`, which is also the clamp.
  Levels stay `i32`, because parsed VVC levels are wider until their range check.
- `Dsp::hevc_inverse` reads and writes `i16` on every ISA. The unclamped second pass fits because
  `(2^15 · L + 2^11) >> 12` with `L ≤ 64N` stays below 2^14 + 1 (test `residual_bound_fits_i16`).
- The VVC inverses are 16-bit in and out: DCT, skip, primary/MTS, LFNST, the 64-point and the chroma LMCS
  scaling. So is joint CbCr, whose negation saturates without changing any reconstruction.
- `vvc_add_residual` no longer copies. It dispatches unless a value exceeds the adder's ±32512 contract
  (transform skip at the limit), which takes the scalar path.
- Exactness: HEVC conformance 136/136, VVC 9/9, and crowd hevc/vvc Default and Fast streams and decodes are
  identical. The relaxed-SIMD checkasm passes too.
- Other figures: HEVC encode −0.46 % Jetson cycles; x86 instructions −1.41 % (VVC encode) and +0.29 % (HEVC
  encode).
- Not done: a reused residual scratch in place of the per-TU `Vec<i16>`. The decoder's tests fingerprint the
  residuals between the inverse and reconstruction passes, so that change means merging the two passes.

**Encoder forward, quantiser and levels (item 7), AV1 and VP9.** Jetson: jperf min of 5 interleaved
rounds, Default, 30 frames 720p 600 kbps, crowd / lowlight / bodycam. x86: callgrind instructions, same
clips. 112/112 streams byte-identical on x86 and NEON (every codec's Default and shipped presets, two
clips, CBR and QP); checkasm passes on AVX2, AVX-512 (SDE `-spr` and `-icx`), NEON and SIMD128.

| Commit | Change | Jetson cycles | Jetson instr. | x86 instr. |
|---|---|---:|---:|---:|
| 1a363676 | AV1/VP9 forward DCT family stores i16 (saturating, never triggers); AV1 quantiser i16 in and out, generic over its storage (AV2 keeps i32), vectors from coefficient 0 with a DC fix-up; AV1 levels i16; VP9 coefficients, quantiser input and largest magnitude i16; transpose kernels removed | AV1 −0.30 / −0.49 / −0.18 %; VP9 −0.72 / −1.03 / −0.63 % | AV1 −0.56 / −0.88 / −0.48 %; VP9 −0.67 / −0.50 / −0.69 % | AV1 −0.58 / −0.73 / −0.49 %; VP9 ≈ 0 |
| (this commit set) | AV1 dequantised coefficients i16 (from coefficient 0, DC fix-up), coefficient gain on i16 (NEON 16-bit products), txfm inverse DCT input i16 on every ISA | AV1 −0.30 / −0.24 / −0.15 % (mean of 3) | AV1 −0.83 / −0.97 / −0.71 % | AV1 −0.85 / −1.07 / −0.76 % |

- **Store-to-load alignment mattered more than the width.** The first i16 quantiser started its vectors at
  coefficient 1, like the 32-bit one. Its 16-byte loads then straddled two of the forward's 16-byte stores,
  and Jetson AV1 cycles rose 0.3–0.7 % (pcsample: NEON `quantize_levels` 1.91 → 2.60 % of the encode)
  while instructions fell. Vectors from coefficient 0, with the DC re-quantised after
  (`txfm::scalar::fix_dc`), removed that and the scalar tail. The dequantiser needed the same change: from
  coefficient 1 its i16 stores cost +0.2–0.4 % cycles at −0.5 % instructions.
- **Not taken: the encoder's reconstruction on the decoder's `av1_inverse_add_i16`.** On the Jetson its
  dav1d path read scratch past the dequantised extent the encoder fills, so streams changed, and it cost
  +0.5–0.8 % cycles.
- **WASM:** encode A/B under node (bench.py clips, 720p 600 kbps, best of 3 alternating rounds) did not
  resolve these changes. The unchanged H.264 control moved by −1 to −22 % between rounds at machine loads of
  40–60.
- **Left for item 7:** H.264, HEVC/VVC and AV2 forward/quantiser/levels; VP9 levels (retry after
  profiling `residual_of`, the token writer and the trellis).
  The trellis `Candidates.levels` stay wide (AV2 levels reach 2^20 − 1; the HEVC/VVC/H.264 trellis
  nearest levels are uncapped).

**WASM** (node, `tools/web/bench.py`, H.264 + HEVC decode, 3 runs alternating old/new): run-to-run noise
was about ±5 % at a machine load of 45 to 57, and no difference above that noise was measured. A quiet
machine is needed to resolve the SIMD128 effect.

**The estimates were optimistic for VP9 and HEVC.** The old code already narrowed to 16 bits inside the
kernels (VP9 packed, HEVC SIMD128 pairs), so the 32-bit storage cost mostly loads and memory. H.264 gained
most: its 8x8 inverse on NEON went from 4 to 8 lanes.

**Ownership.**
- Items 3, 7, 8 and 9 (motion grid, encoder forward/quant/levels, variance accumulator, SAD quads) and
  trellis/RDOQ belong to the encoder agent.
- Items 5 (VVC/HEVC dequant and inverse API) and 6 (AV2 inverse), and 11 and 13 (SGR, film grain, scalar
  MC, grain analysis) are on their own `feature/int-width-*` branches.
