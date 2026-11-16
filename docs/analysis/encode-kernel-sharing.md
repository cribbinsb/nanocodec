# Encoder kernel sharing: audit and plan (2026-11-16)

Owner (Mark, 2026-11-16): "if a kernel can be shared with little performance
deficit, let it be so. Where different reimplementations of similar kernels are
needed lets make sure it's recorded in the code and we adopt a common best
pattern so we aren't reinventing the wheel or dropping performance". Rule:
CLAUDE.md "One kernel per pattern across codecs".

Scope: the nc-dsp kernels the six encoders call (H.264, HEVC, VVC, AV1, VP9,
AV2), including inverse transforms and reconstruction in the encode loop.
Decode-only kernels, `high_depth/` and `av2_dec/` are out of scope. This file
is the record: each family's table says what exists, whether the arithmetic
really differs, and the verdict. Merges update their row when they land.

## Method

- Inventory: read from the code at `39d64df57` (five read-only passes, one per
  family, plus the dispatch tables in `nc-dsp/src/dispatch.rs`). Paths are
  under `crates/nc-dsp/src/` unless they name a codec crate.
- Tiers: S scalar, A2 AVX2, A5 AVX-512, N NEON (N+dot: dotprod), W simd128.
  The AVX-512 table starts from the AVX2 table (`dispatch.rs` `Dsp::avx512`)
  and overrides only some entries; "A5 inh." means the AVX2 kernel runs there.
- Hot: callgrind self instructions of each codec's Default, fixed QP (chart
  QPs), 60 frames 720p, `crowd_mot17_09` / `bodycam_bc1`, x86 AVX2
  (zen5-codecs agent's profiles, `~/nanocodec-data/work/zen5-codecs/prof/base`,
  2026-11-16 07:02). Shares are of the whole program; "a / b" is crowd / bodycam.
- "Little deficit" for a merge: each affected codec's Default within ~0.3 % in
  callgrind instructions, streams byte-identical (`preset_identity.py`).

## Where the encoders spend kernel time (nc-dsp share of the whole program)

| Codec | nc-dsp total | Largest kernel families |
|---|---|---|
| H.264 | 44.7 / 47.4 % | pixel (SAD/SATD) 13.3 / 13.0, diff (variance) 11.2 / 14.8, codec_txfm 10.5 / 8.6, deblock 2.8 / 3.7, mc 1.7 / 2.1 |
| HEVC | 50.4 / 55.4 % | diff 19.7 / 20.7, mc 8.6 / 9.5, codec_txfm 5.2 / 5.5, pixel 4.6 / 6.1, txfm (staged forward) 3.6 / 5.3, sao 2.6 / 2.4 |
| VVC | 47.1 / 50.3 % | diff 9.6 / 10.7, vvc_intra 7.1 / 5.9, codec_txfm 7.0 / 6.4, mc 5.2 / 5.3, txfm 4.8 / 4.7, vvc_alf 4.7 / 7.1, vvc_deblock 2.4 / 3.1 |
| AV1 | 47.2 / 51.3 % | diff 13.2 / 15.7, txfm (forward + quant) 11.5 / 9.4, cdef 4.9 / 6.7, deblock 3.8 / 3.7, pixel 3.5 / 5.1, intra 3.7 / 2.7, mc 2.3 / 2.9 |
| VP9 | 39.0 / 42.1 % | diff 12.4 / 13.1, txfm (AV1 forward) 9.6 / 9.7, vp9_mc 3.5 / 4.6, vp9_loop_filter 3.5 / 3.7, vp9_quant 2.1 / 2.1, vp9_inverse 1.4 / 1.3 |
| AV2 | 37.5 / 45.5 % | diff 10.6 / 12.1, av2_txfm 5.8 / 7.3, cdef 5.3 / 7.2, deblock 2.8 / 2.5, txfm (AV1 quant) 1.9 / 2.1, mc 1.8 / 2.5, av2_intra 1.6 / 1.3 |

`diff` (variance, sub-pixel variance) and `pixel` (SAD, SATD) are already one
implementation for all six codecs; they are the largest share everywhere.

## 1. Forward transforms

| Variant | Module (lines) | Tiers | Technique, layout | Hot | Arithmetic vs family | Verdict |
|---|---|---|---|---|---|---|
| AV1 DCT 4-64 (+ raster, low, bounded) | `txfm/dct.rs` (5351, with inverse), `txfm/generic.rs`, `avx2.rs`, `avx512.rs`, `neon_forward.rs`, `simd128.rs` | S A2 A5 N W | butterfly graph generic over lane type; i32 lanes x86/wasm, i16 lanes on NEON behind an L1 guard | AV1 6.4 / 5.4; VP9 9.6 / 9.7 (VP9 DCT_DCT runs it) | libaom `fwd_txfm2d`: 12/13-bit cosines, stage shifts, saturate | **Base engine** for the cosine family; already shared by AV1, VP9 (DCT_DCT) and the HEVC/VVC staged approximation |
| HEVC/VVC staged rows | `txfm/avx2_forward.rs` (260), `neon_forward.rs` `hevc_rows`, `simd128_forward.rs` | S A2 N W (A5 inh.) | AV1 `fdct16/32` graph on i16 lanes (madd + mulhrs) | HEVC 3.6 / 5.3; VVC 4.8 / 4.7 | AV1 graph scaled by 181/2 (approximate forward, allowed) | Already shared |
| HEVC DCT 4-32, DST4 | `codec_txfm/generic.rs:713-1166`, `avx2_paired.rs` (124), `avx512.rs` | S A2 A5 N W | partial butterfly; odd half as i16-pair `pmaddwd` dot products | HEVC/VVC `staged*_bounded`, `hevc_fdct32`: ~1-1.5 | integer 64/83/36 matrix, shifts log2-1 / log2+6, clamp16 | **Merge with AV2 DCT** (same algorithm, different matrix) |
| VVC rectangular DCT-II | `codec_txfm/vvc_forward*.rs` (431) | S A2 N W | HEVC line graphs | in VVC codec_txfm 7.0 / 6.4 | identical to HEVC (nested basis) | Already shared with HEVC |
| VVC 64 forward | `vvc_tx64/` (351) | S, SIMD only for the 2x2 halve | 2x2 mean, HEVC 32-point, cosine repair | small | approximation glue | Keep (glue) |
| VVC MTS DST7/DCT8 | `vvc_mts/` (657) | S A2 N W | dense matrix, i16 pairs (`pmaddwd`, `smull+addp`, `dot_i16x8`) | small | VVdeC tables | **Family pattern for dense transforms**; AV2 typed to adopt it |
| AV2 DCT 4-64 | `av2_txfm/generic.rs` (484), `paired.rs` (375), `avx512.rs` | S A2 A5 (forward 16-64) N W | partial butterfly, odd half i16 pairs | AV2 5.8 / 7.3 | integer matrix (35 where HEVC has 36, `av2_txfm/mod.rs:16`), columns first, shifts (1,10)..(3,14) | **Merge with HEVC**: basis table, pass order, shifts, clamps as parameters |
| AV2 typed (ADST/FDST/identity) | `av2_txfm/typed.rs`, `typed_lanes.rs` (545) | S A2 N W | dense matrix, i32 broadcast multiplies | small | AVM tables | Keep tables; adopt VVC MTS i16-pair technique |
| VP9 hybrid (DCT/ADST 8,16) | `vp9_txfm.rs` (566), `vp9_txfm/adst.rs` (551) | **S only** | AV1 graphs with 14-bit `VP9_COSPI_64` | VP9 `transform::forward` (codec crate) 0.6 / 1.1 | same graph as AV1, 14-bit cosines (encoder side: rounding is free) | **Run the DCT half on AV1's SIMD lanes**; SIMD for ADST later |
| VP9 4x4 DCT/ADST | `nc-vp9/src/hybrid.rs` (235) | S | i32 | small | libvpx | Keep (small) |
| H.264 4x4, 8x8, DC Hadamard | `codec_txfm/generic.rs:1343-1567`, `avx512_h264.rs`, `neon.rs` | S A2 A5 N W | `Line` driver on i32 lanes (NEON i16) | H.264 ~3-4 | integer 1/2 basis, `+32>>6` | Keep: different basis, already on the shared `Line` driver |
| Reversible WHT4 | `reversible_wht4.rs` (65) | S | lifting | lossless only | identical | Already shared (VP9, AV1, AV2) |

## 2. Inverse transforms (encode-loop reconstruction)

| Variant | Module (lines) | Tiers | Technique | Hot | Arithmetic | Verdict |
|---|---|---|---|---|---|---|
| AV1 DCT (square, sparse, 64) and all types | `txfm/avx2_inverse.rs`, `neon_inverse.rs`, `simd128_inverse.rs`, `narrow_inverse.rs` (616), `av1_inverse/` (2697) | S A2 A5 N W | guarded i16 lanes (madd / mulhrs / vqrdmulh), i32 fallback; EOB-sparse graphs | AV1 ~1 | 12-bit cosines, clamp16 each stage | **Base engine** for cosine inverses |
| VP9 inverse 8-32 | `vp9_inverse/packed/` (1274), oracle `vp9_txfm.rs:186-275` | S A2 (A5 inh.) N W; 4x4 scalar | AV1 `dct::idct*` graphs on 8 wrapping i16 lanes | VP9 1.4 / 1.3 | 14-bit cosines, `WRAPLOW` instead of clamp, shifts 5/6 | **Merge driver into `narrow_inverse`**: stage policy (wrap vs clamp), cosine table, shifts as parameters; VP9 gains 16-lane AVX2 and sparse graphs |
| HEVC/VVC inverse DCT/DST (+ fused add) | `codec_txfm/generic.rs:1166-1287`, `inverse_add.rs`, `simd128_inverse.rs` | S A2 (A5 inh.) N W | partial butterfly, zero-extent skipping | HEVC ~1.3, VVC ~2.4 (`hevc_idct16/32_bounded`) | shift 7, clamp16, shift 12 | **Merge with AV2 inverse** (same engine) |
| AV2 DCT inverse | `av2_txfm/generic.rs:120-387`, `paired.rs:106-181` | S A2 W (paired) N (generic) | partial butterfly, corner bound | AV2 ~1.4 / 1.6 | different matrix, shifts (7,10)..(6,13), residual clamp | **Merge with HEVC**; not with AV1 (cosine butterflies) |
| VVC MTS inverse | `vvc_mts/generic.rs`, `vvc_primary.rs` | S A2 N W | dense i16 pairs | small | VVdeC | Dense-pair family |
| VVC 64 inverse | `vvc_tx64/mod.rs:194` | S (autovec) | plain sums over DCT64 | small | HEVC basis extended to 64 | Route through the HEVC partial butterfly at N=64 after the HEVC/AV2 merge |
| VVC LFNST | `codec_txfm/vvc_lfnst.rs` | S | matrix | small | normative | Keep (VVC only) |
| H.264 4x4, 8x8 | `codec_txfm/generic.rs:1357-1475`, `h264_inverse8.rs` | S A2 A5 N W | i32 `Line`, 8x8 add on i16 | H.264 2.9 / 1.7 (`h264_add_inverse_4x4`) | integer basis | Keep |

## 3. Quantise / dequantise

| Variant | Module | Tiers | Technique | Hot | Formula | Verdict |
|---|---|---|---|---|---|---|
| Shared reciprocal dead-zone quantiser (AV1, AV2, VP9) | `txfm/avx2.rs`, `avx512_quant.rs`, `neon.rs`, `simd128.rs`, `scalar.rs`; terms in `txfm/mod.rs` `QuantTerms` | S A2 A5 N W | exact reciprocal (`mul_epu32` even/odd lanes; NEON zero-group skip), vector from lane 0, DC fix-up; i16 or i32 coefficients and levels, mixed (VP9: i16 into i32) | AV1 3.1 / 2.2; AV2 1.9 / 2.1; VP9 2.1 / 2.1 | `min(((abs << input_shift) + offset) / divisor, cap)` per DC/AC terms | **Shared** |
| VP9 quantise (+ largest_magnitude) | `vp9_quant/` (now parameters and the magnitude kernels only) | S A2 A5 N W via the shared kernel (magnitude: S A2 N W) | shared kernel | VP9 2.1 / 2.1 | `((abs<<s) + floor(step/div)) / step`, DC and AC caps | **Merged** (2026-11-16): `txfm::QuantTerms` takes per-step offset, divisor and cap; VP9 states its quantiser with numerator and divisor scaled by 128 (quotients unchanged). Its own quantize kernels (5 tiers) are gone. |
| H.264 / HEVC / VVC quantise | `codec_txfm/generic.rs:2005` | S A2 (A5 inh.) N W | i32 lanes `(abs*mf + off) >> qbits` | H.264 1.7 / 1.2; HEVC/VVC ~0.8 | 32-bit multiply-shift, per-position mf | Keep apart from the reciprocal family (cheaper, normative scale) |
| HEVC/VVC `quantize_from` | `codec_txfm/generic.rs:1922` | same | same formula, one scalar scale | as above | identical to `quantize` with `mf=[scale]` | **Merge into `quantize`** |
| HEVC/VVC dequantise | `codec_txfm/generic.rs:1308-1342` | S A2 N W | clamp, multiply, shift, round | HEVC ~0.6 | Eq. 8-309 | Low-priority merge with H.264 dequant |
| AV1 dequantise | `txfm/generic.rs:826` | S A2 A5 N W | i32 lanes | AV1 0.7 | `((abs*step) & 0xFFFFFF) >> ls`, clamp16 | **Base** |
| AV2 dequantise | `nc-av2/src/transform.rs:656-700` | S (codec crate) | wrapping u32 | AV2 0.5 | AV1's plus `(+4)>>3` for Q3 steps | **Move to AV1's kernel** with a pre-round parameter |
| VP9 dequantise | `nc-vp9/src/transform.rs:289` | S (codec crate) | | small | `(abs*step) >> dq_shift` (AV1's mask never fires under VP9's cap) | **Use AV1's kernel** |

## 4. Intra prediction

No intra kernel has an AVX-512 tier. Several encoder intra paths live in codec
crates, scalar.

| Variant | Module | Tiers | Technique | Hot | Arithmetic | Verdict |
|---|---|---|---|---|---|---|
| DC / V / H (all codecs) | per codec; AV1 `intra_prediction.rs:38` (AV2 calls it) | fill | fill | AV1 `dc_value` 1.0 / 0.7 | identical base; HEVC adds a luma edge filter, VVC PDPC, H.264 chroma quadrant rules | Shared fill with codec post-passes |
| Shared 2-tap run `intra_runs::run2` | `intra_runs/` | S A2 (A5 uses A2) N W | AV1's technique: `pmaddubsw` + `pmulhrsw`, 32/16/8/4 per step (AVX2); byte `umull`/`umlal` + rounding narrow (NEON); `extmul` (SIMD128) | | `((32-w)a + wb + 16) >> 5` | **Shared** (2026-11-16): HEVC angular lines, AV1 directional, AV2 chroma, VVC chroma |
| HEVC angular | `hevc_intra/mod.rs` `angular_lines` | via `run2` | one run per line (whole-sample lines at weight 32) | HEVC 1.1 / 0.9 | 2-tap, 1/32 | **Merged** onto `run2` |
| AV1 directional | `intra_prediction/directional.rs`, `av1_intra/{avx2,neon,simd128}` | S A2 N W | `maddubs` interleaved pairs + `mulhrs`, **32 wide** (AVX2), byte `vmull` (NEON) | AV1 1.1 / 0.8 | 1/64 position, `w=(pos>>1)&31`: same 2-tap 1/32 formula | **run2 base technique** |
| AV2 chroma `run2` | `av2_intra/mod.rs` (table slot) | via `run2` | | in AV2 1.6 / 1.3 | same 2-tap | **Merged** onto `run2` (its 32-bit-accumulator and NEON copies removed) |
| VVC chroma angular | `vvc_intra/angular_generic.rs` | via `run2` | u8 main reference, PDPC on i16 lanes after the run | in VVC 7.1 / 5.9 | HEVC's 2-tap, then PDPC | **Merged** onto `run2` |
| VVC luma angular | `vvc_intra/angular_generic.rs`, `intra_runs::run4` | S A2 N W | `run4`: AVX2 two `pmaddubsw` pair sums; NEON u16 magnitude sums + saturating narrow (AV2's technique); SIMD128 i16 lanes | VVC 4.9 / 4.2 (`vvc_intra::avx2::predict`) | 4-tap cubic/Gaussian, 6-bit taps | **Shared kernel `run4`** (2026-11-16) |
| AV2 luma `run4` | `intra_prediction/av2.rs`, `av2_intra/` | S A2 N W | A2/W 32-bit accumulators; NEON byte split +/- sums | AV2 1.1 / 1.0 | 4-tap `FOUR_TAP`, 7-bit taps up to 152 | **Keep separate**: its taps exceed i8 and its positive sums (255 x 152) exceed i16, so `intra_runs::run4`'s byte-tap, 16-bit-sum form cannot hold them; its NEON already uses the family's u16 magnitude-sum technique |
| HEVC planar | `hevc_intra/lanes.rs:63` | S A2 N W | u8 refs, I16 lanes | HEVC 0.6 / 0.5 | Eq. 8-48 | **Merge with VVC planar** (HEVC layout) |
| VVC planar/DC/H/V + PDPC | `vvc_intra/generic.rs:9` | S A2 N W | **I32 lanes** over i32 refs | in VVC | planar identical to HEVC; PDPC weights fit i16 | **Merge**: HEVC's I16 layout + optional PDPC |
| AV1 smooth | `intra_prediction.rs:342`, `av1_intra/` | S A2 N W | u16 lanes | small | libaom weights, shift 8 | Keep (AV1 only) |
| AV2 smooth | `intra_prediction/av2.rs:532`, `av2_intra/generic.rs` | S A2 N W | I16 lanes | small | linear ramp + halving blend (weight = VVC PDPC formula) | Keep; share the weight helper |
| Paeth (AV1, AV2) | `intra_prediction.rs:397` | AV1 S A2 N W; **AV2 S only** (`av2.rs:240`) | one row per vector | small | identical `paeth()` | **AV2 to use AV1's SIMD** |
| Filter intra | `intra_prediction/filter.rs`, `av1_intra/` | S A2 N W | 4x2 patch per vector | AV1 0.6 | AV1 only | Keep |
| H.264 8x8 diagonals | `h264_intra/rows.rs` | S A2 N W (autovec rows) | border + rows | H.264 0.5 | avg2/avg3 on filtered refs | **Merge with VP9 rows** |
| H.264 4x4 diagonals | `nc-h264/src/intra4x4.rs:146` | **S**, per-sample `dyn Fn` | none | inside `code_intra_4x4_with` 6.7 / | same avg2/avg3 as VP9 side 4 | **Use VP9/H.264 rows module** |
| VP9 D45..D207, TM | `vp9_intra/rows.rs`, `reference.rs` | S A2 N W (autovec) | border + rows | VP9 0.9 / 0.6 | fixed 8 angles; D45 tail and above-right padding differ from H.264 | **Merge with H.264** (padding rule parameter) |
| H.264 plane, VP9 TM | `nc-h264/src/predict.rs:183`, `vp9_intra/rows.rs:66` | S | | small | unique | Keep |
| [1 2 1] reference filter | `nc-hevc/src/predict.rs:241`, `nc-vvc/src/intra.rs:280` | S (codec crates) | | in HEVC `intra_refs` 1.6 / 1.8 | identical HEVC/VVC | **Move to nc-dsp, share** |
| AV1 CfL | `nc-av1/src/cfl.rs` | S | | small | AV1 model | Keep (VVC CCLM is decoder-only) |

## 5. Motion compensation

| Variant | Module (lines) | Tiers | Technique | Hot | Arithmetic | Verdict |
|---|---|---|---|---|---|---|
| AV1 put/prep 8-tap 1/16 | `mc/av1_avx2.rs` (926), `av1_neon.rs` (429), `av1_simd128.rs` (522), `generic.rs` | S A2 N(+dot) W; A5 = whole-pel copy only | halved taps in i8, `pshufb`+`maddubs` horizontal, i16 intermediate, `pmaddwd` vertical on row pairs, `mulhrsw` rounding; 2 rows per ymm at w4/8; NEON bias-flipped `sdot` | AV1 2.3 / 2.9; AV2 1.8 / 2.5 | round0 3, round1 11 (compound 7) | **Base technique** for the 8-tap family; AV2 already uses it |
| HEVC/VVC separable (`SeparableTaps<N>`) | `mc/generic.rs:365-540`, `avx2.rs`, `avx512.rs`, `neon_dotprod.rs` (HEVC luma only), `separable_simd128.rs` (325) | S A2 A5 (32/64) N(+dot luma) W | per-tap widen + `pmullw`, i16 horizontal, `pmaddwd` vertical | HEVC 8.6 / 9.5; VVC 5.2 / 5.3 | full taps, unrounded first pass, `(v+2048)>>12`; HEVC = VVC at shared phases | **Merge onto the AV1 maddubs kernel** (tap table, halving, first-pass round/shift as parameters); `neon_dotprod` generic over the table; `separable_simd128` into `av1_simd128` |
| AV1 compound average | `mc/generic.rs:323`, `avx512.rs:109` | S A2 A5 N W | i16 | small | `(p0+p1+16)>>5` | Keep; HEVC `hevc_bi` (decoder only) could share later |
| H.264 luma/chroma | `mc/generic.rs:842-1283`, `neon_h264.rs` | S A2 A5(copy) N W | symmetric 6-tap, `j` in 32-bit, quarter-pel `avg` | H.264 1.7 / 2.1 | 6-tap + bilinear | Keep (structurally different) |
| VP9 convolve | `vp9_mc.rs` (420), `vp9_mc_simd/` (1431) | S A2 (A5 inh.) N W (+relaxed dot) | **u8 intermediate**, `maddubs` both passes; NEON `vmull_u8` magnitude split | VP9 3.5 / 4.6 | VP9 tables (odd taps), round 7 and clip after each pass | Keep (different tables and rounding); port `sdot` row primitive to NEON |
| AV1 scaled / VP9 scaled | `av1_mc/scaled/`, `vp9_mc_scaled.rs` | S A2 N W | per-sample phase | resized refs only | | Keep |

## 6. Loop filters

| Variant | Module (lines) | Tiers | Technique | Hot | Arithmetic | Verdict |
|---|---|---|---|---|---|---|
| AV1 edge (deblock family: Astra is adding the AVX-512 tier to `av1_edge`, `av2_chunk`, `lpf_bytes`; results to be added here by Astra) | `deblock/av1.rs` (628 scalar), `av1_packed.rs` (486), `lpf_bytes/` | S A2 N W (A5 inh.) | u8 lanes, 8 lines per vector, libvpx byte masks, widen only flat lanes | AV1 3.8 / 3.7 | filter4/6/8/14, hev/mask/flat/flat2 (AV1 §7.14.6) | **Merge with VP9** |
| VP9 edge | `vp9_deblock.rs` (250 scalar), `vp9_loop_filter/` (620) | S A2 (8-line) N W (16-line paired) | same `lpf_bytes` lanes and `narrow`; `masks` and `group` copied from `av1_packed` | VP9 3.5 / 3.7 | filter4/8 identical to AV1; filter16 = 15-tap p7..q7 vs AV1 13-tap filter14 | **Merge with AV1**: parameter = wide filter rule (taps, outer, centre, flat2 span) |
| AV1 paired 16-line (`*_wide`) | `lpf_bytes/*_wide.rs`, `deblock/{runs,av1_grid,av1_line,av1_tile}.rs` | S A2 N W | two 8-byte groups, dav1d 8x16 transpose | decoder only | same | **Technique to spread** to the AV1/VP9 encoder path |
| H.264 luma + chroma pair | `deblock/h264_bytes/` (1480), `h264_mb.rs` | S A2 N W | 16 u8 lanes, one per line | H.264 2.8 / 3.7 | §8.7 bS, alpha/beta, tC0; strong filter unclipped | Keep; fold `H264Bytes` into `lpf_bytes::Bytes` |
| HEVC luma | `deblock/hevc.rs` (373), `samples.rs` | S A2 N W (I16x8) | i16, 8 lines, 8x8 transpose | HEVC 1.6 / 1.8 | §8.7.2.5 strong (2tC clip) / weak | **Merge with VVC short filter** (strong clip vector parameter) |
| VVC luma | `vvc_deblock/inplace.rs` (601), `mod.rs` | S A2 N W (I16x8) | i16, transposes to p7..q7 | VVC 1.7 / 2.6 | long 7/5/3 (§8.8.3) + short filter with (3-i)tC clip; weak/decisions = HEVC | Keep the long filter; **short path merges with HEVC** |
| HEVC / VVC / H.264 chroma weak | `hevc_chroma.rs`, `vvc_deblock/chroma_edge.rs:161`, `h264_bytes/mod.rs:100` | S A2 N W | | small | all `Clip3(+-t, (4(q0-p0)+p1-q1+4)>>3)` | **One chroma delta core**, gating per codec |
| AV2 | `deblock/av2*.rs` (1448) | S A2 N W (lpf_bytes) | one lane per segment | AV2 2.8 / 2.5 | different algorithm (ramps up to 8 per side) | Keep; already on `lpf_bytes` |
| CDEF | `cdef/` (3497) | S A2 A5 N W | byte `pmaddubsw` interior | AV1 4.9 / 6.7, AV2 5.3 / 7.2 | AV1 §7.15 = AV2 8-bit | Already shared; NEON lacks `block4_pair` |
| SAO | `sao/` (1680) | S A2 A5 N W | byte apply; NEON byte stats | HEVC 2.6 / 2.4, VVC 1.1 / 1.2 | HEVC §8.7.3 = H.266 §8.8.4 | Already shared; spread NEON byte stats |
| VVC ALF | `vvc_alf/` (2988) | S A2 N(+dot) W | i16 `pmaddwd` | VVC 4.7 / 7.1 | 7x7 diamond with clipping | Keep; `vvc_alf_gram` duplicates `gram::gram_rows` (merge) |
| AV1 LR (Wiener, SGR) | `av1_restoration/` (3423) | S A2 N W | dav1d Wiener port | ~0 at Default | separable, no clip | Keep |

## 7. Metrics and reconstruction

| Kernel | Tiers | Users | Duplicate of | Verdict |
|---|---|---|---|---|
| SAD, SAD x4, grid search, SATD | S A2 A5 (partial, measured) N(+dot) W | all | none | Shared already |
| variance, sub-pixel variance, `variance_energy` | S A2 A5 N(+dot) W | all | | Shared already |
| `diff::sse_8x8_row` | S A2 N W | encoder | `variance().sse` per cell | add A5/dot or use variance |
| `convert::sse_row` (`Dsp::sse`) | S A5 W; **A2 and NEON fall back to scalar** | AV1 CfL, encoder | third SSE kernel | **Merge with diff SSE** |
| `diff::add_residual` | S A2 N W | all | `reconstruct`, `constant_add` | **Merge the three add-and-clip kernels** |
| `reconstruct` (pred + residual -> dst, saturating) | S A2 N W | AV2 decoder only | `add_residual` | merge |
| `constant_add` (DC only) | S A2 N W | AV1 inverse | VP9/VVC `add_constant` (codec crates, scalar) | route codec DC shortcuts through it |
| `block_sums` vs `sample::moments_8x8_row` | | encoder | same (sum, sum of squares) | merge |
| `block_copy::avx512` | installed nowhere in `Dsp::avx512` | | | install or fix the module doc |

Scalar add-and-clip and SSE loops written in codec crates (to move onto the
shared kernels): `nc-hevc/src/backend.rs:2481` (`reconstruct<N>`),
`nc-vp9/src/recon.rs:326-346`, `nc-vvc/src/recon.rs:1109,1179`,
`nc-av1/src/cdef.rs:134,215` (same SSE written twice), `nc-av1/src/cfl.rs:107`,
`nc-h264/src/intra4x4.rs:351`, `nc-vvc/src/isp.rs:404`, `nc-vvc/src/mts.rs:264`.

## 8. Codec-crate hot loops (no explicit SIMD: codec crates forbid unsafe)

Bodycam / crowd, Default, self instructions. Coordinate quantise and scan
work with the trellis agent (it owns costing, trellis, RDOQ); this work moves
kernels only.

| Loop | Share | Pattern | Shared kernel to use or build |
|---|---|---|---|
| `nc_vvc::recon::quantize_rect` | 3.6 / 4.5 | quantise | `codec_txfm` `quantize`/`quantize_from` (check why not used) |
| `nc_av2::block_coder::Quantizer::code` / `::reconstruct` | 3.8 / 4.4, 1.8 / 1.9 | quantise, dequantise + inverse | AV1 `quantize_from_wide`, AV1 `dequantize` |
| `nc_av2::transform::{dequantize,reconstruct_into}` | 0.5, 0.8 | dequantise | AV1 `dequantize` with pre-round |
| `nc_vp9::recon::ResidualBuffers::add_residual` | 2.8 / 2.5 | add and clip | `Dsp::add_residual` |
| `nc_vp9::transform::{quantize,forward}` | 0.7 / 0.5, 1.1 / 0.6 | quantise, hybrid forward | `vp9_quant`/AV1 quantiser; AV1 forward lanes |
| `nc_vvc::coeff::walk_sized`, `last_significant` | 3.6 / 5.5, 1.6 | scan statistics | new shared kernel: last nonzero in scan, nonzero count, level-magnitude classes, fed by codec scan tables |
| `nc_av2::coeff::levels::base_context` | 1.4 / 1.8 | neighbour magnitude context | same scan-statistics kernel (context tables stay in the codec) |
| `nc_hevc::backend::intra_refs`, `nc_hevc::predict::oriented` | 1.6 / 1.8, 1.5 / 1.9 | reference build/substitute/filter, angular | `hevc_intra` angular kernel + shared [1 2 1] filter (section 4) |
| `nc_av2::intra_edges::predict` | 1.8 / 2.3 | edge preparation + directional | shared run2/run4 (section 4) |
| `nc_vvc::alf::Analysis::decide`, AV1 `apply_cdef`, SAO stats | 2.3, 2.4 / 1.3, 2.1 | filter search statistics | `gram::gram_rows` as the shared start |

## 9. Merge plan (hottest and cleanest first)

| # | Merge | Codecs (hot share) | Status |
|---|---|---|---|
| 1 | VP9 quantise into the AV1 quantiser | VP9 2.1 | landed (see log) |
| 2 | AV1/VP9 packed loop filter: one driver, wide-filter rule as a parameter | AV1 3.8, VP9 3.5 | owned by Astra (deblock family AVX-512 tier, brief `work/briefs/astra-avx512-deblock-2026-11-16.md`); Astra adds its results here |
| 3 | HEVC/VVC MC onto the AV1 maddubs/sdot kernels | HEVC 9, VVC 5 | planned |
| 4 | Intra `run2` (HEVC, AV1, AV2, VVC chroma) and `run4` (VVC luma; AV2 stays separate) | VVC 6-7, AV1 2-3, AV2 2-3, HEVC 1-3 | landed (see log) |
| 5 | VP9 inverse driver into `narrow_inverse` | VP9 1.3 | planned |
| 6 | HEVC/AV2 partial-butterfly engine (forward + inverse, N up to 64) | AV2 6-7, HEVC/VVC 2-4 | planned |
| 7 | HEVC `quantize_from` into `quantize`; AV2/VP9 dequant onto AV1's | small | planned |
| 8 | Add-and-clip and SSE kernels to one each; codec-crate loops onto them | VP9 2.8 and others | planned |
| 9 | HEVC/VVC short luma deblock and the chroma delta core | HEVC 1.7, VVC 2-3 | planned |
| 10 | VVC/HEVC planar, H.264/VP9 diagonal rows, AV2 Paeth onto AV1 | small | planned |

## Stays separate (and why)

- **H.264 transforms, quantiser, MC, luma deblock:** a different integer basis,
  32-bit multiply-shift quantiser, 6-tap MC with intermediate `j`, and §8.7's
  unclipped strong filter. They already use the shared `Line` driver and
  byte-lane layouts.
- **AV1 cosine butterflies vs HEVC/AV2 integer matrices:** per-stage clamped
  12-bit cosine graphs against matrix dot products; not the same arithmetic.
- **VP9 MC:** different tables (odd taps, so AV1's halving does not apply) and
  a u8 intermediate with rounding after each pass.
- **AV2 deblock:** a different algorithm (ramps), already on `lpf_bytes`.
- **VVC long deblock filters, ALF, LFNST, MTS tables; AV1 filter intra, smooth,
  LR:** single-codec tools.
- **CDEF, SAO, SAD/SATD/variance:** already one implementation each.
