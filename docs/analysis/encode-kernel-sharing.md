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
| AV1 DCT 4-64 (+ raster, low, bounded) | `txfm/dct.rs` (5351, with inverse), `txfm/generic.rs`, `avx2.rs`, `avx512.rs`, `avx512_forward.rs`, `neon_forward.rs`, `simd128.rs` | S A2 A5 N W | shared butterfly graph; ICL full32/low16-square and NEON use guarded i16 rows with wide fallback; ICL16 retains its original dword graph | AV1 6.4 / 5.4; VP9 9.6 / 9.7 (VP9 DCT_DCT runs it) | libaom `fwd_txfm2d`: 12/13-bit cosines, stage shifts, saturate | **Base engine** for the cosine family; already shared by AV1, VP9 (DCT_DCT) and the HEVC/VVC staged approximation |
| HEVC/VVC staged rows | `txfm/avx2_forward.rs`, `avx512_forward.rs`, `neon_forward.rs` `hevc_rows`, `simd128_forward.rs` | S A2 A5 N W | shared AV1 `fdct16/32` graph on i16 lanes (madd + mulhrs); ICL32 carries 32 rows per ZMM | HEVC 3.6 / 5.3; VVC 4.8 / 4.7 | AV1 graph scaled by 181/2 (approximate forward, allowed) | Already shared; sixteen rows keep their natural sixteen-word layout |
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

ICL forward extension (Astra, 2026-11-16): `txfm/avx512_forward.rs` supplies
32-word columns to the shared full32 and low16-square graphs. Full32 uses
a register transpose; low16-square transposes only the retained 16x32
rectangle through the existing sixteen-word adapter. The existing L1 limit
of 32000 admits packed rows; larger rows keep the exact wide graph in cold
helpers. Original 16-point kernels and the shared SIMD facade are retained.
HEVC/VVC staged32 shares the word graph, existing scale and exact DC.
Layout reference: SVT-AV1 v4.3.0, BSD-2 + AOM patent license
(`THIRD_PARTY.md`). Dependent VNNI accumulation in dense HEVC lines and the
provisional intra widening were rejected after measured losses.
The selected bundle improves Zen 5 throughput by 1.41% AV1 Default,
1.49% AV1 Fast, 4.29% VP9, 1.30% HEVC and 2.68% VVC on three public CTC
clips; AV1 Faster/Veryfast are unresolved. The unchanged-kernel H.264
control is 0.96% slower, also recorded. These are fixed-public-clip timings,
not general content-population estimates. Full verification, trial history,
limitations and ns/call: `docs/handoff/astra-avx512-encode.md`.

ICL linear ALF (Astra, 2026-11-16): the existing paired-tap layout now
carries 32 pixels, reusing coefficient broadcasts across four rows and
retaining AVX2 tails. Quiet Zen 5 64x4 row: 119.12 ->64.76 ns; VVC Default
throughput +0.62% on the same public three-clip pair. Other codec controls
and the H.264 timing loss are preserved; SDE ICL H.264 executed work differs
by only +0.0016%, with identical hot encoder/DSP self-counts. See
`docs/handoff/astra-icl-alf.md` for limits, checks and evidence.

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
| VVC planar/DC/H/V + PDPC | `vvc_intra/generic.rs` | S A2 N W | planar: the shared HEVC planar kernel (u8 refs, i16 lanes); DC/H/V fills; PDPC one i16 pass over the samples its weights reach | in VVC | planar identical to HEVC (eq. 8-48) | **Merged** (2026-11-16) onto `hevc_intra::lanes::planar` |
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
| AV1 put/prep 8-tap 1/16 | `mc/av1_avx2.rs`, `av1_avx512.rs` (rows of 32), `av1_neon.rs` (429), `av1_simd128.rs` (522), `generic.rs` | S A2 A5 (32/64 wide, put, prep, mixed) N(+dot) W | halved taps in i8, `pshufb`+`maddubs` horizontal, i16 intermediate, `pmaddwd` vertical on row pairs, `mulhrsw` rounding; 2 rows per ymm at w4/8; NEON bias-flipped `sdot` | AV1 2.3 / 2.9; AV2 1.8 / 2.5 | round0 3, round1 11 (compound 7) | **Base technique** for the 8-tap family; AV2 already uses it |
| HEVC/VVC separable (`SeparableTaps<N>`) | x86: `mc/av1_avx2.rs` `separable` with `av1_avx512` rows (shared with AV1); others `mc/generic.rs` (NEON), `neon_dotprod.rs` (HEVC luma only), `separable_simd128.rs` | S A2 A5 (shared, 32/64 wide, luma and chroma) N(+dot luma) W | AVX2: AV1's `vpshufb` + `vpmaddubsw` horizontal, `vpmaddwd` vertical, two rows per step at 4/8 wide; 4-tap chroma on two tap pairs | HEVC 8.6 / 9.5; VVC 5.2 / 5.3 | full taps (fit i8), unrounded first pass, `(v+2048)>>12`; HEVC = VVC at shared phases | **AVX2 merged** (2026-11-16) onto the AV1 kernel with a `Rounding` scheme (AV1: halved taps and its roundings; HEVC/VVC: full taps). **AVX-512 merged** (2026-11-16): the same passes on rows of 32 outputs in four 128-bit lanes (`av1_avx512`, `WidePasses`); the old 32-word-lane `generic::hevc` AVX-512 path is gone. **SIMD128 merged** (2026-11-16): `separable_simd128` folded into `av1_simd128` (rounding schemes shared in `mc/rounding.rs`); node on the box, min of two runs, new/base: HEVC luma vertical 0.74-0.90 (AV1's rolling row window), luma 2-D 0.97-1.04, luma horizontal 1.00-1.09, chroma 2-D 1.04-1.07, sum of the HEVC rows 0.97 (unchanged AV1 rows 0.97-1.04, the noise). Still to do: `neon_dotprod` generic over the table |
| AV1 compound average | `mc/generic.rs:323`, `avx512.rs:109` | S A2 A5 N W | i16 | small | `(p0+p1+16)>>5` | Keep; HEVC `hevc_bi` (decoder only) could share later |
| H.264 luma/chroma | `mc/generic.rs:842-1283`, `neon_h264.rs` | S A2 A5(copy) N W | symmetric 6-tap, `j` in 32-bit, quarter-pel `avg` | H.264 1.7 / 2.1 | 6-tap + bilinear | Keep (structurally different) |
| VP9 convolve | `vp9_mc.rs` (420), `vp9_mc_simd/` (1431) | S A2 (A5 inh.) N W (+relaxed dot) | **u8 intermediate**, `maddubs` both passes; NEON `vmull_u8` magnitude split | VP9 3.5 / 4.6 | VP9 tables (odd taps), round 7 and clip after each pass | Keep (different tables and rounding); port `sdot` row primitive to NEON |
| AV1 scaled / VP9 scaled | `av1_mc/scaled/`, `vp9_mc_scaled.rs` | S A2 N W | per-sample phase | resized refs only | | Keep |

## 6. Loop filters

| Variant | Module (lines) | Tiers | Technique | Hot | Arithmetic | Verdict |
|---|---|---|---|---|---|---|
| AV1 edge | `deblock/av1.rs` (628 scalar), `av1_packed.rs` (486), `lpf_bytes/` | S A2 A5(ICL/GFNI) N W | u8 lanes, 8/16/32 independent lines, libvpx byte masks, widen only flat lanes | AV1 3.8 / 3.7 | filter4/6/8/14, hev/mask/flat/flat2 (AV1 §7.14.6) | **Merge with VP9** |
| VP9 edge | `vp9_deblock.rs` (250 scalar), `vp9_loop_filter/` (620) | S A2 (8-line) A5(ICL/GFNI, 32-line) N W (16-line paired) | same `lpf_bytes` lanes and `narrow`; `masks` and `group` copied from `av1_packed` | VP9 3.5 / 3.7 | filter4/8 identical to AV1; filter16 = 15-tap p7..q7 vs AV1 13-tap filter14 | **Merge with AV1**: parameter = wide filter rule (taps, outer, centre, flat2 span) |
| AV1/VP9 grouped byte lanes | `lpf_bytes/{*_wide,gfni,avx512}.rs`, `deblock/{runs,av1_grid,av1_line,av1_tile}.rs` | S A2 A5(ICL/GFNI) N W | paired/four 8-byte groups, shared transpose and tile IO | encoder and decoder | same | Shared AVX-512 layout for AV1/VP9; GFNI covers short tails |
| H.264 luma + chroma pair | `deblock/h264_bytes/` (1480), `h264_mb.rs` | S A2 N W | 16 u8 lanes, one per line | H.264 2.8 / 3.7 | §8.7 bS, alpha/beta, tC0; strong filter unclipped | Keep; fold `H264Bytes` into `lpf_bytes::Bytes` |
| HEVC luma | `deblock/hevc.rs` (373), `samples.rs` | S A2 N W (I16x8) | i16, 8 lines, 8x8 transpose | HEVC 1.6 / 1.8 | §8.7.2.5 strong (2tC clip) / weak | **Merged** (2026-11-16): strong and normal filters in `deblock/short_luma.rs`, shared with VVC; HEVC passes the ±2tC clip vector |
| VVC luma | `vvc_deblock/inplace.rs` (601), `mod.rs` | S A2 N W (I16x8) | i16, transposes to p7..q7 | VVC 1.7 / 2.6 | long 7/5/3 (§8.8.3) + short filter with (3-i)tC clip; weak/decisions = HEVC | Keep the long filter; short filters **merged** with HEVC (`short_luma`, 2026-11-16; VVC passes ±3tC/±2tC/±tC) |
| HEVC / VVC / H.264 chroma weak | `hevc_chroma.rs`, `vvc_deblock/chroma_edge.rs:161`, `h264_bytes/mod.rs:100` | S A2 N W | | small | all `Clip3(+-t, (4(q0-p0)+p1-q1+4)>>3)` | **Stays separate** (2026-11-16): the formula is six vector operations, and the three copies sit on three lane families (HEVC `lpf_bytes` words on p1..q1 byte tiles, VVC `I16FilterLanes` on p3..q3 with `shuffle_bytes` pair decisions, H.264 saturated bytes). A shared core needs a new trait with an impl per ISA lane type (about 40 lines) to remove about 10 |
| AV2 | `deblock/av2*.rs` (1448) | S A2 A5(ICL/GFNI) N W (lpf_bytes) | eight segment decision lanes; A5 batches four ramp rows | AV2 2.8 / 2.5 | different algorithm (ramps up to 8 per side) | Keep; already on `lpf_bytes` |
| CDEF | `cdef/` (3497) | S A2 A5 N W | byte constraints; AVX2 paired MAC/Q15 rounding, ICL register tables/GFNI/VNNI | AV1 4.9 / 6.7, AV2 5.3 / 7.2 | AV1 §7.15 = AV2 8-bit | Already shared; NEON `block4_pair` added (2026-11-16, as SIMD128's: two interior 4x4 blocks per call) |
| SAO | `sao/` (1680) | S A2 A5 N W | byte apply; NEON byte stats | HEVC 2.6 / 2.4, VVC 1.1 / 1.2 | HEVC §8.7.3 = H.266 §8.8.4 | Already shared; spread NEON byte stats |
| VVC ALF | `vvc_alf/` | S A2 A5(linear row) N(+dot) W | i16 paired `pmaddwd`, ICL 32 independent pixels; other statistics inherit A2 | VVC 4.7 / 7.1 | 7x7 diamond with clipping | Keep; `vvc_alf_gram` duplicates `gram::gram_rows` (merge) |
| AV1 LR (Wiener, SGR) | `av1_restoration/` (3423) | S A2 N W | dav1d Wiener port | ~0 at Default | separable, no clip | Keep |

AVX-512 implementation (Astra, 2026-11-16): AV1/AV2/VP9 share the grouped
`lpf_bytes` backend and tile IO. The owner requires the full Ice Lake/GFNI
feature contract globally, with older AVX-512 CPUs selecting AVX2. Equations
and all existing ISA tiers remain. Measurements and validation:
[`astra-avx512-deblock`](../handoff/astra-avx512-deblock.md). The separate
AV1/VP9 outer-filter equation unification remains the sharing agent's work.

## 7. Metrics and reconstruction

| Kernel | Tiers | Users | Duplicate of | Verdict |
|---|---|---|---|---|
| SAD, SAD x4, grid search, SATD | S A2 A5 (partial, measured) N(+dot) W | all | none | Shared already |
| variance, sub-pixel variance, `variance_energy` | S A2 A5 N(+dot) W | all | | Shared already |
| `diff::sse_8x8_row` | S A2 N W | encoder; `Dsp::sse` (2026-11-16) | `variance().sse` per cell | **Shared** SSE kernel: `Dsp::sse` (planes) runs it over whole 8x8 cells and sums the edges, so `convert::sse_row` (S A5 W, A2/NEON scalar) is gone |
| `diff::reconstruct` (`Dsp::add_residual`, `Dsp::reconstruct_i16`) | S A2 N W | all; VP9 reconstruction (was a scalar loop in `nc-vp9/src/recon.rs`); AV2 encoder and decoder | `constant_add` | **Merged** (2026-11-16): one width-class kernel, prediction in place or from a block, residual rows or a constant, 16-bit saturating add (`I16Lanes::add_sat`); the `reconstruct` module (8 lanes a step at every width) is gone, so AV2 gains 16/32-lane AVX2 rows |
| `constant_add` (DC only), `Dsp::add_constant` | S A2 N W | AV1 inverse; VP9 and VVC DC-only blocks (2026-11-16) | | **Shared**: VP9's and VVC's scalar `add_constant` removed |
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
| `nc_av1::backend::quantized_eob` | 1.6 (crowd) | end of block | **Done** (2026-11-16): shared `Dsp::end_of_block` over the codec's `eob_table` (scan index + 1 per raster position; libaom/SVT `iscan`); AVX2 16, AVX-512 32, NEON/SIMD128 8 levels a step. AV1 Default -1.8 % instructions |
| `nc_encoder::intra_estimate::estimate` | 1.0 (AV1 crowd) | sums, squares, four directional SSEs over every other row and column | **Done** (2026-11-16): `Dsp::intra_estimate_stats` (even samples by low-byte mask or `uzp1`; `vpmaddwd`, `umull`+`uadalp`, `dot`) |
| `nc_analysis::preanalysis::analyze` finishing loop | 1.05M instructions/P, lowlight | mean/variance/SAD/static classification across blocks | **Moved** (2026-11-16): `Dsp::analysis_stats_row`, one graph across 8 AVX2 / 16 ICL / 4 NEON or SIMD128 blocks. Caller + kernel 1,049,121 ->333,694 Ir/P (-68.2%), exact streams; [handoff](../handoff/astra-preanalysis-stats.md) |
| `nc_vvc::coeff::walk_sized`, `last_significant` | 3.6 / 5.5, 1.6 | scan statistics | **Done** (2026-11-16): AV2's end of block (`5dd9da00b`) and VVC `last_significant` run `Dsp::end_of_block_wide` over end-of-block tables of their scans (VVC: one per grouped rectangle, `scan::rect_eob`). Still open: nonzero count, level-magnitude classes |
| `nc_av2::coeff::levels::base_context` | 1.4 / 1.8 | neighbour magnitude context | same scan-statistics kernel (context tables stay in the codec) |
| `nc_hevc::backend::intra_refs`, `nc_hevc::predict::oriented` | 1.6 / 1.8, 1.5 / 1.9 | reference build/substitute/filter, angular | `hevc_intra` angular kernel + shared [1 2 1] filter (section 4) |
| `nc_av2::intra_edges::predict` | 1.8 / 2.3 | edge preparation + directional | shared run2/run4 (section 4) |
| `nc_vvc::alf::Analysis::decide`, AV1 `apply_cdef`, SAO stats | 2.3, 2.4 / 1.3, 2.1 | filter search statistics | `gram::gram_rows` as the shared start |

## 9. Merge plan (hottest and cleanest first)

| # | Merge | Codecs (hot share) | Status |
|---|---|---|---|
| 1 | VP9 quantise into the AV1 quantiser | VP9 2.1 | landed (see log) |
| 2 | AV1/VP9 packed loop filter: one driver, wide-filter rule as a parameter | AV1 3.8, VP9 3.5 | owned by Astra (deblock family AVX-512 tier, brief `work/briefs/astra-avx512-deblock-2026-11-16.md`); Astra adds its results here |
| 3 | HEVC/VVC MC onto the AV1 maddubs/sdot kernels | HEVC 9, VVC 5 | AVX2 and AVX-512 (rows of 32) landed; NEON dotprod and SIMD128 to do |
| 4 | Intra `run2` (HEVC, AV1, AV2, VVC chroma) and `run4` (VVC luma; AV2 stays separate) | VVC 6-7, AV1 2-3, AV2 2-3, HEVC 1-3 | landed (see log) |
| 5 | VP9 inverse driver into `narrow_inverse` | VP9 1.3 | planned |
| 6 | HEVC/AV2 partial-butterfly engine (forward + inverse, N up to 64) | AV2 6-7, HEVC/VVC 2-4 | planned |
| 7 | HEVC `quantize_from` into `quantize`; AV2/VP9 dequant onto AV1's | small | planned |
| 8 | Add-and-clip and SSE kernels to one each; codec-crate loops onto them | VP9 2.8 and others | VP9/VVC reconstruction onto `add_residual`/`add_constant` landed; `reconstruct` merged into `add_residual`, plane SSE onto `sse_8x8_row` (2026-11-16) |
| 9 | HEVC/VVC short luma deblock and the chroma delta core | HEVC 1.7, VVC 2-3 | luma short filters landed (see log); chroma core stays separate (section 6) |
| 10 | VVC/HEVC planar (landed), H.264/VP9 diagonal rows, AV2 Paeth onto AV1 | small | planar landed; rest planned |

## Handover state (2026-11-16 14:35, after `882703533`)

Landed from the third brief: item 6 luma (`b7ae5102c`, HEVC/VVC short luma
filters in `deblock/short_luma.rs`); the chroma delta core stays separate
(section 6 row). Item 8 step 1 landed (below); item 7 is not
started.

- **Item 8, step 1 (landed).** AV2's level engine (`odd_matrix`, `dot`,
  `inverse_level`, `forward_level`, `inverse_dct`, `forward_dct`) moved to
  `codec_txfm/butterfly.rs`, generic over a `DctBasis` (DC weight and each
  level's odd matrix; `Av2Basis` from `DCT_BASIS`, `HevcBasis` from
  `DCT32`). HEVC's `hevc_idct*`, `*_bounded`, `hevc_fdct*`, `*_low` and
  `hevc_odd_forward` (about 330 lines) became `hevc_fdct` / `hevc_idct`
  calls; a unit test checks them against the HEVC matrix.
  - The NEON regression of the parked commit (sparse inverses +10 to
    +14 %) was the engine's `quad_rounded` padding: a level with 1 or 2
    live odd inputs ran 2 or 4 products per output. Without it (Jetson
    `nc-checkasm --bench --isa neon`, min of two interleaved runs against
    `98b609f22`): add_low4 8x8 -2.1 %, 16x16 -11.2 %, 32x32 -8.2 %;
    split_dc 16x16 -3.5 %, 32x32 -8.4 %; full inverses 8x8 -6.9 %, 32x32
    -5.0 %; forward 16x16 -5.1 %. Slower: 4x4 forward +4.0 % (+1.1 ns),
    4x4 inverse +2.3 %, add_low4/full 4x4 +4.1 % (+2 ns), forward 32x32
    +1.3 %. AV2 within noise (-0.8 to +1.0 %; the 4x4 forward +2.3 % in
    both variants, a kernel the change does not touch).
  - Whole program, Jetson (`jperf`, crowd 720p, 30 frames, chart QPs,
    cycles, min of 3): HEVC encode +0.09 % (instructions -0.22 %), VVC
    -0.76 %, AV2 -0.56 %; HEVC decode (60 frames, QP 36, min of 5)
    -0.54 %. x86 callgrind in the merge log. Streams identical
    (`preset_identity` HEVC/VVC/AV2, 90/90); hevc-v1 147/147.
  - Method trap: syncing a source tree to the Jetson with `rsync -a` (or
    `git archive`) keeps old mtimes, so cargo does not rebuild and every
    "variant" is the same binary. Sync with `rsync -rlp --checksum`
    (changed files get the current time) and compare the binaries' hashes.
  - Later steps: the paired i16 engines (HEVC `avx2_paired.rs` and
    `simd128_inverse`, AV2 `paired.rs`) onto one basis-parameterised pass.
- **Item 7, VP9 inverse into `narrow_inverse`.** Not started. Differences
  read from the code (`vp9_inverse/packed/mod.rs` vs
  `txfm/narrow_inverse.rs`): coefficient layout (VP9 row-major `i16`, AV1
  column-major `coeffs[c * coded_h + r]`), no L1 guard (VP9's wrapping is
  the arithmetic itself, so the narrow path is always exact), no row shift
  and a final shift of 5 (8x8) or 6, VP9's own `iadst8/16` on its
  `Products` trait (libvpx rotations) instead of AV1's ADST graphs, a
  separate non-adding `transform` entry point and `dc_add`. VP9 already has
  EOB-sparse DCT corners (`dct_add` with `K`); the gain left is 16-lane
  AVX2.
  - Read 2026-11-16 (not started; no code): the lane types cannot be
    shared, only the driver. AV1's `Narrow` is exact only inside the L1
    guard: it narrows with a saturating pack (`packs_epi32`) and pre-adds
    equal-weight pairs in 16 bits before `pmulhrsw`; VP9 must wrap like
    libvpx's `WRAPLOW` for every input (truncating pack after masking, and
    `(a + b) * cospi_16_64` in 32 bits), with 14-bit cosines (`COSINE`),
    where AV1's lanes hard-code 12. So VP9 keeps its `Lanes`/`Products`
    types and implements `NarrowLanes` (loads, `store_groups`, `transpose`,
    `add_to_pixels`, const `round_shift`; a norm type with an always-true
    `within_limit`). The driver needs a codec parameter for: the load
    (AV1 column-major lanes load directly; VP9 row-major needs `load_tile`'s
    transpose), the row shift (AV1 1/2, VP9 none), the column shift (AV1 4,
    VP9 5/6), the guard and wide fallback (AV1 only), the 2:1 prescale (AV1
    only) and the 1-D kernels (VP9 `corner_dct` with `Sparse` and its
    `adst`). `dc_add` and the non-adding `transform` stay VP9's.
  - Expected gain: x86 only (a 16-lane AVX2 VP9 type, new, with the
    wrapping `pmaddwd` products of `packed/avx2.rs` on 256-bit halves);
    NEON and SIMD128 stay at 8 lanes, so the Jetson gains nothing but the
    shared loops. Measure VP9 decode share first (Jetson and x86) to size
    it before writing the 16-lane type.

Earlier plan notes (2026-11-16 12:10):

- **Item 6, HEVC/VVC short luma deblock and the chroma delta core.** Luma
  landed (`b7ae5102c`); the chroma core stays separate (section 6 row).
  - Luma: HEVC `deblock/hevc.rs` `apply` (per-lane tC vector, decision masks,
    strong only when a segment chose it) and VVC
    `vvc_deblock/inplace.rs` `strong` / `weak` (scalar tC per edge) are the
    same §8.7.2.5.6/.7 arithmetic. The weak filter is equal (VVC's extra
    `clip(…, ±tC)` and `clip(…, ±tC/2)` are no-ops, since |delta| <= tC and
    the p1 correction is already clamped to ±tC/2). The strong filter differs
    only in the clip: HEVC `±2tC` on all three samples, VVC `±(3 - i)·tC`.
  - Plan: one `strong_side` / `weak` pair on `I16FilterLanes` taking a
    per-position clip vector and a tC vector (splatted for VVC); VVC
    `short_edge` and `short_filters` and HEVC `apply` call it. Measure
    HEVC and VVC Default (callgrind, release); both are hot (HEVC 1.7,
    VVC 2-3).
  - Chroma: `deblock/hevc_chroma.rs` (on `lpf_bytes` `Bytes::W` words),
    `vvc_deblock/chroma_edge.rs` `simd` (`I16FilterLanes`) and H.264
    `h264_bytes` all compute `Clip3(-tC, tC, (4(q0-p0) + p1 - q1 + 4) >> 3)`.
    They sit on different lane traits, so the shared core needs a small
    trait both lane types implement (sub, mul, add, shr/sra, min, max), or
    VVC chroma moves onto `lpf_bytes` words like HEVC's.
- **Item 7, VP9 inverse driver into `txfm/narrow_inverse.rs`.** Not
  started. `vp9_inverse/packed/` runs AV1's `dct::idct*` graphs on 8
  wrapping i16 lanes. The differences to parameterise: stage policy (VP9
  `WRAPLOW` wrap vs AV1's per-stage clamp, guarded by `l1_limit`), the
  14-bit cosine table, and the shifts (VP9 5/6). Gains: 16-lane AVX2 and
  the EOB-sparse graphs for VP9.
- **Item 8, one partial-butterfly engine for HEVC and AV2.** Step 1 landed (above).
  The sources are `codec_txfm/generic.rs:713-1287` (HEVC DCT and inverse)
  and `av2_txfm/generic.rs`, `paired.rs`. The differences are the basis
  table (35 vs 36), pass order (AV2 columns first), shifts and clamps. AV2
  has the larger share (5.8 / 7.3).

Method notes for the next agent:
- Callgrind runs locally only (the clips stay on the PC): build release
  binaries on the box (`nc-remote run --fetch target/release/nanocodec
  --fetch-to DIR`), then `valgrind --tool=callgrind`. The previous runs
  used `NC_DSP_ISA=avx2`, crowd 720p, 30 frames, chart QPs
  (H.264 43, HEVC 46, VVC 44, VP9/AV1 214, AV2 176).
- Callgrind cannot run AVX-512. For AVX-512 kernels, use
  `nc-checkasm --bench --isa avx2|avx512` on the box. Check stream identity
  across ISAs by running one binary with `NC_DSP_ISA=avx2` and with
  `avx512` on the box's public CTC clips (`~/nanocodec-data/ctc-sources`).
- SIMD128: the box has node but no `wasm32-wasip1` target. Build
  `nc-checkasm.wasm` on the PC, put it in the worktree (untracked files
  sync), and run `--bench --isa simd128` there. PC timings were unusable
  under load.
- `git stash` then `land.sh` lands only what is committed, and
  `git commit` takes everything staged, including earlier `git rm`s.

## Merge log (callgrind, release builds, Default, crowd 720p, 30 frames, chart QPs)

| Change | H.264 | HEVC | VVC | VP9 | AV1 | AV2 |
|---|---|---|---|---|---|---|
| Quantiser, intra runs, VVC planar, reconstruction adds, fixes (vs `d155dee64`) | 0.00 | +0.77 | -1.10 | +0.03 | +0.15 | -0.07 |
| + HEVC/VVC MC on the AV1 AVX2 kernel, run2 fixed lengths, 256-bit planar (vs `d155dee64`) | -0.05 | -6.25 | -3.93 | +0.02 | +0.12 | +0.05 |
| + end of block and intra estimate kernels (vs the line above) | +0.16 (`intra_or_inter` no longer inlined) | -0.35 | -0.04 | +0.02 | -2.02 | 0.00 |
| Same, rebased on `3df562c10` (whose AV1 end of block was a codec-local auto-vectorised pass; vs that head) | +0.17 | -0.37 | | | -0.32 | |
| MC AVX-512 rows, VVC last significant on the shared kernel, one reconstruction kernel, plane SSE on `sse_8x8_row` (vs `6fabd2d77`; an AVX2 run, so the AVX-512 rows do not count here) | 0.00 | 0.00 | -1.12 | +0.06 | +0.01 | -0.84 |

| HEVC/VVC short luma filters in `deblock/short_luma.rs` (vs `7b3c81724`) | | +0.01 | -0.06 | | | |
| HEVC 1-D DCTs on the AV2 partial-butterfly engine, no quad padding (vs `98b609f22`) | | -0.08 | -0.24 | | | -0.01 |
| SVT packed-byte 4/8-wide variance/energy/block sums (`2975ad0fa` vs immediate parent `d77fd2d57`; memory-operand row inserts, not the rejected scalar-gather widening trial) | | -0.243 | -0.073 | | -0.326 | |
| Preanalysis finishing statistics across blocks (`f28e49ca3` vs `b7bdce7f6`) | | -0.668 | -0.359 | | -0.534 | |

Streams byte-identical at every step (`preset_identity`, all six codecs;
the AVX-512 rows also AVX2 against AVX-512 on the Ice Lake box).

Not moved into kernels (2026-11-16), with the reason: `nc_analysis::noise::update` is a
selection over the block grid (bucket counts and `select_nth`), not pixel arithmetic;
`zero_residual` already calls shared row kernels. The preanalysis finishing
loop has since moved to `Dsp::analysis_stats_row` (above); its pixel moments
and SAD row kernels were already shared.

## Hot kernels still on 128-bit lanes (AVX2 tier)

Zen 5 rewards 256/512-bit work (owner, 2026-11-16: SVT about 44 % vector
uops, mostly 256/512-bit, against our about 27 %). Read from the code at
the merges above; hot shares are section 0's.

| Kernel | Lanes | Hot | Note |
|---|---|---|---|
| H.264 deblock (`deblock/h264_bytes/x86.rs`) | 128-bit bytes (SSE4.1) | H.264 2.8 / 3.7 | one lane per line of a 16-line edge |
| AV1/VP9 deblock (`lpf_bytes/avx2.rs`, `deblock/avx2.rs` `av1_edge`) | 128-bit (8 lines) | AV1 3.8, VP9 3.5 | Astra's AVX-512 work; the 16-line paired `*_wide` lanes exist (256-bit) for the decoder |
| HEVC luma deblock, VVC deblock (`deblock/hevc.rs`, `vvc_deblock/`) | I16x8 | HEVC 1.7, VVC 2-3 | |
| HEVC/VVC/AV1/AV2 intra runs of 16 (`intra_runs` `step16`) | 128-bit | intra | 32-wide runs are 256-bit |
| `constant_add` | 128-bit bytes | small | 32/64-wide blocks could take 256-bit |
| H.264 inverse 8x8 add (`codec_txfm/avx2.rs` `I16x8`) | I16x8 | H.264 | |
| Variance 4/8 wide (`diff/x86_narrow.rs`) | 16 word lanes AVX2 (2/4 rows), 32 ICL (4/8 rows) | all codecs | SVT packed-byte source/prediction pairs, one `pmaddubsw` difference, bounded i16 sums and paired square products; ICL VNNI square accumulation. Variance, energy and block sums share one graph. The earlier scalar-gather widening trial remains rejected; this port uses memory-operand row loads/inserts. Residual and narrow sub-pixel paths keep their existing layout. |
| AV1/HEVC/VVC MC put (`mc/av1_avx2.rs`) | 256-bit; 512-bit rows of 32 at widths 32 and 64 (2026-11-16) | AV1 2-3, HEVC 9, VVC 5 | Ice Lake (box, `nc-checkasm --bench`, ns/call AVX2 -> AVX-512): AV1 2-D 32x32 257 -> 187, 64x64 959 -> 688; HEVC luma 2-D 32x32 300 -> 224 (the old AVX-512 path took 410), 64x64 1149 -> 829 (old 1555); horizontal-only 3-9 % |
| Sub-pixel variance 16 wide | already one row per 256-bit vector (AVX2 `I16x16`) and two rows per 512-bit vector (AVX-512) | | nothing 128-bit to widen at 16 (2026-11-16 check) |

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
- **AV1 10/12-bit (`u16`) decode families** (Gemini, integrated 2026-11-16):
  `av1_mc::high_mc`/`high_blend`, `av1_inverse::high_simd`,
  `av1_restoration::high_lr`, `deblock::av1_high`, `av1_intra::high`,
  `av1_film_grain::high_simd`, the `u16` CDEF blocks. Each is one family over
  the shared lane traits (`I16Lanes`, `I16FilterLanes`, `I32Lanes`) on AVX2,
  AVX-512 and NEON, as dav1d's 16 bpc code is separate from its 8 bpc code:
  word lanes with `i32` products where the byte kernels pack bytes (MC: no
  tap halving, 12-bit `InterRound0` 5; deblock: `u16` samples where AV1's
  8-bit filter is on `lpf_bytes`; inverse: 10/12-bit stage ranges on the AV1
  generic graph). CDEF, the inverse graph and SGR (`sgr_lanes`) are shared
  with 8 bits; the saturating add is the shared `I16Lanes::add_sat`. VP9's
  `u16` MC and deblocking stay separate (per-pass clip, 15-tap filter).
  No SIMD128 tier yet: on wasm32 the `u16` tables are the scalar references.

## WASM u16 decode families (feature/ext-wasm10, review only, 2026-11-16)

The browser deep-output branch instantiates the existing AV1 MC/compound/
inverse lane graphs with SIMD128, the existing HEVC/VVC bounded inverse and
vector MC/weight graphs, VP9 vector MC/deblock, and shared u8/u16 SAO edge
application. Scalar references stay; AV1 compound u8 dispatch stays on its
previous WASM functions. No codec-local copies or encoder/preset changes.
See [the handoff](../handoff/ext-wasm10.md) for profile selection, exactness,
coverage limits and before/after measurements. The coordinator merges this
branch; this note does not claim it is landed.
