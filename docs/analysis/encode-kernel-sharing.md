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

## Ordered deblock batches (Astra, 2026-11-17)

H264 `deblock/h264_bytes/{grouped,wide}.rs` carries two/four independent
sixteen-byte edges in AVX2/ICL, sharing one byte decision graph. ICL two-edge
macroblocks use all 32 word lanes for widened equations. Dependency-overlapping
edges stay ordered; SSE/NEON/SIMD128 keep their natural widths. HEVC/VVC short
luma share `deblock/{batch,short_runs}.rs`: two/four independent eight-word
tiles per AVX2/ICL register, with per-segment thresholds and a codec-specific
strong clipping table. Their narrow tiers share the eight-lane graph. VVC
long/narrow filters keep the existing kernels. Transpose layout attribution:
dav1d BSD-2 loopfilter16_avx512; arithmetic remains codec-specific where required.

Whole Default encode on three public 720p CTC sources, five paired rounds:
Jetson cycles H264 -0.90%, HEVC level (+0.007%), VVC -0.42%; Zen5 CPU H264
+1.37%, HEVC -0.26%, VVC -0.52%. H264 x86 timing loss remains unresolved;
a 15-round same-executable pair-width probe is level within 0.1%. No general
H264 x86 speedup is claimed. ns/call per tier, losing trials, all-platform
checkasm and 297/297 identity receipts: `docs/handoff/astra-kernels-20261117.md`.

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
| HEVC/VVC `quantize_from` | `codec_txfm/generic.rs:1922`; NEON `codec_txfm/neon.rs` `quantize_from` | same; N on 16-bit lanes | same formula, one scalar scale; NEON (2026-11-20): `umlal` of the 16-bit magnitude into the 32-bit numerator, saturating narrow, and the AV1 quantiser's all-zero group skip (`largest_zero_coefficient`); x86 keeps the generic i32 lanes (queued: the same 16-bit route and skip on AVX2) | as above | identical to `quantize` with `mf=[scale]` | **Merge into `quantize`** |
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
| AV1 put/prep 8-tap 1/16 | `mc/av1_avx2.rs`, `av1_avx512.rs` (rows of 32), `av1_neon.rs` (shared AV1/HEVC/VVC), `av1_simd128.rs` (522), `generic.rs` | S A2 A5 (32/64 wide, put, prep, mixed) N(+dot) W | halved taps in i8, `pshufb`+`maddubs` horizontal, i16 intermediate, `pmaddwd` vertical on row pairs, `mulhrsw` rounding; 2 rows per ymm at w4/8; NEON H16 shared-shuffle `sdot`, V16 transposed bytes, HV six-tap paired rows / eight-tap paired columns | AV1 2.3 / 2.9; AV2 1.8 / 2.5 | round0 3, round1 11 (compound 7) | **Base technique** for the 8-tap family; AV2 already uses it |
| HEVC/VVC separable (`SeparableTaps<N>`) | x86: `mc/av1_avx2.rs` `separable` with `av1_avx512` rows (shared with AV1); NEON eight-tap `av1_neon.rs` + `neon_dotprod.rs` (shared AV1/HEVC/VVC), four-tap `mc/generic.rs`; WASM `av1_simd128.rs` | S A2 A5 (shared, 32/64 wide, luma and chroma) N(+dot luma) W | AVX2: AV1's `vpshufb` + `vpmaddubsw` horizontal, `vpmaddwd` vertical, two rows per step at 4/8 wide; 4-tap chroma on two tap pairs | HEVC 8.6 / 9.5; VVC 5.2 / 5.3 | full taps (fit i8), unrounded first pass, `(v+2048)>>12`; HEVC = VVC at shared phases | **AVX2 merged** (2026-11-16) onto the AV1 kernel with a `Rounding` scheme (AV1: halved taps and its roundings; HEVC/VVC: full taps). **AVX-512 merged** (2026-11-16): the same passes on rows of 32 outputs in four 128-bit lanes (`av1_avx512`, `WidePasses`); the old 32-word-lane `generic::hevc` AVX-512 path is gone. **SIMD128 merged** (2026-11-16): `separable_simd128` folded into `av1_simd128` (rounding schemes shared in `mc/rounding.rs`); node on the box, min of two runs, new/base: HEVC luma vertical 0.74-0.90 (AV1's rolling row window), luma 2-D 0.97-1.04, luma horizontal 1.00-1.09, chroma 2-D 1.04-1.07, sum of the HEVC rows 0.97 (unchanged AV1 rows 0.97-1.04, the noise). **NEON merged** (2026-11-18): the dav1d signed-byte H/V/HV schedules serve AV1 regular/smooth/sharp/mixed, HEVC luma/prep and VVC normal/affine luma, parameterised by tables and rounding. Four-tap chroma remains generic (2026-11-20: a dot-product put on the family's passes, one `sdot` per four outputs, cut HEVC/VVC encode instructions 0.1-0.5 % on the Jetson but not cycles; not landed, `docs/handoff/lowlevel.md`). See [fractional MC handoff](../handoff/astra-neon-mc.md) for selected native measurements and validation |
| AV1 compound average | `mc/generic.rs:323`, `avx512.rs:109` | S A2 A5 N W | i16 | small | `(p0+p1+16)>>5` | Keep; HEVC `hevc_bi` (decoder only) could share later |
| H.264 luma/chroma | `mc/generic.rs:842-1283`, `neon_h264.rs` | S A2 A5(copy) N W | symmetric 6-tap, `j` in 32-bit, quarter-pel `avg` | H.264 1.7 / 2.1 | 6-tap + bilinear | Keep (structurally different) |
| VP9 convolve | `vp9_mc.rs`, `vp9_mc_simd/`, `mc` byte-dot rows | S A2 A5 (32-wide strips) N(+dot) W (+relaxed dot) | Shared byte-pair rows, **u8 intermediate** | VP9 3.5 / 4.6 | Full taps, round by 7 and clip after each pass | Share the input row arithmetic across all four ISAs. VP9 keeps its per-pass narrowing and byte intermediate |
| AV1 scaled / VP9 scaled | `av1_mc/scaled/`, `vp9_mc_scaled.rs` | S A2 N W | per-sample phase | resized refs only | | Keep |

### Shared byte-dot rows (2026-11-19)

AV1, AV2, HEVC, VVC and VP9 use the overlapping source loads and packed
tap-pair gather in `mc/avx2_byte_dot.rs`. In the wide AVX-512 tier VP9
also uses the 32-output gather and sliding byte column in `mc/av1_avx512.rs`.
VP9's 4- and 8-wide AVX2 rows pair two output rows in the same vector.
VP9 uses signed SDOT sums from `mc/neon_dotprod.rs` for eight-tap
horizontal filtering on capable AArch64 cores. The vertical I8MM tier
uses the same rolling four-row transpose for vertical-only and two-axis VP9.
For two-axis blocks the input to that pass is the byte-clipped horizontal
prediction. The widening-multiply NEON path covers other shapes and CPUs.

The relaxed SIMD128 filters share the source window, pair masks, byte
gather, vertical interleave and dot in `mc/simd128_byte_dot.rs`. A VP9
phase can produce an unrounded sum outside signed 16 bits. Its two sign
groups feed the saturating subtract and its intermediate clips to bytes.
AV1, HEVC and VVC use their signed intermediate for the second pass.

The whole-encode comparison used 30 public Johnny frames. AV1 and VP9
used qindex 135, HEVC and VVC QP 36, all at Default. AV2 Faster was
included at qindex 135 in the c8a check. Callgrind used the cached gate
control at `f23b8614c` (AVX2). Zen 5 used four shuffled links each with
four paired rounds. Jetson used five alternating pairs.

| Preset | AVX2 callgrind instructions | Zen 5 instructions | Zen 5 cycles | Jetson instructions | Jetson cycles |
|---|---:|---:|---:|---:|---:|
| AV1 Default | +0.116% | +0.146% | +0.065% | +0.001% | +0.651% |
| VP9 Default | +0.065% | +0.112% | +0.149% | −0.136% | +1.179% |
| HEVC Default | +0.042% | +0.069% | +0.087% | −0.186% | −0.035% |
| VVC Default | +0.064% | +0.076% | +0.290% | −0.078% | +0.396% |
| AV2 Faster | | +0.047% | −0.051% | | |

The shared rows matched 306/306 preset streams. Native checkasm passed
2,909,101 cases on each x86 tier and 3,064,091 on NEON. Plain and relaxed
SIMD128 passed 2,909,101 and 3,171,245 cases. The outlined-SIMD checks found
zero functions. Measurements are under `work/astra-kernels-20261119/`.

### AArch64 high-bit-depth eight-tap rows (2026-11-26)

AV1 sharp and VP9 full-tap interpolation share the eight-output halfword
SDOT row in `simd/sve2.rs`. In VP9, libvpx's rolling four-row vertical
schedule consumes the horizontally clipped samples. The vertical pass
clips again before the optional rounded average. On Graviton4 the
SVE2 tier cut 10-bit VP9 whole-decode time by 1.34% on crowd and 1.04% on pan,
with identical frame hashes. NEON and SVE2 each passed 4,758,064 Checkasm
cases.

HEVC Main 10 luma preparation uses the same halfword row for horizontal
and two-axis widths of 8 through 64. The shared separable driver keeps the
HEVC intermediate scale and output bias. On Graviton4 the fractional
prep kernel was 6–21% faster across 10- and 12-bit test fixtures. Decode of
the Main 10 weighted-prediction conformance stream was 2.73% faster. The
two 90-frame general clips stayed within 0.4% of baseline, with identical
decoded frame hashes in all three.

VVC Main 10 also uses this row for its six-fractional-bit luma predictor,
including alternate half-pel coefficients. The shared separable passes
keep VVC's uncentred intermediate. On Graviton4, whole decode was 2.53%
faster on DMVR_A_3 and 1.00% faster on the 90-frame VVenC crowd stream,
with identical decoded frame hashes.

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

AV1 lane layout (av1-dec-gap-why, 2026-11-19): `deblock/lanes.rs`,
`lanes/kernel.rs` and `lpf_bytes/avx2_full.rs` filter one edge line's units
32 lines a vector with per-unit thresholds and widths (dav1d's
`loopfilter_avx2.asm` design), AVX2 only so far (AVX-512 keeps the ICL
grouped kernels until an SDE comparison, NEON and SIMD128 the eight-line
kernels). VP9 through the same kernel (a policy: 16-tap wide filter, narrow
mask on p3..q3) was exact but not faster (kernel equal, queueing +2.3 %
whole decode) and stays on its packed kernel. Details:
`docs/handoff/av1-decoder.md` "Deblocking in dav1d's lane layout". The
other tiers were measured on 2026-11-19 and keep their kernels: AVX-512
lanes (32 lines with ternlog/mask/GFNI, or 64 lines) were exact but slower
on Zen 5 (+0.3 to +1.0 % decode time) or in instructions (64 lines +3.5 %),
NEON lanes +1.5 % Jetson cycles (handoff "Lane layout on the other ISA
tiers").

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
| `block_copy` plain / rounded average / byte-to-word prep | S A2 A5 N W | all encoder prediction; reconstruction; AV1/VP9/AV2 encode and decode | removed private AV1/VP9/AV2 row copies | **Shared**: per-instance table, one exact-tail row schedule, width-specific entries. NEON moves 16 bytes / widens 16 samples; AVX2 moves32 / widens16; ICL moves64 / widens32; SIMD128 moves16 / widens16. Prep precision is `(sample << shift) - bias`; VP9 compound uses the rounded byte average. Native evidence and all-tier receipts: `docs/handoff/astra-shared-copies.md`. |

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
| H.264 deblock (`deblock/h264_bytes/{x86,wide,grouped}.rs`) | single-edge SSE; two-edge AVX2, four-edge ICL | H.264 2.8 / 3.7 | Ordered independent batches landed 3bcd6e7a0; actual two-edge ICL macroblocks use 32 word lanes for widened equations. Single-edge tails retain SSE. |
| AV1/VP9 deblock (`lpf_bytes/avx2.rs`, `deblock/avx2.rs` `av1_edge`) | 128-bit (8 lines) | AV1 3.8, VP9 3.5 | Astra's AVX-512 work; the 16-line paired `*_wide` lanes exist (256-bit) for the decoder |
| HEVC luma deblock, VVC short deblock (`deblock/{batch,short_runs}.rs`) | I16x8 single tile; I16x16 AVX2, I16x32 ICL batches | HEVC 1.7, VVC 2-3 | Ordered disjoint eight-line tiles landed 3bcd6e7a0; VVC short and long filters use the same ordered tile traversal. |
| VVC long luma deblock (`vvc_deblock/long_luma.rs`) | I16x8 single tile; I16x16 AVX2, I16x32 ICL batches | VVC | Vector endpoint decisions and signed rounded word blends for the §8.8.3.6.7 taps. The NEON vertical transpose works on bytes, shared with H.264. On Jetson, eight-line long probes use 23–25% fewer cycles and two VVC decode streams use 2.26–2.72% fewer cycles. |
| HEVC/VVC/AV1/AV2 intra runs of 16 (`intra_runs` `step16`) | 128-bit | intra | 32-wide runs are 256-bit |
| `constant_add` | 128-bit bytes | small | 32/64-wide blocks could take 256-bit |
| H.264 inverse 8x8 add (`codec_txfm/avx2.rs` `I16x8`) | I16x8 | H.264 | |
| Variance 4/8 wide (`diff/x86_narrow.rs`) | 16 word lanes AVX2 (2/4 rows), 32 ICL (4/8 rows) | all codecs | SVT packed-byte source/prediction pairs, one `pmaddubsw` difference, bounded i16 sums and paired square products; ICL VNNI square accumulation. Variance, energy and block sums share one graph. The earlier scalar-gather widening trial remains rejected; this port uses memory-operand row loads/inserts. Residual and narrow sub-pixel paths keep their existing layout. |
| AV1/HEVC/VVC MC put (`mc/av1_avx2.rs`) | 256-bit; 512-bit rows of 32 at widths 32 and 64 (2026-11-16) | AV1 2-3, HEVC 9, VVC 5 | Ice Lake (box, `nc-checkasm --bench`, ns/call AVX2 -> AVX-512): AV1 2-D 32x32 257 -> 187, 64x64 959 -> 688; HEVC luma 2-D 32x32 300 -> 224 (the old AVX-512 path took 410), 64x64 1149 -> 829 (old 1555); horizontal-only 3-9 % |
| Sub-pixel variance 16 wide | already one row per 256-bit vector (AVX2 `I16x16`) and two rows per 512-bit vector (AVX-512) | | nothing 128-bit to widen at 16 (2026-11-16 check) |

## Network inference (2026-11-22)

`nn/` (`Dsp::nn_*`): the integer inference kernels of a small NNUE-style
network. A new family with one implementation for every codec and for the
engine; nothing in it is codec arithmetic. Exact integers, so every tier
returns the scalar reference's result and a network decides the same on x86,
Arm and WebAssembly (`nc-checkasm` `check_nn`).

| Kernel | Tiers | Pattern |
|---|---|---|
| `nn_add_columns_i16`, `nn_sub_columns_i16` (accumulator ± feature columns) | S A2 A5 N W | 64 sums held in registers across the row list (4 A2, 2 A5, 8 N/W vectors); a width that is not a multiple of 64 finishes a vector at a time |
| `nn_clip_i16_to_u8` (accumulator activation) | S A2 A5 N W | arithmetic shift, signed-saturating narrow to a byte (which caps at 127), maximum with zero |
| `nn_dense_linear` (`bias + Σ w·x`, `i8` weights, 7-bit inputs) | S A2 A5 N W | four outputs per input load and one shared reduction. A2 `vpmaddubsw` + `vpmaddwd`; A5 VNNI `vpdpbusd`; N `smull`/`smlal2` + `sadalp`; W `extmul` low + high, `extadd_pairwise` |
| `nn_dense_clipped` | as above | the linear kernel into a stack block of 64 sums, then the 32-bit activation kernel: one dot product per tier |

The 16-bit pair stage of the A2, N and W dot products is exact because inputs
are at most 127: a pair sums to at most 2 · 127 · 128 = 32512 in magnitude
(`nn::MAX_PAIR_SUM`). Section 5's MC byte-dot rows use the same instruction
(`vpmaddubsw`, samples against signed taps) but are convolutions: a few fixed
taps slid along gathered sample pairs, where this is one long weight row per
output, so the two do not share a kernel. Not built: an Arm dot-product
(`sdot`) tier, which would follow `simd/neon_dot.rs`.

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

## Jetson NEON audit (Astra, 2026-11-17)

Baseline runtime 3bcd6e7a0, native release/fat LTO on Cortex-A78AE, core 3.
Each of six codecs' Default (`medium`) and Fast was sampled on public CTC
FourPeople, KristenAndSara and Johnny, 1280x720, 96 frames each, fixed QP
H264 43 / HEVC 46 / VVC 44 / AV1 214 / VP9 214 / AV2 176. The board's
exclusive timing lock was held; separate uninstrumented jperf PMU runs
record instructions and cycles. `pcsample.py` samples were grouped by
function using the frozen binary's symbols and executable mappings.
Library samples remain in the denominator, with no inference from the
nearest glibc exported symbol's name. These are coarse rankings (333–1301
samples/point), not precise cost estimates. Raw profiles and counters:
`~/nanocodec-data/work/astra-kernels-20261117/task2-profile/`.

| Codec/preset | Samples | Hottest own-binary self samples | Next two |
|---|---:|---|---|
| av1/fast | 538 | `nc_dsp::deblock::neon::av1_edge` 7.4% | `nc_dsp::pixel::neon_dotprod::sad_grid_best_x4::<2>` 4.5%; `nc_dsp::diff::neon_dotprod::variance::<64>` 4.1% |
| av1/medium | 781 | `nc_dsp::diff::neon_dotprod::variance::<64>` 7.9% | `nc_dsp::deblock::neon::av1_edge` 5.1%; `nc_dsp::pixel::neon_dotprod::sad_grid_best_x4::<2>` 4.4% |
| av2/fast | 753 | `nc_dsp::diff::neon_dotprod::variance::<32>` 7.3% | `<nc_common::view::PlaneViewMut>::copy_from` 6.9%; `nc_dsp::diff::neon_dotprod::variance::<64>` 6.1% |
| av2/medium | 955 | `nc_dsp::diff::neon_dotprod::variance::<64>` 6.3% | `nc_dsp::deblock::neon::av2_chunk` 5.9%; `<nc_common::view::PlaneViewMut>::copy_from` 5.1% |
| h264/fast | 333 | `nc_encoder::decision::choose_inter_mode::<nc_h264::backend::H264Backend>` 5.4% | `nc_dsp::diff::neon_dotprod::variance_energy::<16>` 4.8%; `nc_h264::backend::write_coded` 4.2% |
| h264/medium | 526 | `nc_encoder::decision::choose_inter_mode::<nc_h264::backend::H264Backend>` 10.3% | `nc_dsp::pixel::neon_dotprod::sad_grid_best_x4::<2>` 7.0%; `nc_dsp::pixel::neon_dotprod::sad_scan_16x16_even_rows` 6.1% |
| hevc/fast | 488 | `nc_dsp::diff::neon_dotprod::variance::<32>` 7.4% | `nc_dsp::pixel::neon_dotprod::sad_grid_best_x4::<2>` 5.7%; `<nc_encoder::encoder::Encoder<nc_hevc::backend::HevcBackend>>::code_picture` 4.1% |
| hevc/medium | 879 | `nc_dsp::diff::neon_dotprod::subpel::<64>` 10.2% | `nc_dsp::diff::neon_dotprod::variance::<64>` 4.9%; `nc_dsp::pixel::neon_dotprod::sad_grid_best_x4::<2>` 4.0% |
| vp9/fast | 833 | `nc_dsp::diff::neon_dotprod::subpel::<64>` 7.7% | `nc_dsp::txfm::neon_forward::narrow_columns::<32, 16>` 6.7%; `nc_dsp::vp9_loop_filter::packed::run_paired::<16>` 6.4% |
| vp9/medium | 970 | `nc_dsp::diff::neon_dotprod::subpel::<64>` 10.5% | `nc_dsp::txfm::neon_forward::narrow_columns::<32, 32>` 9.8%; `nc_dsp::txfm::neon_forward::narrow_rows::<32, true, 32>` 6.1% |
| vvc/fast | 772 | `nc_dsp::diff::neon_dotprod::variance::<64>` 5.4% | `nc_dsp::codec_txfm::neon::staged32_bounded` 5.1%; `nc_dsp::diff::neon_dotprod::variance::<32>` 4.7% |
| vvc/medium | 1301 | `nc_dsp::mc::neon_dotprod::vertical_2d::<64, 8>` 5.6% | `nc_dsp::codec_txfm::neon::staged32_bounded` 5.2%; `nc_dsp::diff::neon_dotprod::variance::<64>` 3.8% |

Priority and lane audit:

- **Selected:** AV1 single-edge NEON processes eight byte lines although
  the existing shared AV1/VP9 grouped adapter can process sixteen. This is
  5.1% of Default / 7.4% of Fast samples. Reuse the sixteen-line graph,
  retaining the narrow tail and scalar fallback. No new filter arithmetic.
- **Selected:** shared `diff::sse_8x8_row` is 1.9–2.6% of samples in
  AV1/HEVC/AV2/VP9. Its old NEON graph widens byte differences, multiplies
  twice and pairwise-accumulates twice per sixteen samples; byte UDOT can
  keep the two cell reductions independent. All six codecs call this family.
- **Slower than scalar:** four-wide block sums: 4x4 16.99 vs13.33 ns,
  4x8 36.03 vs22.60, 4x16 79.50 vs41.16. Four-wide statistics use I16x4
  and eight-wide energy/sums I16x8, while byte row packing can fill sixteen
  lanes. A libaom-style prototype remains outside the source branch; it
  is not selected merely because it exists.
- **Other slower cases:** VP9 intra d63 4x4 44.30 vs13.96 ns, d45 4x4
  30.63 vs14.35; HEVC vertical luma MC 32x32 1921.89 vs1282.05;
  deinterleave rows 1280x4 194.88 vs155.78; several AV2 angular intra
  shapes. These are candidates, not all fixed by this pass. MC and inverse
  ownership is respected; the full table preserves every measured result.
- **Scalar on NEON:** the reversible WHT4, VP9 four-point inverse fallback
  and VP9 codec-local small hybrid forward paths, VVC LFNST and 64-point
  inverse glue retain scalar graphs. Small normative transforms and glue
  need a workload benefit before adding an architecture-specific copy.
- Wide variance/subpel are hottest in several points but already consume
  sixteen byte samples per vector. No wider-NEON claim is made for them.

The complete quiet checkasm pass compares **1,106 matched cases across
325 named kernels**, scalar then NEON under one board lock; min of five
20ms samples per case. Dotprod dispatch is part of the NEON tier on this
CPU. Each row below shows the slowest NEON case for that kernel, both
times for that same case, speedup range across all its measured cases,
and count of cases more than 5% slower than scalar. The zero-based case
number distinguishes repeated names/shapes with different fixed inputs.
Complete per-case values: [CSV](data/neon-kernels-20261117.csv). Results
are ns per API call, not a workload-weighted prediction. A speedup below
1 means NEON was slower. This baseline table remains unchanged by later
selected-kernel timings so losing cases cannot disappear from the record.

| Kernel | Cases | Slowest NEON case | Scalar ns | NEON ns | Speedup range | NEON >5% slower |
|---|---:|---|---:|---:|---:|---:|
| `add_residual` | 5 | 64x64 #0 | 3469.01 | 776.87 | 1.05–4.51x | 0 |
| `av1_coefficient_gain` | 4 | 32x32 #0 | 745.27 | 333.54 | 1.69–2.29x | 0 |
| `av1_deblock14_h` | 3 | 8x16 #0 | 865.36 | 94.97 | 6.91–9.11x | 0 |
| `av1_deblock14_v` | 3 | 8x16 #0 | 868.28 | 130.99 | 6.57–7.21x | 0 |
| `av1_deblock4_h` | 3 | 8x16 #0 | 274.43 | 82.49 | 2.90–3.33x | 0 |
| `av1_deblock4_v` | 3 | 8x16 #0 | 277.13 | 121.79 | 2.28–2.55x | 0 |
| `av1_deblock6_h` | 3 | 8x16 #0 | 349.16 | 90.91 | 3.16–3.84x | 0 |
| `av1_deblock6_v` | 3 | 8x16 #0 | 353.66 | 127.67 | 2.77–2.98x | 0 |
| `av1_deblock8_h` | 3 | 8x16 #0 | 424.02 | 95.19 | 3.36–4.45x | 0 |
| `av1_deblock8_v` | 3 | 8x16 #0 | 428.47 | 130.90 | 3.25–3.50x | 0 |
| `av1_dequantize` | 4 | 32x32 #0 | 1172.62 | 849.23 | 1.38–1.67x | 0 |
| `av1_edge_flat_h` | 8 | 14x64 #0 | 3474.80 | 951.71 | 1.64–3.65x | 0 |
| `av1_edge_flat_v` | 8 | 14x64 #0 | 3487.16 | 1246.88 | 1.46–2.80x | 0 |
| `av1_edge_narrow_h` | 8 | 14x64 #0 | 3481.63 | 433.86 | 1.64–8.02x | 0 |
| `av1_edge_narrow_v` | 8 | 14x64 #0 | 3483.15 | 571.69 | 1.46–6.09x | 0 |
| `av1_edges_equal_h` | 1 | 64x64 #0 | 109494.82 | 8099.86 | 13.52–13.52x | 0 |
| `av1_edges_equal_v` | 1 | 64x64 #0 | 109941.95 | 9180.08 | 11.98–11.98x | 0 |
| `av1_edges_masked_h` | 1 | 64x64 #0 | 73296.07 | 5492.60 | 13.34–13.34x | 0 |
| `av1_edges_masked_v` | 1 | 64x64 #0 | 73574.53 | 6154.69 | 11.95–11.95x | 0 |
| `av1_edges_mixed_h` | 1 | 64x64 #0 | 73763.93 | 7054.26 | 10.46–10.46x | 0 |
| `av1_edges_mixed_v` | 1 | 64x64 #0 | 74127.97 | 9046.80 | 8.19–8.19x | 0 |
| `av1_filter_intra` | 4 | 32x32 #0 | 19351.04 | 2404.69 | 3.61–8.10x | 0 |
| `av1_forward_dct` | 4 | 32x32 #0 | 9290.39 | 2185.14 | 2.71–4.95x | 0 |
| `av1_forward_dct_64` | 1 | 64x64 #0 | 33900.62 | 6606.73 | 5.13–5.13x | 0 |
| `av1_forward_dct_raster` | 4 | 32x32 #0 | 7763.79 | 2293.78 | 1.99–3.69x | 0 |
| `av1_grid_equal_h` | 1 | 64x64 #0 | 109778.15 | 3543.11 | 30.98–30.98x | 0 |
| `av1_grid_equal_v` | 1 | 64x64 #0 | 109960.85 | 5110.64 | 21.52–21.52x | 0 |
| `av1_grid_masked_h` | 1 | 64x64 #0 | 73770.63 | 2951.09 | 25.00–25.00x | 0 |
| `av1_grid_masked_v` | 1 | 64x64 #0 | 73811.22 | 4259.41 | 17.33–17.33x | 0 |
| `av1_grid_mixed_h` | 1 | 64x64 #0 | 73553.21 | 6721.78 | 10.94–10.94x | 0 |
| `av1_grid_mixed_v` | 1 | 64x64 #0 | 73720.95 | 9352.94 | 7.88–7.88x | 0 |
| `av1_intra_angle_z1_51` | 5 | 64x64 #0 | 697.91 | 720.77 | 0.87–4.29x | 1 |
| `av1_intra_angle_z2_129` | 5 | 64x64 #0 | 3877.32 | 2116.74 | 1.00–1.83x | 0 |
| `av1_intra_angle_z2_157` | 5 | 64x64 #0 | 3650.84 | 1963.33 | 0.97–1.86x | 0 |
| `av1_intra_angle_z3_197` | 5 | 64x64 #0 | 3137.86 | 1325.69 | 1.03–3.13x | 0 |
| `av1_intra_paeth` | 5 | 64x64 #0 | 2718.67 | 1837.71 | 1.42–1.68x | 0 |
| `av1_intra_smooth` | 5 | 64x64 #0 | 2803.49 | 1518.61 | 0.98–1.85x | 0 |
| `av1_intra_smooth_h` | 5 | 64x64 #0 | 1822.40 | 928.43 | 0.98–1.96x | 0 |
| `av1_intra_smooth_v` | 5 | 64x64 #0 | 1858.84 | 783.05 | 0.98–2.37x | 0 |
| `av1_inverse_adst_adst` | 9 | 16x16 #0 | 3624.55 | 759.47 | 1.93–4.77x | 0 |
| `av1_inverse_dc` | 19 | 64x64 #0 | 1366.95 | 258.78 | 1.24–5.28x | 0 |
| `av1_inverse_dc_full` | 19 | 64x64 #0 | 45012.34 | 7177.12 | 1.76–6.27x | 0 |
| `av1_inverse_dct_add` | 4 | 32x32 #0 | 12241.58 | 2538.52 | 2.19–6.03x | 0 |
| `av1_inverse_dct_add_64` | 1 | 64x64 #0 | 41564.48 | 6846.19 | 6.07–6.07x | 0 |
| `av1_inverse_dct_dct` | 19 | 64x64 #0 | 45011.70 | 7177.44 | 1.75–6.32x | 0 |
| `av1_inverse_eob10` | 19 | 64x64 #0 | 22938.95 | 3903.25 | 1.78–6.64x | 0 |
| `av1_inverse_eob10_full` | 19 | 64x64 #0 | 45016.52 | 7178.45 | 1.77–6.27x | 0 |
| `av1_inverse_full_low16` | 2 | 64x64 #0 | 41572.48 | 6844.55 | 4.82–6.07x | 0 |
| `av1_inverse_full_low32` | 2 | 64x64 #0 | 41575.95 | 18132.25 | 2.16–2.29x | 0 |
| `av1_inverse_full_low8` | 2 | 64x64 #0 | 41579.95 | 6843.72 | 4.81–6.08x | 0 |
| `av1_inverse_i16_adst_adst` | 9 | 16x16 #0 | 2954.08 | 581.77 | 3.48–6.01x | 0 |
| `av1_inverse_i16_dct_dct` | 19 | 64x64 #0 | 45075.41 | 6869.58 | 3.91–6.82x | 0 |
| `av1_inverse_i16_eob10` | 19 | 64x64 #0 | 22741.51 | 5155.06 | 3.42–5.29x | 0 |
| `av1_inverse_i16_idtx` | 14 | 32x32 #0 | 5998.95 | 403.40 | 4.66–14.87x | 0 |
| `av1_inverse_idtx` | 14 | 32x32 #0 | 7350.99 | 1138.09 | 2.35–6.85x | 0 |
| `av1_inverse_sparse_low16` | 2 | 64x64 #0 | 30713.33 | 5202.86 | 5.90–7.06x | 0 |
| `av1_inverse_sparse_low32` | 2 | 64x64 #0 | 44843.52 | 18282.46 | 2.30–2.45x | 0 |
| `av1_inverse_sparse_low8` | 2 | 64x64 #0 | 22767.30 | 3723.42 | 6.11–7.26x | 0 |
| `av1_line_equal_h` | 1 | 4x64 #0 | 8769.58 | 521.23 | 16.82–16.82x | 0 |
| `av1_line_equal_v` | 1 | 4x64 #0 | 8782.19 | 664.25 | 13.22–13.22x | 0 |
| `av1_line_masked_h` | 1 | 4x64 #0 | 5607.72 | 492.44 | 11.39–11.39x | 0 |
| `av1_line_masked_v` | 1 | 4x64 #0 | 5614.21 | 584.09 | 9.61–9.61x | 0 |
| `av1_line_mixed_h` | 1 | 4x64 #0 | 5831.07 | 749.03 | 7.78–7.78x | 0 |
| `av1_line_mixed_v` | 1 | 4x64 #0 | 5845.27 | 894.95 | 6.53–6.53x | 0 |
| `av1_predict_2d` | 9 | 64x64 #0 | 142413.54 | 6898.05 | 17.10–23.20x | 0 |
| `av1_predict_h` | 9 | 64x64 #0 | 142816.95 | 2677.06 | 42.87–69.71x | 0 |
| `av1_quantize` | 4 | 32x32 #0 | 4259.80 | 1451.81 | 2.08–2.93x | 0 |
| `av1_quantize_from` | 4 | 32x32 #0 | 3869.17 | 1364.90 | 2.11–2.83x | 0 |
| `av1_run_flat_h` | 8 | 14x64 #0 | 3469.95 | 859.46 | 1.64–4.04x | 0 |
| `av1_run_flat_v` | 8 | 14x64 #0 | 3489.06 | 1031.32 | 1.53–3.38x | 0 |
| `av1_run_narrow_h` | 8 | 8x64 #0 | 1759.48 | 398.86 | 1.64–8.74x | 0 |
| `av1_run_narrow_v` | 8 | 14x64 #0 | 3484.81 | 480.26 | 1.55–7.26x | 0 |
| `av1_sharp_2d` | 9 | 64x64 #0 | 142151.12 | 7999.48 | 14.81–19.63x | 0 |
| `av1_single_equal_h` | 1 | 4x64 #0 | 8691.23 | 892.09 | 9.74–9.74x | 0 |
| `av1_single_equal_v` | 1 | 4x64 #0 | 8707.03 | 950.05 | 9.16–9.16x | 0 |
| `av1_single_masked_h` | 1 | 4x64 #0 | 5548.58 | 669.90 | 8.28–8.28x | 0 |
| `av1_single_masked_v` | 1 | 4x64 #0 | 5554.50 | 698.55 | 7.95–7.95x | 0 |
| `av1_single_mixed_h` | 1 | 4x64 #0 | 5753.74 | 774.00 | 7.43–7.43x | 0 |
| `av1_single_mixed_v` | 1 | 4x64 #0 | 5765.57 | 890.82 | 6.47–6.47x | 0 |
| `av1_smooth_2d` | 9 | 64x64 #0 | 142316.39 | 6897.95 | 17.13–23.19x | 0 |
| `av1_wide_2d` | 3 | 128x128 #0 | 269219.14 | 27715.60 | 7.79–9.71x | 0 |
| `av1_wide_copy` | 3 | 128x128 #0 | 269485.09 | 977.02 | 32.27–275.82x | 0 |
| `av1_wide_h` | 3 | 128x128 #0 | 269415.47 | 10839.00 | 17.13–25.23x | 0 |
| `av1_wide_v` | 3 | 128x128 #0 | 269342.38 | 10071.84 | 12.29–26.74x | 0 |
| `av2_deblock_line_h` | 1 | 32x32 #0 | 1527.69 | 363.33 | 4.20–4.20x | 0 |
| `av2_deblock_line_v` | 1 | 32x32 #0 | 1499.20 | 334.58 | 4.48–4.48x | 0 |
| `av2_deblock_rough_h` | 1 | 32x32 #0 | 454.25 | 151.59 | 3.00–3.00x | 0 |
| `av2_deblock_rough_v` | 1 | 32x32 #0 | 422.23 | 236.68 | 1.78–1.78x | 0 |
| `av2_forward_dct` | 5 | 64x64 #0 | 21526.42 | 14137.05 | 1.47–2.27x | 0 |
| `av2_forward_typed` | 4 | 32x32 #0 | 6537.99 | 7084.38 | 0.92–1.52x | 1 |
| `av2_intra_z1_51_chroma` | 5 | 64x64 #0 | 1182.41 | 1152.21 | 0.86–1.10x | 1 |
| `av2_intra_z1_51_luma` | 5 | 64x64 #0 | 4615.26 | 4129.26 | 0.81–1.12x | 4 |
| `av2_intra_z2_129_luma` | 5 | 64x64 #0 | 7424.22 | 7618.89 | 0.84–0.97x | 4 |
| `av2_intra_z3_197_chroma` | 5 | 64x64 #0 | 5978.30 | 5992.53 | 0.86–1.00x | 2 |
| `av2_intra_z3_197_luma` | 5 | 64x64 #0 | 8278.35 | 9309.42 | 0.72–0.93x | 5 |
| `av2_inverse_dct` | 5 | 64x64 #0 | 13390.73 | 5035.13 | 2.03–2.93x | 0 |
| `av2_inverse_typed` | 4 | 32x32 #0 | 19862.37 | 7366.54 | 1.35–3.50x | 0 |
| `block_sums` | 13 | 64x64 #0 | 1112.02 | 220.27 | 0.52–10.95x | 3 |
| `cdef8_strided_both` | 1 | 8x8 #0 | 2886.71 | 203.08 | 14.21–14.21x | 0 |
| `cdef8_strided_off` | 1 | 8x8 #0 | 92.58 | 20.27 | 4.57–4.57x | 0 |
| `cdef8_strided_pri` | 1 | 8x8 #0 | 1050.90 | 82.97 | 12.67–12.67x | 0 |
| `cdef8_strided_sec` | 1 | 8x8 #0 | 1999.97 | 121.53 | 16.46–16.46x | 0 |
| `cdef_filter` | 2 | 8x8 #0 | 2913.79 | 224.78 | 4.80–12.96x | 0 |
| `cdef_filter_u8` | 2 | 8x8 #0 | 2880.74 | 202.51 | 10.65–14.23x | 0 |
| `cdef_filter_u8_pri` | 2 | 8x8 #0 | 1049.46 | 82.19 | 7.88–12.77x | 0 |
| `cdef_filter_u8_sec` | 2 | 8x8 #0 | 2011.67 | 119.49 | 11.38–16.84x | 0 |
| `cdef_find_dir` | 1 | 8x8 #0 | 425.58 | 82.28 | 5.17–5.17x | 0 |
| `compound_average` | 3 | 64x64 #0 | 3775.88 | 586.88 | 6.43–10.62x | 0 |
| `copy_plane` | 3 | 1884x8 #0 | 639.19 | 656.15 | 0.97–0.99x | 0 |
| `deinterleave_uv` | 1 | 64x32 #0 | 232.68 | 223.08 | 1.04–1.04x | 0 |
| `deinterleave_uv_rows` | 3 | 1884x4 #0 | 314.97 | 388.57 | 0.80–0.87x | 3 |
| `downscale_2x` | 1 | 32x32 #0 | 1657.19 | 308.31 | 5.38–5.38x | 0 |
| `downscale_2x_rows` | 3 | 1884x4 #0 | 730.69 | 358.24 | 2.04–2.88x | 0 |
| `fullpel_put` | 3 | 64x64 #0 | 143679.18 | 170.32 | 278.19–843.58x | 0 |
| `gram_rows` | 1 | 64x4 #0 | 3627.84 | 1841.11 | 1.97–1.97x | 0 |
| `h264_batch_strong_h` | 3 | 16x4 #0 | 1366.56 | 304.67 | 2.55–4.49x | 0 |
| `h264_batch_strong_v` | 3 | 16x4 #0 | 1331.83 | 426.54 | 2.18–3.12x | 0 |
| `h264_batch_weak_h` | 3 | 16x4 #0 | 1415.87 | 258.62 | 2.87–5.47x | 0 |
| `h264_batch_weak_v` | 3 | 16x4 #0 | 1413.00 | 387.46 | 2.41–3.65x | 0 |
| `h264_chroma` | 9 | 64x64 #0 | 4945.40 | 1439.27 | 1.59–7.41x | 0 |
| `h264_deblock_chroma_h` | 1 | 8x8 #0 | 127.03 | 63.07 | 2.01–2.01x | 0 |
| `h264_deblock_chroma_v` | 1 | 8x8 #0 | 127.81 | 88.74 | 1.44–1.44x | 0 |
| `h264_deblock_luma_h` | 1 | 8x8 #0 | 367.24 | 70.79 | 5.19–5.19x | 0 |
| `h264_deblock_luma_v` | 1 | 8x8 #0 | 366.09 | 99.07 | 3.70–3.70x | 0 |
| `h264_deblock_mb_luma` | 1 | 16x16 #0 | 2625.79 | 373.84 | 7.02–7.02x | 0 |
| `h264_deblock_mb_luma_intra` | 1 | 16x16 #0 | 2579.16 | 397.00 | 6.50–6.50x | 0 |
| `h264_dequantize_4x4` | 1 | 16x16 #0 | 712.39 | 129.13 | 5.52–5.52x | 0 |
| `h264_edges_dense_strong` | 1 | 16x8 #0 | 2529.87 | 526.77 | 4.80–4.80x | 0 |
| `h264_edges_dense_weak` | 1 | 16x8 #0 | 2677.93 | 444.63 | 6.02–6.02x | 0 |
| `h264_edges_pair_strong` | 1 | 16x4 #0 | 1322.62 | 316.20 | 4.18–4.18x | 0 |
| `h264_edges_pair_weak` | 1 | 16x4 #0 | 1392.27 | 271.80 | 5.12–5.12x | 0 |
| `h264_forward_4x4` | 1 | 16x16 #0 | 470.41 | 121.60 | 3.87–3.87x | 0 |
| `h264_forward_8x8` | 1 | 8x8 #0 | 176.08 | 60.23 | 2.92–2.92x | 0 |
| `h264_forward_quantize_mb` | 1 | 16x16 #0 | 1041.89 | 510.02 | 2.04–2.04x | 0 |
| `h264_hadamard_4x4` | 1 | 4x4 #0 | 15.95 | 12.70 | 1.26–1.26x | 0 |
| `h264_inverse_4x4` | 1 | 16x16 #0 | 305.22 | 203.84 | 1.50–1.50x | 0 |
| `h264_inverse_8x8` | 1 | 8x8 #0 | 114.20 | 75.28 | 1.52–1.52x | 0 |
| `h264_luma_f` | 9 | 64x64 #0 | 217081.83 | 4302.12 | 8.24–50.46x | 0 |
| `h264_luma_j` | 9 | 64x64 #0 | 106247.02 | 4126.59 | 4.53–25.75x | 0 |
| `h264_mb_dense_strong` | 1 | 16x8 #0 | 2540.12 | 515.59 | 4.93–4.93x | 0 |
| `h264_mb_dense_weak` | 1 | 16x8 #0 | 2691.26 | 435.75 | 6.18–6.18x | 0 |
| `h264_mb_pair_strong` | 1 | 16x4 #0 | 1336.36 | 316.16 | 4.23–4.23x | 0 |
| `h264_mb_pair_weak` | 1 | 16x4 #0 | 1401.66 | 275.67 | 5.08–5.08x | 0 |
| `h264_quantize_4x4` | 1 | 16x16 #0 | 406.74 | 288.03 | 1.41–1.41x | 0 |
| `h264_quantize_8x8` | 1 | 8x8 #0 | 78.24 | 70.82 | 1.10–1.10x | 0 |
| `h264_single_strong_h` | 3 | 16x4 #0 | 1332.79 | 287.58 | 2.64–4.63x | 0 |
| `h264_single_strong_v` | 3 | 16x4 #0 | 1329.93 | 387.16 | 2.25–3.44x | 0 |
| `h264_single_weak_h` | 3 | 16x4 #0 | 1410.92 | 237.98 | 2.93–5.93x | 0 |
| `h264_single_weak_v` | 3 | 16x4 #0 | 1408.78 | 358.02 | 2.45–3.93x | 0 |
| `hevc_add_dc_picture` | 2 | 32x32 #0 | 879.70 | 663.35 | 1.33–1.82x | 0 |
| `hevc_add_full_picture` | 2 | 32x32 #0 | 14319.62 | 5261.95 | 2.46–2.72x | 0 |
| `hevc_add_low4_picture` | 2 | 32x32 #0 | 2548.52 | 1227.04 | 2.08–2.72x | 0 |
| `hevc_add_low8_picture` | 2 | 32x32 #0 | 3105.08 | 1570.53 | 1.98–2.66x | 0 |
| `hevc_batch_strong_h` | 4 | 8x64 #0 | 2133.28 | 403.19 | 4.01–5.29x | 0 |
| `hevc_batch_strong_v` | 4 | 8x64 #0 | 2135.46 | 550.30 | 3.15–3.88x | 0 |
| `hevc_batch_weak_h` | 4 | 8x64 #0 | 1760.01 | 383.15 | 3.47–4.59x | 0 |
| `hevc_batch_weak_v` | 4 | 8x64 #0 | 1764.09 | 530.09 | 2.72–3.33x | 0 |
| `hevc_chroma_2d` | 9 | 64x64 #0 | 10127.26 | 4927.81 | 1.59–2.71x | 0 |
| `hevc_deblock_chroma4_h` | 1 | 4x4 #0 | 80.83 | 61.40 | 1.32–1.32x | 0 |
| `hevc_deblock_chroma4_v` | 1 | 4x4 #0 | 81.65 | 68.48 | 1.19–1.19x | 0 |
| `hevc_deblock_chroma_h` | 1 | 8x8 #0 | 108.44 | 59.32 | 1.83–1.83x | 0 |
| `hevc_deblock_chroma_v` | 1 | 8x8 #0 | 107.29 | 75.91 | 1.41–1.41x | 0 |
| `hevc_deblock_luma_h` | 1 | 8x8 #0 | 258.81 | 109.96 | 2.35–2.35x | 0 |
| `hevc_deblock_luma_v` | 1 | 8x8 #0 | 279.29 | 146.12 | 1.91–1.91x | 0 |
| `hevc_dequantize` | 4 | 32x32 #0 | 3118.19 | 630.35 | 4.86–5.07x | 0 |
| `hevc_forward_bounded` | 4 | 32x32 #0 | 11628.62 | 5284.85 | 2.20–3.27x | 0 |
| `hevc_forward_dct` | 4 | 32x32 #0 | 8786.31 | 5068.57 | 1.72–2.51x | 0 |
| `hevc_forward_dst` | 1 | 4x4 #0 | 60.99 | 35.83 | 1.70–1.70x | 0 |
| `hevc_forward_staged_both` | 2 | 32x32 #0 | 14899.88 | 3332.51 | 3.96–4.47x | 0 |
| `hevc_forward_staged_rows` | 2 | 32x32 #0 | 12490.20 | 4116.45 | 3.03–3.48x | 0 |
| `hevc_intra_angular_m21` | 4 | 32x32 #0 | 1389.18 | 169.02 | 2.69–11.65x | 0 |
| `hevc_intra_angular_p13` | 4 | 32x32 #0 | 1389.23 | 168.78 | 2.70–11.72x | 0 |
| `hevc_intra_planar` | 4 | 32x32 #0 | 983.78 | 373.41 | 2.63–3.23x | 0 |
| `hevc_inverse_add_dc` | 4 | 32x32 #0 | 878.36 | 663.57 | 1.32–2.05x | 0 |
| `hevc_inverse_add_full` | 4 | 32x32 #0 | 14333.46 | 5263.28 | 1.88–2.72x | 0 |
| `hevc_inverse_add_low4` | 4 | 32x32 #0 | 2548.00 | 1226.76 | 1.88–2.72x | 0 |
| `hevc_inverse_add_low8` | 3 | 32x32 #0 | 3102.03 | 1569.59 | 1.98–2.65x | 0 |
| `hevc_inverse_dct` | 4 | 32x32 #0 | 12552.52 | 5547.21 | 1.09–2.26x | 0 |
| `hevc_inverse_dense` | 5 | 32x32 #0 | 12566.43 | 5542.32 | 1.09–2.27x | 0 |
| `hevc_inverse_dst` | 1 | 4x4 #0 | 58.49 | 32.94 | 1.78–1.78x | 0 |
| `hevc_inverse_split_dc` | 4 | 32x32 #0 | 12974.82 | 1900.84 | 1.28–6.83x | 0 |
| `hevc_inverse_split_full` | 4 | 32x32 #0 | 12975.63 | 5964.49 | 1.08–2.18x | 0 |
| `hevc_inverse_split_low4` | 4 | 32x32 #0 | 12977.31 | 1944.57 | 1.28–6.67x | 0 |
| `hevc_inverse_split_low8` | 3 | 32x32 #0 | 12974.39 | 2311.86 | 1.28–5.61x | 0 |
| `hevc_luma_2d` | 9 | 64x64 #0 | 18592.66 | 8280.53 | 1.65–2.50x | 0 |
| `hevc_luma_h` | 9 | 64x64 #0 | 9694.43 | 2651.62 | 2.01–3.84x | 0 |
| `hevc_luma_v` | 9 | 64x64 #0 | 9877.37 | 7441.78 | 0.67–1.42x | 5 |
| `hevc_quantize` | 4 | 32x32 #0 | 1634.74 | 1165.50 | 1.40–1.51x | 0 |
| `hevc_quantize_from` | 4 | 32x32 #0 | 1133.15 | 1062.24 | 1.07–1.26x | 0 |
| `hevc_single_strong_h` | 4 | 8x64 #0 | 2123.45 | 724.13 | 2.86–3.00x | 0 |
| `hevc_single_strong_v` | 4 | 8x64 #0 | 2125.67 | 813.84 | 2.49–2.62x | 0 |
| `hevc_single_weak_h` | 4 | 8x64 #0 | 1760.22 | 605.47 | 2.91–3.11x | 0 |
| `hevc_single_weak_v` | 4 | 8x64 #0 | 1762.15 | 682.62 | 2.37–2.59x | 0 |
| `hevc_split_dc_picture` | 2 | 32x32 #0 | 12977.17 | 1910.24 | 2.29–6.79x | 0 |
| `hevc_split_full_picture` | 2 | 32x32 #0 | 12982.57 | 5974.59 | 1.08–2.17x | 0 |
| `hevc_split_low4_picture` | 2 | 32x32 #0 | 12982.72 | 1950.35 | 2.22–6.66x | 0 |
| `hevc_split_low8_picture` | 2 | 32x32 #0 | 12982.97 | 2316.64 | 1.76–5.60x | 0 |
| `moments` | 1 | 8x8 #0 | 45.40 | 11.74 | 3.87–3.87x | 0 |
| `moments_row4` | 3 | 1884x8 #0 | 4329.62 | 481.82 | 8.73–8.99x | 0 |
| `moments_row8` | 3 | 1884x8 #0 | 8660.10 | 782.07 | 10.48–11.20x | 0 |
| `pair_mean_rows` | 3 | 1884x8 #0 | 713.95 | 706.32 | 1.01–7.76x | 0 |
| `pair_pick_rows` | 3 | 1884x8 #0 | 3177.94 | 536.47 | 5.35–7.12x | 0 |
| `plane_sse` | 3 | 1884x8 #0 | 6419.26 | 1970.17 | 2.10–3.30x | 0 |
| `residual` | 9 | 64x64 #0 | 2608.10 | 233.15 | 1.69–19.79x | 0 |
| `residual_sad` | 5 | 64x64 #0 | 3296.59 | 462.46 | 1.47–13.01x | 0 |
| `sad` | 9 | 64x64 #0 | 351.06 | 195.57 | 1.68–5.47x | 0 |
| `sad_4x4_macroblock` | 1 | 16x16 #0 | 617.71 | 52.15 | 11.84–11.84x | 0 |
| `sad_grid_best` | 1 | 64x64 #0 | 3138.89 | 234.01 | 13.41–13.41x | 0 |
| `sad_grid_best_even` | 2 | 64x64 #1 | 1966.77 | 160.00 | 12.28–12.29x | 0 |
| `sad_grid_best_x4` | 1 | 64x64 #0 | 12636.33 | 855.18 | 14.78–14.78x | 0 |
| `sad_grid_best_x4_even` | 1 | 64x64 #0 | 7654.93 | 564.27 | 13.57–13.57x | 0 |
| `sad_grid_x4` | 1 | 64x64 #0 | 12701.98 | 857.57 | 14.81–14.81x | 0 |
| `sad_grid_x4_even` | 1 | 64x64 #0 | 7664.80 | 568.54 | 13.48–13.48x | 0 |
| `sad_quads` | 2 | 16x16 #0 | 239.32 | 19.77 | 9.47–12.11x | 0 |
| `sad_scan_x33` | 1 | 16x16 #0 | 1712.85 | 172.42 | 9.93–9.93x | 0 |
| `sad_x4` | 9 | 64x64 #0 | 1398.05 | 609.83 | 2.29–7.87x | 0 |
| `sao_band_apply` | 1 | 32x32 #0 | 1768.37 | 889.28 | 1.99–1.99x | 0 |
| `sao_band_masked` | 1 | 32x32 #0 | 1893.32 | 1024.57 | 1.85–1.85x | 0 |
| `sao_edge_apply` | 1 | 32x32 #0 | 5595.08 | 1648.92 | 3.39–3.39x | 0 |
| `sao_edge_stats` | 1 | 32x32 #0 | 4783.01 | 1687.48 | 2.83–2.83x | 0 |
| `satd` | 9 | 64x64 #0 | 26177.51 | 1527.26 | 3.03–17.14x | 0 |
| `satd_4x4_tiles` | 5 | 64x64 #0 | 8670.04 | 2296.69 | 2.97–3.80x | 0 |
| `satd_8x8_quads` | 1 | 16x16 #0 | 1703.04 | 104.86 | 16.24–16.24x | 0 |
| `satd_8x8_quads_split` | 1 | 16x16 #0 | 1715.18 | 160.37 | 10.70–10.70x | 0 |
| `satd_quads_picture` | 1 | 16x16 #0 | 1703.66 | 107.52 | 15.85–15.85x | 0 |
| `satd_quads_picture_split` | 1 | 16x16 #0 | 1717.59 | 163.22 | 10.52–10.52x | 0 |
| `sse_8x8_row` | 1 | 8x8 #0 | 36.58 | 13.08 | 2.80–2.80x | 0 |
| `sub_pixel_variance` | 9 | 64x64 #0 | 27767.28 | 2033.43 | 7.23–22.64x | 0 |
| `variance` | 13 | 64x64 #0 | 2349.25 | 301.83 | 0.97–14.14x | 0 |
| `variance_energy` | 13 | 64x64 #0 | 3738.54 | 380.64 | 0.96–13.61x | 0 |
| `vp9_avg_strided_2d` | 5 | 64x64 #0 | 20884.07 | 5251.43 | 3.50–6.41x | 0 |
| `vp9_avg_strided_copy` | 5 | 64x64 #0 | 410.46 | 437.88 | 0.94–2.45x | 1 |
| `vp9_avg_strided_h` | 5 | 64x64 #0 | 8886.84 | 2408.02 | 1.97–5.13x | 0 |
| `vp9_avg_strided_v` | 5 | 64x64 #0 | 14346.53 | 2408.38 | 4.47–12.97x | 0 |
| `vp9_bilinear_2d` | 5 | 64x64 #0 | 18921.21 | 1434.12 | 6.75–18.90x | 0 |
| `vp9_bilinear_copy` | 5 | 64x64 #0 | 362.84 | 242.22 | 1.50–2.89x | 0 |
| `vp9_bilinear_h` | 5 | 64x64 #0 | 7957.70 | 708.98 | 2.88–12.08x | 0 |
| `vp9_bilinear_v` | 5 | 64x64 #0 | 13517.29 | 674.21 | 6.89–33.23x | 0 |
| `vp9_filter16_h_flat` | 1 | 8x16 #0 | 1509.75 | 146.62 | 10.30–10.30x | 0 |
| `vp9_filter16_h_mixed` | 1 | 8x16 #0 | 819.27 | 146.83 | 5.58–5.58x | 0 |
| `vp9_filter16_h_narrow` | 1 | 8x16 #0 | 302.20 | 77.62 | 3.89–3.89x | 0 |
| `vp9_filter16_h_reject` | 1 | 8x16 #0 | 108.17 | 66.55 | 1.63–1.63x | 0 |
| `vp9_filter16_v_flat` | 1 | 8x16 #0 | 1487.45 | 185.94 | 8.00–8.00x | 0 |
| `vp9_filter16_v_mixed` | 1 | 8x16 #0 | 805.58 | 185.71 | 4.34–4.34x | 0 |
| `vp9_filter16_v_narrow` | 1 | 8x16 #0 | 298.95 | 95.41 | 3.13–3.13x | 0 |
| `vp9_filter16_v_reject` | 1 | 8x16 #0 | 108.10 | 70.16 | 1.54–1.54x | 0 |
| `vp9_filter4_h_flat` | 1 | 8x4 #0 | 258.41 | 66.15 | 3.91–3.91x | 0 |
| `vp9_filter4_h_mixed` | 1 | 8x4 #0 | 181.52 | 65.95 | 2.75–2.75x | 0 |
| `vp9_filter4_h_narrow` | 1 | 8x4 #0 | 256.83 | 66.14 | 3.88–3.88x | 0 |
| `vp9_filter4_h_reject` | 1 | 8x4 #0 | 108.30 | 61.26 | 1.77–1.77x | 0 |
| `vp9_filter4_v_flat` | 1 | 8x4 #0 | 257.10 | 84.97 | 3.03–3.03x | 0 |
| `vp9_filter4_v_mixed` | 1 | 8x4 #0 | 173.33 | 85.08 | 2.04–2.04x | 0 |
| `vp9_filter4_v_narrow` | 1 | 8x4 #0 | 263.52 | 85.17 | 3.09–3.09x | 0 |
| `vp9_filter4_v_reject` | 1 | 8x4 #0 | 108.00 | 66.15 | 1.63–1.63x | 0 |
| `vp9_filter8_h_flat` | 1 | 8x8 #0 | 257.41 | 84.42 | 3.05–3.05x | 0 |
| `vp9_filter8_h_mixed` | 1 | 8x8 #0 | 181.37 | 84.40 | 2.15–2.15x | 0 |
| `vp9_filter8_h_narrow` | 1 | 8x8 #0 | 258.87 | 72.13 | 3.59–3.59x | 0 |
| `vp9_filter8_h_reject` | 1 | 8x8 #0 | 108.12 | 63.17 | 1.71–1.71x | 0 |
| `vp9_filter8_v_flat` | 1 | 8x8 #0 | 255.06 | 106.27 | 2.40–2.40x | 0 |
| `vp9_filter8_v_mixed` | 1 | 8x8 #0 | 176.79 | 106.14 | 1.67–1.67x | 0 |
| `vp9_filter8_v_narrow` | 1 | 8x8 #0 | 257.05 | 92.21 | 2.79–2.79x | 0 |
| `vp9_filter8_v_reject` | 1 | 8x8 #0 | 107.84 | 68.17 | 1.58–1.58x | 0 |
| `vp9_intra_d117` | 4 | 32x32 #0 | 1206.85 | 246.27 | 1.12–4.90x | 0 |
| `vp9_intra_d153` | 4 | 32x32 #0 | 1302.99 | 254.72 | 1.31–5.12x | 0 |
| `vp9_intra_d207` | 4 | 32x32 #0 | 1295.34 | 189.33 | 1.43–6.84x | 0 |
| `vp9_intra_d45` | 4 | 32x32 #0 | 1316.88 | 142.49 | 0.47–9.24x | 1 |
| `vp9_intra_d63` | 4 | 32x32 #0 | 2070.62 | 198.99 | 0.32–10.41x | 1 |
| `vp9_intra_dc` | 4 | 32x32 #0 | 81.10 | 80.25 | 1.00–1.01x | 0 |
| `vp9_intra_tm` | 4 | 32x32 #0 | 993.55 | 145.55 | 1.19–6.83x | 0 |
| `vp9_inverse_aa` | 2 | 16x16 #0 | 1851.19 | 705.52 | 2.62–3.72x | 0 |
| `vp9_inverse_ad` | 2 | 16x16 #0 | 1743.04 | 597.19 | 2.92–4.13x | 0 |
| `vp9_inverse_da` | 2 | 16x16 #0 | 1730.50 | 593.64 | 2.92–4.19x | 0 |
| `vp9_inverse_dd` | 3 | 32x32 #0 | 8377.32 | 2580.54 | 3.25–4.90x | 0 |
| `vp9_line_equal_h` | 1 | 8x64 #0 | 2371.49 | 424.89 | 5.58–5.58x | 0 |
| `vp9_line_equal_v` | 1 | 8x64 #0 | 2383.29 | 517.01 | 4.61–4.61x | 0 |
| `vp9_line_flat_equal_h` | 1 | 8x64 #0 | 12067.55 | 894.16 | 13.50–13.50x | 0 |
| `vp9_line_flat_equal_v` | 1 | 8x64 #0 | 11837.15 | 1078.91 | 10.97–10.97x | 0 |
| `vp9_line_flat_masked_h` | 1 | 8x64 #0 | 7677.16 | 712.95 | 10.77–10.77x | 0 |
| `vp9_line_flat_masked_v` | 1 | 8x64 #0 | 7544.43 | 855.30 | 8.82–8.82x | 0 |
| `vp9_line_flat_mixed_h` | 1 | 8x64 #0 | 6979.53 | 790.48 | 8.83–8.83x | 0 |
| `vp9_line_flat_mixed_v` | 1 | 8x64 #0 | 6856.46 | 1017.06 | 6.74–6.74x | 0 |
| `vp9_line_masked_h` | 1 | 8x64 #0 | 1602.18 | 415.06 | 3.86–3.86x | 0 |
| `vp9_line_masked_v` | 1 | 8x64 #0 | 1612.74 | 477.96 | 3.37–3.37x | 0 |
| `vp9_line_mixed_h` | 1 | 8x64 #0 | 2227.10 | 519.62 | 4.29–4.29x | 0 |
| `vp9_line_mixed_v` | 1 | 8x64 #0 | 2158.23 | 669.37 | 3.22–3.22x | 0 |
| `vp9_mc_2d` | 5 | 64x64 #0 | 18921.55 | 5053.74 | 3.42–6.37x | 0 |
| `vp9_mc_copy` | 5 | 64x64 #0 | 362.40 | 240.48 | 1.51–2.88x | 0 |
| `vp9_mc_h` | 5 | 64x64 #0 | 7952.10 | 2407.11 | 1.89–4.61x | 0 |
| `vp9_mc_v` | 5 | 64x64 #0 | 13516.32 | 2408.98 | 4.34–12.21x | 0 |
| `vp9_put_strided` | 5 | 64x64 #0 | 20049.63 | 5173.63 | 3.43–6.39x | 0 |
| `vp9_quantize` | 3 | 32x32 #0 | 3607.60 | 1391.32 | 2.35–2.59x | 0 |
| `vp9_sharp_2d` | 5 | 64x64 #0 | 18899.79 | 5056.75 | 3.42–6.37x | 0 |
| `vp9_sharp_copy` | 5 | 64x64 #0 | 362.72 | 241.88 | 1.50–2.89x | 0 |
| `vp9_sharp_h` | 5 | 64x64 #0 | 7949.45 | 2407.08 | 1.89–4.64x | 0 |
| `vp9_sharp_v` | 5 | 64x64 #0 | 13518.76 | 2414.72 | 4.36–12.22x | 0 |
| `vp9_single_equal_h` | 1 | 8x64 #0 | 2410.12 | 563.97 | 4.27–4.27x | 0 |
| `vp9_single_equal_v` | 1 | 8x64 #0 | 2415.56 | 714.30 | 3.38–3.38x | 0 |
| `vp9_single_flat_equal_h` | 1 | 8x64 #0 | 11976.56 | 1090.28 | 10.98–10.98x | 0 |
| `vp9_single_flat_equal_v` | 1 | 8x64 #0 | 11864.18 | 1410.44 | 8.41–8.41x | 0 |
| `vp9_single_flat_masked_h` | 1 | 8x64 #0 | 7589.19 | 790.26 | 9.60–9.60x | 0 |
| `vp9_single_flat_masked_v` | 1 | 8x64 #0 | 7523.18 | 986.46 | 7.63–7.63x | 0 |
| `vp9_single_flat_mixed_h` | 1 | 8x64 #0 | 6975.18 | 786.01 | 8.87–8.87x | 0 |
| `vp9_single_flat_mixed_v` | 1 | 8x64 #0 | 6900.65 | 1023.89 | 6.74–6.74x | 0 |
| `vp9_single_masked_h` | 1 | 8x64 #0 | 1602.43 | 457.09 | 3.51–3.51x | 0 |
| `vp9_single_masked_v` | 1 | 8x64 #0 | 1612.88 | 548.60 | 2.94–2.94x | 0 |
| `vp9_single_mixed_h` | 1 | 8x64 #0 | 2211.30 | 519.27 | 4.26–4.26x | 0 |
| `vp9_single_mixed_v` | 1 | 8x64 #0 | 2166.29 | 669.92 | 3.23–3.23x | 0 |
| `vp9_smooth_2d` | 5 | 64x64 #0 | 18910.84 | 5050.44 | 3.47–6.40x | 0 |
| `vp9_smooth_copy` | 5 | 64x64 #0 | 362.72 | 241.81 | 1.50–2.89x | 0 |
| `vp9_smooth_h` | 5 | 64x64 #0 | 7952.24 | 2405.49 | 1.83–4.60x | 0 |
| `vp9_smooth_v` | 5 | 64x64 #0 | 13518.45 | 2406.68 | 4.39–12.26x | 0 |
| `vvc_alf_filter_block_row` | 1 | 64x4 #0 | 11601.23 | 703.20 | 16.50–16.50x | 0 |
| `vvc_alf_gram` | 1 | 64x8 #0 | 25520.73 | 8287.82 | 3.08–3.08x | 0 |
| `vvc_alf_laplacians` | 1 | 64x4 #0 | 597.75 | 347.73 | 1.72–1.72x | 0 |
| `vvc_alf_samples` | 1 | 64x4 #0 | 1318.43 | 251.54 | 5.24–5.24x | 0 |
| `vvc_batch_strong_h` | 3 | 8x4 #0 | 2307.86 | 329.13 | 3.81–7.01x | 0 |
| `vvc_batch_strong_v` | 3 | 8x4 #0 | 2066.47 | 397.88 | 3.21–5.19x | 0 |
| `vvc_batch_weak_h` | 3 | 8x4 #0 | 2137.49 | 318.29 | 3.67–6.72x | 0 |
| `vvc_batch_weak_v` | 3 | 8x4 #0 | 1888.31 | 387.48 | 3.03–4.87x | 0 |
| `vvc_forward64` | 1 | 64x64 #0 | 25728.42 | 6285.71 | 4.09–4.09x | 0 |
| `vvc_forward_dct` | 5 | 32x16 #0 | 7149.99 | 2455.22 | 2.90–3.71x | 0 |
| `vvc_single_strong_h` | 3 | 8x4 #0 | 2333.79 | 307.80 | 4.00–7.58x | 0 |
| `vvc_single_strong_v` | 3 | 8x4 #0 | 2069.62 | 370.25 | 3.33–5.59x | 0 |
| `vvc_single_weak_h` | 3 | 8x4 #0 | 2161.24 | 277.05 | 3.88–7.80x | 0 |
| `vvc_single_weak_v` | 3 | 8x4 #0 | 1896.37 | 352.05 | 3.21–5.39x | 0 |

Selected-pass result (runtime 255a54a64): NEON SSE one-cell API 13.08 ->
11.31 ns; AV1 64-line horizontal edges 6.75–10.82% faster, vertical
14.50–18.99% faster. Canonical ARM and x86 preset identity both 297/297;
all native/SDE/SIMD128 checkasm and outlined-intrinsic guards pass.
Five quiet paired rounds on every codec/preset give whole cycles: H264
Default/Fast +0.86/-1.09%, HEVC -0.66/+0.08%, VVC -0.03/+0.75%, AV1
+0.03/+0.04% (level), VP9 -1.24/+0.03%, AV2 -0.64/-0.51%. Instructions
fall at all twelve points (AV1 -0.77/-0.60%). Do not claim an AV1
whole-encode speedup or hide the unfavorable codec timings. Full values,
all kernel cases, sources and final conformance receipt:
[handoff](../handoff/astra-kernels-20261117.md).


### Updated AV1 Arm priority 1: shared CDEF arithmetic (2026-11-17)

The NEON interior filter already uses sixteen byte lanes; only boundaries
and high depth use eight word lanes. The follow-up retains the shared
AV1/AV2 family and scalar oracle. Direction costs use dav1d's mirrored
SMULL/SMLAL reduction without repeating the previously rejected one-pass
layout. Rounded filter deltas fit signed bytes (weighted sum bounded by
228), allowing signed saturating byte addition after range translation.
Matched Orin medians: direction 82.73 -> 77.58 ns, picture-stride primary
filter 83.06 -> 79.82 ns; both-strength filtering 203.35 -> 205.62 ns (mixed,
unchanged off control 20.31 -> 20.90 ns). Whole Veryfast cycles -0.597%,
Faster +0.212% (level), with instructions -0.129/-0.105%; all streams
identical. The older full audit above remains its original frozen
measurement. See the [follow-up receipt](../handoff/astra-kernels-20261117.md#updated-task-2-av1-arm-priority-1-cdef-2026-11-17)
for five-round diagnosis comparisons, every loss and validation status.


### Full dav1d NEON CDEF follow-up (2026-11-17)

The shared AV1/AV2 eight-bit NEON path ports dav1d 1.5.4's byte
constraints, split bounded byte accumulators, sentinel padding and
one-pass direction graph, with BSD attribution. Assembly tuning borrows
validated source windows, specializes directions, separates strength
frames and packs direction reductions. High depth and every ISA stay.

Against `07615e9b5` on the Jetson, direction is 77.65 → 74.07 ns; 8x8 both
204.87 → 188.52 ns, primary 79.49 → 72.87 ns, secondary 120.36 → 113.88 ns.
4x4 both is 69.14 → 67.45 ns, primary 34.97 → 35.40 ns, secondary
46.72 → 47.38 ns; retain the small single-strength losses. Release
callgrind 8x8 both is 941 → 858 Ir/call; 4x4 both is 340 → 368 despite
its slightly better native time.

Five quiet paired whole-encode rounds of shipped AV1 presets against
`07615e9b5`: Veryfast cycles −0.264%, Faster −0.899%; instructions
−0.318/−0.302%, all 48 streams identical. These modest gains leave further
CDEF opportunity. See the [full receipt](../handoff/astra-kernels-20261117.md#full-neon-cdef-landing-candidate-0c6b1add8)
for the matched dav1d comparison, retained 4x4 costs and validation.
