# Third-party code and data

nanocodec reimplements algorithms from permissively licensed projects. The
modules that do so name their source in their module documentation.

## libaom — BSD-2-Clause

Copyright (c) 2016, Alliance for Open Media. All rights reserved.
Source: https://aomedia.googlesource.com/aom (pinned v3.15.3, see
`tools/anchors/versions.env`).

Used in `crates/nc-av1`:
- `src/tables/generated.rs`: default CDFs, quantiser lookup tables, scan
  orders and coefficient context offsets. Extracted mechanically by
  `tools/codegen/gen_av1_tables.py`.
- `src/range_coder.rs`: the `od_ec` range encoder.
- `src/cdf.rs`: CDF adaptation (`update_cdf`) and the symbol cost
  approximation (`av1_cost_symbol`).
- `src/mvcoding.rs`: motion-vector difference coding and default CDFs.
- `src/coeff.rs`: coefficient coding and context derivation.
- `src/cfl.rs`: chroma-from-luma subsampling and signed rounding from
  `av1/common/cfl.c` and `cfl.h`.
- `src/backend.rs`: partition syntax including edge handling.
- `src/predict.rs`: intra prediction formulas, edge extension and smooth
  weights from `aom_dsp/intrapred.c`, `intrapred_common.h` and
  `av1/common/reconintra.c`.
- `src/loopfilter.rs`: deblocking edge selection, filter length and level
  thresholds (`av1/common/av1_loopfilter.c`) and the pick-from-Q level fit
  (`av1/encoder/picklpf.c`).
- `src/grain.rs`: the film grain seed sequence
  (`av1/encoder/encoder_utils.c`), the AR coefficient shift choice, chroma
  scaling as a function of luma and the luma-correlation rescaling of
  `aom_noise_model_get_grain_parameters` (`aom_dsp/noise_model.c`).
- `src/restoration.rs`: the encoder's Wiener loop restoration search
  (`av1/encoder/pickrst.c`): the separable symmetric decomposition
  (`wiener_decompose_sep_sym`), tap quantization, the refinement around the
  quantized taps (`finer_tile_search_wiener`) and the per-unit RD switch with
  taps coded against the previous unit's (`count_wiener_bits`), reimplemented
  on statistics folded by the filter's symmetry.

Used in `crates/nc-analysis`:
- `src/grain.rs`: the noise model design of `aom_dsp/noise_model.c` (source
  noise against a denoised source on flat blocks found after a plane fit, AR
  fit of that noise, an extra chroma term on co-located luma noise, noise
  strength per intensity bin), reimplemented with a 3x3 blur as the
  denoiser and Yule-Walker equations for the AR fit.

Used in `crates/nc-dsp`:
- `src/txfm/dct.rs`: the 1-D DCT butterflies (`av1_fdct4..64`,
  `av1_idct4..64`), translated mechanically by `tools/codegen/gen_dsp_dct.py`;
  `src/txfm/generic.rs`: the 2-D shift and cosine-bit configuration of
  `fwd_txfm2d` / `inv_txfm2d_add_c`.
- `src/intra_prediction/av1_block/`: intra edge filter strengths and
  kernels, edge upsampling, corner filter, rectangular DC / smooth / Paeth,
  directional zones with upsampled edges
  (`av1/common/reconintra.{c,h}`, `aom_dsp/intrapred.c`), and 4:2:0 chroma
  from luma subsampling, padding, averaging and scaling
  (`av1/common/cfl.{c,h}`). Golden hashes in
  `crates/nc-checkasm/tests/av1_block_intra_golden.rs` come from libaom's
  `reconintra.c` builders and `cfl.c`.
- `src/av1_inverse/`: the inverse ADST4/8/16 butterflies, identity scales,
  sine constants (`av1/common/av1_inv_txfm1d.c`, `av1_txfm.c`) and the 2-D
  rounding, clamping, flips and 64-point zero-out of `inv_txfm2d_add_c`
  with its per-size row shifts (`av1/common/av1_inv_txfm2d.c`). Golden
  hashes in `crates/nc-checkasm/tests/av1_inverse_golden.rs` come from
  libaom's `av1_inv_txfm2d_add_*_c` and `av1_highbd_iwht4x4_16_add_c`.
- `src/mc/mod.rs`: the regular `EIGHTTAP` interpolation filters
  (`av1/common/filter.h`); `src/diff/mod.rs`: the bilinear sub-pixel
  variance definition (`aom_dsp/variance.c`).
- `src/av1_mc/`: AV1 inter prediction: the `BILINEAR` and 4-tap filter
  selection (`av1/common/filter.h`), the wedge master masks, codebooks,
  smooth inter-intra weights and OBMC masks (`av1/common/reconinter.c`,
  `tools/gen_wedge_masks_data.py`), and the warp filter taps, `div_lut`,
  shear setup and least-squares warp estimation
  (`av1/common/warped_motion.c`), reimplemented from the AV1 spec
  (§7.11.3) and checked bit-exact against libaom's C kernels.
- `src/av1_restoration/`: AV1 loop restoration and superres: the Wiener
  rounding (`av1_wiener_convolve_add_src_c`), the self-guided filter
  (`av1_selfguided_restoration_c`, `av1_sgr_params`, `av1_x_by_xplus1`,
  `av1_one_by_x`, `av1_decode_xq`), unit and stripe layout and boundary
  rows (`av1/common/restoration.c`), and `av1_resize_filter_normative`
  with the upscale step and offset (`av1/common/resize.c`), checked
  bit-exact against libaom.
- `src/av1_film_grain/`: AV1 film grain synthesis: `gaussian_sequence`,
  the grain template sizes and seeds, and the scaling lookup
  (`av1/decoder/grain_synthesis.c`), structured after the AV1 spec
  (§7.18.3) and checked bit-exact against `av1_add_film_grain`.
- `src/cdef/`: the CDEF block filter (`cdef_filter_block_internal`) and
  direction search (`cdef_find_dir_c`, and the partial-sum formulation of
  its SIMD version) from `av1/common/cdef_block.c` and
  `cdef_block_simd.h`.
- `src/intra_prediction/directional.rs`: the directional intra derivative
  table and zone rules (`dr_intra_derivative`,
  `av1_dr_prediction_z{1,2,3}_c` in `av1/common/reconintra.{c,h}`);
  `src/intra_prediction/filter.rs`: the filter intra taps
  (`av1_filter_intra_taps`, `av1/common/reconintra.c`).
- `src/deblock/av1.rs`: the loop filter masks and 4/6/8/14-tap filters
  (`aom_dsp/loopfilter.c`).

Used in `crates/nc-encoder`:
- `src/cdef_search.rs`: the greedy CDEF strength-set selection idea of
  `joint_strength_search` (`av1/encoder/pickcdef.c`), reimplemented.

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

The AV1 bitstream format is covered by the Alliance for Open Media Patent
License 1.0.

## kvazaar — BSD-3-Clause

Copyright (c) 2021, Tampere University, ITU/ISO/IEC, project contributors.
All rights reserved.
Source: https://github.com/ultravideo/kvazaar (v2.3.2, commit 1f11f9a).

Used in `crates/nc-hevc`:
- `src/cabac.rs`: the CABAC encoder engine and its state transition and
  `rangeTabLps` tables.
- `src/contexts.rs`: context initialisation values for I slices.
- `src/residual.rs`: the structure of `residual_coding()` and its context
  selection.
- `src/transform.rs`: the values of the 32-point DCT matrix (they are the
  HEVC specification's `transMatrix`).

Used in `crates/nc-dsp`: `src/codec_txfm/mod.rs` holds a copy of the same
32-point matrix values.

> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice,
>   this list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the name of the Tampere University or ITU/ISO/IEC nor the names of
>   its contributors may be used to endorse or promote products derived from
>   this software without specific prior written permission.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
> ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
> LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
> CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
> SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
> INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
> CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

HEVC is covered by patents licensed through pools (Access Advance, Via LA);
this licence grants no patent rights.

## openh264 — BSD-2-Clause

Copyright (c) 2013, Cisco Systems. All rights reserved.
Source: https://github.com/cisco/openh264 (pinned v2.7.0, see
`tools/anchors/versions.env`).

Used in `crates/nc-h264`:
- `src/tables/generated.rs`: CABAC context initialisation values, the LPS
  range table and the state transition table (H.264 Tables 9-12..9-33,
  9-44, 9-45). Extracted mechanically by `tools/codegen/gen_h264_tables.py`.
- `src/tables/cavlc.rs`: the decoder's CAVLC code tables (H.264 Tables 9-5,
  9-7, 9-8, 9-9a, 9-10) as (code, length) pairs from openh264's encoder
  tables, and the `coded_block_pattern` mapping (Table 9-4) from its decoder
  tables. Extracted mechanically by `tools/codegen/gen_h264_cavlc_tables.py`
  from an openh264 v2.4.1 tree (normative tables, unchanged since).
- `src/decoder/{vlc,cavlc}.rs`: the decoder's `residual_block_cavlc()`
  follows openh264's `WelsResidualBlockCavlc` design (algorithm guide,
  reimplemented): each block parsed from a local bit window with one
  end-of-data check, every code one table lookup, trailing-one signs read
  together, branch-free level arithmetic.

The licence text is the BSD-2-Clause text reproduced under libaom above,
with "Cisco Systems" as the copyright holder.

## SVT-AV1 — BSD-3-Clause-Clear

Source: https://gitlab.com/AOMediaCodec/SVT-AV1 (pinned v4.3.0,
commit e3d63405f9ba8219913a7276c2c47d10e5d022fe; see `tools/anchors/versions.env`).

Used as an algorithm guide and reimplemented in:

- `crates/nc-analysis`: real-time pre-analysis and hierarchical motion search.
- `crates/nc-dsp`: EOB-derived sparse inverse dispatch (algorithm guide:
  `ASM_AVX2/av1_inv_txfm_avx2.c` low8/16/32 tables); butterfly arithmetic is
  generated from the libaom normative reference credited above.
- `crates/nc-encoder`: variance partition planning and light mode decision,
  including `product_coding_loop.c:md_subpel_search_fixed_stage` coordinate
  search and initial-offset bias; the CDEF quantiser strength model
  (`svt_pick_cdef_from_qp`, `cdef_model.rs`) and the light-PD1
  pre-transform skip score (`lpd1_should_perform_tx`, `pre_tx_skip.rs`),
  shared by nc-av1 and nc-av2.
- `crates/nc-av1`: keyframe quantizer scaling, light residual decisions
  (`lpd1_should_perform_tx` score controls), CDEF search cadence,
  sampled fixed-point SUB picture variance (`pic_analysis_process.c`), and
  transform-domain luma RD gating (`product_coding_loop.c` light-PD1).
- `crates/nc-av1/src/restoration.rs`: the real-time Wiener restrictions of
  `WnFilterCtrls` (`enc_mode_config.c`: fewer taps, luma only, no
  refinement, restoration only on frames other frames reference).

The pinned source LICENSE.md states:

> BSD 3-Clause Clear License
> The Clear BSD License
> 
> Copyright (c) 2021, Alliance for Open Media
> 
> All rights reserved.
> 
> Redistribution and use in source and binary forms, with or without
> modification, are permitted (subject to the limitations in the disclaimer below)
> provided that the following conditions are met:
> 
> 1. Redistributions of source code must retain the above copyright
>    notice, this list of conditions and the following disclaimer.
> 
> 2. Redistributions in binary form must reproduce the above copyright
>    notice, this list of conditions and the following disclaimer in
>    the documentation and/or other materials provided with the distribution.
> 
> 3. Neither the name of the Alliance for Open Media nor the names of its
>    contributors may be used to endorse or promote products derived from
>    this software without specific prior written permission.
> 
> NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE.
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
> EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
> OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
> THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
> SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
> OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
> HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## Thor — BSD-2-Clause

Copyright (c) 2015, Cisco Systems. All rights reserved.
Source: https://github.com/cisco/thor.
Full notice: [LICENSE](docs/licenses/thor/LICENSE).

Ideas only, reimplemented (no code copied):
- `crates/nc-encoder/src/zero_residual.rs`: the early-skip principle of
  `enc/encode_block.c` (`check_early_skip_sub_block`,
  `check_early_skip_transform_coeff`): judge each sub-block's residual
  against a fraction of the first quantiser level, with low frequencies
  taken from sub-block averages. The same module also follows libaom's
  `predict_skip_txfm` (`av1/encoder/tx_search.c`), quantiser-relative
  skip prediction.

## dav1d — BSD-2-Clause

AV1 film-grain row scaling, chroma indexing and overlap arithmetic in
`nc-dsp/src/av1_film_grain/kernels` are independently reimplemented from
dav1d 1.5.4 `src/filmgrain_tmpl.c` (2018–2025 VideoLAN and dav1d authors).
The SIMD implementation uses our shared lane framework.

Copyright (c) 2018, VideoLAN and dav1d authors. All rights reserved.
Source: https://code.videolan.org/videolan/dav1d
Full notice: [COPYING](docs/licenses/dav1d/COPYING).

The AV1 decoder's coefficient level pass
(`crates/nc-av1/src/decoder/coefficients/levels.rs`) is a port, in Rust, of
the level loop of dav1d 1.5.4 `src/recon_tmpl.c` (`decode_coefs`,
`get_lo_ctx`): the level-buffer layout and fixed-offset context neighbours.

The AV1 symbol decoder (`crates/nc-av1/src/decoder_range.rs`: a 64-bit
window, renormalisation by one shift, refills of whole bytes) and its
symbol kernels (`crates/nc-dsp/src/av1_symbol`: every interval bound and
the CDF adaptation in 16-bit lanes, the decoded symbol counted from the
comparison mask, the interval read back from the stored bounds) follow
the design of dav1d 1.5.4 `src/msac.c` and `src/x86/msac.asm`, and are
reimplemented in Rust.

CDEF (`crates/nc-dsp/src/cdef/band.rs`, `cdef/direction_avx2.rs`) is
reimplemented in Rust from dav1d 1.5.4 `src/cdef_apply_tmpl.c` (`cdef_brow`:
in-place filtering of a picture by bands with unfiltered line buffers, each
8x8 unit's direction, luma and chroma together, only coded units of
filter blocks with a strength set, `adjust_strength`) and
`src/x86/cdef_avx2.asm` (`cdef_dir_8bpc`: rows and their mirror order in
the two halves of a 256-bit register, mirrored line pairs squared by one
`pmaddwd`, the best direction by `phminposuw`). Copyright © 2018 VideoLAN
and dav1d authors; Copyright © 2018 Two Orioles, LLC.

The AVX-512 interior CDEF filter (`crates/nc-dsp/src/cdef/avx512.rs`)
widens the existing packed-byte AVX2 schedule inspired by dav1d/SVT-AV1:
constrained signed byte differences, paired `pmaddubsw` tap weighting, and
exact word rounding. It uses eight independent rows per vector with the
existing scalar CDEF equations and oracle.

AV1 loop restoration (`crates/nc-dsp/src/av1_restoration/`: `mod.rs`,
`wiener.rs`, `self_guided.rs`, `sgr_lanes.rs`, `avx2.rs`, `neon.rs`,
`simd128.rs`) ports dav1d 1.5.4 `src/lr_apply_tmpl.c`,
`src/looprestoration_tmpl.c`, `src/x86/looprestoration_avx2.asm` and
`src/arm/64/looprestoration.S`: in-place restoration by stripe, the Wiener
ring of 16-bit horizontal rows, the 8-bit Wiener arithmetic (byte-pair
`pmaddubsw` / 16-bit `mla`, the clamp as a saturating add with dav1d's
offsets, vertical pair sums with `pmaddwd`) and the self-guided row
schedule (horizontal box sums, box statistics, filtered rows), reimplemented
in Rust over our lane types.
The AV1 decoder's motion field projection
(`crates/nc-av1/src/decoder/motion_field.rs`: the field built one
superblock row at a time into a one-row buffer, per-reference span tables,
runs of equal kept vectors projected once) follows dav1d 1.5.4
`src/refmvs.c` (`load_tmvs`, `dav1d_refmvs_init_frame`), reimplemented in
Rust.

Structure only, no code: `crates/nc-dsp/src/mc/separable_simd128.rs` (the SIMD128 HEVC/H.266
separable prediction) follows the pass structure of dav1d 1.5.4 `put_8tap` as `mc/av1_simd128.rs`
does (rolling horizontal rows per strip, vertical pass on interleaved row pairs against tap pairs),
with HEVC's arithmetic.

AV1 warped motion (`crates/nc-dsp/src/av1_mc/warp_kernels.rs`) follows
dav1d 1.5.4 `warp_affine_8x8_c` (`src/mc_tmpl.c`): an 8x8 block per call
from its 15x15 source window, a 16-bit horizontal intermediate and a
vertical pass, each output with its own filter phase; reimplemented in Rust
over our lane types; the AVX2 kernel (`av1_mc/warp_avx2.rs`) follows
dav1d's `warp_affine_8x8_8bpc_avx2` (`src/x86/mc_avx2.asm`): byte-pair
multiply-adds over taps reordered 0, 2, 4, 6, 1, 3, 5, 7. The baseline NEON
kernel (`av1_mc/warp_neon.rs`) follows `warp_filter_horz_neon` and
`warp_affine_8x8_8bpc_neon` (`src/arm/64/mc.S`) and `transpose_8x8b_xtl`
(`src/arm/64/util.S`): centered signed-byte horizontal products and pairwise
reductions, a rolling row window and transposed coefficients for vertical
widening multiply-adds. BSD-2-Clause; Copyright 2018 VideoLAN/dav1d authors,
Janne Grunau and Martin Storsjo. Our kernel restores the horizontal centering
bias before the vertical pass and retains variable-shift i32 outputs.
The masked compound blend (`av1_mc/mask_kernels.rs`)
follows dav1d's `mask_c`: predictions and weight pairs interleaved for one
multiply-add per pair.

Ideas only, no code: `crates/nc-dsp/src/txfm/narrow_inverse.rs` (with its
AVX2, NEON and SIMD128 lanes; the AVX2 lanes follow the `pmaddwd` /
`pmulhrsw` choices of `src/x86/itx_avx2.asm`) keeps 8-bit
AV1 inverse-transform intermediates in 16-bit lanes, as dav1d's
`itx` assembly does. nanocodec adds a proved L1 guard with an exact 32-bit
fallback instead of relying on the conformance range.

Ideas only, no code: `crates/nc-dsp/src/av1_inverse/` uses the DCT DC-only
shortcut and EOB-bounded row-pass design described in dav1d 1.5.4
`src/itx_tmpl.c`. The Rust implementation retains nanocodec's Q12 constants,
rounding, shared lane types and coefficient layout.

Layout reference, no assembly copied:
`crates/nc-dsp/src/lpf_bytes/neon_wide.rs` reviews dav1d 1.5.4
`src/arm/64/loopfilter.S` for sixteen byte-mask lanes with paired eight-word
smoothing lanes, and `src/arm/64/util.S` (`transpose_8x16b`) for the
byte/word/dword transpose stages. The Rust adapter retains nanocodec's
shared AV1 filter arithmetic. These BSD-2-Clause sources credit
VideoLAN/dav1d authors (2018), Martin Storsjo (2015 in `util.S`, 2019 in
`loopfilter.S`), and Janne Grunau (2015 in `util.S`).

Ideas only, no code: `crates/nc-dsp/src/constant_add/` adds a DC-only
residual to 8-bit samples as a saturating byte add and subtract, as the DC
paths of dav1d 1.5.4's `src/x86/itx_avx2.asm` do.

## libyuv — BSD-3-Clause

Copyright 2011 The LibYuv Project Authors. All rights reserved.
Source: https://chromium.googlesource.com/libyuv/libyuv
Full notice: [LICENSE](docs/licenses/libyuv/LICENSE).

Used in `crates/nc-dsp`:
- `src/convert/to_rgba_avx2.rs`, `to_rgba_neon.rs`, `to_rgba_simd128.rs`:
  the I420 to RGBA row-function structure of libyuv's `I422ToARGBRow`
  family (chroma terms shared by a pixel pair, channels in 16-bit lanes
  until one saturating pack, then byte and 16-bit interleaves). The
  arithmetic is our own exact 8.8 BT.601 form, not libyuv's 6-bit one.
- `src/convert/rgba_avx2.rs`: the RGBA to I420 approach of libyuv's
  `ARGBToYRow_AVX2` / `ARGBToUVRow_AVX2` (byte-weight `pmaddubsw` and
  horizontal adds on packed pixels, no planar transpose), with our exact
  weights and exact 2x2 means.

## wasm-bindgen — MIT OR Apache-2.0

The `nc-web` browser binding crate uses wasm-bindgen0.2.105 and its generated
JavaScript/WASM interoperability code. Source and license texts:
https://github.com/wasm-bindgen/wasm-bindgen/tree/0.2.105
The encoder and decoder algorithms remain in the nanocodec Rust crates.


## libvpx — BSD-3-Clause

Copyright (c) 2010 The WebM Project authors. All rights reserved.
Source: https://github.com/webmproject/libvpx, v1.16.0,
commit `1024874c5919305883187e2953de8fcb4c3d7fa6`.

`crates/nc-vp9/src/bool_coder.rs` reimplements the byte-emission algorithm
from `vpx_dsp/bitwriter.h` and `bitwriter.c`. Its bounded reader follows the
VP9 boolean decoding specification. Keyframe header syntax was checked
against `vp9/decoder/vp9_decodeframe.c`.
`crates/nc-vp9/src/tables.rs` contains generated default intra coefficient,
partition/mode probabilities, scan/context tables, Pareto token models and
8-bit quantizer lookup tables from `vp9/common/`. Regenerate using
`tools/codegen/gen_vp9_tables.py` against the pinned source.
`coeff.rs` implements coefficient token processes cross-checked against
`vp9/decoder/vp9_detokenize.c`. `lossless.rs` and `dct.rs` reimplement the
4x4 Walsh-Hadamard/DCT transforms from `vp9/encoder/vp9_dct.c`,
`vpx_dsp/fwd_txfm.c` and `vpx_dsp/inv_txfm.c`. `intra.rs` implements
partition/intra syntax and context evolution checked against
`vp9/encoder/vp9_bitstream.c` and the decoder.
`tools/codegen/gen_vp9_bool_vector.py` compiles original upstream C routines
as a temporary external oracle; the resulting test vector is retained.
`motion.rs` reimplements default-context motion-vector difference coding from
`vp9_entropymv.c`, `vp9_encodemv.c` and `vp9_decodemv.c`, and bilinear
interpolation from `vp9_filter.c` and `vpx_dsp/vpx_convolve.c`.
`hybrid.rs` reimplements 4x4 ADST/DCT pairs from `vp9/encoder/vp9_dct.c`,
`vp9/common/vp9_idct.c` and `vpx_dsp/inv_txfm.c`.
`prediction.rs` implements DC, vertical, horizontal and true-motion prediction
and missing-neighbor rules from `vpx_dsp/intrapred.c` and `vp9_reconintra.c`.
`filter.rs` reimplements the scalar filter4/filter8 taps, thresholds and
fixed-8x8/TX4 edge scheduling from `vpx_dsp/loopfilter.c` and
`vp9/common/vp9_loopfilter.c`.
The full Profile 0 decoder in `crates/nc-vp9/src/decoder/` reimplements the
decoding processes of `vp9/decoder/vp9_decodeframe.c` (headers, tiles,
partitions, prediction and reconstruction), `vp9_decodemv.c` (mode info and
candidate vectors), `vp9_detokenize.c`, `vp9_decoder.c` / `vp9_dx_iface.c`
(reference slots, superframes), `vp9/common/vp9_pred_common.[ch]`,
`vp9_mvref_common.h`, `vp9_entropymode.c` (past independence, adaptation),
`vp9_reconintra.c` (frame-edge rules), `vp9_scale.c` and
`vp9_loopfilter.c` (`vp9_loop_filter_frame_init`, the per-superblock bit
masks of `vp9_build_mask` / `vp9_adjust_mask` and their tables); `nc-dsp`'s
`vp9_convolve_scaled` reimplements `vpx_dsp/vpx_convolve.c` `convolve`, and
`vp9_mc_scaled.rs` follows `vpx_scaled_2d`'s two-pass structure. `nc-dsp`'s VP9 inverse
transform + add (`vp9_inverse/`) follows the eob-class structure of
`vpx_dsp/inv_txfm.c` and its SIMD versions (`vpx_idct*_1_add`, `_12`/`_10`/
`_38`/`_34`/`_135` partial transforms, doubled-cosine `mulhrs` rotations);
`txfm/sparse_lanes.rs` derives those partial graphs. The AVX2 VP9 vertical
interpolation (`vp9_mc_simd/avx2.rs`) follows
`vpx_dsp/x86/vpx_subpixel_8t_intrin_avx2.c` (`vpx_filter_block1d*_v8_avx2`).
The encoder's real-time shortcuts in `crates/nc-vp9/src/backend.rs`
(`vp9.nonrd_skip_scale`, `vp9.residual_bits_per_level`) reimplement the
transform-block skip thresholds of `model_rd_for_sb_y`, the DC-only path of
`vp9_xform_quant_dc` / `vpx_fdctNxN_1`, and the level-magnitude rate of
`block_yrd` from `vp9/encoder/vp9_pickmode.c` and `vp9_encodemb.c`.
`tools/codegen/check_vp9_kernels.py` verifies these modules' compact golden
vectors/hashes against original C functions extracted from the pinned commit.
It compiles temporary external oracles, reads expected values directly from
the Rust tests and retains neither reference binaries nor raw video. Reference
implementations are not linked into nanocodec.

Full notices: [LICENSE](docs/licenses/libvpx/LICENSE) and
[separate patent grant](docs/licenses/libvpx/PATENTS). Preserve these with
source and binary distributions that include the derived implementation.

## AVM — BSD-3-Clause-Clear; AOM Patent License 1.0

Copyright (c) 2021, Alliance for Open Media. All rights reserved.
Source: https://github.com/AOMediaCodec/avm, v1.0.0,
commit `966a7d7cd6fcf60360caf5dc413b2aeeb65e144d`.

`crates/nc-av2/src/obu.rs` implements final AV2 framing from specification
§5.2 / §6.2, cross-checked against `av2/common/obu_util.c` and the OBU header
writer in `av2/encoder/bitstream.c`. AVM remains an external reference tool,
not a linked nanocodec dependency. `tools/dev/av2_obu_smoke.py` compiles the
original OBU parser, bit-buffer reader and integer/LEB128 helpers in temporary
storage as an external differential oracle; no compiled reference code is shipped
in the Rust implementation.

Additional native AV2 modules reference the same pinned release:
- `src/sequence.rs` and `src/frame_header.rs`: final sequence tool groups and
  closed-loop keyframe prefixes, audited against `av2/encoder/bitstream.c`
  and the decoder's `obu.c` / `decodeframe.c`.
- `src/entropy.rs`: range normalization/termination, AV2 probability scaling,
  true bypass coding and trained-rate CDF updates from
  `avm_dsp/{entenc.c,entdec.c,entcode.c,prob.h}`, independently structured in Rust.
- `src/coeff.rs` (with `src/coeff/`), `src/tx_size.rs`, `src/coeff_tables.rs`,
  `src/dc_tile.rs` and `examples/gray_keyframe.rs`: TX4..TX64 coefficient
  contexts and scans, low/high-range coding and selected default probabilities
  from `av2/common/entropy_inits_{modes,coeffs}.h`, `av2/common/txb_common.h`,
  `av2/common/scan.c`, `av2/common/hr_coding.c`, `av2/encoder/encodetxb.c` and
  `av2/decoder/decodetxb.c`. `examples/gen_coeff_tables.py` reproducibly extracts
  the fixed-CDF coefficient subset from pinned original headers.
  `tools/codegen/check_av2_coeff.py` links the pinned AVM library into a
  temporary test program (contexts and decoding of `examples/coeff_vectors.rs`
  output); nothing from it is shipped.
- `examples/sequence_oracle.c`, `examples/entropy_oracle.c` and the sequence
  oracle Python driver call/compile original reference routines solely for
  differential tests. Golden sequence/entropy outputs in `tests/data` are
  generated by those original routines; no reference binaries are shipped.
- `src/transform.rs`, `src/partition.rs`, `src/intra_modes.rs`, `src/motion.rs`,
  `src/mv_refs.rs` and `src/prediction.rs`: transform arithmetic, partition and
  mode syntax, motion-vector coding/reference derivation and bilinear rounding
  checked against the pinned AVM implementation. The corresponding
  `tools/codegen/check_av2_*.py` drivers compile original C routines as external
  differential oracles. Compatible prediction and reversible-transform kernels
  are shared through `nc-dsp`; AV2-specific rounding remains explicit.
- `src/intra_edges.rs` and `nc-dsp/src/intra_prediction/av2.rs`: intra edge
  availability (`has_top_right`, `has_bottom_left`), edge construction and the
  DC, smooth, Paeth and directional (2-tap chroma, 4-tap luma) predictors of
  `av2/common/reconintra.c` and `avm_dsp/intrapred.c`, reimplemented in Rust.
- `nc-dsp/src/av2_txfm/`: the square `DCT_DCT` kernels (4×4..64×64) of
  `av2/common/idct.c` (`inv_txfm_c`), `av2/encoder/hybrid_fwd_txfm.c`
  (`fwd_txfm_c`), `av2/common/txb_common.c` and `av2/common/common_data.h`,
  restructured as one even/odd butterfly generated from a 64-entry cosine
  table. `tools/codegen/check_av2_dct.py` compiles the original 2-D drivers,
  kernels and decoder dequantization as the differential oracle.
  `nc-dsp/src/av2_txfm/typed.rs` and `adst_tables.rs` (generated): the
  identity, ADST (tuned DST-VII) and flipped-ADST kernels of the same files
  for the other transform types, and `src/tx_type.rs`: `TX_TYPE`, the
  reduced transform sets and `av2_read_tx_type` of `av2/common/blockd.h`,
  `entropymode.h` and `av2/decoder/decodemv.c`.
  `tools/codegen/check_av2_tx_types.py` regenerates the tables and checks every
  (size, type) pair against the original 2-D drivers.

- `src/deblock.rs` (with `src/deblock/`) and `nc-dsp/src/deblock/av2.rs`: the
  deblocking filter of `av2/common/av2_loopfilter.c` (`df_quant_from_qindex`,
  `df_side_from_qindex` and its `side_thresholds` table, edge and filter-length
  selection of `set_lpf_parameters`), the per-segment filter of
  `avm_dsp/loopfilter.c` (`filt_choice_highbd`, `filt_generic_asym_highbd`) and
  the `setup_loopfilter` header syntax, reimplemented and checked bit-exact
  against avmdec and dav2d on encoded streams.

- `src/coeff/tcq.rs`: trellis-coded quantisation of `av2/common/quant_common.c`
  (`tcq_init_state`, `tcq_next_state`, `tcq_quant`), `quant_common.h`
  (`tcq_enable`) and the TCQ high-range and dequantisation paths of
  `av2/decoder/decodetxb.c`, reimplemented in Rust; the encoder's 8-state
  Viterbi search is our own. Checked against avmdec and dav2d on encoded
  streams.

- `src/cdef.rs`: CDEF of `av2/common/cdef.c` and `cdef_block.c`
  (`av2_cdef_compute_sb_list`, `is_8x8_block_skip`, `adjust_strength`,
  `av2_cdef_filter_fb`) and the `setup_cdef` header syntax, on nc-dsp's
  AV1-derived CDEF kernels (bit-exact for 8-bit video), checked against
  avmdec and dav2d on encoded streams.

- `crates/nc-av2/src/coeff/levels.rs`: the default scans of
  `av2/common/scan.c` and the level-context helpers of `txb_common.h`
  (`get_lower_levels_ctx_*`, `get_br_ctx_*`, `get_lp2tx_ctx` and their
  chroma variants), reimplemented in Rust.
- `crates/nc-av2/src/tx_partition.rs`: the square-block transform layout of
  `TX_PARTITION_SPLIT`, `av2_get_tx_size`, `get_lossless_tx_size` and the
  coding order of `decode_token_recon_block`.

Full notices: [LICENSE](docs/licenses/avm/LICENSE) and
[separate patent license](docs/licenses/avm/PATENTS). The Clear BSD software
license does not itself provide a patent grant.

## dav2d — BSD-2-Clause (external reference only)

Copyright © 2018–2026, VideoLAN and dav2d authors.
Source: https://code.videolan.org/videolan/dav2d,
commit `237229ebfefd4eb8e0e25ad9636803bb93bae0c6`.

The independent AV2 conformance executable is built with scalar 8-bit support.
Its copyright and separate patent documents were inspected at this exact pin:
[COPYING](docs/licenses/dav2d/COPYING), [PATENTS](docs/licenses/dav2d/PATENTS).
No dav2d implementation is linked into nanocodec. Actual emitted-subset picture
checks, rather than draft-era README wording, determine compatibility with AVM.

## VVenC — BSD-3-Clause-Clear

Source: https://github.com/fraunhoferhhi/vvenc, commit `a455d51e335c97b40eeea7bb90e3ad52b84c6e83`.

Used as an algorithm/syntax reference in `crates/nc-vvc`: `headers.rs`
(VLCWriter), `contexts.rs` (I/P-slice initialization tables and the dual-window
probability estimator), `syntax.rs`, `coeff.rs` and `coeff_contexts.rs` (CU and
full residual binarization/context tables), `motion.rs` (UnitTools spatial/history
merge and AMVP derivation), `intra.rs` (reference substitution, planar/DC/axis
prediction and PDPC), `recon.rs` (quantization), `filter.rs` and
`nc-dsp/src/vvc_deblock/` (LoopFilter thresholds and short/long deblocking),
`nc-dsp/src/vvc_intra/` (planar/DC/all square angular prediction, PDPC,
angle/inverse-angle tables and InterpolationFilter cubic coefficients, scalar/SIMD;
`angular_generic.rs` uses the Q15 PDPC blend from
`source/Lib/CommonLib/x86/IntraPredX86.h::pdpcBlend`),
and `sao.rs` (edge-offset signalling).
`nc-dsp/src/codec_txfm/vvc_forward_narrow.rs` follows its rectangular pass
scaling while reusing our exact HEVC line graphs for narrow strips. Reimplemented as a restricted
native Rust codec; VVenC is not linked or called by the encoder. The arithmetic
byte-output machinery in `cabac.rs` is adapted from our kvazaar-derived HEVC
coder under the BSD-3-Clause notice above. VVdeC is also a syntax reference for decoder
header parsing (see below).

> The Clear BSD License
>
> Copyright (c) 2019-2026, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVenC Authors.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification,
> are permitted (subject to the limitations in the disclaimer below) provided that
> the following conditions are met:
>
>      * Redistributions of source code must retain the above copyright notice,
>      this list of conditions and the following disclaimer.
>
>      * Redistributions in binary form must reproduce the above copyright
>      notice, this list of conditions and the following disclaimer in the
>      documentation and/or other materials provided with the distribution.
>
>      * Neither the name of the copyright holder nor the names of its
>      contributors may be used to endorse or promote products derived from this
>      software without specific prior written permission.
>
> NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
> THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
> CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
> PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
> CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
> EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
> PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
> BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
> IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.

## VVdeC — BSD-3-Clause-Clear

Source: https://github.com/fraunhoferhhi/vvdec, commit `d2614690cf13ceee1631dd6d30141755b8a689de`.

`crates/nc-vvc/src/decoder_headers/` reimplements H.266 parameter-set, picture and slice syntax
with bounded Rust readers, using `HLSyntaxReader` as a syntax reference.
`crates/nc-dsp/src/mc/generic.rs` (`hevc_bi`, HEVC default bi-prediction) averages the two
predictions as a paired 32-bit sum, the layout of VVdeC's `addAvg_SSE` (`BufferX86.h`).
VVdeC is not linked or called by our decoder; its executable remains an
independent conformance oracle.

> The Clear BSD License
>
> Copyright (c) 2018-2026, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The VVdeC Authors.
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without modification,
> are permitted (subject to the limitations in the disclaimer below) provided that
> the following conditions are met:
>
>      * Redistributions of source code must retain the above copyright notice,
>      this list of conditions and the following disclaimer.
>
>      * Redistributions in binary form must reproduce the above copyright
>      notice, this list of conditions and the following disclaimer in the
>      documentation and/or other materials provided with the distribution.
>
>      * Neither the name of the copyright holder nor the names of its
>      contributors may be used to endorse or promote products derived from this
>      software without specific prior written permission.
>
> NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
> THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
> CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
> LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
> PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
> CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
> EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
> PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
> BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
> IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
> ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
> POSSIBILITY OF SUCH DAMAGE.


`crates/nc-vvc/src/decoder_blocks/` adds coding-tree partitioning, rectangular
residual syntax and dependent quantization, with `CABACReader`, `UnitPartitioner`
and `ContextModelling` as syntax references under the same license.

`crates/nc-dsp/src/codec_txfm/vvc_inverse.rs` derives rectangular and dependent
inverse-scaling parameters from the same BSD-Clear VVdeC `Quant` reference;
it reuses the existing HEVC scaling and DCT-II kernels.

`crates/nc-vvc/src/affine.rs` and `affine_candidates.rs` reimplement affine
motion (sub-block vectors and their bandwidth fallback, sub-block prediction,
the sub-block merge and affine AMVP candidate lists) with `UnitTools`
(`setAllAffineMv`, `getAffineMergeCand`, `fillAffineMvpCand`,
`xInheritedAffineMv`) and `InterPrediction::xPredAffineBlk` as references,
the H.266 luma, affine luma and chroma interpolation taps in
`crates/nc-dsp/src/mc/mod.rs` come from `InterpolationFilter`, and the
affine-aware deblocking filter lengths and the 5-sample long filter in
`crates/nc-vvc/src/filter.rs` and `crates/nc-dsp/src/vvc_deblock/` follow
`LoopFilter`, all under the same license.

`crates/nc-dsp/src/vvc_intra/rectangular.rs`, `vvc_intra/cclm.rs`,
`vvc_recon.rs` and the general intra reconstruction/reference/filter control in
`crates/nc-vvc/src/decoder_blocks/` use the same pinned BSD-Clear VVdeC
`IntraPrediction`, `TrQuant` and `LoopFilter` as normative references. Small
prepared-reference fixtures and reconstruction fingerprints were captured from
that reference; no reference decoder is linked into nanocodec.


The general P-picture parser and motion derivation in
`crates/nc-vvc/src/decoder_blocks/{inter_syntax,motion,unit,tree}.rs` and
reference management in `crates/nc-vvc/src/decoder_dpb.rs` use the same pinned
BSD-Clear VVdeC `CABACReader`, `UnitTools` and `DecLib` as references. Independent
coefficient, motion and complete-picture fingerprints cover CodingToolsSets_B;
interpolation and deblocking arithmetic reuse existing shared DSP kernels.


`crates/nc-dsp/src/vvc_intra/mip_weights.rs` contains normative matrix-intra
weights from the same pinned BSD-Clear VVdeC `MipData.h`, copyright (c)
2018-2026 Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V.
and the VVdeC Authors. The license and disclaimer above apply to those tables.
Matrix prediction and multiple-reference-line equations in `vvc_intra/` and
inverse transform skip in `codec_txfm/vvc_inverse.rs` use `MatrixIntraPrediction`,
`IntraPrediction` and `Quant` as references. The small prepared-reference MIP/MRL
fixtures were captured from that decoder, as was the lossless intra picture hash.

The general B-picture contexts, both-list merge/AMVP/affine/MMVD/temporal motion,
CIIP and geometric prediction, reference/output ordering and deblocking control
in `crates/nc-vvc/src/decoder_blocks/` and `decoder_dpb.rs` use the same pinned
BSD-Clear VVdeC `Contexts`, `CABACReader`, `UnitTools`, `InterPrediction`,
`LoopFilter` and `DecLib` references. Shared high-precision interpolation in
`crates/nc-dsp/src/mc/precision.rs` and geometric weights in `vvc_geo.rs` follow
`InterpolationFilter` and `Rom`. Independent coefficient, motion, mask and
picture fingerprints are captured test evidence; no reference decoder is linked.

The general intra transform tables in `nc-dsp/src/codec_txfm/vvc_primary_tables.rs`
and `vvc_lfnst_tables.rs` reproduce normative integer matrices from the same
pinned BSD-Clear VVdeC `RomTr.cpp` and `RomLFNST.cpp`. The copyright, license and
disclaimer above apply to these tables. General ISP, transform-skip, MTS/LFNST,
LMCS and nonlinear/chroma ALF implementations use `CABACReader`, `UnitTools`,
`TrQuant`, `IntraPrediction`, `Reshape`, `Buffer` and `AdaptiveLoopFilter` as
arithmetic and syntax references. Independent transform fixtures and official
picture-region fingerprints were captured from that reference decoder.

SBT coding/zero-out constraints and decoder affine PROF refinement in
`nc-vvc/src/decoder_blocks/` and `nc-dsp/src/vvc_optical_flow.rs` use the same
BSD-Clear VVdeC `CABACReader`, `UnitTools`, `TrQuant` and `InterPrediction`
references. `nc-vvc/src/affine.rs` shares the model derivatives and normative
bidirectional reference-area limit. Independent first-B-picture fingerprints
are captured from VVdeC, with decoded output streamed into hashes.

Decoder BDOF and DMVR in `nc-dsp/src/vvc_bdof.rs`, `vvc_dmvr.rs` and
`nc-vvc/src/decoder_blocks/refinement.rs` use the same BSD-Clear VVdeC
`InterPrediction`, `InterpolationFilter`, `RdCost` and `UnitTools` arithmetic
references. The accompanying prepared prediction/search fixtures were captured
from that pinned reference decoder; picture outputs are streamed into hashes.

Shared AVX2/NEON/SIMD128 lane implementations in
`nc-dsp/src/vvc_refinement/` independently implement the same BDOF, DMVR and
PROF equations, with the existing scalar references and captured VVdeC
fixtures. They use the project's existing SIMD lane framework. The same
BSD-Clear VVdeC attribution above applies.

`nc-common/src/cabac.rs` (the CABAC decoding engine shared by the H.264 and HEVC
decoders) follows the design of the pinned BSD-Clear VVdeC `BinDecoder`: the
branch-free decision selecting range and offset by mask, and several bypass bins
read together (`decodeBinsEP`, here as one division). `nc-hevc/src/dec/residual.rs`
tabulates the significance contexts by scan index as VVdeC's `CABACReader` does.
Independent Rust implementations; the BSD-Clear VVdeC attribution above applies.

## VMAFx — EUPL-1.2 (external tool only, not vendored)

Source: https://github.com/VMAFx/vmafx (a libvmaf fork with CUDA feature
extractors). The eval harness runs its `vmaf` command-line tool as a separate
program (`eval/nceval/metrics.py`, `NC_VMAFX`), built outside this
repository (our internal notes (not published) "GPU VMAF"). No VMAFx code is
copied, linked or reimplemented here, and it is not part of any shipped
artifact.

## libvmaf — BSD+Patent

Copyright 2016-2020 Netflix, Inc.
Source: https://github.com/Netflix/vmaf (pinned v3.2.1, see
`tools/anchors/versions.env`).
Full notice: [LICENSE](docs/licenses/libvmaf/LICENSE).

Used in `eval/nceval`:
- `msssim.py`: the float MS-SSIM of `libvmaf/src/feature/ms_ssim.c` and its
  iqa helpers (`iqa/ssim_tools.c`, `convolve.c`, `decimate.c`): the 11-tap
  Gaussian window, the 9x9 9/7 low-pass decimation table and the per-scale
  exponents, reimplemented on OpenCV filters.

## Standard reference software — JM, HM, VTM (ideas and non-normative constants)

The ITU-T / ISO/IEC reference encoders are the source of several standard
encoder practices used here, reimplemented in Rust, not copied: the
λ = 0.85 · 2^((QP − 12) / 3) mode-decision multiplier (JM, HM), HM's
dead-zone quantiser rounding offsets (171/512 intra, 85/512 inter, in
`crates/nc-dsp/src/codec_txfm/mod.rs`), HM's sign-data-hiding decision
(`crates/nc-vvc/src/recon.rs`), and VTM's default chroma QP mapping table.
HM and VTM are distributed under the BSD-3-Clause licence (copyright the
ITU-T/ISO/IEC and their contributors); JM under the copyright notice in its own
distribution. Sources: https://vcgit.hhi.fraunhofer.de/jvet/HM,
https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM,
https://vcgit.hhi.fraunhofer.de/jvet/JM.

## AVX-512 motion-cost layouts — SVT-AV1 and libaom

`crates/nc-dsp/src/pixel/avx512.rs` reimplements 64-byte SAD accumulation
and source-load reuse across candidates, studied in SVT-AV1 v4.3.0
`Source/Lib/ASM_AVX512/compute_sad_intrin_avx512.c` (Copyright Intel 2019,
BSD-2-Clause and AOM Patent License, as stated in that file) and libaom
v3.15.3 `aom_dsp/sad_hwy.h` (Copyright Alliance for Open Media 2025,
BSD-2-Clause and AOM Patent License). The hierarchy and tie semantics use
nanocodec's shared grid implementation. No assembly was copied.

`crates/nc-dsp/src/diff/avx512.rs` also uses the narrow-sum/widened-square
layout studied in SVT-AV1 v4.3.0 `variance_avx512.c` (Intel 2021,
BSD-2-Clause and AOM Patent License), with bounds derived from our 64x64
block contract.

`crates/nc-dsp/src/sample/avx512.rs` also reviews that file's full-width
load and widened sum layout. Its pair sums derive from our AVX2 code;
moments reduce four independent blocks in registers. No external code was copied.

The memory batch reviews SVT-AV1 v4.3.0
`Source/Lib/ASM_AVX512/copy_wxh_avx512.c` (Copyright Meta Platforms 2024,
BSD-2-Clause and AOM Patent License) for its narrow fixed-copy / wide-row
split. `crates/nc-dsp/src/block_copy/avx512.rs` keeps the existing narrow
copies and implements masked-tail plane copies independently in Rust.

The AV1 packed filter's four-line direct memory paths in
`crates/nc-dsp/src/deblock/av1_packed.rs` and `lpf_bytes/` review dav1d
1.5.4 `src/x86/loopfilter_avx2.asm` (VideoLAN/dav1d authors 2018–2021,
Two Orioles LLC 2018, BSD-2-Clause) for short edge loads and stores in
registers. The existing shared normative filter arithmetic is retained;
no assembly was copied.

The paired SIMD128 AV1 filter transpose in
`crates/nc-dsp/src/lpf_bytes/simd128_wide.rs` reviews dav1d 1.5.4
`src/x86/loopfilter_sse.asm` (`TRANSPOSE_16X16B`, VideoLAN/dav1d authors
2018–2021 and Two Orioles LLC 2018, BSD-2-Clause). Its byte/word/dword unpack stages are
adapted to paired 8x8 groups in Rust SIMD128, retaining our shared filter
arithmetic; no assembly is copied.
