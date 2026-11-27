# Third-party code and data

nanocodec reimplements algorithms from permissively licensed projects. The
modules that do so name their source in their module documentation.

## libaom — BSD-2-Clause

Copyright (c) 2016, Alliance for Open Media. All rights reserved.
Source: https://aomedia.googlesource.com/aom (pinned v3.15.3, see
`tools/anchors/versions.env`).

`crates/nc-common/src/eob.rs` shares libaom/AVM end-of-block group
boundaries between AV1 and AV2.

Used in `crates/nc-av1`:
- `src/tables/generated.rs`: default CDFs, quantiser lookup tables, scan
  orders and coefficient context offsets. Extracted mechanically by
  `tools/codegen/gen_av1_tables.py`.
- `src/range_coder.rs`: the `od_ec` range encoder.
- `src/cdf.rs`: CDF adaptation (`update_cdf`) and the symbol cost
  approximation (`av1_cost_symbol`).
- `src/mvcoding.rs`: motion-vector difference coding and default CDFs.
- `src/coeff.rs`: coefficient coding and context derivation, including dense
  level preparation and raster contexts from `av1_txb_init_levels` and
  `av1_get_nz_map_contexts`.
- `src/segment_id.rs`: segment-id spatial prediction and context
  (`av1_get_spatial_seg_pred`) and id interleaving (`av1_neg_interleave`,
  `av1_neg_deinterleave`).
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
- `src/av1_forward.rs`: the forward ADST4/8/16 butterflies and identity
  scales (`av1/encoder/av1_fwd_txfm1d.c`), the 2-D column/row driver with its
  flips, rectangular scale and per-size shifts and cosine precisions
  (`av1/encoder/av1_fwd_txfm2d.c`), and the 11-bit cosine table
  (`av1/common/av1_txfm.c`).
- `src/mc/mod.rs`: the regular `EIGHTTAP` interpolation filters
  (`av1/common/filter.h`); `src/diff/mod.rs`: the bilinear sub-pixel
  variance definition (`aom_dsp/variance.c`).
- `src/diff/neon_sse.rs`: unsigned byte dot-product squared differences,
  following `aom_dsp/arm/sse_neon_dotprod.c` (libaom v3.15.3); independent
  cell reductions and exact-width paired-row tails.
- `src/diff/neon_dotprod.rs`: exact two-row packing for eight-wide ordinary
  variance and vector combination before horizontal reduction, following
  `aom_dsp/arm/variance_neon_dotprod.c` (libaom v3.15.3).
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
- `src/spatial.rs`: the spatial SVC reference structures (which slot each
  layer predicts from and refreshes, per temporal unit, for L2T*/L3T* and the
  K-SVC `_KEY` modes) follow `examples/svc_encoder_rtc.cc`
  `set_layer_pattern` (modes 5-10); no code is copied.
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

Used in `crates/nc-common`: `src/cabac_encoder.rs`, the CABAC arithmetic
encoding engine shared by the HEVC and H.266 encoders.

Used in `crates/nc-hevc`:
- `src/cabac.rs`: the state transition and `rangeTabLps` tables.
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

In `crates/nc-dsp/src/codec_txfm/neon.rs`, the H.264 chroma forward path
uses the fused pixel subtraction and four 4x4 transforms of
`WelsDctFourT4_AArch64_neon`, implemented with the existing Rust butterfly.

The same module's NEON inverse-add uses the fused predictor addition and
clipping of `WelsIDctT4Rec_AArch64_neon` with wide intermediate arithmetic.

The four-block x86 chroma forward kernel in
`crates/nc-dsp/src/codec_txfm/avx2_h264.rs` uses the packed AVX2 schedule of
`WelsDctFourT4_avx2` from `codec/common/x86/dct.asm`.
The same module uses the sixteen-word coefficient layout of
`WelsQuantFour4x4_avx2` in `codec/encoder/core/x86/quant.asm` for 8×8
macroblock quantisation, with nanocodec's post-product rounding retained.

The x86 inverse-add in `crates/nc-dsp/src/codec_txfm/avx2_h264_inverse.rs`
fuses the transform and predictor addition after `WelsIDctFourT4Rec_avx2`.
It keeps independent 32-bit butterflies to handle the complete scaled
coefficient range.

The licence text is the BSD-2-Clause text reproduced under libaom above,
with "Cisco Systems" as the copyright holder.

## libavc — Apache-2.0

Copyright (C) 2015 The Android Open Source Project.
Originally developed and contributed by Ittiam Systems Pvt. Ltd, Bangalore.
Source: https://github.com/ittiam-systems/libavc (revision
`4af87ce6aaad5dd222a044758a93c0bdaa1ba107`).

The H.264 NEON plane predictor in `crates/nc-dsp/src/h264_intra/plane.rs`
adapts the weighted edges and incremental row ramps from
`common/armv8/ih264_intra_pred_luma_16x16_av8.s`. The Apache License,
Version 2.0 is at https://www.apache.org/licenses/LICENSE-2.0.

The H.264 CABAC writer in `crates/nc-h264/src/cabac/mod.rs` batches
bypass bins using the constant-range update described in
`encoder/ih264e_cabac.c` (`ih264e_cabac_encode_bypass_bins`).

The H.264 8×8 inverse output in
`crates/nc-dsp/src/codec_txfm/h264_inverse8.rs` applies a signed rounding
shift before widening the destination bytes, as in
`common/armv8/ih264_iquant_itrans_recon_av8.s`.

The vertical H.264 half- and quarter-pel interpolation in
`crates/nc-dsp/src/mc/avx2_h264_vertical.rs` follows the two-row
byte tap schedule in `common/x86/ih264_inter_pred_filters_ssse3.c`.
H.264 horizontal and vertical half- and quarter-pel interpolation and the
diagonal quarter-pel combination in `crates/nc-dsp/src/mc/avx2_h264_vertical.rs`
follow the two-row byte taps in `common/x86/ih264_inter_pred_filters_ssse3.c`.

The two-axis half-pel and adjacent quarter-pel schedule in
`crates/nc-dsp/src/mc/avx2_h264_center.rs` follows the horizontal
byte products and vertical word products in the same libavc file.
The 4×4 H.264 intra predictor in `crates/nc-dsp/src/h264_intra/four.rs`
and its target modules use the paired and truncated outer averages
from `common/x86/ih264_luma_intra_pred_filters_ssse3.c` and
`common/armv8/ih264_intra_pred_luma_4x4_av8.s`.

## libhevc — Apache-2.0

Copyright (C) 2012 Ittiam Systems Pvt Ltd, Bangalore.
Source: https://github.com/ittiam-systems/libhevc (revision
`05da776e6e565546072c39192a7ed190e7f15275`).

The shared HEVC NEON 8×8 inverse and reconstruction in
`crates/nc-dsp/src/codec_txfm/neon_inverse8.rs` uses the lane-MAC schedule of
`common/arm64/ihevc_itrans_recon_8x8.s`. The corresponding 4×4 DCT and DST
in `crates/nc-dsp/src/codec_txfm/neon_inverse4.rs` follow
`ihevc_itrans_recon_4x4.s` and `ihevc_itrans_recon_4x4_ttype1.s`. The 16×16
reconstruction in `crates/nc-dsp/src/codec_txfm/neon_inverse16.rs` uses
the lane MACs and transposed scratch of `ihevc_itrans_recon_16x16.s`. The
32×32 variant in `crates/nc-dsp/src/codec_txfm/neon_inverse32.rs` follows
`ihevc_itrans_recon_32x32.s` in the same directory. The shared 10-bit HEVC and VVC luma
interpolation in `crates/nc-dsp/src/high_depth/luma_mc_neon.rs` follows the
NEON row schedules in `common/arm/ihevc_hbd_inter_pred_filters_neon_intr.c`.
The Apache License, Version 2.0 is at
https://www.apache.org/licenses/LICENSE-2.0.
The shared HEVC/VVC SAO edge application in `crates/nc-dsp/src/sao`
adapts the signed byte table, packed word offsets, and vertical
sign reuse from `common/arm/ihevc_hbd_sao_neon_intr.c` and
`common/arm64/ihevc_sao_edge_offset_class1.s`.
Band offset application in the same module uses the band index table
from `common/arm64/ihevc_sao_band_offset_luma.s` and the high-depth
variant in `common/arm/ihevc_hbd_sao_neon_intr.c`.

The Linux Arm SVE2 SAO kernel in `crates/nc-dsp/src/sao/sve2_high.S`
uses libhevc's edge offset table and vertical sign reuse from
`common/arm/ihevc_hbd_sao_neon_intr.c`.


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
- `crates/nc-av1/src/range_coder.rs` and `crates/nc-av2/src/entropy.rs`:
  the 64-bit-window range-coder flush with backward carry propagation and
  the binary-symbol shortcut (`Source/Lib/Codec/bitstream_unit.c`,
  `aom_write_symbol`); `crates/nc-av1/src/coeff.rs`: the end-of-block-1
  shortcut of `av1_write_coeffs_txb_1d` (`entropy_coding.c`) and
  `av1_cost_coeffs_txb_loop_cost_one_eob` (`rd_cost.c`).
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

### Low-bit-depth forward AVX-512 layout — BSD-2-Clause

The encoder's `txfm/avx512_forward.rs` uses the register layout of
SVT-AV1 v4.3.0 `Source/Lib/ASM_AVX512/lowbd_fwd_txfm2d_avx512.c` as a
reference: thirty-two word lanes, packed rotations and a 32x32 unpack/lane
transpose. That source credits Copyright 2026 Meta Platforms, Inc. and
affiliates, under BSD-2-Clause and the AOM Patent License 1.0. nanocodec
retains its existing libaom butterfly graphs, range checks and rounding;
only the Rust ISA adapter/layout is added. Full notices: [BSD-2-Clause](docs/licenses/svt-av1-forward/LICENSE-BSD2.md)
and [AOM Patent License](docs/licenses/svt-av1-forward/PATENTS.md).


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
The SIMD implementation uses our shared lane framework. The template
generation in `crates/nc-dsp/src/av1_film_grain/mod.rs` follows dav1d
1.5.4 `src/x86/filmgrain_avx2.asm` by batching successive Gaussian indices
and keeping the causal autoregression samples in a rolling window.
The high-depth luma block application in
`crates/nc-dsp/src/av1_film_grain/kernels/high_x86_blocks.rs` follows the
template addressing and overlap schedule in dav1d 1.5.4
`src/x86/filmgrain16_avx2.asm` and `src/x86/filmgrain16_avx512.asm`.

Copyright (c) 2018, VideoLAN and dav1d authors. All rights reserved.
Source: https://code.videolan.org/videolan/dav1d
The wide H-only AV1-family AVX-512 ICL interpolation in
`crates/nc-dsp/src/mc/av1_avx512.rs` follows the shared byte gathers,
four-tap dot products and pack-first rounding in dav1d 1.5.4
`src/x86/mc_avx512.asm` (`PUT_8TAP_H` and `PREP_8TAP_H`,
BSD-2-Clause).

The AV1 AVX2 fused warp module (`crates/nc-dsp/src/av1_mc/warp_avx2/fused.rs`)
ports the packed-byte vertical transpose, alternating row pairs, scalar phase
addressing and register schedule of `src/x86/mc_avx2.asm`
(`warp_affine_8x8_8bpc`). Its Rust wrapper preserves full signed intermediate
outputs, validates table/window bounds and falls back to the clamped kernel.

The byte-pair AVX2 interpolation rows in
`crates/nc-dsp/src/mc/avx2_byte_dot.rs` retain the schedule of dav1d's
`put_8tap` and `prep_8tap` in `src/x86/mc_avx2.asm`, shared by AV1,
HEVC, VVC and VP9 with codec-specific rounding.
The shared wide vertical pass in `crates/nc-dsp/src/mc/` also follows
the rolling adjacent-row interleaves in dav1d 1.5.4's
`src/x86/mc_avx2.asm` and `src/x86/mc_avx512.asm` (BSD-2-Clause).

The AV1 self-guided stripe (`crates/nc-dsp/src/av1_restoration/sgr_stripe.rs`,
its AVX2 arithmetic in `sgr_avx2.rs`) ports dav1d 1.5.4
`src/x86/looprestoration_avx2.asm` (`sgr_filter_5x5_8bpc`,
`sgr_filter_3x3_8bpc`, `sgr_filter_mix_8bpc`): one pass per row of
horizontal sums, running vertical sums, `A`/`B` packed in one 32-bit word,
the 565 / 222 + 343 neighbourhood weights and the projection on the
difference, over our padded stripe window. Its NEON arithmetic
(`sgr_neon.rs`) follows `src/arm/64/looprestoration.S` and
`looprestoration_common.S` (`sgr_box35_row_h_8bpc_neon`'s `ext` taps and
widening sums, `sgr_box5_vert_neon`'s saturating `z` and table lookup of
`x` in place of a division).
The Ice Lake variant (`sgr_avx2/icl.rs`) follows
`src/x86/looprestoration_avx512.asm` for VBMI factor lookup, VNNI
sum-of-squares accumulation and sixteen-lane projection.

The AV1 AVX2 filter intra patches (`crates/nc-dsp/src/av1_intra/avx2.rs`,
`filter_patches`) follow the byte-pair multiply-adds of `ipred_filter_8bpc`
(`FILTER_XMM`) in `src/x86/ipred_avx2.asm` and its `filter_intra_taps`
layout.

The AV1 AVX2 whole-block directional predictor
(`crates/nc-dsp/src/av1_intra/zones_avx2.rs`) follows `ipred_z1_8bpc`,
`ipred_z2_8bpc` and `ipred_z3_8bpc` in the same file: per-lane edge
positions, the `pshufb` gather of sample pairs, `pmaddubsw` against
weight pairs with the `pmulhrsw` rounding, and zone 2's
per-lane blend of the above and left predictions. Its NEON version
(`zones_neon.rs`) follows `ipred_z1_fill1`, `ipred_z2_fill1` and
`ipred_z3_fill1` in `src/arm/64/ipred.S`: the same per-lane positions,
the pair gathered by `tbl` from two registers and blended by
`umull`/`umlal` with a rounding narrow.

Full notice: [COPYING](docs/licenses/dav1d/COPYING).

The AV1 decoder's coefficient level pass
(`crates/nc-av1/src/decoder/coefficients/levels.rs`) is a port, in Rust, of
the level loop of dav1d 1.5.4 `src/recon_tmpl.c` (`decode_coefs`,
`get_lo_ctx`): the level-buffer layout and fixed-offset context neighbours.

The AV1 symbol decoder (`crates/nc-av1/src/decoder_range.rs`: a 64-bit
window, renormalisation by one shift, refills of whole bytes) and its
symbol kernels (`crates/nc-dsp/src/av1_symbol`: every interval bound and
the CDF adaptation in 16-bit lanes, the decoded symbol counted from the
comparison mask, the interval read back from the bounds) follow
the design of dav1d 1.5.4 `src/msac.c` and `src/x86/msac.asm`, and are
reimplemented in Rust. The fused symbol steps (`av1_symbol/sse2_step.rs`,
`av1_symbol/simd128_step.rs`, and `av1_symbol/neon_step.rs`) use the shared
CDF load, comparison-mask signed adaptation, interval scaling and branchless
interval selection design of dav1d 1.5.4 `src/x86/msac.asm` and
`src/arm/64/msac.S` (`decode_update`), Copyright © 2019 VideoLAN and dav1d
authors and Martin Storsjo, BSD-2-Clause. Their biased signed adaptation also
preserves zero-probability entries allowed by our broader kernel contract.

CDEF (`crates/nc-dsp/src/cdef/band.rs`, `cdef/direction_avx2.rs`) is
reimplemented in Rust from dav1d 1.5.4 `src/cdef_apply_tmpl.c` (`cdef_brow`:
in-place filtering of a picture by bands with unfiltered line buffers, each
8x8 unit's direction, luma and chroma together, only coded units of
filter blocks with a strength set, `adjust_strength`) and
`src/x86/cdef_avx2.asm` (`cdef_dir_8bpc`: rows and their mirror order in
the two halves of a 256-bit register, mirrored line pairs squared by one
`pmaddwd`, the best direction by `phminposuw`). Copyright © 2018 VideoLAN
and dav1d authors; Copyright © 2018 Two Orioles, LLC.

The shared eight-tap NEON motion-compensation family
(`crates/nc-dsp/src/mc/av1_neon.rs`, `av1_neon/horizontal_dot.rs`,
`horizontal_i8mm.rs`, `av1_neon/vertical_dot.rs`, `neon_dotprod.rs`) reimplements dav1d 1.5.4
`src/arm/64/mc_dotprod.S` and `src/arm/64/mc.S`: packed signed-byte
horizontal dot products with shared shuffles and folded rounding, six-tap
I8MM horizontal rows using two USMMLA instructions, transposed
four-row vertical dot products with rolling table updates, and horizontal
i16 rows accumulated vertically. The same passes serve AV1, HEVC and VVC
with their own coefficients and rounding. VP9 shares the horizontal
signed-dot rows in `neon_dotprod.rs` and rounds them to bytes. Exact-size
tail loads replace upstream's padded-read assumptions. Copyright © 2018, 2024 VideoLAN and
dav1d authors, Janne Grunau and Martin Storsjo; Copyright © 2024 Arm Limited.
BSD-2-Clause; the retained source notices are in
[docs/licenses/dav1d/MC.txt](docs/licenses/dav1d/MC.txt), with the project notice
above and in [docs/licenses/dav1d/COPYING](docs/licenses/dav1d/COPYING).

The scalar model of dav1d's arm64 inverse transforms
(`crates/nc-dsp/src/av1_inverse/dav1d_model.rs`) reimplements in Rust the
arithmetic, coefficient tables (`idct_coeffs`, `idct64_coeffs`,
`iadst4/8/16_coeffs`), identity fusions and end-of-block tables of dav1d
1.5.4 `src/arm/64/itx.S` (8-bit `inv_txfm_add_*_neon`), Copyright © 2018
VideoLAN and dav1d authors, Copyright © 2019 Martin Storsjo, BSD-2-Clause.
That assembly itself is vendored unchanged as
`crates/nc-dsp/src/av1_inverse/dav1d_itx_neon.s` (with `src/arm/asm.S` and
`src/arm/64/util.S`, preprocessed by `tools/dev/vendor_dav1d_itx.sh`; its
licence header is kept) and linked through `global_asm!` in
`av1_inverse/dav1d_neon.rs`, the AV1 decoder's NEON inverse transforms.
nc-checkasm's conformant-block check
(`crates/nc-checkasm/src/check_av1_conformant.rs`) reimplements the
floating-point forward transforms and per-size scale factors of dav1d
1.5.4 `tests/checkasm/itx.c`, Copyright © 2018 VideoLAN and dav1d authors,
Copyright © 2018 Two Orioles, LLC, BSD-2-Clause.

The AV1 lane-layout loop filters (`crates/nc-dsp/src/deblock/lanes.rs`,
`deblock/lanes/kernel.rs`, `lpf_bytes/avx2_full.rs`) reimplement in Rust the
design of dav1d 1.5.4 `src/x86/loopfilter_avx2.asm` (`lpf_h_sb_y`, `lpf_v_sb_y`,
`lpf_h_sb_uv`, `lpf_v_sb_uv`, `FILTER`, `TRANSPOSE_16X16B`): 32 lines per
vector with each 4-line unit's thresholds and filter width unpacked from a
4-byte entry, the group dispatch on its widest unit with narrower units masked
per lane, the 4/8/16-column transposes of 32 rows, the byte-pair `pmaddubsw`
running smoothing sums and `pmulhrsw` rounding. Copyright © 2018-2021 VideoLAN
and dav1d authors; Copyright © 2018 Two Orioles, LLC. BSD-2-Clause.

The shared AV1/AV2 interior CDEF filters
(`crates/nc-dsp/src/cdef/avx2_interior.rs`, `cdef/avx512.rs`) reimplement
8bpc filtering from dav1d 1.5.4 `src/x86/cdef_avx2.asm`
(`ACCUMULATE_TAP_BYTE`, `ADJUST_PIXEL`, Copyright © 2018 VideoLAN and
dav1d authors; Copyright © 2018 Two Orioles, LLC) and
`src/x86/cdef_avx512.asm` (`cdef_filter_*_8bpc`, Copyright © 2020 VideoLAN
and dav1d authors; Copyright © 2020 Two Orioles, LLC), BSD-2-Clause.
The AVX2 schedule combines signed byte constraints with paired weighting
and rounded-high pixel adjustment. The AVX512ICL schedule selects four taps
per pixel from 8x8 register tables, shifts unsigned byte magnitudes with
GFNI, and accumulates them with signed weights through VNNI dot products.
The byte border filter uses the same reference's BITALG availability masks.
The 8×8 border loads a single twelve-column halo and shares it across quarters.
Source storage, exact-size row loads and scalar equations use nanocodec's
shared family contracts. The GFNI matrix builder is also shared with loop
filters (`simd/gfni.rs`). The full dav1d notice is above and in
[docs/licenses/dav1d/COPYING](docs/licenses/dav1d/COPYING).

The shared four-group loop-filter lanes (`crates/nc-dsp/src/lpf_bytes/avx512.rs`,
`deblock/avx512.rs`, `deblock/batch.rs` and `deblock/hevc_high.rs` (HEVC/VVC
short-luma word tiles for `u8` and `u16`),
and `vp9_loop_filter/packed/avx512.rs`) use the grouped
transpose/widen/filter layout of dav1d 1.5.4 `src/x86/loopfilter_avx512.asm`
(8-bit) and `src/x86/loopfilter16_avx512.asm` (high bit depth) as design
references, reimplemented with Rust intrinsics and our shared scalar equations.
Copyright © 2018, 2022 VideoLAN and dav1d authors; Copyright © 2018, 2022
Two Orioles, LLC. BSD-2-Clause; full notice above and in
[docs/licenses/dav1d/COPYING](docs/licenses/dav1d/COPYING).
GFNI affine byte shifts (`lpf_bytes/gfni.rs`, for short and paired groups too)
follow the same reference: direct unsigned shifts and sign-extending matrices
replace word widening. The loop-filter tier requires AVX-512 plus GFNI;
CPUs without GFNI retain AVX2. VBMI is not required. The shared byte/word
arithmetic serves AV1, VP9 and AV2's distinct ramp equations.

AV1 loop restoration (`crates/nc-dsp/src/av1_restoration/`: `mod.rs`,
`wiener.rs`, `wiener_stripe.rs`, `self_guided.rs`, `sgr_lanes.rs`, `avx2.rs`,
`neon.rs`, `simd128.rs`) ports dav1d 1.5.4 `src/lr_apply_tmpl.c`,
`src/looprestoration_tmpl.c`, `src/x86/looprestoration_avx2.asm` and
`src/arm/64/looprestoration.S`: in-place restoration by stripe, the Wiener
ring of 16-bit horizontal rows, the stripe-wide tap setup and rotating
row-pointer schedule (shared by AVX2 and NEON in `wiener_stripe.rs`), independent horizontal/vertical five-tap
specialization when the outer coefficient is zero, the 8-bit Wiener arithmetic (byte-pair
`pmaddubsw` / 16-bit `mla`, the clamp as a saturating add with dav1d's
offsets, vertical pair sums with `pmaddwd`) and the self-guided row
schedule (horizontal box sums, box statistics, filtered rows), reimplemented
in Rust over our lane types. The AVX2 self-guided factor uses dav1d's
rounded reciprocal form, with exact integer outputs over the full legal
256-value factor domain.
The AV1 decoder's motion field projection
(`crates/nc-av1/src/decoder/motion_field.rs`: the field built one
superblock row at a time into a one-row buffer, per-reference span tables,
runs of equal kept vectors projected once) follows dav1d 1.5.4
`src/refmvs.c` (`load_tmvs`, `dav1d_refmvs_init_frame`), reimplemented in
Rust. Its motion store and kept-vector store (`decoder/motion.rs`
`MotionGrid`, `motion_field.rs` `SavedMotion`) follow the same file's
`splat_mv`: rows padded past the frame edge so that each block is written
unclipped, one fixed-width run per row.

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
`pmulhrsw` choices of `src/x86/itx_avx2.asm` and `src/x86/itx16_avx2.asm`)
keeps 8-bit (`ROW_BITS = 16`) and guarded 10-bit (`ROW_BITS = 18`) AV1
inverse-transform intermediates in 16-bit lanes, as dav1d's `itx` and `itx16`
assembly does. nanocodec adds proved L1 row/column bounds (`narrow_row_limit_10`)
with exact 18-bit/32-bit fallbacks instead of relying on the conformance range.

Ideas only, no code: `crates/nc-dsp/src/av1_inverse/` uses the DCT DC-only
shortcut and EOB-bounded row/column sub-band pruning described in dav1d 1.5.4
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

High-bit-depth (`u16`, 10- and 12-bit) and shared `u8`/`u16` AV1 SIMD
(`crates/nc-dsp/`), reimplemented in Rust over our lane types from dav1d
1.5.4 (BSD-2-Clause, VideoLAN and dav1d authors, Two Orioles LLC, Martin
Storsjo, Janne Grunau):
- `src/cdef/`: shared `CdefSink` interior/edge/dual-chroma filtering and
  `u16` direction search (`src/cdef_tmpl.c`, `src/x86/cdef16_avx2.asm`,
  `src/x86/cdef16_avx512.asm`, `src/arm/64/cdef16.S`).
- `src/av1_mc/high_mc.rs` and `src/av1_mc/high_blend.rs`: `u16` unscaled and
  scaled 8-tap / bilinear interpolation (including direct memory-offset
  even/odd `pmaddwd` tap-pair horizontal convolution and 2-rows-per-vector
  narrow blocks), 8x8 affine warp, saturating `adds_i16` compound average, and
  shared `u8`/`u16` distance, difference-weighted mask, wedge/mask, inter-intra
  and OBMC blends (`src/mc_tmpl.c`, `src/x86/mc16_avx2.asm`,
  `src/x86/mc16_avx512.asm`, `src/arm/64/mc16.S`). The SVE2 horizontal eight-tap
  schedule uses halfword dot products following `src/arm/64/mc16_sve.S`.
- `src/av1_inverse/high_simd.rs` and `src/av1_inverse/generic.rs`: 10- and
  12-bit SIMD inverse transforms (`I32Lanes` and guarded 10-bit `NarrowPixel`),
  DC-only `eob == 1` fast paths and EOB-bounded row/column sub-band pruning
  (`src/itx_tmpl.c`, `src/x86/itx16_avx2.asm`, `src/x86/itx16_avx512.asm`,
  `src/arm/64/itx16.S`).
- `src/av1_restoration/high_lr.rs` and `src/av1_restoration/sgr_lanes.rs`:
  register-sliding 10- and 12-bit 5/7-tap Wiener stripe convolution and shared
  8-/16-lane self-guided restoration (`src/looprestoration_tmpl.c`,
  `src/x86/looprestoration16_avx2.asm`,
  `src/x86/looprestoration16_avx512.asm`, `src/arm/64/looprestoration16.S`).
- `src/deblock/av1_high.rs`: batched 4/6/8/14-tap `u16` loop filter with
  8x8 and wide 16-/32-row vertical-edge register transpose and modular `u16`
  14-tap smoothing (`src/lpf_tmpl.c`, `src/x86/loopfilter16_avx2.asm`,
  `src/x86/loopfilter16_avx512.asm`, `src/arm/64/loopfilter16.S`).
- `src/av1_intra/high.rs`: `u16` directional/transpose/filter-intra patches
  and shared `u8`/`u16` Paeth, 1-pass `pmaddwd` smooth and one-pass `CfL`
  (`pmulhrsw`/`sqrdmulh`) predictors (`src/ipred_tmpl.c`,
  `src/x86/ipred16_avx2.asm`, `src/x86/ipred16_avx512.asm`,
  `src/arm/64/ipred16.S`).
- `src/cdef/avx512_high4.rs`: paired near and far word taps, two-table
  permutes and signed multiply-accumulate for 4×4 edge CDEF from
  `src/x86/cdef16_avx512.asm`.
- `src/av1_film_grain/high_simd.rs`: 10- and 12-bit interpolated Q15 lookup,
  luma/chroma row synthesis and overlap blending (`src/fg_apply_tmpl.c`,
  `src/x86/filmgrain16_avx2.asm`, `src/x86/filmgrain16_avx512.asm`,
  `src/arm/64/filmgrain16.S`).
- `src/av1_film_grain/kernels/high_sve2.S`: halfword gathers for the
  high-depth luma scaling lookup in dav1d's `src/arm/64/filmgrain16.S`
  application schedule.
- `src/av1_film_grain/kernels/avx512.rs`: 8-bit two-table byte lookup,
  paired luma block rows, grain overlap and chroma scaling indices from
  `src/x86/filmgrain_avx512.asm`. The shared block schedule and NEON
  application use `src/arm/64/filmgrain.S` and `src/arm/filmgrain.h`.

The browser SIMD128 extensions to `av1_mc/high_mc.rs`, `av1_mc/high_blend.rs`,
`av1_inverse/high_simd.rs`, `av1_restoration/high_lr.rs`, `deblock/av1_high.rs`
and `cdef/simd128.rs` instantiate these same shared Rust algorithms with the
existing WASM lane types; they introduce no further external implementation.

NEON CDEF direction costs (`crates/nc-dsp/src/cdef/direction.rs` and
`simd/neon.rs`) use the direct mirrored-square `smull`/`smlal` reduction
from dav1d 1.5.4 `src/arm/64/cdef_tmpl.S`; the existing shared partial-sum
graph and scalar direction/tie reference are retained.

The full shared AV1/AV2 NEON CDEF implementation in
`src/cdef/neon_direction.rs`, `neon_filter.rs` and `neon_padding.rs` ports
dav1d 1.5.4 `src/arm/64/cdef.S` (Copyright 2018 VideoLAN/dav1d authors,
2019 Martin Storsjo) and `cdef_tmpl.S` (2020 Martin Storsjo), BSD-2-Clause:
compact 16/8-stride padding, the byte and unavailable-edge filters, split
byte accumulators with halving rounding, and the one-pass direction sums.
The existing scalar reference and AV1/AV2 shared dispatch are retained. The Ice Lake 8×8 interior
filter also uses dav1d’s shared halo rows for all four quarters.

The AV1 decoder's whole-sample intra block copy in
`crates/nc-av1/src/decoder/tile/intrabc.rs` follows the direct in-bounds
prediction path of dav1d 1.5.4 `src/recon_tmpl.c` (`mc`, BSD-2-Clause).

The high-depth AV1 Ice Lake filters in
`crates/nc-dsp/src/av1_mc/high_mc/{horizontal_icl,vertical_icl,separable_icl}.rs` use word
dots from dav1d 1.5.4 `src/x86/mc16_avx512.asm` for packed sixteen-wide
horizontal and vertical pairs, thirty-two-wide rolling vertical windows,
wide horizontal stripes, and the sixteen- and thirty-two-wide separable
filters.

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
`nc-dsp/src/vp9_loop_filter/packed/neon.rs` follows the sixteen-line dual
layout of `vpx_dsp/arm/loopfilter_neon.c` (`vpx_lpf_*_dual_neon`), using
shared byte-mask arithmetic and paired eight-word smoothing.
`crates/nc-dsp/src/vp9_mc_simd/vertical_i8mm.rs` uses the sliding
four-row transpose and merge schedule from
`vpx_dsp/arm/vpx_convolve8_neon_i8mm.c` at commit
`0ab8891af8fb3c68f1135f320d551ccc78b8a253`. It retains full-width
accumulators for the unscaled coefficients and exact-size row loads.
`crates/nc-dsp/src/vp9_mc_high/sve2.rs` follows the four-row SVE2
vertical schedule in `vpx_dsp/arm/highbd_vpx_convolve8_sve2.c` at commit
`98a119c2c67d41f6354281f2cb719c8229c30601`. It uses fixed-size transposes
and stores within the existing VP9 clipping and averaging contract.
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
`vpx_dsp/x86/vpx_subpixel_8t_intrin_avx2.c` (the rolling vertical filters
and the two-dot `vpx_filter_block1d4_h8_avx2` horizontal filter), and
the 4-tier high-bit-depth (`u16`, 10- and 12-bit) VP9 kernels in
`nc-dsp/src/deblock/vp9_high.rs`, `nc-dsp/src/vp9_mc_high.rs` and
`nc-dsp/src/vp9_inverse/high_simd.rs` follow the transposed vertical-edge
loop filter, zero-phase/1-D/paired-tap `madd` convolution and EOB-corner
inverse-transform structure of `vpx_dsp/x86/highbd_loopfilter_sse2.c`,
`vpx_dsp/x86/highbd_convolve_avx2.c` and `vpx_dsp/x86/highbd_idct*_avx2.c`.
The encoder's real-time shortcuts in `crates/nc-vp9/src/backend.rs`
(`vp9.nonrd_skip_scale`, `vp9.residual_bits_per_level`) reimplement the
transform-block skip thresholds of `model_rd_for_sb_y`, the DC-only path of
`vp9_xform_quant_dc` / `vpx_fdctNxN_1`, and the level-magnitude rate of
`block_yrd` from `vp9/encoder/vp9_pickmode.c` and `vp9_encodemb.c`.
The spatial SVC layer structure in `crates/nc-vp9/src/backend.rs` and
`crates/nc-encoder/src/spatial.rs` (full SVC and K-SVC: inter-layer
prediction on every unit or on key units only, hidden lower layers shown by
`show_existing_frame`) follows `vp9/encoder/vp9_svc_layercontext.c`
(`INTER_LAYER_PRED_ON`, `INTER_LAYER_PRED_OFF_NONKEY`) and
`examples/vp9_spatial_svc_encoder.c`; no code is copied.
`tools/codegen/check_vp9_kernels.py` verifies these modules' compact golden
vectors/hashes against original C functions extracted from the pinned commit.
It compiles temporary external oracles, reads expected values directly from
the Rust tests and retains neither reference binaries nor raw video. Reference
implementations are not linked into nanocodec.
The Rust coefficient reader uses the direct Pareto tail branches and
associated energy classes of libvpx v1.16.0's
`vp9/decoder/vp9_detokenize.c` (`decode_coefs`, WebM Project, BSD-3-Clause).

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

`crates/nc-common/src/eob.rs` shares AVM/libaom end-of-block group
boundaries between AV2 and AV1.

Additional native AV2 modules reference the same pinned release:
- `src/sequence.rs` and `src/frame_header.rs`: final sequence tool groups and
  closed-loop keyframe prefixes, audited against `av2/encoder/bitstream.c`
  and the decoder's `obu.c` / `decodeframe.c`.
- `src/entropy.rs`: range normalization/termination, AV2 probability scaling,
  true bypass coding and trained-rate CDF updates from
  `avm_dsp/{entenc.c,entdec.c,entcode.c,prob.h}`, independently structured in Rust.
- `src/coeff.rs` (with `src/coeff/`), `src/tx_size.rs` and `src/coeff_tables.rs`: TX4..TX64 coefficient
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

## dav2d — BSD-2-Clause

Copyright © 2018–2026, VideoLAN and dav2d authors.
Source: https://code.videolan.org/videolan/dav2d,
commit `237229ebfefd4eb8e0e25ad9636803bb93bae0c6`.

The independent AV2 conformance executable is built with scalar 8-bit support.
Its copyright and separate patent documents were inspected at this exact pin:
[COPYING](docs/licenses/dav2d/COPYING), [PATENTS](docs/licenses/dav2d/PATENTS).
No dav2d code is linked into nanocodec. Actual emitted-subset picture
checks, rather than draft-era README wording, determine compatibility with AVM.

Reimplemented in Rust from dav2d (licence text checked at the pin above,
2026-11-15), each module naming its source in its doc comment:

- `crates/nc-av2/src/general/` (the general AV2 decoder for other encoders'
  8-bit 4:2:0 streams): `sequence.rs` and `tiles.rs` from `src/obu.c`
  (`parse_seq_hdr`, `parse_tile_info`, `calc_uniform_tiles`), `bits.rs` from
  `src/getbits.c`, `references.rs` from `src/obu.c` (`get_ref_frames`,
  `derive_pri_sec_ref`), `frame_header.rs` and `frame_header/` from
  `src/obu.c` (`parse_frame_hdr` with `read_frame_size`,
  `find_tip_ref_frames`, `parse_tile_info_frmhdr` and `rescale_matrix`;
  the non-separable Wiener dictionary and tap ranges, `dav2d_div_recip`
  and `dav2d_ccso_quant_sz` from `src/tables.c`), `output.rs` from
  `src/lib.c` (`dav2d_queue_output`, `queue_flush`), `bits.rs`'s
  sub-exponential codes from `src/getbits.c`, `block.rs` and `block/`
  from `src/decode.c` (`dav2d_decode_tile_sbrow`, `decode_sb`,
  `decode_b`, `reset_context`), `src/env.h` (partition and intra
  contexts) and `src/tables.c` (`dav2d_block_dimensions`); `block/palette.rs`
  from `src/recon_tmpl.c` (`read_pal_plane`, `copy_pal_block_y`),
  `src/decode.c` (`read_pal_indices`) and `src/pal.c` (`pal_idx_finish`);
  `block/quantizer_matrices.rs` is generated from `src/quantizer.c`
  (`qm_tbl_32x16`, `qm_tbl_32x32_t`, `dav2d_init_predefined_qm_tables`) by
  `tools/codegen/gen_av2_qm_tables.py`. The general
  `decode_b`, `reset_context`, `read_mv_residual`, `recon_b`'s intra block
  copy vector), `src/recon_tmpl.c` (`recon_b`, `decode_coefs`, `mc`,
  `bawp`), `src/refmvs.c` (`block/motion.rs` and `block/motion/`: the
  spatial motion store, the MV and warp banks, `dav2d_refmvs_find`, the
  frame's projection set-up `dav2d_refmvs_init_frame`, `load_tmvs` with the
  trajectories, the TIP field filters, the splats and the vector
  quantizing), `src/decode.c` (`block/motion_resolution.rs`,
  `block/warp_motion.rs`: `recon_b`'s motion resolution, `splat_*_mv`,
  `derive_warpmv`, `extend_warpmv`), `src/warpmv.c` (`general/warp.rs`:
  `dav2d_get_shear_params`, `dav2d_set_affine_mv2d`,
  `dav2d_find_affine_int`, the divisor reciprocals), `src/env.h`
  (partition and intra contexts, `fix_int_mv_precision`, `mv_reduce_prec`,
  `get_warpmv_2d`, `get_gmv_2d`, `warp_type`) and `src/tables.c`
  (`dav2d_block_dimensions`). The general
  decoder's CDF defaults (`general/cdfs/defaults.rs`) are AVM v1.0.0's
  (`av2/common/entropy_inits_modes.h`), extracted by
  `tools/codegen/gen_av2_general_cdfs.py`. Where dav2d and AVM v1.0.0
  differ (the layer dependency maps of the sequence header; a repeated
  output order, which AVM rejects and dav2d drops), AVM is followed.
- `crates/nc-dsp/src/av2_dec/itx.rs`, `itx_1d.rs` and `stx.rs` (the general
  decoder's inverse, cross-chroma and secondary transforms) from
  `src/itx_tmpl.c`, `src/itx_1d.c`, `src/stx_tmpl.c` and the size, shift,
  DDT-mask, CCTX-angle and end-of-block tables of `src/tables.c` and
  `src/scan.c`; `itx_kernels.rs` and `stx_tables.rs` are generated from
  `src/itx_1d.c` and `src/stx_tables.c` by
  `tools/codegen/gen_av2_dec_tables.py`.
- `crates/nc-dsp/src/av2_dec/ipred.rs` and `ipred/` (every intra predictor
  of the general decoder, 8-bit 4:2:0): from `src/ipred_tmpl.c`,
  `src/ipred_prepare_tmpl.c`, `src/ipred.h`, `src/ipred_prepare.h`,
  `src/intra_edge.h`, `src/ibp.c` and `src/derivation.h` (`derive_alpha`);
  the smooth, DC bi-prediction, reciprocal, MHCCP divisor and DIP weight
  tables (`ipred/tables.rs`) are extracted from `src/tables.c` by
  `tools/codegen/gen_av2_dec_ipred_tables.py`.
- `crates/nc-dsp/src/av2_dec/mc.rs` (intra block copy's bilinear
  prediction and block-adaptive scaling): from `src/mc_tmpl.c`
  (`put_bilin_c`, `morph_c`).
- The general decoder's in-loop filters: `crates/nc-av2/src/general/filters/`
  (`edges.rs` from `src/lf_mask.c`, `deblock.rs` from
  `src/db_apply_tmpl.c` and `src/deblock_tmpl.c`'s threshold set-up,
  `stripes.rs` and `gdf.rs` from `src/lr_apply_tmpl.c`),
  `crates/nc-av2/src/general/block/loop_filters.rs` and `restoration.rs`
  from `src/decode.c` (`decode_b`'s mask updates, `read_sb_restoration`,
  `read_restoration_info`, `read_wiener_filter`), and
  `crates/nc-dsp/src/av2_dec/gdf.rs` from `src/looprestoration_tmpl.c`
  (`gdf_prep_c`, `gdf_add_c`), and its AVX2 preparation kernel in
  `av2_dec/gdf/avx2.rs` from `src/x86/looprestoration_avx2.asm`
  (`gdf_prep_8bpc_avx2`, Copyright © 2026, VideoLAN and dav2d authors
  and Two Orioles, LLC). `av2_dec/gdf/vector8.rs` adapts that schedule to
  the shared eight-lane kernels. `gdf_tables.rs` is generated from
  `src/gdf_tables.c` by `tools/codegen/gen_av2_gdf_tables.py`;
  `filters/restoration.rs` from `src/lr_apply_tmpl.c`, and
  `crates/nc-dsp/src/av2_dec/wiener.rs` from `src/looprestoration_tmpl.c`
  (`ns_wiener_single_y_c`, `wiener_multi`, `wiener_uv_luma_c`,
  `wiener_uv_chroma_c`) and `src/looprestoration.h`
  (`setup_wiener_qval_approx_data`); `wiener_tables.rs` is generated from
  `src/wiener_tables.c` and `src/tables.c` by
  `tools/codegen/gen_av2_wiener_tables.py`.
- Film grain: `crates/nc-av2/src/general/film_grain.rs` from `src/obu.c`
  (`parse_fgm_hdr`), `src/lib.c` (`has_grain`) and `src/fg_apply_tmpl.c`
  (`dav2d_apply_grain`); AV2's grain block layout in
  `crates/nc-dsp/src/av1_film_grain/` (`GrainBlocks`) from
  `src/filmgrain_tmpl.c` (`fgy_32x32xn_c`, `fguv_32x32xn_c`).


The AV2 secondary transform in `crates/nc-dsp/src/av2_dec/stx/` adapts
the paired input rows from `src/x86/stx_avx2.asm` and the Ice Lake
word dot products from `src/x86/stx_avx512.asm`.

## VVenC — BSD-3-Clause-Clear

Source: https://github.com/fraunhoferhhi/vvenc, commit `a455d51e335c97b40eeea7bb90e3ad52b84c6e83`.

Used as an algorithm/syntax reference in `crates/nc-vvc`: `headers.rs`
(VLCWriter), `contexts.rs` (I/P-slice initialization tables and the dual-window
probability estimator), `syntax.rs`, `coeff.rs` and `coeff_contexts.rs` (CU and
full residual binarization/context tables), `motion.rs` (UnitTools spatial/history
merge and AMVP derivation), `intra.rs` (reference substitution, planar/DC/axis
prediction and PDPC), `recon.rs` (quantization), `filter.rs` and
`nc-dsp/src/vvc_deblock/` (LoopFilter thresholds and short/long deblocking,
including the compact p3..q3 short-edge layout from `LoopFilterX86.h`),
`nc-dsp/src/vvc_intra/` (planar/DC/all square angular prediction, PDPC,
angle/inverse-angle tables and InterpolationFilter cubic coefficients, scalar/SIMD;
`angular_generic.rs` uses the Q15 PDPC blend from
`source/Lib/CommonLib/x86/IntraPredX86.h::pdpcBlend`),
and `sao.rs` (edge-offset signalling).
`nc-dsp/src/codec_txfm/vvc_forward_narrow.rs` follows its rectangular pass
scaling while reusing our exact HEVC line graphs for narrow strips. Reimplemented as a restricted
native Rust codec; VVenC is not linked or called by the encoder. The arithmetic
coder is the kvazaar-derived engine shared with HEVC (`nc-common`
`cabac_encoder.rs`, under the BSD-3-Clause notice above). VVdeC is also a syntax reference for decoder
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

## VTM (H.266 reference software) — BSD-3-Clause

Source: https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM, version 24.0,
commit `ca331de230f27e0489df9d2fa6c006e2dcd2ed89`. Copyright (c) 2010-2026, ITU/ISO/IEC.

`crates/nc-vvc/src/decoder_headers/slices.rs` and `subpictures.rs` derive the slice
and subpicture layout of H.266 §6.5.1 and §7.3.2.4/7.3.2.5 in our own Rust, checked
against VTM's `PPS::initRectSliceMap`, `PPS::initSubPic` and `HLSyntaxReader` (the same
inferences and slice ordering).
`crates/nc-dsp/src/mc/rpr.rs` takes the reference-picture-resampling filter coefficients
(RPR luma, affine luma and chroma sets) from VTM's `InterpolationFilter`. No other VTM code is linked or copied; its decoder is the
reference our foreign-corpus streams are compared with.

## VVdeC — BSD-3-Clause-Clear

Source: https://github.com/fraunhoferhhi/vvdec, commit `d2614690cf13ceee1631dd6d30141755b8a689de`.

`crates/nc-vvc/src/decoder_headers/` reimplements H.266 parameter-set, picture and slice syntax
with bounded Rust readers, using `HLSyntaxReader` as a syntax reference.
`crates/nc-dsp/src/mc/generic.rs` (`hevc_bi`, HEVC default bi-prediction) averages the two
predictions as a paired 32-bit sum, the layout of VVdeC's `addAvg_SSE` (`BufferX86.h`).
`crates/nc-dsp/src/vvc_deblock/inplace.rs` also follows the separate short/long
filter layout in `LoopFilterX86.h`, while sharing our scalar decision and filter arithmetic.
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

Shared AVX2/AVX-512/NEON/SIMD128 lane implementations in
`nc-dsp/src/vvc_refinement/`, `mc/precision_simd.rs`, `mc/separable_u16_simd.rs`,
`hevc_weight/lanes.rs`, `deblock/hevc_high.rs`, `codec_txfm/inverse_add.rs`,
`mc/rpr.rs`, `vvc_geo.rs`,
`vvc_alf/`, `vvc_deblock/`, `vvc_intra/`, `vvc_lmcs.rs`, `sao/high_simd.rs` and
`high_depth/vvc_recon.rs` independently implement the same H.265/H.266 and Main 10
(`u16`) equations against the existing scalar references and captured VVdeC
fixtures, over the project's shared SIMD lane framework (no VVdeC assembly or C++
SIMD copied). The same BSD-Clear VVdeC attribution above applies.

`nc-common/src/cabac.rs` (the CABAC decoding engine shared by the H.264, HEVC
and H.266 decoders) and `nc-vvc/src/decoder_cabac.rs` follow the design of the
pinned BSD-Clear VVdeC `BinDecoder`: the branch-free decision selecting range
and offset by mask, bulk byte-buffer refill, and several bypass bins read
together (`decodeBinsEP`, here as exact power-of-two shifts when `range == 256`
else one division). `nc-vvc/src/contexts.rs` keeps the dual-window probability
estimator in shifted `u16` state (`p0 << 5`, `p1 << 1`) and
`nc-vvc/src/decoder_blocks/coeff.rs` structures subblock residual decoding as a
three-pass padded-grid walk (`CABACReader::residual_coding_subblock`), while
`nc-hevc/src/dec/residual.rs` tabulates the significance contexts by scan index
as VVdeC's `CABACReader` does. Independent Rust implementations; the BSD-Clear
VVdeC attribution above applies.

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

Used in `crates/nc-dsp/src/ms_ssim.rs`: the same MS-SSIM (window taps, 9x9
low-pass table, exponents and steps) reimplemented in Rust, for the browser
demo's analysis mode.

## Standard reference software — JM, HM, VTM (ideas and non-normative constants)

The ITU-T / ISO/IEC reference encoders are the source of several standard
encoder practices used here, reimplemented in Rust, not copied: the
λ = 0.85 · 2^((QP − 12) / 3) mode-decision multiplier (JM, HM), HM's
dead-zone quantiser rounding offsets (171/512 intra, 85/512 inter, in
`crates/nc-dsp/src/codec_txfm/mod.rs`), HM's sign-data-hiding decision
(`crates/nc-vvc/src/recon.rs`), VTM's default chroma QP mapping table, and
the structure of HM's SAO encoder (`TEncSampleAdaptiveOffset`: per-CTB
statistics, `deriveOffsets` / `estIterOffset` offset estimation, band
position search, merge and per-component RD choice) in `crates/nc-encoder/src/sao/`.
HM and VTM are distributed under the BSD-3-Clause licence (copyright the
ITU-T/ISO/IEC and their contributors); JM under the copyright notice in its own
distribution. Sources: https://vcgit.hhi.fraunhofer.de/jvet/HM,
https://vcgit.hhi.fraunhofer.de/jvet/VVCSoftware_VTM,
https://vcgit.hhi.fraunhofer.de/jvet/JM.

The VVC decoder also follows VTM 24.0 (commit
`ca331de230f27e0489df9d2fa6c006e2dcd2ed89`, BSD-3-Clause, copyright
2010-2026 ITU/ISO/IEC) as the normative reference where VVdeC does not cover a
tool or the two differ, reimplemented in Rust:
`crates/nc-vvc/src/decoder_blocks/quant_groups.rs` (quantisation-group starts,
`CU::predictQP`, `cu_qp_delta` and `cu_chroma_qp_offset` binarisation), the
weighted-prediction selection in `decoder_blocks/inter_prediction.rs`
(`InterPrediction::xPredInterBi`) and intra block copy in
`decoder_blocks/ibc.rs` (`CABACReader::cu_skip_flag` and `pred_mode`,
`PU::getIBCMergeCandidates`, `PU::fillIBCMvpCand`, the IBC branch of
`DeblockingFilter::xGetBoundaryStrengthSingle`), and picture-timing SEI
`pt_display_elemental_periods_minus1` / `pt_pic_struct` parsing plus
complementary top/bottom field-pair output interleaving (`sps_field_seq_flag`) in
`decoder_metadata.rs` and `decoder_dpb.rs` (`SEIPictureTiming` and VTM's
field-sequence output pairing). The VTM decoder is the reference of the
foreign-stream corpus (`tools/dev/vvc_foreign_corpus.py`); it is run, not
linked.

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

The narrow statistics graph (`crates/nc-dsp/src/diff/x86_narrow.rs`), used
by the AVX2 and AVX-512 tiers for 4/8-wide variance, prediction energy and block
sums, reimplements the packed low-eight-byte row layout, source/reference
interleaving with `pmaddubsw` weights (1,-1), bounded narrow sums and
interleaved sum/SSE reduction from SVT-AV1 v4.3.0
`Source/Lib/ASM_AVX2/variance_avx2.c` (`variance8_kernel_avx2`,
`variance_final_from_32bit_sum_avx2`, Copyright 2016 Alliance for Open Media),
BSD-2-Clause and AOM Patent License (the libaom notices
above). Four-wide row packing, exact-size tails and optional prediction/source
statistics follow our block
contracts; the scalar reference and NEON/SIMD128 implementations remain independent.
The AVX2 sixteen-wide variance also follows the same file's
`variance16_kernel_avx2` memory-operand row inserts and low/high byte pairing.

`crates/nc-dsp/src/sample/avx512.rs` also reviews that file's full-width
load and widened sum layout. Its pair sums derive from our AVX2 code;
moments reduce four independent blocks in registers. No external code was copied.

The memory batch reviews SVT-AV1 v4.3.0
`Source/Lib/ASM_AVX512/copy_wxh_avx512.c` (Copyright Meta Platforms 2024,
BSD-2-Clause and AOM Patent License) for its narrow fixed-copy / wide-row
split. `crates/nc-dsp/src/block_copy/avx512.rs` keeps short copies narrow and
uses exact-size tails in the shared copy/average/prep row schedule, implemented
independently in Rust.

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

## SVT-AV1 NEON dot-product subpixel schedule — BSD-2-Clause

`crates/nc-dsp/src/diff/neon_dotprod.rs` reimplements the independent
column accumulators and two-row even-phase bilinear schedule in pinned
SVT-AV1 v4.3.0 `Source/Lib/ASM_NEON_DOTPROD/variance_neon_dotprod.c`
(`FUSED_SUBPEL_VAR_WXH_XY`). That file credits Copyright (c) 2023, Alliance
for Open Media. All rights reserved, and specifies BSD-2-Clause and the
Alliance for Open Media Patent License 1.0. The BSD-2-Clause text is
reproduced under libaom above. Our implementation retains the shared
signed sum result and adds a single-row tail for odd block heights.

## rav1e — BSD-2-Clause and Alliance for Open Media Patent License 1.0

Copyright (c) 2019, The rav1e contributors. All rights reserved.
Source: https://github.com/xiph/rav1e (revision
`31435de9d76fddd38f6dcc31d4014574cebb2092`).
Notices: [LICENSE](docs/licenses/rav1e/LICENSE) and
[PATENTS](docs/licenses/rav1e/PATENTS).

The shared x86 SATD in `crates/nc-dsp/src/pixel/` adapts the packed 4×4
Hadamard, the 8×8 transform transpose and final absolute-max reduction
from `src/x86/satd.asm`.
