#!/usr/bin/env python3
"""Generate the general AV2 decoder's probability models from AVM v1.0.0's
default tables (crates/nc-av2/src/general/cdfs.rs and general/cdfs/defaults.rs).

    tools/codegen/gen_av2_general_cdfs.py --source AVM_SRC --build AVM_BUILD [--write]

AVM_SRC is an AVM checkout at the pin below (e.g.
~/nanocodec-data/anchors/vp9-av2/src/avm-v1.0.0), AVM_BUILD its build
directory (for avm_config.h). Each model family in SPECS names its field, its
array axes, its symbol count and the AVM expression of one CDF (axes a, b, c;
q is the coefficient quantizer category of quantized families). A small C
program compiled against the pinned headers prints every default with its
adaptation rates, checks the axes with static assertions, and hashes 65 AVM
`update_cdf` steps per model; the generated test replays them through our
`Cdf`. Without --write the files are only checked.

Tool agents add the families their syntax reads to SPECS (docs/handoff/av2.md
"The general block layer").
"""
import argparse
import itertools
import subprocess
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import tempdirs  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PIN = '966a7d7cd6fcf60360caf5dc413b2aeeb65e144d'
OUT = ROOT / 'crates/nc-av2/src/general'

# (field, axes, symbols, AVM expression, flags, what it codes). Flags:
# 'q' the family has a table per coefficient quantizer category; 'r' its
# AVM rows are wider than its alphabet (a ragged table).
SPECS = [
    ('part_split', [2, 64], 2, 'default_do_split_cdf[a][b]', '',
     'whether a block splits (plane tree, context)'),
    ('part_square', [8], 2, 'default_do_square_split_cdf[0][a]', '',
     'whether a 128x128 or 256x256 block splits into four squares'),
    ('part_dir', [2, 64], 2, 'default_rect_type_cdf[a][b]', '',
     'the direction of a binary split'),
    ('part_ext', [2, 64], 2, 'default_do_ext_partition_cdf[a][0][b]', '',
     'whether a split is three-way'),
    ('part_4way', [2, 64], 2, 'default_do_uneven_4way_partition_cdf[a][0][b]', '',
     'whether a three-way split is the uneven four-way one instead'),
    ('region_type', [4], 2, 'default_region_type_cdf[a]', '',
     'whether an inter-frame region is intra only (separate trees)'),
    ('skip_mode', [3], 2, 'default_skip_mode_cdfs[a]', '',
     'whether an inter block is in skip mode'),
    ('is_inter', [4], 2, 'default_intra_inter_cdf[a]', '',
     'whether a block of an inter frame is inter'),
    ('intrabc', [3], 2, 'default_intrabc_cdf[a]', '',
     'whether an intra block uses intra block copy'),
    ('skip_txfm', [6], 2, 'default_skip_txfm_cdfs[a]', '',
     'whether a block has no residual (contexts 3..5 in skip mode)'),
    ('gdf', [], 2, 'default_gdf_cdf', '',
     'whether a guided-detail-filter block is filtered'),
    ('cdef_index0', [4], 2, 'default_cdef_strength_index0_cdf[a]', '',
     'whether a CDEF block uses strength set 0'),
    ('cdef_index3', [], 2, 'default_cdef_cdf[0]', 'r',
     'the CDEF strength set beyond 0 of a frame with three sets'),
    ('cdef_index4', [], 3, 'default_cdef_cdf[1]', 'r', 'the same with four sets'),
    ('cdef_index5', [], 4, 'default_cdef_cdf[2]', 'r', 'the same with five sets'),
    ('cdef_index6', [], 5, 'default_cdef_cdf[3]', 'r', 'the same with six sets'),
    ('cdef_index7', [], 6, 'default_cdef_cdf[4]', 'r', 'the same with seven sets'),
    ('cdef_index8', [], 7, 'default_cdef_cdf[5]', 'r', 'the same with eight sets'),
    ('ccso', [3, 4], 2, 'default_ccso_cdf[a][b]', '',
     'whether a CCSO block of a plane is filtered'),
    ('delta_q', [], 8, 'default_delta_q_cdf', '',
     'a superblock quantizer delta (the last symbol escapes)'),
    # Segment ids (dav2d `seg_pred`, `seg_id_ext`, `seg_id`).
    ('seg_pred', [3], 2, 'default_segment_pred_cdf[a]', '',
     'whether a block takes its segment id from the primary reference\'s map'),
    ('seg_id_ext', [3], 2, 'default_seg_id_ext_flag_cdf[a]', '',
     'whether a segment id difference is in the upper eight (sixteen segments)'),
    ('seg_id', [3], 8, 'default_spatial_pred_seg_tree_cdf[a]', '',
     'a segment id difference from the spatial prediction, lower eight'),
    ('seg_id_upper', [3], 8, 'default_spatial_pred_seg_tree_cdf1[a]', '',
     'a segment id difference from the spatial prediction, upper eight'),
    # Loop restoration units (docs/handoff/av2.md: the loop-filter group).
    ('rst_switchable', [2], 2, 'default_switchable_flex_restore_cdf[a][0]', '',
     'a switchable unit: off (step 0), then PC-Wiener rather than NS-Wiener (step 1)'),
    ('rst_ns_wiener', [], 2, 'default_wienerns_restore_cdf', '',
     'whether a non-separable Wiener unit is filtered'),
    ('rst_pc_wiener', [], 2, 'default_pc_wiener_restore_cdf', '',
     'whether a PC-Wiener unit is filtered'),
    ('wiener_ns_len', [2], 2, 'default_wienerns_length_cdf[a]', '',
     'whether a unit filter codes more taps (luma, chroma)'),
    ('wiener_ns_sym', [], 2, 'default_wienerns_uv_sym_cdf', '',
     'whether a chroma unit filter pairs its outer taps'),
    ('wiener_ns_cf', [], 4, 'default_wienerns_4part_cdf[0]', '',
     'the group of a unit filter tap (four-part code)'),
    # Intra modes (docs/handoff/av2.md "The general block layer": the intra group).
    ('dpcm_y', [], 2, 'default_dpcm_cdf', '', 'whether a lossless luma block is DPCM coded'),
    ('dpcm_dir_y', [], 2, 'default_dpcm_vert_horz_cdf', '', 'the direction of luma DPCM (horizontal)'),
    ('dpcm_uv', [], 2, 'default_dpcm_uv_cdf', '', 'whether a lossless chroma block is DPCM coded'),
    ('dpcm_dir_uv', [], 2, 'default_dpcm_uv_vert_horz_cdf', '', 'the direction of chroma DPCM (horizontal)'),
    ('y_mode_set', [], 4, 'default_y_mode_set_cdf', '', 'the luma mode set (0: the first 13 indices)'),
    ('y_mode_index', [3], 8, 'default_y_mode_idx_cdf[a]', '', 'the luma mode index in set 0 (the last symbol escapes)'),
    ('y_mode_offset', [3], 6, 'default_y_mode_idx_offset_cdf[a]', '', 'the luma mode index past the escape'),
    ('fsc', [4, 6], 2, 'default_fsc_mode_cdf[a][b]', '', 'whether a block uses forward skip coding'),
    ('mrl_index', [3], 4, 'default_mrl_index_cdf[a]', '', 'the reference line of a directional luma block'),
    ('multi_mrl', [3], 2, 'default_multi_line_mrl_cdf[a]', '', 'whether two reference lines are blended'),
    ('palette_y', [], 2, 'default_palette_y_mode_cdf', '', 'whether a luma block uses a palette'),
    ('palette_size', [], 7, 'default_palette_y_size_cdf', '', 'the luma palette size less two'),
    ('palette_identity', [4], 3, 'default_identity_row_cdf_y[a]', '',
     'a palette index row: coded, one repeated index, or a copy of the row before'),
    ('palette_index2', [5], 2, 'default_palette_y_color_index_cdf[0][a]', 'r',
     'a palette color index (palettes of 2 colors)'),
    ('palette_index3', [5], 3, 'default_palette_y_color_index_cdf[1][a]', 'r',
     'a palette color index, 3 colors'),
    ('palette_index4', [5], 4, 'default_palette_y_color_index_cdf[2][a]', 'r',
     'a palette color index, 4 colors'),
    ('palette_index5', [5], 5, 'default_palette_y_color_index_cdf[3][a]', 'r',
     'a palette color index, 5 colors'),
    ('palette_index6', [5], 6, 'default_palette_y_color_index_cdf[4][a]', 'r',
     'a palette color index, 6 colors'),
    ('palette_index7', [5], 7, 'default_palette_y_color_index_cdf[5][a]', 'r',
     'a palette color index, 7 colors'),
    ('palette_index8', [5], 8, 'default_palette_y_color_index_cdf[6][a]', 'r',
     'a palette color index, 8 colors'),
    ('dip', [3], 2, 'default_intra_dip_cdf[a]', '', 'whether a luma block uses data-driven intra prediction'),
    ('dip_mode', [], 6, 'default_intra_dip_mode_n6_cdf', '', 'the data-driven intra prediction mode'),
    ('cfl', [3], 2, 'default_cfl_cdf[a]', '', 'whether a chroma block is predicted from luma'),
    ('uv_mode', [2], 8, 'default_uv_mode_cdf[a]', '', 'the chroma mode index (the last symbol escapes)'),
    ('mhccp', [], 2, 'default_cfl_mhccp_switch_cdf', '', 'whether chroma from luma is the multi-hypothesis model'),
    ('mhccp_direction', [4], 3, 'default_filter_dir_cdf[a]', '', 'the multi-hypothesis model\'s direction'),
    ('cfl_type', [], 2, 'default_cfl_index_cdf', '', 'whether chroma from luma signals its scales'),
    ('cfl_sign', [], 8, 'default_cfl_sign_cdf', '', 'the joint signs of the chroma from luma scales'),
    ('cfl_alpha', [6], 8, 'default_cfl_alpha_cdf[a]', '', 'a chroma from luma scale magnitude'),
    # Intra block copy (the intra group).
    ('intrabc_mode', [], 2, 'default_intrabc_mode_cdf', '',
     'whether a block copy takes its predicted vector as is (no residual)'),
    ('intrabc_precision', [], 2, 'default_intrabc_bv_precision_cdf[0]', '',
     'whether a block copy vector residual is in quarter samples'),
    ('morph_pred', [3], 2, 'default_morph_pred_cdf[a]', '',
     'whether a block copy is scaled by a linear model of its edges'),
    # Block copy vector residuals: AVM's `ndvc` models, which start from
    # `default_nmv_context` (the shell classes per precision: 3 whole, 5 quarter samples).
    ('dv_shell_set', [], 2, 'default_nmv_context.joint_shell_set_cdf', '',
     'whether a vector residual\'s shell class is in the upper half'),
    ('dv_class_low_integer', [], 7, 'default_nmv_context.joint_shell_class_cdf_0[3]', 'r',
     'a whole-sample residual\'s shell class, lower half'),
    ('dv_class_low_quarter', [], 8, 'default_nmv_context.joint_shell_class_cdf_0[5]', 'r',
     'a quarter-sample residual\'s shell class, lower half'),
    ('dv_class_high_integer', [], 7, 'default_nmv_context.joint_shell_class_cdf_1[3]', 'r',
     'a whole-sample residual\'s shell class, upper half'),
    ('dv_class_high_quarter', [], 8, 'default_nmv_context.joint_shell_class_cdf_1[5]', 'r',
     'a quarter-sample residual\'s shell class, upper half'),
    ('dv_offset_low', [2], 2, 'default_nmv_context.shell_offset_low_class_cdf[a]', '',
     'the shell offset of class 0 or 1'),
    ('dv_offset_class2', [], 2, 'default_nmv_context.shell_offset_class2_cdf', '',
     'whether the shell offset of class 2 is above 0'),
    ('dv_offset_high', [16], 2, 'default_nmv_context.shell_offset_other_class_cdf[0][a]', '',
     'a bit of the shell offset of class 3 or more'),
    ('dv_column_greater', [2], 2, 'default_nmv_context.col_mv_greater_flags_cdf[a]', '',
     'whether the smaller component steps further up the shell'),
    ('dv_column_index', [4], 2, 'default_nmv_context.col_mv_index_cdf[a]', '',
     'which component is the smaller one'),
    # Inter blocks (docs/handoff/av2.md "The general block layer": the inter group).
    ('tip', [3], 2, 'default_tip_cdf[a]', '', 'whether an inter block predicts from the TIP frame'),
    ('tip_mode', [], 2, 'default_tip_pred_mode_cdf', '', 'whether a TIP block codes a vector residual'),
    ('tip_drl', [3], 2, 'default_tip_drl_cdf[a]', '', 'a TIP block\'s DRL index bit'),
    ('skip_mode_drl', [3], 2, 'default_skip_drl_cdf[a]', '', 'a skip-mode block\'s DRL index bit'),
    ('comp', [5], 2, 'default_comp_inter_cdf[a]', '', 'whether an inter block has two references'),
    ('single_ref', [3, 6], 2, 'default_single_ref_cdf[a][b]', '',
     'whether a single reference is the rank tried (context, rank)'),
    ('comp0_ref', [3, 6], 2, 'default_comp_ref0_cdf[a][b]', '',
     'whether a compound block\'s first reference is the rank tried'),
    ('comp1_ref', [3, 2, 6], 2, 'default_comp_ref1_cdf[a][b][c]', '',
     'whether its second is (context, same direction as the first, rank)'),
    ('warp', [5], 2, 'default_inter_warp_mode_cdf[a]', '', 'whether a single-reference block warps'),
    ('warp_newmv', [], 2, 'default_is_warpmv_or_warp_newmv_cdf', '', 'whether a warp block is WARPMV, not WARPNEWMV'),
    ('inter_mode', [5], 3, 'default_inter_single_mode_cdf[a]', '', 'a single-reference mode (NEAR, GLOBAL, NEW)'),
    ('amvd', [9, 3], 2, 'default_amvd_mode_cdf[a][b]', '', 'whether a residual is adaptive (mode, context)'),
    ('bawp', [2], 2, 'default_bawp_cdf[a]', '', 'block-adaptive weighted prediction (luma, chroma)'),
    ('bawp_explicit', [3], 2, 'default_explicit_bawp_cdf[a]', '', 'whether the weight is explicit'),
    ('bawp_explicit_scale', [], 2, 'default_explicit_bawp_scale_cdf', '', 'the explicit weight\'s scale'),
    ('warp_extend', [3], 2, 'default_warp_extend_cdf[a]', '', 'whether a warp extends a neighbour\'s model'),
    ('warp_causal', [4], 2, 'default_warp_causal_cdf[a]', '', 'whether a warp is fitted to the neighbours'),
    ('inter_intra', [4], 2, 'default_interintra_cdf[a]', '', 'whether a block blends an intra prediction'),
    ('inter_intra_mode', [4], 4, 'default_interintra_mode_cdf[a]', '', 'the inter-intra mode'),
    ('inter_intra_wedge', [], 2, 'default_wedge_interintra_cdf', '', 'whether inter-intra uses a wedge'),
    ('warp_inter_intra', [4], 2, 'default_warp_interintra_cdf[a]', '', 'whether a WARPMV block blends an intra prediction'),
    ('wedge_quad', [], 4, 'default_wedge_quad_cdf', '', 'a wedge angle\'s quadrant'),
    ('wedge_angle', [4], 5, 'default_wedge_angle_cdf[a]', '', 'the angle in the quadrant'),
    ('wedge_dist', [], 4, 'default_wedge_dist_cdf', '', 'a wedge\'s distance'),
    ('wedge_dist2', [], 3, 'default_wedge_dist_cdf2', '', 'a wedge\'s distance at angles without the centre'),
    ('jmvd_amvd_scale', [], 3, 'default_jmvd_amvd_scale_mode_cdf', '', 'the joint residual scale, adaptive'),
    ('jmvd_scale', [], 5, 'default_jmvd_scale_mode_cdf', '', 'the joint residual scale'),
    ('drl', [3, 5], 2, 'default_drl_cdf[a][b]', '', 'a DRL index bit (bit, mode context)'),
    ('mv_precision_default', [3], 2, 'default_pb_mv_most_probable_precision_cdf[a]', '',
     'whether a residual takes the frame\'s precision'),
    ('mv_precision', [2, 3], 3, 'default_pb_mv_precision_cdf[a][b]', '', 'the residual precision otherwise'),
    ('warp_ref0', [], 2, 'default_warp_ref_idx0_cdf[0]', '', 'whether the warp model index passes 0'),
    ('warp_ref1', [], 2, 'default_warp_ref_idx1_cdf[0]', '', 'whether it passes 1'),
    ('warp_ref2', [], 2, 'default_warp_ref_idx2_cdf[0]', '', 'whether it passes 2'),
    ('warpmv_with_mvd', [], 2, 'default_warpmv_with_mvd_flag_cdf', '', 'whether a WARPMV block codes a residual'),
    ('warp_delta_precision', [31], 2, 'default_warp_precision_idx_cdf[(const int[]){18, 17, 16, 15, 14, 13, 12, 11, 24, 28, 0, 10, 9, 8, 22, 26, 23, 7, 6, 5, 20, 27, 21, 4, 3, 2, 0, 25, 19, 1, 0}[a]]', '',
     'whether warp deltas are coarse (block size, ours to AVM\'s)'),
    ('warp_delta', [2], 8, 'default_warp_delta_param_cdf[a]', '', 'a warp delta parameter magnitude'),
    ('warp_delta_high', [2], 8, 'default_warp_delta_param_high_cdf[a]', '', 'its continuation'),
    ('warp_delta_sign', [], 2, 'default_warp_param_sign_cdf', '', 'a warp delta\'s sign'),
    ('comp_mode_same_refs', [5], 4, 'default_inter_compound_mode_same_refs_cdf[a]', '',
     'a compound mode of one reference twice'),
    ('comp_mode_joint', [2], 2, 'default_inter_compound_mode_is_joint_cdf[a]', '', 'whether a compound mode is `JOINT_NEWMV`'),
    ('comp_mode', [5], 5, 'default_inter_compound_mode_non_joint_type_cdf[a]', '', 'a compound mode otherwise'),
    ('opfl', [2], 2, 'default_use_optflow_cdf[a]', '', 'whether a compound mode is refined by optical flow'),
    ('refine_mv', [11], 2, 'default_refinemv_flag_cdf[a + 1]', '', 'whether a compound block refines its vectors (by mode)'),
    ('comp_type_masked', [12], 2, 'default_comp_group_idx_cdfs[a]', '', 'whether a compound blend is masked'),
    ('comp_type_weighted', [], 2, 'default_compound_type_cdf', '', 'whether the mask is the difference, not a wedge'),
    ('cwp', [4], 2, 'default_cwp_idx_cdf[0][a]', '', 'a compound weight index bit'),
    ('interp_filter', [8], 3, 'default_switchable_interp_cdf[a]', '', 'a block\'s interpolation filter'),
    ('amvd_joint', [], 4, 'default_nmv_context.amvd_joints_cdf', '', 'which adaptive residual components are nonzero'),
    ('amvd_index', [2], 8, 'default_nmv_context.comps[a].amvd_indices_cdf', '', 'an adaptive residual magnitude (y, x)'),
    # Inter vector residuals: AVM's `nmvc` models (from `default_nmv_context`), per precision.
    ('mv_shell_set', [], 2, 'default_nmv_context.joint_shell_set_cdf', '', 'whether the shell class is in the upper half'),
    ('mv_class_low0', [], 5, 'default_nmv_context.joint_shell_class_cdf_0[0]', 'r',
     'a vector residual\'s shell class, lower half, precision 0'),
    ('mv_class_high0', [], 6, 'default_nmv_context.joint_shell_class_cdf_1[0]', 'r',
     'the same, upper half'),
    ('mv_class_low1', [], 6, 'default_nmv_context.joint_shell_class_cdf_0[1]', 'r',
     'a vector residual\'s shell class, lower half, precision 1'),
    ('mv_class_high1', [], 6, 'default_nmv_context.joint_shell_class_cdf_1[1]', 'r',
     'the same, upper half'),
    ('mv_class_low2', [], 6, 'default_nmv_context.joint_shell_class_cdf_0[2]', 'r',
     'a vector residual\'s shell class, lower half, precision 2'),
    ('mv_class_high2', [], 7, 'default_nmv_context.joint_shell_class_cdf_1[2]', 'r',
     'the same, upper half'),
    ('mv_class_low3', [], 7, 'default_nmv_context.joint_shell_class_cdf_0[3]', 'r',
     'a vector residual\'s shell class, lower half, precision 3'),
    ('mv_class_high3', [], 7, 'default_nmv_context.joint_shell_class_cdf_1[3]', 'r',
     'the same, upper half'),
    ('mv_class_low4', [], 7, 'default_nmv_context.joint_shell_class_cdf_0[4]', 'r',
     'a vector residual\'s shell class, lower half, precision 4'),
    ('mv_class_high4', [], 8, 'default_nmv_context.joint_shell_class_cdf_1[4]', 'r',
     'the same, upper half'),
    ('mv_class_low5', [], 8, 'default_nmv_context.joint_shell_class_cdf_0[5]', 'r',
     'a vector residual\'s shell class, lower half, precision 5'),
    ('mv_class_high5', [], 8, 'default_nmv_context.joint_shell_class_cdf_1[5]', 'r',
     'the same, upper half'),
    ('mv_class_low6', [], 8, 'default_nmv_context.joint_shell_class_cdf_0[6]', 'r',
     'a vector residual\'s shell class, lower half, precision 6'),
    ('mv_class_high6', [], 8, 'default_nmv_context.joint_shell_class_cdf_1[6]', 'r',
     'the same, upper half'),
    ('mv_shell_last', [], 2, 'default_nmv_context.joint_shell_last_two_classes_cdf', '',
     'which of the last two shell classes, at eighth samples'),
    ('mv_offset_low', [2], 2, 'default_nmv_context.shell_offset_low_class_cdf[a]', '', 'the shell offset of class 0 or 1'),
    ('mv_offset_class2', [], 2, 'default_nmv_context.shell_offset_class2_cdf', '', 'whether the class 2 offset is above 0'),
    ('mv_offset_high', [16], 2, 'default_nmv_context.shell_offset_other_class_cdf[0][a]', '', 'a bit of a higher offset'),
    ('mv_column_greater', [2], 2, 'default_nmv_context.col_mv_greater_flags_cdf[a]', '', 'whether the smaller component steps up'),
    ('mv_column_index', [4], 2, 'default_nmv_context.col_mv_index_cdf[a]', '', 'which component is the smaller one'),
    # The residual (docs/handoff/av2.md "The general block layer": the residual group).
    ('lossless_tx_size', [4, 2], 2, 'default_lossless_tx_size_cdf[a][b]', '',
     'whether a lossless block uses its largest transform (size group, inter)'),
    ('tx_split', [2, 2, 9], 2, 'default_txfm_do_partition_cdf[a][b][c]', '',
     'whether a block splits its transform (forward skip coding, inter, size)'),
    ('tx_part_1d', [2, 2, 2], 2, 'default_txfm_2or3_way_partition_type_cdf[a][b][c]', '',
     'whether a one-direction transform split is four-way'),
    ('tx_part_2d', [2, 2, 14], 7, 'default_txfm_4way_partition_type_cdf[a][b][c]', '',
     'the transform split type of a block both of whose sides split'),
    ('tx_part_2d_reduced', [2, 2, 14], 7, 'default_txfm_4way_partition_type_reduced_cdf[a][b][c]', '',
     'the same in sequences with the reduced transform partition set'),
    ('tx_type_lossless', [], 2, 'default_lossless_inter_tx_type_cdf', '',
     'whether a lossless inter 4x4 transform is the identity'),
    ('tx_type_long32_dct', [2], 2, 'default_tx_ext_32_cdf[a]', '',
     'whether the long side of a 32-long transform is a DCT (inter)'),
    ('tx_type_intra_short', [4], 4, 'default_intra_ext_tx_short_side_cdf[a]', '',
     'the short-side type of a long intra transform'),
    ('tx_type_intra', [4], 7, 'default_intra_ext_tx_cdf[1][a]', 'r', 'an intra luma transform type'),
    ('tx_type_intra_reduced', [4], 2, 'default_intra_ext_tx_cdf[2][a]', 'r',
     'the same with the reduced transform set'),
    ('tx_type_inter_short', [3, 4], 4, 'default_inter_ext_tx_short_side_cdf[a][b]', '',
     'the short-side type of a long inter transform (end-of-block context, size)'),
    ('tx_type_inter_set', [2, 3, 4], 2, 'default_inter_tx_type_set_cdf[a][b][c]', '',
     'whether an inter type is in the second set (16x16, end-of-block context, size)'),
    ('tx_type_inter_index', [2, 3], 8, 'default_inter_tx_type_idx_cdf[a][b]', '',
     'an inter type of the first set'),
    ('tx_type_inter_offset1', [3], 8, 'default_inter_tx_type_offset_1_cdf[a]', '',
     'an inter type of the second set, transforms other than 16x16'),
    ('tx_type_inter_offset2', [3], 4, 'default_inter_tx_type_offset_2_cdf[a]', '',
     'an inter type of the second set, 16x16 transforms'),
    ('tx_type_inter_dct_idtx', [3, 4], 2, 'default_inter_ext_tx_cdf[3][a][b]', 'r',
     'whether an inter type is the DCT, not the identity (reduced sets 1 and 2, 32x32)'),
    ('tx_type_inter_dct_idtx_iddct', [3, 4], 4, 'default_inter_ext_tx_cdf[4][a][b]', 'r',
     'an inter type of the DCT, identity and one-direction DCT set (reduced set 3)'),
    ('stx', [2, 5], 4, 'default_stx_cdf[a][b]', '', 'the secondary transform kernel (inter, size)'),
    ('stx_set_adst', [], 4, 'default_most_probable_stx_set_cdf_ADST_ADST', '',
     'the secondary transform set of an `ADST_ADST` transform'),
    ('stx_set', [], 7, 'default_most_probable_stx_set_cdf', '', 'the secondary transform set'),
    ('cctx', [], 7, 'default_cctx_type_cdf', '', 'the cross-chroma transform'),
    # Coefficients, per quantizer category (AVM `av2_default_*` tables).
    ('txb_skip', [2, 5, 10], 2, 'av2_default_txb_skip_cdfs[q][a][b][c]', 'q',
     'whether a luma or U transform has no coefficients (inter or forward skip, size, context)'),
    ('txb_skip_v', [12], 2, 'av2_default_v_txb_skip_cdfs[q][a]', 'q',
     'whether a V transform has no coefficients'),
    ('eob16', [3], 5, 'av2_default_eob_multi16_cdfs[q][a]', 'q', 'the end-of-block class of 16 coefficients'),
    ('eob32', [3], 6, 'av2_default_eob_multi32_cdfs[q][a]', 'q', 'the same of 32'),
    ('eob64', [3], 7, 'av2_default_eob_multi64_cdfs[q][a]', 'q', 'the same of 64'),
    ('eob128', [3], 8, 'av2_default_eob_multi128_cdfs[q][a]', 'q', 'the same of 128'),
    ('eob256', [3], 8, 'av2_default_eob_multi256_cdfs[q][a]', 'q', 'the same of 256 (escapes)'),
    ('eob512', [3], 8, 'av2_default_eob_multi512_cdfs[q][a]', 'q', 'the same of 512 (escapes)'),
    ('eob1024', [3], 8, 'av2_default_eob_multi1024_cdfs[q][a]', 'q', 'the same of 1024 (escapes)'),
    ('eob_extra', [], 2, 'av2_default_eob_extra_cdfs[q]', 'q', 'the end of block\'s high offset bit'),
    ('eob_base_hf', [5, 4], 3, 'av2_default_coeff_base_eob_multi_cdfs[q][a][b]', 'q',
     'the last luma level, high frequencies'),
    ('base_hf', [5, 20, 2], 4, 'av2_default_coeff_base_multi_cdfs[q][a][b][c]', 'q',
     'a luma base level, high frequencies (size, context, TCQ)'),
    ('range_hf', [7], 4, 'av2_default_coeff_lps_multi_cdfs[q][a]', 'q', 'a luma low range, high frequencies'),
    ('eob_base_lf', [5, 4], 5, 'av2_default_coeff_base_lf_eob_multi_cdfs[q][a][b]', 'q',
     'the last luma level, low frequencies'),
    ('base_lf', [5, 33, 2], 6, 'av2_default_coeff_base_lf_multi_cdfs[q][a][b][c]', 'q',
     'a luma base level, low frequencies'),
    ('range_lf', [14], 4, 'av2_default_coeff_lps_lf_multi_cdfs[q][a]', 'q', 'a luma low range, low frequencies'),
    ('parity_dc', [5], 4, 'av2_default_coeff_base_ph_cdfs[q][a]', 'q', 'a parity-hidden DC level'),
    ('dc_sign', [2, 2, 3], 2, 'av2_default_dc_sign_cdfs[q][a][b][c]', 'q',
     'a DC sign (chroma, parity hidden, context)'),
    ('bob_base', [3, 3], 3, 'av2_default_coeff_base_bob_multi_cdfs[q][a][b]', 'q',
     'the first level of a forward-skip block'),
    ('range_idtx', [3, 7], 4, 'av2_default_coeff_lps_multi_cdfs_idtx[q][a][b]', 'q',
     'a forward-skip low range'),
    ('base_idtx', [3, 7], 4, 'av2_default_coeff_base_multi_cdfs_idtx[q][a][b]', 'q',
     'a forward-skip base level'),
    ('sign_idtx', [3, 9], 2, 'av2_default_idtx_sign_cdfs[q][a][b]', 'q', 'a forward-skip sign'),
    ('eob_base_uv_hf', [4], 3, 'av2_default_coeff_base_eob_multi_uv_cdfs[q][a]', 'q',
     'the last chroma level, high frequencies'),
    ('base_uv_hf', [12], 4, 'av2_default_coeff_base_multi_uv_cdfs[q][a]', 'q',
     'a chroma base level, high frequencies'),
    ('range_uv_hf', [4], 4, 'av2_default_coeff_lps_multi_uv_cdfs[q][a]', 'q',
     'a chroma low range, high frequencies'),
    ('eob_base_uv_lf', [4], 5, 'av2_default_coeff_base_lf_eob_multi_uv_cdfs[q][a]', 'q',
     'the last chroma level, low frequencies'),
    ('base_uv_lf', [12], 6, 'av2_default_coeff_base_lf_multi_uv_cdfs[q][a]', 'q',
     'a chroma base level, low frequencies'),
]


def rust_type(base, dims):
    for size in reversed(dims):
        base = f'[{base}; {size}]'
    return base


def probe(source, build):
    """Defaults per category and family, and the update-trace hash per category."""
    c = ['#include <stdio.h>', '#include <stdint.h>', '#include <string.h>',
         '#include "av2/common/av2_common_int.h"',
         '#include "av2/common/entropy_inits_coeffs.h"',
         '#include "av2/common/entropy_inits_modes.h"',
         '#include "av2/common/entropy_inits_mv.h"',
         'int main(void) { for(int q=0;q<4;q++) { uint64_t hash=14695981039346656037ULL;']
    for _, dims, n, expression, flags, _ in SPECS:
        for i, size in enumerate(dims):
            c.append(f'for(int {chr(97+i)}=0;{chr(97+i)}<{size};{chr(97+i)}++) {{')
        relation = '>=' if 'r' in flags else '=='
        c.append(f'{{ _Static_assert(sizeof({expression}) {relation} sizeof(avm_cdf_prob) * ({n} + 4), "axes");')
        c.append(f'avm_cdf_prob model[{n}+4]; memcpy(model,{expression},sizeof(model));')
        c.append(f'for(int j=0;j<{n}+4;j++)printf("%u ",model[j]); puts("");')
        c.append(f'for(int step=0;step<65;step++) {{ update_cdf(model,(step*7+step/3)%{n},{n});'
                 f' for(int j=0;j<{n};j++) hash=(hash^model[j])*1099511628211ULL; }} }}')
        c.extend('}' for _ in dims)
    c.append('printf("HASH %llu\\n",(unsigned long long)hash); } return 0; }')
    with tempdirs.temp_dir("av2-general-cdf") as directory:
        path = Path(directory) / 'probe.c'
        path.write_text('\n'.join(c))
        subprocess.run(['cc', '-O1', '-I' + str(source), '-I' + str(build), str(path),
                        '-o', str(path.with_suffix(''))], check=True)
        output = iter(subprocess.check_output([str(path.with_suffix(''))], text=True).splitlines())
    defaults = [[] for _ in SPECS]
    hashes = []
    for _ in range(4):
        for index, (_, dims, n, _, _, _) in enumerate(SPECS):
            rows = []
            for _ in itertools.product(*(range(size) for size in dims)):
                row = list(map(int, next(output).split()))
                assert len(row) == n + 4 and row[n] == 0
                rows.append(row)
            defaults[index].append(rows)
        hashes.append(int(next(output).split()[1]))
    assert next(output, None) is None, 'extra output'
    return defaults, hashes


def generate(defaults, hashes):
    tables = ['// Generated by tools/codegen/gen_av2_general_cdfs.py; do not edit.',
              '// Original AVM copyright Alliance for Open Media.',
              '// BSD-3-Clause-Clear + separate AOM patent terms; see THIRD_PARTY.md.',
              'use super::Initial;']
    for (name, dims, n, _, flags, _), categories in zip(SPECS, defaults):
        quantized = 'q' in flags
        if not quantized:
            assert all(category == categories[0] for category in categories)
        rows = iter(sum(categories, []) if quantized else categories[0])

        def array(shape):
            if not shape:
                row = next(rows)
                return f'Initial {{ inverse: {row[:n]}, rates: {row[n + 1:]} }}'
            return '[' + ',\n'.join(array(shape[1:]) for _ in range(shape[0])) + ']'
        shape = ([4] if quantized else []) + dims
        tables.append(f'pub(super) static {name.upper()}: {rust_type(f"Initial<{n}>", shape)} = {array(shape)};')
    tables.append('#[cfg(test)] pub(super) const TRACE_HASHES: [u64; 4] = ['
                  + ', '.join(f'{h:_}' for h in hashes) + '];')

    def build(expression, depth):
        if depth == 0:
            return expression + '.build()'
        if depth == 1:
            return expression + '.map(Initial::build)'
        return expression + '.map(|value| ' + build('value', depth - 1) + ')'

    def visit(root, action):
        lines = []
        for name, dims, *_ in SPECS:
            expression = f'{root}.{name}'
            if not dims:
                lines.append(action(expression if root == 'self' else '&mut ' + expression))
                continue
            expression = '&mut ' + expression
            for depth in range(len(dims)):
                lines.append(f'for v{depth} in {expression} {{')
                expression = f'v{depth}'
            lines.append(action(expression))
            lines.extend('}' for _ in dims)
        return lines

    def visit_pair(action):
        lines = []
        for name, dims, *_ in SPECS:
            left, right = f'self.{name}', f'other.{name}'
            if not dims:
                lines.append(action(left, f'&{right}'))
                continue
            for depth in range(len(dims)):
                source = f'&{right}' if depth == 0 else right
                lines.append(f'for (v{depth}, w{depth}) in {left}.iter_mut().zip({source}) {{')
                left, right = f'v{depth}', f'w{depth}'
            lines.append(action(left, right))
            lines.extend('}' for _ in dims)
        return lines

    model = ['//! The general AV2 decoder\'s probability models: every AVM v1.0.0 default',
             '//! its syntax reads so far, with AVM\'s adaptation rates (generated by',
             '//! `tools/codegen/gen_av2_general_cdfs.py`, which tool agents extend).',
             '//!', '//! Kept apart from [`crate::cdf_models::TileCdfs`], the encoder\'s subset,',
             '//! so the encoder\'s hot model copies stay small; both use the shared',
             '//! [`Cdf`] and entropy reader.',
             'use crate::entropy::Cdf;', '#[path = "cdfs/defaults.rs"] mod defaults;',
             '#[derive(Clone, Copy)] struct Initial<const N: usize> { inverse: [u16; N], rates: [u8; 3] }',
             'impl<const N: usize> Initial<N> { fn build(self) -> Cdf<N> {',
             'Cdf::new(self.inverse, self.rates).expect("independently verified AVM defaults") } }',
             '/// The coefficient CDF quantizer category of a frame base qindex (AVM `get_q_ctx`).',
             '#[allow(dead_code, reason = "the coefficient families use it")]',
             'fn category(qindex: u16) -> usize { match qindex { 0..=90 => 0, 91..=140 => 1, 141..=190 => 2, _ => 3 } }',
             '/// One tile\'s models.',
             '#[derive(Debug, Clone, PartialEq, Eq)] pub struct Cdfs {']
    for name, dims, n, expression, _, what in SPECS:
        model += [f'/// {what[0].upper() + what[1:]} (AVM `{expression}`).',
                  f'pub {name}: {rust_type(f"Cdf<{n}>", dims)},']
    model += ['}', 'impl Cdfs {', '/// The defaults of a frame at base quantizer `qindex`.',
              '#[must_use] #[allow(clippy::too_many_lines, reason = "generated: one line per model family")]',
              'pub fn new(qindex: u16) -> Self {',
              '#[allow(unused_variables, reason = "the coefficient families use it")] let q = category(qindex);',
              'Self {']
    for name, dims, _, _, flags, _ in SPECS:
        model.append(f'{name}: {build(f"defaults::{name.upper()}" + ("[q]" if "q" in flags else ""), len(dims))},')
    model += ['} }', '/// Move every model back along its adaptation-rate schedule (AVM',
              '/// `av2_reset_cdf_symbol_counters`) before later frames inherit them.',
              '#[allow(clippy::too_many_lines, reason = "generated: one loop per model family")]',
              'pub fn restart_adaptation(&mut self) {']
    model += visit('self', lambda e: f'{e}.restart_adaptation();')
    model += ['}', '/// Weigh in a second reference\'s final models at one eighth (AVM',
              '/// `av2_avg_cdf_symbols`, dav2d `dav2d_cdf_pri_sec_average`).',
              '#[allow(clippy::too_many_lines, reason = "generated: one loop per model family")]',
              'pub fn average_with(&mut self, other: &Self) {']
    model += visit_pair(lambda a, b: f'{a}.average_with({b});')
    model += ['}', '/// Divide every model by `1 << shift`: the first term of the average',
              '/// of several tiles\' final models (dav2d `dav2d_cdf_shift`).',
              '#[allow(clippy::too_many_lines, reason = "generated: one loop per model family")]',
              'pub fn scale_down(&mut self, shift: u32) {']
    model += visit('self', lambda e: f'{e}.scale_down(shift);')
    model += ['}', '/// Add `other` divided by `1 << shift`: a further term of the average',
              '/// of several tiles\' final models (dav2d `dav2d_cdf_shift_accumulate`).',
              '#[allow(clippy::too_many_lines, reason = "generated: one loop per model family")]',
              'pub fn add_scaled(&mut self, other: &Self, shift: u32) {']
    model += visit_pair(lambda a, b: f'{a}.add_scaled({b}, shift);')
    model += ['} }', '#[cfg(test)] mod tests { use super::*; use crate::entropy::Writer;',
              'fn trace<const N: usize>(cdf: &mut Cdf<N>, hash: &mut u64) { let mut writer = Writer::new();',
              'for step in 0..65 { writer.symbol((step * 7 + step / 3) % N, cdf).unwrap();',
              'for &value in cdf.inverse() { *hash = (*hash ^ u64::from(value)).wrapping_mul(1_099_511_628_211); } } }',
              '/// Our models update as AVM\'s `update_cdf` does, from the same defaults.',
              '#[test] #[allow(clippy::too_many_lines, reason = "generated: one loop per model family")]',
              'fn original_c_update_traces() {',
              'for (q, expected) in [0, 91, 141, 191].into_iter().zip(defaults::TRACE_HASHES) {',
              'let mut models = Cdfs::new(q); let mut hash = 14_695_981_039_346_656_037;']
    model += visit('models', lambda e: f'trace({e}, &mut hash);')
    model += ['assert_eq!(hash, expected); } } }']
    return {'cdfs/defaults.rs': '\n'.join(tables) + '\n', 'cdfs.rs': '\n'.join(model) + '\n'}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    head = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == PIN, f'AVM at {head}, expected {PIN}'
    defaults, hashes = probe(args.source, args.build)
    for name, text in generate(defaults, hashes).items():
        with tempdirs.temp_dir("av2-general-cdf-rust") as directory:
            path = Path(directory) / Path(name).name
            path.write_text(text)
            subprocess.run([str(Path.home() / '.cargo/bin/rustfmt'), '--edition', '2024',
                            '--config', 'skip_children=true', str(path)], check=True)
            text = path.read_text()
        target = OUT / name
        if args.write:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        else:
            assert target.read_text() == text, f'{name} differs; regenerate with --write'
    print(f'PASS: {len(SPECS)} model families; update-trace hashes {hashes}')


if __name__ == '__main__':
    main()
