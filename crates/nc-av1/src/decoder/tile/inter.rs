//! Inter blocks: their mode info syntax (AV1 §5.11.23
//! `inter_block_mode_info`, §5.11.25 `read_ref_frames` of single and
//! compound blocks and skip mode, §5.11.26 modes, compound modes and
//! `drl_mode`, §5.11.26 `assign_mv`, §5.11.28 `read_interintra_mode`,
//! §5.11.27 `read_motion_mode`, §5.11.29 `read_compound_type`, §5.11.30
//! interpolation filters) and the warp model of locally warped blocks
//! (§7.10.4 `find_warp_samples`, §7.11.3.8).
//!
//! Neighbour scans (overlappable neighbours, warp samples, filter
//! contexts) follow libaom `av1/common/obmc.h`, `mvref_common.c`
//! (`av1_findSamples`, `av1_selectSamples`) and `pred_common.c`
//! (BSD-2-Clause, see `THIRD_PARTY.md`).
use super::super::block_info::{InterFlags, Neighbours};
use super::super::block_size::BlockSize;
use super::super::frame_header::{INTERP_SWITCHABLE, TOTAL_REFS, WarpType};
use super::super::motion::{INTRA_FRAME, InterMode, MotionCell, NONE_FRAME};
use super::super::mv_stack::{MvBlock, MvStack, find_mv_stack};
use super::{BlockPos, FrameDecoder, MI_SIZE};
use crate::backend::{Av1Mv, PlaneIndex};
use crate::decoder_range::SymbolReader;
use crate::mvcoding::MvPrecision;
use nc_common::decode::DecodeError;
use nc_dsp::av1_mc::filter::{Av1Filter, Av1Filters};
use nc_dsp::av1_mc::mask::{AV1_WEDGE_SIZES, AV1_WEDGE_TYPES, Av1InterIntraMode};
use nc_dsp::av1_mc::scale::Av1ScaleSizes;
use nc_dsp::av1_mc::warp::{Av1Shear, Av1WarpSample, av1_setup_shear, av1_warp_estimation};

/// Reference names (§6.10.24).
const LAST_FRAME: i8 = 1;
const LAST2_FRAME: i8 = 2;
const LAST3_FRAME: i8 = 3;
const GOLDEN_FRAME: i8 = 4;
const BWDREF_FRAME: i8 = 5;
const ALTREF2_FRAME: i8 = 6;
const ALTREF_FRAME: i8 = 7;
/// `single_ref_p1` .. `single_ref_p6` as indices of the CDF row, and the
/// same bits' contexts for the compound reference bits that share them.
const SINGLE_REF_P1: usize = 0;
const SINGLE_REF_P2: usize = 1;
const SINGLE_REF_P3: usize = 2;
const SINGLE_REF_P4: usize = 3;
const SINGLE_REF_P5: usize = 4;
const SINGLE_REF_P6: usize = 5;
/// `uni_comp_ref`, `uni_comp_ref_p1`, `uni_comp_ref_p2`, and `comp_ref`,
/// `comp_ref_p1`, `comp_ref_p2` (`comp_bwdref`, `comp_bwdref_p1`) as
/// indices of their CDF rows.
const COMP_P: usize = 0;
const COMP_P1: usize = 1;
const COMP_P2: usize = 2;
/// Smallest side of compound blocks (`is_comp_ref_allowed`), 4x4 units.
const COMPOUND_MIN_MI: usize = 2;
/// `compound_mode` symbols.
const COMPOUND_MODES: usize = InterMode::COMPOUND.len();
/// `COMP_NEWMV_CTXS`: `NewMvContext` values the compound mode context
/// tells apart.
const COMP_NEWMV_CTXS: usize = 5;
/// `Compound_Mode_Ctx_Map` (§8.3.2 `compound_mode`), by `RefMvContext / 2`
/// and `NewMvContext`.
const COMPOUND_MODE_CTX_MAP: [[usize; COMP_NEWMV_CTXS]; 3] =
    [[0, 1, 1, 1, 1], [1, 2, 3, 4, 4], [4, 4, 5, 6, 7]];
/// `interp_filter` symbols (`SWITCHABLE_FILTERS`), also the "no filter"
/// neighbour type of the filter context.
const SWITCHABLE_FILTERS: usize = 3;
/// Filter context offsets of compound blocks (`INTER_FILTER_COMP_OFFSET`)
/// and of the horizontal filter (`INTER_FILTER_DIR_OFFSET`).
const INTER_FILTER_COMP_OFFSET: usize = SWITCHABLE_FILTERS + 1;
const INTER_FILTER_DIR_OFFSET: usize = 2 * (SWITCHABLE_FILTERS + 1);
/// `interintra_mode` and `motion_mode` symbols.
const INTERINTRA_MODES: usize = 4;
const MOTION_MODES: usize = 3;
/// `MiSize` range that may use inter-intra (`BLOCK_8X8` ..= `BLOCK_32X32`
/// in size order, which leaves out 8x32 and 32x8).
const INTERINTRA_SIZES: std::ops::RangeInclusive<usize> = 3..=9;
/// `LEAST_SQUARES_SAMPLES_MAX`: warp samples kept.
const LEAST_SQUARES_SAMPLES_MAX: usize = 8;
/// Sample selection threshold range (`av1_selectSamples`), in 1/8 samples.
const SAMPLE_THRESHOLD: (usize, usize) = (16, 112);
/// Eighths per sample.
const SUBPEL_PER_SAMPLE: i32 = 8;
/// `max_neighbor_obmc` by `Mi_Width_Log2` (or height): OBMC neighbours
/// used per edge.
pub(super) const MAX_NEIGHBOR_OBMC: [usize; 6] = [0, 1, 2, 3, 4, 4];
/// Largest neighbour step of the OBMC scan (`BLOCK_64X64`), 4x4 units.
pub(super) const OBMC_MAX_STEP: usize = 16;
/// Smallest side of blocks that may use OBMC and warped motion
/// (`is_motion_variation_allowed_bsize`), 4x4 units.
const MOTION_VARIATION_MIN_MI: usize = 2;

/// A local warp model that passed the shear test (§7.11.3.6).
#[derive(Debug, Clone, Copy)]
pub(super) struct WarpModel {
    pub params: [i32; 6],
    pub shear: Av1Shear,
}

/// `motion_mode` (§6.10.26).
#[derive(Debug, Clone, Copy)]
pub(super) enum MotionMode {
    Simple,
    Obmc,
    /// `WARPED_CAUSAL`. Its model is fitted where the block is predicted
    /// ([`FrameDecoder::local_warp`]), which keeps the mode info small.
    Warped,
}

/// Inter-intra prediction of a block (`interintra`).
#[derive(Debug, Clone, Copy)]
pub(super) struct InterIntra {
    pub mode: Av1InterIntraMode,
    /// `wedge_interintra`: the wedge index, or `None` for a smooth mask.
    pub wedge: Option<u8>,
}

/// How a compound block combines its two predictions (`compound_type`,
/// §6.10.29).
#[derive(Debug, Clone, Copy)]
pub(super) enum CompoundType {
    /// `COMPOUND_AVERAGE`.
    Average,
    /// `COMPOUND_DISTANCE`: weighted by the references' distances.
    Distance,
    /// `COMPOUND_WEDGE` with `wedge_index` and `wedge_sign`.
    Wedge { index: u8, sign: bool },
    /// `COMPOUND_DIFFWTD`; `inverse` is `mask_type`.
    DifferenceWeighted { inverse: bool },
}

/// The mode info of an inter block.
#[derive(Debug, Clone, Copy)]
pub(super) struct InterModes {
    /// `RefFrame[0]`, and `RefFrame[1]`: a reference for compound blocks,
    /// `INTRA_FRAME` for inter-intra, `NONE` otherwise.
    pub refs: [i8; 2],
    pub mode: InterMode,
    /// `Mv[]`; the second is zero for single-reference blocks.
    pub mv: [Av1Mv; 2],
    pub filters: Av1Filters,
    pub motion: MotionMode,
    pub interintra: Option<InterIntra>,
    /// The blend of a compound block.
    pub compound: Option<CompoundType>,
    /// What neighbours' contexts read of the block (`skip_mode`,
    /// `comp_group_idx`, `compound_idx`).
    pub flags: InterFlags,
}

impl InterModes {
    /// The block as the motion store records it.
    pub(super) fn cell(&self, size: BlockSize) -> MotionCell {
        MotionCell::inter(size, self.refs, self.mode, self.mv, self.filters)
    }
    /// The references the block predicts from (1 or 2).
    pub(super) fn lists(&self) -> usize {
        1 + usize::from(self.compound.is_some())
    }
}

/// The frame-level inter tools the block layer follows.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // the specification's flags
pub(in super::super) struct InterTools {
    /// `interpolation_filter`: a filter, or [`INTERP_SWITCHABLE`].
    pub interpolation_filter: u8,
    /// `is_motion_mode_switchable`.
    pub switchable_motion_mode: bool,
    pub allow_warped_motion: bool,
    /// `enable_interintra_compound`.
    pub interintra: bool,
    /// `enable_dual_filter`.
    pub dual_filter: bool,
    /// `enable_masked_compound`: compound blocks code `comp_group_idx`.
    pub masked_compound: bool,
    /// `enable_jnt_comp`: compound blocks code `compound_idx`.
    pub jnt_comp: bool,
    /// `use_ref_frame_mvs`: temporal MV candidates.
    pub use_ref_frame_mvs: bool,
    /// Order-hint distance from the frame to each reference.
    pub ref_offsets: [i32; TOTAL_REFS],
    /// The references whose vectors the frame keeps for later frames
    /// (§7.19), when the sequence uses temporal vectors.
    pub saved_sides: Option<[bool; TOTAL_REFS]>,
    /// The sizes each reference of another size than the frame scales
    /// between (`is_scaled`).
    pub scales: [Option<Av1ScaleSizes>; TOTAL_REFS],
}

/// Whether `size` may use wedge masks (`av1_is_wedge_used`).
pub(super) fn wedge_used(size: BlockSize) -> bool {
    AV1_WEDGE_SIZES.contains(&(size.width(), size.height()))
}

impl FrameDecoder<'_> {
    /// `inter_block_mode_info()` (§5.11.23); `skip_mode` blocks take
    /// their references and mode from the frame. In a frame with intra
    /// block copy every block decoded here is one (§5.11.7): it predicts
    /// from `INTRA_FRAME` with a DV.
    // Out of line, as are the prediction and the residual: inlined into one
    // 42 KB `inter_block`, the block's code did not stay in the 32 KB L1
    // instruction cache (a third of the decoder's I1 misses).
    #[inline(never)]
    pub(super) fn inter_modes(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
        skip_mode: bool,
    ) -> Result<InterModes, DecodeError> {
        // Each helper has one call site (intra block copy shares the
        // stack's), so the single-reference path inlines whole into the
        // block walk. The frame's flag, not a per-block one, selects intra
        // block copy: a parameter carried through the walk cost ~0.3%.
        let intrabc = self.header.allow_intrabc;
        let refs = if intrabc {
            [INTRA_FRAME, NONE_FRAME]
        } else {
            match self.compound_frame_references(r, block, near, skip_mode) {
                Some(refs) => refs,
                None => [self.read_single_reference(r, near), NONE_FRAME],
            }
        };
        let stack = self.mv_stack(block, refs);
        if intrabc {
            return self.intrabc_modes(r, block, &stack);
        }
        if refs[1] > INTRA_FRAME {
            self.compound_modes(r, block, near, &stack, refs, skip_mode)
        } else {
            self.single_modes(r, block, refs[0], &stack)
        }
    }

    /// The references of a skip-mode or compound block (§5.11.25), or
    /// `None` when the block is predicted from one reference, which the
    /// caller then reads.
    #[inline]
    fn compound_frame_references(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
        skip_mode: bool,
    ) -> Option<[i8; 2]> {
        if !skip_mode && !self.header.reference_select {
            return None;
        }
        self.read_compound_frame_references(r, block, near, skip_mode)
    }

    /// [`Self::compound_frame_references`] in a frame with skip mode or
    /// compound blocks: `SkipModeFrame[]`, or `comp_mode` and the compound
    /// references.
    // Out of line: our own streams code no compound frames.
    #[cold]
    #[inline(never)]
    fn read_compound_frame_references(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
        skip_mode: bool,
    ) -> std::option::Option<[i8; 2]> {
        if let (true, Some(frames)) = (skip_mode, self.header.skip_mode_frames) {
            return Some(frames);
        }
        let compound = self.header.reference_select
            && block.size.width_mi().min(block.size.height_mi()) >= COMPOUND_MIN_MI
            && r.boolean(&mut self.cdfs.comp_mode[near.comp_mode_context()]);
        if compound {
            Some(self.read_compound_references(r, near))
        } else {
            None
        }
    }

    /// The reference MV stack of `block` predicted from `refs`.
    #[expect(
        clippy::inline_always,
        reason = "one call site; left to the inliner it went out of line once intra block copy joined it (+15M instructions on static 720p)"
    )]
    #[inline(always)]
    pub(super) fn mv_stack(&self, block: BlockPos, refs: [i8; 2]) -> MvStack {
        let top_right = (
            block.mi_row as isize - 1,
            (block.mi_col + block.size.width_mi()) as isize,
        );
        find_mv_stack(
            &self.maps.motion,
            self.temporal.field(),
            &self.mv_frame,
            MvBlock {
                mi_col: block.mi_col,
                mi_row: block.mi_row,
                size: block.size,
                refs,
                top_right_decoded: self.decoded.luma_decoded(top_right.0, top_right.1),
                tile: self.tile,
            },
        )
    }

    /// The rest of `inter_block_mode_info()` for a block predicted from
    /// `reference` alone (or with inter-intra).
    fn single_modes(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        reference: i8,
        stack: &MvStack,
    ) -> Result<InterModes, DecodeError> {
        let (mv, mode) = self.inter_mode(r, stack)?;
        let interintra = self.read_interintra(r, block.size);
        let refs = if interintra.is_some() {
            [reference, INTRA_FRAME]
        } else {
            [reference, NONE_FRAME]
        };
        let mut modes = InterModes {
            refs,
            mode,
            mv: [mv, Av1Mv::ZERO],
            filters: Av1Filters::both(Av1Filter::Regular),
            motion: MotionMode::Simple,
            interintra,
            compound: None,
            flags: InterFlags::SINGLE,
        };
        if interintra.is_none() {
            modes.motion = self.read_motion_mode(r, block, &modes);
        }
        modes.filters = self.read_interp_filters(r, block, &modes);
        Ok(modes)
    }

    /// The rest of `inter_block_mode_info()` for a compound block with
    /// `refs`: its mode and vectors, blend and filters.
    // Out of line: our own streams have no compound blocks.
    #[cold]
    #[inline(never)]
    fn compound_modes(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        near: Neighbours,
        stack: &MvStack,
        refs: [i8; 2],
        skip_mode: bool,
    ) -> Result<InterModes, DecodeError> {
        let (mode, index) = if skip_mode {
            (InterMode::NearestNearestMv, 0)
        } else {
            let ctx = COMPOUND_MODE_CTX_MAP[stack.ref_mv_context() >> 1]
                [stack.new_mv_context().min(COMP_NEWMV_CTXS - 1)];
            let mode =
                InterMode::COMPOUND[r.symbol(&mut self.cdfs.compound_mode[ctx], COMPOUND_MODES)];
            let index = if mode == InterMode::NewNewMv {
                self.drl_index(r, stack, 0)
            } else if mode.has_near_mv() {
                self.drl_index(r, stack, 1)
            } else {
                0
            };
            (mode, index)
        };
        let mut mv = [Av1Mv::ZERO; 2];
        for (list, mv) in mv.iter_mut().enumerate() {
            *mv = self.assign_mv(r, stack, mode.single(list), index, list)?;
        }
        let compound = if skip_mode {
            CompoundType::Average
        } else {
            self.read_compound_type(r, block.size, near, refs)
        };
        let mut modes = InterModes {
            refs,
            mode,
            mv,
            filters: Av1Filters::both(Av1Filter::Regular),
            motion: MotionMode::Simple,
            interintra: None,
            compound: Some(compound),
            // `comp_group_idx` is set for masked blends, `compound_idx`
            // for all but distance weights.
            flags: InterFlags::new(
                skip_mode,
                matches!(
                    compound,
                    CompoundType::Wedge { .. } | CompoundType::DifferenceWeighted { .. }
                ),
                !matches!(compound, CompoundType::Distance),
            ),
        };
        modes.filters = self.read_interp_filters(r, block, &modes);
        Ok(modes)
    }

    /// `assign_mv()` (§5.11.26) of list `list` coded with single mode
    /// `mode` and `RefMvIdx` `index`.
    fn assign_mv(
        &mut self,
        r: &mut SymbolReader<'_>,
        stack: &MvStack,
        mode: InterMode,
        index: usize,
        list: usize,
    ) -> Result<Av1Mv, DecodeError> {
        Ok(match mode {
            InterMode::GlobalMv => stack.global(list),
            InterMode::NearestMv => stack.vector(0, list),
            InterMode::NearMv => stack.vector(index, list),
            _ => {
                let position = if stack.count <= 1 { 0 } else { index };
                let precision = self.mv_precision();
                self.cdfs
                    .mv
                    .read(r, stack.vector(position, list), precision)?
            }
        })
    }

    /// The references of a compound block (§5.11.25): a unidirectional
    /// pair, or one forward and one backward reference.
    fn read_compound_references(&mut self, r: &mut SymbolReader<'_>, near: Neighbours) -> [i8; 2] {
        let counts = near.reference_counts();
        // `comp_ref_type`: 0 is `UNIDIR_COMP_REFERENCE`.
        let bidirectional = r.boolean(&mut self.cdfs.comp_ref_type[near.comp_ref_type_context()]);
        let cdfs = &mut self.cdfs;
        if !bidirectional {
            let mut bit = |p: usize, ctx: usize, r: &mut SymbolReader<'_>| {
                r.boolean(&mut cdfs.uni_comp_ref[ctx][p])
            };
            return if bit(COMP_P, counts.single_ref_context(SINGLE_REF_P1), r) {
                [BWDREF_FRAME, ALTREF_FRAME]
            } else if bit(COMP_P1, counts.uni_comp_ref_p1_context(), r) {
                if bit(COMP_P2, counts.single_ref_context(SINGLE_REF_P5), r) {
                    [LAST_FRAME, GOLDEN_FRAME]
                } else {
                    [LAST_FRAME, LAST3_FRAME]
                }
            } else {
                [LAST_FRAME, LAST2_FRAME]
            };
        }
        let mut forward = |p: usize, single: usize, r: &mut SymbolReader<'_>| {
            r.boolean(&mut cdfs.comp_ref[counts.single_ref_context(single)][p])
        };
        let first = if forward(COMP_P, SINGLE_REF_P3, r) {
            if forward(COMP_P2, SINGLE_REF_P5, r) {
                GOLDEN_FRAME
            } else {
                LAST3_FRAME
            }
        } else if forward(COMP_P1, SINGLE_REF_P4, r) {
            LAST2_FRAME
        } else {
            LAST_FRAME
        };
        let mut backward = |p: usize, single: usize, r: &mut SymbolReader<'_>| {
            r.boolean(&mut cdfs.comp_bwdref[counts.single_ref_context(single)][p])
        };
        let second = if backward(COMP_P, SINGLE_REF_P2, r) {
            ALTREF_FRAME
        } else if backward(COMP_P1, SINGLE_REF_P6, r) {
            ALTREF2_FRAME
        } else {
            BWDREF_FRAME
        };
        [first, second]
    }

    /// `read_ref_frames()` of a single-reference block (§5.11.25).
    fn read_single_reference(&mut self, r: &mut SymbolReader<'_>, near: Neighbours) -> i8 {
        let counts = near.reference_counts();
        let cdfs = &mut self.cdfs.single_ref;
        let mut bit = |p: usize, r: &mut SymbolReader<'_>| {
            r.boolean(&mut cdfs[counts.single_ref_context(p)][p])
        };
        if bit(SINGLE_REF_P1, r) {
            if bit(SINGLE_REF_P2, r) {
                ALTREF_FRAME
            } else if bit(SINGLE_REF_P6, r) {
                ALTREF2_FRAME
            } else {
                BWDREF_FRAME
            }
        } else if bit(SINGLE_REF_P3, r) {
            if bit(SINGLE_REF_P5, r) {
                GOLDEN_FRAME
            } else {
                LAST3_FRAME
            }
        } else if bit(SINGLE_REF_P4, r) {
            LAST2_FRAME
        } else {
            LAST_FRAME
        }
    }

    /// `read_compound_type()` (§5.11.29) of a compound block outside skip
    /// mode.
    fn read_compound_type(
        &mut self,
        r: &mut SymbolReader<'_>,
        size: BlockSize,
        near: Neighbours,
        refs: [i8; 2],
    ) -> CompoundType {
        let comp_group_idx = self.inter_tools.masked_compound
            && r.boolean(&mut self.cdfs.comp_group_idx[near.comp_group_idx_context()]);
        if !comp_group_idx {
            if !self.inter_tools.jnt_comp {
                return CompoundType::Average;
            }
            let distance = |name: i8| self.inter_tools.ref_offsets[name as usize].abs();
            let ctx = near.compound_idx_context(distance(refs[0]) == distance(refs[1]));
            let compound_idx = r.boolean(&mut self.cdfs.compound_idx[ctx]);
            return if compound_idx {
                CompoundType::Average
            } else {
                CompoundType::Distance
            };
        }
        // `compound_type`: 0 is `COMPOUND_WEDGE`, which only wedge sizes
        // code.
        let index = size.index();
        let wedge = wedge_used(size) && !r.boolean(&mut self.cdfs.compound_type[index]);
        if wedge {
            let index = r.symbol(&mut self.cdfs.wedge_index[index], AV1_WEDGE_TYPES);
            CompoundType::Wedge {
                index: index as u8,
                sign: r.literal(1) == 1,
            }
        } else {
            CompoundType::DifferenceWeighted {
                inverse: r.literal(1) == 1,
            }
        }
    }

    /// The precision of coded vector differences.
    fn mv_precision(&self) -> MvPrecision {
        if self.header.force_integer_mv {
            MvPrecision::Integer
        } else if self.header.allow_high_precision_mv {
            MvPrecision::Eighth
        } else {
            MvPrecision::Quarter
        }
    }

    /// The inter mode, `drl_mode` and vector (§5.11.26).
    fn inter_mode(
        &mut self,
        r: &mut SymbolReader<'_>,
        stack: &MvStack,
    ) -> Result<(Av1Mv, InterMode), DecodeError> {
        if !r.boolean(&mut self.cdfs.new_mv[stack.new_mv_context()]) {
            let index = self.drl_index(r, stack, 0);
            let precision = self.mv_precision();
            let mv = self.cdfs.mv.read(r, stack.vector(index, 0), precision)?;
            return Ok((mv, InterMode::NewMv));
        }
        if !r.boolean(&mut self.cdfs.zero_mv[stack.zero_mv_context()]) {
            return Ok((stack.global(0), InterMode::GlobalMv));
        }
        if !r.boolean(&mut self.cdfs.ref_mv[stack.ref_mv_context()]) {
            return Ok((stack.vector(0, 0), InterMode::NearestMv));
        }
        let index = self.drl_index(r, stack, 1);
        Ok((stack.vector(index, 0), InterMode::NearMv))
    }

    /// `RefMvIdx` from the `drl_mode` run starting at list position `first`
    /// (0 for `NEWMV`, 1 for `NEARMV`; §5.11.26 `read_drl_mode`).
    fn drl_index(&mut self, r: &mut SymbolReader<'_>, stack: &MvStack, first: usize) -> usize {
        let mut index = first;
        for position in first..first + 2 {
            if stack.count > position + 1 {
                if !r.boolean(&mut self.cdfs.drl[stack.drl_context(position)]) {
                    break;
                }
                index = position + 1;
            }
        }
        index
    }

    /// `interintra`, its mode and wedge (§5.11.28), where the sequence
    /// and block size allow it.
    fn read_interintra(&mut self, r: &mut SymbolReader<'_>, size: BlockSize) -> Option<InterIntra> {
        if !self.inter_tools.interintra || !INTERINTRA_SIZES.contains(&size.index()) {
            return None;
        }
        let group = size.size_group();
        if !r.boolean(&mut self.cdfs.interintra[group]) {
            return None;
        }
        let mode = match r.symbol(&mut self.cdfs.interintra_mode[group], INTERINTRA_MODES) {
            0 => Av1InterIntraMode::Dc,
            1 => Av1InterIntraMode::Vertical,
            2 => Av1InterIntraMode::Horizontal,
            _ => Av1InterIntraMode::Smooth,
        };
        let index = size.index();
        let wedge = if wedge_used(size) && r.boolean(&mut self.cdfs.wedge_interintra[index]) {
            Some(r.symbol(&mut self.cdfs.wedge_index[index], AV1_WEDGE_TYPES) as u8)
        } else {
            None
        };
        Some(InterIntra { mode, wedge })
    }

    /// `read_motion_mode()` (§5.11.27, libaom `motion_mode_allowed`), with
    /// the warp model of a locally warped block.
    fn read_motion_mode(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        modes: &InterModes,
    ) -> MotionMode {
        if !self.inter_tools.switchable_motion_mode
            || block.size.width_mi().min(block.size.height_mi()) < MOTION_VARIATION_MIN_MI
            || self.global_warp_block(block.size, modes)
            || !self.has_overlappable_neighbour(block)
        {
            return MotionMode::Simple;
        }
        let index = block.size.index();
        let samples = self.warp_samples(block, modes.refs[0]);
        let warp_allowed =
            samples.len >= 1 && self.inter_tools.allow_warped_motion && !self.scaled(modes.refs[0]);
        if !warp_allowed || self.header.force_integer_mv {
            let obmc = r.boolean(&mut self.cdfs.obmc[index]);
            return if obmc {
                MotionMode::Obmc
            } else {
                MotionMode::Simple
            };
        }
        match r.symbol(&mut self.cdfs.motion_mode[index], MOTION_MODES) {
            0 => MotionMode::Simple,
            1 => MotionMode::Obmc,
            _ => MotionMode::Warped,
        }
    }

    /// The local warp model of a `WARPED_CAUSAL` block (§7.11.3.8), or
    /// `None` when the fitted model is unusable (`LocalValid` 0), which
    /// predicts by translation. The neighbours it is fitted to are those
    /// `read_motion_mode` found (the block's own units do not count).
    pub(super) fn local_warp(&self, block: BlockPos, modes: &InterModes) -> Option<WarpModel> {
        let samples = self.warp_samples(block, modes.refs[0]);
        warp_model(samples.as_slice(), block, modes.mv[0])
    }

    /// A `GLOBALMV` block of at least 8x8 whose reference has a
    /// non-translational global model (libaom `is_global_mv_block`), outside
    /// integer-vector frames: it is warped by that model.
    pub(super) fn global_warp_block(&self, size: BlockSize, modes: &InterModes) -> bool {
        !self.header.force_integer_mv && self.global_warp_mode(size, modes)
    }

    /// `is_global_mv_block` of the block's first reference.
    fn global_warp_mode(&self, size: BlockSize, modes: &InterModes) -> bool {
        self.global_warp_list(size, modes, 0)
    }

    /// `is_global_mv_block` of reference list `list` of the block.
    pub(super) fn global_warp_list(
        &self,
        size: BlockSize,
        modes: &InterModes,
        list: usize,
    ) -> bool {
        let kind = self.header.global_motion[modes.refs[list] as usize].kind;
        modes.mode.is_global()
            && kind > WarpType::Translation
            && size.width_mi().min(size.height_mi()) >= MOTION_VARIATION_MIN_MI
    }

    /// Whether an inter block borders an inter block above or to the left
    /// (libaom `av1_count_overlappable_neighbors`).
    fn has_overlappable_neighbour(&self, block: BlockPos) -> bool {
        let mut found = false;
        self.overlappable_above(block, usize::MAX, |_, _, _| found = true);
        if !found {
            self.overlappable_left(block, usize::MAX, |_, _, _| found = true);
        }
        found
    }

    /// Visit up to `limit` inter blocks along the row above `block` (libaom
    /// `foreach_overlappable_nb_above`): `visit(offset, units, cell)` with
    /// the neighbour's column offset and overlap width in 4x4 units. Pairs
    /// of 4-wide blocks count as one, represented by the right one.
    pub(super) fn overlappable_above(
        &self,
        block: BlockPos,
        limit: usize,
        mut visit: impl FnMut(usize, usize, &MotionCell),
    ) {
        if !block.available(PlaneIndex::Y).0 {
            return;
        }
        let grid = &self.maps.motion;
        let end = (block.mi_col + block.size.width_mi()).min(self.header.width / MI_SIZE);
        let mut count = 0;
        let mut col = block.mi_col;
        while col < end && count < limit {
            let mut cell = grid.at(block.mi_row - 1, col);
            let mut step = cell.width_mi().min(OBMC_MAX_STEP);
            if step == 1 {
                col &= !1;
                cell = grid.at(block.mi_row - 1, col + 1);
                step = 2;
            }
            if cell.is_inter() {
                count += 1;
                visit(col - block.mi_col, block.size.width_mi().min(step), cell);
            }
            col += step;
        }
    }

    /// [`Self::overlappable_above`] down the column left of `block`.
    pub(super) fn overlappable_left(
        &self,
        block: BlockPos,
        limit: usize,
        mut visit: impl FnMut(usize, usize, &MotionCell),
    ) {
        if !block.available(PlaneIndex::Y).1 {
            return;
        }
        let grid = &self.maps.motion;
        let end = (block.mi_row + block.size.height_mi()).min(self.header.height / MI_SIZE);
        let mut count = 0;
        let mut row = block.mi_row;
        while row < end && count < limit {
            let mut cell = grid.at(row, block.mi_col - 1);
            let mut step = cell.height_mi().min(OBMC_MAX_STEP);
            if step == 1 {
                row &= !1;
                cell = grid.at(row + 1, block.mi_col - 1);
                step = 2;
            }
            if cell.is_inter() {
                count += 1;
                visit(row - block.mi_row, block.size.height_mi().min(step), cell);
            }
            row += step;
        }
    }

    /// The neighbours predicted from `reference` alone that a locally
    /// warped block fits its model to (libaom `av1_findSamples`), for
    /// blocks that may use warped motion.
    fn warp_samples(&self, block: BlockPos, reference: i8) -> WarpSamples {
        let mut samples = WarpSamples::default();
        let (w4, h4) = block.units();
        let grid = &self.maps.motion;
        let (mi_cols, mi_rows) = (self.header.width / MI_SIZE, self.header.height / MI_SIZE);
        let matches = |cell: &MotionCell| cell.refs == [reference, NONE_FRAME];
        let mut push = |cell: &MotionCell, (row, col): (isize, isize), (sign_r, sign_c)| {
            if matches(cell) {
                samples.push(sample(cell, (row, col), (sign_r, sign_c)));
            }
        };
        let (above, left) = block.available(PlaneIndex::Y);
        let (mut do_top_left, mut do_top_right) = (true, true);
        if above {
            let first = grid.at(block.mi_row - 1, block.mi_col);
            let width = first.width_mi();
            if w4 <= width {
                let col_offset = -((block.mi_col % width) as isize);
                do_top_left &= col_offset == 0;
                do_top_right &= col_offset + width as isize <= w4 as isize;
                push(first, (0, col_offset), (-1, 1));
            } else {
                let mut i = 0;
                while i < w4.min(mi_cols - block.mi_col) {
                    let cell = grid.at(block.mi_row - 1, block.mi_col + i);
                    push(cell, (0, i as isize), (-1, 1));
                    i += cell.width_mi();
                }
            }
        }
        if left {
            let first = grid.at(block.mi_row, block.mi_col - 1);
            let height = first.height_mi();
            if h4 <= height {
                let row_offset = -((block.mi_row % height) as isize);
                do_top_left &= row_offset == 0;
                push(first, (row_offset, 0), (1, -1));
            } else {
                let mut i = 0;
                while i < h4.min(mi_rows - block.mi_row) {
                    let cell = grid.at(block.mi_row + i, block.mi_col - 1);
                    push(cell, (i as isize, 0), (1, -1));
                    i += cell.height_mi();
                }
            }
        }
        if do_top_left && above && left {
            let cell = grid.at(block.mi_row - 1, block.mi_col - 1);
            push(cell, (0, 0), (-1, -1));
        }
        let right = block.mi_col + w4;
        let top_right_decoded = self
            .decoded
            .luma_decoded(block.mi_row as isize - 1, right as isize);
        if do_top_right
            && w4.max(h4) <= OBMC_MAX_STEP
            && top_right_decoded
            && right < self.tile.col_end
            && above
        {
            let cell = grid.at(block.mi_row - 1, right);
            push(cell, (0, w4 as isize), (-1, 1));
        }
        samples
    }

    /// `read_interp_filter` (§5.11.30, libaom `read_mb_interp_filter`):
    /// the frame's filter, or the block's.
    #[inline]
    fn read_interp_filters(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        modes: &InterModes,
    ) -> Av1Filters {
        let frame = self.inter_tools.interpolation_filter;
        if frame == INTERP_SWITCHABLE {
            self.read_switchable_filters(r, block, modes)
        } else {
            Av1Filters::both(Av1Filter::ALL[usize::from(frame)])
        }
    }

    /// The filters of a block in a frame with switchable filters.
    fn read_switchable_filters(
        &mut self,
        r: &mut SymbolReader<'_>,
        block: BlockPos,
        modes: &InterModes,
    ) -> nc_dsp::av1_mc::filter::Av1Filters {
        // Warped blocks need no filter (`av1_is_interp_needed`): locally
        // warped ones, and global ones whose every reference has a
        // non-translational model (libaom's `is_nontrans_global_motion`
        // also counts identity models as non-translational); nor do
        // skip-mode blocks.
        let global = modes.mode.is_global()
            && block.size.width_mi().min(block.size.height_mi()) >= MOTION_VARIATION_MIN_MI
            && modes.refs[..modes.lists()].iter().all(|&name| {
                self.header.global_motion[name as usize].kind != WarpType::Translation
            });
        if modes.flags.skip_mode() || global || matches!(modes.motion, MotionMode::Warped) {
            return Av1Filters::both(Av1Filter::Regular);
        }
        let compound = modes.compound.is_some();
        let mut read = [Av1Filter::Regular; 2];
        for (dir, filter) in read.iter_mut().enumerate() {
            let ctx = self.filter_context(block, modes.refs[0], compound, dir);
            let symbol = r.symbol(&mut self.cdfs.interp_filter[ctx], SWITCHABLE_FILTERS);
            *filter = Av1Filter::ALL[symbol];
            if !self.inter_tools.dual_filter {
                break;
            }
        }
        if !self.inter_tools.dual_filter {
            read[1] = read[0];
        }
        Av1Filters {
            x: read[1],
            y: read[0],
        }
    }

    /// The `interp_filter[dir]` context (§8.3.2, libaom
    /// `av1_get_pred_context_switchable_interp`): the neighbours' filters
    /// in that direction where they use the block's first reference, and
    /// whether the block is `compound`.
    fn filter_context(&self, block: BlockPos, reference: i8, compound: bool, dir: usize) -> usize {
        let grid = &self.maps.motion;
        let kind = |cell: &MotionCell| {
            if cell.refs.contains(&reference) {
                let filters = cell.filters();
                (if dir == 0 { filters.y } else { filters.x }) as usize
            } else {
                SWITCHABLE_FILTERS
            }
        };
        let (above, left) = block.available(PlaneIndex::Y);
        let left = if left {
            kind(grid.at(block.mi_row, block.mi_col - 1))
        } else {
            SWITCHABLE_FILTERS
        };
        let above = if above {
            kind(grid.at(block.mi_row - 1, block.mi_col))
        } else {
            SWITCHABLE_FILTERS
        };
        let neighbours = if left == above || above == SWITCHABLE_FILTERS {
            left
        } else if left == SWITCHABLE_FILTERS {
            above
        } else {
            SWITCHABLE_FILTERS
        };
        dir * INTER_FILTER_DIR_OFFSET
            + usize::from(compound) * INTER_FILTER_COMP_OFFSET
            + neighbours
    }
}

/// Up to [`LEAST_SQUARES_SAMPLES_MAX`] warp samples (`NumSamples`).
#[derive(Debug, Default)]
struct WarpSamples {
    samples: [Av1WarpSample; LEAST_SQUARES_SAMPLES_MAX],
    len: usize,
}

impl WarpSamples {
    /// Add a sample unless the list is full.
    fn push(&mut self, sample: Av1WarpSample) {
        if self.len < LEAST_SQUARES_SAMPLES_MAX {
            self.samples[self.len] = sample;
            self.len += 1;
        }
    }
    fn as_slice(&self) -> &[Av1WarpSample] {
        &self.samples[..self.len]
    }
}

/// One warp sample from neighbour `cell` at 4x4 offset (`row`, `col`)
/// from the block (libaom `record_samples`): the neighbour's centre and
/// where its vector moves it, in 1/8 samples.
fn sample(
    cell: &MotionCell,
    (row, col): (isize, isize),
    (sign_r, sign_c): (isize, isize),
) -> Av1WarpSample {
    let half_w = (cell.width_mi() * MI_SIZE / 2) as isize;
    let half_h = (cell.height_mi() * MI_SIZE / 2) as isize;
    let x = (col * MI_SIZE as isize + sign_c * half_w - 1) as i32 * SUBPEL_PER_SAMPLE;
    let y = (row * MI_SIZE as isize + sign_r * half_h - 1) as i32 * SUBPEL_PER_SAMPLE;
    let mv = cell.mv[0];
    Av1WarpSample {
        source: (x, y),
        destination: (x + i32::from(mv.col), y + i32::from(mv.row)),
    }
}

/// The local warp model of a `WARPED_CAUSAL` block with vector `mv`
/// (§7.11.3.8): samples whose motion differs much from the block's are
/// dropped first (libaom `av1_selectSamples`, keeping at least one).
fn warp_model(samples: &[Av1WarpSample], block: BlockPos, mv: Av1Mv) -> Option<WarpModel> {
    let (w, h) = (block.size.width(), block.size.height());
    let threshold = w.max(h).clamp(SAMPLE_THRESHOLD.0, SAMPLE_THRESHOLD.1) as i32;
    let mut kept = WarpSamples::default();
    for s in samples {
        let dx = s.destination.0 - s.source.0 - i32::from(mv.col);
        let dy = s.destination.1 - s.source.1 - i32::from(mv.row);
        if samples.len() == 1 || dx.abs() + dy.abs() <= threshold {
            kept.push(*s);
        }
    }
    if kept.len == 0 {
        kept.push(samples[0]);
    }
    let params = av1_warp_estimation(
        kept.as_slice(),
        (i32::from(mv.row), i32::from(mv.col)),
        (
            (block.mi_col * MI_SIZE) as u32,
            (block.mi_row * MI_SIZE) as u32,
        ),
        (w as u32, h as u32),
    )?;
    let shear = av1_setup_shear(&params)?;
    Some(WarpModel { params, shear })
}
