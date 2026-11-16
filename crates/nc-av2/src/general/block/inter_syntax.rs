//! An inter block's syntax (dav2d `decode_b`'s inter branch, AVM
//! `read_inter_block_mode_info`): the TIP flag, single or compound
//! references, the mode, adaptive MVD, the joint scale, compound warps,
//! DRL indices, the residual precision and residuals with derived signs,
//! MV refinement, the compound blend and weight, block-adaptive weights,
//! inter-intra, the warp syntax and the interpolation filter.
//!
//! Ported from dav2d commit 237229e `src/decode.c` (`decode_b`,
//! `read_amvd`, `read_mv_residual`, `read_wedge_idx`, `jmvd_scale`) and
//! `src/env.h` (`get_comp_ctx`, `get_snglref_ctx`, `get_compref_ctx`,
//! `get_warp_ctx`, `get_filter_ctx`); BSD-2-Clause, see
//! `docs/licenses/dav2d`.
use super::inter_record::{
    CompoundType, EQUAL_WEIGHT, InterMode, InterRecord, MotionMode, NO_REFERENCE, Reference,
    TIP_FRAME, WARP_DELTA_ABSENT,
};
use super::motion::Mv;
use super::record::BlockRecord;
use super::size::BlockSize;
use super::syntax::{Neighbour, Neighbours};
use super::{Tile, context};
use crate::entropy::Cdf;
use crate::general::cdfs::Cdfs;
use crate::general::frame_header::filter_mode;
use nc_common::decode::{DecodeError, Result};

/// The residual precisions (dav2d `mv_prec`, 0..6 for 8, 4, 2, 1, 1/2,
/// 1/4, 1/8 samples): whole samples, and the finest.
const PRECISION_INTEGER: u8 = 3;
const PRECISION_EIGHTH: u8 = 6;
/// Precisions at or above this never derive a residual sign.
const PRECISION_SIGN_DERIVE_MAX: u8 = 5;
/// The frame precision of eighth samples (`mv_precision` 3).
const FRAME_PRECISION_EIGHTH: u8 = 3;
/// The reduced precisions a block may pick instead of the frame's, by
/// whether the frame's is eighth samples (dav2d `mv_prec_tbl`).
const REDUCED_PRECISIONS: [[u8; 3]; 2] = [[3, 1, 0], [4, 3, 1]];
/// A residual's shell classes: `9 + precision` in all, the lower half
/// first; the eighth-sample precision's last two share a symbol.
const SHELL_CLASS_BASE: u32 = 9;
const SHELL_LAST_PAIR_SUM: u32 = 21;
/// The block-adaptive weight kinds: implicit (1), explicit (2, 3).
const BAWP_EXPLICIT: u8 = 2;
/// Bits of the explicit kind's context above the kind.
const BAWP_CONTEXT_SHIFT: u32 = 2;
/// The warp model candidates (index 0..=3).
const WARP_REFERENCES: u8 = 3;
/// WARPMV codes a residual only from the first two candidates.
const WARP_RESIDUAL_MAX_REFERENCE: u8 = 2;
/// Warp delta magnitudes: the first symbol's escape, steps by precision.
const WARP_DELTA_ESCAPE: usize = 7;
const WARP_DELTA_STEP: i8 = 2;
/// Compound weight indices coded (four flags) and their weights in
/// sixteenths, for references on one side or both (dav2d
/// `cwp_weighting_factor`).
const WEIGHT_FLAGS: usize = 4;
const WEIGHTS: [[i8; 5]; 2] = [[8, 12, 4, 10, 6], [8, 12, 4, 20, -4]];
/// TIP blocks' global weights (dav2d `dav2d_tip_wts`).
pub(super) const TIP_WEIGHTS: [i8; 8] = [8, 12, 16, 18, 20, 4, 6, -4];
/// Blocks whose larger side is at most this many units may use wedges.
const WEDGE_MAX_UNITS: u32 = 16;
/// The switchable filters (regular, smooth, sharp): a context neighbour
/// without one counts as this.
const SWITCHABLE_FILTERS: u8 = 3;
/// Filter contexts of compound blocks follow the single ones'.
const COMPOUND_FILTER_CONTEXT: usize = 4;
/// The wedge index of each (angle, distance) code; -1 is not used
/// (dav2d `wedge_angle_dist2idx`).
const WEDGE_INDICES: [[i8; 4]; 20] = [
    [-1, 0, 1, 2],
    [3, 4, 5, 6],
    [7, 8, 9, 10],
    [11, 12, 13, 14],
    [15, 16, 17, 18],
    [-1, 19, 20, 21],
    [22, 23, 24, 25],
    [26, 27, 28, 29],
    [30, 31, 32, 33],
    [34, 35, 36, 37],
    [-1, 38, 39, 40],
    [-1, 41, 42, 43],
    [-1, 44, 45, 46],
    [-1, 47, 48, 49],
    [-1, 50, 51, 52],
    [-1, 53, 54, 55],
    [-1, 56, 57, 58],
    [-1, 59, 60, 61],
    [-1, 62, 63, 64],
    [-1, 65, 66, 67],
];
/// Angles per wedge quadrant.
const WEDGE_QUADRANT_ANGLES: usize = 5;
/// Angle codes whose distances start at 1: outside 1..=9, and 5.
const WEDGE_CENTRED_ANGLES: std::ops::RangeInclusive<usize> = 1..=9;
const WEDGE_UNCENTRED_ANGLE: usize = 5;

/// What the model-selection code of a residual's shell needs: the offset
/// and component-split families (AVM's `nmvc` for inter vectors, `ndvc`
/// for block copies).
pub(super) struct ShellModels {
    pub offset_low: fn(&mut Cdfs, usize) -> &mut Cdf<2>,
    pub offset_class2: fn(&mut Cdfs) -> &mut Cdf<2>,
    pub offset_high: fn(&mut Cdfs, usize) -> &mut Cdf<2>,
    pub column_greater: fn(&mut Cdfs, usize) -> &mut Cdf<2>,
    pub column_index: fn(&mut Cdfs, usize) -> &mut Cdf<2>,
}

/// The inter vectors' shell models.
const INTER_SHELL: ShellModels = ShellModels {
    offset_low: |m, class| &mut m.mv_offset_low[class],
    offset_class2: |m| &mut m.mv_offset_class2,
    offset_high: |m, bit| &mut m.mv_offset_high[bit],
    column_greater: |m, step| &mut m.mv_column_greater[step],
    column_index: |m, context| &mut m.mv_column_index[context],
};

/// Classes 0 and 1 code their offset in one flag, class 2 in a
/// truncated unary of up to three.
const CLASS_TWO: u32 = 2;
const CLASS_TWO_MAX_OFFSET: u32 = 3;
/// Column-index contexts stop at class 3.
const MAX_COLUMN_CONTEXT: u32 = 3;
/// Pair-index steps: the second flag needs a shell index of 4, the
/// uniform remainder of 6.
const PAIR_SECOND_MIN: u32 = 4;
const PAIR_REMAINDER_MIN: u32 = 6;

impl Tile<'_, '_> {
    /// Read an inter block's syntax (everything up to its transform
    /// partition).
    #[allow(clippy::too_many_lines, reason = "dav2d decode_b's inter branch")]
    pub(super) fn inter_syntax(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        neighbours: Neighbours,
    ) -> Result<(InterRecord, u8)> {
        let size = record
            .luma
            .ok_or(DecodeError::Malformed("AV2 inter block without luma"))?;
        let (bw4, bh4) = size.units();
        let mut inter = InterRecord {
            references: [0, NO_REFERENCE],
            mode: InterMode::Near,
            drl: [0; 2],
            residual: [Mv::default(); 2],
            precision: [0; 2],
            amvd: false,
            motion_mode: MotionMode::Translation,
            warp_reference: 0,
            warp_with_residual: false,
            warp_delta: [0; 4],
            warp_inter_intra: false,
            inter_intra_mode: 0,
            wedge_index: -1,
            wedge_sign: false,
            mask_sign: false,
            compound: CompoundType::None,
            weight: EQUAL_WEIGHT,
            refine_mv: 0,
            bawp: [0; 2],
            filter: filter_mode::REGULAR,
        };
        let tip = self.tip_flag(depth, record, neighbours)?;
        let compound = if record.skip_mode {
            true
        } else if !tip && self.frame.reference_select && bw4 * bh4 >= 4 {
            let context = self.compound_context(neighbours);
            self.bool(|m| &mut m.comp[context])?
        } else {
            false
        };
        // dav2d `mvprec_def`: 1 the frame's precision, 2 a reduced one.
        let mut precision_context = 1;
        let has_filter = if record.skip_mode {
            self.skip_mode_syntax(depth, &mut inter, neighbours)?;
            false
        } else if compound {
            self.compound_syntax(depth, size, &mut inter, neighbours, &mut precision_context)?
        } else {
            self.single_syntax(
                depth,
                record,
                tip,
                &mut inter,
                neighbours,
                &mut precision_context,
            )?
        };
        inter.filter = self.interp_filter(depth, (size, record.skip_mode), &inter, has_filter)?;
        Ok((inter, precision_context))
    }

    /// `is_tip`: whether the block predicts from the TIP frame.
    fn tip_flag(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        neighbours: Neighbours,
    ) -> Result<bool> {
        let size = record.luma.unwrap_or(BlockSize::B4x4);
        let (bw4, bh4) = size.units();
        if record.skip_mode
            || self.frame.tip.mode == 0
            || record.luma != record.chroma
            || bw4.min(bh4) < 2
        {
            return Ok(false);
        }
        let context = (0..neighbours.count)
            .filter(|&n| self.neighbour_motion(neighbours.at[n])[0] == TIP_FRAME)
            .count();
        let tip = self.bool(|m| &mut m.tip[context])?;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-tip[ctx={context},{}]: r={range}",
            u8::from(tip)
        );
        Ok(tip)
    }

    /// A context neighbour's references (none where absent).
    fn neighbour_motion(&self, neighbour: Option<Neighbour>) -> [Reference; 2] {
        neighbour.map_or([NO_REFERENCE; 2], |neighbour| {
            [
                self.neighbour_value(neighbour, |edge| &edge.reference[0]) as i8,
                self.neighbour_value(neighbour, |edge| &edge.reference[1]) as i8,
            ]
        })
    }

    /// The compound flag's context (dav2d `get_comp_ctx`, with intra's
    /// direction -1 counting as true where dav2d tests it as a truth
    /// value).
    fn compound_context(&self, neighbours: Neighbours) -> usize {
        let refs = &self.references;
        let side = |n: usize| {
            let neighbour = neighbours.at[n].expect("a counted neighbour");
            let references = self.neighbour_motion(Some(neighbour));
            let intrabc = self.neighbour_value(neighbour, |edge| &edge.intrabc) != 0;
            (references, intrabc)
        };
        let one_sided = |(references, intrabc): ([Reference; 2], bool)| {
            usize::from(!intrabc && refs.direction(references[0]) != 0)
        };
        match neighbours.count {
            2 => {
                let (a, b) = (side(0), side(1));
                match (a.0[1] == NO_REFERENCE, b.0[1] == NO_REFERENCE) {
                    (true, true) => {
                        usize::from((refs.direction(a.0[0]) == 1) ^ (refs.direction(b.0[0]) == 1))
                    }
                    (true, false) => 2 + one_sided(a),
                    (false, true) => 2 + one_sided(b),
                    (false, false) => 4,
                }
            }
            1 => {
                let a = side(0);
                if a.0[1] == NO_REFERENCE {
                    usize::from(!a.1 && refs.direction(a.0[0]) == 1)
                } else {
                    3
                }
            }
            _ => 1,
        }
    }

    /// The counts of each reference (and intra, index 0; TIP, index 8)
    /// among the context neighbours, as the reference flags' contexts
    /// read them.
    fn reference_counts(&self, neighbours: Neighbours) -> ([i32; 9], i32) {
        let mut counts = [0; 9];
        for n in 0..neighbours.count {
            for reference in self.neighbour_motion(neighbours.at[n]) {
                counts[(i32::from(reference) + 1) as usize] += 1;
            }
        }
        let remaining = neighbours.count as i32 * 2 - counts[0] - counts[8];
        (counts, remaining)
    }

    /// A skip-mode block: its DRL index, references (a neighbour's pair,
    /// else the frame's) and the averaged compound.
    fn skip_mode_syntax(
        &mut self,
        depth: usize,
        inter: &mut InterRecord,
        neighbours: Neighbours,
    ) -> Result<()> {
        let max_drl_bits = self.frame.max_drl_bits;
        let mut drl = 0;
        let mut context = 0;
        while drl < max_drl_bits && self.bool(|m| &mut m.skip_mode_drl[context])? {
            drl += 1;
            context += usize::from(context < 2);
        }
        inter.drl[0] = drl;
        let range = self.reader.range();
        trace!(self, depth, "Post-drl[{drl},{drl}]: r={range}");
        inter.references = self.references.skip_mode;
        for n in 0..neighbours.count {
            let motion = self.neighbour_motion(neighbours.at[n]);
            if motion[0] == TIP_FRAME {
                let [a, b] = self.frame.tip.references.map(|r| r as Reference);
                inter.references = [a.min(b), a.max(b)];
                break;
            } else if motion[1] != NO_REFERENCE {
                inter.references = motion;
                break;
            } else if motion[0] != NO_REFERENCE {
                break;
            }
        }
        inter.compound = CompoundType::Average;
        inter.mode = InterMode::NearNear;
        Ok(())
    }

    /// Whether the block's top or left neighbours (at the far and near
    /// units) predict from `reference` (dav2d `decode_b`'s `match_refs`,
    /// with the superblock-top cache, AVM #1091).
    fn neighbours_use(&self, size: BlockSize, reference: Reference) -> bool {
        let (bw4, bh4) = size.units();
        let b = self.bounds;
        let left = context::left_index(self.row);
        let column = self.column as usize;
        let matches = |edge: &context::Edge, index: usize| {
            edge.reference[0][index] as i8 == reference
                || edge.reference[1][index] as i8 == reference
        };
        let cached = |index: usize| {
            self.above_cache.reference[0][index] as i8 == reference
                || self.above_cache.reference[1][index] as i8 == reference
        };
        let at_superblock_top = self.row & (self.geometry.superblock_units - 1) == 0;
        let offset = column & (context::LEFT_UNITS - 1);
        matches(&self.contexts.left, left)
            || (self.row + bh4 <= b.row_end
                && matches(&self.contexts.left, left + bh4 as usize - 1))
            || if at_superblock_top {
                cached(offset & !1)
                    || (((self.column + bw4 - 2) & !1) < b.column_end
                        && cached((offset + bw4 as usize - 2) & !1))
            } else {
                matches(&self.contexts.above, column)
                    || (self.column + bw4 <= b.column_end
                        && matches(&self.contexts.above, column + bw4 as usize - 1))
            }
    }

    /// The motion modes of the spatial neighbours (translation where
    /// absent).
    fn neighbour_motion_modes(&self, size: BlockSize) -> [u8; 2] {
        self.spatial_neighbours(size).map(|neighbour| {
            neighbour.map_or(MotionMode::Translation as u8, |neighbour| {
                self.neighbour_value(neighbour, |edge| &edge.motion_mode)
            })
        })
    }

    /// A compound block's references, mode, residuals and blend; returns
    /// whether it codes an interpolation filter.
    #[allow(clippy::too_many_lines, reason = "dav2d decode_b's compound branch")]
    fn compound_syntax(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &mut InterRecord,
        neighbours: Neighbours,
        precision_context: &mut u8,
    ) -> Result<bool> {
        let (bw4, bh4) = size.units();
        inter.references = self.compound_references(neighbours)?;
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-ref[{},{}]: r={range}", inter.references[0], inter.references[1]
        );
        let [r0, r1] = inter.references;
        let refs = self.references;
        let context = self.reference_mode_context(size, inter.references, true);
        let mut mode = if r0 == r1 {
            let mode = InterMode::NearNear as u8
                + self.symbol(|m| &mut m.comp_mode_same_refs[context])? as u8;
            // The same-reference list has no NEWMV_NEARMV.
            mode + u8::from(mode > InterMode::NearNew as u8)
        } else {
            let joint_context =
                usize::from(refs.distance[r0 as usize] != -refs.distance[r1 as usize]);
            if self.bool(|m| &mut m.comp_mode_joint[joint_context])? {
                InterMode::JointNew as u8
            } else {
                InterMode::NearNear as u8 + self.symbol(|m| &mut m.comp_mode[context])? as u8
            }
        };
        let mut coded = mode;
        let optical_flow = refs.optical_flow;
        if optical_flow != 0
            && mode != InterMode::GlobalGlobal as u8
            && bw4.min(bh4) >= 2
            && refs.future[r0 as usize] != refs.future[r1 as usize]
        {
            let opfl_context = usize::from(mode > InterMode::NearNear as u8);
            if optical_flow == 2 || self.bool(|m| &mut m.opfl[opfl_context])? {
                mode += 6 - u8::from(mode >= InterMode::GlobalGlobal as u8);
                if optical_flow == 1 {
                    coded = mode;
                }
            }
        }
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-comp_inter_mode[ctx={context},{coded}]: r={range}"
        );
        inter.mode = InterMode::from_u8(mode).ok_or(DecodeError::Malformed("AV2 compound mode"))?;
        let coded = InterMode::from_u8(coded).ok_or(DecodeError::Malformed("AV2 compound mode"))?;
        let new_mode = matches!(
            inter.mode,
            InterMode::NearNew
                | InterMode::NewNear
                | InterMode::NewNew
                | InterMode::JointNew
                | InterMode::OpflNearNew
                | InterMode::OpflNewNear
                | InterMode::OpflNewNew
                | InterMode::OpflJointNew
        );
        if self.sequence.inter.adaptive_mvd && new_mode {
            let mode_context = match coded {
                InterMode::NewNear => 1,
                InterMode::OpflNearNew => 2,
                InterMode::OpflNewNear => 3,
                InterMode::JointNew => 5,
                InterMode::OpflJointNew => 6,
                InterMode::NewNew => 7,
                InterMode::OpflNewNew => 8,
                _ => 0,
            };
            inter.amvd = self.amvd(depth, neighbours, r0, mode_context)?;
        }
        let joint = matches!(inter.mode, InterMode::JointNew | InterMode::OpflJointNew);
        let mut joint_scale = 0;
        if joint {
            joint_scale = if inter.amvd {
                self.symbol(|m| &mut m.jmvd_amvd_scale)?
            } else {
                self.symbol(|m| &mut m.jmvd_scale)?
            };
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-jmvd_scale_mode[{joint_scale}]: r={range}"
            );
        }
        if coded == InterMode::NewNew
            && bw4.min(bh4) > 1
            && !self.frame.force_integer_mv
            && r0 != r1
            && self.frame.optical_flow != 2
            && self.frame.motion_modes.mask & (1 << MotionMode::WarpCausal as u8) != 0
            && self.neighbours_use(size, r0)
            && self.neighbours_use(size, r1)
        {
            let [x1, x2] = self.neighbour_motion_modes(size);
            let causal = MotionMode::WarpCausal as u8;
            let warp_context = usize::from(x1 >= causal || x2 >= causal)
                + usize::from(x1 == causal)
                + usize::from(x2 == causal);
            if self.bool(|m| &mut m.warp_causal[warp_context])? {
                inter.motion_mode = MotionMode::WarpCausal;
            }
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-comp_newmv_warp[{}]: r={range}", inter.motion_mode as u8
            );
        }
        if inter.mode != InterMode::GlobalGlobal {
            self.compound_drl(depth, inter, coded, context)?;
        }
        let precision = self.mv_precision(depth, size, inter, new_mode, precision_context)?;
        inter.precision = [precision; 2];
        if inter.mode != InterMode::GlobalGlobal {
            self.compound_residuals(depth, inter, precision, joint_scale)?;
        }
        let refine = self.refine_mv_flag(depth, size, inter, coded)?;
        inter.refine_mv = refine;
        inter.compound = CompoundType::Average;
        if coded <= InterMode::JointNew
            && refine != 1
            && !(coded == InterMode::JointNew && inter.amvd)
            && self.sequence.inter.masked_compound
            && bw4.min(bh4) >= 2
        {
            self.compound_type(depth, size, inter, coded, neighbours)?;
        }
        inter.weight = EQUAL_WEIGHT;
        if refine == 0
            && joint_scale == 0
            && self.sequence.inter.cwp
            && inter.compound == CompoundType::Average
            && matches!(coded, InterMode::NearNear | InterMode::JointNew)
        {
            self.compound_weight(depth, inter, coded)?;
        }
        Ok(inter.mode <= InterMode::JointNew
            && inter.refine_mv == 0
            && inter.motion_mode == MotionMode::Translation
            && (inter.mode != InterMode::GlobalGlobal || bw4.min(bh4) == 1))
    }

    /// A compound block's two references (dav2d `decode_b`'s compound
    /// reference loop): flags per rank, a pair of one reference twice in
    /// sequences that allow it.
    fn compound_references(&mut self, neighbours: Neighbours) -> Result<[Reference; 2]> {
        let count = self.references.count as i32;
        if count <= 1 {
            return Ok([0, 0]);
        }
        let same = i32::from(self.sequence.references.same_reference_compounds);
        let (counts, mut remaining) = self.reference_counts(neighbours);
        let mut references = [0; 2];
        let mut n = 0;
        let mut maybe_same = same != 0;
        let mut direction = false;
        let mut i: i32 = 0;
        while i < count + n - 2 + i32::from(maybe_same) {
            let current = counts[(i + 1) as usize];
            remaining -= current;
            let rank = i as usize;
            let bit = if n == 0 && (i == 2 || (i >= count - 2 && i + 1 >= same)) {
                true
            } else {
                let context = (current - remaining + 1).clamp(0, 2) as usize;
                if n == 0 {
                    self.bool(|m| &mut m.comp0_ref[context][rank])?
                } else {
                    let side = usize::from(direction ^ self.references.future[rank]);
                    self.bool(|m| &mut m.comp1_ref[context][side][rank])?
                }
            };
            if bit {
                references[n as usize] = i as Reference;
                n += 1;
                if n == 2 {
                    break;
                }
                direction = self.references.future[rank];
            }
            if maybe_same {
                maybe_same = !bit && i + 1 < same;
                if bit {
                    i -= 1;
                    remaining += current;
                }
            }
            i += 1;
        }
        if n < 2 {
            references[1] = (count - 1) as Reference;
            if n == 0 {
                references[0] = (count - 1 - i32::from(same < count)) as Reference;
            }
        }
        Ok(references)
    }

    /// The mode and DRL context of a block's references: how many of the
    /// above and left neighbours (near and far) share them, and whether
    /// any coded a residual (dav2d `get_snglref_ctx`, `get_compref_ctx`).
    fn reference_mode_context(
        &self,
        size: BlockSize,
        references: [Reference; 2],
        compound: bool,
    ) -> usize {
        use InterMode::{
            JointNew, NearNew, New, NewNear, NewNew, OpflJointNew, OpflNearNew, OpflNewNear,
            OpflNewNew,
        };
        let (bw4, bh4) = size.units();
        let b = self.bounds;
        let have_top = self.row > b.row_start;
        let have_left = self.column > b.column_start;
        let have_top_right = self.column + bw4 <= b.column_end;
        let have_bottom_left = self.row + bh4 <= b.row_end;
        let tip = self.frame.tip.references.map(|r| r as Reference);
        let in_modes = |mode: u8, modes: &[InterMode]| modes.iter().any(|&m| m as u8 == mode);
        let first_new = [
            New,
            NewNear,
            NewNew,
            JointNew,
            OpflNewNear,
            OpflNewNew,
            OpflJointNew,
        ];
        // The joint modes are missing here (AVM `compound_ref1_mode`).
        let second_new = [NearNew, NewNew, OpflNearNew, OpflNewNew];
        let any_new = [
            New,
            NearNew,
            NewNear,
            NewNew,
            JointNew,
            OpflNearNew,
            OpflNewNear,
            OpflNewNew,
            OpflJointNew,
        ];
        let mut new = false;
        let mut count = |edge: &context::Edge, index: usize| -> bool {
            let r = [
                edge.reference[0][index] as i8,
                edge.reference[1][index] as i8,
            ];
            let mode = edge.mode[index];
            if compound {
                if r[0] == TIP_FRAME && tip == references {
                    new |= mode == New as u8;
                    true
                } else if r == references {
                    new |= in_modes(mode, &any_new);
                    true
                } else {
                    false
                }
            } else if r[0] == references[0] {
                new |= in_modes(mode, &first_new);
                true
            } else if r[1] == references[0] {
                new |= in_modes(mode, &second_new);
                true
            } else {
                false
            }
        };
        let column = self.column as usize;
        let left = context::left_index(self.row);
        let mut above = false;
        if have_top {
            above |= count(&self.contexts.above, column);
            if have_top_right {
                above |= count(&self.contexts.above, column + bw4 as usize - 1);
            }
        }
        let mut beside = false;
        if have_left {
            beside |= count(&self.contexts.left, left);
            if have_bottom_left {
                beside |= count(&self.contexts.left, left + bh4 as usize - 1);
            }
        }
        usize::from(beside) + usize::from(above) + 2 * usize::from(new)
    }

    /// `amvd`: whether the residual is adaptive.
    fn amvd(
        &mut self,
        depth: usize,
        neighbours: Neighbours,
        reference: Reference,
        mode_context: usize,
    ) -> Result<bool> {
        let adaptive = |tile: &Self, n: usize| {
            n < neighbours.count && {
                let neighbour = neighbours.at[n].expect("a counted neighbour");
                tile.neighbour_motion(Some(neighbour))[0] == reference
                    && tile.neighbour_value(neighbour, |edge| &edge.amvd) != 0
            }
        };
        let context = usize::from(adaptive(self, 0)) + usize::from(adaptive(self, 1));
        let amvd = self.bool(|m| &mut m.amvd[mode_context][context])?;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-amvd[ctx={mode_context}|{context},{}]: r={range}",
            u8::from(amvd)
        );
        Ok(amvd)
    }

    /// A compound block's DRL indices: one per reference for `NEAR_NEAR`
    /// and `NEAR_NEW`, the second after the first for one reference twice.
    fn compound_drl(
        &mut self,
        depth: usize,
        inter: &mut InterRecord,
        coded: InterMode,
        mode_context: usize,
    ) -> Result<()> {
        let indices = 1 + usize::from(coded <= InterMode::NearNew);
        let max_drl_bits = self.frame.max_drl_bits;
        let (mut n, mut context) = (0, 0);
        for r in 0..indices {
            while n < max_drl_bits && self.bool(|m| &mut m.drl[context][mode_context])? {
                n += 1;
                context += usize::from(context < 2);
            }
            inter.drl[r] = n;
            n = if coded == InterMode::NearNear && inter.references[0] == inter.references[1] {
                inter.drl[0] + u8::from(inter.drl[0] < max_drl_bits)
            } else {
                0
            };
            context = usize::from(n.min(2));
        }
        if indices == 1 {
            inter.drl[1] = inter.drl[0];
        }
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-drl[{},{}]: r={range}", inter.drl[0], inter.drl[1]
        );
        Ok(())
    }

    /// The residual precision: the frame's, or a reduced one in sequences
    /// with flexible resolution.
    fn mv_precision(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &InterRecord,
        new_mode: bool,
        precision_context: &mut u8,
    ) -> Result<u8> {
        let mut precision = PRECISION_INTEGER + self.frame.mv_precision;
        if precision > PRECISION_INTEGER
            && !inter.amvd
            && self.sequence.inter.flexible_mv_resolution
            && new_mode
        {
            let [p1, p2] = self.spatial_neighbours(size).map(|neighbour| {
                neighbour.map_or(0, |n| self.neighbour_value(n, |edge| &edge.mv_precision))
            });
            let context = usize::from(p1 & 1) + usize::from(p2 & 1);
            let reduced = !self.bool(|m| &mut m.mv_precision_default[context])?;
            if reduced {
                let rest = usize::from((p1 | p2) >> 1);
                let index = self.symbol(|m| {
                    &mut m.mv_precision[rest][usize::from(precision - PRECISION_INTEGER - 1)]
                })?;
                precision = REDUCED_PRECISIONS[usize::from(precision == PRECISION_EIGHTH)][index];
                *precision_context = 2;
            }
            let range = self.reader.range();
            let (rest, frame) = if reduced {
                (
                    i32::from((p1 | p2) >> 1),
                    i32::from(self.frame.mv_precision) - 1,
                )
            } else {
                (-1, -1)
            };
            trace!(
                self,
                depth, "Post-mv_precision[ctx={context}|{rest}|{frame},{precision}]: r={range}"
            );
        }
        Ok(precision)
    }

    /// A compound block's residuals: each NEW reference's, signs coded or
    /// derived from their sum; a joint mode codes the nearer reference's
    /// and projects it to the other (dav2d `decode_b`).
    fn compound_residuals(
        &mut self,
        depth: usize,
        inter: &mut InterRecord,
        precision: u8,
        joint_scale: usize,
    ) -> Result<()> {
        let joint = matches!(inter.mode, InterMode::JointNew | InterMode::OpflJointNew);
        let [r0, r1] = inter.references.map(|r| r as usize);
        let refs = self.references;
        let (mut start, mut end) = (0, 2);
        let mut distances = [0_i32; 2];
        if joint {
            distances = [
                i32::from(refs.abs_distance[r0]),
                i32::from(refs.abs_distance[r1]),
            ];
            start = usize::from(distances[0] < distances[1]);
            if refs.future[r0] != refs.future[r1] {
                distances[1 - start] = -distances[1 - start];
            }
            end = start + 1;
        }
        let modes = inter.mode.per_reference();
        let (mut sum, mut nonzero) = (0, 0);
        for (n, &mode) in modes.iter().enumerate().take(end).skip(start) {
            if mode != InterMode::New {
                continue;
            }
            if inter.amvd {
                inter.residual[n] = self.amvd_residual()?;
            } else {
                let mv = self.mv_residual(precision)?;
                sum += mv.y + mv.x;
                nonzero += i32::from(mv.y != 0) + i32::from(mv.x != 0);
                inter.residual[n] = mv;
            }
        }
        if matches!(inter.mode, InterMode::NearNear | InterMode::OpflNearNear) {
            return Ok(());
        }
        let derive = self.sequence.inter.mvd_sign_derivation
            && inter.drl == [0, 0]
            && nonzero >= 3 * (end - start) as i32 - 2
            && !self.frame.screen_content
            && self.frame.mv_precision != FRAME_PRECISION_EIGHTH
            && precision < PRECISION_SIGN_DERIVE_MAX
            && matches!(
                inter.mode,
                InterMode::NewNew
                    | InterMode::OpflNewNew
                    | InterMode::JointNew
                    | InterMode::OpflJointNew
            )
            && inter.motion_mode == MotionMode::Translation;
        let derived_at = if derive { nonzero } else { -1 };
        let parity = (sum >> (PRECISION_EIGHTH - precision)) & 1 != 0;
        let mut seen = 0;
        for (n, &mode) in modes.iter().enumerate().take(end).skip(start) {
            if mode != InterMode::New {
                continue;
            }
            let mut mv = inter.residual[n];
            for component in [&mut mv.y, &mut mv.x] {
                if *component != 0 {
                    seen += 1;
                    let negative = if seen == derived_at {
                        parity
                    } else {
                        self.bypass()?
                    };
                    if negative {
                        *component = -*component;
                    }
                }
            }
            inter.residual[n] = mv;
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-mvdiff[{n},y:{},x:{}]: r={range}", mv.y, mv.x
            );
        }
        if joint {
            // The reference not coded takes the coded one's residual,
            // scaled by their distances, at eighth samples.
            let other = end & 1;
            let coded = 1 - other;
            inter.precision[other] = PRECISION_EIGHTH;
            inter.precision[coded] = precision;
            let mut mv = inter.residual[coded]
                .project(distances[other].clamp(-31, 31), distances[coded].min(31));
            joint_scale_residual(&mut mv, inter.amvd, joint_scale);
            inter.residual[other] = mv;
        }
        Ok(())
    }

    /// `refine_mv`: decoder-side MV refinement of a compound block whose
    /// references lie equally far on both sides (2: implied).
    fn refine_mv_flag(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &InterRecord,
        coded: InterMode,
    ) -> Result<u8> {
        let (bw4, bh4) = size.units();
        let refs = self.references;
        let [r0, r1] = inter.references.map(|r| r as usize);
        let switchable_new = refs.optical_flow == 1
            && matches!(
                inter.mode,
                InterMode::NearNew | InterMode::NewNear | InterMode::NewNew | InterMode::JointNew
            );
        if !refs.refine_mv
            || bw4.min(bh4) < 2
            || bw4 * bh4 <= 4
            || inter.mode == InterMode::GlobalGlobal
            || refs.distance[r0] != -refs.distance[r1]
            || refs.distance[r0] == 0
            || switchable_new
        {
            return Ok(0);
        }
        if matches!(
            coded,
            InterMode::NearNear | InterMode::OpflNearNear | InterMode::OpflJointNew
        ) {
            return Ok(2);
        }
        let context = coded as usize - InterMode::NearNear as usize;
        let refine = self.bool(|m| &mut m.refine_mv[context])?;
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-refinemv[ctx={},{}]: r={range}",
            context + 1,
            u8::from(refine)
        );
        Ok(u8::from(refine))
    }

    /// The compound blend: a wedge or difference mask, or the average.
    fn compound_type(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &mut InterRecord,
        coded: InterMode,
        neighbours: Neighbours,
    ) -> Result<()> {
        let (bw4, bh4) = size.units();
        let refs = self.references;
        let furthest = refs.furthest_future.map_or(-2, |rank| rank as Reference);
        let side_context = |tile: &Self, n: usize| -> usize {
            if n >= neighbours.count {
                return 0;
            }
            let neighbour = neighbours.at[n].expect("a counted neighbour");
            let motion = tile.neighbour_motion(Some(neighbour));
            if motion[1] == NO_REFERENCE {
                usize::from(motion[0] == furthest) * 2
            } else {
                usize::from(
                    tile.neighbour_value(neighbour, |edge| &edge.comp_type)
                        > CompoundType::Average as u8,
                )
            }
        };
        let (c0, c1) = (side_context(self, 0), side_context(self, 1));
        let [r0, r1] = inter.references.map(|r| r as usize);
        let context = c0
            + c1
            + usize::from(c0 != 0 && c1 != 0)
            + usize::from(refs.abs_distance[r0] == refs.abs_distance[r1]) * 6;
        let masked = self.bool(|m| &mut m.comp_type_masked[context])?;
        let mut value = -1;
        let mut wedge_sign = -1;
        if masked {
            inter.mode = coded;
            if bw4.max(bh4) <= WEDGE_MAX_UNITS && !self.bool(|m| &mut m.comp_type_weighted)? {
                inter.compound = CompoundType::Wedge;
                inter.wedge_index = self.wedge_index()?;
                inter.wedge_sign = self.bypass()?;
                value = i32::from(inter.wedge_index);
                wedge_sign = i32::from(inter.wedge_sign);
            } else {
                inter.compound = CompoundType::Difference;
                inter.mask_sign = self.bypass()?;
                value = i32::from(inter.mask_sign);
            }
        }
        let range = self.reader.range();
        let letter = ['?', 'w', 'm'][inter.compound as usize - 1];
        trace!(
            self,
            depth,
            "Post-comp_inter_type[ctx={context},{},{letter}={value}|{wedge_sign}]: r={range}",
            inter.compound as u8 - 1
        );
        Ok(())
    }

    /// A wedge index: quadrant, angle, distance (dav2d `read_wedge_idx`).
    fn wedge_index(&mut self) -> Result<i8> {
        let quadrant = self.symbol(|m| &mut m.wedge_quad)?;
        let angle =
            WEDGE_QUADRANT_ANGLES * quadrant + self.symbol(|m| &mut m.wedge_angle[quadrant])?;
        let distance = if !WEDGE_CENTRED_ANGLES.contains(&angle) || angle == WEDGE_UNCENTRED_ANGLE {
            1 + self.symbol(|m| &mut m.wedge_dist2)?
        } else {
            self.symbol(|m| &mut m.wedge_dist)?
        };
        Ok(WEDGE_INDICES[angle][distance])
    }

    /// The compound weight of a `NEAR_NEAR` or `JOINT_NEW` average.
    fn compound_weight(
        &mut self,
        depth: usize,
        inter: &mut InterRecord,
        coded: InterMode,
    ) -> Result<()> {
        let mut index = 0;
        while index < WEIGHT_FLAGS && self.bool(|m| &mut m.cwp[index])? {
            index += 1;
        }
        let [r0, r1] = inter.references.map(|r| r as usize);
        let one_side = usize::from(self.references.future[r0] == self.references.future[r1]);
        inter.weight = WEIGHTS[one_side][index];
        if coded != inter.mode && inter.weight != EQUAL_WEIGHT {
            // Optical flow in every block, which a weight turns off.
            inter.mode = coded;
            inter.refine_mv = 0;
        }
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-compweightpred_idx[{}]: r={range}", inter.weight
        );
        Ok(())
    }

    /// A single-reference block: its reference (or TIP), mode, adaptive
    /// MVD, block-adaptive weights, inter-intra or warp syntax, DRL index,
    /// precision and residual; returns whether it codes a filter.
    #[allow(clippy::too_many_lines, reason = "dav2d decode_b's single branch")]
    fn single_syntax(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        tip: bool,
        inter: &mut InterRecord,
        neighbours: Neighbours,
        precision_context: &mut u8,
    ) -> Result<bool> {
        let size = record.luma.unwrap_or(BlockSize::B4x4);
        let (bw4, bh4) = size.units();
        inter.compound = CompoundType::None;
        if tip {
            inter.references[0] = TIP_FRAME;
            inter.weight = TIP_WEIGHTS[usize::from(self.frame.tip.global_weight)];
        } else {
            let count = self.references.count;
            let mut rank = 0;
            if count > 1 {
                let (counts, mut remaining) = self.reference_counts(neighbours);
                loop {
                    let current = counts[rank + 1];
                    remaining -= current;
                    let context = (current - remaining + 1).clamp(0, 2) as usize;
                    if self.bool(|m| &mut m.single_ref[context][rank])? {
                        break;
                    }
                    rank += 1;
                    if rank >= count - 1 {
                        break;
                    }
                }
            }
            inter.references[0] = rank as Reference;
        }
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-ref[{},-1]: r={range}", inter.references[0]
        );
        let context = self.reference_mode_context(size, inter.references, false);
        if tip {
            let new = self.bool(|m| &mut m.tip_mode)?;
            inter.mode = if new { InterMode::New } else { InterMode::Near };
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-tip_mode[{}]: r={range}", inter.mode as u8
            );
        } else {
            let mut warp = false;
            if bw4.min(bh4) >= 2 && self.frame.warp_delta {
                let warp_context = self.warp_context(size, inter.references[0]);
                warp = self.bool(|m| &mut m.warp[warp_context])?;
            }
            inter.mode = if warp {
                if !self.frame.force_integer_mv && !self.bool(|m| &mut m.warp_newmv)? {
                    InterMode::WarpNew
                } else {
                    InterMode::Warp
                }
            } else {
                InterMode::Near
                    .offset(self.symbol(|m| &mut m.inter_mode[context])? as i32)
                    .ok_or(DecodeError::Malformed("AV2 inter mode"))?
            };
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-single_inter_mode[ctx={context},{}]: r={range}", inter.mode as u8
            );
        }
        if self.sequence.inter.adaptive_mvd && inter.mode == InterMode::New {
            inter.amvd = self.amvd(depth, neighbours, inter.references[0], 4)?;
        }
        if tip {
            // TIP blocks have no further motion syntax.
        } else if inter.mode <= InterMode::New {
            self.bawp_syntax(depth, record, inter)?;
            self.inter_intra(depth, size, inter)?;
        } else {
            self.warp_syntax(depth, size, inter)?;
        }
        if inter.mode != InterMode::Warp && inter.mode != InterMode::Global {
            let max_drl_bits = self.frame.max_drl_bits;
            let mut drl = 0;
            let mut drl_context = 0;
            loop {
                if drl >= max_drl_bits {
                    break;
                }
                let flag = if tip {
                    self.bool(|m| &mut m.tip_drl[drl_context])?
                } else {
                    self.bool(|m| &mut m.drl[drl_context][context])?
                };
                if !flag {
                    break;
                }
                drl += 1;
                drl_context += usize::from(drl_context < 2);
            }
            inter.drl[0] = drl;
            let range = self.reader.range();
            trace!(self, depth, "Post-drl[{drl},-1]: r={range}");
        }
        let new_mode = matches!(inter.mode, InterMode::New | InterMode::WarpNew);
        let precision = self.mv_precision(depth, size, inter, new_mode, precision_context)?;
        inter.precision = [precision; 2];
        if new_mode || (inter.mode == InterMode::Warp && inter.warp_with_residual) {
            self.single_residual(depth, inter, precision)?;
        }
        if inter.mode == InterMode::WarpNew
            && inter.motion_mode == MotionMode::WarpDelta
            && (inter.warp_reference == 0
                || (self.sequence.inter.motion_modes.six_parameter_warp_delta
                    && inter.warp_reference == 1))
        {
            self.warp_delta(depth, size, inter)?;
        } else if inter.motion_mode == MotionMode::WarpDelta {
            inter.warp_delta = [0; 4];
        }
        inter.warp_inter_intra = false;
        if inter.mode == InterMode::Warp && bw4.min(bh4) >= 2 && bw4.max(bh4) <= WEDGE_MAX_UNITS {
            let group = super::intra::size_group(size);
            inter.warp_inter_intra = self.bool(|m| &mut m.warp_inter_intra[group])?;
            if inter.warp_inter_intra {
                inter.inter_intra_mode = self.symbol(|m| &mut m.inter_intra_mode[group])? as u8;
                inter.wedge_index = if self.bool(|m| &mut m.inter_intra_wedge)? {
                    self.wedge_index()?
                } else {
                    -1
                };
            }
            let range = self.reader.range();
            let (mode, wedge) = if inter.warp_inter_intra {
                (
                    i32::from(inter.inter_intra_mode),
                    i32::from(inter.wedge_index),
                )
            } else {
                (-1, -1)
            };
            trace!(
                self,
                depth,
                "Post-warp_ii[{},{mode},{wedge}]: r={range}",
                u8::from(inter.warp_inter_intra)
            );
        }
        Ok(!tip
            && inter.mode <= InterMode::New
            && (inter.mode != InterMode::Global || bw4.min(bh4) == 1))
    }

    /// The warp flag's context: how many top and left neighbours predict
    /// from `reference` with a warp (dav2d `get_warp_ctx`).
    fn warp_context(&self, size: BlockSize, reference: Reference) -> usize {
        let (bw4, bh4) = size.units();
        let b = self.bounds;
        let have_top = self.row > b.row_start;
        let have_left = self.column > b.column_start;
        let at_superblock_top = self.row & (self.geometry.superblock_units - 1) == 0;
        let have_top_right = if at_superblock_top {
            ((self.column + bw4 - 2) & !1) < b.column_end
        } else {
            self.column + bw4 <= b.column_end
        };
        let have_bottom_left = self.row + bh4 <= b.row_end;
        let warped = |references: [u8; 2], mode: u8| {
            (references[0] as i8 == reference || references[1] as i8 == reference)
                && mode >= MotionMode::WarpCausal as u8
        };
        let edge_at = |edge: &context::Edge, index: usize| {
            usize::from(warped(
                [edge.reference[0][index], edge.reference[1][index]],
                edge.motion_mode[index],
            ))
        };
        let cache_at = |index: usize| {
            let cache = &self.above_cache;
            usize::from(warped(
                [cache.reference[0][index], cache.reference[1][index]],
                cache.motion_mode[index],
            ))
        };
        let column = self.column as usize;
        let offset = column & (context::LEFT_UNITS - 1);
        let left = context::left_index(self.row);
        let mut context = 0;
        if have_top {
            if at_superblock_top {
                context += cache_at(offset & !1);
                if have_top_right && bw4 >= 4 {
                    context += cache_at((offset + bw4 as usize - 2) & !1);
                }
            } else {
                context += edge_at(&self.contexts.above, column);
                if have_top_right {
                    context += edge_at(&self.contexts.above, column + bw4 as usize - 1);
                }
            }
        }
        if have_left {
            context += edge_at(&self.contexts.left, left);
            if have_bottom_left {
                context += edge_at(&self.contexts.left, left + bh4 as usize - 1);
            }
        }
        context
    }

    /// Block-adaptive weighted prediction of a NEAR or NEW block.
    fn bawp_syntax(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        inter: &mut InterRecord,
    ) -> Result<()> {
        let size = record.luma.unwrap_or(BlockSize::B4x4);
        let (bw4, bh4) = size.units();
        if !self.frame.bawp || inter.mode == InterMode::Global || bw4.min(bh4) < 2 {
            return Ok(());
        }
        let mut luma = u8::from(self.bool(|m| &mut m.bawp[0])?);
        let mut chroma = 0;
        if luma != 0 {
            let context = if inter.mode == InterMode::New {
                2 - usize::from(inter.amvd)
            } else {
                0
            };
            luma += u8::from(self.bool(|m| &mut m.bawp_explicit[context])?);
            if luma == BAWP_EXPLICIT {
                luma += u8::from(self.bool(|m| &mut m.bawp_explicit_scale)?);
                luma |= (context as u8) << BAWP_CONTEXT_SHIFT;
            }
            if record.chroma.is_some() {
                chroma = u8::from(self.bool(|m| &mut m.bawp[1])?);
            }
        }
        inter.bawp = [luma, chroma];
        let range = self.reader.range();
        trace!(self, depth, "Post-bawp[{},{chroma}]: r={range}", luma & 3);
        Ok(())
    }

    /// Inter-intra of a NEAR or NEW block without block-adaptive weights.
    fn inter_intra(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &mut InterRecord,
    ) -> Result<()> {
        let (bw4, bh4) = size.units();
        if self.frame.motion_modes.mask & (1 << MotionMode::InterIntra as u8) == 0
            || inter.bawp[0] != 0
            || bw4 * bh4 <= 2
            || bw4.max(bh4) > WEDGE_MAX_UNITS
            || !matches!(
                inter.mode,
                InterMode::Near | InterMode::Global | InterMode::New
            )
        {
            return Ok(());
        }
        let group = super::intra::size_group(size);
        if self.bool(|m| &mut m.inter_intra[group])? {
            inter.motion_mode = MotionMode::InterIntra;
            inter.inter_intra_mode = self.symbol(|m| &mut m.inter_intra_mode[group])? as u8;
            inter.wedge_index = -1;
            if bw4.min(bh4) > 1 && self.bool(|m| &mut m.inter_intra_wedge)? {
                inter.wedge_index = self.wedge_index()?;
            }
        }
        let range = self.reader.range();
        let on = inter.motion_mode == MotionMode::InterIntra;
        let (mode, wedge) = if on {
            (
                i32::from(inter.inter_intra_mode),
                i32::from(inter.wedge_index),
            )
        } else {
            (-1, -1)
        };
        trace!(
            self,
            depth, "Post-interintra[{},{mode},{wedge}]: r={range}", inter.motion_mode as u8
        );
        Ok(())
    }

    /// The warp syntax of a WARPMV or WARPNEWMV block: the warp kind, the
    /// model index, WARPMV's residual flag.
    fn warp_syntax(
        &mut self,
        depth: usize,
        size: BlockSize,
        inter: &mut InterRecord,
    ) -> Result<()> {
        inter.motion_mode = MotionMode::WarpDelta;
        let reference = inter.references[0];
        if inter.mode == InterMode::WarpNew && self.neighbours_use(size, reference) {
            let [x1, x2] = self.neighbour_motion_modes(size);
            let causal = MotionMode::WarpCausal as u8;
            let extend_context = usize::from(x1 >= causal) + usize::from(x2 >= causal);
            let modes = self.frame.motion_modes.mask;
            let causal_bit = 1 << causal;
            let delta_bit = 1 << MotionMode::WarpDelta as u8;
            if modes & (1 << MotionMode::WarpExtend as u8) != 0
                && self.bool(|m| &mut m.warp_extend[extend_context])?
            {
                inter.motion_mode = MotionMode::WarpExtend;
            } else if modes & (causal_bit | delta_bit) == causal_bit | delta_bit {
                let causal_context = usize::from(extend_context > 0)
                    + usize::from(x1 == causal)
                    + usize::from(x2 == causal);
                inter.motion_mode = if self.bool(|m| &mut m.warp_causal[causal_context])? {
                    MotionMode::WarpCausal
                } else {
                    MotionMode::WarpDelta
                };
            } else if modes & causal_bit != 0 {
                inter.motion_mode = MotionMode::WarpCausal;
            }
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-sngl_newmv_warp[{}]: r={range}", inter.motion_mode as u8
            );
        }
        if inter.motion_mode == MotionMode::WarpDelta {
            let mut index = 0;
            while index < WARP_REFERENCES {
                let more = match index {
                    0 => self.bool(|m| &mut m.warp_ref0)?,
                    1 => self.bool(|m| &mut m.warp_ref1)?,
                    _ => self.bool(|m| &mut m.warp_ref2)?,
                };
                if !more {
                    break;
                }
                index += 1;
            }
            inter.warp_reference = index;
            let range = self.reader.range();
            trace!(self, depth, "Post-warp_ref_idx[{index}/4]: r={range}");
        }
        if inter.mode == InterMode::Warp && inter.warp_reference < WARP_RESIDUAL_MAX_REFERENCE {
            inter.warp_with_residual = self.bool(|m| &mut m.warpmv_with_mvd)?;
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-warpmv_with_mvd[{}]: r={range}",
                u8::from(inter.warp_with_residual)
            );
        }
        Ok(())
    }

    /// A single reference's residual with coded or derived signs.
    fn single_residual(
        &mut self,
        depth: usize,
        inter: &mut InterRecord,
        precision: u8,
    ) -> Result<()> {
        let mut mv;
        let mut derived_at = -1;
        let mut parity = false;
        if inter.amvd {
            mv = self.amvd_residual()?;
        } else {
            mv = self.mv_residual(precision)?;
            let nonzero = i32::from(mv.x != 0) + i32::from(mv.y != 0);
            let derive = inter.mode != InterMode::Warp
                && nonzero != 0
                && self.sequence.inter.mvd_sign_derivation
                && inter.motion_mode == MotionMode::Translation
                && !self.frame.screen_content
                && self.frame.mv_precision != FRAME_PRECISION_EIGHTH
                && precision < PRECISION_SIGN_DERIVE_MAX;
            if derive {
                derived_at = nonzero;
                parity = ((mv.x + mv.y) >> (PRECISION_EIGHTH - precision)) & 1 != 0;
            }
        }
        let mut seen = 0;
        for component in [&mut mv.y, &mut mv.x] {
            if *component != 0 {
                seen += 1;
                let negative = if seen == derived_at {
                    parity
                } else {
                    self.bypass()?
                };
                if negative {
                    *component = -*component;
                }
            }
        }
        inter.residual[0] = mv;
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-mvdiff[0,y:{},x:{}]: r={range}", mv.y, mv.x
        );
        Ok(())
    }

    /// A warp delta's two or four parameters.
    fn warp_delta(&mut self, depth: usize, size: BlockSize, inter: &mut InterRecord) -> Result<()> {
        let coarse = self.bool(|m| &mut m.warp_delta_precision[size as usize])?;
        let six =
            self.sequence.inter.motion_modes.six_parameter_warp_delta && inter.warp_reference == 1;
        let count = if six { 4 } else { 2 };
        let step = WARP_DELTA_STEP >> u8::from(coarse);
        for n in 0..count {
            // The diagonal parameters (0, 3) and the others have models of
            // their own.
            let kind = usize::from(n == 1 || n == 2);
            let mut value = self.symbol(|m| &mut m.warp_delta[kind])?;
            if value == WARP_DELTA_ESCAPE && coarse {
                value += self.symbol(|m| &mut m.warp_delta_high[kind])?;
            }
            let mut value = value as i8;
            if value != 0 {
                if self.bool(|m| &mut m.warp_delta_sign)? {
                    value = -value;
                }
                value *= step;
            }
            inter.warp_delta[n] = value;
        }
        if count == 2 {
            inter.warp_delta[2] = WARP_DELTA_ABSENT;
        }
        let range = self.reader.range();
        let shift = u32::from(!coarse);
        let shown = |n: usize| {
            if n < count {
                i32::from(inter.warp_delta[n]) >> shift
            } else {
                0
            }
        };
        trace!(
            self,
            depth,
            "Post-warp_param_signal[{},{},{},{}]: r={range}",
            shown(0),
            shown(1),
            shown(2),
            shown(3)
        );
        Ok(())
    }

    /// The interpolation filter: SHARP for skip mode, TIP and refined
    /// blocks, coded per block in switchable frames when the block may
    /// use one, else the frame's.
    fn interp_filter(
        &mut self,
        depth: usize,
        (size, skip_mode): (BlockSize, bool),
        inter: &InterRecord,
        has_filter: bool,
    ) -> Result<u8> {
        if skip_mode
            || inter.references[0] == TIP_FRAME
            || inter.refine_mv != 0
            || inter.mode >= InterMode::OpflNearNear
        {
            return Ok(filter_mode::SHARP);
        }
        if self.frame.filter != filter_mode::SWITCHABLE {
            return Ok(self.frame.filter);
        }
        if !has_filter {
            return Ok(filter_mode::REGULAR);
        }
        let context = self.filter_context(size, inter.references);
        let filter = self.symbol(|m| &mut m.interp_filter[context])? as u8;
        let range = self.reader.range();
        trace!(
            self,
            depth, "Post-subpelfilter[ctx={context},{filter}]: r={range}"
        );
        Ok(filter)
    }

    /// The filter context: the spatial neighbours' filters where they
    /// predict from the block's first reference (dav2d `get_filter_ctx`).
    fn filter_context(&self, size: BlockSize, references: [Reference; 2]) -> usize {
        let reference = references[0];
        let compound = usize::from(references[1] != NO_REFERENCE) * COMPOUND_FILTER_CONTEXT;
        let [f0, f1] = self.spatial_neighbours(size).map(|neighbour| {
            neighbour.map_or(SWITCHABLE_FILTERS, |n| {
                let motion = self.neighbour_motion(Some(n));
                if motion[0] == reference || motion[1] == reference {
                    self.neighbour_value(n, |edge| &edge.filter)
                } else {
                    SWITCHABLE_FILTERS
                }
            })
        });
        compound
            + usize::from(if f0 == f1 || f1 == SWITCHABLE_FILTERS {
                f0
            } else if f0 == SWITCHABLE_FILTERS {
                f1
            } else {
                SWITCHABLE_FILTERS
            })
    }

    /// An adaptive residual: which components are nonzero, then each
    /// one's magnitude index (dav2d `read_amvd`; signs follow).
    fn amvd_residual(&mut self) -> Result<Mv> {
        /// Indices below this step by two from 2; the others are powers
        /// of two.
        const LINEAR_INDICES: usize = 3;
        let joint = self.symbol(|m| &mut m.amvd_joint)?;
        let mut mv = Mv::default();
        let magnitude = |index: usize| {
            if index < LINEAR_INDICES {
                2 + 2 * index as i32
            } else {
                1 << index
            }
        };
        if joint & 2 != 0 {
            mv.y = magnitude(self.symbol(|m| &mut m.amvd_index[0])?);
        }
        if joint & 1 != 0 {
            mv.x = magnitude(self.symbol(|m| &mut m.amvd_index[1])?);
        }
        Ok(mv)
    }

    /// A residual's magnitudes at `precision` (dav2d `read_mv_residual` on
    /// the inter models): the shell class, its offset, the component
    /// split; signs follow.
    fn mv_residual(&mut self, precision: u8) -> Result<Mv> {
        let symbols = SHELL_CLASS_BASE + u32::from(precision);
        let lower = symbols >> 1;
        let class = if self.bool(|m| &mut m.mv_shell_set)? {
            let mut class = lower + 1 + self.mv_class_upper(precision)? as u32;
            if u32::from(precision) + class == SHELL_LAST_PAIR_SUM {
                class += u32::from(self.bool(|m| &mut m.mv_shell_last)?);
            }
            class
        } else {
            self.mv_class_lower(precision)? as u32
        };
        self.shell_vector(class, u32::from(precision), &INTER_SHELL)
    }

    /// The lower-half shell class at `precision`.
    fn mv_class_lower(&mut self, precision: u8) -> Result<usize> {
        match precision {
            0 => self.symbol(|m| &mut m.mv_class_low0),
            1 => self.symbol(|m| &mut m.mv_class_low1),
            2 => self.symbol(|m| &mut m.mv_class_low2),
            3 => self.symbol(|m| &mut m.mv_class_low3),
            4 => self.symbol(|m| &mut m.mv_class_low4),
            5 => self.symbol(|m| &mut m.mv_class_low5),
            _ => self.symbol(|m| &mut m.mv_class_low6),
        }
    }

    /// The upper-half shell class at `precision`, from the half's start.
    fn mv_class_upper(&mut self, precision: u8) -> Result<usize> {
        match precision {
            0 => self.symbol(|m| &mut m.mv_class_high0),
            1 => self.symbol(|m| &mut m.mv_class_high1),
            2 => self.symbol(|m| &mut m.mv_class_high2),
            3 => self.symbol(|m| &mut m.mv_class_high3),
            4 => self.symbol(|m| &mut m.mv_class_high4),
            5 => self.symbol(|m| &mut m.mv_class_high5),
            _ => self.symbol(|m| &mut m.mv_class_high6),
        }
    }

    /// A residual's magnitudes from its shell class (dav2d
    /// `read_mv_residual` after the class): the offset in the class, then
    /// how the shell index splits between the components, at
    /// `precision`, in eighth samples.
    pub(super) fn shell_vector(
        &mut self,
        class: u32,
        precision: u32,
        models: &ShellModels,
    ) -> Result<Mv> {
        let offset = match class {
            0 | 1 => u32::from(self.bool(|m| (models.offset_low)(m, class as usize))?),
            CLASS_TWO => {
                let mut offset = u32::from(self.bool(|m| (models.offset_class2)(m))?);
                while offset != 0 && offset < CLASS_TWO_MAX_OFFSET && self.bypass()? {
                    offset += 1;
                }
                offset
            }
            _ => {
                let mut offset = 0;
                for bit in 0..class {
                    offset |=
                        u32::from(self.bool(|m| (models.offset_high)(m, bit as usize))?) << bit;
                }
                offset
            }
        };
        let index = if class == 0 {
            offset
        } else {
            offset + (1 << class)
        };
        if index == 0 {
            return Ok(Mv::default());
        }
        let mut pair = 0;
        if index >= 2 && self.bool(|m| (models.column_greater)(m, 0))? {
            pair = 1;
            if index >= PAIR_SECOND_MIN && self.bool(|m| (models.column_greater)(m, 1))? {
                pair = 2;
                if index >= PAIR_REMAINDER_MIN {
                    pair += self.uniform((index >> 1) - 1)?;
                }
            }
        }
        let shift = u32::from(PRECISION_EIGHTH) - precision;
        let component = |value: u32| (value << shift) as i32;
        if pair * 2 == index {
            let value = component(pair);
            return Ok(Mv { y: value, x: value });
        }
        let context = class.min(MAX_COLUMN_CONTEXT) as usize;
        let row_smaller = self.bool(|m| (models.column_index)(m, context))?;
        let (small, large) = (component(pair), component(index - pair));
        Ok(if row_smaller {
            Mv { y: small, x: large }
        } else {
            Mv { y: large, x: small }
        })
    }
}

/// Scale a joint mode's projected residual (dav2d `jmvd_scale`): adaptive
/// residuals double or halve both components, others one of them.
fn joint_scale_residual(mv: &mut Mv, amvd: bool, mode: usize) {
    match (amvd, mode) {
        (true, 1) => {
            mv.y *= 2;
            mv.x *= 2;
        }
        (true, 2) => {
            mv.y /= 2;
            mv.x /= 2;
        }
        (false, 1) => mv.y *= 2,
        (false, 2) => mv.x *= 2,
        (false, 3) => mv.y /= 2,
        (false, 4) => mv.x /= 2,
        _ => {}
    }
}
