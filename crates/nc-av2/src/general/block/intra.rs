//! An intra block's mode syntax (dav2d `decode_b`, the `b->intra &&
//! !intrabc` branch; AVM `read_intra_luma_mode`, `read_intra_uv_mode`,
//! `read_cfl_alphas`, `read_mhccp_filter_dir`, `read_mrl_index`,
//! `read_fsc_mode`, `read_intra_dip_mode`, DPCM): the luma mode with its
//! angle, forward skip coding, the reference line, the chroma mode with
//! chroma from luma (explicit, implicit or multi-hypothesis), data-driven
//! intra prediction; then the transform partition, the reconstruction and
//! the contexts later blocks read.
use super::Tile;
use super::context::{self, NO_DIRECTION};
use super::motion::Motion;
use super::palette::Palette;
use super::record::BlockRecord;
use super::size::BlockSize;
use super::syntax::Neighbour;
use super::transform::{LumaTransforms, TxPartition};
use nc_common::decode::{DecodeError, Result};

/// dav2d `enum IntraPredMode` values of the modes the syntax names.
pub mod mode {
    /// `DC_PRED`.
    pub const DC: u8 = 0;
    /// `VERT_PRED`.
    pub const VERTICAL: u8 = 1;
    /// `HOR_PRED`.
    pub const HORIZONTAL: u8 = 2;
    /// `SMOOTH_PRED`.
    pub const SMOOTH: u8 = 9;
    /// `SMOOTH_V_PRED`.
    pub const SMOOTH_VERTICAL: u8 = 10;
    /// `SMOOTH_H_PRED`.
    pub const SMOOTH_HORIZONTAL: u8 = 11;
    /// `PAETH_PRED`.
    pub const PAETH: u8 = 12;
    /// `CFL_PRED` (chroma only).
    pub const CFL: u8 = 13;
}

/// How a chroma block is predicted from luma (dav2d `CFL_EXPLICIT`,
/// `CFL_IMPLICIT`, `CFL_MHCCP`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CflType {
    /// Signalled scales.
    #[default]
    Explicit = 0,
    /// Scales derived from the neighbours.
    Implicit = 1,
    /// The multi-hypothesis cross-component model.
    MultiHypothesis = 2,
}

/// The intra part of a block (dav2d `Av2Block`'s intra fields).
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools, reason = "the block's flags")]
pub struct IntraModes {
    /// Luma mode (dav2d `IntraPredMode`) and angle delta (-3..=3).
    pub y_mode: u8,
    pub y_angle: i8,
    /// Chroma mode and angle delta.
    pub uv_mode: u8,
    pub uv_angle: i8,
    /// DPCM per plane type (lossless only).
    pub dpcm: [bool; 2],
    /// Forward skip coding.
    pub fsc: bool,
    /// Reference line (0..=3) and whether two lines are blended.
    pub mrl_index: u8,
    pub multi_mrl: bool,
    /// Data-driven intra prediction: 0 off, else the transpose bit (4)
    /// over the mode plus one (dav2d `b->dip`).
    pub dip: u8,
    /// Chroma from luma: type, scales (U, V; explicit only) and the
    /// multi-hypothesis direction.
    pub cfl_type: CflType,
    pub cfl_alpha: [i8; 2],
    pub mhccp_direction: u8,
    /// The luma palette (size 0 without one).
    pub palette: Palette,
    /// The luma directional mode index, [`NO_DIRECTION`] if none.
    pub direction: u8,
    /// Whether the above / left luma and chroma neighbours use a smooth
    /// mode (dav2d `is_sm`: luma, chroma).
    pub smooth_above: [bool; 2],
    pub smooth_left: [bool; 2],
}

/// The modes coded without escape, in index order (dav2d
/// `reordered_nondir_y_mode`).
const NON_DIRECTIONAL: [u8; 5] = [
    mode::DC,
    mode::SMOOTH,
    mode::SMOOTH_VERTICAL,
    mode::SMOOTH_HORIZONTAL,
    mode::PAETH,
];
/// Each direction's nominal mode, by direction index (dav2d
/// `reordered_dir_y_mode`).
const DIRECTIONS: [u8; 8] = [3, 8, 1, 5, 4, 6, 2, 7];
/// Angle deltas per direction (-3..=3).
const DELTAS: u8 = 7;
/// Directional mode indices: eight directions of seven deltas.
const DIRECTIONAL_MODES: u8 = 56;
/// The directional mode index of plain vertical and horizontal (DPCM).
const VERTICAL_INDEX: u8 = 17;
const HORIZONTAL_INDEX: u8 = 45;
/// Mode indices of the default directional order (dav2d
/// `default_mode_list_y`): nominal directions, then deltas ±2, ±1, ±3.
const DEFAULT_DIRECTIONS: [u8; 56] = [
    17, 45, 3, 10, 24, 31, 38, 52, 15, 19, 43, 47, 1, 5, 8, 12, 22, 26, 29, 33, 36, 40, 50, 54, 16,
    18, 44, 46, 2, 4, 9, 11, 23, 25, 30, 32, 37, 39, 51, 53, 14, 20, 42, 48, 0, 6, 7, 13, 21, 27,
    28, 34, 35, 41, 49, 55,
];
/// Luma mode index symbols: set 0 codes indices up to the escape (7),
/// sets 1.. code sixteen each from `16 * set - 3`.
const Y_INDEX_ESCAPE: usize = 7;
const Y_SET_SIZE: usize = 16;
const Y_SET_START: usize = 3;
const Y_SET_BITS: u32 = 4;
/// Chroma mode index symbols past the escape (7) add three bits; at most
/// 13 indices.
const UV_INDEX_ESCAPE: usize = 7;
const UV_ESCAPE_BITS: u32 = 3;
const UV_INDICES: usize = 13;
/// The chroma default order after the non-directional modes (dav2d
/// `default_mode_list_uv`) and each direction's place in it.
const DEFAULT_UV_DIRECTIONS: [u8; 8] = [1, 2, 3, 4, 8, 5, 6, 7];
const DIRECTION_TO_UV_INDEX: [usize; 8] = [2, 4, 0, 5, 3, 6, 1, 7];
/// A chroma angle inherits the luma angle when the luma direction is at
/// most this many indices away from the DPCM direction.
const DPCM_ANGLE_REACH: u8 = 4;
/// Forward skip coding is read for blocks up to 32x32 samples, with a
/// size context per block size (dav2d `fsc_bsize_groups`); inter frames'
/// blocks outside intra regions take context 3.
const FSC_MAX_UNITS: u32 = 8;
const FSC_INTER_CONTEXT: usize = 3;
/// Chroma from luma: explicit signs come as one joint symbol (+1) of two
/// ternary signs; scale magnitudes 1..=8.
const CFL_SIGN_BASE: usize = 3;
const CFL_ALPHA_CONTEXTS_PER_SIGN: usize = 3;
/// Chroma blocks above 16 (lossless 1) chroma units (64 samples) have no
/// chroma from luma; multi-hypothesis up to 8 (lossless 1) and from two
/// units.
const CFL_MAX_UNITS: [u32; 2] = [16, 1];
const MHCCP_MAX_UNITS: [u32; 2] = [8, 1];
/// Above this side (in luma units) a block may use chroma from luma even
/// when its chroma tree split differently.
const SDP_CFL_MIN_UNITS: u32 = 16;
/// Palettes are read for DC blocks up to 64x64 with at least 4 units
/// across both sides.
const PALETTE_MAX_UNITS: u32 = 16;
const PALETTE_MIN_UNIT_SUM: u32 = 4;
/// Data-driven intra prediction: blocks of at least 8 units with both
/// sides at least two; its mode is a symbol plus one.
const DIP_MIN_SIDE: u32 = 2;
const DIP_MIN_UNITS: u32 = 8;
const DIP_TRANSPOSE_SHIFT: u32 = 4;
/// Side of the per-64x64 maps of luma directions and forward skip coding
/// chroma blocks of separate trees read (dav2d
/// `luma_intra_dir_mode_map`).
pub const LUMA_MAP_SIDE: usize = 16;

/// The size group of a block (dav2d `size_group_lookup`): 0..=3 by its
/// smaller and larger side logs.
#[must_use]
pub fn size_group(size: BlockSize) -> usize {
    let (w, h) = size.log2_units();
    match (w.min(h), w.max(h)) {
        (0, 0..=2) => 0,
        (0, 3) | (1, 1..=3) => 1,
        (0 | 1, 4) | (2, _) => 2,
        _ => 3,
    }
}

/// The forward skip coding size context (dav2d `fsc_bsize_groups`).
fn fsc_size_context(size: BlockSize) -> usize {
    let (w, h) = size.log2_units();
    match (w.min(h), w.max(h)) {
        (0, 0) => 0,
        (0, 1) => 1,
        (1, 1) => 2,
        (0 | 1, 2) => 3,
        (2, 2) | (0 | 1, 3) => 4,
        _ => 5,
    }
}

impl Tile<'_, '_> {
    /// An intra block: modes, transform partition, reconstruction, contexts.
    pub(super) fn intra_block(&mut self, depth: usize, record: &BlockRecord) -> Result<()> {
        let size = record
            .luma
            .or(record.chroma)
            .ok_or(DecodeError::Malformed("AV2 block without a tree"))?;
        let neighbours = self.spatial_neighbours(size);
        let lossless = self.lossless();
        let mut modes = IntraModes {
            direction: NO_DIRECTION,
            ..IntraModes::default()
        };
        if record.luma.is_some() {
            self.luma_modes(depth, size, neighbours, lossless, &mut modes)?;
        }
        if let Some(chroma) = record.chroma {
            self.chroma_modes(
                depth,
                size,
                (chroma, record.luma.is_some()),
                lossless,
                &mut modes,
            )?;
        }
        if record.luma.is_some() {
            self.palette_and_dip(depth, size, neighbours, &mut modes)?;
        }
        let transforms = if record.luma.is_some() {
            self.transform_partition(depth, size, record, modes.fsc)?
        } else {
            LumaTransforms::Partition(TxPartition::None)
        };
        if record.luma.is_some() {
            modes.smooth_above[0] = self.smooth_luma(Neighbour::Above(record.column as usize));
            modes.smooth_left[0] =
                self.smooth_luma(Neighbour::Left(context::left_index(record.row)));
        }
        if record.chroma.is_some() {
            let (above, left) = Self::chroma_unit_indices(record);
            modes.smooth_above[1] = is_smooth(self.contexts.above.uv_mode[above]);
            modes.smooth_left[1] = is_smooth(self.contexts.left.uv_mode[left]);
        }
        self.reconstruct(depth, record, &modes, transforms)?;
        self.record_intra_contexts(record, &modes, 0);
        if let (Some(size), Some(store)) = (record.luma, &mut self.motion) {
            // Later block copies' lists see no vector here (dav2d
            // `splat_intraref`).
            store.splat(record.row, record.column, size, Motion::Intra);
            if self.sequence.references.mv_bank {
                store.update_bank(record.row, record.column, size);
            }
        }
        Ok(())
    }

    /// Whether the frame (its one segment) is lossless.
    pub(super) fn lossless(&self) -> bool {
        self.frame.lossless.segments & 1 != 0
    }

    /// A neighbour's value of a field, or 0 when absent.
    fn spatial_value(&self, at: Option<Neighbour>, field: fn(&context::Edge) -> &Vec<u8>) -> usize {
        at.map_or(0, |neighbour| {
            usize::from(self.neighbour_value(neighbour, field))
        })
    }

    /// The luma syntax: DPCM or the mode with its angle, forward skip
    /// coding, the reference line.
    fn luma_modes(
        &mut self,
        depth: usize,
        size: BlockSize,
        neighbours: [Option<Neighbour>; 2],
        lossless: bool,
        modes: &mut IntraModes,
    ) -> Result<()> {
        modes.dpcm[0] = lossless && self.bool(|m| &mut m.dpcm_y)?;
        if modes.dpcm[0] {
            let horizontal = self.bool(|m| &mut m.dpcm_dir_y)?;
            (modes.y_mode, modes.direction) = if horizontal {
                (mode::HORIZONTAL, HORIZONTAL_INDEX)
            } else {
                (mode::VERTICAL, VERTICAL_INDEX)
            };
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-ydpcm[dir={}]: r={range}",
                u8::from(horizontal)
            );
        } else {
            self.luma_mode(depth, size, modes)?;
        }
        let (bw4, bh4) = size.units();
        if bw4.max(bh4) <= FSC_MAX_UNITS && self.sequence.transform.intra_identity {
            let size_context = fsc_size_context(size);
            let context = if !self.frame.is_intra() && !self.intra_region {
                FSC_INTER_CONTEXT
            } else {
                self.spatial_value(neighbours[0], |e| &e.fsc)
                    + self.spatial_value(neighbours[1], |e| &e.fsc)
            };
            modes.fsc = self.bool(|m| &mut m.fsc[context][size_context])?;
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-fsc[ctx={context}|{size_context},{}]: r={range}",
                u8::from(modes.fsc)
            );
        }
        if !modes.dpcm[0]
            && modes.direction != NO_DIRECTION
            && self.sequence.intra.multiple_reference_lines
        {
            let context = self.spatial_value(neighbours[0], |e| &e.mrl)
                + self.spatial_value(neighbours[1], |e| &e.mrl);
            modes.mrl_index = self.symbol(|m| &mut m.mrl_index[context])? as u8;
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-mrl_index[ctx={context},{}]: r={range}", modes.mrl_index
            );
            if modes.mrl_index > 0 {
                let context = self.spatial_value(neighbours[0], |e| &e.multi_mrl)
                    + self.spatial_value(neighbours[1], |e| &e.multi_mrl);
                modes.multi_mrl = self.bool(|m| &mut m.multi_mrl[context])?;
                let range = self.reader.range();
                trace!(
                    self,
                    depth,
                    "Post-multi_line_mrl[ctx={context},{}]: r={range}",
                    u8::from(modes.multi_mrl)
                );
            }
        }
        Ok(())
    }

    /// The luma mode: a set, an index in it, and the directional index
    /// list the neighbours reorder (dav2d `decode_b`, `intra_y_mode`).
    fn luma_mode(&mut self, depth: usize, size: BlockSize, modes: &mut IntraModes) -> Result<()> {
        let (bw4, bh4) = size.units();
        let w4 = bw4.min(self.geometry.columns - self.column);
        let h4 = bh4.min(self.geometry.rows - self.row);
        let above = (w4 == bw4)
            .then(|| self.contexts.above.direction[(self.column + bw4 - 1) as usize])
            .filter(|&direction| direction != NO_DIRECTION);
        let left = (h4 == bh4)
            .then(|| self.contexts.left.direction[context::left_index(self.row) + bh4 as usize - 1])
            .filter(|&direction| direction != NO_DIRECTION);
        let set = self.symbol(|m| &mut m.y_mode_set)?;
        let mut context = None;
        let index = if set == 0 {
            let ctx = usize::from(above.is_some()) + usize::from(left.is_some());
            context = Some(ctx);
            let mut index = self.symbol(|m| &mut m.y_mode_index[ctx])?;
            if index == Y_INDEX_ESCAPE {
                index += self.symbol(|m| &mut m.y_mode_offset[ctx])?;
            }
            index
        } else {
            set * Y_SET_SIZE - Y_SET_START + self.reader.literal(Y_SET_BITS)? as usize
        };
        if let Some(&nominal) = NON_DIRECTIONAL.get(index) {
            modes.y_mode = nominal;
            modes.y_angle = 0;
        } else {
            let direction =
                directional_list(index - NON_DIRECTIONAL.len(), bw4 * bh4, left, above)?;
            modes.direction = direction;
            modes.y_mode = DIRECTIONS[usize::from(direction / DELTAS)];
            modes.y_angle = (direction % DELTAS) as i8 - 3;
        }
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-intra_y_mode[set={set},idx={index},ctx={},mode={},angle={}]: r={range}",
            context.map_or(-1, |c| c as i32),
            modes.y_mode,
            modes.y_angle
        );
        Ok(())
    }

    /// The chroma syntax: DPCM, or chroma from luma and the chroma mode,
    /// with the chroma from luma parameters.
    fn chroma_modes(
        &mut self,
        depth: usize,
        size: BlockSize,
        (chroma, has_luma): (BlockSize, bool),
        lossless: bool,
        modes: &mut IntraModes,
    ) -> Result<()> {
        // A chroma tree's block takes the co-located luma direction.
        let luma_direction = |tile: &Self| {
            let map = (tile.row as usize & (LUMA_MAP_SIDE - 1)) * LUMA_MAP_SIDE
                + (tile.column as usize & (LUMA_MAP_SIDE - 1));
            tile.luma_direction_map[map]
        };
        modes.dpcm[1] = lossless && self.bool(|m| &mut m.dpcm_uv)?;
        if modes.dpcm[1] {
            let horizontal = self.bool(|m| &mut m.dpcm_dir_uv)?;
            let (uv_mode, index) = if horizontal {
                (mode::HORIZONTAL, HORIZONTAL_INDEX)
            } else {
                (mode::VERTICAL, VERTICAL_INDEX)
            };
            modes.uv_mode = uv_mode;
            if !has_luma {
                modes.direction = luma_direction(self);
            }
            let direction = modes.direction;
            modes.uv_angle = if direction.abs_diff(index) >= DPCM_ANGLE_REACH {
                0
            } else {
                (direction % DELTAS) as i8 - 3
            };
            let range = self.reader.range();
            trace!(
                self,
                depth,
                "Post-uvdpcm[dir={}]: r={range}",
                u8::from(horizontal)
            );
            return Ok(());
        }
        let ll = usize::from(lossless);
        let (cbw4, cbh4) = chroma.units();
        let (cbw4, cbh4) = ((cbw4 >> 1).max(1), (cbh4 >> 1).max(1));
        let (bw4, bh4) = size.units();
        let mhccp_allowed = self.sequence.intra.mhccp
            && cbw4.max(cbh4) <= MHCCP_MAX_UNITS[ll]
            && cbw4 * cbh4 >= 2 - ll as u32;
        let cfl_allowed = (self.sequence.intra.cfl || mhccp_allowed)
            && (bw4.max(bh4) > SDP_CFL_MIN_UNITS || !self.sdp_cfl_disallowed)
            && cbw4.max(cbh4) <= CFL_MAX_UNITS[ll];
        let (above, left) = Self::chroma_unit_indices_at(self.chroma_row, self.chroma_column);
        let mut cfl_context = None;
        let mut is_cfl = false;
        if cfl_allowed {
            let ctx = usize::from(self.contexts.above.uv_mode[above] == mode::CFL)
                + usize::from(self.contexts.left.uv_mode[left] == mode::CFL);
            cfl_context = Some(ctx);
            is_cfl = self.bool(|m| &mut m.cfl[ctx])?;
        }
        let mut uv_index = None;
        let mut uv_context = None;
        if is_cfl {
            modes.uv_mode = mode::CFL;
            modes.uv_angle = 0;
        } else {
            if !has_luma {
                modes.direction = luma_direction(self);
            }
            let direction = modes.direction;
            let ctx = usize::from(direction != NO_DIRECTION);
            uv_context = Some(ctx);
            let mut index = self.symbol(|m| &mut m.uv_mode[ctx])?;
            if index == UV_INDEX_ESCAPE {
                index += self.reader.literal(UV_ESCAPE_BITS)? as usize;
            }
            if index >= UV_INDICES {
                return Err(DecodeError::Malformed("AV2 chroma mode index"));
            }
            uv_index = Some(index);
            (modes.uv_mode, modes.uv_angle) = chroma_mode(index, ctx, direction);
        }
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-intra_uv_mode[cfl={},idx={},ctx={}|{},mode={},angle={}]: r={range}",
            u8::from(is_cfl),
            uv_index.map_or(-1, |i| i as i32),
            cfl_context.map_or(-1, |c| c as i32),
            uv_context.map_or(-1, |c| c as i32),
            modes.uv_mode,
            modes.uv_angle
        );
        if is_cfl {
            self.cfl_parameters(depth, size, mhccp_allowed, modes)?;
        }
        Ok(())
    }

    /// Chroma from luma's type and its scales or direction.
    fn cfl_parameters(
        &mut self,
        depth: usize,
        size: BlockSize,
        mhccp_allowed: bool,
        modes: &mut IntraModes,
    ) -> Result<()> {
        modes.cfl_alpha = [0; 2];
        if mhccp_allowed && (!self.sequence.intra.cfl || self.bool(|m| &mut m.mhccp)?) {
            let group = size_group(size);
            modes.cfl_type = CflType::MultiHypothesis;
            modes.mhccp_direction = self.symbol(|m| &mut m.mhccp_direction[group])? as u8;
        } else if self.bool(|m| &mut m.cfl_type)? {
            modes.cfl_type = CflType::Implicit;
        } else {
            modes.cfl_type = CflType::Explicit;
            let sign = self.symbol(|m| &mut m.cfl_sign)? + 1;
            let (sign_u, sign_v) = (sign / CFL_SIGN_BASE, sign % CFL_SIGN_BASE);
            for (plane, (own, other)) in
                [(sign_u, sign_v), (sign_v, sign_u)].into_iter().enumerate()
            {
                if own == 0 {
                    continue;
                }
                let context = usize::from(own == 2) * CFL_ALPHA_CONTEXTS_PER_SIGN + other;
                let magnitude = self.symbol(|m| &mut m.cfl_alpha[context])? as i8 + 1;
                modes.cfl_alpha[plane] = if own == 1 { -magnitude } else { magnitude };
            }
        }
        let range = self.reader.range();
        let multi = modes.cfl_type == CflType::MultiHypothesis;
        trace!(
            self,
            depth,
            "Post-cfl[type={},{}={}|{}]: r={range}",
            modes.cfl_type as u8,
            if multi { "mhdir" } else { "alpha" },
            if multi {
                i32::from(modes.mhccp_direction)
            } else {
                i32::from(modes.cfl_alpha[0])
            },
            modes.cfl_alpha[1]
        );
        Ok(())
    }

    /// The palette (its colours; the index map comes after data-driven
    /// intra prediction's syntax) and data-driven intra prediction of a DC
    /// luma block.
    fn palette_and_dip(
        &mut self,
        depth: usize,
        size: BlockSize,
        neighbours: [Option<Neighbour>; 2],
        modes: &mut IntraModes,
    ) -> Result<()> {
        let (bw4, bh4) = size.units();
        if self.frame.screen_content
            && modes.y_mode == mode::DC
            && bw4.max(bh4) <= PALETTE_MAX_UNITS
            && bw4 + bh4 >= PALETTE_MIN_UNIT_SUM
        {
            if self.bool(|m| &mut m.palette_y)? {
                modes.palette = self.palette_colors(depth)?;
            } else {
                let range = self.reader.range();
                trace!(self, depth, "Post-ypal[0]: r={range}");
            }
        }
        if modes.y_mode == mode::DC
            && modes.palette.size == 0
            && self.sequence.intra.dip
            && bw4.min(bh4) >= DIP_MIN_SIDE
            && bw4 * bh4 >= DIP_MIN_UNITS
        {
            let context = self.spatial_value(neighbours[0], |e| &e.dip)
                + self.spatial_value(neighbours[1], |e| &e.dip);
            if self.bool(|m| &mut m.dip[context])? {
                let transpose = u8::from(self.bypass()?);
                let index = self.symbol(|m| &mut m.dip_mode)? as u8;
                modes.dip = (transpose << DIP_TRANSPOSE_SHIFT) | (index + 1);
            }
            let range = self.reader.range();
            let dip = modes.dip;
            trace!(
                self,
                depth,
                "Post-dip[ctx={context},{},tp={},mode={}]: r={range}",
                u8::from(dip != 0),
                dip >> DIP_TRANSPOSE_SHIFT,
                (dip.saturating_sub(1)) & 7
            );
        }
        if modes.palette.size != 0 {
            self.palette_index_map(size, usize::from(modes.palette.size))?;
            let range = self.reader.range();
            trace!(self, depth, "Post-y-pal-indices: r={range}");
        }
        Ok(())
    }

    /// Whether a luma neighbour is an intra block of a smooth mode (dav2d
    /// `sm_flag`).
    fn smooth_luma(&self, at: Neighbour) -> bool {
        self.neighbour_value(at, |e| &e.intra) != 0
            && is_smooth(self.neighbour_value(at, |e| &e.mode))
    }

    /// The above and left chroma-unit indices of the current chroma block.
    fn chroma_unit_indices(record: &BlockRecord) -> (usize, usize) {
        Self::chroma_unit_indices_at(record.chroma_row, record.chroma_column)
    }

    /// The above and left chroma-unit indices at a luma position (4:2:0:
    /// half the luma units; the left edge per superblock row).
    pub(super) fn chroma_unit_indices_at(row: u32, column: u32) -> (usize, usize) {
        ((column >> 1) as usize, context::left_index(row) >> 1)
    }

    /// Record the intra (or block copy's) modes and its block-adaptive
    /// scale flag `morph_pred` in the contexts and the separate trees' luma
    /// maps (dav2d `decode_b` `set_ctx` of the intra and intrabc branches).
    pub(super) fn record_intra_contexts(
        &mut self,
        record: &BlockRecord,
        modes: &IntraModes,
        morph_pred: u8,
    ) {
        if let Some(size) = record.luma {
            let (bw4, bh4) = size.units();
            let column = record.column as usize;
            let row = context::left_index(record.row);
            let values = [
                u8::from(modes.fsc),
                modes.y_mode,
                modes.direction,
                u8::from(modes.mrl_index != 0),
                u8::from(modes.multi_mrl),
                u8::from(modes.dip != 0),
                modes.palette.size,
                morph_pred,
            ];
            for (edge, start, count) in [
                (&mut self.contexts.above, column, bw4),
                (&mut self.contexts.left, row, bh4),
            ] {
                let end = (start + count as usize).min(edge.mode.len());
                for (field, value) in [
                    (&mut edge.fsc, values[0]),
                    (&mut edge.mode, values[1]),
                    (&mut edge.direction, values[2]),
                    (&mut edge.mrl, values[3]),
                    (&mut edge.multi_mrl, values[4]),
                    (&mut edge.dip, values[5]),
                    (&mut edge.palette_size, values[6]),
                    (&mut edge.morph_pred, values[7]),
                ] {
                    field[start..end].fill(value);
                }
                if modes.palette.size != 0 {
                    edge.palette[start..end].fill(modes.palette.colors);
                }
            }
            if self.sequence.partition.separate_trees && record.chroma.is_none() {
                let side = LUMA_MAP_SIDE;
                let (row, column) = (
                    record.row as usize & (side - 1),
                    record.column as usize & (side - 1),
                );
                for y in row..(row + bh4 as usize).min(side) {
                    let line = y * side;
                    let end = (column + bw4 as usize).min(side);
                    self.luma_direction_map[line + column..line + end].fill(modes.direction);
                    self.luma_fsc_map[line + column..line + end].fill(u8::from(modes.fsc));
                }
            }
        }
        if let Some(chroma) = record.chroma {
            let (cbw4, cbh4) = chroma.units();
            let (above, left) = Self::chroma_unit_indices(record);
            let (w, h) = (((cbw4 >> 1).max(1)) as usize, ((cbh4 >> 1).max(1)) as usize);
            let above_end = (above + w).min(self.contexts.above.uv_mode.len());
            self.contexts.above.uv_mode[above..above_end].fill(modes.uv_mode);
            self.contexts.left.uv_mode[left..left + h].fill(modes.uv_mode);
        }
    }
}

/// Whether a mode is one of the smooth modes (dav2d `sm_flag`).
fn is_smooth(mode: u8) -> bool {
    matches!(
        mode,
        mode::SMOOTH | mode::SMOOTH_VERTICAL | mode::SMOOTH_HORIZONTAL
    )
}

/// The directional mode index of list position `index` (dav2d `decode_b`
/// `custom_mode_list_y`): the left and above directions first, for blocks
/// of more than four units their ±1..±4 neighbours, then the default
/// order. Blocks of at most two units use the default order.
fn directional_list(index: usize, units: u32, left: Option<u8>, above: Option<u8>) -> Result<u8> {
    let mut list = Vec::with_capacity(usize::from(DIRECTIONAL_MODES));
    let mut taken = 0_u64;
    if units > 2 {
        for direction in [left, above].into_iter().flatten() {
            if taken & (1 << direction) == 0 {
                list.push(direction);
                taken |= 1 << direction;
            }
        }
    }
    if list.is_empty() {
        return DEFAULT_DIRECTIONS
            .get(index)
            .copied()
            .ok_or(DecodeError::Malformed("AV2 luma mode index"));
    }
    let neighbours = list.len();
    if units > 4 && index >= neighbours {
        for distance in 1..=4_u8 {
            for n in 0..neighbours {
                for delta in [DIRECTIONAL_MODES - distance, distance] {
                    let direction = (list[n] + delta) % DIRECTIONAL_MODES;
                    if taken & (1 << direction) == 0 {
                        list.push(direction);
                        taken |= 1 << direction;
                    }
                }
            }
        }
    }
    if index >= list.len() {
        for &direction in &DEFAULT_DIRECTIONS {
            if taken & (1 << direction) == 0 {
                list.push(direction);
            }
        }
    }
    list.get(index)
        .copied()
        .ok_or(DecodeError::Malformed("AV2 luma mode index"))
}

/// The chroma mode and angle of index `index` with context `context` (1
/// when the luma mode is directional, which then comes first and lends
/// its angle).
fn chroma_mode(index: usize, context: usize, direction: u8) -> (u8, i8) {
    if index < context {
        return (
            DIRECTIONS[usize::from(direction / DELTAS)],
            (direction % DELTAS) as i8 - 3,
        );
    }
    let index = index - context;
    if let Some(&nominal) = NON_DIRECTIONAL.get(index) {
        return (nominal, 0);
    }
    let mut position = index - NON_DIRECTIONAL.len();
    if context == 1 && position >= DIRECTION_TO_UV_INDEX[usize::from(direction / DELTAS)] {
        position += 1;
    }
    (DEFAULT_UV_DIRECTIONS[position], 0)
}
