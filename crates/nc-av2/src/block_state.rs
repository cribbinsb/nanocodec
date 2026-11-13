//! Per-tile neighbour state of the block-based AV2 coder, shared by the
//! engine-path encoder (`block_coder`) and the decoder (`tile_decoder`).
//!
//! Holds what later blocks' contexts read: per 8×8 position whether it is
//! coded, inter, skipped and its intra luma mode (AVM `MB_MODE_INFO` fields
//! read through `xd->neighbors_line_buffer`), the motion-vector grid, and the
//! per-plane coefficient entropy contexts in 4×4 units (AVM
//! `ENTROPY_CONTEXT`). Contexts follow AVM v1.0.0 `av2_get_intra_inter_context`,
//! `av2_get_skip_txfm_context`, `get_y_mode_idx_ctx` and
//! `av2_get_pred_context_switchable_interp`; coefficient
//! contexts come from [`coeff::Context::derive`] (AVM `get_txb_ctx`).
//! It also collects the deblocking edge grid ([`EdgeGrid`]) of the tile and
//! answers CDEF's per-8×8 skip question.
use crate::coeff::{self, TxSize};
use crate::deblock::EdgeGrid;
use crate::interp_filter;
use crate::intra_modes::Mode;
use crate::mv_refs::{ListConfig, ReferenceGrid};
use crate::prediction::InterpFilter;
use nc_common::decode::{DecodeError, Result};

/// Mode-information granularity: every coded block is a multiple of 8×8.
pub(crate) const INFO_UNIT: usize = 8;
/// Superblock side: above neighbours of a superblock's top row are not
/// spatial neighbours (AVM `is_at_sb_top_boundary`).
const SUPERBLOCK: usize = crate::partition::SUPERBLOCK;
/// Coefficient entropy contexts are kept per 4×4 samples of each plane.
const CONTEXT_UNIT: usize = 4;
/// Most 4×4 context units one transform spans (TX64).
const MAX_TX_UNITS: usize = 16;
/// Context units outside the picture.
const OUTSIDE: [u8; MAX_TX_UNITS] = [0; MAX_TX_UNITS];

/// What a coded block leaves for its neighbours' contexts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct BlockInfo {
    pub is_inter: bool,
    pub skip: bool,
    /// Intra luma mode; inter blocks read as DC (AVM `av2_get_joint_mode`).
    pub luma_mode: Mode,
    /// An inter block's interpolation filter (AVM `interp_fltr`); `None`
    /// for intra blocks, whose reference never matches.
    pub filter: Option<InterpFilter>,
    /// Coded in skip mode (AVM `skip_mode`).
    pub skip_mode: bool,
}

/// Neighbour state of one tile.
#[derive(Debug)]
pub(crate) struct BlockState {
    width: usize,
    height: usize,
    info: Vec<Option<BlockInfo>>,
    pub motion: ReferenceGrid,
    /// Transform and skip information of every coded block, for deblocking.
    pub edges: EdgeGrid,
    coefficients: [CoeffPlane; 3],
}

#[derive(Debug)]
struct CoeffPlane {
    above: Vec<u8>,
    left: Vec<u8>,
}

impl BlockState {
    /// State for a `width` x `height` tile (multiples of eight) whose inter
    /// blocks use `lists`.
    /// # Errors
    /// Rejects unsupported dimensions.
    pub fn new(width: usize, height: usize, lists: ListConfig) -> Result<Self> {
        let coefficients = std::array::from_fn(|plane| {
            let (w, h) = if plane == 0 {
                (width, height)
            } else {
                (width / 2, height / 2)
            };
            CoeffPlane {
                above: vec![0; w / CONTEXT_UNIT],
                left: vec![0; h / CONTEXT_UNIT],
            }
        });
        Ok(Self {
            width,
            height,
            info: vec![None; width / INFO_UNIT * (height / INFO_UNIT)],
            motion: ReferenceGrid::new(width, height, lists)?,
            edges: EdgeGrid::new(width, height),
            coefficients,
        })
    }

    fn info_at(&self, x: isize, y: isize) -> Option<BlockInfo> {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return None;
        }
        self.info[y as usize / INFO_UNIT * (self.width / INFO_UNIT) + x as usize / INFO_UNIT]
    }

    /// The first two available of bottom-left, above-right, left and above
    /// (AVM `fetch_spatial_neighbors_with_line_buffer`); a neighbour may
    /// appear twice.
    fn line_neighbours(&self, x: usize, y: usize, size: usize) -> [Option<BlockInfo>; 2] {
        self.first_neighbours(x, y, size, true)
    }

    /// As [`Self::line_neighbours`], but above-right and above are not
    /// available at a superblock's top row (AVM `fetch_spatial_neighbors`).
    fn spatial_neighbours(&self, x: usize, y: usize, size: usize) -> [Option<BlockInfo>; 2] {
        self.first_neighbours(x, y, size, !y.is_multiple_of(SUPERBLOCK))
    }

    /// The first two available of bottom-left, above-right (if
    /// `above_allowed`), left and above (if `above_allowed`).
    fn first_neighbours(
        &self,
        x: usize,
        y: usize,
        size: usize,
        above_allowed: bool,
    ) -> [Option<BlockInfo>; 2] {
        let (x, y, last) = (x as isize, y as isize, size as isize - 1);
        let mut found = [None; 2];
        let mut count = 0;
        let candidates = [
            (x - 1, y + last, true),
            (x + last, y - 1, above_allowed),
            (x - 1, y, true),
            (x, y - 1, above_allowed),
        ];
        for (nx, ny, allowed) in candidates {
            if let Some(info) = self.info_at(nx, ny).filter(|_| allowed) {
                found[count] = Some(info);
                count += 1;
                if count == found.len() {
                    break;
                }
            }
        }
        found
    }

    /// `intra_inter` symbol context, in 0..4.
    pub fn intra_inter_context(&self, x: usize, y: usize, size: usize) -> usize {
        match self.line_neighbours(x, y, size) {
            [Some(a), Some(b)] => match (!a.is_inter, !b.is_inter) {
                (true, true) => 3,
                (false, false) => 0,
                _ => 1,
            },
            [Some(a), None] => 2 * usize::from(!a.is_inter),
            _ => 0,
        }
    }

    /// `switchable_interp` symbol context, in 0..4
    /// ([`interp_filter::context`]).
    pub fn filter_context(&self, x: usize, y: usize, size: usize) -> usize {
        interp_filter::context(
            self.spatial_neighbours(x, y, size)
                .map(|info| info.and_then(|i| i.filter)),
        )
    }

    /// `skip_txfm` symbol context, in 0..3.
    pub fn skip_context(&self, x: usize, y: usize, size: usize) -> usize {
        self.line_neighbours(x, y, size)
            .iter()
            .flatten()
            .map(|info| usize::from(info.skip))
            .sum()
    }

    /// `skip_mode` symbol context, in 0..3: the skip-mode blocks among the
    /// same neighbours (AVM `av2_get_skip_mode_context`).
    pub fn skip_mode_context(&self, x: usize, y: usize, size: usize) -> usize {
        self.line_neighbours(x, y, size)
            .iter()
            .flatten()
            .map(|info| usize::from(info.skip_mode))
            .sum()
    }

    /// Whether the coded 8×8 block at luma `(x, y)` skipped its luma
    /// transforms (AVM `skip_txfm[PLANE_TYPE_Y]`, read by `is_8x8_block_skip`
    /// for CDEF); intra blocks never do. Positions not yet coded read as not
    /// skipped.
    pub fn skips_transform(&self, x: usize, y: usize) -> bool {
        self.info_at(x as isize, y as isize)
            .is_some_and(|info| info.skip)
    }

    /// Whether the blocks above and left of luma `(x, y)` both skipped
    /// their transforms; `None` unless both are coded (the pre-transform
    /// skip's neighbour signal, SVT `lpd1_should_perform_tx`).
    pub fn above_and_left_skip(&self, x: usize, y: usize) -> Option<bool> {
        let above = self.info_at(x as isize, y as isize - 1)?;
        let left = self.info_at(x as isize - 1, y as isize)?;
        Some(above.skip && left.skip)
    }

    /// Whether the block covering luma `(x, y)` is already coded (AVM
    /// `is_mi_coded`, here kept for the whole tile).
    pub fn is_coded(&self, x: usize, y: usize) -> bool {
        self.info_at(x as isize, y as isize).is_some()
    }

    /// Luma modes of the above-right and bottom-left neighbours, for the
    /// intra mode context and list.
    pub fn mode_neighbours(&self, x: usize, y: usize, size: usize) -> (Option<Mode>, Option<Mode>) {
        let last = size as isize - 1;
        let mode = |info: Option<BlockInfo>| info.map(|i| i.luma_mode);
        (
            mode(self.info_at(x as isize + last, y as isize - 1)),
            mode(self.info_at(x as isize - 1, y as isize + last)),
        )
    }

    /// Record a coded block covering `size` x `size` samples at `(x, y)`.
    pub fn record(&mut self, x: usize, y: usize, size: usize, info: BlockInfo) {
        let stride = self.width / INFO_UNIT;
        for row in (y..(y + size).min(self.height)).step_by(INFO_UNIT) {
            for col in (x..(x + size).min(self.width)).step_by(INFO_UNIT) {
                self.info[row / INFO_UNIT * stride + col / INFO_UNIT] = Some(info);
            }
        }
    }

    /// Coefficient context of the `tx` transform of `plane` at plane position
    /// `(x, y)`, in a block whose plane side is `1 << plane_block_log2`
    /// (AVM `get_txb_ctx`). `u_coded` (V only): whether the block's most
    /// recent U transform had coefficients.
    /// # Errors
    /// Rejects geometry the context derivation does not allow.
    pub fn coeff_context(
        &self,
        plane: usize,
        (x, y): (usize, usize),
        tx: TxSize,
        plane_block_log2: u32,
        is_inter: bool,
        u_coded: bool,
    ) -> Result<coeff::Context> {
        let contexts = &self.coefficients[plane];
        let units = tx.units();
        let (ux, uy) = (x / CONTEXT_UNIT, y / CONTEXT_UNIT);
        // Contexts start at zero at the picture's top and left edges.
        let above = if y == 0 {
            &OUTSIDE[..]
        } else {
            &contexts.above[ux..ux + units]
        };
        let left = if x == 0 {
            &OUTSIDE[..]
        } else {
            &contexts.left[uy..uy + units]
        };
        coeff::Context::derive(
            plane as u8,
            plane_block_log2,
            tx,
            is_inter,
            above,
            left,
            u_coded,
        )
    }

    /// Store the entropy context of a coded (or skipped: all zero) transform.
    pub fn update_coeff_context(
        &mut self,
        plane: usize,
        (x, y): (usize, usize),
        tx: TxSize,
        coefficients: &[i32],
    ) {
        self.set_coeff_context(plane, (x, y), tx, coeff::entropy_context(coefficients));
    }

    /// Set the entropy contexts of the transform of `tx` at `(x, y)` in
    /// `plane` to `value` ([`coeff::entropy_context`] of its levels).
    pub fn set_coeff_context(
        &mut self,
        plane: usize,
        (x, y): (usize, usize),
        tx: TxSize,
        value: u8,
    ) {
        let units = tx.units();
        let contexts = &mut self.coefficients[plane];
        contexts.above[x / CONTEXT_UNIT..x / CONTEXT_UNIT + units].fill(value);
        contexts.left[y / CONTEXT_UNIT..y / CONTEXT_UNIT + units].fill(value);
    }

    /// Zero the entropy contexts of every plane over the block of side
    /// `size` at `(x, y)`, as a block without residual leaves them.
    pub fn clear_coeff_contexts(&mut self, x: usize, y: usize, size: usize) {
        for (plane, contexts) in self.coefficients.iter_mut().enumerate() {
            let (px, py, side) = plane_area(plane, x, y, size);
            contexts.above[px..px + side].fill(0);
            contexts.left[py..py + side].fill(0);
        }
    }

    /// Snapshot of the coefficient contexts a block at `(x, y)` of `size`
    /// may change, for rate trials.
    pub fn save_coeff_contexts(&self, x: usize, y: usize, size: usize) -> SavedContexts {
        SavedContexts {
            planes: std::array::from_fn(|plane| {
                let (px, py, side) = plane_area(plane, x, y, size);
                let contexts = &self.coefficients[plane];
                let mut above = [0; MAX_SAVED_CONTEXTS];
                let mut left = [0; MAX_SAVED_CONTEXTS];
                above[..side].copy_from_slice(&contexts.above[px..px + side]);
                left[..side].copy_from_slice(&contexts.left[py..py + side]);
                (above, left)
            }),
            x,
            y,
            size,
        }
    }

    /// Restore a snapshot taken by [`BlockState::save_coeff_contexts`].
    pub fn restore_coeff_contexts(&mut self, saved: &SavedContexts) {
        for (plane, (above, left)) in saved.planes.iter().enumerate() {
            let (px, py, side) = plane_area(plane, saved.x, saved.y, saved.size);
            let contexts = &mut self.coefficients[plane];
            contexts.above[px..px + side].copy_from_slice(&above[..side]);
            contexts.left[py..py + side].copy_from_slice(&left[..side]);
        }
    }

    /// Fail unless every block left of and above `(x, y)` is coded.
    /// # Errors
    /// Reports a coding-order violation.
    pub fn check_order(&self, x: usize, y: usize) -> Result<()> {
        let missing = |nx: isize, ny: isize| nx >= 0 && ny >= 0 && self.info_at(nx, ny).is_none();
        if missing(x as isize - 1, y as isize) || missing(x as isize, y as isize - 1) {
            return Err(DecodeError::Malformed("AV2 block coding order"));
        }
        Ok(())
    }
}

/// Context units along one side of the largest (64×64 luma) block.
const MAX_SAVED_CONTEXTS: usize = 64 / CONTEXT_UNIT;

/// Coefficient contexts saved around one block, in fixed arrays: saving
/// is per inter trial, and six small allocations each time were a visible
/// share of the allocator and `memcpy` time on the Jetson.
#[derive(Debug, Clone)]
pub(crate) struct SavedContexts {
    planes: [([u8; MAX_SAVED_CONTEXTS], [u8; MAX_SAVED_CONTEXTS]); 3],
    x: usize,
    y: usize,
    size: usize,
}

/// Context-unit range of `plane` covered by a luma block.
fn plane_area(plane: usize, x: usize, y: usize, size: usize) -> (usize, usize, usize) {
    let shift = usize::from(plane != 0);
    (
        (x >> shift) / CONTEXT_UNIT,
        (y >> shift) / CONTEXT_UNIT,
        (size >> shift) / CONTEXT_UNIT,
    )
}
