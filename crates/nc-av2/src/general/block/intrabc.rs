//! An intra block copy block (dav2d `decode_b`'s `b->intrabc` branch,
//! `recon_b`'s vector resolution and the block-copy prediction of
//! `bytefn(dav2d_recon_b)`, `mc`, `bawp`; AVM `read_intrabc_info`): the
//! vector's syntax (predictor or residual, DRL index, precision, the
//! block-adaptive scale flag), the predictor list
//! ([`super::motion::MotionStore::block_copy_candidates`]), the bilinear
//! copy from the frame being decoded, then the residual as an inter
//! block's (inter transform types, no intra prediction).
//!
//! dav2d's clip of the vector to decoded areas is disabled (`#if 0`): the
//! vector is used as coded, and a source outside the frame repeats its
//! edge samples.
use super::Tile;
use super::context::NO_DIRECTION;
use super::intra::{IntraModes, mode};
use super::motion::{DEFAULT_LEFT_EXTRA, Motion, Mv};
use super::record::BlockRecord;
use super::size::BlockSize;
use super::transform::LumaTransforms;
use nc_common::decode::{DecodeError, Result};
use nc_dsp::av2_dec::{ipred, mc};

/// Eighth-sample vector units per sample, as a shift.
const VECTOR_FRACTION_BITS: u32 = 3;
/// Vector residual precisions (dav2d `mv_prec`): whole and quarter
/// samples.
const PRECISION_INTEGER: u32 = 3;
const PRECISION_QUARTER: u32 = 5;
/// The finest precision (eighth samples): residuals are scaled up to it.
const PRECISION_EIGHTH: u32 = 6;
/// Shell classes: the lower half holds `(9 + precision) / 2 + 1`. (The
/// flag that splits the eighth-sample precision's last two classes,
/// `mv_prec + class == 21`, is out of block copies' reach.)
const CLASS_BASE: u32 = 9;
/// Classes 0 and 1 code their offset in one flag, class 2 in a truncated
/// unary of up to three.
const CLASS_TWO: u32 = 2;
const CLASS_TWO_MAX_OFFSET: u32 = 3;
/// Column-index contexts stop at class 3.
const MAX_COLUMN_CONTEXT: u32 = 3;
/// Pair-index steps: the second flag needs a shell index of 4, the
/// uniform remainder of 6.
const PAIR_SECOND_MIN: u32 = 4;
const PAIR_REMAINDER_MIN: u32 = 6;
/// The block-adaptive scale (`morph_pred`): unscaled is 256 (8 fraction
/// bits); without edges the offset is -128.
const MORPH_UNIT: i32 = 256;
const MORPH_NO_EDGE_BETA: i32 = -128;
const MORPH_SHIFT: u32 = 8;
/// Edge samples of the block-adaptive scale's fit, as log2 counts, by
/// whether both edges exist, the height and width class (4, 8, 16 or
/// more samples) and edge (above, left) (dav2d `n_edge_samples`).
const MORPH_EDGE_SAMPLES: [[[[u32; 2]; 3]; 3]; 2] = [
    [
        [[2, 2], [3, 2], [4, 2]],
        [[2, 3], [3, 3], [4, 3]],
        [[2, 4], [3, 4], [4, 4]],
    ],
    [
        [[2, 2], [2, 2], [4, 0]],
        [[2, 2], [3, 3], [3, 3]],
        [[0, 4], [3, 3], [4, 4]],
    ],
];
/// The fit's size classes stop at 16 samples (log2 of 4x4 units).
const MORPH_MAX_CLASS: u32 = 2;
/// Bilinear phases are 1/16 samples; luma's eighth-sample vectors double.
const PHASE_BITS: u32 = 4;
/// The largest block copy is 64 samples a side; its window adds a column
/// and a row.
const WINDOW_SIDE: usize = 65;
/// Samples per 4x4 unit, as a shift.
const UNIT_LOG2: u32 = 2;

/// A block copy's syntax: as dav2d's `Av2Block` keeps it.
#[derive(Debug, Clone, Copy)]
struct BlockCopy {
    /// The predictor is used as is (`is_refmv`).
    predictor_only: bool,
    /// The predictor's index in the list.
    drl_index: usize,
    /// Whole-sample residual (and predictor).
    integer: bool,
    /// The residual (zero with `predictor_only`).
    residual: Mv,
    /// Scaled by a linear model of its edges (`morph_pred`).
    morph: bool,
}

impl Tile<'_, '_> {
    /// An intra block-copy block: its syntax, transform partition,
    /// vector, prediction, residual and contexts; its luma transform
    /// layout for the loop filters.
    pub(super) fn intrabc_block(
        &mut self,
        depth: usize,
        record: &BlockRecord,
    ) -> Result<LumaTransforms> {
        let size = record
            .luma
            .ok_or(DecodeError::Malformed("AV2 block copy without luma"))?;
        let copy = self.block_copy_syntax(depth, size)?;
        let transforms = self.transform_partition(depth, size, record, false)?;
        let mv = self.block_copy_vector(depth, record, size, &copy)?;
        self.predict_block_copy(record, mv, copy.morph)?;
        let modes = IntraModes {
            y_mode: mode::DC,
            uv_mode: mode::DC,
            direction: NO_DIRECTION,
            ..IntraModes::default()
        };
        self.reconstruct(depth, record, &modes, transforms)?;
        self.record_intra_contexts(record, &modes, u8::from(copy.morph));
        Ok(transforms)
    }

    /// The vector syntax (dav2d `decode_b`, `else if (b->intrabc)`).
    fn block_copy_syntax(&mut self, depth: usize, size: BlockSize) -> Result<BlockCopy> {
        let max_drl_bits = self
            .frame
            .intra_block_copy
            .map_or(0, |copy| usize::from(copy.max_drl_bits));
        let predictor_only = self.bool(|m| &mut m.intrabc_mode)?;
        let mut drl_index = 0;
        while drl_index < max_drl_bits && self.bypass()? {
            drl_index += 1;
        }
        let force_integer = self.frame.force_integer_mv;
        let integer = if predictor_only || force_integer {
            force_integer
        } else {
            !self.bool(|m| &mut m.intrabc_precision)?
        };
        let mut residual = Mv::default();
        if !predictor_only {
            residual = self.vector_residual(integer)?;
            if residual.y != 0 && self.bypass()? {
                residual.y = -residual.y;
            }
            if residual.x != 0 && self.bypass()? {
                residual.x = -residual.x;
            }
            let range = self.reader.range();
            trace!(
                self,
                depth, "Post-mvdiff[y:{},x:{}]: r={range}", residual.y, residual.x
            );
        }
        // Block-adaptive scaling of copies: key and intra-only frames of
        // sequences with block-adaptive weights, with screen content tools.
        let mut morph_context = None;
        let mut morph = false;
        if self.intra_frame() && self.sequence.inter.bawp && self.frame.screen_content {
            let context = self
                .spatial_neighbours(size)
                .iter()
                .flatten()
                .map(|&neighbour| {
                    usize::from(self.neighbour_value(neighbour, |edge| &edge.morph_pred))
                })
                .sum::<usize>();
            morph = self.bool(|m| &mut m.morph_pred[context])?;
            morph_context = Some(context);
        }
        let range = self.reader.range();
        trace!(
            self,
            depth,
            "Post-intrabc_info[mode={},drl={drl_index},prec={},morphctx={},morph={}]: r={range}",
            u8::from(predictor_only),
            u8::from(!integer),
            morph_context.map_or(-1, |context| context as i32),
            u8::from(morph)
        );
        Ok(BlockCopy {
            predictor_only,
            drl_index,
            integer,
            residual,
            morph,
        })
    }

    /// A vector residual's magnitudes in eighth samples (dav2d
    /// `read_mv_residual` on the block-copy models): a shell class (lower
    /// or upper half), the offset in the class, then how the shell index
    /// splits between the components.
    fn vector_residual(&mut self, integer: bool) -> Result<Mv> {
        let precision = if integer {
            PRECISION_INTEGER
        } else {
            PRECISION_QUARTER
        };
        let lower_classes = u32::midpoint(CLASS_BASE, precision);
        let class = if self.bool(|m| &mut m.dv_shell_set)? {
            let upper = if integer {
                self.symbol(|m| &mut m.dv_class_high_integer)?
            } else {
                self.symbol(|m| &mut m.dv_class_high_quarter)?
            } as u32;
            lower_classes + 1 + upper
        } else if integer {
            self.symbol(|m| &mut m.dv_class_low_integer)? as u32
        } else {
            self.symbol(|m| &mut m.dv_class_low_quarter)? as u32
        };
        let offset = match class {
            0 | 1 => u32::from(self.bool(|m| &mut m.dv_offset_low[class as usize])?),
            CLASS_TWO => {
                let mut offset = u32::from(self.bool(|m| &mut m.dv_offset_class2)?);
                while offset != 0 && offset < CLASS_TWO_MAX_OFFSET && self.bypass()? {
                    offset += 1;
                }
                offset
            }
            _ => {
                let mut offset = 0;
                for bit in 0..class {
                    offset |= u32::from(self.bool(|m| &mut m.dv_offset_high[bit as usize])?) << bit;
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
        if index >= 2 && self.bool(|m| &mut m.dv_column_greater[0])? {
            pair = 1;
            if index >= PAIR_SECOND_MIN && self.bool(|m| &mut m.dv_column_greater[1])? {
                pair = 2;
                if index >= PAIR_REMAINDER_MIN {
                    pair += self.uniform((index >> 1) - 1)?;
                }
            }
        }
        let shift = PRECISION_EIGHTH - precision;
        let component = |value: u32| (value << shift) as i16;
        if pair * 2 == index {
            let value = component(pair);
            return Ok(Mv { y: value, x: value });
        }
        let context = class.min(MAX_COLUMN_CONTEXT) as usize;
        let row_smaller = self.bool(|m| &mut m.dv_column_index[context])?;
        let (small, large) = (component(pair), component(index - pair));
        Ok(if row_smaller {
            Mv { y: small, x: large }
        } else {
            Mv { y: large, x: small }
        })
    }

    /// The block's vector: the listed predictor (or a default when it is
    /// zero) plus the residual; recorded for later blocks' lists (dav2d
    /// `recon_b`, `splat_intrabc_mv`).
    fn block_copy_vector(
        &mut self,
        depth: usize,
        record: &BlockRecord,
        size: BlockSize,
        copy: &BlockCopy,
    ) -> Result<Mv> {
        let max_drl_bits = self.frame.intra_block_copy.map_or(0, |c| c.max_drl_bits);
        let drl_reorder = self.sequence.references.drl_reorder;
        let store = self.motion.as_ref().ok_or(DecodeError::Malformed(
            "AV2 block copy without a motion store",
        ))?;
        let candidates = store.block_copy_candidates(
            (record.row, record.column),
            size,
            drl_reorder,
            max_drl_bits,
        );
        trace!(self, depth, "find_mv_refs(intra)");
        for (n, candidate) in candidates.list[..candidates.count].iter().enumerate() {
            trace!(
                self,
                depth + 1,
                "mv[{n}/{}]: y={},x={},w={}",
                candidates.count,
                candidate.mv.y,
                candidate.mv.x,
                candidate.weight
            );
        }
        let mut mv = candidates
            .list
            .get(copy.drl_index)
            .ok_or(DecodeError::Malformed("AV2 block copy DRL index"))?
            .mv;
        if mv == Mv::default() {
            // AVM forces a nonzero predictor: a superblock up, or left
            // (and 256 samples further) in a tile's first superblock row.
            let superblock = 1_i32 << self.frame.superblock_log2;
            let first_row = record.row < self.bounds.row_start + self.geometry.superblock_units;
            if first_row {
                mv.x = -((superblock + DEFAULT_LEFT_EXTRA) << VECTOR_FRACTION_BITS) as i16;
            } else {
                mv.y = -(superblock << VECTOR_FRACTION_BITS) as i16;
            }
        }
        if !copy.predictor_only {
            if copy.integer {
                mv = mv.to_integer();
            }
            mv.y = mv.y.wrapping_add(copy.residual.y);
            mv.x = mv.x.wrapping_add(copy.residual.x);
        }
        trace!(self, depth, "final 2dmv: y={},x={}", mv.y, mv.x);
        let bank = self.sequence.references.mv_bank;
        if let Some(store) = &mut self.motion {
            store.splat(record.row, record.column, size, Motion::BlockCopy(mv));
            if bank {
                store.add_to_bank(record.row, record.column, size, mv);
            }
        }
        Ok(mv)
    }

    /// Copy the block's luma (and, in a shared tree, chroma) from the frame
    /// being decoded, then scale luma when `morph` (dav2d `recon_b`'s
    /// `b->intrabc` luma path and its `!sdp_active && b->intrabc` chroma
    /// path).
    fn predict_block_copy(&mut self, record: &BlockRecord, mv: Mv, morph: bool) -> Result<()> {
        let size = record.luma.unwrap_or(BlockSize::B4x4);
        let (bw4, bh4) = size.units();
        self.copy_plane(0, (record.row, record.column), (bw4, bh4), mv)?;
        if morph {
            self.morph_block_copy(record, size, mv);
        }
        if let Some(chroma) = record.chroma {
            let (cbw4, cbh4) = chroma.units();
            for plane in 1..=2 {
                self.copy_plane(
                    plane,
                    (record.chroma_row, record.chroma_column),
                    (cbw4, cbh4),
                    mv,
                )?;
            }
        }
        Ok(())
    }

    /// The bilinear copy of a `bw4` x `bh4` luma-unit block of `plane` at
    /// luma unit (`row`, `column`) displaced by `mv` (dav2d `mc` with
    /// `DAV2D_FILTER_BILINEAR` on the current picture): the source window
    /// repeats the edge samples of the 8x8-rounded frame.
    fn copy_plane(
        &mut self,
        plane: usize,
        (row, column): (u32, u32),
        (bw4, bh4): (u32, u32),
        mv: Mv,
    ) -> Result<()> {
        let subsampled = u32::from(plane > 0);
        let (w, h) = (
            ((bw4 << UNIT_LOG2) >> subsampled) as usize,
            ((bh4 << UNIT_LOG2) >> subsampled) as usize,
        );
        if !w.is_power_of_two()
            || !h.is_power_of_two()
            || w < 2
            || h < 2
            || w >= WINDOW_SIDE
            || h >= WINDOW_SIDE
        {
            return Err(DecodeError::Malformed("AV2 block copy size"));
        }
        let shift = VECTOR_FRACTION_BITS + subsampled;
        let phase_mask = (1 << shift) - 1;
        let phase = |v: i16| (i32::from(v) & phase_mask) << (PHASE_BITS - shift);
        let origin = |units: u32, v: i16| {
            (((units << UNIT_LOG2) >> subsampled) as isize) + (isize::from(v) >> shift)
        };
        let (x, y) = (origin(column, mv.x), origin(row, mv.y));
        let visible = (
            ((self.geometry.columns << UNIT_LOG2) >> subsampled) as usize,
            ((self.geometry.rows << UNIT_LOG2) >> subsampled) as usize,
        );
        let stride = self.planes.stride[plane];
        let mut window = std::mem::take(&mut self.window);
        nc_dsp::emulate_edges(
            (&self.planes.data[plane], stride),
            visible,
            (x, y),
            (w + 1, h + 1),
            (&mut window, w + 1),
        );
        let destination = self
            .planes
            .offset(plane, row >> subsampled, column >> subsampled);
        mc::put_bilinear(
            &mut self.planes.data[plane],
            destination,
            stride,
            &window,
            0,
            w + 1,
            (w, h),
            (phase(mv.x), phase(mv.y)),
        );
        self.window = window;
        Ok(())
    }

    /// Scale a copied luma block by a least-squares fit of its source's
    /// edges to its own (dav2d `bawp` for a block copy: one block of at
    /// most 64x64, so never the cached scale of a larger one). Sources
    /// whose edges leave the tile stay unscaled.
    fn morph_block_copy(&mut self, record: &BlockRecord, size: BlockSize, mv: Mv) {
        let (bw4, bh4) = size.units();
        let (row, column) = (record.row, record.column);
        let b = self.bounds;
        let samples = |units: u32| (units << UNIT_LOG2) as i32;
        let whole = |v: i16| {
            let v = i32::from(v);
            (v + 3 + i32::from(v >= 0)) >> VECTOR_FRACTION_BITS
        };
        let (ref_x, ref_y) = (samples(column) + whole(mv.x), samples(row) + whole(mv.y));
        let w4 = bw4.min(self.geometry.columns - column);
        let h4 = bh4.min(self.geometry.rows - row);
        if ref_y + samples(h4) > samples(b.row_end)
            || ref_x + samples(w4) > samples(b.column_end)
            || ref_y - 1 < samples(b.row_start)
            || ref_x - 1 < samples(b.column_start)
        {
            return;
        }
        let have_left = column > b.column_start;
        let have_above = row > b.row_start;
        let (lw4, lh4) = (
            w4.ilog2().min(MORPH_MAX_CLASS),
            h4.ilog2().min(MORPH_MAX_CLASS),
        );
        let counts =
            MORPH_EDGE_SAMPLES[usize::from(have_above && have_left)][lh4 as usize][lw4 as usize];
        let above_log2 = if have_above { counts[0] } else { 0 };
        let left_log2 = if have_left { counts[1] } else { 0 };
        let count_log2 = above_log2
            + if above_log2 == left_log2 {
                u32::from(above_log2 != 0)
            } else {
                left_log2
            };
        let stride = self.planes.stride[0] as isize;
        let plane = &self.planes.data[0];
        let at = |y: i32, x: i32| i32::from(plane[(y as isize * stride + x as isize) as usize]);
        let (block_x, block_y) = (samples(column), samples(row));
        let (mut sum_x, mut sum_y, mut sum_xy, mut sum_xx) = (0, 0, 0, 0);
        let mut add = |source: i32, own: i32| {
            sum_x += source;
            sum_y += own;
            sum_xy += source * own;
            sum_xx += source * source;
        };
        if above_log2 != 0 {
            let width = 4 << lw4;
            let step = width >> above_log2;
            for i in (step >> 1..width).step_by(step as usize) {
                add(at(ref_y - 1, ref_x + i), at(block_y - 1, block_x + i));
            }
        }
        if left_log2 != 0 {
            let height = 4 << lh4;
            let step = height >> left_log2;
            for i in (step >> 1..height).step_by(step as usize) {
                add(at(ref_y + i, ref_x - 1), at(block_y + i, block_x - 1));
            }
        }
        let (alpha, beta) = if count_log2 == 0 {
            (MORPH_UNIT, MORPH_NO_EDGE_BETA)
        } else {
            let product = |a: i32, b: i32| ((i64::from(a) * i64::from(b)) >> count_log2) as i32;
            let num = sum_xy - product(sum_x, sum_y);
            let den = sum_xx - product(sum_x, sum_x);
            let alpha = ipred::derive_alpha(num, den, MORPH_UNIT);
            let diff = (sum_y << MORPH_SHIFT) - sum_x * alpha;
            let beta = (diff.abs() >> count_log2) * diff.signum();
            (alpha, beta)
        };
        let offset = self.planes.offset(0, row, column);
        let stride = self.planes.stride[0];
        mc::morph(
            &mut self.planes.data[0],
            offset,
            stride,
            (alpha, beta),
            ((bw4 << UNIT_LOG2) as usize, (bh4 << UNIT_LOG2) as usize),
        );
    }
}
