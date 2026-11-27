//! The pixel side of intra reconstruction: edge preparation and
//! prediction, chroma from luma, the secondary, cross-chroma and inverse
//! transforms, on the frame being reconstructed.
use super::Tile;
use super::coefficients::{Residual, tx_type};
use super::intra::CflType;
use super::intra::{IntraModes, mode};
use super::record::BlockRecord;
use super::size::BlockSize;
use super::transform::{TxPartition, TxSize};
use crate::general::sample::Av2Sample;
use nc_common::Sample;
use nc_dsp::av2_dec::ipred::{
    self, ANGLE_DIP_FLAG, ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, ANGLE_IS_LUMA,
    ANGLE_MRL_IDX_SHIFT, ANGLE_MULTI_MRL_FLAG, ANGLE_SEQHDR_IBP_FLAG, ANGLE_SMOOTH_LEFT_EDGE_FLAG,
    ANGLE_SMOOTH_TOP_EDGE_FLAG, ANGLE_USE_EDGE_FILTER_FLAG, CFL_ALPHA_U_MASK, CFL_ALPHA_U_SHIFT,
    CFL_ALPHA_V_MASK, CFL_ALPHA_V_SHIFT, CFL_HAS_LEFT, CFL_HAS_TOP, CFL_IS_TOP_SB_EDGE,
    CFL_MH_MAX_EDGE_SAMPLES, CFL_MH_MAX_LUMA_SIZE, CFL_MH_TOP_STRIDE, CflChroma, CflMhDir,
    EdgeRequest, IntraPredMode, SampleRef,
};
use nc_dsp::av2_dec::itx::{self, Dpcm, RectTxSize, TxType};
use nc_dsp::av2_dec::stx;

/// The edge buffer's corner (dav2d `scratch.edge + 128`), nine further on
/// with reference lines beyond the first.
const EDGE_CORNER: usize = 128;
const EDGE_MRL_EXTRA: usize = 9;
/// Above-right availability counts at most 16 units (a ctz guard bit).
const COUNT_GUARD: u32 = 0x1_0000;
/// Chroma transforms of 64 samples have no above-right or below-left edge.
const CHROMA_EDGE_MAX_SAMPLES: usize = 64;
/// Superblock masks cover 64 units.
const MASK_UNITS: u32 = 64;
/// Unit positions in the superblock's coded masks.
const MASK_POSITION: i32 = 63;

/// Edge availability of one transform: above-right and below-left units.
#[derive(Debug, Clone, Copy)]
struct EdgeCounts {
    top_right: i32,
    bottom_left: i32,
}

/// Luma samples per 4x4 unit, as a shift.
const UNIT: u32 = 2;
/// Chroma coefficients are indexed in steps of 16 per chroma unit.
const COEFFICIENTS_PER_UNIT: usize = 16;
/// The modes whose secondary transform is not transposed (dav2d
/// `recon_b_luma_tx`'s `mask`): horizontal-leaning ones.
const UNTRANSPOSED_MODES: [u8; 4] = [2, 6, 8, mode::SMOOTH_HORIZONTAL];

/// The frame being reconstructed: three planes padded to whole
/// superblocks, so transforms that overhang the picture stay inside.
#[derive(Debug, Clone)]
pub struct Planes<T> {
    /// Y, U and V samples, row-major.
    pub data: [Vec<T>; 3],
    /// Each plane's row stride.
    pub stride: [usize; 3],
}

impl<T: Sample> Planes<T> {
    /// Planes for a frame `columns` x `rows` 4x4 units, rounded up to
    /// `superblock` units.
    #[must_use]
    pub fn new(columns: u32, rows: u32, superblock: u32) -> Self {
        let width = (columns.next_multiple_of(superblock) << UNIT) as usize;
        let height = (rows.next_multiple_of(superblock) << UNIT) as usize;
        let plane = |w: usize, h: usize| vec![T::default(); w * h];
        Self {
            data: [
                plane(width, height),
                plane(width / 2, height / 2),
                plane(width / 2, height / 2),
            ],
            stride: [width, width / 2, width / 2],
        }
    }

    /// The offset of plane `plane`'s sample at 4x4 unit (`row`, `column`)
    /// of that plane.
    #[must_use]
    pub fn offset(&self, plane: usize, row: u32, column: u32) -> usize {
        ((row << UNIT) as usize) * self.stride[plane] + ((column << UNIT) as usize)
    }
}

/// `value` when `on`.
fn flag(on: bool, value: i32) -> i32 {
    if on { value } else { 0 }
}

/// See [`flag`].
fn flag32(on: bool, value: u32) -> u32 {
    if on { value } else { 0 }
}

/// The smooth-edge flags: per edge with intra bi-prediction, else both
/// when either neighbour is smooth (dav2d `is_sm_flag`).
fn smooth_flags(ibp: bool, above: bool, left: bool) -> i32 {
    if ibp {
        flag(above, ANGLE_SMOOTH_TOP_EDGE_FLAG) | flag(left, ANGLE_SMOOTH_LEFT_EDGE_FLAG)
    } else {
        flag(
            above || left,
            ANGLE_SMOOTH_TOP_EDGE_FLAG | ANGLE_SMOOTH_LEFT_EDGE_FLAG,
        )
    }
}

/// The kernels' size of a transform.
fn rect(tx: TxSize) -> RectTxSize {
    RectTxSize::from_raw(tx.dav2d_index() as u8).unwrap_or(RectTxSize::Tx4x4)
}

/// The DPCM direction of a lossless DPCM block predicted by `mode`.
fn dpcm(mode: u8) -> Dpcm {
    if mode == mode::VERTICAL {
        Dpcm::Vertical
    } else {
        Dpcm::Horizontal
    }
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// Predict one luma transform (dav2d `recon_b_luma_tx`, "b->intra").
    pub(super) fn predict_luma(
        &mut self,
        record: &BlockRecord,
        modes: &IntraModes,
        tx: TxSize,
        (mode, angle): (u8, i32),
        partition: TxPartition,
    ) {
        let (tw4, th4) = tx.units();
        let (bx, by) = (self.column as i32, self.row as i32);
        let five = matches!(partition, TxPartition::Horizontal5 | TxPartition::Vertical5);
        let is_hv5 = five && (self.row > record.row || self.column > record.column);
        let counts = if is_hv5 {
            EdgeCounts {
                top_right: 0,
                bottom_left: 0,
            }
        } else {
            let mask = (bx & MASK_POSITION, by & MASK_POSITION);
            self.edge_counts(0, (bx, by), (tw4 as i32, th4 as i32), (bx, by), mask)
        };
        let sequence = &self.sequence.intra;
        let mrl = modes.mrl_index;
        let multi = modes.multi_mrl && tx != TxSize::TX4;
        let mut ibp = sequence.ibp && tx != TxSize::TX4 && mrl == 0;
        let smooth = smooth_flags(ibp, modes.smooth_above[0], modes.smooth_left[0]);
        if modes.y_angle & 1 != 0 {
            ibp = false;
        }
        let dip = modes.dip != 0;
        let flags = ANGLE_IS_LUMA
            | smooth
            | flag(sequence.edge_filter, ANGLE_USE_EDGE_FILTER_FLAG)
            | flag(ibp, ANGLE_IBP_FLAG)
            | flag(sequence.ibp, ANGLE_SEQHDR_IBP_FLAG)
            | (i32::from(mrl) << ANGLE_MRL_IDX_SHIFT)
            | flag(multi, ANGLE_MULTI_MRL_FLAG)
            | flag(self.column > self.bounds.column_start, ANGLE_HAS_LEFT_FLAG)
            | flag(self.row > self.bounds.row_start, ANGLE_HAS_TOP_FLAG)
            | flag(dip, ANGLE_DIP_FLAG);
        let angle = if dip { i32::from(modes.dip - 1) } else { angle };
        let topleft = EDGE_CORNER + if mrl != 0 { EDGE_MRL_EXTRA } else { 0 };
        let top_sb_edge = self.row & (self.geometry.superblock_units - 1) == 0;
        let request = EdgeRequest {
            x: bx,
            y: by,
            end_x: self.bounds.column_end as i32,
            end_y: self.bounds.row_end as i32,
            n_tr: counts.top_right,
            n_bl: counts.bottom_left,
            mode: IntraPredMode::from_index(mode).unwrap_or(IntraPredMode::Dc),
            tw4: tw4 as i32,
            th4: th4 as i32,
            intra_flags: angle | flags,
        };
        let limits = (
            ((self.geometry.columns - self.column) << UNIT) as i32,
            ((self.geometry.rows - self.row) << UNIT) as i32,
        );
        self.predict(0, &request, top_sb_edge, topleft, limits);
    }

    /// Prepare the edge of a transform and predict it into its plane.
    fn predict(
        &mut self,
        plane: usize,
        request: &EdgeRequest,
        top_sb_edge: bool,
        topleft: usize,
        (max_width, max_height): (i32, i32),
    ) {
        let depth = self.depth();
        let offset = self
            .planes
            .offset(plane, request.y as u32, request.x as u32);
        let stride = self.planes.stride[plane];
        let data = &mut self.planes.data[plane];
        // The pre-filter row above a superblock is the picture's own row
        // here: loop filters run after the whole tile.
        let row_above = (top_sb_edge && request.y > 0)
            .then(|| SampleRef::new(data, offset - stride - ((request.x as usize) << UNIT)));
        let mode = ipred::prepare_intra_edges(
            request,
            data,
            offset,
            stride,
            row_above,
            &mut self.edge,
            topleft,
            depth,
        );
        let (tw, th) = (
            (request.tw4 << UNIT) as usize,
            (request.th4 << UNIT) as usize,
        );
        ipred::predict(
            mode,
            data,
            offset,
            stride,
            &self.edge,
            topleft,
            tw,
            th,
            request.intra_flags,
            max_width,
            max_height,
            depth,
        );
    }

    /// Above-right and below-left availability of a transform at (`x`,
    /// `y`) of `w` x `h` units in plane type `plane` (dav2d
    /// `recon_b_luma_tx` and `recon_b`'s chroma `n_tr`, `n_bl`), `luma`
    /// its luma position (which the superblock tests use, as dav2d does)
    /// and (`mx`, `my`) its place in the superblock's coded masks.
    #[allow(clippy::many_single_char_names, reason = "dav2d's x, y, w, h")]
    fn edge_counts(
        &self,
        plane: usize,
        (x, y): (i32, i32),
        (w, h): (i32, i32),
        luma: (i32, i32),
        (mx, my): (i32, i32),
    ) -> EdgeCounts {
        let shift = plane as u32;
        let superblock = (self.geometry.superblock_units >> shift) as i32;
        let luma_superblock = self.geometry.superblock_units as i32;
        let (column_start, column_end) = (
            self.bounds.column_start as i32,
            self.bounds.column_end as i32 >> shift,
        );
        let (row_start, row_end) = (
            self.bounds.row_start as i32,
            self.bounds.row_end as i32 >> shift,
        );
        let mut counts = EdgeCounts {
            top_right: 0,
            bottom_left: 0,
        };
        let coded = &self.coded[plane];
        if luma.1 > row_start {
            let mut w2 = w.min(column_end - x - w);
            if luma.1 & (luma_superblock - 1) == 0 {
                counts.top_right = w2;
            } else {
                let end = ((x + superblock) & !(superblock - 1)).min(column_end);
                w2 = w2.min(end - x - w);
                if plane > 0 {
                    w2 = w.min(end - x - w);
                }
                if w2 != 0 {
                    let line = coded[(my - 1).rem_euclid(MASK_UNITS as i32) as usize];
                    let bits = (line >> ((mx + w) as u32 & (MASK_UNITS - 1))) as u32;
                    counts.top_right = ((COUNT_GUARD | !bits).trailing_zeros() as i32).min(w2);
                }
            }
        }
        if luma.0 > column_start {
            let end = ((y + superblock) & !(superblock - 1)).min(row_end);
            let h2 = h.min(end - y - h);
            if h2 == 0 {
                counts.bottom_left = 0;
            } else if luma.0 & (luma_superblock - 1) == 0 {
                counts.bottom_left = h2;
            } else {
                let mask = 1_u64 << ((mx - 1) as u32 & (MASK_UNITS - 1));
                let mut n = 0;
                while n < h2 {
                    let row = (my + n + h) as usize;
                    if coded.get(row).is_none_or(|line| line & mask == 0) {
                        break;
                    }
                    n += 1;
                }
                counts.bottom_left = n;
            }
        }
        counts
    }

    /// Add one luma transform's residual: the secondary transform, then
    /// the inverse (dav2d `recon_b_luma_tx`, "eob != -1").
    pub(super) fn add_luma_residual(
        &mut self,
        record: &BlockRecord,
        modes: &IntraModes,
        (tx, y_mode): (TxSize, u8),
        residual: Residual,
        mut eob: u16,
    ) {
        let size = rect(tx);
        let depth = self.depth();
        let extra = (residual.tx_type >> tx_type::EXTRA_SHIFT) as u8;
        let mut coefficients = std::mem::take(&mut self.luma_coefficients);
        if extra != 0 {
            let intra = record.intra && !record.intrabc;
            let transpose = !(intra && UNTRANSPOSED_MODES.contains(&y_mode));
            eob = stx::apply_secondary_with_dsp(
                self.dsp,
                &mut coefficients,
                size,
                extra,
                transpose,
                eob,
                depth,
            );
        }
        let mut kind = TxType(residual.tx_type & tx_type::TYPE_MASK);
        if self.lossless() && record.intra && !record.intrabc && modes.dpcm[0] {
            kind = kind.with_dpcm(dpcm(modes.y_mode));
        } else if self.sequence.transform.inter_ddt && !record.intra {
            // (Flipped) ADST becomes the (flipped) DDT in inter blocks.
            kind = kind.with_inter_ddt(size);
        }
        let offset = self.planes.offset(0, self.row, self.column);
        let stride = self.planes.stride[0];
        itx::inverse_transform_add_with_dsp(
            self.dsp,
            &mut self.planes.data[0],
            (offset, stride),
            &mut coefficients,
            (size, kind),
            eob,
            depth,
        );
        self.luma_coefficients = coefficients;
    }

    /// Predict a chroma block from luma (dav2d `cfl`).
    pub(super) fn predict_cfl(
        &mut self,
        record: &BlockRecord,
        chroma: BlockSize,
        modes: &IntraModes,
        tx: TxSize,
    ) {
        let (cbx, cby) = (record.chroma_column, record.chroma_row);
        let has_top = cby > self.bounds.row_start;
        let has_left = cbx > self.bounds.column_start;
        let superblock = self.geometry.superblock_units;
        let top_edge = cby & (superblock - 1) == 0;
        let (utw4, uth4) = tx.units();
        let ctw4 = utw4.min((self.geometry.columns - cbx).div_ceil(2));
        let cth4 = uth4.min((self.geometry.rows - cby).div_ceil(2));
        let (ctw, cth) = tx.samples();
        let edges = flag32(has_top, CFL_HAS_TOP)
            | flag32(has_left, CFL_HAS_LEFT)
            | flag32(top_edge, CFL_IS_TOP_SB_EDGE);
        if modes.cfl_type == CflType::MultiHypothesis {
            self.predict_mhccp(record, modes, (ctw4, cth4), (ctw, cth), edges);
            return;
        }
        let filter = u32::from(self.sequence.intra.cfl_downsample_filter);
        let alpha = |value: i8, shift: u32, mask: u32| ((value as u32) << shift) & mask;
        let flags = filter
            | edges
            | alpha(modes.cfl_alpha[0], CFL_ALPHA_U_SHIFT, CFL_ALPHA_U_MASK)
            | alpha(modes.cfl_alpha[1], CFL_ALPHA_V_SHIFT, CFL_ALPHA_V_MASK);
        let (cbw4, cbh4) = chroma.units();
        let (wpad, hpad) = (
            (cbw4.div_ceil(2) - ctw4) as usize,
            (cbh4.div_ceil(2) - cth4) as usize,
        );
        let depth = self.depth();
        let luma_offset = self.planes.offset(0, cby, cbx);
        let chroma_offset = self.planes.offset(1, cby >> 1, cbx >> 1);
        let [luma_stride, chroma_stride, _] = self.planes.stride;
        let [luma, u, v] = &mut self.planes.data;
        // Two luma rows above, or the one row above at a superblock's top.
        let luma_top = if top_edge {
            luma_offset.saturating_sub(luma_stride)
        } else {
            luma_offset - 2 * luma_stride
        };
        let u_top = top_edge.then_some(chroma_offset.saturating_sub(chroma_stride));
        let (u_top_row, v_top_row) = match u_top {
            Some(at) => (Some(u[at..].to_vec()), Some(v[at..].to_vec())),
            None => (None, None),
        };
        ipred::cfl_pred_420(
            modes.cfl_type == CflType::Implicit,
            SampleRef::new(luma, luma_top),
            SampleRef::new(luma, luma_offset),
            luma_stride,
            CflChroma {
                buf: u,
                offset: chroma_offset,
                top_sb_edge: u_top_row.as_deref().map(|row| SampleRef::new(row, 0)),
            },
            CflChroma {
                buf: v,
                offset: chroma_offset,
                top_sb_edge: v_top_row.as_deref().map(|row| SampleRef::new(row, 0)),
            },
            chroma_stride,
            wpad,
            hpad,
            ctw,
            cth,
            flags,
            depth,
        );
    }

    /// Multi-hypothesis chroma from luma (dav2d `cfl`'s `CFL_MHCCP`
    /// branch): the edge counts, the downsampled luma, the fitted model
    /// per plane.
    #[allow(clippy::too_many_lines, reason = "dav2d's one MHCCP sequence")]
    fn predict_mhccp(
        &mut self,
        record: &BlockRecord,
        modes: &IntraModes,
        (ctw4, cth4): (u32, u32),
        (ctw, cth): (usize, usize),
        edges: u32,
    ) {
        let (cbx, cby) = (record.chroma_column, record.chroma_row);
        let (has_t, has_l) = (edges & CFL_HAS_TOP != 0, edges & CFL_HAS_LEFT != 0);
        let top_edge = edges & CFL_IS_TOP_SB_EDGE != 0;
        let direction = modes.mhccp_direction;
        let (dir_l, dir_t) = (
            i32::from(direction == CflMhDir::Left as u8),
            i32::from(direction == CflMhDir::Top as u8),
        );
        let (ssbx, ssby) = ((cbx >> 1) as i32, (cby >> 1) as i32);
        let (cbx4, cby4) = (((cbx & 63) >> 1) as i32, ((cby & 63) >> 1) as i32);
        let (ctw4, cth4) = (ctw4 as i32, cth4 as i32);
        let superblock = self.geometry.superblock_units as i32;
        let (mut refw, mut refh, mut left_columns, mut top_rows) = (0, 0, 0, 0);
        let (mut n_tr, mut n_bl) = (0, 0);
        if has_t {
            let csb = superblock >> 1;
            let tile_end = (self.bounds.column_end >> 1) as i32;
            let mut w = (ctw4.min(tile_end - ssbx - ctw4)).max(0);
            if top_edge {
                n_tr = w;
            } else {
                let end = ((ssbx + csb) & !(csb - 1)).min(tile_end);
                w = ctw4.min(end - ssbx - ctw4);
                if w != 0 {
                    let line = self.coded[1][(cby4 - 1).rem_euclid(64) as usize];
                    let bits = (line >> ((cbx4 + ctw4) as u32 & 63)) as u32;
                    n_tr = ((COUNT_GUARD | !bits).trailing_zeros() as i32).min(w);
                }
            }
            refw += (ctw4 + n_tr) * 4;
            top_rows = 1 + dir_t;
        }
        if has_l {
            let csb = superblock >> 1;
            let end = ((ssby + csb) & !(csb - 1))
                .min((self.bounds.row_end >> 1) as i32)
                .max(0);
            let h = cth4.min(end - ssby - cth4);
            if (cbx as i32) & (superblock - 1) == 0 || h == 0 {
                n_bl = h;
            } else {
                let mask = 1_u64 << ((cbx4 - 1) as u32 & 63);
                while n_bl < h
                    && self.coded[1]
                        .get((cby4 + n_bl + cth4) as usize)
                        .is_some_and(|l| l & mask != 0)
                {
                    n_bl += 1;
                }
            }
            refh += (cth4 + n_bl) * 4;
            left_columns = 1 + dir_l;
        }
        let (mut needed_top, mut needed_left) = (0, 0);
        let (ht, hl) = (i32::from(has_t), i32::from(has_l));
        if has_t {
            refw = (refw + hl - 1 - i32::from(!has_l)).min(128 / 2 - 2);
            needed_top = refw + dir_l;
        }
        if has_l {
            refh = (refh + ht - 1 - i32::from(!has_t)).min(128 / 2 - 2);
            needed_left = refh + dir_t;
            refh -= ht;
        }
        if has_t && dir_t != 0 {
            let first = i32::from(!has_l && needed_top > ctw as i32);
            needed_top = first + needed_top.max(ctw as i32 + left_columns);
        } else if has_l && dir_l != 0 {
            let first = i32::from(!has_t && needed_left > cth as i32);
            needed_left = first + needed_left.max(cth as i32 + top_rows);
        }
        let flags = edges | u32::from(direction);
        let filter = u32::from(self.sequence.intra.cfl_downsample_filter);
        let depth = self.depth();
        let mut luma = vec![T::default(); CFL_MH_MAX_LUMA_SIZE];
        let mut imat = [[T::default(); CFL_MH_MAX_EDGE_SAMPLES]; 2];
        let mut mat = [[0_i32; 4]; 3];
        let luma_offset = self.planes.offset(0, cby, cbx);
        let luma_stride = self.planes.stride[0];
        {
            let plane = &self.planes.data[0];
            let row_above =
                (top_edge && cby > 0).then(|| SampleRef::new(plane, luma_offset - luma_stride));
            ipred::cfl_mh_gen_y_420(
                &mut luma,
                SampleRef::new(plane, luma_offset),
                row_above,
                luma_stride,
                needed_top,
                needed_left - top_rows,
                ctw,
                cth,
                flags,
                filter,
            );
        }
        if has_t || has_l {
            ipred::cfl_mh_gen_mat(&mut mat, &mut imat, &luma, refw, refh, flags, depth);
        }
        let source = SampleRef::new(&luma, top_rows as usize * CFL_MH_TOP_STRIDE);
        let mh = match direction {
            1 => CflMhDir::Top,
            2 => CflMhDir::Left,
            _ => CflMhDir::Center,
        };
        for plane in 1..=2 {
            let offset = self.planes.offset(plane, cby >> 1, cbx >> 1);
            let stride = self.planes.stride[plane];
            let mut alpha = [0, 0, 1 << 16, 0];
            if has_t || has_l {
                let data = &self.planes.data[plane];
                let row_above =
                    (top_edge && cby > 0).then(|| SampleRef::new(data, offset - stride));
                ipred::cfl_mh_calc_alphas(
                    &mut alpha,
                    SampleRef::new(data, offset),
                    row_above,
                    stride,
                    refw,
                    refh,
                    &mat,
                    &imat,
                    flags,
                    depth,
                );
            }
            ipred::cfl_mh_pred(
                mh,
                &mut self.planes.data[plane],
                offset,
                stride,
                source,
                ctw,
                cth,
                &alpha,
                flags,
                depth,
            );
        }
    }

    /// Predict one chroma transform of a plane (dav2d `recon_b`'s chroma
    /// "intra prediction").
    pub(super) fn predict_chroma(
        &mut self,
        record: &BlockRecord,
        plane: usize,
        (x, y): (u32, u32),
        tx: TxSize,
        (mode, angle): (u8, i32),
        modes: &IntraModes,
    ) {
        let (tw4, th4) = tx.units();
        let (cbx, cby) = (record.chroma_column, record.chroma_row);
        let (px, py) = ((cbx >> 1) + x, (cby >> 1) + y);
        let (tw, th) = tx.samples();
        let has_top = cby + (y << 1) > self.bounds.row_start;
        let has_left = cbx + (x << 1) > self.bounds.column_start;
        let mut counts = EdgeCounts {
            top_right: 0,
            bottom_left: 0,
        };
        let superblock = self.geometry.superblock_units;
        let mask = (
            (((cbx & 63) >> 1) + x) as i32,
            (((cby & 63) >> 1) + y) as i32,
        );
        let all = self.edge_counts(
            1,
            (px as i32, py as i32),
            (tw4 as i32, th4 as i32),
            ((cbx + x) as i32, (cby + y) as i32),
            mask,
        );
        if has_top && tw < CHROMA_EDGE_MAX_SAMPLES {
            counts.top_right = all.top_right;
        }
        if has_left && th < CHROMA_EDGE_MAX_SAMPLES {
            counts.bottom_left = all.bottom_left;
        }
        let sequence = &self.sequence.intra;
        let mut ibp = sequence.ibp && tx != TxSize::TX4;
        let smooth = smooth_flags(ibp, modes.smooth_above[1], modes.smooth_left[1]);
        ibp &= mode == mode::DC;
        let flags = smooth
            | flag(ibp, ANGLE_IBP_FLAG)
            | flag(sequence.ibp, ANGLE_SEQHDR_IBP_FLAG)
            | flag(sequence.edge_filter, ANGLE_USE_EDGE_FILTER_FLAG)
            | flag(has_left, ANGLE_HAS_LEFT_FLAG)
            | flag(has_top, ANGLE_HAS_TOP_FLAG);
        let request = EdgeRequest {
            x: px as i32,
            y: py as i32,
            end_x: (self.bounds.column_end >> 1) as i32,
            end_y: (self.bounds.row_end >> 1) as i32,
            n_tr: counts.top_right,
            n_bl: counts.bottom_left,
            mode: IntraPredMode::from_index(mode).unwrap_or(IntraPredMode::Dc),
            tw4: tw4 as i32,
            th4: th4 as i32,
            intra_flags: angle | flags,
        };
        let top_sb_edge = (cby + y) & (superblock - 1) == 0;
        let limits = (
            ((self.geometry.columns as i32) - (cbx + x) as i32) << UNIT,
            ((self.geometry.rows as i32) - (cby + y) as i32) << UNIT,
        );
        self.predict(plane, &request, top_sb_edge, EDGE_CORNER, limits);
    }

    /// Add a chroma transform's residuals (both planes): the
    /// cross-chroma transform, then each plane's inverse (dav2d `recon_b`,
    /// "inverse transform").
    pub(super) fn add_chroma_residuals(
        &mut self,
        record: &BlockRecord,
        modes: &IntraModes,
        (x, y): (u32, u32),
        tx: TxSize,
        index: usize,
        intra: bool,
    ) {
        let depth = self.depth();
        let mut store = std::mem::take(&mut self.chroma_residuals);
        let [u_residual, v_residual] = store.residuals[index];
        let (Some(mut u_res), Some(mut v_res)) = (u_residual, v_residual) else {
            self.chroma_residuals = store;
            return;
        };
        let eob = |r: &Residual| r.eob.map_or(-1, i32::from);
        let start = index * COEFFICIENTS_PER_UNIT;
        let cross = u_res.tx_type >> tx_type::EXTRA_SHIFT;
        if cross != 0 && eob(&u_res) >= i32::from(intra) {
            let (tw, th) = tx.samples();
            let length = tw.min(32) * th.min(32);
            let [u, v] = &mut store.coefficients;
            itx::cross_chroma(
                &mut u[start..start + length],
                &mut v[start..start + length],
                &itx::CCTX_ANGLES[usize::from(cross) - 1],
                depth,
            );
            let larger = if eob(&v_res) > eob(&u_res) {
                v_res.eob
            } else {
                u_res.eob
            };
            (u_res.eob, v_res.eob) = (larger, larger);
            u_res.tx_type &= tx_type::TYPE_MASK;
            v_res.tx_type = u_res.tx_type;
        }
        let size = rect(tx);
        let (column, row) = (
            (record.chroma_column >> 1) + x,
            (record.chroma_row >> 1) + y,
        );
        let dpcm_block = self.lossless() && intra && modes.dpcm[1];
        for (plane, residual) in [u_res, v_res].into_iter().enumerate() {
            let Some(eob) = residual.eob else { continue };
            let mut kind = TxType(residual.tx_type & tx_type::TYPE_MASK);
            if dpcm_block {
                kind = kind.with_dpcm(dpcm(modes.uv_mode));
            } else if self.sequence.transform.inter_ddt && !record.intra {
                kind = kind.with_inter_ddt(size);
            }
            let offset = self.planes.offset(plane + 1, row, column);
            let stride = self.planes.stride[plane + 1];
            itx::inverse_transform_add_with_dsp(
                self.dsp,
                &mut self.planes.data[plane + 1],
                (offset, stride),
                &mut store.coefficients[plane][start..],
                (size, kind),
                eob,
                depth,
            );
        }
        self.chroma_residuals = store;
    }
}
