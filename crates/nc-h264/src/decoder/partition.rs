//! Slice data partitioning (Extended profile, §7.3.2.9, §8.2.3): a slice's
//! header and its category-2 syntax elements (macroblock types, prediction,
//! coded block patterns, QP deltas) come in partition A, the residual of
//! its intra macroblocks (category 3) in partition B and that of its inter
//! macroblocks (category 4) in partition C, each its own NAL unit; B and C
//! are absent when their category is empty.
//!
//! [`PendingSlice`] holds a partition A until its B and C have arrived
//! (`super::Decoder` decides when); [`PartitionedCavlc`] then reads each
//! syntax element from its partition.

use std::sync::Arc;

use super::cavlc::{CavlcSlice, CavlcSyntax, CavlcTables};
use super::chroma::ChromaArray;
use super::mbstate::{MbState, Neighbours};
use super::pps::Pps;
use super::reader::Reader;
use super::slice::{SliceHeader, SliceType};
use super::sps::{MAX_FRAME_MBS, Sps};
use super::syntax::{Block4x4, Coeffs, MbSyntax};
use super::{DecodeError, Result};

/// `slice_id` range (§7.4.2.9.1: below `PicSizeInMbs`).
const MAX_SLICE_ID: u32 = MAX_FRAME_MBS as u32 - 1;
/// `redundant_pic_cnt` range (§7.4.3).
const MAX_REDUNDANT_PIC_CNT: u32 = 127;

/// Which residual partition a NAL unit is (`nal_unit_type` 3 or 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Residuals {
    /// Partition B: category 3, the residual of I and SI macroblocks.
    Intra,
    /// Partition C: category 4, the residual of P and B macroblocks.
    Inter,
}

/// A slice whose partition A has arrived, with its B and C so far.
#[derive(Debug)]
pub(super) struct PendingSlice {
    /// Partition A's RBSP: the slice header, `slice_id` and category 2.
    pub a: Vec<u8>,
    pub nal_ref_idc: u32,
    pub header: SliceHeader,
    pub slice_id: u32,
    pub sps: Arc<Sps>,
    pub pps: Arc<Pps>,
    /// Partitions B and C: their RBSPs.
    pub b: Option<Vec<u8>>,
    pub c: Option<Vec<u8>>,
}

impl PendingSlice {
    /// Attach the residual partition `rbsp`.
    pub fn attach(&mut self, which: Residuals, rbsp: Vec<u8>) -> Result<()> {
        let slot = match which {
            Residuals::Intra => &mut self.b,
            Residuals::Inter => &mut self.c,
        };
        if slot.is_some() {
            return Err(DecodeError::Malformed("repeated slice data partition"));
        }
        *slot = Some(rbsp);
        Ok(())
    }
}

/// `slice_id`, read after the slice header of partition A.
pub(super) fn parse_slice_id(r: &mut Reader<'_>) -> Result<u32> {
    r.ue_max(MAX_SLICE_ID, "slice_id")
}

/// The header of partition B or C (§7.3.2.9.2, §7.3.2.9.3): `slice_id`
/// and `redundant_pic_cnt`; leaves `r` at its slice data. (Separate colour
/// planes, whose `colour_plane_id` would follow, are not decoded.)
pub(super) fn parse_residual_header(r: &mut Reader<'_>, pps: &Pps) -> Result<(u32, u32)> {
    let slice_id = parse_slice_id(r)?;
    let redundant_pic_cnt = if pps.redundant_pic_cnt_present {
        r.ue_max(MAX_REDUNDANT_PIC_CNT, "redundant_pic_cnt")?
    } else {
        0
    };
    Ok((slice_id, redundant_pic_cnt))
}

/// The CAVLC syntax of a partitioned slice: the elements of category 2
/// from partition A, the residual of an intra macroblock from B and of an
/// inter one from C (a missing partition reads as empty data).
pub(super) struct PartitionedCavlc<'a, 't, T> {
    a: CavlcSyntax<'a, 't, T>,
    b: CavlcSyntax<'a, 't, T>,
    c: CavlcSyntax<'a, 't, T>,
    constrained_intra_pred: bool,
}

impl<'a, 't, T> PartitionedCavlc<'a, 't, T> {
    /// The syntax over partition A's slice data and B's and C's (each at
    /// its slice data). Partitioned slices are the Extended profile's,
    /// whose SPS codes no chroma format: 4:2:0 (§7.4.2.1.1).
    pub fn new(
        [a, b, c]: [Reader<'a>; 3],
        tables: &'t CavlcTables,
        constrained_intra_pred: bool,
    ) -> Self {
        let chroma = ChromaArray::Yuv420;
        Self {
            a: CavlcSyntax::new(a, tables, chroma),
            b: CavlcSyntax::new(b, tables, chroma),
            c: CavlcSyntax::new(c, tables, chroma),
            constrained_intra_pred,
        }
    }

    /// The partition `cur`'s residual is read from, and the neighbours its
    /// `nC` sees: with `constrained_intra_pred_flag`, an intra macroblock's
    /// inter neighbours count as blocks without coefficients (`nN` = 0).
    ///
    /// That is the rule of the reference decoder (JM's `predict_nnz`) and
    /// of the earlier editions of §9.2.1, which the streams follow; the
    /// 02/2016 text makes such a neighbour unavailable instead (step 5),
    /// which changes `nC` whenever the other neighbour is available.
    fn residual<'n>(
        &mut self,
        nb: &Neighbours<'n>,
        cur: &MbState,
    ) -> (&mut CavlcSyntax<'a, 't, T>, Neighbours<'n>) {
        if !cur.is_intra() {
            return (&mut self.c, *nb);
        }
        let mut nb = *nb;
        if self.constrained_intra_pred {
            let counted = |m: &'n MbState| if m.is_intra() { m } else { &NO_COEFFICIENTS };
            nb.a = nb.a.map(counted);
            nb.b = nb.b.map(counted);
        }
        (&mut self.b, nb)
    }
}

/// A neighbour whose blocks count as having no coefficients.
static NO_COEFFICIENTS: MbState = MbState::EMPTY;

impl<T> CavlcSlice for PartitionedCavlc<'_, '_, T> {
    fn mb_skip_run(&mut self) -> Result<u32> {
        self.a.mb_skip_run()
    }

    fn more_rbsp_data(&self) -> bool {
        self.a.more_rbsp_data()
    }
}

impl<T> MbSyntax for PartitionedCavlc<'_, '_, T> {
    #[cfg(feature = "analysis")]
    fn bits_consumed(&self) -> usize {
        self.a.bits_consumed() + self.b.bits_consumed() + self.c.bits_consumed()
    }

    fn mb_type(&mut self, slice: SliceType, nb: &Neighbours<'_>) -> Result<u32> {
        self.a.mb_type(slice, nb)
    }

    fn sub_mb_type(&mut self, slice: SliceType) -> Result<u8> {
        self.a.sub_mb_type(slice)
    }

    fn transform_size_8x8_flag(&mut self, nb: &Neighbours<'_>) -> Result<bool> {
        self.a.transform_size_8x8_flag(nb)
    }

    fn intra_pred_mode(&mut self) -> Result<Option<u8>> {
        self.a.intra_pred_mode()
    }

    fn intra_chroma_pred_mode(&mut self, nb: &Neighbours<'_>) -> Result<u8> {
        self.a.intra_chroma_pred_mode(nb)
    }

    fn ref_idx(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &MbState,
        list: usize,
        x4: usize,
        y4: usize,
        num_ref_idx_active: usize,
    ) -> Result<u8> {
        self.a.ref_idx(nb, cur, list, x4, y4, num_ref_idx_active)
    }

    fn mvd(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &MbState,
        list: usize,
        x4: usize,
        y4: usize,
    ) -> Result<[i32; 2]> {
        self.a.mvd(nb, cur, list, x4, y4)
    }

    fn coded_block_pattern(&mut self, nb: &Neighbours<'_>, cur: &MbState) -> Result<u8> {
        self.a.coded_block_pattern(nb, cur)
    }

    fn mb_qp_delta(&mut self) -> Result<i32> {
        self.a.mb_qp_delta()
    }

    fn luma_dc<const PLANE: usize>(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        out: &mut Coeffs,
    ) -> Result<()> {
        let (syntax, nb) = self.residual(nb, cur);
        syntax.luma_dc::<PLANE>(&nb, cur, out)
    }

    fn block_4x4<const PLANE: usize>(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        block: Block4x4,
        out: &mut Coeffs,
    ) -> Result<()> {
        let (syntax, nb) = self.residual(nb, cur);
        syntax.block_4x4::<PLANE>(&nb, cur, block, out)
    }

    fn luma_8x8<const PLANE: usize>(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        b8: usize,
        out: &mut Coeffs,
    ) -> Result<()> {
        let (syntax, nb) = self.residual(nb, cur);
        syntax.luma_8x8::<PLANE>(&nb, cur, b8, out)
    }

    fn chroma_dc(
        &mut self,
        nb: &Neighbours<'_>,
        cur: &mut MbState,
        comp: usize,
        out: &mut Coeffs,
    ) -> Result<()> {
        let (syntax, nb) = self.residual(nb, cur);
        syntax.chroma_dc(&nb, cur, comp, out)
    }

    /// `I_PCM` samples are category 3 (§7.3.5): partition B.
    fn pcm_samples(&mut self, out: &mut [u16], bits: [u32; 2]) -> Result<()> {
        self.b.pcm_samples(out, bits)
    }
}
