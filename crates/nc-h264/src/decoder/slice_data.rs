//! `slice_data()` (§7.3.4): the macroblock loop of one slice, with
//! `mb_skip_run` (CAVLC) or `mb_skip_flag` / `end_of_slice_flag` (CABAC),
//! decoding and reconstructing each macroblock as it is parsed, in raster
//! order or, with slice groups, along `NextMbAddress` (eq. 8-16). MBAFF
//! frames are decoded pair by pair (`mb_field_decoding_flag`, its inference
//! for skipped pairs), each macroblock with the context of its frame or
//! field coding.

use super::cabac_syntax::CabacSyntax;
use super::cavlc::{CavlcSlice, CavlcSyntax};
use super::deblock::{DeblockContext, RowDeblocker, SliceFilter};
use super::mb::{self, MbExtra, SliceParams};
use super::mbstate::{MbState, Neighbours};
use super::recon::{self, ReconContext};
use super::sample::H264Sample;
use super::slice::SliceType;
use super::slice_groups;
#[cfg(feature = "analysis")]
use super::stats::{self, MbContext};
use super::structure::{Parity, field_rows};
use super::syntax::MbSyntax;
use super::transform::Residual;
use super::{DecodeError, Planes, Result};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;

/// What the field macroblocks of an MBAFF frame are decoded with, by
/// parity (top, bottom): their field reference lists, weights and direct
/// prediction (§8.4.2.1).
pub(super) struct FieldMbContexts<'a, T: H264Sample> {
    pub params: [SliceParams<'a>; 2],
    pub recon: [ReconContext<'a, T>; 2],
}

/// The state one slice's macroblocks are decoded into.
pub(super) struct SliceDecoder<'a, T: H264Sample> {
    pub params: SliceParams<'a>,
    pub recon: ReconContext<'a, T>,
    /// The picture's planes.
    pub planes: Planes<'a, T>,
    /// Deblocking control of the picture's slices so far.
    pub slices: &'a [SliceFilter],
    /// `MbState::slice` of the picture's first slice.
    pub first_slice: u32,
    /// Filters macroblock rows a row behind decoding.
    pub deblocker: &'a mut RowDeblocker,
    pub mbs: &'a mut [MbState],
    /// `MbToSliceGroupMap` of a picture with slice groups.
    pub slice_groups: Option<&'a [u8]>,
    pub width_mbs: usize,
    /// A field picture: its macroblocks are field macroblocks.
    pub field: bool,
    /// An MBAFF frame: the contexts of its field macroblocks.
    pub pair_fields: Option<FieldMbContexts<'a, T>>,
    /// `MbState::slice` of this slice's macroblocks.
    pub slice_no: u32,
    pub residual: &'a mut Residual<T::Coeff>,
    /// `I_PCM` sample scratch (reused: only `I_PCM` macroblocks write it).
    pub extra: &'a mut MbExtra,
    /// `QP_Y,PRED`: the slice QP, then the previous macroblock's.
    pub qp: i32,
    /// Macroblocks decoded so far.
    pub decoded: usize,
    /// Where block records go when analysis is enabled.
    #[cfg(feature = "analysis")]
    pub stats: Option<&'a mut StreamStats>,
    /// Whether the slice is CABAC-coded (analysis).
    #[cfg(feature = "analysis")]
    pub cabac: bool,
}

impl<T: H264Sample> SliceDecoder<'_, T> {
    /// The macroblock after `addr` in the slice (eq. 8-16).
    #[inline]
    fn next_addr(&self, addr: usize) -> usize {
        match self.slice_groups {
            None => addr + 1,
            Some(map) => slice_groups::next_mb(map, addr),
        }
    }

    /// Decode (or skip) macroblock `addr` and reconstruct it; `skipped`
    /// decides from its neighbours whether it is skipped.
    fn macroblock<S: MbSyntax>(
        &mut self,
        syntax: &mut S,
        addr: usize,
        skipped: impl FnOnce(&mut S, &Neighbours<'_>) -> bool,
    ) -> Result<()> {
        if addr >= self.mbs.len() {
            return Err(DecodeError::Malformed("slice runs past the picture"));
        }
        if self.mbs[addr].slice >= self.first_slice {
            return Err(DecodeError::Malformed("macroblock decoded twice"));
        }
        // Decoded in place: every neighbour precedes `addr`.
        let (before, rest) = self.mbs.split_at_mut(addr);
        let cur = &mut rest[0];
        cur.reset(
            self.slice_no,
            self.params.slice_type == SliceType::B,
            self.field,
        );
        let nb = Neighbours::of(before, self.width_mbs, addr, self.slice_no);
        #[cfg(feature = "analysis")]
        let (qp_pred, start) = (self.qp, syntax.bits_consumed());
        if skipped(syntax, &nb) {
            if self.params.slice_type == SliceType::B {
                mb::direct_16x16::<T>(&self.params, &nb, cur, addr, true)?;
                cur.qp = self.qp as i8;
            } else {
                mb::skip::<T>(&self.params, &nb, cur, self.qp)?;
            }
            syntax.no_qp_delta();
        } else {
            mb::decode(
                syntax,
                &self.params,
                &nb,
                addr,
                &mut self.qp,
                cur,
                self.residual,
                self.extra,
            )?;
        }
        let position = (addr % self.width_mbs, addr / self.width_mbs);
        #[cfg(feature = "analysis")]
        if let Some(s) = self.stats.as_deref_mut() {
            let mb = MbContext {
                position,
                qp_pred,
                bits: syntax.bits_consumed().saturating_sub(start),
                cabac: self.cabac,
            };
            stats::macroblock(s, cur, &mb);
        }
        recon::reconstruct(
            &self.recon,
            &mut self.planes,
            position,
            cur,
            &nb,
            self.residual,
            self.extra,
        )?;
        self.decoded += 1;
        if let Some(row) = self.deblocker.decoded(addr, self.width_mbs) {
            let ctx = DeblockContext {
                dsp: self.recon.dsp,
                mbs: self.mbs,
                width_mbs: self.width_mbs,
                first_slice: self.first_slice,
                slices: self.slices,
                chroma: self.recon.chroma,
                depths: self.recon.depths,
                field: self.field,
            };
            self.deblocker.filter_through(row, &mut self.planes, &ctx);
        }
        Ok(())
    }

    /// The CABAC macroblock loop from `first_mb`. Out of line, as the
    /// others: its macroblock steps then inline into it (inlined into the
    /// slice instead, they did not: +0.7 % instructions, callgrind).
    #[inline(never)]
    pub fn cabac(&mut self, syntax: &mut CabacSyntax<'_, T>, first_mb: usize) -> Result<()> {
        let mut addr = first_mb;
        let slice_type = self.params.slice_type;
        loop {
            self.macroblock(syntax, addr, |syntax, nb| {
                !slice_type.is_intra() && syntax.mb_skip_flag(slice_type, nb)
            })?;
            addr = self.next_addr(addr);
            if syntax.end_of_slice() {
                return Ok(());
            }
        }
    }

    /// The CAVLC macroblock loop from `first_mb`, of a whole slice or of
    /// a partitioned one.
    #[inline(never)]
    pub fn cavlc<S: CavlcSlice>(&mut self, syntax: &mut S, first_mb: usize) -> Result<()> {
        let mut addr = first_mb;
        loop {
            if !self.params.slice_type.is_intra() {
                let run = syntax.mb_skip_run()? as usize;
                if run > self.mbs.len() {
                    return Err(DecodeError::Malformed("mb_skip_run"));
                }
                for _ in 0..run {
                    self.macroblock(syntax, addr, |_, _| true)?;
                    addr = self.next_addr(addr);
                }
                if run > 0 && !syntax.more_rbsp_data() {
                    return Ok(());
                }
            }
            self.macroblock(syntax, addr, |_, _| false)?;
            addr = self.next_addr(addr);
            if !syntax.more_rbsp_data() {
                return Ok(());
            }
        }
    }

    /// The CABAC macroblock loop of an MBAFF frame from macroblock
    /// `first_mb` (the top of a pair). A skipped top macroblock takes the
    /// pair's `mb_field_decoding_flag`, which follows the bottom one's
    /// `mb_skip_flag` (§7.3.4); both skip flags take their contexts from
    /// the inferred flag until it is read.
    #[inline(never)]
    pub fn cabac_pairs(&mut self, syntax: &mut CabacSyntax<'_, T>, first_mb: usize) -> Result<()> {
        let slice_type = self.params.slice_type;
        let inter = slice_type != SliceType::I;
        let mut top = first_mb;
        loop {
            self.check_pair(top)?;
            let mut field = self.inferred_field(top);
            let (left_field, above_field) = self.neighbour_pairs_field(top);
            let skip_top = inter && {
                let nb = Neighbours::of_pair(
                    &self.mbs[..top],
                    self.width_mbs,
                    (top, field),
                    self.slice_no,
                );
                syntax.mb_skip_flag(slice_type, &nb)
            };
            let mut skip_bottom = None;
            if skip_top {
                // The bottom's skip flag comes next; it sees the top as a
                // skipped macroblock of this slice.
                self.mbs[top].reset(self.slice_no, slice_type == SliceType::B, field);
                let nb = Neighbours::of_pair(
                    &self.mbs[..=top],
                    self.width_mbs,
                    (top + 1, field),
                    self.slice_no,
                );
                let skipped = syntax.mb_skip_flag(slice_type, &nb);
                if !skipped {
                    field = syntax.mb_field_decoding_flag(left_field, above_field);
                }
                skip_bottom = Some(skipped);
            } else {
                field = syntax.mb_field_decoding_flag(left_field, above_field);
            }
            self.pair_macroblock(syntax, top, field, skip_top)?;
            let skip_bottom = match skip_bottom {
                Some(skipped) => skipped,
                None => {
                    inter && {
                        let nb = Neighbours::of_pair(
                            &self.mbs[..=top],
                            self.width_mbs,
                            (top + 1, field),
                            self.slice_no,
                        );
                        syntax.mb_skip_flag(slice_type, &nb)
                    }
                }
            };
            self.pair_macroblock(syntax, top + 1, field, skip_bottom)?;
            top += 2;
            if syntax.end_of_slice() {
                return Ok(());
            }
        }
    }

    /// The CAVLC macroblock loop of an MBAFF frame from macroblock
    /// `first_mb`. A run of skipped macroblocks that ends on a top one is
    /// followed by the pair's `mb_field_decoding_flag` (the bottom one is
    /// coded); a pair skipped whole infers it.
    #[inline(never)]
    pub fn cavlc_pairs(
        &mut self,
        syntax: &mut CavlcSyntax<'_, '_, T>,
        first_mb: usize,
    ) -> Result<()> {
        let mut addr = first_mb;
        let mut field = false;
        loop {
            if self.params.slice_type != SliceType::I {
                let run = syntax.mb_skip_run()? as usize;
                if run > self.mbs.len() {
                    return Err(DecodeError::Malformed("mb_skip_run"));
                }
                for skipped in 0..run {
                    if addr.is_multiple_of(2) {
                        self.check_pair(addr)?;
                        field = if skipped + 1 == run && syntax.more_rbsp_data() {
                            syntax.mb_field_decoding_flag()?
                        } else {
                            self.inferred_field(addr)
                        };
                    }
                    self.pair_macroblock(syntax, addr, field, true)?;
                    addr += 1;
                }
                if run > 0 && !syntax.more_rbsp_data() {
                    return Ok(());
                }
            }
            if addr.is_multiple_of(2) {
                self.check_pair(addr)?;
                field = syntax.mb_field_decoding_flag()?;
            }
            self.pair_macroblock(syntax, addr, field, false)?;
            addr += 1;
            if !syntax.more_rbsp_data() {
                return Ok(());
            }
        }
    }

    /// The pair whose top macroblock is `top` lies in the picture and is
    /// not decoded yet.
    fn check_pair(&self, top: usize) -> Result<()> {
        if !top.is_multiple_of(2) || top + 1 >= self.mbs.len() {
            return Err(DecodeError::Malformed("slice runs past the picture"));
        }
        if self.mbs[top].slice >= self.first_slice {
            return Err(DecodeError::Malformed("macroblock decoded twice"));
        }
        Ok(())
    }

    /// Whether the left and the above pair of the pair at `top` are field
    /// pairs of this slice (`ctxIdxInc` of `mb_field_decoding_flag`).
    fn neighbour_pairs_field(&self, top: usize) -> (bool, bool) {
        let pair = top / 2;
        let field_pair = |available: bool, index: usize| {
            available && {
                let mb = &self.mbs[2 * index];
                mb.slice == self.slice_no && mb.field
            }
        };
        (
            field_pair(!pair.is_multiple_of(self.width_mbs), pair.wrapping_sub(1)),
            field_pair(pair >= self.width_mbs, pair.wrapping_sub(self.width_mbs)),
        )
    }

    /// The `mb_field_decoding_flag` inferred for the pair at `top` when
    /// both its macroblocks are skipped (§7.4.4): the left pair's, else the
    /// above pair's (of this slice), else frame.
    fn inferred_field(&self, top: usize) -> bool {
        let pair = top / 2;
        let of = |available: bool, index: usize| {
            available
                .then(|| &self.mbs[2 * index])
                .filter(|mb| mb.slice == self.slice_no)
                .map(|mb| mb.field)
        };
        of(!pair.is_multiple_of(self.width_mbs), pair.wrapping_sub(1))
            .or_else(|| of(pair >= self.width_mbs, pair.wrapping_sub(self.width_mbs)))
            .unwrap_or(false)
    }

    /// Decode (`skipped` or coded) and reconstruct macroblock `addr` of an
    /// MBAFF frame, a `field` macroblock or not: a field macroblock with
    /// its parity's lists, in its field's rows of the pair.
    fn pair_macroblock<S: MbSyntax>(
        &mut self,
        syntax: &mut S,
        addr: usize,
        field: bool,
        skipped: bool,
    ) -> Result<()> {
        let parity = addr % 2;
        let b_slice = self.params.slice_type == SliceType::B;
        let (params, recon) = match (&self.pair_fields, field) {
            (Some(fields), true) => (&fields.params[parity], &fields.recon[parity]),
            _ => (&self.params, &self.recon),
        };
        let (before, rest) = self.mbs.split_at_mut(addr);
        let cur = &mut rest[0];
        cur.reset(self.slice_no, b_slice, field);
        let nb = Neighbours::of_pair(before, self.width_mbs, (addr, field), self.slice_no);
        #[cfg(feature = "analysis")]
        let (qp_pred, start) = (self.qp, syntax.bits_consumed());
        if skipped {
            if b_slice {
                mb::direct_16x16::<T>(params, &nb, cur, addr, true)?;
                cur.qp = self.qp as i8;
            } else {
                mb::skip::<T>(params, &nb, cur, self.qp)?;
            }
            syntax.no_qp_delta();
        } else {
            mb::decode(
                syntax,
                params,
                &nb,
                addr,
                &mut self.qp,
                cur,
                self.residual,
                self.extra,
            )?;
        }
        let pair = addr / 2;
        let (px, py) = (pair % self.width_mbs, pair / self.width_mbs);
        #[cfg(feature = "analysis")]
        if let Some(s) = self.stats.as_deref_mut() {
            let mb = MbContext {
                position: (px, 2 * py + parity),
                qp_pred,
                bits: syntax.bits_consumed().saturating_sub(start),
                cabac: self.cabac,
            };
            stats::macroblock(s, cur, &mb);
        }
        if field {
            let parity = if parity == 0 {
                Parity::Top
            } else {
                Parity::Bottom
            };
            let mut planes = field_rows(&mut self.planes, parity);
            recon::reconstruct(
                recon,
                &mut planes,
                (px, py),
                cur,
                &nb,
                self.residual,
                self.extra,
            )?;
        } else {
            let position = (px, 2 * py + parity);
            recon::reconstruct(
                recon,
                &mut self.planes,
                position,
                cur,
                &nb,
                self.residual,
                self.extra,
            )?;
        }
        self.decoded += 1;
        Ok(())
    }
}
