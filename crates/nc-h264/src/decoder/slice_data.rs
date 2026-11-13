//! `slice_data()` (§7.3.4): the macroblock loop of one slice, with
//! `mb_skip_run` (CAVLC) or `mb_skip_flag` / `end_of_slice_flag` (CABAC),
//! decoding and reconstructing each macroblock as it is parsed.

use super::cabac_syntax::CabacSyntax;
use super::cavlc::CavlcSyntax;
use super::deblock::{DeblockContext, RowDeblocker, SliceFilter};
use super::mb::{self, MbExtra, SliceParams};
use super::mbstate::{MbState, Neighbours};
use super::recon::{self, ReconContext};
use super::slice::SliceType;
#[cfg(feature = "analysis")]
use super::stats::{self, MbContext};
use super::syntax::MbSyntax;
use super::transform::Residual;
use super::{DecodeError, Planes, Result};
#[cfg(feature = "analysis")]
use nc_common::stream_stats::StreamStats;

/// The state one slice's macroblocks are decoded into.
pub(super) struct SliceDecoder<'a> {
    pub params: SliceParams<'a>,
    pub recon: ReconContext<'a>,
    /// The picture's planes.
    pub planes: Planes<'a>,
    /// Deblocking control of the picture's slices so far.
    pub slices: &'a [SliceFilter],
    /// `MbState::slice` of the picture's first slice.
    pub first_slice: u32,
    /// Filters macroblock rows a row behind decoding.
    pub deblocker: &'a mut RowDeblocker,
    pub mbs: &'a mut [MbState],
    pub width_mbs: usize,
    /// `MbState::slice` of this slice's macroblocks.
    pub slice_no: u32,
    pub residual: &'a mut Residual,
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

impl SliceDecoder<'_> {
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
        *cur = MbState::EMPTY;
        cur.slice = self.slice_no;
        let nb = Neighbours::of(before, self.width_mbs, addr, self.slice_no);
        #[cfg(feature = "analysis")]
        let (qp_pred, start) = (self.qp, syntax.bits_consumed());
        if skipped(syntax, &nb) {
            mb::skip(&self.params, &nb, cur, self.qp)?;
            syntax.no_qp_delta();
        } else {
            mb::decode(
                syntax,
                &self.params,
                &nb,
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
                monochrome: self.recon.monochrome,
            };
            self.deblocker.filter_through(row, &mut self.planes, &ctx);
        }
        Ok(())
    }

    /// The CABAC macroblock loop from `first_mb`.
    pub fn cabac(&mut self, syntax: &mut CabacSyntax<'_>, first_mb: usize) -> Result<()> {
        let mut addr = first_mb;
        loop {
            let p_slice = self.params.slice_type == SliceType::P;
            self.macroblock(syntax, addr, |syntax, nb| {
                p_slice && syntax.mb_skip_flag(nb)
            })?;
            addr += 1;
            if syntax.end_of_slice() {
                return Ok(());
            }
        }
    }

    /// The CAVLC macroblock loop from `first_mb`.
    pub fn cavlc(&mut self, syntax: &mut CavlcSyntax<'_, '_>, first_mb: usize) -> Result<()> {
        let mut addr = first_mb;
        loop {
            if self.params.slice_type == SliceType::P {
                let run = syntax.mb_skip_run()? as usize;
                if run > self.mbs.len() {
                    return Err(DecodeError::Malformed("mb_skip_run"));
                }
                for _ in 0..run {
                    self.macroblock(syntax, addr, |_, _| true)?;
                    addr += 1;
                }
                if run > 0 && !syntax.reader.more_rbsp_data() {
                    return Ok(());
                }
            }
            self.macroblock(syntax, addr, |_, _| false)?;
            addr += 1;
            if !syntax.reader.more_rbsp_data() {
                return Ok(());
            }
        }
    }
}
