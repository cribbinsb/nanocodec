//! In-loop filtering in bands of CTB rows as decoding completes them
//! (H.265 §8.7): deblocking, SAO and the reference border of each band
//! while its samples are still in cache, instead of three passes over the
//! whole picture after its last CTB.
//!
//! A band waits until the samples later stages read are final:
//! - deblocking CTB row `r` waits for row `r + 1` to be decoded, because
//!   intra prediction in row `r + 1` reads row `r`'s unfiltered last line;
//! - SAO of row `r` waits for row `r + 1` to be deblocked, because that
//!   row's horizontal edges at its top change row `r`'s last three lines,
//!   and SAO reads one line into the row below;
//! - the border of row `r` follows its SAO.
//!
//! The order of each filter over a picture is the specification's (see
//! the `deblock` and `sao` module docs), so the result is the same as
//! filtering the whole picture at once. Rows complete in raster order in
//! every CTB scan (tiles finish a CTB row only with their last tile
//! column), and a picture's remaining bands are filtered when it ends.

use nc_common::decode::DecodeError;
use nc_dsp::Dsp;

use super::params::Pps;
use super::picture::PictureState;
use super::{deblock, sao};

/// Working memory of the in-loop filters, kept between pictures.
#[derive(Debug, Default)]
pub(super) struct LoopFilter {
    /// Working memory of SAO.
    sao: sao::Work,
}

/// How far a picture's decoding and filtering have come.
#[derive(Debug, Clone)]
pub(super) struct Progress {
    /// Whether each CTB (raster order) is decoded.
    ctbs: Vec<bool>,
    /// CTBs decoded per CTB row.
    decoded: Vec<u32>,
    /// CTB rows deblocked.
    deblocked: usize,
    /// CTB rows with SAO applied and their border extended: final.
    finished: usize,
}

impl Progress {
    /// A picture of `columns` x `rows` CTBs, nothing decoded.
    pub fn new(columns: usize, rows: usize) -> Self {
        Self {
            ctbs: vec![false; columns * rows],
            decoded: vec![0; rows],
            deblocked: 0,
            finished: 0,
        }
    }

    /// Count CTB `ctb` (raster address) of row `row` decoded.
    ///
    /// # Errors
    ///
    /// If it was decoded before (overlapping slice segments).
    pub fn ctb_decoded(&mut self, ctb: usize, row: usize) -> Result<(), DecodeError> {
        if std::mem::replace(&mut self.ctbs[ctb], true) {
            return Err(DecodeError::Malformed("CTB decoded twice"));
        }
        self.decoded[row] += 1;
        Ok(())
    }

    /// Whether every CTB of the picture is decoded.
    pub fn complete(&self) -> bool {
        self.ctbs.iter().all(|&d| d)
    }
}

impl LoopFilter {
    /// Filter every band whose inputs are final; with `end`, the picture is
    /// complete and every remaining band is filtered.
    pub fn advance(&mut self, dsp: &Dsp, pps: &Pps, pic: &mut PictureState, end: bool) {
        let layout = pic.layout.clone();
        let rows = layout.ctb_h;
        let ctb = 1usize << layout.log2_ctb;
        let band = |row: usize| row * ctb..(row + 1) * ctb;
        let tiles_cross = pps.tiles.as_ref().is_none_or(|t| t.loop_filter_across);
        loop {
            let row = pic.progress.deblocked;
            if row == rows {
                break;
            }
            let decoded = |r: usize| pic.progress.decoded[r] as usize == layout.ctb_w;
            if !(end || (row + 1 < rows && decoded(row) && decoded(row + 1))) {
                break;
            }
            deblock::deblock_band(
                dsp,
                pic,
                [pps.cb_qp_offset, pps.cr_qp_offset],
                tiles_cross,
                row,
            );
            pic.progress.deblocked += 1;
        }
        loop {
            let row = pic.progress.finished;
            let deblocked = pic.progress.deblocked;
            if row == rows || !(row + 1 < deblocked || deblocked == rows) {
                break;
            }
            sao::apply_band(dsp, pic, tiles_cross, &mut self.sao, row);
            let lines = band(row);
            for (c, plane) in pic.samples.planes_mut().iter_mut().enumerate() {
                let shift = usize::from(c > 0);
                let end = (lines.end >> shift).min(plane.height());
                plane.extend_border_rows((lines.start >> shift).min(end)..end);
            }
            pic.progress.finished += 1;
        }
    }
}
