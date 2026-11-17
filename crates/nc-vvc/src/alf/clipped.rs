//! Decoder nonlinear luma banks using the shared ALF classifier and boundary rows.
use super::{Analysis, BLOCK, CtbAlf, CtbFilter, TAPS, VVC_ALF_REACH, VVC_ALF_TRANSPOSES};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Pixel;

pub(crate) struct ClippedFilters {
    pub classes: [u8; 25],
    pub coefficients: Vec<[i16; TAPS]>,
    pub limits: Vec<[i16; TAPS]>,
}
impl<T: Pixel> Analysis<'_, T> {
    pub(crate) fn apply_clipped(
        &self,
        picture: &mut Picture<T>,
        decisions: &[CtbAlf],
        bank: &ClippedFilters,
    ) {
        if !decisions.contains(&Some(CtbFilter::Signalled)) {
            return;
        }
        let transposed: [[([i16; TAPS], [i16; TAPS]); 4]; 25] = std::array::from_fn(|class| {
            let index = usize::from(bank.classes[class]);
            let coeff = bank.coefficients[index];
            let lim = bank.limits[index];
            VVC_ALF_TRANSPOSES.map(|order| (order.map(|i| coeff[i]), order.map(|i| lim[i])))
        });
        let mut output = picture.plane_mut(PlaneIndex::Y).view_mut();
        let mut coefficients = vec![[0; TAPS]; self.block_columns];
        let mut limits = vec![[0; TAPS]; self.block_columns];
        let stride = output.stride();
        for ((x0, y0, width, height), decision) in self.runs(decisions) {
            if decision != Some(CtbFilter::Signalled) {
                continue;
            }
            let blocks = width.div_ceil(BLOCK);
            let x_blk = (x0 - self.origin.0) / BLOCK;
            let mut y = y0;
            while y < y0 + height {
                if y.is_multiple_of(BLOCK) {
                    let row_base = (y - self.origin.1) / BLOCK * self.block_columns + x_blk;
                    for ((c_dst, l_dst), &[cls, tr]) in coefficients[..blocks]
                        .iter_mut()
                        .zip(&mut limits[..blocks])
                        .zip(&self.classes[row_base..row_base + blocks])
                    {
                        let (coeff, lim) = transposed[usize::from(cls)][usize::from(tr)];
                        *c_dst = coeff;
                        *l_dst = lim;
                    }
                    if y + BLOCK <= y0 + height {
                        let col = self.column(x0, VVC_ALF_REACH);
                        let mut shifts = [0u32; BLOCK];
                        let rows: [[&[T]; 7]; BLOCK] = std::array::from_fn(|r| {
                            let (tap_r, shift) = self.tap_rows(y + r);
                            shifts[r] = shift;
                            tap_r.map(|row| &row[col..])
                        });
                        let mut lines = output.data_mut()[y * stride..]
                            .chunks_mut(stride)
                            .map(|line| &mut line[x0..x0 + width]);
                        let mut out: [&mut [T]; BLOCK] = std::array::from_fn(|_| {
                            lines.next().expect("four rows in the picture")
                        });
                        self.kernels().vvc_alf_filter_clipped_block_row(
                            &rows,
                            &coefficients[..blocks],
                            &limits[..blocks],
                            &shifts,
                            &mut out,
                        );
                        y += BLOCK;
                        continue;
                    }
                }
                let (rows, shift) = self.tap_rows(y);
                let rows = rows.map(|r| &r[self.column(x0, VVC_ALF_REACH)..]);
                self.kernels().vvc_alf_filter_clipped_row(
                    &rows,
                    &coefficients[..blocks],
                    &limits[..blocks],
                    shift,
                    &mut output.row_mut(y)[x0..x0 + width],
                );
                y += 1;
            }
        }
    }
}
