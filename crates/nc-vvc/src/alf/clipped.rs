//! Decoder nonlinear luma banks using the shared ALF classifier and boundary rows.
use super::{Analysis, BLOCK, CtbAlf, CtbFilter, PAD, TAPS, VVC_ALF_REACH, VVC_ALF_TRANSPOSES};
use nc_common::{Picture, PlaneIndex};

pub(crate) struct ClippedFilters {
    pub classes: [u8; 25],
    pub coefficients: Vec<[i16; TAPS]>,
    pub limits: Vec<[i16; TAPS]>,
}
impl Analysis<'_> {
    pub(crate) fn apply_clipped(
        &self,
        picture: &mut Picture,
        decisions: &[CtbAlf],
        bank: &ClippedFilters,
    ) {
        let mut output = picture.plane_mut(PlaneIndex::Y).view_mut();
        let mut coefficients = vec![[0; TAPS]; self.block_columns];
        let mut limits = vec![[0; TAPS]; self.block_columns];
        for ((x0, y0, width, height), decision) in self.runs(decisions) {
            if decision != Some(CtbFilter::Signalled) {
                continue;
            }
            let blocks = width.div_ceil(BLOCK);
            for y in y0..y0 + height {
                if y.is_multiple_of(BLOCK) {
                    for block in 0..blocks {
                        let class = self.class(x0 + block * BLOCK, y);
                        let index = usize::from(bank.classes[usize::from(class.index)]);
                        let order = VVC_ALF_TRANSPOSES[usize::from(class.transpose)];
                        coefficients[block] = order.map(|i| bank.coefficients[index][i]);
                        limits[block] = order.map(|i| bank.limits[index][i]);
                    }
                }
                let (rows, shift) = self.tap_rows(y);
                let rows = rows.map(|r| &r[x0 + PAD - VVC_ALF_REACH..]);
                self.dsp.vvc_alf_filter_clipped_row(
                    &rows,
                    &coefficients[..blocks],
                    &limits[..blocks],
                    shift,
                    &mut output.row_mut(y)[x0..x0 + width],
                );
            }
        }
    }
}
