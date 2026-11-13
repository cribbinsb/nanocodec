//! Checks `nc_dsp::av1_inverse_nonzero_rows`, the row bound the inverse
//! transform takes from the end of block, against this crate's scan tables:
//! for every transform size, scan class (2-D zig-zag, `H_*` columns, `V_*`
//! rows) and end of block, every position of the scan prefix must lie in a
//! row below the bound. The decoder stores coefficients column-major
//! (position `c * coded_height + r`), as the inverse reads them.

use nc_dsp::{Av1TxSize, Av1TxType, av1_inverse_nonzero_rows};

use crate::tables::generated::{
    DEFAULT_SCAN_4X4, DEFAULT_SCAN_4X8, DEFAULT_SCAN_4X16, DEFAULT_SCAN_8X4, DEFAULT_SCAN_8X8,
    DEFAULT_SCAN_8X16, DEFAULT_SCAN_8X32, DEFAULT_SCAN_16X4, DEFAULT_SCAN_16X8, DEFAULT_SCAN_16X16,
    DEFAULT_SCAN_16X32, DEFAULT_SCAN_32X8, DEFAULT_SCAN_32X16, DEFAULT_SCAN_32X32,
};

/// The 2-D scan of a coded `width` x `height` area (64 codes as 32).
fn default_scan(width: usize, height: usize) -> &'static [u16] {
    match (width, height) {
        (4, 4) => &DEFAULT_SCAN_4X4,
        (8, 8) => &DEFAULT_SCAN_8X8,
        (16, 16) => &DEFAULT_SCAN_16X16,
        (4, 8) => &DEFAULT_SCAN_4X8,
        (8, 4) => &DEFAULT_SCAN_8X4,
        (8, 16) => &DEFAULT_SCAN_8X16,
        (16, 8) => &DEFAULT_SCAN_16X8,
        (16, 32) => &DEFAULT_SCAN_16X32,
        (32, 16) => &DEFAULT_SCAN_32X16,
        (4, 16) => &DEFAULT_SCAN_4X16,
        (16, 4) => &DEFAULT_SCAN_16X4,
        (8, 32) => &DEFAULT_SCAN_8X32,
        (32, 8) => &DEFAULT_SCAN_32X8,
        _ => &DEFAULT_SCAN_32X32,
    }
}

/// The storage positions of `kind`'s scan over `tx`'s coded area.
fn scan_positions(tx: Av1TxSize, kind: Av1TxType) -> Vec<usize> {
    use Av1TxType::{HAdst, HDct, HFlipadst, VAdst, VDct, VFlipadst};
    let (w, h) = (tx.coded_width(), tx.coded_height());
    match kind {
        // Column by column: storage order.
        HDct | HAdst | HFlipadst => (0..w * h).collect(),
        // Row by row.
        VDct | VAdst | VFlipadst => (0..h)
            .flat_map(|r| (0..w).map(move |c| c * h + r))
            .collect(),
        _ => default_scan(w, h).iter().map(|&p| usize::from(p)).collect(),
    }
}

#[test]
fn row_bound_covers_every_scan_prefix() {
    for tx in Av1TxSize::ALL {
        let h = tx.coded_height();
        for kind in Av1TxType::ALL.into_iter().filter(|kind| kind.fits(tx)) {
            let scan = scan_positions(tx, kind);
            assert_eq!(scan.len(), tx.coded_area(), "{tx:?} {kind:?} scan length");
            let mut rows_used = 0;
            for (eob, &position) in scan.iter().enumerate().map(|(i, p)| (i + 1, p)) {
                rows_used = rows_used.max(position % h + 1);
                let bound = av1_inverse_nonzero_rows(tx, kind, eob);
                assert!(
                    rows_used <= bound && bound <= h,
                    "{tx:?} {kind:?} eob {eob}: rows used {rows_used}, bound {bound}"
                );
            }
        }
    }
}
