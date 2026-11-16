//! The statistics of the engine's directional intra estimate
//! (`nc_encoder::intra_estimate`): over every other row and column of an
//! `N`-square source block, the sum and sum of squares of the samples and
//! the squared error of four open-loop predictions from the source edge
//! around it (vertical, horizontal, 45 degrees down-left, 45 degrees
//! down-right). Every prediction row is one run of the edge array, so each
//! sum is a byte-difference square accumulated in 32 bits.
//!
//! The edge is the engine's: `edge[N - 1 - r]` the left neighbour of row
//! `r`, `edge[N]` the top-left corner, `edge[N + 1 + c]` the sample above
//! column `c` (`c < 2N`). Sums fit 32 bits: at most `32 · 32` samples of
//! squares up to `255²`.
//!
//! Every ISA takes the even samples of a row without a gather: x86 and
//! SIMD128 mask the low byte of each 16-bit lane, NEON unzips the even
//! bytes; the squared differences accumulate by `vpmaddwd` (x86),
//! `vabd`/`umull`/`uadalp` (NEON) or `dot` (SIMD128).

#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
mod avx2;
#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
mod simd128;

use crate::Isa;

/// Largest block side.
pub const ESTIMATE_MAX_SIDE: usize = 64;
/// Samples of the edge array (the left column, the corner, then twice the
/// side above).
pub const ESTIMATE_EDGE_LEN: usize = 4 * ESTIMATE_MAX_SIDE + 1;
/// Rows and columns measured: every `SAMPLE_STEP`th.
const SAMPLE_STEP: usize = 2;
/// Block sides the kernels take.
pub const ESTIMATE_SIDES: [usize; 5] = [4, 8, 16, 32, 64];

/// The sums of one block (module docs).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EstimateStats {
    /// Sum of the measured samples.
    pub sum: u32,
    /// Sum of their squares.
    pub squares: u32,
    /// Squared error of the vertical, horizontal, down-left and down-right
    /// predictions, in that order.
    pub sse: [u32; 4],
}

/// The statistics of the `n`-square block at the start of `src` (rows
/// `stride` apart) against `edge`. `unsafe`: it may run its ISA's
/// instructions, so only a table built for a supported ISA calls it.
pub(crate) type StatsFn =
    unsafe fn(src: &[u8], stride: usize, n: usize, edge: &[u8]) -> EstimateStats;

/// The scalar reference.
pub(crate) fn scalar(src: &[u8], stride: usize, n: usize, edge: &[u8]) -> EstimateStats {
    let mut stats = EstimateStats::default();
    let square = |a: u8, b: u8| {
        let d = u32::from(a.abs_diff(b));
        d * d
    };
    for r in (0..n).step_by(SAMPLE_STEP) {
        let row = &src[r * stride..r * stride + n];
        let left = edge[n - 1 - r];
        for c in (0..n).step_by(SAMPLE_STEP) {
            let v = row[c];
            stats.sum += u32::from(v);
            stats.squares += u32::from(v) * u32::from(v);
            stats.sse[0] += square(v, edge[n + 1 + c]);
            stats.sse[1] += square(v, left);
            stats.sse[2] += square(v, edge[n + 2 + r + c]);
            stats.sse[3] += square(v, edge[n - r + c]);
        }
    }
    stats
}

/// The kernel of `isa`.
pub(crate) fn kernel(isa: Isa) -> StatsFn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => avx2::stats,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => neon::stats,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd128::stats,
        _ => scalar,
    }
}
