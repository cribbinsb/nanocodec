//! The row kernels of loop restoration, one table per [`crate::Dsp`]:
//! the Wiener horizontal and vertical rows ([`super::wiener`]) and the
//! self-guided box sums, box statistics, filtered rows and projection
//! ([`super::self_guided`]), with safe wrappers that check every buffer
//! against the kernel contract before calling the ISA's version.
//!
//! # Kernel contract
//!
//! Every kernel works on rows of `n` values and may read and write up to
//! [`AV1_LR_OVERREAD`] values past them in every buffer except the final
//! 8-bit destinations (`wiener_v`, `sgr_blend`), which it writes exactly.
//! SIMD kernels index their slices (a wrong length panics, it never reads
//! out of bounds); they are `unsafe` only because they execute their
//! ISA's instructions, so they are reachable only through a table built
//! after [`crate::Isa::is_supported`].

#![allow(unsafe_code)]

use crate::Isa;
use nc_common::PlaneViewMut;

use super::AV1_LR_OVERREAD;
use super::self_guided::{blend, box_ab, box_sums, filter_around, filter_between, filter_on};
use super::wiener::{WienerTaps, h_row, v_row};

/// Complete stripe kernel with the padded-window and exact-output contract.
/// The kernel validates all extents before raw SIMD access.
pub(crate) type WienerStripeFn = unsafe fn(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    window: (&[u8], usize),
    size: (usize, usize),
    ring: &mut [i16],
    dst: (&mut PlaneViewMut<'_>, usize, usize),
);
/// Wiener rows the vertical filter reads.
pub(crate) const WIENER_ROWS: usize = 7;

/// `out[..n]` from `src[..n + 6]` (the Wiener horizontal pass).
pub(crate) type WienerHFn = unsafe fn(src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize);
/// `dst` from the same columns of seven intermediate rows.
pub(crate) type WienerVFn =
    unsafe fn(rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u8]);
/// Horizontal box sums and sums of squares of one radius.
pub(crate) type BoxSumsFn = unsafe fn(src: &[u8], sum: &mut [i32], squares: &mut [i32], n: usize);
/// Box statistics `A`, `B` from `N` rows of horizontal sums.
pub(crate) type BoxAbFn<const N: usize> = unsafe fn(
    sums: [&[i32]; N],
    squares: [&[i32]; N],
    scale: u32,
    ab: (&mut [i32], &mut [i32]),
    n: usize,
);
/// A radius-2 output row between two box rows.
pub(crate) type BetweenFn =
    unsafe fn(a: [&[i32]; 2], b: [&[i32]; 2], src: &[u8], out: &mut [i32], n: usize);
/// A radius-2 output row on a box row.
pub(crate) type OnFn = unsafe fn(a: &[i32], b: &[i32], src: &[u8], out: &mut [i32], n: usize);
/// A radius-1 output row.
pub(crate) type AroundFn =
    unsafe fn(a: [&[i32]; 3], b: [&[i32]; 3], src: &[u8], out: &mut [i32], n: usize);
/// The projection of the filtered rows onto `dst`.
pub(crate) type BlendFn =
    unsafe fn(src: &[u8], filtered: [&[i32]; 2], weights: [i32; 2], dst: &mut [u8]);

/// One ISA's loop restoration kernels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    pub(crate) wiener_stripe: Option<WienerStripeFn>,
    pub(crate) wiener_h: WienerHFn,
    pub(crate) wiener_v: WienerVFn,
    pub(crate) box_sums3: BoxSumsFn,
    pub(crate) box_sums5: BoxSumsFn,
    pub(crate) box_ab3: BoxAbFn<3>,
    pub(crate) box_ab5: BoxAbFn<5>,
    pub(crate) between: BetweenFn,
    pub(crate) on: OnFn,
    pub(crate) around: AroundFn,
    pub(crate) blend: BlendFn,
}

/// The scalar references.
pub(crate) const SCALAR: Kernels = Kernels {
    wiener_stripe: None,
    wiener_h: h_row,
    wiener_v: v_row,
    box_sums3: box_sums::<1>,
    box_sums5: box_sums::<2>,
    box_ab3: box_ab::<3>,
    box_ab5: box_ab::<5>,
    between: filter_between,
    on: filter_on,
    around: filter_around,
    blend,
};

/// The kernels of `isa` (AVX-512 tables inherit AVX2's).
pub(crate) fn kernels(isa: Isa) -> Kernels {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 | Isa::Avx512 => super::avx2::KERNELS,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => super::neon::KERNELS,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => super::simd128::KERNELS,
        _ => SCALAR,
    }
}

/// Whether every row holds `n` values and the overread.
fn rows_cover(rows: &[&[i32]], n: usize) -> bool {
    rows.iter().all(|row| row.len() >= n + AV1_LR_OVERREAD)
}

impl Kernels {
    /// Dispatch a complete stripe when this ISA has a prepared kernel.
    pub(super) fn wiener_stripe(
        &self,
        taps: (WienerTaps, WienerTaps),
        window: (&[u8], usize),
        size: (usize, usize),
        ring: &mut [i16],
        dst: (&mut PlaneViewMut<'_>, usize, usize),
    ) -> bool {
        let Some(kernel) = self.wiener_stripe else {
            return false;
        };
        // SAFETY: the table is selected for a supported ISA. The complete
        // stripe kernel checks window, ring and destination extents.
        unsafe { kernel(taps.0, taps.1, window, size, ring, dst) };
        true
    }

    /// The Wiener horizontal row: `out[..n]` from `src[..n + 6]`.
    pub(crate) fn wiener_h(&self, src: &[u8], taps: WienerTaps, out: &mut [i16], n: usize) {
        assert!(src.len() >= n + 6 + AV1_LR_OVERREAD && out.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.wiener_h)(src, taps, out, n) }
    }

    /// The Wiener vertical row into all of `dst`.
    pub(crate) fn wiener_v(&self, rows: &[&[i16]; WIENER_ROWS], taps: WienerTaps, dst: &mut [u8]) {
        assert!(rows.iter().all(|r| r.len() >= dst.len() + AV1_LR_OVERREAD));
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.wiener_v)(rows, taps, dst) }
    }

    /// Horizontal box sums of `radius` (1 or 2) for `n` box columns.
    pub(crate) fn sgr_box_sums(
        &self,
        radius: usize,
        src: &[u8],
        sum: &mut [i32],
        squares: &mut [i32],
        n: usize,
    ) {
        assert!(src.len() >= n + 2 * radius + AV1_LR_OVERREAD);
        assert!(sum.len() >= n + AV1_LR_OVERREAD && squares.len() >= n + AV1_LR_OVERREAD);
        let kernel = match radius {
            1 => self.box_sums3,
            2 => self.box_sums5,
            _ => unreachable!("self-guided radii are 1 and 2"),
        };
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { kernel(src, sum, squares, n) }
    }

    /// Radius-1 box statistics of `n` columns.
    pub(crate) fn sgr_box_ab3(
        &self,
        sums: [&[i32]; 3],
        squares: [&[i32]; 3],
        scale: u32,
        (a, b): (&mut [i32], &mut [i32]),
        n: usize,
    ) {
        assert!(rows_cover(&sums, n) && rows_cover(&squares, n));
        assert!(a.len() >= n + AV1_LR_OVERREAD && b.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.box_ab3)(sums, squares, scale, (a, b), n) }
    }

    /// Radius-2 box statistics of `n` columns.
    pub(crate) fn sgr_box_ab5(
        &self,
        sums: [&[i32]; 5],
        squares: [&[i32]; 5],
        scale: u32,
        (a, b): (&mut [i32], &mut [i32]),
        n: usize,
    ) {
        assert!(rows_cover(&sums, n) && rows_cover(&squares, n));
        assert!(a.len() >= n + AV1_LR_OVERREAD && b.len() >= n + AV1_LR_OVERREAD);
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.box_ab5)(sums, squares, scale, (a, b), n) }
    }

    /// Checks shared by the filtered-row kernels.
    fn check_filter(a: &[&[i32]], b: &[&[i32]], src: &[u8], out: &[i32], n: usize) {
        assert!(rows_cover(a, n + 2) && rows_cover(b, n + 2));
        assert!(src.len() >= n + AV1_LR_OVERREAD && out.len() >= n + AV1_LR_OVERREAD);
    }

    /// A radius-2 output row between box rows.
    pub(crate) fn sgr_filter_between(
        &self,
        a: [&[i32]; 2],
        b: [&[i32]; 2],
        src: &[u8],
        out: &mut [i32],
        n: usize,
    ) {
        Self::check_filter(&a, &b, src, out, n);
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.between)(a, b, src, out, n) }
    }

    /// A radius-2 output row on a box row.
    pub(crate) fn sgr_filter_on(
        &self,
        a: &[i32],
        b: &[i32],
        src: &[u8],
        out: &mut [i32],
        n: usize,
    ) {
        Self::check_filter(&[a], &[b], src, out, n);
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.on)(a, b, src, out, n) }
    }

    /// A radius-1 output row.
    pub(crate) fn sgr_filter_around(
        &self,
        a: [&[i32]; 3],
        b: [&[i32]; 3],
        src: &[u8],
        out: &mut [i32],
        n: usize,
    ) {
        Self::check_filter(&a, &b, src, out, n);
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.around)(a, b, src, out, n) }
    }

    /// The projection onto all of `dst`.
    pub(crate) fn sgr_blend(
        &self,
        src: &[u8],
        filtered: [&[i32]; 2],
        weights: [i32; 2],
        dst: &mut [u8],
    ) {
        let n = dst.len();
        assert!(src.len() >= n + AV1_LR_OVERREAD && rows_cover(&filtered, n));
        // SAFETY: the table was built for a supported ISA; lengths checked.
        unsafe { (self.blend)(src, filtered, weights, dst) }
    }
}
