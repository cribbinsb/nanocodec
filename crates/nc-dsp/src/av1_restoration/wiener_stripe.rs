//! The Wiener filter of one 8-bit stripe (AV1 §7.17.4): the ISA-neutral
//! schedule of the complete-stripe kernels ([`super::avx2`],
//! [`super::neon`]), after dav1d 1.5.4's stripe loop
//! (`src/x86/looprestoration_avx2.asm` `wiener_filter7_8bpc`,
//! `src/arm/64/looprestoration.S` `wiener_filter7_8bpc_neon`;
//! BSD-2-Clause, see `THIRD_PARTY.md`): the taps are prepared once, the
//! seven horizontally filtered rows live in a ring whose pointers rotate
//! (no per-row indirect call or modulo), and an axis whose outer tap is
//! zero runs a five-tap instance (dav1d's `wiener_filter5`). Each ISA
//! supplies the row arithmetic through [`WienerLanes`].

#![allow(
    clippy::many_single_char_names,
    reason = "the spec's and dav1d's sample and row names"
)]
#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use nc_common::PlaneViewMut;

use super::kernels::WIENER_ROWS;
use super::wiener::WienerTaps;
use super::{AV1_LR_BORDER, AV1_LR_OVERREAD};

/// One ISA's Wiener row arithmetic, with the row kernels' rounding and
/// clamps (`super::wiener`).
///
/// # Safety
///
/// Every method executes the ISA's instructions: callers run inside the
/// ISA's `target_feature` entry point.
pub(super) trait WienerLanes {
    /// The horizontal taps, prepared.
    type Horizontal: Copy;
    /// The vertical taps, prepared.
    type Vertical: Copy;
    /// Outputs per horizontal step; one step reads `H_STEP + 8` bytes. At
    /// most [`AV1_LR_OVERREAD`].
    const H_STEP: usize;
    /// Outputs per vertical step (values read from every row). At most
    /// [`AV1_LR_OVERREAD`].
    const V_STEP: usize;

    unsafe fn horizontal(taps: WienerTaps) -> Self::Horizontal;
    unsafe fn vertical(taps: WienerTaps) -> Self::Vertical;
    /// `blocks` horizontal steps of the row at `src` into `out` (outer taps
    /// skipped when `FIVE`).
    unsafe fn h_row<const FIVE: bool>(
        taps: Self::Horizontal,
        src: *const u8,
        out: *mut i16,
        blocks: usize,
    );
    /// `n` output bytes at `dst` from the seven rows (rows 0 and 6 not read
    /// when `FIVE`); the last step's surplus is not written.
    unsafe fn v_row<const FIVE: bool>(
        rows: &[*const i16; WIENER_ROWS],
        taps: Self::Vertical,
        dst: *mut u8,
        n: usize,
    );
}

/// Wiener-filter a `w`x`h` region of a stripe window into `dst` at
/// `(x, y)` with the ISA's lanes `V`, choosing the five-tap form per axis
/// whose outer tap is zero (luma may have only one such axis).
///
/// # Safety
///
/// `V`'s ISA must be enabled in the caller. Extents are checked here.
#[inline(always)]
pub(super) unsafe fn run<V: WienerLanes>(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    source: (&[u8], usize),
    size: (usize, usize),
    ring: &mut [i16],
    output: (&mut PlaneViewMut<'_>, usize, usize),
) {
    // SAFETY: the caller's contract; each instance checks all spans.
    unsafe {
        match (horizontal[0] == 0, vertical[0] == 0) {
            (false, false) => {
                stripe::<V, false, false>(horizontal, vertical, source, size, ring, output);
            }
            (true, false) => {
                stripe::<V, true, false>(horizontal, vertical, source, size, ring, output);
            }
            (false, true) => {
                stripe::<V, false, true>(horizontal, vertical, source, size, ring, output);
            }
            (true, true) => {
                stripe::<V, true, true>(horizontal, vertical, source, size, ring, output);
            }
        }
    }
}

/// [`run`] for one choice of tap counts.
#[inline(always)]
unsafe fn stripe<V: WienerLanes, const H5: bool, const V5: bool>(
    horizontal: WienerTaps,
    vertical: WienerTaps,
    (window, stride): (&[u8], usize),
    (w, h): (usize, usize),
    ring: &mut [i16],
    (dst, x, y): (&mut PlaneViewMut<'_>, usize, usize),
) {
    assert!(x <= dst.width() && w <= dst.width() - x);
    assert!(y <= dst.height() && h <= dst.height() - y);
    assert!(V::H_STEP <= AV1_LR_OVERREAD && V::V_STEP <= AV1_LR_OVERREAD);
    if w == 0 || h == 0 {
        return;
    }
    let row_len = w.checked_add(AV1_LR_OVERREAD).expect("Wiener row extent");
    let source_len = row_len
        .checked_add(2 * AV1_LR_BORDER)
        .expect("Wiener source row");
    let source_rows = h
        .checked_add(2 * AV1_LR_BORDER)
        .expect("Wiener source rows");
    let source_end = (source_rows - 1)
        .checked_mul(stride)
        .and_then(|offset| offset.checked_add(source_len))
        .expect("Wiener source extent");
    assert!(window.len() >= source_end);
    assert!(
        ring.len()
            >= WIENER_ROWS
                .checked_mul(row_len)
                .expect("Wiener ring extent")
    );
    let dst_stride = dst.stride();
    // SAFETY: checked spans cover every window/ring row and output rectangle.
    // Horizontal writes ceil(w/H_STEP)·H_STEP values and reads 8 bytes
    // more, vertical reads ceil(w/V_STEP)·V_STEP, all within the overread.
    // The seven scratch rows are distinct and horizontal completes before
    // vertical reads; pointers rotate among them.
    unsafe {
        let horizontal = V::horizontal(horizontal);
        let vertical = V::vertical(vertical);
        let blocks = w.div_ceil(V::H_STEP);
        let start = ring.as_mut_ptr();
        let mut rows: [*const i16; WIENER_ROWS] =
            core::array::from_fn(|k| start.add(k * row_len).cast_const());
        for (k, &row) in rows[..WIENER_ROWS - 1].iter().enumerate() {
            V::h_row::<H5>(
                horizontal,
                window.as_ptr().add(k * stride),
                row.cast_mut(),
                blocks,
            );
        }
        let output = dst.data_mut().as_mut_ptr().add(y * dst_stride + x);
        for r in 0..h {
            V::h_row::<H5>(
                horizontal,
                window.as_ptr().add((r + WIENER_ROWS - 1) * stride),
                rows[WIENER_ROWS - 1].cast_mut(),
                blocks,
            );
            V::v_row::<V5>(&rows, vertical, output.add(r * dst_stride), w);
            let [a, b, c, d, e, f, g] = rows;
            rows = [b, c, d, e, f, g, a];
        }
    }
}
