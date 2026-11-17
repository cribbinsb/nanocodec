//! The common copy schedule: specialize codec widths and move four rows
//! per iteration. ISA modules supply only exact-width memory operations.

#![allow(
    clippy::inline_always,
    reason = "inline the schedule into its ISA entry"
)]

pub(super) trait Ops {
    const COPY_BYTES: usize;
    const PREP_BYTES: usize;
    /// `AArch64` has enough integer registers to address four independent
    /// strided rows cheaply; x86/WASM use a smaller loop without their
    /// extra saved registers/locals. The row operations remain shared.
    const ROWS: usize = 4;
    /// ISA entry for one width. Keeping these out of the dispatcher's frame
    /// prevents the arbitrary-width fallback's register saves on tiny blocks.
    unsafe fn copy_width<const W: usize, const AVG: bool>(
        src: &[u8],
        ss: usize,
        dst: &mut [u8],
        ds: usize,
        w: usize,
        h: usize,
    );
    /// One fixed-width precision, with a runtime bias for centered domains.
    unsafe fn prep_width<const W: usize, const SHIFT: i32>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
        bias: i16,
    );
    /// The common unbiased precision, preserving a register-only call ABI.
    unsafe fn prep_unbiased_width<const W: usize>(
        src: &[u8],
        ss: usize,
        dst: &mut [i16],
        ds: usize,
        w: usize,
        h: usize,
    );
    /// Read and write exactly N bytes, optionally averaging with the output.
    unsafe fn copy<const N: usize, const AVG: bool>(src: *const u8, dst: *mut u8);
    /// Read N bytes and write N words, each `(sample << SHIFT) - bias`.
    unsafe fn prep<const N: usize, const SHIFT: i32>(src: *const u8, dst: *mut i16, bias: i16);
}

#[inline(always)]
unsafe fn copy_row<K: Ops, const AVG: bool>(src: *const u8, dst: *mut u8, w: usize) {
    let mut x = 0;
    // Large planes use four vectors per iteration; codec widths fold this
    // into fixed moves. Keep the remaining spans exact for unpadded tails.
    // SAFETY: each group lies wholly inside the validated row.
    unsafe {
        macro_rules! group {
            ($n:literal) => {
                if K::COPY_BYTES == $n {
                    while w - x >= 4 * $n {
                        for r in 0..4 {
                            K::copy::<$n, AVG>(src.add(x + r * $n), dst.add(x + r * $n));
                        }
                        x += 4 * $n;
                    }
                }
            };
        }
        group!(64);
        group!(32);
        group!(16);
    }
    // SAFETY: the caller provides w readable/writable samples. Every chunk
    // consumes exactly its width, including the final unpadded row.
    unsafe {
        macro_rules! chunk {
            ($n:literal) => {
                if K::COPY_BYTES >= $n {
                    while w - x >= $n {
                        K::copy::<$n, AVG>(src.add(x), dst.add(x));
                        x += $n;
                    }
                }
            };
        }
        chunk!(64);
        chunk!(32);
        chunk!(16);
        chunk!(8);
        chunk!(4);
        chunk!(2);
        chunk!(1);
    }
}

#[inline(always)]
unsafe fn prep_row<K: Ops, const SHIFT: i32>(src: *const u8, dst: *mut i16, w: usize, bias: i16) {
    let mut x = 0;
    // SAFETY: as for copy_row, with destination offsets measured in words.
    unsafe {
        macro_rules! chunk {
            ($n:literal) => {
                if K::PREP_BYTES >= $n {
                    while w - x >= $n {
                        K::prep::<$n, SHIFT>(src.add(x), dst.add(x), bias);
                        x += $n;
                    }
                }
            };
        }
        chunk!(32);
        chunk!(16);
        chunk!(8);
        chunk!(4);
        chunk!(2);
        chunk!(1);
    }
}

#[inline(always)]
pub(super) unsafe fn copy_width<K: Ops, const W: usize, const AVG: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    let w = if W == 0 { w } else { W };
    // SAFETY: the caller checked these rectangle invariants before calling
    // the kernel. Excluding overlapping rows removes irrelevant stride-one
    // vectorizer paths, particularly in small copies.
    unsafe {
        core::hint::assert_unchecked(ss >= w && ds >= w);
    }
    // Narrow blocks save more setup than four-row unrolling saves work.
    let rows = if W != 0 && W <= 8 { 1 } else { K::ROWS };
    let mut y = 0;
    // SAFETY: entry contracts cover h complete, nonoverlapping rows. Four
    // independent rows reduce loop bookkeeping without packing scalar loads.
    unsafe {
        while h - y >= rows {
            for r in 0..rows {
                copy_row::<K, AVG>(
                    src.as_ptr().add((y + r) * ss),
                    dst.as_mut_ptr().add((y + r) * ds),
                    w,
                );
            }
            y += rows;
        }
        while y < h {
            copy_row::<K, AVG>(src.as_ptr().add(y * ss), dst.as_mut_ptr().add(y * ds), w);
            y += 1;
        }
    }
}

#[inline(always)]
pub(super) unsafe fn copy<K: Ops, const AVG: bool>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    w: usize,
    h: usize,
) {
    if w == 0 || h == 0 {
        return;
    }
    // SAFETY: forwarded from the ISA entry; specializing width removes the
    // inner loop for small codec blocks and prevents per-row memcpy calls.
    unsafe {
        match w {
            4 => K::copy_width::<4, AVG>(src, ss, dst, ds, w, h),
            8 => K::copy_width::<8, AVG>(src, ss, dst, ds, w, h),
            16 => K::copy_width::<16, AVG>(src, ss, dst, ds, w, h),
            32 => K::copy_width::<32, AVG>(src, ss, dst, ds, w, h),
            64 => K::copy_width::<64, AVG>(src, ss, dst, ds, w, h),
            _ => K::copy_width::<0, AVG>(src, ss, dst, ds, w, h),
        }
    }
}

#[inline(always)]
pub(super) unsafe fn prep_width<K: Ops, const W: usize, const SHIFT: i32>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    w: usize,
    h: usize,
    bias: i16,
) {
    let w = if W == 0 { w } else { W };
    // SAFETY: the caller checked these rectangle invariants before calling
    // the kernel. Excluding overlapping rows removes irrelevant stride-one
    // vectorizer paths, particularly in small copies.
    unsafe {
        core::hint::assert_unchecked(ss >= w && ds >= w);
    }
    // Narrow blocks save more setup than four-row unrolling saves work.
    let rows = if W != 0 && W <= 8 { 1 } else { K::ROWS };
    let mut y = 0;
    // SAFETY: entry contracts cover every row and bound the scaled words.
    unsafe {
        while h - y >= rows {
            for r in 0..rows {
                prep_row::<K, SHIFT>(
                    src.as_ptr().add((y + r) * ss),
                    dst.as_mut_ptr().add((y + r) * ds),
                    w,
                    bias,
                );
            }
            y += rows;
        }
        while y < h {
            prep_row::<K, SHIFT>(
                src.as_ptr().add(y * ss),
                dst.as_mut_ptr().add(y * ds),
                w,
                bias,
            );
            y += 1;
        }
    }
}

#[inline(always)]
pub(super) unsafe fn prep<K: Ops, const SHIFT: i32>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    w: usize,
    h: usize,
    bias: i16,
) {
    if w == 0 || h == 0 {
        return;
    }
    // SAFETY: forwarded from the ISA entry, with word strides for dst.
    unsafe {
        match w {
            4 => K::prep_width::<4, SHIFT>(src, ss, dst, ds, w, h, bias),
            8 => K::prep_width::<8, SHIFT>(src, ss, dst, ds, w, h, bias),
            16 => K::prep_width::<16, SHIFT>(src, ss, dst, ds, w, h, bias),
            32 => K::prep_width::<32, SHIFT>(src, ss, dst, ds, w, h, bias),
            64 => K::prep_width::<64, SHIFT>(src, ss, dst, ds, w, h, bias),
            _ => K::prep_width::<0, SHIFT>(src, ss, dst, ds, w, h, bias),
        }
    }
}

/// Keep immediate shifts in the vector operations; precision is selected
/// once per block, never once per sample or vector.
macro_rules! prep_entry {
    ($ops:ty, $src:expr, $ss:expr, $dst:expr, $ds:expr, $w:expr, $h:expr, $shift:expr, $bias:expr) => {{
        match $shift {
            0 => super::rows::prep::<$ops, 0>($src, $ss, $dst, $ds, $w, $h, $bias),
            1 => super::rows::prep::<$ops, 1>($src, $ss, $dst, $ds, $w, $h, $bias),
            2 => super::rows::prep::<$ops, 2>($src, $ss, $dst, $ds, $w, $h, $bias),
            3 => super::rows::prep::<$ops, 3>($src, $ss, $dst, $ds, $w, $h, $bias),
            4 => super::rows::prep::<$ops, 4>($src, $ss, $dst, $ds, $w, $h, $bias),
            5 => super::rows::prep::<$ops, 5>($src, $ss, $dst, $ds, $w, $h, $bias),
            6 => super::rows::prep::<$ops, 6>($src, $ss, $dst, $ds, $w, $h, $bias),
            7 => super::rows::prep::<$ops, 7>($src, $ss, $dst, $ds, $w, $h, $bias),
            _ => panic!("block prep shift exceeds seven"),
        }
    }};
}
pub(super) use prep_entry;

/// Width selection alone stays in the common caller's small frame.
#[inline(always)]
pub(super) unsafe fn prep_unbiased<K: Ops>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    w: usize,
    h: usize,
) {
    if w == 0 || h == 0 {
        return;
    }
    // SAFETY: the supported ISA entry forwards its validated rectangle.
    unsafe {
        match w {
            4 => K::prep_unbiased_width::<4>(src, ss, dst, ds, w, h),
            8 => K::prep_unbiased_width::<8>(src, ss, dst, ds, w, h),
            16 => K::prep_unbiased_width::<16>(src, ss, dst, ds, w, h),
            32 => K::prep_unbiased_width::<32>(src, ss, dst, ds, w, h),
            64 => K::prep_unbiased_width::<64>(src, ss, dst, ds, w, h),
            _ => K::prep_unbiased_width::<0>(src, ss, dst, ds, w, h),
        }
    }
}
