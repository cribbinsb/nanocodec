//! H.264 8x8 inverse transform plus prediction add (§8.5.13.2) of a
//! decoder's already-scaled coefficients, in 16-bit lanes: one vector per
//! row of eight, twice the lanes of the 32-bit form.
//!
//! # Why 16 bits is exact
//!
//! Every value the butterflies of eqs. 8-355..8-370 name is bounded to
//! `-2^15..2^15 - 1` for 8-bit video by bitstream conformance (§8.5.13.2:
//! "the values of ... shall be in the range of −2^(7 + bitDepth) to
//! 2^(7 + bitDepth) − 1"), and so are the scaled inputs (§8.5.13.1, clamped
//! there by the decoder). The steps are additions, subtractions and
//! arithmetic shifts of named values only, so wrapping 16-bit arithmetic
//! computes them exactly: a partial sum may wrap, but its named result is
//! the same modulo 2^16 and in range. The final `(h + 32) >> 6` would wrap
//! for `h` near `2^15`; it is computed as the identical `((h >> 5) + 1) >> 1`.
//!
//! A non-conforming stream gets different but defined pictures (wrapping,
//! never a panic), as with openh264's and ffmpeg's 16-bit 8-bit-depth paths;
//! the residual added stays within `±2^9`, so the prediction add cannot wrap.
//!
//! The scalar reference [`add_scaled_8x8`] is the same arithmetic on plain
//! wrapping `i16`; every ISA runs [`add_scaled_8x8_lanes`] on its eight-lane
//! vector.

// The lane helpers must inline into each ISA's `#[target_feature]` entry.
#![allow(clippy::inline_always)]

use crate::simd::I16x8Shuffles;

/// Side of the block.
const N: usize = 8;
/// The final rounding `(h + 32) >> 6` as two shifts: `h >> FIRST_SHIFT`,
/// plus one, then one more shift.
const FIRST_SHIFT: u32 = 5;

/// The 8-point inverse (eqs. 8-355..8-370) over values of any lane type:
/// `d[k]` holds input `k` of every line, the result output `k`.
#[inline(always)]
fn idct8<T: Copy>(
    d: [T; N],
    add: impl Fn(T, T) -> T,
    sub: impl Fn(T, T) -> T,
    sra: impl Fn(T, u32) -> T,
) -> [T; N] {
    let a0 = add(d[0], d[4]);
    let a4 = sub(d[0], d[4]);
    let a2 = sub(sra(d[2], 1), d[6]);
    let a6 = add(d[2], sra(d[6], 1));
    let (b0, b2, b4, b6) = (add(a0, a6), add(a4, a2), sub(a4, a2), sub(a0, a6));
    let a1 = sub(sub(sub(d[5], d[3]), d[7]), sra(d[7], 1));
    let a3 = sub(sub(add(d[1], d[7]), d[3]), sra(d[3], 1));
    let a5 = add(add(sub(d[7], d[1]), d[5]), sra(d[5], 1));
    let a7 = add(add(add(d[3], d[5]), d[1]), sra(d[1], 1));
    let b1 = add(a1, sra(a7, 2));
    let b7 = sub(a7, sra(a1, 2));
    let b3 = add(a3, sra(a5, 2));
    let b5 = sub(sra(a3, 2), a5);
    [
        add(b0, b7),
        add(b2, b5),
        add(b4, b3),
        add(b6, b1),
        sub(b6, b1),
        sub(b4, b3),
        sub(b2, b5),
        sub(b0, b7),
    ]
}

/// `(h + 32) >> 6` without the 16-bit overflow of `h + 32`.
#[inline(always)]
fn round6(h: i16) -> i16 {
    ((h >> FIRST_SHIFT) + 1) >> 1
}

/// Whether the scaled block has no AC coefficient; if so its DC's constant
/// inverse `(d + 32) >> 6` (every output of a DC-only §8.5.13.2 transform).
#[inline(always)]
fn dc_only(coeffs: &[i16; 64]) -> Option<i16> {
    (coeffs[1..].iter().fold(0, |any, &c| any | c) == 0).then(|| round6(coeffs[0]))
}

/// Scalar reference: inverse transform the scaled 8x8 `coeffs` (raster)
/// and add it to the 8x8 block of `dst` (rows `stride` apart), clamped to
/// 8 bits.
///
/// # Safety
///
/// None beyond the kernel table's signature (`dst` covers the block).
pub(crate) unsafe fn add_scaled_8x8(coeffs: &[i16; 64], dst: &mut [u8], stride: usize) {
    let residual = if let Some(dc) = dc_only(coeffs) {
        [[dc; N]; N]
    } else {
        let lines = |rows: [[i16; N]; N]| -> [[i16; N]; N] {
            rows.map(|row| idct8(row, i16::wrapping_add, i16::wrapping_sub, |x, s| x >> s))
        };
        let transpose = |m: [[i16; N]; N]| -> [[i16; N]; N] {
            core::array::from_fn(|r| core::array::from_fn(|c| m[c][r]))
        };
        let rows: [[i16; N]; N] =
            core::array::from_fn(|r| core::array::from_fn(|c| coeffs[r * N + c]));
        transpose(lines(transpose(lines(rows)))).map(|line| line.map(round6))
    };
    for (r, line) in residual.iter().enumerate() {
        for (sample, &h) in dst[r * stride..][..N].iter_mut().zip(line) {
            *sample = (i16::from(*sample) + h).clamp(0, 255) as u8;
        }
    }
}

/// [`add_scaled_8x8`] on eight-lane vectors: rows are loaded, transposed so
/// that one 1-D pass transforms all rows at once, transposed back for the
/// column pass, and the output rows added to `dst`.
///
/// # Safety
///
/// `V`'s ISA must be available (`codec_txfm` module docs) and have eight
/// lanes. `dst` must cover eight rows of eight samples at `stride`.
#[inline(always)]
pub(crate) unsafe fn add_scaled_8x8_lanes<V: I16x8Shuffles>(
    coeffs: &[i16; 64],
    dst: &mut [u8],
    stride: usize,
) {
    debug_assert_eq!(V::LANES, N);
    let residual = if let Some(dc) = dc_only(coeffs) {
        if dc == 0 {
            return;
        }
        [V::splat(dc); N]
    } else {
        let pass = |lines: [V; N]| {
            idct8(
                lines,
                |a: V, b| a.add(b),
                |a: V, b| a.sub(b),
                |a: V, s| a.sra(s),
            )
        };
        let rows: [V; N] = core::array::from_fn(|r| V::load(&coeffs[r * N..]));
        pass(transpose(pass(transpose(rows)))).map(V::rounded_shr::<6>)
    };
    let base = dst.as_mut_ptr();
    for (r, h) in residual.into_iter().enumerate() {
        // SAFETY: the dispatcher validated one 8x8 destination at stride.
        // The eight-byte row borrow ends before the next iteration.
        let row = unsafe { core::slice::from_raw_parts_mut(base.add(r * stride), N) };
        h.add(V::load_u8(row)).store_u8_sat(row);
    }
}

/// Transpose an 8x8 tile of 16-bit lanes: three rounds of interleaving row
/// `i` with row `i + 4` (each round moves one bit of the row index into the
/// column index).
#[inline(always)]
fn transpose<V: I16x8Shuffles>(mut v: [V; N]) -> [V; N] {
    for _ in 0..3 {
        let mut next = v;
        for i in 0..N / 2 {
            next[2 * i] = v[i].zip_lo(v[i + N / 2]);
            next[2 * i + 1] = v[i].zip_hi(v[i + N / 2]);
        }
        v = next;
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The spec's 8x8 inverse in wide integers, or `None` when an
    /// intermediate leaves the 16-bit range (a non-conforming block).
    fn spec_inverse(c: &[i16; 64]) -> Option<[i64; 64]> {
        let in_range = |x: i64| (-(1 << 15)..1 << 15).contains(&x);
        let mut ok = true;
        let mut checked = |v: [i64; N]| {
            ok &= v.iter().all(|&x| in_range(x));
            v
        };
        let line = |d: [i64; N], checked: &mut dyn FnMut([i64; N]) -> [i64; N]| {
            let a = checked([
                d[0] + d[4],
                d[0] - d[4],
                (d[2] >> 1) - d[6],
                d[2] + (d[6] >> 1),
                d[5] - d[3] - d[7] - (d[7] >> 1),
                d[1] + d[7] - d[3] - (d[3] >> 1),
                d[7] - d[1] + d[5] + (d[5] >> 1),
                d[3] + d[5] + d[1] + (d[1] >> 1),
            ]);
            let b = checked([
                a[0] + a[3],
                a[1] + a[2],
                a[1] - a[2],
                a[0] - a[3],
                a[4] + (a[7] >> 2),
                a[7] - (a[4] >> 2),
                a[5] + (a[6] >> 2),
                (a[5] >> 2) - a[6],
            ]);
            checked([
                b[0] + b[5],
                b[1] + b[7],
                b[2] + b[6],
                b[3] + b[4],
                b[3] - b[4],
                b[2] - b[6],
                b[1] - b[7],
                b[0] - b[5],
            ])
        };
        let mut rows = [[0i64; N]; N];
        for r in 0..N {
            rows[r] = line(
                core::array::from_fn(|k| i64::from(c[r * N + k])),
                &mut checked,
            );
        }
        let mut out = [0i64; 64];
        for col in 0..N {
            let v = line(core::array::from_fn(|r| rows[r][col]), &mut checked);
            for r in 0..N {
                out[r * N + col] = (v[r] + 32) >> 6;
            }
        }
        ok.then_some(out)
    }

    /// On every conforming block the 16-bit reference equals the spec's
    /// wide arithmetic; random blocks from sparse to dense, small to near
    /// the 16-bit limit, with the non-conforming ones skipped.
    #[test]
    fn sixteen_bit_inverse_equals_the_spec_on_conforming_blocks() {
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        let mut conforming = 0;
        for trial in 0..20_000 {
            let magnitude = [64, 1024, 8192, 32767][trial % 4];
            let density = 1 + trial % 64;
            let mut coeffs = [0i16; 64];
            for _ in 0..density {
                let at = (next() % 64) as usize;
                coeffs[at] = ((next() % (2 * magnitude + 1)) as i64 - magnitude as i64) as i16;
            }
            let Some(want) = spec_inverse(&coeffs) else {
                continue;
            };
            conforming += 1;
            let pred: Vec<u8> = (0..64).map(|_| (next() % 256) as u8).collect();
            let mut got = pred.clone();
            // SAFETY: `got` covers the block.
            unsafe { add_scaled_8x8(&coeffs, &mut got, N) };
            for i in 0..64 {
                let expected = (i64::from(pred[i]) + want[i]).clamp(0, 255);
                assert_eq!(i64::from(got[i]), expected, "trial {trial} sample {i}");
            }
        }
        assert!(conforming > 10_000, "{conforming} conforming blocks");
    }

    /// The rounding identity behind [`round6`], over every 16-bit value.
    #[test]
    fn split_rounding_equals_the_wide_rounding() {
        for h in i16::MIN..=i16::MAX {
            assert_eq!(i32::from(round6(h)), (i32::from(h) + 32) >> 6, "{h}");
        }
    }
}
