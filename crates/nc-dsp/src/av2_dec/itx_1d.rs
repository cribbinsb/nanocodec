//! The 1-D inverse transforms of the general AV2 decoder's [`super::itx`]:
//! the DCT (4 to 32 points), the matrix ADST / flipped ADST and inter DDT /
//! flipped DDT (4 to 16), the identity (4 to 32) and the lossless WHT.
//!
//! Ported from dav2d `src/itx_1d.c` (commit 237229e; Copyright © 2018-2026
//! `VideoLAN` and dav2d authors, BSD-2-Clause, see `docs/licenses/dav2d`),
//! bit-exact with AVM v1.0.0. dav2d transforms a strided column in place;
//! here every kernel takes one contiguous line (the 2-D caller gathers and
//! scatters columns), which keeps the arithmetic identical.
//!
//! All arithmetic is 32-bit, as dav2d's `int`: inputs are at most 16 bits
//! (coefficients, or the clamped row output), so the largest sum, a 32-point
//! DCT output, stays below `32 · 90 · 2^15 < 2^27`.

use super::itx_kernels::{
    ADST4, ADST8, ADST16, DCT8_ODD, DCT16_ODD, DCT32_ODD, DDT8, DDT16, FLIPADST4, FLIPADST16,
};

/// Largest number of points of one 1-D transform (a 64-point side is
/// inverted by the 32-point DCT, then upsampled).
pub(super) const MAX_POINTS: usize = 32;

/// Largest half of a DCT ([`MAX_POINTS`] / 2): the even and odd parts.
const MAX_HALF: usize = MAX_POINTS / 2;

/// Largest dense (ADST, DDT) transform.
const MAX_DENSE_POINTS: usize = 16;

/// Weights of the 4-point DCT butterfly (dav2d `inv_dct4_1d_c`): the even
/// pair's gain, then the odd pair's two weights.
const DCT4_EVEN: i32 = 64;
const DCT4_ODD_MAJOR: i32 = 83;
const DCT4_ODD_MINOR: i32 = 35;

/// Gains of the identity transforms by size (4, 8, 16, 32 points): `128`
/// times `sqrt(n / 4)`, the DCT's normalisation (dav2d
/// `inv_identity*_1d_c`).
const IDENTITY4_GAIN: i32 = 128;
const IDENTITY8_GAIN: i32 = 181;
const IDENTITY16_GAIN: i32 = 256;
const IDENTITY32_GAIN: i32 = 362;

/// One 1-D inverse transform of a whole line.
pub(super) type Transform1d = fn(&mut [i32]);

/// Inverse DCT of `c` (4, 8, 16 or 32 points). An `n`-point DCT inverts its
/// even inputs with the `n / 2`-point DCT and adds or subtracts the odd
/// inputs' matrix product (dav2d `inv_dct_1d_c`); the integer matrices nest,
/// so this recursion is the full matrix product.
pub(super) fn dct(c: &mut [i32]) {
    let n = c.len();
    if n == 4 {
        dct4(c);
        return;
    }
    let half = n / 2;
    let odd_matrix: &[i8] = match n {
        8 => &DCT8_ODD,
        16 => &DCT16_ODD,
        32 => &DCT32_ODD,
        _ => panic!("no {n}-point AV2 DCT"),
    };
    let mut even = [0; MAX_HALF];
    for (e, &v) in even.iter_mut().zip(c.iter().step_by(2)) {
        *e = v;
    }
    dct(&mut even[..half]);
    let mut odd = [0; MAX_HALF];
    for (o, weights) in odd.iter_mut().zip(odd_matrix.chunks_exact(half)) {
        *o = weights
            .iter()
            .zip(c.iter().skip(1).step_by(2))
            .map(|(&w, &v)| i32::from(w) * v)
            .sum();
    }
    for i in 0..half {
        c[i] = even[i] + odd[i];
        c[n - 1 - i] = even[i] - odd[i];
    }
}

/// The 4-point DCT butterfly (dav2d `inv_dct4_1d_c`).
fn dct4(c: &mut [i32]) {
    let a0 = c[0] * DCT4_EVEN + c[2] * DCT4_EVEN;
    let a1 = c[0] * DCT4_EVEN - c[2] * DCT4_EVEN;
    let b0 = c[1] * DCT4_ODD_MAJOR + c[3] * DCT4_ODD_MINOR;
    let b1 = c[1] * DCT4_ODD_MINOR - c[3] * DCT4_ODD_MAJOR;
    c[0] = a0 + b0;
    c[1] = a1 + b1;
    c[2] = a1 - b1;
    c[3] = a0 - b0;
}

/// A dense `n`-point transform (dav2d `inv_dst_1d_c`): output `i` is row `i`
/// of `matrix` times the input; `flip` writes the outputs in reverse order,
/// which is how dav2d derives the 8-point flipped ADST and the flipped DDTs
/// from the unflipped matrices.
fn dense(c: &mut [i32], matrix: &[i8], flip: bool) {
    let n = c.len();
    let mut sums = [0; MAX_DENSE_POINTS];
    for (sum, weights) in sums.iter_mut().zip(matrix.chunks_exact(n)) {
        *sum = weights
            .iter()
            .zip(c.iter())
            .map(|(&w, &v)| i32::from(w) * v)
            .sum();
    }
    for (i, &sum) in sums[..n].iter().enumerate() {
        c[if flip { n - 1 - i } else { i }] = sum;
    }
}

fn adst4(c: &mut [i32]) {
    dense(c, &ADST4, false);
}

fn adst8(c: &mut [i32]) {
    dense(c, &ADST8, false);
}

fn adst16(c: &mut [i32]) {
    dense(c, &ADST16, false);
}

fn flipadst4(c: &mut [i32]) {
    dense(c, &FLIPADST4, false);
}

fn flipadst8(c: &mut [i32]) {
    dense(c, &ADST8, true);
}

fn flipadst16(c: &mut [i32]) {
    dense(c, &FLIPADST16, false);
}

fn ddt8(c: &mut [i32]) {
    dense(c, &DDT8, false);
}

fn ddt16(c: &mut [i32]) {
    dense(c, &DDT16, false);
}

fn flipddt8(c: &mut [i32]) {
    dense(c, &DDT8, true);
}

fn flipddt16(c: &mut [i32]) {
    dense(c, &DDT16, true);
}

/// The identity: a per-size gain (dav2d `inv_identity*_1d_c`).
fn identity(c: &mut [i32]) {
    let gain = match c.len() {
        4 => IDENTITY4_GAIN,
        8 => IDENTITY8_GAIN,
        16 => IDENTITY16_GAIN,
        32 => IDENTITY32_GAIN,
        n => panic!("no {n}-point AV2 identity"),
    };
    for v in c {
        *v *= gain;
    }
}

/// The 1-D transform of kind `kind` over `log2_side4` (log2 of the side in
/// 4-sample units: 0 for 4 points .. 4 for 64), as dav2d's `dav2d_tx1d_fns`;
/// `None` where dav2d has none. A 64-point side takes the 32-point DCT: AVM
/// codes only its 32 lowest frequencies and the caller upsamples.
pub(super) fn transform(kind: super::itx::Tx1d, log2_side4: u8) -> Option<Transform1d> {
    use super::itx::Tx1d::{Adst, Dct, Ddt, FlipAdst, FlipDdt, Identity};
    let f: Transform1d = match (kind, log2_side4) {
        (Dct, 0..=4) => dct,
        (Identity, 0..=3) => identity,
        (Adst, 0) => adst4,
        (Adst, 1) => adst8,
        (Adst, 2) => adst16,
        (FlipAdst, 0) => flipadst4,
        (FlipAdst, 1) => flipadst8,
        (FlipAdst, 2) => flipadst16,
        (Ddt, 1) => ddt8,
        (Ddt, 2) => ddt16,
        (FlipDdt, 1) => flipddt8,
        (FlipDdt, 2) => flipddt16,
        _ => return None,
    };
    Some(f)
}

/// The 4-point inverse Walsh-Hadamard transform of lossless blocks (dav2d
/// `dav2d_inv_wht4_1d_c`, AV1 §7.13.2.10's lifting steps).
pub(super) fn wht4(c: &mut [i32]) {
    let t0 = c[0] + c[1];
    let t2 = c[2] - c[3];
    let t4 = (t0 - t2) >> 1;
    let t3 = t4 - c[3];
    let t1 = t4 - c[1];
    c[0] = t0 - t3;
    c[1] = t3;
    c[2] = t1;
    c[3] = t2 + t1;
}
