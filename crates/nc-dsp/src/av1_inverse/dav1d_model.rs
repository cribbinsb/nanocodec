//! Scalar model of dav1d 1.5.4's arm64 NEON 8-bit inverse transforms
//! (`src/arm/64/itx.S`, `inv_txfm_add_*_8bpc_neon`): the reference the
//! vendored assembly is checked against. It reproduces the assembly bit for
//! bit on *every* i16 coefficient input, conformant or not, so it follows
//! the instruction stream rather than dav1d's C (`src/itx_tmpl.c`,
//! `src/itx_1d.c`), which agrees with the assembly only on conformant
//! input. On conformant input it equals libaom and our scalar reference
//! (see the tests).
//!
//! Reimplemented in Rust from dav1d 1.5.4 `src/arm/64/itx.S`, Copyright
//! (c) 2018, `VideoLAN` and dav1d authors; Copyright (c) 2019, Martin Storsjo
//! (BSD-2-Clause; see `THIRD_PARTY.md`). Line numbers below are of that file.
//! Checked bit-exact against that assembly on a Jetson (aarch64) for every
//! size and type, with full-range, extreme, small and DC-only coefficients
//! consistent with the end of block, and with nonzero coefficients beyond
//! it (which the skipped groups must ignore and leave uncleared).
//!
//! # Arithmetic, per NEON instruction
//!
//! Every value between instructions is an i16 lane; each lane is computed
//! independently, so the model works on one line of samples at a time and
//! the assembly's transposes are only index changes here.
//!
//! - `smull`/`smlal`/`smlsl` (the `smull_smlal` / `smull_smlsl` macros) form
//!   exact 32-bit sums; `neg` and the ADST4 `mul`/`add` are 32-bit wrapping
//!   (no input reaches the wrap).
//! - `sqrshrn #12` rounds and *saturates* to 16 bits: every rotation output.
//! - `sqadd`/`sqsub`/`sqneg`/`sqshl` saturate; `sqrdmulh` (the rectangular
//!   prescale, identity scales, DCT64's first rotations) saturates only for
//!   `-32768 * -32768`.
//! - `srshr`, `srhadd` and `sshr` are exact (no overflow is possible).
//! - Several rotations negate the 32-bit sum *before* `sqrshrn` (DCT16,
//!   DCT32, DCT64), the ADSTs negate saturated 16-bit outputs (`sqneg`).
//! - The add to the prediction is `uaddw`, a wrapping 16-bit add of the
//!   pixel, then `sqxtun` clips to 0..=255. The wrap is unreachable: every
//!   residual has been shifted right by at least 2 by then.
//! - The identity transforms are fused with the row shift (`srhadd`, or
//!   skipped where `x2` then `>> 1` cancel) and, with a 32-sample side, with
//!   the column pass and output shift; this differs from "transform, then
//!   shift" exactly where an intermediate would saturate.
//!
//! # Structure and end of block
//!
//! A `DCT_DCT` block with only a DC coefficient (`eob == 1`) takes the
//! `idct_dc` shortcut (it reads and clears only `coeffs[0]`). Otherwise the
//! row pass (with the 2:1 `sqrdmulh` prescale), the row shift and the column
//! pass follow the size's driver. Drivers whose coded height exceeds 8 run
//! the row pass in groups of 8 rows, and skip every group after the first
//! whose end-of-block threshold (`eob - 1 < threshold`, from `eob_32x32` ..
//! `eob_8x32` and the `eob_half` arguments of `def_fn_*`) is not reached:
//! those rows are taken as zero, *whatever the coefficients hold*, and are
//! not cleared. Every coefficient the assembly reads is cleared. The
//! identity-identity transforms with a 32-sample side are 8x8 blocks of one
//! fused per-sample formula, skipped by row and by column block.
//!
//! # Coefficients and types
//!
//! Coefficient (row `r`, column `c`) is `coeffs[c * coded_height + r]`
//! (dav1d's `coeff[y + x * sh]`, coded sides capped at 32). `eob` is the
//! number of coded scan positions (dav1d's argument is `eob - 1`).
//!
//! dav1d's `inv_txfm_add_<a>_<b>_<w>x<h>` applies `<a>` horizontally (the
//! first, row pass) and `<b>` vertically: [`Av1TxType::AdstDct`] (vertical
//! ADST) is `dct_adst`, [`Av1TxType::VDct`] (vertical DCT) is
//! `identity_dct` (`src/itx.h` `assign_itx12_fn`; confirmed against the
//! assembly). The assembly implements a subset of the types [`Av1TxType::fits`]
//! allows, those the transform sets can signal ([`supports`]).

use super::{Av1TxSize, Av1TxType, Kernel1d};

/// `idct_coeffs` (itx.S:62): DCT4 (`[1]` is `2896 * 8` for `sqrdmulh`),
/// DCT8, DCT16 and DCT32 rotation constants.
const IDCT_COEFFS: [i16; 32] = [
    2896,
    2896 * 8,
    1567,
    3784,
    799,
    4017,
    3406,
    2276,
    401,
    4076,
    3166,
    2598,
    1931,
    3612,
    3920,
    1189,
    201,
    4091,
    3035,
    2751,
    1751,
    3703,
    3857,
    1380,
    995,
    3973,
    3513,
    2106,
    2440,
    3290,
    4052,
    601,
];

/// `idct64_coeffs` (itx.S:77), one row per `inv_dct64_step1_neon` call: eight
/// `sqrdmulh` constants (`x8`) for the input rotations, then four `smull`
/// constants (the table's trailing zeros are padding).
const IDCT64_COEFFS: [[i16; 12]; 4] = [
    [
        101 * 8,
        4095 * 8,
        2967 * 8,
        -2824 * 8,
        1660 * 8,
        3745 * 8,
        3822 * 8,
        -1474 * 8,
        4076,
        401,
        4017,
        799,
    ],
    [
        4036 * 8,
        -700 * 8,
        2359 * 8,
        3349 * 8,
        3461 * 8,
        -2191 * 8,
        897 * 8,
        3996 * 8,
        -3166,
        -2598,
        -799,
        -4017,
    ],
    [
        501 * 8,
        4065 * 8,
        3229 * 8,
        -2520 * 8,
        2019 * 8,
        3564 * 8,
        3948 * 8,
        -1092 * 8,
        3612,
        1931,
        2276,
        3406,
    ],
    [
        4085 * 8,
        -301 * 8,
        2675 * 8,
        3102 * 8,
        3659 * 8,
        -1842 * 8,
        1285 * 8,
        3889 * 8,
        -3920,
        -1189,
        -3406,
        -2276,
    ],
];

/// The DCT64 odd inputs each `inv_dct64_step1_neon` call reads
/// (`def_dct64_func`, itx.S:2821–2868).
const IDCT64_STEP1_INPUTS: [[usize; 4]; 4] = [
    [1, 31, 17, 15],
    [7, 25, 23, 9],
    [5, 27, 21, 11],
    [3, 29, 19, 13],
];

/// `iadst4_coeffs` (itx.S:99); `[3]` doubles as the 32-bit `mul` constant.
const IADST4_COEFFS: [i16; 4] = [1321, 3803, 2482, 3344];

/// `iadst8_coeffs` (itx.S:104); its second row repeats `idct_coeffs`.
const IADST8_COEFFS: [i16; 8] = [4076, 401, 3612, 1931, 2598, 3166, 1189, 3920];

/// `iadst16_coeffs` (itx.S:111).
const IADST16_COEFFS: [i16; 16] = [
    4091, 201, 3973, 995, 3703, 1751, 3290, 2440, 2751, 3035, 2106, 3513, 1380, 3857, 601, 4052,
];

/// `sqrdmulh` by this is `Round2(x * 2896, 12)`: the 2:1 prescale and the
/// DC shortcut's DCT gain (`#2896*8`).
const INV_SQRT2_Q15: i16 = 2896 * 8;
/// `sqrdmulh` by this is `Round2(x * (5793 - 4096), 12)`: the fractional
/// part of identity4's `sqrt(2)` (`#(5793-4096)*8`).
const IDENTITY4_FRACTION: i16 = (5793 - 4096) * 8;
/// The same for identity16's `2 * sqrt(2)` (`#2*(5793-4096)*8`).
const IDENTITY16_FRACTION: i16 = 2 * (5793 - 4096) * 8;
/// `sqrshrn` shift after every rotation.
const ROTATION_SHIFT: u32 = 12;
/// `srshr` shift of the column output before the add (`load_add_store`).
const OUTPUT_SHIFT: u32 = 4;
/// Rows per row-pass group (one 8-lane register per coefficient column).
const ROW_GROUP: usize = 8;
/// Side of the identity-identity 32 drivers' blocks.
const IDENTITY_BLOCK: usize = 8;

/// `eob_32x32` (itx.S:2162): the `eob - 1` each further 8-row (or column)
/// group needs. Each table's last entry (1024, 512, 256) exceeds every
/// `eob - 1` and is left out.
const EOB_32X32: [usize; 3] = [36, 136, 300];
/// `eob_16x32` (itx.S:2166).
const EOB_16X32: [usize; 3] = [36, 151, 279];
/// `eob_16x32_shortside` (itx.S:2170).
const EOB_16X32_SHORTSIDE: [usize; 1] = [36];
/// `eob_8x32` (itx.S:2174).
const EOB_8X32: [usize; 3] = [43, 107, 171];

/// A driver's `eob_half` (the second 8-row group's threshold) by transform
/// class: 2-D, `H_*` (row transform only) and `V_*` (column only).
struct HalfThresholds {
    two_d: usize,
    horizontal: usize,
    vertical: usize,
}

/// `def_fns_416` (itx.S:1657), used by 4x16.
const EOB_HALF_4X16: HalfThresholds = HalfThresholds {
    two_d: 29,
    horizontal: 8,
    vertical: 32,
};
/// `def_fns_816` (itx.S:1826), used by 8x16.
const EOB_HALF_8X16: HalfThresholds = HalfThresholds {
    two_d: 43,
    horizontal: 8,
    vertical: 64,
};
/// `def_fn_16x16` (itx.S:1476): 36, but 8 for `H_DCT` and `V_DCT`.
const EOB_HALF_16X16: usize = 36;
const EOB_HALF_16X16_1D: usize = 8;

// ---------------------------------------------------------------------------
// Lane operations, one per NEON instruction.

/// `sqadd`.
fn sqadd(a: i16, b: i16) -> i16 {
    a.saturating_add(b)
}

/// `sqsub`.
fn sqsub(a: i16, b: i16) -> i16 {
    a.saturating_sub(b)
}

/// `sqneg`.
fn sqneg(a: i16) -> i16 {
    a.saturating_neg()
}

/// `sqshl #1`.
fn sqshl1(a: i16) -> i16 {
    a.saturating_mul(2)
}

/// `srshr #shift` (`shift >= 1`).
fn srshr(a: i16, shift: u32) -> i16 {
    ((i32::from(a) + (1 << (shift - 1))) >> shift) as i16
}

/// `srhadd`.
fn srhadd(a: i16, b: i16) -> i16 {
    ((i32::from(a) + i32::from(b) + 1) >> 1) as i16
}

/// Saturate to 16 bits.
fn saturate(x: i64) -> i16 {
    x.clamp(i64::from(i16::MIN), i64::from(i16::MAX)) as i16
}

/// `sqrdmulh` by a constant.
fn sqrdmulh(a: i16, c: i16) -> i16 {
    saturate((2 * i64::from(a) * i64::from(c) + (1 << 15)) >> 16)
}

/// `smull` + `smlal`: `a * ca + b * cb` in 32 bits.
fn mul_add(a: i16, b: i16, ca: i16, cb: i16) -> i32 {
    (i32::from(a) * i32::from(ca)).wrapping_add(i32::from(b) * i32::from(cb))
}

/// `smull` + `smlsl`: `a * ca - b * cb` in 32 bits.
fn mul_sub(a: i16, b: i16, ca: i16, cb: i16) -> i32 {
    (i32::from(a) * i32::from(ca)).wrapping_sub(i32::from(b) * i32::from(cb))
}

/// `sqrshrn #12`.
fn narrow(x: i32) -> i16 {
    saturate((i64::from(x) + (1 << (ROTATION_SHIFT - 1))) >> ROTATION_SHIFT)
}

/// `uaddw` of the pixel (wrapping) then `sqxtun`.
fn add_pixel(pixel: u8, residual: i16) -> u8 {
    let sum = residual.wrapping_add(i16::from(pixel));
    sum.clamp(0, i16::from(u8::MAX)) as u8
}

// ---------------------------------------------------------------------------
// 1-D DCTs.

/// `idct_4` (itx.S:437).
fn idct4(x: [i16; 4]) -> [i16; 4] {
    let c = &IDCT_COEFFS;
    let t3 = narrow(mul_add(x[1], x[3], c[3], c[2]));
    let t2 = narrow(mul_sub(x[1], x[3], c[2], c[3]));
    let t0 = narrow(mul_add(x[0], x[2], c[0], c[0]));
    let t1 = narrow(mul_sub(x[0], x[2], c[0], c[0]));
    [sqadd(t0, t3), sqadd(t1, t2), sqsub(t1, t2), sqsub(t0, t3)]
}

/// `idct_8` (itx.S:707).
fn idct8(x: [i16; 8]) -> [i16; 8] {
    let c = &IDCT_COEFFS;
    let e = idct4([x[0], x[2], x[4], x[6]]);
    let t4a = narrow(mul_sub(x[1], x[7], c[4], c[5]));
    let t7a = narrow(mul_add(x[1], x[7], c[5], c[4]));
    let t5a = narrow(mul_sub(x[5], x[3], c[6], c[7]));
    let t6a = narrow(mul_add(x[5], x[3], c[7], c[6]));

    let t4 = sqadd(t4a, t5a);
    let t5a = sqsub(t4a, t5a);
    let t7 = sqadd(t7a, t6a);
    let t6a = sqsub(t7a, t6a);

    let t5 = narrow(mul_sub(t6a, t5a, c[0], c[0]));
    let t6 = narrow(mul_add(t6a, t5a, c[0], c[0]));
    [
        sqadd(e[0], t7),
        sqadd(e[1], t6),
        sqadd(e[2], t5),
        sqadd(e[3], t4),
        sqsub(e[3], t4),
        sqsub(e[2], t5),
        sqsub(e[1], t6),
        sqsub(e[0], t7),
    ]
}

/// `idct_16` (itx.S:1027).
fn idct16(x: [i16; 16]) -> [i16; 16] {
    let c = &IDCT_COEFFS;
    let e = idct8([x[0], x[2], x[4], x[6], x[8], x[10], x[12], x[14]]);
    let t8a = narrow(mul_sub(x[1], x[15], c[8], c[9]));
    let t15a = narrow(mul_add(x[1], x[15], c[9], c[8]));
    let t9a = narrow(mul_sub(x[9], x[7], c[10], c[11]));
    let t14a = narrow(mul_add(x[9], x[7], c[11], c[10]));
    let t10a = narrow(mul_sub(x[5], x[11], c[12], c[13]));
    let t13a = narrow(mul_add(x[5], x[11], c[13], c[12]));
    let t11a = narrow(mul_sub(x[13], x[3], c[14], c[15]));
    let t12a = narrow(mul_add(x[13], x[3], c[15], c[14]));

    let t9 = sqsub(t8a, t9a);
    let t8 = sqadd(t8a, t9a);
    let t14 = sqsub(t15a, t14a);
    let t15 = sqadd(t15a, t14a);
    let t10 = sqsub(t11a, t10a);
    let t11 = sqadd(t11a, t10a);
    let t12 = sqadd(t12a, t13a);
    let t13 = sqsub(t12a, t13a);

    let t9a = narrow(mul_sub(t14, t9, c[2], c[3]));
    let t14a = narrow(mul_add(t14, t9, c[3], c[2]));
    let t13a = narrow(mul_sub(t13, t10, c[2], c[3]));
    let t10a = narrow(mul_add(t13, t10, c[3], c[2]).wrapping_neg());

    let t11a = sqsub(t8, t11);
    let t8a = sqadd(t8, t11);
    let t12a = sqsub(t15, t12);
    let t15a = sqadd(t15, t12);
    let t9 = sqadd(t9a, t10a);
    let t10 = sqsub(t9a, t10a);
    let t13 = sqsub(t14a, t13a);
    let t14 = sqadd(t14a, t13a);

    let t11 = narrow(mul_sub(t12a, t11a, c[0], c[0]));
    let t12 = narrow(mul_add(t12a, t11a, c[0], c[0]));
    let t10a = narrow(mul_sub(t13, t10, c[0], c[0]));
    let t13a = narrow(mul_add(t13, t10, c[0], c[0]));

    let odd = [t15a, t14, t13a, t12, t11, t10a, t9, t8a];
    std::array::from_fn(|i| {
        if i < 8 {
            sqadd(e[i], odd[i])
        } else {
            sqsub(e[15 - i], odd[15 - i])
        }
    })
}

/// `inv_dct32_odd_8h_x16_neon` (itx.S:1848): the odd half of DCT32 from
/// its odd inputs `x[k] = in[2k + 1]`. Output `k` is the assembly's register
/// `v(16 + k)` (`t16` .. `t31`).
fn idct32_odd(x: [i16; 16]) -> [i16; 16] {
    let a = &IDCT_COEFFS[16..];
    let c = &IDCT_COEFFS;
    let t16a = narrow(mul_sub(x[0], x[15], a[0], a[1]));
    let t31a = narrow(mul_add(x[0], x[15], a[1], a[0]));
    let t17a = narrow(mul_sub(x[8], x[7], a[2], a[3]));
    let t30a = narrow(mul_add(x[8], x[7], a[3], a[2]));
    let t18a = narrow(mul_sub(x[4], x[11], a[4], a[5]));
    let t29a = narrow(mul_add(x[4], x[11], a[5], a[4]));
    let t19a = narrow(mul_sub(x[12], x[3], a[6], a[7]));
    let t28a = narrow(mul_add(x[12], x[3], a[7], a[6]));
    let t20a = narrow(mul_sub(x[2], x[13], a[8], a[9]));
    let t27a = narrow(mul_add(x[2], x[13], a[9], a[8]));
    let t21a = narrow(mul_sub(x[10], x[5], a[10], a[11]));
    let t26a = narrow(mul_add(x[10], x[5], a[11], a[10]));
    let t22a = narrow(mul_sub(x[6], x[9], a[12], a[13]));
    let t25a = narrow(mul_add(x[6], x[9], a[13], a[12]));
    let t23a = narrow(mul_sub(x[14], x[1], a[14], a[15]));
    let t24a = narrow(mul_add(x[14], x[1], a[15], a[14]));

    let t17 = sqsub(t16a, t17a);
    let t16 = sqadd(t16a, t17a);
    let t30 = sqsub(t31a, t30a);
    let t31 = sqadd(t31a, t30a);
    let t18 = sqsub(t19a, t18a);
    let t19 = sqadd(t19a, t18a);
    let t20 = sqadd(t20a, t21a);
    let t21 = sqsub(t20a, t21a);
    let t22 = sqsub(t23a, t22a);
    let t23 = sqadd(t23a, t22a);
    let t24 = sqadd(t24a, t25a);
    let t25 = sqsub(t24a, t25a);
    let t26 = sqsub(t27a, t26a);
    let t27 = sqadd(t27a, t26a);
    let t28 = sqadd(t28a, t29a);
    let t29 = sqsub(t28a, t29a);

    let t17a = narrow(mul_sub(t30, t17, c[4], c[5]));
    let t30a = narrow(mul_add(t30, t17, c[5], c[4]));
    let t18a = narrow(mul_add(t29, t18, c[5], c[4]).wrapping_neg());
    let t29a = narrow(mul_sub(t29, t18, c[4], c[5]));
    let t21a = narrow(mul_sub(t26, t21, c[6], c[7]));
    let t26a = narrow(mul_add(t26, t21, c[7], c[6]));
    let t22a = narrow(mul_add(t25, t22, c[7], c[6]).wrapping_neg());
    let t25a = narrow(mul_sub(t25, t22, c[6], c[7]));

    let t29 = sqsub(t30a, t29a);
    let t30 = sqadd(t30a, t29a);
    let t18 = sqsub(t17a, t18a);
    let t17 = sqadd(t17a, t18a);
    let t19a = sqsub(t16, t19);
    let t16a = sqadd(t16, t19);
    let t20a = sqsub(t23, t20);
    let t23a = sqadd(t23, t20);
    let t21 = sqsub(t22a, t21a);
    let t22 = sqadd(t22a, t21a);
    let t24a = sqadd(t24, t27);
    let t27a = sqsub(t24, t27);
    let t25 = sqadd(t25a, t26a);
    let t26 = sqsub(t25a, t26a);
    let t28a = sqsub(t31, t28);
    let t31a = sqadd(t31, t28);

    let t18a = narrow(mul_sub(t29, t18, c[2], c[3]));
    let t29a = narrow(mul_add(t29, t18, c[3], c[2]));
    let t19 = narrow(mul_sub(t28a, t19a, c[2], c[3]));
    let t28 = narrow(mul_add(t28a, t19a, c[3], c[2]));
    let t20 = narrow(mul_add(t27a, t20a, c[3], c[2]).wrapping_neg());
    let t27 = narrow(mul_sub(t27a, t20a, c[2], c[3]));
    let t21a = narrow(mul_add(t26, t21, c[3], c[2]).wrapping_neg());
    let t26a = narrow(mul_sub(t26, t21, c[2], c[3]));

    let t23 = sqsub(t16a, t23a);
    let out16 = sqadd(t16a, t23a);
    let t24 = sqsub(t31a, t24a);
    let out31 = sqadd(t31a, t24a);
    let t22a = sqsub(t17, t22);
    let out17 = sqadd(t17, t22);
    let out30 = sqadd(t30, t25);
    let t25a = sqsub(t30, t25);
    let t21 = sqsub(t18a, t21a);
    let out18 = sqadd(t18a, t21a);
    let out19 = sqadd(t19, t20);
    let t20a = sqsub(t19, t20);
    let out29 = sqadd(t29a, t26a);
    let t26 = sqsub(t29a, t26a);
    let out28 = sqadd(t28, t27);
    let t27a = sqsub(t28, t27);

    let t20 = narrow(mul_sub(t27a, t20a, c[0], c[0]));
    let t27 = narrow(mul_add(t27a, t20a, c[0], c[0]));
    let t26a = narrow(mul_add(t26, t21, c[0], c[0]));
    let t21a = narrow(mul_sub(t26, t21, c[0], c[0]));
    let t22 = narrow(mul_sub(t25a, t22a, c[0], c[0]));
    let t25 = narrow(mul_add(t25a, t22a, c[0], c[0]));
    let t23a = narrow(mul_sub(t24, t23, c[0], c[0]));
    let t24a = narrow(mul_add(t24, t23, c[0], c[0]));

    [
        out16, out17, out18, out19, t20, t21a, t22, t23a, t24a, t25, t26a, t27, out28, out29,
        out30, out31,
    ]
}

/// DCT32: `idct_16` on the even inputs, `inv_dct32_odd` on the odd ones,
/// then the saturating butterfly of `store2` (itx.S:2058) / `combine`
/// (itx.S:2118): `out[i] = e[i] + t[31 - i]`, `out[31 - i] = e[i] - t[31 - i]`.
fn idct32(x: [i16; 32]) -> [i16; 32] {
    let even = idct16(std::array::from_fn(|k| x[2 * k]));
    let odd = idct32_odd(std::array::from_fn(|k| x[2 * k + 1]));
    std::array::from_fn(|i| {
        if i < 16 {
            sqadd(even[i], odd[15 - i])
        } else {
            sqsub(even[31 - i], odd[i - 16])
        }
    })
}

/// `inv_dct64_step1_neon` (itx.S:2558): four odd inputs to eight
/// intermediates (`t32a t33 t34a t35 t60 t61a t62 t63a`).
fn idct64_step1(x: [i16; 4], k: &[i16; 12]) -> [i16; 8] {
    let t63a = sqrdmulh(x[0], k[1]);
    let t32a = sqrdmulh(x[0], k[0]);
    let t62a = sqrdmulh(x[1], k[2]);
    let t33a = sqrdmulh(x[1], k[3]);
    let t61a = sqrdmulh(x[2], k[5]);
    let t34a = sqrdmulh(x[2], k[4]);
    let t60a = sqrdmulh(x[3], k[6]);
    let t35a = sqrdmulh(x[3], k[7]);

    let t32 = sqadd(t32a, t33a);
    let t33 = sqsub(t32a, t33a);
    let t34 = sqsub(t35a, t34a);
    let t35 = sqadd(t35a, t34a);
    let t60 = sqadd(t60a, t61a);
    let t61 = sqsub(t60a, t61a);
    let t62 = sqsub(t63a, t62a);
    let t63 = sqadd(t63a, t62a);

    let t34a = narrow(mul_add(t61, t34, k[8], k[9]).wrapping_neg());
    let t61a = narrow(mul_sub(t61, t34, k[9], k[8]));
    let t33a = narrow(mul_sub(t62, t33, k[9], k[8]));
    let t62a = narrow(mul_add(t62, t33, k[8], k[9]));

    let t32a = sqadd(t32, t35);
    let t35a = sqsub(t32, t35);
    let t33 = sqadd(t33a, t34a);
    let t34 = sqsub(t33a, t34a);
    let t60a = sqsub(t63, t60);
    let t63a = sqadd(t63, t60);
    let t61 = sqsub(t62a, t61a);
    let t62 = sqadd(t62a, t61a);

    let t61a = narrow(mul_add(t61, t34, k[10], k[11]));
    let t34a = narrow(mul_sub(t61, t34, k[11], k[10]));
    let t60 = narrow(mul_add(t60a, t35a, k[10], k[11]));
    let t35 = narrow(mul_sub(t60a, t35a, k[11], k[10]));
    [t32a, t33, t34a, t35, t60, t61a, t62, t63a]
}

/// `inv_dct64_step2_neon` (itx.S:2619), in place on the 32 step-1 outputs
/// (four groups of eight, in step-1 order), walking `i` up and `j` down.
fn idct64_step2(t: &mut [i16; 32]) {
    let c = &IDCT_COEFFS;
    for i in 0..4 {
        let j = 7 - i;
        let t32a = t[i];
        let t39a = t[j + 8];
        let t63a = t[j];
        let t56a = t[i + 8];
        let t40a = t[i + 16];
        let t47a = t[j + 24];
        let t55a = t[j + 16];
        let t48a = t[i + 24];

        let t32 = sqadd(t32a, t39a);
        let t39 = sqsub(t32a, t39a);
        let t63 = sqadd(t63a, t56a);
        let t56 = sqsub(t63a, t56a);
        let t40 = sqsub(t47a, t40a);
        let t47 = sqadd(t47a, t40a);
        let t48 = sqadd(t48a, t55a);
        let t55 = sqsub(t48a, t55a);

        let t56a = narrow(mul_add(t56, t39, c[3], c[2]));
        let t39a = narrow(mul_sub(t56, t39, c[2], c[3]));
        let t40a = narrow(mul_add(t55, t40, c[3], c[2]).wrapping_neg());
        let t55a = narrow(mul_sub(t55, t40, c[2], c[3]));

        let t32a = sqadd(t32, t47);
        let t47a = sqsub(t32, t47);
        let t39 = sqadd(t39a, t40a);
        let t40 = sqsub(t39a, t40a);
        let t48a = sqsub(t63, t48);
        let t63a = sqadd(t63, t48);
        let t55 = sqsub(t56a, t55a);
        let t56 = sqadd(t56a, t55a);

        t[i] = t32a;
        t[j] = t39;
        t[i + 8] = narrow(mul_sub(t55, t40, c[0], c[0])); // t40a
        t[j + 8] = narrow(mul_sub(t48a, t47a, c[0], c[0])); // t47
        t[i + 16] = narrow(mul_add(t48a, t47a, c[0], c[0])); // t48
        t[j + 16] = narrow(mul_add(t55, t40, c[0], c[0])); // t55a
        t[i + 24] = t56;
        t[j + 24] = t63a;
    }
}

/// DCT64 of 32 coded inputs (`def_dct64_func`, itx.S:2749; the upper 32
/// inputs are zero, §7.13.3, and are never read): DCT32 of the even inputs,
/// the four step-1 calls and step 2 on the odd ones, then the saturating
/// butterfly of `horz_dct_64x8` / `vert_dct_8x64` (itx.S:2890, 2941).
fn idct64(x: [i16; 64]) -> [i16; 64] {
    let even = idct32(std::array::from_fn(|k| if k < 16 { x[2 * k] } else { 0 }));
    let mut odd = [0; 32];
    for (group, (inputs, k)) in IDCT64_STEP1_INPUTS.iter().zip(&IDCT64_COEFFS).enumerate() {
        odd[group * 8..][..8].copy_from_slice(&idct64_step1(inputs.map(|n| x[n]), k));
    }
    idct64_step2(&mut odd);
    std::array::from_fn(|i| {
        if i < 32 {
            sqadd(even[i], odd[31 - i])
        } else {
            sqsub(even[63 - i], odd[i - 32])
        }
    })
}

// ---------------------------------------------------------------------------
// 1-D ADSTs.

/// `iadst_4x4` / `iadst_8x4` (itx.S:464, 499).
fn iadst4(input: [i16; 4]) -> [i16; 4] {
    let k = IADST4_COEFFS.map(i32::from);
    let [in0, in1, in2, in3] = input.map(i32::from);
    let sum = in0.wrapping_sub(in2).wrapping_add(in3);
    let plus = (in0 * k[0])
        .wrapping_add(in2 * k[1])
        .wrapping_add(in3 * k[2]);
    let minus = (in0 * k[2])
        .wrapping_sub(in2 * k[0])
        .wrapping_sub(in3 * k[1]);
    let odd = in1 * k[3];
    [
        narrow(plus.wrapping_add(odd)),
        narrow(minus.wrapping_add(odd)),
        narrow(sum.wrapping_mul(k[3])),
        narrow(plus.wrapping_add(minus).wrapping_sub(odd)),
    ]
}

/// `iadst_8` (itx.S:752).
fn iadst8(x: [i16; 8]) -> [i16; 8] {
    let a = &IADST8_COEFFS;
    let c = &IDCT_COEFFS;
    let t0a = narrow(mul_add(x[7], x[0], a[0], a[1]));
    let t1a = narrow(mul_sub(x[7], x[0], a[1], a[0]));
    let t2a = narrow(mul_add(x[5], x[2], a[2], a[3]));
    let t3a = narrow(mul_sub(x[5], x[2], a[3], a[2]));
    let t4a = narrow(mul_add(x[3], x[4], a[4], a[5]));
    let t5a = narrow(mul_sub(x[3], x[4], a[5], a[4]));
    let t6a = narrow(mul_add(x[1], x[6], a[6], a[7]));
    let t7a = narrow(mul_sub(x[1], x[6], a[7], a[6]));

    let t0 = sqadd(t0a, t4a);
    let t4 = sqsub(t0a, t4a);
    let t1 = sqadd(t1a, t5a);
    let t5 = sqsub(t1a, t5a);
    let t2 = sqadd(t2a, t6a);
    let t6 = sqsub(t2a, t6a);
    let t3 = sqadd(t3a, t7a);
    let t7 = sqsub(t3a, t7a);

    let t4a = narrow(mul_add(t4, t5, c[3], c[2]));
    let t5a = narrow(mul_sub(t4, t5, c[2], c[3]));
    let t6a = narrow(mul_sub(t7, t6, c[3], c[2]));
    let t7a = narrow(mul_add(t7, t6, c[2], c[3]));

    let out0 = sqadd(t0, t2);
    let t2 = sqsub(t0, t2);
    let out7 = sqneg(sqadd(t1, t3));
    let t3 = sqsub(t1, t3);
    let out1 = sqneg(sqadd(t4a, t6a));
    let t6 = sqsub(t4a, t6a);
    let out6 = sqadd(t5a, t7a);
    let t7 = sqsub(t5a, t7a);

    let out3 = sqneg(narrow(mul_add(t2, t3, c[0], c[0])));
    let out4 = narrow(mul_sub(t2, t3, c[0], c[0]));
    let out5 = sqneg(narrow(mul_sub(t6, t7, c[0], c[0])));
    let out2 = narrow(mul_add(t6, t7, c[0], c[0]));
    [out0, out1, out2, out3, out4, out5, out6, out7]
}

/// `iadst_16` (itx.S:1124).
fn iadst16(x: [i16; 16]) -> [i16; 16] {
    let a = &IADST16_COEFFS;
    let c = &IDCT_COEFFS;
    let t0 = narrow(mul_add(x[15], x[0], a[0], a[1]));
    let t1 = narrow(mul_sub(x[15], x[0], a[1], a[0]));
    let t2 = narrow(mul_add(x[13], x[2], a[2], a[3]));
    let t3 = narrow(mul_sub(x[13], x[2], a[3], a[2]));
    let t4 = narrow(mul_add(x[11], x[4], a[4], a[5]));
    let t5 = narrow(mul_sub(x[11], x[4], a[5], a[4]));
    let t6 = narrow(mul_add(x[9], x[6], a[6], a[7]));
    let t7 = narrow(mul_sub(x[9], x[6], a[7], a[6]));
    let t8 = narrow(mul_add(x[7], x[8], a[8], a[9]));
    let t9 = narrow(mul_sub(x[7], x[8], a[9], a[8]));
    let t10 = narrow(mul_add(x[5], x[10], a[10], a[11]));
    let t11 = narrow(mul_sub(x[5], x[10], a[11], a[10]));
    let t12 = narrow(mul_add(x[3], x[12], a[12], a[13]));
    let t13 = narrow(mul_sub(x[3], x[12], a[13], a[12]));
    let t14 = narrow(mul_add(x[1], x[14], a[14], a[15]));
    let t15 = narrow(mul_sub(x[1], x[14], a[15], a[14]));

    let t8a = sqsub(t0, t8);
    let t0a = sqadd(t0, t8);
    let t9a = sqsub(t1, t9);
    let t1a = sqadd(t1, t9);
    let t2a = sqadd(t2, t10);
    let t10a = sqsub(t2, t10);
    let t3a = sqadd(t3, t11);
    let t11a = sqsub(t3, t11);
    let t4a = sqadd(t4, t12);
    let t12a = sqsub(t4, t12);
    let t5a = sqadd(t5, t13);
    let t13a = sqsub(t5, t13);
    let t6a = sqadd(t6, t14);
    let t14a = sqsub(t6, t14);
    let t7a = sqadd(t7, t15);
    let t15a = sqsub(t7, t15);

    let t8 = narrow(mul_add(t8a, t9a, c[5], c[4]));
    let t9 = narrow(mul_sub(t8a, t9a, c[4], c[5]));
    let t10 = narrow(mul_add(t10a, t11a, c[7], c[6]));
    let t11 = narrow(mul_sub(t10a, t11a, c[6], c[7]));
    let t12 = narrow(mul_sub(t13a, t12a, c[5], c[4]));
    let t13 = narrow(mul_add(t13a, t12a, c[4], c[5]));
    let t14 = narrow(mul_sub(t15a, t14a, c[7], c[6]));
    let t15 = narrow(mul_add(t15a, t14a, c[6], c[7]));

    let t4 = sqsub(t0a, t4a);
    let t0 = sqadd(t0a, t4a);
    let t5 = sqsub(t1a, t5a);
    let t1 = sqadd(t1a, t5a);
    let t2 = sqadd(t2a, t6a);
    let t6 = sqsub(t2a, t6a);
    let t3 = sqadd(t3a, t7a);
    let t7 = sqsub(t3a, t7a);
    let t8a = sqadd(t8, t12);
    let t12a = sqsub(t8, t12);
    let t9a = sqadd(t9, t13);
    let t13a = sqsub(t9, t13);
    let t10a = sqadd(t10, t14);
    let t14a = sqsub(t10, t14);
    let t11a = sqadd(t11, t15);
    let t15a = sqsub(t11, t15);

    let t4a = narrow(mul_add(t4, t5, c[3], c[2]));
    let t5a = narrow(mul_sub(t4, t5, c[2], c[3]));
    let t6a = narrow(mul_sub(t7, t6, c[3], c[2]));
    let t7a = narrow(mul_add(t7, t6, c[2], c[3]));
    let t12 = narrow(mul_add(t12a, t13a, c[3], c[2]));
    let t13 = narrow(mul_sub(t12a, t13a, c[2], c[3]));
    let t14 = narrow(mul_sub(t15a, t14a, c[3], c[2]));
    let t15 = narrow(mul_add(t15a, t14a, c[2], c[3]));

    let t2a = sqsub(t0, t2);
    let out0 = sqadd(t0, t2);
    let t3a = sqsub(t1, t3);
    let out15 = sqneg(sqadd(t1, t3));
    let t15a = sqsub(t13, t15);
    let out13 = sqneg(sqadd(t13, t15));
    let out2 = sqadd(t12, t14);
    let t14a = sqsub(t12, t14);
    let out1 = sqneg(sqadd(t8a, t10a));
    let t10 = sqsub(t8a, t10a);
    let out14 = sqadd(t9a, t11a);
    let t11 = sqsub(t9a, t11a);
    let out3 = sqneg(sqadd(t4a, t6a));
    let t6 = sqsub(t4a, t6a);
    let out12 = sqadd(t5a, t7a);
    let t7 = sqsub(t5a, t7a);

    let out8 = narrow(mul_sub(t2a, t3a, c[0], c[0]));
    let out7 = sqneg(narrow(mul_add(t2a, t3a, c[0], c[0])));
    let out5 = sqneg(narrow(mul_add(t14a, t15a, c[0], c[0])));
    let out10 = narrow(mul_sub(t14a, t15a, c[0], c[0]));
    let out4 = narrow(mul_add(t6, t7, c[0], c[0]));
    let out11 = sqneg(narrow(mul_sub(t6, t7, c[0], c[0])));
    let out6 = narrow(mul_add(t10, t11, c[0], c[0]));
    let out9 = sqneg(narrow(mul_sub(t10, t11, c[0], c[0])));
    [
        out0, out1, out2, out3, out4, out5, out6, out7, out8, out9, out10, out11, out12, out13,
        out14, out15,
    ]
}

// ---------------------------------------------------------------------------
// Identity transforms, as each driver fuses them.

/// `inv_identity_*_x4` (itx.S:564): `x * sqrt(2)`.
fn identity4(x: i16) -> i16 {
    sqadd(x, sqrdmulh(x, IDENTITY4_FRACTION))
}

/// `inv_identity_*_x16` (itx.S:1313), also `identity_8x8` (itx.S:1359):
/// `x * 2 * sqrt(2)`.
fn identity16(x: i16) -> i16 {
    sqadd(sqadd(x, x), sqrdmulh(x, IDENTITY16_FRACTION))
}

/// `identity_8x16_shift1` / `identity_8x8_shift1` (itx.S:1343, 1351):
/// identity16 fused with a row shift of 1.
fn identity16_shift1(x: i16) -> i16 {
    sqadd(x, srshr(sqrdmulh(x, IDENTITY16_FRACTION), 1))
}

/// The identity row transform of `width` samples fused with its row
/// shift, as the drivers of sizes without a 32-sample side implement it.
fn identity_row(width: usize, shift: u32, x: i16) -> i16 {
    match (width, shift) {
        (4, 0) => identity4(x),
        // `identity_8x4_shift1` (itx.S:592), 4x16.
        (4, 1) => srhadd(x, sqrdmulh(x, IDENTITY4_FRACTION)),
        // `inv_identity_4h_x8` (itx.S:851), 8x4.
        (8, 0) => sqshl1(x),
        // 8x8 and 8x16: "The identity shl #1 and downshift srshr #1 cancel
        // out" (itx.S:875, 1743).
        (8, 1) => x,
        (16, 1) => identity16_shift1(x),
        // `identity_8x16_shift2` (itx.S:1335), 16x16.
        (16, 2) => srhadd(x, sqrdmulh(x, IDENTITY16_FRACTION) >> 1),
        _ => unreachable!("no identity row of {width} samples with shift {shift}"),
    }
}

/// The identity column transform of `height` samples
/// (`inv_identity_*_x4/x8/x16`).
fn identity_column(height: usize, x: i16) -> i16 {
    match height {
        4 => identity4(x),
        8 => sqshl1(x),
        16 => identity16(x),
        _ => unreachable!("no identity column of {height} samples"),
    }
}

/// The residual of coefficient `x` of an identity-identity transform with a
/// 32-sample side: the whole 2-D transform and output shift fused into one
/// formula (itx.S:2178–2323).
fn identity32_residual(width: usize, height: usize, x: i16) -> i16 {
    match (width, height) {
        (32, 32) => srshr(x, 2),
        // `identity_8x8_shift1`, `load_add_store_8x8 shiftbits=2`.
        (16, 32) => srshr(identity16_shift1(sqrdmulh(x, INV_SQRT2_Q15)), 2),
        // `sqshl #1`, `identity_8x8`, `load_add_store_8x8 shiftbits=4`.
        (32, 16) => srshr(identity16(sqshl1(sqrdmulh(x, INV_SQRT2_Q15))), OUTPUT_SHIFT),
        (8, 32) => srshr(srshr(x, 1), 2),
        (32, 8) => srshr(x, 3),
        _ => unreachable!("no fused identity transform at {width}x{height}"),
    }
}

// ---------------------------------------------------------------------------
// 2-D drivers.

/// Whether dav1d's assembly has `kind` at `tx` (`src/arm/itx.h`): every
/// type up to 16x8 / 8x16 / 16x4 / 4x16, twelve at 16x16 (no 1-D ADST),
/// `DCT_DCT` and `IDTX` with a 32-sample side, only `DCT_DCT` with a
/// 64-sample side.
#[must_use]
pub(crate) fn supports(tx: Av1TxSize, kind: Av1TxType) -> bool {
    use Av1TxType::{DctDct, HAdst, HFlipadst, Idtx, VAdst, VFlipadst};
    let longest = tx.width().max(tx.height());
    let one_d_adst = matches!(kind, VAdst | HAdst | VFlipadst | HFlipadst);
    match longest {
        64 => kind == DctDct,
        32 => matches!(kind, DctDct | Idtx),
        16 if tx.width() == 16 && tx.height() == 16 => !one_d_adst,
        _ => kind.fits(tx),
    }
}

/// Inverse transform `coeffs` (column-major, module docs) of `kind` at `tx`
/// and add it to the 8-bit `dst`, exactly as dav1d's
/// `inv_txfm_add_<row>_<column>_<w>x<h>_8bpc_neon` does, clearing every
/// coefficient the assembly reads. `eob` is the number of coded scan
/// positions (at least 1).
///
/// # Panics
///
/// If the assembly has no such transform ([`supports`]), `eob` is 0, or a
/// buffer is too small.
pub(crate) fn inverse_add(
    tx: Av1TxSize,
    kind: Av1TxType,
    coeffs: &mut [i16],
    eob: usize,
    dst: &mut [u8],
    stride: usize,
) {
    assert!(supports(tx, kind), "dav1d has no {kind:?} at {tx:?}");
    assert!(eob >= 1, "an inverse transform needs a coded coefficient");
    assert!(coeffs.len() >= tx.coded_area(), "too few coefficients");
    assert!(
        dst.len() >= (tx.height() - 1) * stride + tx.width(),
        "destination too small"
    );
    if kind == Av1TxType::DctDct && eob == 1 {
        dc_add(tx, coeffs, dst, stride);
    } else if kind == Av1TxType::Idtx && tx.width().max(tx.height()) == 32 {
        identity32_add(tx, coeffs, eob, dst, stride);
    } else {
        two_pass_add(tx, kind, coeffs, eob, dst, stride);
    }
}

/// `idct_dc` (itx.S:277; 4x4 inline at itx.S:666): the DC through both
/// passes as one value, added to every pixel.
fn dc_add(tx: Av1TxSize, coeffs: &mut [i16], dst: &mut [u8], stride: usize) {
    let mut dc = sqrdmulh(std::mem::take(&mut coeffs[0]), INV_SQRT2_Q15);
    if tx.is_two_to_one() {
        dc = sqrdmulh(dc, INV_SQRT2_Q15);
    }
    let shift = tx.row_shift();
    if shift > 0 {
        dc = srshr(dc, shift);
    }
    let residual = srshr(sqrdmulh(dc, INV_SQRT2_Q15), OUTPUT_SHIFT);
    for row in dst.chunks_mut(stride).take(tx.height()) {
        for pixel in &mut row[..tx.width()] {
            *pixel = add_pixel(*pixel, residual);
        }
    }
}

/// The `eob - 1` thresholds of the row groups after the first (empty: one
/// group holds every coded row).
fn row_group_thresholds(tx: Av1TxSize, kind: Av1TxType) -> &'static [usize] {
    use Av1TxType::{HAdst, HDct, HFlipadst, VAdst, VDct, VFlipadst};
    let by_class = |half: &'static HalfThresholds| -> &'static [usize] {
        match kind {
            HDct | HAdst | HFlipadst => std::slice::from_ref(&half.horizontal),
            VDct | VAdst | VFlipadst => std::slice::from_ref(&half.vertical),
            _ => std::slice::from_ref(&half.two_d),
        }
    };
    match (tx.width(), tx.height()) {
        (4, 16) => by_class(&EOB_HALF_4X16),
        (8, 16) => by_class(&EOB_HALF_8X16),
        (16, 16) if matches!(kind, HDct | VDct) => &[EOB_HALF_16X16_1D],
        (16, 16) => &[EOB_HALF_16X16],
        (16, 32 | 64) => &EOB_16X32,
        (32 | 64, 16) => &EOB_16X32_SHORTSIDE,
        (32 | 64, 32 | 64) => &EOB_32X32,
        (8, 32) => &EOB_8X32,
        _ => &[],
    }
}

/// Groups the assembly runs: the first, then each whose threshold `eob - 1`
/// reaches (`cmp w3, wN; b.lt`; the thresholds increase, so the first miss
/// ends the walk).
fn groups_run(thresholds: &[usize], eob: usize) -> usize {
    1 + thresholds.iter().take_while(|&&t| eob > t).count()
}

/// The general driver: the row pass over the coded rows the end of block
/// keeps, then the column pass and the add (`inv_txfm_add_<w>x<h>_neon`,
/// `def_fn_*`, `inv_txfm_add_dct_dct_*`).
fn two_pass_add(
    tx: Av1TxSize,
    kind: Av1TxType,
    coeffs: &mut [i16],
    eob: usize,
    dst: &mut [u8],
    stride: usize,
) {
    let (width, height) = (tx.width(), tx.height());
    let (coded_width, coded_height) = (tx.coded_width(), tx.coded_height());
    let (vertical, horizontal) = kind.kernels();
    let shift = tx.row_shift();
    let rows = (groups_run(row_group_thresholds(tx, kind), eob) * ROW_GROUP).min(coded_height);

    // Row pass output, row-major; rows the end of block skips stay zero.
    let mut rows_out = vec![0i16; coded_height * width];
    for (y, out) in rows_out.chunks_exact_mut(width).take(rows).enumerate() {
        for (x, sample) in out[..coded_width].iter_mut().enumerate() {
            *sample = std::mem::take(&mut coeffs[x * coded_height + y]);
            if tx.is_two_to_one() {
                *sample = sqrdmulh(*sample, INV_SQRT2_Q15);
            }
        }
        if horizontal == Kernel1d::Identity {
            for sample in out.iter_mut() {
                *sample = identity_row(width, shift, *sample);
            }
        } else {
            transform(horizontal, out);
            if shift > 0 {
                for sample in out.iter_mut() {
                    *sample = srshr(*sample, shift);
                }
            }
        }
    }

    let mut column = [0i16; 64];
    for x in 0..width {
        let column = &mut column[..height];
        column.fill(0);
        for (y, sample) in column[..coded_height].iter_mut().enumerate() {
            *sample = rows_out[y * width + x];
        }
        if vertical == Kernel1d::Identity {
            for sample in column.iter_mut() {
                *sample = identity_column(height, *sample);
            }
        } else {
            transform(vertical, column);
        }
        for (y, &residual) in column.iter().enumerate() {
            let pixel = &mut dst[y * stride + x];
            *pixel = add_pixel(*pixel, srshr(residual, OUTPUT_SHIFT));
        }
    }
}

/// The identity-identity drivers with a 32-sample side
/// (`inv_txfm_add_identity_identity_{32x32,16x32,32x16,8x32,32x8}`,
/// itx.S:2178–2323): 8x8 blocks of [`identity32_residual`], walked by row
/// and by column block, each axis cut off by its own end-of-block table.
fn identity32_add(tx: Av1TxSize, coeffs: &mut [i16], eob: usize, dst: &mut [u8], stride: usize) {
    let (width, height) = (tx.width(), tx.height());
    let (row_thresholds, column_thresholds): (&[usize], &[usize]) = match (width, height) {
        (32, 32) => (&EOB_32X32, &EOB_32X32),
        (16, 32) => (&EOB_16X32, &EOB_16X32_SHORTSIDE),
        (32, 16) => (&EOB_16X32_SHORTSIDE, &EOB_16X32),
        (8, 32) => (&EOB_8X32, &[]),
        (32, 8) => (&[], &EOB_8X32),
        _ => unreachable!("no fused identity transform at {width}x{height}"),
    };
    let rows = groups_run(row_thresholds, eob) * IDENTITY_BLOCK;
    let columns = groups_run(column_thresholds, eob) * IDENTITY_BLOCK;
    for y in 0..rows.min(height) {
        for x in 0..columns.min(width) {
            let coefficient = std::mem::take(&mut coeffs[x * height + y]);
            let pixel = &mut dst[y * stride + x];
            *pixel = add_pixel(*pixel, identity32_residual(width, height, coefficient));
        }
    }
}

/// One 1-D DCT, ADST or flipped ADST over `line` (its length is the
/// transform side). `FLIPADST` is the ADST with its outputs written to the
/// registers in reverse order (`inv_flipadst_*`).
fn transform(kernel: Kernel1d, line: &mut [i16]) {
    match (kernel, line.len()) {
        (Kernel1d::Dct, 4) => apply(line, idct4),
        (Kernel1d::Dct, 8) => apply(line, idct8),
        (Kernel1d::Dct, 16) => apply(line, idct16),
        (Kernel1d::Dct, 32) => apply(line, idct32),
        (Kernel1d::Dct, 64) => apply(line, idct64),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 4) => apply(line, iadst4),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 8) => apply(line, iadst8),
        (Kernel1d::Adst | Kernel1d::FlipAdst, 16) => apply(line, iadst16),
        _ => unreachable!("no {kernel:?} of {} samples", line.len()),
    }
    if kernel == Kernel1d::FlipAdst {
        line.reverse();
    }
}

/// Run a fixed-size 1-D transform in place on `line`.
fn apply<const N: usize>(line: &mut [i16], f: fn([i16; N]) -> [i16; N]) {
    let input: [i16; N] = line.try_into().expect("line is the transform side");
    line.copy_from_slice(&f(input));
}

#[cfg(test)]
mod tests {
    use super::{inverse_add, supports};
    use crate::av1_inverse::{Av1TxSize, Av1TxType};
    use crate::{Dsp, Isa};
    use nc_common::PlaneViewMut;

    /// A small deterministic generator (xorshift64*).
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
        }

        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }

        fn signed(&mut self, magnitude: i32) -> i32 {
            (self.below(2 * magnitude as u64 + 1) as i32) - magnitude
        }
    }

    /// Conformant blocks: few coefficients, small enough that no stage of
    /// either implementation saturates (§7.13.3 requires that of a stream).
    /// The model must then equal libaom's arithmetic (our scalar reference).
    #[test]
    fn matches_the_scalar_reference_on_conformant_blocks() {
        const WIDTH: usize = 64;
        let dsp = Dsp::with_isa(Isa::Scalar).expect("scalar");
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let mut cases = 0;
        for tx in Av1TxSize::ALL {
            for kind in Av1TxType::ALL
                .into_iter()
                .filter(|&kind| supports(tx, kind))
            {
                for trial in 0..40 {
                    let area = tx.coded_area();
                    let mut coeffs = vec![0i16; area];
                    // DC only (the shortcut), else every position coded.
                    let eob = if trial == 0 { 1 } else { area };
                    let budget = 3000;
                    let count = if trial == 0 {
                        1
                    } else {
                        1 + rng.below(6) as usize
                    };
                    for _ in 0..count {
                        let at = if eob == 1 {
                            0
                        } else {
                            rng.below(area as u64) as usize
                        };
                        coeffs[at] = rng.signed(budget / count as i32) as i16;
                    }
                    let wide: Vec<i32> = coeffs.iter().map(|&c| i32::from(c)).collect();
                    let prediction: Vec<u8> = (0..WIDTH * tx.height())
                        .map(|_| rng.below(256) as u8)
                        .collect();

                    let mut expected = prediction.clone();
                    let mut view =
                        PlaneViewMut::new(&mut expected, WIDTH, tx.width(), tx.height()).unwrap();
                    dsp.av1_inverse_add(tx, kind, &wide, eob, &mut view);

                    let mut actual = prediction;
                    inverse_add(tx, kind, &mut coeffs, eob, &mut actual, WIDTH);
                    assert_eq!(actual, expected, "{tx:?} {kind:?} trial {trial}");
                    assert!(
                        coeffs.iter().all(|&c| c == 0),
                        "{tx:?} {kind:?}: not cleared"
                    );
                    cases += 1;
                }
            }
        }
        assert!(cases > 0);
    }

    #[test]
    fn supports_the_signalable_types() {
        let count = |tx: Av1TxSize| Av1TxType::ALL.iter().filter(|&&k| supports(tx, k)).count();
        let expected = |w: usize, h: usize| match (w.max(h), w == h) {
            (64, _) => 1,
            (32, _) => 2,
            (16, true) => 12,
            _ => 16,
        };
        for tx in Av1TxSize::ALL {
            assert_eq!(count(tx), expected(tx.width(), tx.height()), "{tx:?}");
        }
    }
}
