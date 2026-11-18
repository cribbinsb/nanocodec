//! High-bit-depth (`u16`, 10- and 12-bit) VP9 inverse-transform SIMD kernels
//! (`4x4..32x32` DCT and ADST) across AVX2, AVX-512, NEON, and SIMD128.
//!
//! Shares the 1-D ADST graphs ([`super::packed::adst`]) and EOB-pruned sparse
//! DCT graphs ([`super::packed::corner_dct`]) with the 8-bit packed path via
//! [`Wide32`], and shares constant block addition with
//! [`crate::av1_inverse::high_simd::HighAv1InverseFns`].
//!
//! Stage range proof: across `idct{8,16,32}` and `iadst{8,16}`, the largest
//! unrounded `i32` sum has L1 operator gain `166_955.4` over its 1-D input.
//! Bounding 1-D inputs by `SAFE_BOUND = 12_000` keeps every intermediate below
//! `2.004e9 < i32::MAX` (covering all normal 10-bit blocks and almost all
//! 12-bit blocks); extreme blocks fall back before touching `dst` to the
//! 64-bit scalar reference [`crate::vp9_txfm::vp9_inverse_add_high`].
//!
//! Attribution: EOB corner classes and DC-only shortcut follow libvpx
//! (`vpx_dsp/x86/highbd_idct*_avx2.c`, BSD-3-Clause; see `THIRD_PARTY.md`).

#![allow(clippy::inline_always)]

use nc_common::BitDepth;

use crate::av1_inverse::high_simd::HighConstantAddFn;
use crate::vp9_txfm::vp9_inverse_add_high;
use crate::{Isa, Vp9Pass};

/// High-bit-depth VP9 fused inverse transform + clipped sample addition.
#[allow(clippy::too_many_arguments)]
pub(crate) type Vp9InverseAdd16Fn = unsafe fn(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    nonzero: (usize, usize),
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
    dc_fn: HighConstantAddFn,
);

#[allow(clippy::too_many_arguments)]
fn scalar_add_high(
    log2: u8,
    columns: Vp9Pass,
    rows: Vp9Pass,
    input: &[i32],
    _nonzero: (usize, usize),
    dst: &mut [u16],
    dst_stride: usize,
    depth: BitDepth,
    _dc_fn: HighConstantAddFn,
) {
    vp9_inverse_add_high(log2, (columns, rows), input, (dst, dst_stride), depth);
}

pub(crate) fn add_kernel(isa: Isa) -> Vp9InverseAdd16Fn {
    match isa {
        #[cfg(target_arch = "x86_64")]
        Isa::Avx2 => simd::x86::inverse_add_avx2,
        #[cfg(target_arch = "x86_64")]
        Isa::Avx512 => simd::x86::inverse_add_avx512,
        #[cfg(target_arch = "aarch64")]
        Isa::Neon => simd::neon::inverse_add_neon,
        #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
        Isa::Simd128 => simd::simd128::inverse_add_simd128,
        _ => scalar_add_high,
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
mod simd {
    use core::mem::MaybeUninit;

    use nc_common::BitDepth;

    use super::super::packed::{COS_BITS, Products, WideLanes, corner_dct, line, output_shift};
    use crate::Vp9Pass;
    use crate::av1_inverse::high_simd::HighConstantAddFn;
    use crate::simd::I32Lanes;
    use crate::txfm::inverse_lanes::InverseLanes;
    use crate::vp9_txfm::{
        COS8, COS16, COS24, SHIFT4, SIN1, SIN2, SIN3, SIN4, vp9_inverse_add_high,
    };

    const MAX_LANES: usize = 16;
    const SAFE_BOUND: i32 = 12_000;
    const SAFE_BOUND_4X4: i32 = 24_000;
    const MAX_SIMD_BITS: u32 = 12;

    /// 32-bit transform stage lanes without 16-bit stage wrapping.
    #[repr(transparent)]
    #[derive(Clone, Copy)]
    struct Wide32<V: I32Lanes>(V);

    impl<V: I32Lanes> InverseLanes for Wide32<V> {
        #[inline(always)]
        fn splat(value: i32) -> Self {
            Self(V::splat(value))
        }
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            Self(self.0.add(other.0))
        }
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            Self(self.0.sub(other.0))
        }
        #[inline(always)]
        fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
            Self(V::half_btf(w0, a.0, w1, b.0, bit))
        }
        #[inline(always)]
        fn mul_round(w: i32, a: Self, bit: u32) -> Self {
            Self(a.0.mul_scalar(w).round_shift(bit))
        }
        #[inline(always)]
        fn clamp16(self) -> Self {
            self
        }
    }

    impl<V: I32Lanes> Products for Wide32<V> {
        type Narrow = Self;
        #[inline(always)]
        fn add(self, other: Self) -> Self {
            Self(self.0.add(other.0))
        }
        #[inline(always)]
        fn sub(self, other: Self) -> Self {
            Self(self.0.sub(other.0))
        }
        #[inline(always)]
        fn narrow(self) -> Self {
            self
        }
        #[inline(always)]
        fn round14(self) -> Self {
            Self(self.0.round_shift(COS_BITS))
        }
    }

    impl<V: I32Lanes> WideLanes for Wide32<V> {
        type Wide = Self;
        #[inline(always)]
        fn wide(self, weight: i32) -> Self {
            if weight == 1 {
                self
            } else {
                Self(self.0.mul_scalar(weight))
            }
        }
        #[inline(always)]
        fn products(w0: i32, a: Self, w1: i32, b: Self) -> Self {
            Self(a.0.mul_scalar(w0).add(b.0.mul_scalar(w1)))
        }
    }

    #[inline(always)]
    unsafe fn dc_add_high(
        dc: i32,
        side: usize,
        shift: u32,
        dst: &mut [u16],
        dst_stride: usize,
        depth: BitDepth,
        dc_fn: HighConstantAddFn,
    ) {
        if dc == 0 {
            return;
        }
        let round14 = |v: i64| ((v + (1 << (COS_BITS - 1))) >> COS_BITS) as i32;
        let r1 = round14(i64::from(dc) * i64::from(COS16));
        let r2 = round14(i64::from(r1) * i64::from(COS16));
        let residual = (r2 + (1 << (shift - 1))) >> shift;
        if residual != 0 {
            // SAFETY: forwarded ISA and destination contracts.
            unsafe { dc_fn(residual, side, side, dst, dst_stride, depth) };
        }
    }

    #[inline(always)]
    fn line4<V: I32Lanes>(pass: Vp9Pass, [a, b, c, d]: [V; 4]) -> [V; 4] {
        match pass {
            Vp9Pass::Dct => {
                let even0 = V::half_btf(COS16, a, COS16, c, COS_BITS);
                let even1 = V::half_btf(COS16, a, -COS16, c, COS_BITS);
                let odd0 = V::half_btf(COS24, b, -COS8, d, COS_BITS);
                let odd1 = V::half_btf(COS8, b, COS24, d, COS_BITS);
                [
                    even0.add(odd1),
                    even1.add(odd0),
                    even1.sub(odd0),
                    even0.sub(odd1),
                ]
            }
            Vp9Pass::Adst => {
                let diff = a.sub(c).add(d);
                let first = a
                    .mul_scalar(SIN1)
                    .add(c.mul_scalar(SIN4))
                    .add(d.mul_scalar(SIN2));
                let second = a
                    .mul_scalar(SIN2)
                    .sub(c.mul_scalar(SIN1))
                    .sub(d.mul_scalar(SIN4));
                let middle = b.mul_scalar(SIN3);
                [
                    first.add(middle).round_shift(COS_BITS),
                    second.add(middle).round_shift(COS_BITS),
                    diff.mul_scalar(SIN3).round_shift(COS_BITS),
                    first.add(second).sub(middle).round_shift(COS_BITS),
                ]
            }
        }
    }

    #[inline(always)]
    fn transpose4<V4: I32Lanes>(rows: &[V4; 4]) -> [V4; 4] {
        let mut tile = [MaybeUninit::<i32>::uninit(); 16];
        V4::store_transposed(rows, &mut tile, 4);
        // SAFETY: `store_transposed` wrote all 16 `i32` cells of `tile`.
        let s = unsafe { core::slice::from_raw_parts(tile.as_ptr().cast::<i32>(), 16) };
        [
            V4::load(&s[0..]),
            V4::load(&s[4..]),
            V4::load(&s[8..]),
            V4::load(&s[12..]),
        ]
    }

    #[inline(always)]
    fn inverse4_lanes<V4: I32Lanes>(
        passes: (Vp9Pass, Vp9Pass),
        input: &[i32],
        nonzero_rows: usize,
        dst: &mut [u16],
        dst_stride: usize,
        depth: BitDepth,
    ) {
        debug_assert_eq!(V4::LANES, 4);
        let input = &input[..16];
        let mut in_rows = [V4::splat(0); 4];
        let mut in_max = V4::splat(0);
        for (r, row) in in_rows.iter_mut().enumerate().take(nonzero_rows.min(4)) {
            let v = V4::load(&input[r * 4..]);
            in_max = in_max.max(v.abs());
            *row = v;
        }
        let bound = V4::splat(SAFE_BOUND_4X4);
        if in_max.gt(bound).count_nonzero() != 0 {
            vp9_inverse_add_high(2, passes, input, (dst, dst_stride), depth);
            return;
        }
        let y = line4(passes.1, transpose4(&in_rows));
        let row_max = y[0].abs().max(y[1].abs()).max(y[2].abs()).max(y[3].abs());
        if row_max.gt(bound).count_nonzero() != 0 {
            vp9_inverse_add_high(2, passes, input, (dst, dst_stride), depth);
            return;
        }
        let out = line4(passes.0, transpose4(&y));
        let max_val = depth.max();
        for (r, v) in out.into_iter().enumerate() {
            v.round_shift(SHIFT4)
                .add_to_u16(&mut dst[r * dst_stride..], max_val);
        }
    }

    #[inline(always)]
    fn pass1d<V: I32Lanes, const N: usize>(
        kind: Vp9Pass,
        active_len: usize,
        input: &[Wide32<V>; N],
    ) -> [Wide32<V>; N] {
        if kind == Vp9Pass::Dct {
            match N {
                8 => {
                    if active_len <= 4 {
                        corner_dct::<Wide32<V>, N, 4>(input)
                    } else {
                        corner_dct::<Wide32<V>, N, 8>(input)
                    }
                }
                16 => {
                    if active_len <= 4 {
                        corner_dct::<Wide32<V>, N, 4>(input)
                    } else if active_len <= 8 {
                        corner_dct::<Wide32<V>, N, 8>(input)
                    } else {
                        corner_dct::<Wide32<V>, N, 16>(input)
                    }
                }
                32 => {
                    if active_len <= 4 {
                        corner_dct::<Wide32<V>, N, 4>(input)
                    } else if active_len <= 8 {
                        corner_dct::<Wide32<V>, N, 8>(input)
                    } else if active_len <= 16 {
                        corner_dct::<Wide32<V>, N, 16>(input)
                    } else {
                        corner_dct::<Wide32<V>, N, 32>(input)
                    }
                }
                _ => unreachable!("validated VP9 transform size"),
            }
        } else {
            let mut out = *input;
            line::<Wide32<V>, N>(kind, input, &mut out);
            out
        }
    }

    #[inline(always)]
    fn inverse_large_lanes<V: I32Lanes, const N: usize>(
        log2: u8,
        passes: (Vp9Pass, Vp9Pass),
        input: &[i32],
        (nz_rows, nz_cols): (usize, usize),
        (dst, dst_stride): (&mut [u16], usize),
        depth: BitDepth,
    ) {
        let lanes = V::LANES;
        debug_assert!(lanes <= MAX_LANES && N.is_multiple_of(lanes));
        let input = &input[..N * N];
        let nz_r = nz_rows.clamp(1, N);
        let nz_c = nz_cols.clamp(1, N);
        let active_rows = nz_r.div_ceil(lanes) * lanes;
        let active_cols = nz_c.div_ceil(lanes) * lanes;
        let bound = V::splat(SAFE_BOUND);
        let mut temp_grid = [[MaybeUninit::<i32>::uninit(); N]; N];
        let temp = temp_grid.as_flattened_mut();
        let mut tile_buf = [MaybeUninit::<i32>::uninit(); MAX_LANES * MAX_LANES];
        let mut row_max = V::splat(0);

        for r0 in (0..active_rows).step_by(lanes) {
            let mut x = [Wide32(V::splat(0)); N];
            let mut in_max = V::splat(0);
            for c0 in (0..active_cols).step_by(lanes) {
                let mut rows_in = [V::splat(0); MAX_LANES];
                for (i, slot) in rows_in.iter_mut().take(lanes).enumerate() {
                    let v = V::load(&input[(r0 + i) * N + c0..]);
                    in_max = in_max.max(v.abs());
                    *slot = v;
                }
                V::store_transposed(&rows_in[..lanes], &mut tile_buf[..lanes * lanes], lanes);
                // SAFETY: `store_transposed` wrote all `lanes * lanes` cells.
                let s = unsafe {
                    core::slice::from_raw_parts(tile_buf.as_ptr().cast::<i32>(), lanes * lanes)
                };
                for j in 0..lanes {
                    x[c0 + j] = Wide32(V::load(&s[j * lanes..]));
                }
            }
            if in_max.gt(bound).count_nonzero() != 0 {
                vp9_inverse_add_high(log2, passes, input, (dst, dst_stride), depth);
                return;
            }
            let y = pass1d::<V, N>(passes.1, nz_c, &x);
            for c0 in (0..N).step_by(lanes) {
                let mut cols_out = [V::splat(0); MAX_LANES];
                for j in 0..lanes {
                    let v = y[c0 + j].0;
                    row_max = row_max.max(v.abs());
                    cols_out[j] = v;
                }
                V::store_transposed(&cols_out[..lanes], &mut temp[r0 * N + c0..], N);
            }
        }
        if row_max.gt(bound).count_nonzero() != 0 {
            vp9_inverse_add_high(log2, passes, input, (dst, dst_stride), depth);
            return;
        }
        // SAFETY: rows `0..active_rows` across all `0..N` columns of `temp` were written above.
        let temp_init =
            unsafe { core::slice::from_raw_parts(temp.as_ptr().cast::<i32>(), active_rows * N) };
        let shift = output_shift(N);
        let max_val = depth.max();
        for c0 in (0..N).step_by(lanes) {
            let mut x = [Wide32(V::splat(0)); N];
            for (r, slot) in x.iter_mut().enumerate().take(active_rows) {
                *slot = Wide32(V::load(&temp_init[r * N + c0..]));
            }
            let out = pass1d::<V, N>(passes.0, nz_r, &x);
            for (r, v) in out.into_iter().enumerate() {
                v.0.round_shift(shift)
                    .add_to_u16(&mut dst[r * dst_stride + c0..], max_val);
            }
        }
    }

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(super) unsafe fn dispatch_lanes<V4: I32Lanes, V8: I32Lanes, VWide: I32Lanes>(
        log2: u8,
        columns: Vp9Pass,
        rows: Vp9Pass,
        input: &[i32],
        (nz_rows, nz_cols): (usize, usize),
        dst: &mut [u16],
        dst_stride: usize,
        depth: BitDepth,
        dc_fn: HighConstantAddFn,
    ) {
        let passes = (columns, rows);
        let side = 1_usize << log2;
        if depth.bits() > MAX_SIMD_BITS || nz_rows > side || nz_cols > side {
            vp9_inverse_add_high(log2, passes, input, (dst, dst_stride), depth);
            return;
        }
        if passes == (Vp9Pass::Dct, Vp9Pass::Dct) && nz_rows <= 1 && nz_cols <= 1 {
            let dc = if nz_rows == 0 || nz_cols == 0 {
                0
            } else {
                input[0]
            };
            let shift = if log2 == 2 {
                SHIFT4
            } else {
                output_shift(side)
            };
            // SAFETY: forwarded ISA and destination contracts.
            unsafe { dc_add_high(dc, side, shift, dst, dst_stride, depth, dc_fn) };
            return;
        }
        match log2 {
            2 => inverse4_lanes::<V4>(passes, input, nz_rows, dst, dst_stride, depth),
            3 => inverse_large_lanes::<V8, 8>(
                3,
                passes,
                input,
                (nz_rows, nz_cols),
                (dst, dst_stride),
                depth,
            ),
            4 => inverse_large_lanes::<VWide, 16>(
                4,
                passes,
                input,
                (nz_rows, nz_cols),
                (dst, dst_stride),
                depth,
            ),
            5 => inverse_large_lanes::<VWide, 32>(
                5,
                passes,
                input,
                (nz_rows, nz_cols),
                (dst, dst_stride),
                depth,
            ),
            _ => unreachable!("validated VP9 transform size"),
        }
    }

    #[cfg(target_arch = "x86_64")]
    pub(super) mod x86 {
        use nc_common::BitDepth;

        use super::dispatch_lanes;
        use crate::Vp9Pass;
        use crate::av1_inverse::high_simd::HighConstantAddFn;
        use crate::simd::avx2::{I32x4, I32x8};
        use crate::simd::avx512::I32x16;

        #[target_feature(enable = "avx2")]
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn inverse_add_avx2(
            log2: u8,
            columns: Vp9Pass,
            rows: Vp9Pass,
            input: &[i32],
            nonzero: (usize, usize),
            dst: &mut [u16],
            dst_stride: usize,
            depth: BitDepth,
            dc_fn: HighConstantAddFn,
        ) {
            // SAFETY: AVX2 is enabled by this wrapper.
            unsafe {
                dispatch_lanes::<I32x4, I32x8, I32x8>(
                    log2, columns, rows, input, nonzero, dst, dst_stride, depth, dc_fn,
                );
            }
        }

        #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn inverse_add_avx512(
            log2: u8,
            columns: Vp9Pass,
            rows: Vp9Pass,
            input: &[i32],
            nonzero: (usize, usize),
            dst: &mut [u16],
            dst_stride: usize,
            depth: BitDepth,
            dc_fn: HighConstantAddFn,
        ) {
            // SAFETY: AVX-512F/BW/VL are enabled by this wrapper.
            unsafe {
                dispatch_lanes::<I32x4, I32x8, I32x16>(
                    log2, columns, rows, input, nonzero, dst, dst_stride, depth, dc_fn,
                );
            }
        }
    }

    #[cfg(target_arch = "aarch64")]
    pub(super) mod neon {
        use nc_common::BitDepth;

        use super::dispatch_lanes;
        use crate::Vp9Pass;
        use crate::av1_inverse::high_simd::HighConstantAddFn;
        use crate::simd::neon::I32x4;

        #[target_feature(enable = "neon")]
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn inverse_add_neon(
            log2: u8,
            columns: Vp9Pass,
            rows: Vp9Pass,
            input: &[i32],
            nonzero: (usize, usize),
            dst: &mut [u16],
            dst_stride: usize,
            depth: BitDepth,
            dc_fn: HighConstantAddFn,
        ) {
            // SAFETY: NEON is enabled by this wrapper.
            unsafe {
                dispatch_lanes::<I32x4, I32x4, I32x4>(
                    log2, columns, rows, input, nonzero, dst, dst_stride, depth, dc_fn,
                );
            }
        }
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    pub(super) mod simd128 {
        use nc_common::BitDepth;

        use super::dispatch_lanes;
        use crate::Vp9Pass;
        use crate::av1_inverse::high_simd::HighConstantAddFn;
        use crate::simd::simd128::I32x4;

        #[target_feature(enable = "simd128")]
        #[allow(clippy::too_many_arguments)]
        pub(crate) unsafe fn inverse_add_simd128(
            log2: u8,
            columns: Vp9Pass,
            rows: Vp9Pass,
            input: &[i32],
            nonzero: (usize, usize),
            dst: &mut [u16],
            dst_stride: usize,
            depth: BitDepth,
            dc_fn: HighConstantAddFn,
        ) {
            // SAFETY: SIMD128 is enabled by this wrapper.
            unsafe {
                dispatch_lanes::<I32x4, I32x4, I32x4>(
                    log2, columns, rows, input, nonzero, dst, dst_stride, depth, dc_fn,
                );
            }
        }
    }
}
