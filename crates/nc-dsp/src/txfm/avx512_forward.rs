//! Thirty-two word lanes for the shared AV1/VP9 forward columns and
//! HEVC/VVC staged rows. The butterfly graph remains `dct::fdct32`; this
//! adapter supplies packed multiply-adds and a register transpose.
//!
//! Layout reference: SVT-AV1 v4.3.0 `lowbd_fwd_txfm2d_avx512.c`,
//! Copyright 2026 Meta Platforms, Inc. and affiliates (BSD-2-Clause and
//! AOM Patent License 1.0; `THIRD_PARTY.md`). No cosine graph is copied.
//! The existing narrow-stage tests prove that 8-bit residuals scaled by
//! four, including every equal-weight pre-add, stay within i16. Products
//! widen to i32 before rounding. AV1 rows use the existing exact L1 envelope and retain wide fallback.

#![allow(clippy::inline_always)]

use super::{
    avx2_forward::{Column as HalfColumn, transpose16},
    cospi, dct,
    forward_lanes::{ForwardLanes, NARROW_ROW_L1_LIMIT},
    generic,
};
use crate::simd::{I32Lanes, avx512::I32x16};
use core::arch::x86_64::{
    __m512i, _mm_cvtsi32_si128, _mm256_abs_epi16, _mm256_adds_epu16, _mm256_cmpgt_epu16_mask,
    _mm256_set1_epi16, _mm256_setzero_si256, _mm256_storeu_si256, _mm512_abs_epi16,
    _mm512_add_epi16, _mm512_add_epi32, _mm512_adds_epu16, _mm512_castsi512_si256,
    _mm512_cmpgt_epu16_mask, _mm512_cvtepi16_epi32, _mm512_extracti64x4_epi64, _mm512_loadu_si512,
    _mm512_madd_epi16, _mm512_mulhrs_epi16, _mm512_mullo_epi32, _mm512_packs_epi32,
    _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_i64x2,
    _mm512_slli_epi16, _mm512_sra_epi32, _mm512_srai_epi16, _mm512_storeu_si512, _mm512_sub_epi16,
    _mm512_unpackhi_epi16, _mm512_unpackhi_epi32, _mm512_unpackhi_epi64, _mm512_unpacklo_epi16,
    _mm512_unpacklo_epi32, _mm512_unpacklo_epi64,
};
use core::mem::MaybeUninit;

const SIDE: usize = 32;
const INPUT_SHIFT: i32 = crate::codec_txfm::generic::ROW_INPUT_SHIFT;
const COS_BITS: u32 = 12;
const Q15_BITS: u32 = 15;
const HEVC_OUTPUT_SCALE: i32 = 181;

#[derive(Clone, Copy)]
struct Column(__m512i);

impl ForwardLanes for Column {
    #[inline(always)]
    fn splat(value: i32) -> Self {
        // SAFETY: this private type exists only inside AVX-512 wrappers.
        unsafe { Self(_mm512_set1_epi16(value as i16)) }
    }
    #[inline(always)]
    fn add(self, other: Self) -> Self {
        // SAFETY: as above; every stage fits i16 (module contract).
        unsafe { Self(_mm512_add_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn sub(self, other: Self) -> Self {
        // SAFETY: as above.
        unsafe { Self(_mm512_sub_epi16(self.0, other.0)) }
    }
    #[inline(always)]
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        // SAFETY: private AVX-512 values; paired products fit i32 and
        // rounded outputs fit i16. The pack restores lane-local order.
        unsafe {
            if (w0 == w1 || w0 == -w1) && w0.abs() < (1 << bit) {
                let sum = if w0 == w1 {
                    _mm512_add_epi16(a.0, b.0)
                } else {
                    _mm512_sub_epi16(a.0, b.0)
                };
                return Self(_mm512_mulhrs_epi16(
                    sum,
                    _mm512_set1_epi16((w0 << (Q15_BITS - bit)) as i16),
                ));
            }
            let weight = _mm512_set1_epi32((w0 & 0xffff) | (w1 << 16));
            let round = _mm512_set1_epi32(1 << (bit - 1));
            let shift = _mm_cvtsi32_si128(bit as i32);
            let lo = _mm512_madd_epi16(_mm512_unpacklo_epi16(a.0, b.0), weight);
            let hi = _mm512_madd_epi16(_mm512_unpackhi_epi16(a.0, b.0), weight);
            Self(_mm512_packs_epi32(
                _mm512_sra_epi32(_mm512_add_epi32(lo, round), shift),
                _mm512_sra_epi32(_mm512_add_epi32(hi, round), shift),
            ))
        }
    }
}

/// Swap row/column indices of a 32-square word tile. The unpack ladder
/// transposes each eight-square sub-tile, then the 128-bit groups exchange
/// their block-row and block-column indices.
#[inline(always)]
fn transpose(rows: [Column; SIDE]) -> [Column; SIDE] {
    // SAFETY: only ICL wrappers construct these private vector values.
    unsafe {
        let mut words = rows;
        for k in (0..SIDE).step_by(2) {
            words[k] = Column(_mm512_unpacklo_epi16(rows[k].0, rows[k + 1].0));
            words[k + 1] = Column(_mm512_unpackhi_epi16(rows[k].0, rows[k + 1].0));
        }
        let mut dwords = words;
        for group in (0..SIDE).step_by(4) {
            for pair in 0..2 {
                let (a, b) = (words[group + pair].0, words[group + pair + 2].0);
                dwords[group + 2 * pair] = Column(_mm512_unpacklo_epi32(a, b));
                dwords[group + 2 * pair + 1] = Column(_mm512_unpackhi_epi32(a, b));
            }
        }
        let mut qwords = dwords;
        for group in (0..SIDE).step_by(8) {
            for pair in 0..4 {
                let (a, b) = (dwords[group + pair].0, dwords[group + pair + 4].0);
                qwords[group + 2 * pair] = Column(_mm512_unpacklo_epi64(a, b));
                qwords[group + 2 * pair + 1] = Column(_mm512_unpackhi_epi64(a, b));
            }
        }
        let mut out = qwords;
        for k in 0..8 {
            let (a, b, c, d) = (
                qwords[k].0,
                qwords[k + 8].0,
                qwords[k + 16].0,
                qwords[k + 24].0,
            );
            let (ab0, ab1) = (
                _mm512_shuffle_i64x2::<0x88>(a, b),
                _mm512_shuffle_i64x2::<0xdd>(a, b),
            );
            let (cd0, cd1) = (
                _mm512_shuffle_i64x2::<0x88>(c, d),
                _mm512_shuffle_i64x2::<0xdd>(c, d),
            );
            out[k] = Column(_mm512_shuffle_i64x2::<0x88>(ab0, cd0));
            out[k + 16] = Column(_mm512_shuffle_i64x2::<0xdd>(ab0, cd0));
            out[k + 8] = Column(_mm512_shuffle_i64x2::<0x88>(ab1, cd1));
            out[k + 24] = Column(_mm512_shuffle_i64x2::<0xdd>(ab1, cd1));
        }
        out
    }
}

/// Initialize one row of the wide intermediate from a word vector.
#[inline(always)]
unsafe fn store_wide(value: __m512i, out: &mut [MaybeUninit<i32>]) {
    let out = &mut out[..SIDE];
    // SAFETY: the slice covers both sixteen-dword stores; the caller
    // enables AVX-512. Sign extension is exact, without a saturating step.
    unsafe {
        _mm512_storeu_si512(
            out.as_mut_ptr().cast(),
            _mm512_cvtepi16_epi32(_mm512_castsi512_si256(value)),
        );
        _mm512_storeu_si512(
            out.as_mut_ptr().add(SIDE / 2).cast(),
            _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64::<1>(value)),
        );
    }
}

/// AV1 columns: exact libaom input shift 2, Q12 graph and middle shift 4.
#[inline(always)]
unsafe fn column_values<const LOW16: bool>(src: &[i16], stride: usize) -> [Column; SIDE] {
    let mut x = [Column::splat(0); SIDE];
    for (r, value) in x.iter_mut().enumerate() {
        let row = &src[r * stride..][..SIDE];
        // SAFETY: checked complete row; ICL wrapper is the only caller.
        *value = unsafe {
            Column(_mm512_slli_epi16::<{ INPUT_SHIFT as u32 }>(
                _mm512_loadu_si512(row.as_ptr().cast()),
            ))
        };
    }
    let mut y = x;
    let keep = if LOW16 { SIDE / 2 } else { SIDE };
    if LOW16 {
        dct::fdct32_low16(&x, &mut y, cospi(COS_BITS), COS_BITS);
    } else {
        dct::fdct32(&x, &mut y, cospi(COS_BITS), COS_BITS);
    }
    for value in &mut y[..keep] {
        // SAFETY: final column range <=32640, so the rounding add fits.
        value.0 =
            unsafe { _mm512_srai_epi16::<4>(_mm512_add_epi16(value.0, _mm512_set1_epi16(8))) };
    }
    y
}

/// Transpose only the retained sixteen column frequencies: two 16-square
/// tiles instead of a padded 32-square transpose. Lane r of output c is
/// lane c of input r; discarded graph scratch is never consumed.
#[inline(always)]
fn transpose_low(rows: &[Column; SIDE]) -> [HalfColumn; SIDE] {
    // SAFETY: private ICL values, sixteen valid frequency rows.
    unsafe {
        let mut lo = [_mm256_setzero_si256(); SIDE / 2];
        let mut hi = lo;
        for k in 0..SIDE / 2 {
            lo[k] = _mm512_castsi512_si256(rows[k].0);
            hi[k] = _mm512_extracti64x4_epi64::<1>(rows[k].0);
        }
        let mut out = [HalfColumn::splat(0); SIDE];
        for (c, row) in transpose16(lo).into_iter().enumerate() {
            out[c] = HalfColumn(row);
        }
        for (c, row) in transpose16(hi).into_iter().enumerate() {
            out[c + SIDE / 2] = HalfColumn(row);
        }
        out
    }
}

#[inline(always)]
fn low_rows_fit(words: &[HalfColumn; SIDE]) -> bool {
    // SAFETY: middle-shifted inputs cannot be i16::MIN; unsigned
    // saturation remains above the established narrow-row envelope.
    unsafe {
        let mut sum = _mm256_setzero_si256();
        for value in words {
            sum = _mm256_adds_epu16(sum, _mm256_abs_epi16(value.0));
        }
        _mm256_cmpgt_epu16_mask(sum, _mm256_set1_epi16(NARROW_ROW_L1_LIMIT as i16)) == 0
    }
}

/// Unsigned saturating L1 sums: any saturation remains above the narrow
/// guard, so it cannot admit an overflowing word butterfly. No approximation
/// is made: wide fallback keeps every coefficient when a line exceeds it.
#[inline(always)]
fn line_l1(words: &[Column; SIDE]) -> __m512i {
    // SAFETY: middle-shifted values cannot be i16::MIN; unsigned
    // saturation cannot make a sum pass the signed-word safety guard.
    unsafe {
        let mut sum = _mm512_setzero_si512();
        for value in words {
            sum = _mm512_adds_epu16(sum, _mm512_abs_epi16(value.0));
        }
        sum
    }
}

#[inline(always)]
fn within(l1: __m512i, limit: u16) -> bool {
    // SAFETY: private ICL values; unsigned compare preserves saturated sums.
    unsafe { _mm512_cmpgt_epu16_mask(l1, _mm512_set1_epi16(limit as i16)) == 0 }
}

#[inline(always)]
unsafe fn wide_words<'a>(
    words: &[Column; SIDE],
    scratch: &'a mut [MaybeUninit<i32>; SIDE * SIDE],
) -> &'a [i32] {
    for (r, v) in words.iter().enumerate() {
        // SAFETY: each complete word row initializes thirty-two wide slots.
        unsafe { store_wide(v.0, &mut scratch[r * SIDE..]) };
    }
    // SAFETY: every slot was initialized immediately above.
    unsafe { crate::codec_txfm::generic::written(scratch, SIDE * SIDE) }
}

#[inline(always)]
unsafe fn word_rows<const RASTER: bool>(words: &[Column; SIDE], out: &mut [i16]) {
    let mut y = [Column::splat(0); SIDE];
    dct::fdct32(words, &mut y, cospi(COS_BITS), COS_BITS);
    let values = if RASTER { transpose(y) } else { y };
    for (row, value) in out[..SIDE * SIDE]
        .as_chunks_mut::<SIDE>()
        .0
        .iter_mut()
        .zip(values)
    {
        // SAFETY: complete word destination, ICL wrapper enabled. The L1
        // guard proves all intermediates/outputs fit, so no clamp is needed.
        unsafe { _mm512_storeu_si512(row.as_mut_ptr().cast(), value.0) };
    }
}

/// Keep the exceptional wide graph out of the packed kernel's hot code:
/// inlining both precisions nearly doubled its instruction-cache footprint.
#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn wide32<const RASTER: bool, const BOUNDED: bool>(
    words: &[Column; SIDE],
    zero_bound: u32,
    out: &mut [i16],
) -> bool {
    let mut scratch = [MaybeUninit::uninit(); SIDE * SIDE];
    // SAFETY: complete column outputs, enabled ISA, unchanged wide graph.
    let transposed = unsafe { wide_words(words, &mut scratch) };
    if BOUNDED && generic::line_l1_within::<I32x16, SIDE>(transposed, zero_bound) {
        return false;
    }
    // SAFETY: initialized intermediate and validated full output.
    unsafe { generic::forward_32_row_pass::<I32x16, RASTER>(transposed, out) };
    true
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn forward32<const RASTER: bool>(src: &[i16], stride: usize, out: &mut [i16]) {
    // SAFETY: dispatch checked the complete source block and enabled ICL.
    let words = transpose(unsafe { column_values::<false>(src, stride) });
    let l1 = line_l1(&words);
    if within(l1, NARROW_ROW_L1_LIMIT as u16) {
        // SAFETY: same exact L1 envelope used by the NEON forward family;
        // its outward-rounded interval tests prove every butterfly stage.
        unsafe { word_rows::<RASTER>(&words, out) };
    } else {
        // SAFETY: preserve every large row in the unchanged wide graph.
        unsafe { wide32::<RASTER, false>(&words, 0, out) };
    }
}

/// Exact low16-square forward: thirty-two word column lanes, then sixteen
/// guarded word row lanes (the same adapter as staged HEVC). Large row
/// inputs widen the already computed columns instead of recomputing them.
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
pub(super) unsafe fn low32<const RASTER: bool>(src: &[i16], stride: usize, out: &mut [i16]) {
    const KEEP: usize = SIDE / 2;
    // SAFETY: validated complete block, enabled ICL. Only the low sixteen
    // column frequencies are kept; transpose only those valid outputs.
    let words = transpose_low(&unsafe { column_values::<true>(src, stride) });
    let out = &mut out[..SIDE * SIDE];
    for row in out[..KEEP * SIDE].as_chunks_mut::<SIDE>().0 {
        row[KEEP..].fill(0);
    }
    out[KEEP * SIDE..].fill(0);
    if low_rows_fit(&words) {
        let x = words;
        let mut y = x;
        dct::fdct32_low16(&x, &mut y, cospi(COS_BITS), COS_BITS);
        let tile = core::array::from_fn(|k| y[k].0);
        let rows = if RASTER { transpose16(tile) } else { tile };
        for (dest, row) in out.as_chunks_mut::<SIDE>().0.iter_mut().zip(rows) {
            // SAFETY: complete sixteen-word prefix of each checked row;
            // the same L1 proof as the full graph admits every word stage.
            unsafe { _mm256_storeu_si256(dest.as_mut_ptr().cast(), row) };
        }
    } else {
        // SAFETY: the complete columns retain their exact word values;
        // this helper widens them before every row operation.
        unsafe { low32_wide::<RASTER>(&words, out) };
    }
}

#[cold]
#[inline(never)]
#[target_feature(enable = "avx512f,avx512bw,avx512vl")]
unsafe fn low32_wide<const RASTER: bool>(words: &[HalfColumn; SIDE], out: &mut [i16]) {
    const KEEP: usize = SIDE / 2;
    let mut x = [<I32x16 as I32Lanes>::splat(0); SIDE];
    let mut row = [0i16; KEEP];
    for (dest, value) in x.iter_mut().zip(words) {
        // SAFETY: complete temporary word row, enabled ICL. Loading it
        // through the existing lane facade preserves the wide fallback.
        unsafe { _mm256_storeu_si256(row.as_mut_ptr().cast(), value.0) };
        *dest = I32x16::load_i16(&row);
    }
    let mut y = x;
    dct::fdct32_low16(&x, &mut y, cospi(COS_BITS), COS_BITS);
    if RASTER {
        I32x16::store_transposed_sat16(
            &y[..KEEP],
            crate::codec_txfm::generic::as_uninit(out),
            SIDE,
        );
    } else {
        for (dest, row) in out.as_chunks_mut::<SIDE>().0.iter_mut().zip(&y[..KEEP]) {
            row.store_sat16(dest);
        }
    }
}

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(super) unsafe fn bounded32(src: &[i16], stride: usize, limit: u32, out: &mut [i16]) -> bool {
    // SAFETY: complete validated source block and enabled ICL.
    let words = transpose(unsafe { column_values::<false>(src, stride) });
    let l1 = line_l1(&words);
    let narrow = within(l1, NARROW_ROW_L1_LIMIT as u16);
    let zero_bound = limit.saturating_sub(super::FORWARD_ROW_ROUNDING[1]);
    // A saturated sum is always above this low bound. Larger bounds use
    // the original wide norm when the narrow guard did not admit the line.
    if (zero_bound <= NARROW_ROW_L1_LIMIT as u32 && within(l1, zero_bound as u16))
        || (zero_bound > NARROW_ROW_L1_LIMIT as u32 && narrow)
    {
        return false;
    }
    if narrow {
        // SAFETY: proved narrow row range; this call initializes all outputs.
        unsafe { word_rows::<true>(&words, out) };
    } else {
        // SAFETY: original norm/row graph, untouched output on zero.
        return unsafe { wide32::<true, true>(&words, zero_bound, out) };
    }
    true
}

/// HEVC/VVC staged 32-point rows, in the same row-major i32 intermediate
/// as the AVX2 sixteen-line adapter. The DC coefficient remains exact.
/// # Safety
/// ISA enabled; src covers 32 rows of 32 residuals in [-255,255], out 1024 slots.
#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vnni")]
pub(crate) unsafe fn hevc_rows32(src: &[i16], stride: usize, out: &mut [MaybeUninit<i32>]) {
    let mut loaded = [Column::splat(0); SIDE];
    for (r, value) in loaded.iter_mut().enumerate() {
        let row = &src[r * stride..][..SIDE];
        // SAFETY: one complete checked row, with the ISA enabled.
        *value = unsafe { Column(_mm512_loadu_si512(row.as_ptr().cast())) };
    }
    let mut input = transpose(loaded);
    let mut dc = Column::splat(0);
    for value in &mut input {
        value.0 = _mm512_slli_epi16::<{ INPUT_SHIFT as u32 }>(value.0);
        dc = dc.add(*value);
    }
    let mut output = [Column::splat(0); SIDE];
    dct::fdct32(&input, &mut output, cospi(COS_BITS), COS_BITS);
    let rows = transpose(output);
    let shift = _mm_cvtsi32_si128(SIDE.ilog2() as i32 + INPUT_SHIFT);
    let round = _mm512_set1_epi32(1 << (SIDE.ilog2() as i32 + INPUT_SHIFT - 1));
    let scale = _mm512_set1_epi32(HEVC_OUTPUT_SCALE);
    for (r, row) in rows.into_iter().enumerate() {
        let dest = &mut out[r * SIDE..][..SIDE];
        let values = [
            _mm512_castsi512_si256(row.0),
            _mm512_extracti64x4_epi64::<1>(row.0),
        ];
        for (v, dest) in values
            .into_iter()
            .zip(dest.as_chunks_mut::<{ SIDE / 2 }>().0)
        {
            let value = _mm512_sra_epi32(
                _mm512_add_epi32(_mm512_mullo_epi32(_mm512_cvtepi16_epi32(v), scale), round),
                shift,
            );
            // SAFETY: checked sixteen-dword destination, enabled ISA.
            unsafe { _mm512_storeu_si512(dest.as_mut_ptr().cast(), value) };
        }
    }
    let mut exact_dc = [0i16; SIDE];
    // SAFETY: complete word vector destination; enabled ISA.
    unsafe { _mm512_storeu_si512(exact_dc.as_mut_ptr().cast(), dc.0) };
    for (r, v) in exact_dc.into_iter().enumerate() {
        out[r * SIDE].write(i32::from(v));
    }
}

#[cfg(test)]
mod tests {
    use super::{Column, ForwardLanes, SIDE, transpose, transpose_low};
    use core::arch::x86_64::{_mm256_storeu_si256, _mm512_loadu_si512, _mm512_storeu_si512};

    #[test]
    fn transpose_low_matches_retained_coordinates() {
        if !crate::Isa::Avx512.is_supported() {
            return;
        }
        let source: [[i16; SIDE]; SIDE / 2] =
            core::array::from_fn(|r| core::array::from_fn(|c| (r * SIDE + c) as i16));
        // SAFETY: runtime ICL check; loads/stores cover complete vectors.
        // Poison discarded rows so accidental consumption is observable.
        let output = unsafe {
            let mut rows = [Column::splat(-12345); SIDE];
            for (dest, row) in rows.iter_mut().zip(source) {
                *dest = Column(_mm512_loadu_si512(row.as_ptr().cast()));
            }
            transpose_low(&rows).map(|value| {
                let mut row = [0i16; SIDE / 2];
                _mm256_storeu_si256(row.as_mut_ptr().cast(), value.0);
                row
            })
        };
        for (c, column) in output.iter().enumerate() {
            for (r, &value) in column.iter().enumerate() {
                assert_eq!(value, source[r][c]);
            }
        }
    }

    #[test]
    fn transpose32_matches_coordinates() {
        if !crate::Isa::Avx512.is_supported() {
            return;
        }
        let source: [[i16; SIDE]; SIDE] =
            core::array::from_fn(|r| core::array::from_fn(|c| (r * SIDE + c) as i16));
        // SAFETY: the complete ICL tier is available; each load/store
        // covers thirty-two words and transpose uses registers only.
        let output = unsafe {
            transpose(source.map(|row| Column(_mm512_loadu_si512(row.as_ptr().cast())))).map(|v| {
                let mut row = [0i16; SIDE];
                _mm512_storeu_si512(row.as_mut_ptr().cast(), v.0);
                row
            })
        };
        for (r, row) in output.iter().enumerate() {
            for (c, &value) in row.iter().enumerate() {
                assert_eq!(value, source[c][r]);
            }
        }
    }
}
