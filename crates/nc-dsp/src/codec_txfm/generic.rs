//! The HEVC and H.264 transforms and quantisers, generic over
//! [`I32Lanes`]: each 1-D pass transforms `V::LANES` lines at once, with
//! register tile transposes (`store_transposed`) between passes, as in the
//! AV1 drivers (`txfm::generic`).
//!
//! Two 2-D drivers cover every transform:
//! - [`rows_then_columns`] (HEVC forward, every H.264 transform): the input
//!   is transposed first so the row pass is lane-parallel, the row pass
//!   stores its output transposed back to row-major, and the column pass
//!   writes coefficient rows directly.
//! - [`columns_then_rows`] (HEVC inverse): the column pass is lane-parallel
//!   as loaded; its output is transposed for the row pass, whose output is
//!   transposed again into the row-major residual.
//!
//! Scratch buffers are uninitialised and fully written by the pass before
//! the one that reads them.
//!
//! The functions are `unsafe` only because `V`'s instructions need its ISA
//! (see the `simd` module docs): call them from a `#[target_feature]`
//! function of that ISA, or with the scalar `i32` lane type.

#![allow(clippy::inline_always)]
#![allow(unsafe_code)]

use core::marker::PhantomData;
use core::mem::MaybeUninit;

use super::{
    DCT32, DST4, H264Quantizer, HevcForwardMode, HevcForwardStaging, HevcQuantizer,
    QUANT_MF_4X4_POS, QUANT_MF_8X8_POS,
};
use crate::simd::I32Lanes;
use crate::txfm::generic::store_pass;

/// The largest block: 32x32.
pub(crate) const MAX_AREA: usize = MAX_SIDE * MAX_SIDE;
/// Largest transform side.
const MAX_SIDE: usize = 32;

/// A 1-D transform of `V::LANES` lines: `x[i]` holds element `i` of every
/// line, `y[k]` receives output `k`. Generic transforms implement it for
/// every lane type; an ISA module may implement its own for its lane type
/// (`avx2_paired`).
pub(crate) trait Line<V: I32Lanes> {
    fn apply(x: &[V], y: &mut [V]);

    /// As [`Line::apply`], but only outputs `0..N/2` are required (the
    /// others may hold anything): the column pass of a forward that keeps
    /// only its low frequencies ([`HevcForwardMode::low_frequencies`]).
    #[inline(always)]
    fn apply_low(x: &[V], y: &mut [V]) {
        Self::apply(x, y);
    }
}

/// Rounding (and clamping) after one pass.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pass {
    /// `round_shift` amount (0: none).
    pub shift: u32,
    /// Clamp the result to 16 bits.
    pub clamp: bool,
}

impl Pass {
    #[inline(always)]
    pub(super) fn apply<V: I32Lanes>(self, y: &mut [V]) {
        for lane in y {
            let shifted = lane.round_shift(self.shift);
            *lane = if self.clamp {
                shifted.clamp16()
            } else {
                shifted
            };
        }
    }
}

/// Input samples the drivers load: residuals (`i16`) or coefficients.
pub(crate) trait Element: Copy {
    fn load<V: I32Lanes>(src: &[Self]) -> V;
}

impl Element for i16 {
    #[inline(always)]
    fn load<V: I32Lanes>(src: &[Self]) -> V {
        V::load_i16(src)
    }
}

impl Element for i32 {
    #[inline(always)]
    fn load<V: I32Lanes>(src: &[Self]) -> V {
        V::load(src)
    }
}

/// A fully written scratch buffer, viewed as initialised.
///
/// # Safety
///
/// Every element of `buf[..len]` must have been written.
#[inline(always)]
pub(crate) unsafe fn written(buf: &[MaybeUninit<i32>], len: usize) -> &[i32] {
    debug_assert!(len <= buf.len());
    // SAFETY: the caller wrote `buf[..len]`; `MaybeUninit<i32>` has the
    // layout of `i32`.
    unsafe { core::slice::from_raw_parts(buf.as_ptr().cast::<i32>(), len) }
}

/// An initialised output slice viewed as possibly-uninitialised storage, so
/// `store_transposed` can write into it. Sound: writes through the view only
/// ever store initialised `i32`s.
#[inline(always)]
pub(crate) fn as_uninit(out: &mut [i32]) -> &mut [MaybeUninit<i32>] {
    // SAFETY: `MaybeUninit<i32>` has the layout of `i32`, the view borrows
    // `out` mutably for its lifetime, and nothing de-initialises elements.
    unsafe { core::slice::from_raw_parts_mut(out.as_mut_ptr().cast(), out.len()) }
}

/// 2-D transform, rows first: `input` is `N` rows at `stride`; `out`
/// (`N * N`) receives row-major output (`out[v * N + u]`, `v` from the
/// column pass).
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= N`.
#[inline(always)]
pub(crate) unsafe fn rows_then_columns<V, E, R, C, const N: usize>(
    input: &[E],
    stride: usize,
    first: Pass,
    second: Pass,
    out: &mut [i32],
) where
    V: I32Lanes,
    E: Element,
    R: Line<V>,
    C: Line<V>,
{
    // SAFETY: forwarded driver contract; no coefficient bound requested.
    unsafe {
        rows_then_columns_bounded::<V, E, R, C, N>(
            input,
            stride,
            first,
            second,
            out,
            None,
            HevcForwardMode::EXACT,
        )
    };
}

/// Every HEVC DCT matrix entry has magnitude <= 90 (verified against the
/// matrix in tests).
const MAX_DCT_BASIS: u64 = 90;

/// The largest absolute column sum of the row-pass output whose column
/// coefficients all provably have magnitude <= `limit`: a column dot
/// product has magnitude <= 90 * sum(abs(column)), the positive rounding
/// endpoint bounds both signs, and clamping can only reduce magnitude; the
/// exact rounded inequality rearranged once.
fn column_limit(second: Pass, limit: u32) -> u64 {
    let rounding = 1u64 << (second.shift - 1);
    ((u64::from(limit) << second.shift) + (1u64 << second.shift) - 1 - rounding) / MAX_DCT_BASIS
}

/// Whether every coefficient of the exact column pass over the row-pass
/// output `m` provably has magnitude <= `limit` (the column dot products'
/// bound, directly or around each column's mean). Staged columns take the
/// same decision: they approximate those coefficients. Only the first
/// `columns` columns are bounded (the others are not coded,
/// [`HevcForwardMode::low_frequencies`]). `sums` receives each bounded
/// column's absolute sum (the direct bound's input).
#[inline(always)]
pub(crate) fn proven_zero<V: I32Lanes, const N: usize>(
    m: &[i32],
    second: Pass,
    limit: u32,
    columns: usize,
    sums: &mut [i32; N],
) -> bool {
    let lanes = V::LANES;
    let column_limit = column_limit(second, limit);
    let mut centers = [0i32; N];
    for c0 in (0..columns).step_by(lanes) {
        let mut sum = V::splat(0);
        let mut signed = V::splat(0);
        for r in 0..N {
            let value = V::load(&m[r * N + c0..]);
            sum = sum.add(value.abs());
            signed = signed.add(value);
        }
        sum.store(&mut sums[c0..]);
        signed.round_shift(N.ilog2()).store(&mut centers[c0..]);
    }
    if sums[..columns]
        .iter()
        .all(|&sum| sum as u64 <= column_limit)
    {
        return true;
    }
    // Centering raises bodycam/crowd zero hits from34.0%/46.7% to
    //47.5%/53.1%; total P instructions fall another1.05%/0.17%.
    // The DC row is64 throughout: its rounded column result is the
    // rounded mean above. Other rows sum to zero, so subtracting any
    // integer center leaves their exact dot products unchanged.
    if centers[..columns]
        .iter()
        .all(|&dc| dc.unsigned_abs() <= limit)
    {
        let mut centered = [0i32; N];
        for c0 in (0..columns).step_by(lanes) {
            let center = V::load(&centers[c0..]);
            let mut sum = V::splat(0);
            for r in 0..N {
                sum = sum.add(V::load(&m[r * N + c0..]).sub(center).abs());
            }
            sum.store(&mut centered[c0..]);
        }
        if centered[..columns]
            .iter()
            .all(|&sum| sum as u64 <= column_limit)
        {
            return true;
        }
    }
    false
}

/// As `rows_then_columns`, optionally proving all coefficients quantize to
/// zero after the first pass. False leaves out untouched.
#[inline(always)]
unsafe fn rows_then_columns_bounded<V, E, R, C, const N: usize>(
    input: &[E],
    stride: usize,
    first: Pass,
    second: Pass,
    out: &mut [i32],
    zero_limit: Option<u32>,
    mode: HevcForwardMode,
) -> bool
where
    V: I32Lanes,
    E: Element,
    R: Line<V>,
    C: Line<V>,
{
    let lanes = V::LANES;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    // Pass 0: the input transposed, `t[c * N + r] = input[r][c]`.
    let mut t = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    for r0 in (0..N).step_by(lanes) {
        for c0 in (0..N).step_by(lanes) {
            for (i, lane) in x[..lanes].iter_mut().enumerate() {
                *lane = E::load(&input[(r0 + i) * stride + c0..]);
            }
            V::store_transposed(&x[..lanes], &mut t[c0 * N + r0..], N);
        }
    }
    // SAFETY: every (r0, c0) tile of the N x N buffer was written above.
    let t = unsafe { written(&t, N * N) };
    let mut m = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    for r0 in (0..N).step_by(lanes) {
        for (j, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&t[j * N + r0..]);
        }
        if mode.staging == HevcForwardStaging::Exact {
            R::apply(&x, &mut y);
            first.apply(&mut y);
        } else {
            staged_hevc_pass(&x, &mut y, first, ROW_INPUT_SHIFT);
        }
        // Row r0 + i's outputs become row r0 + i of `m`.
        store_pass::<V>(&y, r0, N, &mut m);
    }
    // SAFETY: rows r0..r0 + LANES of `m` were written for every r0.
    let m = unsafe { written(&m, N * N) };
    // SAFETY: every row of m was initialized above; ISA contract forwarded.
    unsafe { columns_bounded::<V, C, N>(m, second, out, zero_limit, mode) }
}

/// Columns per proven-zero group of `HevcForwardMode::zero_columns`: a
/// fixed decision granularity, independent of the SIMD lane count.
const ZERO_COLUMN_GROUP: usize = 8;

#[inline(always)]
unsafe fn columns_bounded<V: I32Lanes, C: Line<V>, const N: usize>(
    m: &[i32],
    second: Pass,
    out: &mut [i32],
    zero_limit: Option<u32>,
    mode: HevcForwardMode,
) -> bool {
    let staged = mode.staging == HevcForwardStaging::RowsAndColumns;
    // The columns (horizontal frequencies) and rows (vertical) computed.
    let kept = if mode.low_frequencies { N / 2 } else { N };
    let input_scale = if staged {
        column_input_scale(&m[..N * N])
    } else {
        0
    };
    let lanes = V::LANES;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    let mut sums = [0; N];
    if zero_limit.is_some_and(|limit| proven_zero::<V, N>(m, second, limit, kept, &mut sums)) {
        return false;
    }
    // Groups proven zero by the direct bound (`HevcForwardMode::
    // zero_columns`): on lowlight 360p hevc-fast 16x16 blocks past the
    // block bound, the upper eight columns in 57-85% (QP 36-42).
    let skip_limit = match zero_limit {
        Some(limit) if mode.zero_columns && !staged => column_limit(second, limit),
        _ => 0,
    };
    let mut zero_group = [false; N];
    for (group, sums) in zero_group
        .iter_mut()
        .zip(sums.as_chunks::<ZERO_COLUMN_GROUP>().0)
    {
        *group = skip_limit > 0 && sums.iter().all(|&sum| sum as u64 <= skip_limit);
    }
    for c0 in (0..N).step_by(lanes) {
        let all_zero = if lanes <= ZERO_COLUMN_GROUP {
            zero_group[c0 / ZERO_COLUMN_GROUP]
        } else {
            zero_group[c0 / ZERO_COLUMN_GROUP..(c0 + lanes).div_ceil(ZERO_COLUMN_GROUP)]
                .iter()
                .all(|&zero| zero)
        };
        if c0 >= kept || all_zero {
            for v in 0..N {
                out[v * N + c0..v * N + c0 + lanes].fill(0);
            }
            continue;
        }
        // Wide vectors may cover both a skipped and a live group. Compute
        // the live half with the same graph, then mask only the original
        // eight-column groups. Narrow ISAs compile away this extra work.
        let mut active = V::splat(-1);
        if lanes > ZERO_COLUMN_GROUP {
            let mut mask = [0; N];
            for (i, value) in mask[..lanes].iter_mut().enumerate() {
                *value = if zero_group[(c0 + i) / ZERO_COLUMN_GROUP] {
                    0
                } else {
                    -1
                };
            }
            active = V::load(&mask);
        }
        for (r, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&m[r * N + c0..]);
        }
        if staged {
            staged_hevc_pass(&x, &mut y, second, input_scale);
        } else if mode.low_frequencies {
            // Only the kept (low) vertical frequencies are stored below:
            // computing just those halves the odd outputs' products.
            C::apply_low(&x, &mut y);
            second.apply(&mut y[..kept]);
        } else {
            C::apply(&x, &mut y);
            second.apply(&mut y);
        }
        for (v, lane) in y.iter().enumerate() {
            if v < kept {
                if lanes > ZERO_COLUMN_GROUP {
                    active
                        .select(*lane, V::splat(0))
                        .store(&mut out[v * N + c0..]);
                } else {
                    lane.store(&mut out[v * N + c0..]);
                }
            } else {
                out[v * N + c0..v * N + c0 + lanes].fill(0);
            }
        }
    }
    true
}

/// Input scale of a staged row pass: residuals (`|r| <= 255`) times 4,
/// which reduces the staged graph's intermediate rounding noise.
pub(crate) const ROW_INPUT_SHIFT: i32 = 2;

/// Largest magnitude a staged pass's scaled input may have: the staged
/// rows' own input range (`4 · 255`), which keeps every butterfly stage of
/// a 32-point graph within 16 bits (`32 · 1020` plus rounding), so NEON can
/// run the graph on eight 16-bit lanes with the same results.
const STAGED_INPUT_LIMIT: i32 = 1020;
/// Largest input up-scale of a staged column pass (the rows' `2^2`).
const COLUMN_MAX_UPSCALE: i32 = 2;

/// The input scale `2^t` of a staged column pass over the row-pass output
/// `m` (clamped to [`STAGED_COLUMN_LIMIT`]): the largest `t <= 2` with
/// `max |m| · 2^t <= 1020`, one per block, so small residuals keep their
/// precision and large ones fit the graph.
pub(crate) fn column_input_scale(m: &[i32]) -> i32 {
    let peak = m
        .iter()
        .map(|v| v.unsigned_abs().min(STAGED_COLUMN_LIMIT as u32))
        .max()
        .unwrap_or(0) as i32;
    let scaled = |t: i32| {
        if t >= 0 {
            peak << t
        } else {
            (peak + (1 << (-t - 1))) >> -t
        }
    };
    let mut t = COLUMN_MAX_UPSCALE;
    while scaled(t) > STAGED_INPUT_LIMIT {
        t -= 1;
    }
    t
}

/// Largest row-pass output a staged column pass takes: the exact rows'
/// bound (L1 norm of an HEVC basis row times 255, shifted: 32640); staged
/// rows approximate it within 0.011% and are clamped to it.
pub(crate) const STAGED_COLUMN_LIMIT: i32 = 32640;

/// Approximate one HEVC forward pass with AV1's staged butterflies: AV1's
/// staged basis has a common HEVC factor 64·√2, approximated by 181/2
/// within 0.011%. Inputs are scaled by `2^input_shift` first (a negative
/// shift clamps to [`STAGED_COLUMN_LIMIT`] and rounds); the DC output stays
/// exact, which keeps the complete DC
/// row or column of the 2-D result exact.
#[inline(always)]
fn staged_hevc_pass<V: I32Lanes, const N: usize>(
    x: &[V; N],
    y: &mut [V; N],
    pass: Pass,
    input_shift: i32,
) {
    use crate::txfm::{cospi, dct};
    let mut dc = V::splat(0);
    let mut scaled = [V::splat(0); N];
    for (source, target) in x.iter().zip(&mut scaled) {
        dc = dc.add(*source);
        *target = if input_shift >= 0 {
            source.shl(input_shift as u32)
        } else {
            source
                .max(V::splat(-STAGED_COLUMN_LIMIT))
                .min(V::splat(STAGED_COLUMN_LIMIT))
                .round_shift(input_shift.unsigned_abs())
        };
    }
    match N {
        16 => dct::fdct16::<V>(
            scaled.as_slice().try_into().unwrap(),
            y.as_mut_slice().try_into().unwrap(),
            cospi(12),
            12,
        ),
        32 => dct::fdct32::<V>(
            scaled.as_slice().try_into().unwrap(),
            y.as_mut_slice().try_into().unwrap(),
            cospi(12),
            12,
        ),
        _ => unreachable!("staged HEVC passes require 16 or 32 points"),
    }
    let output_shift = (pass.shift as i32 + 1 + input_shift) as u32;
    for value in y.iter_mut() {
        *value = value.mul_scalar(181).round_shift(output_shift);
    }
    y[0] = dc.mul_scalar(64).round_shift(pass.shift);
    if pass.clamp {
        for value in y {
            *value = value.clamp16();
        }
    }
}

/// 2-D transform, columns first (HEVC inverse): `input` is row-major
/// `N * N` (`input[k * N + x]`), clamped to 16 bits on load; `out` (`N *
/// N`) receives the row-major result.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= N`.
#[inline(always)]
pub(crate) unsafe fn columns_then_rows<V, T, const N: usize>(
    input: &[i32],
    first: Pass,
    second: Pass,
    out: &mut [i32],
) where
    V: I32Lanes,
    T: Line<V>,
{
    let lanes = V::LANES;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    // `g[x * N + y]`: the column pass's output, column-major.
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    for x0 in (0..N).step_by(lanes) {
        for (k, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&input[k * N + x0..]).clamp16();
        }
        T::apply(&x, &mut y);
        first.apply(&mut y);
        store_pass::<V>(&y, x0, N, &mut g);
    }
    // SAFETY: rows x0..x0 + LANES of `g` were written for every x0.
    let g = unsafe { written(&g, N * N) };
    let out = as_uninit(&mut out[..N * N]);
    for y0 in (0..N).step_by(lanes) {
        for (column, lane) in x.iter_mut().enumerate() {
            *lane = V::load(&g[column * N + y0..]);
        }
        T::apply(&x, &mut y);
        second.apply(&mut y);
        store_pass::<V>(&y, y0, N, out);
    }
}

// ------------------------------------------------------------------ HEVC

/// Four weights share one NEON constant-vector load (`I32Lanes::mul_add4`).
/// The half-basis comes from the canonical table, with no arithmetic changes.
const HEVC_DOT_GROUP: usize = 4;
const HEVC_ODD_INPUTS: usize = DCT32.len() / 2;
const HEVC_ODD_WEIGHTS: [[[i32; HEVC_DOT_GROUP]; HEVC_ODD_INPUTS / HEVC_DOT_GROUP]; DCT32.len()] = {
    let mut weights = [[[0; HEVC_DOT_GROUP]; HEVC_ODD_INPUTS / HEVC_DOT_GROUP]; DCT32.len()];
    let mut row = 0;
    while row < DCT32.len() {
        let mut i = 0;
        while i < HEVC_ODD_INPUTS {
            weights[row][i / HEVC_DOT_GROUP][i % HEVC_DOT_GROUP] = DCT32[row][i] as i32;
            i += 1;
        }
        row += 1;
    }
    weights
};

/// The odd outputs of an `N = 2H`-point HEVC forward DCT from the
/// differences `o[i] = x[i] − x[N−1−i]`: `y[k] = Σ basis(k, i) · o[i]` for
/// odd `k`. Plain constant-bound loops over arrays, which the compiler
/// unrolls completely (iterator adaptors here were left as loops and cost
/// a third of the 32-point transform).
///
/// Only the first `outputs` odd outputs (`k < 2 · outputs`) are computed.
#[allow(
    clippy::needless_range_loop,
    reason = "constant indexed loops preserve forward-transform unrolling"
)]
#[inline(always)]
fn hevc_odd_forward<V: I32Lanes, const H: usize>(o: &[V; H], y: &mut [V], outputs: usize) {
    // Current callers have H=4,8,16. Keep the small/tail cases exact too.
    assert!(H > 0 && H <= HEVC_ODD_INPUTS);
    // Row `k` of the N-point matrix is row `k · 32 / N` of DCT32.
    let row_step = DCT32.len() / (2 * H);
    for j in 0..outputs {
        let k = 2 * j + 1;
        if H == HEVC_ODD_INPUTS {
            let basis = &HEVC_ODD_WEIGHTS[k * row_step];
            let mut acc = V::splat(0);
            for group in 0..H / HEVC_DOT_GROUP {
                let i = group * HEVC_DOT_GROUP;
                // Same wrapping sums, in input order. NEON loads four weights
                // once rather than loading and broadcasting each signed byte.
                acc = acc.mul_add4([o[i], o[i + 1], o[i + 2], o[i + 3]], &basis[group]);
            }
            y[k] = acc;
        } else {
            // Grouping shorter dots slightly slows the 16-point NEON kernel;
            // retain its original constant-folded products and accumulation.
            let basis = &DCT32[k * row_step];
            let mut acc = o[0].mul_scalar(basis[0].into());
            for i in 1..H {
                acc = acc.add(o[i].mul_scalar(basis[i].into()));
            }
            y[k] = acc;
        }
    }
}

/// HEVC 4-point forward DCT.
#[inline(always)]
fn hevc_fdct4<V: I32Lanes>(x: &[V], y: &mut [V]) {
    let (e0, e1) = (x[0].add(x[3]), x[1].add(x[2]));
    let (o0, o1) = (x[0].sub(x[3]), x[1].sub(x[2]));
    y[0] = e0.add(e1).shl(6);
    y[2] = e0.sub(e1).shl(6);
    y[1] = o0.mul_scalar(83).add(o1.mul_scalar(36));
    y[3] = o0.mul_scalar(36).sub(o1.mul_scalar(83));
}

/// HEVC 4-point inverse DCT.
#[inline(always)]
fn hevc_idct4<V: I32Lanes>(c: &[V], y: &mut [V]) {
    let e0 = c[0].add(c[2]).shl(6);
    let e1 = c[0].sub(c[2]).shl(6);
    let o0 = c[1].mul_scalar(83).add(c[3].mul_scalar(36));
    let o1 = c[1].mul_scalar(36).sub(c[3].mul_scalar(83));
    y[0] = e0.add(o0);
    y[1] = e1.add(o1);
    y[2] = e1.sub(o1);
    y[3] = e0.sub(o0);
}

/// Defines the `$n`-point HEVC DCTs by even/odd decomposition ("partial
/// butterfly"): the even outputs are the half-size DCT of `x[i] + x[N−1−i]`,
/// the odd outputs a matrix product with the differences; inversely, the
/// even coefficients give `E` through the half-size inverse, the odd ones
/// `O`, and `y[i] = E[i] + O[i]`, `y[N−1−i] = E[i] − O[i]`. Exact integer
/// arithmetic, so the result is the matrix product.
macro_rules! hevc_dct_level {
    ($fdct:ident, $idct:ident, $n:literal, $log2:literal, $half_f:ident, $half_i:ident) => {
        #[inline(always)]
        fn $fdct<V: I32Lanes>(x: &[V], y: &mut [V]) {
            hevc_fdct_split!(x, y, $n, $half_f, $n / 2);
        }

        #[inline(always)]
        fn $idct<V: I32Lanes>(c: &[V], y: &mut [V]) {
            const H: usize = $n / 2;
            let mut even_in = [V::splat(0); H];
            for (j, e) in even_in.iter_mut().enumerate() {
                *e = c[2 * j];
            }
            let mut even = [V::splat(0); H];
            $half_i(&even_in, &mut even);
            for i in 0..H {
                let mut odd = V::splat(0);
                for k in 0..H {
                    let basis = i32::from(DCT32[(2 * k + 1) << (5 - $log2)][i]);
                    odd = odd.add(c[2 * k + 1].mul_scalar(basis));
                }
                y[i] = even[i].add(odd);
                y[$n - 1 - i] = even[i].sub(odd);
            }
        }
    };
}

/// One even/odd level of an `$n`-point forward DCT on `$x` into `$y`: the
/// even outputs are `$half` (the half-size forward, called directly so it
/// inlines) of the sums, and the first `$odd_outputs` odd outputs come from
/// the differences.
macro_rules! hevc_fdct_split {
    ($x:expr, $y:expr, $n:literal, $half:ident, $odd_outputs:expr) => {{
        const H: usize = $n / 2;
        let (x, y) = ($x, $y);
        let mut even = [V::splat(0); H];
        let mut odd = [V::splat(0); H];
        for i in 0..H {
            even[i] = x[i].add(x[$n - 1 - i]);
            odd[i] = x[i].sub(x[$n - 1 - i]);
        }
        let mut even_out = [V::splat(0); H];
        $half(&even, &mut even_out);
        for (j, &e) in even_out.iter().enumerate() {
            y[2 * j] = e;
        }
        hevc_odd_forward::<V, H>(&odd, y, $odd_outputs);
    }};
}

hevc_dct_level!(hevc_fdct8, hevc_idct8, 8, 3, hevc_fdct4, hevc_idct4);
hevc_dct_level!(hevc_fdct16, hevc_idct16, 16, 4, hevc_fdct8, hevc_idct8);
hevc_dct_level!(hevc_fdct32, hevc_idct32, 32, 5, hevc_fdct16, hevc_idct16);

/// The low half (`y[0..8]`) of the 16-point forward DCT: the 8-point DCT
/// of the sums and four of the eight odd rows.
#[inline(always)]
fn hevc_fdct16_low<V: I32Lanes>(x: &[V], y: &mut [V]) {
    hevc_fdct_split!(x, y, 16, hevc_fdct8, 4);
}

/// The low half (`y[0..16]`) of the 32-point forward DCT, as
/// [`hevc_fdct16_low`] one level up.
#[inline(always)]
fn hevc_fdct32_low<V: I32Lanes>(x: &[V], y: &mut [V]) {
    hevc_fdct_split!(x, y, 32, hevc_fdct16_low, 8);
}

/// A forward and an inverse HEVC 1-D transform.
macro_rules! hevc_lines {
    ($($forward:ident, $inverse:ident, $f:ident, $f_low:ident, $i:ident);*) => {$(
        pub(crate) struct $forward;
        impl<V: I32Lanes> Line<V> for $forward {
            #[inline(always)]
            fn apply(x: &[V], y: &mut [V]) {
                $f(x, y);
            }

            #[inline(always)]
            fn apply_low(x: &[V], y: &mut [V]) {
                $f_low(x, y);
            }
        }
        pub(crate) struct $inverse;
        impl<V: I32Lanes> Line<V> for $inverse {
            #[inline(always)]
            fn apply(x: &[V], y: &mut [V]) {
                $i(x, y);
            }
        }
    )*};
}

// The 4- and 8-point forwards and the DST compute every output for the low
// half too: no low-frequency mode uses them.
hevc_lines!(HevcFdct4, HevcIdct4, hevc_fdct4, hevc_fdct4, hevc_idct4;
            HevcFdct8, HevcIdct8, hevc_fdct8, hevc_fdct8, hevc_idct8;
            HevcFdct16, HevcIdct16, hevc_fdct16, hevc_fdct16_low, hevc_idct16;
            HevcFdct32, HevcIdct32, hevc_fdct32, hevc_fdct32_low, hevc_idct32;
            HevcFdst4, HevcIdst4, hevc_fdst4, hevc_fdst4, hevc_idst4);

/// An inverse line transform that may skip trailing zero coefficients
/// (the generic bounded inverses below serve AVX2 and NEON; SIMD128 has its
/// own 16-bit-pair kernels, `simd128_inverse`).
pub(crate) trait BoundedLine {
    /// As [`Line::apply`], where `c[k]` is zero for every `k >= active`
    /// (their products are skipped; the result is the same).
    fn apply_bounded<V: I32Lanes>(c: &[V], y: &mut [V], active: usize);
}

/// The 4-point inverse with a zero bound: too small to gain from it.
#[inline(always)]
fn hevc_idct4_bounded<V: I32Lanes>(c: &[V], y: &mut [V], _active: usize) {
    hevc_idct4(c, y);
}

/// As `hevc_dct_level`'s inverses, with coefficients from `active` on known
/// to be zero: the even half gets the bound `ceil(active / 2)`, the odd
/// products stop at `floor(active / 2)`. Same integer sums without zero
/// terms, so the result is unchanged.
macro_rules! hevc_idct_bounded_level {
    ($name:ident, $line:ident, $n:literal, $log2:literal, $half:ident) => {
        #[inline(always)]
        fn $name<V: I32Lanes>(c: &[V], y: &mut [V], active: usize) {
            const H: usize = $n / 2;
            let mut even_in = [V::splat(0); H];
            for (j, e) in even_in.iter_mut().enumerate() {
                *e = c[2 * j];
            }
            let mut even = [V::splat(0); H];
            $half(&even_in, &mut even, active.div_ceil(2));
            let odd_active = (active / 2).min(H);
            for i in 0..H {
                let mut odd = V::splat(0);
                for k in 0..odd_active {
                    let basis = i32::from(DCT32[(2 * k + 1) << (5 - $log2)][i]);
                    odd = odd.add(c[2 * k + 1].mul_scalar(basis));
                }
                y[i] = even[i].add(odd);
                y[$n - 1 - i] = even[i].sub(odd);
            }
        }

        impl BoundedLine for $line {
            #[inline(always)]
            fn apply_bounded<V: I32Lanes>(c: &[V], y: &mut [V], active: usize) {
                $name(c, y, active);
            }
        }
    };
}

hevc_idct_bounded_level!(hevc_idct8_bounded, HevcIdct8, 8, 3, hevc_idct4_bounded);
hevc_idct_bounded_level!(hevc_idct16_bounded, HevcIdct16, 16, 4, hevc_idct8_bounded);
hevc_idct_bounded_level!(hevc_idct32_bounded, HevcIdct32, 32, 5, hevc_idct16_bounded);

impl BoundedLine for HevcIdct4 {
    #[inline(always)]
    fn apply_bounded<V: I32Lanes>(c: &[V], y: &mut [V], _active: usize) {
        hevc_idct4(c, y);
    }
}
impl BoundedLine for HevcIdst4 {
    #[inline(always)]
    fn apply_bounded<V: I32Lanes>(c: &[V], y: &mut [V], _active: usize) {
        hevc_idst4(c, y);
    }
}

/// HEVC 4-point forward DST: `y[k] = Σ DST4[k][j] · x[j]`.
#[inline(always)]
fn hevc_fdst4<V: I32Lanes>(x: &[V], y: &mut [V]) {
    for (k, row) in DST4.iter().enumerate() {
        y[k] = x[0]
            .mul_scalar(row[0])
            .add(x[1].mul_scalar(row[1]))
            .add(x[2].mul_scalar(row[2]))
            .add(x[3].mul_scalar(row[3]));
    }
}

/// HEVC 4-point inverse DST: `y[j] = Σ DST4[k][j] · c[k]`.
#[inline(always)]
fn hevc_idst4<V: I32Lanes>(c: &[V], y: &mut [V]) {
    for (j, out) in y.iter_mut().enumerate().take(4) {
        *out = c[0]
            .mul_scalar(DST4[0][j])
            .add(c[1].mul_scalar(DST4[1][j]))
            .add(c[2].mul_scalar(DST4[2][j]))
            .add(c[3].mul_scalar(DST4[3][j]));
    }
}

/// Forward DCT with an exact all-zero coefficient bound between passes.
///
/// # Safety
/// As `rows_then_columns`; V's lane count divides N.
#[inline(always)]
pub(crate) unsafe fn hevc_forward_bounded<
    V: I32Lanes,
    T: Line<V>,
    const N: usize,
    const LOG2: u32,
>(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    let (first, second) = hevc_forward_passes(LOG2);
    // SAFETY: forwarded contract, using the HEVC DCT matrices and shifts.
    unsafe {
        rows_then_columns_bounded::<V, i16, T, T, N>(
            residual,
            stride,
            first,
            second,
            out,
            Some(zero_limit),
            mode,
        )
    }
}

/// The column pass of a bounded HEVC forward (as [`hevc_forward_bounded`]
/// after its rows) over `m`, the row-major `N`x`N` output of a row pass
/// an ISA kernel ran itself.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(crate) unsafe fn hevc_forward_columns<V, T, const N: usize, const LOG2: u32>(
    m: &[i32],
    zero_limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool
where
    V: I32Lanes,
    T: Line<V>,
{
    let (_, second) = hevc_forward_passes(LOG2);
    // SAFETY: forwarded contract.
    unsafe { columns_bounded::<V, T, N>(m, second, out, Some(zero_limit), mode) }
}

/// Packed staged rows, followed by the 4-lane column pass (exact, or staged
/// for [`HevcForwardStaging::RowsAndColumns`]).
/// # Safety
/// NEON must be available; residual and output slices cover the N-square block.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub(crate) unsafe fn hevc_forward_neon_rows<
    T: Line<crate::simd::neon::I32x4>,
    const N: usize,
    const LOG2: u32,
>(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    let staged_columns = mode.staging == HevcForwardStaging::RowsAndColumns;
    let mut m = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    // SAFETY: checked caller slices; this function shares the NEON contract.
    unsafe { crate::txfm::neon_forward::hevc_rows::<N>(residual, stride, &mut m[..N * N]) };
    // SAFETY: packed rows initialized the complete N-square intermediate.
    let m = unsafe { written(&m, N * N) };
    let (_, second) = hevc_forward_passes(LOG2);
    if !staged_columns {
        // SAFETY: initialized block, supported ISA and checked output.
        return unsafe {
            columns_bounded::<crate::simd::neon::I32x4, T, N>(
                m,
                second,
                out,
                Some(zero_limit),
                mode,
            )
        };
    }
    let kept = if mode.low_frequencies { N / 2 } else { N };
    if proven_zero::<crate::simd::neon::I32x4, N>(m, second, zero_limit, kept, &mut [0; N]) {
        return false;
    }
    // SAFETY: checked output; this function shares the NEON contract.
    unsafe { crate::txfm::neon_forward::hevc_columns::<N>(m, second.shift, out) };
    if mode.low_frequencies {
        keep_low_frequencies::<N>(out);
    }
    true
}

/// SIMD128 packed staged rows (`txfm::simd128_forward::hevc_rows`),
/// followed by the 4-lane column pass (exact, or the generic staged columns
/// for [`HevcForwardStaging::RowsAndColumns`]).
/// # Safety
/// Residual and output slices cover the N-square block.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline(always)]
pub(crate) unsafe fn hevc_forward_simd128_rows<
    T: Line<crate::simd::simd128::I32x4>,
    const N: usize,
    const LOG2: u32,
>(
    residual: &[i16],
    stride: usize,
    zero_limit: u32,
    out: &mut [i32],
    mode: HevcForwardMode,
) -> bool {
    let mut m = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    // SAFETY: checked caller slices; SIMD128 is a compile-time feature.
    unsafe { crate::txfm::simd128_forward::hevc_rows::<N>(residual, stride, &mut m[..N * N]) };
    // SAFETY: packed rows initialized the complete N-square intermediate.
    let m = unsafe { written(&m, N * N) };
    let (_, second) = hevc_forward_passes(LOG2);
    // SAFETY: initialized block, supported ISA and checked output.
    unsafe {
        columns_bounded::<crate::simd::simd128::I32x4, T, N>(m, second, out, Some(zero_limit), mode)
    }
}

/// Zeroes all but the low `N/2 x N/2` coefficients of a row-major block
/// ([`HevcForwardMode::low_frequencies`]).
#[cfg(target_arch = "aarch64")]
fn keep_low_frequencies<const N: usize>(out: &mut [i32]) {
    for (v, row) in out[..N * N].as_chunks_mut::<N>().0.iter_mut().enumerate() {
        let from = if v < N / 2 { N / 2 } else { 0 };
        row[from..].fill(0);
    }
}

/// HM's forward shifts for a `2^log2` transform (8-bit video).
const fn hevc_forward_passes(log2: u32) -> (Pass, Pass) {
    (
        Pass {
            shift: log2 - 1,
            clamp: false,
        },
        Pass {
            shift: log2 + 6,
            clamp: true,
        },
    )
}

/// The normative inverse shifts: 7 with clamping, then `20 − BitDepth`.
pub(crate) const HEVC_INVERSE_PASSES: (Pass, Pass) = (
    Pass {
        shift: 7,
        clamp: true,
    },
    Pass {
        shift: 12,
        clamp: false,
    },
);

/// HEVC forward transform `T` of side `N = 2^LOG2`.
///
/// # Safety
///
/// As [`rows_then_columns`].
#[inline(always)]
pub(crate) unsafe fn hevc_forward<V: I32Lanes, T: Line<V>, const N: usize, const LOG2: u32>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32],
) {
    let (first, second) = hevc_forward_passes(LOG2);
    // SAFETY: forwarded contract.
    unsafe { rows_then_columns::<V, i16, T, T, N>(residual, stride, first, second, out) }
}

/// HEVC inverse transform `T` of side `N`.
///
/// # Safety
///
/// As [`columns_then_rows`].
#[inline(always)]
pub(crate) unsafe fn hevc_inverse<V: I32Lanes, T: Line<V>, const N: usize>(
    coeffs: &[i32],
    out: &mut [i32],
) {
    let (first, second) = HEVC_INVERSE_PASSES;
    // SAFETY: forwarded contract.
    unsafe { columns_then_rows::<V, T, N>(coeffs, first, second, out) }
}

/// The rows and columns of an `N`x`N` coefficient block up to its last
/// nonzero one: (1 + last nonzero row, 1 + last nonzero column), (0, 0) for
/// an all-zero block.
#[inline(always)]
pub(crate) fn nonzero_extent<const N: usize>(coeffs: &[i32]) -> (usize, usize) {
    let mut column_any = [0i32; MAX_SIDE];
    let mut rows = 0;
    for (k, row) in coeffs[..N * N].as_chunks::<N>().0.iter().enumerate() {
        let mut row_any = 0;
        for (any, &c) in column_any.iter_mut().zip(row) {
            *any |= c;
            row_any |= c;
        }
        if row_any != 0 {
            rows = k + 1;
        }
    }
    let columns = column_any[..N]
        .iter()
        .rposition(|&c| c != 0)
        .map_or(0, |x| x + 1);
    (rows, columns)
}

/// [`hevc_inverse`] skipping what trailing zero coefficients cannot change:
/// the column pass runs only over columns up to the last nonzero one
/// (rounded up to whole vectors; the others are all zero), with products
/// only up to the last nonzero row; the row pass then reads only those
/// columns. Encoder blocks are mostly sparse (low frequencies only). Exact:
/// it drops zero terms only.
///
/// # Safety
///
/// As [`hevc_inverse`].
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
pub(crate) unsafe fn hevc_inverse_bounded<V, T, const N: usize>(coeffs: &[i32], out: &mut [i32])
where
    V: I32Lanes,
    T: BoundedLine,
{
    let extent = nonzero_extent::<N>(coeffs);
    // SAFETY: same ISA and slice requirements as this function.
    unsafe { hevc_inverse_extent::<V, T, N>(coeffs, out, extent) }
}

/// The bounded inverse with an already measured nonzero extent, allowing
/// reconstruction to select its DC shortcut without scanning twice.
///
/// # Safety
/// As [`hevc_inverse_bounded`]; `extent` must come from `nonzero_extent`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
#[inline(always)]
pub(crate) unsafe fn hevc_inverse_extent<V: I32Lanes, T: BoundedLine, const N: usize>(
    coeffs: &[i32],
    out: &mut [i32],
    extent: (usize, usize),
) {
    let (rows, columns) = extent;
    if rows == 0 {
        out[..N * N].fill(0);
        return;
    }
    let (first, second) = HEVC_INVERSE_PASSES;
    let lanes = V::LANES;
    let columns = columns.div_ceil(lanes) * lanes;
    let mut x = [V::splat(0); N];
    let mut y = [V::splat(0); N];
    // `g[x * N + y]`: the column pass's output, column-major; columns from
    // `columns` on are zero and never read.
    let mut g = [MaybeUninit::<i32>::uninit(); MAX_AREA];
    for x0 in (0..columns).step_by(lanes) {
        for (k, lane) in x[..rows].iter_mut().enumerate() {
            *lane = V::load(&coeffs[k * N + x0..]).clamp16();
        }
        T::apply_bounded(&x, &mut y, rows);
        first.apply(&mut y);
        store_pass::<V>(&y, x0, N, &mut g);
    }
    // SAFETY: columns 0..columns of `g` were written above.
    let g = unsafe { written(&g, columns * N) };
    let out = as_uninit(&mut out[..N * N]);
    x.fill(V::splat(0));
    for y0 in (0..N).step_by(lanes) {
        for (column, lane) in x[..columns].iter_mut().enumerate() {
            *lane = V::load(&g[column * N + y0..]);
        }
        T::apply_bounded(&x, &mut y, columns);
        second.apply(&mut y);
        store_pass::<V>(&y, y0, N, out);
    }
}

/// Levels are clamped to `±2^16` before scaling: larger ones saturate the
/// result anyway (module docs).
pub(crate) const HEVC_DEQUANT_LEVEL_LIMIT: i32 = 1 << 16;

/// Scaled products are clamped to this magnitude before the shift, where
/// larger ones would saturate anyway: `2^(16 + bdShift) >> per`.
pub(crate) fn hevc_dequant_product_limit(q: &HevcQuantizer) -> i32 {
    (1 << (16 + q.bd_shift)) >> q.per
}

/// HEVC scaling (eq. 8-309, flat scaling factor):
/// `clamp16((((l · m) << per) + 2^(bd − 1)) >> bd)`, exact for every level:
/// `l` is first clamped to `±2^16` and `l · m` to `±2^(16 + bd) >> per`,
/// beyond which the result saturates either way.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs).
#[inline(always)]
pub(crate) unsafe fn hevc_dequantize<V: I32Lanes>(
    levels: &[i32],
    q: &HevcQuantizer,
    out: &mut [i32],
) {
    let lanes = V::LANES;
    let level_limit = HEVC_DEQUANT_LEVEL_LIMIT;
    let product_limit = hevc_dequant_product_limit(q);
    let done = levels.len() / lanes * lanes;
    for i in (0..done).step_by(lanes) {
        let l = V::load(&levels[i..])
            .max(V::splat(-level_limit))
            .min(V::splat(level_limit));
        let m = l
            .mul_scalar(q.level_scale)
            .max(V::splat(-product_limit))
            .min(V::splat(product_limit));
        m.shl(q.per)
            .round_shift(q.bd_shift)
            .clamp16()
            .store(&mut out[i..]);
    }
    if done < levels.len() {
        // SAFETY: the scalar lane type needs no ISA.
        unsafe { hevc_dequantize::<i32>(&levels[done..], q, &mut out[done..]) };
    }
}

// ----------------------------------------------------------------- H.264

/// H.264 4x4 forward core transform rows of `Cf`.
pub(crate) struct H264Fdct4;
impl<V: I32Lanes> Line<V> for H264Fdct4 {
    #[inline(always)]
    fn apply(x: &[V], y: &mut [V]) {
        let (s03, d03) = (x[0].add(x[3]), x[0].sub(x[3]));
        let (s12, d12) = (x[1].add(x[2]), x[1].sub(x[2]));
        y[0] = s03.add(s12);
        y[1] = d03.shl(1).add(d12);
        y[2] = s03.sub(s12);
        y[3] = d03.sub(d12.shl(1));
    }
}

/// H.264 4x4 inverse transform (eq. 8-338..8-353).
pub(crate) struct H264Idct4;
impl<V: I32Lanes> Line<V> for H264Idct4 {
    #[inline(always)]
    fn apply(c: &[V], y: &mut [V]) {
        let e0 = c[0].add(c[2]);
        let e1 = c[0].sub(c[2]);
        let e2 = c[1].sra(1).sub(c[3]);
        let e3 = c[1].add(c[3].sra(1));
        y[0] = e0.add(e3);
        y[1] = e1.add(e2);
        y[2] = e1.sub(e2);
        y[3] = e0.sub(e3);
    }
}

/// The 4-point Hadamard of H.264 eq. 8-320.
pub(crate) struct H264Hadamard4;
impl<V: I32Lanes> Line<V> for H264Hadamard4 {
    #[inline(always)]
    fn apply(x: &[V], y: &mut [V]) {
        let (s01, d01) = (x[0].add(x[1]), x[0].sub(x[1]));
        let (s23, d23) = (x[2].add(x[3]), x[2].sub(x[3]));
        y[0] = s01.add(s23);
        y[1] = s01.sub(s23);
        y[2] = d01.sub(d23);
        y[3] = d01.add(d23);
    }
}

/// H.264 8-point forward transform: the exact product with `F`, whose row
/// `k` is 8 times the inverse transform's basis vector `k`.
pub(crate) struct H264Fdct8;
impl<V: I32Lanes> Line<V> for H264Fdct8 {
    #[inline(always)]
    fn apply(x: &[V], y: &mut [V]) {
        let s: [V; 4] = [
            x[0].add(x[7]),
            x[1].add(x[6]),
            x[2].add(x[5]),
            x[3].add(x[4]),
        ];
        let d: [V; 4] = [
            x[0].sub(x[7]),
            x[1].sub(x[6]),
            x[2].sub(x[5]),
            x[3].sub(x[4]),
        ];
        let (e0, e1) = (s[0].add(s[3]), s[1].add(s[2]));
        let (e2, e3) = (s[0].sub(s[3]), s[1].sub(s[2]));
        y[0] = e0.add(e1).shl(3);
        y[4] = e0.sub(e1).shl(3);
        y[2] = e2.shl(3).add(e3.shl(2));
        y[6] = e2.shl(2).sub(e3.shl(3));
        y[1] = dot4(&d, [12, 10, 6, 3]);
        y[3] = dot4(&d, [10, -3, -12, -6]);
        y[5] = dot4(&d, [6, -12, 3, 10]);
        y[7] = dot4(&d, [3, -6, 10, -12]);
    }
}

/// `Σ w[i] · d[i]`.
#[inline(always)]
fn dot4<V: I32Lanes>(d: &[V; 4], w: [i32; 4]) -> V {
    d[0].mul_scalar(w[0])
        .add(d[1].mul_scalar(w[1]))
        .add(d[2].mul_scalar(w[2]))
        .add(d[3].mul_scalar(w[3]))
}

/// H.264 8-point inverse transform (eq. 8-355..8-370).
pub(crate) struct H264Idct8;
impl<V: I32Lanes> Line<V> for H264Idct8 {
    #[inline(always)]
    fn apply(d: &[V], y: &mut [V]) {
        let a0 = d[0].add(d[4]);
        let a4 = d[0].sub(d[4]);
        let a2 = d[2].sra(1).sub(d[6]);
        let a6 = d[2].add(d[6].sra(1));
        let (b0, b2, b4, b6) = (a0.add(a6), a4.add(a2), a4.sub(a2), a0.sub(a6));
        let a1 = d[5].sub(d[3]).sub(d[7]).sub(d[7].sra(1));
        let a3 = d[1].add(d[7]).sub(d[3]).sub(d[3].sra(1));
        let a5 = d[7].sub(d[1]).add(d[5]).add(d[5].sra(1));
        let a7 = d[3].add(d[5]).add(d[1]).add(d[1].sra(1));
        let b1 = a1.add(a7.sra(2));
        let b7 = a7.sub(a1.sra(2));
        let b3 = a3.add(a5.sra(2));
        let b5 = a3.sra(2).sub(a5);
        y[0] = b0.add(b7);
        y[1] = b2.add(b5);
        y[2] = b4.add(b3);
        y[3] = b6.add(b1);
        y[4] = b6.sub(b1);
        y[5] = b4.sub(b3);
        y[6] = b2.sub(b5);
        y[7] = b0.sub(b7);
    }
}

const NO_ROUNDING: Pass = Pass {
    shift: 0,
    clamp: false,
};
/// The final `(x + 32) >> 6` of the H.264 inverse transforms (and the
/// forward 8x8's `1/64` scale).
const ROUND_6: Pass = Pass {
    shift: 6,
    clamp: false,
};

/// H.264 forward 4x4 of every block of a region, blocks in raster order.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= 4`.
#[inline(always)]
pub(crate) unsafe fn h264_forward_4x4<V: I32Lanes>(
    residual: &[i16],
    stride: usize,
    blocks_wide: usize,
    out: &mut [[i32; 16]],
) {
    for (b, block) in out.iter_mut().enumerate() {
        let origin = (b / blocks_wide) * 4 * stride + (b % blocks_wide) * 4;
        // SAFETY: forwarded contract.
        unsafe {
            rows_then_columns::<V, i16, H264Fdct4, H264Fdct4, 4>(
                &residual[origin..],
                stride,
                NO_ROUNDING,
                NO_ROUNDING,
                block,
            );
        }
    }
}

/// A 4x4 transform `T` (rows, then columns, then `last`) of every block.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= 4`.
#[inline(always)]
pub(crate) unsafe fn h264_blocks_4x4<V: I32Lanes, T: Line<V>>(
    input: &[[i32; 16]],
    last: Pass,
    out: &mut [[i32; 16]],
) {
    for (block, result) in input.iter().zip(out.iter_mut()) {
        // SAFETY: forwarded contract.
        unsafe { rows_then_columns::<V, i32, T, T, 4>(block, 4, NO_ROUNDING, last, result) };
    }
}

/// H.264 inverse 4x4 of every block.
///
/// # Safety
///
/// As [`h264_blocks_4x4`].
#[inline(always)]
pub(crate) unsafe fn h264_inverse_4x4<V: I32Lanes>(input: &[[i32; 16]], out: &mut [[i32; 16]]) {
    // SAFETY: forwarded contract.
    unsafe { h264_blocks_4x4::<V, H264Idct4>(input, ROUND_6, out) }
}

/// H.264 4x4 Hadamard of every block.
///
/// # Safety
///
/// As [`h264_blocks_4x4`].
#[inline(always)]
pub(crate) unsafe fn h264_hadamard_4x4<V: I32Lanes>(input: &[[i32; 16]], out: &mut [[i32; 16]]) {
    // SAFETY: forwarded contract.
    unsafe { h264_blocks_4x4::<V, H264Hadamard4>(input, NO_ROUNDING, out) }
}

/// H.264 forward 8x8: `(F · X · Fᵀ + 32) >> 6`.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= 8`.
#[inline(always)]
pub(crate) unsafe fn h264_forward_8x8<V: I32Lanes>(
    residual: &[i16],
    stride: usize,
    out: &mut [i32; 64],
) {
    // SAFETY: forwarded contract.
    unsafe {
        rows_then_columns::<V, i16, H264Fdct8, H264Fdct8, 8>(
            residual,
            stride,
            NO_ROUNDING,
            ROUND_6,
            out,
        );
    }
}

/// H.264 inverse 8x8 (§8.5.13.2).
///
/// # Safety
///
/// As [`h264_forward_8x8`].
#[inline(always)]
pub(crate) unsafe fn h264_inverse_8x8<V: I32Lanes>(coeffs: &[i32; 64], out: &mut [i32; 64]) {
    // SAFETY: forwarded contract.
    unsafe {
        rows_then_columns::<V, i32, H264Idct8, H264Idct8, 8>(coeffs, 8, NO_ROUNDING, ROUND_6, out);
    }
}

/// H.264 scaling: `l · ls[i % ls.len()]`, then `<< shift` (`shift >= 0`)
/// or a rounding shift right by `-shift`.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs).
#[inline(always)]
pub(crate) unsafe fn h264_dequantize<V: I32Lanes>(
    levels: &[i32],
    ls: &[i32],
    shift: i32,
    out: &mut [i32],
) {
    let lanes = V::LANES;
    let period = ls.len();
    let done = if period.is_multiple_of(lanes) {
        levels.len() / lanes * lanes
    } else {
        0
    };
    for i in (0..done).step_by(lanes) {
        let d = V::load(&levels[i..]).mul(V::load(&ls[i % period..]));
        let d = if shift >= 0 {
            d.shl(shift.unsigned_abs())
        } else {
            d.round_shift(shift.unsigned_abs())
        };
        d.store(&mut out[i..]);
    }
    for i in done..levels.len() {
        let d = levels[i].wrapping_mul(ls[i % period]);
        out[i] = if shift >= 0 {
            d << shift
        } else {
            I32Lanes::round_shift(d, shift.unsigned_abs())
        };
    }
}

/// Add `value` to every sample of the `n` x `n` block of `dst` (rows
/// `stride` apart), clamped to 8 bits: the inverse of a block whose only
/// non-zero coefficient is its DC, which every H.264 inverse transform
/// spreads evenly (`(d + 32) >> 6` at every position, §8.5.12.2 and
/// §8.5.13.2 with all other inputs zero).
#[inline(always)]
pub(super) fn add_constant<V: I32Lanes>(value: i32, n: usize, dst: &mut [u8], stride: usize) {
    let lanes = V::splat(value);
    for r in 0..n {
        for c in (0..n).step_by(V::LANES) {
            lanes.add_to_u8(&mut dst[r * stride + c..]);
        }
    }
}

/// Add the `n` x `n` residual `block` (raster) to the block of `dst`,
/// clamped to 8 bits.
#[inline(always)]
fn add_block<V: I32Lanes>(block: &[i32], n: usize, dst: &mut [u8], stride: usize) {
    for r in 0..n {
        for c in (0..n).step_by(V::LANES) {
            V::load(&block[r * n + c..]).add_to_u8(&mut dst[r * stride + c..]);
        }
    }
}

/// H.264 residual reconstruction of 4x4 blocks in place: each block of
/// `levels` (raster order, `blocks_wide` per row of blocks) is scaled
/// (§8.5.12.1 as [`h264_dequantize`] with `ls` and `shift`), its position 0
/// replaced by `dc[b]` (the separately scaled DC of §8.5.10 / §8.5.11)
/// when `dc` is not empty, inverse transformed (§8.5.12.2) and added to its
/// 4x4 block of `dst`, which holds the prediction. A block with no
/// non-zero input leaves its samples as they are; one with only a DC adds
/// the DC's constant inverse, without a transform.
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= 4`.
#[inline(always)]
pub(crate) unsafe fn h264_add_inverse_4x4<V: I32Lanes>(
    levels: &[[i32; 16]],
    dc: &[i32],
    (ls, shift): (&[i32; 16], i32),
    blocks_wide: usize,
    dst: &mut [u8],
    stride: usize,
) {
    for (b, block) in levels.iter().enumerate() {
        let origin = (b / blocks_wide) * 4 * stride + (b % blocks_wide) * 4;
        let dst = &mut dst[origin..];
        let ac = block[1..].iter().fold(0, |any, &l| any | l);
        let dc_level = dc.get(b).copied();
        if ac == 0 {
            let d = dc_level.unwrap_or_else(|| scale_level(block[0], ls[0], shift));
            if d != 0 {
                add_constant::<V>(I32Lanes::round_shift(d, ROUND_6.shift), 4, dst, stride);
            }
            continue;
        }
        let mut scaled = [0i32; 16];
        // SAFETY: forwarded contract.
        unsafe {
            h264_dequantize::<V>(block, ls, shift, &mut scaled);
            if let Some(d) = dc_level {
                scaled[0] = d;
            }
            add_inverse_4x4::<V>(&scaled, dst, stride);
        }
    }
}

/// H.264 reconstruction of 4x4 blocks whose coefficients are already
/// scaled (a decoder's, scaled with any scaling matrix while parsing): as
/// [`h264_add_inverse_4x4`] with unit scale factors and no separate DCs,
/// without the scaling pass.
///
/// # Safety
///
/// As [`h264_add_inverse_4x4`].
#[inline(always)]
pub(crate) unsafe fn h264_add_scaled_4x4<V: I32Lanes>(
    coeffs: &[[i32; 16]],
    blocks_wide: usize,
    dst: &mut [u8],
    stride: usize,
) {
    for (b, block) in coeffs.iter().enumerate() {
        let origin = (b / blocks_wide) * 4 * stride + (b % blocks_wide) * 4;
        let dst = &mut dst[origin..];
        if block[1..].iter().fold(0, |any, &c| any | c) == 0 {
            if block[0] != 0 {
                add_constant::<V>(
                    I32Lanes::round_shift(block[0], ROUND_6.shift),
                    4,
                    dst,
                    stride,
                );
            }
            continue;
        }
        // SAFETY: forwarded contract.
        unsafe { add_inverse_4x4::<V>(block, dst, stride) };
    }
}

/// H.264 reconstruction of the 16 luma 4x4 blocks of a macroblock
/// (raster order, already scaled) whose coded and AC blocks the caller
/// knows: block `b` is skipped unless bit `b` of `coded` is set, gets its
/// DC's constant inverse unless bit `b` of `ac` is set too, else the full
/// inverse transform. As [`h264_add_scaled_4x4`] with four blocks per row,
/// without scanning the coefficients for zeros: a decoder knows which
/// blocks it parsed (openh264's reconstruction is driven by its non-zero
/// counts the same way).
///
/// # Safety
///
/// As [`h264_add_inverse_4x4`] for a 16x16 region of `dst`.
#[inline(always)]
pub(crate) unsafe fn h264_add_scaled_mb<V: I32Lanes>(
    coeffs: &[[i32; 16]; 16],
    (coded, ac): (u16, u16),
    dst: &mut [u8],
    stride: usize,
) {
    let mut remaining = coded;
    while remaining != 0 {
        let b = remaining.trailing_zeros() as usize;
        remaining &= remaining - 1;
        let origin = (b / 4) * 4 * stride + (b % 4) * 4;
        let dst = &mut dst[origin..];
        if (ac >> b) & 1 == 0 {
            add_constant::<V>(
                I32Lanes::round_shift(coeffs[b][0], ROUND_6.shift),
                4,
                dst,
                stride,
            );
        } else {
            // SAFETY: forwarded contract.
            unsafe { add_inverse_4x4::<V>(&coeffs[b], dst, stride) };
        }
    }
}

/// Inverse transform the scaled 4x4 block (§8.5.12.2) and add it to `dst`.
///
/// # Safety
///
/// As [`h264_add_inverse_4x4`].
#[inline(always)]
unsafe fn add_inverse_4x4<V: I32Lanes>(scaled: &[i32; 16], dst: &mut [u8], stride: usize) {
    let mut residual = [0i32; 16];
    // SAFETY: forwarded contract.
    unsafe {
        rows_then_columns::<V, i32, H264Idct4, H264Idct4, 4>(
            scaled,
            4,
            NO_ROUNDING,
            ROUND_6,
            &mut residual,
        );
    }
    add_block::<V>(&residual, 4, dst, stride);
}

/// H.264 residual reconstruction of an 8x8 block in place: `levels` scaled
/// (§8.5.13.1 as [`h264_dequantize`] with `ls` and `shift`), inverse
/// transformed (§8.5.13.2) and added to the 8x8 block of `dst`, which holds
/// the prediction. Zero and DC-only blocks as in [`h264_add_inverse_4x4`].
///
/// # Safety
///
/// `V`'s ISA must be available (module docs); `V::LANES <= 8`.
#[inline(always)]
pub(crate) unsafe fn h264_add_inverse_8x8<V: I32Lanes>(
    levels: &[i32; 64],
    (ls, shift): (&[i32; 64], i32),
    dst: &mut [u8],
    stride: usize,
) {
    if levels[1..].iter().fold(0, |any, &l| any | l) == 0 {
        let d = scale_level(levels[0], ls[0], shift);
        if d != 0 {
            add_constant::<V>(I32Lanes::round_shift(d, ROUND_6.shift), 8, dst, stride);
        }
        return;
    }
    let mut scaled = [0i32; 64];
    // SAFETY: forwarded contract.
    unsafe {
        h264_dequantize::<V>(levels, ls, shift, &mut scaled);
        add_inverse_8x8::<V>(&scaled, dst, stride);
    }
}

/// [`h264_add_scaled_4x4`] for one 8x8 block (§8.5.13.2).
///
/// # Safety
///
/// As [`h264_add_inverse_8x8`].
#[inline(always)]
pub(crate) unsafe fn h264_add_scaled_8x8<V: I32Lanes>(
    coeffs: &[i32; 64],
    dst: &mut [u8],
    stride: usize,
) {
    if coeffs[1..].iter().fold(0, |any, &c| any | c) == 0 {
        if coeffs[0] != 0 {
            add_constant::<V>(
                I32Lanes::round_shift(coeffs[0], ROUND_6.shift),
                8,
                dst,
                stride,
            );
        }
        return;
    }
    // SAFETY: forwarded contract.
    unsafe { add_inverse_8x8::<V>(coeffs, dst, stride) };
}

/// Inverse transform the scaled 8x8 block (§8.5.13.2) and add it to `dst`.
///
/// # Safety
///
/// As [`h264_add_inverse_8x8`].
#[inline(always)]
unsafe fn add_inverse_8x8<V: I32Lanes>(scaled: &[i32; 64], dst: &mut [u8], stride: usize) {
    let mut residual = [0i32; 64];
    // SAFETY: forwarded contract.
    unsafe {
        rows_then_columns::<V, i32, H264Idct8, H264Idct8, 8>(
            scaled,
            8,
            NO_ROUNDING,
            ROUND_6,
            &mut residual,
        );
    }
    add_block::<V>(&residual, 8, dst, stride);
}

/// One level of [`h264_dequantize`].
#[inline(always)]
fn scale_level(level: i32, ls: i32, shift: i32) -> i32 {
    let d = level.wrapping_mul(ls);
    if shift >= 0 {
        d << shift
    } else {
        I32Lanes::round_shift(d, shift.unsigned_abs())
    }
}

/// H.264 4:2:0 chroma residual of one component, transformed and
/// quantised: the four 4x4 forward transforms of the 8x8 `residual` (at
/// `stride`, blocks in raster order), the 2x2 Hadamard of their DCs (eq.
/// 8-328, which serves the forward transform too) quantised with one more
/// bit (as `quantize_dc`), and the AC levels in `ac` with each block's
/// position 0 zero (its DC is coded in `ChromaDCLevel`). Returns the DC
/// levels and the number of non-zero AC levels.
///
/// # Safety
///
/// `V4`'s and `V`'s ISAs must be available (module docs); `V4::LANES <= 4`.
/// `residual` covers 8 rows of 8 at `stride`.
#[inline(always)]
pub(crate) unsafe fn h264_forward_quantize_chroma<V4: I32Lanes, V: I32Lanes>(
    residual: &[i16],
    stride: usize,
    q: &H264Quantizer,
    ac: &mut [[i32; 4 * 4]; 4],
) -> ([i32; 4], u32) {
    // SAFETY: forwarded contract.
    unsafe { h264_forward_4x4::<V4>(residual, stride, 2, ac) };
    let c: [i32; 4] = core::array::from_fn(|b| ac[b][0]);
    let mut dc = [
        c[0] + c[1] + c[2] + c[3],
        c[0] - c[1] + c[2] - c[3],
        c[0] + c[1] - c[2] - c[3],
        c[0] - c[1] - c[2] + c[3],
    ];
    let (dc_qbits, dc_offset) = q.forward_terms(1);
    let mf = &QUANT_MF_4X4_POS[q.rem];
    // SAFETY: the scalar lane type needs no ISA.
    unsafe { quantize::<i32>(&mut dc, &mf[..1], dc_offset, dc_qbits, i32::MAX) };
    for block in ac.iter_mut() {
        block[0] = 0;
    }
    let (qbits, offset) = q.forward_terms(0);
    // SAFETY: forwarded contract.
    let nonzero = unsafe { quantize::<V>(ac.as_flattened_mut(), mf, offset, qbits, i32::MAX) };
    (dc, nonzero)
}

/// An ISA's H.264 8x8 forward transform, as a type so that drivers inline
/// it ([`h264_forward_quantize_mb`]).
pub(crate) trait H264Forward8x8 {
    /// The transform, as [`super::H264Forward8Fn`].
    ///
    /// # Safety
    ///
    /// As [`h264_forward_8x8`], with the implementation's ISA available.
    unsafe fn forward(residual: &[i16], stride: usize, out: &mut [i32; 64]);
}

/// [`h264_forward_8x8`] on `V` lanes.
pub(crate) struct LanesForward8x8<V>(PhantomData<V>);

impl<V: I32Lanes> H264Forward8x8 for LanesForward8x8<V> {
    #[inline(always)]
    unsafe fn forward(residual: &[i16], stride: usize, out: &mut [i32; 64]) {
        // SAFETY: forwarded contract.
        unsafe { h264_forward_8x8::<V>(residual, stride, out) }
    }
}

/// Four inter 8x8 blocks, retaining the caller's proven-zero shortcuts.
/// Resolve the ISA and quantisation terms once per macroblock.
///
/// # Safety
///
/// The ISAs of `Q` and `F` must be available. `residual` covers a 16x16
/// block at `stride`.
#[inline(always)]
pub(crate) unsafe fn h264_forward_quantize_mb<F: H264Forward8x8, Q: I32Lanes>(
    residual: &[i16],
    stride: usize,
    zero_mask: u8,
    q: &H264Quantizer,
    out: &mut [[i32; 64]; 4],
) -> [u32; 4] {
    let (qbits, offset) = q.forward_terms(1);
    let mf = &QUANT_MF_8X8_POS[q.rem];
    let mut counts = [0; 4];
    for (i, block) in out.iter_mut().enumerate() {
        if zero_mask & (1 << i) != 0 {
            block.fill(0);
            continue;
        }
        let start = (i / 2) * 8 * stride + (i % 2) * 8;
        // SAFETY: the caller covers all four quadrants and both ISAs.
        unsafe {
            F::forward(&residual[start..], stride, block);
            counts[i] = quantize::<Q>(block, mf, offset, qbits, i32::MAX);
        }
    }
    counts
}

// ------------------------------------------------------------ quantiser

/// Quantise raw HEVC coefficients into a separate level buffer, preserving
/// the originals needed by sign-data hiding without a preliminary copy.
pub(crate) unsafe fn hevc_quantize_from<V: I32Lanes>(
    coeffs: &[i32],
    q: &HevcQuantizer,
    out: &mut [i32],
) -> u32 {
    let scale = V::splat(q.scale);
    let offset = V::splat(q.offset);
    let maximum = V::splat(super::HEVC_MAX_LEVEL);
    let mut zeros = V::splat(0);
    let done = coeffs.len() / V::LANES * V::LANES;
    for (input, output) in coeffs[..done]
        .chunks_exact(V::LANES)
        .zip(out[..done].chunks_exact_mut(V::LANES))
    {
        let raw = V::load(input);
        let level = raw
            .abs()
            .mul(scale)
            .add(offset)
            .shr_logical(q.qbits)
            .min(maximum)
            .with_sign_of(raw);
        level.store(output);
        zeros = count_zero_lanes(zeros, level);
    }
    let mut nonzero = vector_nonzero(done, zeros);
    for i in done..coeffs.len() {
        let raw = coeffs[i];
        let magnitude = I32Lanes::shr_logical(
            raw.wrapping_abs()
                .wrapping_mul(q.scale)
                .wrapping_add(q.offset),
            q.qbits,
        );
        let level = I32Lanes::with_sign_of(Ord::min(magnitude, super::HEVC_MAX_LEVEL), raw);
        out[i] = level;
        nonzero += u32::from(level != 0);
    }
    nonzero
}

/// Maximum lanes used by these quantisers (AVX2 `I32x8`). The AVX-512
/// table retains these AVX2 entries; its wider transform lanes are separate.
const MAX_I32_LANES: usize = 8;

/// `zeros` plus one in each lane where `level` is zero. The quantisers count
/// zero levels per lane and reduce once per call: a horizontal count per
/// vector (NEON `addv` and a transfer to a general register) serialised the
/// loop on its running total.
#[inline(always)]
fn count_zero_lanes<V: I32Lanes>(zeros: V, level: V) -> V {
    zeros.sub(level.eq(V::splat(0)))
}

/// Nonzero levels among the first `done` (a whole number of vectors), from
/// the per-lane zero counts of [`count_zero_lanes`].
#[inline(always)]
fn vector_nonzero<V: I32Lanes>(done: usize, zeros: V) -> u32 {
    let mut lanes = [0i32; MAX_I32_LANES];
    zeros.store(&mut lanes);
    let zero_total: i32 = lanes[..V::LANES].iter().sum();
    done as u32 - zero_total as u32
}

/// Dead-zone quantisation shared by HEVC and H.264 (`ScaleQuantizeFn`).
///
/// # Safety
///
/// `V`'s ISA must be available (module docs).
#[inline(always)]
pub(crate) unsafe fn quantize<V: I32Lanes>(
    coeffs: &mut [i32],
    mf: &[i32],
    offset: i32,
    qbits: u32,
    max_level: i32,
) -> u32 {
    let lanes = V::LANES;
    let period = mf.len();
    let done = if period.is_multiple_of(lanes) {
        coeffs.len() / lanes * lanes
    } else {
        0
    };
    let offset_v = V::splat(offset);
    let max_v = V::splat(max_level);
    let mut zeros = V::splat(0);
    // One period of coefficients at a time, zipped with the factors in
    // whole vectors: no per-vector index arithmetic or bounds checks (the
    // indexed form cost ~37 instructions per 8 coefficients).
    for block in coeffs[..done].chunks_exact_mut(period) {
        for (c, m) in block.chunks_exact_mut(lanes).zip(mf.chunks_exact(lanes)) {
            let v = V::load(c);
            let magnitude = v
                .abs()
                .mul(V::load(m))
                .add(offset_v)
                .shr_logical(qbits)
                .min(max_v);
            let level = magnitude.with_sign_of(v);
            level.store(c);
            zeros = count_zero_lanes(zeros, level);
        }
    }
    let mut nonzero = vector_nonzero(done, zeros);
    for i in done..coeffs.len() {
        let c = coeffs[i];
        let magnitude = I32Lanes::shr_logical(
            c.wrapping_abs()
                .wrapping_mul(mf[i % period])
                .wrapping_add(offset),
            qbits,
        );
        let level = I32Lanes::with_sign_of(Ord::min(magnitude, max_level), c);
        coeffs[i] = level;
        nonzero += u32::from(level != 0);
    }
    nonzero
}

/// Bit `i` set where `levels[i]` is non-zero (at most 64 levels): the
/// scalar reference of `CodecTxfmFns::nonzero_mask`.
pub(crate) fn nonzero_mask(levels: &[i32]) -> u64 {
    levels
        .iter()
        .enumerate()
        .fold(0, |mask, (i, &level)| mask | (u64::from(level != 0) << i))
}

/// Defines one ISA's `CodecTxfmFns` constructor, `codec_txfm_fns`, and its
/// kernels: `$v4` lanes for 4-point transforms, `$v8` for the others,
/// `$v` for the quantisers, each kernel under the attributes given; the
/// 16- and 32-point HEVC forward lines are the generic butterflies unless
/// given (lines on `$v8`). The kernels' safety contract is the
/// `codec_txfm` module's.
macro_rules! codec_txfm_kernels {
    ($(#[$attr:meta])*; $v4:ty, $v8:ty, $v:ty; $fdct16:ty, $fdct32:ty) => {
        use $crate::codec_txfm::generic::{self as g, HevcFdct4, HevcFdct8, HevcFdst4,
            HevcIdct4, HevcIdct8, HevcIdct16, HevcIdct32, HevcIdst4};
        use $crate::codec_txfm::{CodecTxfmFns, HevcQuantizer};

        /// The transform table of this ISA.
        pub(crate) fn codec_txfm_fns() -> CodecTxfmFns {
            CodecTxfmFns {
                hevc_forward: [hevc_fdst4, hevc_fdct4, hevc_fdct8, hevc_fdct16, hevc_fdct32],
                vvc_forward_dct: $crate::codec_txfm::vvc_forward::forward,
                hevc_inverse: [hevc_idst4, hevc_idct4, hevc_idct8, hevc_idct16, hevc_idct32],
                hevc_inverse_add: [hevc_add_dst4, hevc_add_dct4, hevc_add_dct8, hevc_add_dct16, hevc_add_dct32],
                hevc_dequantize,
                hevc_quantize_from,
                hevc_forward_bounded: [hevc_forward4_bounded, hevc_forward8_bounded, hevc_forward16_bounded, hevc_forward32_bounded],
                h264_forward_4x4,
                h264_inverse_4x4,
                h264_hadamard_4x4,
                h264_forward_8x8,
                h264_forward_quantize_mb,
                h264_inverse_8x8,
                h264_dequantize,
                h264_add_inverse_4x4,
                h264_add_inverse_8x8,
                h264_add_scaled_4x4,
                h264_add_scaled_8x8,
                h264_add_scaled_mb,
                h264_forward_quantize_chroma,
                quantize,
                nonzero_mask,
            }
        }

        $(#[$attr])*
        unsafe fn nonzero_mask(levels: &[i32]) -> u64 {
            g::nonzero_mask(levels)
        }

        $crate::codec_txfm::generic::codec_txfm_kernels!(@hevc [$(#[$attr])*]
            hevc_fdst4, hevc_idst4, HevcFdst4, HevcIdst4, $v4, 4, 2);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@hevc [$(#[$attr])*]
            hevc_fdct4, hevc_idct4, HevcFdct4, HevcIdct4, $v4, 4, 2);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@hevc [$(#[$attr])*]
            hevc_fdct8, hevc_idct8, HevcFdct8, HevcIdct8, $v8, 8, 3);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@hevc [$(#[$attr])*]
            hevc_fdct16, hevc_idct16, $fdct16, HevcIdct16, $v8, 16, 4);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@hevc [$(#[$attr])*]
            hevc_fdct32, hevc_idct32, $fdct32, HevcIdct32, $v8, 32, 5);

        $crate::codec_txfm::generic::codec_txfm_kernels!(@add [$(#[$attr])*] hevc_add_dst4, $v4, HevcIdst4, 4, false);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@add [$(#[$attr])*] hevc_add_dct4, $v4, HevcIdct4, 4, true);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@add [$(#[$attr])*] hevc_add_dct8, $v8, HevcIdct8, 8, true);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@add [$(#[$attr])*] hevc_add_dct16, $v8, HevcIdct16, 16, true);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@add [$(#[$attr])*] hevc_add_dct32, $v8, HevcIdct32, 32, true);

        $crate::codec_txfm::generic::codec_txfm_kernels!(@bounded [$(#[$attr])*] hevc_forward4_bounded, $v4, HevcFdct4, 4, 2);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@bounded [$(#[$attr])*] hevc_forward8_bounded, $v8, HevcFdct8, 8, 3);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@bounded [$(#[$attr])*] hevc_forward16_bounded, $v8, $fdct16, 16, 4);
        $crate::codec_txfm::generic::codec_txfm_kernels!(@bounded [$(#[$attr])*] hevc_forward32_bounded, $v8, $fdct32, 32, 5);

        $(#[$attr])*
        unsafe fn hevc_dequantize(levels: &[i32], q: &HevcQuantizer, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe { g::hevc_dequantize::<$v>(levels, q, out) }
        }

        $(#[$attr])*
        unsafe fn h264_forward_4x4(
            residual: &[i16],
            stride: usize,
            blocks_wide: usize,
            out: &mut [[i32; 16]],
        ) {
            // SAFETY: see the module docs.
            unsafe { g::h264_forward_4x4::<$v4>(residual, stride, blocks_wide, out) }
        }

        $(#[$attr])*
        unsafe fn h264_inverse_4x4(input: &[[i32; 16]], out: &mut [[i32; 16]]) {
            // SAFETY: see the module docs.
            unsafe { g::h264_inverse_4x4::<$v4>(input, out) }
        }

        $(#[$attr])*
        unsafe fn h264_hadamard_4x4(input: &[[i32; 16]], out: &mut [[i32; 16]]) {
            // SAFETY: see the module docs.
            unsafe { g::h264_hadamard_4x4::<$v4>(input, out) }
        }

        $(#[$attr])*
        unsafe fn h264_forward_8x8(residual: &[i16], stride: usize, out: &mut [i32; 64]) {
            // SAFETY: see the module docs.
            unsafe { g::h264_forward_8x8::<$v8>(residual, stride, out) }
        }

        $(#[$attr])*
        unsafe fn h264_forward_quantize_mb(
            residual: &[i16],
            stride: usize,
            zero_mask: u8,
            q: &$crate::codec_txfm::H264Quantizer,
            out: &mut [[i32; 64]; 4],
        ) -> [u32; 4] {
            // SAFETY: see the module docs.
            unsafe {
                g::h264_forward_quantize_mb::<g::LanesForward8x8<$v8>, $v>(
                    residual, stride, zero_mask, q, out,
                )
            }
        }

        $(#[$attr])*
        unsafe fn h264_inverse_8x8(coeffs: &[i32; 64], out: &mut [i32; 64]) {
            // SAFETY: see the module docs.
            unsafe { g::h264_inverse_8x8::<$v8>(coeffs, out) }
        }

        $(#[$attr])*
        unsafe fn h264_add_inverse_4x4(
            levels: &[[i32; 16]],
            dc: &[i32],
            scale: (&[i32; 16], i32),
            blocks_wide: usize,
            dst: &mut [u8],
            stride: usize,
        ) {
            // SAFETY: see the module docs.
            unsafe { g::h264_add_inverse_4x4::<$v4>(levels, dc, scale, blocks_wide, dst, stride) }
        }

        $(#[$attr])*
        unsafe fn h264_forward_quantize_chroma(
            residual: &[i16],
            stride: usize,
            q: &$crate::codec_txfm::H264Quantizer,
            ac: &mut [[i32; 16]; 4],
        ) -> ([i32; 4], u32) {
            // SAFETY: see the module docs.
            unsafe { g::h264_forward_quantize_chroma::<$v4, $v>(residual, stride, q, ac) }
        }

        $(#[$attr])*
        unsafe fn h264_add_scaled_4x4(
            coeffs: &[[i32; 16]],
            blocks_wide: usize,
            dst: &mut [u8],
            stride: usize,
        ) {
            // SAFETY: see the module docs.
            unsafe { g::h264_add_scaled_4x4::<$v4>(coeffs, blocks_wide, dst, stride) }
        }

        $(#[$attr])*
        unsafe fn h264_add_scaled_mb(
            coeffs: &[[i32; 16]; 16],
            masks: (u16, u16),
            dst: &mut [u8],
            stride: usize,
        ) {
            // SAFETY: see the module docs.
            unsafe { g::h264_add_scaled_mb::<$v4>(coeffs, masks, dst, stride) }
        }

        $(#[$attr])*
        unsafe fn h264_add_scaled_8x8(coeffs: &[i32; 64], dst: &mut [u8], stride: usize) {
            // SAFETY: see the module docs.
            unsafe { g::h264_add_scaled_8x8::<$v8>(coeffs, dst, stride) }
        }

        $(#[$attr])*
        unsafe fn h264_add_inverse_8x8(
            levels: &[i32; 64],
            scale: (&[i32; 64], i32),
            dst: &mut [u8],
            stride: usize,
        ) {
            // SAFETY: see the module docs.
            unsafe { g::h264_add_inverse_8x8::<$v8>(levels, scale, dst, stride) }
        }

        $(#[$attr])*
        unsafe fn h264_dequantize(levels: &[i32], ls: &[i32], shift: i32, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe { g::h264_dequantize::<$v>(levels, ls, shift, out) }
        }

        $(#[$attr])*
        unsafe fn hevc_quantize_from(coeffs: &[i32], q: &HevcQuantizer, out: &mut [i32]) -> u32 {
            // SAFETY: see the module docs.
            unsafe { g::hevc_quantize_from::<$v>(coeffs, q, out) }
        }

        $(#[$attr])*
        unsafe fn quantize(
            coeffs: &mut [i32],
            mf: &[i32],
            offset: i32,
            qbits: u32,
            max_level: i32,
        ) -> u32 {
            // SAFETY: see the module docs.
            unsafe { g::quantize::<$v>(coeffs, mf, offset, qbits, max_level) }
        }
    };
    ($(#[$attr:meta])*; $v4:ty, $v8:ty, $v:ty) => {
        $crate::codec_txfm::generic::codec_txfm_kernels!($(#[$attr])*; $v4, $v8, $v;
            $crate::codec_txfm::generic::HevcFdct16, $crate::codec_txfm::generic::HevcFdct32);
    };
    (@add [$(#[$attr:meta])*] $name:ident, $lanes:ty, $transform:ty, $n:literal, $dct:literal) => {
        $(#[$attr])*
        unsafe fn $name(coeffs: &[i32], extent: (usize, usize), dst: &mut [u8], stride: usize) {
            // SAFETY: validated buffers and supported ISA, as module docs.
            unsafe { $crate::codec_txfm::inverse_add::inverse_add::<$lanes, $transform, $n>(coeffs, extent, dst, stride, $dct) }
        }
    };
    (@bounded [$(#[$attr:meta])*] $name:ident, $lanes:ty, $transform:ty, $n:literal, $log2:literal) => {
        $(#[$attr])*
        unsafe fn $name(residual: &[i16], stride: usize, zero_limit: u32, out: &mut [i32], mode: $crate::codec_txfm::HevcForwardMode) -> bool {
            // SAFETY: see module docs.
            unsafe { g::hevc_forward_bounded::<$lanes, $transform, $n, $log2>(residual, stride, zero_limit, out, mode) }
        }
    };
    (@hevc [$(#[$attr:meta])*]
     $forward:ident, $inverse:ident, $f:ty, $i:ty, $lanes:ty, $n:literal, $log2:literal) => {
        $(#[$attr])*
        unsafe fn $forward(residual: &[i16], stride: usize, out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe { g::hevc_forward::<$lanes, $f, $n, $log2>(residual, stride, out) }
        }

        $(#[$attr])*
        unsafe fn $inverse(coeffs: &[i32], out: &mut [i32]) {
            // SAFETY: see the module docs.
            unsafe { g::hevc_inverse::<$lanes, $i, $n>(coeffs, out) }
        }
    };
}

pub(crate) use codec_txfm_kernels;

/// Defines `install_bounded_inverse`, which puts this ISA's
/// [`hevc_inverse_bounded`] kernels (8, 16 and 32 points on `$v` lanes) in
/// a table. The scalar table keeps the plain [`hevc_inverse`]: it is the
/// oracle the bounded kernels are checked against.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
macro_rules! hevc_bounded_inverse_kernels {
    ($(#[$attr:meta])*; $v:ty) => {
        $(#[$attr])*
        unsafe fn hevc_idct8_bounded(coeffs: &[i32], out: &mut [i32]) {
            use $crate::codec_txfm::generic as g;
            // SAFETY: see the module docs.
            unsafe { g::hevc_inverse_bounded::<$v, g::HevcIdct8, 8>(coeffs, out) }
        }

        $(#[$attr])*
        unsafe fn hevc_idct16_bounded(coeffs: &[i32], out: &mut [i32]) {
            use $crate::codec_txfm::generic as g;
            // SAFETY: see the module docs.
            unsafe { g::hevc_inverse_bounded::<$v, g::HevcIdct16, 16>(coeffs, out) }
        }

        $(#[$attr])*
        unsafe fn hevc_idct32_bounded(coeffs: &[i32], out: &mut [i32]) {
            use $crate::codec_txfm::generic as g;
            // SAFETY: see the module docs.
            unsafe { g::hevc_inverse_bounded::<$v, g::HevcIdct32, 32>(coeffs, out) }
        }

        /// Install the bounded HEVC inverses (table indices 2..=4: 8, 16
        /// and 32 points).
        pub(crate) fn install_bounded_inverse(table: &mut $crate::codec_txfm::CodecTxfmFns) {
            table.hevc_inverse[2] = hevc_idct8_bounded;
            table.hevc_inverse[3] = hevc_idct16_bounded;
            table.hevc_inverse[4] = hevc_idct32_bounded;
        }
    };
}

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub(crate) use hevc_bounded_inverse_kernels;

#[cfg(test)]
mod forward_dot_tests {
    use super::{DCT32, hevc_odd_forward};

    /// Scalar checkasm shares this graph, so retain an independent matrix
    /// oracle for grouping, cutoffs, and the wrapping arithmetic contract.
    fn check<const H: usize>() {
        let values = [0, 1, -1, 32640, -32640, i32::MIN, i32::MAX, 65535];
        for rotation in 0..values.len() {
            let input = core::array::from_fn::<_, H, _>(|i| values[(i + rotation) % values.len()]);
            for outputs in 0..=H {
                let mut expected = [191; 32];
                for j in 0..outputs {
                    let k = 2 * j + 1;
                    expected[k] = input
                        .iter()
                        .zip(&DCT32[k * (32 / (2 * H))])
                        .fold(0i32, |sum, (&x, &weight)| {
                            sum.wrapping_add(x.wrapping_mul(i32::from(weight)))
                        });
                }
                let mut got = [191; 32];
                hevc_odd_forward::<i32, H>(&input, &mut got, outputs);
                assert_eq!(got, expected, "H={H} outputs={outputs} rotation={rotation}");
            }
        }
    }

    #[test]
    fn grouped_odd_outputs_match_independent_matrix() {
        check::<1>();
        check::<2>();
        check::<3>();
        check::<4>();
        check::<5>();
        check::<8>();
        check::<16>();
    }
}
