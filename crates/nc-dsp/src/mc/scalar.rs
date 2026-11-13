//! Scalar reference prediction, written out per sample as the specs state
//! it: AV1 always runs both passes (an identity filter folds away exactly);
//! HEVC and H.264 follow their equations case by case.

use super::{
    Av1Family, BiFn, CompoundFns, H264_TAPS, HevcChromaTaps, HevcLumaTaps, McFns, PrepFn, Regular,
    SeparableTaps, Sharp, Smooth, TAPS, VvcAffineLumaTaps, VvcChromaTaps, VvcLumaTaps,
};

pub(crate) fn mc_fns() -> McFns {
    McFns {
        compound: CompoundFns {
            prep: [
                prep_family::<Regular>(),
                prep_family::<Smooth>(),
                prep_family::<Sharp>(),
            ],
            average: [
                av1_average::<4>,
                av1_average::<8>,
                av1_average::<16>,
                av1_average::<32>,
                av1_average::<64>,
            ],
            hevc_prep: hevc_prep_fns(),
            hevc_bi: hevc_bi_fns(),
        },
        av1: av1_family::<Regular>(),
        av1_smooth: av1_family::<Smooth>(),
        av1_sharp: av1_family::<Sharp>(),
        hevc_luma: separable_family::<HevcLumaTaps, 8>(),
        hevc_chroma: separable_family::<HevcChromaTaps, 4>(),
        h264_luma: [
            h264_luma::<4>,
            h264_luma::<8>,
            h264_luma::<16>,
            h264_luma::<32>,
            h264_luma::<64>,
        ],
        h264_chroma: [
            h264_chroma::<4>,
            h264_chroma::<8>,
            h264_chroma::<16>,
            h264_chroma::<32>,
            h264_chroma::<64>,
        ],
        vvc_luma: separable_family::<VvcLumaTaps, 8>(),
        vvc_affine_luma: separable_family::<VvcAffineLumaTaps, 8>(),
        vvc_chroma: separable_family::<VvcChromaTaps, 4>(),
        h264_phases: None,
        av1_mixed: None,
    }
}

/// The kernels of separable table `T` by width class.
fn separable_family<T: SeparableTaps<N>, const N: usize>() -> [super::ConvolveFn; 5] {
    [
        separable::<T, N, 4>,
        separable::<T, N, 8>,
        separable::<T, N, 16>,
        separable::<T, N, 32>,
        separable::<T, N, 64>,
    ]
}

/// `InterRound0` and `InterRound1` for 8-bit, non-compound prediction.
const ROUND0: u32 = 3;
const ROUND1: u32 = 11;
/// `InterRound1` for compound prediction, and `InterPostRound` of the
/// equal-weight average (AV1 §7.11.3.15: `Round2(p0 + p1, 1 + 4)`).
const ROUND1_COMPOUND: u32 = 7;
const AVERAGE_ROUND: u32 = 5;

fn round2(value: i32, bits: u32) -> i32 {
    (value + (1 << (bits - 1))) >> bits
}

/// The AV1 kernels of filter family `F` by width class.
fn av1_family<F: Av1Family>() -> [super::ConvolveFn; 5] {
    [
        av1_convolve::<4, F>,
        av1_convolve::<8, F>,
        av1_convolve::<16, F>,
        av1_convolve::<32, F>,
        av1_convolve::<64, F>,
    ]
}

fn av1_convolve<const W: usize, F: Av1Family>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    av1_passes::<W, F>(src, src_stride, h, (frac_x, frac_y), ROUND1, |r, c, v| {
        dst[r * dst_stride + c] = clip_u8(v);
    });
}

/// The AV1 compound-precision kernels of filter family `F` by width class.
fn prep_family<F: Av1Family>() -> [PrepFn; 5] {
    [
        av1_prep::<4, F>,
        av1_prep::<8, F>,
        av1_prep::<16, F>,
        av1_prep::<32, F>,
        av1_prep::<64, F>,
    ]
}

fn av1_prep<const W: usize, F: Av1Family>(
    src: &[u8],
    src_stride: usize,
    dst: &mut [i16],
    dst_stride: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    av1_passes::<W, F>(
        src,
        src_stride,
        h,
        (frac_x, frac_y),
        ROUND1_COMPOUND,
        |r, c, v| {
            dst[r * dst_stride + c] = v as i16;
        },
    );
}

fn av1_average<const W: usize>(
    p0: &[i16],
    p1: &[i16],
    src_stride: usize,
    dst: &mut [u8],
    dst_stride: usize,
    h: usize,
) {
    for r in 0..h {
        for c in 0..W {
            let at = r * src_stride + c;
            let sum = i32::from(p0[at]) + i32::from(p1[at]);
            dst[r * dst_stride + c] = clip_u8(round2(sum, AVERAGE_ROUND));
        }
    }
}

/// Both AV1 filter passes of a `W`-wide block (AV1 §7.11.3.4), the
/// vertical one rounded by `round1`, handing `store` each output.
fn av1_passes<const W: usize, F: Av1Family>(
    src: &[u8],
    src_stride: usize,
    h: usize,
    (frac_x, frac_y): (usize, usize),
    round1: u32,
    mut store: impl FnMut(usize, usize, i32),
) {
    let fx = F::taps(W, frac_x);
    let fy = F::taps(h, frac_y);
    let rows = h + TAPS - 1;
    let mut intermediate = vec![0i32; rows * W];
    for r in 0..rows {
        let row = &src[r * src_stride..];
        for c in 0..W {
            let sum: i32 = (0..TAPS)
                .map(|t| i32::from(fx[t]) * i32::from(row[c + t]))
                .sum();
            intermediate[r * W + c] = round2(sum, ROUND0);
        }
    }
    for r in 0..h {
        for c in 0..W {
            let sum: i32 = (0..TAPS)
                .map(|t| i32::from(fy[t]) * intermediate[(r + t) * W + c])
                .sum();
            store(r, c, round2(sum, round1));
        }
    }
}

fn clip_u8(value: i32) -> u8 {
    value.clamp(0, 255) as u8
}

/// HEVC uni-prediction rounding of a 14-bit sample (§8.5.3.3.4.2, 8-bit).
fn hevc_round(value: i32) -> u8 {
    clip_u8((value + 32) >> 6)
}

fn separable<T: SeparableTaps<N>, const N: usize, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    hevc_separable::<W, N>(
        src,
        ss,
        dst,
        ds,
        h,
        &T::TABLE[frac_x],
        &T::TABLE[frac_y],
        frac_x,
        frac_y,
    );
}

/// HEVC §8.5.3.3.3 at 8-bit depth (`shift1 = 0`, `shift2 = 6`) followed by
/// uni-prediction rounding.
fn hevc_separable<const W: usize, const N: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    fx: &[i16; N],
    fy: &[i16; N],
    frac_x: usize,
    frac_y: usize,
) {
    for r in 0..h {
        for c in 0..W {
            let value = hevc_sample(src, ss, (r, c), (fx, fy), (frac_x, frac_y));
            dst[r * ds + c] = hevc_round(value);
        }
    }
}

/// The HEVC compound-precision kernels (luma, chroma) by width class: the
/// scalar reference of every ISA's `CompoundFns::hevc_prep`.
pub(crate) fn hevc_prep_fns() -> [[PrepFn; 5]; 2] {
    [
        [
            hevc_prep::<HevcLumaTaps, 8, 4>,
            hevc_prep::<HevcLumaTaps, 8, 8>,
            hevc_prep::<HevcLumaTaps, 8, 16>,
            hevc_prep::<HevcLumaTaps, 8, 32>,
            hevc_prep::<HevcLumaTaps, 8, 64>,
        ],
        [
            hevc_prep::<HevcChromaTaps, 4, 4>,
            hevc_prep::<HevcChromaTaps, 4, 8>,
            hevc_prep::<HevcChromaTaps, 4, 16>,
            hevc_prep::<HevcChromaTaps, 4, 32>,
            hevc_prep::<HevcChromaTaps, 4, 64>,
        ],
    ]
}

/// The HEVC default bi-prediction kernels (luma, chroma) by width class:
/// the scalar reference of every ISA's `CompoundFns::hevc_bi`.
pub(crate) fn hevc_bi_fns() -> [[BiFn; 5]; 2] {
    [
        [
            hevc_bi::<HevcLumaTaps, 8, 4>,
            hevc_bi::<HevcLumaTaps, 8, 8>,
            hevc_bi::<HevcLumaTaps, 8, 16>,
            hevc_bi::<HevcLumaTaps, 8, 32>,
            hevc_bi::<HevcLumaTaps, 8, 64>,
        ],
        [
            hevc_bi::<HevcChromaTaps, 4, 4>,
            hevc_bi::<HevcChromaTaps, 4, 8>,
            hevc_bi::<HevcChromaTaps, 4, 16>,
            hevc_bi::<HevcChromaTaps, 4, 32>,
            hevc_bi::<HevcChromaTaps, 4, 64>,
        ],
    ]
}

/// HEVC default bi-prediction (eq. 8-251, 8-bit: `shift2 = 7`,
/// `offset2 = 64`): [`hevc_prep`] of the second list, then
/// `clamp((first + second + 64) >> 7)` with both predictions restored from
/// their centring.
#[allow(clippy::too_many_arguments)]
fn hevc_bi<T: SeparableTaps<N>, const N: usize, const W: usize>(
    src: &[u8],
    ss: usize,
    first: &[i16],
    fs: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let mut second = [[0i16; W]; MAX_PREP_ROWS];
    hevc_prep::<T, N, W>(src, ss, second.as_flattened_mut(), W, h, frac_x, frac_y);
    let offset = i32::from(super::HEVC_PREP_OFFSET);
    for (r, out) in dst.chunks_mut(ds).take(h).enumerate() {
        for (c, o) in out[..W].iter_mut().enumerate() {
            let sum = i32::from(first[r * fs + c]) + i32::from(second[r][c]) + 2 * offset;
            *o = clip_u8((sum + 64) >> 7);
        }
    }
}

/// HEVC `predSamplesLX` (§8.5.3.3.3, 8-bit), centered by 8192: samples before
/// weighted sample prediction, for bi-prediction and explicit weights.
fn hevc_prep<T: SeparableTaps<N>, const N: usize, const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [i16],
    ds: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let (fx, fy) = (&T::TABLE[frac_x], &T::TABLE[frac_y]);
    let centre = N / 2 - 1;
    // Unbiased 2-D overshoot can reach 33150. Center before narrowing.
    match (frac_x, frac_y) {
        (0, 0) => {
            for (r, out) in dst.chunks_mut(ds).take(h).enumerate() {
                let line = &src[(r + centre) * ss + centre..];
                for (o, &s) in out[..W].iter_mut().zip(line) {
                    *o = center_hevc_sample(i32::from(s) << 6);
                }
            }
        }
        (_, 0) => {
            for (r, out) in dst.chunks_mut(ds).take(h).enumerate() {
                let line = &src[(r + centre) * ss..];
                for (c, o) in out[..W].iter_mut().enumerate() {
                    *o = center_hevc_sample(taps_dot(
                        fx,
                        line[c..c + N].iter().map(|&s| i32::from(s)),
                    ));
                }
            }
        }
        (0, _) => {
            for (r, out) in dst.chunks_mut(ds).take(h).enumerate() {
                for (c, o) in out[..W].iter_mut().enumerate() {
                    let column = (0..N).map(|t| i32::from(src[(r + t) * ss + c + centre]));
                    *o = center_hevc_sample(taps_dot(fy, column));
                }
            }
        }
        _ => {
            let mut rows = [[0i32; W]; MAX_PREP_ROWS];
            for (r, row) in rows[..h + N - 1].iter_mut().enumerate() {
                let line = &src[r * ss..];
                for (c, sum) in row.iter_mut().enumerate() {
                    *sum = taps_dot(fx, line[c..c + N].iter().map(|&s| i32::from(s)));
                }
            }
            for (r, out) in dst.chunks_mut(ds).take(h).enumerate() {
                for (c, o) in out[..W].iter_mut().enumerate() {
                    *o = center_hevc_sample(
                        taps_dot(fy, rows[r..r + N].iter().map(|row| row[c])) >> 6,
                    );
                }
            }
        }
    }
}

/// Store the full HEVC filter range in signed lanes without losing overshoot.
#[inline]
fn center_hevc_sample(sample: i32) -> i16 {
    let centered = sample - i32::from(super::HEVC_PREP_OFFSET);
    debug_assert!((i32::from(i16::MIN)..=i32::from(i16::MAX)).contains(&centered));
    centered as i16
}

/// The filter `taps` applied to `samples`.
fn taps_dot<const N: usize>(taps: &[i16; N], samples: impl Iterator<Item = i32>) -> i32 {
    taps.iter()
        .zip(samples)
        .map(|(&t, s)| i32::from(t) * s)
        .sum()
}

/// Rows of horizontal sums a 2-D prep needs at most: a 64-row block and
/// the 8-tap filter's 7 extra rows.
const MAX_PREP_ROWS: usize = 64 + 7;

/// One 14-bit HEVC prediction sample at block position `(r, c)` of the
/// window `src` (equations 8-228 to 8-240 with `shift1 = 0`, `shift2 = 6`,
/// `shift3 = 6`).
fn hevc_sample<const N: usize>(
    src: &[u8],
    ss: usize,
    (r, c): (usize, usize),
    (fx, fy): (&[i16; N], &[i16; N]),
    (frac_x, frac_y): (usize, usize),
) -> i32 {
    let centre = N / 2 - 1;
    let px = |r: usize, c: usize| i32::from(src[r * ss + c]);
    let tap = |taps: &[i16; N], t: usize| i32::from(taps[t]);
    match (frac_x, frac_y) {
        // A whole-sample position: s << shift3.
        (0, 0) => px(r + centre, c + centre) << 6,
        (_, 0) => (0..N).map(|t| tap(fx, t) * px(r + centre, c + t)).sum(),
        (0, _) => (0..N).map(|t| tap(fy, t) * px(r + t, c + centre)).sum(),
        _ => {
            let column: i32 = (0..N)
                .map(|t| {
                    let row: i32 = (0..N).map(|u| tap(fx, u) * px(r + t, c + u)).sum();
                    tap(fy, t) * row
                })
                .sum();
            column >> 6
        }
    }
}

/// H.264 luma (§8.4.2.2.1): `src` starts 2 up and 2 left of the block.
fn h264_luma<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let px = |r: usize, c: usize| i32::from(src[r * ss + c]);
    let six =
        |f: &dyn Fn(usize) -> i32| -> i32 { (0..6).map(|t| i32::from(H264_TAPS[t]) * f(t)).sum() };
    let clip1 = |v: i32| v.clamp(0, 255);
    for r in 0..h {
        for c in 0..W {
            // Integer samples G (and its right / lower neighbours).
            let g = |dx: usize, dy: usize| px(r + 2 + dy, c + 2 + dx);
            // Horizontal half samples b (row offset dy) and vertical h (column dx).
            let b1 = |row: usize| six(&|t| px(row, c + t));
            let b = |dy: usize| clip1((b1(r + 2 + dy) + 16) >> 5);
            let hh = |dx: usize| clip1((six(&|t| px(r + t, c + 2 + dx)) + 16) >> 5);
            let j = || clip1((six(&|t| b1(r + t)) + 512) >> 10);
            let avg = |p: i32, q: i32| (p + q + 1) >> 1;
            let value = match (frac_x, frac_y) {
                (0, 0) => g(0, 0),
                (1, 0) => avg(g(0, 0), b(0)),
                (2, 0) => b(0),
                (3, 0) => avg(g(1, 0), b(0)),
                (0, 1) => avg(g(0, 0), hh(0)),
                (0, 2) => hh(0),
                (0, 3) => avg(g(0, 1), hh(0)),
                (1, 1) => avg(b(0), hh(0)),
                (3, 1) => avg(b(0), hh(1)),
                (1, 3) => avg(hh(0), b(1)),
                (3, 3) => avg(hh(1), b(1)),
                (2, 1) => avg(b(0), j()),
                (2, 3) => avg(j(), b(1)),
                (1, 2) => avg(hh(0), j()),
                (3, 2) => avg(j(), hh(1)),
                _ => j(),
            };
            dst[r * ds + c] = value as u8;
        }
    }
}

/// H.264 chroma (§8.4.2.2.2): bilinear at eighth-sample positions.
fn h264_chroma<const W: usize>(
    src: &[u8],
    ss: usize,
    dst: &mut [u8],
    ds: usize,
    h: usize,
    frac_x: usize,
    frac_y: usize,
) {
    let (dx, dy) = (frac_x as i32, frac_y as i32);
    let px = |r: usize, c: usize| i32::from(src[r * ss + c]);
    for r in 0..h {
        for c in 0..W {
            let sum = (8 - dx) * (8 - dy) * px(r, c)
                + dx * (8 - dy) * px(r, c + 1)
                + (8 - dx) * dy * px(r + 1, c)
                + dx * dy * px(r + 1, c + 1);
            dst[r * ds + c] = clip_u8((sum + 32) >> 6);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::mc_fns;
    use crate::mc::Filter;

    /// Every filter of every codec preserves a flat block at every
    /// sub-sample position (each filter's taps sum to its unit gain).
    #[test]
    fn flat_blocks_stay_flat() {
        let fns = mc_fns();
        let stride = 16 + 8;
        let window = vec![77u8; stride * (16 + 8)];
        for filter in [
            Filter::Av1,
            Filter::Av1Smooth,
            Filter::Av1Sharp,
            Filter::HevcLuma,
            Filter::HevcChroma,
            Filter::H264Luma,
            Filter::H264Chroma,
            Filter::VvcLuma,
            Filter::VvcAffineLuma,
            Filter::VvcChroma,
        ] {
            for fx in 0..filter.positions() {
                for fy in 0..filter.positions() {
                    let mut out = [0u8; 16 * 16];
                    let kernel = fns.table(filter)[2];
                    #[allow(unsafe_code)]
                    // SAFETY: the scalar table needs no CPU support and the
                    // window covers 16x16 plus every filter's taps.
                    unsafe {
                        kernel(&window, stride, &mut out, 16, 16, fx, fy);
                    }
                    assert!(out.iter().all(|&v| v == 77), "{filter:?} ({fx},{fy})");
                }
            }
        }
    }

    /// Whole-sample positions copy the sample under the block's corner.
    #[test]
    fn whole_sample_positions_copy() {
        let fns = mc_fns();
        let stride = 8 + 8;
        let window: Vec<u8> = (0..stride * (8 + 8)).map(|i| (i * 7 % 251) as u8).collect();
        for filter in [
            Filter::Av1,
            Filter::Av1Smooth,
            Filter::Av1Sharp,
            Filter::HevcLuma,
            Filter::HevcChroma,
            Filter::H264Luma,
            Filter::H264Chroma,
            Filter::VvcLuma,
            Filter::VvcAffineLuma,
            Filter::VvcChroma,
        ] {
            let mut out = [0u8; 64];
            #[allow(unsafe_code)]
            // SAFETY: as above.
            unsafe {
                (fns.table(filter)[1])(&window, stride, &mut out, 8, 8, 0, 0);
            }
            let b = filter.before();
            for r in 0..8 {
                assert_eq!(
                    &out[r * 8..r * 8 + 8],
                    &window[(r + b) * stride + b..(r + b) * stride + b + 8],
                    "{filter:?}"
                );
            }
        }
    }
}
