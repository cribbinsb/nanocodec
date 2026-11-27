//! H.266 general intra reconstruction adapters to the shared residual kernels.
use crate::{BlockSize, Dsp};
use nc_common::{BitDepth, PlaneViewMut, Sample};

/// The largest residual magnitude [`Dsp::add_residual`] takes: its 16-bit
/// add of an 8-bit sample must not wrap.
const ADD_RESIDUAL_LIMIT: i16 = i16::MAX - u8::MAX as i16;

/// H.266 joint `CbCr` inverse mixing (§8.7.2; mask 1: Cr coded, 2/3: Cb
/// coded) of a 16-bit inverse-transform output. Negation saturates, so
/// `CoeffMin` = −32768 becomes 32767 rather than the spec's 32768 (and
/// 16383 rather than 16384 after the halving): any magnitude above 255
/// saturates the 8-bit reconstruction alike, and LMCS chroma scaling
/// clips its input to −256..255 first (§8.7.5), so the picture is the
/// spec's.
/// # Panics
/// If the mask is invalid or outputs are short.
pub fn vvc_joint_chroma(coded: &[i16], mask: u8, negative: bool, cb: &mut [i16], cr: &mut [i16]) {
    assert!((1..=3).contains(&mask) && cb.len() >= coded.len() && cr.len() >= coded.len());
    for ((&value, cb), cr) in coded.iter().zip(cb).zip(cr) {
        let signed = if negative {
            value.saturating_neg()
        } else {
            value
        };
        (*cb, *cr) = match mask {
            1 => (signed >> 1, value),
            2 => (value, signed >> 1),
            3 => (value, signed),
            _ => unreachable!(),
        };
    }
}

/// H.266 inverse adaptive colour transform (§8.7.4.6, YCgCo-R to the
/// coded colour space) of one transform unit's residuals in place: each
/// input clipped to `BitDepth + 1` bits plus sign first (VTM
/// `colorSpaceConvert`), then `t = Y − (Cg >> 1)`, `Y = Cg + t`,
/// `Cb = t − (Co >> 1)`, `Cr = Co + Cb`. Up to 13 bits every value fits
/// `i16` (inputs within ±2^14, outputs within ±2^15).
/// # Panics
/// If the planes differ in length or `depth` is above 13 bits.
pub fn vvc_act_inverse(
    (y, cb, cr): (&mut [i16], &mut [i16], &mut [i16]),
    depth: nc_common::BitDepth,
) {
    assert!(y.len() == cb.len() && y.len() == cr.len());
    assert!(depth.bits() <= ACT_MAX_BIT_DEPTH);
    let limit = (1i32 << (depth.bits() + 1)) - 1;
    for ((y, cb), cr) in y.iter_mut().zip(cb.iter_mut()).zip(cr.iter_mut()) {
        let [luma, cg, co] = [*y, *cb, *cr].map(|v| i32::from(v).clamp(-limit - 1, limit));
        let t = luma - (cg >> 1);
        let blue = t - (co >> 1);
        *y = (cg + t) as i16;
        *cb = blue as i16;
        *cr = (co + blue) as i16;
    }
}

/// The deepest samples whose ACT residuals [`vvc_act_inverse`] holds in
/// `i16`.
const ACT_MAX_BIT_DEPTH: u32 = 13;

impl Dsp {
    /// Add an inverse VVC residual to a packed prediction. Rectangles whose
    /// edges are at least four use the existing dispatched residual adder
    /// when every value is within its contract (it is for transformed
    /// blocks); narrow ISP/chroma edges, and the rare transform-skip or
    /// joint-chroma residual at the 16-bit limits, take a scalar fallback.
    /// # Panics
    /// If dimensions are not powers of two in 1..=64 or buffers are short.
    pub fn vvc_add_residual(
        &self,
        (width, height): (usize, usize),
        prediction: &mut [u8],
        residual: &[i16],
    ) {
        let length = check_add_residual((width, height), prediction.len(), residual.len());
        let residual = &residual[..length];
        let dispatched = BlockSize::new(width, height).filter(|_| {
            residual
                .iter()
                .all(|r| (-ADD_RESIDUAL_LIMIT..=ADD_RESIDUAL_LIMIT).contains(r))
        });
        if let Some(size) = dispatched {
            let mut view = PlaneViewMut::new(prediction, width, width, height).unwrap();
            self.add_residual(size, &mut view, residual, width);
        } else {
            add_clipped(&mut prediction[..length], residual, BitDepth::EIGHT);
        }
    }

    /// Add an inverse VVC residual directly into a strided picture window.
    /// # Panics
    /// As [`Self::vvc_add_residual`].
    pub fn vvc_add_residual_into(&self, dst: &mut PlaneViewMut<'_, u8>, residual: &[i16]) {
        let (width, height) = (dst.width(), dst.height());
        let length = check_add_residual((width, height), width * height, residual.len());
        let residual = &residual[..length];
        let dispatched = BlockSize::new(width, height).filter(|_| {
            residual
                .iter()
                .all(|r| (-ADD_RESIDUAL_LIMIT..=ADD_RESIDUAL_LIMIT).contains(r))
        });
        if let Some(size) = dispatched {
            self.add_residual(size, dst, residual, width);
        } else {
            for (y, row_res) in residual.chunks_exact(width).enumerate() {
                add_clipped(dst.row_mut(y), row_res, BitDepth::EIGHT);
            }
        }
    }
}

/// Check the inputs of [`Dsp::vvc_add_residual`] and return the block's
/// sample count.
///
/// # Panics
/// If a dimension is not a power of two in 1..=64 or a buffer is short.
pub(crate) fn check_add_residual(
    (width, height): (usize, usize),
    prediction_len: usize,
    residual_len: usize,
) -> usize {
    let supported = |n: usize| n.is_power_of_two() && (1..=64).contains(&n);
    assert!(supported(width) && supported(height));
    let length = width * height;
    assert!(prediction_len >= length && residual_len >= length);
    length
}

/// `recSamples = Clip1(predSamples + resSamples)` (H.266 §8.7.5.1) at
/// `depth`, sample by sample: the scalar reference of the residual add.
pub(crate) fn add_clipped<T: Sample>(prediction: &mut [T], residual: &[i16], depth: BitDepth) {
    // Folds to the constant 8 in the 8-bit instance.
    let depth = depth.of::<T>();
    for (dst, &src) in prediction.iter_mut().zip(residual) {
        *dst = depth.clip_to(dst.to_i32() + i32::from(src));
    }
}

#[cfg(any(
    target_arch = "x86_64",
    target_arch = "aarch64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
#[allow(unsafe_code, clippy::inline_always)]
pub(crate) mod high_simd {
    use nc_common::BitDepth;

    use crate::simd::I16Lanes;

    #[inline(always)]
    fn add_row<W: I16Lanes, H: I16Lanes>(
        prediction: &mut [u16],
        residual: &[i16],
        depth: BitDepth,
    ) {
        let len = prediction.len();
        let residual = &residual[..len];
        let max_w = W::splat(depth.max() as i16);
        let max_h = H::splat(depth.max() as i16);
        let zero_w = W::splat(0);
        let zero_h = H::splat(0);
        let mut x = 0;
        while x + W::LANES <= len {
            W::load_u16(&prediction[x..])
                .add_sat(W::load(&residual[x..]))
                .max(zero_w)
                .min(max_w)
                .store_u16(&mut prediction[x..]);
            x += W::LANES;
        }
        while x + H::LANES <= len {
            H::load_u16(&prediction[x..])
                .add_sat(H::load(&residual[x..]))
                .max(zero_h)
                .min(max_h)
                .store_u16(&mut prediction[x..]);
            x += H::LANES;
        }
        while x < len {
            prediction[x] = depth.clip_to(i32::from(prediction[x]) + i32::from(residual[x]));
            x += 1;
        }
    }

    #[inline(always)]
    fn add_strided<W: I16Lanes, H: I16Lanes>(
        residual: &[i16],
        n: usize,
        dst: &mut [u16],
        stride: usize,
        depth: BitDepth,
    ) {
        for (row, line) in residual[..n * n].chunks_exact(n).enumerate() {
            add_row::<W, H>(&mut dst[row * stride..row * stride + n], line, depth);
        }
    }

    macro_rules! isa_add_residual {
        ($mod_name:ident, $w:ty, $h:ty, $feature:literal) => {
            pub(crate) mod $mod_name {
                use nc_common::BitDepth;

                use super::{add_row, add_strided};

                #[target_feature(enable = $feature)]
                pub(crate) unsafe fn add_clipped(
                    prediction: &mut [u16],
                    residual: &[i16],
                    depth: BitDepth,
                ) {
                    add_row::<$w, $h>(prediction, residual, depth);
                }

                #[target_feature(enable = $feature)]
                pub(crate) unsafe fn add_residual_strided(
                    residual: &[i16],
                    n: usize,
                    dst: &mut [u16],
                    stride: usize,
                    depth: BitDepth,
                ) {
                    add_strided::<$w, $h>(residual, n, dst, stride, depth);
                }
            }
        };
    }

    #[cfg(target_arch = "x86_64")]
    isa_add_residual!(
        avx2,
        crate::simd::avx2::I16x16,
        crate::simd::avx2::I16x4,
        "avx2"
    );

    #[cfg(target_arch = "x86_64")]
    isa_add_residual!(
        avx512,
        crate::simd::avx512_i16::I16x32,
        crate::simd::avx2::I16x4,
        "avx512f,avx512bw,avx512vl"
    );

    #[cfg(target_arch = "aarch64")]
    isa_add_residual!(
        neon,
        crate::simd::neon::I16x8,
        crate::simd::neon::I16x4,
        "neon"
    );

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    isa_add_residual!(
        simd128,
        crate::simd::simd128::I16x8,
        crate::simd::simd128::I16x4,
        "simd128"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    /// The inverse ACT undoes the forward YCgCo-R (VTM `colorSpaceConvert`
    /// forward: `Co = R − B`, `t = B + (Co >> 1)`, `Cg = G − t`, `Y = t +
    /// (Cg >> 1)`) exactly, the coded planes being G, B, R.
    #[test]
    fn act_inverse_undoes_the_forward_transform() {
        let rgb = [(0, 0, 0), (1023, 0, 512), (-300, 700, -1023), (5, -6, 7)];
        for (g, b, r) in rgb {
            let co = r - b;
            let t = b + (co >> 1);
            let cg = g - t;
            let y = t + (cg >> 1);
            let (mut py, mut pcb, mut pcr) = ([y], [cg], [co]);
            vvc_act_inverse((&mut py, &mut pcb, &mut pcr), nc_common::BitDepth::TEN);
            assert_eq!((py[0], pcb[0], pcr[0]), (g, b, r));
        }
        // Inputs beyond BitDepth + 1 bits are clipped first.
        let (mut py, mut pcb, mut pcr) = ([5000i16], [0i16], [0i16]);
        vvc_act_inverse((&mut py, &mut pcb, &mut pcr), nc_common::BitDepth::EIGHT);
        assert_eq!((py[0], pcb[0], pcr[0]), (511, 511, 511));
    }
    #[test]
    fn joint_negative_odd_rounding() {
        let coded = [-7, -1, 0, 1, 7];
        let (mut cb, mut cr) = ([0; 5], [0; 5]);
        vvc_joint_chroma(&coded, 2, true, &mut cb, &mut cr);
        assert_eq!(cb, coded);
        assert_eq!(cr, [3, 0, 0, -1, -4]);
        vvc_joint_chroma(&coded, 1, false, &mut cb, &mut cr);
        assert_eq!(cb, [-4, -1, 0, 0, 3]);
        assert_eq!(cr, coded);
    }
    /// The 16-bit minimum is a legal inverse output (§8.7.4.1 clips to
    /// CoeffMin..CoeffMax); it once tripped an assertion. Its negation
    /// saturates (beyond any 8-bit reconstruction difference).
    #[test]
    fn joint_mixing_accepts_the_16_bit_minimum() {
        let coded = [-32768, 32767];
        let (mut cb, mut cr) = ([0; 2], [0; 2]);
        vvc_joint_chroma(&coded, 3, true, &mut cb, &mut cr);
        assert_eq!(cb, coded);
        assert_eq!(cr, [32767, -32767]);
        vvc_joint_chroma(&coded, 2, true, &mut cb, &mut cr);
        assert_eq!(cr, [16383, -16384]);
    }
    #[test]
    fn reconstruction_matches_wide_integer_clipping() {
        let dsp = Dsp::new();
        for width in [2, 4, 8, 16, 32] {
            for height in [2, 4, 8, 16, 32] {
                let length = width * height;
                let residual: Vec<_> = (0..length)
                    .map(|i| [-32768, -32513, -32512, -128, 0, 127, 32512, 32513, 32767][i % 9])
                    .collect();
                let mut actual: Vec<_> = (0..length).map(|i| (i * 53) as u8).collect();
                let expected: Vec<_> = actual
                    .iter()
                    .zip(&residual)
                    .map(|(&p, &r)| (i64::from(p) + i64::from(r)).clamp(0, 255) as u8)
                    .collect();
                dsp.vvc_add_residual((width, height), &mut actual, &residual);
                assert_eq!(actual, expected);
            }
        }
    }
}
