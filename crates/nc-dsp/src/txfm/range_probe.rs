//! Range proofs for the guarded 16-bit NEON transform graphs (test only).
//!
//! Each graph node is a linear combination of the 1-D transform's inputs
//! plus accumulated rounding error, so its magnitude is at most
//! `input L1 * largest coefficient + error`. Running the generated graphs on
//! [`Gain`], which carries outward-rounded Q40 intervals of every input's
//! coefficient, gives for each node the largest input L1 that keeps it in
//! `i16`. Each node also inherits the limits of its ancestors and of the
//! equal-weight pre-adds that the 16-bit `half_btf` computes first, so the
//! limit of the outputs covers the whole graph.
//!
//! The same intervals also bound a node by the largest input magnitude times
//! the sum of its coefficient magnitudes, plus the error. That gives an
//! amplitude limit: the largest input magnitude for which the graph fits
//! `i16` with no guard at all (16-point columns of 8-bit residuals).

use super::forward_lanes::ForwardLanes;
use super::inverse_lanes::InverseLanes;
use super::{COSPI_12, cospi, dct};

const SCALE: i128 = 1 << 40;
/// Most inputs of any probed graph.
const MAX_INPUTS: usize = 64;

#[derive(Clone, Copy)]
struct Gain {
    lo: [i128; MAX_INPUTS],
    hi: [i128; MAX_INPUTS],
    error: i128,
    /// Largest input L1 norm for which this node and its ancestors fit.
    limit: i128,
    /// Largest input magnitude for which this node and its ancestors fit.
    amplitude_limit: i128,
}

impl Gain {
    /// Input `i` itself: coefficient 1 on input `i`.
    fn input(i: usize) -> Self {
        let mut v = Self::zero();
        v.lo[i] = SCALE;
        v.hi[i] = SCALE;
        v
    }

    fn zero() -> Self {
        Self {
            lo: [0; MAX_INPUTS],
            hi: [0; MAX_INPUTS],
            error: 0,
            limit: i128::from(i16::MAX),
            amplitude_limit: i128::from(i16::MAX),
        }
    }

    /// Largest coefficient magnitude and the sum of coefficient magnitudes.
    fn gains(&self) -> (i128, i128) {
        let magnitudes = self
            .lo
            .iter()
            .zip(self.hi)
            .map(|(&lo, hi)| lo.abs().max(hi.abs()));
        (magnitudes.clone().max().unwrap(), magnitudes.sum())
    }

    fn record(mut self) -> Self {
        let (gain, sum) = self.gains();
        let headroom = i128::from(i16::MAX) * SCALE - self.error;
        if gain > 0 {
            self.limit = self.limit.min(headroom / gain);
            self.amplitude_limit = self.amplitude_limit.min(headroom / sum);
        }
        self
    }

    fn combine(a: Self, aw: i128, b: Self, bw: i128, bit: u32) -> Self {
        let divisor = 1i128 << bit;
        let endpoint = |i: usize, upper: bool| {
            let weighted = |v: Self, w: i128| {
                if (w >= 0) == upper {
                    v.hi[i] * w
                } else {
                    v.lo[i] * w
                }
            };
            let numerator = weighted(a, aw) + weighted(b, bw);
            if upper {
                -(-numerator).div_euclid(divisor)
            } else {
                numerator.div_euclid(divisor)
            }
        };
        let error = a.error * aw.abs() + b.error * bw.abs();
        Self {
            lo: core::array::from_fn(|i| endpoint(i, false)),
            hi: core::array::from_fn(|i| endpoint(i, true)),
            error: -(-error).div_euclid(divisor) + if bit > 0 { SCALE / 2 } else { 0 },
            limit: a.limit.min(b.limit),
            amplitude_limit: a.amplitude_limit.min(b.amplitude_limit),
        }
        .record()
    }

    /// `half_btf` as the 16-bit lanes compute it: equal-magnitude weights
    /// (below `2^bit`) add or subtract first, so that sum must fit too.
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        let mut result = Self::combine(a, i128::from(w0), b, i128::from(w1), bit);
        if (w0 == w1 || w0 == -w1) && w0.abs() < (1 << bit) {
            let preadd = Self::combine(a, 1, b, if w0 == w1 { 1 } else { -1 }, 0);
            result.limit = result.limit.min(preadd.limit);
            result.amplitude_limit = result.amplitude_limit.min(preadd.amplitude_limit);
        }
        result
    }
}

impl ForwardLanes for Gain {
    fn splat(v: i32) -> Self {
        assert_eq!(v, 0);
        Self::zero()
    }
    fn add(self, rhs: Self) -> Self {
        Self::combine(self, 1, rhs, 1, 0)
    }
    fn sub(self, rhs: Self) -> Self {
        Self::combine(self, 1, rhs, -1, 0)
    }
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        Gain::half_btf(w0, a, w1, b, bit)
    }
}

impl InverseLanes for Gain {
    fn splat(v: i32) -> Self {
        assert_eq!(v, 0);
        Self::zero()
    }
    fn add(self, rhs: Self) -> Self {
        Self::combine(self, 1, rhs, 1, 0)
    }
    fn sub(self, rhs: Self) -> Self {
        Self::combine(self, 1, rhs, -1, 0)
    }
    fn half_btf(w0: i32, a: Self, w1: i32, b: Self, bit: u32) -> Self {
        Gain::half_btf(w0, a, w1, b, bit)
    }
    fn mul_round(w: i32, a: Self, bit: u32) -> Self {
        Self::combine(a, i128::from(w), Self::zero(), 0, bit)
    }
    /// Within the proven limit every node fits 16 bits, so the clamp is the
    /// identity; the node's own limit was recorded when it was computed.
    fn clamp16(self) -> Self {
        self
    }
}

/// Run `probe` on a thread whose stack holds the expanded graph temporaries
/// (each [`Gain`] lane is 2 KiB).
fn with_large_stack(probe: impl FnOnce() + Send + 'static) {
    const STACK_BYTES: usize = 16 * 1024 * 1024;
    std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(probe)
        .unwrap()
        .join()
        .unwrap();
}

/// The L1 and amplitude limits of a graph's outputs, with inputs
/// `0..nonzero` live.
#[allow(clippy::large_stack_arrays)]
fn output_limits<const N: usize>(
    graph: impl Fn(&[Gain; N], &mut [Gain; N]),
    nonzero: usize,
) -> (i128, i128) {
    let input: [Gain; N] = core::array::from_fn(|i| {
        if i < nonzero {
            Gain::input(i)
        } else {
            Gain::zero()
        }
    });
    let mut output = [Gain::zero(); N];
    graph(&input, &mut output);
    (
        output.iter().map(|v| v.limit).min().unwrap(),
        output.iter().map(|v| v.amplitude_limit).min().unwrap(),
    )
}

/// The L1 limit of a graph's outputs, with inputs `0..nonzero` live.
fn output_limit<const N: usize>(
    graph: impl Fn(&[Gain; N], &mut [Gain; N]),
    nonzero: usize,
) -> i128 {
    output_limits(graph, nonzero).0
}

#[test]
fn forward_row_graphs_fit_i16_for_l1_at_most_the_row_limit() {
    with_large_stack(|| {
        let limit16 = output_limit::<16>(|x, y| dct::fdct16(x, y, cospi(12), 12), 16);
        let limit32 = output_limit::<32>(|x, y| dct::fdct32(x, y, cospi(12), 12), 32);
        // Only the 32 retained outputs of the 64-point row are computed.
        let limit64 = output_limit::<64>(
            |x, y| {
                dct::fdct64_low32(x, y, cospi(10), 10);
                for unused in &mut y[32..] {
                    *unused = Gain::zero();
                }
            },
            64,
        );
        println!("ROW_L1_LIMIT N16 {limit16} N32 {limit32} N64 {limit64}");
        let required = i128::from(super::forward_lanes::NARROW_ROW_L1_LIMIT);
        assert!(limit16 >= required && limit32 >= required && limit64 >= required);
    });
}

#[test]
fn inverse_graphs_fit_i16_for_l1_at_most_the_inverse_limit() {
    with_large_stack(|| {
        let bit = 12;
        let limits = [
            (
                "idct8",
                output_limit::<8>(|x, y| dct::idct8(x, y, &COSPI_12, bit), 8),
            ),
            (
                "idct16",
                output_limit::<16>(|x, y| dct::idct16(x, y, &COSPI_12, bit), 16),
            ),
            (
                "idct32",
                output_limit::<32>(|x, y| dct::idct32(x, y, &COSPI_12, bit), 32),
            ),
            (
                "idct32_low8",
                output_limit::<32>(|x, y| dct::idct32_low8(x, y, &COSPI_12, bit), 8),
            ),
            (
                "idct32_low16",
                output_limit::<32>(|x, y| dct::idct32_low16(x, y, &COSPI_12, bit), 16),
            ),
            (
                "idct64",
                output_limit::<64>(|x, y| dct::idct64(x, y, &COSPI_12, bit), 32),
            ),
            (
                "idct64_low8",
                output_limit::<64>(|x, y| dct::idct64_low8(x, y, &COSPI_12, bit), 8),
            ),
            (
                "idct64_low16",
                output_limit::<64>(|x, y| dct::idct64_low16(x, y, &COSPI_12, bit), 16),
            ),
            (
                "idct64_low32",
                output_limit::<64>(|x, y| dct::idct64_low32(x, y, &COSPI_12, bit), 32),
            ),
        ];
        for (name, limit) in limits {
            println!("INVERSE_L1_LIMIT {name} {limit}");
            assert!(
                limit >= i128::from(super::inverse_lanes::NARROW_INVERSE_L1_LIMIT),
                "{name}: every node and equal-weight pre-add must fit"
            );
        }
    });
}

/// 16-point forward columns need no guard: 8-bit residuals shifted by 2
/// keep every node in `i16`. (Their rows do need one: the row graph only
/// fits for row inputs up to about 2047, while column outputs reach about
/// 2886, so rows use the L1 guard checked above.)
#[test]
fn forward_16_columns_fit_i16_for_8bit_residuals() {
    use super::forward_lanes::FORWARD_16_COLUMN_INPUT_MAX;
    with_large_stack(|| {
        let (_, limit) = output_limits::<16>(|x, y| dct::fdct16(x, y, cospi(13), 13), 16);
        println!("FORWARD16 column amplitude limit {limit}");
        assert!(limit >= FORWARD_16_COLUMN_INPUT_MAX);
    });
}

/// Bound the actual column outputs before checking the row graph, rather
/// than assuming independent full-range i16 row inputs.
#[test]
#[allow(
    clippy::large_stack_arrays,
    reason = "runs on the dedicated proof stack"
)]
fn forward_8_both_passes_fit_i16_for_8bit_residuals() {
    const SIDE: usize = 8;
    const INPUT_MAX: i128 = 255 << 2;
    const COS_BIT: u32 = 13;
    with_large_stack(|| {
        let input = core::array::from_fn(Gain::input);
        let mut columns = [Gain::zero(); SIDE];
        dct::fdct8(&input, &mut columns, cospi(COS_BIT), COS_BIT);
        let (_, amplitude_limit) =
            output_limits::<SIDE>(|x, y| dct::fdct8(x, y, cospi(COS_BIT), COS_BIT), SIDE);
        assert!(INPUT_MAX <= amplitude_limit);
        let row_input_max = columns
            .iter()
            .map(|v| {
                let (_, gain_sum) = v.gains();
                let column_bound = (INPUT_MAX * gain_sum + v.error + SCALE - 1) / SCALE;
                // AV1's between-pass rounded right shift by one.
                (column_bound + 1) / 2
            })
            .max()
            .unwrap();
        println!("FORWARD8 amplitude limit {amplitude_limit}, row input bound {row_input_max}");
        assert!(row_input_max <= amplitude_limit);
    });
}

/// The proven-zero bound of the bounded forwards
/// (`txfm::FORWARD_ROW_ROUNDING`): every row output has coefficient
/// magnitudes at most 1 and accumulated rounding error at most the bound.
#[test]
fn forward_row_outputs_are_bounded_by_l1_plus_rounding() {
    with_large_stack(|| {
        fn check<const N: usize>(graph: impl Fn(&[Gain; N], &mut [Gain; N]), bound: u32) {
            let input: [Gain; N] = core::array::from_fn(Gain::input);
            let mut out = [Gain::zero(); N];
            graph(&input, &mut out);
            for g in &out {
                let (gain, _) = g.gains();
                println!(
                    "FORWARD_ROW N{N} gain {} error {}",
                    gain as f64 / SCALE as f64,
                    g.error as f64 / SCALE as f64
                );
                assert!(gain <= SCALE, "row coefficient above 1");
                assert!(
                    g.error <= i128::from(bound) * SCALE,
                    "row rounding above the bound"
                );
            }
        }
        let [b16, b32] = super::FORWARD_ROW_ROUNDING;
        check::<16>(|x, y| dct::fdct16(x, y, cospi(12), 12), b16);
        check::<32>(|x, y| dct::fdct32(x, y, cospi(12), 12), b32);
    });
}

#[test]
fn typed_graphs_fit_i16_for_l1_at_most_their_limits() {
    use super::narrow_inverse::{ADST_L1_LIMIT, l1_limit};
    use crate::av1_inverse::Kernel1d;
    use crate::av1_inverse::one_d::{iadst8, iadst16, identity};
    with_large_stack(|| {
        let bit = 12;
        let limits = [
            (
                Kernel1d::Adst,
                8,
                output_limit::<8>(|x, y| iadst8(x, y, &COSPI_12, bit), 8),
            ),
            (
                Kernel1d::Adst,
                16,
                output_limit::<16>(|x, y| iadst16(x, y, &COSPI_12, bit), 16),
            ),
            (Kernel1d::Identity, 8, output_limit::<8>(identity, 8)),
            (Kernel1d::Identity, 16, output_limit::<16>(identity, 16)),
            (Kernel1d::Identity, 32, output_limit::<32>(identity, 32)),
        ];
        for (kind, n, limit) in limits {
            println!("TYPED_L1_LIMIT {kind:?}{n} {limit}");
            assert!(limit >= i128::from(l1_limit(kind, n)), "{kind:?}{n}");
        }
        assert_eq!(l1_limit(Kernel1d::FlipAdst, 16), ADST_L1_LIMIT);
    });
}
