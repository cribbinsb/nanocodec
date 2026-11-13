//! AV1 symbol kernels (`nc_dsp::av1_symbol`): the compiled-in baseline SIMD
//! version (SSE2, NEON or SIMD128; no runtime dispatch) against the scalar
//! reference, for every alphabet size up to 16, random valid inverse CDFs
//! (including the extremes 0 and 32767), ranges, values and rates.
use crate::{check::CheckReport, rng::Rng};
use nc_dsp::av1_symbol::{av1_adapt_cdf, av1_find_symbol, reference};

/// Probabilities are 15-bit; an inverse CDF entry stays below this.
const PROB_TOP: usize = 1 << 15;
/// The adaptation rates AV1 uses (§8.2.6: 4..=7).
const RATES: std::ops::RangeInclusive<u32> = 4..=7;

pub(crate) fn check(iterations: usize, rng: &mut Rng, report: &mut CheckReport) {
    for _ in 0..iterations {
        check_size::<3>(rng, report);
        check_size::<4>(rng, report);
        check_size::<5>(rng, report);
        check_size::<6>(rng, report);
        check_size::<7>(rng, report);
        check_size::<8>(rng, report);
        check_size::<9>(rng, report);
        check_size::<10>(rng, report);
        check_size::<11>(rng, report);
        check_size::<13>(rng, report);
        check_size::<16>(rng, report);
    }
}

/// A random inverse CDF of `symbols` symbols: decreasing, below 32768,
/// zero from the last symbol on; a quarter of them pinned to the extremes.
fn random_icdf<const N: usize>(rng: &mut Rng, symbols: usize) -> [u16; N] {
    let mut icdf = [0u16; N];
    let mut previous = PROB_TOP - 1;
    let extreme = rng.below(4) == 0;
    for p in &mut icdf[..symbols - 1] {
        previous = if extreme {
            previous
        } else {
            rng.below(previous + 1)
        };
        *p = previous as u16;
    }
    icdf
}

fn check_size<const N: usize>(rng: &mut Rng, report: &mut CheckReport) {
    for symbols in 2..=N {
        let icdf: [u16; N] = random_icdf(rng, symbols);
        let range = (PROB_TOP + rng.below(PROB_TOP)) as u32;
        let value = match rng.below(4) {
            0 => 0,
            1 => range - 1,
            _ => rng.below(range as usize) as u32,
        };
        let fast = av1_find_symbol(&icdf, symbols, range, value);
        let slow = reference::find_symbol(&icdf, symbols, range, value);
        report.cases += 1;
        if fast != slow {
            report.failures.push(format!(
                "av1_find_symbol {icdf:?} symbols {symbols} range {range} value {value}: \
                 {fast:?} != {slow:?}"
            ));
        }
        let symbol = rng.below(symbols);
        let rate = *RATES.start() + rng.below(RATES.clone().count()) as u32;
        let (mut fast, mut slow) = (icdf, icdf);
        av1_adapt_cdf(&mut fast, symbol, symbols, rate);
        reference::adapt_cdf(&mut slow, symbol, symbols, rate);
        report.cases += 1;
        if fast != slow {
            report.failures.push(format!(
                "av1_adapt_cdf {icdf:?} symbol {symbol} rate {rate}: {fast:?} != {slow:?}"
            ));
        }
    }
}
