//! Symbol rate estimation from adaptive CDFs, shared by the AV1 and AV2
//! backends.
//!
//! Both codecs store 15-bit inverse CDFs (`32768 − cdf`, last symbol's entry
//! 0). Rate-distortion decisions price far more symbols than are coded, so
//! a symbol's cost is read from a table rather than found by trial coding:
//! [`probability_bits`] maps a probability to bits with a 128-entry mantissa
//! table (libaom `av1_cost_symbol`, BSD-2-Clause, Alliance for Open Media),
//! and [`SymbolCosts`] caches every symbol's cost of one CDF state as
//! integers in 1/512 bit (libaom `av1_cost_tokens_from_cdf`, SVT-AV1's
//! per-context rate tables). The refresh cadence is lazy: a codec drops a
//! CDF's cache whenever the CDF adapts ([`SymbolCosts::invalidate`]) and the
//! next price recomputes it, so estimates always use the probabilities as
//! they stand and never re-sample unchanged CDFs. [`BitCounter`] adds such
//! costs up for one candidate.

/// Probability scale: CDFs are 15-bit fixed point (AV1 §4.10.7; AVM keeps
/// the same scale).
pub const PROB_TOP: u32 = 1 << 15;

/// Cost units per bit of [`SymbolCosts::cost_q9`] and [`BitCounter`]
/// (libaom `AV1_PROB_COST_SHIFT`).
pub const COST_UNITS_PER_BIT: u32 = 1 << 9;

/// Mantissa bits of [`probability_bits`]'s table index.
const MANTISSA_BITS: u32 = 7;
/// Bits below the leading one of a normalised 15-bit probability.
const NORMALISED_BITS: u32 = 14;
/// `−log2(m / 2^MANTISSA_BITS)` at the centre of each mantissa bucket
/// `m ∈ [2^7, 2^8)`, like libaom's `av1_prob_cost` table.
const MANTISSA_COST: [f32; 1 << MANTISSA_BITS] = mantissa_cost_table();

/// `−log2(p / 32768)` for a 15-bit probability `p`: normalise `p` to
/// [2^14, 2^15), whole bits from the shift, the rest from a 128-entry
/// table (libaom `av1_cost_symbol`). Mode decision calls this per candidate,
/// where a floating-point `log2` showed up in profiles.
#[must_use]
pub fn probability_bits(p: u32) -> f32 {
    let (shift, index) = bucket(p);
    shift as f32 + MANTISSA_COST[index]
}

/// The whole bits and mantissa table index of a 15-bit probability (clamped
/// to `1..PROB_TOP`): the normalising shift to [2^14, 2^15) and the top
/// [`MANTISSA_BITS`] bits below the leading one.
#[inline]
fn bucket(p: u32) -> (usize, usize) {
    let p = p.clamp(1, PROB_TOP - 1);
    let shift = p.leading_zeros() - (u32::BITS - (NORMALISED_BITS + 1));
    let normalised = p << shift;
    let index = (normalised >> (NORMALISED_BITS - MANTISSA_BITS)) as usize - MANTISSA_COST.len();
    (shift as usize, index)
}

/// [`probability_cost_q9`] for every (shift, mantissa index) of [`bucket`],
/// computed at compile time with the same `f32` arithmetic as
/// [`probability_bits`], so the lookup equals the float formula exactly
/// (rate estimates call it for every priced symbol, where the `f32`
/// rounding was a large share of their cost).
const COST_Q9: [[u16; 1 << MANTISSA_BITS]; NORMALISED_BITS as usize + 1] = {
    let mut table = [[0; 1 << MANTISSA_BITS]; NORMALISED_BITS as usize + 1];
    let mut shift = 0;
    while shift < table.len() {
        let mut index = 0;
        while index < MANTISSA_COST.len() {
            let bits = shift as f32 + MANTISSA_COST[index];
            table[shift][index] = (bits * COST_UNITS_PER_BIT as f32).round() as u16;
            index += 1;
        }
        shift += 1;
    }
    table
};

/// Cost in bits of symbol `s` under inverse CDF `icdf`.
#[must_use]
pub fn symbol_bits(icdf: &[u16], s: usize) -> f32 {
    let high = if s == 0 {
        PROB_TOP
    } else {
        u32::from(icdf[s - 1])
    };
    probability_bits(high - u32::from(icdf[s]))
}

/// [`probability_bits`] in 1/512 bit ([`COST_UNITS_PER_BIT`]), rounded.
// Inlined: rate estimates call it for every priced symbol.
#[inline]
#[must_use]
pub fn probability_cost_q9(p: u32) -> u32 {
    let (shift, index) = bucket(p);
    u32::from(COST_Q9[shift][index])
}

/// [`symbol_bits`] in 1/512 bit ([`COST_UNITS_PER_BIT`]), rounded: read
/// from the integer table, which equals the rounded float formula for every
/// probability (tested). The float form cost a `roundf` call per symbol,
/// a software routine in WebAssembly, on every cache refresh.
#[inline]
#[must_use]
pub fn symbol_cost_q9(icdf: &[u16], s: usize) -> u32 {
    let high = if s == 0 {
        PROB_TOP
    } else {
        u32::from(icdf[s - 1])
    };
    probability_cost_q9(high - u32::from(icdf[s]))
}

/// The cached symbol costs of one adaptive CDF of up to `N` symbols. The
/// owning CDF calls [`Self::invalidate`] whenever it adapts. Equality of the
/// owning CDF should ignore the cache.
#[derive(Debug, Clone, Copy)]
pub struct SymbolCosts<const N: usize> {
    /// Whether `costs` matches the owner's current probabilities.
    valid: bool,
    /// [`symbol_cost_q9`] of each symbol, when `valid`.
    costs: [u16; N],
}

impl<const N: usize> Default for SymbolCosts<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> SymbolCosts<N> {
    /// An empty cache, filled on the first price.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            valid: false,
            costs: [0; N],
        }
    }

    /// Cost of `symbol` under `icdf` (the owner's current inverse CDF) in
    /// 1/512 bit, computing every symbol's cost once per CDF state.
    // Inlined lookup, out-of-line refill: rate estimates price a cached
    // symbol far more often than they refill (the refill inlined into the
    // coefficient walk pushed the lookup out of line, +0.5 % instructions).
    #[inline]
    pub fn cost_q9(&mut self, icdf: &[u16; N], symbol: usize) -> u32 {
        if !self.valid {
            self.refill(icdf);
        }
        u32::from(self.costs[symbol])
    }

    /// Recompute every symbol's cost from `icdf`.
    #[cold]
    #[inline(never)]
    fn refill(&mut self, icdf: &[u16; N]) {
        for (s, cached) in self.costs.iter_mut().enumerate() {
            // At most 15 bits of 512 units: fits in 16 bits.
            *cached = symbol_cost_q9(icdf, s) as u16;
        }
        self.valid = true;
    }

    /// Drop the cached costs: the owner's probabilities changed.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
}

/// The running cost of a candidate's symbols, in integer 1/512 bit so the
/// sum has no floating-point dependency chain.
#[derive(Debug, Default, Clone, Copy)]
pub struct BitCounter {
    /// Cost so far, in 1/512 bit.
    cost_q9: u32,
}

impl BitCounter {
    /// Add a symbol cost in 1/512 bit.
    pub fn add_q9(&mut self, cost_q9: u32) {
        self.cost_q9 += cost_q9;
    }

    /// Add `bits` equiprobable (bypass) bits.
    pub fn add_bypass(&mut self, bits: u32) {
        self.cost_q9 += bits * COST_UNITS_PER_BIT;
    }

    /// The cost so far in 1/512 bit.
    #[must_use]
    pub fn q9(&self) -> u32 {
        self.cost_q9
    }

    /// The cost so far in bits.
    #[must_use]
    pub fn bits(&self) -> f32 {
        self.cost_q9 as f32 / COST_UNITS_PER_BIT as f32
    }
}

const fn mantissa_cost_table() -> [f32; 1 << MANTISSA_BITS] {
    let buckets = 1 << MANTISSA_BITS;
    let mut table = [0.0; 1 << MANTISSA_BITS];
    let mut i = 0;
    while i < buckets {
        let ratio = (buckets as f64 + i as f64 + 0.5) / buckets as f64;
        table[i] = (1.0 - log2_unit(ratio)) as f32;
        i += 1;
    }
    table
}

/// `log2(x)` for `x ∈ [1, 2)` by repeated squaring, one result bit per step.
const fn log2_unit(mut x: f64) -> f64 {
    const RESULT_BITS: u32 = 24;
    let mut result = 0.0;
    let mut bit = 0.5;
    let mut i = 0;
    while i < RESULT_BITS {
        x *= x;
        if x >= 2.0 {
            x /= 2.0;
            result += bit;
        }
        bit /= 2.0;
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cost_table_equals_the_float_formula_for_every_probability() {
        for p in 0..=PROB_TOP {
            let float = (probability_bits(p) * COST_UNITS_PER_BIT as f32).round() as u32;
            assert_eq!(probability_cost_q9(p), float, "p {p}");
        }
    }

    #[test]
    fn table_cost_tracks_log2() {
        for p in [1, 2, 100, 4096, 16384, 20000, 32767] {
            let exact = -(f64::from(p) / f64::from(PROB_TOP)).log2();
            assert!(
                (f64::from(probability_bits(p)) - exact).abs() < 0.01,
                "p={p}"
            );
        }
    }

    #[test]
    fn symbol_cost_equals_the_rounded_symbol_bits() {
        let icdf = [30000_u16, 20000, 9000, 1, 0];
        for s in 0..icdf.len() {
            let float = (symbol_bits(&icdf, s) * COST_UNITS_PER_BIT as f32).round() as u32;
            assert_eq!(symbol_cost_q9(&icdf, s), float, "symbol {s}");
        }
    }

    #[test]
    fn cache_follows_invalidation() {
        let mut icdf = [16384_u16, 0];
        let mut costs = SymbolCosts::new();
        let half = symbol_cost_q9(&icdf, 0);
        assert!(half.abs_diff(COST_UNITS_PER_BIT) < COST_UNITS_PER_BIT / 100);
        assert_eq!(costs.cost_q9(&icdf, 0), half);
        icdf[0] = 24576;
        assert_eq!(costs.cost_q9(&icdf, 0), half);
        costs.invalidate();
        assert_eq!(costs.cost_q9(&icdf, 0), symbol_cost_q9(&icdf, 0));
        assert_eq!(costs.cost_q9(&icdf, 1), symbol_cost_q9(&icdf, 1));
    }

    #[test]
    fn counter_sums_units() {
        let mut counter = BitCounter::default();
        counter.add_bypass(3);
        counter.add_q9(COST_UNITS_PER_BIT / 2);
        assert!((counter.bits() - 3.5).abs() < f32::EPSILON);
    }
}
