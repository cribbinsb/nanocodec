//! A small seeded random source (`SplitMix64`) for test data and the
//! encoder's syntax-stress mode: reproducible from a seed, no external
//! dependency. Never used for a normal encoding decision.

/// `SplitMix64` generator (Steele, Lea & Flood, 2014). Not cryptographic.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// A generator seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform integer in `0..n` (`n` > 0); the tiny modulo bias does not
    /// matter for test data.
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// True with probability `chance`/256.
    pub fn chance(&mut self, chance: u64) -> bool {
        self.next_u64() & 0xFF < chance
    }

    /// A uniform integer in `-span..=span`.
    pub fn symmetric(&mut self, span: i32) -> i32 {
        self.below((2 * span + 1) as u64) as i32 - span
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn deterministic_and_bounded() {
        let (mut a, mut b) = (SplitMix64::new(7), SplitMix64::new(7));
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert!(a.below(13) < 13);
            assert!(a.symmetric(5).abs() <= 5);
            b.below(13);
            b.symmetric(5);
        }
    }
}
