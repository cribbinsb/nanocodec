//! 64-bit FNV-1a hashing, the fingerprint the golden and decoder tests
//! compare samples and syntax against (and the C harnesses that produced
//! their golden files used).

/// The FNV-1a 64-bit offset basis: the hash of no input.
pub const FNV1A_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
/// The FNV-1a 64-bit prime.
const FNV1A_PRIME: u64 = 0x0000_0100_0000_01b3;

/// `hash` extended by `value` (one byte in the standard hash; some tests
/// fold whole values the same way).
#[must_use]
pub const fn fnv1a_step(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(FNV1A_PRIME)
}

/// The FNV-1a hash of `bytes`.
#[must_use]
pub fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes
        .into_iter()
        .fold(FNV1A_OFFSET, |hash, byte| fnv1a_step(hash, u64::from(byte)))
}

#[cfg(test)]
mod tests {
    use super::fnv1a;

    #[test]
    fn matches_the_reference_vectors() {
        assert_eq!(fnv1a([]), 0xcbf2_9ce4_8422_2325);
        assert_eq!(fnv1a(*b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(fnv1a(*b"foobar"), 0x8594_4171_f739_67e8);
    }
}
