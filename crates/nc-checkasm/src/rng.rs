//! The checks' random test inputs: the shared `SplitMix64` source with
//! `usize` bounds and byte fills, reproducible from a seed.

use nc_common::{PlaneView, SplitMix64, fnv1a};

/// Seeded random source for test inputs. Not cryptographic.
#[derive(Debug, Clone)]
pub struct Rng(SplitMix64);

impl Rng {
    /// A generator seeded with `seed`.
    #[must_use]
    pub const fn new(seed: u64) -> Self {
        Self(SplitMix64::new(seed))
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0.next_u64()
    }

    /// A uniform value in `0..bound` (`bound > 0`); the tiny modulo bias is
    /// irrelevant for test data.
    pub fn below(&mut self, bound: usize) -> usize {
        self.0.below(bound as u64) as usize
    }

    /// A random byte.
    pub fn byte(&mut self) -> u8 {
        self.next_u64() as u8
    }

    /// Fill `buf` with random bytes.
    pub fn fill(&mut self, buf: &mut [u8]) {
        for b in buf {
            *b = self.byte();
        }
    }

    /// A `w`x`h` plane of one of four patterns (`1` black/white, `2` a flat
    /// grey with noise, `3` a ramp with noise, otherwise random bytes): the
    /// frames the libaom golden harnesses drew.
    pub fn pattern_plane(&mut self, w: usize, h: usize, pattern: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(w * h);
        for r in 0..h {
            for c in 0..w {
                let b = self.byte();
                out.push(match pattern {
                    1 => (b & 1) * 255,
                    2 => 128 + b % 16,
                    3 => (c * 3 + r * 5 + usize::from(b % 4)) as u8,
                    _ => b,
                });
            }
        }
        out
    }
}

/// The FNV-1a hash of `plane`'s visible samples, row by row.
#[must_use]
pub fn plane_fnv1a(plane: &PlaneView<'_>) -> u64 {
    fnv1a(
        plane
            .rows()
            .flat_map(|r| r[..plane.width()].iter().copied()),
    )
}
