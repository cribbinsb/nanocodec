//! Random legal decisions for the syntax-coverage stress test.
//!
//! Normal encodes reach only the syntax their mode decision likes: 8x8 inter
//! blocks next to 16x16 ones, far vectors, intra islands in inter frames or
//! large block-QP swings may never occur. Our encoder and decoder share
//! prediction code, so a bug on such a path (AV1's MV-reference scan next to
//! 8x8 inter blocks was one) goes unseen until a third-party decoder meets a
//! stream that uses it. With `debug.random_decisions = <seed>` the engine
//! replaces its partition, prediction-mode, vector and block-QP choices with
//! seeded random *legal* ones, and `tools/ci/syntax_stress.py` checks the
//! streams against every standard decoder.
//!
//! This is a test mode, never an encoding mode: compression is meaningless.
//! When the parameter is 0 (the default) nothing here is constructed and the
//! normal decision paths run unchanged, byte for byte.

use crate::backend::PredictionMode;
use crate::mv::{MV_UNITS_PER_SAMPLE, Mv};
use crate::shape::{PartitionShape, ShapeSet};

/// Chance (in 1/256) that a splittable node is split: about one leaf in
/// three at each depth, so every block size occurs and sizes mix.
const SPLIT_CHANCE: u64 = 96;
/// Chance (of 256) that a leaf whose node allows rectangular shapes stays
/// square.
const SHAPE_CHANCE: u64 = 128;
/// Chance (in 1/256) that an inter-frame leaf is coded intra.
const INTRA_CHANCE: u64 = 40;
/// Chance (in 1/256) that an inter leaf takes one of the codec's cheap
/// (predicted) vectors rather than a random one.
const CANDIDATE_CHANCE: u64 = 128;
/// Largest whole-sample component of a random "near" vector.
const NEAR_RANGE_SAMPLES: i32 = 8;
/// Largest whole-sample component of a random "far" vector: reaches well past
/// the picture edge of the small stress clips (reference clamping), while
/// staying inside every codec's vector range at the levels we signal.
const FAR_RANGE_SAMPLES: i32 = 96;
/// Chance (in 1/256) that a random vector is "far".
const FAR_CHANCE: u64 = 32;
/// Largest block quantizer offset, native units; keeps H.264 `mb_qp_delta`
/// between neighbours inside its ±26 range.
const QP_OFFSET_SPAN: i32 = 12;
/// Mixed into the seed handed to the backend, so its sequence differs from
/// the engine's.
const BACKEND_SEED_SALT: u64 = 0x5EED_BAC4_E2D0_0000;

/// Seeded random source for test-only decisions (`SplitMix64`: tiny, fast,
/// reproducible). Public so a backend can randomise its own syntax choices
/// ([`CodecBackend::enable_random_decisions`]) from the same seed.
///
/// [`CodecBackend::enable_random_decisions`]: crate::CodecBackend::enable_random_decisions
#[derive(Debug, Clone)]
pub struct DebugRng {
    state: u64,
}

impl DebugRng {
    /// A source seeded with `seed`.
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

    /// True with probability `chance`/256.
    pub fn chance(&mut self, chance: u64) -> bool {
        self.next_u64() & 0xFF < chance
    }

    /// A uniform integer in `0..n` (`n` > 0).
    pub fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A uniform integer in `-span..=span`.
    pub fn symmetric(&mut self, span: i32) -> i32 {
        self.below((2 * span + 1) as u64) as i32 - span
    }
}

/// The engine's random partition, mode, vector and block-QP choices.
#[derive(Debug, Clone)]
pub(crate) struct RandomDecisions {
    rng: DebugRng,
}

impl RandomDecisions {
    /// The source for `debug.random_decisions = seed`; `None` when 0 (off).
    pub(crate) fn from_seed(seed: f32) -> Option<Self> {
        // The parameter is an integer seed carried in an f32 (exact to 2^24).
        let seed = seed as u64;
        (seed != 0).then_some(Self {
            rng: DebugRng::new(seed),
        })
    }

    /// The seed a backend derives its own random choices from, independent
    /// of the engine's sequence.
    pub(crate) fn backend_seed(&self) -> u64 {
        self.rng.clone().next_u64() ^ BACKEND_SEED_SALT
    }

    fn chance(&mut self, chance: u64) -> bool {
        self.rng.chance(chance)
    }

    /// True 50% of the time on spatial inter-layer enhancement leaves, so
    /// syntax stress exercises both `LAST_FRAME` and `LAST2_FRAME` + scaled MC.
    pub(crate) fn secondary_reference(&mut self) -> bool {
        self.chance(128)
    }

    /// True ~1/3 of the time on compound-capable dual-reference leaves, so
    /// syntax stress exercises compound `[LAST_FRAME, LAST2_FRAME]` prediction.
    pub(crate) fn compound_reference(&mut self) -> bool {
        self.chance(85)
    }

    fn next(&mut self) -> u64 {
        self.rng.next_u64()
    }

    fn symmetric(&mut self, span: i32) -> i32 {
        self.rng.symmetric(span)
    }

    /// Whether to split a node the walk may split (not a forced split).
    pub(crate) fn split(&mut self) -> bool {
        self.chance(SPLIT_CHANCE)
    }

    /// A rectangular shape from `shapes` for a node coded as a leaf, or
    /// `None` (half the time, and always for an empty set, which draws
    /// nothing).
    pub(crate) fn shape(&mut self, shapes: ShapeSet) -> Option<PartitionShape> {
        let count = shapes.iter().count() as u64;
        if count == 0 || self.chance(SHAPE_CHANCE) {
            return None;
        }
        shapes.iter().nth(self.rng.below(count) as usize)
    }

    /// The mode of an inter-frame leaf: intra, one of the codec's cheap
    /// vectors (`candidates`, zero first), or a random quarter-sample vector.
    /// `allows` rejects vectors the frame forbids (intra refresh); those
    /// leaves fall back to intra.
    pub(crate) fn inter_mode(
        &mut self,
        candidates: &[Mv],
        allows: impl Fn(Mv) -> bool,
    ) -> PredictionMode {
        if self.chance(INTRA_CHANCE) {
            return PredictionMode::IntraDc;
        }
        let mv = if !candidates.is_empty() && self.chance(CANDIDATE_CHANCE) {
            candidates[(self.next() % candidates.len() as u64) as usize]
        } else {
            self.vector()
        };
        if allows(mv) {
            PredictionMode::Inter(mv)
        } else {
            PredictionMode::IntraDc
        }
    }

    /// A random vector, quarter-sample phases included.
    pub(crate) fn vector(&mut self) -> Mv {
        let range = if self.chance(FAR_CHANCE) {
            FAR_RANGE_SAMPLES
        } else {
            NEAR_RANGE_SAMPLES
        };
        let span = range * MV_UNITS_PER_SAMPLE;
        Mv {
            x: self.symmetric(span) as i16,
            y: self.symmetric(span) as i16,
        }
    }

    /// The 8x8 sub-block vectors handed to codecs that split a block's motion
    /// finer than the partition (H.264): random, with repeats so that both
    /// merged and split sub-partitions occur.
    pub(crate) fn sub_block_vectors(&mut self, out: &mut [Mv]) {
        let shared = self.vector();
        for mv in out {
            *mv = if self.chance(CANDIDATE_CHANCE) {
                shared
            } else {
                self.vector()
            };
        }
    }

    /// A block quantizer offset keeping `frame + offset` inside `bounds`.
    pub(crate) fn qp_offset(&mut self, frame: u32, bounds: (u32, u32)) -> i16 {
        let offset = self.symmetric(QP_OFFSET_SPAN);
        let coded =
            (i64::from(frame) + i64::from(offset)).clamp(i64::from(bounds.0), i64::from(bounds.1));
        (coded - i64::from(frame)) as i16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_seed_is_off() {
        assert!(RandomDecisions::from_seed(0.0).is_none());
        assert!(RandomDecisions::from_seed(7.0).is_some());
    }

    #[test]
    fn same_seed_same_decisions() {
        let mut a = RandomDecisions::from_seed(42.0).unwrap();
        let mut b = RandomDecisions::from_seed(42.0).unwrap();
        for _ in 0..100 {
            assert_eq!(a.vector(), b.vector());
            assert_eq!(a.split(), b.split());
        }
    }

    #[test]
    fn decisions_cover_their_ranges() {
        let mut r = RandomDecisions::from_seed(1.0).unwrap();
        let (mut splits, mut intra, mut sub_pel, mut far) = (0, 0, 0, 0);
        for _ in 0..4096 {
            splits += usize::from(r.split());
            match r.inter_mode(&[Mv::ZERO], |_| true) {
                PredictionMode::IntraDc => intra += 1,
                PredictionMode::Inter(mv) => {
                    sub_pel += usize::from(!mv.is_full_pel());
                    let limit = (NEAR_RANGE_SAMPLES * MV_UNITS_PER_SAMPLE) as i16;
                    far += usize::from(mv.x.abs() > limit || mv.y.abs() > limit);
                }
            }
        }
        assert!(splits > 1000 && splits < 2000, "{splits}");
        assert!(intra > 400, "{intra}");
        assert!(sub_pel > 1000, "{sub_pel}");
        assert!(far > 50, "{far}");
    }

    #[test]
    fn qp_offsets_stay_in_bounds() {
        let mut r = RandomDecisions::from_seed(3.0).unwrap();
        for _ in 0..1000 {
            let offset = r.qp_offset(48, (0, 51));
            assert!((-12..=3).contains(&offset));
        }
        let mut seen_negative = false;
        for _ in 0..1000 {
            seen_negative |= r.qp_offset(30, (0, 51)) < 0;
        }
        assert!(seen_negative);
    }

    #[test]
    fn forbidden_vectors_fall_back_to_intra() {
        let mut r = RandomDecisions::from_seed(9.0).unwrap();
        for _ in 0..200 {
            if let PredictionMode::Inter(_) = r.inter_mode(&[Mv::ZERO], |_| false) {
                panic!("a forbidden vector was chosen");
            }
        }
    }
}
