//! Which reference slots an inter frame predicts from when it does not
//! name them (`explicit_ref_frame_map` 0), and which of them its entropy
//! contexts start from when it does not say.
//!
//! Ported from dav2d `obu.c` `get_ref_frames` and `derive_pri_sec_ref`
//! (AVM `av2_get_ref_frames`, `choose_primary_secondary_ref_frame`).

/// Most references a frame predicts from (AVM `INTER_REFS_PER_FRAME`).
pub const MAX_REFERENCES: usize = 7;
/// Largest order difference kept (AVM `RELATIVE_DIST_BITS` 8).
const MAX_ORDER_DIFFERENCE: i32 = 127;
/// Order differences past this add one to the score each (before it the
/// score halves its distance to 128).
const NEAR_DISTANCES: i32 = 6;
/// Score of an infinitely distant past reference.
const DISTANCE_SCALE: i32 = 128;
/// Weight of a reference's size (log2 of its area) in the score.
const AREA_WEIGHT: i32 = 1 << 5;

/// What ranking needs to know of the frame held in a reference slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotInfo {
    /// Identity of the frame: slots filled by one frame share it.
    pub frame: u32,
    /// Display order (the order hint unwrapped; dav2d `frame_offset`).
    pub order: u32,
    /// Base quantizer index.
    pub quantizer: u16,
    /// An inter frame (not key or intra-only): only those lend their
    /// entropy contexts.
    pub inter: bool,
    /// Temporal layer.
    pub temporal_layer: u8,
    /// Embedded layer: each layer below the current frame's counts as one
    /// more of display distance.
    pub embedded_layer: u8,
    /// Luma width and height.
    pub size: (u32, u32),
}

/// The frame being decoded, as ranking sees it.
#[derive(Debug, Clone, Copy)]
pub struct Current {
    /// Display order.
    pub order: u32,
    /// Embedded layer.
    pub embedded_layer: u8,
    /// The temporal and embedded layers it may predict from (bit `n`:
    /// layer `n`); references of other layers are left out, though they
    /// still count as future frames.
    pub layer_masks: (u8, u8),
    /// Luma size once known: references more than twice as large or less
    /// than a sixteenth of it in either dimension are left out.
    pub size: Option<(u32, u32)>,
}

/// `a - b` in display order, clamped (dav2d `get_poc_diff`).
#[must_use]
pub fn order_difference(a: u32, b: u32) -> i32 {
    (i64::from(a) - i64::from(b)).clamp(
        -i64::from(MAX_ORDER_DIFFERENCE),
        i64::from(MAX_ORDER_DIFFERENCE),
    ) as i32
}

/// A slot's place in the ranking.
#[derive(Debug, Clone, Copy)]
struct Scored {
    slot: u8,
    score: i32,
    order: u32,
    embedded_layer: u8,
    /// Current order minus the reference's: positive for a past reference.
    difference: i32,
    quantizer: u16,
}

/// The slots `current` predicts from, best first: distinct frames ordered
/// by a score of distance, size and quantizer, at most
/// [`MAX_REFERENCES`]. Returns the slots and how many are distinct (the
/// rest repeat the first).
#[must_use]
pub fn rank(slots: &[Option<SlotInfo>], current: Current) -> ([u8; MAX_REFERENCES], usize) {
    let difference = |info: &SlotInfo| order_difference(current.order, info.order);
    let any_future = slots.iter().flatten().any(|info| difference(info) < 0);
    let mut ranked: Vec<Scored> = Vec::with_capacity(slots.len());
    let mut last_frame = None;
    for (slot, info) in slots.iter().enumerate() {
        let Some(info) = info else { continue };
        let (temporal_mask, embedded_mask) = current.layer_masks;
        if Some(info.frame) == last_frame
            || temporal_mask & (1 << info.temporal_layer) == 0
            || embedded_mask & (1 << info.embedded_layer) == 0
        {
            continue;
        }
        if current
            .size
            .is_some_and(|size| !size_allowed(size, info.size))
        {
            continue;
        }
        let difference = difference(info);
        // AVM `av2_get_ref_frames` `tdist` (dav2d `get_ref_frames`).
        let distance =
            difference.abs() + i32::from(current.embedded_layer) - i32::from(info.embedded_layer);
        let distance_score = if any_future {
            distance << 6
        } else {
            DISTANCE_SCALE - (DISTANCE_SCALE >> distance.min(NEAR_DISTANCES))
                + (distance - NEAR_DISTANCES).max(0)
        };
        // At least one sample; 64 bits hold the largest area (65536²).
        let area = (u64::from(info.size.0) * u64::from(info.size.1)).max(1);
        let area_log2 = area.ilog2() as i32;
        let scored = Scored {
            slot: slot as u8,
            score: distance_score - area_log2 * AREA_WEIGHT + i32::from(info.quantizer),
            order: info.order,
            embedded_layer: info.embedded_layer,
            difference,
            quantizer: info.quantizer,
        };
        // A frame already ranked from another slot.
        if ranked.iter().any(|other| {
            other.score == scored.score
                && other.order == scored.order
                && other.embedded_layer == scored.embedded_layer
        }) {
            continue;
        }
        // Ascending score; equal scores keep slot order.
        let place = ranked
            .iter()
            .rposition(|other| other.score <= scored.score)
            .map_or(0, |index| index + 1);
        ranked.insert(place, scored);
        last_frame = Some(info.frame);
    }
    if ranked.len() > MAX_REFERENCES {
        demote_furthest(&mut ranked);
    }
    let mut chosen = [ranked.first().map_or(0, |first| first.slot); MAX_REFERENCES];
    for (place, scored) in chosen.iter_mut().zip(&ranked) {
        *place = scored.slot;
    }
    (chosen, ranked.len().min(MAX_REFERENCES))
}

/// Whether a reference of size `reference` may predict a frame of `size`
/// (at most twice as large, at least a sixteenth, per dimension).
fn size_allowed(size: (u32, u32), reference: (u32, u32)) -> bool {
    let wide = |a: u32| u64::from(a);
    2 * wide(size.0) >= wide(reference.0)
        && 2 * wide(size.1) >= wide(reference.1)
        && wide(size.0) <= 16 * wide(reference.0)
        && wide(size.1) <= 16 * wide(reference.1)
}

/// With all eight slots distinct, the one left out is the furthest
/// high-quantizer frame on the side that has more of them, rather than the
/// worst score: it moves to the end (dav2d `get_ref_frames`).
fn demote_furthest(ranked: &mut [Scored]) {
    let quantizers = ranked.iter().map(|scored| i32::from(scored.quantizer));
    let (low, high) = (
        quantizers.clone().min().unwrap_or(0),
        quantizers.max().unwrap_or(0),
    );
    let threshold = (high + low + 1) >> 1;
    // Past and future: the furthest index and its distance, and the count.
    let mut furthest = [(0_usize, 0_i32); 2];
    let mut counts = [0_usize; 2];
    for (index, scored) in ranked.iter().enumerate() {
        if i32::from(scored.quantizer) < threshold || scored.difference == 0 {
            continue;
        }
        let side = usize::from(scored.difference < 0);
        if scored.difference.abs() > furthest[side].1 {
            furthest[side] = (index, scored.difference.abs());
        }
        counts[side] += 1;
    }
    let side = match counts[0].cmp(&counts[1]) {
        std::cmp::Ordering::Greater => 0,
        std::cmp::Ordering::Less => 1,
        std::cmp::Ordering::Equal => usize::from(furthest[0].1 < furthest[1].1),
    };
    let index = furthest[side].0;
    if index < MAX_REFERENCES {
        ranked[index..=MAX_REFERENCES].rotate_left(1);
    }
}

/// The ranks `0..count` ordered by `precedes` the way dav2d orders them
/// (`find_tip_ref_frames`, `dav2d_refmvs_init_frame`; dav2d 8f30abf3,
/// 54856741): each rank walks the order so far and swaps into the first
/// place it precedes, the displaced rank walking on. This is a sort, but
/// not a stable one: a rank inserted before a run of equal ranks moves
/// the first of them behind the others, and AVM's references come out in
/// that order.
#[must_use]
pub fn swap_insertion_order(count: usize, precedes: impl Fn(usize, usize) -> bool) -> Vec<usize> {
    let mut order: Vec<usize> = Vec::with_capacity(count);
    for n in 0..count {
        let mut current = n;
        for held in &mut order {
            if precedes(current, *held) {
                std::mem::swap(held, &mut current);
            }
        }
        order.push(current);
    }
    order
}

/// The ranks (into `references`, a frame's slots best first) whose final
/// entropy contexts a frame of `quantizer` at display order `order` starts
/// from when it does not name one, and the second choice that averaging
/// uses: the inter frames nearest in quantizer, then in order, then the
/// later one (dav2d `derive_pri_sec_ref`).
#[must_use]
pub fn context_sources(
    references: &[Option<SlotInfo>],
    quantizer: u16,
    order: u32,
) -> (Option<u8>, Option<u8>) {
    // (rank, quantizer difference, order difference, order), best first.
    let mut best: [Option<(u8, i32, i32, u32)>; 2] = [None; 2];
    for (rank, info) in references.iter().enumerate() {
        let Some(info) = info.filter(|info| info.inter) else {
            continue;
        };
        let candidate = (
            rank as u8,
            (i32::from(info.quantizer) - i32::from(quantizer)).abs(),
            order_difference(order, info.order).abs(),
            info.order,
        );
        let better = |held: Option<(u8, i32, i32, u32)>| {
            held.is_none_or(|(_, quantizer_gap, order_gap, held_order)| {
                candidate.1 < quantizer_gap
                    || (candidate.1 == quantizer_gap
                        && (candidate.2 < order_gap
                            || (candidate.2 == order_gap
                                && order_difference(held_order, candidate.3) < 0)))
            })
        };
        if better(best[0]) {
            best = [Some(candidate), best[0]];
        } else if better(best[1]) {
            best[1] = Some(candidate);
        }
    }
    (
        best[0].map(|chosen| chosen.0),
        best[1].map(|chosen| chosen.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swap_insertion_moves_the_first_of_equal_ranks_back() {
        // Orders 2, 1, 3, 1, 0: dav2d's walk gives 4, 3, 1, 0, 2 (a stable
        // sort would give 4, 1, 3, 0, 2).
        let orders = [2, 1, 3, 1, 0];
        let order = swap_insertion_order(orders.len(), |a, b| orders[a] < orders[b]);
        assert_eq!(order, [4, 3, 1, 0, 2]);
    }

    #[allow(clippy::unnecessary_wraps, reason = "an occupied slot")]
    fn frame(frame: u32, order: u32, quantizer: u16, inter: bool) -> Option<SlotInfo> {
        Some(SlotInfo {
            frame,
            order,
            quantizer,
            inter,
            temporal_layer: 0,
            embedded_layer: 0,
            size: (640, 360),
        })
    }

    fn current(order: u32) -> Current {
        Current {
            order,
            embedded_layer: 0,
            layer_masks: (1, 1),
            size: Some((640, 360)),
        }
    }

    /// The slots of avmenc's low-delay stream (dav2d `DEBUG_FRAME_HDR`
    /// `refs[...]`): the key frame in every slot, then frame `n` in slot
    /// `1 + (n - 1) % 7`.
    #[test]
    fn low_delay_slots_rank_as_dav2d_does() {
        let key = frame(0, 0, 170, false);
        let mut slots = [key; 8];
        assert_eq!(rank(&slots, current(1)), ([0; 7], 1));
        slots[1] = frame(1, 1, 210, true);
        assert_eq!(rank(&slots, current(2)).0[..2], [0, 1]);
        slots[2] = frame(2, 2, 210, true);
        let (chosen, count) = rank(&slots, current(3));
        assert_eq!((&chosen[..3], count), (&[2, 0, 1][..], 3));
        assert_eq!(chosen[3..], [2; 4]);
        slots[3] = frame(3, 3, 210, true);
        assert_eq!(rank(&slots, current(4)).0[..4], [3, 0, 2, 1]);
    }

    #[test]
    fn contexts_come_from_the_nearest_inter_frame() {
        let references = [
            frame(3, 3, 210, true),
            frame(0, 0, 170, false),
            frame(2, 2, 210, true),
            frame(1, 1, 200, true),
        ];
        assert_eq!(context_sources(&references, 210, 4), (Some(0), Some(2)));
        assert_eq!(context_sources(&references, 200, 4), (Some(3), Some(0)));
        assert_eq!(context_sources(&references[1..2], 210, 4), (None, None));
    }
}
