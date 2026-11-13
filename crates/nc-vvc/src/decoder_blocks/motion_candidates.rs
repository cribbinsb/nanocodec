//! Both-list spatial and history candidates for general inter pictures.
//! Shares the vector representation and precision rounding with the encoder.
use super::motion::{Motion, Vector};
use crate::motion::Mv16;
pub(super) fn merge(
    [above, left, right, below, corner]: [Option<Motion>; 5],
    history: &[Motion],
    references: [usize; 2],
    maximum: u8,
    temporal: Option<Motion>,
) -> Vec<Motion> {
    let differs =
        |a: Option<Motion>, b: Option<Motion>| a.filter(|a| b.is_none_or(|b| a.lists != b.lists));
    let mut list: Vec<_> = [
        above,
        differs(left, above),
        differs(right, above),
        differs(below, left),
    ]
    .into_iter()
    .flatten()
    .collect();
    if list.len() < 4 {
        list.extend(differs(differs(corner, left), above));
    }
    let maximum = usize::from(maximum);
    list.extend(temporal);
    list.truncate(maximum);
    for (age, &candidate) in history.iter().rev().enumerate() {
        if list.len() >= maximum.saturating_sub(1) {
            break;
        }
        if age >= 2
            || (differs(Some(candidate), left).is_some()
                && differs(Some(candidate), above).is_some())
        {
            list.push(candidate);
        }
    }
    if list.len() > 1 && list.len() < maximum {
        let lists = std::array::from_fn(|i| match (list[0].lists[i], list[1].lists[i]) {
            (Some(a), Some(b)) => {
                let sum = Mv16 {
                    x: a.vector.x + b.vector.x,
                    y: a.vector.y + b.vector.y,
                }
                .round(1);
                Some(Vector {
                    reference: a.reference,
                    vector: Mv16 {
                        x: sum.x >> 1,
                        y: sum.y >> 1,
                    },
                })
            }
            (a, b) => a.or(b),
        });
        list.push(Motion {
            lists,
            half: list[0].half && list[1].half,
            ..Motion::default()
        });
    }
    let zero_start = list.len();
    let count = if references[1] == 0 {
        references[0]
    } else {
        references[0].min(references[1])
    };
    while list.len() < maximum {
        let index = list.len() - zero_start;
        let reference = if index < count { index as u8 } else { 0 };
        list.push(Motion {
            lists: references.map(|n| {
                (n != 0).then_some(Vector {
                    reference,
                    vector: Mv16::ZERO,
                })
            }),
            ..Motion::default()
        });
    }
    list
}
pub(super) fn amvp(
    [above, left, right, below, corner]: [Option<Motion>; 5],
    history: &[Motion],
    list: usize,
    reference: u8,
    references: [&[i32]; 2],
    shift: u32,
    temporal: Option<Mv16>,
) -> [Mv16; 2] {
    let matching = |motion: Motion, source: usize| {
        motion.lists[source]
            .filter(|v| {
                references[source].get(usize::from(v.reference))
                    == references[list].get(usize::from(reference))
            })
            .map(|v| v.vector.round(shift))
    };
    let same = |motion: Option<Motion>| {
        motion.and_then(|m| matching(m, list).or_else(|| matching(m, 1 - list)))
    };
    let first = same(below).or_else(|| same(left));
    let second = same(right).or_else(|| same(above)).or_else(|| same(corner));
    let mut candidates: Vec<_> = first
        .into_iter()
        .chain(second.filter(|&v| first != Some(v)))
        .collect();
    if candidates.len() < 2 {
        candidates.extend(temporal.map(|v| v.round(shift)));
    }
    for &candidate in history.iter().take(4) {
        for source in [list, 1 - list] {
            if candidates.len() == 2 {
                break;
            }
            candidates.extend(matching(candidate, source));
        }
    }
    candidates.resize(2, Mv16::ZERO);
    [candidates[0], candidates[1]]
}
