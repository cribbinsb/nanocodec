//! Motion-list edge cases independent of the downloaded conformance corpus.
use super::*;
fn motion(reference: u8, x: i32) -> Motion {
    Motion {
        lists: [
            Some(Vector {
                reference,
                vector: Mv16 { x, y: 0 },
            }),
            None,
        ],
        ..Motion::default()
    }
}
#[test]
fn decoded_motion_wraps_at_storage_boundary() {
    assert_eq!(wrap_component(131_071), 131_071);
    assert_eq!(wrap_component(131_072), -131_072);
    assert_eq!(wrap_component(-131_073), 131_071);
}
#[test]
fn zero_merge_candidates_stop_advancing_after_active_references() {
    let list = merge([None; 5], &[], [2, 0], 6, None);
    assert_eq!(
        list.iter()
            .map(|m| m.lists[0].unwrap().reference)
            .collect::<Vec<_>>(),
        [0, 1, 0, 0, 0, 0]
    );
}
#[test]
fn spatial_merge_deduplicates_and_averages_with_ties_toward_zero() {
    let a = motion(0, 4);
    let b = motion(1, -1);
    let list = merge(
        [Some(a), Some(a), Some(b), None, None],
        &[],
        [2, 0],
        6,
        None,
    );
    assert_eq!(&list[..3], &[a, b, motion(0, 1)]);
}
#[test]
fn amvp_matches_reference_picture_and_preserves_history_order() {
    let refs = [10, 20, 10];
    let spatial = [None, Some(motion(2, 5)), None, None, None];
    let history = [motion(1, 40), motion(0, 12), motion(0, 24)];
    assert_eq!(
        amvp(spatial, &history, 0, 0, [&refs, &[]], 2, None),
        [
            motion(0, 4).lists[0].unwrap().vector,
            motion(0, 12).lists[0].unwrap().vector
        ]
    );
}
