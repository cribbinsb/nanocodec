//! Isolated trailing ±1 levels dropped from a quantised block (SVT-AV1's
//! `shave_coeff`), shared by HEVC, VVC and VP9; each supplies its scan and its
//! fast search for the previous nonzero level.
//!
//! An isolated high-frequency ±1 costs its last-position and significance
//! bits for almost no distortion gain; it is mostly noise.

/// Drop trailing ±1 levels that stand at least `min_gap` scan positions
/// after the previous nonzero level, from the end of the block backwards.
/// `positions` maps scan index to raster index; `previous_nonzero(levels,
/// end)` is the highest scan index below `end` with a nonzero level. The
/// first nonzero level of the block is always kept. Returns the new end
/// of block (one past the last nonzero scan index), or zero if empty.
pub fn shave_isolated_ones<L, P>(
    levels: &mut [L],
    positions: &[P],
    min_gap: usize,
    mut previous_nonzero: impl FnMut(&[L], usize) -> Option<usize>,
) -> usize
where
    L: Copy + Default + Into<i32>,
    P: Copy + Into<usize>,
{
    let Some(mut last) = previous_nonzero(levels, positions.len()) else {
        return 0;
    };
    while levels[positions[last].into()].into().abs() == 1 {
        let Some(previous) = previous_nonzero(levels, last) else {
            break;
        };
        if last - previous < min_gap {
            break;
        }
        levels[positions[last].into()] = L::default();
        last = previous;
    }
    last + 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drops_isolated_trailing_ones_and_keeps_the_first_level() {
        let positions: Vec<usize> = (0..16).collect();
        let previous = |levels: &[i32], end: usize| (0..end).rev().find(|&i| levels[i] != 0);
        let mut levels = [2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1];
        assert_eq!(shave_isolated_ones(&mut levels, &positions, 8, previous), 3);
        assert_eq!(levels[15], 0);
        assert_eq!(levels[2], 1, "a close ±1 stays");
        let mut lone = [0; 16];
        lone[9] = 1;
        assert_eq!(shave_isolated_ones(&mut lone, &positions, 1, previous), 10);
        assert_eq!(lone[9], 1, "the first level is kept");
        assert_eq!(
            shave_isolated_ones(&mut [0; 16], &positions, 1, previous),
            0
        );
    }
}
