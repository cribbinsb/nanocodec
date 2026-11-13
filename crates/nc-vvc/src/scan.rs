//! H.266 §9.3 grouped diagonal coefficient scans, prepared at compile time.
//!
//! Like HEVC's cached scans, these tables keep syntax traversal out of the
//! per-candidate allocation path. Transform blocks with both sides at least
//! 4 (up to 32, square or rectangular) use 4x4 groups (§7.3.11.11:
//! `log2SbW` = `log2SbH` = 2); the one- and two-sample-wide luma blocks of
//! intra sub-partitions use 1x16 / 2x8 groups (or their transposes), so a
//! luma group always holds sixteen coefficients.

/// Coefficients of one coefficient group: [`rect_positions`] visits the
/// groups in [`rect_groups`] order, this many positions each.
pub(crate) const GROUP_SIZE: usize = 16;

/// The width and height of a `width` x `height` luma block's coefficient
/// groups (§7.3.11.11 `log2SbW`, `log2SbH`: 4x4 unless a side is below 4,
/// then that side's width and sixteen samples in all).
pub(crate) const fn group_shape(width: usize, height: usize) -> (usize, usize) {
    match (width, height) {
        (1 | 2, _) => (width, GROUP_SIZE / width),
        (_, 1 | 2) => (GROUP_SIZE / height, height),
        _ => (4, 4),
    }
}

/// The coefficient-group grid of a `width` x `height` block: columns, rows.
pub(crate) const fn group_grid(width: usize, height: usize) -> (usize, usize) {
    let (gw, gh) = group_shape(width, height);
    (width / gw, height / gh)
}

/// The up-right diagonal scan of a `width` x `height` array (§6.5.3):
/// anti-diagonals from the top-left, each from bottom-left to top-right.
const fn diagonal<const N: usize>(width: usize, height: usize) -> [usize; N] {
    assert!(N == width * height);
    let mut order = [0; N];
    let (mut sum, mut index) = (0, 0);
    while sum < width + height - 1 {
        let mut x = sum.saturating_sub(height - 1);
        while x <= sum && x < width {
            order[index] = (sum - x) * width + x;
            index += 1;
            x += 1;
        }
        sum += 1;
    }
    order
}

/// The positions of a `width` x `height` block, group by group in
/// `groups` order (the diagonal scan of its [`group_grid`]), each group in
/// its own diagonal order.
const fn grouped<const N: usize>(width: usize, height: usize, groups: &[usize]) -> [usize; N] {
    assert!(N == width * height);
    let (gw, gh) = group_shape(width, height);
    let local: [usize; GROUP_SIZE] = diagonal(gw, gh);
    let columns = width / gw;
    let mut order = [0; N];
    let mut index = 0;
    while index < N {
        let group = groups[index / GROUP_SIZE];
        let within = local[index % GROUP_SIZE];
        order[index] =
            (group / columns * gh + within / gw) * width + group % columns * gw + within % gw;
        index += 1;
    }
    order
}

/// The inverse of a grouped scan: each raster position's scan index.
const fn inverse<const N: usize>(scan: &[usize; N]) -> [usize; N] {
    let mut out = [0; N];
    let mut index = 0;
    while index < N {
        out[scan[index]] = index;
        index += 1;
    }
    out
}

/// Defines the group order, grouped scan and its inverse of every
/// transform size, and the lookups over them.
macro_rules! scans {
    ($(($w:literal, $h:literal, $groups:ident, $positions:ident, $indices:ident)),* $(,)?) => {
        $(
            const $groups: [usize; $w * $h / GROUP_SIZE] =
                diagonal(group_grid($w, $h).0, group_grid($w, $h).1);
            const $positions: [usize; $w * $h] = grouped($w, $h, &$groups);
            const $indices: [usize; $w * $h] = inverse(&$positions);
        )*

        /// The coefficient groups of a `width` x `height` block in coding
        /// order, as raster indices of its [`group_grid`].
        pub(crate) const fn rect_groups(width: usize, height: usize) -> &'static [usize] {
            match (width, height) {
                $(($w, $h) => &$groups,)*
                _ => panic!("unsupported VVC transform size"),
            }
        }

        /// The raster positions of a `width` x `height` block in coding
        /// order, [`GROUP_SIZE`] per group.
        pub(crate) const fn rect_positions(width: usize, height: usize) -> &'static [usize] {
            match (width, height) {
                $(($w, $h) => &$positions,)*
                _ => panic!("unsupported VVC transform size"),
            }
        }

        /// The scan index of each raster position of a `width` x `height`
        /// block (the inverse of [`rect_positions`]).
        pub(crate) fn rect_indices(width: usize, height: usize) -> &'static [usize] {
            match (width, height) {
                $(($w, $h) => &$indices,)*
                _ => panic!("unsupported VVC transform size"),
            }
        }
    };
}

scans!(
    (1, 16, G1X16, S1X16, I1X16),
    (1, 32, G1X32, S1X32, I1X32),
    (2, 8, G2X8, S2X8, I2X8),
    (2, 16, G2X16, S2X16, I2X16),
    (2, 32, G2X32, S2X32, I2X32),
    (8, 2, G8X2, S8X2, I8X2),
    (16, 1, G16X1, S16X1, I16X1),
    (16, 2, G16X2, S16X2, I16X2),
    (32, 1, G32X1, S32X1, I32X1),
    (32, 2, G32X2, S32X2, I32X2),
    (4, 4, G4X4, S4X4, I4X4),
    (4, 8, G4X8, S4X8, I4X8),
    (4, 16, G4X16, S4X16, I4X16),
    (4, 32, G4X32, S4X32, I4X32),
    (8, 4, G8X4, S8X4, I8X4),
    (8, 8, G8X8, S8X8, I8X8),
    (8, 16, G8X16, S8X16, I8X16),
    (8, 32, G8X32, S8X32, I8X32),
    (16, 4, G16X4, S16X4, I16X4),
    (16, 8, G16X8, S16X8, I16X8),
    (16, 16, G16X16, S16X16, I16X16),
    (16, 32, G16X32, S16X32, I16X32),
    (32, 4, G32X4, S32X4, I32X4),
    (32, 8, G32X8, S32X8, I32X8),
    (32, 16, G32X16, S32X16, I32X16),
    (32, 32, G32X32, S32X32, I32X32),
);

/// [`rect_positions`] of a `side`-square block.
#[cfg(test)]
pub(crate) const fn positions(side: usize) -> &'static [usize] {
    rect_positions(side, side)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_scans_cover_each_coefficient_once() {
        assert_eq!(
            positions(4),
            [0, 4, 1, 8, 5, 2, 12, 9, 6, 3, 13, 10, 7, 14, 11, 15]
        );
        let narrow = [(1, 16), (1, 32), (2, 8), (2, 16), (2, 32)];
        let sizes = [4, 8, 16, 32]
            .into_iter()
            .flat_map(|w| [4, 8, 16, 32].map(|h| (w, h)))
            .chain(narrow)
            .chain(narrow.map(|(w, h)| (h, w)));
        for (width, height) in sizes {
            let (gw, gh) = group_shape(width, height);
            let scan = rect_positions(width, height);
            for (index, &position) in scan.iter().enumerate() {
                assert_eq!(rect_indices(width, height)[position], index);
            }
            let mut sorted = scan.to_vec();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..width * height).collect::<Vec<_>>());
            for group in scan.as_chunks::<GROUP_SIZE>().0 {
                let origin = group[0];
                assert!(group.iter().all(|&p| p / width / gh == origin / width / gh
                    && p % width / gw == origin % width / gw));
            }
        }
    }

    #[test]
    fn rectangular_groups_follow_the_group_grid_diagonal() {
        // 16x8: a 4x2 group grid.
        assert_eq!(rect_groups(16, 8), [0, 4, 1, 5, 2, 6, 3, 7]);
        assert_eq!(rect_groups(8, 16), [0, 2, 1, 4, 3, 6, 5, 7]);
    }
}
