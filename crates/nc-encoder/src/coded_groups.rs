//! The coefficient groups (4x4 sub-blocks) of a transform block that hold a
//! non-zero level, as one bit per group in coding (scan) order, shared by
//! HEVC and VVC. Each codec builds the set from its own levels and scan, once
//! per block, and its shaving, sign hiding, rate estimates and syntax writer
//! then visit only the coded groups instead of testing every group again.

/// Which coefficient groups of a transform block hold a non-zero level: bit
/// `i` for the `i`-th group in scan order (at most 64, a 32x32 block of 4x4
/// groups). The codec states whether its set is exact or may keep a stale bit
/// after a level was zeroed (a walk then finds nothing in that group).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CodedGroups(u64);

impl CodedGroups {
    /// Only the first group (the one holding the DC) is coded.
    pub const FIRST: Self = Self(1);

    /// The groups whose bits are set in `mask`, bit `i` for the `i`-th group
    /// in scan order.
    #[must_use]
    pub const fn from_scan_mask(mask: u64) -> Self {
        Self(mask)
    }

    /// Whether no group is coded.
    #[must_use]
    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Whether the group at scan index `index` is coded.
    #[must_use]
    pub fn contains(self, index: usize) -> bool {
        self.0 >> index & 1 != 0
    }

    /// The scan index of the last coded group.
    #[must_use]
    pub fn last(self) -> Option<usize> {
        (!self.is_empty()).then(|| (u64::BITS - 1 - self.0.leading_zeros()) as usize)
    }

    /// The highest coded scan index below `end` (a group index).
    #[must_use]
    pub fn last_below(self, end: usize) -> Option<usize> {
        let below = if end >= u64::BITS as usize {
            self.0
        } else {
            self.0 & ((1 << end) - 1)
        };
        Self(below).last()
    }

    /// The set with its trailing groups dropped while `coded(index)` says
    /// they hold no level: exact again after a change that can only have
    /// emptied the last groups (shaving trailing levels).
    #[must_use]
    pub fn trimmed(self, coded: impl Fn(usize) -> bool) -> Self {
        let mut set = self;
        while let Some(last) = set.last() {
            if coded(last) {
                break;
            }
            set.0 &= !(1 << last);
        }
        set
    }

    /// The scan indices of the coded groups, in scan order.
    pub fn iter(self) -> impl Iterator<Item = usize> {
        let mut bits = self.0;
        std::iter::from_fn(move || {
            (bits != 0).then(|| {
                let i = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                i
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walks_and_trims_in_scan_order() {
        let set = CodedGroups::from_scan_mask(1 | 1 << 7 | 1 << 63);
        assert_eq!(set.iter().collect::<Vec<_>>(), [0, 7, 63]);
        assert_eq!(set.last(), Some(63));
        assert_eq!(set.last_below(63), Some(7));
        assert_eq!(set.last_below(64), Some(63));
        assert_eq!(set.last_below(0), None);
        assert!(set.contains(7) && !set.contains(8));
        assert_eq!(
            set.trimmed(|i| i == 7),
            CodedGroups::from_scan_mask(1 | 1 << 7)
        );
        assert_eq!(CodedGroups::default().last(), None);
    }
}
