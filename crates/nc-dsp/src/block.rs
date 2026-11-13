//! Block sizes the pixel kernels support.

/// Supported block edge lengths, in samples. Every codec's block sizes are
/// built from these (H.264 macroblocks and partitions, HEVC CUs/TUs, AV1
/// blocks down to 4x4).
const EDGES: [usize; 5] = [4, 8, 16, 32, 64];

/// The smallest edge: edge class `c` is `MIN_EDGE << c`. Every kernel call
/// asks for its block's width and height; a shift needs no table load and
/// no bounds check, unlike indexing [`EDGES`].
const MIN_EDGE: usize = EDGES[0];

/// A block size with width and height each one of 4, 8, 16, 32 or 64.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockSize {
    width_class: u8,
    height_class: u8,
}

impl BlockSize {
    /// The block size `width` x `height`, or `None` if either edge is not one of
    /// 4, 8, 16, 32 or 64.
    #[must_use]
    pub const fn new(width: usize, height: usize) -> Option<Self> {
        match (edge_class(width), edge_class(height)) {
            (Some(w), Some(h)) => Some(Self {
                width_class: w,
                height_class: h,
            }),
            _ => None,
        }
    }

    /// Width in samples.
    #[must_use]
    pub const fn width(self) -> usize {
        MIN_EDGE << self.width_class
    }

    /// Height in samples.
    #[must_use]
    pub const fn height(self) -> usize {
        MIN_EDGE << self.height_class
    }

    /// Index of the width in 4, 8, 16, 32, 64; kernels are dispatched per width.
    pub(crate) const fn width_class(self) -> usize {
        self.width_class as usize
    }

    /// Every supported block size, in width-major order.
    pub fn all() -> impl Iterator<Item = BlockSize> {
        EDGES.iter().flat_map(|&w| {
            EDGES
                .iter()
                .map(move |&h| BlockSize::new(w, h).expect("edges are valid"))
        })
    }
}

#[expect(
    clippy::manual_is_power_of_two,
    reason = "`is_power_of_two` is a population count: ~30 instructions without `popcnt`"
)]
const fn edge_class(edge: usize) -> Option<u8> {
    // A power of two from 4 to 64: its class is the shift above 4. One set
    // bit is tested as `edge & (edge - 1)` (`edge` is at least 4 there):
    // `is_power_of_two` is a population count, ~30 instructions without
    // the `popcnt` extension, and every kernel call builds a size.
    if edge >= MIN_EDGE && edge <= EDGES[EDGES.len() - 1] && edge & (edge - 1) == 0 {
        Some((edge.trailing_zeros() - MIN_EDGE.trailing_zeros()) as u8)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::BlockSize;

    #[test]
    fn construction_and_accessors() {
        let b = BlockSize::new(16, 64).unwrap();
        assert_eq!((b.width(), b.height(), b.width_class()), (16, 64, 2));
        assert!(BlockSize::new(12, 16).is_none());
        assert!(BlockSize::new(16, 128).is_none());
        assert_eq!(BlockSize::all().count(), 25);
    }
}
