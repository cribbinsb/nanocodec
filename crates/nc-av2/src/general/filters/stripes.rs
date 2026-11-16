//! Loop-restoration stripes (dav2d `dav2d_lr_slice`, `lr_sbrow`,
//! `lr_stripe`; AVM `av2_loop_restoration_filter_unit`'s stripes): the
//! frame's rows in stripes of 64, the first 8 shorter, so that each stripe
//! boundary sits 8 rows above a 64-row band boundary. Restoration filters
//! (GDF, the Wiener filters) read beyond their stripe only the
//! [`STRIPE_CONTEXT_ROWS`] rows next to it, and those from the deblocked
//! frame (the rows dav2d saves before CDEF changes them); further rows
//! repeat the last of them.

/// Rows of the deblocked frame a stripe sees above and below it.
pub const STRIPE_CONTEXT_ROWS: usize = 2;
/// Height of a stripe, and how much shorter the first is.
const STRIPE_HEIGHT: usize = 64;
const STRIPE_OFFSET: usize = 8;

/// One stripe: rows `top..bottom` of a frame `frame_height` rows tall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stripe {
    /// First row.
    pub top: usize,
    /// Row after the last.
    pub bottom: usize,
    /// The frame's height.
    pub frame_height: usize,
}

impl Stripe {
    /// Rows in the stripe.
    #[must_use]
    pub fn height(self) -> usize {
        self.bottom - self.top
    }

    /// The frame row a filter reads for the row `offset` rows below the
    /// stripe's top (negative: above it), when it reads at most `context`
    /// rows beyond the stripe: the row and whether it is the deblocked
    /// frame's. Inside the stripe it is the row itself; next to a stripe
    /// boundary inside the frame, the deblocked rows; at the frame's top
    /// and bottom, the stripe's edge row.
    #[must_use]
    pub fn context_row(self, offset: isize, context: usize) -> (usize, bool) {
        let height = self.height() as isize;
        if (0..height).contains(&offset) {
            (self.top + offset as usize, false)
        } else if offset < 0 {
            if self.top == 0 {
                (self.top, false)
            } else {
                (self.top - offset.unsigned_abs().min(context), true)
            }
        } else if self.bottom == self.frame_height {
            (self.bottom - 1, false)
        } else {
            let below = (offset - height) as usize;
            (self.bottom + below.min(context - 1), true)
        }
    }
}

/// The stripes of a frame `height` rows tall, top to bottom.
pub fn stripes(height: usize) -> impl Iterator<Item = Stripe> {
    let mut top = 0;
    std::iter::from_fn(move || {
        if top >= height {
            return None;
        }
        let end = if top == 0 {
            STRIPE_HEIGHT - STRIPE_OFFSET
        } else {
            top + STRIPE_HEIGHT
        };
        let stripe = Stripe {
            top,
            bottom: end.min(height),
            frame_height: height,
        };
        top = stripe.bottom;
        Some(stripe)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stripes_start_eight_rows_above_each_band() {
        let tops: Vec<_> = stripes(128).map(|s| (s.top, s.bottom)).collect();
        assert_eq!(tops, [(0, 56), (56, 120), (120, 128)]);
        let short: Vec<_> = stripes(40).map(|s| (s.top, s.bottom)).collect();
        assert_eq!(short, [(0, 40)]);
    }

    #[test]
    fn context_rows_come_from_the_deblocked_frame() {
        let stripe = Stripe {
            top: 56,
            bottom: 120,
            frame_height: 128,
        };
        assert_eq!(stripe.context_row(-1, 2), (55, true));
        assert_eq!(stripe.context_row(-6, 2), (54, true));
        assert_eq!(stripe.context_row(64, 2), (120, true));
        assert_eq!(stripe.context_row(69, 2), (121, true));
        let first = Stripe {
            top: 0,
            bottom: 56,
            frame_height: 128,
        };
        assert_eq!(first.context_row(-3, 2), (0, false));
        let last = Stripe {
            top: 120,
            bottom: 128,
            frame_height: 128,
        };
        assert_eq!(last.context_row(9, 2), (127, false));
    }
}
