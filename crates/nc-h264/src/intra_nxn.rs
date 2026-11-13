//! What `Intra_4x4` and `Intra_8x8` blocks (`I_NxN` macroblocks, H.264
//! §8.3.1, §8.3.2) share in the encoder: the mode numbering, the mode
//! signalling estimate, and `Intra_8x8`'s SATD mode search.
//!
//! The search takes each mode's evaluation as a closure: the shortlist
//! evaluates DC, vertical, horizontal and the predicted mode, then only the
//! two diagonal modes next to whichever of the first three won (5–7 of the
//! nine modes); the full search evaluates every mode the neighbours allow.
//! `Intra_4x4` ranks its modes the same way in its own loop (`intra4x4`).

/// `Intra4x4PredMode` and `Intra8x8PredMode` values (Tables 8-2 and 8-3
/// number the directions alike).
pub use nc_dsp::h264_intra_8x8_mode::{
    COUNT as MODE_COUNT, DC, DIAGONAL_DOWN_LEFT, DIAGONAL_DOWN_RIGHT, HORIZONTAL, HORIZONTAL_DOWN,
    HORIZONTAL_UP, VERTICAL, VERTICAL_LEFT, VERTICAL_RIGHT,
};

/// Signalling bits of a mode: `prev_intra*_pred_mode_flag` alone when it
/// equals the predicted mode, plus the 3-bit `rem_intra*_pred_mode`
/// otherwise (CABAC makes both cheaper, but their ratio is what matters).
const PREDICTED_MODE_BITS: f32 = 1.0;
const OTHER_MODE_BITS: f32 = 4.0;

/// The signalling bits of `mode` for a block whose predicted mode is
/// `predicted`.
pub(crate) fn mode_bits(mode: u8, predicted: u8) -> f32 {
    if mode == predicted {
        PREDICTED_MODE_BITS
    } else {
        OTHER_MODE_BITS
    }
}

/// A block's mode, its prediction `P` and its cost.
pub(crate) type Choice<P> = (u8, P, f32);

/// The cheaper of two choices (the first on a tie).
fn better<P>(a: Choice<P>, b: Choice<P>) -> Choice<P> {
    if b.2 < a.2 { b } else { a }
}

/// The cheapest mode by `evaluate` among every mode `allows` admits (DC
/// always), evaluating DC first.
pub(crate) fn best_of_all_modes<P>(
    allows: impl Fn(u8) -> bool,
    evaluate: impl Fn(u8) -> Choice<P>,
) -> Choice<P> {
    (0..MODE_COUNT)
        .filter(|&m| m != DC && allows(m))
        .fold(evaluate(DC), |best, m| better(best, evaluate(m)))
}

/// The cheapest mode by `evaluate` from the shortlist in the module
/// comment, for a block whose predicted mode is `predicted`.
pub(crate) fn best_of_shortlist<P>(
    predicted: u8,
    allows: impl Fn(u8) -> bool,
    evaluate: impl Fn(u8) -> Choice<P>,
) -> Choice<P> {
    let mut best = evaluate(DC);
    for m in [VERTICAL, HORIZONTAL, predicted] {
        if m != DC && allows(m) {
            best = better(best, evaluate(m));
        }
    }
    let refinements: &[u8] = match best.0 {
        VERTICAL => &[VERTICAL_LEFT, VERTICAL_RIGHT],
        HORIZONTAL => &[HORIZONTAL_UP, HORIZONTAL_DOWN],
        DC => &[DIAGONAL_DOWN_LEFT, DIAGONAL_DOWN_RIGHT],
        _ => &[],
    };
    for &m in refinements {
        if m != predicted && allows(m) {
            best = better(best, evaluate(m));
        }
    }
    best
}
