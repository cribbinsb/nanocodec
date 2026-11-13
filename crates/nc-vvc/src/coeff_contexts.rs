//! H.266 coefficient CABAC initialization tables (I/P slices only).
//! Tables from `VVenC` Contexts.cpp, BSD-3-Clause-Clear; see `THIRD_PARTY.md`.
use crate::contexts::Context;
#[cfg_attr(test, derive(PartialEq, Eq))]
#[derive(Debug, Clone)]
pub(crate) struct CoeffContexts {
    pub group: [[Context; GROUP_CONTEXTS]; 2],
    pub sig: [[Context; SIG_CONTEXTS]; 2],
    pub parity: [[Context; LEVEL_CONTEXTS]; 2],
    pub gt1: [[Context; LEVEL_CONTEXTS]; 2],
    pub gt2: [[Context; LEVEL_CONTEXTS]; 2],
    pub last_x: [[Context; LAST_CONTEXTS]; 2],
    pub last_y: [[Context; LAST_CONTEXTS]; 2],
    /// `sig_coeff_flag` contexts of quantiser states 2 and 3 under
    /// dependent quantisation (§9.3.4.2.8: ctxIdx offset by
    /// `Max(0, QState − 1)` sets); states 0 and 1 use `sig`.
    pub dependent_sig: [[[Context; SIG_CONTEXTS]; 2]; 2],
    /// `sh_dep_quant_used_flag`: levels are coded through the quantiser
    /// state machine (`coeff::walk`, `dep_quant.rs`).
    pub dependent: bool,
}

// Per-channel context counts: luma's, the larger. Chroma's sets are shorter
// and leave the tail of each array unused. Fixed arrays make the per-candidate
// snapshots of the RD searches a copy rather than fourteen allocations.
const GROUP_CONTEXTS: usize = 2;
const SIG_CONTEXTS: usize = 12;
const LEVEL_CONTEXTS: usize = 21;
const LAST_CONTEXTS: usize = 20;
impl CoeffContexts {
    #[allow(
        clippy::too_many_lines,
        reason = "normative initialization table rows kept together"
    )]
    pub fn new(qp: u32, inter: bool) -> Self {
        Self {
            group: [
                init([18, 31], [25, 30], [8, 5], qp, inter),
                init([25, 15], [25, 45], [5, 8], qp, inter),
            ],
            sig: [
                init(
                    [25, 19, 28, 14, 25, 20, 29, 30, 19, 37, 30, 38],
                    [17, 41, 42, 29, 25, 49, 43, 37, 33, 58, 51, 30],
                    [12, 9, 9, 10, 9, 9, 9, 10, 8, 8, 8, 10],
                    qp,
                    inter,
                ),
                init(
                    [25, 27, 28, 37, 34, 53, 53, 46],
                    [17, 34, 35, 21, 41, 59, 60, 38],
                    [12, 12, 9, 13, 4, 5, 8, 9],
                    qp,
                    inter,
                ),
            ],
            parity: [
                init(
                    [
                        33, 25, 18, 26, 34, 27, 25, 26, 19, 42, 35, 33, 19, 27, 35, 35, 34, 42, 20,
                        43, 20,
                    ],
                    [
                        18, 17, 33, 18, 26, 42, 25, 33, 26, 42, 27, 25, 34, 42, 42, 35, 26, 27, 42,
                        20, 20,
                    ],
                    [
                        8, 9, 12, 13, 13, 13, 10, 13, 13, 13, 13, 13, 13, 13, 13, 13, 10, 13, 13,
                        13, 13,
                    ],
                    qp,
                    inter,
                ),
                init(
                    [33, 25, 26, 42, 19, 27, 26, 50, 35, 20, 43],
                    [25, 25, 26, 11, 19, 27, 33, 42, 35, 35, 43],
                    [8, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13],
                    qp,
                    inter,
                ),
            ],
            gt1: [
                init(
                    [
                        25, 25, 11, 27, 20, 21, 33, 12, 28, 21, 22, 34, 28, 29, 29, 30, 36, 29, 45,
                        30, 23,
                    ],
                    [
                        0, 17, 26, 19, 35, 21, 25, 34, 20, 28, 29, 33, 27, 28, 29, 22, 34, 28, 44,
                        37, 38,
                    ],
                    [
                        9, 5, 10, 13, 13, 10, 9, 10, 13, 13, 13, 9, 10, 10, 10, 13, 8, 9, 10, 10,
                        13,
                    ],
                    qp,
                    inter,
                ),
                init(
                    [40, 33, 27, 28, 21, 37, 36, 37, 45, 38, 46],
                    [0, 25, 19, 20, 13, 14, 57, 44, 30, 30, 23],
                    [8, 8, 9, 12, 12, 10, 5, 9, 9, 9, 13],
                    qp,
                    inter,
                ),
            ],
            gt2: [
                init(
                    [
                        25, 1, 40, 25, 33, 11, 17, 25, 25, 18, 4, 17, 33, 26, 19, 13, 33, 19, 20,
                        28, 22,
                    ],
                    [
                        17, 0, 1, 17, 25, 18, 0, 9, 25, 33, 34, 9, 25, 18, 26, 20, 25, 18, 19, 27,
                        29,
                    ],
                    [
                        1, 5, 9, 9, 9, 6, 5, 9, 10, 10, 9, 9, 9, 9, 9, 9, 6, 8, 9, 9, 10,
                    ],
                    qp,
                    inter,
                ),
                init(
                    [40, 9, 25, 18, 26, 35, 25, 26, 35, 28, 37],
                    [17, 9, 25, 10, 18, 4, 17, 33, 19, 20, 29],
                    [1, 5, 8, 8, 9, 6, 6, 9, 8, 8, 9],
                    qp,
                    inter,
                ),
            ],
            last_x: [
                init(
                    [
                        13, 5, 4, 21, 14, 4, 6, 14, 21, 11, 14, 7, 14, 5, 11, 21, 30, 22, 13, 42,
                    ],
                    [
                        6, 13, 12, 6, 6, 12, 14, 14, 13, 12, 29, 7, 6, 13, 36, 28, 14, 13, 5, 26,
                    ],
                    [8, 5, 4, 5, 4, 4, 5, 4, 1, 0, 4, 1, 0, 0, 0, 0, 1, 0, 0, 0],
                    qp,
                    inter,
                ),
                init([12, 4, 3], [12, 4, 18], [5, 4, 4], qp, inter),
            ],
            dependent_sig: [
                [
                    init(
                        [11, 38, 46, 54, 27, 39, 39, 39, 44, 39, 39, 39],
                        [19, 38, 38, 46, 34, 54, 54, 39, 6, 39, 39, 39],
                        [9, 13, 8, 8, 8, 8, 8, 5, 8, 0, 0, 0],
                        qp,
                        inter,
                    ),
                    init(
                        [19, 46, 38, 39, 52, 39, 39, 39],
                        [35, 45, 53, 54, 44, 39, 39, 39],
                        [8, 12, 12, 8, 4, 0, 0, 0],
                        qp,
                        inter,
                    ),
                ],
                [
                    init(
                        [18, 39, 39, 39, 27, 39, 39, 39, 0, 39, 39, 39],
                        [19, 39, 54, 39, 19, 39, 39, 39, 56, 39, 39, 39],
                        [8, 8, 8, 8, 8, 0, 4, 4, 0, 0, 0, 0],
                        qp,
                        inter,
                    ),
                    init(
                        [11, 39, 39, 39, 19, 39, 39, 39],
                        [34, 38, 62, 39, 26, 39, 39, 39],
                        [8, 8, 8, 8, 4, 0, 0, 0],
                        qp,
                        inter,
                    ),
                ],
            ],
            dependent: false,
            last_y: [
                init(
                    [
                        13, 5, 4, 6, 13, 11, 14, 6, 5, 3, 14, 22, 6, 4, 3, 6, 22, 29, 20, 34,
                    ],
                    [
                        5, 5, 12, 6, 6, 4, 6, 14, 5, 12, 14, 7, 13, 5, 13, 21, 14, 20, 12, 34,
                    ],
                    [8, 5, 8, 5, 5, 4, 5, 5, 4, 0, 5, 4, 1, 0, 0, 1, 4, 0, 0, 0],
                    qp,
                    inter,
                ),
                init([12, 4, 3], [11, 4, 18], [6, 5, 5], qp, inter),
            ],
        }
    }
}
impl CoeffContexts {
    /// The `sig_coeff_flag` context `index` of channel `ch` in quantiser
    /// state `state` (always 0 without dependent quantisation).
    pub(crate) fn significance(&mut self, ch: usize, state: u8, index: usize) -> &mut Context {
        match state {
            0 | 1 => &mut self.sig[ch][index],
            _ => &mut self.dependent_sig[usize::from(state) - 2][ch][index],
        }
    }

    /// [`Self::significance`] as it stands, for pricing.
    pub(crate) fn sig_context(&self, ch: usize, state: u8, index: usize) -> Context {
        match state {
            0 | 1 => self.sig[ch][index],
            _ => self.dependent_sig[usize::from(state) - 2][ch][index],
        }
    }
}

fn init<const N: usize, const M: usize>(
    intra: [u8; N],
    predicted: [u8; N],
    windows: [u8; N],
    qp: u32,
    inter: bool,
) -> [Context; M] {
    const { assert!(N <= M) };
    let values = if inter { predicted } else { intra };
    let mut contexts = [Context::UNUSED; M];
    for ((context, value), window) in contexts.iter_mut().zip(values).zip(windows) {
        *context = Context::new(value, window, qp);
    }
    contexts
}
