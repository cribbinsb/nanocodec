//! General-slice CABAC initialization, including B/P bank switching (§9.3.2).
//! Numeric tables from pinned BSD-Clear `VVdeC` `Contexts.cpp`; see `THIRD_PARTY.md`.
use crate::{
    coeff_contexts::CoeffContexts,
    contexts::{Context, Contexts},
};

pub(super) fn bank<const N: usize, const M: usize>(
    rows: [[u8; M]; 3],
    windows: [u8; M],
    qp: u32,
    kind: u8,
) -> [Context; N] {
    let mut result = [Context::UNUSED; N];
    for ((out, value), window) in result.iter_mut().zip(rows[usize::from(kind)]).zip(windows) {
        *out = Context::new(value, window, qp);
    }
    result
}

pub(super) fn contexts(qp: u32, kind: u8) -> Contexts {
    let mut result = Contexts::new(qp, kind != 2);
    if kind != 0 {
        return result;
    }
    result.sao_merge = bank::<1, 1>([[2], [60], [60]], [0], qp, kind)[0];
    result.sao_type = bank::<1, 1>([[2], [5], [13]], [4], qp, kind)[0];
    result.alf_ctb = bank(
        [[33, 52, 46], [13, 23, 46], [62, 39, 39]],
        [0, 0, 0],
        qp,
        kind,
    );
    result.skip = bank(
        [[57, 60, 46], [57, 59, 45], [0, 26, 28]],
        [5, 4, 8],
        qp,
        kind,
    );
    result.merge = bank::<1, 1>([[6], [21], [26]], [4], qp, kind)[0];
    result.merge_idx = bank::<1, 1>([[18], [20], [34]], [4], qp, kind)[0];
    result.mvd = bank([[51, 36], [44, 43], [14, 45]], [9, 5], qp, kind);
    result.root_cbf = bank::<1, 1>([[12], [5], [6]], [4], qp, kind)[0];
    result.mpm = bank::<1, 1>([[44], [36], [45]], [6], qp, kind)[0];
    result.planar = bank([[13, 6], [12, 20], [13, 28]], [1, 5], qp, kind);
    result.isp_mode = bank::<1, 1>([[33], [33], [33]], [9], qp, kind)[0];
    result.isp_split = bank::<1, 1>([[43], [36], [43]], [2], qp, kind)[0];
    result.cbf_y_isp = bank([[5, 14], [20, 7], [5, 7]], [8, 9], qp, kind);
    result.cbf_y = bank::<1, 1>([[15], [23], [15]], [5], qp, kind)[0];
    result.cbf_cb = bank::<1, 1>([[25], [25], [12]], [5], qp, kind)[0];
    result.cbf_cr = bank([[9, 36], [25, 29], [33, 28]], [2, 1], qp, kind);
    result
}

fn row<const N: usize, const M: usize>(values: [u8; M], windows: [u8; M], qp: u32) -> [Context; N] {
    bank([values; 3], windows, qp, 0)
}

pub(super) fn coefficients(qp: u32, kind: u8) -> CoeffContexts {
    let mut result = CoeffContexts::new(qp, kind != 2);
    if kind == 0 {
        significance(&mut result, qp);
        levels(&mut result, qp);
        last_position(&mut result, qp);
    }
    result
}

fn significance(result: &mut CoeffContexts, qp: u32) {
    result.group = [row([25, 45], [8, 5], qp), row([25, 14], [5, 8], qp)];
    result.sig = [
        row(
            [17, 41, 49, 36, 1, 49, 50, 37, 48, 51, 58, 45],
            [12, 9, 9, 10, 9, 9, 9, 10, 8, 8, 8, 10],
            qp,
        ),
        row(
            [9, 49, 50, 36, 48, 59, 59, 38],
            [12, 12, 9, 13, 4, 5, 8, 9],
            qp,
        ),
    ];
}

fn levels(result: &mut CoeffContexts, qp: u32) {
    result.parity = [
        row(
            [
                33, 40, 25, 41, 26, 42, 25, 33, 26, 34, 27, 25, 41, 42, 42, 35, 33, 27, 35, 42, 43,
            ],
            [
                8, 9, 12, 13, 13, 13, 10, 13, 13, 13, 13, 13, 13, 13, 13, 13, 10, 13, 13, 13, 13,
            ],
            qp,
        ),
        row(
            [33, 25, 26, 34, 19, 27, 33, 42, 43, 35, 43],
            [8, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13],
            qp,
        ),
    ];
    result.gt1 = [
        row(
            [
                0, 0, 33, 34, 35, 21, 25, 34, 35, 28, 29, 40, 42, 43, 29, 30, 49, 36, 37, 45, 38,
            ],
            [
                9, 5, 10, 13, 13, 10, 9, 10, 13, 13, 13, 9, 10, 10, 10, 13, 8, 9, 10, 10, 13,
            ],
            qp,
        ),
        row(
            [0, 40, 34, 43, 36, 37, 57, 52, 45, 38, 46],
            [8, 8, 9, 12, 12, 10, 5, 9, 9, 9, 13],
            qp,
        ),
    ];
    result.gt2 = [
        row(
            [
                25, 0, 0, 17, 25, 26, 0, 9, 25, 33, 19, 0, 25, 33, 26, 20, 25, 33, 27, 35, 22,
            ],
            [
                1, 5, 9, 9, 9, 6, 5, 9, 10, 10, 9, 9, 9, 9, 9, 9, 6, 8, 9, 9, 10,
            ],
            qp,
        ),
        row(
            [25, 1, 25, 33, 26, 12, 25, 33, 27, 28, 37],
            [1, 5, 8, 8, 9, 6, 6, 9, 8, 8, 9],
            qp,
        ),
    ];
}

fn last_position(result: &mut CoeffContexts, qp: u32) {
    result.last_x = [
        row(
            [
                6, 6, 12, 14, 6, 4, 14, 7, 6, 4, 29, 7, 6, 6, 12, 28, 7, 13, 13, 35,
            ],
            [8, 5, 4, 5, 4, 4, 5, 4, 1, 0, 4, 1, 0, 0, 0, 0, 1, 0, 0, 0],
            qp,
        ),
        row([19, 5, 4], [5, 4, 4], qp),
    ];
    result.last_y = [
        row(
            [
                5, 5, 20, 13, 13, 19, 21, 6, 12, 12, 14, 14, 5, 4, 12, 13, 7, 13, 12, 41,
            ],
            [8, 5, 8, 5, 5, 4, 5, 5, 4, 0, 5, 4, 1, 0, 0, 1, 4, 0, 0, 0],
            qp,
        ),
        row([11, 5, 27], [6, 5, 5], qp),
    ];
}
