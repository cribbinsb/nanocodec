//! H.266 affine candidate lists of one-reference CUs: the sub-block merge
//! list (§8.5.5.2: inherited, then constructed, then zero models; no
//! sub-block temporal candidate) and the affine AMVP list (§8.5.5.7), from
//! the coded neighbours in the `MotionGrid` and, in pictures using
//! temporal candidates, the collocated motion.
//!
//! Derived from `VVdeC` `UnitTools::getAffineMergeCand`,
//! `fillAffineMvpCand` and `xInheritedAffineMv` (BSD-3-Clause-Clear;
//! `THIRD_PARTY.md`).
use crate::affine::{Model, round_shift};
use crate::motion::{MotionGrid, Mv16};
use nc_encoder::Block;

/// `MaxNumSubblockMergeCand` this encoder signals
/// (`sps_five_minus_max_num_subblock_merge_cand` = 5 − this).
pub(crate) const MAX_SUBBLOCK_MERGE: usize = 5;
/// Affine AMVP candidates (§8.5.5.7: always two).
pub(crate) const AFFINE_AMVP: usize = 2;
/// §8.5.5.5: control-point differences at `1 << SHIFT` per sample.
const SHIFT: u32 = 7;

/// The sub-block merge list of `cu`, `merge_subblock_idx` order, with
/// `limit` (`MaxNumSubblockMergeCand`) entries; `six` is
/// `sps_6param_affine_enabled_flag`.
pub(crate) fn merge_list(grid: &MotionGrid, cu: &Block, limit: usize, six: bool) -> Vec<Model> {
    let mut list = Vec::with_capacity(limit);
    let corners = Corners::of(cu);
    // Inherited: the first affine CU at A0, A1, then at B0, B1, B2.
    for positions in [
        &[corners.a0, corners.a1][..],
        &[corners.b0, corners.b1, corners.b2],
    ] {
        if let Some(neighbour) = positions.iter().find_map(|&(x, y)| grid.affine_at(x, y)) {
            list.push(inherit(grid, cu, neighbour, neighbour.1.six));
        }
    }
    // Constructed from the corners' vectors (§8.5.5.6): top-left B2, B3,
    // A2; top-right B1, B0; bottom-left A1, A0; bottom-right the collocated
    // block. 6-parameter models from three corners (a missing top-left,
    // top-right or bottom-left one completed as the parallelogram's fourth
    // corner), then 4-parameter models.
    let first =
        |positions: &[(isize, isize)]| positions.iter().find_map(|&(x, y)| grid.mv_at(x, y));
    let lt = first(&[corners.b2, corners.b3, corners.a2]);
    let rt = first(&[corners.b1, corners.b0]);
    let lb = first(&[corners.a1, corners.a0]);
    let rb = grid.temporal_corner(cu);
    let fourth = |a: Mv16, b: Mv16, c: Mv16| {
        Mv16 {
            x: a.x + b.x - c.x,
            y: a.y + b.y - c.y,
        }
        .clip()
    };
    let three = |a: Option<Mv16>, b: Option<Mv16>, c: Option<Mv16>| {
        six.then(|| a.zip(b).zip(c).map(|((a, b), c)| (a, b, c)))
            .flatten()
    };
    let constructed = [
        three(lt, rt, lb).map(|(lt, rt, lb)| model([lt, rt, lb], true)),
        three(lt, rt, rb).map(|(lt, rt, rb)| model([lt, rt, fourth(rb, lt, rt)], true)),
        three(lt, lb, rb).map(|(lt, lb, rb)| model([lt, fourth(rb, lt, lb), lb], true)),
        three(rt, lb, rb).map(|(rt, lb, rb)| model([fourth(rt, lb, rb), rt, lb], true)),
        lt.zip(rt)
            .map(|(lt, rt)| model([lt, rt, Mv16::ZERO], false)),
        lt.zip(lb).map(|(lt, lb)| {
            model(
                [lt, top_right_from_bottom_left(cu, lt, lb), Mv16::ZERO],
                false,
            )
        }),
    ];
    list.extend(constructed.into_iter().flatten());
    list.truncate(limit);
    list.resize(limit, model([Mv16::ZERO; 3], false));
    list
}

/// The affine AMVP list of `cu` for a `six`-parameter model, its control
/// points rounded to quarter samples (§8.5.5.7; no AMVR for affine CUs).
pub(crate) fn amvp_list(grid: &MotionGrid, cu: &Block, six: bool) -> [Model; AFFINE_AMVP] {
    const QUARTER: u32 = 2;
    let round = |m: Model| Model {
        cp: m.cp.map(|v| v.round(QUARTER)),
        six,
    };
    let mut list = Vec::with_capacity(AFFINE_AMVP);
    let corners = Corners::of(cu);
    for positions in [
        &[corners.a0, corners.a1][..],
        &[corners.b0, corners.b1, corners.b2],
    ] {
        if let Some(neighbour) = positions.iter().find_map(|&(x, y)| grid.affine_at(x, y)) {
            list.push(round(inherit(grid, cu, neighbour, six)));
        }
    }
    if list.len() < AFFINE_AMVP {
        let first = |positions: &[(isize, isize)]| {
            positions
                .iter()
                .find_map(|&(x, y)| grid.mv_at(x, y))
                .map(|v| v.round(QUARTER))
        };
        let corner = [
            first(&[corners.b2, corners.b3, corners.a2]),
            first(&[corners.b1, corners.b0]),
            first(&[corners.a1, corners.a0]),
        ];
        let available = |n: usize| corner[..n].iter().all(Option::is_some);
        if available(3) || (!six && available(2)) {
            let cp = corner.map(Option::unwrap_or_default);
            list.push(Model { cp, six });
        }
        // Then translational models from single corners, bottom-left
        // first, then from the collocated vector.
        for v in corner.into_iter().rev().flatten() {
            if list.len() >= AFFINE_AMVP {
                break;
            }
            list.push(Model { cp: [v; 3], six });
        }
        if list.len() < AFFINE_AMVP
            && let Some(v) = grid.temporal_candidate(cu)
        {
            list.push(Model {
                cp: [v.round(QUARTER); 3],
                six,
            });
        }
    }
    list.resize(
        AFFINE_AMVP,
        Model {
            cp: [Mv16::ZERO; 3],
            six,
        },
    );
    [list[0], list[1]]
}

/// Whether the motion around `cu` varies: an affine neighbour, or a
/// corner neighbour (§8.5.5.6's control-point positions) whose vector is
/// not `mv`. Where every neighbour moves by `mv`, a fitted model rarely
/// beats the vector (the encoder's gate for affine AMVP).
pub(crate) fn motion_varies(grid: &MotionGrid, cu: &Block, mv: Mv16) -> bool {
    let c = Corners::of(cu);
    [c.a0, c.a1, c.a2, c.b0, c.b1, c.b2, c.b3]
        .into_iter()
        .any(|(x, y)| grid.affine_at(x, y).is_some() || grid.mv_at(x, y).is_some_and(|v| v != mv))
}

/// The neighbour positions of `cu` (§8.5.5.2: A0 below-left, A1 left,
/// A2 left of the top-left sample, B0 above-right, B1 above, B2
/// above-left, B3 above the top-left sample).
struct Corners {
    a0: (isize, isize),
    a1: (isize, isize),
    a2: (isize, isize),
    b0: (isize, isize),
    b1: (isize, isize),
    b2: (isize, isize),
    b3: (isize, isize),
}

impl Corners {
    fn of(cu: &Block) -> Self {
        let (x, y) = (cu.x as isize, cu.y as isize);
        let (w, h) = (cu.width() as isize, cu.height() as isize);
        Self {
            a0: (x - 1, y + h),
            a1: (x - 1, y + h - 1),
            a2: (x - 1, y),
            b0: (x + w, y - 1),
            b1: (x + w - 1, y - 1),
            b2: (x - 1, y - 1),
            b3: (x, y - 1),
        }
    }
}

fn model(cp: [Mv16; 3], six: bool) -> Model {
    Model { cp, six }
}

/// §8.5.5.6 constructed model 5 (`LT`, `LB`): the top-right control point
/// of the 4-parameter model through `lt` and `lb`.
fn top_right_from_bottom_left(cu: &Block, lt: Mv16, lb: Mv16) -> Mv16 {
    let shift = SHIFT + u32::from(cu.log2_width) - u32::from(cu.log2_height);
    let (x0, y0) = (i64::from(lt.x) << SHIFT, i64::from(lt.y) << SHIFT);
    Mv16 {
        x: round_shift(x0 + (i64::from(lb.y - lt.y) << shift), SHIFT),
        y: round_shift(y0 - (i64::from(lb.x - lt.x) << shift), SHIFT),
    }
    .clip()
}

/// §8.5.5.5: the control points of `cu` extrapolated from the affine
/// neighbour (`block`, `model`), for a `six`-parameter model. A neighbour
/// in the CTU row above lends its bottom sub-block row instead (line
/// buffer), as a 4-parameter model.
fn inherit(grid: &MotionGrid, cu: &Block, (block, model): (Block, Model), six: bool) -> Model {
    let (nx, mut ny) = (block.x as i64, block.y as i64);
    let (nw, nh) = (block.width(), block.height());
    let [mut lt, mut rt, lb] = model.cp;
    let bottom = block.y + nh;
    let top_ctu_boundary = bottom.is_multiple_of(grid.ctu_size()) && bottom == cu.y;
    if top_ctu_boundary {
        let row = (bottom - 1) as isize;
        let sub = |x: usize| grid.mv_at(x as isize, row).expect("a coded neighbour");
        lt = sub(block.x);
        rt = sub(block.x + nw - 1);
        ny += nh as i64;
    }
    let scale = |d: i32, log2: u32| i64::from(d) << (SHIFT - log2);
    let (log2_w, log2_h) = (nw.ilog2(), nh.ilog2());
    let hor = (scale(rt.x - lt.x, log2_w), scale(rt.y - lt.y, log2_w));
    let ver = if model.six && !top_ctu_boundary {
        (scale(lb.x - lt.x, log2_h), scale(lb.y - lt.y, log2_h))
    } else {
        (-hor.1, hor.0)
    };
    let base = (i64::from(lt.x) << SHIFT, i64::from(lt.y) << SHIFT);
    let at = |x: usize, y: usize| {
        let (dx, dy) = (x as i64 - nx, y as i64 - ny);
        Mv16 {
            x: round_shift(base.0 + hor.0 * dx + ver.0 * dy, SHIFT),
            y: round_shift(base.1 + hor.1 * dx + ver.1 * dy, SHIFT),
        }
        .clip()
    };
    let cp2 = if six {
        at(cu.x, cu.y + cu.height())
    } else {
        Mv16::ZERO
    };
    Model {
        cp: [at(cu.x, cu.y), at(cu.x + cu.width(), cu.y), cp2],
        six,
    }
}
