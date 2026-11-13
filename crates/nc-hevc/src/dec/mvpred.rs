//! Motion vector prediction (H.265 §8.5.3.2): the merge candidate list
//! with spatial, temporal, combined bi-predictive and zero candidates
//! (§8.5.3.2.2–5), AMVP (§8.5.3.2.6–7) and the temporal predictor read
//! from the collocated picture (§8.5.3.2.8–9).

#![allow(
    clippy::many_single_char_names,
    reason = "H.265 sample and block variables keep their specification names (x, y, w, h, n)"
)]

use nc_encoder::Mv;

use super::picture::{Motion, PictureState, PuMotion, RefPic, flag};

/// Most merge candidates (`MaxNumMergeCand` ≤ 5).
const MAX_MERGE: usize = 5;
/// `l0CandIdx` / `l1CandIdx` of combined bi-predictive candidates (§8.5.3.2.4).
const COMBINATIONS: [(usize, usize); 12] = [
    (0, 1),
    (1, 0),
    (0, 2),
    (2, 0),
    (1, 2),
    (2, 1),
    (0, 3),
    (3, 0),
    (1, 3),
    (3, 1),
    (2, 3),
    (3, 2),
];
/// TMVP reads the collocated motion on a 16x16 grid (§8.5.3.2.8).
const COL_GRID_LOG2: u32 = super::picture::TMVP_GRID_LOG2;

/// A prediction block and the coding block it belongs to.
#[derive(Debug, Clone, Copy)]
pub(super) struct Pu {
    pub cb: (usize, usize),
    pub cb_size: usize,
    pub x: usize,
    pub y: usize,
    pub w: usize,
    pub h: usize,
    pub part_idx: usize,
    /// The second partition of a vertical split (`Nx2N`, `nLx2N`, `nRx2N`)
    /// excludes A1; of a horizontal split, B1.
    pub vertical_split: bool,
    pub horizontal_split: bool,
}

/// The collocated picture of a slice.
#[derive(Debug, Clone, Copy)]
pub(super) struct Collocated<'a> {
    pub motion: &'a Motion,
    pub poc: i32,
}

/// What motion prediction of one slice needs.
pub(super) struct Predictor<'a> {
    pub pic: &'a PictureState,
    pub slice: u16,
    pub lists: &'a [Vec<RefPic>; 2],
    pub poc: i32,
    pub col: Option<Collocated<'a>>,
    pub collocated_from_l0: bool,
    /// `NoBackwardPredFlag` (§8.5.3.2.9).
    pub no_backward_pred: bool,
    pub log2_par_mrg_level: u8,
    pub max_merge: usize,
    pub b_slice: bool,
    pub num_ref_idx: [usize; 2],
}

/// `Clip3(−128, 127, ·)` of a POC distance.
fn clip_distance(d: i32) -> i32 {
    d.clamp(-128, 127)
}

/// POC-distance scaling of a vector (equations 8-180 to 8-183).
fn scale(mv: Mv, td: i32, tb: i32) -> Mv {
    let (td, tb) = (clip_distance(td), clip_distance(tb));
    if td == 0 {
        return mv;
    }
    let tx = (16384 + (td.abs() >> 1)) / td;
    let factor = ((tb * tx + 32) >> 6).clamp(-4096, 4095);
    let component = |v: i16| {
        let p = factor * i32::from(v);
        (p.signum() * ((p.abs() + 127) >> 8)).clamp(-32768, 32767) as i16
    };
    Mv {
        x: component(mv.x),
        y: component(mv.y),
    }
}

impl Predictor<'_> {
    /// Prediction block availability (§6.4.2) of the block at `(xn, yn)`,
    /// and its motion when available and not intra.
    fn neighbour(&self, pu: &Pu, xn: isize, yn: isize) -> Option<PuMotion> {
        let (xcb, ycb) = (pu.cb.0 as isize, pu.cb.1 as isize);
        let s = pu.cb_size as isize;
        let same_cb = xn >= xcb && xn < xcb + s && yn >= ycb && yn < ycb + s;
        let available = if same_cb {
            // Inside the coding block every earlier partition is
            // available, except the third for the second NxN one (§6.4.2).
            !(pu.w * 2 == pu.cb_size
                && pu.h * 2 == pu.cb_size
                && pu.part_idx == 1
                && ycb + pu.h as isize <= yn
                && xcb + pu.w as isize > xn)
        } else {
            self.pic.available((pu.x, pu.y), (xn, yn), self.slice)
        };
        if !available {
            return None;
        }
        let i = self.pic.block(xn as usize, yn as usize);
        if self.pic.info[i].flags & flag::INTRA != 0 {
            return None;
        }
        Some(self.pic.motion_at(i))
    }

    fn ref_pic(&self, list: usize, idx: i8) -> Option<RefPic> {
        self.lists[list].get(usize::try_from(idx).ok()?).copied()
    }

    /// The merge candidate `merge_idx` (§8.5.3.2.2).
    pub fn merge(&self, pu: &Pu, merge_idx: usize) -> PuMotion {
        let original = *pu;
        let mut pu = *pu;
        if self.log2_par_mrg_level > 2 && pu.cb_size == 8 {
            // singleMCLFlag: all PUs of an 8x8 CU share the 2Nx2N list.
            pu = Pu {
                x: pu.cb.0,
                y: pu.cb.1,
                w: 8,
                h: 8,
                part_idx: 0,
                vertical_split: false,
                horizontal_split: false,
                ..pu
            };
        }
        let mut list = [PuMotion::NONE; MAX_MERGE];
        let mut n = self.spatial_merge(&pu, &mut list, merge_idx + 1);
        if n <= merge_idx {
            let mut temporal = PuMotion::NONE;
            for l in 0..if self.b_slice { 2 } else { 1 } {
                if let Some(mv) = self.temporal(&pu, l, 0) {
                    temporal.mv[l] = mv;
                    temporal.ref_idx[l] = 0;
                }
            }
            if temporal.uses(0) || temporal.uses(1) {
                list[n] = temporal;
                n += 1;
            }
        }
        if n <= merge_idx && self.b_slice {
            n = self.combined(&mut list, n, merge_idx);
        }
        if n <= merge_idx {
            // Zero candidates (§8.5.3.2.5).
            let refs = if self.b_slice {
                self.num_ref_idx[0].min(self.num_ref_idx[1])
            } else {
                self.num_ref_idx[0]
            };
            let mut zero = 0;
            while n <= merge_idx {
                let r = if zero < refs { zero as i8 } else { 0 };
                list[n] = PuMotion {
                    mv: [Mv::default(); 2],
                    ref_idx: [r, if self.b_slice { r } else { -1 }],
                };
                n += 1;
                zero += 1;
            }
        }
        let mut chosen = list[merge_idx];
        // 8x4 and 4x8 blocks are restricted to uni-prediction (8.5.3.2.2).
        if original.w + original.h == 12 && chosen.uses(0) && chosen.uses(1) {
            chosen.ref_idx[1] = -1;
            chosen.mv[1] = Mv::default();
        }
        chosen
    }

    /// Spatial merge candidates (§8.5.3.2.3) in order A1, B1, B0, A0, B2.
    fn spatial_merge(&self, pu: &Pu, list: &mut [PuMotion; MAX_MERGE], needed: usize) -> usize {
        let (x, y, w, h) = (pu.x as isize, pu.y as isize, pu.w as isize, pu.h as isize);
        let level = self.log2_par_mrg_level;
        let same_region =
            |xn: isize, yn: isize| (x >> level) == (xn >> level) && (y >> level) == (yn >> level);
        let get = |xn: isize, yn: isize, excluded: bool| {
            if excluded || same_region(xn, yn) {
                None
            } else {
                self.neighbour(pu, xn, yn)
            }
        };
        let differs = |c: Option<PuMotion>, other: Option<PuMotion>| match (c, other) {
            (Some(c), Some(o)) if c == o => None,
            (c, _) => c,
        };
        let mut n = 0;
        let mut push = |c: Option<PuMotion>| {
            if let Some(c) = c {
                list[n] = c;
                n += 1;
            }
            n
        };
        // Candidates are derived in list order and only until the one
        // `merge_idx` selects exists. Pruning compares with the neighbours'
        // motion whether or not they were themselves pruned.
        let a1 = get(x - 1, y + h - 1, pu.vertical_split && pu.part_idx == 1);
        if push(a1) >= needed {
            return n;
        }
        let b1 = get(x + w - 1, y - 1, pu.horizontal_split && pu.part_idx == 1);
        if push(differs(b1, a1)) >= needed {
            return n;
        }
        let b0 = get(x + w, y - 1, false);
        if push(differs(b0, b1)) >= needed {
            return n;
        }
        let a0 = get(x - 1, y + h, false);
        let count = push(differs(a0, a1));
        if count < needed && count < 4 {
            let b2 = get(x - 1, y - 1, false);
            return push(differs(differs(b2, a1), b1));
        }
        count
    }

    /// Combined bi-predictive candidates (§8.5.3.2.4).
    fn combined(&self, list: &mut [PuMotion; MAX_MERGE], orig: usize, merge_idx: usize) -> usize {
        let mut n = orig;
        if orig < 2 || orig >= self.max_merge {
            return n;
        }
        for &(i0, i1) in COMBINATIONS.iter().take(orig * (orig - 1)) {
            let (c0, c1) = (list[i0], list[i1]);
            if c0.uses(0) && c1.uses(1) {
                let p0 = self.ref_pic(0, c0.ref_idx[0]).map(|r| r.poc);
                let p1 = self.ref_pic(1, c1.ref_idx[1]).map(|r| r.poc);
                if p0 != p1 || c0.mv[0] != c1.mv[1] {
                    list[n] = PuMotion {
                        mv: [c0.mv[0], c1.mv[1]],
                        ref_idx: [c0.ref_idx[0], c1.ref_idx[1]],
                    };
                    n += 1;
                }
            }
            if n == self.max_merge || n > merge_idx {
                break;
            }
        }
        n
    }

    /// The temporal predictor of list `list` for `ref_idx` (§8.5.3.2.8).
    fn temporal(&self, pu: &Pu, list: usize, ref_idx: usize) -> Option<Mv> {
        let col = self.col?;
        let log2_ctb = self.pic.layout.log2_ctb;
        let (xbr, ybr) = (pu.x + pu.w, pu.y + pu.h);
        if pu.y >> log2_ctb == ybr >> log2_ctb && ybr < self.pic.height && xbr < self.pic.width {
            let at = |v: usize| (v >> COL_GRID_LOG2) << COL_GRID_LOG2;
            if let Some(mv) = self.collocated(col, (at(xbr), at(ybr)), list, ref_idx) {
                return Some(mv);
            }
        }
        let (xc, yc) = (pu.x + (pu.w >> 1), pu.y + (pu.h >> 1));
        let at = |v: usize| (v >> COL_GRID_LOG2) << COL_GRID_LOG2;
        self.collocated(col, (at(xc), at(yc)), list, ref_idx)
    }

    /// Collocated motion vectors (§8.5.3.2.9).
    fn collocated(
        &self,
        col: Collocated<'_>,
        (x, y): (usize, usize),
        list: usize,
        ref_idx: usize,
    ) -> Option<Mv> {
        let (motion, col_lists) = col.motion.at(x, y)?;
        let source = if !motion.uses(0) {
            1
        } else if !motion.uses(1) {
            0
        } else if self.no_backward_pred {
            list
        } else {
            usize::from(self.collocated_from_l0)
        };
        if !motion.uses(source) {
            return None;
        }
        let col_ref = col_lists[source].get(motion.ref_idx[source] as usize)?;
        let target = self.lists[list].get(ref_idx)?;
        if target.long_term != col_ref.long_term {
            return None;
        }
        let mv = motion.mv[source];
        let col_diff = col.poc - col_ref.poc;
        let cur_diff = self.poc - target.poc;
        Some(if target.long_term || col_diff == cur_diff {
            mv
        } else {
            scale(mv, col_diff, cur_diff)
        })
    }

    /// The AMVP predictor `mvp_lX_flag` of list `list` for `ref_idx`
    /// (§8.5.3.2.6, §8.5.3.2.7).
    pub fn amvp(&self, pu: &Pu, list: usize, ref_idx: usize, mvp_flag: usize) -> Mv {
        let target = self.lists[list][ref_idx];
        let (x, y, w, h) = (pu.x as isize, pu.y as isize, pu.w as isize, pu.h as isize);
        let a = [
            self.neighbour(pu, x - 1, y + h),
            self.neighbour(pu, x - 1, y + h - 1),
        ];
        let b = [
            self.neighbour(pu, x + w, y - 1),
            self.neighbour(pu, x + w - 1, y - 1),
            self.neighbour(pu, x - 1, y - 1),
        ];
        let is_scaled = a.iter().any(Option::is_some);
        let mut mv_a = a
            .iter()
            .flatten()
            .find_map(|m| self.same_picture(m, list, target));
        if mv_a.is_none() {
            mv_a = a
                .iter()
                .flatten()
                .find_map(|m| self.scaled(m, list, target));
        }
        let mut mv_b = b
            .iter()
            .flatten()
            .find_map(|m| self.same_picture(m, list, target));
        if !is_scaled {
            if mv_b.is_some() && mv_a.is_none() {
                mv_a = mv_b;
            }
            mv_b = b
                .iter()
                .flatten()
                .find_map(|m| self.scaled(m, list, target));
        }
        let mut candidates = [Mv::default(); 2];
        let mut n = 0;
        if let Some(m) = mv_a {
            candidates[n] = m;
            n += 1;
        }
        if let Some(m) = mv_b
            && !(mv_a == Some(m))
        {
            candidates[n] = m;
            n += 1;
        }
        if n < 2
            && mvp_flag >= n
            && let Some(m) = self.temporal(pu, list, ref_idx)
        {
            candidates[n] = m;
        }
        candidates[mvp_flag]
    }

    /// A neighbour's vector that refers to the target picture itself,
    /// from list X first, then Y.
    fn same_picture(&self, m: &PuMotion, list: usize, target: RefPic) -> Option<Mv> {
        [list, 1 - list].into_iter().find_map(|l| {
            (m.uses(l) && self.ref_pic(l, m.ref_idx[l])?.id == target.id).then_some(m.mv[l])
        })
    }

    /// A neighbour's vector to any picture of the same long-term-ness,
    /// scaled by POC distance between short-term pictures.
    fn scaled(&self, m: &PuMotion, list: usize, target: RefPic) -> Option<Mv> {
        [list, 1 - list].into_iter().find_map(|l| {
            if !m.uses(l) {
                return None;
            }
            let r = self.ref_pic(l, m.ref_idx[l])?;
            if r.long_term != target.long_term {
                return None;
            }
            Some(if r.long_term || target.long_term {
                m.mv[l]
            } else {
                scale(m.mv[l], self.poc - r.poc, self.poc - target.poc)
            })
        })
    }
}
