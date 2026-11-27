//! Intra edge preparation: resolves the coded mode, angle and edge
//! availability to a predictor ([`ImplMode`]) and copies the edges that
//! predictor reads from the picture into the edge buffer (layout in
//! [`super`]), extending unavailable samples as AV2 defines: past the
//! available run by repeating its last sample, a missing edge from the
//! other edge's first sample, or mid-grey-ish constants without either.
//!
//! Ported from dav2d `src/ipred_prepare_tmpl.c` (`dav2d_prepare_intra_edges`)
//! and `src/ipred_prepare.h` (commit 237229e, BSD-2-Clause, see
//! `docs/licenses/dav2d`).

use nc_common::{BitDepth, Sample};

use super::{
    ANGLE_DIP_FLAG, ANGLE_HAS_LEFT_FLAG, ANGLE_HAS_TOP_FLAG, ANGLE_IBP_FLAG, ANGLE_MASK,
    ANGLE_MULTI_MRL_FLAG, ANGLE_SEQHDR_IBP_FLAG, ANGLE_USE_EDGE_FILTER_FLAG, ImplMode,
    IntraPredMode, SampleRef, mrl_idx,
};

/// Directional angles (degrees) splitting the zones.
const VERTICAL: i32 = 90;
/// See [`VERTICAL`].
const HORIZONTAL: i32 = 180;
/// The corner filter needs a transform of at least this many 4-sample
/// units of width plus height.
const CORNER_FILTER_MIN_UNITS: i32 = 6;
/// log2 of the 4-sample unit of positions and counts.
const UNIT_LOG2: u32 = 2;
/// Bits of the corner filter (weights 5, 6, 5 out of 16).
const CORNER_FILTER_BITS: u32 = 4;

/// The edges a predictor reads (`EdgeMask`).
#[derive(Debug, Clone, Copy, Default)]
#[allow(clippy::struct_excessive_bools, reason = "dav2d's EdgeMask bit-field")]
struct EdgeNeeds {
    left: bool,
    top: bool,
    topleft: bool,
    topright: bool,
    bottomleft: bool,
}

impl EdgeNeeds {
    /// `intra_prediction_edges[mode]`.
    fn of(mode: ImplMode) -> Self {
        let (left, top, topleft, topright, bottomleft) = match mode {
            ImplMode::Dc => (true, true, false, false, false),
            ImplMode::Vert | ImplMode::TopDc => (false, true, false, false, false),
            ImplMode::Hor | ImplMode::LeftDc => (true, false, false, false, false),
            ImplMode::Dc128 => (false, false, false, false, false),
            ImplMode::Z1 => (false, true, true, true, false),
            ImplMode::Z2 | ImplMode::Paeth => (true, true, true, false, false),
            ImplMode::Z3 => (true, false, true, false, true),
            ImplMode::Smooth => (true, true, false, true, true),
            ImplMode::SmoothV => (false, true, false, false, true),
            ImplMode::SmoothH => (true, false, false, true, false),
            ImplMode::Dip => (true, true, true, true, true),
        };
        Self {
            left,
            top,
            topleft,
            topright,
            bottomleft,
        }
    }
}

/// The block and its tile for [`prepare_intra_edges`], all in 4-sample
/// units of the plane: dav2d's `x, y, w, h, n_tr, n_bl, mode, tw4, th4,
/// intra_flags`.
#[derive(Debug, Clone, Copy)]
pub struct EdgeRequest {
    /// The block's column in the plane.
    pub x: i32,
    /// The block's row in the plane.
    pub y: i32,
    /// The tile's end column (exclusive).
    pub end_x: i32,
    /// The tile's (or picture's) end row (exclusive).
    pub end_y: i32,
    /// Available units of the top-right edge.
    pub n_tr: i32,
    /// Available units of the bottom-left edge.
    pub n_bl: i32,
    /// The coded mode; never [`IntraPredMode::Cfl`] (chroma from luma
    /// prepares its DC as [`IntraPredMode::Dc`], as dav2d's caller does).
    pub mode: IntraPredMode,
    /// Transform width.
    pub tw4: i32,
    /// Transform height.
    pub th4: i32,
    /// The absolute angle (directional modes; DIP's mode with
    /// [`ANGLE_DIP_FLAG`]) OR'ed with the `ANGLE_*` flags, as [`super::predict`]
    /// receives them.
    pub intra_flags: i32,
}

/// The decoded `intra_flags`.
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools, reason = "the ANGLE_* flag bits")]
struct Flags {
    angle: i32,
    edge_filter: bool,
    dip: bool,
    ibp: bool,
    seq_ibp: bool,
    mrl: usize,
    multi_mrl: bool,
    have_left: bool,
    have_top: bool,
}

impl Flags {
    fn new(flags: i32) -> Self {
        Self {
            angle: flags & ANGLE_MASK,
            edge_filter: flags & ANGLE_USE_EDGE_FILTER_FLAG != 0,
            dip: flags & ANGLE_DIP_FLAG != 0,
            ibp: flags & ANGLE_IBP_FLAG != 0,
            seq_ibp: flags & ANGLE_SEQHDR_IBP_FLAG != 0,
            mrl: mrl_idx(flags),
            multi_mrl: flags & ANGLE_MULTI_MRL_FLAG != 0,
            have_left: flags & ANGLE_HAS_LEFT_FLAG != 0,
            have_top: flags & ANGLE_HAS_TOP_FLAG != 0,
        }
    }
}

/// The predictor for a coded mode, and whether it is directional.
fn resolve_mode(mode: IntraPredMode, f: &Flags) -> (ImplMode, bool) {
    use IntraPredMode as M;
    match mode {
        M::Vert
        | M::Hor
        | M::DiagDownLeft
        | M::DiagDownRight
        | M::VertRight
        | M::HorDown
        | M::HorUp
        | M::VertLeft => {
            // Bi-prediction needs the zone predictor even without its edge.
            let zone = if f.angle <= VERTICAL {
                if f.angle < VERTICAL && (f.have_top || f.ibp) {
                    ImplMode::Z1
                } else {
                    ImplMode::Vert
                }
            } else if f.angle < HORIZONTAL {
                ImplMode::Z2
            } else if f.angle > HORIZONTAL && (f.have_left || f.ibp) {
                ImplMode::Z3
            } else {
                ImplMode::Hor
            };
            (zone, true)
        }
        M::Dc if f.dip => (ImplMode::Dip, false),
        M::Dc => {
            let mode = match (f.have_left, f.have_top) {
                (false, false) => ImplMode::Dc128,
                (false, true) => ImplMode::TopDc,
                (true, false) => ImplMode::LeftDc,
                (true, true) => ImplMode::Dc,
            };
            (mode, false)
        }
        M::Paeth => {
            debug_assert!(!f.dip);
            let mode = match (f.have_left, f.have_top) {
                (false, false) => ImplMode::Dc128,
                (false, true) => ImplMode::Vert,
                (true, false) => ImplMode::Hor,
                (true, true) => ImplMode::Paeth,
            };
            (mode, false)
        }
        M::Smooth => (ImplMode::Smooth, false),
        M::SmoothV => (ImplMode::SmoothV, false),
        M::SmoothH => (ImplMode::SmoothH, false),
        M::Cfl => unreachable!("CfL predicts its DC through Dc"),
    }
}

/// Is `mode` one of the directional zone predictors.
fn is_zone(mode: ImplMode) -> bool {
    matches!(mode, ImplMode::Z1 | ImplMode::Z2 | ImplMode::Z3)
}

/// The picture rows above the block that the top edges come from: line
/// `mrl_idx` and line 0, `stride` apart vertically (0 for the pre-filter
/// row at a superblock's top, which serves every line).
#[derive(Clone, Copy)]
struct TopRows<'a, T> {
    line: SampleRef<'a, T>,
    line0: SampleRef<'a, T>,
    stride: isize,
}

/// The values of samples no edge provides, around mid-grey at the depth.
#[derive(Clone, Copy)]
struct MissingFill<T> {
    /// A missing left edge without a top edge (`((1 << bitdepth) >> 1) + 1`).
    left: T,
    /// A missing top edge without a left edge (`((1 << bitdepth) >> 1) - 1`).
    top: T,
    /// The corner without either edge (`(1 << bitdepth) >> 1`).
    corner: T,
}

impl<T: Sample> MissingFill<T> {
    fn new(depth: BitDepth) -> Self {
        let mid = depth.mid();
        Self {
            left: T::from_i32(mid + 1),
            top: T::from_i32(mid - 1),
            corner: T::from_i32(mid),
        }
    }
}

/// The edge buffer being written, with the fill of missing samples.
struct EdgeOut<'a, T> {
    buf: &'a mut [T],
    topleft: usize,
    missing: MissingFill<T>,
}

impl<T: Sample> EdgeOut<'_, T> {
    fn index(&self, i: isize) -> usize {
        super::index(self.topleft, i)
    }

    fn get(&self, i: isize) -> T {
        self.buf[self.index(i)]
    }

    fn set(&mut self, i: isize, v: T) {
        let at = self.index(i);
        self.buf[at] = v;
    }

    /// Fills `len` samples from `start` with `v`.
    fn fill(&mut self, start: isize, len: usize, v: T) {
        let at = self.index(start);
        self.buf[at..at + len].fill(v);
    }

    /// Copies `src` to the samples from `start`.
    fn copy(&mut self, start: isize, src: &[T]) {
        let at = self.index(start);
        self.buf[at..at + src.len()].copy_from_slice(src);
    }
}

/// Fills the edge buffer `edge` (corner at `topleft`; layout and size in
/// [`super`]) for the block of `req` whose first sample is at `offset` in
/// `picture` (rows `stride` apart), and returns the predictor to call
/// (`dav2d_prepare_intra_edges`).
///
/// The top edges come from `top_sb_edge`, the pre-filter row above a
/// superblock's top (at the plane's first column), when given, else from
/// the picture. Edges the returned predictor does not read are left as
/// they were; unavailable ones are extended as AV2 defines, around the
/// mid-grey of `depth`.
pub fn prepare_intra_edges<T: Sample>(
    req: &EdgeRequest,
    picture: &[T],
    offset: usize,
    stride: usize,
    top_sb_edge: Option<SampleRef<'_, T>>,
    edge: &mut [T],
    topleft: usize,
    depth: BitDepth,
) -> ImplMode {
    debug_assert!(req.y < req.end_y && req.x < req.end_x);
    let f = Flags::new(req.intra_flags);
    let (mode, is_dir) = resolve_mode(req.mode, &f);
    debug_assert!(f.mrl == 0 || is_dir);
    let corner_filter = is_zone(mode)
        && f.have_left
        && f.have_top
        && f.mrl == 0
        && f.edge_filter
        && req.tw4 + req.th4 >= CORNER_FILTER_MIN_UNITS
        && (f.seq_ibp || mode == ImplMode::Z2);

    let mut needs = EdgeNeeds::of(mode);
    if matches!(mode, ImplMode::Z1 | ImplMode::Z3) && f.ibp {
        // Bi-prediction also predicts from the opposite edge.
        needs = EdgeNeeds::of(ImplMode::Dip);
    }

    let pic = SampleRef::new(picture, offset);
    let stride = stride as isize;
    let reads_top = needs.top || needs.topleft || needs.topright;
    let reads_left = needs.left || needs.bottomleft;
    let top =
        (f.have_top && (reads_top || (reads_left && !f.have_left))).then(|| match top_sb_edge {
            Some(row) => {
                let line = row.shifted((req.x << UNIT_LOG2) as isize);
                TopRows {
                    line,
                    line0: line,
                    stride: 0,
                }
            }
            None => TopRows {
                line: pic.shifted(-((f.mrl as isize + 1) * stride)),
                line0: pic.shifted(-stride),
                stride,
            },
        });

    let shape = Shape::new(req, &f, mode, is_dir);
    let missing = MissingFill::new(depth.of::<T>());
    let mut out = EdgeOut {
        buf: edge,
        topleft,
        missing,
    };
    if needs.left || corner_filter {
        left_edge(&mut out, req, &f, &shape, needs, pic, stride, top);
    } else if needs.bottomleft {
        debug_assert_eq!(mode, ImplMode::SmoothV);
        let value = if !f.have_left {
            top.map_or(missing.left, |t| t.line.at(0))
        } else if req.n_bl <= 0 {
            let rows = shape.th.min((req.end_y - req.y) << UNIT_LOG2);
            pic.at(stride * (rows as isize - 1) - 1)
        } else {
            pic.at(stride * shape.th as isize - 1)
        };
        out.set(-(1 + shape.th as isize), value);
    }

    if needs.top || corner_filter {
        top_edge(&mut out, req, &f, &shape, needs, pic, top);
    } else if needs.topright {
        debug_assert_eq!(mode, ImplMode::SmoothH);
        let value = match top {
            None if f.have_left => pic.at(-1),
            None => missing.top,
            Some(t) if req.n_tr <= 0 => {
                let cols = shape.tw.min((req.end_x - req.x) << UNIT_LOG2);
                t.line.at(cols as isize - 1)
            }
            Some(t) => t.line.at(shape.tw as isize),
        };
        out.set(1 + shape.tw as isize, value);
    }

    if needs.topleft {
        corner(&mut out, &f, &shape, pic, top, corner_filter);
    }
    mode
}

/// Sizes of the edge runs.
struct Shape {
    /// Transform width and height in samples.
    tw: i32,
    th: i32,
    /// The reference line, for the zone predictors only.
    diag_mrl: usize,
    /// Offset of line 0's corner from line `mrl_idx`'s.
    e_stride: isize,
    /// Is the mode DIP's (its extensions are a quarter of the side).
    dip: bool,
    is_dir: bool,
}

impl Shape {
    fn new(req: &EdgeRequest, f: &Flags, mode: ImplMode, is_dir: bool) -> Self {
        let tw = req.tw4 << UNIT_LOG2;
        let th = req.th4 << UNIT_LOG2;
        let diag_mrl = if is_zone(mode) { f.mrl } else { 0 };
        let e_stride = ((tw + th) * 2) as isize + diag_mrl as isize * 3 + 1;
        Self {
            tw,
            th,
            diag_mrl,
            e_stride,
            dip: f.dip,
            is_dir,
        }
    }

    /// The length of a side `side` long with its extension `ext` (the
    /// other side for directional modes), and the line-0 run's length.
    fn run(&self, side: i32, other: i32, needs_side: bool, needs_ext: bool) -> (i32, i32) {
        let mut sz = if needs_side { side } else { 1 };
        let mut sz2 = side;
        if needs_ext {
            sz += if self.dip {
                side >> 2
            } else if self.is_dir {
                other + 2 * self.diag_mrl as i32
            } else {
                1
            };
            sz2 = sz - 2 * self.diag_mrl as i32;
        }
        (sz, sz2)
    }
}

/// The left column and its bottom-left extension, of line `mrl_idx` and,
/// with multiple reference lines, line 0.
fn left_edge<T: Sample>(
    out: &mut EdgeOut<'_, T>,
    req: &EdgeRequest,
    f: &Flags,
    shape: &Shape,
    needs: EdgeNeeds,
    pic: SampleRef<'_, T>,
    stride: isize,
    top: Option<TopRows<'_, T>>,
) {
    let th = shape.th;
    let (sz, sz2) = shape.run(th, shape.tw, needs.left, needs.bottomleft);
    let left = -(shape.diag_mrl as isize + 1);
    let left2 = shape.e_stride - 1;
    let column = |col: isize, i: i32| pic.at(stride * i as isize + col);

    if !f.have_left {
        let no_left = out.missing.left;
        out.fill(
            left + 1 - sz as isize,
            sz as usize,
            top.map_or(no_left, |t| t.line.at(0)),
        );
        if f.multi_mrl {
            let v = top.map_or(no_left, |t| t.line0.at(0));
            out.fill(left2 + 1 - sz2 as isize, sz2 as usize, v);
        }
        return;
    }
    let col = -1 - f.mrl as isize;
    let mut have = if needs.left {
        th.min((req.end_y - req.y) << UNIT_LOG2)
    } else {
        1
    };
    if needs.bottomleft && req.n_bl > 0 {
        have += (req.n_bl << UNIT_LOG2).min(sz - th);
    }
    for i in 0..have {
        out.set(left - i as isize, column(col, i));
    }
    if have < sz {
        let last = out.get(left + 1 - have as isize);
        out.fill(left + 1 - sz as isize, (sz - have) as usize, last);
    }
    if f.multi_mrl {
        let have = have.min(sz2);
        for i in 0..have {
            out.set(left2 - i as isize, column(-1, i));
        }
        if have < sz2 {
            let last = out.get(left2 + 1 - have as isize);
            out.fill(left2 + 1 - sz2 as isize, (sz2 - have) as usize, last);
        }
    }
}

/// The top row and its top-right extension, of line `mrl_idx` and, with
/// multiple reference lines, line 0.
fn top_edge<T: Sample>(
    out: &mut EdgeOut<'_, T>,
    req: &EdgeRequest,
    f: &Flags,
    shape: &Shape,
    needs: EdgeNeeds,
    pic: SampleRef<'_, T>,
    top: Option<TopRows<'_, T>>,
) {
    let tw = shape.tw;
    let (sz, sz2) = shape.run(tw, shape.th, needs.top, needs.topright);
    let start = shape.diag_mrl as isize + 1;
    let start2 = shape.e_stride + 1;

    let Some(rows) = top else {
        let no_top = out.missing.top;
        let v = if f.have_left {
            pic.at(-1 - f.mrl as isize)
        } else {
            no_top
        };
        out.fill(start, sz as usize, v);
        if f.multi_mrl {
            out.fill(
                start2,
                sz2 as usize,
                if f.have_left { pic.at(-1) } else { no_top },
            );
        }
        return;
    };
    let mut have = if needs.top {
        tw.min((req.end_x - req.x) << UNIT_LOG2)
    } else {
        1
    };
    out.copy(start, rows.line.slice(0, have as usize));
    if needs.topright && req.n_tr > 0 {
        // The top-right run continues from the block's right edge.
        have += (req.n_tr << UNIT_LOG2).min(sz - tw);
        out.copy(
            start + tw as isize,
            rows.line.slice(tw as isize, (have - tw) as usize),
        );
    }
    if have < sz {
        let last = out.get(start + have as isize - 1);
        out.fill(start + have as isize, (sz - have) as usize, last);
    }
    if f.multi_mrl {
        let have = have.min(sz2);
        out.copy(start2, rows.line0.slice(0, have as usize));
        if have < sz2 {
            let last = out.get(start2 + have as isize - 1);
            out.fill(start2 + have as isize, (sz2 - have) as usize, last);
        }
    }
}

/// The corner samples of line `mrl_idx`, line 0's corner, and the corner
/// filter.
fn corner<T: Sample>(
    out: &mut EdgeOut<'_, T>,
    f: &Flags,
    shape: &Shape,
    pic: SampleRef<'_, T>,
    top: Option<TopRows<'_, T>>,
    filter: bool,
) {
    let no_corner = out.missing.corner;
    debug_assert_eq!(shape.diag_mrl, f.mrl);
    let m = f.mrl as isize;
    match top {
        Some(rows) if f.have_left => {
            for i in -m..0 {
                out.set(i, rows.line.at(-(m + 1) - i * rows.stride));
            }
            for i in 0..=m {
                out.set(i, rows.line.at(-(m + 1 - i)));
            }
        }
        _ => {
            let v = if f.have_left {
                pic.at(-(1 + m))
            } else {
                top.map_or(no_corner, |t| t.line.at(0))
            };
            out.fill(-m, 2 * f.mrl + 1, v);
        }
    }
    let corner0 = match (f.have_left, top) {
        (true, Some(rows)) => rows.line0.at(-1),
        (true, None) => pic.at(-1),
        (false, Some(rows)) => rows.line0.at(0),
        (false, None) => no_corner,
    };
    out.set(shape.e_stride, corner0);

    if filter {
        let c = out.get(0).to_i32();
        let sum = c + (out.get(-1).to_i32() + c + out.get(1).to_i32()) * 5;
        out.set(
            0,
            T::from_i32((sum + (1 << (CORNER_FILTER_BITS - 1))) >> CORNER_FILTER_BITS),
        );
    }
}
