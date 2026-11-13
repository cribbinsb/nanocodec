//! HEVC intra sample prediction (§8.4.4.2): reference sample gathering and
//! substitution (§8.4.4.2.2), reference filtering (§8.4.4.2.3), and the
//! planar (§8.4.4.2.5), DC (§8.4.4.2.6) and angular (§8.4.4.2.6) modes.
//!
//! References are kept in one linear array `p` of `4n + 1` samples: the left
//! column from the bottom (`p[-1][2n-1]`) up to the corner (`p[-1][-1]`),
//! then the row above from left to right (`p[0][-1]` .. `p[2n-1][-1]`).
//! [`Refs::left`] and [`Refs::top`] index it the way the specification does.
//!
//! Main profile, 4:2:0: filtering and the DC / pure horizontal and vertical
//! edge filters apply to luma only. The encoder never sets
//! `strong_intra_smoothing_enabled_flag`; the decoder's
//! [`predict_smoothed_into`] honours it.

use nc_common::PlaneViewMut;
use nc_dsp::{Dsp, HevcPlanarEdges};

/// Reference value when no neighbour is available: `1 << (BitDepth − 1)`.
const NO_NEIGHBOURS: u8 = 128;
/// Largest intra transform block side.
const MAX_SIZE: usize = crate::transform::MAX_SIZE;
/// Reference samples of the largest block.
const MAX_REFS: usize = 4 * MAX_SIZE + 1;
/// Strong smoothing applies to luma blocks of this side only (§8.4.4.2.3).
const STRONG_SMOOTHING_SIZE: usize = 32;
/// `1 << (BitDepthY - 5)`: the largest deviation from a straight line of
/// either reference edge that still allows strong smoothing (8-bit).
const STRONG_SMOOTHING_THRESHOLD: i32 = 1 << 3;
/// DC and pure horizontal / vertical edge filters apply below this size.
const EDGE_FILTER_MAX_SIZE: usize = 32;

/// `INTRA_PLANAR`.
pub const PLANAR: u8 = 0;
/// `INTRA_DC`.
pub const DC: u8 = 1;
/// `INTRA_ANGULAR10` (pure horizontal).
pub const HORIZONTAL: u8 = 10;
/// `INTRA_ANGULAR26` (pure vertical).
pub const VERTICAL: u8 = 26;
/// Number of luma intra prediction modes.
pub const MODES: u8 = 35;
/// `INTRA_ANGULAR34`, the chroma mode replacing an explicit chroma mode
/// equal to the luma mode.
const ANGULAR34: u8 = 34;
/// `intra_chroma_pred_mode` 4: chroma uses the luma mode (DM).
pub const CHROMA_DM: u8 = 4;
/// The chroma modes of `intra_chroma_pred_mode` 0..=3.
const EXPLICIT_CHROMA_MODES: [u8; 4] = [PLANAR, VERTICAL, HORIZONTAL, DC];

/// `IntraPredModeC` for `intra_chroma_pred_mode` (0..=4) and the luma mode
/// (§8.4.3, Table 8-2; 4:2:0 needs no further mapping).
#[must_use]
pub fn chroma_mode(intra_chroma_pred_mode: u8, luma: u8) -> u8 {
    match EXPLICIT_CHROMA_MODES.get(usize::from(intra_chroma_pred_mode)) {
        Some(&mode) if mode == luma => ANGULAR34,
        Some(&mode) => mode,
        None => luma,
    }
}

/// `intraPredAngle` of modes 2..=34 (Table 8-4).
const ANGLE: [i32; 33] = [
    32, 26, 21, 17, 13, 9, 5, 2, 0, -2, -5, -9, -13, -17, -21, -26, -32, -26, -21, -17, -13, -9,
    -5, -2, 0, 2, 5, 9, 13, 17, 21, 26, 32,
];
/// `invAngle` of modes 11..=25 (Table 8-5).
const INV_ANGLE: [i32; 15] = [
    -4096, -1638, -910, -630, -482, -390, -315, -256, -315, -390, -482, -630, -910, -1638, -4096,
];

/// The reference samples of one transform block.
#[derive(Debug, Clone)]
pub struct Refs {
    n: usize,
    samples: [u8; MAX_REFS],
}

impl Default for Refs {
    /// Storage for [`Refs::gather_into`].
    fn default() -> Self {
        Self {
            n: 0,
            samples: [0; MAX_REFS],
        }
    }
}

/// How availability varies along the sides of a block whose references
/// [`Refs::gather`] collects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sides {
    /// Ask for every run.
    PerUnit,
    /// The neighbours beside the block's own rows (left) and columns
    /// (above) precede it in decoding order and lie in one CTB per side,
    /// so one slice and tile: one answer covers each side. Only the
    /// below-left and above-right extensions vary. Not with constrained
    /// intra prediction, where each neighbouring block's mode counts.
    Uniform,
}

impl Refs {
    /// Gather and substitute the references of the `n`x`n` block at `(x, y)`
    /// of `plane` (plane coordinates). Availability (§6.4.1) is uniform over
    /// `unit`-sample runs aligned to the block (one minimum 4x4 luma
    /// transform block); `available(xn, yn)` is asked once per run, only
    /// for positions inside the picture. With [`Sides::Uniform`] it is
    /// asked once for the runs beside the block's own rows (left) and
    /// once for those above its own columns.
    pub fn gather(
        plane: &nc_common::Plane,
        (x, y): (usize, usize),
        n: usize,
        unit: usize,
        sides: Sides,
        available: impl Fn(usize, usize) -> bool,
    ) -> Self {
        let mut refs = Self::default();
        refs.gather_into(plane, (x, y), n, unit, sides, available);
        refs
    }

    /// [`Self::gather`] into `self`, reusing its storage: no sample is
    /// cleared first, as every one of the `4n + 1` is written (gathered or
    /// substituted).
    pub fn gather_into(
        &mut self,
        plane: &nc_common::Plane,
        (x, y): (usize, usize),
        n: usize,
        unit: usize,
        sides: Sides,
        available: impl Fn(usize, usize) -> bool,
    ) {
        let view = plane.view();
        let (width, height) = (plane.width(), plane.height());
        let (data, stride) = (view.data(), view.stride());
        self.n = n;
        let samples = &mut self.samples;
        // Availability by run: the left column's units bottom up, the
        // corner, then the top row's units left to right (at most 16 per
        // side for a 32x32 block of 4-sample units).
        let side_units = 2 * n / unit;
        let mut known = 0u64;
        let corner = 2 * n;
        if x > 0 {
            // Left column, bottom up: samples[i] is row y + 2n − 1 − i.
            // The block's own rows are samples n..2n.
            let own_left = (sides == Sides::Uniform).then(|| available(x - 1, y));
            for (u, start) in (0..2 * n).step_by(unit).enumerate() {
                let top_row = y + 2 * n - start - unit;
                let usable = match own_left {
                    Some(own) if start >= n => own,
                    _ => top_row < height && available(x - 1, top_row),
                };
                if !usable {
                    continue;
                }
                for i in start..start + unit {
                    samples[i] = data[(y + 2 * n - 1 - i) * stride + x - 1];
                }
                known |= 1 << u;
            }
            if y > 0 && available(x - 1, y - 1) {
                samples[corner] = data[(y - 1) * stride + x - 1];
                known |= 1 << side_units;
            }
        }
        if y > 0 {
            let row = &data[(y - 1) * stride..];
            let own_above = (sides == Sides::Uniform).then(|| available(x, y - 1));
            let usable = |start: usize| match own_above {
                Some(own) if start < n => own,
                _ => x + start < width && available(x + start, y - 1),
            };
            let mut start = 0;
            while start < 2 * n {
                if !usable(start) {
                    start += unit;
                    continue;
                }
                // Consecutive available units are copied as one run.
                let mut end = start + unit;
                while end < 2 * n && usable(end) {
                    end += unit;
                }
                let first = corner + 1 + start;
                samples[first..first + end - start].copy_from_slice(&row[x + start..x + end]);
                for u in start / unit..end / unit {
                    known |= 1 << (side_units + 1 + u);
                }
                start = end;
            }
        }
        let units = 2 * side_units + 1;
        if known == (1 << units) - 1 {
            return;
        }
        let count = 4 * n + 1;
        // The samples of run `u`.
        let run = |u: usize| match u.cmp(&side_units) {
            std::cmp::Ordering::Less => u * unit..(u + 1) * unit,
            std::cmp::Ordering::Equal => corner..corner + 1,
            std::cmp::Ordering::Greater => {
                let start = corner + 1 + (u - side_units - 1) * unit;
                start..start + unit
            }
        };
        if known == 0 {
            samples[..count].fill(NO_NEIGHBOURS);
            return;
        }
        // §8.4.4.2.2: the bottom-most reference takes the first available
        // one, then each missing sample copies the one before it in this
        // order.
        samples[0] = samples[run(known.trailing_zeros() as usize).start];
        for u in (0..units).filter(|&u| known & (1 << u) == 0) {
            for i in run(u).filter(|&i| i > 0) {
                samples[i] = samples[i - 1];
            }
        }
    }

    /// `p[-1][y]` for `y` in `-1..2n`.
    fn left(&self, y: isize) -> i32 {
        i32::from(self.samples[(2 * self.n as isize - 1 - y) as usize])
    }

    /// `p[x][-1]` for `x` in `-1..2n`.
    fn top(&self, x: isize) -> i32 {
        i32::from(self.samples[(2 * self.n as isize + 1 + x) as usize])
    }

    /// The [1 2 1] filtered references (§8.4.4.2.3, end samples unchanged).
    fn filtered(&self) -> Self {
        let count = 4 * self.n + 1;
        let mut samples = self.samples;
        for (dst, w) in samples[1..count - 1]
            .iter_mut()
            .zip(self.samples[..count].windows(3))
        {
            let [a, b, c] = [w[0], w[1], w[2]].map(u32::from);
            *dst = ((a + 2 * b + c + 2) >> 2) as u8;
        }
        Self { n: self.n, samples }
    }
}

impl Refs {
    /// The bilinear references of `strong_intra_smoothing_enabled_flag`
    /// (§8.4.4.2.3, equations 8-50 to 8-54) when both edges are close to
    /// straight lines, `None` otherwise.
    fn strongly_smoothed(&self) -> Option<Self> {
        let n = self.n;
        if n != STRONG_SMOOTHING_SIZE {
            return None;
        }
        let corner = self.top(-1);
        let (top_end, left_end) = (self.top(2 * n as isize - 1), self.left(2 * n as isize - 1));
        let straight =
            |end: i32, middle: i32| (corner + end - 2 * middle).abs() < STRONG_SMOOTHING_THRESHOLD;
        if !straight(top_end, self.top(n as isize - 1))
            || !straight(left_end, self.left(n as isize - 1))
        {
            return None;
        }
        let mut samples = self.samples;
        let span = 2 * n as i32;
        for i in 0..2 * n - 1 {
            let k = i as i32 + 1;
            // Left column bottom-up and top row left to right, both
            // interpolated between the corner and their far end.
            let left = ((span - k) * corner + k * left_end + n as i32) >> 6;
            let top = ((span - k) * corner + k * top_end + n as i32) >> 6;
            samples[2 * n - 1 - i] = left as u8;
            samples[2 * n + 1 + i] = top as u8;
        }
        Some(Self { n, samples })
    }
}

/// `filterFlag` of §8.4.4.2.3 for a luma block of side `n`.
fn needs_filter(mode: u8, n: usize) -> bool {
    if mode == DC || n == 4 {
        return false;
    }
    let min_dist = (i32::from(mode) - i32::from(VERTICAL))
        .abs()
        .min((i32::from(mode) - i32::from(HORIZONTAL)).abs());
    let threshold = match n {
        8 => 7,
        16 => 1,
        _ => 0,
    };
    min_dist > threshold
}

/// Predict the `n`x`n` block from `refs` with `mode` into `out` (row-major).
/// `luma` enables filtering and the luma edge filters.
pub fn predict(dsp: &Dsp, refs: &Refs, mode: u8, luma: bool, out: &mut [u8]) {
    if predict_oriented(dsp, refs, mode, luma, out) {
        transpose_in_place(&mut out[..refs.n * refs.n], refs.n);
    }
}

/// Like [`predict`], but horizontal angular modes (2..=17) are produced
/// transposed (column `j` of the block as row `j` of `out`), which is how
/// they are computed. Returns whether `out` is transposed. The SATD of a
/// transposed difference equals the SATD of the difference, so a mode search
/// can compare against a transposed source instead of transposing every
/// candidate.
pub fn predict_oriented(dsp: &Dsp, refs: &Refs, mode: u8, luma: bool, out: &mut [u8]) -> bool {
    oriented(dsp, refs, mode, luma, false, out)
}

/// [`predict`] with `strong_intra_smoothing_enabled_flag` = `strong`, straight
/// into the picture: `scratch` (at least `n²` samples) takes the kernel's
/// output, which `dst` (the `n`x`n` block) receives row by row, transposed
/// for the horizontal modes in the same pass.
pub fn predict_smoothed_into(
    dsp: &Dsp,
    refs: &Refs,
    (mode, luma, strong): (u8, bool, bool),
    scratch: &mut [u8],
    dst: &mut PlaneViewMut<'_>,
) {
    let n = refs.n;
    let block = &mut scratch[..n * n];
    if oriented(dsp, refs, mode, luma, strong, block) {
        for row in 0..n {
            for (col, d) in dst.row_mut(row)[..n].iter_mut().enumerate() {
                *d = block[col * n + row];
            }
        }
    } else {
        for (row, line) in block.chunks_exact(n).enumerate() {
            copy_row(line, &mut dst.row_mut(row)[..n]);
        }
    }
}

/// Copy one block row: by its size, so that the copy compiles to vector
/// moves rather than a call (or WASM `memory.copy`).
fn copy_row(src: &[u8], dst: &mut [u8]) {
    fn fixed<const N: usize>(src: &[u8], dst: &mut [u8]) {
        dst[..N].copy_from_slice(&src[..N]);
    }
    match src.len() {
        4 => fixed::<4>(src, dst),
        8 => fixed::<8>(src, dst),
        16 => fixed::<16>(src, dst),
        32 => fixed::<32>(src, dst),
        _ => dst.copy_from_slice(src),
    }
}

/// [`predict_oriented`] with optional strong smoothing.
fn oriented(dsp: &Dsp, refs: &Refs, mode: u8, luma: bool, strong: bool, out: &mut [u8]) -> bool {
    let n = refs.n;
    let filtered;
    let p = if luma && needs_filter(mode, n) {
        filtered = strong
            .then(|| refs.strongly_smoothed())
            .flatten()
            .unwrap_or_else(|| refs.filtered());
        &filtered
    } else {
        refs
    };
    let out = &mut out[..n * n];
    match mode {
        PLANAR => {
            planar(dsp, p, n, out);
            false
        }
        DC => {
            dc(p, n, luma, out);
            false
        }
        _ => angular(dsp, p, mode, n, luma, out),
    }
}

/// Transpose the `n`x`n` block `block` in place.
fn transpose_in_place(block: &mut [u8], n: usize) {
    for row in 0..n {
        for col in row + 1..n {
            block.swap(row * n + col, col * n + row);
        }
    }
}

/// §8.4.4.2.5, by the planar kernel.
fn planar(dsp: &Dsp, p: &Refs, n: usize, out: &mut [u8]) {
    let mut above = [0u8; MAX_SIZE];
    let mut left = [0u8; MAX_SIZE];
    for i in 0..n {
        above[i] = p.top(i as isize) as u8;
        left[i] = p.left(i as isize) as u8;
    }
    let edges = HevcPlanarEdges {
        above: &above[..n],
        left: &left[..n],
        top_right: p.top(n as isize) as u8,
        bottom_left: p.left(n as isize) as u8,
    };
    dsp.hevc_intra_planar(&edges, n, out);
}

/// §8.4.4.2.6 (DC), with the luma edge filter for blocks below 32x32.
fn dc(p: &Refs, n: usize, luma: bool, out: &mut [u8]) {
    let sum: i32 = (0..n as isize).map(|i| p.top(i) + p.left(i)).sum();
    let log2 = n.trailing_zeros();
    let dc = (sum + n as i32) >> (log2 + 1);
    out.fill(dc as u8);
    if luma && n < EDGE_FILTER_MAX_SIZE {
        out[0] = ((p.left(0) + 2 * dc + p.top(0) + 2) >> 2) as u8;
        for i in 1..n {
            out[i] = ((p.top(i as isize) + 3 * dc + 2) >> 2) as u8;
            out[i * n] = ((p.left(i as isize) + 3 * dc + 2) >> 2) as u8;
        }
    }
}

/// §8.4.4.2.6 (angular modes 2..=34), in line order: returns `true` (the
/// block is transposed) for horizontal modes.
fn angular(dsp: &Dsp, p: &Refs, mode: u8, n: usize, luma: bool, out: &mut [u8]) -> bool {
    let angle = ANGLE[usize::from(mode) - 2];
    let vertical = mode >= 18;
    // The main reference ("ref[]" of the specification), indexed from -n.
    // Vertical modes project along the row above, horizontal ones along the
    // left column; a negative angle extends it with the other side.
    let main = |i: isize| if vertical { p.top(i) } else { p.left(i) };
    let side = |i: isize| if vertical { p.left(i) } else { p.top(i) };
    let mut buf = [0u8; 3 * MAX_SIZE + 1];
    let origin = n; // buf[origin + i] is ref[i]
    for i in 0..=n {
        buf[origin + i] = main(i as isize - 1) as u8;
    }
    let reach = (n as i32 * angle) >> 5;
    if angle < 0 {
        if reach < -1 {
            let inv = INV_ANGLE[usize::from(mode) - 11];
            for i in reach..=-1 {
                buf[(origin as i32 + i) as usize] =
                    side((((i * inv + 128) >> 8) - 1) as isize) as u8;
            }
        }
    } else {
        for i in n + 1..=2 * n {
            buf[origin + i] = main(i as isize - 1) as u8;
        }
    }
    // Project one line at a time: rows for vertical modes, columns for
    // horizontal ones (written as rows, i.e. transposed).
    dsp.hevc_intra_angular(&buf[..=3 * n], n, angle, out);
    // Pure vertical / horizontal luma: smooth the first column (vertical) or
    // row (horizontal) towards the side references; in line order both are
    // the first sample of every line.
    if luma && n < EDGE_FILTER_MAX_SIZE && angle == 0 {
        let corner = p.top(-1);
        for (k, line) in out.chunks_exact_mut(n).enumerate() {
            let sample = main(0) + ((side(k as isize) - corner) >> 1);
            line[0] = sample.clamp(0, 255) as u8;
        }
    }
    !vertical
}

/// The three most probable modes (§8.4.2) given the left and above
/// neighbours' luma modes (`None`: unavailable, inter, or above outside
/// the current CTB row; all count as DC).
#[must_use]
pub fn most_probable(left: Option<u8>, above: Option<u8>) -> [u8; 3] {
    let a = left.unwrap_or(DC);
    let b = above.unwrap_or(DC);
    if a == b {
        if a < 2 {
            [PLANAR, DC, VERTICAL]
        } else {
            [a, 2 + ((a + 29) % 32), 2 + ((a - 2 + 1) % 32)]
        }
    } else {
        let third = if a != PLANAR && b != PLANAR {
            PLANAR
        } else if a != DC && b != DC {
            DC
        } else {
            VERTICAL
        };
        [a, b, third]
    }
}

/// How a luma mode is signalled against the candidate list.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModeCode {
    /// `prev_intra_luma_pred_flag` 1 with `mpm_idx`.
    Mpm(u8),
    /// `rem_intra_luma_pred_mode`.
    Rem(u8),
}

impl ModeCode {
    /// The code of `mode` given the candidates `mpm`.
    #[must_use]
    pub fn of(mode: u8, mpm: [u8; 3]) -> Self {
        if let Some(i) = mpm.iter().position(|&m| m == mode) {
            return Self::Mpm(i as u8);
        }
        let below = mpm.iter().filter(|&&m| m < mode).count() as u8;
        Self::Rem(mode - below)
    }

    /// Bins of the code (flag + bypass bins); the flag counted as one bit.
    #[must_use]
    pub fn bits(self) -> f32 {
        match self {
            Self::Mpm(0) => 2.0,
            Self::Mpm(_) => 3.0,
            Self::Rem(_) => 6.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chroma_modes_follow_table_8_2() {
        // intra_chroma_pred_mode 0..=3: planar, vertical, horizontal, DC;
        // 34 where that equals the luma mode; 4 takes the luma mode.
        assert_eq!(chroma_mode(0, 7), PLANAR);
        assert_eq!(chroma_mode(1, 7), VERTICAL);
        assert_eq!(chroma_mode(2, 7), HORIZONTAL);
        assert_eq!(chroma_mode(3, 7), DC);
        assert_eq!(chroma_mode(1, VERTICAL), ANGULAR34);
        assert_eq!(chroma_mode(3, DC), ANGULAR34);
        assert_eq!(chroma_mode(CHROMA_DM, 7), 7);
    }
    use nc_common::Plane;

    fn ramp_plane() -> Plane {
        let mut plane = Plane::new(64, 64, 0).expect("plane");
        let mut view = plane.view_mut();
        for y in 0..64 {
            for (x, v) in view.row_mut(y).iter_mut().enumerate() {
                *v = ((x * 3 + y * 2) & 0xff) as u8;
            }
        }
        plane
    }

    #[test]
    fn no_neighbours_predicts_mid_grey() {
        let plane = ramp_plane();
        let refs = Refs::gather(&plane, (0, 0), 8, 4, Sides::PerUnit, |_, _| true);
        let mut out = [0u8; 64];
        for mode in 0..MODES {
            predict(&Dsp::new(), &refs, mode, true, &mut out);
            assert!(out.iter().all(|&v| v == NO_NEIGHBOURS), "mode {mode}");
        }
    }

    #[test]
    fn vertical_copies_the_row_above_and_horizontal_the_left_column() {
        let plane = ramp_plane();
        let refs = Refs::gather(&plane, (16, 16), 8, 4, Sides::PerUnit, |_, _| true);
        let view = plane.view();
        let mut out = [0u8; 64];
        // Chroma rules (no edge filter) make the copies exact.
        predict(&Dsp::new(), &refs, VERTICAL, false, &mut out);
        for y in 0..8 {
            assert_eq!(&out[y * 8..y * 8 + 8], &view.row(15)[16..24]);
        }
        predict(&Dsp::new(), &refs, HORIZONTAL, false, &mut out);
        for y in 0..8 {
            assert!(
                out[y * 8..y * 8 + 8]
                    .iter()
                    .all(|&v| v == view.row(16 + y)[15])
            );
        }
    }

    #[test]
    fn every_mode_and_size_stays_inside_its_references() {
        let plane = ramp_plane();
        let mut out = [0u8; MAX_SIZE * MAX_SIZE];
        for n in [4, 8, 16, 32] {
            let refs = Refs::gather(&plane, (16, 16), n, 4, Sides::PerUnit, |_, _| true);
            for mode in 0..MODES {
                predict(&Dsp::new(), &refs, mode, true, &mut out);
                predict(&Dsp::new(), &refs, mode, false, &mut out);
            }
        }
    }

    #[test]
    fn mode_codes_round_trip() {
        let mpm = most_probable(Some(10), Some(26));
        assert_eq!(mpm, [10, 26, PLANAR]);
        let mut rems: Vec<u8> = (0..MODES)
            .filter_map(|m| match ModeCode::of(m, mpm) {
                ModeCode::Rem(r) => Some(r),
                ModeCode::Mpm(_) => None,
            })
            .collect();
        rems.dedup();
        assert_eq!(rems, (0..32).collect::<Vec<_>>());
    }

    #[test]
    fn angular_neighbour_candidates_wrap() {
        assert_eq!(most_probable(Some(2), Some(2)), [2, 33, 3]);
        assert_eq!(most_probable(Some(34), None.or(Some(34))), [34, 33, 3]);
        assert_eq!(most_probable(None, None), [PLANAR, DC, VERTICAL]);
    }
}
