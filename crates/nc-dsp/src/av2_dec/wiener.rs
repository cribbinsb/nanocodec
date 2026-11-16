//! The AV2 loop restoration filters (dav2d `looprestoration_tmpl.c`;
//! AVM v1.0.0 `restoration.c`): the non-separable Wiener filter of luma
//! (one filter, or one per class) and of chroma (with a cross-component
//! part on the co-located luma), and the pixel-classified Wiener filter
//! (PC-Wiener: a pretrained filter per class).
//!
//! The kernels read a window of the plane with [`WINDOW_BORDER`] samples of
//! context on every side, filled by the caller (dav2d's stripe and edge
//! rules live in `nc_av2::general::filters::restoration`). Classes are
//! chosen per 4x4 block from second-derivative features of the 6x6
//! samples around it and how many of its neighbours code no residual
//! ([`ClassModel`]).

use nc_common::{BitDepth, Sample};

/// Rows and columns of context a window holds on each side of its block.
pub const WINDOW_BORDER: usize = 4;
/// Taps of a luma filter (symmetric pairs).
pub const LUMA_TAPS: usize = 16;
/// Taps of a chroma filter: six chroma pairs, then twelve luma neighbours.
pub const CHROMA_TAPS: usize = 18;
/// Taps of a pretrained filter: twelve symmetric pairs and the centre.
pub const PRETRAINED_TAPS: usize = 13;
/// Chroma taps on the chroma plane itself.
const CHROMA_OWN_TAPS: usize = 6;
/// Fixed-point bits of the filters' taps.
const TAP_BITS: u32 = 7;
/// Luma pairs' offsets (dy, dx) (dav2d `wiener_ns_config_y`).
const LUMA_PAIRS: [(isize, isize); LUMA_TAPS] = [
    (1, 0),
    (0, 1),
    (2, 0),
    (0, 2),
    (1, 1),
    (-1, 1),
    (2, 1),
    (2, -1),
    (1, 2),
    (1, -2),
    (3, 0),
    (0, 3),
    (4, 0),
    (0, 4),
    (3, 3),
    (3, -3),
];
/// Chroma pairs' offsets (dav2d `wiener_ns_config_uv`).
const CHROMA_PAIRS: [(isize, isize); CHROMA_OWN_TAPS] =
    [(1, 0), (0, 1), (1, 1), (-1, 1), (2, 0), (0, 2)];
/// The cross-component taps' luma neighbours (dav2d
/// `wiener_ns_config_uv_from_y`).
const LUMA_NEIGHBOURS: [(isize, isize); CHROMA_TAPS - CHROMA_OWN_TAPS] = [
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
    (1, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (2, 0),
    (-2, 0),
    (0, 2),
    (0, -2),
];
/// Pretrained pairs' offsets (dav2d `pc_wiener_config`).
const PRETRAINED_PAIRS: [(isize, isize); PRETRAINED_TAPS - 1] = [
    (1, 0),
    (0, 1),
    (2, 0),
    (0, 2),
    (1, 1),
    (-1, 1),
    (2, 1),
    (2, -1),
    (1, 2),
    (1, -2),
    (3, 0),
    (0, 3),
];

/// A window of a plane around a block: `samples[(y + WINDOW_BORDER) *
/// stride + x + WINDOW_BORDER]` is the block's sample (x, y).
#[derive(Debug, Clone, Copy)]
pub struct Window<'a, T> {
    /// The samples.
    pub samples: &'a [T],
    /// Row stride.
    pub stride: usize,
}

impl<T: Sample> Window<'_, T> {
    /// The sample at (`x`, `y`) of the block (up to [`WINDOW_BORDER`]
    /// outside).
    #[inline]
    #[must_use]
    pub fn at(&self, x: isize, y: isize) -> i32 {
        let border = WINDOW_BORDER as isize;
        let index = (y + border) as usize * self.stride + (x + border) as usize;
        self.samples[index].to_i32()
    }
}

/// A luma filter.
#[derive(Debug, Clone, Copy)]
pub enum LumaFilter<'a> {
    /// A non-separable Wiener filter (the frame's or the unit's).
    NonSeparable(&'a [i8]),
    /// A pretrained PC-Wiener filter.
    Pretrained(&'a [i8; PRETRAINED_TAPS]),
}

/// The rounded, clipped output of a filter sum in tap units.
fn output<T: Sample>(sum: i32, depth: BitDepth) -> T {
    depth.clip_to::<T>((sum + (1 << (TAP_BITS - 1))) >> TAP_BITS)
}

/// Filter the 4x4 block at (`x`, `y`) of `window` into `out` (row stride
/// `stride`, the block's first sample at index 0; dav2d
/// `ns_wiener_single_y_c`, `wiener_multi`'s filtering).
pub fn luma_block<T: Sample>(
    window: &Window<'_, T>,
    filter: LumaFilter<'_>,
    (x, y): (usize, usize),
    out: &mut [T],
    stride: usize,
    depth: BitDepth,
) {
    for dy in 0..4 {
        for dx in 0..4 {
            let (px, py) = ((x + dx) as isize, (y + dy) as isize);
            let m = window.at(px, py);
            let pair = |(oy, ox): (isize, isize)| {
                window.at(px + ox, py + oy) + window.at(px - ox, py - oy)
            };
            let sum = match filter {
                LumaFilter::NonSeparable(taps) => {
                    let mut sum = m << TAP_BITS;
                    for (&offset, &tap) in LUMA_PAIRS.iter().zip(taps) {
                        sum += (pair(offset) - 2 * m) * i32::from(tap);
                    }
                    sum
                }
                LumaFilter::Pretrained(taps) => {
                    let mut sum = 2 * m * i32::from(taps[PRETRAINED_TAPS - 1]);
                    for (&offset, &tap) in PRETRAINED_PAIRS.iter().zip(taps) {
                        sum += i32::from(tap) * pair(offset);
                    }
                    sum
                }
            };
            out[dy * stride + dx] = output(sum, depth);
        }
    }
}

/// Filter the chroma sample at (`x`, `y`) of `window` with `taps`, its
/// cross-component part read from `luma` (the co-located luma, downsampled
/// to chroma resolution and as wide a window; dav2d `wiener_uv_luma_c`
/// and `wiener_uv_chroma_c`).
#[must_use]
pub fn chroma_sample<T: Sample>(
    window: &Window<'_, T>,
    luma: &Window<'_, T>,
    taps: &[i8; CHROMA_TAPS],
    (x, y): (usize, usize),
    depth: BitDepth,
) -> T {
    let (px, py) = (x as isize, y as isize);
    let m = window.at(px, py);
    let mut sum = m << TAP_BITS;
    for (&(oy, ox), &tap) in CHROMA_PAIRS.iter().zip(taps) {
        let pair = window.at(px + ox, py + oy) + window.at(px - ox, py - oy);
        sum += (pair - 2 * m) * i32::from(tap);
    }
    let centre = luma.at(px, py);
    for (&(oy, ox), &tap) in LUMA_NEIGHBOURS.iter().zip(&taps[CHROMA_OWN_TAPS..]) {
        sum += (luma.at(px + ox, py + oy) - centre) * i32::from(tap);
    }
    output(sum, depth)
}

/// Feature weights and offsets of the class model (dav2d
/// `setup_wiener_qval_approx_data`'s `mode_weights`, `mode_offsets`): the
/// first row decides the skip-only index, the others each feature's.
const MODE_WEIGHTS: [[i32; 3]; 4] = [
    [-527, 15325, 321],
    [26436, -17705, 17905],
    [366, -147, -194],
    [202, -267, -179],
];
const MODE_OFFSETS: [i32; 4] = [-547, -21565, -573, -680];
/// Feature normalisers (dav2d `pc_wiener_normalizer`).
const NORMALIZERS: [u32; 3] = [3739, 3273, 3074];
/// Largest skip count of a block (a 3x3 neighbourhood, centre weighted 4,
/// each of its own 4x4 block's neighbours by row up to 6).
pub const MAX_SKIP_COUNT: usize = 36;
/// Bits of each feature's index in the 9-bit feature index, and the skip
/// flag above them.
const FEATURE_INDEX_BITS: u32 = 3;
const SKIP_INDEX_SHIFT: u32 = 9;

/// The class model of a frame's quantizer step (dav2d
/// `WienerQValApproxData`): each feature's threshold as a function of the
/// block's skip count, a slope plus a correction table.
#[derive(Debug, Clone)]
pub struct ClassModel {
    corrections: [[i32; MAX_SKIP_COUNT + 1]; 3],
    offsets: [i32; 3],
    slopes: [i32; 3],
    /// Skip counts below this set the skip bit of the index (`None`: it
    /// never changes).
    changeover: Option<usize>,
    /// The index offset of the skip bit's low state (`idx0_low << 9`).
    offset: usize,
}

/// `apply_sign((|v| + 4096) >> 13, v)`.
fn round13(v: i32) -> i32 {
    ((v.abs() + (1 << 12)) >> 13) * v.signum()
}

/// One model row's response at skip count `s`.
fn response(row: usize, s: i32, qstep: i32) -> i32 {
    let skips = s * 7;
    let product = (skips * qstep + 128) >> 8;
    let [a, b, c] = MODE_WEIGHTS[row];
    round13(a * (skips << 5) + b * qstep + c * product)
}

impl ClassModel {
    /// The model of quantizer step `qstep` (dav2d
    /// `setup_wiener_qval_approx_data`).
    #[must_use]
    pub fn new(qstep: i32) -> Self {
        let level = |s: i32| {
            let qval = 255 * (MODE_OFFSETS[0] + response(0, s, qstep));
            (((qval.max(0) + 8192) >> 14).min(255)) >> 5
        };
        let first = level(0);
        let mut low = first;
        let mut changeover = None;
        for s in 1..=MAX_SKIP_COUNT as i32 {
            let value = level(s);
            if value != first {
                low = value;
                changeover = Some(s as usize);
                break;
            }
        }
        let mut model = Self {
            corrections: [[0; MAX_SKIP_COUNT + 1]; 3],
            offsets: [0; 3],
            slopes: [0; 3],
            changeover,
            offset: (low as usize) << SKIP_INDEX_SHIFT,
        };
        for i in 0..3 {
            let [a, _, c] = MODE_WEIGHTS[i + 1];
            let slope =
                (7 * 32 * a + ((7 * i64::from(c) * i64::from(qstep)) >> 8) as i32 + 4096) >> 13;
            let table = |s: i32| MODE_OFFSETS[i + 1] + response(i + 1, s, qstep) - s * slope;
            let base = table(0);
            model.slopes[i] = slope;
            model.offsets[i] = base;
            for (s, correction) in model.corrections[i].iter_mut().enumerate() {
                // dav2d keeps the correction as `int8_t`.
                *correction = i32::from((table(s as i32) - base) as i8);
            }
        }
        model
    }

    /// The feature index (into a class table) of the 4x4 block at (`x`,
    /// `y`) of `window` whose 3x3 neighbourhood has skip count `skips`
    /// (dav2d `wiener_multi`'s classification).
    #[must_use]
    pub fn feature_index<T: Sample>(
        &self,
        window: &Window<'_, T>,
        (x, y): (usize, usize),
        skips: usize,
        depth: BitDepth,
    ) -> usize {
        let features = features(window, (x as isize, y as isize));
        let shift = depth.above_eight();
        let round = (1 << shift) >> 1;
        let below = self.changeover.is_some_and(|changeover| skips < changeover);
        let mut index = usize::from(below) << SKIP_INDEX_SHIFT;
        for i in 0..3 {
            let feature = ((features[i] * NORMALIZERS[i] + round) >> shift) as i32;
            let threshold =
                self.slopes[i] * skips as i32 + self.corrections[i][skips] + self.offsets[i];
            let level =
                ((((feature + 255 * threshold).max(0) + 8192) >> 14).min(255) >> 5) as usize;
            index |= level << (FEATURE_INDEX_BITS * (2 - i as u32));
        }
        self.offset + index
    }
}

/// The three second-derivative features (vertical, anti-diagonal,
/// diagonal) summed over the 6x6 samples around the 4x4 block at (`x`,
/// `y`) (dav2d `calc_features_2x2` over the block and its ring).
fn features<T: Sample>(window: &Window<'_, T>, (x, y): (isize, isize)) -> [u32; 3] {
    let mut sums = [0_u32; 3];
    for py in y - 1..y + 5 {
        for px in x - 1..x + 5 {
            let m = 2 * window.at(px, py);
            let vertical = window.at(px, py - 1) + window.at(px, py + 1) - m;
            let anti_diagonal = window.at(px + 1, py - 1) + window.at(px - 1, py + 1) - m;
            let diagonal = window.at(px - 1, py - 1) + window.at(px + 1, py + 1) - m;
            sums[0] += vertical.unsigned_abs();
            sums[1] += anti_diagonal.unsigned_abs();
            sums[2] += diagonal.unsigned_abs();
        }
    }
    sums
}

/// Each 4x4 block's skip count from its row's and the rows above and
/// below's no-residual flags (dav2d `cache_skip_count_row`): 4 for the
/// block itself and 1 for each side neighbour, rows weighted 1, 4, 1. A
/// row's flags extend past its first and last block by repeating them.
#[must_use]
pub fn skip_counts(rows: [&[bool]; 3]) -> Vec<usize> {
    let row_count = |row: &[bool], b: usize| {
        let skipped = |i: isize| {
            let i = i.clamp(0, row.len() as isize - 1) as usize;
            usize::from(!row[i])
        };
        4 * skipped(b as isize) + skipped(b as isize - 1) + skipped(b as isize + 1)
    };
    (0..rows[1].len())
        .map(|b| row_count(rows[0], b) + 4 * row_count(rows[1], b) + row_count(rows[2], b))
        .collect()
}
