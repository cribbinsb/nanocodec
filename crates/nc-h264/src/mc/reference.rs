//! Plain per-sample reference of H.264 inter prediction (§8.4.2.2), used
//! only by tests to check the nc-dsp kernels independently.
//!
//! Luma: the 6-tap filter (1, −5, 20, 20, −5, 1) for half samples, averages
//! for quarter samples (§8.4.2.2.1, Table 8-12). Chroma (4:2:0): bilinear
//! interpolation at 1/8 sample (§8.4.2.2.2). Reference coordinates outside
//! the picture are clamped to its edges (eq. 8-228..8-231), which is what
//! the decoder does, so the reference's own border is never read.
//!

use nc_common::PlaneView;
use nc_encoder::Mv;

/// Luma macroblock size.
const MB: usize = 16;
/// Chroma block size of a macroblock (4:2:0).
const MB_CHROMA: usize = 8;
/// Extra samples the 6-tap filter reads before / after a position.
const TAPS_BEFORE: usize = 2;
const TAPS_AFTER: usize = 3;
/// Side of the integer-sample window the 16x16 prediction reads.
const WINDOW: usize = MB + TAPS_BEFORE + TAPS_AFTER + 1;

/// The H.264 6-tap half-sample filter, unnormalised (`b1`, `h1` of eq. 8-241).
fn tap6(s: [i32; 6]) -> i32 {
    s[0] - 5 * s[1] + 20 * s[2] + 20 * s[3] - 5 * s[4] + s[5]
}

fn clip1(v: i32) -> i32 {
    v.clamp(0, 255)
}

/// Predict the 16x16 luma block at `(x, y)` from `reference` displaced by
/// `mv` (quarter samples), into `out` (raster).
pub fn predict_luma_16x16(
    reference: &PlaneView<'_>,
    x: usize,
    y: usize,
    mv: Mv,
    out: &mut [u8; 256],
) {
    let (mvx, mvy) = (i32::from(mv.x), i32::from(mv.y));
    let frac = ((mvx & 3) as usize, (mvy & 3) as usize);
    // Integer window: window[r][c] = G(c - 2, r - 2) relative to the block's
    // whole-sample position (eq. 8-228..8-231 clamp the coordinates).
    let left = x as i32 + (mvx >> 2) - TAPS_BEFORE as i32;
    let top = y as i32 + (mvy >> 2) - TAPS_BEFORE as i32;
    let (width, height) = (reference.width() as i32, reference.height() as i32);
    let mut window = [[0i32; WINDOW]; WINDOW];
    for (r, row) in window.iter_mut().enumerate() {
        let src = reference.row((top + r as i32).clamp(0, height - 1) as usize);
        for (c, v) in row.iter_mut().enumerate() {
            *v = i32::from(src[(left + c as i32).clamp(0, width - 1) as usize]);
        }
    }
    if frac == (0, 0) {
        for (j, row) in out.as_chunks_mut::<MB>().0.iter_mut().enumerate() {
            for (i, v) in row.iter_mut().enumerate() {
                *v = window[j + TAPS_BEFORE][i + TAPS_BEFORE] as u8;
            }
        }
        return;
    }
    let planes = HalfSamples::new(&window);
    for (j, row) in out.as_chunks_mut::<MB>().0.iter_mut().enumerate() {
        for (i, v) in row.iter_mut().enumerate() {
            *v = planes.sample(&window, i, j, frac) as u8;
        }
    }
}

/// The half-sample values around a 16x16 block (eq. 8-241..8-247), with
/// `b1` kept unnormalised for the centre filter.
struct HalfSamples {
    /// `b1` at (i, row) for every window row: horizontal half samples.
    horizontal: [[i32; MB]; WINDOW],
    /// `h` at (i, j), i in 0..=16: vertical half samples, normalised.
    vertical: [[i32; MB + 1]; MB],
    /// `j` at (i, j): centre half samples, normalised.
    centre: [[i32; MB]; MB],
}

impl HalfSamples {
    fn new(window: &[[i32; WINDOW]; WINDOW]) -> Self {
        let mut horizontal = [[0i32; MB]; WINDOW];
        for (row, out) in window.iter().zip(horizontal.iter_mut()) {
            for (i, v) in out.iter_mut().enumerate() {
                *v = tap6([
                    row[i],
                    row[i + 1],
                    row[i + 2],
                    row[i + 3],
                    row[i + 4],
                    row[i + 5],
                ]);
            }
        }
        let mut vertical = [[0i32; MB + 1]; MB];
        for (j, out) in vertical.iter_mut().enumerate() {
            for (i, v) in out.iter_mut().enumerate() {
                let col = i + TAPS_BEFORE;
                let taps = core::array::from_fn(|k| window[j + k][col]);
                *v = clip1((tap6(taps) + 16) >> 5);
            }
        }
        let mut centre = [[0i32; MB]; MB];
        for (j, out) in centre.iter_mut().enumerate() {
            for (i, v) in out.iter_mut().enumerate() {
                let taps = core::array::from_fn(|k| horizontal[j + k][i]);
                *v = clip1((tap6(taps) + 512) >> 10);
            }
        }
        Self {
            horizontal,
            vertical,
            centre,
        }
    }

    /// `b` at (i, j): normalised horizontal half sample.
    fn horizontal_at(&self, i: usize, j: usize) -> i32 {
        clip1((self.horizontal[j + TAPS_BEFORE][i] + 16) >> 5)
    }

    /// The prediction sample at (i, j) for fraction `frac` (Table 8-12).
    fn sample(
        &self,
        window: &[[i32; WINDOW]; WINDOW],
        i: usize,
        j: usize,
        frac: (usize, usize),
    ) -> i32 {
        let full = |di: usize, dj: usize| window[j + dj + TAPS_BEFORE][i + di + TAPS_BEFORE];
        let avg = |p: i32, q: i32| (p + q + 1) >> 1;
        let half_h = self.horizontal_at(i, j); // b
        let half_h_below = || self.horizontal_at(i, j + 1); // s
        let half_v = self.vertical[j][i]; // h
        let half_v_right = self.vertical[j][i + 1]; // m
        let mid = self.centre[j][i]; // j
        match frac {
            (0, 0) => full(0, 0),
            (0, 1) => avg(full(0, 0), half_v),           // d
            (0, 2) => half_v,                            // h
            (0, 3) => avg(full(0, 1), half_v),           // n
            (1, 0) => avg(full(0, 0), half_h),           // a
            (1, 1) => avg(half_h, half_v),               // e
            (1, 2) => avg(half_v, mid),                  // i
            (1, 3) => avg(half_v, half_h_below()),       // p
            (2, 0) => half_h,                            // b
            (2, 1) => avg(half_h, mid),                  // f
            (2, 2) => mid,                               // j
            (2, 3) => avg(mid, half_h_below()),          // q
            (3, 0) => avg(full(1, 0), half_h),           // c
            (3, 1) => avg(half_h, half_v_right),         // g
            (3, 2) => avg(mid, half_v_right),            // k
            (3, 3) => avg(half_v_right, half_h_below()), // r
            _ => unreachable!("fractions are two bits"),
        }
    }
}

/// Predict the 8x8 chroma block at chroma position `(x, y)` from `reference`
/// with the luma vector `mv` (which is in 1/8 chroma samples for 4:2:0).
pub fn predict_chroma_8x8(
    reference: &PlaneView<'_>,
    x: usize,
    y: usize,
    mv: Mv,
    out: &mut [u8; 64],
) {
    let (mvx, mvy) = (i32::from(mv.x), i32::from(mv.y));
    let (fx, fy) = (mvx & 7, mvy & 7);
    let left = x as i32 + (mvx >> 3);
    let top = y as i32 + (mvy >> 3);
    let (width, height) = (reference.width() as i32, reference.height() as i32);
    let sample = |dx: i32, dy: i32| {
        let row = reference.row((top + dy).clamp(0, height - 1) as usize);
        i32::from(row[(left + dx).clamp(0, width - 1) as usize])
    };
    for j in 0..MB_CHROMA as i32 {
        for i in 0..MB_CHROMA as i32 {
            // Eq. 8-266.
            let v = ((8 - fx) * (8 - fy) * sample(i, j)
                + fx * (8 - fy) * sample(i + 1, j)
                + (8 - fx) * fy * sample(i, j + 1)
                + fx * fy * sample(i + 1, j + 1)
                + 32)
                >> 6;
            out[(j as usize) * MB_CHROMA + i as usize] = v as u8;
        }
    }
}
