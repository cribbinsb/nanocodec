//! H.266 CCLM template selection and luma gathering; arithmetic uses shared DSP.
//!
//! Luma is downsampled to the chroma grid (§8.4.5.2.14): 4:2:0 by the
//! six-tap (or collocated five-tap) filter, 4:2:2 by the horizontal
//! `[1 2 1]` filter, which is the six-tap filter of one row taken twice,
//! and 4:4:4 not at all.
use super::{Sequence, Unit, references::Availability};
use nc_common::{BitDepth, Picture, PlaneIndex, PlaneView, Sample};
use nc_dsp::{vvc_cclm_downsample, vvc_cclm_downsample_collocated, vvc_cclm_predict};

/// The largest chroma transform block side (64 in 4:4:4).
const MAX_CHROMA_SIDE: usize = 64;

#[derive(Debug)]
pub(super) struct CclmScratch<T: Sample> {
    luma: Box<[T; MAX_CHROMA_SIDE * MAX_CHROMA_SIDE]>,
    key: (u16, u16, u8, u8),
    pair_count: u8,
    coords: [(isize, isize); 4],
    pair_luma: [T; 4],
    valid: bool,
}

impl<T: Sample> Default for CclmScratch<T> {
    fn default() -> Self {
        Self {
            luma: Box::new([T::default(); MAX_CHROMA_SIDE * MAX_CHROMA_SIDE]),
            key: (0, 0, 0, 0),
            pair_count: 0,
            coords: [(0, 0); 4],
            pair_luma: [T::default(); 4],
            valid: false,
        }
    }
}

impl<T: Sample> CclmScratch<T> {
    #[inline]
    pub(super) fn invalidate(&mut self) {
        self.valid = false;
    }
}

/// How luma is downsampled to chroma: the subsampling (log2 `SubWidthC`,
/// `SubHeightC`), whether the left and above neighbours are available,
/// `sps_chroma_vertical_collocated_flag` and the CTU size (log2).
#[derive(Clone, Copy)]
struct Downsampling {
    subsampling: (usize, usize),
    left: bool,
    above: bool,
    collocated: bool,
    ctu_log2: u8,
}

#[inline]
fn downsample_point<T: Sample>(
    luma: &PlaneView<'_, T>,
    (nx, ny): (usize, usize),
    d: Downsampling,
    (cx, cy): (isize, isize),
) -> T {
    let (w_max, h_max) = (luma.width() as isize - 1, luma.height() as isize - 1);
    let sample = |px: isize, py: isize| {
        let sx = (nx as isize + px).clamp(0, w_max) as usize;
        let sy = (ny as isize + py).clamp(0, h_max) as usize;
        luma.row(sy)[sx]
    };
    let (sub_x, sub_y) = d.subsampling;
    if sub_x == 0 {
        return sample(cx, cy);
    }
    let (px, mut py) = (cx << sub_x, cy << sub_y);
    // One luma row: 4:2:2, or 4:2:0's row above a CTU.
    let single_row = sub_y == 0 || (cy == -1 && ny.is_multiple_of(1 << d.ctu_log2));
    if single_row {
        py = cy;
    }
    let lx = if cx == 0 && !d.left { px } else { px - 1 };
    if d.collocated && !single_row {
        let uy = if cy == 0 && !d.above { py } else { py - 1 };
        vvc_cclm_downsample_collocated([
            sample(px, uy),
            sample(lx, py),
            sample(px, py),
            sample(px + 1, py),
            sample(px, py + 1),
        ])
    } else {
        let dy = if single_row { py } else { py + 1 };
        vvc_cclm_downsample([
            sample(lx, py),
            sample(px, py),
            sample(px + 1, py),
            sample(lx, dy),
            sample(px, dy),
            sample(px + 1, dy),
        ])
    }
}

fn downsample_block<T: Sample>(
    luma: &PlaneView<'_, T>,
    (nx, ny): (usize, usize),
    (width, height): (usize, usize),
    d: Downsampling,
    dst: &mut [T],
) {
    let collocated = d.collocated;
    let interior_x = nx > 0 && nx + 2 * width < luma.width();
    let interior_y = ny + 2 * height <= luma.height() && (!collocated || ny > 0);
    if d.subsampling == (1, 1) && interior_x && interior_y {
        for dy in 0..height {
            let py = ny + 2 * dy;
            let row0 = luma.row(py);
            let out_row = &mut dst[dy * width..(dy + 1) * width];
            if collocated {
                let uy = if dy == 0 && !d.above { py } else { py - 1 };
                let row_up = luma.row(uy);
                let row_dn = luma.row((py + 1).min(luma.height() - 1));
                for (dx, slot) in out_row.iter_mut().enumerate() {
                    let px = nx + 2 * dx;
                    let lx = if dx == 0 && !d.left { px } else { px - 1 };
                    *slot = vvc_cclm_downsample_collocated([
                        row_up[px],
                        row0[lx],
                        row0[px],
                        row0[px + 1],
                        row_dn[px],
                    ]);
                }
            } else {
                let row1 = luma.row(py + 1);
                for (dx, slot) in out_row.iter_mut().enumerate() {
                    let px = nx + 2 * dx;
                    let lx = if dx == 0 && !d.left { px } else { px - 1 };
                    *slot = vvc_cclm_downsample([
                        row0[lx],
                        row0[px],
                        row0[px + 1],
                        row1[lx],
                        row1[px],
                        row1[px + 1],
                    ]);
                }
            }
        }
        return;
    }
    for dy in 0..height {
        for dx in 0..width {
            dst[dy * width + dx] = downsample_point(luma, (nx, ny), d, (dx as isize, dy as isize));
        }
    }
}

/// CCLM prediction (H.266 §8.4.5.2.14) of chroma `plane` of `block` at
/// `depth`, reusing the downsampled luma block and template points across
/// `Cb` and `Cr` of the same transform unit.
pub(super) fn predict_cached<T: Sample>(
    cache: &mut CclmScratch<T>,
    picture: &Picture<T>,
    available: &Availability,
    sequence: &Sequence,
    unit: &Unit,
    block: nc_encoder::Block,
    (plane, depth): (PlaneIndex, BitDepth),
    out: &mut [T],
) {
    let n = block;
    let (x, y) = sequence.scaled(true, (n.x, n.y));
    let (width, height) = sequence.scaled(true, (n.width(), n.height()));
    let key = (n.x as u16, n.y as u16, width as u8, height as u8);
    let chroma = picture.plane(plane).view();
    if !(cache.valid && cache.key == key) {
        let left = available.has(1, x as isize - 1, y as isize);
        let above = available.has(1, x as isize, y as isize - 1);
        let luma = picture.plane(PlaneIndex::Y).view();
        let flags = Downsampling {
            subsampling: sequence.subsampling(),
            left,
            above,
            collocated: sequence.tools.chroma_collocated[1],
            ctu_log2: sequence.ctu_log2,
        };
        let mut top_count = if above && unit.chroma_mode != 68 {
            width
        } else {
            0
        };
        let mut left_count = if left && unit.chroma_mode != 69 {
            height
        } else {
            0
        };
        if top_count > 0 && unit.chroma_mode == 69 {
            top_count += (width..width + width.min(height))
                .take_while(|&dx| available.has(1, (x + dx) as isize, y as isize - 1))
                .count();
        }
        if left_count > 0 && unit.chroma_mode == 68 {
            left_count += (height..height + width.min(height))
                .take_while(|&dy| available.has(1, x as isize - 1, (y + dy) as isize))
                .count();
        }
        let mut pair_count = 0usize;
        for (top, count, other) in [
            (true, top_count, left_count),
            (false, left_count, top_count),
        ] {
            if count == 0 {
                continue;
            }
            let single = usize::from(other == 0);
            let start = count >> (2 + single);
            let step = (count >> (1 + single)).max(1);
            for i in 0..count.min(2 << single) {
                let position = start + i * step;
                let coord = if top {
                    (position as isize, -1)
                } else {
                    (-1, position as isize)
                };
                cache.coords[pair_count] = coord;
                cache.pair_luma[pair_count] = downsample_point(&luma, (n.x, n.y), flags, coord);
                pair_count += 1;
            }
        }
        downsample_block(
            &luma,
            (n.x, n.y),
            (width, height),
            flags,
            &mut cache.luma[..width * height],
        );
        cache.key = key;
        cache.pair_count = pair_count as u8;
        cache.valid = true;
    }
    let pair_count = usize::from(cache.pair_count);
    let mut pairs = [[T::default(); 2]; 4];
    for (i, slot) in pairs[..pair_count].iter_mut().enumerate() {
        let (cx, cy) = cache.coords[i];
        let value = chroma.row((y as isize + cy) as usize)[(x as isize + cx) as usize];
        *slot = [cache.pair_luma[i], value];
    }
    vvc_cclm_predict(
        &cache.luma[..width * height],
        &pairs[..pair_count],
        out,
        depth,
    );
}
