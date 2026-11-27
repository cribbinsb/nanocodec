//! General intra availability, reference substitution and smoothing (§8.4.5.2).
use nc_common::{BitDepth, Picture, PlaneIndex, Sample};

/// Stack-backed reference edge buffer (max `2 * 64 + 4` samples, with headroom).
pub(super) struct RefBuf {
    buf: [i32; 260],
    len: usize,
}

impl std::ops::Deref for RefBuf {
    type Target = [i32];

    #[inline]
    fn deref(&self) -> &[i32] {
        &self.buf[..self.len]
    }
}

#[derive(Debug, Default)]
pub(super) struct Availability {
    dimensions: [usize; 2],
    stride: usize,
    /// `0` when incomplete, else `epoch_base + tile + 1`: another tile's or
    /// earlier slice's samples are unavailable without clearing between slices.
    cells: [Vec<u16>; 2],
    epoch_base: u16,
    tag: u16,
    /// CTU size (log2) when the wavefront rule applies.
    wavefront: Option<u8>,
    /// First luma column that the current CU may not refer to.
    right_limit: usize,
    /// The chroma planes' subsampling (log2 `SubWidthC`, `SubHeightC`).
    subsampling: (usize, usize),
}
impl Availability {
    /// `wavefront`: the CTU size (log2) of a sequence with entropy
    /// synchronisation, where a CU sees nothing in CTU columns to the right
    /// of its own (H.266 §6.4.4).
    #[cfg(test)]
    pub fn new(dimensions: [usize; 2], wavefront: Option<u8>) -> Self {
        let mut avail = Self::default();
        avail.reset(dimensions, wavefront, (0, (1, 1)));
        avail
    }
    /// Prepare for a new slice of `dimensions` whose largest tile index is
    /// `max_tile` and whose chroma has `subsampling`, clearing the grid only
    /// on geometry change or `u16` wrap.
    pub fn reset(
        &mut self,
        dimensions: [usize; 2],
        wavefront: Option<u8>,
        (max_tile, subsampling): (u16, (usize, usize)),
    ) {
        self.subsampling = subsampling;
        let stride = dimensions[0].div_ceil(4);
        let count = stride * dimensions[1].div_ceil(4);
        let need_clear = self.dimensions != dimensions
            || self.cells[0].len() != count
            || u32::from(self.epoch_base) + u32::from(max_tile) + 2 > u32::from(u16::MAX);
        self.dimensions = dimensions;
        self.stride = stride;
        self.wavefront = wavefront;
        self.right_limit = usize::MAX;
        if need_clear {
            for grid in &mut self.cells {
                grid.clear();
                grid.resize(count, 0);
            }
            self.epoch_base = 0;
        }
        self.tag = self.epoch_base + 1;
    }
    /// Advance `epoch_base` past `max_tile` after a slice completes.
    pub fn finish_slice(&mut self, max_tile: u16) {
        self.epoch_base = self.epoch_base.saturating_add(max_tile.saturating_add(1));
    }
    /// Set the CU whose neighbours the following queries are about.
    #[inline]
    pub fn enter(&mut self, unit: &super::Unit) {
        self.tag = self.epoch_base + unit.tile + 1;
        if let Some(ctu_log2) = self.wavefront {
            self.right_limit = ((unit.node.x >> ctu_log2) + 1) << ctu_log2;
        }
    }
    /// The subsampling of `channel` (0 luma, 1 chroma).
    #[inline]
    fn scale(&self, channel: usize) -> (usize, usize) {
        if channel == 0 {
            (0, 0)
        } else {
            self.subsampling
        }
    }
    /// Whether sample `(x, y)` of `channel` (in that channel's samples) is
    /// reconstructed and available to the current CU.
    #[inline]
    pub fn has(&self, channel: usize, x: isize, y: isize) -> bool {
        let (sx, sy) = self.scale(channel);
        if x < 0
            || y < 0
            || (x as usize) << sx >= self.right_limit
            || (x as usize) << sx >= self.dimensions[0]
            || (y as usize) << sy >= self.dimensions[1]
        {
            return false;
        }
        let cell = ((y as usize) << sy >> 2) * self.stride + ((x as usize) << sx >> 2);
        self.cells[channel][cell] == self.tag
    }
    #[inline]
    pub fn mark_transform(&mut self, unit: &super::Unit, transform: &super::Transform) {
        let tag = self.epoch_base + unit.tile + 1;
        for channel in if unit.combined {
            0..2
        } else {
            unit.channel..unit.channel + 1
        } {
            let n = if channel == 0 {
                transform.block
            } else {
                let Some(block) = transform.chroma_block else {
                    continue;
                };
                block
            };
            let cols = n.width().div_ceil(4);
            let grid = &mut self.cells[channel];
            for y in (n.y..n.y + n.height()).step_by(4) {
                let base = (y >> 2) * self.stride + (n.x >> 2);
                grid[base..base + cols].fill(tag);
            }
        }
    }
}

pub(super) struct References {
    pub top: RefBuf,
    pub left: RefBuf,
    pub corner: i32,
}

/// Slice-reusable reference gathering buffers (zero per-block initialization).
#[derive(Debug)]
pub(super) struct RefScratch {
    raw: [i32; 520],
    top: [i32; 260],
    left: [i32; 260],
}

impl Default for RefScratch {
    fn default() -> Self {
        Self {
            raw: [0; 520],
            top: [0; 260],
            left: [0; 260],
        }
    }
}

pub(super) struct RefView<'a> {
    pub top: &'a [i32],
    pub left: &'a [i32],
    pub corner: i32,
}

const UNSET: i32 = -1;

fn substitute_chain(raw: &mut [i32], mid: i32) {
    let mut last = raw.iter().copied().find(|&v| v != UNSET).unwrap_or(mid);
    for slot in raw {
        if *slot == UNSET {
            *slot = last;
        } else {
            last = *slot;
        }
    }
}

/// The intra reference samples (§8.4.5.2.8): when no neighbour is
/// available every one is the mid-grey `1 << (BitDepth − 1)`, else each
/// unavailable one copies the last available one before it.
impl References {
    pub fn gather_mrl_into<'a, T: Sample>(
        scratch: &'a mut RefScratch,
        picture: &Picture<T>,
        available: &Availability,
        (x, y, width, height): (usize, usize, usize, usize),
        (line, depth): (u8, BitDepth),
    ) -> RefView<'a> {
        let offset = isize::from(line);
        let view = picture.plane(PlaneIndex::Y).view();
        let (top_len, left_len) = (
            2 * width + usize::from(line),
            2 * height + usize::from(line),
        );
        let total = left_len + 1 + top_len;
        let raw = &mut scratch.raw[..total];
        raw.fill(UNSET);
        let lx = x as isize - offset - 1;
        for (idx, dy) in (0..left_len).rev().enumerate() {
            let py = y as isize + dy as isize - offset;
            if available.has(0, lx, py) {
                raw[idx] = view.row(py as usize)[lx as usize].to_i32();
            }
        }
        let cy = y as isize - offset - 1;
        if available.has(0, lx, cy) {
            raw[left_len] = view.row(cy as usize)[lx as usize].to_i32();
        }
        if cy >= 0 && (cy as usize) < view.height() {
            let top_row = view.row(cy as usize);
            for dx in 0..top_len {
                let px = x as isize + dx as isize - offset;
                if available.has(0, px, cy) {
                    raw[left_len + 1 + dx] = top_row[px as usize].to_i32();
                }
            }
        }
        substitute_chain(raw, depth.mid());
        let corner = raw[left_len];
        scratch.top[..top_len].copy_from_slice(&raw[left_len + 1..total]);
        for i in 0..left_len {
            scratch.left[i] = scratch.raw[left_len - 1 - i];
        }
        RefView {
            top: &scratch.top[..top_len],
            left: &scratch.left[..left_len],
            corner,
        }
    }

    pub fn gather_into<'a, T: Sample>(
        scratch: &'a mut RefScratch,
        picture: &Picture<T>,
        available: &Availability,
        (plane, depth): (PlaneIndex, BitDepth),
        (x, y, width, height): (usize, usize, usize, usize),
        smooth: bool,
    ) -> RefView<'a> {
        let channel = usize::from(plane != PlaneIndex::Y);
        let (sx, sy) = available.scale(channel);
        let view = picture.plane(plane).view();
        let (left_len, top_len) = (2 * height, 2 * width);
        let total = left_len + 1 + top_len;
        let raw = &mut scratch.raw[..total];
        let tag = available.tag;
        let cells = &available.cells[channel];
        let cell_stride = available.stride;
        let x_limit = (available.dimensions[0] >> sx).min(available.right_limit >> sx);
        let y_limit = available.dimensions[1] >> sy;
        let (cell_width, cell_height) = (4 >> sx, 4 >> sy);
        let mut any_unset = false;
        if x > 0 && x - 1 < x_limit {
            let ux = x - 1;
            let cell_x = (ux << sx) >> 2;
            let valid_rows = left_len.min(y_limit.saturating_sub(y));
            if valid_rows < left_len {
                any_unset = true;
                raw[..left_len - valid_rows].fill(UNSET);
            }
            let data = view.data();
            let p_stride = view.stride();
            let mut dy = 0;
            while dy < valid_rows {
                let ry = y + dy;
                let run = (cell_height - (ry & (cell_height - 1))).min(valid_rows - dy);
                let idx_end = left_len - dy;
                if cells[((ry << sy) >> 2) * cell_stride + cell_x] == tag {
                    for k in 0..run {
                        raw[idx_end - 1 - k] = data[(ry + k) * p_stride + ux].to_i32();
                    }
                } else {
                    any_unset = true;
                    raw[idx_end - run..idx_end].fill(UNSET);
                }
                dy += run;
            }
            if y > 0
                && y - 1 < y_limit
                && cells[(((y - 1) << sy) >> 2) * cell_stride + cell_x] == tag
            {
                raw[left_len] = data[(y - 1) * p_stride + ux].to_i32();
            } else {
                any_unset = true;
                raw[left_len] = UNSET;
            }
        } else {
            any_unset = true;
            raw[..=left_len].fill(UNSET);
        }
        if y > 0 && y - 1 < y_limit && x < x_limit {
            let top_row = view.row(y - 1);
            let row_cells = &cells[(((y - 1) << sy) >> 2) * cell_stride..];
            let valid_cols = top_len.min(x_limit - x);
            let mut dx = 0;
            while dx < valid_cols {
                let rx = x + dx;
                let run = (cell_width - (rx & (cell_width - 1))).min(valid_cols - dx);
                let out_base = left_len + 1 + dx;
                if row_cells[(rx << sx) >> 2] == tag {
                    for k in 0..run {
                        raw[out_base + k] = top_row[rx + k].to_i32();
                    }
                } else {
                    any_unset = true;
                    raw[out_base..out_base + run].fill(UNSET);
                }
                dx += run;
            }
            if valid_cols < top_len {
                any_unset = true;
                raw[left_len + 1 + valid_cols..total].fill(UNSET);
            }
        } else {
            any_unset = true;
            raw[left_len + 1..total].fill(UNSET);
        }
        if any_unset {
            substitute_chain(raw, depth.mid());
        }
        if smooth {
            let mut prev = raw[0];
            for i in 1..total - 1 {
                let cur = raw[i];
                raw[i] = (prev + 2 * cur + raw[i + 1] + 2) >> 2;
                prev = cur;
            }
        }
        let corner = raw[left_len];
        scratch.top[..top_len].copy_from_slice(&raw[left_len + 1..total]);
        for i in 0..left_len {
            scratch.left[i] = scratch.raw[left_len - 1 - i];
        }
        RefView {
            top: &scratch.top[..top_len],
            left: &scratch.left[..left_len],
            corner,
        }
    }

    pub fn gather<T: Sample>(
        picture: &Picture<T>,
        available: &Availability,
        (plane, depth): (PlaneIndex, BitDepth),
        (x, y, width, height): (usize, usize, usize, usize),
        smooth: bool,
    ) -> Self {
        let mut scratch = RefScratch::default();
        let view = Self::gather_into(
            &mut scratch,
            picture,
            available,
            (plane, depth),
            (x, y, width, height),
            smooth,
        );
        let (top_len, left_len) = (view.top.len(), view.left.len());
        let corner = view.corner;
        Self {
            top: RefBuf {
                buf: scratch.top,
                len: top_len,
            },
            left: RefBuf {
                buf: scratch.left,
                len: left_len,
            },
            corner,
        }
    }
}
