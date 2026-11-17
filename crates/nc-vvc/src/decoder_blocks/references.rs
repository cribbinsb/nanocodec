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

pub(super) struct Availability {
    dimensions: [usize; 2],
    stride: usize,
    completed: [Vec<bool>; 2],
    /// Tile index of each completed cell, and of the current CU: another
    /// tile's samples are unavailable.
    tiles: [Vec<u16>; 2],
    tile: u16,
    /// CTU size (log2) when the wavefront rule applies.
    wavefront: Option<u8>,
    /// First luma column that the current CU may not refer to.
    right_limit: usize,
}
impl Availability {
    /// `wavefront`: the CTU size (log2) of a sequence with entropy
    /// synchronisation, where a CU sees nothing in CTU columns to the right
    /// of its own (H.266 §6.4.4).
    pub fn new(dimensions: [usize; 2], wavefront: Option<u8>) -> Self {
        let stride = dimensions[0].div_ceil(4);
        let count = stride * dimensions[1].div_ceil(4);
        Self {
            dimensions,
            stride,
            completed: [vec![false; count], vec![false; count]],
            tiles: [vec![0; count], vec![0; count]],
            tile: 0,
            wavefront,
            right_limit: usize::MAX,
        }
    }
    /// Set the CU whose neighbours the following queries are about.
    pub fn enter(&mut self, unit: &super::Unit) {
        self.tile = unit.tile;
        if let Some(ctu_log2) = self.wavefront {
            self.right_limit = ((unit.node.x >> ctu_log2) + 1) << ctu_log2;
        }
    }
    #[inline]
    pub fn has(&self, channel: usize, x: isize, y: isize) -> bool {
        if x < 0
            || y < 0
            || (x as usize) << channel >= self.right_limit
            || (x as usize) >= self.dimensions[0] >> channel
            || (y as usize) >= self.dimensions[1] >> channel
        {
            return false;
        }
        let cell = ((y as usize) << channel >> 2) * self.stride + ((x as usize) << channel >> 2);
        self.completed[channel][cell] && self.tiles[channel][cell] == self.tile
    }
    pub fn mark_transform(&mut self, unit: &super::Unit, transform: &super::Transform) {
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
            for y in (n.y..n.y + n.height()).step_by(4) {
                let base = (y >> 2) * self.stride + (n.x >> 2);
                self.completed[channel][base..base + cols].fill(true);
                self.tiles[channel][base..base + cols].fill(unit.tile);
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
        let view = picture.plane(plane).view();
        let (left_len, top_len) = (2 * height, 2 * width);
        let total = left_len + 1 + top_len;
        let raw = &mut scratch.raw[..total];
        raw.fill(UNSET);
        let lx = x as isize - 1;
        if lx >= 0 {
            let ux = lx as usize;
            for (idx, dy) in (0..left_len).rev().enumerate() {
                let py = (y + dy) as isize;
                if available.has(channel, lx, py) {
                    raw[idx] = view.row(y + dy)[ux].to_i32();
                }
            }
            if available.has(channel, lx, y as isize - 1) {
                raw[left_len] = view.row(y - 1)[ux].to_i32();
            }
        }
        if y > 0 {
            let top_row = view.row(y - 1);
            let py = y as isize - 1;
            for dx in 0..top_len {
                let px = (x + dx) as isize;
                if available.has(channel, px, py) {
                    raw[left_len + 1 + dx] = top_row[x + dx].to_i32();
                }
            }
        }
        substitute_chain(raw, depth.mid());
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
