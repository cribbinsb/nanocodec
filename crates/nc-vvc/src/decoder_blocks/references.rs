//! General intra availability, reference substitution and smoothing (§8.4.5.2).
use nc_common::{BitDepth, Picture, PlaneIndex, Sample};

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
            for y in (n.y..n.y + n.height()).step_by(4) {
                for x in (n.x..n.x + n.width()).step_by(4) {
                    let cell = (y >> 2) * self.stride + (x >> 2);
                    self.completed[channel][cell] = true;
                    self.tiles[channel][cell] = unit.tile;
                }
            }
        }
    }
}

pub(super) struct References {
    pub top: Vec<i32>,
    pub left: Vec<i32>,
    pub corner: i32,
}
/// The intra reference samples (§8.4.5.2.8): when no neighbour is
/// available every one is the mid-grey `1 << (BitDepth − 1)`, else each
/// unavailable one copies the last available one before it.
impl References {
    pub fn gather_mrl<T: Sample>(
        picture: &Picture<T>,
        available: &Availability,
        (x, y, width, height): (usize, usize, usize, usize),
        (line, depth): (u8, BitDepth),
    ) -> Self {
        let offset = isize::from(line);
        let view = picture.plane(PlaneIndex::Y).view();
        let sample = |px: isize, py: isize| {
            available
                .has(0, px, py)
                .then(|| view.row(py as usize)[px as usize].to_i32())
        };
        let (top_length, left_length) = (
            2 * width + usize::from(line),
            2 * height + usize::from(line),
        );
        let mut chain = Vec::with_capacity(top_length + left_length + 1);
        for dy in (0..left_length).rev() {
            chain.push(sample(
                x as isize - offset - 1,
                y as isize + dy as isize - offset,
            ));
        }
        chain.push(sample(x as isize - offset - 1, y as isize - offset - 1));
        for dx in 0..top_length {
            chain.push(sample(
                x as isize + dx as isize - offset,
                y as isize - offset - 1,
            ));
        }
        let mut last = chain.iter().find_map(|&v| v).unwrap_or(depth.mid());
        let samples: Vec<_> = chain
            .into_iter()
            .map(|v| {
                last = v.unwrap_or(last);
                last
            })
            .collect();
        Self {
            top: samples[left_length + 1..].to_vec(),
            left: samples[..left_length].iter().rev().copied().collect(),
            corner: samples[left_length],
        }
    }
    pub fn gather<T: Sample>(
        picture: &Picture<T>,
        available: &Availability,
        (plane, depth): (PlaneIndex, BitDepth),
        (x, y, width, height): (usize, usize, usize, usize),
        smooth: bool,
    ) -> Self {
        let channel = usize::from(plane != PlaneIndex::Y);
        let view = picture.plane(plane).view();
        let sample = |px: isize, py: isize| {
            available
                .has(channel, px, py)
                .then(|| view.row(py as usize)[px as usize].to_i32())
        };
        let mut chain = Vec::with_capacity(2 * (width + height) + 1);
        for dy in (0..2 * height).rev() {
            chain.push(sample(x as isize - 1, (y + dy) as isize));
        }
        chain.push(sample(x as isize - 1, y as isize - 1));
        for dx in 0..2 * width {
            chain.push(sample((x + dx) as isize, y as isize - 1));
        }
        let mut last = chain.iter().find_map(|&v| v).unwrap_or(depth.mid());
        let mut raw: Vec<_> = chain
            .into_iter()
            .map(|v| {
                last = v.unwrap_or(last);
                last
            })
            .collect();
        if smooth {
            let source = raw.clone();
            for i in 1..raw.len() - 1 {
                raw[i] = (source[i - 1] + 2 * source[i] + source[i + 1] + 2) >> 2;
            }
        }
        Self {
            top: raw[2 * height + 1..].to_vec(),
            left: raw[..2 * height].iter().rev().copied().collect(),
            corner: raw[2 * height],
        }
    }
}
