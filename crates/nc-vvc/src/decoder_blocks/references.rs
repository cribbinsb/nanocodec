//! General intra availability, reference substitution and smoothing (§8.4.5.2).
use nc_common::{Picture, PlaneIndex};

pub(super) struct Availability {
    dimensions: [usize; 2],
    stride: usize,
    completed: [Vec<bool>; 2],
}
impl Availability {
    pub fn new(dimensions: [usize; 2]) -> Self {
        let stride = dimensions[0].div_ceil(4);
        let count = stride * dimensions[1].div_ceil(4);
        Self {
            dimensions,
            stride,
            completed: [vec![false; count], vec![false; count]],
        }
    }
    pub fn has(&self, channel: usize, x: isize, y: isize) -> bool {
        x >= 0
            && y >= 0
            && (x as usize) < self.dimensions[0] >> channel
            && (y as usize) < self.dimensions[1] >> channel
            && self.completed[channel]
                [((y as usize) << channel >> 2) * self.stride + ((x as usize) << channel >> 2)]
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
                    self.completed[channel][(y >> 2) * self.stride + (x >> 2)] = true;
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
impl References {
    pub fn gather_mrl(
        picture: &Picture,
        available: &Availability,
        (x, y, width, height): (usize, usize, usize, usize),
        line: u8,
    ) -> Self {
        let offset = isize::from(line);
        let view = picture.plane(PlaneIndex::Y).view();
        let sample = |px: isize, py: isize| {
            available
                .has(0, px, py)
                .then(|| i32::from(view.row(py as usize)[px as usize]))
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
        let mut last = chain.iter().find_map(|&v| v).unwrap_or(128);
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
    pub fn gather(
        picture: &Picture,
        available: &Availability,
        plane: PlaneIndex,
        (x, y, width, height): (usize, usize, usize, usize),
        smooth: bool,
    ) -> Self {
        let channel = usize::from(plane != PlaneIndex::Y);
        let view = picture.plane(plane).view();
        let sample = |px: isize, py: isize| {
            available
                .has(channel, px, py)
                .then(|| i32::from(view.row(py as usize)[px as usize]))
        };
        let mut chain = Vec::with_capacity(2 * (width + height) + 1);
        for dy in (0..2 * height).rev() {
            chain.push(sample(x as isize - 1, (y + dy) as isize));
        }
        chain.push(sample(x as isize - 1, y as isize - 1));
        for dx in 0..2 * width {
            chain.push(sample((x + dx) as isize, y as isize - 1));
        }
        let mut last = chain.iter().find_map(|&v| v).unwrap_or(128);
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
