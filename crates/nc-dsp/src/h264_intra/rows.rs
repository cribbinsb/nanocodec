//! The three non-directional H.264 8x8 modes and the packed row
//! selectors for the six remaining modes. Each directional output
//! is one selection from a 16-byte window of pair or triple averages.

use super::{H264Intra8x8Refs, LEFT0, TOP0, h264_intra_8x8_mode as mode};

/// Fill the three plain modes without preparing directional averages.
#[inline]
pub(super) fn plain(refs: &H264Intra8x8Refs, mode: u8, out: &mut [u8; 64]) {
    match mode {
        mode::VERTICAL => {
            let top = &refs.border[TOP0..TOP0 + 8];
            for row in out.as_chunks_mut::<8>().0 {
                row.copy_from_slice(top);
            }
        }
        mode::HORIZONTAL => {
            for (y, row) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                row.fill(refs.l(y));
            }
        }
        mode::DC => out.fill(refs.dc()),
        _ => unreachable!("the caller selected a plain Intra_8x8 mode"),
    }
}

/// Where the 16 consecutive pair and triple averages begin in the border.
pub(super) const FIRST: [u8; 6] = [TOP0 as u8, 1, 2, 1, TOP0 as u8, 0];

/// A pair's index is 0..15 and a triple's is 16..31 within each mode.
const PAIR_COUNT: isize = 16;

const fn selector(mode: u8, x: isize, y: isize, first: isize) -> u8 {
    let top = TOP0 as isize;
    let left = LEFT0 as isize;
    let corner = super::CORNER as isize;
    let triple = PAIR_COUNT;
    let index = match mode {
        mode::DIAGONAL_DOWN_LEFT => triple + top + x + y,
        mode::DIAGONAL_DOWN_RIGHT => triple + corner - 1 + x - y,
        mode::VERTICAL_RIGHT => {
            let z = 2 * x - y;
            let k = x - (y >> 1);
            if z >= 0 && z % 2 == 0 {
                corner + k
            } else if z > 0 {
                triple + corner + k - 1
            } else {
                triple + corner + z
            }
        }
        mode::HORIZONTAL_DOWN => {
            let z = 2 * y - x;
            let j = y - (x >> 1);
            if z >= 0 && z % 2 == 0 {
                left - j
            } else if z > 0 {
                triple + left - j
            } else {
                triple + left - 1 - z
            }
        }
        mode::VERTICAL_LEFT => {
            let k = x + (y >> 1);
            let offset = if y % 2 == 0 { 0 } else { triple };
            offset + top + k
        }
        mode::HORIZONTAL_UP => {
            let z = x + 2 * y;
            let j = y + (x >> 1);
            if z > 13 {
                // The first two border bytes are both the lowest left pixel.
                0
            } else if z % 2 == 0 {
                left - j - 1
            } else {
                triple + left - j - 2
            }
        }
        _ => panic!("expected a directional Intra_8x8 mode"),
    };
    (index - first) as u8
}

const fn indices() -> [[u8; 64]; 6] {
    let mut modes = [[0; 64]; 6];
    let mut mode = 0;
    while mode < modes.len() {
        let mut pixel = 0;
        while pixel < 64 {
            let index = selector(
                mode::DIAGONAL_DOWN_LEFT + mode as u8,
                (pixel % 8) as isize,
                (pixel / 8) as isize,
                FIRST[mode] as isize,
            );
            assert!(index < 32);
            modes[mode][pixel] = index;
            pixel += 1;
        }
        mode += 1;
    }
    modes
}

pub(super) const INDICES: [[u8; 64]; 6] = indices();

/// Ice Lake selects pairs from its first vector and triples from the second.
#[cfg(target_arch = "x86_64")]
const fn icl_indices() -> [[u8; 64]; 6] {
    let mut modes = INDICES;
    let mut mode = 0;
    while mode < modes.len() {
        let mut pixel = 0;
        while pixel < modes[mode].len() {
            if modes[mode][pixel] >= 16 {
                modes[mode][pixel] += 48;
            }
            pixel += 1;
        }
        mode += 1;
    }
    modes
}

#[cfg(target_arch = "x86_64")]
pub(super) const ICL_INDICES: [[u8; 64]; 6] = icl_indices();

/// On lane-local swizzles each table's unavailable indices zero the lane.
#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
const fn split_masks() -> [[[u8; 64]; 2]; 6] {
    let mut masks = [[[0x80; 64]; 2]; 6];
    let mut mode = 0;
    while mode < masks.len() {
        let mut pixel = 0;
        while pixel < 64 {
            let index = INDICES[mode][pixel] as usize;
            masks[mode][index / 16][pixel] = (index % 16) as u8;
            pixel += 1;
        }
        mode += 1;
    }
    masks
}

#[cfg(any(target_arch = "x86_64", target_arch = "wasm32"))]
pub(super) const MASKS: [[[u8; 64]; 2]; 6] = split_masks();
