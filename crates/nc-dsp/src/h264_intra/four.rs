//! H.264 4×4 intra samples and the packed adjacent averages from libavc.

use crate::Isa;

use super::h264_intra_8x8_mode::{
    COUNT, DC, DIAGONAL_DOWN_LEFT, DIAGONAL_DOWN_RIGHT, HORIZONTAL, HORIZONTAL_DOWN, HORIZONTAL_UP,
    VERTICAL, VERTICAL_LEFT, VERTICAL_RIGHT,
};

const TOP: usize = 6;
const LEFT: usize = 4;

/// Index into the directional values of a two-tap average starting at line
/// position `k`.
const fn pair(k: i32) -> u8 {
    k as u8
}

/// Index into the directional values of a three-tap average starting at
/// line position `k`.
const fn triple(k: i32) -> u8 {
    (16 + k) as u8
}

/// The value sample `(x, y)` of directional `mode` takes (§8.3.1.2.4–9, the
/// standard's formulas with `p[i, -1]` and `p[-1, j]` written as line
/// positions).
const fn directional_value(mode: u8, x: i32, y: i32) -> u8 {
    let top = TOP as i32;
    let left = LEFT as i32;
    match mode {
        // avg3(p[x+y, -1], p[x+y+1, -1], p[x+y+2, -1]); at (3, 3) the
        // repeated p[7, -1] makes it (p[6, -1] + 3p[7, -1] + 2) >> 2.
        DIAGONAL_DOWN_LEFT => triple(top + x + y),
        // Above, on and below the diagonal alike: the three samples centred
        // on line position top - 1 + (x - y).
        DIAGONAL_DOWN_RIGHT => triple(top - 2 + x - y),
        VERTICAL_RIGHT => {
            let i = x - (y >> 1);
            match 2 * x - y {
                0 | 2 | 4 | 6 => pair(top + i - 1),
                1 | 3 | 5 => triple(top + i - 2),
                -1 => triple(left),
                _ => triple(left + 1 - y),
            }
        }
        HORIZONTAL_DOWN => {
            let j = y - (x >> 1);
            match 2 * y - x {
                0 | 2 | 4 | 6 => pair(left - j),
                1 | 3 | 5 => triple(left - j),
                -1 => triple(left),
                _ => triple(top + x - 3),
            }
        }
        VERTICAL_LEFT => {
            let i = x + (y >> 1);
            if y % 2 == 0 {
                pair(top + i)
            } else {
                triple(top + i)
            }
        }
        HORIZONTAL_UP => {
            let j = y + (x >> 1);
            match x + 2 * y {
                0 | 2 | 4 => pair(left - j - 1),
                1 | 3 => triple(left - j - 2),
                // (p[-1, 2] + 3p[-1, 3] + 2) >> 2 over the repeated p[-1, 3].
                5 => triple(0),
                // p[-1, 3]: the average of its two copies.
                _ => pair(0),
            }
        }
        _ => 0,
    }
}

/// For each directional mode (by `Intra4x4PredMode`), the directional value
/// of each sample, raster order: each prediction is sixteen table loads
/// from the shared pairs and triples prepared along the border.
const DIRECTIONAL: [[u8; 16]; COUNT as usize] = {
    let mut table = [[0; 16]; COUNT as usize];
    let mut mode = 0;
    while mode < COUNT {
        let mut i = 0;
        while i < 16 {
            table[mode as usize][i] = directional_value(mode, (i % 4) as i32, (i / 4) as i32);
            i += 1;
        }
        mode += 1;
    }
    table
};

/// Raw 4×4 prediction border. Unavailable above-right samples already
/// repeat the last sample above the block.
#[derive(Debug, Clone, Copy)]
pub struct H264Intra4x4Edges {
    line: [u8; 16],
    has_top: bool,
    has_left: bool,
}

impl H264Intra4x4Edges {
    /// Form the border and DC value from the available neighbours.
    #[must_use]
    pub fn new(top: [u8; 8], left: [u8; 4], corner: u8, (has_top, has_left): (bool, bool)) -> Self {
        let [l0, l1, l2, l3] = left;
        let [t0, t1, t2, t3, t4, t5, t6, t7] = top;
        let line = [
            l3, l3, l2, l1, l0, corner, t0, t1, t2, t3, t4, t5, t6, t7, t7, t7,
        ];
        Self {
            line,
            has_top,
            has_left,
        }
    }

    #[inline]
    fn plain(&self, mode: u8) -> [u8; 16] {
        let line = &self.line;
        match mode {
            VERTICAL => core::array::from_fn(|i| line[TOP + i % 4]),
            HORIZONTAL => core::array::from_fn(|i| line[LEFT - i / 4]),
            _ => [self.dc(); 16],
        }
    }

    fn dc(&self) -> u8 {
        let sum = |samples: &[u8]| samples.iter().copied().map(u16::from).sum::<u16>();
        let top = || sum(&self.line[TOP..TOP + 4]);
        let left = || sum(&self.line[LEFT - 3..=LEFT]);
        let value = match (self.has_top, self.has_left) {
            (true, true) => (top() + left() + 4) >> 3,
            (true, false) => (top() + 2) >> 2,
            (false, true) => (left() + 2) >> 2,
            (false, false) => 128,
        };
        value as u8
    }
}

/// One block's cached adjacent averages for trying several modes.
#[derive(Debug, Clone, Copy)]
pub struct H264Intra4x4Refs {
    edges: H264Intra4x4Edges,
    table: [u8; 32],
}

type PrepareFn = unsafe fn(&[u8; 16]) -> [u8; 32];
type SelectFn = unsafe fn(&[u8; 32], usize) -> [u8; 16];
type SingleFn = unsafe fn(&[u8; 16], usize) -> [u8; 16];

#[derive(Debug, Clone, Copy)]
pub(crate) struct Kernels {
    prepare: PrepareFn,
    select: SelectFn,
    single: SingleFn,
}

impl Kernels {
    pub(crate) fn for_isa(isa: Isa) -> Self {
        match isa {
            #[cfg(target_arch = "x86_64")]
            Isa::Avx2 | Isa::Avx512 => x86::KERNELS,
            #[cfg(target_arch = "aarch64")]
            Isa::Neon | Isa::Sve2 => neon::KERNELS,
            #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
            Isa::Simd128 => wasm::KERNELS,
            _ => Self {
                prepare: prepare_scalar,
                select: select_scalar,
                single: single_scalar,
            },
        }
    }

    #[inline]
    pub(crate) fn prepare(&self, edges: H264Intra4x4Edges) -> H264Intra4x4Refs {
        // SAFETY: the table was built for a checked ISA. Every byte is present.
        let table = unsafe { (self.prepare)(&edges.line) };
        H264Intra4x4Refs { edges, table }
    }

    #[inline]
    pub(crate) fn predict(&self, refs: &H264Intra4x4Refs, mode: u8) -> [u8; 16] {
        if mode <= DC {
            return refs.edges.plain(mode);
        }
        self.directional(&refs.table, mode)
    }

    #[inline]
    pub(crate) fn single(&self, edges: H264Intra4x4Edges, mode: u8) -> [u8; 16] {
        if mode <= DC {
            return edges.plain(mode);
        }
        assert!(mode < COUNT, "invalid H.264 intra4 mode");
        // SAFETY: the mode is in range and the table matches the selected ISA.
        unsafe { (self.single)(&edges.line, usize::from(mode)) }
    }

    #[inline]
    fn directional(&self, table: &[u8; 32], mode: u8) -> [u8; 16] {
        assert!(mode < COUNT, "invalid H.264 intra4 mode");
        // SAFETY: the mode and ISA are valid, and the prepared values occupy
        // exactly 32 bytes.
        unsafe { (self.select)(table, usize::from(mode)) }
    }
}

fn prepare_scalar(line: &[u8; 16]) -> [u8; 32] {
    let mut table = [0; 32];
    for (i, values) in line.windows(2).enumerate() {
        table[i] = super::avg2(values[0], values[1]);
    }
    for (i, values) in line.windows(3).enumerate() {
        table[16 + i] = super::avg3(values[0], values[1], values[2]);
    }
    table
}

fn select_scalar(table: &[u8; 32], mode: usize) -> [u8; 16] {
    DIRECTIONAL[mode].map(|i| table[usize::from(i)])
}

fn single_scalar(line: &[u8; 16], mode: usize) -> [u8; 16] {
    select_scalar(&prepare_scalar(line), mode)
}

#[cfg(any(
    target_arch = "x86_64",
    all(target_arch = "wasm32", target_feature = "simd128")
))]
const MASKS: [[[u8; 16]; 2]; 9] = {
    let mut masks = [[[0x80; 16]; 2]; 9];
    let mut mode = 0;
    while mode < 9 {
        let mut i = 0;
        while i < 16 {
            let index = DIRECTIONAL[mode][i] as usize;
            masks[mode][index / 16][i] = (index % 16) as u8;
            i += 1;
        }
        mode += 1;
    }
    masks
};

#[cfg(target_arch = "aarch64")]
#[allow(unsafe_code)]
#[path = "four_neon.rs"]
mod neon;
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[allow(unsafe_code)]
#[path = "four_wasm.rs"]
mod wasm;
#[cfg(target_arch = "x86_64")]
#[allow(unsafe_code)]
#[path = "four_x86.rs"]
mod x86;
