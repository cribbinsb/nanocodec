//! Constant tables of the H.264 decoding process used by the encoder.
//!
//! Small normative tables are written out here with their spec references;
//! the large CABAC tables are generated from openh264 (`generated`), as are
//! the decoder's CAVLC code tables (`cavlc`).

pub mod cavlc;
mod generated;

pub use generated::{
    CABAC_INIT_MN, CONTEXT_COUNT, DEBLOCK_ALPHA, DEBLOCK_BETA, DEBLOCK_TC0, RANGE_TAB_LPS,
    TRANS_IDX,
};

/// Frame zig-zag scan of a 4x4 block (H.264 §8.5.6, Table 8-13): scan
/// position → raster index (`row * 4 + column`).
pub const ZIGZAG_4X4: [usize; 16] = [0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15];

/// Frame zig-zag scan of an 8x8 block (H.264 §8.5.7, Table 8-14): scan
/// position → raster index (`row * 8 + column`).
pub const ZIGZAG_8X8: [usize; 64] = [
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, //
    12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28, //
    35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, //
    58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
];

/// ctxIdxInc of `significant_coeff_flag` in a frame-coded 8x8 block, by
/// scan position (H.264 Table 9-43, frame coded column).
pub const SIGNIFICANT_CTX_INC_8X8: [u8; 63] = [
    0, 1, 2, 3, 4, 5, 5, 4, 4, 3, 3, 4, 4, 4, 5, 5, //
    4, 4, 4, 4, 3, 3, 6, 7, 7, 7, 8, 9, 10, 9, 8, 7, //
    7, 6, 11, 12, 13, 11, 6, 7, 8, 9, 14, 10, 9, 8, 6, 11, //
    12, 13, 11, 6, 9, 14, 10, 9, 11, 12, 13, 11, 14, 10, 12,
];

/// ctxIdxInc of `last_significant_coeff_flag` in an 8x8 block, by scan
/// position (H.264 Table 9-43).
pub const LAST_CTX_INC_8X8: [u8; 63] = [
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, //
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, //
    3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, //
    5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8,
];

/// `QPc` as a function of `qPI` for `qPI >= 30` (H.264 §8.5.8, Table 8-15);
/// below 30, `QPc = qPI`.
const CHROMA_QP_FROM_30: [u8; 22] = [
    29, 30, 31, 32, 32, 33, 34, 34, 35, 35, 36, 36, 37, 37, 37, 38, 38, 38, 39, 39, 39, 39,
];

/// Chroma QP for luma `qp` and the PPS `chroma_qp_index_offset` (8-bit
/// video, so `QpBdOffsetC = 0` and `qPI = Clip3(0, 51, QPY + offset)`,
/// H.264 §8.5.8).
pub fn chroma_qp(qp: u8, offset: i8) -> u8 {
    let qpi = (i32::from(qp) + i32::from(offset)).clamp(0, 51) as usize;
    CHROMA_QP[qpi]
}

/// `QPc` for every `qPI` (one lookup, no branch: the deblocking filter
/// asks per edge).
const CHROMA_QP: [u8; 52] = {
    let mut table = [0; 52];
    let mut qpi = 0;
    while qpi < 52 {
        table[qpi] = if qpi < 30 {
            qpi as u8
        } else {
            CHROMA_QP_FROM_30[qpi - 30]
        };
        qpi += 1;
    }
    table
};

/// One row of H.264 Table A-1 (level limits) that the encoder checks.
#[derive(Debug, Clone, Copy)]
pub struct LevelLimits {
    /// `level_idc`.
    pub level_idc: u8,
    /// `MaxFS`: maximum frame size in macroblocks.
    pub max_frame_mbs: u32,
    /// `MaxMBPS`: maximum macroblock processing rate.
    pub max_mbs_per_second: u32,
}

/// H.264 Table A-1, in increasing order (level 1b omitted).
pub const LEVELS: [LevelLimits; 15] = [
    LevelLimits {
        level_idc: 10,
        max_frame_mbs: 99,
        max_mbs_per_second: 1485,
    },
    LevelLimits {
        level_idc: 11,
        max_frame_mbs: 396,
        max_mbs_per_second: 3000,
    },
    LevelLimits {
        level_idc: 12,
        max_frame_mbs: 396,
        max_mbs_per_second: 6000,
    },
    LevelLimits {
        level_idc: 13,
        max_frame_mbs: 396,
        max_mbs_per_second: 11880,
    },
    LevelLimits {
        level_idc: 21,
        max_frame_mbs: 792,
        max_mbs_per_second: 19800,
    },
    LevelLimits {
        level_idc: 22,
        max_frame_mbs: 1620,
        max_mbs_per_second: 20250,
    },
    LevelLimits {
        level_idc: 30,
        max_frame_mbs: 1620,
        max_mbs_per_second: 40500,
    },
    LevelLimits {
        level_idc: 31,
        max_frame_mbs: 3600,
        max_mbs_per_second: 108_000,
    },
    LevelLimits {
        level_idc: 32,
        max_frame_mbs: 5120,
        max_mbs_per_second: 216_000,
    },
    LevelLimits {
        level_idc: 40,
        max_frame_mbs: 8192,
        max_mbs_per_second: 245_760,
    },
    LevelLimits {
        level_idc: 42,
        max_frame_mbs: 8704,
        max_mbs_per_second: 522_240,
    },
    LevelLimits {
        level_idc: 50,
        max_frame_mbs: 22080,
        max_mbs_per_second: 589_824,
    },
    LevelLimits {
        level_idc: 51,
        max_frame_mbs: 36864,
        max_mbs_per_second: 983_040,
    },
    LevelLimits {
        level_idc: 52,
        max_frame_mbs: 36864,
        max_mbs_per_second: 2_073_600,
    },
    LevelLimits {
        level_idc: 60,
        max_frame_mbs: 139_264,
        max_mbs_per_second: 4_177_920,
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zigzag_is_a_permutation() {
        let mut seen = [false; 16];
        for &p in &ZIGZAG_4X4 {
            seen[p] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn chroma_qp_table_endpoints() {
        assert_eq!(chroma_qp(29, 0), 29);
        assert_eq!(chroma_qp(30, 0), 29);
        assert_eq!(chroma_qp(51, 0), 39);
        assert_eq!(chroma_qp(33, -3), 29);
        assert_eq!(chroma_qp(2, -3), 0);
        assert_eq!(chroma_qp(50, 3), 39);
    }
}
