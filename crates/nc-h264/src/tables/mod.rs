//! Constant tables of the H.264 decoding process used by the encoder.
//!
//! Small normative tables are written out here with their spec references;
//! the large CABAC tables are generated from openh264 (`generated`), as are
//! the decoder's CAVLC code tables (`cavlc`), except the 4:2:2 chroma DC
//! ones (`cavlc_422`).

pub mod cavlc;
pub mod cavlc_422;
mod generated;

pub use generated::{
    CABAC_INIT_MN, CONTEXT_COUNT, DEBLOCK_ALPHA, DEBLOCK_BETA, DEBLOCK_TC0, RANGE_TAB_LPS,
    TRANS_IDX,
};

/// Frame zig-zag scan of a 4x4 block (H.264 §8.5.6, Table 8-13): scan
/// position → raster index (`row * 4 + column`).
pub use nc_dsp::H264_ZIGZAG_4X4 as ZIGZAG_4X4;

pub use nc_dsp::H264_ZIGZAG_8X8 as ZIGZAG_8X8;

/// Field scan of a 4x4 block (H.264 §8.5.6, Table 8-13, field
/// macroblocks): scan position → raster index.
pub const FIELD_SCAN_4X4: [usize; 16] = [0, 4, 1, 8, 12, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15];

/// Field scan of an 8x8 block (H.264 §8.5.7, Table 8-14, field
/// macroblocks): scan position → raster index.
pub const FIELD_SCAN_8X8: [usize; 64] = [
    0, 8, 16, 1, 9, 24, 32, 17, 2, 25, 40, 48, 56, 33, 10, 3, //
    18, 41, 49, 57, 26, 11, 4, 19, 34, 42, 50, 58, 27, 12, 5, 20, //
    35, 43, 51, 59, 28, 13, 6, 21, 36, 44, 52, 60, 29, 14, 22, 37, //
    45, 53, 61, 30, 7, 15, 38, 46, 54, 62, 23, 31, 39, 47, 55, 63,
];

/// ctxIdxInc of `significant_coeff_flag` in a field-coded 8x8 block, by
/// scan position (H.264 Table 9-43, field coded column).
pub const SIGNIFICANT_CTX_INC_8X8_FIELD: [u8; 63] = [
    0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 7, 8, 4, 5, //
    6, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10, 8, 11, 12, 11, //
    9, 9, 10, 10, 8, 11, 12, 11, 9, 9, 10, 10, 8, 13, 13, 9, //
    9, 10, 10, 8, 13, 13, 9, 9, 10, 10, 14, 14, 14, 14, 14,
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
    /// `MaxBR`, in units of `cpbBrVclFactor` bits/s (Table A-2).
    pub max_br: u32,
    /// `MaxCPB`, in units of `cpbBrVclFactor` bits.
    pub max_cpb: u32,
}

/// `cpbBrVclFactor` of the High profile (Table A-2): bits per `MaxBR` /
/// `MaxCPB` unit.
pub const HIGH_CPB_BR_VCL_FACTOR: u64 = 1250;

/// H.264 Table A-1, in increasing order (level 1b omitted).
pub const LEVELS: [LevelLimits; 19] = [
    LevelLimits {
        level_idc: 10,
        max_frame_mbs: 99,
        max_mbs_per_second: 1485,
        max_br: 64,
        max_cpb: 175,
    },
    LevelLimits {
        level_idc: 11,
        max_frame_mbs: 396,
        max_mbs_per_second: 3000,
        max_br: 192,
        max_cpb: 500,
    },
    LevelLimits {
        level_idc: 12,
        max_frame_mbs: 396,
        max_mbs_per_second: 6000,
        max_br: 384,
        max_cpb: 1000,
    },
    LevelLimits {
        level_idc: 13,
        max_frame_mbs: 396,
        max_mbs_per_second: 11880,
        max_br: 768,
        max_cpb: 2000,
    },
    LevelLimits {
        level_idc: 20,
        max_frame_mbs: 396,
        max_mbs_per_second: 11880,
        max_br: 2000,
        max_cpb: 2000,
    },
    LevelLimits {
        level_idc: 21,
        max_frame_mbs: 792,
        max_mbs_per_second: 19800,
        max_br: 4000,
        max_cpb: 4000,
    },
    LevelLimits {
        level_idc: 22,
        max_frame_mbs: 1620,
        max_mbs_per_second: 20250,
        max_br: 4000,
        max_cpb: 4000,
    },
    LevelLimits {
        level_idc: 30,
        max_frame_mbs: 1620,
        max_mbs_per_second: 40500,
        max_br: 10000,
        max_cpb: 10000,
    },
    LevelLimits {
        level_idc: 31,
        max_frame_mbs: 3600,
        max_mbs_per_second: 108_000,
        max_br: 14000,
        max_cpb: 14000,
    },
    LevelLimits {
        level_idc: 32,
        max_frame_mbs: 5120,
        max_mbs_per_second: 216_000,
        max_br: 20000,
        max_cpb: 20000,
    },
    LevelLimits {
        level_idc: 40,
        max_frame_mbs: 8192,
        max_mbs_per_second: 245_760,
        max_br: 20000,
        max_cpb: 25000,
    },
    LevelLimits {
        level_idc: 41,
        max_frame_mbs: 8192,
        max_mbs_per_second: 245_760,
        max_br: 50000,
        max_cpb: 62500,
    },
    LevelLimits {
        level_idc: 42,
        max_frame_mbs: 8704,
        max_mbs_per_second: 522_240,
        max_br: 50000,
        max_cpb: 62500,
    },
    LevelLimits {
        level_idc: 50,
        max_frame_mbs: 22080,
        max_mbs_per_second: 589_824,
        max_br: 135_000,
        max_cpb: 135_000,
    },
    LevelLimits {
        level_idc: 51,
        max_frame_mbs: 36864,
        max_mbs_per_second: 983_040,
        max_br: 240_000,
        max_cpb: 240_000,
    },
    LevelLimits {
        level_idc: 52,
        max_frame_mbs: 36864,
        max_mbs_per_second: 2_073_600,
        max_br: 240_000,
        max_cpb: 240_000,
    },
    LevelLimits {
        level_idc: 60,
        max_frame_mbs: 139_264,
        max_mbs_per_second: 4_177_920,
        max_br: 240_000,
        max_cpb: 240_000,
    },
    LevelLimits {
        level_idc: 61,
        max_frame_mbs: 139_264,
        max_mbs_per_second: 8_355_840,
        max_br: 480_000,
        max_cpb: 480_000,
    },
    LevelLimits {
        level_idc: 62,
        max_frame_mbs: 139_264,
        max_mbs_per_second: 16_711_680,
        max_br: 800_000,
        max_cpb: 800_000,
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
