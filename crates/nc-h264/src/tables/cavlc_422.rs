//! The CAVLC code tables of 4:2:2 chroma DC blocks (eight coefficients),
//! which the generated tables (`cavlc.rs`, from openh264's 4:2:0-only
//! encoder) lack: H.264 Table 9-5's `nC == −2` column and Table 9-9b,
//! transcribed from the Recommendation (and checked against JM's `vlc.c`).
//! Entries are `(code value, code length)` as in `cavlc.rs`; a length of 0
//! marks a combination that has no code.

/// Table 9-5 at `nC == −2`, indexed `[TotalCoeff][TrailingOnes]`.
pub const COEFF_TOKEN_CHROMA_DC_422: [[(u8, u8); 4]; 9] = [
    [(1, 1), (0, 0), (0, 0), (0, 0)],
    [(15, 7), (1, 2), (0, 0), (0, 0)],
    [(14, 7), (13, 7), (1, 3), (0, 0)],
    [(7, 9), (12, 7), (11, 7), (1, 5)],
    [(6, 9), (5, 9), (10, 7), (1, 6)],
    [(7, 10), (6, 10), (4, 9), (9, 7)],
    [(7, 11), (6, 11), (5, 10), (8, 7)],
    [(7, 12), (6, 12), (5, 11), (4, 10)],
    [(7, 13), (5, 12), (4, 12), (4, 11)],
];

/// Table 9-9b, `total_zeros` of 4:2:2 chroma DC blocks, indexed
/// `[TotalCoeff - 1][total_zeros]`.
pub const TOTAL_ZEROS_CHROMA_DC_422: [[(u8, u8); 8]; 7] = [
    [
        (1, 1),
        (2, 3),
        (3, 3),
        (2, 4),
        (3, 4),
        (1, 4),
        (1, 5),
        (0, 5),
    ],
    [
        (0, 3),
        (1, 2),
        (1, 3),
        (4, 3),
        (5, 3),
        (6, 3),
        (7, 3),
        (0, 0),
    ],
    [
        (0, 3),
        (1, 3),
        (1, 2),
        (2, 2),
        (6, 3),
        (7, 3),
        (0, 0),
        (0, 0),
    ],
    [
        (6, 3),
        (0, 2),
        (1, 2),
        (2, 2),
        (7, 3),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    [
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    [
        (0, 2),
        (1, 2),
        (1, 1),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
    [
        (0, 1),
        (1, 1),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
        (0, 0),
    ],
];
