//! VP9 forward probability updates in the compressed header: the
//! `diff_update_prob` syntax with its recentred, remapped sub-exponential
//! code (VP9 §6.3.2-6.3.5; libvpx `vp9_dsubexp.c` / `vp9_subexp.c`,
//! BSD-3-Clause, see `THIRD_PARTY.md`).

use crate::bool_coder::{BoolReader, BoolWriter};

/// Probability of the "update" flag of every diff-updated probability.
pub(crate) const DIFF_UPDATE_PROB: u8 = 252;
const MAX_PROB: i32 = 255;
/// `inv_map_table`: coded index to recentred distance.
const INV_MAP: [u8; 255] = [
    7, 20, 33, 46, 59, 72, 85, 98, 111, 124, 137, 150, 163, 176, 189, 202, 215, 228, 241, 254, 1,
    2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 47, 48, 49, 50, 51, 52, 53, 54,
    55, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 75, 76, 77, 78, 79, 80,
    81, 82, 83, 84, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102, 103, 104,
    105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 125,
    126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 138, 139, 140, 141, 142, 143, 144, 145,
    146, 147, 148, 149, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 164, 165, 166,
    167, 168, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186,
    187, 188, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 242, 243, 244, 245, 246, 247, 248,
    249, 250, 251, 252, 253, 253,
];
/// `decode_uniform` alphabet: 8 bits, the first 65 values in 7.
const UNIFORM_BITS: u32 = 8;
const UNIFORM_SHORT: i32 = (1 << UNIFORM_BITS) - 191;

fn inv_recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v & 1 != 0 {
        m - ((v + 1) >> 1)
    } else {
        m + (v >> 1)
    }
}

/// `inv_remap_prob`: the new probability coded as `delta` from `old`.
fn inv_remap(delta: usize, old: u8) -> u8 {
    let v = i32::from(INV_MAP[delta]);
    let m = i32::from(old) - 1;
    let p = if (m << 1) <= MAX_PROB {
        1 + inv_recenter_nonneg(v, m)
    } else {
        MAX_PROB - inv_recenter_nonneg(v, MAX_PROB - 1 - m)
    };
    p as u8
}

/// `map_table`: recentred distance (minus 1) to coded index, the inverse
/// of the first 254 entries of [`INV_MAP`] (its last entry repeats).
const MAP: [u8; 254] = {
    let mut map = [0; 254];
    let mut delta = 0;
    while delta < map.len() {
        map[INV_MAP[delta] as usize - 1] = delta as u8;
        delta += 1;
    }
    map
};

/// `recenter_nonneg`, the inverse of [`inv_recenter_nonneg`].
fn recenter_nonneg(v: i32, m: i32) -> i32 {
    if v > 2 * m {
        v
    } else if v >= m {
        (v - m) << 1
    } else {
        ((m - v) << 1) - 1
    }
}

/// The code of `new` relative to `old` (the inverse of [`inv_remap`];
/// libvpx `remap_prob`). `new` differs from `old`.
fn remap(new: u8, old: u8) -> usize {
    debug_assert_ne!(new, old, "an unchanged probability has no code");
    let (v, m) = (i32::from(new) - 1, i32::from(old) - 1);
    let distance = if (m << 1) <= MAX_PROB {
        recenter_nonneg(v, m)
    } else {
        recenter_nonneg(MAX_PROB - 1 - v, MAX_PROB - 1 - m)
    };
    usize::from(MAP[distance as usize - 1])
}

/// Bits of the sub-exponential code of `delta` (all literal bits).
fn subexp_bits(delta: usize) -> u32 {
    match delta {
        0..16 => 5,
        16..32 => 6,
        32..64 => 8,
        _ => {
            let v = delta as i32 - 64;
            3 + if v < UNIFORM_SHORT {
                UNIFORM_BITS - 1
            } else {
                UNIFORM_BITS
            }
        }
    }
}

fn write_literal(writer: &mut BoolWriter, value: u32, bits: u32) {
    for bit in (0..bits).rev() {
        writer.write((value >> bit) & 1 != 0, 128);
    }
}

/// `encode_term_subexp`.
fn write_subexp(writer: &mut BoolWriter, delta: usize) {
    let delta = delta as u32;
    match delta {
        0..16 => {
            writer.write(false, 128);
            write_literal(writer, delta, 4);
        }
        16..32 => {
            writer.write(true, 128);
            writer.write(false, 128);
            write_literal(writer, delta - 16, 4);
        }
        32..64 => {
            for bit in [true, true, false] {
                writer.write(bit, 128);
            }
            write_literal(writer, delta - 32, 5);
        }
        _ => {
            for _ in 0..3 {
                writer.write(true, 128);
            }
            let v = delta as i32 - 64;
            if v < UNIFORM_SHORT {
                write_literal(writer, v as u32, UNIFORM_BITS - 1);
            } else {
                let rest = v - UNIFORM_SHORT;
                write_literal(
                    writer,
                    (UNIFORM_SHORT + (rest >> 1)) as u32,
                    UNIFORM_BITS - 1,
                );
                writer.write(rest & 1 != 0, 128);
            }
        }
    }
}

/// `decode_term_subexp`.
fn read_subexp(reader: &mut BoolReader<'_>) -> usize {
    if !reader.read(128) {
        return reader.read_literal(4) as usize;
    }
    if !reader.read(128) {
        return reader.read_literal(4) as usize + 16;
    }
    if !reader.read(128) {
        return reader.read_literal(5) as usize + 32;
    }
    let v = reader.read_literal(UNIFORM_BITS - 1) as i32;
    let v = if v < UNIFORM_SHORT {
        v
    } else {
        (v << 1) - UNIFORM_SHORT + i32::from(reader.read(128))
    };
    v as usize + 64
}

/// Write one diff-update: the flag, and the new probability if it differs.
pub(crate) fn write_update(writer: &mut BoolWriter, old: u8, new: u8) {
    writer.write(new != old, DIFF_UPDATE_PROB);
    if new != old {
        write_subexp(writer, remap(new, old));
    }
}

/// `vp9_diff_update_prob`.
pub(crate) fn read_update(reader: &mut BoolReader<'_>, prob: &mut u8) {
    if reader.read(DIFF_UPDATE_PROB) {
        let delta = read_subexp(reader);
        *prob = inv_remap(delta.min(INV_MAP.len() - 1), *prob);
    }
}

/// Bits of updating `old` to `new` beyond not updating it.
pub(crate) fn update_cost(old: u8, new: u8) -> f32 {
    crate::tree::bit_cost(true, DIFF_UPDATE_PROB) - crate::tree::bit_cost(false, DIFF_UPDATE_PROB)
        + subexp_bits(remap(new, old)) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remap_inverts_inv_remap() {
        for old in 1..=255u8 {
            for new in (1..=255u8).filter(|&new| new != old) {
                assert_eq!(inv_remap(remap(new, old), old), new, "{old} -> {new}");
            }
        }
    }

    #[test]
    fn every_update_round_trips() {
        let pairs: Vec<(u8, u8)> = (1..=255u8)
            .step_by(7)
            .flat_map(|old| (1..=255u8).step_by(5).map(move |new| (old, new)))
            .collect();
        let mut writer = BoolWriter::new();
        for &(old, new) in &pairs {
            write_update(&mut writer, old, new);
        }
        let packet = writer.finish();
        let mut reader = BoolReader::new(&packet).unwrap();
        for &(old, new) in &pairs {
            let mut prob = old;
            read_update(&mut reader, &mut prob);
            assert_eq!(prob, new, "{old} -> {new}");
        }
    }
}
