//! Coefficient-syntax vectors for the original-AVM oracle
//! (`tools/codegen/check_av2_coeff.py`).
//!
//! Prints tiles of Y/U/V transform blocks for every size, prediction kind,
//! quantizer category and adaptation mode. Each block line carries the
//! neighbour inputs, our derived contexts, the entropy context and the
//! coefficients; each tile line carries the bytes our writer produced. The
//! oracle derives contexts with AVM `get_txb_ctx`, decodes the bytes with
//! AVM `av2_read_sig_txtype` / `av2_read_coeffs_txb` and compares.
//! Magnitudes stay below 2^16 so the oracle can undo AVM's dequantization.
use nc_av2::coeff::{self, Context, TxSize};
use nc_av2::entropy::Writer;
use std::fmt::Write as _;

/// One qindex per CDF quantizer category, plus lossless.
const QINDICES: [u8; 5] = [0, 60, 120, 170, 230];
/// Neighbour units printed per side: enough for TX64.
const UNITS: usize = 16;
/// Largest magnitude the oracle's dequantizer inverts exactly.
const MAX_ORACLE_LEVEL: u32 = 1 << 16;
/// Block groups (Y, U, V) per size and prediction kind in each tile.
const GROUPS: usize = 4;

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
    fn below(&mut self, n: u32) -> u32 {
        self.next() % n
    }
}

/// Valid entropy-context bytes, zero half of the time.
fn neighbours(rng: &mut Lcg) -> [u8; UNITS] {
    std::array::from_fn(|_| {
        if rng.below(2) == 0 {
            0
        } else {
            (rng.below(8) | rng.below(3) << 3) as u8
        }
    })
}

fn coefficients(rng: &mut Lcg, tx: TxSize) -> Vec<i32> {
    let len = tx.coded_len();
    let eob = match rng.below(5) {
        0 => 0,
        1 => 1 + rng.below(3) as usize,
        2 => len,
        _ => 1 + rng.below(len as u32) as usize,
    };
    let density = 1 + rng.below(4);
    let mut block = vec![0; len];
    // Raster-order prefixes still exercise every scan position and context.
    for value in &mut block[..eob] {
        let magnitude = match rng.below(8) {
            0..=2 if rng.below(density) == 0 => 0,
            0..=4 => 1 + rng.below(3),
            5 | 6 => rng.below(40),
            _ => rng.below(MAX_ORACLE_LEVEL),
        } as i32;
        *value = if rng.below(2) == 0 {
            magnitude
        } else {
            -magnitude
        };
    }
    block
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = Lcg(2026);
    let mut out = String::new();
    for qindex in QINDICES {
        for adaptive in [false, true] {
            let sizes: &[TxSize] = if qindex == 0 {
                &[TxSize::Tx4]
            } else {
                &TxSize::ALL
            };
            let mut writer = Writer::new();
            if adaptive {
                writer.enable_adaptation(qindex);
            }
            let mut lines = String::new();
            for &luma_tx in sizes {
                for is_inter in [false, true] {
                    for _ in 0..GROUPS {
                        let mut u_coded = false;
                        for plane in 0..3u8 {
                            let tx = if plane == 0 {
                                luma_tx
                            } else {
                                luma_tx.min(TxSize::Tx32)
                            };
                            let block_log2 = tx.log2() + rng.below(2);
                            let (above, left) = (neighbours(&mut rng), neighbours(&mut rng));
                            let ctx = Context::derive(
                                plane, block_log2, tx, is_inter, &above, &left, u_coded,
                            )?;
                            let block = coefficients(&mut rng, tx);
                            coeff::write(&mut writer, &block, tx, ctx, qindex)?;
                            if plane == 1 {
                                u_coded = block.iter().any(|&c| c != 0);
                            }
                            write!(
                                lines,
                                "B {} {plane} {} {block_log2} {} {}",
                                tx.log2(),
                                u8::from(is_inter),
                                ctx.skip,
                                ctx.dc_sign
                            )?;
                            for value in above.iter().chain(&left) {
                                write!(lines, " {value}")?;
                            }
                            write!(lines, " {}", coeff::entropy_context(&block))?;
                            for value in &block {
                                write!(lines, " {value}")?;
                            }
                            lines.push('\n');
                        }
                    }
                }
            }
            let packet = writer.finish();
            write!(out, "T {qindex} {} {}", u8::from(adaptive), packet.len())?;
            for byte in &packet {
                write!(out, " {byte}")?;
            }
            out.push('\n');
            out.push_str(&lines);
            out.push_str("E\n");
        }
    }
    print!("{out}");
    Ok(())
}
