//! Exact pricing checks against the reconstructing syntax walk.
use super::*;

/// Keep the old reconstructing path selectable in the same test binary.
#[derive(Default)]
struct ReferenceCost(BitCost);
impl Bins for ReferenceCost {
    fn bin(&mut self, c: &mut Context, value: bool) -> Result<bool> {
        self.0.bin(c, value)
    }
    fn bypass(&mut self, value: u32, bits: u32) -> Result<u32> {
        self.0.bypass(value, bits)
    }
}

fn random(state: &mut u64) -> u32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state as u32
}

#[test]
fn source_pricing_matches_reconstructed_cost_and_every_context() {
    let mut rng = 0xd46e_5c71_482a_b931;
    for width in [1, 2, 4, 8, 16, 32, 64] {
        for height in [1, 2, 4, 8, 16, 32, 64] {
            if !is_coded_size(width, height) {
                continue;
            }
            let count = coded_side(width) * coded_side(height);
            for dependent in [false, true] {
                for ch in [0, 1] {
                    let mut reference = CoeffContexts::new(random(&mut rng) % 64, ch != 0);
                    reference.dependent = dependent;
                    for pattern in 0_usize..24 {
                        let mut source = vec![0; count];
                        for value in &mut source {
                            let sample = random(&mut rng);
                            *value = match pattern % 6 {
                                0 => 0,
                                1 if !sample.is_multiple_of(8) => 0,
                                2 if !sample.is_multiple_of(3) => 0,
                                3 => (sample % 3) as i32 - 1,
                                4 => (sample % 11) as i32 - 5,
                                _ => (sample % 65535) as i32 - 32767,
                            };
                        }
                        // A nonzero block is the encoder's contract. The DC-only
                        // arm also exercises large grids with one coded group.
                        source[0] = 1;
                        if pattern.is_multiple_of(6) {
                            // Also put the last level at a later group's first
                            // position: its halo is zero even when earlier
                            // groups contain levels and need their own halos.
                            let group = (pattern / 6) * (count / GROUP_SIZE) / 4;
                            let scan =
                                crate::scan::rect_positions(coded_side(width), coded_side(height));
                            source[scan[group * GROUP_SIZE]] = 7;
                        }
                        let initial = [0., 123.125, 16_777_216.][pattern % 3];
                        let mut expected = ReferenceCost(BitCost(initial));
                        let mut actual = BitCost(initial);
                        let mut candidate = reference.clone();
                        emit_rect(&mut expected, &mut reference, &source, (width, height), ch);
                        emit_rect(&mut actual, &mut candidate, &source, (width, height), ch);
                        assert_eq!(
                            actual.0.to_bits(),
                            expected.0.0.to_bits(),
                            "{width}x{height} ch={ch} dependent={dependent} pattern={pattern}"
                        );
                        assert_eq!(
                            candidate, reference,
                            "{width}x{height} ch={ch} dependent={dependent} pattern={pattern}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn every_source_template_neighbour_precedes_its_coding() {
    for width in [1, 2, 4, 8, 16, 32] {
        for height in [1, 2, 4, 8, 16, 32] {
            if !is_coded_size(width, height) {
                continue;
            }
            let scan = crate::scan::rect_indices(width, height);
            for y in 0..height {
                for x in 0..width {
                    for (dx, dy) in TEMPLATE {
                        if x + dx < width && y + dy < height {
                            assert!(scan[(y + dy) * width + x + dx] > scan[y * width + x]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "fixed-workload instruction/cycle probe; select old grid with NC_COEFF_PRICE_REFERENCE"]
fn pricing_cost_probe() {
    use std::hint::black_box;
    let reference = std::env::var_os("NC_COEFF_PRICE_REFERENCE").is_some();
    let iterations = std::env::var("NC_COEFF_PRICE_ITERATIONS")
        .map_or(1000, |s| s.parse().expect("integer iterations"));
    let mut blocks = Vec::new();
    for side in [4, 8, 16, 32] {
        for span in [1, 4, 16, side * side] {
            let mut levels = vec![0; side * side];
            for &p in &crate::scan::positions(side)[..span] {
                levels[p] = (p % 7 + 1) as i32;
            }
            blocks.push((side, levels));
        }
    }
    if let Ok(case) = std::env::var("NC_COEFF_PRICE_CASE") {
        let case: usize = case.parse().expect("integer case index");
        let selected = blocks.remove(case);
        blocks.clear();
        blocks.push(selected);
    }
    let seed = CoeffContexts::new(37, false);
    let mut checksum = 0_u64;
    for _ in 0..iterations {
        for (side, levels) in &blocks {
            let mut contexts = seed.clone();
            let cost = if reference {
                let mut cost = ReferenceCost::default();
                emit(&mut cost, &mut contexts, black_box(levels), *side, 0);
                cost.0.0
            } else {
                let mut cost = BitCost::default();
                emit(&mut cost, &mut contexts, black_box(levels), *side, 0);
                cost.0
            };
            checksum = checksum.wrapping_add(u64::from(black_box(cost.to_bits())));
            black_box(contexts);
        }
    }
    println!("calls={} checksum={checksum}", iterations * blocks.len());
}
