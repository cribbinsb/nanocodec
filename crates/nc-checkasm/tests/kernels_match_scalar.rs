//! `cargo test` entry point: every SIMD kernel this machine supports must
//! match scalar. (The `nc-checkasm` binary runs the same checks with more
//! iterations.)

use nc_checkasm::check;
use nc_dsp::Isa;

const SEED: u64 = 0xC0DE;
const ITERATIONS: usize = 20;

#[test]
fn every_supported_isa_matches_scalar() {
    for isa in Isa::ALL {
        if let Some(report) = check(isa, SEED, ITERATIONS) {
            assert!(
                report.passed(),
                "{} mismatches, first: {}",
                report.failures.len(),
                report.failures[0]
            );
        }
    }
}
