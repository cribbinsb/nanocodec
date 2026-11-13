//! Kernel bit-exactness: every SIMD kernel against scalar, and the AV1
//! kernels against libaom golden hashes.
//!
//! One test binary for the crate (`autotests = false`, one `[[test]]` in
//! Cargo.toml): a binary per file spent most of the test build linking.
//! `tools/ci/check_test_modules.py` fails if a `tests/*.rs` file is not a
//! module here, since Cargo would otherwise skip it silently.
mod av1_block_intra_golden;
mod av1_film_grain_golden;
mod av1_inverse_golden;
mod av1_mc_golden;
mod av1_restoration_golden;
mod kernels_match_scalar;
