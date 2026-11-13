//! The VP9 backend through the shared engine, and malformed streams.
//!
//! One test binary for the crate (`autotests = false`, one `[[test]]` in
//! Cargo.toml): a binary per file spent most of the test build linking.
//! `tools/ci/check_test_modules.py` fails if a `tests/*.rs` file is not a
//! module here, since Cargo would otherwise skip it silently.
mod malformed;
mod roundtrip;
