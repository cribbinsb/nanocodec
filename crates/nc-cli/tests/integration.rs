//! End-to-end CLI runs: decoding, rate control and film grain.
//!
//! One test binary for the crate (`autotests = false`, one `[[test]]` in
//! Cargo.toml): a binary per file spent most of the test build linking.
//! `tools/ci/check_test_modules.py` fails if a `tests/*.rs` file is not a
//! module here, since Cargo would otherwise skip it silently.
mod decode_cli;
mod film_grain_cli;
mod frame_hold;
mod keyframe_cli;
mod rate_control_cli;
mod restoration_cli;
mod sse_log_cli;
mod static_recovery;
mod stdio_cli;
