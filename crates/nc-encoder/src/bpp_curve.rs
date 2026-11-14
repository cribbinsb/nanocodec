//! A setting that depends on the stream's bits per pixel (target bits per
//! frame over luma samples): two registered points, interpolated linearly
//! in log2 bits per pixel and held flat outside them.
//!
//! Used where a rate-control lever should act differently on starved and
//! on generous streams (`skip_budget.rs` floor, `quality_cap.rs` noise
//! floor clamp): at generous rates a lever that stops spending may only
//! trim very fine refinement, at starved rates every bit counts for what
//! changes.

/// Bits per pixel of the starved point: 300 kbps at 854x480, 15 fps is
/// 0.049; 150 kbps at 720p30 is 0.005.
pub(crate) const STARVED_BPP: f64 = 0.02;
/// Bits per pixel of the generous point: 1 Mbps at 854x480, 15 fps is 0.16.
pub(crate) const GENEROUS_BPP: f64 = 0.15;

/// `starved` at or below [`STARVED_BPP`], `generous` at or above
/// [`GENEROUS_BPP`], log-linear between.
pub(crate) fn interpolate(bpp: f64, starved: f64, generous: f64) -> f64 {
    let t = ((bpp.max(f64::MIN_POSITIVE) / STARVED_BPP).log2()
        / (GENEROUS_BPP / STARVED_BPP).log2())
    .clamp(0., 1.);
    starved * (1. - t) + generous * t
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ends_hold_and_the_middle_is_log_linear() {
        assert_eq!(interpolate(0.001, 0.5, 0.1), 0.5);
        assert_eq!(interpolate(1.0, 0.5, 0.1), 0.1);
        let middle = (STARVED_BPP * GENEROUS_BPP).sqrt();
        assert!((interpolate(middle, 0.5, 0.1) - 0.3).abs() < 1e-12);
    }
}
