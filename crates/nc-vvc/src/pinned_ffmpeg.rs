//! The pinned `FFmpeg` the VVC conformance tests decode with, shared by the
//! unit tests (`angular_conformance`) and `tests/conformance.rs` (included
//! there by path). Test-only.

use std::process::Command;

/// `FFmpeg` releases before this mis-decode VVC ALF streams: they read
/// uninitialised `CC-ALF` indexes whenever ALF is on (`FFmpeg` ticket 11325).
const MIN_FFMPEG_VERSION: (u32, u32) = (8, 0);

/// The `FFmpeg` to check against: `NC_FFMPEG`, the pinned decode-only build
/// (`anchors/<arch>/bin/ffmpeg-decode`, `build_anchors.sh ffmpeg`), never one
/// on `PATH`. Panics unless it reports a release >= [`MIN_FFMPEG_VERSION`].
pub(crate) fn pinned_ffmpeg() -> String {
    let ffmpeg = std::env::var("NC_FFMPEG").expect("NC_FFMPEG must point to FFmpeg >= 8.0");
    let banner = Command::new(&ffmpeg)
        .arg("-version")
        .output()
        .unwrap()
        .stdout;
    let banner = String::from_utf8_lossy(&banner);
    let version = banner
        .strip_prefix("ffmpeg version ")
        .map(|v| v.trim_start_matches('n'))
        .and_then(|v| {
            let mut parts = v.split(|c: char| !c.is_ascii_digit());
            Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
        });
    assert!(
        version.is_some_and(|v: (u32, u32)| v >= MIN_FFMPEG_VERSION),
        "{ffmpeg} is not FFmpeg >= 8.0 (older ones mis-decode VVC ALF, ticket 11325): {}",
        banner.lines().next().unwrap_or("")
    );
    ffmpeg
}
