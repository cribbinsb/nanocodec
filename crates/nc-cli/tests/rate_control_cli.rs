//! Bitrate-mode CLI controls and actual per-frame QP bounds across all codecs.
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nc-cli-rc-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn path(path: &Path) -> String {
    path.to_str().unwrap().into()
}
#[test]
fn bitrate_without_qp_logs_dynamic_actual_qp_and_decodes_exactly() {
    let scratch = Scratch::new();
    let input = scratch.0.join("source.yuv");
    let bytes: Vec<u8> = (0..30 * 96 * 64 * 3 / 2)
        .map(|i| ((i * 29 + (i / 96) * 11 + i / (96 * 64) * 23) % 251) as u8)
        .collect();
    fs::write(&input, bytes).unwrap();
    for codec in ["h264", "hevc", "av1"] {
        let stream = scratch.0.join(format!(
            "{codec}.{}",
            if codec == "av1" { "ivf" } else { "bin" }
        ));
        let log = scratch.0.join(format!("{codec}.csv"));
        let recon = scratch.0.join(format!("{codec}.recon.yuv"));
        let decoded = scratch.0.join(format!("{codec}.decoded.yuv"));
        let (min, max) = if codec == "av1" { (100, 220) } else { (10, 45) };
        let result = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
            .args([
                "encode".into(),
                "--codec".into(),
                codec.into(),
                "--input".into(),
                path(&input),
                "--output".into(),
                path(&stream),
                "--recon".into(),
                path(&recon),
                "--width".into(),
                "96".into(),
                "--height".into(),
                "64".into(),
                "--fps".into(),
                "15".into(),
                "--frames".into(),
                "30".into(),
                "--intra-period".into(),
                "10".into(),
                "--bitrate".into(),
                "25".into(),
                "--buffer-ms".into(),
                "500".into(),
                "--peak-bitrate".into(),
                "50".into(),
                "--min-qp".into(),
                min.to_string(),
                "--max-qp".into(),
                max.to_string(),
                "--rate-log".into(),
                path(&log),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{codec}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        let text = fs::read_to_string(&log).unwrap();
        let rows: Vec<_> = text
            .lines()
            .skip(1)
            .map(|line| line.split(',').collect::<Vec<_>>())
            .collect();
        assert_eq!(rows.len(), 30);
        let quantizers: Vec<u32> = rows.iter().map(|row| row[3].parse().unwrap()).collect();
        assert!(
            quantizers.iter().all(|&q| (min..=max).contains(&q)),
            "{codec}: {quantizers:?}"
        );
        assert!(
            quantizers.iter().any(|&q| q != quantizers[0]),
            "controller must adapt {codec}"
        );
        assert!(
            rows.iter().all(|row| row[2] == row[3]),
            "RC metadata and actual QP differ {codec}"
        );
        assert!(rows.iter().all(|row| row[9] == "25000"));
        let result = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
            .args([
                "decode",
                "--codec",
                codec,
                "--input",
                stream.to_str().unwrap(),
                "--output",
                decoded.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{codec}: {}",
            String::from_utf8_lossy(&result.stderr)
        );
        assert_eq!(fs::read(decoded).unwrap(), fs::read(recon).unwrap());
    }
}
