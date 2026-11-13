//! End-to-end native CLI default/fast stream decoding and error handling.
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};
struct Scratch(PathBuf);
impl Scratch {
    fn new() -> Self {
        let name = format!(
            "nc-cli-decode-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let path = std::env::temp_dir().join(name);
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn run(args: &[String]) {
    let output = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
        .args(args)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "args={args:?}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}
fn path(path: &Path) -> String {
    path.to_str().unwrap().into()
}
#[test]
fn all_codecs_default_fast_key_p_and_repeated_headers_match_encoder_recon() {
    let scratch = Scratch::new();
    let dir = &scratch.0;
    let input = dir.join("source.yuv");
    let frame_bytes = 64 * 48 * 3 / 2;
    let bytes: Vec<u8> = (0..8 * frame_bytes)
        .map(|i| ((i * 13 + i / frame_bytes * 19 + (i / 64) * 7) % 251) as u8)
        .collect();
    fs::write(&input, bytes).unwrap();
    for codec in ["h264", "hevc", "av1"] {
        for fast in [false, true] {
            let name = format!("{codec}-{fast}");
            let stream = dir.join(format!(
                "{name}.{}",
                if codec == "av1" { "ivf" } else { "bin" }
            ));
            let recon = dir.join(format!("{name}.encoder.yuv"));
            let decoded = dir.join(format!("{name}.decoder.yuv"));
            let mut args = vec![
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
                "64".into(),
                "--height".into(),
                "48".into(),
                "--fps".into(),
                "30000/1001".into(),
                "--qp".into(),
                if codec == "av1" { "190" } else { "38" }.into(),
                "--frames".into(),
                "8".into(),
                "--intra-period".into(),
                "3".into(),
            ];
            if fast {
                if codec == "av1" {
                    for setting in [
                        "av1.pre_tx_skip=1",
                        "av1.pre_tx_skip_score=70",
                        "av1.pre_tx_skip_energy=30",
                        "av1.transform_rd_skip=2",
                        "av1.transform_rd_skip_chroma=1",
                        "decision.subpel_walk=2",
                    ] {
                        args.extend(["--param".into(), setting.into()]);
                    }
                } else {
                    args.extend([
                        "--preset".into(),
                        if codec == "hevc" {
                            "hevc-fast"
                        } else {
                            "h264-fast"
                        }
                        .into(),
                    ]);
                }
            }
            run(&args);
            // IVF timestamps need not be evenly spaced; decoding is packet order.
            if codec == "av1" {
                let mut ivf = fs::read(&stream).unwrap();
                let mut pos = 32;
                let mut timestamp = 11u64;
                while pos < ivf.len() {
                    let size = u32::from_le_bytes(ivf[pos..pos + 4].try_into().unwrap()) as usize;
                    ivf[pos + 4..pos + 12].copy_from_slice(&timestamp.to_le_bytes());
                    timestamp = timestamp * 2 + 1;
                    pos += 12 + size;
                }
                fs::write(&stream, ivf).unwrap();
            }
            run(&[
                "decode".into(),
                "--codec".into(),
                codec.into(),
                "--input".into(),
                path(&stream),
                "--output".into(),
                path(&decoded),
            ]);
            assert_eq!(fs::read(&recon).unwrap().len(), 8 * frame_bytes);
            assert_eq!(
                fs::read(&decoded).unwrap(),
                fs::read(&recon).unwrap(),
                "{name}"
            );
        }
    }
}
#[test]
fn malformed_input_and_same_output_path_fail_without_panicking_or_truncating_input() {
    let scratch = Scratch::new();
    let input = scratch.0.join("bad.ivf");
    fs::write(&input, b"broken").unwrap();
    for output in [input.clone(), scratch.0.join("output.yuv")] {
        let result = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
            .args([
                "decode",
                "--codec",
                "av1",
                "--input",
                input.to_str().unwrap(),
                "--output",
                output.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(!result.status.success());
        assert!(!String::from_utf8_lossy(&result.stderr).contains("panicked"));
        assert_eq!(fs::read(&input).unwrap(), b"broken");
    }
}
