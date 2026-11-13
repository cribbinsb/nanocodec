//! `av1.restoration`: AV1 Wiener loop restoration. The stream's restored
//! pictures, as our decoder reconstructs them, equal the encoder's
//! reconstruction (unit syntax placed before each superblock's partition,
//! filters applied after CDEF), and restoration changes the pictures.
use std::{
    fs,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

/// Neither dimension a multiple of 64 (partial units and superblocks).
const WIDTH: usize = 200;
const HEIGHT: usize = 136;
const FRAMES: usize = 4;

struct Scratch(PathBuf);
impl Scratch {
    /// A smooth moving pattern with noise: restoration finds filters that
    /// pay at a coarse quantizer.
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "nc-cli-restoration-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir(&path).unwrap();
        let mut state = 1u32;
        let mut bytes = Vec::new();
        for frame in 0..FRAMES {
            for (w, h) in [
                (WIDTH, HEIGHT),
                (WIDTH / 2, HEIGHT / 2),
                (WIDTH / 2, HEIGHT / 2),
            ] {
                for y in 0..h {
                    for x in 0..w {
                        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                        let smooth = 128.0
                            + 70.0
                                * (((x + 2 * frame) as f64) / 7.0).sin()
                                * ((y as f64) / 11.0).cos();
                        bytes.push((smooth as i32 + (state >> 28) as i32 - 8).clamp(0, 255) as u8);
                    }
                }
            }
        }
        fs::write(path.join("source.yuv"), bytes).unwrap();
        Self(path)
    }

    /// Encode with `restoration` into `name`.ivf, keeping the reconstruction
    /// in `name`.yuv.
    fn encode(&self, name: &str, restoration: u8) -> (PathBuf, PathBuf) {
        let (stream, recon) = (
            self.0.join(format!("{name}.ivf")),
            self.0.join(format!("{name}.yuv")),
        );
        let run = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
            .args(["encode", "--codec", "av1", "--input"])
            .arg(self.0.join("source.yuv"))
            .arg("--output")
            .arg(&stream)
            .arg("--recon")
            .arg(&recon)
            .args([
                "--width",
                &WIDTH.to_string(),
                "--height",
                &HEIGHT.to_string(),
            ])
            .args([
                "--fps",
                "30",
                "--frames",
                &FRAMES.to_string(),
                "--qp",
                "160",
            ])
            .args(["--param", &format!("av1.restoration={restoration}")])
            .args(["--param", "av1.restoration_row_step=1"])
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        (stream, recon)
    }

    fn decode(&self, stream: &PathBuf) -> Vec<u8> {
        let output = self.0.join("decoded.yuv");
        let run = Command::new(env!("CARGO_BIN_EXE_nanocodec"))
            .args(["decode", "--codec", "av1", "--input"])
            .arg(stream)
            .arg("--output")
            .arg(&output)
            .output()
            .unwrap();
        assert!(
            run.status.success(),
            "{}",
            String::from_utf8_lossy(&run.stderr)
        );
        fs::read(output).unwrap()
    }
}
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[test]
fn restored_stream_decodes_to_the_encoder_reconstruction() {
    let scratch = Scratch::new();
    let (plain, plain_recon) = scratch.encode("plain", 0);
    let (restored, restored_recon) = scratch.encode("restored", 1);
    let restored_recon = fs::read(restored_recon).unwrap();
    assert_eq!(scratch.decode(&restored), restored_recon);
    assert_eq!(scratch.decode(&plain), fs::read(plain_recon).unwrap());
    assert_ne!(
        restored_recon,
        fs::read(scratch.0.join("plain.yuv")).unwrap(),
        "restoration filtered nothing"
    );
}
