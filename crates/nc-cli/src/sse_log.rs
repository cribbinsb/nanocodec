//! `--sse-log`: the squared error of every shown picture against a
//! reference clip (`--sse-reference`), so the evaluation harness scores
//! PSNR without ever reading the reconstruction itself
//! (`eval/nceval/metrics.py`, `EncoderPsnrMetric`).
//!
//! One CSV row per picture `--display-recon` would get (the picture shown at
//! each input frame's time; a simulated loss shows nothing and has no row):
//! `frame,sse_y,sse_u,sse_v`, each the exact integer sum over its plane.
//! The reference is raw I420 at the stream size, read from its first frame.

use std::io::Write;
use std::path::Path;

use nc_common::{Picture, PlaneIndex, yuv};
use nc_dsp::Dsp;

use crate::stdio;

/// The open reference clip and log.
pub(crate) struct SseLog {
    reference: stdio::Input,
    /// The current reference frame, sized like the last shown picture.
    frame: Option<Picture>,
    out: stdio::Output,
    dsp: Dsp,
}

impl SseLog {
    pub(crate) fn create(reference: &Path, log: &Path) -> Result<Self, String> {
        let reference = stdio::open_raw(reference)?;
        let mut out = stdio::create(log)?;
        writeln!(out, "frame,sse_y,sse_u,sse_v").map_err(|e| e.to_string())?;
        Ok(Self {
            reference,
            frame: None,
            out,
            dsp: Dsp::new(),
        })
    }

    /// Log the error of `shown`, the picture shown at input frame `index`,
    /// against the next reference frame.
    pub(crate) fn write(&mut self, index: u32, shown: &Picture) -> Result<(), String> {
        let size = (shown.width(), shown.height());
        if self
            .frame
            .as_ref()
            .is_none_or(|f| (f.width(), f.height()) != size)
        {
            self.frame = Some(Picture::new(size.0, size.1, 0).map_err(|e| e.to_string())?);
        }
        let frame = self.frame.as_mut().expect("allocated above");
        if !yuv::read_frame(&mut self.reference, frame).map_err(|e| e.to_string())? {
            return Err(format!("--sse-reference ended before frame {index}"));
        }
        let [y, u, v] = PlaneIndex::ALL.map(|plane| {
            self.dsp
                .sse(&frame.plane(plane).view(), &shown.plane(plane).view())
        });
        writeln!(self.out, "{index},{y},{u},{v}").map_err(|e| e.to_string())
    }

    pub(crate) fn finish(mut self) -> Result<(), String> {
        self.out.flush().map_err(|e| e.to_string())
    }
}
