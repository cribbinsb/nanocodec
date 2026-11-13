//! Replay HEVC trace geometry and thresholds on generated samples. Pipe gzip
//! output to `--filter-runs -`; no video or compression dependency is needed.
//! Single mode submits one uniform segment per call, not the old edge API.
//! Single chroma segments currently use the scalar two-line tail: a batch gain
//! can include coalescing into SIMD, not just less dispatch. Samples reset per
//! phase; this does not reproduce vertical-to-horizontal picture dependencies.
//! Exempt sample restoration belongs to the decoder and is not timed here.

use nc_common::PlaneViewMut;
use nc_dsp::{Dsp, EdgeRun, HevcChromaFilter, HevcLumaFilter, Isa};
use std::time::{Duration, Instant};

#[derive(Default)]
struct Phase {
    width: usize,
    height: usize,
    vertical: bool,
    chroma: bool,
    luma: Vec<EdgeRun<HevcLumaFilter>>,
    uv: [Vec<EdgeRun<HevcChromaFilter>>; 2],
}

#[derive(Default)]
struct Report {
    elapsed: Duration,
    checksum: u64,
    phases: usize,
    runs: usize,
    lines: usize,
}

pub(crate) fn replay(data: &[u8], batch: bool, isa: Option<Isa>) -> Result<(), String> {
    let text = std::str::from_utf8(data).map_err(|e| e.to_string())?;
    let dsp = isa.map_or_else(
        || Ok(Dsp::new()),
        |isa| Dsp::with_isa(isa).ok_or("unsupported ISA"),
    )?;
    let mut geometry = None;
    let mut phase: Option<Phase> = None;
    let mut report = Report::default();
    for line in text.lines() {
        let mut fields = line.split_whitespace();
        let Some(kind) = fields.next() else { continue };
        if kind == "F" {
            flush(&mut phase, &dsp, batch, &mut report)?;
            fields.next().ok_or("missing picture order count")?;
            let width = number(&mut fields)?;
            let height = number(&mut fields)?;
            let ctb = number(&mut fields)?;
            if width == 0 || height == 0 || !matches!(ctb, 16 | 32 | 64) {
                return Err("invalid HEVC picture geometry".into());
            }
            geometry = Some((width, height));
            continue;
        }
        let plane = match kind {
            "L" => 0,
            "C0" => 1,
            "C1" => 2,
            _ => return Err(format!("unknown HEVC record {kind}")),
        };
        let direction = number(&mut fields)?;
        if direction > 1 {
            return Err("invalid edge direction".into());
        }
        let vertical = direction == 1;
        let chroma = plane != 0;
        let (width, height) = geometry.ok_or("edge before picture header")?;
        if phase
            .as_ref()
            .is_some_and(|p| p.vertical != vertical || p.chroma != chroma)
        {
            flush(&mut phase, &dsp, batch, &mut report)?;
        }
        let active = phase.get_or_insert_with(|| Phase {
            width: if chroma { width.div_ceil(2) } else { width },
            height: if chroma { height.div_ceil(2) } else { height },
            vertical,
            chroma,
            ..Phase::default()
        });
        let x = number(&mut fields)?;
        let y = number(&mut fields)?;
        let lines = number(&mut fields)?;
        let unit = if chroma { 2 } else { 4 };
        if lines == 0 || lines > 8 || !lines.is_multiple_of(unit) {
            return Err("invalid HEVC segment count".into());
        }
        let beta = if chroma { 0 } else { threshold(&mut fields)? };
        let mut tc = [0; 4];
        for value in &mut tc[..if chroma { 4 } else { 2 }] {
            *value = threshold(&mut fields)?;
        }
        // Traces include decoder-owned PCM/exempt flags after these fields.
        // Generated samples are non-PCM, so only the filter calls are replayed.
        for (segment, &tc) in tc[..lines / unit].iter().enumerate() {
            let x = x + if vertical { 0 } else { segment * unit };
            let y = y + if vertical { segment * unit } else { 0 };
            if chroma {
                active.uv[plane - 1].push(EdgeRun {
                    x,
                    y,
                    lines: unit,
                    filter: HevcChromaFilter { tc },
                });
            } else {
                active.luma.push(EdgeRun {
                    x,
                    y,
                    lines: unit,
                    filter: HevcLumaFilter { beta, tc },
                });
            }
        }
    }
    flush(&mut phase, &dsp, batch, &mut report)?;
    println!(
        "hevc-filter-runs isa={} dispatch={} phases={} runs={} lines={} elapsed_ns={} checksum={:016x}",
        dsp.isa(),
        if batch { "batch" } else { "single-segment" },
        report.phases,
        report.runs,
        report.lines,
        report.elapsed.as_nanos(),
        report.checksum
    );
    Ok(())
}

fn number<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Result<usize, String> {
    fields
        .next()
        .ok_or("missing trace field")?
        .parse()
        .map_err(|_| "invalid trace integer".into())
}

fn threshold<'a>(fields: &mut impl Iterator<Item = &'a str>) -> Result<u8, String> {
    u8::try_from(number(fields)?).map_err(|_| "invalid trace threshold".into())
}

fn flush(
    phase: &mut Option<Phase>,
    dsp: &Dsp,
    batch: bool,
    report: &mut Report,
) -> Result<(), String> {
    let Some(p) = phase.take() else { return Ok(()) };
    if p.chroma {
        for runs in &p.uv {
            measure(&p, runs, report, |view, runs| {
                if batch {
                    dsp.hevc_deblock_chroma_runs(view, p.vertical, runs);
                } else {
                    for run in runs {
                        dsp.hevc_deblock_chroma_runs(view, p.vertical, std::slice::from_ref(run));
                    }
                }
            })?;
        }
    } else {
        measure(&p, &p.luma, report, |view, runs| {
            if batch {
                dsp.hevc_deblock_luma_runs(view, p.vertical, runs);
            } else {
                for run in runs {
                    dsp.hevc_deblock_luma_runs(view, p.vertical, std::slice::from_ref(run));
                }
            }
        })?;
    }
    Ok(())
}

fn measure<F: Copy>(
    phase: &Phase,
    runs: &[EdgeRun<F>],
    report: &mut Report,
    apply: impl FnOnce(&mut PlaneViewMut<'_>, &[EdgeRun<F>]),
) -> Result<(), String> {
    if runs.is_empty() {
        return Ok(());
    }
    // Scratch padding accommodates final filter groups. This is not a claim
    // about the decoder's storage layout (chroma may have fewer padded rows).
    // Only a band's touched rows are generated, keeping replay memory bounded.
    let width = phase.width.next_multiple_of(8);
    let plane_height = phase.height.next_multiple_of(8);
    let first_row = runs
        .iter()
        .map(|r| {
            if phase.vertical {
                r.y
            } else {
                r.y.saturating_sub(4)
            }
        })
        .min()
        .unwrap();
    let end_row = runs.iter().try_fold(0, |end, r| {
        r.y.checked_add(if phase.vertical { r.lines } else { 4 })
            .map(|row| end.max(row))
            .ok_or("run extent overflow")
    })?;
    if end_row > plane_height {
        return Err("run outside coded plane".into());
    }
    let height = end_row - first_row;
    let local_runs: Vec<_> = runs
        .iter()
        .map(|r| EdgeRun {
            y: r.y - first_row,
            ..*r
        })
        .collect();
    let size = width.checked_mul(height).ok_or("plane size overflow")?;
    let mut pixels = Vec::new();
    pixels.try_reserve_exact(size).map_err(|e| e.to_string())?;
    pixels.extend((0..size).map(|i| match report.phases % 3 {
        0 => 128,
        1 => 120 + (i.wrapping_mul(97).wrapping_add(i / width * 53) % 8) as u8,
        _ => i.wrapping_mul(97).wrapping_add(i / width * 53) as u8,
    }));
    let mut view = PlaneViewMut::new(&mut pixels, width, width, height).ok_or("invalid plane")?;
    let start = Instant::now();
    apply(&mut view, &local_runs);
    report.elapsed += start.elapsed();
    for pixel in pixels {
        report.checksum = report
            .checksum
            .wrapping_mul(31)
            .wrapping_add(u64::from(pixel));
    }
    report.phases += 1;
    report.runs += runs.len();
    report.lines += runs.iter().map(|run| run.lines).sum::<usize>();
    Ok(())
}
