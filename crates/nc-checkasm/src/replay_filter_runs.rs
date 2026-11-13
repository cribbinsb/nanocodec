//! Replay captured AV1 descriptors on generated pixels, retaining pass order.
//! This measures dispatch/traversal and kernel behavior, not decoder speed.

use nc_common::PlaneViewMut;
use nc_dsp::{Av1EdgeFilter, Av1FilterLength, Dsp, EdgeRun, Isa};
use std::{io::Read, time::Instant};

struct Pass {
    width: usize,
    height: usize,
    vertical: bool,
    runs: Vec<EdgeRun<Av1EdgeFilter>>,
}

pub(crate) fn replay(path: &str, batch: bool, isa: Option<Isa>) -> Result<(), String> {
    let data = if path == "-" {
        let mut bytes = Vec::new();
        std::io::stdin()
            .read_to_end(&mut bytes)
            .map_err(|e| e.to_string())?;
        bytes
    } else {
        std::fs::read(path).map_err(|e| e.to_string())?
    };
    if data.starts_with(b"F ") {
        return crate::replay_hevc_runs::replay(&data, batch, isa);
    }
    let passes = parse(&data)?;
    let dsp = isa.map_or_else(
        || Ok(Dsp::new()),
        |isa| Dsp::with_isa(isa).ok_or_else(|| "unsupported ISA".to_owned()),
    )?;
    let mut elapsed = std::time::Duration::ZERO;
    let mut checksum = 0u64;
    let mut count = 0;
    let mut lines = 0;
    for (index, pass) in passes.iter().enumerate() {
        let mut plane: Vec<u8> = (0..pass.width * pass.height)
            .map(|i| {
                let hash = i.wrapping_mul(97).wrapping_add(i / pass.width * 53);
                match index % 3 {
                    0 => 128,
                    1 => 120 + (hash % 8) as u8,
                    _ => hash as u8,
                }
            })
            .collect();
        let mut view = PlaneViewMut::new(&mut plane, pass.width, pass.width, pass.height)
            .ok_or("invalid generated plane")?;
        // Allocating/resetting generated samples and parsing descriptors are
        // outside timing. Each pass sees identical input in both dispatch modes.
        let start = Instant::now();
        if batch {
            dsp.av1_loop_filter_runs(&mut view, pass.vertical, &pass.runs);
        } else {
            for run in &pass.runs {
                dsp.av1_loop_filter_edge(
                    &mut view,
                    (run.x, run.y),
                    pass.vertical,
                    run.lines,
                    run.filter,
                );
            }
        }
        elapsed += start.elapsed();
        for value in plane {
            checksum = checksum.wrapping_mul(31).wrapping_add(u64::from(value));
        }
        count += pass.runs.len();
        lines += pass.runs.iter().map(|run| run.lines).sum::<usize>();
    }
    println!(
        "filter-runs isa={} dispatch={} passes={} runs={count} lines={lines} elapsed_ns={} checksum={checksum:016x}",
        dsp.isa(),
        if batch { "batch" } else { "single" },
        passes.len(),
        elapsed.as_nanos()
    );
    Ok(())
}

fn parse(data: &[u8]) -> Result<Vec<Pass>, String> {
    if !data.len().is_multiple_of(12) {
        return Err("run trace must contain 12-byte records".into());
    }
    let mut passes: Vec<Pass> = Vec::new();
    for record in data.as_chunks::<12>().0 {
        let fields: [usize; 4] = core::array::from_fn(|i| {
            usize::from(u16::from_le_bytes([record[4 + 2 * i], record[5 + 2 * i]]))
        });
        match record[0] {
            1 => {
                if record[2] > 1 || fields[2] == 0 || fields[3] == 0 {
                    return Err("invalid run trace geometry".into());
                }
                passes.push(Pass {
                    width: fields[2],
                    height: fields[3],
                    vertical: record[2] == 1,
                    runs: Vec::new(),
                });
            }
            2 => {
                let length = match record[1] {
                    4 => Av1FilterLength::Four,
                    6 => Av1FilterLength::Six,
                    8 => Av1FilterLength::Eight,
                    14 => Av1FilterLength::Fourteen,
                    _ => return Err("invalid AV1 filter length".into()),
                };
                let thresh = u8::try_from(fields[3]).map_err(|_| "invalid AV1 threshold")?;
                if !fields[2].is_multiple_of(4) {
                    return Err("trace run is not a multiple of four lines".into());
                }
                passes
                    .last_mut()
                    .ok_or("run before pass header")?
                    .runs
                    .push(EdgeRun {
                        x: fields[0],
                        y: fields[1],
                        lines: fields[2],
                        filter: Av1EdgeFilter {
                            length,
                            limit: record[2],
                            blimit: record[3],
                            thresh,
                        },
                    });
            }
            _ => return Err("unknown run trace record".into()),
        }
    }
    Ok(passes)
}
