//! Bounded native motion diagnostic; isolated coefficient bytes include coder termination.
use nc_av2::transform::TxSize;
use nc_av2::{coeff, entropy, intra, motion, mv_refs, partition, prediction, transform};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;
#[path = "../src/search.rs"]
mod search;

fn source(kind: usize, changed: bool) -> Picture {
    let mut picture = Picture::new(64, 64, 0).unwrap();
    for (p, index) in PlaneIndex::ALL.into_iter().enumerate() {
        let mut plane = picture.plane_mut(index).view_mut();
        for y in 0..plane.height() {
            for x in 0..plane.width() {
                let shift = usize::from(changed && kind == 1) * if p == 0 { 2 } else { 1 };
                let sx = x.saturating_sub(shift);
                let i = y * if p == 0 { 64 } else { 32 } + sx;
                let value = if kind == 2 {
                    (sx * 3 + y * 5 + p * 31) % 256
                } else {
                    (i * 37
                        + i / 64 * 11
                        + 19
                        + usize::from(changed && kind == 0) * 23 * ((i / 8) % 2))
                        % 256
                };
                plane.row_mut(y)[x] = value as u8;
            }
        }
    }
    picture
}
#[derive(Default)]
struct Stats {
    y: u64,
    uv: u64,
    recon: u64,
    bytes: usize,
    mv_bytes: usize,
    changed: usize,
}
fn measure(
    source: &Picture,
    reference: &Picture,
    q: u8,
    x: usize,
    y: usize,
    vector: (i32, i32),
    predictor: (i32, i32),
    dsp: &Dsp,
) -> Stats {
    let mut result = Stats::default();
    let mut writer = entropy::Writer::new();
    let mut u_nonzero = false;
    for (p, bx, by) in [
        (0, x, y),
        (0, x + 4, y),
        (0, x, y + 4),
        (0, x + 4, y + 4),
        (1, x / 2, y / 2),
        (2, x / 2, y / 2),
    ] {
        let mut pixels = [0; 16];
        prediction::predict(
            dsp,
            prediction::NATIVE_FILTER,
            reference.plane(PlaneIndex::ALL[p]),
            bx,
            by,
            vector,
            u8::from(p != 0),
            &mut pixels,
        )
        .unwrap();
        let view = source.plane(PlaneIndex::ALL[p]).view();
        let residual: [i32; 16] = std::array::from_fn(|i| {
            i32::from(view.row(by + i / 4)[bx + i % 4]) - i32::from(pixels[i])
        });
        let sse = residual
            .iter()
            .map(|v| i64::from(*v).pow(2) as u64)
            .sum::<u64>();
        if p == 0 {
            result.y += sse;
        } else {
            result.uv += sse;
        }
        let mut levels = [0; 16];
        transform::quantize_residual(
            dsp,
            TxSize::Tx4,
            &residual,
            (q, transform::NEAREST_ROUNDING_Q7),
            &mut levels,
        )
        .unwrap();
        let mut reconstructed = [0; 16];
        transform::reconstruct(
            dsp,
            (TxSize::Tx4, transform::TxType::DctDct),
            &levels,
            q,
            &mut reconstructed,
        )
        .unwrap();
        for i in 0..16 {
            let value = (i32::from(pixels[i]) + reconstructed[i]).clamp(0, 255);
            result.recon +=
                (i64::from(view.row(by + i / 4)[bx + i % 4]) - i64::from(value)).pow(2) as u64;
        }
        coeff::write(
            &mut writer,
            &levels,
            coeff::TxSize::Tx4,
            coeff::Context {
                is_inter: true,
                plane: p as u8,
                skip: if p == 1 {
                    6
                } else if p == 2 {
                    6 * u8::from(u_nonzero)
                } else {
                    0
                },
                dc_sign: 0,
            },
            q,
        )
        .unwrap();
        if p == 1 {
            u_nonzero = levels.iter().any(|&v| v != 0);
        }
    }
    result.bytes = writer.finish().len();
    if vector != (0, 0) {
        let mut writer = entropy::Writer::new();
        motion::write_mv(&mut writer, vector.0 - predictor.0, vector.1 - predictor.1).unwrap();
        result.mv_bytes = writer.finish().len();
        result.changed = 1;
    }
    result
}
fn main() {
    println!(
        "pattern,q,path,luma_pred_sse,chroma_pred_sse,recon_sse,isolated_coeff_bytes,isolated_mv_bytes,moving_blocks"
    );
    let dsp = Dsp::new();
    for (kind, name) in ["stress", "translated_texture", "static_gradient"]
        .into_iter()
        .enumerate()
    {
        for q in [0, 100, 200] {
            let reference = intra::encode_keyframe(&source(kind, false), q)
                .unwrap()
                .reconstruction;
            let source = source(kind, true);
            for (path, encoded) in [
                (
                    "zero",
                    nc_av2::inter::encode_inter(&source, &reference, q, 1).unwrap(),
                ),
                (
                    "finalist_rd",
                    nc_av2::inter::encode_inter_motion(&source, &reference, q, 1).unwrap(),
                ),
            ] {
                let mut error = 0_u64;
                for plane in PlaneIndex::ALL {
                    let original = source.plane(plane).view();
                    let reconstructed = encoded.reconstruction.plane(plane).view();
                    for row in 0..original.height() {
                        error += original
                            .row(row)
                            .iter()
                            .zip(reconstructed.row(row))
                            .map(|(&a, &b)| (i64::from(a) - i64::from(b)).pow(2) as u64)
                            .sum::<u64>();
                    }
                }
                eprintln!(
                    "packet,{name},{q},{path},bytes={},reconstruction_sse={error}",
                    encoded.packet.len()
                );
            }
            for (path, subpel) in [("zero", 0.0), ("integer", 0.0), ("subpel", 2.0)] {
                let mut search = search::Search::new(q);
                search.set_params(search::SearchParams {
                    subpel_steps: subpel,
                    ..Default::default()
                });
                let mut grid =
                    mv_refs::ReferenceGrid::new(64, 64, mv_refs::ListConfig::BASIC).unwrap();
                let mut sum = Stats::default();
                let mut writer = entropy::Writer::new();
                partition::write_8x8(&mut writer, 64, 64, |_, x, y| {
                    let candidates = grid.candidates(x, y, 8)?;
                    let vector = if path == "zero" {
                        (0, 0)
                    } else {
                        search.find(&source, &reference, x, y, &candidates.vectors[..2])[1]
                    };
                    let result = measure(
                        &source,
                        &reference,
                        q,
                        x,
                        y,
                        vector,
                        candidates.vectors[0],
                        &dsp,
                    );
                    sum.y += result.y;
                    sum.uv += result.uv;
                    sum.recon += result.recon;
                    sum.bytes += result.bytes;
                    sum.mv_bytes += result.mv_bytes;
                    sum.changed += result.changed;
                    grid.commit(x, y, 8, vector, vector != (0, 0))?;
                    Ok(())
                })
                .unwrap();
                println!(
                    "{name},{q},{path},{},{},{},{},{},{}",
                    sum.y, sum.uv, sum.recon, sum.bytes, sum.mv_bytes, sum.changed
                );
            }
        }
    }
}
