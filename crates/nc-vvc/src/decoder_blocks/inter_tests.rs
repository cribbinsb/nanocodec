//! Independent official P-picture syntax checks, enabled after corpus download.
use super::*;
use std::fmt::Write;
#[test]
#[ignore = "requires downloaded official VVC corpus"]
fn official_predicted_syntax() {
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    let bytes = std::fs::read(format!(
        "{root}/CodingToolsSets_B_2/CodingToolsSets_B_2.bit"
    ))
    .unwrap();
    let (mut sequence, mut picture) = (None, None);
    let mut frames = 0;
    let mut dpb = crate::decoder_dpb::References::default();
    let mut previous = None;
    let mut trace = String::new();
    for nal in nc_common::decode::annex_b_units(&bytes).unwrap() {
        let rbsp = nc_common::decode::unescape_rbsp(&nal[2..]).unwrap();
        match nal[1] >> 3 {
            15 => sequence = Some(Sequence::parse(&rbsp).unwrap()),
            16 => {
                picture =
                    Some(PictureParameters::parse(&rbsp, sequence.as_ref().unwrap()).unwrap());
            }
            kind @ (0..=3 | 7..=10) => {
                let (s, p) = (sequence.as_ref().unwrap(), picture.as_ref().unwrap());
                let header = SliceHeader::parse(&rbsp, kind, s, p, None).unwrap();
                let mut units =
                    parse(&rbsp, s, p, &header).unwrap_or_else(|e| panic!("frame {frames}: {e:?}"));
                let references: Vec<_> = header.references[0]
                    .entries
                    .iter()
                    .take(usize::from(header.active[0]))
                    .map(|entry| match entry {
                        crate::decoder_headers::ReferenceEntry::Short(delta) => {
                            frames as i32 + delta
                        }
                        crate::decoder_headers::ReferenceEntry::Long(_) => panic!("long term"),
                    })
                    .collect();
                derive_motion(
                    &mut units,
                    p.dimensions.map(|v| v as usize),
                    s,
                    &header,
                    [&references, &[]],
                    None,
                )
                .unwrap();
                record_units(&mut trace, &units, frames);
                let (_, pocs) = dpb.prepare(previous.take(), s, &header, kind, 1).unwrap();
                let references = pocs.each_ref().map(|list| dpb.pictures(list));
                let dsp = nc_dsp::Dsp::new();
                inverse_residuals(&mut units, s, &header, &dsp);
                let mut reconstructed = reconstruct(
                    &units,
                    s,
                    p.dimensions.map(|v| v as usize),
                    header.picture.joint_sign,
                    [&references[0], &references[1]],
                    &dsp,
                    None,
                )
                .unwrap();
                compare_blocks(&reconstructed, &units, frames);
                deblock(&mut reconstructed, &units, s, &header, &dsp);
                compare_filtered(&reconstructed, frames);
                assert_picture(&reconstructed, frames);
                previous = Some(reconstructed);
                frames += 1;
            }
            _ => {}
        }
    }
    assert_eq!(frames, 9);
    // Independently captured from pinned VVdeC d2614690. Sort by picture,
    // component and geometry so oracle scheduling cannot affect the digest.
    let mut motion = Vec::new();
    let mut levels = Vec::new();
    let mut poc = 0;
    for line in trace.lines() {
        if let Some(value) = line.strip_prefix("NC_POC ") {
            poc = value.parse::<i32>().unwrap();
        }
        if let Some(values) = line.strip_prefix("NC_MOTION ") {
            motion.push(numbers(values));
        }
        if let Some(values) = line.strip_prefix("NC_LEVELS ") {
            let mut row = vec![poc];
            row.extend(numbers(values));
            levels.push(row);
        }
    }
    assert_eq!(motion.len(), 1695);
    assert_eq!(levels.len(), 1467);
    assert_eq!(fingerprint(motion), 0xb31c_51c0_5a4b_8e51);
    assert_eq!(fingerprint(levels), 0xb066_7bdd_4203_ba22);
    if let Ok(path) = std::env::var("NC_VVC_SYNTAX_TRACE") {
        std::fs::write(path, trace).unwrap();
    }
}

fn numbers(values: &str) -> Vec<i32> {
    values
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect()
}
fn fingerprint(mut rows: Vec<Vec<i32>>) -> u64 {
    rows.sort_unstable();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for value in rows.into_iter().flatten() {
        for byte in value.to_le_bytes() {
            hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
        }
    }
    hash
}

pub(super) fn compare_blocks(picture: &nc_common::Picture, units: &[Unit], frame: usize) {
    let Ok(path) = std::env::var("NC_VVC_INTER_RECO_TRACE") else {
        return;
    };
    let trace = std::fs::read_to_string(path).unwrap();
    let mut expected = std::collections::BTreeMap::new();
    for line in trace
        .lines()
        .filter_map(|line| line.strip_prefix("NC_RECO "))
    {
        let values: Vec<u64> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        if values[0] == frame as u64 {
            expected.insert(values[1..6].to_vec(), values[6]);
        }
    }
    for (unit, transform) in units.iter().flat_map(|unit| {
        unit.transforms
            .iter()
            .map(move |transform| (unit, transform))
    }) {
        for plane in if unit.combined {
            0..3
        } else if unit.channel == 0 {
            0..1
        } else {
            1..3
        } {
            let shift = usize::from(plane != 0);
            let node = transform.block;
            let (x, y, w, h) = (
                node.x >> shift,
                node.y >> shift,
                node.width() >> shift,
                node.height() >> shift,
            );
            let view = picture.plane(nc_common::PlaneIndex::ALL[plane]).view();
            let mut hash = 0xcbf2_9ce4_8422_2325u64;
            for dy in 0..h {
                for &v in &view.row(y + dy)[x..x + w] {
                    hash = (hash ^ u64::from(v)).wrapping_mul(0x0100_0000_01b3);
                }
            }
            let key = [plane, x, y, w, h].map(|v| v as u64).to_vec();
            assert_eq!(
                Some(&hash),
                expected.get(&key),
                "frame {frame} block {key:?} motion {:?} intra {}/{}",
                unit.motion,
                unit.luma_mode,
                unit.chroma_mode
            );
        }
    }
}

fn compare_filtered(picture: &nc_common::Picture, frame: usize) {
    let Ok(path) = std::env::var("NC_VVC_INTER_RECO_TRACE") else {
        return;
    };
    let trace = std::fs::read_to_string(path).unwrap();
    for line in trace
        .lines()
        .filter_map(|line| line.strip_prefix("NC_FINAL "))
    {
        let v: Vec<u64> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        if v[0] != frame as u64 {
            continue;
        }
        let (plane, x, y) = (v[1] as usize, v[2] as usize, v[3] as usize);
        let view = picture.plane(nc_common::PlaneIndex::ALL[plane]).view();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for dy in y..(y + 4).min(view.height()) {
            for &v in &view.row(dy)[x..(x + 4).min(view.width())] {
                hash = (hash ^ u64::from(v)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        assert_eq!(
            hash, v[4],
            "filtered frame {frame}, plane {plane}, x{x} y{y}"
        );
    }
}

fn record_units(trace: &mut String, units: &[Unit], frames: usize) {
    for unit in units {
        if let Some(m) = unit.motion.and_then(|m| m.lists[0]) {
            writeln!(
                trace,
                "NC_MOTION {frames} {} {} {} {} {} {} {}",
                unit.node.x,
                unit.node.y,
                unit.node.width,
                unit.node.height,
                m.reference,
                m.vector.x,
                m.vector.y
            )
            .unwrap();
        }
    }
    for transform in units.iter().flat_map(|unit| &unit.transforms) {
        for (plane, levels) in transform
            .levels
            .iter()
            .enumerate()
            .filter(|(_, l)| !l.is_empty())
        {
            let shift = usize::from(plane != 0);
            writeln!(trace, "NC_POC {frames}").unwrap();
            write!(
                trace,
                "NC_LEVELS {plane} {} {} {} {}",
                transform.block.x >> shift,
                transform.block.y >> shift,
                transform.block.width() >> shift,
                transform.block.height() >> shift
            )
            .unwrap();
            for level in levels {
                write!(trace, " {level}").unwrap();
            }
            writeln!(trace).unwrap();
        }
    }
}

fn assert_picture(picture: &nc_common::Picture, frame: usize) {
    // Hashes of complete 8-bit output from pinned VVdeC, captured through a pipe.
    let expected = [
        0x0b63_f7f4_9701_9678,
        0x0368_d95c_7824_21b3,
        0xd868_80e3_6f43_be4e,
        0x572a_0e4d_cd8b_2b9e,
        0xb3a5_47fd_a57b_49ff,
        0xdacf_f350_2c3c_f907,
        0xe87f_cae5_1125_f7b5,
        0x4770_d9a0_59ba_50b5,
        0xfb41_bd21_daaf_df86,
    ];
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for plane in nc_common::PlaneIndex::ALL {
        let view = picture.plane(plane).view();
        for y in 0..view.height() {
            for &byte in view.row(y) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    assert_eq!(hash, expected[frame], "complete picture {frame}");
}
