//! Official-corpus syntax fingerprints captured independently from `VVdeC`.
use super::*;
#[test]
#[ignore = "requires downloaded official VVC corpus"]
fn official_intra_syntax() {
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    let bytes = std::fs::read(format!(
        "{root}/CodingToolsSets_A_2/CodingToolsSets_A_2.bit"
    ))
    .unwrap();
    let mut sequence = None;
    let mut picture = None;
    let mut frames = 0;
    let (mut modes, mut coefficients) = (FNV_OFFSET, FNV_OFFSET);
    let mut blocks = 0;
    let mut residuals = FNV_OFFSET;
    for unit in nc_common::decode::annex_b_units(&bytes).unwrap() {
        let rbsp = nc_common::decode::unescape_rbsp(&unit[2..]).unwrap();
        match unit[1] >> 3 {
            15 => sequence = Some(Sequence::parse(&rbsp).unwrap()),
            16 => {
                picture =
                    Some(PictureParameters::parse(&rbsp, sequence.as_ref().unwrap()).unwrap());
            }
            kind @ (0..=3 | 7..=10) => {
                let (s, p) = (sequence.as_ref().unwrap(), picture.as_ref().unwrap());
                let header = SliceHeader::parse(&rbsp, kind, s, p, None).unwrap();
                let mut units = parse(&rbsp, s, p, &header).unwrap();
                for unit in &units {
                    if unit.channel == 0 {
                        fingerprint(
                            &mut modes,
                            [
                                unit.node.x as i32,
                                unit.node.y as i32,
                                i32::from(unit.luma_mode),
                            ],
                        );
                    }
                }
                blocks += fingerprint_levels(&mut coefficients, &units);
                inverse_residuals(&mut units, s, &header, &nc_dsp::Dsp::new());
                fingerprint_levels(&mut residuals, &units);
                let reconstructed = reconstruct(
                    &units,
                    s,
                    p.dimensions.map(|v| v as usize),
                    header.picture.joint_sign,
                    [&[], &[]],
                    &nc_dsp::Dsp::new(),
                    None,
                )
                .unwrap();
                compare_reconstruction(&reconstructed, &units, frames);
                corruptions(&rbsp, s, p, &header);
                assert_eq!(units.len(), [1468, 1425][frames]);
                frames += 1;
            }
            _ => {}
        }
    }
    assert_eq!(frames, 2);
    // Independently captured from VVdeC d2614690, not from this decoder.
    // Fingerprints cover every nonzero TU's plane/geometry/signed levels and
    // every luma CU's location/direction, in bitstream order over both pictures.
    assert_eq!(blocks, 2051);
    // Reference residual blocks reordered by plane/location to CABAC order.
    assert_eq!(residuals, 0xb89b_6c0f_7277_be43);
    assert_eq!(coefficients, 0xe0af_9a8d_8806_93b5);
    assert_eq!(modes, 0x8e58_f1c6_5b14_7920);
}
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
fn fingerprint(hash: &mut u64, values: impl IntoIterator<Item = i32>) {
    for value in values {
        for byte in value.to_le_bytes() {
            *hash = (*hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
}

fn corruptions(rbsp: &[u8], s: &Sequence, p: &PictureParameters, header: &SliceHeader) {
    for end in (header.payload_offset..rbsp.len() - 2).step_by(97) {
        assert!(
            parse(&rbsp[..end], s, p, header).is_err(),
            "accepted truncated slice at {end}"
        );
    }
    // Exercise corrupt arithmetic payloads with deterministic mutations;
    // a mutation may remain valid, but it must never panic or recurse
    // past the bounded coding-tree depth.
    let mut seed = 0x76c0_ffee_u32;
    for _ in 0..64 {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let mut corrupted = rbsp.to_vec();
        let index = header.payload_offset + seed as usize % (rbsp.len() - header.payload_offset);
        corrupted[index] ^= 1 << (seed >> 29);
        let _ = parse(&corrupted, s, p, header);
    }
}

fn compare_reconstruction(picture: &nc_common::Picture, units: &[Unit], frame: usize) {
    let trace = std::env::var("NC_VVC_RECO_TRACE")
        .map(|path| std::fs::read_to_string(path).unwrap())
        .unwrap_or_default();
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
    let mut records = Vec::new();
    let mut count = 0;
    for (unit, transform) in units.iter().flat_map(|unit| {
        unit.transforms
            .iter()
            .map(move |transform| (unit, transform))
    }) {
        for plane in if unit.channel == 0 { 0..1 } else { 1..3 } {
            let n = transform.block;
            let (x, y, width, height) = (
                n.x >> unit.channel,
                n.y >> unit.channel,
                n.width() >> unit.channel,
                n.height() >> unit.channel,
            );
            let view = picture
                .plane(
                    [
                        nc_common::PlaneIndex::Y,
                        nc_common::PlaneIndex::U,
                        nc_common::PlaneIndex::V,
                    ][plane],
                )
                .view();
            let mut hash = FNV_OFFSET;
            for dy in 0..height {
                for &value in &view.row(y + dy)[x..x + width] {
                    hash = (hash ^ u64::from(value)).wrapping_mul(0x0000_0100_0000_01b3);
                }
            }
            let key = [plane, x, y, width, height].map(|v| v as u64).to_vec();
            records.push([key[0], key[1], key[2], key[3], key[4], hash]);
            if !trace.is_empty() {
                assert_eq!(
                    Some(&hash),
                    expected.get(&key),
                    "frame {frame} block {key:?} mode {} joint {}",
                    if plane == 0 {
                        unit.luma_mode
                    } else {
                        unit.chroma_mode
                    },
                    transform.joint
                );
            }
            count += 1;
        }
    }
    if !trace.is_empty() {
        assert_eq!(count, expected.len());
    }
    records.sort_unstable();
    let mut hash = FNV_OFFSET;
    for row in records {
        for value in row {
            for byte in value.to_le_bytes() {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
    }
    // Independent VVdeC hashes of all blocks before in-loop filtering.
    assert_eq!(hash, [0x707b_247b_3172_0035, 0x4464_1c04_5aaa_fa8b][frame]);
    eprintln!("{count} reconstructed blocks match VVdeC");
}

fn fingerprint_levels(hash: &mut u64, units: &[Unit]) -> usize {
    let mut count = 0;
    for transform in units.iter().flat_map(|unit| &unit.transforms) {
        for (plane, values) in transform
            .levels
            .iter()
            .enumerate()
            .filter(|(_, v)| !v.is_empty())
        {
            let shift = usize::from(plane != 0);
            let block = transform.block;
            fingerprint(
                hash,
                [
                    plane as i32,
                    (block.x >> shift) as i32,
                    (block.y >> shift) as i32,
                    (block.width() >> shift) as i32,
                    (block.height() >> shift) as i32,
                ],
            );
            fingerprint(hash, values.iter().copied());
            count += 1;
        }
    }
    count
}
