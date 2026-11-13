//! Downloaded-corpus checks for extended intra syntax.
use super::*;
#[test]
#[ignore = "requires downloaded official VVC corpus"]
fn official_extended_intra_syntax() {
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    for name in ["LOSSLESS_A_3", "QUANT_D_4", "QUANT_C_2"] {
        let bytes = std::fs::read(format!("{root}/{name}/{name}.bit")).unwrap();
        let (mut sequence, mut picture) = (None, None);
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
                        parse(&rbsp, s, p, &header).unwrap_or_else(|e| panic!("{name}: {e:?}"));
                    eprintln!("{name}: {} units", units.len());
                    eprintln!(
                        "MRL {} MIP {}",
                        units.iter().filter(|u| u.reference_line != 0).count(),
                        units.iter().filter(|u| u.mip.is_some()).count()
                    );
                    let dsp = nc_dsp::Dsp::new();
                    inverse_residuals(&mut units, s, &header, &dsp);
                    let mut decoded = reconstruct(
                        &units,
                        s,
                        p.dimensions.map(|v| v as usize),
                        header.picture.joint_sign,
                        [&[], &[]],
                        &dsp,
                        None,
                    )
                    .unwrap();
                    if name == "LOSSLESS_A_3" {
                        super::inter_tests::compare_blocks(&decoded, &units, 0);
                    }
                    deblock(&mut decoded, &units, s, &header, &dsp);
                    let mut hash = 0xcbf2_9ce4_8422_2325u64;
                    for plane in nc_common::PlaneIndex::ALL {
                        let view = decoded.plane(plane).view();
                        for y in 0..view.height() {
                            for &byte in view.row(y) {
                                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                            }
                        }
                    }
                    // Complete first-picture output captured independently from VVdeC.
                    assert_eq!(
                        hash,
                        if name == "LOSSLESS_A_3" {
                            0x31cd_1f8d_0212_0dd5
                        } else if name == "QUANT_D_4" {
                            0xcd1a_5f19_fca7_4cdf
                        } else {
                            0xf3ec_06cf_f4d8_ca64
                        }
                    );
                    break;
                }
                _ => {}
            }
        }
    }
}

#[test]
#[ignore = "requires downloaded official VVC corpus"]
fn official_bidirectional_syntax() {
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    for name in ["QUANT_D_4", "QUANT_C_2"] {
        let bytes = std::fs::read(format!("{root}/{name}/{name}.bit")).unwrap();
        let (mut sequence, mut picture) = (None, None);
        let mut count = 0;
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
                    if header.kind == 2 {
                        continue;
                    }
                    let mut units = parse(&rbsp, s, p, &header).unwrap_or_else(|error| {
                        panic!("{name} POC {}: {error:?}", header.picture.order.lsb)
                    });
                    eprintln!(
                        "{name} POC {}: {} units",
                        header.picture.order.lsb,
                        units.len()
                    );
                    let expected = match (name, header.picture.order.lsb) {
                        ("QUANT_D_4", 2) => (12_016, 0xee01_d263_3b76_acae),
                        ("QUANT_D_4", 1) => (432, 0x14c7_42a2_73db_4ea7),
                        ("QUANT_C_2", 2) => (33_948, 0x2826_0f3d_81bd_a07b),
                        ("QUANT_C_2", 1) => (36_761, 0x85c4_e573_94f1_f38e),
                        _ => unreachable!(),
                    };
                    assert_eq!(coefficient_fingerprint(&units), expected);
                    if name == "QUANT_D_4" && header.picture.order.lsb == 2 {
                        derive_and_compare_motion(
                            &mut units,
                            s,
                            &header,
                            p.dimensions.map(|v| v as usize),
                        );
                    }

                    count += 1;
                }
                _ => {}
            }
        }
        assert_eq!(count, 2);
    }
}

/// Signed coefficients captured independently from pinned `VVdeC` before inverse scaling.
fn coefficient_fingerprint(units: &[Unit]) -> (usize, u64) {
    let mut rows = Vec::new();
    for transform in units.iter().flat_map(|u| &u.transforms) {
        for (plane, levels) in transform.levels.iter().enumerate() {
            if levels.is_empty() {
                continue;
            }
            let shift = usize::from(plane != 0);
            let b = transform.block;
            let mut row = vec![
                plane as i32,
                (b.x >> shift) as i32,
                (b.y >> shift) as i32,
                (b.width() >> shift) as i32,
                (b.height() >> shift) as i32,
            ];
            row.extend(levels);
            rows.push(row);
        }
    }
    rows.sort_unstable();
    let count = rows.len();
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for byte in rows.into_iter().flatten().flat_map(i32::to_le_bytes) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
    }
    (count, hash)
}

fn derive_and_compare_motion(
    units: &mut [Unit],
    s: &Sequence,
    h: &SliceHeader,
    dimensions: [usize; 2],
) {
    let Ok(_path) = std::env::var("NC_VVC_B_MOTION_TRACE") else {
        return;
    };
    let end = units.len();
    let refs: [Vec<i32>; 2] = std::array::from_fn(|list| {
        h.references[list]
            .entries
            .iter()
            .take(usize::from(h.active[list]))
            .map(|r| match r {
                crate::decoder_headers::ReferenceEntry::Short(d) => h.picture.order.lsb as i32 + d,
                crate::decoder_headers::ReferenceEntry::Long(_) => panic!("long term"),
            })
            .collect()
    });
    derive_motion(
        &mut units[..end],
        dimensions,
        s,
        h,
        [&refs[0], &refs[1]],
        None,
    )
    .unwrap();
    compare_derived_motion(units, h.picture.order.lsb as i32);
}
pub(super) fn compare_derived_motion(units: &[Unit], poc: i32) {
    let Ok(path) = std::env::var("NC_VVC_B_MOTION_TRACE") else {
        return;
    };
    let mut compared = 0;
    let trace = std::fs::read_to_string(path).unwrap();
    let expected: std::collections::BTreeMap<_, _> = trace
        .lines()
        .filter_map(|l| l.strip_prefix("NC_BMOTION "))
        .map(|l| {
            l.split_whitespace()
                .map(|v| v.parse::<i32>().unwrap())
                .collect::<Vec<_>>()
        })
        .filter(|v| v[0] == poc)
        .map(|v| ((v[1], v[2]), v[5..11].to_vec()))
        .collect();
    for u in units {
        if let Some(m) = u
            .motion
            .filter(|_| u.geometric.is_none() && u.subblocks.is_empty())
        {
            let actual: Vec<_> = m
                .lists
                .into_iter()
                .flat_map(|v| {
                    v.map_or([-1, 0, 0], |v| {
                        [i32::from(v.reference), v.vector.x, v.vector.y]
                    })
                })
                .collect();
            compared += 1;
            assert_eq!(
                Some(&actual),
                expected.get(&(u.node.x as i32, u.node.y as i32)),
                "CU {:?} syntax {:?}",
                u.node,
                u.inter
            );
        }
    }
    eprintln!("{compared} non-geometric, non-subblock inter CUs match VVdeC motion");
}
