//! Official general B-picture syntax checks independent of motion reconstruction.
use super::*;

#[test]
#[ignore = "requires official corpus"]
fn official_general_inter_syntax() {
    let name = std::env::var("NC_VVC_GENERAL_STREAM").unwrap_or_else(|_| "8b400_B_2".into());
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    let bytes = std::fs::read(format!("{root}/{name}/{name}.bit")).unwrap();
    let (mut sequence, mut parameters) = (None, None);
    let mut banks: [Option<Box<crate::decoder_headers::Adaptation>>; 8] =
        std::array::from_fn(|_| None);
    let limit = std::env::var("NC_VVC_GENERAL_FRAMES").map_or(49, |v| v.parse().unwrap());
    let mut pictures = 0;
    for nal in nc_common::decode::annex_b_units(&bytes).unwrap() {
        let rbsp = nc_common::decode::unescape_rbsp(&nal[2..]).unwrap();
        match nal[1] >> 3 {
            15 => sequence = Some(Sequence::parse(&rbsp).unwrap()),
            16 => {
                parameters =
                    Some(PictureParameters::parse(&rbsp, sequence.as_ref().unwrap()).unwrap());
            }
            17 | 18 => {
                let aps = crate::decoder_headers::Adaptation::parse(&rbsp).unwrap();
                if aps.kind() == 0 {
                    let id = usize::from(aps.id);
                    banks[id] = Some(Box::new(aps));
                }
            }
            kind @ (0..=3 | 7..=10) => {
                let (s, p) = (sequence.as_ref().unwrap(), parameters.as_ref().unwrap());
                let h = SliceHeader::parse(&rbsp, kind, s, p, None).unwrap();
                let counts = filter_counts(&banks, &h).unwrap();
                let parsed = parse_filtered(&rbsp, s, p, &h, counts).unwrap_or_else(|e| {
                    panic!(
                        "{name} picture {pictures} POC {}: {e:?}",
                        h.picture.order.lsb
                    )
                });
                eprintln!(
                    "{name} POC {}: {} units, {} SBT",
                    h.picture.order.lsb,
                    parsed.units.len(),
                    parsed.units.iter().filter(|u| u.sbt.is_some()).count()
                );
                pictures += 1;
                if pictures == limit {
                    break;
                }
            }
            _ => {}
        }
    }
    assert_eq!(pictures, limit);
}

#[test]
#[ignore = "requires official corpus"]
fn official_general_first_b_picture() {
    decode_prefix(2, None);
}

#[test]
#[ignore = "requires official corpus"]
fn official_general_refined_b_pictures() {
    let frames = std::env::var("NC_VVC_GENERAL_FRAMES").map_or(3, |v| v.parse().unwrap());
    let expected = std::env::var("NC_VVC_PREFIX_HASH").map_or_else(
        |_| {
            assert_eq!(frames, 3);
            assert_eq!(
                std::env::var("NC_VVC_GENERAL_STREAM").unwrap_or_else(|_| "8b400_B_2".into()),
                "8b400_B_2"
            );
            // Independent VVdeC output in display order: POC 0, 8, 16.
            0x17e3_855e_53b1_6608
        },
        |value| u64::from_str_radix(&value, 16).unwrap(),
    );
    decode_prefix(frames, Some(expected));
}

#[test]
#[ignore = "requires official corpus"]
fn official_geometric_transform_edge_deblocking() {
    // POC 8 has an uncoded TU boundary inside a geometric CU at (304,204).
    // Comparing its geometric motion field there incorrectly filters three
    // luma pixels. The independent VVdeC hash covers POCs 0, 8 and 16.
    decode_named_prefix("8b420_A_2", 3, Some(0x3989_a3af_3b20_6f41));
}

fn decode_prefix(limit: usize, reference_hash: Option<u64>) {
    let name = std::env::var("NC_VVC_GENERAL_STREAM").unwrap_or_else(|_| "8b400_B_2".into());
    decode_named_prefix(&name, limit, reference_hash);
}

fn decode_named_prefix(name: &str, limit: usize, reference_hash: Option<u64>) {
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    let bytes = std::fs::read(format!("{root}/{name}/{name}.bit")).unwrap();
    let mut decoder = crate::Decoder::new();
    let mut pictures = Vec::new();
    let mut count = 0;
    for nal in nc_common::decode::annex_b_units(&bytes).unwrap() {
        if let Some(picture) = decoder.push_nal(nal).unwrap() {
            pictures.push(picture);
        }
        while let Some(picture) = decoder.next_output() {
            pictures.push(picture);
        }
        if matches!(nal[1] >> 3, 0..=3 | 7..=10) {
            count += 1;
        }
        if count == limit {
            break;
        }
    }
    decoder.flush().unwrap();
    while let Some(picture) = decoder.next_output() {
        pictures.push(picture);
    }
    assert_eq!(pictures.len(), limit);
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for picture in pictures {
        for plane in nc_common::PlaneIndex::ALL
            .into_iter()
            .take(if decoder.is_monochrome() { 1 } else { 3 })
        {
            let view = picture.plane(plane).view();
            for y in 0..view.height() {
                for &byte in view.row(y) {
                    hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
                }
            }
        }
    }
    // Independent streamed VVdeC output, POC 0 then POC 16.
    let expected = reference_hash.unwrap_or_else(|| match name {
        "8b400_A_2" => 0xce5d_eec6_1029_afda,
        "8b400_B_2" => 0xa0a2_c709_227e_fffe,
        "8b420_A_2" => 0x28ff_398c_a163_7695,
        "8b420_B_2" => 0x6d7e_6ce8_20ef_0151,
        _ => panic!("unknown reference {name}"),
    });
    assert_eq!(hash, expected, "{name} first {limit} pictures");
}
