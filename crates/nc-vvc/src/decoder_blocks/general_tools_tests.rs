//! Independent official first-picture syntax and reconstruction checks.
use super::*;
#[test]
#[ignore = "requires official corpus; optional NC_VVC_GENERAL_RECO_TRACE"]
fn official_general_intra_reconstruction() {
    let name = std::env::var("NC_VVC_GENERAL_STREAM").unwrap_or_else(|_| "8b400_A_2".into());
    let path = format!(
        "{}/nanocodec-data/conformance/vvc/streams/{name}/{name}.bit",
        std::env::var("HOME").unwrap()
    );
    let bytes = std::fs::read(path).unwrap();
    let mut sequence = None;
    let mut parameters = None;
    let mut banks: [Option<Box<crate::decoder_headers::Adaptation>>; 8] =
        std::array::from_fn(|_| None);
    let mut reshape_banks: [Option<Box<crate::decoder_headers::Adaptation>>; 8] =
        std::array::from_fn(|_| None);
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
                if aps.kind() == 1 {
                    let id = usize::from(aps.id);
                    reshape_banks[id] = Some(Box::new(aps));
                } else if aps.kind() == 0 {
                    let id = usize::from(aps.id);
                    banks[id] = Some(Box::new(aps));
                }
            }
            kind @ (0..=3 | 7..=10) => {
                let s = sequence.as_ref().unwrap();
                let p = parameters.as_ref().unwrap();
                let header = SliceHeader::parse(&rbsp, kind, s, p, None).unwrap();
                let reshape = reshape_model(s, &header, &reshape_banks).unwrap();
                let counts = filter_counts(&banks, &header).unwrap();
                let mut parsed = parse_filtered(&rbsp, s, p, &header, counts).unwrap();
                eprintln!(
                    "{name}: {} CUs, {} CTUs, ISP={}, MTS={}, LFNST={}",
                    parsed.units.len(),
                    parsed.filters.len(),
                    parsed.units.iter().filter(|u| u.isp.is_some()).count(),
                    parsed.units.iter().filter(|u| u.mts != 0).count(),
                    parsed.units.iter().filter(|u| u.lfnst != 0).count()
                );
                if let Ok(position) = std::env::var("NC_VVC_DEBUG_POSITION") {
                    let xy: Vec<usize> = position.split(',').map(|v| v.parse().unwrap()).collect();
                    for unit in &parsed.units {
                        if (unit.node.x..unit.node.x + unit.node.width).contains(&xy[0])
                            && (unit.node.y..unit.node.y + unit.node.height).contains(&xy[1])
                        {
                            eprintln!("{unit:?}");
                        }
                    }
                }
                let dsp = nc_dsp::Dsp::new();
                inverse_residuals(&mut parsed.units, s, &header, &dsp);
                let mut picture = reconstruct(
                    &parsed.units,
                    s,
                    p.dimensions.map(|v| v as usize),
                    header.picture.joint_sign,
                    [&[], &[]],
                    &dsp,
                    reshape.as_ref().map(|m| (m, header.picture.chroma_scale)),
                )
                .unwrap();
                compare(&picture, "NC_RECO ", 0);
                if let Some(model) = &reshape {
                    dsp.vvc_inverse_lmcs(
                        model,
                        &mut picture.plane_mut(nc_common::PlaneIndex::Y).view_mut(),
                    );
                }
                deblock(&mut picture, &parsed.units, s, &header, &dsp);
                compare(&picture, "NC_DBF ", 0);
                apply_filters(&mut picture, &parsed.filters, s, &header, &banks, &dsp).unwrap();
                compare(&picture, "NC_FINAL ", 0);
                assert_picture(&name, &picture, s.chroma);
                return;
            }
            _ => {}
        }
    }
    panic!("no picture");
}
pub(crate) fn compare(picture: &nc_common::Picture, prefix: &str, poc: i32) {
    let Ok(path) = std::env::var("NC_VVC_GENERAL_RECO_TRACE") else {
        return;
    };
    let trace = std::fs::read_to_string(path).unwrap();
    let mut count = 0;
    let filtered = prefix != "NC_RECO ";
    for line in trace.lines().filter_map(|l| l.strip_prefix(prefix)) {
        let mut v: Vec<usize> = line
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        if v[0] != poc as usize {
            continue;
        }
        if filtered {
            v.insert(4, 4);
            v.insert(5, 4);
        }
        let [_, plane, x, y, width, height, expected] = v.as_slice() else {
            panic!("invalid trace")
        };
        let view = picture.plane(nc_common::PlaneIndex::ALL[*plane]).view();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for row in *y..y + height {
            for &pixel in &view.row(row)[*x..x + width] {
                hash = (hash ^ u64::from(pixel)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        assert_eq!(
            hash, *expected as u64,
            "POC {poc} {prefix}plane {plane} ({x},{y}) {width}x{height}"
        );
        count += 1;
    }
    if prefix == "NC_DBF " && count == 0 {
        return;
    }
    assert!(count > 100);
    eprintln!("{count} regions match VVdeC (filtered={filtered})");
}

fn assert_picture(name: &str, picture: &nc_common::Picture, chroma: bool) {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for plane in nc_common::PlaneIndex::ALL
        .into_iter()
        .take(if chroma { 3 } else { 1 })
    {
        let view = picture.plane(plane).view();
        for y in 0..view.height() {
            for &byte in view.row(y) {
                hash = (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    // Independently captured from pinned VVdeC, with raw output streamed into a hash.
    let expected = match name {
        "8b400_A_2" => 0xf839_40f3_6914_b4bd,
        "8b400_B_2" => 0x2446_d749_29e7_1294,
        "8b420_A_2" => 0x1ec2_e245_7638_5ba5,
        "8b420_B_2" => 0x9c97_d24f_1eba_148e,
        _ => panic!("no independent reference hash for {name}"),
    };
    assert_eq!(hash, expected, "{name} first-picture output");
}
