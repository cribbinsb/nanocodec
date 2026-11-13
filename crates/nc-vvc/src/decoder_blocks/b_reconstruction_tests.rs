//! Downloaded-corpus B-picture reconstruction checks against independent `VVdeC` output.
use super::*;
#[test]
#[ignore = "requires downloaded official VVC corpus"]
fn official_quant_d_bidirectional_pictures() {
    decode_quant(
        "QUANT_D_4",
        [
            Some(0xcd1a_5f19_fca7_4cdf),
            Some(0xa947_8300_1fbf_adc9),
            Some(0xda60_a2d9_2a74_dae1),
        ],
    );
}
#[test]
#[ignore = "requires downloaded official VVC corpus"]
fn official_quant_c_bidirectional_pictures() {
    decode_quant(
        "QUANT_C_2",
        [
            Some(0xf3ec_06cf_f4d8_ca64),
            Some(0x2461_9c17_2392_9f74),
            Some(0xafc8_e37b_56f3_20c0),
        ],
    );
}
fn decode_quant(name: &str, hashes: [Option<u64>; 3]) {
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    let bytes = std::fs::read(format!("{root}/{name}/{name}.bit")).unwrap();
    let (mut sequence, mut parameters) = (None, None);
    let mut previous = None;
    let mut references = crate::decoder_dpb::References::default();
    let dsp = nc_dsp::Dsp::new();
    for nal in nc_common::decode::annex_b_units(&bytes).unwrap() {
        let rbsp = nc_common::decode::unescape_rbsp(&nal[2..]).unwrap();
        match nal[1] >> 3 {
            15 => sequence = Some(Sequence::parse(&rbsp).unwrap()),
            16 => {
                parameters =
                    Some(PictureParameters::parse(&rbsp, sequence.as_ref().unwrap()).unwrap());
            }
            kind @ (0..=3 | 7..=10) => {
                let (s, p) = (sequence.as_ref().unwrap(), parameters.as_ref().unwrap());
                let h = SliceHeader::parse(&rbsp, kind, s, p, None).unwrap();
                let dimensions = p.dimensions.map(|v| v as usize);
                let mut units = parse(&rbsp, s, p, &h).unwrap();
                let (poc, lists) = references
                    .prepare(previous.take(), s, &h, kind, nal[1] & 7)
                    .unwrap();
                derive_motion(
                    &mut units,
                    dimensions,
                    s,
                    &h,
                    [&lists[0], &lists[1]],
                    references
                        .temporal(poc, [&lists[0], &lists[1]], &h, s.ctu_log2)
                        .as_ref(),
                )
                .unwrap();
                super::extended_tests::compare_derived_motion(&units, poc);
                references.record_motion(MotionField::new(&units, dimensions, poc, lists.clone()));
                inverse_residuals(&mut units, s, &h, &dsp);
                let pictures = lists.each_ref().map(|list| references.pictures(list));
                let mut picture = reconstruct(
                    &units,
                    s,
                    dimensions,
                    h.picture.joint_sign,
                    [&pictures[0], &pictures[1]],
                    &dsp,
                    None,
                )
                .unwrap();
                if poc != 0 {
                    compare_regions(&picture, poc);
                }
                deblock(&mut picture, &units, s, &h, &dsp);
                if poc != 0 {
                    compare_filtered(&picture, poc);
                }
                if let Some(expected) = hashes[poc as usize] {
                    assert_eq!(fingerprint(&picture), expected, "{name} POC {poc}");
                }
                if poc == 1 {
                    return;
                }
                previous = Some(picture);
            }
            _ => {}
        }
    }
    panic!("missing B picture");
}
fn fingerprint(picture: &nc_common::Picture) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325u64;
    for plane in nc_common::PlaneIndex::ALL {
        let view = picture.plane(plane).view();
        for y in 0..view.height() {
            for &v in view.row(y) {
                hash = (hash ^ u64::from(v)).wrapping_mul(0x0100_0000_01b3);
            }
        }
    }
    hash
}
fn compare_regions(picture: &nc_common::Picture, poc: i32) {
    let Ok(path) = std::env::var("NC_VVC_B_MOTION_TRACE") else {
        return;
    };
    let trace = std::fs::read_to_string(path).unwrap();
    let mut count = 0;
    for values in trace.lines().filter_map(|l| l.strip_prefix("NC_RECO ")) {
        let values: Vec<u64> = values
            .split_whitespace()
            .map(|s| s.parse().unwrap())
            .collect();
        if values[0] != poc as u64 {
            continue;
        }
        let (plane, x, y, w, h) = (
            values[1] as usize,
            values[2] as usize,
            values[3] as usize,
            values[4] as usize,
            values[5] as usize,
        );
        let view = picture.plane(nc_common::PlaneIndex::ALL[plane]).view();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for row in y..y + h {
            for &pixel in &view.row(row)[x..x + w] {
                hash = (hash ^ u64::from(pixel)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        assert_eq!(hash, values[6], "POC {poc} plane {plane} ({x},{y}) {w}x{h}");
        count += 1;
    }
    eprintln!("{count} B-picture reconstruction regions match VVdeC");
}

fn compare_filtered(picture: &nc_common::Picture, poc: i32) {
    let Ok(path) = std::env::var("NC_VVC_B_MOTION_TRACE") else {
        return;
    };
    let trace = std::fs::read_to_string(path).unwrap();
    for line in trace.lines().filter_map(|l| l.strip_prefix("NC_FINAL ")) {
        let v: Vec<u64> = line
            .split_whitespace()
            .map(|v| v.parse().unwrap())
            .collect();
        if v[0] != poc as u64 {
            continue;
        }
        let (plane, x, y) = (v[1] as usize, v[2] as usize, v[3] as usize);
        let view = picture.plane(nc_common::PlaneIndex::ALL[plane]).view();
        let mut hash = 0xcbf2_9ce4_8422_2325u64;
        for row in y..y + 4 {
            for &pixel in &view.row(row)[x..x + 4] {
                hash = (hash ^ u64::from(pixel)).wrapping_mul(0x0100_0000_01b3);
            }
        }
        assert_eq!(hash, v[4], "filtered plane {plane} at {x},{y}");
    }
}
