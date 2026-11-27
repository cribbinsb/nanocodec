//! The browser entry point against the native VVC decoder on official
//! conformance streams (downloaded corpus, so ignored by default): each
//! stream goes through [`BrowserDecoder`] one picture per packet, as the page
//! and `tools/web/packets.mjs` split it, and must show exactly the pictures
//! the native decoder outputs, as the CLI decodes it, in the same order.
//! These seven streams change picture size (`RPR_*`, `BOUNDARY_A_3`) or code field
//! pictures shown as frames (`FIELD_*`); the browser conformance run counted
//! them as failing on its per-picture split of the reference (2026-11-23).

use super::*;
use nc_common::decode::annex_b_units;

/// The VVC picture header NAL type (H.266 Table 5).
const PH_NUT: u8 = 19;
/// VVC NAL types below this are coded slices (H.266 §7.4.2.2).
const FIRST_NON_VCL_NUT: u8 = 12;
/// Raw output samples are three planes, 4:2:0.
const PLANES: usize = 3;
const START_CODE: [u8; 4] = [0, 0, 0, 1];

const STREAMS: [&str; 7] = [
    "BOUNDARY_A_3",
    "FIELD_A_4",
    "FIELD_B_2",
    "RPR_A_4",
    "RPR_B_3",
    "RPR_C_3",
    "RPR_D_1",
];

/// The stream's packets, one per picture: a picture opens at a picture
/// header NAL or a slice with `sh_picture_header_in_slice_header_flag`
/// (H.266 §7.3.7.1), and NALs after a picture's last slice go with the next
/// one (`annexBFrames` in tools/web/packets.mjs).
fn picture_packets(stream: &[u8]) -> Vec<Vec<u8>> {
    let (mut packets, mut current, mut tail) = (Vec::new(), Vec::new(), Vec::new());
    let mut has_slice = false;
    for nal in annex_b_units(stream).unwrap() {
        let kind = nal[1] >> 3;
        let vcl = kind < FIRST_NON_VCL_NUT;
        if has_slice && (kind == PH_NUT || (vcl && nal[2] & 0x80 != 0)) {
            packets.push(std::mem::replace(&mut current, std::mem::take(&mut tail)));
            has_slice = false;
        }
        let target = if vcl || !has_slice {
            current.append(&mut tail);
            &mut current
        } else {
            &mut tail
        };
        target.extend_from_slice(&START_CODE);
        target.extend_from_slice(nal);
        has_slice |= vcl;
    }
    current.append(&mut tail);
    packets.push(current);
    packets
}

/// The native decoder's raw output pictures, as `nanocodec decode` writes them.
fn native_pictures(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut decoder = nc_vvc::Decoder::default();
    let mut pictures = Vec::new();
    let drain = |decoder: &mut nc_vvc::Decoder, pictures: &mut Vec<Vec<u8>>| {
        while let Some(picture) = decoder.next_output() {
            let mut raw = Vec::new();
            picture.write_yuv(&mut raw, PLANES).unwrap();
            pictures.push(raw);
        }
    };
    for nal in annex_b_units(stream).unwrap() {
        if decoder.push_nal_shown(nal).unwrap() {
            let mut raw = Vec::new();
            ShownFrame::Eight(decoder.shown().unwrap())
                .write_yuv(&mut raw, PLANES)
                .unwrap();
            pictures.push(raw);
        }
        drain(&mut decoder, &mut pictures);
    }
    decoder.flush().unwrap();
    drain(&mut decoder, &mut pictures);
    pictures
}

/// The browser module's shown pictures, in the same raw layout (deep samples
/// as 16-bit little-endian words).
fn browser_pictures(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut decoder = BrowserDecoder::new("vvc").unwrap();
    let mut pictures = Vec::new();
    let take = |decoder: &BrowserDecoder, pictures: &mut Vec<Vec<u8>>| {
        pictures.push(if decoder.bit_depth() > 8 {
            let samples = decoder.decoded_i420_u16().unwrap();
            samples.iter().flat_map(|s| s.to_le_bytes()).collect()
        } else {
            decoder.decoded_i420().unwrap()
        });
    };
    for packet in picture_packets(stream) {
        decoder.decode(&packet).unwrap();
        if decoder.frame_shown() {
            take(&decoder, &mut pictures);
        }
        while decoder.next_frame().unwrap() {
            take(&decoder, &mut pictures);
        }
    }
    decoder.finish().unwrap();
    while decoder.next_frame().unwrap() {
        take(&decoder, &mut pictures);
    }
    pictures
}

#[test]
#[ignore = "requires downloaded official VVC corpus"]
fn browser_shows_the_native_pictures_on_resized_and_field_streams() {
    let root = std::env::var("NC_VVC_CORPUS").unwrap_or_else(|_| {
        format!(
            "{}/nanocodec-data/conformance/vvc/streams",
            std::env::var("HOME").unwrap()
        )
    });
    for name in STREAMS {
        let stream = std::fs::read(format!("{root}/{name}/{name}.bit")).unwrap();
        let (native, browser) = (native_pictures(&stream), browser_pictures(&stream));
        assert_eq!(browser.len(), native.len(), "{name}: picture count");
        let first = browser.iter().zip(&native).position(|(b, n)| b != n);
        assert_eq!(first, None, "{name}: first differing picture");
    }
}
