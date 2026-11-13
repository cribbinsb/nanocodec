//! Bounded hostile-packet checks for the stateful native decoder.
//! All fixtures are tiny in-memory pictures; no external decoder or fuzz runtime.
use nc_av2::{obu, sequence, stream};
use nc_common::{Picture, PlaneIndex};
use std::panic::{AssertUnwindSafe, catch_unwind};

fn picture(side: usize, phase: usize) -> Picture {
    let mut picture = Picture::new(side, side, 0).unwrap();
    for plane in PlaneIndex::ALL {
        let mut view = picture.plane_mut(plane).view_mut();
        for y in 0..view.height() {
            for (x, sample) in view.row_mut(y).iter_mut().enumerate() {
                *sample = ((x * 37 + y * 19 + phase * 29 + x * y * 3) % 256) as u8;
            }
        }
    }
    picture
}

fn equal_pixels(actual: &Picture, expected: &Picture) {
    assert_eq!(
        (actual.width(), actual.height()),
        (expected.width(), expected.height())
    );
    for plane in PlaneIndex::ALL {
        let actual = actual.plane(plane).view();
        let expected = expected.plane(plane).view();
        for row in 0..actual.height() {
            assert_eq!(actual.row(row), expected.row(row));
        }
    }
}

struct Fixture {
    key: Vec<u8>,
    forward: Vec<u8>,
    expected: Picture,
}

impl Fixture {
    fn new(side: usize, quantizer: u8) -> Self {
        Self::with_tx8(side, quantizer, false)
    }

    fn with_tx8(side: usize, quantizer: u8, tx8: bool) -> Self {
        let mut encoder = stream::Encoder::default();
        encoder.set_tx8(tx8);
        let key = encoder.encode(&picture(side, 0), quantizer, false).unwrap();
        let forward = encoder.encode(&picture(side, 1), quantizer, false).unwrap();
        Self {
            key: key.packet,
            forward: forward.packet,
            expected: forward.reconstruction,
        }
    }

    /// A corrupt entropy payload can occasionally describe another valid image.
    /// Such packets are allowed to succeed. Every rejection must be transactional:
    /// a subsequent valid P frame must still reconstruct against the original key.
    fn check(&self, packet: &[u8], must_reject: bool) -> bool {
        let mut decoder = stream::Decoder::default();
        decoder.decode_ref(&self.key).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| decoder.decode_ref(packet).map(|_| ())));
        assert!(result.is_ok(), "decoder panicked on {} bytes", packet.len());
        let rejected = result.unwrap().is_err();
        if must_reject {
            assert!(rejected, "accepted malformed {}-byte packet", packet.len());
        }
        if rejected {
            equal_pixels(decoder.decode_ref(&self.forward).unwrap(), &self.expected);
        }
        rejected
    }
}

#[test]
fn tx8_payload_errors_preserve_reference() {
    let fixture = Fixture::with_tx8(16, 100, true);
    let mut rejections = 0;
    for packet in [&fixture.key, &fixture.forward] {
        for length in 0..packet.len() {
            fixture.check(&packet[..length], true);
        }
        for index in 0..128 {
            let mut corrupt = packet.clone();
            let position = index * 37 % corrupt.len();
            corrupt[position] ^= 1 << (index % 8);
            rejections += usize::from(fixture.check(&corrupt, false));
        }
    }
    assert!(rejections > 16);
}

#[test]
fn truncated_packets_and_payload_mutations_preserve_reference_on_error() {
    let mut cases = 0;
    let mut rejected_mutations = 0;
    for fixture in [Fixture::new(8, 0), Fixture::new(16, 100)] {
        // Framing has an explicit payload size, so every proper prefix is invalid.
        for packet in [&fixture.key, &fixture.forward] {
            for length in 0..packet.len() {
                assert!(
                    cases < 1600,
                    "fixture unexpectedly exceeds bounded case budget"
                );
                fixture.check(&packet[..length], true);
                cases += 1;
            }
            // Deterministically spread flips across both framing and entropy,
            // rather than exhaustively exercising every bit of the same picture.
            for index in 0..128 {
                let mut corrupt = packet.clone();
                let position = index * 37 % corrupt.len();
                corrupt[position] ^= 1 << (index % 8);
                rejected_mutations += usize::from(fixture.check(&corrupt, false));
                cases += 1;
            }
        }
    }
    assert!(cases < 2000);
    assert!(
        rejected_mutations > 64,
        "mutation set failed to exercise rejection paths"
    );
}

fn rewrite_frame(packet: &[u8], change: impl FnOnce(&mut obu::Header, &mut Vec<u8>)) -> Vec<u8> {
    let mut remaining = packet;
    let mut rewritten = Vec::new();
    let mut change = Some(change);
    while !remaining.is_empty() {
        let (unit, rest) = obu::parse(remaining).unwrap();
        remaining = rest;
        let mut header = unit.header;
        let mut payload = unit.payload.to_vec();
        if matches!(header.obu_type, 4 | 7) {
            change.take().unwrap()(&mut header, &mut payload);
        }
        obu::write(header, &payload, &mut rewritten).unwrap();
    }
    rewritten
}

#[test]
fn hostile_framing_layers_dimensions_and_post_picture_errors_are_transactional() {
    let fixture = Fixture::new(8, 0);
    let mut cases = vec![
        vec![],
        vec![0],
        vec![0x80; 8],
        vec![0xff; 9],
        // Claimed 32-bit maximum size and an overflowing 33-bit size.
        vec![0xff, 0xff, 0xff, 0xff, 0x0f, 0x1c],
        vec![0x80, 0x80, 0x80, 0x80, 0x10, 0x1c],
        vec![1, 0x9c], // extension flag with missing extension byte
        vec![1, 0],    // unsupported OBU type
    ];
    for (extended_id, embedded_id, temporal_id) in [(1, 0, 0), (0, 1, 0), (0, 0, 1)] {
        cases.push(rewrite_frame(&fixture.forward, |header, _| {
            header.extended_id = extended_id;
            header.embedded_id = embedded_id;
            header.temporal_id = temporal_id;
        }));
    }
    cases.push(rewrite_frame(&fixture.forward, |_, payload| {
        payload.clear();
    }));
    cases.push(rewrite_frame(&fixture.forward, |_, payload| {
        payload.fill(0xff);
    }));
    // Force failure only after one whole picture has already been reconstructed.
    let mut suffix_error = fixture.forward.clone();
    suffix_error.push(0);
    cases.push(suffix_error);
    let mut two_frames = fixture.forward.clone();
    two_frames.extend(&fixture.forward);
    cases.push(two_frames);
    for (max_width, max_height) in [(16, 16), (65536, 65536), (4096, 4096), (7, 8)] {
        let mut sequence_only = Vec::new();
        let header = sequence::SequenceHeader {
            id: 0,
            max_width,
            max_height,
            level: 31,
            single_picture: false,
            order_hint_bits: 8,
            crop: sequence::CropWindow::NONE,
            layering: nc_av2::sequence::Layering::FLAT,
            chroma_delta_q: false,
            dc_delta_q: false,
            parity_hiding: false,
            tcq: nc_av2::sequence::TcqFrames::Off,
            ccso: false,
            reference_lists: nc_av2::mv_refs::ListConfig::BASIC,
        };
        obu::write(
            obu::Header {
                obu_type: 1,
                temporal_id: 0,
                embedded_id: 0,
                extended_id: 0,
            },
            &sequence::write(&header).unwrap(),
            &mut sequence_only,
        )
        .unwrap();
        cases.push(sequence_only.clone()); // valid new sequence, but no picture
        sequence_only.extend(&fixture.forward); // dimensions conflict with existing reference
        cases.push(sequence_only);
    }
    for packet in cases {
        fixture.check(&packet, true);
    }
    // A missing-reference error must not prevent later normal startup.
    let mut decoder = stream::Decoder::default();
    assert!(decoder.decode_ref(&fixture.forward).is_err());
    decoder.decode_ref(&fixture.key).unwrap();
    equal_pixels(
        decoder.decode_ref(&fixture.forward).unwrap(),
        &fixture.expected,
    );
}
