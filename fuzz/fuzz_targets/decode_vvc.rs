//! Arbitrary input to the public VVC decoder through the browser, packet
//! and command-line entry points (`common` describes the input layout).
#![no_main]
#![forbid(unsafe_code)]

mod common;

use common::{Mode, framing};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let Some((config, data)) = common::config(data) else {
        return;
    };
    let mut decoder = nc_vvc::Decoder::new();
    match config.mode {
        Mode::Browser => {
            for packet in common::packets(data) {
                if decoder.decode_shown(packet).is_ok() {
                    common::read_shown(decoder.shown());
                }
            }
        }
        Mode::CApi => {
            for packet in common::packets(data) {
                if let Ok(picture) = decoder.decode(packet) {
                    common::read_picture(&picture);
                }
            }
        }
        Mode::Cli | Mode::CliAllLayers => {
            let drain = |decoder: &mut nc_vvc::Decoder| {
                while let Some(picture) = decoder.next_output() {
                    common::read_picture(&picture);
                }
            };
            let result = framing::read_nal_units(&mut &data[..], |nal| {
                if decoder.push_nal_shown(nal).map_err(|e| e.to_string())? {
                    common::read_shown(Some(decoder.shown().expect("a decoded picture is shown")));
                }
                drain(&mut decoder);
                Ok(())
            });
            if result.is_ok() && decoder.flush().is_ok() {
                drain(&mut decoder);
            }
        }
    }
});
