//! Arbitrary input to the public AV2 decoder through the browser, packet and
//! command-line (IVF) entry points (`common` describes the input layout).
#![no_main]
#![forbid(unsafe_code)]

mod common;

use common::{Mode, framing};
use libfuzzer_sys::fuzz_target;
use nc_encoder::Codec;

fuzz_target!(|data: &[u8]| {
    let Some((config, data)) = common::config(data) else {
        return;
    };
    let mut decoder = nc_av2::Decoder::default();
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
            let _ = framing::read_ivf(&mut &data[..], |header, frame| {
                if framing::ivf_fourcc(Codec::Av2).is_none_or(|fourcc| header.fourcc() != fourcc) {
                    return Err("IVF codec identifier disagrees with the codec".into());
                }
                decoder
                    .decode_shown(frame.packet)
                    .map_err(|e| e.to_string())?;
                common::read_shown(Some(decoder.shown().expect("a decoded picture is shown")));
                Ok(())
            });
        }
    }
});
