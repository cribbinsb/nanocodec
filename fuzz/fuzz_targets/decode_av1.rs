//! Arbitrary input to the public AV1 decoder through the browser, C API and
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
    let mut decoder = nc_av1::Decoder::new();
    decoder.set_film_grain(!config.skip_film_grain);
    if decoder.set_operating_point(config.operating_point).is_err() {
        return;
    }
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
                if let Ok(Some(picture)) = decoder.decode_optional(packet) {
                    common::read_picture(&picture);
                }
            }
        }
        Mode::Cli | Mode::CliAllLayers => {
            let all_layers = config.mode == Mode::CliAllLayers;
            let _ = framing::read_ivf(&mut &data[..], |header, frame| {
                if framing::ivf_fourcc(Codec::Av1).is_none_or(|fourcc| header.fourcc() != fourcc) {
                    return Err("IVF codec identifier disagrees with the codec".into());
                }
                if all_layers {
                    for (picture, _) in decoder
                        .decode_all_layers(frame.packet)
                        .map_err(|e| e.to_string())?
                    {
                        common::read_picture(&picture);
                        decoder.recycle(picture);
                    }
                } else if decoder
                    .decode_and_show(frame.packet)
                    .map_err(|e| e.to_string())?
                {
                    common::read_shown(Some(decoder.shown().expect("a decoded picture is shown")));
                }
                Ok(())
            });
        }
    }
});
