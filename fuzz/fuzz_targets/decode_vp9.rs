//! Arbitrary input to the public VP9 decoder through the browser, C API and
//! command-line (IVF, superframe) entry points (`common` describes the
//! input layout).
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
    let mut decoder = nc_vp9::Decoder::new();
    decoder.set_base_only(config.operating_point > 0);
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
            let all_layers = config.mode == Mode::CliAllLayers;
            let _ = framing::read_ivf(&mut &data[..], |header, frame| {
                if framing::ivf_fourcc(Codec::Vp9).is_none_or(|fourcc| header.fourcc() != fourcc) {
                    return Err("IVF codec identifier disagrees with the codec".into());
                }
                if all_layers {
                    for layer in decoder
                        .decode_all_layers(frame.packet)
                        .map_err(|e| e.to_string())?
                    {
                        common::read_window(&layer.picture, (0, 0), (layer.width, layer.height));
                        decoder.recycle(layer.picture);
                    }
                } else if decoder
                    .decode_kept(frame.packet)
                    .map_err(|e| e.to_string())?
                {
                    let (picture, size) = decoder.shown_frame().expect("a kept frame is shown");
                    common::read_window(picture, (0, 0), size);
                }
                Ok(())
            });
        }
    }
});
