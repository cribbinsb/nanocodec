//! One function per decoder: the decode paths of one fuzz input (layout in
//! `mod.rs`), shared by the libFuzzer targets and the native/WASM replay
//! tool (`fuzz/replay`).

use nc_encoder::Codec;

use super::{Mode, framing};

/// Drive the public H.264 decoder through the browser, C API and command-line entry points.
pub fn h264(data: &[u8]) {
    let Some((config, data)) = super::config(data) else {
        return;
    };
    let mut decoder = nc_h264::Decoder::new();
    match config.mode {
        Mode::Browser => {
            for packet in super::packets(data) {
                if decoder.decode_shown(packet).is_ok() {
                    super::read_shown(decoder.shown());
                }
            }
        }
        Mode::CApi => {
            for packet in super::packets(data) {
                if let Ok(picture) = decoder.decode(packet) {
                    super::read_picture(&picture);
                }
            }
        }
        Mode::Cli | Mode::CliAllLayers => {
            let mut queue = nc_h264::OutputQueue::default();
            let read = |frame: nc_h264::DecodedFrame| {
                super::read_window(&frame.picture, frame.origin, frame.size);
            };
            let _ = framing::read_nal_units(&mut &data[..], |nal| {
                if let Some(frame) = decoder.decode_nal(nal).map_err(|e| e.to_string())? {
                    queue.push(frame).into_iter().for_each(read);
                }
                Ok(())
            });
            queue.flush().into_iter().for_each(read);
        }
    }
}

/// Drive the public HEVC decoder through the browser, C API and command-line entry points.
pub fn hevc(data: &[u8]) {
    let Some((config, data)) = super::config(data) else {
        return;
    };
    let mut decoder = nc_hevc::Decoder::new();
    match config.mode {
        Mode::Browser => {
            for packet in super::packets(data) {
                if decoder.decode_shown(packet).is_ok() {
                    super::read_shown(decoder.shown());
                }
            }
        }
        Mode::CApi => {
            for packet in super::packets(data) {
                if let Ok(picture) = decoder.decode(packet) {
                    super::read_picture(&picture);
                }
            }
        }
        Mode::Cli | Mode::CliAllLayers => {
            let drain = |decoder: &mut nc_hevc::Decoder| {
                while let Some(frame) = decoder.next_output() {
                    super::read_window(&frame.picture, frame.origin, frame.size);
                }
            };
            let result = framing::read_nal_units(&mut &data[..], |nal| {
                decoder.push_nal(nal).map_err(|e| e.to_string())?;
                drain(&mut decoder);
                Ok(())
            });
            if result.is_ok() && decoder.flush().is_ok() {
                drain(&mut decoder);
            }
        }
    }
}

/// Drive the public VVC decoder through the browser, packet and command-line entry points.
pub fn vvc(data: &[u8]) {
    let Some((config, data)) = super::config(data) else {
        return;
    };
    let mut decoder = nc_vvc::Decoder::new();
    match config.mode {
        Mode::Browser => {
            for packet in super::packets(data) {
                if decoder.decode_shown(packet).is_ok() {
                    super::read_shown(decoder.shown());
                }
            }
        }
        Mode::CApi => {
            for packet in super::packets(data) {
                if let Ok(picture) = decoder.decode(packet) {
                    super::read_picture(&picture);
                }
            }
        }
        Mode::Cli | Mode::CliAllLayers => {
            let drain = |decoder: &mut nc_vvc::Decoder| {
                while let Some(picture) = decoder.next_output() {
                    super::read_picture(&picture);
                }
            };
            let result = framing::read_nal_units(&mut &data[..], |nal| {
                if decoder.push_nal_shown(nal).map_err(|e| e.to_string())? {
                    super::read_shown(Some(decoder.shown().expect("a decoded picture is shown")));
                }
                drain(&mut decoder);
                Ok(())
            });
            if result.is_ok() && decoder.flush().is_ok() {
                drain(&mut decoder);
            }
        }
    }
}

/// Drive the public AV1 decoder through the browser, C API and command-line (IVF) entry points.
pub fn av1(data: &[u8]) {
    let Some((config, data)) = super::config(data) else {
        return;
    };
    let mut decoder = nc_av1::Decoder::new();
    decoder.set_film_grain(!config.skip_film_grain);
    if decoder.set_operating_point(config.operating_point).is_err() {
        return;
    }
    match config.mode {
        Mode::Browser => {
            for packet in super::packets(data) {
                if decoder.decode_shown(packet).is_ok() {
                    super::read_shown(decoder.shown());
                }
            }
        }
        Mode::CApi => {
            for packet in super::packets(data) {
                if let Ok(Some(picture)) = decoder.decode_optional(packet) {
                    super::read_picture(&picture);
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
                        super::read_picture(&picture);
                        decoder.recycle(picture);
                    }
                } else if decoder
                    .decode_and_show(frame.packet)
                    .map_err(|e| e.to_string())?
                {
                    super::read_shown(Some(decoder.shown().expect("a decoded picture is shown")));
                }
                Ok(())
            });
        }
    }
}

/// Drive the public VP9 decoder through the browser, C API and command-line (IVF, superframe) entry points.
pub fn vp9(data: &[u8]) {
    let Some((config, data)) = super::config(data) else {
        return;
    };
    let mut decoder = nc_vp9::Decoder::new();
    decoder.set_base_only(config.operating_point > 0);
    match config.mode {
        Mode::Browser => {
            for packet in super::packets(data) {
                if decoder.decode_shown(packet).is_ok() {
                    super::read_shown(decoder.shown());
                }
            }
        }
        Mode::CApi => {
            for packet in super::packets(data) {
                if let Ok(picture) = decoder.decode(packet) {
                    super::read_picture(&picture);
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
                        super::read_window(&layer.picture, (0, 0), (layer.width, layer.height));
                        decoder.recycle(layer.picture);
                    }
                } else if decoder
                    .decode_kept(frame.packet)
                    .map_err(|e| e.to_string())?
                {
                    let (picture, size) = decoder.shown_frame().expect("a kept frame is shown");
                    super::read_window(picture, (0, 0), size);
                }
                Ok(())
            });
        }
    }
}

/// Drive the public AV2 decoder through the browser, packet and command-line (IVF) entry points.
pub fn av2(data: &[u8]) {
    let Some((config, data)) = super::config(data) else {
        return;
    };
    let mut decoder = nc_av2::Decoder::default();
    match config.mode {
        Mode::Browser => {
            for packet in super::packets(data) {
                if decoder.decode_shown(packet).is_ok() {
                    super::read_shown(decoder.shown());
                }
            }
        }
        Mode::CApi => {
            for packet in super::packets(data) {
                if let Ok(picture) = decoder.decode(packet) {
                    super::read_picture(&picture);
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
                super::read_shown(Some(decoder.shown().expect("a decoded picture is shown")));
                Ok(())
            });
        }
    }
}
