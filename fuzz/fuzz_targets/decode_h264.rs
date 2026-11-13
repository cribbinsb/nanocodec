//! Arbitrary input to the public H.264 decoder through the browser, C API
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
    let mut decoder = nc_h264::Decoder::new();
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
            let mut queue = nc_h264::OutputQueue::default();
            let read = |frame: nc_h264::DecodedFrame| {
                common::read_window(&frame.picture, frame.origin, frame.size);
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
});
