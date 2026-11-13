//! Emit sequence payload hex vectors for comparison with pinned AVM's writer.
use nc_av2::sequence::{SequenceHeader, write};

fn main() {
    for single_picture in [false, true] {
        for (max_width, max_height) in [(1, 1), (64, 48), (1920, 1080), (65536, 65536)] {
            let header = SequenceHeader {
                id: 15,
                max_width,
                max_height,
                level: 31,
                single_picture,
                order_hint_bits: if single_picture { 0 } else { 8 },
                crop: nc_av2::sequence::CropWindow::NONE,
                layering: nc_av2::sequence::Layering::FLAT,
                chroma_delta_q: false,
                dc_delta_q: false,
                parity_hiding: false,
                tcq: nc_av2::sequence::TcqFrames::Off,
                ccso: false,
                reference_lists: nc_av2::mv_refs::ListConfig::BASIC,
            };
            let payload = write(&header).expect("valid fixture");
            for byte in payload {
                print!("{byte:02x}");
            }
            println!();
        }
    }
}
