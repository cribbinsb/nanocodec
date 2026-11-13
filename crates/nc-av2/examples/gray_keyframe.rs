//! Minimal AV2 tile probe: one native 64x64 DC-predicted gray keyframe.
//!
//! This is a conformance probe, not a general encoder. Default CDF entries are
//! from AVM v1.0.0 `entropy_inits_{modes,coeffs}.h` (Clear BSD + AOM patent terms).
use nc_av2::{dc_tile, frame_header, obu, sequence};
use std::io::Write;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let levels: Vec<i32> = if args.len() == 1 {
        vec![0]
    } else {
        args[1..]
            .iter()
            .map(|s| s.parse())
            .collect::<Result<_, _>>()?
    };
    let sequence = sequence::SequenceHeader {
        id: 0,
        max_width: 64,
        max_height: 64,
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
    let mut packet = Vec::new();
    obu::write(
        obu::Header {
            obu_type: 2,
            temporal_id: 0,
            embedded_id: 0,
            extended_id: obu::GLOBAL_XLAYER_ID,
        },
        &[],
        &mut packet,
    )?;
    let header = obu::Header {
        obu_type: 1,
        temporal_id: 0,
        embedded_id: 0,
        extended_id: 0,
    };
    obu::write(header, &sequence::write(&sequence)?, &mut packet)?;
    for (index, dc) in levels.into_iter().enumerate() {
        let frame = frame_header::KeyframeHeader {
            adaptive_cdf: false,
            order_hint: (index % 256) as u16,
            quantizer: 255,
            quant_deltas: nc_av2::transform::QuantDeltas::NONE,
            transform_select: false,
            tx_set: nc_av2::tx_type::TxSetMode::IntraDct,
            levels: nc_av2::coeff::LevelCoding::Plain,
            deblock: nc_av2::deblock::DeblockParams::OFF,
            cdef: nc_av2::cdef::CdefParams::OFF,
            ccso: nc_av2::ccso::CcsoParams::OFF,
            delta_q: None,
        };
        let mut payload = frame_header::write_keyframe(&sequence, frame)?;
        payload.extend(dc_tile::encode(dc)?);
        obu::write(
            obu::Header {
                obu_type: frame_header::CLOSED_LOOP_KEY,
                ..header
            },
            &payload,
            &mut packet,
        )?;
    }
    std::io::stdout().lock().write_all(&packet)?;
    Ok(())
}
