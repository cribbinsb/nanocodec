//! Native 8-bit 4:2:0 AV2 keyframes using 8×8 DC/H/V/Paeth blocks and 4×4 WHT/DCT.
//!
//! This initial picture subset reuses reconstruction immediately for subsequent
//! prediction and keeps coefficient/edge scratch storage per picture. A bounded
//! predictor search selects DC/H/V/Paeth without speculative transforms.
//! Loop filters and frame CDF adaptation remain disabled.
use crate::transform::QuantDeltas;
use crate::{
    coeff, entropy, frame_header, intra_modes, obu, partition, sequence, transform, tx_partition,
};
use coeff::TxSize;
use intra_modes::Mode;
use nc_common::decode::{DecodeError, Result};
use nc_common::{Picture, PlaneIndex};
use nc_dsp::Dsp;

/// Encoded packet and encoder reconstruction for external conformance checks.
#[derive(Debug)]
pub struct EncodedFrame {
    /// Final AV2 OBU stream, including sequence and keyframe headers.
    pub packet: Vec<u8>,
    /// Planar 8-bit 4:2:0 reconstructed pixels.
    pub reconstruction: Picture,
}
/// Side of a coding block in luma and chroma samples, as base-two logarithms.
const LUMA_BLOCK_LOG2: u32 = 3;
const CHROMA_BLOCK_LOG2: u32 = 2;
struct Plane {
    above: Vec<u8>,
    left: Vec<u8>,
}
pub(crate) struct Tile {
    qindex: u8,
    planes: [Plane; 3],
    pub(crate) reconstruction: Picture,
    pub(crate) dsp: Dsp,
}
impl Tile {
    pub(crate) fn new(width: usize, height: usize, qindex: u8) -> Result<Self> {
        if width == 0
            || height == 0
            || !width.is_multiple_of(8)
            || !height.is_multiple_of(8)
            || width > 4096
            || width.checked_mul(height).is_none_or(|n| n > 2304 * 4096)
        {
            return Err(DecodeError::Malformed(
                "AV2 intra requires bounded dimensions divisible by eight",
            ));
        }
        let planes = std::array::from_fn(|p| {
            let (w, h) = if p == 0 {
                (width, height)
            } else {
                (width / 2, height / 2)
            };
            Plane {
                above: vec![0; w / 4],
                left: vec![0; h / 4],
            }
        });
        Ok(Self {
            qindex,
            planes,
            reconstruction: Picture::new(width, height, 0)
                .map_err(|_| DecodeError::Malformed("AV2 picture allocation failed"))?,
            dsp: Dsp::new(),
        })
    }

    /// Entropy contexts above and left of a transform; zero outside the picture.
    fn edges(&self, p: usize, x: usize, y: usize, tx: TxSize) -> ([u8; 2], [u8; 2]) {
        let plane = &self.planes[p];
        let units = tx.units();
        let (mut above, mut left) = ([0; 2], [0; 2]);
        if y > 0 {
            above[..units].copy_from_slice(&plane.above[x / 4..x / 4 + units]);
        }
        if x > 0 {
            left[..units].copy_from_slice(&plane.left[y / 4..y / 4 + units]);
        }
        (above, left)
    }
    /// Contexts of a TX4 in plane `p`: luma TX4s split an 8×8 block, chroma
    /// TX4s cover their whole 4×4 block.
    pub(crate) fn context(
        &self,
        p: usize,
        x: usize,
        y: usize,
        u_nonzero: bool,
    ) -> Result<coeff::Context> {
        let (above, left) = self.edges(p, x, y, TxSize::Tx4);
        let block_log2 = if p == 0 {
            LUMA_BLOCK_LOG2
        } else {
            CHROMA_BLOCK_LOG2
        };
        coeff::Context::derive(
            p as u8,
            block_log2,
            TxSize::Tx4,
            false,
            &above,
            &left,
            u_nonzero,
        )
    }
    /// Contexts of the TX8 covering a whole 8×8 luma block.
    pub(crate) fn context8(&self, x: usize, y: usize) -> Result<coeff::Context> {
        let (above, left) = self.edges(0, x, y, TxSize::Tx8);
        coeff::Context::derive(0, LUMA_BLOCK_LOG2, TxSize::Tx8, false, &above, &left, false)
    }
    pub(crate) fn update_context8(&mut self, x: usize, y: usize, coefficients: &[i32; 64]) {
        let context = coeff::entropy_context(coefficients);
        self.planes[0].above[x / 4..x / 4 + 2].fill(context);
        self.planes[0].left[y / 4..y / 4 + 2].fill(context);
    }
    pub(crate) fn reconstruct8(
        &mut self,
        x: usize,
        y: usize,
        prediction: &[i32; 64],
        coefficients: &[i32; 64],
    ) -> Result<()> {
        let mut residual = [0; 64];
        transform::reconstruct(
            &self.dsp,
            (TxSize::Tx8, transform::TxType::DctDct),
            coefficients,
            self.qindex,
            &mut residual,
        )?;
        let mut view = self.reconstruction.plane_mut(PlaneIndex::Y).view_mut();
        for i in 0..64 {
            view.row_mut(y + i / 8)[x + i % 8] = (prediction[i] + residual[i]).clamp(0, 255) as u8;
        }
        self.update_context8(x, y, coefficients);
        Ok(())
    }
    pub(crate) fn reconstruct(
        &mut self,
        p: usize,
        x: usize,
        y: usize,
        prediction: &[i32; 16],
        coefficients: &[i32; 16],
    ) -> Result<()> {
        let mut residual = [0; 16];
        if coefficients.iter().any(|&c| c != 0) {
            transform::reconstruct(
                &self.dsp,
                (TxSize::Tx4, transform::TxType::DctDct),
                coefficients,
                self.qindex,
                &mut residual,
            )?;
        }
        let mut view = self.reconstruction.plane_mut(PlaneIndex::ALL[p]).view_mut();
        for row in 0..4 {
            for col in 0..4 {
                view.row_mut(y + row)[x + col] =
                    (prediction[row * 4 + col] + residual[row * 4 + col]).clamp(0, 255) as u8;
            }
        }
        self.update_context(p, x, y, coefficients);
        Ok(())
    }
    pub(crate) fn update_context(
        &mut self,
        p: usize,
        x: usize,
        y: usize,
        coefficients: &[i32; 16],
    ) {
        let context = coeff::entropy_context(coefficients);
        let plane = &mut self.planes[p];
        plane.above[x / 4] = context;
        plane.left[y / 4] = context;
    }
    pub(crate) fn save_contexts(&self, x: usize, y: usize) -> [[u8; 4]; 3] {
        std::array::from_fn(|p| {
            let (px, py, extra) = if p == 0 {
                (x / 4, y / 4, 1)
            } else {
                (x / 8, y / 8, 0)
            };
            [
                self.planes[p].above[px],
                self.planes[p].above[px + extra],
                self.planes[p].left[py],
                self.planes[p].left[py + extra],
            ]
        })
    }
    pub(crate) fn restore_contexts(&mut self, x: usize, y: usize, saved: [[u8; 4]; 3]) {
        for (p, values) in saved.into_iter().enumerate() {
            let (px, py, extra) = if p == 0 {
                (x / 4, y / 4, 1)
            } else {
                (x / 8, y / 8, 0)
            };
            self.planes[p].above[px] = values[0];
            self.planes[p].above[px + extra] = values[1];
            self.planes[p].left[py] = values[2];
            self.planes[p].left[py + extra] = values[3];
        }
    }
}
pub(crate) fn blocks(x: usize, y: usize) -> [(usize, usize, usize); 6] {
    [
        (0, x, y),
        (0, x + 4, y),
        (0, x, y + 4),
        (0, x + 4, y + 4),
        (1, x / 2, y / 2),
        (2, x / 2, y / 2),
    ]
}

struct Modes {
    above: Vec<Mode>,
    left: Vec<Mode>,
}
impl Modes {
    fn new(width: usize, height: usize) -> Self {
        Self {
            above: vec![Mode::Dc; width / 8],
            left: vec![Mode::Dc; height / 8],
        }
    }
    fn neighbors(&self, x: usize, y: usize) -> (Option<Mode>, Option<Mode>) {
        (
            (y > 0).then(|| self.above[x / 8]),
            (x > 0).then(|| self.left[y / 8]),
        )
    }
    fn commit(&mut self, x: usize, y: usize, mode: Mode) {
        self.above[x / 8] = mode;
        self.left[y / 8] = mode;
    }
}

// Compare only the first transform's already available edges. This bounded
// predictor search avoids speculative transforms or reconstruction copies.
pub(crate) fn choose_mode(
    source: &Picture,
    reconstructed: &Picture,
    x: usize,
    y: usize,
    chroma: bool,
) -> Mode {
    Mode::BASIC
        .into_iter()
        .min_by_key(|&mode| {
            let planes: &[PlaneIndex] = if chroma {
                &[PlaneIndex::U, PlaneIndex::V]
            } else {
                &[PlaneIndex::Y]
            };
            planes
                .iter()
                .map(|&plane| {
                    let prediction = intra_modes::predict4(reconstructed.plane(plane), x, y, mode);
                    let source = source.plane(plane).view();
                    prediction
                        .iter()
                        .enumerate()
                        .map(|(i, &value)| {
                            let delta = i32::from(source.row(y + i / 4)[x + i % 4]) - value;
                            (delta * delta) as u64
                        })
                        .sum::<u64>()
                })
                .sum::<u64>()
        })
        .unwrap_or(Mode::Dc)
}

/// Encode one lossless keyframe from planar Y/U/V pixels.
/// # Errors
/// Rejects invalid dimensions, input length or unsupported syntax constraints.
pub fn encode_lossless(source: &Picture) -> Result<EncodedFrame> {
    encode_keyframe(source, 0)
}

/// Encode one keyframe at native qindex 0..255; zero is lossless.
/// # Errors
/// Rejects unsupported dimensions, allocation failures or syntax constraints.
pub fn encode_keyframe(source: &Picture, qindex: u8) -> Result<EncodedFrame> {
    encode_keyframe_with_adaptation(source, qindex, false)
}
pub(crate) fn encode_keyframe_with_adaptation(
    source: &Picture,
    qindex: u8,
    adaptive_cdf: bool,
) -> Result<EncodedFrame> {
    encode_keyframe_with_options(source, qindex, adaptive_cdf, false)
}

/// Encode a keyframe with optional CDF adaptation and lossy 8×8 luma transforms.
/// # Errors
/// Rejects invalid source dimensions and unsupported coding parameters.
pub fn encode_keyframe_with_options(
    source: &Picture,
    qindex: u8,
    adaptive_cdf: bool,
    tx8: bool,
) -> Result<EncodedFrame> {
    let tx8 = tx8 && qindex != 0;
    let (width, height) = (source.width(), source.height());
    let mut picture = Tile::new(width, height, qindex)?;
    let mut writer = entropy::Writer::new();
    if adaptive_cdf {
        writer.enable_adaptation(qindex);
    }
    let mut modes = Modes::new(width, height);
    partition::write_8x8(&mut writer, width, height, |writer, x, y| {
        let (above, left) = modes.neighbors(x, y);
        let luma = choose_mode(source, &picture.reconstruction, x, y, false);
        let chroma = choose_mode(source, &picture.reconstruction, x / 2, y / 2, true);
        intra_modes::write(writer, qindex == 0, above, left, luma, chroma)?;
        modes.commit(x, y, luma);
        if qindex != 0 {
            tx_partition::write(writer, LUMA_BLOCK_LOG2, false, !tx8)?;
        }
        if tx8 {
            let prediction =
                intra_modes::predict8(picture.reconstruction.plane(PlaneIndex::Y), x, y, luma);
            let view = source.plane(PlaneIndex::Y).view();
            let residual: [i32; 64] =
                std::array::from_fn(|i| i32::from(view.row(y + i / 8)[x + i % 8]) - prediction[i]);
            let mut coefficients = [0; 64];
            transform::quantize_residual(
                &picture.dsp,
                TxSize::Tx8,
                &residual,
                (qindex, transform::NEAREST_ROUNDING_Q7),
                &mut coefficients,
            )?;
            coeff::write(
                writer,
                &coefficients,
                TxSize::Tx8,
                picture.context8(x, y)?,
                qindex,
            )?;
            picture.reconstruct8(x, y, &prediction, &coefficients)?;
        }
        let mut u_nonzero = false;
        for (p, x, y) in blocks(x, y).into_iter().skip(if tx8 { 4 } else { 0 }) {
            let prediction = intra_modes::predict4(
                picture.reconstruction.plane(PlaneIndex::ALL[p]),
                x,
                y,
                if p == 0 { luma } else { chroma },
            );
            let view = source.plane(PlaneIndex::ALL[p]).view();
            let residual: [i32; 16] =
                std::array::from_fn(|i| i32::from(view.row(y + i / 4)[x + i % 4]) - prediction[i]);
            let mut coefficients = [0; 16];
            transform::quantize_residual(
                &picture.dsp,
                TxSize::Tx4,
                &residual,
                (qindex, transform::NEAREST_ROUNDING_Q7),
                &mut coefficients,
            )?;
            coeff::write(
                writer,
                &coefficients,
                TxSize::Tx4,
                picture.context(p, x, y, u_nonzero)?,
                qindex,
            )?;
            if p == 1 {
                u_nonzero = coefficients.iter().any(|&c| c != 0);
            }
            picture.reconstruct(p, x, y, &prediction, &coefficients)?;
        }
        Ok(())
    })?;
    let packet = keyframe_packet(width, height, qindex, adaptive_cdf, writer.finish())?;
    Ok(EncodedFrame {
        packet,
        reconstruction: picture.reconstruction,
    })
}

fn keyframe_packet(
    width: usize,
    height: usize,
    qindex: u8,
    adaptive_cdf: bool,
    tile: Vec<u8>,
) -> Result<Vec<u8>> {
    let seq = sequence::SequenceHeader {
        id: 0,
        max_width: width as u32,
        max_height: height as u32,
        level: 31,
        single_picture: false,
        order_hint_bits: 8,
        crop: sequence::CropWindow::NONE,
        layering: crate::sequence::Layering::FLAT,
        chroma_delta_q: false,
        dc_delta_q: false,
        parity_hiding: false,
        tcq: crate::sequence::TcqFrames::Off,
        ccso: false,
        reference_lists: crate::mv_refs::ListConfig::BASIC,
    };
    let frame = frame_header::KeyframeHeader {
        order_hint: 0,
        quantizer: qindex,
        quant_deltas: QuantDeltas::NONE,
        transform_select: qindex != 0,
        tx_set: crate::tx_type::TxSetMode::IntraDct,
        levels: crate::coeff::LevelCoding::Plain,
        adaptive_cdf,
        deblock: crate::deblock::DeblockParams::OFF,
        cdef: crate::cdef::CdefParams::OFF,
        ccso: crate::ccso::CcsoParams::OFF,
        delta_q: None,
    };
    let mut payload = frame_header::write_keyframe(&seq, frame)?;
    payload.extend(tile);
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
    obu::write(header, &sequence::write(&seq)?, &mut packet)?;
    obu::write(
        obu::Header {
            obu_type: frame_header::CLOSED_LOOP_KEY,
            ..header
        },
        &payload,
        &mut packet,
    )?;
    Ok(packet)
}

/// Decode one native-subset keyframe to a planar picture.
/// # Errors
/// Rejects unsupported sequence/frame/tile tools, framing or arithmetic errors.
pub fn decode_keyframe(mut packet: &[u8]) -> Result<Picture> {
    let mut sequence = None;
    let mut frame = None;
    while !packet.is_empty() {
        let (unit, rest) = obu::parse(packet)?;
        packet = rest;
        if unit.header.obu_type != 2
            && (unit.header.extended_id != 0
                || unit.header.embedded_id != 0
                || unit.header.temporal_id != 0)
        {
            return Err(DecodeError::Unsupported(
                "AV2 intra decoder requires base layer",
            ));
        }
        match unit.header.obu_type {
            2 if unit.payload.is_empty() => {}
            1 => {
                sequence = Some(sequence::parse(unit.payload)?);
            }
            frame_header::CLOSED_LOOP_KEY => {
                let seq = sequence.ok_or(DecodeError::Malformed("AV2 frame before sequence"))?;
                if frame.is_some() {
                    return Err(DecodeError::Malformed(
                        "AV2 single-frame decoder received multiple frames",
                    ));
                }
                let (header, offset) = frame_header::parse_keyframe(&seq, unit.payload)?;
                let qindex = header.quantizer;
                if qindex != 0 && !header.transform_select {
                    return Err(DecodeError::Unsupported(
                        "AV2 intra requires selected four-by-four transforms",
                    ));
                }
                let (width, height) = (seq.max_width as usize, seq.max_height as usize);
                let mut picture = Tile::new(width, height, qindex)?;
                let mut reader = entropy::Reader::new(&unit.payload[offset..])?;
                if header.adaptive_cdf {
                    reader.enable_adaptation(qindex);
                }
                let mut modes = Modes::new(width, height);
                partition::read_8x8(&mut reader, width, height, |reader, x, y| {
                    let (above, left) = modes.neighbors(x, y);
                    let (luma, chroma) = intra_modes::read(reader, qindex == 0, above, left)?;
                    modes.commit(x, y, luma);
                    let tx8 = qindex != 0 && !tx_partition::read(reader, LUMA_BLOCK_LOG2, false)?;
                    if tx8 {
                        let prediction = intra_modes::predict8(
                            picture.reconstruction.plane(PlaneIndex::Y),
                            x,
                            y,
                            luma,
                        );
                        let mut coefficients = [0; 64];
                        let context = picture.context8(x, y)?;
                        coeff::read(reader, &mut coefficients, TxSize::Tx8, context, qindex)?;
                        picture.reconstruct8(x, y, &prediction, &coefficients)?;
                    }
                    let mut u_nonzero = false;
                    for (p, x, y) in blocks(x, y).into_iter().skip(if tx8 { 4 } else { 0 }) {
                        let prediction = intra_modes::predict4(
                            picture.reconstruction.plane(PlaneIndex::ALL[p]),
                            x,
                            y,
                            if p == 0 { luma } else { chroma },
                        );
                        let mut coefficients = [0; 16];
                        let context = picture.context(p, x, y, u_nonzero)?;
                        coeff::read(reader, &mut coefficients, TxSize::Tx4, context, qindex)?;
                        if p == 1 {
                            u_nonzero = coefficients.iter().any(|&c| c != 0);
                        }
                        picture.reconstruct(p, x, y, &prediction, &coefficients)?;
                    }
                    Ok(())
                })?;
                frame = Some(picture.reconstruction);
            }
            _ => return Err(DecodeError::Malformed("AV2 OBU outside intra subset")),
        }
    }
    frame.ok_or(DecodeError::Malformed("AV2 stream contains no picture"))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tx8_picture_quantizer_boundaries_and_edges() {
        for (w, h) in [(8, 8), (16, 16), (72, 40)] {
            let mut source = Picture::new(w, h, 0).unwrap();
            for p in PlaneIndex::ALL {
                let mut view = source.plane_mut(p).view_mut();
                for y in 0..view.height() {
                    for (x, pixel) in view.row_mut(y).iter_mut().enumerate() {
                        *pixel = (x * 71 + y * 37 + x * y * 3) as u8;
                    }
                }
            }
            for q in [1, 24, 25, 90, 91, 140, 141, 190, 191, 255] {
                for adaptive in [false, true] {
                    let encoded = encode_keyframe_with_options(&source, q, adaptive, true).unwrap();
                    let decoded = decode_keyframe(&encoded.packet).unwrap();
                    for p in PlaneIndex::ALL {
                        let actual = decoded.plane(p).view();
                        let expected = encoded.reconstruction.plane(p).view();
                        for y in 0..actual.height() {
                            assert_eq!(actual.row(y), expected.row(y));
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn textured_picture_roundtrip_all_quantizers() {
        for (w, h) in [(8, 8), (16, 16), (64, 64), (72, 40), (128, 72)] {
            let mut state = 27u32;
            let pixels: Vec<_> = (0..w * h * 3 / 2)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    (state >> 24) as u8
                })
                .collect();
            let mut source = Picture::new(w, h, 0).unwrap();
            let mut offset = 0;
            for p in PlaneIndex::ALL {
                let mut v = source.plane_mut(p).view_mut();
                for row in 0..v.height() {
                    let line = v.row_mut(row);
                    let n = line.len();
                    line.copy_from_slice(&pixels[offset..offset + n]);
                    offset += n;
                }
            }
            for qindex in [0, 1, 90, 140, 190, 255] {
                let encoded = encode_keyframe(&source, qindex).unwrap();
                let decoded = decode_keyframe(&encoded.packet).unwrap();
                assert_eq!((decoded.width(), decoded.height()), (w, h));
                for p in PlaneIndex::ALL {
                    let expected = source.plane(p).view();
                    let actual = decoded.plane(p).view();
                    let reconstructed = encoded.reconstruction.plane(p).view();
                    for row in 0..expected.height() {
                        if qindex == 0 {
                            assert_eq!(expected.row(row), actual.row(row));
                        }
                        assert_eq!(actual.row(row), reconstructed.row(row));
                    }
                }
            }
        }
    }
}
