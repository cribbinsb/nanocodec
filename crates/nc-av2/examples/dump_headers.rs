//! Print the headers of an AV2 stream (IVF or raw OBU) as the general
//! decoder parses them: a diagnostic to hold against dav2d's
//! `DEBUG_SEQ_HDR` / `DEBUG_FRAME_HDR` output.
//!
//! `cargo run -p nanocodec-av2 --example dump_headers -- [--trace] FILE`
//!
//! Without `--trace`, one line per OBU with the parsed sequence header.
//! With it, dav2d's frame-header trace format: `OBU type=N` per OBU and
//! `HDR: <checkpoint>: off=<bit offset>` per checkpoint, then `REFS:` with
//! an inter frame's slot per reference rank, and at the end `OUTPUT:` with
//! the display order of every picture output, so that
//! `tools/dev/av2_foreign/header_trace.py` can diff the two field by field.
//! The reference slots are tracked from the headers alone (no pictures).
use std::rc::Rc;

use nc_av2::general::frame_header::{self, Context, MAX_SLOTS, Parsed, SlotHeader, obu_type};
use nc_av2::general::output::{OutputOrder, Release};
use nc_av2::general::sequence::{self, SequenceHeader};
use nc_av2::obu;
use nc_common::decode::BitReader;

/// IVF file and frame header sizes.
const IVF_FILE_HEADER: usize = 32;
const IVF_FRAME_HEADER: usize = 12;
/// OBU type of the sequence header.
const SEQUENCE_HEADER: u8 = 1;

/// The OBU bytes of `file`: an IVF file's packets joined, or the file.
fn obu_bytes(file: &[u8]) -> Vec<u8> {
    if !file.starts_with(b"DKIF") {
        return file.to_vec();
    }
    let mut bytes = Vec::new();
    let mut position = IVF_FILE_HEADER;
    while let Some(header) = file.get(position..position + IVF_FRAME_HEADER) {
        let size = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
        let start = position + IVF_FRAME_HEADER;
        let Some(packet) = file.get(start..start + size) else {
            break;
        };
        bytes.extend_from_slice(packet);
        position = start + size;
    }
    bytes
}

/// The reference slots as headers only, and the display order of each
/// picture output so far.
struct Walk {
    sequence: Option<SequenceHeader>,
    slots: Vec<Option<SlotHeader>>,
    frames: u32,
    order: OutputOrder,
    shown: Vec<u32>,
}

impl Walk {
    /// Output the frame of display order `order` (with what it releases).
    fn show(&mut self, order: u32) {
        for release in self.order.show(&self.slots, order).unwrap_or_default() {
            self.shown.push(match release {
                Release::Current => order,
                Release::Slot(slot) => {
                    self.slots[slot]
                        .as_ref()
                        .expect("a hidden frame")
                        .header
                        .order
                }
            });
        }
    }
}

impl Walk {
    /// Parse one frame OBU's header, printing its checkpoints, and update
    /// the slots as decoding it would.
    fn frame(&mut self, unit: &obu::Obu<'_>) -> Result<(), Box<dyn std::error::Error>> {
        let sequence = self
            .sequence
            .as_ref()
            .ok_or("frame before a sequence header")?;
        let kind = unit.header.obu_type;
        let mut bits = BitReader::new(unit.payload);
        if !obu_type::is_header_only(kind) && !bits.read_bit()? {
            // A later tile group of the frame: no header.
            return Ok(());
        }
        let context = Context {
            sequence,
            slots: &self.slots,
            obu_type: kind,
            temporal_layer: unit.header.temporal_id,
            embedded_layer: unit.header.embedded_id,
        };
        let mut trace = |name: &'static str, offset: usize| println!("HDR: {name}: off={offset}");
        match frame_header::parse(&context, &mut bits, &mut trace)? {
            Parsed::ShowExisting(shown) => {
                let held = self.slots[usize::from(shown.slot)]
                    .clone()
                    .ok_or("empty slot")?;
                self.show(held.header.order);
                if held.header.frame_type == frame_header::FrameType::Key {
                    for slot in &mut self.slots {
                        *slot = Some(held.clone());
                    }
                }
            }
            Parsed::Frame(header) => {
                if !header.is_intra() {
                    let slots: Vec<String> = header.references.iter().map(u8::to_string).collect();
                    println!("REFS: {}", slots.join(","));
                }
                if header.show_immediate {
                    self.show(header.order);
                }
                let refresh = header.refresh;
                let held = SlotHeader {
                    header: Rc::from(header),
                    frame: self.frames,
                };
                self.frames += 1;
                for (index, slot) in self.slots.iter_mut().enumerate() {
                    if refresh & (1 << index) != 0 {
                        *slot = Some(held.clone());
                    }
                }
            }
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let trace = args.first().is_some_and(|arg| arg == "--trace");
    if trace {
        args.remove(0);
    }
    let path = args.first().ok_or("usage: dump_headers [--trace] FILE")?;
    let bytes = obu_bytes(&std::fs::read(path)?);
    let mut walk = Walk {
        sequence: None,
        slots: vec![None; MAX_SLOTS],
        frames: 0,
        order: OutputOrder::default(),
        shown: Vec::new(),
    };
    let mut remaining = bytes.as_slice();
    while !remaining.is_empty() {
        let (unit, rest) = obu::parse(remaining)?;
        remaining = rest;
        let header = unit.header;
        if trace {
            println!("OBU type={}", header.obu_type);
            if header.obu_type == SEQUENCE_HEADER {
                let parsed = sequence::parse(unit.payload)?;
                if walk.sequence.as_ref() != Some(&parsed) {
                    walk.order = OutputOrder::new(parsed.inter.order_hint_bits);
                }
                walk.sequence = Some(parsed);
            } else if obu_type::is_frame(header.obu_type) {
                walk.frame(&unit)?;
            }
            continue;
        }
        print!(
            "OBU type={} size={} temporal={}",
            header.obu_type,
            unit.payload.len(),
            header.temporal_id
        );
        if header.obu_type == SEQUENCE_HEADER {
            match sequence::parse(unit.payload) {
                Ok(parsed) => print!(" {parsed:?}"),
                Err(error) => print!(" ERROR {error}"),
            }
        }
        println!();
    }
    if trace {
        for slot in walk.order.flush(&walk.slots) {
            let held = walk.slots[slot].as_ref().expect("a hidden frame");
            walk.shown.push(held.header.order);
        }
        let shown: Vec<String> = walk.shown.iter().map(u32::to_string).collect();
        println!("OUTPUT: {}", shown.join(","));
    }
    Ok(())
}
