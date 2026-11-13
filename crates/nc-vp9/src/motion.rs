//! VP9 NEWMV differences: syntax, symbol counts and bit costs (VP9
//! §6.4.19-6.4.20; libvpx `vp9_encodemv.c` `encode_mv_component`,
//! `vp9_decodemv.c` `read_mv_component`, `vp9_entropymv.c` `vp9_inc_mv`;
//! BSD-3-Clause, see `THIRD_PARTY.md`).
//!
//! Differences are in eighth samples. The lowest magnitude bit is coded
//! when the frame allows high precision and the reference vector is small
//! (`use_mv_hp`), else inferred as 1 (even components).

use nc_common::decode::{DecodeError, Result};

#[cfg(test)]
use crate::bool_coder::BoolWriter;
use crate::bool_coder::{BitSink, BoolReader};
use crate::probs::{
    Counts, FrameContext, MV_CLASS, MV_CLASS0, MV_FP, MV_JOINT, MvComponentCounts, MvComponentProbs,
};
use crate::tree::{self, Tree};

/// Largest NEWMV difference component the coded classes reach.
pub(crate) const MAX_DIFFERENCE: i32 = 16_382;
/// `kMvRefThresh`: reference components below 8 samples (1/8 units)
/// allow 1/8 differences.
const HIGH_PRECISION_LIMIT: i32 = 64;
/// Integer offset bits of class 0 (`CLASS0_BITS`).
const CLASS0_BITS: usize = 1;

/// One magnitude decomposed as the syntax codes it (`vp9_get_mv_class`).
struct Component {
    negative: bool,
    class: usize,
    integer: usize,
    fraction: usize,
    /// The 1/8 bit of `magnitude - 1`.
    high: usize,
}

impl Component {
    fn of(value: i32) -> Self {
        let z = value.unsigned_abs() as usize - 1;
        let class = if z < 16 { 0 } else { (z >> 3).ilog2() as usize };
        let offset = z - if class == 0 { 0 } else { 8 << class };
        Self {
            negative: value < 0,
            class,
            integer: offset >> 3,
            fraction: (offset >> 1) & 3,
            high: offset & 1,
        }
    }

    fn value(&self) -> i32 {
        let base = if self.class == 0 { 0 } else { 8 << self.class };
        let magnitude = (base + (self.integer << 3) + (self.fraction << 1) + self.high + 1) as i32;
        if self.negative { -magnitude } else { magnitude }
    }

    /// Integer offset bits of the class.
    fn bits(&self) -> usize {
        self.class + CLASS0_BITS - 1
    }
}

/// Where the bits of a difference go: a writer, a cost sum, or nowhere.
trait Sink {
    fn bit(&mut self, bit: bool, prob: u8, slot: usize);
    fn symbol(&mut self, tree: &Tree, probs: &[u8], base_slot: usize, value: usize);
}

impl<W: BitSink> Sink for W {
    fn bit(&mut self, bit: bool, prob: u8, slot: usize) {
        self.write_mode(bit, prob, slot);
    }
    fn symbol(&mut self, tree: &Tree, probs: &[u8], base_slot: usize, value: usize) {
        for &(node, bit) in tree.path(value) {
            let node = usize::from(node);
            self.write_mode(bit, probs[node], base_slot + node);
        }
    }
}

/// The direct bit count (the oracle of [`MvCosts`]).
#[cfg(test)]
struct Cost(f32);

#[cfg(test)]
impl Sink for Cost {
    fn bit(&mut self, bit: bool, prob: u8, _slot: usize) {
        self.0 += tree::bit_cost(bit, prob);
    }
    fn symbol(&mut self, tree: &Tree, probs: &[u8], _base_slot: usize, value: usize) {
        self.0 += tree::cost(tree, probs, value);
    }
}

fn joint(row: i32, col: i32) -> usize {
    usize::from(col != 0) + 2 * usize::from(row != 0)
}

fn emit_component(
    sink: &mut impl Sink,
    axis: usize,
    value: i32,
    probs: &MvComponentProbs,
    usehp: bool,
) {
    use crate::recorder;
    let c = Component::of(value);
    sink.bit(c.negative, probs.sign, recorder::mv_sign_slot(axis));
    sink.symbol(
        MV_CLASS,
        &probs.classes,
        recorder::mv_classes_slot(axis),
        c.class,
    );
    if c.class == 0 {
        sink.symbol(
            MV_CLASS0,
            &probs.class0,
            recorder::mv_class0_slot(axis),
            c.integer,
        );
        sink.symbol(
            MV_FP,
            &probs.class0_fp[c.integer],
            recorder::mv_class0_fp_slot(axis, c.integer),
            c.fraction,
        );
    } else {
        for i in 0..c.bits() {
            sink.bit(
                c.integer & (1 << i) != 0,
                probs.bits[i],
                recorder::mv_bits_slot(axis, i),
            );
        }
        sink.symbol(MV_FP, &probs.fp, recorder::mv_fp_slot(axis), c.fraction);
    }
    if usehp {
        let (prob, slot) = if c.class == 0 {
            (probs.class0_hp, recorder::mv_class0_hp_slot(axis))
        } else {
            (probs.hp, recorder::mv_hp_slot(axis))
        };
        sink.bit(c.high != 0, prob, slot);
    }
}

fn emit(sink: &mut impl Sink, row: i32, col: i32, context: &FrameContext, usehp: bool) {
    sink.symbol(
        MV_JOINT,
        &context.mv_joints,
        crate::recorder::MV_JOINT_SLOT,
        joint(row, col),
    );
    for (axis, value) in [row, col].into_iter().enumerate() {
        if value != 0 {
            emit_component(sink, axis, value, &context.mv[axis], usehp);
        }
    }
}

/// `vp9_inc_mv`: the 1/8 bit is counted whether or not it was coded.
fn count(row: i32, col: i32, counts: &mut Counts) {
    counts.mv_joints[joint(row, col)] += 1;
    for (axis, value) in [row, col].into_iter().enumerate() {
        if value == 0 {
            continue;
        }
        let c = Component::of(value);
        let comp: &mut MvComponentCounts = &mut counts.mv[axis];
        comp.sign[usize::from(c.negative)] += 1;
        comp.classes[c.class] += 1;
        if c.class == 0 {
            comp.class0[c.integer] += 1;
            comp.class0_fp[c.integer][c.fraction] += 1;
            comp.class0_hp[c.high] += 1;
        } else {
            for i in 0..c.bits() {
                comp.bits[i][(c.integer >> i) & 1] += 1;
            }
            comp.fp[c.fraction] += 1;
            comp.hp[c.high] += 1;
        }
    }
}

fn check(row: i32, col: i32, usehp: bool) -> Result<()> {
    if [row, col]
        .iter()
        .any(|v| (!usehp && v & 1 != 0) || !(-MAX_DIFFERENCE..=MAX_DIFFERENCE).contains(v))
    {
        return Err(DecodeError::Unsupported(
            "VP9 motion difference outside the coded range or precision",
        ));
    }
    Ok(())
}

/// `use_mv_hp`: 1/8 precision applies near a small reference vector.
pub(crate) fn uses_high_precision(reference: crate::types::Mv) -> bool {
    reference.row.abs() < HIGH_PRECISION_LIMIT && reference.col.abs() < HIGH_PRECISION_LIMIT
}

/// Encode and count a NEWMV difference in eighth samples.
/// # Errors
/// Rejects odd components and differences outside the coded range, before
/// writing anything.
pub(crate) fn encode(
    writer: &mut impl BitSink,
    row: i32,
    col: i32,
    context: &FrameContext,
    counts: &mut Counts,
    usehp: bool,
) -> Result<()> {
    check(row, col, usehp)?;
    emit(writer, row, col, context, usehp);
    count(row, col, counts);
    Ok(())
}

/// Estimated bits of a NEWMV difference (coded range; odd components only
/// with `usehp`), summed symbol by symbol.
#[cfg(test)]
fn cost(row: i32, col: i32, context: &FrameContext, usehp: bool) -> f32 {
    let mut cost = Cost(0.0);
    emit(&mut cost, row, col, context, usehp);
    cost.0
}

/// Bits of every vector symbol under one frame's probabilities: mode
/// decision prices vectors per candidate, and these only change between
/// frames. [`MvCosts::cost`] adds them in the order the symbols are coded,
/// so the sums equal a symbol-by-symbol count bit for bit.
#[derive(Debug, Clone)]
pub(crate) struct MvCosts {
    joints: [f32; 4],
    components: [ComponentCosts; 2],
}

#[derive(Debug, Clone)]
struct ComponentCosts {
    sign: [f32; 2],
    classes: [f32; MV_CLASSES],
    class0: [f32; 2],
    bits: [[f32; 2]; crate::probs::MV_OFFSET_BITS],
    class0_fp: [[f32; 4]; 2],
    fp: [f32; 4],
    class0_hp: [f32; 2],
    hp: [f32; 2],
}

/// `MV_CLASSES`.
const MV_CLASSES: usize = 11;

/// The bits of each of a tree's `N` symbols.
fn symbol_costs<const N: usize>(tree: &Tree, probs: &[u8]) -> [f32; N] {
    std::array::from_fn(|value| tree::cost(tree, probs, value))
}

impl ComponentCosts {
    fn new(probs: &MvComponentProbs) -> Self {
        let bits = |prob: u8| [false, true].map(|bit| tree::bit_cost(bit, prob));
        Self {
            sign: bits(probs.sign),
            classes: symbol_costs(MV_CLASS, &probs.classes),
            class0: symbol_costs(MV_CLASS0, &probs.class0),
            bits: probs.bits.map(bits),
            class0_fp: probs.class0_fp.map(|p| symbol_costs(MV_FP, &p)),
            fp: symbol_costs(MV_FP, &probs.fp),
            class0_hp: bits(probs.class0_hp),
            hp: bits(probs.hp),
        }
    }

    /// [`emit_component`]'s bits, summed onto `total` in its order.
    fn add(&self, total: &mut f32, value: i32, usehp: bool) {
        let c = Component::of(value);
        *total += self.sign[usize::from(c.negative)];
        *total += self.classes[c.class];
        if c.class == 0 {
            *total += self.class0[c.integer];
            *total += self.class0_fp[c.integer][c.fraction];
        } else {
            for i in 0..c.bits() {
                *total += self.bits[i][(c.integer >> i) & 1];
            }
            *total += self.fp[c.fraction];
        }
        if usehp {
            let costs = if c.class == 0 {
                &self.class0_hp
            } else {
                &self.hp
            };
            *total += costs[c.high];
        }
    }
}

impl MvCosts {
    pub(crate) fn new(context: &FrameContext) -> Self {
        Self {
            joints: symbol_costs(MV_JOINT, &context.mv_joints),
            components: [0, 1].map(|axis| ComponentCosts::new(&context.mv[axis])),
        }
    }

    /// Estimated bits of a NEWMV difference (coded range; odd components
    /// only with `usehp`).
    pub(crate) fn cost(&self, row: i32, col: i32, usehp: bool) -> f32 {
        let mut total = 0.0;
        total += self.joints[joint(row, col)];
        for (axis, value) in [row, col].into_iter().enumerate() {
            if value != 0 {
                self.components[axis].add(&mut total, value, usehp);
            }
        }
        total
    }
}

fn read_component(
    reader: &mut BoolReader<'_>,
    probs: &MvComponentProbs,
    usehp: bool,
) -> Result<i32> {
    let negative = reader.read(probs.sign);
    let class = tree::read(reader, MV_CLASS, &probs.classes);
    let (integer, fraction) = if class == 0 {
        let integer = tree::read(reader, MV_CLASS0, &probs.class0);
        (
            integer,
            tree::read(reader, MV_FP, &probs.class0_fp[integer]),
        )
    } else {
        let mut integer = 0;
        for i in 0..class + CLASS0_BITS - 1 {
            integer |= usize::from(reader.read(probs.bits[i])) << i;
        }
        (integer, tree::read(reader, MV_FP, &probs.fp))
    };
    // The 1/8 bit defaults to 1 when not coded.
    let high = if usehp {
        let prob = if class == 0 {
            probs.class0_hp
        } else {
            probs.hp
        };
        usize::from(reader.read(prob))
    } else {
        1
    };
    let value = Component {
        negative,
        class,
        integer,
        fraction,
        high,
    }
    .value();
    if value.abs() > MAX_DIFFERENCE {
        return Err(DecodeError::Malformed(
            "VP9 motion difference exceeds coded range",
        ));
    }
    Ok(value)
}

/// Decode and count a NEWMV difference in eighth samples.
/// # Errors
/// Truncation and out-of-range differences. Discard the reader on error.
pub(crate) fn decode(
    reader: &mut BoolReader<'_>,
    context: &FrameContext,
    counts: &mut Counts,
    usehp: bool,
) -> Result<(i32, i32)> {
    let joint = tree::read(reader, MV_JOINT, &context.mv_joints);
    let row = if joint & 2 != 0 {
        read_component(reader, &context.mv[0], usehp)?
    } else {
        0
    };
    let col = if joint & 1 != 0 {
        read_component(reader, &context.mv[1], usehp)?
    } else {
        0
    };
    count(row, col, counts);
    Ok((row, col))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_pinned_libvpx_motion_writer() {
        // Generated with the unmodified v1.16.0 encode_mv_component,
        // vp9_write_token, default_nmv_context and vpx_writer C routines.
        let expected = [
            0x0a, 0x34, 0x28, 0x85, 0xd4, 0xde, 0x6b, 0xfd, 0x64, 0xa0, 0xc8, 0x3c, 0x89, 0x3b,
            0x9f, 0x8b, 0x64, 0xef, 0xfa, 0x27, 0x9e, 0x47, 0xaf, 0xee, 0x7f, 0xfc, 0xef, 0x8c,
            0xf5, 0xbf, 0xff, 0xf4, 0x3f, 0xb8, 0x3f, 0x23, 0xff, 0xe8, 0xff, 0xf1, 0xff, 0xff,
            0xff, 0xd2, 0xff, 0xfd, 0x97, 0xff, 0xff, 0xff, 0xfe, 0x6b, 0xff, 0xff, 0xff, 0xfd,
            0xe4, 0x00,
        ];
        let values = [
            0, 2, -2, 6, -8, 14, 16, 18, -32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16382,
        ];
        let (context, mut counts) = (FrameContext::default(), Counts::default());
        let mut writer = BoolWriter::new();
        for (index, row) in values.into_iter().enumerate() {
            encode(
                &mut writer,
                row,
                if index % 3 == 0 { 0 } else { -row },
                &context,
                &mut counts,
                false,
            )
            .unwrap();
        }
        assert_eq!(writer.finish(), expected);
    }

    #[test]
    fn every_even_component_and_joint_roundtrip() {
        let (context, mut counts) = (FrameContext::default(), Counts::default());
        let mut writer = BoolWriter::new();
        for value in (-MAX_DIFFERENCE..=MAX_DIFFERENCE).step_by(2) {
            encode(&mut writer, value, 0, &context, &mut counts, false).unwrap();
            encode(&mut writer, 0, value, &context, &mut counts, false).unwrap();
            encode(&mut writer, value, -value, &context, &mut counts, false).unwrap();
            assert!(cost(value, -value, &context, false) > 0.0 || value == 0);
        }
        let packet = writer.finish();
        let mut reader = BoolReader::new(&packet).unwrap();
        for value in (-MAX_DIFFERENCE..=MAX_DIFFERENCE).step_by(2) {
            assert_eq!(
                decode(&mut reader, &context, &mut counts, false).unwrap(),
                (value, 0)
            );
            assert_eq!(
                decode(&mut reader, &context, &mut counts, false).unwrap(),
                (0, value)
            );
            assert_eq!(
                decode(&mut reader, &context, &mut counts, false).unwrap(),
                (value, -value)
            );
        }
    }

    #[test]
    fn cached_costs_equal_the_direct_sums() {
        let mut context = FrameContext::default();
        context.mv[1].classes[3] = 17;
        context.mv[0].hp = 3;
        let cached = MvCosts::new(&context);
        for value in (-MAX_DIFFERENCE..=MAX_DIFFERENCE).step_by(5) {
            for usehp in [false, true] {
                let value = if usehp { value } else { value & !1 };
                let direct = cost(value, -value, &context, usehp);
                assert_eq!(
                    cached.cost(value, -value, usehp).to_bits(),
                    direct.to_bits()
                );
                let direct = cost(0, value, &context, usehp);
                assert_eq!(cached.cost(0, value, usehp).to_bits(), direct.to_bits());
            }
        }
    }

    #[test]
    fn every_high_precision_component_roundtrips() {
        let (context, mut counts) = (FrameContext::default(), Counts::default());
        let mut writer = BoolWriter::new();
        let values = (-MAX_DIFFERENCE..=MAX_DIFFERENCE).filter(|&v| v != 0);
        for value in values.clone() {
            encode(&mut writer, value, -value, &context, &mut counts, true).unwrap();
        }
        let packet = writer.finish();
        let mut reader = BoolReader::new(&packet).unwrap();
        for value in values {
            assert_eq!(
                decode(&mut reader, &context, &mut counts, true).unwrap(),
                (value, -value)
            );
        }
    }

    #[test]
    fn invalid_vectors_do_not_write_partial_syntax() {
        let (context, mut counts) = (FrameContext::default(), Counts::default());
        let mut writer = BoolWriter::new();
        for value in [1, -1, 16384, -16384, i32::MIN, i32::MAX] {
            assert!(encode(&mut writer, 0, value, &context, &mut counts, false).is_err());
        }
        for value in [16384, -16384, i32::MIN, i32::MAX] {
            assert!(encode(&mut writer, 0, value, &context, &mut counts, true).is_err());
        }
        assert_eq!(writer.finish(), BoolWriter::new().finish());
    }
}
