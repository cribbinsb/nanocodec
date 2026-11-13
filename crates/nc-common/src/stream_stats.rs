//! Coding-tool analysis a decoder can record while it parses a stream:
//! what an encoder actually chose, block by block, as opposed to what its
//! headers permit.
//!
//! A decoder holds an `Option<StreamStats>` (off by default: decoding then
//! pays one predictable branch per call site and allocates nothing) and, at
//! its parse points, pushes codec-neutral [`BlockRecord`]s: coding blocks,
//! prediction blocks, transform blocks and loop-filter units, each a
//! rectangle in luma samples with its prediction kind, modes, vectors,
//! coded flags, QP and bits. Picture-level facts (slice type, QP, header
//! switches) go in as keyed values with [`StreamStats::fact`].
//!
//! The records of the picture being decoded sit in one reusable buffer of
//! fixed-size records ([`StreamStats::records`] holds the last completed
//! picture's) so a viewer can draw them without per-block calls. When a
//! picture completes, [`aggregate`] turns its records into keyed tallies
//! ([`FrameStats`]); `nanocodec stat` writes those as JSON and
//! `eval/nceval/streamstats.py` summarises them. The page and the CLI use
//! the same aggregation.

use std::collections::BTreeMap;
use std::fmt::Write as _;

/// What a record describes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RecordKind {
    /// A coding block (HEVC CU, H.264 macroblock): size, prediction,
    /// partitioning, QP, all its bits.
    #[default]
    Coding = 0,
    /// A prediction block: intra modes, or motion.
    Prediction = 1,
    /// A transform block (a leaf of the transform tree).
    Transform = 2,
    /// A loop-filter unit (HEVC SAO CTB).
    Filter = 3,
}

/// How a block is predicted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Pred {
    /// Intra prediction.
    #[default]
    Intra = 0,
    /// Inter with explicitly coded motion (AMVP, H.264 mvd).
    Inter = 1,
    /// Skipped: inherited motion and no residual.
    Skip = 2,
    /// Merged motion with residual syntax (HEVC merge).
    Merge = 3,
}

/// How a coding block is split into prediction blocks (HEVC `PartMode`
/// names; H.264 16x8 is `HalfHorizontal`, `I_4x4` is `Sixteenth`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Shape {
    /// One prediction block (2Nx2N, 16x16).
    #[default]
    Whole = 0,
    /// Two stacked halves (2NxN, 16x8).
    HalfHorizontal = 1,
    /// Two side-by-side halves (`Nx2N`, 8x16).
    HalfVertical = 2,
    /// Four quarters (`NxN`, 8x8).
    Quarter = 3,
    /// Asymmetric: a quarter-height top part (2NxnU).
    QuarterTop = 4,
    /// Asymmetric: a quarter-height bottom part (2NxnD).
    QuarterBottom = 5,
    /// Asymmetric: a quarter-width left part (nLx2N).
    QuarterLeft = 6,
    /// Asymmetric: a quarter-width right part (nRx2N).
    QuarterRight = 7,
    /// Sixteen 4x4 blocks (H.264 `I_4x4`).
    Sixteenth = 8,
    /// Raw samples (PCM).
    Raw = 9,
}

/// A codec-neutral class of intra prediction mode.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum IntraClass {
    /// Planar / plane / smooth.
    #[default]
    Planar = 0,
    /// DC.
    Dc = 1,
    /// A directional mode (including pure horizontal and vertical).
    Angular = 2,
    /// Anything else (e.g. palette, intra block copy).
    Other = 3,
}

/// [`BlockRecord::flags`] bits.
pub mod flags {
    /// Luma residual coded (the coded bits are bit `plane`).
    pub const CODED_Y: u8 = 1;
    /// Cb residual coded.
    pub const CODED_CB: u8 = 2;
    /// Cr residual coded.
    pub const CODED_CR: u8 = 4;
    /// Transform skipped (any plane of the block).
    pub const TRANSFORM_SKIP: u8 = 8;
    /// A QP delta was coded in this coding block.
    pub const QP_DELTA_CODED: u8 = 16;
    /// Bi-prediction.
    pub const BI: u8 = 32;
}

/// One block's decisions. Positions and sizes are luma samples; vectors
/// are in the units [`StreamStats::fact`] `mv_units_per_sample` gives.
/// Fields a kind does not use stay zero.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BlockRecord {
    /// What the record describes.
    pub kind: RecordKind,
    /// How the block is predicted.
    pub pred: Pred,
    /// Coding blocks: how they split into prediction blocks.
    pub shape: Shape,
    /// Coding-tree depth (coding blocks) or transform-tree depth.
    pub depth: u8,
    /// Top-left corner, luma samples.
    pub x: u16,
    /// Top-left corner, luma samples.
    pub y: u16,
    /// Width, luma samples.
    pub w: u16,
    /// Height, luma samples.
    pub h: u16,
    /// Intra: the luma mode (codec numbering). Merge / skip: the merge
    /// index. Inter: the predictor index. Filter: the luma filter type.
    pub mode: u8,
    /// Intra: the chroma mode (codec numbering). Filter: the chroma type.
    pub mode2: u8,
    /// Intra: [`IntraClass`]. Filter: how its parameters were signalled.
    pub class: u8,
    /// [`flags`].
    pub flags: u8,
    /// Luma QP in the codec's scale (H.264 / HEVC QP, AV1 / VP9 qindex).
    pub qp: i16,
    /// The QP delta coded in the block ([`flags::QP_DELTA_CODED`]).
    pub qp_delta: i8,
    /// Reference index per list (-1: unused).
    pub ref_idx: [i8; 2],
    /// Motion vector per list, `[x, y]`.
    pub mv: [[i16; 2]; 2],
    /// Coded motion vector difference per list.
    pub mvd: [[i16; 2]; 2],
    /// Non-zero coefficient levels (transform blocks).
    pub nonzero: u16,
    /// Entropy-coded bits the block's syntax took (coding: all of it;
    /// prediction: the motion vector differences; transform: the
    /// residual; filter: its parameters).
    pub bits: u32,
}

impl BlockRecord {
    /// Area in luma samples.
    #[must_use]
    pub fn area(&self) -> u64 {
        u64::from(self.w) * u64::from(self.h)
    }
}

/// Halfwords per record in [`pack_records`].
pub const PACKED_HALFWORDS: usize = 24;

/// The records as a flat `i16` array, [`PACKED_HALFWORDS`] per record,
/// for a viewer to read without per-record calls (`web/analysis.js` reads
/// the same layout): kind, pred, shape, depth, x, y, w, h, mode, mode2,
/// class, flags, qp, `qp_delta`, `ref_idx[0]`, `ref_idx[1]`, mv0 x, mv0 y,
/// mv1 x, mv1 y, mvd0 x, mvd0 y, nonzero (saturated), bits (saturated).
/// `out` is cleared first and reused by the caller.
pub fn pack_records(records: &[BlockRecord], out: &mut Vec<i16>) {
    let saturate = |v: u32| v.min(i16::MAX as u32) as i16;
    // Positions and sizes are below 32768 for pictures up to 8K.
    let coordinate = |v: u16| v.min(i16::MAX as u16) as i16;
    out.clear();
    out.reserve(records.len() * PACKED_HALFWORDS);
    for r in records {
        out.extend_from_slice(&[
            r.kind as i16,
            r.pred as i16,
            r.shape as i16,
            i16::from(r.depth),
            coordinate(r.x),
            coordinate(r.y),
            coordinate(r.w),
            coordinate(r.h),
            i16::from(r.mode),
            i16::from(r.mode2),
            i16::from(r.class),
            i16::from(r.flags),
            r.qp,
            i16::from(r.qp_delta),
            i16::from(r.ref_idx[0]),
            i16::from(r.ref_idx[1]),
            r.mv[0][0],
            r.mv[0][1],
            r.mv[1][0],
            r.mv[1][1],
            r.mvd[0][0],
            r.mvd[0][1],
            saturate(u32::from(r.nonzero)),
            saturate(r.bits),
        ]);
    }
}

/// How often a value occurred and its total weight.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Tally {
    /// Occurrences recorded.
    pub count: u64,
    /// Sum of their weights.
    pub weight: u64,
}

/// The tallies of one picture, keyed by statistic and value.
#[derive(Debug, Clone, Default)]
pub struct FrameStats {
    /// Statistic → value → tally.
    pub tallies: BTreeMap<&'static str, BTreeMap<i64, Tally>>,
}

impl FrameStats {
    /// Count one occurrence of `value` for statistic `what`, with `weight`.
    pub fn add(&mut self, what: &'static str, value: i64, weight: u64) {
        let tally = self
            .tallies
            .entry(what)
            .or_default()
            .entry(value)
            .or_default();
        tally.count += 1;
        tally.weight += weight;
    }

    /// The tally of `value` under statistic `what` (zero when absent).
    #[must_use]
    pub fn get(&self, what: &str, value: i64) -> Tally {
        self.tallies
            .get(what)
            .and_then(|values| values.get(&value))
            .copied()
            .unwrap_or_default()
    }

    /// As one JSON object `{"<what>": {"<value>": [count, weight]}}`.
    fn write_json(&self, out: &mut String) {
        out.push('{');
        for (j, (what, values)) in self.tallies.iter().enumerate() {
            let sep = if j == 0 { "" } else { ", " };
            let _ = write!(out, "{sep}\"{what}\": {{");
            for (k, (value, tally)) in values.iter().enumerate() {
                let sep = if k == 0 { "" } else { ", " };
                let _ = write!(out, "{sep}\"{value}\": [{}, {}]", tally.count, tally.weight);
            }
            out.push('}');
        }
        out.push('}');
    }
}

/// Analysis of a stream: the records of the picture being decoded and of
/// the last completed one, and the tallies of every completed picture.
#[derive(Debug, Clone)]
pub struct StreamStats {
    current: Vec<BlockRecord>,
    current_facts: FrameStats,
    completed: Vec<BlockRecord>,
    /// Per completed picture in decoding order: its facts and the
    /// aggregate of its records.
    pub frames: Vec<FrameStats>,
    started: bool,
    /// Whether completed pictures are aggregated into `frames` (else only
    /// the last picture's records are kept: bounded memory for a viewer).
    tallies: bool,
}

impl Default for StreamStats {
    fn default() -> Self {
        Self {
            current: Vec::new(),
            current_facts: FrameStats::default(),
            completed: Vec::new(),
            frames: Vec::new(),
            started: false,
            tallies: true,
        }
    }
}

impl StreamStats {
    /// Analysis that keeps only the last completed picture's records, no
    /// per-picture tallies (a viewer aggregates what it shows itself).
    #[must_use]
    pub fn records_only() -> Self {
        Self {
            tallies: false,
            ..Self::default()
        }
    }

    /// Start a new picture, completing the previous one.
    pub fn begin_frame(&mut self) {
        self.finish();
        self.started = true;
    }

    /// Complete the picture in progress, if any (also done by the next
    /// [`StreamStats::begin_frame`]).
    pub fn finish(&mut self) {
        if !self.started {
            return;
        }
        self.started = false;
        let mut frame = std::mem::take(&mut self.current_facts);
        if self.tallies {
            aggregate(&self.current, &mut frame);
            self.frames.push(frame);
        }
        std::mem::swap(&mut self.current, &mut self.completed);
        self.current.clear();
    }

    /// Record a picture-level fact (weight: e.g. bytes, or 0).
    pub fn fact(&mut self, what: &'static str, value: i64, weight: u64) {
        self.current_facts.add(what, value, weight);
    }

    /// Record one block.
    pub fn push(&mut self, record: BlockRecord) {
        self.current.push(record);
    }

    /// The records of the last completed picture (decoders complete a
    /// picture when they finish decoding it).
    #[must_use]
    pub fn records(&self) -> &[BlockRecord] {
        &self.completed
    }

    /// The per-picture tallies as JSON: `{"frames": [{"<what>":
    /// {"<value>": [count, weight], ...}, ...}, ...]}`.
    #[must_use]
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\"frames\": [");
        for (i, frame) in self.frames.iter().enumerate() {
            out.push_str(if i == 0 { "\n" } else { ",\n" });
            frame.write_json(&mut out);
        }
        out.push_str("\n]}\n");
        out
    }
}

/// The size of a block as one value: `w * SIZE_CODE_SCALE + h`.
pub const SIZE_CODE_SCALE: i64 = 1000;

/// A vector's size bucket in its units: 0 for zero, then
/// `1 + floor(log2(max(|x|, |y|)))`.
#[must_use]
pub fn size_bucket([x, y]: [i16; 2]) -> i64 {
    let largest = i32::from(x).unsigned_abs().max(i32::from(y).unsigned_abs());
    if largest == 0 {
        0
    } else {
        1 + i64::from(largest.ilog2())
    }
}

/// The fractional precision a vector of `units` per sample needs: 0
/// integer, then 1 + log2 of the finest fraction (1 half, 2 quarter, 3
/// eighth sample).
#[must_use]
pub fn precision([x, y]: [i16; 2], units: i32) -> i64 {
    let fraction = |v: i16| i32::from(v).rem_euclid(units);
    let bits = units.trailing_zeros();
    let finest = |f: i32| if f == 0 { 0 } else { bits - f.trailing_zeros() };
    i64::from(finest(fraction(x)).max(finest(fraction(y))))
}

/// Statistic keys of [`aggregate`]. Weights are luma areas unless noted.
pub mod keys {
    /// Coding blocks: size code, by prediction, depth, [`super::Pred`],
    /// [`super::Shape`] by intra / inter, QP, coded QP deltas (weight 0),
    /// bits (weight: bits, by prediction).
    pub const CB_SIZE: &str = "cb_size";
    /// `cb_size_by_pred`.
    pub const CB_SIZE_BY_PRED: [&str; 4] = [
        "cb_size_intra",
        "cb_size_inter",
        "cb_size_skip",
        "cb_size_merge",
    ];
    /// `cb_depth`.
    pub const CB_DEPTH: &str = "cb_depth";
    /// `cb_pred`.
    pub const CB_PRED: &str = "cb_pred";
    /// `cb_shape_intra`.
    pub const CB_SHAPE_INTRA: &str = "cb_shape_intra";
    /// `cb_shape_inter`.
    pub const CB_SHAPE_INTER: &str = "cb_shape_inter";
    /// `cb_qp`.
    pub const CB_QP: &str = "cb_qp";
    /// `cb_qp_delta`.
    pub const CB_QP_DELTA: &str = "cb_qp_delta";
    /// Coded planes of non-skipped coding blocks: value = `flags & 7`.
    pub const CB_CODED: &str = "cb_coded";
    /// `cb_bits`.
    pub const CB_BITS: &str = "cb_bits";
    /// Prediction blocks: [`super::Pred`], size code (inter blocks), merge index,
    /// predictor index, reference index, vector precision, zero vectors,
    /// vector and difference size buckets, bi-prediction, bits (weight:
    /// bits), intra luma mode, [`super::IntraClass`], chroma mode.
    pub const PB_PRED: &str = "pb_pred";
    /// `pb_size`.
    pub const PB_SIZE: &str = "pb_size";
    /// `merge_idx`.
    pub const MERGE_IDX: &str = "merge_idx";
    /// `mvp_idx`.
    pub const MVP_IDX: &str = "mvp_idx";
    /// `ref_idx`.
    pub const REF_IDX: &str = "ref_idx";
    /// `mv_precision`.
    pub const MV_PRECISION: &str = "mv_precision";
    /// `mv_zero`.
    pub const MV_ZERO: &str = "mv_zero";
    /// `mv_size`.
    pub const MV_SIZE: &str = "mv_size";
    /// `mvd_size`.
    pub const MVD_SIZE: &str = "mvd_size";
    /// `bi_pred`.
    pub const BI_PRED: &str = "bi_pred";
    /// `mvd_bits`.
    pub const MVD_BITS: &str = "mvd_bits";
    /// `intra_mode`.
    pub const INTRA_MODE: &str = "intra_mode";
    /// `intra_class`.
    pub const INTRA_CLASS: &str = "intra_class";
    /// `intra_chroma_mode`.
    pub const INTRA_CHROMA_MODE: &str = "intra_chroma_mode";
    /// Intra luma mode by prediction block size: value `size_code *
    /// INTRA_MODE_SCALE + mode`.
    pub const INTRA_MODE_BY_SIZE: &str = "intra_mode_by_size";
    /// The mode's multiplier in [`INTRA_MODE_BY_SIZE`] values.
    pub const INTRA_MODE_SCALE: i64 = 100;
    /// Intra blocks whose chroma mode number equals the luma one (HEVC:
    /// the derived mode, or an explicit mode that coincides): value 0 / 1.
    pub const INTRA_CHROMA_SAME: &str = "intra_chroma_same";
    /// Transform blocks of intra / inter coding blocks: size code, depth;
    /// coded flag per plane (weight: luma area), transform skip, non-zero
    /// levels (weight: levels), residual bits (weight: bits).
    pub const TB_SIZE: [&str; 2] = ["tb_size_intra", "tb_size_inter"];
    /// `tb_depth`.
    pub const TB_DEPTH: [&str; 2] = ["tb_depth_intra", "tb_depth_inter"];
    /// `tb_coded`.
    pub const TB_CODED: [&str; 3] = ["tb_coded_y", "tb_coded_cb", "tb_coded_cr"];
    /// `tb_transform_skip`.
    pub const TB_TRANSFORM_SKIP: &str = "tb_transform_skip";
    /// `tb_nonzero`.
    pub const TB_NONZERO: &str = "tb_nonzero";
    /// `tb_bits`.
    pub const TB_BITS: &str = "tb_bits";
    /// Filter units: luma and chroma type, signalling; bits (weight: bits).
    pub const FILTER_LUMA: &str = "filter_luma";
    /// `filter_chroma`.
    pub const FILTER_CHROMA: &str = "filter_chroma";
    /// `filter_signal`.
    pub const FILTER_SIGNAL: &str = "filter_signal";
    /// `filter_bits`.
    pub const FILTER_BITS: &str = "filter_bits";
}

/// The facts a decoder records that [`aggregate`] reads.
pub mod facts {
    /// Motion vector units per luma sample (4: quarter sample).
    pub const MV_UNITS_PER_SAMPLE: &str = "mv_units_per_sample";
}

/// Default motion vector units per sample when no fact says otherwise.
const DEFAULT_MV_UNITS: i32 = 4;

/// Add the tallies of a picture's records to `frame`.
pub fn aggregate(records: &[BlockRecord], frame: &mut FrameStats) {
    let units = frame
        .tallies
        .get(facts::MV_UNITS_PER_SAMPLE)
        .and_then(|values| values.keys().next())
        .map_or(DEFAULT_MV_UNITS, |&u| u as i32);
    for r in records {
        let area = r.area();
        let size = i64::from(r.w) * SIZE_CODE_SCALE + i64::from(r.h);
        match r.kind {
            RecordKind::Coding => coding(frame, r, area, size),
            RecordKind::Prediction => prediction(frame, r, area, size, units),
            RecordKind::Transform => transform(frame, r, area, size),
            RecordKind::Filter => {
                frame.add(keys::FILTER_LUMA, i64::from(r.mode), area);
                frame.add(keys::FILTER_CHROMA, i64::from(r.mode2), area);
                frame.add(keys::FILTER_SIGNAL, i64::from(r.class), area);
                frame.add(keys::FILTER_BITS, 0, u64::from(r.bits));
            }
        }
    }
}

fn coding(frame: &mut FrameStats, r: &BlockRecord, area: u64, size: i64) {
    let pred = r.pred as usize;
    frame.add(keys::CB_SIZE, size, area);
    frame.add(keys::CB_SIZE_BY_PRED[pred], size, area);
    frame.add(keys::CB_DEPTH, i64::from(r.depth), area);
    frame.add(keys::CB_PRED, pred as i64, area);
    match r.pred {
        Pred::Intra => frame.add(keys::CB_SHAPE_INTRA, r.shape as i64, area),
        Pred::Inter | Pred::Merge => frame.add(keys::CB_SHAPE_INTER, r.shape as i64, area),
        Pred::Skip => {}
    }
    if r.pred != Pred::Skip {
        let planes = flags::CODED_Y | flags::CODED_CB | flags::CODED_CR;
        frame.add(keys::CB_CODED, i64::from(r.flags & planes), area);
    }
    frame.add(keys::CB_QP, i64::from(r.qp), area);
    if r.flags & flags::QP_DELTA_CODED != 0 {
        frame.add(keys::CB_QP_DELTA, i64::from(r.qp_delta), 0);
    }
    frame.add(keys::CB_BITS, pred as i64, u64::from(r.bits));
}

fn prediction(frame: &mut FrameStats, r: &BlockRecord, area: u64, size: i64, units: i32) {
    frame.add(keys::PB_PRED, r.pred as i64, area);
    if r.pred == Pred::Intra {
        frame.add(keys::INTRA_CHROMA_SAME, i64::from(r.mode2 == r.mode), area);
        frame.add(keys::INTRA_MODE, i64::from(r.mode), area);
        frame.add(
            keys::INTRA_MODE_BY_SIZE,
            size * keys::INTRA_MODE_SCALE + i64::from(r.mode),
            area,
        );
        frame.add(keys::INTRA_CLASS, i64::from(r.class), area);
        frame.add(keys::INTRA_CHROMA_MODE, i64::from(r.mode2), area);
        return;
    }
    match r.pred {
        Pred::Skip | Pred::Merge => frame.add(keys::MERGE_IDX, i64::from(r.mode), area),
        _ => frame.add(keys::MVP_IDX, i64::from(r.mode), area),
    }
    frame.add(keys::PB_SIZE, size, area);
    frame.add(keys::BI_PRED, i64::from(r.flags & flags::BI != 0), area);
    for list in 0..2 {
        if r.ref_idx[list] < 0 {
            continue;
        }
        let mv = r.mv[list];
        frame.add(keys::REF_IDX, i64::from(r.ref_idx[list]), area);
        frame.add(keys::MV_PRECISION, precision(mv, units), area);
        frame.add(keys::MV_ZERO, i64::from(mv == [0, 0]), area);
        frame.add(keys::MV_SIZE, size_bucket(mv), area);
        if r.pred == Pred::Inter {
            frame.add(keys::MVD_SIZE, size_bucket(r.mvd[list]), area);
        }
    }
    frame.add(keys::MVD_BITS, 0, u64::from(r.bits));
}

fn transform(frame: &mut FrameStats, r: &BlockRecord, area: u64, size: i64) {
    let inter = usize::from(r.pred != Pred::Intra);
    frame.add(keys::TB_SIZE[inter], size, area);
    frame.add(keys::TB_DEPTH[inter], i64::from(r.depth), area);
    for (plane, key) in keys::TB_CODED.iter().enumerate() {
        frame.add(key, i64::from(r.flags >> plane & 1), area);
    }
    frame.add(
        keys::TB_TRANSFORM_SKIP,
        i64::from(r.flags & flags::TRANSFORM_SKIP != 0),
        area,
    );
    frame.add(keys::TB_NONZERO, 0, u64::from(r.nonzero));
    frame.add(keys::TB_BITS, 0, u64::from(r.bits));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_aggregate_per_frame_and_write_json() {
        let mut stats = StreamStats::default();
        stats.begin_frame();
        stats.fact("slice_qp", 30, 0);
        let block = BlockRecord {
            w: 16,
            h: 16,
            qp: 30,
            ..BlockRecord::default()
        };
        stats.push(block);
        stats.push(block);
        stats.begin_frame();
        stats.finish();
        assert_eq!(stats.frames.len(), 2);
        assert_eq!(stats.records(), &[]);
        assert_eq!(
            stats.frames[0].get(keys::CB_SIZE, 16016),
            Tally {
                count: 2,
                weight: 512
            }
        );
        assert_eq!(stats.frames[1].get(keys::CB_SIZE, 16016), Tally::default());
        assert!(
            stats
                .to_json()
                .starts_with("{\"frames\": [\n{\"cb_bits\": {\"0\": [2, 0]}")
        );
    }

    #[test]
    fn packs_records_in_the_documented_layout() {
        let record = BlockRecord {
            kind: RecordKind::Prediction,
            pred: Pred::Merge,
            x: 16,
            y: 8,
            w: 32,
            h: 16,
            mode: 1,
            qp: -3,
            ref_idx: [0, -1],
            mv: [[-5, 7], [0, 0]],
            bits: 100_000,
            ..BlockRecord::default()
        };
        let mut out = vec![9; 3];
        pack_records(&[record, record], &mut out);
        assert_eq!(out.len(), 2 * PACKED_HALFWORDS);
        assert_eq!(
            out[..PACKED_HALFWORDS],
            [
                1,
                3,
                0,
                0,
                16,
                8,
                32,
                16,
                1,
                0,
                0,
                0,
                -3,
                0,
                0,
                -1,
                -5,
                7,
                0,
                0,
                0,
                0,
                0,
                i16::MAX
            ]
        );
    }

    #[test]
    fn buckets_and_precision() {
        assert_eq!(size_bucket([0, 0]), 0);
        assert_eq!(size_bucket([1, 0]), 1);
        assert_eq!(size_bucket([-4, 3]), 3);
        assert_eq!(precision([-8, 4], 4), 0);
        assert_eq!(precision([-6, 4], 4), 1);
        assert_eq!(precision([2, -3], 4), 2);
        assert_eq!(precision([1, 0], 8), 3);
    }
}
