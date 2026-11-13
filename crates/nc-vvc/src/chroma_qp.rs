//! H.266 §7.3.2.4 / §7.4.3.4 chroma QP mapping table: syntax and derivation.
//!
//! VVC signals the luma-to-chroma QP map in the SPS as a piecewise-linear
//! curve. We send one table shared by Cb and Cr, with one point per QP step
//! wherever it departs from the identity, so any non-decreasing map whose
//! steps are 0 or 1 is exact. The encoder's map reproduces HEVC's fixed 4:2:0
//! table (HEVC §8.6.1, Table 8-10) behind a chroma QP offset, so chroma keeps
//! the share of bits HEVC's tuned configuration gives it. Derivation checked
//! against `VVenC` `ChromaQpMappingTable` (BSD-3-Clause-Clear; `THIRD_PARTY.md`).
use crate::headers::MAX_QP;
use nc_common::{
    BitWriter,
    decode::{BitReader, DecodeError, Result},
};

/// Table entries: one per QP 0..=63 (8-bit video: `QpBdOffset` is 0).
const ENTRIES: usize = MAX_QP as usize + 1;
/// `sps_qp_table_start_minus26` is relative to 26.
const START_BIAS: i32 = 26;
/// HEVC Table 8-10: `QPc` for `qPi` 30..=43; below it `QPc` = `qPi`, above it `qPi` − 6.
const HEVC_TABLE_FIRST: i32 = 30;
const HEVC_TABLE: [i32; 14] = [29, 30, 31, 32, 33, 33, 34, 34, 35, 35, 36, 36, 37, 37];
/// HEVC's offset of `QPc` below `qPi` above its table.
const HEVC_HIGH_OFFSET: i32 = 6;
/// The smallest offset at which [`ChromaQpTable::hevc`] is the identity:
/// HEVC's map never lowers a QP by more than [`HEVC_HIGH_OFFSET`].
#[cfg(test)]
const IDENTITY_OFFSET: i32 = HEVC_HIGH_OFFSET;

/// `ChromaQpTable[0]`: the chroma QP of each luma QP (before chroma offsets).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChromaQpTable([u8; ENTRIES]);

impl ChromaQpTable {
    /// The identity map.
    #[cfg(test)]
    pub(crate) fn identity() -> Self {
        Self(std::array::from_fn(|qp| qp as u8))
    }

    /// HEVC's 4:2:0 map of `qp + offset` (HEVC §8.6.1, with
    /// `pps_cb_qp_offset` = `offset`), continued with slope one above HEVC's
    /// range, never above `qp` itself and clipped to VVC's 0..=63.
    ///
    /// The SPS table passes through its start point unchanged (§7.4.3.4:
    /// `qpOutVal[0] = qpInVal[0]`) and steps down by one below it, so it
    /// cannot map a QP above itself (coarser chroma would need
    /// `pps_cb_qp_offset`). Positive offsets therefore only soften HEVC's
    /// finer chroma at high QP; `IDENTITY_OFFSET` and above give the identity.
    pub(crate) fn hevc(offset: i32) -> Self {
        Self(std::array::from_fn(|qp| {
            let qpi = qp as i32 + offset;
            let mapped = if qpi < HEVC_TABLE_FIRST {
                qpi
            } else if let Some(&value) = HEVC_TABLE.get((qpi - HEVC_TABLE_FIRST) as usize) {
                value
            } else {
                qpi - HEVC_HIGH_OFFSET
            };
            mapped.clamp(0, qp as i32) as u8
        }))
    }

    /// The engine chroma policy's map (`nc_encoder::chroma_policy`): chroma
    /// at luma's QP up to QP 36, then up to 3 QP finer (the JVET CTC map),
    /// rounded to whole QPs and, like [`Self::hevc`], never above `qp`
    /// (a positive `chroma.offset` above the curve would need
    /// `pps_cb_qp_offset`; the policy's default 0 does not).
    pub(crate) fn policy(policy: nc_encoder::chroma_policy::ChromaPolicy) -> Self {
        Self(std::array::from_fn(|qp| {
            let mapped = policy.chroma_q(qp as f64).round() as i32;
            mapped.clamp(0, qp as i32) as u8
        }))
    }

    /// The chroma QP of luma QP `qp`.
    pub(crate) fn map(&self, qp: u32) -> u32 {
        u32::from(self.0[qp.min(MAX_QP) as usize])
    }

    /// `sps_qp_table_start_minus26`, `sps_num_points_in_qp_table_minus1` and
    /// the points, after `sps_same_qp_table_for_chroma_flag` = 1.
    ///
    /// The derivation starts at a point that maps to itself and, below it,
    /// steps down by one; above the last point it steps up by one. So the
    /// first point is the end of the table's identity prefix and the last is
    /// where its final slope-one run begins; every step between is a point.
    pub(crate) fn write(&self, w: &mut BitWriter) {
        let table = self.0.map(i32::from);
        let identity_prefix = table
            .iter()
            .enumerate()
            .take_while(|&(qp, &mapped)| mapped == qp as i32)
            .count();
        assert!(identity_prefix > 0, "the table must map QP 0 to 0");
        let start = (identity_prefix - 1).min(ENTRIES - 2);
        let slope_one_from = (start + 1..ENTRIES)
            .rev()
            .take_while(|&qp| table[qp] == (table[qp - 1] + 1).min(MAX_QP as i32))
            .last()
            .unwrap_or(ENTRIES)
            .max(start + 2);
        let end = slope_one_from - 1;
        w.put_se(start as i32 - START_BIAS);
        w.put_ue((end - start - 1) as u32);
        for qp in start..end {
            // One QP per point: sps_delta_qp_in_val_minus1 = 0, so
            // sps_delta_qp_diff_val (XOR 0) is the output step itself.
            w.put_ue(0);
            w.put_ue((table[qp + 1] - table[qp]) as u32);
        }
    }

    /// Parse the syntax [`write`](Self::write) emits (any valid points) and
    /// derive the table (§7.4.3.4, equations 57-60).
    pub(crate) fn parse(r: &mut BitReader<'_>) -> Result<Self> {
        // qp_table_start_minus26 is range-checked before the bias is added,
        // so a value near i32::MAX cannot overflow (§7.4.3.4).
        let start_minus26 = r.read_se()?;
        if !(-START_BIAS..MAX_QP as i32 - START_BIAS).contains(&start_minus26) {
            return Err(DecodeError::Malformed("VVC chroma QP table start"));
        }
        let start = START_BIAS + start_minus26;
        let points = r.read_ue_max(MAX_QP, "VVC chroma QP table points")? as usize + 1;
        if points > (MAX_QP as i32 - start) as usize {
            return Err(DecodeError::Malformed("VVC chroma QP table points"));
        }
        let mut qp_in = vec![start];
        let mut qp_out = vec![start];
        let mut deltas_in = Vec::with_capacity(points);
        for _ in 0..points {
            let delta_in_minus1 = r.read_ue()?;
            let diff = r.read_ue()?;
            if delta_in_minus1 > MAX_QP || diff > MAX_QP {
                return Err(DecodeError::Malformed("VVC chroma QP table delta"));
            }
            let (last_in, last_out) = (qp_in[qp_in.len() - 1], qp_out[qp_out.len() - 1]);
            let next_in = last_in + delta_in_minus1 as i32 + 1;
            let next_out = last_out + (delta_in_minus1 ^ diff) as i32;
            if next_in > MAX_QP as i32 || next_out > MAX_QP as i32 {
                return Err(DecodeError::Malformed("VVC chroma QP table range"));
            }
            qp_in.push(next_in);
            qp_out.push(next_out);
            deltas_in.push(delta_in_minus1 as i32 + 1);
        }
        let mut table = [0i32; ENTRIES];
        let first = start as usize;
        table[first] = start;
        for qp in (0..first).rev() {
            table[qp] = (table[qp + 1] - 1).clamp(0, MAX_QP as i32);
        }
        for (j, &delta_in) in deltas_in.iter().enumerate() {
            let rounding = delta_in >> 1;
            let base = table[qp_in[j] as usize];
            for (m, qp) in (qp_in[j] + 1..=qp_in[j + 1]).enumerate() {
                let step = (qp_out[j + 1] - qp_out[j]) * (m as i32 + 1) + rounding;
                table[qp as usize] = base + step / delta_in;
            }
        }
        for qp in qp_in[points] as usize + 1..ENTRIES {
            table[qp] = (table[qp - 1] + 1).clamp(0, MAX_QP as i32);
        }
        Ok(Self(table.map(|mapped| mapped as u8)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(table: ChromaQpTable) -> ChromaQpTable {
        let mut w = BitWriter::new();
        table.write(&mut w);
        w.trailing_bits();
        let bytes = w.into_bytes();
        ChromaQpTable::parse(&mut BitReader::new(&bytes)).expect("valid table")
    }

    #[test]
    fn policy_map_follows_luma_then_gets_three_finer() {
        let table = ChromaQpTable::policy(nc_encoder::chroma_policy::ChromaPolicy { offset: 0.0 });
        for (qpi, qpc) in [(0, 0), (36, 36), (38, 37), (42, 39), (51, 48), (63, 60)] {
            assert_eq!(table.map(qpi), qpc, "qPi {qpi}");
        }
        assert_eq!(round_trip(table), table);
    }

    #[test]
    fn hevc_map_matches_hevc_table_8_10() {
        let table = ChromaQpTable::hevc(0);
        for (qpi, qpc) in [
            (0, 0),
            (29, 29),
            (30, 29),
            (35, 33),
            (43, 37),
            (44, 38),
            (51, 45),
        ] {
            assert_eq!(table.map(qpi), qpc, "qPi {qpi}");
        }
        let offset = ChromaQpTable::hevc(-3);
        assert_eq!((offset.map(2), offset.map(33), offset.map(63)), (0, 29, 54));
        let softened = ChromaQpTable::hevc(3);
        assert_eq!((softened.map(20), softened.map(63)), (20, 60));
        for offset in IDENTITY_OFFSET..=12 {
            assert_eq!(ChromaQpTable::hevc(offset), ChromaQpTable::identity());
        }
    }

    #[test]
    fn written_tables_derive_back_exactly() {
        for table in [ChromaQpTable::identity()]
            .into_iter()
            .chain((-12..=12).map(ChromaQpTable::hevc))
        {
            assert_eq!(round_trip(table), table);
        }
    }

    #[test]
    fn interpolated_points_follow_the_derivation() {
        // VTM's SDR table: points (17,17) (22,23) (34,35) (42,39), coded as
        // start 17 and (delta_in_minus1, delta_out) (4, 6), (11, 12), (7, 4).
        let mut w = BitWriter::new();
        w.put_se(17 - START_BIAS);
        w.put_ue(2);
        for (delta_in_minus1, delta_out) in [(4u32, 6u32), (11, 12), (7, 4)] {
            w.put_ue(delta_in_minus1);
            w.put_ue(delta_in_minus1 ^ delta_out);
        }
        w.trailing_bits();
        let bytes = w.into_bytes();
        let table = ChromaQpTable::parse(&mut BitReader::new(&bytes)).unwrap();
        assert_eq!(table.map(16), 16);
        assert_eq!(table.map(22), 23);
        assert_eq!(table.map(34), 35);
        assert_eq!(table.map(42), 39);
        assert_eq!(table.map(43), 40);
        // Between points: 17 + (6 * m + 2) / 5.
        assert_eq!(table.map(18), 17 + (6 + 2) / 5);
    }

    /// A start offset near `i32::MAX` used to overflow when biased.
    #[test]
    fn start_is_range_checked_before_the_bias() {
        let mut w = BitWriter::new();
        w.put_se(i32::MAX);
        w.put_ue(0);
        w.trailing_bits();
        let bytes = w.into_bytes();
        assert_eq!(
            ChromaQpTable::parse(&mut BitReader::new(&bytes)).err(),
            Some(DecodeError::Malformed("VVC chroma QP table start"))
        );
    }
}
