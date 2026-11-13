//! H.266 §9.3 rectangular residuals and dependent-quantization state transitions.
//! Shares the existing CABAC engine and ordinary coefficient context banks.
//! Additional state contexts follow BSD-Clear `VVdeC` (`THIRD_PARTY.md`).
use super::initialization::bank;
use crate::{coeff_contexts::CoeffContexts, contexts::Context, decoder_cabac::Reader};
use nc_common::decode::{DecodeError, Result};

const GROUP_START: [usize; 10] = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24];
const STATE_TRANSITIONS: u32 = 32_040;

pub(super) struct Models {
    pub common: CoeffContexts,
    extra_significance: [[[Context; 12]; 2]; 2],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            common: super::initialization::coefficients(qp, kind),
            extra_significance: [
                [
                    bank(
                        [
                            [26, 45, 53, 46, 49, 54, 61, 39, 35, 39, 39, 39],
                            [19, 38, 38, 46, 34, 54, 54, 39, 6, 39, 39, 39],
                            [11, 38, 46, 54, 27, 39, 39, 39, 44, 39, 39, 39],
                        ],
                        [9, 13, 8, 8, 8, 8, 8, 5, 8, 0, 0, 0],
                        qp,
                        kind,
                    ),
                    bank(
                        [
                            [19, 54, 39, 39, 50, 39, 39, 39, 0, 39, 39, 39],
                            [19, 39, 54, 39, 19, 39, 39, 39, 56, 39, 39, 39],
                            [18, 39, 39, 39, 27, 39, 39, 39, 0, 39, 39, 39],
                        ],
                        [8, 8, 8, 8, 8, 0, 4, 4, 0, 0, 0, 0],
                        qp,
                        kind,
                    ),
                ],
                [
                    bank(
                        [
                            [34, 45, 38, 31, 58, 39, 39, 39],
                            [35, 45, 53, 54, 44, 39, 39, 39],
                            [19, 46, 38, 39, 52, 39, 39, 39],
                        ],
                        [8, 12, 12, 8, 4, 0, 0, 0],
                        qp,
                        kind,
                    ),
                    bank(
                        [
                            [34, 38, 54, 39, 41, 39, 39, 39],
                            [34, 38, 62, 39, 26, 39, 39, 39],
                            [11, 39, 39, 39, 19, 39, 39, 39],
                        ],
                        [8, 8, 8, 8, 4, 0, 0, 0],
                        qp,
                        kind,
                    ),
                ],
            ],
        }
    }
    fn significance(&mut self, channel: usize, state: u32, index: usize) -> &mut Context {
        if state < 2 {
            &mut self.common.sig[channel][index]
        } else {
            &mut self.extra_significance[channel][state as usize - 2][index]
        }
    }
}

#[derive(Debug)]
pub(super) struct Scan {
    pub(super) positions: Vec<usize>,
    pub(super) groups: Vec<usize>,
    pub(super) group_size: usize,
    pub(super) columns: usize,
    pub(super) rows: usize,
}
impl Scan {
    pub(super) fn new(width: usize, height: usize) -> Self {
        // Narrow chroma transforms use 2x2 or 2x8 groups; the ordinary
        // square path keeps the encoder's precomputed 4x4-group scans.
        let (gw, gh) = match (width, height) {
            (1, _) => (1, height.min(16)),
            (_, 1) => (width.min(16), 1),
            (2, 2 | 4) | (4, 2) => (2, 2),
            (2, _) => (2, 8),
            (_, 2) => (8, 2),
            _ => (4, 4),
        };
        let columns = width.min(32) / gw;
        let rows = height.min(32) / gh;
        let groups = diagonal(columns, rows);
        let local = diagonal(gw, gh);
        let positions = groups
            .iter()
            .flat_map(|&group| {
                local.iter().map(move |&position| {
                    (group / columns * gh + position / gw) * width
                        + group % columns * gw
                        + position % gw
                })
            })
            .collect();
        Self {
            positions,
            groups,
            group_size: gw * gh,
            columns,
            rows,
        }
    }
}
fn diagonal(width: usize, height: usize) -> Vec<usize> {
    let mut order = Vec::with_capacity(width * height);
    for sum in 0..width + height - 1 {
        for x in sum.saturating_sub(height - 1)..=sum.min(width - 1) {
            order.push((sum - x) * width + x);
        }
    }
    order
}
/// Last occupied grouped-scan position for CU transform-tool constraints.
pub(super) fn last_scan(levels: &[i32], width: usize, height: usize) -> Option<usize> {
    Scan::new(width, height)
        .positions
        .iter()
        .rposition(|&p| levels[p] != 0)
}
fn last_prefix(
    r: &mut Reader<'_>,
    models: &mut [Context],
    side: usize,
    channel: usize,
    limit_side: usize,
) -> Result<usize> {
    let log2 = side.ilog2() as usize;
    let offset = if channel == 0 {
        [0, 0, 0, 3, 6, 10, 15][log2]
    } else {
        0
    };
    let shift = if channel == 0 {
        (log2 + 1) >> 2
    } else {
        (side >> 3).min(2)
    };
    let limit = GROUP_START
        .iter()
        .rposition(|&v| v < limit_side.min(32))
        .unwrap();
    let mut group = 0;
    while group < limit && r.bin(&mut models[offset + (group >> shift)])? {
        group += 1;
    }
    Ok(group)
}
fn last_suffix(r: &mut Reader<'_>, group: usize) -> Result<usize> {
    Ok(GROUP_START[group]
        + if group > 3 {
            r.bypass_bits(((group - 2) >> 1) as u32)? as usize
        } else {
            0
        })
}
fn last_position(
    r: &mut Reader<'_>,
    models: &mut Models,
    width: usize,
    height: usize,
    channel: usize,
    scan: &Scan,
    zero_out: bool,
) -> Result<usize> {
    let gx = last_prefix(
        r,
        &mut models.common.last_x[channel],
        width,
        channel,
        if zero_out { width.min(16) } else { width },
    )?;
    let gy = last_prefix(
        r,
        &mut models.common.last_y[channel],
        height,
        channel,
        if zero_out { height.min(16) } else { height },
    )?;
    let x = last_suffix(r, gx)?;
    let y = last_suffix(r, gy)?;
    scan.positions
        .iter()
        .position(|&p| p == y * width + x)
        .ok_or(DecodeError::Malformed("VVC last coefficient position"))
}
fn neighbours(
    values: &[i32],
    pos: usize,
    width: usize,
    height: usize,
) -> impl Iterator<Item = u32> + '_ {
    let (x, y) = (pos % width, pos / width);
    [(1, 0), (2, 0), (0, 1), (1, 1), (0, 2)]
        .into_iter()
        .filter(move |&(dx, dy)| x + dx < width && y + dy < height)
        .map(move |(dx, dy)| values[pos + dy * width + dx].unsigned_abs())
}
fn context(
    values: &[i32],
    pos: usize,
    width: usize,
    height: usize,
    channel: usize,
) -> (usize, usize) {
    let (mut sum, mut count) = (0, 0);
    for value in neighbours(values, pos, width, height) {
        sum += value.min(4 + (value & 1));
        count += u32::from(value != 0);
    }
    let diagonal = pos % width + pos / width;
    let sig = ((sum + 1) >> 1).min(3) as usize
        + usize::from(diagonal < 2) * 4
        + usize::from(channel == 0 && diagonal < 5) * 4;
    let level = (sum - count).min(4) as usize
        + 1
        + if diagonal == 0 {
            if channel == 0 { 15 } else { 5 }
        } else if channel == 0 {
            if diagonal < 3 {
                10
            } else if diagonal < 10 {
                5
            } else {
                0
            }
        } else {
            0
        };
    (sig, level)
}
fn rice(values: &[i32], pos: usize, width: usize, height: usize, base: u32) -> u32 {
    match neighbours(values, pos, width, height)
        .sum::<u32>()
        .saturating_sub(5 * base)
        .min(31)
    {
        0..=6 => 0,
        7..=13 => 1,
        14..=27 => 2,
        _ => 3,
    }
}
pub(super) fn remainder(r: &mut Reader<'_>, rice: u32) -> Result<u32> {
    const CUTOFF: u32 = 5;
    const MAX_ESCAPE: u32 = 12;
    let mut prefix = 0;
    while prefix < CUTOFF + MAX_ESCAPE && r.bypass()? {
        prefix += 1;
    }
    if prefix < CUTOFF {
        return Ok((prefix << rice) + r.bypass_bits(rice)?);
    }
    let escape = prefix - CUTOFF;
    let base = (CUTOFF + (1 << escape) - 1) << rice;
    Ok(base
        + r.bypass_bits(if escape == MAX_ESCAPE {
            15
        } else {
            escape + rice
        })?)
}

pub(super) fn read(
    r: &mut Reader<'_>,
    models: &mut Models,
    width: usize,
    height: usize,
    channel: usize,
    dependent: bool,
    hiding: bool,
    zero_out: bool,
) -> Result<Vec<i32>> {
    if ![1, 2, 4, 8, 16, 32, 64].contains(&width)
        || ![1, 2, 4, 8, 16, 32, 64].contains(&height)
        || channel > 1
    {
        return Err(DecodeError::Unsupported("VVC residual geometry"));
    }
    if !zero_out && !dependent && hiding && width == height && (4..=32).contains(&width) {
        return crate::coeff::code(
            r,
            &mut models.common,
            &vec![0; width * height],
            width,
            channel,
        );
    }
    read_general(
        r, models, width, height, channel, dependent, hiding, zero_out,
    )
}
fn read_general(
    r: &mut Reader<'_>,
    models: &mut Models,
    width: usize,
    height: usize,
    channel: usize,
    dependent: bool,
    hiding: bool,
    zero_out: bool,
) -> Result<Vec<i32>> {
    let scan = Scan::new(width, height);
    let last = last_position(r, models, width, height, channel, &scan, zero_out)?;
    let mut walk = Walk {
        values: vec![0; width * height],
        adjustment: vec![0; width * height],
        negative: vec![false; width * height],
        state: 0,
        budget: width.min(32) * height.min(32) * 28 / 16,
        width,
        height,
        channel,
        dependent,
        hiding,
    };
    let mut significant = vec![false; scan.columns * scan.rows];
    for group in (0..=last / scan.group_size).rev() {
        let raster = scan.groups[group];
        if zero_out
            && ((width == 32 && raster % scan.columns >= 4)
                || (height == 32 && raster / scan.columns >= 4))
        {
            continue;
        }
        let adjacent = (raster % scan.columns + 1 < scan.columns && significant[raster + 1])
            || (raster / scan.columns + 1 < scan.rows && significant[raster + scan.columns]);
        if group != 0
            && group != last / scan.group_size
            && !r.bin(&mut models.common.group[channel][usize::from(adjacent)])?
        {
            continue;
        }
        significant[raster] = true;
        walk.group(r, models, &scan, group, last)?;
    }
    for ((value, adjustment), negative) in walk
        .values
        .iter_mut()
        .zip(walk.adjustment)
        .zip(walk.negative)
    {
        if dependent {
            *value = 2 * *value - i32::from(adjustment);
        }
        if negative {
            *value = -*value;
        }
        if !(-32768..=32767).contains(value) {
            return Err(DecodeError::Malformed("VVC coefficient dynamic range"));
        }
    }
    Ok(walk.values)
}
struct Walk {
    values: Vec<i32>,
    adjustment: Vec<u8>,
    negative: Vec<bool>,
    state: u32,
    budget: usize,
    width: usize,
    height: usize,
    channel: usize,
    dependent: bool,
    hiding: bool,
}
impl Walk {
    fn advance(&mut self, parity: u32) {
        self.state = if self.dependent {
            (STATE_TRANSITIONS >> (4 * self.state + 2 * parity)) & 3
        } else {
            0
        };
    }
    fn group(
        &mut self,
        r: &mut Reader<'_>,
        models: &mut Models,
        scan: &Scan,
        group: usize,
        last: usize,
    ) -> Result<()> {
        let first = group * scan.group_size;
        let end = (first + scan.group_size - 1).min(last);
        let mut nonzero = Vec::with_capacity(scan.group_size);
        let mut regular = Vec::with_capacity(scan.group_size);
        let mut bypass = Vec::with_capacity(scan.group_size);
        for index in (first..=end).rev() {
            let pos = scan.positions[index];
            if self.budget < 4 {
                bypass.push((index, pos));
                continue;
            }
            let (sig, mut level) =
                context(&self.values, pos, self.width, self.height, self.channel);
            let inferred = index == last || (index == first && group != 0 && nonzero.is_empty());
            let present = if inferred {
                true
            } else {
                self.budget -= 1;
                r.bin(models.significance(self.channel, self.state, sig))?
            };
            if present {
                if index == last {
                    level = 0;
                }
                self.adjustment[pos] = (self.state >> 1) as u8;
                self.budget -= 1;
                self.values[pos] = if r.bin(&mut models.common.gt1[self.channel][level])? {
                    self.budget -= 2;
                    let parity = r.bin(&mut models.common.parity[self.channel][level])?;
                    let greater = r.bin(&mut models.common.gt2[self.channel][level])?;
                    2 + i32::from(parity) + 2 * i32::from(greater)
                } else {
                    1
                };
                regular.push(pos);
                nonzero.push((index, pos));
            }
            self.advance(self.values[pos] as u32 & 1);
        }
        for pos in regular {
            if self.values[pos] >= 4 {
                let parameter = rice(&self.values, pos, self.width, self.height, 4);
                self.values[pos] += 2 * remainder(r, parameter)? as i32;
            }
        }
        for (index, pos) in bypass {
            let parameter = rice(&self.values, pos, self.width, self.height, 0);
            let zero = (if self.state < 2 { 1 } else { 2 }) << parameter;
            let rem = remainder(r, parameter)?;
            self.values[pos] = match rem.cmp(&zero) {
                std::cmp::Ordering::Equal => 0,
                std::cmp::Ordering::Less => rem as i32 + 1,
                std::cmp::Ordering::Greater => rem as i32,
            };
            if self.values[pos] != 0 {
                self.adjustment[pos] = (self.state >> 1) as u8;
                nonzero.push((index, pos));
            }
            self.advance(self.values[pos] as u32 & 1);
        }
        let hidden = self.hiding
            && !self.dependent
            && nonzero
                .first()
                .zip(nonzero.last())
                .is_some_and(|(a, b)| a.0 - b.0 > crate::coeff::SIGN_HIDING_THRESHOLD);
        let sum = nonzero.iter().map(|&(_, p)| self.values[p]).sum::<i32>();
        for (index, &(_, pos)) in nonzero.iter().enumerate() {
            self.negative[pos] = if hidden && index + 1 == nonzero.len() {
                sum & 1 != 0
            } else {
                r.bypass()?
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cabac::CabacWriter;

    #[test]
    fn luma_scans_match_the_encoder_tables() {
        let narrow = [(1, 16), (1, 32), (2, 8), (2, 16), (2, 32)];
        for (width, height) in narrow.into_iter().chain(narrow.map(|(w, h)| (h, w))) {
            let scan = Scan::new(width, height);
            assert_eq!(scan.positions, crate::scan::rect_positions(width, height));
            assert_eq!(scan.groups, crate::scan::rect_groups(width, height));
        }
    }

    #[test]
    fn rectangular_scans_cover_the_effective_transform() {
        for width in [2, 4, 8, 16, 32, 64] {
            for height in [2, 4, 8, 16, 32, 64] {
                let scan = Scan::new(width, height);
                let mut positions = scan.positions.clone();
                positions.sort_unstable();
                let expected: Vec<_> = (0..height.min(32))
                    .flat_map(|y| (0..width.min(32)).map(move |x| y * width + x))
                    .collect();
                assert_eq!(positions, expected, "{width}x{height}");
                assert_eq!(scan.group_size * scan.groups.len(), positions.len());
            }
        }
    }
    #[test]
    fn general_reader_matches_shared_square_syntax() {
        for side in [4, 8, 16, 32] {
            for inter in [false, true] {
                for channel in 0..2 {
                    for pattern in 0..4 {
                        let mut source: Vec<i32> = (0..side * side)
                            .map(|i| match pattern {
                                0 if i == side * side - 1 => -32767,
                                1 if i % 17 == 0 => 32767,
                                2 => i as i32 % 11 - 5,
                                3 => {
                                    if i % 2 == 0 {
                                        -32767
                                    } else {
                                        32767
                                    }
                                }
                                _ => 0,
                            })
                            .collect();
                        // Give the shared writer legal hidden signs, without
                        // consulting the new decoder's scan or state machinery.
                        for group in crate::scan::positions(side).chunks(16) {
                            let first = group.iter().position(|&p| source[p] != 0);
                            let last = group.iter().rposition(|&p| source[p] != 0);
                            if let (Some(first), Some(last)) = (first, last) {
                                let sum: i32 = group.iter().map(|&p| source[p].abs()).sum();
                                if last - first > crate::coeff::SIGN_HIDING_THRESHOLD {
                                    source[group[first]] = source[group[first]].abs()
                                        * if sum % 2 == 1 { -1 } else { 1 };
                                }
                            }
                        }
                        let mut writer = CabacWriter::new();
                        crate::coeff::code(
                            &mut writer,
                            &mut CoeffContexts::new(37, inter),
                            &source,
                            side,
                            channel,
                        )
                        .unwrap();
                        writer.encode_terminate(true);
                        let mut bits = writer.finish();
                        bits.trailing_bits();
                        let bytes = bits.into_bytes();
                        let mut reader = Reader::new(&bytes).unwrap();
                        // Deliberately bypass the square fast-path delegation.
                        let actual = read_general(
                            &mut reader,
                            &mut Models::new(37, if inter { 1 } else { 2 }),
                            side,
                            side,
                            channel,
                            false,
                            true,
                            false,
                        )
                        .unwrap();
                        assert_eq!(
                            actual, source,
                            "side={side} inter={inter} channel={channel} pattern={pattern}"
                        );
                        assert!(reader.terminate().unwrap());
                    }
                }
            }
        }
    }
}
