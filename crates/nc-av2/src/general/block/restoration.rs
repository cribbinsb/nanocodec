//! Loop restoration units (dav2d `read_sb_restoration`,
//! `read_restoration_info`, `read_wiener_filter`; AVM `read_lr_params`):
//! at the top-left superblock of each unit, its type and, for a
//! non-separable Wiener unit of a plane without frame filters, its
//! filters: one of the last four filters the tile coded (a bank per luma
//! class and per chroma plane) as is, or refined tap by tap from it.
//!
//! Units are kept by the 64x64 filter block holding their top-left luma
//! position ([`crate::general::filters::FilterMaps::restoration`]).

use super::Tile;
use crate::general::bits::inverse_recenter;
use crate::general::filters::RestorationUnit;
use crate::general::frame_header::restoration::{
    CHROMA_TAP_MASKS, CHROMA_TAP_RANGE, LUMA_TAP_MASKS, LUMA_TAP_RANGE, MAX_CLASSES, MAX_TAPS,
    Taps, kind,
};
use crate::general::sample::Av2Sample;
use nc_common::decode::Result;

/// Filters a bank remembers.
const BANK_SIZE: usize = 4;
/// Taps of a luma filter.
const LUMA_TAPS: usize = 16;
/// First tap the asymmetric chroma code pairs with the next.
const FIRST_PAIRED_TAP: usize = 6;
/// The four-part tap code's first group is this many bits narrower than
/// the tap range (dav2d `decode_4way`'s `n_bits - 4`).
const FOUR_PART_SAVING: u32 = 4;
/// Luma samples per 4x4 unit, as a shift.
const UNIT_LOG2: u32 = 2;

/// The last filters a tile coded for one luma class or chroma plane
/// (dav2d `WienerBankY`, `WienerBankUV`).
#[derive(Debug, Clone, Copy)]
struct Bank {
    filters: [Taps; BANK_SIZE],
    /// Filters held (at most [`BANK_SIZE`]) and the newest one's slot.
    size: usize,
    newest: usize,
}

impl Bank {
    /// A bank holding the middle of every tap's range (dav2d
    /// `setup_tile`).
    fn new(ranges: &[(u8, i8)]) -> Self {
        let mut filters = [[0; MAX_TAPS]; BANK_SIZE];
        for (tap, &(bits, minimum)) in filters[0].iter_mut().zip(ranges) {
            *tap = minimum + ((1_i16 << bits) >> 1) as i8;
        }
        Self {
            filters,
            size: 0,
            newest: 0,
        }
    }

    /// Keep a newly coded filter.
    fn push(&mut self, filter: &Taps) {
        self.newest = (self.newest + 1) % BANK_SIZE;
        self.filters[self.newest] = *filter;
        self.size = (self.size + 1).min(BANK_SIZE);
    }
}

/// The tile's banks: one per luma class, one per chroma plane.
#[derive(Debug, Clone)]
pub struct Banks {
    luma: Vec<Bank>,
    chroma: [Bank; 2],
}

impl Banks {
    /// Fresh banks for a tile.
    #[must_use]
    pub fn new() -> Self {
        Self {
            luma: vec![Bank::new(&LUMA_TAP_RANGE); MAX_CLASSES],
            chroma: [Bank::new(&CHROMA_TAP_RANGE); 2],
        }
    }
}

impl Default for Banks {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Av2Sample> Tile<'_, '_, T> {
    /// The restoration units of planes `planes` starting at the current
    /// superblock (dav2d `read_sb_restoration`).
    pub(super) fn restoration_units(&mut self, planes: std::ops::Range<usize>) -> Result<()> {
        let restoration = self.frame.restoration;
        let superblock = (self.geometry.superblock_units << UNIT_LOG2) as usize;
        for plane in planes {
            let frame_kind = restoration.planes[plane].kind;
            if frame_kind == kind::NONE {
                continue;
            }
            let ss = usize::from(plane > 0);
            let unit_log2 = u32::from(restoration.unit_log2[ss]);
            let unit = 1_usize << unit_log2;
            let x = ((self.column - self.bounds.column_start) << UNIT_LOG2) as usize >> ss;
            let y = ((self.row - self.bounds.row_start) << UNIT_LOG2) as usize >> ss;
            if (x | y) & (unit - 1) != 0 {
                continue;
            }
            let width = (self.bounds.column_end << UNIT_LOG2) as usize >> ss;
            let height = (self.bounds.row_end << UNIT_LOG2) as usize >> ss;
            let half = unit >> 1;
            // The last unit of a row or column absorbs a remainder below
            // half a unit.
            let (fx, fy) = (
                (self.column << UNIT_LOG2) as usize >> ss,
                (self.row << UNIT_LOG2) as usize >> ss,
            );
            if (y != 0 && fy + half > height) || (x != 0 && fx + half > width) {
                continue;
            }
            // Superblocks are never more than one unit wide or (4:2:0)
            // tall.
            let unit_rows = (((height - fy + half).min(superblock >> ss)) >> unit_log2).max(1);
            for row in 0..unit_rows {
                let luma_y = (self.row << UNIT_LOG2) as usize + (row << (unit_log2 as usize + ss));
                let block = self.filter_block((luma_y >> UNIT_LOG2) as u32, self.column);
                let unit = self.restoration_unit(plane, frame_kind)?;
                let range = self.reader.range();
                trace!(
                    self,
                    0, "Post-restoration[p={plane},type={}]: r={range}", unit.kind
                );
                self.filters.restoration[plane][block] = Some(unit);
            }
        }
        Ok(())
    }

    /// One unit's type and filters (dav2d `read_restoration_info`).
    fn restoration_unit(&mut self, plane: usize, frame_kind: u8) -> Result<RestorationUnit> {
        let unit_kind = if frame_kind == kind::SWITCHABLE {
            if self.bool(|m| &mut m.rst_switchable[0])? {
                kind::NONE
            } else if self.bool(|m| &mut m.rst_switchable[1])? {
                kind::PC_WIENER
            } else {
                kind::NS_WIENER
            }
        } else {
            let on = if frame_kind == kind::NS_WIENER {
                self.bool(|m| &mut m.rst_ns_wiener)?
            } else {
                self.bool(|m| &mut m.rst_pc_wiener)?
            };
            if on { frame_kind } else { kind::NONE }
        };
        let mut unit = RestorationUnit {
            kind: unit_kind,
            filters: [[0; MAX_TAPS]; MAX_CLASSES],
        };
        let wiener = self.frame.restoration.planes[plane].wiener;
        if unit_kind != kind::NS_WIENER || wiener.frame_filters {
            return Ok(unit);
        }
        if plane > 0 {
            let mut bank = self.restoration_banks.chroma[plane - 1];
            let (exact, reference) = self.bank_reference(&bank)?;
            let filter = &mut unit.filters[0];
            if exact {
                *filter = bank.filters[reference];
                bank.size = bank.size.max(1);
            } else {
                self.unit_filter(filter, &bank.filters[reference], true)?;
                bank.push(filter);
            }
            self.restoration_banks.chroma[plane - 1] = bank;
            return Ok(unit);
        }
        // Every class's reference first, then the refinements.
        let classes = usize::from(wiener.classes);
        let mut references = [(false, 0); MAX_CLASSES];
        for (class, reference) in references.iter_mut().enumerate().take(classes) {
            let bank = self.restoration_banks.luma[class];
            *reference = self.bank_reference(&bank)?;
        }
        for (class, &(exact, reference)) in references.iter().enumerate().take(classes) {
            let mut bank = self.restoration_banks.luma[class];
            let filter = &mut unit.filters[class];
            if exact {
                filter[..LUMA_TAPS].copy_from_slice(&bank.filters[reference][..LUMA_TAPS]);
                bank.size = bank.size.max(1);
            } else {
                self.unit_filter(filter, &bank.filters[reference], false)?;
                bank.push(filter);
            }
            self.restoration_banks.luma[class] = bank;
        }
        Ok(unit)
    }

    /// Whether the unit reuses a bank filter exactly, and which (counted
    /// back from the newest; dav2d `read_wiener_ref`).
    fn bank_reference(&mut self, bank: &Bank) -> Result<(bool, usize)> {
        let exact = self.bypass()?;
        let mut back = 0;
        while back + 1 < bank.size && !self.bypass()? {
            back += 1;
        }
        Ok((exact, (bank.newest + BANK_SIZE - back) % BANK_SIZE))
    }

    /// A unit filter refined from `reference` (dav2d `read_wiener_filter`):
    /// how many tap groups it codes, for chroma whether its outer taps
    /// pair, then each coded tap in the four-part code around the
    /// reference's.
    fn unit_filter(&mut self, filter: &mut Taps, reference: &Taps, chroma: bool) -> Result<()> {
        let (masks, ranges): (&[u32], &[(u8, i8)]) = if chroma {
            (&CHROMA_TAP_MASKS, &CHROMA_TAP_RANGE)
        } else {
            (&LUMA_TAP_MASKS, &LUMA_TAP_RANGE)
        };
        let mut groups = 0;
        while groups < masks.len() - 1
            && self.bool(|m| &mut m.wiener_ns_len[usize::from(chroma)])?
        {
            groups += 1;
        }
        let paired = chroma && groups != 0 && self.bool(|m| &mut m.wiener_ns_sym)?;
        let mut mask = masks[groups];
        let mut tap = 0;
        while tap < ranges.len() {
            if mask & 1 != 0 {
                let (bits, minimum) = ranges[tap];
                let centre = i32::from(reference[tap]) - i32::from(minimum);
                filter[tap] = (self.four_part(centre, u32::from(bits))? + i32::from(minimum)) as i8;
                if paired && tap >= FIRST_PAIRED_TAP {
                    filter[tap + 1] = filter[tap];
                    tap += 1;
                    mask >>= 1;
                }
            }
            tap += 1;
            mask >>= 1;
        }
        Ok(())
    }

    /// A value in `0..2^bits` near `reference`: a group symbol, then the
    /// group's bits, recentred (dav2d `decode_4way`).
    fn four_part(&mut self, reference: i32, bits: u32) -> Result<i32> {
        let group = self.symbol(|m| &mut m.wiener_ns_cf)? as u32;
        let width = bits + group + u32::from(group == 0) - FOUR_PART_SAVING;
        let rest = self.reader.literal(width)?;
        let value = if group == 0 {
            rest
        } else {
            (1 << (bits + group - FOUR_PART_SAVING)) + rest
        };
        let count = 1_u32 << bits;
        let reference = reference as u32;
        Ok(if reference * 2 <= count {
            inverse_recenter(reference, value)
        } else {
            count - 1 - inverse_recenter(count - 1 - reference, value)
        } as i32)
    }
}
