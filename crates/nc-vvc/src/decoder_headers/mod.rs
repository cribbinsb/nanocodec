//! Bounded H.266 parameter-set syntax, independent of the encoder's writers.
//! Syntax checked against BSD-licensed `VVdeC`; see `THIRD_PARTY.md`.
mod aps;
pub(super) use aps::{Adaptation, Data as AdaptationData};
mod picture;
pub(super) use picture::PictureHeader;
mod pps;
mod ptl;
pub(super) use pps::PictureParameters;
mod rpl;
pub(super) use rpl::Entry as ReferenceEntry;
mod slice;
mod sps;
pub(super) use slice::SliceHeader;
mod tools;
mod vui;
mod weights;
use nc_common::decode::{BitReader, DecodeError, Result};
pub(super) use sps::Sequence;

pub(crate) struct Reader<'a>(BitReader<'a>);
impl<'a> Reader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self(BitReader::new(bytes))
    }
    fn flag(&mut self) -> Result<bool> {
        self.0.read_bit()
    }
    fn bits(&mut self, count: u32) -> Result<u32> {
        self.0.read_bits(count)
    }
    fn ue(&mut self, max: u32, name: &'static str) -> Result<u32> {
        let value = self.0.read_ue()?;
        if value > max {
            return Err(DecodeError::Malformed(name));
        }
        Ok(value)
    }
    fn se(&mut self, min: i32, max: i32, name: &'static str) -> Result<i32> {
        let value = self.0.read_se()?;
        if !(min..=max).contains(&value) {
            return Err(DecodeError::Malformed(name));
        }
        Ok(value)
    }
    fn zeroes(&mut self, count: u32) -> Result<()> {
        for _ in 0..count {
            if self.flag()? {
                return Err(DecodeError::Malformed("VVC reserved zero bit"));
            }
        }
        Ok(())
    }
    fn align(&mut self) -> Result<()> {
        self.0.align_zero()
    }
    fn flags(&mut self, count: usize) -> Result<Vec<bool>> {
        (0..count).map(|_| self.flag()).collect()
    }
    pub(crate) fn finish(&mut self, extensions: bool) -> Result<()> {
        if extensions {
            while self.0.remaining_bits() > 8 || {
                let n = self.0.remaining_bits();
                n > 0 && self.0.clone().read_bits(n as u32)? != 1 << (n - 1)
            } {
                self.flag()?;
            }
        }
        self.0.trailing_bits()?;
        if self.0.remaining_bits() != 0 {
            return Err(DecodeError::Malformed("VVC RBSP trailing bytes"));
        }
        Ok(())
    }
}
