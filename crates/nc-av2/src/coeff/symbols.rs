//! One symbol interface over the entropy writer and reader, so the
//! coefficient walk is written once for both directions.
//!
//! Writer methods code the given value and return it; reader methods ignore
//! the given value (the walk passes whatever its empty source implies) and
//! return the decoded one. High-range remainders use AVM's adaptive
//! truncated Rice code (`av2/common/hr_coding.c` `get_adaptive_param`,
//! `av2/encoder/encodetxb.c` `write_adaptive_hr`,
//! `av2/decoder/decodetxb.c` `read_adaptive_hr`).
//!
//! A third implementation, [`Pricer`], codes nothing: it adds up the rate
//! of every symbol under the writer's current probabilities, so that the
//! trellis prices candidate levels with the very model selection the coder
//! uses. [`FrozenPricer`] prices a writer's frozen rate estimates the same
//! way, straight from the live models, for the coefficient walk.
use super::levels::Region;
use super::models::BaseFamily;
use super::{Block, models};
use crate::cdf_models::TileCdfs;
use crate::entropy::{Cdf, Reader, Writer, fixed_cost_q9};
use nc_common::decode::{DecodeError, Result};
use nc_encoder::symbol_cost::COST_UNITS_PER_BIT;

/// Running-average thresholds that raise the Rice parameter above one.
const RICE_THRESHOLDS: [u32; 5] = [4, 8, 16, 32, 64];
/// Longest unary prefix before the Exp-Golomb escape.
const MAX_RICE_PREFIX: u32 = 6;
/// Unary prefix extension beyond the Rice parameter.
const RICE_PREFIX_EXTENSION: u32 = 4;
/// Exp-Golomb lengths beyond 20 bits cannot belong to a valid level.
const MAX_GOLOMB_BITS: u32 = 20;

pub(super) trait SymbolCoder {
    /// Code one symbol with the fixed CDF, or the adaptive tile model when
    /// adaptation is enabled.
    fn code_model<const N: usize>(
        &mut self,
        symbol: usize,
        fixed: &[u16],
        select: impl Fn(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> Result<usize>;

    /// Code `bits` bypass bits, most significant first.
    fn code_literal(&mut self, value: u32, bits: u32) -> Result<u32>;

    /// Code a high-range remainder given the running average of earlier ones.
    fn code_high_range(&mut self, value: u32, average: u32) -> Result<u32>;

    /// Code the base symbol of a position of `block` with `family`'s models
    /// in `context` ([`models::base`]).
    #[allow(
        clippy::inline_always,
        reason = "once per coded position of every block write"
    )]
    #[inline(always)]
    fn code_base(
        &mut self,
        symbol: usize,
        block: Block,
        family: BaseFamily,
        context: usize,
    ) -> Result<usize>
    where
        Self: Sized,
    {
        models::base(self, symbol, block, family, context)
    }

    /// Code the low-range increment of a position in `region` in `context`
    /// under coefficient quantizer category `q` ([`models::low_range`]).
    fn code_low_range(
        &mut self,
        symbol: usize,
        q: usize,
        region: Region,
        context: usize,
    ) -> Result<usize>
    where
        Self: Sized,
    {
        models::low_range(self, symbol, q, region, context)
    }
}

/// Rice parameter and unary limit for a running average.
fn rice(average: u32) -> (u32, u32) {
    let parameter = 1 + RICE_THRESHOLDS
        .iter()
        .take_while(|&&t| average >= t)
        .count() as u32;
    (
        parameter,
        (parameter + RICE_PREFIX_EXTENSION).min(MAX_RICE_PREFIX),
    )
}

/// Bits of a high-range remainder `value` after a running average
/// `average`: the unary prefix, its terminating one and the Rice suffix, or
/// past the prefix limit the Exp-Golomb escape (as [`Writer`] codes it).
pub(super) fn high_range_bits(value: u32, average: u32) -> u32 {
    let (m, prefix_limit) = rice(average);
    let unary = value >> m;
    if unary < prefix_limit {
        return unary + 1 + m;
    }
    let k = m + 1;
    let x = value - (prefix_limit << m) + (1 << k);
    let bits = u32::BITS - x.leading_zeros();
    prefix_limit + 2 * bits - 1 - k
}

/// Prices symbols instead of coding them: each model symbol at the
/// writer's current probabilities ([`Writer::price_q9`], no adaptation),
/// bypass bits at one bit each.
pub(super) struct Pricer<'a> {
    writer: &'a mut Writer,
    /// Rate priced since the last [`Self::take_q9`], in 1/512 bit.
    q9: u32,
}

impl<'a> Pricer<'a> {
    pub(super) fn new(writer: &'a mut Writer) -> Self {
        Self { writer, q9: 0 }
    }

    /// The rate priced since the last call, in 1/512 bit; starts again at 0.
    pub(super) fn take_q9(&mut self) -> u32 {
        std::mem::take(&mut self.q9)
    }
}

impl SymbolCoder for Pricer<'_> {
    fn code_model<const N: usize>(
        &mut self,
        symbol: usize,
        fixed: &[u16],
        select: impl Fn(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> Result<usize> {
        self.q9 += self.writer.price_q9(symbol, fixed, select);
        Ok(symbol)
    }

    fn code_literal(&mut self, value: u32, bits: u32) -> Result<u32> {
        self.q9 += bits * COST_UNITS_PER_BIT;
        Ok(value)
    }

    fn code_high_range(&mut self, value: u32, average: u32) -> Result<u32> {
        self.q9 += high_range_bits(value, average) * COST_UNITS_PER_BIT;
        Ok(value)
    }
}

/// Prices a writer's frozen rate estimate ([`Writer::estimates_frozen_now`])
/// from the live models directly: the same rates the writer would add
/// symbol by symbol, without its per-symbol mode dispatch, and with the
/// base and low-range models (every coded position's) indexed without a
/// coder round trip. The sums are integers, so the total is exact.
pub(super) struct FrozenPricer<'a> {
    /// The live models, or `None` when the tile does not adapt.
    models: Option<&'a mut TileCdfs>,
    /// Rate priced so far, in 1/512 bit.
    q9: u32,
}

impl<'a> FrozenPricer<'a> {
    pub(super) fn new(models: Option<&'a mut TileCdfs>) -> Self {
        Self { models, q9: 0 }
    }

    /// The rate priced in total, in 1/512 bit.
    pub(super) fn total_q9(&self) -> u32 {
        self.q9
    }
}

impl SymbolCoder for FrozenPricer<'_> {
    fn code_model<const N: usize>(
        &mut self,
        symbol: usize,
        fixed: &[u16],
        select: impl Fn(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> Result<usize> {
        let count = if self.models.is_some() {
            N
        } else {
            fixed.len()
        };
        if symbol >= count {
            return Err(DecodeError::Malformed("AV2 symbol outside CDF"));
        }
        self.q9 += if let Some(models) = self.models.as_deref_mut() {
            select(models).cost_q9(symbol)
        } else {
            fixed_cost_q9(fixed, symbol)
        };
        Ok(symbol)
    }

    fn code_literal(&mut self, value: u32, bits: u32) -> Result<u32> {
        self.q9 += bits * COST_UNITS_PER_BIT;
        Ok(value)
    }

    fn code_high_range(&mut self, value: u32, average: u32) -> Result<u32> {
        self.q9 += high_range_bits(value, average) * COST_UNITS_PER_BIT;
        Ok(value)
    }

    #[allow(
        clippy::inline_always,
        reason = "once per coded position of every rate estimate"
    )]
    #[inline(always)]
    fn code_base(
        &mut self,
        symbol: usize,
        block: Block,
        family: BaseFamily,
        context: usize,
    ) -> Result<usize> {
        let Some(models) = self.models.as_deref() else {
            return models::base(self, symbol, block, family, context);
        };
        self.q9 += models::base_cost_q9(models, symbol, block.tx, family, context);
        Ok(symbol)
    }

    #[inline]
    fn code_low_range(
        &mut self,
        symbol: usize,
        q: usize,
        region: Region,
        context: usize,
    ) -> Result<usize> {
        let Some(models) = self.models.as_deref() else {
            return models::low_range(self, symbol, q, region, context);
        };
        self.q9 += models::low_range_cost_q9(models, symbol, region, context);
        Ok(symbol)
    }
}

impl SymbolCoder for Writer {
    fn code_model<const N: usize>(
        &mut self,
        symbol: usize,
        fixed: &[u16],
        select: impl Fn(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> Result<usize> {
        self.model(symbol, fixed, select)?;
        Ok(symbol)
    }

    fn code_literal(&mut self, value: u32, bits: u32) -> Result<u32> {
        self.literal(value, bits)?;
        Ok(value)
    }

    fn code_high_range(&mut self, value: u32, average: u32) -> Result<u32> {
        let (m, prefix_limit) = rice(average);
        let unary = value >> m;
        if unary < prefix_limit {
            self.literal(0, unary)?;
            self.literal(1, 1)?;
            self.literal(value & ((1 << m) - 1), m)?;
        } else {
            self.literal(0, prefix_limit)?;
            let k = m + 1;
            let x = value - (prefix_limit << m) + (1 << k);
            let bits = u32::BITS - x.leading_zeros();
            self.literal(0, bits - 1 - k)?;
            self.literal(x, bits)?;
        }
        Ok(value)
    }
}

impl SymbolCoder for Reader<'_> {
    fn code_model<const N: usize>(
        &mut self,
        _symbol: usize,
        fixed: &[u16],
        select: impl Fn(&mut TileCdfs) -> &mut Cdf<N>,
    ) -> Result<usize> {
        self.model(fixed, select)
    }

    fn code_literal(&mut self, _value: u32, bits: u32) -> Result<u32> {
        self.literal(bits)
    }

    fn code_high_range(&mut self, _value: u32, average: u32) -> Result<u32> {
        let (m, prefix_limit) = rice(average);
        for unary in 0..prefix_limit {
            if self.literal(1)? != 0 {
                return Ok((unary << m) + self.literal(m)?);
            }
        }
        let k = m + 1;
        let mut extra = 0;
        while self.literal(1)? == 0 {
            extra += 1;
            if extra + k > MAX_GOLOMB_BITS {
                return Err(DecodeError::Malformed("AV2 coefficient Rice overflow"));
            }
        }
        let bits = extra + k;
        Ok((prefix_limit << m) + (1 << bits) + self.literal(bits)? - (1 << k))
    }
}
