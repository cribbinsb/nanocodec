//! AV1 inter prediction kernels (AV1 §7.11.3), 8-bit: everything the
//! decoder needs to turn motion data into a prediction block.
//!
//! - [`filter`]: the four interpolation filter families with dual
//!   (separate horizontal/vertical) filters and the 4-tap variants for block
//!   edges of at most 4 samples (§7.11.3.4).
//! - [`convolve`]: block inter prediction (§7.11.3.4) at final (8-bit) or
//!   compound precision, unscaled or with a per-sample step for scaled
//!   references; [`scale`] derives the start positions and steps
//!   (motion vector scaling, §7.11.3.3).
//! - [`blend`]: compound average and distance weights (§7.11.3.15), mask
//!   blends for compound and inter-intra prediction (§7.11.3.14) and the
//!   OBMC blend (§7.11.3.10).
//! - [`mask`]: wedge masks (§7.11.3.11), difference-weighted masks
//!   (§7.11.3.12) and smooth inter-intra masks (§7.11.3.13).
//! - [`warp`]: the warp filter (§7.11.3.5), shear setup (§7.11.3.6) and
//!   warp estimation (§7.11.3.8).
//!
//! Everything is a pure function of its inputs. Reference samples come in
//! as a *window* the caller gathers (with clamped coordinates where it
//! leaves the picture, as §7.11.3.4 reads `ref[Clip3(..)][Clip3(..)]`), so
//! the kernels need no picture geometry; only the warp, whose source
//! positions vary per 8x8 block, reads the whole reference plane.
//!
//! The scalar code is the reference, written as the spec states it. The
//! common case of our own streams (one reference, unscaled, one filter
//! family, power-of-two blocks from 4 to 128) goes through the existing SIMD
//! prediction kernels ([`crate::Dsp::predict`]), tiling 128-sample edges into
//! existing 64-sample kernels. The scalar full-block equation remains the oracle.
//!
//! Filter, mask and warp tables: libaom v3.15.3 `av1/common/filter.h`,
//! `reconinter.c`, `warped_motion.c` and `tools/gen_wedge_masks_data.py`
//! (BSD-2-Clause, Alliance for Open Media; see `THIRD_PARTY.md`).

pub mod blend;
pub mod convolve;
pub mod filter;
pub mod mask;
pub(crate) mod mask_kernels;
pub mod scale;
pub(crate) mod scaled;
pub mod warp;
#[cfg(target_arch = "x86_64")]
mod warp_avx2;
pub(crate) mod warp_kernels;
#[cfg(target_arch = "aarch64")]
mod warp_neon;

/// `FILTER_BITS`: precision of every interpolation and warp filter tap.
pub(crate) const FILTER_BITS: u32 = 7;
/// `InterRound0`: rounding of the horizontal filter pass (8-bit).
pub(crate) const INTER_ROUND0: u32 = 3;
/// `InterRound1` of non-compound prediction (8-bit): the output is pixels.
pub(crate) const INTER_ROUND1_FINAL: u32 = 2 * FILTER_BITS - INTER_ROUND0;
/// `InterRound1` of compound prediction (`COMPOUND_ROUND1_BITS`).
pub(crate) const INTER_ROUND1_COMPOUND: u32 = 7;
/// `InterPostRound` of compound prediction: the extra precision of a
/// compound prediction over pixels, removed when the two are blended.
pub(crate) const INTER_POST_ROUND: u32 = 2 * FILTER_BITS - INTER_ROUND0 - INTER_ROUND1_COMPOUND;
/// Largest 8-bit sample value.
pub(crate) const PIXEL_MAX: i32 = 255;

/// `Round2(value, bits)` for `bits >= 1` (AV1 §4.7).
pub(crate) const fn round2(value: i32, bits: u32) -> i32 {
    (value + (1 << (bits - 1))) >> bits
}

/// `Round2Signed(value, bits)` (AV1 §4.7): rounds the magnitude.
pub(crate) const fn round2_signed(value: i64, bits: u32) -> i64 {
    if value >= 0 {
        (value + (1 << (bits - 1))) >> bits
    } else {
        -((-value + (1 << (bits - 1))) >> bits)
    }
}

/// `Clip1` for 8-bit samples.
pub(crate) fn clip_pixel(value: i32) -> u8 {
    value.clamp(0, PIXEL_MAX) as u8
}
