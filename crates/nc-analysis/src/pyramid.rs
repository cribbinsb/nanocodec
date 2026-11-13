//! The luma pyramid: half- and quarter-resolution copies of a frame for
//! hierarchical motion search (the search runs at quarter resolution; the
//! half level is the step in between).
//!
//! Each level is a 2x box-filtered downscale of the level above (odd edges
//! drop their last sample), with padded borders so searches can reach past
//! the picture edge exactly as at full resolution (searches reaching past
//! the half level's border gather their samples). Only HME level 1 reads
//! the half level's border, so it is extended on demand
//! ([`Pyramid::ensure_half_border`]): with level 1 off (the H.264 speed
//! presets) the border is never written.

use nc_common::{Plane, PlaneView, PlaneViewMut, Result};
use nc_dsp::Dsp;

/// Border of the half level, in its samples: the 32-sample SB overhanging
/// the picture's edge. HME level 1 gathers windows reaching further out
/// (`motion::Level::gathered`); its full reach (104) was ~250 KB of border
/// stores per 720p frame.
const HALF_BORDER: usize = 32;
/// Border of the quarter level: the quarter-resolution search reaches at
/// most `HME_LIMIT` samples past its 16-sample block (the window centre
/// within `HME_LIMIT - HME_RANGE_X`, plus the window), and the block itself
/// overhangs a picture narrower than it. The half level's border here
/// wrote ~90 KB more border stores per 720p frame.
const QUARTER_BORDER: usize = crate::motion::HME_LIMIT as usize + crate::motion::HME_BLOCK;

/// How the half level is made from the full-resolution luma.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Downsample {
    /// The rounded mean of each 2x2 box.
    #[default]
    Filtered,
    /// Even rows only, the rounded mean of each horizontal pair: half the
    /// luma reads (SVT-AV1's faster presets decimate their ME pyramid too).
    EvenRowsMean,
    /// Even rows and even columns: plain decimation.
    Decimated,
}

impl Downsample {
    /// Fill `half` (rows `top..top + half.height()` of the level) from
    /// `luma`, whose top row is full-resolution row `2 * top`.
    pub(crate) fn fill(self, dsp: &Dsp, luma: &PlaneView<'_>, half: &mut PlaneViewMut<'_>) {
        let rows = half.height();
        match self {
            Self::Filtered => {
                let source = luma
                    .sub_view(0, 0, 2 * half.width(), 2 * rows)
                    .expect("the source covers the level's rows");
                dsp.downscale_2x(&source, half);
            }
            Self::EvenRowsMean | Self::Decimated => {
                let even = PlaneView::new(luma.data(), 2 * luma.stride(), luma.width(), rows)
                    .expect("the source covers the level's even rows");
                dsp.pairs_2x(&even, half, self == Self::Decimated);
            }
        }
    }
}

/// Half- and quarter-resolution luma.
#[derive(Debug, Clone)]
pub struct Pyramid {
    half: Plane,
    quarter: Plane,
    /// Whether `half`'s border holds the current level's edge samples.
    half_border: bool,
    /// Levels holding the current frame's samples.
    built: Levels,
    /// How the half level is made (the quarter level is always filtered:
    /// it reads the small half level).
    downsample: Downsample,
}

/// How much of a [`Pyramid`] holds the current frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Levels {
    None,
    Half,
    Both,
}

impl Pyramid {
    /// Empty levels for a `width` x `height` luma plane.
    ///
    /// # Errors
    ///
    /// If the picture is too small to have a quarter-resolution level
    /// (under 4 samples in either dimension).
    pub fn new(width: usize, height: usize) -> Result<Self> {
        Ok(Self {
            half: Plane::new(width / 2, height / 2, HALF_BORDER)?,
            quarter: Plane::new(width / 4, height / 4, QUARTER_BORDER)?,
            half_border: false,
            built: Levels::None,
            downsample: Downsample::Filtered,
        })
    }

    /// How the half level is made from now on.
    pub fn set_downsample(&mut self, downsample: Downsample) {
        self.downsample = downsample;
    }

    /// How the half level is made.
    #[must_use]
    pub fn downsample(&self) -> Downsample {
        self.downsample
    }

    /// Forget the levels: `luma` holds a new frame.
    pub fn invalidate(&mut self) {
        self.built = Levels::None;
    }

    /// Whether every level holds the current frame.
    #[must_use]
    pub fn is_built(&self) -> bool {
        self.built == Levels::Both
    }

    /// The half level, for the caller to fill now from this frame's luma as
    /// [`Self::downsample`] says (pre-analysis does, a strip at a time),
    /// after which it counts as built.
    pub fn half_to_fill(&mut self) -> &mut Plane {
        self.half_border = false;
        self.built = Levels::Half;
        &mut self.half
    }

    /// Build every level from `luma` not yet holding this frame and extend
    /// the quarter level's border (the half level's waits for
    /// [`Self::ensure_half_border`]).
    ///
    /// # Panics
    ///
    /// If `luma` is not the size this pyramid was created for.
    pub fn build(&mut self, dsp: &Dsp, luma: &Plane) {
        if self.built == Levels::Both {
            return;
        }
        if self.built == Levels::None {
            assert_eq!(
                (luma.width() / 2, luma.height() / 2),
                (self.half.width(), self.half.height()),
                "pyramid built from a plane of a different size"
            );
            self.downsample
                .fill(dsp, &luma.view(), &mut self.half.view_mut());
            self.half_border = false;
        }
        dsp.downscale_2x(&self.half.view(), &mut self.quarter.view_mut());
        self.quarter.extend_borders_with(dsp);
        self.built = Levels::Both;
    }

    /// Extend the half level's border, if this build has not yet: before a
    /// search reads the half level ([`Self::half`]).
    pub fn ensure_half_border(&mut self, dsp: &Dsp) {
        if !self.half_border {
            self.half.extend_borders_with(dsp);
            self.half_border = true;
        }
    }

    /// Half resolution. Its border is valid only after
    /// [`Self::ensure_half_border`].
    #[must_use]
    pub fn half(&self) -> &Plane {
        &self.half
    }

    /// Quarter resolution.
    #[must_use]
    pub fn quarter(&self) -> &Plane {
        &self.quarter
    }
}

#[cfg(test)]
mod tests {
    use nc_common::Plane;
    use nc_dsp::Dsp;

    use super::{Downsample, Pyramid};

    #[test]
    fn levels_average_boxes() {
        let mut luma = Plane::new(16, 16, 8).unwrap();
        {
            let mut v = luma.view_mut();
            for y in 0..16 {
                for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                    *s = if x < 8 { 0 } else { 200 };
                }
            }
        }
        let mut p = Pyramid::new(16, 16).unwrap();
        p.build(&Dsp::new(), &luma);
        assert!(p.is_built());
        assert_eq!(p.half().view().row(0), &[0, 0, 0, 0, 200, 200, 200, 200]);
        assert_eq!(p.quarter().view().row(1), &[0, 0, 200, 200]);
    }

    #[test]
    fn decimated_levels_read_even_rows() {
        let mut luma = Plane::new(8, 4, 8).unwrap();
        {
            let mut v = luma.view_mut();
            for y in 0..4 {
                for (x, s) in v.row_mut(y).iter_mut().enumerate() {
                    *s = if y % 2 == 0 { (10 * x) as u8 } else { 255 };
                }
            }
        }
        let mut p = Pyramid::new(8, 4).unwrap();
        p.set_downsample(Downsample::EvenRowsMean);
        p.build(&Dsp::new(), &luma);
        assert_eq!(p.half().view().row(0), &[5, 25, 45, 65]);
        let mut p = Pyramid::new(8, 4).unwrap();
        p.set_downsample(Downsample::Decimated);
        p.build(&Dsp::new(), &luma);
        assert_eq!(p.half().view().row(1), &[0, 20, 40, 60]);
    }
}
