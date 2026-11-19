//! One CTB's SAO statistics (HM `getBlkStats`), gathered on first use: a
//! picture that reuses types reads only those, and a merge candidate of
//! another type adds its own on demand.
//!
//! Edge statistics cover the CTB's samples whose two neighbours along the
//! class lie inside the picture (the others are never filtered, HEVC
//! §8.7.3.2); band statistics cover the whole CTB. Both read the deblocked
//! picture, which is what SAO classifies (§8.7.3.1).

use nc_common::{Picture, PlaneIndex};
use nc_dsp::{Dsp, SAO_EO_NEIGHBOURS, SaoBandStats, SaoStats};

use super::{EDGE_CLASSES, Settings};

/// The statistics gathered so far for one plane of the CTB.
#[derive(Default)]
struct PlaneStats {
    edge: [Option<SaoStats>; EDGE_CLASSES as usize],
    band: Option<SaoBandStats>,
}

/// One CTB's planes and the statistics gathered from them.
pub(super) struct CtbStats<'a> {
    dsp: &'a Dsp,
    source: &'a Picture,
    deblocked: &'a Picture,
    /// The CTB in CTB units, and its luma side.
    position: (usize, usize),
    ctb: usize,
    row_step: usize,
    planes: [PlaneStats; 3],
}

impl<'a> CtbStats<'a> {
    pub(super) fn new(
        dsp: &'a Dsp,
        (source, deblocked): (&'a Picture, &'a Picture),
        position: (usize, usize),
        settings: &Settings,
    ) -> Self {
        Self {
            dsp,
            source,
            deblocked,
            position,
            ctb: settings.ctb,
            row_step: settings.row_step,
            planes: Default::default(),
        }
    }

    /// The CTB in `plane` samples, `(x, y, side)` (4:2:0 chroma: half).
    fn origin(&self, plane: PlaneIndex) -> (usize, usize, usize) {
        let side = if plane == PlaneIndex::Y {
            self.ctb
        } else {
            self.ctb / 2
        };
        (self.position.0 * side, self.position.1 * side, side)
    }

    /// Edge statistics of `plane` for `class`.
    pub(super) fn edge(&mut self, plane: PlaneIndex, class: u8) -> SaoStats {
        if let Some(stats) = self.planes[plane as usize].edge[usize::from(class)] {
            return stats;
        }
        let deblocked = self.deblocked.plane(plane).view();
        let rect = class_rect(
            class,
            (deblocked.width(), deblocked.height()),
            self.origin(plane),
        );
        let stats = self.dsp.sao_edge_stats_sampled(
            class,
            &self.source.plane(plane).view(),
            &deblocked,
            rect,
            self.row_step,
        );
        self.planes[plane as usize].edge[usize::from(class)] = Some(stats);
        stats
    }

    /// Band statistics of `plane`.
    pub(super) fn band(&mut self, plane: PlaneIndex) -> SaoBandStats {
        if let Some(stats) = self.planes[plane as usize].band {
            return stats;
        }
        let deblocked = self.deblocked.plane(plane).view();
        let (x, y, side) = self.origin(plane);
        let rect = (
            x,
            y,
            (x + side).min(deblocked.width()),
            (y + side).min(deblocked.height()),
        );
        let stats = self.dsp.sao_band_stats_sampled(
            &self.source.plane(plane).view(),
            &deblocked,
            rect,
            self.row_step,
        );
        self.planes[plane as usize].band = Some(stats);
        stats
    }
}

/// The samples of the CTB `(cx, cy, ctb)` (plane samples) of a `width` x
/// `height` plane that have both neighbours of `class` inside the plane,
/// as `(x0, y0, x1, y1)` (possibly empty).
#[must_use]
pub fn class_rect(
    class: u8,
    (width, height): (usize, usize),
    (cx, cy, ctb): (usize, usize, usize),
) -> (usize, usize, usize, usize) {
    let [(ax, ay), (bx, by)] = SAO_EO_NEIGHBOURS[usize::from(class)];
    let (left, right) = (usize::from(ax < 0 || bx < 0), usize::from(ax > 0 || bx > 0));
    let (top, bottom) = (usize::from(ay < 0 || by < 0), usize::from(ay > 0 || by > 0));
    (
        cx.max(left),
        cy.max(top),
        (cx + ctb).min(width.saturating_sub(right)),
        (cy + ctb).min(height.saturating_sub(bottom)),
    )
}
