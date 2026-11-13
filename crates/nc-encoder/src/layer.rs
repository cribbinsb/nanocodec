//! Temporal-layer position of a frame in a low-delay mini-GOP.
//!
//! Flat IPP is one layer. Hierarchical P (L1T2: period 2; L1T3: period 4,
//! layers T0 T2 T1 T2) codes base frames from further back, so each layer
//! has its own bit cost and quantizer. Rate control receives this per
//! frame so it can keep one model per layer (SVT keeps a rate correction
//! factor per mini-GOP position).

/// Where a frame sits in the temporal-layer structure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LayerPosition {
    /// Temporal layer id; 0 is the base layer.
    pub layer: u8,
    /// Number of temporal layers; 1 is flat IPP.
    pub layers: u8,
    /// Position within the mini-GOP, 0..period.
    pub position: u8,
    /// Mini-GOP length in frames.
    pub period: u8,
}

impl LayerPosition {
    /// Every frame of a flat IPP chain.
    pub const FLAT: Self = Self {
        layer: 0,
        layers: 1,
        position: 0,
        period: 1,
    };

    /// Whether this frame is in the top layer of a layered structure, and
    /// therefore never referenced (droppable).
    #[must_use]
    pub const fn is_top_layer(self) -> bool {
        self.layers > 1 && self.layer + 1 == self.layers
    }

    /// The positions of one mini-GOP of this structure, in coding order.
    /// Layers are dyadic: position `p > 0` is in layer
    /// `layers − 1 − trailing_zeros(p)` (L1T3: T0 T2 T1 T2).
    #[must_use]
    pub fn cycle(self) -> Vec<LayerPosition> {
        (0..self.period)
            .map(|position| LayerPosition {
                layer: if position == 0 {
                    0
                } else {
                    self.layers - 1 - position.trailing_zeros() as u8
                },
                position,
                ..self
            })
            .collect()
    }

    /// Whether this frame belongs to a flat (single-layer) structure.
    #[must_use]
    pub const fn is_flat(self) -> bool {
        self.layers <= 1
    }
}
