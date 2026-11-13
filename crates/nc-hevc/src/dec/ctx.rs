//! Every CABAC context variable of Main-profile slice data, with the
//! initial values of all three `initType`s (H.265 §9.3.2.2, Tables 9-5 to
//! 9-37), kept in one flat array so that WPP and dependent slice segments
//! can save and restore the whole state (§9.3.2.4, §9.3.2.5).

use crate::cabac::Context;

/// One syntax element's run of contexts in the flat array.
#[derive(Debug, Clone, Copy)]
pub(super) struct Element {
    /// Index of its first context.
    pub start: usize,
    /// Initial values per `initType` (0: I, 1: P or B with
    /// `cabac_init_flag`, 2: B or P with `cabac_init_flag`).
    init: [&'static [u8]; 3],
}

/// "Not used" initial value of elements absent from I slices (154 is HM's
/// `CNU`); any value works because those elements are never decoded there.
const CNU: u8 = 154;

macro_rules! elements {
    ($($name:ident: [$i:expr, $p:expr, $b:expr];)*) => {
        elements!(@step 0usize; $($name: [$i, $p, $b];)*);
        /// All elements, for initialisation.
        const ALL: &[Element] = &[$($name),*];
    };
    (@step $start:expr; $name:ident: [$i:expr, $p:expr, $b:expr]; $($rest:tt)*) => {
        pub(super) const $name: Element = Element { start: $start, init: [&$i, &$p, &$b] };
        elements!(@step $start + $name.init[0].len(); $($rest)*);
    };
    (@step $start:expr;) => {
        /// Number of contexts.
        const COUNT: usize = $start;
    };
}

elements! {
    SAO_MERGE: [[153], [153], [153]];
    SAO_TYPE: [[200], [185], [160]];
    SPLIT_CU: [[139, 141, 157], [107, 139, 126], [107, 139, 126]];
    TRANSQUANT_BYPASS: [[154], [154], [154]];
    SKIP: [[CNU; 3], [197, 185, 201], [197, 185, 201]];
    PRED_MODE: [[CNU], [149], [134]];
    PART_MODE: [[184, CNU, CNU, CNU], [154, 139, 154, 154], [154, 139, 154, 154]];
    PREV_INTRA_LUMA: [[184], [154], [183]];
    CHROMA_PRED: [[63], [152], [152]];
    RQT_ROOT_CBF: [[CNU], [79], [79]];
    MERGE_FLAG: [[CNU], [110], [154]];
    MERGE_IDX: [[CNU], [122], [137]];
    INTER_PRED_IDC: [[CNU; 5], [95, 79, 63, 31, 31], [95, 79, 63, 31, 31]];
    REF_IDX: [[CNU; 2], [153, 153], [153, 153]];
    MVP_FLAG: [[CNU], [168], [168]];
    SPLIT_TRANSFORM: [[153, 138, 138], [124, 138, 94], [224, 167, 122]];
    CBF_LUMA: [[111, 141], [153, 111], [153, 111]];
    CBF_CHROMA: [[94, 138, 182, 154, 154], [149, 107, 167, 154, 154], [149, 92, 167, 154, 154]];
    ABS_MVD_GT0: [[CNU], [140], [169]];
    ABS_MVD_GT1: [[CNU], [198], [198]];
    CU_QP_DELTA: [[154, 154], [154, 154], [154, 154]];
    TRANSFORM_SKIP: [[139, 139], [139, 139], [139, 139]];
    LAST_X: [
        [110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63],
        [125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108],
        [125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93]
    ];
    LAST_Y: [
        [110, 110, 124, 125, 140, 153, 125, 127, 140, 109, 111, 143, 127, 111, 79, 108, 123, 63],
        [125, 110, 94, 110, 95, 79, 125, 111, 110, 78, 110, 111, 111, 95, 94, 108, 123, 108],
        [125, 110, 124, 110, 95, 94, 125, 111, 111, 79, 125, 126, 111, 111, 79, 108, 123, 93]
    ];
    CODED_SUB_BLOCK: [[91, 171, 134, 141], [121, 140, 61, 154], [121, 140, 61, 154]];
    SIG_COEFF: [
        [111, 111, 125, 110, 110, 94, 124, 108, 124, 107, 125, 141, 179, 153, 125, 107, 125, 141,
         179, 153, 125, 107, 125, 141, 179, 153, 125, 140, 139, 182, 182, 152, 136, 152, 136, 153,
         136, 139, 111, 136, 139, 111],
        [155, 154, 139, 153, 139, 123, 123, 63, 153, 166, 183, 140, 136, 153, 154, 166, 183, 140,
         136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 123, 123, 107, 121, 107, 121, 167,
         151, 183, 140, 151, 183, 140],
        [170, 154, 139, 153, 139, 123, 123, 63, 124, 166, 183, 140, 136, 153, 154, 166, 183, 140,
         136, 153, 154, 166, 183, 140, 136, 153, 154, 170, 153, 138, 138, 122, 121, 122, 121, 167,
         151, 183, 140, 151, 183, 140]
    ];
    GREATER1: [
        [140, 92, 137, 138, 140, 152, 138, 139, 153, 74, 149, 92, 139, 107, 122, 152, 140, 179,
         166, 182, 140, 227, 122, 197],
        [154, 196, 196, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 137, 169, 194,
         166, 167, 154, 167, 137, 182],
        [154, 196, 167, 167, 154, 152, 167, 182, 182, 134, 149, 136, 153, 121, 136, 122, 169, 208,
         166, 167, 154, 152, 167, 182]
    ];
    GREATER2: [[138, 153, 136, 167, 152, 152], [107, 167, 91, 122, 107, 167], [107, 167, 91, 107, 107, 167]];
}

/// Offset of the chroma contexts within `SIG_COEFF`.
pub(super) const SIG_CHROMA: usize = 27;
/// Offset of the chroma contexts within `LAST_X` / `LAST_Y`.
pub(super) const LAST_CHROMA: usize = 15;
/// Offset of the chroma contexts within `GREATER1`.
pub(super) const GREATER1_CHROMA: usize = 16;
/// Offset of the chroma contexts within `GREATER2`.
pub(super) const GREATER2_CHROMA: usize = 4;
/// Offset of the chroma contexts within `CODED_SUB_BLOCK`.
pub(super) const CODED_SUB_BLOCK_CHROMA: usize = 2;
/// The `inter_pred_idc` context of its second bin / of 8x4 and 4x8 PUs.
pub(super) const INTER_PRED_IDC_LAST: usize = 4;

/// The whole context state of one slice segment.
#[derive(Debug, Clone)]
pub(super) struct Contexts([Context; COUNT]);

impl Contexts {
    /// Initialise every context for `init_type` at `slice_qp` (§9.3.2.2).
    pub fn new(init_type: usize, slice_qp: i32) -> Self {
        let mut all = [Context::new(CNU, slice_qp); COUNT];
        for element in ALL {
            for (i, &v) in element.init[init_type].iter().enumerate() {
                all[element.start + i] = Context::new(v, slice_qp);
            }
        }
        Self(all)
    }

    /// Context `inc` of `element`.
    #[inline]
    pub fn at(&mut self, element: Element, inc: usize) -> &mut Context {
        &mut self.0[element.start + inc]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_init_type_lists_every_context() {
        for element in ALL {
            let n = element.init[0].len();
            assert!(element.init.iter().all(|t| t.len() == n));
        }
        assert_eq!(SIG_COEFF.init[0].len(), 42);
        assert_eq!(GREATER1.init[0].len(), 24);
        assert_eq!(GREATER2.start + 6, COUNT);
    }
}
