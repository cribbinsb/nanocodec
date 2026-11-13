//! Probability-model selection for each coefficient syntax element.
//!
//! Every function names one AVM model family and indexes the fixed table
//! (`coeff_tables`, used without adaptation) and the adaptive tile model
//! (`TileCdfs`) identically, so the two never select different identities.
//! Families and axes follow AVM v1.0.0 `av2/common/entropymode.h`
//! (`txb_skip_cdf`, `eob_flag_cdf*`, `coeff_base*_cdf`, `coeff_br*_cdf`,
//! `dc_sign_cdf`, `inter_ext_tx_cdf`, `inter_tx_type_set`, `inter_tx_type_idx`,
//! `inter_tx_type_offset_1/2`, `intra_ext_tx_cdf`).
use super::levels::Region;
use super::symbols::SymbolCoder;
use super::{Block, PLANE_V, PLANE_Y, TxSize};
use crate::cdf_models::TileCdfs;
use crate::coeff_tables as t;
use crate::tx_type::InterSet;
use nc_common::decode::Result;

/// End-of-block plane contexts: intra luma, inter luma, chroma
/// (AVM `get_eob_plane_ctx`).
const EOB_CHROMA_CONTEXT: usize = 2;
/// Lossless inter TX4 flag without adaptation: probability one half.
const LOSSLESS_INTER_TX: [u16; 2] = [16384, 0];

/// Which base-symbol models code a position: its region's, the last
/// coefficient's family when `last`, and for a luma position of a TCQ
/// transform in a Q1 state the second quantizer's set (AVM
/// `coeff_base*_cdf[..][q_i]`; never for the last coefficient or chroma).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct BaseFamily {
    pub region: Region,
    pub last: bool,
    pub upper: bool,
}

impl BaseFamily {
    /// The family of a plain (non-TCQ) position.
    pub(super) const fn plain(region: Region, last: bool) -> Self {
        Self {
            region,
            last,
            upper: false,
        }
    }
}

/// All-zero flag. Y and U share the per-size models; V has its own.
pub(super) fn all_zero<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    block: Block,
) -> Result<usize> {
    let (q, skip) = (block.q, usize::from(block.ctx.skip));
    if block.ctx.plane == PLANE_V {
        return coder.code_model(symbol, &t::V_SKIP[q][skip], |m| &mut m.v_skip[skip]);
    }
    let (inter, tx) = (usize::from(block.ctx.is_inter), block.tx.index());
    coder.code_model(symbol, &t::SKIP[q][inter][tx][skip], |m| {
        &mut m.skip[inter][tx][skip]
    })
}

/// End-of-block class minus one. Larger sizes cap the model at eight
/// symbols (`EOB_MAX_SYMS - 3`); the caller codes the overflow bits.
pub(super) fn eob_class<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    block: Block,
) -> Result<usize> {
    let q = block.q;
    let p = if block.ctx.plane == PLANE_Y {
        usize::from(block.ctx.is_inter)
    } else {
        EOB_CHROMA_CONTEXT
    };
    match block.tx {
        TxSize::Tx4 => coder.code_model(symbol, &t::EOB16[q][p], |m| &mut m.eob16[p]),
        TxSize::Tx8 => coder.code_model(symbol, &t::EOB64[q][p], |m| &mut m.eob64[p]),
        TxSize::Tx16 => coder.code_model(symbol, &t::EOB256[q][p], |m| &mut m.eob256[p]),
        TxSize::Tx32 | TxSize::Tx64 => {
            coder.code_model(symbol, &t::EOB1024[q][p], |m| &mut m.eob1024[p])
        }
    }
}

/// Most significant end-of-block offset bit.
pub(super) fn eob_extra<C: SymbolCoder>(coder: &mut C, symbol: usize, q: usize) -> Result<usize> {
    coder.code_model(symbol, &t::EXTRA[q], |m| &mut m.extra)
}

/// Inter luma transform type for TX4..TX32, by EOB context and size: AVM's
/// `EXT_TX_SET_DCT_IDTX` models (inter set index 3).
pub(super) fn inter_tx_type<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    tx: TxSize,
    eob_context: usize,
) -> Result<usize> {
    let size = tx.index();
    coder.code_model(symbol, &t::INTER_TX[eob_context][size], |m| {
        &mut m.inter_tx[eob_context][size]
    })
}

/// Inter luma set flag of `reduced_tx_set_used = 0` (TX4..TX16): whether
/// the type's index lies past `INTER_TX_TYPE_INDEX_COUNT`
/// (AVM `inter_tx_type_set[eset - 1][eob_context][size]`).
pub(super) fn inter_tx_set_flag<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    set: InterSet,
    tx: TxSize,
    eob_context: usize,
) -> Result<usize> {
    let (s, size) = (set.model_index(), tx.index());
    coder.code_model(symbol, &t::INTER_TX_SET[s][eob_context][size], |m| {
        &mut m.inter_tx_set[s][eob_context][size]
    })
}

/// Inter luma type index below `INTER_TX_TYPE_INDEX_COUNT`
/// (AVM `inter_tx_type_idx[eset - 1][eob_context]`).
pub(super) fn inter_tx_index<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    set: InterSet,
    eob_context: usize,
) -> Result<usize> {
    let s = set.model_index();
    coder.code_model(symbol, &t::INTER_TX_IDX[s][eob_context], |m| {
        &mut m.inter_tx_idx[s][eob_context]
    })
}

/// Inter luma type offset past `INTER_TX_TYPE_INDEX_COUNT`
/// (AVM `inter_tx_type_offset_1` for `EXT_TX_SET_ALL16`, `_offset_2` for
/// `EXT_TX_SET_DTT9_IDTX_1DDCT`).
pub(super) fn inter_tx_offset<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    set: InterSet,
    eob_context: usize,
) -> Result<usize> {
    match set {
        InterSet::All16 => coder.code_model(symbol, &t::INTER_TX_OFFSET1[eob_context], |m| {
            &mut m.inter_tx_offset1[eob_context]
        }),
        InterSet::Dtt9Idtx1dDct => {
            coder.code_model(symbol, &t::INTER_TX_OFFSET2[eob_context], |m| {
                &mut m.inter_tx_offset2[eob_context]
            })
        }
    }
}

/// Intra luma `EXT_NEW_TX_SET` index for TX4..TX16 under
/// `reduced_tx_set_used = 0`, by size: AVM's `intra_ext_tx_cdf[1]`
/// (seven symbols).
pub(super) fn intra_full_tx_type<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    tx: TxSize,
) -> Result<usize> {
    let size = tx.index();
    coder.code_model(symbol, &t::INTRA_TX_FULL[size], |m| {
        &mut m.intra_tx_full[size]
    })
}

/// Intra luma transform type for TX4..TX16 under `reduced_tx_set_used`
/// 1 or 3, by size: AVM's `intra_ext_tx_cdf[2]` (two symbols).
pub(super) fn intra_tx_type<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    tx: TxSize,
) -> Result<usize> {
    let size = tx.index();
    coder.code_model(symbol, &t::INTRA_TX[size], |m| &mut m.intra_tx[size])
}

/// Lossless inter TX4: WHT (0) or IDTX (1).
pub(super) fn lossless_inter_tx_type<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
) -> Result<usize> {
    coder.code_model(symbol, &LOSSLESS_INTER_TX, |m| &mut m.lossless_inter_tx)
}

/// Base symbol. Luma models are per transform size; chroma models are shared
/// by all sizes. The last coefficient has its own (zero-free) families.
pub(super) fn base<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    block: Block,
    family: BaseFamily,
    ctx: usize,
) -> Result<usize> {
    let (q, tx) = (block.q, block.tx.index());
    if family.upper {
        return match family.region {
            Region::LumaLow => coder.code_model(symbol, &t::BASE_LF_TCQ[q][tx][ctx], |m| {
                &mut m.tcq().base_lf[tx][ctx]
            }),
            _ => coder.code_model(symbol, &t::BASE_TCQ[q][tx][ctx], |m| {
                &mut m.tcq().base[tx][ctx]
            }),
        };
    }
    match (family.region, family.last) {
        (Region::LumaLow, true) => coder.code_model(symbol, &t::BASE_LF_EOB[q][tx][ctx], |m| {
            &mut m.base_lf_eob[tx][ctx]
        }),
        (Region::LumaDefault, true) => coder.code_model(symbol, &t::BASE_EOB[q][tx][ctx], |m| {
            &mut m.base_eob[tx][ctx]
        }),
        (Region::ChromaLow, true) => coder.code_model(symbol, &t::UV_BASE_LF_EOB[q][ctx], |m| {
            &mut m.uv_base_lf_eob[ctx]
        }),
        (Region::ChromaDefault, true) => {
            coder.code_model(symbol, &t::UV_BASE_EOB[q][ctx], |m| &mut m.uv_base_eob[ctx])
        }
        (Region::LumaLow, false) => {
            coder.code_model(symbol, &t::BASE_LF[q][tx][ctx], |m| &mut m.base_lf[tx][ctx])
        }
        (Region::LumaDefault, false) => {
            coder.code_model(symbol, &t::BASE[q][tx][ctx], |m| &mut m.base[tx][ctx])
        }
        (Region::ChromaLow, false) => {
            coder.code_model(symbol, &t::UV_BASE_LF[q][ctx], |m| &mut m.uv_base_lf[ctx])
        }
        (Region::ChromaDefault, false) => {
            coder.code_model(symbol, &t::UV_BASE[q][ctx], |m| &mut m.uv_base[ctx])
        }
    }
}

/// The rate of base `symbol` under the live adaptive model [`base`]
/// selects, in 1/512 bit: a frozen estimate prices every coded position,
/// so it indexes the model directly instead of going through a coder.
#[allow(
    clippy::inline_always,
    reason = "once per coded position of every rate estimate"
)]
#[inline(always)]
pub(super) fn base_cost_q9(
    models: &TileCdfs,
    symbol: usize,
    tx: TxSize,
    family: BaseFamily,
    ctx: usize,
) -> u32 {
    let tx = tx.index();
    if family.upper {
        let tcq = models
            .built_tcq()
            .expect("a TCQ frame builds its models before coding");
        return match family.region {
            Region::LumaLow => tcq.base_lf[tx][ctx].cost_q9(symbol),
            _ => tcq.base[tx][ctx].cost_q9(symbol),
        };
    }
    match (family.region, family.last) {
        (Region::LumaLow, true) => models.base_lf_eob[tx][ctx].cost_q9(symbol),
        (Region::LumaDefault, true) => models.base_eob[tx][ctx].cost_q9(symbol),
        (Region::ChromaLow, true) => models.uv_base_lf_eob[ctx].cost_q9(symbol),
        (Region::ChromaDefault, true) => models.uv_base_eob[ctx].cost_q9(symbol),
        (Region::LumaLow, false) => models.base_lf[tx][ctx].cost_q9(symbol),
        (Region::LumaDefault, false) => models.base[tx][ctx].cost_q9(symbol),
        (Region::ChromaLow, false) => models.uv_base_lf[ctx].cost_q9(symbol),
        (Region::ChromaDefault, false) => models.uv_base[ctx].cost_q9(symbol),
    }
}

/// The rate of low-range `symbol` under the live adaptive model
/// [`low_range`] selects, in 1/512 bit.
#[inline]
pub(super) fn low_range_cost_q9(
    models: &TileCdfs,
    symbol: usize,
    region: Region,
    ctx: usize,
) -> u32 {
    match region {
        Region::LumaLow => models.br_lf[ctx].cost_q9(symbol),
        Region::LumaDefault => models.br[ctx].cost_q9(symbol),
        Region::ChromaLow | Region::ChromaDefault => models.uv_br[ctx].cost_q9(symbol),
    }
}

/// Low-range increment; shared by all sizes. Chroma DC has none.
pub(super) fn low_range<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    q: usize,
    region: Region,
    ctx: usize,
) -> Result<usize> {
    match region {
        Region::LumaLow => coder.code_model(symbol, &t::BR_LF[q][ctx], |m| &mut m.br_lf[ctx]),
        Region::LumaDefault => coder.code_model(symbol, &t::BR[q][ctx], |m| &mut m.br[ctx]),
        Region::ChromaLow | Region::ChromaDefault => {
            coder.code_model(symbol, &t::UV_BR[q][ctx], |m| &mut m.uv_br[ctx])
        }
    }
}

/// Luma DC sign (luma plane type), in the hidden-parity group when the
/// block hides its DC parity (AVM `dc_sign_cdf[0][is_hidden][ctx]`).
pub(super) fn dc_sign<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    block: Block,
    hidden: bool,
) -> Result<usize> {
    let ctx = usize::from(block.ctx.dc_sign);
    if hidden {
        coder.code_model(symbol, &t::SIGN_HIDDEN[block.q][ctx], |m| {
            &mut m.sign_hidden[ctx]
        })
    } else {
        coder.code_model(symbol, &t::SIGN[block.q][ctx], |m| &mut m.sign[0][ctx])
    }
}

/// The hidden-parity DC level index, 0..=3 (AVM `coeff_base_ph_cdf`).
pub(super) fn hidden_level<C: SymbolCoder>(
    coder: &mut C,
    symbol: usize,
    block: Block,
    context: usize,
) -> Result<usize> {
    coder.code_model(symbol, &t::BASE_PH[block.q][context], |m| {
        &mut m.base_ph[context]
    })
}
