//! H.266 §7.3.10 coding-unit and transform residual syntax.
//!
//! Binarization checked against `VVenC` `CABACWriter` (BSD-3-Clause-Clear;
//! see `THIRD_PARTY.md`). Quadtree coding units and the rectangular leaves
//! of one multi-type tree level, each one transform unit unless larger than
//! the largest transform (`transform_tree`).
use crate::{
    cabac::CabacWriter,
    coeff::Bins,
    contexts::Contexts,
    motion::{Amvr, Motion, Mv16, Predictors},
    transform_tree,
};
use nc_common::decode::Result;

/// How an inter CU's motion is signalled (§7.3.11.5, §7.3.11.7).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterSyntax {
    /// Regular merge candidate `merge_idx` (a skip CU when no residual).
    Merge(usize),
    /// Sub-block (affine) merge candidate `merge_subblock_idx`.
    SubblockMerge(usize),
    /// Affine AMVP: `mvp_l0_flag` and the MVD of each control point
    /// (quarter samples; the second and third are coded relative to the
    /// first, §7.4.11.7).
    AffineAmvp { mvp: usize, mvd: [[i32; 2]; 3] },
    /// AMVP: `mvp_l0_flag`, the MVD precision, and the MVD in its steps.
    Amvp {
        mvp: usize,
        amvr: Amvr,
        mvd: [i32; 2],
    },
}

/// A coded inter CU's motion and its signalling.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Inter {
    pub motion: Motion,
    pub syntax: InterSyntax,
}

/// Backend syntax choice for an engine-selected prediction mode.
#[derive(Debug, Clone)]
pub(crate) struct Coding {
    /// The CU's motion; `None` for an intra CU.
    pub inter: Option<Inter>,
    /// The coding unit (its luma size sets the transform units).
    pub cu: nc_encoder::Block,
    /// Levels by plane, in `transform_tree` layout.
    pub levels: [Vec<i32>; 3],
    pub skip: bool,
    pub intra_mode: u8,
    pub chroma_mode: u8,
    /// The intra sub-partitions of an intra CU (`isp.rs`), if split.
    pub isp: Option<crate::isp::Split>,
    /// The `CuQpDeltaVal` the CU's first transform unit with a coded block
    /// carries (`quant_group.rs`); `None` when it codes none.
    pub qp_delta: Option<i32>,
    /// `mts_idx` (`mts.rs`): 0 (DCT-II) unless signalled.
    pub mts: u8,
}

/// Decisions recorded during the shared walk. Like HEVC, entropy coding waits
/// until frame filtering has chosen SAO parameters, which precede each CTU.
#[derive(Debug)]
pub(crate) enum Item {
    Ctu {
        x: usize,
        y: usize,
    },
    /// A quadtree node's split (only nodes that code split bins).
    Split {
        allowed: crate::coding_tree::Allowed,
        contexts: crate::coding_tree::SplitContexts,
        split: crate::coding_tree::TreeSplit,
    },
    Leaf {
        contexts: crate::motion::CuContexts,
        coding: Coding,
    },
}

/// Coding unit `coding` in a picture with (`inter`) or without P slices,
/// with the SPS's optional `tools`.
pub(crate) fn block(
    w: &mut CabacWriter,
    c: &mut Contexts,
    (inter, tools): (bool, crate::headers::Tools),
    p: &crate::motion::CuContexts,
    coding: &Coding,
) {
    let &Coding {
        inter: motion,
        cu,
        ref levels,
        skip,
        intra_mode,
        chroma_mode,
        isp,
        qp_delta,
        mts,
    } = coding;
    let mut qp_delta = qp_delta;
    let intra = motion.is_none();
    if inter {
        w.encode_bin(&mut c.skip[p.skip_ctx], skip);
        if skip {
            let syntax = motion.expect("a skip CU merges").syntax;
            merge_data(w, c, tools, p, &cu, syntax);
            return;
        }
        w.encode_bin(&mut c.intra[p.intra_ctx], intra);
    }
    if let Some(Inter { motion, syntax }) = motion {
        let merged = matches!(
            syntax,
            InterSyntax::Merge(_) | InterSyntax::SubblockMerge(_)
        );
        w.encode_bin(&mut c.merge, merged);
        if merged {
            merge_data(w, c, tools, p, &cu, syntax);
        } else {
            if affine_signalled(tools, &cu) {
                let model = match motion {
                    Motion::Affine(model) => Some(model.six),
                    Motion::Vector(_) => None,
                };
                affine_flags(w, c, tools, p.affine_ctx, model).expect("valid affine flags");
            }
            match syntax {
                InterSyntax::Amvp { mvp, amvr, mvd } => {
                    write_mvd(w, c, mvd);
                    w.encode_bin(&mut c.mvp, mvp != 0);
                    if mvd != [0; 2] {
                        amvr_precision(w, c, amvr).expect("valid AMVR precision");
                    }
                }
                InterSyntax::AffineAmvp { mvp, mvd } => {
                    let Motion::Affine(model) = motion else {
                        unreachable!("affine AMVP codes an affine model");
                    };
                    for &delta in &mvd[..model.points()] {
                        write_mvd(w, c, delta);
                    }
                    w.encode_bin(&mut c.mvp, mvp != 0);
                }
                InterSyntax::Merge(_) | InterSyntax::SubblockMerge(_) => unreachable!(),
            }
            let present = levels.iter().flatten().any(|&v| v != 0);
            w.encode_bin(&mut c.root_cbf, present);
            if !present {
                return;
            }
        }
    } else {
        crate::isp::code_split(w, c, tools, &cu, isp).expect("valid ISP flags");
        crate::intra::mode(w, c, p.mpm, intra_mode, isp.is_some()).expect("valid intra mode");
        crate::chroma::code(w, c, intra_mode, chroma_mode).expect("valid chroma mode");
    }
    if let Some(split) = isp {
        isp_units(w, c, &cu, split, levels, qp_delta);
        return;
    }
    let max_tb_log2 = tools.max_tb_log2();
    let count = transform_tree::count(&cu, max_tb_log2);
    let tu = transform_tree::unit_of(&cu, max_tb_log2);
    let luma_flag = intra || transform_tree::splits(&cu, max_tb_log2);
    for index in 0..count {
        let unit = std::array::from_fn(|plane| {
            let samples = transform_tree::plane_samples(&tu, plane);
            transform_tree::unit_levels(&levels[plane], index, samples)
        });
        transform_unit(
            w,
            c,
            unit,
            (tu.width(), tu.height()),
            luma_flag,
            &mut qp_delta,
        );
    }
    if tools.mts.is_some() && crate::mts::signalled(tools.mts, intra, &cu, &levels[0]) {
        crate::mts::code(w, &mut c.mts_idx, mts).expect("valid mts_idx");
    } else {
        debug_assert_eq!(mts, 0, "mts_idx without its syntax");
    }
}

/// `cu_qp_delta_abs`/`cu_qp_delta_sign_flag` of a transform unit with a
/// coded block, when the CU still carries its group's delta (§7.3.11.10).
fn write_qp_delta(w: &mut CabacWriter, c: &mut Contexts, qp_delta: &mut Option<i32>) {
    if let Some(delta) = qp_delta.take() {
        crate::quant_group::code_delta(w, &mut c.cu_qp_delta, delta).expect("valid QP delta");
    }
}

/// §7.3.11.10 `transform_unit`: the coded flags and residuals of one TU.
/// `luma_flag`: `tu_y_coded_flag` is coded even when both chroma flags are
/// zero (intra CUs, CUs of several TUs); otherwise it is inferred 1.
/// `size` is the TU's luma width and height; `qp_delta`: the CU's delta
/// while still to be coded.
fn transform_unit(
    w: &mut CabacWriter,
    c: &mut Contexts,
    levels: [&[i32]; 3],
    (width, height): (usize, usize),
    luma_flag: bool,
    qp_delta: &mut Option<i32>,
) {
    let present = levels.each_ref().map(|v| v.iter().any(|&c| c != 0));
    w.encode_bin(&mut c.cbf_cb, present[1]);
    w.encode_bin(&mut c.cbf_cr[usize::from(present[1])], present[2]);
    if luma_flag || present[1] || present[2] {
        w.encode_bin(&mut c.cbf_y, present[0]);
    }
    if present.contains(&true) {
        write_qp_delta(w, c, qp_delta);
    }
    for (plane, level) in levels.iter().enumerate() {
        if present[plane] {
            let shift = usize::from(plane != 0);
            crate::coeff::emit_rect(
                w,
                &mut c.coeff,
                level,
                (width >> shift, height >> shift),
                shift,
            );
        }
    }
}

/// §7.3.11.8-10 the transform units of an ISP CU: one per sub-partition,
/// the CU's chroma flags and residuals with the last (§7.3.11.10 `chromaAvailable`).
fn isp_units(
    w: &mut CabacWriter,
    c: &mut Contexts,
    cu: &nc_encoder::Block,
    split: crate::isp::Split,
    levels: &[Vec<i32>; 3],
    mut qp_delta: Option<i32>,
) {
    let parts = crate::isp::parts(cu, split);
    let mut coded = [false; crate::isp::PARTS];
    for (index, part) in parts.iter().enumerate() {
        let luma = transform_tree::unit_levels(&levels[0], index, part.area());
        coded[index] = luma.iter().any(|&v| v != 0);
        let last = index == crate::isp::PARTS - 1;
        let chroma = [&levels[1], &levels[2]].map(|v| last && v.iter().any(|&l| l != 0));
        if last {
            w.encode_bin(&mut c.cbf_cb, chroma[0]);
            w.encode_bin(&mut c.cbf_cr[usize::from(chroma[0])], chroma[1]);
        }
        crate::isp::code_luma_flag(w, c, index, &coded[..index], coded[index])
            .expect("a coded last sub-partition after empty ones");
        if coded[index] || chroma.contains(&true) {
            write_qp_delta(w, c, &mut qp_delta);
        }
        if coded[index] {
            crate::coeff::emit_rect(w, &mut c.coeff, luma, (part.width(), part.height()), 0);
        }
        for (plane, present) in chroma.into_iter().enumerate() {
            if present {
                let size = (cu.width() / 2, cu.height() / 2);
                crate::coeff::emit_rect(w, &mut c.coeff, &levels[plane + 1], size, 1);
            }
        }
    }
}

/// §7.3.11.7 `merge_data` of a skip or merge CU (`syntax` a merge).
fn merge_data(
    w: &mut CabacWriter,
    c: &mut Contexts,
    tools: crate::headers::Tools,
    p: &crate::motion::CuContexts,
    cu: &nc_encoder::Block,
    syntax: InterSyntax,
) {
    let subblock = match syntax {
        InterSyntax::SubblockMerge(index) => Some(index),
        _ => None,
    };
    if subblock_merge_signalled(tools, cu) {
        subblock_merge_flag(w, c, p.affine_ctx, subblock.is_some()).expect("valid flag");
    }
    match (syntax, subblock) {
        (_, Some(index)) => {
            subblock_merge_index(w, c, index).expect("valid sub-block merge index");
        }
        (InterSyntax::Merge(index), None) => {
            merge_index(w, c, tools.merge_candidates, index).expect("valid merge index");
        }
        _ => unreachable!("merge_data codes a merge"),
    }
}

/// Whether CU `cu` codes `merge_subblock_flag` (§7.3.11.7:
/// `MaxNumSubblockMergeCand` > 0, at least 8x8).
pub(crate) fn subblock_merge_signalled(
    tools: crate::headers::Tools,
    cu: &nc_encoder::Block,
) -> bool {
    tools.affine.enabled() && cu.width() >= 8 && cu.height() >= 8
}

/// Whether AMVP CU `cu` codes `inter_affine_flag` (§7.3.11.5: at least
/// 16x16).
pub(crate) fn affine_signalled(tools: crate::headers::Tools, cu: &nc_encoder::Block) -> bool {
    tools.affine.enabled() && cu.width() >= 16 && cu.height() >= 16
}

/// `merge_subblock_flag` at context `ctx` (affine neighbours).
pub(crate) fn subblock_merge_flag(
    io: &mut impl Bins,
    c: &mut Contexts,
    ctx: usize,
    subblock: bool,
) -> Result<bool> {
    io.bin(&mut c.subblock_merge[ctx], subblock)
}

/// §7.3.11.7 `merge_subblock_idx`: truncated unary up to
/// `MaxNumSubblockMergeCand` − 1, the first bin context coded, the rest
/// bypass (§9.3.4.2, Table 132).
pub(crate) fn subblock_merge_index(
    io: &mut impl Bins,
    c: &mut Contexts,
    index: usize,
) -> Result<usize> {
    let last = crate::affine_candidates::MAX_SUBBLOCK_MERGE - 1;
    if !io.bin(&mut c.subblock_merge_idx, index > 0)? {
        return Ok(0);
    }
    let mut decoded = 1;
    while decoded < last && io.bypass(u32::from(index > decoded), 1)? == 1 {
        decoded += 1;
    }
    Ok(decoded)
}

/// Estimated bits of a sub-block merge: the flag and index.
pub(crate) fn subblock_merge_bits(c: &Contexts, ctx: usize, index: usize) -> f32 {
    let last = crate::affine_candidates::MAX_SUBBLOCK_MERGE - 1;
    c.subblock_merge[ctx].bits(true)
        + c.subblock_merge_idx.bits(index > 0)
        + index.min(last - 1) as f32
}

/// `inter_affine_flag` (context `ctx`) and, with 6-parameter models
/// enabled, `cu_affine_type_flag` of an AMVP CU: `six` is `None` for a
/// translational CU, else whether its model has 6 parameters.
pub(crate) fn affine_flags(
    io: &mut impl Bins,
    c: &mut Contexts,
    tools: crate::headers::Tools,
    ctx: usize,
    six: Option<bool>,
) -> Result<Option<bool>> {
    if !io.bin(&mut c.affine[ctx], six.is_some())? {
        return Ok(None);
    }
    if !tools.affine.six() {
        return Ok(Some(false));
    }
    io.bin(&mut c.affine_type, six == Some(true)).map(Some)
}

/// §7.3.11.7 `merge_idx` of a list of `size` (`MaxNumMergeCand`):
/// truncated unary up to `size` − 1, the first bin context coded, the rest
/// bypass (§9.3.4.2, Table 132); absent from a one-entry list.
pub(crate) fn merge_index(
    io: &mut impl Bins,
    c: &mut Contexts,
    size: usize,
    index: usize,
) -> Result<usize> {
    debug_assert!(index < size);
    if size == 1 || !io.bin(&mut c.merge_idx, index > 0)? {
        return Ok(0);
    }
    let mut decoded = 1;
    while decoded < size - 1 && io.bypass(u32::from(index > decoded), 1)? == 1 {
        decoded += 1;
    }
    Ok(decoded)
}

/// Estimated bits of [`merge_index`], without changing state.
pub(crate) fn merge_index_bits(c: &Contexts, size: usize, index: usize) -> f32 {
    if size == 1 {
        return 0.;
    }
    c.merge_idx.bits(index > 0) + index.min(size - 2) as f32
}

/// The MVD of `mv` from predictor `pred` (both on `amvr`'s grid), in its
/// steps (§7.4.11.8: `MvdL0` is shifted by `AmvrShift`).
pub(crate) fn mvd_of(mv: Mv16, pred: Mv16, amvr: Amvr) -> [i32; 2] {
    let shift = amvr.shift();
    [(mv.x - pred.x) >> shift, (mv.y - pred.y) >> shift]
}

/// §7.3.10.5 `amvr_flag` and `amvr_precision_idx` (after a nonzero MVD):
/// the flag, then whether the precision is not half a sample (always, here),
/// then integer or four samples.
pub(crate) fn amvr_precision(io: &mut impl Bins, c: &mut Contexts, amvr: Amvr) -> Result<Amvr> {
    if !io.bin(&mut c.amvr_flag, amvr != Amvr::Quarter)? {
        return Ok(Amvr::Quarter);
    }
    if !io.bin(&mut c.amvr_not_half, true)? {
        return Err(nc_common::decode::DecodeError::Unsupported(
            "VVC half-sample AMVR",
        ));
    }
    Ok(if io.bin(&mut c.amvr_four, amvr == Amvr::FourSamples)? {
        Amvr::FourSamples
    } else {
        Amvr::Integer
    })
}

/// Estimated bits of [`amvr_precision`], without changing state.
fn amvr_bits(c: &Contexts, amvr: Amvr) -> f32 {
    c.amvr_flag.bits(amvr != Amvr::Quarter)
        + if amvr == Amvr::Quarter {
            0.
        } else {
            c.amvr_not_half.bits(true) + c.amvr_four.bits(amvr == Amvr::FourSamples)
        }
}

/// The cheapest way to code AMVP vector `mv`: its precision, predictor
/// index and estimated bits (MVD, `mvp_l0_flag` and the AMVR bins). A
/// coarser precision needs `mv` on its grid and a nonzero MVD (a zero MVD
/// codes no `amvr_flag`, which means quarter samples).
pub(crate) fn amvp_coding(c: &Contexts, p: &Predictors, mv: Mv16) -> (Amvr, usize, f32) {
    let mut best = (Amvr::Quarter, 0, f32::INFINITY);
    for amvr in Amvr::ALL.into_iter().filter(|a| a.holds(mv)) {
        for (index, &pred) in p.amvp[amvr as usize].iter().enumerate() {
            let delta = mvd_of(mv, pred, amvr);
            let zero = delta == [0; 2];
            if zero && amvr != Amvr::Quarter {
                continue;
            }
            let bits = mvd_bits(c, delta)
                + c.mvp.bits(index != 0)
                + if zero { 0. } else { amvr_bits(c, amvr) };
            if bits < best.2 {
                best = (amvr, index, bits);
            }
        }
    }
    best
}

/// The cheapest affine AMVP coding of quarter-sample `model` against the
/// predictors `preds`: `mvp_l0_flag`, the coded MVDs (the later control
/// points' relative to the first, §7.4.11.7) and their estimated bits.
pub(crate) fn affine_amvp_coding(
    c: &Contexts,
    preds: &[crate::affine::Model; crate::affine_candidates::AFFINE_AMVP],
    model: &crate::affine::Model,
) -> (usize, [[i32; 2]; 3], f32) {
    let mut best = (0, [[0; 2]; 3], f32::INFINITY);
    for (index, pred) in preds.iter().enumerate() {
        let mut mvd = [[0; 2]; 3];
        for i in 0..model.points() {
            mvd[i] = mvd_of(model.cp[i], pred.cp[i], Amvr::Quarter);
            if i > 0 {
                mvd[i] = [mvd[i][0] - mvd[0][0], mvd[i][1] - mvd[0][1]];
            }
        }
        let bits = mvd[..model.points()]
            .iter()
            .map(|&delta| mvd_bits(c, delta))
            .sum::<f32>()
            + c.mvp.bits(index != 0);
        if bits < best.2 {
            best = (index, mvd, bits);
        }
    }
    best
}

pub(crate) fn write_mvd(w: &mut CabacWriter, c: &mut Contexts, delta: [i32; 2]) {
    for d in delta {
        w.encode_bin(&mut c.mvd[0], d != 0);
    }
    for d in delta {
        if d != 0 {
            w.encode_bin(&mut c.mvd[1], d.unsigned_abs() > 1);
        }
    }
    for d in delta {
        if d.unsigned_abs() > 1 {
            let code = d.unsigned_abs() - 2;
            let prefix = ((code >> 1) + 1).ilog2();
            w.encode_bypass((1 << prefix) - 1, prefix);
            w.encode_bypass(code - (((1 << prefix) - 1) << 1), prefix + 2);
        }
        if d != 0 {
            w.encode_bypass(u32::from(d < 0), 1);
        }
    }
}

/// MVD contexts plus order-one exponential-Golomb tails of `delta` (in
/// steps), without changing state.
pub(crate) fn mvd_bits(c: &Contexts, delta: [i32; 2]) -> f32 {
    delta
        .into_iter()
        .map(|d| {
            let a = d.unsigned_abs();
            c.mvd[0].bits(a > 0)
                + if a == 0 {
                    0.
                } else {
                    c.mvd[1].bits(a > 1)
                        + 1.
                        + if a <= 1 {
                            0.
                        } else {
                            (2 * ((((a - 2) >> 1) + 1).ilog2()) + 2) as f32
                        }
                }
        })
        .sum()
}
