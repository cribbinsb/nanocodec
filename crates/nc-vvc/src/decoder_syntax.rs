//! Bounded H.266 §9.3 decoding of intra, skip, merge and AMVP CU syntax.
use crate::{
    contexts::Contexts,
    decoder_cabac::Reader,
    headers::Tools,
    motion::{Amvr, Motion, MotionGrid, Mv16, Predictors},
    quant_group::DeltaRead,
    syntax::{Coding, Inter, InterSyntax, merge_index},
    transform_tree,
};
use nc_common::decode::{DecodeError, Result};

/// Zero levels for the coefficient walk's encoder-side input, which a
/// decoding walk reads but does not use: the largest TU's samples.
const ZERO_LEVELS: [i32; 1 << (2 * transform_tree::MAX_TB_LOG2)] =
    [0; 1 << (2 * transform_tree::MAX_TB_LOG2)];

/// Coding unit `cu`; coefficient-free planes keep empty levels.
/// Rectangular CUs (multi-type tree leaves) are inter only in our streams.
/// `tools`: the SPS's optional tools; `grid`: the coded neighbours
/// (affine candidate lists), whose regular predictors are `p`.
pub(crate) fn block(
    r: &mut Reader<'_>,
    c: &mut Contexts,
    (inter, tools): (bool, Tools),
    (grid, p): (&MotionGrid, &Predictors),
    cu: nc_encoder::Block,
    delta_pending: bool,
) -> Result<Coding> {
    let neighbours = Neighbours { grid, p, cu, tools };
    let mut coding = Coding {
        inter: None,
        cu,
        levels: Default::default(),
        skip: false,
        intra_mode: crate::intra::DC,
        chroma_mode: crate::intra::DC,
        isp: None,
        qp_delta: None,
        mts: 0,
    };
    let mut delta = if delta_pending {
        DeltaRead::Pending
    } else {
        DeltaRead::Absent
    };
    if inter && r.bin(&mut c.skip[p.skip_ctx])? {
        coding.skip = true;
        coding.inter = Some(merge_data(r, c, &neighbours)?);
        return Ok(coding);
    }
    let intra = !inter || r.bin(&mut c.intra[p.intra_ctx])?;
    if intra && !cu.is_square() {
        return Err(DecodeError::Unsupported("VVC rectangular intra CU"));
    }
    if intra {
        coding.isp = crate::isp::code_split(r, c, tools, &cu, None)?;
        if coding.isp.is_some() && !crate::isp::supported(&cu, tools.mts.is_some()) {
            return Err(DecodeError::Unsupported(
                "VVC intra sub-partitions of this size",
            ));
        }
        coding.intra_mode = crate::intra::supported(crate::intra::mode(
            r,
            c,
            p.mpm,
            crate::intra::DC,
            coding.isp.is_some(),
        )?)?;
        coding.chroma_mode = crate::chroma::code(r, c, coding.intra_mode, coding.intra_mode)?;
        if let Some(split) = coding.isp {
            coding.levels = isp_units(r, c, &cu, split, &mut delta)?;
            coding.qp_delta = delta.value();
            return Ok(coding);
        }
    } else if r.bin(&mut c.merge)? {
        coding.inter = Some(merge_data(r, c, &neighbours)?);
    } else {
        coding.inter = Some(amvp(r, c, &neighbours)?);
        if !r.bin(&mut c.root_cbf)? {
            return Ok(coding);
        }
    }
    let max_tb_log2 = tools.max_tb_log2();
    let luma_flag = intra || transform_tree::splits(&cu, max_tb_log2);
    let tu = transform_tree::unit_of(&cu, max_tb_log2);
    let count = transform_tree::count(&cu, max_tb_log2);
    for index in 0..count {
        let unit = transform_unit(r, c, (tu.width(), tu.height()), luma_flag, &mut delta)?;
        for (levels, unit) in coding.levels.iter_mut().zip(unit) {
            transform_tree::place(levels, count, index, unit);
        }
    }
    coding.qp_delta = delta.value();
    if tools.mts.is_some() && crate::mts::signalled(tools.mts, intra, &cu, &coding.levels[0]) {
        coding.mts = crate::mts::code(r, &mut c.mts_idx, 0)?;
    }
    Ok(coding)
}

/// What the motion syntax of CU `cu` derives its candidates from.
struct Neighbours<'a> {
    grid: &'a MotionGrid,
    p: &'a Predictors,
    cu: nc_encoder::Block,
    tools: Tools,
}

/// §7.3.11.7 `merge_data` of a skip or merge CU.
fn merge_data(r: &mut Reader<'_>, c: &mut Contexts, n: &Neighbours<'_>) -> Result<Inter> {
    let subblock = crate::syntax::subblock_merge_signalled(n.tools, &n.cu)
        && crate::syntax::subblock_merge_flag(r, c, n.p.affine_ctx, false)?;
    if subblock {
        let index = crate::syntax::subblock_merge_index(r, c, 0)?;
        let list = crate::affine_candidates::merge_list(
            n.grid,
            &n.cu,
            crate::affine_candidates::MAX_SUBBLOCK_MERGE,
            n.tools.affine.six(),
        );
        return Ok(Inter {
            motion: Motion::Affine(list[index]),
            syntax: InterSyntax::SubblockMerge(index),
        });
    }
    let index = merge_index(r, c, n.tools.merge_candidates, 0)?;
    Ok(Inter {
        motion: Motion::Vector(n.p.merge[index]),
        syntax: InterSyntax::Merge(index),
    })
}

/// The AMVP motion of an inter CU (§7.3.11.5: affine flags, MVDs,
/// `mvp_l0_flag`, AMVR).
fn amvp(r: &mut Reader<'_>, c: &mut Contexts, n: &Neighbours<'_>) -> Result<Inter> {
    let six = if crate::syntax::affine_signalled(n.tools, &n.cu) {
        crate::syntax::affine_flags(r, c, n.tools, n.p.affine_ctx, None)?
    } else {
        None
    };
    if let Some(six) = six {
        return affine_amvp(r, c, n, six);
    }
    let mvd = read_mvd(r, c)?;
    let mvp = usize::from(r.bin(&mut c.mvp)?);
    let amvr = if mvd == [0; 2] {
        Amvr::Quarter
    } else {
        crate::syntax::amvr_precision(r, c, Amvr::Quarter)?
    };
    let mv = add_mvd(n.p.amvp[amvr as usize][mvp], mvd, amvr)?;
    Ok(Inter {
        motion: Motion::Vector(mv),
        syntax: InterSyntax::Amvp { mvp, amvr, mvd },
    })
}

/// An affine AMVP CU of a `six`-parameter model: one MVD per control point,
/// the later ones relative to the first (§8.5.5.? `cpMvLX`), quarter
/// samples (no affine AMVR).
fn affine_amvp(
    r: &mut Reader<'_>,
    c: &mut Contexts,
    n: &Neighbours<'_>,
    six: bool,
) -> Result<Inter> {
    let points = if six { 3 } else { 2 };
    let mut mvd = [[0; 2]; 3];
    for delta in &mut mvd[..points] {
        *delta = read_mvd(r, c)?;
    }
    let mvp = usize::from(r.bin(&mut c.mvp)?);
    let pred = crate::affine_candidates::amvp_list(n.grid, &n.cu, six)[mvp];
    let mut cp = [Mv16::ZERO; 3];
    for (i, point) in cp.iter_mut().enumerate().take(points) {
        let first = if i == 0 { [0; 2] } else { mvd[0] };
        let delta = [mvd[i][0] + first[0], mvd[i][1] + first[1]];
        *point = add_mvd(pred.cp[i], delta, Amvr::Quarter)?;
    }
    Ok(Inter {
        motion: Motion::Affine(crate::affine::Model { cp, six }),
        syntax: InterSyntax::AffineAmvp { mvp, mvd },
    })
}

/// Predictor `pred` plus difference `mvd` in `amvr` steps (§8.5.2.1:
/// `uLX` wraps to 18 bits; our streams stay inside the range).
fn add_mvd(pred: Mv16, mvd: [i32; 2], amvr: Amvr) -> Result<Mv16> {
    let shift = amvr.shift();
    let mv = Mv16 {
        x: pred.x + (mvd[0] << shift),
        y: pred.y + (mvd[1] << shift),
    };
    if mv.clip() != mv {
        return Err(DecodeError::Unsupported("VVC motion vector range"));
    }
    Ok(mv)
}

/// §7.3.11.10 `transform_unit` of luma size `size`, as `syntax::transform_unit`.
fn transform_unit(
    r: &mut Reader<'_>,
    c: &mut Contexts,
    (width, height): (usize, usize),
    luma_flag: bool,
    delta: &mut DeltaRead,
) -> Result<[Vec<i32>; 3]> {
    let cb = r.bin(&mut c.cbf_cb)?;
    let cr = r.bin(&mut c.cbf_cr[usize::from(cb)])?;
    let y = (!luma_flag && !cb && !cr) || r.bin(&mut c.cbf_y)?;
    if y || cb || cr {
        read_qp_delta(r, c, delta)?;
    }
    let mut levels: [Vec<i32>; 3] = Default::default();
    for (plane, present) in [y, cb, cr].into_iter().enumerate() {
        if present {
            let shift = usize::from(plane != 0);
            let size = (width >> shift, height >> shift);
            levels[plane] = crate::coeff::code_rect(
                r,
                &mut c.coeff,
                &ZERO_LEVELS[..transform_tree::coded_samples(size.0, size.1)],
                size,
                shift,
            )?;
        }
    }
    Ok(levels)
}

/// The transform units of an ISP CU, as `syntax::isp_units`: luma levels
/// of the sub-partitions back to back, the CU's chroma with the last.
fn isp_units(
    r: &mut Reader<'_>,
    c: &mut Contexts,
    cu: &nc_encoder::Block,
    split: crate::isp::Split,
    delta: &mut DeltaRead,
) -> Result<[Vec<i32>; 3]> {
    let mut levels: [Vec<i32>; 3] = Default::default();
    let mut coded = [false; crate::isp::PARTS];
    for (index, part) in crate::isp::parts(cu, split).iter().enumerate() {
        let last = index == crate::isp::PARTS - 1;
        let mut chroma = [false; 2];
        if last {
            chroma[0] = r.bin(&mut c.cbf_cb)?;
            chroma[1] = r.bin(&mut c.cbf_cr[usize::from(chroma[0])])?;
        }
        coded[index] = crate::isp::code_luma_flag(r, c, index, &coded[..index], false)?;
        if coded[index] || chroma.contains(&true) {
            read_qp_delta(r, c, delta)?;
        }
        if coded[index] {
            let size = (part.width(), part.height());
            let unit =
                crate::coeff::code_rect(r, &mut c.coeff, &ZERO_LEVELS[..part.area()], size, 0)?;
            transform_tree::place(&mut levels[0], crate::isp::PARTS, index, unit);
        }
        for (plane, present) in chroma.into_iter().enumerate() {
            if present {
                let size = (cu.width() / 2, cu.height() / 2);
                levels[plane + 1] = crate::coeff::code_rect(
                    r,
                    &mut c.coeff,
                    &ZERO_LEVELS[..size.0 * size.1],
                    size,
                    1,
                )?;
            }
        }
    }
    Ok(levels)
}

/// `cu_qp_delta_abs`/`cu_qp_delta_sign_flag` when `delta` is still to be
/// read (§7.3.11.10).
fn read_qp_delta(r: &mut Reader<'_>, c: &mut Contexts, delta: &mut DeltaRead) -> Result<()> {
    if *delta == DeltaRead::Pending {
        *delta = DeltaRead::Read(crate::quant_group::code_delta(r, &mut c.cu_qp_delta, 0)?);
    }
    Ok(())
}

pub(crate) fn read_mvd(r: &mut Reader<'_>, c: &mut Contexts) -> Result<[i32; 2]> {
    let nonzero = [r.bin(&mut c.mvd[0])?, r.bin(&mut c.mvd[0])?];
    let greater = [
        nonzero[0] && r.bin(&mut c.mvd[1])?,
        nonzero[1] && r.bin(&mut c.mvd[1])?,
    ];
    let mut delta = [0; 2];
    for i in 0..2 {
        if !nonzero[i] {
            continue;
        }
        let magnitude = if greater[i] {
            let mut prefix = 0;
            while r.bypass()? {
                prefix += 1;
                if prefix > 15 {
                    return Err(DecodeError::Unsupported("VVC MVD prefix"));
                }
            }
            2 + (((1 << prefix) - 1) << 1) + r.bypass_bits(prefix + 1)?
        } else {
            1
        };
        delta[i] = if r.bypass()? {
            -(magnitude as i32)
        } else {
            magnitude as i32
        };
    }
    Ok(delta)
}
