//! General intra tools and transform-skip flags (§7.3.11), sharing ordinary MPM syntax.
//! Contexts and conditional syntax checked against BSD-Clear `VVdeC`.
use super::initialization::bank;
use super::{Context, DecodeError, Parser, Reader, Result, Split, Transform, Unit, intra};

pub(super) struct Models {
    bdpcm: [Context; 4],
    mip: [Context; 4],
    reference_line: [Context; 2],
    skip: [Context; 2],
    bdpcm_cbf: [Context; 3],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            bdpcm: bank(
                [[19, 21, 0, 28], [40, 36, 0, 13], [19, 35, 1, 27]],
                [1, 4, 1, 0],
                qp,
                kind,
            ),
            mip: bank(
                [[56, 57, 50, 26], [41, 57, 58, 26], [33, 49, 50, 25]],
                [9, 10, 9, 6],
                qp,
                kind,
            ),
            reference_line: bank([[25, 59], [25, 58], [25, 60]], [5, 8], qp, kind),
            skip: bank([[25, 17], [25, 9], [25, 9]], [1, 1], qp, kind),
            bdpcm_cbf: [
                bank::<1, 1>([[6], [5], [12]], [1], qp, kind)[0],
                bank::<1, 1>([[37], [28], [21]], [0], qp, kind)[0],
                bank::<1, 1>([[45], [45], [36]], [0], qp, kind)[0],
            ],
        }
    }
}
impl Parser<'_> {
    fn bdpcm(&mut self, r: &mut Reader<'_>, unit: &mut Unit, channel: usize) -> Result<()> {
        let Some(log2) = self.sequence.transforms.skip_log2 else {
            return Ok(());
        };
        let node = unit.node;
        if self.sequence.transforms.bdpcm
            && node.width >> channel <= 1 << log2
            && node.height >> channel <= 1 << log2
            && r.bin(&mut self.intra_models.bdpcm[2 * channel])?
        {
            unit.bdpcm[channel] =
                1 + u8::from(r.bin(&mut self.intra_models.bdpcm[2 * channel + 1])?);
        }
        Ok(())
    }
    pub(super) fn luma_prediction(&mut self, r: &mut Reader<'_>, unit: &mut Unit) -> Result<()> {
        self.bdpcm(r, unit, 0)?;
        if unit.bdpcm[0] != 0 {
            unit.luma_mode = if unit.bdpcm[0] == 2 {
                intra::VERTICAL
            } else {
                intra::HORIZONTAL
            };
            return Ok(());
        }
        let node = unit.node;
        if self.sequence.tools.mip {
            let context = if node.width > 2 * node.height || node.height > 2 * node.width {
                3
            } else {
                [
                    (node.x.wrapping_sub(1), node.y),
                    (node.x, node.y.wrapping_sub(1)),
                ]
                .into_iter()
                .filter(|&(x, y)| self.neighbour(x, y, 0).is_some_and(|u| u.mip.is_some()))
                .count()
            };
            if r.bin(&mut self.intra_models.mip[context])? {
                let transpose = r.bypass()?;
                let count: u32 = if node.width == 4 && node.height == 4 {
                    16
                } else if node.width == 4
                    || node.height == 4
                    || (node.width == 8 && node.height == 8)
                {
                    8
                } else {
                    6
                };
                let bits = count.ilog2();
                let threshold = (1 << (bits + 1)) - count;
                let mut mode = r.bypass_bits(bits)?;
                if mode >= threshold {
                    mode = 2 * mode + u32::from(r.bypass()?) - threshold;
                }
                unit.mip = Some((mode as u8, transpose));
                return Ok(());
            }
        }
        if self.sequence.tools.mrl
            && !node.y.is_multiple_of(1 << self.sequence.ctu_log2)
            && r.bin(&mut self.intra_models.reference_line[0])?
        {
            unit.reference_line = if r.bin(&mut self.intra_models.reference_line[1])? {
                2
            } else {
                1
            };
        }
        self.isp_mode(r, unit)?;
        let left = self
            .neighbour(node.x.wrapping_sub(1), node.y + node.height - 1, 0)
            .map_or(intra::PLANAR, |u| u.luma_mode);
        let above = if node.y.is_multiple_of(1 << self.sequence.ctu_log2) {
            intra::PLANAR
        } else {
            self.neighbour(node.x + node.width - 1, node.y.wrapping_sub(1), 0)
                .map_or(intra::PLANAR, |u| u.luma_mode)
        };
        let candidates = intra::mpm(left, above);
        unit.luma_mode = if unit.reference_line == 0 {
            intra::mode(
                r,
                &mut self.contexts,
                candidates,
                intra::PLANAR,
                unit.isp.is_some(),
            )?
        } else {
            let mut index = 1;
            while index < 5 && r.bypass()? {
                index += 1;
            }
            candidates[index]
        };
        Ok(())
    }
    fn isp_mode(&mut self, r: &mut Reader<'_>, unit: &mut Unit) -> Result<()> {
        let node = unit.node;
        if self.sequence.tools.isp
            && unit.reference_line == 0
            && node.width * node.height > 16
            && node.width.max(node.height) <= 1 << self.sequence.max_transform_log2
            && r.bin(&mut self.contexts.isp_mode)?
        {
            unit.isp = Some(if r.bin(&mut self.contexts.isp_split)? {
                crate::isp::Split::Vertical
            } else {
                crate::isp::Split::Horizontal
            });
            unit.transforms = super::transform_units::isp_units(node, unit.isp.unwrap());
        }
        Ok(())
    }
    pub(super) fn chroma_prediction(&mut self, r: &mut Reader<'_>, unit: &mut Unit) -> Result<()> {
        self.bdpcm(r, unit, 1)?;
        if unit.bdpcm[1] != 0 {
            unit.chroma_mode = if unit.bdpcm[1] == 2 {
                intra::VERTICAL
            } else {
                intra::HORIZONTAL
            };
            return Ok(());
        }
        let node = unit.node;
        if !unit.combined {
            unit.luma_mode = self
                .neighbour(node.x + node.width / 2, node.y + node.height / 2, 0)
                .ok_or(DecodeError::Malformed("VVC missing collocated luma"))?
                .luma_mode;
        }
        unit.chroma_mode =
            if self.sequence.tools.cclm && self.cclm_allowed(unit) && r.bin(&mut self.cclm[0])? {
                67 + if r.bin(&mut self.cclm[1])? {
                    1 + u8::from(r.bypass()?)
                } else {
                    0
                }
            } else {
                crate::chroma::code(r, &mut self.contexts, unit.luma_mode, unit.luma_mode)?
            };
        Ok(())
    }
    fn cclm_allowed(&self, unit: &Unit) -> bool {
        if self.header.kind != 2 || unit.combined || self.sequence.ctu_log2 <= 5 {
            return true;
        }
        let [first, second] = unit.node.cclm_splits;
        let chroma = first == Split::Quad
            || first == Split::None
            || (first == Split::Horizontal && matches!(second, Split::None | Split::Vertical));
        let depth = self.sequence.ctu_log2 - 6;
        let luma = self
            .neighbour(unit.node.x, unit.node.y, 0)
            .is_some_and(|u| {
                // A 64×64 luma node using ISP cannot supply CCLM, even
                // though its coding tree has no further split (§8.4.4.2).
                u.node.qt_depth > depth || (u.node.mtt_depth == 0 && u.isp.is_none())
            });
        chroma && luma
    }
    pub(super) fn coded_flag(
        &mut self,
        r: &mut Reader<'_>,
        unit: &Unit,
        plane: usize,
        previous: bool,
    ) -> Result<bool> {
        let channel = usize::from(plane != 0);
        let context = if unit.bdpcm[channel] != 0 {
            &mut self.intra_models.bdpcm_cbf[plane]
        } else {
            match plane {
                0 => &mut self.contexts.cbf_y,
                1 => &mut self.contexts.cbf_cb,
                _ => &mut self.contexts.cbf_cr[usize::from(previous)],
            }
        };
        r.bin(context)
    }
    pub(super) fn transform_skip(
        &mut self,
        r: &mut Reader<'_>,
        unit: &Unit,
        transform: &mut Transform,
        plane: usize,
    ) -> Result<()> {
        let channel = usize::from(plane != 0);
        transform.skip[plane] = unit.bdpcm[channel] != 0;
        let block = if plane == 0 {
            transform.block
        } else {
            transform.chroma_block.unwrap()
        };
        if !transform.skip[plane]
            && unit.sbt.is_none()
            && (unit.isp.is_none() || plane != 0)
            && self.sequence.transforms.skip_log2.is_some_and(|log2| {
                block.width() >> channel <= 1 << log2 && block.height() >> channel <= 1 << log2
            })
        {
            transform.skip[plane] = r.bin(&mut self.intra_models.skip[channel])?;
        }
        Ok(())
    }
}
