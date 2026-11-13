//! H.266 §6.4.1 / §9.3 coding-tree split availability and context selection.
//! Normative decisions checked against BSD-Clear `VVdeC`, with decoder-local state.
use super::initialization::bank;
use crate::{contexts::Context, decoder_cabac::Reader};
use nc_common::decode::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Split {
    None,
    Quad,
    Horizontal,
    Vertical,
    TripleHorizontal,
    TripleVertical,
}
#[derive(Debug, Clone, Copy)]
pub(super) struct Node {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
    pub qt_depth: u8,
    pub mtt_depth: u8,
    pub implicit_depth: u8,
    pub parent: Split,
    pub child: usize,
    pub cclm_splits: [Split; 2],
}
impl Node {
    pub fn root(x: usize, y: usize, side: usize) -> Self {
        Self {
            x,
            y,
            width: side,
            height: side,
            qt_depth: 0,
            mtt_depth: 0,
            implicit_depth: 0,
            parent: Split::None,
            child: 0,
            cclm_splits: [Split::None; 2],
        }
    }
    pub fn outside(self, dimensions: [usize; 2]) -> bool {
        self.x + self.width > dimensions[0] || self.y + self.height > dimensions[1]
    }
    pub fn children(self, split: Split, implicit: bool) -> Vec<Self> {
        let rectangles = match split {
            Split::None => Vec::new(),
            Split::Quad => vec![
                (0, 0, self.width / 2, self.height / 2),
                (self.width / 2, 0, self.width / 2, self.height / 2),
                (0, self.height / 2, self.width / 2, self.height / 2),
                (
                    self.width / 2,
                    self.height / 2,
                    self.width / 2,
                    self.height / 2,
                ),
            ],
            Split::Horizontal => vec![
                (0, 0, self.width, self.height / 2),
                (0, self.height / 2, self.width, self.height / 2),
            ],
            Split::Vertical => vec![
                (0, 0, self.width / 2, self.height),
                (self.width / 2, 0, self.width / 2, self.height),
            ],
            Split::TripleHorizontal => vec![
                (0, 0, self.width, self.height / 4),
                (0, self.height / 4, self.width, self.height / 2),
                (0, 3 * self.height / 4, self.width, self.height / 4),
            ],
            Split::TripleVertical => vec![
                (0, 0, self.width / 4, self.height),
                (self.width / 4, 0, self.width / 2, self.height),
                (3 * self.width / 4, 0, self.width / 4, self.height),
            ],
        };
        rectangles
            .into_iter()
            .enumerate()
            .map(|(index, (x, y, width, height))| Self {
                x: self.x + x,
                y: self.y + y,
                width,
                height,
                qt_depth: self.qt_depth + u8::from(split == Split::Quad),
                mtt_depth: if split == Split::Quad {
                    0
                } else {
                    self.mtt_depth + 1
                },
                implicit_depth: self.implicit_depth
                    + u8::from(implicit && matches!(split, Split::Horizontal | Split::Vertical)),
                parent: split,
                child: index,
                cclm_splits: if self.width == 64 && self.height == 64 {
                    [split, Split::None]
                } else if self.width == 64 && self.height == 32 {
                    [self.cclm_splits[0], split]
                } else {
                    self.cclm_splits
                },
            })
            .collect()
    }
}
#[derive(Debug, Clone, Copy)]
pub(super) struct Limits {
    pub min_qt: usize,
    pub min_cb: usize,
    pub max_bt: usize,
    pub max_tt: usize,
    pub max_depth: u8,
    pub chroma: bool,
    pub dual: bool,
    pub inter_only: bool,
}
#[derive(Debug, Clone, Copy)]
#[allow(clippy::struct_excessive_bools)] // Independently available normative split modes.
struct Choices {
    none: bool,
    quad: bool,
    bh: bool,
    bv: bool,
    th: bool,
    tv: bool,
}
impl Choices {
    fn new(n: Node, limits: Limits, dimensions: [usize; 2]) -> Self {
        let Limits {
            min_qt,
            min_cb,
            max_bt,
            max_tt,
            max_depth,
            chroma,
            dual,
            inter_only,
        } = limits;
        let (w, h) = (n.width, n.height);
        let mut c = Self {
            none: true,
            quad: true,
            bh: true,
            bv: true,
            th: true,
            tv: true,
        };
        if dual && (w > 64 || h > 64) {
            return Self {
                none: false,
                quad: true,
                bh: false,
                bv: false,
                th: false,
                tv: false,
            };
        }
        c.quad = matches!(n.parent, Split::None | Split::Quad) && w > min_qt && (!chroma || w > 8);
        let mut mtt = n.mtt_depth < max_depth + n.implicit_depth;
        let right_inside = n.x + w <= dimensions[0];
        let bottom_inside = n.y + h <= dimensions[1];
        if !right_inside || !bottom_inside {
            let binary = w <= max_bt && h <= max_bt && w <= 64 && h <= 64 && mtt;
            c.none = false;
            c.th = false;
            c.tv = false;
            c.quad |= !binary;
            c.bh = binary && !bottom_inside && (right_inside || !c.quad);
            c.bv = binary && bottom_inside && !right_inside && (!chroma || w > 8);
            c.quad |= !c.bh && !c.bv;
            return c;
        }
        mtt &= w > min_cb || h > min_cb;
        mtt &= (w <= max_bt && h <= max_bt) || (w <= max_tt && h <= max_tt);
        if !mtt {
            c.bh = false;
            c.bv = false;
            c.th = false;
            c.tv = false;
            return c;
        }
        c.bh = w <= max_bt && h <= max_bt && h > min_cb && (w <= 64 || h > 64);
        c.bv = w <= max_bt && h <= max_bt && w > min_cb && (w > 64 || h <= 64);
        if n.child == 1 {
            c.bh &= n.parent != Split::TripleHorizontal;
            c.bv &= n.parent != Split::TripleVertical;
        }
        c.th = w <= max_tt && h <= max_tt && w <= 64 && h <= 64 && h > 2 * min_cb;
        c.tv = w <= max_tt && h <= max_tt && w <= 64 && h <= 64 && w > 2 * min_cb;
        if chroma {
            let area = w * h / 4;
            c.bh &= area > 16;
            c.bv &= area > 16 && w > 8;
            c.th &= area > 32;
            c.tv &= area > 32 && w > 16;
        }
        if inter_only {
            if w * h == 32 {
                c.bh = false;
                c.bv = false;
            }
            if w * h == 64 {
                c.th = false;
                c.tv = false;
            }
        }
        c
    }
}
pub(super) struct Models {
    split: [Context; 9],
    quad: [Context; 6],
    direction: [Context; 5],
    binary: [Context; 4],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            split: bank(
                [
                    [18, 27, 15, 18, 28, 45, 26, 7, 23],
                    [11, 35, 53, 12, 6, 30, 13, 15, 31],
                    [19, 28, 38, 27, 29, 38, 20, 30, 31],
                ],
                [12, 13, 8, 8, 13, 12, 5, 9, 9],
                qp,
                kind,
            ),
            quad: bank(
                [
                    [26, 36, 38, 18, 34, 21],
                    [20, 14, 23, 18, 19, 6],
                    [27, 6, 15, 25, 19, 37],
                ],
                [0, 8, 8, 12, 12, 8],
                qp,
                kind,
            ),
            direction: bank(
                [
                    [43, 42, 37, 42, 44],
                    [43, 35, 37, 34, 52],
                    [43, 42, 29, 27, 44],
                ],
                [9, 8, 9, 8, 5],
                qp,
                kind,
            ),
            binary: bank(
                [[28, 29, 28, 29], [43, 37, 21, 22], [36, 45, 36, 45]],
                [12, 13, 12, 13],
                qp,
                kind,
            ),
        }
    }
    pub fn read(
        &mut self,
        r: &mut Reader<'_>,
        n: Node,
        limits: Limits,
        dimensions: [usize; 2],
        neighbours: [Option<Node>; 2],
    ) -> Result<Split> {
        let c = Choices::new(n, limits, dimensions);
        let hor = usize::from(c.bh) + usize::from(c.th);
        let ver = usize::from(c.bv) + usize::from(c.tv);
        let total = 2 * usize::from(c.quad) + hor + ver;
        if total == 0 {
            return Ok(Split::None);
        }
        let [left, above] = neighbours;
        if c.none {
            let context = usize::from(left.is_some_and(|b| b.height < n.height))
                + usize::from(above.is_some_and(|b| b.width < n.width))
                + [0, 0, 0, 3, 3, 6, 6][total];
            if !r.bin(&mut self.split[context])? {
                return Ok(Split::None);
            }
        }
        if c.quad {
            let context = usize::from(left.is_some_and(|b| b.qt_depth > n.qt_depth))
                + usize::from(above.is_some_and(|b| b.qt_depth > n.qt_depth))
                + if n.qt_depth >= 2 { 3 } else { 0 };
            if hor + ver == 0 || r.bin(&mut self.quad[context])? {
                return Ok(Split::Quad);
            }
        }
        let vertical = if ver == 0 {
            false
        } else if hor == 0 {
            true
        } else {
            let context = if ver < hor {
                3
            } else if ver > hor {
                4
            } else if let (Some(left), Some(above)) = (left, above) {
                match (n.width / above.width).cmp(&(n.height / left.height)) {
                    std::cmp::Ordering::Equal => 0,
                    std::cmp::Ordering::Less => 1,
                    std::cmp::Ordering::Greater => 2,
                }
            } else {
                0
            };
            r.bin(&mut self.direction[context])?
        };
        let binary = if vertical { c.bv } else { c.bh };
        let triple = if vertical { c.tv } else { c.th };
        let binary = binary
            && (!triple
                || r.bin(
                    &mut self.binary[usize::from(n.mtt_depth <= 1) + 2 * usize::from(vertical)],
                )?);
        Ok(match (vertical, binary) {
            (false, true) => Split::Horizontal,
            (true, true) => Split::Vertical,
            (false, false) => Split::TripleHorizontal,
            (true, false) => Split::TripleVertical,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn limits() -> Limits {
        Limits {
            min_qt: 8,
            min_cb: 4,
            max_bt: 32,
            max_tt: 32,
            max_depth: 3,
            chroma: false,
            dual: true,
            inter_only: false,
        }
    }
    #[test]
    fn triple_middle_cannot_repeat_parallel_binary_split() {
        for (triple, vertical) in [
            (Split::TripleHorizontal, false),
            (Split::TripleVertical, true),
        ] {
            let children = Node::root(0, 0, 32).children(triple, false);
            let choices = Choices::new(children[1], limits(), [64, 64]);
            assert!(!choices.quad);
            assert_eq!(choices.bh, vertical);
            assert_eq!(choices.bv, !vertical);
            assert_eq!(
                children.iter().map(|n| n.width * n.height).sum::<usize>(),
                32 * 32
            );
        }
    }
    #[test]
    fn picture_boundary_forces_split_and_extends_implicit_depth() {
        let root = Node::root(0, 224, 32);
        let choices = Choices::new(root, limits(), [416, 240]);
        assert!(!choices.none && choices.bh && !choices.bv && !choices.th && !choices.tv);
        let children = root.children(Split::Horizontal, true);
        assert_eq!(children[0].implicit_depth, 1);
        assert!(!children[0].outside([416, 240]));
        assert!(children[1].outside([416, 240]));
    }
    #[test]
    fn smallest_chroma_unit_cannot_split() {
        let limits = Limits {
            chroma: true,
            ..limits()
        };
        let choices = Choices::new(Node::root(0, 0, 8), limits, [32, 32]);
        assert!(choices.none);
        assert!(!choices.quad && !choices.bh && !choices.bv && !choices.th && !choices.tv);
    }
}
