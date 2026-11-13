//! H.266 transform-skip residual syntax, sharing CABAC, scans and Rice decoding.
//! Contexts and three-pass reconstruction follow BSD-Clear `VVdeC`.
use super::{
    Context, DecodeError, Reader, Result,
    coeff::{Scan, remainder},
    initialization::bank,
};

pub(super) struct Models {
    group: [Context; 3],
    significant: [Context; 3],
    parity: Context,
    greater: [Context; 4],
    large: [Context; 4],
    sign: [Context; 6],
}
impl Models {
    pub fn new(qp: u32, kind: u8) -> Self {
        Self {
            group: bank(
                [[18, 35, 45], [18, 12, 29], [18, 20, 38]],
                [5, 8, 8],
                qp,
                kind,
            ),
            significant: bank(
                [[25, 50, 37], [40, 35, 44], [25, 28, 38]],
                [13, 13, 8],
                qp,
                kind,
            ),
            parity: bank::<1, 1>([[11], [3], [11]], [6], qp, kind)[0],
            greater: bank(
                [[3, 4, 4, 5], [2, 10, 3, 3], [10, 3, 3, 3]],
                [1, 1, 1, 1],
                qp,
                kind,
            ),
            large: bank(
                [[19, 11, 4, 6], [18, 11, 4, 28], [11, 5, 5, 14]],
                [4, 2, 1, 6],
                qp,
                kind,
            ),
            sign: bank(
                [
                    [35, 25, 46, 28, 33, 38],
                    [5, 10, 53, 43, 25, 46],
                    [12, 17, 46, 28, 25, 46],
                ],
                [1, 4, 4, 5, 8, 8],
                qp,
                kind,
            ),
        }
    }
}
struct Walk {
    width: usize,
    values: Vec<i32>,
    budget: usize,
    bdpcm: bool,
}
impl Walk {
    fn neighbours(&self, position: usize) -> [i32; 2] {
        [
            if position.is_multiple_of(self.width) {
                0
            } else {
                self.values[position - 1]
            },
            if position < self.width {
                0
            } else {
                self.values[position - self.width]
            },
        ]
    }
    fn first_pass(
        &mut self,
        reader: &mut Reader<'_>,
        models: &mut Models,
        positions: &[usize],
        signs: &mut [Option<bool>],
    ) -> Result<usize> {
        let mut count = 0;
        let mut parsed = 0;
        for (i, &position) in positions.iter().enumerate() {
            if self.budget < 4 {
                break;
            }
            let neighbours = self.neighbours(position);
            let nonzero = neighbours.iter().filter(|&&v| v != 0).count();
            let significant = if count == 0 && i + 1 == positions.len() {
                true
            } else {
                self.budget -= 1;
                reader.bin(&mut models.significant[nonzero])?
            };
            if significant {
                let [left, above] = neighbours;
                let sign_context =
                    if (left == 0 && above == 0) || left.signum() * above.signum() < 0 {
                        0
                    } else if left >= 0 && above >= 0 {
                        1
                    } else {
                        2
                    };
                let negative =
                    reader.bin(&mut models.sign[sign_context + 3 * usize::from(self.bdpcm)])?;
                let large = reader.bin(&mut models.large[if self.bdpcm { 3 } else { nonzero }])?;
                self.budget -= 2;
                let parity = if large {
                    self.budget -= 1;
                    reader.bin(&mut models.parity)?
                } else {
                    false
                };
                self.values[position] =
                    (1 + i32::from(large) + i32::from(parity)) * if negative { -1 } else { 1 };
                signs[i] = Some(negative);
                count += 1;
            }
            parsed += 1;
        }
        Ok(parsed)
    }
    fn second_pass(
        &mut self,
        reader: &mut Reader<'_>,
        models: &mut Models,
        positions: &[usize],
    ) -> Result<usize> {
        let mut parsed = 0;
        for &position in positions {
            if self.budget < 4 {
                break;
            }
            let mut value = self.values[position].abs();
            for (i, context) in models.greater.iter_mut().enumerate() {
                if value >= 2 * (i as i32 + 1) {
                    value += 2 * i32::from(reader.bin(context)?);
                    self.budget -= 1;
                }
            }
            self.values[position] = value;
            parsed += 1;
        }
        Ok(parsed)
    }
    fn group(
        &mut self,
        reader: &mut Reader<'_>,
        models: &mut Models,
        positions: &[usize],
    ) -> Result<()> {
        let mut signs = [None; 16];
        let first = self.first_pass(reader, models, positions, &mut signs)?;
        let second = self.second_pass(reader, models, positions)?;
        for (i, &position) in positions.iter().enumerate() {
            let cutoff = if i < second {
                10
            } else if i < first {
                2
            } else {
                0
            };
            let mut value = self.values[position].abs();
            if value >= cutoff {
                value += (remainder(reader, 1)? as i32) << u32::from(i < first);
                if value != 0 && i >= first {
                    signs[i] = Some(reader.bypass()?);
                }
            }
            if !self.bdpcm && cutoff != 0 && value != 0 {
                let prediction = self
                    .neighbours(position)
                    .map(i32::abs)
                    .into_iter()
                    .max()
                    .unwrap();
                value = if value == 1 && prediction > 0 {
                    prediction
                } else {
                    value - i32::from(value <= prediction)
                };
            }
            self.values[position] = value;
        }
        for (&position, negative) in positions.iter().zip(signs) {
            if negative == Some(true) {
                self.values[position] = -self.values[position];
            }
            if !(-32768..=32767).contains(&self.values[position]) {
                return Err(DecodeError::Malformed(
                    "VVC transform-skip coefficient range",
                ));
            }
        }
        Ok(())
    }
}

pub(super) fn read(
    reader: &mut Reader<'_>,
    models: &mut Models,
    size: (usize, usize),
    bdpcm: bool,
) -> Result<Vec<i32>> {
    let (width, height) = size;
    let scan = Scan::new(width, height);
    let mut walk = Walk {
        width,
        values: vec![0; width * height],
        budget: width * height * 7 / 4,
        bdpcm,
    };
    let mut groups = vec![false; scan.groups.len()];
    let mut any = false;
    for (index, &raster) in scan.groups.iter().enumerate() {
        let left = raster % scan.columns != 0 && groups[raster - 1];
        let above = raster >= scan.columns && groups[raster - scan.columns];
        let significant = (index + 1 == groups.len() && !any)
            || reader.bin(&mut models.group[usize::from(left) + usize::from(above)])?;
        if significant {
            groups[raster] = true;
            any = true;
            let start = index * scan.group_size;
            walk.group(
                reader,
                models,
                &scan.positions[start..start + scan.group_size],
            )?;
        }
    }
    Ok(walk.values)
}
