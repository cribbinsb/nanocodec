//! Four-lane optical-flow gradients, six-by-six flow sums and BDOF correction.
#![allow(unsafe_code, clippy::inline_always)]
use crate::simd::I32Lanes;
use crate::vvc_bdof::{MAX_PADDED, extend, flow};

#[inline(always)]
fn terms<V: I32Lanes>(gx: V, gy: V, difference: V) -> [V; 5] {
    let sx = gx.max(V::splat(-1)).min(V::splat(1));
    let sy = gy.max(V::splat(-1)).min(V::splat(1));
    [
        gx.abs(),
        gy.abs(),
        sx.mul(difference),
        sy.mul(difference),
        sy.mul(gx),
    ]
}

#[inline(always)]
pub(super) unsafe fn bdof<V: I32Lanes>(size: (usize, usize), inputs: [&[i16]; 2], out: &mut [u8]) {
    debug_assert_eq!(V::LANES, 4);
    let (width, height) = size;
    let stride = width + 2;
    let count = stride * (height + 2);
    let mut samples = [[0i16; MAX_PADDED]; 2];
    let mut gradients = [[[0i32; MAX_PADDED]; 2]; 2];
    for list in 0..2 {
        samples[list][..count].copy_from_slice(&inputs[list][..count]);
        for y in 1..=height {
            for x in (1..=width).step_by(4) {
                let at = y * stride + x;
                // SAFETY: four-lane wrapper, complete one-sample border.
                let values = unsafe { super::generic::gradient::<V>(&samples[list], stride, at) };
                for axis in 0..2 {
                    values[axis].store(&mut gradients[list][axis][at..]);
                }
            }
        }
        for axis in &mut gradients[list] {
            extend(axis, width, height);
        }
        extend(&mut samples[list], width, height);
    }
    for y0 in (0..height).step_by(4) {
        for x0 in (0..width).step_by(4) {
            let mut sums = [0i32; 5];
            let mut vector_sums = [V::splat(0); 5];
            for y in y0..y0 + 6 {
                let at = y * stride + x0;
                let gx = V::load(&gradients[0][0][at..])
                    .add(V::load(&gradients[1][0][at..]))
                    .sra(1);
                let gy = V::load(&gradients[0][1][at..])
                    .add(V::load(&gradients[1][1][at..]))
                    .sra(1);
                let difference = V::load_i16(&samples[1][at..])
                    .sra(4)
                    .sub(V::load_i16(&samples[0][at..]).sra(4));
                let values = terms(gx, gy, difference);
                for i in 0..5 {
                    vector_sums[i] = vector_sums[i].add(values[i]);
                }
                // The two remaining columns are scalar: no speculative read
                // crosses the validated 18×18 padded region at its last row.
                for at in at + 4..at + 6 {
                    let gx = (gradients[0][0][at] + gradients[1][0][at]) >> 1;
                    let gy = (gradients[0][1][at] + gradients[1][1][at]) >> 1;
                    let difference =
                        i32::from(samples[1][at] >> 4) - i32::from(samples[0][at] >> 4);
                    let values = [
                        gx.abs(),
                        gy.abs(),
                        gx.signum() * difference,
                        gy.signum() * difference,
                        gy.signum() * gx,
                    ];
                    for i in 0..5 {
                        sums[i] += values[i];
                    }
                }
            }
            for i in 0..5 {
                let mut lanes = [0i32; 4];
                vector_sums[i].store(&mut lanes);
                sums[i] += lanes.iter().sum::<i32>();
            }
            let vector = flow(sums);
            for y in y0..y0 + 4 {
                let at = (y + 1) * stride + x0 + 1;
                let dx = V::load(&gradients[0][0][at..]).sub(V::load(&gradients[1][0][at..]));
                let dy = V::load(&gradients[0][1][at..]).sub(V::load(&gradients[1][1][at..]));
                let value = V::load_i16(&samples[0][at..])
                    .add(V::load_i16(&samples[1][at..]))
                    .add(dx.mul_scalar(vector[0]))
                    .add(dy.mul_scalar(vector[1]))
                    .add(V::splat(64))
                    .sra(7)
                    .max(V::splat(0))
                    .min(V::splat(255));
                let mut lanes = [0i32; 4];
                value.store(&mut lanes);
                for (dst, value) in out[y * width + x0..][..4].iter_mut().zip(lanes) {
                    *dst = value as u8;
                }
            }
        }
    }
}
