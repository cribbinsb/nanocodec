//! Lane arithmetic for bilinear DMVR, search SAD and affine PROF.
#![allow(unsafe_code, clippy::inline_always)]
use crate::simd::{I16Lanes, I32Lanes};

#[inline(always)]
pub(super) unsafe fn bilinear<V: I16Lanes>(
    size: (usize, usize),
    input: &nc_common::PlaneView<'_>,
    phases: (usize, usize),
    out: &mut [i16],
) {
    let (width, height) = size;
    let (fx, fy) = (phases.0 as u8, phases.1 as i16);
    let end = width / V::LANES * V::LANES;
    for y in 0..height {
        let top = input.row(y);
        let bottom = input.row(y + 1);
        let row = &mut out[y * width..][..width];
        for x in (0..end).step_by(V::LANES) {
            let horizontal = |row: &[u8]| {
                V::mul_add_u8_pair(&row[x..], &row[x + 1..], 16 - fx, fx)
                    .add(V::splat(2))
                    .sra(2)
            };
            // Both stages fit i16: horizontal <=1020, vertical sum <=16328.
            horizontal(top)
                .mul(V::splat(16 - fy))
                .add(horizontal(bottom).mul(V::splat(fy)))
                .add(V::splat(8))
                .sra(4)
                .store(&mut row[x..]);
        }
        for x in end..width {
            let horizontal = |row: &[u8]| {
                (i32::from(row[x]) * i32::from(16 - fx) + i32::from(row[x + 1]) * i32::from(fx) + 2)
                    >> 2
            };
            row[x] =
                ((horizontal(top) * i32::from(16 - fy) + horizontal(bottom) * i32::from(fy) + 8)
                    >> 4) as i16;
        }
    }
}

#[inline(always)]
pub(super) unsafe fn dmvr<V: I32Lanes>(
    size: (usize, usize),
    inputs: [&[i16]; 2],
) -> ([i32; 2], bool) {
    let (width, height) = size;
    let stride = width + 4;
    crate::vvc_dmvr::search(size, |dx, dy| {
        let mut sum = V::splat(0);
        let offset = dy * stride as i32 + dx;
        for y in (0..height).step_by(2) {
            for x in (0..width).step_by(V::LANES) {
                let at = ((y + 2) * stride + x + 2) as i32;
                // Widen before subtraction so even arbitrary signed i16 windows
                // agree with the scalar reference, not just normative 10-bit input.
                sum = sum.add(
                    V::load_i16(&inputs[0][(at + offset) as usize..])
                        .sub(V::load_i16(&inputs[1][(at - offset) as usize..]))
                        .abs(),
                );
            }
        }
        let mut lanes = [0i32; 8];
        sum.store(&mut lanes);
        lanes[..V::LANES].iter().sum()
    })
}

#[inline(always)]
pub(super) unsafe fn gradient<V: I32Lanes>(samples: &[i16], stride: usize, at: usize) -> [V; 2] {
    [
        V::load_i16(&samples[at + 1..])
            .sra(6)
            .sub(V::load_i16(&samples[at - 1..]).sra(6)),
        V::load_i16(&samples[at + stride..])
            .sra(6)
            .sub(V::load_i16(&samples[at - stride..]).sra(6)),
    ]
}

#[inline(always)]
pub(super) unsafe fn prof<V: I32Lanes>(
    samples: &[i16; 36],
    corrections: &[[i32; 2]; 16],
    out: &mut [i16; 16],
) {
    debug_assert_eq!(V::LANES, 4);
    for y in 0..4 {
        let at = (y + 1) * 6 + 1;
        // SAFETY: caller supplies a supported four-lane ISA and padded 6×6 block.
        let [gx, gy] = unsafe { gradient::<V>(samples, 6, at) };
        let cx = core::array::from_fn::<_, 4, _>(|x| corrections[y * 4 + x][0]);
        let cy = core::array::from_fn::<_, 4, _>(|x| corrections[y * 4 + x][1]);
        let delta = gx
            .mul(V::load(&cx))
            .add(gy.mul(V::load(&cy)))
            .max(V::splat(-8192))
            .min(V::splat(8191));
        let value = V::load_i16(&samples[at..]).add(delta).clamp16();
        let mut lanes = [0i32; 4];
        value.store(&mut lanes);
        for (dst, value) in out[y * 4..][..4].iter_mut().zip(lanes) {
            *dst = value as i16;
        }
    }
}
