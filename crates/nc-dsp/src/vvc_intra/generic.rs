//! Lane-wise H.266 planar/DC/axis prediction. All sums fit signed 32 bits.
#![allow(unsafe_code, clippy::inline_always)]
use super::VvcIntraMode;
use crate::simd::I32Lanes;

/// # Safety
/// Called only with V's ISA enabled and a supported side divisible by its lanes.
#[inline(always)]
pub(super) unsafe fn predict<V: I32Lanes>(
    n: usize,
    mode: VvcIntraMode,
    top: &[i32],
    left: &[i32],
    corner: i32,
    out: &mut [u8],
) {
    let mode = match mode {
        VvcIntraMode::Angular { direction: 18, .. } => VvcIntraMode::Horizontal,
        VvcIntraMode::Angular { direction: 50, .. } => VvcIntraMode::Vertical,
        mode => mode,
    };
    let dc = (top[..n].iter().chain(&left[..n]).sum::<i32>() + n as i32) / (2 * n as i32);
    let scale = (2 * n.ilog2() - 2) >> 2;
    let distances: [i32; 32] = std::array::from_fn(|i| i as i32 + 1);
    let weights: [i32; 32] = std::array::from_fn(|i| 32 >> ((2 * i) >> scale).min(31));
    for y in 0..n {
        let wt = V::splat(weights[y]);
        let side = V::splat(left[y]);
        for x in (0..n).step_by(V::LANES) {
            let above = V::load(&top[x..]);
            let wl = V::load(&weights[x..]);
            let value = match mode {
                VvcIntraMode::Angular { .. } => {
                    unreachable!("angular directions use angular_generic")
                }
                VvcIntraMode::Planar => {
                    let distance = V::load(&distances[x..]);
                    let horizontal = V::splat(n as i32)
                        .sub(distance)
                        .mul(side)
                        .add(distance.mul_scalar(top[n]));
                    let vertical = above
                        .mul_scalar((n - 1 - y) as i32)
                        .add(V::splat(((y + 1) as i32) * left[n]));
                    horizontal
                        .add(vertical)
                        .add(V::splat(n as i32))
                        .sra(n.ilog2() + 1)
                }
                VvcIntraMode::Dc => V::splat(dc),
                VvcIntraMode::Horizontal => side,
                VvcIntraMode::Vertical => above,
            };
            let correction = match mode {
                VvcIntraMode::Angular { .. } => {
                    unreachable!("angular directions use angular_generic")
                }
                VvcIntraMode::Planar | VvcIntraMode::Dc => {
                    wl.mul(side.sub(value)).add(wt.mul(above.sub(value)))
                }
                VvcIntraMode::Horizontal => wt.mul(above.sub(V::splat(corner))),
                VvcIntraMode::Vertical => wl.mul(side.sub(V::splat(corner))),
            };
            let value = value.add(correction.add(V::splat(32)).sra(6));
            let dst = &mut out[y * n + x..y * n + x + V::LANES];
            dst.fill(0);
            value.add_to_u8(dst);
        }
    }
}

/// The ISA's `predict`: planar, DC and the axes on i32 lanes (`$small` for
/// 4x4, `$wide` otherwise); other angular directions on i16 lanes as wide
/// as the block side allows (`$a4`, `$a8`, `$a16`).
macro_rules! kernels {
    ($(#[$attr:meta])*; $small:ty, $wide:ty; $a4:ty, $a8:ty, $a16:ty) => {
        $(#[$attr])*
        pub(super) unsafe fn predict(n: usize, mode: super::VvcIntraMode, top: &[i32], left: &[i32], corner: i32, transposed_ok: bool, out: &mut [u8]) -> bool {
            use super::VvcIntraMode::Angular;
            // SAFETY: target feature enabled; every supported side is a
            // multiple of four and a lane count of at most the side is used.
            unsafe {
                match mode {
                    Angular { direction, luma } if direction != 18 && direction != 50 => match n {
                        4 => super::angular_generic::predict::<$a4>(n, direction, luma, top, left, corner, transposed_ok, out),
                        8 => super::angular_generic::predict::<$a8>(n, direction, luma, top, left, corner, transposed_ok, out),
                        _ => super::angular_generic::predict::<$a16>(n, direction, luma, top, left, corner, transposed_ok, out),
                    },
                    _ if n == 4 => { super::generic::predict::<$small>(n, mode, top, left, corner, out); false }
                    _ => { super::generic::predict::<$wide>(n, mode, top, left, corner, out); false }
                }
            }
        }
    };
}
pub(crate) use kernels;
