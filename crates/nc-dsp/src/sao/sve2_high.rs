//! SVE2 Main 10/12 edge-offset application. The word lookup follows
//! libhevc's NEON offset table, with predicated source and output rows.

use nc_common::BitDepth;

use super::Rect;

core::arch::global_asm!(include_str!("sve2_high.S"), options(raw));

#[repr(C)]
struct Job {
    source: *const u16,
    output: *mut u16,
    source_stride: usize,
    output_stride: usize,
    neighbour_a: isize,
    neighbour_b: isize,
    width: usize,
    rows: usize,
    maximum: i16,
    table: [i16; 8],
}

const _: () = {
    assert!(core::mem::offset_of!(Job, maximum) == 64);
    assert!(core::mem::offset_of!(Job, table) == 66);
};

unsafe extern "C" {
    fn nc_sao_edge_u16_sve2(job: *const Job);
}

/// # Safety
/// SVE2 is enabled, and dispatch validated the destination rectangle and
/// the source extent for both neighbours.
pub(crate) unsafe fn apply(
    input: &[u16],
    ss: usize,
    out: &mut [u16],
    ds: usize,
    (a, b): (isize, isize),
    [oa, ob, oc, od]: [i16; 4],
    rect: Rect,
    depth: BitDepth,
) {
    let width = rect.x1 - rect.x0;
    let rows = rect.y1 - rect.y0;
    if width == 0 || rows == 0 {
        return;
    }
    if depth.bits() >= i16::BITS {
        // SAFETY: NEON is part of the SVE2 tier, and the geometry is unchanged.
        unsafe {
            super::high_simd::neon::apply(
                input,
                ss,
                out,
                ds,
                (a, b),
                [oa, ob, oc, od],
                rect,
                depth,
            );
        }
        return;
    }
    let source = rect.y0 * ss + rect.x0;
    let output = rect.y0 * ds + rect.x0;
    let job = Job {
        // SAFETY: the validated rectangle begins within each complete plane.
        source: unsafe { input.as_ptr().add(source) },
        // SAFETY: at least one valid output sample is present.
        output: unsafe { out.as_mut_ptr().add(output) },
        source_stride: ss.wrapping_mul(2),
        output_stride: ds.wrapping_mul(2),
        neighbour_a: a * 2,
        neighbour_b: b * 2,
        width,
        rows,
        maximum: depth.max() as i16,
        table: [oa, ob, 0, oc, od, 0, 0, 0],
    };
    // SAFETY: the source footprint includes both neighbours, the output is
    // distinct, and masked tail loads and stores stay inside the rectangle.
    unsafe { nc_sao_edge_u16_sve2(&raw const job) };
}
