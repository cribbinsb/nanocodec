//! CDEF of decoded frames (AV1 §7.15): the frame header's strength sets
//! applied through the shared [`crate::cdef_map`] with the decoder's own
//! map of coded units and indices.
use super::frame_header::CdefParams;
pub(super) use crate::cdef_map::{CdefMap, CdefScratch};
use nc_common::Picture;
use nc_dsp::{CdefStrengths, DepthKernels, Pixel};

/// Apply the frame's CDEF to `recon` in place.
pub(super) fn apply<T: Pixel>(
    kernels: DepthKernels<'_>,
    recon: &mut Picture<T>,
    map: &CdefMap,
    params: &CdefParams,
    scratch: &mut CdefScratch<T>,
) {
    let strengths: Vec<CdefStrengths> = (0..1usize << params.bits)
        .map(|s| CdefStrengths {
            luma_primary: params.y_pri[s],
            luma_secondary: params.y_sec[s],
            chroma_primary: params.uv_pri[s],
            chroma_secondary: params.uv_sec[s],
        })
        .collect();
    crate::cdef_map::apply(kernels, recon, map, &strengths, params.damping, scratch);
}
