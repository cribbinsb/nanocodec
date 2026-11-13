//! Benchmarks of AV2 directional intra prediction, whose interpolation
//! runs (4-tap luma, 2-tap chroma) are this ISA's kernels.

use nc_dsp::intra_prediction::av2::{AV2_INTRA_MAX_SIDE, Av2IntraEdges, Av2Predictor};
use nc_dsp::{BlockSize, Dsp};

use crate::bench::time;

/// Samples written on each edge (as the checkasm test).
const EDGE_SAMPLES: usize = 2 * AV2_INTRA_MAX_SIDE + 2;

/// One angle per zone, luma and chroma, at every side.
pub(crate) fn bench_av2_intra(
    dsp: &Dsp,
    src: &[u8],
    push: &mut impl FnMut(&'static str, BlockSize, f64),
) {
    let mut edges = Av2IntraEdges::new();
    edges.above_mut()[..EDGE_SAMPLES].copy_from_slice(&src[..EDGE_SAMPLES]);
    edges.left_mut()[..EDGE_SAMPLES].copy_from_slice(&src[EDGE_SAMPLES..2 * EDGE_SAMPLES]);
    edges.set_corner(src[2 * EDGE_SAMPLES]);
    for side in [4, 8, 16, 32, 64] {
        let size = BlockSize::new(side, side).expect("intra sizes are block sizes");
        let mut out = vec![0; side * side];
        for (name, angle, luma) in [
            ("av2_intra_z1_51_luma", 51, true),
            ("av2_intra_z2_129_luma", 129, true),
            ("av2_intra_z3_197_luma", 197, true),
            ("av2_intra_z1_51_chroma", 51, false),
            ("av2_intra_z3_197_chroma", 197, false),
        ] {
            let predictor = Av2Predictor::Directional { angle, luma };
            push(
                name,
                size,
                time(|| {
                    dsp.av2_intra_predict(&mut edges, predictor, side, &mut out);
                    u32::from(out[side * side - 1])
                }),
            );
        }
    }
}
