//! Three word-dot accumulators for the 8×8 secondary transform, following
//! dav2d's `stx8_8bpc_avx512icl` (BSD-2-Clause, `VideoLAN` and dav2d authors).

use core::arch::x86_64::{
    __m512i, _mm_mask_storeu_epi16, _mm512_add_epi32, _mm512_castsi512_si128,
    _mm512_castsi512_si256, _mm512_cvtepi8_epi16, _mm512_dpwssd_epi32, _mm512_extracti32x4_epi32,
    _mm512_loadu_si512, _mm512_mask_storeu_epi16, _mm512_packs_epi32, _mm512_permutex2var_epi8,
    _mm512_permutex2var_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_srai_epi32,
};

use super::{COEFF8X8_MAPPING, STX8_OUTPUTS};

// Two loads cover the 96 bytes of a pair of weight rows. Make one signed
// halfword pair per output, in the same order as the subsequent mapping.
const PERMUTE: [[u8; 64]; 3] = {
    let mut indices = [[0; 64]; 3];
    let mut group = 0;
    while group < 3 {
        let mut lane = 0;
        while lane < 16 {
            let output = group * 16 + lane;
            indices[group][2 * lane] = output as u8;
            indices[group][2 * lane + 1] = (output + 80) as u8;
            lane += 1;
        }
        group += 1;
    }
    indices
};

/// Select the mapped coefficient from the two lane-striped packs.
struct OutputMapping {
    lanes: [[u16; 32]; 2],
    mask: [u32; 2],
}

const fn output_mapping(map: usize, transpose: bool) -> OutputMapping {
    let mut lanes = [[0; 32]; 2];
    let mut mask = [0; 2];
    let mut index = 0;
    while index < STX8_OUTPUTS {
        let raster = COEFF8X8_MAPPING[map][index] as usize;
        let position = if transpose {
            raster
        } else {
            8 * (raster % 8) + raster / 8
        };
        let group = index / 16;
        let lane = index % 16;
        let source = 32 * (group / 2) + 8 * (lane / 4) + 4 * (group % 2) + lane % 4;
        lanes[position / 32][position % 32] = source as u16;
        mask[position / 32] |= 1 << (position % 32);
        index += 1;
    }
    OutputMapping { lanes, mask }
}

const OUTPUT: [[OutputMapping; 2]; 3] = [
    [output_mapping(0, false), output_mapping(0, true)],
    [output_mapping(1, false), output_mapping(1, true)],
    [output_mapping(2, false), output_mapping(2, true)],
];

#[target_feature(enable = "avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
pub(super) fn secondary8(
    cf: &mut [i16],
    stride: usize,
    (transpose, map): (bool, usize),
    kernel: &[i8],
    inputs: usize,
) {
    const N: usize = STX8_OUTPUTS;
    let coefficients = &cf[..inputs];
    assert!(cf.len() >= 7 * stride + 8);
    assert!(kernel.len() >= coefficients.len().next_multiple_of(2) * N);
    let mut sum = [_mm512_setzero_si512(); 3];
    // SAFETY: the table provides three complete vectors.
    let permute: [__m512i; 3] = PERMUTE
        .each_ref()
        .map(|permutation| unsafe { _mm512_loadu_si512(permutation.as_ptr().cast()) });
    for i in (0..coefficients.len()).step_by(2) {
        let a = i32::from(coefficients[i] as u16);
        let b = i32::from(coefficients.get(i + 1).copied().unwrap_or(0));
        let input = _mm512_set1_epi32(a | (b << 16));
        // SAFETY: the two overlapping loads cover the 96 bytes validated above.
        unsafe {
            let ptr = kernel.as_ptr().add(i * N);
            let first = _mm512_loadu_si512(ptr.cast());
            let last = _mm512_loadu_si512(ptr.add(32).cast());
            for group in 0..3 {
                let pairs = _mm512_permutex2var_epi8(first, permute[group], last);
                let weights = _mm512_cvtepi8_epi16(_mm512_castsi512_si256(pairs));
                sum[group] = _mm512_dpwssd_epi32(sum[group], input, weights);
            }
        }
    }
    let round = |dot| {
        let bias = _mm512_add_epi32(_mm512_set1_epi32(64), _mm512_srai_epi32::<31>(dot));
        _mm512_srai_epi32::<7>(_mm512_add_epi32(dot, bias))
    };
    let a = _mm512_packs_epi32(round(sum[0]), round(sum[1]));
    let b = _mm512_packs_epi32(round(sum[2]), _mm512_setzero_si512());
    let OutputMapping { lanes, mask } = &OUTPUT[map][usize::from(transpose)];
    cf[..32].fill(0);
    // SAFETY: the explicit span check covers all eight output columns.
    unsafe {
        for half in 0..2 {
            let index = _mm512_loadu_si512(lanes[half].as_ptr().cast());
            let out = _mm512_permutex2var_epi16(a, index, b);
            let dst = cf.as_mut_ptr().add(half * 4 * stride);
            let mask = mask[half];
            if stride == 8 {
                _mm512_mask_storeu_epi16(dst.cast(), mask, out);
            } else {
                _mm_mask_storeu_epi16(dst.cast(), mask as u8, _mm512_castsi512_si128(out));
                _mm_mask_storeu_epi16(
                    dst.add(stride).cast(),
                    (mask >> 8) as u8,
                    _mm512_extracti32x4_epi32::<1>(out),
                );
                _mm_mask_storeu_epi16(
                    dst.add(2 * stride).cast(),
                    (mask >> 16) as u8,
                    _mm512_extracti32x4_epi32::<2>(out),
                );
                _mm_mask_storeu_epi16(
                    dst.add(3 * stride).cast(),
                    (mask >> 24) as u8,
                    _mm512_extracti32x4_epi32::<3>(out),
                );
            }
        }
    }
}
