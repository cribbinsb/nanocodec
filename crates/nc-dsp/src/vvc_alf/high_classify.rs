//! High-bit-depth (`u16`) VVC ALF 2x2-subsampled Laplacian group sums and
//! 4x4 block classification across AVX2, AVX-512, NEON, and SIMD128.
//!
//! At up to [`super::VVC_ALF_MAX_BIT_DEPTH`] (10 bits), each Laplacian
//! `|2c - a - b|` is at most `2046`, each four-column group sum is at most
//! `8184` (fitting signed 16-bit lanes), and each block sum of two groups
//! over four grid rows is at most `65472` (`< 2^16`). Consequently, the
//! cross products `d1 * hv0` and `hv1 * d0` in `classify` stay below
//! `65472^2 = 4_286_582_784 < 2^32` and are compared as **unsigned** 32-bit
//! products (`> 2^31` overflows signed 32-bit comparison).

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx2 {
    use core::arch::x86_64::{
        __m256i, _mm_cvtsi32_si128, _mm_loadu_si128, _mm_storeu_si128, _mm256_abs_epi16,
        _mm256_add_epi16, _mm256_add_epi32, _mm256_and_si256, _mm256_blend_epi16,
        _mm256_blendv_epi8, _mm256_broadcastsi128_si256, _mm256_castsi256_si128,
        _mm256_cmpgt_epi32, _mm256_cvtepu16_epi32, _mm256_extracti128_si256, _mm256_hadd_epi16,
        _mm256_loadu_si256, _mm256_max_epi32, _mm256_min_epi32, _mm256_min_epu32,
        _mm256_mullo_epi32, _mm256_or_si256, _mm256_packus_epi32, _mm256_permute4x64_epi64,
        _mm256_permutevar8x32_epi32, _mm256_set1_epi32, _mm256_setr_epi8, _mm256_setr_epi32,
        _mm256_setzero_si256, _mm256_shuffle_epi8, _mm256_slli_epi32, _mm256_srl_epi32,
        _mm256_srli_epi32, _mm256_storeu_si256, _mm256_sub_epi16, _mm256_sub_epi32,
        _mm256_unpackhi_epi64, _mm256_unpacklo_epi64, _mm256_xor_si256,
    };
    use nc_common::BitDepth;

    use super::super::{
        ACTIVITY_CLASS, ACTIVITY_CLASSES, ACTIVITY_SCALE, ACTIVITY_SHIFT_ABOVE_DEPTH,
        BOUNDARY_ACTIVITY_SCALE, MAX_ACTIVITY, TRANSPOSE, classify_at, laplacians_at,
    };

    const ODD_LANES: i32 = 0b1010_1010;

    #[target_feature(enable = "avx2")]
    fn load16(row: &[u16], at: usize) -> __m256i {
        let samples = &row[at..at + 16];
        // SAFETY: AVX2 enabled; `samples` holds 16 `u16` values (32 bytes).
        unsafe { _mm256_loadu_si256(samples.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2")]
    fn pick16(rows: &[&[u16]; 4], row: usize, at: usize) -> __m256i {
        _mm256_blend_epi16::<ODD_LANES>(load16(rows[row], at), load16(rows[row + 1], at))
    }

    #[target_feature(enable = "avx2")]
    fn lap16(twice: __m256i, a: __m256i, b: __m256i) -> __m256i {
        _mm256_abs_epi16(_mm256_sub_epi16(twice, _mm256_add_epi16(a, b)))
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn laplacians(rows: &[&[u16]; 4], out: &mut [[u16; 4]]) {
        let to_groups = _mm256_setr_epi8(
            0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15, 0, 1, 4, 5, 8, 9, 12, 13, 2, 3,
            6, 7, 10, 11, 14, 15,
        );
        let (steps, _) = out.as_chunks_mut::<4>();
        let whole = steps.len() * 4;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let x = 16 * step + 1;
            let up0 = pick16(rows, 0, x - 1);
            let up1 = pick16(rows, 0, x);
            let up2 = pick16(rows, 0, x + 1);
            let c0 = pick16(rows, 1, x - 1);
            let c1 = pick16(rows, 1, x);
            let c2 = pick16(rows, 1, x + 1);
            let dn0 = pick16(rows, 2, x - 1);
            let dn1 = pick16(rows, 2, x);
            let dn2 = pick16(rows, 2, x + 1);
            let twice = _mm256_add_epi16(c1, c1);
            let sums = _mm256_hadd_epi16(
                _mm256_hadd_epi16(lap16(twice, up1, dn1), lap16(twice, c0, c2)),
                _mm256_hadd_epi16(lap16(twice, up0, dn2), lap16(twice, dn0, up2)),
            );
            let groups = _mm256_shuffle_epi8(sums, to_groups);
            // SAFETY: AVX2 enabled; `out_chunk` holds 4 `[u16; 4]` groups (32 bytes).
            unsafe { _mm256_storeu_si256(out_chunk.as_mut_ptr().cast(), groups) };
        }
        if whole < out.len() {
            let tail = [
                &rows[0][4 * whole..],
                &rows[1][4 * whole..],
                &rows[2][4 * whole..],
                &rows[3][4 * whole..],
            ];
            laplacians_at(&tail, &mut out[whole..]);
        }
    }

    #[target_feature(enable = "avx2")]
    fn load_groups(groups: &[[u16; 4]], at: usize) -> __m256i {
        let g = &groups[at..at + 4];
        // SAFETY: AVX2 enabled; `g` holds 4 groups (32 bytes).
        unsafe { _mm256_loadu_si256(g.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx2")]
    fn lookup(table: &[u8; 16], index: __m256i) -> __m256i {
        // SAFETY: AVX2 enabled; `table` holds 16 bytes.
        let t = unsafe { _mm_loadu_si128(table.as_ptr().cast()) };
        _mm256_shuffle_epi8(_mm256_broadcastsi128_si256(t), index)
    }

    #[target_feature(enable = "avx2")]
    fn quad_sums(
        groups: &[[u16; 4]],
        at: usize,
        pair_words: __m256i,
        quad_dwords: __m256i,
    ) -> __m256i {
        let sums = _mm256_add_epi16(load_groups(groups, at), load_groups(groups, at + 1));
        _mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(sums, pair_words), quad_dwords)
    }

    #[target_feature(enable = "avx2")]
    pub(crate) unsafe fn classify(
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
        depth: BitDepth,
    ) {
        let scale = _mm256_set1_epi32(if boundary {
            BOUNDARY_ACTIVITY_SCALE
        } else {
            ACTIVITY_SCALE
        } as i32);
        let shift = _mm_cvtsi32_si128((depth.bits() + ACTIVITY_SHIFT_ABOVE_DEPTH) as i32);
        let sign = _mm256_set1_epi32(i32::MIN);
        let pair_words = _mm256_setr_epi8(
            0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15, 0, 1, 8, 9, 2, 3, 10, 11, 4, 5,
            12, 13, 6, 7, 14, 15,
        );
        let quad_dwords = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let mut transpose = [0u8; 16];
        transpose[..8].copy_from_slice(&TRANSPOSE);
        let (steps, _) = out.as_chunks_mut::<8>();
        let whole = steps.len() * 8;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let first = 8 * step;
            let q0 = quad_sums(groups, first, pair_words, quad_dwords);
            let q1 = quad_sums(groups, first + 4, pair_words, quad_dwords);
            let vertical_d0 = _mm256_unpacklo_epi64(q0, q1);
            let horizontal_d1 = _mm256_unpackhi_epi64(q0, q1);
            let ver = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(vertical_d0));
            let hor = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(horizontal_d1));
            let dg0 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(vertical_d0));
            let dg1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256::<1>(horizontal_d1));
            let activity = _mm256_min_epu32(
                _mm256_srl_epi32(_mm256_mullo_epi32(_mm256_add_epi32(ver, hor), scale), shift),
                _mm256_set1_epi32(MAX_ACTIVITY as i32),
            );
            let vertical = _mm256_cmpgt_epi32(ver, hor);
            let (hv1, hv0) = (_mm256_max_epi32(ver, hor), _mm256_min_epi32(ver, hor));
            let hv_dir = _mm256_blendv_epi8(_mm256_set1_epi32(3), _mm256_set1_epi32(1), vertical);
            let first_diag = _mm256_cmpgt_epi32(dg0, dg1);
            let (d1, d0) = (_mm256_max_epi32(dg0, dg1), _mm256_min_epi32(dg0, dg1));
            let d_dir =
                _mm256_blendv_epi8(_mm256_set1_epi32(2), _mm256_setzero_si256(), first_diag);
            // Unsigned 32-bit comparison: d1 * hv0 and hv1 * d0 reach 65472^2 < 2^32.
            let diagonal = _mm256_cmpgt_epi32(
                _mm256_xor_si256(_mm256_mullo_epi32(d1, hv0), sign),
                _mm256_xor_si256(_mm256_mullo_epi32(hv1, d0), sign),
            );
            let strong = _mm256_blendv_epi8(hv1, d1, diagonal);
            let weak = _mm256_blendv_epi8(hv0, d0, diagonal);
            let main = _mm256_blendv_epi8(hv_dir, d_dir, diagonal);
            let secondary = _mm256_blendv_epi8(d_dir, hv_dir, diagonal);
            let one = _mm256_cmpgt_epi32(strong, _mm256_add_epi32(weak, weak));
            let two = _mm256_cmpgt_epi32(
                _mm256_add_epi32(strong, strong),
                _mm256_add_epi32(_mm256_slli_epi32::<3>(weak), weak),
            );
            let strength = _mm256_sub_epi32(_mm256_sub_epi32(_mm256_setzero_si256(), one), two);
            let step_classes = _mm256_add_epi32(
                _mm256_slli_epi32::<1>(_mm256_and_si256(main, _mm256_set1_epi32(1))),
                strength,
            );
            let directional = _mm256_and_si256(
                _mm256_mullo_epi32(step_classes, _mm256_set1_epi32(i32::from(ACTIVITY_CLASSES))),
                one,
            );
            let index = _mm256_add_epi32(lookup(&ACTIVITY_CLASS, activity), directional);
            let transposes = lookup(
                &transpose,
                _mm256_add_epi32(
                    _mm256_slli_epi32::<1>(main),
                    _mm256_srli_epi32::<1>(secondary),
                ),
            );
            let pairs = _mm256_or_si256(index, _mm256_slli_epi32::<8>(transposes));
            let packed = _mm256_permute4x64_epi64::<0b1000>(_mm256_packus_epi32(pairs, pairs));
            // SAFETY: AVX2 enabled; `out_chunk` holds 8 `[u8; 2]` pairs (16 bytes).
            unsafe {
                _mm_storeu_si128(
                    out_chunk.as_mut_ptr().cast(),
                    _mm256_castsi256_si128(packed),
                );
            }
        }
        if whole < out.len() {
            classify_at(&groups[whole..], boundary, &mut out[whole..], depth);
        }
    }
}

#[cfg(target_arch = "x86_64")]
pub(crate) mod avx512 {
    use core::arch::x86_64::{
        __m512i, _mm_cvtsi32_si128, _mm_loadu_si128, _mm256_storeu_si256, _mm512_abs_epi16,
        _mm512_add_epi16, _mm512_add_epi32, _mm512_and_si512, _mm512_broadcast_i32x4,
        _mm512_castsi512_si256, _mm512_cmpgt_epu32_mask, _mm512_cvtepi32_epi16,
        _mm512_cvtepu16_epi32, _mm512_extracti64x4_epi64, _mm512_loadu_si512, _mm512_madd_epi16,
        _mm512_mask_blend_epi16, _mm512_mask_blend_epi32, _mm512_maskz_mov_epi32,
        _mm512_maskz_set1_epi32, _mm512_max_epu32, _mm512_min_epu32, _mm512_mullo_epi32,
        _mm512_or_si512, _mm512_packs_epi32, _mm512_permutex2var_epi16, _mm512_set_epi16,
        _mm512_set1_epi16, _mm512_set1_epi32, _mm512_setzero_si512, _mm512_shuffle_epi8,
        _mm512_slli_epi32, _mm512_srl_epi32, _mm512_srli_epi32, _mm512_storeu_si512,
        _mm512_sub_epi16,
    };
    use nc_common::BitDepth;

    use super::super::{
        ACTIVITY_CLASS, ACTIVITY_CLASSES, ACTIVITY_SCALE, ACTIVITY_SHIFT_ABOVE_DEPTH,
        BOUNDARY_ACTIVITY_SCALE, MAX_ACTIVITY, TRANSPOSE,
    };

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    fn load32(row: &[u16], at: usize) -> __m512i {
        let samples = &row[at..at + 32];
        // SAFETY: AVX-512F/BW/VL enabled; `samples` holds 32 `u16` values (64 bytes).
        unsafe { _mm512_loadu_si512(samples.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    fn pick32(rows: &[&[u16]; 4], row: usize, at: usize) -> __m512i {
        _mm512_mask_blend_epi16(
            0xAAAA_AAAA,
            load32(rows[row], at),
            load32(rows[row + 1], at),
        )
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    fn lap32(twice: __m512i, a: __m512i, b: __m512i) -> __m512i {
        _mm512_abs_epi16(_mm512_sub_epi16(twice, _mm512_add_epi16(a, b)))
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub(crate) unsafe fn laplacians(rows: &[&[u16]; 4], out: &mut [[u16; 4]]) {
        let group_shuf: [u8; 16] = [0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15];
        // SAFETY: AVX-512F/BW/VL enabled; `group_shuf` holds 16 bytes.
        let to_groups =
            _mm512_broadcast_i32x4(unsafe { _mm_loadu_si128(group_shuf.as_ptr().cast()) });
        let ones = _mm512_set1_epi16(1);
        let (steps, _) = out.as_chunks_mut::<8>();
        let whole = steps.len() * 8;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let x = 32 * step + 1;
            let up0 = pick32(rows, 0, x - 1);
            let up1 = pick32(rows, 0, x);
            let up2 = pick32(rows, 0, x + 1);
            let c0 = pick32(rows, 1, x - 1);
            let c1 = pick32(rows, 1, x);
            let c2 = pick32(rows, 1, x + 1);
            let dn0 = pick32(rows, 2, x - 1);
            let dn1 = pick32(rows, 2, x);
            let dn2 = pick32(rows, 2, x + 1);
            let twice = _mm512_add_epi16(c1, c1);
            let vh_pairs = _mm512_packs_epi32(
                _mm512_madd_epi16(lap32(twice, up1, dn1), ones),
                _mm512_madd_epi16(lap32(twice, c0, c2), ones),
            );
            let dd_pairs = _mm512_packs_epi32(
                _mm512_madd_epi16(lap32(twice, up0, dn2), ones),
                _mm512_madd_epi16(lap32(twice, dn0, up2), ones),
            );
            let sums = _mm512_packs_epi32(
                _mm512_madd_epi16(vh_pairs, ones),
                _mm512_madd_epi16(dd_pairs, ones),
            );
            let groups = _mm512_shuffle_epi8(sums, to_groups);
            // SAFETY: AVX-512F/BW/VL enabled; `out_chunk` holds 8 `[u16; 4]` groups (64 bytes).
            unsafe { _mm512_storeu_si512(out_chunk.as_mut_ptr().cast(), groups) };
        }
        if whole < out.len() {
            let tail = [
                &rows[0][4 * whole..],
                &rows[1][4 * whole..],
                &rows[2][4 * whole..],
                &rows[3][4 * whole..],
            ];
            // SAFETY: AVX-512 implies AVX2; tail slices are in bounds.
            unsafe { super::avx2::laplacians(&tail, &mut out[whole..]) };
        }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    fn load_groups8(groups: &[[u16; 4]], at: usize) -> __m512i {
        let g = &groups[at..at + 8];
        // SAFETY: AVX-512F/BW/VL enabled; `g` holds 8 groups (64 bytes).
        unsafe { _mm512_loadu_si512(g.as_ptr().cast()) }
    }

    #[target_feature(enable = "avx512f,avx512bw,avx512vl")]
    pub(crate) unsafe fn classify(
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
        depth: BitDepth,
    ) {
        let scale = _mm512_set1_epi32(if boundary {
            BOUNDARY_ACTIVITY_SCALE
        } else {
            ACTIVITY_SCALE
        } as i32);
        let shift = _mm_cvtsi32_si128((depth.bits() + ACTIVITY_SHIFT_ABOVE_DEPTH) as i32);
        let mut transpose_bytes = [0u8; 16];
        transpose_bytes[..8].copy_from_slice(&TRANSPOSE);
        // SAFETY: AVX-512F/BW/VL enabled; each table holds 16 bytes.
        let (act_tbl, tr_tbl) = unsafe {
            (
                _mm512_broadcast_i32x4(_mm_loadu_si128(ACTIVITY_CLASS.as_ptr().cast())),
                _mm512_broadcast_i32x4(_mm_loadu_si128(transpose_bytes.as_ptr().cast())),
            )
        };
        let idx_vh = _mm512_set_epi16(
            61, 57, 53, 49, 45, 41, 37, 33, 29, 25, 21, 17, 13, 9, 5, 1, 60, 56, 52, 48, 44, 40,
            36, 32, 28, 24, 20, 16, 12, 8, 4, 0,
        );
        let idx_dd = _mm512_add_epi16(idx_vh, _mm512_set1_epi16(2));
        let (steps, _) = out.as_chunks_mut::<16>();
        let whole = steps.len() * 16;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let first = 16 * step;
            let s0 = _mm512_add_epi16(load_groups8(groups, first), load_groups8(groups, first + 1));
            let s1 = _mm512_add_epi16(
                load_groups8(groups, first + 8),
                load_groups8(groups, first + 9),
            );
            let vh = _mm512_permutex2var_epi16(s0, idx_vh, s1);
            let dd = _mm512_permutex2var_epi16(s0, idx_dd, s1);
            let ver = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(vh));
            let hor = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(vh));
            let dg0 = _mm512_cvtepu16_epi32(_mm512_castsi512_si256(dd));
            let dg1 = _mm512_cvtepu16_epi32(_mm512_extracti64x4_epi64::<1>(dd));
            let activity = _mm512_min_epu32(
                _mm512_srl_epi32(_mm512_mullo_epi32(_mm512_add_epi32(ver, hor), scale), shift),
                _mm512_set1_epi32(MAX_ACTIVITY as i32),
            );
            let vertical = _mm512_cmpgt_epu32_mask(ver, hor);
            let (hv1, hv0) = (_mm512_max_epu32(ver, hor), _mm512_min_epu32(ver, hor));
            let hv_dir =
                _mm512_mask_blend_epi32(vertical, _mm512_set1_epi32(3), _mm512_set1_epi32(1));
            let first_diag = _mm512_cmpgt_epu32_mask(dg0, dg1);
            let (d1, d0) = (_mm512_max_epu32(dg0, dg1), _mm512_min_epu32(dg0, dg1));
            let d_dir =
                _mm512_mask_blend_epi32(first_diag, _mm512_set1_epi32(2), _mm512_setzero_si512());
            let diagonal =
                _mm512_cmpgt_epu32_mask(_mm512_mullo_epi32(d1, hv0), _mm512_mullo_epi32(hv1, d0));
            let strong = _mm512_mask_blend_epi32(diagonal, hv1, d1);
            let weak = _mm512_mask_blend_epi32(diagonal, hv0, d0);
            let main = _mm512_mask_blend_epi32(diagonal, hv_dir, d_dir);
            let secondary = _mm512_mask_blend_epi32(diagonal, d_dir, hv_dir);
            let one = _mm512_cmpgt_epu32_mask(strong, _mm512_add_epi32(weak, weak));
            let two = _mm512_cmpgt_epu32_mask(
                _mm512_add_epi32(strong, strong),
                _mm512_add_epi32(_mm512_slli_epi32::<3>(weak), weak),
            );
            let strength = _mm512_add_epi32(
                _mm512_maskz_set1_epi32(one, 1),
                _mm512_maskz_set1_epi32(two, 1),
            );
            let step_classes = _mm512_add_epi32(
                _mm512_slli_epi32::<1>(_mm512_and_si512(main, _mm512_set1_epi32(1))),
                strength,
            );
            let directional = _mm512_maskz_mov_epi32(
                one,
                _mm512_mullo_epi32(step_classes, _mm512_set1_epi32(i32::from(ACTIVITY_CLASSES))),
            );
            let index = _mm512_add_epi32(_mm512_shuffle_epi8(act_tbl, activity), directional);
            let transposes = _mm512_shuffle_epi8(
                tr_tbl,
                _mm512_add_epi32(
                    _mm512_slli_epi32::<1>(main),
                    _mm512_srli_epi32::<1>(secondary),
                ),
            );
            let pairs = _mm512_or_si512(index, _mm512_slli_epi32::<8>(transposes));
            let packed = _mm512_cvtepi32_epi16(pairs);
            // SAFETY: AVX-512F/BW/VL enabled; `out_chunk` holds 16 `[u8; 2]` pairs (32 bytes).
            unsafe { _mm256_storeu_si256(out_chunk.as_mut_ptr().cast(), packed) };
        }
        if whole < out.len() {
            // SAFETY: AVX-512 implies AVX2; tail slices are in bounds.
            unsafe {
                super::avx2::classify(&groups[whole..], boundary, &mut out[whole..], depth);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
pub(crate) mod neon {
    use core::arch::aarch64::{
        int32x4_t, uint8x16_t, uint16x4_t, uint16x4x4_t, uint16x8_t, uint16x8x4_t, uint32x4_t,
        vabdq_u16, vaddq_u16, vaddq_u32, vandq_u32, vbslq_u16, vbslq_u32, vcgtq_u32, vcombine_u16,
        vdupq_n_s32, vdupq_n_u32, vget_high_u16, vget_low_u16, vld1q_u8, vld1q_u16, vld4q_u16,
        vmaxq_u32, vminq_u32, vmovl_high_u16, vmovl_u16, vmovn_u32, vmulq_n_u32, vmulq_u32,
        vorrq_u32, vpaddq_u16, vqtbl1q_u8, vreinterpretq_u8_u32, vreinterpretq_u32_u8, vshlq_n_u32,
        vshlq_u32, vshrq_n_u32, vst1q_u16, vst4_u16, vsubq_u32,
    };
    use nc_common::BitDepth;

    use super::super::{
        ACTIVITY_CLASS, ACTIVITY_CLASSES, ACTIVITY_SCALE, ACTIVITY_SHIFT_ABOVE_DEPTH,
        BOUNDARY_ACTIVITY_SCALE, MAX_ACTIVITY, TRANSPOSE, classify_at, laplacians_at,
    };

    #[target_feature(enable = "neon")]
    fn load8(row: &[u16], at: usize) -> uint16x8_t {
        let s = &row[at..at + 8];
        // SAFETY: NEON enabled; `s` holds 8 `u16` values.
        unsafe { vld1q_u16(s.as_ptr()) }
    }

    #[target_feature(enable = "neon")]
    fn pick16(rows: &[&[u16]; 4], row: usize, at: usize, odd: uint16x8_t) -> [uint16x8_t; 2] {
        [
            vbslq_u16(odd, load8(rows[row + 1], at), load8(rows[row], at)),
            vbslq_u16(odd, load8(rows[row + 1], at + 8), load8(rows[row], at + 8)),
        ]
    }

    #[target_feature(enable = "neon")]
    fn lap_pair(c: [uint16x8_t; 2], a: [uint16x8_t; 2], b: [uint16x8_t; 2]) -> uint16x8_t {
        let lo = vabdq_u16(vaddq_u16(c[0], c[0]), vaddq_u16(a[0], b[0]));
        let hi = vabdq_u16(vaddq_u16(c[1], c[1]), vaddq_u16(a[1], b[1]));
        vpaddq_u16(lo, hi)
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn laplacians(rows: &[&[u16]; 4], out: &mut [[u16; 4]]) {
        let odd_lanes: [u16; 8] = [0, 0xffff, 0, 0xffff, 0, 0xffff, 0, 0xffff];
        // SAFETY: NEON enabled; `odd_lanes` holds 8 `u16` values.
        let odd = unsafe { vld1q_u16(odd_lanes.as_ptr()) };
        let (steps, _) = out.as_chunks_mut::<4>();
        let whole = steps.len() * 4;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let x = 16 * step + 1;
            let up0 = pick16(rows, 0, x - 1, odd);
            let up1 = pick16(rows, 0, x, odd);
            let up2 = pick16(rows, 0, x + 1, odd);
            let c0 = pick16(rows, 1, x - 1, odd);
            let c1 = pick16(rows, 1, x, odd);
            let c2 = pick16(rows, 1, x + 1, odd);
            let dn0 = pick16(rows, 2, x - 1, odd);
            let dn1 = pick16(rows, 2, x, odd);
            let dn2 = pick16(rows, 2, x + 1, odd);
            let straight = vpaddq_u16(lap_pair(c1, up1, dn1), lap_pair(c1, c0, c2));
            let diagonal = vpaddq_u16(lap_pair(c1, up0, dn2), lap_pair(c1, dn0, up2));
            let groups = uint16x4x4_t(
                vget_low_u16(straight),
                vget_high_u16(straight),
                vget_low_u16(diagonal),
                vget_high_u16(diagonal),
            );
            // SAFETY: NEON enabled; `out_chunk` holds 16 `u16` values.
            unsafe { vst4_u16(out_chunk.as_mut_ptr().cast(), groups) };
        }
        if whole < out.len() {
            let tail = [
                &rows[0][4 * whole..],
                &rows[1][4 * whole..],
                &rows[2][4 * whole..],
                &rows[3][4 * whole..],
            ];
            laplacians_at(&tail, &mut out[whole..]);
        }
    }

    #[target_feature(enable = "neon")]
    fn lookup(table: uint8x16_t, index: uint32x4_t) -> uint32x4_t {
        vreinterpretq_u32_u8(vqtbl1q_u8(table, vreinterpretq_u8_u32(index)))
    }

    #[target_feature(enable = "neon")]
    fn classes4(
        [ver, hor, dg0, dg1]: [uint32x4_t; 4],
        scale: u32,
        neg_shift: int32x4_t,
        tables: [uint8x16_t; 2],
    ) -> uint16x4_t {
        let activity = vminq_u32(
            vshlq_u32(vmulq_n_u32(vaddq_u32(ver, hor), scale), neg_shift),
            vdupq_n_u32(MAX_ACTIVITY),
        );
        let vertical = vcgtq_u32(ver, hor);
        let (hv1, hv0) = (vmaxq_u32(ver, hor), vminq_u32(ver, hor));
        let hv_dir = vbslq_u32(vertical, vdupq_n_u32(1), vdupq_n_u32(3));
        let first_diag = vcgtq_u32(dg0, dg1);
        let (d1, d0) = (vmaxq_u32(dg0, dg1), vminq_u32(dg0, dg1));
        let d_dir = vbslq_u32(first_diag, vdupq_n_u32(0), vdupq_n_u32(2));
        let diagonal = vcgtq_u32(vmulq_u32(d1, hv0), vmulq_u32(hv1, d0));
        let strong = vbslq_u32(diagonal, d1, hv1);
        let weak = vbslq_u32(diagonal, d0, hv0);
        let main = vbslq_u32(diagonal, d_dir, hv_dir);
        let secondary = vbslq_u32(diagonal, hv_dir, d_dir);
        let one = vcgtq_u32(strong, vaddq_u32(weak, weak));
        let two = vcgtq_u32(vaddq_u32(strong, strong), vmulq_n_u32(weak, 9));
        let strength = vsubq_u32(vsubq_u32(vdupq_n_u32(0), one), two);
        let step_classes = vaddq_u32(vshlq_n_u32::<1>(vandq_u32(main, vdupq_n_u32(1))), strength);
        let directional = vandq_u32(vmulq_n_u32(step_classes, u32::from(ACTIVITY_CLASSES)), one);
        let index = vaddq_u32(lookup(tables[0], activity), directional);
        let transposes = lookup(
            tables[1],
            vaddq_u32(vshlq_n_u32::<1>(main), vshrq_n_u32::<1>(secondary)),
        );
        vmovn_u32(vorrq_u32(index, vshlq_n_u32::<8>(transposes)))
    }

    #[target_feature(enable = "neon")]
    fn load_groups8(groups: &[[u16; 4]], at: usize) -> uint16x8x4_t {
        let g = &groups[at..at + 8];
        // SAFETY: NEON enabled; `g` holds 8 `[u16; 4]` groups.
        unsafe { vld4q_u16(g.as_ptr().cast()) }
    }

    #[target_feature(enable = "neon")]
    pub(crate) unsafe fn classify(
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
        depth: BitDepth,
    ) {
        let scale = if boundary {
            BOUNDARY_ACTIVITY_SCALE
        } else {
            ACTIVITY_SCALE
        };
        let neg_shift = vdupq_n_s32(-((depth.bits() + ACTIVITY_SHIFT_ABOVE_DEPTH) as i32));
        let mut transpose = [0u8; 16];
        transpose[..8].copy_from_slice(&TRANSPOSE);
        // SAFETY: NEON enabled; each table holds 16 bytes.
        let tables = unsafe {
            [
                vld1q_u8(ACTIVITY_CLASS.as_ptr()),
                vld1q_u8(transpose.as_ptr()),
            ]
        };
        let (steps, _) = out.as_chunks_mut::<8>();
        let whole = steps.len() * 8;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let (left, right) = (
                load_groups8(groups, 8 * step),
                load_groups8(groups, 8 * step + 1),
            );
            let s0 = vaddq_u16(left.0, right.0);
            let s1 = vaddq_u16(left.1, right.1);
            let s2 = vaddq_u16(left.2, right.2);
            let s3 = vaddq_u16(left.3, right.3);
            let low = classes4(
                [
                    vmovl_u16(vget_low_u16(s0)),
                    vmovl_u16(vget_low_u16(s1)),
                    vmovl_u16(vget_low_u16(s2)),
                    vmovl_u16(vget_low_u16(s3)),
                ],
                scale,
                neg_shift,
                tables,
            );
            let high = classes4(
                [
                    vmovl_high_u16(s0),
                    vmovl_high_u16(s1),
                    vmovl_high_u16(s2),
                    vmovl_high_u16(s3),
                ],
                scale,
                neg_shift,
                tables,
            );
            // SAFETY: NEON enabled; `out_chunk` holds 8 `[u8; 2]` pairs (16 bytes).
            unsafe { vst1q_u16(out_chunk.as_mut_ptr().cast(), vcombine_u16(low, high)) };
        }
        if whole < out.len() {
            classify_at(&groups[whole..], boundary, &mut out[whole..], depth);
        }
    }
}

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub(crate) mod simd128 {
    use core::arch::wasm32::{
        i8x16_swizzle, i16x8_abs, i16x8_add, i16x8_shuffle, i16x8_sub, i32x4_add, i32x4_mul,
        i32x4_shl, i32x4_sub, u16x8_narrow_i32x4, u32x4_extend_high_u16x8, u32x4_extend_low_u16x8,
        u32x4_gt, u32x4_max, u32x4_min, u32x4_shr, u32x4_splat, v128, v128_and, v128_bitselect,
        v128_load, v128_or, v128_store, v128_store64_lane,
    };
    use nc_common::BitDepth;

    use super::super::{
        ACTIVITY_CLASS, ACTIVITY_CLASSES, ACTIVITY_SCALE, ACTIVITY_SHIFT_ABOVE_DEPTH,
        BOUNDARY_ACTIVITY_SCALE, MAX_ACTIVITY, TRANSPOSE, classify_at, laplacians_at,
    };

    #[target_feature(enable = "simd128")]
    fn load8(row: &[u16], at: usize) -> v128 {
        let s = &row[at..at + 8];
        // SAFETY: SIMD128 enabled; `s` holds 8 `u16` values (16 bytes).
        unsafe { v128_load(s.as_ptr().cast()) }
    }

    #[target_feature(enable = "simd128")]
    fn pick8(rows: &[&[u16]; 4], row: usize, at: usize) -> v128 {
        i16x8_shuffle::<0, 9, 2, 11, 4, 13, 6, 15>(load8(rows[row], at), load8(rows[row + 1], at))
    }

    #[target_feature(enable = "simd128")]
    fn pair_sums(a: v128, b: v128) -> v128 {
        i16x8_add(
            i16x8_shuffle::<0, 2, 4, 6, 8, 10, 12, 14>(a, b),
            i16x8_shuffle::<1, 3, 5, 7, 9, 11, 13, 15>(a, b),
        )
    }

    #[target_feature(enable = "simd128")]
    fn lap8(twice: v128, a: v128, b: v128) -> v128 {
        i16x8_abs(i16x8_sub(twice, i16x8_add(a, b)))
    }

    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn laplacians(rows: &[&[u16]; 4], out: &mut [[u16; 4]]) {
        let (steps, _) = out.as_chunks_mut::<2>();
        let whole = steps.len() * 2;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let x = 8 * step + 1;
            let up0 = pick8(rows, 0, x - 1);
            let up1 = pick8(rows, 0, x);
            let up2 = pick8(rows, 0, x + 1);
            let c0 = pick8(rows, 1, x - 1);
            let c1 = pick8(rows, 1, x);
            let c2 = pick8(rows, 1, x + 1);
            let dn0 = pick8(rows, 2, x - 1);
            let dn1 = pick8(rows, 2, x);
            let dn2 = pick8(rows, 2, x + 1);
            let twice = i16x8_add(c1, c1);
            let straight = pair_sums(lap8(twice, up1, dn1), lap8(twice, c0, c2));
            let diagonal = pair_sums(lap8(twice, up0, dn2), lap8(twice, dn0, up2));
            let sums = pair_sums(straight, diagonal);
            let groups = i16x8_shuffle::<0, 2, 4, 6, 1, 3, 5, 7>(sums, sums);
            // SAFETY: SIMD128 enabled; `out_chunk` holds 2 `[u16; 4]` groups (16 bytes).
            unsafe { v128_store(out_chunk.as_mut_ptr().cast(), groups) };
        }
        if whole < out.len() {
            let tail = [
                &rows[0][4 * whole..],
                &rows[1][4 * whole..],
                &rows[2][4 * whole..],
                &rows[3][4 * whole..],
            ];
            laplacians_at(&tail, &mut out[whole..]);
        }
    }

    #[target_feature(enable = "simd128")]
    fn load_groups2(groups: &[[u16; 4]], at: usize) -> v128 {
        let g = &groups[at..at + 2];
        // SAFETY: SIMD128 enabled; `g` holds 2 `[u16; 4]` groups (16 bytes).
        unsafe { v128_load(g.as_ptr().cast()) }
    }

    #[target_feature(enable = "simd128")]
    pub(crate) unsafe fn classify(
        groups: &[[u16; 4]],
        boundary: bool,
        out: &mut [[u8; 2]],
        depth: BitDepth,
    ) {
        let scale = u32x4_splat(if boundary {
            BOUNDARY_ACTIVITY_SCALE
        } else {
            ACTIVITY_SCALE
        });
        let shift = depth.bits() + ACTIVITY_SHIFT_ABOVE_DEPTH;
        let mut transpose_bytes = [0u8; 16];
        transpose_bytes[..8].copy_from_slice(&TRANSPOSE);
        // SAFETY: SIMD128 enabled; each table holds 16 bytes.
        let (activity_class, transpose) = unsafe {
            (
                v128_load(ACTIVITY_CLASS.as_ptr().cast()),
                v128_load(transpose_bytes.as_ptr().cast()),
            )
        };
        let (steps, _) = out.as_chunks_mut::<4>();
        let whole = steps.len() * 4;
        for (step, out_chunk) in steps.iter_mut().enumerate() {
            let first = 4 * step;
            let p0 = i16x8_add(load_groups2(groups, first), load_groups2(groups, first + 1));
            let p1 = i16x8_add(
                load_groups2(groups, first + 2),
                load_groups2(groups, first + 3),
            );
            let straight = i16x8_shuffle::<0, 4, 8, 12, 1, 5, 9, 13>(p0, p1);
            let diagonals = i16x8_shuffle::<2, 6, 10, 14, 3, 7, 11, 15>(p0, p1);
            let (ver, hor) = (
                u32x4_extend_low_u16x8(straight),
                u32x4_extend_high_u16x8(straight),
            );
            let (dg0, dg1) = (
                u32x4_extend_low_u16x8(diagonals),
                u32x4_extend_high_u16x8(diagonals),
            );
            let activity = u32x4_min(
                u32x4_shr(i32x4_mul(i32x4_add(ver, hor), scale), shift),
                u32x4_splat(MAX_ACTIVITY),
            );
            let vertical = u32x4_gt(ver, hor);
            let (hv1, hv0) = (u32x4_max(ver, hor), u32x4_min(ver, hor));
            let hv_dir = v128_bitselect(u32x4_splat(1), u32x4_splat(3), vertical);
            let first_diag = u32x4_gt(dg0, dg1);
            let (d1, d0) = (u32x4_max(dg0, dg1), u32x4_min(dg0, dg1));
            let d_dir = v128_bitselect(u32x4_splat(0), u32x4_splat(2), first_diag);
            let diagonal = u32x4_gt(i32x4_mul(d1, hv0), i32x4_mul(hv1, d0));
            let strong = v128_bitselect(d1, hv1, diagonal);
            let weak = v128_bitselect(d0, hv0, diagonal);
            let main = v128_bitselect(d_dir, hv_dir, diagonal);
            let secondary = v128_bitselect(hv_dir, d_dir, diagonal);
            let one = u32x4_gt(strong, i32x4_add(weak, weak));
            let two = u32x4_gt(i32x4_add(strong, strong), i32x4_mul(weak, u32x4_splat(9)));
            let strength = i32x4_sub(i32x4_sub(u32x4_splat(0), one), two);
            let step_classes = i32x4_add(i32x4_shl(v128_and(main, u32x4_splat(1)), 1), strength);
            let directional = v128_and(
                i32x4_mul(step_classes, u32x4_splat(u32::from(ACTIVITY_CLASSES))),
                one,
            );
            let index = i32x4_add(i8x16_swizzle(activity_class, activity), directional);
            let transposes = i8x16_swizzle(
                transpose,
                i32x4_add(i32x4_shl(main, 1), u32x4_shr(secondary, 1)),
            );
            let pairs = v128_or(index, i32x4_shl(transposes, 8));
            let packed = u16x8_narrow_i32x4(pairs, pairs);
            // SAFETY: SIMD128 enabled; `out_chunk` holds 4 `[u8; 2]` pairs (8 bytes).
            unsafe { v128_store64_lane::<0>(packed, out_chunk.as_mut_ptr().cast()) };
        }
        if whole < out.len() {
            classify_at(&groups[whole..], boundary, &mut out[whole..], depth);
        }
    }
}
