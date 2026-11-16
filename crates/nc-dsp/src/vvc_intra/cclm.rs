//! H.266 §8.4.5.2.14 cross-component linear prediction arithmetic, at any
//! bit depth (the model's arithmetic does not depend on it; only the
//! missing-reference value and the clip do).
//! BSD-Clear `VVdeC` equations; reference availability/selection stays in the codec.

use nc_common::{BitDepth, Sample};

/// Downsample six substituted luma samples `[left, centre, right, lower_left,
/// lower_centre, lower_right]` for non-collocated 4:2:0 CCLM. Repeat the first
/// row as the second for the single-row template at a CTU boundary.
#[must_use]
pub fn vvc_cclm_downsample<T: Sample>(samples: [T; 6]) -> T {
    let [left, centre, right, lower_left, lower_centre, lower_right] = samples.map(T::to_i32);
    T::from_i32((left + 2 * centre + right + lower_left + 2 * lower_centre + lower_right + 4) >> 3)
}

/// Downsample `[above, left, centre, right, below]` for collocated 4:2:0 CCLM.
#[must_use]
pub fn vvc_cclm_downsample_collocated<T: Sample>(samples: [T; 5]) -> T {
    let [above, left, centre, right, below] = samples.map(T::to_i32);
    T::from_i32((above + left + 4 * centre + right + below + 4) >> 3)
}

/// Predict chroma from downsampled luma and two/four selected `[luma, chroma]`
/// reference pairs at `depth`, clipped to it (`Clip1`). An empty reference
/// list predicts the midpoint `1 << (BitDepth − 1)`.
/// # Panics
/// If the reference count is not zero, two or four, or output is short.
pub fn vvc_cclm_predict<T: Sample>(luma: &[T], pairs: &[[T; 2]], out: &mut [T], depth: BitDepth) {
    assert!(matches!(pairs.len(), 0 | 2 | 4) && out.len() >= luma.len());
    let (a, b, shift) = model(pairs, depth);
    for (&value, dst) in luma.iter().zip(out) {
        *dst = depth.clip_to(((a * value.to_i32()) >> shift) + b);
    }
}

fn model<T: Sample>(pairs: &[[T; 2]], depth: BitDepth) -> (i32, i32, u32) {
    const DIVISORS: [i32; 16] = [0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0];

    if pairs.is_empty() {
        return (0, depth.mid(), 0);
    }
    let samples = if pairs.len() == 2 {
        [pairs[1], pairs[0], pairs[1], pairs[0]]
    } else {
        [pairs[0], pairs[1], pairs[2], pairs[3]]
    };
    // The specified sorting network matters when luma values tie: their
    // paired chroma values must retain the normative grouping.
    let (mut low, mut high) = ([0, 2], [1, 3]);
    if samples[low[0]][0] > samples[low[1]][0] {
        low.swap(0, 1);
    }
    if samples[high[0]][0] > samples[high[1]][0] {
        high.swap(0, 1);
    }
    if samples[low[0]][0] > samples[high[1]][0] {
        std::mem::swap(&mut low, &mut high);
    }
    if samples[low[1]][0] > samples[high[0]][0] {
        std::mem::swap(&mut low[1], &mut high[0]);
    }
    let mean = |indices: [usize; 2], component: usize| {
        (samples[indices[0]][component].to_i32() + samples[indices[1]][component].to_i32() + 1) >> 1
    };
    let (low_y, low_c) = (mean(low, 0), mean(low, 1));
    let difference = mean(high, 0) - low_y;
    if difference == 0 {
        return (0, low_c, 0);
    }
    let chroma_difference = mean(high, 1) - low_c;
    let mut x = difference.ilog2() as i32;
    let norm = ((difference << 4 >> x) & 15) as usize;
    let reciprocal = DIVISORS[norm] | 8;
    x += i32::from(norm != 0);
    let y = if chroma_difference == 0 {
        0
    } else {
        chroma_difference.abs().ilog2() as i32 + 1
    };
    let mut a = (chroma_difference * reciprocal + ((1 << y) >> 1)) >> y;
    let mut shift = 3 + x - y;
    if shift < 1 {
        shift = 1;
        a = a.signum() * 15;
    }
    (a, low_c - ((a * low_y) >> shift), shift as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn linear_models_and_missing_references() {
        let eight = BitDepth::EIGHT;
        let luma: [u8; 6] = [0, 32, 64, 128, 192, 255];
        let mut out = [0; 6];
        vvc_cclm_predict(&luma, &[], &mut out, eight);
        assert_eq!(out, [128; 6]);
        vvc_cclm_predict(&luma, &[[32, 32], [192, 192]], &mut out, eight);
        assert_eq!(out, luma);
        vvc_cclm_predict(&luma, &[[32, 223], [192, 63]], &mut out, eight);
        assert_eq!(out, luma.map(|v| 255 - v));
        vvc_cclm_predict(&luma, &[[50, 30], [50, 50]], &mut out, eight);
        // Equal luma preserves the specified grouping: duplicated second pair.
        assert_eq!(out, [50; 6]);
    }

    #[test]
    fn ten_bit_midpoint_and_clip() {
        let ten = BitDepth::TEN;
        let luma: [u16; 4] = [0, 100, 900, 1023];
        let mut out = [0; 4];
        vvc_cclm_predict(&luma, &[], &mut out, ten);
        assert_eq!(out, [512; 4]);
        vvc_cclm_predict(&luma, &[[100, 600], [900, 1023]], &mut out, ten);
        assert_eq!(out[3], 1023, "the prediction clips to 10 bits, not 8");
        assert!((599..=601).contains(&out[1]), "{out:?}");
    }
}
