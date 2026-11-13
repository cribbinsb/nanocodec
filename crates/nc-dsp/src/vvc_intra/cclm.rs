//! H.266 §8.4.5.2 cross-component linear prediction arithmetic.
//! BSD-Clear `VVdeC` equations; reference availability/selection stays in the codec.

/// Downsample six substituted luma samples `[left, centre, right, lower_left,
/// lower_centre, lower_right]` for non-collocated 4:2:0 CCLM. Repeat the first
/// row as the second for the single-row template at a CTU boundary.
#[must_use]
pub fn vvc_cclm_downsample(samples: [u8; 6]) -> u8 {
    let [left, centre, right, lower_left, lower_centre, lower_right] = samples.map(u32::from);
    ((left + 2 * centre + right + lower_left + 2 * lower_centre + lower_right + 4) >> 3) as u8
}

/// Downsample `[above, left, centre, right, below]` for collocated 4:2:0 CCLM.
#[must_use]
pub fn vvc_cclm_downsample_collocated(samples: [u8; 5]) -> u8 {
    let [above, left, centre, right, below] = samples.map(u32::from);
    ((above + left + 4 * centre + right + below + 4) >> 3) as u8
}

/// Predict chroma from downsampled luma and two/four selected `[luma, chroma]`
/// reference pairs. An empty reference list predicts the 8-bit midpoint.
/// # Panics
/// If the reference count is not zero, two or four, or output is short.
pub fn vvc_cclm_predict(luma: &[u8], pairs: &[[u8; 2]], out: &mut [u8]) {
    assert!(matches!(pairs.len(), 0 | 2 | 4) && out.len() >= luma.len());
    let (a, b, shift) = model(pairs);
    for (&value, dst) in luma.iter().zip(out) {
        *dst = (((a * i32::from(value)) >> shift) + b).clamp(0, 255) as u8;
    }
}

fn model(pairs: &[[u8; 2]]) -> (i32, i32, u32) {
    const DIVISORS: [i32; 16] = [0, 7, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1, 0];

    if pairs.is_empty() {
        return (0, 128, 0);
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
        (i32::from(samples[indices[0]][component]) + i32::from(samples[indices[1]][component]) + 1)
            >> 1
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
        let luma = [0, 32, 64, 128, 192, 255];
        let mut out = [0; 6];
        vvc_cclm_predict(&luma, &[], &mut out);
        assert_eq!(out, [128; 6]);
        vvc_cclm_predict(&luma, &[[32, 32], [192, 192]], &mut out);
        assert_eq!(out, luma);
        vvc_cclm_predict(&luma, &[[32, 223], [192, 63]], &mut out);
        assert_eq!(out, luma.map(|v| 255 - v));
        vvc_cclm_predict(&luma, &[[50, 30], [50, 50]], &mut out);
        // Equal luma preserves the specified grouping: duplicated second pair.
        assert_eq!(out, [50; 6]);
    }
}
