//! Short runs of one value in per-unit contexts and maps (a block's runs are
//! 1 to 16 entries, mostly powers of two): set by fixed-size stores, where a
//! runtime-length `fill` was a `memset` call per run (on the Jetson the
//! decoder's short fills were ~0.6 % of a 720p decode in libc).

/// Set every entry of `run` to `value`.
#[inline]
pub(super) fn fill_run<T: Copy>(run: &mut [T], value: T) {
    match run.len() {
        1 => run[0] = value,
        2 => set::<T, 2>(run, value),
        4 => set::<T, 4>(run, value),
        8 => set::<T, 8>(run, value),
        16 => set::<T, 16>(run, value),
        _ => run.fill(value),
    }
}

#[inline]
fn set<T: Copy, const N: usize>(run: &mut [T], value: T) {
    let run: &mut [T; N] = run.try_into().expect("a run of N entries");
    *run = [value; N];
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_length_is_filled() {
        for len in 0..=20 {
            let mut run = vec![0u8; len];
            fill_run(&mut run, 7);
            assert!(run.iter().all(|&v| v == 7), "length {len}");
        }
    }
}
