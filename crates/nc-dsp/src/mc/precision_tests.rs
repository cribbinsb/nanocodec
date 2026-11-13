//! Precision retention and final-rounding checks for shared HEVC/VVC prediction.
use crate::{BlockSize, Dsp, Filter};
use nc_common::{PlaneView, PlaneViewMut};
#[test]
fn retained_precision_rounds_to_existing_dispatched_prediction() {
    let dsp = Dsp::new();
    for filter in [
        Filter::HevcLuma,
        Filter::HevcChroma,
        Filter::VvcLuma,
        Filter::VvcAffineLuma,
        Filter::VvcChroma,
    ] {
        for (w, h) in [(4, 4), (8, 4), (4, 16)] {
            let stride = w + filter.taps() - 1;
            let rows = h + filter.taps() - 1;
            let source: Vec<_> = (0..stride * rows)
                .map(|i| (i * 71 + i * i * 3) as u8)
                .collect();
            let window = PlaneView::new(&source, stride, stride, rows).unwrap();
            for fy in 0..filter.positions() {
                for fx in 0..filter.positions() {
                    let mut wide = vec![0; w * h];
                    dsp.predict_high_precision(filter, (w, h), &window, (fx, fy), false, &mut wide);
                    let mut bytes = vec![0; w * h];
                    let mut out = PlaneViewMut::new(&mut bytes, w, w, h).unwrap();
                    dsp.predict(
                        filter,
                        BlockSize::new(w, h).unwrap(),
                        &window,
                        fx,
                        fy,
                        &mut out,
                    );
                    let rounded: Vec<_> = wide
                        .into_iter()
                        .map(|v| ((i32::from(v) + 32) >> 6).clamp(0, 255) as u8)
                        .collect();
                    assert_eq!(rounded, bytes, "{filter:?} {w}x{h} phase {fx},{fy}");
                }
            }
        }
    }
}
#[test]
fn intermediate_samples_preserve_overshoot_and_alternate_half_filter() {
    let dsp = Dsp::new();
    let mut source = vec![0u8; 11 * 11];
    source[3 * 11 + 1] = 255;
    source[3 * 11 + 3] = 255;
    source[3 * 11 + 4] = 255;
    source[3 * 11 + 6] = 255;
    let window = PlaneView::new(&source, 11, 11, 11).unwrap();
    let mut result = [0; 16];
    dsp.predict_high_precision(Filter::VvcLuma, (4, 4), &window, (8, 0), false, &mut result);
    assert_eq!(result[0], 88 * 255);
    dsp.predict_high_precision(Filter::VvcLuma, (4, 4), &window, (8, 0), true, &mut result);
    assert_eq!(result[0], 46 * 255);
}
