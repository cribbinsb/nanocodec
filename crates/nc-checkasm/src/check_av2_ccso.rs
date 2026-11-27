//! AV2 CCSO packed offsets and edge classes on complete and partial spans.

use nc_dsp::Dsp;
use nc_dsp::av2_dec::ccso::{CcsoFilter, CcsoMode};

use crate::{CheckReport, rng::Rng};

pub(crate) fn check(dsp: &Dsp, reference: &Dsp, rng: &mut Rng, report: &mut CheckReport) {
    let rows: [[u8; 512]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| rng.byte()));
    let spans = rows.each_ref().map(<[u8; 512]>::as_slice);
    let samples: [u8; 258] = std::array::from_fn(|_| rng.byte());
    let codes = std::array::from_fn(|_| rng.byte());
    let offsets = std::array::from_fn(|i| match i {
        0 => i8::MIN,
        1 => i8::MAX,
        _ => rng.byte() as i8,
    });
    let modes = [
        (CcsoMode::Bands, 2),
        (CcsoMode::Two(1), 5),
        (CcsoMode::Two(255), 8),
        (CcsoMode::Three(0), 5),
        (CcsoMode::Three(64), 5),
        (CcsoMode::Three(128), 6),
        (CcsoMode::Three(255), 8),
    ];
    for (mode, shift) in modes {
        let filter = CcsoFilter::new(mode, shift, codes, offsets);
        for chroma in [false, true] {
            for width in [0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 255, 256] {
                let mut expected = samples;
                let mut actual = samples;
                filter.apply(reference, &mut expected[1..][..width], spans, chroma);
                filter.apply(dsp, &mut actual[1..][..width], spans, chroma);
                report.cases += 1;
                if actual != expected {
                    report.failures.push(format!(
                        "AV2 CCSO mismatch with {mode:?}, shift {shift}, chroma {chroma}, width {width}"
                    ));
                }
            }
        }
    }
}
