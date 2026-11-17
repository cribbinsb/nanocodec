//! Fixed-work encoder-facing copy counts. Arguments: ISA, side,
//! put/prep/recon, calls. No input or decoded video files are written.

#![forbid(unsafe_code)]

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{BlockSize, Dsp, Filter, Isa};
use std::hint::black_box;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let isa = match args[0].as_str() {
        "scalar" => Isa::Scalar,
        "avx2" => Isa::Avx2,
        "avx512" => Isa::Avx512,
        "neon" => Isa::Neon,
        "simd128" => Isa::Simd128,
        _ => panic!("ISA must be scalar/avx2/avx512/neon/simd128"),
    };
    let side: usize = args[1].parse().expect("block side");
    let kind = args[2].as_str();
    let calls: usize = args[3].parse().expect("calls");
    let size = BlockSize::new(side, side).expect("block size");
    let dsp = Dsp::with_isa(isa).expect("ISA available");
    let stride = side + 7;
    let input: Vec<u8> = (0..stride * stride).map(|i| (i * 37 + 11) as u8).collect();
    let mut output = vec![0; side * side];
    let mut words = vec![0_i16; side * side];
    let source = PlaneView::new(&input[3 * stride + 3..], stride, side, side).unwrap();
    for _ in 0..calls {
        let dsp = black_box(&dsp);
        match kind {
            "put" => dsp.predict_samples(
                Filter::Av1,
                size,
                (&input, stride),
                (0, 0),
                (&mut output, side),
            ),
            "prep" => dsp.predict_compound_samples(
                Filter::Av1,
                size,
                (&input, stride),
                (0, 0),
                (&mut words, side),
            ),
            "recon" => {
                let mut dst = PlaneViewMut::new(&mut output, side, side, side).unwrap();
                dsp.copy_plane(&source, &mut dst);
            }
            _ => panic!("operation must be put/prep/recon"),
        }
        black_box(&output);
        black_box(&words);
    }
    println!("{} {}", output[0], words[0]);
}
