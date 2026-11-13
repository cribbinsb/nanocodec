//! Fixed-work CDEF driver for load-independent callgrind comparisons with SVT.
//! Arguments: ISA (scalar/avx2/neon/simd128), block side, strength case, calls.

#![forbid(unsafe_code)]

use std::hint::black_box;

use nc_common::{PlaneView, PlaneViewMut};
use nc_dsp::{Dsp, Isa};

const SIDE: usize = 32;
const ORIGIN: usize = 8;
const STRENGTHS: [(u8, u8); 3] = [(8, 2), (15, 0), (0, 4)];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let isa = match args[0].as_str() {
        "scalar" => Isa::Scalar,
        "avx2" => Isa::Avx2,
        "neon" => Isa::Neon,
        "simd128" => Isa::Simd128,
        _ => panic!("ISA must be scalar/avx2/neon/simd128"),
    };
    let side: usize = args[1].parse().expect("block side");
    let kind: usize = args[2].parse().expect("strength case");
    let calls: usize = args[3].parse().expect("calls");
    let dsp = Dsp::with_isa(isa).expect("ISA available");
    let input: [u8; SIDE * SIDE] =
        core::array::from_fn(|i| 120 + u8::from(i % SIDE >= ORIGIN) * 4 + (i % 3) as u8);
    let mut output = input;
    let source = PlaneView::new(&input, SIDE, SIDE, SIDE).expect("source");
    for _ in 0..calls {
        let mut dst = PlaneViewMut::new(&mut output, SIDE, SIDE, SIDE).expect("output");
        dsp.cdef_filter_interior(
            side,
            &source,
            (ORIGIN, ORIGIN),
            STRENGTHS[kind],
            3,
            4,
            &mut dst,
        );
        black_box(output[ORIGIN * SIDE + ORIGIN]);
    }
    println!("{}", output[ORIGIN * SIDE + ORIGIN]);
}
