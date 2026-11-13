//! Fixed-call VP9 interpolation probe for load-independent instruction profiles.
use nc_dsp::{Dsp, Isa, Vp9Filter};
use std::hint::black_box;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert_eq!(
        args.len(),
        6,
        "usage: vp9_mc_probe ISA SIDE KIND AXIS CALLS"
    );
    let isa = match args[1].as_str() {
        "scalar" => Isa::Scalar,
        "avx2" => Isa::Avx2,
        "neon" => Isa::Neon,
        "simd128" => Isa::Simd128,
        _ => panic!("unknown ISA"),
    };
    let side: usize = args[2].parse().expect("side");
    let kind = if args[3] == "0" {
        Vp9Filter::Regular
    } else {
        Vp9Filter::Bilinear
    };
    let axis: usize = args[4].parse().expect("axis");
    let calls: u32 = args[5].parse().expect("calls");
    let dsp = Dsp::with_isa(isa).expect("ISA supported");
    let stride = side + 7;
    let src: Vec<u8> = (0..stride * stride)
        .map(|i| (i * 37 + i / stride * 19) as u8)
        .collect();
    let mut dst = vec![0; side * side];
    let phase = [(0, 0), (7, 0), (0, 7), (7, 9)][axis];
    let mut checksum = 0u32;
    for _ in 0..calls {
        dsp.vp9_convolve(kind, black_box(&src), stride, phase, (side, side), &mut dst);
        checksum = checksum.wrapping_add(u32::from(black_box(dst[0])));
    }
    println!("{checksum}");
}
