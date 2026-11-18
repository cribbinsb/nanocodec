//! Fixed-work fractional MC driver for release instruction counts.
#![forbid(unsafe_code)]
use nc_dsp::{BlockSize, Dsp, Filter, Isa};
use std::hint::black_box;
const SS: usize = 160;
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let side: usize = args[1].parse().expect("width");
    let filter = match args[2].as_str() {
        "regular" => Filter::Av1,
        "sharp" => Filter::Av1Sharp,
        "hevc" => Filter::HevcLuma,
        "vvc" => Filter::VvcLuma,
        _ => panic!("filter"),
    };
    let (fx, fy) = match args[3].as_str() {
        "h" => (1, 0),
        "v" => (0, 1),
        "hv" => (1, 1),
        _ => panic!("axis"),
    };
    let prep = args[4] == "prep";
    let calls: usize = args[5].parse().expect("calls");
    let dsp = Dsp::with_isa(Isa::Neon).expect("NEON");
    let size = BlockSize::new(side, side).expect("size");
    let src: Vec<u8> = (0..SS * SS)
        .map(|i| ((i * 37 + i / SS * 17) % 251) as u8)
        .collect();
    let mut pixels = vec![0u8; side * side];
    let mut words = vec![0i16; side * side];
    for _ in 0..calls {
        if prep {
            dsp.predict_compound_samples(
                filter,
                size,
                (black_box(&src), SS),
                (fx, fy),
                (black_box(&mut words), side),
            );
        } else {
            dsp.predict_samples(
                filter,
                size,
                (black_box(&src), SS),
                (fx, fy),
                (black_box(&mut pixels), side),
            );
        }
    }
    let checksum: i64 = pixels.iter().map(|&x| i64::from(x)).sum::<i64>()
        + words.iter().map(|&x| i64::from(x)).sum::<i64>();
    println!("{checksum}");
}
