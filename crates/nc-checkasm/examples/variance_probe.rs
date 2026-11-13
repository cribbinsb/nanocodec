//! Fixed-call public variance probe; references report SSE and variance.
use nc_common::PlaneView;
use nc_dsp::{BlockSize, Dsp, Isa};
use std::hint::black_box;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).is_some_and(|s| s == "--version") {
        println!(
            "variance_probe {}",
            option_env!("NC_BUILD_COMMIT").unwrap_or("unknown")
        );
        return;
    }
    assert_eq!(
        args.len(),
        6,
        "usage: variance_probe ISA WIDTH HEIGHT STRIDE CALLS"
    );
    let isa: Isa = args[1].parse().expect("ISA");
    let w: usize = args[2].parse().expect("width");
    let h: usize = args[3].parse().expect("height");
    let stride: usize = args[4].parse().expect("stride");
    let calls: u64 = args[5].parse().expect("calls");
    let len = 1 + (h - 1) * stride + w;
    let source: Vec<u8> = (0..len).map(|i| (i * 37 + i / stride * 19) as u8).collect();
    let prediction: Vec<u8> = (0..len)
        .map(|i| (i * 11 + i / stride * 7 + 43) as u8)
        .collect();
    let src = PlaneView::new(&source[1..], stride, w, h).expect("source");
    let pred = PlaneView::new(&prediction[1..], stride, w, h).expect("prediction");
    let size = BlockSize::new(w, h).expect("block size");
    let dsp = Dsp::with_isa(isa).expect("ISA supported");
    let scalar = Dsp::with_isa(Isa::Scalar).expect("scalar");
    assert_eq!(
        dsp.variance(size, &src, &pred),
        scalar.variance(size, &src, &pred)
    );
    let mut checksum = 0u64;
    // References specialize the power-of-two area into a shift. Match that
    // finishing work rather than timing a runtime integer division.
    let area_shift = (w * h).ilog2();
    for _ in 0..calls {
        let value = dsp.variance(size, black_box(&src), black_box(&pred));
        let sum = i64::from(value.sum);
        let variance = value.sse - ((sum * sum) >> area_shift) as u32;
        checksum = checksum.wrapping_add(u64::from(value.sse) + u64::from(variance));
    }
    println!("{}", black_box(checksum));
}
