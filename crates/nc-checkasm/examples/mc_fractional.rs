//! Scratch: ns/call of the AV1 MC kernels (put and prep) per filter, axis
//! and square size, timed as nc-checkasm's `time` (20 ms calibrated, min of 5).
#![forbid(unsafe_code)]
use nc_dsp::{BlockSize, Dsp, Filter, Isa};
use std::hint::black_box;
use std::time::{Duration, Instant};

const TARGET: Duration = Duration::from_millis(20);
const SS: usize = 160;

fn time(mut f: impl FnMut() -> u32) -> f64 {
    let mut calls = 16u64;
    let calls = loop {
        let start = Instant::now();
        for _ in 0..calls {
            black_box(f());
        }
        let e = start.elapsed();
        if e >= TARGET / 4 {
            break ((calls as f64 * TARGET.as_secs_f64() / e.as_secs_f64()) as u64).max(1);
        }
        calls *= 2;
    };
    (0..5)
        .map(|_| {
            let s = Instant::now();
            for _ in 0..calls {
                black_box(f());
            }
            s.elapsed().as_nanos() as f64 / calls as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let dsp = Dsp::with_isa(Isa::Neon).expect("NEON MC bench");
    eprintln!("isa {:?}", dsp.isa());
    let reference = Dsp::with_isa(Isa::Scalar).expect("scalar reference");
    let src: Vec<u8> = (0..SS * SS)
        .map(|i| ((i * 37 + i / SS * 17) % 251) as u8)
        .collect();
    let mut put = vec![0u8; 64 * 64];
    let mut prep = vec![0i16; 64 * 64];
    for side in [8usize, 16, 32, 64] {
        let size = BlockSize::new(side, side).unwrap();
        for (fname, filter) in [
            ("regular", Filter::Av1),
            ("smooth", Filter::Av1Smooth),
            ("sharp", Filter::Av1Sharp),
        ] {
            for (axis, fx, fy) in [("h", 5, 0), ("v", 0, 9), ("hv", 5, 9)] {
                let mut expected = put.clone();
                reference.predict_samples(
                    filter,
                    size,
                    (&src, SS),
                    (fx, fy),
                    (&mut expected, side),
                );
                dsp.predict_samples(filter, size, (&src, SS), (fx, fy), (&mut put, side));
                assert_eq!(put, expected, "put {fname} {axis} {side}");
                let mut expected = prep.clone();
                reference.predict_compound_samples(
                    filter,
                    size,
                    (&src, SS),
                    (fx, fy),
                    (&mut expected, side),
                );
                dsp.predict_compound_samples(filter, size, (&src, SS), (fx, fy), (&mut prep, side));
                assert_eq!(prep, expected, "prep {fname} {axis} {side}");
                let p = time(|| {
                    dsp.predict_samples(filter, size, (&src, SS), (fx, fy), (&mut put, side));
                    u32::from(put[0])
                });
                let q = time(|| {
                    dsp.predict_compound_samples(
                        filter,
                        size,
                        (&src, SS),
                        (fx, fy),
                        (&mut prep, side),
                    );
                    prep[0] as u32
                });
                println!("{fname}\tw{side}\t{axis}\tput\t{p:.1}");
                println!("{fname}\tw{side}\t{axis}\tprep\t{q:.1}");
            }
        }
    }
}
