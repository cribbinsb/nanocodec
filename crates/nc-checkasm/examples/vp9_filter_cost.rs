//! Fixed-work VP9 filter probe; counter slopes remove process startup.
//! Arguments: width, lines, vertical (0/1), pattern (0..4), iterations.
use nc_dsp::{Dsp, Vp9FilterLimits, Vp9FilterWidth};
use std::hint::black_box;
const SIDE: usize = 64;
const EDGE: usize = 16;
const ORIGIN: usize = EDGE * SIDE + EDGE;
const LEVEL: u8 = 32;
fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(args.len(), 5);
    let [width, lines, vertical, pattern, iterations]: [usize; 5] = args[..].try_into().unwrap();
    let width = match width {
        4 => Vp9FilterWidth::Four,
        8 => Vp9FilterWidth::Eight,
        16 => Vp9FilterWidth::Sixteen,
        _ => panic!("width must be 4, 8 or 16"),
    };
    assert!(matches!(lines, 8 | 16 | 32));
    assert!(vertical <= 1 && pattern <= 4);
    let input: [u8; SIDE * SIDE] = core::array::from_fn(|i| {
        let (across, line) = if vertical == 0 {
            (i / SIDE, i % SIDE)
        } else {
            (i % SIDE, i / SIDE)
        };
        match pattern {
            0 => (i * 37 + i / SIDE * 11) as u8,
            1 => 120 + (across % 3) as u8 * 2,
            2 => 120 + u8::from(across >= EDGE) * 4,
            3 => {
                if line % 16 < 8 {
                    120 + u8::from(across >= EDGE) * 4
                } else {
                    (i * 37) as u8
                }
            }
            _ => 120 + ((across + line) % 2) as u8,
        }
    });
    let mut pixels = input;
    let dsp = Dsp::new();
    let (step, along) = if vertical == 0 { (SIDE, 1) } else { (1, SIDE) };
    for _ in 0..iterations {
        pixels.copy_from_slice(black_box(&input));
        dsp.vp9_filter_edge(
            &mut pixels,
            ORIGIN,
            step,
            along,
            lines,
            width,
            Vp9FilterLimits::new(LEVEL),
        );
        black_box(&pixels);
    }
    let checksum = pixels.iter().fold(0_u64, |s, &v| {
        s.wrapping_mul(257).wrapping_add(u64::from(v))
    });
    println!("{checksum}");
}
