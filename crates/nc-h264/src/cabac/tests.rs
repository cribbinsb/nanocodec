//! The byte-oriented engine against the literal §9.3.4 engine on random
//! bin sequences: the slice data must be bit-identical.

use nc_common::BitWriter;

use super::reference::SpecEncoder;
use super::{BinSink, CabacEncoder, Context, SliceKind};
use crate::tables::CONTEXT_COUNT;

/// `SplitMix64`.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

fn header(rng: &mut Rng) -> BitWriter {
    let mut out = BitWriter::new();
    for _ in 0..(rng.next() % 4) {
        out.put_bits((rng.next() & 0xff) as u32, 8);
    }
    out
}

/// One bin of a fuzz sequence.
#[derive(Debug, Clone, Copy)]
enum Bin {
    Bypass(bool),
    /// A terminate bin with value 0.
    Terminate,
    Decision(usize, bool),
}

fn code(sink: &mut impl BinSink, bins: &[Bin]) {
    for &bin in bins {
        match bin {
            Bin::Bypass(bin) => sink.encode_bypass(bin),
            Bin::Terminate => sink.encode_terminate(false),
            Bin::Decision(ctx, bin) => sink.encode_decision(ctx, bin),
        }
    }
}

#[test]
fn byte_engine_matches_the_spec_engine() {
    let mut rng = Rng(0x00c0_ffee);
    for case in 0..4000 {
        let kind = if case % 2 == 0 {
            SliceKind::I
        } else {
            SliceKind::P
        };
        let qp = (rng.next() % 52) as u8;
        let prefix = header(&mut rng);
        let mut fast = CabacEncoder::new(prefix.clone(), kind, 0, qp);
        let mut spec = SpecEncoder::new(prefix, kind, qp);
        // Skewed bins (mostly MPS-like runs) exercise long renormalisation
        // runs, carries and outstanding bytes; a few contexts are hot.
        let length = rng.next() % 3000;
        let bias = rng.next() % 16;
        let bins: Vec<Bin> = (0..length)
            .map(|_| {
                let r = rng.next();
                match r % 16 {
                    0 => Bin::Bypass((r >> 8) & 1 != 0),
                    1 if (r >> 8).is_multiple_of(64) => Bin::Terminate,
                    _ => {
                        let ctx = if (r >> 12).is_multiple_of(4) {
                            ((r >> 16) % CONTEXT_COUNT as u64) as usize
                        } else {
                            ((r >> 16) % 8) as usize
                        };
                        Bin::Decision(ctx, (r >> 40) % 16 < bias)
                    }
                }
            })
            .collect();
        for &bin in &bins {
            match bin {
                Bin::Bypass(bin) => spec.encode_bypass(bin),
                Bin::Terminate => spec.encode_terminate(false),
                Bin::Decision(ctx, bin) => spec.encode_decision(ctx, bin),
            }
        }
        // The fast engine codes runs of random length through writers (as
        // macroblocks are), and single bins through the encoder itself.
        let mut rest = bins.as_slice();
        while !rest.is_empty() {
            let (run, after) = rest.split_at((rng.next() as usize % 64).min(rest.len()));
            if run.is_empty() {
                code(&mut fast, &after[..1]);
                rest = &after[1..];
            } else {
                fast.write(|writer| code(writer, run));
                rest = after;
            }
        }
        fast.encode_terminate(true);
        spec.encode_terminate(true);
        assert_eq!(fast.finish(), spec.finish(), "case {case}");
    }
}

#[test]
fn context_init_matches_spec_examples() {
    // ctxIdx 276 has no (m, n); mb_qp_delta ctxIdx 60 is (0, 41): at any
    // QP preCtxState = 41, i.e. pStateIdx 22 with MPS 0.
    let ctx = Context::init(0, 41, 30);
    assert_eq!((ctx.p_state_idx(), ctx.val_mps()), (22, false));
    // (20, -15) at QP 26: (20*26 >> 4) - 15 = 17 → pStateIdx 46, MPS 0.
    let ctx = Context::init(20, -15, 26);
    assert_eq!((ctx.p_state_idx(), ctx.val_mps()), (46, false));
}

#[test]
fn terminated_slice_ends_with_stop_bit() {
    let mut enc = CabacEncoder::new(BitWriter::new(), SliceKind::I, 0, 26);
    for i in 0..100 {
        enc.encode_decision(60, i % 7 == 0);
        enc.encode_bypass(i % 3 == 0);
    }
    enc.encode_terminate(true);
    let bytes = enc.finish();
    let last = *bytes.last().unwrap();
    assert_ne!(last, 0, "the final byte holds the stop bit");
}
