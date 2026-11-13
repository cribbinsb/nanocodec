//! Round trips through writer and reader for every size, plane, prediction
//! kind and quantizer category, plus context derivation checks. The original
//! AVM oracle for contexts and coded bytes is `tools/codegen/check_av2_coeff.py`.
use super::*;

/// One representative qindex per CDF quantizer category, plus the boundaries.
const QINDICES: [u8; 8] = [1, 90, 91, 140, 141, 190, 191, 255];

struct Lcg(u32);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        self.0 >> 8
    }
}

/// A block whose nonzero levels fill a random scan prefix with a mix of
/// small, saturating and 20-bit magnitudes.
fn random_block(rng: &mut Lcg, tx: TxSize) -> Vec<i32> {
    let scan = levels::scan(tx);
    let eob = match rng.next() % 4 {
        0 => 0,
        1 => 1 + rng.next() as usize % 4,
        _ => 1 + rng.next() as usize % tx.coded_len(),
    };
    let mut block = vec![0; tx.coded_len()];
    for &pos in &scan[..eob] {
        let magnitude = match rng.next() % 16 {
            0..=6 => 0,
            7..=12 => 1 + rng.next() % 9,
            13 | 14 => rng.next() % 400,
            _ => rng.next() % (MAX_LEVEL + 1),
        } as i32;
        block[usize::from(pos)] = if rng.next().is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        };
    }
    block
}

fn contexts(rng: &mut Lcg, plane: u8, is_inter: bool) -> Context {
    let skip_contexts = if plane == PLANE_V { 12 } else { 10 };
    Context {
        is_inter,
        plane,
        skip: (rng.next() % skip_contexts) as u8,
        dc_sign: (rng.next() % 3) as u8,
    }
}

fn round_trip(adaptive: bool, qindex: u8, cases: &[(TxSize, Context, Vec<i32>)]) {
    let mut writer = Writer::new();
    if adaptive {
        writer.enable_adaptation(qindex);
    }
    for (tx, ctx, block) in cases {
        write(&mut writer, block, *tx, *ctx, qindex).unwrap();
    }
    let packet = writer.finish();
    let mut reader = Reader::new(&packet).unwrap();
    if adaptive {
        reader.enable_adaptation(qindex);
    }
    for (index, (tx, ctx, block)) in cases.iter().enumerate() {
        let mut decoded = vec![7; tx.coded_len()];
        read(&mut reader, &mut decoded, *tx, *ctx, qindex).unwrap();
        assert_eq!(&decoded, block, "block {index} {tx:?} {ctx:?} q={qindex}");
    }
}

#[test]
fn every_size_plane_prediction_and_quantizer_round_trips() {
    let mut rng = Lcg(91);
    for qindex in QINDICES {
        for adaptive in [false, true] {
            let mut cases = Vec::new();
            for tx in TxSize::ALL {
                for plane in [PLANE_Y, PLANE_U, PLANE_V] {
                    if plane != PLANE_Y && tx == TxSize::Tx64 {
                        continue;
                    }
                    for is_inter in [false, true] {
                        for _ in 0..3 {
                            let ctx = contexts(&mut rng, plane, is_inter);
                            cases.push((tx, ctx, random_block(&mut rng, tx)));
                        }
                    }
                }
            }
            round_trip(adaptive, qindex, &cases);
        }
    }
}

#[test]
fn every_end_of_block_round_trips() {
    for tx in TxSize::ALL {
        let scan = levels::scan(tx);
        let cases: Vec<_> = (0..=tx.coded_len())
            .map(|eob| {
                let mut block = vec![0; tx.coded_len()];
                for (index, &pos) in scan[..eob].iter().enumerate() {
                    block[usize::from(pos)] = if index % 2 == 0 {
                        index as i32 * 29 + 1
                    } else {
                        -(index as i32 * 19 + 1)
                    };
                }
                let ctx = Context {
                    is_inter: eob % 2 != 0,
                    dc_sign: (eob % 3) as u8,
                    ..Context::default()
                };
                (tx, ctx, block)
            })
            .collect();
        round_trip(true, 100, &cases);
    }
}

#[test]
fn lossless_tx4_round_trips_in_every_plane() {
    let mut rng = Lcg(5);
    let cases: Vec<_> = (0..90)
        .map(|i| {
            let plane = (i % 3) as u8;
            let ctx = contexts(&mut rng, plane, i % 2 == 0);
            (TxSize::Tx4, ctx, random_block(&mut rng, TxSize::Tx4))
        })
        .collect();
    round_trip(false, 0, &cases);
    round_trip(true, 0, &cases);
}

#[test]
fn rejects_unsupported_combinations() {
    let mut writer = Writer::new();
    let ctx = Context::default();
    let block = vec![0; TxSize::Tx8.coded_len()];
    assert!(write(&mut writer, &block, TxSize::Tx8, ctx, 0).is_err());
    assert!(write(&mut writer, &block[..16], TxSize::Tx8, ctx, 9).is_err());
    let chroma = Context {
        plane: PLANE_U,
        ..ctx
    };
    assert!(write(&mut writer, &vec![0; 1024], TxSize::Tx64, chroma, 9).is_err());
    let too_large = [i32::try_from(MAX_LEVEL).unwrap() + 1; 16];
    assert!(write(&mut writer, &too_large, TxSize::Tx4, ctx, 9).is_err());
    let bad = Context { skip: 10, ..ctx };
    assert!(write(&mut writer, &block, TxSize::Tx8, bad, 9).is_err());
}

#[test]
fn luma_skip_context_follows_avm_table() {
    // AVM `skip_contexts[top][left]` from `get_txb_ctx`.
    const TABLE: [[u8; 5]; 5] = [
        [1, 2, 2, 3, 3],
        [2, 2, 3, 3, 4],
        [2, 3, 3, 4, 4],
        [3, 3, 4, 4, 5],
        [3, 4, 4, 5, 5],
    ];
    for (top, row) in TABLE.iter().enumerate() {
        for (left, &expected) in row.iter().enumerate() {
            // Above: two units whose OR has level `top`; left: one per unit.
            let above = [top as u8 & 1 | 16, top as u8 & 6];
            let left_units = [left as u8, 8];
            let ctx =
                Context::derive(0, 4, TxSize::Tx8, false, &above, &left_units, false).unwrap();
            assert_eq!(ctx.skip, expected, "top={top} left={left}");
        }
    }
    let saturated = Context::derive(0, 5, TxSize::Tx8, true, &[7, 7], &[6, 1], false).unwrap();
    assert_eq!(saturated.skip, 5);
    let whole = Context::derive(0, 4, TxSize::Tx16, false, &[7; 4], &[7; 4], false).unwrap();
    assert_eq!(whole.skip, 0);
}

#[test]
fn chroma_skip_contexts_and_v_offsets() {
    let zero = [0; 8];
    let coded = [0, 0, 0, 1, 0, 0, 0, 0];
    let u = Context::derive(PLANE_U, 4, TxSize::Tx16, false, &coded, &zero, true).unwrap();
    assert_eq!(u.skip, 7);
    let v = |block_log2, u_coded| {
        Context::derive(
            PLANE_V,
            block_log2,
            TxSize::Tx16,
            false,
            &coded,
            &coded,
            u_coded,
        )
        .unwrap()
        .skip
    };
    assert_eq!(v(4, false), 2);
    assert_eq!(v(4, true), 8);
    assert_eq!(v(5, false), 5);
    assert_eq!(v(5, true), 11);
    // Only the units covered by the transform count.
    let outside = Context::derive(PLANE_U, 3, TxSize::Tx4, false, &coded, &zero, false).unwrap();
    assert_eq!(outside.skip, 6);
}

#[test]
fn dc_sign_context_balances_neighbour_signs() {
    let negative = 1 << 3;
    let positive = 2 << 3;
    let derive = |above: &[u8], left: &[u8]| {
        Context::derive(0, 4, TxSize::Tx16, false, above, left, false)
            .unwrap()
            .dc_sign
    };
    assert_eq!(derive(&[0; 4], &[0; 4]), 0);
    assert_eq!(derive(&[negative, positive, 0, 0], &[0; 4]), 0);
    assert_eq!(derive(&[negative, 0, 0, 0], &[positive, negative, 0, 0]), 1);
    assert_eq!(
        derive(
            &[positive; 4],
            &[negative; 3]
                .iter()
                .chain(&[0])
                .copied()
                .collect::<Vec<_>>()
        ),
        2
    );
    assert!(Context::derive(0, 4, TxSize::Tx16, false, &[3 << 3; 4], &[0; 4], false).is_err());
    assert!(Context::derive(0, 4, TxSize::Tx16, false, &[0; 3], &[0; 4], false).is_err());
    assert!(Context::derive(0, 3, TxSize::Tx16, false, &[0; 4], &[0; 4], false).is_err());
}

#[test]
fn entropy_context_caps_level_and_records_dc_sign() {
    assert_eq!(entropy_context(&[0; 16]), 0);
    assert_eq!(entropy_context(&[0, 2, -1, 0]), 3);
    assert_eq!(entropy_context(&[-1, 9, 0, 0]), 7 | 1 << 3);
    assert_eq!(entropy_context(&[4, 0, 0, 0]), 4 | 2 << 3);
    assert_eq!(entropy_context(&[i32::MIN, i32::MAX]), 7 | 1 << 3);
}

/// The context byte `write_typed` gathers while coding equals the one
/// computed from the whole block, for sparse and dense random blocks.
#[test]
fn written_entropy_context_matches_the_block() {
    let mut rng = Lcg(31);
    let ctx = Context::derive(0, 4, TxSize::Tx8, false, &[0; 2], &[0; 2], false).unwrap();
    for density in [1, 4, 16, 64] {
        for _ in 0..50 {
            let mut levels = [0i32; 64];
            for level in &mut levels {
                if rng.next() % 64 < density {
                    *level = (rng.next() % 9) as i32 - 4;
                }
            }
            let mut w = Writer::new();
            let written = write_typed(&mut w, &levels, TxSize::Tx8, ctx, 120, Kind::DCT_ONLY);
            assert_eq!(written.unwrap(), entropy_context(&levels), "{levels:?}");
        }
    }
}

/// Luma blocks of every set mode round-trip with each type our encoder may
/// give them, through both the fixed and the adaptive models, plain and
/// trellis-coded (TCQ's state-selected models and paired remainders).
#[test]
fn typed_luma_blocks_round_trip_in_every_set() {
    let mut rng = Lcg(23);
    for (set, coding) in [
        TxSetMode::Full,
        TxSetMode::IntraDctAdst,
        TxSetMode::IntraDct,
    ]
    .into_iter()
    .flat_map(|set| [(set, LevelCoding::Plain), (set, LevelCoding::Tcq)])
    {
        for adaptive in [false, true] {
            let mut cases = Vec::new();
            for tx in TxSize::ALL {
                for is_inter in [false, true] {
                    for _ in 0..12 {
                        let ctx = contexts(&mut rng, PLANE_Y, is_inter);
                        let block = random_block(&mut rng, tx);
                        let eob = levels::end_of_block(&block, tx);
                        let types = set.luma_types(tx, is_inter);
                        let tx_type = if eob == 0 || !(is_inter || eob > 1) || types.is_empty() {
                            TxType::DctDct
                        } else {
                            types[rng.next() as usize % types.len()]
                        };
                        cases.push((tx, ctx, block, tx_type));
                    }
                }
            }
            let mut writer = Writer::new();
            if adaptive {
                writer.enable_adaptation(100);
            }
            for (tx, ctx, block, tx_type) in &cases {
                let kind = Kind {
                    set,
                    tx_type: *tx_type,
                    levels: coding,
                };
                write_typed(&mut writer, block, *tx, *ctx, 100, kind).unwrap();
            }
            let packet = writer.finish();
            let mut reader = Reader::new(&packet).unwrap();
            if adaptive {
                reader.enable_adaptation(100);
            }
            for (tx, ctx, block, tx_type) in &cases {
                let mut decoded = vec![0; tx.coded_len()];
                let read_type =
                    read_typed(&mut reader, &mut decoded, *tx, *ctx, 100, (set, coding)).unwrap();
                assert_eq!(
                    (&decoded, read_type),
                    (block, *tx_type),
                    "{set:?} {coding:?} {tx:?} {ctx:?}"
                );
            }
        }
    }
}

/// The full sets' 1-D inter types and non-DCT intra types are outside our
/// subset; chroma stays DCT-only.
#[test]
fn full_set_rejects_types_outside_the_subset() {
    let mut block = vec![0; TxSize::Tx8.coded_len()];
    block[..3].copy_from_slice(&[5, -2, 1]);
    let full = |tx_type| Kind {
        set: TxSetMode::Full,
        tx_type,
        levels: LevelCoding::Plain,
    };
    let inter = Context {
        is_inter: true,
        ..Context::default()
    };
    let intra = Context::default();
    let chroma = Context {
        plane: PLANE_U,
        ..inter
    };
    let mut writer = Writer::new();
    for (ctx, tx_type) in [
        (inter, TxType::VDct),
        (inter, TxType::HFlipadst),
        (intra, TxType::AdstAdst),
        (chroma, TxType::AdstAdst),
    ] {
        assert!(write_typed(&mut writer, &block, TxSize::Tx8, ctx, 100, full(tx_type)).is_err());
    }
    for tx_type in [TxType::FlipadstAdst, TxType::Idtx, TxType::DctDct] {
        write_typed(&mut writer, &block, TxSize::Tx8, inter, 100, full(tx_type)).unwrap();
    }
}
