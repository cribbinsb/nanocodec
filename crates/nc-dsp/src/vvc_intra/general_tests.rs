//! Independent prepared-reference fixtures for general intra and ISP prediction.
use crate::{Dsp, VvcIntraMode};
#[test]
#[ignore = "requires pinned VVdeC D_PRED trace (NC_VVC_GENERAL_PRED_TRACE)"]
fn official_general_predictions() {
    let trace =
        std::fs::read_to_string(std::env::var("NC_VVC_GENERAL_PRED_TRACE").unwrap()).unwrap();
    let dsp = Dsp::new();
    let mut meta = Vec::new();
    let mut edge = Vec::new();
    let mut count = [0; 3];
    for line in trace.lines() {
        let values = |s: &str| {
            s.split_whitespace()
                .map(|v| v.parse::<i32>().unwrap())
                .collect::<Vec<_>>()
        };
        if let Some(s) = line.strip_prefix("NC_META ") {
            meta = values(s);
        }
        if let Some(s) = line.strip_prefix("NC_EDGE ") {
            edge = values(s);
        }
        if let Some(s) = line.strip_prefix("NC_PRED ") {
            let expected = values(s);
            assert_eq!(&edge[..6], &expected[..6]);
            let size = (edge[3] as usize, edge[4] as usize);
            let direction = edge[5] as u8;
            let top = &edge[7..7 + meta[4] as usize];
            let left = &edge[7 + meta[4] as usize..];
            let mut output = vec![0; size.0 * size.1];
            let category = if meta[0] != 0 {
                dsp.vvc_intra_predict_isp(
                    (meta[2] as usize, meta[3] as usize),
                    size,
                    direction,
                    top,
                    left,
                    edge[6],
                    &mut output,
                );
                0
            } else if meta[1] != 0 {
                // The MRL trace's prepared buffer includes the offset line.
                continue;
            } else {
                let mode = match direction {
                    0 => VvcIntraMode::Planar,
                    1 => VvcIntraMode::Dc,
                    _ => VvcIntraMode::Angular {
                        direction,
                        luma: edge[0] == 0,
                    },
                };
                dsp.vvc_intra_predict_rect(size, mode, top, left, edge[6], &mut output);
                if size.0 == 64 || size.1 == 64 { 1 } else { 2 }
            };
            let actual: Vec<_> = output.into_iter().map(i32::from).collect();
            assert_eq!(
                actual,
                expected[6..],
                "block {:?} metadata {meta:?}",
                &edge[..6]
            );
            count[category] += 1;
        }
    }
    eprintln!(
        "matched ISP={}, 64-point={}, ordinary={}",
        count[0], count[1], count[2]
    );
    assert!(count[0] > 0 && count.iter().sum::<usize>() > 100);
}
