//! Forced-direction syntax gate independent of the encoder's candidate ranking.
//! Every mode must decode correctly even when a normal search would not pick it.
use crate::{
    cabac::CabacWriter, chroma_qp::ChromaQpTable, contexts::Contexts, headers, intra,
    motion::MotionGrid, recon, syntax,
};
use nc_common::{Picture, PlaneIndex, yuv};
use nc_dsp::Dsp;
use nc_encoder::Block;
use std::process::Command;

/// `FFmpeg` releases before this mis-decode VVC ALF streams: they read
/// uninitialised `CC-ALF` indexes whenever ALF is on (`FFmpeg` ticket 11325).
const MIN_FFMPEG_VERSION: (u32, u32) = (8, 0);

/// The `FFmpeg` to check against: `NC_FFMPEG`, the pinned decode-only build
/// (`anchors/<arch>/bin/ffmpeg-decode`, `build_anchors.sh ffmpeg`), never one
/// on `PATH`. Panics unless it reports a release >= [`MIN_FFMPEG_VERSION`].
fn pinned_ffmpeg() -> String {
    let ffmpeg = std::env::var("NC_FFMPEG").expect("NC_FFMPEG must point to FFmpeg >= 8.0");
    let banner = Command::new(&ffmpeg)
        .arg("-version")
        .output()
        .unwrap()
        .stdout;
    let banner = String::from_utf8_lossy(&banner);
    let version = banner
        .strip_prefix("ffmpeg version ")
        .map(|v| v.trim_start_matches('n'))
        .and_then(|v| {
            let mut parts = v.split(|c: char| !c.is_ascii_digit());
            Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
        });
    assert!(
        version.is_some_and(|v: (u32, u32)| v >= MIN_FFMPEG_VERSION),
        "{ffmpeg} is not FFmpeg >= 8.0 (older ones mis-decode VVC ALF, ticket 11325): {}",
        banner.lines().next().unwrap_or("")
    );
    ffmpeg
}

/// Round-to-nearest quantiser offset (1/512 step): the gate checks syntax,
/// not the encoder's dead zone.
const NEAREST_ROUNDING_512THS: i32 = 256;
/// A 64x64 picture of 32x32 CTUs.
const GEOMETRY: headers::Geometry = headers::Geometry {
    width: 64,
    height: 64,
    coded_width: 64,
    coded_height: 64,
    ctu_log2: 5,
    partitioning: headers::Partitioning::QUADTREE,
};

struct ForcedFrame {
    dsp: Dsp,
    source: Picture,
    picture: Picture,
    grid: MotionGrid,
    contexts: Contexts,
    writer: CabacWriter,
    qp: u32,
    direction: u8,
    leaf_log2: u8,
    chroma_index: Option<usize>,
}
impl ForcedFrame {
    fn new(qp: u32, direction: u8, leaf_log2: u8) -> Self {
        let mut source = Picture::new(64, 64, 0).unwrap();
        for plane in PlaneIndex::ALL {
            let mut view = source.plane_mut(plane).view_mut();
            for y in 0..view.height() {
                for (x, pixel) in view.row_mut(y).iter_mut().enumerate() {
                    *pixel = ((x * 19 + y * 31 + plane as usize * 17)
                        ^ (x * y + usize::from(direction))) as u8;
                }
            }
        }
        Self {
            dsp: Dsp::new(),
            source,
            picture: Picture::new(64, 64, 0).unwrap(),
            grid: MotionGrid::new(64, 64, GEOMETRY.ctu_log2, headers::DEFAULT_MERGE_CANDIDATES),
            contexts: Contexts::new(qp, false),
            writer: CabacWriter::new(),
            qp,
            direction,
            leaf_log2,
            chroma_index: None,
        }
    }
    /// Write quadtree node `block`'s `split_cu_flag` (intra slice).
    fn code_split(&mut self, block: Block, split: bool) {
        let node = crate::coding_tree::Node {
            block,
            mtt_depth: 0,
        };
        let allowed = crate::coding_tree::Allowed::of(
            &node,
            headers::Partitioning::QUADTREE,
            (GEOMETRY.coded_width, GEOMETRY.coded_height),
        );
        let tree_split = if split {
            crate::coding_tree::TreeSplit::Quad
        } else {
            crate::coding_tree::TreeSplit::None
        };
        crate::coding_tree::code(
            &mut self.writer,
            &mut self.contexts.tree,
            allowed,
            self.grid.split_contexts(&node, allowed),
            tree_split,
        )
        .unwrap();
    }
    fn block(&mut self, block: Block) {
        if block.log2_size() > 3 {
            let split = block.log2_size() > self.leaf_log2;
            self.code_split(block, split);
            if split {
                for child in block.quadrants() {
                    self.block(child);
                }
                return;
            }
        }
        // Rotate neighbors as well as the current mode, to exercise MPM wrap,
        // neighboring-angle cases and the non-MPM truncated-binary branch.
        let selected = ((usize::from(self.direction)
            + block.x / block.size() * 7
            + block.y / block.size() * 13)
            % 67) as u8;
        let chroma_mode = self
            .chroma_index
            .map_or(selected, |index| crate::chroma::candidates(selected)[index]);
        let mut levels: [Vec<i32>; 3] = std::array::from_fn(|_| Vec::new());
        for plane in PlaneIndex::ALL {
            let shift = usize::from(plane != PlaneIndex::Y);
            let pred = intra::prediction(
                &self.dsp,
                &self.picture,
                Some(&self.grid),
                plane,
                (&block, &block),
                if plane == PlaneIndex::Y {
                    selected
                } else {
                    chroma_mode
                },
            );
            let mut scratch = recon::QuantScratch::default();
            let coeff = recon::quantize(
                &self.dsp,
                &self.source,
                plane,
                block.x >> shift,
                block.y >> shift,
                &pred,
                recon::Quant {
                    qp: self.qp,
                    rounding_512ths: NEAREST_ROUNDING_512THS,
                    zero_bound_percent: recon::EXACT_ZERO_BOUND_PERCENT,
                    shave_gap: 0,
                    forward: nc_dsp::HevcForwardMode::EXACT,
                    transform: crate::mts::Selection::Dct2,
                },
                None,
                &mut scratch,
            )
            .to_vec();
            recon::store(
                &self.dsp,
                &mut self.picture,
                plane,
                (block.x >> shift, block.y >> shift, block.size() >> shift),
                &pred,
                &coeff,
                recon::Scaling {
                    qp: self.qp,
                    dependent: false,
                    transform: crate::mts::Selection::Dct2,
                },
            );
            levels[plane as usize] = coeff;
        }
        let units = crate::transform_tree::coded_units(
            &block,
            None,
            &levels,
            crate::transform_tree::MAX_TB_LOG2,
        );
        syntax::block(
            &mut self.writer,
            &mut self.contexts,
            (false, crate::headers::Tools::default()),
            &self.grid.predictors(&block).contexts(),
            &syntax::Coding {
                inter: None,
                cu: block,
                levels,
                skip: false,
                intra_mode: selected,
                chroma_mode,
                isp: None,
                qp_delta: None,
                mts: 0,
            },
        );
        self.grid.record(&block, None, false, selected, &units);
    }
    fn encode(mut self) -> (Vec<u8>, Vec<u8>) {
        for y in (0..64).step_by(32) {
            for x in (0..64).step_by(32) {
                self.grid.begin_ctu(x);
                // Every CTU's SAO is off, so each merges with a neighbour.
                let neighbour = crate::sao::CtuSao::default();
                crate::sao::code(
                    &mut self.writer,
                    &mut self.contexts,
                    neighbour,
                    (x > 0).then_some(neighbour),
                    (y > 0).then_some(neighbour),
                )
                .unwrap();
                self.block(Block::square(x, y, 5));
            }
        }
        self.writer.encode_terminate(true);
        let mut coded = self.writer.finish();
        coded.trailing_bits();
        let mut slice = headers::slice(self.qp, false, 0, None, false, None);
        slice.extend(coded.into_bytes());
        let mut packet = Vec::new();
        headers::nal(
            headers::SPS_NUT,
            &headers::sps(
                &GEOMETRY,
                nc_common::ColourInfo::default(),
                &ChromaQpTable::identity(),
                headers::Tools::default(),
            ),
            &mut packet,
        );
        headers::nal(
            headers::PPS_NUT,
            &headers::pps(&GEOMETRY, false),
            &mut packet,
        );
        headers::nal(headers::IDR_N_LP, &slice, &mut packet);
        crate::filter::apply(
            &self.dsp,
            &mut self.picture,
            &self.grid,
            &crate::filter::DeblockQp {
                frame: [self.qp; 2],
                blocks: None,
            },
        );
        let mut expected = Vec::new();
        yuv::write_frame(&mut expected, &self.picture).unwrap();
        (packet, expected)
    }
}

#[test]
#[ignore = "requires NC_FFMPEG (FFmpeg >= 8.0) and NC_VVDEC pointing to vvdecapp"]
fn all_directions_match_standard_decoders() {
    let vvdec = std::env::var("NC_VVDEC").unwrap();
    for leaf_log2 in [3, 4, 5] {
        for qp in [0, 37] {
            let mut stream = Vec::new();
            let mut expected = Vec::new();
            let mut decoder = crate::Decoder::new();
            for (direction, chroma_index) in (0..67).flat_map(|direction| {
                [None, Some(0), Some(1), Some(2), Some(3)].map(|chroma| (direction, chroma))
            }) {
                let mut frame = ForcedFrame::new(qp, direction, leaf_log2);
                frame.chroma_index = chroma_index;
                let (packet, recon) = frame.encode();
                let decoded = decoder.decode(&packet).unwrap();
                let mut own = Vec::new();
                yuv::write_frame(&mut own, &decoded).unwrap();
                assert!(
                    own == recon,
                    "own direction={direction} log2={leaf_log2} qp={qp}"
                );
                stream.extend(packet);
                expected.extend(recon);
            }
            let path = std::env::temp_dir().join(format!(
                "nc-vvc-angular-{}-{leaf_log2}-{qp}.vvc",
                std::process::id()
            ));
            std::fs::write(&path, stream).unwrap();
            let ffmpeg = Command::new(pinned_ffmpeg())
                .args(["-v", "error", "-threads", "1", "-f", "vvc", "-i"])
                .arg(&path)
                .args(["-f", "rawvideo", "-pix_fmt", "yuv420p", "-"])
                .output()
                .unwrap();
            let vvdec = Command::new(&vvdec)
                .args(["-v", "0", "-t", "1", "-b"])
                .arg(&path)
                .args(["-o", "/dev/stdout"])
                .output()
                .unwrap();
            std::fs::remove_file(path).unwrap();
            for (name, result) in [("ffmpeg", ffmpeg), ("vvdec", vvdec)] {
                assert!(
                    result.status.success(),
                    "{name}: {}",
                    String::from_utf8_lossy(&result.stderr)
                );
                assert_eq!(
                    result.stdout.len(),
                    expected.len(),
                    "{name} log2={leaf_log2} qp={qp}"
                );
                assert!(
                    result.stdout == expected,
                    "{name} log2={leaf_log2} qp={qp} first difference {:?}",
                    result
                        .stdout
                        .iter()
                        .zip(&expected)
                        .position(|(a, b)| a != b)
                );
            }
        }
    }
}
