"""Side-by-side work budget: openh264 vs nanocodec-h264 from two callgrind
profiles of the same encode (docs/handoff/h264-vs-openh264.md).

usage: python3 -I classify.py OPENH264.cg OPENH264_NM_SYMS NANOCODEC.cg
  OPENH264_NM_SYMS: `nm openh264-harness | sort` (names its asm kernels).
  NANOCODEC.cg should come from a build with line tables
  (CARGO_PROFILE_RELEASE_DEBUG=line-tables-only) so inlined code can be
  attributed by source file.
"""
import sys, re, collections
sys.path.insert(0, '.')
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cgparse
FR = 20
CATS = ['frame ingest/padding', 'pre-analysis', 'motion search (integer)', 'sub-pel',
        'mode decision / cost estimation', 'intra search', 'prediction/MC',
        'residual + transform + quant', 'inverse + recon', 'entropy coding', 'deblocking',
        'bookkeeping/overhead']
OH = [
 (r'Expand|ReadYUV|fread|_IO_|WelsMoveMemory|CopySpatialPic|Padding|harness|main$', 0),
 (r'WelsVP::|VAA|Vaa|BackgroundDetection|SceneChange|Preprocess|PreProcess|AnalyzeSpatial|VaaCalc|SampleVariance', 1),
 (r'MotionEstimate|DiamondSearch|MeSadCost|PredictSad|CheckDirectional|FeatureSearch|LineFullSearch|SadFour|SampleSad', 2),
 (r'MeRefineFracPixel|InterMbRefinement|CalculateSatdCost|McHorVer|PixelAvg|Satd16x16|Satd16x8|Satd8x16|Satd8x8', 3),
 (r'WelsMdI4x4|WelsMdI16x16|WelsMdIntra|Combined3|SatdThree|Satd4x4|LumaPred|IChromaPred|PredIntra4x4|FillNeighborCacheIntra|MdIntraInit|I16x16Pred|ChromaPred', 5),
 (r'WelsMd|PredMv|PredSkip|PSkip|Pskip|FillNeighborCache|TryP|Partition|JudgeBGD|BackgroundMbEnc|UpdateMbMv|MdInterInit|CheckFrame', 4),
 (r'McChroma|McCopy|McLuma|McFunc|WelsMcCopy', 6),
 (r'IDct|Rec(?!ord)|Copy4x4|Copy8x8|Copy16x16|Dequant|IHadamard', 8),
 (r'Dct|Quant|CalculateSingleCtr|Scan4x4|GetNoneZero|Hadamard|EncInterY|EncRecUV|ScanDc', 7),
 (r'Cavlc|WriteMb|WriteBlock|SpatialWrite|BsWrite|BitWrite|WelsWrite|Nal|Ue|Se|Bs[A-Z]|WriteSlice|WriteRef', 9),
 (r'Deblock|Filtering', 10),
]
NC = [
 (r'read_frame|extend_borders|copy_from|yuv::|padded_view|FrameSink|ivf|write_frame', 0),
 (r'preanalysis|sad_quads|moments_8x8|downscale_2x|noise::|pyramid', 1),
 (r'nc_analysis::motion|sad_grid|sad_scan|keep_min|search_vector|global::|grid_costs|nc_analysis', 2),
 (r'sub_pixel_variance|refine_subpel', 3),
 (r'intra4x4|predict_intra|predict::|choose_intra|ChromaMode|Intra16|intra16|predict::plane', 5),
 (r'decision::|Measurer|source_variance|partition_plan|predict_mv|neighbour_at|predicted_vectors|mode_bits|QuadrantCosts|sub_block_vectors|proven_zero|zero_sad_threshold|choose_inter|partition::|Neighbours|variance_w|mbinfo|mvpred', 4),
 (r'mc::|h264_luma|h264_chroma|Dsp>::predict$|predict_inter|copy_prediction', 6),
 (r'inverse|dequantize|add_residual', 8),
 (r'transform_macroblock|transform_chroma|forward|quantize|residual_of|Dsp>::residual$|residual_w|coefficient_cost|quantize_dc', 7),
 (r'cabac|syntax::|residual::write|write_nal|bitwriter|BitWriter|nal::', 9),
 (r'deblock|loop_filter_frame', 10),
 # Top-level drivers: their own (inlined) self cost is split by source file
 # below; kernels they call directly count as mode decision.
 (r'CodecBackend>::encode_block|>::code_node|choose_|encode_inter|encode_intra', 4),
 (r'nanocodec::(encode|main)$|Encoder<.*>::encode', 11),
]
FILE_RULES = [(r'intra4x4|predict\.rs', 5), (r'syntax\.rs|cabac|bitwriter|residual\.rs', 9),
              (r'mvpred|partition|decision|mbinfo', 4), (r'macroblock\.rs|codec_txfm', 7),
              (r'motion\.rs|grid\.rs|global\.rs', 2), (r'preanalysis|noise|pyramid', 1), (r'deblock', 10)]

def cat_of(name, rules):
    for pat, c in rules:
        if re.search(pat, name):
            return c
    return None

def run(path, symfile, rules, use_files=False):
    names, selfc, edges = cgparse.parse(path, symfile)
    callers = collections.defaultdict(list)
    calls_in = collections.Counter()
    for (a, b), (n, inc) in edges.items():
        callers[b].append((a, inc)); calls_in[b] += n
    memo = {}
    def cats(fn, depth=0):
        # distribution over categories for fn's self cost
        if fn in memo: return memo[fn]
        c = cat_of(names.get(fn, '?'), rules)
        if c is not None: d = {c: 1.0}
        elif depth > 6 or not callers[fn]: d = {11: 1.0}
        else:
            tot = sum(i for _, i in callers[fn]) or 1
            d = collections.Counter()
            for a, i in callers[fn]:
                for k, v in cats(a, depth + 1).items(): d[k] += v * i / tot
        memo[fn] = d
        return d
    out = collections.Counter()
    if use_files:
        n2, files, sc2 = cgparse.parse_files(path)
        for (fn, fi), c in sc2.items():
            nm = n2.get(fn, '?')
            if re.search(r'encode_block|code_node|analyze_written', nm):
                k = cat_of(files.get(fi, ''), FILE_RULES)
                if k is None: k = 4 if 'code_node' in nm else (2 if 'analyze' in nm else 11)
                out[k] += c
            else:
                for k, v in cats(fn).items(): out[k] += c * v
    else:
        for fn, c in selfc.items():
            for k, v in cats(fn).items(): out[k] += c * v
    return out, names, calls_in, selfc

oh, ohn, ohcalls, ohself = run(sys.argv[1], sys.argv[2], OH)
nc, ncn, nccalls, ncself = run(sys.argv[3], None, NC, use_files=True)
print('| Category | openh264-baseline | nanocodec-h264 | ratio |')
print('|---|---|---|---|')
for i, c in enumerate(CATS):
    a, b = oh[i] / FR / 1e6, nc[i] / FR / 1e6
    print(f'| {c} | {a:.1f} | {b:.1f} | {"%.1f×" % (b/a) if a > 0.05 else "—"} |')
print(f'| **total** | **{sum(oh.values())/FR/1e6:.1f}** | **{sum(nc.values())/FR/1e6:.1f}** | **{sum(nc.values())/sum(oh.values()):.1f}×** |')

def counts(names, calls, pats):
    for label, pat in pats:
        n = sum(c for f, c in calls.items() if re.search(pat, names.get(f, '')))
        print(f'{label}: {n/FR:.0f}/frame = {n/FR/3600:.2f}/MB')
print('\n--- openh264 counts')
counts(ohn, ohcalls, [
 ('SAD 16x16', r'^WelsSampleSad16x16'), ('SAD four 16x16 (x4 positions)', r'SadFour16x16'),
 ('SAD 8x8', r'^WelsSampleSad8x8'), ('SAD four 8x8/8x16/16x8', r'SadFour(8x8|8x16|16x8)'),
 ('SATD 16x16', r'Satd16x16'), ('SATD 8x8/16x8/8x16', r'Satd(8x8|16x8|8x16)'), ('SATD 4x4', r'Satd4x4'), ('SATD three 4x4', r'SatdThree'),
 ('I16 combined3', r'Intra16x16Combined3'), ('chroma combined3', r'Chroma8x8Combined3'),
 ('MC chroma', r'McChroma'), ('MC copy', r'McCopy'), ('half-pel filter calls', r'McHorVer'), ('PixelAvg', r'PixelAvg'),
 ('DCT four 4x4', r'DctFourT4'), ('DCT 4x4', r'DctT4'), ('quant four 4x4', r'QuantFour'), ('IDCT four rec', r'IDctFourT4Rec'),
 ('CAVLC blocks written', r'WriteBlockResidualCavlc'), ('MeRefineFracPixel', r'MeRefineFracPixel'), ('diamond search', r'WelsDiamondSearch'),
 ('MotionEstimateSearch', r'WelsMotionEstimateSearch'), ('PSkip enc tries', r'WelsMdPSkipEnc'), ('I4x4 MD', r'WelsMdI4x4\('), ('I16 MD', r'WelsMdI16x16\('),
])
print('\n--- nanocodec counts')
counts(ncn, nccalls, [
 ('sub-pixel variance', r'sub_pixel_variance_w'), ('variance (whole-sample)', r'avx2::variance_w'),
 ('Measurer::measure', r'Measurer>::measure'), ('analysis grid positions (sad_grid_best_64x64)', r'avx2::sad_grid_best'),
 ('analysis scan rows', r'avx2::sad_scan_16x16'), ('pre-analysis sad_quads rows', r'avx2::sad_quads_8x8_row'),
 ('SATD 8x8 tiles calls', r'avx2::satd_8x8_tiles'), ('SATD 4x4 tiles calls', r'avx2::satd_4x4_tiles'),
 ('MC luma calls', r'avx2::h264_luma_w'), ('MC chroma calls', r'avx2::h264_chroma_w'),
 ('fwd 4x4', r'avx2::h264_forward_4x4'), ('fwd 8x8', r'avx2::h264_forward_8x8'), ('quantize', r'avx2::quantize$'),
 ('inverse 4x4', r'avx2::h264_inverse_4x4'), ('inverse 8x8', r'avx2::h264_inverse_8x8'),
 ('CABAC decision bins', r'CabacEncoder>::encode_decision'), ('CABAC bypass bins', r'CabacEncoder>::encode_bypass'),
 ('encode_block', r'CodecBackend>::encode_block'), ('intra4x4 coding', r'code_intra_4x4'), ('deblock luma calls', r'deblock::avx2::luma'),
])
