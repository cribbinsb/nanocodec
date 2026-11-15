#!/usr/bin/env python3
"""Build and run the foreign 8-bit VVC decoder corpus (VVenC and VTM streams).

The official suite has only nine 8-bit streams, so most Main-profile tools are
exercised only by 10-bit vectors our decoder rejects. This tool encodes short
8-bit clips with VVenC and VTM (CTC configurations plus one stream per tool),
records per-picture MD5s from the VTM reference decoder (with VVdeC and ffmpeg
as cross-checks), and compares our decoder with them. Palette mode has no stream:
H.266 A.3.1 requires sps_palette_enabled_flag = 0 in the Main 10 profiles (VTM's
decoder rejects such a stream under that profile). Decoded pixels stay in pipes; only bitstreams and logs are kept, under
`~/nanocodec-data/work/vvc-foreign`.

    vvc_foreign_corpus.py build [--only NAME,...] [--jobs N]   # encode + reference hashes
    vvc_foreign_corpus.py run --bin .../nanocodec [--json out]  # our decoder, table by tool
    vvc_foreign_corpus.py pin --bin .../nanocodec               # check in the passing set's hashes
    vvc_foreign_corpus.py check --bin .../nanocodec             # regression: the pinned set still passes

`build` is deterministic for a given VTM/VVenC build; `check` refuses a stream
whose SHA-256 differs from the pinned one instead of comparing against it.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
import time
import threading
from pathlib import Path

HOME = Path.home()
DATA = HOME / 'nanocodec-data'
WORK = DATA / 'work/vvc-foreign'
SRC = DATA / 'work/standards-eval/src'
VTM = SRC / 'VTM/bin/EncoderAppStatic'
VTM_CFG = SRC / 'VTM/cfg'
VVENC = SRC / 'vvenc/bin/release-static/vvencFFapp'
VVENC_CFG = SRC / 'vvenc/cfg'
VTM_DECODER = SRC / 'VTM/bin/DecoderAppStatic'
VVDEC = DATA / 'anchors/x86_64/bin/vvdecapp'
FFMPEG = DATA / 'anchors/x86_64/bin/ffmpeg-decode'
PINNED = Path(__file__).resolve().parents[1] / 'ci/vvc_foreign_main8.json'

# Cached 8-bit 4:2:0 clips cut from /mldata (name, width, height, fps, source kbps).
CLIPS = {
    'bc1': ('bodycam_bc1', 1280, 720, 30, 1420.6),
    'jaad': ('pan_jaad_0117', 1280, 720, 30, 4754.3),
    'mot': ('crowd_mot17_09', 1280, 720, 30, 2998.0),
}
SOURCE_FRAMES = 33

# VTM switches that turn every optional coding tool off: the per-tool streams
# enable exactly one on top of this, so a failure names its tool.
VTM_MINIMAL = (
    'MTS=0 MTSImplicit=0 SBT=0 ISP=0 MMVD=0 Affine=0 SbTMVP=0 LMChroma=0 DepQuant=0 IMV=0 '
    'ALF=0 CCALF=0 BCW=0 CIIP=0 Geo=0 IBC=0 LMCSEnable=0 MRL=0 MIP=0 JointCbCr=0 PROF=0 '
    'ChromaTS=0 LFNST=0 BIO=0 DMVR=0 SMVD=0 SAO=0 TransformSkip=0 BDPCM=0 SignHideFlag=0 '
    'AffineAmvr=0 AllowDisFracMMVD=0')

BASES = {
    'ai': 'encoder_intra_vtm.cfg',
    'ldp': 'encoder_lowdelay_P_vtm.cfg',
    'ldb': 'encoder_lowdelay_vtm.cfg',
    'ra': 'encoder_randomaccess_vtm.cfg',
}


def vtm(name, base, tools, options='', clip='bc1', size=(640, 360), frames=9, qp=37, minimal=True):
    return dict(name=name, encoder='vtm', base=base, tools=tools, options=options, clip=clip,
                size=size, frames=frames, qp=qp, minimal=minimal)


def vvenc(name, cfg, tools, options='', clip='jaad', size=(640, 360), frames=17, qp=37):
    return dict(name=name, encoder='vvenc', base=cfg, tools=tools, options=options, clip=clip,
                size=size, frames=frames, qp=qp, minimal=False)


def streams():
    out = []
    # CTC configurations, every default tool on.
    out += [
        vtm('vtm-ctc-ai', 'ai', 'CTC all-intra', clip='jaad', frames=8, qp=42, minimal=False),
        vtm('vtm-ctc-ldp', 'ldp', 'CTC low-delay P', frames=17, minimal=False),
        vtm('vtm-ctc-ldb', 'ldb', 'CTC low-delay B', frames=17, minimal=False),
        vtm('vtm-ctc-ra', 'ra', 'CTC random access', clip='jaad', frames=33, minimal=False),
        vtm('vtm-ctc-ra-720p', 'ra', 'CTC random access', clip='mot', size=(1280, 720), frames=17,
            minimal=False),
        vtm('vtm-ctc-ldb-720p', 'ldb', 'CTC low-delay B', clip='jaad', size=(1280, 720), frames=9,
            minimal=False),
    ]
    # Minimal baselines: partitioning, residual coding, deblocking only.
    out += [
        vtm('vtm-min-ai', 'ai', 'baseline intra (MTT, dual tree)', clip='jaad', frames=4, qp=42),
        vtm('vtm-min-ldp', 'ldp', 'baseline P (MTT)'),
        vtm('vtm-min-ldb', 'ldb', 'baseline low-delay B'),
        vtm('vtm-min-ra', 'ra', 'baseline random access (reordering)', frames=17),
        vtm('vtm-qt-only', 'ldp', 'quadtree only (no MTT)',
            'MaxMTTHierarchyDepth=0 MaxMTTHierarchyDepthISliceL=0 MaxMTTHierarchyDepthISliceC=0 '
            'MaxBTLumaISlice=8 MaxTTLumaISlice=8 MaxBTChromaISlice=8 MaxTTChromaISlice=8 '
            'MaxBTNonISlice=8 MaxTTNonISlice=8'),
        vtm('vtm-single-tree', 'ldp', 'single tree intra', 'DualITree=0'),
        vtm('vtm-ctu32', 'ldp', 'CTU 32', 'CTUSize=32 MaxCUWidth=32 MaxCUHeight=32 Log2MaxTbSize=5 MaxBTLumaISlice=32 MaxBTChromaISlice=32 '
            'MaxBTNonISlice=32 MaxTTLumaISlice=32 MaxTTChromaISlice=32 MaxTTNonISlice=32'),
        vtm('vtm-ctu64', 'ldp', 'CTU 64', 'CTUSize=64 MaxBTNonISlice=64'),
        vtm('vtm-tb32', 'ldp', 'max transform 32', 'Log2MaxTbSize=5'),
        vtm('vtm-400', 'ldb', '4:0:0', 'ChromaFormatIDC=400'),
    ]
    # One tool each over the minimal baseline.
    intra = [
        ('isp', 'ISP', 'ISP=1'), ('mip', 'MIP', 'MIP=1'), ('mrl', 'MRL', 'MRL=1'),
        ('cclm', 'CCLM', 'LMChroma=1'), ('lfnst', 'LFNST', 'LFNST=1'),
        ('mts', 'MTS explicit', 'MTS=3'), ('mts-implicit', 'MTS implicit', 'MTSImplicit=1'),
        ('depquant', 'dependent quantisation', 'DepQuant=1'),
        ('signhide', 'sign data hiding', 'SignHideFlag=1'), ('jccr', 'JCCR', 'JointCbCr=1'),
        ('lmcs', 'LMCS', 'LMCSEnable=1'), ('alf', 'ALF', 'ALF=1'),
        ('ccalf', 'ALF + CCALF', 'ALF=1 CCALF=1'), ('sao', 'SAO', 'SAO=1'),
        ('ts', 'transform skip', 'TransformSkip=1 ChromaTS=1'),
        ('bdpcm', 'BDPCM', 'TransformSkip=1 ChromaTS=1 BDPCM=1'),
        ('ibc', 'IBC', 'IBC=1'),
        ('scaling-default', 'scaling lists (default)', 'ScalingList=1'),
        ('scaling-file', 'scaling lists (explicit)',
         f'ScalingList=2 ScalingListFile={VTM_CFG}/scaling_list/sample_scaling_list_file.txt'),
    ]
    for tag, tools, options in intra:
        out.append(vtm(f'vtm-ai-{tag}', 'ai', tools, options, clip='jaad', frames=3, qp=42))
        out.append(vtm(f'vtm-ldp-{tag}', 'ldp', tools, options))
    inter = [
        ('ldp', 'sbt', 'SBT', 'SBT=1'), ('ldp', 'affine', 'affine', 'Affine=1'),
        ('ldp', 'affine-prof', 'affine + PROF', 'Affine=1 PROF=1'),
        ('ldp', 'sbtmvp', 'SbTMVP', 'SbTMVP=1'), ('ldp', 'mmvd', 'MMVD', 'MMVD=1'),
        ('ldp', 'mmvd-fpel', 'MMVD full-pel', 'MMVD=1 AllowDisFracMMVD=1'),
        ('ldp', 'ciip', 'CIIP', 'CIIP=1'), ('ldp', 'amvr', 'AMVR', 'IMV=1'),
        ('ldp', 'affine-amvr', 'affine AMVR', 'Affine=1 IMV=1 AffineAmvr=1'),
        ('ldp', 'tmvp-off', 'no temporal MVP', 'TMVPMode=0'),
        ('ldp', 'wp', 'weighted prediction P', 'WeightedPredP=1 WeightedPredMethod=4'),
        ('ldp', 'dbk-off', 'deblocking off', 'DeblockingFilterDisable=1'),
        ('ldp', 'cu-dqp', 'CU QP delta', 'MaxDeltaQP=3 MaxCuDQPSubdiv=2'),
        ('ldp', 'chroma-qp', 'PPS and slice chroma QP offsets',
         'CbQpOffset=2 CrQpOffset=-3 SliceChromaQPOffsetPeriodicity=1 '
         'SliceCbQpOffsetIntraOrPeriodic=1 SliceCrQpOffsetIntraOrPeriodic=-1'),
        ('ldp', 'cu-chroma-qp', 'CU chroma QP offsets',
         'CbQpOffsetList=2,-2 CrQpOffsetList=-2,2 CbCrQpOffsetList=1,-1 MaxCuChromaQpOffsetSubdiv=2 '
         'SliceCuChromaQpOffsetEnabled=1 JointCbCr=1'),
        ('ldp', 'ladf', 'LADF',
         'LADF=1 LadfNumIntervals=3 LadfQpOffset=2,0,-2 LadfIntervalLowerBound=70,140'),
        ('ldp', 'ph-in-sh-off', 'separate picture header', 'EnablePictureHeaderInSliceHeader=0'),
        ('ldp', 'aud', 'AUD + hash SEI', 'AccessUnitDelimiter=1 SEIDecodedPictureHash=1'),
        ('ldp', 'wpp', 'WPP', 'WaveFrontSynchro=1'),
        ('ldp', 'tiles', 'tiles 2x2', 'EnablePicPartitioning=1 TileColumnWidthArray=3 TileRowHeightArray=2'),
        ('ldp', 'tiles-nolf', 'tiles, no cross-tile filtering',
         'EnablePicPartitioning=1 TileColumnWidthArray=3 TileRowHeightArray=2 DisableLoopFilterAcrossTiles=1 '
         'SAO=1 ALF=1'),
        ('ldp', 'rect-slices', 'rectangular slices',
         'EnablePicPartitioning=1 TileColumnWidthArray=3 TileRowHeightArray=2 RasterScanSlices=0 '
         'RectSliceFixedWidth=1 RectSliceFixedHeight=1'),
        ('ldp', 'rect-slices-nolf', 'rectangular slices, no cross-slice filtering',
         'EnablePicPartitioning=1 TileColumnWidthArray=3 TileRowHeightArray=2 RasterScanSlices=0 '
         'RectSliceFixedWidth=1 RectSliceFixedHeight=1 DisableLoopFilterAcrossSlices=1 SAO=1 ALF=1'),
        ('ldp', 'raster-slices', 'raster-scan slices',
         'EnablePicPartitioning=1 TileColumnWidthArray=1 TileRowHeightArray=1 RasterScanSlices=1 '
         'RasterSliceSizes=4'),
        ('ldp', 'subpics', 'subpictures',
         'EnablePicPartitioning=1 TileColumnWidthArray=5 TileRowHeightArray=2 RasterScanSlices=0 '
         'RectSliceFixedWidth=1 RectSliceFixedHeight=1 SubPicInfoPresentFlag=1 NumSubPics=2 '
         'SubPicCtuTopLeftX=0,0 SubPicCtuTopLeftY=0,2 SubPicWidth=5,5 SubPicHeight=2,1 '
         'SubPicTreatedAsPicFlag=1,1 LoopFilterAcrossSubpicEnabledFlag=0,0'),
        ('ldp', 'rpr', 'RPR 2x',
         'ScalingRatioHor=2.0 ScalingRatioVer=2.0 SwitchPocPeriod=4'),
        ('ldp', 'rpr-1p5', 'RPR 1.5x',
         'ScalingRatioHor=1.5 ScalingRatioVer=1.5 SwitchPocPeriod=4'),
        ('ldp', 'wrap', 'wrap-around', 'WrapAround=1 WrapAroundOffset=640'),
        ('ldp', 'virtual-boundaries', 'virtual boundaries',
         'VirtualBoundariesPresentInSPSFlag=1 NumVerVirtualBoundaries=1 NumHorVirtualBoundaries=1 '
         'VirtualBoundariesPosX=320 VirtualBoundariesPosY=184 SAO=1 ALF=1'),
        ('ldp', 'gdr', 'GDR', 'GdrEnabled=1 GdrPocStart=2 GdrPeriod=8 GdrInterval=4'),
        ('ldb', 'geo', 'GPM', 'Geo=1'), ('ldb', 'bcw', 'BCW', 'BCW=1'),
        ('ldb', 'wp-b', 'weighted prediction B', 'WeightedPredB=1 WeightedPredMethod=4'),
        ('ldb', 'ciip-b', 'CIIP (B)', 'CIIP=1'), ('ldb', 'mmvd-b', 'MMVD (B)', 'MMVD=1'),
        ('ldb', 'affine-b', 'affine (B)', 'Affine=1 PROF=1'),
        ('ldb', 'sbtmvp-b', 'SbTMVP (B)', 'SbTMVP=1'),
        ('ra', 'bdof', 'BDOF', 'BIO=1'), ('ra', 'dmvr', 'DMVR', 'DMVR=1'),
        ('ra', 'smvd', 'SMVD', 'SMVD=1'), ('ra', 'bcw-ra', 'BCW (RA)', 'BCW=1'),
        ('ra', 'ltr', 'composite reference config (no LTR signalled)', 'CompositeLTReference=1'),
        ('ra', 'lmcs-ra', 'LMCS (RA)', 'LMCSEnable=1'),
        ('ra', 'alf-ra', 'ALF + CCALF (RA)', 'ALF=1 CCALF=1'),
    ]
    for base, tag, tools, options in inter:
        frames = 17 if base == 'ra' or tag in ('gdr', 'rpr', 'rpr-1p5') else 9
        out.append(vtm(f'vtm-{base}-{tag}' if not tag.endswith(('-b', '-ra')) else f'vtm-{tag}',
                       base, tools, options, frames=frames))
    # Quantisation groups in the dual tree and at several depths.
    out += [
        vtm('vtm-ai-cu-dqp', 'ai', 'CU QP delta (dual tree)', 'MaxDeltaQP=3 MaxCuDQPSubdiv=2',
            clip='jaad', frames=3, qp=42),
        vtm('vtm-ai-cu-dqp-deep', 'ai', 'CU QP delta (dual tree, subdiv 6)',
            'MaxDeltaQP=4 MaxCuDQPSubdiv=6', clip='jaad', frames=3, qp=42),
        vtm('vtm-ai-cu-chroma-qp', 'ai', 'CU chroma QP offsets (dual tree)',
            'CbQpOffsetList=2,-2 CrQpOffsetList=-2,2 CbCrQpOffsetList=1,-1 MaxCuChromaQpOffsetSubdiv=2 '
            'SliceCuChromaQpOffsetEnabled=1 JointCbCr=1', clip='jaad', frames=3, qp=42),
        vtm('vtm-ldb-cu-dqp-deep', 'ldb', 'CU QP delta (subdiv 4, all tools)',
            'MaxDeltaQP=4 MaxCuDQPSubdiv=4', minimal=False),
        vtm('vtm-single-tree-cu-dqp', 'ldp', 'CU QP delta (single tree intra)',
            'DualITree=0 MaxDeltaQP=3 MaxCuDQPSubdiv=4'),
    ]
    # VVenC presets: random access, low delay, all intra.
    for preset in ('faster', 'fast', 'medium', 'slow'):
        out.append(vvenc(f'vvenc-ra-{preset}', f'randomaccess_{preset}.cfg', f'VVenC {preset} random access',
                         '--IntraPeriod 16 --GOPSize 16 --DecodingRefreshType idr'))
        out.append(vvenc(f'vvenc-ld-{preset}', f'experimental/lowdelay_{preset}.cfg',
                         f'VVenC {preset} low delay', clip='bc1'))
        out.append(vvenc(f'vvenc-ai-{preset}', f'randomaccess_{preset}.cfg', f'VVenC {preset} all intra',
                         '--IntraPeriod 1 --GOPSize 1 --MCTF 0 --BIM 0', frames=4, qp=42))
    out.append(vvenc('vvenc-ra-medium-720p', 'randomaccess_medium.cfg', 'VVenC medium random access',
                     '--IntraPeriod 32 --GOPSize 32', clip='mot', size=(1280, 720), frames=33))
    return out


def source(spec):
    """Scaled 8-bit 4:2:0 source for a stream; made once, shared by all streams."""
    name, width, height, _, _ = CLIPS[spec['clip']]
    out_w, out_h = spec['size']
    path = WORK / 'src' / f'{spec["clip"]}_{out_w}x{out_h}.yuv'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix('.tmp')
        subprocess.run(
            ['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-s',
             f'{width}x{height}', '-i', str(DATA / 'clips/main' / f'{name}.yuv'), '-vf',
             f'scale={out_w}:{out_h}', '-frames:v', str(SOURCE_FRAMES), '-f', 'rawvideo', str(temp)],
            check=True)
        temp.rename(path)
    return path


def encode_command(spec, stream):
    width, height = spec['size']
    fps = CLIPS[spec['clip']][3]
    if spec['encoder'] == 'vtm':
        command = [str(VTM), '-c', str(VTM_CFG / BASES[spec['base']]), '-i', str(source(spec)),
                   '-b', str(stream), '-o', '', '-wdt', str(width), '-hgt', str(height), '-fr', str(fps),
                   '-f', str(spec['frames']), '-q', str(spec['qp']), '--InputBitDepth=8',
                   '--InternalBitDepth=8', '--OutputBitDepth=8', '--TemporalSubsampleRatio=1',
                   '--TemporalFilter=0', '--Level=4.1']
        tokens = (VTM_MINIMAL.split() if spec['minimal'] else []) + shlex.split(spec['options'])
        return command + [f'--{token}' for token in tokens]
    return [str(VVENC), '-c', str(VVENC_CFG / spec['base']), '-i', str(source(spec)), '-b', str(stream),
            '-s', f'{width}x{height}', '-fr', str(fps), '-f', str(spec['frames']), '-q', str(spec['qp']),
            '--InputBitDepth', '8', '--InternalBitDepth', '8', '-t', '1'] + shlex.split(spec['options'])


def frame_hashes(command, frame_bytes, fifo=None, timeout=600):
    """Per-picture MD5s of a decoder's raw output, read from its stdout or a FIFO."""
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL if fifo else subprocess.PIPE,
                               stderr=subprocess.PIPE)
    errors = bytearray()

    def drain():
        for chunk in iter(lambda: process.stderr.read1(65536), b''):
            errors.extend(chunk)
            del errors[:-4096]
    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()
    timer = threading.Timer(timeout, process.kill)
    timer.start()
    hashes, whole, tail = [], hashlib.md5(), 0
    try:
        # A decoder that fails before opening the FIFO must not block the reader.
        pixels = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK) if fifo else process.stdout.fileno()
        pending = bytearray()
        while True:
            try:
                chunk = os.read(pixels, 1 << 20)
            except BlockingIOError:
                chunk = None
            if chunk:
                pending.extend(chunk)
                while len(pending) >= frame_bytes:
                    frame = bytes(pending[:frame_bytes])
                    del pending[:frame_bytes]
                    whole.update(frame)
                    hashes.append(hashlib.md5(frame).hexdigest())
            elif not fifo and chunk == b'':
                break
            elif fifo and process.poll() is not None:
                if chunk != b'':
                    continue
                break
            else:
                time.sleep(0.002)
        if fifo:
            os.close(pixels)
        whole.update(bytes(pending))
        tail = len(pending)
        code = process.wait()
    finally:
        timer.cancel()
        if process.poll() is None:
            process.kill()
            process.wait()
    drainer.join(5)
    return dict(code=code, frames=hashes, md5=whole.hexdigest(), tail=tail,
                error=errors.decode(errors='replace').strip()[-600:])


def reference(stream, frame_bytes):
    """The VTM reference decoder's pictures (cropped, at their coded size)."""
    with tempfile.TemporaryDirectory(dir=WORK / 'tmp') as folder:
        fifo = os.path.join(folder, 'out.yuv')
        os.mkfifo(fifo)
        return frame_hashes([str(VTM_DECODER), '-b', str(stream), '-o', fifo, '-d', '8'], frame_bytes, fifo)


def vvdec_reference(stream, frame_bytes):
    with tempfile.TemporaryDirectory(dir=WORK / 'tmp') as folder:
        fifo = os.path.join(folder, 'out.yuv')
        os.mkfifo(fifo)
        return frame_hashes([str(VVDEC), '-b', str(stream), '-o', fifo, '-t', '1', '-v', '1'],
                            frame_bytes, fifo)


def agreement(other, main):
    if other['code'] == 0 and other['md5'] == main['md5']:
        return 'agrees'
    return f'differs ({len(other["frames"])} pictures, exit {other["code"]}) {other["error"][-160:]}'


def second_reference(stream, frame_bytes, monochrome):
    return frame_hashes([str(FFMPEG), '-v', 'error', '-i', str(stream), '-f', 'rawvideo', '-pix_fmt',
                         'gray' if monochrome else 'yuv420p', '-'], frame_bytes)


def build_one(spec):
    stream = WORK / 'streams' / f'{spec["name"]}.266'
    log = WORK / 'logs' / f'{spec["name"]}.log'
    command = encode_command(spec, stream)
    if not stream.exists():
        temp = stream.with_suffix('.tmp')
        with open(log, 'w') as sink:
            sink.write(' '.join(shlex.quote(part) for part in command) + '\n')
            sink.flush()
            patched = [str(temp) if part == str(stream) else part for part in command]
            code = subprocess.run(patched, stdout=sink, stderr=subprocess.STDOUT).returncode
        if code or not temp.exists() or temp.stat().st_size == 0:
            temp.unlink(missing_ok=True)
            return dict(name=spec['name'], status='encode-failed', log=str(log))
        temp.rename(stream)
    width, height = spec['size']
    monochrome = 'ChromaFormatIDC=400' in spec['options']
    frame_bytes = width * height if monochrome else width * height * 3 // 2
    main = reference(stream, frame_bytes)
    vvdec = vvdec_reference(stream, frame_bytes)
    other = second_reference(stream, frame_bytes, monochrome)
    data = stream.read_bytes()
    clip = CLIPS[spec['clip']]
    kbps = len(data) * 8 * clip[3] / spec['frames'] / 1000
    return dict(
        name=spec['name'], status='ok' if main['code'] == 0 and main['frames'] else 'reference-failed',
        encoder=spec['encoder'], tools=spec['tools'], width=width, height=height, monochrome=monochrome,
        coded_frames=spec['frames'], bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
        kbps=round(kbps, 1), source_kbps=clip[4], within_half_source=kbps <= clip[4] / 2,
        frames=main['frames'], md5=main['md5'], tail=main['tail'], reference_error=main['error'],
        vvdec=agreement(vvdec, main), ffmpeg=agreement(other, main),
        command=' '.join(shlex.quote(part) for part in command))


def load_manifest():
    path = WORK / 'manifest.json'
    return json.loads(path.read_text()) if path.exists() else {}


def build(args):
    for folder in ('streams', 'logs', 'tmp', 'src'):
        (WORK / folder).mkdir(parents=True, exist_ok=True)
    specs = streams()
    if args.only:
        specs = [spec for spec in specs if spec['name'] in args.only.split(',')]
    for spec in specs:
        source(spec)
    manifest = load_manifest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for row in pool.map(build_one, specs):
            manifest[row['name']] = row
            note = '' if row['status'] != 'ok' else (
                f'{len(row["frames"])} pictures, {row["kbps"]} kbps'
                f'{"" if row["within_half_source"] else " OVER HALF SOURCE"}, VVdeC {row["vvdec"]}, '
                f'ffmpeg {row["ffmpeg"]}')
            print(f'{row["status"]:16} {row["name"]}: {note}', flush=True)
    (WORK / 'manifest.json').write_text(json.dumps(manifest, indent=1, sort_keys=True) + '\n')


def run_one(binary, row, timeout, whole=True):
    stream = WORK / 'streams' / f'{row["name"]}.266'
    frame_bytes = row['width'] * row['height'] * (2 if row['monochrome'] else 3) // 2
    ours = frame_hashes([str(binary), 'decode', '--codec', 'vvc', '--input', str(stream), '--output',
                         '/dev/stdout'], frame_bytes, timeout=timeout)
    expected = row['frames']
    first = next((index for index, (a, b) in enumerate(zip(ours['frames'], expected)) if a != b), None)
    if ours['code'] != 0:
        error = ours['error'].splitlines()[-1] if ours['error'] else f'exit {ours["code"]}'
        return f'rejected after {len(ours["frames"])} pictures: {error}'
    if first is not None:
        return f'mismatch at picture {first} of {len(expected)}'
    if len(ours['frames']) != len(expected) or ours['tail'] != row['tail'] or (
            whole and ours['md5'] != row['md5']):
        return f'output size differs after {len(ours["frames"])} of {len(expected)} pictures'
    return 'bit-exact'


def run(args):
    manifest = {name: row for name, row in load_manifest().items() if row['status'] == 'ok'}
    if args.only:
        manifest = {name: manifest[name] for name in args.only.split(',')}
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {name: pool.submit(run_one, args.bin, row, args.timeout) for name, row in manifest.items()}
        for name, future in pending.items():
            results[name] = future.result()
    width = max(map(len, results))
    for name in sorted(results, key=lambda name: (results[name] != 'bit-exact', results[name][:12], name)):
        print(f'{name:{width}}  {manifest[name]["tools"]:42.42}  {results[name]}')
    passed = sum(result == 'bit-exact' for result in results.values())
    print(f'foreign 8-bit VVC corpus: {passed}/{len(results)} bit-exact against the VTM decoder')
    if args.json:
        args.json.write_text(json.dumps(results, indent=1, sort_keys=True) + '\n')
    return results, manifest


def pin(args):
    """Check in the hashes of every stream our decoder now passes (regression set)."""
    results, manifest = run(args)
    keep = ('encoder', 'tools', 'width', 'height', 'monochrome', 'sha256', 'frames', 'tail', 'command')
    pinned = [dict(name=name, **{key: manifest[name][key] for key in keep})
              for name in sorted(results) if results[name] == 'bit-exact']
    PINNED.write_text(json.dumps(pinned, indent=1) + '\n')
    print(f'pinned {len(pinned)} streams in {PINNED}')


def check(args):
    """Regression gate: every pinned stream is present, unchanged and still bit-exact."""
    pinned = json.loads(PINNED.read_text())
    if args.only:
        pinned = [row for row in pinned if row['name'] in args.only.split(',')]

    def one(row):
        stream = WORK / 'streams' / f'{row["name"]}.266'
        if not stream.exists():
            return 'missing stream (run the build action)'
        if hashlib.sha256(stream.read_bytes()).hexdigest() != row['sha256']:
            return 'stream differs from the pinned SHA-256'
        return run_one(args.bin, dict(row, md5=None), args.timeout, whole=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = dict(zip((row['name'] for row in pinned), pool.map(one, pinned)))
    for name, result in results.items():
        if result != 'bit-exact':
            print(f'{name}: {result}')
    passed = sum(result == 'bit-exact' for result in results.values())
    print(f'pinned foreign 8-bit VVC streams: {passed}/{len(results)} bit-exact')
    return int(passed != len(results) or not results)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=('build', 'run', 'pin', 'check'))
    parser.add_argument('--bin', type=Path)
    parser.add_argument('--only', default='')
    parser.add_argument('--jobs', type=int, default=12)
    parser.add_argument('--timeout', type=int, default=300)
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if args.action == 'build':
        build(args)
        return 0
    elif args.bin is None:
        parser.error('--bin is required')
    elif args.action == 'run':
        run(args)
    elif args.action == 'pin':
        pin(args)
    else:
        return check(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
