#!/usr/bin/env python3
"""Block-by-block comparison of two HEVC streams of one source (ours against a reference encoder).

    tools/dev/hevc_block_compare.py --ours A.hevc --theirs B.hevc --source CLIP --width W --height H \
        [--first 1] [--last N] [--analysis-bin DIR]

Both streams are decoded by our analysis decoder (`nanocodec stat --codec hevc --records`, an
analysis build: docs/PROCEDURES.md "Inspect a stream") and laid on an 8x8 luma grid per picture;
the decoded pictures and the source come from ffmpeg pipes (no YUV files). Printed per stream, over
the pictures FIRST..LAST (P pictures of a one-key stream): area, bits and luma squared error by
prediction class (intra, AMVP, skip, merge with residual), vector precision, CU and transform sizes;
then the cross table (our class by theirs) with each cell's bits, error and its share of the luma
cost gap `SSE + lambda * bits` at HM's lambda for the reference's mean QP. Code both streams at the
same flat QP (`--param rate.cadence_fixed_qp=0` for ours) for the table to compare like with like.
Numbers only. Example and findings: docs/handoff/hevc.md "VC-1K: block-by-block diagnosis".
"""
import argparse
import os
import subprocess
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))
from nceval import tempdirs  # noqa: E402

INTRA, AMVP, SKIP, MERGE = 0, 1, 2, 3
NAMES = ['intra', 'amvp', 'skip', 'merge']
#: Halfwords per packed record (nc_common::stream_stats::PACKED_HALFWORDS).
HALFWORDS = 24
#: HM's lambda at QP 12 in squared error per bit; it doubles every 3 QP.
LAMBDA_AT_QP12 = 0.57


def frames_of(path):
    """The packed records of every picture of a `stat --records` file."""
    raw = open(path, 'rb').read()
    out, pos = [], 0
    while pos < len(raw):
        n = int(np.frombuffer(raw, '<u4', 1, pos)[0])
        pos += 4
        out.append(np.frombuffer(raw, '<i2', n * HALFWORDS, pos).reshape(n, HALFWORDS).astype(np.int64))
        pos += n * HALFWORDS * 2
    return out


def maps(rec, w, h):
    """Per 8x8 cell: category, CU size, mv x/y, fractional, luma-coded TU size (0 uncoded), bits."""
    gw, gh = (w + 7) // 8, (h + 7) // 8
    m = {k: np.zeros((gh, gw), np.float64) for k in ('cat', 'cu', 'mvx', 'mvy', 'tu', 'bits', 'resbits', 'mvbits', 'qp', 'cb', 'anytu')}
    for r in rec:
        kind, pred, x, y, bw, bh = r[0], r[1], r[4], r[5], r[6], r[7]
        sl = (slice(y // 8, (y + bh + 7) // 8), slice(x // 8, (x + bw + 7) // 8))
        cells = max(1, ((min(y + bh, h) - y + 7) // 8) * ((min(x + bw, w) - x + 7) // 8))
        if kind == 0:
            m['cu'][sl] = bw
            m['bits'][sl] = r[23] / cells
            m['qp'][sl] = r[12]
            if pred == 0:
                m['cat'][sl] = INTRA
        elif kind == 1:
            if pred != 0:
                m['cat'][sl] = {1: AMVP, 2: SKIP, 3: MERGE}[int(pred)]
                m['mvx'][sl] = r[16]
                m['mvy'][sl] = r[17]
                m['mvbits'][sl] = r[23] / cells
        elif kind == 2:
            m['anytu'][sl] = bw
            m['resbits'][sl] += r[23] / cells
            if r[11] & 1:
                m['tu'][sl] = bw
            if r[11] & 6:
                m['cb'][sl] = 1
    return m


def decoded(path):
    """An ffmpeg pipe of `path` as raw 4:2:0 pictures."""
    return subprocess.Popen(['ffmpeg', '-v', 'quiet', '-i', str(path), '-f', 'rawvideo', '-pix_fmt', 'yuv420p', '-'],
                            stdout=subprocess.PIPE)


def cell_sse(a, b, w, h):
    d = (a.astype(np.int32) - b.astype(np.int32)) ** 2
    ph, pw = (h + 7) // 8 * 8, (w + 7) // 8 * 8
    p = np.zeros((ph, pw), np.int64)
    p[:h, :w] = d
    return p.reshape(ph // 8, 8, pw // 8, 8).sum((1, 3))


def records(stream, analysis_bin, scratch):
    """`nanocodec stat --records` of `stream`, as per-picture record arrays."""
    out = Path(scratch) / (Path(stream).name + '.rec')
    subprocess.run([str(Path(analysis_bin) / 'nanocodec'), 'stat', '--codec', 'hevc', '--input', str(stream),
                    '--records', str(out)], check=True)
    return frames_of(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--ours', required=True)
    parser.add_argument('--theirs', required=True)
    parser.add_argument('--source', required=True, help='the source clip (anything ffmpeg reads)')
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--first', type=int, default=1)
    parser.add_argument('--last', type=int)
    parser.add_argument('--analysis-bin', default=os.environ.get(
        'NC_ANALYSIS_BIN', str(Path.home() / 'nanocodec-data/targets/analysis/release')))
    args = parser.parse_args()
    w, h, first = args.width, args.height, args.first
    with tempdirs.temp_dir("hevc-block-compare") as scratch:
        ro, rx = (records(s, args.analysis_bin, scratch) for s in (args.ours, args.theirs))
    last = args.last if args.last is not None else min(len(ro), len(rx)) - 1
    src, po, px = decoded(args.source), decoded(args.ours), decoded(args.theirs)
    fs = w * h * 3 // 2
    acc = {}

    def add(key, val):
        acc[key] = acc.get(key, 0.0) + float(val)

    cells = 0
    for f in range(last + 1):
        s, a, b = (p.stdout.read(fs) for p in (src, po, px))
        if f < first:
            continue
        sy = np.frombuffer(s, np.uint8, w * h).reshape(h, w)
        eo = cell_sse(np.frombuffer(a, np.uint8, w * h).reshape(h, w), sy, w, h)
        ex = cell_sse(np.frombuffer(b, np.uint8, w * h).reshape(h, w), sy, w, h)
        mo, mx = maps(ro[f], w, h), maps(rx[f], w, h)
        cells += eo.size
        add('qp_o', mo['qp'].mean())
        add('qp_x', mx['qp'].mean())
        for who, m, e in (('o', mo, eo), ('x', mx, ex)):
            frac = ((m['mvx'].astype(int) | m['mvy'].astype(int)) & 3) != 0
            zero = (m['mvx'] == 0) & (m['mvy'] == 0)
            inter = m['cat'] != INTRA
            for c in range(4):
                sel = m['cat'] == c
                add(f'{who}.area.{c}', sel.sum())
                add(f'{who}.bits.{c}', m['bits'][sel].sum())
                add(f'{who}.sse.{c}', e[sel].sum())
                add(f'{who}.resbits.{c}', m['resbits'][sel].sum())
                add(f'{who}.mvbits.{c}', m['mvbits'][sel].sum())
                add(f'{who}.frac.{c}', (sel & frac).sum())
                add(f'{who}.zero.{c}', (sel & zero).sum())
                add(f'{who}.ycoded.{c}', (sel & (m['tu'] > 0)).sum())
            for size in (8, 16, 32, 64):
                add(f'{who}.cu.{size}', (m['cu'] == size).sum())
                add(f'{who}.cuinter.{size}', ((m['cu'] == size) & inter).sum())
                add(f'{who}.cuintra.{size}', ((m['cu'] == size) & ~inter).sum())
            for size in (4, 8, 16, 32):
                add(f'{who}.tu.{size}', ((m['tu'] == size) & inter).sum())
                add(f'{who}.anytu.{size}', ((m['anytu'] == size) & inter).sum())
                add(f'{who}.tuintra.{size}', ((m['tu'] == size) & ~inter).sum())
        # cross tabulation: our class by theirs
        fx = ((mx['mvx'].astype(int) | mx['mvy'].astype(int)) & 3) != 0
        zx = (mx['mvx'] == 0) & (mx['mvy'] == 0)
        for co in range(4):
            so = mo['cat'] == co
            for cx in range(4):
                sel = so & (mx['cat'] == cx)
                k = f'x.{co}.{cx}'
                add(k + '.area', sel.sum())
                add(k + '.obits', mo['bits'][sel].sum())
                add(k + '.xbits', mx['bits'][sel].sum())
                add(k + '.osse', eo[sel].sum())
                add(k + '.xsse', ex[sel].sum())
                add(k + '.xfrac', (sel & fx).sum())
                add(k + '.xzero', (sel & zx).sum())
                add(k + '.xycoded', (sel & (mx['tu'] > 0)).sum())
                add(k + '.xresbits', mx['resbits'][sel].sum())
                add(k + '.oresbits', mo['resbits'][sel].sum())
                for size in (8, 16, 32, 64):
                    add(k + f'.xcu{size}', (sel & (mx['cu'] == size)).sum())
                    add(k + f'.ocu{size}', (sel & (mo['cu'] == size)).sum())
    n = last + 1 - first
    qp = acc['qp_x'] / n
    lam = LAMBDA_AT_QP12 * 2 ** ((qp - 12) / 3)
    print(f'{args.ours} vs {args.theirs}: pictures {first}..{last}, mean QP ours {acc["qp_o"] / n:.2f} theirs {qp:.2f}, lambda {lam:.1f}')
    tot = {who: (sum(acc[f'{who}.bits.{c}'] for c in range(4)), sum(acc[f'{who}.sse.{c}'] for c in range(4))) for who in 'ox'}
    for who in 'ox':
        bits, sse = tot[who]
        print(f' {"ours" if who == "o" else "theirs"}: bits/frame {bits / n:.0f}  PSNR-Y {10 * np.log10(255 ** 2 * cells * 64 / sse):.3f}  J/cell {(sse + lam * bits) / cells:.2f}')
        print('   cat    area%  bits%  bits/cell res/cell mv/cell sse/cell  frac% zero% Ycoded%')
        for c in range(4):
            a = acc[f'{who}.area.{c}']
            if a == 0:
                continue
            print(f'   {NAMES[c]:6s} {100 * a / cells:5.1f} {100 * acc[f"{who}.bits.{c}"] / bits:6.1f} {acc[f"{who}.bits.{c}"] / a:8.2f} {acc[f"{who}.resbits.{c}"] / a:7.2f} {acc[f"{who}.mvbits.{c}"] / a:7.2f} {acc[f"{who}.sse.{c}"] / a:8.1f} {100 * acc[f"{who}.frac.{c}"] / a:6.1f} {100 * acc[f"{who}.zero.{c}"] / a:5.1f} {100 * acc[f"{who}.ycoded.{c}"] / a:6.1f}')
        print('   CU size area%  all/inter/intra: ' + '  '.join(f'{s}: {100 * acc[f"{who}.cu.{s}"] / cells:.1f}/{100 * acc[f"{who}.cuinter.{s}"] / cells:.1f}/{100 * acc[f"{who}.cuintra.{s}"] / cells:.1f}' for s in (8, 16, 32, 64)))
        print('   inter luma-coded TU area%: ' + '  '.join(f'{s}: {100 * acc[f"{who}.tu.{s}"] / cells:.1f}' for s in (4, 8, 16, 32))
              + '   inter TU tree area%: ' + '  '.join(f'{s}: {100 * acc[f"{who}.anytu.{s}"] / cells:.1f}' for s in (4, 8, 16, 32))
              + '   intra coded: ' + '  '.join(f'{s}: {100 * acc[f"{who}.tuintra.{s}"] / cells:.1f}' for s in (4, 8, 16, 32)))
    print(' cross (ours row, theirs column): area% | ours bits/cell, sse/cell | their bits/cell, sse/cell | dJ/cell (ours-theirs) and share of the total J gap')
    gap = (tot['o'][1] + lam * tot['o'][0]) - (tot['x'][1] + lam * tot['x'][0])
    for co in range(4):
        for cx in range(4):
            k = f'x.{co}.{cx}'
            a = acc[k + '.area']
            if a < 0.002 * cells:
                continue
            dj = (acc[k + '.osse'] + lam * acc[k + '.obits']) - (acc[k + '.xsse'] + lam * acc[k + '.xbits'])
            print(f'   {NAMES[co]:6s}->{NAMES[cx]:6s} {100 * a / cells:5.1f}% | {acc[k + ".obits"] / a:7.2f} {acc[k + ".osse"] / a:7.1f} | {acc[k + ".xbits"] / a:7.2f} {acc[k + ".xsse"] / a:7.1f} | dJ {dj / a:8.1f}  gap share {100 * dj / gap:6.1f}%'
                  f' | theirs there: frac {100 * acc[k + ".xfrac"] / a:4.0f}% zero {100 * acc[k + ".xzero"] / a:4.0f}% Ycoded {100 * acc[k + ".xycoded"] / a:4.0f}% CU ' + '/'.join(f'{100 * acc[k + f".xcu{s}"] / a:.0f}' for s in (8, 16, 32, 64))
                  + ' ours CU ' + '/'.join(f'{100 * acc[k + f".ocu{s}"] / a:.0f}' for s in (8, 16, 32, 64)))
    print(f' total J gap per cell {gap / cells:.2f} ({100 * gap / (tot["x"][1] + lam * tot["x"][0]):.1f}% of their J)')




if __name__ == '__main__':
    main()
