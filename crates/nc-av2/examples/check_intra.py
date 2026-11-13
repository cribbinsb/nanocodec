#!/usr/bin/env python3
"""Bounded streamed AV2 picture gate: native decode vs AVM and dav2d.

Raw pictures stay in memory. One CPU slot covers this tiny batch; compressed
packets live in a temporary directory. Uses the repository evaluation framework.
Usage: NC_ANCHORS=... python3 check_intra.py --encoder /path/to/intra_frame
"""
import argparse
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3] / 'eval'))
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.conformance import gate_stream
from nceval.slots import cpu_slots, low_priority
from nceval.yuv import YuvFormat, frame_digest

parser=argparse.ArgumentParser()
parser.add_argument('--encoder',type=pathlib.Path,required=True)
parser.add_argument('--tx8',action='store_true')
args=parser.parse_args()
with cpu_slots(1), tempfile.TemporaryDirectory(prefix='nc-av2-picture-') as tmp:
    for width,height in [(8,8),(16,16),(64,64),(72,40)]:
        packet=bytearray();hashes=[]
        for qindex in [0,1,90,140,190,255]:
            for pattern in ['flat','texture']:
                pixels=(bytes([128])*(width*height*3//2) if pattern=='flat'
                        else bytes((i*37+i//width*11+19)%256 for i in range(width*height*3//2)))
                encoded=subprocess.run(low_priority([str(args.encoder),str(width),str(height),str(qindex)]+(['--tx8'] if args.tx8 else [])),input=pixels,capture_output=True,check=True).stdout
                reconstructed=subprocess.run(low_priority([str(args.encoder),'--decode']),input=encoded,capture_output=True,check=True).stdout
                assert len(reconstructed)==len(pixels)
                if qindex==0: assert reconstructed==pixels
                hashes.append(frame_digest(reconstructed))
                packet.extend(encoded)
        path=pathlib.Path(tmp)/f'{width}-{height}.obu';path.write_bytes(packet)
        gate=gate_stream(path,Codec.AV2,YuvFormat(width,height),len(hashes),load_paths(),expected_hashes=hashes,serial=True,hold_slots=False)
        print(f'{width}x{height}: {len(hashes)} frames, {len(packet)} bytes, passed={gate.passed}, gate={gate.level}, reason={gate.reason}',flush=True)
        if not gate.passed:sys.exit(1)
