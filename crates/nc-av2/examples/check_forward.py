#!/usr/bin/env python3
"""Streamed full two-family AV2 key/P gate; one slot for a bounded tiny batch."""
import argparse
import pathlib
import subprocess
import sys
import tempfile
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[3]/'eval'))
from nceval.codecs import Codec
from nceval.config import load_paths
from nceval.conformance import gate_stream
from nceval.slots import cpu_slots,low_priority
from nceval.yuv import YuvFormat, frame_digest
parser=argparse.ArgumentParser()
parser.add_argument('--encoder',type=pathlib.Path,required=True)
parser.add_argument('--motion',action='store_true')
parser.add_argument('--adaptive',action='store_true')
parser.add_argument('--tx8',action='store_true')
args=parser.parse_args()
with cpu_slots(1),tempfile.TemporaryDirectory(prefix='nc-av2-forward-') as temp:
    for width,height in [(8,8),(16,16),(64,64),(72,40)]+([(128,136)] if args.motion else []):
        frame_len=width*height*3//2
        source=b''.join(bytes((i*37+i//width*11+19+(index//2)*23*((i//8)%2))%256 for i in range(frame_len)) for index in range(5))
        packet=bytearray();expected=[]
        for qindex in [0,1,100,200,255]:
            coded=subprocess.run(low_priority([str(args.encoder),str(width),str(height),str(qindex)]+(['--motion'] if args.motion else [])+(['--adaptive'] if args.adaptive else [])+(['--tx8'] if args.tx8 else [])),input=source,capture_output=True,check=True).stdout
            decoded=subprocess.run(low_priority([str(args.encoder),'--decode']),input=coded,capture_output=True,check=True).stdout
            assert len(decoded)==len(source)
            if qindex==0:assert decoded==source
            expected.extend(frame_digest(decoded[i:i+frame_len]) for i in range(0,len(decoded),frame_len))
            packet.extend(coded)
        path=pathlib.Path(temp)/f'{width}-{height}.obu';path.write_bytes(packet)
        result=gate_stream(path,Codec.AV2,YuvFormat(width,height),len(expected),load_paths(),expected_hashes=expected,serial=True,hold_slots=False)
        print(f'{width}x{height}: {len(expected)} frames, {len(packet)} bytes, passed={result.passed}, gate={result.level}, reason={result.reason}',flush=True)
        if not result.passed:
            failure=pathlib.Path('/tmp/nc-av2-forward-failed.obu');failure.write_bytes(packet)
            print(f'compressed failure retained at {failure}',flush=True)
            sys.exit(1)
