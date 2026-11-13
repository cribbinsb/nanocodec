#!/usr/bin/env python3
"""Fetch ITU H.266.1 first-edition revised vectors and select 8-bit Main streams.

Archives stay outside the repository. Extract only compressed streams and small
metadata, never reference YUV. Selection uses ffprobe's SPS-derived profile and
pixel format, not filenames. Expected output MD5 comes from the official zip.
"""
import argparse
import hashlib
import io
import json
import re
import subprocess
import urllib.request
import zipfile
from pathlib import Path

INDEX = "https://www.itu.int/wftp3/av-arch/jvet-site/bitstream_exchange/VVC/FDIS_r1/"
MAX_ARCHIVE = 64 << 20
MAX_EXTRACTED_STREAMS = 512 << 20


def fetch(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        data = response.read(MAX_ARCHIVE + 1)
    if len(data) > MAX_ARCHIVE:
        raise ValueError(f"oversize download: {url}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.home() / 'nanocodec-data/conformance/vvc')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1] / 'ci/vvc_conformance_main8.json')
    args = parser.parse_args()
    archives = args.root / 'archives'
    streams = args.root / 'streams'
    archives.mkdir(parents=True, exist_ok=True)
    streams.mkdir(parents=True, exist_ok=True)
    links = re.findall(r'HREF="([^"]+\.zip)"', fetch(INDEX).decode(), re.I)
    entries, excluded = [], []
    extracted_bytes = 0
    for href in links:
        name = Path(href).name
        archive = archives / name
        if not archive.exists():
            data = fetch('https://www.itu.int' + href)
            temporary = archive.with_suffix('.part')
            temporary.write_bytes(data)
            temporary.replace(archive)
        with zipfile.ZipFile(io.BytesIO(archive.read_bytes())) as z:
            members = [m for m in z.infolist() if m.filename.lower().endswith(('.bit', '.bin', '.266', '.vvc')) and '__MACOSX' not in m.filename]
            for m in members:
                if m.file_size > MAX_ARCHIVE:
                    raise ValueError(f'oversize bitstream: {name}/{m.filename}')
                extracted_bytes += m.file_size
                if extracted_bytes > MAX_EXTRACTED_STREAMS:
                    raise ValueError('corpus exceeds compressed-stream size budget')
                relative = Path(archive.stem) / Path(m.filename).name
                target = streams / relative
                target.parent.mkdir(exist_ok=True)
                data = z.read(m)
                target.write_bytes(data)
                probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
                    'stream=profile,pix_fmt,width,height', '-of', 'json', str(target)],
                    capture_output=True, text=True, timeout=30, check=True)
                info = json.loads(probe.stdout).get('streams', [])
                if len(info) != 1:
                    raise ValueError(f'unclassified stream {relative}: {probe.stdout}')
                info = info[0]
                row = dict(name=target.stem, bitstream=str(relative), **info)
                if info.get('profile') not in ('Main 10', 'Main 10 Still Picture') or info.get('pix_fmt') not in ('yuv420p', 'gray'):
                    excluded.append(row)
                    continue
                hashes = []
                for md in z.infolist():
                    if md.filename.lower().endswith('.yuv.md5') and md.file_size < 4096:
                        hashes.extend(re.findall(r'\b[0-9a-fA-F]{32}\b', z.read(md).decode()))
                if len(hashes) != 1:
                    raise ValueError(f'expected one official output MD5: {name}: {hashes}')
                row.update(md5=hashes[0].lower(), sha256=hashlib.sha256(data).hexdigest(),
                    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), url='https://www.itu.int'+href)
                entries.append(row)
                print(f'selected {row["name"]}', flush=True)
    args.manifest.write_text(json.dumps(entries, indent=2) + '\n')
    (args.root / 'excluded.json').write_text(json.dumps(excluded, indent=2) + '\n')
    print(f'{len(entries)} selected; {len(excluded)} excluded by profile/format')


if __name__ == '__main__':
    main()
