#!/usr/bin/env python3
"""Decode the Argon AV1 conformance streams (AOM, open-sourced by Graphcore
and Broadcom: argon_coveragetool_av1_base_and_extended_profiles_v2.1.1.zip,
https://aomedia.org/av1-video-decoder-verification-tool/) with our decoder
and aomdec, and compare their whole output MD5s.

Argon ships reference MD5s of `aomdec --rawvideo --all-layers` output; our
CLI writes the conformance runner's form (tools/ci/conformance.py: the
operating point's output frames, monochrome with mid-grey chroma), so the
reference here is aomdec run on the same input in that form (`--i420`).
Each stream (Annex B in the `*_core*` and `*_stress` sets, Section 5 in
`*_not_annexb*`) is first rewritten as IVF, one temporal unit per frame,
every OBU with its size field (§5.2, Annex B.2); both decoders read that
file. Nothing decoded is written to disk.

    python3 -I tools/dev/av1_argon.py ARGON_DIR --fetch [--sets ...]
    python3 -I tools/dev/av1_argon.py ARGON_DIR --bin NANOCODEC --aomdec AOMDEC
        [--sets profile0_core,profile0_not_annexb] [--jobs 8] [--json OUT]

ARGON_DIR holds the `argon_coveragetool_av1_base_and_extended_profiles_v2.1`
tree (untrusted data: only read). `--fetch` fills it with the chosen sets'
streams straight from the published zip by HTTP range requests (the whole
zip is 7 GB; the four default sets' streams about 0.8 GB). Large scale tile sets need tile list
decoding (not supported) and the error set holds invalid streams: neither
is a default set.
"""

import argparse
import concurrent.futures
import hashlib
import io
import json
import struct
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = ("https://aom-cwg-av1-argon-streams-public.s3.us-east-1.amazonaws.com/"
           "argon_coveragetool_av1_base_and_extended_profiles_v2.1.1.zip")
ROOT_NAME = "argon_coveragetool_av1_base_and_extended_profiles_v2.1"
RANGE_BUFFER = 1 << 22

DEFAULT_SETS = ("profile0_core,profile0_core_special,profile0_not_annexb,"
                "profile0_not_annexb_special")
OBU_TEMPORAL_DELIMITER = 2
OBU_HAS_SIZE = 0x02
OBU_HAS_EXTENSION = 0x04
OBU_TYPE_SHIFT = 3
OBU_TYPE_MASK = 0x0F
TIMEOUT = 900
CHUNK = 1 << 20


def leb128(data, pos):
    value = 0
    for i in range(8):
        byte = data[pos + i]
        value |= (byte & 0x7F) << (7 * i)
        if not byte & 0x80:
            return value, pos + i + 1
    raise ValueError("leb128 longer than 8 bytes")


def encode_leb128(value):
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def with_size(obu):
    """An OBU (header + payload, no size field) with its size field set."""
    header = obu[0]
    header_len = 2 if header & OBU_HAS_EXTENSION else 1
    return (bytes([header | OBU_HAS_SIZE]) + obu[1:header_len]
            + encode_leb128(len(obu) - header_len) + obu[header_len:])


def annex_b_units(data):
    """Temporal units of an Annex B stream as Section 5 bytes (Annex B.2)."""
    pos = 0
    while pos < len(data):
        unit_size, pos = leb128(data, pos)
        end = pos + unit_size
        out = bytearray()
        while pos < end:
            frame_size, pos = leb128(data, pos)
            frame_end = pos + frame_size
            while pos < frame_end:
                obu_length, pos = leb128(data, pos)
                obu = data[pos:pos + obu_length]
                pos += obu_length
                header_len = 2 if obu[0] & OBU_HAS_EXTENSION else 1
                if obu[0] & OBU_HAS_SIZE:
                    # The OBU's own size may stop short of obu_length.
                    size, after = leb128(obu, header_len)
                    obu = obu[:after + size]
                    out += obu
                else:
                    out += with_size(obu)
        yield bytes(out)


def section5_units(data):
    """Temporal units of a Section 5 stream: split at temporal delimiters."""
    pos, unit = 0, bytearray()
    while pos < len(data):
        header = data[pos]
        header_len = 2 if header & OBU_HAS_EXTENSION else 1
        if not header & OBU_HAS_SIZE:
            raise ValueError("Section 5 OBU without a size field")
        size, after = leb128(data, pos + header_len)
        end = after + size
        if (header >> OBU_TYPE_SHIFT) & OBU_TYPE_MASK == OBU_TEMPORAL_DELIMITER and unit:
            yield bytes(unit)
            unit = bytearray()
        unit += data[pos:end]
        pos = end
    if unit:
        yield bytes(unit)


def write_ivf(units, path):
    with open(path, "wb") as f:
        # IVF header; the size fields are not read by either decoder.
        f.write(b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"AV01", 0, 0, 30, 1, len(units), 0))
        for index, unit in enumerate(units):
            f.write(struct.pack("<IQ", len(unit), index) + unit)


def hashed(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest, size = hashlib.md5(), 0
    try:
        while chunk := process.stdout.read(CHUNK):
            digest.update(chunk)
            size += len(chunk)
        _, err = process.communicate(timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        return None, size, "timeout"
    error = err.decode(errors="replace").strip().splitlines()
    return digest.hexdigest() if process.returncode == 0 else None, size, error[-1] if error else ""


class RangeReader(io.RawIOBase):
    """A seekable read-only view of a URL through HTTP range requests."""

    def __init__(self, url):
        super().__init__()
        self.url, self.pos = url, 0
        head = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(head, timeout=60) as response:
            self.size = int(response.headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.pos, io.SEEK_END: self.size}[whence]
        self.pos = base + offset
        return self.pos

    def readinto(self, buffer):
        count = min(len(buffer), self.size - self.pos)
        if count <= 0:
            return 0
        request = urllib.request.Request(
            self.url, headers={"Range": f"bytes={self.pos}-{self.pos + count - 1}"})
        with urllib.request.urlopen(request, timeout=300) as response:
            data = response.read()
        buffer[:len(data)] = data
        self.pos += len(data)
        return len(data)


def fetch(argon, sets):
    """Extract the streams of `sets` from the published zip into `argon`."""
    archive = zipfile.ZipFile(io.BufferedReader(RangeReader(ZIP_URL), RANGE_BUFFER))
    wanted = tuple(f"{ROOT_NAME}/{name}/streams/" for name in sets)
    count = 0
    for info in archive.infolist():
        if info.is_dir() or not info.filename.startswith(wanted):
            continue
        target = argon / Path(info.filename).relative_to(ROOT_NAME)
        if target.exists() and target.stat().st_size == info.file_size:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(archive.read(info))
        count += 1
    print(f"fetched {count} streams into {argon}")


def run_one(stream, args, annex_b):
    data = stream.read_bytes()
    try:
        units = list(annex_b_units(data) if annex_b else section5_units(data))
    except (ValueError, IndexError) as e:
        return {"status": "framing", "error": str(e)}
    with tempfile.TemporaryDirectory(prefix="nanocodec-argon-") as scratch:
        ivf = Path(scratch) / "stream.ivf"
        write_ivf(units, ivf)
        reference, ref_bytes, ref_error = hashed(
            [args.aomdec, "--i420", "-o", "/dev/stdout", str(ivf)])
        ours, our_bytes, our_error = hashed(
            [args.bin, "decode", "--codec", "av1", "--input", str(ivf), "--output", "/dev/stdout"])
    if reference is None:
        status = "reference-failed"
    elif ours is None:
        status = "error"
    else:
        status = "pass" if ours == reference else "mismatch"
    return {"status": status, "bytes": [our_bytes, ref_bytes], "error": our_error or ref_error}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("argon", type=Path)
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--bin")
    parser.add_argument("--aomdec")
    parser.add_argument("--sets", default=DEFAULT_SETS)
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.fetch:
        fetch(args.argon, args.sets.split(","))
        return 0
    if not (args.bin and args.aomdec):
        parser.error("--bin and --aomdec are required to decode")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(args.jobs) as pool:
        futures = {}
        for name in args.sets.split(","):
            annex_b = "not_annexb" not in name
            for stream in sorted((args.argon / name / "streams").glob("*.obu")):
                futures[pool.submit(run_one, stream, args, annex_b)] = f"{name}/{stream.name}"
        for future in concurrent.futures.as_completed(futures):
            results[futures[future]] = future.result()
    summary = {}
    for key, result in results.items():
        counts = summary.setdefault(key.split("/")[0], {})
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    errors = {}
    for result in results.values():
        if result["status"] == "error":
            errors[result["error"]] = errors.get(result["error"], 0) + 1
    print(json.dumps({"summary": summary, "errors": errors}, indent=1))
    if args.json:
        args.json.write_text(json.dumps({"summary": summary, "errors": errors,
                                         "streams": dict(sorted(results.items()))}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
