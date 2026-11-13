#!/usr/bin/env python3
"""Check our VVC decoder against official 8-bit Main output hashes, without YUV files."""
import argparse
import concurrent.futures
import hashlib
import json
import selectors
import subprocess
import time
from pathlib import Path


def run_one(binary, root, entry, timeout):
    path = root / entry['bitstream']
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
        return dict(status='error', error='input SHA-256 mismatch')
    return hash_output([str(binary), 'decode', '--codec', 'vvc', '--input', str(path),
                                '--output', '/dev/stdout'], entry['md5'], timeout)


def hash_output(command, expected, timeout):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    digest, size, errors = hashlib.md5(), 0, bytearray()
    deadline = time.monotonic() + timeout
    timed_out = False
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, 'pixels')
            selector.register(process.stderr, selectors.EVENT_READ, 'errors')
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 1)):
                    chunk = key.fileobj.read1(65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                    elif key.data == 'pixels':
                        digest.update(chunk)
                        size += len(chunk)
                    else:
                        errors.extend(chunk)
                        del errors[:-8192]
            if timed_out:
                process.kill()
            try:
                process.wait(timeout=max(0.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                process.wait()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stdout.close()
        process.stderr.close()
    status = 'timeout' if timed_out else 'error' if process.returncode else 'pass' if digest.hexdigest() == expected else 'mismatch'
    return dict(status=status, bytes=size, md5=digest.hexdigest(), expected_md5=expected,
                error=errors.decode(errors='replace').strip()[-1000:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin', type=Path, required=True)
    parser.add_argument('--streams', type=Path, default=Path.home() / 'nanocodec-data/conformance/vvc/streams')
    parser.add_argument('--manifest', type=Path, default=Path(__file__).with_name('vvc_conformance_main8.json'))
    parser.add_argument('--jobs', type=int, choices=range(1, 5), default=4)
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--only', default='')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    if not 0 < args.timeout <= 600:
        parser.error('--timeout must be between 1 and 600 seconds')
    entries = json.loads(args.manifest.read_text())
    if args.only:
        wanted = set(args.only.split(','))
        names = {entry['name'] for entry in entries}
        if wanted - names:
            parser.error(f'unknown streams: {sorted(wanted - names)}')
        entries = [entry for entry in entries if entry['name'] in wanted]
    if not entries:
        parser.error('empty conformance selection')
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        pending = {pool.submit(run_one, args.bin, args.streams, entry, args.timeout): entry['name'] for entry in entries}
        for future in concurrent.futures.as_completed(pending):
            name = pending[future]
            try:
                result = future.result()
            except (OSError, ValueError) as error:
                result = dict(status='error', error=str(error))
            results[name] = result
            print(f'{result["status"]:8} {name}: {result.get("error", "")}', flush=True)
    passed = sum(row['status'] == 'pass' for row in results.values())
    print(f'VVC 8-bit Main: {passed}/{len(results)} match official output MD5')
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, sort_keys=True)+'\n')
    return int(passed != len(results))


if __name__ == '__main__':
    raise SystemExit(main())
