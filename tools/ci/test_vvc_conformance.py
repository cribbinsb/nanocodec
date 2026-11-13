#!/usr/bin/env python3
"""Regression tests for bounded streaming conformance execution."""
import hashlib
import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('vvc_conformance', Path(__file__).with_name('vvc_conformance.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StreamingGate(unittest.TestCase):
    def run_fixture(self, program, expected=b'pixels', timeout=2):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'decoder'
            binary.write_text('#!/usr/bin/env python3\n' + program)
            binary.chmod(0o700)
            (root / 'input.bit').write_bytes(b'stream')
            entry = dict(bitstream='input.bit', sha256=hashlib.sha256(b'stream').hexdigest(),
                         md5=hashlib.md5(expected).hexdigest())
            return runner.run_one(binary, root, entry, timeout)

    def test_pixels_and_full_stderr_pipe_do_not_deadlock(self):
        result = self.run_fixture("import sys\nsys.stderr.write('x'*200000)\nsys.stdout.buffer.write(b'pixels')\n")
        self.assertEqual(result['status'], 'pass')
        self.assertLessEqual(len(result['error']), 1000)

    def test_hang_before_output_is_bounded(self):
        result = self.run_fixture('import time\ntime.sleep(10)\n', timeout=0.1)
        self.assertEqual(result['status'], 'timeout')

    def test_nonzero_exit_never_passes_matching_pixels(self):
        result = self.run_fixture("import sys\nsys.stdout.buffer.write(b'pixels')\nsys.exit(1)\n")
        self.assertEqual(result['status'], 'error')

    def test_wrong_pixels_fail(self):
        result = self.run_fixture("print('wrong')\n")
        self.assertEqual(result['status'], 'mismatch')


if __name__ == '__main__':
    unittest.main()
