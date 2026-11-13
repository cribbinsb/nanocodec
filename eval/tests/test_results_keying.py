"""Shared results DB: builds of one encoder name keep separate curves (no eviction)."""

import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nceval import remeasure
from nceval.framestats import FrameInfo
from nceval.results import EncodeRecord, ResultsDb

OLD_ENCODES = """
CREATE TABLE encodes (
    id INTEGER PRIMARY KEY, encoder TEXT NOT NULL, structure TEXT NOT NULL,
    clip TEXT NOT NULL, quantizer INTEGER NOT NULL, frames INTEGER NOT NULL,
    bytes INTEGER NOT NULL, kbps REAL NOT NULL,
    conformance TEXT NOT NULL CHECK (conformance IN ('pass', 'fail')),
    conformance_reason TEXT NOT NULL DEFAULT '', platform TEXT NOT NULL,
    fingerprint TEXT NOT NULL, harness_commit TEXT NOT NULL, created REAL NOT NULL,
    UNIQUE (encoder, structure, clip, quantizer, frames, platform));
CREATE TABLE metrics (encode_id INTEGER NOT NULL REFERENCES encodes(id) ON DELETE CASCADE,
    name TEXT NOT NULL, value REAL NOT NULL, PRIMARY KEY (encode_id, name));
"""


def record(fingerprint, q, frames=150, kbps=None):
    return EncodeRecord("nanocodec-hevc", "ip10s", "clip", q, frames, 1000, kbps or 100.0 * q,
                        True, "", "x86_64", fingerprint, "c")


def store_curve(db, fingerprint, frames=150, base=1.0):
    for q in (30, 33, 36, 39):
        db.store(record(fingerprint, q, frames, base * q),
                 {"psnr_yuv": 40.0 - q / 10}, [FrameInfo(10, "P")] * frames,
                 [{"psnr_y": 40.0}] * frames)
        time.sleep(0.002)  # distinct creation times order the curves


class FingerprintKeying(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "results.sqlite"

    def tearDown(self):
        self.tmp.cleanup()

    def test_another_build_no_longer_evicts_a_curve(self):
        db = ResultsDb(self.path)
        store_curve(db, "scoreboard")
        store_curve(db, "other-agent", base=2.0)
        self.assertTrue(db.curve_is_current("nanocodec-hevc", "ip10s", "clip", 150, "x86_64",
                                            "scoreboard"))
        self.assertTrue(db.curve_is_current("nanocodec-hevc", "ip10s", "clip", 150, "x86_64",
                                            "other-agent"))
        # Readers get the latest curve unless they name one.
        latest = db.rd_points("nanocodec-hevc", "ip10s", "clip", "x86_64", "psnr_yuv")
        named = db.rd_points("nanocodec-hevc", "ip10s", "clip", "x86_64", "psnr_yuv",
                             fingerprint="scoreboard")
        self.assertEqual([r for r, _ in latest], [60.0, 66.0, 72.0, 78.0])
        self.assertEqual([r for r, _ in named], [30.0, 33.0, 36.0, 39.0])
        db.clear_curve("nanocodec-hevc", "ip10s", "clip", 150, "x86_64", "other-agent")
        self.assertTrue(db.curve_is_current("nanocodec-hevc", "ip10s", "clip", 150, "x86_64",
                                            "scoreboard"))
        self.assertEqual(len(db.rd_points("nanocodec-hevc", "ip10s", "clip", "x86_64",
                                          "psnr_yuv")), 4)

    def test_relabel_supersedes_a_partial_curve_under_the_new_key(self):
        db = ResultsDb(self.path)
        store_curve(db, "old")
        db.store(record("new", 30), {"psnr_yuv": 37.0})
        self.assertEqual(db.relabel_curve("nanocodec-hevc", "ip10s", "clip", 150, "x86_64",
                                          "old", "new"), 4)
        self.assertTrue(db.curve_is_current("nanocodec-hevc", "ip10s", "clip", 150, "x86_64",
                                            "new"))

    def test_old_database_is_migrated_keeping_rows_ids_and_metrics(self):
        with sqlite3.connect(self.path) as conn:
            conn.executescript(OLD_ENCODES)
            conn.execute("INSERT INTO encodes VALUES (7, 'nanocodec-hevc', 'ip10s', 'clip', 30, "
                         "150, 1000, 300.0, 'pass', '', 'x86_64', 'fp', 'c', 1.0)")
            conn.execute("INSERT INTO metrics VALUES (7, 'psnr_yuv', 37.0)")
        db = ResultsDb(self.path)
        self.assertEqual(db.encode_metrics(7), {"psnr_yuv": 37.0})
        db.store(record("fp2", 30), {"psnr_yuv": 38.0})
        self.assertEqual(db.encode_metrics(7), {"psnr_yuv": 37.0})
        ResultsDb(self.path)  # a second open finds the new key and does nothing

    def test_stored_points_pick_the_requested_length(self):
        db = ResultsDb(self.path)
        store_curve(db, "val150", frames=150)
        store_curve(db, "full", frames=300)
        source = remeasure.Source(self.path, "ip10s")
        with patch.object(remeasure.platform, "machine", return_value="x86_64"):
            found = remeasure.stored_points([source], "nanocodec-hevc", "clip", 150)
            self.assertEqual({(frames, fp) for _, frames, fp in found[1]}, {(150, "val150")})
            latest = remeasure.stored_points([source], "nanocodec-hevc", "clip")
            self.assertEqual({fp for _, _, fp in latest[1]}, {"full"})


if __name__ == "__main__":
    unittest.main()
