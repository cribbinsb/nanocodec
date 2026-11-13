"""Results database (SQLite).

One row per encode, keyed by everything that defines it, so a rerun of the
same test point replaces its old result instead of duplicating it. The key
includes the configuration fingerprint (2026-11-08): agents sweep the same
encoder name with different builds into this shared file, and keyed by name
alone each sweep evicted the others' rows (the scoreboard's curves stopped
being current mid-run). A curve is now one fingerprint's rows; readers that
do not name a fingerprint get the most recently measured one. Metrics
and per-frame data are stored only for encodes that passed the conformance
gate. Every encode and speed row carries the fingerprint of the encoder
configuration and measurement chain it was produced with (nceval.fingerprint),
so stale results can be found and redone.

    encodes(id, encoder, structure, clip, quantizer, frames, bytes, kbps,
            conformance, conformance_reason, platform, fingerprint,
            harness_commit, created)
    metrics(encode_id, name, value)                     sequence values
    frames(encode_id, frame, bytes, frame_type, temporal_id, qp,
           psnr_y, psnr_u, psnr_v, vmaf, vmaf_neg)      per-frame data (§4.5a)
    speed(encoder, structure, clip, quantizer, frames, platform, core, runs,
          ms_per_frame, cpu_ms_per_frame, instructions, cycles, fingerprint,
          harness_commit, created)
"""

from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .framestats import FrameInfo

# How long a write waits for another process holding the database lock.
DB_LOCK_TIMEOUT_SECONDS = 300

SCHEMA = """
CREATE TABLE IF NOT EXISTS encodes (
    id INTEGER PRIMARY KEY,
    encoder TEXT NOT NULL,
    structure TEXT NOT NULL,
    clip TEXT NOT NULL,
    quantizer INTEGER NOT NULL,
    frames INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    kbps REAL NOT NULL,
    conformance TEXT NOT NULL CHECK (conformance IN ('pass', 'fail')),
    conformance_reason TEXT NOT NULL DEFAULT '',
    gate_level TEXT NOT NULL DEFAULT '',
    platform TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    harness_commit TEXT NOT NULL,
    created REAL NOT NULL,
    UNIQUE (encoder, structure, clip, quantizer, frames, platform, fingerprint)
);
CREATE TABLE IF NOT EXISTS metrics (
    encode_id INTEGER NOT NULL REFERENCES encodes(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    PRIMARY KEY (encode_id, name)
);
CREATE TABLE IF NOT EXISTS frames (
    encode_id INTEGER NOT NULL REFERENCES encodes(id) ON DELETE CASCADE,
    frame INTEGER NOT NULL,
    bytes INTEGER NOT NULL,
    frame_type TEXT NOT NULL,
    temporal_id INTEGER,
    qp REAL,
    psnr_y REAL, psnr_u REAL, psnr_v REAL,
    vmaf REAL, vmaf_neg REAL, vmaf_v1 REAL, ms_ssim REAL,
    PRIMARY KEY (encode_id, frame)
);
CREATE TABLE IF NOT EXISTS speed (
    id INTEGER PRIMARY KEY,
    encoder TEXT NOT NULL,
    structure TEXT NOT NULL,
    clip TEXT NOT NULL,
    quantizer INTEGER NOT NULL,
    frames INTEGER NOT NULL,
    platform TEXT NOT NULL,
    core INTEGER NOT NULL,
    runs INTEGER NOT NULL,
    ms_per_frame REAL NOT NULL,      -- median wall time
    cpu_ms_per_frame REAL NOT NULL,  -- median user+sys CPU time
    instructions REAL,               -- median, from perf stat (NULL if unavailable)
    cycles REAL,
    fingerprint TEXT NOT NULL,
    harness_commit TEXT NOT NULL,
    created REAL NOT NULL
);
"""

MIN_CURVE_POINTS = 4


@dataclass(frozen=True)
class SpeedRecord:
    encoder: str
    structure: str
    clip: str
    quantizer: int
    frames: int
    platform: str
    core: int
    runs: int
    ms_per_frame: float
    cpu_ms_per_frame: float
    instructions: float | None
    cycles: float | None
    fingerprint: str
    harness_commit: str


@dataclass(frozen=True)
class EncodeRecord:
    encoder: str
    structure: str
    clip: str
    quantizer: int
    frames: int
    bytes: int
    kbps: float
    passed: bool
    reason: str
    platform: str
    fingerprint: str
    harness_commit: str
    gate_level: str = ""  # nceval.gatepolicy.GateLevel of a passed stream


class ResultsDb:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        # The sweep reads and stores results from worker threads: one shared
        # connection, and EVERY statement (reads too) goes through self.lock.
        # Unlocked concurrent reads on a shared sqlite3 connection return
        # garbled rows.
        # Several sweeps (one per agent/codec) share this database file:
        # wait for another process's write rather than failing with
        # "database is locked", and use WAL so readers never block writers.
        self.conn = sqlite3.connect(path, check_same_thread=False,
                                    timeout=DB_LOCK_TIMEOUT_SECONDS)
        self.conn.execute("PRAGMA journal_mode = WAL")
        self.lock = threading.Lock()
        self.conn.executescript(SCHEMA)
        self._add_column("encodes", "gate_level", "TEXT NOT NULL DEFAULT ''")
        self._key_encodes_by_fingerprint()
        self.conn.execute("PRAGMA foreign_keys = ON")
        for column in ("vmaf_v1", "ms_ssim"):
            self._add_column("frames", column, "REAL")
        self.conn.commit()

    def _key_encodes_by_fingerprint(self) -> None:
        """Migrate an older database whose encodes were unique by encoder name alone.

        SQLite cannot change a UNIQUE constraint in place: rebuild the table
        (foreign keys off, so the metrics and frames rows survive the drop),
        keeping every row and its id. A concurrent migration is tolerated.
        """
        def table_sql() -> str:
            return self.conn.execute("SELECT sql FROM sqlite_master WHERE type='table' "
                                     "AND name='encodes'").fetchone()[0]
        if "platform, fingerprint)" in table_sql():
            return
        self.conn.commit()
        self.conn.execute("PRAGMA foreign_keys = OFF")
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            if "platform, fingerprint)" not in table_sql():
                columns = ", ".join(row[1] for row in
                                    self.conn.execute("PRAGMA table_info(encodes)"))
                create = SCHEMA.split(";")[0].replace("CREATE TABLE IF NOT EXISTS encodes",
                                                      "CREATE TABLE encodes_keyed")
                self.conn.execute(create)
                self.conn.execute(f"INSERT INTO encodes_keyed ({columns}) "
                                  f"SELECT {columns} FROM encodes")
                self.conn.execute("DROP TABLE encodes")
                self.conn.execute("ALTER TABLE encodes_keyed RENAME TO encodes")
            self.conn.commit()
        except BaseException:
            self.conn.rollback()
            raise

    def _add_column(self, table: str, column: str, declaration: str) -> None:
        """Migrate an older database; tolerate a concurrent migration."""
        columns = lambda: {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column in columns():
            return
        try:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")
        except sqlite3.OperationalError:
            if column not in columns():
                raise

    # --- encodes -----------------------------------------------------------

    def store(self, record: EncodeRecord, metrics: dict[str, float],
              frame_info: list[FrameInfo] | None = None,
              frame_metrics: list[dict[str, float]] | None = None) -> None:
        if (metrics or frame_info) and not record.passed:
            raise ValueError("metrics must not be stored for a stream that failed conformance")
        with self.lock, self.conn:
            self.conn.execute(
                "DELETE FROM encodes WHERE encoder=? AND structure=? AND clip=? AND quantizer=? "
                "AND frames=? AND platform=? AND fingerprint=?",
                (record.encoder, record.structure, record.clip, record.quantizer,
                 record.frames, record.platform, record.fingerprint))
            cur = self.conn.execute(
                "INSERT INTO encodes (encoder, structure, clip, quantizer, frames, bytes, kbps, "
                "conformance, conformance_reason, gate_level, platform, fingerprint, "
                "harness_commit, created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.encoder, record.structure, record.clip, record.quantizer, record.frames,
                 record.bytes, record.kbps, "pass" if record.passed else "fail", record.reason,
                 record.gate_level, record.platform, record.fingerprint, record.harness_commit,
                 time.time()))
            self.conn.executemany(
                "INSERT INTO metrics (encode_id, name, value) VALUES (?,?,?)",
                [(cur.lastrowid, name, value) for name, value in metrics.items()])
            if frame_info:
                self._insert_frames(cur.lastrowid, frame_info, frame_metrics or [])

    def _insert_frames(self, encode_id: int, frame_info: list[FrameInfo],
                       frame_metrics: list[dict[str, float]]) -> None:
        if frame_metrics and len(frame_metrics) != len(frame_info):
            raise ValueError(f"{len(frame_info)} coded frames but {len(frame_metrics)} "
                             f"per-frame metric rows")
        rows = []
        for i, info in enumerate(frame_info):
            m = frame_metrics[i] if frame_metrics else {}
            rows.append((encode_id, i, info.bytes, info.frame_type, info.temporal_id, info.qp,
                         m.get("psnr_y"), m.get("psnr_u"), m.get("psnr_v"),
                         m.get("vmaf"), m.get("vmaf_neg"), m.get("vmaf_v1"), m.get("ms_ssim")))
        self.conn.executemany(
            "INSERT INTO frames (encode_id, frame, bytes, frame_type, temporal_id, qp, "
            "psnr_y, psnr_u, psnr_v, vmaf, vmaf_neg, vmaf_v1, ms_ssim) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)

    def curve_is_current(self, encoder: str, structure: str, clip: str, frames: int,
                         platform: str, fingerprint: str,
                         failures_count_as_current: bool = True) -> bool:
        """True if this curve was fully measured with the current fingerprint.

        By default curves with a conformance failure count as current:
        re-running an unchanged encoder would fail the same way, and the
        failure stays listed. Pass failures_count_as_current=False to retry
        them (e.g. after an environment problem such as a full disk).
        """
        rows = self._query(
            "SELECT e.conformance, "
            "(SELECT COUNT(*) FROM frames f WHERE f.encode_id = e.id) "
            "FROM encodes e WHERE encoder=? AND structure=? AND clip=? AND frames=? "
            "AND platform=? AND fingerprint=?", (encoder, structure, clip, frames, platform,
                                                 fingerprint))
        if not rows:
            return False
        if any(status == "fail" for status, _ in rows):
            return failures_count_as_current
        # Passed points above the source cap are stored without per-frame data
        # (they are gated, never scored); they neither count nor block.
        scored = [n for _, n in rows if n]
        return len(rows) >= MIN_CURVE_POINTS and all(n == frames for n in scored)

    def relabel_curve(self, encoder: str, structure: str, clip: str, frames: int,
                      platform: str, old_fingerprint: str, new_fingerprint: str) -> int:
        """Re-key a curve to a new fingerprint (proven identical output); rows changed."""
        with self.lock, self.conn:
            # A partial curve already under the new key is superseded.
            self.conn.execute(
                "DELETE FROM encodes WHERE encoder=? AND structure=? AND clip=? AND frames=? "
                "AND platform=? AND fingerprint=?",
                (encoder, structure, clip, frames, platform, new_fingerprint))
            return self.conn.execute(
                "UPDATE encodes SET fingerprint=? WHERE encoder=? AND structure=? AND clip=? "
                "AND frames=? AND platform=? AND fingerprint=?",
                (new_fingerprint, encoder, structure, clip, frames, platform,
                 old_fingerprint)).rowcount

    def clear_curve(self, encoder: str, structure: str, clip: str, frames: int,
                    platform: str, fingerprint: str) -> None:
        """Forget one fingerprint's curve before re-measuring it, so old points never mix in.

        Other fingerprints' curves (other builds, other agents) are kept.
        """
        with self.lock, self.conn:
            self.conn.execute(
                "DELETE FROM encodes WHERE encoder=? AND structure=? AND clip=? AND frames=? "
                "AND platform=? AND fingerprint=?",
                (encoder, structure, clip, frames, platform, fingerprint))

    def latest_fingerprint(self, encoder: str, structure: str, clip: str, platform: str,
                           frames: int | None = None) -> str | None:
        """The fingerprint of the most recently stored encode of one curve."""
        rows = self._query(
            "SELECT fingerprint FROM encodes WHERE encoder=? AND structure=? AND clip=? "
            "AND platform=? AND (? IS NULL OR frames=?) ORDER BY created DESC LIMIT 1",
            (encoder, structure, clip, platform, frames, frames))
        return rows[0][0] if rows else None

    def rd_points(self, encoder: str, structure: str, clip: str, platform: str,
                  metric: str, frames: int | None = None,
                  fingerprint: str | None = None) -> list[tuple[float, float]]:
        """(kbps, metric) for every *passing* encode of one curve, by rate.

        `frames` selects one encode length when a namespace holds several
        (e.g. VAL-150 and full-length curves of the same clip); `fingerprint`
        one configuration (default: the most recently measured).
        """
        fingerprint = fingerprint or self.latest_fingerprint(encoder, structure, clip,
                                                             platform, frames)
        rows = self._query(
            "SELECT e.kbps, m.value FROM encodes e JOIN metrics m ON m.encode_id = e.id "
            "WHERE e.encoder=? AND e.structure=? AND e.clip=? AND e.platform=? "
            "AND e.conformance='pass' AND m.name=? AND (? IS NULL OR e.frames=?) "
            "AND e.fingerprint=? ORDER BY e.kbps",
            (encoder, structure, clip, platform, metric, frames, frames, fingerprint))
        return [(float(r), float(q)) for r, q in rows]

    def encoders(self, structure: str, platform: str) -> list[str]:
        rows = self._query(
            "SELECT DISTINCT encoder FROM encodes WHERE structure=? AND platform=? "
            "ORDER BY encoder", (structure, platform))
        return [r[0] for r in rows]

    def failures(self, clips: list[str] | None = None) -> list[tuple[str, str, str, int, str]]:
        rows = self._query(
            "SELECT encoder, structure, clip, quantizer, conformance_reason FROM encodes "
            "WHERE conformance='fail' ORDER BY encoder, structure, clip, quantizer")
        return [r for r in rows if clips is None or r[2] in clips]

    def gate_levels(self, clips: list[str] | None = None) -> dict[str, dict[str, int]]:
        """Passed encodes per encoder and gate level ('' = before levels were recorded)."""
        rows = self._query("SELECT encoder, clip, gate_level FROM encodes WHERE conformance='pass'")
        levels: dict[str, dict[str, int]] = {}
        for encoder, clip, level in rows:
            if clips is None or clip in clips:
                counts = levels.setdefault(encoder, {})
                counts[level] = counts.get(level, 0) + 1
        return levels

    # --- speed -------------------------------------------------------------

    def store_speed(self, record: SpeedRecord) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "DELETE FROM speed WHERE encoder=? AND structure=? AND clip=? AND platform=?",
                (record.encoder, record.structure, record.clip, record.platform))
            self.conn.execute(
                "INSERT INTO speed (encoder, structure, clip, quantizer, frames, platform, core, "
                "runs, ms_per_frame, cpu_ms_per_frame, instructions, cycles, fingerprint, "
                "harness_commit, created) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record.encoder, record.structure, record.clip, record.quantizer, record.frames,
                 record.platform, record.core, record.runs, record.ms_per_frame,
                 record.cpu_ms_per_frame, record.instructions, record.cycles,
                 record.fingerprint, record.harness_commit, time.time()))

    def speed_is_current(self, encoder: str, structure: str, clip: str, platform: str,
                         fingerprint: str) -> bool:
        rows = self._query(
            "SELECT fingerprint FROM speed WHERE encoder=? AND structure=? AND clip=? "
            "AND platform=?", (encoder, structure, clip, platform))
        return bool(rows) and rows[0][0] == fingerprint

    def speed_rows(self, structure: str, clip: str, platform: str) -> list[tuple]:
        return self._query(
            "SELECT encoder, quantizer, frames, ms_per_frame, cpu_ms_per_frame, instructions "
            "FROM speed WHERE structure=? AND clip=? AND platform=? ORDER BY cpu_ms_per_frame",
            (structure, clip, platform))

    def speed_platforms(self) -> list[str]:
        return [r[0] for r in self._query("SELECT DISTINCT platform FROM speed ORDER BY platform")]

    def swept_points(self, encoder: str, structure: str, clip: str,
                     platform: str) -> list[tuple[int, float]]:
        """(quantizer, kbps) of every passing encode of one curve (latest fingerprint)."""
        fingerprint = self.latest_fingerprint(encoder, structure, clip, platform)
        return [(int(q), float(k)) for q, k in self._query(
            "SELECT quantizer, kbps FROM encodes WHERE encoder=? AND structure=? AND clip=? "
            "AND platform=? AND conformance='pass' AND fingerprint=?",
            (encoder, structure, clip, platform, fingerprint))]

    def curve_encodes(self, encoder: str, structure: str, clip: str,
                      platform: str) -> list[tuple[int, int, int, float, str, str]]:
        """(id, quantizer, frames, kbps, conformance, reason) of every encode of a curve.

        The curve is its most recently measured fingerprint's.
        """
        fingerprint = self.latest_fingerprint(encoder, structure, clip, platform)
        return self._query(
            "SELECT id, quantizer, frames, kbps, conformance, conformance_reason FROM encodes "
            "WHERE encoder=? AND structure=? AND clip=? AND platform=? AND fingerprint=? "
            "ORDER BY kbps", (encoder, structure, clip, platform, fingerprint))

    def encode_metrics(self, encode_id: int) -> dict[str, float]:
        return {name: float(value) for name, value in self._query(
            "SELECT name, value FROM metrics WHERE encode_id=?", (encode_id,))}

    def frame_rows(self, encode_id: int) -> list[tuple]:
        """(frame, bytes, frame_type, psnr_y, vmaf_v1) per coded frame, in order."""
        return self._query(
            "SELECT frame, bytes, frame_type, psnr_y, vmaf_v1 FROM frames WHERE encode_id=? "
            "ORDER BY frame", (encode_id,))

    def _query(self, sql: str, params: tuple = ()) -> list[tuple]:
        with self.lock:
            return self.conn.execute(sql, params).fetchall()
