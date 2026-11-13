"""The maintained reference (anchor) evaluation: `python3 -m nceval reference`.

Everything is driven by one checked-in config (eval/configs/reference.json):
clip sets, anchors, coding structures, metrics, report anchors and the speed
runs. Every step is idempotent:

  clips   extract.py / add_noise.py reuse outputs that are already complete;
  sweep   curves measured with the current fingerprint are skipped
          (nceval.fingerprint: pinned tool versions, adapter flags,
          measurement chain); missing per-frame data is backfilled from the
          kept bitstreams;
  speed   re-timed only when an encoder's fingerprint changed (or --force-speed);
          optionally also on a remote machine (--platform-remote jetson);
  report  always regenerated from the database, written as a dated markdown
          file under $NC_DATA_ROOT/reports/ and summarised on stdout.
"""

from __future__ import annotations

import json
import posixpath
import platform
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from math import exp, log
from pathlib import Path

from . import localenv, report, speed
from .clips import Clip, ClipSet, load_clipset
from .config import Paths
from .encoders import ANCHORS, EncoderAdapter
from .results import ResultsDb
from .sweep import REFERENCE_TARGETS_KBPS, Sweep, SweepConfig, nominal_targets

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "eval" / "configs" / "reference.json"
STEPS = ("sweep", "speed", "report")
LOCAL_PLATFORM = platform.machine()


@dataclass(frozen=True)
class Suite:
    name: str
    selection: Path
    noise_sources: list[str]
    noise_gains: list[str]
    anchors: list[str]
    structures: list[SweepConfig]
    report_splits: list[str]
    per_clip: bool


@dataclass(frozen=True)
class Selection:
    """Optional restriction of a run (e.g. to get a priority result first).

    Empty means "everything in the config". Anchors are run in the order given,
    so the first-named encoder's curves finish first.
    """
    anchors: tuple[str, ...] = ()
    suites: tuple[str, ...] = ()
    structures: tuple[str, ...] = ()  # structure tags, e.g. "ip10s-tl1"

    def pick_anchors(self, suite: Suite) -> list[str]:
        if not self.anchors:
            return suite.anchors
        return [a for a in self.anchors if a in suite.anchors]

    def pick_suite(self, suite: Suite) -> bool:
        return not self.suites or suite.name in self.suites

    def pick_structure(self, cfg: SweepConfig) -> bool:
        return not self.structures or cfg.structure_tag in self.structures


class Reference:
    def __init__(self, config_path: Path, paths: Paths, db: ResultsDb, harness_commit: str,
                 selection: Selection = Selection()):
        self.selection = selection
        self.config = json.loads(config_path.read_text())
        self.config_path = config_path
        self.paths = paths
        self.db = db
        self.harness_commit = harness_commit
        workers = self.config.get("sweep_workers", 8)
        self.suites = {s["name"]: _parse_suite(s, workers) for s in self.config["suites"]}
        for suite in self.suites.values():
            unknown = [a for a in suite.anchors if a not in ANCHORS]
            if unknown:
                raise SystemExit(f"suite {suite.name}: unknown anchors {unknown}")
        self.sweeper = Sweep(paths, db, harness_commit)

    # --- steps -------------------------------------------------------------

    def run(self, steps: tuple[str, ...], remote: str | None, force_speed: bool) -> Path | None:
        if "sweep" in steps:
            self.prepare_clips()
            self.sweep()
        if "speed" in steps:
            self.speed_local(force_speed)
            if remote:
                self.speed_remote(remote, force_speed)
        if "report" in steps:
            return self.write_report()
        return None

    def prepare_clips(self) -> None:
        for suite in self._selected_suites():
            _run_script(["clipset/extract.py", str(suite.selection)])
            if suite.noise_sources:
                args = ["clipset/add_noise.py", str(self.clipset_file(suite))]
                for name in suite.noise_sources:
                    args += ["--clip", name]
                for gain in suite.noise_gains:
                    args += ["--gain", gain]
                _run_script(args)

    def sweep(self) -> None:
        for suite in self._selected_suites():
            clips = list(self.clipset(suite).clips)
            encoders = [ANCHORS[a] for a in self.selection.pick_anchors(suite)]
            for cfg in suite.structures:
                if not self.selection.pick_structure(cfg):
                    continue
                print(f"== sweep {suite.name} {cfg.structure_tag}: {len(encoders)} encoders x "
                      f"{len(clips)} clips", flush=True)
                summary = self.sweeper.run(encoders, clips, cfg)
                print(f"== {suite.name} {cfg.structure_tag}: {summary}", flush=True)

    def speed_local(self, force: bool) -> None:
        spec = self.config["speed"]
        for suite, cfg, clip, encoders in self._speed_jobs():
            fingerprints = {e.name: self.sweeper.fingerprint(e, cfg, clip) for e in encoders}
            if not force and all(self.db.speed_is_current(e.name, cfg.structure_tag, clip.name,
                                                          LOCAL_PLATFORM, fingerprints[e.name])
                                 for e in encoders):
                continue
            print(f"== speed {LOCAL_PLATFORM} {suite.name}/{clip.name}", flush=True)
            quantizers = self._speed_quantizers(encoders, cfg, clip)
            results = speed.measure(encoders, clip, cfg.structure_for(clip), quantizers,
                                    clip.frames, spec["core"], spec["runs"], self.paths)
            speed.store(results, self.db, cfg.structure_tag, clip, clip.frames, LOCAL_PLATFORM,
                        spec["core"], spec["runs"], fingerprints, self.harness_commit)

    def speed_remote(self, name: str, force: bool) -> None:
        remote = _resolve_remote(self.config["remotes"][name])
        spec = self.config["speed"]
        remote_platform = _ssh(remote, "uname -m").strip()
        free_mb = int(_ssh(remote, "free -m | awk '/Mem:/ {print $7}'").strip())
        if free_mb < remote["min_free_ram_mb"]:
            print(f"!! {name}: only {free_mb} MB RAM available (need "
                  f"{remote['min_free_ram_mb']}); skipping remote speed runs")
            return
        self._sync_code(remote)
        for suite, cfg, clip, encoders in self._speed_jobs():
            fingerprints = {e.name: self.sweeper.fingerprint(e, cfg, clip) for e in encoders}
            if not force and all(self.db.speed_is_current(e.name, cfg.structure_tag, clip.name,
                                                          remote_platform, fingerprints[e.name])
                                 for e in encoders):
                continue
            print(f"== speed {name} ({remote_platform}) {suite.name}/{clip.name}", flush=True)
            self._sync_clip(remote, suite, clip)
            quantizers = self._speed_quantizers(encoders, cfg, clip)
            remote_clipset = f"{remote['data_root']}/clips/{suite.name}.json"
            cmd = (f"cd {shlex.quote(remote['workdir'])}/eval && "
                   f"NC_DATA_ROOT={shlex.quote(remote['data_root'])} python3 -m nceval speed "
                   f"--clipset {shlex.quote(remote_clipset)} --clip {clip.name} "
                   f"--encoders {','.join(e.name for e in encoders)} "
                   f"--quantizers {_format_quantizers(quantizers)} "
                   f"--intra-period-seconds {cfg.intra_period_seconds:g} "
                   + ("--single-keyframe " if cfg.single_keyframe else "") +
                   f"--temporal-layers {cfg.temporal_layers} "
                   f"--core {remote['core']} --runs {spec['runs']} --json")
            results = [speed.SpeedResult(**r) for r in json.loads(_ssh(remote, cmd))]
            speed.store(results, self.db, cfg.structure_tag, clip, clip.frames, remote_platform,
                        remote["core"], spec["runs"], fingerprints, self.harness_commit)

    def write_report(self) -> Path:
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M")
        out_dir = self.paths.data_root / "reports"
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"reference-{stamp}.md"
        headline, body = self._report_sections()
        out.write_text("\n".join([
            f"# Reference evaluation — {stamp}", "",
            f"Harness commit `{self.harness_commit}`, config "
            f"`{self.config_path.relative_to(REPO_ROOT)}`, BD-rate platform {LOCAL_PLATFORM}.",
            "BD-rate: negative = fewer bits than the anchor at equal quality; means are over "
            "clips (each clip weighs the same); clips with too little curve overlap are "
            "excluded and counted.", "",
            *headline, "", *body]))
        print("\n".join(headline))
        print(f"\nfull report: {out}")
        return out

    # --- report building ---------------------------------------------------

    def _report_sections(self) -> tuple[list[str], list[str]]:
        metrics = self.config["metrics"]
        anchors = self.config["report_anchors"]
        head = self.config["headline"]
        head_suite = self.suites[head["suite"]]
        head_cfg = _parse_structure(head["structure"], 1)

        headline = [f"## Headline: {head_suite.name}, {head_cfg.structure_tag}, all clips", ""]
        clipset = self.clipset(head_suite)
        for anchor in anchors:
            headline += [f"BD-rate vs **{anchor}** (overall, all clips):", "",
                         "| encoder | " + " | ".join(metrics) + " |",
                         "|---|" + "---:|" * len(metrics)]
            per_metric = {m: report.summarise(report.compute(
                self.db, clipset, anchor, m, head_cfg.structure_tag, LOCAL_PLATFORM,
                head_suite.anchors)) for m in metrics}
            for enc in head_suite.anchors:
                if enc == anchor:
                    continue
                headline.append(f"| {enc} | " + " | ".join(
                    _pct(per_metric[m].get(enc)) for m in metrics) + " |")
            headline.append("")
        headline += self._speed_summary()

        body: list[str] = []
        for suite in self._selected_suites():
            clipset = self.clipset(suite)
            for cfg in suite.structures:
                body += [f"## {suite.name}: {cfg.structure_tag}", ""]
                for split in suite.report_splits:
                    chosen = _split(clipset, split)
                    for anchor in anchors:
                        for metric in metrics:
                            rows = report.compute(self.db, chosen, anchor, metric,
                                                  cfg.structure_tag, LOCAL_PLATFORM,
                                                  suite.anchors)
                            body += _markdown_bd_table(
                                rows, f"{split} clips ({len(chosen.clips)}), vs {anchor}, "
                                      f"{metric}", suite.per_clip)
                body.append("")
            body += self._noise_section(suite, clipset)
            body += ["### Conformance", "",
                     report.format_failures(self.db, [c.name for c in clipset.clips]), ""]
        body += self._speed_tables()
        return headline, body

    def _noise_section(self, suite: Suite, clipset: ClipSet) -> list[str]:
        """Synthetic-noise clips: clean-reference quality, not BD-rate (see report.py)."""
        if not any(c.content_class in report.SEPARATE_CLASSES for c in clipset.clips):
            return []
        scale = lambda clip: nominal_targets(clip)[0] / REFERENCE_TARGETS_KBPS[0]  # noqa: E731
        fixed = report.NOISE_REPORT_KBPS_720P30
        lines = [f"### {suite.name}: synthetic noise (scored against the clean original)", "",
                 "No BD-rate: against the clean reference, quality is not monotonic in "
                 "bitrate. Best = highest clean-reference quality over the sweep (and the "
                 f"bitrate where it occurs); fixed-rate columns are interpolated at "
                 f"{'/'.join(f'{k:g}' for k in fixed)} kbps (720p30-equivalent).", ""]
        for cfg in suite.structures:
            for metric in ("psnr_yuv", "vmaf_neg" if cfg.measurement_profile == "legacy" else "vmaf_v1"):
                rows = report.noise_summary(self.db, clipset, metric, cfg.structure_tag,
                                            LOCAL_PLATFORM, suite.anchors, scale)
                if not rows:
                    continue
                lines += [f"**{cfg.structure_tag}, {metric}**", "",
                          "| clip | encoder | best | at kbps | " +
                          " | ".join(f"@{k:g}" for k in fixed) + " |",
                          "|---|---|---:|---:|" + "---:|" * len(fixed)]
                for r in rows:
                    cells = [f"{r.at_fixed_kbps[k]:.2f}" if r.at_fixed_kbps[k] is not None
                             else "-" for k in fixed]
                    lines.append(f"| {r.clip} | {r.encoder} | {r.best_quality:.2f} | "
                                 f"{r.best_kbps:.0f} | " + " | ".join(cells) + " |")
                lines.append("")
        return lines

    def _speed_summary(self) -> list[str]:
        anchor = self.config["speed"]["anchor"]
        lines = [f"Encode speed (CPU time, one pinned core), geometric mean ratio vs {anchor}:",
                 "", "| set | platform | " + " | ".join(
                     a for a in self._all_speed_encoders()) + " |",
                 "|---|---|" + "---:|" * len(self._all_speed_encoders())]
        for platform_name in self._speed_platforms():
            for suite, cfg, clips in self._speed_sets():
                ratios = self._speed_ratios(cfg, clips, platform_name, anchor)
                if ratios:
                    lines.append(f"| {suite.name} | {platform_name} | " + " | ".join(
                        f"{ratios[e]:.2f}x" if e in ratios else "-"
                        for e in self._all_speed_encoders()) + " |")
        return lines

    def _speed_tables(self) -> list[str]:
        anchor = self.config["speed"]["anchor"]
        lines = ["## Speed detail", ""]
        for platform_name in self._speed_platforms():
            for suite, cfg, clips in self._speed_sets():
                for clip in clips:
                    rows = self.db.speed_rows(cfg.structure_tag, clip.name, platform_name)
                    if not rows:
                        continue
                    base = next((r[4] for r in rows if r[0] == anchor), None)
                    lines += [f"### {platform_name}: {clip.name} ({clip.width}x{clip.height}"
                              f"@{clip.fps:g}, {rows[0][2]} frames, {cfg.structure_tag})", "",
                              "| encoder | q | wall ms/f | cpu ms/f | x anchor |",
                              "|---|---:|---:|---:|---:|"]
                    for enc, q, _, wall, cpu, _ in rows:
                        ratio = f"{cpu / base:.2f}x" if base else "-"
                        lines.append(f"| {enc} | {q} | {wall:.2f} | {cpu:.2f} | {ratio} |")
                    lines.append("")
        conditions = speed.machine_conditions()
        if conditions:
            lines += [f"Local measurement conditions: " + "; ".join(conditions), ""]
        return lines

    def _speed_ratios(self, cfg: SweepConfig, clips: list[Clip], platform_name: str,
                      anchor: str) -> dict[str, float]:
        logs: dict[str, list[float]] = {}
        for clip in clips:
            rows = {r[0]: r[4] for r in self.db.speed_rows(cfg.structure_tag, clip.name,
                                                           platform_name)}
            if anchor not in rows:
                continue
            for enc, cpu in rows.items():
                logs.setdefault(enc, []).append(log(cpu / rows[anchor]))
        return {e: exp(sum(v) / len(v)) for e, v in logs.items()}

    def _speed_platforms(self) -> list[str]:
        return self.db.speed_platforms()

    def _all_speed_encoders(self) -> list[str]:
        names: list[str] = []
        for suite, _, _ in self._speed_sets():
            names += [a for a in suite.anchors if a not in names]
        return names

    # --- helpers -----------------------------------------------------------

    def clipset_file(self, suite: Suite) -> Path:
        name = json.loads(suite.selection.read_text())["name"]
        return self.paths.clips / f"{name}.json"

    def clipset(self, suite: Suite) -> ClipSet:
        path = self.clipset_file(suite)
        if not path.is_file():
            raise SystemExit(f"{path} missing: run 'nceval reference --only sweep' first")
        return load_clipset(path, self.paths.data_root)

    def _selected_suites(self) -> list[Suite]:
        return [s for s in self.suites.values() if self.selection.pick_suite(s)]

    def _speed_sets(self) -> list[tuple[Suite, SweepConfig, list[Clip]]]:
        sets = []
        for s in self.config["speed"]["sets"]:
            suite = self.suites[s["suite"]]
            if not self.selection.pick_suite(suite):
                continue
            clips = {c.name: c for c in self.clipset(suite).clips}
            sets.append((suite, _parse_structure(s["structure"], 1),
                         [clips[n] for n in s["clips"]]))
        return sets

    def _speed_jobs(self):
        speed_anchor = self.config["speed"]["anchor"]
        for suite, cfg, clips in self._speed_sets():
            if not self.selection.pick_structure(cfg):
                continue
            names = self.selection.pick_anchors(suite)
            if speed_anchor not in names:
                names = [speed_anchor, *names]  # ratios need the anchor timed alongside
            encoders = [ANCHORS[a] for a in names]
            for clip in clips:
                yield suite, cfg, clip, encoders

    def _speed_quantizers(self, encoders: list[EncoderAdapter], cfg: SweepConfig,
                          clip: Clip) -> dict[str, tuple[int, float]]:
        # The configured target is for 720p30; scale it like the sweep targets.
        scale = nominal_targets(clip)[0] / REFERENCE_TARGETS_KBPS[0]
        target = self.config["speed"]["target_kbps_720p30"] * scale
        return speed.choose_quantizers(self.db, encoders, cfg.structure_tag, clip,
                                       LOCAL_PLATFORM, target)

    def _sync_code(self, remote: dict) -> None:
        dest = f"{remote['host']}:{remote['workdir']}/"
        _ssh(remote, f"mkdir -p {shlex.quote(remote['workdir'])}/tools/anchors")
        subprocess.run(["rsync", "-a", "--delete", "--exclude", "__pycache__",
                        str(REPO_ROOT / "eval"), dest], check=True)
        subprocess.run(["rsync", "-a", str(REPO_ROOT / "tools" / "anchors" / "versions.env"),
                        f"{dest}tools/anchors/"], check=True)
        subprocess.run(["rsync", "-a", str(REPO_ROOT / "tools" / "openh264-harness"),
                        f"{dest}tools/"], check=True)

    def _sync_clip(self, remote: dict, suite: Suite, clip: Clip) -> None:
        data_root = remote["data_root"]
        rel = clip.yuv.relative_to(self.paths.data_root)
        _ssh(remote, f"mkdir -p {shlex.quote(str(Path(data_root) / rel.parent))}")
        subprocess.run(["rsync", "-a", str(clip.yuv), f"{remote['host']}:{data_root}/{rel}"],
                       check=True)
        subprocess.run(["rsync", "-a", str(self.clipset_file(suite)),
                        f"{remote['host']}:{data_root}/clips/"], check=True)


def _parse_suite(raw: dict, workers: int) -> Suite:
    return Suite(name=raw["name"], selection=REPO_ROOT / raw["selection"],
                 noise_sources=raw.get("noise_sources", []),
                 noise_gains=raw.get("noise_gains", []), anchors=raw["anchors"],
                 structures=[_parse_structure(s, workers) for s in raw["structures"]],
                 report_splits=raw.get("report_splits", ["all"]),
                 per_clip=raw.get("per_clip", False))


def _parse_structure(raw: dict, workers: int) -> SweepConfig:
    """One structure entry; {"single_keyframe": true} is the long-GOP set (no period)."""
    single = bool(raw.get("single_keyframe", False))
    return SweepConfig(intra_period_seconds=float(raw.get("intra_period_seconds", 0) if single
                                                  else raw["intra_period_seconds"]),
                       temporal_layers=int(raw["temporal_layers"]), workers=workers,
                       single_keyframe=single,
                       measurement_profile=raw.get("measurement_profile", "vmaf-v1"))


def _split(clipset: ClipSet, split: str) -> ClipSet:
    if split == "all":
        return clipset
    return ClipSet(name=clipset.name, clips=tuple(c for c in clipset.clips if c.split == split))


def _markdown_bd_table(rows: list[report.ClipBd], title: str, per_clip: bool) -> list[str]:
    classes = report.ordered_classes(rows)
    lines = [f"**{title}**", "", "| encoder | " + " | ".join(classes) + " | overall | excl. |",
             "|---|" + "---:|" * (len(classes) + 2)]
    for enc, s in sorted(report.summarise(rows).items()):
        cells = [f"{s.by_class[c]:.1f}%" if c in s.by_class else "-" for c in classes]
        lines.append(f"| {enc} | " + " | ".join(cells) +
                     f" | {_pct(s)} | {s.clips_excluded or ''} |")
    if per_clip:
        lines += ["", "| encoder | clip | BD-rate | overlap |", "|---|---|---:|---:|"]
        for r in sorted(rows, key=lambda r: (r.encoder, r.clip)):
            if r.result and r.result.bd_rate_percent is not None:
                flag = "" if r.result.reliable else " (unreliable)"
                lines.append(f"| {r.encoder} | {r.clip} | {r.result.bd_rate_percent:.1f}%"
                             f"{flag} | {r.result.overlap_fraction:.2f} |")
            else:
                lines.append(f"| {r.encoder} | {r.clip} | - | {r.note} |")
    lines.append("")
    return lines


def _pct(summary: report.EncoderSummary | None) -> str:
    return f"{summary.overall:.1f}%" if summary and summary.overall is not None else "-"


def _format_quantizers(quantizers: dict[str, tuple[int, float]]) -> str:
    return ",".join(f"{name}={q}:{kbps:.1f}" for name, (q, kbps) in quantizers.items())


def _run_script(args: list[str]) -> None:
    subprocess.run([sys.executable, *args], cwd=REPO_ROOT / "eval", check=True)


def _resolve_remote(remote: dict) -> dict:
    """A config remote made concrete: a `$NAME` host read from the local settings
    (nceval/localenv.py), and paths relative to the remote's home made absolute."""
    host = remote["host"]
    resolved = dict(remote, host=localenv.require(host[1:]) if host.startswith("$") else host)
    home = _ssh(resolved, 'printf %s "$HOME"')
    for key in ("workdir", "data_root"):
        resolved[key] = posixpath.join(home, resolved[key])
    return resolved


def _ssh(remote: dict, command: str) -> str:
    proc = subprocess.run(["ssh", "-o", "BatchMode=yes", remote["host"], command],
                          stdin=subprocess.DEVNULL, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ssh {remote['host']} failed ({proc.returncode}): "
                           f"{proc.stderr.strip()[-500:]}")
    return proc.stdout
