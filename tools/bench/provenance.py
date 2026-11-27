"""What exactly a BP-3 report measured: the build identity of every binary, the tools, the
machines and the datasets, from the runs' own records (owner, 2026-11-24: "ABSOLUTELY CRITICAL
that the version used is recorded"). `report.py` puts this block at the top of report.json and
report.txt, and V25 refuses a report whose identity is missing or mixed.

Sources, all written before or while the runs ran (nothing is re-run):
  - each run's `manifest.json`: the tools commit it was staged at, and every binary's SHA-256
    (V2 checked each box's binaries against it);
  - the anchor stores under `~/nanocodec-data/anchors/*/`: `SHA256SUMS` maps a hash to its
    store, `BUILD.json` gives our encoder's commit, version and build, `build-flags.json` each
    reference's source commit, flags, LTO/PGO and the toolchain;
  - the runs' driver logs: the instance types, zones and machine images the boxes ran on;
  - the dataset manifests at the run's tools commit (`git show`), hashed.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Callable

ANCHORS = Path.home() / "nanocodec-data" / "anchors"
REPO = Path(__file__).resolve().parents[2]
#: The reference binaries and the build-flags.json entry that built each.
TOOL_OF_BINARY = {"x264": "x264", "x265": "x265", "SvtAv1EncApp": "svtav1", "aomenc": "libaom",
                  "svc_encoder_rtc": "libaom", "vpxenc": "libvpx", "kvazaar": "kvazaar",
                  "vvencFFapp": "vvenc", "openh264-harness": "openh264"}
OURS = "nanocodec"


class ProvenanceRefusal(RuntimeError):
    """V25: a report's build identity is missing or not one identity."""


def build_index(anchors: Path = ANCHORS) -> dict[str, dict]:
    """SHA-256 -> {store, path} for every binary of every anchor store."""
    index = {}
    for sums in anchors.glob("*/SHA256SUMS*"):
        for line in sums.read_text().splitlines():
            if line.strip():
                digest, path = line.split(maxsplit=1)
                index[digest] = {"store": sums.parent.name, "path": path.strip()}
    return index


def store_records(store: str, anchors: Path = ANCHORS) -> tuple[dict, dict]:
    """(BUILD.json, build-flags.json) of a store ({} where absent)."""
    def load(name: str) -> dict:
        path = anchors / store / name
        return json.loads(path.read_text()) if path.exists() else {}
    return load("BUILD.json"), load("build-flags.json")


def machines(run: Path) -> dict:
    """Instance types, zones and images the run's boxes ran on, from its driver logs."""
    types, zones, images = set(), set(), set()
    for log in run.glob("series/*/*.log"):
        text = log.read_text(errors="replace")
        for instance_type, zone in re.findall(r"launched \S+ \((\S+)\) in (\S+)", text):
            types.add(instance_type)
            zones.add(zone)
        images |= set(re.findall(r"image (ami-[0-9a-f]+)", text))
    return {"types": sorted(types), "zones": sorted(zones),
            "images": sorted(images) or "not recorded (logged from 2026-11-24 16:30)"}


def dataset_manifests(commit: str, inputs: set[str]) -> dict[str, str]:
    """{manifest path: SHA-256} of the dataset manifests at `commit` that hold the runs' clips."""
    listing = subprocess.run(["git", "-C", str(REPO), "ls-tree", "--name-only", commit,
                              "eval/clipset/"], capture_output=True, text=True).stdout.split()
    found = {}
    for path in (p for p in listing if p.endswith(".manifest.json")):
        text = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{path}"],
                              capture_output=True).stdout
        try:
            names = {c["name"] for c in json.loads(text)["clips"]}
        except (ValueError, KeyError, TypeError):
            continue
        if names & inputs:
            found[path] = hashlib.sha256(text).hexdigest()
    return found


def binary_of(job: dict) -> str:
    """The binary a job runs (its argv's first element, without the box's {BIN})."""
    return Path(job["argv"][0].replace("{BIN}/", "")).name


def provenance(runs: list[Path], timing_runs: list[Path] = (),
               anchors: Path | None = None, keep: Callable[[str], bool] | None = None) -> dict:
    """The report's provenance block; V25 refuses a missing or mixed identity. `keep` names
    the families the report takes from these runs (a mixed report, `report.Part`): only their
    binaries are the block's encoders."""
    return identity(runs, timing_runs, anchors, keep)[0]


def identity(runs: list[Path], timing_runs: list[Path] = (), anchors: Path | None = None,
             keep: Callable[[str], bool] | None = None) -> tuple[dict, dict[str, str]]:
    """(the provenance block, every binary the runs' manifests record: name -> SHA-256)."""
    anchors = anchors or ANCHORS
    manifests = {str(r): json.loads((r / "manifest.json").read_text())
                 for r in [*runs, *timing_runs]}
    used, inputs = set(), set()
    for run in [*runs, *timing_runs]:
        for job in json.loads((run / "jobs.json").read_text()):
            if keep is None or keep(job["family"]):
                used.add(binary_of(job))
                inputs.add(job["input"])
    binaries: dict[str, str] = {}
    for run, manifest in manifests.items():
        if not manifest.get("binaries") or not manifest.get("procedure_commit"):
            raise ProvenanceRefusal(f"V25: {run} records no binaries or tools commit")
        for name, digest in manifest["binaries"].items():
            if binaries.setdefault(name, digest) != digest:
                raise ProvenanceRefusal(f"V25: {name} differs between the runs "
                                        f"({binaries[name][:12]} and {digest[:12]} in {run})")
    index = build_index(anchors)
    encoders = {}
    for name in sorted(used & set(binaries)):
        where = index.get(binaries[name])
        if where is None:
            raise ProvenanceRefusal(f"V25: {name} ({binaries[name][:12]}) has no build record "
                                    "in any anchor store")
        build, flags = store_records(where["store"], anchors)
        entry = {"sha256": binaries[name], "store": where["store"]}
        # our encoder, or a history day's build of it (nanocodec-<commit>: its store's BUILD.json)
        if name == OURS or name.startswith(OURS + "-"):
            if not build.get(OURS, {}).get("commit"):
                raise ProvenanceRefusal(f"V25: store {where['store']} records no commit for "
                                        "our encoder")
            entry.update(build[OURS])
        else:
            tool = flags.get("tools", {}).get(TOOL_OF_BINARY.get(name, name))
            if not tool or not tool.get("commit"):
                raise ProvenanceRefusal(f"V25: {name} has no source commit in "
                                        f"{where['store']}/build-flags.json")
            entry.update({k: tool[k] for k in ("commit", "ref", "lto", "pgo", "compile_flags")
                          if k in tool})
        entry["toolchain"] = flags.get("compiler", {})
        entry["profile"] = flags.get("profile")
        encoders[name] = entry
    other = {name: {"sha256": digest, **index.get(digest, {"store": "not in an anchor store"})}
             for name, digest in sorted(binaries.items()) if name not in encoders}
    commits = {run: m["procedure_commit"] for run, m in manifests.items()}
    first_commit = next(iter(commits.values()))
    return ({"encoders": encoders, "other_binaries": other, "tools_commits": commits,
             "machines": {run: machines(Path(run)) for run in commits},
             "datasets": dataset_manifests(first_commit, inputs),
             "quality_runs": [str(r) for r in runs],
             "timing_runs": [str(r) for r in timing_runs]}, binaries)


def merged(parts: list[tuple[dict, dict[str, str], bool]]) -> dict:
    """One block for a mixed report: each part is (its block, its recorded binaries, whether
    it takes our families). V25 per family: every encoder has one identity across the parts,
    and every binary that no ours-part uses (the competitors', the decoders', the scorer's)
    is byte-identical in every part that records it, so the reused competitor points ran the
    binaries the new runs carry. Only our encoder may differ: the parts that do not take our
    families hold the superseded build, which the block does not list."""
    ours_used = {OURS} if any(ours for _, _, ours in parts) else set()
    recorded: dict[str, str] = {}
    for _, binaries, _ in parts:
        for name, digest in binaries.items():
            if name in ours_used:
                continue
            if recorded.setdefault(name, digest) != digest:
                raise ProvenanceRefusal(f"V25: {name} differs between the report's parts "
                                        f"({recorded[name][:12]} and {digest[:12]})")
    encoders: dict[str, dict] = {}
    for block, _, _ in parts:
        for name, entry in block["encoders"].items():
            if encoders.setdefault(name, entry)["sha256"] != entry["sha256"]:
                raise ProvenanceRefusal(f"V25: {name} has two identities in the report "
                                        f"({encoders[name]['sha256'][:12]} and "
                                        f"{entry['sha256'][:12]})")
    other = {name: entry for block, _, _ in parts
             for name, entry in block["other_binaries"].items() if name not in encoders}

    def union(key: str) -> list[str]:
        return list(dict.fromkeys(r for block, _, _ in parts for r in block[key]))
    return {"encoders": dict(sorted(encoders.items())), "other_binaries": dict(sorted(other.items())),
            "tools_commits": {k: v for block, _, _ in parts for k, v in block["tools_commits"].items()},
            "machines": {k: v for block, _, _ in parts for k, v in block["machines"].items()},
            "datasets": {k: v for block, _, _ in parts for k, v in block["datasets"].items()},
            "quality_runs": union("quality_runs"), "timing_runs": union("timing_runs")}


PROCEDURE = "docs/benchmark-procedure.md"


def procedure_version(commit: str | None = None) -> str:
    """The procedure version (e.g. BP-3.13) in the procedure's title at `commit`, or in the
    working tree's (the report code's own) when None."""
    if commit is None:
        text = (REPO / PROCEDURE).read_text()
    else:
        text = subprocess.run(["git", "-C", str(REPO), "show", f"{commit}:{PROCEDURE}"],
                              capture_output=True, text=True).stdout
    found = re.search(r"\((BP-[0-9.]+)[,)]", text.splitlines()[0] if text else "")
    return found.group(1) if found else "unknown"


def procedure(block: dict) -> dict:
    """The procedure versions a report states (owner, 2026-11-24: the pages read them): each
    run's, from its tools commit, and the report code's; they differ where a report was
    written by later code than its runs."""
    runs = {run: procedure_version(commit) for run, commit in block["tools_commits"].items()}
    return {"runs": sorted(set(runs.values())), "by_run": runs, "report": procedure_version()}


def text(block: dict) -> list[str]:
    """The provenance block as the report's first lines."""
    ours = block["encoders"].get(OURS, {})
    lines = [f"PROVENANCE: ours {ours.get('version', '?')} ({ours.get('build', '?')}), "
             f"sha256 {ours.get('sha256', '?')[:16]}"]
    for name, e in block["encoders"].items():
        if name != OURS:
            lines.append(f"  {name}: {e.get('ref', '')} commit {e['commit'][:12]}, LTO "
                         f"{e.get('lto')}, PGO {e.get('pgo')}, {e.get('toolchain', {}).get('cc', '?')}")
    lines.append("  tools commits: " + ", ".join(sorted({c[:12] for c in
                                                        block["tools_commits"].values()})))
    if "procedure" in block:
        lines.append(f"  procedure: runs {', '.join(block['procedure']['runs'])}; report code "
                     f"{block['procedure']['report']}")
    types = sorted({t for m in block["machines"].values() for t in m["types"]})
    lines.append(f"  machines: {', '.join(types)}; datasets: " + ", ".join(
        f"{Path(p).name} {h[:12]}" for p, h in block["datasets"].items()))
    return lines
