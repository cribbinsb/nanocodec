#!/usr/bin/env python3
"""Summarise one release's committed conformance results into summary.json.

A release's decoder conformance runs (tools/ci/conformance.py --json per
native suite, tools/ci/vvc_conformance.py --json for the official-MD5 VVC
gate, tools/ci/wasm_conformance.sh with JSON_DIR for the browser module) are
committed under docs/results/conformance-<date>/. This script reads them with
the manifests and writes that directory's summary.json: per suite the pass
count, its denominator and what the denominator is (which part of the
official suite, what is left out and why), the reference decoder and the
failing streams. Published conformance counts are taken from summary.json.

    python3 -I tools/ci/conformance_summary.py docs/results/conformance-2026-11-23 \
        --commit 98069a8b3
"""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CI = ROOT / "tools/ci"

#: Native suites: result file stem -> (manifest, reference decoder, denominator).
NATIVE = {
    "h264": ("h264_conformance.json", "JM 19 ldecod (whole and per-picture MD5); FFmpeg's whole-output MD5 "
             "recorded as a cross-check",
             "ITU-T H.264.1: the AVCv1, FRExt and Professional-profiles (High 10) zips, 193 streams fetched "
             "(tools/dev/conformance_fetch.py --suite h264); the 3 FRExt 4:2:2 streams FREXT1-3_TANDBERG_A "
             "are not in the manifest; High 4:2:2 / 4:4:4 zips (Hi422, PPH422, PPH444, PPCV444), MVC and "
             "SVC are not fetched. Required: tools/ci/h264_conformance_passing.txt"),
    "h264-x264": ("h264_conformance_x264.json", "JM 19 ldecod; FFmpeg cross-check",
                  "8 x264 streams for I/P-only High tools (tools/dev/h264_conformance_supplement.sh), "
                  "not part of the ITU-T suite"),
    "hevc": ("hevc_conformance_v1.json", "HM TAppDecoder (whole and per-picture MD5)",
             "the whole JCT-VC HEVC_v1 set (ITU-T H.265.1): 147 Main, Main 10 and Main Still Picture streams"),
    "av1": ("av1_conformance_libaom.json", "aomdec --i420 (libaom v3.15.3)",
            "libaom v3.15.3 Main-profile test vectors: the 176 av1-1-b8-* and 66 av1-1-b10-* "
            "(tools/conformance/av1_manifest.txt), film grain applied"),
    "av1-supplementary": ("av1_conformance_supplementary.json",
                          "aomdec --i420 (libaom v3.15.3); each stream kept only where aomdec and dav1d "
                          "1.5.4 agree on every frame",
                          "28 short streams of libaom v3.15.3 (aomenc; svc_encoder_rtc with its ROI map) "
                          "and SVT-AV1 v4.3.0 for Main-profile tools the libaom vectors lack: quantizer "
                          "matrices, superblock quantizer and loop filter deltas, segmentation (AQ modes, "
                          "ROI loop filter / reference / skip features), lossless, odd frame sizes, switch "
                          "frames (tools/dev/av1_conformance_supplement.sh); not an official suite"),
    "av1-profiles": ("av1_conformance_profiles.json",
                     "aomdec (libaom v3.15.3): --i420 for 4:2:0 and monochrome, --rawvideo for 4:2:2 "
                     "and 4:4:4; dav1d 1.5.4 as a cross-check",
                     "27 High and Professional profile streams of aomenc and ffmpeg's libaom "
                     "(tools/dev/av1_profile_streams.sh); not an official suite"),
    "vp9": ("vp9_conformance_libvpx.json", "vpxdec-hbd --i420 (libvpx v1.16.0); FFmpeg cross-check "
            "(disagrees on the 64 resize vectors, which FFmpeg scales to the first size: those are checked "
            "against vpxdec alone)",
            "libvpx v1.16.0 VP9 test vectors of profile 0 (303 vp90-2-*, 8-bit 4:2:0) and profile 2 "
            "(2 vp92-2-*, 10- and 12-bit 4:2:0); profiles 1 and 3 (4:4:4, 4:2:2, 4:4:0) not listed"),
    "vvc": ("vvc_conformance_v1.json", "VTM 24.0 DecoderAppStatic (whole and per-picture MD5)",
            "226 of the 283 JVET H.266.1 first-edition streams: Main 10 and Main 10 Still Picture, 4:2:0 "
            "or 4:0:0 (9 8-bit, 217 10-bit). Left out: 4:2:2 / 4:4:4, multilayer (OLS, VPS, SPATSCAL, "
            "ILRPL), palette, ACT; HRD_B_2 (VTM fails it) and VPS_C_2 (no probe result). STILL_B_1 is "
            "scored against VTM's output, which differs from the MD5 shipped in its zip"),
}

#: The VVC gate against the zips' own MD5s (not VTM's).
VVC_MAIN8 = ("vvc-main8", "vvc_conformance_main8.json", "the MD5 shipped in each official zip",
             "the 9 8-bit Main 4:2:0 streams of H.266.1, checked against the suite's own MD5s")

#: Browser corpus file stem (web/pkg build) -> manifest; the relaxed-SIMD build is reported beside it.
WASM = {"h264": "h264_conformance.json", "hevc": "hevc_conformance_v1.json",
        "vvc": "vvc_conformance_v1.json", "av1": "av1_conformance_libaom.json",
        "vp9": "vp9_conformance_libvpx.json"}


def manifest_entries(name):
    data = json.loads((CI / name).read_text())
    return {e["name"]: e for e in (data["streams"] if isinstance(data, dict) else data)}


def native_suite(results_dir, stem, manifest, reference, denominator):
    results = json.loads((results_dir / f"{stem}.json").read_text())
    entries = manifest_entries(manifest)
    depth = Counter()
    depth_pass = Counter()
    for name, r in results.items():
        bits = max(entries[name].get("bit_depth", [8]))
        depth[bits] += 1
        depth_pass[bits] += r["status"] == "pass"
    failing = sorted(n for n, r in results.items() if r["status"] != "pass")
    row = {"passed": sum(r["status"] == "pass" for r in results.values()), "run": len(results),
           "manifest": len(entries), "by_bit_depth": {str(b): [depth_pass[b], depth[b]] for b in sorted(depth)},
           "reference": reference, "denominator": denominator,
           "failing": {n: results[n]["status"] + (f": {results[n]['error']}" if results[n].get("error") else "")
                       for n in failing}}
    if stem == "h264":
        required = {line.strip() for line in (CI / "h264_conformance_passing.txt").read_text().splitlines()
                    if line.strip() and not line.startswith("#")}
        row["required"] = len(required)
        row["required_failing"] = sorted(required & set(failing))
    return row


def argon_suite(path):
    """The AOM Argon streams run by tools/dev/av1_argon.py: per set, the
    streams whose output equals aomdec's (in the `*_error` sets of invalid
    streams, those our decoder ends without a panic, signal or timeout)."""
    results = json.loads(path.read_text())
    streams = results["streams"]
    passed = ("pass", "robust", "unframed")
    by_set = {}
    for name, r in streams.items():
        counts = by_set.setdefault(name.split("/")[0], [0, 0])
        counts[0] += r["status"] in passed
        counts[1] += 1
    sets = sorted(by_set)
    return {"passed": sum(r["status"] in passed for r in streams.values()), "run": len(streams),
            "by_set": by_set,
            "reference": "aomdec --rawvideo (libaom v3.15.3) on the same IVF rewrite of each stream, "
                         "against our decoder's --unbounded 1 --monochrome-chroma 0 output; the "
                         "*_error sets (invalid streams) check only that our decoder ends without a "
                         "panic, signal or timeout; the Argon reference MD5s (aomdec --rawvideo "
                         "--all-layers) are not compared",
            "denominator": "AOM Argon v2.1.1 (argon_coveragetool_av1_base_and_extended_profiles_v2.1.1"
                           f".zip): every stream of the sets {', '.join(sets)}; the large scale tile "
                           "sets (tile list OBUs, an exception) are not run",
            "failing": {n: r["status"] + (f": {r['error']}" if r.get("error") else "")
                        for n, r in sorted(streams.items()) if r["status"] not in passed}}


def wasm_suites(results_dir):
    out = {}
    for codec, manifest in WASM.items():
        builds = {}
        for build in ("pkg", "relaxed"):
            path = results_dir / "wasm" / f"{build}-{codec}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            builds[build] = {"passed": r["passed"], "compared": r["compared"],
                             "deeper_than_8_bits": [r["deep_passed"], r["deep_compared"]],
                             "required_failing": r["required_failing"],
                             "failing": {n: s["problem"] for n, s in r["streams"].items() if s["problem"]}}
        if builds:
            first = json.loads((results_dir / "wasm" / f"{next(iter(builds))}-{codec}.json").read_text())
            out[codec] = {"manifest": first["manifest_streams"],
                          "denominator": f"the native suite's 4:2:0 progressive streams of {manifest} "
                                         f"(left out: {len(first['skipped'])} 4:0:0 or interlaced)",
                          **builds}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--commit", required=True, help="the commit whose release build was run")
    parser.add_argument("--note", action="append", default=[],
                        help="a fact about the run the result files do not hold (repeatable)")
    args = parser.parse_args()
    summary = {"commit": args.commit, "date": args.results_dir.name.removeprefix("conformance-"),
               "notes": args.note,
               "native": {stem: native_suite(args.results_dir, stem, *spec)
                          for stem, spec in NATIVE.items() if (args.results_dir / f"{stem}.json").exists()}}
    stem, manifest, reference, denominator = VVC_MAIN8
    if (args.results_dir / f"{stem}.json").exists():
        results = json.loads((args.results_dir / f"{stem}.json").read_text())
        summary["native"][stem] = {
            "passed": sum(r["status"] == "pass" for r in results.values()), "run": len(results),
            "manifest": len(json.loads((CI / manifest).read_text())), "reference": reference,
            "denominator": denominator,
            "failing": {n: r["status"] for n, r in results.items() if r["status"] != "pass"}}
    if (args.results_dir / "argon.json").exists():
        summary["native"]["av1-argon"] = argon_suite(args.results_dir / "argon.json")
    summary["wasm"] = wasm_suites(args.results_dir)
    (args.results_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    for group in ("native", "wasm"):
        for name, row in summary[group].items():
            for build in ("pkg", "relaxed") if group == "wasm" else (None,):
                r = row.get(build) if build else row
                if r:
                    print(f"{group:6} {name:10} {build or '':8} {r['passed']}/{r.get('run', r.get('compared'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
