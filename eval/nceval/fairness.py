"""The benchmark fairness record: every encoder's exact command, built from one file.

docs/benchmark-fairness.toml holds, per encoder, the CBR and Fixed QP command templates
and a reason for every flag group (what it controls, why the value, why it is fair,
what asymmetry remains). The benchmark scripts take their command lines from
`command` only, and `render` writes the procedure's fairness tables, so the record, the
document and the runs cannot drift (eval/tests/test_fairness_record.py).

    families()                         -> the charted families (name -> template values)
    command(family, mode, clip values) -> argv
    render()                           -> the markdown tables of docs/benchmark-procedure.md section 17
"""

from __future__ import annotations

import argparse
import os
import re
import tomllib
from pathlib import Path

RECORD = Path(__file__).resolve().parents[2] / "docs" / "benchmark-fairness.toml"
#: The charted families and their template values (BP-2 section 2).
OURS_RUNGS = {"h264": ("veryfast", "faster", "fast", "medium", "slow", "slower"),
              "hevc": ("faster", "fast", "medium", "slow", "slower", "veryslow"),
              "vvc": ("faster", "fast", "medium", "slow", "slower", "veryslow"),
              "av1": ("veryfast", "faster", "fast", "medium", "slow", "slower"),
              "vp9": ("veryfast", "faster", "fast", "medium", "slow", "slower"),
              "av2": ("faster", "fast", "medium", "slow", "slower")}
#: Every real-time preset from each reference's fastest to medium (review 2: no gaps, so a reference
#: front is not interpolation).
X26X_PRESETS = {"x264": ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium"),
                "x265": ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium")}
METRIC_TUNES = ("psnr", "ssim")
#: The zero line (section 9): the only family charted with metric twins (D20).
ZERO_LINE = ("x264", "medium")
SVT_PRESETS = (8, 9, 10, 11, 12, 13)
KVAZAAR_PRESETS = ("ultrafast", "superfast", "veryfast")
VVENC_PRESETS = ("faster", "fast")
VPX_SPEEDS, AOM_SPEEDS = (7, 9), (8, 10)


def load(path: Path = RECORD) -> dict:
    return tomllib.loads(path.read_text())


def _encoders(record: dict) -> dict[str, dict]:
    return {e["tool"].split()[0]: e for e in record["encoder"]}


def families() -> dict[str, tuple[str, dict]]:
    """{family: (tool, template values)} for every charted family."""
    out: dict[str, tuple[str, dict]] = {}
    for codec, rungs in OURS_RUNGS.items():
        for rung in rungs:
            out[f"nanocodec-{codec}@{rung}"] = ("nanocodec", {
                "CODEC": codec, "PRESET_ARGS": [] if rung == "medium" else ["--preset", rung]})
    for tool, presets in X26X_PRESETS.items():
        for preset in presets:
            # metric twins only for the zero line, x264 medium (owner, 2026-11-25, D20)
            tunes = (None, *METRIC_TUNES) if (tool, preset) == ZERO_LINE else (None,)
            for tune in tunes:
                name = f"{tool}-{preset}" + (f"~{tune}" if tune else "")
                if tool == "x264":
                    values = {"TUNE": "zerolatency" + (f",{tune}" if tune else ""), "TUNE_EXTRA": []}
                else:
                    values = {"TUNE": tune or "zerolatency", "TUNE_EXTRA": ["--no-cutree"] if tune else []}
                out[name] = (tool, {"PRESET": preset, **values})
    for name, complexity, cabac, aq in (("openh264-baseline", "medium", "0", "1"),
                                        ("openh264-baseline~cabac", "high", "1", "1")):
        values = {"COMPLEXITY": complexity, "CABAC": cabac, "AQ": aq}
        out[name] = ("openh264-harness", {**values, "SKIP": "0", "SKIP_ARGS": []})
        if "~" not in name:  # as libvpx and libaom: a '+drop' row for the main family only
            out[name + DROP_SUFFIX] = ("openh264-harness",
                                       {**values, "SKIP": "1", "SKIP_ARGS": OPENH264_DROP_ARGS})
    for preset in SVT_PRESETS:
        out[f"svtav1-m{preset}"] = ("SvtAv1EncApp", {"PRESET": str(preset)})
    # (main-row tool, '+drop' row tool): libaom's main rows run the example's settings through
    # aomenc with the shared bucket, its '+drop' rows the example as shipped (procedure section 3)
    for (tool, drop_tool), kind, speeds in ((("vpxenc", "vpxenc"), "libvpx-vp9-cpu", VPX_SPEEDS),
                                            (("aomenc", "svc_encoder_rtc"), "libaom-av1-cpu", AOM_SPEEDS)):
        for speed in speeds:
            for tune in (None,):  # no metric twins off the zero line (D20)
                name = f"{kind}{speed}" + (f"~{tune}" if tune else "")
                values = {"PRESET": str(speed), "TUNE_EXTRA": ["--tune=ssim"] if tune else []}
                out[name] = (tool, {**values, "DROP": "0", "MAX_Q": MAIN_ROW_MAX_Q})
                if not tune:
                    out[name + DROP_SUFFIX] = (drop_tool, {**values, "DROP": VPX_AOM_DROP_THRESHOLD,
                                                           "MAX_Q": SHIPPED_DROP_MAX_Q})
    for preset in KVAZAAR_PRESETS:
        out[f"kvazaar-{preset}"] = ("kvazaar", {"PRESET": preset})
    for preset in VVENC_PRESETS:
        out[f"vvenc-{preset}"] = ("vvencFFapp", {"PRESET": preset})
    return out


def phase_values(record: dict) -> dict[str, str]:
    """The current phase's shared values (initial fullness): {INIT_MS}, {VBV_INIT}, {KEY_CAP}."""
    return dict(record["phase"][record["phase"]["current"]])


#: Our encoder's conditions in the dev screens (owner, 2026-11-24: "if the lookahead is off
#: in the benchmarks it should be off in dev too"): BENCHMARK, the record's own flags
#: (`own_conditions`), unless NC_CONDITIONS=default asks for the encoder's shipped defaults
#: (studying them; said loudly by every run that uses it).
CONDITIONS_ENV = "NC_CONDITIONS"
BENCHMARK, SHIPPED_DEFAULTS = "benchmark", "default"
OURS_TOOL = "nanocodec"
#: Where our record command's condition flags sit: after the rate (or quantizer) value, up
#: to the rung's own arguments.
RATE_VALUE = {"cbr": "{KBPS}", "qp": "{QP}"}
PRESET_ARGS = "{PRESET_ARGS}"


def conditions() -> str:
    """BENCHMARK or SHIPPED_DEFAULTS (NC_CONDITIONS); anything else is refused."""
    value = os.environ.get(CONDITIONS_ENV, BENCHMARK)
    if value not in (BENCHMARK, SHIPPED_DEFAULTS):
        raise SystemExit(f"{CONDITIONS_ENV}={value}: use {BENCHMARK} or {SHIPPED_DEFAULTS}")
    return value


def own_conditions(mode: str, record: dict | None = None) -> list[str]:
    """Our encoder's condition flags in `mode` ("cbr" or "qp") as the benchmark runs it: the
    record's command after the rate or quantizer value, up to the rung's arguments, with the
    phase's values filled ({KBPS} stays for the caller). The one definition: the dev screens
    code our encoder with these, so dev and benchmark conditions cannot differ."""
    record = record or load()
    template = _encoders(record)[OURS_TOOL][mode]
    start = template.index(RATE_VALUE[mode]) + 1
    flags = template[start:template.index(PRESET_ARGS)]
    phase = phase_values(record)
    out = []
    for arg in flags:
        for name, value in phase.items():
            arg = arg.replace("{" + name + "}", value)
        out.append(arg)
    return out


#: The labelled frame-dropping rows (review 3 C1): CBR only, for the encoders whose authors
#: ship them with dropping on.
DROP_SUFFIX = "+drop"
#: openh264's '+drop' row: Annex B carries no timestamps, so the harness lists the input frame
#: of each coded picture beside the stream; the box fills {OUTPUT} and keeps the list in the
#: encode record (box.py `take_coded_frames`, task T16).
OPENH264_DROP_ARGS = ["--coded-frames", "{OUTPUT}.frames"]
VPX_AOM_DROP_THRESHOLD = "30"
#: libvpx's largest quantizer (owner, 2026-11-25): its full range on the main rows, which code
#: every frame; the real-time example's 52 (examples/vpx_temporal_svc_encoder.c:814), which
#: presupposes frame dropping, on the '+drop' rows only.
MAIN_ROW_MAX_Q = "63"
SHIPPED_DROP_MAX_Q = "52"


class NotCharted(ValueError):
    """The family has no command in this mode (a +drop row in Fixed QP; a tune twin of an
    encoder whose CBR program has no --tune)."""


#: VVenC's clock must be a multiple of the frame rate; 27 MHz is its own advice for NTSC.
VVENC_NTSC_TICKS = 27_000_000

#: Placeholders a plan may leave for the box to fill (paths that exist only there).
BOX_PLACEHOLDERS = ("BIN", "CFG", "INPUT", "OUTPUT")


def command(family: str, mode: str, clip: dict, record: dict | None = None) -> list[str]:
    """The family's argv in `mode` ("cbr" or "qp"); `clip` gives BIN INPUT OUTPUT W H
    FPS_NUM FPS_DEN N KEYINT CFG and KBPS or QP; the record's current phase gives the initial
    fullness. A value may be its own placeholder for BOX_PLACEHOLDERS (a plan leaves the box's
    paths); any other unfilled placeholder is refused."""
    record = record or load()
    tool, values = families()[family]
    entry = _encoders(record)[tool]
    if mode not in entry.get("modes", ("cbr", "qp")):
        raise NotCharted(f"{family}: charted in {', '.join(entry['modes'])} only (the record)")
    if family.endswith(DROP_SUFFIX) and mode != "cbr":
        raise NotCharted(f"{family}: frame dropping is a CBR row only")
    if "~" in family and mode in entry.get("untuned_modes", ()):
        raise NotCharted(f"{family}: {tool}'s {mode} program has no tune twin")
    template = entry[mode]
    fps_float = int(clip["FPS_NUM"]) / int(clip["FPS_DEN"])
    filled = {**phase_values(record), **{k: str(v) for k, v in clip.items()},
              "FPS_FLOAT": f"{fps_float:g}"}
    num, den = int(clip["FPS_NUM"]), int(clip["FPS_DEN"])
    # a multiple of FPS_NUM is a multiple of the rate num/den: 27000000 itself for NTSC
    filled["VVENC_TICKS"] = str(VVENC_NTSC_TICKS // num * num)
    if "KBPS" in clip and str(clip["KBPS"]).isdigit():  # VVenC's lowest accepted peak
        filled["KBPS_X1500"] = str(int(clip["KBPS"]) * 1500)
    strings = {**filled, **{k: v for k, v in values.items() if isinstance(v, str)}}

    def fill(arg: str) -> str:
        for name, value in strings.items():
            arg = arg.replace("{" + name + "}", value)
        if set(re.findall(r"{(\w+)}", arg)) - set(BOX_PLACEHOLDERS):
            raise ValueError(f"{family} {mode}: unfilled {arg}")
        return arg

    argv: list[str] = []
    for arg in template:
        key = arg[1:-1] if arg.startswith("{") and arg.endswith("}") else None
        # a list placeholder expands to its arguments, each filled like any other
        argv += [fill(a) for a in values[key]] if key and isinstance(values.get(key), list) \
            else [fill(arg)]
    return argv


def render(record: dict | None = None) -> str:
    """The fairness tables: shared constraints, then one table per encoder."""
    record = record or load()
    lines = ["### Shared constraints", "", "| Constraint | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in record["shared"].items()]
    lines += ["", f"Phase now: **{record['phase']['current']}**; values per phase:", "",
              "| Phase | " + " | ".join(phase_values(record)) + " |",
              "|---|" + "---|" * len(phase_values(record))]
    lines += ["| " + name + " | " + " | ".join(str(v) for v in values.values()) + " |"
              for name, values in record["phase"].items() if name != "current"]
    for e in record["encoder"]:
        lines += ["", f"### {e['family']} ({e['tool']})", "",
                  f"- Version: {e['version']}", f"- Build: {e['build']}", "",
                  "| Flags | Controls | Value | Why fair | Remaining asymmetry |", "|---|---|---|---|---|"]
        lines += [f"| `{w['flags']}` | {w['controls']} | {w['value']} | {w['fair']} | {w['remaining']} |"
                  for w in e["why"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("action", choices=("render",))
    ap.parse_args()
    print(render(), end="")


if __name__ == "__main__":
    main()
