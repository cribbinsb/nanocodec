"""The results site's data and the README's headline facts are generated, current and figure-checked.

The browser side (every figure a rendered page shows is in that page's data) is
web/tests/results-pages.py; these are the checks that need no browser.
"""

import json
import re

from nceval import site

FIGURE = re.compile(r"\d[\d,]*(?:\.\d+)?\s*(?:%|fps\b|kbps\b|×)")
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
# Method constants the pages' prose may name: the ladder's reference frame rate
# and the confidence level of the bootstrap intervals.
METHOD_CONSTANTS = {"30 fps", "95%"}
# Names in the README block whose digits are not figures.
NAMES = re.compile(r"Ryzen 9 5950X|x264|H\.264|AV1|VP9|AV2|720p")


def test_generated_files_are_current():
    stale = site.stale(site.outputs())
    assert not stale, f"run python3 -m nceval.site: {[str(p) for p in stale]}"


def test_readme_figures_are_only_in_the_generated_block():
    text = site.README.read_text()
    begin, end = text.index(site.README_BEGIN), text.index(site.README_END)
    outside = text[:begin] + text[end:]
    assert not FIGURE.findall(outside)


def test_readme_block_figures_are_in_the_data():
    text = site.README.read_text()
    block = text[text.index(site.README_BEGIN) + len(site.README_BEGIN):text.index(site.README_END)]
    chart = json.loads((site.SITE_DATA / "chart.json").read_text())
    allowed = set()
    for view in chart["views"]:
        if view.get("clips"):
            allowed.add(str(view["clips"]))
        allowed.update(str(v) for v in view.get("tools", {}).values())
        for values in view["values"].values():
            for x in [values["j"], *values["fps"].values()]:
                if x is not None:
                    allowed.update({f"{abs(x):.0f}", f"{abs(x):.1f}"})
    allowed.add(str(sum(1 for p in chart["points"].values()
                        if p["ours"] and p["preset"] == site.DEFAULT_PRESET)))
    figures = NUMBER.findall(NAMES.sub(" ", block))
    assert figures and all(f in allowed for f in figures), \
        [f for f in figures if f not in allowed]


def test_pages_type_no_figures():
    pages = sorted((site.REPO / "web" / "results").glob("*.html"))
    assert pages
    for page in pages:
        text = re.sub(r"<[^>]+>", " ", page.read_text())
        typed = [f for f in FIGURE.findall(text) if f not in METHOD_CONSTANTS]
        assert not typed, (page.name, typed)


def test_command_lines_are_the_harness_settings():
    chart = json.loads((site.SITE_DATA / "chart.json").read_text())
    x264 = chart["points"][site.TUNED_ZERO]["cbr_command"]
    assert "--tune zerolatency,psnr" in x264 and "--bframes 0" in x264 and "--threads 1" in x264
    svt = chart["points"]["svtav1-m12"]["cbr_command"]
    assert svt.startswith("SvtAv1EncApp --rtc 1 --preset 12") and "--lp 1" in svt
    ours = chart["points"]["nanocodec-h264@h264-fast"]["cbr_command"]
    assert ours.startswith("nanocodec encode --codec h264") and ours.endswith("--preset h264-fast")
