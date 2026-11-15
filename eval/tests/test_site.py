"""The results site's data and the README's headline facts are generated (by land.sh) and figure-checked.

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


def test_the_generated_files_are_regenerated_only_when_an_input_changed():
    assert site.inputs_changed("HEAD") is False
    assert site.INPUTS and all(not p.startswith("/") for p in site.INPUTS)


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
    ours = chart["points"]["nanocodec-h264@fast"]["cbr_command"]
    assert ours.startswith("nanocodec encode --codec h264") and ours.endswith("--preset fast")


def test_fixed_qp_speeds_are_the_chart_timing():
    """Every plotted fixed-QP point sits at the chart's own single-thread timing of its
    family (the records the CBR chart uses), never at the lens run's fps."""
    timing = site.timing_fps()
    chart = json.loads((site.SITE_DATA / "chart.json").read_text())
    view = next((v for v in chart["views"] if v["id"] == "qp"), None)
    if view is None:
        return
    cbr = next(v for v in chart["views"] if v["id"] == "dev-psnr")["values"]
    for family, value in view["values"].items():
        for platform, fps in value["fps"].items():
            if fps is None:
                continue
            assert fps == timing[family][platform], (family, platform, fps)
            if family in cbr and cbr[family]["fps"][platform]:  # the CBR chart's own point
                assert abs(fps - cbr[family]["fps"][platform]) <= 0.1, (family, platform, fps)


def test_no_local_path_is_published():
    """The site's data and the demo's stream manifest name programs and files, never a
    directory of the machine they were made on."""
    for path in sorted(site.SITE_DATA.glob("*.json")) + [site.REPO / "web/streams-ref/manifest.json"]:
        assert not re.search(r"/mldata|/home/|~/|anaconda", path.read_text()), path.name
