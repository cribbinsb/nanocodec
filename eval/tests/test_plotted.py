"""Every shipped preset is plotted: nceval.plotted is the one family list, and every
view and tool that plots families takes it from there (a hand-kept copy lost AV1
Veryfast and AV2 Faster from some views)."""

import json
import sys
from pathlib import Path

from nceval import chart, plotted, rows, site
from nceval.codecs import Codec
from nceval.ctc import conditions
from nceval.encoders.nanocodec import PRESETS_DIR, shipped_presets

REPO = Path(__file__).resolve().parents[2]


def shipped_single_layer() -> set[str]:
    """Straight from the registry, independently of nceval.plotted."""
    names = set()
    for codec in Codec:
        for stem in shipped_presets(codec):
            if stem.startswith(("l2-", "l3-")):
                continue
            names.add(f"nanocodec-{codec.value}" + ("" if stem == "medium" else f"@{stem}"))
    return names


def test_every_shipped_preset_is_plotted():
    shipped = shipped_single_layer()
    assert set(plotted.own_names()) == shipped
    assert {name for name, _ in chart.OURS} == shipped
    assert {e.name for e in conditions.OURS} == shipped  # the fixed-QP lens
    assert {r.encoder for r in rows.ROWS} == {n for n in shipped
                                                  if not n.endswith(("@slow", "@slower", "@veryslow"))}


def test_every_codec_has_an_order_a_label_and_a_bar():
    assert set(plotted.CODEC_ORDER) == set(Codec)
    assert set(plotted.CODEC_LABELS) == set(Codec) and set(plotted.BARS) == set(Codec)


def test_every_plotted_family_has_a_label_and_a_head_to_head_pair():
    labels = plotted.labels()
    pairs = {ours for ours, _ in chart.SELECTED_PAIRS}
    known = set(plotted.plotted_names())
    for name in plotted.own_names():
        assert name in labels and name in pairs
    assert all(other in known for _, other in chart.SELECTED_PAIRS)


def test_a_new_preset_file_is_plotted_without_another_edit(tmp_path):
    for codec in Codec:
        (tmp_path / codec.value).mkdir()
        for stem in shipped_presets(codec, PRESETS_DIR)[1:]:
            (tmp_path / codec.value / f"{stem}.toml").write_text("")
    (tmp_path / "hevc" / "veryfast.toml").write_text("")
    names = plotted.own_names(tmp_path)
    assert "nanocodec-hevc@veryfast" in names
    assert plotted.labels(tmp_path)["nanocodec-hevc@veryfast"] == "HEVC Veryfast"
    pairs = plotted.pairs(tmp_path)
    assert ("nanocodec-hevc@veryfast", "nanocodec-hevc@faster") in pairs
    assert ("nanocodec-hevc@veryfast", "svtav1-m12") in pairs  # HEVC's bar
    # VP9 Veryfast's bar is now HEVC's own Veryfast, not its nearest slower rung
    assert ("nanocodec-vp9@veryfast", "nanocodec-hevc@veryfast") in pairs


def test_layered_presets_are_not_chart_points():
    assert not [n for n in plotted.own_names() if "@l2-" in n or "@l3-" in n]


def test_site_check_wants_every_shipped_preset_in_every_view():
    chart_data = json.loads((site.SITE_DATA / "chart.json").read_text())
    assert site.coverage_problems(chart_data) == []
    view = next(v for v in chart_data["views"] if v["id"] == "dev-psnr")
    marked = next(iter(view["not_measured"]), None)
    view["not_measured"] = {}
    if marked:  # an unmarked missing preset is a problem
        assert any(marked in p for p in site.coverage_problems(chart_data))
    view["not_measured"] = {"nanocodec-h264": "stale"}  # a mark on a measured one too
    assert any("nanocodec-h264 is marked" in p for p in site.coverage_problems(chart_data))


def test_release_and_chart_refresh_take_every_shipped_preset():
    sys.path.insert(0, str(REPO / "tools" / "dev"))
    import chart_refresh
    assert chart_refresh.family_list(chart_refresh.ALL_FAMILIES) == plotted.own_names()
    assert chart_refresh.family_list("nanocodec-av1,") == ["nanocodec-av1"]
