"""land_scope classifies changed paths into the landing checks they need (land.sh)."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("land_scope", ROOT / "tools" / "dev" / "land_scope.py")
land_scope = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(land_scope)
DOCS_SPEC = importlib.util.spec_from_file_location("check_docs", ROOT / "tools" / "dev" / "check_docs.py")
check_docs = importlib.util.module_from_spec(DOCS_SPEC)
DOCS_SPEC.loader.exec_module(check_docs)


@pytest.fixture(scope="module")
def graph():
    dirs, dependents = land_scope.workspace_graph(ROOT)
    return dirs, dependents, land_scope.embedded_files(ROOT, dirs)


@pytest.mark.parametrize("path, kind", [
    ("docs/STATUS.md", "docs"),
    ("README.md", "docs"),
    ("docs/results/chart.svg", "docs"),
    ("eval/nceval/tune.py", "eval"),
    ("tools/dev/fastbd.py", "python"),
    ("tools/ci/vp9_conformance.sh", "shell"),
    ("web/app.js", "web"),
    ("fuzz/fuzz_targets/h264.rs", "fuzz"),
    ("presets/candidates/av1-x.toml", "presets"),
    ("Cargo.lock", "gate"),
    ("tools/ci/check.sh", "gate"),
    ("tools/web/build.sh", "gate"),
    ("tools/dev/land.sh", "gate"),
    ("crates/nc-vvc/README.md", "crate"),
    ("Makefile", "unknown"),
])
def test_each_path_class(graph, path, kind):
    dirs, _, embedded = graph
    assert land_scope.classify(path, ROOT, dirs, embedded)[0] == kind, path


def test_embedded_presets_select_the_crates_that_include_them(graph):
    dirs, _, embedded = graph
    kind, crates = land_scope.classify("presets/hevc-fast.toml", ROOT, dirs, embedded)
    assert kind == "included"
    assert {"nc-cli", "nc-web", "nc-hevc"} <= crates


def test_a_codec_change_reaches_its_front_ends_only(graph):
    _, dependents, _ = graph
    assert land_scope.closure({"nc-vvc"}, dependents) == {"nc-vvc", "nc-cli", "nc-web"}


@pytest.mark.parametrize("crate", ["nc-dsp", "nc-encoder", "nc-hevc", "nc-cli"])
def test_a_change_that_can_alter_a_stream_runs_the_conformance_check(graph, crate):
    _, dependents, _ = graph
    affected = land_scope.closure({crate}, dependents)
    assert land_scope.needs_conformance([f"crates/{crate}/src/lib.rs"], affected, False)


def test_eval_web_and_docs_changes_skip_the_conformance_check(graph):
    _, dependents, _ = graph
    assert not land_scope.needs_conformance(["eval/nceval/tune.py", "docs/STATUS.md"],
                                            set(), False)
    assert not land_scope.needs_conformance(["crates/nc-web/src/lib.rs"],
                                            land_scope.closure({"nc-web"}, dependents), False)
    assert land_scope.needs_conformance(["eval/nceval/conformance.py"], set(), False)
    assert land_scope.needs_conformance(["Cargo.lock"], set(), True)


def test_only_a_change_to_the_page_runs_the_chromium_page_tests():
    assert land_scope.changes_page("web", "web/app.js")
    assert land_scope.changes_page("crate", "crates/nc-web/src/lib.rs")
    assert not land_scope.changes_page("crate", "crates/nc-vvc/src/lib.rs")
    assert not land_scope.changes_page("crate", "crates/nc-dsp/src/x86/cdef.rs")
    assert not land_scope.changes_page("docs", "docs/PROCEDURES.md")


def test_train_rank_puts_docs_first_and_page_changes_last():
    assert land_scope.rank(ROOT, ["docs/STATUS.md", "README.md"]) == 0
    assert land_scope.rank(ROOT, ["docs/STATUS.md", "crates/nc-vvc/src/lib.rs"]) == 1
    assert land_scope.rank(ROOT, ["presets/av1-fast.toml"]) == 1
    assert land_scope.rank(ROOT, ["web/app.js", "docs/handoff/browser-demo.md"]) == 2
    assert land_scope.rank(ROOT, ["crates/nc-web/src/lib.rs"]) == 2
    assert land_scope.rank(ROOT, []) == 1


def test_a_kernel_change_reaches_every_crate_but_nc_common(graph):
    dirs, dependents, _ = graph
    assert land_scope.closure({"nc-dsp"}, dependents) == set(dirs) - {"nc-common"}


def test_docs_check_finds_broken_links_and_bad_svg(tmp_path):
    (tmp_path / "img.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'/>")
    (tmp_path / "bad.svg").write_text("<svg>")
    (tmp_path / "doc.md").write_text(
        "[ok](img.svg) ![ok](img.svg#x) [web](https://example.org) [here](#top)\n"
        "`[code](missing.md)`\n```\n[fenced](missing.md)\n```\n"
        "[gone](missing.md) <img src=\"nope.png\">\n")
    problems = (check_docs.check_markdown(tmp_path / "doc.md", tmp_path)
                + check_docs.check_svg(tmp_path / "img.svg", tmp_path)
                + check_docs.check_svg(tmp_path / "bad.svg", tmp_path))
    assert [p.split(": ", 1)[1].split(" (")[0] for p in problems] == [
        "broken link missing.md", "broken link nope.png", "not well-formed SVG"]
