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
    ("crates/nc-encoder/presets/candidates/av1-x.toml", "presets"),
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
    dirs, dependents, embedded = graph
    kind, crates = land_scope.classify("crates/nc-encoder/presets/hevc/fast.toml", ROOT, dirs, embedded)
    assert kind == "included"
    # nc-encoder's preset table embeds it (presets.rs); its front ends follow,
    # and the CLI's tests pin the shipped values.
    assert crates == {"nanocodec-encoder", "nanocodec-cli"}
    assert {"nanocodec-hevc", "nanocodec-cli", "nanocodec-web"} <= land_scope.closure(crates, dependents)


def test_a_codec_change_reaches_its_front_ends_only(graph):
    _, dependents, _ = graph
    assert land_scope.closure({"nanocodec-vvc"}, dependents) == {
        "nanocodec-vvc", "nanocodec-cli", "nanocodec-web", "nanocodec"}


@pytest.mark.parametrize("crate", ["nanocodec-dsp", "nanocodec-encoder", "nanocodec-hevc", "nanocodec-cli"])
def test_a_change_that_can_alter_a_stream_runs_the_conformance_check(graph, crate):
    _, dependents, _ = graph
    affected = land_scope.closure({crate}, dependents)
    assert land_scope.needs_conformance([f"crates/{crate}/src/lib.rs"], affected, False)


def test_eval_web_and_docs_changes_skip_the_conformance_check(graph):
    _, dependents, _ = graph
    assert not land_scope.needs_conformance(["eval/nceval/tune.py", "docs/STATUS.md"],
                                            set(), False)
    assert not land_scope.needs_conformance(["crates/nc-web/src/lib.rs"],
                                            land_scope.closure({"nanocodec-web"}, dependents), False)
    assert land_scope.needs_conformance(["eval/nceval/conformance.py"], set(), False)
    assert land_scope.needs_conformance(["Cargo.lock"], set(), True)


@pytest.mark.parametrize("paths, codecs", [
    (["crates/nc-vp9/src/lib.rs"], ["vp9"]),
    (["crates/nc-vp9/src/lib.rs", "crates/nc-encoder/presets/av1/fast.toml"], ["av1", "vp9"]),
    (["crates/nc-encoder/src/lib.rs"], ["all"]),
    (["crates/nc-cli/src/main.rs"], ["all"]),
    (["crates/nc-encoder/presets/hevc/fast.toml"], ["hevc"]),
    (["eval/nceval/conformance.py"], ["all"]),
    (["crates/nc-web/src/lib.rs"], ["0"]),
])
def test_the_conformance_check_codes_only_the_changed_codecs(graph, paths, codecs):
    dirs, dependents, embedded = graph
    crates = set()
    for path in paths:
        if not path.startswith("crates/nc-encoder/presets/"):  # presets name their codec (plan_for_paths)
            crates |= land_scope.classify(path, ROOT, dirs, embedded)[1]
    affected = land_scope.closure(crates, dependents)
    assert land_scope.conformance_codecs(paths, affected, False) == codecs


def test_browser_module_and_checkasm_run_only_for_their_own_crates():
    def plan_of(path):
        out = land_scope.plan_for_paths(ROOT, [path])
        return dict(line.split(" ", 1) for line in out.splitlines() if " " in line)
    codec = plan_of("crates/nc-vp9/src/lib.rs")
    assert codec["web"] == "0" and codec["checkasm"] == "0"
    kernels = plan_of("crates/nc-dsp/src/lib.rs")
    assert kernels["web"] == "1" and kernels["checkasm"] == "1"
    assert plan_of("crates/nc-web/src/lib.rs")["web"] == "1"


def test_only_a_change_to_the_page_runs_the_chromium_page_tests():
    assert land_scope.changes_page("web", "web/app.js")
    assert land_scope.changes_page("crate", "crates/nc-web/src/lib.rs")
    assert not land_scope.changes_page("crate", "crates/nc-vvc/src/lib.rs")
    assert not land_scope.changes_page("crate", "crates/nc-dsp/src/x86/cdef.rs")
    assert not land_scope.changes_page("docs", "docs/PROCEDURES.md")


def test_the_plan_names_the_directly_touched_crates_for_their_unit_tests():
    def touched(path):
        out = land_scope.plan_for_paths(ROOT, [path])
        return next(line.split()[1:] for line in out.splitlines() if line.split()[0] == "touched")
    assert touched("crates/nc-vvc/src/lib.rs") == ["nanocodec-vvc"]
    assert touched("docs/STATUS.md") == []


def test_a_kernel_change_reaches_every_crate_but_nc_common(graph):
    dirs, dependents, _ = graph
    assert land_scope.closure({"nanocodec-dsp"}, dependents) == set(dirs) - {"nanocodec-common"}


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
