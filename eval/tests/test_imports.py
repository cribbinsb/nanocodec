"""Every nceval module imports on its own, and their top-level imports form no cycle.

A cycle (scoreboard -> tune -> tune_realuse -> realuse -> scoreboard, 2026-11-09)
broke `python3 -m nceval ...` for everyone; shared names live in leaf modules
(nceval.stages, nceval.rows) instead.
"""

import ast
import subprocess
import sys
from pathlib import Path

import pytest

EVAL = Path(__file__).resolve().parents[1]
PACKAGE = EVAL / "nceval"


def module_name(path: Path) -> str:
    parts = path.with_suffix("").relative_to(EVAL).parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


MODULES = sorted(module_name(p) for p in PACKAGE.rglob("*.py"))


@pytest.mark.parametrize("module", MODULES)
def test_module_imports_in_a_fresh_interpreter(module):
    result = subprocess.run([sys.executable, "-c", f"import {module}"], cwd=EVAL,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-600:]


def top_level_imports(path: Path) -> set[str]:
    """nceval modules `path` imports at module level (relative imports)."""
    name = module_name(path)
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ImportFrom) and node.level:
            base = package.rsplit(".", node.level - 1)[0] if node.level > 1 else package
            target = f"{base}.{node.module}" if node.module else base
            found |= {target} | {f"{target}.{alias.name}" for alias in node.names}
    return found & set(MODULES)


def test_no_import_cycles():
    graph = {module_name(p): top_level_imports(p) for p in PACKAGE.rglob("*.py")}
    state: dict[str, int] = {}

    def visit(module: str, path: list[str]) -> None:
        state[module] = 1
        for dependency in sorted(graph[module]):
            if state.get(dependency) == 1:
                pytest.fail("import cycle: " + " -> ".join(path[path.index(dependency):]
                                                          + [dependency]))
            if not state.get(dependency):
                visit(dependency, path + [dependency])
        state[module] = 2
    for module in sorted(graph):
        if not state.get(module):
            visit(module, [module])
