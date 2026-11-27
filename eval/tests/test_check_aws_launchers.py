"""tools/ci/check_aws_launchers.py: launches go through aws_session, on the guide's types and
regions, on demand only with a reason, and every launcher has a reclaim test."""
import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools/ci/check_aws_launchers.py"
spec = importlib.util.spec_from_file_location("check_aws_launchers", SCRIPT)
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)

TYPES, REGIONS = {"c8g.8xlarge"}, {"eu-north-1"}


def problems(path, source):
    return check.problems(path, source, TYPES, REGIONS)


def test_the_repository_passes():
    assert check.main() == 0


def test_a_type_or_region_outside_the_guide_fails_but_a_docstring_may_name_one():
    found = problems("tools/dev/x.py", '"""Uses c7a.24xlarge."""\nT = "c7a.24xlarge"\n'
                                       'R = "ap-south-1"\nOK = ("c8g.8xlarge", "eu-north-1")\n')
    assert len(found) == 2 and "c7a.24xlarge" in found[0] and "ap-south-1" in found[1]


def test_a_direct_launch_and_an_unexplained_on_demand_fail():
    found = problems("tools/dev/x.py", "ec2.run_instances(MinCount=1)\nS(on_demand=True)\n")
    assert any("run_instances" in f for f in found) and any("on_demand" in f for f in found)
    assert not problems("tools/dev/aws_session.py", "ec2.run_instances(MinCount=1)\n")


def test_a_launcher_without_a_reclaim_test_fails():
    found = problems("tools/dev/new_tool.py", "spec = SessionSpec(region='eu-north-1')\n")
    assert found == ["tools/dev/new_tool.py: launches instances (SessionSpec) but has no "
                     "entry in RECLAIM_TESTS"]
    assert not problems("tools/dev/aws_measure.py", "spec = SessionSpec()\n")


def test_every_registered_reclaim_test_exists():
    assert check.missing_tests() == []


def test_a_tree_without_git_is_walked(tmp_path):
    """The build box's synced copy has no .git (2026-11-24: `git ls-files` exited 128 there
    and failed the verifier's checks)."""
    (tmp_path / "tools/dev").mkdir(parents=True)
    (tmp_path / "tools/dev/x.py").write_text("x = 1\n")
    (tmp_path / "eval/tests").mkdir(parents=True)
    (tmp_path / "eval/tests/test_x.py").write_text("x = 1\n")
    (tmp_path / "crates").mkdir()
    (tmp_path / "crates/y.py").write_text("y = 1\n")
    assert check.tracked(tmp_path) == ["tools/dev/x.py"]
