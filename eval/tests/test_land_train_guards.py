"""land_train_guards refuses a commit that reports tripped nceval.train guards without a
justification (tools/dev/land_train_guards.py, nceval.train_guards)."""
import importlib.util
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "land_train_guards", ROOT / "tools" / "dev" / "land_train_guards.py")
land_train_guards = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(land_train_guards)

TRIPPED = "h264 fast: a.x\n\n- nceval.train guards tripped: final-check: bd_worst_psnr_y +3.19 > +2.00\n"


def commit(repo: Path, message: str) -> None:
    (repo / "f").write_text(message)
    subprocess.run(["git", "add", "f"], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "-m", message], cwd=repo, check=True)


def test_refuses_tripped_guards_until_justified(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    commit(tmp_path, "base")
    base = subprocess.run(["git", "rev-parse", "HEAD"], cwd=tmp_path, check=True,
                          capture_output=True, text=True).stdout.strip()
    monkeypatch.chdir(tmp_path)
    commit(tmp_path, "unrelated change")
    assert land_train_guards.main([base]) == 0
    commit(tmp_path, TRIPPED + "Guard-justification: <required: what the guards flag>\n")
    assert land_train_guards.main([base]) == 1
    subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
                    "--amend", "-m", TRIPPED + "Guard-justification: one clip, a better key\n"],
                   cwd=tmp_path, check=True)
    assert land_train_guards.main([base]) == 0
