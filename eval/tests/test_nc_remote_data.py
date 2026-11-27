"""tools/dev/nc_remote.py: --data destinations and the warning for paths the sync leaves out."""
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import nc_remote  # noqa: E402


def worktree(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    subprocess.run(["git", "init", "-q", str(tree)], check=True)
    (tree / ".gitignore").write_text("out/\n")
    return tree


def test_data_lands_in_the_job_tree_or_under_the_box_home(tmp_path, monkeypatch):
    monkeypatch.setattr(nc_remote.Path, "home", lambda: tmp_path)
    tree = worktree(tmp_path)
    assert nc_remote.data_destination(tree / "eval/x.npz", tree, "job") \
        == "ncbox/jobs/job/src/eval/x.npz"
    assert nc_remote.data_destination(tmp_path / "nanocodec-data/work/x.npz", tree, "job") \
        == "nanocodec-data/work/x.npz"


def test_large_raw_and_ignored_paths_are_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(nc_remote, "SYNC_MAX_BYTES", 10)
    tree = worktree(tmp_path)
    (tree / "big.npz").write_bytes(b"x" * 11)
    (tree / "small.py").write_bytes(b"x")
    (tree / "clip.yuv").write_bytes(b"x")
    (tree / "out").mkdir()
    (tree / "out/table.npz").write_bytes(b"x")
    reasons = {name: nc_remote.sync_exclusion(tree / name, tree)
               for name in ("big.npz", "small.py", "clip.yuv", "out/table.npz")}
    assert reasons == {"big.npz": "over 0 MB", "small.py": None, "clip.yuv": "raw video",
                       "out/table.npz": "ignored by git"}
    assert nc_remote.sync_exclusion(tmp_path / "elsewhere", tree) is None


def test_the_warning_names_excluded_paths_not_uploaded(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(nc_remote, "SYNC_MAX_BYTES", 10)
    tree = worktree(tmp_path)
    (tree / "big.npz").write_bytes(b"x" * 11)
    (tree / "out").mkdir()
    (tree / "out/t.npz").write_bytes(b"x")
    nc_remote.warn_excluded("python3 fit.py big.npz --table=out/t.npz", tree, [])
    err = capsys.readouterr()
    text = err.out + err.err
    assert "big.npz is not synced" in text and "out/t.npz is not synced" in text
    nc_remote.warn_excluded("python3 fit.py big.npz", tree, [(tree / "big.npz").resolve()])
    err = capsys.readouterr()
    assert "not synced" not in err.out + err.err


def test_synced_tool_directories_go_up_as_real_files_and_keep_the_boxs_own_links(
        tmp_path, monkeypatch):
    """--sync copies a link's target, never the link: the PC's absolute links (vpxdec ->
    /home/mark/...) landed dangling on the build box (2026-11-23). A working link of the
    box's own (clang-format -> its venv) is kept; a dangling one is replaced."""
    pc, box_home = tmp_path / "pc", tmp_path / "box"
    tools, elsewhere = pc / "nanocodec-data/anchors/x86_64/bin", pc / "nanocodec-data/vp9-av2"
    tools.mkdir(parents=True)
    elsewhere.mkdir(parents=True)
    (elsewhere / "vpxdec").write_text("real vpxdec")
    (tools / "vpxdec").symlink_to(elsewhere / "vpxdec")  # absolute, outside the tree
    (tools / "clang-format").write_text("the PC's")
    on_box = box_home / "nanocodec-data/anchors/x86_64/bin"
    on_box.mkdir(parents=True)
    (box_home / "venv").mkdir()
    (box_home / "venv/clang-format").write_text("the box's")
    (on_box / "clang-format").symlink_to("../../../../venv/clang-format")
    (on_box / "vpxdec").symlink_to("/nonexistent/home/mark/vp9-av2/vpxdec")  # dangling
    monkeypatch.setattr(nc_remote.Path, "home", lambda: pc)

    def rsync(box, mode, *args):
        local = [a.split(":", 1)[1] if ":" in a and a.startswith(nc_remote.SSH_USER) else a
                 for a in args]
        local[-1] = str(box_home / local[-1])
        subprocess.run(["rsync", *mode, "--mkpath", *local], check=True)
    monkeypatch.setattr(nc_remote, "rsync", rsync)
    monkeypatch.setattr(nc_remote, "ssh", lambda box, command, **kw: subprocess.run(
        ["bash", "-c", command], cwd=box_home, **kw))
    box = type("Box", (), {"host": "box"})()
    nc_remote.sync(box, worktree(tmp_path), "job", [tools])
    assert not (on_box / "vpxdec").is_symlink()
    assert (on_box / "vpxdec").read_text() == "real vpxdec"
    assert (on_box / "clang-format").is_symlink()
    assert (on_box / "clang-format").read_text() == "the box's"
