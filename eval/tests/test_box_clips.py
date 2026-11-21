"""Raw-video clips on rented machines: the frames a box takes of a clip are the bytes a
local measure reads (nceval.clipframes.frames_command, aws_heldout_chart.decode_line), and
the main chart's clip-set file gives its clips (realuse.write_natural_val_clipset)."""
import hashlib
import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from nceval.clipframes import frames_command
from nceval.clips import Clip
from nceval.config import Paths
from nceval.realuse import clipset_clips, natural_val, write_natural_val_clipset

TOOL = Path(__file__).resolve().parents[2] / "tools" / "dev" / "aws_heldout_chart.py"


def heldout_tool():
    spec = importlib.util.spec_from_file_location("aws_heldout_chart", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # its dataclasses look their module up
    spec.loader.exec_module(module)
    return module


def raw_clip(tmp_path: Path, frames: int = 4) -> Clip:
    yuv = tmp_path / "c.yuv"
    yuv.write_bytes(bytes(range(256)) * (16 * 8 * 3 // 2 * frames // 256 + 1))
    return Clip("c", yuv, 16, 8, 30.0, frames, 1000.0, "static", "val")


def test_a_raw_clips_frames_are_its_first_bytes_as_stored(tmp_path):
    clip = raw_clip(tmp_path)
    out = subprocess.run(frames_command(clip, 3), capture_output=True, check=True).stdout
    assert out == clip.yuv.read_bytes()[:3 * clip.frame_bytes]


def test_an_encoded_clip_is_decoded(tmp_path):
    clip = replace(raw_clip(tmp_path), yuv=tmp_path / "c.mp4")
    command = frames_command(clip, 3)
    assert command[0].endswith("ffmpeg") and "-frames:v" in command and str(clip.yuv) in command


def test_frames_are_read_from_frame_zero(tmp_path):
    with pytest.raises(ValueError):
        frames_command(replace(raw_clip(tmp_path), start_frame=2), 3)


def test_decode_lines_name_how_the_box_takes_each_clip(tmp_path):
    tool = heldout_tool()
    clip = raw_clip(tmp_path)
    assert tool.decode_line(clip, 150) == f"c c.yuv 4 {16 * 8 * 3 // 2} raw\n"
    encoded = replace(clip, yuv=tmp_path / "c.mp4")
    assert tool.decode_line(encoded, 2).split() == ["c", "c.mp4", "2", "192", "encoded"]


def test_the_box_command_copies_a_raw_clip_byte_for_byte(tmp_path):
    """prepare_command's own decode line, run here on a raw clip."""
    tool = heldout_tool()
    clip = raw_clip(tmp_path)
    command = tool.prepare_command("v", [], [clip], 3)
    decode = command.split("bash -c ", 1)[1].split(" _ < decode.txt", 1)[0]
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "c.yuv").write_bytes(clip.yuv.read_bytes())
    out = tmp_path / "out"
    out.mkdir()
    script = decode.strip("'").replace(tool.REMOTE_CLIPS, str(out))
    subprocess.run(["bash", "-c", script, "_", *tool.decode_line(clip, 3).split()],
                   cwd=tmp_path, check=True, capture_output=True)
    assert hashlib.sha256((out / "c.yuv").read_bytes()).hexdigest() == hashlib.sha256(
        clip.yuv.read_bytes()[:3 * clip.frame_bytes]).hexdigest()


def test_the_main_charts_clip_set_file_holds_its_clips(tmp_path):
    entry = {"yuv": "clips/main/{}.yuv", "width": 16, "height": 8, "fps": 30.0, "frames": 4,
             "source_kbps": 1000.0, "content_class": "static"}
    clips = [{**entry, "name": n, "yuv": entry["yuv"].format(n), "split": s}
             for n, s in (("a", "val"), ("b", "train"), ("c", "val"), ("d", "test"))]
    (tmp_path / "clips").mkdir()
    (tmp_path / "clips" / "main.json").write_text(json.dumps({"name": "main", "clips": clips}))
    paths = Paths(data_root=tmp_path, anchors=tmp_path, nanocodec_bin=tmp_path)
    out = write_natural_val_clipset(paths, tmp_path / "work" / "main-val.json")
    assert clipset_clips(str(out), paths) == natural_val(paths)
    assert [c.name for c in natural_val(paths)] == ["a", "c"]


def test_a_round_merges_only_the_results_it_brought_back(tmp_path):
    tool = heldout_tool()
    results, cache = tmp_path / "results", tmp_path / "cache"
    (results / "w/c/f").mkdir(parents=True)
    old, new = results / "w/c/f/old.json", results / "w/c/f/new.json"
    old.write_text("{}")
    new.write_text('{"a": 1}')
    tool.merge_results(results, cache, [new])
    assert (cache / "w/c/f/new.json").read_text() == '{"a": 1}'
    assert not (cache / "w/c/f/old.json").exists()
