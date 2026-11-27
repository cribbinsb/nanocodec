"""tools/dev/aws_svc_encodes.py: a spot reclaim mid-run keeps the streams already pulled
back, and the rerun uploads and runs only the plan's commands still missing (nothing is
rented: the session is a stand-in)."""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools" / "dev"))
import aws_svc_encodes  # noqa: E402
from aws_session import RemoteFailure  # noqa: E402

STEMS = ("a-300", "a-600", "b-300")


class FakeSession:
    """A box that codes `codes` of the uploaded plan's streams into the pulled directory,
    then is reclaimed when `reclaim` is set."""

    uploads: list[list[str]] = []

    def __init__(self, spec, codes: int, reclaim: bool):
        self.codes, self.reclaim, self.out = codes, reclaim, None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def upload(self, root: Path, remote: str) -> None:
        self.plan = (root / "plan.sh").read_text().splitlines()
        FakeSession.uploads.append(self.plan)

    @contextlib.contextmanager
    def pulling(self, remote: str, local: Path, marker: str):
        self.out = local
        yield

    def run_detached(self, command: str, log: str, minutes: float) -> None:
        for line in self.plan[:self.codes]:
            stem = aws_svc_encodes.OUTPUT.search(line)[1]
            (self.out / f"{stem}.ivf").write_bytes(b"stream")
        if self.reclaim:
            raise RemoteFailure("the box was reclaimed (spot interruption)", [])


def run(monkeypatch, tmp_path: Path, codes: int, reclaim: bool) -> int:
    monkeypatch.setattr(aws_svc_encodes, "Session",
                        lambda spec: FakeSession(spec, codes, reclaim))
    # the clips' upload is not under test: only the plan goes up
    monkeypatch.setattr(aws_svc_encodes, "stage", lambda lines, binary, root: (
        (root / "plan.sh").write_text("".join(f"{line}\n" for line in lines))))
    monkeypatch.setattr(sys, "argv", ["aws_svc_encodes.py", "--plan", str(tmp_path / "plan"),
                                      "--binary", str(tmp_path / "nanocodec"),
                                      "--out", str(tmp_path / "out")])
    return aws_svc_encodes.main()


def test_a_reclaim_mid_run_reruns_only_the_streams_not_back(monkeypatch, tmp_path):
    FakeSession.uploads = []
    (tmp_path / "plan").write_text("".join(
        f"$NC_SVC_BIN --input $NC_SVC_CLIPS/{stem[0]}.yuv --output $NC_SVC_OUT/{stem}.ivf\n"
        for stem in STEMS))
    with pytest.raises(RemoteFailure):
        run(monkeypatch, tmp_path, codes=2, reclaim=True)
    back = sorted(p.stem for p in (tmp_path / "out").iterdir())
    assert back == ["a-300", "a-600"]  # pulled before the reclaim, kept
    assert run(monkeypatch, tmp_path, codes=3, reclaim=False) == 0
    assert len(FakeSession.uploads) == 2 and len(FakeSession.uploads[0]) == 3
    assert [aws_svc_encodes.OUTPUT.search(line)[1] for line in FakeSession.uploads[1]] == \
        ["b-300"]
    assert sorted(p.stem for p in (tmp_path / "out").iterdir()) == sorted(STEMS)
    assert run(monkeypatch, tmp_path, codes=0, reclaim=False) == 0  # all back: no box
    assert len(FakeSession.uploads) == 2
