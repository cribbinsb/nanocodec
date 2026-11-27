"""The measurement boxes' packages, and the baked image that holds them.

A fresh warm box spent most of its ~7 min readiness installing these packages on the stock
Ubuntu image (2026-11-25, a staged screen's two rounds). `tools/dev/aws_bake.py` bakes them
once per architecture into our own image (tagged Project=nanocodec, Purpose=IMAGE_PURPOSE,
Arch, SetupHash); warm boxes launch from it when its SetupHash matches PACKAGE_STEPS, and
box setup skips the install when the image's marker says it is done. Any change to
PACKAGE_STEPS changes the hash: boxes fall back to the stock image (and install as before)
until the image is baked again.
"""

from __future__ import annotations

import hashlib

#: Venv and pip packages apt does not package (tune's optuna, imported by nceval.screen).
REMOTE_HOME = "/home/ubuntu"
REMOTE_VENV = f"{REMOTE_HOME}/venv"
PIP_PACKAGES = ("optuna",)
#: Where a baked image marks its packages as installed (one file per SetupHash).
MARKER_DIR = "/opt/nanocodec-baked"
IMAGE_PURPOSE = "measure-image"


def package_steps() -> list[str]:
    """Every package a measurement box needs, as shell commands (box setup's first part)."""
    return [
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq update",
        "sudo DEBIAN_FRONTEND=noninteractive apt-get -qq install -y python3-numpy "
        "python3-scipy python3-opencv python3-xxhash python3-venv rsync ffmpeg valgrind "
        "> /dev/null",
        f"python3 -m venv --system-site-packages {REMOTE_VENV}",
        f"{REMOTE_VENV}/bin/pip install -q {' '.join(PIP_PACKAGES)}",
    ]


def setup_hash() -> str:
    return hashlib.sha256("\n".join(package_steps()).encode()).hexdigest()[:16]


def install_unless_baked() -> str:
    """One shell command: the package steps, unless this image is baked with this hash."""
    return (f"if [ -f {MARKER_DIR}/{setup_hash()} ]; then echo 'packages: baked image'; "
            f"else {' && '.join(package_steps())}; fi")


def image_tags(arch: str) -> dict[str, str]:
    """The tags of the baked image a box of `arch` ("amd64"/"arm64") launches from."""
    return {"Purpose": IMAGE_PURPOSE, "Arch": arch, "SetupHash": setup_hash()}
