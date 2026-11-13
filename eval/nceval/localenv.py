"""Machine-local settings that stay out of the repository, such as the Jetson's SSH target.

The public repository names no hosts or home paths. Each machine keeps them in
`$NC_DATA_ROOT/local.env` (default `~/nanocodec-data/local.env`), a shell file of
`export NAME=value` lines that shell tools source and Python tools read here. A
variable already in the environment wins over the file.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path

LOCAL_ENV_NAME = "local.env"


def local_env_file() -> Path:
    data_root = Path(os.environ.get("NC_DATA_ROOT", Path.home() / "nanocodec-data"))
    return data_root / LOCAL_ENV_NAME


def setting(name: str) -> str | None:
    """`name` from the environment, else from the local settings file, else None."""
    if os.environ.get(name):
        return os.environ[name]
    path = local_env_file()
    if not path.is_file():
        return None
    for line in path.read_text().splitlines():
        words = shlex.split(line, comments=True)
        if words[:1] == ["export"]:
            words = words[1:]
        for word in words:
            key, sep, value = word.partition("=")
            if sep and key == name:
                return value
    return None


def require(name: str) -> str:
    """`name`'s value; exits with a hint when it is set nowhere."""
    value = setting(name)
    if not value:
        raise SystemExit(f"{name} is not set: add `export {name}=...` to {local_env_file()} "
                         "or export it")
    return value
