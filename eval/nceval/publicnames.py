"""The source credit the site prints beside a clip: the dataset or collection it comes
from, by name (never a path), with the owners Xiph.org hosts credited themselves.

Clip and collection names are published as they are (owner, 2026-11-15). What this
module fixes is the credit: Netflix's El Fuente / Chimera clips and SVT's were filed
under "xiph", the folder they were fetched into.
"""

from __future__ import annotations

from pathlib import Path

#: Collection folder (lower case) -> its credit, where the folder name is not one.
CREDITS = {
    "svt": "SVT Open Content",
    "mot": "MOTChallenge (MOT17 / MOT20)",
    "bdd100k_mot": "BDD100K MOT",
    "jaad": "JAAD",
    "personpath22": "PersonPath22",
    "meva": "MEVA",
    "wildtrack": "WILDTRACK",
    "uvg_vcm": "UVG-VCM",
    "chad": "CHAD",
    "caucafall": "CAUCAFall",
    "gmdcsa24": "GMDCSA-24",
    "aboda": "ABODA",
    "chirla": "CHIRLA",
}
#: Xiph.org hosts several owners' clips: Netflix's (El Fuente, Chimera) are credited to it.
XIPH, NETFLIX_FOLDER = "xiph", "netflix"
XIPH_CREDIT = "Xiph.org test media (derf's collection)"
NETFLIX_CREDIT = "Netflix Open Content (El Fuente / Chimera), via Xiph.org"
#: Files of our own store named after the public MOTChallenge sequence they are.
MOT_PREFIX = "mot"
#: Folders of our video store that are not a collection.
STORE_FOLDERS = {"base", "mp4", "video", "excerpts", "clips", "other", "videos"}
OWN_STORE = "own collection"


def credit(source: str) -> str:
    """The collection a clip whose source file is `source` comes from."""
    path = Path(source)
    folders = [part.lower() for part in path.parts[:-1]]
    if XIPH in folders:
        return NETFLIX_CREDIT if NETFLIX_FOLDER in folders else XIPH_CREDIT
    for folder in reversed(folders):
        if folder in CREDITS:
            return CREDITS[folder]
    stem = path.stem.lower()
    if stem.startswith(MOT_PREFIX) and stem[len(MOT_PREFIX):len(MOT_PREFIX) + 2].isdigit():
        return CREDITS["mot"]
    named = [part for part in path.parts[2:-1] if part.lower() not in STORE_FOLDERS
             and part not in ("downloaded_datasets", "tracking_original")]
    return named[0] if named else OWN_STORE
