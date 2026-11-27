#!/usr/bin/env python3
"""Bake the measurement image: the warm boxes' packages installed once, per architecture.

    python3 tools/dev/aws_bake.py [--arch arm64|amd64|both] [--region eu-north-1]

Run it whenever measure_image.package_steps() changes (the SetupHash in the image's tags no
longer matches, so warm boxes fall back to the stock image and install as before). It
launches one small spot box of the architecture from the stock Ubuntu image, runs the
package steps, marks them done (measure_image.MARKER_DIR), creates an image tagged
Project=nanocodec, Purpose=measure-image, Arch, SetupHash, waits until it is available,
then deregisters every older measure image of that architecture and deletes its snapshots.
The box is terminated by its session on every exit. Cost: a few cents per architecture.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_image  # noqa: E402
from aws_session import PROJECT_TAG, Session, SessionSpec, say, utc_now  # noqa: E402

#: The smallest spot box of each architecture that installs the packages in minutes.
#: (The cost guide's allowed types: the smallest of each architecture.)
BAKE_TYPES = {"arm64": ("c8g.large", "c7g.large"),
              "amd64": ("c7i.4xlarge",)}
BAKE_BUDGET_USD = 0.30
BAKE_LIFETIME_MINUTES = 45
INSTALL_MINUTES = 20
IMAGE_WAIT_MINUTES = 25


def bake(arch: str, region: str) -> str:
    tags = {"Project": PROJECT_TAG, **measure_image.image_tags(arch)}
    types = BAKE_TYPES[arch]
    spec = SessionSpec(region=region, instance_type=types[0], arch=arch,
                       purpose=measure_image.IMAGE_PURPOSE, lifetime_minutes=BAKE_LIFETIME_MINUTES,
                       budget_usd=BAKE_BUDGET_USD, root_gb=16, fallback_types=types[1:],
                       launch_wait_minutes=5)
    with Session(spec) as session:
        marker = f"{measure_image.MARKER_DIR}/{measure_image.setup_hash()}"
        session.run_detached(" && ".join(measure_image.package_steps())
                             + f" && sudo mkdir -p {measure_image.MARKER_DIR} && sudo touch {marker}"
                             + " && sudo apt-get -qq clean && sync",
                             "bake.log", INSTALL_MINUTES)
        name = f"nanocodec-measure-{arch}-{measure_image.setup_hash()}-{utc_now():%Y%m%d%H%M}"
        image = session.ec2.create_image(InstanceId=session.instance_id, Name=name, NoReboot=True,
                                         TagSpecifications=[
                                             {"ResourceType": "image", "Tags": tag_list(tags)},
                                             {"ResourceType": "snapshot", "Tags": tag_list(tags)}])
        image_id = image["ImageId"]
        say(f"image {image_id} ({name}): waiting until available")
        session.ec2.get_waiter("image_available").wait(
            ImageIds=[image_id], WaiterConfig={"Delay": 15, "MaxAttempts": IMAGE_WAIT_MINUTES * 4})
        say(f"image {image_id} available")
    retire_older(session.ec2, arch, keep=image_id)
    return image_id


def tag_list(tags: dict[str, str]) -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in tags.items()]


def retire_older(ec2, arch: str, keep: str) -> None:
    """Deregister every other measure image of `arch` (ours only) and delete its snapshots."""
    images = ec2.describe_images(Owners=["self"], Filters=[
        {"Name": "tag:Project", "Values": [PROJECT_TAG]},
        {"Name": "tag:Purpose", "Values": [measure_image.IMAGE_PURPOSE]},
        {"Name": "tag:Arch", "Values": [arch]}])["Images"]
    for image in images:
        if image["ImageId"] == keep:
            continue
        snapshots = [m["Ebs"]["SnapshotId"] for m in image.get("BlockDeviceMappings", [])
                     if "Ebs" in m and "SnapshotId" in m["Ebs"]]
        ec2.deregister_image(ImageId=image["ImageId"])
        time.sleep(2)
        for snapshot in snapshots:
            ec2.delete_snapshot(SnapshotId=snapshot)
        say(f"retired image {image['ImageId']} and {len(snapshots)} snapshot(s)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arch", choices=("arm64", "amd64", "both"), default="both")
    ap.add_argument("--region", default="eu-north-1")
    a = ap.parse_args()
    for arch in (("arm64", "amd64") if a.arch == "both" else (a.arch,)):
        print(f"{arch}: {bake(arch, a.region)}", flush=True)


if __name__ == "__main__":
    main()
