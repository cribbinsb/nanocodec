#!/usr/bin/env python3
"""Which AWS instance a workload should rent, checked against today's prices before launch.

    python3 tools/dev/aws_choose.py WORKLOAD [--hours H] [--boxes N] [--root-gb G]
    python3 tools/dev/aws_choose.py all

WORKLOAD is one of WORKLOADS (docs/aws-cost-guide.md, the measured cost guide every tool
and agent follows). For each of the workload's candidates, in the guide's order, it
prints every region's cheapest spot zone price, the price per 1000 vCPU-hours, whether
the type is offered there, the region's free spot quota (vCPUs), and the estimated cost
of N boxes for H hours (instance + root volume + public IPv4, as aws_session bills it).
The pick is the first candidate offered in the home region with quota room; a spill
region is named only when the home region has none, with its price ratio, since spill
spot cost 2.8-3.6x eu-north-1's on 2026-11-24 (c7i.24xlarge $1.58-2.00/h against $0.56/h).

Read-only: it creates nothing (describe calls and the Service Quotas lookup only).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aws_session import (GP3_USD_PER_GB_MONTH, HOURS_PER_MONTH,  # noqa: E402
                         PUBLIC_IPV4_USD_PER_HOUR, SPILL_REGIONS, spot_quota, spot_usage,
                         utc_now)

HOME_REGION = "eu-north-1"
#: A spill region is worth naming only within this factor of the home price per vCPU
#: (aws_warm drains boxes above 1.5x for the same reason).
SPILL_PRICE_FACTOR = 1.5
DEFAULT_ROOT_GB = 16


@dataclass(frozen=True)
class Workload:
    """A workload's candidates, best first (docs/aws-cost-guide.md has the measurements)."""
    types: tuple[str, ...]
    note: str


WORKLOADS = {
    "timing": Workload(("c8i.2xlarge", "c8g.2xlarge"),
                       "benchmark absolute timing (BP-3.10): one encode at a time, spot"),
    "ab": Workload(("c8i.2xlarge", "c8g.2xlarge"),
                   "paired CPU A/B (aws_cpu_ab): the benchmark's two families, spot"),
    "packed": Workload(("c7i.24xlarge", "c6i.24xlarge"),
                       "untimed packed encodes and CPU metric scoring (x86 tools)"),
    "packed-arm": Workload(("c8g.8xlarge", "c8g.16xlarge"),
                           "packed encodes and scoring on Graviton4 (aws_measure --arch aarch64)"),
    "build": Workload(("c7i.24xlarge", "c6i.24xlarge"),
                      "the build box (nc-remote): x86, many concurrent jobs"),
    "build-arm": Workload(("c8g.8xlarge",),
                          "a native aarch64 build (cold release build 243 s, $0.011)"),
}


@dataclass(frozen=True)
class Row:
    """One candidate type in one region."""
    instance_type: str
    region: str
    vcpus: int
    price: float | None  # cheapest spot zone, USD/h; None: not offered
    quota_free: int

    def per_kvcpu(self) -> float | None:
        return None if self.price is None else 1000 * self.price / self.vcpus


def estimate(price: float, hours: float, boxes: int, root_gb: int) -> float:
    """The expected cost of N boxes for H hours at `price` (aws_session's cost model)."""
    storage = root_gb * GP3_USD_PER_GB_MONTH / HOURS_PER_MONTH
    return boxes * hours * (price + storage + PUBLIC_IPV4_USD_PER_HOUR)


def pick(rows: list[Row], types: tuple[str, ...], boxes: int) -> tuple[Row | None, str]:
    """The first candidate offered at home with quota room for `boxes`; else the cheapest
    spill row within SPILL_PRICE_FACTOR of the home price per vCPU; else none (wait)."""
    home = [r for t in types for r in rows
            if r.instance_type == t and r.region == HOME_REGION and r.price is not None]
    for row in home:
        if row.quota_free >= boxes * row.vcpus:
            return row, "home region"
    reference = min((r.per_kvcpu() for r in home), default=None)
    spill = sorted((r for r in rows if r.region != HOME_REGION and r.price is not None
                    and r.quota_free >= boxes * r.vcpus), key=Row.per_kvcpu)
    if spill and reference is not None and spill[0].per_kvcpu() <= SPILL_PRICE_FACTOR * reference:
        return spill[0], f"spill: {spill[0].per_kvcpu() / reference:.2f}x the home price per vCPU"
    if spill:
        ratio = (f"{spill[0].per_kvcpu() / reference:.1f}x home" if reference
                 else "not offered at home")
        return None, (f"no home room; cheapest spill {spill[0].instance_type} in "
                      f"{spill[0].region} is {ratio}: wait for home quota")
    return None, "no candidate has room in any region: wait"


def survey(types: tuple[str, ...]) -> list[Row]:
    """Today's rows for `types` in the home region and its spill regions."""
    import boto3
    rows = []
    for region in (HOME_REGION, *SPILL_REGIONS.get(HOME_REGION, ())):
        ec2 = boto3.client("ec2", region_name=region)
        free = spot_quota(region) - spot_usage(ec2)[0]
        offered = {o["InstanceType"] for o in ec2.describe_instance_type_offerings(
            LocationType="region", Filters=[{"Name": "instance-type", "Values": list(types)}])[
                "InstanceTypeOfferings"]}
        info = {i["InstanceType"]: i["VCpuInfo"]["DefaultVCpus"] for i in
                ec2.describe_instance_types(InstanceTypes=sorted(offered))["InstanceTypes"]} \
            if offered else {}
        history = ec2.describe_spot_price_history(
            InstanceTypes=sorted(offered), ProductDescriptions=["Linux/UNIX"],
            StartTime=utc_now())["SpotPriceHistory"] if offered else []
        for t in types:
            prices = [float(h["SpotPrice"]) for h in history if h["InstanceType"] == t]
            rows.append(Row(t, region, info.get(t, 1), min(prices) if prices else None, free))
    return rows


def report(name: str, hours: float, boxes: int, root_gb: int) -> None:
    work = WORKLOADS[name]
    rows = survey(work.types)
    print(f"== {name}: {work.note}")
    for r in rows:
        if r.price is None:
            print(f"  {r.instance_type:14s} {r.region:13s} not offered")
            continue
        print(f"  {r.instance_type:14s} {r.region:13s} ${r.price:.4f}/h  "
              f"${r.per_kvcpu():6.2f}/1000 vCPU-h  quota free {r.quota_free:5d}  "
              f"{boxes} x {hours:g} h ~ ${estimate(r.price, hours, boxes, root_gb):.2f}")
    chosen, why = pick(rows, work.types, boxes)
    print(f"  pick: {chosen.instance_type} in {chosen.region} ({why})" if chosen
          else f"  pick: none ({why})")


def main(argv: list[str]) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("workload", choices=[*WORKLOADS, "all"])
    p.add_argument("--hours", type=float, default=1.0)
    p.add_argument("--boxes", type=int, default=1)
    p.add_argument("--root-gb", type=int, default=DEFAULT_ROOT_GB)
    a = p.parse_args(argv)
    for name in (WORKLOADS if a.workload == "all" else (a.workload,)):
        report(name, a.hours, a.boxes, a.root_gb)


if __name__ == "__main__":
    main(sys.argv[1:])
