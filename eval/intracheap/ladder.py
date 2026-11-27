"""Key BD-J and key CPU of every family of a one-frame `nceval.fixedqp gap` result.

    python3 eval/intracheap/ladder.py RESULT.json

Per anchor (each codec's medium key): the test family's J % against it and its
whole-process CPU per key, in ms and as a multiple of the anchor's in the same run.
"""
import json
import sys


def main(path: str) -> None:
    result = json.load(open(path))
    seconds: dict[str, list[float]] = {}
    for key, values in result["cpu_s"].items():
        seconds.setdefault(key.split("|", 1)[0], []).extend(values)
    ms = {family: 1000 * sum(v) / len(v) for family, v in seconds.items()}
    rows = []
    for pair, value in result["pairs"].items():
        test, anchor = pair.split(" vs ")
        s = value["summary"]
        rows.append((anchor, ms[test], test, s["j"], s["lo"], s["hi"]))
    anchor = None
    for row in sorted(rows):
        if row[0] != anchor:
            anchor = row[0]
            print(f"\n{anchor}: {ms[anchor]:.1f} ms per key process")
        _, cpu, test, j, lo, hi = row
        print(f"  {test:<34} J {j:+6.1f} [{lo:+5.1f}, {hi:+5.1f}]  {cpu:7.1f} ms  x{cpu / ms[anchor]:.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
