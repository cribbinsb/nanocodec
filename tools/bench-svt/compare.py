#!/usr/bin/env python3
"""Join svt_bench output with nc-checkasm --bench output: ns/call side by side.

Usage: compare.py <svt_bench output> <nc-checkasm --bench output>

For each kernel the SVT harness timed, prints SVT-AV1's time, nanocodec's
best SIMD time for the same kernel and size, and their ratio (> 1 means
nanocodec is slower). Inputs may hold several rounds; the fastest time of
each kernel is kept.
"""

import sys


def main() -> None:
    svt = {}
    for line in open(sys.argv[1]):
        kernel, size, ns = line.split()
        svt[(kernel, size)] = min(float(ns), svt.get((kernel, size), float("inf")))
    ours = {}
    for line in open(sys.argv[2]):
        parts = line.split()
        if len(parts) < 4 or parts[2] == "scalar" or parts[0] == "kernel":
            continue
        try:
            ns = float(parts[3])
        except ValueError:
            continue
        key = (parts[0], parts[1])
        ours[key] = min(ns, ours.get(key, float("inf")))
    print(f"{'kernel':<22}{'size':>7}{'SVT-AV1 ns':>12}{'nanocodec ns':>14}{'ratio':>8}")
    for key, svt_ns in svt.items():
        our_ns = ours.get(key)
        if our_ns is None:
            print(f"{key[0]:<22}{key[1]:>7}{svt_ns:>12.1f}{'-':>14}")
            continue
        print(f"{key[0]:<22}{key[1]:>7}{svt_ns:>12.1f}{our_ns:>14.1f}{our_ns / svt_ns:>7.2f}x")


if __name__ == "__main__":
    main()
