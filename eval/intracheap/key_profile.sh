#!/bin/bash
# One key frame, timed and profiled with hardware cycles (perf), ours beside
# SVT-AV1's still-picture mode: whole-process CPU seconds, cycles, and the
# self-cycle table by symbol. Usage: key_profile.sh BIN CLIP.yuv W H OUT_DIR
set -eu
BIN=$1; CLIP=$2; W=$3; H=$4; OUT=$5
SVT=${SVT:-$HOME/nanocodec-data/anchors/x86_64/bin/SvtAv1EncApp}
mkdir -p "$OUT"
ours() { # codec qp [params...]
  local codec=$1 qp=$2; shift 2
  echo "$BIN" encode --codec "$codec" --input "$CLIP" --width "$W" --height "$H" --fps 30/1 \
    --frames 1 --intra-period 1 --temporal-layers 1 --output "$OUT/o.bin" --qp "$qp" "$@"
}
svt() { # preset qp
  echo "$SVT" --avif 1 --preset "$1" --rc 0 --aq-mode 0 --qp "$2" --keyint 1 --scd 0 --lp 1 \
    --fps-num 30 --fps-denom 1 -w "$W" -h "$H" -n 1 -i "$CLIP" -b "$OUT/s.bin"
}
timeit() { # label command...
  local label=$1; shift
  rm -f -- "$OUT/o.bin" "$OUT/s.bin"
  sudo -n perf stat -r 5 -x, -e task-clock,cycles,instructions -o "$OUT/stat.txt" -- taskset -c 5 "$@" > /dev/null 2>&1
  echo "$label bytes=$(stat -c %s "$OUT"/[os].bin | tail -1) $(awk -F, '{printf "%s=%s ", $3, $1}' "$OUT/stat.txt")"
}
record() { # label command...
  local label=$1; shift
  sudo -n perf record -q -g -c 100000 -e cycles -o "$OUT/$label.data" -- taskset -c 5 "$@" > "$OUT/$label.record.log" 2>&1
  sudo -n perf report -i "$OUT/$label.data" --no-children --sort symbol --stdio --percent-limit 0.4 2>/dev/null \
    | grep -E '^ +[0-9]' | head -70 > "$OUT/$label.self.txt"
  sudo -n perf report -i "$OUT/$label.data" --children --sort symbol --stdio --percent-limit 2 -g none 2>/dev/null \
    | grep -E '^ +[0-9]' | head -90 > "$OUT/$label.incl.txt"
  sudo -n rm -f -- "$OUT/$label.data"
}
for qp in ${OUR_QPS:-120 180}; do
  for codec in ${CODECS:-av1}; do
    timeit "ours-$codec-q$qp" $(ours "$codec" "$qp")
    record "ours-$codec-q$qp" $(ours "$codec" "$qp")
  done
done
for qp in ${SVT_QPS:-35 50}; do
  for preset in 8 6 4; do
    timeit "svt-m$preset-q$qp" $(svt "$preset" "$qp")
  done
  record "svt-m8-q$qp" $(svt 8 "$qp")
  record "svt-m4-q$qp" $(svt 4 "$qp")
done
rm -f -- "$OUT/o.bin" "$OUT/s.bin" "$OUT/stat.txt"
