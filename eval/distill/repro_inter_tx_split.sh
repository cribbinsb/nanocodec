# Repro of the open bug av1-inter-tx-split-starved-cbr-level-overrun (ideas ledger): run on the build box from the worktree root.
set -o pipefail
B=target/gate/nanocodec; Y=~/nanocodec-data/clips/broad_independent_v1/external_antare_refused_bwc_1080p.yuv
run() { # name preset kbps params...
  n=$1; p=$2; k=$3; shift 3; a=(); for x in "$@"; do a+=(--param "$x"); done
  s=$(date +%s.%N)
  $B encode --codec av1 --preset $p --input $Y --width 1920 --height 1080 --fps 10 --frames 100 --bitrate $k "${a[@]}" --output out/tx-$n.ivf --rate-log out/tx-$n.csv > out/tx-$n.err 2>&1; rc=$?
  e=$(date +%s.%N)
  bytes=$(stat -c %s out/tx-$n.ivf 2>/dev/null); maxq=$(cut -d, -f4 out/tx-$n.csv 2>/dev/null | tail -n +2 | sort -n | tail -1)
  echo "$n preset=$p kbps=$k rc=$rc bytes=$bytes kbps_out=$(( ${bytes:-0}*8/10/1000 )) maxqp=$maxq secs=$(echo "$e-$s"|bc) $(tail -c 200 out/tx-$n.err | tr '\n' ' ')"
}
cargo build -q --profile gate -p nanocodec-cli || exit 1
for k in 150 300; do
  run base$k fast $k
  run all$k fast $k av1.inter_tx_split=1 av1.transform_rd_gain_scale=1 av1.transform_rd_skip=0 av1.transform_rd_skip_chroma=0 av1.transform_rd_uv_max_noise_sigma=255
  run split$k fast $k av1.inter_tx_split=1
  run gain$k fast $k av1.transform_rd_gain_scale=1
  run skip$k fast $k av1.transform_rd_skip=0
  run skipc$k fast $k av1.transform_rd_skip_chroma=0
  run uv$k fast $k av1.transform_rd_uv_max_noise_sigma=255
done
run vfbase veryfast 150
run vfall veryfast 150 av1.inter_tx_split=1 av1.partial_forward=0 av1.transform_rd_gain_scale=1 av1.transform_rd_skip=0 av1.transform_rd_skip_chroma=0 av1.transform_rd_uv_max_noise_sigma=255
