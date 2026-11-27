#!/usr/bin/env bash
# encode.sh <clip-name> <config: ai|ldp|ra|min|minai> <qp> <frames> [extra avmenc args...]
# One avmenc v1.0.0 stream of a quick360 clip into the foreign-stream corpus
# (~/nanocodec-data/work/av2-foreign/streams), which compare.py decodes.
# Optional environment: NC_AV2_CLIPS, the clip directory (default quick360);
# TAG, a name part for streams made with extra arguments
# (<clip>.<config>.<TAG>.f<frames>.ivf).
# The corpus of 2026-11-15: three clips (bodycam_airport_360p,
# crowd_mot17_09_360p, pan_jaad_0117_360p) at QP 210: ai, ldp, min and minai
# with 10 frames, ra with 9, minai with 1.
set -euo pipefail
clip=$1; cfg=$2; qp=$3; frames=$4; shift 4
root=$HOME/nanocodec-data
yuv=${NC_AV2_CLIPS:-$root/clips/quick360}/$clip.yuv
out=$root/work/av2-foreign/streams/$clip.$cfg${TAG:+.$TAG}.f$frames.ivf
mkdir -p "$(dirname "$out")"
common=(--good --passes=1 --cpu-used=9 --threads=1 --row-mt=0 --end-usage=q "--qp=$qp"
        --disable-warning-prompt --bit-depth=8 --input-bit-depth=8 --width=640 --height=360
        --i420 --fps=30/1 "--limit=$frames" --ivf)
case $cfg in
  ai)  args=(--kf-min-dist=0 --kf-max-dist=0 --lag-in-frames=0) ;;
  ldp) args=(--lag-in-frames=0 --auto-alt-ref=0 --enable-fwd-kf=0 --kf-min-dist=9999 --kf-max-dist=9999 --enable-onesided-comp=0) ;;
  ra)  args=(--kf-min-dist=9999 --kf-max-dist=9999) ;;
  min*) args=(--lag-in-frames=0 --auto-alt-ref=0 --enable-fwd-kf=0 --kf-min-dist=9999 --kf-max-dist=9999
        --enable-keyframe-filtering=0 --monotonic-output-order=1 --tile-columns=0 --tile-rows=0 --aq-mode=0 --deltaq-mode=0 --enable-tpl-model=0
        --sb-size=64 --enable-sdp=0 --enable-extended-sdp=0 --enable-ext-partitions=0 --enable-uneven-4way-partitions=0
        --enable-rect-partitions=0 --min-partition-size=8
        --enable-intra-dip=0 --enable-intra-edge-filter=0 --enable-mrls=0 --enable-cfl-intra=0 --enable-mhccp=0
        --enable-ibp=0 --enable-palette=0 --enable-intrabc=0
        --enable-interintra-comp=0 --enable-warp-causal=0 --enable-warp-delta=0 --enable-warp-extend=0
        --enable-warped-motion=0 --enable-six-param-warp-delta=0 --enable-masked-comp=0 --enable-interinter-wedge=0
        --enable-diff-wtd-comp=0 --enable-smooth-interintra=0 --enable-interintra-wedge=0 --enable-onesided-comp=0
        --enable-ref-frame-mvs=0 --enable-tip=0 --enable-mv-traj=0 --enable-bawp=0 --enable-cwp=0
        --enable-imp-msk-bld=0 --enable-lf-sub-pu=0 --enable-opfl-refine=0 --enable-refinemv=0
        --enable-adaptive-mvd=0 --enable-mvd-sign-derive=0 --enable-flex-mvres=0 --enable-global-motion=0
        --enable-short-refresh-frame-flags=0 --enable-joint-mvd=0 --enable-skip-mode=0 --enable-bru=0
        --enable-refmvbank=0
        --enable-fsc=0 --enable-idtx-intra=0 --enable-ist=0 --enable-inter-ist=0 --enable-inter-ddt=0
        --enable-cctx=0 --enable-tcq=0 --enable-parity-hiding=0 --enable-avg-cdf=0 --enable-flip-idtx=0
        --use-inter-dct-only=1 --use-intra-dct-only=1 --enable-qm=0
        --enable-cdef=1 --enable-cdef-on-skip-txfm=2 --enable-gdf=0 --enable-restoration=0 --enable-ccso=0
        --enable-pc-wiener=0 --enable-wiener-nonsep=0) ;;
  *) echo "unknown config $cfg" >&2; exit 2 ;;
esac
start=$(date +%s)
# The minimal sets also take every switch the sequence-header probing found our decoder to need.
if [[ $cfg == min* ]]; then
  args+=(--max-pb-aspect-ratio=2 --select-adaptive-ds=0 --enable-drl-reorder=0 --max-drl-refbvs=2
         --enable-chroma-dctonly=1 --reduced-tx-part-set=1)
fi
# minai: the minimal tool set with every frame a key frame (later options override).
if [[ $cfg == minai ]]; then args+=(--kf-min-dist=0 --kf-max-dist=0); fi
"$root/anchors/vp9-av2/bin/avmenc" "${common[@]}" "${args[@]}" "$@" -o "$out" "$yuv" > "$out.log" 2>&1
echo "$clip.$cfg qp=$qp frames=$frames bytes=$(stat -c %s "$out") kbps=$(( $(stat -c %s "$out") * 8 * 30 / frames / 1000 )) secs=$(( $(date +%s) - start ))"
