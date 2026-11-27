#!/usr/bin/env bash
# Supplementary H.264 conformance streams for the Baseline and Extended
# tools the ITU-T H.264.1 suite we fetch covers thinly or not at all: data
# partitioning (no suite stream), redundant pictures (none), arbitrary
# slice order (none), SI slices (none) and slice groups with ASO (JM 19's
# SP pictures are not made: neither JM's nor FFmpeg's decode reproduces
# lencod's reconstruction of them; the suite's SP1_BT_A and sp2_bt_b cover
# SP slices). JM's
# encoder (lencod) makes them; their reference is lencod's reconstruction,
# kept only where a standard decoder's output equals it: JM's (ldecod), or
# for the streams JM decodes wrongly (redundant pictures: ldecod drops
# pictures) FFmpeg's. `--suite h264-jm` of tools/ci/conformance.py checks
# them (manifest: tools/dev/conformance_manifest.py --suite h264-jm).
#
#   JM_BIN=<dir with lencod, ldecod> tools/dev/h264_conformance_jm.sh [out-dir]
#
# Without JM_BIN, JM is cloned at JM_COMMIT and built in the scratch dir.
# ASO streams are JM streams whose slices are then put in reverse order
# within each picture (tools/dev/h264_slice_order.py); their reference is
# the decode of the stream in coding order. Output:
# <out-dir>/<name>/{<name>.264,<name>.yuv}, default
# ~/nanocodec-data/conformance/h264/jm. Runs on the build box (lencod takes
# about a minute a stream): nc-remote run --clip <the source clip> -- ...
set -euo pipefail

root=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-$HOME/nanocodec-data/conformance/h264/jm}
clip=${NC_JM_CLIP:-$HOME/nanocodec-data/clips/quick/crowd_mot17_09.yuv}
clip_size=1280x720
frames=12
JM_COMMIT=8b34eee1576952dc2a04cd2fdb52febfde4030b2  # JM 19.1, 2026-04-13
scratch=$(mktemp -d "${TMPDIR:-/tmp}/nanocodec-h264-conformance-jm-XXXXXX")
trap 'rm -rf "$scratch"' EXIT

if [[ -z ${JM_BIN:-} ]]; then
    git clone -q https://vcgit.hhi.fraunhofer.de/jvet/JM.git "$scratch/jm"
    git -C "$scratch/jm" checkout -q "$JM_COMMIT"
    make -C "$scratch/jm" -j"$(nproc)" >"$scratch/jm-build.log" 2>&1
    JM_BIN=$scratch/jm/bin
fi
lencod=$(ls "$JM_BIN"/lencod* | head -1)
ldecod=$(ls "$JM_BIN"/ldecod* | head -1)
cfg_dir=$(dirname "$lencod")/../cfg
[[ -d $cfg_dir ]] || cfg_dir=${JM_CFG:?JM_CFG: the JM cfg directory}

ffmpeg -y -hide_banner -loglevel error -nostdin -f rawvideo -pix_fmt yuv420p -s "$clip_size" \
    -i "$clip" -frames:v "$frames" -vf scale=176:144:flags=bicubic -f rawvideo \
    -pix_fmt yuv420p "$scratch/src.yuv"

common=(-p "InputFile=$scratch/src.yuv" -p "FramesToBeEncoded=$frames" -p SourceWidth=176
        -p SourceHeight=144 -p QPISlice=30 -p QPPSlice=31 -p QPBSlice=33 -p IntraPeriod=6)
extended=(-d "$cfg_dir/encoder_extended.cfg" -p ProfileIDC=88 -p LevelIDC=30)
baseline=(-d "$cfg_dir/encoder_baseline.cfg" -p ProfileIDC=66 -p LevelIDC=30)

# name | profile | lencod options | slice order (keep or reverse)
# (jm_ext_si: with lencod's RD optimisation on, JM's decode of its SI pictures
# differs from lencod's reconstruction; off, they agree.)
streams=(
  "jm_ext_dp_b|extended|-p PartitionMode=1 -p NumberBFrames=1|keep"
  "jm_ext_dp_cip|extended|-p PartitionMode=1 -p NumberBFrames=0 -p UseConstrainedIntraPred=1 -p RandomIntraMBRefresh=8|keep"
  "jm_ext_dp_slices|extended|-p PartitionMode=1 -p NumberBFrames=1 -p SliceMode=1 -p SliceArgument=33|keep"
  "jm_ext_dp_aso|extended|-p PartitionMode=1 -p NumberBFrames=0 -p SliceMode=1 -p SliceArgument=25|reverse"
  "jm_base_redundant|baseline|-p UseRedundantPicture=1 -p NumRedundantHierarchy=1 -p PrimaryGOPLength=4 -p NumberReferenceFrames=4|keep"
  "jm_base_aso|baseline|-p SliceMode=1 -p SliceArgument=20|reverse"
  "jm_base_fmo_dispersed_aso|baseline|-p num_slice_groups_minus1=1 -p slice_group_map_type=1 -p SliceMode=1 -p SliceArgument=20|reverse"
  "jm_ext_si|extended|-p NumberBFrames=0 -p SPPicturePeriodicity=3 -p SI_FRAMES=1 -p QPSISlice=32 -p QPSP2Slice=30 -p RDOptimization=0|keep"
)

mkdir -p "$out"
for entry in "${streams[@]}"; do
    IFS='|' read -r name profile options order <<<"$entry"
    dir=$out/$name
    mkdir -p "$dir" "$scratch/$name"
    if [[ $profile == extended ]]; then profile_options=("${extended[@]}"); else profile_options=("${baseline[@]}"); fi
    # shellcheck disable=SC2086 # options are a word list by design
    (cd "$scratch/$name" && "$lencod" "${profile_options[@]}" "${common[@]}" $options \
        -p OutputFile=coded.264 -p ReconFile=rec.yuv >lencod.log 2>&1) \
        || { tail -20 "$scratch/$name/lencod.log" >&2; echo "$name: lencod failed" >&2; exit 1; }
    (cd "$scratch/$name" && "$ldecod" -p InputFile=coded.264 -p OutputFile=jm.yuv >/dev/null 2>&1) || true
    ffmpeg -v quiet -nostdin -y -i "$scratch/$name/coded.264" -f rawvideo -pix_fmt yuv420p \
        "$scratch/$name/ffmpeg.yuv" || true
    agrees=
    for decoder in jm ffmpeg; do
        if cmp -s "$scratch/$name/rec.yuv" "$scratch/$name/$decoder.yuv"; then agrees=$decoder; break; fi
    done
    [[ -n $agrees ]] || { echo "$name: no decoder reproduces lencod's reconstruction" >&2; exit 1; }
    if [[ $order == reverse ]]; then
        python3 -I "$root/tools/dev/h264_slice_order.py" "$scratch/$name/coded.264" "$dir/$name.264"
    else
        cp "$scratch/$name/coded.264" "$dir/$name.264"
    fi
    mv "$scratch/$name/rec.yuv" "$dir/$name.yuv"
    echo "$name: $(stat -c %s "$dir/$name.264") bytes, $agrees decode == lencod reconstruction"
done
