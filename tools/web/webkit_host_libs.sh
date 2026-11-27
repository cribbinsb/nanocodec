#!/usr/bin/env bash
# Playwright's WebKit on an Ubuntu 26.04 host without root (this PC; the build
# box). Playwright 1.60 ships only its Ubuntu 24.04 WebKit (WPE MiniBrowser)
# here, which needs five Ubuntu 24.04 libraries 26.04 no longer has (libicu74,
# libxml2 2.9, libevent 2.1, libmanette, libwoff1). This unpacks them, SHA-256
# checked, under $NC_DATA_ROOT/tools/webkit-noble-libs/root, where
# tools/web/webkit-minibrowser.sh (the executable tools/web/browser_engines.py
# gives WebKit) puts them last on the library path. Idempotent.
set -euo pipefail
dest=${NC_DATA_ROOT:-$HOME/nanocodec-data}/tools/webkit-noble-libs
[[ -f $dest/root/.complete ]] && exit 0
pool=http://archive.ubuntu.com/ubuntu/pool/main
debs=(
  "i/icu/libicu74_74.2-1ubuntu3.1_amd64.deb c9a70989678660eed9a1e904c74fa043da8bec8e2036856fc16e31ced79b04f8"
  "libx/libxml2/libxml2_2.9.14+dfsg-1.3ubuntu3.10_amd64.deb c5a065f85c1e6f605741e365e40cad71c3ceb0e85e2791400bd35b8896e12ad1"
  "libe/libevent/libevent-2.1-7t64_2.1.12-stable-9ubuntu2.2_amd64.deb 083ced6efb23476cb4de4e441b741af02db5653e67fa7fcd6cf7e5f7b78f6101"
  "libm/libmanette/libmanette-0.2-0_0.2.7-1build2_amd64.deb 8fbb67e94abf563c01398ad7458df1a6824f19a0b4337f67d61c0889771805db"
  "w/woff2/libwoff1_1.0.2-2build1_amd64.deb e23a6287ff611ff60a5a598de21f8f60f6236b6ad9443376318e6cc7914219c1"
)
mkdir -p "$dest/debs" "$dest/root"
for entry in "${debs[@]}"; do
  read -r path sum <<< "$entry"
  file=$dest/debs/$(basename "$path")
  [[ -f $file ]] || curl -fsS -o "$file" "$pool/$path"
  echo "$sum  $file" | sha256sum -c --quiet
  dpkg-deb -x "$file" "$dest/root"
done
touch "$dest/root/.complete"
