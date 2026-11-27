#!/bin/sh
# WebKit's executable for tools/web/browser_engines.py on a host that lacks
# its Ubuntu 24.04 libraries: Playwright's own MiniBrowser wrapper, with the
# libraries tools/web/webkit_host_libs.sh unpacked appended to the library
# path. NC_WEBKIT_DIR is Playwright's webkit-NNNN directory, NC_WEBKIT_LIBS the
# unpacked libraries' directory.
wk="${NC_WEBKIT_DIR:?}/minibrowser-wpe"
export WEBKIT_EXEC_PATH="$wk/bin"
export WEBKIT_INJECTED_BUNDLE_PATH="$wk/lib"
export WEBKIT_INSPECTOR_RESOURCES_PATH="$wk/share"
export LD_LIBRARY_PATH="$wk/lib:$wk/sys/lib:${NC_WEBKIT_LIBS:?}"
export WEBKIT_FORCE_COMPLEX_TEXT=1
# Never a window on the user's session (Playwright passes --headless as well).
unset DISPLAY WAYLAND_DISPLAY
exec "$wk/bin/MiniBrowser" "$@"
