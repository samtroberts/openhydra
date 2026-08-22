#!/bin/sh
# OpenHydra .deb/.rpm pre-remove (Layer 3): drop the `openhydra` symlink we created in post-install.
# Only remove OUR link — a symlink that resolves to an openhydra-agent binary — never a real binary
# or a symlink someone else created.
if [ -L /usr/bin/openhydra ]; then
  tgt="$(readlink -f /usr/bin/openhydra 2>/dev/null || true)"
  case "$tgt" in
    */openhydra-agent) rm -f /usr/bin/openhydra ;;
  esac
fi
exit 0
