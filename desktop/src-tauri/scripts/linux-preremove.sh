#!/bin/sh
# OpenHydra .deb/.rpm pre-remove (Layer 3): drop the `openhydra` symlink we created in post-install.
# Only remove it if it's a symlink (never clobber a user's own real `openhydra` binary).
if [ -L /usr/bin/openhydra ]; then
  rm -f /usr/bin/openhydra
fi
exit 0
