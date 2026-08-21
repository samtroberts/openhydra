#!/bin/sh
# OpenHydra .deb/.rpm post-install (Layer 3): expose the bundled `openhydra-agent` sidecar as an
# `openhydra` command on PATH, so the Connectors terminal snippets and `openhydra launch` work out of
# the box for package installs — no in-app "Install CLI" step needed. See docs/CLI_ON_PATH_PLAN_v1.md.
#
# Idempotent; never fails the install (a missing sidecar just skips the link).
set -e

# Prefer PATH resolution (Tauri installs the sidecar into /usr/bin, so it's usually already there as
# `openhydra-agent`); fall back to the other locations Tauri may use.
AGENT="$(command -v openhydra-agent 2>/dev/null || true)"
if [ -z "$AGENT" ]; then
  for p in /usr/bin/openhydra-agent /usr/lib/openhydra/openhydra-agent /usr/lib/OpenHydra/openhydra-agent; do
    if [ -x "$p" ]; then AGENT="$p"; break; fi
  done
fi

if [ -n "$AGENT" ]; then
  ln -sf "$AGENT" /usr/bin/openhydra
fi

exit 0
