#!/usr/bin/env bash
# Build openhydra-scorer: llama.cpp's llama-server at the commit in LLAMACPP_COMMIT, with OpenHydra's
# prompt-scoring patch (POST /v1/prompt_logprobs, a startup memory check, --exit-on-stdin-eof), its
# loader for the Qwen3.5 GGUFs Ollama publishes (llamacpp-ollama-compat.patch) and tokenizer-only
# subcommands (llamacpp-token-count.patch), as one self-contained binary the agent runs as its
# side-scorer (agent/src/side_scorer.rs) and a gateway counts tokens with (agent/src/token_count.rs).
#
#   scorer/build.sh <out-path>
#
# Portable on purpose: GGML_NATIVE=OFF, so the build machine's CPU features don't end up in users'
# binaries (x86 builds target AVX2; Apple builds run on every Apple Silicon Mac), static, and on
# macOS with the Metal shaders embedded. LLAMACPP_SRC=<pristine llama.cpp checkout at that commit>
# skips the fetch (offline builds). Also writes <out>.notices.txt: the licences of the code compiled in.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?usage: scorer/build.sh <out-path>}"
COMMIT="$(tr -d ' \r\n' < "$HERE/LLAMACPP_COMMIT")"
PATCH="$HERE/llamacpp-prompt-scoring.patch"
# Loads the Qwen3.5 GGUFs Ollama publishes (its own layout; see the patch), applied after PATCH.
COMPAT_PATCH="$HERE/llamacpp-ollama-compat.patch"
# --export-vocab / --count-tokens: a verifier's tokenizer, without the model's weights.
COUNT_PATCH="$HERE/llamacpp-token-count.patch"
WORK="${SCORER_WORK:-$HERE/.build}"
SRC="$WORK/llama.cpp"
mkdir -p "$WORK"

if [[ -n "${LLAMACPP_SRC:-}" ]]; then
  # Only the pinned commit, unmodified: what the notices and the patches were checked against.
  HEAD_COMMIT="$(git -C "$LLAMACPP_SRC" rev-parse HEAD 2>/dev/null)" \
    || { echo "scorer/build.sh: LLAMACPP_SRC ($LLAMACPP_SRC) isn't a git checkout" >&2; exit 1; }
  [[ "$HEAD_COMMIT" == "$COMMIT" ]] \
    || { echo "scorer/build.sh: LLAMACPP_SRC is at $HEAD_COMMIT, not the pinned $COMMIT" >&2; exit 1; }
  git -C "$LLAMACPP_SRC" diff --quiet HEAD -- \
    || { echo "scorer/build.sh: LLAMACPP_SRC has local changes; use a pristine checkout" >&2; exit 1; }
  # Nor untracked or ignored files (they'd be copied in), top-level build directories aside, which
  # aren't copied (review round 4, R4-8).
  EXTRA="$(git -C "$LLAMACPP_SRC" status --porcelain --untracked-files=all --ignored | grep -Ev '^(\?\?|!!) build[^/]*/' || true)"
  [[ -z "$EXTRA" ]] \
    || { echo "scorer/build.sh: LLAMACPP_SRC has files outside the pinned commit; use a pristine checkout:" >&2; echo "$EXTRA" | head -5 >&2; exit 1; }
  rm -rf "$SRC"
  mkdir -p "$SRC"
  # Everything but top-level build directories (a tar --exclude 'build*' would also drop
  # cmake/build-info.cmake and friends on BSD tar).
  for entry in "$LLAMACPP_SRC"/* "$LLAMACPP_SRC"/.[!.]*; do
    [[ -e "$entry" ]] || continue
    case "$(basename "$entry")" in build*) ;; *) cp -R "$entry" "$SRC/" ;; esac
  done
  (cd "$SRC" && patch -p1 --forward --silent < "$PATCH" && patch -p1 --forward --silent < "$COMPAT_PATCH" \
    && patch -p1 --forward --silent < "$COUNT_PATCH")
else
  if [[ ! -d "$SRC/.git" ]]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
  git -C "$SRC" checkout -q --force FETCH_HEAD
  git -C "$SRC" clean -qfdx -e 'build-scorer'
  git -C "$SRC" apply "$PATCH"
  git -C "$SRC" apply "$COMPAT_PATCH"
  git -C "$SRC" apply "$COUNT_PATCH"
fi

FLAGS=(
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_NATIVE=OFF -DGGML_CUDA=OFF
  -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
)
case "$(uname -s)" in
  # Without a deployment target the binary requires the build machine's macOS (dyld refuses it on
  # anything older); 13.3 is what llama.cpp's own macOS release builds target.
  Darwin) FLAGS+=(-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3) ;;
  # The C/C++ runtime linked in statically: MSVC's default (/MD) needs the VC++ redistributable's
  # vcruntime140(_1).dll and msvcp140.dll, which Windows doesn't ship. And ggml's own thread pool
  # instead of OpenMP (no OpenMP runtime DLL either).
  MINGW*|MSYS*|CYGWIN*) FLAGS+=(-DGGML_OPENMP=OFF -DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded) ;;
  # ggml's own thread pool instead of OpenMP: no libgomp the user's system might lack.
  *) FLAGS+=(-DGGML_OPENMP=OFF) ;;
esac
cmake -S "$SRC" -B "$SRC/build-scorer" "${FLAGS[@]}" > "$WORK/configure.log" 2>&1 \
  || { tail -30 "$WORK/configure.log" >&2; exit 1; }
# An explicit job count: a bare --parallel is an unbounded `make -j` with the Makefile generator, which
# runs every compile at once and runs a CI runner out of memory.
JOBS="${SCORER_JOBS:-$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)}"
cmake --build "$SRC/build-scorer" --target llama-server --config Release --parallel "$JOBS" > "$WORK/build.log" 2>&1 \
  || { tail -30 "$WORK/build.log" >&2; exit 1; }

BIN="$(find "$SRC/build-scorer/bin" -type f \( -name llama-server -o -name llama-server.exe \) | head -n 1)"
[[ -n "$BIN" ]] || { echo "scorer/build.sh: no llama-server built (see $WORK/build.log)" >&2; exit 1; }
# The patches must be in (the agent starts it with these flags) before it replaces $OUT.
# Captured first: piped straight into `grep -q`, the binary can die of SIGPIPE once grep has matched,
# and pipefail would then fail the check of a binary that passes it.
HELP="$("$BIN" --help 2>&1 || true)"
grep -q -- '--exit-on-stdin-eof' <<<"$HELP" || { echo "scorer/build.sh: $BIN lacks the OpenHydra patch" >&2; exit 1; }
USAGE="$("$BIN" --count-tokens 2>&1 || true)"
grep -q -- '--count-tokens <vocab.gguf>' <<<"$USAGE" || { echo "scorer/build.sh: $BIN lacks the token-count patch" >&2; exit 1; }
# Licence notices of everything compiled in, shipped beside it (<out>.notices.txt, without .exe).
"$(command -v python3 || command -v python)" "$HERE/notices.py" llamacpp "$SRC" "$WORK/notices.txt"
mkdir -p "$(dirname "$OUT")"
cp "$BIN" "$OUT"
cp "$WORK/notices.txt" "${OUT%.exe}.notices.txt"
echo "scorer/build.sh: built $OUT (llama.cpp ${COMMIT:0:7} + prompt-scoring patch)"
