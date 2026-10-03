#!/usr/bin/env bash
# Build openhydra-scorer: llama.cpp's llama-server at the commit in LLAMACPP_COMMIT, with OpenHydra's
# prompt-scoring patch (POST /v1/prompt_logprobs, a startup memory check, --exit-on-stdin-eof), as
# one self-contained binary the agent runs as its side-scorer (agent/src/side_scorer.rs).
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
WORK="${SCORER_WORK:-$HERE/.build}"
SRC="$WORK/llama.cpp"
mkdir -p "$WORK"

if [[ -n "${LLAMACPP_SRC:-}" ]]; then
  rm -rf "$SRC"
  mkdir -p "$SRC"
  # Everything but top-level build directories (a tar --exclude 'build*' would also drop
  # cmake/build-info.cmake and friends on BSD tar).
  for entry in "$LLAMACPP_SRC"/* "$LLAMACPP_SRC"/.[!.]*; do
    [[ -e "$entry" ]] || continue
    case "$(basename "$entry")" in build*) ;; *) cp -R "$entry" "$SRC/" ;; esac
  done
  (cd "$SRC" && patch -p1 --forward --silent < "$PATCH")
else
  if [[ ! -d "$SRC/.git" ]]; then
    git init -q "$SRC"
    git -C "$SRC" remote add origin https://github.com/ggml-org/llama.cpp
  fi
  git -C "$SRC" fetch -q --depth 1 origin "$COMMIT"
  git -C "$SRC" checkout -q --force FETCH_HEAD
  git -C "$SRC" clean -qfdx -e 'build-scorer'
  git -C "$SRC" apply "$PATCH"
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
cmake -S "$SRC" -B "$SRC/build-scorer" "${FLAGS[@]}" > "$WORK/configure.log"
cmake --build "$SRC/build-scorer" --target llama-server --config Release --parallel > "$WORK/build.log"

BIN="$(find "$SRC/build-scorer/bin" -type f \( -name llama-server -o -name llama-server.exe \) | head -n 1)"
[[ -n "$BIN" ]] || { echo "scorer/build.sh: no llama-server built (see $WORK/build.log)" >&2; exit 1; }
# The patch must be in (the agent starts it with these flags) before it replaces $OUT.
"$BIN" --help 2>&1 | grep -q -- '--exit-on-stdin-eof' || { echo "scorer/build.sh: $BIN lacks the OpenHydra patch" >&2; exit 1; }
# Licence notices of everything compiled in, shipped beside it (<out>.notices.txt, without .exe).
"$(command -v python3 || command -v python)" "$HERE/notices.py" llamacpp "$SRC" "$WORK/notices.txt"
mkdir -p "$(dirname "$OUT")"
cp "$BIN" "$OUT"
cp "$WORK/notices.txt" "${OUT%.exe}.notices.txt"
echo "scorer/build.sh: built $OUT (llama.cpp ${COMMIT:0:7} + prompt-scoring patch)"
