#!/bin/bash
set -euo pipefail

# Build BO C11 core as WASM via Emscripten
# Prerequisites: emsdk activated (emcc in PATH)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/wasm"

# Verify prerequisites
if ! command -v emcc &> /dev/null; then
  echo "ERROR: emcc not found. Activate emsdk first:"
  echo "  source /path/to/emsdk/emsdk_env.sh"
  exit 1
fi

echo "=== Compiling WASM ==="
mkdir -p "$OUTPUT_DIR"

EXPORTED_FUNCTIONS='[
  "_wl_bo_get_last_error",
  "_wl_bo_space_create",
  "_wl_bo_space_add_continuous",
  "_wl_bo_space_add_integer",
  "_wl_bo_space_add_categorical",
  "_wl_bo_space_add_condition",
  "_wl_bo_space_finalize",
  "_wl_bo_space_free",
  "_wl_bo_create",
  "_wl_bo_observe",
  "_wl_bo_suggest",
  "_wl_bo_suggest_liar",
  "_wl_bo_get_n_obs",
  "_wl_bo_get_best_score",
  "_wl_bo_get_n_contexts",
  "_wl_bo_free",
  "_malloc",
  "_free"
]'

EXPORTED_RUNTIME_METHODS='["ccall","cwrap","getValue","setValue","HEAPF64","HEAPU8","HEAP32"]'

emcc \
  "${PROJECT_DIR}/csrc/bo.c" \
  "${PROJECT_DIR}/csrc/wl_api.c" \
  -I "${PROJECT_DIR}/csrc" \
  -o "${OUTPUT_DIR}/bo.js" \
  -std=c11 \
  -s MODULARIZE=1 \
  -s SINGLE_FILE=1 \
  -s SINGLE_FILE_BINARY_ENCODE=0 \
  -s EXPORT_NAME=createBO \
  -s EXPORTED_FUNCTIONS="${EXPORTED_FUNCTIONS}" \
  -s EXPORTED_RUNTIME_METHODS="${EXPORTED_RUNTIME_METHODS}" \
  -s ALLOW_MEMORY_GROWTH=1 \
  -s INITIAL_MEMORY=16777216 \
  -s ENVIRONMENT='web,node' \
  -O2 \
  -lm

echo "=== Verifying exports ==="
bash "${SCRIPT_DIR}/verify-exports.sh"

echo "=== Writing BUILD_INFO ==="
cat > "${OUTPUT_DIR}/BUILD_INFO" <<EOF
upstream: none (C11 from scratch)
build_date: $(date -u +%Y-%m-%dT%H:%M:%SZ)
emscripten: $(emcc --version | head -1)
build_flags: -O2 -std=c11 SINGLE_FILE=1
wasm_embedded: true
EOF

echo "=== Build complete ==="
ls -lh "${OUTPUT_DIR}/bo.js"
cat "${OUTPUT_DIR}/BUILD_INFO"
