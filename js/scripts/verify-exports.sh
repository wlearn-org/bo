#!/bin/bash
set -euo pipefail

# Verify that the built WASM module exports all expected symbols.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WASM_FILE="${PROJECT_DIR}/wasm/bo.js"

if [ ! -f "$WASM_FILE" ]; then
  echo "ERROR: ${WASM_FILE} not found. Run build-wasm.sh first."
  exit 1
fi

EXPECTED_EXPORTS=(
  wl_bo_get_last_error
  wl_bo_space_create
  wl_bo_space_add_continuous
  wl_bo_space_add_integer
  wl_bo_space_add_categorical
  wl_bo_space_add_condition
  wl_bo_space_finalize
  wl_bo_space_free
  wl_bo_create
  wl_bo_observe
  wl_bo_suggest
  wl_bo_suggest_liar
  wl_bo_get_n_obs
  wl_bo_get_best_score
  wl_bo_get_n_contexts
  wl_bo_free
)

missing=0
for fn in "${EXPECTED_EXPORTS[@]}"; do
  if ! grep -q "_${fn}" "$WASM_FILE"; then
    echo "MISSING: _${fn}"
    missing=$((missing + 1))
  fi
done

if [ $missing -gt 0 ]; then
  echo "ERROR: ${missing} exports missing from ${WASM_FILE}"
  exit 1
fi

echo "All ${#EXPECTED_EXPORTS[@]} exports verified."
