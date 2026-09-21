#!/bin/bash
# Standard change validation: stratified, linspace, and Haar at stride 1, plus LOC.
# Run this after every realization edit.
set -uo pipefail
cd "$(dirname "$0")" || exit 2
PYTHON="${GULPS_PYTHON:-python3}"
TARGET=$(cargo metadata --no-deps --format-version 1 | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])') || exit 2
SEG="${GULPS_REALIZATION_CORPUS_DIR:-$PWD/corpus}"
if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "missing Python: $PYTHON; set GULPS_PYTHON to a Python with gulps installed" >&2
  exit 2
fi
for required in feasible_linspace.npy feasible_haar.npy; do
  if [[ ! -f "$SEG/$required" ]]; then
    echo "missing locked corpus $SEG/$required" >&2
    echo "the Haar and linspace corpora are not tracked; copy them into corpus/" >&2
    exit 2
  fi
done
if ! "$PYTHON" scripts/generate_realization_edge_corpus.py \
  --output "$SEG/feasible_stratified.npy" --check-existing; then
  echo "locked stratified corpus is missing, stale, or corrupt: $SEG" >&2
  echo "materialize the registered fixture; do not overwrite it by regeneration" >&2
  exit 2
fi
cargo build --quiet --release --features diagnostics --bin can_sandwich || exit 1
status=0
echo "== non-iterative realization guard =="
if [[ -e src/constructive.rs ]]; then
  echo "iterative realization code is not permitted" >&2
  status=1
else
  grep -R -nE "polish_recovery|perturb_recovery|constructive::|levenberg|multistart" \
    src
  guard_status=$?
  if [[ $guard_status -eq 0 ]]; then
    echo "iterative realization code is not permitted" >&2
    status=1
  elif [[ $guard_status -ne 1 ]]; then
    echo "could not complete the non-iterative realization guard" >&2
    status=1
  fi
fi
run_atomic_corpus() {
  local label=$1
  local path=$2
  echo "== $label =="
  if ! "$TARGET/release/can_sandwich" bench-npy "$path" 1 \
    | grep -E "OVERALL|by rung|unusable gate-lifted|latency"; then
    status=1
  fi
}
run_atomic_corpus "stratified exact/near Weyl strata" "$SEG/feasible_stratified.npy"
run_atomic_corpus "linspace" "$SEG/feasible_linspace.npy"
run_atomic_corpus "haar" "$SEG/feasible_haar.npy"
echo "== public realization pipeline =="
if ! "$PYTHON" scripts/validate_realization_pipeline_corpus.py \
  "$SEG/feasible_stratified.pipeline.npz" --max-case-seconds 0.5; then
  status=1
fi
echo "== stable tail rows (1001 repeats) =="
if ! "$TARGET/release/can_sandwich" bench-row "$SEG/feasible_linspace.npy" 712010 1001; then
  status=1
fi
if ! "$TARGET/release/can_sandwich" bench-row "$SEG/feasible_haar.npy" 22874 1001; then
  status=1
fi
echo "== production library LOC =="
library_sources=()
for path in src/*.rs; do
  library_sources+=("$path")
done
if ! wc -l "${library_sources[@]}" | tail -n 1; then
  status=1
fi
echo "== benchmark/diagnostic LOC =="
if ! wc -l benchmark/diagnostics.rs; then
  status=1
fi
exit "$status"
