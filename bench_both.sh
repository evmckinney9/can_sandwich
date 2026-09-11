#!/bin/bash
# Standard change validation: stratified, linspace, and Haar at stride 1, plus LOC.
# Run this after every realization edit.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
PYTHON="${GULPS_PYTHON:-$ROOT/.venv/bin/python}"
SEG="${GULPS_REALIZATION_CORPUS_DIR:-$ROOT/.local/corpora/realization-v5}"
if [[ ! -x "$PYTHON" ]]; then
  echo "missing project Python: $PYTHON; run 'make bootstrap'" >&2
  exit 2
fi
for required in feasible_linspace.npy feasible_haar.npy; do
  if [[ ! -f "$SEG/$required" ]]; then
    echo "missing locked corpus $SEG/$required" >&2
    echo "run 'make corpora CORPUS_SOURCE=/path/to/segments'" >&2
    exit 2
  fi
done
if ! "$PYTHON" "$ROOT/dev/research/scripts/generate_realization_edge_corpus.py" \
  --output "$SEG/feasible_stratified.npy" --check-existing; then
  echo "locked stratified corpus is missing, stale, or corrupt: $SEG" >&2
  echo "materialize the registered fixture; do not overwrite it by regeneration" >&2
  exit 2
fi
cargo build --quiet --release --features diagnostics --bin can_sandwich || exit 1
status=0
echo "== non-iterative realization guard =="
if [[ -e src/constructive.rs ]] || \
   grep -R -nE "polish_recovery|perturb_recovery|constructive::|levenberg|multistart" \
     src ../core/src/realization; then
  echo "iterative realization code is not permitted" >&2
  status=1
fi
run_atomic_corpus() {
  local label=$1
  local path=$2
  echo "== $label =="
  if ! ../target/release/can_sandwich bench-npy "$path" 1 2>/dev/null \
    | grep -E "OVERALL|by rung|unusable gate-lifted|latency"; then
    status=1
  fi
}
run_atomic_corpus "stratified exact/near Weyl strata" "$SEG/feasible_stratified.npy"
run_atomic_corpus "linspace" "$SEG/feasible_linspace.npy"
run_atomic_corpus "haar" "$SEG/feasible_haar.npy"
echo "== public realization pipeline =="
if ! "$PYTHON" "$ROOT/dev/research/scripts/validate_realization_pipeline_corpus.py" \
  "$SEG/feasible_stratified.pipeline.npz" --max-case-seconds 0.5; then
  status=1
fi
echo "== stable tail rows (1001 repeats) =="
../target/release/can_sandwich bench-row $SEG/feasible_linspace.npy 712010 1001 2>/dev/null
../target/release/can_sandwich bench-row $SEG/feasible_haar.npy 22874 1001 2>/dev/null
echo "== production library LOC =="
ls src/*.rs | grep -v main.rs | xargs wc -l | tail -1
echo "== benchmark/diagnostic LOC =="
wc -l src/main.rs
exit "$status"
