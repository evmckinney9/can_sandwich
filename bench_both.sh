#!/bin/bash
# Standard change validation: stratified, linspace, and Haar at stride 1, plus LOC.
# Run this after every realization edit.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
SEG="$ROOT/.claude/scripts/diagnostics/segments"
if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/generate_realization_edge_corpus.py" \
  --output "$SEG/feasible_stratified.npy" --check-existing >/dev/null 2>&1; then
  "$ROOT/.venv/bin/python" "$ROOT/scripts/generate_realization_edge_corpus.py" \
    --output "$SEG/feasible_stratified.npy"
fi
cargo build --quiet --release --features diagnostics --bin can_sandwich || exit 1
status=0
echo "== non-iterative realization guard =="
if [[ -e src/constructive.rs ]] || \
   rg -n "polish_recovery|perturb_recovery|constructive::|levenberg|multistart" \
     src ../core/src/realization; then
  echo "iterative realization code is not permitted" >&2
  status=1
fi
run_atomic_corpus() {
  local label=$1
  local path=$2
  echo "== $label =="
  if ! ./target/release/can_sandwich bench-npy "$path" 1 2>/dev/null \
    | grep -E "OVERALL|by rung|unusable gate-lifted|latency"; then
    status=1
  fi
}
run_atomic_corpus "stratified exact/near Weyl strata" "$SEG/feasible_stratified.npy"
run_atomic_corpus "linspace" "$SEG/feasible_linspace.npy"
run_atomic_corpus "haar" "$SEG/feasible_haar.npy"
echo "== public realization pipeline =="
if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/repro_realization_hole.py"; then
  status=1
fi
if ! "$ROOT/.venv/bin/python" "$ROOT/scripts/validate_realization_pipeline_corpus.py" \
  "$SEG/feasible_stratified.pipeline.npz" --max-case-seconds 0.5; then
  status=1
fi
echo "== stable tail rows (1001 repeats) =="
./target/release/can_sandwich bench-row $SEG/feasible_linspace.npy 712010 1001 2>/dev/null
./target/release/can_sandwich bench-row $SEG/feasible_haar.npy 22874 1001 2>/dev/null
echo "== production library LOC =="
wc -l \
  src/lib.rs src/can_sandwich.rs src/problem.rs src/arb_roots.rs \
  src/axis_quartic.rs src/one_plus_three.rs src/klein.rs \
  src/pair22.rs src/secular.rs src/three_givens.rs src/sandwich.rs \
  src/chord.rs src/cpoly.rs src/resonance.rs | tail -1
echo "== benchmark/diagnostic LOC =="
wc -l src/main.rs src/data.rs src/spin_selector.rs src/kernel_frame.rs | tail -1
exit "$status"
