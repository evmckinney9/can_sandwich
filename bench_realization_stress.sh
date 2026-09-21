#!/bin/bash
# Generate fresh structured corpora and replay every row through both the
# atomic solver and public decomposition pipeline. Run the regular suite first.
set -uo pipefail
cd "$(dirname "$0")" || exit 2
PYTHON="${GULPS_PYTHON:-python3}"
TARGET=$(cargo metadata --no-deps --format-version 1 | "$PYTHON" -c 'import json, sys; print(json.load(sys.stdin)["target_directory"])') || exit 2

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "missing Python: $PYTHON; set GULPS_PYTHON to a Python with gulps installed" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 SEED [SEED ...]" >&2
  exit 2
fi

for seed in "$@"; do
  if [[ ! "$seed" =~ ^[0-9]+$ ]]; then
    echo "seed must be a nonnegative decimal integer: $seed" >&2
    exit 2
  fi
done

tmp=$(mktemp -d /tmp/gulps-realization-stress.XXXXXX) || exit 2
cleanup() {
  if [[ "$tmp" == /tmp/gulps-realization-stress.* && -d "$tmp" ]]; then
    rm -rf -- "$tmp"
  fi
}
trap cleanup EXIT

cargo build --quiet --release --features diagnostics --bin can_sandwich || exit 1
status=0
for seed in "$@"; do
  corpus="$tmp/feasible_stratified_seed_${seed}.npy"
  echo "== structured realization stress seed $seed =="
  if ! "$PYTHON" scripts/generate_realization_edge_corpus.py \
    --seed "$seed" --output "$corpus"; then
    status=1
    continue
  fi
  if ! "$TARGET/release/can_sandwich" bench-npy "$corpus" 1; then
    status=1
  fi
  if ! "$PYTHON" \
    scripts/validate_realization_pipeline_corpus.py \
    "${corpus%.npy}.pipeline.npz" --max-case-seconds 0.5; then
    status=1
  fi
done
exit "$status"
