#!/bin/bash
# Generate fresh structured corpora and replay every row through both the
# atomic solver and public decomposition pipeline. Run the regular suite first.
set -uo pipefail
cd "$(dirname "$0")"
ROOT=$(cd ../.. && pwd)
PYTHON="${GULPS_PYTHON:-$ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "missing project Python: $PYTHON; run 'make bootstrap'" >&2
  exit 2
fi

if [[ $# -eq 0 ]]; then
  echo "usage: $0 SEED [SEED ...]" >&2
  exit 2
fi

tmp=$(mktemp -d /tmp/gulps-realization-stress.XXXXXX)
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
  if ! "$PYTHON" "$ROOT/dev/research/scripts/generate_realization_edge_corpus.py" \
    --seed "$seed" --output "$corpus"; then
    status=1
    continue
  fi
  if ! ../target/release/can_sandwich bench-npy "$corpus" 1; then
    status=1
  fi
  if ! "$PYTHON" \
    "$ROOT/dev/research/scripts/validate_realization_pipeline_corpus.py" \
    "${corpus%.npy}.pipeline.npz" --max-case-seconds 0.5; then
    status=1
  fi
done
exit "$status"
