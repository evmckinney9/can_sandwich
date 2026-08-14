#!/bin/bash
# Standard change-validation: both corpora stride-1 + LOC. Run after every edit.
set -e
cd "$(dirname "$0")"
SEG=/home/evm9/gulps/.claude/scripts/diagnostics/segments
cargo build --release --features diagnostics --bin can_sandwich 2>&1 | grep -E "^error" && exit 1
echo "== linspace =="
./target/release/can_sandwich bench-npy $SEG/feasible_linspace.npy 1 2>/dev/null | grep -E "OVERALL|by rung|latency"
echo "== haar =="
./target/release/can_sandwich bench-npy $SEG/feasible_haar.npy 1 2>/dev/null | grep -E "OVERALL|by rung|latency"
echo "== stable tail rows (1001 repeats) =="
./target/release/can_sandwich bench-row $SEG/feasible_linspace.npy 712010 1001 2>/dev/null
./target/release/can_sandwich bench-row $SEG/feasible_haar.npy 22874 1001 2>/dev/null
echo "== production library LOC =="
wc -l \
  src/lib.rs src/can_sandwich.rs src/problem.rs src/arb_roots.rs \
  src/axis_quartic.rs src/one_plus_three.rs src/klein.rs \
  src/pair22.rs src/secular.rs \
  src/three_givens.rs src/sandwich.rs | tail -1
echo "== benchmark/diagnostic LOC =="
wc -l src/main.rs src/data.rs src/probe_dump.rs src/spin_selector.rs | tail -1
