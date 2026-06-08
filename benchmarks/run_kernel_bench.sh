#!/usr/bin/env bash
# Before/after microbench for the TurboQuant distance kernel.
# Stashes ONLY the optimized quantize.rs to measure the baseline, then restores
# it and measures the optimized kernel. The example binary itself is unchanged
# between runs, so the comparison is apples-to-apples.
set -u
cd /home/joe/dev/rhypedb || exit 1

STASHED=0
if ! git diff --quiet -- crates/rhypedb-vector/src/quantize.rs; then
    git stash push -m kernelbase -- crates/rhypedb-vector/src/quantize.rs && STASHED=1
fi

echo "############ BASELINE (original kernel) ############"
cargo run --release --example kernel_bench -p rhypedb-vector 2>&1

echo
echo "############ restoring optimized kernel ############"
if [ "$STASHED" = 1 ]; then
    git stash pop
fi
git diff --stat -- crates/rhypedb-vector/src/quantize.rs | tail -2

echo
echo "############ AFTER (optimized kernel) ############"
cargo run --release --example kernel_bench -p rhypedb-vector 2>&1

echo
echo "############ DONE — git status ############"
git status --short -- crates/rhypedb-vector/src/quantize.rs
