#!/usr/bin/env bash
# Before/after microbench for the TurboQuant vector crate.
# Stashes ONLY the optimized source files (quantize.rs + index.rs) to measure the
# baseline, then restores them and measures the optimized version. The example
# binary itself is unchanged between runs, so the comparison is apples-to-apples.
set -u
cd /home/joe/dev/rhypedb || exit 1

SRC="crates/rhypedb-vector/src/quantize.rs crates/rhypedb-vector/src/index.rs"
STASHED=0
if ! git diff --quiet -- $SRC; then
    git stash push -m kernelbase -- $SRC && STASHED=1
fi

echo "############ BASELINE (original kernel) ############"
cargo run --release --example kernel_bench -p rhypedb-vector 2>&1

echo
echo "############ restoring optimized kernel ############"
if [ "$STASHED" = 1 ]; then
    git stash pop
fi
git diff --stat -- $SRC | tail -3

echo
echo "############ AFTER (optimized kernel) ############"
cargo run --release --example kernel_bench -p rhypedb-vector 2>&1

echo
echo "############ DONE — git status ############"
git status --short -- $SRC
