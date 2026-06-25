#!/usr/bin/env bash
#
# Ship-guard for the crash-fuzz fault injector (Overboard cmqshgpnx, Inc 7).
#
# `crash-fuzz` is a TEST-ONLY Cargo feature (in rhypedb-storage, re-exported by
# rhypedb-engine) that arms an in-process fault injector at WAL/flush/compaction/
# vectorize boundaries. It must NEVER be enabled in a production build of the
# `rhypedb-server` binary — the default build is meant to be byte-identical with
# the feature absent.
#
# This guard resolves the server's SHIPPED feature graph (default AND
# --no-default-features, dev-dependency edges excluded — they never link into the
# binary) and fails if `crash-fuzz` appears anywhere in it. A feature-tree check is
# chosen over an `nm`/symbol diff on purpose: the `Site` enum and the
# `crash_inject::hit` call sites are ALWAYS compiled (they no-op when the feature is
# off), so those symbols exist even in a clean prod binary — a symbol scan would
# false-positive, whereas the resolved feature set is the exact, authoritative
# invariant.
#
# Run locally: ./scripts/assert-no-crash-fuzz.sh
set -euo pipefail

cd "$(dirname "$0")/.."

fail=0

# Match `crash-fuzz` only as a whole feature token in `cargo tree`'s `{p} {f}`
# output (features are space/comma delimited), so a confusable name like
# `crash-fuzz-extended` or a path component never false-positives.
PATTERN='(^|[ ,])crash-fuzz([ ,]|$)'

# Resolve the server binary's SHIPPED feature graph for a given feature flag set
# and fail if `crash-fuzz` appears on any node. `-e features,no-dev` adds feature
# edges while dropping dev-dependency edges (test-only crates that never link into
# the binary, e.g. a future crash-fuzz E2E that pulls the engine with the feature).
# The `'{p} {f}'` format prints each package with its enabled feature list.
check_pkg() {
  local label="$1"
  shift
  echo "==> $label"
  local out
  out="$(cargo tree -p rhypedb-server -e features,no-dev -f '{p} {f}' "$@" 2>/dev/null)"
  if echo "$out" | grep -qiE "$PATTERN"; then
    echo "  FAIL: crash-fuzz reaches the prod server feature tree:"
    echo "$out" | grep -iE "$PATTERN" | sed 's/^/    /'
    fail=1
  else
    echo "  ok: no crash-fuzz in this feature tree"
  fi
}

# Catch ANY crate (not just the server's path) putting crash-fuzz in its own
# `default` — that would activate it on a normal (shipped) edge. Dev edges are
# excluded for the same reason as above.
check_ws() {
  echo "==> whole workspace (default features, shipped edges)"
  local out
  out="$(cargo tree --workspace -e features,no-dev -f '{p} {f}' 2>/dev/null)"
  if echo "$out" | grep -qiE "$PATTERN"; then
    echo "  FAIL: a crate enables crash-fuzz by default:"
    echo "$out" | grep -iE "$PATTERN" | sed 's/^/    /'
    fail=1
  else
    echo "  ok: no crate defaults crash-fuzz on"
  fi
}

check_pkg "rhypedb-server (default features)"
check_pkg "rhypedb-server (--no-default-features)" --no-default-features
check_ws

if [[ "$fail" -ne 0 ]]; then
  echo
  echo "crash-fuzz must stay test-only. Remove it from any default/production"
  echo "feature path of rhypedb-server (it belongs behind 'cargo test --features"
  echo "crash-fuzz' only)."
  exit 1
fi

echo
echo "PASS: the crash-fuzz fault injector does not reach the production server binary."
