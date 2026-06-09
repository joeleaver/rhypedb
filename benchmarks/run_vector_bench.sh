#!/usr/bin/env bash
# Suite 2 (vector) driver: resolve the dataset dimension, generate a matching
# rhypedb schema, start the bench server, run scenario_09, tear the server down.
# Postgres+pgvector is expected up via `docker compose up -d` (pgvector image).
#
# Usage: run_vector_bench.sh [dataset] [subset] [queries] [k]
#   defaults: glove-25-angular 20000 500 10
set -uo pipefail
cd "$(dirname "$0")/.."   # repo root

DATASET="${1:-glove-25-angular}"
SUBSET="${2:-20000}"
QUERIES="${3:-500}"
K="${4:-10}"
DATA=/tmp/vecbench-rhypedb-data
HTTP_PORT=4500
TCP_PORT=4501

# 1) Resolve the dataset dimension. Synthetic datasets encode it in the name
# (synthetic-384-angular -> 384); ann-benchmarks datasets read it from the hdf5.
if [[ "$DATASET" == synthetic-* ]]; then
  DIM=$(echo "$DATASET" | cut -d- -f2)
else
  DIM=$(.venv-bench/bin/python -c "
from benchmarks.harness import vector_data
import h5py
with h5py.File(vector_data._download('$DATASET'), 'r') as f:
    print(f['train'].shape[1])
") || { echo "failed to resolve dataset dimension"; exit 1; }
fi
echo "dataset=$DATASET dim=$DIM subset=$SUBSET queries=$QUERIES k=$K"

# 2) Fresh data dir + dimension-matched schema.
pkill -f 'rhypedb-server.*vecbench-rhypedb-data' 2>/dev/null || true
timeout 10 bash -c 'until ! (exec 3<>/dev/tcp/127.0.0.1/'"$TCP_PORT"') 2>/dev/null; do :; done' || true
rm -rf "$DATA"; mkdir -p "$DATA"
printf 'type Vec {\n    embedding: Vector<%s>\n}\n' "$DIM" > "$DATA/vec.sdl"

# 3) Start the server.
target/release/rhypedb-server \
  --schema "$DATA/vec.sdl" --data-dir "$DATA" \
  --listen 127.0.0.1:"$HTTP_PORT" --tcp-listen 127.0.0.1:"$TCP_PORT" --no-sync \
  >/tmp/vecbench_server.log 2>&1 &
SRV=$!
if ! timeout 20 bash -c 'until (exec 3<>/dev/tcp/127.0.0.1/'"$TCP_PORT"') 2>/dev/null; do :; done'; then
  echo "SERVER FAILED TO START:"; cat /tmp/vecbench_server.log; kill "$SRV" 2>/dev/null || true; exit 1
fi
echo "server up pid=$SRV"

# 4) Run the scenario (connects to the running server) + tear down.
# pgvector ef_search matched to rhypedb's internal `.similar` ef (= k.max(50)).
# RHYPEDB_EF / RHYPEDB_RERANK (env, default 0/off) drive the new `.similar`
# recall knobs; a non-default value is tagged into the output filename so an
# A/B (baseline vs rerank) run doesn't clobber.
EF_SEARCH="${5:-50}"
RHYPEDB_EF="${RHYPEDB_EF:-0}"
RHYPEDB_RERANK="${RHYPEDB_RERANK:-0}"
TAG=""
[[ "$RHYPEDB_EF" != "0" ]] && TAG="${TAG}_ef${RHYPEDB_EF}"
[[ "$RHYPEDB_RERANK" != "0" ]] && TAG="${TAG}_rerank${RHYPEDB_RERANK}"
OUT="benchmarks/results/scenario_09_${DATASET}_${SUBSET}${TAG}.json"
.venv-bench/bin/python -m benchmarks.suite1.scenario_09_vector \
  --dataset "$DATASET" --subset "$SUBSET" --queries "$QUERIES" --k "$K" \
  --ef-search "$EF_SEARCH" --rhypedb-ef "$RHYPEDB_EF" --rhypedb-rerank "$RHYPEDB_RERANK" \
  --tcp-port "$TCP_PORT" --impl "${IMPL:-tcp,pg}" \
  --data-dir "$DATA" --out "$OUT"
RC=$?

kill "$SRV" 2>/dev/null || true
timeout 10 bash -c 'until ! (exec 3<>/dev/tcp/127.0.0.1/'"$TCP_PORT"') 2>/dev/null; do :; done' || true
exit $RC
