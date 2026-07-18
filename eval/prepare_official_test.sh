#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DATA_DIR" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$1"
PY="${PY:-python}"

cd "$REPO_ROOT"
"$PY" data/babylm/fetch_offtest.py --data-dir "$DATA_DIR"
"$PY" data/babylm/clean.py \
  --input-split test \
  --raw-dir "$DATA_DIR/raw/test" \
  --out-dir "$DATA_DIR/clean/test"
"$PY" data/babylm/prepare_test.py --data-dir "$DATA_DIR"
