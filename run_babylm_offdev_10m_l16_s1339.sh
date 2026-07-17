#!/usr/bin/env bash
# Compatibility entry point for the seed-1339 offdev L16 pair.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$REPO_ROOT/run_babylm_offdev_10m_l16_s1338.sh" 1339
