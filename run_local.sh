#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ ! -x ".venv/bin/python" ]; then
  python3 -m venv .venv
fi

.venv/bin/python -m pip install -r requirements.txt

export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export MPLCONFIGDIR="${TMPDIR:-/tmp}/qsar_qspr_mplconfig"
mkdir -p "$MPLCONFIGDIR"

exec .venv/bin/streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless true

