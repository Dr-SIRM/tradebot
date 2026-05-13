#!/usr/bin/env bash
# Download a 14-asset daily universe for TSM portfolio testing.
#
# Picked to span uncorrelated macro factors:
#   - Crypto (BTC, ETH)              - already have
#   - US large cap (SPY, QQQ)        - already have
#   - International equity (EFA, EEM, VEA)
#   - Bonds (TLT, IEF, HYG)
#   - Commodities (gold, silver, oil, DBA agriculture)
#   - REITs (VNQ)
#   - FX (EURUSD)                    - already have
#
# yfinance daily history goes back decades for liquid ETFs. Window 2010-2026
# gives every asset enough warmup for the 252-day momentum signal.
#
# Re-running this script is safe: it skips files that already exist.
# Use --force to re-download everything.
#
# Usage:
#   bash scripts/download_universe.sh           # download missing only
#   bash scripts/download_universe.sh --force   # re-download all
set -euo pipefail

FORCE=0
if [[ "${1:-}" == "--force" ]]; then FORCE=1; fi

cd "$(dirname "$0")/.."
mkdir -p data/raw

# Prefer project venv (has deps from requirements.txt); else system python3/python.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON=python
else
  echo "error: create .venv (see README) or install python3 on PATH" >&2
  exit 1
fi

START="2010-01-01"
END="2026-05-11"

# Format: <ticker>:<output_filename>:<asset_class>
# asset_class isn't used by the downloader (it just routes ticker→source) but
# is documented here so the spec yaml can pick it up.
ASSETS=(
  # Bonds
  "TLT:tlt_1d.csv:equity"
  "IEF:ief_1d.csv:equity"
  "HYG:hyg_1d.csv:equity"
  # International equity
  "EFA:efa_1d.csv:equity"
  "EEM:eem_1d.csv:equity"
  "VEA:vea_1d.csv:equity"
  # Commodities (ETFs — easier than futures contracts)
  "SLV:slv_1d.csv:commodity"
  "USO:uso_1d.csv:commodity"
  "DBA:dba_1d.csv:commodity"
  # REITs
  "VNQ:vnq_1d.csv:equity"
)

n_done=0
n_skipped=0
for entry in "${ASSETS[@]}"; do
  IFS=':' read -r ticker outfile class <<< "$entry"
  outpath="data/raw/${outfile}"
  if [[ -f "$outpath" && $FORCE -eq 0 ]]; then
    echo "[skip]   $ticker → $outfile (already exists)"
    n_skipped=$((n_skipped + 1))
    continue
  fi
  echo "[fetch]  $ticker → $outfile"
  "$PYTHON" -m data.download equity "$ticker" 1d "$START" "$END" -o "$outpath" || {
    echo "[FAIL]   $ticker — moving on"
    continue
  }
  n_done=$((n_done + 1))
done

echo
echo "Done. Downloaded: $n_done, skipped: $n_skipped"
echo "Now run:  $PYTHON run_tsm_backtest.py --spec config/tsm_universe.yaml --out logs/tsm_universe"
