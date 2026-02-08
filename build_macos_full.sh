#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"

PY=""
for c in python3.12 python3.11 python3; do
  if command -v "$c" >/dev/null 2>&1; then
    PY="$c"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "ERROR: python3 not found. Install Python 3.11 from python.org first." >&2
  exit 1
fi

"$PY" - <<'PY'
import sys
if sys.version_info < (3, 11):
    raise SystemExit(
        "ERROR: Python 3.11+ is required for the macOS build.\n"
        "Your current python is: " + sys.version + "\n\n"
        "Fix: install Python 3.11 from https://www.python.org/downloads/macos/ and re-run.\n"
        "Tip: use python3.11 explicitly if python3 points to the system Python."
    )
print("Using", sys.version)
PY

"$PY" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# Full build includes FAISS + sentence-transformers (torch/transformers).
# Recommend Python 3.11 or 3.12 on macOS for best wheel availability.
python -m pip install -r requirements.txt

pyinstaller --noconfirm --clean panacea_desktop.spec

echo
echo "Build output:"
echo "  dist/Miriam.app"
