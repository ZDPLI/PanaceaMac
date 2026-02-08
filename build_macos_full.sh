#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Full build includes FAISS + sentence-transformers (torch/transformers).
# Recommend Python 3.11 or 3.12 on macOS for best wheel availability.
python -m pip install -r requirements.txt

pyinstaller --noconfirm --clean panacea_desktop.spec

echo
echo "Build output:"
echo "  dist/Miriam.app"

