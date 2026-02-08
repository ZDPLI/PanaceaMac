#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-11.0}"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lite.txt

pyinstaller --noconfirm --clean panacea_desktop.spec

echo
echo "Build output:"
echo "  dist/Miriam.app"
