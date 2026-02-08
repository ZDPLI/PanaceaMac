# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

import sys

from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.datastruct import Tree

def _safe_collect_submodules(pkg: str) -> list[str]:
    try:
        return collect_submodules(pkg)
    except Exception:
        return []


def _safe_collect_data_files(pkg: str) -> list[tuple[str, str]]:
    try:
        return collect_data_files(pkg)
    except Exception:
        return []


hiddenimports = []
hiddenimports += _safe_collect_submodules('PySide6')
hiddenimports += _safe_collect_submodules('docx')
a_datas = []
a_datas += _safe_collect_data_files('PySide6')
a_datas += _safe_collect_data_files('sounddevice')
a_datas += _safe_collect_data_files('soundfile')
"""Bundle local sources as datas to prevent missing-module crashes when PyInstaller misses a submodule."""
a_datas += [Tree('panacea_desktop', prefix='panacea_desktop')]
hiddenimports += _safe_collect_submodules('PyPDF2')
hiddenimports += _safe_collect_submodules('rank_bm25')
hiddenimports += _safe_collect_submodules('panacea_desktop')
hiddenimports += _safe_collect_submodules('sentence_transformers')
hiddenimports += _safe_collect_submodules('transformers')
hiddenimports += _safe_collect_submodules('torch')
hiddenimports += _safe_collect_submodules('faiss')
hiddenimports += _safe_collect_submodules('sounddevice')
hiddenimports += _safe_collect_submodules('soundfile')
hiddenimports += _safe_collect_submodules('pyttsx3')
hiddenimports += _safe_collect_submodules('pydicom')
hiddenimports += _safe_collect_submodules('nibabel')
hiddenimports += _safe_collect_submodules('PIL')


a = Analysis(
    ['run_app.py'],
    pathex=['.'],
    binaries=[],
    datas=a_datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Miriam',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=(False if sys.platform == "darwin" else True),
    console=False,
)

if sys.platform == "darwin":
    # Build a proper macOS .app bundle for double-click launch.
    app = BUNDLE(
        exe,
        name="Miriam.app",
        bundle_identifier="com.panacea.miriam",
        info_plist={
            "CFBundleDisplayName": "Miriam",
            "NSMicrophoneUsageDescription": "Optional: used for voice features.",
        },
    )
