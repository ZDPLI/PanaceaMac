# Miriam (Desktop)

A cross-platform desktop application that is functionally equivalent to the Panacea Telegram bot:
- Multi-dialog chat with persistent SQLite history
- Prompt packs: GP / DERM / HISTO / RADIO
- Modes: light / medium / heavy / consensus (3 candidates + arbiter)
- OpenAI-compatible providers: Novita / OpenRouter / Custom endpoint
- RAG over local documents (PDF/DOCX/TXT/RTF) with chunking and retrieval
- Token streaming (incremental output)
- Export chat to TXT/MD/JSON
- Optional image attachments for OpenAI-vision compatible models (data-url)
- Radiology study import (CT/MRI): DICOM folder/ZIP or NIfTI → generates preview slices and injects them into the model context

## Run locally

### Windows

1) Install Python 3.11+ (recommended)
2) In a terminal:

```bat
cd panacea_desktop
py -m venv .venv
.venv\Scripts\activate
py -m pip install -r requirements.txt
py -m panacea_desktop
```

The app stores its SQLite DB here:
- Windows: `%APPDATA%\MiriamDesktop\miriam.sqlite3`

### macOS

1) Install Python 3.11 or 3.12 (recommended for binary wheels)
2) In a terminal:

```bash
cd panacea_desktop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r ../requirements-lite.txt
python -m panacea_desktop
```

The app stores its SQLite DB here:
- `~/Library/Application Support/MiriamDesktop/miriam.sqlite3`

## Configure providers

Open **Settings** and set:
- Provider mode: auto / novita / openrouter / custom
- Base URL and API key
- Models for light/medium/heavy/arbiter

All providers are used via `POST {base_url}/chat/completions` (OpenAI-compatible).

## Radiology (CT/MRI)

Open **More → Radiology study (CT/MRI)** and select one of:
- A DICOM folder (will be zipped internally for portability)
- A DICOM ZIP
- A NIfTI file (`.nii` / `.nii.gz`)

The app will generate a small set of axial preview slices (PNG) and attach them to subsequent requests.


## Build .exe

### One-file exe
Run:

```bat
build_windows_onefile.bat
```

Output:
- `dist\Miriam.exe`

### One-dir build
Run:

```bat
build_windows_onedir.bat
```

Output:
- `dist\Miriam\Miriam.exe`

## Build macOS .app

On macOS, build with PyInstaller to produce `dist/Miriam.app`.

### Lite build (recommended)
This excludes FAISS / sentence-transformers (the app still works; RAG retrieval modes `lexical`/`bm25` are available).

```bash
./build_macos_lite.sh
```

### Full build
This includes FAISS + sentence-transformers (torch/transformers). Prefer Python 3.11/3.12.

```bash
./build_macos_full.sh
```

## Notes
- PySide6 (Qt) is used for the desktop UI.
- RAG retrieval modes:
  - `lexical` (simple token/IDF)
  - `bm25` (recommended default; requires `rank-bm25`)
  - `faiss` (local embeddings + FAISS cosine search; requires `sentence-transformers`, `numpy`, `faiss-cpu`)
- In `faiss` mode, the embedding model is configurable in Settings. By default it uses `sentence-transformers/all-MiniLM-L6-v2`.
  - The model may download on first run if not present; you can also point to a local folder path.

- Radiology: open **More → Radiology study (CT/MRI)** to import a DICOM folder/ZIP or NIfTI file.
  - The app stores the imported source and generates a small set of preview PNG slices under `%APPDATA%\MiriamDesktop\files\imaging_previews` (Windows) or `~/Library/Application Support/MiriamDesktop/files/imaging_previews` (macOS).
  - These previews are sent to the model as images, and metadata is injected as a `[RADIOLOGY CONTEXT]` block.


## Cross-dialog memory & prompts

- Settings → **Memory**: manage pinned/disabled memories, set max injected items, enable/disable auto-extraction.
- Settings → **Prompts**: override System/Arbiter prompts per prompt pack (gp/derm/histo).
