# macOS Setup / Build

This folder is meant to be copied to a macOS machine and built there.

## Prereqs (macOS)

1. Install Python 3.11 or 3.12.
2. Open Terminal and `cd` into this folder.

## Build a .app (recommended: lite)

The lite build excludes FAISS / sentence-transformers (no torch/transformers). RAG `lexical`/`bm25` still works.

```bash
chmod +x ./build_macos_lite.sh
./build_macos_lite.sh
```

Output:
- `dist/Miriam.app`

## Full build (optional)

```bash
chmod +x ./build_macos_full.sh
./build_macos_full.sh
```

## If macOS blocks the app (Gatekeeper)

If you downloaded the zip from the internet, macOS may quarantine the app bundle. You can remove quarantine like this:

```bash
xattr -dr com.apple.quarantine dist/Miriam.app
```

