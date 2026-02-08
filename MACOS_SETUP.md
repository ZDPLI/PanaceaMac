# macOS Setup / Build

This folder is meant to be copied to a macOS machine and built there.

## Prereqs (macOS)

1. Install Python 3.11 or 3.12.
2. Open Terminal and `cd` into this folder.

## Build a .app (recommended: lite)

The lite build excludes FAISS / sentence-transformers (no torch/transformers). RAG `lexical`/`bm25` still works.
It also excludes optional voice dependencies by default, so it can build on a clean macOS Big Sur install without Homebrew.

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

## Optional: Voice dependencies (mic / TTS)

Voice input (mic) uses `sounddevice` + `soundfile` and typically requires native audio libraries on macOS.
On a clean system without Homebrew, keep Voice disabled and the app will work normally.

## If macOS blocks the app (Gatekeeper)

If you downloaded the zip from the internet, macOS may quarantine the app bundle. You can remove quarantine like this:

```bash
xattr -dr com.apple.quarantine dist/Miriam.app
```
