"""Entry point for PyInstaller.

IMPORTANT:
Do NOT point PyInstaller directly at `panacea_desktop/main.py`.
When `main.py` is executed as a script, it has no parent package and
relative imports like `from .core...` will raise:
    ImportError: attempted relative import with no known parent package

This file is executed as a top-level script and imports the package
module, ensuring a proper package context both in source and when frozen.
"""

from panacea_desktop.qt_app import main


if __name__ == "__main__":
    main()
