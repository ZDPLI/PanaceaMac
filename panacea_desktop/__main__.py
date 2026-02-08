"""Package entrypoint for local runs.

Allows:
  python -m panacea_desktop
"""

from .qt_app import main


if __name__ == "__main__":
    main()

