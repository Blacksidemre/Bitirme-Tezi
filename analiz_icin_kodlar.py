"""Backward-compatible launcher for the modern analysis package.

Usage:
    python analiz_icin_kodlar.py
    python analiz_icin_kodlar.py --data veriseti.xlsx --output-dir outputs/latest
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atakum_housing.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
