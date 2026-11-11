"""Real-network smoke test for the Windows machine. Not part of pytest.

    .venv\\Scripts\\python.exe scripts\\smoke_live.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from supere.smoke import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
