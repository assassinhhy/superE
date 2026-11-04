from __future__ import annotations

from pathlib import Path

import pytest

from supere.storage.db import Database


@pytest.fixture
def db(tmp_path: Path) -> Database:
    handle = Database(tmp_path / "superE.sqlite")
    yield handle
    handle.close()
