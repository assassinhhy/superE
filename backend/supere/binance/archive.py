"""Read data.binance.vision zip files. Checksums are ``sha256  filename``."""

from __future__ import annotations

import hashlib
import io
import zipfile

from supere.errors import ChecksumError


def parse_checksum(text: str) -> str:
    token = text.strip().split()[0]
    if len(token) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in token):
        raise ChecksumError(f"无法识别的 CHECKSUM: {text[:80]!r}")
    return token.lower()


def verify_bytes(blob: bytes, checksum_text: str) -> None:
    expected = parse_checksum(checksum_text)
    digest = hashlib.sha256(blob).hexdigest()
    if digest != expected:
        raise ChecksumError("SHA-256 不一致")


def zip_text(blob: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if not names:
            raise ChecksumError("zip 里没有文件")
        return archive.read(names[0]).decode("utf-8")
