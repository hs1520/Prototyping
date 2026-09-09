"""SHA-256 of a text, used as a content identity for board revisions and replay keys."""
from __future__ import annotations

import hashlib


def sha256_text(text: str) -> str:
    """Hex SHA-256 of *text* encoded as UTF-8."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
