"""
common.py
=========
Shared constants + helpers used by all three legal agents.
"""
from __future__ import annotations

import json
import re
from typing import Any

NOT_FOUND = "Not found in the provided documents."
MISSING = "[MISSING INFORMATION]"

TRUSTED_DOMAINS = [
    "indiacode.nic.in",
    "main.sci.gov.in",
    "indiankanoon.org",
    "egazette.nic.in",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def safe(value: Any) -> Any:
    """Return MISSING for None / empty values; otherwise the value itself."""
    if value is None:
        return MISSING
    if isinstance(value, str) and not value.strip():
        return MISSING
    if isinstance(value, (list, tuple, dict)) and len(value) == 0:
        return MISSING
    return value


def dedupe(items: list[Any]) -> list[Any]:
    seen: set[str] = set()
    out = []
    for it in items:
        key = json.dumps(it, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out
