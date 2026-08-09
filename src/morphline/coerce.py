"""Typed scalar extraction from pandas rows.

Indexing a pandas row returns a very broad union — anything from ``str`` to
``Timestamp`` to ``complexfloating`` — so ``float(row["age"])`` does not type
check under ``mypy --strict`` even when it is obviously correct at runtime.
These helpers put the narrowing in one reviewed place instead of scattering
``cast`` calls (or, worse, ``# type: ignore``) through the ingestion code.

They also fail loudly on genuinely unconvertible values, which is the behaviour
worth having at a dataset boundary: a column that unexpectedly holds text
should raise here rather than silently become ``NaN`` three stages later.
"""

from __future__ import annotations

import math
from typing import Any


def as_float(value: Any) -> float:
    """Coerce a pandas scalar to ``float``.

    Args:
        value: A scalar pulled from a DataFrame row.

    Returns:
        The value as a float.

    Raises:
        TypeError: If the value cannot be interpreted as a number.
    """
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"expected a numeric value, got {value!r} ({type(value).__name__})"
        ) from exc


def as_optional_float(value: Any) -> float | None:
    """Coerce a pandas scalar to ``float``, mapping null and NaN to ``None``.

    Used where absence is meaningful and must not collapse to zero — surface
    hole counts being the case that matters (§2.2).

    Args:
        value: A scalar pulled from a DataFrame row.

    Returns:
        The value as a float, or ``None`` if it was null or NaN.
    """
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


def as_str(value: Any) -> str:
    """Coerce a pandas scalar to ``str``.

    Args:
        value: A scalar pulled from a DataFrame row.

    Returns:
        The value as a string.
    """
    return str(value)
