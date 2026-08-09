"""morphline — a BIDS-aware longitudinal neuroimaging derivatives pipeline.

Stage boundaries are the point of this package. Ingestion is the only layer
that knows what a FreeSurfer file looks like; everything downstream reads the
canonical schema in :mod:`morphline.schema` and nothing else (BUILD_PLAN §1.4).
"""

from __future__ import annotations

__version__ = "0.1.0"
