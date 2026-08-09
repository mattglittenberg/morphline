"""Canonical schema contract tests (BUILD_PLAN §1.5)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from morphline.schema import (
    CANONICAL_COLUMNS,
    OBSERVATION_KEY,
    conform,
    empty_canonical,
    euler_number,
    read_canonical,
    validate,
    write_canonical,
)


def minimal_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": "synthetic-v1",
        "dataset_version": "0.1.0",
        "subject_id": "sub-0001",
        "session_id": "ses-01",
        "region": "lh-hippocampus",
        "measure_type": "volume",
        "value": 4000.0,
        "source_file": "/tmp/aseg.stats",
        "source_file_checksum": "abc123",
        "parser_version": "1.0.0",
        "ingested_at": "2026-08-08T00:00:00Z",
    }
    row.update(overrides)
    return row


def test_empty_canonical_has_every_column() -> None:
    df = empty_canonical()
    for column in CANONICAL_COLUMNS:
        assert column in df.columns


def test_conform_fills_missing_optional_columns() -> None:
    df = conform(pd.DataFrame([minimal_row()]))
    assert "site" in df.columns
    assert df["site"].isna().all()
    assert "qc_status" in df.columns


def test_conform_rejects_missing_required_column() -> None:
    row = minimal_row()
    del row["subject_id"]
    with pytest.raises(ValueError, match="missing required columns"):
        conform(pd.DataFrame([row]))


def test_validate_rejects_duplicate_observation_keys() -> None:
    """Duplicated subject x session x region x measure rows are a bug (§5.2)."""
    df = conform(pd.DataFrame([minimal_row(), minimal_row()]))
    with pytest.raises(ValueError, match="duplicated observation keys"):
        validate(df)


def test_validate_accepts_same_region_in_different_sessions() -> None:
    df = conform(pd.DataFrame([minimal_row(), minimal_row(session_id="ses-02")]))
    validate(df)


def test_observation_key_is_the_documented_grain() -> None:
    assert OBSERVATION_KEY == (
        "dataset",
        "subject_id",
        "session_id",
        "region",
        "measure_type",
    )


def test_parquet_roundtrip_preserves_values(tmp_path: Path) -> None:
    df = conform(pd.DataFrame([minimal_row(site="site-a", value=4123.5)]))
    path = write_canonical(df, tmp_path / "obs.parquet")
    back = read_canonical(path)
    assert back.loc[0, "value"] == pytest.approx(4123.5)
    assert back.loc[0, "site"] == "site-a"
    assert back.loc[0, "subject_id"] == "sub-0001"


def test_write_rejects_invalid_frame(tmp_path: Path) -> None:
    """Persistence is a validation point, not just serialization."""
    df = conform(pd.DataFrame([minimal_row(), minimal_row()]))
    with pytest.raises(ValueError, match="duplicated"):
        write_canonical(df, tmp_path / "obs.parquet")


def test_provenance_columns_survive_roundtrip(tmp_path: Path) -> None:
    """Traceability requires the source file to survive persistence (§1.5)."""
    df = conform(pd.DataFrame([minimal_row(source_file="/data/sub-1/aseg.stats")]))
    back = read_canonical(write_canonical(df, tmp_path / "obs.parquet"))
    assert back.loc[0, "source_file"] == "/data/sub-1/aseg.stats"
    assert back.loc[0, "source_file_checksum"] == "abc123"


class TestEulerNumber:
    """Euler derivation, per §2.2."""

    @pytest.mark.parametrize(
        ("holes", "expected"),
        [(0, 2.0), (1, 0.0), (42, -82.0), (17, -32.0), (100, -198.0)],
    )
    def test_derivation(self, holes: int, expected: float) -> None:
        assert euler_number(holes) == expected

    def test_none_propagates(self) -> None:
        assert euler_number(None) is None

    def test_nan_propagates(self) -> None:
        assert euler_number(float("nan")) is None

    def test_more_holes_means_more_negative(self) -> None:
        """More negative means more defects — the sign convention matters."""
        assert euler_number(50) < euler_number(10)  # type: ignore[operator]
