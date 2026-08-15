"""The report's figures (BUILD_PLAN §2.8).

Two properties matter more than any individual chart. First, **a figure with no
data returns None rather than an empty axis frame** — an empty frame reads as
"measured and found nothing", which is a different claim from "not applicable
here", and this pipeline spends real effort elsewhere keeping those apart.
Second, **the Plotly bundle is inlined exactly once**: §2.8 requires a
self-contained file, and inlining per figure would multiply several megabytes by
the chart count.
"""

from __future__ import annotations

from typing import Any

import pytest

from morphline.report import charts

EMPTY: dict[str, Any] = {}


@pytest.fixture
def accounting() -> dict[str, Any]:
    """A minimal accounting report with every charted section populated."""
    return {
        "funnel": [
            {
                "boundary": "raw files",
                "unit": "files",
                "count": 100,
                "lost": 0,
                "causes": "",
                "unexplained": 0,
            },
            {
                "boundary": "parsed files",
                "unit": "files",
                "count": 96,
                "lost": 4,
                "causes": "EMPTY_FILE=4",
                "unexplained": 0,
            },
        ],
        "qc_summary": {
            "by_status": {"PASS": 90, "WARNING": 6},
            "by_site": {"site-a": {"PASS": 45, "WARNING": 3}, "site-b": {"PASS": 45, "WARNING": 3}},
            "by_flag": {"euler_low": 4, "region_outlier": 2},
        },
        "missingness_by_site": {
            "site-a": {"missing_derivative": 3},
            "site-b": {"missing_acquisition": 1},
        },
        "batch_sizes": {"observations_per_site": {"site-a": 45, "site-b": 12}},
        "completers": {
            "applicable": True,
            "continuous": {
                "age_baseline": {"standardized_difference": -0.045},
                "etiv_baseline": {"standardized_difference": 0.31},
            },
        },
    }


@pytest.fixture
def model() -> dict[str, Any]:
    """Model results with one significant and one non-significant region."""
    term = "time:dx_baseline[T.patient]"
    return {
        "fdr_alpha": 0.05,
        "fits": [
            {
                "region": "lh-hippocampus",
                "estimable": True,
                "coefficients": {term: -92.8},
                "std_errors": {term: 30.0},
                "q_values": {term: 0.04},
            },
            {
                "region": "rh-hippocampus",
                "estimable": True,
                "coefficients": {term: -76.6},
                "std_errors": {term: 40.0},
                "q_values": {term: 0.14},
            },
        ],
        "sensitivity": {"applicable": False, "rows": []},
    }


PRIMARY = "time:dx_baseline[T.patient]"


def test_every_chart_returns_none_on_empty_input() -> None:
    """A chart with nothing to show must not render an empty frame."""
    assert charts.funnel_chart(EMPTY) is None
    assert charts.qc_status_chart(EMPTY) is None
    assert charts.qc_flags_chart(EMPTY) is None
    assert charts.missingness_chart(EMPTY) is None
    assert charts.batch_sizes_chart(EMPTY, 20) is None
    assert charts.confound_crosstab_chart(EMPTY) is None
    assert charts.forest_chart(EMPTY, PRIMARY) is None
    assert charts.sensitivity_chart(EMPTY) is None
    assert charts.completer_smd_chart(EMPTY) is None


def test_sensitivity_is_none_when_the_arms_are_not_comparable(model: dict[str, Any]) -> None:
    """A run where harmonization changed nothing must not plot one fit twice.

    ``applicable`` comes from whether the two frames' *values* differ, so this
    guards the same degenerate case the tables already handle — ``config/test.yaml``
    puts every batch below ``min_batch_size`` by design.
    """
    assert charts.sensitivity_chart(model) is None


def test_sensitivity_renders_once_the_arms_actually_differ(model: dict[str, Any]) -> None:
    """And is non-vacuous: the same input with differing arms does produce a figure."""
    model["sensitivity"] = {
        "applicable": True,
        "rows": [
            {
                "region": "lh-hippocampus",
                "harmonized": -92.8,
                "unharmonized": 58.0,
                "comparable": True,
            }
        ],
    }
    fig = charts.sensitivity_chart(model)
    assert fig is not None
    assert {trace.name for trace in fig.data} == {"unharmonized", "harmonized (primary)"}


def test_funnel_marks_unexplained_loss_as_critical(accounting: dict[str, Any]) -> None:
    """Unexplained loss is a bug, not a quantity, and is coloured as one."""
    clean = charts.funnel_chart(accounting)
    assert clean is not None
    assert set(clean.data[0].marker.color) == {charts.SERIES_1}

    accounting["funnel"][1]["unexplained"] = 4
    flagged = charts.funnel_chart(accounting)
    assert flagged is not None
    assert charts.STATUS_CRITICAL in flagged.data[0].marker.color


def test_qc_status_uses_the_reserved_status_palette(accounting: dict[str, Any]) -> None:
    """Status colours are reserved; a series colour must never impersonate one."""
    fig = charts.qc_status_chart(accounting)
    assert fig is not None
    assert fig.layout.barmode == "stack"
    colours = {trace.name: trace.marker.color for trace in fig.data}
    assert colours == {"PASS": charts.STATUS_GOOD, "WARNING": charts.STATUS_WARNING}
    assert fig.layout.showlegend, "WARNING is sub-3:1 on light; colour must not carry it alone"


def test_forest_separates_significance_without_relying_on_colour(model: dict[str, Any]) -> None:
    """FDR significance is encoded by marker fill plus a legend, never by hue."""
    fig = charts.forest_chart(model, PRIMARY)
    assert fig is not None
    assert len(fig.data) == 2
    symbols = {trace.marker.symbol for trace in fig.data}
    assert symbols == {"circle", "circle-open"}
    assert {trace.marker.color for trace in fig.data} == {charts.SERIES_1}
    assert fig.layout.showlegend


def test_batch_sizes_flags_below_threshold_with_a_label(accounting: dict[str, Any]) -> None:
    """status-warning is sub-3:1 on light, so the label is the mitigation."""
    fig = charts.batch_sizes_chart(accounting, 20)
    assert fig is not None
    assert charts.STATUS_WARNING in fig.data[0].marker.color
    assert any("below threshold" in t for t in fig.data[0].text)


def test_single_series_charts_carry_no_legend(accounting: dict[str, Any]) -> None:
    """A legend for one series is chrome; the title already names it."""
    for fig in (charts.qc_flags_chart(accounting), charts.completer_smd_chart(accounting)):
        assert fig is not None
        assert not fig.layout.showlegend


def test_plotly_is_inlined_exactly_once(accounting: dict[str, Any], model: dict[str, Any]) -> None:
    """§2.8 wants one self-contained file, not one bundle per figure."""
    rendered = charts.build(accounting, EMPTY, model, PRIMARY, 20)
    assert len(rendered) >= 4
    fragments = list(rendered.values())
    carrying = [f for f in fragments if "Plotly.newPlot" in f and len(f) > 1_000_000]
    assert len(carrying) == 1, "the bundle must ride on exactly one fragment"


def test_build_omits_charts_whose_data_is_absent(model: dict[str, Any]) -> None:
    """Absent sections are omitted so the template can say why, not left blank."""
    rendered = charts.build(EMPTY, EMPTY, model, PRIMARY, None)
    assert "funnel" not in rendered
    assert "qc_status" not in rendered
    assert "forest" in rendered


def test_theme_script_carries_a_dark_step_for_every_light_series() -> None:
    """Dark mode is selected, not flipped: every series hue needs its own step."""
    script = charts.theme_script()
    for light, dark in charts.DARK_FOR_LIGHT.items():
        assert light in script and dark in script
    assert "prefers-color-scheme: dark" in script
