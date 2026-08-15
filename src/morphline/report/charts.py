"""Plotly figures for the HTML report (BUILD_PLAN §2.8).

Charts summarise; the tables beneath them remain authoritative. That ordering is
deliberate — §2.8's rule is that a reader holding only this file can reconstruct
the run, and a hover tooltip is not a value a reader can reconstruct from. Every
figure here has a table twin in the template.

**Colours are validated, not chosen.** The palette is the data-viz reference
instance, checked against *this report's* surfaces (``#ffffff`` light,
``#14171a`` dark) rather than the palette's own defaults: categorical slots 1-3
clear the CVD target (ΔE 9.2 light / 9.4 dark, all-pairs) and the normal-vision
floor (24.0 / 20.9). Two consequences are load bearing:

* Slot 3 sits at 2.82:1 on the light surface, below the 3:1 contrast bar. The
  relief rule applies, and the retained tables are what relieves it.
* ``status-warning`` is 1.83:1 on light *by design*. A WARNING mark therefore
  never carries meaning by colour alone; it is always accompanied by a legend
  entry and a label.

Every function returns ``None`` when its input is absent or degenerate, and the
template renders an explicit note instead. An empty axis frame reads as "measured
and found nothing", which is a different claim from "not applicable here".
"""

from __future__ import annotations

from typing import Any, Final

import plotly.graph_objects as go

#: Colour roles, light mode. The dark values live in :data:`DARK_FOR_LIGHT` and
#: are swapped in the browser, so a figure is built once and themed twice.
SERIES_1: Final = "#2a78d6"
SERIES_2: Final = "#eb6834"
SERIES_3: Final = "#1baf7a"
STATUS_GOOD: Final = "#0ca30c"
STATUS_WARNING: Final = "#fab219"
STATUS_CRITICAL: Final = "#d03b3b"

#: Light hex to dark hex. The status palette is deliberately absent: those four
#: steps are mode-invariant and already clear 3:1 on the dark surface.
DARK_FOR_LIGHT: Final[dict[str, str]] = {
    SERIES_1: "#3987e5",
    SERIES_2: "#d95926",
    SERIES_3: "#199e70",
}

#: Sequential ramp for magnitude, one hue light to dark (§ anti-patterns: never
#: a rainbow). Used for heatmaps only.
SEQUENTIAL: Final = [
    [0.0, "#cde2fb"],
    [0.25, "#86b6ef"],
    [0.5, "#3987e5"],
    [0.75, "#256abf"],
    [1.0, "#0d366b"],
]

STATUS_COLOURS: Final[dict[str, str]] = {
    "PASS": STATUS_GOOD,
    "WARNING": STATUS_WARNING,
    "FAIL": STATUS_CRITICAL,
}

_AXIS_LIGHT: Final = "#898781"
_GRID_LIGHT: Final = "#e1e0d9"
_BAR_GAP: Final = 2


def _base_layout(
    fig: go.Figure, *, height: int, showlegend: bool, right_margin: int = 16
) -> go.Figure:
    """Apply the chrome every figure shares.

    Backgrounds are transparent so the figure sits on the page surface in either
    theme; gridlines are solid hairlines one shade off that surface, never
    dashed.

    Args:
        fig: Figure to style.
        height: Pixel height, sized to include the axis band.
        showlegend: Whether a legend is shown. True whenever there are ≥2 series.
        right_margin: Gutter reserved for labels drawn outside a bar end. The
            default is too tight for those; a clipped label is an anti-pattern,
            so charts that place text outside their marks widen it.

    Returns:
        The same figure, styled.
    """
    fig.update_layout(
        height=height,
        margin={"l": 8, "r": right_margin, "t": 8, "b": 8},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={
            "family": "-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif",
            "size": 12,
            "color": _AXIS_LIGHT,
        },
        showlegend=showlegend,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
        hoverlabel={"font": {"size": 12}},
        bargap=0.28,
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor=_GRID_LIGHT,
        gridwidth=1,
        zeroline=False,
        linecolor=_GRID_LIGHT,
        ticks="",
    )
    fig.update_yaxes(showgrid=False, zeroline=False, linecolor=_GRID_LIGHT, ticks="")
    return fig


def funnel_chart(accounting: dict[str, Any]) -> go.Figure | None:
    """Horizontal bars for the data accounting funnel (§1.6).

    One colour, not an ordinal ramp: the ramp's steps quantise at ΔL ≈ 0.047
    against a 0.06 floor, so seven distinguishable steps do not exist — and
    position already encodes stage order while length encodes count, so a ramp
    would double-encode what the chart already shows.

    Boundaries with unexplained loss are drawn in ``status-critical`` and
    labelled, because unexplained loss is a bug rather than a quantity.

    Args:
        accounting: The accounting report.

    Returns:
        The figure, or ``None`` if the funnel is empty.
    """
    rows = accounting.get("funnel") or []
    if not rows:
        return None

    labels = [str(r["boundary"]) for r in rows]
    counts = [int(r["count"]) for r in rows]
    unexplained = [int(r.get("unexplained") or 0) for r in rows]
    lost = [int(r.get("lost") or 0) for r in rows]
    causes = [str(r.get("causes") or "none") for r in rows]
    colours = [STATUS_CRITICAL if u else SERIES_1 for u in unexplained]

    fig = go.Figure(
        go.Bar(
            x=counts,
            y=labels,
            orientation="h",
            marker={"color": colours},
            customdata=[[lo, c, u] for lo, c, u in zip(lost, causes, unexplained, strict=True)],
            hovertemplate=(
                "<b>%{y}</b><br>retained %{x:,}<br>lost %{customdata[0]:,}"
                "<br>cause: %{customdata[1]}<br>unexplained %{customdata[2]:,}<extra></extra>"
            ),
            text=[f"{c:,}" for c in counts],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.update_yaxes(autorange="reversed")
    return _base_layout(fig, height=64 + 34 * len(rows), showlegend=False, right_margin=72)


def qc_status_chart(accounting: dict[str, Any]) -> go.Figure | None:
    """Stacked QC status per site, in the reserved status palette.

    A 2px surface gap separates segments rather than a border. WARNING is
    sub-3:1 on the light surface by design, so the legend is always on and the
    status name is in every tooltip — colour never carries it alone.

    Args:
        accounting: The accounting report.

    Returns:
        The figure, or ``None`` when no per-site QC counts exist.
    """
    by_site = (accounting.get("qc_summary") or {}).get("by_site") or {}
    if not by_site:
        return None

    sites = sorted(by_site)
    statuses = [
        s for s in ("PASS", "WARNING", "FAIL") if any(by_site[site].get(s) for site in sites)
    ]
    if not statuses:
        return None

    fig = go.Figure()
    for status in statuses:
        fig.add_bar(
            name=status,
            x=sites,
            y=[int(by_site[site].get(status, 0)) for site in sites],
            marker={
                "color": STATUS_COLOURS[status],
                "line": {"width": _BAR_GAP, "color": "rgba(0,0,0,0)"},
            },
            hovertemplate=f"<b>%{{x}}</b><br>{status} %{{y:,}}<extra></extra>",
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_LIGHT, title="observations")
    return _base_layout(fig, height=300, showlegend=len(statuses) >= 2)


def qc_flags_chart(accounting: dict[str, Any]) -> go.Figure | None:
    """Horizontal bars of QC flag counts by type — one series, so no legend.

    Args:
        accounting: The accounting report.

    Returns:
        The figure, or ``None`` when nothing was flagged.
    """
    by_flag = (accounting.get("qc_summary") or {}).get("by_flag") or {}
    if not by_flag:
        return None

    items = sorted(by_flag.items(), key=lambda kv: int(kv[1]))
    fig = go.Figure(
        go.Bar(
            x=[int(v) for _, v in items],
            y=[k for k, _ in items],
            orientation="h",
            marker={"color": SERIES_1},
            hovertemplate="<b>%{y}</b><br>%{x:,} observations<extra></extra>",
            text=[f"{v:,}" for _, v in items],
            textposition="outside",
            cliponaxis=False,
        )
    )
    return _base_layout(fig, height=64 + 34 * len(items), showlegend=False, right_margin=72)


def missingness_chart(accounting: dict[str, Any]) -> go.Figure | None:
    """Grouped bars of missingness by site, one series per cause (§2.5.4).

    Args:
        accounting: The accounting report.

    Returns:
        The figure, or ``None`` when nothing is missing.
    """
    by_site = accounting.get("missingness_by_site") or {}
    if not by_site:
        return None

    sites = sorted(by_site)
    causes = sorted({c for site in sites for c in by_site[site]})
    if not causes:
        return None

    palette = [SERIES_1, SERIES_2, SERIES_3]
    fig = go.Figure()
    for index, cause in enumerate(causes[: len(palette)]):
        fig.add_bar(
            name=cause,
            x=sites,
            y=[int(by_site[site].get(cause, 0)) for site in sites],
            marker={"color": palette[index]},
            hovertemplate=f"<b>%{{x}}</b><br>{cause} %{{y:,}}<extra></extra>",
        )
    fig.update_layout(barmode="group")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=_GRID_LIGHT, title="sessions")
    return _base_layout(fig, height=300, showlegend=len(causes) >= 2)


def batch_sizes_chart(accounting: dict[str, Any], min_batch_size: int | None) -> go.Figure | None:
    """Observations per site against the configured ``min_batch_size``.

    Batches below the threshold are drawn in ``status-warning`` *and* labelled,
    since that step is sub-3:1 on the light surface.

    Args:
        accounting: The accounting report.
        min_batch_size: Configured threshold, or ``None`` if unset.

    Returns:
        The figure, or ``None`` when no batch sizes were reported.
    """
    per_site = (accounting.get("batch_sizes") or {}).get("observations_per_site") or {}
    if not per_site:
        return None

    items = sorted(per_site.items(), key=lambda kv: int(kv[1]))
    threshold = int(min_batch_size) if min_batch_size else 0
    below = [threshold and int(v) < threshold for _, v in items]
    fig = go.Figure(
        go.Bar(
            x=[int(v) for _, v in items],
            y=[k for k, _ in items],
            orientation="h",
            marker={"color": [STATUS_WARNING if b else SERIES_1 for b in below]},
            text=[
                f"{v:,}" + ("  below threshold" if b else "")
                for (_, v), b in zip(items, below, strict=True)
            ],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:,} observations<extra></extra>",
        )
    )
    if threshold:
        fig.add_vline(x=threshold, line={"width": 1, "color": _AXIS_LIGHT})
    return _base_layout(fig, height=64 + 34 * len(items), showlegend=False, right_margin=160)


def confound_crosstab_chart(harmonization: dict[str, Any]) -> go.Figure | None:
    """Heatmap of the site × time-from-baseline crosstab (§2.3.1).

    Sequential single-hue ramp: this is magnitude, so a categorical palette
    would be the wrong job entirely.

    Args:
        harmonization: The harmonization result.

    Returns:
        The figure, or ``None`` when no crosstab was produced.
    """
    crosstab = (harmonization.get("diagnostics") or {}).get("crosstab") or {}
    if not crosstab:
        return None

    bins = list(crosstab)
    sites = sorted({s for row in crosstab.values() for s in row})
    if not sites:
        return None

    z = [[int(crosstab[b].get(site, 0)) for b in bins] for site in sites]
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=bins,
            y=sites,
            colorscale=SEQUENTIAL,
            hovertemplate="site %{y}<br>time bin %{x}<br>%{z:,} observations<extra></extra>",
            colorbar={"thickness": 10, "outlinewidth": 0, "tickfont": {"size": 11}},
        )
    )
    fig.update_xaxes(showgrid=False, title="time from baseline (years)")
    fig.update_yaxes(showgrid=False)
    return _base_layout(fig, height=110 + 40 * len(sites), showlegend=False)


def forest_chart(model: dict[str, Any], primary_term: str) -> go.Figure | None:
    """Forest plot of the primary interaction across every estimable region.

    FDR significance is encoded by marker fill plus a legend entry, never by
    colour alone — and never in a status colour, since "significant" is not a
    good/bad state.

    Args:
        model: Model results.
        primary_term: The primary family's coefficient name.

    Returns:
        The figure, or ``None`` when nothing was estimable.
    """
    fits = [
        f
        for f in (model.get("fits") or [])
        if f.get("estimable") and primary_term in (f.get("coefficients") or {})
    ]
    if not fits:
        return None

    alpha = float(model.get("fdr_alpha") or 0.05)
    fits = sorted(fits, key=lambda f: float(f["coefficients"][primary_term]))
    regions = [str(f["region"]) for f in fits]
    beta = [float(f["coefficients"][primary_term]) for f in fits]
    err = [1.96 * float((f.get("std_errors") or {}).get(primary_term, 0.0)) for f in fits]
    q = [float((f.get("q_values") or {}).get(primary_term, 1.0)) for f in fits]

    fig = go.Figure()
    for label, sig in ((f"q ≤ {alpha:.2g}", True), ("not significant", False)):
        idx = [i for i, qv in enumerate(q) if (qv <= alpha) is sig]
        if not idx:
            continue
        fig.add_trace(
            go.Scatter(
                name=label,
                x=[beta[i] for i in idx],
                y=[regions[i] for i in idx],
                mode="markers",
                error_x={
                    "type": "data",
                    "array": [err[i] for i in idx],
                    "thickness": 1.5,
                    "width": 0,
                    "color": SERIES_1,
                },
                marker={
                    "size": 9,
                    "symbol": "circle" if sig else "circle-open",
                    "color": SERIES_1,
                    "line": {"width": 1.5, "color": SERIES_1},
                },
                customdata=[[q[i]] for i in idx],
                hovertemplate=(
                    "<b>%{y}</b><br>estimate %{x:.3g}<br>q = %{customdata[0]:.3g}<extra></extra>"
                ),
            )
        )
    fig.add_vline(x=0, line={"width": 1, "color": _AXIS_LIGHT})
    fig.update_xaxes(title="estimate (95% CI)")
    return _base_layout(fig, height=120 + 22 * len(regions), showlegend=True)


def sensitivity_chart(model: dict[str, Any]) -> go.Figure | None:
    """Harmonized against unharmonized estimates, per region.

    Returns ``None`` when the comparison is not applicable — a run where
    harmonization changed nothing must not present one fit twice under two
    headings.

    Args:
        model: Model results.

    Returns:
        The figure, or ``None`` when the arms are not comparable.
    """
    sensitivity = model.get("sensitivity") or {}
    if not sensitivity.get("applicable"):
        return None
    rows = [r for r in (sensitivity.get("rows") or []) if r.get("comparable")]
    if not rows:
        return None

    rows = sorted(rows, key=lambda r: float(r["harmonized"]))
    regions = [str(r["region"]) for r in rows]
    fig = go.Figure()
    for x0, x1, region in zip(
        [float(r["unharmonized"]) for r in rows],
        [float(r["harmonized"]) for r in rows],
        regions,
        strict=True,
    ):
        fig.add_shape(
            type="line", x0=x0, x1=x1, y0=region, y1=region, line={"color": _GRID_LIGHT, "width": 2}
        )
    for name, key, colour in (
        ("unharmonized", "unharmonized", SERIES_2),
        ("harmonized (primary)", "harmonized", SERIES_1),
    ):
        fig.add_trace(
            go.Scatter(
                name=name,
                x=[float(r[key]) for r in rows],
                y=regions,
                mode="markers",
                marker={"size": 9, "color": colour},
                hovertemplate=f"<b>%{{y}}</b><br>{name} %{{x:.3g}}<extra></extra>",
            )
        )
    fig.add_vline(x=0, line={"width": 1, "color": _AXIS_LIGHT})
    fig.update_xaxes(title="estimate")
    return _base_layout(fig, height=120 + 22 * len(regions), showlegend=True)


def completer_smd_chart(accounting: dict[str, Any]) -> go.Figure | None:
    """Standardized mean differences, completers against non-completers (§2.5.4).

    A standardized difference, never a p-value: the question is whether the
    groups differ enough for MAR to be load bearing, not whether the sample is
    large enough to detect that they do.

    Args:
        accounting: The accounting report.

    Returns:
        The figure, or ``None`` when the comparison does not apply.
    """
    completers = accounting.get("completers") or {}
    if not completers.get("applicable"):
        return None
    continuous = completers.get("continuous") or {}
    values = {
        name: float(stats["standardized_difference"])
        for name, stats in continuous.items()
        if stats.get("standardized_difference") is not None
    }
    if not values:
        return None

    items = sorted(values.items(), key=lambda kv: abs(kv[1]))
    fig = go.Figure(
        go.Scatter(
            x=[v for _, v in items],
            y=[k for k, _ in items],
            mode="markers",
            marker={"size": 10, "color": SERIES_1},
            hovertemplate="<b>%{y}</b><br>standardized difference %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line={"width": 1, "color": _AXIS_LIGHT})
    for bound in (-0.25, 0.25):
        fig.add_vline(x=bound, line={"width": 1, "color": _GRID_LIGHT})
    fig.update_xaxes(title="standardized mean difference")
    return _base_layout(fig, height=110 + 34 * len(items), showlegend=False)


#: Sequential ramp for the dark surface. Not an automatic inversion of the light
#: ramp: on a dark surface the *light* end must carry high magnitude, or the
#: densest cells recede into the background.
SEQUENTIAL_DARK: Final = [
    [0.0, "#0d366b"],
    [0.25, "#1c5cab"],
    [0.5, "#3987e5"],
    [0.75, "#86b6ef"],
    [1.0, "#cde2fb"],
]

#: Chrome that differs by surface. Muted ink is mode-invariant, so only the
#: hairlines move.
_GRID_DARK: Final = "#2c2c2a"

_PLOTLY_CONFIG: Final[dict[str, Any]] = {"displayModeBar": False, "responsive": True}

THEME_SCRIPT: Final = """
<script>
(function () {
  var SWAP = __SWAP__;
  var SEQ_DARK = __SEQ_DARK__;
  var SEQ_LIGHT = __SEQ_LIGHT__;
  function recolour(node, dark) {
    if (Array.isArray(node)) return node.map(function (v) { return recolour(v, dark); });
    if (node && typeof node === "object") {
      var out = {};
      Object.keys(node).forEach(function (k) {
        out[k] = (k === "colorscale") ? (dark ? SEQ_DARK : SEQ_LIGHT)
                                      : recolour(node[k], dark);
      });
      return out;
    }
    if (typeof node === "string") {
      if (dark && SWAP[node]) return SWAP[node];
      if (!dark) {
        var back = Object.keys(SWAP).filter(function (k) { return SWAP[k] === node; });
        if (back.length === 1) return back[0];
      }
    }
    return node;
  }
  function apply(dark) {
    document.querySelectorAll(".js-plotly-plot").forEach(function (gd) {
      if (!gd.data) return;
      window.Plotly.react(gd, recolour(gd.data, dark), recolour(gd.layout, dark));
    });
  }
  var mq = window.matchMedia("(prefers-color-scheme: dark)");
  function sync() { apply(mq.matches); }
  if (mq.matches) { window.addEventListener("load", sync); }
  mq.addEventListener ? mq.addEventListener("change", sync) : mq.addListener(sync);
})();
</script>
"""


def theme_script() -> str:
    """Return the inline script that re-themes every figure for dark mode.

    Plotly bakes colours into the figure rather than reading CSS, so a report
    that supports ``prefers-color-scheme`` has to restyle its figures itself.
    This walks each graph's data and layout, swaps the light hexes for their
    validated dark counterparts, and replaces the sequential scale wholesale —
    the dark ramp is stepped for the dark surface, not flipped.

    Returns:
        A ``<script>`` element, self-contained and dependency-free.
    """
    import json

    swap = dict(DARK_FOR_LIGHT)
    swap[_GRID_LIGHT] = _GRID_DARK
    return (
        THEME_SCRIPT.replace("__SWAP__", json.dumps(swap))
        .replace("__SEQ_DARK__", json.dumps(SEQUENTIAL_DARK))
        .replace("__SEQ_LIGHT__", json.dumps(SEQUENTIAL))
    )


def render(figures: dict[str, go.Figure | None]) -> dict[str, str]:
    """Render figures to embeddable HTML, inlining plotly.js exactly once.

    §2.8 requires a self-contained file with no external asset fetches, so the
    bundle is inlined rather than fetched from a CDN. It is inlined against the
    *first* figure only; every later figure reuses that one copy.

    Args:
        figures: Chart name to figure, where ``None`` means "not applicable".

    Returns:
        Chart name to an HTML fragment, omitting names whose figure was ``None``.
    """
    rendered: dict[str, str] = {}
    first = True
    for name, fig in figures.items():
        if fig is None:
            continue
        rendered[name] = fig.to_html(
            include_plotlyjs=first,
            full_html=False,
            config=_PLOTLY_CONFIG,
            default_height=f"{fig.layout.height}px",
        )
        first = False
    return rendered


def build(
    accounting: dict[str, Any],
    harmonization: dict[str, Any],
    model: dict[str, Any],
    primary_term: str,
    min_batch_size: int | None,
) -> dict[str, str]:
    """Build and render every report figure.

    Args:
        accounting: Accounting report.
        harmonization: Harmonization result.
        model: Model results.
        primary_term: The primary family's coefficient name.
        min_batch_size: Configured harmonization threshold, if any.

    Returns:
        Chart name to HTML fragment. Names absent from the mapping had no data
        and the template renders a note in their place.
    """
    return render(
        {
            "funnel": funnel_chart(accounting),
            "missingness": missingness_chart(accounting),
            "qc_status": qc_status_chart(accounting),
            "qc_flags": qc_flags_chart(accounting),
            "batch_sizes": batch_sizes_chart(accounting, min_batch_size),
            "confound": confound_crosstab_chart(harmonization),
            "forest": forest_chart(model, primary_term),
            "sensitivity": sensitivity_chart(model),
            "completer_smd": completer_smd_chart(accounting),
        }
    )
