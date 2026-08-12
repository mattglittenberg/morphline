"""Cross-check the in-repo estimator against ``neuroCombat``.

Build.md deviation 1 replaces ``neuroHarmonize`` with an in-repo ComBat, on the
grounds that its ``numpy==1.26.4`` pin would hold the whole project on numpy
1.x. A reviewer's first question is whether the replacement is actually the
same method, and "we wrote it from the paper" is an assertion. This makes it
checkable.

``neuroCombat`` is a different package from ``neuroHarmonize`` and declares no
dependencies at all, so it cannot drag numpy backwards. It is an optional extra
rather than a dev dependency::

    uv sync --extra combat-xcheck
    uv run pytest -m optional

Without it installed this module skips rather than fails, and CI's default
``pytest -m "not slow"`` never collects it.

**This cross-check earned its place before it was written.** Reading
``neuroCombat``'s prior estimation to build the comparison is what revealed that
the in-repo shrinkage pooled across batches within a region, where Johnson et
al. pool across regions within a batch — a real defect, invisible to every test
that only asked whether recovery worked, because with three sites the wrong-axis
prior still shrinks a little and still recovers well.

Two conventions to keep straight when reading this file:

* ``neuroCombat`` takes ``dat`` as **features x samples**, the transpose of
  every other frame in this repo.
* Its ``estimates`` carry ``gamma.star`` and ``delta.star`` as batch x feature
  matrices, where ``delta.star`` is the **variance**; morphline's
  ``delta_star`` is its square root, the divisor actually used in the
  back-transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from morphline.combat import run_combat

neuroCombat = pytest.importorskip("neuroCombat", reason="install the combat-xcheck extra")

pytestmark = pytest.mark.optional

N_FEATURES = 6
N_PER_BATCH = 40
BATCHES = ("site-a", "site-b", "site-c")


@pytest.fixture(scope="module")
def case() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """A deterministic multi-batch, multi-feature case with injected effects.

    Returns:
        The long-format frame morphline consumes, the ``covars`` frame
        neuroCombat consumes, and the features x samples matrix.
    """
    rng = np.random.default_rng(20260811)

    shifts = {"site-a": -0.8, "site-b": 0.3, "site-c": 0.9}
    scales = {"site-a": 1.0, "site-b": 1.6, "site-c": 0.7}

    sites: list[str] = []
    for batch in BATCHES:
        sites.extend([batch] * N_PER_BATCH)
    n_samples = len(sites)

    age = rng.normal(70.0, 8.0, n_samples)
    sex = rng.choice(["F", "M"], n_samples)

    matrix = np.zeros((N_FEATURES, n_samples), dtype=np.float64)
    for feature in range(N_FEATURES):
        base = 10.0 + feature
        signal = base - 0.05 * (age - 70.0) + 0.4 * (sex == "M")
        noise = np.asarray([rng.normal(0.0, 0.5 * scales[s]) for s in sites])
        offset = np.asarray([shifts[s] * (1.0 + 0.2 * feature) for s in sites])
        matrix[feature] = signal + offset + noise

    rows: list[dict[str, object]] = []
    for feature in range(N_FEATURES):
        for sample in range(n_samples):
            rows.append(
                {
                    "subject_id": f"sub-{sample:04d}",
                    "region": f"region-{feature:02d}",
                    "measure_type": "volume",
                    "site": sites[sample],
                    "age_baseline": float(age[sample]),
                    "sex": str(sex[sample]),
                    "value": float(matrix[feature, sample]),
                }
            )
    long_frame = pd.DataFrame(rows)

    covars = pd.DataFrame(
        {"site": sites, "age_baseline": age, "sex": sex},
    )
    return long_frame, covars, matrix


def _reference(covars: pd.DataFrame, matrix: np.ndarray, *, eb: bool = True) -> dict[str, object]:
    """Run neuroCombat with settings matching the in-repo defaults."""
    return neuroCombat.neuroCombat(
        dat=pd.DataFrame(matrix),
        covars=covars,
        batch_col="site",
        categorical_cols=["sex"],
        continuous_cols=["age_baseline"],
        eb=eb,
        parametric=True,
        mean_only=False,
        ref_batch=None,
    )


def _ours(long_frame: pd.DataFrame, *, eb: bool = True) -> pd.DataFrame:
    """Run the in-repo estimator and return values as features x samples."""
    result = run_combat(
        long_frame,
        batch_column="site",
        covariates=("age_baseline", "sex"),
        empirical_bayes=eb,
    )
    adjusted = long_frame.assign(adjusted=result.values)
    return adjusted.pivot(index="region", columns="subject_id", values="adjusted")


def _aligned(long_frame: pd.DataFrame, ours: pd.DataFrame) -> np.ndarray:
    """Return our adjusted values in the reference's row/column order."""
    regions = sorted(long_frame["region"].unique())
    subjects = sorted(long_frame["subject_id"].unique(), key=lambda s: int(s.split("-")[1]))
    return ours.loc[regions, subjects].to_numpy(dtype=np.float64)


class TestAdjustedValues:
    """The output both implementations exist to produce."""

    def test_adjusted_values_match_without_shrinkage(
        self, case: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    ) -> None:
        """The location/scale adjustment alone, before any prior is involved.

        Checked first because it isolates the standardization, the design
        matrix, and the back-transform. A mismatch here is an algebra bug; a
        mismatch only in the shrunk case is a prior bug.

        Agreement is to about seven significant figures rather than to machine
        precision, and the gap is not ours. ``neuroCombat`` solves the design by
        explicit normal equations, ``la.inv(mod.T @ mod) @ mod.T``, while this
        implementation uses the SVD-based ``np.linalg.lstsq``. Forming and
        inverting the normal equations squares the condition number, so the two
        genuinely disagree in the last few digits and morphline is on the more
        accurate side of the difference. Tightening this tolerance would be
        asserting that we reproduce a numerical weakness.
        """
        long_frame, covars, matrix = case
        theirs = np.asarray(_reference(covars, matrix, eb=False)["data"], dtype=np.float64)
        ours = _aligned(long_frame, _ours(long_frame, eb=False))

        assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-6)
        assert not np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)

    def test_adjusted_values_match_with_empirical_bayes(
        self, case: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    ) -> None:
        long_frame, covars, matrix = case
        theirs = np.asarray(_reference(covars, matrix)["data"], dtype=np.float64)
        ours = _aligned(long_frame, _ours(long_frame))

        assert np.allclose(ours, theirs, rtol=1e-6, atol=1e-6)


class TestEstimatedParameters:
    """Agreeing outputs can mask compensating parameter errors."""

    def test_gamma_star_matches(self, case: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]) -> None:
        long_frame, covars, matrix = case
        estimates = _reference(covars, matrix)["estimates"]
        assert isinstance(estimates, dict)
        theirs = np.asarray(estimates["gamma.star"], dtype=np.float64)

        result = run_combat(long_frame, batch_column="site", covariates=("age_baseline", "sex"))
        for feature in range(N_FEATURES):
            params = result.fit.parameters[(f"region-{feature:02d}", "volume")]
            for position, batch in enumerate(BATCHES):
                assert params.gamma_star[batch] == pytest.approx(
                    float(theirs[position, feature]), rel=1e-6, abs=1e-9
                )

    def test_delta_star_matches_as_a_variance(
        self, case: tuple[pd.DataFrame, pd.DataFrame, np.ndarray]
    ) -> None:
        """neuroCombat's ``delta.star`` is the variance; ours is its root."""
        long_frame, covars, matrix = case
        estimates = _reference(covars, matrix)["estimates"]
        assert isinstance(estimates, dict)
        theirs = np.asarray(estimates["delta.star"], dtype=np.float64)

        result = run_combat(long_frame, batch_column="site", covariates=("age_baseline", "sex"))
        for feature in range(N_FEATURES):
            params = result.fit.parameters[(f"region-{feature:02d}", "volume")]
            for position, batch in enumerate(BATCHES):
                assert params.delta_star[batch] ** 2 == pytest.approx(
                    float(theirs[position, feature]), rel=1e-6
                )


def test_the_shrinkage_axis_is_the_one_neurocombat_uses(
    case: tuple[pd.DataFrame, pd.DataFrame, np.ndarray],
) -> None:
    """A regression test for the defect this cross-check found.

    ``neuroCombat`` estimates a batch's prior from that batch's effects across
    features (``gamma_bar = np.mean(gamma_hat, axis=1)``). Pooling the other way
    — across batches within a feature — still produces plausible-looking output
    and still recovers injected effects reasonably, which is why nothing else in
    the suite caught it. Only an independent implementation does.

    With three batches and six features the two axes give materially different
    answers, so agreement to 1e-6 pins the axis rather than merely the formula.
    """
    long_frame, covars, matrix = case
    theirs = np.asarray(_reference(covars, matrix)["data"], dtype=np.float64)
    unshrunk = np.asarray(_reference(covars, matrix, eb=False)["data"], dtype=np.float64)

    assert not np.allclose(theirs, unshrunk, rtol=1e-3), (
        "shrinkage must actually move the estimates, or this proves nothing"
    )
    assert np.allclose(_aligned(long_frame, _ours(long_frame)), theirs, rtol=1e-6, atol=1e-6)
