import numpy as np
import pytest
from numpy.linalg import multi_dot

from fastmatrix import dimensions_from_shapes, multiply_chain, plan_dimensions


def test_plan_selects_known_optimum():
    plan = plan_dimensions((10, 100, 5, 50))
    assert plan.scalar_multiplications == 7_500
    assert plan.parenthesization() == "((A0 @ A1) @ A2)"


def test_matches_numpy_reference():
    rng = np.random.default_rng(42)
    matrices = [rng.normal(size=s) for s in [(8, 5), (5, 11), (11, 3), (3, 7)]]
    actual = multiply_chain(matrices, backend="numpy", dtype="float64")
    np.testing.assert_allclose(actual, multi_dot(matrices), rtol=1e-12, atol=1e-12)


def test_single_matrix_preserved_numerically():
    matrix = np.arange(12).reshape(3, 4)
    actual = multiply_chain([matrix], backend="numpy", dtype="float64")
    np.testing.assert_array_equal(actual, matrix)


def test_rejects_incompatible_chain():
    with pytest.raises(ValueError, match="Incompatible"):
        dimensions_from_shapes([(2, 3), (4, 5)])


def test_plan_cache_reuses_object():
    first = plan_dimensions((10, 20, 30))
    second = plan_dimensions((10, 20, 30))
    assert first is second
