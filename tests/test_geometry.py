import numpy as np
import pytest
from itertools import islice

from coherent_point_drift.geometry import *

def test_random_rotations():
    np.testing.assert_equal(next(randomRotations(3, 4)), next(randomRotations(3, 4)))
    with pytest.raises(AssertionError):
        np.testing.assert_equal(*islice(randomRotations(3), 2))
    with pytest.raises(AssertionError):
        np.testing.assert_equal(next(randomRotations(3)), next(randomRotations(3)))

def test_rmsd():
    rng = np.random.RandomState(4)
    points = rng.uniform(0, 1, (10, 2))
    P = np.eye(len(points))

    assert RMSD(points, points + 1, P) > 0.5
    assert RMSD(points, points, P) == 0

def test_std():
    assert std(np.zeros((5, 3))) == 0
    unit = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    unit = np.concatenate([unit, -unit])
    assert std(unit) == 1
