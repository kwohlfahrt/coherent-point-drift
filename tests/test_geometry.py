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
