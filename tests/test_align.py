from coherent_point_drift.align import *
from coherent_point_drift.util import last
from coherent_point_drift.geometry import RMSD, rigidXform
from pickle import load

import numpy as np
import pytest
from itertools import islice

def test_global():
    with open("tests/fixtures/ref.pickle", "rb") as f:
        X = load(f)
    with open("tests/fixtures/deg.pickle", "rb") as f:
        Y = load(f)
    with open("tests/fixtures/xform.pickle", "rb") as f:
        expected = load(f)
    P, xform = globalAlignment(X, Y, w=0.5, mirror=True)

    for r, e in zip(xform, expected):
        np.testing.assert_almost_equal(r, e)

def test_perfect():
    with open("tests/fixtures/ref.pickle", "rb") as f:
        X = load(f)
    P, (R, t, s) = last(islice(driftRigid(X, X), 100))

    np.testing.assert_almost_equal(R, np.eye(2))
    np.testing.assert_almost_equal(t, np.zeros(2))
    np.testing.assert_almost_equal(s, 1)
