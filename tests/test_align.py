from coherent_point_drift.align import *
from coherent_point_drift.util import last
from coherent_point_drift.geometry import RMSD, rigidXform
from coherent_point_drift.main import loadPoints, loadXform
from pickle import load
from pathlib import Path

import numpy as np
import pytest
from itertools import islice

def test_global():
    X = loadPoints(Path("tests/fixtures/ref.txt"))
    Y = loadPoints(Path("tests/fixtures/deg.txt"))
    expected = loadXform(Path("tests/fixtures/xform.pickle"))

    P, xform = globalAlignment(X, Y, w=0.5, mirror=True)

    for r, e in zip(xform, expected):
        np.testing.assert_almost_equal(r, e)

def test_perfect_rigid():
    X = loadPoints(Path("tests/fixtures/ref.txt"))
    P, (R, t, s) = last(islice(driftRigid(X, X), 100))

    np.testing.assert_almost_equal(R, np.eye(2))
    np.testing.assert_almost_equal(t, np.zeros(2))
    np.testing.assert_almost_equal(s, 1)
