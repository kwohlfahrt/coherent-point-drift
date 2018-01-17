from coherent_point_drift.align import *
from coherent_point_drift.util import last
from coherent_point_drift.main import loadPoints, loadXform
from pathlib import Path

import numpy as np
from itertools import islice


def test_global():
    X = loadPoints(Path("tests/fixtures/ref.txt"))
    Y = loadPoints(Path("tests/fixtures/deg.txt"))
    expected = loadXform(Path("tests/fixtures/xform.pickle"))

    P, xform = globalAlignment(X, Y, w=0.5, mirror=True)

    np.testing.assert_almost_equal(xform.R, expected.R)
    np.testing.assert_almost_equal(xform.t, expected.t)
    np.testing.assert_almost_equal(xform.s, expected.s)


def test_perfect_rigid():
    X = loadPoints(Path("tests/fixtures/ref.txt"))
    P, xform = last(islice(driftRigid(X, X), 100))

    np.testing.assert_almost_equal(xform.R, np.eye(2))
    np.testing.assert_almost_equal(xform.t, np.zeros(2))
    np.testing.assert_almost_equal(xform.s, 1)


def test_affine():
    X = loadPoints(Path("tests/fixtures/ref.txt"))
    Y = loadPoints(Path("tests/fixtures/deg.txt"))
    expected = loadXform(Path("tests/fixtures/affine.pickle"))

    P, xform = last(islice(driftAffine(X, Y, w=0.5), 100))

    np.testing.assert_almost_equal(xform.B, expected.B)
    np.testing.assert_almost_equal(xform.t, expected.t)


def test_perfect_affine():
    X = loadPoints(Path("tests/fixtures/ref.txt"))
    P, xform = last(islice(driftAffine(X, X), 100))

    np.testing.assert_almost_equal(xform.B, np.eye(2))
    np.testing.assert_almost_equal(xform.t, np.zeros(2))
