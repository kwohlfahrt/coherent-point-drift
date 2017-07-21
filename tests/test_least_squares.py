import numpy as np
from coherent_point_drift.least_squares import *
from coherent_point_drift.geometry import rigidXform, randomRotations, rotationMatrix
from coherent_point_drift.util import last
from itertools import islice

def test_least_squares():
    rng = np.random.RandomState(4)

    ndim = 3

    R = rotationMatrix(*next(randomRotations(ndim, rng)))
    t = rng.normal(size=ndim)
    s = rng.lognormal(size=1)[0]

    X = rng.normal(size=(10, ndim))
    Y = rigidXform(X, R, t, s)

    alignment = align(X, Y)

    np.testing.assert_almost_equal(np.linalg.inv(R), alignment[0])
    np.testing.assert_almost_equal(1/s, alignment[2])
    np.testing.assert_almost_equal(np.linalg.inv(R).dot(-t) / s, alignment[1])
    np.testing.assert_almost_equal(rigidXform(Y, *align(X, Y)), X)


def test_cpd_prior():
    from coherent_point_drift.align import driftRigid
    rng = np.random.RandomState(4)

    ndim = 3

    R = rotationMatrix(*next(randomRotations(ndim, rng)))
    t = rng.normal(size=ndim)
    s = rng.normal(size=1)[0]

    X = rng.normal(size=(10, ndim))
    Y = rigidXform(X, R, t, s)

    cpd = last(islice(driftRigid(X, Y, w=np.eye(len(X))), 200))
    for params in zip(align(X, Y), cpd):
        np.testing.assert_almost_equal(*params)


def test_mirror():
    from coherent_point_drift.align import driftRigid
    # L-shape
    X = np.array([[1, 0], [0, 0], [0, 1], [0, 2], [0, 3]])
    Y = X * np.array([[-1, 1]])
    np.testing.assert_almost_equal(rigidXform(Y, *align(X, Y, mirror=True)), X)
