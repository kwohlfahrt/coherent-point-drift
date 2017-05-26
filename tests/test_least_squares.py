import numpy as np
from coherent_point_drift.least_squares import *
from coherent_point_drift.geometry import rigidXform, randomRotations, rotationMatrix
from coherent_point_drift.util import last
from itertools import islice
import random

def test_least_squares():
    np.random.seed(4)
    random.seed(6)

    ndim = 3

    R = rotationMatrix(*next(randomRotations(ndim)))
    t = np.random.normal(size=ndim)
    s = np.random.normal(size=1)[0]

    X = np.random.normal(size=(10, ndim))
    Y = rigidXform(X, R, t, s)

    alignment = align(X, Y)

    np.testing.assert_almost_equal(np.linalg.inv(R), alignment[0])
    np.testing.assert_almost_equal(1/s, alignment[2])
    np.testing.assert_almost_equal(np.linalg.inv(R).dot(-t) / s, alignment[1])
    np.testing.assert_almost_equal(rigidXform(Y, *align(X, Y)), X)


def test_cpd_prior():
    from coherent_point_drift.align import driftRigid
    np.random.seed(4)
    random.seed(6)

    ndim = 3

    R = rotationMatrix(*next(randomRotations(ndim)))
    t = np.random.normal(size=ndim)
    s = np.random.normal(size=1)[0]

    X = np.random.normal(size=(10, ndim))
    Y = rigidXform(X, R, t, s)

    cpd = last(islice(driftRigid(X, Y, w=np.eye(len(X))), 200))
    for params in zip(align(X, Y), cpd):
        np.testing.assert_almost_equal(*params)
