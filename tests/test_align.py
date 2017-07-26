from coherent_point_drift.align import *
from coherent_point_drift.geometry import RMSD, rigidXform
from pickle import load

import numpy as np
import pytest

def test_global():
    with open("tests/fixtures/ref.pickle", "rb") as f:
        X = load(f)
    with open("tests/fixtures/deg.pickle", "rb") as f:
        Y = load(f)
    with open("tests/fixtures/xform.pickle", "rb") as f:
        expected = load(f)
    xform = globalAlignment(X, Y, w=0.5, mirror=True)

    for r, e in zip(xform, expected):
        np.testing.assert_almost_equal(r, e)
