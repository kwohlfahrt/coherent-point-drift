#!/usr/bin/env python3
from math import pi

def globalAlignment(X, Y, w=0.9, step=pi/4):
    from numpy import zeros
    from geometry import spacedRotations, RMSD, rigidXform
    from itertools import chain, islice
    from util import argmin

    D = X.shape[1]

    estimates = (islice(driftRigid(X, Y, w, (rotation, zeros(D), 1.0)), 200)
                 for rotation in spacedRotations(D, step))
    return argmin(chain.from_iterable(estimates),
                  key=lambda xform: RMSD(X, rigidXform(Y, *xform)))

def driftRigid(X, Y, w=0.9, initial_guess=None):
    from numpy.linalg import svd, det
    from numpy import exp, trace, diag
    from numpy import eye, zeros
    from numpy import seterr
    from math import pi
    from geometry import pairwiseDistanceSquared, rigidXform

    if not (X.ndim == Y.ndim == 2):
        raise ValueError("Expecting 2D input data, got {}D and {}D"
                         .format(X.ndim, Y.ndim))
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Expecting points with matching dimensionality, got {} and {}"
                         .format(X.shape[1:], Y.shape[1:]))
    if not (0 <= w <= 1):
        raise ValueError("w must be in the range [0..1], got {}"
                         .format(w))

    D = X.shape[1]
    N = len(X)
    M = len(Y)

    if initial_guess is not None:
        R, t, s = initial_guess
    else:
        R, t, s = eye(D), zeros(D), 1.0

    old_exceptions = seterr(divide='raise', over='raise', under='raise', invalid='raise')

    sigma_squared = 1 / (D*M*N) * pairwiseDistanceSquared(X, rigidXform(Y, R, t, s)).sum()

    while True:
        # E-step
        pairwise_dist_squared = pairwiseDistanceSquared(X, rigidXform(Y, R, t, s))
        try:
            P = (exp(-1/(2*sigma_squared) * pairwise_dist_squared)
                / (exp(-1/(2*sigma_squared) * pairwise_dist_squared).sum(axis=0)
                    + (2 * pi * sigma_squared) ** (D/2)
                    * w / (1-w) * M / N))
        except FloatingPointError:
            seterr(**old_exceptions)
            break

        # M-step
        N_p = P.sum()
        mu_x = 1 / N_p * X.T.dot(P.T.sum(axis=1))
        mu_y = 1 / N_p * Y.T.dot(P.sum(axis=1))
        X_hat = X - mu_x.T
        Y_hat = Y - mu_y.T
        A = X_hat.T.dot(P.T).dot(Y_hat)
        U, _, VT = svd(A)
        C = eye(D)
        C[-1, -1] = det(U.dot(VT))
        R = U.dot(C).dot(VT)
        s = trace(A.T.dot(R)) / trace(Y_hat.T.dot(diag(P.sum(axis=1))).dot(Y_hat))
        t = mu_x - s * R.dot(mu_y)
        sigma_squared = 1 / (N_p * D) * (trace(X_hat.T.dot(diag(P.T.sum(axis=1))).dot(X_hat))
                                         - s * trace(A.T.dot(R)))
        yield R, t, s

if __name__ == "__main__":
    from math import sin, cos, pi, degrees
    from numpy.random import rand, seed, shuffle
    from numpy import array, eye
    from matplotlib import pyplot as plt
    from geometry import rotationMatrix, RMSD, rigidXform

    seed(4)

    N = 12
    drop = 2
    D = 2
    repeats = 2

    reference = rand(N, D)
    plt.figure()
    plt.scatter(reference[:, 0], reference[:, 1], marker='v', color='black')

    colors = rand(repeats, 3)
    errors = []

    for i in range(repeats):
        shuffle(reference) # To remove different points
        if D == 3:
            rotation = rotationMatrix(*(rand(3) * 2 * pi))
        elif D == 2:
            rotation = rotationMatrix(rand() * 2 * pi)
        scale = rand() + 0.5
        translation = rand(D)
        color = rand(3)

        moved = rigidXform(reference[:N-drop], rotation, translation, scale)
        plt.scatter(moved[:, 0], moved[:, 1], marker='o', color=colors[i], alpha=0.5)

        R, t, s = globalAlignment(reference, moved, w=0.9, step=pi/8)
        fitted = rigidXform(moved, R, t, s)
        plt.scatter(fitted[:, 0], fitted[:, 1], marker='+', color='green')
        errors.append(RMSD(reference, fitted))

    plt.figure()
    plt.scatter(range(len(errors)), errors, color=colors)
    plt.show()
