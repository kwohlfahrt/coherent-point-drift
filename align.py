from math import pi

def globalAlignment(X, Y, w=0.9, step=pi/6, maxiter=200):
    from numpy import zeros
    from geometry import spacedRotations, RMSD, rigidXform
    from itertools import chain, islice
    from util import argmin

    D = X.shape[1]
    estimates = (islice(driftRigid(X, Y, w, (rotation, zeros(D), 1.0)), maxiter)
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
