from math import pi

def tryAlignment(X, Y, w, maxiter, rotation):
    from .util import last
    from itertools import islice
    from .geometry import rotationMatrix

    return last(islice(driftRigid(X, Y, w, rotationMatrix(*rotation)), maxiter))

def globalAlignment(X, Y, w=0.7, nsteps=12, maxiter=200):
    from .geometry import spacedRotations, RMSD, rigidXform
    from functools import partial
    from multiprocessing import Pool

    D = X.shape[1]
    with Pool() as p:
        xforms = p.imap_unordered(partial(tryAlignment, X, Y, w, maxiter),
                                  spacedRotations(D, nsteps))
        solution = min(xforms, key=lambda xform: RMSD(X, rigidXform(Y, *xform)))
    return solution

# X is the reference, Y is the points
def driftRigid(X, Y, w=0.7, initial_R=None):
    from numpy.linalg import svd, det, norm
    from numpy import exp, trace, diag
    from numpy import eye, zeros
    from numpy import seterr
    from math import pi
    from .geometry import pairwiseDistanceSquared, rigidXform

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

    # Pre-normalize input data
    pre_offsets = tuple(map(lambda x: -x.mean(axis=0), (X, Y)))
    X = X + pre_offsets[0]
    Y = Y + pre_offsets[1]
    pre_scales = tuple(map(lambda x: 1 / abs(x).max(), (X, Y)))
    X = X * pre_scales[0]
    Y = Y * pre_scales[1]

    R = initial_R if initial_R is not None else eye(D)
    t = 0
    s = 1

    sigma_squared = 1 / (D*M*N) * pairwiseDistanceSquared(rigidXform(Y, R, t, s), X).sum()
    old_exceptions = seterr(divide='ignore', over='ignore', under='ignore', invalid='raise')
    while True:
        # E-step
        pairwise_dist_squared = pairwiseDistanceSquared(rigidXform(Y, R, t, s), X)
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

        # Compensate for input normalization
        scale_out = s * pre_scales[1] / pre_scales[0]
        translation_out = (s * pre_scales[1] * R.dot(pre_offsets[1]) / pre_scales[0]
                           + t / pre_scales[0] - pre_offsets[0])
        yield R, translation_out, scale_out
