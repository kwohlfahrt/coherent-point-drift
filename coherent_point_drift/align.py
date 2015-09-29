from math import pi

def tryAlignment(f, X, Y, w, maxiter, rotation):
    from .util import last
    from itertools import islice
    from .geometry import rotationMatrix

    # FIXME: Messy
    if f is driftRigid:
        initial_guess = rotationMatrix(*rotation), None, None
    elif f == driftAffine:
        initial_guess = rotationMatrix(*rotation), None
    return last(islice(f(X, Y, w, initial_guess), maxiter))

def globalAlignment(f, X, Y, w=0.5, nsteps=12, maxiter=200):
    from .geometry import spacedRotations, RMSD, rigidXform
    from functools import partial
    from multiprocessing import Pool

    D = X.shape[1]
    with Pool() as p:
        xforms = p.imap_unordered(partial(tryAlignment, f, X, Y, w, maxiter),
                                  spacedRotations(D, nsteps))
        solution = min(xforms, key=lambda xform: RMSD(X, rigidXform(Y, *xform)))
    return solution

def driftAffine(X, Y, w=0.5, initial_guess=(None, None), guess_scale=True):
    from numpy.linalg import inv
    from numpy import exp, trace, diag, std, eye
    from numpy import seterr
    from math import pi
    from .geometry import pairwiseDistanceSquared, affineXform

    D = X.shape[1]
    N = len(X)
    M = len(Y)

    B, t = initial_guess
    if B is None:
        B = eye(D)
    if guess_scale:
        s = std(X) / std(affineXform(Y, B=B))
        B = s * B
    if t is None:
        t = X.mean(axis=0) - affineXform(Y, B=B).mean(axis=0)

    sigma_squared = 1 / (D*M*N) * pairwiseDistanceSquared(affineXform(Y, B, t), X).sum()
    old_exceptions = seterr(divide='ignore', over='ignore', under='ignore', invalid='raise')
    while True:
        # E-step
        pairwise_dist_squared = pairwiseDistanceSquared(affineXform(Y, B, t), X)
        try:
            # The original algorithm expects unit variance, so normalize (2πς**2)**D/2 to compensate
            # No other parts are scale-dependent
            P = (exp(-1/(2*sigma_squared) * pairwise_dist_squared)
                / (exp(-1/(2*sigma_squared) * pairwise_dist_squared).sum(axis=0)
                    + (2 * pi * sigma_squared) ** (D/2)
                    * w / (1-w) * M / N / std(X) ** D))
        except FloatingPointError:
            seterr(**old_exceptions)
            break

        # M-step
        N_p = P.sum()
        mu_x = 1 / N_p * X.T.dot(P.T.sum(axis=1))
        mu_y = 1 / N_p * Y.T.dot(P.sum(axis=1))
        X_hat = X - mu_x.T
        Y_hat = Y - mu_y.T

        #This part is different to driftRigid
        B = (X_hat.T.dot(P.T).dot(Y_hat)
             .dot(inv(Y_hat.T.dot(diag(P.sum(axis=1))).dot(Y_hat))))
        t = mu_x - B.dot(mu_y)
        sigma_squared = 1 / (N_p * D) * (trace(X_hat.T.dot(diag(P.T.sum(axis=1))).dot(X_hat))
                                         - trace(X_hat.T.dot(P.T).dot(Y_hat).dot(B.T)))
        yield B, t


# X is the reference, Y is the points
def driftRigid(X, Y, w=0.5, initial_guess=(None, None, None)):
    from numpy.linalg import svd, det, norm
    from numpy import exp, trace, diag, std
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

    R, t, s = initial_guess
    if R is None:
        R = eye(D)
    if s is None:
        s = std(X) / std(R.dot(Y.T).T)
    if t is None:
        t = X.mean(axis=0) - rigidXform(Y, R, 0, s).mean(axis=0)

    sigma_squared = 1 / (D*M*N) * pairwiseDistanceSquared(rigidXform(Y, R, t, s), X).sum()
    old_exceptions = seterr(divide='ignore', over='ignore', under='ignore', invalid='raise')
    while True:
        # E-step
        pairwise_dist_squared = pairwiseDistanceSquared(rigidXform(Y, R, t, s), X)
        try:
            # The original algorithm expects unit variance, so normalize (2πς**2)**D/2 to compensate
            # No other parts are scale-dependent
            P = (exp(-1/(2*sigma_squared) * pairwise_dist_squared)
                / (exp(-1/(2*sigma_squared) * pairwise_dist_squared).sum(axis=0)
                    + (2 * pi * sigma_squared) ** (D/2)
                    * w / (1-w) * M / N / std(X) ** D))
        except FloatingPointError:
            seterr(**old_exceptions)
            break

        # M-step
        N_p = P.sum()
        mu_x = 1 / N_p * X.T.dot(P.T.sum(axis=1))
        mu_y = 1 / N_p * Y.T.dot(P.sum(axis=1))
        X_hat = X - mu_x.T
        Y_hat = Y - mu_y.T

        # This part is different to driftAffine
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
