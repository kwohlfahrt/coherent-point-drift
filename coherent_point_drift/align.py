from math import pi

def tryAlignment(X, Y, w, maxiter, initial_guess):
    from .util import last
    from itertools import islice

    return last(islice(driftRigid(X, Y, w, initial_guess), maxiter))

def globalAlignment(X, Y, w=0.5, nsteps=7, maxiter=200, mirror=False, processes=None):
    from .geometry import spacedRotations, RMSD, rigidXform, affineXform, rotationMatrix
    from functools import partial
    from itertools import starmap
    from numpy import eye
    from multiprocessing import Pool
    from itertools import chain

    D = X.shape[1]
    with Pool(processes) as p:
        xforms = p.imap_unordered(partial(tryAlignment, X, Y, w, maxiter),
                                  map(lambda R: (rotationMatrix(*R), None, None),
                                      spacedRotations(D, nsteps)))
        if mirror:
            reflection = eye(Y.shape[1])
            reflection[0, 0] = -1
            Y = affineXform(Y, reflection)
            mirror_xforms = p.imap_unordered(partial(tryAlignment, X, Y, w, maxiter),
                                             map(lambda R: (rotationMatrix(*R), None, None),
                                                 spacedRotations(D, nsteps)))
            mirror_xforms = starmap(lambda R, t, s: (R.dot(reflection), t, s), mirror_xforms)
        else:
            mirror_xforms = ()
        solution = min(chain(xforms, mirror_xforms), key=lambda xform: RMSD(X, rigidXform(Y, *xform)))
    return solution

def eStep(X, Y, prior, sigma_squared):
    from numpy import exp
    from .geometry import pairwiseDistanceSquared, std

    D = X.shape[1]
    N = len(X)
    M = len(Y)

    dist = pairwiseDistanceSquared(Y, X)
    overlap = prior * exp(-dist / (2 * sigma_squared))

    # The original algorithm expects unit variance, so normalize (2πς**2)**D/2 to compensate
    # No other parts are scale-dependent
    P = overlap / (overlap.sum(axis=0)
                   + (2 * pi * sigma_squared) ** (D / 2)
                   * (1 - prior.sum(axis=0)) / N / std(X) ** D)
    return P

# X is the reference, Y is the points
def driftAffine(X, Y, w=0.5, initial_guess=(None, None), guess_scale=True):
    from numpy.linalg import inv
    from numpy import trace, diag, eye, full
    from numpy import seterr
    from math import pi
    from .geometry import pairwiseDistanceSquared, affineXform, std

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
    if isinstance(w, float):
        prior = full((M, N), (1-w)/M, dtype='double')
    else:
        prior = asarray(w)

    sigma_squared = pairwiseDistanceSquared(affineXform(Y, B, t), X).sum() / (D * M * N)
    old_exceptions = seterr(divide='ignore', over='ignore', under='ignore', invalid='raise')
    while True:
        # E-step
        try:
            P = eStep(X, affineXform(Y, B, t), prior, sigma_squared)
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
        sigma_squared = (trace(X_hat.T.dot(diag(P.T.sum(axis=1))).dot(X_hat))
                         - trace(X_hat.T.dot(P.T).dot(Y_hat).dot(B.T))) / (N_p * D)
        yield B, t

def driftRigid(X, Y, w=0.5, initial_guess=(None, None, None)):
    from numpy.linalg import svd, det, norm
    from numpy import trace, diag, eye, full, asarray
    from numpy import seterr
    from math import pi
    from .geometry import pairwiseDistanceSquared, rigidXform, std

    if not (X.ndim == Y.ndim == 2):
        raise ValueError("Expecting 2D input data, got {}D and {}D"
                         .format(X.ndim, Y.ndim))
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Expecting points with matching dimensionality, got {} and {}"
                         .format(X.shape[1:], Y.shape[1:]))

    D = X.shape[1]
    N = len(X)
    M = len(Y)

    R, t, s = initial_guess
    if R is None:
        R = eye(D)
    if s is None:
        s = std(X) / std(rigidXform(Y, R=R))
    if t is None:
        t = X.mean(axis=0) - rigidXform(Y, R=R, s=s).mean(axis=0)
    if isinstance(w, float):
        if not (0 <= w <= 1):
            raise ValueError("w must be in the range [0..1], got {}"
                            .format(w))
        prior = full((M, N), (1-w)/M, dtype='double')
    else:
        prior = asarray(w)

    sigma_squared = pairwiseDistanceSquared(rigidXform(Y, R, t, s), X).sum() / (D * M * N)
    old_exceptions = seterr(divide='ignore', over='ignore', under='ignore', invalid='raise')
    while True:
        # E-step
        try:
            P = eStep(X, rigidXform(Y, R, t, s), prior, sigma_squared)
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
        sigma_squared = (trace(X_hat.T.dot(diag(P.T.sum(axis=1))).dot(X_hat))
                         - s * trace(A.T.dot(R))) / (N_p * D)
        yield R, t, s
