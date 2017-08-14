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
        initializers = [
            (rotationMatrix(*R), None, None) for R in spacedRotations(D, nsteps)
        ]
        xforms = p.imap_unordered(partial(tryAlignment, X, Y, w, maxiter), initializers)
        if mirror:
            reflection = eye(Y.shape[1])
            reflection[0, 0] = -1

            mirror_xforms = p.imap_unordered(
                partial(tryAlignment, X, affineXform(Y, reflection), w, maxiter), initializers
            )
            def mirror_xform(P, xform):
                R, t, s = xform
                return P, (R.dot(reflection), t, s)
            mirror_xforms = starmap(mirror_xform, mirror_xforms)
        else:
            mirror_xforms = ()
        xforms = chain(xforms, mirror_xforms)
        solution = min(xforms, key=lambda xform: RMSD(X, rigidXform(Y, *xform[1]), xform[0]))
    return solution

def eStep(X, Y, prior, sigma_squared):
    from numpy import exp
    from .geometry import pairwiseDistanceSquared, std

    D = X.shape[1]
    N = len(X)
    M = len(Y)

    dist = pairwiseDistanceSquared(X, Y)
    overlap = prior * exp(-dist / (2 * sigma_squared))

    # The original algorithm expects unit variance, so normalize (2πς**2)**D/2 to compensate
    # No other parts are scale-dependent
    P = overlap / (overlap.sum(axis=1, keepdims=True)
                   + (2 * pi * sigma_squared) ** (D / 2)
                   * (1 - prior.sum(axis=1, keepdims=True)) / N / std(X) ** D)
    return P

# X is the reference, Y is the points
def driftAffine(X, Y, w=0.5, initial_guess=(None, None), guess_scale=True):
    from numpy.linalg import inv
    from numpy import trace, eye, full
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
        prior = full((N, M), (1-w)/M, dtype='double')
    else:
        prior = asarray(w)

    sigma_squared = pairwiseDistanceSquared(X, affineXform(Y, B, t)).sum() / (D * M * N)
    while True:
        # E-step
        P = eStep(X, affineXform(Y, B, t), prior, sigma_squared)

        # M-step
        N_p = P.sum()
        mu_x = 1 / N_p * X.T.dot(P.sum(axis=1))
        mu_y = 1 / N_p * Y.T.dot(P.sum(axis=0))
        X_hat = X - mu_x.T
        Y_hat = Y - mu_y.T

        #This part is different to driftRigid
        B = (X_hat.T.dot(P).dot(Y_hat)
             .dot(inv((Y_hat.T *  P.sum(axis=0, keepdims=True)).dot(Y_hat))))
        t = mu_x - B.dot(mu_y)
        old_sigma_squared = sigma_squared
        sigma_squared = (trace((X_hat.T * P.T.sum(axis=1, keepdims=True).T).dot(X_hat))
                         - trace(X_hat.T.dot(P).dot(Y_hat).dot(B.T))) / (N_p * D)
        yield P, (B, t)
        if abs(sigma_squared) < 1e-15 or abs(old_sigma_squared - sigma_squared) < 1e-15:
            break

def driftRigid(X, Y, w=0.5, initial_guess=(None, None, None)):
    from numpy.linalg import svd, det, norm
    from numpy import trace, eye, full, asarray
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
        prior = full((N, M), (1-w)/M, dtype='double')
    else:
        prior = asarray(w)

    sigma_squared = pairwiseDistanceSquared(X, rigidXform(Y, R, t, s)).sum() / (D * M * N)
    while True:
        # E-step
        P = eStep(X, rigidXform(Y, R, t, s), prior, sigma_squared)

        # M-step
        N_p = P.sum()
        mu_x = 1 / N_p * X.T.dot(P.sum(axis=1))
        mu_y = 1 / N_p * Y.T.dot(P.sum(axis=0))
        X_hat = X - mu_x.T
        Y_hat = Y - mu_y.T

        # This part is different to driftAffine
        A = X_hat.T.dot(P).dot(Y_hat)
        U, _, VT = svd(A)
        C = eye(D)
        C[-1, -1] = det(U.dot(VT))
        R = U.dot(C).dot(VT)
        s = trace(A.T.dot(R)) / trace((Y_hat.T * P.sum(axis=0, keepdims=True)).dot(Y_hat))
        t = mu_x - s * R.dot(mu_y)
        old_sigma_squared = sigma_squared
        sigma_squared = (trace((X_hat.T * P.sum(axis=1, keepdims=True).T).dot(X_hat))
                         - s * trace(A.T.dot(R))) / (N_p * D)
        yield P, (R, t, s)
        if abs(sigma_squared) < 1e-15 or abs(old_sigma_squared - sigma_squared) < 1e-15:
            # Sigma squared == 0 on positive fit, but occasionally ~= -1e17
            break
