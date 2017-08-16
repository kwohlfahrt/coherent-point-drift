from math import pi

def tryAlignment(X, Y, w, maxiter, initial_guess):
    from .util import last
    from itertools import islice

    P, xform = last(islice(driftRigid(X, initial_guess @ Y, w), maxiter))
    return P, xform @ initial_guess

def globalAlignment(X, Y, w=0.5, nsteps=7, maxiter=200, mirror=False, processes=None):
    from .geometry import spacedRotations, RMSD, RigidXform, rotationMatrix
    from functools import partial
    from itertools import starmap
    from numpy import eye
    from multiprocessing import Pool
    from itertools import chain

    D = X.shape[1]
    with Pool(processes) as p:
        initializers = list(map(
            RigidXform, starmap(rotationMatrix, spacedRotations(D, nsteps))
        ))
        xforms = p.imap_unordered(partial(tryAlignment, X, Y, w, maxiter), initializers)
        if mirror:
            reflection = eye(Y.shape[1])
            reflection[0, 0] = -1
            reflection = RigidXform(reflection)

            mirror_xforms = p.imap_unordered(
                partial(tryAlignment, X, reflection @ Y, w, maxiter), initializers
            )
            mirror_xforms = ((P, xform @ reflection) for P, xform in mirror_xforms)
        else:
            mirror_xforms = ()
        xforms = chain(xforms, mirror_xforms)
        solution = min(xforms, key=lambda xform: RMSD(X, xform[1] @ Y, xform[0]))
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
def driftAffine(X, Y, w=0.5):
    from numpy.linalg import inv
    from numpy import trace, eye, full, asarray
    from math import pi
    from .geometry import pairwiseDistanceSquared, AffineXform, std

    D = X.shape[1]
    N = len(X)
    M = len(Y)

    B = (std(X) / std(Y)) * eye(D)
    t = X.mean(axis=0) - (AffineXform(B) @ Y).mean(axis=0)

    if isinstance(w, float):
        prior = full((N, M), (1-w)/M, dtype='double')
    else:
        prior = asarray(w)

    sigma_squared = pairwiseDistanceSquared(X, AffineXform(B, t) @ Y).sum() / (D * M * N)
    while True:
        # E-step
        P = eStep(X, AffineXform(B, t) @ Y, prior, sigma_squared)

        # M-step
        N_p = P.sum()
        mu_x = 1 / N_p * X.T @ P.sum(axis=1)
        mu_y = 1 / N_p * Y.T @ P.sum(axis=0)
        X_hat = X - mu_x.T
        Y_hat = Y - mu_y.T

        #This part is different to driftRigid
        B = (X_hat.T @ P @ Y_hat) @ inv((Y_hat.T * P.sum(axis=0, keepdims=True)) @ Y_hat)
        t = mu_x - B @ mu_y
        old_sigma_squared = sigma_squared
        sigma_squared = (trace((X_hat.T * P.sum(axis=1, keepdims=True).T) @ X_hat)
                         - trace(X_hat.T @ P @ Y_hat @ B.T)) / (N_p * D)
        yield P, AffineXform(B, t)
        if abs(sigma_squared) < 1e-12 or abs(old_sigma_squared - sigma_squared) < 1e-12:
            break

def driftRigid(X, Y, w=0.5):
    from numpy.linalg import svd, det, norm
    from numpy import trace, eye, full, asarray
    from math import pi
    from .geometry import pairwiseDistanceSquared, RigidXform, std

    if not (X.ndim == Y.ndim == 2):
        raise ValueError("Expecting 2D input data, got {}D and {}D"
                         .format(X.ndim, Y.ndim))
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Expecting points with matching dimensionality, got {} and {}"
                         .format(X.shape[1:], Y.shape[1:]))

    D = X.shape[1]
    N = len(X)
    M = len(Y)

    R = eye(D)
    s = std(X) / std(Y)
    t = X.mean(axis=0) - Y.mean(axis=0)

    if isinstance(w, float):
        if not (0 <= w <= 1):
            raise ValueError("w must be in the range [0..1], got {}"
                            .format(w))
        prior = full((N, M), (1-w)/M, dtype='double')
    else:
        prior = asarray(w)

    sigma_squared = pairwiseDistanceSquared(X, RigidXform(R, t, s) @ Y).sum() / (D * M * N)
    while True:
        # E-step
        P = eStep(X, RigidXform(R, t, s) @ Y, prior, sigma_squared)

        # M-step
        N_p = P.sum()
        mu_x = 1 / N_p * X.T @ P.sum(axis=1)
        mu_y = 1 / N_p * Y.T @ P.sum(axis=0)
        X_hat = X - mu_x.T
        Y_hat = Y - mu_y.T

        # This part is different to driftAffine
        A = X_hat.T @ P @ Y_hat
        U, _, VT = svd(A)
        C = eye(D)
        C[-1, -1] = det(U @ VT)
        R = U @ C @ VT
        s = trace(A.T @ R) / trace((Y_hat.T * P.sum(axis=0, keepdims=True)) @ Y_hat)
        t = mu_x - s * R @ mu_y
        old_sigma_squared = sigma_squared
        sigma_squared = (trace((X_hat.T * P.sum(axis=1, keepdims=True).T) @ X_hat)
                         - s * trace(A.T @ R)) / (N_p * D)
        yield P, RigidXform(R, t, s)
        if abs(sigma_squared) < 1e-12 or abs(old_sigma_squared - sigma_squared) < 1e-12:
            # Sigma squared == 0 on positive fit, but occasionally ~= -1e17
            break
