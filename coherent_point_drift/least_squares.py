from .geometry import rigidXform

def score(X, Y):
    return ((X - Y) ** 2).sum()

def align(X, Y, mirror=False):
    import numpy as np
    import numpy.linalg as la

    mu_y = Y.mean(axis=0)
    mu_x = X.mean(axis=0)

    X_hat = X - mu_x
    Y_hat = Y - mu_y

    ss_x = (la.norm(X_hat, axis=1) ** 2).mean()
    ss_y = (la.norm(Y_hat, axis=1) ** 2).mean()

    sigma = X_hat.T @ Y_hat / len(X)

    U, D, VT = la.svd(sigma)
    S = np.eye(len(D))
    if la.det(sigma) < 0:
        S[-1, -1] = -1

    R = U @ S @ VT
    s = 1 / ss_y * np.trace(np.diag(D) @ S)
    t = mu_x - s * R @ mu_y
    xform = R, t, s
    if mirror:
        ndim = Y.shape[-1]
        mirrored = np.array([[1] * (Y.shape[-1] - 1) + [-1]]) * Y
        R, t, s = align(X, mirrored, mirror=False)
        R[-1, -1] *= -1
        xform = min(xform, (R, t, s), key=lambda x: score(X, rigidXform(Y, *x)))
    return xform
