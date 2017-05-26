def align(X, Y):
    import numpy as np
    import numpy.linalg as la

    mu_y = Y.mean(axis=0)
    mu_x = X.mean(axis=0)

    X_hat = X - mu_x
    Y_hat = Y - mu_y

    ss_x = (la.norm(X_hat, axis=1) ** 2).mean()
    ss_y = (la.norm(Y_hat, axis=1) ** 2).mean()

    sigma = X_hat.T.dot(Y_hat) / len(X)

    U, D, VT = la.svd(sigma)
    S = np.eye(len(D))
    if la.det(sigma) < 0:
        S[-1, -1] = -1

    R = U.dot(S).dot(VT)
    s = 1 / ss_y * np.trace(np.diag(D).dot(S))
    t = mu_x - s * R.dot(mu_y)
    return R, t, s
