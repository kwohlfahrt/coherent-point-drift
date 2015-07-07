from numpy import array

def RMSD(X, Y):
    from numpy import sqrt
    dist = pairwiseDistanceSquared(X, Y)
    return sqrt(1/dist.shape[0] * dist.min(axis=0).sum())

def pairwiseDistanceSquared(X, Y):
    return ((X[None, :, :] - Y[:, None, :]) ** 2).sum(axis=2)

def rotationMatrix(*theta):
    from numpy import eye, roll
    from math import cos, sin

    if len(theta) == 1:
        theta, = theta
        R = eye(2) * cos(theta)
        R[1, 0] = sin(theta)
        R[0, 1] = -sin(theta)
    elif len(theta) == 3:
        R = eye(3)
        for axis, axis_theta in enumerate(theta):
            axis_R = eye(3)
            axis_R[1:, 1:] = rotationMatrix(axis_theta)
            axis_R = roll(roll(axis_R, axis, 0), axis, 1)
            R = R.dot(axis_R)
    else:
        raise ValueError("Only defined for D in [2..3], not {}"
                         .format(len(theta)))
    return R

def spacedRotations(D, step):
    from math import pi, cos, sin
    from itertools import product as cartesian
    from util import frange


    if D == 2:
        thetas = ((theta,) for theta in frange(0, 2*pi, step))
    elif D == 3:
        thetas = cartesian(frange(0, 2*pi, step), repeat=D)
    else:
        raise NotImplementedError("Only defined for D in [2..3], not {}"
                                  .format(D))
    yield from (rotationMatrix(*theta) for theta in thetas)

def rigidXform(X, R=array(1), t=0.0, s=1.0):
    return s * R.dot(X.T).T + t
