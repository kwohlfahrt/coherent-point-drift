from numpy import array
from functools import partial

def RMSD(X, Y, P):
    from numpy import sqrt

    P = P / P.sum(axis=0, keepdims=True)
    return sqrt((pairwiseDistanceSquared(X, Y) * P).mean()) / std(X)

def pairwiseDistanceSquared(X, Y):
    # R[i, j] = distance(X[i], Y[j]) ** 2
    return ((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2)

def rotationMatrix(*angles):
    from numpy import eye, array
    from math import cos, sin

    if len(angles) == 1:
        theta, = angles
        R = eye(2) * cos(theta)
        R[1, 0] = sin(theta)
        R[0, 1] = -sin(theta)
    elif len(angles) == 2:
        theta, (x, y, z) = angles
        c = cos(theta)
        s = sin(theta)
        R = array([[c+x*x*(1-c),   x*y*(1-c)-z*s, (1-c)*x*z+y*s],
                   [y*x*(1-c)+z*s, c+y*y*(1-c),   y*z*(1-c)-x*s],
                   [z*x*(1-c)-y*s, z*y*(1-c)+x*s, c+z*z*(1-c)]])
    else:
        raise NotImplementedError("Only implemented for D in [2..3], not {}"
                                  .format(len(angles)))
    return R

def spacedRotations(D, N):
    from math import pi, sin, cos, sqrt
    from itertools import product as cartesian, repeat
    from .util import frange

    if D == 2:
        yield from zip(frange(-pi, pi, 2*pi/N))
    elif D == 3:
        # Ken Shoemake
        # Graphics Gems III, pp 124-132
        from .quaternion import Quaternion
        for X, *theta in cartesian(frange(0, 1, 1/N), *repeat(frange(0, 2*pi, 2*pi/N), 2)):
            R = (sqrt(1-X), sqrt(X))
            yield Quaternion(sin(theta[0]) * R[0], cos(theta[0]) * R[0],
                             sin(theta[1]) * R[1], cos(theta[1]) * R[1]).axis_angle
    else:
        raise NotImplementedError("Only defined for D in [2..3], not {}"
                                  .format(D))
# For spaced points on a sphere, see Saff & Kuijlaars,
# The Mathematical Intelligencer Winter 1997, Volume 19, Issue 1, pp 5-11

def randomRotations(D, rng=None):
    from math import pi, sin, cos, sqrt
    from numpy.random import RandomState
    from itertools import product as cartesian, repeat
    from .util import frange

    if not isinstance(rng, RandomState):
        rng = RandomState(rng)

    if D == 2:
        while True:
            yield (rng.uniform(-pi, pi),)
    elif D == 3:
        # Ken Shoemake
        # Graphics Gems III, pp 124-132
        from .quaternion import Quaternion
        while True:
            X = rng.uniform(0, 1)
            theta = rng.uniform(0, 2*pi), rng.uniform(0, 2*pi)
            R = (sqrt(1-X), sqrt(X))
            yield Quaternion(sin(theta[0]) * R[0], cos(theta[0]) * R[0],
                             sin(theta[1]) * R[1], cos(theta[1]) * R[1]).axis_angle
    else:
        raise NotImplementedError("Only defined for D in [2..3], not {}"
                                  .format(D))

def std(x):
    from numpy import sqrt

    return sqrt(((x - x.mean(axis=0)) ** 2).sum(axis=1).mean())

class RigidXform:
    def __init__(self, R=None, t=None, s=None):
        self.R = R
        self.t = t
        self.s = s

    def matrices(self, ndim):
        from numpy import eye, zeros

        R = self.R if self.R is not None else eye(ndim)
        t = self.t if self.t is not None else zeros(ndim)
        s = self.s if self.s is not None else 1.0
        return R, t, s

    @property
    def ndim(self):
        if self.R is not None:
            return self.R.shape[0]
        elif self.t is not None:
            return len(self.t)
        else:
            return None

    @property
    def inverse(self):
        from numpy.linalg import inv
        R = inv(self.R) if self.R is not None else None
        s = 1/self.s if self.s is not None else None
        t = -self.t if self.t is not None else None

        return type(self)(R=R, s=s) @ type(self)(t=t)

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            # Compose
            ndim = self.ndim or other.ndim
            if ndim is None:
                return type(self)()

            self_R, self_t, self_s = self.matrices(ndim)
            other_R, other_t, other_s = other.matrices(ndim)

            R = self_R @ other_R
            t = self_s * self_R @ other_t + self_t
            s = self_s * other_s
            return type(self)(R, t, s)
        else:
            # Apply
            ndim = other.shape[1]
            R, t, s = self.matrices(ndim)
            return s * (R.dot(other.T)).T + t

    def __eq__(self, other):
        return self.R == other.R and self.t == other.t and self.s == other.s

    def __str__(self):
        R = '\n'.join(map(' '.join, map(partial(map, str), self.R)))
        t = ' '.join(map(str, self.t))
        s = str(self.s)
        return '\n\n'.join([R, t, s])

    @classmethod
    def normalize(cls, X):
        N, D = X.shape
        s = 1 / std(X)
        t = -X.mean(axis=0) * s
        return cls(t=t, s=s)

class AffineXform:
    def __init__(self, B=None, t=None):
        self.B = B
        self.t = t

    def matrices(self, ndim):
        from numpy import eye, zeros

        B = self.B if self.B is not None else eye(ndim)
        t = self.t if self.t is not None else zeros(ndim)
        return B, t

    @property
    def ndim(self):
        if self.B is not None:
            return self.B.shape[0]
        elif self.t is not None:
            return len(self.t)
        else:
            return None

    @property
    def inverse(self):
        from numpy.linalg import inv
        B = inv(self.B) if self.B is not None else None
        t = -self.t if self.t is not None else None

        return type(self)(B=B) @ type(self)(t=t)

    def __matmul__(self, other):
        if isinstance(other, type(self)):
            # Compose
            ndim = self.ndim or other.ndim
            if ndim is None:
                return type(self)()

            self_B, self_t = self.matrices(ndim)
            other_B, other_t = other.matrices(ndim)

            B = self_B @ other_B
            t = self_B @ other_t + self_t
            return type(self)(B, t)
        else:
            # Apply
            ndim = other.shape[1]
            B, t = self.matrices(ndim)
            return (B.dot(other.T)).T + t


    def __eq__(self, other):
        return self.B == other.B and self.t == other.t

    def __str__(self):
        B = '\n'.join(map(' '.join, map(partial(map, str), self.B)))
        t = ' '.join(map(str, self.t))
        return '\n\n'.join([B, t])

    @classmethod
    def normalize(cls, X):
        from numpy import eye

        N, D = X.shape
        s = 1 / std(X)
        t = -X.mean(axis=0) * s
        return cls(B=eye(D) * s, t=t)
