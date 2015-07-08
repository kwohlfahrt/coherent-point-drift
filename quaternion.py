from math import sqrt, cos, sin, acos, pi

class Quaternion:
    def __init__(self, s, i, j, k):
        self.s = s
        self.i = i
        self.j = j
        self.k = k

    @classmethod
    def fromAxisAngle(cls, v, theta):
        if len(v) != 3:
            raise ValueError("v must be a 3D vector.")

        theta = theta % (2 * pi) / 2
        try:
            # Numpy-compatible array
            v = v / v.norm() * sin(theta)
        except AttributeError:
            norm = sqrt(sum(map(lambda x: x ** 2, v)))
            v = map(lambda x: x / norm * sin(theta), v)
        return cls(cos(theta), *v)

    def __iter__(self):
        yield from (self.s, self.i, self.j, self.k)

    def __repr__(self):
        return ("Quaternion({}, {}i, {}j, {}k)"
                .format(self.s, self.i, self.j, self.k))

    def __eq__(self, other):
        return (self.s == other.s
                and self.i == other.i
                and self.j == other.j
                and self.k == other.k)

    def __hash__(self):
        return hash(self.s, self.i, self.j, self.k)

    def __add__(self, other):
        return Quaternion(self.s + other.s,
                          self.i + other.i,
                          self.j + other.j,
                          self.k + other.k)

    def __sub__(self, other):
        return Quaternion(self.s - other.s,
                          self.i - other.i,
                          self.j - other.j,
                          self.k - other.k)

    def __mul__(self, other):
        try:
            return Quaternion(
                self.s * other.s - self.i * other.i - self.j * other.j - self.k * other.k,
                self.s * other.i + self.i * other.s + self.j * other.k - self.k * other.j,
                self.s * other.j - self.i * other.k + self.j * other.s + self.k * other.i,
                self.s * other.k + self.i * other.j - self.j * other.i + self.k * other.s
            )
        except AttributeError:
            return Quaternion(other * self.s, other * self.i, other * self.j, other * self.k)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return Quaternion(self.s / other, self.i / other, self.j / other, self.k / other)

    def __abs__(self):
        return self.norm()

    def __round__(self, ndigits=0):
        return Quaternion(*(round(i, ndigits) for i in (self.s, self.i, self.j, self.k)))

    def norm(self):
        return sqrt(sum(map(lambda x: x ** 2), (self.s, self.i, self.j, self.k)))

    def reciprocal(self):
        return self.conjugate() / self.norm() ** 2

    def conjugate(self, other=None):
        if not other:
            return Quaternion(self.s, -self.i, -self.j, -self.k)
        else:
            return self * other * self.reciprocal()

    def unit(self):
        return self / self.norm()

    def matrix(self):
        r, i, j, k = self
        return [[1 - 2*j*j - 2*k*k, 2*(i*j - k*r), 2*(i*k + j*r)],
                [2*(i*j + k*r), 1 - 2*i*i - 2*k*k, 2*(j*k - i*r)],
                [2*(i*k - j*r), 2*(j*k + i*r), 1 - 2*i*i - 2*j*j]]

    @property
    def vector(self):
        return self.i, self.j, self.k

    @vector.setter
    def vector(self, *v):
        self.i, self.j, self.k = v

    @property
    def axis_angle(self):
        theta = 2 * acos(self.s)
        try:
            v = tuple(map(lambda x: x / sin(theta / 2), self.vector))
        except ZeroDivisionError:
            # No rotation, so v doesn't matter
            v = (1, 0, 0)
        return theta, v
