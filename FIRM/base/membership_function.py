
class MF(object):

    def __init__(self):
        pass

    def __call__(self, x: float):
        return min(1, max(0, self._execute(x)))


#########################################
# USEFUL PRE-BAKED MEMBERSHIP FUNCTIONS #
#########################################


# Triangular MF
class TriangularMF(MF):
    """
        Creates a normalized triangular membership function.
        Requires a <= b <= c and the semantics is the following:
        ::
            1   |   .
                |  / \\
                | /   \\
            0   |/     \\
                ---------
                 a  b  c
        Args:
            a: universe of discourse coordinate of the leftmost vertex.
            b: universe of discourse coordinate of the upper vertex.
            c: universe of discourse coordinate of the rightmost vertex.
    """

    def __init__(self, a: float, b: float, c: float):
        self._a = a
        self._b = b
        self._c = c
        if (a > b):
            raise Exception("Error in triangular fuzzy set: a=%.2f should be <= b=%.2f" % (a, b))
        elif (b > c):
            raise Exception("Error in triangular fuzzy set: b=%.2f should be <= c=%.2f" % (b, c))

    def _execute(self, x):
        if x < self._b:
            if self._a != self._b:
                return (x - self._a) * (1 / (self._b - self._a))
            else:
                return 1
        else:
            if self._b != self._c:
                return 1 + (x - self._b) * (-1 / (self._c - self._b))
            else:
                return 1

    def __repr__(self):
        return "<Triangular MF (%f, %f, %f)>" % (self._a, self._b, self._c)

# Trapezoidal MF
class TrapezoidalMF(MF):
    """
        Creates a normalized trapezoidal membership function.
        Requires a <= b <= c <= d.
        Args:
            a: universe of discourse coordinate of the leftmost vertex.
            b: universe of discourse coordinate of the upper left vertex.
            c: universe of discourse coordinate of the upper right vertex.
            d: universe of discourse coordinate of the rightmost vertex.
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        self._a = a
        self._b = b
        self._c = c
        self._d = d

    def _execute(self, x):
        if x < self._b:
            if self._a != self._b:
                return (x - self._a) * (1 / (self._b - self._a))
            else:
                return 1
        elif x >= self._b and x <= self._c:
            return 1
        else:
            if self._c != self._d:
                return 1 + (x - self._c) * (-1 / (self._d - self._c))
            else:
                return 1
    def __repr__(self):
        return "<Trapezoidal MF (%f, %f, %f, %f)>" % (self._a, self._b, self._c,self._d)

# Crisp MF
class CrispMF(MF):

    def __init__(self, a):
        self._a = a

    def _execute(self, x):
        return int(self._a == x)

    def __repr__(self):
        return "<Crisp MF (Category %s)>" % self._a