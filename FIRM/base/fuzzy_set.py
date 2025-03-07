import FIRM.base.membership_function as membership_function


class FuzzySet(object):
    def __init__(self, mem_fun: membership_function.MF, label: str):
        self._mem_fun = mem_fun
        self._label = label

    def __call__(self, x: float):
        return self._mem_fun(x)

    def get_term(self):
        """
        Return the linguistic label associated to this fuzzy set.
        """
        return self._label

    def __repr__(self):
        return "<Fuzzy set, mem_fun=%s, term='%s'>" % (self._mem_fun, self._label)
