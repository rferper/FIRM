import FIRM.base.fuzzy_set as fuzzy_set
import FIRM.base.membership_function as membership_function
import numpy as np
from matplotlib.pyplot import figure, axes, plot, show, title, subplots, legend, colorbar



class LinguisticVariable(object):
    """
    Creates a new linguistic variable. Args: fs_list: a list of FuzzySet instances. name: a string providing a brief
    description of the concept represented by the linguistic variable (optional). universe_of_discourse: a list of
    two elements, specifying min and max of the universe of discourse. Optional, but it must be specified to exploit
    plotting facilities.
    """

    def __init__(self, name='', fs_list=[], universe_of_discourse=None):

        if len(fs_list) == 0:
            raise Exception("ERROR: please specify at least one fuzzy set")
        self.universe_of_discourse = universe_of_discourse
        self.fs_list = fs_list
        self.name = name

    def get_fs_dict(self):
        fs_dict = {}
        for v in self.fs_list:
            if isinstance(v, fuzzy_set.FuzzySet):
                fs_dict[v.get_term()] = v
            else:
                raise Exception("ERROR: please provide only fuzzy set objects as arguments")
        return fs_dict

    def add_fs(self, *args):
        for v in args:
            if isinstance(v, fuzzy_set.FuzzySet):
                self.fs_list[v.get_term()] = v
            else:
                raise Exception("ERROR: please provide only fuzzy set objects as arguments")

    @property
    def get_labels(self):
        labels = []
        for v in self.fs_list:
            labels.append(v.get_term())
        return labels

    def eval_fuzzy_set(self, ind_fs, x):
        return self.fs_list[ind_fs](x)

    def eval_max_fuzzy_set(self, x):
        labels = self.get_labels
        return labels[np.argmax(list(map(lambda v: v(x), self.fs_list)))]

    def eval_max_fuzzy_set2(self, x):
        index = np.argmax(list(map(lambda v: v(x), self.fs_list)))
        return self.eval_fuzzy_set(index, x)

    def get_index(self, term):
        for n, fs in enumerate(self.fs_list):
            if fs.get_term() == term:
                return n
        return -1

    def get_universe_of_discourse(self):
        return self.universe_of_discourse

    def plot(self):
        """
        This method returns a matplotlib ax, representing all fuzzy sets contained in the liguistic variable.

        Args:
            ax: the matplotlib axis to plot to.

        Returns:
            A matplotlib axis, representing all fuzzy sets contained in the liguistic variable.
        """
        fig, ax = subplots(1, 1)
        mi, ma = self.get_universe_of_discourse()
        x = np.linspace(mi, ma, 10000)
        for nn, fs in enumerate(self.fs_list):
            y = [fs(xx) for xx in x]
            ax.plot(x, y, label=fs.get_term())
            ax.legend()
            ax.set_xlabel(self.name)
            ax.set_ylabel("Membership degree")
            ax.set_ylim(bottom=-0.05, top=1.05)
        return fig, ax

    def __repr__(self):
        if self.name is None:
            text = "N/A"
        else:
            text = self.name
        return "<Linguistic variable '" + text + "', contains fuzzy sets %s, universe of discourse: %s>" % (
            str(self.fs_list), str(self.universe_of_discourse))


class UniformTriangle(LinguisticVariable):
    """
    Creates a new linguistic variable, whose universe of discourse is automatically divided in a given number of
    fuzzy sets. The sets are all symmetrical, normalized, and for each element of the universe their memberships sum
    up to 1.

        Args: n_sets: (integer) number of fuzzy sets in which the universe of discourse must be divided. terms: list
        of strings containing linguistic terms for the fuzzy sets (must be appropriate to the number of fuzzy sets).
        universe_of_discourse: a list of two elements, specifying min and max of the universe of discourse.
    """

    def __init__(self, name='', n_sets=3, labels=None, universe_of_discourse=None):

        if n_sets < 2:
            raise Exception("Cannot create linguistic variable with less than 2 fuzzy sets.")
        if universe_of_discourse is None:
            raise Exception("Cannot create linguistic variable without universe of discourse.")

        control_points = [x * 1 / (n_sets - 1) for x in range(n_sets)]
        low = universe_of_discourse[0]
        high = universe_of_discourse[1]
        control_points = [low + (high - low) * x for x in control_points]

        if labels is None:
            labels = ['case %d' % (i + 1) for i in range(n_sets)]

        fs_list = [fuzzy_set.FuzzySet(membership_function.TriangularMF(low, low, control_points[1]), labels[0])]

        for n in range(1, n_sets - 1):
            fs_list.append(
                fuzzy_set.FuzzySet(membership_function.TriangularMF(control_points[n - 1],
                                                                    control_points[n],
                                                                    control_points[n + 1]),
                                   labels[n])
            )
        fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(control_points[-2], high, high), labels[-1]))

        super().__init__(name, fs_list, universe_of_discourse)


class CrispLinguisticVariable(LinguisticVariable):
    """
        Creates a new crisp linguistic variable, whose universe of discourse is automatically divided as many crisp
        fuzzy sets as number of categories

        Args:
            name: name of the linguistic variable
            list_categories: a list with all the categories
    """

    def __init__(self, name='', list_categories=[]):
        n_fs = len(list_categories)
        fs_list = []
        dict_categories = {}
        for i in range(n_fs):
            dict_categories[list_categories[i]] = i
            fs_list.append(fuzzy_set.FuzzySet(membership_function.CrispMF(list_categories[i]), list_categories[i]))
        self.dict_categories = dict_categories
        super().__init__(name, fs_list, [0, n_fs - 1])
