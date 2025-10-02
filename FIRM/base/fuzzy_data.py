import pandas as pd
import numpy as np
import FIRM.base.membership_function as membership_function
import FIRM.base.fuzzy_set as fuzzy_set
import FIRM.base.fuzzy_linguistic_variable as fuzzy_linguistic_variable
from pyFTS.partitioners import CMeans, FCM, Entropy, Huarng
from pyFTS.common import Membership as mf


class FuzzyData(object):
    """
        Creates a new fuzzy data.
        Args:
            fv_list: a list of LinguisticVariables instances.
            name: a name of for the fuzzy data
    """

    def __init__(self, name='', fv_list=[]):

        if not fv_list:
            raise Exception("ERROR: please specify at least one fuzzy variable")
        self.fv_list = fv_list
        self.name = name

    @property
    def get_names(self):
        labels = []
        for v in self.fv_list:
            labels.append(v.name)
        return labels

    @property
    def get_nlabels(self):
        n_variables = len(self.fv_list)
        nlabels_variables = [0] * n_variables
        for i in range(n_variables):
            nlabels_variables[i] = len(self.fv_list[i].get_labels)
        return nlabels_variables

    def mean_degrees(self, dataset: pd.core.frame.DataFrame):
        if self.fv_list is None:
            raise Exception("ERROR: please specify at least one fuzzy variable")
        names_variables = dataset.columns.values
        results = pd.DataFrame(columns=names_variables)
        i = 0
        for fv in self.fv_list:
            j = 0
            column = [0] * len(fv.fs_list)
            for fs in fv.fs_list:
                column[j] = np.mean(dataset[names_variables[i]].apply(fs))
                j = j + 1
            results[names_variables[i]] = pd.Series(column)
            i = i + 1
        return results

    # Plot

    def __repr__(self):
        if self.name is None:
            text = "N/A"
        else:
            text = self.name
        return "<Fuzzy Data'" + text + "', contains linguistic variables %s" % (str(self.fv_list))


class FuzzyDataUniformTriangular(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels, labels=None):
        if labels is None:
            labels = ['Label ' + str(i) for i in list(range(0, n_labels))]
        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float or variable.dtype == int:
                fv_list.append(fuzzy_linguistic_variable.UniformTriangle(name=name_v, n_sets=n_labels,
                                                                         labels=labels,
                                                                         universe_of_discourse=[
                                                                             np.min(variable),
                                                                             np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


class FuzzyDataCMeans(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels=3, labels=None):

        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float or variable.dtype == int:
                part = CMeans.CMeansPartitioner(data=variable, npart=n_labels, func=mf.trimf)
                keys = list(part.sets.keys())
                fs_list = []
                if labels is None:
                    labels = keys
                parameters = part.sets[keys[0]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[0],
                                                                                    c=parameters[1],
                                                                                    d=parameters[2]), labels[0]))
                for i in range(1, len(keys) - 1):
                    parameters = part.sets[keys[i]].parameters
                    fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(a=parameters[0],
                                                                                       b=parameters[1],
                                                                                       c=parameters[2]), labels[i]))
                parameters = part.sets[keys[len(keys) - 1]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[1],
                                                                                    c=parameters[2],
                                                                                    d=parameters[2]),
                                                  labels[len(keys) - 1]))
                fv_list.append(fuzzy_linguistic_variable.LinguisticVariable(name=name_v,
                                                                            fs_list=fs_list,
                                                                            universe_of_discourse=[
                                                                                np.min(variable),
                                                                                np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


class FuzzyDataFCM(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels=3, labels=None):

        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float or variable.dtype == int:
                part = FCM.FCMPartitioner(data=variable, npart=n_labels, func=mf.trimf)
                keys = list(part.sets.keys())
                fs_list = []
                if labels is None:
                    labels = keys
                parameters = part.sets[keys[0]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[0],
                                                                                    c=parameters[1],
                                                                                    d=parameters[2]), labels[0]))
                for i in range(1, len(keys) - 1):
                    parameters = part.sets[keys[i]].parameters
                    fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(a=parameters[0],
                                                                                       b=parameters[1],
                                                                                       c=parameters[2]), labels[i]))
                parameters = part.sets[keys[len(keys) - 1]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[1],
                                                                                    c=parameters[2],
                                                                                    d=parameters[2]),
                                                  labels[len(keys) - 1]))
                fv_list.append(fuzzy_linguistic_variable.LinguisticVariable(name=name_v,
                                                                            fs_list=fs_list,
                                                                            universe_of_discourse=[
                                                                                np.min(variable),
                                                                                np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


class FuzzyDataEntropy(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels=3, labels=None):

        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float or variable.dtype == int:
                part = Entropy.EntropyPartitioner(data=variable, npart=n_labels, func=mf.trimf)
                keys = list(part.sets.keys())
                fs_list = []
                if labels is None:
                    labels = keys
                parameters = part.sets[keys[0]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[0],
                                                                                    c=parameters[1],
                                                                                    d=parameters[2]), labels[0]))
                for i in range(1, len(keys) - 1):
                    parameters = part.sets[keys[i]].parameters
                    fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(a=parameters[0],
                                                                                       b=parameters[1],
                                                                                       c=parameters[2]), labels[i]))
                parameters = part.sets[keys[len(keys) - 1]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[1],
                                                                                    c=parameters[2],
                                                                                    d=parameters[2]),
                                                  labels[len(keys) - 1]))
                fv_list.append(fuzzy_linguistic_variable.LinguisticVariable(name=name_v,
                                                                            fs_list=fs_list,
                                                                            universe_of_discourse=[
                                                                                np.min(variable),
                                                                                np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


class FuzzyDataHuarng(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels=3, labels=None):

        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float or variable.dtype == int:
                part = Huarng.HuarngPartitioner(data=variable, npart=n_labels, func=mf.trimf)
                keys = list(part.sets.keys())
                fs_list = []
                if labels is None:
                    labels = keys
                parameters = part.sets[keys[0]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[0],
                                                                                    c=parameters[1],
                                                                                    d=parameters[2]), labels[0]))
                for i in range(1, len(keys) - 1):
                    parameters = part.sets[keys[i]].parameters
                    fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(a=parameters[0],
                                                                                       b=parameters[1],
                                                                                       c=parameters[2]), labels[i]))
                parameters = part.sets[keys[len(keys) - 1]].parameters
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[1],
                                                                                    c=parameters[2],
                                                                                    d=parameters[2]),
                                                  labels[len(keys) - 1]))
                fv_list.append(fuzzy_linguistic_variable.LinguisticVariable(name=name_v,
                                                                            fs_list=fs_list,
                                                                            universe_of_discourse=[
                                                                                np.min(variable),
                                                                                np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)


class FuzzyDataQuantiles(FuzzyData):
    def __init__(self, name: str, dataset: pd.core.frame.DataFrame, n_labels=3, labels=None):
        if labels is None:
            labels = ['Label ' + str(i) for i in list(range(0, n_labels))]
        names_variables = dataset.columns.values
        fv_list = []
        for name_v in names_variables:
            variable = dataset[name_v]
            if variable.dtype == float or variable.dtype == int:
                parameters = [np.quantile(variable, 1 / (n_labels + 1) * a) for a in range(0, n_labels + 2)]
                fs_list = []
                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[0],
                                                                                    b=parameters[0],
                                                                                    c=parameters[1],
                                                                                    d=parameters[2]), labels[0]))
                for i in range(1, len(labels) - 1):
                    fs_list.append(fuzzy_set.FuzzySet(membership_function.TriangularMF(a=parameters[i],
                                                                                       b=parameters[i + 1],
                                                                                       c=parameters[i + 2]), labels[i]))

                fs_list.append(fuzzy_set.FuzzySet(membership_function.TrapezoidalMF(a=parameters[n_labels - 1],
                                                                                    b=parameters[n_labels],
                                                                                    c=parameters[n_labels + 1],
                                                                                    d=parameters[n_labels + 1]),
                                                  labels[len(labels) - 1]))
                fv_list.append(fuzzy_linguistic_variable.LinguisticVariable(name=name_v,
                                                                            fs_list=fs_list,
                                                                            universe_of_discourse=[
                                                                                np.min(variable),
                                                                                np.max(variable)]))
            if variable.dtype == object:
                fv_list.append(fuzzy_linguistic_variable.CrispLinguisticVariable(name_v, variable.unique()))
        super().__init__(name, fv_list)
