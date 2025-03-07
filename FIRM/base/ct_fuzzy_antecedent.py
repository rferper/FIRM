import numpy as np
import pandas as pd
import itertools


class CRFuzzyAntecedent(object):
    """
    Object that represents a fuzzy rule in terms of a list of tuples, in which the last one corresponds to the consequent.
    """

    def __init__(self, lant: list):
        """
        Initializes the object that represents the fuzzy rule

        Args:
            lrule: A list of tuples (x,y), where x is the variable and y the index of the linguistic variable considered
        """
        self.lant = lant
        self.antecedents = []
        self.evaluated = 0

    def __repr__(self):
        return '<Fuzzy Antecedent: {}>'.format(self.lant)


    def get_num_features(self):
        return len(self.lant)

    def evaluate_antecedent_example(self, example: np.array, fuzzy_data, T):
        ant = self.lant
        n_features = len(ant)
        out = 1
        for i in range(n_features):
            out = T(out, fuzzy_data.fv_list[ant[i][0]].fs_list[ant[i][1]](example.iloc[ant[i][0]]))
        return out

    def evaluate_rule_database(self, data, fuzzy_data, T):
        self.evaluated = 1
        self.antecedents = np.array(data.apply(lambda x: self.evaluate_antecedent_example(x, fuzzy_data, T), axis=1))
        return self

    def fcoverage(self):
        n_examples = len(self.antecedents)
        ant = self.antecedents
        fcoverage = sum(ant) / n_examples
        return fcoverage

    def sentence_rule(self, fuzzy_data):
        features = fuzzy_data.fv_list
        ant = self.lant
        out = ['IF (']
        ping = 0
        for i in range(len(ant)):
            feature = features[ant[i][0]]
            labels = feature.get_labels
            if ping == 1:
                out.append('AND')
            out.append(feature.name)
            out.append('IS')
            out.append(str(labels[ant[i][1]]))
            ping = 1
        out.append(')')
        return ' '.join(out)