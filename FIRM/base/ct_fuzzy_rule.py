import numpy as np
import pandas as pd
import itertools


class CRFuzzyRule(object):
    """
    Object that represents a fuzzy rule in terms of a list of tuples, in which the last one corresponds to the consequent.
    """

    def __init__(self, lrule: list):
        """
        Initializes the object that represents the fuzzy rule

        Args:
            lrule: A list of tuples (x,y), where x is the variable and y the index of the linguistic variable considered
        """
        self.lrule = lrule
        self.antecedents = []
        self.consequents = []
        self.evaluations = []
        self.evaluated = 0

    def __repr__(self):
        return '<Fuzzy Rule: {}>'.format(self.lrule)


    def get_num_features(self):
        return len(self.lrule)-1

    def evaluate_antecedent_example(self, example: np.array, fuzzy_data, T):
        ant = self.lrule[:-1]
        n_features = len(ant)
        out = 1
        for i in range(n_features):
            out = T(out, fuzzy_data.fv_list[ant[i][0]].fs_list[ant[i][1]](example.iloc[ant[i][0]]))
        return out

    def evaluate_consequent_example(self, example: np.array, fuzzy_data):
        con = self.lrule[-1]
        return fuzzy_data.fv_list[con[0]].fs_list[con[1]](example.iloc[con[0]])

    def evaluate_rule_example(self, example: np.array, fuzzy_data, T, I):
        ant = self.evaluate_antecedent_example(example, fuzzy_data, T)
        con = self.evaluate_consequent_example(example, fuzzy_data)
        return T(ant, I(ant, con))

    def evaluate_rule_database(self, data, fuzzy_data, T, I):
        self.evaluated = 1
        self.antecedents = np.array(data.apply(lambda x: self.evaluate_antecedent_example(x, fuzzy_data, T), axis=1))
        self.consequents = np.array(
            data.apply(lambda x: self.evaluate_consequent_example(x, fuzzy_data), axis=1))
        self.evaluations = np.array(list(map(lambda x, y: T(x, I(x, y)), self.antecedents, self.consequents)))
        return self

    def fcoverage(self):
        n_examples = len(self.antecedents)
        ant = self.antecedents
        fcoverage = sum(ant) / n_examples
        return fcoverage

    def fsupport(self):
        n_examples = len(self.antecedents)
        eval = self.evaluations
        fsupport = sum(eval) / n_examples
        return fsupport

    def fconfidence(self):
        n_examples = len(self.antecedents)
        ant = self.antecedents
        eval = self.evaluations
        if sum(self.antecedents) > 0:
            fconfidence = sum(eval) / sum(ant)
        else:
            fconfidence = 0
        return fconfidence

    def fwracc(self):
        n_examples = len(self.antecedents)
        ant = self.antecedents
        con = self.consequents
        eval = self.evaluations
        fwracc = 1 / n_examples * (sum(eval) - sum(ant) * sum(con) / n_examples)
        return fwracc

    def sentence_rule(self, fuzzy_data):
        features = fuzzy_data.fv_list
        ant = self.lrule[:-1]
        con = self.lrule[-1]
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
        feature = features[con[0]]
        labels = feature.get_labels
        out.append(')')
        out.append('THEN')
        out.append(feature.name)
        out.append('IS')
        out.append(str(labels[con[1]]))
        return ' '.join(out)
