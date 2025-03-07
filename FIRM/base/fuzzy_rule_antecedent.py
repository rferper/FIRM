import numpy as np
import pandas as pd
import itertools
import FIRM.base.fuzzy_rule as fuzzy_rule


class FuzzyRuleAntecedent(object):
    """
    Object that represents an antecedent of a fuzzy rule in terms of a binary sequence
    """

    def __init__(self, bant: list):
        self.ID = int(''.join(map(str, itertools.chain.from_iterable(bant))), 2)
        self.bant = bant
        self.antecedents = []
        self.evaluated = 0
        self.best_ind_target = 0
        self.fconfidence = 0

    def __eq__(self, other):
        return self.ID == other.ID

    def __repr__(self):
        return '<Fuzzy Antecedent: {}>'.format(self.bant)

    def get_used_features(self):
        used_features = set()
        n_features = len(self.bant)
        for i in range(n_features):
            r = np.nonzero(self.bant[i])[0]
            if len(r) > 0:
                used_features.add(i)
        return used_features

    def evaluate_example(self, example, fuzzy_data, T):
        n_features = len(self.bant)
        out = 1
        for i in range(n_features):
            r = np.nonzero(self.bant[i])[0]
            if len(r) > 0:
                r = int(r)
                out = T(out, fuzzy_data.fv_list[i].fs_list[r](example.iloc[i]))
        return out

    def evaluate_antecedent_database(self, data, fuzzy_data, T, I):
        self.evaluated = 1
        self.antecedents = np.array(data.apply(lambda x: self.evaluate_example(x, fuzzy_data, T), axis=1))
        return self

    def fcoverage(self):
        return sum(self.antecedents) / len(self.antecedents)

    def best_consequent(self, data, fuzzy_data, T, I):
        nlabels_target = len(fuzzy_data.fv_list[-1].get_labels)
        for ind_target in range(nlabels_target):
            v = [0] * nlabels_target
            v[ind_target] = 1
            rule = fuzzy_rule.FuzzyRule(self.bant + [v])
            con = np.array(data.apply(lambda x: rule.evaluate_consequent_example(x, fuzzy_data.fv_list[-1]), axis=1))
            eval = np.array(list(map(lambda x, y: T(x, I(x, y)), self.antecedents, con)))
            if sum(self.antecedents) > 0:
                fconfidence = sum(eval) / sum(self.antecedents)
            else:
                fconfidence = 0
            if fconfidence > self.fconfidence:
                self.best_ind_target = ind_target
                self.fconfidence = fconfidence
        return self

    def to_sentence_rule(self, fuzzy_data):
        nlabels_target = len(fuzzy_data.fv_list[-1].get_labels)
        v = [0] * nlabels_target
        v[self.best_ind_target] = 1
        rule = fuzzy_rule.FuzzyRule(self.bant + [v])
        return rule.sentence_rule(fuzzy_data)
