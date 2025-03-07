import numpy as np
import pandas as pd
import itertools


class FuzzyRule(object):
    """
    Object that represents a fuzzy rule in terms of a binary sequence
    """

    def __init__(self, brule: list):
        """
        Initializes the object that represents the fuzzy rule

        Args:
            brule: A numpy array, with the binary representation of the rule
        """
        self.ID = int(''.join(map(str, itertools.chain.from_iterable(brule))), 2)
        self.brule = brule
        self.antecedents = []
        self.consequents = []
        self.evaluations = []
        self.evaluated = 0

    def __repr__(self):
        return '<Fuzzy Rule: {}>'.format(self.brule)

    def __eq__(self, other):
        return self.ID == other.ID

    def get_num_features(self):
        n_used_features = 0
        ant = self.brule[:-1]
        n_features = len(ant)
        for i in range(n_features):
            r = np.nonzero(ant[i])[0]
            if len(r) > 0:
                n_used_features = n_used_features + 1
        return n_used_features

    def get_used_features(self):
        used_features = []
        ant = self.brule[:-1]
        n_features = len(ant)
        for i in range(n_features):
            r = np.nonzero(ant[i])[0]
            if len(r) > 0:
                used_features.append(used_features)
        return used_features

    def evaluate_antecedent_example(self, example: np.array, fuzzy_data, T):
        ant = self.brule[:-1]
        n_features = len(ant)
        out = 1
        for i in range(n_features):
            r = np.nonzero(ant[i])[0]
            if len(r) > 0:
                r = int(r)
                out = T(out, fuzzy_data.fv_list[i].fs_list[r](example.iloc[i]))
        return out

    def evaluate_consequent_example(self, example: np.array, fuzzy_variable):
        con = self.brule[-1]
        try:
            r = int(np.nonzero(con)[0])
            return fuzzy_variable.fs_list[int(r)](example.iloc[-1])
        except:
            print("Consequent is empty")

    def evaluate_rule_example(self, example: np.array, fuzzy_data, T, I):
        ant = self.evaluate_antecedent_example(example, fuzzy_data, T)
        con = self.evaluate_consequent_example(example, fuzzy_data.fv_list[-1])
        return T(ant, I(ant, con))

    def evaluate_rule_database(self, data, fuzzy_data, T, I):
        self.evaluated = 1
        self.antecedents = np.array(data.apply(lambda x: self.evaluate_antecedent_example(x, fuzzy_data, T), axis=1))
        self.consequents = np.array(
            data.apply(lambda x: self.evaluate_consequent_example(x, fuzzy_data.fv_list[-1]), axis=1))
        self.evaluations = np.array(list(map(lambda x, y: T(x, I(x, y)), self.antecedents, self.consequents)))
        return self

    def test_rule(self):
        out = pd.DataFrame(columns=['fsupport', 'fwracc', 'optimistic_estimate'])
        n_examples = len(self.antecedents)
        fsupport = sum(self.evaluations) / n_examples
        fwracc = 1 / n_examples * (sum(self.evaluations) - sum(self.antecedents) * sum(self.consequents) / n_examples)
        optimistic_estimate = sum(self.evaluations) / n_examples * (1 - sum(self.consequents) / n_examples)
        out.loc[0] = [fsupport, fwracc, optimistic_estimate]
        return out
    def fwracc_weights(self, T, weights):
        modified_eval = np.array(list(map(lambda x, y: T(x, y), self.evaluations, weights)))
        modified_ant = np.array(list(map(lambda x, y: T(x, y), self.antecedents, weights)))
        modified_con = np.array(list(map(lambda x, y: T(x, y), self.consequents, weights)))
        fwracc_weights = 1 / sum(weights) * (sum(modified_eval) - sum(modified_ant) * sum(modified_con) / sum(weights))
        return fwracc_weights

    def sentence_rule(self, fuzzy_data):
        features = fuzzy_data.fv_list[:-1]
        target = fuzzy_data.fv_list[-1]
        ant = self.brule[:-1]
        out = ['IF (']
        ping = 0
        for i in range(len(features)):
            feature = features[i]
            labels = feature.get_labels
            r = np.nonzero(ant[i])[0]
            if len(r) > 0:
                r = int(r)
                if ping == 1:
                    out.append('AND')
                out.append(feature.name)
                out.append('IS')
                out.append(str(labels[r]))
                ping = 1
        labels = target.get_labels
        out.append(')')
        out.append('THEN')
        out.append(target.name)
        out.append('IS')
        out.append(labels[int(np.nonzero(self.brule[-1])[0])])
        return ' '.join(out)

    def measures(self, I):
        out = pd.DataFrame(columns=['nfeatures',
                                    'fcoverage',
                                    'fsupport',
                                    'fconfidence',
                                    'fwracc'])
        n_used_features = self.get_num_features()
        n_examples = len(self.antecedents)
        ant = self.antecedents
        con = self.consequents
        eval = self.evaluations
        fcoverage = sum(ant) / n_examples
        fsupport = sum(eval) / n_examples
        if sum(self.antecedents) > 0:
            fconfidence = sum(eval) / sum(ant)
        else:
            fconfidence = 0
        fwracc = 1 / n_examples * (sum(eval) - sum(ant) * sum(con) / n_examples)
        out.loc[0] = [n_used_features, fcoverage, fsupport, fconfidence, fwracc]
        return out