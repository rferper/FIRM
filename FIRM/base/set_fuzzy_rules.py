import pandas as pd
import FIRM.base.operator_power as operator_power
import FIRM.base.fuzzy_rule as fuzzy_rule


def N(x, w):
    try:
        return (1 - x) / (1 - w * x)
    except:
        return 1


class SetFuzzyRules(object):
    """
    Object that represents a collection of fuzzy rules in terms of a binary sequence
    """

    def __init__(self, rule_list=[]):
        self.rule_list = rule_list

    def measures(self, I, fuzzy_dataset):
        out = pd.DataFrame(columns=['sentence_rule',
                                    'num_features',
                                    'fcoverage',
                                    'fsupport',
                                    'fconfidence',
                                    'fwracc'])
        out['sentence_rule'] = list(map(lambda x: fuzzy_rule.FuzzyRule.sentence_rule(x, fuzzy_dataset), self.rule_list))
        for i in range(len(self.rule_list)):
            measures = self.rule_list[i].measures(I)
            out.iloc[i, 1:] = measures.values[0]
        return out

    def overall_coverage(self, T, w):
        n_examples = len(self.rule_list[0].antecedents)
        ANTS = pd.DataFrame(list(map(lambda rule: rule.antecedents, self.rule_list)))
        foverallcoverage = float(0)
        for i in range(n_examples):
            foverallcoverage = foverallcoverage + operator_power.OperatorPower(lambda x, y: N(T(N(x, w), N(y, w)), w),
                                                                               list(ANTS.iloc[:, i]))
        foverallcoverage = foverallcoverage / n_examples
        print('FOverallCoverage: {}'.format(round(foverallcoverage, 3)))
        return foverallcoverage
