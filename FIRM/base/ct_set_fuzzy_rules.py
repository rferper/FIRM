import pandas as pd
import FIRM.base.operator_power as operator_power
import FIRM.base.ct_fuzzy_rule as fuzzy_rule


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

    def measures(self,fuzzy_dataset):
        out = pd.DataFrame(columns=['sentence_rule',
                                    'num_features',
                                    'fcoverage',
                                    'fsupport',
                                    'fconfidence',
                                    'fwracc'])
        out['sentence_rule'] = list(map(lambda x: fuzzy_rule.CRFuzzyRule.sentence_rule(x, fuzzy_dataset), self.rule_list))
        out['num_features'] = list(map(lambda x: fuzzy_rule.CRFuzzyRule.get_num_features(x), self.rule_list))
        out['fcoverage'] = list(map(lambda x: fuzzy_rule.CRFuzzyRule.fcoverage(x), self.rule_list))
        out['fsupport'] = list(map(lambda x: fuzzy_rule.CRFuzzyRule.fsupport(x), self.rule_list))
        out['fconfidence'] = list(map(lambda x: fuzzy_rule.CRFuzzyRule.fconfidence(x), self.rule_list))
        out['fwracc'] = list(map(lambda x: fuzzy_rule.CRFuzzyRule.fwracc(x), self.rule_list))
        out_sorted = out.sort_values(by='fconfidence', ascending=False)
        return out_sorted

    def jaccard_similarity(self,rule_list2):
        set1 = set()
        for rule in self.rule_list:
            set1.add(tuple(rule.lrule))
        set2 = set()
        for rule in rule_list2.rule_list:
            set2.add(tuple(rule.lrule))
        # Compute intersection and union
        intersection = set1.intersection(set2)
        union = set1.union(set2)
        # Calculate Jaccard Index
        if len(union) == 0:
            return 0  # Avoid division by zero
        return len(intersection) / len(union)

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