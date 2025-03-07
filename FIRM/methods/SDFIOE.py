import numpy as np
import math
import pandas as pd
import copy
import progressbar
import FIRM.base.fuzzy_rule as fuzzy_rule
import FIRM.base.set_fuzzy_rules as set_fuzzy_rules
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# ----- INPUT ----- #
# Data: DataFrame with input Data
# FuzzyData: DataFrame with all the fuzzy linguistic variables
# ind_target_class: Index Target Class
# I: Fuzzy implication Function
# T: t-norm
# K: Number of output rules
# max_features: Maximum number of features in the antecedent
# min_coverage: Minimum coverage for the considered rules
# w: Parameter for computing the overall coverage

def SDFIOE(dataset, fuzzy_dataset, ind_target_class, T, I, K=10, max_features=5, min_support=0.05):
    n_features = len(fuzzy_dataset.fv_list) - 1
    max_features = min(n_features, max_features)
    nlabels_features = [0] * n_features
    for i in range(n_features):
        nlabels_features[i] = len(fuzzy_dataset.fv_list[i].get_labels)
    nlabels_target = len(fuzzy_dataset.fv_list[-1].get_labels)
    number_rules = math.prod(np.array(nlabels_features, dtype='int64') + 1) - 1

    def refine_rule(parent, m, selected_rules, rules_explored):
        if parent.get_num_features() >= max_features:
            rules_explored = rules_explored + math.prod(np.array(nlabels_features[m + 1:], dtype='int64') + 1) - 1
            bar.update(rules_explored)
            return selected_rules, rules_explored
        else:
            for k in range(m + 1, n_features):
                nLL = len(fuzzy_dataset.fv_list[k].get_labels)
                for j in range(nLL):
                    rules_explored = rules_explored + 1
                    children_ant = copy.deepcopy(parent.brule[:-1])
                    children_ant[k][j] = 1
                    children_con = [[0] * nlabels_target]
                    children_con[0][ind_target_class] = 1
                    children_brule = children_ant + children_con
                    children = fuzzy_rule.FuzzyRule(children_brule)
                    children = children.evaluate_rule_database(dataset, fuzzy_dataset, T, I)
                    measures = children.test_rule()
                    if float(measures.fsupport.iloc[0]) >= min_support:
                        if len(selected_rules) < K:
                            children_df = pd.DataFrame([[children, float(measures.fwracc.iloc[0])]],
                                                       columns=['binary_rule', 'fwracc'])
                            selected_rules = pd.concat([selected_rules, children_df], ignore_index=True)
                            selected_rules = selected_rules.sort_values(by=['fwracc'], ascending=False)
                            selected_rules, rules_explored = refine_rule(children, k, selected_rules, rules_explored)
                        else:
                            if (float(measures.optimistic_estimate.iloc[0]) >= selected_rules.iloc[-1]['fwracc']):
                                if float(measures.fwracc.iloc[0]) > selected_rules.iloc[-1]['fwracc']:
                                    selected_rules.iloc[-1] = [children, float(measures.fwracc.iloc[0])]
                                    selected_rules = selected_rules.sort_values(by=['fwracc'], ascending=False)
                                selected_rules, rules_explored = refine_rule(children, k, selected_rules,
                                                                             rules_explored)
                            else:
                                if k < n_features - 1:
                                    rules_explored = rules_explored + math.prod(
                                        np.array(nlabels_features[k + 1:], dtype='int64') + 1) - 1
                                    bar.update(rules_explored)
                    else:
                        if k < n_features - 1:
                            rules_explored = rules_explored + math.prod(
                                np.array(nlabels_features[k + 1:], dtype='int64') + 1) - 1
                            bar.update(rules_explored)
        return selected_rules, rules_explored

    root_ant = [[0] * nlabels_features[n] for n in range(n_features)]
    v = [0] * nlabels_target
    root = fuzzy_rule.FuzzyRule(root_ant + [v])
    selected_rules = pd.DataFrame(columns=['binary_rule', 'fwracc'])
    rules_explored = 0
    widgets = [
        progressbar.Percentage(), ' ',
        progressbar.Counter(format='(%(value)02d/%(max_value)d rules considered)'),
        progressbar.Bar(),
        progressbar.Timer(), '~',
        progressbar.ETA(),
    ]
    with progressbar.ProgressBar(max_value=number_rules, widgets=widgets) as bar:
        selected_rules, rules_explored = refine_rule(root, -1, selected_rules, rules_explored)

    if len(selected_rules['binary_rule']) == 0:
        print('FAIL')
        return [], [0, 0, 0, 0, -1]

    selected_rules = selected_rules.reset_index(drop=True)
    selected_rules = set_fuzzy_rules.SetFuzzyRules(selected_rules['binary_rule'])
    out = selected_rules.measures(I, fuzzy_dataset)

    print('| ------ SUMMARY ------ |')
    print('Mean Num_features: {}'.format(round(np.mean(out.num_features), 3)))
    print('Mean FCoverage: {}'.format(round(np.mean(out.fcoverage), 3)))
    print('Mean FSupport: {}'.format(round(np.mean(out.fsupport), 3)))
    print('Mean FConfidence: {}'.format(round(np.mean(out.fconfidence), 3)))
    print('Mean FWRAcc: {}'.format(round(np.mean(out.fwracc), 3)))

    return selected_rules, out
