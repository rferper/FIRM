import numpy as np
import pandas as pd
import FIRM.base.set_fuzzy_rules as set_fuzzy_rules


# ----- INPUT ----- #
# Data: DataFrame with input Data
# FuzzyData: DataFrame with all the fuzzy linguistic variables
# set_rules: Set of rules to postprocess
# I: Fuzzy implication Function
# T: t-norm
# K: Number of output rules
# w: Parameter for computing the overall coverage


def WCSDFI(dataset, fuzzy_dataset, set_rules, T, I, N, K=10):
    n_examples = len(dataset)
    n_features = len(fuzzy_dataset.fv_list) - 1
    nlabels_features = [0] * n_features
    for i in range(n_features):
        nlabels_features[i] = len(fuzzy_dataset.fv_list[i].get_labels)
    weights_old = np.array([0] * n_examples, dtype=np.float64)
    weights_new = np.array([1] * n_examples, dtype=np.float64)
    explored_rules = pd.DataFrame(columns=['binary_rule', 'fwracc'])
    explored_rules['binary_rule'] = set_rules.rule_list
    explored_rules['fwracc'] = explored_rules['binary_rule'].apply(lambda x: x.fwracc_weights(T, weights_new))
    selected_rules = []
    while len(selected_rules) < K and not (weights_old == weights_new).all():
        i_new = explored_rules['fwracc'].argmax()
        new_rule = explored_rules['binary_rule'].iloc[i_new]
        selected_rules.append(new_rule)
        # drop selected rule
        explored_rules = explored_rules.drop(index=explored_rules.iloc[i_new].name)
        # modify weights
        weights_old = weights_new
        weights_new = np.array(list(map(lambda x, y: T(x, N(y)), weights_old, new_rule.antecedents)))
        explored_rules['fwracc'] = explored_rules['binary_rule'].apply(lambda x: x.fwracc_weights(T, weights_new))

    selected_rules = set_fuzzy_rules.SetFuzzyRules(selected_rules)
    out = selected_rules.measures(I, fuzzy_dataset)
    out = out.sort_values(by=['fwracc'], ascending=False, ignore_index=True)

    print('| ------ SUMMARY ------ |')
    print('Mean Num_features: {}'.format(round(np.mean(out.num_features), 3)))
    print('Mean FCoverage: {}'.format(round(np.mean(out.fcoverage), 3)))
    print('Mean FSupport: {}'.format(round(np.mean(out.fsupport), 3)))
    print('Mean FConfidence: {}'.format(round(np.mean(out.fconfidence), 3)))
    print('Mean FWRAcc: {}'.format(round(np.mean(out.fwracc), 3)))

    return selected_rules, out