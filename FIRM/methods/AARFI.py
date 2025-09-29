import copy
from FIRM.base.ct_fuzzy_rule import CRFuzzyRule
from FIRM.base.ct_fuzzy_antecedent import CRFuzzyAntecedent
from FIRM.base.ct_set_fuzzy_rules import SetFuzzyRules
def compatible_con(list_of_tuples, new_tuple):
    """
    Checks if a tuple is compatible to be the consequent of an antecedent of a rule.
    """
    x, y = new_tuple  # Unpack the new tuple
    # Check if x is already present as the first component in any tuple in the list
    if x not in [a for a, b in list_of_tuples]:
        return True
    else:
        return False

# Generate fuzzy 1-itemsets
def generate_fuzzy_1itemsets(fuzzy_dataset):
    """
    Generate fuzzy itemsets by combining items with linguistic variables.
    """
    itemsets = []
    for item in range(len(fuzzy_dataset.fv_list)):
        for lv in range(len(fuzzy_dataset.fv_list[item].get_labels)):
            itemsets.append([(item,lv)])
    return itemsets

def frequent_itemsets(itemsets,min_cov,dataset,fuzzy_dataset,T):
    """
    Given a list of itemsets, return only the frequent ones w.r.t a minimum coverage.
    """
    freq_itemsets = []
    for itemset in itemsets:
        antecedent = CRFuzzyAntecedent(itemset)
        antecedent = antecedent.evaluate_rule_database(dataset,fuzzy_dataset,T)
        if antecedent.fcoverage() >= min_cov:
            freq_itemsets.append(itemset)
    return freq_itemsets

def generate_nitemsets(frequent_n_1_itemsets):
    """
    Generate n-itemsets from (n-1)-itemsets.
    """
    n = len(frequent_n_1_itemsets)
    candidate_nitemsets = []

    # Join Step: Combine pairs of (n-1)-itemsets
    for i in range(n):
        for j in range(i + 1, n):
            itemset1 = frequent_n_1_itemsets[i]
            itemset2 = frequent_n_1_itemsets[j]

            # Check if the first (n-2) items are the same
            if itemset1[:-1] == itemset2[:-1]:
                candidate = sorted(itemset1 + [itemset2[-1]])
                candidate_nitemsets.append(candidate)

    return candidate_nitemsets

def eliminate_candidates(candidates):
    """
    Eliminates candidates that contain tuples of the type (x, y) and (x, z),
    where y != z (same first component but different second components).
    """
    valid_candidates = []

    for candidate in candidates:
        # Create a dictionary to track the first component (x) and its corresponding second component (y)
        first_component_map = {}
        is_valid = True

        for x, y in candidate:
            if x in first_component_map:
                # If the first component (x) already exists, check if the second component (y) is the same
                if first_component_map[x] != y:
                    is_valid = False  # Invalid candidate, as it has (x, y) and (x, z) where y != z
                    break
            else:
                # Store the first component and its corresponding second component
                first_component_map[x] = y

        if is_valid:
            valid_candidates.append(candidate)

    return valid_candidates

def apriori(dataset, fuzzy_dataset, T, min_cov = 0.1, max_feat = 5):
    """
    Generalized Apriori algorithm for the given fuzzy dataset.
    Returns all frequent itemsets (from 1-itemsets to n-itemsets) as a list of lists.
    """
    all_frequent_itemsets = []

    # Step 1: Generate frequent 1-itemsets
    freq_itemsets_1 = frequent_itemsets(generate_fuzzy_1itemsets(fuzzy_dataset),min_cov,dataset,fuzzy_dataset,T)
    all_frequent_itemsets.extend(freq_itemsets_1)

    # Step 2: Iteratively generate frequent n-itemsets
    n = 2
    freq_itemsets_prev = freq_itemsets_1
    while freq_itemsets_prev and n<=max_feat:
        freq_itemsets_n = frequent_itemsets(eliminate_candidates(generate_nitemsets(freq_itemsets_prev)),min_cov,dataset,fuzzy_dataset,T)
        if not freq_itemsets_n:
            break  # Stop if no more frequent itemsets are found
        all_frequent_itemsets.extend(freq_itemsets_n)
        freq_itemsets_prev = freq_itemsets_n
        n += 1
    return all_frequent_itemsets

def AARFI(dataset, fuzzy_dataset, T, I, min_cov=0.3, min_supp=0.3, min_conf=0.8, max_feat=5):
    rules = []
    ant_candidates = apriori(dataset, fuzzy_dataset, T, min_cov, max_feat)
    con_candidates = generate_fuzzy_1itemsets(fuzzy_dataset)

    for con in con_candidates:
        for ant in ant_candidates:
            if compatible_con(ant, con[0]):
                # make deep copies to avoid shared mutation
                ant_copy = copy.deepcopy(ant)
                con_copy = copy.deepcopy(con)
                rule = CRFuzzyRule(ant_copy + con_copy)
                rule = rule.evaluate_rule_database(dataset, fuzzy_dataset, T, I)
                if rule.fsupport() >= min_supp and rule.fconfidence() >= min_conf:
                    rules.append(rule)
    return SetFuzzyRules(rules)

def redundacy_prunning(rules, epsilon = 0.05):
    rules =  rules.rule_list
    result = []
    for i, rule1 in enumerate(rules):
        is_subset = False
        for j, rule2 in enumerate(rules):
            if i != j and rule2.lrule[-1] == rule1.lrule[-1] and set(rule2.lrule[:-1]).issubset(
                    set(rule1.lrule[:-1])) and abs(rule1.fconfidence() - rule2.fconfidence()) < epsilon:
                is_subset = True
                break
        if not is_subset:
            result.append(rule1)

    return redundacy_prunning(SetFuzzyRules(result))