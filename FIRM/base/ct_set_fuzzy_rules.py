import numpy as np
import pandas as pd
from typing import Callable, Iterable, List, Tuple

import FIRM.base.operator_power as operator_power  # if you still need it elsewhere
import FIRM.base.ct_fuzzy_rule as fuzzy_rule


def N(x: np.ndarray, w: float):
    """
    Parametric negation used in your S-norm construction.
    Vectorized for numpy arrays; falls back to scalar division rules automatically.
    """
    # (1 - x) / (1 - w*x). Safe where denominator==0 -> 1 (your original intent)
    denom = 1.0 - w * x
    out = (1.0 - x) / denom
    # handle denom==0 -> 1
    if isinstance(out, np.ndarray):
        mask = (denom == 0)
        if np.any(mask):
            out = out.copy()
            out[mask] = 1.0
        return out.astype(np.float32, copy=False)
    # scalar
    return 1.0 if denom == 0 else float(out)


class SetFuzzyRules(object):
    """
    Collection of fuzzy rules.
    """

    def __init__(self, rule_list: List[fuzzy_rule.CRFuzzyRule] | None = None):
        self.rule_list: List[fuzzy_rule.CRFuzzyRule] = rule_list or []

    # ------------------ Quality measures table ------------------

    def measures(self, fuzzy_dataset) -> pd.DataFrame:
        """
        Build a dataframe of rule-level metrics efficiently.
        Assumes each rule already called .evaluate_rule_database(...).
        """
        if not self.rule_list:
            return pd.DataFrame(columns=['sentence_rule','num_features','fcoverage','fsupport','fconfidence','fwracc'])

        rows = []
        for r in self.rule_list:
            rows.append({
                'sentence_rule': r.sentence_rule(fuzzy_dataset),
                'num_features':  r.get_num_features(),
                'fcoverage':     r.fcoverage(),
                'fsupport':      r.fsupport(),
                'fconfidence':   r.fconfidence(),
                'fwracc':        r.fwracc(),
            })
        df = pd.DataFrame.from_records(rows)
        return df.sort_values(by='fconfidence', ascending=False, kind='mergesort').reset_index(drop=True)

    # ------------------ Jaccard similarity ------------------

    def jaccard_similarity(self, other: "SetFuzzyRules") -> float:
        """
        Jaccard over sets of tuple-encoded rules.
        """
        set1 = {tuple(r.lrule) for r in self.rule_list}
        set2 = {tuple(r.lrule) for r in other.rule_list}
        if not set1 and not set2:
            return 0.0
        inter = len(set1 & set2)
        union = len(set1 | set2)
        return inter / union if union else 0.0

    # ------------------ Overall coverage (vectorized) ------------------

    def overall_coverage(self, T: Callable[[np.ndarray, np.ndarray], np.ndarray], w: float) -> float:
        """
        Compute overall coverage across all rules and all examples using:
            S(x, y) = N( T( N(x), N(y) ), w )
        Then aggregate all rule antecedents per example with S as the n-ary S-norm,
        and finally average over examples.
        """
        if not self.rule_list:
            print('FOverallCoverage: 0.0')
            return 0.0

        # Ensure rules are evaluated and consistent in length
        n_examples = None
        for r in self.rule_list:
            if r.antecedents is None or r.antecedents.size == 0:
                raise ValueError("All rules must be evaluated before calling overall_coverage().")
            if n_examples is None:
                n_examples = r.antecedents.size
            elif r.antecedents.size != n_examples:
                raise ValueError("All rules must have antecedents of equal length.")

        # Stack antecedent memberships: shape = (n_rules, n_examples)
        A = np.vstack([r.antecedents for r in self.rule_list]).astype(np.float32, copy=False)

        # Vectorized S-norm derived from T via negation N with parameter w:
        # S(x, y) = N( T( N(x), N(y) ), w )
        def S_arrays(x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return N(T(N(x, w), N(y, w)), w)

        acc = A[0]
        for i in range(1, A.shape[0]):
            acc = S_arrays(acc, A[i])

        # acc now = per-example overall coverage; mean over examples:
        foverallcoverage = float(acc.mean())
        print(f'FOverallCoverage: {foverallcoverage:.3f}')
        return foverallcoverage
