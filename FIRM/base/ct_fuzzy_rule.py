import numpy as np
import pandas as pd
from typing import List, Tuple, Callable, Dict, Optional

# -------- helpers ----------------------------------------------------------

def _vectorize_binary(f: Callable) -> Callable:
    def vf(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        try:
            return f(a, b)
        except Exception:
            return np.vectorize(f, otypes=[float])(a, b)  # elementwise
    return vf

def _reduce_tnorm(arrays: List[np.ndarray], vT: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
    if not arrays:
        return np.ones(0, dtype=np.float32)
    out = arrays[0]
    for a in arrays[1:]:
        out = vT(out, a)
    return out

# -------- main class -------------------------------------------------------

class CRFuzzyRule(object):
    """
    Fuzzy rule as a list of (feature_index, linguistic_index) tuples.
    The last tuple is the consequent; others are antecedents.
    Requires user-provided T and I.
    """

    __slots__ = ("lrule", "antecedents", "consequents", "evaluations", "evaluated")

    def __init__(self, lrule: List[Tuple[int, int]]):
        self.lrule = lrule
        self.antecedents = np.array([], dtype=np.float32)
        self.consequents = np.array([], dtype=np.float32)
        self.evaluations = np.array([], dtype=np.float32)
        self.evaluated = 0

    def __repr__(self):
        return f"<Fuzzy Rule: {self.lrule}>"

    def get_num_features(self) -> int:
        return len(self.lrule) - 1

    @staticmethod
    def _get_membership_vector(
        data: pd.DataFrame,
        feature_idx: int,
        ling_idx: int,
        fuzzy_data,
        cache: Dict[Tuple[int, int], np.ndarray],
        dtype=np.float32,
        chunk_size: Optional[int] = None,
    ) -> np.ndarray:
        key = (feature_idx, ling_idx)
        if key in cache:
            return cache[key]

        col = data.iloc[:, feature_idx].to_numpy()
        mu = fuzzy_data.fv_list[feature_idx].fs_list[ling_idx]

        try:
            vals = mu(col).astype(dtype)
        except Exception:
            if chunk_size is None:
                chunk_size = 100_000 if col.shape[0] > 1_000_000 else 50_000
            out = np.empty(col.shape[0], dtype=dtype)
            for s in range(0, col.shape[0], chunk_size):
                e = min(s + chunk_size, col.shape[0])
                try:
                    out[s:e] = mu(col[s:e])  # try vector input per chunk
                except Exception:
                    out[s:e] = np.fromiter((mu(v) for v in col[s:e]), count=e - s, dtype=dtype)
            vals = out

        cache[key] = vals
        return vals

    def evaluate_rule_database(
        self,
        data: pd.DataFrame,
        fuzzy_data,
        T: Callable[[np.ndarray, np.ndarray], np.ndarray],
        I: Callable[[np.ndarray, np.ndarray], np.ndarray],
        *,
        cache: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
        dtype=np.float32,
    ):
        """
        Vectorized evaluation for all rows using REQUIRED T and I.
        Stores:
          - self.antecedents = T-reduced antecedent memberships
          - self.consequents = consequent memberships
          - self.evaluations = T(antecedents, I(antecedents, consequents))
        """
        self.evaluated = 1
        if cache is None:
            cache = {}

        vT = _vectorize_binary(T)
        vI = _vectorize_binary(I)

        # Antecedent vectors
        ant_pairs = self.lrule[:-1]
        ant_vecs = [
            self._get_membership_vector(data, fi, li, fuzzy_data, cache, dtype=dtype)
            for (fi, li) in ant_pairs
        ]
        antecedents = _reduce_tnorm(ant_vecs, vT) if ant_vecs else np.ones(len(data), dtype=dtype)

        # Consequent vector
        con_f, con_l = self.lrule[-1]
        consequents = self._get_membership_vector(data, con_f, con_l, fuzzy_data, cache, dtype=dtype)

        # Implication + final evaluation
        implied = vI(antecedents, consequents)
        evaluations = vT(antecedents, implied)

        # store
        self.antecedents = antecedents.astype(dtype, copy=False)
        self.consequents = consequents.astype(dtype, copy=False)
        self.evaluations = evaluations.astype(dtype, copy=False)
        return self

    # ---- metrics (vectorized) --------------------------------------------

    def fcoverage(self) -> float:
        return float(self.antecedents.mean()) if self.antecedents.size else 0.0

    def fsupport(self) -> float:
        return float(self.evaluations.mean()) if self.evaluations.size else 0.0

    def fconfidence(self) -> float:
        s_ant = float(self.antecedents.sum())
        return float(self.evaluations.sum() / s_ant) if s_ant > 0.0 else 0.0

    def fwracc(self) -> float:
        n = self.antecedents.size
        if n == 0:
            return 0.0
        return float((self.evaluations.sum() - self.antecedents.sum() * self.consequents.sum() / n) / n)

    # ---- pretty printer ---------------------------------------------------

    def sentence_rule(self, fuzzy_data) -> str:
        features = fuzzy_data.fv_list
        ant = self.lrule[:-1]
        con = self.lrule[-1]
        parts = ["IF ("]
        for i, (fi, li) in enumerate(ant):
            if i > 0:
                parts.append("AND")
            parts += [features[fi].name, "IS", str(features[fi].get_labels[li])]
        parts += [")", "THEN", features[con[0]].name, "IS", str(features[con[0]].get_labels[con[1]])]
        return " ".join(parts)

