import copy
from typing import Dict, List, Tuple, Iterable
import numpy as np

from FIRM.base.ct_fuzzy_rule import CRFuzzyRule
from FIRM.base.ct_fuzzy_antecedent import CRFuzzyAntecedent
from FIRM.base.ct_set_fuzzy_rules import SetFuzzyRules


# ---------- helpers --------------------------------------------------------

def compatible_con(ant: Iterable[Tuple[int,int]], con: Tuple[int,int]) -> bool:
    """A consequent must refer to a feature not already used in antecedent."""
    ant_features = {i for (i, _) in ant}
    return con[0] not in ant_features

def _vectorize_binary(f):
    def vf(a, b):
        try:
            return f(a, b)
        except Exception:
            return np.vectorize(f, otypes=[float])(a, b)
    return vf


# ---------- 1-itemsets & cache --------------------------------------------

def generate_fuzzy_1itemsets(fuzzy_dataset) -> List[frozenset]:
    """
    Each item is a (feature_idx, ling_idx) tuple. Itemsets are frozensets of items.
    """
    out = []
    for f_idx, fv in enumerate(fuzzy_dataset.fv_list):
        for l_idx, _ in enumerate(fv.get_labels):
            out.append(frozenset({(f_idx, l_idx)}))
    return out

def _precompute_singleton_memberships(dataset, fuzzy_dataset) -> Dict[Tuple[int,int], np.ndarray]:
    """
    Cache μ(x) vectors for every singleton (f,l) once.
    """
    cache: Dict[Tuple[int,int], np.ndarray] = {}
    for f_idx, fv in enumerate(fuzzy_dataset.fv_list):
        col = dataset.iloc[:, f_idx].to_numpy()
        for l_idx, _ in enumerate(fv.get_labels):
            mu = fv.fs_list[l_idx]
            try:
                v = mu(col).astype(np.float32)
            except Exception:
                v = np.fromiter((mu(x) for x in col), count=len(col), dtype=np.float32)
            cache[(f_idx, l_idx)] = v
    return cache


# ---------- Frequent itemsets ---------------------------------------------

def frequent_itemsets(
    itemsets: List[frozenset],
    min_cov: float,
    singleton_mu: Dict[Tuple[int,int], np.ndarray],
    T,  # binary t-norm accepting arrays
) -> Tuple[List[frozenset], Dict[frozenset, np.ndarray]]:
    """
    Filter itemsets by fuzzy coverage and return their antecedent vectors.
    Returns:
      - list of frequent itemsets
      - dict mapping itemset -> antecedent vector (μ reduced by T)
    """
    vT = _vectorize_binary(T)
    ant_vecs: Dict[frozenset, np.ndarray] = {}
    freq: List[frozenset] = []

    for S in itemsets:
        items = sorted(S)  # deterministic order
        # reduce membership vectors via T
        vec = singleton_mu[items[0]]
        for it in items[1:]:
            vec = vT(vec, singleton_mu[it])

        cov = float(vec.mean())
        if cov >= min_cov:
            freq.append(S)
            ant_vecs[S] = vec.astype(np.float32, copy=False)

    return freq, ant_vecs


def generate_nitemsets(prev_level: List[frozenset]) -> List[frozenset]:
    """
    Standard Apriori join: join L_{k-1} with itself to make C_k.
    Also enforces no duplicate feature in an itemset.
    """
    Ck = set()
    L = [sorted(s) for s in prev_level]
    n = len(L)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = L[i], L[j]
            # join if first k-2 items equal
            if a[:-1] == b[:-1]:
                cand = a[:-1] + sorted([a[-1], b[-1]])
                # enforce unique feature indices
                features = [fi for (fi, _) in cand]
                if len(features) == len(set(features)):
                    Ck.add(frozenset(cand))
    return list(Ck)


def eliminate_by_apriori_property(candidates: List[frozenset], prev_level_set: set) -> List[frozenset]:
    """
    Prune C_k: every (k-1)-subset must be in L_{k-1}.
    """
    pruned = []
    for c in candidates:
        k = len(c)
        ok = True
        for it in c:
            if frozenset(c - {it}) not in prev_level_set:
                ok = False
                break
        if ok:
            pruned.append(c)
    return pruned


# ---------- Main Apriori ---------------------------------------------------

def apriori(dataset, fuzzy_dataset, T, min_cov: float = 0.1, max_feat: int = 5):
    """
    Returns: (all_frequent_itemsets, antecedent_vectors)
      - all_frequent_itemsets: list[frozenset]
      - antecedent_vectors: dict[itemset -> np.ndarray] for quick reuse
    """
    singleton_mu = _precompute_singleton_memberships(dataset, fuzzy_dataset)

    all_freq: List[frozenset] = []
    ant_vectors: Dict[frozenset, np.ndarray] = {}

    # L1
    Lk, ant_k = frequent_itemsets(generate_fuzzy_1itemsets(fuzzy_dataset), min_cov, singleton_mu, T)
    all_freq.extend(Lk)
    ant_vectors.update(ant_k)

    k = 2
    while Lk and k <= max_feat:
        # Ck
        Ck = generate_nitemsets(Lk)
        Ck = eliminate_by_apriori_property(Ck, set(Lk))

        # evaluate Ck using cached singletons (reduce by T)
        Lk, ant_k = frequent_itemsets(Ck, min_cov, singleton_mu, T)
        all_freq.extend(Lk)
        ant_vectors.update(ant_k)
        k += 1

    return all_freq, ant_vectors


# ---------- Rule generation (AARFI) ---------------------------------------

def AARFI(dataset, fuzzy_dataset, T, I, min_cov=0.3, min_supp=0.3, min_conf=0.8, max_feat=5):
    """
    Generates association rules with fuzzy antecedents and singleton consequents.
    Uses cached antecedent vectors from Apriori to avoid re-evaluation.
    """
    vT = _vectorize_binary(T)
    vI = _vectorize_binary(I)

    # Mine antecedent candidates and get their vectors
    ant_candidates, ant_vecs = apriori(dataset, fuzzy_dataset, T, min_cov, max_feat)

    # Precompute all singleton consequents' vectors once
    singleton_mu = _precompute_singleton_memberships(dataset, fuzzy_dataset)
    con_candidates = [next(iter(s)) for s in generate_fuzzy_1itemsets(fuzzy_dataset)]  # list of (f,l)

    rules = []
    for ant in ant_candidates:
        ant_vec = ant_vecs[ant]
        sum_ant = float(ant_vec.sum())
        if sum_ant == 0.0:
            continue

        ant_features = {fi for (fi, _) in ant}

        for con in con_candidates:
            if con[0] in ant_features:
                continue  # respect compatibility

            con_vec = singleton_mu[con]
            implied = vI(ant_vec, con_vec)
            eval_vec = vT(ant_vec, implied)

            fsupport = float(eval_vec.mean())
            if fsupport < min_supp:
                continue

            fconfidence = float(eval_vec.sum() / sum_ant)
            if fconfidence < min_conf:
                continue

            # Build the full rule object (vectorized evaluate to fill fields if you prefer)
            lrule = sorted(ant) + [con]
            rule = CRFuzzyRule(lrule)
            # Use CRFuzzyRule's vectorized evaluation with membership cache to avoid recomputation
            # If your CRFuzzyRule supports cache injection, pass it; else this call will re-evaluate.
            rule.evaluate_rule_database(dataset, fuzzy_dataset, T, I)
            rules.append(rule)

    return SetFuzzyRules(rules)


# ---------- Redundancy pruning --------------------------------------------

def redundancy_pruning(rules: SetFuzzyRules, epsilon: float = 0.05) -> SetFuzzyRules:
    """
    Remove rules whose antecedent is a superset of another rule's antecedent
    with the same consequent and nearly equal confidence.
    """
    by_con: Dict[Tuple[int,int], List[CRFuzzyRule]] = {}
    for r in rules.rule_list:
        con = r.lrule[-1]
        by_con.setdefault(con, []).append(r)

    kept: List[CRFuzzyRule] = []
    for con, group in by_con.items():
        # sort by antecedent size ascending to prefer keeping shorter rules
        group_sorted = sorted(group, key=lambda r: len(r.lrule) - 1)
        selected: List[CRFuzzyRule] = []
        for i, r1 in enumerate(group_sorted):
            A1 = set(r1.lrule[:-1])
            c1 = r1.fconfidence()
            redundant = False
            for r2 in selected:  # only compare to already kept (smaller) ones
                A2 = set(r2.lrule[:-1])
                if A2.issubset(A1) and abs(c1 - r2.fconfidence()) < epsilon:
                    redundant = True
                    break
            if not redundant:
                selected.append(r1)
        kept.extend(selected)

    return SetFuzzyRules(kept)
