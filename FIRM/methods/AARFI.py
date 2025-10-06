import copy
import logging
from typing import Dict, List, Tuple, Iterable, Optional, Callable
import numpy as np
import pandas as pd

from FIRM.base.ct_fuzzy_rule import CRFuzzyRule
from FIRM.base.ct_fuzzy_antecedent import CRFuzzyAntecedent
from FIRM.base.ct_set_fuzzy_rules import SetFuzzyRules


# ---------- logging helpers ------------------------------------------------

def _get_logger(logger: Optional[logging.Logger], verbose: bool) -> logging.Logger:
    """
    Return a logger. If none provided, create a module-level logger.
    When verbose=True and the returned logger has no handlers, attach a basic handler.
    """
    lg = logger or logging.getLogger(__name__)
    if verbose and not lg.handlers:
        # Basic, non-duplicating handler; you can configure upstream as you prefer.
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)
        lg.propagate = False
    return lg

def _vlog(verbose: bool, lg: logging.Logger, msg: str, *args):
    if verbose:
        lg.info(msg, *args)

def _vdebug(verbose: bool, lg: logging.Logger, msg: str, *args):
    # occasionally you might want more detailed lines gated on verbose as well
    if verbose:
        lg.debug(msg, *args)


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

def generate_fuzzy_1itemsets(
    fuzzy_dataset,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[frozenset]:
    """
    Each item is a (feature_idx, ling_idx) tuple. Itemsets are frozensets of items.
    """
    lg = _get_logger(logger, verbose)
    out = []
    total_labels = 0
    for f_idx, fv in enumerate(fuzzy_dataset.fv_list):
        n_labels = len(fv.get_labels)
        total_labels += n_labels
        for l_idx, _ in enumerate(fv.get_labels):
            out.append(frozenset({(f_idx, l_idx)}))
    _vlog(verbose, lg, "Generated %d fuzzy 1-itemsets across %d features (%d total labels).",
          len(out), len(fuzzy_dataset.fv_list), total_labels)
    return out

def _precompute_singleton_memberships(
    dataset,
    fuzzy_dataset,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Dict[Tuple[int,int], np.ndarray]:
    """
    Cache μ(x) vectors for every singleton (f,l) once.
    """
    lg = _get_logger(logger, verbose)
    cache: Dict[Tuple[int,int], np.ndarray] = {}
    n_rows = len(dataset)
    _vlog(verbose, lg, "Precomputing singleton memberships for %d rows and %d features...",
          n_rows, len(fuzzy_dataset.fv_list))

    for f_idx, fv in enumerate(fuzzy_dataset.fv_list):
        col = dataset.iloc[:, f_idx].to_numpy()
        for l_idx, _ in enumerate(fv.get_labels):
            mu = fv.fs_list[l_idx]
            try:
                v = mu(col).astype(np.float32)
            except Exception:
                v = np.fromiter((mu(x) for x in col), count=len(col), dtype=np.float32)
            cache[(f_idx, l_idx)] = v
    _vlog(verbose, lg, "Cached %d singleton membership vectors.", len(cache))
    return cache


# ---------- Frequent itemsets ---------------------------------------------

def frequent_itemsets(
    itemsets: List[frozenset],
    min_cov: float,
    singleton_mu: Dict[Tuple[int,int], np.ndarray],
    T,  # binary t-norm accepting arrays
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[frozenset], Dict[frozenset, np.ndarray]]:
    """
    Filter itemsets by fuzzy coverage and return their antecedent vectors.
    Returns:
      - list of frequent itemsets
      - dict mapping itemset -> antecedent vector (μ reduced by T)
    """
    lg = _get_logger(logger, verbose)
    vT = _vectorize_binary(T)
    ant_vecs: Dict[frozenset, np.ndarray] = {}
    freq: List[frozenset] = []

    _vlog(verbose, lg, "Evaluating %d candidate itemsets (min_cov=%.4f)...",
          len(itemsets), min_cov)

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

    _vlog(verbose, lg, "Selected %d frequent itemsets (%.1f%% pass rate).",
          len(freq), 100.0 * len(freq) / max(1, len(itemsets)))
    return freq, ant_vecs


def generate_nitemsets(
    prev_level: List[frozenset],
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[frozenset]:
    """
    Standard Apriori join: join L_{k-1} with itself to make C_k.
    Also enforces no duplicate feature in an itemset.
    """
    lg = _get_logger(logger, verbose)
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
    out = list(Ck)
    _vlog(verbose, lg, "Joined %d -> %d candidates for next level.", len(prev_level), len(out))
    return out


def eliminate_by_apriori_property(
    candidates: List[frozenset],
    prev_level_set: set,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> List[frozenset]:
    """
    Prune C_k: every (k-1)-subset must be in L_{k-1}.
    """
    lg = _get_logger(logger, verbose)
    pruned = []
    dropped = 0
    for c in candidates:
        ok = True
        for it in c:
            if frozenset(c - {it}) not in prev_level_set:
                ok = False
                break
        if ok:
            pruned.append(c)
        else:
            dropped += 1
    _vlog(verbose, lg, "Apriori pruning: kept %d / %d (dropped %d).",
          len(pruned), len(candidates), dropped)
    return pruned


# ---------- Main Apriori ---------------------------------------------------

def apriori(
    dataset,
    fuzzy_dataset,
    T,
    min_cov: float = 0.1,
    max_feat: int = 5,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    Returns: (all_frequent_itemsets, antecedent_vectors)
      - all_frequent_itemsets: list[frozenset]
      - antecedent_vectors: dict[itemset -> np.ndarray] for quick reuse
    """
    lg = _get_logger(logger, verbose)
    _vlog(verbose, lg, "Starting Apriori (min_cov=%.4f, max_feat=%d)...", min_cov, max_feat)

    singleton_mu = _precompute_singleton_memberships(
        dataset, fuzzy_dataset, verbose=verbose, logger=lg
    )

    all_freq: List[frozenset] = []
    ant_vectors: Dict[frozenset, np.ndarray] = {}

    # L1
    L1_candidates = generate_fuzzy_1itemsets(fuzzy_dataset, verbose=verbose, logger=lg)
    Lk, ant_k = frequent_itemsets(L1_candidates, min_cov, singleton_mu, T, verbose=verbose, logger=lg)
    all_freq.extend(Lk)
    ant_vectors.update(ant_k)
    _vlog(verbose, lg, "Level-1: %d frequent itemsets.", len(Lk))

    k = 2
    while Lk and k <= max_feat:
        _vlog(verbose, lg, "---- Level %d ----", k)
        # Ck
        Ck = generate_nitemsets(Lk, verbose=verbose, logger=lg)
        Ck = eliminate_by_apriori_property(Ck, set(Lk), verbose=verbose, logger=lg)

        # evaluate Ck using cached singletons (reduce by T)
        Lk, ant_k = frequent_itemsets(Ck, min_cov, singleton_mu, T, verbose=verbose, logger=lg)
        _vlog(verbose, lg, "Level-%d frequent itemsets: %d", k, len(Lk))
        all_freq.extend(Lk)
        ant_vectors.update(ant_k)
        k += 1

    _vlog(verbose, lg, "Apriori finished: total frequent itemsets = %d", len(all_freq))
    return all_freq, ant_vectors


# ---------- Rule generation (AARFI) ---------------------------------------

def AARFI(
    dataset,
    fuzzy_dataset,
    T,
    I,
    min_cov=0.3,
    min_supp=0.3,
    min_conf=0.8,
    max_feat=5,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
):
    """
    Generates association rules with fuzzy antecedents and singleton consequents.
    Uses cached antecedent vectors from Apriori to avoid re-evaluation.
    """
    lg = _get_logger(logger, verbose)
    vT = _vectorize_binary(T)
    vI = _vectorize_binary(I)

    _vlog(verbose, lg,
          "AARFI start (min_cov=%.3f, min_supp=%.3f, min_conf=%.3f, max_feat=%d)...",
          min_cov, min_supp, min_conf, max_feat)

    # Mine antecedent candidates and get their vectors
    ant_candidates, ant_vecs = apriori(
        dataset, fuzzy_dataset, T, min_cov, max_feat, verbose=verbose, logger=lg
    )
    _vlog(verbose, lg, "Antecedent candidates: %d", len(ant_candidates))

    # Precompute all singleton consequents' vectors once
    singleton_mu = _precompute_singleton_memberships(
        dataset, fuzzy_dataset, verbose=verbose, logger=lg
    )
    con_candidates = [next(iter(s)) for s in generate_fuzzy_1itemsets(
        fuzzy_dataset, verbose=verbose, logger=lg
    )]  # list of (f,l)
    _vlog(verbose, lg, "Consequent candidates (singletons): %d", len(con_candidates))

    rules = []
    examined = 0
    kept = 0
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
                examined += 1
                continue

            fconfidence = float(eval_vec.sum() / sum_ant)
            if fconfidence < min_conf:
                examined += 1
                continue

            # Build the full rule object (vectorized evaluate to fill fields if you prefer)
            lrule = sorted(ant) + [con]
            rule = CRFuzzyRule(lrule)
            # Use CRFuzzyRule's vectorized evaluation with membership cache to avoid recomputation
            # If your CRFuzzyRule supports cache injection, pass it; else this call will re-evaluate.
            rule.evaluate_rule_database(dataset, fuzzy_dataset, T, I)
            rules.append(rule)
            kept += 1
            examined += 1

    _vlog(verbose, lg, "AARFI finished. Rules kept: %d (from %d evaluated pairs).", kept, examined)
    return SetFuzzyRules(rules)


# ---------- Redundancy pruning --------------------------------------------

def redundancy_pruning(
    rules: SetFuzzyRules,
    epsilon: float = 0.05,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> SetFuzzyRules:
    """
    Remove rules whose antecedent is a superset of another rule's antecedent
    with the same consequent and nearly equal confidence.
    """
    lg = _get_logger(logger, verbose)
    _vlog(verbose, lg, "Redundancy pruning (epsilon=%.4f) over %d rules...", epsilon, len(rules.rule_list))

    by_con: Dict[Tuple[int,int], List[CRFuzzyRule]] = {}
    for r in rules.rule_list:
        con = r.lrule[-1]
        by_con.setdefault(con, []).append(r)

    kept: List[CRFuzzyRule] = []
    dropped = 0
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
            else:
                dropped += 1
        kept.extend(selected)

    _vlog(verbose, lg, "Pruning complete: kept %d, dropped %d.", len(kept), dropped)
    return SetFuzzyRules(kept)


def AARFI_F(
    dataset,
    fuzzy_dataset,
    T,                      # t-norm used for Apriori antecedent mining only
    I,                      # kept for API symmetry; not used here unless you want it
    F: Callable[[np.ndarray, np.ndarray], np.ndarray],  # vectorized: F(a, c) -> eval; must broadcast
    *,
    min_cov: float = 0.3,   # Apriori coverage (for antecedents)
    min_supp: float = 0.3,
    min_conf: float = 0.8,
    max_feat: int = 5,
    chunk_rows: int = 100_000,  # set 0/None to process all rows at once
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[SetFuzzyRules, pd.DataFrame]:
    """
    Mine rules using a general vectorized F(x,y)=T(x,I(x,y)), and return:
      - SetFuzzyRules (un-evaluated rule objects)
      - pandas DataFrame with columns: ['rule','coverage','support','confidence','lrule','n_antecedents']

    This does NOT call `CRFuzzyRule.evaluate_rule_database(...)`.
    The metrics shown in the DataFrame are the precomputed ones from F.
    """
    lg = _get_logger(logger, verbose)
    _vlog(verbose, lg,
          "AARFI_F (min_cov=%.3f, min_supp=%.3f, min_conf=%.3f, max_feat=%d)",
          min_cov, min_supp, min_conf, max_feat)

    # ---- 1) Mine antecedents and reuse cached vectors from Apriori ----
    ant_list, ant_vecs = apriori(
        dataset, fuzzy_dataset, T, min_cov, max_feat, verbose=verbose, logger=lg
    )
    if not ant_list:
        return SetFuzzyRules([]), pd.DataFrame(columns=["rule","coverage","support","confidence","lrule","n_antecedents"])

    # ---- 2) Stack all singleton consequents once: C shape (N, Cn) ----
    singleton_mu = _precompute_singleton_memberships(
        dataset, fuzzy_dataset, verbose=verbose, logger=lg
    )
    con_items = [next(iter(s)) for s in generate_fuzzy_1itemsets(
        fuzzy_dataset, verbose=verbose, logger=lg
    )]
    C = np.stack([singleton_mu[c] for c in con_items], axis=1).astype(np.float32, copy=False)
    N, Cn = C.shape
    con_feat_idx = np.array([c[0] for c in con_items], dtype=np.int32)

    if not chunk_rows or chunk_rows <= 0:
        chunk_rows = N

    rows = []
    rule_objs: List[CRFuzzyRule] = []
    examined_pairs = 0
    kept_pairs = 0

    for ant in ant_list:
        a = ant_vecs[ant].astype(np.float32, copy=False)  # (N,)
        sum_a = float(a.sum()) + 1e-12
        coverage_a = float(a.mean())

        # Feature-compatibility mask (structural)
        ant_features = {fi for (fi, _) in ant}
        mask_feat = ~np.isin(con_feat_idx, list(ant_features))  # (Cn,)
        if not np.any(mask_feat):
            continue

        # Accumulate numerators: n_j = sum_i F(a_i, C_ij), chunked over rows to save RAM
        numerators = np.zeros(Cn, dtype=np.float64)
        for s in range(0, N, chunk_rows):
            e = min(s + chunk_rows, N)
            a_blk = a[s:e][:, None]       # (B, 1)
            C_blk = C[s:e, :]             # (B, Cn)
            blk = F(a_blk, C_blk)         # must broadcast -> (B, Cn)
            numerators += blk.sum(axis=0, dtype=np.float64)

        fsupport_all = (numerators / float(N)).astype(np.float32, copy=False)  # (Cn,)
        fconf_all    = (numerators / sum_a).astype(np.float32, copy=False)     # (Cn,)

        ok = mask_feat & (fsupport_all >= min_supp) & (fconf_all >= min_conf)
        examined_pairs += int(mask_feat.sum())
        if not np.any(ok):
            continue

        idxs = np.nonzero(ok)[0]
        # Compute evaluation columns only for kept consequents (small K) for display correctness (optional)
        # If you don't need per-row eval vectors, this step is unnecessary; we already have the metrics.
        # evalK = F(a[:, None], C[:, idxs])  # (N, K)  <-- omit to save RAM/time

        for j in idxs.tolist():
            lrule = sorted(ant) + [con_items[j]]
            r = CRFuzzyRule(lrule)  # no evaluation; we only need the string
            rule_str = r.sentence_rule(fuzzy_dataset)

            rows.append({
                "rule": rule_str,
                "coverage": coverage_a,
                "support": float(fsupport_all[j]),
                "confidence": float(fconf_all[j]),
                "lrule": tuple(lrule),
                "n_antecedents": r.get_num_features(),
            })
            rule_objs.append(r)

        kept_pairs += len(idxs)

    _vlog(verbose, lg, "AARFI_F finished. Rules kept: %d (from %d examined pairs).",
          kept_pairs, examined_pairs)

    # Build DataFrame and sort for nicer presentation
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["confidence","support","coverage"], ascending=False, kind="mergesort").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["rule","coverage","support","confidence","lrule","n_antecedents"])

    return SetFuzzyRules(rule_objs), df



