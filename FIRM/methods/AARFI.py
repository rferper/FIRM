import copy
import logging
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Optional, Callable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

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
) -> Tuple[List[frozenset], Dict[frozenset, Tuple[frozenset, Tuple[int,int]]]]:
    """
    Standard Apriori join: join L_{k-1} with itself to make C_k.
    Also enforces no duplicate feature in an itemset.
    Returns candidates and an incremental construction map:
      candidate -> (parent_(k-1)-itemset, added_singleton_item)
    """
    lg = _get_logger(logger, verbose)
    Ck = set()
    parent_map: Dict[frozenset, Tuple[frozenset, Tuple[int,int]]] = {}
    L = [sorted(s) for s in prev_level]

    # Group itemsets by their shared prefix (first k-2 items).
    # Only itemsets within the same prefix group can join, so this avoids
    # the O(n^2) all-pairs scan — critical when |L_{k-1}| is large (e.g. online_news L2).
    prefix_groups: Dict[tuple, List[list]] = defaultdict(list)
    for items in L:
        prefix_groups[tuple(items[:-1])].append(items)

    for prefix, group in prefix_groups.items():
        ng = len(group)
        for i in range(ng):
            for j in range(i + 1, ng):
                a, b = group[i], group[j]
                cand = list(prefix) + sorted([a[-1], b[-1]])
                # enforce unique feature indices
                features = [fi for (fi, _) in cand]
                if len(features) == len(set(features)):
                    cand_fs = frozenset(cand)
                    if cand_fs not in Ck:
                        Ck.add(cand_fs)
                        # Incremental vector construction:
                        # cand = parent + singleton_added_item
                        parent_map[cand_fs] = (frozenset(a), b[-1])
    out = list(Ck)
    _vlog(verbose, lg, "Joined %d -> %d candidates for next level.", len(prev_level), len(out))
    return out, parent_map


def frequent_itemsets_incremental(
    candidates: List[frozenset],
    parent_map: Dict[frozenset, Tuple[frozenset, Tuple[int,int]]],
    min_cov: float,
    prev_ant_vecs: Dict[frozenset, np.ndarray],
    singleton_mu: Dict[Tuple[int,int], np.ndarray],
    T,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[frozenset], Dict[frozenset, np.ndarray]]:
    """
    Evaluate C_k incrementally from L_{k-1} vectors:
      vec(cand) = T(vec(parent), mu(added_item))
    """
    lg = _get_logger(logger, verbose)
    vT = _vectorize_binary(T)
    ant_vecs: Dict[frozenset, np.ndarray] = {}
    freq: List[frozenset] = []

    _vlog(verbose, lg, "Evaluating %d candidates incrementally (min_cov=%.4f)...",
          len(candidates), min_cov)

    for S in candidates:
        parent, add_item = parent_map[S]
        vec = vT(prev_ant_vecs[parent], singleton_mu[add_item])
        cov = float(vec.mean())
        if cov >= min_cov:
            freq.append(S)
            ant_vecs[S] = vec.astype(np.float32, copy=False)

    _vlog(verbose, lg, "Incremental selection: %d frequent itemsets (%.1f%% pass rate).",
          len(freq), 100.0 * len(freq) / max(1, len(candidates)))
    return freq, ant_vecs


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
    singleton_mu: Optional[Dict[Tuple[int,int], np.ndarray]] = None,
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

    if singleton_mu is None:
        singleton_mu = _precompute_singleton_memberships(
            dataset, fuzzy_dataset, verbose=verbose, logger=lg
        )
    else:
        _vlog(verbose, lg, "Using precomputed singleton memberships (%d vectors).", len(singleton_mu))

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
        Ck, parent_map = generate_nitemsets(Lk, verbose=verbose, logger=lg)
        Ck = eliminate_by_apriori_property(Ck, set(Lk), verbose=verbose, logger=lg)
        parent_map = {c: parent_map[c] for c in Ck}

        # Evaluate Ck incrementally from previous level vectors.
        prev_ant_k = ant_k
        Lk, ant_k = frequent_itemsets_incremental(
            Ck,
            parent_map,
            min_cov,
            prev_ant_k,
            singleton_mu,
            T,
            verbose=verbose,
            logger=lg,
        )
        _vlog(verbose, lg, "Level-%d frequent itemsets: %d", k, len(Lk))
        all_freq.extend(Lk)
        ant_vectors.update(ant_k)
        k += 1

    _vlog(verbose, lg, "Apriori finished: total frequent itemsets = %d", len(all_freq))
    return all_freq, ant_vectors


def beam_apriori(
    dataset,
    fuzzy_dataset,
    T,
    min_cov: float = 0.1,
    max_feat: int = 5,
    beam_width: int = 128,
    max_children_per_node: int = 32,
    singleton_mu: Optional[Dict[Tuple[int,int], np.ndarray]] = None,
    score_fn: Optional[Callable[[float, int], float]] = None,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[frozenset], Dict[frozenset, np.ndarray]]:
    """
    Beam-search heuristic over antecedent itemsets.

    Keeps only the top `beam_width` candidates per level using:
      score(cov, k) = cov + 0.01 * k  (default)
    where cov is fuzzy coverage and k is antecedent size.
    """
    lg = _get_logger(logger, verbose)
    vT = _vectorize_binary(T)
    if score_fn is None:
        score_fn = lambda cov, k: cov + 0.01 * float(k)

    if singleton_mu is None:
        singleton_mu = _precompute_singleton_memberships(
            dataset, fuzzy_dataset, verbose=verbose, logger=lg
        )

    one_items = [next(iter(s)) for s in generate_fuzzy_1itemsets(
        fuzzy_dataset, verbose=verbose, logger=lg
    )]

    # Level 1
    L1_candidates = [frozenset({it}) for it in one_items]
    L1, ant1 = frequent_itemsets(
        L1_candidates, min_cov, singleton_mu, T, verbose=verbose, logger=lg
    )
    if not L1:
        return [], {}

    scored_l1 = sorted(
        ((score_fn(float(ant1[s].mean()), 1), s) for s in L1),
        key=lambda x: x[0],
        reverse=True,
    )
    beam_level = [s for _, s in scored_l1[:max(1, beam_width)]]

    selected: List[frozenset] = list(beam_level)
    ant_vectors: Dict[frozenset, np.ndarray] = {s: ant1[s] for s in beam_level}
    _vlog(verbose, lg, "Beam level 1: kept %d itemsets.", len(beam_level))

    k = 2
    while beam_level and k <= max_feat:
        _vlog(verbose, lg, "---- Beam level %d ----", k)

        cand_vecs: Dict[frozenset, np.ndarray] = {}
        for parent in beam_level:
            parent_vec = ant_vectors[parent]
            parent_items = sorted(parent)
            parent_features = {fi for (fi, _) in parent_items}
            parent_last = parent_items[-1]

            child_count = 0
            for add_item in one_items:
                if add_item[0] in parent_features:
                    continue
                # Enforce lexicographic growth to avoid duplicates.
                if add_item <= parent_last:
                    continue
                child = frozenset(parent | {add_item})
                if child in cand_vecs:
                    continue
                vec = vT(parent_vec, singleton_mu[add_item]).astype(np.float32, copy=False)
                cov = float(vec.mean())
                if cov >= min_cov:
                    cand_vecs[child] = vec
                    child_count += 1
                    if max_children_per_node > 0 and child_count >= max_children_per_node:
                        break

        if not cand_vecs:
            break

        scored = sorted(
            ((score_fn(float(v.mean()), k), s) for s, v in cand_vecs.items()),
            key=lambda x: x[0],
            reverse=True,
        )
        beam_level = [s for _, s in scored[:max(1, beam_width)]]

        for s in beam_level:
            ant_vectors[s] = cand_vecs[s]
        selected.extend(beam_level)
        _vlog(verbose, lg, "Beam level %d: candidates=%d kept=%d.", k, len(cand_vecs), len(beam_level))
        k += 1

    _vlog(verbose, lg, "Beam Apriori finished: selected antecedents=%d", len(selected))
    return selected, ant_vectors


# ---------- Per-antecedent worker (shared by AARFI_F and ARFI_beam) -------

def _eval_antecedent(
    ant: frozenset,
    ant_vec: np.ndarray,
    C: np.ndarray,
    C_f: Optional[np.ndarray],
    con_feat_idx: np.ndarray,
    con_items: list,
    F,
    separable_F: bool,
    N: int,
    chunk_rows: int,
    min_supp: float,
    min_conf: float,
    fuzzy_dataset,
) -> List[dict]:
    """Evaluate one antecedent against all compatible consequents. Returns matching row dicts."""
    a = ant_vec
    sum_a = float(a.sum()) + 1e-12
    coverage_a = float(a.mean())

    ant_features = {fi for (fi, _) in ant}
    compat_idx = np.nonzero(~np.isin(con_feat_idx, list(ant_features)))[0]
    if compat_idx.size == 0:
        return []

    n_compat = compat_idx.size
    numerators_compat = np.zeros(n_compat, dtype=np.float64)

    if separable_F:
        C_f_compat = C_f[:, compat_idx]
        for s in range(0, N, chunk_rows):
            e = min(s + chunk_rows, N)
            numerators_compat += (a[s:e, None] * C_f_compat[s:e]).sum(axis=0, dtype=np.float64)
    else:
        C_compat = C[:, compat_idx]
        for s in range(0, N, chunk_rows):
            e = min(s + chunk_rows, N)
            numerators_compat += F(a[s:e, None], C_compat[s:e]).sum(axis=0, dtype=np.float64)

    fsupport = (numerators_compat / float(N)).astype(np.float32, copy=False)
    fconf = (numerators_compat / sum_a).astype(np.float32, copy=False)

    ok = (fsupport >= min_supp) & (fconf >= min_conf)
    if not np.any(ok):
        return []

    idxs_local = np.nonzero(ok)[0]
    idxs = compat_idx[idxs_local]

    result = []
    for pos, j in zip(idxs_local.tolist(), idxs.tolist()):
        lrule = sorted(ant) + [con_items[j]]
        r = CRFuzzyRule(lrule)
        result.append({
            "rule": r.sentence_rule(fuzzy_dataset),
            "coverage": coverage_a,
            "support": float(fsupport[pos]),
            "confidence": float(fconf[pos]),
            "lrule": tuple(lrule),
            "n_antecedents": r.get_num_features(),
        })
    return result


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
    con_chunk: int = 64,
    prune_epsilon: float = 0.05,
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
          "AARFI start (min_cov=%.3f, min_supp=%.3f, min_conf=%.3f, max_feat=%d, con_chunk=%d)...",
          min_cov, min_supp, min_conf, max_feat, con_chunk)

    # Precompute singleton memberships once and reuse in Apriori + consequents.
    singleton_mu = _precompute_singleton_memberships(
        dataset, fuzzy_dataset, verbose=verbose, logger=lg
    )

    # Mine antecedent candidates and get their vectors
    ant_candidates, ant_vecs = apriori(
        dataset,
        fuzzy_dataset,
        T,
        min_cov,
        max_feat,
        singleton_mu=singleton_mu,
        verbose=verbose,
        logger=lg,
    )
    _vlog(verbose, lg, "Antecedent candidates: %d", len(ant_candidates))

    # Build all singleton consequents once
    con_candidates = [next(iter(s)) for s in generate_fuzzy_1itemsets(
        fuzzy_dataset, verbose=verbose, logger=lg
    )]  # list of (f,l)
    _vlog(verbose, lg, "Consequent candidates (singletons): %d", len(con_candidates))
    C = np.stack([singleton_mu[c] for c in con_candidates], axis=1).astype(np.float32, copy=False)
    con_feat_idx = np.array([c[0] for c in con_candidates], dtype=np.int32)
    n_cons = C.shape[1]
    if not con_chunk or con_chunk <= 0:
        con_chunk = n_cons

    rules = []
    examined = 0
    kept = 0
    for ant in ant_candidates:
        ant_vec = ant_vecs[ant]
        sum_ant = float(ant_vec.sum())
        if sum_ant == 0.0:
            continue

        ant_features = {fi for (fi, _) in ant}
        compatible = np.nonzero(~np.isin(con_feat_idx, list(ant_features)))[0]
        examined += int(compatible.size)
        if compatible.size == 0:
            continue

        a_col = ant_vec[:, None]
        for s in range(0, compatible.size, con_chunk):
            e = min(s + con_chunk, compatible.size)
            idx_blk = compatible[s:e]
            C_blk = C[:, idx_blk]

            implied_blk = vI(a_col, C_blk)
            eval_blk = vT(a_col, implied_blk).astype(np.float32, copy=False)

            fsupport_blk = eval_blk.mean(axis=0)
            fconfidence_blk = eval_blk.sum(axis=0) / sum_ant
            ok = (fsupport_blk >= min_supp) & (fconfidence_blk >= min_conf)
            if not np.any(ok):
                continue

            for pos in np.nonzero(ok)[0].tolist():
                j = int(idx_blk[pos])
                con = con_candidates[j]
                lrule = sorted(ant) + [con]
                rule = CRFuzzyRule(lrule)
                # Reuse already computed vectors instead of evaluating the whole rule again.
                rule.antecedents = ant_vec
                rule.consequents = C[:, j]
                rule.evaluations = eval_blk[:, pos]
                rule.evaluated = 1
                rules.append(rule)
                kept += 1

    _vlog(verbose, lg, "AARFI finished. Rules kept: %d (from %d evaluated pairs).", kept, examined)
    pruned = redundancy_pruning(
        SetFuzzyRules(rules),
        epsilon=prune_epsilon,
        verbose=verbose,
        logger=lg,
    )
    return pruned


# ---------- Redundancy pruning --------------------------------------------

def redundancy_pruning(
    rules: SetFuzzyRules,
    epsilon: float = 0.05,
    confidence_map: Optional[Dict[Tuple[int,int], float]] = None,
    *,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> SetFuzzyRules:
    """
    Remove rules whose antecedent is a superset of another rule's antecedent
    with the same consequent and nearly equal confidence.
    If confidence_map is provided, it is used instead of r.fconfidence().
    The map must be keyed by tuple(r.lrule).
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
            if confidence_map is not None:
                c1 = float(confidence_map.get(tuple(r1.lrule), r1.fconfidence()))
            else:
                c1 = r1.fconfidence()
            redundant = False
            for r2 in selected:  # only compare to already kept (smaller) ones
                A2 = set(r2.lrule[:-1])
                if confidence_map is not None:
                    c2 = float(confidence_map.get(tuple(r2.lrule), r2.fconfidence()))
                else:
                    c2 = r2.fconfidence()
                if A2.issubset(A1) and abs(c1 - c2) < epsilon:
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
    prune_epsilon: float = 0.05,
    separable_F: bool = False,  # set True only when F(x,y) = x*g(y); precomputes g(C) once
    n_jobs: int = 1,            # parallel workers over antecedents; -1 = all cores (non-separable F only)
    ant_batch_size: int = 512,  # antecedents per BLAS batch (separable_F=True path)
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

    # ---- 1) Precompute singleton memberships once, and mine antecedents ----
    singleton_mu = _precompute_singleton_memberships(
        dataset, fuzzy_dataset, verbose=verbose, logger=lg
    )
    ant_list, ant_vecs = apriori(
        dataset,
        fuzzy_dataset,
        T,
        min_cov,
        max_feat,
        singleton_mu=singleton_mu,
        verbose=verbose,
        logger=lg,
    )
    if not ant_list:
        return SetFuzzyRules([]), pd.DataFrame(columns=["rule","coverage","support","confidence","lrule","n_antecedents"])

    # ---- 2) Stack all singleton consequents once: C shape (N, Cn) ----
    con_items = [next(iter(s)) for s in generate_fuzzy_1itemsets(
        fuzzy_dataset, verbose=verbose, logger=lg
    )]
    C = np.stack([singleton_mu[c] for c in con_items], axis=1).astype(np.float32, copy=False)
    N, Cn = C.shape
    con_feat_idx = np.array([c[0] for c in con_items], dtype=np.int32)

    if not chunk_rows or chunk_rows <= 0:
        chunk_rows = N

    # Optional precomputation: only valid when F(x,y) = x * g(y).
    # F(1, C) = g(C), so F(a, c) = a * g(c). Avoids recomputing g(C) per antecedent.
    C_f = F(np.ones((1, Cn), dtype=np.float32), C).astype(np.float32, copy=False) if separable_F else None

    if separable_F:
        # Batch matrix multiply: numerators[i, j] = dot(A[i], C_f[:, j]) in float64.
        # Replaces per-antecedent joblib dispatch with a single BLAS dgemm per batch.
        C_f64 = C_f.astype(np.float64, copy=False)
        rows = []
        for bs in range(0, len(ant_list), ant_batch_size):
            be = min(bs + ant_batch_size, len(ant_list))
            batch_ants = ant_list[bs:be]
            M = len(batch_ants)

            A = np.stack([ant_vecs[ant] for ant in batch_ants]).astype(np.float64)  # (M, N)
            sum_A = A.sum(axis=1)            # (M,)
            coverage_batch = A.mean(axis=1)  # (M,)

            numerators = A @ C_f64           # (M, Cn) — single BLAS dgemm call
            fsupport = (numerators / N).astype(np.float32)
            fconf    = (numerators / (sum_A[:, None] + 1e-12)).astype(np.float32)

            # Compatibility: consequent feature must differ from all antecedent features
            compat = np.ones((M, Cn), dtype=bool)
            for i, ant in enumerate(batch_ants):
                ant_feats = np.fromiter((fi for fi, _ in ant), dtype=np.int32, count=len(ant))
                compat[i] = ~np.isin(con_feat_idx, ant_feats)

            ok = (fsupport >= min_supp) & (fconf >= min_conf) & compat
            ant_idxs, con_idxs = np.nonzero(ok)
            for ai, cj in zip(ant_idxs.tolist(), con_idxs.tolist()):
                ant = batch_ants[ai]
                lrule = sorted(ant) + [con_items[cj]]
                r = CRFuzzyRule(lrule)
                rows.append({
                    "rule": r.sentence_rule(fuzzy_dataset),
                    "coverage": float(coverage_batch[ai]),
                    "support": float(fsupport[ai, cj]),
                    "confidence": float(fconf[ai, cj]),
                    "lrule": tuple(lrule),
                    "n_antecedents": r.get_num_features(),
                })
    else:
        # Non-separable F: keep per-antecedent evaluation with optional parallelism.
        batch = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_eval_antecedent)(
                ant, ant_vecs[ant].astype(np.float32, copy=False),
                C, C_f, con_feat_idx, con_items, F, separable_F,
                N, chunk_rows, min_supp, min_conf, fuzzy_dataset,
            )
            for ant in ant_list
        )
        rows = [row for ant_rows in batch for row in ant_rows]

    rule_objs = [CRFuzzyRule(list(row["lrule"])) for row in rows]
    _vlog(verbose, lg, "AARFI_F finished. Rules kept: %d (from %d antecedents).",
          len(rows), len(ant_list))

    # Build DataFrame and sort for nicer presentation
    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["confidence","support","coverage"], ascending=False, kind="mergesort").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["rule","coverage","support","confidence","lrule","n_antecedents"])

    conf_map = {tuple(lr): float(c) for lr, c in zip(df["lrule"], df["confidence"])} if not df.empty else None
    pruned = redundancy_pruning(
        SetFuzzyRules(rule_objs),
        epsilon=prune_epsilon,
        confidence_map=conf_map,
        verbose=verbose,
        logger=lg,
    )
    if not df.empty:
        # Keep DataFrame rows that survived pruning, then align rule_list order to DataFrame order.
        pruned_by_lrule: Dict[Tuple, CRFuzzyRule] = {tuple(r.lrule): r for r in pruned.rule_list}
        df = df[df["lrule"].isin(pruned_by_lrule.keys())].reset_index(drop=True)
        aligned_rules = [pruned_by_lrule[tuple(lr)] for lr in df["lrule"].tolist()]
        pruned = SetFuzzyRules(aligned_rules)
    return pruned, df


def ARFI_beam(
    dataset,
    fuzzy_dataset,
    T,
    I,
    F: Callable[[np.ndarray, np.ndarray], np.ndarray],
    *,
    min_cov: float = 0.3,
    min_supp: float = 0.3,
    min_conf: float = 0.8,
    max_feat: int = 5,
    beam_width: int = 128,
    max_children_per_node: int = 32,
    chunk_rows: int = 100_000,
    score_fn: Optional[Callable[[float, int], float]] = None,
    prune_epsilon: float = 0.05,
    separable_F: bool = False,  # set True only when F(x,y) = x*g(y); precomputes g(C) once
    n_jobs: int = 1,            # parallel workers over antecedents; -1 = all cores (non-separable F only)
    ant_batch_size: int = 512,  # antecedents per BLAS batch (separable_F=True path)
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[SetFuzzyRules, pd.DataFrame]:
    """
    Beam-search version focused on speed for large datasets.

    Output format matches AARFI_F:
      - SetFuzzyRules (un-evaluated rule objects)
      - DataFrame columns: ['rule','coverage','support','confidence','lrule','n_antecedents']
    """
    lg = _get_logger(logger, verbose)
    _vlog(
        verbose,
        lg,
        "ARFI_beam (min_cov=%.3f, min_supp=%.3f, min_conf=%.3f, max_feat=%d, beam_width=%d, max_children_per_node=%d)",
        min_cov,
        min_supp,
        min_conf,
        max_feat,
        beam_width,
        max_children_per_node,
    )

    singleton_mu = _precompute_singleton_memberships(
        dataset, fuzzy_dataset, verbose=verbose, logger=lg
    )
    ant_list, ant_vecs = beam_apriori(
        dataset,
        fuzzy_dataset,
        T,
        min_cov=min_cov,
        max_feat=max_feat,
        beam_width=beam_width,
        max_children_per_node=max_children_per_node,
        singleton_mu=singleton_mu,
        score_fn=score_fn,
        verbose=verbose,
        logger=lg,
    )
    if not ant_list:
        return SetFuzzyRules([]), pd.DataFrame(columns=["rule","coverage","support","confidence","lrule","n_antecedents"])

    con_items = [next(iter(s)) for s in generate_fuzzy_1itemsets(
        fuzzy_dataset, verbose=verbose, logger=lg
    )]
    C = np.stack([singleton_mu[c] for c in con_items], axis=1).astype(np.float32, copy=False)
    N, Cn = C.shape
    con_feat_idx = np.array([c[0] for c in con_items], dtype=np.int32)

    if not chunk_rows or chunk_rows <= 0:
        chunk_rows = N

    # Optional precomputation: only valid when F(x,y) = x * g(y).
    C_f = F(np.ones((1, Cn), dtype=np.float32), C).astype(np.float32, copy=False) if separable_F else None

    if separable_F:
        # Batch matrix multiply: same optimisation as AARFI_F.
        C_f64 = C_f.astype(np.float64, copy=False)
        rows = []
        for bs in range(0, len(ant_list), ant_batch_size):
            be = min(bs + ant_batch_size, len(ant_list))
            batch_ants = ant_list[bs:be]
            M = len(batch_ants)

            A = np.stack([ant_vecs[ant] for ant in batch_ants]).astype(np.float64)  # (M, N)
            sum_A = A.sum(axis=1)
            coverage_batch = A.mean(axis=1)

            numerators = A @ C_f64           # (M, Cn)
            fsupport = (numerators / N).astype(np.float32)
            fconf    = (numerators / (sum_A[:, None] + 1e-12)).astype(np.float32)

            compat = np.ones((M, Cn), dtype=bool)
            for i, ant in enumerate(batch_ants):
                ant_feats = np.fromiter((fi for fi, _ in ant), dtype=np.int32, count=len(ant))
                compat[i] = ~np.isin(con_feat_idx, ant_feats)

            ok = (fsupport >= min_supp) & (fconf >= min_conf) & compat
            ant_idxs, con_idxs = np.nonzero(ok)
            for ai, cj in zip(ant_idxs.tolist(), con_idxs.tolist()):
                ant = batch_ants[ai]
                lrule = sorted(ant) + [con_items[cj]]
                r = CRFuzzyRule(lrule)
                rows.append({
                    "rule": r.sentence_rule(fuzzy_dataset),
                    "coverage": float(coverage_batch[ai]),
                    "support": float(fsupport[ai, cj]),
                    "confidence": float(fconf[ai, cj]),
                    "lrule": tuple(lrule),
                    "n_antecedents": r.get_num_features(),
                })
    else:
        batch = Parallel(n_jobs=n_jobs, prefer='threads')(
            delayed(_eval_antecedent)(
                ant, ant_vecs[ant].astype(np.float32, copy=False),
                C, C_f, con_feat_idx, con_items, F, separable_F,
                N, chunk_rows, min_supp, min_conf, fuzzy_dataset,
            )
            for ant in ant_list
        )
        rows = [row for ant_rows in batch for row in ant_rows]

    rule_objs = [CRFuzzyRule(list(row["lrule"])) for row in rows]
    _vlog(verbose, lg, "ARFI_beam finished. Rules kept: %d (from %d antecedents).",
          len(rows), len(ant_list))

    if rows:
        df = pd.DataFrame(rows)
        df = df.sort_values(["confidence","support","coverage"], ascending=False, kind="mergesort").reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=["rule","coverage","support","confidence","lrule","n_antecedents"])

    conf_map = {tuple(lr): float(c) for lr, c in zip(df["lrule"], df["confidence"])} if not df.empty else None
    pruned = redundancy_pruning(
        SetFuzzyRules(rule_objs),
        epsilon=prune_epsilon,
        confidence_map=conf_map,
        verbose=verbose,
        logger=lg,
    )
    if not df.empty:
        # Keep DataFrame rows that survived pruning, then align rule_list order to DataFrame order.
        pruned_by_lrule: Dict[Tuple, CRFuzzyRule] = {tuple(r.lrule): r for r in pruned.rule_list}
        df = df[df["lrule"].isin(pruned_by_lrule.keys())].reset_index(drop=True)
        aligned_rules = [pruned_by_lrule[tuple(lr)] for lr in df["lrule"].tolist()]
        pruned = SetFuzzyRules(aligned_rules)
    return pruned, df



