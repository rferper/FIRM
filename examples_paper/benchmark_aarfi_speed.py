import argparse
import statistics
import time
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

# Allow running this script directly from examples_paper.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import FIRM.base.fuzzy_data as fuzzy_data
from FIRM.methods.AARFI import AARFI_F


def process_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Match the notebook preprocessing used in examples_paper/test2.ipynb.
    """
    out = df.copy()
    for col in out.columns:
        s = out[col]

        if pd.api.types.is_integer_dtype(s):
            out[col] = s.astype("float64")
            continue

        is_cat_like = (
            pd.api.types.is_object_dtype(s)
            or isinstance(s.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(s)
        )
        if is_cat_like:
            n_unique = s.nunique(dropna=True)
            if n_unique > 10:
                top10 = s.value_counts(dropna=True).index[:10]
                out[col] = s.where(s.isna() | s.isin(top10), "Unknown").astype("object")
            else:
                out[col] = s.astype("object")

    return out


def make_dataset(csv_path: str, n_labels: int) -> Tuple[pd.DataFrame, object]:
    resolved_csv = Path(csv_path)
    if not resolved_csv.is_absolute():
        resolved_csv = PROJECT_ROOT / resolved_csv

    df = process_df(pd.read_csv(resolved_csv))
    dataset = df.copy()

    int_cols = dataset.select_dtypes(include=["int"]).columns
    dataset[int_cols] = dataset[int_cols].astype(float)

    if n_labels == 3:
        labels = ["L", "M", "H"]
    else:
        labels = [f"L{i+1}" for i in range(n_labels)]

    fuzzy_dataset = fuzzy_data.FuzzyDataQuantiles("symmetric", dataset, n_labels, labels)
    return dataset, fuzzy_dataset


def benchmark(name: str, fn: Callable[[], int], repeats: int) -> List[float]:
    times: List[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        n_rules = fn()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"{name:>16}: {dt:8.4f}s | n_rules={n_rules}")
    return times


def crisp_apriori_timing_test4(
    dataset: pd.DataFrame,
    fuzzy_dataset,
    min_support: float,
    min_conf: float,
    max_feat: int,
) -> Tuple[float, float, int] | Tuple[None, None, None]:
    try:
        from mlxtend.frequent_patterns import apriori as mlxtend_apriori, association_rules
    except Exception:
        return None, None, None

    # Same crisp preprocessing pattern as examples_paper/test4.ipynb
    data = dataset.copy()
    for i in range(len(fuzzy_dataset.fv_list)):
        data[dataset.columns[i]] = dataset[dataset.columns[i]].map(
            lambda x: fuzzy_dataset.fv_list[i].eval_max_fuzzy_set(x)
        )

    encoded = pd.get_dummies(data, columns=data.columns)

    t0 = time.perf_counter()
    df_freq = mlxtend_apriori(
        encoded,
        min_support=min_support,
        use_colnames=True,
        verbose=0,
        max_len=max_feat + 1,
        low_memory=True,
    )
    apriori_dt = time.perf_counter() - t0

    t1 = time.perf_counter()
    df_ar = association_rules(df_freq, metric="confidence", min_threshold=min_conf)
    df_rules_filtered = df_ar[
        (df_ar["antecedents"].apply(len) <= max_feat) &
        (df_ar["consequents"].apply(len) <= 1)
    ].reset_index(drop=True)
    full_dt = time.perf_counter() - t0
    _ = t1  # keep marker for readability if you later print split times
    return apriori_dt, full_dt, len(df_rules_filtered)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark AARFI_F speed.")
    parser.add_argument("--csv", default="assets/online_news.csv", help="Input CSV path.")
    parser.add_argument("--repeats", type=int, default=5, help="Benchmark repeats.")
    parser.add_argument("--min-cov", type=float, default=0.1)
    parser.add_argument("--min-supp", type=float, default=0.1)
    parser.add_argument("--min-conf", type=float, default=0.4)
    parser.add_argument("--max-feat", type=int, default=3)
    parser.add_argument("--labels", type=int, default=3)
    parser.add_argument("--row-chunk", type=int, default=100000, help="Row chunk size for AARFI_F.")
    args = parser.parse_args()

    dataset, fuzzy_dataset = make_dataset(args.csv, args.labels)

    I = lambda x, y: 1 - x + x * (y ** 0.01)
    T = lambda x, y: np.maximum(x + y - 1, 0)
    F = lambda x, y: x * (y ** 0.01)

    print("Configuration:")
    print(f"  csv={args.csv}")
    print(f"  rows={len(dataset)}, cols={len(dataset.columns)}")
    print(
        "  thresholds="
        f"(min_cov={args.min_cov}, min_supp={args.min_supp}, "
        f"min_conf={args.min_conf}, max_feat={args.max_feat})"
    )
    print(f"  repeats={args.repeats}")
    print()

    def run_aarfi_f() -> int:
        rules, _ = AARFI_F(
            dataset,
            fuzzy_dataset,
            T=T,
            I=I,
            F=F,
            min_cov=args.min_cov,
            min_supp=args.min_supp,
            min_conf=args.min_conf,
            max_feat=args.max_feat,
            chunk_rows=args.row_chunk,
            verbose=False,
        )
        return len(rules.rule_list)

    print("Timing crisp Apriori (test4 style):")
    crisp_apriori_times: List[float] = []
    crisp_full_times: List[float] = []
    crisp_rules: List[int] = []
    for _ in range(args.repeats):
        apr_dt, full_dt, n_rules = crisp_apriori_timing_test4(
            dataset,
            fuzzy_dataset,
            min_support=args.min_cov,
            min_conf=args.min_conf,
            max_feat=args.max_feat,
        )
        if apr_dt is None:
            print("   crisp_apriori: skipped (mlxtend not installed)")
            break
        crisp_apriori_times.append(apr_dt)
        crisp_full_times.append(full_dt)
        crisp_rules.append(n_rules)
        print(
            f"{'CRISP apriori':>16}: {apr_dt:8.4f}s | "
            f"full={full_dt:8.4f}s | n_rules={n_rules}"
        )
    print()

    print("Timing AARFI_F:")
    aarfi_f_times = benchmark("AARFI_F", run_aarfi_f, args.repeats)
    print()

    print("Summary (median over repeats):")
    if crisp_apriori_times:
        print(f"  Crisp Apriori median: {statistics.median(crisp_apriori_times):.4f}s")
        print(f"  Crisp Full median:    {statistics.median(crisp_full_times):.4f}s")
        print(f"  Crisp rules median:   {int(statistics.median(crisp_rules))}")
    med_aarfi_f = statistics.median(aarfi_f_times)
    print(f"  AARFI_F median: {med_aarfi_f:.4f}s")


if __name__ == "__main__":
    main()
