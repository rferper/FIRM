import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import FIRM.base.fuzzy_data as fuzzy_data
from FIRM.methods.AARFI import AARFI, AARFI_F, ARFI_beam


def process_df(df: pd.DataFrame) -> pd.DataFrame:
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
        labels = [f"L{i + 1}" for i in range(n_labels)]

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark ARFI_beam speed.")
    parser.add_argument("--csv", default="assets/global_house.csv", help="Input CSV path.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--min-cov", type=float, default=0.3)
    parser.add_argument("--min-supp", type=float, default=0.3)
    parser.add_argument("--min-conf", type=float, default=0.8)
    parser.add_argument("--max-feat", type=int, default=3)
    parser.add_argument("--labels", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=64)
    parser.add_argument("--max-children", type=int, default=16)
    parser.add_argument("--row-chunk", type=int, default=100000)
    parser.add_argument("--con-chunk", type=int, default=64, help="Consequent chunk size for normal AARFI.")
    parser.add_argument("--compare-aarfi-f", action="store_true", help="Also benchmark AARFI_F.")
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
    print(
        "  beam="
        f"(beam_width={args.beam_width}, max_children={args.max_children}, "
        f"row_chunk={args.row_chunk})"
    )
    print(f"  repeats={args.repeats}")
    print()

    def run_arfi_beam() -> int:
        rules, _ = ARFI_beam(
            dataset,
            fuzzy_dataset,
            T=T,
            I=I,
            F=F,
            min_cov=args.min_cov,
            min_supp=args.min_supp,
            min_conf=args.min_conf,
            max_feat=args.max_feat,
            beam_width=args.beam_width,
            max_children_per_node=args.max_children,
            chunk_rows=args.row_chunk,
            verbose=False,
        )
        return len(rules.rule_list)

    print("Timing ARFI_beam:")
    beam_times = benchmark("ARFI_beam", run_arfi_beam, args.repeats)
    print()

    def run_aarfi() -> int:
        rules = AARFI(
            dataset,
            fuzzy_dataset,
            T=T,
            I=I,
            min_cov=args.min_cov,
            min_supp=args.min_supp,
            min_conf=args.min_conf,
            max_feat=args.max_feat,
            con_chunk=args.con_chunk,
            verbose=False,
        )
        return len(rules.rule_list)

    print("Timing AARFI (normal):")
    aarfi_times = benchmark("AARFI", run_aarfi, args.repeats)
    print()

    med_beam = statistics.median(beam_times)
    med_aarfi = statistics.median(aarfi_times)
    print("Summary (median over repeats):")
    print(f"  ARFI_beam median: {med_beam:.4f}s")
    print(f"  AARFI median:     {med_aarfi:.4f}s")
    if med_aarfi > 0:
        print(f"  ARFI_beam/AARFI:  {med_beam / med_aarfi:.3f}x")

    if args.compare_aarfi_f:
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

        print("Timing AARFI_F:")
        aarfi_f_times = benchmark("AARFI_F", run_aarfi_f, args.repeats)
        med_aarfi_f = statistics.median(aarfi_f_times)
        print()
        print(f"  AARFI_F median:   {med_aarfi_f:.4f}s")
        if med_aarfi_f > 0:
            print(f"  ARFI_beam/AARFI_F: {med_beam / med_aarfi_f:.3f}x")


if __name__ == "__main__":
    main()
