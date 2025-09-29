# -*- coding: utf-8 -*-
"""
Check the required properties for functions F: [0,1]^2 -> [0,1].

Conditions to check:
1) Boundary: F(0,y)=0 and F(x,1)=x
2) Range: F(x,y) in [0,1]
3) Monotonicity: increasing in both variables
4) Not commutative and far from max/product
5) Strong asymmetry: F(x,y) >> F(y,x)

Two candidate functions are included:
  - F_exp(x,y; alpha) = x * (exp(alpha*y)-1)/(exp(alpha)-1)
  - F_power(x,y; k)   = x * y^k
"""

import math
from typing import Callable, Tuple, Dict

# --------------------------- CONFIG ---------------------------
GRID_STEPS = 51            # resolution per axis (>= 21 recommended)
TOL = 1e-6                 # numeric tolerance for equalities/monotonicity
DIFF_THR = 5e-2            # "very different from max/product" mean-abs-error threshold
ASYM_DELTA = 0.10          # only compare pairs with y >= x + ASYM_DELTA
ASYM_TARGET_RATIO = 5.0    # require median(F(x,y)/F(y,x)) >= this (strong asymmetry)
REPORT_EXAMPLES = 5        # how many example pairs to print when useful
# --------------------------------------------------------------


def F_exp(x: float, y: float, alpha: float = 8.0) -> float:
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("x,y must be in [0,1]")
    denom = math.exp(alpha) - 1.0
    return x * (math.exp(alpha * y) - 1.0) / denom


def F_power(x: float, y: float, k: int = 3) -> float:
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        raise ValueError("x,y must be in [0,1]")
    return x * (y ** k)


def _grid_points(n: int):
    step = 1.0 / (n - 1)
    return [i * step for i in range(n)]


def check_all(
    F: Callable[[float, float], float],
    name: str = "F",
    grid_steps: int = GRID_STEPS,
    tol: float = TOL,
    diff_thr: float = DIFF_THR,
    asym_delta: float = ASYM_DELTA,
    asym_target: float = ASYM_TARGET_RATIO,
) -> Dict[str, bool]:
    xs = _grid_points(grid_steps)
    ys = xs

    # Metrics we’ll aggregate
    in_range_ok = True
    range_violations = []

    # 1) Boundary
    boundary_ok = True
    boundary_violations = []

    # 2) Monotonicity
    mono_x_ok = True
    mono_y_ok = True
    mono_x_viol = []
    mono_y_viol = []

    # 3) Not commutative
    comm_ok = False
    comm_examples = []

    # 4) Far from max/product
    mae_vs_prod = 0.0
    mae_vs_max = 0.0
    count = 0

    # 5) Strong asymmetry
    ratios = []      # F(x,y)/(F(y,x)) for pairs with y >= x + delta
    ratio_examples = []

    # Precompute values to avoid recomputation
    table = [[None for _ in ys] for _ in xs]
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            val = F(x, y)
            table[i][j] = val
            # Range
            if not (-tol <= val <= 1.0 + tol):
                in_range_ok = False
                range_violations.append(((x, y), val))

    # Boundary checks
    # F(0,y)=0
    x0_idx = 0
    for j, y in enumerate(ys):
        v = table[x0_idx][j]
        if abs(v - 0.0) > tol:
            boundary_ok = False
            boundary_violations.append(("F(0,y)=0", (0.0, y), v))
    # F(x,1)=x
    y1_idx = len(ys) - 1
    for i, x in enumerate(xs):
        v = table[i][y1_idx]
        if abs(v - x) > tol:
            boundary_ok = False
            boundary_violations.append(("F(x,1)=x", (x, 1.0), v))

    # Monotonicity: in x (for fixed y)
    for j, y in enumerate(ys):
        prev = table[0][j]
        for i in range(1, len(xs)):
            cur = table[i][j]
            if cur + tol < prev:  # should be non-decreasing
                mono_x_ok = False
                mono_x_viol.append(((xs[i-1], y), (xs[i], y), (prev, cur)))
            prev = cur

    # Monotonicity: in y (for fixed x)
    for i, x in enumerate(xs):
        prev = table[i][0]
        for j in range(1, len(ys)):
            cur = table[i][j]
            if cur + tol < prev:
                mono_y_ok = False
                mono_y_viol.append(((x, ys[j-1]), (x, ys[j]), (prev, cur)))
            prev = cur

    # Not commutative; differences & distances to product/max
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            v = table[i][j]
            v_xy = v
            v_yx = table[j][i]
            if abs(v_xy - v_yx) > 5e-4 and len(comm_examples) < REPORT_EXAMPLES:
                comm_ok = True
                comm_examples.append(((x, y), v_xy, v_yx))
            # distances
            mae_vs_prod += abs(v - (x * y))
            mae_vs_max += abs(v - (x if x >= y else y))
            count += 1

    mae_vs_prod /= max(1, count)
    mae_vs_max /= max(1, count)
    far_from_prod = mae_vs_prod >= diff_thr
    far_from_max = mae_vs_max >= diff_thr

    # Strong asymmetry ratios for y >= x + delta (exclude tiny denominators)
    eps = 1e-12
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if y >= x + asym_delta:
                num = table[i][j]
                den = table[j][i]
                if den > eps:
                    r = num / den
                    ratios.append((r, (x, y), num, den))

    if ratios:
        ratios.sort(key=lambda t: t[0])  # ascending
        # median ratio
        mid = len(ratios) // 2
        if len(ratios) % 2 == 1:
            median_ratio = ratios[mid][0]
        else:
            median_ratio = 0.5 * (ratios[mid - 1][0] + ratios[mid][0])
        # keep a few large examples
        for r, xy, num, den in ratios[-REPORT_EXAMPLES:]:
            ratio_examples.append((xy, r, num, den))
    else:
        median_ratio = float('nan')

    strong_asym_ok = (not math.isnan(median_ratio)) and (median_ratio >= asym_target)

    # Print report
    print(f"\n=== Report for {name} ===")
    print(f"Grid: {grid_steps}x{grid_steps}, tol={tol}")
    print(f"[Range]    in [0,1]: {'PASS' if in_range_ok else 'FAIL'}")
    if not in_range_ok:
        print("  First violations (x,y)->value:")
        for (xy, v) in range_violations[:REPORT_EXAMPLES]:
            print(f"   {xy} -> {v:.6g}")

    print(f"[Boundary] F(0,y)=0 & F(x,1)=x: {'PASS' if boundary_ok else 'FAIL'}")
    if not boundary_ok:
        for what, (x, y), v in boundary_violations[:REPORT_EXAMPLES]:
            print(f"  {what} violated at ({x:.3f},{y:.3f}) -> {v:.6g}")

    print(f"[Monotone in x]  {'PASS' if mono_x_ok else 'FAIL'}")
    if not mono_x_ok:
        for a, b, (va, vb) in mono_x_viol[:REPORT_EXAMPLES]:
            print(f"  F{a}={va:.6g} > F{b}={vb:.6g}")

    print(f"[Monotone in y]  {'PASS' if mono_y_ok else 'FAIL'}")
    if not mono_y_ok:
        for a, b, (va, vb) in mono_y_viol[:REPORT_EXAMPLES]:
            print(f"  F{a}={va:.6g} > F{b}={vb:.6g}")

    print(f"[Not commutative] {'PASS' if comm_ok else 'FAIL'}")
    if comm_ok:
        print("  Example pairs (x,y): F(x,y) vs F(y,x)")
        for (x, y), vxy, vyx in comm_examples:
            print(f"   ({x:.2f},{y:.2f}): {vxy:.6g} vs {vyx:.6g}")

    print(f"[Far from product]  MAE={mae_vs_prod:.5f} (thr {diff_thr}) -> {'PASS' if far_from_prod else 'FAIL'}")
    print(f"[Far from max]      MAE={mae_vs_max:.5f} (thr {diff_thr}) -> {'PASS' if far_from_max else 'FAIL'}")

    print(f"[Strong asymmetry]  median F(x,y)/F(y,x) for y>=x+{asym_delta:.2f}: "
          f"{median_ratio:.3f} (target {asym_target}) -> {'PASS' if strong_asym_ok else 'FAIL'}")
    if ratio_examples:
        print("  Largest ratio examples (x,y): ratio [F(x,y) / F(y,x)]")
        for (x, y), r, num, den in ratio_examples:
            print(f"   ({x:.2f},{y:.2f}): {r:.3f}  [{num:.6g} / {den:.6g}]")

    return {
        "range_ok": in_range_ok,
        "boundary_ok": boundary_ok,
        "mono_x_ok": mono_x_ok,
        "mono_y_ok": mono_y_ok,
        "not_commutative_ok": comm_ok,
        "far_from_prod_ok": far_from_prod,
        "far_from_max_ok": far_from_max,
        "strong_asym_ok": strong_asym_ok,
    }


# --------------------------- RUN DEMO ---------------------------
if __name__ == "__main__":
    # Exponential family (strongly asymmetric)
    alpha = 8.0
    print("Testing F_exp with alpha =", alpha)
    check_all(lambda x, y: F_exp(x, y, alpha=alpha), name=f"F_exp(alpha={alpha})")

    # Power family
    k = 5
    print("\nTesting F_power with k =", k)
    check_all(lambda x, y: F_power(x, y, k=k), name=f"F_power(k={k})")
