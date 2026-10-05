"""Phase 3: single-factor tests on the in-sample period only.

Usage: python scripts/20_factor_tests.py
The hold-out period (HOLDOUT_START onward) is excluded here and is used once, at the end.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (BACKTEST_START, DATA_PROCESSED, FACTOR_PRIORS, HOLDOUT_START,  # noqa: E402
                    N_QUANTILES, PASS_LS_POSITIVE, PASS_MIN_RANK_IC, PASS_MIN_T, REPORTS)
from src.evaluation.factor_tests import test_factor  # noqa: E402

pd.set_option("display.width", 220)
REPORTS.mkdir(exist_ok=True)

panel = pd.read_parquet(DATA_PROCESSED / "factors_monthly.parquet")
# The last in-sample rebalance date must have its forward return end before the hold-out starts
last_dates = sorted(d for d in panel["date"].unique() if d < pd.Timestamp(HOLDOUT_START))
panel = panel[(panel["date"] >= BACKTEST_START) & (panel["date"] < last_dates[-1])]
print(f"In-sample: {panel['date'].min().date()} to {panel['date'].max().date()} "
      f"({panel['date'].nunique()} rebalance dates; returns through {last_dates[-1].date()})")

rows, rics, qrets = {}, {}, {}
for f in FACTOR_PRIORS:
    rows[f], rics[f], qrets[f] = test_factor(panel, f, N_QUANTILES)
res = pd.DataFrame(rows).T

res["pass"] = (res["RankIC"] >= PASS_MIN_RANK_IC) & (res["t"] >= PASS_MIN_T)
if PASS_LS_POSITIVE:
    res["pass"] &= res["LS_ann"] > 0
res["reversed"] = (res["RankIC"] <= -PASS_MIN_RANK_IC) & (res["t"] <= -PASS_MIN_T)
res.to_csv(REPORTS / "phase3_factor_tests.csv")

show = res.copy()
pct = ["Q1", "Q2", "Q3", "Q4", "Q5", "LS_ann", "LS_vol", "RankIC>0"]
for c in pct:
    show[c] = (show[c] * 100).round(2)
for c in ["IC", "RankIC", "RankIC_std"]:
    show[c] = show[c].round(4)
for c in ["ICIR", "t", "LS_sharpe", "monotonic", "autocorr"]:
    show[c] = show[c].astype(float).round(2)
print("\n== Single-factor results (returns in %, quantile returns are annualised excess over the "
      "equal-weighted universe) ==")
print(show.drop(columns="months").to_string())
print(f"\nPass rule: RankIC >= {PASS_MIN_RANK_IC}, t >= {PASS_MIN_T}, Q5-Q1 > 0")
print("passed:", list(res.index[res["pass"]]))
print("significant in the opposite direction:", list(res.index[res["reversed"]]))

ric_df = pd.DataFrame(rics)
yearly = ric_df.groupby(ric_df.index.year).mean().round(3)
print("\n== Mean Rank IC by year ==")
print(yearly.to_string())
yearly.to_csv(REPORTS / "phase3_rank_ic_by_year.csv")

# Figures
fig, ax = plt.subplots(figsize=(10, 5))
ric_df.cumsum().plot(ax=ax)
ax.set_title("Cumulative Rank IC (in-sample)")
ax.axhline(0, color="grey", lw=0.8)
fig.tight_layout()
fig.savefig(REPORTS / "phase3_cum_rank_ic.png", dpi=150)

fig, axes = plt.subplots(2, 5, figsize=(20, 7), sharex=True)
for ax, f in zip(axes.flat, FACTOR_PRIORS):
    (1 + qrets[f]).cumprod().plot(ax=ax, legend=False)
    ax.set_title(f)
axes.flat[0].legend([f"Q{i}" for i in range(1, N_QUANTILES + 1)], fontsize=7)
fig.suptitle("Cumulative excess return by quintile (Q5 = highest factor value)")
fig.tight_layout()
fig.savefig(REPORTS / "phase3_quintiles.png", dpi=150)
print(f"\nSaved: {REPORTS / 'phase3_factor_tests.csv'}, phase3_rank_ic_by_year.csv, "
      "phase3_cum_rank_ic.png, phase3_quintiles.png")
