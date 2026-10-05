"""Phase 2: build the monthly factor panel and print sanity checks."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BACKTEST_START, DATA_PROCESSED, FACTOR_PRIORS  # noqa: E402
from src.factors.build import build_factor_panel  # noqa: E402

pd.set_option("display.width", 160)
panel = build_factor_panel()
names = list(FACTOR_PRIORS)

print("\n== Panel ==")
print("rows:", len(panel), "| months:", panel["date"].nunique(),
      "|", panel["date"].min().date(), "to", panel["date"].max().date())
print("stocks per rebalance date, by year (median):")
print(panel.groupby(panel["date"].dt.year)["symbol"].count().div(
    panel.groupby(panel["date"].dt.year)["date"].nunique()).round(0).to_string())

print("\nSW level-1 industries on last date:", panel[panel["date"] == panel["date"].max()]["industry"].nunique(),
      "| codes:", sorted(panel["industry"].unique())[:40])

print("\n== Missing share of raw factors before filling (by year) ==")
print(pd.read_csv(DATA_PROCESSED / "factor_missing_by_year.csv", index_col=0).round(3).to_string())

print("\n== fwd_ret ==")
print(panel["fwd_ret"].describe(percentiles=[0.01, 0.5, 0.99]).round(4).to_string())

last = panel[panel["date"] == panel["date"].max()]
print(f"\n== Raw factor distribution on {last['date'].iloc[0].date()} ==")
print(last[[f"{n}_raw" for n in names]].describe(percentiles=[0.01, 0.5, 0.99]).T.round(4).to_string())

print("\n== Processed factors: mean / std across all dates (should be about 0 / 1) ==")
print(panel[names].agg(["mean", "std"]).T.round(3).to_string())

sample = panel[panel["date"] >= BACKTEST_START]
corr = sample.groupby("date")[names].corr().groupby(level=1).mean().loc[names, names]
print(f"\n== Average cross-sectional correlation of processed factors (from {BACKTEST_START}) ==")
print(corr.round(2).to_string())
