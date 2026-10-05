"""Phase 5: long-only monthly backtest with costs and A-share trading limits (in-sample window).
Usage: python scripts/40_backtest.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import BT_SIGNALS, DATA_PROCESSED, DATA_RAW, REPORTS, TOP_FRACTION  # noqa: E402
from src.backtest.engine import (ew_universe, execution_dates, index_returns,  # noqa: E402
                                 load_exec_data, metrics, run_backtest, yearly_by_start)
from src.data.utils import to_ns  # noqa: E402

pd.set_option("display.width", 200)
scores = pd.read_parquet(DATA_PROCESSED / "predictions_insample.parquet")
scores["date"] = to_ns(scores["date"])
rebal = sorted(scores["date"].unique())
cal = to_ns(pd.read_parquet(DATA_RAW / "trade_calendar.parquet")["trade_date"])
exec_d = execution_dates(rebal, cal)
print(f"Holding periods: {len(rebal) - 1}, first trade {exec_d[0].date()}, last valuation {exec_d[-1].date()}")

mkt = load_exec_data(sorted(scores["symbol"].unique()), exec_d)

results, series = {}, {}
universe = ew_universe(scores, rebal, exec_d, mkt)
csi500 = index_returns(DATA_RAW / "benchmarks" / "csi500.parquet", exec_d)
csi1000 = index_returns(DATA_RAW / "benchmarks" / "csi1000.parquet", exec_d)

for sig in BT_SIGNALS:
    bt = run_backtest(scores, sig, rebal, exec_d, mkt, TOP_FRACTION)
    bt.to_csv(REPORTS / f"phase5_{sig}_periods.csv")
    series[f"{sig} (net)"] = bt["net"]
    series[f"{sig} (gross)"] = bt["gross"]
    for kind in ("gross", "net"):
        m = metrics(bt[kind], universe)
        m["vs_csi500"] = m["ann_ret"] - metrics(csi500)["ann_ret"]
        m["vs_csi1000"] = m["ann_ret"] - metrics(csi1000)["ann_ret"]
        m["turnover_mo"] = bt["turnover"].mean()
        m["cost_ann"] = bt["cost"].mean() * 12
        m["avg_holdings"] = bt["n_hold"].mean()
        results[f"{sig} ({kind})"] = m
    print(f"{sig}: average blocked buys per month {bt['n_target_blocked'].mean():.1f}, "
          f"average stuck holdings {bt['n_stuck'].mean():.1f}")

for name, r in [("EW universe", universe), ("CSI 500", csi500), ("CSI 1000", csi1000)]:
    results[name] = metrics(r)
    series[name] = r

res = pd.DataFrame(results).T
res.to_csv(REPORTS / "phase5_backtest.csv")
show = res.copy()
for c in ["ann_ret", "ann_vol", "max_dd", "excess_ann", "vs_csi500", "vs_csi1000", "win_rate",
          "turnover_mo", "cost_ann"]:
    show[c] = (show[c] * 100).round(2)
for c in ["sharpe", "info_ratio", "avg_holdings"]:
    show[c] = show[c].astype(float).round(2)
print("\n== Backtest (in %, monthly data; excess_ann, info_ratio and win_rate are vs EW universe; "
      "Sharpe uses a zero risk-free rate) ==")
print(show.to_string())

yearly = yearly_by_start(series, exec_d)
print("\n== Returns by year of period start (%; first and last years are partial) ==")
print((yearly * 100).round(2).to_string())
yearly.to_csv(REPORTS / "phase5_yearly.csv")

fig, ax = plt.subplots(figsize=(11, 5))
nav = (1 + pd.DataFrame(series)[[f"{BT_SIGNALS[0]} (net)", f"{BT_SIGNALS[0]} (gross)",
                                 f"{BT_SIGNALS[1]} (net)", "EW universe", "CSI 500", "CSI 1000"]]).cumprod()
nav.plot(ax=ax, title="Net asset value (monthly)")
ax.axhline(1, color="grey", lw=0.8)
fig.tight_layout()
fig.savefig(REPORTS / "phase5_nav.png", dpi=150)
print("\nSaved: phase5_backtest.csv, phase5_yearly.csv, phase5_nav.png, phase5_<signal>_periods.csv")
