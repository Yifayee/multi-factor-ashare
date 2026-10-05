"""Final step: run the fixed pipeline ONCE on the 2024-2025 hold-out period.

Nothing here is tuned: factor definitions, signs, combination methods, model parameters
and backtest rules are exactly those decided on 2016-2023. Report the result as it comes out.
Usage: python scripts/50_holdout.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATA_PROCESSED, DATA_RAW, FACTOR_PRIORS, HOLDOUT_START, LGBM_PARAMS,  # noqa: E402
                    N_QUANTILES, REPORTS, RETRAIN_EVERY, TOP_FRACTION, TRAIN_START, VALID_MONTHS)
from src.backtest.engine import (ew_universe, execution_dates, index_returns,  # noqa: E402
                                 load_exec_data, metrics, run_backtest, yearly_by_start)
from src.data.utils import to_ns  # noqa: E402
from src.evaluation.factor_tests import ic_series, ic_summary, test_factor  # noqa: E402
from src.models.combine import add_target, equal_weight, walk_forward  # noqa: E402

pd.set_option("display.width", 200)
features = list(FACTOR_PRIORS)
panel = add_target(pd.read_parquet(DATA_PROCESSED / "factors_monthly.parquet"))
panel["date"] = to_ns(panel["date"])
dates = sorted(panel["date"].unique())

# First hold-out signal: last in-sample rebalance date (its return is earned in January 2024)
first = max(d for d in dates if d < pd.Timestamp(HOLDOUT_START))
signal_dates = [d for d in dates if d >= first][:-1]   # last date has no next-month return
end_date = dates[-1]
print(f"Hold-out signals: {pd.Timestamp(signal_dates[0]).date()} to {pd.Timestamp(signal_dates[-1]).date()} "
      f"({len(signal_dates)} months), valuation through {pd.Timestamp(end_date).date()}")

# 1. Single factors: Rank IC in hold-out vs in-sample
ins = pd.read_csv(REPORTS / "phase3_factor_tests.csv", index_col=0)
hp = panel[panel["date"].isin(signal_dates)]
rows = {}
for f in features:
    s = ic_summary(ic_series(hp, f, "spearman"))
    rows[f] = {"RankIC_insample": ins.loc[f, "RankIC"], "RankIC_holdout": s["mean"],
               "t_holdout": s["t"], "passed_insample": ins.loc[f, "pass"]}
fac = pd.DataFrame(rows).T
print("\n== Single-factor Rank IC: in-sample (2016-2023) vs hold-out ==")
print(fac.round(4).to_string())

# 2. Combined signals (LightGBM keeps the same expanding-window rule)
preds, _, imp = walk_forward(panel, features, signal_dates, RETRAIN_EVERY, VALID_MONTHS,
                             LGBM_PARAMS, TRAIN_START)
ev = hp[["date", "symbol", "fwd_ret", *features]].copy()
ev["ew_all"] = equal_weight(ev, features)
ev = ev.merge(preds[["date", "symbol", "lgbm"]], on=["date", "symbol"], how="left")
comb = pd.DataFrame({k: test_factor(ev, k, N_QUANTILES)[0] for k in ["ew_all", "lgbm"]}).T
print("\n== Combined signals in hold-out ==")
print(comb[["RankIC", "ICIR", "t", "RankIC>0", "LS_ann", "monotonic"]].astype(float).round(4).to_string())

# 3. Backtest with the same rules
cal = to_ns(pd.read_parquet(DATA_RAW / "trade_calendar.parquet")["trade_date"])
rebal = list(signal_dates) + [end_date]
exec_d = execution_dates(signal_dates, cal) + [pd.Timestamp(end_date)]
scores = ev[["date", "symbol", "ew_all", "lgbm"]]
mkt = load_exec_data(sorted(scores["symbol"].unique()), exec_d)

universe = ew_universe(scores, rebal, exec_d, mkt)
csi500 = index_returns(DATA_RAW / "benchmarks" / "csi500.parquet", exec_d)
csi1000 = index_returns(DATA_RAW / "benchmarks" / "csi1000.parquet", exec_d)
results, series = {}, {}
for sig in ["ew_all", "lgbm"]:
    bt = run_backtest(scores, sig, rebal, exec_d, mkt, TOP_FRACTION)
    bt.to_csv(REPORTS / f"holdout_{sig}_periods.csv")
    series[f"{sig} (net)"] = bt["net"]
    m = metrics(bt["net"], universe)
    m["vs_csi500"] = m["ann_ret"] - metrics(csi500)["ann_ret"]
    m["vs_csi1000"] = m["ann_ret"] - metrics(csi1000)["ann_ret"]
    m["turnover_mo"], m["cost_ann"] = bt["turnover"].mean(), bt["cost"].mean() * 12
    results[f"{sig} (net)"] = m
for name, r in [("EW universe", universe), ("CSI 500", csi500), ("CSI 1000", csi1000)]:
    results[name], series[name] = metrics(r), r

res = pd.DataFrame(results).T
res.to_csv(REPORTS / "holdout_backtest.csv")
show = res.copy()
for c in ["ann_ret", "ann_vol", "max_dd", "excess_ann", "vs_csi500", "vs_csi1000", "win_rate",
          "turnover_mo", "cost_ann"]:
    show[c] = (show[c] * 100).round(2)
for c in ["sharpe", "info_ratio"]:
    show[c] = show[c].astype(float).round(2)
print("\n== Hold-out backtest, net of costs (in %; excess_ann / info_ratio / win_rate vs EW universe) ==")
print(show.to_string())

yearly = yearly_by_start(series, exec_d)
print("\n== Hold-out returns by year (%) ==")
print((yearly * 100).round(2).to_string())

fac.to_csv(REPORTS / "holdout_factor_ic.csv")
comb.to_csv(REPORTS / "holdout_combined.csv")
yearly.to_csv(REPORTS / "holdout_yearly.csv")
fig, ax = plt.subplots(figsize=(10, 4.5))
(1 + pd.DataFrame(series)).cumprod().plot(ax=ax, title="Hold-out net asset value (2024-2025)")
ax.axhline(1, color="grey", lw=0.8)
fig.tight_layout()
fig.savefig(REPORTS / "holdout_nav.png", dpi=150)
print("\nSaved: holdout_factor_ic.csv, holdout_combined.csv, holdout_backtest.csv, "
      "holdout_yearly.csv, holdout_nav.png")
