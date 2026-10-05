"""Phase 4: compare equal-weight, linear regression and LightGBM factor combinations.

Evaluation window: PRED_START to the last in-sample month (hold-out still untouched).
Usage: python scripts/30_combine_factors.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (DATA_PROCESSED, FACTOR_PRIORS, HOLDOUT_START, LGBM_PARAMS,  # noqa: E402
                    N_QUANTILES, PASSED_FACTORS, PRED_START, REPORTS, RETRAIN_EVERY,
                    TRAIN_START, VALID_MONTHS)
from src.evaluation.factor_tests import test_factor  # noqa: E402
from src.models.combine import add_target, equal_weight, walk_forward  # noqa: E402

pd.set_option("display.width", 220)
features = list(FACTOR_PRIORS)
panel = add_target(pd.read_parquet(DATA_PROCESSED / "factors_monthly.parquet"))

in_sample = sorted(d for d in panel["date"].unique() if d < pd.Timestamp(HOLDOUT_START))
pred_dates = [d for d in in_sample[:-1] if d >= pd.Timestamp(PRED_START)]
print(f"Prediction months: {pd.Timestamp(pred_dates[0]).date()} to {pd.Timestamp(pred_dates[-1]).date()} "
      f"({len(pred_dates)}), expanding-window refit every {RETRAIN_EVERY} months")

preds, coefs, imp = walk_forward(panel, features, pred_dates, RETRAIN_EVERY, VALID_MONTHS,
                                 LGBM_PARAMS, TRAIN_START)

ev = panel[panel["date"].isin(pred_dates)][["date", "symbol", "fwd_ret", *features]].copy()
ev["ew_passed"] = equal_weight(ev, PASSED_FACTORS)
ev["ew_all"] = equal_weight(ev, features)
ev = ev.merge(preds, on=["date", "symbol"], how="left")
ev[["date", "symbol", "fwd_ret", "ew_passed", "ew_all", "linear", "lgbm"]].to_parquet(
    DATA_PROCESSED / "predictions_insample.parquet", index=False)

rows, qrets, rics = {}, {}, {}
for name in ["ew_passed", "ew_all", "linear", "lgbm"]:
    rows[name], rics[name], qrets[name] = test_factor(ev, name, N_QUANTILES)
res = pd.DataFrame(rows).T
res.to_csv(REPORTS / "phase4_combination.csv")

show = res.drop(columns="months").copy()
for c in ["Q1", "Q2", "Q3", "Q4", "Q5", "LS_ann", "LS_vol", "RankIC>0"]:
    show[c] = (show[c] * 100).round(2)
for c in ["IC", "RankIC", "RankIC_std"]:
    show[c] = show[c].round(4)
for c in ["ICIR", "t", "LS_sharpe", "monotonic", "autocorr"]:
    show[c] = show[c].astype(float).round(2)
print("\n== Combined signals over the same months (returns in %, annualised excess over "
      "equal-weighted universe, before costs) ==")
print(show.to_string())
print("\nNote: ew_passed uses factors selected on 2016-2023 results, so it carries "
      "selection bias inside this window; ew_all, linear and lgbm use all 10 factors.")

print("\n== Linear regression coefficients at each refit ==")
print(coefs.round(4).to_string())
print("\n== LightGBM gain importance share at each refit ==")
print(imp.round(3).to_string())

ric = pd.DataFrame(rics)
print("\n== Mean Rank IC by year ==")
print(ric.groupby(ric.index.year).mean().round(3).to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
ric.cumsum().plot(ax=axes[0], title="Cumulative Rank IC")
ls = pd.DataFrame({k: v[N_QUANTILES] - v[1] for k, v in qrets.items()})
(1 + ls).cumprod().plot(ax=axes[1], title="Cumulative Q5 - Q1 (before costs)")
fig.tight_layout()
fig.savefig(REPORTS / "phase4_combination.png", dpi=150)
print(f"\nSaved: phase4_combination.csv, phase4_combination.png, predictions_insample.parquet")
