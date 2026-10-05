"""Factor combination: equal weight, linear regression, LightGBM.

Target: cross-sectional percentile rank of next-month return (removes the market
move and dampens outliers). Training uses an expanding window that only contains
rebalance dates strictly before the prediction date, so every training label is
already realised at prediction time.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def add_target(panel: pd.DataFrame) -> pd.DataFrame:
    panel = panel.copy()
    panel["target"] = panel.groupby("date")["fwd_ret"].rank(pct=True)
    return panel


def equal_weight(panel: pd.DataFrame, factors: list) -> pd.Series:
    return panel[factors].mean(axis=1)


def walk_forward(panel: pd.DataFrame, features: list, pred_dates, retrain_every: int,
                 valid_months: int, lgbm_params: dict, train_start: str):
    """Returns predictions for each date in pred_dates plus fitted-model diagnostics."""
    data = panel[panel["date"] >= train_start].dropna(subset=["target"])
    all_dates = sorted(data["date"].unique())
    preds, coefs, importances = [], [], []
    lin = gbm = None
    for i, t in enumerate(pred_dates):
        if i % retrain_every == 0:
            train_dates = [d for d in all_dates if d < t]
            train = data[data["date"].isin(train_dates)]
            X, y = train[features].values, train["target"].values

            lin = LinearRegression().fit(X, y)
            coefs.append(pd.Series(lin.coef_, index=features, name=pd.Timestamp(t)))

            valid_dates = set(train_dates[-valid_months:])
            is_val = train["date"].isin(valid_dates).values
            gbm = lgb.LGBMRegressor(**lgbm_params)
            gbm.fit(X[~is_val], y[~is_val], eval_set=[(X[is_val], y[is_val])],
                    callbacks=[lgb.early_stopping(50, verbose=False)])
            importances.append(pd.Series(gbm.booster_.feature_importance("gain"),
                                         index=features, name=pd.Timestamp(t)))
            print(f"  refit at {pd.Timestamp(t).date()}: {len(train_dates)} months, "
                  f"{len(train)} rows, lgbm best iteration {gbm.best_iteration_}")
        cur = panel[panel["date"] == t]
        preds.append(pd.DataFrame({
            "date": t, "symbol": cur["symbol"].values,
            "linear": lin.predict(cur[features].values),
            "lgbm": gbm.predict(cur[features].values, num_iteration=gbm.best_iteration_),
        }))
    imp = pd.DataFrame(importances)
    imp = imp.div(imp.sum(axis=1), axis=0)
    return pd.concat(preds, ignore_index=True), pd.DataFrame(coefs), imp
