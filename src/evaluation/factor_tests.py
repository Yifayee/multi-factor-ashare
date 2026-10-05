"""Single-factor tests: IC, Rank IC, ICIR, quantile portfolios.

All factors are already multiplied by their prior sign, so a positive IC means the
factor works in the direction assumed before testing.
"""
import numpy as np
import pandas as pd


def ic_series(panel: pd.DataFrame, factor: str, method: str = "pearson") -> pd.Series:
    """Cross-sectional correlation between factor and next-month return, per date."""
    d = panel[["date", factor, "fwd_ret"]].dropna()
    if method == "spearman":
        d[[factor, "fwd_ret"]] = d.groupby("date")[[factor, "fwd_ret"]].rank()
    return d.groupby("date").apply(lambda g: g[factor].corr(g["fwd_ret"]), include_groups=False)


def ic_summary(ic: pd.Series) -> dict:
    n = ic.count()
    mean, sd = ic.mean(), ic.std()
    return {"mean": mean, "std": sd, "icir": mean / sd, "t": mean / sd * np.sqrt(n),
            "pos_ratio": (ic > 0).mean(), "n_months": n}


def quantile_returns(panel: pd.DataFrame, factor: str, q: int) -> pd.DataFrame:
    """Equal-weighted next-month return of each factor quantile (1 = lowest, q = highest),
    in excess of the equal-weighted universe return that month."""
    d = panel[["date", factor, "fwd_ret"]].dropna().copy()
    d["bucket"] = d.groupby("date")[factor].transform(
        lambda x: pd.qcut(x.rank(method="first"), q, labels=False) + 1)
    ret = d.groupby(["date", "bucket"])["fwd_ret"].mean().unstack()
    market = d.groupby("date")["fwd_ret"].mean()
    return ret.sub(market, axis=0)


def annualise(monthly: pd.Series) -> float:
    m = monthly.dropna()
    return (1 + m).prod() ** (12 / len(m)) - 1 if len(m) else np.nan


def factor_autocorr(panel: pd.DataFrame, factor: str) -> float:
    """Average rank correlation of a stock's factor value between consecutive months.
    Low values mean the portfolio would turn over heavily."""
    wide = panel.pivot(index="date", columns="symbol", values=factor)
    r = wide.rank(axis=1)
    corrs = [r.iloc[i].corr(r.iloc[i - 1]) for i in range(1, len(r))]
    return float(np.nanmean(corrs))


def test_factor(panel: pd.DataFrame, factor: str, q: int) -> tuple:
    ic = ic_series(panel, factor, "pearson")
    ric = ic_series(panel, factor, "spearman")
    qret = quantile_returns(panel, factor, q)
    ls = qret[q] - qret[1]
    q_ann = {f"Q{i}": annualise(qret[i]) for i in range(1, q + 1)}
    mono = pd.Series(list(q_ann.values())).corr(pd.Series(range(1, q + 1)), method="spearman")
    s_ic, s_ric = ic_summary(ic), ic_summary(ric)
    row = {
        "IC": s_ic["mean"], "RankIC": s_ric["mean"], "RankIC_std": s_ric["std"],
        "ICIR": s_ric["icir"], "t": s_ric["t"], "RankIC>0": s_ric["pos_ratio"],
        **q_ann,
        "LS_ann": annualise(ls), "LS_vol": ls.std() * np.sqrt(12),
        "LS_sharpe": ls.mean() / ls.std() * np.sqrt(12), "monotonic": mono,
        "autocorr": factor_autocorr(panel, factor), "months": s_ric["n_months"],
    }
    return row, ric, qret
