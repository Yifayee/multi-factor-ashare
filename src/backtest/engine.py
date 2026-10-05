"""Monthly long-only backtest.

Timing: scores use data up to the close of rebalance date t; trades execute at the close of
the next trading day (e0). Holdings are then kept until the next execution day (e1).

Trading rules
- Cannot buy a stock that is suspended or at limit-up on e0.
- Cannot sell a stock that is suspended or at limit-down on e0; it stays in the portfolio.
- A stock that has stopped trading for good is sold at its last traded price.
Costs: commission + slippage on both sides, stamp duty on sells.
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (COMMISSION, DATA_RAW, LIMIT_TOL, SLIPPAGE, STAMP_CHANGE_DATE,
                    STAMP_DUTY_NEW, STAMP_DUTY_OLD)
from src.data.utils import to_ns


def price_limit(symbol: str, date: pd.Timestamp) -> float:
    if symbol.startswith("68"):
        return 0.20
    if symbol.startswith("30") and date >= pd.Timestamp("2020-08-24"):
        return 0.20
    return 0.10


def execution_dates(rebal_dates, calendar: pd.Series) -> list:
    cal = pd.DatetimeIndex(sorted(calendar))
    return [cal[cal > d][0] for d in rebal_dates]


def load_exec_data(symbols, exec_dates) -> dict:
    """Per stock, on each execution date: last price (hfq, carried forward), traded flag,
    limit-up / limit-down flags, and whether the stock has stopped trading for good."""
    ed = pd.DatetimeIndex(exec_dates)
    px, traded, up, down, dead = {}, {}, {}, {}, {}
    for s in tqdm(symbols, desc="loading prices"):
        raw = pd.read_parquet(DATA_RAW / "prices" / "raw" / f"{s}.parquet", columns=["date", "close"])
        hfq = pd.read_parquet(DATA_RAW / "prices" / "hfq" / f"{s}.parquet", columns=["date", "close"])
        raw["date"], hfq["date"] = to_ns(raw["date"]), to_ns(hfq["date"])
        raw = raw.sort_values("date").set_index("date")["close"]
        h = hfq.sort_values("date").set_index("date")["close"]
        chg = raw / raw.shift(1) - 1
        lim = pd.Series([price_limit(s, d) for d in chg.index], index=chg.index)
        px[s] = h.reindex(h.index.union(ed)).ffill().reindex(ed).values
        traded[s] = ed.isin(h.index)
        up[s] = (chg >= lim - LIMIT_TOL).reindex(ed, fill_value=False).values
        down[s] = (chg <= -lim + LIMIT_TOL).reindex(ed, fill_value=False).values
        dead[s] = ed > h.index.max()
    to_df = lambda d: pd.DataFrame(d, index=ed)  # noqa: E731
    return {"px": to_df(px), "traded": to_df(traded), "up": to_df(up), "down": to_df(down),
            "dead": to_df(dead)}


def run_backtest(scores: pd.DataFrame, signal: str, rebal: list, exec_d: list, mkt: dict,
                 top_frac: float) -> pd.DataFrame:
    """scores: columns date, symbol, <signal>. Returns one row per holding period."""
    w = pd.Series(dtype=float)  # current weights, sum to 1 when invested
    out = []
    for i in range(len(rebal) - 1):
        t, e0, e1 = rebal[i], exec_d[i], exec_d[i + 1]
        sc = scores[scores["date"] == t].set_index("symbol")[signal].dropna()
        target = set(sc[sc >= sc.quantile(1 - top_frac)].index)

        traded, up, down, dead = (mkt[k].loc[e0] for k in ("traded", "up", "down", "dead"))
        held = set(w.index)
        can_sell = {s for s in held if (traded.get(s, False) and not down.get(s, False)) or dead.get(s, False)}
        stuck = {s for s in held if s not in can_sell}
        buyable = {s for s in target - stuck if traded.get(s, False) and (not up.get(s, False) or s in held)}

        stuck_w = w.reindex(list(stuck)).fillna(0.0)
        free = 1.0 - stuck_w.sum()
        new_w = pd.concat([stuck_w, pd.Series(free / len(buyable), index=sorted(buyable))]) \
            if buyable else stuck_w
        new_w = new_w.groupby(level=0).sum()

        allsym = new_w.index.union(w.index)
        diff = new_w.reindex(allsym, fill_value=0.0) - w.reindex(allsym, fill_value=0.0)
        buys, sells = diff.clip(lower=0).sum(), (-diff).clip(lower=0).sum()
        stamp = STAMP_DUTY_NEW if e0 >= pd.Timestamp(STAMP_CHANGE_DATE) else STAMP_DUTY_OLD
        cost = buys * (COMMISSION + SLIPPAGE) + sells * (COMMISSION + SLIPPAGE + stamp)

        p0, p1 = mkt["px"].loc[e0, new_w.index], mkt["px"].loc[e1, new_w.index]
        r = (p1 / p0 - 1).fillna(0.0)
        gross = float((new_w * r).sum())
        net = (1 - cost) * (1 + gross) - 1
        w = new_w * (1 + r) / (1 + gross)

        out.append({"date": e1, "start": e0, "gross": gross, "net": net, "cost": cost,
                    "turnover": 0.5 * (buys + sells), "n_hold": len(new_w), "n_stuck": len(stuck),
                    "n_target_blocked": len(target - buyable - stuck)})
    return pd.DataFrame(out).set_index("date")


def ew_universe(scores: pd.DataFrame, rebal: list, exec_d: list, mkt: dict) -> pd.Series:
    """Equal-weighted return of every stock in the factor universe, no costs or limits."""
    rets = {}
    for i in range(len(rebal) - 1):
        e0, e1 = exec_d[i], exec_d[i + 1]
        syms = [s for s in scores.loc[scores["date"] == rebal[i], "symbol"] if mkt["traded"].loc[e0].get(s, False)]
        r = mkt["px"].loc[e1, syms] / mkt["px"].loc[e0, syms] - 1
        rets[e1] = r.mean()
    return pd.Series(rets)


def index_returns(path, exec_d: list) -> pd.Series:
    idx = pd.read_parquet(path)
    idx["date"] = to_ns(idx["date"])
    c = idx.set_index("date")["close"].sort_index()
    ed = pd.DatetimeIndex(exec_d)
    p = c.reindex(c.index.union(ed)).ffill().reindex(ed)
    return pd.Series(p.values[1:] / p.values[:-1] - 1, index=ed[1:])


def metrics(r: pd.Series, bench: pd.Series = None) -> dict:
    r = r.dropna()
    nav = (1 + r).cumprod()
    out = {"ann_ret": nav.iloc[-1] ** (12 / len(r)) - 1,
           "ann_vol": r.std() * np.sqrt(12),
           "sharpe": r.mean() / r.std() * np.sqrt(12),
           "max_dd": (nav / nav.cummax() - 1).min()}
    if bench is not None:
        a = (r - bench.reindex(r.index)).dropna()
        out["excess_ann"] = out["ann_ret"] - ((1 + bench.reindex(r.index)).prod() ** (12 / len(r)) - 1)
        out["info_ratio"] = a.mean() / a.std() * np.sqrt(12)
        out["win_rate"] = (a > 0).mean()
    return out


def yearly_by_start(series: dict, exec_d: list) -> pd.DataFrame:
    """Compound period returns into years by each period's START date, so a year means
    periods that begin in that year (e.g. 2 Jan to 2 Jan of the next year)."""
    start_of = dict(zip(pd.DatetimeIndex(exec_d[1:]), pd.DatetimeIndex(exec_d[:-1])))
    df = pd.DataFrame(series)
    years = [start_of[d].year for d in df.index]
    return df.groupby(years).apply(lambda x: (1 + x).prod() - 1)
