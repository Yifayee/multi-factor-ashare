"""Build the monthly factor panel: one row per (rebalance date, stock)."""
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (DATA_PROCESSED, DATA_RAW, FACTOR_PRIORS, MAD_K, MIN_LISTED_DAYS,
                    PRICE_END, PRICE_START)
from src.data.fundamentals import build_pit_fundamentals, frontier
from src.data.utils import logger, to_ns
from src.factors.preprocess import process_cross_section


def rebalance_dates() -> pd.DatetimeIndex:
    cal = to_ns(pd.read_parquet(DATA_RAW / "trade_calendar.parquet")["trade_date"])
    cal = cal[(cal >= PRICE_START) & (cal <= PRICE_END)]
    return pd.DatetimeIndex(cal.groupby(cal.dt.to_period("M")).max().sort_values())


def stock_snapshots(symbol: str, rebal: pd.DatetimeIndex, report_periods) -> tuple:
    """Price-based factors and forward returns for one stock at each rebalance date it traded."""
    raw = pd.read_parquet(DATA_RAW / "prices" / "raw" / f"{symbol}.parquet",
                          columns=["date", "close", "amount", "turnover"])
    hfq = pd.read_parquet(DATA_RAW / "prices" / "hfq" / f"{symbol}.parquet", columns=["date", "close"])
    d = raw.merge(hfq.rename(columns={"close": "close_hfq"}), on="date").sort_values("date")
    d["date"] = to_ns(d["date"])
    d = d.set_index("date")
    px = d["close_hfq"]
    ret = px.pct_change()
    d["k"] = d["close_hfq"] / d["close"]  # adjustment ratio, used to align per-share data
    mcap = d["amount"] / (d["turnover"].where(d["turnover"] > 0) / 100)
    d["size_raw"] = np.log(mcap.where(mcap > 0))
    d["mom_12_1_raw"] = px.shift(21) / px.shift(252) - 1
    d["rev_1m_raw"] = px / px.shift(21) - 1
    d["vol_60d_raw"] = ret.rolling(60, min_periods=40).std()
    d["turn_20d_raw"] = d["turnover"].rolling(20, min_periods=15).mean()
    d["n_days"] = np.arange(1, len(d) + 1)

    # Forward return to the next rebalance date, using the last traded price on or before it.
    # A stock that is suspended or delisted during the month keeps its last traded price.
    p_asof = px.reindex(px.index.union(rebal)).ffill().reindex(rebal)
    next_p = pd.Series(p_asof.values, index=rebal).shift(-1)

    snap = d[d.index.isin(rebal)].copy()
    snap["fwd_ret"] = next_p.reindex(snap.index).values / snap["close_hfq"].values - 1
    snap["symbol"] = symbol
    cols = ["symbol", "close", "k", "size_raw", "mom_12_1_raw", "rev_1m_raw", "vol_60d_raw",
            "turn_20d_raw", "n_days", "fwd_ret"]
    snap = snap[cols].reset_index()

    k_rep = pd.DataFrame({"symbol": symbol, "report_period": pd.to_datetime(sorted(report_periods))})
    if len(k_rep):
        k_rep["k_rep"] = d["k"].reindex(d.index.union(k_rep["report_period"])).ffill().bfill() \
            .reindex(k_rep["report_period"]).values  # bfill: reports dated before our price history
    return snap, k_rep


def build_factor_panel() -> pd.DataFrame:
    rebal = rebalance_dates()
    uni = pd.read_parquet(DATA_RAW / "universe.parquet")
    symbols = sorted(p.stem for p in (DATA_RAW / "prices" / "raw").glob("*.parquet"))
    fund = build_pit_fundamentals(set(symbols))
    periods = fund.groupby("symbol")["report_period"].apply(list).to_dict()

    snaps, kreps = [], []
    for sym in tqdm(symbols, desc="stocks"):
        s, k = stock_snapshots(sym, rebal, periods.get(sym, []))
        snaps.append(s)
        kreps.append(k)
    panel = pd.concat(snaps, ignore_index=True).sort_values(["date", "symbol"])
    kreps = pd.concat(kreps, ignore_index=True)

    # Latest available report strictly before each rebalance date
    fr = frontier(fund).merge(kreps, on=["symbol", "report_period"], how="left")
    fcols = ["symbol", "avail_date", "report_period", "bps", "roe", "accruals", "growth", "equity", "k_rep"]
    fr = fr[fcols].rename(columns={"roe": "roe_rep", "accruals": "accruals_rep", "growth": "growth_rep"})
    panel = pd.merge_asof(panel.sort_values("date"), fr.sort_values("avail_date"),
                          left_on="date", right_on="avail_date", by="symbol",
                          allow_exact_matches=False)

    # Book-to-price in the share units of the report date: price per old share = close * k_t / k_rep
    px_rep_units = panel["close"] * panel["k"] / panel["k_rep"]
    panel["bp_raw"] = panel["bps"] / px_rep_units
    panel["roe_raw"] = panel["roe_rep"]
    panel["ep_raw"] = panel["roe_rep"] * panel["bp_raw"]  # E/P = (E/B) * (B/P)
    panel["accruals_raw"] = panel["accruals_rep"]
    panel["growth_raw"] = panel["growth_rep"]

    # Latest available annual report for the loss proxy
    ann = fund[fund["report_period"].dt.month == 12][["symbol", "avail_date", "loss2y"]]
    panel = pd.merge_asof(panel.sort_values("date"), ann.sort_values("avail_date").rename(
        columns={"avail_date": "ann_avail"}), left_on="date", right_on="ann_avail", by="symbol",
        allow_exact_matches=False)

    # Point-in-time SW level-1 industry
    ind = pd.read_parquet(DATA_RAW / "industry_sw_hist.parquet")[["symbol", "start_date", "industry_code"]]
    ind["start_date"] = to_ns(ind["start_date"])
    ind = ind.dropna(subset=["start_date"]).sort_values("start_date")
    ind["industry"] = ind["industry_code"].astype(str).str[:2]
    panel = pd.merge_asof(panel.sort_values("date"), ind[["symbol", "start_date", "industry"]],
                          left_on="date", right_on="start_date", by="symbol")
    panel["industry"] = panel["industry"].fillna("NA")

    # Universe filters
    st_proxy = (panel["equity"] <= 0) | (panel["loss2y"] == True)  # noqa: E712
    keep = (panel["n_days"] >= MIN_LISTED_DAYS) & panel["size_raw"].notna() & ~st_proxy
    logger.info("Filters: young=%d, no mcap=%d, ST proxy=%d of %d rows",
                (panel["n_days"] < MIN_LISTED_DAYS).sum(), panel["size_raw"].isna().sum(),
                st_proxy.sum(), len(panel))
    panel = panel[keep].copy()

    raw_cols = [f"{f}_raw" for f in FACTOR_PRIORS]
    missing = panel.groupby(panel["date"].dt.year)[raw_cols].apply(lambda x: x.isna().mean())

    processed = []
    for _, g in tqdm(panel.groupby("date"), desc="cross-sections"):
        processed.append(process_cross_section(g, FACTOR_PRIORS, MAD_K))
    panel = panel.join(pd.concat(processed))

    keep_cols = ["date", "symbol", "industry", "fwd_ret", *raw_cols, *FACTOR_PRIORS]
    out = panel[keep_cols].sort_values(["date", "symbol"]).reset_index(drop=True)
    out.to_parquet(DATA_PROCESSED / "factors_monthly.parquet", index=False)
    missing.to_csv(DATA_PROCESSED / "factor_missing_by_year.csv")
    return out
