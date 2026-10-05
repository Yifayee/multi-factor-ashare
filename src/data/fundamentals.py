"""Point-in-time fundamentals: one row per (stock, report period) with the date it became public.

Availability rule (from scripts/06_check_announcements.py):
- Only the balance-sheet table's date for Q1/H1/Q3 looks like a first-disclosure date;
  all statements of one report are published together, so that date is used for every field.
- If that date is missing or later than the statutory deadline, the deadline is used.
- Annual reports: no reliable date in the free data, so the deadline (Apr 30 next year) is used.
A report may be used on a rebalance date only if avail_date is strictly earlier.
"""
import numpy as np
import pandas as pd

from config import DATA_RAW
from src.data.utils import logger, to_ns

FIELDS = {
    "zcfz": {"股东权益合计": "equity", "资产-总资产": "assets", "公告日期": "ann_bs"},
    "lrb": {"净利润": "np"},
    "xjll": {"经营性现金流-现金流量净额": "ocf"},
    "yjbb": {"每股净资产": "bps", "净利润-同比增长": "np_yoy"},
}
FLOWS = ["np", "ocf"]  # year-to-date flows that need TTM conversion


def _load_table(name: str, cols: dict, universe: set) -> pd.DataFrame:
    files = sorted((DATA_RAW / "financials" / name).glob("*.parquet"))
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df["symbol"] = df["股票代码"].astype(str).str.zfill(6)
    df = df[df["symbol"].isin(universe)]
    df = df[["symbol", "report_period", *cols]].rename(columns=cols)
    df["report_period"] = to_ns(df["report_period"])
    for c in cols.values():
        if c != "ann_bs":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.drop_duplicates(["symbol", "report_period"], keep="last")


def _deadline(period: pd.Series) -> pd.Series:
    y, m = period.dt.year, period.dt.month
    out = pd.Series(pd.NaT, index=period.index, dtype="datetime64[ns]")
    out[m == 3] = pd.to_datetime(y[m == 3].astype(str) + "-04-30")
    out[m == 6] = pd.to_datetime(y[m == 6].astype(str) + "-08-31")
    out[m == 9] = pd.to_datetime(y[m == 9].astype(str) + "-10-31")
    out[m == 12] = pd.to_datetime((y[m == 12] + 1).astype(str) + "-04-30")
    return out


def _add_ttm(df: pd.DataFrame) -> pd.DataFrame:
    """TTM = YTD(P) + Annual(Y-1) - YTD(same quarter, Y-1); for annual reports TTM = YTD."""
    df = df.copy()
    df["year"], df["month"] = df["report_period"].dt.year, df["report_period"].dt.month
    prev_same = df[["symbol", "year", "month", *FLOWS]].copy()
    prev_same["year"] += 1
    prev_annual = df[df["month"] == 12][["symbol", "year", *FLOWS]].copy()
    prev_annual["year"] += 1
    df = df.merge(prev_same, on=["symbol", "year", "month"], how="left", suffixes=("", "_ly"))
    df = df.merge(prev_annual, on=["symbol", "year"], how="left", suffixes=("", "_lya"))
    for f in FLOWS:
        df[f"{f}_ttm"] = np.where(df["month"] == 12, df[f], df[f] + df[f"{f}_lya"] - df[f"{f}_ly"])
    return df.drop(columns=[c for c in df.columns if c.endswith(("_ly", "_lya"))])


def build_pit_fundamentals(universe: set) -> pd.DataFrame:
    tables = [_load_table(n, c, universe) for n, c in FIELDS.items()]
    df = tables[0]
    for t in tables[1:]:
        df = df.merge(t, on=["symbol", "report_period"], how="outer")
    df = _add_ttm(df)

    deadline = _deadline(df["report_period"])
    ann = pd.to_datetime(df["ann_bs"], errors="coerce")
    use_ann = (df["month"] != 12) & ann.notna() & (ann <= deadline) & (ann > df["report_period"])
    df["avail_date"] = np.where(use_ann, ann, deadline)
    df["avail_date"] = to_ns(df["avail_date"])
    logger.info("Fundamentals: %d reports, %.1f%% dated by announcement, rest by statutory deadline",
                len(df), 100 * use_ann.mean())

    # Derived ratios at report level
    eq = df["equity"].where(df["equity"] > 0)
    df["roe"] = df["np_ttm"] / eq
    df["accruals"] = (df["np_ttm"] - df["ocf_ttm"]) / df["assets"].where(df["assets"] > 0)
    df["growth"] = df["np_yoy"]

    # Two consecutive annual losses or negative equity: proxy for ST status (no free history)
    annual = df[df["month"] == 12].sort_values(["symbol", "year"])
    annual["loss2y"] = (annual["np"] < 0) & (annual.groupby("symbol")["np"].shift(1) < 0)
    df = df.merge(annual[["symbol", "report_period", "loss2y"]], on=["symbol", "report_period"], how="left")
    return df.sort_values(["symbol", "avail_date", "report_period"]).reset_index(drop=True)


def frontier(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only reports that move a stock's latest available period forward.
    Prevents an older report that becomes available later (e.g. annual on Apr 30)
    from replacing a newer one (Q1 published Apr 25)."""
    df = df.sort_values(["symbol", "avail_date", "report_period"])
    latest = df.groupby("symbol")["report_period"].cummax()
    return df[df["report_period"] == latest]
