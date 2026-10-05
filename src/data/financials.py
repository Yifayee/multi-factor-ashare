"""Quarterly financial statements with announcement dates (point-in-time source).

Each call returns ALL stocks for one report period, which is far cheaper than per-stock calls.
Raw Chinese column names are kept here; standardisation happens in Phase 2 after inspection.
"""
import pandas as pd
import akshare as ak

from config import DATA_RAW, FIN_END_YEAR, FIN_START_YEAR
from src.data.utils import logger, save_parquet, with_retry

TABLES = {
    "yjbb": ak.stock_yjbb_em,   # EPS, revenue, net profit, ROE, BPS, gross margin, announcement date
    "zcfz": ak.stock_zcfz_em,   # balance sheet summary
    "lrb": ak.stock_lrb_em,     # income statement summary
    "xjll": ak.stock_xjll_em,   # cash flow summary
}


def report_periods():
    for year in range(FIN_START_YEAR, FIN_END_YEAR + 1):
        for mmdd in ("0331", "0630", "0930", "1231"):
            yield f"{year}{mmdd}"


def download_financials() -> list:
    failed = []
    for name, fn in TABLES.items():
        for period in report_periods():
            path = DATA_RAW / "financials" / name / f"{period}.parquet"
            if path.exists():
                continue
            try:
                df = with_retry(fn, date=period)
            except Exception as exc:
                logger.error("%s %s failed: %s", name, period, exc)
                failed.append((name, period))
                continue
            if df is None or df.empty:
                logger.warning("%s %s empty (report not published yet?)", name, period)
                continue
            df["report_period"] = pd.Timestamp(period)
            save_parquet(df, path)
    logger.info("Financials done, failed=%d", len(failed))
    return failed
