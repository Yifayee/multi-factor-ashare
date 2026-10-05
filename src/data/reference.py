"""Reference data: trading calendar, SW industry history, benchmark indices."""
import pandas as pd
import akshare as ak

from config import BENCHMARKS, DATA_RAW
from src.data.utils import logger, save_parquet, with_retry

SW_URL = "https://www.swsresearch.com/swindex/pdf/SwClass2021/StockClassifyUse_stock.xls"
SW_MANUAL = DATA_RAW / "manual" / "StockClassifyUse_stock.xls"


def download_calendar() -> pd.DataFrame:
    cal = with_retry(ak.tool_trade_date_hist_sina)
    cal["trade_date"] = pd.to_datetime(cal["trade_date"])
    save_parquet(cal, DATA_RAW / "trade_calendar.parquet")
    logger.info("Calendar: %d trading days", len(cal))
    return cal


def _read_sw_excel(path) -> pd.DataFrame:
    """Same parsing as akshare.stock_industry_clf_hist_sw, applied to a local file."""
    df = pd.read_excel(path, dtype={"股票代码": "str", "行业代码": "str"})
    return df.rename(columns={"股票代码": "symbol", "计入日期": "start_date",
                              "行业代码": "industry_code", "更新日期": "update_time"})


def download_industry() -> pd.DataFrame:
    """Shenwan 2021 classification with change history (start_date per assignment).

    The SWS site's certificate chain fails verification in some Python setups,
    so a manually downloaded copy of the same file is used as fallback.
    """
    try:
        ind = with_retry(ak.stock_industry_clf_hist_sw, retries=2)
        source = "akshare"
    except Exception as exc:
        if not SW_MANUAL.exists():
            raise FileNotFoundError(
                f"akshare failed ({type(exc).__name__}). Download {SW_URL} in a browser "
                f"and save it as {SW_MANUAL}") from exc
        ind = _read_sw_excel(SW_MANUAL)
        source = "manual file"
    ind["symbol"] = ind["symbol"].astype(str).str.zfill(6)
    ind["start_date"] = pd.to_datetime(ind["start_date"], errors="coerce")
    save_parquet(ind, DATA_RAW / "industry_sw_hist.parquet")
    logger.info("Industry (%s): %d records, %d stocks", source, len(ind), ind["symbol"].nunique())
    return ind


def download_benchmarks() -> None:
    for name, code in BENCHMARKS.items():
        df = with_retry(ak.stock_zh_index_daily, symbol=code)
        df["date"] = pd.to_datetime(df["date"])
        df["index"] = name
        save_parquet(df, DATA_RAW / "benchmarks" / f"{name}.parquet")
        logger.info("Benchmark %s: %s to %s", name, df["date"].min().date(), df["date"].max().date())
