"""Daily bars per stock, from Eastmoney with Sina as fallback.

Two files per stock, same schema regardless of source:
- raw (unadjusted): for price-limit checks and float market cap
- hfq (backward-adjusted): for returns, momentum, volatility

Standard schema: date, open, high, low, close, volume (shares), amount (CNY),
turnover (% of float shares), symbol, source
"""
import time

import akshare as ak
import numpy as np
import pandas as pd
from tqdm import tqdm

from config import (COOLDOWN_AFTER_FAILS, COOLDOWN_SECONDS, DATA_RAW,
                    PRICE_END, PRICE_SOURCES_LISTED, PRICE_START)
from src.data.utils import logger, save_parquet, with_market_prefix, with_retry

STD_COLS = ["date", "open", "high", "low", "close", "volume", "amount", "turnover"]
EM_COLS = {"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low",
           "成交量": "volume", "成交额": "amount", "换手率": "turnover"}


def _fix_volume_units(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """If amount/(volume*close) is ~100, volume is in lots; convert to shares."""
    ratio = (df["amount"] / (df["volume"] * df["close"])).median()
    if 20 < ratio < 500:
        logger.info("%s: volume looks like lots (ratio %.1f), converting to shares", symbol, ratio)
        df["volume"] = df["volume"] * 100
    return df


def _finish(df: pd.DataFrame, symbol: str, source: str) -> pd.DataFrame:
    df = df[STD_COLS].copy()
    for col in STD_COLS[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["date"] = pd.to_datetime(df["date"]).astype("datetime64[ns]")
    df = df[(df["date"] >= PRICE_START) & (df["date"] <= PRICE_END)]
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["symbol"] = symbol
    df["source"] = source
    return df


def fetch_em(symbol: str):
    """Eastmoney: one request per adjustment type. Volume is reported in lots (100 shares)."""
    out = []
    for adjust in ("", "hfq"):
        df = with_retry(ak.stock_zh_a_hist, symbol=symbol, period="daily",
                        start_date=PRICE_START, end_date=PRICE_END, adjust=adjust)
        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame()
        df = df.rename(columns=EM_COLS)
        df["volume"] = df["volume"] * 100  # lots -> shares
        out.append(_finish(df, symbol, "em"))
    return out[0], out[1]


def fetch_sina(symbol: str):
    """Sina: raw bars + hfq factor table, hfq prices computed locally at full precision."""
    code = with_market_prefix(symbol)
    raw = with_retry(ak.stock_zh_a_daily, symbol=code, adjust="")
    if raw is None or raw.empty:
        return pd.DataFrame(), pd.DataFrame()
    raw["turnover"] = raw["turnover"] * 100  # fraction -> percent, same unit as Eastmoney
    raw = _finish(raw, symbol, "sina")

    factor = with_retry(ak.stock_zh_a_daily, symbol=code, adjust="hfq-factor")
    factor = factor.rename(columns={factor.columns[0]: "date"})
    factor["date"] = pd.to_datetime(factor["date"]).astype("datetime64[ns]")
    factor["hfq_factor"] = pd.to_numeric(factor["hfq_factor"], errors="coerce")
    factor = factor.dropna().sort_values("date")

    # Each factor applies from its date onward
    hfq = pd.merge_asof(raw, factor, on="date", direction="backward")
    if hfq["hfq_factor"].isna().any():
        raise ValueError(f"{symbol}: hfq factor missing for some dates")
    for col in ("open", "high", "low", "close"):
        hfq[col] = hfq[col] * hfq["hfq_factor"]
    return raw, hfq.drop(columns="hfq_factor")


def fetch_tx(symbol: str):
    """Tencent: one request per year per adjustment type, so slower. Units are checked, not assumed."""
    code = with_market_prefix(symbol)
    start = f"{PRICE_START[:4]}-{PRICE_START[4:6]}-{PRICE_START[6:]}"
    end = f"{PRICE_END[:4]}-{PRICE_END[4:6]}-{PRICE_END[6:]}"
    out = []
    for adjust in ("", "hfq"):
        df = with_retry(ak.stock_zh_a_hist_tx, symbol=code, start_date=start, end_date=end,
                        adjust=adjust, timeout=20)
        if df is None or df.empty:
            return pd.DataFrame(), pd.DataFrame()
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") * 100  # akshare divides by 100
        out.append(_finish(df, symbol, "tx"))
    raw, hfq = out
    raw = _fix_volume_units(raw, symbol)
    return raw, rebuild_multiplicative_hfq(raw, hfq)


def rebuild_multiplicative_hfq(raw: pd.DataFrame, hfq_tx: pd.DataFrame,
                               rounding: float = 0.01) -> pd.DataFrame:
    """Rebuild a multiplicative hfq series from Tencent's adjusted prices.

    Cross-checking against Sina showed Tencent's adjusted price is affine in the raw price
    between corporate actions: adj = a * raw + b. Splits/bonus shares change the slope a;
    cash dividends only add to the intercept b. Percentage returns of such a series are
    compressed by b, so they cannot be used directly.

    Walk forward through the series, keeping the current segment's anchor point and slope.
    A day whose adj is off the segment's line by more than rounding error is an ex-right day.
      - normal day:   r_t = raw_t / raw_{t-1} - 1
      - ex-right day: r_t = (adj_t - adj_{t-1}) / (a_old * raw_{t-1}),
        which equals ((1 + split) * raw_t + dividend) / raw_{t-1} - 1
    """
    m = raw[["date", "close"]].merge(hfq_tx[["date", "close"]], on="date", suffixes=("_raw", "_adj"))
    m = m.sort_values("date").reset_index(drop=True)
    r_px, a_px = m["close_raw"].to_numpy(float), m["close_adj"].to_numpy(float)
    n = len(m)
    ret = np.zeros(n)
    ex_days = []

    s = 0                 # segment anchor index
    a_hat = None          # slope of current segment
    a_last = 1.0          # slope of previous segment, fallback when the current one is unknown
    span = 0.0            # largest |raw - raw_anchor| used to estimate a_hat

    for t in range(1, n):
        d_raw = r_px[t] - r_px[s]
        if a_hat is not None:
            pred = a_px[s] + a_hat * d_raw
            tol = 1.2 * rounding + (rounding / span) * abs(d_raw)
            on_line = abs(a_px[t] - pred) <= tol
        else:
            on_line = True  # cannot test yet; first moves of a new segment define its slope

        if on_line:
            ret[t] = r_px[t] / r_px[t - 1] - 1
            if abs(d_raw) > span:
                span = abs(d_raw)
                a_hat = (a_px[t] - a_px[s]) / d_raw
        else:
            a_old = a_hat if a_hat is not None else a_last
            ret[t] = (a_px[t] - a_px[t - 1]) / (a_old * r_px[t - 1])
            ex_days.append(m["date"].iloc[t])
            a_last, s, a_hat, span = a_old, t, None, 0.0

    close_hfq = r_px[0] * np.cumprod(1 + ret)
    factor = close_hfq / r_px
    out = raw[raw["date"].isin(m["date"])].sort_values("date").reset_index(drop=True).copy()
    for col in ("open", "high", "low", "close"):
        out[col] = out[col] * factor
    out["source"] = "tx_rebuilt"
    out.attrs["ex_days"] = ex_days
    return out


FETCHERS = {"em": fetch_em, "sina": fetch_sina, "tx": fetch_tx}


def fetch_with_fallback(symbol: str, sources=PRICE_SOURCES_LISTED):
    for src in sources:
        try:
            raw, hfq = FETCHERS[src](symbol)
            return raw, hfq, src
        except Exception as exc:
            logger.warning("%s via %s failed: %s", symbol, src, type(exc).__name__)
    raise RuntimeError(f"{symbol}: all sources failed")


def download_prices(symbols, sources=PRICE_SOURCES_LISTED) -> list:
    """Download and cache raw + hfq for each stock. Re-running skips finished stocks."""
    raw_dir, hfq_dir = DATA_RAW / "prices" / "raw", DATA_RAW / "prices" / "hfq"
    failed, empty, used = [], [], {}
    consecutive_fails = 0
    for sym in tqdm(symbols, desc="prices"):
        if (raw_dir / f"{sym}.parquet").exists() and (hfq_dir / f"{sym}.parquet").exists():
            continue
        try:
            raw, hfq, src = fetch_with_fallback(sym, sources)
            consecutive_fails = 0
        except RuntimeError as exc:
            logger.error(str(exc))
            failed.append(sym)
            consecutive_fails += 1
            if consecutive_fails >= COOLDOWN_AFTER_FAILS:
                logger.warning("%d stocks failed in a row, cooling down %ds",
                               consecutive_fails, COOLDOWN_SECONDS)
                time.sleep(COOLDOWN_SECONDS)
                consecutive_fails = 0
            continue
        if raw.empty:
            empty.append(sym)
            continue
        save_parquet(raw, raw_dir / f"{sym}.parquet")
        save_parquet(hfq, hfq_dir / f"{sym}.parquet")
        used[src] = used.get(src, 0) + 1
    logger.info("prices done: saved=%s, failed=%d, empty=%d", used, len(failed), len(empty))
    tag = "_".join(sources)
    (DATA_RAW / "prices").mkdir(parents=True, exist_ok=True)
    pd.Series(empty, name="symbol", dtype=str).to_csv(DATA_RAW / "prices" / f"empty_{tag}.csv", index=False)
    pd.Series(failed, name="symbol", dtype=str).to_csv(DATA_RAW / "prices" / f"failed_{tag}.csv", index=False)
    return failed
