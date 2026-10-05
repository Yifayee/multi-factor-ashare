"""Shared helpers for data download: retry, caching, logging."""
import logging
import random
import time
from pathlib import Path

import pandas as pd

from config import MAX_RETRIES, REQUEST_SLEEP, RETRY_BASE_SLEEP

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("data")


def polite_sleep() -> None:
    time.sleep(random.uniform(*REQUEST_SLEEP))


def with_retry(fn, *args, retries: int = MAX_RETRIES, base_sleep: float = RETRY_BASE_SLEEP, **kwargs):
    """Call fn with exponential backoff. Raises the last exception if all attempts fail."""
    for attempt in range(1, retries + 1):
        try:
            result = fn(*args, **kwargs)
            polite_sleep()
            return result
        except Exception as exc:  # network / parsing errors from upstream sites
            if attempt == retries:
                raise
            wait = base_sleep * 2 ** (attempt - 1) + random.uniform(0, 1.0)
            logger.warning("%s failed (%s), retry %d/%d in %.1fs",
                           getattr(fn, "__name__", fn), type(exc).__name__, attempt, retries - 1, wait)
            time.sleep(wait)


def save_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def is_target_a_share(symbol: str, prefixes) -> bool:
    return isinstance(symbol, str) and len(symbol) == 6 and symbol.startswith(prefixes)


def with_market_prefix(symbol: str) -> str:
    """'600000' -> 'sh600000', '000001' -> 'sz000001' (Sina/Tencent style)."""
    return ("sh" if symbol.startswith("6") else "sz") + symbol


def to_ns(s):
    """Normalise datetimes to nanosecond precision; parquet may return microseconds,
    and merge_asof refuses keys with different units."""
    return pd.to_datetime(s).astype("datetime64[ns]")
