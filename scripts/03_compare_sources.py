"""Check that hfq returns agree across sources, since listed and delisted stocks
come from different sources. Price levels may differ (different base date);
daily returns must not.

Compares Sina hfq returns with (1) Tencent's adjusted series as published and
(2) the multiplicative series rebuilt from it (see rebuild_multiplicative_hfq).

Usage: python scripts/03_compare_sources.py --symbols 000001,600519,000002
"""
import argparse
import sys
from pathlib import Path

import akshare as ak
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.prices import fetch_sina, fetch_tx, _finish, rebuild_multiplicative_hfq  # noqa: E402
from src.data.utils import with_market_prefix, with_retry  # noqa: E402
from config import PRICE_END, PRICE_START  # noqa: E402


def tx_published_hfq(symbol: str) -> pd.DataFrame:
    start = f"{PRICE_START[:4]}-{PRICE_START[4:6]}-{PRICE_START[6:]}"
    end = f"{PRICE_END[:4]}-{PRICE_END[4:6]}-{PRICE_END[6:]}"
    df = with_retry(ak.stock_zh_a_hist_tx, symbol=with_market_prefix(symbol),
                    start_date=start, end_date=end, adjust="hfq", timeout=20)
    df["turnover"] = 0.0
    return _finish(df, symbol, "tx")


def report(name: str, base: pd.DataFrame, other: pd.DataFrame) -> None:
    m = base[["date", "close"]].merge(other[["date", "close"]], on="date", suffixes=("_a", "_b"))
    m = m.sort_values("date").reset_index(drop=True)
    ra, rb = m["close_a"].pct_change(), m["close_b"].pct_change()
    diff = (ra - rb).abs()
    print(f"  [{name}] common dates={len(m)}, corr={ra.corr(rb):.6f}, max |diff|={diff.max():.4%}, "
          f"days |diff|>0.1%: {(diff > 0.001).sum()}")
    cum_a, cum_b = (1 + ra.fillna(0)).prod() - 1, (1 + rb.fillna(0)).prod() - 1
    print(f"    cumulative return over common dates: sina={cum_a:.2%}, other={cum_b:.2%}")
    worst = m.assign(ret_sina=ra, ret_other=rb, diff=diff).nlargest(3, "diff")
    print(worst[["date", "ret_sina", "ret_other", "diff"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", default="000001,600519,000002")
    args = parser.parse_args()
    for sym in args.symbols.split(","):
        _, sina = fetch_sina(sym)
        raw_tx, rebuilt = fetch_tx(sym)
        published = tx_published_hfq(sym)
        print(f"\n{sym}:")
        report("tx published", sina, published)
        report("tx rebuilt", sina, rebuilt)
        print(f"    ex-right days detected: {[d.date().isoformat() for d in rebuilt.attrs['ex_days']]}")
