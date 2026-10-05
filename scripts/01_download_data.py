"""Phase 1 entry point. Run the steps in this order:
    python scripts/01_download_data.py --step smoke        # test sources, saves nothing
    python scripts/01_download_data.py --step reference    # universe, calendar, industry, benchmarks
    python scripts/01_download_data.py --step financials   # quarterly statements with announcement dates
    python scripts/01_download_data.py --step listed       # prices for listed stocks (Sina first)
    python scripts/01_download_data.py --step delisted     # prices for delisted stocks (Eastmoney first)
Every step skips files that already exist, so it is safe to re-run after a failure.
"""
import argparse
import sys
from pathlib import Path

import akshare as ak

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import PRICE_SOURCES_DELISTED, PRICE_SOURCES_LISTED  # noqa: E402
from src.data.financials import download_financials  # noqa: E402
from src.data.prices import FETCHERS, download_prices  # noqa: E402
from src.data.reference import download_benchmarks, download_calendar, download_industry  # noqa: E402
from src.data.universe import build_universe  # noqa: E402


def smoke(sources) -> None:
    """Try each source on 1 listed + 3 delisted stocks, plus one financial-statement call."""
    uni = build_universe(use_cache=True)
    listed = uni[uni.status == "listed"]["symbol"].head(1).tolist()
    delisted = uni[uni.status == "delisted"]["symbol"].sample(3, random_state=2).tolist()
    for src in sources:
        print(f"\n=== source: {src} ===")
        for sym in listed + delisted:
            tag = "listed" if sym in listed else "delisted"
            try:
                raw, hfq = FETCHERS[src](sym)
            except Exception as exc:
                print(f"{sym} ({tag}): FAILED {type(exc).__name__}")
                continue
            if raw.empty:
                print(f"{sym} ({tag}): EMPTY")
                continue
            print(f"{sym} ({tag}): {len(raw)} rows, {raw.date.min().date()} to {raw.date.max().date()}, "
                  f"last raw close={raw.close.iloc[-1]:.2f}, last hfq close={hfq.close.iloc[-1]:.2f}, "
                  f"median turnover%={raw.turnover.median():.3f}, "
                  f"median amount/(volume*close)={(raw.amount / (raw.volume * raw.close)).median():.3f}")

    print("\n=== financials (Eastmoney datacenter) ===")
    try:
        fin = ak.stock_yjbb_em(date="20231231")
        print(f"yjbb 20231231: {len(fin)} rows, {fin.shape[1]} columns")
    except Exception as exc:
        print(f"yjbb 20231231: FAILED {type(exc).__name__}")


def main(step: str, sources) -> None:
    if step == "smoke":
        smoke(sources or ("em", "sina", "tx"))
    elif step == "reference":
        # Run each part independently so one failure does not block the others
        for fn in (lambda: build_universe(use_cache=True), download_calendar,
                   download_industry, download_benchmarks):
            try:
                fn()
            except Exception as exc:
                print(f"FAILED: {exc}")
    elif step == "financials":
        download_financials()
    elif step in ("listed", "delisted"):
        uni = build_universe(use_cache=True)
        symbols = uni.loc[uni.status == step, "symbol"].tolist()
        default = PRICE_SOURCES_LISTED if step == "listed" else PRICE_SOURCES_DELISTED
        failed = download_prices(symbols, sources or default)
        if failed:  # one more pass for transient failures
            download_prices(failed, sources or default)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", default="smoke",
                        choices=["smoke", "reference", "financials", "listed", "delisted"])
    parser.add_argument("--sources", default="", help="override source order, e.g. sina,em")
    args = parser.parse_args()
    main(args.step, tuple(s for s in args.sources.split(",") if s))
