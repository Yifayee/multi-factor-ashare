"""Phase 1 data quality report. Prints facts only; decisions are made after reading them."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW  # noqa: E402


def load_prices(tag: str) -> pd.DataFrame:
    files = list((DATA_RAW / "prices" / tag).glob("*.parquet"))
    return pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)


def main() -> None:
    uni = pd.read_parquet(DATA_RAW / "universe.parquet")
    print("== Universe ==")
    print(uni["status"].value_counts().to_string())

    raw = load_prices("raw")
    print("\n== Prices (raw) ==")
    print("stocks with data:", raw["symbol"].nunique())
    print("date range:", raw["date"].min().date(), "to", raw["date"].max().date())
    have = set(raw["symbol"])
    delisted = uni[uni.status == "delisted"]
    cov = delisted["symbol"].isin(have).mean() if len(delisted) else float("nan")
    print(f"delisted stocks with price data: {delisted['symbol'].isin(have).sum()}/{len(delisted)} ({cov:.1%})")
    print("rows with turnover<=0 or missing:", int((raw["turnover"].fillna(0) <= 0).sum()))
    print("stocks by source:", raw.groupby("source")["symbol"].nunique().to_dict())
    # Unit check: amount / (volume * close) should be close to 1 if volume is in shares
    ratio = raw["amount"] / (raw["volume"] * raw["close"])
    print("amount/(volume*close) median by source:", ratio.groupby(raw["source"]).median().round(3).to_dict())
    print("stocks per year:")
    print(raw.groupby(raw["date"].dt.year)["symbol"].nunique().to_string())

    ind = pd.read_parquet(DATA_RAW / "industry_sw_hist.parquet")
    print("\n== Industry ==")
    print("columns:", list(ind.columns))
    print("universe coverage:", f"{uni['symbol'].isin(ind['symbol']).mean():.1%}")
    print("delisted coverage:", f"{delisted['symbol'].isin(ind['symbol']).mean():.1%}")
    print("earliest start_date:", ind["start_date"].min())

    print("\n== Financials (yjbb) ==")
    fin = pd.concat(pd.read_parquet(f) for f in (DATA_RAW / "financials" / "yjbb").glob("*.parquet"))
    fin["ann"] = pd.to_datetime(fin["最新公告日期"])
    lag = (fin["ann"] - fin["report_period"]).dt.days
    print("rows:", len(fin), "| periods:", fin["report_period"].nunique())
    code_col = "股票代码"
    per = fin.groupby("report_period")[code_col].agg(["size", "nunique"])
    print("rows vs unique stocks per period (last 4):")
    print(per.tail(4).to_string())
    print("example columns:", list(fin.columns))
    print("announcement lag (days) quantiles:")
    print(lag.quantile([0.01, 0.25, 0.5, 0.75, 0.99]).to_string())
    print("announcement before period end (should be ~0):", int((lag < 0).sum()))
    print("missing announcement date:", int(fin["ann"].isna().sum()))
    print("\nfinancial table columns:")
    for name in ("zcfz", "lrb", "xjll"):
        f = next((DATA_RAW / "financials" / name).glob("*.parquet"), None)
        if f is not None:
            print(name, list(pd.read_parquet(f).columns))


if __name__ == "__main__":
    main()
