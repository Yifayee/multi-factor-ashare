"""How early is each financial table's announcement date, by report type?
Restricted to our universe. A median lag far beyond the statutory deadline means the
date is a 'latest update' date, not the first disclosure."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW  # noqa: E402

# Statutory deadlines, days after period end: Q1 Apr 30, H1 Aug 31, Q3 Oct 31, annual Apr 30 next year
DEADLINE_DAYS = {3: 30, 6: 62, 9: 31, 12: 120}
DATE_COL = {"yjbb": "最新公告日期", "zcfz": "公告日期", "lrb": "公告日期", "xjll": "公告日期"}

uni = set(pd.read_parquet(DATA_RAW / "universe.parquet")["symbol"])
for name, col in DATE_COL.items():
    df = pd.concat(pd.read_parquet(f) for f in (DATA_RAW / "financials" / name).glob("*.parquet"))
    df = df[df["股票代码"].astype(str).str.zfill(6).isin(uni)]
    lag = (pd.to_datetime(df[col], errors="coerce") - df["report_period"]).dt.days
    month = df["report_period"].dt.month
    late = lag > month.map(DEADLINE_DAYS)
    print(f"\n== {name} ({col}) rows in universe: {len(df)}, dup (code, period): "
          f"{df.duplicated(['股票代码', 'report_period']).sum()}")
    summary = pd.DataFrame({"lag": lag, "month": month, "late": late}).groupby("month").agg(
        n=("lag", "size"), p10=("lag", lambda x: x.quantile(0.1)), median=("lag", "median"),
        p90=("lag", lambda x: x.quantile(0.9)), share_after_deadline=("late", "mean"))
    print(summary.round(3).to_string())
