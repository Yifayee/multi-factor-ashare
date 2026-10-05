"""Diagnostics for the factor panel: extreme forward returns and missing industries."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_PROCESSED, DATA_RAW  # noqa: E402

pd.set_option("display.width", 200)
panel = pd.read_parquet(DATA_PROCESSED / "factors_monthly.parquet")
dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
next_date = dict(zip(dates[:-1], dates[1:]))

ext = panel[(panel["fwd_ret"] < -0.7) | (panel["fwd_ret"] > 3)].copy()
print(f"extreme fwd_ret rows (< -70% or > +300%): {len(ext)} of {panel['fwd_ret'].notna().sum()}")

rows = []
for _, r in ext.reindex(ext["fwd_ret"].abs().sort_values(ascending=False).index).head(25).iterrows():
    hfq = pd.read_parquet(DATA_RAW / "prices" / "hfq" / f"{r.symbol}.parquet").set_index("date")
    raw = pd.read_parquet(DATA_RAW / "prices" / "raw" / f"{r.symbol}.parquet").set_index("date")
    nd = next_date.get(r.date)
    h_next = hfq.loc[:nd, "close"].iloc[-1] if nd is not None else None
    r_next = raw.loc[:nd, "close"].iloc[-1] if nd is not None else None
    rows.append({"symbol": r.symbol, "date": r.date.date(), "fwd_ret": round(r.fwd_ret, 3),
                 "source": hfq["source"].iloc[0], "hfq_t": round(hfq.loc[r.date, "close"], 3),
                 "hfq_next": round(h_next, 3), "raw_t": raw.loc[r.date, "close"], "raw_next": r_next,
                 "min_hfq_ever": round(hfq["close"].min(), 3)})
print(pd.DataFrame(rows).to_string(index=False))

src = {p.stem: pd.read_parquet(p, columns=["source"])["source"].iloc[0]
       for p in (DATA_RAW / "prices" / "hfq").glob("*.parquet")}
ext["source"] = ext["symbol"].map(src)
print("\nextreme rows by source:", ext["source"].value_counts().to_dict())
neg = [s for s, p in ((p.stem, p) for p in (DATA_RAW / "prices" / "hfq").glob("*.parquet"))
       if (pd.read_parquet(p, columns=["close"])["close"] <= 0).any()]
print("stocks with non-positive hfq close anywhere:", len(neg), neg[:20])

print("\nshare of rows with industry 'NA', by year:")
print(panel.assign(na=panel["industry"] == "NA").groupby(panel["date"].dt.year)["na"].mean().round(4).to_string())
