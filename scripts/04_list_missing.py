"""List universe stocks that have no saved price file, with the source order that was tried."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW  # noqa: E402

uni = pd.read_parquet(DATA_RAW / "universe.parquet")
have = {p.stem for p in (DATA_RAW / "prices" / "raw").glob("*.parquet")}
missing = uni[~uni["symbol"].isin(have)]
print(f"universe={len(uni)}, with prices={len(have)}, missing={len(missing)}")
print(missing.groupby("status").size().to_string())
print(missing[["symbol", "name", "status"]].to_string(index=False))
