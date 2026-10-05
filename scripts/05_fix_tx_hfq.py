"""Rebuild multiplicative hfq for stocks already downloaded from Tencent (offline, no network).
Run only after 03_compare_sources.py shows the rebuilt series matches Sina."""
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import DATA_RAW  # noqa: E402
from src.data.prices import rebuild_multiplicative_hfq  # noqa: E402
from src.data.utils import save_parquet  # noqa: E402

raw_dir, hfq_dir = DATA_RAW / "prices" / "raw", DATA_RAW / "prices" / "hfq"
done = skipped = 0
n_ex = []
for path in tqdm(sorted(hfq_dir.glob("*.parquet"))):
    hfq = pd.read_parquet(path)
    if hfq["source"].iloc[0] != "tx":  # sina files and already rebuilt files are left alone
        skipped += 1
        continue
    raw = pd.read_parquet(raw_dir / path.name)
    out = rebuild_multiplicative_hfq(raw, hfq)
    n_ex.append(len(out.attrs["ex_days"]))
    out.attrs = {}  # attrs hold Timestamps, which parquet metadata cannot serialise
    save_parquet(out, path)
    done += 1
print(f"rebuilt={done}, skipped={skipped}")
print("ex-right days per stock:", pd.Series(n_ex).describe().round(1).to_dict())
