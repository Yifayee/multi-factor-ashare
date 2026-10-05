"""Global configuration. All research choices are fixed here before any backtest is run."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

# Price data window: 2014-2015 is warm-up for 12-month momentum and volatility factors
PRICE_START = "20140101"
PRICE_END = "20251231"

# Financial reports: start one year earlier so TTM and YoY figures exist in 2014
FIN_START_YEAR = 2013
FIN_END_YEAR = 2025

# Research periods (decided up front, not tuned)
BACKTEST_START = "2016-01-01"
HOLDOUT_START = "2024-01-01"  # untouched until the final evaluation

# Only Shanghai/Shenzhen A-shares: main boards, ChiNext (30x), STAR (68x). Excludes BSE and B-shares.
A_SHARE_PREFIXES = ("00", "30", "60", "68")

# Benchmarks (Sina index symbols)
BENCHMARKS = {
    "csi300": "sh000300",
    "csi500": "sh000905",
    "csi1000": "sh000852",  # replaces CSI All Share: Sina only returned 2011-2016 for sh000985
}

# Network politeness (data sites throttle fast scrapers)
REQUEST_SLEEP = (1.0, 2.5)      # random pause after each successful request, seconds
MAX_RETRIES = 3
RETRY_BASE_SLEEP = 3.0          # backoff: 3s, 6s, ...
COOLDOWN_AFTER_FAILS = 3        # consecutive stocks failing on every source
COOLDOWN_SECONDS = 300
# Source order by stock status, based on the Phase 1 smoke test:
# Sina served listed stocks reliably but not delisted ones; Eastmoney served delisted ones but throttles hard
PRICE_SOURCES_LISTED = ("sina", "em")
PRICE_SOURCES_DELISTED = ("em", "tx")

# ---------------- Phase 2: factors (fixed before looking at any test result) ----------------
MIN_LISTED_DAYS = 120      # exclude stocks with fewer trading days in our data (new listings)
MAD_K = 3.0                # winsorize at median +/- 3 * 1.4826 * MAD (about 3 sigma)

# Expected direction from theory / prior literature: +1 means higher value -> higher future return.
# Signs are applied before testing so that Phase 3 never flips a factor because of its own result.
FACTOR_PRIORS = {
    "size": -1,       # log float market cap, small-cap premium
    "bp": +1,         # book-to-price
    "ep": +1,         # TTM earnings-to-price
    "roe": +1,        # TTM return on equity
    "accruals": -1,   # (TTM net profit - TTM operating cash flow) / total assets
    "growth": +1,     # latest reported YoY net profit growth
    "mom_12_1": +1,   # return from t-12m to t-1m
    "rev_1m": -1,     # last-month return, short-term reversal
    "vol_60d": -1,    # 60-day volatility, low-volatility anomaly
    "turn_20d": -1,   # 20-day average turnover
}
