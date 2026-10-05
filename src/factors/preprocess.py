"""Cross-sectional preprocessing, applied date by date."""
import numpy as np
import pandas as pd


def mad_winsorize(x: pd.Series, k: float) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median()
    if not np.isfinite(mad) or mad == 0:
        return x
    width = k * 1.4826 * mad
    return x.clip(med - width, med + width)


def zscore(x: pd.Series) -> pd.Series:
    sd = x.std()
    return (x - x.mean()) / sd if sd and np.isfinite(sd) else x * 0.0


def neutralize(y: pd.Series, industry: pd.Series, size: pd.Series = None) -> pd.Series:
    """Residual of OLS on industry dummies (+ size). Industry dummies span the intercept."""
    X = pd.get_dummies(industry, dtype=float)
    if size is not None:
        X["size"] = size.values
    beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
    return pd.Series(y.values - X.values @ beta, index=y.index)


def process_cross_section(df: pd.DataFrame, factors: dict, k: float) -> pd.DataFrame:
    """df: one date. factors: {name: prior_sign}. Returns processed columns named like the factor."""
    out = pd.DataFrame(index=df.index)
    size_z = zscore(mad_winsorize(df["size_raw"], k))
    for name, sign in factors.items():
        x = df[f"{name}_raw"] * sign
        x = mad_winsorize(x, k)
        x = x.fillna(x.groupby(df["industry"]).transform("median")).fillna(x.median())
        x = zscore(x)
        x = neutralize(x, df["industry"], None if name == "size" else size_z)
        out[name] = zscore(x)
    return out
