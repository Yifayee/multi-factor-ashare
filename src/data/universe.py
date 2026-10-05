"""Stock universe including delisted names, to avoid survivorship bias."""
import pandas as pd
import akshare as ak

from config import A_SHARE_PREFIXES, DATA_RAW, PRICE_START
from src.data.utils import is_target_a_share, logger, save_parquet, with_retry


def _listed() -> pd.DataFrame:
    df = with_retry(ak.stock_info_a_code_name)
    df = df.rename(columns={"code": "symbol", "name": "name"})
    df["status"] = "listed"
    df["delist_date"] = pd.NaT
    return df[["symbol", "name", "status", "delist_date"]]


def _delisted() -> pd.DataFrame:
    sh = with_retry(ak.stock_info_sh_delist, symbol="全部")
    sh = sh.rename(columns={"公司代码": "symbol", "公司简称": "name",
                            "上市日期": "list_date", "暂停上市日期": "delist_date"})
    sz = with_retry(ak.stock_info_sz_delist, symbol="终止上市公司")
    sz = sz.rename(columns={"证券代码": "symbol", "证券简称": "name",
                            "上市日期": "list_date", "终止上市日期": "delist_date"})
    df = pd.concat([sh[["symbol", "name", "list_date", "delist_date"]],
                    sz[["symbol", "name", "list_date", "delist_date"]]], ignore_index=True)
    df["symbol"] = df["symbol"].astype(str).str.zfill(6)
    df["delist_date"] = pd.to_datetime(df["delist_date"], errors="coerce")
    df["status"] = "delisted"
    return df[["symbol", "name", "status", "delist_date"]]


def build_universe(use_cache: bool = False) -> pd.DataFrame:
    path = DATA_RAW / "universe.parquet"
    if use_cache and path.exists():
        return pd.read_parquet(path)
    listed = _listed()
    delisted = _delisted()
    # Only delistings that fall inside our sample matter
    delisted = delisted[delisted["delist_date"] >= pd.Timestamp(PRICE_START)]
    uni = pd.concat([listed, delisted], ignore_index=True)
    uni = uni[uni["symbol"].apply(lambda s: is_target_a_share(s, A_SHARE_PREFIXES))]
    # A code can appear in both lists if it was reused; keep the listed record
    uni = uni.drop_duplicates(subset="symbol", keep="first").reset_index(drop=True)
    save_parquet(uni, path)
    logger.info("Universe: %d listed, %d delisted in sample",
                (uni.status == "listed").sum(), (uni.status == "delisted").sum())
    return uni
