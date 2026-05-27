# feature_collapse_parallel.py

import polars as pl
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def aggregate_5m_to_30m(f5: pl.DataFrame) -> pl.DataFrame:
    return (
        f5
        .group_by_dynamic(
            index_column="datetime",
            every="30m",
            period="30m",
            closed="right"
        )
        .agg([
            pl.col("ret_5m").mean().alias("ret_5m_mean"),
            pl.col("log_ret_5m").sum().alias("log_ret_5m_sum"),
            pl.col("vol_change").mean().alias("vol_change_mean"),
            pl.col("bar_range").mean().alias("bar_range_mean"),
        ])
        .sort("datetime")
    )


def join_daily(f30: pl.DataFrame, f1d: pl.DataFrame) -> pl.DataFrame:
    return (
        f30
        .join_asof(
            f1d.sort("datetime"),
            left_on="datetime",
            right_on="datetime",
            strategy="backward"
        )
    )


def combine_features_for_ticker(ticker: str, feature_dir: Path) -> pl.DataFrame:
    f5  = pl.read_parquet(feature_dir / f"{ticker}_5m.parquet")
    f30 = pl.read_parquet(feature_dir / f"{ticker}_30m.parquet")
    f1d = pl.read_parquet(feature_dir / f"{ticker}_1d.parquet")

    f5_agg = aggregate_5m_to_30m(f5)
    f30 = join_daily(f30.sort("datetime"), f1d)

    merged = f30.join(f5_agg, on="datetime", how="left")
    return merged.with_columns(pl.lit(ticker).alias("ticker"))

def normalize_schema(df: pl.DataFrame) -> pl.DataFrame:
    # Convert all integer-like columns to Float64
    return df.with_columns([
        pl.col(col).cast(pl.Float64)
        for col, dtype in df.schema.items()
        if dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64
        )
    ])


def process_ticker(ticker: str, feature_dir: Path, tmp_dir: Path) -> str:
    df = combine_features_for_ticker(ticker, feature_dir)

    # NEW: enforce consistent schema
    df = normalize_schema(df)

    out_file = tmp_dir / f"{ticker}.parquet"
    df.write_parquet(out_file)
    return ticker



if __name__ == "__main__":
    import json

    with open("universe.json") as f:
        tickers = json.load(f)["tickers"]

    FEATURE_DIR = Path("features")
    OUT_PATH = Path("collapsed_30m.parquet")
    TMP_DIR = Path("tmp_parallel")
    TMP_DIR.mkdir(exist_ok=True)

    # Remove old output
    if OUT_PATH.exists():
        OUT_PATH.unlink()

    # --- PARALLEL EXECUTION ---
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_ticker, t, FEATURE_DIR, TMP_DIR): t
            for t in tickers
        }

        for future in as_completed(futures):
            t = futures[future]
            try:
                future.result()
                print("done:", t)
            except Exception as e:
                print(f"Error processing {t}: {e}")

    # --- CONCATENATE RESULTS ---
    dfs = [pl.read_parquet(p) for p in TMP_DIR.glob("*.parquet")]
    final_df = pl.concat(dfs)
    final_df.write_parquet(OUT_PATH)

    print("All tickers appended to collapsed_30m.parquet")
