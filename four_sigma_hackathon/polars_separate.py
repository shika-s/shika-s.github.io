# polars_separate_files.py
# calculates the features for each of the tickers for all times separately and saves them to an output dir
#
# Date filter is read from submission/config.json (key: data.feature_build_start).
# Filtering at read time slashes disk usage:
#   2000-2025 (full):   ~36 GB intermediate features/
#   2015-2025 (10 yrs): ~14 GB
#   2020-2025 (5 yrs):  ~7 GB
import json
from datetime import datetime
import polars as pl
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DATA_DIR = Path("data/train")
OUT_DIR = Path("features")
OUT_DIR.mkdir(exist_ok=True)

# Read filter date from config.json (fallback to 2015-01-01 if config missing)
try:
    with open("submission/config.json") as f:
        START_DATE = json.load(f)["data"]["feature_build_start"]
except (FileNotFoundError, KeyError):
    START_DATE = "2015-01-01"
START_DT = datetime.strptime(START_DATE, "%Y-%m-%d")
print(f"[polars_separate] filtering rows >= {START_DATE}")



def compute_5min_features(df: pl.DataFrame) -> pl.DataFrame:
    # Filter regular session
    df = (
        df
        .filter(
            (pl.col("datetime").dt.time() >= pl.time(9, 30)) &
            (pl.col("datetime").dt.time() <= pl.time(16, 0))
        )
        .sort("datetime")
    )

    # First pass: bar-level features
    df = df.with_columns([
        (pl.col("close").pct_change()).alias("ret_5m"),
        (pl.col("close").log().diff()).alias("log_ret_5m"),
        (pl.col("volume") / pl.col("volume").shift(1)).alias("vol_change_raw"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("bar_range"),
        ((pl.col("close") - pl.col("open")) /
         (pl.col("high") - pl.col("low") + 1e-9)).alias("signed_range"),
        ((pl.col("high") - pl.col("close")) / pl.col("close")).alias("upper_wick"),
        ((pl.col("close") - pl.col("low")) / pl.col("close")).alias("lower_wick"),
    ])

    # Clean volume change
    df = df.with_columns(
        pl.when(pl.col("vol_change_raw").is_finite())
          .then(pl.col("vol_change_raw"))
          .otherwise(0.0)
          .alias("vol_change")
    ).drop("vol_change_raw")

    # Second pass: rolling features (now ret_5m exists)
    df = df.with_columns([
        pl.col("ret_5m").rolling_std(window_size=12).alias("vol_5m_12"),  # 1 hour vol
        pl.col("ret_5m").rolling_mean(window_size=6).alias("mom_5m_6"),   # 30m momentum
        pl.col("bar_range").rolling_mean(window_size=6).alias("range_5m_6"),
        pl.col("volume").rolling_mean(window_size=20).alias("vol_mean_20"),
        ((pl.col("volume") - pl.col("volume").rolling_mean(20)) /
         pl.col("volume").rolling_std(20)).alias("vol_zscore_20"),
    ])
    
    # After your rolling features block:
    df = df.with_columns([
        # Advanced Feature 1: Volatility Burst Ratio
        (pl.col("vol_5m_12") / pl.col("vol_5m_12").shift(12)).alias("vol_burst"),

        # Advanced Feature 2: Micro-Reversal Z-Score
        (
            (pl.col("ret_5m") - pl.col("ret_5m").rolling_mean(20)) /
            pl.col("ret_5m").rolling_std(20)
        ).alias("ret_z"),
    ])


    return df


def compute_30min_features(df: pl.DataFrame) -> pl.DataFrame:
    df = (
        df
        .filter(
            (pl.col("datetime").dt.time() >= pl.time(9, 30)) &
            (pl.col("datetime").dt.time() <= pl.time(16, 0))
        )
        .sort("datetime")
    )

    # Bar-level features
    df = df.with_columns([
        (pl.col("close").pct_change()).alias("ret_30m"),
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_30m"),
        (pl.col("volume") / pl.col("volume").shift(1)).alias("volume_change_30m"),
    ])

    # Rolling features
    df = df.with_columns([
        pl.col("ret_30m").rolling_mean(2).alias("mom_30m_2"),
        pl.col("ret_30m").rolling_mean(4).alias("mom_30m_4"),

        pl.col("range_30m").rolling_std(4).alias("vol_30m_4"),
        (
            (pl.col("range_30m") - pl.col("range_30m").rolling_mean(20)) /
            pl.col("range_30m").rolling_std(20)
        ).alias("range_z_30m"),

        (
            (pl.col("volume") - pl.col("volume").rolling_mean(20)) /
            pl.col("volume").rolling_std(20)
        ).alias("volume_z_30m"),
    ])

    # Session structure
    df = df.with_columns([
        (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute() - 570).alias("minutes_from_open"),
        (960 - (pl.col("datetime").dt.hour() * 60 + pl.col("datetime").dt.minute())).alias("minutes_to_close"),
    ])

    df = df.with_columns([
        (pl.col("minutes_from_open") / 390).alias("session_progress")
    ])

    #  JPS - added dollar_volume
    df = df.with_columns([
        (pl.col("volume") * pl.col("close")).alias("dollar_volume")
    ])
        
    #  JPS - added vwap
    df = (
        df
        .with_columns(pl.col("datetime").dt.date().alias("date"))
        .with_columns([
            pl.col("dollar_volume").cum_sum().over("date").alias("cu_dollar_volume"),
            pl.col("volume").cum_sum().over("date").alias("cu_volume"),
        ])
        .with_columns((pl.col("cu_dollar_volume") / pl.col("cu_volume")).alias("vwap")
        )
    )

    return df


def compute_1d_features(df: pl.DataFrame) -> pl.DataFrame:
    df = df.sort("datetime")

    # Basic daily returns
    df = df.with_columns([
        (pl.col("close").pct_change()).alias("ret_1d"),
        (pl.col("close").pct_change(5)).alias("ret_5d"),
        (pl.col("close").pct_change(21)).alias("ret_21d"),
    ])

    # Daily range
    df = df.with_columns([
        ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("range_1d")
    ])

    # Gap return
    df = df.with_columns([
        ((pl.col("open") - pl.col("close").shift(1)) /
         pl.col("close").shift(1)).alias("gap_return")
    ])

    # Volatility regime
    df = df.with_columns([
        pl.col("ret_1d").rolling_std(window_size=20).alias("vol_20d")
    ])

    # Volume regime
    df = df.with_columns([
        (
            (pl.col("volume") - pl.col("volume").rolling_mean(20)) /
            pl.col("volume").rolling_std(20)
        ).alias("volume_z_20d")
    ])

    # Moving averages
    df = df.with_columns([
        pl.col("close").rolling_mean(5).alias("ma_5d"),
        pl.col("close").rolling_mean(10).alias("ma_10d"),
        pl.col("close").rolling_mean(20).alias("ma_20d"),
    ])


    # Price position in daily range
    df = df.with_columns([
        ((pl.col("close") - pl.col("low")) /
         (pl.col("high") - pl.col("low") + 1e-9)).alias("pos_in_day")
    ])

    return df



def process_ticker(ticker: str):
    # Filter at read time — drops bars before START_DT, dramatically reducing
    # downstream memory + disk for intermediate features.
    df5  = pl.read_parquet(DATA_DIR / f"{ticker}_5min.parquet").filter(pl.col("datetime") >= pl.lit(START_DT))
    df30 = pl.read_parquet(DATA_DIR / f"{ticker}_30min.parquet").filter(pl.col("datetime") >= pl.lit(START_DT))
    df1d = pl.read_parquet(DATA_DIR / f"{ticker}_1day.parquet").filter(pl.col("datetime") >= pl.lit(START_DT))

    # Compute features
    f5  = compute_5min_features(df5)
    f30 = compute_30min_features(df30)
    f1d = compute_1d_features(df1d)

    # --- NEW: enforce consistent schema ---
    f5  = enforce_schema(f5)
    f30 = enforce_schema(f30)
    f1d = enforce_schema(f1d)

    # Save each timeframe separately
    f5.write_parquet(OUT_DIR / f"{ticker}_5m.parquet")
    f30.write_parquet(OUT_DIR / f"{ticker}_30m.parquet")
    f1d.write_parquet(OUT_DIR / f"{ticker}_1d.parquet")

    print(ticker)
    return ticker


def enforce_schema(df: pl.DataFrame) -> pl.DataFrame:
    # Convert all integer-like columns to Float64
    # (volume is the usual culprit)
    return df.with_columns([
        pl.col(col).cast(pl.Float64)
        for col, dtype in df.schema.items()
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64)
    ])


def run_parallel(tickers, n_workers=6):
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for t in ex.map(process_ticker, tickers):
            print("done:", t)


if __name__ == "__main__":
    import json

    with open("universe.json") as f:
        tickers = json.load(f)["tickers"]

    run_parallel(tickers, n_workers=6)
    print("All features saved separately.")
