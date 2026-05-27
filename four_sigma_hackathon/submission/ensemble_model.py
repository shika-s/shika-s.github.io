"""
Ensemble submission: averages predictions from LightGBM + XGBoost + RandomForest.

Loads three trained models from:
    submission/weights.pkl       (LightGBM)
    submission/weights_xgb.pkl   (XGBoost)
    submission/weights_rf.pkl    (RandomForest)

Plus the sector mapping from submission/clusters.pkl.

Feature pipeline is identical to lightGBM_model.py — we just produce three
predictions per row, then average them. RandomForest can't handle NaN, so we
fillna(0) for its prediction step only.

To use this instead of single-model lightGBM_model.py, point the scorer at
this file or rename it to model.py.
"""

import pathlib
import pickle
from datetime import time as dt_time

import numpy as np
import pandas as pd


REGULAR_OPEN  = dt_time(9, 30)
REGULAR_CLOSE = dt_time(16, 0)

DEFAULT_PREPARE_START      = pd.Timestamp("2020-01-01")
DEFAULT_PREDICT_CHUNK_SIZE = 1_000_000

FEATURES = [
    # per-ticker (30-min)
    "momentum", "volume_change", "bar_range",
    "rolling_mean_5", "rolling_std_5", "rolling_volume_ratio",
    "rolling_mean_13", "rolling_std_13",
    # cross-sectional (vs. universe)
    "momentum_z", "momentum_rank", "bar_range_z", "volume_change_z",
    "rolling_mean_5_z", "rolling_mean_13_z", "rolling_volume_ratio_z",
    # sector-relative (vs. cluster)
    "momentum_relative_to_sector", "momentum_sector_z",
    # 5-min microstructure (intra-bar shape)
    "intrabar_std", "close_position", "volume_concentration", "tail_strength",
    # extras: time-of-day, dollar volume, VWAP distance, overnight gap
    "minutes_from_open", "minutes_to_close", "dollar_volume", "vwap_distance", "gap_return",
]

# Equal-weight ensemble by default — see notebook ensemble cell for tuning.
DEFAULT_WEIGHTS = {"lgb": 1/3, "xgb": 1/3, "rf": 1/3}


def _per_ticker_features(df, ticker):
    """30-min OHLCV → 8 single-stock features."""
    by_day = df.groupby(df["datetime"].dt.date, sort=False)
    momentum = by_day["close"].pct_change()
    vol_change = (df["volume"] / by_day["volume"].shift(1)) \
        .replace([np.inf, -np.inf], np.nan).fillna(0.0)
    bar_range = (df["high"] - df["low"]) / df["close"]

    rolling_mean_5 = momentum.rolling(5, min_periods=3).mean()
    rolling_std_5  = momentum.rolling(5, min_periods=3).std()
    rolling_vol_mean = df["volume"].rolling(5, min_periods=3).mean()
    rolling_volume_ratio = (df["volume"] / rolling_vol_mean) \
        .replace([np.inf, -np.inf], np.nan)
    rolling_mean_13 = momentum.rolling(13, min_periods=10).mean()
    rolling_std_13  = momentum.rolling(13, min_periods=10).std()

    return pd.DataFrame({
        "datetime":             df["datetime"],
        "ticker":               ticker,
        "momentum":             momentum,
        "volume_change":        vol_change,
        "bar_range":            bar_range,
        "rolling_mean_5":       rolling_mean_5,
        "rolling_std_5":        rolling_std_5,
        "rolling_volume_ratio": rolling_volume_ratio,
        "rolling_mean_13":      rolling_mean_13,
        "rolling_std_13":       rolling_std_13,
    }).dropna(subset=["momentum"])


def _per_ticker_extras_features(df_30, df_d, ticker):
    """Compute 5 extra features: time-of-day, dollar volume, VWAP distance, gap return.

    df_30: filtered + sorted 30-min OHLCV (caller has already filtered to regular hours).
    df_d:  daily OHLCV for this ticker (raw, will be sorted here).
    """
    # Time-of-day (390 minutes per session — 9:30 to 16:00)
    minutes_in_day = df_30["datetime"].dt.hour * 60 + df_30["datetime"].dt.minute
    minutes_from_open = (minutes_in_day - 9 * 60 - 30).clip(lower=0)
    minutes_to_close  = (16 * 60 - minutes_in_day).clip(lower=0)

    # Dollar volume — close × volume captures economic significance
    dollar_volume = df_30["close"] * df_30["volume"]

    # Intraday VWAP via cumulative sums per day
    df_30 = df_30.copy()
    df_30["date"] = df_30["datetime"].dt.date
    cum_dv  = dollar_volume.groupby(df_30["date"]).cumsum()
    cum_vol = df_30["volume"].groupby(df_30["date"]).cumsum()
    vwap = cum_dv / cum_vol
    vwap_distance = (df_30["close"] - vwap) / vwap

    # Gap return from daily file: (today_open - yesterday_close) / yesterday_close
    df_d = df_d.sort_values("datetime").reset_index(drop=True)
    df_d["prev_close"] = df_d["close"].shift(1)
    df_d["gap_return"] = (df_d["open"] - df_d["prev_close"]) / df_d["prev_close"]
    df_d["date"] = df_d["datetime"].dt.date
    gap_return = df_30["date"].map(dict(zip(df_d["date"], df_d["gap_return"])))

    return pd.DataFrame({
        "datetime":          df_30["datetime"],
        "ticker":            ticker,
        "minutes_from_open": minutes_from_open.astype("int16"),
        "minutes_to_close":  minutes_to_close.astype("int16"),
        "dollar_volume":     dollar_volume.astype("float32"),
        "vwap_distance":     vwap_distance.astype("float32"),
        "gap_return":        gap_return.astype("float32"),
    })


def _per_ticker_5min_features(df_5, ticker):
    """5-min OHLCV → 4 microstructure features per 30-min window."""
    by_day = df_5.groupby(df_5["datetime"].dt.date, sort=False)
    df_5 = df_5.copy()
    df_5["ret"]    = by_day["close"].pct_change()
    df_5["window"] = df_5["datetime"].dt.floor("30min")

    agg = df_5.groupby("window", sort=True).agg(
        intrabar_std=("ret", "std"),
        high_30=("high", "max"),
        low_30=("low", "min"),
        close_30=("close", "last"),
        vol_max=("volume", "max"),
        vol_sum=("volume", "sum"),
        tail_strength=("ret", "last"),
    )
    agg["close_position"]       = (agg["close_30"] - agg["low_30"]) / (agg["high_30"] - agg["low_30"])
    agg["volume_concentration"] = agg["vol_max"] / agg["vol_sum"]

    return agg[["intrabar_std", "close_position", "volume_concentration", "tail_strength"]] \
        .reset_index().rename(columns={"window": "datetime"}) \
        .assign(ticker=ticker)


class Model:

    def load(self, weights_path, config):
        weights_dir = pathlib.Path(weights_path).parent

        # Load all 3 models
        with open(weights_path, "rb") as f:
            self.m_lgb = pickle.load(f)
        with open(weights_dir / "weights_xgb.pkl", "rb") as f:
            self.m_xgb = pickle.load(f)
        with open(weights_dir / "weights_rf.pkl", "rb") as f:
            self.m_rf = pickle.load(f)

        # Sector mapping
        with open(weights_dir / "clusters.pkl", "rb") as f:
            self.cluster_map = pickle.load(f)

        # Config-driven knobs
        data_cfg = (config or {}).get("data", {})
        self.prepare_start = pd.Timestamp(
            data_cfg.get("prepare_start", DEFAULT_PREPARE_START)
        )
        self.predict_chunk_size = int(
            data_cfg.get("predict_chunk_size", DEFAULT_PREDICT_CHUNK_SIZE)
        )

        # Ensemble weights (sum to 1)
        ens_cfg = (config or {}).get("ensemble", {})
        self.weights = {
            "lgb": ens_cfg.get("w_lgb", DEFAULT_WEIGHTS["lgb"]),
            "xgb": ens_cfg.get("w_xgb", DEFAULT_WEIGHTS["xgb"]),
            "rf":  ens_cfg.get("w_rf",  DEFAULT_WEIGHTS["rf"]),
        }

    def prepare(self, data_dir):
        data_dir = pathlib.Path(data_dir)

        # Phase 1: per-ticker, read both 30-min and 5-min, compute features, merge
        files = sorted(data_dir.glob("*_30min.parquet"))
        parts = []
        for i, f in enumerate(files):
            ticker = f.stem.replace("_30min", "")

            df_30 = pd.read_parquet(f)
            df_30 = df_30[(df_30["datetime"].dt.time >= REGULAR_OPEN) &
                          (df_30["datetime"].dt.time <= REGULAR_CLOSE)]
            df_30 = df_30.sort_values("datetime").reset_index(drop=True)
            feat_30 = _per_ticker_features(df_30, ticker)

            f_5 = data_dir / f"{ticker}_5min.parquet"
            if f_5.exists():
                df_5 = pd.read_parquet(f_5)
                df_5 = df_5[(df_5["datetime"].dt.time >= REGULAR_OPEN) &
                            (df_5["datetime"].dt.time <= REGULAR_CLOSE)]
                df_5 = df_5.sort_values("datetime").reset_index(drop=True)
                feat_5 = _per_ticker_5min_features(df_5, ticker)
                feat = feat_30.merge(feat_5, on=["datetime", "ticker"], how="left")
            else:
                feat = feat_30
                for col in ["intrabar_std", "close_position", "volume_concentration", "tail_strength"]:
                    feat[col] = np.nan

            # Extras: time-of-day, dollar volume, VWAP distance, gap return
            f_d = data_dir / f"{ticker}_1day.parquet"
            if f_d.exists():
                df_d = pd.read_parquet(f_d)
                feat_extras = _per_ticker_extras_features(df_30, df_d, ticker)
                feat = feat.merge(feat_extras, on=["datetime", "ticker"], how="left")
            else:
                for col in ["minutes_from_open", "minutes_to_close",
                            "dollar_volume", "vwap_distance", "gap_return"]:
                    feat[col] = np.nan

            parts.append(feat)
            if (i + 1) % 200 == 0:
                print(f"  prepare phase 1 (per-ticker): {i + 1} / {len(files)}")

        print("  concatenating into long DataFrame...")
        features_df = pd.concat(parts, ignore_index=True)
        parts = None

        # Filter to training window
        n_before = len(features_df)
        features_df = features_df[features_df["datetime"] >= self.prepare_start].reset_index(drop=True)
        print(f"  filtered to >= {self.prepare_start.date()}: {len(features_df):,} rows (dropped {n_before - len(features_df):,})")

        # Cluster id
        features_df["cluster"] = (
            features_df["ticker"].map(self.cluster_map).fillna(-1).astype("int16")
        )

        # Phase 2a: cross-sectional
        print("  prepare phase 2a: cross-sectional features...")
        ts_groups = features_df.groupby("datetime", sort=False)
        for col in [
            "momentum", "volume_change", "bar_range",
            "rolling_mean_5", "rolling_mean_13", "rolling_volume_ratio",
        ]:
            cs_mean = ts_groups[col].transform("mean")
            cs_std  = ts_groups[col].transform("std")
            features_df[f"{col}_z"] = (features_df[col] - cs_mean) / cs_std
        features_df["momentum_rank"] = ts_groups["momentum"].rank(pct=True)

        # Phase 2b: sector-relative
        print("  prepare phase 2b: sector-relative features...")
        sector_groups = features_df.groupby(["datetime", "cluster"], sort=False)
        sector_mean = sector_groups["momentum"].transform("mean")
        sector_std  = sector_groups["momentum"].transform("std")
        features_df["momentum_relative_to_sector"] = features_df["momentum"] - sector_mean
        features_df["momentum_sector_z"] = (features_df["momentum"] - sector_mean) / sector_std

        # Phase 3: predict from each model in chunks; average with configured weights
        print(f"  prepare phase 3: predicting in chunks of {self.predict_chunk_size:,}...")
        n = len(features_df)
        preds = np.zeros(n, dtype=np.float64)
        for start in range(0, n, self.predict_chunk_size):
            end = min(start + self.predict_chunk_size, n)
            X = features_df[FEATURES].iloc[start:end]
            X_fill = X.fillna(0.0)   # RF needs no NaN

            preds[start:end] = (
                self.weights["lgb"] * self.m_lgb.predict(X)
                + self.weights["xgb"] * self.m_xgb.predict(X)
                + self.weights["rf"]  * self.m_rf.predict(X_fill)
            )
        features_df["pred"] = preds

        # Phase 4: build scores dict
        print("  prepare phase 4: building scores dict...")
        self.scores = {
            ts: dict(zip(g["ticker"].values, g["pred"].values))
            for ts, g in features_df.groupby("datetime", sort=False)
        }
        print(f"  done: {len(self.scores):,} timestamps")

    def predict(self, timestamp):
        ts = pd.Timestamp(timestamp)
        if ts in self.scores:
            return pd.Series(self.scores[ts])
        return pd.Series(dtype="float64")
