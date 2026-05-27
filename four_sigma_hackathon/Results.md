# Model Results

Validation IC results across all model experiments. All metrics on the **last 90 days** of training data (~708 timestamps).

**Scoring metric:** mean Spearman rank correlation (mean IC). IR = mean / std (diagnostic only).

## Best Result

The 3-tree ensemble of LightGBM + XGBoost + RandomForest (equal weights), trained on **26 features** with the **rank-target** objective using **the full 2015+ data window**, gives the best result:

| Submission | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|
| **3-tree equal ensemble (2015+ training)** | **+0.0727** | **0.1003** | **0.724** | **79.7%** |

This beats the trivial **flipped-momentum baseline** (+0.0478) by ~+0.025 mean IC and is right on **~8 of every 10 bars**.

The previous shipped submission (LGB+RF only on 2020+ data) hit +0.0713 / IR 0.707; switching to 2015+ training and re-introducing XGBoost into an equal-weighted 3-tree blend lifted both mean IC (+0.0014) and IR (+0.017). ElasticNet was trained on 2015+ data as well but excluded — it remains too weak to add diversity (see ensemble selection sections below).

## Full experiment log

### Baselines

| # | Feature set | Model | Target | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|---|---|
| 0 | momentum | naive baseline (return as score) | — | −0.0478 | 0.1253 | −0.38 | 33.6% |
| 0b | −momentum | flipped baseline (sign-flip) | — | +0.0478 | 0.1253 | 0.38 | 66.4% |

The negative IC of raw momentum reveals **short-term mean reversion** — flipping the sign gives a free +0.0478 baseline.

### Adding feature richness (return target)

| # | Feature set | Model | Target | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|---|---|
| 2 | base (3) | LightGBM | return | +0.0412 | 0.1120 | 0.37 | 65.3% |
| 3 | base + rolling stats (8) | LightGBM | return | +0.0441 | 0.1240 | 0.36 | 64.4% |
| 4 | + cross-sectional z-scores (15) | LightGBM | return | +0.0465 | 0.1139 | 0.41 | 65.7% |
| 5 | + sector clusters (17) | LightGBM | return | +0.0426 | 0.0757 | 0.56 | 73.4% |

**Key finding:** Adding sector clusters lowered mean IC slightly but **dramatically reduced std IC** (0.114 → 0.076), more than doubling pct > 0 (65.7% → 73.4%). The feature-engineering progression converged on a stable ~+0.045 mean IC.

### Ranker objective experiments (failed)

| # | Feature set | Model | Target | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|---|---|
| 6 | 17 features | LGBMRanker (rank_xendcg, exp gain) | rank | −0.0079 | 0.2069 | −0.04 | 50.3% |
| 7 | 17 features | LGBMRanker (rank_xendcg, linear gain) | rank | −0.0081 | 0.2053 | −0.04 | 50.0% |

**Key finding:** NDCG-based ranking objectives have built-in top-of-list bias that doesn't match Spearman scoring. Both LGBMRanker attempts produced near-random results despite reasonable NDCG values. **Reverted to LGBMRegressor.**

### Cross-algorithm comparison (return target, 17 features)

| # | Model | Target | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|---|
| 5 | LightGBM | return | +0.0426 | 0.0757 | 0.56 | 73.4% |
| 8 | XGBoost | return | +0.0412 | 0.0782 | 0.53 | 73.3% |
| 10 | RandomForest (30% sample) | return | +0.0437 | 0.0783 | 0.56 | 74.4% |

**Key finding:** All three tree algorithms converged to nearly identical performance (within noise) — the algorithmic ceiling for this feature set is ~+0.043 mean IC.

### Adding 5-min microstructure features (return target, 21 features)

| # | Model | Target | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|---|
| 11 | LightGBM | return | **+0.0567** | 0.0945 | 0.60 | 76.6% |
| 12 | XGBoost | return | +0.0565 | 0.0947 | 0.60 | 75.1% |
| 13 | RandomForest (30% sample) | return | +0.0557 | 0.0927 | 0.601 | 75.7% |

**Key finding:** Adding 4 microstructure features from 5-min data (`tail_strength`, `intrabar_std`, `close_position`, `volume_concentration`) gave a **+0.014 mean IC lift** across all algorithms. `tail_strength` (the return of the last 5-min bar) became the #1 most important feature.

### Switching to rank target (21 features)

| # | Model | Target | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|---|
| 14 | LightGBM | rank | **+0.0690** | 0.1014 | **0.680** | 78.2% |
| 15 | LightGBM (n_estimators=1500) | rank | +0.0690 | 0.1014 | 0.681 | 78.4% |
| 16 | XGBoost | rank | +0.0686 | 0.1015 | 0.676 | 77.8% |
| 17 | RandomForest (30% sample) | rank | +0.0676 | 0.0973 | **0.695** | **78.8%** |

**Key finding:** Switching from MSE-on-returns to MSE-on-cross-sectional-rank gave a further **+0.012 mean IC lift**. Loss landscape now aligned with Spearman scoring. `close_position` jumped to #1 feature; all three algorithms again converged. RandomForest had the highest IR thanks to lowest std.

### Ensemble (rank target, 21 features)

| # | Weights | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|
| 18a | equal (1/3 each) | **+0.0696** | 0.1011 | 0.688 | 79.0% |
| 18b | LGB-heavy (0.5 / 0.25 / 0.25) | +0.0695 | 0.1012 | 0.687 | 78.8% |
| 18c | LGB+RF only (0.5 / 0 / 0.5) | +0.0695 | 0.1003 | 0.693 | 79.4% |

**Key finding:** Modest ensemble lift (+0.0006 mean IC over best single model) — the three tree models are too correlated for big ensemble gains. Equal weights chosen for shipped submission.

### Adding 5 extra features (rank target, 26 features)

5 features added: `minutes_from_open`, `minutes_to_close`, `dollar_volume`, `vwap_distance`, `gap_return`. Time-of-day, economic significance of volume, intraday VWAP positioning, and overnight gap.

| # | Model | Target | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|---|
| 19 | LightGBM | rank | **+0.0712** | 0.1022 | **0.696** | 79.2% |
| 20 | XGBoost | rank | +0.0705 | 0.1016 | 0.694 | — |
| 21 | RandomForest (30% sample) | rank | +0.0688 | 0.0992 | 0.694 | — |
| 22 | ElasticNet (50% sample) | rank | +0.0639 | 0.1017 | 0.628 | 77.1% |

**Key finding:** All tree models lifted +0.0012 to +0.0022 mean IC. `dollar_volume` (#3 importance), `minutes_from_open` (#4), and `minutes_to_close` (#5) jumped into the top 5 features for LightGBM. ElasticNet got a smaller lift because time-of-day and dollar_volume scale poorly with linear models.

### Ensemble selection (rank target, 26 features)

Tested every weighting combination of the 4 models:

| # | Configuration | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|
| 23a | Equal 3-tree (LGB+XGB+RF) | +0.0714 | 0.1015 | 0.704 | 78.7% |
| 23b | LGB-heavy (0.5/0.25/0.25) | +0.0714 | 0.1018 | 0.701 | 79.2% |
| **23c** | **LGB+RF only (0.5/0.0/0.5)** | **+0.0713** | **0.1009** | **0.707** | **79.9%** |
| 23d | Equal 4-way (with ENet) | +0.0711 | 0.1021 | 0.697 | 79.4% |
| 23e | Trees 80% / ENet 20% | +0.0712 | 0.1019 | 0.699 | 79.7% |
| 23f | LGB+RF+ENet (no XGB) | +0.0710 | 0.1016 | 0.698 | 79.7% |

**Key finding:** **LGB+RF only beats every other ensemble on every metric except mean IC (where it's behind by 0.0001 — within noise).** Two takeaways:

1. **XGBoost adds no diversity over LightGBM** — both are gradient boosting with very similar inductive bias. Mixing them just averages correlated predictions. Replacing XGB's weight with more LGB or RF reduces variance.
2. **ElasticNet drags down the ensemble** — its lower individual IC (~0.007 below trees) outweighs its diversity benefit. The features that gave the recent IC lift (time-of-day, dollar_volume) are tree-only signals that linear models can't use well.

The submission was originally **LGB + RF (equal weights, 26 features, rank target, 2020+ training)** — this was the best 26-feature configuration on the 5-year window. It was later superseded by the 2015+ retrain (see next section).

### Extending the training window: 2015+ vs 2020+ (rank target, 26 features)

Hypothesis: the 2020-cutoff was originally chosen to skip COVID volatility, but the model uses cross-sectional rank features that should be regime-robust. Extending back to 2015 doubles the training data (~7M rows → ~12M rows train sample after the 30% subsample for RF) without changing the architecture.

Standalone results (each model retrained on 2015+):

| # | Model | Target | Mean IC | Std IC | IR | pct > 0 | vs 2020+ baseline |
|---|---|---|---|---|---|---|---|
| 24 | LightGBM | rank | +0.0721 | 0.0994 | **0.725** | 80.8% | +0.0009 mean / +0.029 IR |
| 25 | XGBoost | rank | **+0.0723** | 0.0994 | **0.727** | 79.8% | +0.0018 mean / +0.033 IR |
| 26 | RandomForest (30% sample) | rank | +0.0696 | 0.1019 | 0.683 | 78.8% | +0.0008 mean / −0.011 IR |
| 27 | ElasticNet (50% sample) | rank | +0.0639 | 0.1022 | 0.625 | 77.3% | ≈ flat |

**Key finding:** All three trees show meaningful improvement, with XGBoost gaining the most (+0.033 IR). The pattern is consistent: **more data mostly tightens variance rather than raising the mean** — the std IC dropped ~0.0028 across all models, lifting IR by ~0.03. This is exactly what we'd hope for when adding training data (lower regime-specific overfitting).

Ensemble selection on the 2015+ retrains:

| # | Configuration | Mean IC | Std IC | IR | pct > 0 |
|---|---|---|---|---|---|
| **28a** | **3-tree equal (LGB+XGB+RF, 1/3 each)** | **+0.0727** | **0.1003** | **0.724** | **79.7%** |
| 28b | LGB-heavy (0.5 / 0.25 / 0.25) | +0.0726 | 0.1001 | 0.726 | 79.7% |
| 28c | LGB+RF only (0.5 / 0 / 0.5) | +0.0724 | 0.1003 | 0.722 | 79.7% |
| 29a | Equal 4-way (with ENet) | +0.0717 | 0.1019 | 0.704 | 79.4% |
| 29b | Trees 80% / ENet 20% | +0.0719 | 0.1016 | 0.708 | 79.4% |
| 29c | LGB+RF+ENet (no XGB) | +0.0716 | 0.1016 | 0.705 | 79.7% |

**Two reversals from the 2020+ ensemble:**

1. **XGBoost now earns its slot.** With all three trees on the same data window, XGB and LGB de-correlate enough that adding XGB into the blend lifts mean IC by +0.0003 and IR by +0.002 over LGB+RF only. Previously (on 2020+ data) XGB was redundant; on 2015+ data it adds genuine diversity.
2. **ENet still doesn't earn its slot.** Even retrained on 2015+, ElasticNet's standalone IR (0.625) is too low to add value to a tree blend. Every 4-way variant is worse than the 3-tree blends.

The shipped submission switched to **3-tree equal weights (LGB + XGB + RF, 1/3 each), 26 features, rank target, 2015+ training**.

## Progression at a glance

| Stage | Mean IC | Δ |
|---|---|---|
| Naive momentum | −0.0478 | — |
| Flipped baseline (1-line rule) | +0.0478 | +0.0956 |
| LightGBM 3 features | +0.0412 | −0.0066 |
| + rolling stats (8 feat) | +0.0441 | +0.0029 |
| + cross-sectional (15 feat) | +0.0465 | +0.0024 |
| + sector clusters (17 feat) | +0.0426 | −0.0039 (but IR jumped 0.41 → 0.56) |
| + 5-min microstructure (21 feat) | +0.0567 | **+0.0141** |
| + rank target | +0.0690 | **+0.0123** |
| + ensemble (3 trees, 21 feat) | +0.0696 | +0.0006 |
| + 5 extras: time/volume/vwap/gap (26 feat) | +0.0712 | **+0.0016** |
| + LGB+RF ensemble (drop XGB) | +0.0713 | +0.0001 (mean) / +0.019 (IR) |
| + Extend training window to 2015+ | +0.0724 | +0.0011 (mean) / +0.015 (IR) |
| + Re-introduce XGB into 3-tree equal-weights ensemble | +0.0727 | +0.0003 (mean) / +0.002 (IR) |

**Four biggest wins:** 5-min microstructure features (+0.014), switching to rank target (+0.012), the 5 extras (+0.002 mean / +0.011 IR via reduced std), and extending training back to 2015 (+0.0014 mean / +0.017 IR via reduced variance).

## Feature descriptions

All 21 features are computed at the 30-min bar level — each row describes the bar starting at the labeled timestamp. The forward-looking target is the *next* 30-min bar's return, set up so the model uses backward-looking context to predict the upcoming bar.

### Per-ticker features (8)

These describe a single stock's bar without reference to other stocks. Computed within each trading day to avoid contamination from the overnight gap.

| Feature | Formula | What it captures |
|---|---|---|
| `momentum` | `(close - close_prev) / close_prev` within day | The current bar's return — the dominant directional signal |
| `volume_change` | `volume / volume_prev` within day | Bar-to-bar volume change ratio (1.0 = unchanged, >1 = elevated) |
| `bar_range` | `(high - low) / close` | Intra-bar volatility / range as % of price |
| `rolling_mean_5` | mean of last 5 bars' `momentum` | ~2.5 hour rolling return (intra-session trend) |
| `rolling_std_5` | std of last 5 bars' `momentum` | Recent return volatility (~2.5 hour window) |
| `rolling_volume_ratio` | `volume / mean(volume, last 5 bars)` | Is current volume elevated vs recent norm |
| `rolling_mean_13` | mean of last 13 bars' `momentum` | ~Full-day rolling return (session-spanning trend) |
| `rolling_std_13` | std of last 13 bars' `momentum` | Full-day return volatility |

### Cross-sectional features (vs. universe) (7)

These compare a stock to **all other stocks at the same timestamp**, isolating idiosyncratic from market-wide moves. All computed via `groupby(datetime)` on the per-ticker features.

| Feature | Formula | What it captures |
|---|---|---|
| `momentum_z` | `(momentum - cs_mean) / cs_std` per timestamp | Standardized momentum — how unusual is this stock's move vs peers right now |
| `momentum_rank` | percentile rank of momentum per timestamp (0 to 1) | Rank-based version of `momentum_z` — robust to outliers, captures non-linear positioning |
| `bar_range_z` | z-score of `bar_range` vs universe | Is this stock unusually volatile this bar |
| `volume_change_z` | z-score of `volume_change` vs universe | Is volume change unusual vs peers right now |
| `rolling_mean_5_z` | z-score of `rolling_mean_5` vs universe | Cross-sectional positioning of 2.5-hour return |
| `rolling_mean_13_z` | z-score of `rolling_mean_13` vs universe | Cross-sectional positioning of full-day return |
| `rolling_volume_ratio_z` | z-score of `rolling_volume_ratio` vs universe | Is sustained volume elevation unusual vs peers |

### Sector-relative features (2)

These compare a stock to **its sector cluster** at the same timestamp. Sectors are derived via spectral clustering on residual return correlations (see Sector Clustering below).

| Feature | Formula | What it captures |
|---|---|---|
| `momentum_relative_to_sector` | `momentum - sector_mean(momentum)` per timestamp | Idiosyncratic momentum after removing sector factor — the stock's outperformance vs its peer group |
| `momentum_sector_z` | `(momentum - sector_mean) / sector_std` per timestamp | Standardized sector-relative momentum |

### 5-min microstructure features (4)

These describe what happened *inside* each 30-min bar by aggregating the 6 underlying 5-min bars. Captures bar shape and trading-pattern detail that single OHLCV can't express.

| Feature | Formula | What it captures |
|---|---|---|
| `intrabar_std` | std of 6 underlying 5-min returns | Within-bar volatility (different from `bar_range` = high − low) |
| `close_position` | `(close - low) / (high - low)` of the 30-min bar | Where in the bar's range it closed: 0 = closed at low, 1 = closed at high |
| `volume_concentration` | `max(5-min volume) / sum(5-min volume)` | Spike vs steady volume distribution within the bar (high = front/back-loaded) |
| `tail_strength` | return of the last 5-min bar within the window | Did momentum accelerate or fade at the bar's end |

### Sector clustering

Pseudo-sectors are derived from data, since the hackathon dataset is anonymized (no real sector labels):

1. Read daily OHLCV for all 1000 tickers, 2015-2025
2. Compute log returns
3. **Subtract daily cross-sectional mean** (removes the market factor — without this, almost all stocks correlate ~0.5 with each other and clustering fails)
4. Compute pairwise correlation matrix on residuals
5. Convert to affinity: `max(0, corr)` (clip negatives to 0)
6. **Spectral clustering** with K=20 (after AgglomerativeClustering produced 1 mega-cluster + 19 singletons)
7. Save mapping `{ticker → cluster_id}` to `submission/clusters.pkl`

Result: ~94% of tickers in usable clusters of 11–204 members. Largest cluster (204) probably tech; remaining clusters likely correspond to GICS-style sectors (financials, energy, healthcare, etc.) but are unverified.

### Key feature engineering decisions

- **All features are backward-looking.** The target alone is forward-looking (next bar's return).
- **Rolling features use min_periods < window** to tolerate the NaN at first bar of each day.
- **Day-grouped operations** for `momentum`, `volume_change` prevent overnight-gap contamination.
- **Feature importance progression** showed `momentum_relative_to_sector`, `tail_strength`, and `close_position` as the most predictive features across all model architectures.

## Feature importance — 21-feature rank-target run (intermediate)

Captured before the 5 extras were added. Each model uses a different importance metric, so absolute values aren't directly comparable. Below shows **% of total importance** within each model (so columns sum to 100%) plus the **average rank across all 3 models** for cross-model comparison.

Sorted by average rank (best across all 3 models at top):

| # | Feature | LightGBM % | XGBoost % | RandomForest % | Avg rank |
|---|---|---|---|---|---|
| 1 | `close_position` | **19.4** | 12.6 | **18.9** | **1.3** |
| 2 | `tail_strength` | 16.3 | **16.5** | 17.1 | **1.7** |
| 3 | `rolling_volume_ratio` | 11.8 | 8.4 | 6.6 | 3.0 |
| 4 | `momentum_relative_to_sector` | 4.3 | 5.8 | 6.5 | 5.3 |
| 5 | `rolling_volume_ratio_z` | 9.0 | 6.1 | 4.3 | 5.7 |
| 6 | `volume_concentration` | 5.6 | 5.1 | 4.6 | 6.3 |
| 7 | `momentum_sector_z` | 3.7 | 5.0 | 5.9 | 6.7 |
| 8 | `rolling_mean_13_z` | 4.7 | 4.9 | 2.5 | 9.7 |
| 9 | `rolling_mean_5_z` | 3.3 | 4.0 | 2.9 | 10.0 |
| 9 | `volume_change` | 2.4 | 3.2 | 4.6 | 10.0 |
| 11 | `intrabar_std` | 3.7 | 4.0 | 2.6 | 10.3 |
| 12 | `momentum_rank` | 1.9 | 3.2 | 5.0 | 10.7 |
| 13 | `bar_range_z` | 2.7 | 3.0 | 2.3 | 13.3 |
| 14 | `momentum` | 2.4 | 2.9 | 2.5 | 14.0 |
| 14 | `volume_change_z` | 1.8 | 2.9 | 2.5 | 14.0 |
| 16 | `momentum_z` | 1.0 | 2.4 | 4.2 | 15.3 |
| 17 | `bar_range` | 1.5 | 2.4 | 1.4 | 17.3 |
| 18 | `rolling_mean_13` | 1.4 | 2.2 | 1.5 | 17.7 |
| 19 | `rolling_std_13` | 1.2 | 2.1 | 1.5 | 18.0 |
| 20 | `rolling_mean_5` | 1.0 | 2.0 | 1.5 | 19.7 |
| 21 | `rolling_std_5` | 0.7 | 1.7 | 1.1 | 21.0 |

### What the importance table reveals

**1. Universal top 3.** `close_position`, `tail_strength`, `rolling_volume_ratio` are top-3 in **every model**. Together they account for 33–47% of each model's attention. These three features are the dominant signal — losing any one would hurt the model significantly.

**2. The 5-min microstructure features dominate.** `close_position` and `tail_strength` (both 5-min-derived) take the #1 and #2 spots. `volume_concentration` (also 5-min) lands in the top 6. The 5-min microstructure addition was the single biggest IC lift in the project for good reason.

**3. Sector clustering paid off.** `momentum_relative_to_sector` (#4 avg rank) and `momentum_sector_z` (#7) are both in the top 7 across all models. The spectral clustering on residual returns produced sectors the model finds meaningful — even though they're unverified pseudo-sectors.

**4. Universe z-scores are mixed.** Volume-based z-scores (`rolling_volume_ratio_z`, `volume_change_z`) outperform return-based z-scores (`momentum_z`, `bar_range_z`). The model doesn't gain much from "is this stock's return unusual vs peers" — but does gain from "is this stock's volume unusual vs peers."

**5. Long-window rolling features are weak.** `rolling_mean_13`, `rolling_std_13`, `rolling_mean_5`, `rolling_std_5` are at the bottom (avg ranks 17–20). Their *z-scored* versions rank higher, which suggests rolling stats are useful only when normalized cross-sectionally — the raw values add little.

**6. RandomForest disagrees most.** RF puts `momentum_z` (#16 avg) and `momentum_rank` (#12 avg) much higher than the boosters do (LGB has them at 19 and 14). This is the source of ensemble diversity — RF "sees" something in pure cross-sectional rank features that boosting misses.

### Practical implication

For a leaner model, the bottom 5 (`bar_range`, `rolling_mean_13`, `rolling_std_13`, `rolling_mean_5`, `rolling_std_5`) could likely be removed with minimal IC loss. They contribute <2% of each model's attention. A leaner 16-feature model would train faster with similar performance.

## Feature importance — final 26-feature rank-target run, 2015+ training (shipped)

Importances from the three models that compose the shipped equal-weighted ensemble (LGB + XGB + RF, all trained on 2015+ data, 26 features, rank target). Each model uses a different importance metric, so absolute values aren't directly comparable. Below shows **% of total importance** within each model (so each column sums to 100%) plus the **average rank across all 3 models** for cross-model comparison.

Sorted by average rank across all 3 models:

| # | Feature | LightGBM % | XGBoost % | RandomForest % | Avg rank |
|---|---|---|---|---|---|
| 1 | `close_position` | **19.8** | **15.0** | **21.0** | **1.0** |
| 2 | `tail_strength` | 17.9 | 11.0 | 16.9 | **2.3** |
| 3 | `dollar_volume` ⭐ | 12.9 | 6.9 | 8.5 | **3.7** |
| 4 | `minutes_to_close` ⭐ | 5.2 | **14.5** | 4.2 | **5.0** |
| 4 | `minutes_from_open` ⭐ | 8.5 | 7.9 | 4.7 | **5.0** |
| 6 | `momentum_sector_z` | 4.3 | 5.8 | 5.0 | 6.0 |
| 7 | `momentum_relative_to_sector` | 3.4 | 3.9 | 5.8 | 6.3 |
| 8 | `momentum_rank` | 1.7 | 2.3 | 5.1 | 10.0 |
| 8 | `rolling_mean_5_z` | 2.4 | 3.3 | 2.3 | 10.0 |
| 10 | `rolling_mean_13_z` | 4.0 | 3.9 | 1.9 | 10.3 |
| 10 | `bar_range_z` | 3.0 | 2.6 | 2.2 | 10.3 |
| 12 | `intrabar_std` | 2.3 | 2.6 | 2.2 | 11.7 |
| 13 | `momentum` | 2.0 | 2.1 | 2.2 | 13.0 |
| 13 | `volume_concentration` | 1.7 | 2.0 | 2.5 | 13.0 |
| 15 | `momentum_z` | 0.6 | 2.0 | 3.8 | 15.7 |
| 16 | `vwap_distance` ⭐ | 1.4 | 1.7 | 1.8 | 16.3 |
| 17 | `rolling_volume_ratio_z` | 1.0 | 1.4 | 1.2 | 18.7 |
| 17 | `rolling_volume_ratio` | 0.9 | 1.3 | 2.0 | 18.7 |
| 19 | `rolling_std_13` | 1.2 | 1.5 | 0.8 | 19.0 |
| 20 | `bar_range` | 0.9 | 1.4 | 1.0 | 19.7 |
| 21 | `rolling_mean_13` | 1.0 | 1.3 | 1.0 | 20.0 |
| 22 | `gap_return` ⭐ | 1.4 | 1.3 | 0.5 | 21.0 |
| 23 | `rolling_mean_5` | 0.6 | 1.2 | 0.9 | 22.3 |
| 24 | `volume_change` | 0.5 | 1.1 | 1.4 | 22.7 |
| 25 | `rolling_std_5` | 0.6 | 1.1 | 0.6 | 24.3 |
| 26 | `volume_change_z` | 0.6 | 1.1 | 0.6 | 25.0 |

⭐ = features added in the 26-feature run (vs the earlier 21-feature run)

### What the importance table reveals

**1. The "universal top 3" is now unanimous.** `close_position`, `tail_strength`, and `dollar_volume` are #1, #2, #3 across **every** model — and `close_position` is #1 in all three (avg rank 1.0). The longer training window concentrated importance on these three: with 2015+ data, LGB, XGB, and RF all converged on the same dominant signals, whereas on 2020+ data the rankings were noisier. Together these three features capture **44–55%** of each model's attention.

**2. Time-of-day signals stayed in the top 5 but consolidated.** `minutes_to_close` and `minutes_from_open` are tied at #4 (avg rank 5.0). XGBoost still values `minutes_to_close` very highly (#2 at 14.5%) — the original "XGB sees time-of-day differently" pattern persists.

**3. Sector features remain the second tier.** `momentum_sector_z` (#6) and `momentum_relative_to_sector` (#7) are right behind the top 5 across all three models. Spectral clustering on residual returns continues to pay for itself.

**4. RandomForest still leans on rank features.** RF places `momentum_rank` (#8 avg) and `momentum_z` (#15 avg) much higher than LGB does (LGB ranks them #13 and #24). This is the source of ensemble diversity — RF "sees" something in pure cross-sectional rank features that boosting misses, which is why RF + LGB beats either alone even though they have similar standalone IC.

**5. `vwap_distance` and `gap_return` are still middle/bottom.** Both around 1–2% of model attention — useful but not dominant. `gap_return` is the lowest-ranked of the 5 extras (#22), confirming the earlier observation that trees don't extract much from overnight gaps relative to bar-internal signals.

**6. Bottom of the table is universally weak.** `volume_change_z`, `rolling_std_5`, `volume_change`, `rolling_mean_5` are all <1.5% across every model. Same set of weak features as the 2020+ training — the longer window didn't rescue any of them.

### Practical implication for a leaner model

The bottom 5–6 features (`gap_return`, `rolling_mean_5`, `volume_change`, `rolling_std_5`, `volume_change_z`) could be removed for a 20–21 feature model with minimal IC loss (~<0.001). The dominant signal lives in the top 10 features, which together account for ~75% of each model's attention.

`gap_return` is interesting — it ranked higher in the linear ElasticNet (top 10 coefficients) but trees don't value it as much. This is one of the few features where linear and trees genuinely disagree.

## Configuration of submission

- **Model:** 3-tree ensemble (LightGBM + XGBoost + RandomForest), equal weights (1/3 each). ElasticNet trained but excluded.
- **Features:** 26 total
  - 8 per-ticker (30-min): momentum, volume_change, bar_range, rolling_mean_5/13, rolling_std_5/13, rolling_volume_ratio
  - 7 cross-sectional vs. universe: momentum_z, momentum_rank, bar_range_z, volume_change_z, rolling_mean_5_z, rolling_mean_13_z, rolling_volume_ratio_z
  - 2 sector-relative (vs. cluster): momentum_relative_to_sector, momentum_sector_z
  - 4 5-min microstructure: intrabar_std, close_position, volume_concentration, tail_strength
  - 5 extras: minutes_from_open, minutes_to_close, dollar_volume, vwap_distance, gap_return
- **Sector clustering:** spectral clustering on residual return correlations 2015–2025, K=20
- **Training window:** 2015-01-01 to (test cutoff − 90 days) — full ~10 year window
- **Validation:** last 90 days (~708 timestamps)
- **Target:** cross-sectional percentile rank of next-bar return
- **Hyperparameters:** see `submission/config.json`

## Files

- `submission/lightGBM_model.py` — single-model submission (LightGBM only)
- `submission/ensemble_model.py` — ensemble submission (3-model average)
- `submission/weights.pkl`, `weights_xgb.pkl`, `weights_rf.pkl` — trained models
- `submission/clusters.pkl` — sector cluster mapping
- `submission/config.json` — all hyperparameters and metadata
