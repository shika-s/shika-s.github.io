# Four Sigma Quantitative Finance Hackathon

> Team submission to the Four Sigma Quantitative Finance Hackathon (April–May 2026). Task: predict 30-minute stock-return rankings across 1,000 anonymized U.S. tickers, scored by Spearman rank correlation (IC) over a held-out test year.

## Team-authored files

| File | Author |
|---|---|
| [`shikha_feature_engineering_scaffold_colab.ipynb`](shikha_feature_engineering_scaffold_colab.ipynb) | Shikha Sharma — my Colab fork of the scaffold |
| [`submission/`](submission/) | Shikha Sharma — final ensemble (LightGBM / RF / XGBoost), weights, and config |
| [`test_submission.py`](test_submission.py) | Shikha Sharma — submission validation |
| [`jennifer_feature-engineering-scaffold.ipynb`](jennifer_feature-engineering-scaffold%20copy.ipynb) | Jennifer (teammate) — feature engineering work |
| [`merge_flat_directory_to_30m.py`](merge_flat_directory_to_30m.py), [`polars_separate.py`](polars_separate.py) | Jennifer (teammate) |
| [`Results.md`](Results.md), [`Model Results.xlsx`](Model%20Results.xlsx) | Team-produced results |

## My contribution

The team discussed feature ideas together and I built my own feature engineering pipeline in [`shikha_feature_engineering_scaffold_colab.ipynb`](shikha_feature_engineering_scaffold_colab.ipynb), creating features across five categories:

- Rolling statistics on 30-minute bars
- Cross-sectional z-scores
- Pseudo-sectors derived from spectral clustering on daily data
- Sector-momentum features built on those pseudo-sectors
- 5-minute bar features

I also built the final submission in [`submission/`](submission/) — a 3-tree equal-weighted LightGBM + Random Forest + XGBoost ensemble that scored +0.0713 mean Spearman IC.

## Data

The training data (per-ticker parquet files at 1-day, 30-minute, and 5-minute frequencies) belongs to Four Sigma and is not redistributed here.
