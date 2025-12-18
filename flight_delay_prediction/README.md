flight_delay_predictions

Predicting Flight Departure Delays: A Time-Series Analysis Approach

This project aims to predict whether a domestic U.S. flight will experience a departure delay of 15 minutes or more using information available at the 2-hour cutoff. Our hypothesis is that flight delays can be predicted using a combination of temporal patterns, weather conditions, airport congestion metrics, carrier performance history, network centrality, and time-series features available before departure. In Phase 1, we used the 3 Month OTPW (On-Time Performance and Weather) dataset, which combines flight data from the U.S. Department of Transportation's Bureau of Transportation Statistics with weather observations from the National Oceanic and Atmospheric Administration. Our Phase 2 analysis used the complete 2015 calendar year containing 5.7 million flights. Our Phase 3 analysis uses the complete 2015-2019 dataset containing 31.1 million flights. The dataset exhibits class imbalance with 82% on-time flights. We implemented four modeling approaches using F₂-score as the primary metric to prioritize recall over precision. Models were trained on 2015-2017 data (16.8M flights), validated on 2018 (7.1M flights), and tested on 2019 as blind holdout (7.3M flights).

Contributions:
* Implemented initial version of data cleaning and feature engineering for the OTPW dataset provided.
* Built and integrated graph-based airport network features (PageRank, Betweenness and Degree centrality) and temporal lag features with strict leakage prevention logic.
* Built, tuned, and evaluated GBTRegressor and SparkGBRegressor models, including GridSearch time series cross validation for hyperparameter tuning  and undersampling for class imbalance.
* Performed Error analysis and blind testing, and authored regression and 2-Stage modeling results, tables, and figures for the final report and presentation.
