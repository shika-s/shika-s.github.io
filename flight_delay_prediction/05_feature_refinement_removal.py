# Databricks notebook source
# Load 5-year CP5 dataset
df = spark.read.parquet("dbfs:/student-groups/Group_4_4/checkpoint_5_final_clean_2015-2019.parquet")

print("Rows:", df.count())
print("Columns:", len(df.columns))
df.printSchema()

# COMMAND ----------

from pyspark.sql import functions as F

GROUP_PATH = "dbfs:/student-groups/Group_4_4"
path_cp5_5y = f"{GROUP_PATH}/checkpoint_5_final_clean_2015-2019.parquet"

df = spark.read.parquet(path_cp5_5y)

print("Initial columns:", len(df.columns))

# 1) Drop all high-correlation helper features
cols_high_corr = [c for c in df.columns if c.endswith("_high_corr")]
print("Dropping _high_corr columns:", len(cols_high_corr))

# 2) Drop zero / near-zero importance features from Phase 2 analysis
zero_low_importance = [
    # simple flags that showed zero importance
    "is_peak_month",
    "extreme_wind",
    "is_weekend",
    "is_holiday_month",
    "rapid_weather_change",
    "extreme_temperature",
    "low_visibility",
    "QUARTER",
    "wind_direction_sin",
    "wind_direction_cos",
    "is_rainy",
    "extreme_precipitation",
    "time_of_day_morning",
    "departure_month",
    "is_first_flight_of_aircraft",
    "is_business_hours",
    "weather_obs_lag_hours",
    "time_of_day_evening",
    "YEAR",
    "time_of_day_night",
    "time_of_day_early_morning",
    "departure_dayofweek",
    "time_of_day_afternoon",
    "month_cos",
    "distance_long",
    "distance_medium",
]

# keep only the ones that actually exist in this dataframe
zero_low_importance = [c for c in zero_low_importance if c in df.columns]
print("Dropping zero/low-importance columns:", len(zero_low_importance))

# 3) Apply drops
df = df.drop(*cols_high_corr).drop(*zero_low_importance)

print("Columns after feature refinement:", len(df.columns))

# COMMAND ----------

# Save refined feature set for Phase 3 modeling
out_path_refined = f"{GROUP_PATH}/checkpoint_5_final_clean_2015-2019_refined.parquet"

df.write.mode("overwrite").parquet(out_path_refined)
print("Saved refined 5Y features to:", out_path_refined)