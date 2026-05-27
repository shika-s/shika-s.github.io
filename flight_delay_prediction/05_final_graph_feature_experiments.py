# Databricks notebook source
df = spark.read.parquet("dbfs:/student-groups/Group_4_4/data_12M/df_joined_1Y_features_plus_gf.parquet")
df.printSchema()


# COMMAND ----------

import pyspark.sql.functions as F

target_del15 = "DEP_DEL15"

num_cols = [
    "dep_delay15_24h_rolling_avg_by_origin_dayofweek",
    "dep_delay_24h_rolling_avg_by_origin_dayofweek",
    "dep_delay15_24h_rolling_avg_by_origin_carrier",
    "dep_delay15_24h_rolling_avg_by_origin",
    "dep_delay_24h_rolling_avg_by_origin_carrier",
    "dep_delay_24h_rolling_avg_by_origin",
    "CRS_DEP_TIME",
    "DEP_HOUR",
    "peak_travel_hour",
    "peak_travel_month",
    "HourlyRelativeHumidity",
    "HourlyPrecipitation",
    "HourlyWindSpeed",
    "DISTANCE",
    "DISTANCE_GROUP",
    "HourlyVisibility",
    "pagerank",
    "degree",
    "betweenness"
]

corr_rows_del15 = []
for c in num_cols:
    corr_val = df.select(F.corr(c, target_del15)).first()[0]
    corr_rows_del15.append((c, float(corr_val) if corr_val is not None else None))

corr_del15 = spark.createDataFrame(corr_rows_del15, ["feature", "corr_with_DEP_DEL15"])
corr_del15 = corr_del15.orderBy(F.desc("corr_with_DEP_DEL15"))

corr_del15.show(50, False)


# COMMAND ----------

target_delay = "DEP_DELAY"

corr_rows_delay = []
for c in num_cols:
    corr_val = df.select(F.corr(c, target_delay)).first()[0]
    corr_rows_delay.append((c, float(corr_val) if corr_val is not None else None))

corr_delay = spark.createDataFrame(corr_rows_delay, ["feature", "corr_with_DEP_DELAY"])
corr_delay = corr_delay.orderBy(F.desc("corr_with_DEP_DELAY"))

corr_delay.show(50, False)


# COMMAND ----------

import matplotlib.pyplot as plt

pdf_del15 = corr_del15.toPandas().dropna()

plt.figure(figsize=(10, 4))
plt.bar(pdf_del15["feature"], pdf_del15["corr_with_DEP_DEL15"])
plt.title("Feature Correlation with DEP_DEL15")
plt.ylabel("Correlation")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()


# COMMAND ----------

pdf_delay = corr_delay.toPandas().dropna()

plt.figure(figsize=(10, 4))
plt.bar(pdf_delay["feature"], pdf_delay["corr_with_DEP_DELAY"])
plt.title("Feature Correlation with DEP_DELAY")
plt.ylabel("Correlation")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()
