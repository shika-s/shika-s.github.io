# Databricks notebook source
# MAGIC %md
# MAGIC ## 1: Data Cleaning

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Target Variable Overview and Label Cleaning
# MAGIC Before performing exploratory analysis, we cleaned the raw OTPW dataset to ensure data quality.

# COMMAND ----------

from pyspark.sql.functions import col

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))


# COMMAND ----------

# OTPW
df_otpw = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
display(df_otpw)

# COMMAND ----------

path = "dbfs:/mnt/mids-w261/OTPW_3M_2015.csv"

df = (spark.read.format("csv")
      .option("header", "true")
      .option("inferSchema", "true") 
      .load(path)
     ).cache()

print(f"Rows: {df.count():,}, Cols: {len(df.columns)}")
df.printSchema()
display(df.limit(5))

# COMMAND ----------

# Check total rows, columns, and a few sample records
print(f"Rows: {df.count():,}, Columns: {len(df.columns)}")
display(df.limit(5))


# COMMAND ----------

total = df.count()

target_summary = (
    df.withColumn(
        "DEP_DEL15_status",
        F.when(F.col("DEP_DEL15").isNull(), "Null")
         .when(F.col("DEP_DEL15") == 1, "Delayed (≥15 min)")
         .otherwise("On Time (<15 min)")
    )
    .groupBy("DEP_DEL15_status")
    .agg(F.count("*").alias("count"))
    .withColumn("percentage", F.round(F.col("count")/F.lit(total)*100, 2))
    .orderBy("DEP_DEL15_status")
)

display(target_summary)


# COMMAND ----------

# Remove rows where target variable is null
df = df.filter(df.DEP_DEL15.isNotNull())

print(f"After removing null targets: {df.count():,} rows remain.")

# COMMAND ----------

# MAGIC %md
# MAGIC **1.1: Summary**
# MAGIC -  Approximately 20% of flights are delayed (≥15 min), 77% are on-time, 
# MAGIC -  and 3% have missing target labels. 
# MAGIC -  remove the rows with null DEP_DEL15 values 
# MAGIC -  and keep both delayed and on-time flights for modeling.

# COMMAND ----------

total = df.count()

# Count non-null values for every column
non_null_agg = df.agg(*[F.count(c).alias(c) for c in df.columns])

# Reshape into (column, non_null)
nulls_long = non_null_agg.selectExpr(
    "stack({}, {}) as (column, non_null)".format(
        len(df.columns),
        ", ".join([f"'{c}', `{c}`" for c in df.columns])
    )
).withColumn("null_count", F.lit(total) - F.col("non_null")) \
 .withColumn("null_pct", F.round(F.col("null_count") / F.lit(total) * 100, 2)) \
 .orderBy(F.desc("null_pct"))

display(nulls_long)

# COMMAND ----------

# Drop columns with excessive missingness (>80%), always keep the target
threshold = 80.0
cols_to_drop = [r["column"] for r in nulls_long.filter(F.col("null_pct") > threshold).collect()
                if r["column"] != "DEP_DEL15"]

print(f"Drop {len(cols_to_drop)} columns (> {threshold}% null).")
df = df.drop(*cols_to_drop)
print(f"Remaining columns: {len(df.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC - removed columns with more than 80% missing values. 
# MAGIC - Kept the target column [DEP_DEL15] intact

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Column Selection and Type Normalization (for EDA)

# COMMAND ----------

# Select relevant columns ----
cols_to_check  = [
    "DEP_DEL15",
    "CRS_DEP_HOUR", "DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER", "ORIGIN", "DEST",
    "DISTANCE",
    "HourlyDryBulbTemperature",
    "HourlyWindSpeed",
    "HourlyPrecipitation"
]
df_base = df.select(*[c for c in cols_keep if c in df.columns])


# COMMAND ----------

df1 = df_base
row_cnt = df1.count()

null_report = (
    df1.select([
        F.count(F.when(F.col(c).isNull(), c)).alias(f"{c}__nulls")
        for c in df1.columns
    ]).withColumn("rows", F.lit(row_cnt))
)

# Display as (column, nulls, null_rate)
long_nulls = []
for c in df1.columns:
    long_nulls.append(
        F.struct(F.lit(c).alias("column"),
                 F.col(f"{c}__nulls").alias("nulls"),
                 (F.col(f"{c}__nulls")/F.col("rows")).alias("null_rate"))
    )

null_long_df = null_report.select(F.array(*long_nulls).alias("arr")) \
                          .select(F.explode("arr").alias("rec")) \
                          .select("rec.*") \
                          .orderBy(F.desc("null_rate"))
null_long_df.show(truncate=False)

# COMMAND ----------

# Remove non-numeric characters before casting to double; unparseable values become null
def to_double_safe(colname: str):
    # Keep digits, dot, minus; empty after cleanup -> null
    cleaned = F.regexp_replace(F.col(colname), r"[^0-9\.\-]", "")
    return F.when(cleaned == "", None).otherwise(cleaned).cast("double")

df2 = df1
for c in ["HourlyDryBulbTemperature", "HourlyWindSpeed", "HourlyPrecipitation"]:
    if c in df2.columns:
        df2 = df2.withColumn(c, to_double_safe(c))

# COMMAND ----------

# ---- Step 6: Enforce valid ranges and canonical dtypes ----
# Ensure label is 0/1, hour in 0–23, weekday in 1–7, distance positive and plausible
if "DEP_DEL15" in df2.columns:
    df2 = df2.withColumn(
        "DEP_DEL15",
        F.when(F.col("DEP_DEL15").isin(0, 1), F.col("DEP_DEL15")).otherwise(None)
    ).withColumn("DEP_DEL15", F.col("DEP_DEL15").cast("int"))

if "CRS_DEP_HOUR" in df2.columns:
    df2 = df2.withColumn(
            "CRS_DEP_HOUR",
            F.when((F.col("CRS_DEP_HOUR").cast("int") >= 0) & (F.col("CRS_DEP_HOUR").cast("int") <= 23),
                   F.col("CRS_DEP_HOUR").cast("int"))
             .otherwise(F.lit(None))  # invalid hours become null
        )

if "DAY_OF_WEEK" in df2.columns:
    df2 = df2.withColumn(
            "DAY_OF_WEEK",
            F.when((F.col("DAY_OF_WEEK").cast("int") >= 1) & (F.col("DAY_OF_WEEK").cast("int") <= 7),
                   F.col("DAY_OF_WEEK").cast("int"))
             .otherwise(F.lit(None))
        )
    
if "DISTANCE" in df2.columns:
    df2 = df2.withColumn(
        "DISTANCE",
        F.when((F.col("DISTANCE") > 0) & (F.col("DISTANCE") < 5000), F.col("DISTANCE")).otherwise(None)
        .cast("double")
    )

# Airport/carrier codes: keep alphanumeric only (avoid punctuation/hidden chars)
for c in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]:
    if c in df2.columns:
        df2 = df2.withColumn(c, F.regexp_replace(F.col(c), r"[^A-Z0-9]", ""))
        df2 = df2.withColumn(c, F.when(F.length(F.col(c)) == 0, None).otherwise(F.col(c)))


# COMMAND ----------

# Sanity-check distributions to confirm conversions look reasonable
qc_cols = [c for c in ["DEP_DEL15","CRS_DEP_HOUR","DAY_OF_WEEK","DISTANCE",
                       "HourlyDryBulbTemperature","HourlyWindSpeed","HourlyPrecipitation"] if c in df2.columns]

# Basic stats for numeric columns
df2.select([F.mean(c).alias(f"{c}_mean") for c in qc_cols if dict(df2.dtypes)[c] in ("int", "double")]).show()

# Cardinalities for key categorical IDs
for c in ["OP_UNIQUE_CARRIER", "ORIGIN", "DEST"]:
    if c in df2.columns:
        print(c, "distinct =", df2.select(c).distinct().count())


# COMMAND ----------

# ---- Step 8: Baseline imputation (median for numeric fields) ----
# Keep it simple for Phase 1; can refine later (e.g., groupwise median by ORIGIN×hour)
median_cols = [c for c in ["HourlyDryBulbTemperature","HourlyWindSpeed","HourlyPrecipitation","DISTANCE"] if c in df2.columns]
if median_cols:
    med_row = df2.select([F.expr(f"percentile_approx({c}, 0.5)").alias(c) for c in median_cols]).first()
    med_map = {c: med_row[c] for c in median_cols if med_row[c] is not None}
    df3 = df2.fillna(med_map)
else:
    df3 = df2

# Optionally drop rows missing required keys for modeling (keep a copy before dropping)
required_keys = [c for c in ["ORIGIN","DEST","OP_UNIQUE_CARRIER","CRS_DEP_HOUR","DAY_OF_WEEK"] if c in df3.columns]
df_clean = df3.na.drop(subset=required_keys)  # or keep df3 if you want to impute keys later

df_clean.printSchema()
print("Rows before:", row_cnt, " | after basic cleaning:", df_clean.count())


# COMMAND ----------

from functools import reduce
from pyspark.sql import functions as F

after_cnt = df_clean.count()
summaries = []

for c, t in df_clean.dtypes:
    # Make sure the column actually exists
    if c not in df_clean.columns:
        continue

    # Base summary: column name, dtype, null rate
    base = df_clean.select(
        F.lit(c).alias("column"),
        F.lit(t).alias("dtype"),
        (F.sum(F.col(c).isNull().cast("int")) / F.lit(after_cnt)).alias("null_rate")
    )

    # If numeric column, include min / median / max
    if t in ("int", "bigint", "double", "float"):
        base = df_clean.select(
            F.lit(c).alias("column"),
            F.lit(t).alias("dtype"),
            (F.sum(F.col(c).isNull().cast("int")) / F.lit(after_cnt)).alias("null_rate"),
            F.min(F.col(c)).alias("min"),
            F.expr(f"percentile_approx({c}, 0.5)").alias("median"),
            F.max(F.col(c)).alias("max")
        )

    summaries.append(base)

# Combine all summaries
dq_summary = reduce(lambda a, b: a.unionByName(b, allowMissingColumns=True), summaries)
dq_summary.orderBy(F.desc("null_rate")).show(truncate=False)



# COMMAND ----------

# MAGIC %md
# MAGIC ## 2: Exploratory Data Analysis (EDA)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1: Time-Based Delay Patterns
# MAGIC In this section, we explore how flight delay rates vary across different time dimensions.
# MAGIC - **By Hour of Day:** Early-morning flights (5–8 AM) tend to have the lowest delay rates, while evening flights (5–9 PM) experience the highest delays due to accumulated congestion.
# MAGIC - **By Day of Week:** Delay patterns often rise toward the weekend, reflecting increased flight volume and operational complexity.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.1 Hourly Delay Rate

# COMMAND ----------


# 2.1: Extract scheduled departure hour (0–23)
# CRS_DEP_TIME is stored as HHMM (e.g., 945 = 9:45am)
# df = df_clean.withColumn("CRS_DEP_HOUR", (F.col("CRS_DEP_TIME")/100).cast("int"))

# Average delay rate by hour of the day
delay_by_hour = (
    df_clean.groupBy("CRS_DEP_HOUR")
      .agg(F.round(F.mean("DEP_DEL15") * 100, 2).alias("delay_rate_pct"))
      .orderBy(F.desc("delay_rate_pct"))
)
display(delay_by_hour)

# COMMAND ----------

# MAGIC %md
# MAGIC **2.1.1 Observation**
# MAGIC
# MAGIC The hourly trend shows that flights departing **early in the morning (around 5–8 AM)** 
# MAGIC have the **lowest delay rates**, while those departing in the **late afternoon and evening (around 5–9 PM)** 
# MAGIC experience the **highest delays**.  
# MAGIC This pattern reflects the **cumulative operational congestion** that builds up throughout the day.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.1.2 Day-of-Week Delay Rate

# COMMAND ----------

# Average delay rate by day of week (1=Mon, 7=Sun)
delay_by_day = (
    df_clean.groupBy("DAY_OF_WEEK")
      .agg(F.round(F.mean("DEP_DEL15") * 100, 2).alias("delay_rate_pct"))
      .orderBy(F.desc("delay_rate_pct"))
)
display(delay_by_day)

# COMMAND ----------

# MAGIC %md
# MAGIC **2.1.2 Observation**
# MAGIC
# MAGIC Flights departing on **Mondays** and **weekends (especially Sundays)** 
# MAGIC show the **highest delay rates**, while **midweek flights (Tuesday–Wednesday)** 
# MAGIC tend to have the **lowest delays**.  
# MAGIC This pattern likely reflects heavier passenger traffic and tighter scheduling 
# MAGIC at the beginning and end of the week.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Airline Delay Patterns
# MAGIC In this section, we analyze **airline-level delay behavior** to identify carriers 
# MAGIC that tend to have higher on-time performance issues.  
# MAGIC Examining delay rates by carrier helps uncover whether delays are primarily influenced 
# MAGIC by **airline-specific operational efficiency**, **network size**, or **route congestion**. 

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.2 Airline Delay Rate vs Flight Volume

# COMMAND ----------

# Airline-level delay statistics
delay_by_carrier = (
    df_clean.groupBy("OP_UNIQUE_CARRIER")
      .agg(
          F.count("*").alias("num_flights"),
          F.round(F.mean("DEP_DEL15") * 100, 2).alias("delay_rate_pct")
      )
      .orderBy(F.desc("delay_rate_pct"))
)

display(delay_by_carrier)

# COMMAND ----------

# MAGIC %md
# MAGIC **2.2 Observation**
# MAGIC
# MAGIC This dual-axis chart compares **flight volume** (bars) and **delay rate (%)** (line) 
# MAGIC across major U.S. carriers.  
# MAGIC Smaller airlines such as **Frontier (F9)** and **Envoy (MQ)** exhibit 
# MAGIC the highest delay rates (above 28%), possibly due to limited scheduling flexibility 
# MAGIC and resource constraints.  
# MAGIC In contrast, large carriers like **Southwest (WN)**, **American (AA)**, 
# MAGIC and **Delta (DL)** manage far more flights while maintaining moderate delay rates 
# MAGIC (around 16–21%), indicating stronger operational efficiency and network resilience.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Airport Delay Patterns
# MAGIC In this section, we analyze **departure delay patterns by airport** to identify 
# MAGIC which airports tend to experience the highest delay rates.  
# MAGIC Understanding airport-level performance helps reveal the impact of local congestion, 
# MAGIC weather conditions, and hub operations on flight punctuality.
# MAGIC

# COMMAND ----------

# Airport-level delay statistics
delay_by_airport = (
    df_clean.groupBy("ORIGIN")
      .agg(
          F.count("*").alias("num_flights"),
          F.round(F.mean("DEP_DEL15") * 100, 2).alias("delay_rate_pct")
      )
      .orderBy(F.desc("delay_rate_pct"))
)

display(delay_by_airport)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.1 Top 20 Busiest Airports

# COMMAND ----------

topN_busiest = (delay_by_airport.orderBy(F.desc("num_flights")).limit(20))
topN_sorted_by_busiest = topN_busiest.orderBy(F.desc("delay_rate_pct"))
display(topN_sorted_by_busiest)


# COMMAND ----------

# MAGIC %md
# MAGIC **2.3.1 Observation**
# MAGIC
# MAGIC Among the busiest airports, **ORD** (Chicago O’Hare), **ATL** (Atlanta), and **DFW** (Dallas/Fort Worth) 
# MAGIC handle the highest flight volumes but also experience significant delays.  
# MAGIC High delay rates are often observed at major hubs due to **air traffic congestion, 
# MAGIC weather sensitivity, and connecting flight dependencies**.  
# MAGIC Conversely, airports like **PHX** and **SEA** maintain relatively low delay percentages 
# MAGIC despite heavy traffic, suggesting more efficient scheduling and operational management.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.3.2 Top 20 Most Delayed Airports

# COMMAND ----------

topN_Most_Delay = (delay_by_airport.orderBy(F.desc("delay_rate_pct")).limit(20))
topN_sorted_by_delay = topN_Most_Delay.orderBy(F.desc("delay_rate_pct"))
display(topN_sorted_by_delay)

# COMMAND ----------

# MAGIC %md
# MAGIC **2.3.2 Observation**  
# MAGIC The top 20 most delayed airports reveal that **smaller regional airports** such as **ILG** (Wilmington), **UST** (St. Augustine), and **OTH** (North Bend) show **highest delay rates**—often exceeding 35–40%—despite having very few scheduled flights.  
# MAGIC This suggests that **limited infrastructure, staffing constraints, or higher sensitivity to weather conditions** can lead to large fluctuations in delay performance at low-volume airports.  
# MAGIC
# MAGIC In contrast, **major hubs like ORD** (Chicago O’Hare) and **DEN**(Denver) also appear among the top 20 but with **lower relative delay rates (~25–30%)**, which are likely driven by **air-traffic congestion rather than capacity shortages**.  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Weather Impact on Delays

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.1 Temperature vs Delay Patterns 

# COMMAND ----------

# Calculate delay rate by temperature bins (every 5°F)
temp_delay = (
    df_clean
      .withColumn("Temp_bin", F.floor(F.col("HourlyDryBulbTemperature") / 5) * 5)
      .groupBy("Temp_bin")
      .agg(
          F.round(F.mean("DEP_DEL15") * 100, 2).alias("delay_rate_pct"),
          F.count("*").alias("num_flights")
      )
      .orderBy("Temp_bin")
)
display(temp_delay)

# COMMAND ----------

# MAGIC %md
# MAGIC **2.4.1 Observation**
# MAGIC
# MAGIC The chart reveals a **U-shaped relationship** between temperature and flight delays.  
# MAGIC Flights departing in **moderate temperatures (40–70°F)** experience the **lowest delay rates** (around 12–17%),  
# MAGIC while **extreme cold (below 20°F)**  correspond to **higher delays**, reaching up to **35%**.  

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.2 Wind Speed vs Delay Patterns

# COMMAND ----------

# Calculate delay rate and flight count by wind speed bins (every 5 mph)
wind_delay = (
    df_clean
      .withColumn("Wind_bin", F.floor(F.col("HourlyWindSpeed") / 2.5) * 2.5)
      .groupBy("Wind_bin")
      .agg(
          F.round(F.mean("DEP_DEL15") * 100, 2).alias("delay_rate_pct"),
          F.count("*").alias("num_flights")
      )
      .orderBy("Wind_bin")
)
display(wind_delay)

# COMMAND ----------

# MAGIC %md
# MAGIC **2.4.2 Observation**
# MAGIC
# MAGIC Most flights occur under moderate wind speeds (around 5–15 mph), where delay rates remain relatively low.  
# MAGIC As wind speed increases beyond 20 mph, delay rates rise sharply while flight volume drops, suggesting that strong winds significantly disrupt flight operations.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.4.3 Precipitation vs Delay Patterns  
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC To ensure statistical reliability, bins with fewer than **100 flights** were excluded, as their delay rates fluctuate excessively due to small sample sizes.  
# MAGIC Additionally, the **no-precipitation group (0.00 inches)** was removed since it represents over 90% of all flights and dominates the visualization.  

# COMMAND ----------

BIN = 0.01

df_precip_num = (
    df_clean
      .withColumn("precip_raw", F.col("HourlyPrecipitation").cast("string"))
      .withColumn(
          "precip_num",
          F.when(F.col("precip_raw") == "T", 0.0)
           .otherwise(F.regexp_replace("precip_raw", "[^0-9\\.]", ""))
           .cast("double")
      )
)
precip_binned = (
    df_precip_num
      .filter(F.col("precip_num").isNotNull())             
      .withColumn("bin_idx", F.floor(F.col("precip_num")/F.lit(BIN)).cast("int"))
      .withColumn("Precip_bin",
                  (F.col("bin_idx")*F.lit(BIN)).cast(DecimalType(4,2))) 
      .groupBy("bin_idx","Precip_bin")
      .agg(
          F.count("*").alias("num_flights"),
          F.round(F.mean("DEP_DEL15")*100, 2).alias("delay_rate_pct")
      )
      .orderBy("bin_idx")
      .drop("bin_idx")
)

display(precip_binned)

# COMMAND ----------

# Filter out bins with very few samples and precipitation = 0
precip_filtered = precip_binned.filter(
    (F.col("Precip_bin") > 0) & (F.col("num_flights") >= 100)
)
display(precip_filtered)

# COMMAND ----------

# MAGIC %md
# MAGIC **2.4.3 Observation**
# MAGIC
# MAGIC After excluding bins with precipitation = 0 and fewer than 100 flights,
# MAGIC the distribution reveals a meaningful relationship between rainfall intensity and flight delays.  
# MAGIC
# MAGIC - **Light rain (<0.05 inches)** already shows a mild increase in delay rate (~29%).  
# MAGIC - **Moderate rain (0.10–0.15 inches)** corresponds to a noticeable rise, around **35–40% delay rate**.  
# MAGIC - For **heavier precipitation (>0.20 inches)**, the sample size drops sharply, but the delay rate remains consistently high (30–37%).  
# MAGIC
# MAGIC Overall, **delay probability tends to rise with increasing precipitation**, especially once rainfall exceeds **0.10 inches**, indicating that wet weather significantly disrupts flight operations.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Distance Impact on Delays
# MAGIC Understanding how flight distance and route characteristics influence delay rates helps identify whether delays are primarily due to operational congestion or weather dependencies.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2.5.1 Distance Delay Patterns

# COMMAND ----------

# Calculate delay rate by distance bin (every 250 miles)

distance_delay = (
    df_clean.filter(F.col("DISTANCE").isNotNull())
      .withColumn("distance_bin", (F.floor(F.col("DISTANCE") / 250) * 250))
      .groupBy("distance_bin")
      .agg(
          F.count("*").alias("num_flights"),
          F.round(F.mean("DEP_DEL15") * 100, 2).alias("delay_rate_pct")
      )
      .orderBy("distance_bin")
)

display(distance_delay)


# COMMAND ----------

# MAGIC %md
# MAGIC **2.5.1 Observation**
# MAGIC
# MAGIC Short-haul flights (0–500 miles) have the highest flight volume but maintain a relatively moderate delay rate around 18–20%.  
# MAGIC As the flight distance increases, the delay rate **rises noticeably**, peaking around **1500–2500 miles** (≈20–33%).
# MAGIC
# MAGIC For very long-haul flights (>3500 miles), the delay rate continues increaseing.