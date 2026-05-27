# Databricks notebook source
# MAGIC %md
# MAGIC # Flight Sentry
# MAGIC Predicting Flight Departure Delays: A Time-Series Classification Approach
# MAGIC FP Phase 2 EDA, baseline pipeline, Scalability, Efficiency, Distributed/parallel Training, and Scoring Pipeline

# COMMAND ----------

from pyspark.sql.functions import col
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pandas as pd



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # Know your mount
# MAGIC Here is the mounting for this class, your source for the original data! Remember, you only have Read access, not Write! Also, become familiar with `dbutils` the equivalent of `gcp` in DataProc

# COMMAND ----------

data_BASE_DIR = "dbfs:/mnt/mids-w261/"
display(dbutils.fs.ls(f"{data_BASE_DIR}"))

# COMMAND ----------

# MAGIC %md
# MAGIC # Data for the Project
# MAGIC
# MAGIC For the project you will have 4 sources of data:
# MAGIC
# MAGIC 1. Airlines Data: This is the raw data of flights information. You have 3 months, 6 months, 1 year, and full data from 2015 to 2019. Remember the maxima: "Test, Test, Test", so a lot of testing in smaller samples before scaling up! Location of the data? `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/`, `dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/`, etc. (Below the dbutils to get the folders)
# MAGIC 2. Weather Data: Raw data for weather information. Same as before, we are sharing 3 months, 6 months, 1 year
# MAGIC 3. Stations data: Extra information of the location of the different weather stations. Location `dbfs:/mnt/mids-w261/datasets_final_project_2022/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet/`
# MAGIC 4. OTPW Data: This is our joined data (We joined Airlines and Weather). This is the main dataset for your project, the previous 3 are given for reference. You can attempt your own join for Extra Credit. Location `dbfs:/mnt/mids-w261/OTPW_60M/OTPW_60M/` and more, several samples are given!

# COMMAND ----------

# OTPW
#df_custom = spark.read.format("csv").option("header","true").load(f"dbfs:/mnt/mids-w261/OTPW_3M_2015.csv")
#display(df_custom)

out_path = "dbfs:/student-groups/Group_4_4/JOINED_3M_2015.parquet"
df_custom = spark.read.parquet(out_path)
display(df_custom.limit(100))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # EDA

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where to Checkpoint your data?
# MAGIC There is a folder created in the Mount called `student-groups`. Please create a folder there with the name `Group_{section}_{number}` for example, Group Section 01 Number 01 will be `Group_01_01`. Any folder that doesn't follow this convection will be deleted without warning. Thanks! 

# COMMAND ----------

# Create folder
section = "04"
number = "04"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}"
dbutils.fs.mkdirs(folder_path)




# COMMAND ----------




#TODO - Can we derive a column indicating aircraft delay from previous flight? airport delay ? carrier_delay ? 

# group rows of df_custom by weather_dely, security_delay, late_aircraft_delay, carrier_delay, NAS_delay
df_delay_reason = df_custom.filter((df_custom['weather_delay'] > 0.0) | (df_custom['security_delay'] > 0.0) | (df_custom['late_aircraft_delay'] > 0.0) | (df_custom['carrier_delay'] > 0.0) | (df_custom['NAS_delay'] > 0.0))



# Overlayed boxplots for df_delay_reason
delay_cols = ['weather_delay', 'security_delay', 'late_aircraft_delay', 'carrier_delay', 'NAS_delay']
delay_data = df_delay_reason.select(*delay_cols).toPandas().apply(pd.to_numeric, errors='coerce')

plt.figure(figsize=(10, 6))
delay_data.boxplot(column=delay_cols)
plt.title('Boxplot of Delay Reasons')
plt.xlabel('Delay Reason')
plt.ylabel('Values')
plt.xticks(rotation=45)
plt.show()





# COMMAND ----------

# count of null values in otpw df
null_counts = [
    F.count(
        F.when(
            (
                (F.isnan(F.col(c)) if dict(df_custom.dtypes)[c] in ["float", "double"] else F.lit(False))
                | F.col(c).isNull()
            ),
            c
        )
    ).alias(c)
    for c in df_custom.columns
]

df_custom.select(null_counts).display()




# COMMAND ----------

# count of rows in custom join dataset
df_custom.count()

# COMMAND ----------

# check to see if there are duplicate rows in the otpw 
df_custom.groupBy('flight_id', 'ORIGIN', 'DEST','OP_CARRIER', 'OP_CARRIER_FL_NUM' ).count().filter(F.col('count') > 1).display()

# COMMAND ----------

# find duplicate rows in df_custom
df_custom.sort('FL_DATE', 'OP_CARRIER','TAIL_NUM','OP_CARRIER_FL_NUM', 'ORIGIN', 'DEST','DEP_TIME', 'ARR_TIME').display() 

# COMMAND ----------

df_custom.filter(df_custom['CANCELLED']==0).groupBy('FL_DATE', 'TAIL_NUM', 'CRS_DEP_TIME', 'CRS_ARR_TIME', 'OP_CARRIER', 'OP_CARRIER_FL_NUM').count().filter(F.col('count') > 1).display()   

# COMMAND ----------

#find number of rows and columns of df_custom
df_custom.count(), len(df_custom.columns)



# COMMAND ----------

#Find percentage of rows null in each column for df_custom, sort by null count desc
df_custom_null = df_custom.select([F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in df_custom.columns]).toPandas().transpose().reset_index()
df_custom_null.columns = ['column', 'null_count']
df_custom_null['null_pct'] = df_custom_null['null_count']*100/df_custom.count()
#df_custom_null.sort_values(by='null_pct', ascending=False).display()

# COMMAND ----------

# create histogram with null column data ,label the x-axis as percentage of data missing, y-axis as number of columns

df_custom_null['null_pct'].hist(bins=100)
plt.xlabel('Percentage of data missing')
plt.ylabel('Number of columns')
plt.show()  

# COMMAND ----------

#FILTER out rows where DEP_DELAY is null (cancelled/diverted)
df_custom = df_custom.cache()

#Display counts in OTPW where DEP_DELAY is null 
print("Rows where DEP_DELAY is null")
display(df_custom.filter(df_custom['DEP_DELAY'].isNull()).count())


# Display counts in OTPW grouped by  column DIVERTED
df_custom.groupby(['DIVERTED', 'CANCELLED']).count().display()

# Filtering out rows where flights are diverted or cancelled
df_custom_filtered = df_custom.filter(df_custom['CANCELLED']==0).filter(df_custom['DIVERTED']==0)

# Count of rows where DEP_DELAY is null
print("Rows where DEP_DELAY is null")
display(df_custom_filtered.filter(df_custom_filtered['DEP_DELAY'].isNull()).count())




# COMMAND ----------

# Drop columns with mostly null values in df_custom

# Find columns with >50 % null values in the df_custom_null
mostly_null_df_custom = df_custom_null[df_custom_null['null_pct']> 50].sort_values(by='null_pct', ascending=False)
print("Columns with >50 % null values in df_custom_null", mostly_null_df_custom['column'].count())

print("Shape before dropping motly null columns", (df_custom_filtered.count(), len(df_custom_filtered.columns)))
if mostly_null_df_custom is not None: df_custom_filtered = df_custom_filtered.drop(*mostly_null_df_custom['column'])
print("Shape after dropping mostly null columns=", (df_custom_filtered.count(), len(df_custom_filtered.columns)))

# Drop columns causing data leakage
leakage_risk_columns = ['ARR_DEL15', 'ARR_DELAY', 'weather_delay', 'security_delay', 'late_aircraft_delay', 'carrier_delay', 'NAS_delay', 'DEP_TIME', 'ARR_TIME', 'TAXI_OUT', 'TAXI_IN', 'WHEELS_OFF', 'WHEELS_ON', 'ACTUAL_ELAPSED_TIME', 'AIR_TIME']

if leakage_risk_columns is not None: df_custom_filtered = df_custom_filtered.drop(*leakage_risk_columns)
print("Shape after dropping leakage risk columns=", (df_custom_filtered.count(), len(df_custom_filtered.columns)))

# Drop columns cancelled, diverted, cancel_reason
cancel_columns = ['CANCELLED', 'CANCELLATION_CODE', 'DIVERTED', 'DIVERTED_REASON']
if cancel_columns is not None: df_custom_filtered = df_custom_filtered.drop(*cancel_columns)
print("Shape after dropping cancel columns=", (df_custom_filtered.count(), len(df_custom_filtered.columns)))


# Drop columns without predictive information
non_predictive_columns = ['OP_UNIQUE_CARRIER', 'ORIGIN_STATE_NM', 'DEST_STATE_NM', 'ORIGIN_CITY_NAME', 'ORIGIN_AIRPORT_SEQ_ID', 'OP_CARRIER_AIRLINE_ID', 'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'DEST_AIRPORT_SEQ_ID', 'DEST_AIRPORT_ID', 'ORIGIN_CITY_MARKET_ID', 'DEST_CITY_NAME',  'DEST_STATE_ABR', 'DEP_DELAY_NEW','CRS_ARR_TIME', 'ARR_DELAY_NEW',  'ARR_DELAY_GROUP', 'ARR_TIME_BLK', 'CRS_ELAPSED_TIME','origin_airport_name', 'origin_station_name', 'origin_station_id', 'origin_iata_code', 'origin_icao',  'origin_station_lat', 'origin_station_lon', 'origin_airport_lat', 'origin_airport_lon', 'origin_station_dis', 'dest_airport_name', 'dest_station_name', 'dest_station_id', 'dest_iata_code', 'dest_icao', 'dest_station_lat', 'dest_station_lon', 'dest_airport_lat', 'dest_airport_lon', 'dest_station_dis', 'sched_depart_date_time', 'STATION', 'LATITUDE', 'LONGITUDE', 'NAME', 'REPORT_TYPE', 'SOURCE', 'REM', '_row_desc', 'WindEquipmentChangeDate', 'four_hours_prior_depart_UTC', 'ORIGIN_STATE_FIPS', 'DEST_STATE_FIPS', 'ORIGIN_WAC', 'DEST_WAC', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR', 'DEST_CITY_MARKET_ID', 'origin_region', 'dest_region', 'FLIGHTS' ]

if non_predictive_columns is not None: df_custom_filtered = df_custom_filtered.drop(*non_predictive_columns)
print("Shape after dropping non_predictive_columns =", (df_custom_filtered.count(), len(df_custom_filtered.columns))) 

# Convert  numerical columns to float

weather_related_columns = [
    "HourlyAltimeterSetting","HourlySkyConditions", "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
    "HourlyPrecipitation", "HourlyRelativeHumidity", "HourlySeaLevelPressure", "HourlyStationPressure", "HourlyWetBulbTemperature","HourlyVisibility", "HourlyWindDirection", "HourlyWindSpeed", 'ELEVATION'
    ]

target_columns = [
    "DEP_DEL15", 'DEP_DELAY', 'DEP_DELAY_GROUP'
]

numerical_features = weather_related_columns + target_columns


# Convert numerical features to float
for col in numerical_features:
    if col in df_custom_filtered.columns:
        df_custom_filtered = df_custom_filtered.withColumn(col, F.col(col).cast("float"))



# COMMAND ----------

display(df_custom_filtered.limit(100))

# COMMAND ----------

#check for null values in all columns

def check_for_null_values(df, df_nulls):
    if df_nulls is None:
        df_nulls = []
        columns_to_check = df.columns
    else:
        columns_to_check = df_nulls

    # Step 1: find null counts for each column and total row count
    result = df.select(
        [F.count(F.when(F.col(c).isNull(), c)).alias(c) for c in columns_to_check] +
        [F.count("*").alias("_total_count")]
    ).first()

    total_count = result["_total_count"]

    # Step 2: Build results
    null_results = []
    for col in columns_to_check:
        null_count = result[col]
        if null_count > 0:  # Only include columns with nulls
            null_pct = round(null_count * 100 / total_count, 2)
            null_results.append({'column': col, 'null_count': null_count, 'null_pct': null_pct})

    # Step 3: Display results
    if null_results:
        df_nulls = pd.DataFrame(null_results).sort_values('null_pct', ascending=False)
        df_nulls = df_nulls[df_nulls["null_count"] > 0]
        display(df_nulls)
    else:
        df_nulls = []
        print("No nulls found")


    return df_nulls



# COMMAND ----------

# check for null values
df_nulls = check_for_null_values(df_custom_filtered, None)


# COMMAND ----------

# Handle high percentage of null values in column HourlySkyConditions by using most frequent value (mode for categorical data) by month near the origin
# Mode Imputation for HourlySkyConditions
mode_sky_conditions = df_custom_filtered.groupBy("ORIGIN", "MONTH").agg(F.mode("HourlySkyConditions").alias("mode_sky_conditions"))
df_custom_filtered = df_custom_filtered.join(mode_sky_conditions, on=["ORIGIN", "MONTH"], how="left")
df_custom_filtered = df_custom_filtered.withColumn("HourlySkyConditions", F.coalesce(F.col("HourlySkyConditions"), F.col("mode_sky_conditions")))
df_custom_filtered = df_custom_filtered.drop("mode_sky_conditions")

# COMMAND ----------

# Handle medium percentage null values for the columns HourlyPrecipitation and HourlySeaLevelPressure by using the median ( numeric data) based on airport and month
# Median is better than mean because it is less affected by outliers.

median_cols = ['HourlyPrecipitation', 'HourlySeaLevelPressure']
for col in median_cols:
    median_expr = F.expr(f"percentile_approx({col}, 0.5)")
    medians = df_custom_filtered.groupBy("ORIGIN", "MONTH").agg(median_expr.alias(f"{col}_median"))
    df_custom_filtered = df_custom_filtered.join(medians, on=["ORIGIN", "MONTH"], how="left")
    df_custom_filtered = df_custom_filtered.withColumn(col, F.coalesce(F.col(col), F.col(f"{col}_median")))
    df_custom_filtered = df_custom_filtered.drop(f"{col}_median")


# COMMAND ----------

# check for null values
if (len(df_nulls) > 0):
    df_nulls = check_for_null_values(df_custom_filtered, df_nulls['column'].to_list())





# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F

# Impute the null values for the weather data using rolling average from previous rows. 
# Cannot use value from rows in front.


# Rolling Mean Imputation for weather related columns
# Create a window based on departure time range 
rolling_window = (Window
               .partitionBy("ORIGIN")
               .orderBy(F.unix_timestamp("prediction_utc"))  # Combined datetime column
               .rangeBetween(-3600 * 24, -3600 * 1))  # -24 hours window in seconds

cols_with_nulls = set(weather_related_columns) - set(['ELEVATION'])

for col in cols_with_nulls:
    df_custom_filtered = df_custom_filtered.withColumn(f"{col}_rolling_mean", F.avg(F.col(col)).over(rolling_window))
    df_custom_filtered = df_custom_filtered.withColumn(col, F.coalesce(F.col(col), F.col(f"{col}_rolling_mean")))
    df_custom_filtered = df_custom_filtered.drop(f"{col}_rolling_mean")


# COMMAND ----------

# check null counts
if len(df_nulls) > 0:
    df_nulls = check_for_null_values(df_custom_filtered, df_nulls['column'].tolist())




# COMMAND ----------


# check for nulls and if any, impute with global median
if len(df_nulls) > 0:
    
    # Calculate all global medians in one query
    columns_with_nulls = df_nulls['column'].tolist()

    global_medians_result = df_custom_filtered.select([
        F.expr(f"percentile_approx({col}, 0.5)").alias(f"{col}_median") 
        for col in columns_with_nulls
    ]).first()

    # Apply all imputation
    for col in columns_with_nulls:
        global_median = global_medians_result[f"{col}_median"]
        df_custom_filtered = df_custom_filtered.withColumn(
            col, F.coalesce(F.col(col), F.lit(global_median))
        )



# COMMAND ----------

# check null values again
df_nulls = check_for_null_values(df_custom_filtered, None)
assert len(df_nulls) == 0

# COMMAND ----------

df_custom_filtered.write.mode('overwrite').parquet(f"{folder_path}/df_custom_3M_initial_features.parquet")

# COMMAND ----------


display(dbutils.fs.ls(f"{folder_path}"))

# COMMAND ----------

df_custom_filtered = spark.read.parquet(f"{folder_path}/df_custom_3M_initial_features.parquet/")

# COMMAND ----------

# Split the 3M dataset
# Test set:  Last two weeks of last month (3)
test_data_3m = df_custom_filtered.filter((F.col("MONTH") == 3) & (F.col('DAY_OF_MONTH') > 14))

# Skipping validation data
# Validation set: First two weeks of last month (3)
#validation_data_3m = df_custom_filtered.filter((F.col("MONTH") == 3) & (F.col('DAY_OF_MONTH') <= 14))

# Train set: First two months  (1 , 2) plus last two weeks of month (3)
train_data_3m = df_custom_filtered.filter( (F.col("MONTH") == 1)  | (F.col("MONTH") == 2) | ((F.col("MONTH") == 3) & (F.col('DAY_OF_MONTH') <= 14)   ) )

print(f"Train data count: {train_data_3m.count()}")
#print(f"Validation data count: {validation_data_3m.count()}")
print(f"Test data count: {test_data_3m.count()}")

# COMMAND ----------

#datacheckpoint
# Save data split as  parquet files
train_data_3m.write.mode('overwrite').parquet(f"{folder_path}/df_custom_3M_train_data.parquet")
#validation_data_3m.write.mode('overwrite').parquet(f"{folder_path}/df_custom_3M_validation_data.parquet") 
test_data_3m.write.mode('overwrite').parquet(f"{folder_path}/df_custom_3M_test_data.parquet") 


# COMMAND ----------

train_data_3m = spark.read.parquet(f"{folder_path}/df_custom_3M_train_data.parquet")
#validation_data_3m = spark.read.parquet(f"{folder_path}/df_custom_3M_validation_data.parquet") 
test_data_3m = spark.read.parquet(f"{folder_path}/df_custom_3M_test_data.parquet") 

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Get label counts for each dataset
train_counts = train_data_3m.groupBy("DEP_DEL15").count().toPandas().set_index("DEP_DEL15")["count"]
#val_counts = validation_data_3m.groupBy("DEP_DEL15").count().toPandas().set_index("DEP_DEL15")["count"]
test_counts = test_data_3m.groupBy("DEP_DEL15").count().toPandas().set_index("DEP_DEL15")["count"]

# Combine into DataFrame
data = pd.DataFrame({
    'Train': train_counts,
    #'Validation': val_counts,
    'Test': test_counts
})

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(data.index))  # Label positions (0, 1)
width = 0.25  # Width of bars

bars1 = ax.bar(x - width, data['Train'], width, label='Train', color='skyblue')
#bars2 = ax.bar(x, data['Validation'], width, label='Validation', color='lightgreen')
bars3 = ax.bar(x + width, data['Test'], width, label='Test', color='salmon')

# Customize
ax.set_xlabel('Delay Actual', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('Label Distribution Across Datasets', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(['No Delay', 'Delay > 15 mins'])
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bars in [bars1,  bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# COMMAND ----------

# Handle class imbalance using class weights
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql import functions as F

# Step 1: Calculate class weights
# Get class distribution
class_counts = train_data_3m.groupBy("DEP_DEL15").count().collect()
class_0_count = [row['count'] for row in class_counts if row['DEP_DEL15'] == 0][0]
class_1_count = [row['count'] for row in class_counts if row['DEP_DEL15'] == 1][0]

total = class_0_count + class_1_count

# Calculate weights (inverse of class frequency)
weight_0 = total / (2 * class_0_count)
weight_1 = total / (2 * class_1_count)

print(f"Class 0 weight: {weight_0:.3f}")
print(f"Class 1 weight: {weight_1:.3f}")

# Step 2: Add weight column to dataframe
train_data_3m = train_data_3m.withColumn(
    "class_weight",
    F.when(F.col("DEP_DEL15") == 0, weight_0)
     .when(F.col("DEP_DEL15") == 1, weight_1)
)




# COMMAND ----------

#One hot encode categorical features
from pyspark.ml.feature import StringIndexer, OneHotEncoder


# Define categorical features with few unique values (for one-hot encoding)
# "origin_type", "dest_type" are missing in custom join
categorical_features = [
    "QUARTER","DAY_OF_MONTH","DAY_OF_WEEK", "OP_CARRIER", "ORIGIN", "DEST",  "DISTANCE_GROUP", "MONTH"
]

# Define numerical features 
numerical_features = [
    "HourlyAltimeterSetting","HourlySkyConditions", "HourlyDewPointTemperature", "HourlyDryBulbTemperature",
    "HourlyPrecipitation", "HourlyRelativeHumidity", "HourlySeaLevelPressure", "HourlyStationPressure", "HourlyWetBulbTemperature","HourlyVisibility", "HourlyWindDirection", "HourlyWindSpeed"
    
]

# Convert numerical features to float
for col in numerical_features:
    if col in df_custom_filtered.columns:
        df_custom_filtered = df_custom_filtered.withColumn(col, F.col(col).cast("float"))



# Create stages for the pipeline
stages = []

# Process categorical features with one-hot encoding
for feature in categorical_features:
    # Create string indexer
    indexer = StringIndexer(inputCol=feature, outputCol=f"{feature}_indexed", handleInvalid="keep")
    # Create one-hot encoder
    encoder = OneHotEncoder(inputCol=f"{feature}_indexed", outputCol=f"{feature}_encoded")
    # Add stages
    stages += [indexer, encoder]



# Collect all transformed features
transformed_features = [f"{feature}_encoded" for feature in categorical_features] + \
                       numerical_features

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# Create vector assembler
assembler = VectorAssembler(inputCols=transformed_features, outputCol="features_unscaled", handleInvalid="keep")
stages.append(assembler)

# Create standard scaler
scaler = StandardScaler(inputCol="features_unscaled", outputCol="features_scaled", withStd=True, withMean=True)
stages.append(scaler)

# Create and fit the pipeline
pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train_data_3m)

# Transform the datasets
train_data_3m_transformed = pipeline_model.transform(train_data_3m)
train_data_3m_transformed = train_data_3m_transformed.cache()
#validation_data_3m_transformed = pipeline_model.transform(validation_data_3m)
#validation_data_3m_transformed = validation_data_3m_transformed.cache()
test_data_3m_transformed = pipeline_model.transform(test_data_3m)
test_data_3m_transformed = test_data_3m_transformed.cache()

# Prepare data for modeling
train_data_3m_ml = train_data_3m_transformed.select("DEP_DEL15", "features_scaled", "class_weight")
#validation_data_3m_ml = validation_data_3m_transformed.select("DEP_DEL15", "features_scaled")
test_data_3m_ml = test_data_3m_transformed.select("DEP_DEL15", "features_scaled")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 Stage POC 

# COMMAND ----------

# Two stage prediction

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.regression import LinearRegression
from pyspark.sql import functions as F

# ========== TRAINING ==========

# 1. Prepare data for modeling

train_data_3m_ml_2step = train_data_3m_transformed.select("DEP_DEL15", "DEP_DELAY", "features_scaled", "class_weight")
#validation_data_3m_ml_2step = validation_data_3m_transformed.select("DEP_DEL15", "DEP_DELAY", "features_scaled")
test_data_3m_ml_2step = test_data_3m_transformed.select("DEP_DEL15", "DEP_DELAY", "features_scaled")



# 3. Train Random Forest classifier
print("Training classifier...")
rf = RandomForestClassifier(featuresCol="features_scaled", 
                            labelCol="DEP_DEL15", 
                            weightCol="class_weight",
                            predictionCol="DEP_DEL15_pred",
                            probabilityCol="DEP_DEL15_probability",
                            numTrees=100,
                            maxDepth=10,
                            seed=42)
rf_model = rf.fit(train_data_3m_ml_2step)



# 4. Train regressor on delayed flights only
print("Training regressor...")
delayed_flights = train_data_3m_ml_2step.filter(F.col("DEP_DEL15") == 1)
print(f"Training regressor on {delayed_flights.count()} delayed flights")

lr = LinearRegression(
    featuresCol="features_scaled",
    labelCol="DEP_DELAY",
    predictionCol="DEP_DELAY_pred",
    maxIter=100,
    regParam=0.1,
    elasticNetParam=0.0
)

lr_model = lr.fit(delayed_flights)

print("Completed Training regressor...")

# COMMAND ----------

# ========== PREDICTION ==========

from pyspark.mllib.evaluation import BinaryClassificationMetrics

def two_stage_predict(test_df, rf_model, lr_model):
    """
    Complete two-stage prediction pipeline
    
    Returns DataFrame with:
    - is_delayed_pred: 0 or 1
    - delay_probability: probability of delay
    - predicted_delay_minutes: 0 if not delayed, or predicted minutes if delayed
    """
    
    # Stage 1: Classify
    classified = rf_model.transform(test_df)
    
    # Stage 2: Regress for predicted delays
    delayed_mask = classified.filter(F.col("DEP_DEL15_pred") == 1)
    not_delayed_mask = classified.filter(F.col("DEP_DEL15_pred") == 0)
    
    # Apply regression to delayed predictions
    if delayed_mask.count() > 0:
        delayed_with_amount = lr_model.transform(delayed_mask)
    else:
        delayed_with_amount = delayed_mask.withColumn("DEP_DELAY_pred", F.lit(0.0))
    
    # Set 0 delay for not-delayed predictions
    not_delayed_with_amount = not_delayed_mask.withColumn("DEP_DELAY_pred", F.lit(0.0))
    
    # Combine results
    all_predictions = delayed_with_amount.unionByName(
        not_delayed_with_amount, 
        allowMissingColumns=True
    )
    
    return all_predictions

# Make predictions
predictions = two_stage_predict(test_data_3m_ml_2step, rf_model, lr_model)

# ========== EVALUATION ==========

# Show sample predictions
predictions.select(
    "DEP_DEL15", "DEP_DEL15_pred",
    "DEP_DELAY", "DEP_DELAY_pred",
    "DEP_DEL15_probability"
).show(20, truncate=False)

# Evaluate classification stage
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.mllib.evaluation import BinaryClassificationMetrics

evaluator = BinaryClassificationEvaluator(
    labelCol="DEP_DEL15",
    rawPredictionCol="DEP_DEL15_pred",
    metricName="areaUnderROC"
)



if "DEP_DEL15" in predictions.columns:
    auc = evaluator.evaluate(predictions)
    print(f"Classification AUC: {auc:.4f}")

# Evaluate regression stage (only on actually delayed flights)
delayed_actual = predictions.filter(F.col("DEP_DEL15") == 1)

if delayed_actual.count() > 0:
    from pyspark.ml.evaluation import RegressionEvaluator
    
    reg_evaluator = RegressionEvaluator(
        labelCol="DEP_DELAY",
        predictionCol="DEP_DELAY_pred",
        metricName="rmse"
    )
    
    rmse = reg_evaluator.evaluate(delayed_actual)
    
    reg_evaluator.setMetricName("mae")
    mae = reg_evaluator.evaluate(delayed_actual)
    
    reg_evaluator.setMetricName("r2")
    r2 = reg_evaluator.evaluate(delayed_actual)
    
    print(f"\nRegression Metrics (on delayed flights):")
    print(f"  RMSE: {rmse:.2f} ")
    print(f"  MAE: {mae:.2f} ")
    print(f"  R²: {r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Evaluation Metrics

# COMMAND ----------

## Performance: accuracy
import seaborn as sns
from pyspark.mllib.evaluation import BinaryClassificationMetrics, MulticlassMetrics
from sklearn.metrics import confusion_matrix

# Show sample predictions
predictions.select(
    "DEP_DEL15", "DEP_DEL15_pred",
    "DEP_DELAY", "DEP_DELAY_pred",
    "DEP_DEL15_probability"
).show(20, truncate=False)

# Evaluate classification stage
# Instantiate metrics object
metrics_b = BinaryClassificationMetrics(predictions.select("DEP_DEL15_pred", "DEP_DEL15").rdd.map(lambda row: (row.DEP_DEL15_pred, row.DEP_DEL15)))
metrics_m = MulticlassMetrics(predictions.select("DEP_DEL15_pred", "DEP_DEL15").rdd.map(lambda row: (row.DEP_DEL15_pred, row.DEP_DEL15)))

# Statistics by class
labels = [0.0, 1.0]
for label in sorted(labels):
    print("Class %s auprc = %s" % (label, metrics_b.areaUnderPR))
    print("Class %s precision = %s" % (label, metrics_m.precision(label)))
    print("Class %s recall = %s" % (label, metrics_m.recall(label)))
    print("Class %s F1 Measure = %s" % (label, metrics_m.fMeasure(label, 1.0)))
    print("Class %s F2 Measure = %s" % (label, metrics_m.fMeasure(label, 2.0)))

# Weighted stats
print("Weighted recall = %s" % metrics_m.weightedRecall)
print("Weighted precision = %s" % metrics_m.weightedPrecision)
print("Weighted F(1) Score = %s" % metrics_m.weightedFMeasure())
print("Weighted F(0.5) Score = %s" % metrics_m.weightedFMeasure(beta=0.5))
print("Weighted F(2) Score = %s" % metrics_m.weightedFMeasure(beta=2.0))
print("Weighted false positive rate = %s" % metrics_m.weightedFalsePositiveRate)

## Confusion Matrix
y_pred = predictions.select('DEP_DEL15_pred').collect()
y_orig = predictions.select('DEP_DEL15').collect()

def confusion_matrix_sklearn(y_orig, y_pred):
    """
    To plot the confusion_matrix with percentages
    prediction:  predicted values
    original:    original values
    """
    cm = confusion_matrix(y_orig, y_pred)
    labels = np.asarray(
        [
            ["{0:0.0f}".format(item) + "\n{0:.2%}".format(item / cm.flatten().sum())]
            for item in cm.flatten()
        ]
    ).reshape(2, 2)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=labels, fmt="", cmap="Blues")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    
confusion_matrix_sklearn(y_orig, y_pred)

# COMMAND ----------

# MAGIC %md
# MAGIC ![MIDS261_Final_Project_Phase2_POC.jpg](./MIDS261_Final_Project_Phase2_POC.jpg "MIDS261_Final_Project_Phase2_POC.jpg")

# COMMAND ----------

# MAGIC %md
# MAGIC Phase1: 
# MAGIC RMSE: 97.49 minutes
# MAGIC MAE: 53.59 min
# MAGIC R2: -0.912

# COMMAND ----------

# MAGIC %md
# MAGIC