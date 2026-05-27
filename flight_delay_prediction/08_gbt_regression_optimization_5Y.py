# Databricks notebook source
# MAGIC %md
# MAGIC # Flight Sentry 
# MAGIC Predicting Flight Departure Delays: A Time-Series Classification Approach
# MAGIC FP Phase 3  regression pipeline, Scalability, Efficiency, Distributed/parallel Training, and Scoring Pipeline

# COMMAND ----------

from pyspark.sql.functions import col,when,  isnan

import pyspark.sql.functions as F
import matplotlib.pyplot as plt
import pandas as pd
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

# Team folder
section = "4"
number = "4"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}/"
dbutils.fs.mkdirs(folder_path)
display(dbutils.fs.ls(f"{folder_path}"))

regression_train_checkpoint = 'checkpoint_6_regression_train_2015_2018.parquet/'
regression_test_checkpoint = 'checkpoint_6_regression_test_2019.parquet/'

# COMMAND ----------

# MAGIC %md
# MAGIC #### Helper Functions

# COMMAND ----------

#helper function to check for null values in all columns

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

def read_file_and_count_nulls(file):
    df = spark.read.parquet(file)
    #number of rows and number of columns for df
    print(f"Number of rows: {df.count()}")
    print(f"Number of columns: {len(df.columns)}")
    df_nulls = check_for_null_values(df, None)
    return df

def write_file(df, file):
    df.write.mode("overwrite").parquet(file)
    print(f"File written to {file}")
    return None

def display_size(df):
    print(f"Size of the dataset: {df.count()} rows")
    print(f"Number of columns: {len(df.columns)}")

# Extract categorical and numerical columns based on column type

def create_column_list_by_type(df):

    categorical_columns = []
    numerical_columns = []
    timestamp_columns = []
    date_column = []
    for field in df.schema.fields:
        if "string" in field.dataType.simpleString():
            categorical_columns.append(field.name)
        elif "timestamp" in field.dataType.simpleString():
            timestamp_columns.append(field.name)
        elif "date" in field.dataType.simpleString():
            date_column.append(field.name)
        else:
            numerical_columns.append(field.name)

    print("Categorical Feature Count: ", len(categorical_columns))
    print("Numerical Feature Count: ", len(numerical_columns))
    print("Timestamp Feature Count: ", len(timestamp_columns))
    print("Date Feature Count: ", len(date_column))
    return categorical_columns, numerical_columns, timestamp_columns, date_column

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load cleaned and feature engineered Data for 5Y: 2015-2019

# COMMAND ----------

# DATA Details
filepath = "dbfs:/student-groups/Group_4_4/"

file1 = "checkpoint_5_final_clean_2015-2019.parquet"
#file2 = "checkpoint_5_final_clean_2015-2019_refined.parquet"

#file0 = "dbfs:/student-groups/Group_4_4/2015_final_feature_engineered_data_with_dep_delay"

# COMMAND ----------

# Load Cleaned Feature Engineered Data

df_5y = read_file_and_count_nulls(filepath + file1)


# COMMAND ----------

# MAGIC %md
# MAGIC # Handle null columns

# COMMAND ----------

# drop 4 columns with null values (may revisit later, time permitting)
print("Size before dropping")
display_size(df_5y)
df_5y = df_5y.drop("same_day_prior_delay_percentage", "route_delays_30d", 'dest_delay_rate_today', 'carrier_delays_at_origin_30d')

print("size after dropping")
display_size(df_5y)

# COMMAND ----------

display(df_5y.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### TRAIN/TEST SPLIT

# COMMAND ----------

# MAGIC %md
# MAGIC We will split the 5Y 2015-2019 data by using the first 4 years for Training and last year for evaluation. 

# COMMAND ----------

 # Split the 5y dataset
# Test set:  1y 2019
test_data_5y = df_5y.filter((F.col("YEAR") == 2019) )
print("Size of the test dataset:")
display_size(test_data_5y)

# Train set: First 4 years
train_data_5y = df_5y.filter((F.col("YEAR") < 2019) )
print("Size of the train dataset:")
display_size(train_data_5y) 




# COMMAND ----------

# MAGIC %md
# MAGIC ### Class Imbalance

# COMMAND ----------



# Get label counts for each dataset
train_counts = train_data_5y.groupBy("DEP_DEL15").count().toPandas().set_index("DEP_DEL15")["count"]
test_counts = test_data_5y.groupBy("DEP_DEL15").count().toPandas().set_index("DEP_DEL15")["count"]

# Combine into DataFrame
data = pd.DataFrame({
    'Train': train_counts,
    'Test': test_counts
})

# Create grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(data.index))  # Label positions (0, 1)
width = 0.25  # Width of bars

bars1 = ax.bar(x - width, data['Train'], width, label='Train', color='skyblue')
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

# MAGIC %md
# MAGIC Class Imbalance can be addressed using different techniques like Class Weights or sampling techniques like UpSampling or DownSampling. For our case, with high data, Up Sampling the minority class does not make sense as it will increase the size of the dataset significantly. This technique is more appropriate for small datasets. With large dataset like ours, which has about 5M rows, down sampling the majority class makes more sense as we can afford to lose some data. 
# MAGIC
# MAGIC By combining Down Sampling with Cross Validation, we can ensure that all available training data is used for cross validation, so there is minimal loss of information. We will be implementing down sampling within the Cross Validation folds. It will not be applied to the entire data. This will allow minimum wastage of data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Target Variable : DEP_DELAY Analysis

# COMMAND ----------


# show min,max,median,mean of DEP_DELAY column  
train_data_5y.select('DEP_DELAY').describe().display()  

# COMMAND ----------

# display outliers in dep_delay column
from pyspark.sql import functions as F

def get_outlier_stats(df, col='DEP_DELAY'):
    """Calculate outlier thresholds using IQR method."""
    
    
    # Calculate percentiles
    percentiles = df.approxQuantile(col, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 0.01)
    
    q1, q3 = percentiles[2], percentiles[4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    stats = {
        'p1': percentiles[0],
        'p5': percentiles[1],
        'q1': q1,
        'median': percentiles[3],
        'q3': q3,
        'p95': percentiles[5],
        'p99': percentiles[6],
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    # Count outliers
    total = df.count()
    outliers_low = df.filter(F.col(col) < lower_bound).count()
    outliers_high = df.filter(F.col(col) > upper_bound).count()
    
    stats['total'] = total
    stats['outliers_low'] = outliers_low
    stats['outliers_high'] = outliers_high
    stats['outliers_pct'] = (outliers_low + outliers_high) / total * 100
    
    return stats

# Get stats
stats = get_outlier_stats(train_data_5y, 'DEP_DELAY')
print("=== Outlier Statistics ===")
for k, v in stats.items():
    print(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}")
                                                                             

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ##### Key Statistics
# MAGIC
# MAGIC Statistic|Value|Interpretation|
# MAGIC |--|--|--|
# MAGIC p1|-234 min|1% of flights depart 234+ min early (data quality issue?)|
# MAGIC p5|-9 min|5% of flights depart 9+ min early|
# MAGIC |Q1|-5 min|25% of flights depart 5+ min early|
# MAGIC |Median|-2 min|Typical flight departs 2 min early|
# MAGIC |Q3|7 min|75% of flights have delays ≤ 7 min|
# MAGIC |p95|65 min|95% of flights have delays ≤ 65 min|
# MAGIC |p99|2710 min|1% have delays > 45 hours (likely data errors)|
# MAGIC
# MAGIC
# MAGIC ##### Observations:
# MAGIC We observe extreme range in the data, (from -234 to 2710 minutes).
# MAGIC
# MAGIC 1. MOST FLIGHTS ARE ON-TIME OR EARLY                                  
# MAGIC   • Median = -2 min (typical flight is 2 min early)                  
# MAGIC    • 75% of flights have delays ≤ 7 min                               
# MAGIC                                                                          
# MAGIC  2. RIGHT-SKEWED DISTRIBUTION                                        
# MAGIC     • Long tail of delays on the right                                 
# MAGIC     • 12.7% of flights are "outliers" (delayed > 25 min)              
# MAGIC                                                                         
# MAGIC   3. OUTLIERS ARE MOSTLY DELAYS (not early departures)                  
# MAGIC     • High outliers: 3,038,585 (99.7% of all outliers)                
# MAGIC     • Low outliers: 8,024 (0.3% of all outliers)                      
# MAGIC                                                                   
# MAGIC   4. EXTREME VALUES SUGGEST DATA QUALITY ISSUES                         
# MAGIC     • p1 = -234 min (flights 4 hours early?)                          
# MAGIC     • p99 = 2710 min (flights 45 hours late?) 
# MAGIC
# MAGIC   IQR = Q3 - Q1 = 7 - (-5) = 12 minutes
# MAGIC
# MAGIC Lower bound = Q1 - 1.5 × IQR = -5 - 18 = -23 min
# MAGIC Upper bound = Q3 + 1.5 × IQR = 7 + 18 = 25 min   
# MAGIC
# MAGIC
# MAGIC
# MAGIC Outlier Counts:
# MAGIC Category|Count|Percentage|Description|
# MAGIC |--|--|--|--|
# MAGIC Normal|20,823,275|87.24%|Delay between -23 and 25 min
# MAGIC Low outliers|8,0240.03%|Very early (< -23 min)|
# MAGIC High outliers|3,038,585|12.73%|Delayed > 25 min|
# MAGIC Total outliers|3,046,609|12.76%
# MAGIC
# MAGIC We will focus on the delays wich are positive only. 
# MAGIC Since the distribution is right skewed, we will take log of the dep_delay column to reduce the impact of the outliers. 
# MAGIC

# COMMAND ----------

#Replace negative values with zero and log transform
# Define column names
def create_delay_log_column(df, df_name, delay_column_name, log_delay_column_name):
    '''Create a new column with the log of the delay column, replacing negative values with zero. '''   
    delay_column_name = 'DEP_DELAY'
    log_delay_column_name = 'DEP_DELAY_LOG'

    # Create new column: replace negatives with 0, then take log(x + 1)
    df = df.withColumn(
        log_delay_column_name,
        F.log(
            F.when(F.col(delay_column_name) < 0, 0)
            .otherwise(F.col(delay_column_name)) + 1
        )
    )

    # Verify: check for any issues
    print(f"Original column stats ({delay_column_name}):")
    df.select(
        F.min(delay_column_name).alias('min'),
        F.max(delay_column_name).alias('max'),
        F.avg(delay_column_name).alias('avg')
    ).show()

    print(f"New log column stats ({log_delay_column_name}):")
    df.select(
        F.min(log_delay_column_name).alias('min'),
        F.max(log_delay_column_name).alias('max'),
        F.avg(log_delay_column_name).alias('avg')
    ).show()

    # Verify no negative values in log column (log(0+1) = 0 is the minimum)
    print(f"For Dataset {df_name}: Rows with negative log values: {df.filter(F.col(log_delay_column_name) < 0).count()}")
    return df

train_data_5y = create_delay_log_column(train_data_5y, 'train', 'DEP_DELAY', 'DEP_DELAY_LOG')
test_data_5y = create_delay_log_column(test_data_5y,  'test', 'DEP_DELAY', 'DEP_DELAY_LOG')


# COMMAND ----------

from pyspark.sql import functions as F

def get_outlier_stats(df, col, is_log_transformed=False):
    """Calculate outlier thresholds using IQR method."""
    
    # If log-transformed, back-transform for interpretable stats
    if is_log_transformed:
        df = df.withColumn('_temp_col', F.exp(F.col(col)) - 1)
        analysis_col = '_temp_col'
    else:
        analysis_col = col
    
    # Calculate percentiles
    percentiles = df.approxQuantile(analysis_col, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99], 0.01)
    
    q1, q3 = percentiles[2], percentiles[4]
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    stats = {
        'column': col,
        'is_log': is_log_transformed,
        'p1': percentiles[0],
        'p5': percentiles[1],
        'q1': q1,
        'median': percentiles[3],
        'q3': q3,
        'p95': percentiles[5],
        'p99': percentiles[6],
        'iqr': iqr,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound
    }
    
    # Count outliers
    total = df.count()
    outliers_low = df.filter(F.col(analysis_col) < lower_bound).count()
    outliers_high = df.filter(F.col(analysis_col) > upper_bound).count()
    
    stats['total'] = total
    stats['outliers_low'] = outliers_low
    stats['outliers_high'] = outliers_high
    stats['outliers_pct'] = (outliers_low + outliers_high) / total * 100
    
    return stats



def plot_outlier_analysis(df, col, is_log_transformed=False, sample_fraction=0.1):
    """Create comprehensive outlier analysis for any column."""
    
    # Get stats
    stats = get_outlier_stats(df, col, is_log_transformed)
    
    # Prepare data for plotting
    if is_log_transformed:
        pdf = df.sample(fraction=sample_fraction, seed=42).withColumn(
            'plot_values', F.exp(F.col(col)) - 1
        ).select('plot_values').toPandas()
        title_suffix = "(back-transformed to minutes)"
        col_display = f"{col} → minutes"
    else:
        pdf = df.sample(fraction=sample_fraction, seed=42).select(col).toPandas()
        pdf.columns = ['plot_values']
        title_suffix = "(original scale)"
        col_display = col
    
    fig = plt.figure(figsize=(14, 10))
    
    # 1. Pie chart
    ax1 = fig.add_subplot(2, 2, 1)
    normal_count = stats['total'] - stats['outliers_low'] - stats['outliers_high']
    sizes = [normal_count, stats['outliers_high'], stats['outliers_low']]
    labels = [f"Normal\n({normal_count/stats['total']*100:.1f}%)", 
              f"High Outliers\n({stats['outliers_high']/stats['total']*100:.1f}%)",
              f"Low Outliers\n({stats['outliers_low']/stats['total']*100:.1f}%)"]
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    # Remove zero-sized slices
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        sizes, labels, colors = zip(*non_zero)
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Outlier Distribution', fontweight='bold')
    
    # 2. Box plot
    ax2 = fig.add_subplot(2, 2, 2)
    # Cap extreme values for visualization
    plot_cap = stats['p99']
    pdf_capped = pdf[pdf['plot_values'] <= plot_cap]
    bp = ax2.boxplot(pdf_capped['plot_values'], vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('steelblue')
    ax2.set_ylabel('Value')
    ax2.set_title(f'Box Plot (capped at p99={plot_cap:.0f})', fontweight='bold')
    
    # 3. Histogram
    ax3 = fig.add_subplot(2, 2, 3)
    pdf_hist = pdf[pdf['plot_values'] <= stats['p95']]
    ax3.hist(pdf_hist['plot_values'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axvline(stats['median'], color='green', linestyle='-', linewidth=2, label=f"Median ({stats['median']:.1f})")
    ax3.axvline(stats['upper_bound'], color='red', linestyle='--', linewidth=2, label=f"Upper bound ({stats['upper_bound']:.1f})")
    ax3.set_xlabel('Value')
    ax3.set_ylabel('Count')
    ax3.set_title('Distribution (up to p95)', fontweight='bold')
    ax3.legend()
    
    # 4. Summary statistics
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis('off')
    summary = f"""
    COLUMN: {col_display}
    
    PERCENTILES:
    • p1:     {stats['p1']:>10.1f}
    • p5:     {stats['p5']:>10.1f}
    • Q1:     {stats['q1']:>10.1f}
    • Median: {stats['median']:>10.1f}
    • Q3:     {stats['q3']:>10.1f}
    • p95:    {stats['p95']:>10.1f}
    • p99:    {stats['p99']:>10.1f}
    
    OUTLIER BOUNDARIES (IQR={stats['iqr']:.1f}):
    • Lower: {stats['lower_bound']:>10.1f}
    • Upper: {stats['upper_bound']:>10.1f}
    
    COUNTS:
    • Total:        {stats['total']:>12,}
    • High outliers:{stats['outliers_high']:>12,}
    • Low outliers: {stats['outliers_low']:>12,}
    • Outlier %:    {stats['outliers_pct']:>11.2f}%
    """
    ax4.text(0.1, 0.5, summary, fontsize=10, family='monospace',
             verticalalignment='center', transform=ax4.transAxes,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Summary Statistics', fontweight='bold')
    
    plt.suptitle(f'Outlier Analysis: {col_display} {title_suffix}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    #plt.savefig(f'/outlier_{col}.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return stats


def compare_distributions(df, col_original='DEP_DELAY', col_log='DEP_DELAY_LOG', sample_fraction=0.1):
    """Compare original and log-transformed distributions side by side."""
    
    # Get stats for both
    stats_original = get_outlier_stats(df, col_original, is_log_transformed=False)
    stats_log = get_outlier_stats(df, col_log, is_log_transformed=True)
    
    # Sample data
    df_sample = df.sample(fraction=sample_fraction, seed=42)
    
    pdf_original = df_sample.select(col_original).toPandas()
    pdf_original.columns = ['value']
    
    pdf_log = df_sample.withColumn('value', F.exp(F.col(col_log)) - 1).select('value').toPandas()
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: Original DEP_DELAY
    # Histogram
    ax1 = axes[0, 0]
    pdf_orig_capped = pdf_original[pdf_original['value'] <= stats_original['p95']]
    ax1.hist(pdf_orig_capped['value'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(stats_original['median'], color='green', linestyle='-', linewidth=2)
    ax1.axvline(stats_original['upper_bound'], color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Delay (minutes)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'DEP_DELAY (Original)\nMedian={stats_original["median"]:.1f}', fontweight='bold')
    
    # Box plot
    ax2 = axes[0, 1]
    pdf_orig_box = pdf_original[pdf_original['value'] <= stats_original['p99']]
    bp1 = ax2.boxplot(pdf_orig_box['value'], vert=True, patch_artist=True)
    bp1['boxes'][0].set_facecolor('steelblue')
    ax2.set_ylabel('Delay (minutes)')
    ax2.set_title('DEP_DELAY Box Plot', fontweight='bold')
    
    # Pie chart
    ax3 = axes[0, 2]
    normal = stats_original['total'] - stats_original['outliers_high'] - stats_original['outliers_low']
    sizes = [normal, stats_original['outliers_high']]
    labels = [f"Normal\n({normal/stats_original['total']*100:.1f}%)", 
              f"Outliers\n({stats_original['outliers_high']/stats_original['total']*100:.1f}%)"]
    colors = ['#2ecc71', '#e74c3c']
    ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax3.set_title('DEP_DELAY Outliers', fontweight='bold')
    
    # Row 2: Log-transformed (back-transformed for comparison)
    # Histogram
    ax4 = axes[1, 0]
    pdf_log_capped = pdf_log[pdf_log['value'] <= stats_log['p95']]
    ax4.hist(pdf_log_capped['value'], bins=50, color='coral', alpha=0.7, edgecolor='black')
    ax4.axvline(stats_log['median'], color='green', linestyle='-', linewidth=2)
    ax4.axvline(stats_log['upper_bound'], color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Delay (minutes, back-transformed)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'DEP_DELAY_LOG (Back-transformed)\nMedian={stats_log["median"]:.1f}', fontweight='bold')
    
    # Box plot
    ax5 = axes[1, 1]
    pdf_log_box = pdf_log[pdf_log['value'] <= stats_log['p99']]
    bp2 = ax5.boxplot(pdf_log_box['value'], vert=True, patch_artist=True)
    bp2['boxes'][0].set_facecolor('coral')
    ax5.set_ylabel('Delay (minutes, back-transformed)')
    ax5.set_title('DEP_DELAY_LOG Box Plot', fontweight='bold')
    
    # Pie chart
    ax6 = axes[1, 2]
    normal = stats_log['total'] - stats_log['outliers_high'] - stats_log['outliers_low']
    sizes = [normal, stats_log['outliers_high']]
    labels = [f"Normal\n({normal/stats_log['total']*100:.1f}%)", 
              f"Outliers\n({stats_log['outliers_high']/stats_log['total']*100:.1f}%)"]
    colors = ['#2ecc71', '#e74c3c']
    ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax6.set_title('DEP_DELAY_LOG Outliers', fontweight='bold')
    
    plt.suptitle('Comparison: Original vs Log-Transformed DEP_DELAY', fontsize=14, fontweight='bold')
    plt.tight_layout()
    #plt.savefig('/delay_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.show()
    
    # Print comparison table
    print("\n" + "="*70)
    print("COMPARISON: DEP_DELAY vs DEP_DELAY_LOG (back-transformed)")
    print("="*70)
    print(f"{'Metric':<20} {'DEP_DELAY':>15} {'DEP_DELAY_LOG':>15} {'Difference':>15}")
    print("-"*70)
    print(f"{'Median':<20} {stats_original['median']:>15.1f} {stats_log['median']:>15.1f} {stats_log['median']-stats_original['median']:>15.1f}")
    print(f"{'Q1':<20} {stats_original['q1']:>15.1f} {stats_log['q1']:>15.1f} {stats_log['q1']-stats_original['q1']:>15.1f}")
    print(f"{'Q3':<20} {stats_original['q3']:>15.1f} {stats_log['q3']:>15.1f} {stats_log['q3']-stats_original['q3']:>15.1f}")
    print(f"{'p95':<20} {stats_original['p95']:>15.1f} {stats_log['p95']:>15.1f} {stats_log['p95']-stats_original['p95']:>15.1f}")
    print(f"{'Upper bound':<20} {stats_original['upper_bound']:>15.1f} {stats_log['upper_bound']:>15.1f} {stats_log['upper_bound']-stats_original['upper_bound']:>15.1f}")
    print(f"{'Outlier %':<20} {stats_original['outliers_pct']:>14.1f}% {stats_log['outliers_pct']:>14.1f}% {stats_log['outliers_pct']-stats_original['outliers_pct']:>14.1f}%")
    print("="*70)
    
    return stats_original, stats_log

# Analyze DEP_DELAY (original)
print("=== DEP_DELAY (Original) ===")
stats_original = plot_outlier_analysis(train_data_5y, 'DEP_DELAY', is_log_transformed=False)

# Analyze DEP_DELAY_LOG (log-transformed, back-transform for display)
print("\n=== DEP_DELAY_LOG (Back-transformed to minutes) ===")
stats_log = plot_outlier_analysis(train_data_5y, 'DEP_DELAY_LOG', is_log_transformed=True)

# Compare both side-by-side
print("\n=== Side-by-Side Comparison ===")
stats_orig, stats_log = compare_distributions(train_data_5y, 'DEP_DELAY', 'DEP_DELAY_LOG')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Column Cardinality

# COMMAND ----------

# find unique values for categorical columns in df_5y
def print_cardinality_count(df, cat_col):
    """ 
    Print cardinality count for each categorical column in df_5y
    """
    for column in cat_col:
        print(f"\n{'='*50}")
        print(f"Column: {column}")
        print(f"{'='*50}")
        
        # Get distinct count
        distinct_count = df.select(column).distinct().count()
        print(f"Distinct values: {distinct_count}")

    
print_cardinality_count(train_data_5y, categorical_columns)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC
# MAGIC #### Categorical Column Cardinality Summary
# MAGIC
# MAGIC
# MAGIC | Column | Distinct Values | Encoding Recommendation |
# MAGIC |--------|-----------------|------------------------|
# MAGIC | DEST | 363 | Target Encoding (high cardinality) |
# MAGIC | ORIGIN | 362 | Target Encoding (high cardinality) |
# MAGIC | day_hour_interaction | 168 | Target Encoding (medium-high cardinality) |
# MAGIC | ORIGIN_STATE_ABR | 53 | Target Encoding or One-Hot |
# MAGIC | DEST_STATE_ABR | 53 | Target Encoding or One-Hot |
# MAGIC | OP_UNIQUE_CARRIER | 19 | One-Hot or Target Encoding |
# MAGIC | sky_condition_parsed | 6 | One-Hot Encoding |
# MAGIC | airline_reputation_category | 5 | One-Hot Encoding |
# MAGIC | season | 4 | One-Hot Encoding |
# MAGIC | turnaround_category | 4 | One-Hot Encoding |
# MAGIC | origin_type | 3 | One-Hot Encoding |
# MAGIC | weather_condition_category | 3 | One-Hot Encoding |
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC The numerical columns with high cardinality column  'flight_id', 'TAIL_NUM', have a very high cardinality, so they will not be used for feature modeling. 
# MAGIC
# MAGIC Categorical Columns with  High Cardinality
# MAGIC
# MAGIC Column: ORIGIN, DEST, day_hour_interaction , ORIGIN_STATE_ABR, DEST_STATE_ABR, OP_UNIQUE_CARRIER
# MAGIC We will address the above columns next by  target encoding  using the mean delay values at the location from train data. 
# MAGIC Global smoothing will be applied to prevent smaller aiports from getting extreme values. There may be some leakage during cross fold validation, but no leakage during holdout testing. 
# MAGIC  
# MAGIC

# COMMAND ----------


target_encoded_columns = ["ORIGIN", "DEST","ORIGIN_STATE_ABR", "DEST_STATE_ABR",  "OP_UNIQUE_CARRIER", 'day_hour_interaction']
# calculate on train, apply to both
def add_target_encodings(df, train_df, categorical_cols, target_col="DEP_DELAY", min_samples=100):
    """Add target encodings for multiple categorical columns."""
    
    global_mean = train_df.agg(F.mean(target_col)).collect()[0][0]
    
    for col in categorical_cols:
        # Calculate stats from training data
        stats = train_df.groupBy(col).agg(
            F.mean(target_col).alias(f"{col}_mean"),
            F.count("*").alias(f"{col}_count")
        )
        
        # Apply smoothing
        stats = stats.withColumn(
            f"{col}_encoded",
            (F.col(f"{col}_count") * F.col(f"{col}_mean") + min_samples * global_mean) / 
            (F.col(f"{col}_count") + min_samples)
        )
        
        # Join to dataframe
        df = df.join(
            stats.select(col, f"{col}_encoded"),
            on=col,
            how="left"
        ).fillna({f"{col}_encoded": global_mean})
    
    return df


# Apply to dataset

train_encoded = add_target_encodings(train_data_5y, train_data_5y,target_encoded_columns)      # train stats for train
test_encoded = add_target_encodings(test_data_5y, train_data_5y,target_encoded_columns)        # train stats for test!

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Drop _indexed columns as they have been target encoded

# COMMAND ----------



def drop_indexed_columns(df):
    # Preview before dropping
    indexed_cols = [c for c in df.columns if "_indexed" in c]

    print("Columns to be dropped:")
    print("-" * 40)
    for col in sorted(indexed_cols):
        print(f"  {col}")
    print("-" * 40)
    print(f"Total: {len(indexed_cols)} columns")

    # Drop columns
    df = df.drop(*indexed_cols)

    # Verify they're gone
    remaining_indexed = [c for c in train_encoded.columns if "_indexed" in c]
    print(f"\nRemaining '_indexed' columns: {len(remaining_indexed)}")
    print(remaining_indexed)
    return df


train_encoded = drop_indexed_columns(train_encoded)
test_encoded = drop_indexed_columns(test_encoded)



# COMMAND ----------

# check final size before saving checkpoint
display_size(train_encoded)
display_size(test_encoded)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Data Checkpoint for Regression
# MAGIC

# COMMAND ----------

#Data Checkpoint 
checkpoint_write = False
if checkpoint_write:
    
    print("Data Write Checkpoint is enabled")
    # Save data split as  parquet files
    write_file(train_encoded,f"{folder_path}{regression_train_checkpoint}")
    write_file(test_encoded, f"{folder_path}{regression_test_checkpoint}") 
else:
  print("Data Write Checkpoint is disabled")

#Checkpoint read
checkpoint_read = True
if checkpoint_read:
    print("Data Read Checkpoint is enabled")
    train_5y = read_file_and_count_nulls(f"{folder_path}{regression_train_checkpoint}")
    test_5y = read_file_and_count_nulls(f"{folder_path}{regression_test_checkpoint}")
else:
    print("Data Read Checkpoint is disabled")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Feature Selection for ML

# COMMAND ----------

# Check column counts again
# Extract categorical and numerical columns based on column type


categorical_columns, numerical_columns, timestamp_columns, date_column = create_column_list_by_type(train_5y)

print("="*80)
print("Categorical Columns:\n", categorical_columns)
print("="*80)
print("Target Encoded Columns:\n", target_encoded_columns)
print("="*80)
print("Target Encoded Count:", len(target_encoded_columns))
print("="*80)
print("Numerical Columns:\n", numerical_columns)
print("="*80)
print("Categorical Count:", len(categorical_columns))
print("="*80)
print("Numerical Count:", len(numerical_columns))
print("="*80)
print("Timestamp Columns:\n", timestamp_columns)
print("="*80)
print("Timestamp Count:", len(timestamp_columns))
print("="*80)
print("Date Column:\n", date_column)
print("="*80)
print("Date Column Count:", len(date_column))
print("="*80)
print("Total Columns:", len(categorical_columns) + len(numerical_columns) + len(timestamp_columns) + len(date_column))
print("="*80)
print("Total Features:", len(categorical_columns) + len(numerical_columns) - len(target_encoded_columns))
print("="*80)




# COMMAND ----------

# MAGIC %md
# MAGIC Categorical Feature Count:  12
# MAGIC Numerical Feature Count:  129
# MAGIC Timestamp Feature Count:  2
# MAGIC Date Feature Count:  1
# MAGIC
# MAGIC
# MAGIC ================================================================================
# MAGIC
# MAGIC Categorical Columns:
# MAGIC  ['day_hour_interaction', 'OP_UNIQUE_CARRIER', 'DEST_STATE_ABR', 'ORIGIN_STATE_ABR', 'DEST', 'ORIGIN', 'origin_type', 'season', 'weather_condition_category', 'airline_reputation_category', 'turnaround_category', 'sky_condition_parsed']
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC
# MAGIC Target Encoded Columns:
# MAGIC  ['ORIGIN', 'DEST', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR', 'OP_UNIQUE_CARRIER', 'day_hour_interaction']
# MAGIC ================================================================================
# MAGIC Target Encoded Count: 6
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC
# MAGIC Numerical Columns:
# MAGIC  ['asof_minutes', 'YEAR', 'QUARTER', 'DAY_OF_MONTH', 'DAY_OF_WEEK', 'OP_CARRIER_FL_NUM', 'CRS_ARR_TIME', 'ORIGIN_AIRPORT_ID', 'DEST_AIRPORT_ID', 'DISTANCE_high_corr', 'DISTANCE_GROUP_high_corr', 'HourlyDryBulbTemperature', 'HourlyDewPointTemperature', 'HourlyWetBulbTemperature_high_corr', 'HourlyWindDirection', 'HourlyWindGustSpeed', 'HourlyVisibility', 'HourlyRelativeHumidity', 'HourlyStationPressure', 'HourlySeaLevelPressure_high_corr', 'HourlyAltimeterSetting', 'origin_airport_lat', 'origin_airport_lon', 'dest_airport_lat', 'dest_airport_lon', 'origin_station_dis', 'dest_station_dis', 'DEP_DEL15', 'DEP_DELAY', 'departure_dayofweek', 'is_weekend', 'is_peak_month', 'time_of_day_early_morning', 'time_of_day_morning', 'time_of_day_afternoon', 'time_of_day_evening', 'time_of_day_night', 'rolling_origin_num_flights_24h_high_corr', 'rolling_origin_num_delays_24h', 'rolling_origin_delay_ratio_24h_high_corr', 'dep_delay15_24h_rolling_avg_by_origin_high_corr', 'dep_delay15_24h_rolling_avg_by_origin_carrier_high_corr', 'dep_delay15_24h_rolling_avg_by_origin_dayofweek', 'dep_delay15_24h_rolling_avg_by_origin_log', 'dep_delay15_24h_rolling_avg_by_origin_carrier_log', 'dep_delay15_24h_rolling_avg_by_origin_dayofweek_log', 'is_superbowl_week', 'is_major_event', 'weather_severity_index_high_corr', 'distance_medium', 'distance_long', 'distance_very_long', 'is_airport_maintenance', 'is_natural_disaster', 'airline_reputation_score', 'airport_traffic_density', 'carrier_flight_count', 'weather_obs_lag_hours', 'log_distance', 'is_rainy', 'prev_flight_dep_del15', 'prev_flight_crs_elapsed_time', 'hours_since_prev_flight', 'is_first_flight_of_aircraft', 'is_holiday_month', 'num_airport_wide_delays', 'oncoming_flights', 'prior_day_delay_rate', 'prior_flights_today', 'time_based_congestion_ratio', 'temp_humidity_interaction_high_corr', 'rapid_weather_change', 'temp_anomaly', 'precip_anomaly_high_corr', 'dep_time_sin', 'dep_time_cos', 'arr_time_sin', 'arr_time_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'wind_direction_sin', 'wind_direction_cos', 'extreme_precipitation', 'extreme_wind', 'extreme_temperature', 'origin_degree_centrality', 'dest_degree_centrality_high_corr', 'origin_pagerank_high_corr', 'dest_pagerank_high_corr', 'origin_betweenness_high_corr', 'dest_betweenness', 'delay_propagation_score', 'network_delay_cascade', 'days_since_epoch', 'origin_1yr_delay_rate', 'dest_1yr_delay_rate', 'rolling_30day_volume', 'days_since_last_delay_route', 'days_since_carrier_last_delay_at_origin', 'route_delay_rate_30d', 'peak_hour_x_traffic', 'weekend_x_route_volume', 'weather_x_airport_delays', 'temp_x_holiday', 'route_delay_rate_x_peak_hour', 'rolling_delay_rate_squared_high_corr', 'log_distance_squared_high_corr', 'traffic_density_squared_high_corr', 'carrier_encoded_x_hour', 'origin_encoded_x_weather', 'origin_encoded_x_visibility', 'origin_encoded_x_precipitation', 'origin_encoded_x_wind', 'origin_x_dest_encoded', 'carrier_x_origin_encoded', 'carrier_x_dest_encoded', 'rf_prob_delay', 'rf_prob_delay_binned', 'dep_delay15_24h_rolling_avg_by_origin_carrier_weighted', 'dep_delay15_24h_rolling_avg_by_origin_weighted', 'DEP_DELAY_LOG', 'ORIGIN_encoded', 'DEST_encoded', 'ORIGIN_STATE_ABR_encoded', 'DEST_STATE_ABR_encoded', 'OP_UNIQUE_CARRIER_encoded', 'day_hour_interaction_encoded']
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC
# MAGIC Categorical Count: 12
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC
# MAGIC Numerical Count: 129
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC
# MAGIC Timestamp Columns:
# MAGIC  ['prediction_utc', 'origin_obs_utc']
# MAGIC ================================================================================
# MAGIC Timestamp Count: 2
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC Date Column:
# MAGIC  ['FL_DATE']
# MAGIC ================================================================================
# MAGIC Date Column Count: 1
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC Total Columns: 144
# MAGIC ================================================================================
# MAGIC
# MAGIC
# MAGIC Total Features: 135
# MAGIC ================================================================================

# COMMAND ----------


# FEATURE SELECTION

# Categorical Count: 6
# Numerical Count: 129



def create_feature_list(categorical_columns, numerical_columns, timestamp_columns, date_column,target_encoded_columns ):
    
    categorical_leakage_Columns = ['CANCELLATION_CODE']
    #grid_zero_imp_c = zero_importance_grid_search_categorical 
    categorical_features = list(set(categorical_columns) - set(categorical_leakage_Columns) - set(target_encoded_columns)) 

    numerical_leakage_Columns = ['DEP_DELAY','DEP_DELAY_LOG', 'DEP_DEL15', 'CANCELLED', 'DIVERTED',]

    #grid_zero_imp_n = zero_importance_grid_search_numerical
    numerical_features = list(set(numerical_columns) - set(numerical_leakage_Columns) - set(date_column) - set(timestamp_columns))
    return categorical_features, numerical_features

    

categorical_features, numerical_features = create_feature_list(categorical_columns,
                                                                numerical_columns, 
                                                                timestamp_columns,
                                                                date_column, 
                                                                target_encoded_columns )

print("="*80)
print("categorical_features=", len(categorical_features))
print("="*80)
print("numerical_features=", len(numerical_features))
print("="*80)
print("Categorical Features:\n", categorical_features)
print("="*80)
print("Numerical Features:\n", numerical_features)
print("="*80)

       

# COMMAND ----------

# MAGIC %md
# MAGIC ================================================================================
# MAGIC categorical_features= 6
# MAGIC ================================================================================
# MAGIC numerical_features= 126
# MAGIC ================================================================================
# MAGIC Categorical Features:
# MAGIC  ['weather_condition_category', 'season', 'sky_condition_parsed', 'turnaround_category', 'airline_reputation_category', 'origin_type']
# MAGIC ================================================================================
# MAGIC Numerical Features:
# MAGIC  ['origin_1yr_delay_rate', 'is_peak_month', 'HourlyDryBulbTemperature', 'asof_minutes', 'OP_CARRIER_FL_NUM', 'hours_since_prev_flight', 'days_since_epoch', 'carrier_x_origin_encoded', 'dest_airport_lat', 'dep_time_sin', 'oncoming_flights', 'rolling_origin_delay_ratio_24h_high_corr', 'dest_betweenness', 'dep_delay15_24h_rolling_avg_by_origin_high_corr', 'extreme_wind', 'log_distance_squared_high_corr', 'origin_encoded_x_precipitation', 'is_superbowl_week', 'DEST_AIRPORT_ID', 'HourlySeaLevelPressure_high_corr', 'month_sin', 'HourlyWindDirection', 'ORIGIN_encoded', 'day_hour_interaction_encoded', 'dep_delay15_24h_rolling_avg_by_origin_dayofweek_log', 'dep_delay15_24h_rolling_avg_by_origin_carrier_weighted', 'DAY_OF_MONTH', 'route_delay_rate_30d', 'QUARTER', 'wind_direction_sin', 'ORIGIN_AIRPORT_ID', 'origin_degree_centrality', 'peak_hour_x_traffic', 'wind_direction_cos', 'carrier_x_dest_encoded', 'time_based_congestion_ratio', 'is_rainy', 'prev_flight_dep_del15', 'arr_time_cos', 'weather_severity_index_high_corr', 'prior_flights_today', 'carrier_flight_count', 'rolling_delay_rate_squared_high_corr', 'dest_1yr_delay_rate', 'extreme_precipitation', 'rf_prob_delay', 'carrier_encoded_x_hour', 'time_of_day_morning', 'origin_x_dest_encoded', 'day_of_week_sin', 'is_weekend', 'prev_flight_crs_elapsed_time', 'dest_pagerank_high_corr', 'dep_delay15_24h_rolling_avg_by_origin_weighted', 'DISTANCE_GROUP_high_corr', 'is_holiday_month', 'dest_degree_centrality_high_corr', 'dep_delay15_24h_rolling_avg_by_origin_carrier_log', 'route_delay_rate_x_peak_hour', 'CRS_ARR_TIME', 'days_since_carrier_last_delay_at_origin', 'origin_airport_lon', 'is_airport_maintenance', 'is_first_flight_of_aircraft', 'rf_prob_delay_binned', 'airline_reputation_score', 'rolling_30day_volume', 'days_since_last_delay_route', 'precip_anomaly_high_corr', 'ORIGIN_STATE_ABR_encoded', 'weather_x_airport_delays', 'DAY_OF_WEEK', 'weather_obs_lag_hours', 'rapid_weather_change', 'DISTANCE_high_corr', 'temp_humidity_interaction_high_corr', 'is_natural_disaster', 'distance_very_long', 'prior_day_delay_rate', 'extreme_temperature', 'origin_airport_lat', 'log_distance', 'dep_time_cos', 'YEAR', 'delay_propagation_score', 'origin_station_dis', 'network_delay_cascade', 'dep_delay15_24h_rolling_avg_by_origin_carrier_high_corr', 'time_of_day_night', 'rolling_origin_num_delays_24h', 'HourlyWindGustSpeed', 'dep_delay15_24h_rolling_avg_by_origin_log', 'HourlyDewPointTemperature', 'traffic_density_squared_high_corr', 'rolling_origin_num_flights_24h_high_corr', 'distance_medium', 'origin_encoded_x_weather', 'time_of_day_early_morning', 'HourlyWetBulbTemperature_high_corr', 'dest_airport_lon', 'DEST_encoded', 'HourlyVisibility', 'dest_station_dis', 'DEST_STATE_ABR_encoded', 'origin_encoded_x_wind', 'departure_dayofweek', 'HourlyStationPressure', 'temp_anomaly', 'day_of_week_cos', 'weekend_x_route_volume', 'is_major_event', 'airport_traffic_density', 'HourlyAltimeterSetting', 'num_airport_wide_delays', 'temp_x_holiday', 'origin_encoded_x_visibility', 'arr_time_sin', 'origin_pagerank_high_corr', 'OP_UNIQUE_CARRIER_encoded', 'HourlyRelativeHumidity', 'origin_betweenness_high_corr', 'time_of_day_afternoon', 'month_cos', 'dep_delay15_24h_rolling_avg_by_origin_dayofweek', 'distance_long', 'time_of_day_evening']
# MAGIC ================================================================================

# COMMAND ----------



# create encoded categorical feature names to help understand feature importance
def create_encoded_feature_names(categorical_features):
    # find cardinality count for the categorical features
    print("Categorical Features:\n", categorical_features)
    print("="*80)
    categorical_feature_encoded = []
    for col in categorical_features:
        for i in range(0,train_data_5y.select(col).distinct().count()):
            encoded_col_name = col + "_"+str(i)
            categorical_feature_encoded.append(encoded_col_name)
    return categorical_feature_encoded

categorical_feature_encoded_names = create_encoded_feature_names(categorical_features)
print("Number of Categorical Features:", len(categorical_feature_encoded_names))
print(categorical_feature_encoded_names)
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categorical Features Summary
# MAGIC
# MAGIC ### Original Categorical Features (6)
# MAGIC
# MAGIC | # | Feature |
# MAGIC |---|---------|
# MAGIC | 1 | weather_condition_category |
# MAGIC | 2 | season |
# MAGIC | 3 | sky_condition_parsed |
# MAGIC | 4 | turnaround_category |
# MAGIC | 5 | airline_reputation_category |
# MAGIC | 6 | origin_type |
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### One-Hot Encoded Features (25)
# MAGIC
# MAGIC | # | Feature |
# MAGIC |---|---------|
# MAGIC | 1 | weather_condition_category_0 |
# MAGIC | 2 | weather_condition_category_1 |
# MAGIC | 3 | weather_condition_category_2 |
# MAGIC | 4 | season_0 |
# MAGIC | 5 | season_1 |
# MAGIC | 6 | season_2 |
# MAGIC | 7 | season_3 |
# MAGIC | 8 | sky_condition_parsed_0 |
# MAGIC | 9 | sky_condition_parsed_1 |
# MAGIC | 10 | sky_condition_parsed_2 |
# MAGIC | 11 | sky_condition_parsed_3 |
# MAGIC | 12 | sky_condition_parsed_4 |
# MAGIC | 13 | sky_condition_parsed_5 |
# MAGIC | 14 | turnaround_category_0 |
# MAGIC | 15 | turnaround_category_1 |
# MAGIC | 16 | turnaround_category_2 |
# MAGIC | 17 | turnaround_category_3 |
# MAGIC | 18 | airline_reputation_category_0 |
# MAGIC | 19 | airline_reputation_category_1 |
# MAGIC | 20 | airline_reputation_category_2 |
# MAGIC | 21 | airline_reputation_category_3 |
# MAGIC | 22 | airline_reputation_category_4 |
# MAGIC | 23 | origin_type_0 |
# MAGIC | 24 | origin_type_1 |
# MAGIC | 25 | origin_type_2 |
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Cross Validation for time series data

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Cross Validation
# MAGIC Size of the train dataset: 23869884 rows
# MAGIC Size of the test dataset: 7259007 rows
# MAGIC
# MAGIC For Cross Validation, we dont split the datasetup front, instead, we split the dataset during the cross validation on a per fold basis. For each iteration, (1*i) year of data will be used for training and 1 year will be used for testing. So, for 5 year data we will use first 4 year as train data and last 1 year data for holdout evaluation. 
# MAGIC
# MAGIC
# MAGIC raw_data → [Feature Selection] → [Encoding] → [Scaling] → preprocessed_data → [Time Series CV]
# MAGIC
# MAGIC The "leakage" from scaling is negligible with large time-series datasets. Target encoding and time-based CV  matter much more!
# MAGIC
# MAGIC Holdout configuration
# MAGIC
# MAGIC N_HOLDOUT_YEAR = 1  #  Last 1 Year held out
# MAGIC
# MAGIC
# MAGIC CV configuration - Expanding Window
# MAGIC
# MAGIC N_TRAIN_Year = (n)    # Use n years for training in nth fold
# MAGIC
# MAGIC N_TEST_YEAR = 1     # Test on (n +1)th year
# MAGIC
# MAGIC
# MAGIC ### Class Imbalance
# MAGIC
# MAGIC Why downsampling is best:
# MAGIC
# MAGIC Even after downsampling to 50:50, we'd still have 2 million rows (1M + 1M), which is more than enough for most models. Reducing from 5M to 2M rows results in 60% faster training. Memory efficiency - Uses less RAM/GPU memory. No overfitting risk - Unlike upsampling which can cause overfitting. Minimal information loss - With 4M majority samples,  can afford to discard 3M
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### REGRESSION WITH CROSS VALIDATION

# COMMAND ----------

# Spark configurations to handle shuffle issues better
spark.conf.set("spark.sql.shuffle.partitions", "200")
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")

# Set checkpoint directory (one-time setup)
spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoint")

# Increase memory if needed
#spark.conf.set("spark.executor.memory", "8g")
#spark.conf.set("spark.driver.memory", "4g")

# COMMAND ----------

# MAGIC %md
# MAGIC #### GridSearch with Time Series Cross Validation

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import time

from pyspark.sql import functions as F
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor
)


import pandas as pd
import time

# ============================================================================
# CROSS VALIDATION HELPER FUNCTIONS
#  undersample_majority_class
# ============================================================================


def undersample_majority_class(train_df, balance_col, sampling_strategy=1.0, seed=42):
    """Undersample majority class to balance dataset."""

    delayed_df = train_df.filter(F.col(balance_col) == 1)
    ontime_df = train_df.filter(F.col(balance_col) == 0)
    
    delayed_count = delayed_df.count()
    ontime_count = ontime_df.count()
    
    print(f"    Original: On-time={ontime_count:,}, Delayed={delayed_count:,}")
    
    target_ontime_count = int(delayed_count / sampling_strategy)
    sampling_fraction = min(1.0, target_ontime_count / ontime_count)
    
    ontime_sampled = ontime_df.sample(False, sampling_fraction, seed=seed)
    balanced_df = delayed_df.union(ontime_sampled)
    
    actual_ontime = ontime_sampled.count()
    print(f"    Balanced: On-time={actual_ontime:,}, Delayed={delayed_count:,}")
    
    return balanced_df

# ============================================================================
# CROSS VALIDATION HELPER FUNCTIONS
# extract_feature_importance
# ============================================================================

def extract_feature_importance_with_names( model_name, importances, categorical_feature_encoded_names, numerical_features):
    """
    Extracts feature importance from a trained model.
    Parameters:
        importances: List of feature importances.
        feature_names: List of feature names.
    Returns:
        List of tuples containing feature names and their corresponding importance
        sorted in descending order of importance.
    """
    

    # Build expanded feature names
    expanded_features = numerical_features + categorical_feature_encoded_names
    #print(f"\nTotal expanded features: {len(expanded_features)}")
    # print expanded_features values
    #print(f"Expanded features: {expanded_features}")
    print(f"Feature length: {len(expanded_features)}")
    print(f"Importances length: {len(importances)}")
    feature_importance_list = []
    # Check lengths
    if len(expanded_features) == len(importances):
        # Create one row per feature
        for feature, importance in zip(expanded_features, importances):
            feature_importance_list.append({
                'model': model_name,
                'feature': feature,
                'importance': importance
            })
        
        feature_importance_df = pd.DataFrame(feature_importance_list)\
                                .sort_values(by='importance', ascending=False)
        
        print("\nTop 40 Most Important Features:")
        for idx, row in feature_importance_df.head(20).iterrows():
            print(f"{row['feature']:50s} {row['importance']:.6f}")
        return feature_importance_list
    else:
        print(f"Still mismatched: {len(expanded_features)} vs {len(importances)}")
        #print first Non zero importances
        for idx, importance in enumerate(importances):
            if importance > 0:
                print(f"Feature {idx}: {importance}")
        return None



# COMMAND ----------

# MAGIC %md
# MAGIC  We will use rolling window 

# COMMAND ----------


# Grid Search helper functions

def parameter_sets(param_grid):
    # return parameter names and parameter sets in param_grid.
    parameter_names = [param.name for param in param_grid[0]]
    parameter_values = [p.values() for p in param_grid]
    return parameter_names, parameter_values
  
def create_simple_param_grid_dt(model_type):
  
  if model_type == "decision_tree":
    grid = {
        'maxDepth': [5, 10],
        'maxBins': [ 50, 100],
        'minInstancesPerNode': [ 2,5],
        'minInfoGain': [0.0, 0.1]
    }
  elif model_type == "random_forest":
    grid = {
        'maxDepth': [ 5,10],
        'maxBins': [ 20, 100],
        'minInstancesPerNode': [2, 5],
        'minInfoGain': [0.0, 0.1],
        'numTrees': [ 10, 20],
        'subsamplingRate': [0.8, 1.0]  
    }
  elif model_type == "gradient_boosted_trees":
    grid = {
        'maxDepth': [5, 10],
        'maxBins': [ 20, 50],
        'minInstancesPerNode': [5, 10],
        'minInfoGain': [0.0, 0.1],
        'maxIter': [10, 20, 50],        # number of trees/iterations
        'stepSize': [0.05, 0.1]    # learning rate
    }
  elif model_type == "xgboost":
    grid = {
        'max_depth': [3,5,7],
        'learning_rate': [0.05, 0.01,0.1],
        'num_round': [50, 100,200],
        'colsample_bytree': [0.8, 1.0],
        'reg_lambda': [0.0, 1,10]
    }
  
  param_grid = [dict(zip(grid.keys(), v)) for v in product(*grid.values())]
  return param_grid


def preprocess_pipeline_func():
    """Create a pipeline for feature transformation."""
    
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
    transformed_features =  numerical_features  + [f"{feature}_encoded" for feature in categorical_features] 

    # Create vector assembler
    assembler = VectorAssembler(inputCols=transformed_features, outputCol="features", handleInvalid="keep")
    stages.append(assembler)

    #skipping scaler for Tree family of models
    # Create standard scaler
    #scaler = StandardScaler(inputCol="features_unscaled", outputCol="features_scaled", withStd=True, withMean=False)
    #stages.append(scaler)      

    # Create the preprocessing pipeline
    preprocess_pipeline = Pipeline(stages=stages)

    return preprocess_pipeline
  
###################################################
#Tuning strategy_ xgboost():
def create_param_grid_by_stage_xgb(stage,best_depth=7,best_lr=0.1,best_num_round=100):
    #After stage 1 tuning, updated defaults to stage 1 results


    # Stage 1: Tune structure (most important)
    stage1_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.05, 0.1],
        'num_round': [50, 100]
    }
    # 3 × 2 × 2 = 12 combinations

    # Stage 2: Tune regularization (after finding best depth/lr)
    stage2_grid = {
        'max_depth': [best_depth],
        'learning_rate': [best_lr],
        'num_round': [best_num_round],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_lambda': [0, 1, 10],
        'gamma': [0, 0.1, 0.5]
    }
    # 1 × 1 × 1 × 3 × 3 × 3 = 27 combinations

    # Stage 3: Fine-tune 
    stage3_grid = {
        # Narrow ranges around best values
        'max_depth': [best_depth - 1, best_depth, best_depth + 1],
        'learning_rate': [best_lr * 0.5, best_lr, best_lr * 2],
        'num_round': [best_num_round, best_num_round * 1.5, best_num_round * 2],
    }

    default_grid = {
      'max_depth':5,
      'learning_rate':0.1,
      'n_estimators':100,
      'colsample_bytree':0.8,
      'reg_lambda':1,
      'gamma':0,
      'early_stopping_rounds':10,
    }

    if stage == 1:
      grid =  stage1_grid
    elif stage == 2:
      grid = stage2_grid
    elif stage == 3:
      grid =  stage3_grid
    else: 
      grid = default_grid

    param_grid = [dict(zip(grid.keys(), v)) for v in product(*grid.values())]
    return param_grid
    
  
def get_model(model_name, preprocess_pipeline_func, params):
    """
    Create a pipeline with specified model and parameters.
    
    Args:
        model_type: str - model type ('decision_tree', 'gbt', 'random_forest')
        pipeline_func: function - returns preprocessing stages
        params: dict - hyperparameters for the model
    
    Returns:
        Pipeline object
    """


    if model_name == 'decision_tree':
      model = DecisionTreeRegressor(
          featuresCol=FEATURES_COL,
          labelCol=LABEL_COL,
      )
    elif model_name == 'random_forest':
      model = RandomForestRegressor(
          featuresCol=FEATURES_COL,
          labelCol=LABEL_COL,
      )
    elif model_name == 'gradient_boosted_trees':
      model = GBTRegressor(
          featuresCol=FEATURES_COL,
          labelCol=LABEL_COL,
      )
    elif model_name == 'xgboost':
      model = SparkXGBRegressor(
          features_col=FEATURES_COL,
          label_col=LABEL_COL,
          num_workers=8,              # 64 cores / 4 = 16 workers (leave some for Spark)
          use_gpu=False,
          validation_indicator_col='is_validation',
          # Early stopping
          early_stopping_rounds=20,
          eval_metric='rmse',
          max_depth=params['max_depth'],
          learning_rate=params['learning_rate'],
          n_estimators=params['num_round']
      )
    model.setParams(**params)
    preprocess_pipeline = preprocess_pipeline_func()     #preprocessing_pipeline_func()
    stages = preprocess_pipeline.getStages()
    stages.append(model)
    pipeline = Pipeline(stages=stages)
    return pipeline
  


def cv_eval(train_preds, test_preds):
  """
  Input: transformed df with prediction and label
  Output: desired score 
  """
   
  #rdd_preds = preds.select(['prediction', LABEL_COL]).rdd
  evaluator_rmse = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse")
  evaluator_mae = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="mae")
      
  train_rmse = np.round(evaluator_rmse.evaluate(train_preds),4)
  train_mae = np.round(evaluator_mae.evaluate(train_preds),4)
  test_rmse = np.round(evaluator_rmse.evaluate(test_preds),4)
  test_mae = np.round(evaluator_mae.evaluate(test_preds),4)
  rmse = [train_rmse, test_rmse]
  mae = [train_mae, test_mae]

  return [rmse, mae]

def get_fold_weights(k, weights_option=0):
  # Define weights (more weight on recent folds)
  # Assuming scores are in chronological order [fold1, fold2, fold3, fold4, fold5]
  n_folds = k
  if weights_option == 0:
    # Option 0: Equal weights (simple)
    weights = np.ones(n_folds)
  elif weights_option == 1:
    # Option 1: Linear weights (simple)
    weights = np.arange(1, n_folds + 1)  # [1, 2, 3, 4, 5]
  elif weights_option == 2:
    # Option 2: Exponential weights (more aggressive)
    decay = 0.5  # adjust this - smaller = more emphasis on recent
    weights = np.array([decay ** (n_folds - i - 1) for i in range(n_folds)])
    # e.g., for 5 folds: [0.0625, 0.125, 0.25, 0.5, 1.0]
  return weights


def timeSeriesSplitCV(train_dataset,  param_grid, pipeline_func, model_type, k=3, blocking=False, sampling='under', metric='rmse', verbose=True, balance_col=BALANCE_COL):
  '''
  Perform timSeriesSplit k-fold cross validation 
  '''
  # Initiate trackers
  best_score = 100
  best_param_vals = None
   
  df=train_dataset
  n=df.count()
  df = df.withColumn("row_id", f.row_number().over(Window.partitionBy().orderBy("FL_DATE")))  # . #flight_date
  chunk_size = int(n/(k+1))
  
  #parameter_names, parameter_values = parameter_sets(param_grid)
  print('')
  print(f'Number of validation datapoints for each fold is {chunk_size:,}')
  print("************************************************************")
  
  
  
  for p in param_grid:        #for p in parameters:
    pipeline = get_model(model_type, pipeline_func, p)   #get_model(model_type, pipeline_func, p)
    
    # Print parameter set
    param_print = p
    #param_print = {x[0]:x[1] for x in zip(parameter_names,p)}
    print(f"Parameters: {param_print}")   
    
    # Track score
    scores=[]
    
    # Start k-fold
    for i in range(k):
      
      
      train_df = df.filter(f.col('row_id') <= chunk_size * (i+1)).cache()
     
      # Create dev set
      dev_df = df.filter((f.col('row_id') > chunk_size * (i+1))&(f.col('row_id') <= chunk_size * (i+2))).cache()  

      # Apply sampling on train if selected
      if sampling == 'under':
        train_df = undersample_majority_class(train_df, balance_col, sampling_strategy=1.0, seed=42)
        train_df = train_df.cache()
    
        
      #print info on train and dev set for this fold
          
      # Fit params on the model
      model = pipeline.fit(train_df)
      train_pred = model.transform(train_df)
      dev_pred = model.transform(dev_df)
    
      score = cv_eval(train_pred,dev_pred) 
     

      scores.append(score) #rmse only#dev score only
      print(f'    Number of training datapoints for fold number {i+1} is {train_df.count():,} with a {metric} score of {score:.4f} for train and test data') 
      print('------------------------------------------------------------')
      # Set best parameter set to current one for first fold
      if best_param_vals == None:
        best_param_vals = p
    
    # Take average of all scores
    avg_score = np.average(scores)    
    # Update best score and parameter set to reflect optimal dev performance
    if avg_score < best_score:  # trying to minimize rmse
      previous_best = best_score
      best_score =  avg_score  # should this be score ?   #
      best_parameters = param_print
      best_param_vals = p
      print(f'new best score of {best_score:.4f}')
    else:
      print(f'Result was not better, score was {avg_score:.4f} with best {metric} score {best_score:.4f}')
    print("************************************************************")
  
  print(best_parameters)
  print(f'Best {metric} score is {best_score:.4f} with parameters {best_param_vals}')
  print("************************************************************")
  return best_parameters, best_score
#############################################################




##############################################################
def timeSeriesSplitCV_rolling(train_dataset,  param_grid, pipeline_func, model_type, k=3, sampling='under', metric='rmse', verbose=True, balance_col=BALANCE_COL):
    '''
    Perform timSeriesSplit k-fold cross validation 
    '''
    # Initiate trackers
    best_score = 100
    best_param_vals = None
    
    df=train_dataset
    n=df.count()
    df = df.withColumn("row_id", f.row_number().over(Window.partitionBy().orderBy("FL_DATE")))  # . #flight_date

    train_window_size = 2 #chunks
    total_chunks = train_window_size + k 
    chunk_size = int(n/total_chunks)
    print(f"Method: Rolling Window (train size: {train_window_size} chunks)")
    print(f"Total rows: {n:,}, Chunk size: {chunk_size:,}, Folds: {k}")
    print("=" * 60)
    print('')
    print(f'Number of validation datapoints for each fold is {chunk_size:,}')
    print("************************************************************")
    
    #parameter_names, parameter_values = parameter_sets(param_grid)
    #chunk_size = int(n/(k+1))
    # get total number of parameter sets in the dict param_grid
    param_set_total = len(param_grid)
    print(f'There are {param_set_total} parameter sets to try')
    
   
    for i, p in enumerate(param_grid, start=1):        #for p in parameters:
      print("************************************************************")
      print(f'Running parameter set {i} of {param_set_total}')
            
      pipeline = get_model(model_type, pipeline_func, p)   #get_model(model_type, pipeline_func, p)
      
      # Print parameter set
      param_print = p
      #param_print = {x[0]:x[1] for x in zip(parameter_names,p)}
      print(f"Parameters: {param_print}")   
      
      # Track score
      scores=[]
      
      # Start k-fold
      for i in range(k):


        # If TimeseriesSplit with fixed size rolling window
        train_start = chunk_size*i + 1
        train_end = chunk_size * (i + train_window_size)
        dev_start = train_end + 1
        dev_end = chunk_size * (i + train_window_size + 1)

        train_df = df.filter(
                  (F.col('row_id') >= train_start) & (F.col('row_id') <= train_end)
              ).cache()
        # Create dev set     
        dev_df = df.filter(
                  (F.col('row_id') >= dev_start) & (F.col('row_id') <= dev_end)
              ).cache()
        
        print(f"Fold {i+1}: Train [{train_start:,} - {train_end:,}] ({train_df.count():,}) → Test [{dev_start:,} - {dev_end:,}] ({dev_df.count():,})")
              
        #yield train_df, dev_df

        # Apply sampling on train if selected
        if sampling == 'under':
          train_df = undersample_majority_class(train_df, balance_col, sampling_strategy=1.0, seed=42)
          # Add validation indicator (20% for validation)
        train_df = train_df.withColumn(
          "is_validation",
          F.when(F.rand(seed=42) < 0.2, True).otherwise(False)
        )
        train_df = train_df.cache()
      
          
        #print info on train and dev set for this fold
            
        # Fit params on the model
        model = pipeline.fit(train_df)
        train_pred = model.transform(train_df)
        dev_pred = model.transform(dev_df)
      
        score = cv_eval(train_pred, dev_pred) 
        #print("score", score)
        score = score[0][1]
        scores.append(score) #rmse only#dev score only
        print(f'    Number of training datapoints for fold number {i+1} is {train_df.count():,} with a {metric} score of {score:.4f}') 
        print('------------------------------------------------------------')
        # Set best parameter set to current one for first fold
        if best_param_vals == None:
          best_param_vals = p
      
      # Take average of all scores
      avg_score = np.average(scores)  
      weights_option = 0
      weights = get_fold_weights(k, weights_option)
      # Take WEIGHTED average of all scores
      weighted_avg_score = np.average(scores, weights=weights)  
      # Print comparison
      print(f"Params: {param_print}")
      print(f"  Fold scores: {[round(s, 4) for s in scores]}")
      print(f"  Unweighted avg: {unweighted_avg:.4f} | Weighted avg: {weighted_avg:.4f} | Diff: {weighted_avg - unweighted_avg:.4f}")
      print()
      # Use weighted for selection
      avg_score = weighted_avg_score
      # Update best score and parameter set to reflect optimal dev performance
      if avg_score < best_score:  # trying to minimize rmse
        previous_best = best_score
        best_score =  avg_score  # should this be score ?   #
        best_parameters = param_print
        best_param_vals = p
        print(f'new best score of {best_score:.4f}')
      else:
        print(f'Result was not better, score was {avg_score:.4f} with best {metric} score {best_score:.4f}')
      print("************************************************************")
    
    
    print(f'Best {metric} score is {best_score:.4f} for parameter set {best_param_vals}')
    return best_parameters, best_score



# COMMAND ----------

#from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor,  GBTRegressor, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from xgboost.spark import SparkXGBRegressor, SparkXGBClassifier
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from itertools import product

USE_UNDERSAMPLING = True
SAMPLING_STRATEGY = 1.0 # equal count for majority and minority class
DELAY_THRESHOLD = 15
BALANCE_COL = "DEP_DEL15"

# Model configuration
DATE_COL = "FL_DATE"
LABEL_COL = 'DEP_DELAY_LOG'
FEATURES_COL = 'features'   #"features_scaled"



#model_names = [ 'random_forest', 'gradient_boosted_trees', 'decision_tree',]
model_names = ['xgboost']

for model_name in model_names:
    print(f'Running {model_name} model')
    model_results = []

    tuning_stage = 2 #1
    param_grid = create_param_grid_by_stage_xgb(tuning_stage)



    best_parameters, best_score =  timeSeriesSplitCV_rolling(train_5y,
                                                    param_grid, 
                                                    preprocess_pipeline_func, model_name, 
                                                    k=3, 
                                                    sampling='under', metric='rmse', 
                                                    verbose=True,
                                                    balance_col=BALANCE_COL)
    model_results.append({
        'model_name': model_name,
        'best_parameters': best_parameters,
        'best_score': best_score
    })
    print('-------------------------------------------------------------')
    print(f'Best parameters for {model_name} are {best_parameters} and best score is {best_score}')
    print('-------------------------------------------------------------')
  

#actual_delay = exp(prediction) - 1

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC #### Grid search result summary from different runs
# MAGIC Best Parameter from GridSearch for DecisionTreeRegressor
# MAGIC new best score of 44.4488
# MAGIC ************************************************************
# MAGIC Training on full train dataset, and validating on dev dataset with best parameters from CV:
# MAGIC {'maxDepth': 10, 
# MAGIC 'maxBins': 50, 
# MAGIC 'minInstancesPerNode': 5, 
# MAGIC 'minInfoGain': 0.1}
# MAGIC
# MAGIC We notice that maxDepth and maxBins are at upper end of the range we tested with, suggesting, we may have to try larger values for these parametrs. 
# MAGIC
# MAGIC DecisionTreeRegressor : Best parameter for best score:
# MAGIC Parameters: {'maxDepth': 10, 
# MAGIC 'maxBins': 50, 
# MAGIC 'minInstancesPerNode': 5, 
# MAGIC 'minInfoGain': 0.1}
# MAGIC
# MAGIC RandomForest : Best Parameter for Best Score of 1.3113
# MAGIC Parameters: {'maxDepth': 5, 'maxBins': 20, 'minInstancesPerNode': 2, 'minInfoGain': 0.0, 'numTrees': 20, 'subsamplingRate': 1.0}
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

#Training on full train set and evaluating on holdout set

def calculate_rmse_in_minutes(log_predictions, prediction_col, label_col):
    """Calculate RMSE in original minutes scale."""
    
    metrics_df = log_predictions.select(
        # RMSE
        F.sqrt(
            F.avg(
                F.pow(
                    (F.exp(F.col(prediction_col)) - 1) - (F.exp(F.col(label_col)) - 1),
                    2
                )
            )
        ).alias("rmse"),
        # MAE
        F.avg(
            F.abs(
                (F.exp(F.col(prediction_col)) - 1) - (F.exp(F.col(label_col)) - 1)
            )
        ).alias("mae")
    )
    
    result = metrics_df.collect()[0]
    return result["rmse"], result["mae"]
   
def calculate_error_distribution(predictions, prediction_col, label_col):
    """Calculate error distribution statistics in original minutes scale."""
    
    # Add error column
    errors_df = predictions.withColumn(
        "abs_error",
        F.abs(
            (F.exp(F.col(prediction_col)) - 1) - (F.exp(F.col(label_col)) - 1)
        )
    )
    
    # Calculate percentiles
    percentiles = errors_df.select(
        F.percentile_approx("abs_error", 0.5).alias("median_error"),
        F.percentile_approx("abs_error", 0.9).alias("p90_error"),
        F.percentile_approx("abs_error", 0.99).alias("p99_error")
    ).collect()[0]
    
    return percentiles

def cv_eval_in_minutes(log_predictions, prediction_col, label_col):
  """
  Input: transformed df with prediction and label
  Output: desired score 
  """
   
  rmse_in_minutes, mae_in_minutes = calculate_rmse_in_minutes(log_predictions, prediction_col, label_col)
 
  return [rmse_in_minutes, mae_in_minutes]

def run_holdout_eval(train_df, test_df, preprocess_pipeline_func, model_name, best_parameters, balance_col, label_col, prediction_col):
    print("Balancing using undersampling...")
    train_df = undersample_majority_class(train_df, balance_col, sampling_strategy=0.5, seed=42)
    train_df = train_df.cache()

    pipeline = get_model(model_name, preprocess_pipeline_func, best_parameters)
    print("starting model traing...")
    pipeline = pipeline.fit(train_data_5y)
    print("finished training model")
    # Check if model has feature importance
    print("Extracting feature importance")
    if hasattr(model_name, 'featureImportances'):
        importances = model.featureImportances.toArray()
        feature_importance = extract_feature_importance_with_names( model_name, importances, categorical_feature_encoded_names, numerical_features)
    else:
        feature_importance = None 
    #Transform training and holdout data for comparison
    print("predicting on train data")
    train_pred = pipeline.transform(train_df)
    print("predicting on holdout data")
    holdout_pred = pipeline.transform(test_df)
    train_rmse_minutes, train_mae_minutes = cv_eval_in_minutes(train_pred, prediction_col, label_col)
    holdout_rmse_minutes, holdout_mae_minutes  = cv_eval_in_minutes(holdout_pred, prediction_col, label_col)
    print("-"*50)
    print(f'    Training data (2015-2018) result (minutes): rmse={train_rmse_minutes:.4f}, mae={train_mae_minutes:.4f}')
    print(f'    Final holdout (2019) result (minutes): rmse={holdout_rmse_minutes:.4f}, mae={holdout_mae_minutes:.4f}')
    print("-"*50)
    #calculate error distribution
    train_percentiles = calculate_error_distribution(train_pred, prediction_col, label_col)
    holdout_percentiles = calculate_error_distribution(holdout_pred, prediction_col, label_col)
    print(f'    Training data (2015-2018) error distribution (minutes): median={train_percentiles["median_error"]:.4f}, p90={train_percentiles["p90_error"]:.4f}, p99={train_percentiles["p99_error"]:.4f}')
    print(f'    Final holdout (2019) error distribution (minutes): median={holdout_percentiles["median_error"]:.4f}, p90={holdout_percentiles["p90_error"]:.4f}, p99={holdout_percentiles["p99_error"]:.4f}')
    print("-"*50)
    

    return train_pred, holdout_pred, train_scores, holdout_scores, feature_importance





# COMMAND ----------

from pyspark.sql import functions as F

best_parameters = {
    'maxDepth': 7,
    'learning_rate': 0.1, 
    'num_round': 100,
    #'maxBins': 50,
    #'minInstancesPerNode': 1,
    #'minInfoGain':  0.1
}

model_name = 'xgboost'   #'decision_tree'
    
log_train_predictions, log_holdout_predictions, train_scores, holdout_scores, feature_importance = run_holdout_eval(train_data_5y, test_data_5y, preprocess_pipeline_func, model_name, best_parameters, BALANCE_COL, LABEL_COL, "prediction")

print(f"  5 year Evaluation complete for model: {model_name} with best parameters: {best_parameters}")
""" 
rmse_in_minutes, mae_in_minutes = calculate_rmse_in_minutes(holdout_predictions, "prediction", LABEL_COL)
print(f"RMSE: {rmse_in_minutes:.2f} minutes")
print(f"MAE: {mae_in_minutes:.2f} minutes")

result = calculate_error_distribution(holdout_predictions, "prediction", LABEL_COL)
print(f"Median error: {result['median_error']:.2f} minutes")
print(f"90th percentile error: {result['p90_error']:.2f} minutes")
print(f"99th percentile error: {result['p99_error']:.2f} minutes")
"""

# COMMAND ----------

# MAGIC %md
# MAGIC #### Summary of holdout results using best parameters
# MAGIC #### Decision Tree
# MAGIC Median error: 1.05 minutes
# MAGIC 90th percentile error: 30.70 minutes
# MAGIC 99th percentile error: 186.30 minutes
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

from pyspark.ml.tuning import ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor
import pyspark.sql.functions as f
from pyspark.sql.window import Window

from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml import Pipeline
from itertools import product

USE_UNDERSAMPLING = True
SAMPLING_STRATEGY = 0.5
DELAY_THRESHOLD = 15
BALANCE_COL = "DEP_DEL15"

# Model configuration
DATE_COL = "FL_DATE"
LABEL_COL = "DEP_DELAY"
FEATURE_COL = "features_scaled"


def preprocess_pipeline_func():
    """Create a pipeline for feature transformation."""
    
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
    transformed_features =  numerical_features  + [f"{feature}_encoded" for feature in categorical_features] 

    # Create vector assembler
    assembler = VectorAssembler(inputCols=transformed_features, outputCol="features_unscaled", handleInvalid="keep")
    stages.append(assembler)

    # Create standard scaler
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features_scaled", withStd=True, withMean=False)
    stages.append(scaler)

    # Create the preprocessing pipeline
    preprocess_pipeline = Pipeline(stages=stages)

    return preprocess_pipeline

    

 
def create_simple_param_grid_dt(model_name):
    grid = {
        'maxDepth': [2, 5, 10],
        'maxBins': [10, 20, 50],
        'minInstancesPerNode': [1, 2, 5],
        'minInfoGain': [0.0, 0.1]
    }

    param_grid = [dict(zip(grid.keys(), v)) for v in product(*grid.values())]
    return param_grid

model_name = 'decision_tree'
param_grid = create_simple_param_grid_dt(model_name)

#print(param_grid)
print(param_grid[0].keys())  # Check the keys

best_parameters, best_score =  timeSeriesSplitCV(train_data_5y,
                                                   param_grid, 
                                                   preprocess_pipeline_func, model_name, 
                                                   k=3, 
                                                   blocking=False, sampling='under', metric='rmse', 
                                                   verbose=True,
                                                   balance_col=BALANCE_COL)
print(best_parameters)
print(best_score)

#actual_delay = exp(prediction) - 1

# COMMAND ----------





def cv_eval(preds):
  """
  Input: transformed df with prediction and label
  Output: desired score 
  """
  rdd_preds_m = preds.select(['prediction', 'label']).rdd
  rdd_preds_b = preds.select('label','probability').rdd.map(lambda row: (float(row['probability'][1]), float(row['label'])))
  metrics_m = MulticlassMetrics(rdd_preds_m)
  metrics_b = BinaryClassificationMetrics(rdd_preds_b)
  F2 = np.round(metrics_m.fMeasure(label=1.0, beta=2.0), 4)
  pr = metrics_b.areaUnderPR
  return F2, pr

def timeSeriesSplitCV(dataset, param_grid, pipeline_func, model_type, k=3, blocking=False, sampling=None, metric='f2', verbose=True):
  '''
  Perform timSeriesSplit k-fold cross validation 
  '''
  # Initiate trackers
  best_score = 0
  best_param_vals = None
   
  df=dataset
  n=df.count()
  df = df.withColumn("row_id", f.row_number().over(Window.partitionBy().orderBy("flight_date")))
  chunk_size = int(n/(k+1))
  
  parameter_names, parameters = parameter_sets(param_grid)
  print('')
  print(f'Number of validation datapoints for each fold is {chunk_size:,}')
  print("************************************************************")
  
  if len(parameters) == 1:
    print('you only entered one set of parameters you doofus')
  
  for p in parameters:
    pipeline = get_model(model_type, pipeline_func, p)
    
    # Print parameter set
    param_print = {x[0]:x[1] for x in zip(parameter_names,p)}
    print(f"Parameters: {param_print}")
    
    # Track score
    scores=[]
    
    # Start k-fold
    for i in range(k):
      
      # If TimeseriesSplit 
      if not blocking:
        train_df = df.filter(f.col('row_id') <= chunk_size * (i+1)).cache()
      # If BlockingSplit
      else:
        train_df = df.filter((f.col('row_id') > chunk_size * i)&(f.col('row_id') <= chunk_size * (i+1))).cache()
        
      # Create dev set
      dev_df = df.filter((f.col('row_id') > chunk_size * (i+1))&(f.col('row_id') <= chunk_size * (i+2))).cache()  

      # Apply sampling on train if selected
      if sampling=='down':
        train_df = downsample(train_df)
        train_df = train_df.cache()
      elif sampling=='up':
        train_df = upsample(train_df)
        train_df = train_df.cache()
      elif sampling=='weights':
        train_df = add_class_weights(train_df).cache()
        
      #print info on train and dev set for this fold
      if verbose:
        print('    TRAIN set for fold {} goes from {} to {}, count is {:,} flights ({})'.format((i+1), 
                                                                                       train_df.agg({'flight_date':'min'}).collect()[0][0],
                                                                                       train_df.agg({'flight_date':'max'}).collect()[0][0],
                                                                                       train_df.count(),
                                                                                       sampling + '-sampled' if sampling else 'no sampling'))
        print('    DEV set for fold {} goes from {} to {}, count is {:,} flights'.format((i+1), 
                                                                                       dev_df.agg({'flight_date':'min'}).collect()[0][0],
                                                                                       dev_df.agg({'flight_date':'max'}).collect()[0][0],
                                                                                       dev_df.count()))      
      # Fit params on the model
      model = pipeline.fit(train_df)
      dev_pred = model.transform(dev_df)
      if metric=='f2':
        score = cv_eval(dev_pred)[0]
      elif metric=='pr':
        score = cv_eval(dev_pred)[1]
      scores.append(score)
      print(f'    Number of training datapoints for fold number {i+1} is {train_df.count():,} with a {metric} score of {score:.2f}') 
      print('------------------------------------------------------------')
      # Set best parameter set to current one for first fold
      if best_param_vals == None:
        best_param_vals = p
    
    # Take average of all scores
    avg_score = np.average(scores)    
    # Update best score and parameter set to reflect optimal dev performance
    if avg_score > best_score:
      previous_best = best_score
      best_score = avg_score
      best_parameters = param_print
      best_param_vals = p
      print(f'new best score of {best_score:.2f}')
    else:
      print(f'Result was no better, score was {avg_score:.2f} with best {metric} score {best_score:.2f}')
    print("************************************************************")
  
  # Train on full df
  print('Training on full train dataset, and validating on dev dataset with best parameters from CV:')
  print(best_parameters)
    
  if verbose:
    print('    TRAIN set for best parameter fitted model goes from {} to {}, count is {:,} flights ({})'.format(train_df.agg({'flight_date':'min'}).collect()[0][0],
                                                                                                     train_df.agg({'flight_date':'max'}).collect()[0][0],
                                                                                                     train_df.count(),
                                                                                                     sampling + '-sampled' if sampling else 'no sampling'))
  return best_parameters, best_score

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder

spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setCheckpointDir("/dbfs/tmp/ml_checkpoints")

train_preprocessed_data = train_preprocessed_data.cache()
test_preprocessed_data = test_preprocessed_data.cache()


#maxIter=50,
#maxDepth=5,
#stepSize=0.1,
#subsamplingRate=1.0

gbt = GBTRegressor(
    featuresCol=FEATURES_COL,
    labelCol=LABEL_COL,
    predictionCol='prediction'
    )

# Create parameter grid
paramGrid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5, 10]) \
    .addGrid(gbt.maxIter, [20, 50,  100]) \
    .addGrid(gbt.stepSize, [0.01, 0.1, 0.5]) \
    .addGrid(gbt.subsamplingRate, [0.8, 1.0]) \
    .build()

print(f"Number of parameter combinations: {len(paramGrid)}")
#print(f"Parameter combinations: {paramGrid}")

# Initiate trackers
lowest_score = 5
best_param_vals = None
metric = 'rmse'



# Apply the parameters
for idx, params in enumerate(paramGrid):
    # Track score
    scores=[]
    # Set best parameter set to current one for first iteration
    if best_param_vals == None:
      best_param_vals = params
    param_str = ", ".join([f"{p.name}={v}" for p, v in params.items()])
    print("************************************************************")
    print(f"running with parameter set number {idx}: { param_str}")
    fold_results =  run_gbt_cv(train_preprocessed_data, test_preprocessed_data, params, N_TRAIN_MONTHS,N_TEST_MONTHS,N_FOLDS, seed=42)
    for score in fold_results:
        scores.append(score)
    # Take average of all scores
    avg_score = np.average(scores)    
    # Update best score and parameter set to reflect optimal dev performance
    if avg_score < lowest_score:
      previous_lowest_score = lowest_score
      lowest_score = avg_score
      best_param_vals = params 
      best_param_str = ", ".join([f"{p.name}={v}" for p, v in best_param_vals.items()])
      print(f'new lowest score of {lowest_score:.2f} for parameter set {best_param_str}')
    else:
      print(f'Result was not lower, score was {avg_score:.2f} with lowest {metric} score {lowest_score:.2f}')
    print("************************************************************")

print("Completed Search, best score was {lowest_score:.4f} for parameter set {best_param_str}")

