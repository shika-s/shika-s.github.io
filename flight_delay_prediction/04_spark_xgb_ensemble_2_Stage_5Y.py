# Databricks notebook source
# MAGIC %md
# MAGIC # Flight Sentry - Regression Pipeline
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

# COMMAND ----------

# Team folder
section = "4"
number = "4"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}/"
dbutils.fs.mkdirs(folder_path)
display(dbutils.fs.ls(f"{folder_path}"))

regression_train_checkpoint = 'checkpoint_6_regression_train_2015_2018.parquet/'
regression_test_checkpoint = 'checkpoint_6_regression_test_2019.parquet/'

local_path = '/Workspace/Users/shikhasharma@berkeley.edu/'
dbutils.fs.mkdirs(local_path)
#display(dbutils.fs.ls(f"{local_path}"))

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
        
# create encoded categorical feature names to help understand feature importance
def create_encoded_feature_names(df, categorical_features):
    # find cardinality count for the categorical features
    print("Categorical Features:\n", categorical_features)
    print("="*80)
    categorical_feature_encoded = []
    for col in categorical_features:
        for i in range(0,df.select(col).distinct().count()):
            encoded_col_name = col + "_"+str(i)
            categorical_feature_encoded.append(encoded_col_name)
    return categorical_feature_encoded












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

#Data Checkpoint Read

print("Data Read Checkpoint is enabled")
train_5y = spark.read.parquet(f"{folder_path}{regression_train_checkpoint}") 
test_5y = spark.read.parquet(f"{folder_path}{regression_test_checkpoint}") 
display_size(train_5y)
display_size(test_5y)



# COMMAND ----------

display(train_5y.limit(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Check Column Cardinality

# COMMAND ----------

# Check column counts again
# Extract categorical and numerical columns based on column type

categorical_columns, numerical_columns, timestamp_columns, date_column = create_column_list_by_type(train_5y)
target_encoded_columns  = ['ORIGIN', 'DEST', 'OP_UNIQUE_CARRIER', 'ORIGIN_STATE_ABR', 'DEST_STATE_ABR', 'day_hour_interaction']

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

# find cardinality count for the categorical features

categorical_feature_encoded_names = create_encoded_feature_names(train_5y, categorical_features)
print("Number of Categorical Features:", len(categorical_feature_encoded_names))
print(categorical_feature_encoded_names)
print("="*80)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Categorical Features Summary
# MAGIC
# MAGIC ### Original Categorical Features (7)
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
# MAGIC

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

from pyspark.sql import functions as F
from pyspark.ml.regression import (
    LinearRegression,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GBTRegressor
)
from xgboost.spark import SparkXGBRegressor, SparkXGBClassifier
from pyspark.ml.evaluation import RegressionEvaluator
import pandas as pd
import time
import pyspark.sql.functions as f
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from itertools import product
import numpy as np


# ============================================================================
#  CONFIGURATION
# ============================================================================

# Logging
VERBOSE = False
# Class imbalance handling
USE_UNDERSAMPLING = True
SAMPLING_STRATEGY = 1.0
DELAY_THRESHOLD = 15
BALANCE_COL = "DEP_DEL15"

# Model configuration
DATE_COL = "FL_DATE"
LABEL_COL = "DEP_DELAY_LOG"
FEATURES_COL = "features"




# ============================================================================
# DEFINE ALL MODEL TEMPLATES BASED ON RESULTS FROM GRID SEARCH
# ============================================================================


# Model configurations 
models = {
    "Linear_Regressor": LinearRegression(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=50,
        regParam=0.01,
        elasticNetParam=0.1
    ),
    
    "DecisionTree_Regressor": DecisionTreeRegressor(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxDepth=10,
        minInstancesPerNode=100
    ),
    
    "RandomForest_Regressor": RandomForestRegressor(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        numTrees=50,
        maxDepth=5,
        minInstancesPerNode=200
    ),
    
    "GradientBoost_Regressor": GBTRegressor(
        featuresCol=FEATURES_COL,
        labelCol=LABEL_COL,
        maxIter=50,
        maxDepth=5,
        stepSize=0.1
    ),
    
    "SparkXGB_Regressor": SparkXGBRegressor(
        features_col=FEATURES_COL,
        label_col=LABEL_COL,
        max_depth=7,
        learning_rate=0.1,
        num_round=100
    ),
    "SparkXGB_Classifier": SparkXGBClassifier(
        features_col=FEATURES_COL,
        label_col=LABEL_COL,
        max_depth=7,
        learning_rate=0.1,
        num_round=100
    )
}




# ============================================================================
# HELPER FUNCTIONS
#  undersample_majority_class
# ============================================================================

def undersample_majority_class(train_df, balance_col, sampling_strategy=1.0, seed=42):
    """Undersample majority class to balance dataset."""
    train_df= train_df
    balance_col = balance_col
    sampling_strategy=sampling_strategy
    seed=seed

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



def extract_feature_importance( model_name, importances, transformed_features):
    """
    Extracts feature importance from a trained model.
    Parameters:
        feature_names: List of feature names.
    Returns:
        List of tuples containing feature names and their corresponding importance
        sorted in descending order of importance.
    """
    model_name = model_name
    importances = importances

    # Build expanded feature names
    expanded_features = transformed_features    #numerical_features + categorical_feature_encoded_names
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
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance_df.head(10).iterrows():
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
# MAGIC RESULT from GridSearch Time Series Cross Validation:
# MAGIC based on above run, the lowest score is achieved for following parameter set
# MAGIC
# MAGIC new lowest score of rmse=1.22 for parameter set maxDepth=3, maxIter=20, stepSize=0.1, subsamplingRate=1.0

# COMMAND ----------

# MAGIC %md
# MAGIC #### 2 Stage Architecture

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Stage 1: 
# MAGIC
# MAGIC Classification: 
# MAGIC ====== Best overall GBT config on validation set ======
# MAGIC maxDepth=5, stepSize=0.1, subsamplingRate=0.8, numIters=90, AUC-PR=0.7191, AUC-ROC=0.8945, F0.5=0.6415
# MAGIC
# MAGIC Stage 2: 
# MAGIC
# MAGIC Regression: 
# MAGIC DecisionTreeRegressor : Best parameter for best score: Parameters: {'maxDepth': 10, 'maxBins': 50, 'minInstancesPerNode': 5, 'minInfoGain': 0.1}
# MAGIC
# MAGIC RandomForest : Best Parameter for Best Score of 1.3113 Parameters: {'maxDepth': 5, 'maxBins': 20, 'minInstancesPerNode': 2, 'minInfoGain': 0.0, 'numTrees': 20, 'subsamplingRate': 1.0}
# MAGIC

# COMMAND ----------


def get_configured_model(model_name, r_label_col, b_label_col, feature_col):
    """Return just the model, no preprocessing"""
    
    if model_name == 'rf':
        return RandomForestRegressor(
            featuresCol=feature_col,
            labelCol=r_label_col,
            numTrees=50,
            maxDepth=5,
            maxBins=20,
            minInstancesPerNode=2,
            minInfoGain=0.0,
            subsamplingRate=1.0,
            predictionCol='pred_rf'
        )
    elif model_name == 'gbtr':
        return GBTRegressor(
            featuresCol=feature_col,
            labelCol=r_label_col,
            predictionCol='pred_gbtr',
            maxIter=40,
            maxDepth=5,
            stepSize=0.1,
            subsamplingRate=0.8,
        )
    elif model_name == 'gbtc':
        return GBTClassifier(
            featuresCol=feature_col,
            labelCol=b_label_col,
            predictionCol='pred_gbtc',
            maxIter=40,
            maxDepth=5,
            stepSize=0.1,
            subsamplingRate=0.8,
        )
    elif model_name == 'rfc':
        return RandomForestClassifier(
            featuresCol=feature_col,
            labelCol=b_label_col,
            predictionCol='pred_rfc',
            probabilityCol='class_prob',
            numTrees=40,
            maxDepth=5,
            subsamplingRate=0.8
        )
    elif model_name == 'xgbr':
        return SparkXGBRegressor(
            features_col=feature_col,
            label_col=r_label_col,
            prediction_col='pred_xgbr',
            verbosity=0,
            num_round=100,                    # number of trees
            max_depth=7,
            learning_rate=0.1,                # same as eta
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,                    # L1 regularization
            reg_lambda=1.0,                   # L2 regularization
            early_stopping_rounds=10,         # stops if no improvement
            validation_indicator_col='is_validation',
            eval_metric='rmse',
            num_workers=8,                    # parallel workers
            use_gpu=False
        )
    elif model_name == 'xgbc':
        return SparkXGBClassifier(
            features_col=feature_col,
            label_col=b_label_col,
            prediction_col='pred_xgbc',
            probability_col='prob_xgbc',      # supports probabilities!
            verbosity=0,
            num_round=100,
            max_depth=7,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            early_stopping_rounds=10,
            eval_metric='logloss',
            validation_indicator_col='is_validation',
            num_workers=8,
            use_gpu=False
        )


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

    return preprocess_pipeline, transformed_features

   

# COMMAND ----------

#Training on full train set and evaluating on holdout set
from pyspark.sql import functions as F

def calculate_metrics_in_minutes(df, prediction_col, label_col):
    """Calculate RMSE and MAE in original minutes scale."""
    
    metrics_df = df.select(
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



def cv_eval_in_minutes(log_pred_df, prediction_col="ensemble_pred", label_col="DEP_DELAY"):
    """
    Input: transformed df with prediction and label columns
    Output: [rmse_in_minutes, mae_in_minutes]
    """
    
    rmse_in_minutes, mae_in_minutes = calculate_metrics_in_minutes(log_pred_df, prediction_col, label_col)
    
    return [rmse_in_minutes, mae_in_minutes]


def run_holdout_eval(train_df, test_df, preprocess_pipeline_func, model_name, best_parameters, balance_col, label_col, prediction_col):
    print("Balancing using undersampling...")
    train_df = undersample_majority_class(train_df, balance_col, sampling_strategy=1.0, seed=42)
    # Add validation indicator (20% for validation)
    train_df = train_df.withColumn(
          "is_validation",
          F.when(F.rand(seed=42) < 0.2, True).otherwise(False)
        )
    train_df = train_df.cache()

    pipeline = get_model(model_name, preprocess_pipeline_func, best_parameters)
    print("starting model traing...")
    pipeline = pipeline.fit(train_df)
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

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier
from xgboost.spark import SparkXGBRegressor, SparkXGBClassifier
from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml import Pipeline

def get_feature_importance(fitted_model, feature_names):
    """Extract feature importance from a trained model."""
    
    # Get feature importances (works for RF, GBT, DecisionTree)
    importances = fitted_model.featureImportances.toArray()
    if len(importances) != len(feature_names):
        importances = importances[:len(feature_names)]

    
    # Create dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df


def train_ensemble(train_df, feature_col=FEATURES_COL,balance_col="DEP_DEL15", b_label_col="DEP_DEL15", r_label_col="DEP_DELAY_LOG"):
    """Train multiple models and return them"""
    
    model_names = ['xgbr', 'xgbc']  # 'gbtr', 'gbtc'
    

    # Balance data
    train_df = undersample_majority_class(train_df, balance_col, sampling_strategy=1.0, seed=42)
    # Add validation indicator (20% for validation)
    train_df = train_df.withColumn(
          "is_validation",
          F.when(F.rand(seed=42) < 0.2, True).otherwise(False)
        )
    train_df = train_df.cache()
    # Step 1: Fit preprocessing ONCE
    print("Fitting preprocessing pipeline...")
    preprocess_pipeline, transformed_features = preprocess_pipeline_func()
    fitted_preprocessor = preprocess_pipeline.fit(train_df)
    
    # Step 2: Transform training data ONCE
    print("Transforming training data...")
    train_preprocessed = fitted_preprocessor.transform(train_df)
    train_preprocessed = train_preprocessed.cache()
    
    # Step 3: Train each model on preprocessed data
    fitted_models = {'preprocessor': fitted_preprocessor}  # save preprocessor
    importances = []
    for name in model_names:
        print(f"Training {name}...")
        model = get_configured_model(name, r_label_col, b_label_col, feature_col)
        fitted_model = model.fit(train_preprocessed)
        fitted_models[name] = fitted_model
        print(f"Finished Training {name}...")
        # get feature importance for each model
        try:
            booster = fitted_model.get_booster()
            importance_dict = booster.get_score(importance_type='gain')
            # Create DataFrame directly
            importance_df = pd.DataFrame({
                'feature': transformed_features,
                'importance': [importance_dict.get(f"f{i}", 0) for i in range(len(transformed_features))]
            })
            
            
            # Normalize to 0-1
            importance_df['normalized'] = importance_df['importance'] / importance_df['importance'].sum()
            importance_df = importance_df.sort_values('normalized', ascending=False)
            print("-"*50)
            print(importance_df.head(40))
            print("-"*50)
            importances.append(importance_df)
            #feature_importance = extract_feature_importance(name, importances, transformed_features)
            
            #print(f"Finished getting Feature importance for {name}...")
        except Exception as e:
            print(f"Error getting Feature importance for {name}...")
            print(e)
    return fitted_models, importances


def ensemble_predict(df, fitted_models):      
    """Combine classification probability with regression prediction
    Final = P(delayed) * regression_prediction"""
    
    # Step 1: Preprocess ONCE
    print("Preprocessing data...")
    df_preprocessed = fitted_models['preprocessor'].transform(df)
    
    # Step 2: Apply regression model
    print("Getting regression prediction...")
    df_with_reg = fitted_models['xgbr'].transform(df_preprocessed)
    
     # Step 3: Apply classification model
    print("Getting classification prediction...")
    df_with_both = fitted_models['xgbc'].transform(df_with_reg)
    
    pred_col = 'pred_xgbr'
    prob_col = 'prob_xgbc'
    # Extract probability of class 1 (delayed)
    # Note: GBTClassifier doesn't have probability - use rawPrediction or switch to RFC
    print("Extracting probability of class 1")
    df_with_prob = df_with_both.withColumn(
        prob_col,
        vector_to_array(prob_col)[1]
    )
   
    
    threshold = 0.25
    # Test different strategies
    results_comparison = df_with_prob.withColumn(
        'pred_threshold', 
        F.when(F.col(prob_col) > threshold, F.col(pred_col)).otherwise(0)
    ).withColumn(
        'pred_r_only',
        F.col(pred_col)
    ).withColumn(
        'pred_multiply',
        F.col(prob_col) * F.col(pred_col)
    )
   
    return results_comparison

# COMMAND ----------

# Train
fitted_models, importances = train_ensemble(train_5y)

# Predict
train_results_comparison_df = ensemble_predict(train_5y, fitted_models)
test_results_comparison_df = ensemble_predict(test_5y, fitted_models)

# Evaluate each
for idx,comparison_df in enumerate([train_results_comparison_df, test_results_comparison_df]):
    
    for pred_col in ['pred_threshold', 'pred_r_only', 'pred_multiply']:
        rmse, mae = cv_eval_in_minutes(comparison_df, prediction_col=pred_col, label_col="DEP_DELAY_LOG")
        percentiles = calculate_error_distribution(comparison_df, pred_col, label_col="DEP_DELAY_LOG")
        print("-"*50)
        print("-"*50)
        if idx == 0:
            print(f"    Training data (2015-2018) result (minutes): {pred_col}: rmse={rmse:.4f}, mae={mae:.4f}")
            print("-"*50)
            print(f'    Training data (2015-2018) error distribution (minutes){pred_col} : median={percentiles["median_error"]:.4f}, p90={percentiles["p90_error"]:.4f}, p99={percentiles["p99_error"]:.4f}')
        else:
            print(f"    Final holdout (2019) result (minutes): {pred_col}: rmse={rmse:.4f}, mae={mae:.4f}")
            print("-"*50)
            print(f'    Final holdout (2019) error distribution (minutes) {pred_col}: median={percentiles["median_error"]:.4f}, p90={percentiles["p90_error"]:.4f}, p99={percentiles["p99_error"]:.4f}')
        print("-"*50)
        print("-"*50)
        #print(f"{idx}: {pred_col}: RMSE={rmse:.2f}, MAE={mae:.2f}")
    print("-"*50)
#results.select('ensemble_pred', 'DEP_DELAY', 'prob_delayed', 'pred_rf').show()

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC ## Model Performance Summary
# MAGIC
# MAGIC ### Training Data (2015-2018)
# MAGIC
# MAGIC | Method | RMSE | MAE | Median Error | P90 Error | P99 Error |
# MAGIC |--------|------|-----|--------------|-----------|-----------|
# MAGIC | pred_threshold | 35.18 | 10.72 | 2.80 | 24.59 | 137.43 |
# MAGIC | pred_r_only | 35.16 | 10.95 | 2.76 | 24.53 | 137.21 |
# MAGIC | pred_multiply | 36.35 | 10.15 | **0.73** | **24.29** | 145.98 |
# MAGIC
# MAGIC ### Final Holdout (2019)
# MAGIC
# MAGIC | Method | RMSE | MAE | Median Error | P90 Error | P99 Error |
# MAGIC |--------|------|-----|--------------|-----------|-----------|
# MAGIC | pred_threshold | 42.85 | 12.07 | 2.63 | 26.72 | 161.12 |
# MAGIC | pred_r_only | **42.83** | 12.31 | 2.61 | **26.63** | **161.07** |
# MAGIC | pred_multiply | 44.14 | **11.66** | **0.66** | 27.13 | 169.95 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Observations
# MAGIC
# MAGIC **1. Generalization Gap**
# MAGIC | Method | RMSE Increase (Train → Test) |
# MAGIC |--------|------------------------------|
# MAGIC | pred_threshold | +7.67 (+21.8%) |
# MAGIC | pred_r_only | +7.67 (+21.8%) |
# MAGIC | pred_multiply | +7.79 (+21.4%) |
# MAGIC
# MAGIC All methods show ~22% degradation on holdout - consistent and reasonable for time-based split.
# MAGIC
# MAGIC **2. Method Comparison**
# MAGIC
# MAGIC | Method | Strengths | Weaknesses |
# MAGIC |--------|-----------|------------|
# MAGIC | **pred_r_only** | Best RMSE, best P90/P99 | Higher MAE |
# MAGIC | **pred_threshold** | Balanced performance | Middle of the pack |
# MAGIC | **pred_multiply** | Best median error, best MAE | Worst RMSE, worst P99 (tail errors) |
# MAGIC
# MAGIC **3. Trade-off Insight**
# MAGIC
# MAGIC - `pred_multiply` has the **lowest median error (0.66 min)** - great for typical flights
# MAGIC - But it has the **worst tail errors (P99=170 min)** - bad for extreme delays
# MAGIC - `pred_r_only` is most consistent across all percentiles
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Recommendation
# MAGIC
# MAGIC | Use Case | Best Method |
# MAGIC |----------|-------------|
# MAGIC | **Minimize average error** | `pred_multiply` (lowest MAE) |
# MAGIC | **Minimize large errors** | `pred_r_only` (lowest RMSE, P99) |
# MAGIC | **Balanced / Production** | `pred_threshold` or `pred_r_only` |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC 1. **What matters more ?**
# MAGIC    - Accurate for most flights → `pred_multiply`
# MAGIC    - Avoid big misses on severe delays → `pred_r_only`
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC Feature Importance for XGBRegressor:
# MAGIC
# MAGIC | Rank | Feature | Importance | Normalized |
# MAGIC |------|---------|------------|------------|
# MAGIC | 1 | num_airport_wide_delays | 1152.0 | 9.26% |
# MAGIC | 2 | hours_since_prev_flight | 1096.0 | 8.81% |
# MAGIC | 3 | dep_delay15_24h_rolling_avg_by_origin_dayofweek_... | 829.0 | 6.66% |
# MAGIC | 4 | prior_flights_today | 641.0 | 5.15% |
# MAGIC | 5 | prior_day_delay_rate | 629.0 | 5.06% |
# MAGIC | 6 | prev_flight_crs_elapsed_time | 571.0 | 4.59% |
# MAGIC | 7 | prev_flight_dep_del15 | 345.0 | 2.77% |
# MAGIC | 8 | dep_time_sin | 344.0 | 2.77% |
# MAGIC | 9 | dep_delay15_24h_rolling_avg_by_origin_carrier_... | 332.0 | 2.67% |
# MAGIC | 10 | rolling_origin_num_delays_24h | 274.0 | 2.20% |
# MAGIC | 11 | dep_time_cos | 260.0 | 2.09% |
# MAGIC | 12 | rolling_30day_volume | 249.0 | 2.00% |
# MAGIC | 13 | dep_delay15_24h_rolling_avg_by_origin_dayofweek | 228.0 | 1.83% |
# MAGIC | 14 | rolling_origin_delay_ratio_24h_high_corr | 225.0 | 1.81% |
# MAGIC | 15 | rolling_origin_num_flights_24h_high_corr | 200.0 | 1.61% |
# MAGIC | 16 | log_distance_squared_high_corr | 199.0 | 1.60% |
# MAGIC | 17 | carrier_flight_count | 171.0 | 1.37% |
# MAGIC | 18 | airline_reputation_score | 161.0 | 1.29% |
# MAGIC | 19 | OP_UNIQUE_CARRIER_encoded | 156.0 | 1.25% |
# MAGIC | 20 | dep_delay15_24h_rolling_avg_by_origin_high_corr | 152.0 | 1.22% |
# MAGIC | 21 | days_since_epoch | 150.0 | 1.21% |
# MAGIC | 22 | carrier_encoded_x_hour | 137.0 | 1.10% |
# MAGIC | 23 | time_based_congestion_ratio | 130.0 | 1.05% |
# MAGIC | 24 | arr_time_cos | 124.0 | 1.00% |
# MAGIC | 25 | day_hour_interaction_encoded | 122.0 | 0.98% |
# MAGIC | 26 | route_delay_rate_30d | 121.0 | 0.97% |
# MAGIC | 27 | DEST_encoded | 118.0 | 0.95% |
# MAGIC | 28 | origin_pagerank_high_corr | 117.0 | 0.94% |
# MAGIC | 29 | arr_time_sin | 115.0 | 0.92% |
# MAGIC | 30 | origin_airport_lon | 109.0 | 0.88% |
# MAGIC | 31 | CRS_ARR_TIME | 102.0 | 0.82% |
# MAGIC | 32 | delay_propagation_score | 96.0 | 0.77% |
# MAGIC | 33 | origin_1yr_delay_rate | 95.0 | 0.76% |
# MAGIC | 34 | dest_airport_lat | 94.0 | 0.76% |
# MAGIC | 35 | dest_pagerank_high_corr | 91.0 | 0.73% |
# MAGIC | 36 | days_since_last_delay_route | 89.0 | 0.72% |
# MAGIC | 37 | origin_degree_centrality | 89.0 | 0.72% |
# MAGIC | 38 | carrier_x_origin_encoded | 87.0 | 0.70% |
# MAGIC | 39 | carrier_x_dest_encoded | 84.0 | 0.68% |
# MAGIC | 40 | OP_CARRIER_FL_NUM | 83.0 | 0.67% |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Insights
# MAGIC
# MAGIC **Top Feature Categories:**
# MAGIC
# MAGIC | Category | Features | Combined Weight |
# MAGIC |----------|----------|-----------------|
# MAGIC | **Airport Congestion** | num_airport_wide_delays, prior_flights_today, time_based_congestion_ratio | ~15.5% |
# MAGIC | **Rolling Delay Metrics** | dep_delay15_24h_rolling_avg_*, rolling_origin_* | ~16.4% |
# MAGIC | **Previous Flight Info** | hours_since_prev_flight, prev_flight_*, prior_day_delay_rate | ~21.2% |
# MAGIC | **Time Features** | dep_time_sin/cos, arr_time_sin/cos, day_hour_interaction | ~7.7% |
# MAGIC | **Carrier/Route** | OP_UNIQUE_CARRIER_encoded, carrier_*, route_delay_rate_30d | ~6.1% |
# MAGIC
# MAGIC **Takeaway:** Recent operational history (previous flights, rolling delays, airport congestion) dominates predictions - these capture real-time conditions that drive delays.

# COMMAND ----------

# MAGIC %md
# MAGIC ## XGBoost Classifier Feature Importance
# MAGIC
# MAGIC | Rank | Feature | Importance | Normalized |
# MAGIC |------|---------|------------|------------|
# MAGIC | 1 | num_airport_wide_delays | 925.0 | 10.81% |
# MAGIC | 2 | dep_delay15_24h_rolling_avg_by_origin_dayofweek_... | 772.0 | 9.03% |
# MAGIC | 3 | hours_since_prev_flight | 611.0 | 7.14% |
# MAGIC | 4 | prior_day_delay_rate | 551.0 | 6.44% |
# MAGIC | 5 | prior_flights_today | 547.0 | 6.39% |
# MAGIC | 6 | prev_flight_crs_elapsed_time | 360.0 | 4.21% |
# MAGIC | 7 | dep_time_sin | 311.0 | 3.64% |
# MAGIC | 8 | rolling_origin_num_delays_24h | 241.0 | 2.82% |
# MAGIC | 9 | rolling_30day_volume | 231.0 | 2.70% |
# MAGIC | 10 | rolling_origin_num_flights_24h_high_corr | 225.0 | 2.63% |
# MAGIC | 11 | dep_time_cos | 221.0 | 2.58% |
# MAGIC | 12 | rolling_origin_delay_ratio_24h_high_corr | 218.0 | 2.55% |
# MAGIC | 13 | dep_delay15_24h_rolling_avg_by_origin_carrier_... | 209.0 | 2.44% |
# MAGIC | 14 | prev_flight_dep_del15 | 166.0 | 1.94% |
# MAGIC | 15 | dep_delay15_24h_rolling_avg_by_origin_dayofweek | 161.0 | 1.88% |
# MAGIC | 16 | dep_delay15_24h_rolling_avg_by_origin_high_corr | 132.0 | 1.54% |
# MAGIC | 17 | origin_pagerank_high_corr | 104.0 | 1.22% |
# MAGIC | 18 | OP_UNIQUE_CARRIER_encoded | 91.0 | 1.06% |
# MAGIC | 19 | traffic_density_squared_high_corr | 88.0 | 1.03% |
# MAGIC | 20 | log_distance_squared_high_corr | 85.0 | 0.99% |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Comparison: Classifier vs Regressor
# MAGIC
# MAGIC | Rank | XGBC (Classifier) | XGBR (Regressor) |
# MAGIC |------|-------------------|------------------|
# MAGIC | 1 | num_airport_wide_delays (10.8%) | num_airport_wide_delays (9.3%) |
# MAGIC | 2 | dep_delay15_24h_rolling_avg_... (9.0%) | hours_since_prev_flight (8.8%) |
# MAGIC | 3 | hours_since_prev_flight (7.1%) | dep_delay15_24h_rolling_avg_... (6.7%) |
# MAGIC | 4 | prior_day_delay_rate (6.4%) | prior_flights_today (5.2%) |
# MAGIC | 5 | prior_flights_today (6.4%) | prior_day_delay_rate (5.1%) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Insights
# MAGIC
# MAGIC **Similarities (Both Models Agree):**
# MAGIC - `num_airport_wide_delays` is #1 in both
# MAGIC - Top 5 features are nearly identical, just reordered
# MAGIC - Rolling delay averages are important for both
# MAGIC
# MAGIC **Differences:**
# MAGIC
# MAGIC | Feature | Classifier Rank | Regressor Rank | Interpretation |
# MAGIC |---------|-----------------|----------------|----------------|
# MAGIC | `prior_day_delay_rate` | #4 (6.4%) | #5 (5.1%) | More important for predicting IF delayed |
# MAGIC | `prev_flight_dep_del15` | #14 (1.9%) | #7 (2.8%) | More important for predicting HOW LONG |
# MAGIC | `rolling_30day_volume` | #9 (2.7%) | #12 (2.0%) | Volume matters more for classification |
# MAGIC
# MAGIC **Interpretation:**
# MAGIC - **Classifier** focuses more on delay patterns and rates (binary: will it be delayed?)
# MAGIC - **Regressor** focuses more on duration-related features (continuous: how long?)
# MAGIC
# MAGIC This makes intuitive sense - classification cares about likelihood, regression cares about magnitude.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Feature Category Breakdown
# MAGIC
# MAGIC | Category | XGBC Weight | XGBR Weight |
# MAGIC |----------|-------------|-------------|
# MAGIC | Airport Congestion | ~17.2% | ~15.5% |
# MAGIC | Rolling Delay Metrics | ~18.3% | ~16.4% |
# MAGIC | Previous Flight Info | ~17.8% | ~21.2% |
# MAGIC | Time Features | ~6.2% | ~7.7% |
# MAGIC | Carrier/Route | ~4.8% | ~6.1% |
# MAGIC
# MAGIC The classifier relies slightly more on congestion and delay patterns, while the regressor uses more previous flight information to estimate duration.

# COMMAND ----------

# MAGIC %md
# MAGIC pred_threshold: 0.00, 30.00, 178.08
# MAGIC
# MAGIC pred_r_only: 1.71, 27.79, 175.17
# MAGIC
# MAGIC pred_multiply: 0.50, 34.64, 237.75
# MAGIC
# MAGIC
# MAGIC pred_threshold: RMSE=46.25, MAE=12.76
# MAGIC
# MAGIC pred_rf_only: RMSE=46.14, MAE=13.22
# MAGIC
# MAGIC pred_multiply: RMSE=48.59, MAE=13.52
# MAGIC
# MAGIC
# MAGIC RMSE: 48.5915 minutes
# MAGIC
# MAGIC MAE: 13.5206 minutes
# MAGIC
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Updated Experiment Table
# MAGIC
# MAGIC | Exp # | Phase | Classifier Model | Regression Model | Train Data | Test Data | Balance Strategy | Ensemble Prediction Strategy | Train: [RMSE, MAE] (min.) | Test: [RMSE, MAE] (min.) | Binary Metrics (F1, F2, AuPRC) |
# MAGIC |:-----:|:-----:|------------------|------------------|------------|-----------|------------------|------------------------------|---------------------------|--------------------------|-------------------------------|
# MAGIC | 1 | 1 | Logistic Regression | Linear Regression | 2015 Q1,2 | 2015 Q3 | Class weights | Sequential (Filtered Training) | [72.11, 43.69] | [97.49, 53.59] | - |
# MAGIC | 2 | 2 | RandomForest | GBTRegressor | 2015 Q1,2,3 | 2015 Q4 | Undersample (0.5) | Sequential (Filtered Inference) | [19.43, 11.15] | [74.22, 41.97] | - |
# MAGIC | 3 | 3 | RandomForest | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Threshold-Gated | [37.74, 10.74] | [45.38, 12.24] | - |
# MAGIC | 4 | 3 | RandomForest | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [37.50, 10.96] | [45.16, 12.43] | - |
# MAGIC | 5 | 3 | RandomForest | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Probability-weighted | [40.69, 11.73] | [48.24, 13.33] | - |
# MAGIC | 6 | 3 | GBTClassifier | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Threshold-Gated | [38.48, 10.65] | [46.21, 12.19] | - |
# MAGIC | 7 | 3 | GBTClassifier | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [37.88, 11.07] | [45.58, 12.57] | - |
# MAGIC | 8 | 3 | GBTClassifier | GBTRegressor (weighted) | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [48.11, 17.97] | [42.99, 12.89] | - |
# MAGIC | 9 | 3 | SparkXGBClassifier | SparkXGBRegressor | 2015-2018 | 2019 | Undersample (1.0) | Threshold-Gated | [35.18, 10.72] | [42.85, 12.07] | - |
# MAGIC | 10 | 3 | SparkXGBClassifier | SparkXGBRegressor | 2015-2018 | 2019 | Undersample (1.0) | Regression only | [35.16, 10.95] | [42.83, 12.30] | - |
# MAGIC | 11 | 3 | SparkXGBClassifier | SparkXGBRegressor | 2015-2018 | 2019 | Undersample (1.0) | Probability-weighted | [36.35, 10.15] | [44.14, 11.66] | - |
# MAGIC | 12 | 3 | - | SparkXGBRegressor (weighted) | 2015-2018 | 2019 | Undersample (1.0) + Sample weights | Regression only | [54.85, 22.80] | [41.79, 13.20] | [0.659, 0.697, 0.720] |
# MAGIC | 13 | 3 | - | SparkXGBRegressor | 2015-2018 | 2019 | Undersample (1.0) | Regression only | - | [42.40, 11.93] | [0.663, 0.634, 0.735] |
# MAGIC | 14 | 3 | - | Ensemble (Weighted + Unweighted) | 2015-2018 | 2019 | Undersample (1.0) | Average | - | [42.05, 12.40] | [0.668, 0.666, 0.731] |
# MAGIC | 15 | 3 | - | Ensemble (Weighted + Unweighted) | 2015-2018 | 2019 | Undersample (1.0) | **Max** | - | [**41.69**, 13.21] | [0.659, 0.697, 0.722] |
# MAGIC | 16 | 3 | - | Ensemble (Weighted + Unweighted) | 2015-2018 | 2019 | Undersample (1.0) | Min | - | [42.50, **11.92**] | [0.663, 0.633, 0.734] |
# MAGIC
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Summary of New Experiments (Regression only)
# MAGIC
# MAGIC | Exp # | Model | Strategy | Test RMSE | Test MAE | F1 | F2 | AuPRC | Best For |
# MAGIC |:-----:|-------|----------|-----------|----------|-----|-----|-------|----------|
# MAGIC | 12 | XGB (weighted) | Regression | 41.79 | 13.20 | 0.659 | 0.697 | 0.720 | Recall |
# MAGIC | 13 | XGB (unweighted) | Regression | 42.40 | 11.93 | 0.663 | 0.634 | **0.735** | Precision, AuPRC |
# MAGIC | 14 | Ensemble | Average | 42.05 | 12.40 | **0.668** | 0.666 | 0.731 | **Best F1** |
# MAGIC | 15 | Ensemble | Max | **41.69** | 13.21 | 0.659 | **0.697** | 0.722 | **Best RMSE, F2** |
# MAGIC | 16 | Ensemble | Min | 42.50 | **11.92** | 0.663 | 0.633 | 0.734 | **Best MAE** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Findings
# MAGIC
# MAGIC | Metric | Best Experiment | Value |
# MAGIC |--------|-----------------|-------|
# MAGIC | **Best RMSE** | Exp #15 (Ensemble Max) | **41.69 min** |
# MAGIC | **Best MAE** | Exp #16 (Ensemble Min) | **11.92 min** |
# MAGIC | **Best F1** | Exp #14 (Ensemble Average) | **0.668** |
# MAGIC | **Best F2** | Exp #15 (Ensemble Max) | **0.697** |
# MAGIC | **Best AuPRC** | Exp #13 (Unweighted) | **0.735** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Model Configuration (Regression only)
# MAGIC
# MAGIC ```python
# MAGIC # Common parameters for both models
# MAGIC max_depth = 11
# MAGIC learning_rate = 0.05
# MAGIC n_estimators = 200
# MAGIC reg_alpha = 0.2
# MAGIC reg_lambda = 1.0
# MAGIC subsample = 0.8
# MAGIC colsample_bytree = 0.8
# MAGIC
# MAGIC # Sample weights (for weighted model only)
# MAGIC weight = 1.0 if delay <= 60 min
# MAGIC weight = 2.0 if 60 < delay <= 120 min
# MAGIC weight = 2.5 if delay > 120 min
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC %md
# MAGIC
# MAGIC
# MAGIC #### Summary of 2-Stage Experiments
# MAGIC
# MAGIC
# MAGIC
# MAGIC | Exp # | Phase | Classifier Model | Regression Model | Train Data | Test Data | Balance Strategy | Ensemble Prediction Strategy | Train: [RMSE, MAE] (min.) | Test: [RMSE, MAE] (min.) |
# MAGIC |:-----:|:-----:|------------------|------------------|------------|-----------|------------------|------------------------------|---------------------------|--------------------------|
# MAGIC | 1 | 1 | Logistic Regression | Linear Regression | 2015 Q1,2 | 2015 Q3 | Class weights | Sequential (Filtered Training) | [72.11, 43.69] | [97.49, 53.59] |
# MAGIC | 2 | 2 | RandomForest | GBTRegressor | 2015 Q1,2,3 | 2015 Q4 | Undersample (0.5) | Sequential (Filtered Inference) | [19.43, 11.15] | [74.22, 41.97] |
# MAGIC | 3 | 3 | RandomForest | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Threshold-Gated | [37.74, 10.74] | [45.38, 12.24] |
# MAGIC | 4 | 3 | RandomForest | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [37.50, 10.96] | [45.16, 12.43] |
# MAGIC | 5 | 3 | RandomForest | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Probability-weighted | [40.69, 11.73] | [48.24, 13.33] |
# MAGIC | 6 | 3 | GBTClassifier | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Threshold-Gated | [38.48, 10.65] | [46.21, 12.19] |
# MAGIC | 7 | 3 | GBTClassifier | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [37.88, 11.07] | [45.58, 12.57] |
# MAGIC | 8 | 3 | GBTClassifier | GBTRegressor (weighted) | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [48.11, 17.97] | [42.99, 12.89] |
# MAGIC | 9 | 3 | SparkXGBClassifier | SparkXGBRegressor | 2015-2018 | 2019 | Undersample (1.0) | Threshold-Gated | [35.18, 10.72] | [42.85, 12.07] |
# MAGIC | 10 | 3 | SparkXGBClassifier | SparkXGBRegressor | 2015-2018 | 2019 | Undersample (1.0) | Regression only | [35.16, 10.95] | [42.83, **12.30**] |
# MAGIC | 11 | 3 | SparkXGBClassifier | SparkXGBRegressor | 2015-2018 | 2019 | Undersample (1.0) | Probability-weighted | [36.35, 10.15] | [44.14, 11.66] |
# MAGIC |12 | 3| | SparkXGBRegressor| 2015-2018|2019|Undersample(1.0) | Adaptive Ensemble (Weighted-Non-weighted) prediction only||[**41.69**, 13.21 ]
# MAGIC
# MAGIC ---
# MAGIC Desciption of terms used in table:
# MAGIC - Sequential (Filtered Training): Regressor trained only on delayed samples
# MAGIC - Sequential (Filtered Inference): Regressor trained on all data, predictions gated by classifier
# MAGIC - Threshold-Gated prediction: Returns the predicted delay value when the probability exceeds the specified threshold, otherwise returns zero
# MAGIC - Probability-weighted: P(delay) × predicted value
# MAGIC - Weighted Model : Regression Model trained using weights on delay value
# MAGIC ---
# MAGIC
# MAGIC ##### Key Findings
# MAGIC
# MAGIC ###### Best Performers (Test RMSE)
# MAGIC
# MAGIC | Rank | Exp # | Model Combination | Strategy | Test RMSE | Test MAE |
# MAGIC |:----:|:-----:|-------------------|----------|-----------|----------|
# MAGIC | 1 | 10 | XGBClassifier + XGBRegressor | Regression only | **42.83** | 12.30 |
# MAGIC | 2 | 9 | XGBClassifier + XGBRegressor | Threshold-Gated | 42.85 | 12.07 |
# MAGIC | 2 | 8 | GBTClassifier + GBTRegressor (weighted) | Regression only | 42.99 | 12.89 |
# MAGIC
# MAGIC ###### Best Performers (Test MAE)
# MAGIC
# MAGIC | Rank | Exp # | Model Combination | Strategy | Test RMSE | Test MAE |
# MAGIC |:----:|:-----:|-------------------|----------|-----------|----------|
# MAGIC | 1 | 11 | XGBClassifier + XGBRegressor | Probability-weighted | 44.14 | **11.66** |
# MAGIC | 2 | 9 | XGBClassifier + XGBRegressor | Threshold-Gated | 42.85 | **12.07** |
# MAGIC | 3 | 6 | GBTClassifier + GBTRegressor | Threshold-Gated | 46.21 | **12.19** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### Evolution Across Phases
# MAGIC
# MAGIC | Phase | Key Changes | Impact |
# MAGIC |:-----:|-------------|--------|
# MAGIC | 1 | Baseline: Logistic + Linear, class weights | High error (RMSE: 97.49) |
# MAGIC | 2 | Tree-based models, undersampling | Improved but overfit (Train: 19.43, Test: 74.22) |
# MAGIC | 3 | Full data (2015-2018), XGBoost | Best results (Test RMSE: ~43) |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ##### Key Insights
# MAGIC
# MAGIC 1. **XGBoost outperforms:** Experiments 9-11 with SparkXGBoost achieve the best test performance
# MAGIC 2. **Undersampling 1.0 > 0.5:** Full balance (1.0) gives better results than partial (0.5)
# MAGIC 3. **Regression-only works best for RMSE:** Minimal benefit from classifier gating for RMSE
# MAGIC 4. **Probability-weighted best for MAE:** Exp #11 achieves lowest MAE (11.66 min)
# MAGIC 5. **Weighted model helps generalization:** Exp #8 shows smaller train-test gap with weighted model

# COMMAND ----------

# Check the distribution of probabilities
results.select(
    F.avg('pred_rfc').alias('avg_prob'),
    F.min('pred_rfc').alias('min_prob'),
    F.max('pred_rfc').alias('max_prob'),
    F.percentile_approx('pred_rfc', 0.5).alias('median_prob')
).show()

# Check predictions side by side
results.select('pred_rf', 'pred_rfc', 'ensemble_pred', 'DEP_DELAY').show(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Threshold tuning

# COMMAND ----------

# Threshold tuning
# Test different thresholds
for thresh in [0.2, 0.25, 0.3, 0.35, 0.4]:
    results_temp = test_results_comparison_df.withColumn(
        'pred_thresh',
        F.when(F.col('prob_delayed') > thresh, F.col('pred_gbtr')).otherwise(0)
    )
    rmse, mae = cv_eval_in_minutes(results_temp, prediction_col='pred_thresh', label_col="DEP_DELAY")
    print(f"Threshold {thresh}: RMSE={rmse:.2f}, MAE={mae:.2f}")

# COMMAND ----------

for thresh in [0.1, 0.15, 0.2]:
    results_temp = test_results_comparison_df.withColumn(
        'pred_thresh',
        F.when(F.col('prob_delayed') > thresh, F.col('pred_gbtr')).otherwise(0)
    )
    rmse, mae = cv_eval_in_minutes(results_temp, prediction_col='pred_thresh', label_col="DEP_DELAY")
    print(f"Threshold {thresh}: RMSE={rmse:.2f}, MAE={mae:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC | Threshold | RMSE | MAE |
# MAGIC |-----------|------|-----|
# MAGIC | 0.1 | 45.16 | 12.42 |
# MAGIC | 0.15 | 45.16 | 12.41 |
# MAGIC | 0.2 | 45.17 | 12.38 |
# MAGIC | 0.25 | 45.26 | 12.29 |
# MAGIC | 0.3 | 45.38 | **12.24** |
# MAGIC | rf_only | **45.16** | 12.43 |
# MAGIC
# MAGIC **Pattern:**
# MAGIC
# MAGIC - **Lower threshold** → RMSE improves, MAE worsens
# MAGIC - **Higher threshold** → RMSE worsens, MAE improves
# MAGIC
# MAGIC **Why?**
# MAGIC
# MAGIC - Low threshold (0.1) ≈ rf_only (since min_prob = 0.089, almost everything passes)
# MAGIC - Higher threshold → more flights set to 0 → reduces false delay predictions → better MAE
# MAGIC - But setting actual delays to 0 → increases big errors → worse RMSE
# MAGIC
# MAGIC **Trade-off is clear:**
# MAGIC
# MAGIC | Priority | Best Choice | RMSE | MAE |
# MAGIC |----------|-------------|------|-----|
# MAGIC | Minimize large errors | rf_only or threshold ≤ 0.15 | 45.16 | 12.41-12.43 |
# MAGIC | Minimize average error | threshold = 0.3 | 45.38 | 12.24 |
# MAGIC | Balanced | threshold = 0.2-0.25 | 45.17-45.26 | 12.29-12.38 |
# MAGIC
# MAGIC **Recommendation:** 
# MAGIC
# MAGIC Use **threshold = 0.25** as a good balance:
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Prediction Analysis by Features

# COMMAND ----------

write_result = False
if write_result:
    test_results_comparison_df.write.mode('overwrite').parquet(f'{folder_path}regression_predictions.parquet')

# With partitioning (useful for querying)
write_partitioned_result = True
if write_partitioned_result:
    test_results_comparison_df.write.mode('overwrite').partitionBy('ORIGIN_encoded').parquet(f'{folder_path}regression_predictions_partitioned')

# COMMAND ----------

from pyspark.sql import functions as F

def analyze_by_feature(df, feature_col, prediction_col='pred_threshold', label_col='DEP_DELAY_LOG'):
    """Analyze prediction performance by a categorical feature."""
    
    analysis = df.withColumn(
        'error', F.col(prediction_col) - F.col(label_col)
    ).withColumn(
        'abs_error', F.abs(F.col('error'))
    ).withColumn(
        'squared_error', F.pow(F.col('error'), 2)
    ).withColumn(
        # Back-transform to minutes
        'pred_minutes', F.exp(F.col(prediction_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col(label_col)) - 1
    ).withColumn(
        'error_minutes', F.col('pred_minutes') - F.col('actual_minutes')
    ).withColumn(
        'abs_error_minutes', F.abs(F.col('error_minutes'))
    )
    
    result = analysis.groupBy(feature_col).agg(
        F.count('*').alias('count'),
        F.avg('actual_minutes').alias('avg_actual_delay'),
        F.avg('pred_minutes').alias('avg_pred_delay'),
        F.avg('abs_error_minutes').alias('mae'),
        F.sqrt(F.avg(F.pow(F.col('error_minutes'), 2))).alias('rmse'),
        F.avg('error_minutes').alias('bias')  # positive = overpredict, negative = underpredict
    ).orderBy(F.desc('count'))
    
    return result

def analyze_by_year_month(df, prediction_col='pred_threshold', label_col='DEP_DELAY_LOG'):
    """Analyze prediction performance by Year and Month."""
    
    analysis = df.withColumn(
        'MONTH', F.month(F.col('FL_DATE'))  # Extract month from FL_DATE
    ).withColumn(
        'error', F.col(prediction_col) - F.col(label_col)
    ).withColumn(
        'abs_error', F.abs(F.col('error'))
    ).withColumn(
        'squared_error', F.pow(F.col('error'), 2)
    ).withColumn(
        'pred_minutes', F.exp(F.col(prediction_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col(label_col)) - 1
    ).withColumn(
        'error_minutes', F.col('pred_minutes') - F.col('actual_minutes')
    ).withColumn(
        'abs_error_minutes', F.abs(F.col('error_minutes'))
    )
    
    result = analysis.groupBy('YEAR', 'MONTH').agg(
        F.count('*').alias('count'),
        F.avg('actual_minutes').alias('avg_actual_delay'),
        F.avg('pred_minutes').alias('avg_pred_delay'),
        F.avg('abs_error_minutes').alias('mae'),
        F.sqrt(F.avg(F.pow(F.col('error_minutes'), 2))).alias('rmse'),
        F.avg('error_minutes').alias('bias')
    ).orderBy('YEAR', 'MONTH')
    
    return result




analysis_df = test_results_comparison_df

# By airport
print("=== Performance by Origin Airport ===")
analyze_by_feature(analysis_df, 'ORIGIN').show(20)

# By weekend
print("=== Performance by Weekend ===")
analyze_by_feature(analysis_df, 'is_weekend').show()

# By holiday
print("=== Performance by Holiday ===")
analyze_by_feature(analysis_df, 'is_superbowl_week').show()

# By carrier
print("=== Performance by Carrier ===")
analyze_by_feature(analysis_df, 'OP_UNIQUE_CARRIER').show(20)
# By Day of week
print("=== Performance by Day of Week ===")
analyze_by_feature(analysis_df, 'DAY_OF_WEEK').show()

# By YEAR-MONTH
print("=== Performance by YEAR-Month ===")
analyze_by_year_month(analysis_df).show()



# COMMAND ----------

# MAGIC %md
# MAGIC Key Insight from Results - Systematic Underprediction:
# MAGIC
# MAGIC model consistently underpredicts across ALL segments (negative bias everywhere):
# MAGIC
# MAGIC Segment|Avg |ActualAvg| PredictedBias|
# MAGIC |--|--|--|--|
# MAGIC All airports|~10-21 min|~3-9 min|-7 to -15 min|
# MAGIC Superbowl week=0|14.0|4.8|-9.2|
# MAGIC Superbowl week=1|12.9|4.0|-8.9|
# MAGIC All carriers|~10-21 min|~3-7 min|-6 to -15 min

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance Analysis
# MAGIC
# MAGIC ### Performance by Origin Airport (Top 20)
# MAGIC
# MAGIC | Origin | Count | Avg Actual Delay | Avg Pred Delay | MAE | RMSE | Bias |
# MAGIC |--------|------:|----------------:|---------------:|----:|-----:|-----:|
# MAGIC | ATL | 391,701 | 10.94 | 7.37 | 9.82 | 32.52 | -3.57 |
# MAGIC | ORD | 328,129 | 17.32 | 12.70 | 15.47 | 45.02 | -4.62 |
# MAGIC | DFW | 295,645 | 15.51 | 10.93 | 14.25 | 39.24 | -4.58 |
# MAGIC | DEN | 246,472 | 15.80 | 10.87 | 13.77 | 41.82 | -4.94 |
# MAGIC | CLT | 231,325 | 13.44 | 9.01 | 12.12 | 35.55 | -4.43 |
# MAGIC | LAX | 216,481 | 12.91 | 7.78 | 12.32 | 38.10 | -5.13 |
# MAGIC | IAH | 176,601 | 15.50 | 9.38 | 14.46 | 46.22 | -6.13 |
# MAGIC | PHX | 172,578 | 12.24 | 8.06 | 10.97 | 34.48 | -4.18 |
# MAGIC | SFO | 166,750 | 16.57 | 10.25 | 14.92 | 43.95 | -6.32 |
# MAGIC | LGA | 166,297 | 18.29 | 12.74 | 14.66 | 45.85 | -5.55 |
# MAGIC | LAS | 161,620 | 13.16 | 8.08 | 11.40 | 38.96 | -5.08 |
# MAGIC | DTW | 159,608 | 12.95 | 7.09 | 11.94 | 44.44 | -5.86 |
# MAGIC | MSP | 158,199 | 12.05 | 6.45 | 11.05 | 44.89 | -5.60 |
# MAGIC | BOS | 147,016 | 17.36 | 10.86 | 14.50 | 43.13 | -6.50 |
# MAGIC | SEA | 141,352 | 10.97 | 6.08 | 10.33 | 34.56 | -4.88 |
# MAGIC | MCO | 140,387 | 16.61 | 10.51 | 14.14 | 41.84 | -6.10 |
# MAGIC | DCA | 135,619 | 14.43 | 8.63 | 12.65 | 41.88 | -5.79 |
# MAGIC | EWR | 132,093 | 21.42 | 14.39 | 18.42 | 50.16 | -7.04 |
# MAGIC | JFK | 124,759 | 15.86 | 8.33 | 14.59 | 49.02 | -7.53 |
# MAGIC | PHL | 116,359 | 13.86 | 7.60 | 12.45 | 40.85 | -6.26 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Performance by Weekend
# MAGIC
# MAGIC | Is Weekend | Count | Avg Actual Delay | Avg Pred Delay | MAE | RMSE | Bias |
# MAGIC |:----------:|------:|----------------:|---------------:|----:|-----:|-----:|
# MAGIC | No (0) | 5,338,910 | 14.18 | 8.24 | 12.14 | 42.57 | -5.94 |
# MAGIC | Yes (1) | 1,920,097 | 13.58 | 7.54 | 11.82 | 43.70 | -6.04 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Performance by Holiday (Super Bowl Week)
# MAGIC
# MAGIC | Super Bowl Week | Count | Avg Actual Delay | Avg Pred Delay | MAE | RMSE | Bias |
# MAGIC |:---------------:|------:|----------------:|---------------:|----:|-----:|-----:|
# MAGIC | No (0) | 7,133,789 | 14.04 | 8.08 | 12.06 | 42.87 | -5.97 |
# MAGIC | Yes (1) | 125,218 | 12.90 | 6.88 | 11.46 | 42.94 | -6.02 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Performance by Carrier
# MAGIC
# MAGIC | Carrier | Count | Avg Actual Delay | Avg Pred Delay | MAE | RMSE | Bias |
# MAGIC |:-------:|------:|----------------:|---------------:|----:|-----:|-----:|
# MAGIC | WN | 1,327,434 | 11.77 | 8.10 | 9.13 | **23.58** | -3.67 |
# MAGIC | DL | 987,956 | 10.80 | 5.71 | 9.75 | 39.15 | -5.09 |
# MAGIC | AA | 924,121 | 14.81 | 8.45 | 13.40 | 46.45 | -6.36 |
# MAGIC | OO | 812,828 | 16.32 | 8.16 | 13.79 | 57.04 | -8.16 |
# MAGIC | UA | 618,119 | 16.37 | 9.71 | 15.17 | 48.59 | -6.66 |
# MAGIC | YX | 320,614 | 12.68 | 7.35 | 11.06 | 39.57 | -5.33 |
# MAGIC | MQ | 314,648 | 12.95 | 8.06 | 11.00 | 41.30 | -4.90 |
# MAGIC | B6 | 289,228 | 21.67 | 12.64 | 17.81 | 49.86 | **-9.03** |
# MAGIC | OH | 281,637 | 14.32 | 8.32 | 11.90 | 36.77 | -6.01 |
# MAGIC | AS | 261,008 | 9.77 | 5.38 | 9.04 | 26.57 | -4.39 |
# MAGIC | 9E | 252,344 | 14.25 | 7.91 | 11.89 | 46.24 | -6.34 |
# MAGIC | YV | 220,232 | 17.39 | 8.58 | 15.08 | 55.26 | -8.81 |
# MAGIC | NK | 200,162 | 14.16 | 8.44 | 12.71 | 41.64 | -5.72 |
# MAGIC | F9 | 132,383 | 18.80 | 11.82 | 15.80 | 39.12 | -6.98 |
# MAGIC | EV | 128,175 | 21.55 | 9.89 | 18.58 | **70.21** | **-11.66** |
# MAGIC | G4 | 104,429 | 14.65 | 7.20 | 11.55 | 47.40 | -7.45 |
# MAGIC | HA | 83,689 | **5.01** | **2.22** | **4.25** | 22.45 | -2.79 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Performance by Day of Week
# MAGIC
# MAGIC | Day | Count | Avg Actual Delay | Avg Pred Delay | MAE | RMSE | Bias |
# MAGIC |:---:|------:|----------------:|---------------:|----:|-----:|-----:|
# MAGIC | Mon (1) | 1,082,414 | 15.11 | 8.78 | 12.94 | 44.63 | -6.33 |
# MAGIC | Tue (2) | 1,054,699 | 12.73 | 7.22 | 11.03 | **39.81** | -5.51 |
# MAGIC | Wed (3) | 1,047,178 | 13.20 | 7.49 | 11.34 | 40.88 | -5.71 |
# MAGIC | Thu (4) | 1,069,473 | 15.25 | 9.06 | 12.87 | 43.27 | -6.19 |
# MAGIC | Fri (5) | 1,083,950 | 14.42 | 8.55 | 12.37 | 43.62 | -5.87 |
# MAGIC | Sat (6) | 888,765 | 12.71 | 6.64 | 11.19 | 42.82 | -6.07 |
# MAGIC | Sun (7) | 1,032,528 | 14.50 | 8.37 | 12.46 | **44.78** | -6.12 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Performance by Year-Month (2019)
# MAGIC
# MAGIC | Year | Month | Count | Avg Actual Delay | Avg Pred Delay | MAE | RMSE | Bias |
# MAGIC |:----:|:-----:|------:|----------------:|---------------:|----:|-----:|-----:|
# MAGIC | 2019 | Jan | 565,274 | 13.17 | 7.03 | 11.37 | 42.73 | -6.14 |
# MAGIC | 2019 | Feb | 515,727 | **16.32** | **9.73** | **14.08** | **47.74** | -6.59 |
# MAGIC | 2019 | Mar | 617,712 | 12.33 | 7.21 | 10.72 | 39.50 | -5.11 |
# MAGIC | 2019 | Apr | 595,264 | 13.88 | 7.96 | 11.81 | 41.88 | -5.93 |
# MAGIC | 2019 | May | 620,555 | 15.05 | 8.78 | 12.77 | 42.79 | -6.27 |
# MAGIC | 2019 | Jun | 619,578 | **18.48** | **11.34** | **15.50** | **47.61** | **-7.14** |
# MAGIC | 2019 | Jul | 642,841 | 17.03 | 9.62 | 14.58 | **49.15** | **-7.41** |
# MAGIC | 2019 | Aug | 644,497 | 15.61 | 9.15 | 13.45 | 45.47 | -6.46 |
# MAGIC | 2019 | Sep | 594,140 | **10.09** | **5.11** | **8.95** | **36.36** | -4.98 |
# MAGIC | 2019 | Oct | 629,086 | 11.20 | 6.36 | 9.69 | 36.65 | -4.84 |
# MAGIC | 2019 | Nov | 596,465 | **9.91** | **5.27** | **8.79** | 37.25 | **-4.64** |
# MAGIC | 2019 | Dec | 617,868 | 15.07 | 8.97 | 12.80 | 44.86 | -6.10 |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Insights
# MAGIC
# MAGIC ###  Problem Areas (High RMSE/MAE)
# MAGIC
# MAGIC | Category | Worst Performers | RMSE | Insight |
# MAGIC |----------|------------------|------|---------|
# MAGIC | **Airports** | EWR (Newark) | 50.16 | NYC area airports are challenging |
# MAGIC | **Carriers** | EV (ExpressJet) | 70.21 | Regional carriers harder to predict |
# MAGIC | **Months** | Jun, Jul | 47-49 | Summer travel season is volatile |
# MAGIC
# MAGIC ###  Best Performance
# MAGIC
# MAGIC | Category | Best Performers | RMSE | Insight |
# MAGIC |----------|-----------------|------|---------|
# MAGIC | **Airports** | ATL | 32.52 | High volume but predictable |
# MAGIC | **Carriers** | HA (Hawaiian) | 22.45 | Low delay rates, consistent |
# MAGIC | **Months** | Sep, Oct, Nov | 36-37 | Fall months most predictable |
# MAGIC
# MAGIC ###  Systematic Underprediction
# MAGIC
# MAGIC All bias values are **negative**, meaning the model consistently **underpredicts** delays:
# MAGIC
# MAGIC | Category | Worst Bias | Issue |
# MAGIC |----------|-----------|-------|
# MAGIC | EV carrier | -11.66 min | Underpredicts by ~12 min on average |
# MAGIC | B6 (JetBlue) | -9.03 min | Underpredicts by ~9 min on average |
# MAGIC | Jul | -7.41 min | Summer underprediction |
# MAGIC
# MAGIC

# COMMAND ----------

# Check bias for different strategies
for pred_col in ['pred_xgbr', 'pred_threshold']:
    bias_check = analysis_df.withColumn(
        'pred_minutes', F.exp(F.col(pred_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col('DEP_DELAY_LOG')) - 1
    ).select(
        F.avg('actual_minutes').alias('avg_actual'),
        F.avg('pred_minutes').alias('avg_pred'),
        F.avg(F.col('pred_minutes') - F.col('actual_minutes')).alias('bias')
    )
    print(f"=== {pred_col} ===")
    bias_check.show()

# COMMAND ----------

# MAGIC %md
# MAGIC This confirms the regression model itself is underpredicting, and the threshold makes it worse.
# MAGIC Strategy|Avg Actual|Avg Predicted|Bias|
# MAGIC |--|--|--|--|
# MAGIC pred_gbtr (raw)|14.02|6.69|-7.34|
# MAGIC pred_threshold|14.02|4.80|-9.22|
# MAGIC
# MAGIC Root cause: The regression model only predicts ~47% of the actual delay on average.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Error Analysis by Error Distribution

# COMMAND ----------

from pyspark.sql import functions as F

def analyze_by_delay_bins(df, prediction_col='pred_xgbr', label_col='DEP_DELAY_LOG'):
    """Analyze prediction error by actual delay value bins."""
    
    analysis = df.withColumn(
        'pred_minutes', F.exp(F.col(prediction_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col(label_col)) - 1
    ).withColumn(
        'error_minutes', F.col('pred_minutes') - F.col('actual_minutes')
    ).withColumn(
        'abs_error_minutes', F.abs(F.col('error_minutes'))
    ).withColumn(
        'delay_bin',
        F.when(F.col('actual_minutes') <= 0, '0: On-time (<=0)')
         .when(F.col('actual_minutes') <= 15, '1: Short (1-15)')
         .when(F.col('actual_minutes') <= 30, '2: Medium (16-30)')
         .when(F.col('actual_minutes') <= 60, '3: Long (31-60)')
         .when(F.col('actual_minutes') <= 120, '4: Very Long (61-120)')
         .otherwise('5: Extreme (>120)')
    )
    
    result = analysis.groupBy('delay_bin').agg(
        F.count('*').alias('count'),
        F.avg('actual_minutes').alias('avg_actual'),
        F.avg('pred_minutes').alias('avg_pred'),
        F.avg('error_minutes').alias('bias'),
        F.avg('abs_error_minutes').alias('mae'),
        F.sqrt(F.avg(F.pow(F.col('error_minutes'), 2))).alias('rmse'),
        F.percentile_approx('error_minutes', 0.5).alias('median_bias')
    ).orderBy('delay_bin')
    
    return result

# Run analysis
print("=== Error Distribution by Delay Value ===")
analyze_by_delay_bins(analysis_df, 'pred_xgbr', 'DEP_DELAY_LOG').show()

# COMMAND ----------

def analyze_by_delay_bins_detailed(df, prediction_col='pred_xgbr', label_col='DEP_DELAY_LOG'):
    """More detailed binning."""
    
    analysis = df.withColumn(
        'pred_minutes', F.exp(F.col(prediction_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col(label_col)) - 1
    ).withColumn(
        'error_minutes', F.col('pred_minutes') - F.col('actual_minutes')
    ).withColumn(
        'abs_error_minutes', F.abs(F.col('error_minutes'))
    ).withColumn(
        'delay_bin',
        F.when(F.col('actual_minutes') < 0, '00: Early (<0)')
         .when(F.col('actual_minutes') == 0, '01: On-time (0)')
         .when(F.col('actual_minutes') <= 5, '02: 1-5 min')
         .when(F.col('actual_minutes') <= 10, '03: 6-10 min')
         .when(F.col('actual_minutes') <= 15, '04: 11-15 min')
         .when(F.col('actual_minutes') <= 20, '05: 16-20 min')
         .when(F.col('actual_minutes') <= 30, '06: 21-30 min')
         .when(F.col('actual_minutes') <= 45, '07: 31-45 min')
         .when(F.col('actual_minutes') <= 60, '08: 46-60 min')
         .when(F.col('actual_minutes') <= 90, '09: 61-90 min')
         .when(F.col('actual_minutes') <= 120, '10: 91-120 min')
         .when(F.col('actual_minutes') <= 180, '11: 121-180 min')
         .otherwise('12: >180 min')
    )
    
    result = analysis.groupBy('delay_bin').agg(
        F.count('*').alias('count'),
        F.round(F.avg('actual_minutes'), 1).alias('avg_actual'),
        F.round(F.avg('pred_minutes'), 1).alias('avg_pred'),
        F.round(F.avg('error_minutes'), 1).alias('bias'),
        F.round(F.avg('abs_error_minutes'), 1).alias('mae')
    ).orderBy('delay_bin')
    
    return result

print("=== Detailed Error Distribution ===")
analyze_by_delay_bins_detailed(analysis_df, 'pred_xgbr', 'DEP_DELAY_LOG').show(15, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Key insight :**
# MAGIC
# MAGIC | Delay Bin | Actual | Predicted | Bias |
# MAGIC |-----------|--------|-----------|------|
# MAGIC | On-time | 0 | 2.3 | +2.3 (overpredict) |
# MAGIC | 1-5 min | 2.7 | 4.3 | +1.5 (overpredict) |
# MAGIC | 6-10 min | 7.4 | 5.8 | -1.6 (starts underpredicting) |
# MAGIC | >180 min | 314.8 | 40.6 | -274 (severe underpredict) |
# MAGIC
# MAGIC
# MAGIC - Model **overpredicts** small delays (0-5 min)
# MAGIC - Model **underpredicts** larger delays, worsening as delay increases
# MAGIC

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Get the data as pandas
error_dist = analyze_by_delay_bins_detailed(analysis_df, 'pred_xgbr', 'DEP_DELAY_LOG').toPandas()

# Create figure with multiple subplots

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(error_dist['delay_bin'], error_dist['avg_actual'], 'o-', label='Actual', linewidth=2, markersize=8)
ax.plot(error_dist['delay_bin'], error_dist['avg_pred'], 's-', label='Predicted', linewidth=2, markersize=8)

ax.fill_between(error_dist['delay_bin'], error_dist['avg_actual'], error_dist['avg_pred'], 
                alpha=0.3, color='gray', label='Underprediction Gap')

ax.set_xlabel('Delay Bin')
ax.set_ylabel('Minutes')
ax.set_title('Model Underprediction Increases with Delay Magnitude')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Get the data as pandas
#error_dist = analyze_by_delay_bins_detailed(analysis_df, 'pred_gbtr', 'DEP_DELAY').toPandas()

def plot_error_dist(error_dist):
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Actual vs Predicted by bin
    ax1 = axes[0, 0]
    x = range(len(error_dist))
    width = 0.35
    ax1.bar([i - width/2 for i in x], error_dist['avg_actual'], width, label='Actual', color='steelblue')
    ax1.bar([i + width/2 for i in x], error_dist['avg_pred'], width, label='Predicted', color='coral')
    ax1.set_xticks(x)
    ax1.set_xticklabels(error_dist['delay_bin'], rotation=45, ha='right')
    ax1.set_ylabel('Minutes')
    ax1.set_title('Actual vs Predicted Delay by Bin')
    ax1.legend()

    # 2. Bias by bin
    ax2 = axes[0, 1]
    colors = ['green' if b >= 0 else 'red' for b in error_dist['bias']]
    ax2.bar(x, error_dist['bias'], color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(error_dist['delay_bin'], rotation=45, ha='right')
    ax2.set_ylabel('Bias (minutes)')
    ax2.set_title('Prediction Bias by Delay Bin\n(Positive=Overpredict, Negative=Underpredict)')

    # 3. MAE by bin
    ax3 = axes[1, 0]
    ax3.bar(x, error_dist['mae'], color='purple', alpha=0.7)
    ax3.set_xticks(x)
    ax3.set_xticklabels(error_dist['delay_bin'], rotation=45, ha='right')
    ax3.set_ylabel('MAE (minutes)')
    ax3.set_title('Mean Absolute Error by Delay Bin')

    # 4. Count distribution (log scale)
    ax4 = axes[1, 1]
    ax4.bar(x, error_dist['count'], color='teal', alpha=0.7)
    ax4.set_yscale('log')
    ax4.set_xticks(x)
    ax4.set_xticklabels(error_dist['delay_bin'], rotation=45, ha='right')
    ax4.set_ylabel('Count (log scale)')
    ax4.set_title('Sample Distribution by Delay Bin')

    plt.tight_layout()

    plt.show()
    return

plot_error_dist(error_dist)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment with weighted learning

# COMMAND ----------


# Add weights based on actual delay in TRAINING data only
''' 
train_weighted = train_data_5y.withColumn(
    'actual_minutes', F.exp(F.col('DEP_DELAY')) - 1
).withColumn(
    'weight',
    F.when(F.col('actual_minutes') <= 0, 0.3)        # conservative for on-time
     .when(F.col('actual_minutes') <= 15, 0.5)       # conservative for small
     .when(F.col('actual_minutes') <= 30, 1.0)       # neutral
     .when(F.col('actual_minutes') <= 60, 3.0)       # aggressive for medium
     .when(F.col('actual_minutes') <= 120, 6.0)      # more aggressive
     .otherwise(10.0)                                 # most aggressive for large
)
'''
train_weighted = train_data_5y.withColumn(
    'actual_minutes', F.exp(F.col('DEP_DELAY')) - 1
).withColumn(
    'weight',
    F.when(F.col('actual_minutes') <= 60, 1.0)       # normal weight for most
     .when(F.col('actual_minutes') <= 120, 2.0)      # slight upweight
     .otherwise(4.0)                                  # upweight only extreme
)

# COMMAND ----------

train_weighted.display(limit=5)

# COMMAND ----------


# Balance data
train_df = undersample_majority_class(train_weighted, "DEP_DEL15", sampling_strategy=1.0, seed=42)

# Step 1: Fit preprocessing ONCE
print("Fitting preprocessing pipeline...")
preprocess_pipeline = preprocess_pipeline_func()
fitted_preprocessor = preprocess_pipeline.fit(train_df)

# Step 2: Transform training data ONCE
print("Transforming training data...")
train_preprocessed = fitted_preprocessor.transform(train_df)
test_preprocessed = fitted_preprocessor.transform(test_data_5y)


# Train with weights
model = GBTRegressor(
    featuresCol=FEATURES_COL,
    labelCol=LABEL_COL,
    weightCol='weight',
    maxIter=50,
    maxDepth=6
)

fitted_model = model.fit(train_preprocessed)

# Evaluate on test (no correction needed)
train_predictions = fitted_model.transform(train_preprocessed)
train_rmse, train_mae = cv_eval_in_minutes(train_predictions, 'prediction', 'DEP_DELAY')
test_predictions = fitted_model.transform(test_preprocessed)
test_rmse, test_mae = cv_eval_in_minutes(test_predictions, 'prediction', 'DEP_DELAY')
print(f"Train - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
print(f"Test - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}")

# COMMAND ----------

print("=== After (weighted model) ===")
analyze_by_delay_bins_detailed(test_predictions, 'prediction', 'DEP_DELAY').show()


# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Data
labels = ['On-time', '1-5', '6-10', '11-15', '16-20', '21-30', 
          '31-45', '46-60', '61-90', '91-120', '121-180', '>180']
actual = [0.0, 2.7, 7.4, 12.3, 17.9, 25.2, 37.4, 52.1, 73.3, 103.8, 145.1, 314.8]
predicted = [3.5, 6.4, 8.7, 12.3, 20.4, 24.9, 30.9, 36.8, 43.8, 51.8, 57.2, 65.8]

fig, ax = plt.subplots(figsize=(14, 7))

x = np.arange(len(labels))
width = 0.35

# Bars
bars1 = ax.bar(x - width/2, actual, width, label='Actual Delay', color='steelblue')
bars2 = ax.bar(x + width/2, predicted, width, label='Predicted Delay', color='coral')

# Perfect prediction line
ax.plot(x, actual, 'k--', alpha=0.3, linewidth=1)

# Highlight zones
ax.axvspan(-0.5, 2.5, alpha=0.1, color='lightgray', label='Overprediction Zone')
ax.axvspan(2.5, 5.5, alpha=0.1, color='lightgreen', label='Well-Calibrated Zone')
ax.axvspan(5.5, 11.5, alpha=0.1, color='gray', label='Underprediction Zone')

# Labels
ax.set_xlabel('Delay Bin (minutes)', fontsize=12)
ax.set_ylabel('Average Delay (minutes)', fontsize=12)
ax.set_title('Weighted Model Performance by Delay Severity:\nAccurate for Medium Delays, Fails on Extremes', 
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.legend(loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Add gap annotations for extreme cases
ax.annotate(f'Gap: -249 min', xy=(11, 65.8), xytext=(9.5, 150),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

ax.annotate(f'Gap: -88 min', xy=(10, 57.2), xytext=(8, 100),
            arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
            fontsize=10, color='red')

plt.tight_layout()
#plt.savefig('delay_vs_time_bins_single.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weighted Model Performance by Delay Magnitude
# MAGIC
# MAGIC ### Overall Pattern
# MAGIC
# MAGIC The model shows a **clear asymmetric error pattern**: it overpredicts small delays and severely underpredicts large delays.
# MAGIC
# MAGIC ### Performance Breakdown
# MAGIC
# MAGIC **Well-Calibrated Range (11-30 min delays):**
# MAGIC | Delay Bin | Actual | Predicted | Bias | Observation |
# MAGIC |-----------|--------|-----------|------|-------------|
# MAGIC | 11-15 min | 12.3 | 12.3 | 0.0 |  Perfect calibration |
# MAGIC | 16-20 min | 17.9 | 20.4 | +2.5 | Slight overprediction |
# MAGIC | 21-30 min | 25.2 | 24.9 | -0.2 |  Nearly perfect |
# MAGIC
# MAGIC **Overprediction Zone (0-10 min delays):**
# MAGIC | Delay Bin | Actual | Predicted | Bias | Issue |
# MAGIC |-----------|--------|-----------|------|-------|
# MAGIC | On-time | 0.0 | 3.5 | +3.5 | Predicts delay when none exists |
# MAGIC | 1-5 min | 2.7 | 6.4 | +3.7 | Overpredicts minor delays |
# MAGIC | 6-10 min | 7.4 | 8.7 | +1.3 | Slight overprediction |
# MAGIC
# MAGIC **Severe Underprediction Zone (31+ min delays):**
# MAGIC | Delay Bin | Actual | Predicted | Bias | Issue |
# MAGIC |-----------|--------|-----------|------|-------|
# MAGIC | 31-45 min | 37.4 | 30.9 | -6.5 | Begins underpredicting |
# MAGIC | 46-60 min | 52.1 | 36.8 | -15.3 | Significant gap |
# MAGIC | 61-90 min | 73.3 | 43.8 | -29.5 | Large underprediction |
# MAGIC | 91-120 min | 103.8 | 51.8 | -52.0 | Severe underprediction |
# MAGIC | 121-180 min | 145.1 | 57.2 | -87.8 | Critical gap |
# MAGIC | >180 min | 314.8 | 65.8 | -249.0 |  Catastrophic miss |
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC 1. Sweet spot exists: Model performs best for delays between 11-30 minutes (near-zero bias)
# MAGIC
# MAGIC 2. Regression to the mean: Model predictions cluster around 30-65 minutes regardless of actual delay severity
# MAGIC
# MAGIC 3. Extreme delays are problematic: For delays >3 hours, model only predicts ~66 minutes - missing by 4+ hours on average
# MAGIC
# MAGIC 4. Conservative predictions: Model appears to "cap" predictions around 65 minutes, unable to capture tail events
# MAGIC
# MAGIC 5. Practical impact:
# MAGIC    - **For passengers:** Minor delays ( less than 15 min) are not conidered as actual delay. So minimal impact
# MAGIC    - **For operations:** Cannot rely on model for severe delay scenarios
# MAGIC
# MAGIC ### Recommendation
# MAGIC
# MAGIC Consider a separate model or adjustment factor for severe delays (>45 min), or implement prediction intervals that widen for longer predicted delays.

# COMMAND ----------

weighted_error_dist = analyze_by_delay_bins_detailed(test_predictions, 'prediction', 'DEP_DELAY').toPandas()
plot_error_dist(weighted_error_dist)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Error analysis by Carrier
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

def analyze_by_carrier(df, prediction_col='pred_xgbr', label_col='DEP_DELAY_LOG'):
    """Analyze by original carrier name."""
    
    analysis = df.withColumn(
        'pred_minutes', F.exp(F.col(prediction_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col(label_col)) - 1
    ).withColumn(
        'error_minutes', F.col('pred_minutes') - F.col('actual_minutes')
    ).withColumn(
        'abs_error_minutes', F.abs(F.col('error_minutes'))
    )
    
    result = analysis.groupBy('OP_UNIQUE_CARRIER').agg(  # use original column
        F.count('*').alias('count'),
        F.round(F.avg('actual_minutes'), 1).alias('avg_actual'),
        F.round(F.avg('pred_minutes'), 1).alias('avg_pred'),
        F.round(F.avg('error_minutes'), 1).alias('bias'),
        F.round(F.avg('abs_error_minutes'), 1).alias('mae'),
        F.round(F.sqrt(F.avg(F.pow(F.col('error_minutes'), 2))), 1).alias('rmse')
    ).orderBy(F.desc('count'))
    
    return result

# Get carrier analysis with real names
carrier_analysis = analyze_by_carrier(analysis_df, 'pred_xgbr', 'DEP_DELAY_LOG')
carrier_analysis.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Convert to pandas
carrier_df = carrier_analysis.toPandas()

# Sort by count for better visualization
carrier_df = carrier_df.sort_values('count', ascending=True)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Actual vs Predicted by carrier
ax1 = axes[0, 0]
y_pos = range(len(carrier_df))
height = 0.35
ax1.barh([y - height/2 for y in y_pos], carrier_df['avg_actual'], height, label='Actual', color='steelblue')
ax1.barh([y + height/2 for y in y_pos], carrier_df['avg_pred'], height, label='Predicted', color='coral')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(carrier_df['OP_UNIQUE_CARRIER'])
ax1.set_xlabel('Average Delay (minutes)')
ax1.set_title('Actual vs Predicted Delay by Carrier')
ax1.legend()

# 2. Bias by carrier
ax2 = axes[0, 1]
colors = ['green' if b >= 0 else 'red' for b in carrier_df['bias']]
ax2.barh(y_pos, carrier_df['bias'], color=colors)
ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(carrier_df['OP_UNIQUE_CARRIER'])
ax2.set_xlabel('Bias (minutes)')
ax2.set_title('Prediction Bias by Carrier\n(Negative=Underpredict)')

# 3. MAE by carrier
ax3 = axes[1, 0]
ax3.barh(y_pos, carrier_df['mae'], color='purple', alpha=0.7)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(carrier_df['OP_UNIQUE_CARRIER'])
ax3.set_xlabel('MAE (minutes)')
ax3.set_title('Mean Absolute Error by Carrier')

# 4. Flight count by carrier
ax4 = axes[1, 1]
ax4.barh(y_pos, carrier_df['count'], color='teal', alpha=0.7)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(carrier_df['OP_UNIQUE_CARRIER'])
ax4.set_xlabel('Number of Flights')
ax4.set_title('Flight Count by Carrier')

plt.tight_layout()

plt.show()

# COMMAND ----------

fig, ax = plt.subplots(figsize=(12, 6))

# Sort by bias (worst underprediction first)
carrier_df_sorted = carrier_df.sort_values('bias')

x = range(len(carrier_df_sorted))
width = 0.35

ax.bar([i - width/2 for i in x], carrier_df_sorted['avg_actual'], width, label='Actual Delay', color='steelblue')
ax.bar([i + width/2 for i in x], carrier_df_sorted['avg_pred'], width, label='Predicted Delay', color='coral')

# Add bias annotation
for i, (_, row) in enumerate(carrier_df_sorted.iterrows()):
    ax.annotate(f"{row['bias']:.1f}", 
                xy=(i, max(row['avg_actual'], row['avg_pred']) + 1),
                ha='center', fontsize=9, color='red')
# Add after plotting bars
ax.set_ylim(0, carrier_df_sorted['avg_actual'].max() * 1.2)  # add 20% headroom

ax.set_xticks(x)
ax.set_xticklabels(carrier_df_sorted['OP_UNIQUE_CARRIER'], rotation=45, ha='right')
ax.set_ylabel('Average Delay (minutes)')
ax.set_title('Actual vs Predicted Delay by Carrier\n(Red numbers show bias)')
ax.legend()
plt.tight_layout()
#plt.savefig(f'{group_path}carrier_gap.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# Create summary with ranking
carrier_df_ranked = carrier_df.copy()
carrier_df_ranked['mae_rank'] = carrier_df_ranked['mae'].rank()
carrier_df_ranked['bias_rank'] = carrier_df_ranked['bias'].abs().rank()
carrier_df_ranked = carrier_df_ranked.sort_values('mae')

print("=== Carrier Performance Ranking ===")
print(carrier_df_ranked[['OP_UNIQUE_CARRIER', 'count', 'avg_actual', 'avg_pred', 'bias', 'mae', 'rmse']].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC #### Airport Analysis

# COMMAND ----------

# Airport analysis

airport_df = analyze_by_feature(analysis_df, 'ORIGIN').show(20)

# COMMAND ----------

#fig, axes = plt.subplots(1, 3, figsize=(18, 6))
#fig, axes = plt.subplots(1, 2, figsize=(12, 6))
fig, axes = plt.subplots(3, 1, figsize=(10, 15))
delay_df = error_dist
airport_df = analyze_by_feature(analysis_df, 'ORIGIN').toPandas().head(20)

# 1. Error by Delay Bin
ax1 = axes[0]
x = range(len(delay_df))
width = 0.35
ax1.bar([i - width/2 for i in x], delay_df['avg_actual'], width, label='Actual', color='steelblue')
ax1.bar([i + width/2 for i in x], delay_df['avg_pred'], width, label='Predicted', color='coral')
ax1.set_xticks(x)
ax1.set_xticklabels(delay_df['delay_bin'], rotation=45, ha='right')
ax1.set_ylabel('Minutes')
ax1.set_title('By Delay Magnitude')
ax1.legend()

# 2. By Airport
ax2 = axes[1]
airport_sorted = airport_df     #.sort_values(('count', ascending=True))
y = range(len(airport_sorted))
ax2.barh(y, airport_sorted['avg_actual_delay'], height=0.4, label='Actual', color='steelblue', alpha=0.7)
ax2.barh(y, airport_sorted['avg_pred_delay'], height=0.4, label='Predicted', color='coral', alpha=0.7)
ax2.set_yticks(y)
ax2.set_yticklabels(airport_sorted['ORIGIN'])
ax2.set_xlabel('Minutes')
ax2.set_title('By Airport (Top 15)')
ax2.legend()

# 3. By Carrier
ax3 = axes[2]
carrier_sorted = carrier_df.sort_values('bias')
y = range(len(carrier_sorted))
ax3.barh(y, carrier_sorted['avg_actual'], height=0.4, label='Actual', color='steelblue', alpha=0.7)
ax3.barh(y, carrier_sorted['avg_pred'], height=0.4, label='Predicted', color='coral', alpha=0.7)
ax3.set_yticks(y)
ax3.set_yticklabels(carrier_sorted['OP_UNIQUE_CARRIER'])
ax3.set_xlabel('Minutes')
ax3.set_title('By Carrier')
ax3.legend()



plt.suptitle('Flight Delay: Actual vs Predicted', fontsize=14, fontweight='bold')
plt.tight_layout()
#plt.savefig(f'{}/error_analysis_simple.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature Importance

# COMMAND ----------

# MAGIC %md
# MAGIC

# COMMAND ----------

for importance_df in importances:
    print(importance_df)    
    

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Assuming you have importance DataFrames for both models
# importance_df_xgbc for classifier
# importance_df_xgbr for regressor
importance_df_xgbr= importances[0]
importance_df_xgbc = importances[1]
top_20_xgbc = importance_df_xgbc.head(20).sort_values('normalized', ascending=True)
top_20_xgbr = importance_df_xgbr.head(20).sort_values('normalized', ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Classifier
axes[0].barh(top_20_xgbc['feature'], top_20_xgbc['normalized'] * 100, color='coral')
axes[0].set_xlabel('Importance (%)')
axes[0].set_title('XGBoost Classifier - Top 20 Features')

# Regressor
axes[1].barh(top_20_xgbr['feature'], top_20_xgbr['normalized'] * 100, color='steelblue')
axes[1].set_xlabel('Importance (%)')
axes[1].set_title('XGBoost Regressor - Top 20 Features')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Comparison: Classifier vs Regressor
# MAGIC
# MAGIC ### Key Observations
# MAGIC
# MAGIC **Dominant Feature:**
# MAGIC Both models agree that `prev_flight_dep_del15` (whether the previous flight was delayed) is the most important predictor, but with different magnitudes:
# MAGIC - **Classifier:** ~45% importance (heavily dominant)
# MAGIC - **Regressor:** ~32% importance (still dominant but less extreme)
# MAGIC
# MAGIC **Second Most Important:**
# MAGIC Both models rank `num_airport_wide_delays` as the second most important feature (~10% for both).
# MAGIC
# MAGIC ### Key Differences
# MAGIC
# MAGIC | Aspect | Classifier | Regressor |
# MAGIC |--------|------------|-----------|
# MAGIC | **Concentration** | Highly concentrated on 1 feature | More evenly distributed |
# MAGIC | **Top feature weight** | ~45% | ~32% |
# MAGIC | **Feature diversity** | Few features dominate | Multiple features contribute meaningfully |
# MAGIC
# MAGIC ### Interpretation
# MAGIC
# MAGIC 1. **Classifier focuses on binary signals:** The classifier heavily relies on whether the previous flight was delayed - a strong binary indicator for predicting *if* a delay will occur.
# MAGIC
# MAGIC 2. **Regressor uses more diverse inputs:** The regressor distributes importance across more features (prior_day_delay_rate, days_since_last_delay_route, rolling averages) because predicting *how long* a delay will be requires more nuanced information.
# MAGIC
# MAGIC 3. **Shared important features:** Both models value:
# MAGIC    - Previous flight delay status
# MAGIC    - Airport-wide delay counts
# MAGIC    - Rolling delay averages
# MAGIC    - Delay propagation metrics
# MAGIC
# MAGIC ### Implication for Ensemble
# MAGIC
# MAGIC The different feature emphasis suggests the classifier and regressor capture complementary information - supporting their use together in a two-stage prediction pipeline.

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
importance_df = importance_df_xgbr
top_20 = importance_df.head(20).sort_values('normalized', ascending=False)

plt.figure(figsize=(10, 8))
sns.set_style("whitegrid")

ax = sns.barplot(
    x='normalized', 
    y='feature', 
    data=top_20, 
    palette='Blues_r'
)

# Add value labels
for i, (val, feature) in enumerate(zip(top_20['normalized'], top_20['feature'])):
    ax.text(val + 0.001, i, f'{val*100:.1f}%', va='center', fontsize=9)

plt.xlabel('Importance (Normalized)')
plt.ylabel('Feature')
plt.title('Top 20 Feature Importances', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

top_20 = importance_df.head(20)
top_20['cumulative'] = top_20['normalized'].cumsum()

fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot
x = range(len(top_20))
ax1.bar(x, top_20['normalized'] * 100, color='steelblue', alpha=0.7, label='Individual')
ax1.set_xticks(x)
ax1.set_xticklabels(top_20['feature'], rotation=45, ha='right')
ax1.set_ylabel('Importance (%)')
ax1.set_xlabel('Feature')

# Cumulative line
ax2 = ax1.twinx()
ax2.plot(x, top_20['cumulative'] * 100, 'r-o', linewidth=2, label='Cumulative')
ax2.set_ylabel('Cumulative Importance (%)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Add annotation for total coverage
ax2.axhline(y=top_20['cumulative'].iloc[-1] * 100, color='red', linestyle='--', alpha=0.5)
ax2.text(len(top_20)-1, top_20['cumulative'].iloc[-1] * 100 + 2, 
         f"Top 20 = {top_20['cumulative'].iloc[-1]*100:.1f}%", fontsize=10, color='red')

plt.title('Top 20 Feature Importances with Cumulative Coverage', fontsize=14, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.12, 0.88))
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 20 Feature Importances Summary
# MAGIC
# MAGIC ### Key Finding
# MAGIC The top 20 features capture **77.7% of total model importance**, demonstrating that a relatively small subset of features drives most of the predictive power.
# MAGIC
# MAGIC ### Dominant Features
# MAGIC
# MAGIC **Top 3 features account for ~45% of importance:**
# MAGIC 1. **prev_flight_dep_del15** (~32%): Whether the previous flight was delayed is by far the strongest predictor - indicating delay propagation is the primary driver
# MAGIC 2. **num_airport_wide_delays** (~8%): Current airport congestion level
# MAGIC 3. **prior_day_delay_rate** (~5%): Historical delay patterns from the previous day
# MAGIC
# MAGIC ### Feature Categories
# MAGIC
# MAGIC **Delay History Features (dominant):**
# MAGIC - Previous flight delay status
# MAGIC - Prior day delay rate
# MAGIC - Days since last delay on route
# MAGIC - Rolling delay averages
# MAGIC
# MAGIC **Airport/Operational Features:**
# MAGIC - Airport-wide delay counts
# MAGIC - Delay propagation score
# MAGIC - Rolling 30-day volume
# MAGIC - Origin betweenness (network centrality)
# MAGIC
# MAGIC **Temporal Features:**
# MAGIC - Hours since previous flight
# MAGIC - Departure time (sin/cos encoded)
# MAGIC - Arrival time (cos)
# MAGIC - Day of week interactions
# MAGIC
# MAGIC **Route/Carrier Features:**
# MAGIC - Destination encoded
# MAGIC - Route delay rate (30-day)
# MAGIC - Carrier-weighted rolling averages
# MAGIC
# MAGIC ### Cumulative Curve Interpretation
# MAGIC
# MAGIC The steep initial rise followed by a flattening curve shows:
# MAGIC - **First 5 features:** Capture ~55% of importance
# MAGIC - **First 10 features:** Capture ~68% of importance  
# MAGIC - **First 20 features:** Capture ~78% of importance
# MAGIC
# MAGIC This indicates potential for **dimensionality reduction** - a model using only the top 20-40 features may perform nearly as well as one using all features, with faster training and reduced overfitting risk.

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

top_20 = importance_df.head(20).sort_values('normalized', ascending=True)

# Define feature categories (adjust based on your features)
def get_category(feature):
    if 'rolling' in feature.lower() or 'delay' in feature.lower():
        return 'Delay Metrics'
    elif 'airport' in feature.lower() or 'origin' in feature.lower() or 'dest' in feature.lower():
        return 'Airport'
    elif 'carrier' in feature.lower() or 'airline' in feature.lower():
        return 'Carrier'
    elif 'time' in feature.lower() or 'hour' in feature.lower() or 'day' in feature.lower():
        return 'Time'
    elif 'flight' in feature.lower() or 'prev' in feature.lower():
        return 'Flight Info'
    else:
        return 'Other'

top_20['category'] = top_20['feature'].apply(get_category)

# Color mapping
color_map = {
    'Delay Metrics': '#e74c3c',
    'Airport': '#3498db',
    'Carrier': '#2ecc71',
    'Time': '#9b59b6',
    'Flight Info': '#f39c12',
    'Other': '#95a5a6'
}

colors = [color_map[cat] for cat in top_20['category']]

plt.figure(figsize=(12, 8))
bars = plt.barh(top_20['feature'], top_20['normalized'] * 100, color=colors)
plt.xlabel('Importance (%)')
plt.title('Top 20 Feature Importances by Category')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, label=cat) for cat, color in color_map.items()]
plt.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
plt.show()

# COMMAND ----------

def find_worst_segments(df, feature_col, min_count=1000, prediction_col='pred_xgbr', label_col='DEP_DELAY_LOG'):
    """Find segments where model performs worst."""
    
    result = analyze_by_feature(df, feature_col, prediction_col, label_col)
    
    # Filter for minimum sample size and sort by MAE
    worst = result.filter(F.col('count') >= min_count).orderBy(F.desc('mae'))
    
    return worst

# Find airports where model struggles most
print("=== Worst Performing Airports ===")
find_worst_segments(test_results_comparison_df, 'ORIGIN', min_count=1000).show(10)

# Find carriers where model struggles most
print("=== Worst Performing Carriers ===")
find_worst_segments(test_results_comparison_df, 'OP_UNIQUE_CARRIER', min_count=1000).show(10)

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create Worst Performing Airports DataFrame
airports_data = {
    'ORIGIN': ['EGE', 'ACK', 'ASE', 'MQT', 'HYS', 'ACV', 'HDN', 'SWF', 'DLH', 'PLN'],
    'count': [2221, 1154, 6027, 1363, 1186, 1897, 1050, 1725, 3231, 1047],
    'avg_actual_delay': [31.25, 32.73, 29.11, 29.76, 29.67, 28.91, 23.87, 27.30, 24.50, 25.51],
    'avg_pred_delay': [13.41, 16.32, 13.90, 12.15, 11.61, 12.40, 14.95, 14.89, 9.50, 12.49],
    'mae': [25.13, 23.96, 23.06, 22.80, 21.69, 20.92, 19.83, 19.51, 18.73, 18.71],
    'rmse': [110.79, 84.04, 85.76, 99.52, 103.53, 79.66, 93.26, 65.95, 80.15, 78.41],
    'bias': [-17.83, -16.40, -15.21, -17.61, -18.06, -16.50, -8.92, -12.41, -15.00, -13.02]
}
airports_df = pd.DataFrame(airports_data)

# Create Worst Performing Carriers DataFrame
carriers_data = {
    'carrier': ['EV', 'B6', 'F9', 'UA', 'YV', 'OO', 'AA', 'NK', '9E', 'OH'],
    'count': [128175, 289228, 132383, 618119, 220232, 812828, 924121, 200162, 252344, 281637],
    'avg_actual_delay': [21.55, 21.67, 18.80, 16.37, 17.39, 16.32, 14.81, 14.16, 14.25, 14.32],
    'avg_pred_delay': [10.22, 12.94, 12.15, 10.03, 8.96, 8.55, 8.78, 8.83, 8.33, 8.73],
    'mae': [18.83, 18.01, 16.01, 15.39, 15.34, 14.06, 13.60, 12.98, 12.21, 12.15],
    'rmse': [70.19, 49.84, 39.10, 48.57, 55.23, 57.02, 46.43, 41.62, 46.22, 36.75],
    'bias': [-11.32, -8.73, -6.65, -6.34, -8.43, -7.77, -6.03, -5.33, -5.92, -5.59]
}
carriers_df = pd.DataFrame(carriers_data)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ============================================
# Plot 1: Worst Airports - Actual vs Predicted
# ============================================
ax1 = axes[0, 0]
airports_sorted = airports_df.sort_values('mae', ascending=True)
y = np.arange(len(airports_sorted))
height = 0.35

ax1.barh(y - height/2, airports_sorted['avg_actual_delay'], height, label='Actual', color='steelblue')
ax1.barh(y + height/2, airports_sorted['avg_pred_delay'], height, label='Predicted', color='coral')
ax1.set_yticks(y)
ax1.set_yticklabels(airports_sorted['ORIGIN'])
ax1.set_xlabel('Avg Delay (minutes)')
ax1.set_title('Worst 10 Airports: Actual vs Predicted Delay', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='x', alpha=0.3)

# ============================================
# Plot 2: Worst Airports - RMSE
# ============================================
ax2 = axes[0, 1]
airports_sorted_rmse = airports_df.sort_values('rmse', ascending=True)
y = np.arange(len(airports_sorted_rmse))

colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(airports_sorted_rmse)))
bars = ax2.barh(y, airports_sorted_rmse['rmse'], color=colors)
ax2.set_yticks(y)
ax2.set_yticklabels(airports_sorted_rmse['ORIGIN'])
ax2.set_xlabel('RMSE (minutes)')
ax2.set_title('Worst 10 Airports by RMSE', fontsize=12, fontweight='bold')
ax2.axvline(x=airports_sorted_rmse['rmse'].mean(), color='black', linestyle='--', 
            label=f"Avg: {airports_sorted_rmse['rmse'].mean():.1f}")
ax2.legend()
ax2.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, airports_sorted_rmse['rmse']):
    ax2.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', fontsize=9)

# ============================================
# Plot 3: Worst Carriers - Actual vs Predicted
# ============================================
ax3 = axes[1, 0]
carriers_sorted = carriers_df.sort_values('mae', ascending=True)
y = np.arange(len(carriers_sorted))
height = 0.35

ax3.barh(y - height/2, carriers_sorted['avg_actual_delay'], height, label='Actual', color='steelblue')
ax3.barh(y + height/2, carriers_sorted['avg_pred_delay'], height, label='Predicted', color='coral')
ax3.set_yticks(y)
ax3.set_yticklabels(carriers_sorted['carrier'])
ax3.set_xlabel('Avg Delay (minutes)')
ax3.set_title('Worst 10 Carriers: Actual vs Predicted Delay', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# ============================================
# Plot 4: Worst Carriers - Bias (Underprediction)
# ============================================
ax4 = axes[1, 1]
carriers_sorted_bias = carriers_df.sort_values('bias', ascending=True)
y = np.arange(len(carriers_sorted_bias))

colors = ['#e74c3c' if b < -8 else '#f39c12' if b < -6 else '#f1c40f' for b in carriers_sorted_bias['bias']]
bars = ax4.barh(y, carriers_sorted_bias['bias'], color=colors)
ax4.axvline(x=0, color='black', linestyle='-', linewidth=1)
ax4.set_yticks(y)
ax4.set_yticklabels(carriers_sorted_bias['carrier'])
ax4.set_xlabel('Bias (minutes)')
ax4.set_title('Worst 10 Carriers by Bias\n(Negative = Underprediction)', fontsize=12, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, carriers_sorted_bias['bias']):
    ax4.text(val - 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}', va='center', ha='right', fontsize=9)

plt.tight_layout()
#plt.savefig('worst_performers_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# ============================================
# Plot 1: Airports - MAE vs RMSE with Bubble Size = Count
# ============================================
ax1 = axes[0]
airports_data = {
    'ORIGIN': ['EGE', 'ACK', 'ASE', 'MQT', 'HYS', 'ACV', 'HDN', 'SWF', 'DLH', 'PLN'],
    'count': [2221, 1154, 6027, 1363, 1186, 1897, 1050, 1725, 3231, 1047],
    'mae': [25.13, 23.96, 23.06, 22.80, 21.69, 20.92, 19.83, 19.51, 18.73, 18.71],
    'rmse': [110.79, 84.04, 85.76, 99.52, 103.53, 79.66, 93.26, 65.95, 80.15, 78.41],
    'bias': [-17.83, -16.40, -15.21, -17.61, -18.06, -16.50, -8.92, -12.41, -15.00, -13.02]
}
airports_df = pd.DataFrame(airports_data)

scatter = ax1.scatter(airports_df['mae'], airports_df['rmse'], 
                      s=airports_df['count']/20, 
                      c=airports_df['bias'], cmap='RdYlGn',
                      alpha=0.7, edgecolors='black', linewidth=1)

for i, row in airports_df.iterrows():
    ax1.annotate(row['ORIGIN'], (row['mae'] + 0.3, row['rmse'] + 1), fontsize=9)

ax1.set_xlabel('MAE (minutes)', fontsize=11)
ax1.set_ylabel('RMSE (minutes)', fontsize=11)
ax1.set_title('Worst 10 Airports\n(Bubble size = flight count, Color = bias)', fontsize=12, fontweight='bold')
plt.colorbar(scatter, ax=ax1, label='Bias (minutes)')
ax1.grid(alpha=0.3)

# ============================================
# Plot 2: Carriers - MAE vs RMSE with Bubble Size = Count
# ============================================
ax2 = axes[1]
carriers_data = {
    'carrier': ['EV', 'B6', 'F9', 'UA', 'YV', 'OO', 'AA', 'NK', '9E', 'OH'],
    'count': [128175, 289228, 132383, 618119, 220232, 812828, 924121, 200162, 252344, 281637],
    'mae': [18.83, 18.01, 16.01, 15.39, 15.34, 14.06, 13.60, 12.98, 12.21, 12.15],
    'rmse': [70.19, 49.84, 39.10, 48.57, 55.23, 57.02, 46.43, 41.62, 46.22, 36.75],
    'bias': [-11.32, -8.73, -6.65, -6.34, -8.43, -7.77, -6.03, -5.33, -5.92, -5.59]
}
carriers_df = pd.DataFrame(carriers_data)

scatter2 = ax2.scatter(carriers_df['mae'], carriers_df['rmse'], 
                       s=carriers_df['count']/3000, 
                       c=carriers_df['bias'], cmap='RdYlGn',
                       alpha=0.7, edgecolors='black', linewidth=1)

for i, row in carriers_df.iterrows():
    ax2.annotate(row['carrier'], (row['mae'] + 0.2, row['rmse'] + 0.5), fontsize=10, fontweight='bold')

ax2.set_xlabel('MAE (minutes)', fontsize=11)
ax2.set_ylabel('RMSE (minutes)', fontsize=11)
ax2.set_title('Worst 10 Carriers\n(Bubble size = flight count, Color = bias)', fontsize=12, fontweight='bold')
plt.colorbar(scatter2, ax=ax2, label='Bias (minutes)')
ax2.grid(alpha=0.3)

plt.tight_layout()
#plt.savefig('worst_performers_bubble.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Key Insights Summary
# MAGIC
# MAGIC ### Worst Airports
# MAGIC
# MAGIC | Airport | Avg Actual | Avg Predicted | Gap | RMSE | Issue |
# MAGIC |---------|-----------|---------------|-----|------|-------|
# MAGIC | **EGE** (Eagle, CO) | 31.2 | 13.4 | -17.8 | **110.8** | Highest RMSE - mountain airport |
# MAGIC | **ACK** (Nantucket) | 32.7 | 16.3 | -16.4 | 84.0 | Small island airport |
# MAGIC | **ASE** (Aspen) | 29.1 | 13.9 | -15.2 | 85.8 | Mountain weather |
# MAGIC | **HYS** (Hays, KS) | 29.7 | 11.6 | -18.1 | 103.5 | Severe underprediction |
# MAGIC
# MAGIC ### Worst Carriers
# MAGIC
# MAGIC | Carrier | Avg Actual | Avg Predicted | Gap | RMSE | Issue |
# MAGIC |---------|-----------|---------------|-----|------|-------|
# MAGIC | **EV** (ExpressJet) | 21.5 | 10.2 | -11.3 | **70.2** | Highest RMSE - regional |
# MAGIC | **B6** (JetBlue) | 21.7 | 12.9 | -8.7 | 49.8 | High delays, moderate error |
# MAGIC | **OO** (SkyWest) | 16.3 | 8.6 | -7.8 | 57.0 | Regional carrier |
# MAGIC | **YV** (Mesa) | 17.4 | 9.0 | -8.4 | 55.2 | Regional carrier |
# MAGIC
# MAGIC **Pattern:** Regional carriers and small/mountain airports are hardest to predict - likely due to weather sensitivity and fewer training samples.

# COMMAND ----------

def analyze_by_multiple_features(df, features, prediction_col='pred_xgbr', label_col='DEP_DELAY_LOG'):
    """Analyze by combination of features."""
    
    analysis = df.withColumn(
        'pred_minutes', F.exp(F.col(prediction_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col(label_col)) - 1
    ).withColumn(
        'error_minutes', F.col('pred_minutes') - F.col('actual_minutes')
    ).withColumn(
        'abs_error_minutes', F.abs(F.col('error_minutes'))
    )
    
    result = analysis.groupBy(features).agg(
        F.count('*').alias('count'),
        F.avg('actual_minutes').alias('avg_actual_delay'),
        F.avg('pred_minutes').alias('avg_pred_delay'),
        F.avg('abs_error_minutes').alias('mae'),
        F.sqrt(F.avg(F.pow(F.col('error_minutes'), 2))).alias('rmse')
    ).orderBy(F.desc('count'))
    
    return result

# Weekend + Holiday combination
print("=== Weekend x Holiday ===")
analyze_by_multiple_features(test_results_comparison_df, ['is_weekend', 'is_holiday_month']).show()

# Airport + Carrier
print("=== Top Routes (Origin + Carrier) ===")
analyze_by_multiple_features(test_results_comparison_df, ['ORIGIN', 'OP_UNIQUE_CARRIER']).filter(F.col('count') > 500).show(20)

# COMMAND ----------

import matplotlib.pyplot as plt
import numpy as np

# Top 10 routes
routes = ['ATL-DL', 'DFW-AA', 'CLT-AA', 'CLT-OH', 'ORD-UA', 'MDW-WN', 'DEN-WN', 'LAS-WN', 'SEA-AS', 'MSP-DL']
actual = [9.49, 16.85, 14.51, 11.82, 17.69, 15.23, 14.34, 12.39, 9.70, 10.45]
predicted = [6.83, 12.34, 10.02, 9.13, 13.97, 11.97, 11.93, 9.58, 5.87, 6.37]

fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(routes))
width = 0.35

bars1 = ax.bar(x - width/2, actual, width, label='Actual', color='steelblue')
bars2 = ax.bar(x + width/2, predicted, width, label='Predicted', color='coral')

ax.set_xlabel('Route (Origin-Carrier)')
ax.set_ylabel('Avg Delay (minutes)')
ax.set_title('Top 10 Routes: Actual vs Predicted Delay', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(routes, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Add gap annotation
for i, (a, p) in enumerate(zip(actual, predicted)):
    gap = a - p
    ax.annotate(f'-{gap:.1f}', xy=(i, max(a, p) + 0.5), ha='center', fontsize=8, color='gray')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Top 10 Routes: Actual vs Predicted Delay Summary
# MAGIC
# MAGIC ### Overall Pattern
# MAGIC
# MAGIC The model **consistently underpredicts delays across all top 10 routes**, with gaps ranging from -2.4 to -4.5 minutes.
# MAGIC
# MAGIC ### Route-by-Route Insights
# MAGIC
# MAGIC **Highest Actual Delays:**
# MAGIC - **ORD-UA (United at O'Hare):** ~17.7 min actual, largest delay among top routes
# MAGIC - **DFW-AA (American at Dallas):** ~16.9 min actual, second highest
# MAGIC - **MDW-WN (Southwest at Midway):** ~15.2 min actual
# MAGIC
# MAGIC **Lowest Actual Delays:**
# MAGIC - **ATL-DL (Delta at Atlanta):** ~9.5 min actual, efficient hub operation
# MAGIC - **SEA-AS (Alaska at Seattle):** ~9.5 min actual, well-run hub
# MAGIC - **MSP-DL (Delta at Minneapolis):** ~10.5 min actual
# MAGIC
# MAGIC ### Prediction Gap Analysis
# MAGIC
# MAGIC | Gap Size | Routes | Observation |
# MAGIC |----------|--------|-------------|
# MAGIC | **Largest (-4.5)** | DFW-AA, CLT-AA | American Airlines hubs most underpredicted |
# MAGIC | **Medium (-3.3 to -4.1)** | ORD-UA, MDW-WN, SEA-AS, MSP-DL | Mixed carriers |
# MAGIC | **Smallest (-2.4 to -2.8)** | DEN-WN, CLT-OH, ATL-DL, LAS-WN | Best calibrated routes |
# MAGIC
# MAGIC ### Key Takeaways
# MAGIC
# MAGIC 1. **Southwest (WN) routes are better calibrated:** DEN-WN (-2.4), LAS-WN (-2.8) have smallest gaps
# MAGIC 2. **American Airlines (AA) routes have largest errors:** DFW-AA and CLT-AA both at -4.5 minutes
# MAGIC 3. **High-volume Delta hubs perform well:** ATL-DL has low delay and small gap (-2.7)
# MAGIC 4. **Systematic underprediction:** No route shows overprediction - model bias is consistent

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Reshape for heatmap
heatmap_data = pd.DataFrame({
    'Weekday': [14.56, 13.02],
    'Weekend': [14.11, 11.96]
}, index=['Non-Holiday', 'Holiday'])

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Actual Delay Heatmap
actual_data = pd.DataFrame({
    'Weekday': [14.56, 13.02],
    'Weekend': [14.11, 11.96]
}, index=['Non-Holiday', 'Holiday'])

sns.heatmap(actual_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Minutes'})
axes[0].set_title('Avg Actual Delay', fontsize=12, fontweight='bold')

# Predicted Delay Heatmap
pred_data = pd.DataFrame({
    'Weekday': [8.92, 7.72],
    'Weekend': [8.26, 6.99]
}, index=['Non-Holiday', 'Holiday'])

sns.heatmap(pred_data, annot=True, fmt='.1f', cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Minutes'})
axes[1].set_title('Avg Predicted Delay', fontsize=12, fontweight='bold')

# RMSE Heatmap
rmse_data = pd.DataFrame({
    'Weekday': [42.82, 41.69],
    'Weekend': [44.27, 41.84]
}, index=['Non-Holiday', 'Holiday'])

sns.heatmap(rmse_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=axes[2], cbar_kws={'label': 'Minutes'})
axes[2].set_title('RMSE', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Create Weekend x Holiday DataFrame
weekend_holiday_data = {
    'is_weekend': [0, 1, 0, 1],
    'is_holiday_month': [0, 0, 1, 1],
    'count': [4034304, 1445096, 1304606, 475001],
    'avg_actual_delay': [14.56, 14.11, 13.02, 11.96],
    'avg_pred_delay': [8.92, 8.26, 7.72, 6.99],
    'mae': [12.68, 12.50, 11.44, 10.75],
    'rmse': [42.82, 44.27, 41.69, 41.84]
}
weekend_df = pd.DataFrame(weekend_holiday_data)

# Create labels
weekend_df['label'] = weekend_df.apply(
    lambda x: f"{'Weekend' if x['is_weekend'] else 'Weekday'}\n{'Holiday' if x['is_holiday_month'] else 'Non-Holiday'}", 
    axis=1
)

# Create Top Routes DataFrame
routes_data = {
    'route': ['ATL-DL', 'DFW-AA', 'CLT-AA', 'CLT-OH', 'ORD-UA', 'MDW-WN', 'DEN-WN', 'LAS-WN', 'SEA-AS', 'MSP-DL',
              'BWI-WN', 'DEN-UA', 'DAL-WN', 'ORD-MQ', 'ORD-AA', 'ORD-OO', 'PHX-WN', 'IAH-UA', 'DTW-DL', 'DFW-MQ'],
    'count': [243229, 149773, 98028, 93550, 76651, 75120, 69513, 69469, 68374, 68313,
              68125, 67765, 66096, 65996, 63785, 61285, 60017, 59717, 59546, 57293],
    'avg_actual_delay': [9.49, 16.85, 14.51, 11.82, 17.69, 15.23, 14.34, 12.39, 9.70, 10.45,
                         13.02, 15.31, 13.29, 11.51, 17.08, 20.72, 12.94, 15.58, 10.67, 7.88],
    'avg_pred_delay': [6.83, 12.34, 10.02, 9.13, 13.97, 11.97, 11.93, 9.58, 5.87, 6.37,
                       9.42, 10.73, 9.80, 10.32, 12.71, 14.81, 10.63, 11.28, 6.09, 7.28],
    'mae': [8.95, 15.03, 12.92, 11.22, 16.02, 11.56, 11.22, 9.46, 9.41, 9.98,
            10.43, 14.20, 9.87, 10.99, 15.02, 18.49, 10.10, 15.23, 10.43, 8.76],
    'rmse': [28.13, 35.54, 35.15, 28.55, 42.15, 25.28, 24.11, 21.11, 22.83, 36.84,
             24.71, 40.68, 23.13, 30.19, 38.67, 54.65, 22.70, 38.93, 37.26, 24.26]
}
routes_df = pd.DataFrame(routes_data)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ============================================
# Plot 1: Weekend x Holiday - Actual vs Predicted
# ============================================
ax1 = axes[0, 0]
x = np.arange(len(weekend_df))
width = 0.35

ax1.bar(x - width/2, weekend_df['avg_actual_delay'], width, label='Actual', color='steelblue')
ax1.bar(x + width/2, weekend_df['avg_pred_delay'], width, label='Predicted', color='coral')
ax1.set_xticks(x)
ax1.set_xticklabels(weekend_df['label'])
ax1.set_ylabel('Avg Delay (minutes)')
ax1.set_title('Weekend × Holiday: Actual vs Predicted Delay', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# ============================================
# Plot 2: Weekend x Holiday - Error Metrics
# ============================================
ax2 = axes[0, 1]
x = np.arange(len(weekend_df))
width = 0.35

ax2.bar(x - width/2, weekend_df['mae'], width, label='MAE', color='#2ecc71')
ax2.bar(x + width/2, weekend_df['rmse'], width, label='RMSE', color='#e74c3c')
ax2.set_xticks(x)
ax2.set_xticklabels(weekend_df['label'])
ax2.set_ylabel('Error (minutes)')
ax2.set_title('Weekend × Holiday: Error Metrics', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# ============================================
# Plot 3: Top Routes - Actual vs Predicted (Top 15)
# ============================================
ax3 = axes[1, 0]
top_15 = routes_df.head(15).sort_values('avg_actual_delay', ascending=True)
y = np.arange(len(top_15))
height = 0.35

ax3.barh(y - height/2, top_15['avg_actual_delay'], height, label='Actual', color='steelblue')
ax3.barh(y + height/2, top_15['avg_pred_delay'], height, label='Predicted', color='coral')
ax3.set_yticks(y)
ax3.set_yticklabels(top_15['route'])
ax3.set_xlabel('Avg Delay (minutes)')
ax3.set_title('Top 15 Routes: Actual vs Predicted Delay', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(axis='x', alpha=0.3)

# ============================================
# Plot 4: Top Routes - RMSE (sorted)
# ============================================
ax4 = axes[1, 1]
routes_sorted = routes_df.sort_values('rmse', ascending=True)
y = np.arange(len(routes_sorted))

colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(routes_sorted)))
ax4.barh(y, routes_sorted['rmse'], color=colors)
ax4.set_yticks(y)
ax4.set_yticklabels(routes_sorted['route'])
ax4.set_xlabel('RMSE (minutes)')
ax4.set_title('Top Routes by RMSE (sorted)', fontsize=12, fontweight='bold')
ax4.axvline(x=routes_sorted['rmse'].mean(), color='red', linestyle='--', label=f"Avg: {routes_sorted['rmse'].mean():.1f}")
ax4.legend()
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()

#plt.savefig(f'{local_path}weekend_holiday_routes_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Analysis Summary
# MAGIC
# MAGIC ### Weekend × Holiday Analysis
# MAGIC
# MAGIC **Key Finding:** Holiday periods reduce both actual delays and prediction errors.
# MAGIC
# MAGIC | Segment | Actual Delay | Predicted | Key Insight |
# MAGIC |---------|-------------|-----------|-------------|
# MAGIC | Weekday, Non-Holiday | 14.6 min | 8.9 min | Highest delays, baseline performance |
# MAGIC | Weekend, Non-Holiday | 14.1 min | 8.3 min | Similar to weekday |
# MAGIC | Weekday, Holiday | 13.0 min | 7.7 min | Lower delays, better predictions |
# MAGIC | Weekend + Holiday | 12.0 min | 7.0 min | Lowest delays, best predictions |
# MAGIC
# MAGIC **Observations:**
# MAGIC - **Holiday effect is stronger than weekend effect:** Holiday months reduce actual delays by ~1.5-2 minutes regardless of weekend status
# MAGIC - **Model consistently underpredicts:** All segments show 5-6 minute underprediction gap
# MAGIC - **RMSE is relatively stable:** Ranges from 41.7 to 44.3 across all segments
# MAGIC - **Best performance:** Weekend + Holiday combination (lowest MAE of 10.7 minutes)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### Top Routes (Origin + Carrier) Analysis
# MAGIC
# MAGIC **Best Performing Routes:**
# MAGIC - **LAS-WN (Southwest at Las Vegas):** Lowest RMSE (21.1), MAE of 9.5 minutes
# MAGIC - **DEN-WN (Southwest at Denver):** RMSE of 24.1, well-calibrated predictions
# MAGIC - **PHX-WN (Southwest at Phoenix):** RMSE of 22.7, consistent performance
# MAGIC
# MAGIC **Worst Performing Routes:**
# MAGIC - **ORD-OO (SkyWest at O'Hare):** Highest RMSE (54.6), highest actual delay (20.7 min)
# MAGIC - **ORD-UA (United at O'Hare):** High RMSE (42.2), high actual delay (17.7 min)
# MAGIC - **DEN-UA (United at Denver):** RMSE of 40.7, significant underprediction
# MAGIC
# MAGIC **Key Patterns:**
# MAGIC
# MAGIC 1. **Southwest (WN) routes perform best:** Consistently lower RMSE across LAS, DEN, PHX, MDW, DAL, BWI - likely due to point-to-point operations and predictable patterns
# MAGIC
# MAGIC 2. **O'Hare (ORD) is challenging:** Multiple carriers (OO, UA, AA) show high errors at ORD - hub complexity and weather issues
# MAGIC
# MAGIC 3. **Regional carriers struggle:** SkyWest (OO) at ORD has the worst performance (RMSE 54.6) - regional operations are harder to predict
# MAGIC
# MAGIC 4. **Delta hubs perform well:** ATL-DL has the highest volume (243K) with reasonable RMSE (28.1) - efficient hub operation
# MAGIC
# MAGIC 5. **Consistent underprediction:** All routes show predicted delays 3-6 minutes below actual - systematic bias in the model