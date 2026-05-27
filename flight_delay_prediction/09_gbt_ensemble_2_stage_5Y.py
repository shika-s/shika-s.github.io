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
# MAGIC ### One-Hot Encoded Features (33)
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



def extract_feature_importance( model_name, importances):
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
    expanded_features = numerical_features + categorical_feature_encoded
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

from pyspark.ml.regression import DecisionTreeRegressor,  GBTRegressor, RandomForestRegressor
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.regression import DecisionTreeRegressor



def parameter_sets(param_grid):
    # return parameter names and parameter sets in param_grid.
    parameter_names = [param.name for param in param_grid[0]]
    parameter_values = [p.values() for p in param_grid]
    return parameter_names, parameter_values


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
          featuresCol=FEATURE_COL,
          labelCol=LABEL_COL,
          maxDepth=params['maxDepth'],
          maxBins=params['maxBins'],
          minInstancesPerNode=params['minInstancesPerNode'],
          minInfoGain=params['minInfoGain']
      )



    preprocess_pipeline = preprocess_pipeline_func()     #preprocessing_pipeline_func()
    stages = preprocess_pipeline.getStages()
    stages.append(model)
    pipeline = Pipeline(stages=stages)
    return pipeline

def cv_eval(preds):
  """
  Input: transformed df with prediction and label
  Output: desired score 
  """
   
  #rdd_preds = preds.select(['prediction', LABEL_COL]).rdd
  evaluator_rmse = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="rmse")
  evaluator_mae = RegressionEvaluator(labelCol=LABEL_COL, predictionCol="prediction", metricName="mae")
      
  rmse = np.round(evaluator_rmse.evaluate(preds),4)
  mae = np.round(evaluator_mae.evaluate(preds),4)
  return [rmse, mae]

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
      
      # If TimeseriesSplit 
    
      train_df = df.filter(f.col('row_id') <= chunk_size * (i+1)).cache()
     
      # Create dev set
      dev_df = df.filter((f.col('row_id') > chunk_size * (i+1))&(f.col('row_id') <= chunk_size * (i+2))).cache()  

      # Apply sampling on train if selected
      if sampling == 'under':
        train_df = undersample_majority_class(train_df, balance_col, sampling_strategy=0.5, seed=42)
        train_df = train_df.cache()
    
        
      #print info on train and dev set for this fold
          
      # Fit params on the model
      model = pipeline.fit(train_df)
      dev_pred = model.transform(dev_df)
    
      score = cv_eval(dev_pred)[0]
      
      scores.append(score)
      print(f'    Number of training datapoints for fold number {i+1} is {train_df.count():,} with a {metric} score of {score:.4f}') 
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
            featuresCol=feature_col,
            labelCol=r_label_col,
            predictionCol='pred_xgbr',
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
            featuresCol=feature_col,
            labelCol=b_label_col,
            predictionCol='pred_xgbc',
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
    assembler = VectorAssembler(inputCols=transformed_features, outputCol="features_unscaled", handleInvalid="keep")
    stages.append(assembler)

    # Create standard scaler
    scaler = StandardScaler(inputCol="features_unscaled", outputCol="features_scaled", withStd=True, withMean=False)
    stages.append(scaler)

    # Create the preprocessing pipeline
    preprocess_pipeline = Pipeline(stages=stages)

    return preprocess_pipeline

   

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

from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier


from pyspark.ml.functions import vector_to_array
from pyspark.sql.functions import col
from pyspark.ml.feature import (
    StringIndexer,
    OneHotEncoder,
    VectorAssembler,
    StandardScaler
)
from pyspark.ml import Pipeline


def train_ensemble(train_df, feature_col=FEATURES_COL,balance_col="DEP_DEL15", b_label_col="DEP_DEL15", r_label_col="DEP_DELAY_LOG"):
    """Train multiple models and return them"""
    
    model_names = ['gbtr', 'gbtc'  ]                      
    

    # Balance data
    train_df = undersample_majority_class(train_df, balance_col, sampling_strategy=1.0, seed=42)
    
    # Step 1: Fit preprocessing ONCE
    print("Fitting preprocessing pipeline...")
    preprocess_pipeline = preprocess_pipeline_func()
    fitted_preprocessor = preprocess_pipeline.fit(train_df)
    
    # Step 2: Transform training data ONCE
    print("Transforming training data...")
    train_preprocessed = fitted_preprocessor.transform(train_df)
    train_preprocessed = train_preprocessed.cache()
    
    # Step 3: Train each model on preprocessed data
    fitted_models = {'preprocessor': fitted_preprocessor}  # save preprocessor
    
    for name in model_names:
        print(f"Training {name}...")
        model = get_configured_model(name, r_label_col, b_label_col, feature_col)
        fitted_models[name] = model.fit(train_preprocessed)
        print(f"Finished Training {name}...")
    
    return fitted_models


def ensemble_predict(df, fitted_models):      
    """Combine classification probability with regression prediction
    Final = P(delayed) * regression_prediction"""
    
    # Step 1: Preprocess ONCE
    print("Preprocessing data...")
    df_preprocessed = fitted_models['preprocessor'].transform(df)
    
    # Step 2: Apply regression model
    print("Getting regression prediction...")
    df_with_reg = fitted_models['gbtr'].transform(df_preprocessed)
    
     # Step 3: Apply classification model
    print("Getting classification prediction...")
    df_with_both = fitted_models['gbtc'].transform(df_with_reg)
    
    # Extract probability of class 1 (delayed)
    # Note: GBTClassifier doesn't have probability - use rawPrediction or switch to RFC
    print("Extracting probability of class 1")
    df_with_prob = df_with_both.withColumn(
        'raw_score',
        vector_to_array('rawPrediction')[1]
    )
   

    threshold = 0.25
    # Test different strategies
    results_comparison = df_with_prob.withColumn(
        'pred_threshold', 
        F.when(F.col('raw_score') > threshold, F.col('pred_gbtr')).otherwise(0)
    ).withColumn(
        'pred_r_only',
        F.col('pred_gbtr')
    ).withColumn(
        'pred_multiply',
        F.col('raw_score') * F.col('pred_gbtr')
    )
   
    return results_comparison

# COMMAND ----------

# Train
fitted_models = train_ensemble(train_data_5y)

# Predict
train_results_comparison_df = ensemble_predict(train_data_5y, fitted_models)
test_results_comparison_df = ensemble_predict(test_data_5y, fitted_models)

# Evaluate each
for idx,comparison_df in enumerate([train_results_comparison_df, test_results_comparison_df]):
    
    for pred_col in ['pred_threshold', 'pred_r_only', 'pred_multiply']:
        rmse, mae = cv_eval_in_minutes(comparison_df, prediction_col=pred_col, label_col="DEP_DELAY")
        print(f"{idx}: {pred_col}: RMSE={rmse:.2f}, MAE={mae:.2f}")


#results.select('ensemble_pred', 'DEP_DELAY', 'prob_delayed', 'pred_rf').show()

# COMMAND ----------

# MAGIC %md
# MAGIC     Result from GBTClassifier and REgressor
# MAGIC         Original: On-time=19,570,544, Delayed=4,299,340
# MAGIC     Balanced: On-time=8,598,338, Delayed=4,299,340
# MAGIC Fitting preprocessing pipeline...
# MAGIC Transforming training data...
# MAGIC Training gbtr...
# MAGIC Finished Training gbtr...
# MAGIC Training gbtc...
# MAGIC Finished Training gbtc...
# MAGIC Preprocessing data...
# MAGIC Getting regression prediction...
# MAGIC Getting classification prediction...
# MAGIC Extracting probability of class 1
# MAGIC Preprocessing data...
# MAGIC Getting regression prediction...
# MAGIC Getting classification prediction...
# MAGIC Extracting probability of class 1
# MAGIC 0: pred_threshold: RMSE=38.48, MAE=10.65
# MAGIC 0: pred_r_only: RMSE=37.88, MAE=11.07
# MAGIC 0: pred_multiply: RMSE=9253.56, MAE=68.63
# MAGIC 1: pred_threshold: RMSE=46.21, MAE=12.19
# MAGIC 1: pred_r_only: RMSE=45.58, MAE=12.57
# MAGIC 1: pred_multiply: RMSE=3302.75, MAE=40.34
# MAGIC
# MAGIC
# MAGIC     
# MAGIC     
# MAGIC     Result from Random Forest Classifier and GBTRegressor
# MAGIC     
# MAGIC     Original: On-time=19,570,544, Delayed=4,299,340
# MAGIC     Balanced: On-time=8,598,092, Delayed=4,299,340
# MAGIC Fitting preprocessing pipeline...
# MAGIC Transforming training data...
# MAGIC Training gbtr...
# MAGIC Finished Training gbtr...
# MAGIC Training rfc...
# MAGIC Finished Training rfc...
# MAGIC Preprocessing data...
# MAGIC Getting regression prediction...
# MAGIC Getting classification prediction...
# MAGIC Extracting probability of class 1
# MAGIC Preprocessing data...
# MAGIC Getting regression prediction...
# MAGIC Getting classification prediction...
# MAGIC Extracting probability of class 1
# MAGIC 0: pred_threshold: RMSE=37.74, MAE=10.74
# MAGIC 0: pred_r_only: RMSE=37.50, MAE=10.96
# MAGIC 0: pred_multiply: RMSE=40.69, MAE=11.73
# MAGIC 1: pred_threshold: RMSE=45.38, MAE=12.24
# MAGIC 1: pred_r_only: RMSE=45.16, MAE=12.43
# MAGIC 1: pred_multiply: RMSE=48.24, MAE=13.33

# COMMAND ----------

 for pred_col in ['pred_threshold', 'pred_r_only', 'pred_multiply']:
        percentiles =   calculate_error_distribution(comparison_df, prediction_col=pred_col, label_col="DEP_DELAY")
        print(f" {pred_col}: {percentiles['median_error']:.2f}, {percentiles['p90_error']:.2f}, {percentiles['p99_error']:.2f}")
      

# COMMAND ----------

# Evaluate each
for pred_col in ['pred_threshold', 'pred_rf_only', 'pred_multiply']:
    rmse, mae = cv_eval_in_minutes(results__comparison_df, prediction_col=pred_col, label_col="DEP_DELAY")
    print(f"{pred_col}: RMSE={rmse:.2f}, MAE={mae:.2f}")

# COMMAND ----------

rmse, mae = cv_eval_in_minutes(results, prediction_col="ensemble_pred", label_col="DEP_DELAY")
print(f"RMSE: {rmse:.4f} minutes")
print(f"MAE: {mae:.4f} minutes")

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Ensemble Specs
# MAGIC
# MAGIC |Model| Parameters|
# MAGIC |--|--|
# MAGIC |GBTRegressor|maxIter=40,maxDepth=5,stepSize=0.1, subsamplingRate=0.8|
# MAGIC |GBTClassifier|maxIter=40,maxDepth=5,stepSize=0, subsamplingRate=0.8|
# MAGIC  

# COMMAND ----------

# MAGIC %md
# MAGIC
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
# MAGIC | 6 | 3 | GBTClassifier | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Threshold-Gated | [38.48, 10.65] | [46.21, **12.19**] |
# MAGIC | 7 | 3 | GBTClassifier | GBTRegressor | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [37.88, 11.07] | [45.58, 12.57] |
# MAGIC | 8 | 3 | GBTClassifier | GBTRegressor (weighted) | 2015-2018 | 2019 | Undersample (0.5) | Regression only | [48.11, 17.97] | [**42.99**, 12.89] |
# MAGIC
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
# MAGIC | 1| 8 | GBTClassifier + GBTRegressor (weighted) | Regression only | 42.99 | 12.89 |
# MAGIC
# MAGIC ###### Best Performers (Test MAE)
# MAGIC
# MAGIC | Rank | Exp # | Model Combination | Strategy | Test RMSE | Test MAE |
# MAGIC |:----:|:-----:|-------------------|----------|-----------|----------|
# MAGIC | 1 | 6 | GBTClassifier + GBTRegressor | Threshold-Gated | 46.21 | **12.19** |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC
# MAGIC

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

def analyze_by_feature(df, feature_col, prediction_col='pred_threshold', label_col='DEP_DELAY'):
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


analysis_df = test_results_comparison_df

# By airport
print("=== Performance by Origin Airport ===")
analyze_by_feature(analysis_df, 'ORIGIN_encoded').show(20)

# By weekend
print("=== Performance by Weekend ===")
#analyze_by_feature(analysis_df, 'is_weekend').show()

# By holiday
print("=== Performance by Holiday ===")
analyze_by_feature(analysis_df, 'is_superbowl_week').show()

# By carrier
print("=== Performance by Carrier ===")
analyze_by_feature(analysis_df, 'OP_UNIQUE_CARRIER_encoded').show(20)

# By hour of day
print("=== Performance by Hour ===")
#analyze_by_feature(analysis_df, 'DEP_HOUR').orderBy('DEP_HOUR').show(24)

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

# Check bias for different strategies
for pred_col in ['pred_gbtr', 'pred_threshold']:
    bias_check = analysis_df.withColumn(
        'pred_minutes', F.exp(F.col(pred_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col('DEP_DELAY')) - 1
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

def analyze_by_delay_bins(df, prediction_col='pred_gbtr', label_col='DEP_DELAY'):
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
analyze_by_delay_bins(analysis_df, 'pred_gbtr', 'DEP_DELAY').show()

# COMMAND ----------

def analyze_by_delay_bins_detailed(df, prediction_col='pred_gbtr', label_col='DEP_DELAY'):
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
analyze_by_delay_bins_detailed(analysis_df, 'pred_gbtr', 'DEP_DELAY').show(15, truncate=False)


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

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Get the data as pandas
error_dist = analyze_by_delay_bins_detailed(analysis_df, 'pred_gbtr', 'DEP_DELAY').toPandas()

# Create figure with multiple subplots

fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(error_dist['delay_bin'], error_dist['avg_actual'], 'o-', label='Actual', linewidth=2, markersize=8)
ax.plot(error_dist['delay_bin'], error_dist['avg_pred'], 's-', label='Predicted', linewidth=2, markersize=8)

ax.fill_between(error_dist['delay_bin'], error_dist['avg_actual'], error_dist['avg_pred'], 
                alpha=0.3, color='red', label='Underprediction Gap')

ax.set_xlabel('Delay Bin')
ax.set_ylabel('Minutes')
ax.set_title('Model Underprediction Increases with Delay Magnitude')
ax.legend()
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Experiment with weighted learning

# COMMAND ----------


# Add weights based on actual delay in TRAINING data only

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
train_df = undersample_majority_class(train_weighted, "DEP_DEL15", sampling_strategy=0.5, seed=42)

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

weighted_error_dist = analyze_by_delay_bins_detailed(test_predictions, 'prediction', 'DEP_DELAY').toPandas()
plot_error_dist(weighted_error_dist)

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

def plot_error_over_time(df, time_col, prediction_col='prediction', label_col='DEP_DELAY'):
    """Plot error metrics over time."""
    
    # Get data
    time_df = analyze_by_time(df, time_col, prediction_col, label_col).toPandas()
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Actual vs Predicted over time
    ax1 = axes[0, 0]
    ax1.plot(time_df[time_col], time_df['avg_actual'], 'o-', label='Actual', color='steelblue')
    ax1.plot(time_df[time_col], time_df['avg_pred'], 's-', label='Predicted', color='coral')
    ax1.fill_between(time_df[time_col], time_df['avg_actual'], time_df['avg_pred'], 
                     alpha=0.3, color='red')
    ax1.set_xlabel(time_col)
    ax1.set_ylabel('Average Delay (minutes)')
    ax1.set_title('Actual vs Predicted Over Time')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Bias over time
    ax2 = axes[0, 1]
    colors = ['green' if b >= 0 else 'red' for b in time_df['bias']]
    ax2.bar(range(len(time_df)), time_df['bias'], color=colors)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xticks(range(len(time_df)))
    ax2.set_xticklabels(time_df[time_col], rotation=45, ha='right')
    ax2.set_ylabel('Bias (minutes)')
    ax2.set_title('Prediction Bias Over Time')
    
    # 3. MAE over time
    ax3 = axes[1, 0]
    ax3.plot(time_df[time_col], time_df['mae'], 'o-', color='purple', linewidth=2)
    ax3.fill_between(time_df[time_col], 0, time_df['mae'], alpha=0.3, color='purple')
    ax3.set_xlabel(time_col)
    ax3.set_ylabel('MAE (minutes)')
    ax3.set_title('Mean Absolute Error Over Time')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. RMSE over time
    ax4 = axes[1, 1]
    ax4.plot(time_df[time_col], time_df['rmse'], 'o-', color='darkred', linewidth=2)
    ax4.fill_between(time_df[time_col], 0, time_df['rmse'], alpha=0.3, color='darkred')
    ax4.set_xlabel(time_col)
    ax4.set_ylabel('RMSE (minutes)')
    ax4.set_title('Root Mean Squared Error Over Time')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.suptitle(f'Error Distribution by {time_col}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    #plt.savefig(f'/error_by_{time_col}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return time_df

resuts_df = test_predictions
# Plot by different time dimensions
results_with_time = add_time_features(results_df, 'FL_DATE')

# By month
month_df = plot_error_over_time(results_with_time, 'month')

# By year-month (for trend analysis)
ym_df = plot_error_over_time(results_with_time, 'year_month')

# By hour
#hour_df = plot_error_over_time(results_with_time, 'DEP_HOUR')

# COMMAND ----------

# MAGIC %md
# MAGIC #### Stratified Metrics

# COMMAND ----------

from pyspark.sql import functions as F

def stratified_metrics(df, prediction_col='prediction', label_col='DEP_DELAY'):
    """Calculate metrics separately for on-time vs delayed flights."""
    
    analysis = df.withColumn(
        'pred_minutes', F.exp(F.col(prediction_col)) - 1
    ).withColumn(
        'actual_minutes', F.exp(F.col(label_col)) - 1
    ).withColumn(
        'error_minutes', F.col('pred_minutes') - F.col('actual_minutes')
    ).withColumn(
        'abs_error_minutes', F.abs(F.col('error_minutes'))
    ).withColumn(
        'flight_status',
        F.when(F.col('actual_minutes') <= 0, 'On-time')
         .when(F.col('actual_minutes') <= 15, 'Minor Delay')
         .when(F.col('actual_minutes') <= 60, 'Moderate Delay')
         .otherwise('Severe Delay')
    )
    
    result = analysis.groupBy('flight_status').agg(
        F.count('*').alias('count'),
        F.round(F.count('*') / df.count() * 100, 1).alias('pct'),
        F.round(F.avg('actual_minutes'), 1).alias('avg_actual'),
        F.round(F.avg('pred_minutes'), 1).alias('avg_pred'),
        F.round(F.avg('error_minutes'), 1).alias('bias'),
        F.round(F.avg('abs_error_minutes'), 1).alias('mae'),
        F.round(F.sqrt(F.avg(F.pow(F.col('error_minutes'), 2))), 1).alias('rmse')
    ).orderBy('avg_actual')
    
    return result

# Run stratified analysis
stratified = stratified_metrics(test_predictions)
stratified.show()


# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Create data
data = {
    'Flight Status': ['On-time', 'Minor Delay (1-15 min)', 'Moderate Delay (16-60 min)', 'Severe Delay (>60 min)'],
    'Count': ['4,753,278', '1,198,095', '821,982', '485,652'],
    '%': ['65.5%', '16.5%', '11.3%', '6.7%'],
    'Avg Actual': ['0.0', '6.3', '31.8', '140.1'],
    'Avg Pred': ['3.5', '8.5', '27.7', '52.4'],
    'Bias': ['+3.5', '+2.2', '-4.1', '-87.7'],
    'MAE': ['3.5', '8.2', '23.2', '98.7'],
    'RMSE': ['9.0', '15.1', '30.4', '157.1']
}

df = pd.DataFrame(data)

fig, ax = plt.subplots(figsize=(14, 4))
ax.axis('off')

# Create table
table = ax.table(
    cellText=df.values,
    colLabels=df.columns,
    loc='center',
    cellLoc='center'
)

# Style
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

# Header style
for j in range(len(df.columns)):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

# Highlight severe delay row (worst performance)
for j in range(len(df.columns)):
    table[(4, j)].set_facecolor('#ffc7ce')

# Highlight on-time row (best performance)  
for j in range(len(df.columns)):
    table[(1, j)].set_facecolor('#c6efce')

plt.title('Model Performance by Flight Status\n(All values in minutes except Count and %)', 
          fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
#plt.savefig('/flight_status_performance.png', dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### Error analysis by Carrier
# MAGIC

# COMMAND ----------

from pyspark.sql import functions as F

def analyze_by_carrier(df, prediction_col='pred_gbtr', label_col='DEP_DELAY'):
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
carrier_analysis = analyze_by_carrier(analysis_df, 'pred_gbtr', 'DEP_DELAY')
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

# Airport analysis

airport_df = analyze_by_feature(analysis_df, 'ORIGIN_encoded').show(20)

# COMMAND ----------

#fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
delay_df = error_dist
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
''' 
# 2. By Airport
ax2 = axes[1]
airport_sorted = airport_df.sort_values('bias')
y = range(len(airport_sorted))
ax2.barh(y, airport_sorted['avg_actual'], height=0.4, label='Actual', color='steelblue', alpha=0.7)
ax2.barh(y, airport_sorted['avg_pred'], height=0.4, label='Predicted', color='coral', alpha=0.7)
ax2.set_yticks(y)
ax2.set_yticklabels(airport_sorted['ORIGIN'])
ax2.set_xlabel('Minutes')
ax2.set_title('By Airport (Top 15)')
ax2.legend()
'''
# 3. By Carrier
ax3 = axes[1]
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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import functions as F

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

# Get feature names from your VectorAssembler
# This should match the order you used in VectorAssembler
feature_names = numerical_features + categorical_features   #[f"{feat}_encoded" for feat in categorical_features]


# Extract importance
importance_df = get_feature_importance(fitted_model, feature_names)
#print(importance_df)
# display top 20 features by importance
top_n = 30
top_features = importance_df.head(top_n)
top_features

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ┌──────┬─────────────────────────────────────┬────────────┐
# MAGIC │ Rank │ Feature                             │ Importance │
# MAGIC ├──────┼─────────────────────────────────────┼────────────┤
# MAGIC │  1   │ Hours Since Previous Flight         │   18.0%    │ ◄─┐
# MAGIC │  2   │ Number of Airport-Wide Delays       │   17.2%    │   │ Top 3 = 52%
# MAGIC │  3   │ Previous Flight Delayed (15+ min)   │   16.6%    │ ◄─┘
# MAGIC ├──────┼─────────────────────────────────────┼────────────┤
# MAGIC │  4   │ Prior Day Delay Rate                │    6.8%    │
# MAGIC │  5   │ Rolling Avg (Origin × DOW)          │    6.5%    │
# MAGIC │  6   │ Previous Flight Elapsed Time        │    5.9%    │
# MAGIC │  7   │ Prior Flights Today                 │    3.1%    │
# MAGIC │  8   │ Days Since Route Delay              │    2.4%    │
# MAGIC │  9   │ Destination                         │    2.1%    │
# MAGIC │ 10   │ Rolling Avg (Origin × Carrier)      │    2.1%    │
# MAGIC ├──────┼─────────────────────────────────────┼────────────┤
# MAGIC │ 11   │ Rolling Origin Delays 24h           │    2.1%    │
# MAGIC │ 12   │ Rolling 30-Day Volume               │    2.0%    │
# MAGIC │ 13   │ Carrier × Hour                      │    1.4%    │
# MAGIC │ 14   │ Delay Propagation Score             │    1.4%    │
# MAGIC │ 15   │ Rolling Delay Ratio 24h             │    1.1%    │
# MAGIC │ 16   │ Rolling Origin Flights 24h          │    1.0%    │
# MAGIC │ 17   │ Arrival Time (Cosine)               │    0.9%    │
# MAGIC │ 18   │ Origin                              │    0.8%    │
# MAGIC │ 19   │ Days Since Carrier Delay at Origin  │    0.8%    │
# MAGIC │ 20   │ Route Delay Rate 30d                │    0.7%    │
# MAGIC ├──────┼─────────────────────────────────────┼────────────┤
# MAGIC │ 21   │ Dest Degree Centrality              │    0.6%    │
# MAGIC │ 22   │ Turnaround Category                 │    0.6%    │
# MAGIC │ 23   │ Airline Reputation Score            │    0.6%    │
# MAGIC │ 24   │ Origin Degree Centrality            │    0.5%    │
# MAGIC │ 25   │ Dest State                          │    0.4%    │
# MAGIC │ 26   │ Dest Betweenness                    │    0.4%    │
# MAGIC │ 27   │ Rolling Avg (Origin)                │    0.3%    │
# MAGIC │ 28   │ Carrier                             │    0.3%    │
# MAGIC │ 29   │ Carrier × Dest                      │    0.3%    │
# MAGIC │ 30   │ Origin PageRank                     │    0.3%    │
# MAGIC └──────┴─────────────────────────────────────┴────────────┘
# MAGIC
# MAGIC Summary:
# MAGIC • Top 3 features: 52% of importance
# MAGIC • Top 10 features: 80% of importance  
# MAGIC • Graph network features (centrality, pagerank): ~2.5%
# MAGIC ```
# MAGIC
# MAGIC **Key Insights:**
# MAGIC
# MAGIC | Category | Features | Total Importance |
# MAGIC |----------|----------|------------------|
# MAGIC | Flight History | Hours since prev, prev delayed, elapsed time, propagation | ~42% |
# MAGIC | Airport Conditions | Airport delays, rolling delays, delay ratio | ~21% |
# MAGIC | Historical Patterns | Prior day rate, rolling avgs, route delay rate | ~18% |
# MAGIC | Location/Network | Origin, dest, centrality, pagerank | ~8% |
# MAGIC | Time | Carrier×hour, arrival time | ~2% |

# COMMAND ----------

# MAGIC %md
# MAGIC feature	importance
# MAGIC 1	hours_since_prev_flight	0.180372
# MAGIC 60	num_airport_wide_delays	0.172468
# MAGIC 18	prev_flight_dep_del15	0.166137
# MAGIC 44	prior_day_delay_rate	0.067735
# MAGIC 11	dep_delay15_24h_rolling_avg_by_origin_dayofwee...	0.065122
# MAGIC 26	prev_flight_crs_elapsed_time	0.058984
# MAGIC 21	prior_flights_today	0.031308
# MAGIC 39	days_since_last_delay_route	0.024311
# MAGIC 55	DEST_encoded	0.021018
# MAGIC 12	dep_delay15_24h_rolling_avg_by_origin_carrier_...	0.020867
# MAGIC 49	rolling_origin_num_delays_24h	0.020622
# MAGIC 38	rolling_30day_volume	0.019884
# MAGIC 24	carrier_encoded_x_hour	0.014278
# MAGIC 46	delay_propagation_score	0.013800
# MAGIC 3	rolling_origin_delay_ratio_24h_high_corr	0.010636
# MAGIC 51	rolling_origin_num_flights_24h_high_corr	0.010274
# MAGIC 19	arr_time_cos	0.009498
# MAGIC 10	ORIGIN_encoded	0.008339
# MAGIC 33	days_since_carrier_last_delay_at_origin	0.007712
# MAGIC 13	route_delay_rate_30d	0.006592

# COMMAND ----------

import matplotlib.pyplot as plt

# Data
features = [
    'Hours Since Prev Flight',
    'Num Airport-Wide Delays',
    'Prev Flight Delayed',
    'Prior Day Delay Rate',
    'Rolling Avg (Origin×DOW)',
    'Prev Flight Elapsed Time',
    'Prior Flights Today',
    'Days Since Route Delay',
    'Destination',
    'Rolling Avg (Origin×Carrier)',
    'Rolling Origin Delays 24h',
    'Rolling 30-Day Volume',
    'Carrier × Hour',
    'Delay Propagation Score',
    'Rolling Delay Ratio 24h'
]

importance = [0.1804, 0.1725, 0.1661, 0.0677, 0.0651, 0.0590, 0.0313, 
              0.0243, 0.0210, 0.0209, 0.0206, 0.0199, 0.0143, 0.0138, 0.0106]

# Reverse for horizontal bar chart
features = features[::-1]
importance = importance[::-1]

fig, ax = plt.subplots(figsize=(10, 8))

# Color top 3 differently
colors = ['steelblue'] * 12 + ['#2e8b57'] * 3

bars = ax.barh(features, importance, color=colors[::-1])

# Add value labels
for i, (bar, val) in enumerate(zip(bars, importance)):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
            f'{val:.1%}', va='center', fontsize=9)

ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Feature Importance - Top 15 Features', fontsize=14, fontweight='bold')
ax.set_xlim(0, 0.22)

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='#2e8b57', label='Top 3 (52% total)'),
                   Patch(facecolor='steelblue', label='Others')]
ax.legend(handles=legend_elements, loc='lower right')

plt.tight_layout()
#plt.savefig('/feature_importance_chart.png', dpi=150, bbox_inches='tight',facecolor='white', edgecolor='none')
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Group features by category
categories = {
    'Flight History': [
        ('Hours Since Prev Flight', 0.1804),
        ('Prev Flight Delayed', 0.1661),
        ('Prev Flight Elapsed Time', 0.0590),
        ('Prior Flights Today', 0.0313),
        ('Delay Propagation Score', 0.0138)
    ],
    'Airport Conditions': [
        ('Num Airport-Wide Delays', 0.1725),
        ('Rolling Origin Delays 24h', 0.0206),
        ('Rolling Origin Delay Ratio', 0.0106),
        ('Rolling Origin Flights 24h', 0.0103)
    ],
    'Historical Patterns': [
        ('Prior Day Delay Rate', 0.0677),
        ('Rolling Avg (Origin×DOW)', 0.0651),
        ('Rolling Avg (Origin×Carrier)', 0.0209),
        ('Days Since Route Delay', 0.0243),
        ('Route Delay Rate 30d', 0.0066)
    ],
    'Location/Time': [
        ('Destination', 0.0210),
        ('Carrier × Hour', 0.0143),
        ('Arrival Time', 0.0095),
        ('Origin', 0.0083)
    ]
}

# Sum by category
category_sums = {cat: sum([x[1] for x in feats]) for cat, feats in categories.items()}

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: By category (pie chart)
ax1 = axes[0]
colors = ['#2e8b57', '#4472C4', '#ed7d31', '#7030a0']
wedges, texts, autotexts = ax1.pie(
    category_sums.values(), 
    labels=category_sums.keys(),
    autopct='%1.1f%%',
    colors=colors,
    explode=[0.05, 0.05, 0.05, 0.05]
)
ax1.set_title('Feature Importance by Category', fontsize=12, fontweight='bold')

# Right: Top 10 individual features
ax2 = axes[1]
top_features = [
    'Hours Since Prev Flight',
    'Num Airport-Wide Delays', 
    'Prev Flight Delayed',
    'Prior Day Delay Rate',
    'Rolling Avg (Origin×DOW)',
    'Prev Flight Elapsed Time',
    'Prior Flights Today',
    'Days Since Route Delay',
    'Destination',
    'Rolling Avg (Origin×Carrier)'
]
top_importance = [0.1804, 0.1725, 0.1661, 0.0677, 0.0651, 0.0590, 0.0313, 0.0243, 0.0210, 0.0209]

# Reverse for horizontal bars
top_features = top_features[::-1]
top_importance = top_importance[::-1]

bars = ax2.barh(top_features, top_importance, color='steelblue')

# Highlight top 3
for bar in list(bars)[-3:]:
    bar.set_color('#2e8b57')

# Add value labels
for bar, val in zip(bars, top_importance):
    ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2, 
             f'{val:.1%}', va='center', fontsize=9)

ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Individual Features', fontsize=12, fontweight='bold')
ax2.set_xlim(0, 0.22)

plt.suptitle('GBT Regressor Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
#plt.savefig('/feature_importance_panel.png', dpi=150, bbox_inches='tight',facecolor='white', edgecolor='none')
plt.show()


# COMMAND ----------

# MAGIC %md
# MAGIC │ Rank │ Feature                                     │ Importance │
# MAGIC ├──────┼─────────────────────────────────────────────┼────────────┤
# MAGIC │  1   │ Hours Since Previous Flight                 │   18.0%    │
# MAGIC │  2   │ Number of Airport-Wide Delays               │   17.2%    │
# MAGIC │  3   │ Previous Flight Delayed (15+ min)           │   16.6%    │
# MAGIC ├──────┼─────────────────────────────────────────────┼────────────┤
# MAGIC │  4   │ Prior Day Delay Rate                        │    6.8%    │
# MAGIC │  5   │ Dep Delay Rolling Avg (Origin × DOW)        │    6.5%    │
# MAGIC │  6   │ Previous Flight Elapsed Time                │    5.9%    │
# MAGIC │  7   │ Prior Flights Today                         │    3.1%    │
# MAGIC │  8   │ Days Since Last Delay (Route)               │    2.4%    │
# MAGIC │  9   │ Destination                                 │    2.1%    │
# MAGIC │ 10   │ Dep Delay Rolling Avg (Origin × Carrier)    │    2.1%    │
# MAGIC └──────┴─────────────────────────────────────────────┴────────────┘
# MAGIC
# MAGIC Top 3 features account for 52% of total importance

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

# Split into two columns
features_1_15 = [
    ('1', 'Hours Since Prev Flight', '18.0%'),
    ('2', 'Num Airport-Wide Delays', '17.2%'),
    ('3', 'Prev Flight Delayed', '16.6%'),
    ('4', 'Prior Day Delay Rate', '6.8%'),
    ('5', 'Rolling Avg (Origin×DOW)', '6.5%'),
    ('6', 'Prev Flight Elapsed Time', '5.9%'),
    ('7', 'Prior Flights Today', '3.1%'),
    ('8', 'Days Since Route Delay', '2.4%'),
    ('9', 'Destination', '2.1%'),
    ('10', 'Rolling Avg (Origin×Carrier)', '2.1%'),
    ('11', 'Rolling Origin Delays 24h', '2.1%'),
    ('12', 'Rolling 30-Day Volume', '2.0%'),
    ('13', 'Carrier × Hour', '1.4%'),
    ('14', 'Delay Propagation Score', '1.4%'),
    ('15', 'Rolling Delay Ratio 24h', '1.1%'),
]

features_16_30 = [
    ('16', 'Rolling Origin Flights 24h', '1.0%'),
    ('17', 'Arrival Time (Cosine)', '0.9%'),
    ('18', 'Origin', '0.8%'),
    ('19', 'Days Since Carrier Delay', '0.8%'),
    ('20', 'Route Delay Rate 30d', '0.7%'),
    ('21', 'Dest Degree Centrality', '0.6%'),
    ('22', 'Turnaround Category', '0.6%'),
    ('23', 'Airline Reputation Score', '0.6%'),
    ('24', 'Origin Degree Centrality', '0.5%'),
    ('25', 'Dest State', '0.4%'),
    ('26', 'Dest Betweenness', '0.4%'),
    ('27', 'Rolling Avg (Origin)', '0.3%'),
    ('28', 'Carrier', '0.3%'),
    ('29', 'Carrier × Dest', '0.3%'),
    ('30', 'Origin PageRank', '0.3%'),
]

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

for ax, data, title in zip(axes, [features_1_15, features_16_30], ['Rank 1-15', 'Rank 16-30']):
    ax.axis('off')
    
    df = pd.DataFrame(data, columns=['Rank', 'Feature', 'Importance'])
    
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center',
        cellLoc='center',
        colWidths=[0.12, 0.65, 0.23]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.6)
    
    # Header
    for j in range(3):
        table[(0, j)].set_facecolor('#4472C4')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    
    # Top 3 highlight (first table only)
    if title == 'Rank 1-15':
        for i in range(1, 4):
            for j in range(3):
                table[(i, j)].set_facecolor('#c6efce')
    
    # Left-align features
    for i in range(1, len(df) + 1):
        table[(i, 1)].set_text_props(ha='left')
    
    ax.set_title(title, fontsize=12, fontweight='bold')

plt.suptitle('Feature Importance - Top 30 Features', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
#plt.savefig('/feature_importance_two_col.png', dpi=150, bbox_inches='tight',facecolor='white', edgecolor='none')
plt.show()
