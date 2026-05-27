# Databricks notebook source
# MAGIC %md
# MAGIC # Team 4_4_Graph_Feature

# COMMAND ----------

# Setup
!pip install networkx

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# imports
from graphframes import GraphFrame
from pyspark.sql.functions import col
import re
import heapq
import itertools
import numpy as np
import networkx as nx # 
from collections import defaultdict, deque
import matplotlib.pyplot as plt
%matplotlib inline

# COMMAND ----------


data_BASE_DIR = "dbfs:/mnt/mids-w261/"
# Team folder
section = "4"
number = "4"
folder_path = f"dbfs:/student-groups/Group_{section}_{number}"
data_path_12M = f"{folder_path}/data_12M/"


df_1Y_features = spark.read.parquet(f"{data_path_12M}/df_joined_1Y_2015_features.parquet/")

print(f"Our feature dataset has {df_1Y_features.count()} rows and {len(df_1Y_features.columns)} columns.") 

# print the first 5 rows
display(df_1Y_features.limit(5))


# COMMAND ----------

# MAGIC %md
# MAGIC We will design our graph database with airports as the nodes and the flights as the edges.  The number of flights originating from an airport will be its attribute. The number of flights reaching an airport will also be its attributes. 

# COMMAND ----------

edges = (
    df_1Y_features
    .groupBy("ORIGIN", "DEST")
    .count()
    .withColumnRenamed("ORIGIN", "src")
    .withColumnRenamed("DEST", "dst")
    .withColumnRenamed("count", "weight")
)


# count of number of edges
print(f"There are {edges.count()} edges in our graph.")

#display edges sorted by distance
display(edges.orderBy(col("weight").desc()), limit=10)

# COMMAND ----------

vertices = df_1Y_features.select("ORIGIN").distinct().withColumnRenamed("ORIGIN", "id").union(df_1Y_features.select("DEST").distinct().withColumnRenamed("DEST", "id"))

# count of number of nodes
print(f"There are {vertices.count()} nodes in our graph.")

display(vertices, limit=10)


# COMMAND ----------



## Let's display the graph vertices - GCP Solution, for Databricks, just display(g.vertices)
g = GraphFrame(vertices, edges)
display(g.vertices.limit(10).toPandas())

## Let's display the graph edges sorted in descending order by weight
display(g.edges.orderBy(col("weight").desc()).limit(10).toPandas())

## Let's display inDegrees sorted in descending order
display(g.inDegrees.orderBy(col("inDegree").desc()).limit(10).toPandas())


## Let's display outDegrees sorted in descending order
display(g.outDegrees.orderBy(col("outDegree").desc()).limit(10).toPandas())



# COMMAND ----------

# Calculate pagerank for each node
pr = g.pageRank(resetProbability=0.15, maxIter=10)
display(pr.vertices.orderBy(col("pagerank").desc()).limit(10).toPandas())
# Calculate degree centrality for each node
dc = g.degrees
display(dc.orderBy(col("degree").desc()).limit(10).toPandas())


# COMMAND ----------

# add pagerank to the dataframe df_1Y_features
df_with_pagerank = df_1Y_features.join(pr.vertices, df_1Y_features.ORIGIN == pr.vertices.id, 'left').drop('id')
# add degree centrality to the dataframe df_1Y_features
df_pr_dc = df_with_pagerank.join(dc, df_1Y_features.ORIGIN == dc.id, 'left').drop('id')






# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC #### Betweeness

# COMMAND ----------

edges = (
    df_1Y_features
    .groupBy("ORIGIN", "DEST")
    .count()
    .withColumnRenamed("ORIGIN", "src")
    .withColumnRenamed("DEST", "dst")
    .withColumnRenamed("distance", "weight")
)


# count of number of edges
print(f"There are {edges.count()} edges in our graph.")

nodes = df_1Y_features.select("ORIGIN").distinct().withColumnRenamed("ORIGIN", "id").union(df_1Y_features.select("DEST").distinct().withColumnRenamed("DEST", "id"))



# COMMAND ----------

import networkx as nx
import pandas as pd
import pyspark.sql.functions as F

# Convert PySpark DataFrame to Pandas (collect to driver)
edges_pd = df_1Y_features.groupBy("ORIGIN", "DEST").agg(
    F.avg("DISTANCE").alias("distance"),
    F.count("*").alias("num_flights")
).toPandas()

# Create directed graph
G = nx.DiGraph()

# Add edges with distance as weight
for _, row in edges_pd.iterrows():
    G.add_edge(
        row['ORIGIN'], 
        row['DEST'], 
        weight=row['distance'],
        num_flights=row['num_flights']
    )

# Calculate betweenness centrality
# weight parameter uses edge weights for shortest path calculation
betweenness = nx.betweenness_centrality(G, weight='weight')

# Convert back to DataFrame
betweenness_df = pd.DataFrame([
    {'airport': k, 'betweenness': v} 
    for k, v in betweenness.items()
])

# Convert to Spark DataFrame and join back
betweenness_spark = spark.createDataFrame(betweenness_df)



# COMMAND ----------

df_with_pr_dc_betweenness = df_pr_dc.join(
    betweenness_spark,
    df_pr_dc.ORIGIN == betweenness_spark.airport,
    'left'
).drop('airport')

# handle nulls
pagerank_median = df_with_pr_dc_betweenness.agg(F.expr("percentile_approx(pagerank, 0.5)")).first()[0]
betweenness_median = df_with_pr_dc_betweenness.agg(F.expr("percentile_approx(betweenness, 0.5)")).first()[0]
degree_median = df_with_pr_dc_betweenness.agg(F.expr("percentile_approx(degree, 0.5)")).first()[0]

df_with_pr_dc_betweenness = df_with_pr_dc_betweenness.fillna({
    "pagerank": pagerank_median,
    "betweenness": betweenness_median,
    "degree": degree_median
})


#display number of rows and columns in the new dataframe
print(f"There are {df_with_pr_dc_betweenness.count()} rows and {len(df_with_pr_dc_betweenness.columns)} columns in the new dataframe.")
# display the number of rows and columns in the original dataframe
print(f"There are {df_1Y_features.count()} rows and {len(df_1Y_features.columns)} columns in the original dataframe.")

display(df_with_pr_dc_betweenness, limit=10)



# COMMAND ----------

# write to parquet file
df_with_pr_dc_betweenness.write.mode("overwrite").parquet(f"{data_path_12M}df_joined_1Y_features_plus_gf.parquet")
