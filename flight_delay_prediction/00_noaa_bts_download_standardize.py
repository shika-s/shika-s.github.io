# Databricks notebook source
# MAGIC %md
# MAGIC ### Download NOAA LCD-2 Batch files (1 file per year)

# COMMAND ----------

import os
import re
import xml.etree.ElementTree as ET
from urllib.request import urlopen

import requests


# 1) LIST endpoint (XML listing of keys under v2/archive/)
LISTING_URL = "https://www.ncei.noaa.gov/oa/local-climatological-data/?prefix=v2/archive/"

# 2) DOWNLOAD base for objects
DOWNLOAD_BASE = "https://www.ncei.noaa.gov/oa/local-climatological-data/v2/archive/"

# 3) Years you want (edit as needed)
years = [2020, 2021, 2022, 2023, 2024, 2025]

# 4) Local destination directory (DBFS mount path)
local_dir = "/dbfs/student-groups/Group_4_4/future_joins/NOAA/LCDv2/"
os.makedirs(local_dir, exist_ok=True)

# Matches keys like: v2/archive/lcd_v2.0.0_d2024_c20250710.tar.gz
KEY_RE = re.compile(
    r"^v2/archive/(?P<fname>lcd_v(?P<ver>[\d.]+)_d(?P<year>\d{4})_c(?P<created>\d{8})\.tar\.gz)$"
)

def list_archive_keys() -> list[str]:
    """Return all object keys under v2/archive/ from the XML listing."""
    xml_bytes = urlopen(LISTING_URL).read()
    root = ET.fromstring(xml_bytes)
    keys = []
    for el in root.iter():
        if el.tag.endswith("Key") and el.text:
            keys.append(el.text.strip())
    return keys

def pick_latest_tarball_per_year(keys: list[str], years: list[int]) -> dict[int, str]:
    """
    For each requested year, pick the tarball with the most recent build date (cYYYYMMDD).
    Returns {year: filename}.
    """
    best = {}  # year -> (created_yyyymmdd, filename)
    want = set(years)

    for k in keys:
        m = KEY_RE.match(k)
        if not m:
            continue
        y = int(m.group("year"))
        if y not in want:
            continue

        created = m.group("created")  # yyyymmdd
        fname = m.group("fname")

        if (y not in best) or (created > best[y][0]):
            best[y] = (created, fname)

    missing = sorted(want - set(best.keys()))
    if missing:
        raise FileNotFoundError(
            f"No LCDv2 archive tarball found for year(s): {missing}. "
            f"Check LISTING_URL manually or update your year list."
        )

    return {y: best[y][1] for y in sorted(best.keys())}

def download_stream(url: str, dest_path: str, chunk_bytes: int = 1024 * 1024) -> None:
    """Stream download (safer for multi-GB files than urlretrieve)."""
    tmp_path = dest_path + ".part"

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", "0"))

        downloaded = 0
        with open(tmp_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_bytes):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

        # Basic sanity check if server provided length
        if total and downloaded != total:
            raise IOError(f"Incomplete download for {url}: got {downloaded} bytes, expected {total} bytes.")

    os.replace(tmp_path, dest_path)

# ---- RUN ----
all_keys = list_archive_keys()
year_to_fname = pick_latest_tarball_per_year(all_keys, years)

for year, fname in year_to_fname.items():
    url = DOWNLOAD_BASE + fname
    local_path = os.path.join(local_dir, fname)

    if os.path.exists(local_path):
        print(f"[SKIP] {year}: already exists -> {local_path}")
        continue

    print(f"[DL] {year}: {fname}")
    download_stream(url, local_path)
    print(f"[OK]  saved -> {local_path}")


# COMMAND ----------

# MAGIC %md
# MAGIC ### Download BTS On-Time (Reporting Carrier) Batch files (1 file per month)

# COMMAND ----------

# Monthly BTS On-Time (Reporting Carrier) files for 2020–2024
# Source family:
#   https://www.bts.gov/browse-statistical-products-and-data/bts-publications/airline-service-quality-performance-234-time
# Data files:
#   https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip

BASE_URL_TEMPLATE = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

MONTH_NAMES = [
    "January", "February", "March", "April",
    "May", "June", "July", "August",
    "September", "October", "November", "December",
]

airline_ontime_2020_2024 = [
    {
        "year": year,
        "month": month,                           # 1–12
        "label": f"{MONTH_NAMES[month - 1]} {year}",
        "url": BASE_URL_TEMPLATE.format(year=year, month=month),
    }
    for year in range(2020, 2025)                # 2020, 2021, 2022, 2023, 2024
    for month in range(1, 13)                    # January..December
]

import urllib.request
import os

local_dir = "/dbfs/student-groups/Group_4_4/future_joins/BTS_OnTime/"
os.makedirs(local_dir, exist_ok=True)

for f in airline_ontime_2020_2024:
    filename = f["url"].split("/")[-1]
    local_path = os.path.join(local_dir, filename)
    urllib.request.urlretrieve(f["url"], local_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract BTS On-Time (Reporting Carrier) Batch files

# COMMAND ----------

import zipfile
import os

zip_dir = "/dbfs/student-groups/Group_4_4/future_joins/BTS_OnTime/"
local_zip_dir = "/dbfs/student-groups/Group_4_4/future_joins/BTS_OnTime/"

zip_files = [f for f in os.listdir(local_zip_dir) if f.endswith(".zip")]
success = True

for zip_file in zip_files:
    zip_path = os.path.join(local_zip_dir, zip_file)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(local_zip_dir)
    except Exception as e:
        success = False
        break

if success:
    for zip_file in zip_files:
        os.remove(os.path.join(local_zip_dir, zip_file))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join BTS On-Time (Reporting Carrier) CSV files

# COMMAND ----------

from pyspark.sql.functions import input_file_name

csv_dir = "dbfs:/student-groups/Group_4_4/future_joins/BTS_OnTime/"
df_all = spark.read.option("header", True).option("inferSchema", True).csv(csv_dir + "*.csv")
display(df_all)

parquet_path = "dbfs:/student-groups/Group_4_4/future_joins/BTS_OnTime/parquet_airlines_data_2020_2024.parquet"
df_all.write.mode("overwrite").parquet(parquet_path)

df_parquet = spark.read.parquet(parquet_path)
display(df_parquet)

print(f"CSV sample count: {df_all.count()}")
print(f"Parquet sample count: {df_parquet.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardize schema for BTS On-Time (Reporting Carrier) 

# COMMAND ----------

from pyspark.sql import functions as F
# Path to the Parquet file
schema_path = "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/"
new_file_path = "dbfs:/student-groups/Group_4_4/future_joins/BTS_OnTime/parquet_airlines_data_2020_2024.parquet"
parquet_path = "dbfs:/student-groups/Group_4_4/future_joins/BTS_OnTime/parquet_airlines_data_2020_2024_std.parquet"


# Load the Parquet data
df1 = spark.read.parquet(schema_path)
df2 = spark.read.parquet(new_file_path)

column_mapping = {
    # Date/Time fields
    "QUARTER": "Quarter",
    "MONTH": "Month",
    "DAY_OF_MONTH": "DayofMonth",
    "DAY_OF_WEEK": "DayOfWeek",
    "FL_DATE": "FlightDate",  # Will convert date to string
    "YEAR": "Year",
    
    # Carrier fields
    "FLIGHTS": "Flights",
    "OP_UNIQUE_CARRIER": "Reporting_Airline",
    "OP_CARRIER_AIRLINE_ID": "DOT_ID_Reporting_Airline",
    "OP_CARRIER": "IATA_CODE_Reporting_Airline",
    "TAIL_NUM": "Tail_Number",
    "OP_CARRIER_FL_NUM": "Flight_Number_Reporting_Airline",
    
    # Origin fields
    "ORIGIN_AIRPORT_ID": "OriginAirportID",
    "ORIGIN_AIRPORT_SEQ_ID": "OriginAirportSeqID",
    "ORIGIN_CITY_MARKET_ID": "OriginCityMarketID",
    "ORIGIN": "Origin",
    "ORIGIN_CITY_NAME": "OriginCityName",
    "ORIGIN_STATE_ABR": "OriginState",
    "ORIGIN_STATE_FIPS": "OriginStateFips",
    "ORIGIN_STATE_NM": "OriginStateName",
    "ORIGIN_WAC": "OriginWac",
    
    # Destination fields
    "DEST_AIRPORT_ID": "DestAirportID",
    "DEST_AIRPORT_SEQ_ID": "DestAirportSeqID",
    "DEST_CITY_MARKET_ID": "DestCityMarketID",
    "DEST": "Dest",
    "DEST_CITY_NAME": "DestCityName",
    "DEST_STATE_ABR": "DestState",
    "DEST_STATE_FIPS": "DestStateFips",
    "DEST_STATE_NM": "DestStateName",
    "DEST_WAC": "DestWac",
    
    # Departure fields
    "CRS_DEP_TIME": "CRSDepTime",
    "DEP_TIME": "DepTime",
    "DEP_DELAY": "DepDelay",
    "DEP_DELAY_NEW": "DepDelayMinutes",  # New format uses DepDelayMinutes
    "DEP_DEL15": "DepDel15",
    "DEP_DELAY_GROUP": "DepartureDelayGroups",
    "DEP_TIME_BLK": "DepTimeBlk",
    "TAXI_OUT": "TaxiOut",
    "WHEELS_OFF": "WheelsOff",
    "FIRST_DEP_TIME": "FirstDepTime",
    "TOTAL_ADD_GTIME": "TotalAddGTime",
    "LONGEST_ADD_GTIME": "LongestAddGTime",

    
    # Arrival fields
    "WHEELS_ON": "WheelsOn",
    "TAXI_IN": "TaxiIn",
    "CRS_ARR_TIME": "CRSArrTime",
    "ARR_TIME": "ArrTime",
    "ARR_DELAY": "ArrDelay",
    "ARR_DELAY_NEW": "ArrDelayMinutes",  # New format uses ArrDelayMinutes
    "ARR_DEL15": "ArrDel15",
    "ARR_DELAY_GROUP": "ArrivalDelayGroups",  # Check if this column exists in df2
    "ARR_TIME_BLK": "ArrTimeBlk",
    
    # Flight duration fields
    "CRS_ELAPSED_TIME": "CRSElapsedTime",
    "ACTUAL_ELAPSED_TIME": "ActualElapsedTime",
    "AIR_TIME": "AirTime",
    
    # Distance
    "DISTANCE": "Distance",
    "DISTANCE_GROUP": "DistanceGroup",
    
    # Cancellation fields
    "CANCELLED": "Cancelled",
    "CANCELLATION_CODE": "CancellationCode",
    
    # Delay fields (may not exist in all datasets)
    "CARRIER_DELAY": "CarrierDelay",
    "WEATHER_DELAY": "WeatherDelay",
    "NAS_DELAY": "NASDelay",
    "SECURITY_DELAY": "SecurityDelay",
    "LATE_AIRCRAFT_DELAY": "LateAircraftDelay",
    
    # Diverted flight fields (DIV1)
    "DIVERTED": "Diverted",
    "DIV1_AIRPORT": "Div1Airport",
    "DIV1_AIRPORT_ID": "Div1AirportID",
    "DIV1_AIRPORT_SEQ_ID": "Div1AirportSeqID",
    "DIV1_LONGEST_GTIME": "Div1LongestGTime",
    "DIV1_TAIL_NUM": "Div1TailNum",
    "DIV1_TOTAL_GTIME": "Div1TotalGTime",
    "DIV1_WHEELS_OFF": "Div1WheelsOff",
    "DIV1_WHEELS_ON": "Div1WheelsOn",
    
    # Diverted flight fields (DIV2)
    "DIV2_AIRPORT": "Div2Airport",
    "DIV2_AIRPORT_ID": "Div2AirportID",
    "DIV2_AIRPORT_SEQ_ID": "Div2AirportSeqID",
    "DIV2_LONGEST_GTIME": "Div2LongestGTime",
    "DIV2_TAIL_NUM": "Div2TailNum",
    "DIV2_TOTAL_GTIME": "Div2TotalGTime",
    "DIV2_WHEELS_OFF": "Div2WheelsOff",
    "DIV2_WHEELS_ON": "Div2WheelsOn",
    "DIV_AIRPORT_LANDINGS": "DivAirportLandings",
    "DIV_REACHED_DEST": "DivReachedDest",
    "DIV_ACTUAL_ELAPSED_TIME": "DivActualElapsedTime",
    "DIV_ARR_DELAY": "DivArrDelay",
    "DIV_DISTANCE": "DivDistance",
    
    # Diverted flight fields (DIV3)
    "DIV3_AIRPORT": "Div3Airport",
    "DIV3_AIRPORT_ID": "Div3AirportID",
    "DIV3_AIRPORT_SEQ_ID": "Div3AirportSeqID",
    "DIV3_LONGEST_GTIME": "Div3LongestGTime",
    "DIV3_TAIL_NUM": "Div3TailNum",
    "DIV3_TOTAL_GTIME": "Div3TotalGTime",
    "DIV3_WHEELS_OFF": "Div3WheelsOff",
    "DIV3_WHEELS_ON": "Div3WheelsOn",

    # Diverted flight fields (DIV4)
    "DIV4_AIRPORT": "Div4Airport",
    "DIV4_AIRPORT_ID": "Div4AirportID",
    "DIV4_AIRPORT_SEQ_ID": "Div4AirportSeqID",
    "DIV4_WHEELS_ON": "Div4WheelsOn",
    "DIV4_TOTAL_GTIME": "Div4TotalGTime",
    "DIV4_LONGEST_GTIME": "Div4LongestGTime",
    "DIV4_WHEELS_OFF": "Div4WheelsOff",
    "DIV4_TAIL_NUM": "Div4TailNum",
    
    # Diverted flight fields (DIV5)
    "DIV5_AIRPORT": "Div5Airport",
    "DIV5_AIRPORT_ID": "Div5AirportID",
    "DIV5_AIRPORT_SEQ_ID": "Div5AirportSeqID",
    "DIV5_WHEELS_ON": "Div5WheelsOn",
    "DIV5_TOTAL_GTIME": "Div5TotalGTime",
    "DIV5_LONGEST_GTIME": "Div5LongestGTime",
    "DIV5_WHEELS_OFF": "Div5WheelsOff",
    "DIV5_TAIL_NUM": "Div5TailNum"

}

# Build select expressions with type casting
select_exprs = []
missing_cols = []
mapped_cols = []

# Get df1 schema for type matching
df1_schema_dict = {field.name: field.dataType for field in df1.schema.fields}

# Iterate through df1's columns (TARGET format) and map from df2 (SOURCE)
for target_col in df1.columns:  # target_col = "QUARTER" (df1 format)
    src_col = column_mapping.get(target_col)  # src_col = "Quarter" (df2 column name)
    
    if src_col and src_col in df2.columns:
        # Column exists in df2, take data from df2 and rename to df1 format
        target_type = df1_schema_dict[target_col]
        
        # Special handling for FL_DATE (date to string conversion)
        if target_col == "FL_DATE" and src_col == "FlightDate":
            # Take FlightDate from df2, convert to string, alias as FL_DATE (df1 format)
            select_exprs.append(
                F.date_format(F.col(src_col), "yyyy-MM-dd").cast(target_type).alias(target_col)
            )
        else:
            # Take src_col from df2, cast to df1's type, alias as target_col (df1 format)
            # Example: F.col("Quarter") from df2 -> alias as "QUARTER" (df1 format)
            select_exprs.append(
                F.col(src_col).cast(target_type).alias(target_col)
            )
        mapped_cols.append(target_col)
    else:
        # Column not found in df2, fill with NULL but keep df1's column name and type
        target_type = df1_schema_dict[target_col]
        select_exprs.append(
            F.lit(None).cast(target_type).alias(target_col)
        )
        missing_cols.append(target_col)

# Create standardized dataframe
# df2_standardized will have df1's column names (QUARTER, MONTH, FL_DATE, etc.)
# but with data from df2 (Quarter -> QUARTER, Month -> MONTH, FlightDate -> FL_DATE, etc.)
df2_standardized = df2.select(*select_exprs)
df2_standardized.write.mode("overwrite").parquet(parquet_path)

display(df2_standardized)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Extract NOAA LCD-2 Batch files 

# COMMAND ----------

import tarfile
import os

local_tar_dir = "/dbfs/student-groups/Group_4_4/future_joins/NOAA/LCDv2/"
tar_files = [f for f in os.listdir(local_tar_dir) if f.endswith(".tar.gz")]
success = True

for tar_file in tar_files:
    tar_path = os.path.join(local_tar_dir, tar_file)
    try:
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(local_tar_dir)
    except Exception as e:
        success = False
        break

if success:
    for tar_file in tar_files:
        os.remove(os.path.join(local_tar_dir, tar_file))

display(dbutils.fs.ls("dbfs:/student-groups/Group_4_4/future_joins/NOAA/LCDv2/"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Join NOAA LCD-2 Batch files 

# COMMAND ----------

from pyspark.sql.functions import input_file_name

csv_dir = "dbfs:/student-groups/Group_4_4/future_joins/NOAA/LCDv2/"
df_all = spark.read.option("header", True).option("inferSchema", True).csv(csv_dir + "*.csv")
df_all = df_all.filter(~input_file_name().contains("_2025"))
display(df_all)

parquet_path = "dbfs:/student-groups/Group_4_4/future_joins/NOAA/LCDv2_weather_data_2020_2024.parquet"
df_all.write.mode("overwrite").parquet(parquet_path)

df_parquet = spark.read.parquet(parquet_path)
display(df_parquet)

print(f"CSV sample count: {df_all.count()}")
print(f"Parquet sample count: {df_parquet.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Standardize schema for NOAA LCD-2 reports

# COMMAND ----------

from pyspark.sql.functions import col, year

parquet_in = "dbfs:/student-groups/Group_4_4/future_joins/NOAA/LCDv2_weather_data_2020_2024.parquet"
parquet_out = "dbfs:/student-groups/Group_4_4/future_joins/NOAA/LCDv2_weather_data_2020_2024_std.parquet"

df2 = spark.read.parquet(parquet_in)

df2_standardized = df2.select(
    "*",
    col('MonthlyAverageWindSpeed').alias('AWND'),
    col('MonthlyCoolingDegreeDays').alias('CLDD'),
    col('MonthlyHeatingDegreeDays').alias('HTDD'),
    col('CoolingDegreeDaysSeasonToDate').alias('CDSD'),
    col('HeatingDegreeDaysSeasonToDate').alias('HDSD'),
    col('MonthlyNumberDaysWithSnowfall').alias('DSNW'),
    year(col('DATE')).alias('YEAR')
)

from pyspark.sql import functions as F
from pyspark.sql.window import Window


# 2) Rename STATION -> GHCN_ID
df2_standardized = df2_standardized.withColumnRenamed("STATION", "GHCN_ID")

# 3) Parse GHCN network code + candidate join keys
df2_standardized = (df2_standardized
  .withColumn("network_code", F.substring("GHCN_ID", 3, 1))
  .withColumn("icao_from_ghcn", F.when(F.col("network_code")=="I", F.substring("GHCN_ID", 8, 4)))
  .withColumn("wban_from_ghcn", F.when(F.col("network_code")=="W", F.regexp_extract("GHCN_ID", r"(\d{5})$", 1)))
  .withColumn("wban_from_ghcn", F.lpad("wban_from_ghcn", 5, "0"))
)

# 4) Load ISD history (contains USAF, WBAN, ICAO, BEGIN, END)
isd_path = "/dbfs/student-groups/Group_4_4/future_joins/NOAA/isd-history.csv"

import urllib.request
urllib.request.urlretrieve(
    "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv",
    isd_path
)

isd = (
    spark.read.option("header", True).csv("dbfs:/student-groups/Group_4_4/future_joins/NOAA/isd-history.csv")
    .select("USAF", "WBAN", "ICAO", "BEGIN", "END")
    .withColumn("USAF", F.lpad(F.col("USAF").cast("string"), 6, "0"))
    .withColumn("WBAN", F.lpad(F.col("WBAN").cast("string"), 5, "0"))
    .withColumn("station11", F.concat("USAF", "WBAN"))
)

# 5) Join preference:
#    - If network_code == I: join by ICAO
#    - If network_code == W: join by WBAN (then you get USAF from ISD history)
wx_isd = (df2_standardized
  .join(isd, df2_standardized.icao_from_ghcn == isd.ICAO, "left")
  .withColumn("station11_by_icao", F.col("station11"))
  .drop("station11","USAF","WBAN","ICAO","BEGIN","END")
)

wx_isd = (wx_isd
  .join(isd, wx_isd.wban_from_ghcn == isd.WBAN, "left")
  .withColumn("station11_by_wban", F.col("station11"))
)

# 6) Choose final STATION (ISD station11)
wx_isd = wx_isd.withColumn(
  "STATION",
  F.coalesce("station11_by_icao", "station11_by_wban")
).drop("station11_by_icao","station11_by_wban")

# 7) Now you have both: GHCN_ID and ISD STATION
wx_isd.select("STATION","GHCN_ID","DATE").show(5, False)


wx_isd.write.mode("overwrite").parquet(parquet_out)
display(spark.read.parquet(parquet_out))