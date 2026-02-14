# Databricks notebook source
# MAGIC %md
# MAGIC ## Instructions 

# COMMAND ----------

# MAGIC %md
# MAGIC The main objective of this pipeline is to create a single analytical table where each row is a flight, enriched with:
# MAGIC Core scheduling and routing information (carrier, origin/destination, distance, calendar fields).
# MAGIC Binary delay labels and supporting delay metrics (`DEP_DEL15`, `ARR_DEL15`, `DEP_DELAY`, `ARR_DELAY`).
# MAGIC “As-of” origin weather features sampled from NOAA hourly data before departure (temperature, visibility, wind, precipitation, etc.).
# MAGIC Geographic helper fields (lat/lon for airports and stations, station–airport distances).
# MAGIC The result is a model-ready dataset that can be used for classification (on-time vs delayed) regression on delay minutes, while carefully avoiding information leakage from post-departure variables.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Feature selection and leakage control

# COMMAND ----------

# MAGIC %md
# MAGIC We define a clear feature-selection and leakage-control policy before any heavy joins are performed. First, it distinguishes outcomes from inputs: delay outcomes are treated strictly as labels or evaluation targets, while everything that could realistically be known before departure is treated as candidate predictors. On the input side, the dataset emphasizes variables that are fixed in advance or known operationally before takeoff: calendar information that captures seasonal and weekly patterns; planned schedule details such as planned departure and arrival times and planned flight duration; and structural flight characteristics like carrier identity, route, and distance. These are complemented with “as-of” origin weather features obtained from historical observations up to a fixed cutoff time before departure, ensuring that only past or present meteorological information is used. In contrast, any field that reflects what actually happened after the aircraft began its departure sequence—such as realized delays, taxi-out times, airborne times, or ex-post cause codes—is explicitly excluded from the modeling feature set to avoid data leakage. Those variables are retained only in separate diagnostic groups for analysis and model evaluation, not as predictors. This separation between pre-event and post-event information ensures that the final feature set mirrors the information a real-time system would have at prediction time, guarding against over-optimistic performance estimates and improving the robustness of the resulting models.

# COMMAND ----------

# MAGIC %md
# MAGIC | **Feature**           | **Type**           | **Description**                                         | **Notes / Use in Model**                     |
# MAGIC | --------------------- | ------------------ | ------------------------------------------------------- | -------------------------------------------- |
# MAGIC | `DEP_DEL15`           | Target (Binary)    | 1 if departure delay ≥15 minutes, else 0                | Main prediction target                       |
# MAGIC | `DEP_DELAY`           | Numeric            | Departure delay in minutes (negative = early departure) | Used for validation and correlation          |
# MAGIC | `CRS_DEP_HOUR`        | Numeric (0–23)     | Scheduled departure hour extracted from `CRS_DEP_TIME`  | Captures daily delay pattern                 |
# MAGIC | `DAY_OF_WEEK`         | Categorical (1–7)  | Day of the week (1 = Monday, 7 = Sunday)                | Reflects weekday vs. weekend trends          |
# MAGIC | `MONTH`               | Categorical (1–12) | Month of the year                                       | Captures seasonal variation                  |
# MAGIC | `OP_UNIQUE_CARRIER`   | Categorical        | Airline carrier code (e.g., AA, DL, UA)                 | Delay rates vary by airline                  |
# MAGIC | `ORIGIN`              | Categorical        | Origin airport IATA code                                | Captures airport-level delay patterns        |
# MAGIC | `DEST`                | Categorical        | Destination airport IATA code                           | May add contextual variation                 |
# MAGIC | `DISTANCE`            | Numeric            | Flight distance in miles                                | Longer flights less affected by short delays |
# MAGIC | `TAXI_OUT`            | Numeric            | Taxi-out time in minutes                                | Indicator of airport congestion              |
# MAGIC | `CRS_ELAPSED_TIME`    | Numeric            | Scheduled flight duration in minutes                    | Useful for normalization                     |
# MAGIC | `HourlyPrecipitation` | Numeric            | Precipitation at departure station (inches/hour)        | Proxy for adverse weather                    |
# MAGIC | `HourlyVisibility`    | Numeric            | Visibility at departure station (miles)                 | Lower values may increase delay risk         |
# MAGIC | `HourlyWindSpeed`     | Numeric            | Wind speed at departure station (mph)                   | Captures storm or runway condition impact    |
# MAGIC | `CANCELLED`           | Binary             | 1 if flight was canceled                                | May need to exclude or treat separately      |
# MAGIC | `DIVERTED`            | Binary             | 1 if flight diverted to another airport                 | Usually excluded for modeling                |

# COMMAND ----------

# MAGIC %md
# MAGIC | Column                | Description                                                  | Source | Notes                                       |
# MAGIC | :-------------------- | :----------------------------------------------------------- | :----- | :------------------------------------------ |
# MAGIC | QUARTER               | Calendar quarter of the year (1–4)                           | Flight | some seasons might experience more delays than others: seasonality                      |
# MAGIC | MONTH                 | Month of flight date (1–12)                                  | Flight | Use for monthly trends                      |
# MAGIC | DAY_OF_MONTH          | Day of the month (1–31)                                      | Flight | Temporal feature                            |
# MAGIC | DAY_OF_WEEK           | Day of the week (1=Mon, 7=Sun)                               | Flight | Delays vary by weekday                      |
# MAGIC | FL_DATE               | Flight date                                                  | Flight | Combine with time fields for timestamp      |
# MAGIC | OP_UNIQUE_CARRIER     | Unique airline carrier code (e.g., AA, DL)                   | Flight | Key categorical variable                    |
# MAGIC | OP_CARRIER_AIRLINE_ID | Airline numeric ID from BTS                                  | Flight | Alternate carrier ID                        |
# MAGIC | OP_CARRIER            | Carrier abbreviation                                         | Flight | Duplicate of OP_UNIQUE_CARRIER              |
# MAGIC | TAIL_NUM              | Aircraft tail number                                         | Flight | Often missing or reused                     |
# MAGIC | OP_CARRIER_FL_NUM     | Flight number                                                | Flight | Combine with carrier for unique flight ID   |
# MAGIC | ORIGIN_AIRPORT_ID     | Unique numeric ID for origin airport                         | Flight | Key for joins                               |
# MAGIC | ORIGIN_AIRPORT_SEQ_ID | Unique ID per airport sequence                               | Flight | Not needed for modeling                     |
# MAGIC | ORIGIN_CITY_MARKET_ID | City market ID                                               | Flight | Identifies metro area                       |
# MAGIC | ORIGIN                | Origin airport code (IATA)                                   | Flight | Major key feature                           |
# MAGIC | ORIGIN_CITY_NAME      | Full city name of origin                                     | Flight | Redundant with ORIGIN                       |
# MAGIC | ORIGIN_STATE_ABR      | Origin state abbreviation                                    | Flight | Useful for mapping                          |
# MAGIC | ORIGIN_STATE_FIPS     | State FIPS code                                              | Flight | Redundant geographic ID                     |
# MAGIC | ORIGIN_STATE_NM       | Full state name                                              | Flight | Informational only                          |
# MAGIC | ORIGIN_WAC            | World Area Code for origin                                   | Flight | May be dropped                              |
# MAGIC | DEST_AIRPORT_ID       | Unique numeric ID for destination airport                    | Flight | Key for joins                               |
# MAGIC | DEST_AIRPORT_SEQ_ID   | Destination sequence ID                                      | Flight | Often redundant                             |
# MAGIC | DEST_CITY_MARKET_ID   | Destination city market ID                                   | Flight | Identifies metro area                       |
# MAGIC | DEST                  | Destination airport code (IATA)                              | Flight | Key feature                                 |
# MAGIC | DEST_CITY_NAME        | Full city name of destination                                | Flight | Informational                               |
# MAGIC | DEST_STATE_ABR        | Destination state abbreviation                               | Flight | Useful for mapping                          |
# MAGIC | DEST_STATE_FIPS       | Destination state FIPS code                                  | Flight | Redundant                                   |
# MAGIC | DEST_STATE_NM         | Full destination state name                                  | Flight | Informational only                          |
# MAGIC | DEST_WAC              | World Area Code for destination                              | Flight | May be dropped                              |
# MAGIC | CRS_DEP_TIME          | Scheduled departure time (HHMM local)                        | Flight | Convert to hour for modeling                |
# MAGIC | DEP_TIME              | Actual departure time (HHMM local)                           | Flight | Post-departure → leakage                    |
# MAGIC | DEP_DELAY             | Departure delay in minutes                                   | Flight | Leakage (after event)                       |
# MAGIC | DEP_DELAY_NEW         | Departure delay, no negatives                                | Flight | Leakage (after event)                       |
# MAGIC | DEP_DEL15             | 1 if departure delay ≥15 min                                 | Flight | Post-departure indicator                    |
# MAGIC | DEP_DELAY_GROUP       | Categorical group of departure delay                         | Flight | Leakage variable                            |
# MAGIC | DEP_TIME_BLK          | Scheduled departure block (time interval)                    | Flight | Keep for modeling                           |
# MAGIC | TAXI_OUT              | Taxi-out time in minutes                                     | Flight | Leakage; occurs after departure             |
# MAGIC | WHEELS_OFF            | Time wheels left ground (HHMM)                               | Flight | Leakage                                     |
# MAGIC | WHEELS_ON             | Time wheels touched down (HHMM)                              | Flight | Leakage                                     |
# MAGIC | TAXI_IN               | Taxi-in time (minutes)                                       | Flight | Leakage                                     |
# MAGIC | CRS_ARR_TIME          | Scheduled arrival time (HHMM)                                | Flight | Keep; pre-scheduled info                    |
# MAGIC | ARR_TIME              | Actual arrival time (HHMM)                                   | Flight | Leakage                                     |
# MAGIC | ARR_DELAY             | Arrival delay (minutes)                                      | Flight | Target-related; drop                        |
# MAGIC | ARR_DELAY_NEW         | Non-negative arrival delay                                   | Flight | Redundant                                   |
# MAGIC | ARR_DEL15             | (1 if arrival delay ≥15 min)             | Flight | Binary label                                |
# MAGIC | ARR_DELAY_GROUP       | Grouped arrival delay                                        | Flight | Redundant with ARR_DEL15                    |
# MAGIC | ARR_TIME_BLK          | Scheduled arrival block                                      | Flight | Pre-scheduled; usable                       |
# MAGIC | CANCELLED             | 1 if flight was cancelled                                    | Flight | Keep for classification                     |
# MAGIC | CANCELLATION_CODE     | Code for reason of cancellation (A=Carrier, B=Weather, etc.) | Flight | Important categorical for cancelled flights |
# MAGIC | DIVERTED              | 1 if flight diverted to another airport                      | Flight | Keep; rare event                            |
# MAGIC | CRS_ELAPSED_TIME      | Scheduled elapsed flight time (min)                          | Flight | Useful duration variable                    |
# MAGIC | ACTUAL_ELAPSED_TIME   | Actual total flight time (min)                               | Flight | Leakage                                     |
# MAGIC | AIR_TIME              | In-air flight time (min)                                     | Flight | Leakage                                     |
# MAGIC | FLIGHTS               | Number of flights (usually 1)                                | Flight | Constant; drop                              |
# MAGIC | DISTANCE              | Great circle distance (miles)                                | Flight | Key continuous variable                     |
# MAGIC | DISTANCE_GROUP        | Distance category (1=short haul, etc.)                       | Flight | Categorical; keep                           |
# MAGIC | CARRIER_DELAY         | Delay due to airline (min)                                   | Flight | Post-event; leakage                         |
# MAGIC | WEATHER_DELAY         | Delay due to weather (min)                                   | Flight | Leakage                                     |
# MAGIC | NAS_DELAY             | Delay due to air traffic control (min)                       | Flight | Leakage                                     |
# MAGIC | SECURITY_DELAY        | Delay due to security (min)                                  | Flight | Leakage                                     |
# MAGIC | LATE_AIRCRAFT_DELAY   | Delay due to late incoming aircraft (min)                    | Flight | Leakage                                     |
# MAGIC | FIRST_DEP_TIME        | First departure attempt (for multi-leg flights)              | Flight | Leakage                                     |
# MAGIC | TOTAL_ADD_GTIME       | Total gate time added                                        | Flight | Leakage                                     |
# MAGIC | LONGEST_ADD_GTIME     | Longest gate time added                                      | Flight | Leakage                                     |

# COMMAND ----------

# MAGIC %md
# MAGIC | Category                           | Columns                                                                                                                                                                                               | Action                       | Notes                                                                             |
# MAGIC | :--------------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------- | :-------------------------------------------------------------------------------- |
# MAGIC | **Target**                         | `ARR_DEL15`                                                                                                                                                                                           | Keep                         | Binary classification label (1 = delay ≥15 min)                                   |
# MAGIC | **Temporal Features**              | `MONTH`, `DAY_OF_WEEK`, `CRS_DEP_TIME`, `CRS_ARR_TIME`, `DEP_TIME_BLK`, `ARR_TIME_BLK`                                                                                                                | Keep (or engineer)           | Convert times to hour bins or features; captures time-of-day and seasonal effects |
# MAGIC | **Flight Ops / Routing**           | `OP_UNIQUE_CARRIER`, `ORIGIN`, `DEST`, `DISTANCE`, `DISTANCE_GROUP`, `CANCELLED`, `DIVERTED`                                                                                                          | Keep                         | Core predictive variables; encode categorical features                            |
# MAGIC | **Weather (Hourly)**               | `HourlyPrecipitation`, `HourlyVisibility`, `HourlyWindSpeed`                                                                                                                                          | Keep                         | Key real-time weather indicators                                                  |
# MAGIC | **Weather (Hourly – Derived)**     | `HourlyPrecipitation_D`, `HourlyVisibility_D`, `HourlyWindSpeed_D`                                                                                                                                    | Keep                         | Change/delta features showing recent shifts                                       |
# MAGIC | **Weather (Daily)**                | `DailyPrecipitation`, `DailyMaximumDryBulbTemperature`, `DailyMinimumDryBulbTemperature`, `DailyPeakWindSpeed`                                                                                        | Review                       | Optional aggregates for extended weather context                                  |
# MAGIC | **Geographic / Station Info**      | `LATITUDE`, `LONGITUDE`, `ELEVATION`, `origin_station_dis`                                                                                                                                            | Review                       | Keep if you plan spatial or distance-based analysis                               |
# MAGIC | **Carrier ID / Meta**              | `OP_CARRIER_AIRLINE_ID`, `OP_CARRIER`                                                                                                                                                                 | Drop                         | Redundant with `OP_UNIQUE_CARRIER`                                                |
# MAGIC | **Delay & Post-Arrival Variables** | `DEP_DELAY`, `ARR_DELAY`, `AIR_TIME`, `ACTUAL_ELAPSED_TIME`, `WHEELS_OFF`, `WHEELS_ON`, `TAXI_OUT`, `TAXI_IN`, `CARRIER_DELAY`, `WEATHER_DELAY`, `NAS_DELAY`, `SECURITY_DELAY`, `LATE_AIRCRAFT_DELAY` | Drop (leakage)               | All occur after or depend on delay outcome                                        |
# MAGIC | **Monthly / Climate Features**     | All `Monthly...` fields (`MonthlyAverageRH`, `MonthlyGreatestPrecip`, etc.)                                                                                                                           | Drop                         | More than 90% null and not relevant for single-flight prediction                  |
# MAGIC | **High-Null / Empty Monthly & Short-Duration** | `MonthlyAverageRH`, `MonthlyDaysWithGT001Precip`, `MonthlyDaysWithGT010Precip`, `MonthlyDaysWithGT32Temp`, `MonthlyDaysWithGT90Temp`, `MonthlyDaysWithLT0Temp`, `MonthlyDaysWithLT32Temp`, `MonthlyDepartureFromNormalAverageTemperature`, `MonthlyDepartureFromNormalCoolingDegreeDays`, `MonthlyDepartureFromNormalHeatingDegreeDays`, `MonthlyDepartureFromNormalMaximumTemperature`, `MonthlyDepartureFromNormalMinimumTemperature`, `MonthlyDepartureFromNormalPrecipitation`, `MonthlyDewpointTemperature`, `MonthlyGreatestPrecip`, `MonthlyGreatestPrecipDate`, `MonthlyGreatestSnowDepth`, `MonthlyGreatestSnowDepthDate`, `MonthlyGreatestSnowfall`, `MonthlyGreatestSnowfallDate`, `MonthlyMaxSeaLevelPressureValue`, `MonthlyMaxSeaLevelPressureValueDate`, `MonthlyMaxSeaLevelPressureValueTime`, `MonthlyMaximumTemperature`, `MonthlyMeanTemperature`, `MonthlyMinSeaLevelPressureValue`, `MonthlyMinSeaLevelPressureValueDate`, `MonthlyMinSeaLevelPressureValueTime`, `MonthlyMinimumTemperature`, `MonthlySeaLevelPressure`, `MonthlyStationPressure`, `MonthlyTotalLiquidPrecipitation`, `MonthlyTotalSnowfall`, `MonthlyWetBulb`, `AWND`, `CDSD`, `CLDD`, `DSNW`, `HDSD`, `HTDD`, `NormalsCoolingDegreeDay`, `NormalsHeatingDegreeDay`, `ShortDurationEndDate005`, `ShortDurationEndDate010`, `ShortDurationEndDate015`, `ShortDurationEndDate020`, `ShortDurationEndDate030`, `ShortDurationEndDate045`, `ShortDurationEndDate060`, `ShortDurationEndDate080`, `ShortDurationEndDate100`, `ShortDurationEndDate120`, `ShortDurationEndDate150`, `ShortDurationEndDate180`, `ShortDurationPrecipitationValue005`, `ShortDurationPrecipitationValue010`, `ShortDurationPrecipitationValue015`, `ShortDurationPrecipitationValue020`, `ShortDurationPrecipitationValue030`, `ShortDurationPrecipitationValue045`, `ShortDurationPrecipitationValue060`, `ShortDurationPrecipitationValue080`, `ShortDurationPrecipitationValue100`, `ShortDurationPrecipitationValue120`, `ShortDurationPrecipitationValue150`, `ShortDurationPrecipitationValue180` | Drop   | All null in OTPW;  safe to drop to simplify schema |
# MAGIC | **Backup / Metadata Fields**       | All `Backup...`, `ShortDuration...`, `_row_desc`, `REM`                                                                                                                                               | Drop                         | Join metadata and reference only                                                  |
# MAGIC | **Text Fields**                    | `NAME`, `REPORT_TYPE`, `DailyWeather`, `HourlySkyConditions`, `HourlyPresentWeatherType`                                                                                                              | Optional (drop for baseline) | Free text; may require NLP or feature extraction later                            |
# MAGIC | **Station Linking**                | `STATION`, `DATE`, `SOURCE`                                                                                                                                                                           | Keep for validation only     | Needed for join checks; not a model input                                         |

# COMMAND ----------

# MAGIC %md
# MAGIC #### iv. Airport–Weather Integration Plan (Data Joins, Issues & Rationale)
# MAGIC
# MAGIC Our exploratory review of the four source tables (flights, airport codes, weather, and station metadata) showed that the **biggest blockers are not in the flights themselves but in the lookup tables we need to join to**. The flights table already carries clean IATA airport codes (`ORIGIN`, `DEST`) and consistent date fields (`FL_DATE`, `YEAR`, `MONTH`, …), so it is a good factual backbone. However, our main airport codes file does **not** include time zones and sometimes only stores geolocation as a single `coordinates` string (`"lon, lat"`). At the same time, the external GitHub airport list **does** provide `timezone`, and often better lat/lon, but it doesn’t perfectly align 1:1 with our current codes file. This makes a direct “flights → airports → weather” join fragile, because we would be mixing two partially-overlapping airport catalogs. To fix this, we will first build **one master airport dimension** by joining the GitHub timezones and geolocation into the codes we already use in flights, and we will coalesce coordinates so that every IATA that appears as an origin/destination ends up with: **(a)** a timezone, **(b)** a lat/lon pair, and **(c)** a human-readable name.
# MAGIC
# MAGIC The second major issue is at the **weather** side: NOAA data is hourly and station-based (`STATION`, `DATE`, 100+ weather features), not airport-based. That means there is no native key to connect “ATL, SFO, ORD…” directly to a weather row. Also, stations and airports don’t share the exact same IDs: stations use a different identifier (and sometimes we need to normalize with the `stations.csv` file). On top of that, **weather is in UTC** while our flights are in **local airport time**, so if we don’t add airport timezones first, we can’t reliably pick “the weather hour that corresponds to this departure.” To solve this, we will: (1) compute airport → nearest-(1..3)-station pairs using the unified lat/lon we just created, (2) store that in a small bridge table (`airport_weather_station`), and (3) when we enrich flights, we will convert flight times to UTC using the airport’s timezone and then pick the matching hourly weather from the correct station. This approach gives us a repeatable pattern we can scale from the 3-month sample up to 2015–2021.
# MAGIC
# MAGIC **Key data problems we identified:**
# MAGIC - Our original airport codes file **lacks time zones**, so flight local times cannot be aligned to UTC weather.
# MAGIC - Airport geolocation is sometimes packed as a single text field (`coordinates`), so we must **parse and standardize lat/lon**.
# MAGIC - Weather rows are **station-based, not airport-based**, so we must create an extra **airport → station** bridge.
# MAGIC - Stations may come from **two slightly different sources** (`weather.csv` vs `stations.csv`), so we need **ID normalization**.
# MAGIC - Not all flights’ origin/dest codes are guaranteed to appear in the GitHub airport list, so we will **fallback to the original codes file** to avoid losing rows.

# COMMAND ----------

# MAGIC %md
# MAGIC #### v. Entity–Relationship Blueprint for Flights ↔ Airports ↔ Weather
# MAGIC
# MAGIC This diagram summarizes the core entities we will use to enrich flight records with meteorological data.
# MAGIC

# COMMAND ----------

mermaid_diagram_joins = """
<div class="mermaid">
erDiagram
    %% LEGEND
    %% PK_... = primary key (or business key)
    %% FK_... = foreign key to another table

    FLIGHTS {
        string PK_flight_row
        date FL_DATE
        string FK_origin_iata_code
        string FK_dest_iata_code
        string OP_UNIQUE_CARRIER
        int OP_CARRIER_FL_NUM
        int CRS_DEP_TIME
        int CRS_ARR_TIME
        int YEAR
        int MONTH
        int DAY_OF_MONTH
    }

    MASTER_AIRPORTS {
        string PK_iata_code
        string ident
        string name
        string municipality
        string iso_country
        string iso_region
        string airport_timezone
        float lat
        float lon
    }

    WEATHER {
        string PK_station_id
        datetime PK_obs_datetime
        float LATITUDE
        float LONGITUDE
        string NAME
        float HourlyDryBulbTemperature
        float HourlyVisibility
        float HourlyWindSpeed
    }

    NOAA_STATIONS {
        string PK_station_id_norm
        float lat
        float lon
        string neighbor_id
        float distance_to_neighbor
    }

    AIRPORT_WEATHER_STATION {
        string PK_iata_code
        string PK_station_id
        int PK_rank
        float dist_km
    }

    CHECKPOINTS {
        string checkpoint_name
        string file_path
        int rows
        int columns
        string description
    }

    %% RELATIONSHIPS
    FLIGHTS }o--|| MASTER_AIRPORTS : "origin (FK_origin_iata_code)"
    FLIGHTS }o--|| MASTER_AIRPORTS : "destination (FK_dest_iata_code)"
    MASTER_AIRPORTS ||--o{ AIRPORT_WEATHER_STATION : "airport → nearest stations"
    WEATHER ||--o{ AIRPORT_WEATHER_STATION : "station in bridge"
    WEATHER }o--|| NOAA_STATIONS : "normalize/enrich station"
    FLIGHTS ||--o{ CHECKPOINTS : "pipeline stages"
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({startOnLoad:true});</script>
"""
displayHTML(mermaid_diagram_joins)

# COMMAND ----------

# MAGIC %md
# MAGIC #### vi. To-Do List for Airport–Weather Join Pipeline
# MAGIC
# MAGIC **Phase 0 – Staging (raw → stg)**  
# MAGIC - [ ] Load **flights** (3m / 6m / 1y sample) into `stg_flights`  
# MAGIC - [ ] Load **airport codes** (current `codes.csv`) into `stg_airport_codes`  
# MAGIC - [ ] Load **GitHub airports with timezone** into `stg_airport_tz`  
# MAGIC - [ ] Load **weather** (NOAA hourly) into `stg_weather_hourly`  
# MAGIC - [ ] Load **station metadata** (`stations.csv`) into `stg_noaa_stations`
# MAGIC
# MAGIC **Phase 1 – Unified airport dimension**  
# MAGIC - [ ] Uppercase and trim IATA codes in **both** airport sources  
# MAGIC - [ ] Parse `coordinates` from `stg_airport_codes` → (`codes_lat`, `codes_lon`)  
# MAGIC - [ ] Select from `stg_airport_tz` only the needed fields: (`iata_code`, `airport_timezone`, `gh_lat`, `gh_lon`)  
# MAGIC - [ ] Left-join timezone + better lat/lon into `stg_airport_codes` and **coalesce** → `dim_airport (master_airports)`  
# MAGIC - [ ] Compare `dim_airport` to distinct `ORIGIN` and `DEST` from flights to find **missing airports**
# MAGIC
# MAGIC **Phase 2 – Weather station dimension**  
# MAGIC - [ ] Build `dim_weather_station` = distinct (`STATION`, `LATITUDE`, `LONGITUDE`, `NAME`) from `stg_weather_hourly`  
# MAGIC - [ ] Left-join to `stg_noaa_stations` to fill missing coordinates / IDs  
# MAGIC - [ ] Validate that all stations used in weather have lat/lon
# MAGIC
# MAGIC **Phase 3 – Airport ↔ station bridge (nearest K)**  
# MAGIC - [ ] Broadcast `dim_airport` (only airports with lat/lon)  
# MAGIC - [ ] Cross-join with `dim_weather_station`  
# MAGIC - [ ] Compute **Haversine** distance → `dist_km`  
# MAGIC - [ ] Window/partition by airport and keep top **K=3** nearest stations  
# MAGIC - [ ] Save as `airport_weather_station (iata_code, STATION, dist_km, rank)`
# MAGIC
# MAGIC **Phase 4 – Time alignment**  
# MAGIC - [ ] From flights, build `dep_ts_local` = `FL_DATE` + `CRS_DEP_TIME`  
# MAGIC - [ ] Convert `dep_ts_local` to UTC using `dim_airport.airport_timezone` (origin)  
# MAGIC - [ ] Repeat for arrival using destination airport  
# MAGIC - [ ] (Optional) Materialize `dim_date` / `dim_time` for reporting and easier joins
# MAGIC
# MAGIC **Phase 5 – Final enrichment views + QA**  
# MAGIC - [ ] Create `v_flights_with_origin_weather`:
# MAGIC   - join flights → origin airport → bridge (rank=1) → weather on matching UTC hour  
# MAGIC - [ ] Create `v_flights_with_dest_weather`:
# MAGIC   - join flights → destination airport → bridge (rank=1) → weather on matching UTC hour  
# MAGIC - [ ] Coverage report:
# MAGIC   - % flights with origin weather  
# MAGIC   - % flights with destination weather  
# MAGIC   - airports with `dist_km > 300` (flag for manual review)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration

# COMMAND ----------

# CONFIG: choose data slice and IO paths

# Options: "3M", "1Y", "5Y", "FUTURE"
DATA_SLICE = "5Y"

DATA_CONFIG = {
    "3M": {
        # course 3-month subset
        "flights_path": "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_3m/",
        "weather_path": "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_3m",
        "joined_output": "dbfs:/student-groups/Group_4_4/JOINED_3M.parquet",
        # The 3M dataset is already sliced
        "max_year": None,
        "max_month": None,
    },
    "1Y": {
        # course 1-year subset
        "flights_path": "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data_1y/",
        "weather_path": "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data_1y/",
        "joined_output": "dbfs:/student-groups/Group_4_4/JOINED_1Y.parquet",
        "max_year": None,
        "max_month": None,
    },
    "5Y": {
        # full 2015–2021 flights + weather, but we keep only up to 2019-12
        "flights_path": "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_airlines_data/",
        "weather_path": "dbfs:/mnt/mids-w261/datasets_final_project_2022/parquet_weather_data",
        "joined_output": "dbfs:/student-groups/Group_4_4/JOINED_5Y_2015_2019.parquet",
        "max_year": 2019,
        "max_month": 12,
    },
    "FUTURE": {
        # placeholders for future pipelines
        "flights_path": "dbfs:/student-groups/Group_4_4/future_joins/BTS_OnTime/parquet_airlines_data_2020_2024_std.parquet",
        "weather_path": "dbfs:/student-groups/Group_4_4/future_joins/NOAA/LCDv2_weather_data_2020_2024_std.parquet",
        "joined_output": "dbfs:/student-groups/Group_4_4/JOINED_FUTURE.parquet",
        "max_year": None,
        "max_month": None,
    },
}

cfg = DATA_CONFIG[DATA_SLICE]

print(f"Using configuration: {DATA_SLICE}")
for k, v in cfg.items():
    print(f"  {k}: {v}")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Library imports

# COMMAND ----------

import pandas as pd
import urllib.request
import pyspark.sql.functions as sf
from pyspark.sql import Window as W
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load stats helpers

# COMMAND ----------

# RUN STATS: start timer & helper functions

job_start = datetime.now()
print(f"Job start: {job_start.isoformat()}")

# Ensure dbutils is available (works in Databricks)
try:
    dbutils  # type: ignore[name-defined]
except NameError:
    from pyspark.dbutils import DBUtils
    dbutils = DBUtils(spark)

def get_dir_size(path: str) -> int:
    """
    Recursively compute total size (in bytes) of all files under `path`.
    Works for DBFS paths like 'dbfs:/mnt/...'.
    """
    total = 0
    for f in dbutils.fs.ls(path):
        if f.isDir():
            total += get_dir_size(f.path)
        else:
            total += f.size
    return total

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load helpers

# COMMAND ----------

# Helper function to pretty print databases
def show_df(df, n=5):
    """Pretty print the first `n` rows of a Spark DataFrame using Databricks display."""
    display(df.limit(n))

# Helper function to display columns of a Spark DataFrame
def show_columns(df):
    """Display the column names, data types, and % of null values of a Spark DataFrame."""
    total_rows = df.count()
    null_counts = df.select([sf.count(sf.when(sf.col(c).isNull(), c)).alias(c) for c in df.columns]).collect()[0].asDict()
    percent_null = {c: (null_counts[c] / total_rows * 100) if total_rows > 0 else None for c in df.columns}
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": [t for _, t in df.dtypes],
        "% Null": [percent_null[c] for c in df.columns]
    })
    display(col_info)
    print(f"Total rows: {total_rows}")

# Helper function haversine calculation
def haversine_km_expr(lat1, lon1, lat2, lon2):
    """
    Great-circle distance on a sphere (WGS84 mean Earth radius).
    All arguments are Column[double] in radians.
    Returns Column[double] in kilometers.
    """
    dlat = (lat2 - lat1)
    dlon = (lon2 - lon1)
    a = sf.pow(sf.sin(dlat / 2), 2) + sf.cos(lat1) * sf.cos(lat2) * sf.pow(sf.sin(dlon / 2), 2)
    c = 2 * sf.atan2(sf.sqrt(a), sf.sqrt(1 - a))
    return sf.lit(6371.0088) * c  # mean Earth radius in km

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load databases

# COMMAND ----------

# Flights data

"""
Flights data

     This is a subset of the passenger flight's on-time performance data taken from the TranStats data collection available from the U.S. Department of Transportation (DOT)

        https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ 

    Links to an external site. 

The flight dataset was downloaded from the US Department of Transportation
Links to an external site. and contains flight information from 2015 to 2021
(Note flight data for the period [2015-2019] has the following dimensionality  31,746,841 x 109)
A Data Dictionary for this dataset is located here:

    https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ 
"""

# Load flights based on DATA_SLICE
df_flights = spark.read.parquet(cfg["flights_path"])

# Drop exact duplicates
n_raw = df_flights.count()
df_flights = df_flights.dropDuplicates()
n_distinct = df_flights.count()
print(f"Flights - raw rows: {n_raw:,}, distinct rows: {n_distinct:,}")

# Display Results
# show_df(df_flights, 5)
# show_columns(df_flights)


# COMMAND ----------

# Weather data

"""
Weather table

    As a frequent flyer, we know that flight departure (and arrival)  often get affected by weather conditions, so it makes sense to collect, and process weather data corresponding to the origin and destination airports at the time of departure and arrival, respectively, and build features based upon this data. 
    The weather dataset was downloaded from the National Oceanic and Atmospheric Administration repository 

Links to an external site. and contains weather information from 2015 to 2021

    The dimensionality of the weather data for the period [2015-2019] is 630,904,436 x 177

Data dictionary (subset): 

    Please refer to pages 8-12:  https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf 

Links to an external site. 
A better version of the data dictionary can be read here: https://www.ncei.noaa.gov/pub/data/cdo/documentation/LCD_documentation.pdf
Links to an external site.

    A superset of the features is described here:

        https://www.ncei.noaa.gov/data/global-hourly/doc/isd-format-document.pdf 

    Links to an external site.

A subset of the features is shared here:

    https://docs.google.com/spreadsheets/d/1v0P34NlQKrvXGCACKDeqgxpDwmj3HxaleUiTrY7VRn0/edit#gid=0 
"""

# Load weather based on DATA_SLICE
df_weather = spark.read.parquet(cfg["weather_path"])

print(f"Weather - raw rows: {df_weather.count():,}")

# Allowed hourly report types (no QCLCD daily/monthly summaries)
ALLOWED_RPT = ["FM-15", "FM-16", "FM-12"]  # METAR, SPECI, SYNOP

# Filter to allowed report types AND create obs_utc from DATE
weather_hourly = (
    df_weather
    .filter(sf.col("REPORT_TYPE").isin(ALLOWED_RPT))
    .withColumn("obs_utc", sf.col("DATE").cast("timestamp"))
)

# Preference: METAR > SPECI > SYNOP, then latest timestamp
weather_ranked = weather_hourly.withColumn(
    "report_type_rank",
    sf.when(sf.col("REPORT_TYPE")=="FM-15", 1)
     .when(sf.col("REPORT_TYPE")=="FM-16", 2)
     .when(sf.col("REPORT_TYPE")=="FM-12", 3)
     .otherwise(99)
)

win_st_hr = W.partitionBy("STATION","obs_utc") \
             .orderBy(sf.col("report_type_rank").asc(), sf.col("DATE").desc())

weather_best = (
    weather_ranked
    .withColumn("rn", sf.row_number().over(win_st_hr))
    .filter(sf.col("rn") == 1)
    .drop("rn", "report_type_rank")
)

# Display results
# show_df(df_weather, 5)
# show_columns(df_weather)


# COMMAND ----------

# Weather station data

"""
Airport dataset
    Overall the airport dataset provides some metadata about each airport.
    The airport dataset was downloaded from the US Department of Transportation and has the following dimensionality: 18,097 x 10.
    It is located here:
        dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data

"""
df_stations = spark.read.parquet(f"dbfs:/mnt/mids-w261/datasets_final_project_2022/stations_data/stations_with_neighbors.parquet")

# Display results
# show_df(df_stations, 5)
# show_columns(df_stations)

# COMMAND ----------

# Airport codes
"""
Airport codes Table

    Airport codes may refer to either:
         IATA airport code, a three-letter code that is used in passenger reservation, ticketing, and baggage-handling systems,
        or ICAO airport code which is a four-letter code used by ATC systems and for airports that do not have an IATA airport code (from Wikipedia).
    Here you will need to import an external airport code conversion set (source: https://datahub.io/core/airport-codes 

Links to an external site.) and join the airport codes to the airline's flights table on the IATA code (3-letter code used by passengers)
"""

# Download and load airport codes CSV to Spark DataFrame
url = "https://datahub.io/core/airport-codes/_r/-/data/airport-codes.csv"
local_path = "/tmp/airport-codes.csv"
urllib.request.urlretrieve(url, local_path)
df_codes = spark.read.format("csv").option("header", True).load(local_path)

# Display results
# show_df(df_codes, 5)
# show_columns(df_codes)

# COMMAND ----------

"""
Airport Timezone & Geolocation Table

    This external table augments airport identifiers with the fields we need
    for time alignment and spatial joins. Each record includes:
        • IATA (3-letter passenger code) and ICAO (4-letter ATC code)
        • Airport name and locality metadata
        • Latitude and longitude (decimal degrees)
        • Time zone in IANA format (e.g., "America/Los_Angeles")

    Source:
        https://raw.githubusercontent.com/lxndrblz/Airports/main/airports.csv
        (Project page: https://github.com/lxndrblz/Airports)

    Usage:
        Import this CSV and join to the flights table on IATA (3-letter code used
        in `ORIGIN` / `DEST`). Use the `timezone` column to convert local flight
        times (e.g., FL_DATE + CRS_DEP_TIME) to UTC before weather joins, and use
        `latitude`/`longitude` to compute nearest NOAA weather stations.

    Notes:
        • Normalize IATA to uppercase and drop duplicates before joining.
        • Prefer this table’s lat/lon and timezone; if an airport is missing,
          fall back to the original codes file.
        • The timezone field follows IANA; ensure your Spark build supports
          IANA names when calling `to_utc_timestamp`.
        • /tmp is ephemeral and may be cleared when the cluster shuts down; use DBFS for persistent storage.
"""

dbutils.fs.mkdirs("dbfs:/student-groups/Group_4_4")
local_path = "/dbfs/student-groups/Group_4_4/airport-zones.csv"
url = "https://raw.githubusercontent.com/lxndrblz/Airports/main/airports.csv"
urllib.request.urlretrieve(url, local_path)
df_zones = spark.read.format("csv").option("header", True).load("dbfs:/student-groups/Group_4_4/airport-zones.csv")

# Display results
# show_df(df_zones)
# show_columns(df_zones)

# COMMAND ----------

# MAGIC %md
# MAGIC # Raw Dataframes Joins

# COMMAND ----------

# MAGIC %md
# MAGIC ## Airports

# COMMAND ----------

# Build MASTER_AIRPORTS with timezone +  latitude and longitude
codes = (
    df_codes
    .withColumn("iata_code", sf.upper("iata_code"))
    .withColumn("_coords", sf.split(sf.regexp_replace(sf.col("coordinates"), "\\s+", ""), ","))
    .withColumn("codes_lon", sf.col("_coords").getItem(0).cast("double"))
    .withColumn("codes_lat", sf.col("_coords").getItem(1).cast("double"))
    .drop("_coords")
)

# Display results
# show_df(codes, 5)



# COMMAND ----------

tz = (
    df_zones
    .withColumn("iata_code", sf.upper(sf.col("code")))
    .select(
        sf.col("iata_code"),
        sf.col("time_zone").alias("airport_timezone"),
        sf.col("latitude").cast("double").alias("gh_lat"),
        sf.col("longitude").cast("double").alias("gh_lon"),
        sf.col("name").alias("gh_name")
    )
    .dropna(subset=["iata_code"])
    .dropDuplicates(["iata_code"])
)

# Display results
# show_df(tz, 5)

# COMMAND ----------

df_airports = (
    codes.alias("c")
    .join(tz.alias("g"), on="iata_code", how="left")
    .withColumn("lat", sf.coalesce("g.gh_lat", "c.codes_lat"))
    .withColumn("lon", sf.coalesce("g.gh_lon", "c.codes_lon"))
    .withColumn("airport_timezone", sf.col("airport_timezone"))
    .select("iata_code", "ident", sf.col("c.type").alias("airport_type"), 
            "name", "municipality", "iso_country", "iso_region", "airport_timezone", "lat", "lon")
    .dropna(subset=["iata_code"])
    .dropDuplicates(["iata_code"])
)

# Display results
# show_df(df_airports, 5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Weather Stations

# COMMAND ----------

# Cleanup Weather stations
df_weather_station = (
    df_stations
    .select(
        sf.col("station_id").cast("string").alias("station_id"),
        sf.col("lat").cast("double").alias("lat"),
        sf.col("lon").cast("double").alias("lon")
    )
    .dropna(subset=["station_id","lat","lon"])
    .dropDuplicates(["station_id"])
)

# Display results
# show_df(df_weather_station, 5)
# show_columns(df_weather_station)


# COMMAND ----------

# Prepare airport coordinates in radians
airport_radians = (
    df_airports
    .dropna(subset=["lat", "lon"])
    .withColumn("lat_rad", sf.radians("lat"))
    .withColumn("lon_rad", sf.radians("lon"))
    .select("iata_code", "lat", "lon", "lat_rad", "lon_rad")  
)

# Display results
# show_df(airport_radians, 5)
# show_columns(airport_radians)

# COMMAND ----------

# Prepare stations coordinates in radians
stations_radians = (
    df_weather_station
    .dropna(subset=["lat", "lon"])
    .withColumn("st_lat_rad", sf.radians("lat"))
    .withColumn("st_lon_rad", sf.radians("lon"))
    .select(sf.col("station_id").alias("station_id"), "lat", "lon", "st_lat_rad", "st_lon_rad")
)
# Display results
# show_df(stations_radians, 5)
# show_columns(stations_radians)

# COMMAND ----------

# Cross join (broadcast the small side), compute distance, rank, keep top-3 per airport
airports_stations_cross = (
    sf.broadcast(airport_radians).crossJoin(stations_radians)
    .withColumn("dist_km", haversine_km_expr(sf.col("lat_rad"), sf.col("lon_rad"),
                                             sf.col("st_lat_rad"), sf.col("st_lon_rad")))
    .select(
        sf.col("iata_code"),
        sf.col("station_id").alias("STATION"),
        sf.col("dist_km")
    )
)
station_rank = W.partitionBy("iata_code").orderBy(sf.col("dist_km").asc())
airport_weather_station = (
    airports_stations_cross
    .withColumn("rank", sf.row_number().over(station_rank))
    .filter(sf.col("rank") <= 3)
    .select("iata_code", "STATION", "dist_km", "rank")
)

# COMMAND ----------

# Prediction timestamp (origin): build local time from schedule (no leakage), then T–2h, then local→UTC
fl = df_flights
crs = (
    fl
    .withColumn("CRS_DEP_TIME_str", sf.lpad(sf.col("CRS_DEP_TIME").cast("string"), 4, "0"))
    .withColumn("dep_hh", sf.col("CRS_DEP_TIME_str").substr(1, 2).cast("int"))
    .withColumn("dep_mm", sf.col("CRS_DEP_TIME_str").substr(3, 2).cast("int"))
    .withColumn("FL_DATE_str", sf.col("FL_DATE").cast("string"))
    .withColumn(
        "dep_local_ts",
        sf.to_timestamp(
            sf.concat_ws(" ", "FL_DATE_str", sf.format_string("%02d:%02d:00", sf.col("dep_hh"), sf.col("dep_mm"))),
            "yyyy-MM-dd HH:mm:ss"
        )
    )
)

# Display results
# show_df(crs, 5)
# show_columns(crs)


# COMMAND ----------

fl_origin = (
    crs.alias("f")
    .join(df_airports.alias("a"), sf.col("f.ORIGIN")==sf.col("a.iata_code"), "left")
    .withColumn("airport_timezone", sf.col("a.airport_timezone"))                     # expose TZ as a plain column
    .withColumn("prediction_local_ts", sf.expr("dep_local_ts - INTERVAL 2 HOURS"))    # (equiv to T–2h)
    .withColumn(
        "prediction_utc",
        sf.expr("to_utc_timestamp(prediction_local_ts, airport_timezone)")            # use expr() so TZ column works
    )
    .withColumn(
        "flight_id",
        sf.concat_ws("|",
            sf.col("FL_DATE").cast("string"),
            sf.col("OP_UNIQUE_CARRIER"),
            sf.col("OP_CARRIER_FL_NUM"),
            sf.col("ORIGIN"),
            sf.col("DEST")
        )
    )
    .select("flight_id","FL_DATE","ORIGIN","DEST","prediction_local_ts","prediction_utc")
)

# Display results
# show_df(fl_origin, 5)
# show_columns(fl_origin)

# COMMAND ----------

# As-of weather (origin): join candidate stations; filter obs_utc ≤ prediction_utc and within 6h; choose latest by rank/time
origin_candidates = (
    fl_origin.alias("f")
    .join(airport_weather_station.alias("b"), sf.col("f.ORIGIN") == sf.col("b.iata_code"), how="left")
    .select("f.*", sf.col("b.STATION").alias("cand_station"), sf.col("b.rank").alias("station_rank"))
)

# Display results
# show_df(origin_candidates)
# show_columns(origin_candidates)



# COMMAND ----------

# Restrict to weather rows 'as-of' prediction_utc (no future) and within a bounded lookback window 
weather_required = [
    "HourlyDryBulbTemperature","HourlyDewPointTemperature","HourlyWetBulbTemperature",
    "HourlyPrecipitation","HourlyWindSpeed","HourlyWindDirection","HourlyWindGustSpeed",
    "HourlyVisibility","HourlyRelativeHumidity","HourlyStationPressure","HourlySeaLevelPressure",
    "HourlyAltimeterSetting","HourlySkyConditions","HourlyPresentWeatherType"
]
wx_present = [c for c in weather_required if c in weather_best.columns]
weather = weather_best.select(
    "STATION",
    sf.col("obs_utc"),
    *[sf.col(c) for c in wx_present]
)

# weather_cols = ["STATION", sf.col("DATE").cast("timestamp").alias("obs_utc")] + [sf.col(c) for c in wx_present]
# weather = df_weather.select(*weather_cols)

# In origin_candidates, also carry the distance so we can publish origin_station_dis
origin_candidates = (
    fl_origin.alias("f")
    .join(airport_weather_station.alias("b"), sf.col("f.ORIGIN") == sf.col("b.iata_code"), how="left")
    .select("f.*",
            sf.col("b.STATION").alias("cand_station"),
            sf.col("b.rank").alias("station_rank"),
            sf.col("b.dist_km").alias("cand_station_dis_km"))
)

# 6-hour lookback 
weather_join = (
    origin_candidates.alias("x")
    .join(
        weather.alias("w"),
        on=[
            sf.col("w.STATION") == sf.col("x.cand_station"),
            sf.col("w.obs_utc") <= sf.col("x.prediction_utc"),
            sf.col("w.obs_utc") >= sf.expr("timestampadd(HOUR, -6, x.prediction_utc)")
        ],
        how="left"
    )
)

# Display results
# show_df(weather_join, 5)
# show_columns(weather_join)

# COMMAND ----------

show_df(weather_join, 5)

# COMMAND ----------

# Station selection window, prefer lower station_rank (1, then 2, then 3), and the latest obs_utc within the window
window = W.partitionBy("flight_id").orderBy(sf.col("station_rank").asc(), sf.col("obs_utc").desc())
origin_asof = (
    weather_join
    .withColumn("rn", sf.row_number().over(window))
    .filter(sf.col("rn") == 1)
    .withColumn("asof_minutes", sf.floor((sf.unix_timestamp("prediction_utc") - sf.unix_timestamp("obs_utc"))/60.0))
    .select(
        "flight_id","ORIGIN","prediction_utc",
        sf.col("cand_station").alias("origin_station_id"),
        sf.col("cand_station_dis_km").alias("origin_station_dis"),
        sf.col("obs_utc").alias("origin_obs_utc"),
        "asof_minutes",
        *wx_present,
        "station_rank"
    )
)
# Display results
# show_df(origin_asof, 5)
# show_columns(origin_asof)

# COMMAND ----------

# Origin station lat/lon
origin_asof_enriched = (
    origin_asof.alias("o")
    .join(
        df_weather_station.select(
            sf.col("station_id").alias("STATION"),
            sf.col("lat").alias("origin_station_lat"),
            sf.col("lon").alias("origin_station_lon")
        ).alias("s"),
        sf.col("o.origin_station_id")==sf.col("s.STATION"),
        "left"
    )
)

need_from_w = ["flight_id","prediction_utc","origin_obs_utc","asof_minutes",
               "origin_station_id","origin_station_dis","origin_station_lat","origin_station_lon"] + wx_present
origin_asof_enriched = origin_asof_enriched.select(*[c for c in need_from_w if c in origin_asof_enriched.columns])


# Airport lat/lon (origin & dest)
air_min = df_airports.select(
    "iata_code",
    sf.col("lat").alias("airport_lat"),
    sf.col("lon").alias("airport_lon"),
    sf.col("airport_type")
)
origin_air_geo = air_min.select(
    sf.col("iata_code").alias("ORIGIN"),
    sf.col("airport_lat").alias("origin_airport_lat"),
    sf.col("airport_lon").alias("origin_airport_lon"),
    sf.col("airport_type").alias("origin_type")   
)
dest_air_geo = air_min.select(
    sf.col("iata_code").alias("DEST"),
    sf.col("airport_lat").alias("dest_airport_lat"),
    sf.col("airport_lon").alias("dest_airport_lon"),
    sf.col("airport_type").alias("dest_type")     # <-- new
)

# Dest station (rank-1) for location helpers (no dest weather to avoid leakage)
dest_rank1 = (
    airport_weather_station
    .filter(sf.col("rank")==1)
    .select(
        sf.col("iata_code").alias("DEST"),
        sf.col("STATION").alias("dest_station_id"),
        sf.col("dist_km").alias("dest_station_dis")
    )
)
dest_station_geo = (
    dest_rank1.alias("d")
    .join(
        df_weather_station.select(
            sf.col("station_id").alias("STATION"),
            sf.col("lat").alias("dest_station_lat"),
            sf.col("lon").alias("dest_station_lon")
        ).alias("s"),
        sf.col("d.dest_station_id")==sf.col("s.STATION"),
        "left"
    )
    .select("DEST","dest_station_id","dest_station_dis","dest_station_lat","dest_station_lon")
)

# COMMAND ----------

# Rebuild a stable flight_id on the raw flights for the final join
flights_keyed = (
    df_flights
    .withColumn(
        "flight_id",
        sf.concat_ws("|",
            sf.col("FL_DATE").cast("string"),
            sf.col("OP_UNIQUE_CARRIER"),
            sf.col("OP_CARRIER_FL_NUM"),
            sf.col("ORIGIN"),
            sf.col("DEST")
        )
    )
)

# Display results
# show_df(flights_keyed, 5)
# show_columns(flights_keyed)

# COMMAND ----------

# Assemble final dataset: one row per flight
final_joined = (
    flights_keyed.alias("f")
    .join(origin_asof_enriched.alias("w"), "flight_id", "left")
    .join(origin_air_geo, ["ORIGIN"], "left")
    .join(dest_air_geo,   ["DEST"],   "left")
    .join(dest_station_geo, ["DEST"], "left")
)

# Column groups
model_inputs = [
    # Flight & schedule
    "FL_DATE","YEAR","QUARTER","MONTH","DAY_OF_MONTH","DAY_OF_WEEK",
    "OP_UNIQUE_CARRIER","OP_CARRIER","OP_CARRIER_FL_NUM","TAIL_NUM",
    "CRS_DEP_TIME","CRS_ARR_TIME","CRS_ELAPSED_TIME",
    "ORIGIN","ORIGIN_AIRPORT_ID","ORIGIN_CITY_NAME","ORIGIN_STATE_ABR",
    "DEST","DEST_AIRPORT_ID","DEST_CITY_NAME","DEST_STATE_ABR",
    "DISTANCE","DISTANCE_GROUP",
    # Origin weather (as-of T–2h)
] + wx_present + [
    # Location helpers
    "origin_station_lat","origin_station_lon","origin_airport_lat","origin_airport_lon",
    "dest_station_lat","dest_station_lon","dest_airport_lat","dest_airport_lon",
    "origin_station_dis","dest_station_dis", "origin_type","dest_type"
]

labels_eval = ["DEP_DEL15","DEP_DELAY","ARR_DEL15","ARR_DELAY"]
post_flight = [
    "CARRIER_DELAY","WEATHER_DELAY","NAS_DELAY","SECURITY_DELAY","LATE_AIRCRAFT_DELAY",
    "DEP_TIME","ARR_TIME","TAXI_OUT","TAXI_IN","WHEELS_OFF","WHEELS_ON","ACTUAL_ELAPSED_TIME","AIR_TIME"
]
flags = ["CANCELLED","CANCELLATION_CODE","DIVERTED"]

provenance = ["flight_id","prediction_utc","origin_obs_utc","asof_minutes","origin_station_id","dest_station_id"]

# Keep only columns that exist (different weather slices may miss some)
def present(cols, df_cols): 
    s = set(df_cols); 
    return [c for c in cols if c in s]

keep = provenance \
     + present(model_inputs, final_joined.columns) \
     + present(labels_eval, final_joined.columns) \
     + present(post_flight, final_joined.columns) \
     + present(flags, final_joined.columns)

final_curated = final_joined.select(*keep)


# Optional: limit to an end date if configured
max_year = cfg.get("max_year")
max_month = cfg.get("max_month")

if max_year is not None:
    if max_month is None:
        max_month = 12
    final_curated = final_curated.filter(
        (sf.col("YEAR") < max_year)
        | ((sf.col("YEAR") == max_year) & (sf.col("MONTH") <= max_month))
    )

# Persist for the team using the configured output path
out_path = cfg["joined_output"]
(final_curated
 .write
 .format("parquet")
 .mode("overwrite")
 .save(out_path))

print(f"Wrote joined dataset to: {out_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1Y Join from 5Y Data

# COMMAND ----------

# Build a 1Y (2015) subset only when running the 5Y config
if DATA_SLICE == "5Y":
    df_joined_5Y = spark.read.parquet(DATA_CONFIG["5Y"]["joined_output"])
    print("Source row count (5Y):", df_joined_5Y.count())

    df_2015 = df_joined_5Y.filter(sf.col("YEAR") == 2015)
    print("Destination row count (1Y 2015):", df_2015.count())

    df_2015.write.mode("overwrite").parquet("dbfs:/student-groups/Group_4_4/JOINED_1Y_2015.parquet")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get stats

# COMMAND ----------

# RUN STATS: summary of this join job

import pandas as pd

job_end = datetime.now()
runtime_seconds = (job_end - job_start).total_seconds()
runtime_minutes = round(runtime_seconds / 60, 2)

print(f"Job end:   {job_end.isoformat()}")
print(f"Runtime:   {runtime_seconds:.1f} seconds ({runtime_minutes} minutes)")

# --- Dataset-level stats (final_curated) ---

# Number of samples and features
row_count = final_curated.count()              # triggers a Spark job
feature_count = len(final_curated.columns)

# Date coverage (uses FL_DATE or change to YEAR/MONTH if you prefer)
date_bounds = final_curated.agg(
    sf.min("FL_DATE").alias("min_date"),
    sf.max("FL_DATE").alias("max_date")
).collect()[0]

data_start_date = date_bounds["min_date"]
data_end_date = date_bounds["max_date"]

# --- File sizes (inputs/outputs) ---

INPUT_PATHS = {
    "flights": cfg["flights_path"],
    "weather": cfg["weather_path"],
    # You can add more here if you want:
    # "airport_codes": "dbfs:/path/to/airport-codes.csv",
    # "airports_with_tz": "dbfs:/path/to/airports.csv",
    # "stations": "dbfs:/path/to/stations_with_neighbors.parquet",
}

OUTPUT_PATHS = {
    "joined_output": out_path,   # out_path was used when writing final_curated
}

file_rows = []

for name, path in INPUT_PATHS.items():
    size_bytes = get_dir_size(path)
    file_rows.append({
        "role": "input",
        "name": name,
        "path": path,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    })

for name, path in OUTPUT_PATHS.items():
    size_bytes = get_dir_size(path)
    file_rows.append({
        "role": "output",
        "name": name,
        "path": path,
        "size_mb": round(size_bytes / (1024 * 1024), 2),
    })

files_df = pd.DataFrame(file_rows)

# --- High-level run summary ---

summary_rows = [
    {"metric": "data_slice",        "value": DATA_SLICE},
    {"metric": "job_start",         "value": job_start.isoformat()},
    {"metric": "job_end",           "value": job_end.isoformat()},
    {"metric": "runtime_minutes",   "value": runtime_minutes},
    {"metric": "row_count",         "value": row_count},
    {"metric": "feature_count",     "value": feature_count},
    {"metric": "data_start_date",   "value": str(data_start_date)},
    {"metric": "data_end_date",     "value": str(data_end_date)},
]

summary_df = pd.DataFrame(summary_rows)

display(summary_df)
display(files_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Preview

# COMMAND ----------

df_check = spark.read.parquet(out_path)
display(df_check.limit(10))