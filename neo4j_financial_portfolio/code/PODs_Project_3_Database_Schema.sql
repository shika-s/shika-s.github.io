CREATE TABLE "pods" (
  "pod_id" varchar PRIMARY KEY,
  "pod_name" varchar,
  "pod_asset_class_focus" varchar,
  "pod_geo_focus" varchar,
  "pod_inception_date" datetime
);

CREATE TABLE "assets" (
  "asset_id" varchar PRIMARY KEY,
  "asset_name" varchar,
  "asset_class" varchar,
  "country_of_risk" varchar,
  "sec_type" varchar
);

CREATE TABLE "pod_nav_history" (
  "Dates" datetime,
  "pod_id_time_series" float
);

CREATE TABLE "asset_factor_exposures" (
  "asset_id" varchar PRIMARY KEY,
  "rate_coef" float,
  "equity_coef" float,
  "credit_coef" float,
  "fx_coef" float,
  "inflation_coef" float
);

CREATE TABLE "pods_current_allocation" (
  "pod_id" varchar PRIMARY KEY,
  "asset_id" varchar,
  "weight" float
);

COMMENT ON COLUMN "pod_nav_history"."Dates" IS 'cant display time series data for pod_id';

COMMENT ON COLUMN "pod_nav_history"."pod_id_time_series" IS 'variable doesnt exist this way, just in here for demostrantion purposes';

ALTER TABLE "pods" ADD FOREIGN KEY ("pod_id") REFERENCES "pods_current_allocation" ("pod_id");

ALTER TABLE "asset_factor_exposures" ADD FOREIGN KEY ("asset_id") REFERENCES "assets" ("asset_id");

ALTER TABLE "asset_factor_exposures" ADD FOREIGN KEY ("asset_id") REFERENCES "pods_current_allocation" ("asset_id");

ALTER TABLE "pods_current_allocation" ADD FOREIGN KEY ("asset_id") REFERENCES "assets" ("asset_id");
