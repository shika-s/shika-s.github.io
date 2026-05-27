library(tidyverse)
library(here)


#Using train.csv because it has SalePrice
full_data = read.csv(here("data/external/train.csv"))
full_data_rows <- nrow(full_data)

filtered_by_year <- full_data %>%
  filter(YrSold %in% c(2006,2007))
filtered_by_year_rows <- nrow(filtered_by_year)

full_data_filtered <- filtered_by_year %>%
  filter(BldgType == "1Fam")
full_data_filtered_rows <- nrow(full_data_filtered)


#Data Cleaning (no na data found)
sum(is.na(full_data_filtered$GrLivArea))
sum(is.na(full_data_filtered$Neighborhood))
sum(is.na(full_data_filtered$MoSale))
sum(is.na(full_data_filtered$Bedroom))
sum(is.na(full_data_filtered$FullBath))
sum(is.na(full_data_filtered$HalfBath))
sum(is.na(full_data_filtered$BsmtFullBath))
sum(is.na(full_data_filtered$BsmtHalfBath))
sum(is.na(full_data_filtered$TotalBsmtSF))
sum(is.na(full_data_filtered$YearRemodAdd))
sum(is.na(full_data_filtered$SalePrice))
sum(is.na(full_data_filtered$LotArea))
sum(is.na(full_data_filtered$SeasonSold))
sum(is.na(full_data_filtered$MSZoning))
sum(is.na(full_data_filtered$Neighborhood))
sum(is.na(full_data_filtered$OverallCond))
sum(is.na(full_data_filtered$OverallQual))
sum(is.na(full_data_filtered$YearBuilt))
sum(is.na(full_data_filtered$KitchenQual))



write.csv(full_data_filtered, here("data/interim/full_data_filtered.csv") )

