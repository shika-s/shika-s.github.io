library(tidyverse)
library(here)


#Using train.csv because it has SalePrice
full_data_filtered = read.csv(here("data/interim/full_data_filtered.csv"))


head(full_data_filtered)

full_data_filtered <- full_data_filtered %>%
  mutate(SeasonSold = case_when(
    MoSold %in% c(12, 1, 2) ~ "winter",
    MoSold %in% c(3, 4, 5) ~ "spring",
    MoSold %in% c(6, 7, 8) ~ "summer",
    MoSold %in% c(9, 10, 11) ~ "fall",
    TRUE ~ NA_character_
  ))

full_data_filtered$SeasonSold <- factor(full_data_filtered$SeasonSold)


full_data_filtered <- full_data_filtered %>%
  mutate(ZoneType = case_when(
    MSZoning %in% c('C', 'RH') ~ "Cluster",
    MSZoning %in% c('RM', 'FV') ~ "Normal",
    MSZoning %in% c('RL') ~ "Sparse",
    TRUE ~ NA_character_
  ))

full_data_filtered$ZoneType <- factor(full_data_filtered$ZoneType)

full_data_filtered <- full_data_filtered %>%
  mutate(Age = 2010 - YearRemodAdd 
  )

#dummy(basis) category variable is low automatically
full_data_filtered$OverallQualCategory <- cut(full_data_filtered$OverallQual, 
                                      breaks = c(0, 4, 7, 10),
                                      labels = c("low", "med", "high"), 
                                      right = TRUE)  # Ensures that the upper boundary of each interval is included

full_data_filtered$OverallCondCategory <- cut(full_data_filtered$OverallCond, 
                                      breaks = c(0, 4, 7, 10),
                                      labels = c("low", "med", "high"), 
                                      right = TRUE)

full_data_filtered$TotalBsmtSFLogged <- ifelse(full_data_filtered$TotalBsmtSF > 0,
                                       log(full_data_filtered$TotalBsmtSF),
                                       0)


#Transform neighborhood into location
full_data_filtered <- full_data_filtered %>%
  mutate(NeighborhoodGroup = case_when(
    Neighborhood %in% c("NAmes", "NWAmes", "Gilbert", "SawyerW", "Veenker") ~ "North",
    Neighborhood %in% c("CollgCr", "StoneBr", "Somerst", "Timber", "ClearCr", "Blmngtn") ~ "West",
    Neighborhood %in% c("OldTown", "Edwards", "SWISU", "Crawfor", "BrkSide") ~ "Central",
    Neighborhood %in% c("IDOTRR", "MeadowV", "NPkVill", "BrDale") ~ "East",
    Neighborhood %in% c("Mitchel", "Sawyer", "Blueste", "NoRidge", "NridgHt") ~ "South",
    TRUE ~ NA_character_
  ))
full_data_filtered$NeighborhoodGroup <- factor(full_data_filtered$NeighborhoodGroup)

head(full_data_filtered)


###split between exploratory and confirmation set 30:70
set.seed(100)  # for reproducibility

# Total number of rows
n <- nrow(full_data_filtered)

# Sample 30% of the indices for exploration
explore_indices <- sample(1:n, size = floor(0.3 * n))

# Split the data
explore_data <- full_data_filtered[explore_indices, ]
confirm_data  <- full_data_filtered[-explore_indices, ]

write.csv(explore_data, here("data/processed/explore_data.csv") )
write.csv(confirm_data, here("data/processed/confirm_data.csv") )
