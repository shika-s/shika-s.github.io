library(ggplot2)
library(tidyverse)
library(car)
library(caret) #shows prediction results good (will need to do install.packages("caret"))
library(reshape2)
library(lmtest)
library(stargazer)
library(here)
library(sandwich)

#Read processed exploratory dataset
train_data = read.csv(here("data/processed/explore_data.csv"))
test_data = read.csv(here("data/processed/confirm_data.csv"))

###Initial Model
initial_model <- lm(SalePrice ~ GrLivArea, data = train_data)
summary(initial_model)


###Now using test_data for predictions/verification
#Analysis plots for Initial Model
par(mfrow = c(2, 2))
plot(initial_model)
# Predict using the test_data
predictions_initial_model <- predict(initial_model, newdata = test_data)
test_data$SalePrice

# High-quality base R plot
plot(test_data$SalePrice, predictions_initial_model,
     xlab = "Actual SalePrice",
     ylab = "Predicted SalePrice",
     main = "Actual vs Predicted SalePrice",
     pch = 16,           # solid circle
     cex = 0.6,          # smaller point size
     col = rgb(0.2, 0.4, 0.8, 0.6))  # semi-transparent blue

# reference line
abline(a = 0, b = 1, col = "red", lwd = 2, lty = 2)


###Residual plot on test set
# Calculate residuals
residuals <- test_data$SalePrice - predictions_initial_model

# Plot residuals vs predicted
plot(predictions_initial_model, residuals,
     xlab = "Predicted SalePrice",
     ylab = "Residuals",
     main = "Residuals vs Predicted (Test Set Initial Model)",
     pch = 16,
     cex = 0.6,
     col = rgb(0.8, 0.2, 0.2, 0.6))

abline(h = 0, col = "darkgray", lty = 2)
###Q-Q plot on test set
qqnorm(residuals,
       main = "Q-Q Plot of Residuals (Test Set Initial Model)",
       pch = 16,
       col = "steelblue")
qqline(residuals, col = "red", lwd = 2)

#######Multivariable Regression
###Model 1: GrLivArea (re-doing for organization)
model_1_GrLivArea <- lm(SalePrice ~ GrLivArea, data = train_data)
summary(model_1_GrLivArea)
# summary with robust standard errors
coeftest(model_1_GrLivArea, vcov = vcovHC(model_1_GrLivArea, type = "HC1"))
par(mfrow = c(2, 2))
plot(model_1_GrLivArea)

###Model 1: GrLivArea
model_1_GrLivArea <- lm(SalePrice ~ GrLivArea, data = train_data)
summary(model_1_GrLivArea)
par(mfrow = c(2, 2))
plot(model_1_GrLivArea)
bptest(model_1_GrLivArea) #Breusch-Pagan Test

###Model 1b: sqrtGrLivArea
model_1b_GrLivArea <- lm(SalePrice ~ sqrt(GrLivArea), data = train_data)
summary(model_1b_GrLivArea)
par(mfrow = c(2, 2))
plot(model_1b_GrLivArea)
bptest(model_1b_GrLivArea) #Breusch-Pagan Test


###Model 1c: GrLivArea
model_1c_GrLivArea <- lm(log10(SalePrice) ~ GrLivArea, data = train_data)
summary(model_1c_GrLivArea)
par(mfrow = c(2, 2))
plot(model_1c_GrLivArea)
bptest(model_1c_GrLivArea) #Breusch-Pagan Test

###MODEL 2: Addition of BedroomAbvGr variable
model_2_BedroomAbvGr <- lm(SalePrice ~ GrLivArea + BedroomAbvGr, data = train_data)
summary(model_2_BedroomAbvGr)
vif(model_2_BedroomAbvGr)
par(mfrow = c(2, 2))
plot(model_2_BedroomAbvGr)
bptest(model_2_BedroomAbvGr)






###MODEL 3: Addition of LotArea variable and removal of BedroomAbvGr
model_3_LotArea <- lm(SalePrice ~ GrLivArea + LotArea, data = train_data)
summary(model_3_LotArea)
vif(model_3_LotArea)
par(mfrow = c(2, 2))
plot(model_3_LotArea)

###MODEL 4: Addition of FullBath variable and removal of LotArea
model_4_FullBath <- lm(SalePrice ~ GrLivArea + FullBath, data = train_data)
summary(model_4_FullBath)
vif(model_4_FullBath)
par(mfrow = c(2, 2))
plot(model_4_FullBath)


###MODEL 5: Addition of SeasonSold variable
model_5_SeasonSold <- lm(SalePrice ~ GrLivArea + FullBath + SeasonSold, data = train_data)
summary(model_5_SeasonSold)
vif(model_5_SeasonSold)
par(mfrow = c(2, 2))
plot(model_5_SeasonSold)
#####season is not statistically significant. Will not include for 
#further models

###MODEL 6: Addition of TotalBsmtSF variable and removal of SeasonSold
model_6_TotalBsmtSF <- lm(SalePrice ~ GrLivArea + FullBath + TotalBsmtSF, data = train_data)
summary(model_6_TotalBsmtSF)
vif(model_6_TotalBsmtSF)
par(mfrow = c(2, 2))
plot(model_6_TotalBsmtSF)
model_6_predictions <- predict(model_6_TotalBsmtSF, newdata = test_data)
postResample(pred = model_6_predictions, obs = test_data$SalePrice) #eval for model_6

###MODEL 7: Addition of YearBuilt variable
model_7_YearBuilt <- lm(SalePrice ~ GrLivArea + FullBath + TotalBsmtSF + YearBuilt, data = train_data)
summary(model_7_YearBuilt)
vif(model_7_YearBuilt)
par(mfrow = c(2, 2))
plot(model_7_YearBuilt)
model_7_predictions <- predict(model_7_YearBuilt, newdata = test_data)
postResample(pred = model_7_predictions, obs = test_data$SalePrice) #eval for model_7

###MODEL 8: Removal of FullBath variable
model_8 <- lm(SalePrice ~ GrLivArea + TotalBsmtSF + YearBuilt, data = train_data)
summary(model_8)
vif(model_8)
par(mfrow = c(2, 2))
plot(model_8)
model_8_predictions <- predict(model_8, newdata = test_data)
postResample(pred = model_8_predictions, obs = test_data$SalePrice) #eval for model_8

###MODEL 9: Addition of NeighborhoodGroup variable
model_9_NeighborhoodGroup <- lm(SalePrice ~ GrLivArea + TotalBsmtSF + YearBuilt + NeighborhoodGroup, data = train_data)
summary(model_9_NeighborhoodGroup)
vif(model_9_NeighborhoodGroup)
par(mfrow = c(2, 2))
plot(model_9_NeighborhoodGroup)

###MODEL 10: Addition of GarageCars variable and removal of NeighborhoodGroup
model_10_GarageCars <- lm(SalePrice ~ GrLivArea + TotalBsmtSF + YearBuilt + GarageCars, data = train_data)
summary(model_10_GarageCars)
vif(model_10_GarageCars)
par(mfrow = c(2, 2))
plot(model_10_GarageCars)
model_10_predictions <- predict(model_10_GarageCars, newdata = test_data)
postResample(pred = model_10_predictions, obs = test_data$SalePrice) #eval for model_10

###MODEL 11: Addition of KitchenQual variable
model_11_KitchenQual <- lm(SalePrice ~ GrLivArea + TotalBsmtSF + YearBuilt + GarageCars + KitchenQual, data = train_data)
summary(model_11_KitchenQual)
vif(model_11_KitchenQual)
par(mfrow = c(2, 2))
plot(model_11_KitchenQual)
model_11_predictions <- predict(model_11_KitchenQual, newdata = test_data)
postResample(pred = model_11_predictions, obs = test_data$SalePrice) #eval for model_11

###MODEL 12: Transforming GrLivArea -> sqrt(GrLivArea)
model_12_sqrtGrLivArea <- lm(SalePrice ~ sqrt(GrLivArea) + TotalBsmtSF + YearBuilt + GarageCars + KitchenQual, data = train_data)
summary(model_12_sqrtGrLivArea)
vif(model_12_sqrtGrLivArea)
par(mfrow = c(2, 2))
plot(model_12_sqrtGrLivArea)


###MODEL 13: Transforming TotalBsmtSF -> sqrt(TotalBsmtSF)
model_13_sqrtGrLivArea <- lm(SalePrice ~ sqrt(GrLivArea) + sqrt(TotalBsmtSF) + YearBuilt + GarageCars + KitchenQual, data = train_data)
summary(model_13_sqrtGrLivArea)
vif(model_13_sqrtGrLivArea)
par(mfrow = c(2, 2))
plot(model_13_sqrtGrLivArea)

###MODEL 14: Transforming SalePrice -> log10(SalePrice)
model_14_logSalePrice <- lm(log10(SalePrice) ~ sqrt(GrLivArea) + sqrt(TotalBsmtSF) + YearBuilt + GarageCars + KitchenQual, data = train_data)
summary(model_14_logSalePrice)
vif(model_14_logSalePrice)
par(mfrow = c(2, 2))
plot(model_14_logSalePrice)
###log transforming SalePrice made KitchenQual and GarageCars insignificant

###MODEL 15: Removing KitchenQual and GarageCars
model_15_RemovingKG <- lm(log10(SalePrice) ~ sqrt(GrLivArea) + sqrt(TotalBsmtSF) + YearBuilt, data = train_data)
summary(model_15_RemovingKG)
vif(model_15_RemovingKG)
par(mfrow = c(2, 2))
plot(model_15_RemovingKG)

###MODEL 16: Adding OverallQual Linearly
model_16_OverallQual <- lm(log10(SalePrice) ~ sqrt(GrLivArea) + sqrt(TotalBsmtSF) + YearBuilt + OverallQual, data = train_data)
summary(model_16_OverallQual)
vif(model_16_OverallQual)
par(mfrow = c(2, 2))
plot(model_16_OverallQual)
bptest(model_16_OverallQual)

###MODEL 17: Adding OverallCond Linearly
model_17_OverallCond <- lm(log10(SalePrice) ~ sqrt(GrLivArea) + sqrt(TotalBsmtSF) + YearBuilt + OverallQual + OverallCond, data = train_data)
summary(model_17_OverallCond)
vif(model_17_OverallCond)
par(mfrow = c(2, 2))
plot(model_17_OverallCond)
bptest(model_17_OverallCond)

###MODEL 18: Adding OverallQual OverallCond categorically, changing sqrt->log transforms
#dummy(basis) category variable is low automatically
train_data$OverallQualCategory <- cut(train_data$OverallQual, 
                       breaks = c(0, 4, 7, 10),
                       labels = c("low", "med", "high"), 
                       right = TRUE)  # Ensures that the upper boundary of each interval is included
test_data$OverallQualCategory <- cut(test_data$OverallQual, 
                                     breaks = c(0, 4, 7, 10),
                                     labels = c("low", "med", "high"), 
                                     right = TRUE)  # Ensures that the upper boundary of each interval is included

train_data$OverallCondCategory <- cut(train_data$OverallCond, 
                                      breaks = c(0, 4, 7, 10),
                                      labels = c("low", "med", "high"), 
                                      right = TRUE)
test_data$OverallCondCategory <- cut(test_data$OverallCond, 
                                      breaks = c(0, 4, 7, 10),
                                      labels = c("low", "med", "high"), 
                                      right = TRUE)

train_data$TotalBsmtSFLogged <- ifelse(train_data$TotalBsmtSF > 0,
                     log(train_data$TotalBsmtSF),
                     0)
test_data$TotalBsmtSFLogged <- ifelse(test_data$TotalBsmtSF > 0,
                                       log(test_data$TotalBsmtSF),
                                       0)

model_18_CatLog <- lm(log10(SalePrice) ~ log10(GrLivArea) + TotalBsmtSFLogged + YearBuilt + OverallQualCategory + OverallCondCategory, data = train_data)
summary(model_18_CatLog)
# summary with robust standard errors
coeftest(model_18_CatLog, vcov = vcovHC(model_18_CatLog, type = "HC1"))
vif(model_18_CatLog)
par(mfrow = c(2, 2))
plot(model_18_CatLog)
bptest(model_18_CatLog)
model_18_predictions <- predict(model_18_CatLog, newdata = test_data)
postResample(pred = model_18_predictions, obs = log10(test_data$SalePrice))


###Taking a look at correlations
predictors <- train_data %>% select_if(is.numeric) %>% select(-SalePrice)
cor_matrix <- cor(predictors, use = "complete.obs")
diag(cor_matrix) <- 0
cor_matrix[lower.tri(cor_matrix, diag = TRUE)] <- NA

var_pairs <- which(cor_matrix > 0.7, arr.ind = TRUE)
results <- data.frame(
  var1 = rownames(cor_matrix)[var_pairs[, 1]],
  var2 = colnames(cor_matrix)[var_pairs[, 2]],
  correlation = cor_matrix[var_pairs]
)

print(results)

###Intermediate Model
model_intermediate <- lm(log10(SalePrice) ~ log10(GrLivArea) + OverallQualCategory, data = train_data)
summary(model_intermediate)
# summary with robust standard errors
coeftest(model_intermediate, vcov = vcovHC(model_intermediate, type = "HC1"))
vif(model_intermediate)
par(mfrow = c(2, 2))
plot(model_intermediate)
bptest(model_intermediate)

###STARGAZER FOR FIRST/INTERMEDIATELAST MODEL (model_1_GrLivArea/model_intermediate/model_18_CatLog)
# generating robust standard errors for each model
robust_se_1c <- vcovHC(model_1c_GrLivArea, type = "HC1")
robust_se_intermediate <- vcovHC(model_intermediate, type = "HC1")
robust_se_18_CatLog <- vcovHC(model_18_CatLog, type = "HC1")

stargazer(model_1c_GrLivArea, model_intermediate, model_18_CatLog, type = "latex", out = "stargazer_output.tex",
          title = "Comparison of Regression Models",
          dep.var.labels = "Log10(SalePrice)",
          column.labels = c("Initial Model", "Intermediate Model", "Advanced Model"),
          se = list(sqrt(diag(robust_se_1c)),
                    sqrt(diag(robust_se_intermediate)),
                    sqrt(diag(robust_se_18_CatLog))))





