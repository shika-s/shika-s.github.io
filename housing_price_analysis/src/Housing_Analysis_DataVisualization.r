library(ggplot2)
library(tidyverse)
library(here)
library(reshape2)


#Read processed exploratory dataset
explore_data = read.csv(here("data/processed/explore_data.csv"))

head(explore_data)

summary(explore_data$SalePrice)


#Sale Price Distribution
explore_data %>%
  ggplot() +
  aes(x=SalePrice/1000) +
  geom_histogram(aes(y = after_stat(density)),
               colour = 1, fill = "white", binwidth=20) +
  geom_density()+
  labs(
    x = "Sale Price($1000)") 


#GrLivArea distribution
explore_data %>%
  ggplot() +
  aes(x=GrLivArea) +
  geom_histogram(aes(y = after_stat(density)),
                 colour = 1, fill = "white", binwidth=20) +
  geom_density()

#Season of Sale Distribution
explore_data %>%
  ggplot() +
  aes(x=SeasonSold) +
  geom_bar() +
  geom_density()

# Age of House/Reconstruction Distribution
explore_data %>%
  ggplot() +
  aes(x=Age) +
  geom_bar() +
  geom_density()

#Neighborhood distribution
explore_data %>%
  ggplot() +
  aes(x=Neighborhood) +
  geom_bar() +
  theme(axis.text.x = element_text(angle=90, hjust=1))

#NeighborhoodGroup distribution
explore_data %>%
  ggplot() +
  aes(x=NeighborhoodGroup) +
  geom_bar() +
  theme(axis.text.x = element_text(angle=90, hjust=1))

#Lot Area Distribution
explore_data %>%
  ggplot() +
  aes(x=LotArea/100) +
  geom_histogram(aes(y = after_stat(density)),
                 colour = 1, fill = "white", binwidth=20) +
  geom_density()

#MSZoning Distribution
explore_data %>%
  ggplot() +
  aes(x=MSZoning) +
  geom_bar() 

#Condition1 distribution
explore_data %>%
  ggplot() +
  aes(x=Condition1) +
  geom_bar() 

#Condition1 distribution
explore_data %>%
  ggplot() +
  aes(x=Condition2) +
  geom_bar() 

#MSSubClass distribution
explore_data %>%
  ggplot() +
  aes(x=MSSubClass) +
  geom_bar() 

#Joint Distribution of Seasonal Sale Price and Above Ground Living Area   
explore_data %>%
  ggplot() +
  aes(x = GrLivArea, y = SalePrice, color = SeasonSold) + 
  geom_point() + 
  #scale_color_discrete(breaks=c("Spring","Summer","Fall","Winter"))+
  labs(
    x = "Above Ground Living Area (Sq. Ft.)", 
    y = "Sale Price($)",
    color= "Season of Sale") 


#Joint distribution of Season of Sale and Sale Price
explore_data %>%
  ggplot() +
  aes(x = SeasonSold, y = SalePrice/1000) + 
  geom_boxplot() + 
  labs(
    x = "Season of Sale", 
    y = "SalePrice($100k)") 


#Distribution of SalePrice/GrLivArea and Season of Sale
explore_data %>%
  ggplot() +
  aes(x = SeasonSold, y = SalePrice/GrLivArea) + 
  geom_boxplot() + 
  labs(
    x = "Season of Sale", 
    y = "SalePrice/GrLivArea") 



#Distribution of SalePrice/GrLivArea and Neighborhood
explore_data %>%
  ggplot() +
  aes(x =SalePrice/GrLivArea, y =Neighborhood ) + 
  geom_boxplot() + 
  labs(
    y = "Neighborhood", 
    x = "SalePrice/GrLivArea") 


#Distribution of SalePrice and Neighborhood
explore_data %>%
  ggplot() +
  aes(x =SalePrice/1000, y =Neighborhood ) + 
  geom_boxplot() + 
  labs(
    y = "Neighborhood", 
    x = "SalePrice($100k)") 


#Distribution of SalePrice and Years since  Remodelling/Built
explore_data %>%
  ggplot() +
  aes(x = Age, y = SalePrice/1000 ) + 
  geom_point() + 
  labs(
    x = "Years since  Remodelling/Built", 
    y = "SalePrice") 




#Distribution of SalePrice with Number of Bedrooms Above Ground
explore_data %>%
  ggplot() +
  aes(x = BedroomAbvGr, y = SalePrice/1000 ) + 
  geom_point() + 
  labs(
    x = "BedroomAbvGr", 
    y = "SalePrice") 

#Distribution of Sale Price and Lot Area
explore_data %>%
  ggplot() +
  aes(x = log(LotArea), y = SalePrice/1000 ) + 
  geom_point() + 
  labs(
    x = "LotArea", 
    y = "SalePrice") 


#Distribution of Lot Area and Above Ground Living Area
explore_data %>%
  ggplot() +
  aes(x = log(LotArea), y = log(GrLivArea) ) + 
  geom_point() + 
  labs(
    x = "LotArea", 
    y = "GrLivArea") 


#Distribution of Sale Price and # of FullBath
explore_data %>%
  ggplot() +
  aes(x = FullBath, y = SalePrice ) + 
  geom_point() + 
  labs(
    x = "FullBath", 
    y = "SalePrice") 



#Distribution of # Full Baths and # Bedrooms Above Ground
explore_data %>%
  ggplot() +
  aes(x = FullBath, y = BedroomAbvGr ) + 
  geom_point() + 
  labs(
    x = "FullBath", 
    y = "BedroomAbvGr") 


#Joint Distribution of Sale Price, Neighborhood and Zoning Type
explore_data %>%
  ggplot() +
  aes(y = Neighborhood, x = SalePrice, color = ZoneType ) + 
  geom_point() + 
  labs(
    y = "Neighborhood", 
    x = "SalePrice",
    color= "ZoneType") 

#Joint Distribution of Sale Price, Neighborhood and Zoning Type
explore_data %>%
  ggplot() +
  aes(y = NeighborhoodGroup, x = SalePrice, color = ZoneType ) + 
  geom_point() + 
  labs(
    y = "NeighborhoodGroup", 
    x = "SalePrice",
    color= "ZoneType") 

#Joint Distribution of Sale Price, Neighborhood and Proximity
explore_data %>%
  ggplot() +
  aes(y = Neighborhood, x = SalePrice, color = Condition1 ) + 
  geom_point() + 
  labs(
    y = "Neighborhood", 
    x = "SalePrice",
    color= "Condition1") 

#Distribution of Neighborhood, Zone type and Proximity to Various Conditions
explore_data %>%
  ggplot() +
  aes(y = Neighborhood, x = ZoneType, color = Condition1 ) + 
  geom_point() + 
  labs(
    y = "Neighborhood", 
    x = "ZoneType",
    color= "Condition1") 


#Distribution of Neighborhood, Zone type and Proximity to Various Conditions
explore_data %>%
  ggplot() +
  aes(y = SalePrice, x = MSSubClass, color=ZoneType ) + 
  geom_point() + 
  labs(
    y = "SalePrice", 
    x = "MSSubClass",
    color= "ZoneType") 

#Distribution of SalePrice, OverallCond and KitchenQual
explore_data %>%
  ggplot() +
  aes(y = SalePrice, x = OverallCond, color=KitchenQual ) + 
  geom_point() + 
  labs(
    y = "SalePrice", 
    x = "OverallCond",
    color= "KitchenQual") 

#Distribution of SalePrice, OverallQual and GarageCars
explore_data %>%
  ggplot() +
  aes(y = SalePrice, x = OverallQual, color=GarageCars ) + 
  geom_point() + 
  labs(
    y = "Sale Price", 
    x = "Overall Quality",
    color= "#Car Garage") 



#Distribution of SalePrice, TotalBsmtSF and FullBath
explore_data %>%
  ggplot() +
  aes(y = SalePrice, x = TotalBsmtSF, color=FullBath ) + 
  geom_point() + 
  labs(
    y = "Sale Price", 
    x = "Total Basement SF",
    color= "#Full Bath") 



#Correlations
focus_data <- data.frame(explore_data$SalePrice,
                         explore_data$GrLivArea, 
                         explore_data$Age, 
                         explore_data$SeasonSold,
                         explore_data$ZoneType,
                         explore_data$LotArea,
                         explore_data$FullBath,
                         explore_data$BedroomAbvGr,
                         explore_data$NeighborhoodGroup,
                         explore_data$TotalBsmtSF,
                         explore_data$GarageCars,
                         explore_data$KitchenQual,
                         explore_data$YearBuilt,
                         explore_data$OverallCond,
                         explore_data$OverallQual)

names <- c("SalePrice", "GrLivArea", "Age", "SeasonSold", "ZoneType", "LotArea", "FullBath",
          "Bedroom", "Location", "TotalBsmtSF", "GarageCars","KitchenQual",
          "YearBuilt", "OverallCond","OverallQual")
colnames(focus_data) <- names



# dummify the data
dmy <- dummyVars(" ~ .", data = focus_data)
trsf <- data.frame(predict(dmy, newdata = focus_data))
trsf

#corelation matrix
cormat <- round(cor(trsf), 6)
head(cormat)



# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}



# Melt the correlation matrix
upper_tri <- get_upper_tri(cormat)
upper_tri
melted_cormat <- melt(upper_tri, na.rm = TRUE)
head(melted_cormat)


# Heatmap
heatmap_plot <- ggplot(data = melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ 
  theme(axis.text.x = element_text(angle = 90, vjust = 1, 
                                   size = 8, hjust = 1))+
  coord_fixed()+
  labs(title= "Correlation Matrix for Variables",
       x="",
       y="")





