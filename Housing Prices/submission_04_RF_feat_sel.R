#---------------------------------------------------------------------------------------------------
#
# Housing Prices - Advanced Regression Techniques
#
#---------------------------------------------------------------------------------------------------
#
# Kaggle Playground Competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques
#
#---------------------------------------------------------------------------------------------------

# Load Libraries
library(tidyverse)
library(caret)
library(randomForest)

#---------------------------------------------------------------------------------------------------
# Load Data
#---------------------------------------------------------------------------------------------------

setwd("C:\\Users\\Steffen\\SkyDrive\\Files\\Data\\Development\\Data Science\\Kaggle\\Housing Prices")
train.raw <- read.csv("train.csv")
test.raw <- read.csv("test.csv")

#---------------------------------------------------------------------------------------------------
# Split the Data
#---------------------------------------------------------------------------------------------------

trainIndex <- createDataPartition(train.raw$Id, p = 0.75, list = FALSE)
t1 <- train.raw[trainIndex,]
v1 <- train.raw[-trainIndex,]


#---------------------------------------------------------------------------------------------------
# Data Exploration...
#---------------------------------------------------------------------------------------------------

# Subset - only get integer columns
nums <- sapply(t1, is.integer)
t2 <- t1[, nums]
t2$Id <- NULL

nums <- sapply(v1, is.integer)
v2 <- v1[, nums]
v2$Id <- NULL

nums <- sapply(test.raw, is.integer)
tt <- test.raw[, nums]
tt_id <- tt$Id
tt$Id <- NULL

summary(t2)
summary(v2)
summary(tt)

# Deal with NA's: create histograms first - 157 NA's
# LotFrontage: Linear feet of street connected to property
# ggplot(t2, aes(x = LotFrontage)) + geom_histogram()
# ggplot(t2, aes(x = LotFrontage, y = SalePrice)) + geom_point()     # 2-3 outliers

# Replace with Median Value
t2$LotFrontage[is.na(t2$LotFrontage)] <- median(t2$LotFrontage, na.rm = TRUE)
v2$LotFrontage[is.na(v2$LotFrontage)] <- median(v2$LotFrontage, na.rm = TRUE)
tt$LotFrontage[is.na(tt$LotFrontage)] <- median(tt$LotFrontage, na.rm = TRUE)

#MasVnrArea: Masonry veneer area in square feet - 5 NA's
# ggplot(t2, aes(x = MasVnrArea)) + geom_histogram()
# ggplot(t2, aes(x = MasVnrArea, y = SalePrice)) + geom_point()   # Probably depends also on MasVnrType

# Replace NA's with zero
t2$MasVnrArea[is.na(t2$MasVnrArea)] <- 0
v2$MasVnrArea[is.na(v2$MasVnrArea)] <- 0
tt$MasVnrArea[is.na(tt$MasVnrArea)] <- 0

# GarageYrBlt: Year garage was built - 49 NA's
# ggplot(t2, aes(x = GarageYrBlt)) + geom_histogram()
# ggplot(t2, aes(x = GarageYrBlt, y = SalePrice)) + geom_point()
# # Check relation between GarageYrBlt and YearBuilt
# ggplot(t2, aes(x = GarageYrBlt, y = YearBuilt)) + geom_point()
# ggplot(t2, aes(x = GarageYrBlt - YearBuilt)) + geom_histogram()
# 
# t2b <- t2 %>% filter(is.na(GarageYrBlt))
# ggplot(t2b, aes(x = YearBuilt)) + geom_histogram()

# Replace NA's with YearBuilt values
t2$GarageYrBlt[is.na(t2$GarageYrBlt)] <- t2$YearBuilt[is.na(t2$GarageYrBlt)]
v2$GarageYrBlt[is.na(v2$GarageYrBlt)] <- v2$YearBuilt[is.na(v2$GarageYrBlt)]
tt$GarageYrBlt[is.na(tt$GarageYrBlt)] <- tt$YearBuilt[is.na(tt$GarageYrBlt)]

summary(t2)
summary(v2)
summary(tt)

# Fix training set specific NA's - small numbers only
ggplot(tt, aes(x = BsmtFinSF1)) + geom_histogram()
ggplot(tt, aes(x = BsmtFinSF2)) + geom_histogram()
ggplot(tt, aes(x = BsmtUnfSF)) + geom_histogram()
ggplot(tt, aes(x = TotalBsmtSF)) + geom_histogram()
ggplot(tt, aes(x = BsmtFullBath)) + geom_histogram()
ggplot(tt, aes(x = BsmtHalfBath)) + geom_histogram()
ggplot(tt, aes(x = GarageCars)) + geom_histogram()
ggplot(tt, aes(x = GarageArea)) + geom_histogram()

tt$BsmtFinSF1[is.na(tt$BsmtFinSF1)] <- 0
tt$BsmtFinSF2[is.na(tt$BsmtFinSF2)] <- 0
tt$BsmtUnfSF[is.na(tt$BsmtUnfSF)] <- median(tt$BsmtUnfSF, na.rm = TRUE)
tt$TotalBsmtSF[is.na(tt$TotalBsmtSF)] <- median(tt$TotalBsmtSF, na.rm = TRUE)
tt$BsmtFullBath[is.na(tt$BsmtFullBath)] <- 0
tt$BsmtHalfBath[is.na(tt$BsmtHalfBath)] <- 0
tt$GarageCars[is.na(tt$GarageCars)] <- 2
tt$GarageArea[is.na(tt$GarageArea)] <- median(tt$GarageArea, na.rm = TRUE)



#---------------------------------------------------------------------------------------------------
# Build Model(s)
#---------------------------------------------------------------------------------------------------

set.seed(2531)

x <- t2[, -37]
xtest <- v2[, -37]
y <- t2$SalePrice
ytest <- v2$SalePrice

# Parameter Grid
par_grid <- expand.grid(ntree = c(200, 400, 800, 1500),
                        mtry = c(3, 4, 6, 8, 12, 16))

rsquare <- c()

# Loop through multiple random forests (grid search)
for (i in c(1:nrow(par_grid))) {
      
      fit <- randomForest(x = x, y = y, 
                          importance = FALSE,
                          mtry = par_grid[i, "mtry"],
                          ntree = par_grid[i, "ntree"],
                          xtest = xtest, ytest = ytest)
      
      rsquare <- append(rsquare, mean(fit$rsq))
      print(paste0("R-Square: ", round(mean(fit$rsq), 3), " | mtry = ", par_grid[i, "mtry"], 
                   " | ntree = ", par_grid[i, "ntree"]))
}

fit <- randomForest(x = x, y = y, 
                    importance = TRUE,
                    mtry = 6,
                    ntree = 1500,
                    xtest = xtest, ytest = ytest)

df.imp <- data.frame(variable = names(fit$importance[,1]), 
                     importance = fit$importance[,1])
arrange(df.imp, desc(importance))

# Remove negative importance columns from datasets
t2$MiscVal <- NULL
v2$MiscVal <- NULL
tt$MiscVal <- NULL

t2$LowQualFinSF <- NULL
v2$LowQualFinSF <- NULL
tt$LowQualFinSF <- NULL

t2$PoolArea <- NULL
v2$PoolArea <- NULL
tt$PoolArea <- NULL

# Fit Model again
x <- t2[, -34]
xtest <- v2[, -34]
y <- t2$SalePrice
ytest <- v2$SalePrice

fit <- randomForest(x = x, y = y, 
                    importance = TRUE,
                    mtry = 6,
                    ntree = 1500,
                    xtest = xtest, ytest = ytest)

df.imp <- data.frame(variable = names(fit$importance[,1]), 
                     importance = fit$importance[,1])
arrange(df.imp, desc(importance))
varImpPlot(fit)

mean(fit$rsq)

fit <- randomForest(SalePrice ~ ., data = t2,
                    mtry = 6, ntree = 1500)

#---------------------------------------------------------------------------------------------------
# Submission Prediction and File
#---------------------------------------------------------------------------------------------------

ptt <- predict(fit, newdata = tt)
df_tt <- data.frame(Id = tt_id, SalePrice = ptt)

summary(df_tt)
ggplot(df_tt, aes(x = SalePrice)) + geom_histogram()

# Replace NA's in prediction with median sales price
df_tt$SalePrice[is.na(df_tt$SalePrice)] <- median(df_tt$SalePrice, na.rm = TRUE)

write_csv(df_tt, "submission_04.csv")

# Submission_04: NOT IMPROVED - Score: 0.15580 (prev. 0.15154)

