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
library(corrplot)

#---------------------------------------------------------------------------------------------------
# Load Data
#---------------------------------------------------------------------------------------------------

setwd("C:\\Users\\Steffen\\SkyDrive\\Files\\Data\\Development\\Data Science\\Kaggle\\Housing Prices")
train.raw <- read.csv("train.csv")
test.raw <- read.csv("test.csv")

#---------------------------------------------------------------------------------------------------
# Split the Data
#---------------------------------------------------------------------------------------------------

set.seed(4537)
trainIndex <- createDataPartition(train.raw$Id, p = 0.70, list = FALSE)
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

# Fix test set specific NA's - small numbers only
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
# Detailed Feature Review
#---------------------------------------------------------------------------------------------------

# Plot each variable against SalePrice as scatter plot

# MSSubClass - should be a factor variable!
ggplot(t2, aes(x = t2$MSSubClass, y = SalePrice)) +
      geom_point(size = 3, alpha = 0.3)

# LotFrontage: there is something going on, many values are similar -> vertical lines!
# FIND an explanation for this!
ggplot(t2, aes(x = t2$LotFrontage, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)
ggplot(t1, aes(x = t1$LotFrontage, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# LotArea: some outliers with huge area but low price, why??
ggplot(t2, aes(x = t2$LotArea, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# OverallQual - should also be a factor variable
# Clear correlation, but looks exponential, not linear. Wide spread.
ggplot(t2, aes(x = t2$OverallQual, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

ggplot(t2, aes(as.factor(t2$OverallQual), SalePrice/1000)) +
      geom_boxplot()

# OverallCond - should also be a factor variable
# There is a weak correlation and "5" has an abnormal spread
# Check why there is such a big spread!
ggplot(t2, aes(x = t2$OverallCond, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

ggplot(t2, aes(as.factor(t2$OverallCond), SalePrice/1000)) +
      geom_boxplot()

# YearBuilt - Should maybe not be an integer - change to new variable, "age"
ggplot(t2, aes(x = t2$YearBuilt, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# YearRemodAdd - Same as YearBuilt if no remod done - change type, re-engineer
# Also check why many values are at the minimum. It is not the same as YearBuilt Minimum.
ggplot(t2, aes(x = t2$YearRemodAdd, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# MasVnrArea - Lots of zeros
ggplot(t2, aes(x = t2$MasVnrArea, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# BsmtFinSF1 - Lots of zeros, some correlation plus outlier
ggplot(t2, aes(x = t2$BsmtFinSF1, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# BsmtFinSF2 - lots and lots of zeros. No clear correlation.
ggplot(t2, aes(x = t2$BsmtFinSF2, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# BsmtUnfSF - zeros, weak correlation
ggplot(t2, aes(x = t2$BsmtUnfSF, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# TotalBsmtSF - better correlation if zeros and 1 outlier are ignored
ggplot(t2, aes(x = t2$TotalBsmtSF, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# X1stFlrSF - Correlation and outlier(s)
ggplot(t2, aes(x = t2$X1stFlrSF, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# X2ndFlrSF - Many zeros and correlation
ggplot(t2, aes(x = t2$X2ndFlrSF, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# LowQualFinSF - mostly zeros, poor correlation. But maybe it helps to explain outliers?
ggplot(t2, aes(x = t2$LowQualFinSF, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# GrLivArea - good correlation, 1 outlier
ggplot(t2, aes(x = t2$GrLivArea, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3) + 
      geom_smooth()
t2b <- t2[t2$GrLivArea < 5000,]
ggplot(t2b, aes(x = t2b$GrLivArea, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3) + 
      geom_smooth()

# --- Use Mahalanobis Distance for Outlier Detection --- #
df <- data.frame(t2$GrLivArea, t2$SalePrice)
m.dist <- mahalanobis(df, colMeans(df), cov(df))
summary(m.dist)
df.m.dist <- data.frame(m.dist)
ggplot(df.m.dist, aes(x = df.m.dist$m.dist)) + geom_histogram()
df$mdist <- m.dist
df$abnormal <- 0
df$abnormal[df$mdist >= 10] <- 1
ggplot(df, aes(x = df$t2.GrLivArea, y = df$t2.SalePrice, color = as.factor(df$abnormal))) +
      geom_point(size = 3, alpha = 0.5)

# Detect Outliers with Maha but w/out Sales Price
df2 <- data.frame(t2$GrLivArea, t2$GarageYrBlt, t2$LotArea, t2$LotFrontage)
m.dist2 <- mahalanobis(df2, colMeans(df2), cov(df2))
df2$mdist <- m.dist2
ggplot(df2, aes(x = mdist)) + geom_histogram()
df2$SalePrice <- t2$SalePrice
df2$abnormal <- 0
df2$abnormal[df2$mdist >= 10] <- 1
sum(df2$abnormal)
ggplot(df2, aes(x = df2$t2.GrLivArea, y = df2$SalePrice, color = as.factor(df2$abnormal))) +
      geom_point(size = 3, alpha = 0.5)
ggplot(df2, aes(x = df2$t2.GarageYrBlt, y = df2$SalePrice, color = as.factor(df2$abnormal))) +
      geom_point(size = 3, alpha = 0.5)
ggplot(df2, aes(x = df2$t2.LotArea, y = df2$SalePrice, color = as.factor(df2$abnormal))) +
      geom_point(size = 3, alpha = 0.5)
ggplot(v2, aes(x = v2$LotArea, y = v2$SalePrice)) +
      geom_point(size = 3, alpha = 0.5)


# BsmtFullBath - 3 values only, wide spread. Make factor?
ggplot(t2, aes(x = t2$BsmtFullBath, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# BsmtHalfBath - 3 values only, wide spread, low values for > 1
ggplot(t2, aes(x = t2$BsmtHalfBath, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# FullBath - values 0 - 3. There is correlation
ggplot(t2, aes(x = t2$FullBath, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# HalfBath - values 0 - 2, wide spread
ggplot(t2, aes(x = t2$HalfBath, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# BedroomAbvGr -  0 to 6, low correlation
ggplot(t2, aes(x = t2$BedroomAbvGr, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# KitchenAbvGr: 0 - 3, no real correlation
ggplot(t2, aes(x = t2$KitchenAbvGr, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# TotRmsAbvGrd: there is correlation but wide spread, values: 2 - 12
ggplot(t2, aes(x = t2$TotRmsAbvGrd, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# Fireplaces: 0 - 3, weak correlation
ggplot(t2, aes(x = t2$Fireplaces, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# GarageYrBlt: replace with age - correlation but not linear
ggplot(t2, aes(x = t2$GarageYrBlt, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3) + geom_smooth(se = FALSE, color = "red")

# GarageCars
ggplot(t2, aes(x = t2$GarageCars, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# GarageArea - correlation plus zeroes plus outliers
ggplot(t2, aes(x = t2$GarageArea, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# WoodDeckSF - many zeroes, some correlation
ggplot(t2, aes(x = t2$WoodDeckSF, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# OpenPorchSF - many zeroes, weak correlation
ggplot(t2, aes(x = t2$OpenPorchSF, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# EnclosedPorch - mostly zeroes, weak correlation, outliers
ggplot(t2, aes(x = t2$EnclosedPorch, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# X3SsnPorch - mostly zeroes
ggplot(t2, aes(x = t2$X3SsnPorch, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# ScreenPorch - mostly zeroes
ggplot(t2, aes(x = t2$ScreenPorch, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# PoolArea - mostly zeroes
ggplot(t2, aes(x = t2$PoolArea, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# MiscVal - mostly zeroes - maybe good for outlier detection?
ggplot(t2, aes(x = t2$MiscVal, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# MoSold - values 1 - 12, should be factor var. No linear correlation but some relation is there.
# wide spread - outliers
ggplot(t2, aes(x = t2$MoSold, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)

# YrSold - no correlation, but financial crisis could show some impact
# should be factor var
ggplot(t2, aes(x = t2$YrSold, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)


# Correlation
correlation <- cor(t2)
corrplot(correlation, order = "hclust")


#---------------------------------------------------------------------------------------------------
# Build Model(s)
#---------------------------------------------------------------------------------------------------

# Use XGBoost - no changes to integer variables
library(xgboost)
library(Ckmeans.1d.dp)

# Convert data into sparse matrices
train <- t2
train$SalePrice <- NULL
label.tr <- t2$SalePrice

val <- v2
val$SalePrice <- NULL
label.val <- v2$SalePrice

dtrain <- xgb.DMatrix(data = as.matrix(train), label = label.tr)
dval <- xgb.DMatrix(data = as.matrix(val), label = label.val)

watchlist <- list(train = dtrain, test = dval)

fit_gbtree <- xgb.train(data = dtrain, max.depth = 5, eta = 0.2, nthread = 2,
                        watchlist = watchlist, nround = 100, verbose = TRUE)

xgb.dump(fit_gbtree, with.stats = TRUE)
importance_matrix <- xgb.importance(model = fit_gbtree)

xgb.plot.importance(importance_matrix)

ntrees <- 200
xgboostModelCV <- xgb.cv(data =  dtrain, nrounds = ntrees, nfold = 5, showsd = TRUE, 
                         metrics = "rmse", verbose = TRUE, "eval_metric" = "rmse",
                         "objective" = "reg:linear", "max.depth" = 10, "eta" = 0.1,                               
                         "subsample" = 0.75, "colsample_bytree" = 1)

xvalidationScores <- as.data.frame(xgboostModelCV)

ggplot(xvalidationScores, aes(x = c(1:ntrees), y = test.rmse.mean)) + geom_line()

fit_gbtree <- xgb.train(data = dtrain, max.depth = 10, eta = 0.1, nthread = 2,
                        watchlist = watchlist, nround = 200, verbose = TRUE)

dtest <- xgb.DMatrix(as.matrix(tt))


#---------------------------------------------------------------------------------------------------
# Submission Prediction and File
#---------------------------------------------------------------------------------------------------

ptt <- predict(fit_gbtree, newdata = dtest)
df_tt <- data.frame(Id = tt_id, SalePrice = ptt)

summary(df_tt)
ggplot(df_tt, aes(x = SalePrice)) + geom_histogram()

# Replace NA's in prediction with median sales price
df_tt$SalePrice[is.na(df_tt$SalePrice)] <- median(df_tt$SalePrice, na.rm = TRUE)

write_csv(df_tt, "submission_05.csv")

# Submission_05: Score: 0.15088 (prev. 0.15154, improved by 15 places)

