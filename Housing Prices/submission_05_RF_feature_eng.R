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

# BsmtFullBath
ggplot(t2, aes(x = t2$BsmtFullBath, y = SalePrice/1000)) +
      geom_point(size = 3, alpha = 0.3)



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

write_csv(df_tt, "submission_05.csv")

# Submission_05: NOT IMPROVED - Score: 0.15580 (prev. 0.15154)

