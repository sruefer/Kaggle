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

#---------------------------------------------------------------------------------------------------
# Load Data
#---------------------------------------------------------------------------------------------------

setwd("C:\\Users\\Steffen\\SkyDrive\\Files\\Data\\Development\\Data Science\\Kaggle\\Housing Prices")
train.raw <- read.csv("train.csv")
test.raw <- read.csv("test.csv")

#---------------------------------------------------------------------------------------------------
# Split the Data
#---------------------------------------------------------------------------------------------------

trainIndex <- createDataPartition(train.raw$Id, p = 0.6, list = FALSE)
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


# Deal with NA's: create histograms first - 157 NA's
# LotFrontage: Linear feet of street connected to property
ggplot(t2, aes(x = LotFrontage)) + geom_histogram()
ggplot(t2, aes(x = LotFrontage, y = SalePrice)) + geom_point()     # 2-3 outliers

# Replace with Median Value
t2$LotFrontage[is.na(t2$LotFrontage)] <- median(t2$LotFrontage, na.rm = TRUE)
v2$LotFrontage[is.na(v2$LotFrontage)] <- median(v2$LotFrontage, na.rm = TRUE)
tt$LotFrontage[is.na(tt$LotFrontage)] <- median(tt$LotFrontage, na.rm = TRUE)

#MasVnrArea: Masonry veneer area in square feet - 5 NA's
ggplot(t2, aes(x = MasVnrArea)) + geom_histogram()
ggplot(t2, aes(x = MasVnrArea, y = SalePrice)) + geom_point()   # Probably depends also on MasVnrType

# Replace NA's with zero
t2$MasVnrArea[is.na(t2$MasVnrArea)] <- 0
v2$MasVnrArea[is.na(v2$MasVnrArea)] <- 0
tt$MasVnrArea[is.na(tt$MasVnrArea)] <- 0

# GarageYrBlt: Year garage was built - 49 NA's
ggplot(t2, aes(x = GarageYrBlt)) + geom_histogram()
ggplot(t2, aes(x = GarageYrBlt, y = SalePrice)) + geom_point()
# Check relation between GarageYrBlt and YearBuilt
ggplot(t2, aes(x = GarageYrBlt, y = YearBuilt)) + geom_point()
ggplot(t2, aes(x = GarageYrBlt - YearBuilt)) + geom_histogram()

t2b <- t2 %>% filter(is.na(GarageYrBlt))
ggplot(t2b, aes(x = YearBuilt)) + geom_histogram()

# Replace NA's with YearBuilt values
t2$GarageYrBlt[is.na(t2$GarageYrBlt)] <- t2$YearBuilt[is.na(t2$GarageYrBlt)]
v2$GarageYrBlt[is.na(v2$GarageYrBlt)] <- v2$YearBuilt[is.na(v2$GarageYrBlt)]
tt$GarageYrBlt[is.na(tt$GarageYrBlt)] <- tt$YearBuilt[is.na(tt$GarageYrBlt)]

summary(t2)
summary(v2)
summary(tt)

#---------------------------------------------------------------------------------------------------
# Linear Models with t2
#---------------------------------------------------------------------------------------------------

fit_lm1 <- lm(SalePrice ~ ., data = t2)
summary(fit_lm1)
plot(fit_lm1)           # Residuals Plot


# Stepwise selection
fit_lm2 <- step(fit_lm1, direction = "backward")
summary(fit_lm2)

#---------------------------------------------------------------------------------------------------
# Predictions with Validation Set
#---------------------------------------------------------------------------------------------------

p1 <- data.frame(predict(fit_lm1, newdata = v2[, -37]))
p2 <- data.frame(predict(fit_lm2, newdata = v2[, -37]))
y <- data.frame(v2$SalePrice)

p <- rbind(y, p1, p2)

# Prediction Plotting
ggplot(p, aes(x = p1, y = y)) + 
      geom_point(alpha = 0.3, size = 2) + 
      ylim(0, 800000) + xlim(0, 800000) +
      geom_abline(slope = 1, intercept = 0, col = "blue")

ggplot(p, aes(x = p2, y = y)) + 
      geom_point(alpha = 0.3, size = 2) + 
      ylim(0, 800000) + xlim(0, 800000) +
      geom_abline(slope = 1, intercept = 0, col = "blue")

#---------------------------------------------------------------------------------------------------
# Predictions with Test Set
#---------------------------------------------------------------------------------------------------

ptt <- predict(fit_lm2, newdata = tt)
df_tt <- data.frame(Id = tt_id, SalePrice = ptt)

summary(df_tt)
ggplot(df_tt, aes(x = SalePrice)) + geom_histogram()

# Replace NA's in prediction with median sales price
df_tt$SalePrice[is.na(df_tt$SalePrice)] <- median(df_tt$SalePrice, na.rm = TRUE)

write_csv(df_tt, "submission_01.csv")

# Submission_01: place 1960, score approx 0.2294. Baseline score was 0.4

