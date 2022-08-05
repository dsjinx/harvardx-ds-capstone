##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
##########################################################
#save data set(edx, validation) at local directory, so we do not need to repeat 
#above process, if we get stuck and crashed during the following process
save(edx, validation, file = "mledx.rdata")
load("mledx.rdata")

#library(tidyverse)
#library(data.table)
library(stringr)
#library(caret)
library(Matrix)
library(doParallel)
library(glmnet)

#global mean
g_mean <- mean(edx$rating)

#biases
#making 5-folds cross validation sets
set.seed(2, sample.kind= "Rounding")
ind_cv <- createFolds(edx$userId, k = 5)
train_cv <- list()
test_cv <- list()
for(k in 1:5){
  train_cv[[k]] <- edx[-ind_cv[[k]],]
  test_cv[[k]] <- edx[ind_cv[[k]],] %>% 
    semi_join(train_cv[[k]], by = "movieId") %>% 
    semi_join(train_cv[[k]], by = "userId")
}

rm(k, ind_cv)

#user-bias with ridge regression penalty
lambda_search <- seq(5, 12, 0.1) 
#utilize the multicores on our computer to parallelize the loop
registerDoParallel(cores = 3) #I am using total cores -1 to avoid system frozen
#number of cores only need to be registered once globally
ub_tune <- foreach(l = lambda_search, .combine = "cbind.data.frame") %:% 
  foreach(k = 1:5, .combine = "c", .packages = "data.table") %dopar% {
    u_bias <- train_cv[[k]][
      , .(u_bias = sum(rating - g_mean) / (l + .N)), by = .(userId)]
    
    pred <- u_bias[test_cv[[k]], on = .(userId)][
      , .(err = g_mean + u_bias - rating)]
    
    sqrt(mean(pred$err * pred$err))
  }

setDT(ub_tune)
setnames(ub_tune, as.character(lambda_search))
ub_rmse <- ub_tune[, lapply(.SD, mean)]
#try different ranges of lambda_search to find the turning bottom in plot
qplot(lambda_search, as.numeric(ub_rmse[1, ]), geom = c("point", "line"))

lambda_u <- lambda_search[which.min(ub_rmse[1,])]
u_bias <- sample_train[, .(u_bias = sum(rating - g_mean) / (lambda_u + .N)), 
                       by = .(userId)]

rm(ub_rmse, ub_tune, lambda_search, lambda_u)

#movie-bias with ridge regression penalty
lambda_search <- seq(1, 5, 0.1)
mb_tune <- foreach(l = lambda_search, .combine = "cbind.data.frame") %:% 
  foreach(k = 1:5, .combine = "c", .packages = "data.table") %dopar% {
    m_bias <- u_bias[train_cv[[k]], on = .(userId)][
      , .(m_bias = sum(rating - g_mean - u_bias) / (l + .N)), 
      by = .(movieId)]
    
    pred <- m_bias[u_bias[test_cv[[k]], 
                          on = .(userId)], 
                   on = .(movieId)][
                     , .(err = g_mean + u_bias + m_bias - rating)]
    
    sqrt(mean(pred$err * pred$err))
  }

setDT(mb_tune)
setnames(mb_tune, as.character(lambda_search))
mb_rmse <- mb_tune[, lapply(.SD, mean)]
qplot(lambda_search, as.numeric(mb_rmse[1,]), 
      geom = c("point", "line"))

lambda_m <- lambda_search[which.min(mb_rmse[1,])]
m_bias <- sample_train[u_bias, on = .(userId)][
  , .(m_bias = sum(rating - g_mean - u_bias) / (lambda_m + .N)), 
  by = .(movieId)]

rm(mb_tune, mb_rmse, lambda_search, lambda_m, train_cv, test_cv)

#genres-bias with elastic net penalty
