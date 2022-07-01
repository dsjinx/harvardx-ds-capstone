library(tidyverse)
library(caret)
library(data.table)

#load the edx data from misc folder
#create a sample out of edx, and make a pair of train/test sets  

load("~/Work/Rproj/PH125.9x/harvardx-ds-capstone/misc/edx.Rdata")
edx %>% tibble()
set.seed(9 ,sample.kind = "rounding")
sample <- sample(1:9000000, 9999, replace = FALSE)
edx_sample <- edx[sample,]

ind <- createDataPartition(edx_sample$rating, 1, p = 0.1, list = FALSE)
sample_train <- edx_sample[-ind,]
sample_test <- edx_sample[ind,] %>% 
  semi_join(sample_train, by = "movieId") %>%
  semi_join(sample_train, by = "userId")
sample_train %>% tibble()
sample_test %>% tibble()

rm(edx, edx_sample, ind, sample)

#de-mean 
#global mean
g_mean <- mean(sample_train$rating) 
#user bias
u_bias <- sample_train[, .(mean(rating - g_mean)), by = .(userId)]
#movie bias with voting count regulation (eg. lambda = 0.5)
m_bias <- sample_train[u_bias, on = .(userId)][
  , .(sum(rating - g_mean - V1)/(0.5 + .N)), by = .(movieId)]
#tune the lambda by RMSE
lambda <- seq(0, 5, 0.1) #guessing and reviewing with qplot

rmse <- lapply(lambda, function(l){
  
  m_bias <- sample_train[u_bias, on = .(userId)][
    , .(sum(rating - g_mean - V1)/(l + .N)), by = .(movieId)]
  
  pred <- sample_test[u_bias, on = .(userId)][
    m_bias, on = .(movieId)][
      !is.na(rating), .(V1 + i.V1 + g_mean), by = .(userId, movieId)]
  
  rmse <- sample_test[pred, on = .(userId, movieId)][
    , sqrt(mean((rating - V1)^2))]
  
  return(rmse)
})

lambda <- lambda[which.min(rmse)]
qplot(lambda, as.numeric(rmse), geom = c("point", "line"))

rm(rmse)

#latent factor in sgd
#rating table
rtable <- dcast(sample_train, userId ~ movieId, value.var = "rating")