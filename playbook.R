library(tidyverse)
library(caret)
library(data.table)
library(Matrix)

#load the edx data from misc folder
#create a sample out of edx, and make a pair of train/test sets  

load("~/Work/Rproj/PH125.9x/harvardx-ds-capstone/misc/edx.Rdata")
edx %>% tibble()
set.seed(9 ,sample.kind = "Rounding")
sample <- sample(1:9000000, 9999, replace = FALSE)
edx_sample <- edx[sample,]

set.seed(2 ,sample.kind = "Rounding")
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
#m_bias <- sample_train[u_bias, on = .(userId)][
#  , .(sum(rating - g_mean - V1)/(0.5 + .N)), by = .(movieId)]
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

qplot(lambda, as.numeric(rmse), geom = c("point", "line"))
lambda <- lambda[which.min(rmse)]

rm(rmse)

#latent factor in sgd
#rating residual table from the train set: rating - (g_mean + u_bias + m_bias)
residual_train <- function(l){
  
  m_bias <- sample_train[u_bias, on = .(userId)][
    , .(sum(rating - g_mean - V1)/(l + .N)), by = .(movieId)]
  
  est <- sample_train[u_bias, on = .(userId)][
    m_bias, on = .(movieId)][
      , .(V1 + i.V1 + g_mean), by = .(userId, movieId)]
  
  err <- sample_train[est, on = .(userId, movieId)][
    , .(rating - V1), by = .(userId, movieId)]
  
  return(err)
}

temp_resid <- residual_train(lambda)
rtable <- dcast(temp_resid, userId ~ movieId, value.var = "V1")
sum(!is.na(rtable[,-1])) #highly sparse matrix

#sgd
#?inspect temp_resid$V1 to decide the starting P,Q matrix distribution
#?possibly start with QQ plot
r_mean <- temp_resid[, mean(V1)]
r_sd <- temp_resid[, sd(V1)]
p <- seq(0.05, 0.95, 0.05)
r_quantile <- temp_resid[, quantile(V1, p)]
n_quantile <- qnorm(p, r_mean, r_sd)
qplot(n_quantile, r_quantile) + geom_abline()

#the about normal distribution of residual to allow us making an normal 
#assumption, under which we can initialise the P U for sgd learning
rtable[is.na(rtable)] <- 0
rtable <- Matrix(as.matrix(rtable[,-1]), sparse = TRUE) #exclude userId
rating_train <- rtable@x #non-zero ratings
rtable_i <- rtable@i #row index of non-zero 
rtable_j <- rep(1:rtable@Dim[2], diff(rtable@p)) #col index of non-zero

#warm start P Q matrix for sgd
k <- 83
P <- matrix(rnorm(k*rtable@Dim[1], r_mean, r_sd), nrow =  k)
Q <- matrix(rnorm(k*rtable@Dim[2], r_mean, r_sd), nrow =  k)

sgd <- function(P, Q, y, L_rate, batch_size, epochs){
  #y: rating table to be trained against (rtable in dgCMatrix sparse format)
  #lambda_p/q: not tuned for now, on personal preference
  n <- length(y@x)
  r <- y@x #ratings in dense form
  u <- y@i+1 #rating row index
  i <- rep(1:y@Dim[2], diff(y@p)) #rating column index
  learning_log <- list()  
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
        
        err_ui <- P[,u[ui]] %*% Q[,i[ui]] - r[ui]
        p_grad <- err * Q[,i[ui]] + lambda_p * P[,u[ui]]
        q_grad <- err * P[,u[ui]] + lambda_q * Q[,i[ui]]
        
        P[,u[ui]] <- P[,u[ui]] - L_rate * p_grad
        Q[,i[ui]] <- Q[,i[ui]] - L_rate * q_grad
    }
  
  err <- t(P) %*% Q - r ####solve the index?????? 
  learning_log[[t]] <-sqrt(mean(err^2)) 
  }
  return(learning_log)
}
