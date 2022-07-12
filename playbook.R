library(tidyverse)
library(caret)
library(data.table)
library(Matrix)
library(doParallel)
library(ggplot2)

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
#user and movie biases with count regulation lambda_u and lambda_m respectively
#tune the lambda by cross validation
set.seed(0, sample.kind= "Rounding")
ind_cv <- createFolds(sample_train$userId, k = 5)
train_cv <- list()
test_cv <- list()
for(k in 1:5){
  train_cv[[k]] <- sample_train[-ind_cv[[k]],]
  test_cv[[k]] <- sample_train[ind_cv[[k]],]
}

#implement parallel computing to save time
cl <- makePSOCKcluster(2)
registerDoParallel(cl)
lambda_search <- seq(15, 23, 0.1) #guessing and reviewing with qplot
tune_u <- foreach(k = 1:5) %:% 
    foreach(l = lambda_search, .combine = rbind, 
            #.export = c("train_cv", "test_cv"), 
            .packages = c("data.table")) %dopar% {
      
      u_bias <- train_cv[[k]][
        , .(u_bias = sum(rating - g_mean)/(l + .N)), by = .(userId)]
      
      pred <- test_cv[[k]][u_bias, on = .(userId)][
        , .(pred = u_bias + g_mean), by = .(userId)]
      
      rmse <- test_cv[[k]][pred, on = .(userId)][
        !is.na(rating), sqrt(mean((rating - pred)^2))]
      
      return(rmse)
  }
stopCluster(cl)

#plot the results by facet_grid
for(k in 1:5){
  
}

rmse_u <- lapply(lambda_test, function(l){
  #cross-validation method
  
  train <- 
  test <-  
  
  u_bias <- sample_train[
    , .(u_bias = sum(rating - g_mean)/(l + .N)), by = .(userId)]
  
  pred <- sample_test[u_bias, on = .(userId)][
    , .(pred = u_bias + g_mean), by = .(userId)]
  
  rmse <- sample_test[pred, on = .(userId)][
    !is.na(rating), sqrt(mean((rating - pred)^2))]
  
  return(rmse)
})

qplot(lambda_test, unlist(rmse_u), geom = c("point", "line"))
lambda_u <- lambda_test[which.min(rmse_u)]

lambda_test <- seq(0, 5, 0.1)

rmse_m <- lapply(lambda_test, function(l){
  
  m_bias <- sample_train[u_bias, on = .(userId)][
    , .(m_bias = sum(rating - g_mean - u_bias)/(l + .N)), by = .(movieId)]
  
  pred <- sample_test[u_bias, on = .(userId)][
    m_bias, on = .(movieId)][
      !is.na(rating), .(pred = u_bias + m_bias + g_mean), by = .(userId, movieId)]
  
  rmse <- sample_test[pred, on = .(userId, movieId)][
    , sqrt(mean((rating - pred)^2))]
  
  return(rmse)
})

qplot(lambda, as.numeric(rmse), geom = c("point", "line"))
lambda_m <- lambda[which.min(rmse)]

rm(rmse)

#latent factor in sgd
#rating residual table from the train set: rating - (g_mean + u_bias + m_bias)
residual_train <- function(l){
  
  m_bias <- sample_train[u_bias, on = .(userId)][
    , .(m_bias = sum(rating - g_mean - u_bias)/(l + .N)), by = .(movieId)]
  
  est <- sample_train[u_bias, on = .(userId)][
    m_bias, on = .(movieId)][
      , .(est = u_bias + m_bias + g_mean), by = .(userId, movieId)]
  
  resid <- sample_train[est, on = .(userId, movieId)][
    , .(resid = rating - est), by = .(userId, movieId)]
  
  return(list(resid, m_bias))
}

m_bias <- residual_train(lambda)[[2]]
temp_resid <- residual_train(lambda)[[1]]
rtable <- dcast(temp_resid, userId ~ movieId, value.var = "resid")
sum(!is.na(rtable[,-1]))/(dim(rtable)[1]*(dim(rtable)[2] - 1)) #sparsity

#?inspect temp_resid$V1 to decide the starting P,Q matrix distribution
#?possibly start with QQ plot
r_mean <- temp_resid[, mean(resid)]
r_sd <- temp_resid[, sd(resid)]
p <- seq(0.05, 0.95, 0.05)
r_quantile <- temp_resid[, quantile(V1, p)]
n_quantile <- qnorm(p, r_mean, r_sd)
qplot(n_quantile, r_quantile) + geom_abline()

rm(residual_train, p, temp_resid)

#sgd
#the about normal distribution of residual to allow us making an normal 
#assumption, under which we can initialise the P U for sgd learning
rtable_sparse <- as(as.matrix(rtable[,-1]), "sparseMatrix") #exclude userId
id_NA <- which(is.na(rtable_sparse@x))
range(rtable_sparse@x[-id_NA])
length(which(rtable_sparse ==0))
#inspection tells us we have 3050 perfect pred by 0 resid, which fills the 
#gap between number of non-NA's and @x non zeros
#!!!!!!0 resid is a sign of overfitting, need to consider add regulation on u_bias
rtable_sparse[is.na(rtable_sparse)] <- 0
rating_train <- rtable_sparse@x #non-zero ratings
rtable_i <- rtable_sparse@i + 1 #row index of non-zero 
rtable_j <- rep(1:rtable_sparse@Dim[2], diff(rtable_sparse@p)) #col index of non-zero

rm(rtable, id_NA)

#warm start P Q matrix for sgd
k <- 20
set.seed(0, sample.kind = "Rounding")
P <- matrix(rnorm(k*rtable_sparse@Dim[1], r_mean, r_sd), nrow =  k)
set.seed(0, sample.kind = "Rounding")
Q <- matrix(rnorm(k*rtable_sparse@Dim[2], r_mean, r_sd), nrow =  k)
##?? make a pair of sparse P Q, with only non zero columns corresponding to u i 


sgd <- function(P, Q, y, L_rate, lambda_q, lambda_p, batch_size, epochs){
  #y: rating table to be trained on (rtable_sparse in dgCMatrix sparse format)
  #lambda_p/q: not tuned for now, on personal preference
  #batch_size: should be integer << number of ratings to be trained on
  n <- length(y@x) #number of non zero ratings
  r <- y@x #ratings to be learned on
  u <- rtable_i #rating row index, default starting is 0
  i <- rtable_j #ratings in dense format column index
  learning_log <- list()
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
        
        err_ui <- c(P[,u[ui]] %*% Q[,i[ui]] - r[ui]) 
        p_grad <- err_ui * Q[,i[ui]] + lambda_p * P[,u[ui]]
        q_grad <- err_ui * P[,u[ui]] + lambda_q * Q[,i[ui]]
        
        P[,u[ui]] <- P[,u[ui]] - L_rate * p_grad
        Q[,i[ui]] <- Q[,i[ui]] - L_rate * q_grad
    }
  
    #err <- P[,u] %*% Q[,i] - r #??need to fix
    #learning_log[[t]] <-sqrt(mean(err^2)) 
  }
  #return(learning_log)
}

##?? parallise RMSE
#err_resid <- t(P) %*% Q - rtable
#rmse_learning <- sqrt(mean(t(err_resid)%*%err_resid))
#sgd template tested working
sgd <- function(L_rate,epochs){
  
  learning_log <- list()
  
  for ( t in 1:epochs){
    
    ids <- sample(1:length(x_s@x), 1)
    
    for (id in ids){
      err <- c(p[,u[id]] %*% q[,i[id]] - x_s[id])
      nabla_p <- err * q[, i[id]] + 1 * p[,u[id]]
      nabla_q <- err * p[, u[id]] + 1 * q[,i[id]]  
      p[,u[id]] <- p[,u[id]] - L_rate * nabla_p
      q[,i[id]] <- q[,i[id]] - L_rate * nabla_q    
    }
    err_log <- t(p) %*% q - x_s 
    learning_log[[t]] <- sqrt(sum(apply(err_log, 2, crossprod))/ncol(x_s))
  }
  return(learning_log[[which.min(learning_log)]])
}