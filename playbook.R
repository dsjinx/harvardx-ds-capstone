library(tidyverse)
library(data.table)
library(caret)
library(Matrix)
library(doParallel)


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
rm(k, ind_cv)

#u_bias
#guessing and reviewing with plotting
lambda_search <- seq(5, 16, 0.1)
#implement parallel computing to save time
cl <- makePSOCKcluster(3)
registerDoParallel(cl)
ub_tune <- foreach(k = 1:5) %:% 
    foreach(l = lambda_search, 
            .combine = "c", 
            .packages = "data.table") %dopar% {
      
      u_bias <- train_cv[[k]][
        , .(u_bias = sum(rating - g_mean)/(l + .N)), by = .(userId)]
      
      pred <- test_cv[[k]][u_bias, on = .(userId)][
        , .(pred = u_bias + g_mean), by = .(userId)]
      
      rmse <- test_cv[[k]][pred, on = .(userId)][
        !is.na(rating), sqrt(mean((rating - pred)^2))]
      }
stopCluster(cl) 
rm(cl) #always clear any established clusters after stopping, otherwise it will
#cause error in starting the next foreach parallel

#plot the results to search possible lambda_u
ub_rmse <- as.data.table(t(sapply(ub_tune, c))) %>% 
  setnames(., as.character(lambda_search))
ub_rmse <- ub_rmse[, lapply(.SD, mean)]
qplot(x = lambda_search, y = as.numeric(ub_rmse[1,]), 
      geom = c("point", "line"))
#try different ranges of lambda_search to find the bottom on plot
lambda_u <- lambda_search[which.min(as.numeric(ub_rmse))]
u_bias <- sample_train[
  , .(u_bias = sum(rating - g_mean)/(lambda_u + .N)), 
  by = .(userId)]
rm(ub_rmse, ub_tune, lambda_search)

#m_bias (repeat the u_bias tuning method)
lambda_search <- seq(1, 6, 0.1)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)
mb_tune <- foreach(k = 1:5) %:% 
  foreach(l = lambda_search, 
          .combine = "c",
          .packages = "data.table") %dopar% {
            
      m_bias <- train_cv[[k]][u_bias, on = .(userId)][!is.na(rating),
          .(m_bias = sum(rating - g_mean - u_bias)/(l + .N)), 
          by = .(movieId)]
      
      pred <- test_cv[[k]][u_bias, on = .(userId)][
          m_bias, on = .(movieId)][!is.na(rating),
          .(pred = u_bias + m_bias + g_mean), 
          by = .(userId, movieId)]
      
      rmse <- test_cv[[k]][pred, on = .(userId, movieId)][
        , sqrt(mean((rating - pred)^2))]
      }
stopCluster(cl)
rm(cl)

mb_rmse <- as.data.table(t(sapply(mb_tune, c))) %>% 
  setnames(., as.character(lambda_search))
mb_rmse <- mb_rmse[, lapply(.SD, mean)]
qplot(x = lambda_search, y = as.numeric(mb_rmse[1,]), 
      geom = c("point", "line"))
lambda_m <- lambda_search[which.min(mb_rmse)]
m_bias <- sample_train[u_bias, on = .(userId)][!is.na(rating), 
            .(m_bias = sum(rating - g_mean - u_bias)/(lambda_m + .N)), 
            by = .(movieId)]
rm(mb_tune, mb_rmse, lambda_search)

#latent factors by sgd
#rating residual table from the train set: rating - (g_mean + u_bias + m_bias)
residual_train <- sample_train[u_bias, on = .(userId)][
                          m_bias, on = .(movieId)][
                            , .(resid = g_mean + u_bias + m_bias - rating), 
                            by = .(userId, movieId)]
ot <- sum(residual_train$resid == 0) #check overfittings 

#QQ plot to check the distribution
r_mean <- residual_train[, mean(resid)]
r_sd <- residual_train[, sd(resid)]
p <- seq(0.05, 0.95, 0.05)
r_quantile <- residual_train[, quantile(resid, p)]
n_quantile <- qnorm(p, r_mean, r_sd)
qplot(n_quantile, r_quantile) + geom_abline()

rm(residual_train, p, temp_resid)

#sgd
#the about normal distribution of residual to allow us making an normal 
#assumption, under which we can initialise the P U for sgd learning
rtable <- dcast(residual_train, userId ~ movieId, value.var = "resid")
sum(!is.na(rtable[,-1]))/(dim(rtable)[1]*(dim(rtable)[2] - 1)) #sparsity
u_id <- rtable[,1] 
m_id <- names(rtable) #reserve for future use, may not need????
rtable_sparse <- as(as.matrix(rtable[,-1]), "sparseMatrix") #exclude userId
#change all NA to 0
rtable_sparse[is.na(rtable_sparse)] <- 0
resids <- rtable_sparse@x #training resid
resid_i <- rtable_sparse@i + 1 #row index of resid (user) 
#col index of resid (movie)
resid_j <- rep(1:rtable_sparse@Dim[2], diff(rtable_sparse@p)) 

rm(rtable, id_NA)

#warm start P Q matrix for sgd
k <- 10
set.seed(0, sample.kind = "Rounding")
P <- matrix(rnorm(k*rtable_sparse@Dim[1], r_mean, r_sd), nrow =  k)
set.seed(0, sample.kind = "Rounding")
Q <- matrix(rnorm(k*rtable_sparse@Dim[2], r_mean, r_sd), nrow =  k)
#??any 0 rows cols, should not, otherwise the corresponding u,i should not be in the table

sgd <- function(P, Q, y, L_rate, lambda_q, lambda_p, batch_size, epochs){
  #y: resid to be trained on (rtable_sparse in dgCMatrix sparse format)
  #lambda_p/q: not tuned, try 
  #batch_size: sample size out of total number of training resid 
  r <- y
  n <- length(y) #number of training resid
  u <- resid_i #row index (userid) of 
  i <- resid_j #ratings in dense format column index

  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
        
        err_ui <- c(P[,u[ui]] %*% Q[,i[ui]] - r[ui]) 
        nabla_p <- err_ui * Q[,i[ui]] + lambda_p * P[,u[ui]]
        nabla_q <- err_ui * P[,u[ui]] + lambda_q * Q[,i[ui]]
        
        P[,u[ui]] <- P[,u[ui]] - L_rate * p_grad
        Q[,i[ui]] <- Q[,i[ui]] - L_rate * q_grad
    }
  
  }
}

rmse_sgd <- function(P, Q, y){
  err_sq <- c()
  
    for (i in length(y)){
      err <- c(P[,resid_i[i]] %*% Q[,resid_j[i]] - y[i])
      err_sq <- append(err_sqrt, err^2)
    }
  sqrt(mean(err_sq))
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