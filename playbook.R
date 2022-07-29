library(tidyverse)
library(data.table)
library(stringr)
library(caret)
library(Matrix)
library(doParallel)
library(glmnet)


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
registerDoParallel(cores = 3)
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

#plot the results to search possible lambda_u
ub_rmse <- as.data.table(t(sapply(ub_tune, c))) %>% 
  setnames(., as.character(lambda_search))
ub_rmse <- ub_rmse[, lapply(.SD, mean)]
qplot(x = lambda_search, y = as.numeric(ub_rmse[1,]), 
      geom = c("point", "line"))
#try different ranges of lambda_search to find the bottom on plot
lambda_u <- lambda_search[which.min(as.numeric(ub_rmse))]
u_bias <- sample_train[, .(u_bias = sum(rating - g_mean)/(lambda_u + .N)), 
                        by = .(userId)]
rm(ub_rmse, ub_tune, lambda_search)

#m_bias (repeat the u_bias tuning method)
lambda_search <- seq(1, 6, 0.1)
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

#genres
genres <- str_split(sample_train$genres, "\\|") 
gen_cat <- genres %>% unlist() %>% unique()
n <- length(gen_cat)
gen_mean <- lapply(1:n, function(n){
                  sample_train[u_bias, on = .(userId)][
                    m_bias, on = .(movieId)][
                    genres %like% gen_cat[n],
                    mean(g_mean + u_bias + m_bias - rating)]
                  })
names(gen_mean) <- gen_cat
rm(n)

m_id <- unique(sample_train$movieId)
m_n <- length(m_id)
m_gen <- fpreach(i = 1:5, ,packages = c("stringr", "data.table"), function(i){
          str_split(sample_train[movieId == m_id[i], genres][1], "\\|") 
          })

gen_n <- length(genres)
ind <- vector(mode = "list", length = m_n)
ind <- lapply(1:gen_n, function(n){
          for(i in 1:length(genres[[n]])){
              ind[[n]][i] <- str_which(gen_cat, genres[[n]][i])
          }
        })

cl <- makePSOCKcluster(3)
registerDoParallel(cl)
ind <- foreach(g = 1:gen_n) %:% 
          foreach(i = 1:length(genres[[g]]), .combine = "c", 
                  .packages = "stringr") %dopar% {
                  ind <- str_which(genres_cat, genres[[g]][i])
          }
stopCluster(cl)
rm(cl)

gen <- data.frame(genres_cat)
j <- 1
gen <- while(j <= m_n){
          gen <- cbind(gen, gen_mean)
          gen_mean[-id[[j]]] <- 0
          j <- j + 1
}
names(gen) <- c(gens, m_id)

#rating residual table from the train set: rating - (g_mean + u_bias + m_bias)
residual_train <- sample_train[u_bias, on = .(userId)][
                          m_bias, on = .(movieId)][
                            , .(resid = g_mean + u_bias + m_bias - rating), 
                            by = .(userId, movieId)]
of <- sum(residual_train$resid == 0) #check overfittings 
rm(of)

#QQ plot to check the distribution
r_mean <- residual_train[, mean(resid)]
r_sd <- residual_train[, sd(resid)]
p <- seq(0.05, 0.95, 0.05)
r_quantile <- residual_train[, quantile(resid, p)]
n_quantile <- qnorm(p, r_mean, r_sd)
qplot(n_quantile, r_quantile) + geom_abline()

rm(p, r_quantile, n_quantile)

#the almost normal distribution of residual to allow us making an normal 
#assumption, under which we can initialise the P U for sgd learning
rtable <- dcast(residual_train, userId ~ movieId, value.var = "resid")
sum(!is.na(rtable[,-1]))/(dim(rtable)[1]*(dim(rtable)[2] - 1)) #sparsity
u_id <- as.character(unlist(rtable[,1])) #turn a data table into strings, 
                  #since the fundamental structure of data table is list
rtable_sparse <- as(as.matrix(rtable[,-1]), "sparseMatrix") #exclude userId
#replace all NA with 0 to make sparse
replace_na(rtable_sparse, 0)

# ??use caret with method = glment
cl <- makePSOCKcluster(3)
registerDoParallel(cl)
fit <- cv.glmnet(gen, rtable_sparse, family = "mgaussian", 
                 type.measure = "mse", nfolds = 5, alpha = 0.5, 
                 parallel = TRUE, trace.it = TRUE)
stopCluster(cl)
rm(cl)

#sgd
#rtable_sparse[is.na(rtable_sparse)] <- 0
resids <- rtable_sparse@x #training resid
resid_i <- rtable_sparse@i + 1 #row index of resid (user) 
#col index of resid (movie)
resid_j <- rep(1:rtable_sparse@Dim[2], diff(rtable_sparse@p)) 

rm(rtable, residual_train)

#warm start P Q matrix for sgd
k <- 200
f_mean <- sqrt(r_mean/k)
f_sd<- r_sd/sqrt(k) #by LNN
set.seed(0, sample.kind = "Rounding")
P <- matrix(rnorm(k*rtable_sparse@Dim[1], f_mean, f_sd), nrow = k)
set.seed(0, sample.kind = "Rounding")
Q <- matrix(rnorm(k*rtable_sparse@Dim[2], f_mean, f_sd), nrow = k)

sgd <- function(P, Q, y, L_rate, lambda_p, lambda_q, batch_size, epochs){
  #y: resid to be trained on (rtable_sparse in dgCMatrix sparse format)
  #lambda_p/q: not tuned, try 
  #batch_size: sample size out of total number of training resid 
  #batch_size * epochs should be much larger than length(y) 
  r <- y
  n <- length(y) #number of training resid
  learning_log <- list()
  
  for (t in 1:epochs){
    
    batch_id <- sample(1:n, batch_size, replace = FALSE)
    
    for (ui in batch_id){
        
        err_ui <- c(P[,resid_i[ui]] %*% Q[,resid_j[ui]] - r[ui]) 
        nabla_p <- err_ui * Q[,resid_j[ui]] / n + lambda_p * P[,resid_i[ui]]
        nabla_q <- err_ui * P[,resid_i[ui]] / n + lambda_q * Q[,resid_j[ui]]
        
        P[,resid_i[ui]] <- P[,resid_i[ui]] - L_rate * nabla_p
        Q[,resid_j[ui]] <- Q[,resid_j[ui]] - L_rate * nabla_q
    }
    
    err <- sapply(1:n, function(j){
      P[,resid_i[j]] %*% Q[,resid_j[j]] - r[j]
      })
    learning_log[[t]] <- sqrt(mean(err^2))
    rm(err)
  }
  return(learning_log)
}

learning_result <- sgd(P = P, Q = Q , y = resids, 
                       L_rate = 0.6, lambda_p = 0.3, lambda_q = 0.3, 
                       batch_size = 30, epochs = 500)
learning_result <- unlist(learning_result)
qplot(x = c(1:500), y = learning_result)
rm(learning_result) #clean before restart

#validation
P <- as.data.frame(P) %>% setNames(u_id)
Q <- as.data.frame(Q) %>% setNames(m_id[-1])

rmse <- function(g_mean, u_bias, m_bias, P, Q, valid){
            
            pred <- valid[u_bias, on = .(userId)][
                                m_bias, on = .(movieId)][!is.na(rating), 
                                .(pred = g_mean + u_bias + m_bias), 
                                by = .(userId, movieId)]
            pred_uid <- as.character(unlist(pred[,1]))
            pred_mid <- as.character(unlist(pred[,2]))
            n <- length(pred$pred)
            
            cl <- makePSOCKcluster(3)
            registerDoParallel(cl) 
            pred_resids <- foreach(i = 1:n, 
                                   .combine = "c", 
                                   .packages = "data.table") %dopar% {
                                    P[,pred_uid[i]] %*% Q[,pred_mid[i]]
                                  }
            stopCluster(cl)
            rm(cl)

            pred <- pred[, resid := pred_resids]
            err_rmse <- valid[pred, on = .(userId, movieId)][
                                    , .(err = pred + resid - rating), 
                                    by = .(userId, movieId)]
            return(sqrt(mean(err_rmse$err * err_rmse$err)))
        }

rmse(g_mean = g_mean, u_bias = u_bias, m_bias = m_bias, P = P, Q = Q, 
     valid = sample_test)
