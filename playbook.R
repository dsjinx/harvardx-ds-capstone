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
  test_cv[[k]] <- sample_train[ind_cv[[k]],] %>% 
    semi_join(train_cv[[k]], by = "movieId") %>% 
    semi_join(train_cv[[k]], by = "userId")
}

rm(k, ind_cv)

#u_bias
#guessing and reviewing with plotting
lambda_search <- seq(5, 12, 0.1)
#implement parallel computing to save time
registerDoParallel(cores = 3)
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
qplot(lambda_search, as.numeric(ub_rmse[1, ]), geom = c("point", "line"))
#try different ranges of lambda_search to find the bottom on plot
lambda_u <- lambda_search[which.min(ub_rmse[1,])]
u_bias <- sample_train[, .(u_bias = sum(rating - g_mean) / (lambda_u + .N)), 
                        by = .(userId)]

rm(ub_rmse, ub_tune, lambda_search, lambda_u)
#rm(mb_rmse, mb_tune, lambda_search)
#m_bias (repeat the u_bias tuning method)
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

#genres
genres <- str_split(sample_train$genres, "\\|") 
gen_cat <- genres %>% unlist() %>% unique()
n <- length(gen_cat)
gen_mean <- foreach(g = 1:n, .combine = "c", 
                    .packages = "data.table") %dopar% {
                    m_bias[u_bias[sample_train, on = .(userId)], 
                           on = .(movieId)][genres %like% gen_cat[g],
                    mean(g_mean + u_bias + m_bias - rating)]
                  }

rm(n)

m_id <- unique(sample_train$movieId)
m_n <- length(m_id)
m_gen <- foreach(i = 1:m_n, .packages = c("stringr", "data.table")) %dopar% {
            gens <- sample_train[movieId == m_id[i], genres][1] 
            gens <- str_split(gens, "\\|") %>% unlist()
            }
names(m_gen) <- m_id

ind <- foreach(g = 1:m_n) %dopar% {
          foreach(i = 1:length(m_gen[[g]]), .combine = "c", 
                  .packages = "stringr") %do% {
                  ind <- str_which(gen_cat, m_gen[[g]][i])}
              }

gen <- data.frame(gen_cat)
j <- 1
while(j <= m_n){
      gen[, j+1] <- gen_mean
      gen[-ind[[j]], j+1] <- 0
      j <- j + 1
      }
colnames(gen)[-1] <- m_id

rm(j, genres, ind, m_gen, gen_cat, gen_mean, m_n)

#rating residual table from the train set: rating - (g_mean + u_bias + m_bias)
residual_train <- m_bias[u_bias[sample_train, on = .(userId)]
                         , on = .(movieId)][
                           , .(resid = g_mean + u_bias + m_bias - rating), 
                            by = .(userId, movieId)]
sum(residual_train$resid == 0) #check overfittings 

#QQ plot to check the distribution
r_mean <- residual_train[, mean(resid)]
r_sd <- residual_train[, sd(resid)]
p <- seq(0.05, 0.95, 0.05)
r_quantile <- residual_train[, quantile(resid, p)]
n_quantile <- qnorm(p, r_mean, r_sd)
qplot(n_quantile, r_quantile) + geom_abline()

rm(p, r_quantile, n_quantile)

rtable <- dcast(residual_train, userId ~ movieId, value.var = "resid")
sum(!is.na(rtable[,-1]))/(dim(rtable)[1]*(dim(rtable)[2] - 1)) #sparsity
movieId <- names(rtable[,-1])
rtable_tr <- transpose(rtable, keep.names = "movieId", 
                      make.names = "userId")
rtable_tr$movieId <- as.numeric(rtable_tr$movieId)
m_id_dt <- data.table(movieId = m_id)
rtable <- rtable_tr[m_id_dt, on = .(movieId)]

rm(rtable_tr, movieId, m_id, m_id_dt, m_id)

rtable_y <- setnafill(rtable[,-1], fill = 0)
rtable_y <- as(as.matrix(rtable_y), "sparseMatrix")
gen_x <- as(as.matrix(gen[,-1]), "sparseMatrix")
gen_x <- t(gen_x)
set.seed(1, sample.kind = "Rounding")
ind_y<- createFolds(1: rtable_y@Dim[2], k = ceiling(rtable_y@Dim[2]/1000))

rm(rtable)
#choice between fitting data in limited ram with slow for loop
#speed up the learning with lapply but require bit ram space
#some how the lappy version does not work on windows
b <- length(ind_y)
system.time(u_beta <- lapply(1:b, function(k){
  fit <- cv.glmnet(gen_x, rtable_y[, ind_y[[k]]],
                   family = "mgaussian", 
                   type.measure = "mse", 
                   nfolds = 5, alpha = 0.5,
                   parallel = TRUE, trace.it = TRUE)
  coef(fit, s= "lambda.min")}
  )
)
rm(b)
u_beta <- unlist(u_beta)

u_betas <- list()
system.time(for(k in 1:b){
  fit <- cv.glmnet(gen_x, rtable_y[, ind_y[[k]]],
                     family = "mgaussian", 
                     type.measure = "mse", 
                     nfolds = 5, alpha = 0.5, 
                     parallel = TRUE, trace.it = TRUE)
  u_betas[[k]] <- coef(fit, s= "lambda.min")
  rm(fit)
  gc()
})
rm(k)
u_beta <- unlist(u_beta)

uid_test <- sample_test$userId %>% as.character()
mid_test <- sample_test$movieId %>% as.character()

gl <- length(sample_test$genres)
gen_bias <- foreach(i = 1:gl, .combine = "c") %dopar% {
  gen[,mid_test[i]] %*% u_beta[[uid_test[i]]][-1] + u_beta[[uid_test[i]]][1]
}
rm(gl)

pred <- m_bias[u_bias[sample_test[, .(userId, movieId, rating)][
    , gen_bias := gen_bias], on = .(userId)], on = .(movieId)][
    , ':='(pred = pred <- g_mean + u_bias + m_bias - gen_bias, 
           err = pred - rating)]
sqrt(mean(pred$err * pred$err))

rm(gen, gen_x, ind_y, rtable_y)

#update the gen_x with trained beta
k <- length(u_beta)
u_beta_int <- foreach(k = 1:k, .combine = "c") %dopar% {
  u_beta[[k]][1] + 0
}
uid <- names(u_beta) %>% as.numeric()
u_beta_int <- data.table(uid, u_beta_int)
rm(k)

residual_alt <- u_beta_int[residual_train, on = .(uid = userId)][
  , resid_alt := resid - u_beta_int
]
rtable_alt <- dcast(residual_alt, uid ~ movieId, value.var = "resid_alt")
uid_dt <- as.data.table(uid)
rtable_alt <- rtable_alt[uid_dt, on = .(uid)]
u_beta_alt <- Matrix(0, nrow = 7612, ncol = (length(u_beta[[1]]) - 1))
for(i in 1:7612){
  u_beta_alt[i, ] <- u_beta[[i]][-1]
}

rtable_alt_y <- setnafill(rtable_alt[,-1], fill = 0)
rtable_alt_y <- as(as.matrix(rtable_alt_y), "sparseMatrix")

rm(i, uid, uid_dt, residual_alt, residual_train, rtable_alt)

ind_alt_y <- createFolds(1: rtable_alt_y@Dim[2], 
                     k = ceiling(rtable_alt_y@Dim[2]/1000))
k <- length(ind_alt_y)
gen_alt_x <- lapply(1:k, function(m){
  temp <- cv.glmnet(u_beta_alt, rtable_alt_y[, ind_alt_y[[m]]], 
                    family = "mgaussian", intercept = FALSE, 
                    type.measure = "mse", nfolds = 5, 
                    parallel = TRUE, trace.it = TRUE)
  coef(temp, s = "lambda.min")
})
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
            
            pred_resids <- foreach(i = 1:n, 
                                   .combine = "c", 
                                   .packages = "data.table") %dopar% {
                                    P[,pred_uid[i]] %*% Q[,pred_mid[i]]
                                  }

            pred <- pred[, resid := pred_resids]
            err_rmse <- valid[pred, on = .(userId, movieId)][
                                    , .(err = pred + resid - rating), 
                                    by = .(userId, movieId)]
            return(sqrt(mean(err_rmse$err * err_rmse$err)))
        }

rmse(g_mean = g_mean, u_bias = u_bias, m_bias = m_bias, P = P, Q = Q, 
     valid = sample_test)
